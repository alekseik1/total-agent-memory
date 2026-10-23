"""Multilingual recall-quality floor gate.

Saves a frozen EN/RU/UK corpus (tests/fixtures/recall_quality/corpus.json)
through the real ``memory_save`` MCP tool into an isolated memory directory,
then runs every labelled query through ``memory_recall`` with the default
local embedder and fails when top-1 / top-3 accuracy of any bucket drops
below its floor.

Run it on its own:

    pytest -m eval tests/test_recall_quality_gate.py -s

``-s`` prints the per-bucket table and every miss. The embedding model must
be in the fastembed cache (``python tests/smoke/prewarm_models.py``); without
it the test skips, unless ``TAM_EVAL_REQUIRE_EMBEDDER=1`` is set, in which case
a missing model is a failure (CI sets it so the gate can never pass vacuously).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

CORPUS_PATH = ROOT / "tests" / "fixtures" / "recall_quality" / "corpus.json"
RECALL_LIMIT = 5
TOP_K = 3

# Floors = measured accuracy minus a margin of roughly one query per bucket,
# so a single platform-dependent rank flip (ONNX float drift between CPUs)
# does not fail the gate while a real retrieval regression does.
#
# Measured 2026-10-13, macOS arm64, Python 3.13, fastembed 0.8.0,
# paraphrase-multilingual-MiniLM-L12-v2 (default fast mode, cross-encoder off):
#   bucket          n   top1   top3
#   exact          24  1.000  1.000   (24/24, 24/24)
#   paraphrase     22  0.773  0.864   (17/22, 19/22)
#   cross_lingual  24  0.833  0.958   (20/24, 23/24)
#   temporal        8  1.000  1.000   (8/8, 8/8)
# Three consecutive runs produced byte-identical tables.
FLOORS: dict[str, dict[str, float]] = {
    "exact": {"top1": 0.95, "top3": 0.95},
    "paraphrase": {"top1": 0.72, "top3": 0.81},
    "cross_lingual": {"top1": 0.79, "top3": 0.91},
    "temporal": {"top1": 0.87, "top3": 0.87},
}

# A superseded value must never be returned by recall at all.
MAX_STALE_LEAKS = 0

RUNTIME_ENV = {
    "MEMORY_MODE": "fast",
    "MEMORY_LLM_ENABLED": "false",
    "MEMORY_CROSS_RERANK": "off",
    "MEMORY_ASYNC_ENRICHMENT": "false",
    "MEMORY_QUALITY_GATE_ENABLED": "false",
    "MEMORY_CONTRADICTION_DETECT_ENABLED": "false",
    "MCP_TRANSPORT": "stdio",
}

pytestmark = pytest.mark.eval


@dataclass
class BucketScore:
    total: int = 0
    top1: int = 0
    top3: int = 0
    misses: list[str] = field(default_factory=list)

    def rate(self, metric: str) -> float:
        return getattr(self, metric) / self.total if self.total else 0.0


@dataclass
class EvalResult:
    buckets: dict[str, BucketScore]
    stale_leaks: list[str]

    def table(self) -> str:
        lines = [f"{'bucket':<14} {'n':>3} {'top1':>6} {'top3':>6}"]
        for name, score in sorted(self.buckets.items()):
            lines.append(
                f"{name:<14} {score.total:>3} {score.rate('top1'):>6.3f} {score.rate('top3'):>6.3f}"
            )
        for name, score in sorted(self.buckets.items()):
            lines.extend(f"  miss[{name}] {miss}" for miss in score.misses)
        lines.extend(f"  stale-leak {leak}" for leak in self.stale_leaks)
        return "\n".join(lines)


def load_corpus() -> dict:
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def call(server, tool: str, **args) -> dict:
    return json.loads(asyncio.run(server._do(tool, args)))


def recall_contents(server, project: str, query: str) -> list[str]:
    out = call(server, "memory_recall", query=query, project=project, limit=RECALL_LIMIT)
    return [hit["content"] for group in out["results"].values() for hit in group]


def seed(server, corpus: dict) -> dict[str, str]:
    project = corpus["project"]
    by_key: dict[str, str] = {}
    for fact in corpus["facts"]:
        if "previous" in fact:
            old = call(server, "memory_save", content=fact["previous"], type="fact", project=project)
            assert old.get("saved") is True, f"seeding {fact['key']} (previous) failed: {old}"
            new = call(server, "memory_save", content=fact["content"], type="fact",
                       project=project, supersede=True)
            assert new.get("superseded") == [old["id"]], (
                f"seeding {fact['key']}: supersede did not retire the previous value: {new}"
            )
        else:
            new = call(server, "memory_save", content=fact["content"], type="fact", project=project)
        assert new.get("saved") is True, f"seeding {fact['key']} failed: {new}"
        by_key[fact["key"]] = fact["content"]
    return by_key


def evaluate(server, corpus: dict) -> EvalResult:
    content_by_key = seed(server, corpus)
    stale = {f["previous"]: f["key"] for f in corpus["facts"] if "previous" in f}
    buckets: dict[str, BucketScore] = defaultdict(BucketScore)
    leaks: list[str] = []
    for q in corpus["queries"]:
        hits = recall_contents(server, corpus["project"], q["query"])
        expected = content_by_key[q["expect"]]
        score = buckets[q["bucket"]]
        score.total += 1
        if hits[:1] == [expected]:
            score.top1 += 1
        if expected in hits[:TOP_K]:
            score.top3 += 1
        else:
            got = hits[0][:60] if hits else "no hits"
            score.misses.append(f"[{q['lang']}] {q['query']!r} -> expected {q['expect']}, top1={got!r}")
        leaks.extend(f"{q['query']!r} returned stale value of {stale[h]}" for h in hits if h in stale)
    return EvalResult(dict(buckets), leaks)


def _embedder_ready(server) -> bool:
    return server.store.fastembed is not None


@pytest.fixture(scope="module")
def eval_result(tmp_path_factory):
    memory_dir = tmp_path_factory.mktemp("recall-quality")
    (memory_dir / "blobs").mkdir()
    (memory_dir / "chroma").mkdir()
    with pytest.MonkeyPatch.context() as mp:
        for key, value in RUNTIME_ENV.items():
            mp.setenv(key, value)
        mp.setenv("MEMORY_ACTIVECONTEXT_VAULT", str(memory_dir / "vault"))
        import config
        import server

        config._cache_clear()
        mp.setattr(server, "MEMORY_DIR", memory_dir)
        mp.setattr(server, "store", server.Store())
        mp.setattr(server, "recall", server.Recall(server.store))
        mp.setattr(server, "SID", "recall-quality-gate")
        mp.setattr(server, "BRANCH", "")
        mp.setattr(server, "_v5_modules", {})
        try:
            if not _embedder_ready(server):
                message = (
                    "default embedding model is not cached; run "
                    "`FASTEMBED_CACHE_PATH=... python tests/smoke/prewarm_models.py`"
                )
                if os.environ.get("TAM_EVAL_REQUIRE_EMBEDDER") == "1":
                    pytest.fail(message)
                pytest.skip(message)
            result = evaluate(server, load_corpus())
        finally:
            server.store.db.close()
            config._cache_clear()
    sys.stdout.write("\nrecall-quality gate\n" + result.table() + "\n")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write("### Recall-quality gate\n\n```\n" + result.table() + "\n```\n")
    return result


def test_corpus_is_well_formed():
    corpus = load_corpus()
    keys = [f["key"] for f in corpus["facts"]]
    assert len(keys) == len(set(keys)), "duplicate fact keys"
    contents = [f["content"] for f in corpus["facts"]] + [f["previous"] for f in corpus["facts"] if "previous" in f]
    assert len(contents) == len(set(contents)), "duplicate fact contents make top-1 ambiguous"
    assert {f["lang"] for f in corpus["facts"]} == {"en", "ru", "uk"}
    for q in corpus["queries"]:
        assert q["bucket"] in FLOORS, f"unknown bucket {q['bucket']!r}"
        assert q["expect"] in keys, f"query {q['query']!r} expects unknown fact {q['expect']!r}"
    temporal = {q["expect"] for q in corpus["queries"] if q["bucket"] == "temporal"}
    updated = {f["key"] for f in corpus["facts"] if "previous" in f}
    assert temporal <= updated, f"temporal queries must target updated facts: {temporal - updated}"


@pytest.mark.parametrize("bucket", sorted(FLOORS))
@pytest.mark.parametrize("metric", ["top1", "top3"])
def test_recall_quality_floor(eval_result, bucket, metric):
    score = eval_result.buckets[bucket]
    floor = FLOORS[bucket][metric]
    actual = score.rate(metric)
    assert actual >= floor, (
        f"{bucket} {metric} = {actual:.3f} ({getattr(score, metric)}/{score.total}) "
        f"fell below the floor {floor:.3f}\n{eval_result.table()}"
    )


def test_superseded_values_never_recalled(eval_result):
    assert len(eval_result.stale_leaks) <= MAX_STALE_LEAKS, "\n".join(eval_result.stale_leaks)
