import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_runner(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "benchmarks" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_session_corpus_keeps_assistant_evidence_and_source_mapping():
    runner = load_runner("longmemeval_bench")
    corpus, ids = runner.build_corpus(
        ["empty", "assistant", "mixed"],
        [[], [{"role": "assistant", "content": "The code is amber."}],
         [{"role": "user", "content": "Which code?"},
          {"role": "assistant", "content": "Amber."}]],
    )
    assert ids == ["assistant", "mixed"]
    assert corpus == ["assistant: The code is amber.", "user: Which code?\nassistant: Amber."]
    with pytest.raises(ValueError, match="equal lengths"):
        runner.build_corpus(["missing"], [])
    dated, _ = runner.build_corpus(
        ["dated"], [[{"role": "user", "content": "Yesterday."}]], ["2024/01/02"],
    )
    assert dated == ["[2024/01/02]\nuser: Yesterday."]


@pytest.mark.parametrize("name", ["locomo_bench", "locomo_bench_llm"])
def test_locomo_explicit_database_overrides_inherited_environment(name, tmp_path, monkeypatch):
    runner = load_runner(name)
    monkeypatch.setenv("TAM_MEMORY_DIR", "/unrelated-memory")
    monkeypatch.setenv("CLAUDE_MEMORY_DIR", "/legacy-memory")
    monkeypatch.setenv("MEMORY_LLM_ENABLED", "true")
    runner.setup_env(tmp_path, True)
    assert runner.os.environ["TAM_MEMORY_DIR"] == str(tmp_path)
    assert runner.os.environ["CLAUDE_MEMORY_DIR"] == str(tmp_path)
    assert runner.os.environ["MEMORY_LLM_ENABLED"] == "false"


def test_longmemeval_reports_per_question_evidence_and_latency(tmp_path, monkeypatch):
    runner = load_runner("longmemeval_bench")
    source = tmp_path / "dataset.json"
    report = tmp_path / "report.json"
    question = {
        "question_id": "assistant-only", "question_type": "single-session-assistant",
        "question": "amber", "answer_session_ids": ["answer"],
        "haystack_session_ids": ["distractor", "answer"],
        "haystack_sessions": [
            [{"role": "user", "content": "The weather is sunny."}],
            [{"role": "assistant", "content": "The code is amber."}],
        ],
    }
    abstention = {**question, "question_id": "unknown_abs", "answer_session_ids": []}
    source.write_text(json.dumps([question, abstention]))
    monkeypatch.setattr(runner, "_OUTPUT_PATH", str(report))
    runner.run_benchmark(str(source), ["bm25"], k=1)
    result = json.loads(report.read_text())["modes"]["bm25"]
    assert result["total_r_all"] == 1
    assert result["scored_questions"] == 1
    assert len(result["records"]) == 2
    assert result["records"][1]["r_all"] is None
    assert result["records"][0]["retrieved_session_ids"] == ["answer"]
    assert result["records"][0]["gold_session_ids"] == ["answer"]
    assert result["records"][0]["question_id"] == "assistant-only"
    assert result["p50_latency_ms"] <= result["p95_latency_ms"]
    assert result["p50_latency_ms"] >= 0
