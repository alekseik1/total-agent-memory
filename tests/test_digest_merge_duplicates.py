"""DigestPhase.merge_duplicates: distinct values never merge, the newest record survives, no O(n^2) scan."""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reflection import digest  # noqa: E402
from reflection.digest import DigestPhase  # noqa: E402

BULK_ROWS = 6000
BULK_TIME_BUDGET_SEC = 20.0


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE knowledge(id INTEGER PRIMARY KEY, type TEXT, content TEXT, project TEXT, tags TEXT, "
        "confidence REAL, recall_count INT, created_at TEXT, status TEXT, superseded_by INT)"
    )
    conn.execute(
        "CREATE TABLE knowledge_nodes(knowledge_id INT, node_id TEXT, role TEXT, strength REAL, "
        "UNIQUE(knowledge_id, node_id, role))"
    )
    yield conn
    conn.close()


def _add(db, kid, content, created, recalls=0, project="p", ktype="fact"):
    db.execute(
        "INSERT INTO knowledge VALUES(?,?,?,?,?,?,?,?,?,NULL)",
        (kid, ktype, content, project, "[]", 1.0, recalls, created, "active"),
    )


def _state(db):
    return {r["id"]: (r["status"], r["superseded_by"], r["recall_count"]) for r in db.execute("SELECT * FROM knowledge")}


def test_different_numbers_are_not_duplicates(db):
    for kid, rpm, day in ((1, 100, "03"), (2, 500, "02"), (3, 900, "01")):
        _add(db, kid, f"API rate limit for service A is {rpm} requests per minute in production",
             f"2026-03-{day}", recalls=5 if kid == 2 else 0)
    assert DigestPhase(db).merge_duplicates() == 0
    assert {v[0] for v in _state(db).values()} == {"active"}


def test_negation_flips_meaning_and_blocks_merge(db):
    _add(db, 1, "The nightly backup job is enabled for the production database cluster", "2026-03-02")
    _add(db, 2, "The nightly backup job is not enabled for the production database cluster", "2026-03-01")
    assert DigestPhase(db).merge_duplicates() == 0


def test_newest_survives_even_if_older_has_more_recalls(db):
    text = "Deploys go through the blue green pipeline and finish within ten minutes"
    _add(db, 1, text, "2026-03-03", recalls=0)
    _add(db, 2, text + ".", "2026-03-02", recalls=7)
    assert DigestPhase(db).merge_duplicates() == 1
    state = _state(db)
    assert state[1] == ("active", None, 7)
    assert state[2][:2] == ("superseded", 1)


def test_supersede_never_points_at_a_dead_record(db):
    base = "The staging environment is rebuilt from the main branch every night at three"
    _add(db, 1, base, "2026-03-04")
    _add(db, 2, base + ".", "2026-03-03", recalls=9)
    _add(db, 3, base + "!", "2026-03-02")
    _add(db, 4, base + "?", "2026-03-01", recalls=2)
    DigestPhase(db).merge_duplicates()
    state = _state(db)
    for kid, (status, target, _) in state.items():
        if status == "superseded":
            assert state[target][0] == "active", f"{kid} -> {target} is dead"
    assert sum(1 for v in state.values() if v[0] == "active") == 1
    assert state[1][2] == 11


def test_existing_chain_is_repointed_to_the_survivor(db):
    base = "Alerts are routed to the on call engineer through the paging service"
    _add(db, 1, base, "2026-03-03")
    _add(db, 2, base + ".", "2026-03-02")
    _add(db, 3, "Totally different historical note about the cache layer size", "2026-02-01")
    db.execute("UPDATE knowledge SET status='superseded', superseded_by=2 WHERE id=3")
    DigestPhase(db).merge_duplicates()
    assert _state(db)[3][:2] == ("superseded", 1)


def test_exact_duplicates_collapse_and_other_projects_are_untouched(db):
    text = "Use connection pooling with a maximum of twenty connections per service"
    for kid in (1, 2, 3):
        _add(db, kid, text, f"2026-03-0{kid}")
    _add(db, 4, text, "2026-03-01", project="other")
    assert DigestPhase(db).merge_duplicates() == 2
    state = _state(db)
    assert state[3][0] == "active" and state[4][0] == "active"
    assert state[1][0] == state[2][0] == "superseded"
    assert state[1][1] == 3 and state[2][1] == 3


def test_blocking_avoids_all_pairs_comparison(db, monkeypatch):
    calls = {"n": 0}
    real = digest.SequenceMatcher

    class Counting(real):
        def __init__(self, *args, **kwargs):
            calls["n"] += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(digest, "SequenceMatcher", Counting)
    started = time.monotonic()
    rows = [(i, f"unique topic{i} with word{i} and detail{i} for item{i} describing component{i} behaviour",
             f"2026-01-01T00:{i // 60 % 60:02d}:{i % 60:02d}Z") for i in range(BULK_ROWS)]
    db.executemany(
        "INSERT INTO knowledge VALUES(?,?,?,?,?,?,?,?,?,NULL)",
        [(i, "fact", c, "p", "[]", 1.0, 0, t, "active") for i, c, t in rows],
    )
    assert DigestPhase(db).merge_duplicates() == 0
    assert time.monotonic() - started < BULK_TIME_BUDGET_SEC
    assert calls["n"] < BULK_ROWS * 5  # all-pairs would be ~18 million


def test_near_duplicates_with_one_typo_are_still_found(db):
    _add(db, 1, "The release train leaves every second Tuesday after the QA signoff", "2026-03-02")
    _add(db, 2, "The release train leaves every second Tuesday after the QA sign off", "2026-03-01")
    assert DigestPhase(db).merge_duplicates() == 1
