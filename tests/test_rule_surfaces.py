"""Unit tests for src/rule_surfaces.py (surfacing, follow check, effectiveness)."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import rule_surfaces as rs
from rule_surfaces import ActionEvidence, RuleSurfacer, SurfaceNotFound

ROOT = Path(__file__).resolve().parents[1]
MIGRATION_038 = ROOT / "migrations" / "038_rule_surfaces.sql"
T0 = datetime(2026, 10, 13, 9, 15, tzinfo=UTC)

RULES_DDL = """
CREATE TABLE rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    content TEXT NOT NULL,
    context TEXT DEFAULT '',
    category TEXT NOT NULL,
    scope TEXT DEFAULT 'global',
    priority INTEGER NOT NULL DEFAULT 5,
    source_insight_id INTEGER,
    project TEXT DEFAULT 'general',
    tags TEXT DEFAULT '[]',
    status TEXT DEFAULT 'active',
    fire_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    fail_count INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 0.0,
    last_fired TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class Clock:
    def __init__(self, now: datetime = T0):
        self.now = now

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(RULES_DDL)
    conn.executescript(MIGRATION_038.read_text(encoding="utf-8"))
    yield conn
    conn.close()


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def surfacer(db, clock):
    return RuleSurfacer(db, clock=clock)


def add_rule(db, content, *, scope="global", priority=5, status="active", tags=None,
             shown=0, project="general"):
    cur = db.execute(
        "INSERT INTO rules (session_id, content, category, scope, priority, project, tags, status, "
        "shown_count, created_at, updated_at) VALUES ('s', ?, 'manual', ?, ?, ?, ?, ?, ?, 'x', 'x')",
        (content, scope, priority, project, json.dumps(tags or []), status, shown))
    db.commit()
    return cur.lastrowid


def rule(db, rule_id):
    return dict(db.execute("SELECT * FROM rules WHERE id=?", (rule_id,)).fetchone())


# ── tokenisation ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("text", [None, "", "   ", "the and of", "?!..."])
def test_tokenize_empty_and_stopword_only(text):
    assert rs.tokenize(text) == []


def test_tokenize_stems_english_and_russian():
    assert rs.tokenize("Running migrations") == ["run", "migrat"]
    assert rs.tokenize("Миграции базы данных") == ["миграц", "баз", "дан"]


def test_tokenize_dedupes_and_handles_unicode():
    tokens = rs.tokenize("Café café 🚀 naïve résumé 42")
    assert tokens.count("café") == 1
    assert "42" not in tokens
    assert all(token for token in tokens)


@pytest.mark.parametrize(("text", "expected"), [
    ("Never use SQLite, use Postgres instead", {"sqlite"}),
    ("Don’t mock the database", {"mock", "databas"}),
    ("Force-push to main is forbidden", {"forc", "push"}),
    ("Never use print in the payment service", {"print"}),
    ("Не используй print, используй logging", {"print"}),
    ("Нельзя коммитить секреты", {"коммит", "секрет"}),
    ("никогда не делай деплой в пятницу", {"депл"}),
    ("Не пиши комментарии на русском", {"пиш", "комментар"}),
])
def test_forbidden_terms_detects_prohibitions(text, expected):
    assert rs.forbidden_terms(text) == frozenset(expected)


@pytest.mark.parametrize("text", [None, "", "Always run tests before commit",
                                  "Это не работает", "Use Postgres for queues"])
def test_forbidden_terms_none_for_non_prohibitions(text):
    assert rs.forbidden_terms(text) is None


# ── statistics ───────────────────────────────────────────────────────────


def test_wilson_interval_edges():
    assert rs.wilson_interval(0, 0) == (0.0, 1.0)
    low, high = rs.wilson_interval(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-4)
    assert high == pytest.approx(0.7634, abs=1e-4)
    assert rs.wilson_interval(0, 6)[0] == 0.0
    assert rs.wilson_interval(6, 6)[1] == 1.0
    with pytest.raises(ValueError):
        rs.wilson_interval(7, 6)


@pytest.mark.parametrize(("success", "fail", "expected"), [
    (1, 0, False), (0, 4, False), (0, 5, False), (0, 6, True), (1, 5, False),
    (1, 10, True), (0, 0, False), (10, 10, False),
])
def test_should_suspend(success, fail, expected):
    assert rs.should_suspend(success, fail) is expected


def test_rated_success_rate():
    assert rs.rated_success_rate(0, 0) == 0.0
    assert rs.rated_success_rate(1, 0) == 1.0
    assert rs.rated_success_rate(1, 3) == 0.25


def test_binomial_p_values():
    assert rs.binomial_two_sided_p(10, 10) == pytest.approx(2 / 1024)
    assert rs.binomial_two_sided_p(5, 10) == pytest.approx(1.0)
    assert rs.binomial_two_sided_p(0, 0) == 1.0
    with pytest.raises(ValueError):
        rs.binomial_two_sided_p(1, 2, 1.0)


def test_benjamini_hochberg():
    assert rs.benjamini_hochberg([]) == []
    assert rs.benjamini_hochberg([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])
    assert rs.benjamini_hochberg([0.9]) == [0.9]


# ── surfacing ────────────────────────────────────────────────────────────


def test_surface_rules_without_rules_returns_empty(surfacer):
    assert surfacer.surface_rules("anything", "p", "s1", "test") == []


@pytest.mark.parametrize("query", [None, "", "   "])
def test_empty_query_falls_back_to_priority(db, surfacer, query):
    low = add_rule(db, "Write docstrings", priority=2)
    high = add_rule(db, "Run tests before commit", priority=9)
    surfaces = surfacer.surface_rules(query, "p", "s1", "test")
    assert [s.rule_id for s in surfaces] == [high, low]
    assert {s.ranking for s in surfaces} == {"priority"}


def test_query_ranks_relevant_rules_and_idf_prefers_distinctive_tokens(db, surfacer):
    add_rule(db, "Run tests before every commit", priority=5)
    add_rule(db, "Run linters before every commit", priority=5)
    migration = add_rule(db, "Every SQLite migration needs a PostgreSQL twin", priority=5)
    surfaces = surfacer.surface_rules("add a sqlite migration", "p", "s1", "test")
    assert surfaces[0].rule_id == migration
    assert surfaces[0].ranking == "lexical"
    assert set(surfaces[0].matched_terms) == {"sqlite", "migrat"}
    assert all(s.score > 0 for s in surfaces)


def test_query_without_matches_falls_back_to_priority(db, surfacer):
    a = add_rule(db, "Run tests before commit", priority=3)
    b = add_rule(db, "Write docstrings", priority=7)
    surfaces = surfacer.surface_rules("kubernetes ingress", "p", "s1", "test")
    assert [s.rule_id for s in surfaces] == [b, a]
    assert surfaces[0].ranking == "priority"


def test_russian_query_matches_russian_rule_by_stem(db, surfacer):
    target = add_rule(db, "Перед миграцией базы делай резервную копию")
    add_rule(db, "Пиши тесты на каждую функцию")
    surfaces = surfacer.surface_rules("запускаю миграции базы", "p", "s1", "test")
    assert surfaces[0].rule_id == target
    assert surfaces[0].ranking == "lexical"


def test_priority_weights_equal_relevance(db, surfacer):
    low = add_rule(db, "Docker images must be pinned", priority=2)
    high = add_rule(db, "Docker compose files must be linted", priority=8)
    surfaces = surfacer.surface_rules("docker", "p", "s1", "test")
    assert [s.rule_id for s in surfaces] == [high, low]


def test_scope_and_status_filters(db, surfacer):
    mine = add_rule(db, "deploy checklist", scope="project:p")
    add_rule(db, "deploy freeze", scope="project:other")
    add_rule(db, "deploy retired", status="suspended")
    cat = add_rule(db, "deploy category rule", scope="category:infra")
    ids = {s.rule_id for s in surfacer.surface_rules("deploy", "p", "s1", "test")}
    assert ids == {mine}
    ids = {s.rule_id for s in surfacer.surface_rules("deploy", "p", "s2", "test", categories=["infra"])}
    assert ids == {mine, cat}


def test_phase_filter(db, surfacer):
    core = add_rule(db, "deploy core")
    build = add_rule(db, "deploy build", tags=["phase:build"])
    add_rule(db, "deploy plan", tags=["phase:plan"])
    ids = {s.rule_id for s in surfacer.surface_rules("deploy", "p", "s1", "t", phase="build")}
    assert ids == {core, build}
    with pytest.raises(ValueError):
        surfacer.surface_rules("deploy", "p", "s1", "t", phase="nope")


def test_limit_is_clamped(db, surfacer):
    for i in range(rs.MAX_SURFACE_LIMIT + 5):
        add_rule(db, f"rule number {i} about caching")
    assert len(surfacer.surface_rules("caching", "p", "s1", "t", limit=0)) == 1
    assert len(surfacer.surface_rules("caching", "p", "s2", "t", limit=10_000)) == rs.MAX_SURFACE_LIMIT


def test_surface_rows_and_shown_count(db, surfacer):
    rid = add_rule(db, "Run tests before commit")
    [surface] = surfacer.surface_rules("commit", "proj", "s1", "self_rules_context")
    row = dict(db.execute("SELECT * FROM rule_surfaces WHERE id=?", (surface.surface_id,)).fetchone())
    assert surface.surface_id.startswith(rs.SURFACE_ID_PREFIX)
    assert row["rule_id"] == rid and row["session_id"] == "s1" and row["project"] == "proj"
    assert row["query"] == "commit" and row["source"] == "self_rules_context"
    assert row["followed"] is None
    assert rule(db, rid)["shown_count"] == 1
    assert rule(db, rid)["fire_count"] == 0
    assert surface.to_dict()["surface_id"] == surface.surface_id


def test_surface_reused_within_same_session_and_hour(db, surfacer, clock):
    rid = add_rule(db, "Run tests before commit")
    first = surfacer.surface_rules("commit", "p", "s1", "t")[0]
    clock.now = T0 + timedelta(minutes=30)
    second = surfacer.surface_rules("commit again", "p", "s1", "t")[0]
    assert second.surface_id == first.surface_id and second.reused
    assert rule(db, rid)["shown_count"] == 1
    clock.now = T0 + timedelta(hours=1)
    third = surfacer.surface_rules("commit", "p", "s1", "t")[0]
    assert third.surface_id != first.surface_id and not third.reused
    other_session = surfacer.surface_rules("commit", "p", "s2", "t")[0]
    assert other_session.surface_id not in {first.surface_id, third.surface_id}
    assert rule(db, rid)["shown_count"] == 3


def test_damping_for_often_shown_never_followed(db, surfacer):
    stale = add_rule(db, "Cache invalidation needs a version key", shown=rs.DAMPING_MIN_SHOWN)
    fresh = add_rule(db, "Cache invalidation needs a version key", shown=0)
    surfaces = {s.rule_id: s for s in surfacer.surface_rules("cache invalidation", "p", "s1", "t")}
    assert surfaces[stale].damped and not surfaces[fresh].damped
    assert surfaces[stale].score == pytest.approx(surfaces[fresh].score * rs.DAMPING_FACTOR)


def test_no_damping_once_followed(db, surfacer):
    rid = add_rule(db, "Cache invalidation needs a version key", shown=rs.DAMPING_MIN_SHOWN)
    surface = surfacer.surface_rules("cache", "p", "s1", "t")[0]
    surfacer.mark_followed(surface.surface_id, True)
    again = surfacer.surface_rules("cache", "p", "s2", "t")[0]
    assert again.rule_id == rid and not again.damped


# ── mark_followed ────────────────────────────────────────────────────────


def test_mark_followed_values_and_fire_count(db, surfacer, clock):
    rid = add_rule(db, "Run tests before commit")
    sid = surfacer.surface_rules("commit", "p", "s1", "t")[0].surface_id
    mark = surfacer.mark_followed(sid, True, "ran pytest")
    assert (mark.followed, mark.note, mark.rule_id) == (1, "ran pytest", rid)
    assert mark.followed_at is not None
    assert rule(db, rid)["fire_count"] == 1
    surfacer.mark_followed(sid, True)
    assert rule(db, rid)["fire_count"] == 1  # only the first transition fires
    assert surfacer.mark_followed(sid, False).followed == 0
    assert surfacer.mark_followed(sid, True, applicable=False).followed == -1
    cleared = surfacer.mark_followed(sid, None)
    assert cleared.followed is None and cleared.followed_at is None
    row = db.execute("SELECT followed, followed_at FROM rule_surfaces WHERE id=?", (sid,)).fetchone()
    assert tuple(row) == (None, None)


def test_stored_query_and_note_are_redacted(db, surfacer):
    key = "AKIA" + "Q" * 16
    add_rule(db, "Rotate credentials after leaks")
    sid = surfacer.surface_rules(f"credentials {key}", "p", "s1", "t")[0].surface_id
    surfacer.mark_followed(sid, True, f"rotated {key}")
    row = db.execute("SELECT query, follow_note FROM rule_surfaces WHERE id=?", (sid,)).fetchone()
    assert key not in row[0] and key not in row[1]


def test_mark_followed_truncates_note(db, surfacer):
    add_rule(db, "Run tests before commit")
    sid = surfacer.surface_rules("commit", "p", "s1", "t")[0].surface_id
    assert len(surfacer.mark_followed(sid, True, "x" * 5000).note) == rs.MAX_NOTE_CHARS


def test_mark_followed_unknown_surface(surfacer):
    with pytest.raises(SurfaceNotFound):
        surfacer.mark_followed("rs_missing", True)


# ── followcheck ──────────────────────────────────────────────────────────


def surfaced(db, surfacer, content, query, session="s1"):
    rid = add_rule(db, content)
    surfaces = surfacer.surface_rules(query, "p", session, "t")
    return next(s for s in surfaces if s.rule_id == rid)


def act(text, minutes):
    return ActionEvidence(text, T0 + timedelta(minutes=minutes))


def test_followcheck_followed(db, surfacer):
    add_rule(db, "Write docstrings for public functions")
    s = surfaced(db, surfacer, "Run pytest with coverage before committing migrations", "pytest migrations")
    [verdict] = [v for v in surfacer.followcheck("s1", [act("ran pytest --coverage on the migrations", 5)])
                 if v.surface_id == s.surface_id]
    assert verdict.verdict == "followed" and verdict.followed == 1
    assert set(verdict.overlap) >= {"pytest", "coverag", "migrat"}
    assert verdict.persisted
    assert db.execute("SELECT followed FROM rule_surfaces WHERE id=?", (s.surface_id,)).fetchone()[0] == 1


def test_followcheck_not_applicable_and_unknown(db, surfacer):
    s = surfaced(db, surfacer, "Run pytest with coverage before committing migrations", "pytest")
    [na] = surfacer.followcheck("s1", [act("edited the README banner image", 5)], persist=False)
    assert na.verdict == "not_applicable" and na.followed == -1 and not na.persisted
    [unknown] = surfacer.followcheck("s1", [act("ran pytest quickly", 5)], persist=False)
    assert unknown.verdict == "unknown" and unknown.followed is None
    assert db.execute("SELECT followed FROM rule_surfaces WHERE id=?", (s.surface_id,)).fetchone()[0] is None


def test_followcheck_ignores_evidence_before_surface_or_outside_window(db, surfacer):
    surfaced(db, surfacer, "Run pytest with coverage before committing migrations", "pytest")
    [before] = surfacer.followcheck("s1", [act("ran pytest coverage migrations", -5),
                                           act("unrelated", 1)], persist=False)
    assert before.verdict == "not_applicable"
    [late] = surfacer.followcheck("s1", [act("ran pytest coverage migrations", 200)],
                                  window=timedelta(hours=4), persist=False)
    assert late.verdict == "followed"
    assert surfacer.followcheck("s1", [act("ran pytest coverage migrations", 200)], persist=False) == []
    [outside] = surfacer.followcheck("s1", [act("x", 1), act("ran pytest coverage migrations", 200)],
                                     persist=False)
    assert outside.verdict == "not_applicable" and outside.evidence_count == 1


def test_followcheck_counts_evidence_stamped_with_the_surface_time(db, surfacer):
    # Regression: on a coarse clock (Windows before Python 3.13) the prompt hook's
    # surface and the next tool call share a timestamp; the action was dropped.
    s = surfaced(db, surfacer, "Run pytest with coverage before committing migrations", "pytest migrations")
    at = datetime.fromisoformat(db.execute("SELECT surfaced_at FROM rule_surfaces WHERE id=?",
                                           (s.surface_id,)).fetchone()[0])
    [verdict] = [v for v in surfacer.followcheck("s1", [ActionEvidence("ran pytest --coverage on the migrations", at)],
                                                 persist=False)
                 if v.surface_id == s.surface_id]
    assert verdict.verdict == "followed" and verdict.evidence_count == 1


def test_followcheck_prohibition_violation_is_ignored(db, surfacer):
    s = surfaced(db, surfacer, "Never force-push to the main branch", "branch")
    [verdict] = surfacer.followcheck("s1", [act("git push --force origin main", 3)])
    assert verdict.verdict == "ignored" and verdict.followed == 0
    assert set(verdict.forbidden_hits) >= {"push", "forc"}
    assert db.execute("SELECT followed FROM rule_surfaces WHERE id=?", (s.surface_id,)).fetchone()[0] == 0


def test_followcheck_russian_prohibition(db, surfacer):
    surfaced(db, surfacer, "Никогда не используй print для логирования в сервисе", "логирование")
    [verdict] = surfacer.followcheck("s1", [act("добавил print в сервис", 3)], persist=False)
    assert verdict.verdict == "ignored"


def test_followcheck_prohibition_respected_counts_as_followed(db, surfacer):
    add_rule(db, "Write docstrings for public functions")
    s = surfaced(db, surfacer, "Never use print in the payment service, use structured logging",
                 "payment logging")
    verdicts = surfacer.followcheck("s1", [act("payment service now uses structured logging", 3)],
                                    persist=False)
    verdict = next(v for v in verdicts if v.surface_id == s.surface_id)
    assert verdict.verdict == "followed" and not verdict.forbidden_hits


def test_followcheck_does_not_overwrite_explicit_mark(db, surfacer):
    s = surfaced(db, surfacer, "Never force-push to the main branch", "branch")
    surfacer.mark_followed(s.surface_id, True, "manual")
    [verdict] = surfacer.followcheck("s1", [act("git push --force origin main", 3)])
    assert verdict.verdict == "ignored" and not verdict.persisted
    row = db.execute("SELECT followed, follow_note FROM rule_surfaces WHERE id=?", (s.surface_id,)).fetchone()
    assert tuple(row) == (1, "manual")


def test_followcheck_edge_cases(db, surfacer):
    assert surfacer.followcheck("s1", []) == []
    assert surfacer.followcheck("nobody", [act("anything", 1)]) == []
    with pytest.raises(ValueError):
        surfacer.followcheck("s1", [act("x", 1)], window=timedelta(0))
    with pytest.raises(ValueError):
        ActionEvidence("x", datetime(2026, 1, 1, tzinfo=None))  # noqa: DTZ001 — naive on purpose


def test_followcheck_other_session_is_not_judged(db, surfacer):
    surfaced(db, surfacer, "Never force-push to the main branch", "branch", session="s1")
    assert surfacer.followcheck("s2", [act("git push --force origin main", 3)]) == []


# ── effectiveness report ─────────────────────────────────────────────────


def seed_marks(db, rule_id, followed, ignored, project="p", extra=None):
    values = [1] * followed + [0] * ignored + list(extra or [])
    for index, value in enumerate(values):
        db.execute(
            "INSERT INTO rule_surfaces (id, rule_id, session_id, project, surfaced_at, followed) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (f"rs_{rule_id}_{project}_{index}", rule_id, f"s{index}", project, "2026-10-01T00:00:00.000000Z",
             value))
    db.commit()


def test_report_empty(surfacer):
    report = surfacer.effectiveness_report()
    assert report.rules == ()
    data = report.to_dict()
    assert data["summary"] == {"useful": 0, "harmful": 0, "unverified": 0, "insufficient": 0}
    assert data["caveats"]


def test_report_verdicts_and_bh(db, surfacer):
    useful = add_rule(db, "useful rule")
    harmful = add_rule(db, "ignored rule")
    coin = add_rule(db, "coin rule")
    thin = add_rule(db, "thin rule")
    seed_marks(db, useful, 12, 0)
    seed_marks(db, harmful, 0, 12)
    seed_marks(db, coin, 5, 5, extra=[-1, None])
    seed_marks(db, thin, 2, 1)
    report = {r.rule_id: r for r in surfacer.effectiveness_report(min_n=5).rules}
    assert report[useful].verdict == "useful"
    assert report[harmful].verdict == "harmful"
    assert report[coin].verdict == "unverified"
    assert report[thin].verdict == "insufficient" and report[thin].q_value is None
    assert (report[coin].not_applicable, report[coin].unjudged, report[coin].surfaces) == (1, 1, 12)
    assert report[coin].follow_rate == pytest.approx(0.5)
    assert report[useful].q_value >= report[useful].p_value
    assert report[useful].ci_low > 0.5
    assert report[useful].to_dict()["n"] == 12


def test_report_project_filter_and_deleted_rule(db, surfacer):
    rid = add_rule(db, "rule")
    seed_marks(db, rid, 6, 0, project="a")
    seed_marks(db, 999, 1, 0, project="b")
    only_a = surfacer.effectiveness_report(project="a")
    assert [r.rule_id for r in only_a.rules] == [rid]
    everything = {r.rule_id: r for r in surfacer.effectiveness_report().rules}
    assert everything[999].status == "deleted"


def test_report_rejects_bad_min_n(surfacer):
    with pytest.raises(ValueError):
        surfacer.effectiveness_report(min_n=0)


def test_report_includes_ratings(db, surfacer):
    rid = add_rule(db, "rule")
    db.execute("UPDATE rules SET success_count=3, fail_count=1 WHERE id=?", (rid,))
    seed_marks(db, rid, 1, 0)
    [entry] = surfacer.effectiveness_report().rules
    assert (entry.rated_helpful, entry.rated_unhelpful) == (3, 1)
