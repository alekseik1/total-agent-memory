"""Rule surfacing, follow tracking and effectiveness measurement (SOUL rules).

A *surface* is one rule shown to one session for one query. Surfaces are
recorded in ``rule_surfaces`` (migration 038) so that later evidence — what
the agent actually did — can be matched against them:

* :meth:`RuleSurfacer.surface_rules` ranks active rules by distinctive-token
  overlap with the query (IDF computed per call over the active rule corpus),
  weighted by rule priority and damped for rules that keep being shown but
  are never followed. One surface per (rule, session, hour) is reused.
* :meth:`RuleSurfacer.mark_followed` records an explicit verdict.
* :meth:`RuleSurfacer.followcheck` infers verdicts from action evidence by
  token overlap, with prohibition awareness: a "never X" rule whose X shows
  up in the actions was ignored, not followed.
* :meth:`RuleSurfacer.effectiveness_report` summarises follow rates with
  Wilson intervals and Benjamini-Hochberg-corrected verdicts. It measures,
  it never changes rule status.
"""

from __future__ import annotations

import json
import logging
import math
import re
import uuid
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any, Literal, Protocol

import snowballstemmer

from memory_core.query_terms import QUESTION_STOPWORDS, russian_stem
from memory_core.timestamps import format_utc
from secret_redaction import redact_secrets

LOG = logging.getLogger(__name__)

# ── Rating / suspension policy ─────────────────────────────────────────────
WILSON_Z_95 = 1.959963984540054
# A rule is auto-suspended only after this many rated outcomes...
RULE_SUSPEND_MIN_RATED = 5
# ...and only when even the optimistic end of its 95% success interval is low.
# With z = 1.96 the upper bound for 0 successes in n ratings is z^2 / (n + z^2):
# 0.434 at n=5, 0.390 at n=6, 0.300 only at n=9. 0.4 suspends a rule after six
# straight failures (one helpful rating needs eleven ratings in all).
RULE_SUSPEND_MAX_UPPER = 0.4

# ── Surfacing ──────────────────────────────────────────────────────────────
DEFAULT_SURFACE_LIMIT = 10
MAX_SURFACE_LIMIT = 50
# Rules shown this often without a single follow lose part of their score.
DAMPING_MIN_SHOWN = 10
DAMPING_FACTOR = 0.6
SURFACE_REUSE_BUCKET = timedelta(hours=1)
SURFACE_ID_PREFIX = "rs_"
MAX_STORED_QUERY_CHARS = 2000
MAX_NOTE_CHARS = 1000
VALID_PHASES = frozenset({"van", "plan", "creative", "build", "reflect", "archive"})
PHASE_TAG_PREFIX = "phase:"

# ── Follow check ───────────────────────────────────────────────────────────
DEFAULT_FOLLOW_WINDOW = timedelta(hours=2)
FOLLOW_MIN_OVERLAP = 3
FOLLOW_MIN_DISTINCTIVE = 2
# A token is distinctive when it occurs in at most this share of active rules
# (or in exactly one rule, which covers tiny corpora).
DISTINCTIVE_MAX_DF_RATIO = 0.25
FORBIDDEN_CLAUSE_MAX_TOKENS = 6
FORBIDDEN_MIN_OVERLAP = 2

# ── Effectiveness report ───────────────────────────────────────────────────
DEFAULT_REPORT_MIN_N = 5
REPORT_ALPHA = 0.05
# Null hypothesis of the per-rule test: the rule is followed as often as not.
REPORT_NULL_FOLLOW_RATE = 0.5
SQL_IN_CHUNK = 500

MIN_TOKEN_LENGTH = 2

FOLLOWED = 1
IGNORED = 0
NOT_APPLICABLE = -1

Verdict = Literal["followed", "ignored", "not_applicable", "unknown"]
EffectivenessVerdict = Literal["useful", "harmful", "unverified", "insufficient"]
Ranking = Literal["lexical", "priority"]

_VERDICT_VALUES: dict[str, int | None] = {
    "followed": FOLLOWED,
    "ignored": IGNORED,
    "not_applicable": NOT_APPLICABLE,
    "unknown": None,
}

# ── Tokenisation ───────────────────────────────────────────────────────────
_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)
_LATIN_WORD = re.compile(r"[a-z]+")
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})

# Function words that carry no topic. Negations are here too: they matter for
# prohibition detection (handled on raw text) but not for topical overlap.
EXTRA_STOPWORDS = frozenset((
    "i", "you", "he", "she", "it", "we", "they", "my", "your", "our", "their", "this", "that",
    "these", "those", "there", "here", "not", "no", "never", "don't", "dont", "doesn't", "didn't",
    "isn't", "aren't", "won't", "can't", "cannot", "avoid", "must", "mustn't", "shouldn't",
    "always", "any", "all", "some", "each", "every", "only", "also", "just", "than", "then",
    "into", "onto", "from", "over", "under", "via", "per", "if", "else", "so", "but", "because",
    "while", "after", "before", "up", "down", "out", "off", "again", "more", "most", "less",
    "very", "too", "it's", "its", "instead", "rather", "forbidden", "prohibited", "allowed", "не",
    "ни", "нет", "никогда", "нельзя", "всегда", "это", "этот", "эта", "эти", "тот", "та", "те",
    "там", "тут", "здесь", "я", "мы", "он", "она", "они", "оно", "его", "её", "ее", "их", "наш",
    "ваш", "свой", "мой", "твой", "же", "ли", "бы", "уже", "ещё", "еще", "при", "над", "под",
    "без", "через", "между", "после", "перед", "если", "чтобы", "потому", "но", "да", "так",
    "тоже", "также", "только", "все", "всё", "всех", "весь", "каждый", "любой", "надо", "нужно",
    "можно", "должен", "должна", "вместо", "запрещено", "запрещается", "избегай", "избегайте",
))
STOPWORDS = QUESTION_STOPWORDS | EXTRA_STOPWORDS

# Verbs too generic to identify what a prohibition forbids ("never use X").
GENERIC_ACTION_WORDS = frozenset((
    "use", "using", "used", "uses", "run", "running", "runs", "call", "calling", "make", "making",
    "do", "doing", "does", "perform", "apply", "try", "put", "используй", "используйте",
    "использовать", "использую", "применяй", "применять", "делай", "делайте", "делать", "запускай",
    "запускать", "вызывай", "вызывать", "ставь", "ставить",
))

_PROHIBITION = re.compile(
    r"(?<!\w)(?:"
    r"never|avoid(?:ing)?|don't|dont|do\s+not|does\s+not|must\s+not|mustn't|"
    r"should\s+not|shouldn't|cannot|can't|may\s+not|no\s+longer|refrain\s+from|"
    r"(?:is\s+|are\s+)?(?:forbidden|prohibited|not\s+allowed)(?:\s+to)?|"
    r"нельзя|никогда(?:\s+не)?|запрещено|запрещается|запрещ[её]н[аы]?|избегай(?:те)?|"
    r"не\s+надо|не\s+нужно|не\s+стоит|не\s+смей(?:те)?|хватит|перестань(?:те)?|"
    r"не(?=\s+[а-яё]+(?:и|ите|й|йте|ь|ьте)(?![а-яё]))"
    r")(?!\w)",
    re.UNICODE,
)
# Where the forbidden object ends: punctuation, a contrastive connective, or a
# preposition that starts the circumstances ("never use print | in the service").
_PUNCTUATION_END = re.compile(r"[.;!?\n,:()\[\]{}]|\s[-\u2013\u2014]\s", re.UNICODE)
_CLAUSE_END = re.compile(
    r"[.;!?\n,:()\[\]{}]|\s[-–—]\s|(?<!\w)(?:instead|unless|except|but|rather|"
    r"in|on|at|to|for|within|inside|into|during|when|while|"
    r"вместо|кроме|если|но|лучше|в|во|на|для|при|когда|пока)(?!\w)",
    re.UNICODE,
)


@lru_cache(maxsize=1)
def _english_stemmer() -> snowballstemmer.stemmer:
    return snowballstemmer.stemmer("english")


def _stem(word: str) -> str:
    if _LATIN_WORD.fullmatch(word):
        return _english_stemmer().stemWord(word)
    return russian_stem(word)


def _normalise(text: str) -> str:
    return text.translate(_APOSTROPHES).casefold()


def _content_words(text: str) -> list[str]:
    return [
        word
        for word in _WORD.findall(_normalise(text))
        if len(word) >= MIN_TOKEN_LENGTH and word not in STOPWORDS and not word.isdigit()
    ]


def tokenize(text: str | None) -> list[str]:
    """Stemmed topical tokens of ``text`` (EN + RU), order kept, duplicates removed."""
    if not text:
        return []
    return list(dict.fromkeys(_stem(word) for word in _content_words(text)))


def forbidden_terms(rule_text: str | None) -> frozenset[str] | None:
    """Stemmed tokens a prohibition forbids, or ``None`` when the rule forbids nothing.

    "Never use SQLite, use Postgres instead" -> {"sqlite"}; "Не используй print" ->
    {"print"}. A trailing marker ("force-push is forbidden") takes the clause
    before it.
    """
    if not rule_text:
        return None
    text = _normalise(rule_text)
    found: set[str] = set()
    prohibition = False
    for match in _PROHIBITION.finditer(text):
        prohibition = True
        tail = text[match.end():]
        end = _CLAUSE_END.search(tail)
        words = _object_words(tail[: end.start()] if end else tail)
        if not words:
            head = text[: match.start()]
            starts = [m.end() for m in _PUNCTUATION_END.finditer(head)]
            segment = head[starts[-1]:] if starts else head
            end = _CLAUSE_END.search(segment)
            words = _object_words(segment[: end.start()] if end else segment)
        found.update(_stem(word) for word in words[:FORBIDDEN_CLAUSE_MAX_TOKENS])
    if not prohibition:
        return None
    return frozenset(found)


def _object_words(clause: str) -> list[str]:
    return [word for word in _content_words(clause) if word not in GENERIC_ACTION_WORDS]


# ── Statistics ─────────────────────────────────────────────────────────────


def wilson_interval(successes: int, n: int, z: float = WILSON_Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion; ``(0.0, 1.0)`` when ``n`` is 0."""
    if n <= 0:
        return 0.0, 1.0
    if successes < 0 or successes > n:
        raise ValueError(f"successes must be within 0..{n}, got {successes}")
    p = successes / n
    z2 = z * z
    denominator = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    # The bounds at the boundary are exactly 0 / 1; avoid floating-point residue.
    low = 0.0 if successes == 0 else max(0.0, centre - half)
    high = 1.0 if successes == n else min(1.0, centre + half)
    return low, high


def rated_success_rate(success_count: int, fail_count: int) -> float:
    """Share of rated outcomes that were helpful; 0.0 when nothing was rated."""
    rated = success_count + fail_count
    return success_count / rated if rated > 0 else 0.0


def should_suspend(success_count: int, fail_count: int) -> bool:
    """True when enough ratings exist and the 95% upper bound of success is below the bar."""
    rated = success_count + fail_count
    if rated < RULE_SUSPEND_MIN_RATED:
        return False
    return wilson_interval(success_count, rated)[1] < RULE_SUSPEND_MAX_UPPER


def wilson_lower_bound(success_count: int, fail_count: int) -> float:
    """Lower end of the 95% Wilson interval of the rated success share; 0.0 when unrated."""
    rated = success_count + fail_count
    return wilson_interval(success_count, rated)[0] if rated > 0 else 0.0


@dataclass(frozen=True)
class RuleOutcome:
    rule_id: int
    status: str
    success_count: int
    fail_count: int
    success_rate: float
    lower_bound: float
    upper_bound: float


def record_rule_outcome(db: Database, rule_id: int, success: bool, updated_at: str) -> RuleOutcome | None:
    """Count one explicit rating and refresh success_rate (raw share) and success_lower_bound.

    Commits. None when the rule does not exist. Suspension is the caller's call.
    """
    column = "success_count" if success else "fail_count"
    db.execute(f"UPDATE rules SET {column} = COALESCE({column}, 0) + 1, updated_at = ? WHERE id = ?",
               (updated_at, rule_id))
    row = db.execute("SELECT success_count, fail_count, status FROM rules WHERE id = ?", (rule_id,)).fetchone()
    if row is None:
        db.commit()
        return None
    successes, failures = int(row[0] or 0), int(row[1] or 0)
    rate = rated_success_rate(successes, failures)
    low, high = wilson_interval(successes, successes + failures)
    db.execute("UPDATE rules SET success_rate = ?, success_lower_bound = ? WHERE id = ?", (rate, low, rule_id))
    db.commit()
    return RuleOutcome(rule_id, str(row[2] or ""), successes, failures, rate, low, high)


@dataclass(frozen=True)
class AutoOutcome:
    rule_id: int
    auto_success_count: int
    auto_fail_count: int


def record_auto_outcome(db: Database, rule_id: int, followed: bool, updated_at: str) -> AutoOutcome | None:
    """Count one heuristic (Stop-hook) verdict in the auto_* counters only; commits.

    Explicit ratings (success_count, fail_count, success_rate, success_lower_bound)
    and therefore auto-suspension are never touched. None when the rule does not exist.
    """
    column = "auto_success_count" if followed else "auto_fail_count"
    db.execute(f"UPDATE rules SET {column} = COALESCE({column}, 0) + 1, updated_at = ? WHERE id = ?",
               (updated_at, rule_id))
    row = db.execute("SELECT auto_success_count, auto_fail_count FROM rules WHERE id = ?", (rule_id,)).fetchone()
    db.commit()
    return None if row is None else AutoOutcome(rule_id, int(row[0] or 0), int(row[1] or 0))


def backfill_lower_bounds(db: Database) -> int:
    """Fill success_lower_bound for rated rules that predate the column (migration 040); commits."""
    rows = db.execute(
        "SELECT id, success_count, fail_count FROM rules "
        "WHERE COALESCE(success_count, 0) > 0 AND COALESCE(success_lower_bound, 0) = 0").fetchall()
    for rule_id, successes, failures in rows:
        db.execute("UPDATE rules SET success_lower_bound = ? WHERE id = ?",
                   (wilson_lower_bound(int(successes or 0), int(failures or 0)), rule_id))
    db.commit()
    return len(rows)


def binomial_two_sided_p(k: int, n: int, p0: float = REPORT_NULL_FOLLOW_RATE) -> float:
    """Exact two-sided binomial test p-value (sum of outcomes no likelier than ``k``)."""
    if n <= 0:
        return 1.0
    if not 0.0 < p0 < 1.0:
        raise ValueError(f"p0 must be in (0, 1), got {p0}")
    log_p, log_q = math.log(p0), math.log1p(-p0)

    def log_pmf(i: int) -> float:
        return (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                + i * log_p + (n - i) * log_q)

    observed = log_pmf(k)
    # Relative tolerance against floating-point ties between symmetric outcomes.
    tolerance = 1e-7
    total = sum(math.exp(lp) for lp in map(log_pmf, range(n + 1)) if lp <= observed + tolerance)
    return min(1.0, total)


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (q-values), in input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [1.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        index = order[rank - 1]
        running = min(running, p_values[index] * m / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


# ── Data types ─────────────────────────────────────────────────────────────


class Database(Protocol):
    """The DB-API subset used here (sqlite3.Connection or tam_db PgConnection)."""

    def execute(self, sql: str, parameters: Sequence[Any] = ..., /) -> Any: ...

    def commit(self) -> None: ...


class RuleSurfaceError(Exception):
    """Base class for rule-surface domain errors."""


class SurfaceNotFound(RuleSurfaceError):
    """The surface id does not exist."""


@dataclass(frozen=True)
class RuleSurface:
    surface_id: str
    rule_id: int
    score: float
    matched_terms: tuple[str, ...]
    reused: bool
    damped: bool
    ranking: Ranking
    rule: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            **dict(self.rule),
            "surface_id": self.surface_id,
            "relevance": round(self.score, 4),
            "matched_terms": list(self.matched_terms),
            "surface_reused": self.reused,
            "damped": self.damped,
            "ranking": self.ranking,
        }


@dataclass(frozen=True)
class RankedRule:
    """One active rule scored against a query, before any surface is recorded."""

    rule: dict[str, Any]
    score: float
    weight: float
    matched: tuple[str, ...]
    distinctive: tuple[str, ...]
    damped: bool
    ranking: Ranking

    @property
    def rule_id(self) -> int:
        return int(self.rule["id"])


@dataclass(frozen=True)
class FollowMark:
    surface_id: str
    rule_id: int
    followed: int | None
    note: str
    followed_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionEvidence:
    """Something the agent did: a tool call summary, a diff, a command line."""

    text: str
    at: datetime
    kind: str = ""

    def __post_init__(self) -> None:
        if self.at.tzinfo is None or self.at.utcoffset() is None:
            raise ValueError("ActionEvidence.at must be timezone-aware")


@dataclass(frozen=True)
class FollowVerdict:
    surface_id: str
    rule_id: int
    verdict: Verdict
    evidence_count: int
    overlap: tuple[str, ...]
    distinctive_overlap: tuple[str, ...]
    forbidden_hits: tuple[str, ...]
    persisted: bool

    @property
    def followed(self) -> int | None:
        return _VERDICT_VALUES[self.verdict]

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "followed": self.followed}


@dataclass(frozen=True)
class RuleEffectiveness:
    rule_id: int
    content: str
    status: str
    surfaces: int
    followed: int
    ignored: int
    not_applicable: int
    unjudged: int
    follow_rate: float | None
    ci_low: float
    ci_high: float
    p_value: float | None
    q_value: float | None
    verdict: EffectivenessVerdict
    rated_helpful: int
    rated_unhelpful: int

    @property
    def n(self) -> int:
        return self.followed + self.ignored

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "n": self.n}


@dataclass(frozen=True)
class EffectivenessReport:
    project: str | None
    min_n: int
    alpha: float
    null_follow_rate: float
    rules: tuple[RuleEffectiveness, ...]
    caveats: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        verdicts = Counter(rule.verdict for rule in self.rules)
        return {
            "project": self.project,
            "min_n": self.min_n,
            "alpha": self.alpha,
            "null_follow_rate": self.null_follow_rate,
            "summary": {name: verdicts.get(name, 0)
                        for name in ("useful", "harmful", "unverified", "insufficient")},
            "rules": [rule.to_dict() for rule in self.rules],
            "caveats": list(self.caveats),
        }


REPORT_CAVEATS = (
    "Follow rate measures adoption, not outcome: a followed rule is not proven to improve results.",
    "Observational data: rules are surfaced by relevance, so exposure is confounded with task type.",
    "Inferred verdicts come from token-overlap heuristics and can be wrong in both directions.",
    "'harmful' means consistently ignored (prompt cost without adoption), not that following it hurts.",
    ("Per-rule tests use an exact binomial test against the null follow rate, "
     "Benjamini-Hochberg corrected across the rules with enough data."),
    "Measurement only: no rule status is changed by this report.",
)


# ── Service ────────────────────────────────────────────────────────────────


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _rows(cursor: Any) -> list[dict[str, Any]]:
    names = [column[0] for column in cursor.description or ()]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def _parse_tags(raw: Any) -> list[str]:
    if not raw:
        return []
    try:
        tags = json.loads(raw)
    except (TypeError, ValueError):
        LOG.warning("rule tags are not valid JSON: %r", raw)
        return []
    return [tag for tag in tags if isinstance(tag, str)] if isinstance(tags, list) else []


def filter_phase(rows: Iterable[Mapping[str, Any]], phase: str | None) -> list[dict[str, Any]]:
    """Core rules (no ``phase:*`` tag) plus rules tagged ``phase:<phase>``."""
    if phase is None:
        return [dict(row) for row in rows]
    if phase not in VALID_PHASES:
        raise ValueError(f"Unknown phase '{phase}'. Expected one of: {sorted(VALID_PHASES)}")
    kept: list[dict[str, Any]] = []
    for row in rows:
        phase_tags = [tag for tag in _parse_tags(row.get("tags")) if tag.startswith(PHASE_TAG_PREFIX)]
        if not phase_tags or f"{PHASE_TAG_PREFIX}{phase}" in phase_tags:
            kept.append(dict(row))
    return kept


class _Corpus:
    """Document frequencies of tokens over a rule corpus."""

    def __init__(self, documents: Iterable[Sequence[str]]):
        self.size = 0
        self.df: Counter[str] = Counter()
        for tokens in documents:
            self.size += 1
            self.df.update(set(tokens))

    def idf(self, token: str) -> float:
        df = self.df.get(token, 0)
        return math.log(1.0 + (self.size - df + 0.5) / (df + 0.5))

    def distinctive(self, token: str) -> bool:
        df = self.df.get(token, 0)
        return df <= 1 or (self.size > 0 and df / self.size <= DISTINCTIVE_MAX_DF_RATIO)


class RuleSurfacer:
    """Surfaces rules, records follow verdicts and reports effectiveness."""

    def __init__(self, db: Database, clock: Callable[[], datetime] = _utc_now):
        self._db = db
        self._clock = clock

    # ── surfacing ──────────────────────────────────────────────────────

    def surface_rules(
        self,
        query: str | None,
        project: str,
        session_id: str,
        source: str,
        limit: int = DEFAULT_SURFACE_LIMIT,
        *,
        categories: Sequence[str] | None = None,
        phase: str | None = None,
    ) -> list[RuleSurface]:
        """Rank active rules in scope for ``query`` and record one surface per rule.

        Scope is ``global`` + ``project:<project>`` (+ ``category:<c>``), then the
        phase filter. Rules sharing no token with the query are left out; when no
        rule matches, or the query is empty, rules come in priority order.
        """
        limit = max(1, min(int(limit), MAX_SURFACE_LIMIT))
        ranked = self.rank_rules(query, project, categories=categories, phase=phase)
        return self.record_surfaces(ranked[:limit], query, project, session_id, source)

    def rank_rules(
        self,
        query: str | None,
        project: str,
        *,
        categories: Sequence[str] | None = None,
        phase: str | None = None,
        exclude_rule_ids: Collection[int] = (),
        lexical_only: bool = False,
    ) -> list[RankedRule]:
        """Active rules in scope ranked for ``query``, without recording anything.

        Lexical matches come first by score; with none, every rule in priority
        order (``ranking='priority'``) unless ``lexical_only``, which returns [].
        Rules in ``exclude_rule_ids`` are left out before ranking.
        """
        scopes = ["global", f"project:{project}", *(f"category:{c}" for c in categories or ())]
        placeholders = ",".join("?" for _ in scopes)
        excluded = {int(rule_id) for rule_id in exclude_rule_ids}
        candidates = [row for row in filter_phase(
            _rows(self._db.execute(
                f"SELECT * FROM rules WHERE status='active' AND scope IN ({placeholders}) "
                "ORDER BY priority DESC, id", scopes)),
            phase,
        ) if int(row["id"]) not in excluded]
        if not candidates:
            return []

        ever_followed = self._ever_followed({int(row["id"]) for row in candidates})
        documents = {int(row["id"]): tokenize(row.get("content")) for row in candidates}
        corpus = _Corpus(documents.values())
        query_tokens = set(tokenize(query))

        scored: list[RankedRule] = []
        for row in candidates:
            rule_id = int(row["id"])
            matched = tuple(token for token in documents[rule_id] if token in query_tokens)
            relevance = sum(corpus.idf(token) for token in matched)
            damped = (int(row.get("shown_count") or 0) >= DAMPING_MIN_SHOWN
                      and rule_id not in ever_followed)
            weight = max(1, int(row.get("priority") or 1)) * (DAMPING_FACTOR if damped else 1.0)
            distinctive = tuple(token for token in matched if corpus.distinctive(token))
            scored.append(RankedRule(row, relevance * weight, weight, matched, distinctive, damped, "lexical"))

        lexical = [item for item in scored if item.score > 0]
        if lexical:
            lexical.sort(key=lambda item: (-item.score, -item.weight, item.rule_id))
            return lexical
        if lexical_only:
            return []
        scored.sort(key=lambda item: (-item.weight, item.rule_id))
        return [replace(item, score=item.weight, ranking="priority") for item in scored]

    def record_surfaces(self, ranked: Sequence[RankedRule], query: str | None, project: str,
                        session_id: str, source: str) -> list[RuleSurface]:
        """Record one surface per ranked rule (reused within the session and hour) and commit."""
        stored_query = redact_secrets(query or "")[0][:MAX_STORED_QUERY_CHARS]
        now = self._clock()
        surfaces = []
        for item in ranked:
            row = item.rule
            surface_id, reused = self._surface_for(item.rule_id, project, session_id,
                                                   stored_query, source, now)
            if not reused:
                row["shown_count"] = int(row.get("shown_count") or 0) + 1
            surfaces.append(RuleSurface(surface_id, item.rule_id, item.score, item.matched, reused,
                                        item.damped, item.ranking, row))
        self._db.commit()
        return surfaces

    def _ever_followed(self, rule_ids: set[int]) -> set[int]:
        followed: set[int] = set()
        ids = sorted(rule_ids)
        for start in range(0, len(ids), SQL_IN_CHUNK):
            chunk = ids[start:start + SQL_IN_CHUNK]
            marks = ",".join("?" for _ in chunk)
            cursor = self._db.execute(
                f"SELECT DISTINCT rule_id FROM rule_surfaces WHERE followed = ? AND rule_id IN ({marks})",
                [FOLLOWED, *chunk])
            followed.update(int(row[0]) for row in cursor.fetchall())
        return followed

    def _surface_for(self, rule_id: int, project: str, session_id: str, query: str,
                     source: str, now: datetime) -> tuple[str, bool]:
        bucket_start = now.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
        bucket_end = bucket_start + SURFACE_REUSE_BUCKET
        existing = self._db.execute(
            "SELECT id FROM rule_surfaces WHERE session_id = ? AND rule_id = ? "
            "AND surfaced_at >= ? AND surfaced_at < ? ORDER BY surfaced_at LIMIT 1",
            (session_id, rule_id, format_utc(bucket_start), format_utc(bucket_end))).fetchone()
        if existing is not None:
            return str(existing[0]), True
        surface_id = f"{SURFACE_ID_PREFIX}{uuid.uuid4().hex}"
        stamp = format_utc(now)
        self._db.execute(
            "INSERT INTO rule_surfaces (id, rule_id, session_id, project, query, source, surfaced_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (surface_id, rule_id, session_id, project, query, source, stamp))
        self._db.execute(
            "UPDATE rules SET shown_count = COALESCE(shown_count, 0) + 1 WHERE id = ?", (rule_id,))
        return surface_id, False

    # ── follow verdicts ────────────────────────────────────────────────

    def mark_followed(self, surface_id: str, followed: bool | None, note: str = "",
                      applicable: bool = True) -> FollowMark:
        """Record whether the surfaced rule was followed.

        ``applicable=False`` stores -1 whatever ``followed`` says; ``followed=None``
        clears the verdict. A first transition to "followed" counts as a rule fire.
        """
        if not applicable:
            value: int | None = NOT_APPLICABLE
        elif followed is None:
            value = None
        else:
            value = FOLLOWED if followed else IGNORED
        row = self._db.execute(
            "SELECT rule_id, followed FROM rule_surfaces WHERE id = ?", (surface_id,)).fetchone()
        if row is None:
            raise SurfaceNotFound(f"Rule surface {surface_id!r} not found")
        rule_id, previous = int(row[0]), row[1]
        clean_note = redact_secrets(note or "")[0][:MAX_NOTE_CHARS]
        followed_at = format_utc(self._clock()) if value is not None else None
        self._apply_verdict(surface_id, rule_id, previous, value, clean_note, followed_at)
        self._db.commit()
        return FollowMark(surface_id, rule_id, value, clean_note, followed_at)

    def _apply_verdict(self, surface_id: str, rule_id: int, previous: int | None,
                       value: int | None, note: str, followed_at: str | None) -> None:
        self._db.execute(
            "UPDATE rule_surfaces SET followed = ?, follow_note = ?, followed_at = ? WHERE id = ?",
            (value, note, followed_at, surface_id))
        if value == FOLLOWED and previous != FOLLOWED:
            self._db.execute(
                "UPDATE rules SET fire_count = COALESCE(fire_count, 0) + 1, last_fired = ? "
                "WHERE id = ?", (followed_at, rule_id))

    def followcheck(
        self,
        session_id: str,
        actions: Sequence[ActionEvidence],
        window: timedelta = DEFAULT_FOLLOW_WINDOW,
        *,
        persist: bool = True,
    ) -> list[FollowVerdict]:
        """Infer a verdict for each surface of ``session_id`` from action evidence.

        Evidence counts for a surface only when it happened at or after the surface
        and within ``window`` of it. The same instant counts: hooks run in order, so
        an action stamped with the surface's time came after it on a coarse clock
        (about 15.6 ms on Windows before Python 3.13). Rules: a prohibition whose forbidden object
        appears in one action -> ignored; >= 3 overlapping tokens with >= 2
        distinctive -> followed; evidence with no overlap at all -> not
        applicable; otherwise unknown. With ``persist`` the decisive verdicts are
        written to surfaces that have no verdict yet (explicit marks win).
        """
        if window <= timedelta(0):
            raise ValueError("window must be positive")
        if not actions:
            return []
        stamped = sorted(((action.at.astimezone(UTC), action) for action in actions),
                         key=lambda item: item[0])
        earliest, latest = stamped[0][0], stamped[-1][0]
        surfaces = _rows(self._db.execute(
            "SELECT s.id, s.rule_id, s.surfaced_at, s.followed, r.content "
            "FROM rule_surfaces s JOIN rules r ON r.id = s.rule_id "
            "WHERE s.session_id = ? AND s.surfaced_at >= ? AND s.surfaced_at <= ? "
            "ORDER BY s.surfaced_at, s.id",
            (session_id, format_utc(earliest - window), format_utc(latest))))
        if not surfaces:
            return []
        corpus = _Corpus(tokenize(row[0]) for row in self._db.execute(
            "SELECT content FROM rules WHERE status='active'").fetchall())
        action_tokens = [(at, set(tokenize(action.text))) for at, action in stamped]

        verdicts = []
        for surface in surfaces:
            verdict = self._judge(surface, action_tokens, window, corpus)
            persisted = False
            if persist and verdict.followed is not None and surface["followed"] is None:
                self._apply_verdict(surface["id"], int(surface["rule_id"]), None, verdict.followed,
                                    f"followcheck: {verdict.verdict}", format_utc(self._clock()))
                persisted = True
            verdicts.append(replace(verdict, persisted=persisted))
        if persist:
            self._db.commit()
        return verdicts

    @staticmethod
    def _judge(surface: Mapping[str, Any], action_tokens: Sequence[tuple[datetime, set[str]]],
               window: timedelta, corpus: _Corpus) -> FollowVerdict:
        surfaced = datetime.fromisoformat(str(surface["surfaced_at"]))
        evidence = [tokens for at, tokens in action_tokens if surfaced <= at <= surfaced + window]
        surface_id, rule_id = str(surface["id"]), int(surface["rule_id"])
        if not evidence:
            return FollowVerdict(surface_id, rule_id, "unknown", 0, (), (), (), False)

        rule_tokens = tokenize(surface.get("content"))
        forbidden = forbidden_terms(surface.get("content"))
        if forbidden:
            needed = min(FORBIDDEN_MIN_OVERLAP, len(forbidden))
            for tokens in evidence:
                hits = forbidden & tokens
                if len(hits) >= needed:
                    return FollowVerdict(surface_id, rule_id, "ignored", len(evidence), (), (),
                                         tuple(sorted(hits)), False)

        seen = set().union(*evidence)
        overlap = tuple(token for token in rule_tokens if token in seen)
        distinctive = tuple(token for token in overlap if corpus.distinctive(token))
        if len(overlap) >= FOLLOW_MIN_OVERLAP and len(distinctive) >= FOLLOW_MIN_DISTINCTIVE:
            verdict: Verdict = "followed"
        elif not overlap:
            verdict = "not_applicable"
        else:
            verdict = "unknown"
        return FollowVerdict(surface_id, rule_id, verdict, len(evidence), overlap, distinctive,
                             (), False)

    # ── measurement ────────────────────────────────────────────────────

    def effectiveness_report(self, project: str | None = None,
                             min_n: int = DEFAULT_REPORT_MIN_N) -> EffectivenessReport:
        """Per-rule follow statistics with Wilson 95% CIs and BH-corrected verdicts."""
        if min_n < 1:
            raise ValueError("min_n must be at least 1")
        where, params = ("WHERE project = ?", [project]) if project else ("", [])
        counts = _rows(self._db.execute(
            "SELECT rule_id, COUNT(*) AS surfaces, "
            "SUM(CASE WHEN followed = 1 THEN 1 ELSE 0 END) AS followed, "
            "SUM(CASE WHEN followed = 0 THEN 1 ELSE 0 END) AS ignored, "
            "SUM(CASE WHEN followed = -1 THEN 1 ELSE 0 END) AS not_applicable, "
            "SUM(CASE WHEN followed IS NULL THEN 1 ELSE 0 END) AS unjudged "
            f"FROM rule_surfaces {where} GROUP BY rule_id ORDER BY rule_id", params))
        rules = self._rules_by_id([int(row["rule_id"]) for row in counts])

        tested = [row for row in counts if int(row["followed"] or 0) + int(row["ignored"] or 0) >= min_n]
        p_values = [binomial_two_sided_p(int(row["followed"] or 0),
                                         int(row["followed"] or 0) + int(row["ignored"] or 0))
                    for row in tested]
        q_by_rule = dict(zip((int(row["rule_id"]) for row in tested),
                             zip(p_values, benjamini_hochberg(p_values), strict=True), strict=True))

        results = []
        for row in counts:
            rule_id = int(row["rule_id"])
            followed, ignored = int(row["followed"] or 0), int(row["ignored"] or 0)
            n = followed + ignored
            low, high = wilson_interval(followed, n)
            rate = followed / n if n else None
            p_value, q_value = q_by_rule.get(rule_id, (None, None))
            if q_value is None:
                verdict: EffectivenessVerdict = "insufficient"
            elif q_value < REPORT_ALPHA and rate is not None and rate > REPORT_NULL_FOLLOW_RATE:
                verdict = "useful"
            elif q_value < REPORT_ALPHA and rate is not None and rate < REPORT_NULL_FOLLOW_RATE:
                verdict = "harmful"
            else:
                verdict = "unverified"
            rule = rules.get(rule_id, {})
            results.append(RuleEffectiveness(
                rule_id=rule_id,
                content=str(rule.get("content") or ""),
                status=str(rule.get("status") or "deleted"),
                surfaces=int(row["surfaces"] or 0),
                followed=followed,
                ignored=ignored,
                not_applicable=int(row["not_applicable"] or 0),
                unjudged=int(row["unjudged"] or 0),
                follow_rate=rate,
                ci_low=low,
                ci_high=high,
                p_value=p_value,
                q_value=q_value,
                verdict=verdict,
                rated_helpful=int(rule.get("success_count") or 0),
                rated_unhelpful=int(rule.get("fail_count") or 0),
            ))
        results.sort(key=lambda item: (item.q_value is None, item.q_value or 0.0, -item.n, item.rule_id))
        return EffectivenessReport(project, min_n, REPORT_ALPHA, REPORT_NULL_FOLLOW_RATE,
                                   tuple(results), REPORT_CAVEATS)

    def _rules_by_id(self, rule_ids: Sequence[int]) -> dict[int, dict[str, Any]]:
        found: dict[int, dict[str, Any]] = {}
        for start in range(0, len(rule_ids), SQL_IN_CHUNK):
            chunk = list(rule_ids[start:start + SQL_IN_CHUNK])
            marks = ",".join("?" for _ in chunk)
            for row in _rows(self._db.execute(
                    f"SELECT id, content, status, success_count, fail_count FROM rules "
                    f"WHERE id IN ({marks})", chunk)):
                found[int(row["id"])] = row
        return found
