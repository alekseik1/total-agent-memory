"""Query shapes that change how `memory_recall` searches.

Cheap, regex-only classifiers (English and Russian). The `directives` and
`multi_query` tiers use the first two:

* an *advice* query asks what to do or what to pick ("how should I...",
  "recommend...", "как лучше..."); conventions and preferences of the
  project answer it even when they share no words with the question;
* an *ordering or comparison* query names several things and asks about
  their order or difference ("did X happen before Y", "X or Y", "X vs Y");
  each side is searched on its own, because one combined query ranks
  records that mention both sides, which rarely exist.

The rest steer later stages: a *historical-value* question ("what was it
before?") reverses the value-update order (Stage 4.9); a *compound* query is
split into parts that each get a slot in the top K; an *intent* (decision,
lesson, how-to) multiplies the fused score of the record types that answer it.
"""

from __future__ import annotations

import re

from memory_core.query_terms import lexical_terms

MAX_SUB_QUERIES = 3
MIN_SUB_QUERY_TERMS = 1

_ADVICE = re.compile(
    r"\b("
    r"recommend(?:ation)?s?|suggest(?:ion)?s?|advi[cs]e|"
    r"should (?:i|we|one|you)|how (?:should|do|can) (?:i|we)|"
    r"what(?:'s| is) the (?:best|right|proper|usual|standard|preferred|recommended)|"
    r"best (?:way|practice|option|choice)|"
    r"which (?:one |\w+ )?(?:should|to) (?:use|pick|choose|prefer)|"
    r"convention|guideline|prefer(?:red|ence)?s?|rule of thumb|"
    r"стоит ли|как (?:лучше|правильно|принято|нужно)|что (?:лучше|выбрать|использовать)|"
    r"посоветуй|порекомендуй|рекоменд\w*|какой (?:выбрать|использовать)|по какому (?:правилу|принципу)|"
    r"конвенци\w*|предпочт\w*"
    r")\b",
    re.IGNORECASE,
)

_ORDERING_CUE = re.compile(
    r"\b("
    r"first|last|earlier|later|before|after|order|sequence|which came|happened first|"
    r"vs\.?|versus|compared? (?:to|with)|difference between|or|"
    r"сначала|раньше|позже|до|после|порядок|последовательност\w*|что было (?:первым|раньше)|"
    r"или|сравни\w*|в отличие от|разниц\w* между"
    r")\b",
    re.IGNORECASE,
)

_SPLIT = re.compile(
    r"\s*(?:,|;|:|\?|"
    r"\b(?:or|vs\.?|versus|before|after|earlier than|later than|compared (?:to|with)|and then|then|"
    r"или|до того как|после того как|до|после|раньше чем|позже чем|а потом|потом|и затем)\b"
    r")\s*",
    re.IGNORECASE,
)

_LEADING_FILLER = re.compile(
    r"^(?:(?:which|what|who|when|did|was|were|is|are|does|do|had|has|how)\s+)+"
    r"|^(?:(?:что|кто|когда|какой|какая|какое|какие|было|был|была|были)\s+)+",
    re.IGNORECASE,
)

_TRAILING_FILLER = re.compile(
    r"\s+(?:happen(?:ed)?|come|came|occur(?:red)?|first|last|earlier|later|произошл[оаи]|случил[оаи]сь|сначала)\s*$",
    re.IGNORECASE,
)


def is_advice_query(query: str) -> bool:
    """True when the query asks for a recommendation, rule or preferred choice."""
    return bool(query) and _ADVICE.search(query) is not None


_FILLER_TERMS = frozenset({
    "happened", "happen", "came", "come", "occurred", "occur", "first", "last", "earlier", "later",
    "раньше", "позже", "сначала", "было", "произошло", "случилось", "первым", "потом",
})


def _informative(part: str) -> bool:
    terms = lexical_terms(part)
    return len(terms) >= MIN_SUB_QUERY_TERMS and any(term.lower() not in _FILLER_TERMS for term in terms)


def sub_queries(query: str) -> list[str]:
    """Split an ordering/comparison question into the things it compares.

    Returns an empty list when the query has no ordering cue or fewer than
    two parts with at least MIN_SUB_QUERY_TERMS lexical terms each; the
    caller then runs the plain pipeline only.
    """
    if not query or _ORDERING_CUE.search(query) is None:
        return []
    parts: list[str] = []
    for raw in _SPLIT.split(query):
        part = _TRAILING_FILLER.sub("", _LEADING_FILLER.sub("", (raw or "").strip())).strip(" .!")
        if _informative(part) and part.lower() not in {p.lower() for p in parts}:
            parts.append(part)
    if len(parts) < 2:
        return []
    return parts[:MAX_SUB_QUERIES]


# ── historical-value questions ("what was it before") ──────────────────────
#
# Stage 4.9 of recall puts the newest of several values of one statement
# first. A question about the earlier value wants the opposite order. Three
# kinds of cue, checked in this order:
#
# 1. a comparison frame ("before or after", "earlier than", "что было
#    раньше: X или Y") is an ordering question, never a historical one;
# 2. a strong cue names the earlier state on its own: "previously",
#    "used to", "my old address", "раньше", "бывший", "когда-то";
# 3. a weak cue ("before the migration", "at first", "сначала", "до
#    переезда") counts only in a question that is not a procedure: "before
#    deploying, run tests" and "should I back up before migrating?" are
#    instructions, "what did we use before the migration?" is history.

_HISTORICAL_COMPARISON = re.compile(
    r"\b(?:before|after|earlier|later)\s+(?:or|than)\b"
    r"|\bor\s+(?:before|after|earlier|later)\b"
    r"|\b(?:came|happened|occurred|was\s+done)\s+(?:first|earlier)\b"
    r"|\b(?:до|после|раньше|позже|ранее)\s*,?\s*(?:или|чем)\b"
    r"|\bили\s+(?:до|после|раньше|позже)\b"
    r"|\b(?:что|кто)\s+(?:было|был|была|произошло|случилось)\s+(?:раньше|сначала|первым)\b",
    re.IGNORECASE,
)

# "What car did he get after his old Prius broke down?", "what do we use now instead of the old
# system?": the question is about the later state even though it names the earlier one.
_LATER_STATE = re.compile(
    r"\b(?:after|afterwards|since|now|currently|nowadays|today|instead\s+of|replaced?\s+by)\b"
    r"|\b(?:после|теперь|сейчас|сегодня|вместо|нынешн\w*|текущ\w*)\b",
    re.IGNORECASE,
)

_HISTORICAL_STRONG = re.compile(
    r"\b(?:previously|formerly|originally|initially|beforehand|in\s+the\s+past|back\s+then|no\s+longer|anymore)\b"
    # "used to" as a past habit has a person as subject ("we used to", "did Anna use to");
    # "the tool used to deploy" (purpose) and "get used to" do not. Names: _USED_TO_NAMED.
    r"|\b(?:i|we|you|he|she|they)\s+used\s+to\b"
    r"|\bdid\s+\w+\s+used?\s+to\s+(?:be|live|work|have|own|like|love|enjoy|go|do|drive|run|play|eat|"
    r"watch|read|listen|believe|think|feel|call|date|study|teach|deploy|host|store|use|wear|visit|travel|"
    r"collect|practice|practise|attend|sing|paint|write|cook)\b"
    r"|\b(?:my|our|your|his|her|their|the)\s+(?:old|previous|former|prior|earlier|original)\s+\w+"
    r"|\b(?:previous|former)\s+\w+"
    r"|\bwhat\s+(?:was|were)\s+(?:it|that|they)\s+(?:before|earlier)\b"
    r"|(?<!можно\s)\bраньше\b|(?<!можно\s)\bранее\b|\bпрежде\b(?!\s*,?\s*чем)|\bизначально\b|\bпервоначально\b"
    r"|\bкогда-то\b|\bв\s+прошлом\b|\bбольше\s+не\b|\bдо\s+этого\b"
    r"|\bбывш\w+|(?<!по-)\bпрежн\w+|\bстар(?:ый|ая|ое|ые|ого|ой|ую|ым|ых|ыми|ому|ом)\b",
    re.IGNORECASE,
)

_HISTORICAL_WEAK = re.compile(
    r"\bbefore\s*(?:[?!.]|$)"
    r"|\bbefore\s+(?:that|this|then|it|the\s+(?:change|switch|migration|move|update|upgrade|rename|rewrite|refactor))\b"
    r"|\bbefore\s+(?:(?:i|we|you|he|she|they|it)\s+)?"
    r"(?:chang|switch|migrat|mov|updat|upgrad|replac|renam|rewr|refactor|adopt|introduc)\w*"
    r"|\bat\s+first\b(?!\s+(?:of|sight|glance))|\bin\s+the\s+beginning\b(?!\s+of)"
    r"|\bсначала\b|\bвначале\b|\bпоначалу\b|\bпрежде\s*,?\s*чем\b"
    r"|\bдо\s+(?:перехода|переезда|миграции|смены|замены|обновления|переименования|переписывания|рефакторинга|того)\b",
    re.IGNORECASE,
)

_QUESTION_FORM = re.compile(
    r"\?|^\s*(?:what|which|who|whom|whose|where|when|why|how|did|was|were|is|are|had|has)\b"
    r"|^\s*(?:do|does|have)\s+(?:i|we|you|he|she|they|it)\b"
    r"|^\s*(?:что|какой|какая|какое|какие|каким|какую|где|когда|кто|кого|как|сколько|чем|почему|зачем|куда|откуда)\b"
    r"|\bли\b",
    re.IGNORECASE,
)

_PROCEDURE = re.compile(
    r"\b(?:should|must|need(?:s)?\s+to|have\s+to|has\s+to|remember\s+to|don't\s+forget|do\s+not\s+forget|make\s+sure|"
    r"how\s+(?:do|to|can|should)|is\s+it\s+(?:ok|safe|necessary)|required\s+to)\b"
    r"|\b(?:нужно|надо|нельзя|стоит|следует|должен|должна|должны|обязательно|не\s+забудь|как\s+правильно|как\s+лучше|"
    r"можно\s+ли|нужно\s+ли|надо\s+ли|стоит\s+ли)\b",
    re.IGNORECASE,
)


# "Where Anna used to live": a capitalised name (not the first word) right before "used to".
_USED_TO_NAMED = re.compile(r"(?<=\s)[A-ZА-ЯЁ][a-zа-яё]+\s+used\s+to\b")

# Phrases that look like cues but name no earlier value; removed before the cue
# search. A reference to the dialogue itself ("in our previous conversation you
# recommended...", "в прошлом разговоре") and a time window ("in the past three
# months", "my previous year's time") fired on 16 of 95 LongMemEval-S dev questions.
_HISTORICAL_NEUTRAL = re.compile(
    r"\b(?:previous|earlier|last|prior|former)\s+(?:conversations?|chats?|discussions?|sessions?|messages?|"
    r"exchanges?|talks?|threads?|replies|reply|answers?|responses?)\b"
    r"|\bin\s+the\s+(?:past|last)\s+(?:(?:few|couple(?:\s+of)?|several|\d+|one|two|three|four|five|six|seven|"
    r"eight|nine|ten|eleven|twelve)\s+)?(?:days?|weeks?|weekends?|months?|years?|decades?)\b"
    r"|\b(?:previous|prior|past)\s+(?:day|week|weekend|month|year|decade|night|morning|evening)s?(?:'s)?\b"
    r"|\b(?:прошл|предыдущ)\w*\s+(?:разговор|чат|диалог|бесед|сообщени|переписк|сесси|обсуждени)\w*",
    re.IGNORECASE,
)


def is_historical_value_query(query: str) -> bool:
    """True when the query asks for an earlier value of something that changed.

    Comparison frames ("before or after") and anchors to the later state
    ("after", "now", "instead of") are excluded first; a strong cue decides
    alone; a weak cue needs a question that is not a procedure.
    """
    if not query or not query.strip():
        return False
    if _HISTORICAL_COMPARISON.search(query) or _LATER_STATE.search(query):
        return False
    query = _HISTORICAL_NEUTRAL.sub(" ", query)
    if _HISTORICAL_STRONG.search(query) or _USED_TO_NAMED.search(query):
        return True
    return (_HISTORICAL_WEAK.search(query) is not None
            and _QUESTION_FORM.search(query) is not None
            and _PROCEDURE.search(query) is None)


# ── compound queries ("X and Y", "X, also Y", "X, а также Y") ─────────────
#
# A query that asks for two things is answered by two records; fused into one
# ranking, the better-matched half can push the other out of the top K. The
# split is used by the compound-slots stage of recall (MEMORY_RECALL_COMPOUND_SLOTS).
# Clause-level separators ("; ", "? ", ", also", "а также") always split;
# "and" / ", and" / "as well as" / "и" split only before a new question
# ("... and how do we deploy", "... и где лежат логи") with
# MIN_COMPOUND_PART_TERMS content words on both sides, so "Caroline and
# Melanie", "her family and friends" or a list "A, B, and C" stay one part.

MAX_COMPOUND_PARTS = 3
MIN_COMPOUND_PART_TERMS = 2

_COMPOUND_STRONG = re.compile(
    r"\s*(?:[;?]\s+(?=\S)|,\s*(?:and\s+)?also\b|\band\s+also\b|,?\s*а\s+также\b|,\s*и\s+ещё\b|,\s*и\s+еще\b)\s*",
    re.IGNORECASE,
)
# A new question after "and" / "и": a question word, or an auxiliary with its subject.
_CLAUSE_START = (
    r"(?=(?:what|which|who|whom|whose|where|when|why|how|"
    r"(?:is|are|was|were|do|does|did|can|could|should|will|would|has|have|had)\s+"
    r"(?:i|we|you|he|she|they|it)|"
    r"что|как|где|когда|кто|куда|откуда|почему|зачем|сколько|какой|какая|какое|какие|каким|какую|чем)\b)"
)
_COMPOUND_WEAK = re.compile(
    r"\s*(?:,\s*(?:and|plus|и|плюс)|\s(?:and|и)|,?\s+as\s+well\s+as)\s+" + _CLAUSE_START,
    re.IGNORECASE,
)


# "Answer yes or no." and the like instruct the reader; they are not a part to search for.
_META_PART = re.compile(r"^\s*(?:answer|reply|respond|ответь|ответьте)\b", re.IGNORECASE)


def _content_terms(part: str) -> list[str]:
    return [term for term in lexical_terms(part) if term.lower() not in _FILLER_TERMS]


def compound_parts(query: str) -> list[str]:
    """Split a query that asks for several things into its parts (empty list = not compound)."""
    if not query or not query.strip():
        return []
    parts: list[str] = []
    for chunk in _COMPOUND_STRONG.split(query):
        chunk = (chunk or "").strip(" .!?,;")
        if not chunk:
            continue
        pieces = _COMPOUND_WEAK.split(chunk)
        if len(pieces) > 1 and all(len(_content_terms(piece)) >= MIN_COMPOUND_PART_TERMS for piece in pieces):
            parts.extend(piece.strip(" .!?,;") for piece in pieces)
        else:
            parts.append(chunk)
    kept: list[str] = []
    for part in parts:
        if _META_PART.match(part):
            continue
        if len(_content_terms(part)) >= MIN_COMPOUND_PART_TERMS and part.lower() not in {p.lower() for p in kept}:
            kept.append(part)
    if len(kept) < 2:
        return []
    return kept[:MAX_COMPOUND_PARTS]


# ── intents that name the record type answering the question ───────────────
#
# config.RECALL_INTENT_TYPES maps each intent to record types; recall
# multiplies their fused score (config.get_recall_intent_boosts).

INTENT_DECISION = "decision"
INTENT_LESSON = "lesson"
INTENT_HOWTO = "howto"

_INTENT_PATTERNS: dict[str, re.Pattern[str]] = {
    INTENT_DECISION: re.compile(
        r"\bwhat\s+(?:did|have|had)\s+(?:we|i|you|they|the\s+team)\s+(?:decid\w*|agree\w*|settle\w*)"
        r"|\b(?:decision|decisions|decided|rationale|trade-?offs?)\b"
        r"|\bwhy\s+(?:did|do|have|had|was|were|is|are)\s+(?:we|i|you|they|the\s+team|it)\s+"
        r"(?:\w+\s+)?(?:choose|chose|chosen|pick|picked|select|selected|opt|opted|go\s+with|went\s+with|gone\s+with|"
        r"use|used|using|switch|switched|prefer|preferred|adopt|adopted|move|moved|migrate|migrated)\b"
        r"|\bwhy\s+(?:was|were|is|are)\s+\S+(?:\s+\S+)?\s+(?:chosen|picked|selected|adopted|preferred)\b"
        r"|\bкакое\s+решение\b|\bчто\s+(?:мы\s+|я\s+|они\s+)?(?:решили|решил|решила|постановили)\b"
        r"|\bприн(?:яли|ял|яла|ято)\s+решени\w*|\bпочему\s+(?:мы\s+|я\s+|они\s+)?(?:выбрал\w*|выбра\w*|остановил\w*|"
        r"перешл\w*|перешёл|предпоч\w*|решил\w*|отказал\w*)|\bобоснован\w*",
        re.IGNORECASE,
    ),
    INTENT_LESSON: re.compile(
        # "lesson" also names a class ("scuba diving lessons"); only the learning sense counts.
        r"\blessons?\s+(?:learned|learnt|from|of)\b|\b(?:what|which|any|key|main)\s+lessons?\b"
        r"|\bwhat\s+(?:did|have)\s+(?:we|i|you)\s+learn\w*|\btakeaways?\b"
        # "convention" also names an event ("a gaming convention"); only the rule sense counts.
        r"|\brules?\b|\b(?:our|team|coding|naming|code|style|project|commit|branch)\s+conventions?\b"
        r"|\bconventions?\s+(?:for|on|about|around)\b|\bguidelines?\b|\bbest\s+practices?\b"
        r"|\bpitfalls?\b|\bgotchas?\b|\bwhat\s+(?:to|should\s+(?:i|we))\s+avoid\b|\bmistakes?\b"
        r"|\bурок\w*|\bвывод\w*|\bчему\s+(?:мы\s+|я\s+)?научил\w*|\bправил\w*|\bконвенци\w*|\bсоглашени\w*"
        r"|\bграбл\w*|\bподводн\w+\s+камн\w*|\bчего\s+(?:нужно\s+|надо\s+|стоит\s+)?избегать\b",
        re.IGNORECASE,
    ),
    INTENT_HOWTO: re.compile(
        r"\bhow\s+(?:do|did|can|could|should|would|to)\s+(?:(?:i|we|you|one)\s+)?"
        r"(?:fix|fixed|solve|solved|resolve|resolved|handle|handled|set\s+up|setup|configure|configured|install|"
        r"installed|deploy|deployed|debug|debugged|work\s+around|worked\s+around|get\s+rid|got\s+rid|repair|"
        r"patch|patched|recover|recovered|migrate|migrated|troubleshoot)\b"
        r"|\b(?:solution|solutions|workaround|workarounds|fix\s+for|troubleshoot\w*)\b"
        r"|\bhow\s+(?:was|were)\s+\S+(?:\s+\S+)?\s+(?:fixed|solved|resolved)\b"
        r"|\bкак\s+(?:мы\s+|я\s+)?(?:починить|починили|починил|исправить|исправили|исправил|решить|решили|"
        r"настроить|настроили|установить|задеплоить|обойти|обошли|победить|победили|поднять|подняли|"
        r"восстановить|восстановили|отладить|мигрировать)\b"
        r"|\bрешение\s+(?:проблемы|ошибки|бага)\b|\bобходн\w+\s+пут\w*|\bфикс\w*|\bворкараунд\w*",
        re.IGNORECASE,
    ),
}


def query_intents(query: str) -> frozenset[str]:
    """Intents of the query: INTENT_DECISION, INTENT_LESSON and/or INTENT_HOWTO (empty when none)."""
    if not query or not query.strip():
        return frozenset()
    return frozenset(name for name, pattern in _INTENT_PATTERNS.items() if pattern.search(query))


_USER_TURN = re.compile(r"^\s*(?:\[[^\]]*\]\s*)?(?:user|human|пользователь)\s*:", re.IGNORECASE)


def is_user_turn(content: str) -> bool:
    """True for a conversation record spoken by the user ("[date] user: ...")."""
    return bool(content) and _USER_TURN.match(content) is not None
