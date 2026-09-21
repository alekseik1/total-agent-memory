from __future__ import annotations

import re
from functools import lru_cache

import snowballstemmer

MAX_LEXICAL_TERMS = 32
QUESTION_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "how",
        "when",
        "where",
        "why",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "to",
        "of",
        "in",
        "on",
        "at",
        "for",
        "with",
        "and",
        "or",
        "as",
        "by",
        "about",
        "can",
        "could",
        "would",
        "should",
        "will",
        "please",
        "tell",
        "me",
        "know",
        "remember",
        "что",
        "какой",
        "какая",
        "какие",
        "какое",
        "кто",
        "кого",
        "кому",
        "чей",
        "чья",
        "чьи",
        "как",
        "когда",
        "где",
        "куда",
        "почему",
        "зачем",
        "это",
        "есть",
        "был",
        "была",
        "были",
        "быть",
        "в",
        "во",
        "на",
        "для",
        "от",
        "до",
        "по",
        "из",
        "у",
        "и",
        "или",
        "а",
        "о",
        "об",
        "про",
        "с",
        "со",
        "мне",
        "меня",
        "ты",
        "вы",
        "пожалуйста",
        "расскажи",
        "напомни",
    ]
)
_TERM = re.compile(r"\w+(?:[./:-]\w+)*", re.UNICODE)
_CYRILLIC_WORD = re.compile(r"[а-яё]+")
# A shorter stem ("по", "на") would prefix-match most of the index.
MIN_PREFIX_STEM = 3


def lexical_terms(query: str) -> list[str]:
    terms = list(dict.fromkeys(_TERM.findall(query.casefold())))
    meaningful = [term for term in terms if term not in QUESTION_STOPWORDS]
    return (meaningful or terms)[:MAX_LEXICAL_TERMS]


@lru_cache(maxsize=1)
def _russian_stemmer() -> snowballstemmer.stemmer:
    return snowballstemmer.stemmer("russian")


def russian_stem(word: str) -> str:
    """Snowball stem of a lower-case Cyrillic word; other words come back unchanged."""
    return _russian_stemmer().stemWord(word) if _CYRILLIC_WORD.fullmatch(word) else word


def _fts_phrase(text: str) -> str:
    return '"' + text.replace('"', '""') + '"'


def fts_match_query(query: str) -> str:
    """FTS5 MATCH expression for a free-text query.

    unicode61 matches exact word forms, so Russian terms become stem prefix
    queries ("Маша" -> "маш"*), which also match "Маше" and "Машу". Other
    terms stay exact phrases.
    """
    parts: list[str] = []
    for term in lexical_terms(query):
        stem = russian_stem(term)
        if stem != term and len(stem) >= MIN_PREFIX_STEM:
            parts.append(_fts_phrase(stem) + "*")
        else:
            parts.append(_fts_phrase(term))
    return " OR ".join(dict.fromkeys(parts)) or _fts_phrase(query)
