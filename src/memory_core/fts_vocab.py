"""Does the full-text index know any content word of the query?

A question in one language over records in another ("what does my sister
like" over Russian notes) still gets full-text matches, but only on function
words and short tokens the two share; those matches crowd the fused ranking
while the semantic tier, which reads across languages, holds the answer.
`vocabulary_overlap` answers whether at least one content word of the query
(a token FTS5 would index, at least MIN_CONTENT_TOKEN_CHARS long, not a
function word, not a number) occurs in `knowledge_fts`; recall damps the fts
tier when none does.

SQLite reads the vocabulary through an fts5vocab table in the connection's
temp schema (nothing is written to the database file). PostgreSQL reads the
`knowledge_tsv` side table that replaces FTS5 there (memory_core.pg_fts).
Russian words are looked up by their Snowball stem as a prefix, as
`query_terms.fts_match_query` matches them.
"""

from __future__ import annotations

from memory_core.pg_fts import fts5_tokens
from memory_core.query_terms import lexical_terms, russian_stem

MIN_CONTENT_TOKEN_CHARS = 3
MAX_CONTENT_TOKENS = 16
VOCAB_TABLE = "knowledge_fts_vocab"
# Upper bound of a prefix range over fts5vocab terms.
PREFIX_RANGE_END = "\U0010ffff"
PG_TEXT_COLUMNS = ("content", "context", "tags")

FUNCTION_WORDS = frozenset({
    # English
    "about", "above", "after", "again", "all", "also", "and", "any", "are", "because", "been", "before", "being",
    "both", "but", "can", "could", "did", "does", "doing", "done", "each", "else", "ever", "every", "few", "for",
    "from", "further", "get", "got", "had", "has", "have", "having", "her", "hers", "herself", "him", "himself",
    "his", "how", "into", "its", "itself", "just", "let", "like", "many", "may", "might", "more", "most", "much",
    "must", "my", "myself", "near", "never", "nor", "not", "now", "off", "once", "one", "only", "other", "our",
    "ours", "out", "over", "own", "same", "she", "should", "since", "some", "still", "such", "than", "that", "the",
    "their", "theirs", "them", "then", "there", "these", "they", "this", "those", "through", "too", "under",
    "until", "use", "used", "very", "was", "way", "well", "were", "what", "when", "where", "which", "while", "who",
    "whom", "whose", "why", "will", "with", "would", "yes", "yet", "you", "your", "yours", "yourself",
    # Russian
    "без", "был", "была", "были", "было", "быть", "вам", "вас", "весь", "вот", "все", "всё", "всех", "где",
    "даже", "для", "его", "ее", "её", "если", "есть", "ещё", "еще", "здесь", "или", "как", "какая", "какие",
    "какой", "какое", "когда", "кто", "куда", "либо", "меня", "мне", "мой", "моя", "мои", "мы", "над", "нам",
    "нас", "наш", "наша", "наши", "него", "нее", "неё", "нет", "них", "ним", "она", "они", "оно", "под", "после",
    "при", "про", "раз", "сам", "свой", "себя", "так", "также", "там", "тебе", "тебя", "тоже", "только", "тот",
    "три", "уже", "чем", "чего", "что", "чтобы", "чей", "чья", "эта", "эти", "это", "этот", "этого",
})


def content_tokens(query: str) -> list[str]:
    """Index tokens of the query's content words, in query order (at most MAX_CONTENT_TOKENS)."""
    tokens: dict[str, None] = {}
    for term in lexical_terms(query or ""):
        for token in fts5_tokens(term):
            if (len(token) >= MIN_CONTENT_TOKEN_CHARS and not token.isdigit()
                    and token not in FUNCTION_WORDS):
                tokens.setdefault(token, None)
    return list(tokens)[:MAX_CONTENT_TOKENS]


def _lookup_form(token: str) -> tuple[str, bool]:
    """(term, prefix): Russian words by stem prefix, others exactly."""
    stem = russian_stem(token)
    if stem != token and len(stem) >= MIN_CONTENT_TOKEN_CHARS:
        return stem, True
    return token, False


def vocabulary_overlap(db, query: str, *, is_postgres: bool) -> bool | None:
    """True when a content word of `query` is in the knowledge index; None when the query has none."""
    tokens = content_tokens(query)
    if not tokens:
        return None
    forms = [_lookup_form(token) for token in tokens]
    if is_postgres:
        return _pg_overlap(db, forms)
    return _sqlite_overlap(db, forms)


def _sqlite_overlap(db, forms: list[tuple[str, bool]]) -> bool:
    db.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS temp.{VOCAB_TABLE} USING fts5vocab(main, knowledge_fts, 'row')")
    for term, prefix in forms:
        if prefix:
            row = db.execute(f"SELECT 1 FROM temp.{VOCAB_TABLE} WHERE term >= ? AND term < ? LIMIT 1",
                             (term, term + PREFIX_RANGE_END)).fetchone()
        else:
            row = db.execute(f"SELECT 1 FROM temp.{VOCAB_TABLE} WHERE term = ? LIMIT 1", (term,)).fetchone()
        if row is not None:
            return True
    return False


def _pg_overlap(db, forms: list[tuple[str, bool]]) -> bool:
    query = " | ".join("'" + term.replace("\\", "\\\\").replace("'", "''") + "'" + (":*" if prefix else "")
                       for term, prefix in forms)
    match = " OR ".join(f"{column}_lexemes @@ CAST(? AS tsquery)" for column in PG_TEXT_COLUMNS)
    row = db.execute(f"SELECT 1 FROM knowledge_tsv WHERE {match} LIMIT 1",
                     [query] * len(PG_TEXT_COLUMNS)).fetchone()
    return row is not None
