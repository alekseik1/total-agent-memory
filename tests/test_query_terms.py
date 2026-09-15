import sqlite3

from memory_core.query_terms import MAX_LEXICAL_TERMS, lexical_terms


def test_question_filler_does_not_outrank_factual_match():
    db = sqlite3.connect(":memory:")
    db.execute("CREATE VIRTUAL TABLE docs USING fts5(content)")
    db.executemany(
        "INSERT INTO docs(content) VALUES (?)",
        [
            ("What does the team do? How does it work? What is it?",),
            ("The deployment uses PostgreSQL replication.",),
        ],
    )
    terms = lexical_terms("What does the deployment use?")
    query = " OR ".join(f'"{term}"' for term in terms)
    rows = db.execute(
        "SELECT rowid FROM docs WHERE docs MATCH ? ORDER BY bm25(docs)", (query,)
    ).fetchall()
    assert rows == [(2,)]
    db.close()


def test_keep_negation_identifiers_and_short_language_names():
    assert lexical_terms("Why does Go not use src/auth.py?") == [
        "go",
        "not",
        "use",
        "src/auth.py",
    ]
    assert lexical_terms("Почему не работает Go?") == ["не", "работает", "go"]


def test_deduplicate_bound_and_fallback():
    assert lexical_terms("WHERE where") == ["where"]
    assert lexical_terms("!!!") == []
    assert (
        len(lexical_terms(" ".join(f"term{i}" for i in range(100))))
        == MAX_LEXICAL_TERMS
    )
