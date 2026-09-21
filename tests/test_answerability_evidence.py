import json

import pytest

from ai_layer.answerability import _build_user_prompt, classify_answerability


@pytest.mark.parametrize("padding", [30, 300])
def test_classifier_receives_middle_fact_in_long_source(padding):
    question = "Which database does Morgan use?"
    content = "Session: March 2026\n" + "Unrelated weather notes.\n" * padding
    content += "Morgan uses PostgreSQL for the billing database.\n"
    content += "Unrelated garden notes.\n" * padding
    calls = []

    class Reader:
        def complete(self, **kwargs):
            calls.append(kwargs)
            present = "Morgan uses PostgreSQL for the billing database." in kwargs["user"]
            return json.dumps({"answerable": present, "partial": False, "confidence": 0.9,
                               "missing": None if present else "database", "rationale": "source check"})

    result = classify_answerability(question, [content], llm_client=Reader())
    assert result.answerable
    assert len(calls) == 1
    assert "Session: March 2026" in calls[0]["user"]
    assert len(calls[0]["user"]) < 1000


def test_short_evidence_remains_verbatim():
    content = "Morgan uses PostgreSQL. Alex uses Redis."
    prompt = _build_user_prompt("Morgan database?", [content])
    assert f"[1] {content}" in prompt


def test_clipped_unicode_evidence_has_explicit_omission_and_bounded_size():
    content = "Дата: март\n" + "Записи о погоде.\n" * 100
    content += "Мария использует PostgreSQL для базы заказов.\n"
    content += "Другие заметки.\n" * 100
    prompt = _build_user_prompt("Какую базу использует Мария?", [content])
    assert "Мария использует PostgreSQL" in prompt
    assert "excerpt truncated" in prompt
    assert "\ufffd" not in prompt
    assert len(prompt) < 1000
