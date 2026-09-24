import json

import pytest

from memory_core.classifier import classify
from memory_core.evidence_pack import pack_evidence, pack_evidence_records


@pytest.mark.parametrize("roles", [("user", "assistant"), ("Human", "AI")])
def test_transcripts_are_text_even_with_embedded_code(roles):
    first, second = roles
    text = f"{first}: We meet tomorrow.\n{second}: Noted.\n{first}: Remember the room."
    assert classify(text).type == "text"
    assert classify(text + "\nTraceback (most recent call last):").type == "text"


@pytest.mark.parametrize("text,path", [
    ("user: admin\nassistant: helper\nsystem: linux", None),
    ("---\nuser: admin\nassistant: helper\nuser: another", None),
    ("user: hello\nassistant: hello\nuser: goodbye", "config.yaml"),
    ("host: local\nport: 8000\nmode: fast", None),
])
def test_explicit_yaml_and_distinct_configuration_keys_remain_yaml(text, path):
    assert classify(text, file_path=path).type == "yaml"


def test_middle_evidence_and_all_sources_survive_budget():
    records = [
        {"id": 1, "content": "Date: 2026-03-01\n" + "user: weather was pleasant.\n" * 100
         + "user: The launch owner is Morgan.\nassistant: Morgan, not Alex, owns the launch.\n"
         + "user: unrelated garden notes.\n" * 100},
        {"id": 2, "content": "The launch date is April 7."},
    ]
    result = pack_evidence(records, query="Who owns the launch?", max_chars=1400)
    assert len(result) <= 1400
    assert "Morgan, not Alex" in result
    assert "Date: 2026-03-01" in result
    assert "April 7" in result
    assert '"id":1' in result and '"id":2' in result
    assert "excerpt truncated" in result
    assert "excerpt truncated" not in records[0]["content"]


def test_utf8_budget_preserves_sources_and_unicode():
    records = [{"id": i, "content": "Дата: март\n" + "Повседневные записи.\n" * 100
                + "Владелец запуска — Мария.\n" + "Записи о погоде.\n" * 100}
               for i in (1, 2)]
    result = pack_evidence(records, query="Владелец запуска", max_bytes=2000)
    assert len(result.encode("utf-8")) <= 2000
    assert result.count("Мария") == 2
    assert "\ufffd" not in result


def test_short_evidence_keeps_metadata_and_content():
    records = [{"id": 7, "source_ref": "knowledge:7", "created_at": "2026-03-01",
                "content": "Owner is Morgan.", "evidence_ids": [4, 7]}]
    packed = pack_evidence_records(records, query="owner")
    assert packed == records
    assert json.loads(pack_evidence(records).splitlines()[0])["evidence_ids"] == [4, 7]


@pytest.mark.parametrize("kwargs", [{"max_bytes": True}, {"max_bytes": 1}, {"max_chars": 0}])
def test_invalid_budgets_rejected(kwargs):
    with pytest.raises(ValueError):
        pack_evidence([], **kwargs)


def test_metadata_overflow_is_explicit():
    with pytest.raises(ValueError, match="metadata"):
        pack_evidence([{"id": i, "content": "fact"} for i in range(30)], max_chars=512)


def test_neighbor_anchor_link_survives_rendering():
    packed = pack_evidence([{"id": 2, "anchor_id": 1, "content": "Yes, that one."}])
    assert json.loads(packed.splitlines()[0])["anchor_id"] == 1


def test_rank_weighting_keeps_the_top_hit_whole():
    """Forty long rounds share the budget; the first search hit must not be cut to an excerpt."""
    from memory_core.evidence_context import rank_weighted

    answer = "assistant: " + " ".join(f"{n}. parameter number {n}" for n in range(1, 101))
    hits = [{"id": 0, "content": answer}] + [
        {"id": n, "content": f"user: filler round {n}. " + "unrelated chatter " * 120} for n in range(1, 40)]
    hits.append({"id": 99, "content": "neighbour " * 200, "via": ["session_neighbor"]})

    uniform = pack_evidence_records(hits, query="What was the 27th parameter?", max_chars=48000)
    assert "excerpt truncated" in uniform[0]["content"]

    weighted_hits = rank_weighted(hits)
    assert [hit["evidence_weight"] for hit in weighted_hits[:3]] == [4.0, 2.5, 2.0]
    assert weighted_hits[-1]["evidence_weight"] == 0.5
    weighted = pack_evidence_records(weighted_hits, query="What was the 27th parameter?", max_chars=48000)
    assert weighted[0]["content"] == answer
    assert len({hit["id"] for hit in weighted}) == len(hits)
