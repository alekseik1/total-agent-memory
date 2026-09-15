import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "memory_quality_eval", ROOT / "benchmarks/memory_quality.py"
)
evaluation = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = evaluation
spec.loader.exec_module(evaluation)


def test_contract_dataset_has_distinct_topics_and_no_cross_project_gold():
    data = json.loads((ROOT / "benchmarks/data/memory-contracts-v1.json").read_text())
    cases = evaluation.load_cases(data)
    assert len(cases) == 12
    assert {case.category for case in cases} >= {
        "multi_hop",
        "temporal",
        "isolation",
        "unanswerable",
    }
    data["cases"][0]["evidence"] = [[3]]
    with pytest.raises(ValueError, match="crosses project"):
        evaluation.load_cases(data)


def test_judge_audit_separates_api_errors_and_conflicting_identical_answers(tmp_path):
    rows = [
        {
            "id": "1",
            "question": "When?",
            "gold": "Yesterday",
            "prediction": "Yesterday.",
            "correct": True,
        },
        {
            "id": "2",
            "question": "When?",
            "gold": "Yesterday",
            "prediction": " yesterday ",
            "correct": False,
        },
        {
            "id": "3",
            "question": "When?",
            "gold": "Yesterday",
            "error": "HTTP failure",
            "correct": False,
        },
    ]
    path = tmp_path / "answers.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    assert evaluation.judge_audit(path) == {
        "rows": 3,
        "api_errors": 1,
        "identical_answer_conflicts": 1,
        "correct": 1,
    }
