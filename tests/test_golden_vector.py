"""The shipped embedding model still produces the recorded vector (pooling, truncation, scale).

fastembed changed the pooling of this model between 0.5 and 0.6 without changing its
name, which would silently split a store into two incompatible vector populations.
The recipe in tests/data/golden-vector.json records what produced the stored vectors;
the text is longer than the model's token budget, so truncation is covered too.
Skipped when the model is not in the local cache or the library is outside the range.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

GOLDEN = json.loads((Path(__file__).parent / "data" / "golden-vector.json").read_text(encoding="utf-8"))
RECIPE = GOLDEN["recipe"]
# CI sets this on the job that prewarms the model: there a missing model fails instead of skipping.
REQUIRE_ENV = "TAM_GOLDEN_REQUIRE_MODEL"
MIN_COSINE = 0.9999
# Relative: ONNX kernels differ in the last digits between CPUs (arm64 vs x86_64 is about 0.1 %), while a
# pooling or normalisation change moves the scale by far more.
NORM_REL_TOLERANCE = 5e-3
LIBRARY_RANGE = ((0, 6), (0, 9))


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(piece) for piece in text.split(".")[:2])


def _cached(model_name: str) -> bool:
    from fastembed import TextEmbedding
    from fastembed.common.utils import define_cache_dir

    description = TextEmbedding._get_model_description(model_name)
    repo = description.sources.hf
    snapshots = Path(define_cache_dir()) / f"models--{repo.replace('/', '--')}" / "snapshots"
    return snapshots.is_dir() and any(snapshots.glob("*/*.onnx"))


@pytest.fixture(scope="module")
def model():
    fastembed = pytest.importorskip("fastembed")
    low, high = LIBRARY_RANGE
    if not low <= _version(fastembed.__version__) < high:
        pytest.skip(f"fastembed {fastembed.__version__} is outside the recorded range {RECIPE['library_range']}")
    if not _cached(RECIPE["model"]):
        message = f"{RECIPE['model']} is not in the local fastembed cache"
        if os.environ.get(REQUIRE_ENV) == "1":
            pytest.fail(message)
        pytest.skip(message)
    from memory_core.fastembed_loader import load_model

    return load_model(fastembed.TextEmbedding, RECIPE["model"])


def test_recipe_text_exceeds_the_token_budget(model):
    tokenizer = model.model.tokenizer
    budget = tokenizer.truncation["max_length"]
    tokenizer.no_truncation()
    try:
        untruncated = len(tokenizer.encode(GOLDEN["text"]).ids)
    finally:
        tokenizer.enable_truncation(max_length=budget)
    assert budget == RECIPE["max_tokens"]
    assert untruncated == RECIPE["text_tokens_untruncated"] > budget


def test_model_output_matches_the_recorded_vector(model):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        produced = np.asarray(next(iter(model.embed([RECIPE["passage_prefix"] + GOLDEN["text"]]))), dtype=np.float64)
    recorded = np.asarray(GOLDEN["vector"], dtype=np.float64)

    cosine = float(produced @ recorded / (np.linalg.norm(produced) * np.linalg.norm(recorded)))
    assert cosine >= MIN_COSINE, f"direction changed (cosine {cosine:.6f}): pooling or tokenizer differs"
    assert abs(float(np.linalg.norm(produced)) - GOLDEN["norm"]) <= NORM_REL_TOLERANCE * GOLDEN["norm"], (
        f"scale changed ({np.linalg.norm(produced):.4f} vs {GOLDEN['norm']:.4f}): normalisation differs")


def test_recipe_documents_how_the_vector_was_made():
    assert {"model", "library", "library_range", "pooling", "query_prefix", "passage_prefix", "max_tokens"} <= set(RECIPE)
    assert len(GOLDEN["vector"]) == 384
