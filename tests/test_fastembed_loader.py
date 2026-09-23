"""fastembed models with external weights load from a symlink-free copy of the cache."""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import ClassVar

import pytest

from memory_core import fastembed_loader
from memory_core.fastembed_loader import EXTERNAL_DATA_ERROR, load_model


class FakeFactory:
    """Refuses the cache path the way onnxruntime does; accepts a plain directory."""

    calls: ClassVar[list[dict]] = []

    def __init__(self, model, **kwargs):
        type(self).calls.append(kwargs)
        path = kwargs.get("specific_model_path")
        if path is None:
            raise RuntimeError(f"[ONNXRuntimeError] : 1 : FAIL : {EXTERNAL_DATA_ERROR} for initializer: w.")
        data = os.path.join(path, "model.onnx_data")
        if os.path.islink(data) or not os.path.isfile(data):
            raise RuntimeError("still a symlink")
        self.path = path

    @staticmethod
    def _get_model_description(model):
        return SimpleNamespace(sources=SimpleNamespace(hf="qdrant/big-model-onnx"))


def cache_with_symlinked_snapshot(root):
    repo = root / "models--qdrant--big-model-onnx"
    (repo / "blobs").mkdir(parents=True)
    (repo / "blobs" / "abc").write_bytes(b"weights")
    snapshot = repo / "snapshots" / "rev1"
    snapshot.mkdir(parents=True)
    (snapshot / "model.onnx_data").symlink_to(repo / "blobs" / "abc")
    (snapshot / "model.onnx").write_bytes(b"graph")
    return snapshot


def test_external_data_refusal_loads_from_a_plain_copy(tmp_path):
    cache_with_symlinked_snapshot(tmp_path)
    FakeFactory.calls = []
    model = load_model(FakeFactory, "big", cache_dir=str(tmp_path), threads=2)
    copy = tmp_path / "materialized" / "qdrant--big-model-onnx"
    assert model.path == str(copy)
    assert (copy / "model.onnx_data").read_bytes() == b"weights"
    assert not (copy / "model.onnx_data").is_symlink()
    assert FakeFactory.calls[-1]["threads"] == 2

    FakeFactory.calls = []
    load_model(FakeFactory, "big", cache_dir=str(tmp_path))
    assert len(FakeFactory.calls) == 2  # refused once, then the existing copy is reused


def test_other_errors_propagate(tmp_path):
    def broken(model, **kwargs):
        raise ValueError("model not found")

    with pytest.raises(ValueError, match="model not found"):
        load_model(broken, "missing", cache_dir=str(tmp_path))


def test_missing_snapshot_is_reported(tmp_path):
    with pytest.raises(RuntimeError, match="no downloaded snapshot"):
        fastembed_loader.materialize(FakeFactory, "big", str(tmp_path))


def test_a_model_that_loads_directly_is_not_copied(tmp_path):
    loaded = load_model(lambda model, **kwargs: ("ok", model, kwargs), "small", cache_dir=str(tmp_path))
    assert loaded == ("ok", "small", {"cache_dir": str(tmp_path)})
    assert not (tmp_path / "materialized").exists()
