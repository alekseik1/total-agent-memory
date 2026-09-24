"""Load fastembed models whose weights live in a separate `model.onnx_data` file.

fastembed downloads a model into the HuggingFace cache, where the snapshot
directory holds symlinks into `blobs/`. onnxruntime 1.2x checks that an
external data file lies inside the model's directory, resolves the symlink,
and refuses: "External data path validation failed ... escapes model
directory". Every model above 2 GB is stored that way (multilingual-e5-large,
jina-embeddings-v3), so it could not be loaded at all. The loader copies such
a snapshot once into a plain directory next to the cache and loads it from there.
"""
from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from pathlib import Path

LOGGER = logging.getLogger(__name__)
EXTERNAL_DATA_ERROR = "External data path validation failed"
MATERIALIZED_DIR = "materialized"


def load_model(factory: Callable[..., object], model: str, **kwargs) -> object:
    """`factory(model, **kwargs)`, retried from a symlink-free copy when onnxruntime refuses the cache."""
    try:
        return factory(model, **kwargs)
    except Exception as error:
        if EXTERNAL_DATA_ERROR not in str(error):
            raise
    path = materialize(factory, model, kwargs.get("cache_dir"))
    LOGGER.info("Loading fastembed model from a copy without cache symlinks",
                extra={"model": model, "path": str(path)})
    return factory(model, specific_model_path=str(path), **kwargs)


def materialize(factory, model: str, cache_dir: str | None) -> Path:
    """Copy the model's cached snapshot, following symlinks, into `<cache>/materialized/<repo>`."""
    from fastembed.common.utils import define_cache_dir

    description = factory._get_model_description(model)
    repo = description.sources.hf
    if not repo:
        raise RuntimeError(f"{model} has no HuggingFace source to copy")
    cache = Path(define_cache_dir(cache_dir))
    snapshots = sorted((cache / f"models--{repo.replace('/', '--')}" / "snapshots").glob("*"),
                       key=lambda path: path.stat().st_mtime)
    if not snapshots:
        raise RuntimeError(f"{model}: no downloaded snapshot under {cache}")
    target = cache / MATERIALIZED_DIR / repo.replace("/", "--")
    marker = target / ".complete"
    if not marker.exists():
        partial = target.with_name(target.name + ".partial")
        shutil.rmtree(partial, ignore_errors=True)
        shutil.copytree(snapshots[-1], partial, symlinks=False)
        shutil.rmtree(target, ignore_errors=True)
        partial.rename(target)
        marker.touch()
    return target
