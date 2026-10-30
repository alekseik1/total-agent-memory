"""The suite stops the Chroma systems each test starts (tests/chroma_support.py)."""
import gc
import os
import sys

import pytest

from tests.chroma_support import running_systems, stop_systems_started_since

chromadb = pytest.importorskip("chromadb")

SYSTEMS_PER_CHECK = 5


def _open_descriptors() -> int:
    return len(os.listdir("/dev/fd"))


def test_a_started_system_is_stopped_and_forgotten(tmp_path):
    before = running_systems()
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    client.get_or_create_collection("knowledge", metadata={"hnsw:space": "cosine"})
    assert str(tmp_path / "chroma") in running_systems()

    assert stop_systems_started_since(before) == [str(tmp_path / "chroma")]
    assert running_systems() == before


def test_systems_running_before_are_left_alone(tmp_path):
    kept = chromadb.PersistentClient(path=str(tmp_path / "kept"))
    before = running_systems()
    chromadb.PersistentClient(path=str(tmp_path / "new"))

    assert stop_systems_started_since(before) == [str(tmp_path / "new")]
    assert str(tmp_path / "kept") in running_systems()
    assert kept.get_or_create_collection("knowledge").count() == 0
    stop_systems_started_since(before - {str(tmp_path / "kept")})


def test_the_same_path_opens_again_after_a_stop(tmp_path):
    before = running_systems()
    path = str(tmp_path / "chroma")
    chromadb.PersistentClient(path=path).get_or_create_collection("knowledge").add(ids=["1"], embeddings=[[0.1, 0.2]])
    stop_systems_started_since(before)

    reopened = chromadb.PersistentClient(path=path)
    assert reopened.get_collection("knowledge").count() == 1
    stop_systems_started_since(before)


def _start_and_stop_systems(root, before) -> int:
    """Open descriptors after starting SYSTEMS_PER_CHECK systems under ``root`` and stopping them."""
    for index in range(SYSTEMS_PER_CHECK):
        chromadb.PersistentClient(path=str(root / str(index))).get_or_create_collection("knowledge")
    assert len(stop_systems_started_since(before)) == SYSTEMS_PER_CHECK
    # The stopped bindings sit in reference cycles; collect them as the interpreter eventually would.
    gc.collect()
    return _open_descriptors()


@pytest.mark.skipif(sys.platform == "win32", reason="/dev/fd lists open descriptors on POSIX only")
def test_stopped_systems_leave_no_descriptors_behind(tmp_path):
    before = running_systems()
    # chromadb keeps a few process-wide handles open after its first system; measure from there.
    settled = _start_and_stop_systems(tmp_path / "first", before)
    assert _start_and_stop_systems(tmp_path / "second", before) <= settled
    assert _start_and_stop_systems(tmp_path / "third", before) <= settled


_STARTED_BY_PREVIOUS_TEST: list[str] = []


def test_conftest_stops_the_system_of_a_test_part_1_start(tmp_path):
    path = str(tmp_path / "left-running")
    chromadb.PersistentClient(path=path).get_or_create_collection("knowledge")
    _STARTED_BY_PREVIOUS_TEST.append(path)


def test_conftest_stops_the_system_of_a_test_part_2_check():
    assert _STARTED_BY_PREVIOUS_TEST, "runs right after part 1 in file order"
    assert _STARTED_BY_PREVIOUS_TEST[0] not in running_systems()


def test_nothing_to_do_before_chromadb_is_imported(monkeypatch):
    monkeypatch.delitem(sys.modules, "chromadb.api.shared_system_client")
    assert running_systems() == frozenset()
    assert stop_systems_started_since(frozenset()) == []
