"""The company setup runs under a uid that has no passwd entry (containers, Kubernetes, OpenShift)."""

import getpass
import os
import sys
from pathlib import Path

import pytest

from setup_wizard import company

POSIX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason="systemd units and numeric uids are POSIX")


@pytest.fixture(params=[KeyError("getpwuid(): uid not found: 1000"), OSError("No username set in the environment")],
                ids=["python-3.11-3.12", "python-3.13"])
def no_account(request, monkeypatch):
    def missing():
        raise request.param
    monkeypatch.setattr(getpass, "getuser", missing)


def test_login_name_is_none_without_an_account(no_account):
    assert company.login_name() is None


@POSIX_ONLY
def test_systemd_units_fall_back_to_the_numeric_uid(no_account, tmp_path: Path):
    network = company.NetworkAnswer(host="127.0.0.1", port=8765, public_url="http://127.0.0.1:8765")
    unit = company.systemd_unit(tmp_path, network)
    assert f"User={os.getuid()}" in unit.splitlines()


def test_login_name_is_the_account_when_there_is_one(monkeypatch):
    monkeypatch.setattr(getpass, "getuser", lambda: "alice")
    assert company.login_name() == "alice"
    assert company.service_user() == "alice"
