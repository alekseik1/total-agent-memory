"""scripts/check-no-internal-data.sh: detection per class, reasoned allow-list, clean tree."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "check-no-internal-data.sh"
BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(not BASH or not shutil.which("git"), reason="bash and git required")

# Built from parts so this file itself carries no credential or address shape.
PRIVATE_IP = ".".join(["192", "168", "7", "21"])
KEY = "sk-ant-" + "api03-" + "Zx9Yw8Vu7Ts6Rq5Po4Nm3Lk2"
AWS = "AKIA" + "ABCDEFGHIJKLMNOP"


def _repo(root: Path, files: dict[str, str], allowlist: str = "", hosts: str = "") -> Path:
    """A fresh git repository whose files stay untracked: the scan must cover new files too."""
    repo = root / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy2(SCRIPT, repo / "scripts" / "check-no-internal-data.sh")
    (repo / "scripts" / "internal-data-allowlist.txt").write_text(allowlist, encoding="utf-8")
    (repo / "scripts" / "internal-data-hosts.local.txt").write_text(hosts, encoding="utf-8")
    for name, text in files.items():
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    return repo


def _run(repo: Path, *args: str, hosts_env: str = "") -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("TAM_INTERNAL")}
    if hosts_env:
        env["TAM_INTERNAL_HOSTS"] = hosts_env
    return subprocess.run([BASH, str(repo / "scripts" / "check-no-internal-data.sh"), *args], cwd=repo,
                          env=env, capture_output=True, text=True, timeout=60, check=False)


@pytest.mark.parametrize("name,text,expected", [
    ("rfc1918", f"host = {PRIVATE_IP}\n", PRIVATE_IP),
    ("rfc1918", "db: 10.1.2.3:5432\n", "10.1.2.3"),
    ("rfc1918", "peer 172.20.0.9\n", "172.20.0.9"),
    ("mdns", "ssh build-box.local\n", "build-box.local"),
    ("apikey", f"ANTHROPIC_API_KEY={KEY}\n", KEY),
    ("apikey", f"aws {AWS}\n", AWS),
    ("apikey", "-----BEGIN RSA PRIVATE KEY-----\n", "-----BEGIN RSA PRIVATE KEY-----"),
])
def test_each_class_is_detected(tmp_path, name, text, expected):
    repo = _repo(tmp_path, {"docs/notes.md": text})
    proc = _run(repo)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert f"docs/notes.md:1: {name}: {expected}" in proc.stdout


def test_operator_hosts_come_from_the_list_and_the_environment(tmp_path):
    repo = _repo(tmp_path, {"README.md": "see /Users/ops/Dev on Build-Station-7 and other-box\n"},
                 hosts="# operators\nbuild-station-7\n")
    proc = _run(repo)
    assert proc.returncode == 1 and "operator: Build-Station-7" in proc.stdout
    assert "other-box" not in proc.stdout
    proc = _run(repo, hosts_env="other-box, spare")
    assert "operator: other-box" in proc.stdout


def test_public_data_and_code_attributes_pass(tmp_path):
    repo = _repo(tmp_path, {
        "a.py": "import threading\nlocal = threading.local()\npath = '~/.local/bin'\nip = '8.8.8.8'\nv = '110.1.2.3.4'\n",
        "b.md": "Docs live at https://example.com and 192.0.2.10 is a documentation address.\nsk-learn is not a key\n",
    })
    proc = _run(repo)
    assert proc.returncode == 1  # threading.local looks like a .local host until allow-listed
    assert "a.py:2: mdns: threading.local" in proc.stdout
    assert "192.0.2.10" not in proc.stdout and "8.8.8.8" not in proc.stdout and "110.1.2.3" not in proc.stdout
    assert "b.md" not in proc.stdout and "~/.local" not in proc.stdout


def test_allow_list_accepts_a_finding_with_a_reason(tmp_path):
    repo = _repo(tmp_path, {"tests/t.py": f"HOST = '{PRIVATE_IP}'\n", "src/x.py": f"HOST = '{PRIVATE_IP}'\n"},
                 allowlist="tests/* :: 192\\.168\\.7\\.[0-9]+ :: fixture address for a bind test\n")
    proc = _run(repo)
    assert proc.returncode == 1
    assert "src/x.py:1: rfc1918" in proc.stdout and "tests/t.py" not in proc.stdout


@pytest.mark.parametrize("entry", [
    "tests/* :: 10\\.0\\.0\\.1\n",
    "tests/* :: 10\\.0\\.0\\.1 :: short\n",
    "tests/* :: 10\\.0\\.0\\.1 :: \n",
    "not an entry at all\n",
])
def test_allow_list_entry_without_a_reason_is_a_configuration_error(tmp_path, entry):
    repo = _repo(tmp_path, {"tests/t.py": "ip = '10.0.0.1'\n"}, allowlist=entry)
    proc = _run(repo)
    assert proc.returncode == 2
    assert "reason" in proc.stderr or "expected" in proc.stderr


def test_stale_entries_warn_and_fail_only_on_request(tmp_path):
    repo = _repo(tmp_path, {"a.txt": "nothing here\n"}, allowlist="a.txt :: 10\\.0\\.0\\.1 :: fixture that was removed\n")
    assert _run(repo).returncode == 0
    strict = _run(repo, "--fail-on-stale")
    assert strict.returncode == 1 and "stale allow-list entry" in strict.stderr


# The source distribution is not a git checkout: the tree scan and .gitignore belong to the repository.
needs_checkout = pytest.mark.skipif(not (ROOT / ".git").exists(), reason="needs the git checkout, not the source distribution")


@needs_checkout
def test_the_repository_tree_is_clean():
    proc = subprocess.run([BASH, str(SCRIPT)], cwd=ROOT, capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_a_failing_git_grep_is_not_reported_as_clean(tmp_path):
    repo = _repo(tmp_path, {"README.md": "plain text\n"})
    real_git = shutil.which("git")
    shim = tmp_path / "bin" / "git"
    shim.parent.mkdir()
    shim.write_text(f'#!/bin/sh\n[ "$1" = grep ] && exit 2\nexec "{real_git}" "$@"\n', encoding="utf-8")
    shim.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("TAM_INTERNAL")}
    env["PATH"] = f"{shim.parent}{os.pathsep}{env['PATH']}"
    proc = subprocess.run([BASH, str(repo / "scripts" / "check-no-internal-data.sh")], cwd=repo, env=env,
                          capture_output=True, text=True, timeout=60, check=False)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "git grep failed" in proc.stderr


@needs_checkout
def test_the_repository_ships_no_host_list_and_ignores_the_local_one():
    assert not (ROOT / "scripts" / "internal-data-hosts.txt").exists()
    ignored = subprocess.run(["git", "check-ignore", "-q", "scripts/internal-data-hosts.local.txt"], cwd=ROOT,
                             check=False)
    assert ignored.returncode == 0
