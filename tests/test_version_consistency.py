"""Every place that ships a release number agrees with pyproject.toml.

A release bump touches a dozen files: package metadata, the MCP registry
manifest, the citation file, the changelog, image tags in the Makefile,
compose files, Dockerfiles and CI, and the README install line. Missing one
publishes a registry entry, an image or a citation pointing at the previous
release. Each check below names the file and the field so a failure says
exactly what to edit.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Callable
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SEMVER = r"\d+\.\d+\.\d+(?:[-.]?(?:a|b|rc|dev|post)\d*)?"


def canonical_version() -> str:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]


# An unpacked source distribution (the smoke workflow's source-archive job)
# carries no .git and leaves out CITATION.cff, CHANGELOG.md and .github/.
# There a missing file is expected; in a checkout it is a failure.
IN_CHECKOUT = (ROOT / ".git").exists()


def read(rel: str) -> str:
    path = ROOT / rel
    if not path.is_file():
        if IN_CHECKOUT:
            raise AssertionError(f"{rel} is missing; update this test if it was moved or removed")
        pytest.skip(f"{rel} is not shipped in the source distribution")
    return path.read_text(encoding="utf-8")


def regex_all(rel: str, pattern: str) -> list[str]:
    found = re.findall(pattern, read(rel), flags=re.MULTILINE)
    assert found, f"{rel}: pattern {pattern!r} matched nothing; update this test if the file layout changed"
    return found


def version_py() -> list[str]:
    return regex_all("src/version.py", rf'^VERSION\s*=\s*"({SEMVER})"')


def server_json() -> list[str]:
    data = json.loads(read("server.json"))
    versions = [data["version"]]
    versions.extend(pkg["version"] for pkg in data.get("packages", []) if "version" in pkg)
    return versions


def plugin_json() -> list[str]:
    return [json.loads(read(".claude-plugin/plugin.json"))["version"]]


def citation_cff() -> list[str]:
    return regex_all("CITATION.cff", rf"^version:\s*\"?({SEMVER})\"?\s*$")


def changelog_head() -> list[str]:
    for heading in re.findall(r"^## \[([^\]]+)\]", read("CHANGELOG.md"), flags=re.MULTILINE):
        if heading.strip().lower() != "unreleased":
            return [heading.strip()]
    raise AssertionError("CHANGELOG.md has no released '## [x.y.z]' heading")


def makefile_tags() -> list[str]:
    return regex_all("Makefile", rf"^\w+_IMAGE\s*\?=\s*[\w./-]+:({SEMVER})\s*$")


def compose_team_tags() -> list[str]:
    return regex_all("docker-compose.team.yml", rf"image:\s*\$\{{TAM_TEAM_IMAGE:-[\w./-]+:({SEMVER})\}}")


def dockerfile_arg_tags() -> list[str]:
    found: list[str] = []
    read("docker/Dockerfile.dev")
    for path in sorted((ROOT / "docker").glob("Dockerfile.*")):
        found.extend(re.findall(rf"^ARG \w+=tam-[\w-]+:({SEMVER})\s*$", path.read_text(encoding="utf-8"),
                                flags=re.MULTILINE))
    assert found, "docker/Dockerfile.*: no 'ARG ...=tam-...:<version>' lines found"
    return found


def smoke_workflow_tags() -> list[str]:
    return regex_all(".github/workflows/smoke.yml", rf"tam-compose-candidate:({SEMVER})")


def readme_versions() -> list[str]:
    return (
        regex_all("README.md", rf"badge/version-({SEMVER})-")
        + regex_all("README.md", rf"ghcr\.io/vbcherepanov/total-agent-memory:({SEMVER})")
        + regex_all("README.md", rf"total-agent-memory \(version ({SEMVER})\)")
    )


SOURCES: dict[str, Callable[[], list[str]]] = {
    "src/version.py VERSION": version_py,
    "server.json version + packages[].version": server_json,
    ".claude-plugin/plugin.json version": plugin_json,
    "CITATION.cff version": citation_cff,
    "CHANGELOG.md newest released heading": changelog_head,
    "Makefile *_IMAGE tags": makefile_tags,
    "docker-compose.team.yml TAM_TEAM_IMAGE default tag": compose_team_tags,
    "docker/Dockerfile.* ARG image tags": dockerfile_arg_tags,
    ".github/workflows/smoke.yml compose candidate tag": smoke_workflow_tags,
    "README.md badge / docker run tag / citation line": readme_versions,
}


@pytest.mark.parametrize("label", sorted(SOURCES))
def test_version_matches_pyproject(label):
    expected = canonical_version()
    found = SOURCES[label]()
    wrong = sorted({v for v in found if v != expected})
    assert not wrong, f"{label}: found {wrong}, pyproject.toml [project].version is {expected!r}"


def test_release_dates_agree():
    version_date = regex_all("src/version.py", r'^RELEASE_DATE\s*=\s*"(\d{4}-\d{2}-\d{2})"')[0]
    citation_date = regex_all("CITATION.cff", r"^date-released:\s*\"?(\d{4}-\d{2}-\d{2})\"?\s*$")[0]
    version = canonical_version()
    changelog = re.search(
        rf"^## \[{re.escape(version)}\]\s*-\s*(\d{{4}}-\d{{2}}-\d{{2}})", read("CHANGELOG.md"), flags=re.MULTILINE
    )
    assert changelog, f"CHANGELOG.md: no '## [{version}] - YYYY-MM-DD' heading"
    dates = {
        "src/version.py RELEASE_DATE": version_date,
        "CITATION.cff date-released": citation_date,
        f"CHANGELOG.md [{version}] date": changelog.group(1),
    }
    assert len(set(dates.values())) == 1, f"release dates disagree: {dates}"
