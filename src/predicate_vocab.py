"""Canonical predicates for the temporal knowledge graph.

``vocabularies/predicate_aliases.txt`` (plus optional extra files named by
``MEMORY_PREDICATE_VOCAB_EXTRA``) groups predicates that state the same
relation: ``city``, ``current_city``, ``location`` and ``живет_в`` are all
``lives_in``. ``temporal_kg`` matches assertions by the whole alias group, so a
new object under any form supersedes the current one under another form, while
the stored predicate keeps the caller's wording.

Unknown predicates form a group of their own (exact match, as before).
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

DEFAULT_VOCAB_PATH = Path(__file__).resolve().parent.parent / "vocabularies" / "predicate_aliases.txt"
VOCAB_PATH_ENV = "MEMORY_PREDICATE_VOCAB_PATH"
VOCAB_EXTRA_ENV = "MEMORY_PREDICATE_VOCAB_EXTRA"
_SEPARATORS = re.compile(r"[\s\-]+")


class PredicateVocabularyError(ValueError):
    """The vocabulary file is malformed or assigns one form to two canonicals."""


def normalize(predicate: str) -> str:
    """Lower-case, trimmed, spaces/hyphens as underscores."""
    return _SEPARATORS.sub("_", predicate.strip().lower())


@dataclass(frozen=True)
class PredicateVocabulary:
    canonical_by_form: dict[str, str]
    forms_by_canonical: dict[str, tuple[str, ...]]

    def canonical(self, predicate: str) -> str:
        """Canonical name of ``predicate``; its normalized form when unknown."""
        form = normalize(predicate)
        return self.canonical_by_form.get(form, form)

    def is_known(self, predicate: str) -> bool:
        return normalize(predicate) in self.canonical_by_form

    def match_forms(self, predicate: str) -> tuple[str, ...]:
        """Stored predicate values that mean the same relation as ``predicate``.

        Always contains ``predicate`` itself as given (legacy rows stored any
        casing); for a known predicate, every form of its group as well.
        """
        forms = [predicate]
        canonical = self.canonical_by_form.get(normalize(predicate))
        if canonical is not None:
            forms.extend(form for form in self.forms_by_canonical[canonical] if form != predicate)
        return tuple(forms)


def parse(text: str, *, source: str = "<vocabulary>",
          into: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    """Parse vocabulary lines into canonical -> forms (canonical first)."""
    groups: dict[str, list[str]] = into if into is not None else {}
    owner = {form: canonical for canonical, forms in groups.items() for form in forms}
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        forms = [normalize(part) for part in line.split(",") if part.strip()]
        if not forms:
            continue
        canonical = forms[0]
        group = groups.setdefault(canonical, [canonical])
        owner.setdefault(canonical, canonical)
        if owner[canonical] != canonical:
            raise PredicateVocabularyError(
                f"{source}:{number}: {canonical!r} is already an alias of {owner[canonical]!r}")
        for form in forms[1:]:
            current = owner.get(form)
            if current is not None and current != canonical:
                raise PredicateVocabularyError(
                    f"{source}:{number}: {form!r} already belongs to {current!r}")
            if current is None:
                owner[form] = canonical
                group.append(form)
    return groups


def _build(groups: dict[str, list[str]]) -> PredicateVocabulary:
    canonical_by_form = {form: canonical for canonical, forms in groups.items() for form in forms}
    return PredicateVocabulary(canonical_by_form=canonical_by_form,
                               forms_by_canonical={c: tuple(f) for c, f in groups.items()})


def _paths() -> list[Path]:
    paths = [Path(os.environ.get(VOCAB_PATH_ENV) or DEFAULT_VOCAB_PATH)]
    extra = os.environ.get(VOCAB_EXTRA_ENV, "")
    paths.extend(Path(part) for part in extra.split(os.pathsep) if part.strip())
    return paths


def load(paths: list[Path] | None = None) -> PredicateVocabulary:
    """Read and merge vocabulary files. A missing file is an error: matching
    would silently fall back to exact predicates."""
    groups: dict[str, list[str]] = {}
    for path in paths if paths is not None else _paths():
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as error:
            raise PredicateVocabularyError(f"Predicate vocabulary unreadable: {path}: {error}") from error
        parse(text, source=str(path), into=groups)
    return _build(groups)


_cache: tuple[tuple[str, ...], PredicateVocabulary] | None = None
_cache_lock = threading.Lock()


def get_vocabulary() -> PredicateVocabulary:
    """The vocabulary for the current env configuration (cached per path set)."""
    global _cache
    key = tuple(str(path) for path in _paths())
    with _cache_lock:
        if _cache is None or _cache[0] != key:
            _cache = (key, load([Path(item) for item in key]))
        return _cache[1]


def reset_cache() -> None:
    global _cache
    with _cache_lock:
        _cache = None
