from __future__ import annotations

import re
from dataclasses import dataclass

from memory_core.query_terms import lexical_terms, russian_stem

MAX_BINDING_CHARS = 500
_TURN = re.compile(r'(?m)^(?:\[[^\n]+\]\s*)?([\w][\w .-]{0,40}):[ \t]+')
_ROLE = re.compile(r'(?m)^(?:\[[^\n]+\]\s*)?([\w][\w .-]{0,40}):\s+(?:I\b|my\b|я\b|мой\b|моя\b)', re.IGNORECASE)
GENERIC_ROLES = frozenset(('user', 'assistant', 'system', 'пользователь', 'ассистент'))


class InvalidGrounding(ValueError):
    pass


_WORD = re.compile(r'\w+')
_CYRILLIC = re.compile(r'[а-яё]', re.IGNORECASE)


def contains_name(text: str, name: str) -> bool:
    name = name.strip()
    if not name:
        return False
    if re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', text, re.IGNORECASE):
        return True
    if not _CYRILLIC.search(name):
        return False
    # Russian names inflect ("Маша" is cited as "Маше"): compare Snowball stems word by word.
    wanted = [russian_stem(word) for word in _WORD.findall(name.casefold())]
    words = [russian_stem(word) for word in _WORD.findall(text.casefold())]
    return any(words[index:index + len(wanted)] == wanted for index in range(len(words) - len(wanted) + 1))


def citation_context(content: str, quote: str) -> str:
    start = content.find(quote)
    if start < 0:
        raise InvalidGrounding('Citation does not occur in the source')
    turns = list(_TURN.finditer(content))
    preceding = [turn.start() for turn in turns if turn.start() <= start]
    following = [turn.start() for turn in turns if turn.start() > start + len(quote)]
    return content[preceding[-1] if preceding else 0:following[0] if following else len(content)]


def subject_supported(subject: str, quotes: tuple[str, ...], target_quotes: tuple[str, ...] = ()) -> bool:
    subject = re.sub(r'^(?:the|a|an)\s+', '', subject.strip(), flags=re.IGNORECASE)
    names = re.findall(r'\b[A-ZА-Я][a-zа-я]+(?:[ -][A-ZА-Я][a-zа-я]+)*\b', subject)
    keys = names or lexical_terms(subject)
    if not keys or (names and not all(any(contains_name(quote, name) for quote in quotes) for name in names)):
        return False
    if not names and not any(contains_name(quote, key) for quote in quotes for key in keys):
        return False
    for quote in target_quotes:
        for role in _ROLE.findall(quote):
            if role.casefold() not in GENERIC_ROLES and not contains_name(subject, role):
                return False
    return True


@dataclass(frozen=True)
class MissingRelation:
    subject: str
    relation: str
    time: str = ''

    def query(self, original: str, bridge_quotes: tuple[str, ...] = ()) -> str:
        fields = (self.subject, self.relation, self.time)
        if any(not isinstance(value, str) or len(value) > MAX_BINDING_CHARS for value in fields):
            raise InvalidGrounding('Invalid missing-relation fields')
        if not self.subject.strip() or not self.relation.strip():
            raise InvalidGrounding('A missing relation requires subject and relation')
        personal = self.subject.casefold() in ('user', 'the user', 'пользователь') and re.search(r'\b(i|my|me|я|мой|моя|мне)\b', original, re.IGNORECASE)
        if not personal and not contains_name(original, self.subject) and not any(contains_name(quote, self.subject) for quote in bridge_quotes):
            raise InvalidGrounding('Follow-up subject must occur in the question or cited bridge')
        return ' '.join(value.strip() for value in fields if value.strip())
