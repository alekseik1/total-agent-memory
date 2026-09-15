from __future__ import annotations


def object_schema(properties: dict[str, object]) -> dict[str, object]:
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


TEXT = {'type': 'string'}
CITATION = object_schema({'source_id': {'type': 'integer'}, 'quote': TEXT})
CLAIM = object_schema({'subject': TEXT, 'event': TEXT, 'time': TEXT,
                       'modality': {'type': 'string', 'enum': ['asserted', 'planned', 'negated', 'inferred']},
                       'citations': {'type': 'array', 'items': CITATION}})
DRAFT_SCHEMA = object_schema({
    'status': {'type': 'string', 'enum': ['supported', 'inferred', 'partial', 'insufficient']},
    'answer': TEXT, 'claims': {'type': 'array', 'items': CLAIM},
    'missing': {'anyOf': [object_schema({'subject': TEXT, 'relation': TEXT, 'time': TEXT}), {'type': 'null'}]},
})
VERIFICATION_SCHEMA = object_schema({'supported': {'type': 'boolean'}, 'reason': TEXT})
