from __future__ import annotations

from typing import TypedDict

import config
from ai_layer.grounded_answer import GroundedAnswer, GroundedAnswerService
from ai_layer.grounded_reader import GroundedReader
from ai_layer.negative_evidence import LLMContradictionScorer, ProviderInversionClient
from evidence_endpoint import EvidenceRecall, EvidenceStore
from llm_provider import make_provider
from memory_core.retrieval import MemoryHit, SearchScope, flatten_results


class AnswerOptions(TypedDict, total=False):
    query: str
    project: str
    branch: str
    type: str
    limit: int
    max_bytes: int
    followup: bool


def answer_response(store: EvidenceStore, recall: EvidenceRecall, options: AnswerOptions) -> GroundedAnswer:
    if not options.get('project', '').strip():
        raise ValueError('Grounded answers require an explicit project')
    scope = SearchScope(options['project'], options.get('type', 'all'), options.get('branch'))
    def search(query: str, limit: int) -> list[MemoryHit]:
        return flatten_results(recall.search(query, project=scope.project, ktype=scope.kind,
                                            branch=scope.branch, limit=limit, detail='full', record_usage=False))
    provider, model = make_provider(config.get_phase_provider('reason')), config.get_phase_model('reason')
    reader = GroundedReader(provider, model)
    negative = config.is_negative_retrieval_enabled()
    service = GroundedAnswerService(
        store.db, search, reader, config.get_recall_excluded_tags(),
        contradiction_scorer=LLMContradictionScorer(provider, model) if negative else None,
        inversion_client=ProviderInversionClient(provider, model) if negative else None,
        contradiction_policy=config.get_contradiction_policy())
    result = service.answer(
        options['query'], scope, limit=options.get('limit', 10), max_bytes=options.get('max_bytes', 24000),
        followup=options.get('followup', True))
    return result
