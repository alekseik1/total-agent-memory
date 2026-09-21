import json

from ai_layer.iterative_retriever import iterative_retrieve


def test_decomposition_and_planning_use_the_injected_provider():
    calls = []
    queries = []

    class Client:
        def complete(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return json.dumps(
                    {
                        "canonical": "service database",
                        "decomposed": ["service owner", "owner database"],
                        "hyde": "",
                    }
                )
            return json.dumps(
                {
                    "partial_answer": "owner uses PostgreSQL",
                    "next_query": None,
                    "done": True,
                }
            )

    def search(query, **kwargs):
        queries.append(query)
        return [{"id": 1, "content": "The service owner uses PostgreSQL."}]

    result = iterative_retrieve(
        "Which database does the service owner use?",
        search_fn=search,
        llm_client=Client(),
        llm_model="selected-mini",
    )
    assert result.provenance["rewrite"]["used_decomposition"]
    assert queries == ["Which database does the service owner use?"]
    assert len(calls) == 2
    assert all(call["model"] == "selected-mini" for call in calls)
