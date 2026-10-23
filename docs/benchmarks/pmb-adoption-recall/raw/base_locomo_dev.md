# Tier ablation: locomo dev (cross-rerank off)

| config | n | R@1 | R@5 | R@10 | MRR | p50 ms | multi-hop R@10 | open-domain R@10 | single-hop R@10 | temporal R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 383 | 0.303 | 0.561 | 0.689 | 0.423 | 24 | 0.595 | 0.316 | 0.675 | 0.878 |

Tier fired (full): {"fts": 1.0, "semantic": 1.0, "hyde": 0.0, "multi_repr": 0.0, "fuzzy": 0.0, "graph": 0.0, "episode": 0.0, "atomic_facts": 0.0, "directives": 0.0, "multi_query": 0.0}
