# Tier ablation: lme dev (cross-rerank off)

| config | n | R@1 | R@5 | R@10 | MRR | p50 ms | knowledge-update R@10 | multi-session R@10 | single-session-assistant R@10 | single-session-preference R@10 | single-session-user R@10 | temporal-reasoning R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 95 | 0.463 | 0.832 | 0.916 | 0.619 | 56 | 1.000 | 0.960 | 1.000 | 0.600 | 1.000 | 0.839 |

Tier fired (full): {"fts": 1.0, "semantic": 1.0, "hyde": 0.0, "multi_repr": 0.0, "fuzzy": 0.0, "graph": 0.0, "episode": 0.0, "atomic_facts": 0.0, "directives": 0.0, "multi_query": 0.0}
