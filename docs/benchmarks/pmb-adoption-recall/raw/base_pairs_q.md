# locomo test (cross-rerank off, K=10, entity graph off)

| config | n | R@1 | R@5 | R@10 | MRR | p50 ms | pair-question R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| default | 318 | 0.403 | 0.748 | 0.802 | 0.541 | 41 | 0.802 |

- default: coverage@10 0.583, compound {"n": 0, "coverage@5": 0.0, "coverage@10": 0.0}, fired {"compound": 0, "historical": 0, "fts_damped": 0}
