# locomo test (cross-rerank off, K=10, entity graph off)

| config | n | R@1 | R@5 | R@10 | MRR | p50 ms | pair-and R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| default | 318 | 0.396 | 0.755 | 0.805 | 0.540 | 41 | 0.805 |

- default: coverage@10 0.593, compound {"n": 0, "coverage@5": 0.0, "coverage@10": 0.0}, fired {"compound": 0, "historical": 0, "fts_damped": 0}
