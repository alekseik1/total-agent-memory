# locomo test (cross-rerank off, K=50, entity graph off)

| config | n | R@1 | R@5 | R@10 | MRR | p50 ms | multi-hop R@10 | open-domain R@10 | single-hop R@10 | temporal R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 1153 | 0.314 | 0.586 | 0.672 | 0.439 | 26 | 0.572 | 0.397 | 0.719 | 0.719 |

- default: coverage@10 0.599, compound {"n": 3, "coverage@5": 0.6667, "coverage@10": 0.6667}, fired {"compound": 3, "historical": 8, "fts_damped": 0}
