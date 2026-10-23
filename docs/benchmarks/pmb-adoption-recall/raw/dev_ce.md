# locomo dev (cross-rerank on, K=50, entity graph off)

| config | n | R@1 | R@5 | R@10 | MRR | p50 ms | multi-hop R@10 | open-domain R@10 | single-hop R@10 | temporal R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 383 | 0.467 | 0.739 | 0.828 | 0.586 | 900 | 0.770 | 0.421 | 0.835 | 0.944 |

- default: coverage@10 0.767, compound {"n": 0, "coverage@5": 0.0, "coverage@10": 0.0}, fired {"compound": 0, "historical": 1, "fts_damped": 1}
