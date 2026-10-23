# locomo dev (cross-rerank off, K=50, entity graph off)

| config | n | R@1 | R@5 | R@10 | MRR | p50 ms | multi-hop R@10 | open-domain R@10 | single-hop R@10 | temporal R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 383 | 0.303 | 0.561 | 0.689 | 0.423 | 25 | 0.595 | 0.316 | 0.675 | 0.878 |

- default: coverage@10 0.636, compound {"n": 0, "coverage@5": 0.0, "coverage@10": 0.0}, fired {"compound": 0, "historical": 0, "fts_damped": 0}
