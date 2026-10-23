# locomo dev (cross-rerank off, K=50, entity graph on)

| config | n | R@1 | R@5 | R@10 | MRR | p50 ms | multi-hop R@10 | open-domain R@10 | single-hop R@10 | temporal R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qe_off | 383 | 0.285 | 0.553 | 0.687 | 0.411 | 24 | 0.581 | 0.316 | 0.675 | 0.878 |
| qe_0.3 | 383 | 0.274 | 0.561 | 0.692 | 0.406 | 24 | 0.581 | 0.316 | 0.675 | 0.900 |
| qe_0.5 | 383 | 0.274 | 0.564 | 0.692 | 0.408 | 24 | 0.581 | 0.316 | 0.675 | 0.900 |
| qe_1 | 383 | 0.272 | 0.553 | 0.692 | 0.402 | 24 | 0.581 | 0.316 | 0.675 | 0.900 |

- qe_off: coverage@10 0.636, compound {"n": 0, "coverage@5": 0.0, "coverage@10": 0.0}, fired {"compound": 0, "historical": 1, "fts_damped": 1}
- qe_0.3: coverage@10 0.641, compound {"n": 0, "coverage@5": 0.0, "coverage@10": 0.0}, fired {"compound": 0, "historical": 1, "fts_damped": 1}
- qe_0.5: coverage@10 0.641, compound {"n": 0, "coverage@5": 0.0, "coverage@10": 0.0}, fired {"compound": 0, "historical": 1, "fts_damped": 1}
- qe_1: coverage@10 0.640, compound {"n": 0, "coverage@5": 0.0, "coverage@10": 0.0}, fired {"compound": 0, "historical": 1, "fts_damped": 1}
