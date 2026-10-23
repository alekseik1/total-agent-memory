# locomo test (cross-rerank off, K=50, entity graph on)

| config | n | R@1 | R@5 | R@10 | MRR | p50 ms | multi-hop R@10 | open-domain R@10 | single-hop R@10 | temporal R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qe_off | 1153 | 0.304 | 0.588 | 0.673 | 0.433 | 26 | 0.577 | 0.397 | 0.719 | 0.719 |
| qe_0.3 | 1153 | 0.300 | 0.587 | 0.672 | 0.431 | 26 | 0.577 | 0.397 | 0.718 | 0.719 |
| qe_0.5 | 1153 | 0.300 | 0.586 | 0.670 | 0.431 | 27 | 0.577 | 0.397 | 0.715 | 0.719 |
| qe_1 | 1153 | 0.299 | 0.586 | 0.669 | 0.429 | 28 | 0.577 | 0.397 | 0.715 | 0.710 |

- qe_off: coverage@10 0.598, compound {"n": 3, "coverage@5": 0.5, "coverage@10": 0.6667}, fired {"compound": 3, "historical": 8, "fts_damped": 0}
- qe_0.3: coverage@10 0.597, compound {"n": 3, "coverage@5": 0.6667, "coverage@10": 0.6667}, fired {"compound": 3, "historical": 8, "fts_damped": 0}
- qe_0.5: coverage@10 0.596, compound {"n": 3, "coverage@5": 0.6667, "coverage@10": 0.6667}, fired {"compound": 3, "historical": 8, "fts_damped": 0}
- qe_1: coverage@10 0.595, compound {"n": 3, "coverage@5": 0.6667, "coverage@10": 0.6667}, fired {"compound": 3, "historical": 8, "fts_damped": 0}
