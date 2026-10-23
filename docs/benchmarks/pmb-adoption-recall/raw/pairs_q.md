# locomo test (cross-rerank off, K=10, entity graph off)

| config | n | R@1 | R@5 | R@10 | MRR | p50 ms | pair-question R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| slots_off | 318 | 0.403 | 0.748 | 0.802 | 0.541 | 40 | 0.802 |
| slots_on | 318 | 0.403 | 0.748 | 0.852 | 0.546 | 62 | 0.852 |

- slots_off: coverage@10 0.583, compound {"n": 0, "coverage@5": 0.0, "coverage@10": 0.0}, fired {"compound": 0, "historical": 4, "fts_damped": 0}
- slots_on: coverage@10 0.629, compound {"n": 318, "coverage@5": 0.522, "coverage@10": 0.6292}, fired {"compound": 318, "historical": 4, "fts_damped": 0}
