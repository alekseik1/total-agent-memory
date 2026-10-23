# locomo test (cross-rerank off, K=10, entity graph off)

| config | n | R@1 | R@5 | R@10 | MRR | p50 ms | pair-and R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| slots_off | 318 | 0.396 | 0.755 | 0.805 | 0.540 | 40 | 0.805 |
| slots_on | 318 | 0.396 | 0.755 | 0.852 | 0.545 | 63 | 0.852 |

- slots_off: coverage@10 0.593, compound {"n": 0, "coverage@5": 0.0, "coverage@10": 0.0}, fired {"compound": 0, "historical": 4, "fts_damped": 0}
- slots_on: coverage@10 0.637, compound {"n": 310, "coverage@5": 0.5293, "coverage@10": 0.6388}, fired {"compound": 310, "historical": 4, "fts_damped": 0}
