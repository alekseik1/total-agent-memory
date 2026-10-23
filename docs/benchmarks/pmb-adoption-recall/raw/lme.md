# lme dev (cross-rerank off, K=50, entity graph off)

| config | n | R@1 | R@5 | R@10 | MRR | p50 ms | knowledge-update R@10 | multi-session R@10 | single-session-assistant R@10 | single-session-preference R@10 | single-session-user R@10 | temporal-reasoning R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 95 | 0.463 | 0.832 | 0.916 | 0.619 | 66 | 1.000 | 0.960 | 1.000 | 0.600 | 1.000 | 0.839 |
| intent_off | 95 | 0.463 | 0.832 | 0.916 | 0.619 | 64 | 1.000 | 0.960 | 1.000 | 0.600 | 1.000 | 0.839 |
| damp_off | 95 | 0.463 | 0.832 | 0.916 | 0.619 | 67 | 1.000 | 0.960 | 1.000 | 0.600 | 1.000 | 0.839 |
| slots_off | 95 | 0.463 | 0.832 | 0.916 | 0.619 | 67 | 1.000 | 0.960 | 1.000 | 0.600 | 1.000 | 0.839 |

- default: coverage@10 0.836, compound {"n": 0, "coverage@5": 0.0, "coverage@10": 0.0}, fired {"compound": 0, "historical": 4, "fts_damped": 0}
- intent_off: coverage@10 0.836, compound {"n": 0, "coverage@5": 0.0, "coverage@10": 0.0}, fired {"compound": 0, "historical": 4, "fts_damped": 0}
- damp_off: coverage@10 0.836, compound {"n": 0, "coverage@5": 0.0, "coverage@10": 0.0}, fired {"compound": 0, "historical": 4, "fts_damped": 0}
- slots_off: coverage@10 0.836, compound {"n": 0, "coverage@5": 0.0, "coverage@10": 0.0}, fired {"compound": 0, "historical": 4, "fts_damped": 0}
