# Sudoku Baselines — Seed-Averaged Results (jul09_2026 spec)

Full-board solve rate (%), mean ± seed-std over seeds {1,2,3}. Eval: exact velocity, top_k_v=-1, 180 steps, greedy last (LangFlow top_k=1 canonical).

coverage: 21/21 (algo,diff) groups | 63/63 seed-runs

| Model | easy | medium | hard |
|---|---|---|---|
| AR | 14.7 ± 3.5 (n=3) | 3.4 ± 0.3 (n=3) | 0.5 ± 0.3 (n=3) |
| S-FLM (naive) | 78.8 ± 1.1 (n=3) | 43.8 ± 3.2 (n=3) | 11.1 ± 1.7 (n=3) |
| S-FLM + trunc | 94.4 ± 0.4 (n=3) | 79.8 ± 1.7 (n=3) | 42.4 ± 3.4 (n=3) |
| S-FLM + trunc + adaptive | 95.0 ± 0.8 (n=3) | 76.7 ± 7.3 (n=3) | 42.2 ± 2.8 (n=3) |
| E-FLM (naive) | 88.2 ± 1.2 (n=3) | 62.2 ± 2.3 (n=3) | 19.2 ± 3.3 (n=3) |
| LangFlow + ada sched | 81.2 ± 0.9 (n=3) | 52.4 ± 2.7 (n=3) | 18.2 ± 2.1 (n=3) |
| LangFlow + ada sched + SC | 97.0 ± 0.5 (n=3) | 87.2 ± 1.9 (n=3) | 50.4 ± 4.6 (n=3) |
