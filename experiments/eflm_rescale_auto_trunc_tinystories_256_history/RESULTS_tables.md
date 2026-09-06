15 cell(s) with GenPPL. Entries: GenPPL (entropy); ⚠ = entropy < 3.0 (degenerate, inadmissible). **bold** = best ADMISSIBLE cell.

### Stage 1 — GenPPL over LR x R at the closed-form truncation (m = 1.0)

| lr \ R | 0.05 | 0.1 | 0.5 | 1 |
|---|---|---|---|---|
| 3e-4 | 13.50 (3.70) | 14.33 (3.67) | 12.60 (3.74) | 12.56 (3.83) |
| 1e-3 | 11.89 (3.47) | 12.25 (3.76) | **10.24** (3.74) | 11.80 (3.85) |

### Valid PPL over LR x R (m = 1.0) — NOT a selection metric

| lr \ R | 0.05 | 0.1 | 0.5 | 1 |
|---|---|---|---|---|
| 3e-4 | 72.63 | 56.64 | 26.90 | 18.11 |
| 1e-3 | 77.09 | 56.58 | 26.92 | 18.28 |
| **tau_max** | 4.665 | 3.981 | 2.444 | 1.834 |

Each column integrates the flow bound over its own horizon (tau_max row), so PPL falls with R for mechanical reasons and is comparable only *within* a column. Select on GenPPL + entropy; PPL is kept as a collapse detector (nan / inf / ~1e134).

### Stage 2 — GenPPL over m x R at lr = 1e-3 (TAU_MAX = m · tau*(R))

| m \ R | 0.05 | 0.1 | 0.5 | 1 |
|---|---|---|---|---|
| 0.85 | · | · | 10.97 (3.79) | · |
| 1.0 | 11.89 (3.47) | 12.25 (3.76) | **10.24** (3.74) | 11.80 (3.85) |
| 1.15 | · | · | 13.44 (3.86) | · |
| 1.25 | · | · | 13.78 (3.94) | · |

### Seed replication (mean +- sd over seeds)

| cell | seeds | GenPPL per seed | mean | sd | entropy |
|---|---|---|---|---|---|
| lr 1e-3, R 0.5, m 1.0 | 1,2,3 | 10.24 / 10.94 / 11.93 | **11.04** | 0.85 | 3.74 / 3.76 / 3.80 |
| lr 1e-3, R 1, m 1.0 | 1,2,3 | 11.80 / 12.07 / 12.42 | **12.10** | 0.31 | 3.85 / 3.88 / 3.82 |

### All cells (best GenPPL first)

| cell | seed | tau_max | valid PPL | GenPPL | entropy | ckpts |
|---|---|---|---|---|---|---|
| lr 1e-3, R 0.5, m 1.0 | 1 | 2.4436 | 26.916 | 10.245 | 3.739 | 31 |
| lr 1e-3, R 0.5, m 1.0 | 2 | 2.4436 | 26.556 | 10.942 | 3.761 | 31 |
| lr 1e-3, R 0.5, m 0.85 | 1 | 2.0771 | 47.271 | 10.965 | 3.790 | 31 |
| lr 1e-3, R 1, m 1.0 | 1 | 1.8338 | 18.284 | 11.805 | 3.846 | 31 |
| lr 1e-3, R 0.05, m 1.0 | 1 | 4.6649 | 77.088 | 11.893 | 3.467 | 31 |
| lr 1e-3, R 0.5, m 1.0 | 3 | 2.4436 | 26.723 | 11.928 | 3.796 | 31 |
| lr 1e-3, R 1, m 1.0 | 2 | 1.8338 | 18.041 | 12.069 | 3.884 | 31 |
| lr 1e-3, R 0.1, m 1.0 | 1 | 3.9811 | 56.577 | 12.249 | 3.763 | 31 |
| lr 1e-3, R 1, m 1.0 | 3 | 1.8338 | 18.117 | 12.417 | 3.821 | 31 |
| lr 3e-4, R 1, m 1.0 | 1 | 1.8338 | 18.109 | 12.561 | 3.835 | 31 |
| lr 3e-4, R 0.5, m 1.0 | 1 | 2.4436 | 26.896 | 12.601 | 3.744 | 31 |
| lr 1e-3, R 0.5, m 1.15 | 1 | 2.8102 | 17.056 | 13.442 | 3.858 | 31 |
| lr 3e-4, R 0.05, m 1.0 | 1 | 4.6649 | 72.632 | 13.498 | 3.704 | 31 |
| lr 1e-3, R 0.5, m 1.25 | 1 | 3.0546 | 13.701 | 13.777 | 3.945 | 31 |
| lr 3e-4, R 0.1, m 1.0 | 1 | 3.9811 | 56.638 | 14.335 | 3.673 | 31 |
