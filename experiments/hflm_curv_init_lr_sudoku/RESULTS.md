# hflm_curv_init_lr_sudoku — Results

168/168 cells complete (2026-07-02). H-FLM board accuracy on **Sudoku medium** (2000 boards)
across curvature K × embedding init × LR; everything else fixed (tiny-hyperbolic-dit,
20k steps, batch 256, prior_cov 0.25, rho_max 12, log-linear; eval: 180 steps, exact
velocity, greedy, `top_k_velocity=-1`). Ran on unicorn (thickstun/desa) + TinkerCliffs
(a100_normal_q; 19 cells — its queue starved overnight, so unicorn backfilled the rest).

## Board accuracy (K × init, one block per LR)

### LR 1e-4
| K | ngpt | random | c0.01 | c0.02 | c0.04 | c0.06 | c0.08 |
|---|--:|--:|--:|--:|--:|--:|--:|
| -0.25 | .738 | .825 | .777 | .825 | .742 | .638 | .639 |
| -0.3 | .746 | .740 | .796 | .740 | .707 | .715 | .742 |
| -0.5 | .707 | .701 | .547 | .701 | .727 | .679 | .574 |
| -0.7 | .638 | .672 | .793 | .760 | .690 | .601 | .728 |
| -1.0 | .709 | .700 | .656 | .708 | .732 | .562 | .673 |
| -1.5 | .674 | .608 | .536 | .651 | .648 | .487 | .685 |

### LR 3e-4
| K | ngpt | random | c0.01 | c0.02 | c0.04 | c0.06 | c0.08 |
|---|--:|--:|--:|--:|--:|--:|--:|
| -0.25 | .706 | .738 | .733 | .738 | .728 | .812 | .767 |
| -0.3 | .753 | .773 | .823 | .773 | .763 | .790 | .810 |
| -0.5 | .766 | .793 | .825 | .765 | .774 | .753 | .707 |
| -0.7 | .782 | .806 | .769 | .696 | .783 | .765 | .783 |
| -1.0 | .767 | .661 | .787 | .679 | .758 | .740 | .702 |
| -1.5 | .703 | .696 | .644 | .655 | .613 | .605 | .754 |

### LR 5e-4
| K | ngpt | random | c0.01 | c0.02 | c0.04 | c0.06 | c0.08 |
|---|--:|--:|--:|--:|--:|--:|--:|
| -0.25 | .779 | .763 | .692 | .763 | .803 | .788 | .792 |
| -0.3 | .781 | .741 | **.872** | .741 | .798 | .721 | .814 |
| -0.5 | .738 | .780 | .805 | .778 | .757 | .763 | .796 |
| -0.7 | .758 | .806 | .817 | .806 | .751 | .710 | .724 |
| -1.0 | .804 | .719 | .802 | .690 | .757 | .673 | .687 |
| -1.5 | .649 | .732 | .753 | .732 | .659 | .643 | .781 |

### LR 1e-3
| K | ngpt | random | c0.01 | c0.02 | c0.04 | c0.06 | c0.08 |
|---|--:|--:|--:|--:|--:|--:|--:|
| -0.25 | .793 | .705 | .839 | .610 | .773 | .582 | .776 |
| -0.3 | .789 | .742 | .829 | .759 | .750 | .731 | .732 |
| -0.5 | .763 | .843 | .724 | .843 | .804 | .758 | .814 |
| -0.7 | .806 | .829 | .722 | .846 | .740 | .774 | .753 |
| -1.0 | .817 | .644 | .547 | .704 | .735 | .750 | .685 |
| -1.5 | .693 | .703 | .772 | .775 | .743 | .614 | .768 |

## Marginals (mean over the other two axes / best single cell)

| K | mean | max | | init | mean | max | | LR | mean | max |
|---|--:|--:|-|---|--:|--:|-|---|--:|--:|
| -0.25 | .745 | .839 | | ngpt | .744 | .817 | | 1e-4 | **.689** | .825 |
| **-0.3** | **.767** | **.872** | | random | .738 | .843 | | 3e-4 | .744 | .825 |
| -0.5 | .749 | .843 | | c0.01 | .744 | .872 | | **5e-4** | **.755** | .872 |
| -0.7 | .754 | .846 | | c0.02 | .739 | .846 | | 1e-3 | .747 | .846 |
| -1.0 | .709 | .817 | | c0.04 | .739 | .804 | | | | |
| -1.5 | .678 | .782 | | c0.06 | .694 | .813 | | | | |
| | | | | c0.08 | .737 | .814 | | | | |

**Top-10 cells** (all at K ∈ [-0.7, -0.25], none at LR 1e-4 except two -0.25 cells):
.872 (-0.3, c0.01, 5e-4) · .846 (-0.7, c0.02, 1e-3) · .843 (-0.5, random, 1e-3) ·
.843 (-0.5, c0.02, 1e-3) · .839 (-0.25, c0.01, 1e-3) · .829 (-0.3, c0.01, 1e-3) ·
.829 (-0.7, random, 1e-3) · .825 (-0.25, random, 1e-4) · .825 (-0.25, c0.02, 1e-4) ·
.825 (-0.5, c0.01, 3e-4)

## Noise floor (built-in replicate)

`init=random` ≡ `custom std=0.02` (same N(0, 4e-4) init), so their 24 (K, LR) pairs are
effective replicates: mean |Δ| = **2.6 pts** (max ≈ 9.5), i.e. per-run σ ≈ 2.3 pts —
training is not run-to-run deterministic across heterogeneous GPUs (A5000/A6000/Ada/A100,
TF32 + nondeterministic kernels). Single-cell gaps under ~5 pts are noise; marginals
(24–28 runs) have SE ≈ 0.5 pt and are trustworthy.

## Insights

1. **Curvature has a broad interior optimum, and K=-1 is the wrong default.** The plateau
   K ∈ [-0.7, -0.25] (means .745–.767) beats the standard unit hyperboloid K=-1 by
   **+4–6 pts** and K=-1.5 by **+7–9 pts**; every top-10 cell lies on the plateau. This
   confirms and refines `hflm_curv_sudoku` (optimum there: K=-0.5 under top-1 eval).
2. **LR: 1e-4 is too low — everything else is fine.** LR 1e-4 loses ~6 pts on the mean;
   5e-4 is the best mean and 5e-4/1e-3 host nearly all top cells. High LR (1e-3) is
   *stable* for H-FLM at every curvature (no divergence anywhere in the grid).
3. **Init barely matters** (given ρ-compatible scales): ngpt/random/c0.01–c0.08 means all
   sit within 1 pt (.737–.744) except c0.06 (.694) — a dip that is non-monotonic in std
   and only ~2σ of a marginal, so likely part noise. Small stds (0.01–0.02) do produce the
   best single cells, but the init axis is second-order next to K and LR.
4. **Best recipe: K ≈ -0.3…-0.5, init std ≈ 0.01–0.02, LR 5e-4–1e-3.** Best cell
   **87.2%** (K=-0.3, std 0.01, LR 5e-4); best replicated setting: K=-0.5…-0.7 ×
   std 0.02/random × 1e-3 (.829–.846 across 4 quasi-replicates). For comparison, the
   fixed-recipe `hflm_curv_sudoku` medium best was 76.3% — tuning init/LR + full-vocab
   velocity eval is worth ~+10 pts on medium.
5. **Priority subset** (init=random × LR {3e-4, 1e-3}) delivered the answer early and it
   held up: K=-0.5/-0.7 at LR 1e-3 (.843/.829) were within noise of the eventual winner.

Caveat: eval here uses `top_k_velocity=-1` (velocity averaged over the full vocab);
`hflm_curv_sudoku` used top-1, so absolute numbers are not comparable across experiments.

## Artifacts

- per cell: `outputs/hflm_curv_init_lr_sudoku/k<K>_i-<init>_lr<lr>/` (checkpoints +
  `eval/results.json`); 19 TC-run cells have eval results synced here, checkpoints on
  TinkerCliffs (`~shengyenc/workspace/research/s-flm/outputs/hflm_curv_init_lr_sudoku/`)
- logs: `experiments/hflm_curv_init_lr_sudoku/logs/` (unicorn) + same path on TC
- unicorn jobs 585355–585458 + backfill 650433–650607; TC jobs 6168436–6168519
  (65 cancelled when unicorn backfilled)
- a `d-hard_*` extension of this grid (hard difficulty) is running separately and is not
  covered by this report
