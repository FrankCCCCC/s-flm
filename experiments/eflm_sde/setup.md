# EFLM SDE Sampler

- Pick and load best checkpoints from ``experiments/eflm_rescale_auto_trunc_tinystories_256`` to evaluate the frontier line with various ``eta`` and ``GScheduler``
- Search for best ``GScheduler`` and ``eta`` that can maintain the same GenPPL with highest entropy, make EFLM become competive to DUO, FLM, MDLM, and S-FLM. Design the best ``GScheduler``
- Make sure the settings are aligned with the existing baselines in ``experiments/eflm_rescale_auto_trunc_tinystories_256``

---

# Frontier Line Evaluation

- GenPPL & Entropy Frontier Evaluation
  - Refer to ``/home/sc3379/workspace/research/s-flm-dev/s-flm/papers/Language Modeling with Hyperspherical Flows.pdf``, Figure 10, draw the GenPPL (Y axis) vs Entropy (X axis) frontier line for each NFE 
  - Cells:
    - For each method: {MDLM, DUO, FLM}, evaluate on the seed-varying = {1, 2, 3} pretrained models
      - NFE: {1, 4, 8, 16, 32, 64, 128, 256}
      - T: {0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15, 1.20}
  - Draw 512 samples for each cell
  - Draw the GenPPL (Y axis) vs Entropy (X axis) frontier line for each NFE, output in ``experiments/naive_ar_tinystories_s256``
  - Report the mean and std across 3 seeds as solid line and the shadow of the frontier
