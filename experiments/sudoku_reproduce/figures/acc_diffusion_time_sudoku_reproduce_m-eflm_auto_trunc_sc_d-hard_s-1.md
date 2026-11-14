# EFLM sudoku_reproduce_m-eflm_auto_trunc_sc_d-hard_s-1 (last.ckpt): accuracy along the sampler (2000 validation boards, 180 steps, seed 3); empty-cell acc / board acc of the argmax at the diffusion time x (1 = decoded)

| sampler | x = 0 | x = 0.02 | x = 0.05 | x = 0.1 | x = 0.25 | x = 0.5 | x = 0.75 | x = 1 | best board acc (at x) |
|---|---|---|---|---|---|---|---|---|---|
| flow sampler (exact velocity) | 83.8 / 6.3 | 93.2 / 76.3 | 93.2 / 78.6 | 93.2 / 79.1 | 93.1 / 79.2 | 93.0 / 79.2 | 93.0 / 79.2 | 92.9 / 79.2 | 79.2 (x = 0.28) |
