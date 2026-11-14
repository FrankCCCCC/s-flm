# EFLM sudoku_reproduce_m-eflm_auto_trunc_d-hard_s-1 (last.ckpt): accuracy along the sampler (2000 validation boards, 180 steps, seed 3); empty-cell acc / board acc of the argmax at the diffusion time x (1 = decoded)

| sampler | x = 0 | x = 0.02 | x = 0.05 | x = 0.1 | x = 0.25 | x = 0.5 | x = 0.75 | x = 1 | best board acc (at x) |
|---|---|---|---|---|---|---|---|---|---|
| flow sampler (exact velocity) | 85.2 / 9.4 | 85.2 / 10.3 | 85.0 / 12.8 | 84.7 / 21.6 | 85.0 / 41.6 | 85.0 / 48.4 | 84.9 / 49.4 | 84.9 / 49.7 | 49.7 (x = 0.96) |
