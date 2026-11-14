# MDLM sudoku_reproduce_m-mdlm_d-hard_s-1 (last.ckpt): accuracy along the sampler (2000 validation boards, 180 steps, seed 3); empty-cell acc / board acc of the argmax at the diffusion time x (1 = decoded)

| sampler | x = 0 | x = 0.02 | x = 0.05 | x = 0.1 | x = 0.25 | x = 0.5 | x = 0.75 | x = 1 | best board acc (at x) |
|---|---|---|---|---|---|---|---|---|---|
| ancestral sampler | 82.5 / 2.1 | 82.1 / 4.9 | 82.2 / 9.4 | 81.8 / 17.9 | 81.1 / 34.4 | 80.5 / 37.0 | 80.4 / 37.0 | 80.4 / 37.0 | 37.0 (x = 0.42) |
