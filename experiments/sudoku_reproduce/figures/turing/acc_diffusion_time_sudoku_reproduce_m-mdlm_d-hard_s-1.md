# MDLM sudoku_reproduce_m-mdlm_d-hard_s-1 (last.ckpt): accuracy along the sampler (2000 validation boards, 180 steps, seed 3); empty-cell acc / board acc of the argmax at the diffusion time x (1 = decoded)

| sampler | x = 0 | x = 0.02 | x = 0.05 | x = 0.1 | x = 0.25 | x = 0.5 | x = 0.75 | x = 1 | best board acc (at x) |
|---|---|---|---|---|---|---|---|---|---|
| ancestral sampler | 84.7 / 6.2 | 84.2 / 10.9 | 84.3 / 18.2 | 84.1 / 28.8 | 83.6 / 44.4 | 83.2 / 46.2 | 83.1 / 46.2 | 83.0 / 46.2 | 46.2 (x = 0.44) |
