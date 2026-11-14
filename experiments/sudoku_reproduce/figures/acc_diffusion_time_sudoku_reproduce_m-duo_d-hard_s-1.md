# DUO sudoku_reproduce_m-duo_d-hard_s-1 (last.ckpt): accuracy along the sampler (2000 validation boards, 180 steps, seed 3); empty-cell acc / board acc of the argmax at the diffusion time x (1 = decoded)

| sampler | x = 0 | x = 0.02 | x = 0.05 | x = 0.1 | x = 0.25 | x = 0.5 | x = 0.75 | x = 1 | best board acc (at x) |
|---|---|---|---|---|---|---|---|---|---|
| ancestral sampler | 82.8 / 5.1 | 83.0 / 5.7 | 83.0 / 6.5 | 83.0 / 9.8 | 83.3 / 21.9 | 85.0 / 43.5 | 85.4 / 52.9 | 86.0 / 57.1 | 57.2 (x = 0.99) |
