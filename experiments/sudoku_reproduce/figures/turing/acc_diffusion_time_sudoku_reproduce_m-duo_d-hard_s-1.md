# DUO sudoku_reproduce_m-duo_d-hard_s-1 (last.ckpt): accuracy along the sampler (2000 validation boards, 180 steps, seed 3); empty-cell acc / board acc of the argmax at the diffusion time x (1 = decoded)

| sampler | x = 0 | x = 0.02 | x = 0.05 | x = 0.1 | x = 0.25 | x = 0.5 | x = 0.75 | x = 1 | best board acc (at x) |
|---|---|---|---|---|---|---|---|---|---|
| ancestral sampler | 81.7 / 4.7 | 82.0 / 5.1 | 82.0 / 5.5 | 82.1 / 7.8 | 82.1 / 19.1 | 83.4 / 36.1 | 84.1 / 46.5 | 84.6 / 51.1 | 51.2 (x = 0.99) |
