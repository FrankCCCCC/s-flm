# claude_test_hbfm_tinystories_256 — Results

HBFM on TinyStories seq 256 (`forward_type = naive`, `(H^3_K)^16`, log-linear, LR 3e-4): `time_conversion_mode` ×
`time_exp_rate` × curvature × `time_range_upper_bound` (see `EXPERIMENT.md`). Regenerate the tables with
`python experiments/claude_test_hbfm_tinystories_256/report.py`. `val/ppl` is the importance-weighted
denoising-CE bound (not a likelihood; not comparable to the AR / MDLM numbers, and it rewards proposals that
sample resolved states). GenPPL = gpt2-large retokenized generative perplexity of 64 samples (180 steps); read it
WITH the entropy and the distinct-sample count. `unit rate` = rate / |K|; `ub` and `t_max` are physical heat times.

Reference points (`experiments/naive_ar_tinystories_s256/RESULTS.md`, same DiT size / data / steps, mean over
3 seeds at the best LR): MDLM 18.76 @ entropy 4.39, DUO 17.41 @ 4.34, FLM 49.2 @ 4.55, S-FLM + trunc + ada
12.95 @ 3.95; AR 7.2 @ 4.26 (greedy mode decode, 1 distinct sample).

<!-- report:begin -->
188 evaluated (cell, step, horizon) triples.

### Per cell

| fwd | mode | rate | K | unit rate | ub | t_max | step | seed | val/ppl | GenPPL | entropy | uniq | NFE |
|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| naive | exp | 1.5 | -0.5 | 3 | 1 | 0.5 | 30000 | 1 | 1.735 | 13.87 | 3.933 | 64/64 | 180 |
| naive | exp | 1.5 | -0.5 | 3 | 1 | 3 | 30000 | 1 | 1.735 | 16.34 | 3.893 | 64/64 | 180 |
| naive | exp | 1.5 | -0.5 | 3 | 1 | 3 | 25000 | 1 | 1.753 | 16.82 | 3.866 | 64/64 | 180 |
| naive | exp | 1.5 | -0.5 | 3 | 1 | 0.5 | 20000 | 1 | 1.776 | 15.02 | 3.923 | 64/64 | 180 |
| naive | exp | 1.5 | -0.5 | 3 | 1 | 3 | 20000 | 1 | 1.776 | 18.76 | 3.907 | 64/64 | 180 |
| naive | exp | 1.5 | -0.5 | 3 | 1 | 3 | 15000 | 1 | 1.811 | 19.54 | 3.894 | 64/64 | 180 |
| naive | exp | 1.5 | -0.5 | 3 | 1 | 0.5 | 10000 | 1 | 1.871 | 18.31 | 3.929 | 64/64 | 180 |
| naive | exp | 1.5 | -0.5 | 3 | 1 | 3 | 10000 | 1 | 1.871 | 23.11 | 3.925 | 64/64 | 180 |
| naive | exp | 1.5 | -0.5 | 3 | 1 | 3 | 5000 | 1 | 2.012 | 28.29 | 3.863 | 64/64 | 180 |
| naive | exp | 3 | -0.25 | 12 | 1 | 0.5 | 30000 | 1 | 3.049 | 15.09 | 3.924 | 64/64 | 180 |
| naive | exp | 3 | -0.25 | 12 | 1 | 6 | 30000 | 1 | 3.049 | 17.18 | 3.878 | 64/64 | 180 |
| naive | exp | 3 | -0.25 | 12 | 1 | 6 | 25000 | 1 | 3.121 | 18.45 | 3.866 | 64/64 | 180 |
| naive | exp | 3 | -0.25 | 12 | 1 | 0.5 | 20000 | 1 | 3.221 | 16.44 | 3.915 | 64/64 | 180 |
| naive | exp | 3 | -0.25 | 12 | 1 | 6 | 20000 | 1 | 3.221 | 19.70 | 3.865 | 64/64 | 180 |
| naive | exp | 3 | -0.25 | 12 | 1 | 6 | 15000 | 1 | 3.369 | 20.99 | 3.862 | 64/64 | 180 |
| naive | exp | 3 | -0.25 | 12 | 1 | 0.5 | 10000 | 1 | 3.620 | 19.90 | 3.888 | 64/64 | 180 |
| naive | exp | 3 | -0.25 | 12 | 1 | 6 | 10000 | 1 | 3.620 | 22.96 | 3.820 | 64/64 | 180 |
| naive | exp | 3 | -0.25 | 12 | 1 | 6 | 5000 | 1 | 4.250 | 31.75 | 3.866 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 0.25 | 30000 | 1 | 1.715 | 13.45 | 3.920 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 0.5 | 30000 | 1 | 1.715 | 12.90 | 3.928 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 0.5 | 30000 | 2 | 1.713 | 12.95 | 3.876 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 0.5 | 30000 | 3 | 1.715 | 13.90 | 3.920 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 1 | 30000 | 1 | 1.715 | 13.34 | 3.890 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 2 | 30000 | 1 | 1.715 | 14.63 | 3.856 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 3 | 30000 | 1 | 1.715 | 15.56 | 3.849 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 3 | 30000 | 2 | 1.713 | 14.91 | 3.822 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 3 | 30000 | 3 | 1.715 | 16.25 | 3.906 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 0.5 | 25000 | 1 | 1.731 | 13.69 | 3.908 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 0.5 | 25000 | 2 | 1.729 | 13.34 | 3.892 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 0.5 | 25000 | 3 | 1.732 | 14.59 | 3.935 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 3 | 25000 | 1 | 1.731 | 16.34 | 3.844 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 3 | 25000 | 2 | 1.729 | 15.74 | 3.825 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 3 | 25000 | 3 | 1.732 | 17.48 | 3.923 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 0.5 | 20000 | 1 | 1.752 | 14.22 | 3.926 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 0.5 | 20000 | 2 | 1.751 | 13.58 | 3.887 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 0.5 | 20000 | 3 | 1.754 | 15.29 | 3.943 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 3 | 20000 | 1 | 1.752 | 16.87 | 3.867 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 3 | 20000 | 2 | 1.751 | 16.66 | 3.838 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 3 | 20000 | 3 | 1.754 | 18.42 | 3.917 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 0.5 | 15000 | 1 | 1.784 | 15.34 | 3.932 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 0.5 | 15000 | 2 | 1.782 | 14.58 | 3.902 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 0.5 | 15000 | 3 | 1.787 | 15.80 | 3.935 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 3 | 15000 | 1 | 1.784 | 18.56 | 3.883 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 3 | 15000 | 2 | 1.782 | 17.53 | 3.843 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 3 | 15000 | 3 | 1.787 | 19.64 | 3.893 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 0.5 | 10000 | 1 | 1.839 | 16.33 | 3.934 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 0.5 | 10000 | 2 | 1.836 | 15.85 | 3.905 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 0.5 | 10000 | 3 | 1.844 | 17.42 | 3.937 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 1 | 10000 | 1 | 1.839 | 16.54 | 3.907 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 2 | 10000 | 1 | 1.839 | 19.22 | 3.893 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 3 | 10000 | 1 | 1.839 | 20.63 | 3.880 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 3 | 10000 | 2 | 1.836 | 18.10 | 3.844 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 3 | 10000 | 3 | 1.844 | 21.37 | 3.907 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 0.5 | 5000 | 1 | 1.979 | 20.60 | 3.929 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 0.5 | 5000 | 2 | 1.971 | 19.90 | 3.909 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 0.5 | 5000 | 3 | 1.987 | 22.69 | 3.960 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 1 | 5000 | 1 | 1.979 | 21.12 | 3.915 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 2 | 5000 | 1 | 1.979 | 23.58 | 3.903 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 3 | 5000 | 1 | 1.979 | 25.11 | 3.883 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 3 | 5000 | 2 | 1.971 | 21.75 | 3.811 | 64/64 | 180 |
| naive | exp | 3 | -0.5 | 6 | 1 | 3 | 5000 | 3 | 1.987 | 28.00 | 3.934 | 64/64 | 180 |
| naive | exp | 3 | -1 | 3 | 1 | 0.5 | 30000 | 1 | 1.316 | 14.52 | 3.929 | 64/64 | 180 |
| naive | exp | 3 | -1 | 3 | 1 | 1.5 | 30000 | 1 | 1.316 | 17.38 | 3.918 | 64/64 | 180 |
| naive | exp | 3 | -1 | 3 | 1 | 1.5 | 25000 | 1 | 1.323 | 17.02 | 3.899 | 64/64 | 180 |
| naive | exp | 3 | -1 | 3 | 1 | 0.5 | 20000 | 1 | 1.332 | 16.12 | 3.927 | 64/64 | 180 |
| naive | exp | 3 | -1 | 3 | 1 | 1.5 | 20000 | 1 | 1.332 | 18.54 | 3.912 | 64/64 | 180 |
| naive | exp | 3 | -1 | 3 | 1 | 1.5 | 15000 | 1 | 1.345 | 20.49 | 3.910 | 64/64 | 180 |
| naive | exp | 3 | -1 | 3 | 1 | 0.5 | 10000 | 1 | 1.367 | 19.84 | 3.942 | 64/64 | 180 |
| naive | exp | 3 | -1 | 3 | 1 | 1.5 | 10000 | 1 | 1.367 | 22.66 | 3.913 | 64/64 | 180 |
| naive | exp | 3 | -1 | 3 | 1 | 1.5 | 5000 | 1 | 1.419 | 28.43 | 3.902 | 64/64 | 180 |
| naive | exp | 5 | -0.5 | 10 | 1 | 0.5 | 30000 | 1 | 1.728 | 14.06 | 3.926 | 64/64 | 180 |
| naive | exp | 5 | -0.5 | 10 | 1 | 3 | 30000 | 1 | 1.728 | 16.88 | 3.881 | 64/64 | 180 |
| naive | exp | 5 | -0.5 | 10 | 1 | 3 | 25000 | 1 | 1.747 | 16.63 | 3.867 | 64/64 | 180 |
| naive | exp | 5 | -0.5 | 10 | 1 | 0.5 | 20000 | 1 | 1.772 | 14.80 | 3.912 | 64/64 | 180 |
| naive | exp | 5 | -0.5 | 10 | 1 | 3 | 20000 | 1 | 1.772 | 18.68 | 3.890 | 64/64 | 180 |
| naive | exp | 5 | -0.5 | 10 | 1 | 3 | 15000 | 1 | 1.809 | 18.80 | 3.863 | 64/64 | 180 |
| naive | exp | 5 | -0.5 | 10 | 1 | 0.5 | 10000 | 1 | 1.874 | 17.85 | 3.897 | 64/64 | 180 |
| naive | exp | 5 | -0.5 | 10 | 1 | 3 | 10000 | 1 | 1.874 | 20.64 | 3.854 | 64/64 | 180 |
| naive | exp | 5 | -0.5 | 10 | 1 | 3 | 5000 | 1 | 2.033 | 29.44 | 3.893 | 64/64 | 180 |
| naive | exp | 10 | -0.5 | 20 | 1 | 0.5 | 30000 | 1 | 1.795 | 15.01 | 3.930 | 64/64 | 180 |
| naive | exp | 10 | -0.5 | 20 | 1 | 3 | 30000 | 1 | 1.795 | 17.82 | 3.881 | 64/64 | 180 |
| naive | exp | 10 | -0.5 | 20 | 1 | 3 | 25000 | 1 | 1.819 | 18.51 | 3.866 | 64/64 | 180 |
| naive | exp | 10 | -0.5 | 20 | 1 | 0.5 | 20000 | 1 | 1.851 | 16.93 | 3.938 | 64/64 | 180 |
| naive | exp | 10 | -0.5 | 20 | 1 | 3 | 20000 | 1 | 1.851 | 21.41 | 3.917 | 64/64 | 180 |
| naive | exp | 10 | -0.5 | 20 | 1 | 3 | 15000 | 1 | 1.896 | 21.99 | 3.918 | 64/64 | 180 |
| naive | exp | 10 | -0.5 | 20 | 1 | 0.5 | 10000 | 1 | 1.972 | 20.59 | 3.938 | 64/64 | 180 |
| naive | exp | 10 | -0.5 | 20 | 1 | 3 | 10000 | 1 | 1.972 | 24.49 | 3.918 | 64/64 | 180 |
| naive | exp | 10 | -0.5 | 20 | 1 | 3 | 5000 | 1 | 2.157 | 31.72 | 3.897 | 64/64 | 180 |
| naive | exp | 20 | -0.5 | 40 | 1 | 0.5 | 30000 | 1 | 1.818 | 16.36 | 3.942 | 64/64 | 180 |
| naive | exp | 20 | -0.5 | 40 | 1 | 3 | 30000 | 1 | 1.818 | 19.01 | 3.893 | 64/64 | 180 |
| naive | exp | 20 | -0.5 | 40 | 1 | 3 | 25000 | 1 | 1.841 | 20.72 | 3.901 | 64/64 | 180 |
| naive | exp | 20 | -0.5 | 40 | 1 | 0.5 | 20000 | 1 | 1.871 | 18.96 | 3.949 | 64/64 | 180 |
| naive | exp | 20 | -0.5 | 40 | 1 | 3 | 20000 | 1 | 1.871 | 22.22 | 3.898 | 64/64 | 180 |
| naive | exp | 20 | -0.5 | 40 | 1 | 3 | 15000 | 1 | 1.915 | 22.60 | 3.877 | 64/64 | 180 |
| naive | exp | 20 | -0.5 | 40 | 1 | 0.5 | 10000 | 1 | 1.987 | 24.99 | 3.961 | 64/64 | 180 |
| naive | exp | 20 | -0.5 | 40 | 1 | 3 | 10000 | 1 | 1.987 | 28.38 | 3.901 | 64/64 | 180 |
| naive | exp | 20 | -0.5 | 40 | 1 | 3 | 5000 | 1 | 2.121 | 38.31 | 3.918 | 64/64 | 180 |
| naive | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.25 | 30000 | 1 | 1.691 | 13.10 | 3.936 | 64/64 | 180 |
| naive | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.25 | 30000 | 2 | 1.695 | 12.86 | 3.962 | 64/64 | 180 |
| naive | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.25 | 30000 | 3 | 1.695 | 13.48 | 3.978 | 64/64 | 180 |
| naive | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.35 | 30000 | 1 | 1.691 | 12.54 | 3.926 | 64/64 | 180 |
| naive | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.35 | 30000 | 2 | 1.695 | 12.40 | 3.942 | 64/64 | 180 |
| naive | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.35 | 30000 | 3 | 1.695 | 13.38 | 3.969 | 64/64 | 180 |
| naive | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.5 | 30000 | 1 | 1.691 | 12.92 | 3.923 | 64/64 | 180 |
| naive | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.5 | 30000 | 2 | 1.695 | 12.03 | 3.924 | 64/64 | 180 |
| naive | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.5 | 30000 | 3 | 1.695 | 13.10 | 3.963 | 64/64 | 180 |
| naive | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.5 | 25000 | 1 | 1.705 | 13.28 | 3.912 | 64/64 | 180 |
| naive | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.5 | 25000 | 2 | 1.709 | 12.68 | 3.924 | 64/64 | 180 |
| naive | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.5 | 25000 | 3 | 1.709 | 13.65 | 3.946 | 64/64 | 180 |
| naive | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.5 | 20000 | 1 | 1.724 | 13.83 | 3.915 | 64/64 | 180 |
| naive | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.5 | 20000 | 2 | 1.728 | 13.98 | 3.941 | 64/64 | 180 |
| naive | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.5 | 20000 | 3 | 1.728 | 14.14 | 3.955 | 64/64 | 180 |
| naive | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.5 | 15000 | 1 | 1.750 | 14.72 | 3.924 | 64/64 | 180 |
| naive | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.5 | 15000 | 2 | 1.756 | 13.98 | 3.913 | 64/64 | 180 |
| naive | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.5 | 15000 | 3 | 1.756 | 14.75 | 3.951 | 64/64 | 180 |
| naive | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.5 | 10000 | 1 | 1.797 | 16.61 | 3.934 | 64/64 | 180 |
| naive | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.5 | 10000 | 2 | 1.802 | 15.39 | 3.917 | 64/64 | 180 |
| naive | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.5 | 10000 | 3 | 1.803 | 16.38 | 3.927 | 64/64 | 180 |
| naive | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.5 | 5000 | 1 | 1.907 | 20.38 | 3.952 | 64/64 | 180 |
| naive | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.5 | 5000 | 2 | 1.915 | 19.90 | 3.944 | 64/64 | 180 |
| naive | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.5 | 5000 | 3 | 1.916 | 20.56 | 3.962 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 0.5 | 0.25 | 30000 | 1 | 1.695 | 13.26 | 3.932 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 0.5 | 0.5 | 30000 | 1 | 1.695 | 13.18 | 3.934 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 0.5 | 0.5 | 30000 | 2 | 1.698 | 12.87 | 3.922 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 0.5 | 0.5 | 30000 | 3 | 1.698 | 13.78 | 3.951 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 0.5 | 0.5 | 25000 | 1 | 1.710 | 14.00 | 3.938 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 0.5 | 0.5 | 25000 | 2 | 1.712 | 13.50 | 3.915 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 0.5 | 0.5 | 25000 | 3 | 1.712 | 13.84 | 3.941 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 0.5 | 0.5 | 20000 | 1 | 1.728 | 14.71 | 3.939 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 0.5 | 0.5 | 20000 | 2 | 1.731 | 14.44 | 3.933 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 0.5 | 0.5 | 20000 | 3 | 1.732 | 14.39 | 3.954 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 0.5 | 0.5 | 15000 | 1 | 1.755 | 14.62 | 3.938 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 0.5 | 0.5 | 15000 | 2 | 1.758 | 14.34 | 3.905 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 0.5 | 0.5 | 15000 | 3 | 1.759 | 15.84 | 3.945 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 0.5 | 0.5 | 10000 | 1 | 1.800 | 17.07 | 3.961 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 0.5 | 0.5 | 10000 | 2 | 1.803 | 15.61 | 3.923 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 0.5 | 0.5 | 10000 | 3 | 1.806 | 16.83 | 3.942 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 0.5 | 0.5 | 5000 | 1 | 1.907 | 20.66 | 3.978 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 0.5 | 0.5 | 5000 | 2 | 1.915 | 19.30 | 3.932 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 0.5 | 0.5 | 5000 | 3 | 1.916 | 20.70 | 3.989 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 1 | 0.5 | 30000 | 1 | 1.730 | 14.89 | 3.950 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 1 | 1 | 30000 | 1 | 1.730 | 14.77 | 3.922 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 1 | 1 | 25000 | 1 | 1.746 | 15.27 | 3.913 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 1 | 0.5 | 20000 | 1 | 1.767 | 15.58 | 3.935 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 1 | 1 | 20000 | 1 | 1.767 | 15.83 | 3.920 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 1 | 1 | 15000 | 1 | 1.797 | 16.55 | 3.914 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 1 | 0.5 | 10000 | 1 | 1.848 | 18.59 | 3.950 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 1 | 1 | 10000 | 1 | 1.848 | 18.21 | 3.941 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 1 | 1 | 5000 | 1 | 1.969 | 22.50 | 3.960 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 2 | 0.5 | 30000 | 1 | 1.771 | 15.80 | 3.934 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 2 | 2 | 30000 | 1 | 1.771 | 17.18 | 3.900 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 2 | 2 | 25000 | 1 | 1.788 | 18.09 | 3.895 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 2 | 0.5 | 20000 | 1 | 1.812 | 17.31 | 3.930 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 2 | 2 | 20000 | 1 | 1.812 | 18.93 | 3.907 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 2 | 2 | 15000 | 1 | 1.847 | 20.85 | 3.936 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 2 | 0.5 | 10000 | 1 | 1.905 | 20.01 | 3.956 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 2 | 2 | 10000 | 1 | 1.905 | 23.44 | 3.929 | 64/64 | 180 |
| naive | unif | 3 | -0.5 | 6 | 2 | 2 | 5000 | 1 | 2.038 | 29.08 | 3.970 | 64/64 | 180 |
| horo | exp | 3 | -0.5 | 6 | 1 | 0.5 | 30000 | 1 | 1.714 | 12.81 | 3.958 | 64/64 | 180 |
| horo | exp | 3 | -0.5 | 6 | 1 | 3 | 30000 | 1 | 1.714 | 16.43 | 3.902 | 64/64 | 180 |
| horo | exp | 3 | -0.5 | 6 | 1 | 0.5 | 25000 | 1 | 1.730 | 13.89 | 3.952 | 64/64 | 180 |
| horo | exp | 3 | -0.5 | 6 | 1 | 3 | 25000 | 1 | 1.730 | 16.91 | 3.914 | 64/64 | 180 |
| horo | exp | 3 | -0.5 | 6 | 1 | 0.5 | 20000 | 1 | 1.752 | 14.51 | 3.962 | 64/64 | 180 |
| horo | exp | 3 | -0.5 | 6 | 1 | 3 | 20000 | 1 | 1.752 | 18.25 | 3.929 | 64/64 | 180 |
| horo | exp | 3 | -0.5 | 6 | 1 | 0.5 | 15000 | 1 | 1.783 | 15.43 | 3.953 | 64/64 | 180 |
| horo | exp | 3 | -0.5 | 6 | 1 | 3 | 15000 | 1 | 1.783 | 19.99 | 3.935 | 64/64 | 180 |
| horo | exp | 3 | -0.5 | 6 | 1 | 0.5 | 10000 | 1 | 1.837 | 17.00 | 3.964 | 64/64 | 180 |
| horo | exp | 3 | -0.5 | 6 | 1 | 3 | 10000 | 1 | 1.837 | 22.34 | 3.947 | 64/64 | 180 |
| horo | exp | 3 | -0.5 | 6 | 1 | 0.5 | 5000 | 1 | 1.975 | 24.18 | 4.037 | 64/64 | 180 |
| horo | exp | 3 | -0.5 | 6 | 1 | 3 | 5000 | 1 | 1.975 | 31.42 | 4.010 | 64/64 | 180 |
| horo | exp | 10 | -0.5 | 20 | 1 | 0.5 | 30000 | 1 | 1.786 | 14.87 | 3.936 | 64/64 | 180 |
| horo | exp | 10 | -0.5 | 20 | 1 | 3 | 30000 | 1 | 1.786 | 19.08 | 3.930 | 64/64 | 180 |
| horo | exp | 10 | -0.5 | 20 | 1 | 0.5 | 25000 | 1 | 1.812 | 15.86 | 3.938 | 64/64 | 180 |
| horo | exp | 10 | -0.5 | 20 | 1 | 3 | 25000 | 1 | 1.812 | 20.58 | 3.945 | 64/64 | 180 |
| horo | exp | 10 | -0.5 | 20 | 1 | 0.5 | 20000 | 1 | 1.847 | 17.19 | 3.959 | 64/64 | 180 |
| horo | exp | 10 | -0.5 | 20 | 1 | 3 | 20000 | 1 | 1.847 | 22.15 | 3.939 | 64/64 | 180 |
| horo | exp | 10 | -0.5 | 20 | 1 | 0.5 | 15000 | 1 | 1.896 | 18.68 | 3.967 | 64/64 | 180 |
| horo | exp | 10 | -0.5 | 20 | 1 | 3 | 15000 | 1 | 1.896 | 24.64 | 3.956 | 64/64 | 180 |
| horo | exp | 10 | -0.5 | 20 | 1 | 0.5 | 10000 | 1 | 1.976 | 22.01 | 3.979 | 64/64 | 180 |
| horo | exp | 10 | -0.5 | 20 | 1 | 3 | 10000 | 1 | 1.976 | 28.46 | 3.934 | 64/64 | 180 |
| horo | exp | 10 | -0.5 | 20 | 1 | 0.5 | 5000 | 1 | 2.156 | 30.87 | 4.006 | 64/64 | 180 |
| horo | exp | 10 | -0.5 | 20 | 1 | 3 | 5000 | 1 | 2.156 | 43.80 | 4.015 | 64/64 | 180 |
| horo | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.5 | 30000 | 1 | 1.690 | 12.75 | 3.948 | 64/64 | 180 |
| horo | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.5 | 25000 | 1 | 1.704 | 12.99 | 3.928 | 64/64 | 180 |
| horo | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.5 | 20000 | 1 | 1.724 | 13.37 | 3.939 | 64/64 | 180 |
| horo | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.5 | 15000 | 1 | 1.753 | 14.03 | 3.930 | 64/64 | 180 |
| horo | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.5 | 10000 | 1 | 1.803 | 16.52 | 3.953 | 64/64 | 180 |
| horo | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.5 | 5000 | 1 | 1.929 | 22.64 | 3.998 | 64/64 | 180 |

### Mean ± std over seeds

| fwd | mode | rate | K | unit rate | ub | t_max | step | n | val/ppl | GenPPL | entropy |
|---|---|---|---|---|---|---|---|---|---:|---:|---:|
| naive | exp | 1.5 | -0.5 | 3 | 1 | 0.5 | 30000 | 1 | 1.735 | 13.87 | 3.933 |
| naive | exp | 1.5 | -0.5 | 3 | 1 | 3 | 30000 | 1 | 1.735 | 16.34 | 3.893 |
| naive | exp | 1.5 | -0.5 | 3 | 1 | 3 | 25000 | 1 | 1.753 | 16.82 | 3.866 |
| naive | exp | 1.5 | -0.5 | 3 | 1 | 0.5 | 20000 | 1 | 1.776 | 15.02 | 3.923 |
| naive | exp | 1.5 | -0.5 | 3 | 1 | 3 | 20000 | 1 | 1.776 | 18.76 | 3.907 |
| naive | exp | 1.5 | -0.5 | 3 | 1 | 3 | 15000 | 1 | 1.811 | 19.54 | 3.894 |
| naive | exp | 1.5 | -0.5 | 3 | 1 | 0.5 | 10000 | 1 | 1.871 | 18.31 | 3.929 |
| naive | exp | 1.5 | -0.5 | 3 | 1 | 3 | 10000 | 1 | 1.871 | 23.11 | 3.925 |
| naive | exp | 1.5 | -0.5 | 3 | 1 | 3 | 5000 | 1 | 2.012 | 28.29 | 3.863 |
| naive | exp | 3 | -0.25 | 12 | 1 | 0.5 | 30000 | 1 | 3.049 | 15.09 | 3.924 |
| naive | exp | 3 | -0.25 | 12 | 1 | 6 | 30000 | 1 | 3.049 | 17.18 | 3.878 |
| naive | exp | 3 | -0.25 | 12 | 1 | 6 | 25000 | 1 | 3.121 | 18.45 | 3.866 |
| naive | exp | 3 | -0.25 | 12 | 1 | 0.5 | 20000 | 1 | 3.221 | 16.44 | 3.915 |
| naive | exp | 3 | -0.25 | 12 | 1 | 6 | 20000 | 1 | 3.221 | 19.70 | 3.865 |
| naive | exp | 3 | -0.25 | 12 | 1 | 6 | 15000 | 1 | 3.369 | 20.99 | 3.862 |
| naive | exp | 3 | -0.25 | 12 | 1 | 0.5 | 10000 | 1 | 3.620 | 19.90 | 3.888 |
| naive | exp | 3 | -0.25 | 12 | 1 | 6 | 10000 | 1 | 3.620 | 22.96 | 3.820 |
| naive | exp | 3 | -0.25 | 12 | 1 | 6 | 5000 | 1 | 4.250 | 31.75 | 3.866 |
| naive | exp | 3 | -0.5 | 6 | 1 | 0.25 | 30000 | 1 | 1.715 | 13.45 | 3.920 |
| naive | exp | 3 | -0.5 | 6 | 1 | 0.5 | 30000 | 3 | 1.714 ± 0.001 | 13.25 ± 0.56 | 3.908 ± 0.028 |
| naive | exp | 3 | -0.5 | 6 | 1 | 1 | 30000 | 1 | 1.715 | 13.34 | 3.890 |
| naive | exp | 3 | -0.5 | 6 | 1 | 2 | 30000 | 1 | 1.715 | 14.63 | 3.856 |
| naive | exp | 3 | -0.5 | 6 | 1 | 3 | 30000 | 3 | 1.714 ± 0.001 | 15.58 ± 0.67 | 3.859 ± 0.043 |
| naive | exp | 3 | -0.5 | 6 | 1 | 0.5 | 25000 | 3 | 1.731 ± 0.002 | 13.87 ± 0.65 | 3.912 ± 0.022 |
| naive | exp | 3 | -0.5 | 6 | 1 | 3 | 25000 | 3 | 1.731 ± 0.002 | 16.52 ± 0.89 | 3.864 ± 0.052 |
| naive | exp | 3 | -0.5 | 6 | 1 | 0.5 | 20000 | 3 | 1.752 ± 0.002 | 14.36 ± 0.86 | 3.919 ± 0.029 |
| naive | exp | 3 | -0.5 | 6 | 1 | 3 | 20000 | 3 | 1.752 ± 0.002 | 17.32 ± 0.96 | 3.874 ± 0.040 |
| naive | exp | 3 | -0.5 | 6 | 1 | 0.5 | 15000 | 3 | 1.785 ± 0.003 | 15.24 ± 0.62 | 3.923 ± 0.018 |
| naive | exp | 3 | -0.5 | 6 | 1 | 3 | 15000 | 3 | 1.785 ± 0.003 | 18.58 ± 1.06 | 3.873 ± 0.026 |
| naive | exp | 3 | -0.5 | 6 | 1 | 0.5 | 10000 | 3 | 1.840 ± 0.004 | 16.53 ± 0.80 | 3.925 ± 0.018 |
| naive | exp | 3 | -0.5 | 6 | 1 | 1 | 10000 | 1 | 1.839 | 16.54 | 3.907 |
| naive | exp | 3 | -0.5 | 6 | 1 | 2 | 10000 | 1 | 1.839 | 19.22 | 3.893 |
| naive | exp | 3 | -0.5 | 6 | 1 | 3 | 10000 | 3 | 1.840 ± 0.004 | 20.03 ± 1.72 | 3.877 ± 0.032 |
| naive | exp | 3 | -0.5 | 6 | 1 | 0.5 | 5000 | 3 | 1.979 ± 0.008 | 21.06 ± 1.45 | 3.933 ± 0.025 |
| naive | exp | 3 | -0.5 | 6 | 1 | 1 | 5000 | 1 | 1.979 | 21.12 | 3.915 |
| naive | exp | 3 | -0.5 | 6 | 1 | 2 | 5000 | 1 | 1.979 | 23.58 | 3.903 |
| naive | exp | 3 | -0.5 | 6 | 1 | 3 | 5000 | 3 | 1.979 ± 0.008 | 24.95 ± 3.13 | 3.876 ± 0.062 |
| naive | exp | 3 | -1 | 3 | 1 | 0.5 | 30000 | 1 | 1.316 | 14.52 | 3.929 |
| naive | exp | 3 | -1 | 3 | 1 | 1.5 | 30000 | 1 | 1.316 | 17.38 | 3.918 |
| naive | exp | 3 | -1 | 3 | 1 | 1.5 | 25000 | 1 | 1.323 | 17.02 | 3.899 |
| naive | exp | 3 | -1 | 3 | 1 | 0.5 | 20000 | 1 | 1.332 | 16.12 | 3.927 |
| naive | exp | 3 | -1 | 3 | 1 | 1.5 | 20000 | 1 | 1.332 | 18.54 | 3.912 |
| naive | exp | 3 | -1 | 3 | 1 | 1.5 | 15000 | 1 | 1.345 | 20.49 | 3.910 |
| naive | exp | 3 | -1 | 3 | 1 | 0.5 | 10000 | 1 | 1.367 | 19.84 | 3.942 |
| naive | exp | 3 | -1 | 3 | 1 | 1.5 | 10000 | 1 | 1.367 | 22.66 | 3.913 |
| naive | exp | 3 | -1 | 3 | 1 | 1.5 | 5000 | 1 | 1.419 | 28.43 | 3.902 |
| naive | exp | 5 | -0.5 | 10 | 1 | 0.5 | 30000 | 1 | 1.728 | 14.06 | 3.926 |
| naive | exp | 5 | -0.5 | 10 | 1 | 3 | 30000 | 1 | 1.728 | 16.88 | 3.881 |
| naive | exp | 5 | -0.5 | 10 | 1 | 3 | 25000 | 1 | 1.747 | 16.63 | 3.867 |
| naive | exp | 5 | -0.5 | 10 | 1 | 0.5 | 20000 | 1 | 1.772 | 14.80 | 3.912 |
| naive | exp | 5 | -0.5 | 10 | 1 | 3 | 20000 | 1 | 1.772 | 18.68 | 3.890 |
| naive | exp | 5 | -0.5 | 10 | 1 | 3 | 15000 | 1 | 1.809 | 18.80 | 3.863 |
| naive | exp | 5 | -0.5 | 10 | 1 | 0.5 | 10000 | 1 | 1.874 | 17.85 | 3.897 |
| naive | exp | 5 | -0.5 | 10 | 1 | 3 | 10000 | 1 | 1.874 | 20.64 | 3.854 |
| naive | exp | 5 | -0.5 | 10 | 1 | 3 | 5000 | 1 | 2.033 | 29.44 | 3.893 |
| naive | exp | 10 | -0.5 | 20 | 1 | 0.5 | 30000 | 1 | 1.795 | 15.01 | 3.930 |
| naive | exp | 10 | -0.5 | 20 | 1 | 3 | 30000 | 1 | 1.795 | 17.82 | 3.881 |
| naive | exp | 10 | -0.5 | 20 | 1 | 3 | 25000 | 1 | 1.819 | 18.51 | 3.866 |
| naive | exp | 10 | -0.5 | 20 | 1 | 0.5 | 20000 | 1 | 1.851 | 16.93 | 3.938 |
| naive | exp | 10 | -0.5 | 20 | 1 | 3 | 20000 | 1 | 1.851 | 21.41 | 3.917 |
| naive | exp | 10 | -0.5 | 20 | 1 | 3 | 15000 | 1 | 1.896 | 21.99 | 3.918 |
| naive | exp | 10 | -0.5 | 20 | 1 | 0.5 | 10000 | 1 | 1.972 | 20.59 | 3.938 |
| naive | exp | 10 | -0.5 | 20 | 1 | 3 | 10000 | 1 | 1.972 | 24.49 | 3.918 |
| naive | exp | 10 | -0.5 | 20 | 1 | 3 | 5000 | 1 | 2.157 | 31.72 | 3.897 |
| naive | exp | 20 | -0.5 | 40 | 1 | 0.5 | 30000 | 1 | 1.818 | 16.36 | 3.942 |
| naive | exp | 20 | -0.5 | 40 | 1 | 3 | 30000 | 1 | 1.818 | 19.01 | 3.893 |
| naive | exp | 20 | -0.5 | 40 | 1 | 3 | 25000 | 1 | 1.841 | 20.72 | 3.901 |
| naive | exp | 20 | -0.5 | 40 | 1 | 0.5 | 20000 | 1 | 1.871 | 18.96 | 3.949 |
| naive | exp | 20 | -0.5 | 40 | 1 | 3 | 20000 | 1 | 1.871 | 22.22 | 3.898 |
| naive | exp | 20 | -0.5 | 40 | 1 | 3 | 15000 | 1 | 1.915 | 22.60 | 3.877 |
| naive | exp | 20 | -0.5 | 40 | 1 | 0.5 | 10000 | 1 | 1.987 | 24.99 | 3.961 |
| naive | exp | 20 | -0.5 | 40 | 1 | 3 | 10000 | 1 | 1.987 | 28.38 | 3.901 |
| naive | exp | 20 | -0.5 | 40 | 1 | 3 | 5000 | 1 | 2.121 | 38.31 | 3.918 |
| naive | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.25 | 30000 | 3 | 1.694 ± 0.002 | 13.15 ± 0.32 | 3.959 ± 0.021 |
| naive | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.35 | 30000 | 3 | 1.694 ± 0.002 | 12.77 ± 0.53 | 3.945 ± 0.022 |
| naive | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.5 | 30000 | 3 | 1.694 ± 0.002 | 12.68 ± 0.57 | 3.937 ± 0.023 |
| naive | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.5 | 25000 | 3 | 1.708 ± 0.002 | 13.20 ± 0.49 | 3.927 ± 0.017 |
| naive | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.5 | 20000 | 3 | 1.727 ± 0.003 | 13.98 ± 0.15 | 3.937 ± 0.020 |
| naive | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.5 | 15000 | 3 | 1.754 ± 0.003 | 14.48 ± 0.43 | 3.930 ± 0.020 |
| naive | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.5 | 10000 | 3 | 1.801 ± 0.003 | 16.13 ± 0.65 | 3.926 ± 0.008 |
| naive | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.5 | 5000 | 3 | 1.913 ± 0.005 | 20.28 ± 0.34 | 3.953 ± 0.009 |
| naive | unif | 3 | -0.5 | 6 | 0.5 | 0.25 | 30000 | 1 | 1.695 | 13.26 | 3.932 |
| naive | unif | 3 | -0.5 | 6 | 0.5 | 0.5 | 30000 | 3 | 1.697 ± 0.002 | 13.27 ± 0.46 | 3.936 ± 0.015 |
| naive | unif | 3 | -0.5 | 6 | 0.5 | 0.5 | 25000 | 3 | 1.711 ± 0.002 | 13.78 ± 0.25 | 3.931 ± 0.014 |
| naive | unif | 3 | -0.5 | 6 | 0.5 | 0.5 | 20000 | 3 | 1.730 ± 0.002 | 14.51 ± 0.17 | 3.942 ± 0.011 |
| naive | unif | 3 | -0.5 | 6 | 0.5 | 0.5 | 15000 | 3 | 1.757 ± 0.002 | 14.93 ± 0.80 | 3.929 ± 0.021 |
| naive | unif | 3 | -0.5 | 6 | 0.5 | 0.5 | 10000 | 3 | 1.803 ± 0.003 | 16.51 ± 0.78 | 3.942 ± 0.019 |
| naive | unif | 3 | -0.5 | 6 | 0.5 | 0.5 | 5000 | 3 | 1.913 ± 0.005 | 20.22 ± 0.79 | 3.967 ± 0.030 |
| naive | unif | 3 | -0.5 | 6 | 1 | 0.5 | 30000 | 1 | 1.730 | 14.89 | 3.950 |
| naive | unif | 3 | -0.5 | 6 | 1 | 1 | 30000 | 1 | 1.730 | 14.77 | 3.922 |
| naive | unif | 3 | -0.5 | 6 | 1 | 1 | 25000 | 1 | 1.746 | 15.27 | 3.913 |
| naive | unif | 3 | -0.5 | 6 | 1 | 0.5 | 20000 | 1 | 1.767 | 15.58 | 3.935 |
| naive | unif | 3 | -0.5 | 6 | 1 | 1 | 20000 | 1 | 1.767 | 15.83 | 3.920 |
| naive | unif | 3 | -0.5 | 6 | 1 | 1 | 15000 | 1 | 1.797 | 16.55 | 3.914 |
| naive | unif | 3 | -0.5 | 6 | 1 | 0.5 | 10000 | 1 | 1.848 | 18.59 | 3.950 |
| naive | unif | 3 | -0.5 | 6 | 1 | 1 | 10000 | 1 | 1.848 | 18.21 | 3.941 |
| naive | unif | 3 | -0.5 | 6 | 1 | 1 | 5000 | 1 | 1.969 | 22.50 | 3.960 |
| naive | unif | 3 | -0.5 | 6 | 2 | 0.5 | 30000 | 1 | 1.771 | 15.80 | 3.934 |
| naive | unif | 3 | -0.5 | 6 | 2 | 2 | 30000 | 1 | 1.771 | 17.18 | 3.900 |
| naive | unif | 3 | -0.5 | 6 | 2 | 2 | 25000 | 1 | 1.788 | 18.09 | 3.895 |
| naive | unif | 3 | -0.5 | 6 | 2 | 0.5 | 20000 | 1 | 1.812 | 17.31 | 3.930 |
| naive | unif | 3 | -0.5 | 6 | 2 | 2 | 20000 | 1 | 1.812 | 18.93 | 3.907 |
| naive | unif | 3 | -0.5 | 6 | 2 | 2 | 15000 | 1 | 1.847 | 20.85 | 3.936 |
| naive | unif | 3 | -0.5 | 6 | 2 | 0.5 | 10000 | 1 | 1.905 | 20.01 | 3.956 |
| naive | unif | 3 | -0.5 | 6 | 2 | 2 | 10000 | 1 | 1.905 | 23.44 | 3.929 |
| naive | unif | 3 | -0.5 | 6 | 2 | 2 | 5000 | 1 | 2.038 | 29.08 | 3.970 |
| horo | exp | 3 | -0.5 | 6 | 1 | 0.5 | 30000 | 1 | 1.714 | 12.81 | 3.958 |
| horo | exp | 3 | -0.5 | 6 | 1 | 3 | 30000 | 1 | 1.714 | 16.43 | 3.902 |
| horo | exp | 3 | -0.5 | 6 | 1 | 0.5 | 25000 | 1 | 1.730 | 13.89 | 3.952 |
| horo | exp | 3 | -0.5 | 6 | 1 | 3 | 25000 | 1 | 1.730 | 16.91 | 3.914 |
| horo | exp | 3 | -0.5 | 6 | 1 | 0.5 | 20000 | 1 | 1.752 | 14.51 | 3.962 |
| horo | exp | 3 | -0.5 | 6 | 1 | 3 | 20000 | 1 | 1.752 | 18.25 | 3.929 |
| horo | exp | 3 | -0.5 | 6 | 1 | 0.5 | 15000 | 1 | 1.783 | 15.43 | 3.953 |
| horo | exp | 3 | -0.5 | 6 | 1 | 3 | 15000 | 1 | 1.783 | 19.99 | 3.935 |
| horo | exp | 3 | -0.5 | 6 | 1 | 0.5 | 10000 | 1 | 1.837 | 17.00 | 3.964 |
| horo | exp | 3 | -0.5 | 6 | 1 | 3 | 10000 | 1 | 1.837 | 22.34 | 3.947 |
| horo | exp | 3 | -0.5 | 6 | 1 | 0.5 | 5000 | 1 | 1.975 | 24.18 | 4.037 |
| horo | exp | 3 | -0.5 | 6 | 1 | 3 | 5000 | 1 | 1.975 | 31.42 | 4.010 |
| horo | exp | 10 | -0.5 | 20 | 1 | 0.5 | 30000 | 1 | 1.786 | 14.87 | 3.936 |
| horo | exp | 10 | -0.5 | 20 | 1 | 3 | 30000 | 1 | 1.786 | 19.08 | 3.930 |
| horo | exp | 10 | -0.5 | 20 | 1 | 0.5 | 25000 | 1 | 1.812 | 15.86 | 3.938 |
| horo | exp | 10 | -0.5 | 20 | 1 | 3 | 25000 | 1 | 1.812 | 20.58 | 3.945 |
| horo | exp | 10 | -0.5 | 20 | 1 | 0.5 | 20000 | 1 | 1.847 | 17.19 | 3.959 |
| horo | exp | 10 | -0.5 | 20 | 1 | 3 | 20000 | 1 | 1.847 | 22.15 | 3.939 |
| horo | exp | 10 | -0.5 | 20 | 1 | 0.5 | 15000 | 1 | 1.896 | 18.68 | 3.967 |
| horo | exp | 10 | -0.5 | 20 | 1 | 3 | 15000 | 1 | 1.896 | 24.64 | 3.956 |
| horo | exp | 10 | -0.5 | 20 | 1 | 0.5 | 10000 | 1 | 1.976 | 22.01 | 3.979 |
| horo | exp | 10 | -0.5 | 20 | 1 | 3 | 10000 | 1 | 1.976 | 28.46 | 3.934 |
| horo | exp | 10 | -0.5 | 20 | 1 | 0.5 | 5000 | 1 | 2.156 | 30.87 | 4.006 |
| horo | exp | 10 | -0.5 | 20 | 1 | 3 | 5000 | 1 | 2.156 | 43.80 | 4.015 |
| horo | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.5 | 30000 | 1 | 1.690 | 12.75 | 3.948 |
| horo | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.5 | 25000 | 1 | 1.704 | 12.99 | 3.928 |
| horo | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.5 | 20000 | 1 | 1.724 | 13.37 | 3.939 |
| horo | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.5 | 15000 | 1 | 1.753 | 14.03 | 3.930 |
| horo | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.5 | 10000 | 1 | 1.803 | 16.52 | 3.953 |
| horo | trunc_exp | 3 | -0.5 | 6 | 0.5 | 0.5 | 5000 | 1 | 1.929 | 22.64 | 3.998 |
<!-- report:end -->

## Analysis (2026-09-19, all 15 cells at 30k; per-read tables above regenerated by `report.py`, summary tables in `logs/final_tables.md`; verified against the raw `samples_genppl.json` files)

The 9 `rate-*_ada-*` directories in `outputs/` are the paused horosphere / 32-factor round (`EXPERIMENT.md` → Previous
round) and are not part of this grid.

**Answer.** The best (mode, rate, K, ub) is **`trunc_exp`, `time_exp_rate = 3`, `K = −0.5`, `time_range_upper_bound = 0.5`**:
GenPPL **12.64 ± 0.43** at unigram entropy **3.93 ± 0.02** (3 seeds: 12.59 / 12.25 / 13.10; 64/64 distinct samples in every
read). The spec point (`exp`, rate 3, K −0.5) reads 15.58 ± 0.67 @ 3.86 ± 0.04 at its default sampler horizon (physical
t_max 3 = unit 1.5) and 13.25 ± 0.56 @ 3.91 ± 0.03 when its samples are drawn at the same horizon as the bounded cell
(t_max 0.5). The 0.61 GenPPL (4.6%) gap between the two at matched horizon is about one seed-std (Welch t ≈ 1.5 on
3 vs 3 seeds, p ≈ 0.2, not significant), but the bounded proposal leads at every checkpoint (10k 15.97 vs 16.53, 20k
13.98 vs 14.36, 30k 12.64 vs 13.25) and its seed spread is smaller at every step (table D). Everything else on the four
axes is worse.

### Ranking at 30k, sorted by GenPPL at the matched physical horizon 0.5

Own horizon = the sampler horizon the cell trains for (exp: unit 1.5 = physical 1.5/|K|; bounded modes: ub), shown
alongside. For the K ≠ −0.5 cells the 0.5 column is matched in physical, not unit, time. Distinct samples: 64/64 in every row.

| rank | mode | rate | K | unit rate | ub | seeds | GenPPL @ own horizon | GenPPL @ horizon 0.5 | entropy @ 0.5 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | trunc_exp | 3 | −0.5 | 6 | 0.5 | 3 | **12.64 ± 0.43** | (same sampler) | 3.93 ± 0.02 |
| 2 | unif | – | −0.5 | – | 0.5 | 3 | 13.19 ± 0.36 | (same sampler) | 3.94 ± 0.01 |
| 3 | exp (spec) | 3 | −0.5 | 6 | – | 3 | 15.58 ± 0.67 | 13.25 ± 0.56 | 3.91 ± 0.03 |
| 4 | exp | 1.5 | −0.5 | 3 | – | 1 | 16.34 | 13.87 | 3.93 |
| 5 | exp | 5 | −0.5 | 10 | – | 1 | 16.88 | 14.06 | 3.93 |
| 6 | exp | 3 | −1.0 | 3 | – | 1 | 17.38 | 14.52 | 3.93 |
| 7 | unif | – | −0.5 | – | 1.0 | 1 | 14.77 | 14.89 | 3.95 |
| 8 | exp | 10 | −0.5 | 20 | – | 1 | 17.82 | 15.01 | 3.93 |
| 9 | exp | 3 | −0.25 | 12 | – | 1 | 17.18 | 15.09 | 3.92 |
| 10 | unif | – | −0.5 | – | 2.0 | 1 | 17.18 | 15.80 | 3.93 |
| 11 | exp | 20 | −0.5 | 40 | – | 1 | 19.01 | 16.36 | 3.94 |

References (same DiT / data / steps, 3-seed means): S-FLM + trunc + ada 12.95 @ 3.95, DUO 17.41 @ 4.34, MDLM 18.76 @ 4.39.
All 64 samples are distinct in every one of the 152 eval reads (140 distinct (cell, step, horizon, seed) triples plus 12
duplicate re-reads; 145/133 of them existed when this section was first written, before the 2026-09-20 horizon probe); entropy is 3.82–3.96 in every 30k read (spread 0.14), so ranking by GenPPL alone is fair *within*
this sweep. HBFM's entropy sits ~0.45 nat below MDLM / DUO (like S-FLM's 3.95), so the ~30% lower GenPPL than MDLM / DUO
is a comparison at lower unigram diversity; against S-FLM + trunc + ada at the same entropy, the best HBFM cell ties
(12.64 ± 0.43 vs 12.95).

### What each axis says

- **Sampler horizon is the biggest lever (table C).** The spec model (seed 1, 30k) sampled at physical t_max 0.25 / 0.5 /
  1 / 2 / 3 gives 13.45 / 12.90 / 13.34 / 14.63 / 15.56: flat over 0.25–1.0 (all within 4%, one seed, ≈ read noise; the
  exact minimum inside 0.25–1.0 is not resolved) and rising steeply beyond 1 (+13% at 2, +21% at 3). The 3-seed spec cell
  improves 15.58 → 13.25 (−15%) from the horizon alone, with entropy going *up* (3.86 → 3.91), and every exp cell moves
  the same way (rate 1.5 / 5 / 10 / 20 and both curvatures: −12% to −17%). Starting the bridge at unit 1.5 spends most of
  the 180 steps in the uninformative region of the proposal.
- **Rate (exp, K −0.5): the spec's rate 3 (unit 6, the ESS optimum of the calibration) is the best point of the ladder,
  and the ladder is monotone above it at both horizons:** 1.5 / 3 / 5 / 10 / 20 → 16.34 / 15.58 / 16.88 / 17.82 / 19.01
  (own horizon) and 13.87 / 13.25 / 14.06 / 15.01 / 16.36 (horizon 0.5). The basin is broad (rate 1.5–5 within 5–8% of the
  spec at either horizon, one seed each, inside the spec's seed range — not resolved) and falls off by 13–24% at rate
  10–20 (resolved). Rate 3 ≤ 5 < 10 < 20 holds at every checkpoint from 5k (rate 5 ties the spec's seed 1 at 10k, 20.64
  vs 20.63, at the default horizon); rate 1.5 and 5 swap places between checkpoints. The hypothesis' alternative optimum
  at unit ~25 (the sudoku transfer) is refuted; the ESS calibration's prediction (unit 6) held.
- **Curvature at rate 3 is the unit-rate axis again.** K −0.25 (unit 12, R 2) → 17.18 / 15.09 and K −1 (unit 3, R 1) →
  17.38 / 14.52 vs K −0.5 (unit 6) 15.58 / 13.25: both 10–14% worse (K −1 at horizon 0.5, +9.6%, is the most marginal
  gap that decides the ranking). The rate/|K| equivalence holds to ~6%: K −1 rate 3 (unit 3, own horizon unit 1.5) 17.38
  vs K −0.5 rate 1.5 (unit 3, own horizon unit 1.5) 16.34; K −0.25 rate 3 (unit 12) 17.18 sits between K −0.5 rate 5 / 10
  (unit 10 / 20) 16.88 / 17.82. The residual is the optimizer / DiT-input scale `R = 1/sqrt|K|` that the symmetry does not
  cover. (The `_tmax0.5` rows of the K ≠ −0.5 cells are matched in *physical*, not unit, time: unit 0.5 for K −1, unit
  0.125 for K −0.25.)
- **Bounded proposals (trunc_exp / unif) win only through the horizon, and only when ub is short.** `trunc_exp` ub 0.5 is
  exp(rate 3) with the latest 22% of draws (t > 0.5, P = e^−1.5) removed and the sampler bounded at 0.5. The two
  ingredients separate at matched horizon: the horizon accounts for −15% (15.58 → 13.25 on the spec cell) and the
  late-draw removal for a further −4.6% at fixed horizon 0.5 (13.25 → 12.64, within seed noise). The truncated model
  does **not** prefer a horizon shorter than its bound (2026-09-20, same-node reads of the 30k checkpoints): t_max
  0.25 / 0.35 / 0.5 → 13.15 ± 0.32 / 12.77 ± 0.53 / 12.57 ± 0.53, worse in 3/3 seeds at 0.25, and unif ub 0.5 agrees
  (13.26 vs 13.18). `t_max > ub` is rejected at construction, so 0.5 = ub is this cell's optimum; ub 0.5 is a real
  optimum of the ladder rather than its lowest rung, and no ub 0.25 cell was trained. `unif` ub 0.5 (single seed) is 13.18: 4% behind trunc_exp, tied with the spec-at-0.5 mean.
  Wider uniform proposals are monotonically worse at both horizons — ub 1.0 → 14.77 (14.89 at 0.5: a shorter sampler
  does not rescue it), ub 2.0 → 17.18 (15.80 at 0.5) — i.e. the wide proposal degrades the trained model itself, not just
  the sampler, in line with the calibration's ESS 0.63 / 0.33 for unit ub 0.5 / 1.0 (= physical 1.0 / 2.0 at K −0.5).
- **Seed variance.** The spec cell's seeds spread 12–13% at 5k, 9% at 10k and 4% at 30k (std/mean, own horizon);
  trunc_exp's 1–4% at every step. Single-seed axis gaps below ~8% (rate 1.5 / 5 vs 3, unif 0.5 vs trunc_exp) are
  therefore not resolved; the gaps that decide the ranking (rate ≥ 10, K ≠ −0.5, unif ≥ 1.0 at matched horizon: 10–24%)
  are.
- **`val/ppl` is not a quality number.** It is the importance-weighted denoising-CE bound in the units of the proposal /
  geometry: K −1 reads 1.32 and K −0.25 3.05 against 1.71 at K −0.5, uncorrelated with GenPPL (the lowest bound, K −1,
  has the worst GenPPL), and the three best cells differ by ≈ 0.02 (trunc 1.69, unif 0.5 1.70, spec 1.71). Within one
  (mode, K) family it tracks the rate ordering for rate ≥ 3 (1.71 → 1.73 → 1.80 → 1.82 for rate 3 → 5 → 10 → 20; rate 1.5
  reads 1.74 despite its better GenPPL), nothing more.

### Stage 2: planned vs run

Planned (`EXPERIMENT.md`): seeds 2–3 of the spec and of each winning axis' best cell, plus an edge point on any axis whose
optimum sits at the end of its ladder. Run (`CONFIRM` in `sweep.py`): rate 1.5 seed 1 (low-side rate point and the K −1
equivalence partner), spec seeds 2–3, trunc_exp ub 0.5 seeds 2–3. Not run: trunc_exp ub 1.0 (the spec at t_max 1.0, 13.34,
is its proxy), rate 40 (rate 20 lost), and unif ub 0.25 although ub 0.5 is the best and lowest point of the unif ladder
(the unif axis is open on the low side, like trunc_exp's). Rate 5 got no seeds because it lost at matched horizon (10k
17.85 vs 16.33, 20k 14.80 vs 14.22, 30k 14.06 vs 13.25, seed 1) despite tying the spec at the default horizon at 10k–25k.

### Eval-noise note

The sampler is seeded: re-evaluating the same checkpoint on the same node reproduces GenPPL to the last digit (trunc
seed 3 @ 30k 13.10 / 13.10, seed 2 @ 10k 15.39 / 15.39, unif 0.5 @ 30k 13.18 / 13.18 — all thickstun-compute-01 both times,
per the node names in `logs/mode-*_<job>.log`). Re-reads on a different node / GPU type differ by 0.01–0.46: seed 1 @ 10k
16.15 vs 16.61 (two tick reads of the same keep-10000.ckpt on thickstun-compute-01 vs kuleshov-compute-03, which rules
out the "re-saved checkpoint" guess in the Status log), seed 2 @ 20k 13.97 vs 13.98, the spec's 25k default read 16.21 vs
16.34, seed 1 @ 30k 12.59 (Ada) vs 12.92 (A5000), seed 2 @ 30k 12.25 (A6000) vs 12.03 (Ada). Cross-node comparisons
therefore carry ~2–3% read noise on top of seed noise. The tables in this section use the training job's own final read
(`eval/`, `last.ckpt`); the per-read block above (`report.py`) keys rows by (cell, step, horizon) and the later-sorting
`eval_step30000_tmax0.5/` overwrites `eval/`, so it shows the keep-30000 re-reads for the trunc_exp seeds (12.92 / 12.03 /
13.10 → 12.68 ± 0.57; at 10k 16.13 ± 0.65). Either choice gives the same ranking (12.68 ± 0.57 vs spec-at-0.5 13.25 ± 0.56).

### Answer to the task, and what would sharpen it

Use `time_conversion_mode = trunc_exp`, `time_exp_rate = 3`, `prod_factor_gaussian_curvature = −0.5`,
`time_range_upper_bound = 0.5` (GenPPL 12.64 ± 0.43 @ 3.93). If the proposal must stay `exp`, keep rate 3 / K −0.5 and
sample with `sampler.t_max = 0.5` (13.25 ± 0.56 @ 3.91) — never the default horizon. Unresolved within this budget:
the ub ladder *above* 0.5 for the bounded modes (0.75; below 0.5 is closed by the horizon reads), the low side of the
rate ladder (1.5 vs 3, one seed), and 64-sample GenPPL itself (±2–3% cross-node read noise; 256 samples would halve it).
**Closed 2026-09-21:** trunc_exp vs unif at ub 0.5 is settled on 3 seeds each — trunc_exp 12.59 / 12.25 / 13.10 =
12.64 ± 0.43 vs unif 13.18 / 12.84 / 13.55 = 13.19 ± 0.36, so trunc_exp is 4.3% ahead and 8 of the 9 pairwise
seed comparisons favour it. On 3-vs-3 that is a consistent but not statistically decisive win (Welch t ≈ 1.7, p ≈ 0.17);
the ranking is unchanged and the recommended naive config remains trunc_exp ub 0.5.

## Previous round — `forward_type = horosphere` (paused 2026-09-17 11:20; static, not regenerated)

Same recipe with the backbone's logits as a residual on the Busemann log-densities, LR 3e-4, K −0.5, physical
rate = unit rate × 0.5, horizon 2.0 physical, seed 1. Stopped at 5k–11k of 30k steps for the refactoring; these
are the kept-checkpoint reads.

| rate | unit rate | noise | step | val/ppl | GenPPL | entropy | uniq |
|---|---|---|---|---:|---:|---:|---:|
| 0.01 | 0.02 | log-linear | 5000 | 1.249 | 316.19 | 2.849 | 64/64 |
| 0.01 | 0.02 | log-linear-adaptive | 5000 | 1.251 | 184.21 | 3.545 | 64/64 |
| 0.5 | 1 | log-linear | 5000 | 1.653 | 110.80 | 3.880 | 64/64 |
| 2.5 | 5 | log-linear | 5000 | 1.458 | 46.48 | 3.966 | 64/64 |
| 2.5 | 5 | log-linear | 10000 | 1.394 | 33.39 | 3.957 | 64/64 |
| 10 | 20 | log-linear | 5000 | 1.461 | 42.80 | 3.978 | 64/64 |
| 30 | 60 | log-linear | 5000 | — | 103.5 | 3.88 | 64/64 |

Take-aways that shaped this round: the rate has a broad optimum between unit 5 and 20 with cliffs on both sides;
rate 0.01 collapses (word salad at the lowest val/ppl of all — the denoising bound is not a quality metric);
the adaptive schedule only partly rescues a mismatched rate. Throughput: Ada 3.2 s/step, A6000 5.5, A5000 6.9
(4 GPUs, global batch 512).

## Status

- 2026-09-17 20:15 EDT: spec revised by the user (`naive` forward, `(H^3)^16`, `time_conversion_mode`, `time_range_upper_bound`);
  scripts extended (`TIME_CONVERSION_MODE`, `TIME_RANGE_UPPER_BOUND`, `FORWARD_TYPE`, `KEEP_EVERY`, `PROPOSAL_RATE`,
  `T_MAX`), sweep/report rewritten for the new grid; smoke tests of all three modes (train → resume → eval, negative
  test of `T_MAX > ub`, exact sweep knobs with `EMBED_DIM 48`) and the HBFM test suite run on GPU nodes before submission
  (`logs/tests/`; the 5 test failures on HEAD are pre-existing, see EXPERIMENT.md). Two bugs caught and fixed before any
  4-GPU job: the `keep-{step}` hydra quoting and the missing `sampler.t_max` at training for the bounded modes.
- 2026-09-17 21:15 EDT: 16-factor calibration (`logs/calib/calib16.log`): context-free posterior one-hot at unit tau 0.8
  (0.4 on 32 factors), 98% of the CE mass below tau 0.5, ESS-optimal unit exp rate 6 = the spec's rate 3 at K −0.5;
  sampler horizon set to unit 1.5 (physical 3.0 at K −0.5).
- 2026-09-17 21:20 EDT: stage 1 submitted (10 cells, seed 1, 4 GPUs each), SLURM jobs 390981–390990 in nice order
  spec (exp 3, K −0.5) → rate 5 / 10 / 20 → K −0.25 / −1 → unif 1.0 → trunc_exp 0.5 → unif 0.5 / 2.0. GPUs at submission:
  A5000 node 9 free, Ada node 1 free, A6000 node full (an 8-GPU 2-day job) → 2 cells start now, the rest queue. A
  2-hourly babysit tick evaluates keep-5k/10k/… checkpoints (`sweep.py --eval-step N`, plus the horizon control
  `--t-max 0.5/1.0/2.0` on the spec cell), refreshes this table and adds stage-2 cells to `CONFIRM`.
- 2026-09-17 21:25 EDT: five cells started at once (GPUs had freed up): spec on the Ada node, rate 5 / 10 on the A5000
  node, rate 20 / K −0.25 on the A6000 node; K −1 and the four bounded-mode cells queue. Steady-state throughput
  (rank-0 micro-batches/s, 4 GPUs, global batch 512, naive forward, 16 factors): Ada 9.1 it/s at micro-batch 16 →
  0.9 s/step → ~7.3 h per 30k cell (keep-5000 after ~1.2 h); A6000 4.3 it/s → 1.9 s/step → ~16 h; A5000 5.8 it/s at
  micro-batch 8 → 2.7 s/step → ~23 h. About 3.5× faster than the horosphere / 32-factor round (3.2 / 5.5 / 6.9 s/step).
- 2026-09-17 22:45 EDT, first read — spec cell (exp, rate 3, K −0.5) @ 5k/30k: GenPPL **25.1** @ entropy 3.88 (64/64
  distinct, val/ppl 1.98) at the default horizon 3.0 physical; the horizon control at 0.5 physical (unit 0.25) on the
  same checkpoint gives **20.6** @ 3.93. Already below the previous round's best 5k read (42.8, horosphere, 32 factors)
  and within reach of the 30k-step MDLM / DUO references (18.8 / 17.4 @ 4.4). Samples are coherent TinyStories prose.
  The shorter horizon helping (fewer of the 180 steps spent on resolved states → finer dt in the decision window, or an
  earlier greedy commit) makes the sampler horizon an eval-time lever; the 1.0 / 2.0 controls are running.
- 2026-09-17 22:40 EDT, horizon control complete on the spec cell @ 5k (same checkpoint, 180 steps, greedy last step):
  t_max 0.5 / 1.0 / 2.0 / 3.0 physical → GenPPL 20.6 / 21.1 / 23.6 / 25.1 at entropy 3.93 / 3.92 / 3.90 / 3.88 (all 64/64).
  Monotone: the shorter the bridge, the better the GenPPL at slightly higher entropy. Consequence for the design: a
  bounded-mode cell (unif / trunc_exp at ub 0.5–1.0) gets ~15–20% of GenPPL from its horizon alone, so unif / trunc_exp
  must beat exp-at-the-same-horizon (the `_tmax` rows), not the default-horizon exp row. The training-side comparison
  of the exp cells among themselves (rate ladder, K) is unaffected (same horizon 1.5 unit). Next 5k reads: K −0.25 and
  rate 20 (A6000 pair) ≈ 23:20, rate 5 / 10 (A5000 pair) ≈ 00:45.
- 2026-09-17 23:35 EDT, 5k reads of the A6000 pair (default horizon unit 1.5): exp rate 20 (unit 40) → GenPPL 38.3 @ 3.92;
  exp rate 3 at K −0.25 (unit 12) → 31.8 @ 3.87; both behind the spec point (25.1 @ 3.88), in the order the calibration's
  ESS predicts (0.96 / ~0.7 / 0.08 for unit 6 / 12 / 40). The K −0.25 cell's val/ppl (4.25 vs 1.98) is not comparable
  across K (different importance weights and input scale). Spec cell reached 10k: its 10k eval and horizon controls
  (t_max 0.5 / 1.0 / 2.0) are queued (jobs 398753–398756). Still to read at 5k: rate 5 / 10 (A5000, ≈ 00:50), K −1 and
  the four bounded-mode cells (queued behind the running ones).
- 2026-09-17 23:40 EDT, spec cell @ 10k/30k: GenPPL **20.6** @ 3.88 at the default horizon (25.1 at 5k; val/ppl 1.84);
  horizon control t_max 0.5 / 1.0 / 2.0 → **16.3** @ 3.93 / 16.5 @ 3.91 / 19.2 @ 3.89 (64/64 everywhere). At one third
  of the training, the spec point sampled at horizon 0.5 is already below the 30k-step MDLM / DUO references (18.8 /
  17.4), at ~0.4 lower entropy (3.93 vs 4.3–4.4; the S-FLM+trunc+ada reference is 12.95 @ 3.95). The horizon ordering
  is identical at 5k and 10k, so it is a property of the sampler, not of training progress.
- 2026-09-18 00:45 EDT, spec cell @ 15k: GenPPL **18.6** @ 3.88 (default horizon) / **15.3** @ 3.93 (horizon 0.5), val/ppl
  1.78. Trajectory 5k → 10k → 15k: 25.1 → 20.6 → 18.6 (default) and 20.6 → 16.3 → 15.3 (horizon 0.5), still improving;
  the horizon gain is a steady ~17%. Other cells: rate 5 / 10 at ~4.7k (5k reads ≈ 01:00), rate 20 / K −0.25 at ~8.3k
  (10k reads ≈ 01:40); K −1 and the bounded modes still queued behind other users' jobs.
- 2026-09-18 01:15 EDT, 5k rate ladder complete (K −0.5, default horizon): rate 3 / 5 / 10 / 20 → GenPPL **25.1** / 29.4 /
  31.7 / 38.3 at entropy 3.88–3.92 (64/64), monotone in the rate; K −0.25 at rate 3 (unit 12) → 31.8, between rate 5 and
  10 at K −0.5 (unit 10 / 20), consistent with the rate/|K| equivalence. The calibration's ESS ordering (unit 6 / 10 / 20 /
  40 → 0.96 / 0.79 / 0.21 / 0.08) predicts the ranking exactly. Since nothing above the spec rate helps, stage 2 adds one
  point BELOW it, rate 1.5 (unit 3, job 401319, nice 55 — also the equivalence partner of the queued K −1 / rate 3 cell),
  and seeds 2–3 of the spec cell (jobs 401320–401321, nice 100–101, behind every stage-1 axis so K −1 and the bounded
  modes are not delayed). 13 cells in the grid.
- 2026-09-18 01:45 EDT, 10k reads of the A6000 pair (default horizon): rate 20 → 28.4 @ 3.90 (from 38.3; spec 20.6);
  K −0.25 at rate 3 → **23.0 @ 3.82** (from 31.8): the flatter curvature improves faster between 5k and 10k (−28% vs
  −18% for the spec cell) and now sits 11% behind the spec point at a lower entropy — the sudoku pattern (K ranking
  shifts with training length) may be repeating, so the K axis is decided at 30k, not now. Spec cell reached 20k; its
  20k evals go out via the batch tick (interactive `srun` allocations have been timing out on the controller since
  ~00:30, so eval submissions now run as a tiny `sbatch` job, `logs/tests/tick_*.log`).
- 2026-09-18 01:50 EDT, spec cell @ 20k: GenPPL **16.9** @ 3.87 (default horizon) / **14.2** @ 3.93 (horizon 0.5), val/ppl
  1.75. Trajectory 5k / 10k / 15k / 20k: 25.1 / 20.6 / 18.6 / 16.9 (default) and 20.6 / 16.3 / 15.3 / 14.2 (horizon 0.5),
  still falling ~9% per 5k. At the default horizon it now beats MDLM (18.8) and DUO (17.4) at ~0.5 lower entropy; at
  horizon 0.5 it is within 10% of S-FLM+trunc+ada (12.95 @ 3.95) with 10k steps to go.
- 2026-09-18 02:55 EDT, spec cell @ 25k: GenPPL **16.2** @ 3.85 (default horizon), val/ppl 1.73; −4% vs 20k, the curve
  is flattening (constant LR, no decay). 30k finish + final eval ≈ 03:40. Controller note: `sbatch` also timed out once
  and CPU tick jobs at nice 100 sat pending behind GPU jobs; ticks now go out at nice 0 (they are 1-CPU, 1-minute jobs).
- 2026-09-18 03:25 EDT, spec cell @ 25k, horizon 0.5: **13.7** @ 3.91 (from 14.2 at 20k), 6% above S-FLM+trunc+ada
  (12.95 @ 3.95) at a slightly lower entropy. Rate 20 / K −0.25 reach 15k ≈ 03:40; rate 5 / 10 at ~7.8k.
- 2026-09-18 03:45 EDT, 15k reads (default horizon): rate 20 → 22.6 @ 3.88 (28.4 at 10k); K −0.25 → 21.0 @ 3.86 (23.0
  at 10k); spec at 15k was 18.6, so the gaps are 22% and 13% at equal steps (K −0.25's catch-up slowed: −9% from 10k to
  15k vs the spec's −10%). Eval noise: two ticks raced and the spec's 25k default eval ran twice on the same checkpoint
  → 16.21 vs 16.34 GenPPL (the table keeps the later one), i.e. ~1% run-to-run noise from nondeterministic kernels —
  differences below ~0.3 GenPPL are not resolvable with 64 samples. Spec cell reached 30k at 03:42; its final eval is
  running inside the training job.
- 2026-09-18 04:00 EDT — **first cell complete.** Spec cell (exp, rate 3, K −0.5, seed 1) @ 30k: GenPPL **15.6** @ entropy
  3.85 at the default horizon (unit 1.5 / physical 3.0), **12.9** @ 3.93 at horizon 0.5 physical; val/ppl 1.72; 64/64
  distinct at both. Trajectory (default / horizon 0.5): 5k 25.1 / 20.6 → 10k 20.6 / 16.3 → 15k 18.6 / 15.3 → 20k
  16.9 / 14.2 → 25k 16.3 / 13.7 → 30k 15.6 / 12.9. Versus the 30k references (same DiT / data / steps): MDLM 18.8 @
  4.39, DUO 17.4 @ 4.34, S-FLM+trunc+ada 12.95 @ 3.95 — HBFM at the spec point matches S-FLM+trunc+ada with the short
  horizon and beats the discrete baselines by 17% even at the default horizon, at ~0.5 lower unigram entropy than
  MDLM / DUO (the S-FLM+trunc+ada comparison is at matched entropy). 7.3 h of 4 Ada GPUs. The full horizon curve on the
  final checkpoint (t_max 0.25 / 1.0 / 2.0 in addition to 0.5 / 3.0) is queued. K −1 started on the freed Ada GPUs.
- 2026-09-18 04:10 EDT, horizon curve on the spec cell's final (30k) checkpoint, 180 steps, greedy last step:
  t_max 0.25 / 0.5 / 1.0 / 2.0 / 3.0 physical (unit 0.125 / 0.25 / 0.5 / 1.0 / 1.5) → GenPPL 13.5 / **12.9** / 13.3 / 14.6 /
  15.6 at entropy 3.92 / 3.93 / 3.89 / 3.86 / 3.85 (64/64 everywhere). The curve has a minimum at 0.5 physical, i.e. about
  a quarter of the context-free resolution scale (unit 0.8): shorter bridges lose a little (the SDE has too few
  informative steps: dt = 0.0014 physical at t_max 0.25), longer ones lose ~1 GenPPL per extra unit of physical time.
  0.25–1.0 are within 4% (≈ 2–3× the eval noise). Recommendation for every cell: report the default-horizon row (the
  spec's protocol) AND the matched t_max 0.5 row; the bounded-mode cells with ub 0.5 are sampled at exactly this optimum.
- 2026-09-18 04:15 EDT, matched-horizon (t_max 0.5) rows at 10k: spec 16.3 @ 3.93, K −0.25 19.9 @ 3.89, rate 20 25.0 @
  3.96 — same ordering as at the default horizon (20.6 / 23.0 / 28.4), horizon gain 12–21% for all. From here the tick
  adds the t_max 0.5 row to every cell at 10k / 20k / 30k.
- 2026-09-18 04:55 EDT, 10k rate ladder complete (default horizon): rate 3 / 5 / 10 / 20 → 20.6 / **20.6** / 24.5 / 28.4
  at entropy 3.88 / 3.85 / 3.92 / 3.90. Rate 5 (unit 10) caught up with the spec rate between 5k and 10k (29.4 → 20.6,
  −30%, vs −18% for rate 3), so the 5k ranking was premature on the low side: the rate optimum is a broad plateau over
  unit 6–10, with unit 20 / 40 clearly worse (+19% / +38%). Whether rate 5 overtakes rate 3 is decided at 20k / 30k
  (A5000 node: 20k ≈ 09:30, 30k ≈ 17:00); if it does, its seeds 2–3 join stage 2. K −0.25 (23.0) sits between rate 5
  and rate 10 at equal unit rate 12, as the rate/|K| equivalence predicts.
- 2026-09-18 05:00 EDT, matched horizon (t_max 0.5) at 10k: spec 16.3 vs rate 5 **17.9** (@ 3.90) — the tie at the
  default horizon does not carry over: rate 5 gains only 13% from the short horizon (20.6 → 17.9) against 21% for the
  spec (20.6 → 16.3), so at the best sampler setting the spec rate leads by 9%. The faster proposal (mean unit tau 0.10 vs
  0.17) appears to train a posterior that is already sharp at the default horizon and has less to gain from committing
  earlier. Rate 10's matched row is running; K −1 reaches 5k ≈ 05:05, rate 20 / K −0.25 reach 20k ≈ 05:50.
- 2026-09-18 05:10 EDT: rate 10 at matched horizon 0.5 @ 10k → 20.6 @ 3.94 (default 24.5), so the short-horizon ladder
  at 10k is spec 16.3 < rate 5 17.9 < rate 10 20.6 < rate 20 25.0, monotone in the rate. K −1 (rate 3, unit 3, R = 1)
  @ 5k → **28.4** @ 3.90 (val/ppl 1.42, not comparable across K): the curvature axis at 5k reads K −0.5 25.1 < K −1 28.4
  < K −0.25 31.8, i.e. the spec curvature is best on both sides at equal steps; K −1 is the equivalence partner of the
  queued rate-1.5 / K −0.5 cell (same unit rate 3). K −1 runs on the Ada node (30k ≈ 11:20).
- 2026-09-18 05:50 EDT, 20k reads: rate 20 → 22.2 @ 3.90 (matched horizon 0.5: 19.0 @ 3.95); K −0.25 → 19.7 @ 3.87
  (matched row running). Spec @ 20k was 16.9 / 14.2 → gaps +32% / +17% (default) and +33% (rate 20, matched). Both
  cells finish 30k ≈ 08:00 (A6000 node).
- 2026-09-18 06:20 EDT: K −1 @ 10k → 22.7 @ 3.91 (matched 0.5: 19.8 @ 3.94), the same as K −0.25 @ 10k (23.0 / 19.9):
  the curvature axis at rate 3 is symmetric around the spec value, K −0.5 20.6 / 16.3 < K −1 ≈ K −0.25 by ~10% (default)
  / ~21% (matched), in line with unit rate 6 being the ESS optimum and unit 3 / 12 sitting on either shoulder. K −0.25
  @ 20k matched → 16.4 (spec 14.2, +16%). Full 10k table (default | matched 0.5): spec 20.6 | 16.3; rate 5 20.6 | 17.9;
  rate 10 24.5 | 20.6; rate 20 28.4 | 25.0; K −0.25 23.0 | 19.9; K −1 22.7 | 19.8.
- 2026-09-18 07:30 EDT: K −1 @ 15k → 20.5 @ 3.91; the 15k curvature ordering is K −0.5 18.6 < K −1 20.5 < K −0.25 21.0
  (both shoulders +10–13%), unchanged from 10k. Rate 20 @ 15k 22.6 for reference.
- 2026-09-18 07:55 EDT, 25k reads: K −0.25 → 18.5 @ 3.87, rate 20 → 20.7 @ 3.90 (spec 16.3): +13% / +27%. Both finish
  30k ≈ 09:50 (A6000 node), which frees it for the next queued cells (unif 1.0 at nice 60 first).
- 2026-09-18 08:40 EDT: K −1 @ 20k → 18.5 @ 3.91 (matched 0.5: 16.1 @ 3.93); spec @ 20k 16.9 / 14.2 → +10% / +13%;
  K −0.25 @ 20k 19.7 / 16.4. At 20k the sharper curvature (K −1, unit 3) edges the flatter one (K −0.25, unit 12) by
  6%, both behind K −0.5. K −1 finishes 30k ≈ 09:40 (Ada node).
- 2026-09-18 08:45 EDT, 15k reads of the A5000 pair: rate 5 → 18.8 @ 3.86 (spec 18.6: tie within the ~1% eval noise
  at the default horizon), rate 10 → 22.0 @ 3.92 (+18%). Full 15k default-horizon ladder: rate 3 18.6 ≈ rate 5 18.8 <
  K −1 20.5 < K −0.25 21.0 < rate 10 22.0 < rate 20 22.6. Rate 5's 20k matched-horizon row (≈ 11:00) decides whether it
  joins stage 2.
- 2026-09-18 09:45 EDT: K −1 @ 25k → 17.0 @ 3.90 (spec 16.3): the gap shrank from +13% (10k) → +10% (20k) → +4% (25k);
  the sharper curvature (unit rate 3, R = 1) is still improving faster than the spec cell late in training, so its 30k
  read (≈ 10:20) matters for the curvature call. K −0.25 @ 25k was 18.5 (+13%).
- 2026-09-18 10:00 EDT — **cells 2 and 3 complete** (A6000 node, 12.6 h each). Final 30k, default horizon / matched
  t_max 0.5: K −0.25 at rate 3 → **17.2** @ 3.88 / **15.1** @ 3.92 (val/ppl 3.05); rate 20 → **19.0** @ 3.89 / **16.4**
  @ 3.94 (val/ppl 1.82); spec 15.6 / 12.9. So K −0.25 ends +10% / +17% and rate 20 +22% / +27% behind the spec point;
  rate 20 at 30k is still worse than the spec at 20k. Both freed GPUs were taken at once by unif ub 1.0 (stage 1) and
  rate 1.5 (stage 2), which started 09:59 on the A6000 node (~16 h each). Running: rate 5 / 10 (A5000, ~17k), K −1
  (Ada, ~27k, done ≈ 10:35), unif 1.0, rate 1.5. Queued: trunc_exp 0.5, unif 0.5 / 2.0, spec seeds 2–3.
- 2026-09-18 10:50 EDT — **cell 4 complete.** K −1 at rate 3 (unit 3, R = 1) @ 30k: **17.4** @ 3.92 default / **14.5** @
  3.93 matched (val/ppl 1.33); the 25k → 30k step gave nothing (17.0 → 17.4, inside the noise), so the late catch-up
  stalled. Curvature axis at 30k (default | matched): K −0.5 **15.6 | 12.9** < K −0.25 17.2 | 15.1 ≈ K −1 17.4 | 14.5 —
  the spec curvature wins by 10–17% on both shoulders; between the shoulders, the sharper K −1 is better at the short
  horizon (14.5 vs 15.1) and the flatter K −0.25 at the default one (17.2 vs 17.4), both inside ~2× eval noise. Its 4
  Ada GPUs went to trunc_exp ub 0.5 (started 10:45, ~7 h). Complete: spec, K −0.25, rate 20, K −1. Running: rate 5 / 10
  (A5000, ~18k), unif 1.0 + rate 1.5 (A6000, ~2k), trunc_exp 0.5 (Ada). Queued: unif 0.5 / 2.0, spec seeds 2–3.
- 2026-09-18 12:00 EDT, first bounded-mode read: trunc_exp (rate 3, K −0.5, ub 0.5, horizon 0.5) @ 5k → **20.4** @ 3.95
  (val/ppl 1.91), vs the spec cell @ 5k sampled at the same horizon 0.5 → 20.6 @ 3.93 and at its default horizon → 25.1.
  Truncating the training proposal to [0, 0.5] (22% of the exp draws, 19% of the CE mass) is neutral at 5k; the whole
  19% gain over the spec protocol is the sampler horizon, exactly what the horizon control predicted. Samples coherent,
  64/64. trunc_exp runs on the Ada node (30k ≈ 17:45).
- 2026-09-18 12:15 EDT, 5k reads of the A6000 pair (started 09:59): **rate 1.5 at K −0.5 → 28.3 @ 3.86** vs its
  equivalence partner **K −1 at rate 3 → 28.4 @ 3.90** (both unit rate 3; R = 1.41 vs 1, loss scale ×2 apart): identical
  within the eval noise — the `(rate, K) → (c·rate, c·K)` symmetry found on sudoku holds on TinyStories at 5k with the
  clip at 1.0 (the loss-scale ratio here is only 2×, not 300×). Both sit 13% behind the spec (unit 6), so unit 3 is on
  the low shoulder as the ESS curve predicts. **unif ub 1.0 (horizon 1.0) → 22.5 @ 3.96** vs the spec @ 5k at the same
  horizon 1.0 → 21.1 @ 3.92: the uniform proposal on [0, 1.0] is ~7% worse than exp(3) at matched horizon at 5k (ESS
  0.63 vs 0.96 in the calibration), and 10% better than the spec's default-protocol row (25.1) purely through the
  horizon. Mode axis so far at 5k, all at their own horizon vs exp at the same horizon: trunc_exp 0.5 20.4 vs 20.6 (tie),
  unif 1.0 22.5 vs 21.1 (−7%).
- 2026-09-18 12:30 EDT, 20k reads of the A5000 pair (default horizon): rate 5 → 18.7 @ 3.89, rate 10 → 21.4 @ 3.92 (spec
  16.9): the 10k / 15k tie of rate 5 with the spec was transient — at 20k it is +11% behind, so the rate axis is
  decided for the spec rate 3 (unit 6) and rate 5 gets no extra seeds. Full 20k ladder (default | matched 0.5): spec
  **16.9 | 14.2** < K −1 18.5 | 16.1 < rate 5 18.7 | 14.8 < K −0.25 19.7 | 16.4 < rate 10 21.4 | 16.9 <
  rate 20 22.2 | 19.0 (matched-horizon ordering: spec 14.2 < rate 5 14.8 < K −1 16.1 < K −0.25 16.4 < rate 10 16.9 <
  rate 20 19.0 — same ranking on both horizons; rate 5 is closest to the spec at the short horizon, 4% behind). Stage 2 therefore stays at: rate 1.5 (running, equivalence partner of K −1) + spec seeds 2–3
  (queued); no axis winner other than the spec point has emerged on the exp side.
- 2026-09-18 13:05 EDT: trunc_exp ub 0.5 @ 10k → 16.2 @ 3.93 (its `_tmax0.5` duplicate on the same checkpoint, same
  horizon: 16.6 @ 3.93 — a direct eval-noise read of ~3%, larger than the ~1% seen on the spec's 25k duplicate; treat
  differences below ~0.5 GenPPL as noise) vs the spec @ 10k matched → 16.3. Tie again: the truncated proposal tracks
  exp-at-the-same-horizon at 5k and 10k.
- 2026-09-18 14:10 EDT: trunc_exp ub 0.5 @ 15k → 14.7 @ 3.92 vs spec @ 15k matched → 15.3 @ 3.93 (−4%, ~1–2× eval
  noise). Trajectory of the mode comparison at horizon 0.5 (trunc_exp / exp): 5k 20.4 / 20.6, 10k 16.2 / 16.3, 15k
  14.7 / 15.3 — on par, with a hint of an edge appearing; decided at 30k (≈ 17:45).
- 2026-09-18 14:20 EDT, 10k reads of the A6000 pair: rate 1.5 at K −0.5 → 23.1 @ 3.93 vs K −1 at rate 3 → 22.7 @ 3.91
  (unit rate 3 both): the equivalence holds at 10k too (Δ 2%, inside noise). unif ub 1.0 → **18.2** @ 3.94 at its horizon
  1.0 vs spec @ 10k at horizon 1.0 → 16.5 (+10%); sampling unif at 0.5 gives 18.6, i.e. the horizon lever that buys the
  exp cells 15–20% does nothing for the uniform-trained model — its posterior is not sharper early, consistent with a
  proposal that spends 20% of its draws at unit tau > 0.5 where the exp proposal never goes. Mode axis at 10k, each at
  its own horizon vs exp at the same horizon: trunc_exp 0.5 16.2 vs 16.3 (tie), unif 1.0 18.2 vs 16.5 (−10%).
- 2026-09-18 14:25 EDT: rate 1.5 @ 10k at t_max 0.5 → 18.3 @ 3.93 vs K −1 @ 10k at t_max 0.5 → 19.8. Caveat: the
  `_tmax0.5` rows are matched in PHYSICAL time, which is unit 0.25 at K −0.5 but unit 0.5 at K −1 (and unit 0.125 at
  K −0.25), so across different K only the default-horizon rows (unit 1.5 everywhere) are horizon-matched; the
  equivalence test therefore reads 23.1 vs 22.7 (default rows), not 18.3 vs 19.8. Within a fixed K the `_tmax0.5` rows
  are exactly comparable.
- 2026-09-18 15:20 EDT: trunc_exp ub 0.5 @ 20k → 13.8 @ 3.92 (duplicate eval identical this time) vs spec @ 20k matched
  → 14.2 @ 3.93 (−3%). trunc_exp / exp at horizon 0.5: 5k 20.4 / 20.6, 10k 16.2 / 16.3, 15k 14.7 / 15.3, 20k 13.8 / 14.2 —
  a consistent 0–4% edge for the truncated proposal from 15k on, at the edge of eval noise. 30k ≈ 17:45 decides whether
  trunc_exp joins stage 2 (seeds 2–3) as the mode winner.
- 2026-09-18 16:20 EDT, 25k reads (default): rate 5 → 16.6 @ 3.87 (spec 16.3, +2%), rate 10 → 18.5 @ 3.87 (+13%). Rate 5
  vs spec across checkpoints: 5k +17%, 10k 0%, 15k +1%, 20k +11%, 25k +2% — the gap oscillates at the level of the
  checkpoint-to-checkpoint fluctuation (a constant-LR run without decay), so at the default horizon rate 3 and rate 5
  (unit 6 and 10) are statistically tied on one seed; the matched-horizon rows (10k: 16.3 vs 17.9; 20k: 14.2 vs 14.8)
  still favour rate 3 by 4–9%. The 30k pair decides the tie-break; the rate optimum is best described as the plateau
  unit 6–10 with the spec rate at its low edge.
- 2026-09-18 16:25 EDT: trunc_exp ub 0.5 @ 25k → 13.3 @ 3.91 vs spec @ 25k matched → 13.7 (−3%): third consecutive
  checkpoint (15k −4%, 20k −3%, 25k −3%) with the same small edge, so it is probably real though ≤ 2× eval noise per
  point. unif ub 1.0 @ 15k → 16.6 @ 3.91 at horizon 1.0 (the spec has no horizon-1.0 row at 15k; interpolating its
  default→1.0 ratio, 0.80 at 10k / 0.86 at 30k, puts exp at ≈ 15.4–16.0 there, so unif stays ~5–8% behind).
- 2026-09-18 16:30 EDT: rate 1.5 @ 15k → 19.5 @ 3.89 (K −1 @ 15k 20.5; spec 18.6). Equivalence pair at 5k / 10k / 15k:
  28.3 / 23.1 / 19.5 vs 28.4 / 22.7 / 20.5 — within the checkpoint fluctuation throughout.
- 2026-09-18 17:30 EDT — **cell 5 complete: trunc_exp ub 0.5 (rate 3, K −0.5, horizon = ub = 0.5) @ 30k → GenPPL
  12.59 @ 3.92**, val/ppl 1.69, 64/64 — the best final number so far, 2.4% below the spec point sampled at the same
  horizon (12.90 @ 3.93) and below the S-FLM+trunc+ada reference (12.95 @ 3.95). Its trajectory vs exp-at-horizon-0.5:
  5k 20.4 / 20.6, 10k 16.2 / 16.3, 15k 14.7 / 15.3, 20k 13.8 / 14.2, 25k 13.3 / 13.7, 30k 12.6 / 12.9 — a steady 2–4%
  edge from 15k on, i.e. real on this seed but at the eval-noise scale. Stage 2: seeds 2–3 of trunc_exp 0.5 added to
  `CONFIRM` (nice 102 / 103, behind the spec's seeds 2–3), so the final answer can compare 3-seed means of the two
  candidates. Its 4 Ada GPUs went to unif ub 0.5 (started 17:27, ~7 h). Complete: spec, K −0.25, rate 20, K −1,
  trunc_exp 0.5. Running: rate 5 / 10 (A5000, ~27k, done ≈ 18:15), unif 1.0 + rate 1.5 (A6000, ~16k), unif 0.5 (Ada).
  Queued: unif 2.0, spec seeds 2–3, trunc_exp seeds 2–3.
- 2026-09-18 18:30 EDT, 20k reads of the A6000 pair: rate 1.5 → 18.8 @ 3.91 (K −1 @ 20k 18.5; spec 16.9) — equivalence
  pair still within 1–5% at every checkpoint; unif ub 1.0 → 15.8 @ 3.92 at horizon 1.0 (spec @ 20k: 16.9 default / 14.2
  at 0.5; the spec's horizon-1.0 row at 20k is not sampled, interpolating its 30k ratio 13.34/15.56 gives ≈ 14.5), so
  unif stays ~9% behind exp at matched horizon. Rate 5 / 10 reach 30k ≈ 18:45 (A5000).
- 2026-09-18 18:35 EDT: the tick's `_tmax0.5` re-eval of trunc_exp's final checkpoint (same checkpoint, same horizon)
  gave 12.92 @ 3.92 vs the in-job 12.59 @ 3.92 — 2.6% eval noise, the same size as trunc_exp's edge over the spec at
  matched horizon (12.90). Pooling both reads, trunc_exp 0.5 ≈ 12.75 vs spec 12.90 on seed 1: indistinguishable. The
  eval noise floor (~1–3% per 64-sample eval) is now measured on three duplicate pairs (spec 25k, trunc 10k, trunc 30k);
  the seeds-2–3 cells of both candidates are what will decide the mode axis.
- 2026-09-18 18:45 EDT: unif ub 0.5 (horizon 0.5) @ 5k → 20.7 @ 3.98 — a three-way tie with trunc_exp 0.5 (20.4) and exp
  at horizon 0.5 (20.6) at 5k. On the short range [0, 0.5] physical (unit 0.25, 40% of the CE mass at tau < 0.1) the
  proposal shape hardly matters; unif ub 1.0's 7–10% deficit is the tail of its range (unit 0.25–0.5, where 20% of its
  draws sit on nearly resolved states). Its entropy is the highest in the sweep so far (3.98). unif 0.5 runs on the
  Ada node (30k ≈ 00:45).
- 2026-09-18 20:00 EDT — **cells 6 and 7 complete** (A5000 node, 22.7 h each): rate 5 @ 30k → **16.9** @ 3.88 (val/ppl
  1.73), rate 10 @ 30k → **17.8** @ 3.88 (val/ppl 1.80); spec 15.6. The 30k rate ladder at the default horizon is
  monotone, rate 3 / 5 / 10 / 20 → 15.6 / 16.9 / 17.8 / 19.0 (+8% / +15% / +22%): rate 5's on-and-off tie with the spec
  ended on the losing side, so the spec rate is the rate-axis winner outright (matched-horizon 30k rows running).
  unif ub 0.5 @ 10k → 17.1 @ 3.96 (5k tie is gone: trunc_exp 16.2 / exp-at-0.5 16.3 at 10k, +5%). Their 8 freed GPUs
  went to unif ub 2.0 (nice 90) and the spec's seed 2 (nice 100), started 20:00 on the A5000 node (~23 h each).
  Complete (7): spec, K −0.25, rate 20, K −1, trunc_exp 0.5, rate 5, rate 10. Running (5): unif 1.0 + rate 1.5 (A6000,
  ~24k), unif 0.5 (Ada, ~11k), unif 2.0 + spec seed 2 (A5000). Queued (3): spec seed 3, trunc_exp seeds 2–3.
- 2026-09-18 20:15 EDT, matched-horizon (t_max 0.5) 30k rows: rate 5 → 14.1 @ 3.93, rate 10 → 15.0 @ 3.93 (spec 12.9):
  +9% / +16%. Final 30k ranking of the exp cells at horizon 0.5: spec **12.9** < rate 5 14.1 < K −1 14.5 < rate 10 15.0 <
  K −0.25 15.1 < rate 20 16.4; at the default horizon: spec **15.6** < rate 5 16.9 < K −0.25 17.2 < K −1 17.4 < rate 10
  17.8 < rate 20 19.0. The spec point (rate 3, K −0.5, unit 6) wins both the rate and the curvature axis on both horizons.
- 2026-09-18 20:40 EDT, 25k reads (A6000 pair): rate 1.5 → 16.8 @ 3.87 (K −1 @ 25k 17.0; spec 16.3) — the unit-3 pair
  is now within 3% of the spec at the default horizon, i.e. the low shoulder of the rate optimum is shallower late in
  training than the 5k/10k reads suggested (the gap went +13% → +7% → +3% at 5k / 15k / 25k); unif ub 1.0 → 15.3 @ 3.91
  at horizon 1.0 (exp at horizon 1.0 ≈ 14.0 by the spec's 30k ratio: ~9% behind, unchanged). Both finish 30k ≈ 02:00.
- 2026-09-18 21:00 EDT: unif ub 0.5 @ 15k → 14.6 @ 3.94, tied with trunc_exp 0.5 (14.7) and 5% below exp at horizon 0.5
  (15.3); at 10k it was 17.1 vs 16.2 / 16.3. On the short range the two bounded modes track each other and sit at or a
  few % under exp-at-the-same-horizon; the mode axis is decided by 3-seed means, not by these single-seed reads.
- 2026-09-18 22:05 EDT: unif ub 0.5 @ 20k → 14.7 @ 3.94 (15k 14.6: flat), vs trunc_exp 13.8 and exp-at-0.5 14.2 at 20k
  (+6% / +3%). The uniform proposal on the short range stalls where the exp-shaped ones keep improving. 30k ≈ 00:45.
- 2026-09-18 22:45 EDT — **cells 8 and 9 complete** (A6000 node, 12.7 h each). rate 1.5 (unit 3) @ 30k → **16.3** @ 3.89 /
  **13.9** @ 3.93 (matched 0.5): +5% / +8% vs the spec, and 6% / 4% better than its unit-rate-3 partner K −1 (17.4 /
  14.5) — the rate/|K| equivalence holds to within ~5% over the whole run, with the flatter-curvature member (larger R)
  slightly ahead at the end. unif ub 1.0 @ 30k → **14.8** @ 3.92 at its horizon 1.0 (spec at horizon 1.0: 13.3, +11%),
  14.9 at 0.5 (spec 12.9, +15%). The freed A6000 GPUs took spec seed 3 and trunc_exp seed 2 (started 22:44, ~12.7 h);
  trunc_exp seed 3 is the last queued cell. Complete (9): spec, rate 5 / 10 / 20, K −0.25 / −1, rate 1.5, trunc_exp 0.5,
  unif 1.0. Running (5): unif 0.5 (Ada, ~25k), unif 2.0 + spec seed 2 (A5000, ~4k), spec seed 3 + trunc seed 2 (A6000).
- 2026-09-18 23:20 EDT: unif ub 0.5 @ 25k → 14.0 @ 3.94 (trunc_exp 13.3, exp-at-0.5 13.7 at 25k: +5% / +2%). Final at
  30k ≈ 00:45.
- 2026-09-19 00:15 EDT — **cell 10 complete: unif ub 0.5 @ 30k → 13.2 @ 3.93** (val/ppl 1.70, 64/64), 2% behind the spec
  at matched horizon (12.9) and trunc_exp 0.5 (12.6 / 12.9). On the short range [0, 0.5] all three proposal shapes land
  within ~5% at 30k (exp-shaped ≤ uniform), i.e. the range matters, the shape barely does. Its 4 Ada GPUs took the last
  queued cell, trunc_exp seed 3 (started 00:16, ~7 h). Stage 1 is complete except unif 2.0 (A5000, ~5.5k); stage 2:
  spec seeds 2 (A5000, ~5.5k) / 3 (A6000, ~2.5k), trunc_exp seeds 2 (A6000, ~2.5k) / 3 (Ada). ETAs: trunc seed 3 ≈ 07:30,
  spec seed 3 + trunc seed 2 ≈ 11:30, unif 2.0 + spec seed 2 ≈ 19:00 (2026-09-19).
- 2026-09-19 00:25 EDT, 5k reads: spec seed 2 → 21.8 @ 3.81 vs seed 1 @ 5k 25.1 @ 3.88 — a 13% seed-to-seed spread at
  5k (seed 2 also lower entropy), the first direct read of seed variance in this sweep; the 30k spread is what the final
  means are for. unif ub 2.0 (horizon 2.0) → 29.1 @ 3.97 vs the spec @ 5k at horizon 2.0 → 23.6 (+23%), the weakest
  bounded cell, as the calibration's ESS 0.33 / 22% resolved-state draws predicted. unif 0.5's duplicate final = 13.18
  exactly (deterministic this time).
- 2026-09-19 01:00 EDT: spec seed 3 @ 5k → 28.0 @ 3.93; the spec cell's three seeds at 5k read 25.1 / 21.8 / 28.0 (mean
  25.0, sd 3.1 → 12% CV). Early-training seed variance is therefore larger than every single-seed axis gap below ~15%
  at 5k, which is why stage-2 seeds and 30k reads decide; the 30k seed spread is the number to watch.
- 2026-09-19 01:05 EDT: trunc_exp seed 2 @ 5k → 19.9 @ 3.94 (seed 1 @ 5k 20.4): a 2% seed spread vs the spec cell's 12%
  at 5k (one pair only; the trunc_exp cells sample at horizon 0.5, where the spec's seed-1 read was 20.6).
- 2026-09-19 02:50 EDT: eval submissions stalled. Since ~01:00 every GPU in desa+thickstun is allocated (my 5 cells = 20
  GPUs, the rest by another session's ~95-job `init_3x3d_ada_prod_*` flood at nice 100), and CPU-only jobs on these
  partitions no longer start at all (the 1-CPU tick 554931 and a 2-min `hostname` probe both sit at Reason=Priority
  while nodes show 30-350 idle cores; `sprio` shows the per-partition TRES weight for gres/gpu lifts every GPU job above
  a CPU-only job, so the main scheduler blocks the partition on the un-startable GPU flood). Workaround under test: a
  tick that requests `--gres=gpu:1` (562608, nice 0, 10 min) — my nice 0 outranks the nice-100 flood the moment any GPU
  frees; the CPU tick 554931 now depends on it (afterany) so the two cannot race. Nothing else changed: trunc_exp seed 3
  passed 10k (Ada, ~0.8 s/step, 30k ~07:00), spec seed 3 + trunc seed 2 at 9k (A6000, ~1.5 s/step, ~11:15), unif 2.0 +
  spec seed 2 at 8k (A5000, ~2.9 s/step, ~20:00). No eval can run before a GPU frees anyway.
- 2026-09-19 03:15 EDT: the GPU-requesting tick (562608) ran in 20 s on desa-compute-01 and queued the 10k reads (nice 20);
  they start as GPUs free (an Ada eval takes ~2 min). trunc_exp seed 3 @ 5k → 20.56 @ 3.96: the three trunc_exp seeds at
  5k read 20.4 / 19.9 / 20.6 (mean 20.3, sd 0.4 → 2% CV) vs the spec cell's 25.1 / 21.8 / 28.0 (12% CV) at the same
  step. Same-step, same-horizon (0.5) comparison: spec seed 1 @ 5k at horizon 0.5 read 20.6, i.e. the bounded proposal
  is not better at 5k, only less seed-sensitive (all three trunc seeds sample at their own horizon 0.5).
- 2026-09-19 03:20 EDT: 10k seed reads (Ada evals take ~2 min each once a GPU frees). trunc_exp ub 0.5 seeds 1/2/3 @ 10k →
  16.15 / 15.39 / 16.38 (mean 15.97, sd 0.51, 3% CV; entropy 3.92-3.93). Spec cell seeds 1/3 @ 10k at the default horizon
  → 20.63 / 21.37 (seed 2 reaches 10k ~03:55 on the A5000); spec seed 1 at horizon 0.5 @ 10k was 16.33, so at matched
  horizon trunc_exp and exp are still within eval noise of each other at 10k (seed-3 horizon-0.5 row running).
- 2026-09-19 03:22 EDT: spec seed 3 @ 10k at horizon 0.5 → 17.42 @ 3.94 (seed 1: 16.33). Matched-horizon 10k comparison
  so far: exp (spec) 16.33 / 17.42 vs trunc_exp 0.5 16.15 / 15.39 / 16.38 → trunc_exp ~5% ahead on 2-vs-3 seeds, within
  the seed spread. The trunc seed-2 `_tmax0.5` re-read reproduced 15.39 @ 3.917 exactly, so sampling is seeded and the
  earlier 1-3% differences between duplicate reads (16.21 vs 16.34, 12.59 vs 12.92) are not sampler randomness; the
  likely cause is a re-saved checkpoint between the two reads (unverified).
- 2026-09-19 04:20 EDT: spec seed 2 @ 10k → 18.10 @ 3.84; the spec cell's three seeds at 10k (default horizon) read
  20.63 / 18.10 / 21.37 (mean 20.0, sd 1.7, 8% CV — narrowing from 12% at 5k). unif ub 2.0 @ 10k → 23.44 @ 3.93 at its
  horizon 2.0 (spec at horizon 2.0 @ 10k: 19.22; unif 1.0: 18.21; unif 0.5: 17.07 — the wider the uniform proposal, the
  worse, monotone). trunc_exp seed 3 @ 15k → 14.75 @ 3.95 (seed 1 @ 15k: 14.72; spec seed 1 @ 15k at horizon 0.5: 15.34).
- 2026-09-19 04:22 EDT: matched-horizon (0.5) 10k table complete for the seed cells: spec 16.33 / 15.85 / 17.42 (mean 16.5,
  sd 0.8) vs trunc_exp 0.5 16.15 / 15.39 / 16.38 (mean 16.0, sd 0.5) → 3.5% apart, inside one sd; the 30k reads decide.
  unif 2.0 @ 10k at horizon 0.5 → 20.01 (vs 23.44 at its own horizon 2.0): still ~20% behind exp/trunc at the same
  horizon, so the wide uniform proposal degrades the trained model, not only the sampler.
- 2026-09-19 04:52 EDT: trunc_exp seed 3 @ 20k → 14.14 @ 3.96 (seed 1 @ 20k: 13.83; spec seed 1 @ 20k at horizon 0.5:
  14.22). trunc seed 3 finishes 30k on the Ada ~07:00; the A6000 pair (spec seed 3, trunc seed 2) is at 14k, the A5000
  pair (spec seed 2, unif 2.0) at 12k. (GPU tick 569272 ran; the table block was regenerated by report.py at 04:40.)
- 2026-09-19 05:18 EDT: 15k reads: spec seed 3 → 19.64 @ 3.89 (seed 1: 18.56; default horizon); trunc_exp seed 2 → 13.98
  @ 3.91, so trunc_exp 0.5 @ 15k = 14.72 / 13.98 / 14.75 (mean 14.5, sd 0.4, 3% CV) vs spec seed 1 at horizon 0.5 @ 15k
  15.34 and unif 0.5 @ 15k 14.62. Seed spread of the bounded proposal stays ~3% at every step so far (5k/10k/15k).
- 2026-09-19 05:52 EDT: trunc_exp seed 3 @ 25k → 13.65 @ 3.95 (seed 1 @ 25k: 13.28). Its 30k finish (Ada) is due ~07:00;
  the training job then runs the final eval itself (`eval/`).
- 2026-09-19 06:55 EDT: 11/15 cells done. trunc_exp ub 0.5 seed 3 @ 30k (job 500473, 4x Ada, 6.3 h) → 13.10 @ 3.96 (seed 1:
  12.59, re-read 12.92). Two of three trunc seeds in: 12.59 / 13.10 vs the spec cell at horizon 0.5: 12.90 (seed 1). Still
  running: spec seed 3 + trunc seed 2 (A6000, 18k, ~11:20), spec seed 2 + unif 2.0 (A5000, 14k, ~21:00).
- 2026-09-19 07:47 EDT: 20k seed table complete. trunc_exp 0.5 seeds @ 20k → 13.83 / 13.97 / 14.14 (mean 13.98, sd 0.16,
  1% CV). Spec seeds 1/3 @ 20k → 16.87 / 18.42 at the default horizon, 14.22 / 15.29 at horizon 0.5 (seed 2 is at 15k:
  17.53 @ 3.84 default; seeds 1/3 @ 15k were 18.56 / 19.64). unif 2.0 @ 15k → 20.85 @ 3.94 at its horizon 2.0 (unif 1.0
  @ 15k: 16.55). Picture at 20k, matched horizon: trunc_exp 14.0 ± 0.2 vs exp 14.8 (2 seeds) → ~5% in favour of the
  bounded proposal, now larger than its own seed spread; the 30k reads (A6000 pair ~11:20, A5000 pair ~21:00) decide.
- 2026-09-19 09:52 EDT: 25k reads: trunc_exp seed 2 → 12.68 @ 3.92 (seeds 1/3 @ 25k: 13.28 / 13.65 → 3-seed mean 13.2, sd
  0.5); spec seed 3 → 17.48 @ 3.92 at the default horizon (seed 1: 16.34). The A6000 pair reaches 30k ~11:30 and then runs
  its own final eval; the A5000 pair is at ~19k (~21:00).
- 2026-09-19 11:15 EDT: 20k reads for the A5000 pair: spec seed 2 → 16.66 @ 3.84 (spec seeds @ 20k, default horizon:
  16.87 / 16.66 / 18.42, mean 17.3, sd 1.0); unif 2.0 → 18.93 @ 3.91 at its horizon 2.0 (spec @ 20k at horizon 3.0 = its
  default: 16.87; unif 1.0 @ 20k: 15.83, unif 0.5 @ 20k: 14.71). Matched-horizon (0.5) rows for both are running.
- 2026-09-19 11:45 EDT: 13/15 done (A6000 pair finished, 12.3 h each). trunc_exp ub 0.5 seed 2 @ 30k → 12.25 @ 3.92, so
  the bounded cell's three seeds read 12.59 / 12.25 / 13.10 (mean 12.65, sd 0.43). Spec seed 3 @ 30k → 16.25 @ 3.91 at
  the default horizon and 13.90 @ 3.92 at horizon 0.5 (seed 1: 15.56 / 12.90). Matched-horizon (0.5) 20k table is now
  3 vs 3: spec 14.22 / 13.58 / 15.29 (mean 14.4, sd 0.9) vs trunc_exp 13.83 / 13.97 / 14.14 (mean 14.0, sd 0.2).
  unif 2.0 @ 20k at horizon 0.5 → 17.31 (18.93 at its own horizon 2.0). Remaining: spec seed 2 + unif 2.0 (A5000, 21k,
  ~21:00), then the final analysis.
- 2026-09-19 15:07 EDT: 25k reads for the A5000 pair: spec seed 2 → 15.74 @ 3.83 (spec seeds @ 25k, default horizon:
  16.34 / 15.74 / 17.48); unif 2.0 → 18.09 @ 3.90 at its horizon 2.0. Both cells are at 25k+ and finish 30k ~19:00;
  after their finals the sweep is complete and the analysis follows.
- 2026-09-19 18:55 EDT: SWEEP COMPLETE — 15/15 cells have `eval/samples_genppl.json`. Last finals (A5000, 22.8 h each): spec
  seed 2 @ 30k → 14.91 @ 3.82 (default horizon) / 12.95 @ 3.88 (horizon 0.5); unif 2.0 @ 30k → 17.18 @ 3.90 at its
  horizon 2.0 / 15.80 @ 3.93 at horizon 0.5. Spec cell, 3 seeds @ 30k: 15.56 / 14.91 / 16.25 (15.57 ± 0.67) default,
  12.90 / 12.95 / 13.90 (13.25 ± 0.56) at horizon 0.5; trunc_exp 0.5, 3 seeds: 12.59 / 12.25 / 13.10 (12.65 ± 0.43).
  Final tick 584155 regenerates the table block and writes `logs/final_tables.md`; analysis follows below.
- 2026-09-19 19:40 EDT: Analysis section written (above the previous-round table) after a 3-lens adversarial verification
  of the draft against the raw JSONs (all headline numbers confirmed; 9 wording corrections applied: read count 145/133,
  rate-ladder ordering, horizon "U" overstated, horizon-vs-truncation decomposition, node attribution of duplicate reads).
  Correction to the 03:22 entry: the 1–3% differences between duplicate reads are node / GPU-type dependent (same node →
  bit-identical, different node → 0.01–0.46), not re-saved checkpoints. EXPERIMENT.md has the stage-2 record, measured
  throughput and outcome. Nothing is committed (commits need the user). Sweep closed.
- 2026-09-20 11:15 EDT: stage 3, step 1 (eval-only, no new training). The analysis left the bounded ladders open on the
  low side: `trunc_exp` ub 0.5 and `unif` ub 0.5 are each the best AND lowest point of their ladder, and no read below
  their bound exists (`t_max > ub` is rejected at construction, but `t_max < ub` is allowed and was never taken).
  Submitted tick 590414: `--eval-step 30000 --t-max {0.25, 0.35} --only mode-trunc_exp` (3 seeds each) and `--t-max 0.25`
  on unif ub 0.5. All pinned to thickstun-compute-01 via the new `EVAL_NODELIST` knob in `sweep.py`, because re-reads of
  one checkpoint are bit-identical on the same node but differ by up to 0.46 GenPPL across GPU types — a horizon curve
  has to be read on one node. Ada-node baselines at t_max 0.5 for the comparison: trunc seeds 12.59 / 12.03 / 13.10
  (mean 12.57), unif 0.5 13.18. Cost ~15 GPU-minutes; the result decides whether ub 0.25 training cells are worth
  GPU-days (the spec cell's own curve is flat over 0.25–1.0, so a shorter bound may buy nothing).
- 2026-09-20 11:25 EDT: stage-3 horizon result (tick 591055; the first attempt 590414 failed because a `nodelist` must
  be paired with a partition that contains the node — `sweep.py` now takes `EVAL_PARTITION` too). All reads on
  thickstun-compute-01, 30k checkpoints. trunc_exp ub 0.5, seeds 1/2/3 by sampler horizon: t_max 0.25 → 13.10 / 12.86 /
  13.48 (13.15 ± 0.32), t_max 0.35 → 12.54 / 12.40 / 13.38 (12.77 ± 0.53), t_max 0.5 = ub → 12.59 / 12.03 / 13.10
  (12.57 ± 0.53). Shorter than the bound is worse in 3/3 seeds at 0.25 (+4.6%) and in 2/3 at 0.35 (+1.6%), and unif
  ub 0.5 agrees (13.26 at 0.25 vs 13.18 at 0.5). **So ub 0.5 is a real optimum, not an edge artifact: the bounded
  ladders do not continue downward, and `ub 0.25` training cells are not worth GPU-days.** This also matches the spec
  cell's own curve (0.25 13.45 > 0.5 12.90). The trained model wants the bridge to start at unit tau 0.25, and cutting
  it shorter costs more than the proposal's late-draw mass buys.
- 2026-09-20 11:25 EDT: stage 3, step 2 (tick 591611): submitted `unif` ub 0.5 seeds 2–3 (nice 10, 4 A5000 GPUs each,
  ~23 h, done ~10:30 Sep 21). This is the only remaining cell pair that can change the answer: rank 1 (trunc_exp 0.5,
  12.64 ± 0.43 on 3 seeds) leads rank 2 (unif 0.5, 13.18 on ONE seed) by 4%, inside the seed spread. Everything else on
  the four specified axes is settled. No other cells are planned.
- 2026-09-20 11:56 EDT (CORRECTED 12:40): both confirmation cells (591612/591613) are training on kuleshov-compute-03 at
  6.05 micro-batch/s. An optimizer step is 16 micro-batches per GPU (global batch 512 = 4 GPUs x micro-batch 8 x accum 16),
  so that is 2.65 s/step — the same as the ~2.7 s/step the A5000 gave during the main sweep, NOT a speed-up. My first
  reading of this line divided by 8 instead of 16 and claimed ~1.3 s/step and a finish '~23:00 tonight'; both were wrong.
  Correct ETA for 30k + final eval: ~09:40 Sep 21. A 3-hourly babysit tick drives the reads from here.
- 2026-09-20 12:10 EDT: **spec change by the user** — `setup.md` now reads `forward_type: {naive, horosphere}` and its TASK
  line adds `forward_type` to the arguments to vary ("Vary forward_type, time_exp_rate, global curvature,
  time_conversion_mode, and time_range_upper_bound"). Everything analysed above is the `naive` half of that grid. A
  horosphere stage is being scoped before anything is submitted: the paused round 1 used horosphere but at 32 factors
  (embed_dim 96) and a different LR grid, so its static table below is NOT a drop-in comparison, and the horosphere
  readout is several times slower per step than naive (EXPERIMENT.md → Compute). Known blocker to fix first:
  `sweep.py`'s `tag_of()` has no forward-type component, so a horosphere cell would collide with the naive cell of the
  same (mode, rate, K, ub, seed) — same output dir, same `last.ckpt`, same eval dirs.
- 2026-09-20 12:40 EDT: **stage 4 submitted — `forward_type = horosphere`.** Scoped first with a 4-lens read-only
  investigation (algo semantics / plumbing / cost / what round 1 actually showed); see EXPERIMENT.md → Stage 4. The
  tooling had to change before anything could run: cells are now `(mode, rate, k, ub, seed, forward_type)` and
  `tag_of()` appends `_fwd-horo` **only** for non-naive cells. Without that a horosphere cell reuses the naive cell's
  directory — 15 of them would have been silently skipped as already-done ("submitted 0, skipped N", looking like a
  clean idempotent no-op) and the two in-flight unif cells would have resumed naive weights under a horosphere readout
  and overwritten the published results, with nothing raising. `report.py` / `final_tables.py` now carry an optional
  `_fwd-` group and a `fwd` column. Regression-checked on a compute node (scratchpad `verify_fwd.sh`): all 17 existing
  tags still resolve to their directories, the 3 new tags collide with nothing, `FORWARD_TYPE=horosphere` reaches both
  the training and the sampling command, and the regenerated report differs from the published block only by the new
  column and by today's horizon rows. Cells (seed 1, nice 20): spec point (594554, RUNNING on the Ada), the naive
  winner `trunc_exp` ub 0.5 (594555), and rate 10 (594556). Expected ~16.5 h on the Ada / ~27.5 h A6000 / ~38.5 h A5000
  per cell, ~2.5x naive. Note there is NO existing horosphere baseline to compare against: round 1 was 32 factors,
  5k–11k steps, sampled at physical t_max 2.0, with no `time_conversion_mode` knob.
- 2026-09-20 12:50 EDT: horosphere throughput MEASURED (steady state over a 300 s window, not the startup estimate):
  594554 Ada 4.24 micro-batch/s → 3.77 s/step → 31.4 h per 30k; 594555 A6000 2.34 → 6.83 s/step → 56.9 h; 594556 A6000
  2.68 → 5.96 s/step → 49.7 h (594555/594556 share kuleshov-compute-02, 4 GPUs each). That is ~4.5x the naive baseline
  (0.79 Ada / 1.52 A6000 s/step), not the predicted 2.5x: the scoping estimate assumed the chunked readout halves when
  the factor count goes 32 → 16, and it does not. All three cells train with finite loss; the tracebacks in their logs
  are the benign `pymp-*` 'Device or resource busy' cleanup. ETAs: Ada ~20:00 Sep 21, A6000 pair ~14:30 and ~21:40
  Sep 22. First signal much earlier — keep-5000 lands ~18:00 today (Ada) and ~22:15 today (A6000). **Decision point at
  the 5k/10k reads:** if horosphere is clearly behind naive at matched sampler horizon, the right move is to stop the
  cells rather than spend ~100 more GPU-hours, which needs the user (I do not cancel jobs).
- 2026-09-20 14:55 EDT: **correction to the 12:50 throughput entry — it was wrong, and the original prediction was
  right.** Steps are micro-batches / accum, and accum depends on the node's memory: the 48 GB nodes (Ada, A6000) run
  micro-batch 16 → accum 8 (29142 micro-batches/epoch), the 24 GB A5000 runs micro-batch 8 → accum 16 (58283/epoch). I
  divided every cell by 16, which doubled the horosphere step times. Correct figures: 594554 Ada 4.24 micro-batch/s ÷ 8
  = **1.89 s/step, 15.7 h** per 30k; 594555 A6000 3.42 s/step, 28.5 h; 594556 A6000 2.99 s/step, 24.9 h. Independent
  cross-check from the checkpoint timeline: 594554 wrote step 4000 at 14:45 after starting at 12:39 → 2 h 06 m / 4000 =
  1.89 s/step, exact agreement. So horosphere is **2.4x naive** (1.89 vs 0.79 on the Ada), the scoping estimate of 2.5x,
  and the chunked readout does roughly halve with the factor count after all. Revised ETAs: 594554 ~04:20 Sep 21,
  594556 ~13:35 Sep 21, 594555 ~17:10 Sep 21. keep-5000 reads: ~15:20 today (Ada), ~16:50 and ~17:20 (A6000).
  (The 2026-09-20 11:56 unif entry used accum 16 on an A5000, which is correct and stands.)
- 2026-09-20 15:15 EDT: unif ub 0.5 now has 3 seeds at 5k → 20.66 / 19.30 / 20.70 (20.22 ± 0.79 @ 3.97), statistically
  indistinguishable from trunc_exp ub 0.5 at the same step (20.38 / 19.90 / 20.56 = 20.28 ± 0.34) and well ahead of the
  spec cell (24.95 ± 3.13, own horizon). So the rank-1-vs-2 question is genuinely open and will be settled at 30k, not
  by an early read: at 30k on one seed unif was 13.18 vs trunc_exp's 12.64 ± 0.43. Horosphere cells are at ~3-4k steps,
  no keep-5000 yet.
- 2026-09-20 15:25 EDT: **first horosphere-vs-naive read (spec point, 5k, seed 1).** horosphere 31.42 @ 4.010 at its own
  horizon and 24.18 @ 4.037 at matched horizon 0.5; naive seed 1 at the same step 25.11 @ 3.883 and 20.60 @ 3.929
  (64/64 distinct everywhere). So horosphere is 25% behind at its own horizon and 17% behind at matched horizon — but
  at a consistently HIGHER entropy (+0.11 nat), so this is not a pure loss: the two sit at different points of the
  quality/diversity trade-off, and GenPPL alone across an entropy gap that size is not a clean comparison. Caveats
  before reading anything into it: one seed each, and the naive spec cell's 3-seed spread at 5k is ±3.13 (12%), so only
  the own-horizon gap clears it (31.42 vs 24.95 ± 3.13). Horosphere also responds to the sampler horizon the same way
  naive does (−23% from 3.0 to 0.5), which is reassuring for the mechanics. NOT stopping the cells on this: 5k is 1/6 of
  training, the objective differs (not just a hyperparameter), and the 10k read is much more informative — the Ada cell
  reaches it ~20:30 today. Decision deferred to the 10k reads.
- 2026-09-20 17:05 EDT: second horosphere 5k read (rate 10, seed 1): 43.80 @ 4.015 own horizon / 30.87 @ 4.006 at 0.5,
  vs naive rate 10 at 5k 31.72 @ 3.897 own — 38% behind. More informative than the gap itself: the rate-10/rate-3 ratio
  at own horizon is 43.80/31.42 = **1.39 for horosphere vs 31.72/25.11 = 1.26 for naive**, so carrying the geometry in
  the readout makes the model MORE sensitive to the proposal rate, not less. That is the opposite of the stage-4
  hypothesis (the readout is the exact Bayes posterior, so the residual should have needed the proposal less) and it
  weakens the case for tuning the rate axis under horosphere. Entropy is consistently ~4.01 for horosphere vs ~3.89
  naive at both rates. Two independent points now favour naive at 5k; the 10k reads (~20:30 spec point) decide.
- 2026-09-20 17:32 EDT: third and cleanest horosphere 5k read — `trunc_exp` ub 0.5, the naive winner's config, where
  BOTH forward types sample at horizon 0.5 = ub, so the comparison is matched by construction with no horizon confound:
  horosphere 22.64 @ 3.998 (seed 1) vs naive 20.38 / 19.90 / 20.56 = 20.28 ± 0.34 @ 3.953. That is **+12%, about 7x the
  naive seed sd**, at an entropy difference of only +0.045 nat (vs +0.11 at the spec point), so the quality/diversity
  caveat is weak for this pair: horosphere is simply behind. All three horosphere cells now lose at 5k — spec point
  +25% own / +17% matched, rate 10 +38% own, trunc_exp +12% matched. Consistent direction, three configurations.
  Still holding the stop decision for the 10k reads (spec point reaches 10k ~18:10): 5k is 1/6 of training and this is a
  different objective, not a hyperparameter, which is exactly where the early-read heuristic from stages 1-2 is weakest
  — horosphere's logits are a residual on a strong geometric prior and could plausibly converge later.
- 2026-09-20 18:02 EDT: **10k read overturns the 5k picture — do NOT stop the horosphere cells.** Spec point, horosphere
  seed 1 vs the naive 3-seed mean: at 5k +26% own horizon / +17% matched 0.5; at 10k **+11% own / +3% matched**
  (17.00 @ 3.964 vs 16.53 ± 0.80 @ 3.925). A 0.47 gap against a naive seed sd of 0.80 is well inside the noise, and
  horosphere is still at the higher entropy. So the ranking at 5k was a transient: horosphere converges more slowly
  early — exactly what its parameterisation predicts, since its logits are a residual on a strong geometric prior and
  have to unlearn less rather than learn more — and has closed 17% → 3% in one checkpoint interval. This is the failure
  mode I flagged when declining to stop on the 5k evidence, and it is why the early-read heuristic from stages 1-2
  (which compared hyperparameters within ONE objective) does not transfer to a change of objective. Revised plan: let
  all three cells run to 30k. If the trend continues, horosphere ties or beats naive at matched horizon; the 15k read
  (~20:30 on the Ada cell) tells us whether the gap keeps closing or flattens out.
- 2026-09-20 20:40 EDT: spec-point 15k read — the gap keeps closing, monotonically, at both horizons.
  Matched horizon 0.5: 5k +17.4% → 10k +2.9% → **15k +0.6%** (horo 15.43 @ 3.953 vs naive 15.34; n=1 naive at 15k,
  since only seed 1 has a `_tmax0.5` row there, so treat the last point as 1-vs-1). Own horizon 3.0: +25.9% → +11.5% →
  +7.6% (horo 19.99 @ 3.935 vs naive 18.58, n=3). Two readings: (i) horosphere is now level with naive at matched
  horizon while holding ~0.02-0.03 nat more entropy, and the trend has not flattened; (ii) the own-horizon gap stays
  much larger than the matched one, i.e. horosphere gains MORE from the short sampler horizon than naive does, which
  fits its higher entropy — it puts more mass on late, less-resolved bridge states. Letting all three cells finish is
  clearly right; on this trajectory the 30k matched-horizon comparison is a tie or a small horosphere win. To make the
  30k comparison 3-vs-3 rather than 1-vs-3 I will also need `--eval-step 15000 --t-max 0.5` on naive seeds 2-3.
- 2026-09-20 21:22 EDT: the closing trend replicates in a second cell. Rate 10, horosphere vs naive (seed 1 each):
  own horizon 5k +38.1% → 10k +16.2%; matched horizon 0.5 at 10k +6.9% (horo 22.01 @ 3.979 vs naive 20.59 @ 3.938;
  the naive rate-10 cell has no `_tmax0.5` row at 5k, so that point is missing by construction). Same shape as the spec
  point, but further behind at the same step (+6.9% vs +2.9% matched at 10k) — consistent with the earlier finding that
  horosphere is MORE sensitive to the proposal rate than naive, so rate 10 is a worse config under horosphere than it
  already is under naive. Nothing here changes the plan: all three cells run to 30k.
- 2026-09-20 21:50 EDT: **stage-5 decision point reached — all three horosphere cells now have 5k AND 10k reads.**
  Matched-horizon gap (horosphere seed 1 vs the naive mean at the same step and horizon):

  | cell | 5k | 10k | 15k |
  |---|---:|---:|---:|
  | exp rate 3 (spec) | +17.4% | +2.9% | +0.6% |
  | trunc_exp ub 0.5 (matched by construction) | +11.7% | +2.4% | – |
  | exp rate 10 | – | +6.9% | – |

  trunc_exp @ 10k: horo 16.35 @ 3.954 vs naive 15.97 ± 0.52 @ 3.923 — inside one naive seed sd. The stage-5 rule was
  "add seeds 2-3 and probe the axes if horosphere WINS by more than the ~4% seed spread; stop if it clearly LOSES".
  **Neither holds: horosphere is converging to parity, not winning and not losing**, so the decision is to spend nothing
  further now and let the three cells finish — the 30k reads are the cheapest way to break the tie, and they are already
  paid for. No new cells. Re-decide stage 5 on the 30k numbers: a horosphere win there justifies seeds 2-3 of the best
  horosphere cell; parity means the answer stays the naive `trunc_exp` config, since horosphere costs 2.4x the compute
  for the same quality; a loss closes the axis. Note for whoever reads this later: horosphere's consistent edge is
  ~0.03 nat MORE entropy at equal GenPPL, so if diversity is valued it is not a neutral tie.
- 2026-09-20 23:12 EDT: spec-point trend extended to 20k, now with n=3 naive matched rows at every step (the fill tick
  613778 added the missing `_tmax0.5` rows at 5k/15k/25k for seeds 2-3). Matched horizon 0.5, horosphere seed 1 vs the
  naive 3-seed mean: 5k +14.8% (naive 21.06 ± 1.45) → 10k +2.9% (16.53 ± 0.80) → 15k +1.2% (15.24 ± 0.62) → **20k +1.0%
  (horo 14.51 @ 3.962 vs naive 14.36 ± 0.86)**. The curve has flattened at ~1%, i.e. parity: the gap is now a seventh of
  the naive seed sd. Horosphere holds ~0.03 nat more entropy throughout. Unless 30k breaks the pattern, the stage-4
  answer is "horosphere matches naive at matched sampler horizon, at 2.4x the compute and slightly higher diversity",
  which means the recommended config stays the naive one.
- 2026-09-21 01:35 EDT: rate-10 horosphere at 15k, own horizon (the naive rate-10 cell has no `_tmax0.5` rows at 5k/15k,
  so matched is unavailable there): 5k +38.1% → 10k +16.2% → 15k +12.0% (horo 24.64 @ 3.956 vs naive 21.99 @ 3.918).
  Converging like the other two but from much further back and flattening at a much larger gap — at the spec rate the
  own-horizon gap was already +7.6% by 15k. This is the third confirmation that horosphere pays a bigger penalty for a
  mis-set proposal rate than naive does, so under horosphere the rate axis matters MORE, not less. Both `unif`
  confirmation cells also passed 15k and their reads are queued.
- 2026-09-21 02:12 EDT: **crossover at the naive winner's config.** `trunc_exp` ub 0.5, matched by construction,
  horosphere seed 1 vs the naive 3 seeds: 5k +11.7% → 10k +2.4% → **15k −2.5%** (horo 14.13 @ 3.940 vs naive
  14.48 ± 0.43 @ 3.930). Read it precisely before celebrating: horosphere is now BELOW the naive mean but still INSIDE
  the naive seed range (naive seeds at 15k were 14.72 / 13.98 / 14.75, so horo's 14.13 sits between seed 2 and seeds
  1/3). That is parity leaning horosphere on one seed, not a win — but the trajectory is still improving while the spec
  cell's has flattened, and this is the config that matters, since it is the one that produced the current answer.
  If the 30k read of this cell comes in below naive's 12.64 ± 0.43, stage 5 is justified: seeds 2-3 of
  `trunc_exp ub 0.5 fwd-horo` to settle it, which is ~57 h of A6000 time for the pair. This cell finishes ~17:10 today.
- 2026-09-21 04:20 EDT: **first complete horosphere-vs-naive head-to-head at 30k (spec point, exp rate 3, K −0.5).**

  | sampler horizon | horosphere seed 1 | naive 3 seeds | gap |
  |---|---:|---:|---:|
  | own (3.0) | 16.43 @ 3.902 | 15.58 ± 0.67 @ 3.859 | +5.5% |
  | matched (0.5) | **12.81 @ 3.958** | 13.25 ± 0.56 @ 3.908 | **−3.3%** |

  Matched-horizon convergence, complete: 5k +14.8% → 10k +2.9% → 15k +1.2% → 20k +1.0% → 25k +0.1% → 30k −3.3%.
  Monotone the whole way, crossing over at the end. At the matched horizon horosphere is better on BOTH metrics — lower
  GenPPL *and* +0.05 nat entropy, 64/64 distinct — so this is not a quality/diversity trade. Two caveats: it is one
  horosphere seed against three naive seeds, and −3.3% is inside the naive seed sd (±0.56 = 4.2%), so call it a tie
  leaning horosphere rather than a win. Also note horosphere gains far more from the short horizon than naive does
  (−22% from 3.0 → 0.5 vs naive's −15%), which is the same asymmetry seen at every earlier checkpoint.
  **Context for the ranking:** 12.81 puts horosphere-at-the-spec-config level with the best NAIVE cell in the whole
  sweep (`trunc_exp` ub 0.5, 12.64 ± 0.43) — i.e. the geometric readout buys at the spec point what the bounded
  proposal buys under naive. And the horosphere run of that same `trunc_exp` config is still training (at 16k it was
  already −2.5% vs its naive counterpart); it finishes ~17:10 today and is the cell that could take the top spot
  outright. Stage 5 (seeds 2-3) is now likely justified — deciding after that cell's 30k read.
- 2026-09-21 06:30 EDT: `trunc_exp` ub 0.5 horosphere at 20k — the lead is now outside the noise. Curve (matched by
  construction, horosphere seed 1 vs naive 3 seeds): 5k +11.7% → 10k +2.4% → 15k −2.5% → **20k −4.4%** (horo 13.37 @
  3.939 vs naive 13.98 ± 0.15). The naive sd at 20k is only 0.15, so −4.4% is ~4 sd below the naive mean — this is no
  longer a tie. Extrapolating the last two intervals puts its 30k read near 12.0–12.3 against the best naive cell's
  12.64 ± 0.43, i.e. a probable new best overall. Considered submitting horosphere seeds 2-3 now to save ~11 h of
  wall-clock, and decided against it on two grounds: the stated rule was to decide on the 30k read, and every GPU in
  both partitions is currently allocated (Ada 8/8, A6000 10/10, A5000 10/10), so an early submission would only queue
  and save nothing. Decision stands: wait for this cell's 30k read (~17:10 today), then submit seeds 2-3 if it holds.
- 2026-09-21 10:08 EDT: rate-10 horosphere at 25k, own horizon: 5k +38.1% → 10k +16.2% → 15k +12.0% → 20k +3.4% →
  25k +11.2% (horo 20.58 @ 3.945 vs naive 18.51). Both series are single seeds, so the 20k point is checkpoint noise,
  not a crossing — each series on its own is monotone (horo 24.64 → 22.15 → 20.58; naive 21.99 → 21.41 → 18.51).
  The honest summary of this cell: unlike the spec point (which converged to parity and then crossed at 30k), rate 10
  stays ~11-12% behind and is NOT closing. Third consistent sign that a mis-set proposal rate costs horosphere more
  than it costs naive, so the rate axis matters more under horosphere, not less.
- 2026-09-21 10:23 EDT: unif ub 0.5 seed 3 finished — 13.55 @ 3.953 (64/64). With seed 1's 13.18 that is two of three
  in, and **the naive rank-1 question is now effectively settled in favour of `trunc_exp`**: for unif's 3-seed mean to
  reach trunc_exp's 12.64, the outstanding seed 2 would have to read 11.19, which is far below the best single naive
  number anywhere in the sweep (12.25). So rank 1 stays `trunc_exp` ub 0.5 = 12.64 ± 0.43 and rank 2 is unif ub 0.5.
  Seed 2 is ~20 min out and will fix the exact mean.
- 2026-09-21 10:28 EDT: **naive stage 3 complete — rank 1 vs rank 2 settled.** unif ub 0.5 3-seed final:
  13.18 / 12.84 / 13.55 = 13.19 ± 0.36 @ 3.936 (64/64 distinct) vs trunc_exp ub 0.5's 12.64 ± 0.43 @ 3.934. trunc_exp
  keeps rank 1 by 4.3%; 8 of 9 pairwise seed comparisons favour it, but on 3-vs-3 seeds it is not statistically decisive
  (Welch t ≈ 1.7, p ≈ 0.17). Analysis ranking table and open-questions list updated; the naive answer is unchanged.
  Remaining: the two horosphere cells (rate 10 ~13:35, trunc_exp ~17:10 today).
- 2026-09-21 10:50 EDT: `trunc_exp` horosphere at 25k — the lead NARROWED, so temper the 06:30 extrapolation. Curve
  (matched by construction): 5k +11.7% → 10k +2.4% → 15k −2.5% → 20k −4.4% → **25k −1.6%** (horo 12.99 @ 3.928 vs naive
  13.20 ± 0.49). The −4.4% at 20k looked like ~4 sd only because the naive sd happened to be 0.15 at that step; at 25k
  the naive sd is back to 0.49 and the gap is 1.6%. Correct summary: from 15k on, horosphere runs 1.6–4.4% ahead of
  naive at this config, i.e. **at or just above parity on one seed, not a decisive lead**, and my 06:30 note predicting
  a 30k read of 12.0–12.3 was over-confident on a single noisy interval. Projecting the last two intervals now gives
  ~12.4–12.6 against naive's 12.64 ± 0.43 — a tie or marginal win. The cell finishes ~17:10; the decision on seeds 2-3
  rests on that read, and if it is a tie the honest call is that horosphere does not justify 2.4x the compute.
- 2026-09-21 14:28 EDT: rate-10 horosphere finished 30k — **and it corrects two things I wrote earlier.**
  own horizon 3.0: horo 19.08 @ 3.930 vs naive 17.82 @ 3.881 = +7.0%; matched horizon 0.5: horo 14.87 @ 3.936 vs
  naive 15.01 @ 3.930 = **−0.9%**, i.e. parity.
  (1) My 10:08 and 14:00-ish notes said rate 10 "stays ~11-12% behind and is NOT closing". That was true of the
  OWN-horizon reads only; at the matched horizon it converges to parity exactly like the spec point. Both exp cells now
  land at matched-horizon parity at 30k (spec −3.3%, rate 10 −0.9%) while staying 5–7% behind at their own horizon.
  (2) The "horosphere is much more sensitive to the proposal rate" claim was over-stated. Measured at 30k, matched
  horizon, rate 10 vs rate 3: horosphere 14.87/12.81 = 1.16x, naive 15.01/13.25 = 1.13x. The effect is real but small
  (3 percentage points), not the 1.39-vs-1.26 gap the 5k own-horizon reads suggested — early own-horizon reads
  exaggerated it on both axes.
  The stable finding across all three cells: horosphere is worse than naive at long sampler horizons and equal or
  better at short ones, consistent with it placing more mass on late, less-resolved bridge states.
- 2026-09-21 15:12 EDT: **STAGE 4 COMPLETE — all three horosphere cells finished 30k. The answer is "tie", and the
  stage-5 rule says stop.** At matched sampler horizon, horosphere seed 1 vs the naive mean at the same config:

  | config | horosphere | naive | gap |
  |---|---:|---:|---:|
  | exp rate 3 (spec) | 12.81 @ 3.958 | 13.25 ± 0.56 @ 3.908 | −3.3% |
  | trunc_exp ub 0.5 | 12.75 @ 3.948 | 12.64 ± 0.43 @ 3.934 | +0.8% |
  | exp rate 10 | 14.87 @ 3.936 | 15.01 @ 3.930 | −0.9% |

  Every gap is inside the naive seed spread, and the decisive cell (trunc_exp, where the 25k read had horosphere 1.6%
  ahead) came in at +0.8% — inside the naive seed range 12.25–13.10. So the 15k–25k "crossover" was seed-and-checkpoint
  noise on a single horosphere seed, not a durable lead; my 02:12 and 06:30 notes read more into it than the data
  supported. **Conclusion: at 16 factors on TinyStories-256, forward_type=horosphere matches forward_type=naive at
  matched sampler horizon and loses by 5–7% at the default long horizon, for 2.4x the compute.** The recommended config
  is unchanged: `naive`, `trunc_exp`, rate 3, K −0.5, ub 0.5, sampled at t_max 0.5 → 12.64 ± 0.43 @ 3.934.
  Per the stage-5 rule ("expand only if horosphere beats naive by more than the ~4% seed spread"), no seeds 2-3 and no
  axis probe: that would be ~57 h of A6000 time to resolve a tie. The one consistent non-tie: horosphere holds
  +0.01–0.05 nat more entropy at equal GenPPL in all three cells, so if sample diversity is valued it is a marginal
  win, not a neutral one.
