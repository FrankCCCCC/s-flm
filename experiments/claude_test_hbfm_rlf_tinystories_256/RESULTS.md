# claude_test_hbfm_rlf_tinystories_256 — Results

HBFM on TinyStories seq 256 with random Laplacian features (HyLa) of the input state vs the plain coordinates (see `EXPERIMENT.md`). Regenerate with `python report.py`. `±` = sample std over seeds (ddof=1). val/ppl is the denoising-CE bound, not a likelihood.

<!-- report:begin -->
165 evaluated (cell, step, horizon) triples.

### Per read

| read | fwd | t_max | step | arm | seed | val/ppl | GenPPL | entropy | distinct |
|---|---|---|---|---|---|---:|---:|---:|---:|
| own | naive | 0.25 | 30000 | none | 1 | 1.715 | 13.45 | 3.920 | 64/64 |
| own | naive | 0.5 | 30000 | none | 1 | 1.715 | 12.90 | 3.928 | 64/64 |
| own | naive | 0.5 | 30000 | none | 2 | 1.713 | 12.95 | 3.876 | 64/64 |
| own | naive | 0.5 | 30000 | none | 3 | 1.715 | 13.90 | 3.920 | 64/64 |
| own | naive | 0.5 | 30000 | cap5 | 1 | 1.757 | 15.46 | 4.172 | 64/64 |
| own | naive | 0.5 | 30000 | cap5 | 2 | 1.755 | 14.58 | 4.181 | 64/64 |
| own | naive | 0.5 | 30000 | cap5 | 3 | 1.759 | 15.67 | 4.201 | 64/64 |
| own | naive | 0.5 | 30000 | best | 1 | 1.744 | 14.73 | 4.194 | 64/64 |
| own | naive | 0.5 | 30000 | best | 2 | 1.736 | 13.93 | 4.181 | 64/64 |
| own | naive | 0.5 | 30000 | best | 3 | 1.739 | 14.25 | 4.245 | 64/64 |
| own | naive | 0.5 | 30000 | sphere-L12 | 1 | 1.724 | 13.46 | 3.950 | 64/64 |
| own | naive | 0.5 | 30000 | sphere-L4 | 1 | 1.717 | 13.47 | 3.930 | 64/64 |
| own | naive | 0.5 | 30000 | sphere-L4 | 2 | 1.721 | 13.22 | 3.941 | 64/64 |
| own | naive | 0.5 | 30000 | sphere-L4 | 3 | 1.718 | 13.90 | 3.948 | 64/64 |
| own | naive | 0.5 | 30000 | sphere-L4-lr5e-4 | 1 | 1.710 | 13.30 | 3.940 | 64/64 |
| own | naive | 0.5 | 30000 | sphere-L8 | 1 | 1.720 | 13.21 | 3.930 | 64/64 |
| own | naive | 0.5 | 25000 | none | 1 | 1.731 | 13.69 | 3.908 | 64/64 |
| own | naive | 0.5 | 25000 | none | 2 | 1.729 | 13.34 | 3.892 | 64/64 |
| own | naive | 0.5 | 25000 | none | 3 | 1.732 | 14.59 | 3.935 | 64/64 |
| own | naive | 0.5 | 20000 | none | 1 | 1.752 | 14.22 | 3.926 | 64/64 |
| own | naive | 0.5 | 20000 | none | 2 | 1.751 | 13.58 | 3.887 | 64/64 |
| own | naive | 0.5 | 20000 | none | 3 | 1.754 | 15.29 | 3.943 | 64/64 |
| own | naive | 0.5 | 20000 | cap5 | 3 | 1.804 | 17.16 | 4.220 | 64/64 |
| own | naive | 0.5 | 15000 | none | 1 | 1.784 | 15.34 | 3.932 | 64/64 |
| own | naive | 0.5 | 15000 | none | 2 | 1.782 | 14.58 | 3.902 | 64/64 |
| own | naive | 0.5 | 15000 | none | 3 | 1.787 | 15.80 | 3.935 | 64/64 |
| own | naive | 0.5 | 15000 | cap5 | 2 | 1.831 | 17.78 | 4.206 | 64/64 |
| own | naive | 0.5 | 15000 | cap5 | 3 | 1.840 | 18.24 | 4.212 | 64/64 |
| own | naive | 0.5 | 10000 | none | 1 | 1.839 | 16.33 | 3.934 | 64/64 |
| own | naive | 0.5 | 10000 | none | 2 | 1.836 | 15.85 | 3.905 | 64/64 |
| own | naive | 0.5 | 10000 | none | 3 | 1.844 | 17.42 | 3.937 | 64/64 |
| own | naive | 0.5 | 10000 | cap5 | 3 | 1.899 | 20.41 | 4.222 | 64/64 |
| own | naive | 0.5 | 10000 | sphere-L4 | 1 | 1.845 | 18.10 | 3.995 | 64/64 |
| own | naive | 0.5 | 10000 | sphere-L4 | 2 | 1.849 | 16.37 | 3.943 | 64/64 |
| own | naive | 0.5 | 10000 | sphere-L4 | 3 | 1.847 | 17.80 | 3.996 | 64/64 |
| own | naive | 0.5 | 5000 | none | 1 | 1.979 | 20.60 | 3.929 | 64/64 |
| own | naive | 0.5 | 5000 | none | 2 | 1.971 | 19.90 | 3.909 | 64/64 |
| own | naive | 0.5 | 5000 | none | 3 | 1.987 | 22.69 | 3.960 | 64/64 |
| own | naive | 0.5 | 5000 | cap5 | 1 | 2.045 | 25.97 | 4.203 | 64/64 |
| own | naive | 0.5 | 5000 | cap5 | 2 | 2.028 | 26.94 | 4.235 | 64/64 |
| own | naive | 0.5 | 5000 | cap5 | 3 | 2.048 | 25.94 | 4.198 | 64/64 |
| own | naive | 0.5 | 5000 | cap5 sc0.005 | 1 | 2.010 | 24.62 | 4.188 | 64/64 |
| own | naive | 0.5 | 5000 | cap5 sc0.005 unnorm | 1 | 2.000 | 23.58 | 4.178 | 64/64 |
| own | naive | 0.5 | 5000 | cap5 sc0.05 | 1 | 2.009 | 23.95 | 4.166 | 64/64 |
| own | naive | 0.5 | 5000 | cap5 sc0.05 unnorm | 1 | 2.010 | 24.09 | 4.192 | 64/64 |
| own | naive | 0.5 | 5000 | cap5 sc0.25 | 1 | 2.006 | 24.17 | 4.191 | 64/64 |
| own | naive | 0.5 | 5000 | cap5 sc0.25 unnorm | 1 | 2.003 | 23.47 | 4.171 | 64/64 |
| own | naive | 1 | 30000 | none | 1 | 1.715 | 13.34 | 3.890 | 64/64 |
| own | naive | 1 | 10000 | none | 1 | 1.839 | 16.54 | 3.907 | 64/64 |
| own | naive | 1 | 5000 | none | 1 | 1.979 | 21.12 | 3.915 | 64/64 |
| own | naive | 2 | 30000 | none | 1 | 1.715 | 14.63 | 3.856 | 64/64 |
| own | naive | 2 | 10000 | none | 1 | 1.839 | 19.22 | 3.893 | 64/64 |
| own | naive | 2 | 5000 | none | 1 | 1.979 | 23.58 | 3.903 | 64/64 |
| own | naive | 3 | 30000 | none | 1 | 1.715 | 15.56 | 3.849 | 64/64 |
| own | naive | 3 | 30000 | none | 2 | 1.713 | 14.91 | 3.822 | 64/64 |
| own | naive | 3 | 30000 | none | 3 | 1.715 | 16.25 | 3.906 | 64/64 |
| own | naive | 3 | 30000 | cap5 | 1 | 1.757 | 19.42 | 4.157 | 64/64 |
| own | naive | 3 | 30000 | cap5 | 2 | 1.755 | 19.47 | 4.215 | 64/64 |
| own | naive | 3 | 30000 | cap5 | 3 | 1.759 | 21.08 | 4.211 | 64/64 |
| own | naive | 3 | 30000 | best | 1 | 1.744 | 18.65 | 4.195 | 64/64 |
| own | naive | 3 | 30000 | best | 2 | 1.736 | 19.37 | 4.171 | 64/64 |
| own | naive | 3 | 30000 | best | 3 | 1.739 | 19.52 | 4.247 | 64/64 |
| own | naive | 3 | 30000 | sphere-L12 | 1 | 1.724 | 16.07 | 3.923 | 64/64 |
| own | naive | 3 | 30000 | sphere-L4 | 1 | 1.717 | 16.86 | 3.919 | 64/64 |
| own | naive | 3 | 30000 | sphere-L4 | 2 | 1.721 | 14.84 | 3.855 | 64/64 |
| own | naive | 3 | 30000 | sphere-L4 | 3 | 1.718 | 16.58 | 3.905 | 64/64 |
| own | naive | 3 | 30000 | sphere-L4-lr5e-4 | 1 | 1.710 | 15.70 | 3.905 | 64/64 |
| own | naive | 3 | 30000 | sphere-L8 | 1 | 1.720 | 16.41 | 3.876 | 64/64 |
| own | naive | 3 | 25000 | none | 1 | 1.731 | 16.34 | 3.844 | 64/64 |
| own | naive | 3 | 25000 | none | 2 | 1.729 | 15.74 | 3.825 | 64/64 |
| own | naive | 3 | 25000 | none | 3 | 1.732 | 17.48 | 3.923 | 64/64 |
| own | naive | 3 | 20000 | none | 1 | 1.752 | 16.87 | 3.867 | 64/64 |
| own | naive | 3 | 20000 | none | 2 | 1.751 | 16.66 | 3.838 | 64/64 |
| own | naive | 3 | 20000 | none | 3 | 1.754 | 18.42 | 3.917 | 64/64 |
| own | naive | 3 | 20000 | cap5 | 3 | 1.804 | 22.48 | 4.231 | 64/64 |
| own | naive | 3 | 15000 | none | 1 | 1.784 | 18.56 | 3.883 | 64/64 |
| own | naive | 3 | 15000 | none | 2 | 1.782 | 17.53 | 3.843 | 64/64 |
| own | naive | 3 | 15000 | none | 3 | 1.787 | 19.64 | 3.893 | 64/64 |
| own | naive | 3 | 15000 | cap5 | 2 | 1.831 | 22.79 | 4.222 | 64/64 |
| own | naive | 3 | 15000 | cap5 | 3 | 1.840 | 24.86 | 4.242 | 64/64 |
| own | naive | 3 | 10000 | none | 1 | 1.839 | 20.63 | 3.880 | 64/64 |
| own | naive | 3 | 10000 | none | 2 | 1.836 | 18.10 | 3.844 | 64/64 |
| own | naive | 3 | 10000 | none | 3 | 1.844 | 21.37 | 3.907 | 64/64 |
| own | naive | 3 | 10000 | cap5 | 3 | 1.899 | 27.23 | 4.246 | 64/64 |
| own | naive | 3 | 10000 | sphere-L4 | 1 | 1.845 | 21.72 | 3.998 | 64/64 |
| own | naive | 3 | 10000 | sphere-L4 | 2 | 1.849 | 19.14 | 3.896 | 64/64 |
| own | naive | 3 | 10000 | sphere-L4 | 3 | 1.847 | 21.72 | 3.938 | 64/64 |
| own | naive | 3 | 5000 | none | 1 | 1.979 | 25.11 | 3.883 | 64/64 |
| own | naive | 3 | 5000 | none | 2 | 1.971 | 21.75 | 3.811 | 64/64 |
| own | naive | 3 | 5000 | none | 3 | 1.987 | 28.00 | 3.934 | 64/64 |
| own | naive | 3 | 5000 | cap5 | 1 | 2.045 | 34.08 | 4.237 | 64/64 |
| own | naive | 3 | 5000 | cap5 | 2 | 2.028 | 32.47 | 4.231 | 64/64 |
| own | naive | 3 | 5000 | cap5 | 3 | 2.048 | 34.77 | 4.244 | 64/64 |
| own | naive | 3 | 5000 | cap5 sc0.005 | 1 | 2.010 | 32.47 | 4.221 | 64/64 |
| own | naive | 3 | 5000 | cap5 sc0.005 unnorm | 1 | 2.000 | 30.27 | 4.227 | 64/64 |
| own | naive | 3 | 5000 | cap5 sc0.05 | 1 | 2.009 | 30.81 | 4.224 | 64/64 |
| own | naive | 3 | 5000 | cap5 sc0.05 unnorm | 1 | 2.010 | 31.25 | 4.214 | 64/64 |
| own | naive | 3 | 5000 | cap5 sc0.25 | 1 | 2.006 | 31.40 | 4.244 | 64/64 |
| own | naive | 3 | 5000 | cap5 sc0.25 unnorm | 1 | 2.003 | 32.26 | 4.226 | 64/64 |
| own | horosphere | 0.5 | 30000 | none | 1 | 1.714 | 12.81 | 3.958 | 64/64 |
| own | horosphere | 0.5 | 30000 | none | 2 | 1.716 | 13.00 | 3.932 | 64/64 |
| own | horosphere | 0.5 | 30000 | none | 3 | 1.710 | 13.40 | 3.957 | 64/64 |
| own | horosphere | 0.5 | 30000 | cap5 | 1 | 1.774 | 17.28 | 4.149 | 64/64 |
| own | horosphere | 0.5 | 30000 | cap5 | 2 | 2.618 | 76.27 | 4.283 | 64/64 |
| own | horosphere | 0.5 | 30000 | cap5 | 3 | 2.744 | 74.99 | 4.114 | 64/64 |
| own | horosphere | 0.5 | 25000 | none | 1 | 1.730 | 13.89 | 3.952 | 64/64 |
| own | horosphere | 0.5 | 25000 | cap5 | 2 | 2.646 | 66.98 | 4.075 | 64/64 |
| own | horosphere | 0.5 | 25000 | cap5 | 3 | 2.813 | 87.02 | 4.077 | 64/64 |
| own | horosphere | 0.5 | 20000 | none | 1 | 1.752 | 14.51 | 3.962 | 64/64 |
| own | horosphere | 0.5 | 20000 | cap5 | 1 | 1.840 | 20.88 | 4.163 | 64/64 |
| own | horosphere | 0.5 | 20000 | cap5 | 2 | 2.673 | 72.35 | 4.048 | 64/64 |
| own | horosphere | 0.5 | 20000 | cap5 | 3 | 2.749 | 106.41 | 4.295 | 64/64 |
| own | horosphere | 0.5 | 15000 | none | 1 | 1.783 | 15.43 | 3.953 | 64/64 |
| own | horosphere | 0.5 | 15000 | cap5 | 1 | 1.902 | 24.65 | 4.145 | 64/64 |
| own | horosphere | 0.5 | 15000 | cap5 | 2 | 2.708 | 84.85 | 4.183 | 64/64 |
| own | horosphere | 0.5 | 15000 | cap5 | 3 | 2.799 | 102.57 | 4.169 | 64/64 |
| own | horosphere | 0.5 | 10000 | none | 1 | 1.837 | 17.00 | 3.964 | 64/64 |
| own | horosphere | 0.5 | 10000 | cap5 | 1 | 2.020 | 30.78 | 4.148 | 64/64 |
| own | horosphere | 0.5 | 10000 | cap5 | 2 | 2.754 | 88.31 | 4.118 | 64/64 |
| own | horosphere | 0.5 | 10000 | cap5 | 3 | 2.874 | 105.70 | 4.108 | 64/64 |
| own | horosphere | 0.5 | 5000 | none | 1 | 1.975 | 24.18 | 4.037 | 64/64 |
| own | horosphere | 0.5 | 5000 | cap5 | 1 | 2.388 | 60.26 | 4.121 | 64/64 |
| own | horosphere | 0.5 | 5000 | cap5 | 2 | 2.888 | 88.32 | 3.856 | 64/64 |
| own | horosphere | 0.5 | 5000 | cap5 | 3 | 3.076 | 125.10 | 3.997 | 64/64 |
| own | horosphere | 3 | 30000 | none | 1 | 1.714 | 16.43 | 3.902 | 64/64 |
| own | horosphere | 3 | 30000 | none | 2 | 1.716 | 16.01 | 3.898 | 64/64 |
| own | horosphere | 3 | 30000 | none | 3 | 1.710 | 17.34 | 3.917 | 64/64 |
| own | horosphere | 3 | 30000 | cap5 | 1 | 1.774 | 23.79 | 4.144 | 64/64 |
| own | horosphere | 3 | 30000 | cap5 | 2 | 2.618 | 81.05 | 4.113 | 64/64 |
| own | horosphere | 3 | 30000 | cap5 | 3 | 2.744 | 85.36 | 3.950 | 64/64 |
| own | horosphere | 3 | 25000 | none | 1 | 1.730 | 16.91 | 3.914 | 64/64 |
| own | horosphere | 3 | 20000 | none | 1 | 1.752 | 18.25 | 3.929 | 64/64 |
| own | horosphere | 3 | 20000 | cap5 | 1 | 1.840 | 29.79 | 4.143 | 64/64 |
| own | horosphere | 3 | 15000 | none | 1 | 1.783 | 19.99 | 3.935 | 64/64 |
| own | horosphere | 3 | 15000 | cap5 | 1 | 1.902 | 34.04 | 4.141 | 64/64 |
| own | horosphere | 3 | 10000 | none | 1 | 1.837 | 22.34 | 3.947 | 64/64 |
| own | horosphere | 3 | 10000 | cap5 | 1 | 2.020 | 46.76 | 4.175 | 64/64 |
| own | horosphere | 3 | 5000 | none | 1 | 1.975 | 31.42 | 4.010 | 64/64 |
| own | horosphere | 3 | 5000 | cap5 | 1 | 2.388 | 73.94 | 4.085 | 64/64 |
| paired | naive | 0.5 | 30000 | none | 1 | 1.715 | 12.82 | 3.925 | 64/64 |
| paired | naive | 0.5 | 30000 | none | 2 | 1.713 | 12.95 | 3.876 | 64/64 |
| paired | naive | 0.5 | 30000 | none | 3 | 1.715 | 13.90 | 3.920 | 64/64 |
| paired | naive | 0.5 | 30000 | cap5 | 1 | 1.757 | 15.45 | 4.175 | 64/64 |
| paired | naive | 0.5 | 30000 | cap5 | 2 | 1.755 | 14.42 | 4.184 | 64/64 |
| paired | naive | 0.5 | 30000 | cap5 | 3 | 1.759 | 15.74 | 4.207 | 64/64 |
| paired | naive | 0.5 | 30000 | sphere-L4 | 1 | 1.717 | 13.34 | 3.932 | 64/64 |
| paired | naive | 3 | 30000 | none | 1 | 1.715 | 15.56 | 3.849 | 64/64 |
| paired | naive | 3 | 30000 | none | 2 | 1.713 | 15.09 | 3.829 | 64/64 |
| paired | naive | 3 | 30000 | none | 3 | 1.715 | 16.67 | 3.921 | 64/64 |
| paired | naive | 3 | 30000 | cap5 | 1 | 1.757 | 19.75 | 4.155 | 64/64 |
| paired | naive | 3 | 30000 | cap5 | 2 | 1.755 | 19.14 | 4.212 | 64/64 |
| paired | naive | 3 | 30000 | cap5 | 3 | 1.759 | 21.13 | 4.216 | 64/64 |
| paired | naive | 3 | 30000 | sphere-L4 | 1 | 1.717 | 16.86 | 3.922 | 64/64 |
| paired | horosphere | 0.5 | 30000 | none | 1 | 1.714 | 12.81 | 3.958 | 64/64 |
| paired | horosphere | 0.5 | 30000 | none | 2 | 1.716 | 12.96 | 3.926 | 64/64 |
| paired | horosphere | 0.5 | 30000 | none | 3 | 1.710 | 13.40 | 3.957 | 64/64 |
| paired | horosphere | 0.5 | 30000 | cap5 | 1 | 1.774 | 17.28 | 4.149 | 64/64 |
| paired | horosphere | 0.5 | 30000 | cap5 | 2 | 2.618 | 77.15 | 4.288 | 64/64 |
| paired | horosphere | 0.5 | 30000 | cap5 | 3 | 2.744 | 74.99 | 4.114 | 64/64 |
| paired | horosphere | 3 | 30000 | none | 1 | 1.714 | 16.43 | 3.902 | 64/64 |
| paired | horosphere | 3 | 30000 | none | 2 | 1.716 | 16.09 | 3.902 | 64/64 |
| paired | horosphere | 3 | 30000 | none | 3 | 1.710 | 17.34 | 3.917 | 64/64 |
| paired | horosphere | 3 | 30000 | cap5 | 1 | 1.774 | 23.79 | 4.144 | 64/64 |
| paired | horosphere | 3 | 30000 | cap5 | 2 | 2.618 | 78.03 | 4.104 | 64/64 |
| paired | horosphere | 3 | 30000 | cap5 | 3 | 2.744 | 85.36 | 3.950 | 64/64 |

### Mean ± std over seeds

| read | fwd | t_max | step | arm | n | val/ppl | GenPPL | entropy |
|---|---|---|---|---|---|---:|---:|---:|
| own | naive | 0.25 | 30000 | none | 1 | 1.715 | 13.45 | 3.920 |
| own | naive | 0.5 | 30000 | none | 3 | 1.714 ± 0.001 | 13.25 ± 0.56 | 3.908 ± 0.028 |
| own | naive | 0.5 | 30000 | cap5 | 3 | 1.757 ± 0.002 | 15.24 ± 0.58 | 4.185 ± 0.015 |
| own | naive | 0.5 | 30000 | best | 3 | 1.740 ± 0.004 | 14.30 ± 0.40 | 4.207 ± 0.033 |
| own | naive | 0.5 | 30000 | sphere-L12 | 1 | 1.724 | 13.46 | 3.950 |
| own | naive | 0.5 | 30000 | sphere-L4 | 3 | 1.719 ± 0.002 | 13.53 ± 0.34 | 3.940 ± 0.009 |
| own | naive | 0.5 | 30000 | sphere-L4-lr5e-4 | 1 | 1.710 | 13.30 | 3.940 |
| own | naive | 0.5 | 30000 | sphere-L8 | 1 | 1.720 | 13.21 | 3.930 |
| own | naive | 0.5 | 25000 | none | 3 | 1.731 ± 0.002 | 13.87 ± 0.65 | 3.912 ± 0.022 |
| own | naive | 0.5 | 20000 | none | 3 | 1.752 ± 0.002 | 14.36 ± 0.86 | 3.919 ± 0.029 |
| own | naive | 0.5 | 20000 | cap5 | 1 | 1.804 | 17.16 | 4.220 |
| own | naive | 0.5 | 15000 | none | 3 | 1.785 ± 0.003 | 15.24 ± 0.62 | 3.923 ± 0.018 |
| own | naive | 0.5 | 15000 | cap5 | 2 | 1.835 ± 0.006 | 18.01 ± 0.32 | 4.209 ± 0.004 |
| own | naive | 0.5 | 10000 | none | 3 | 1.840 ± 0.004 | 16.53 ± 0.80 | 3.925 ± 0.018 |
| own | naive | 0.5 | 10000 | cap5 | 1 | 1.899 | 20.41 | 4.222 |
| own | naive | 0.5 | 10000 | sphere-L4 | 3 | 1.847 ± 0.002 | 17.42 ± 0.93 | 3.978 ± 0.030 |
| own | naive | 0.5 | 5000 | none | 3 | 1.979 ± 0.008 | 21.06 ± 1.45 | 3.933 ± 0.025 |
| own | naive | 0.5 | 5000 | cap5 | 3 | 2.041 ± 0.011 | 26.28 ± 0.57 | 4.212 ± 0.020 |
| own | naive | 0.5 | 5000 | cap5 sc0.005 | 1 | 2.010 | 24.62 | 4.188 |
| own | naive | 0.5 | 5000 | cap5 sc0.005 unnorm | 1 | 2.000 | 23.58 | 4.178 |
| own | naive | 0.5 | 5000 | cap5 sc0.05 | 1 | 2.009 | 23.95 | 4.166 |
| own | naive | 0.5 | 5000 | cap5 sc0.05 unnorm | 1 | 2.010 | 24.09 | 4.192 |
| own | naive | 0.5 | 5000 | cap5 sc0.25 | 1 | 2.006 | 24.17 | 4.191 |
| own | naive | 0.5 | 5000 | cap5 sc0.25 unnorm | 1 | 2.003 | 23.47 | 4.171 |
| own | naive | 1 | 30000 | none | 1 | 1.715 | 13.34 | 3.890 |
| own | naive | 1 | 10000 | none | 1 | 1.839 | 16.54 | 3.907 |
| own | naive | 1 | 5000 | none | 1 | 1.979 | 21.12 | 3.915 |
| own | naive | 2 | 30000 | none | 1 | 1.715 | 14.63 | 3.856 |
| own | naive | 2 | 10000 | none | 1 | 1.839 | 19.22 | 3.893 |
| own | naive | 2 | 5000 | none | 1 | 1.979 | 23.58 | 3.903 |
| own | naive | 3 | 30000 | none | 3 | 1.714 ± 0.001 | 15.58 ± 0.67 | 3.859 ± 0.043 |
| own | naive | 3 | 30000 | cap5 | 3 | 1.757 ± 0.002 | 19.99 ± 0.94 | 4.194 ± 0.033 |
| own | naive | 3 | 30000 | best | 3 | 1.740 ± 0.004 | 19.18 ± 0.47 | 4.204 ± 0.038 |
| own | naive | 3 | 30000 | sphere-L12 | 1 | 1.724 | 16.07 | 3.923 |
| own | naive | 3 | 30000 | sphere-L4 | 3 | 1.719 ± 0.002 | 16.09 ± 1.10 | 3.893 ± 0.034 |
| own | naive | 3 | 30000 | sphere-L4-lr5e-4 | 1 | 1.710 | 15.70 | 3.905 |
| own | naive | 3 | 30000 | sphere-L8 | 1 | 1.720 | 16.41 | 3.876 |
| own | naive | 3 | 25000 | none | 3 | 1.731 ± 0.002 | 16.52 ± 0.89 | 3.864 ± 0.052 |
| own | naive | 3 | 20000 | none | 3 | 1.752 ± 0.002 | 17.32 ± 0.96 | 3.874 ± 0.040 |
| own | naive | 3 | 20000 | cap5 | 1 | 1.804 | 22.48 | 4.231 |
| own | naive | 3 | 15000 | none | 3 | 1.785 ± 0.003 | 18.58 ± 1.06 | 3.873 ± 0.026 |
| own | naive | 3 | 15000 | cap5 | 2 | 1.835 ± 0.006 | 23.82 ± 1.46 | 4.232 ± 0.014 |
| own | naive | 3 | 10000 | none | 3 | 1.840 ± 0.004 | 20.03 ± 1.72 | 3.877 ± 0.032 |
| own | naive | 3 | 10000 | cap5 | 1 | 1.899 | 27.23 | 4.246 |
| own | naive | 3 | 10000 | sphere-L4 | 3 | 1.847 ± 0.002 | 20.86 ± 1.49 | 3.944 ± 0.051 |
| own | naive | 3 | 5000 | none | 3 | 1.979 ± 0.008 | 24.95 ± 3.13 | 3.876 ± 0.062 |
| own | naive | 3 | 5000 | cap5 | 3 | 2.041 ± 0.011 | 33.77 ± 1.18 | 4.237 ± 0.007 |
| own | naive | 3 | 5000 | cap5 sc0.005 | 1 | 2.010 | 32.47 | 4.221 |
| own | naive | 3 | 5000 | cap5 sc0.005 unnorm | 1 | 2.000 | 30.27 | 4.227 |
| own | naive | 3 | 5000 | cap5 sc0.05 | 1 | 2.009 | 30.81 | 4.224 |
| own | naive | 3 | 5000 | cap5 sc0.05 unnorm | 1 | 2.010 | 31.25 | 4.214 |
| own | naive | 3 | 5000 | cap5 sc0.25 | 1 | 2.006 | 31.40 | 4.244 |
| own | naive | 3 | 5000 | cap5 sc0.25 unnorm | 1 | 2.003 | 32.26 | 4.226 |
| own | horosphere | 0.5 | 30000 | none | 3 | 1.713 ± 0.003 | 13.07 ± 0.30 | 3.949 ± 0.015 |
| own | horosphere | 0.5 | 30000 | cap5 | 3 | 2.379 ± 0.527 | 56.18 ± 33.69 | 4.182 ± 0.089 |
| own | horosphere | 0.5 | 25000 | none | 1 | 1.730 | 13.89 | 3.952 |
| own | horosphere | 0.5 | 25000 | cap5 | 2 | 2.730 ± 0.118 | 77.00 ± 14.16 | 4.076 ± 0.002 |
| own | horosphere | 0.5 | 20000 | none | 1 | 1.752 | 14.51 | 3.962 |
| own | horosphere | 0.5 | 20000 | cap5 | 3 | 2.420 ± 0.504 | 66.55 ± 43.06 | 4.168 ± 0.124 |
| own | horosphere | 0.5 | 15000 | none | 1 | 1.783 | 15.43 | 3.953 |
| own | horosphere | 0.5 | 15000 | cap5 | 3 | 2.470 ± 0.494 | 70.69 ± 40.84 | 4.166 ± 0.019 |
| own | horosphere | 0.5 | 10000 | none | 1 | 1.837 | 17.00 | 3.964 |
| own | horosphere | 0.5 | 10000 | cap5 | 3 | 2.549 ± 0.462 | 74.93 ± 39.21 | 4.125 ± 0.021 |
| own | horosphere | 0.5 | 5000 | none | 1 | 1.975 | 24.18 | 4.037 |
| own | horosphere | 0.5 | 5000 | cap5 | 3 | 2.784 ± 0.355 | 91.23 ± 32.52 | 3.991 ± 0.133 |
| own | horosphere | 3 | 30000 | none | 3 | 1.713 ± 0.003 | 16.59 ± 0.68 | 3.906 ± 0.010 |
| own | horosphere | 3 | 30000 | cap5 | 3 | 2.379 ± 0.527 | 63.40 ± 34.37 | 4.069 ± 0.104 |
| own | horosphere | 3 | 25000 | none | 1 | 1.730 | 16.91 | 3.914 |
| own | horosphere | 3 | 20000 | none | 1 | 1.752 | 18.25 | 3.929 |
| own | horosphere | 3 | 20000 | cap5 | 1 | 1.840 | 29.79 | 4.143 |
| own | horosphere | 3 | 15000 | none | 1 | 1.783 | 19.99 | 3.935 |
| own | horosphere | 3 | 15000 | cap5 | 1 | 1.902 | 34.04 | 4.141 |
| own | horosphere | 3 | 10000 | none | 1 | 1.837 | 22.34 | 3.947 |
| own | horosphere | 3 | 10000 | cap5 | 1 | 2.020 | 46.76 | 4.175 |
| own | horosphere | 3 | 5000 | none | 1 | 1.975 | 31.42 | 4.010 |
| own | horosphere | 3 | 5000 | cap5 | 1 | 2.388 | 73.94 | 4.085 |
| paired | naive | 0.5 | 30000 | none | 3 | 1.714 ± 0.001 | 13.22 ± 0.59 | 3.907 ± 0.027 |
| paired | naive | 0.5 | 30000 | cap5 | 3 | 1.757 ± 0.002 | 15.20 ± 0.70 | 4.189 ± 0.017 |
| paired | naive | 0.5 | 30000 | sphere-L4 | 1 | 1.717 | 13.34 | 3.932 |
| paired | naive | 3 | 30000 | none | 3 | 1.714 ± 0.001 | 15.78 ± 0.81 | 3.866 ± 0.048 |
| paired | naive | 3 | 30000 | cap5 | 3 | 1.757 ± 0.002 | 20.01 ± 1.02 | 4.194 ± 0.034 |
| paired | naive | 3 | 30000 | sphere-L4 | 1 | 1.717 | 16.86 | 3.922 |
| paired | horosphere | 0.5 | 30000 | none | 3 | 1.713 ± 0.003 | 13.06 ± 0.31 | 3.947 ± 0.018 |
| paired | horosphere | 0.5 | 30000 | cap5 | 3 | 2.379 ± 0.527 | 56.47 ± 33.96 | 4.184 ± 0.092 |
| paired | horosphere | 3 | 30000 | none | 3 | 1.713 ± 0.003 | 16.62 ± 0.65 | 3.907 ± 0.008 |
| paired | horosphere | 3 | 30000 | cap5 | 3 | 2.379 ± 0.527 | 62.39 ± 33.63 | 4.066 ± 0.102 |
<!-- report:end -->

## Early reads (2026-10-01 02:15 EDT; 0 / 8 cells at 30k, `keep-{5000,10000}.ckpt` reads on mixed GPU types)

GenPPL (gpt2-large retokenized, first chunk) @ sample entropy; 64/64 distinct samples everywhere. Baseline rows are the
reused `claude_test_hbfm_tinystories_256` reads of the same checkpoints steps.

| readout | step | t_max | none (seeds) | cap5 (seeds) |
|---|---:|---:|---|---|
| naive | 5k | 0.5 | 20.6 / 19.9 / 22.7 @ 3.93 | 26.0 / 26.9 / 25.9 @ 4.21 |
| naive | 5k | 3.0 | 25.1 / 21.8 / 28.0 @ 3.88 | 34.1 / 32.5 / 34.8 @ 4.24 |
| naive | 10k | 0.5 | 16.3 / 15.9 / 17.4 @ 3.93 | 20.4 (seed 3) @ 4.22 |
| naive | 10k | 3.0 | 20.6 / 18.1 / 21.4 @ 3.88 | 27.2 (seed 3) @ 4.25 |
| naive | 15k | 0.5 | 15.3 / 14.6 / 15.8 @ 3.92 | 17.8 / 18.2 (seeds 2, 3) @ 4.21 |
| naive | 15k | 3.0 | 18.6 / 17.5 / 19.6 @ 3.87 | 22.8 / 24.9 (seeds 2, 3) @ 4.23 |
| naive | 20k | 0.5 | 14.2 / 13.6 / 15.3 @ 3.92 | 17.2 (seed 3) @ 4.22 |
| naive | 20k | 3.0 | 16.9 / 16.7 / 18.4 @ 3.87 | 22.5 (seed 3) @ 4.23 |
| horosphere | 5k | 0.5 | 24.2 (seed 1) @ 4.04 | 60.3 (seed 1) @ 4.12 |
| horosphere | 5k | 3.0 | 31.4 (seed 1) @ 4.01 | 73.9 (seed 1) @ 4.09 |
| horosphere | 10k | 0.5 | 17.0 (seed 1) @ 3.96 | 30.8 (seed 1) @ 4.15 |
| horosphere | 10k | 3.0 | 22.3 (seed 1) @ 3.95 | 46.8 (seed 1) @ 4.18 |
| horosphere | 15k | 0.5 | 15.4 (seed 1) @ 3.95 | 24.7 (seed 1) @ 4.15 |
| horosphere | 15k | 3.0 | 20.0 (seed 1) @ 3.94 | 34.0 (seed 1) @ 4.14 |

- At matched steps the RLF arm is 15–30 % worse in GenPPL under naive (every seed and horizon from 5k to 20k) and
  1.6–2.5× worse under horosphere, at a *higher* entropy (4.2 vs 3.9 nats) — worse quality, not a diversity trade-off.
  The horosphere gap shrinks with training (5k 2.5×, 10k 1.8×, 15k 1.6×) and the naive gap narrows only slowly
  (5k +25 %, 20k +15 % at t_max 0.5), so the 30k reads decide how much of it is a slower start.
- These are cross-GPU-type reads (±0.5 GenPPL); the final comparison uses the single-node paired re-reads.

## 30k finals — running tally (own-node reads; the paired single-node re-reads come when all cells are done)

GenPPL @ entropy (64/64 distinct everywhere); baseline = the reused `claude_test_hbfm_tinystories_256` spec cells.

| readout | arm | seed | t_max 3.0 (default) | t_max 0.5 (matched) | val/ppl bound |
|---|---|---|---:|---:|---:|
| naive | none | 1 / 2 / 3 | 15.56 / 14.91 / 16.25 @ 3.85 → **15.58 ± 0.67** | 12.90 / 12.95 / 13.90 → **13.25 ± 0.56** | 1.715 |
| naive | cap5 | 1 / 2 / 3 | 19.42 / 19.47 / 21.08 @ 4.19 → **19.99 ± 0.94** | 15.46 / 14.58 / 15.67 → **15.24 ± 0.58** | 1.757 |
| horosphere | none | 1 / 2 / 3 | 16.43 / 16.01 / 17.34 @ 3.91 → **16.59 ± 0.68** | 12.81 / 13.00 / 13.40 @ 3.95 → **13.07 ± 0.30** | 1.713 |
| horosphere | cap5 | 1 | 23.79 @ 4.14 | 17.28 @ 4.15 | 1.774 |
| horosphere | cap5 | 2 | **81.05** @ 4.11 | **76.27** @ 4.28 | **2.618** (collapsed run) |
| horosphere | cap5 | 3 | **85.36** @ 3.95 | **74.99** @ 4.11 | **2.744** (collapsed run) |
| horosphere | cap5 | 1 / 2 / 3 | 63.4 ± 34.5 | 56.2 ± 33.7 | 2 of 3 seeds collapsed |

- First final (2026-10-01 07:50 EDT): RLF under naive is +35 % GenPPL at the default horizon and +18 % at the matched
  horizon versus the baseline mean, outside the baseline's seed spread at both horizons, with a worse denoising bound
  (1.759 vs 1.715) and 0.35 nat higher sample entropy.
- **Naive arm complete (18:10 EDT):** RLF 19.99 ± 0.94 vs 15.58 ± 0.67 at the default horizon (+28 %) and
  15.24 ± 0.58 vs 13.25 ± 0.56 at the matched horizon (+15 %); worse in every seed at both horizons (own-node reads;
  Welch t ≈ 6.6 / 4.3), with a worse denoising bound (1.757 vs 1.715) and 0.3 nat higher entropy. By the
  pre-registered rule this is a consistent LOSS, not a tie.
- **Horosphere baseline complete (Oct 2 09:55 EDT):** 16.59 ± 0.68 / 13.07 ± 0.30 — within 0.2 of the naive baseline at
  both horizons (the previous study's "horosphere ties naive" finding, now on 3 seeds), against which the scale-1 HyLa
  horosphere arm (23.8 / 81.1 / 85.4) is unusable.
- **Horosphere seed 3 also collapsed (23:05 EDT):** 85 / 75, bound 2.74. With the scale-1 HyLa, horosphere + RLF
  trains to a usable model in only 1 of 3 seeds (23.8 / 17.3, itself 35–45 % behind the baseline); the naive + RLF arm
  trained in 3 of 3 seeds but 15–28 % behind. Per-checkpoint reads of both collapsed cells are queued to see whether
  they diverged late or never learned (seed 1's own curve was 60 → 31 → 25 → 21 → 17 at 5k → 30k).
- **Horosphere seed 2 collapsed (22:40 EDT):** GenPPL 81 / 76 with a denoising bound of 2.62 (vs 1.77 for seed 1 and
  1.71 for the baseline) — the same seed-fragility as the sudoku RLF cells (one collapsed seed per group, including a
  0 % cell). Per-checkpoint reads of its keep-{5k..25k}.ckpt at t_max 0.5 are queued to locate the collapse.
- Second final (10:05 EDT, horosphere seed 1): +45 % at the default horizon and +35 % at the matched horizon versus the
  baseline's seed 1 (16.43 / 12.81), bound 1.774 vs 1.71. Under both readouts the RLF model is worse at every step and
  every horizon read so far; the horosphere gap, which shrank from 2.5× at 5k to 1.6× at 15k, settles at 1.35–1.45× at
  30k — a slower start AND a worse model.

## Stage 2 — HyLa hyperparameter screen, 5k steps (`sweep.py --screen`; naive, seed 1, own-node reads)

Reference-code regime (github.com/ydtydr/HyLa: λ halved, `lambda_scale` 0.01–0.5 → our `hyla_scale` 0.005–0.25; no
1/√D → `hyla_normalize=false`). Compared at 5k steps with the baseline's three seeds and the stage-1 scale-1 arm.

| config | GenPPL @ t_max 3.0 | GenPPL @ t_max 0.5 | entropy | bound |
|---|---:|---:|---:|---:|
| baseline (none), seeds 1/2/3 | 25.1 / 21.8 / 28.0 → 25.0 | 20.6 / 19.9 / 22.7 → 21.1 | 3.88–3.93 | ~1.98 |
| stage 1: scale 1, normalized, seeds 1/2/3 | 34.1 / 32.5 / 34.8 → 33.8 | 26.0 / 26.9 / 25.9 → 26.3 | 4.20–4.24 | |
| scale 0.005, unnormalized | 30.27 | 23.58 | 4.18–4.23 | 2.000 |
| scale 0.005, normalized | 32.47 | 24.62 | 4.19–4.22 | 2.010 |
| scale 0.05, unnormalized | 31.25 | 24.09 | 4.19 | |
| scale 0.05, normalized | 30.81 | 23.95 | 4.17 | |
| scale 0.25, unnormalized (the sudoku winner) | 32.26 | 23.47 | 4.17 | 2.003 |
| scale 0.25, normalized | 31.40 | 24.17 | 4.19 | 2.006 |

- All six screens (04:15 EDT) land in one band: 30.8–32.5 at the default horizon and 23.5–24.6 at the matched horizon,
  i.e. the reference regime recovers roughly half of the stage-1 gap (33.8 / 26.3) but stays ~25 % / ~12 % behind the
  baseline at 5k with the same 0.3-nat entropy excess, and the six configs are indistinguishable at this step (the
  baseline's own 5k seed spread is ±1.5 / ±3.1). The 30k decision therefore follows the sudoku ranking: scale 0.25
  unnormalized (`best` arm, 3 seeds, running since 03:40–04:10 EDT).

## Why the horosphere + HyLa (scale 1) cells collapsed (2026-10-02 03:25 EDT)

Same gradient attribution as the sudoku diagnosis (`diag_collapse3.py`: random batch, B = 8 × 256, alpha 0.6, per-module
gradient norms at the final checkpoints):

| embedding-table gradient norm | collapsed seed 2 | collapsed seed 3 | healthy seed 1 |
|---|---:|---:|---:|
| full | **217** | **24** | 3.1 |
| HyLa path detached (Busemann readout only) | 1.5 | 1.5 | 1.2 |
| cap lowered to 2 | 69 | 281 | 3.1 |

- No prompts are pinned here, and the bridge states sit at s ≤ 1.9, so the sudoku pin path is not involved; yet in both
  collapsed models 90–99 % of the embedding gradient again comes through the HyLa feature map (the trunk's own
  gradients are 4–5× the healthy ones too: in_proj 45 vs 10). With λ ~ N(0, 1) the feature's gradient in the state
  direction carries a λ·sin(λP + b) term with |λ| up to ~3, so even interior states make the map steep and oscillatory
  in the embedding direction; once the trunk leans on a few sharp features, their gradients push the word embeddings
  around, the features drift under the trunk, and the run degrades (bound 2.6–2.7 vs 1.77) — the same positive feedback
  as on sudoku, minus the pin amplifier. Lowering the cap does nothing for interior states (seed 3: 24 → 281).
- The remedy common to both datasets is the reference-code regime (λ scale 0.005–0.25, which removes the oscillating
  term and was seed-stable on sudoku); the pin stop-gradient fix is additionally required on sudoku. The 30k `best`
  arm (scale 0.25, unnormalized, naive) tests exactly that here.

## Stage 1 — paired single-node re-reads of the 30k checkpoints (thickstun-compute-01, 2026-10-02 06:45 EDT)

Every finished stage-1 cell re-sampled on one node (`sweep.py --paired-eval`, 64 samples, 180 steps); GenPPL @ entropy.

| readout | arm | t_max 0.5 (seeds) | mean ± std | t_max 3.0 (seeds) | mean ± std |
|---|---|---|---:|---|---:|
| naive | none | 12.82 / 12.95 / 13.90 | **13.22 ± 0.59** @ 3.91 | 15.56 / 15.09 / 16.67 | **15.78 ± 0.81** @ 3.87 |
| naive | cap5 (HyLa, scale 1) | 15.45 / 14.42 / 15.74 | 15.20 ± 0.70 @ 4.19 | 19.75 / 19.14 / 21.13 | 20.01 ± 1.02 @ 4.19 |
| horosphere | none | 12.81 / 13.40 (seed 2 running) | 13.11 ± 0.42 @ 3.96 | 16.43 / 17.34 | 16.88 ± 0.65 @ 3.91 |
| horosphere | cap5 (HyLa, scale 1) | 17.28 / 77.15 / 74.99 | 56.5 ± 34.0 (2 collapsed) | 23.79 / 78.03 / 85.36 | 62.4 ± 33.6 |

- Same-node reads reproduce the own-node numbers to within ~0.3 GenPPL and settle the stage-1 question: at the
  paper's scale HyLa costs +15 % (matched horizon) / +27 % (default horizon) under naive, worse in every seed, and is
  unusable under horosphere (2 of 3 seeds collapsed). The stage-3 `best` arm (reference-regime scale 0.25,
  unnormalized, naive, 3 × 30k) is the remaining open question.

## Stage 3 — `best` arm (scale 0.25, unnormalized, naive), 30k finals (own-node reads)

| seed | t_max 3.0 (default) | t_max 0.5 (matched) | bound | baseline seed (naive) t_max 3.0 / 0.5 | scale-1 HyLa seed t_max 3.0 / 0.5 |
|---|---:|---:|---:|---|---|
| 1 | 18.65 @ 4.20 | 14.73 @ 4.19 | 1.744 | 15.56 / 12.90 | 19.42 / 15.46 |
| 2 | 19.37 @ 4.17 | 13.93 @ 4.18 | 1.736 | 14.91 / 12.95 | 19.47 / 14.58 |
| 3 | 19.52 @ 4.25 | 14.25 @ 4.25 | 1.739 | 16.25 / 13.90 | 21.08 / 15.67 |

- **Complete (Oct 3 06:00 EDT, own-node A5000 reads):** `best` = 19.18 ± 0.47 (t_max 3) / 14.30 ± 0.40 (t_max 0.5), bound 1.740, vs the naive baseline 15.58 ± 0.67 / 13.25 ± 0.56, bound 1.715 — worse in every seed at both horizons (+23 % / +8 %), with the same ≈ 0.3-nat entropy
  excess (4.17–4.25 vs 3.9). The reference-code regime (scale 0.25, unnormalized) beats the paper's scale 1 (19.99 ± 0.94 /
  15.24 ± 0.58) by 4–6 % but never reaches the coordinates. No paired single-node re-reads for these two seeds (the
  user closed HyLa; the 0.46 GPU-type spread cannot move a +8 % gap). **Final TinyStories verdict: HyLa does not
  improve HBFM in any configuration tried** (scale 1 capped, reference regime, 5k screens of six configs).
- First final (Oct 2 18:00 EDT): the reference-code regime improves on the paper's scale by 4–5 % (18.65 vs 19.42 at
  the default horizon, 14.73 vs 15.46 matched) and the bound by 0.013, but stays +20 % / +14 % behind the baseline's
  seed 1 with the same 0.3-nat entropy excess — the same picture as the sudoku tie-at-best, and well short of a gain.
- 2026-10-02 19:30 EDT — **HyLa closed by the user** ("cancel all pending HYLA jobs, add an option sphere"): the ch2263
  replicates (`best` × horosphere seeds 1-3, A100 copies of naive seeds 2-3) were cancelled before starting, so the
  horosphere readout of the `best` config is never measured. Seeds 2-3 above were already running and were left to
  finish (their finals land in `eval/`, `eval_tmax0.5/` of their cells). Verdict so far, unchanged by any read: no HyLa
  configuration matches the coordinate baseline on TinyStories (+14 % GenPPL at matched horizon is the best cell);
  together with sudoku (tie at best, seed-fragile) RLF is not an improvement. The successor is the bounded
  hyperbolic-harmonic basis (`model.sphere_degree`, `models/hyperbolic_dit.py::HarmonicFeatures`), tested on sudoku first
  (`experiments/claude_test_hbfm_rlf_sudoku`, stage 4); it comes to TinyStories only if it works there.

## Stage 4 — `sphere-L4` (harmonic basis, naive readout): early read at 10k steps (2026-10-03 09:35 EDT, own-node reads)

| cell | t_max 3.0 | t_max 0.5 | bound |
|---|---:|---:|---:|
| sphere-L4 seed 1 @ 10k | 21.72 @ 3.998 | 18.10 @ 3.995 | 1.845 |
| `none` seed 1 @ 10k | 20.63 @ 3.880 | 16.33 @ 3.934 | 1.839 |
| `none` seeds 1-3 @ 10k | 20.03 ± 1.72 @ 3.877 | 16.53 ± 0.80 @ 3.925 | 1.840 ± 0.004 |
| HyLa cap5 seed 3 @ 10k | 27.23 @ 4.246 | 20.41 @ 4.222 | — |

- At matched steps the harmonic trunk is behind the coordinate trunk: +8 % GenPPL at the default
  horizon and +9 % at the matched one, with a +0.12-nat higher sample entropy, and the
  bound is +0.005 worse — a clear, same-direction gap at both horizons, though less than half of HyLa's (+36 % / +24 %).
- The early-read caveat of the previous study applies (early reads misled across a change of objective; here the
  objective is the same but the input map is new, so convergence speed may differ). The 30k finals decide: seed 1
  ~18:00 EDT, seeds 2-3 ~07:30 EDT Oct 4. Seeds 2-3 get the same 10k read when they pass 10k (~14:00 EDT).

### 10k early reads, all three seeds (2026-10-03 15:30 EDT, own-node reads)

| seed | sphere-L4 t_max 3 | `none` t_max 3 | Δ | sphere-L4 t_max 0.5 | `none` t_max 0.5 | Δ |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 21.72 @ 4.00 | 20.63 @ 3.88 | +5 % | 18.10 @ 4.00 | 16.33 @ 3.93 | +11 % |
| 2 | 19.14 @ 3.90 | 18.10 @ 3.84 | +6 % | 16.37 @ 3.94 | 15.85 @ 3.91 | +3 % |
| 3 | 21.72 @ 3.94 | 21.37 @ 3.91 | +2 % | 17.80 @ 4.00 | 17.42 @ 3.94 | +2 % |
| mean | 20.86 ± 1.49 | 20.03 ± 1.72 | +4 % | 17.42 ± 0.93 | 16.53 ± 0.80 | +5 % |

- Seed-paired (same data seed as the baseline cell), the harmonic trunk is behind in every seed at both horizons,
  by 2–6 % (t_max 3) and 2–11 % (t_max 0.5), with a 0.0–0.1-nat higher sample entropy: a small, consistent deficit
  at a third of training — an order of magnitude smaller than HyLa's at the same point (+36 % / +24 %), but not a
  gain. Finals at 30k: seed 1 ~18:30 EDT, seeds 2-3 ~08:00 EDT Oct 4.

### 30k finals (own-node reads)

| seed | sphere-L4 t_max 3 | `none` t_max 3 | Δ | sphere-L4 t_max 0.5 | `none` t_max 0.5 | Δ | bound sphere / `none` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 (A6000) | 16.86 @ 3.92 | 15.56 @ 3.9 | +8 % | 13.47 @ 3.93 | 12.90 @ 3.9 | +4 % | 1.717 / 1.715 |
| 2 (A5000) | 14.84 @ 3.86 | 14.91 | −0.5 % | 13.22 @ 3.94 | 12.95 | +2 % | 1.721 / — |
| 3 (A5000) | 16.58 @ 3.91 | 16.25 | +2 % | 13.90 @ 3.95 | 13.90 | 0 % | 1.718 / — |
| **mean** | **16.09 ± 1.09** | **15.57 ± 0.67** | **+3 %** | **13.53 ± 0.34** | **13.25 ± 0.56** | **+2 %** | 1.719 / 1.715 |

- Seed 1 (2026-10-03 18:36 EDT): the harmonic trunk finishes at the baseline's sample entropy (3.92 vs 3.9 — HyLa's
  was 4.2) and the same likelihood bound (1.717 vs 1.715), with GenPPL +8 % / +4 % behind the coordinate trunk at the
  two horizons. Against HyLa's finals on the same seed (cap5 19.42 / 15.46; `best` 18.65 / 14.73) it is 10–13 % / 9–13 %
  better. Its 10k deficit (+5 % / +11 %) narrowed to +8 % / +4 %, i.e. the matched-horizon gap halved over training.
  The t_max 0.5 gap (0.57) is within the cross-GPU read spread (0.46) seen in stage 1, so a paired single-node
  re-read on thickstun-compute-01 was submitted for this cell (both horizons).

### Verdict for TinyStories (2026-10-04 06:50 EDT; stage 4 complete, naive readout, 3 seeds, own-node reads)

| input map (naive readout) | GenPPL t_max 3 | GenPPL t_max 0.5 | entropy | bound |
|---|---:|---:|---:|---:|
| coordinates (`none`) | 15.58 ± 0.67 | 13.25 ± 0.56 | 3.9 | 1.715 |
| **sphere L = 4** | **16.09 ± 1.09** | **13.53 ± 0.34** | 3.86–3.92 | 1.719 |
| HyLa, reference regime (`best`) | 19.18 ± 0.46 | 14.30 ± 0.40 | 4.2 | 1.739 |
| HyLa, paper config capped (`cap5`) | 19.99 ± 0.94 | 15.24 ± 0.58 | 4.2 | 1.757 |

- **Sphere ties the coordinates.** Seed-paired it is −0.5 / +2 / +8 % at the default horizon and 0 / +2 / +4 % at the
  matched one; the 3-seed means differ by +3 % / +2 %, inside the seed spread (0.7–1.1 at
  t_max 3, 0.3–0.6 at 0.5) and inside the cross-GPU read spread (0.46) at the matched horizon. Sample entropy
  (3.86–3.92 vs 3.9) and the likelihood bound (1.719 vs 1.715) are the baseline's — the harmonic trunk converges to the
  same model as the coordinate trunk, only from 496 bounded inputs instead of 48. On the GenPPL–entropy frontier it
  sits exactly at the baseline's point (13.5 @ 3.9), i.e. still above S-FLM / DUO / MDLM at matched entropy.
- **Sphere beats HyLa decisively**: 16 % / 5 % better than HyLa's best configuration and
  19 % / 11 % better than the paper's, with HyLa's 0.3-nat entropy inflation and bound penalty gone,
  and no seed instability (HyLa collapsed 2 of 3 horosphere seeds; the sphere seeds' spread equals the baseline's).
- **The 10k early reads were pessimistic** (+4 % / +5 % at a third of training → +3 % / +2 % at 30k; the matched-horizon
  gap closed), the opposite of HyLa, whose gap grew. Early reads again misled on the sign of the final trend.
- **Not run:** the horosphere readout (the stage-3 rule — only if naive shows a gain — was not met) and other L (sudoku
  showed no L dependence; L = 4 costs nothing but trunk width here). The paired single-node re-reads of seed 1
  (jobs 939782-939783) are still queued and will only add a row.

**Final answer for the RLF study (both tasks):** neither input map improves HBFM. HyLa (random Laplacian features)
is worse everywhere and unstable near the boundary; the harmonic basis `sphere` is a drop-in equivalent of the
coordinates — bounded, pin-proof, same accuracy on Sudoku (tie at L = 2 / 4 / 6), same GenPPL / entropy / bound on
TinyStories — which confirms the design note's reading that the coordinates already are the L = 1 harmonic basis and
that extra angular resolution is not what limits HBFM at this scale.

## Stage 5 — tuning screens on TinyStories (seed 1, naive, 30k; COMPLETE 2026-10-05 04:40 EDT)

| arm | GenPPL t_max 3 | GenPPL t_max 0.5 | entropy | bound | inputs |
|---|---:|---:|---:|---:|---:|
| `none` (coordinates), seed 1 | 15.56 | 12.90 | 3.9 | 1.715 | 48 |
| sphere L = 4 (stage 4), seed 1 | 16.86 (paired 16.86) | 13.47 (paired 13.34) | 3.92 | 1.717 | 496 |
| sphere L = 8 | 16.41 | 13.21 | 3.88 / 3.93 | 1.720 | 1392 |
| sphere L = 12 | 16.07 | 13.46 | 3.92 / 3.95 | 1.724 | 2800 |
| **sphere L = 4, LR 5e-4** | **15.70** | **13.30** | 3.91 / 3.94 | **1.710** | 496 |

- **Degree does not help on 50k words either**: L = 8 / 12 land within the seed noise of L = 4 at both horizons and the
  bound gets slightly worse with L (1.717 → 1.720 → 1.724). Together with sudoku (degree hurts under horosphere) this
  closes the positional-encoding hypothesis: beyond L ≈ 2–4 the extra harmonics add nuisance, not resolvable detail,
  at the radii HBFM trains on (median s 0.5–2.4, where f_l ≈ 0 for l ≳ 6).
- **LR 5e-4 is the lever**: on the same seed it takes the sphere trunk from 16.86 / 13.47 to 15.70 / 13.30 and the bound
  from 1.717 to **1.710, below the coordinates' 1.715** (the sphere seeds' bound spread is 0.002) — parity with the
  coordinates on GenPPL and a slightly better likelihood bound. None of the screens beats `none` seed 1 by the > 1
  GenPPL rule, so no seeds 2-3 yet; the LR arm is 0.14 worse / 0.40 worse at the two horizons (both inside the 0.46
  read spread).
- The paired single-node re-reads of sphere-L4 seed 1 (thickstun-compute-01) are 16.86 / 13.34 vs own-node 16.86 /
  13.47: the GPU-type effect is ≤ 0.13 here.

Stage 6 (submitted 11:15 EDT, seed 1, naive): `none-lr5e-4` (the control: does the coordinate trunk gain the same from
LR 5e-4?), `sphere-L4-lr1e-3`, `sphere-L2-lr5e-4` (jobs 990767-990769).

### Stage 6 control (2026-10-05 20:45 EDT, seed 1, naive): the coordinates gain from LR 5e-4 too

| arm (seed 1) | t_max 3 | t_max 0.5 | bound |
|---|---:|---:|---:|
| `none` LR 3e-4 | 15.56 | 12.90 | 1.715 |
| `none` LR 5e-4 | 15.88 @ 3.89 | 13.18 @ 3.93 | 1.707 |
| sphere L = 4, LR 3e-4 | 16.86 | 13.47 | 1.717 |
| sphere L = 4, LR 5e-4 | 15.70 @ 3.91 | 13.30 @ 3.94 | 1.710 |

At matched LR 5e-4 the sphere map and the coordinates tie (−0.18 / +0.12 GenPPL, bound +0.003): unlike Sudoku, the
coordinate trunk also benefits from the higher LR on TinyStories (bound 1.715 → 1.707). No sphere win on TinyStories.
- 2026-10-05 21:05 EDT: sphere L = 4 at LR 1e-3 (seed 1) = 15.59 @ 3.93 / 13.22 @ 3.97, bound **1.702** — the best bound of any
  HBFM TinyStories cell so far (`none` 5e-4: 1.707, 3e-4: 1.715); GenPPL ties. Control `none-lr1e-3` (seed 1) submitted.
- 2026-10-06 14:45 EDT: sphere L = 2 at LR 5e-4 (seed 1) = 15.27 @ 3.90 / 12.98 @ 3.93, bound 1.708 vs `none` at 5e-4 15.88 /
  13.18, 1.707 and `none` at 3e-4 15.56 / 12.90, 1.715: −0.61 / −0.20 GenPPL vs the matched-LR coordinates, equal bound —
  inside the 0.46 cross-GPU read spread, so a tie on one seed. `none-lr1e-3` (the control for sphere-L4 at 1e-3, bound 1.702) running.

### Stage 6 COMPLETE (2026-10-06 20:50 EDT): on TinyStories the LR lever belongs to the coordinates, not the sphere map

| seed 1, naive | LR 3e-4 | LR 5e-4 | LR 1e-3 |
|---|---|---|---|
| `none` (coordinates) | 15.56 / 12.90, 1.715 | 15.88 / 13.18, 1.707 | **14.82 / 12.28, 1.698** |
| sphere L = 4 | 16.86 / 13.47, 1.717 | 15.70 / 13.30, 1.710 | 15.59 / 13.22, 1.702 |
| sphere L = 2 | — | 15.27 / 12.98, 1.708 | — |

(GenPPL t_max 3 / t_max 0.5, bound.) The coordinates at LR 1e-3 are the best HBFM TinyStories cell of the whole study —
−0.77 / −0.94 GenPPL and −0.004 bound vs the sphere map at the same LR. **Verdict of the sphere tuning:** no sphere
configuration beats the raw Poincaré coordinates on TinyStories at matched LR; the only sphere win is Sudoku with the
naive readout at LR 5e-4 (+5 pt, 3 seeds). Side finding: HBFM's TinyStories LR (3e-4) was under-tuned — 1e-3 improves the
coordinate model by 5 % / 5 % GenPPL and 0.017 bound on seed 1 (seeds 2-3 not run).
