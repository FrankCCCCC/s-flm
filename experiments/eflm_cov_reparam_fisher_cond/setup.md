## The Angular Fisher Matrix Near the Origin

With $\Sigma=\operatorname{Cov}_{q_{\mathrm{data}}(v)}[v]$, near the origin:

$$
\boxed{
F(\rho_t, \omega_t)\simeq
(d-1)^2 \kappa^2 \rho_t^2 P_{\omega_t}\Sigma P_{\omega_t}
\simeq4(d-1)^2 r_t^2 P_{\omega_t}\Sigma P_{\omega_t}.
}
$$

When $q_{\mathrm{data}}(v)$ is long-tailed and $\omega_t$ lies along a principal direction, $P_{\omega_t}\Sigma P_{\omega_t}$ is ill-conditioned, which amplifies the KL cost along the leading directions.

---

## The Lower Bound of the Condition Number of Data Distribution Covariance $\Sigma$

Let $v_1$ be the most frequent word in the vocabulary. Then

$$
\operatorname{cond}(\Sigma) \geq \frac{(d-1)\, q_{\mathrm{data}}(v_1)\, \| v_1 - \mu \|^2}{1 - q_{\mathrm{data}}(v_1)}.
$$

Therefore, if $q_{\mathrm{data}}(v)$ is long-tailed and the most frequent word $v_1$ is far from the embedding mean $\mu$, then $\Sigma$ is ill-conditioned.

---

## The Lower Bound of the Condition Number of Posterior Covariance $\Sigma_{t}$

At any state $z_t$ the angular Fisher is exactly a posterior covariance:

$$
\begin{aligned}
F(\rho_t,\omega_t)&=(d-1)^2\sinh^2(\kappa\rho_t)\,\Sigma_t,\qquad
\tilde v_t:=P_{\omega_t}v/D_\kappa(v,\rho_t,\omega_t),\\
\Sigma_t&:=\operatorname{Cov}_{q(v\mid z_t)}[\tilde v_t],\qquad
\mu_t:=\mathbb E_{q(v\mid z_t)}[\tilde v_t].
\end{aligned}
$$

For the posterior mode $v_1^{(t)}$ and $d\ge3$, on the tangent space:

$$
\boxed{
\operatorname{cond}(\Sigma_t)
\ \ge\ \frac{(d-2)\,q(v_1^{(t)}\mid z_t)\,\|\tilde v_1^{(t)}-\mu_t\|^2}
{\operatorname{tr}\Sigma_t-q(v_1^{(t)}\mid z_t)\,\|\tilde v_1^{(t)}-\mu_t\|^2}.
}
$$

**Posterior skew and centroid distance control the conditioning.**

Prior frequencies enter only near the origin; far away the posterior collapses onto one word, $F\to0$, and the bound says nothing.


---

GOAL:

Read Agent.md and slides/sep25_2026/slides.md

Compute the condition number and stable rank of $\Sigma_t$ along $t \in [0, 1]$ and $\Sigma$. Consider the definition of $\Sigma_t :=\operatorname{Cov}_{q(v\mid z_t)}[\tilde v_t]$ and $\tilde v_t:=P_{\omega_t}v/D_\kappa(v,\rho_t,\omega_t)$, and $\Sigma=\operatorname{Cov}_{q_{\mathrm{data}}(v)}[v]$. Use Bayesian rule $q(v | z_t) \propto q(z_t | v) q(v)$ to sample the conditional distribution $q(v | z_t)$. Then, compute the $\Sigma_t$

Write a script called visualization/angular_fisher_sigma_t.py, given one EFLM checkpoint, output one figures with 15 subplots.

1. consider 2 different q(v)
    a. q(v) is the whole word embedding matrix (including word not used in training dataset), drawn the word v based on the training dataset frequency, labeled as ``data_freq``
    b. q(v) is the vocabulary bank of the training dataset, drawn the word v uniformly, labeled as ``uniform``
    c. q(v) is the model predicted likelihood, labeled as ``model_pred``

2. consider 5 metrics, denote \lambda_i are ordered eigen values, \lambda_i >= \lambda_{i+1}
    a. \lambda_1 / (\sum_i \lambda_i)
    b. k90: the minimum number of top \lambda_i whose sum is at least 90% of the energy \sum_i \lambda_i, i.e. k90 = min{ k : \sum_{i \leq k} \lambda_i ≥ 0.9 \sum_i \lambda_i}?
    c. k99: the minimum number of top \lambda_i whose sum is at least 99% of the energy \sum_i \lambda_i, i.e. k99 = min{ k : \sum_{i \leq k} \lambda_i ≥ 0.99 \sum_i \lambda_i}?
    d. eff rank (participation ratio): (\sum \lambda_i)^2 / \sum \lambda_i^2
    e. log10 cond: \log_{10}(\max_i \lambda_i / \min_i \lambda_i)

for each subplot, plot the metric in y axis, diffusion timestep / SNR in x axis, dash line represents forward_type=naive, solid line represents forward_type=gaussian; one color represents one kind of radius

The shaded band is the interquartile range across states: the 25th to 75th percentile of that metric over the 128 states sampled at each t. The line is the median of the same 128 values. Both files use the same band; only the x-axis differs.

add an option for timestep samples in x axis, default: 51, t = 0, 0.02, 0.04, …, 1.0. That comes from np.linspace(0, 1, 51).

Use the script angular_fisher_sigma_t.py to draw the figure, put figure under experiments/eflm_cov_reparam_fisher_cond/figures. All temparory data should be put in experiments/eflm_cov_reparam_fisher_cond/data.

use following checkpoints, put into one figure

| label       | checkpoint (under base)                                                                           |
| ----------- | ------------------------------------------------------------------------------------------------- |
| naive R=0.1 | `eflm_rescale_auto_trunc_tinystories_256/eflmratr_lr-3e-4_r-0.1_m-1.0/checkpoints/last.ckpt`      |
| naive R=0.5 | `eflm_rescale_auto_trunc_tinystories_256/eflmratr_lr-3e-4_r-0.5_m-1.0/checkpoints/last.ckpt`      |
| naive R=1.0 | `eflm_rescale_auto_trunc_tinystories_256/eflmratr_lr-3e-4_r-1_m-1.0/checkpoints/last.ckpt`        |
| gauss R=0.1 | `eflm_rescale_auto_trunc_cov_reparam_tinystories_256/ns-auto_r-0.1_lr-3e-4/checkpoints/last.ckpt` |
| gauss R=0.5 | `eflm_rescale_auto_trunc_cov_reparam_tinystories_256/ns-auto_r-0.5_lr-3e-4/checkpoints/last.ckpt` |
| gauss R=1.0 | `eflm_rescale_auto_trunc_cov_reparam_tinystories_256/ns-auto_r-1.0_lr-3e-4/checkpoints/last.ckpt` |

---

GOAL2

Check the models used in listed in above table, for each forward_type={naive,guassian} and R={0.1,0.5,1.0}, train EFLM with same config with dim=16,48,and 768, with 3 seeds, skip if already existed