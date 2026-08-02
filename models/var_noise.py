"""NN parametrization of the learned noise schedule for
`noise_schedules.VariationalAdaptiveSchedule`.

Structure follows MuLAN (Diffusion Models With Learned Adaptive Noise,
Suppl. E.2): a network reads a context c -- here the DLM's final hidden state,
[B, L, D] -- and emits the coefficients of a basis whose integral is monotone
in t.  The schedule is therefore a CLOSED FORM in t, so d/dt is analytic.

Two deviations from MuLAN, both required by the setup:

  * MuLAN emits one gamma per input dimension (a diagonal schedule).  Here the
    output is a SCALAR per sequence (`scope='global'`, [B]) or per position
    (`scope='positional'`, [B, L]).  The per-position mode is what gives EFLM a
    learnable decoding order.
  * MuLAN maps its polynomial onto gamma in [gamma_min, gamma_max] and then
    sets alpha^2 = sigmoid(-gamma) (variance preserving).  EFLM's interpolant
    x_t = alpha_t e + (1 - alpha_t) eps is not variance preserving, so we emit
    the normalized polynomial tau(c, t) = f(c, t) / f(c, 1) in [0, 1] and let
    `VariationalAdaptiveSchedule` compose it with the base schedule,
    alpha_t = alpha_base(tau).  tau is MuLAN's gamma up to the affine map
    gamma = gamma_min + (gamma_max - gamma_min) tau, so nothing is lost; the
    gain is that tau = t at init reproduces the un-warped baseline exactly.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dit import DDiTBlock, LayerNorm, Rotary

# Keeps f(c, 1) (the normalizer) and the RBF widths away from zero.
_EPS = 1e-6
# Uniform mix on f'(t): f' += _RIDGE, so f is strictly increasing, the
# normalizer f(c, 1) >= _RIDGE (no 0/0 when the basis coefficients collapse),
# and -- because the ridge integrates into BOTH f and f(c, 1) -- tau(c, 1) = 1
# stays exact.  Same trick and magnitude as AdaptiveSchedule's uniform_mix.
_RIDGE = 1e-3


def _inv_softplus(y):
  return math.log(math.expm1(y))


class GammaNet(nn.Module):
  """Maps (c, t) -> (tau, dtau/dt), the monotone time warp of MuLAN.

  Args:
    in_dim: width of the context c.
    embed_dim: width of this network's trunk.
    num_layer: trunk depth (MLP hidden layers / DiT blocks).
    model_type: 'dit' (attention over positions) or 'mlp' (per position).
    scope: 'positional' -> tau is [B, L]; 'global' -> tau is [B].
    basis: 'polynomial' (MuLAN Suppl. E.2) or 'rbf' (Gaussian bumps).
    degree: polynomial degree (odd) or number of RBF bases.

  No dropout anywhere: the loss weight divides two evaluations of this net
  (at t and at the interval edge), and train-mode dropout would give them
  different coefficients -- breaking E[w] = 1 and feeding the schedule
  gradients of a different random warp (found in external review).
  """

  def __init__(self, in_dim, embed_dim, num_layer, model_type,
               scope, basis, degree):
    super().__init__()
    if scope not in ('global', 'positional'):
      raise ValueError(f'Unknown scope: {scope}')
    if basis not in ('polynomial', 'rbf'):
      raise ValueError(f'Unknown basis: {basis}')
    if basis == 'polynomial' and (degree < 1 or degree % 2 == 0):
      raise ValueError(
        f'The polynomial basis needs an odd degree >= 1 (its derivative is a '
        f'perfect square, MuLAN Suppl. E.2), got {degree}.')
    if basis == 'rbf' and degree < 1:
      raise ValueError(f'The rbf basis needs degree >= 1, got {degree}.')
    self.scope = scope
    self.basis = basis
    self.degree = degree
    self.model_type = model_type

    if basis == 'polynomial':
      # f'(t) = (sum_j psi_j t^j)^2 has degree `degree` - 1 -> m + 1 coefs.
      self.n_coef = (degree + 1) // 2
      # exponents of q = psi (*) psi, the expansion of f'(t)
      self.register_buffer(
        'powers', torch.arange(degree, dtype=torch.float32),
        persistent=False)
    else:
      self.n_coef = degree
      self.register_buffer(
        'centers', (torch.arange(degree, dtype=torch.float32) + 0.5) / degree,
        persistent=False)
      self.width = 1.0 / degree

    if model_type == 'mlp':
      layers = [LayerNorm(in_dim), nn.Linear(in_dim, embed_dim), nn.SiLU()]
      for _ in range(num_layer - 1):
        layers += [nn.Linear(embed_dim, embed_dim), nn.SiLU()]
      self.trunk = nn.Sequential(*layers)
      self.blocks = None
    elif model_type == 'dit':
      self.trunk = nn.Sequential(LayerNorm(in_dim),
                                 nn.Linear(in_dim, embed_dim))
      n_heads = max(1, embed_dim // 64)
      self.rotary_emb = Rotary(embed_dim // n_heads)
      self.blocks = nn.ModuleList([
        DDiTBlock(dim=embed_dim, n_heads=n_heads, adaLN=False,
                  dropout=0.0)
        for _ in range(num_layer)])
      self.final_norm = LayerNorm(embed_dim)
    else:
      raise ValueError(f'Unknown model_type: {model_type}')

    self.head = nn.Linear(embed_dim, self.n_coef)
    # Identity init: zero weights + a bias that makes tau(t) = t, so the
    # schedule starts exactly at the base schedule and the MuLAN reweighting
    # starts at w == 1 (see VariationalAdaptiveSchedule.loss_weight).
    # Residual quirk (accepted): at EXACTLY psi = 0 the polynomial warp is the
    # ridge identity with d tau / d psi = 0 (f is quadratic in psi), a
    # gradient-dead saddle.  It is measure-zero, repelled from the anchor init
    # below, and fixing it (gauge p(0) = 1) would pin f'(0) > 0, giving up
    # MuLAN's zero-derivative-endpoint flexibility.
    self.head.weight.data.zero_()
    bias = torch.zeros(self.n_coef)
    if basis == 'polynomial':
      bias[0] = 1.0                      # psi = (1, 0, ...) -> f'(t) = 1
    else:
      bias.fill_(_inv_softplus(1.0))     # equal weight on every bump
    self.head.bias.data.copy_(bias)

  def _coefs(self, c):
    """c: [B, L, in_dim] -> [B, n_coef] (global) or [B, L, n_coef]."""
    h = self.trunk(c)
    if self.blocks is not None:
      rotary_cos_sin = self.rotary_emb(h)
      for block in self.blocks:
        h = block(h, rotary_cos_sin, c=None)
      h = self.final_norm(h)
    if self.scope == 'global':
      h = h.mean(dim=1)
    return self.head(h)

  def _polynomial(self, psi, t):
    # f'(t) = p(t)^2 with p(t) = sum_j psi_j t^j; q = psi convolved with psi
    # holds the coefficients of f'.  f(t) = sum_k q_k t^(k+1) / (k+1).
    m = self.n_coef - 1
    q = [0.0] * (2 * m + 1)
    for i in range(m + 1):
      for j in range(m + 1):
        q[i + j] = q[i + j] + psi[..., i] * psi[..., j]
    q = torch.stack(q, dim=-1)                       # [..., degree]
    scale = self.powers + 1.0                        # k + 1
    t_pow = t ** self.powers                         # [B, 1(, 1), degree]
    f_prime = (q * t_pow).sum(-1)
    f = (q * t_pow * t / scale).sum(-1)
    f_one = (q / scale).sum(-1)
    return f, f_prime, f_one

  def _rbf(self, psi, t):
    # f'(t) = sum_k w_k exp(-(t - mu_k)^2 / (2 s^2)) >= 0, so f is monotone.
    # Its antiderivative uses erf: int exp(-x^2/(2 s^2)) dx
    #   = s sqrt(pi/2) erf(x / (s sqrt(2))).
    w = F.softplus(psi) + _EPS
    s = self.width
    norm = s * math.sqrt(math.pi / 2)
    denom = s * math.sqrt(2.0)

    def _int(u):
      return norm * torch.erf((u - self.centers) / denom)

    base = _int(torch.zeros_like(self.centers))
    f_prime = (w * torch.exp(-(t - self.centers) ** 2 / (2 * s ** 2))).sum(-1)
    f = (w * (_int(t) - base)).sum(-1)
    f_one = (w * (_int(torch.ones_like(self.centers)) - base)).sum(-1)
    return f, f_prime, f_one

  def forward(self, t, c):
    """t: [B]; c: [B, L, in_dim].  Returns (tau, dtau/dt).

    Shapes: [B] for scope='global', [B, L] for scope='positional'.
    """
    psi = self._coefs(c).float()
    # Broadcast t over the coefficient (and position) axes.
    t = t.reshape(t.shape[0], *([1] * (psi.ndim - 1))).float()
    if self.basis == 'polynomial':
      f, f_prime, f_one = self._polynomial(psi, t)
    else:
      f, f_prime, f_one = self._rbf(psi, t)
    # Uniform mix (see _RIDGE): endpoints stay exact, normalizer stays > 0.
    t_flat = t.squeeze(-1)
    f = f + _RIDGE * t_flat
    f_prime = f_prime + _RIDGE
    f_one = f_one + _RIDGE
    return (f / f_one).clamp(0.0, 1.0), f_prime / f_one


class PromptContext(nn.Module):
  """Inference-available schedule context: an encoding of the PROMPT.

  MuLAN Suppl. D distinguishes two conditioning regimes.  Conditioning the
  schedule on a function of x0 that is unavailable at inference (D.2 -- e.g.
  a DLM hidden state of the corrupted truth) underperforms the unconditioned
  baseline (Fig. 4): the reverse process has no consistent stand-in for the
  context.  Conditioning on context that IS available at inference (D.1) is
  sound.  For prefix tasks like sudoku the prompt (the clues) is exactly
  such a context -- and it is what a decoding order should adapt to.

  Deterministic per-sequence encoding z in R^{z_dim}: own token embedding
  (decoupled from the DLM), MLP, masked mean over the PROMPT positions
  (valid_tokens == 0).  Identical construction at training (x0's prompt) and
  sampling (prefix_tokens), computed ONCE and held fixed per trajectory --
  no per-step drift, no oracle leak of the solution.  `expand` broadcasts z
  over positions with a learned positional embedding so a positional
  GammaNet emits one gamma per position from the per-sequence z (MuLAN:
  per-pixel gamma from one latent).
  """

  def __init__(self, vocab_size, z_dim, width, max_len):
    super().__init__()
    self.z_dim = z_dim
    self.embed = nn.Embedding(vocab_size, width)
    # The encoder pools over positions, so positions must enter the input:
    # for sudoku the clue LOCATIONS, not just the digit multiset, determine
    # a sensible decoding order.
    self.enc_pos_embed = nn.Parameter(torch.zeros(1, max_len, width))
    nn.init.normal_(self.enc_pos_embed, std=0.02)
    self.mlp = nn.Sequential(
      nn.Linear(width, width), nn.SiLU(),
      nn.Linear(width, z_dim))
    self.pos_embed = nn.Parameter(torch.zeros(1, max_len, z_dim))
    nn.init.normal_(self.pos_embed, std=0.02)

  def forward(self, tokens, valid_tokens=None):
    """tokens: [B, L] ints -> z [B, z_dim].

    valid_tokens marks GENERATED positions (1); the mean pools the prompt
    (valid_tokens == 0).  None -> pool everything (prompt-only input, as in
    the sampler, or a promptless dataset).
    """
    L = tokens.shape[1]
    h = self.mlp(self.embed(tokens) + self.enc_pos_embed[:, :L])
    if valid_tokens is not None:
      m = (1.0 - valid_tokens.to(h.dtype)).unsqueeze(-1)
      return (h * m).sum(1) / m.sum(1).clamp(min=1.0)
    return h.mean(1)

  def expand(self, z, length):
    """[B, z_dim] -> [B, L, z_dim] context for GammaNet."""
    return z.unsqueeze(1) + self.pos_embed[:, :length]
