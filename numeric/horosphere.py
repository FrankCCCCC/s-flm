import contextlib
from typing import List, Optional, Union

import torch

from numeric.geo_bridge import GeoUtils

@contextlib.contextmanager
def _fp32_matmul():
    """cuBLAS at full float32 (main.py enables TF32 for the DiT): the readout's
    gradient goes through `1 / A` with `A = sin^2(a/2) + cos^2(a/2) e^{-2s}`
    small for words near the state, which TF32's 1e-3 rounding of `<u, phi>`
    would corrupt."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def _horosphere_block(us, ss, decay, phis, factor_dim):
    """One vocabulary chunk of `HyperbolicBoundaryFM.horosphere_geometry`.

    `us` (B, L, m, d) unit bridge directions, `ss` (B, L, m) dimensionless
    radials, `decay` = exp(-2 ss), `phis` (Vc, m, d) unit boundary directions
    of the chunk's words; returns (B, L, Vc). The formula of
    `horosphere_geometry_tensor` on the chunk -- the
    cancellation-free squared-difference half angles -- as one compiled
    reduction over the (B, L, Vc, m, d) broadcast.
    """
    tiny = torch.finfo(us.dtype).tiny
    u = us.unsqueeze(-3)                                    # (B, L, 1, m, d)
    sin_half_sq = (u - phis).square().sum(-1) / 4           # (B, L, Vc, m)
    cos_half_sq = (u + phis).square().sum(-1) / 4
    term = ss.unsqueeze(-2) + (
        sin_half_sq + cos_half_sq * decay.unsqueeze(-2)).clamp_min(tiny).log()
    return -(factor_dim - 1) * term.sum(-1)


def _horosphere_block_dot(us, ss, decay, phis, factor_dim):
    """`_horosphere_block` with the half angles from the inner products,
    `sin^2(a/2) = (1 - <u, phi>) / 2`, `cos^2(a/2) = (1 + <u, phi>) / 2`: one
    (B L, d) x (d, Vc) product per factor, so its backward is two skinny
    matmuls per factor instead of a reduction over (B, L, Vc, m, d) -- 2.4x
    faster. Used only to differentiate (`_HorosphereChunk.backward`): its
    forward loses the target word's tiny sin^2(a/2) to cancellation in
    float32, while its gradient is that of the difference form up to (i)
    components along `u` / `phi` themselves, which the direction
    normalizations (theta = z / |z| in the polar readback, phi = e / |e| in
    `horosphere_geometry`) project away, and (ii) float rounding of `1 / A`
    where the word is already resolved (`A` ~ e^{-2s}, weight ~ 0).
    """
    tiny = torch.finfo(us.dtype).tiny
    out = None
    for i in range(us.shape[-2]):
        dot = us[..., i, :] @ phis[:, i, :].transpose(0, 1)           # (B, L, Vc)
        term = ss[..., i, None] + (
        (1 - dot) / 2 + (1 + dot) / 2 * decay[..., i, None]).clamp_min(tiny).log()
        out = term if out is None else out + term
    return -(factor_dim - 1) * out


_horosphere_block_compiled = torch.compile(_horosphere_block)
_horosphere_block_dot_compiled = torch.compile(_horosphere_block_dot)


class _HorosphereChunk(torch.autograd.Function):
    """Exact forward (`_horosphere_block`, nothing saved but the inputs), and a
    backward that recomputes the chunk in the inner-product form and
    differentiates that (`_horosphere_block_dot`), at full float32 matmul
    precision."""

    @staticmethod
    def forward(ctx, us, ss, decay, phis, factor_dim):
        r"""
        Squared-difference form (the base class): sin^2(a/2) = |u - \phi|^2/4. Cancellation-free, 
        so it is exact for the target word even when the state has nearly resolved it. 
        But autograd through it, even compiled, becomes inductor reduction kernels 
        over (B, L, chunk, 32, 3): 876 ms forward+backward for the readout alone.
        """
        ctx.save_for_backward(us, ss, decay, phis)
        ctx.factor_dim = factor_dim
        block = _horosphere_block_compiled if us.is_cuda else _horosphere_block
        return block(us, ss, decay, phis, factor_dim)

    @staticmethod
    def backward(ctx, grad_out):
        r"""
        Inner-product form: sin^2(a/2) = (1 - \langle u, \phi\rangle)/2. Its backward is 
        two skinny matmuls per factor (cuBLAS), 365 ms forward+backward. But in float32 
        its forward loses the target word's tiny sin² to cancellation once the radial reaches s \approx 5; 
        it fails the repo's fp32 Bayes-posterior test (test_horosphere_readout_is_bayes_posterior, atol 1e-4) 
        by ~1e-3 nats.
        """
        us, ss, decay, phis = ctx.saved_tensors
        block = _horosphere_block_dot_compiled if us.is_cuda else _horosphere_block_dot
        with torch.enable_grad(), _fp32_matmul():
            leaves = [t.detach().requires_grad_(True) for t in (us, ss, decay, phis)]
            out = block(*leaves, ctx.factor_dim)
            grads = torch.autograd.grad(out, leaves, grad_out)
        return (*grads, None)


class HorosphereGeometry:
    # Words per block of `horosphere_geometry`: the forward's (B, L, READOUT_CHUNK, m)
    # intermediates and the backward's ~32 (B, L, READOUT_CHUNK) tensors are
    # ~0.5 GB / ~4 GB at a 16 x 256 micro-batch in float32.
    READOUT_CHUNK = 2048

    @staticmethod
    def horosphere_geometry_chunk(
        theta: torch.Tensor,
        radius: torch.Tensor,
        word_embedding: torch.Tensor,
        prod_factor_dim: Optional[Union[int, List[int]]] = None,
        prod_factor_gaussian_curvature: Optional[Union[float, List[float]]] = None,
        readout_dtype: torch.dtype = torch.float64,
    ):
        """Memory-bounded `horosphere_geometry_tensor`.

        The same quantity -- per word v the sum over factors i of
        `-(d-1) (s_i + log(sin^2(a_iv/2) + cos^2(a_iv/2) e^{-2 s_i}))`, the
        half-angle form of the Busemann log-densities with the cancellation-free
        squared-difference half angles -- but the base class broadcasts one
        (B, L, V, m, d) float64 tensor, 147 GB for a 16 x 256 micro-batch on
        (H^3)^32 with V = 50257. Here the same formula is evaluated in
        `readout_dtype` on `READOUT_CHUNK`-word blocks (`_horosphere_block`,
        torch.compile'd on CUDA) through `_HorosphereChunk`, which saves only the
        block's inputs and differentiates the inner-product form of the same
        quantity in its backward (`_horosphere_block_dot`: two skinny matmuls per
        factor instead of a reduction over (B, L, Vc, m, d)), so the live memory
        is O(B L V) whatever the number of factors and a 16 x 256 micro-batch
        costs ~0.4 s forward + backward on an RTX A5000 (base form: 147 GB,
        out of memory). Computed and returned in `readout_dtype`, like
        `horosphere_geometry_tensor`: float32 is adequate while the dimensionless
        radial stays below ~8 (configs/algo/hbfm.yaml).
        """
        embedding = word_embedding
        dims, curvatures = GeoUtils.validate_prod_factors(
            prod_factor_dim=prod_factor_dim,
            prod_factor_gaussian_curvature=prod_factor_gaussian_curvature,
            embedding_size=embedding.shape[-1])
        if len(set(dims)) != 1:
            raise ValueError(
                f'horosphere_geometry needs one shared factor dimension; got {dims}.')
        factor_dim, num_factors = dims[0], len(dims)
        dtype = readout_dtype
        tiny = torch.finfo(dtype).tiny
        phis = embedding.to(dtype).unflatten(-1, (num_factors, factor_dim))
        phis = phis / phis.norm(dim=-1, p=2, keepdim=True).clamp_min(tiny)
        us = theta.to(dtype).unflatten(-1, (num_factors, factor_dim))
        kappas = radius.new_tensor(
            [1.0 / GeoUtils._curvature_scale(k) for k in curvatures], dtype=dtype)
        ss = radius.to(dtype) * kappas
        decay = (-2.0 * ss).exp()
        outs = [_HorosphereChunk.apply(us, ss, decay, phis_c, factor_dim)
                for phis_c in phis.split(HorosphereGeometry.READOUT_CHUNK, dim=0)]
        return torch.cat(outs, dim=-1)

    @staticmethod
    def horosphere_geometry_tensor(
        theta: torch.Tensor,
        radius: torch.Tensor,
        word_embedding: torch.Tensor,
        prod_factor_dim: Optional[Union[int, List[int]]] = None,
        prod_factor_gaussian_curvature: Optional[Union[float, List[float]]] = None,
        readout_dtype: torch.dtype = torch.float64,
    ):
        """
        Horocycle distance (Poisson kernel) for each word embedding.

        `horosphere_dists[..., v] = sum_i -(d_i - 1) B^i_v(z_i)` -- `B^i_v` the
        Busemann function of word `v`'s boundary point in factor `i`. It is the
        log density of the bridge direction at word `v`, up to a `v`-independent
        constant, so `softmax(horosphere_dists + log p)` is exactly the Bayes
        posterior `q(y | z_t)`; the factors are independent Brownian motions, so
        their Busemann terms simply add. This is the product-manifold form of
        `HyperBridge.horosphere_geometry`, and every consumer of the naive logits
        must treat them as a RESIDUAL on top of this term.

        The half-angle form is the load-bearing one (Invariant 5):
        `cosh s - sinh s <u, phi_v> = e^{+s} sin^2(a_v/2) + e^{-s} cos^2(a_v/2)`,
        with `s = rho_i / R_i` the DIMENSIONLESS radial -- curvature enters only
        here, exactly as in `HyperbolicHeatKernel._angular_boost`. Pulling
        `e^{+s}` out of the log makes it overflow-free, the squared-difference
        forms are cancellation-free, and `clamp_min` keeps the log finite once
        both terms underflow (`s > ~372`, where `sin_half_sq` is exactly 0 for
        the target word). Computed in float64 per Invariant 3.

        Args:
            theta (`torch.Tensor` of shape `(batch_size, max_seq_len, input_theta_dim)`):
                Per-factor unit boundary directions, concatenated.
            radius (`torch.Tensor` of shape `(batch_size, max_seq_len, input_radius_dim)`):
                Intrinsic radial coordinate of each product factor.
            prod_factor_dim (`Union[int, List[int]]`, *optional*):
                Dimension of each factor; they must all be EQUAL here. Both
                lists `None` means the single factor `[embedding_size]` at
                `[-1.0]`.
            prod_factor_gaussian_curvature (`Union[float, List[float]]`, *optional*):
                Curvature `K_i < 0` of each factor, free to differ per factor.

        Returns:
            `torch.Tensor` of shape `(batch_size, max_seq_len, vocab_size)`:
                Horosphere (Busemann) log-densities, float64.
        """
        embedding = word_embedding
        if embedding is None:
            raise ValueError(
                f"word_embedding is None; the horosphere "
                "readout needs the per-word boundary directions. Every model on "
                "this path owns a table -- MLPLMRefactor's lm_head.weight, "
                "OptimalModelRefactor's frozen uniform_sphere_points buffer."
            )
        embedding = embedding.to(readout_dtype)
        dims, curvatures = GeoUtils.validate_prod_factors(
            prod_factor_dim=prod_factor_dim,
            prod_factor_gaussian_curvature=prod_factor_gaussian_curvature,
            embedding_size=embedding.shape[-1],
        )
        if theta.shape[-1] != embedding.shape[-1]:
            raise ValueError(
                f"theta must carry one direction per factor, {embedding.shape[-1]} "
                f"channels in total; got {theta.shape[-1]}."
            )
        if radius.shape[-1] != len(dims):
            raise ValueError(
                f"radius must carry one radial coordinate per product factor "
                f"({len(dims)}); got {radius.shape[-1]}."
            )
        if len(set(dims)) != 1:
            raise ValueError(
                f"horosphere_geometry needs one shared factor dimension; got {dims}."
            )
        theta = theta.to(readout_dtype)
        radius = radius.to(readout_dtype)
        tiny = torch.finfo(readout_dtype).tiny

        # No python loop over factors: one shared factor dimension splits the
        # concatenated boundary axis by a reshape, and the whole formula is
        # elementwise in the resulting factor axis, which the final sum
        # contracts away. Curvature stays per-factor, as the `kappas` vector.
        factor_dim, num_factors = dims[0], len(dims)
        # us: (..., 1, m, d) against phis: (V, m, d) -> (..., V, m, d)
        us = theta.unflatten(-1, (num_factors, factor_dim)).unsqueeze(-3)
        phis = embedding.unflatten(-1, (num_factors, factor_dim))
        phis = phis / phis.norm(dim=-1, p=2, keepdim=True).clamp_min(tiny)
        #   sin^2(a_v/2) = (1 - <u, phi_v>) / 2 = ||u - phi_v||^2 / 4
        #   cos^2(a_v/2) = (1 + <u, phi_v>) / 2 = ||u + phi_v||^2 / 4
        sin_half_sq = (us - phis).square().sum(-1) / 4
        cos_half_sq = (us + phis).square().sum(-1) / 4
        kappas = radius.new_tensor(
            [1.0 / GeoUtils._curvature_scale(k) for k in curvatures]
        )
        ss = (radius * kappas).unsqueeze(-2)
        return -(factor_dim - 1) * (
            ss + (
                sin_half_sq + cos_half_sq * (-2.0 * ss).exp()
            ).clamp_min(tiny).log()
        ).sum(-1)

    @staticmethod
    def compute_horosphere(
        theta: torch.Tensor,
        radius: torch.Tensor,
        word_embedding: torch.Tensor,
        prod_factor_dim: Optional[Union[int, List[int]]] = None,
        prod_factor_gaussian_curvature: Optional[Union[float, List[float]]] = None,
        readout_dtype: torch.dtype = torch.float64,
        forward_chunked: bool = True,
    ):
        if forward_chunked:
            return HorosphereGeometry.horosphere_geometry_chunk(
                theta, radius, word_embedding, prod_factor_dim, prod_factor_gaussian_curvature, readout_dtype
            )
        else:
            return HorosphereGeometry.horosphere_geometry_tensor(
                theta, radius, word_embedding, prod_factor_dim, prod_factor_gaussian_curvature, readout_dtype
            )
