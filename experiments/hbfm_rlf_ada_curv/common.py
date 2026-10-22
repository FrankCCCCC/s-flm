"""Shared loader of the analysis scripts (analysis only): one trained cell -> (cfg, tokenizer, model).

Runs on a GPU when there is one and on CPU otherwise. The DiT's attention and rotary kernels
(flash_attn) are CUDA-only, so on CPU they are replaced by PyTorch's scaled-dot-product attention
and the reference GPT-NeoX rotary embedding -- the same functions (`models/dit.py` keeps the same
fallbacks as comments), in float32 instead of bf16.
"""
import sys

import hydra
import omegaconf
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
import algo  # noqa: E402
import dataloader  # noqa: E402
import main  # noqa: E402,F401  (registers the hydra resolvers)
import models.dit  # noqa: E402

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def _attention(qkv, dropout_p=0.0, causal=False, **kwargs):
  """flash_attn.flash_attn_qkvpacked_func on CPU: qkv [B, S, 3, H, D] -> [B, S, H, D]."""
  q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
  return F.scaled_dot_product_attention(q, k, v, is_causal=causal).transpose(1, 2)


def _rotary(qkv, cos, sin, *args, **kwargs):
  """flash_attn.layers.rotary.apply_rotary_emb_qkv_ on CPU: rotate q and k of qkv [B, S, 3, H, D]."""
  cos = torch.cat([cos, cos], dim=-1)[None, :, None, :]
  sin = torch.cat([sin, sin], dim=-1)[None, :, None, :]
  q, k = (x * cos + models.dit.rotate_half(x) * sin for x in (qkv[:, :, 0], qkv[:, :, 1]))
  return torch.stack([q, k, qkv[:, :, 2]], dim=2)


def load_cell(cell, ckpt='last.ckpt', ema=True):
  if DEVICE == 'cpu':
    models.dit.flash_attn.flash_attn_qkvpacked_func = _attention
    models.dit.flash_attn.layers.rotary.apply_rotary_emb_qkv_ = _rotary
  overrides = [o for o in omegaconf.OmegaConf.load(f'{cell}/.hydra/overrides.yaml')
               if not o.startswith('hydra.')]
  with hydra.initialize(config_path='../../configs', version_base=None):
    cfg = hydra.compose(config_name='config', overrides=overrides)
  tokenizer = dataloader.get_tokenizer(cfg)
  model = algo.HyperbolicBoundaryFM.load_from_checkpoint(
    f'{cell}/checkpoints/{ckpt}', tokenizer=tokenizer, config=cfg, strict=False,
    map_location=DEVICE).to(DEVICE)
  if not ema:
    model.ema = None
  model._eval_mode()
  return cfg, tokenizer, model
