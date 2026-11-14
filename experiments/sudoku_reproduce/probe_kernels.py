#!/usr/bin/env python
"""probe_kernels.py -- which attention backend and GPU kernels one MDLM Sudoku training step uses on this GPU.

Builds the tiny DiT the Sudoku MDLM cells train (8 blocks, 512 wide, 8 heads, 180 tokens), runs one forward and
backward pass at batch 128 under bf16 autocast (as Lightning's bf16 precision does), with TF32 set as main.py sets
it, and prints the SDPA backends PyTorch accepts for these shapes and the GEMM / attention kernels launched.
--sdpa-math and --no-tf32 apply the same switches as main.py's `sdpa_math=True` and `allow_tf32=False`.
Run on a compute node only.

Usage: python experiments/sudoku_reproduce/probe_kernels.py [--sdpa-math] [--no-tf32]
"""
import collections
import os
import sys

import omegaconf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
import models.dit as dit  # noqa: E402

if '--no-tf32' not in sys.argv:  # main.py with allow_tf32: True (configs/config.yaml)
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.set_float32_matmul_precision('high')
if '--sdpa-math' in sys.argv:
  torch.backends.cuda.enable_flash_sdp(False)
  torch.backends.cuda.enable_mem_efficient_sdp(False)
  torch.backends.cuda.enable_cudnn_sdp(False)

print(torch.cuda.get_device_name(), 'capability', torch.cuda.get_device_capability(), 'torch', torch.__version__)
print('enabled: flash', torch.backends.cuda.flash_sdp_enabled(), 'mem_efficient',
      torch.backends.cuda.mem_efficient_sdp_enabled(), 'cudnn', torch.backends.cuda.cudnn_sdp_enabled(),
      'math', torch.backends.cuda.math_sdp_enabled(),
      '| math in bf16 (not fp32):', torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed())
q = torch.randn(128, 8, 180, 64, device='cuda', dtype=torch.bfloat16)
p = torch.backends.cuda.SDPAParams(q, q, q, None, 0.0, False, False)
for name, fn in [('flash', torch.backends.cuda.can_use_flash_attention),
                 ('mem_efficient', torch.backends.cuda.can_use_efficient_attention),
                 ('cudnn', torch.backends.cuda.can_use_cudnn_attention)]:
  print(f'can use {name}:', fn(p, debug=True))

cfg = omegaconf.OmegaConf.create({'algo': {'causal_attention': False, 'adaLN': True},
                                  'model': {'hidden_size': 512, 'cond_dim': 128, 'length': 180,
                                            'n_blocks': 8, 'n_heads': 8, 'dropout': 0.1}})
model = dit.DIT(cfg, vocab_size=11).cuda().train()
xt = torch.randint(0, 11, (128, 180), device='cuda')
sigma = torch.rand(128, device='cuda')


def step():
  with torch.autocast('cuda', dtype=torch.bfloat16):
    loss = model(None, xt, sigma).float().logsumexp(-1).mean()
  loss.backward()
  torch.cuda.synchronize()


step()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
  step()
kernels = collections.Counter(e.name for e in prof.events()
                              if e.device_type == torch.autograd.DeviceType.CUDA)
keys = ('gemm', 'flash', 'fmha', 'attention', 'softmax', 'cutlass', 'xmma', 'tf32', 'sgemm', 'cudnn')
print('GEMM / attention kernels (count, name):')
for name, n in sorted(kernels.items()):
  if any(k in name.lower() for k in keys):
    print(f'  {n:4d}  {name[:160]}')
print('kernel launches in total:', sum(kernels.values()))
