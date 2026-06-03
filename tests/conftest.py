"""Shared fixtures for the HBFM (HyperbolicBoundaryFM) test suite.

These tests are written AGAINST the interface contract in
`experiments/hbfm/ARCH.md` (Section 9, items I1-I12) BEFORE the implementation
exists.  They are expected to FAIL with import / attribute / NotImplemented
errors until the implementer wires up `algo.HyperbolicBoundaryFM`, the
`configs/algo/hbfm.yaml` / `configs/sampler/hbfm.yaml` configs, the `main.py`
registry branch, and `samplers.HBFMSampler`.

Everything here runs fast and CPU-only.  We build the *smallest real config*
(a tiny `sphere-dit` + `SudokuTokenizer`) via Hydra `compose`, exactly the
backbone/data combination the experiment uses, then attach the HBFM algo
fields that `configs/algo/hbfm.yaml` will carry (Section 5).
"""
import os
import sys

import pytest
import torch

# Make the repo root importable (`import algo`, `import geo_bridge`, ...).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

# Keep transformers fully offline; the only network-touching dependency is the
# `metrics.Metrics` PPL tokenizer, which we point at the locally-cached gpt2.
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

# Tiny problem dimensions (CPU-friendly).
VOCAB_SIZE = 12          # SudokuTokenizer vocab
SMALL_D = 8              # general-d path
BINARY_D = 2             # d == 2 closed-form path
SEQ_LEN = 4              # model.length
BATCH = 2
HBFM_T_MIN = 0.01
HBFM_T_MAX = 2.0         # d=2 smoke range (ARCH 5.1 / EXPERIMENT 6.1)


def _compose_config(hidden_size, sampler_predictor='sfm'):
  """Compose a tiny sphere-dit + sudoku config and attach the HBFM algo block.

  We start from `algo=sfm` (which shares every base field with the future
  `hbfm.yaml`) and overlay the HBFM-specific fields from ARCH Section 5.1 so
  the algo object can read `config.algo.hbfm_t_min`, etc.  `sampler` defaults
  to `sfm` so the algo constructor's `get_sampler(config)` call works even
  before `HBFMSampler` is registered; sampler-specific tests override it.

  `n_heads` is chosen so that `head_dim = hidden_size / n_heads >= 2`. The
  rotary fallback (`models/dit.py:27`) asserts `ro_dim <= head_dim`, which
  fails at `head_dim == 1`. The d=2 backbone forward therefore REQUIRES
  `n_heads=1` (head_dim=2); the d=8 path is fine with either, and n_heads=1 is
  harmless there too. This is a backbone-fixture concern, not an HBFM-geometry
  one -- the d=2 bridge geometry is independent of attention head layout.
  """
  from hydra import compose, initialize_config_dir
  from omegaconf import OmegaConf

  cfg_dir = os.path.join(REPO_ROOT, 'configs')
  n_heads = 1 if hidden_size <= 2 else 2
  overrides = [
    'model=tiny-sphere-dit',
    'noise=log-linear',
    f'sampler={sampler_predictor}',
    'algo=sfm',
    'data=sudoku',
    f'model.hidden_size={hidden_size}',
    f'model.length={SEQ_LEN}',
    f'model.n_heads={n_heads}',
    'model.n_blocks=1',
    'model.cond_dim=16',
    'trainer.devices=1',
    'trainer.num_nodes=1',
    'loader.global_batch_size=2',
    # locally-cached tokenizer so metrics.Metrics never hits the network.
    'eval.gen_ppl_eval_model_name_or_path=gpt2',
  ]
  with initialize_config_dir(version_base=None, config_dir=cfg_dir):
    cfg = compose(config_name='config', overrides=overrides)

  # Overlay the HBFM algo fields (ARCH 5.1). These will live in
  # configs/algo/hbfm.yaml once it exists; we set them here so the algo
  # constructor and methods can read them in this pre-implementation phase.
  # Open struct mode so new keys can be added to the composed config.
  OmegaConf.set_struct(cfg, False)
  cfg.algo.name = 'hbfm'
  cfg.algo.hbfm_t_min = HBFM_T_MIN
  cfg.algo.hbfm_t_max = HBFM_T_MAX
  cfg.algo.bridge_dim = None          # null -> config.model.hidden_size
  cfg.algo.input_repr = 'A'
  cfg.algo.proposal_type = 'unif'
  cfg.algo.weighted_ce = False
  cfg.algo.hbfm_log_qxt_time = False
  return cfg


@pytest.fixture
def tokenizer():
  import dataloader
  return dataloader.SudokuTokenizer()


@pytest.fixture
def config_d8():
  """Tiny general-d (d=8) HBFM config."""
  return _compose_config(SMALL_D)


@pytest.fixture
def config_d2():
  """Tiny binary (d=2) HBFM config -> closed-form bridge path."""
  return _compose_config(BINARY_D)


def _make_hbfm(config, tokenizer):
  import algo
  return algo.HyperbolicBoundaryFM(config, tokenizer)


@pytest.fixture
def hbfm_d8(config_d8, tokenizer):
  return _make_hbfm(config_d8, tokenizer)


@pytest.fixture
def hbfm_d2(config_d2, tokenizer):
  return _make_hbfm(config_d2, tokenizer)


@pytest.fixture
def x0():
  """A clean target batch of token ids, shape (B, L)."""
  torch.manual_seed(0)
  return torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN), dtype=torch.long)


@pytest.fixture
def valid_tokens():
  return torch.ones(BATCH, SEQ_LEN, dtype=torch.float32)
