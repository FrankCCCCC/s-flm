"""Phase-4 dataloader contract tests for the TinyStories integration.

These pin the contract from `experiments/tinystories/ARCH.md` (§3 return
contract, §7 test architecture) and EXPERIMENT success criteria 1-4 for the
`tinystories` dataset branch in `get_dataset`.

Conventions mirror `tests/test_hflm_dispatch.py`: `from conftest import
REPO_ROOT` puts the repo root on `sys.path`, and the config is built inline with
`omegaconf` rather than via hydra.

Network + cost: every test downloads `roneneldan/TinyStories` from HF Hub on a
cold cache, and the `train` tests tokenize the FULL train split (~2.1M stories)
before the first assertion -- minutes per run, with a multi-hundred-MB `.dat`
cache written under a throwaway temp dir. The whole module is therefore one
slow/network bucket via the module-level `pytestmark` below, and is intended to
be run locally / skipped in CI. (The fast 1%-slice `tinystories-debug` variant
that previously backed these tests has been removed; only the full `tinystories`
branch remains.)
"""
import os

import pytest

import dataloader
import utils
from conftest import REPO_ROOT  # noqa: F401

# ARCH §7.1: the whole file hits the network; one slow/network bucket.
pytestmark = pytest.mark.network

# ARCH §7.2: block_size 256 (not 1024). Geometry assertions are block_size-
# agnostic; 256 yields >1 block + interior EOS yet is cheaper to group.
BLOCK_SIZE = 256


def _data_config(*, name, cache_dir, block_size=BLOCK_SIZE,
                 wrap=True, streaming=False,
                 insert_train_eos=True, insert_valid_eos=True):
  """Minimal omegaconf config exposing only the fields get_dataset reads.

  Mirrors ARCH §7.4. `loader.num_workers` (read at dataloader.py:693) drives
  `.map()` parallelism; set to 8 here so tokenizing the full train split
  completes in a tolerable time.
  """
  import omegaconf
  cfg = {
    'data': {
      'train': name,
      'valid': name,
      'tokenizer_name_or_path': 'gpt2',
      'cache_dir': str(cache_dir),
      'wrap': wrap,
      'streaming': streaming,
      'insert_train_eos': insert_train_eos,
      'insert_valid_eos': insert_valid_eos,
    },
    'model': {'length': block_size},
    'loader': {'num_workers': 8},
  }
  return omegaconf.OmegaConf.create(cfg)


@pytest.fixture(scope='module')
def cache_dir(tmp_path_factory):
  """Module-scoped throwaway cache dir; never the repo data_cache (ARCH §7.3)."""
  return tmp_path_factory.mktemp('tinystories_cache')


@pytest.fixture(scope='module')
def tokenizer(cache_dir):
  """Production gpt2 tokenizer (BOS=EOS=50256 via BertProcessing, ARCH §7.5)."""
  cfg = _data_config(name='tinystories', cache_dir=cache_dir)
  return dataloader.get_tokenizer(cfg)


# ---------------------------------------------------------------------------
# Criterion 1: loads & returns a non-empty torch dataset, both modes.
# ---------------------------------------------------------------------------

def test_tinystories_train_loads_torch(cache_dir, tokenizer):
  """Criterion 1: `get_dataset(..., 'train')` returns a non-empty dataset with
  an `input_ids` column and no exception."""
  cfg = _data_config(name='tinystories', cache_dir=cache_dir)
  ds = dataloader.get_dataset(cfg, tokenizer, 'train')
  assert len(ds) > 0
  assert 'input_ids' in ds.column_names


def test_tinystories_valid_loads_torch(cache_dir, tokenizer):
  """Criterion 1 + split mapping: `mode='valid'` selects the `validation` split
  of the DatasetDict and loads without KeyError (ARCH §5, EXPERIMENT Risk 5)."""
  cfg = _data_config(name='tinystories', cache_dir=cache_dir)
  ds = dataloader.get_dataset(cfg, tokenizer, 'valid')
  assert len(ds) > 0


# ---------------------------------------------------------------------------
# Criterion 2: cache `.dat` is flat and slash-free under cache_dir.
# ---------------------------------------------------------------------------

def test_cache_path_flat_and_slashfree(cache_dir, tokenizer):
  """Criterion 2: after a non-streaming call, the `.dat` cache is created
  directly under cache_dir with a slash-free leaf name (guards EXPERIMENT
  Failure-mode 1: the friendly name keeps `roneneldan/TinyStories` out of the
  filename). Filename mirrors dataloader.py:705."""
  cfg = _data_config(name='tinystories', cache_dir=cache_dir)
  dataloader.get_dataset(cfg, tokenizer, 'train')

  filename = f'tinystories_train_bs{BLOCK_SIZE}_wrapped.dat'
  full_path = os.path.join(str(cache_dir), filename)
  # The cache must land at a flat path directly under cache_dir. If the slashed
  # repo id `roneneldan/TinyStories` leaked into dataset_name, the `.dat` would
  # instead nest under a `roneneldan/` subdir (EXPERIMENT Failure-mode 1).
  assert utils.fsspec_exists(full_path)
  assert not os.path.isdir(os.path.join(str(cache_dir), 'roneneldan'))


# ---------------------------------------------------------------------------
# Criterion 3: block geometry (shape == block_size, BOS first, EOS last).
# ---------------------------------------------------------------------------

def test_block_shape_and_bos_eos(cache_dir, tokenizer):
  """Criterion 3: each block is exactly `block_size` long, starts with BOS and
  ends with EOS (frame tokens added by `_group_texts`, ARCH §3 / §9)."""
  cfg = _data_config(name='tinystories', cache_dir=cache_dir)
  ds = dataloader.get_dataset(cfg, tokenizer, 'train')
  ex = ds[0]
  assert tuple(ex['input_ids'].shape) == (BLOCK_SIZE,)
  assert tuple(ex['attention_mask'].shape) == (BLOCK_SIZE,)
  assert int(ex['input_ids'][0]) == tokenizer.bos_token_id
  assert int(ex['input_ids'][-1]) == tokenizer.eos_token_id


# ---------------------------------------------------------------------------
# Criterion 4: EOS insertion present when enabled (insert_train_eos=True).
# ---------------------------------------------------------------------------

def test_eos_insertion_present_when_enabled(cache_dir, tokenizer):
  """Criterion 4: with `insert_train_eos=True` (default), a block built from
  short stories contains interior story-boundary EOS tokens.

  BOS==EOS==50256 collision (EXPERIMENT Failure-mode 6 / ARCH §7.6 #5):
  `_group_texts` frames every block as `[BOS] ... [EOS]`, both id 50256, so the
  count of id-50256 tokens already includes the 2 frame tokens. The assertion is
  therefore `> 2` (more than the two frames => at least one interior boundary),
  never an exact equality.

  Only the eos=True path is tested (EXPERIMENT TBD-4 RESOLVED). Setting
  `insert_train_eos=False` would flip `eos_tag` -> a distinct
  `..._wrapped_eosFalse.dat` cache file; no separate eosFalse test is written.
  """
  cfg = _data_config(name='tinystories', cache_dir=cache_dir)
  ds = dataloader.get_dataset(cfg, tokenizer, 'train')
  ex = ds[0]
  eos_count = int((ex['input_ids'] == tokenizer.eos_token_id).sum())
  assert eos_count > 2


# ---------------------------------------------------------------------------
# Criterion 3 parity: train and valid traverse the identical code path.
# ---------------------------------------------------------------------------

def test_parity_train_valid_schema(cache_dir, tokenizer):
  """Criterion 3 parity (ARCH §7.6 #6, EXPERIMENT TBD-3 RESOLVED: code-path
  parity, no logged baseline): `tinystories` train and valid share identical
  `column_names` and per-example `input_ids` length, confirming both ride the
  same generic post-dispatch tail."""
  cfg = _data_config(name='tinystories', cache_dir=cache_dir)
  train_ds = dataloader.get_dataset(cfg, tokenizer, 'train')
  valid_ds = dataloader.get_dataset(cfg, tokenizer, 'valid')
  assert train_ds.column_names == valid_ds.column_names
  assert len(train_ds[0]['input_ids']) == len(valid_ds[0]['input_ids'])
