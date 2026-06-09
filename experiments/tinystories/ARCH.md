# ARCH: TinyStories dataset integration

> **Update (2026-06-09):** the `tinystories-debug` 1%-slice variant described in earlier revisions has been removed; only the full `tinystories` dataset remains. Sections below have been revised accordingly.

Architecture contract for the TinyStories plumbing in `EXPERIMENT.md`. The
test-writer (Phase 4) and implementer (Phase 5) both target this document
verbatim. All 10 EXPERIMENT TBDs are RESOLVED; this spec honors those decisions.

All paths are relative to repo root
`/share/thickstun/sychou/workspace/research/s-flm-dev1/s-flm`.

---

## 1. Scope

**In:**
1. `dataloader.py`: add one `elif` branch to `get_dataset()` (`tinystories`).
   `tinystories` is a `DatasetDict` indexed by `hf_split`; it stays OFF the
   bare-`Dataset` list (the `data = dataset` branch).
2. New `configs/data/tinystories.yaml`.
3. New `tests/test_tinystories_dataloader.py`.
4. `.gitignore`: ensure `data_cache/` is ignored.

**Out (explicitly NOT touched — refuting the hypothesis if any are needed):**
`get_tokenizer`, `get_dataloaders`, `_group_texts`, `trainer_base`, `main`,
`algo`, any `models/*`, any `configs/model/*`, `configs/config.yaml`.
The bare-`Dataset` list itself (`data = dataset` branch) is also unchanged —
`tinystories` is a `DatasetDict` and rides the `else` path.
No new detokenizer, no new split-mapping entry, no new column-removal branch,
no text-field special-case. TinyStories rides entirely on the existing generic
(`else`) paths for split mapping, detokenizer, text field, and column removal.

**Note on `.gitignore` (item 5):** `data_cache/` is ALREADY present at
`.gitignore:175`. Verify it is there; if so this is a no-op and the implementer
makes zero edits to `.gitignore`. Do NOT add a duplicate entry.

---

## 2. Module layout

| File | Action | What |
|------|--------|------|
| `dataloader.py` | modify | 1 new `elif` in `get_dataset()` dispatch |
| `configs/data/tinystories.yaml` | create | full-dataset config |
| `tests/test_tinystories_dataloader.py` | create | Phase-4 assertions for criteria 1–4 |
| `.gitignore` | verify only | `data_cache/` already at line 175 |

---

## 3. Interface contract (target for Phase 4 + Phase 5)

### `get_dataset(config, tokenizer, mode) -> datasets.Dataset`

Unchanged signature. Reads these config fields (all already read today; no new
field introduced):

- `config.data.train` / `config.data.valid` — the dataset name, selected by `mode`.
- `config.data.insert_train_eos` / `config.data.insert_valid_eos` — EOS toggle.
- `config.data.wrap` — block-packing toggle (`True` for the new config).
- `config.data.cache_dir` — `data_cache` (repo-relative) for the new config.
- `config.data.streaming` — `False` for the new config.
- `config.model.length` — `block_size`; cache key encodes `bs{block_size}`.
- `config.loader.num_workers` — `.map()` parallelism (default `12` in
  `configs/config.yaml:38`); the test config MUST set this (use a small value,
  e.g. `1`, for hermetic fast tests — see §7).

**Return contract — for both modes** (`tinystories × {train, valid}`)
with `wrap=True`:

A torch-formatted HF `Dataset` (`.with_format('torch')`) whose every example has:
- `input_ids`: 1-D tensor of length **exactly `block_size`**
  (`= config.model.length`; `1024` under `model=small`).
- `attention_mask`: 1-D tensor of length **exactly `block_size`** (all ones).
- `input_ids[0] == tokenizer.bos_token_id` and
  `input_ids[-1] == tokenizer.eos_token_id` (frame tokens added by
  `_group_texts`, `dataloader.py:673-675`).
- `len(dataset) > 0`.

The new branch only produces/returns the raw `dataset` object; the existing
post-dispatch code (`dataloader.py:812-921`) performs split selection,
tokenization, grouping, caching, and `.with_format('torch')`. The new branch
MUST NOT call `.with_format`, `.remove_columns`, or `save_to_disk` itself —
that all happens in the shared tail.

---

## 4. Exact diff intent — `get_dataset()`

### 4.1 Branch: `tinystories` (full DatasetDict)

**Insertion point:** inside the `if/elif` dispatch chain (`dataloader.py:719-810`),
immediately AFTER the `elif dataset_name == 'ag_news':` block
(ends `dataloader.py:788`) and BEFORE `elif dataset_name == 'synthetic':`
(`dataloader.py:789`). Placement among plain-text HF datasets is conventional;
exact ordinal position is non-load-bearing as long as it is a sibling `elif` in
this chain.

```python
elif dataset_name == 'tinystories':
  dataset = datasets.load_dataset(
    'roneneldan/TinyStories',
    cache_dir=cache_dir,
    streaming=streaming,
    revision=revision)
```

- `cache_dir`, `streaming`, `revision` are the locals already bound at
  `dataloader.py:690-694`. `streaming` comes from the config (`False`).
- No `split=` argument -> `load_dataset` returns a **`DatasetDict`** keyed by
  `train` / `validation`.
- `num_proc` is NOT passed here (matches the `ag_news` / `scientific_papers`
  precedent, which also omit it on the load call; `num_proc` is used later in
  `.map()`).
- No `split=` argument -> `load_dataset` returns a **`DatasetDict`**; it stays
  on the `else` (`data = dataset[hf_split]`) branch and is NOT added to the
  bare-`Dataset` list. See §5 for the explicit contract.

### 4.2 What is NOT affected (verify these stay on `else`)

- **Split mapping (`dataloader.py:696-699`):** `tinystories` uses the
  generic `hf_split = 'validation' if mode == 'valid' else mode`, which is
  exactly the TinyStories split layout (`train`, `validation`). Do NOT add
  `tinystories` to `['text8','lm1b','ag_news']`. Do NOT add it to the
  bare-`Dataset` list either — it must be indexed by `hf_split`.
- **`return data.with_format('torch')` early-return (`dataloader.py:817-819`):**
  guarded by `dataset_name in ('synthetic', 'tiny_gsm', 'sudoku')`. `tinystories`
  is not in that tuple, so it does not early-return; it flows through tokenize +
  group + cache. UNAFFECTED.
- **Detokenizer (`dataloader.py:821-832`):** `tinystories` hits `else ->
  detokenizer = None`. Correct (TinyStories needs no detokenization). UNAFFECTED.
- **Text field (`dataloader.py:847-853`):** hits `else -> example['text']`.
  TinyStories' only column is `text`. UNAFFECTED.
- **Column removal (`dataloader.py:888-899`):** hits `else ->
  remove_columns('text')`. UNAFFECTED.
- **Grouping + cache (`dataloader.py:906-921`):** `wrap=True`, `streaming=False`
  -> `_group_texts` runs, `chunked_dataset.save_to_disk(_path)` writes the
  `.dat`, returns `.with_format('torch')`. UNAFFECTED.

The ONLY edit is §4.1. Anything else is out of scope.

---

## 5. DatasetDict contract (why `tinystories` stays off the bare-`Dataset` list)

Make it impossible to get wrong:

| Name | `load_dataset` call | Return type | Post-dispatch needs | On bare-list? |
|------|---------------------|-------------|---------------------|---------------|
| `tinystories` | no `split=` | **`DatasetDict`** (`{train, validation}`) | `data = dataset[hf_split]` to pick the split | **NO** |

Rule: with no `split=` argument, `load_dataset` returns a `DatasetDict`.
Full `tinystories` returns such a dict; it MUST be indexed by `hf_split`, so it
belongs on the `else` (`data = dataset[hf_split]`) branch — i.e. NOT on the
bare-`Dataset` list. If it were mistakenly added to that list, `data = dataset`
would hand a `DatasetDict` (not a `Dataset`) to tokenization and fail. The
criterion-1 tests (both modes) catch a regression.

---

## 6. Config file contents (verbatim)

### `configs/data/tinystories.yaml`

```yaml
train: tinystories
valid: tinystories
tokenizer_name_or_path: gpt2
cache_dir: data_cache
wrap: True
streaming: False
insert_train_eos: True
insert_valid_eos: True
```

It mirrors `configs/data/wikitext2.yaml` field-for-field; the only deltas are
the `train`/`valid` names and `cache_dir` (`data_cache` instead of the absent
shared `/share/...` path, per EXPERIMENT TBD-2 RESOLVED). Field key order matches
the template. This is a leaf data-group config consumed by hydra as
`data=tinystories`.

---

## 7. Test architecture — `tests/test_tinystories_dataloader.py`

### 7.1 Conventions (match repo)

- `from conftest import REPO_ROOT  # noqa: F401` to put repo root on `sys.path`
  (same as `test_hflm_dispatch.py:23`).
- Build a minimal `omegaconf` config inline (mirror
  `test_hflm_dispatch.py::_sampler_config`); do NOT load full hydra.
- Import `dataloader`; build the tokenizer with `dataloader.get_tokenizer(cfg)`
  (drives the gpt2 + `BertProcessing` BOS/EOS=50256 setup at
  `dataloader.py:970-974`) so tests use the exact production tokenizer path.
- All network/download-hitting tests carry `@pytest.mark.network` (EXPERIMENT
  TBD-8 RESOLVED: local-runnable, CI-skippable). Tests target the full
  `tinystories` dataset, so first run downloads + tokenizes the full split.
  Register the marker in
  `tests/conftest.py` via `pytest_configure` OR rely on a top-level
  `pytestmark = pytest.mark.network` if every test in the file hits the network
  (it does). **Decision:** set module-level `pytestmark = pytest.mark.network`
  so the whole file is one slow/network bucket; no per-function decorator
  needed. (Avoids `PytestUnknownMarkWarning` only if registered; the
  implementer should add a one-line `markers` registration — see §10 open
  question, non-blocking.)

### 7.2 `block_size` choice — use `256`, NOT 1024

The cache `.dat` is keyed on `bs{block_size}` (`dataloader.py:705`) and
`_group_texts` packs to `block_size`. Tests assert *shape == block_size* and
*BOS-first/EOS-last*, which hold for ANY `block_size`; the value `1024` is only
required for the criterion reference values, not for correctness. A smaller
`block_size` means fewer tokens per block and a smaller `.dat` to tokenize/group
-> faster, cheaper tests. **Use `model.length = 256`.** Justification: the
geometry assertions are block_size-agnostic; 256 is large enough that the
TinyStories split yields `>1` block and interior EOS tokens (criterion 4) yet
4x cheaper than 1024. The Phase-7 smoke run (manual, separate) uses real
`model=small` at 1024; the unit tests do not need 1024.

Because the test block_size (256) differs from the production block_size (1024),
the test `.dat` (`tinystories_train_bs256_wrapped.dat`) is a DISTINCT cache
file from a production run at 1024 — no collision, no stale-cache cross-talk.

### 7.3 Cache-dir hygiene

Tests MUST NOT depend on or pollute the shared `data_cache`. Use a per-test
throwaway cache dir:

- Use pytest's `tmp_path` fixture as `cache_dir` for every test (each test gets a
  fresh empty dir). This guarantees a cold cache (criterion 2 asserts the `.dat`
  is *created*) and zero pollution of `data_cache`.
- Trade-off: each test that uses a fresh `tmp_path` re-downloads/re-tokenizes
  the full split. To keep total runtime low, the implementer MAY share one
  `cache_dir` across the criterion-3/4/parity tests via a
  `scope='module'` fixture that yields a single `tempfile.TemporaryDirectory`,
  while the cache-creation test (criterion 2) uses its own fresh `tmp_path` to
  observe a cold->warm transition. **Decision:** module-scoped temp cache_dir
  fixture `cache_dir` shared by all tests; the criterion-2 test asserts the
  `.dat` exists AFTER its `get_dataset` call (creation observed within the call,
  not relative to an empty start), so sharing is fine. Never write into the repo
  `data_cache` from tests.

### 7.4 Config builder helper

```python
def _data_config(*, name, cache_dir, block_size=256,
                 wrap=True, streaming=False,
                 insert_train_eos=True, insert_valid_eos=True):
  """Minimal omegaconf config exposing only the fields get_dataset reads."""
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
    'loader': {'num_workers': 1},
  }
  return omegaconf.OmegaConf.create(cfg)
```

`loader.num_workers = 1` is REQUIRED (`get_dataset` reads it at
`dataloader.py:693`) and small for hermetic, low-overhead `.map()` in tests.

### 7.5 Fixtures

```python
@pytest.fixture(scope='module')
def cache_dir(tmp_path_factory):
  """Module-scoped throwaway cache dir; never the repo data_cache."""
  return tmp_path_factory.mktemp('tinystories_cache')

@pytest.fixture(scope='module')
def tokenizer(cache_dir):
  """Production gpt2 tokenizer (BOS=EOS=50256 via BertProcessing)."""
  cfg = _data_config(name='tinystories', cache_dir=cache_dir)
  return dataloader.get_tokenizer(cfg)
```

### 7.6 Test functions (names, signatures, assertions)

All map to EXPERIMENT success criteria 1–4. Criterion 5 (smoke train) is the
Phase-7 manual gate, NOT a pytest test (EXPERIMENT TBD-6 RESOLVED).

1. `test_tinystories_train_loads_torch(cache_dir, tokenizer)`
   — **Criterion 1.** `cfg = _data_config('tinystories', cache_dir)`;
   `ds = dataloader.get_dataset(cfg, tokenizer, 'train')`. Assert no exception,
   `len(ds) > 0`, and `'input_ids' in ds.column_names`.

2. `test_tinystories_valid_loads_torch(cache_dir, tokenizer)`
   — **Criterion 1 + split mapping.** Same as above with `mode='valid'`.
   Confirms the `valid -> 'validation'` mapping selects the `validation` member
   of the DatasetDict without `KeyError`. Assert `len(ds) > 0`.

3. `test_cache_path_flat_and_slashfree(cache_dir, tokenizer)`
   — **Criterion 2.** Call `get_dataset(..., 'train')`. Then:
   - Construct the expected filename `tinystories_train_bs256_wrapped.dat`
     (mirror `dataloader.py:705`). Assert `os.path.basename(filename) ==
     filename` (no `/` in the leaf name) and that the relative path under
     `cache_dir` contains no `/` before `.dat`.
   - Assert `utils.fsspec_exists(os.path.join(cache_dir, filename))` is `True`.
     (Import `utils` per repo convention.)
   - Guards EXPERIMENT Failure-mode 1 (slash in `roneneldan/TinyStories`):
     the friendly name `tinystories` keeps the leaf flat.

4. `test_block_shape_and_bos_eos(cache_dir, tokenizer)`
   — **Criterion 3.** `ds = get_dataset(..., 'train')`; `ex = ds[0]`. Assert:
   - `ex['input_ids'].shape == (256,)` (== `block_size`).
   - `ex['attention_mask'].shape == (256,)`.
   - `int(ex['input_ids'][0]) == tokenizer.bos_token_id`.
   - `int(ex['input_ids'][-1]) == tokenizer.eos_token_id`.

5. `test_eos_insertion_present_when_enabled(cache_dir, tokenizer)`
   — **Criterion 4.** With `insert_train_eos=True` (default), `ds = get_dataset(
   ..., 'train')`; `ex = ds[0]`. Count `eos_token_id` occurrences in
   `ex['input_ids']`. Assert **`> 1`** (strictly more than the 2 frame tokens is
   the goal, but BOS==EOS==50256 collision means the leading BOS *also* counts
   as an EOS-id occurrence; see note below). Concretely assert the count is
   `> 2` — leading BOS (id 50256) + trailing EOS (id 50256) = 2 frame
   occurrences, so any interior story-boundary EOS pushes the count `> 2`.
   Test eos=True path ONLY (EXPERIMENT TBD-4 RESOLVED); add a one-line comment
   noting `insert_train_eos=False` would change `eos_tag` -> a different cache
   file. Do NOT write a separate eosFalse test.

   **BOS==EOS collision note (EXPERIMENT Failure-mode 6):** under gpt2,
   `bos_token_id == eos_token_id == 50256`. `_group_texts` adds `[BOS] ...
   [EOS]` = two id-50256 frame tokens per block. So `count(input_ids == 50256)`
   includes both frames. The assertion must be `> 2`, NOT `== <exact>`, and
   reads as "more than the two frame tokens, i.e. at least one interior story
   boundary present."

6. `test_parity_with_tinystories_schema(cache_dir, tokenizer)`
   — **Criterion 3 parity (code-path sanity, EXPERIMENT TBD-3 RESOLVED:
   code-path parity, no logged baseline).** Assert the `train` and `valid`
   `tinystories` datasets share identical `column_names` (`['input_ids',
   'attention_mask']`) and identical per-example `input_ids` length (256).
   This confirms train and valid traverse the same generic post-dispatch path.
   (EXPERIMENT names a wikitext2 parity test; wikitext2's shared `cache_dir`
   default is absent on this machine and would force a wikitext2 download.
   Internal train-vs-valid parity exercises the identical code path with
   no extra dataset dependency — **Decision:** use train-vs-valid parity.
   See §10 open question, non-blocking.)

### 7.7 What tests do NOT do

- The tests download + tokenize the full `tinystories` split on first run
  (cached thereafter); they are marked `@pytest.mark.network` and run at a small
  `block_size` (256) to keep the tokenize/group step cheap. Criterion-1/2/3/4
  are all satisfied via the full `tinystories` dataset, exercising the same
  post-dispatch tail the Phase-7 smoke run uses.
- No model forward/backward, no `main.py`, no W&B. Those are the Phase-7
  manual smoke gate (criterion 5).

---

## 8. Data flow

```
config (data=tinystories, model.length, loader.num_workers)
  -> get_dataset(config, tokenizer, mode)
       mode -> dataset_name, insert_eos               (dataloader.py:683-688)
       cache key f'{name}_{mode}_bs{block_size}_wrapped{eos_tag}.dat'  (705)
       if cached .dat exists: load_from_disk + with_format('torch') -> RETURN (710-712)
       else:
         dispatch:
           tinystories -> load_dataset(...)                  [DatasetDict]   (§4.1)
         split select:
           data = dataset[hf_split]   (hf_split: train|validation)
         detokenizer = None (else)
         preprocess_and_tokenize: text=example['text'], +[EOS] if insert_eos
         remove_columns('text') (else)
         _group_texts(block_size, bos=BOS, eos=EOS): pack to [BOS ... EOS]*block_size
         save_to_disk(_path)
         with_format('torch') -> RETURN
```

State: the only persistent state is the on-disk `.dat` cache under `cache_dir`,
keyed on `name`, `mode`, `block_size`, `wrap`, and `eos_tag`. No in-memory
global state introduced.

---

## 9. Edge cases & invariants

- **Invariant:** every returned example has `input_ids`/`attention_mask` of
  length exactly `block_size`, `input_ids[0]==BOS`, `input_ids[-1]==EOS`
  (guaranteed by `_group_texts`, which is unchanged).
- **Invariant:** the `.dat` leaf name contains no `/` (friendly name
  `tinystories`, never `roneneldan/TinyStories`).
- **Stale cache (EXPERIMENT Risk 2):** cache key includes `bs{block_size}` and
  `eos_tag`. Changing `model.length` or `insert_eos` keys a different `.dat`;
  reusing an old run silently hits a non-matching key only if it matches. Tests
  mitigate via throwaway `tmp_path` cache_dir (§7.3); never assume a clean repo
  `data_cache`.
- **Streaming + `save_to_disk` incompatibility (EXPERIMENT Risk 3):** the config
  pins `streaming: False`. A streaming dataset has no `len()`/`save_to_disk`, so
  the caching path requires `streaming=False`.
- **DatasetDict indexing (EXPERIMENT Risk 4):** see §5 table — `tinystories`
  stays OFF the bare-`Dataset` list and is indexed by `hf_split`.
- **Wrong valid split (EXPERIMENT Risk 5):** `tinystories` stays on the generic
  `validation` mapping; criterion-1 valid test catches a regression.
- **BOS==EOS==50256 (EXPERIMENT Risk 6):** criterion-4 count is `> 2`
  (frame-token aware), never an exact equality. See §7.6 #5.
- **First-run network download:** tests are `@pytest.mark.network`; first run
  downloads the full `tinystories` split from HF Hub. Acceptable locally,
  skippable in CI.
- **NOT handled (explicitly):** `wrap=False` for TinyStories (config pins
  `True`); `streaming=True`; the `insert_eos=False` cache
  variant (noted in a test comment only); any tokenizer other than gpt2.

---

## 10. Open questions

No BLOCKING ambiguities. Two non-blocking implementer choices, each with a
default decided in this spec:

1. **Pytest marker registration (non-blocking).** Using
   `@pytest.mark.network` without registering it emits
   `PytestUnknownMarkWarning`. Default: implementer adds a one-line
   `markers = network: ...` registration to `pytest.ini`/`pyproject`/`conftest`
   `pytest_configure`, OR accepts the warning. Does not affect pass/fail. The
   spec does not require touching CI config.

2. **Parity test target (non-blocking).** EXPERIMENT §"Validation plan" lists
   `test_parity_with_wikitext2_schema`. wikitext2's template `cache_dir`
   (`/share/kuleshov/...`) is absent on this machine, so a true wikitext2 parity
   test would trigger a separate wikitext2 download under a temp cache.
   **Decision (this spec):** implement parity as `tinystories`
   train-vs-valid schema equality (§7.6 #6) — same generic code path, no extra
   dataset dependency. If the user specifically wants a cross-dataset wikitext2
   comparison, that is an additive, non-blocking change.
