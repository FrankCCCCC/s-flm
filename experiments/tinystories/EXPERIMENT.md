# EXPERIMENT: TinyStories dataset integration

> **Update (2026-06-09):** the `tinystories-debug` 1%-slice variant described in earlier revisions has been removed; only the full `tinystories` dataset remains. Sections below have been revised accordingly.

**Type:** Integration / plumbing (not a novel-modeling experiment).
**Direction:** Wire `roneneldan/TinyStories` into the existing `dataloader.py`
pipeline so it loads, tokenizes, caches, and feeds training exactly like the
other small HF text datasets (`wikitext2`, `ag_news`). The "experiment" is the
question: *does TinyStories pass cleanly through the existing pipeline,
end-to-end, with correct shapes/caching/EOS behavior and a finite, decreasing
loss on a short train run?*

This spec defines **what success looks like and what must be measured.** It does
not contain training code. The architect/implementer will add the two changes
in the Variables section.

---

## Hypothesis / goal

Falsifiable functional claim:

> Adding (A) a `tinystories` branch to `get_dataset()`, plus (B) a config file,
> is sufficient for the existing
> pipeline to produce torch-formatted, correctly-shaped, EOS-terminated training
> and validation datasets and to train for a few steps with finite, non-diverging
> loss — **with zero changes to any other module** (`get_tokenizer`,
> `get_dataloaders`, `_group_texts`, `trainer_base`, `main`, `algo`).

Refuted if any of: get_dataset errors on either mode; cache path contains a
`/`; block shapes are wrong; EOS insertion is inconsistent with `insert_eos`;
the smoke train run errors, NaNs, or diverges; or any module besides
`dataloader.py` + the new YAML must be touched to make it work.

---

## Variables (what changes vs. what is held fixed)

### Changes (the implementation — two edits, verbatim intent)

- **(A) `dataloader.py` — new branch in `get_dataset()`** (insert in the
  if/elif chain, currently `dataloader.py:719-810`, e.g. next to the other
  plain-text HF datasets):

  ```python
  elif dataset_name == 'tinystories':
    dataset = datasets.load_dataset(
      'roneneldan/TinyStories',
      cache_dir=cache_dir,
      streaming=streaming,
      revision=revision)
  ```

  The friendly, slash-free name `tinystories` is the cache key (the HF repo id
  `roneneldan/TinyStories` contains a `/`, which would corrupt the flat cache
  filename built at `dataloader.py:705-708` — see Risks).

- **(B) One config file** modeled on `configs/data/wikitext2.yaml`:

  `configs/data/tinystories.yaml`
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
  `RESOLVED (TBD-2): cache_dir = relative `data_cache` (user decision). The shared
  default /share/kuleshov/ssahoo/textdiffusion/data does NOT exist on this
  machine. `data_cache` is repo-relative, user-writable, and should be gitignored.`

### Held fixed (must NOT change to make this work)

- Tokenizer: `gpt2` (verified compatible — adds BOS/EOS via `BertProcessing` in
  `get_tokenizer`, `dataloader.py:970-974`).
- `wrap: True` (the convention for every text config here; exercises
  `_group_texts`).
- `block_size = config.model.length` (= **1024** for `configs/model/small.yaml`).
- Split mapping `valid -> 'validation'` (`dataloader.py:696-699`): TinyStories
  has exactly `train` and `validation` splits, so the generic non-special
  mapping is correct — `tinystories` must NOT be added to the
  `['text8','lm1b','ag_news']` special-split list.
- Detokenizer (`else -> None`, `dataloader.py:831-832`) and column removal
  (`else -> remove 'text'`, `dataloader.py:897-899`): TinyStories is a single
  clean `text` column, so both `else` branches are already correct — do NOT add
  `tinystories` to the `ptb` / `scientific_papers` / `ag_news` special cases.
- `preprocess_and_tokenize` text-field selection (`else -> example['text']`,
  `dataloader.py:852-853`): correct as-is.

---

## Baselines / reference (parity, not research metrics)

Confirm TinyStories produces structurally identical output to an existing small
text dataset run through the same code path.

- **Primary parity baseline: `wikitext2`** (`configs/data/wikitext2.yaml`).
  Same tokenizer (gpt2), same `wrap: True`, same generic non-special path,
  same `else`-branch detokenizer/column handling. The expectation is *byte-for-
  byte identical dataset schema and block geometry* (only the row count differs).
- **Secondary parity reference: `ag_news`** (`configs/data/ag_news.yaml`) — also
  single `text`-ish column, confirms the non-special tokenize path.
- No prior TinyStories runs exist in this repo. `TBD-3: no W&B baseline run IDs
  exist for any of these on TinyStories; parity is checked against the *code
  path*, not a logged run. Confirm that is acceptable (it should be for plumbing).`

---

## Success criteria (concrete, checkable)

All five must pass. Reference values assume `model.length = 1024`,
`tokenizer = gpt2` (vocab/BOS=EOS id `50256`).

1. **Loads & returns torch dataset, both modes.**
   `get_dataset(config, tokenizer, mode)` returns a non-empty,
   `.with_format('torch')` dataset for `tinystories × {train, valid}`.
   No exception.

2. **Cache path is flat and slash-free.** After a (non-streaming) call, a single
   `.dat` directory is created directly under `cache_dir` with name
   `tinystories_train_bs1024_wrapped.dat` (and the `valid` analogue).
   Assert the relative path under `cache_dir` contains **no `/`** before the
   `.dat` suffix and that the file/dir exists (`utils.fsspec_exists`).

3. **Block geometry correct.** For `wrap: True`, each example's
   `input_ids` has length **exactly `block_size` (1024)** and `attention_mask`
   has length 1024 (per `_group_texts`, `dataloader.py:656-679`).
   Every block starts with BOS (`input_ids[0] == tokenizer.bos_token_id`) and
   ends with EOS (`input_ids[-1] == tokenizer.eos_token_id`).

4. **EOS-insertion toggle behaves per `insert_eos`.** With
   `insert_train_eos: True`, per-document EOS separators are present in the
   concatenated stream (a `tinystories` block built from short stories contains
   `>1` occurrence of `eos_token_id`, i.e. interior story boundaries, not just
   the trailing block EOS). Re-running with `insert_train_eos: False` (which
   changes the cache filename via `eos_tag='_eosFalse'`, `dataloader.py:702-707`)
   produces a *different* cache file and fewer interior EOS tokens.
   `TBD-4: is the eosFalse variant in-scope for the test, or assert the default
   eos=True path only? Default proposal: test eos=True only, note the toggle.`

5. **End-to-end smoke train run.** A short run completes N steps with **finite,
   non-NaN loss that is lower at step N than at step 0** (not a strict
   monotonicity claim — just net decrease on the logged `trainer/loss`).
   - Vehicle: `main.py` with `data=tinystories`, `model=small` (or a
     smaller model if 1024-length small is too heavy on the smoke node), tiny
     batch, `trainer.max_steps=N`, `wandb.project` per naming below.
   - `TBD-5: N (proposed: 50 steps). Confirm step count.`
   - `TBD-6: run a REAL training step loop via main.py (criterion 5 as written),
     OR is a dataloader-only smoke (iterate one batch, check shape/dtype/finite,
     no model fwd/bwd) sufficient? main.py exercises strictly more, but pulls in
     DDP/EMA/gpt2-large gen-ppl download. Default proposal: do the dataloader
     smoke AS A TEST (Phase 4) + ONE short main.py run as a manual gate (Phase 7).`

---

## Ablations (minimal — isolate the mechanism)

This is plumbing, so "ablations" = the smallest set of axes that isolate where a
failure comes from:

- **Mode:** `train` vs. `valid` — isolates the `valid -> 'validation'` split
  mapping and the train/valid `insert_eos` config fields.
- **Parity dataset:** `tinystories` vs. `wikitext2` through the identical code
  path — isolates "is this a TinyStories problem or a pipeline problem?"

Held single-valued (NOT ablated, to keep it sharp): `wrap=True`,
`block_size=1024`, tokenizer=gpt2, `streaming=False`.

---

## Metrics (W&B keys)

This is integration, so most "metrics" are pass/fail assertions, not curves.

- **Primary (assertions, Phase 4 tests):**
  - dataset returns without exception (both modes) — boolean.
  - `len(dataset) > 0` — count.
  - per-example `input_ids.shape == (1024,)`, `attention_mask.shape == (1024,)`.
  - BOS-first / EOS-last per block — boolean.
  - cache `.dat` exists and is slash-free under `cache_dir` — boolean + path.
- **Secondary (smoke run, Phase 7, logged to W&B by existing `main.py`):**
  - `trainer/loss` (or whatever the existing Lightning module logs — `TBD-7:
    confirm exact W&B loss key; grep `self.log` in `trainer_base.py` /
    `algo.py`). Check: finite at every logged step; `loss[N] < loss[0]`.
  - `trainer/global_step` reaches N.
- **Diagnostic (manual, one-off, not logged):**
  - tokens/example histogram or mean tokens-per-story vs. wikitext2 (sanity that
    TinyStories docs are short, many interior EOS) — print, do not gate on it.
  - wall-clock + cache size on disk.

---

## Compute budget (tiny)

| Run | Where | Est. wall time | Est. GPU-h |
|-----|-------|----------------|------------|
| Phase 4 dataloader tests, full `tinystories` (CPU) | 1 CPU | minutes (download + tokenize the full split) | 0 |
| Phase 4 parity check vs `wikitext2` (cached) | 1 CPU | < 1 min if cached | 0 |
| One-time full `tinystories` load + tokenize + cache to disk | 1 CPU, `num_workers=12` | ~minutes–low tens of min (TinyStories ≈ 2GB raw text) | 0 |
| Phase 7 smoke train, `tinystories`, N≈50 steps | 1 GPU (`gpu` partition has idle nodes; e.g. `sablab-gpu-05`) | < 5 min | < 0.1 |

**Total: well under 1 GPU-hour, < 1 CPU-hour.** No multi-GPU, no multi-seed,
no hyperparameter sweep — this is a correctness gate, not a measurement.
`sinfo` confirms idle GPU nodes on the `gpu` / `default_partition` partitions.

---

## Validation plan

### Phase 4 — automated tests (pytest, under `tests/`)

Add `tests/test_tinystories_dataloader.py`. Follow existing conventions:
`from conftest import REPO_ROOT` (puts repo root on `sys.path`, see
`tests/conftest.py`), build a minimal `omegaconf` config like
`test_hflm_dispatch.py::_sampler_config` rather than loading full hydra.
Tests target the full `tinystories` dataset (they download + tokenize the full
split on first run; cached thereafter).
`TBD-8: tests hit the network (HF download) on first run — acceptable for an
integration test, or must we vendor a tiny fixture / mark @pytest.mark.network
and skip in CI? Default proposal: mark it network/slow, allow local run.`

Tests assert success criteria 1–4:
- `test_tinystories_train_loads_torch` (criterion 1)
- `test_tinystories_valid_loads_torch` (criterion 1, split mapping)
- `test_cache_path_flat_and_slashfree` (criterion 2)
- `test_block_shape_and_bos_eos` (criterion 3 — shape 1024, BOS first, EOS last)
- `test_eos_insertion_present_when_enabled` (criterion 4)
- `test_parity_with_wikitext2_schema` (same `column_names`, same per-example
  `input_ids` length 1024 — confirms identical code path)

Run: `pytest tests/test_tinystories_dataloader.py -v` from repo root.

### Phase 7 — end-to-end smoke train run (manual gate)

Invoke the existing entrypoint (do not write new training code):

```
python main.py \
  data=tinystories \
  model=small \
  trainer.max_steps=50 \
  loader.global_batch_size=<small> \
  wandb.project=tinystories-integration \
  mode=train
```

`TBD-9: exact batch size / whether to shrink model.length for the smoke node —
small@1024 may be heavy. Default proposal: keep small, drop global_batch_size to
fit one GPU; reduce model.length only if OOM.`

Pass = process exits 0, `trainer/loss` is finite at all logged steps and
`loss[50] < loss[0]`, W&B run appears under the project below.

---

## Failure modes to watch for (what invalidates the experiment)

1. **Slash in cache name (mitigated).** Using the raw repo id
   `roneneldan/TinyStories` as `dataset_name` would make `filename` at
   `dataloader.py:705-708` contain a `/`, producing a nested/garbage path or an
   `os.path` mishap. **Mitigation: friendly name `tinystories`.** The test in
   criterion 2 guards this.
2. **Stale cache keyed on `block_size`/`eos`.** The `.dat` cache filename encodes
   `bs{block_size}` and `eos_tag`. Changing `model.length` or `insert_eos` after
   a cached run silently uses the cache only when the key matches; a key change
   forces a re-tokenize. Risk: a test that changes `block_size` but reuses an old
   cache, or a test polluted by a prior run's cache. **Mitigation:** tests
   either assert on a freshly-computed cache or clean the specific `.dat` first;
   never assume a clean cache_dir.
3. **Streaming vs. `save_to_disk` incompatibility.** Streaming datasets have no
   `len()` and skip `save_to_disk`, so the caching path requires
   `streaming=False`. **Mitigation:** `tinystories.yaml` pins `streaming: False`
   (enforced above); a test could assert the config has `streaming == False`.
4. **DatasetDict indexing.** With no `split=` argument, `load_dataset` returns a
   `DatasetDict`; the non-special path does `dataset[hf_split]`
   (`dataloader.py:816`) to pick the `train` / `validation` member.
   **Mitigation:** `tinystories` stays on the `else` (`data = dataset[hf_split]`)
   branch — it must NOT be added to the bare-dataset `data = dataset` list at
   `dataloader.py:812-814`.
5. **Wrong split for valid.** If `tinystories` were mistakenly added to the
   `['text8','lm1b','ag_news']` special-split list, `valid` would map to a
   non-existent `test` split. **Mitigation:** leave it on the generic
   `validation` mapping; criterion 1 (valid mode) catches a regression.
6. **gpt2 padding/special-token assumptions.** `wrap=True` relies on EOS/BOS
   ids; gpt2 sets BOS=EOS=50256 via `BertProcessing`. Not a data-leakage risk,
   but a block whose interior EOS count is tested (criterion 4) must account for
   BOS==EOS id collision when counting. **Mitigation:** count occurrences in the
   *interior* of the block, and treat the criterion as "more than the 2 frame
   tokens" rather than an exact count.
7. **No real eval contamination risk** (this is plumbing, no held-out research
   metric). The only "leakage"-shaped concern is reusing a cache built from a
   different slice/split — covered by (2).

---

## W&B project / run naming convention (exact strings)

- **Project:** `tinystories-integration`
- **Run name (smoke):** `tinystories_small_smoke50` (pattern:
  `{data.train}_{model}_smoke{max_steps}`).
- **Group:** `dataset-integration`
- **Tags:** the repo's default tag list already injects
  `${data.train}`, `${data.valid}`, `${algo.name}`, `${noise.type}`
  (see `configs/config.yaml:98-102`); add tag `integration`.
- Phase 4 pytest does **not** log to W&B (offline assertions).

`TBD-10: confirm the W&B entity/team; default config uses project `debug` with
no entity pin — override only `wandb.project` for the smoke run unless an entity
is required.`

---

## Resolved decisions (all TBDs closed — ready to build)

- **TBD-1 — MOOT:** the original 1% debug-slice decision no longer applies; the
  `tinystories-debug` variant has been removed and only the full `tinystories`
  dataset remains (see the dated note at the top of this file).
- **TBD-2 — RESOLVED:** `cache_dir = data_cache` (repo-relative; the shared default
  does not exist on this machine). Add `data_cache/` to `.gitignore`.
- **TBD-3 — RESOLVED:** code-path parity vs. `wikitext2` is acceptable (plumbing,
  no logged TinyStories baseline run needed).
- **TBD-4 — RESOLVED:** test the `insert_eos=True` default path only; note the
  toggle in a comment, no separate `eosFalse` test.
- **TBD-5 — RESOLVED:** smoke-run step count `N = 50`.
- **TBD-6 — RESOLVED:** Phase 7 = dataloader-only smoke AS A TEST (Phase 4) **plus**
  ONE short `main.py` train run (~50 steps, 1 GPU) as a manual gate.
- **TBD-7 — RESOLVED:** W&B loss key = `trainer/loss` (train; logged at
  `trainer_base.py:332`). Val monitor is `val/loss`. Check finite + `loss[N] < loss[0]`.
- **TBD-8 — RESOLVED:** network-hitting tests are acceptable; mark them
  `@pytest.mark.network`/slow so they run locally but can be skipped in CI.
- **TBD-9 — RESOLVED:** keep `model=small`; drop `loader.global_batch_size` to fit
  one GPU; reduce `model.length` only if OOM.
- **TBD-10 — RESOLVED:** override only `wandb.project=tinystories-integration`; no
  entity pin.
