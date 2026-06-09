# RESULTS: TinyStories dataset integration (Phase-7 smoke train)

> **Update (2026-06-09):** the `tinystories-debug` variant used for the smoke run below has since been removed from the codebase; the integration now exposes only the full `tinystories` dataset, and the Phase-4 tests were updated to target it. The results below are retained as the historical record of the original validation run (which used the 1% debug slice).

**Verdict: CONFIRMED.**

The hypothesis — *"TinyStories integrates cleanly end-to-end through the existing
pipeline with zero changes to other modules"* — holds. All five success criteria
pass. The dataloader path (criteria 1–4) is validated by the Phase-4 test suite
(6/6, reviewer APPROVE) and independently re-confirmed by the real smoke run. The
Phase-7 smoke-train gate (criterion 5) passes: 10/10 logged `trainer/loss` values
are finite, the loss net-decreased from first to last logged step, and the run
completed all 50 steps with exit status `finished`.

W&B run: https://wandb.ai/syctw/tinystories-integration/runs/1dhq6a1m
(entity `syctw`, project `tinystories-integration`, run `1dhq6a1m`, name
`unique-pond-2`, group `dataset-integration`, state `finished`).

---

## Per-criterion results

| # | Criterion | Evidence | Result |
|---|-----------|----------|--------|
| 1 | Loads & returns torch dataset for `{tinystories, tinystories-debug} × {train, valid}`, no exception | Phase-4 tests `test_tinystories_debug_train_loads_torch`, `test_tinystories_debug_valid_loads_torch` (6/6 passing, reviewer APPROVE). Smoke run loaded train+valid via the same path. | **PASS** |
| 2 | Cache path is flat and slash-free `.dat` under `cache_dir` | Phase-4 `test_cache_path_flat_and_slashfree`. On-disk after smoke run: `data_cache/tinystories-debug_train_bs1024_wrapped.dat` and `..._valid_...dat` — flat, no `roneneldan/` subdir. | **PASS** |
| 3 | Block geometry: `input_ids`/`attention_mask` length == 1024, BOS-first / EOS-last | Phase-4 `test_block_shape_and_bos_eos`. Smoke run logged `Batch input_ids.shape torch.Size([8, 1024])` for both train and valid dataloaders. | **PASS** |
| 4 | EOS-insertion present when `insert_train_eos: True` (>1 interior EOS per block) | Phase-4 `test_eos_insertion_present_when_enabled`. Config carried `insert_train_eos: true`, `insert_valid_eos: true`. | **PASS** |
| 5 | End-to-end smoke train: finite, non-NaN `trainer/loss`, net-decreased over N steps, reached `max_steps` | W&B run `1dhq6a1m`: 10/10 loss points finite; loss[first]=10.9423 → loss[last]=10.4505 (net −0.49); `trainer/global_step` reached 49 (= step 50, 0-indexed); state `finished`. | **PASS** |

---

## Criterion 5 specifics (the smoke-train gate)

Pulled from the W&B run history via the `wandb-primary` skill (SDK `wandb.Api()`,
env `sfm` / wandb 0.19.9). `trainer/loss` is logged on_step every 5 steps with
`max_steps=50`, giving exactly **10 logged points** as expected.

| logged `_step` | `trainer/global_step` | `trainer/loss` |
|---:|---:|---:|
| 1  | 4  | 10.94232 |
| 3  | 9  | 10.73002 |
| 5  | 14 | 10.80268 |
| 7  | 19 | 11.33216 |
| 9  | 24 | 10.97570 |
| 11 | 29 | 10.88519 |
| 13 | 34 | 10.57872 |
| 15 | 39 | 11.61543 |
| 17 | 44 | 10.81220 |
| 19 | 49 | 10.45047 |

- **Finite:** all 10 values finite, no NaN/Inf. (Check: `all(isfinite) == True`.)
- **Net decrease:** loss[first] = **10.94232** (global_step 4) →
  loss[last] = **10.45047** (global_step 49). loss[last] < loss[first] by
  **−0.4919**. The spec gates on net decrease, **not** strict monotonicity — the
  curve does bounce up at global_step 19 (11.33) and 39 (11.62), which is fully
  expected for a 50-step run at batch size 8 and does not violate the criterion.
- **Step count:** `trainer/global_step` reached **49** (0-indexed; = the full 50
  steps). Run state is **`finished`**; local log ends with
  `` `Trainer.fit` stopped: `max_steps=50` reached. ``

Config matches the spec (from `wandb-metadata.json`): `data=tinystories-debug`,
`model=small`, `algo=mdlm`, `strategy=single-device`,
`global_batch_size=batch_size=eval_batch_size=8`, `max_steps=50`,
`log_every_n_steps=5`, single device, NVIDIA RTX A6000 (Ampere), CUDA 12.8.

Note: `val/loss`, `val/nll`, `val/bpd`, `val/ppl` are NaN in the run summary —
this is **not** a failure. The smoke run was launched with
`limit_val_batches=0` / `val_check_interval=1000`, so no validation pass ran
inside 50 steps; the NaNs are uninitialized torchmetric states, unrelated to the
gated `trainer/loss`. (Three benign torchmetrics "compute before update"
UserWarnings appear for exactly this reason.)

---

## Criteria 1–4 — validated as given (not re-run)

- **Phase-4 pytest** `tests/test_tinystories_dataloader.py`: **6/6 passing**,
  reviewer (sw-reviewer) verdict **APPROVE**. Test functions present and named
  per spec: `test_tinystories_debug_train_loads_torch`,
  `test_tinystories_debug_valid_loads_torch`, `test_cache_path_flat_and_slashfree`,
  `test_block_shape_and_bos_eos`, `test_eos_insertion_present_when_enabled`,
  `test_parity_with_tinystories_schema`.
  (Post-2026-06-09: these tests now target the full `tinystories` dataset, since
  `tinystories-debug` was removed; see the dated note at the top of this file.)
- **Independent re-confirmation by the real smoke run:** generated the
  train + validation splits, tokenized the 1% slices, and saved flat caches
  `data_cache/tinystories-debug_{train,valid}_bs1024_wrapped.dat` (verified on
  disk; **no `roneneldan/` subdirectory** → criterion 2 holds in production, not
  just in the test). It then produced `torch.Size([8, 1024])` batches for both
  the train and valid dataloaders.
  - (The `data_cache/roneneldan___tiny_stories/` directory is HuggingFace's own
    raw-download cache under `cache_dir` — HF sanitizes the `/` to `___`. It is
    the upstream Arrow download, NOT the tokenized `.dat` cache that criterion 2
    gates on; the two `.dat` directories are correctly flat and slash-free.)

---

## Notes / caveats

- **The integration itself never failed.** Two earlier smoke attempts failed for
  pure cluster GPU-architecture reasons, **not** dataloader issues:
  1. A `gpu-low` node hit a DDP/NCCL `operation not supported` error.
  2. A second attempt hit CUDA `no kernel image is available for execution`
     (binary/arch mismatch).
  Both were resolved by running on a `gpu-mid/high` A6000 node with
  `strategy=single-device`. The dataloader/caching/EOS path produced correct
  output on every attempt.
- The 4 `Traceback` lines near the end of `smoke_run3.log` are benign
  multiprocessing temp-dir finalizers firing at interpreter shutdown **after**
  the run reached `max_steps` and logged successfully (exit code 0). Not a
  failure.
- This is a plumbing/integration gate, not a research measurement: no held-out
  metric, no multi-seed, no p-value. The loss magnitude (~10.5 nats) is just the
  near-random MDLM starting point over 50 steps; it is not meant to indicate
  model quality.

---

## Recommended next steps

1. **Integration is done.** No code changes needed; ship the `tinystories` /
   `tinystories-debug` dataloader branches and the two YAML configs as-is.
2. **(Optional) Full `tinystories` (non-debug) run** if a real training run on
   the full corpus is desired — exercises branch (A)'s full download + cache
   (DatasetDict path) at scale and confirms a sustained downward loss trend over
   more steps. This is a nice-to-have, not a gate.
3. If a longer run is launched, enable validation (`limit_val_batches`/
   `val_check_interval`) so `val/loss` is populated rather than NaN.

---

### Evidence index
- Spec: `/share/thickstun/sychou/workspace/research/s-flm-dev1/s-flm/experiments/tinystories/EXPERIMENT.md`
- Smoke log: `/share/thickstun/sychou/workspace/research/s-flm-dev1/s-flm/experiments/tinystories/smoke_run3.log`
- W&B run files: `/share/thickstun/sychou/workspace/research/s-flm-dev1/s-flm/outputs/tinystories/smoke50/wandb/run-20260608_154627-1dhq6a1m/`
- W&B run (remote): https://wandb.ai/syctw/tinystories-integration/runs/1dhq6a1m
- Phase-4 tests: `/share/thickstun/sychou/workspace/research/s-flm-dev1/s-flm/tests/test_tinystories_dataloader.py`
- Cache (on disk): `/share/thickstun/sychou/workspace/research/s-flm-dev1/s-flm/data_cache/tinystories-debug_{train,valid}_bs1024_wrapped.dat`
