# Support-v3 experiment artifacts

See the [results](../../support-v3-results.md),
[protocol](../../support-v3-protocol.md) and
[dataset card](../../../datasets/support-v3/README.md). Inputs were frozen before
training in `frozen-inputs.json`. **V3 is not promoted.** The explicit
[order addendum](../../support-v3-order-addendum.md) was frozen during training,
before test scoring, in `order-addendum-inputs.json`.

Use the existing optional CPU environment (`Dockerfile.probe`). Mount the repo
at `/app`, a writable scratch directory at `/work`, and the pinned model's cache;
set `PYTHONPATH=/app/src`, `HF_HUB_OFFLINE=1`, `OPENBLAS_NUM_THREADS=1`.
The training command needs no network or cloud credentials:

```sh
python scripts/run_support_v3.py train --work /work/run
python scripts/run_support_v3.py evaluate --work /work/run
python scripts/support_v3_order_check.py evaluate
python scripts/verify_support_v3_results.py
python scripts/diagnose_support_v3.py
```

`training/adapter.json` contains only LoRA, pooler and classifier tensors, loaded
over the hash-pinned upstream base model. Intermediate checkpoint tensors remain
in the scratch directory; all development scores, checkpoint hashes and selection
decisions are published. The selected adapter and all comparators are retained.
Evaluation writes calibration scores/thresholds before reading test scores.
Plain JSON weights require no unsafe pickle loading.
The `v2_original_gate` entry deliberately preserves its original v2 threshold
metadata, including **old** calibration statistics; those nested statistics
are not measurements on v3 calibration. The other gates are recalibrated on v3.

To reproduce inference independently, use the saved selected adapter in a fresh
process and a fresh output directory:

```sh
python scripts/run_support_v3.py evaluate --work /work/verify --output /work/verify-results
```

Compare `calibration.json`, `decisions.json`, `summary.json` and
`input-controls.json` with the published results; runtime metadata is expected
to differ. Fully rerun training into a new `--training` directory to reproduce
optimization (one CPU seed; exact cross-platform bitwise identity not promised).
Fresh-process re-inference was byte-identical for all four primary artifacts
and for the supplemental `order-check-results.json`. To reconstruct rather than
re-infer all metrics, run `scripts/verify_support_v3_results.py`.

The original frozen `verify_support_v3.py` is retained as historical code: its
exact-control-AUC==0.5 assertion fails on tiny float32 differences between equal
inputs in different batch positions/shapes. The post-run verifier additionally
checks same-input numerical spread and class balance, while reconstructing all
unchanged raw metrics. The [results report](../../support-v3-results.md) gives
the exact deviations. This correction changes no model, threshold or prediction.

`diagnostics.json` includes all family metrics and the binding low-scoring useful
calibration cases. `error-audit.json` contains 26 deterministic illustrations,
with original labels retained. All checkpoint development scores are public;
intermediate tensor files can be regenerated but are not duplicated here.

Code and original synthetic content use Apache-2.0. Numerical artifacts and
adapter tensors trained with the SQuAD derivative are distributed under
CC-BY-SA-4.0; see the [source attribution](../../../datasets/support-v1/README.md).
The base model itself is not redistributed.
