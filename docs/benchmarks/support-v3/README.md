# Support-v3 experiment artifacts

See the [protocol](../../support-v3-protocol.md) and
[dataset card](../../../datasets/support-v3/README.md). Inputs are frozen before
training in `frozen-inputs.json`; no outcome is claimed at this protocol commit.

Use the existing optional CPU environment (`Dockerfile.probe`). Mount the repo
at `/app`, a writable scratch directory at `/work`, and the pinned model's cache;
set `PYTHONPATH=/app/src`, `HF_HUB_OFFLINE=1`, `OPENBLAS_NUM_THREADS=1`.
The training command needs no network or cloud credentials:

```sh
python scripts/run_support_v3.py train --work /work/run
python scripts/run_support_v3.py evaluate --work /work/run
```

`training/adapter.json` contains only LoRA, pooler and classifier tensors, loaded
over the hash-pinned upstream base model. Intermediate checkpoint tensors remain
in the scratch directory; all development scores, checkpoint hashes and selection
decisions are published. The selected adapter and all comparators are retained.
Evaluation writes calibration scores/thresholds before reading test scores.
Plain JSON weights require no unsafe pickle loading.

To reproduce inference independently, use the saved selected adapter in a fresh
process and a fresh output directory:

```sh
python scripts/run_support_v3.py evaluate --work /work/verify --output /work/verify-results
```

Compare `calibration.json`, `decisions.json`, `summary.json` and
`input-controls.json` with the published results; runtime metadata is expected
to differ. Fully rerun training into a new `--training` directory to reproduce
optimization (one CPU seed; exact cross-platform bitwise identity not promised).

Code and original synthetic content use Apache-2.0. Numerical artifacts and
adapter tensors trained with the SQuAD derivative are distributed under
CC-BY-SA-4.0; see the [source attribution](../../../datasets/support-v1/README.md).
The base model itself is not redistributed. This describes artifact terms,
not a legal determination about every downstream model-weight use.
