# Reproducible support-classifier artifacts

The preregistration is commit
[`7120fc1`](https://github.com/BVGAL/arborvault/commit/7120fc1c0b82600ae5a9b06775fb713d2ca0ae2c).
It was published before held-out feature extraction and test scoring.
The environment amendment
[`2295d63`](https://github.com/BVGAL/arborvault/commit/2295d63cf1342cb3b2268e4c4ccba095b0c4d4b2)
upgrades the CPU runtime before test extraction; all features use that amended runtime.
See the [protocol](../../support-v1-protocol.md) and
[dataset attribution and limitations](../../../datasets/support-v1/README.md).

- `frozen-inputs.json`: original algorithm, configuration and dataset hashes.
- `features/manifest.json`: runtime versions, input hashes, per-role times and hashes.
- `features/{fit,selection,calibration,test}.json`: rounded frozen CLS features and
  raw logits; only text hashes and numeric data. Each role has a separate shard.
- `results/fitted-probes.json`: selected standardized linear heads, all three L2
  selection scores and both calibration thresholds per method. `previous` keeps
  its original coefficients; only its thresholds are recalibrated.
- `results/decisions.json`: every held-out score and accept/reject decision.
- `results/summary.json`: confusion counts, metrics and paired article-bootstrap intervals.
- `article-breakdown.json` and `error-audit.json`: post-hoc descriptive diagnostics;
  no changes to fitted heads, thresholds or official test labels.
- `run-cost.json`: incremental and cumulative estimated cloud spend.
- `dependency-audit.json`: initial findings, corrected upstream version audit and
  the CPU-wheel auditing limitation.

Numeric features, fitted-head artifacts, decisions and audit annotations in this
experiment are offered under CC BY-SA 4.0, with attribution to the SQuAD 2.0
authors and ArborVault contributors. The dataset's full license is
[provided here](../../../datasets/support-v1/LICENSE-DATA.txt). Implementation and
original prose use the repository's Apache-2.0 license. The downloaded pretrained
encoder is not redistributed; its separate model terms still apply.

From the CPU environment, check the recorded publication without downloading an encoder:

```sh
python scripts/verify_support_artifacts.py
python scripts/run_support_probe.py analyze --output reports/support-v1-reproduced
```

The verifier checks the frozen input chain, every feature-shard checksum, old-head
identity, test decisions and confidence intervals. The second command reruns
fitting and calibration on the fixed training roles and writes into a fresh
directory. These are research heads, not a production API default or a model
trained on live application traffic.
