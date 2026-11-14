# Semantic-abstention artifacts

Preregistered at
[`3a725b4`](https://github.com/BVGAL/arborvault/commit/3a725b49651cd7b41d37cd34da0bdc8195f02ea1)
before new feature extraction and classifier fitting. See the
[protocol](../../support-v2-protocol.md) and
[dataset card](../../../datasets/support-v2/README.md).

- `frozen-inputs.json` locks the algorithm, protocol, configuration and dataset.
- `features/manifest.json` records exact versions, hashes, reused sources and runtime.
- `features/{fit,selection,calibration,test}.json` contains only new vectors,
  raw logits, per-example input hashes and references to already published
  support-v1 fit/selection features. Referenced source files are hash-checked.
- `results/fitted-models.json` contains chosen and ablation heads, complete
  development candidate scores, training objectives and common calibration gates.
- `results/decisions.json` contains every held-out score and decision.
- `results/summary.json` keeps human transfer, combined contrasts and individual
  phenomena separate, with paired article/family bootstrap intervals.
- `run-cost.json` records local execution and the cumulative cloud ledger.
- `diagnostics.json` and `error-audit.json` describe post-hoc errors and score
  behavior; no fitted heads, labels or thresholds were changed by this analysis.

Numeric experiment artifacts are offered under CC BY-SA 4.0 with attribution to
the SQuAD 2.0 authors and ArborVault contributors. The independently authored
contrast corpus and implementation use Apache-2.0; the human data retains
[CC BY-SA 4.0](../../../datasets/support-v1/LICENSE-DATA.txt). The pretrained
encoder is downloaded separately under its model terms and is not redistributed.

In the CPU environment, from the repository root:

```sh
python scripts/verify_support_v2.py
python scripts/run_support_v2.py analyze --output reports/support-v2-reproduced
python scripts/diagnose_support_v2.py
```

The first command verifies the publication chain and reconstructs held-out
decisions from saved features. The second refits the fixed candidate schedule,
performs development-only selection and calibration, then reproduces the test.
Neither command chooses new settings using test outcomes. Use fresh output
directories; overwriting completed experiments is refused.
