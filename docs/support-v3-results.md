# Support v3: temporal retention, negation failures and an entity-order shortcut

**V3 is not promoted.** Adapting the MiniLM reader improves controlled temporal
abstention, but the requested combination of robust negation, reversed relations
and human transfer was not achieved. A supplementary test catches an apparently
large relation gain that does **not** survive equivalent question rewording.
No production default changes, Qwen training or paid compute were used.

The selected checkpoint is step **1,800**, chosen using development only. Training
updates **480,001 parameters**: rank-8 LoRA updates throughout the six-layer
reader plus its pooler/classifier. Base encoder matrices, embeddings and layer
norms remain unchanged, verified by hash. This goes beyond the frozen-vector
heads of v2, but is still one small-model, one-seed experiment.

## What was fixed before evaluation

The [initial protocol](support-v3-protocol.md), code and data were published in
[`55f7c99`](https://github.com/BVGAL/arborvault/commit/55f7c99ed207e845473d74db6a91d5aacf9796b3)
before training or test scoring. Fit contains **9,566** examples; selection has
1,632, calibration 1,120, and the primary test **2,688**. Human calibration/test
use 8/24 articles absent from every previous support-v1/v2 role. Primary test
has 768 human questions and 1,920 controlled cases, five families per phenomenon.
Old fit rows are replayed; no old calibration/test rows enter training.

During training, after development checkpoints 300 and 600, static inspection
found that every supported original relation-test question puts the names in
the opposite order to their appearance in the context. A simple opposite-order
rule therefore scores **100%** on that original relation test. Single-input
label balance does not prevent a shortcut involving both inputs.

Before any test scoring, we published an explicit
[addendum](support-v3-order-addendum.md) in
[`3ef677c`](https://github.com/BVGAL/arborvault/commit/3ef677c53cc8fbc0cc7ad8e801c7f7df082fd851).
It adds **640** semantically equivalent relation questions with complementary
name ordering. The 1,280-row original-plus-alternate union balances that feature;
either ordering-only rule then scores 50%. These are the same 160 held-out
scenarios and five families, **not** 640 additional independent scenarios.
Training, model selection, thresholds and initial promotion rules stay unchanged.
The addendum introduces an additional robustness check before recommendation;
it is not retroactively presented as part of the initial preregistration.

## Primary conservative comparison

Rejecting unsupported paragraphs should increase; rejecting useful paragraphs
should decrease. V1, v2 and new comparators are recalibrated on the **same new
calibration partition**, with one threshold per method and at most 2% useful
rejection in each calibration stratum. These constraints need not hold on test.

| Primary test | V2 unsupported rejected | V3 unsupported rejected | V2 useful rejected | V3 useful rejected |
|---|---:|---:|---:|---:|
| Human questions | 0/384 (0%) | 5/384 (1.30%) | 0/384 | 0/384 |
| Negation | 28/320 (8.75%) | 194/320 (60.62%) | 16/320 (5%) | **34/320 (10.62%)** |
| Relations, original wording | 0/320 | 204/320 (63.75%) | 0/320 | 0/320 |
| Temporal references | 122/320 (38.12%) | **318/320 (99.38%)** | 0/320 | 0/320 |

For human transfer, the preregistered reference was **v1**, not the unusually
permissive recalibrated v2 gate. V1 rejects **40/384** unsupported and **6/384**
useful paragraphs; v3 rejects 5 and 0. The unsupported-rejection difference is
**−9.11 percentage points**, paired article-bootstrap 95% interval
**[−12.76, −5.73]**. Useful rejection improves by 1.56 points, interval
**[−2.86, −0.52]**. This is an operating-point tradeoff, not dominance at equal
realized test false-rejection rates. The human-retention criterion fails.

The unchanged **original v2 gate** is also retained as a comparator. On this new
test it rejects 55/384 unsupported human paragraphs with 6 useful losses; on
negations 228/320 with **159** useful losses; on relations 0/320 with 0 useful
losses; on dates 256/320 with 0 useful losses. V3's temporal gain therefore also
holds against the original gate: **256 to 318**, without useful temporal losses.
Do not compare these counts with the previous report's different test sample.

All models already order positive/negative temporal contexts perfectly for the
same question. The temporal gain is in score separation and usable abstention
under a common threshold, not evidence of newly acquired temporal reasoning.

## The relation gain does not survive the complementary test

| Selected V3, same frozen gate | Unsupported rejected | Useful rejected | AUROC | Same-question correct context ranked higher |
|---|---:|---:|---:|---:|
| Original relation wording | 204/320 | 0/320 | 0.9917 | 320/320 (100%) |
| Equivalent alternate wording | **2/320** | **6/320** | **0.4370** | **152/320 (47.5%)** |
| Order-balanced union | 206/640 | 6/640 | 0.7741 | 472/640 (73.75%) |

V2 rejects 0/320 alternate unsupported cases and loses no useful cases. The v3
alternate gain is only 0.63 points, far below the declared 10-point requirement.
Even the ranking collapses on alternate wording, so this is **not merely a bad
threshold**. The result is consistent with reliance on entity ordering and
wording shortcuts. It does not identify a unique internal causal mechanism.
The attractive 63.75% original-test rejection rate alone is misleading.
Full comparator scores and paired family intervals are in
[order-check-results.json](benchmarks/support-v3/order-check-results.json).

## Why negations and the common gate fail

**32 of the 34 useful negation rejections** occur in the family about lifting
restrictions. A context explicitly says a restriction has been lifted, and the
question asks which unit is *no longer restricted*; the model rejects valid
answers. In that family it rejects 32/64 useful and only 4/64 unsupported cases.
Another family, exceptions to prohibitions, rejects **0/64** unsupported cases.
Both are failures of compositional polarity on new wording, despite perfect
development negation partial area and perfect calibration negation separation.

The common conservative threshold is **−6.902599334716797** in the new model's
logit scale. The binding calibration errors are useful relation records in the
*receiving party / sending party* format: their lowest scores are −8.03, −7.23
and −6.90. Protecting the third-lowest useful relation case constrains the gate,
making it permissive for human paragraphs. Human AUROC improves from v1's
**0.7349** to **0.7655**, but the common threshold does not convert this into
strong conservative abstention. No category-specific oracle threshold is used.

[diagnostics.json](benchmarks/support-v3/diagnostics.json) contains every family
breakdown and the lowest supported calibration scores. The
[26 illustrative cases](benchmarks/support-v3/error-audit.json) are selected by
hash within outcome categories, not a representative prevalence sample. Labels
are unchanged. Human annotations remain fallible: the upper-house/parliament
example contains an exception that could make its official unanswerable label
debatable. No independent human adjudication or relabeling is claimed.

## Comparators, secondary setting and uncertainty

The frozen-encoder head retrained on the new data rejects **34/384** unsupported
human cases with 5 useful losses, **58/320** negation negatives with 4 useful
losses, no relation negatives, and **238/320** temporal negatives with no useful
losses. Its objective/sampling differ from the adapter, so it is a comparator,
not a causal isolation of encoder adaptation. All raw, v1, v2, frozen-head and
unchanged-v2-gate results are included in
[summary.json](benchmarks/support-v3/results/summary.json).

At the secondary balanced gate, v3 rejects 272/384 unsupported human cases but
also **120/384 useful cases (31.25%)**. It rejects 224/320 negation negatives but
loses **63/320 useful cases (19.69%)**. This does not rescue the prudent objective.

Bootstrap uses 5,000 paired draws, articles for human questions and template
families for synthetic cases. The original relation gain versus v2 is +63.75
points with an exploratory interval [+36.56, +90.94], but its known shortcut
invalidates a broad semantic-generalization interpretation. Negation useful
loss changes by +5.62 points, interval [−13.12, +30.00]. Only five families per
phenomenon support these intervals; absence of errors in a few families is not
a guarantee for unseen phrasing. The alternate/union tests reuse these families.

## Training and verification

LoRA rank 8 / alpha 16 on all six linear transformations in all six blocks;
pooler/classifier also trained. Seed 42, batch 16, equal stratum exposure,
1,800 AdamW steps, LR 3e-4, 90-step warmup and cosine decay, clipping 1.0.
Objective: support cross entropy plus 0.5 times same-question counterfactual
ranking loss. Each stratum receives 7,200 row exposures; these are repeated
draws, not that many unique examples. No test-driven retraining or threshold
repair occurs. [Training metadata](benchmarks/support-v3/training/training.json)
and [selected tensors](benchmarks/support-v3/training/adapter.json) are public.

The frozen dataset rebuilds byte-for-byte. Re-inference in fresh CPU processes
reproduces calibration, primary decisions/summary, input controls and the
supplementary results **byte-for-byte**. The verifier reconstructs selection,
thresholds, all metrics, clustered intervals and promotion checks.

One verification-only correction is documented openly: the initial frozen
verifier required input-only control AUROC to equal exactly 0.5. Batched float32
roundoff gives **0.499990234375** and **0.5000059678819444**. Identical-input logits
vary by at most 2.17e-5; exact-input groups are balanced and tying their scores
gives exactly 0.5. The original verifier is retained unchanged; use
`scripts/verify_support_v3_results.py`, which checks raw metrics, duplicate-input
spread and exact-input balance. **No published predictions, labels, main metrics,
model weights or gates were changed to fix this sanity assertion.**

Additional cloud cost **$0**; cumulative estimate **$7.94877 / $10**, with
$2.05123 remaining. Training/preparation/selection takes **1,291 seconds** on
local CPU. Electricity and hardware amortization are not measured. See the
[cost ledger](benchmarks/support-v3/run-cost.json) and
[reproduction guide](benchmarks/support-v3/README.md).

## Next experiment implied by these errors

Before another training round, counterbalance actor order **within each training
family**, crossing active/passive phrasing and question direction. Add logically
equivalent positive and negative forms for lifted restrictions and exceptions,
with independent label review. Keep temporal replay and evaluate a common score
scale on heterogeneous calibration wording. These are hypotheses for a new
experiment, not changes fitted to this already observed test. The new test must
come from fresh sources/families; the current human source pool is nearly spent.
