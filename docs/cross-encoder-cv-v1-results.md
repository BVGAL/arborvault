# Cross-encoder CV v1: five-fold document results

**The broader CV does not confirm the earlier large fine-tuning gain. Qwen base
has the highest pooled coverage-first recall in this comparison.** Its Recall@5
is **71.44%**, versus **67.08%** for the adapted cross-encoder, **65.80%** for the
unadapted cross-encoder and **58.51%** for standard-text E5.

The adapted cross-encoder gains **1.29 points** over its own base, with a paired
document-bootstrap 95% interval of **[-1.12, +3.65]**. It improves recall in three
of five folds, rather than showing a consistent large gain. Against Qwen base its
delta is **-4.36 points [-7.52, -1.02]**; against this E5 reference it is **+8.57
points [+5.22, +11.91]**. These are conditional exploratory intervals with the
limitations below. They do not establish superiority over all dense retrieval
or vector-RAG systems, nor compare equal serving costs.

There is a more favorable ranking signal: cross-encoder MRR rises from **.4776
to .5346**, with a numerical improvement in all five folds. That means the first
annotated hit tends to rank earlier; it does not establish a dependable gain in
recovering the full evidence set within the final context. The fine-tuned model
is therefore not promoted as a recall upgrade.

The preceding **81%** figure came from a different, smaller development cohort
and a model trained on more questions. Its direct subtraction from 67.08% would
confound cohort difficulty, training size and checkpoint selection. Qwen and E5
also score lower on the pooled cohort. The relevant new fine-tuning effect is
the **within-CV** adapted-versus-base comparison, not a cross-experiment drop.

**Query-level abstention makes the tradeoff measurable, but does not solve it.**
For the adapted cross-encoder, the conservative raw-maximum threshold rejects
**15/140 unknown questions (10.71%)**, while rejecting **17/700 answerable questions
(2.43%)**; recall is **65.94%**, down 1.14 points from coverage-first. Its false
abstention rate slightly exceeds the inner calibration's two-point guard on OOF
evaluation. That guard was an empirical calibration rule, not a guarantee.

The five-feature learned gate is worse for this cross-encoder at the observed
operating points: the conservative policy rejects **19/140** unknowns but also
**30/700** answerable questions, with **64.55%** recall. It exceeds both calibration
guards on OOF predictions. Mean fold answerability AUROC falls from **.7165** for
raw maximum score to **.6473** for the learned gate. The balanced learned gate
rejects fewer unknowns and more answerable questions than the balanced raw-score
policy. Added calibration complexity did not generalize here. Small gate-fit sets
(12–25 unknown questions per fold) are a plausible contributor, not an isolated
causal diagnosis.

For Qwen, the conservative learned gate retains **69.23%** recall while rejecting
**28/140 unknowns (20.00%)** and **20/700 answerable questions (2.86%)**. Its recall
loss is 2.21 points, also beyond the two-point inner guard. The balanced gate
rejects **97/140** unknowns but **171/700** answerable questions and leaves **54.83%**
recall. More abstentions are not automatically a useful product improvement.

**Decision:** retain Qwen base as the quality reference and keep production
unchanged. The small cross-encoder remains a candidate where model size and serving
cost matter, but this recipe has not demonstrated a stable recall gain from training.
For its next abstention study, the simple maximum-score gate is the more defensible
candidate to validate; that choice is now informed by CV and needs new confirmation.
Collect more independently reviewed unknown queries and new documents, define the
acceptable false-abstention cost, then validate before another training round or
deployment. No extra cloud run or model promotion follows automatically from this report.


## Scope and frozen choices

The [protocol](cross-encoder-cv-v1-protocol.md) and source were fixed before outer
evaluation. Pooling historical QASPER train/dev/test gives **840 questions on 624
documents**, including **700 answerable and 140 unknown** questions. Every document
is evaluated once out of fold; no question from its document participates in that
fold's weight updates, checkpoint selection, gate fitting or threshold calibration.
The final predictions are from **five separate trained models**, not one model
trained on all 840 questions. Each starts from the same original MiniLM cross-encoder.

Each outer fold reserves separate inner document sets for checkpoint choice, gate
fit and threshold tuning. Only 390–410 questions train each model; 300 documents
per fold, rather than all four outer folds. This sacrifices some training data to
separate calibration. All labels remain the existing human QASPER labels, including
their limitations. No AI audit proposal becomes a training label.

This is retrospective CV on previously inspected data, **not a new untouched test**.
Historical test tables remain unchanged, but their questions are now reused within
this explicit CV experiment. The earlier 120-question development percentages are
not directly comparable to these pooled results. Pretraining contamination cannot
be ruled out, and CV does not erase prior recipe selection on this corpus.

## Coverage-first retrieval

Neural methods use identical candidate text, a BM25 prefilter of at most 64 passages,
then scorer ranking. E5 uses standard path-plus-passage text over all passages of
the known document. All methods return at most five passages and 512 words. The
comparison concerns evidence retrieval, not generated-answer quality or corpus-wide
search. No new Qwen fine-tuning or new tree policy is included.

| Method / policy | Recall@5 | Evidence coverage | MRR | False abstentions | Unknown rejected | Query coverage | Unsupported returns among retained |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cross-encoder-tuned/coverage | 67.08% | 64.45% | 0.5346 | 0/700 | 0/140 | 100.00% | 38.81% |
| cross-encoder-base/coverage | 65.80% | 62.65% | 0.4776 | 0/700 | 0/140 | 100.00% | 40.00% |
| qwen-base/coverage | 71.44% | 68.52% | 0.5506 | 0/700 | 0/140 | 100.00% | 34.29% |
| e5-text/coverage | 58.51% | 55.71% | 0.4296 | 0/700 | 0/140 | 100.00% | 46.55% |

## Abstention: conservative operating point

A query gate accepts the entire retrieved context or returns nothing. Accepted
queries retain their original passages; low individual scores no longer delete
useful context. `max` uses only the top relevance score. `gate` fits five scalar
features with fixed regularized logistic regression, then selects a threshold on
different inner documents. All four models receive the same calibration procedure.

Conservative calibration maximizes rejection of unknown questions subject to no
more than two percentage points of recall loss and two points of extra false
abstention on answerable calibration questions. These are empirical inner-set
guards, not guaranteed limits on the outer folds. False abstentions below are
counted on **all 700 answerable questions**; recall also includes their zero scores.

| Method / policy | Recall@5 | Evidence coverage | MRR | False abstentions | Unknown rejected | Query coverage | Unsupported returns among retained |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cross-encoder-tuned/max-conservative | 65.94% | 63.31% | 0.5286 | 17/700 | 15/140 | 96.19% | 37.38% |
| cross-encoder-tuned/gate-conservative | 64.55% | 61.99% | 0.5178 | 30/700 | 19/140 | 94.17% | 37.42% |
| cross-encoder-base/max-conservative | 64.37% | 61.23% | 0.4687 | 20/700 | 12/140 | 96.19% | 39.11% |
| cross-encoder-base/gate-conservative | 64.37% | 61.30% | 0.4677 | 16/700 | 6/140 | 97.38% | 39.85% |
| qwen-base/max-conservative | 69.42% | 66.79% | 0.5352 | 19/700 | 23/140 | 95.00% | 32.83% |
| qwen-base/gate-conservative | 69.23% | 66.45% | 0.5338 | 20/700 | 28/140 | 94.29% | 32.45% |
| e5-text/max-conservative | 58.25% | 55.52% | 0.4273 | 7/700 | 4/140 | 98.69% | 46.20% |
| e5-text/gate-conservative | 56.65% | 53.86% | 0.4135 | 14/700 | 3/140 | 97.98% | 47.02% |

## Abstention: balanced operating point

The balanced threshold maximizes the mean of answerable keep rate and unknown
rejection rate on inner calibration. It has **no recall-loss guard**. A higher
unknown rejection rate can come at a substantial cost in useful retrieval. Do not
select this operating point solely because its unknown count looks better.

| Method / policy | Recall@5 | Evidence coverage | MRR | False abstentions | Unknown rejected | Query coverage | Unsupported returns among retained |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cross-encoder-tuned/max-balanced | 53.93% | 51.73% | 0.4344 | 163/700 | 76/140 | 71.55% | 31.61% |
| cross-encoder-tuned/gate-balanced | 51.79% | 49.77% | 0.4182 | 173/700 | 61/140 | 72.14% | 34.82% |
| cross-encoder-base/max-balanced | 49.89% | 47.44% | 0.3732 | 191/700 | 71/140 | 68.81% | 34.60% |
| cross-encoder-base/gate-balanced | 54.18% | 51.87% | 0.3967 | 144/700 | 60/140 | 75.71% | 35.53% |
| qwen-base/max-balanced | 53.71% | 51.95% | 0.4191 | 187/700 | 96/140 | 66.31% | 26.03% |
| qwen-base/gate-balanced | 54.83% | 52.79% | 0.4212 | 171/700 | 97/140 | 68.10% | 26.22% |
| e5-text/max-balanced | 43.23% | 41.51% | 0.3199 | 201/700 | 54/140 | 69.64% | 43.76% |
| e5-text/gate-balanced | 38.45% | 36.33% | 0.2753 | 219/700 | 56/140 | 67.26% | 47.26% |

"Unsupported returns" means retained unknown queries or retained answerable queries
with zero annotated-evidence recall, divided by retained queries. It is a diagnostic
for the retrieved context, not an answer hallucination rate. Incomplete annotations
can label valid alternative evidence as unsupported. Conditional recall is also
available in raw results; it excludes rejected answerable questions and must not
replace the unconditional recall column above.

## Fold results and paired uncertainty

| Fold | Fit / selection / gate-fit / threshold / evaluation questions | Evaluation unknown | Selected step | CE tuned recall | CE base recall | Qwen base recall | E5 recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 407 / 75 / 101 / 94 / 163 | 26 | 100 | 62.23% | 64.36% | 72.69% | 57.73% |
| 2 | 397 / 62 / 105 / 102 / 174 | 35 | 75 | 66.55% | 66.67% | 71.22% | 56.47% |
| 3 | 410 / 66 / 106 / 95 / 163 | 28 | 75 | 69.81% | 65.12% | 73.95% | 58.64% |
| 4 | 390 / 62 / 102 / 106 / 180 | 28 | 98 | 62.06% | 61.40% | 68.80% | 56.30% |
| 5 | 395 / 73 / 99 / 113 / 160 | 23 | 99 | 75.36% | 71.90% | 70.86% | 63.69% |

Pooled metrics weight questions equally, not folds equally. All five folds are
shown rather than selecting a favorable split. Checkpoint curves and each fold's
locked coefficients, thresholds and calibration grid are included in the artifacts.

| Paired comparison | Recall delta, points [95% interval] | Unknown rejection delta, points [95% interval] | False-abstention delta, points [95% interval] |
| --- | ---: | ---: | ---: |
| cross-encoder-tuned/coverage minus cross-encoder-base/coverage | +1.29 [-1.12, +3.65] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| cross-encoder-tuned/coverage minus qwen-base/coverage | -4.36 [-7.52, -1.02] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| cross-encoder-tuned/coverage minus e5-text/coverage | +8.57 [+5.22, +11.91] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| cross-encoder-tuned/gate-conservative minus cross-encoder-base/gate-conservative | +0.18 [-2.35, +2.64] | +9.29 [+2.90, +15.61] | +2.00 [+0.71, +3.28] |
| cross-encoder-tuned/gate-conservative minus qwen-base/gate-conservative | -4.68 [-8.15, -1.23] | -6.43 [-15.94, +3.02] | +1.43 [-0.42, +3.18] |
| cross-encoder-tuned/gate-conservative minus e5-text/gate-conservative | +7.89 [+4.15, +11.59] | +11.43 [+5.11, +17.93] | +2.29 [+0.71, +4.11] |
| cross-encoder-tuned/gate-balanced minus cross-encoder-base/gate-balanced | -2.39 [-5.71, +0.89] | +0.71 [-7.30, +8.51] | +4.14 [+0.72, +7.67] |
| cross-encoder-tuned/gate-balanced minus qwen-base/gate-balanced | -3.05 [-7.27, +1.09] | -25.71 [-35.21, -15.97] | +0.29 [-3.71, +4.30] |
| cross-encoder-tuned/gate-balanced minus e5-text/gate-balanced | +13.33 [+9.43, +17.34] | +3.57 [-6.34, +13.48] | -6.57 [-10.54, -2.41] |

Positive false-abstention deltas mean **more answerable questions rejected**, which
is a cost. Intervals use 2,000 paired document-bootstrap replicates stratified by
outer fold. They condition on saved predictions and do not refit models. Overlapping
training sets, training-seed variability, earlier researcher choices and label/domain
uncertainty are not fully represented. These are nominal exploratory intervals,
not a guaranteed confidence interval for the full CV learning procedure, and are
not adjusted for multiple comparisons. All 31 paired comparisons are retained in
the [verified summary](benchmarks/cross-encoder-cv-v1/verified-summary.json).

## Answerability discrimination and probability checks

| Model | Mean fold AUROC: raw max | Mean fold AUROC: learned gate | Mean fold unknown AP | OOF probability Brier | OOF ECE (10 bins) |
| --- | ---: | ---: | ---: | ---: | ---: |
| cross-encoder-tuned | 0.7165 | 0.6473 | 0.3295 | 0.1382 | 0.0600 |
| cross-encoder-base | 0.6678 | 0.6311 | 0.3013 | 0.1363 | 0.0330 |
| qwen-base | 0.7830 | 0.7896 | 0.4717 | 0.1192 | 0.0409 |
| e5-text | 0.5898 | 0.5622 | 0.2328 | 0.1450 | 0.0650 |

AUROC and unknown average precision (AP) above are arithmetic means of the five
fold-specific values. Raw cross-encoder score scales can differ between folds,
so a pooled raw-score AUROC is intentionally not used. Fold-specific AP prevalence
references are retained; aggregate unknown prevalence is 140/840, or 16.67%.
That prevalence follows the pilot's sampling quotas and does not estimate the
unknown-query prevalence in a deployed system. Thresholds and probability calibration
need validation for the intended traffic and cost of rejecting an answerable query.
The learned gate outputs probability estimates; reliability bins, Brier and ECE
describe their calibration on the OOF predictions without guaranteeing it on new
traffic. The raw report also includes descriptive retained-coverage curves, which
were not used to select deployment thresholds.

## Verification, cost and reproducibility

GPU source: `249a4a20b0d443cf887d71fb96e1ef258454abde`. The downloaded inventory contains
**154 hash-verified artifacts**, including checkpoint weights.
Offline verification rebuilt fold membership, candidate sets, checkpoint-selection
metrics, gates and threshold rules before producing all 20 OOF policy reports.
This checks reproducibility against saved scores; it is not independent human
review or a second training seed. The GPU environment passed **30 tests**.
The full local application suite passed **101 tests**, with six optional tensor
checks skipped there and covered separately in the GPU environment.

Experiment runtime after installation: **23.17 minutes**.
Billable allocation including setup, waiting and transfer: **32.46
minutes**. Estimated GPU/storage cost: **$0.49**; cumulative
estimated spend: **$4.99 / $10**. These are runtime/rate
estimates, not a final provider invoice. Pod deletion and absence were verified.
No persistent network volume or production model was left running.

Checkpoint binaries are retained locally and omitted from Git; the public
[weight hash inventory](benchmarks/cross-encoder-cv-v1/model-artifact-hashes.json)
records their fingerprints. No Hugging Face release or production promotion is
claimed. The public artifact directory is not a complete model download; the
full verifier also expects the checkpoint files from an actual run.

Artifacts: [OOF results and verification](benchmarks/cross-encoder-cv-v1/verified-summary.json),
[fold plan](benchmarks/cross-encoder-cv-v1/plan.json),
[environment](benchmarks/cross-encoder-cv-v1/environment.json),
[cost and cleanup](benchmarks/cross-encoder-cv-v1/run-cost.json),
[public checksums](benchmarks/cross-encoder-cv-v1/public-manifest.json).

```sh
python scripts/run_cross_encoder_cv.py --output /path/to/new-cv
python scripts/summarize_cross_encoder_cv.py /path/to/new-cv --output /path/to/verified-summary.json
```

The runner requires the pinned GPU training environment in the protocol; the
verifier runs on CPU with NumPy. This pilot remains limited to short English NLP
papers with the relevant document already supplied. Long documents, multilingual
and multi-document search, actual serving latency and generated answers still
need separate evaluation.
