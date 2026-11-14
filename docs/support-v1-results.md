# Human hard negatives improve support classification, but abstention remains weak

On 1,120 held-out SQuAD 2.0 questions, retraining the small context-support head
raises balanced-mode accuracy from **61.34% to 66.96%**. The paired article-bootstrap
95% interval for the **+5.63 percentage-point** difference is **[+2.77, +8.57]**.
The original relevance-logit baseline reaches **62.23%**; the new head's advantage
is **+4.73 points**, interval **[+1.61, +7.95]**.

The improvement mostly comes from **retaining useful paragraphs**, not from
rejecting more unsupported ones at the balanced operating point. The conservative
setting rejects only **46/560 unsupported cases**, so this remains an inadequate
standalone abstention gate. No production default is changed.

This measures whether a **supplied complete paragraph contains the answer**.
It does not measure document retrieval, branch navigation or generated-answer
accuracy. In particular, 66.96% must not be compared with the previous manual
retrieval Recall@5 of 95.42%: those are different tasks, labels and corpora.

## What was trained and tested

The 22,713,601-parameter MiniLM relevance encoder remains frozen. We train only
the standardized linear head: **384 coefficients plus an intercept**, with
feature means/scales estimated on fit data. Qwen has no role in this experiment.
The model revision, complete-text input format and objective match the
[registered protocol](support-v1-protocol.md).

| Role | Articles | Supported questions | Unsupported questions |
|---|---:|---:|---:|
| Fit | 128 | 2,031 | 2,031 |
| Select L2 | 32 | 384 | 384 |
| Calibrate thresholds | 32 | 384 | 384 |
| Held-out test | 35 | 560 | 560 |

Every paragraph has both labels through two different original human questions.
This supplies on-topic negatives with changed entities, relations, polarity,
dates, quantities or missing details. The original SQuAD 2.0 dev articles are
held out for this experiment; no article or normalized paragraph crosses roles.
Exact normalized questions from previous ArborVault datasets are excluded.
All question/paragraph text fits the encoder without truncation.

A pre-outcome AI-assisted audit of 25 fit pairs found six with ambiguous wording
or plausible alternative support. We excluded those twelve fit examples before
training, without replacing or relabeling them. See the
[dataset card and itemized audit](../datasets/support-v1/README.md).
These are human-authored adversarial benchmark questions, **not organic user
logs**; original labels can still be noisy.

L2=0.1 was selected for the context and question-only heads; L2=10 for the
uninformative context-only control. Selection used article-macro AUROC on the
selection role. The old head's coefficients stayed unchanged. **Every method
received the same new calibration set**, including the old head and raw logit.
No coefficients or thresholds were selected using test outcomes.

## Balanced operating point

Accuracy equals balanced accuracy here because the test has 560 questions per
class. Both error columns should be **lower**. Thresholds maximize balanced
accuracy on calibration, with a fixed tie rule favoring fewer useful rejections.

| Method | Correct / 1,120 | Accuracy | Unsupported accepted / 560 | Supported rejected / 560 | AUROC |
|---|---:|---:|---:|---:|---:|
| **Retrained context head** | **750** | **66.96%** | 272 | **98** | **0.7399** |
| Previous frozen head | 687 | 61.34% | 257 | 176 | 0.6565 |
| Original relevance logit | 697 | 62.23% | **164** | 259 | 0.6578 |
| Question-only control | 650 | 58.04% | 210 | 260 | 0.6186 |
| Context-only control | 560 | 50.00% | 560 | 0 | 0.5000 |

Compared with the previous head, the new one makes **78 fewer false rejections
of useful paragraphs**, while allowing **15 more unsupported paragraphs** through.
That is 63 additional correct decisions overall. Compared with raw relevance,
it makes 161 fewer useful rejections while allowing 108 more unsupported cases.
It therefore does not dominate either baseline on both error types.

The new head accepts 734/1,120 cases, or **65.54% coverage**. Among accepted cases,
272/734 (**37.06%**) are unsupported according to the source labels. The old head's
corresponding figure is 40.09%, and raw relevance's is 35.27% at much lower
coverage (41.52%). These proportions depend on the artificial 50% unsupported
prevalence and are not calibrated deployment guarantees.

The new head's accuracy interval is **[64.55%, 69.46%]**. Differences above use
5,000 paired resamples of the **35 articles**, keeping each article's paragraph
pairs together. They do not treat 1,120 questions as independent observations.
Intervals are conditional on these source labels and this benchmark population.

## Conservative operating point

Calibration permits at most 2% rejection of supported contexts (at most seven
of the 384 calibration positives). This is an empirical calibration constraint,
not a guarantee that the test rate stays below 2%.

| Method | Unsupported rejected / 560 | Unsupported accepted / 560 | Supported rejected / 560 | Accuracy |
|---|---:|---:|---:|---:|
| **Retrained context head** | **46 (8.21%)** | 514 | 12 (2.14%) | 53.04% |
| Previous frozen head | 22 (3.93%) | 538 | 10 (1.79%) | 51.07% |
| Original relevance logit | 15 (2.68%) | 545 | 8 (1.43%) | 50.62% |
| Question-only control | 37 (6.61%) | 523 | 13 (2.32%) | 52.14% |
| Context-only control | 0 | 560 | 0 | 50.00% |

The new head rejects 24 more unsupported cases than the old head, at the cost
of two additional useful rejections. However, **91.79% of unsupported cases still
pass**. Higher AUROC and a positive primary difference do not solve low-error
abstention. The balanced setting rejects more unsupported cases but loses
98/560 useful ones (**17.50%**), a substantial practical cost.

## Controls and error audit

The context-only head is exactly uninformative: every paragraph has opposite
labels and identical features, so it predicts 0.5. The question-only control
exceeds chance (58.04% accuracy, 0.6186 AUROC), demonstrating residual question
style or construction cues. The full head performs better, but that alone does
not establish robust reasoning or absence of other shortcuts.

A post-hoc **AI-assisted** sample inspected six false accepts and six false
rejects, chosen by fixed hash order within each error class. The
[itemized audit](benchmarks/support-v1/error-audit.json) preserves all official
test labels and scores. It suggests the following failure modes:

- Polarity reversal: a degree described as required is accepted for a question
  asking which degree is no longer required; outside competence is accepted
  for a question about inside competence.
- Wrong temporal scope: a paragraph about the fourth parliamentary session is
  accepted for a question about the fifth.
- Wrong relationship: a competitor is treated as evidence for a supporter.
- Useful evidence rejected: an explicit financial amount near the paragraph's
  end, a parenthetical synonym, a paraphrased time interval, and a misspelled
  reference to a network are all missed. Full-text encoding rules out truncation
  for these examples.
- Three sampled cases merit independent label review: two negatives plausibly
  admit an answer and one positive accepts a relative quantity for a numerical
  question. This selected error sample is not a label-error-rate estimate.

The [complete article breakdown](benchmarks/support-v1/article-breakdown.json)
shows new-head balanced accuracy ranging from 46.88% to 81.25% across articles.
Each article has only 32 test questions, so individual article rankings are noisy.

## Interpretation and next evidence needed

There is a positive signal for replacing the old synthetic-data head **on this
paragraph-support benchmark**, with a primary confidence interval above zero.
The actual gain is smaller than an accuracy-only headline might suggest:
strict abstention remains weak and the balanced setting still accepts 272
unsupported contexts. The new head is retained as an experimental artifact.

Before production promotion, independently adjudicate relational, negation and
temporal negatives; pair them with valid paraphrases and typos so lexical novelty
does not become a rejection cue. Then test complete retrieved multi-passage
contexts on fresh application-like questions, with an explicitly chosen error
cost and prevalence. An encoder adaptation experiment could test whether the
frozen representation limits this linear head, but this run does not isolate
that cause. This test is now observed development evidence for future changes;
it must not be reused as a fresh holdout.

The English Wikipedia domain, public dev split, possible pretraining exposure,
crowdworker artifacts, residual label noise and balanced class prevalence limit
transfer. This provides no claim of improvement over vector RAG, corpus-wide
retrieval or an IBM STAIR implementation.

## Reproduction, validation and cost

The initial registration was published at
[`7120fc1`](https://github.com/BVGAL/arborvault/commit/7120fc1c0b82600ae5a9b06775fb713d2ca0ae2c).
The dependency-audit amendment
[`2295d63`](https://github.com/BVGAL/arborvault/commit/2295d63cf1342cb3b2268e4c4ccba095b0c4d4b2)
was published before restarting extraction. The initial runtime stopped during
selection-feature extraction, before calibration/test features or classifier
fitting. All published features use PyTorch 2.13.0+cpu, Transformers 5.22.2,
NumPy 2.5.4, float32 eager inference and seven-decimal feature serialization.

The [artifact directory](benchmarks/support-v1/README.md) includes every feature
shard, fitted head, threshold, held-out decision, confidence interval, audit and
input hash. A fresh analysis reproduced coefficients, decisions and summary
**byte for byte**. The verifier independently reconstructs the stored decisions
and checks all input hashes. Sixteen focused tests pass, including the real
pinned tokenizer/encoder and equality with the previous CLS feature path.
The ordinary local suite passes 132 tests with 13 environment-dependent skips;
the [pre-test GitHub CI](https://github.com/BVGAL/arborvault/actions/runs/43615332362)
passes **136 tests with 9 skips**, including database checks, and its separate
Compose job succeeds.

Final feature extraction took **501.30 seconds** on four local CPU threads for
16,795 unique forward passes across all roles and controls. The fitting,
calibration and analysis command took about 4.20 seconds including container
launch; these are workflow timings, not serving-latency benchmarks.

**Additional cloud spend: $0.00. Cumulative estimated spend: $7.94877 of $10.00.**
No Runpod or other paid compute was allocated. Local electricity and hardware
amortization are unmeasured and excluded. See the [cost record](benchmarks/support-v1/run-cost.json)
and [dependency-audit details](benchmarks/support-v1/dependency-audit.json).
