# Support v2: preregistered semantic abstention experiment

This protocol, code, configuration and dataset are published before extracting
held-out features or selecting the new classifier. The prior support-v1 test
has been observed and is not reused as a fresh test or added to training.

## Objective and data

Improve rejection of unsupported but relevant paragraphs involving negation,
reversed relations and wrong temporal references, while limiting rejection of
supported paragraphs. Assess transfer on **new human questions** separately
from performance on controlled contrasts. The primary comparison is the new
selected classifier versus the frozen support-v1 head at the conservative
operating point, reporting **both** error rates, not accuracy alone.

The [dataset card](../datasets/support-v2/README.md) defines support as an explicit
non-null requested answer, describes the human source, licenses and synthetic
annotation limits, and gives every split count. Fit has 7,902 examples; selection
and calibration each have 1,536; test has 1,024 human and 1,152 contrast examples.
Every contrast question and context individually has both labels. Event years
and distracting reference-document years both occur in temporal contexts.

Human fit/selection come from support-v1. New human calibration/test articles
are completely disjoint from all support-v1 articles and from each other.
Synthetic roles have separate template families and fictional entities. The
test templates were authored and reviewed before model outcomes; this is a
training-held-out wording test, not independent human benchmark construction.

## Frozen representation and candidates

Keep `cross-encoder/ms-marco-MiniLM-L6-v2`, revision
`233902d25c440f23af6f7d6e94d2946bac0bee0a`, frozen. Use whole question/paragraph
pairs, maximum 512 tokens, float32 eager CPU inference, right padding, four CPU
threads, batch size 16, and last-layer 384-dimensional CLS features. Serialize
features and original relevance logits to seven decimal places. Only question
and paragraph text enter the model. Use the already tested CPU environment:
PyTorch 2.13.0+cpu, Transformers 5.22.2 and NumPy 2.5.4, with pinned safetensors
and remote model code disabled.

Reuse only the old **fit and selection** feature caches when their source hashes,
input hashes and representation match. Persist references instead of duplicating
old vectors. All new human articles and all contrasts require new inference.
The previous head's coefficients and standardization remain unchanged.

Fit the following on the combined fit rows with equal row weights:

1. The previous standardized linear logistic objective with L2 in **{0.1, 1, 10}**.
   This is the data-only architecture ablation; it has 385 trainable coefficients.
2. A **32-unit tanh hidden layer** and logistic output, with fit-only feature
   standardization. It has 12,353 coefficients. Train float64 full-batch Adam
   (beta1=0.9, beta2=0.999, epsilon=1e-8), learning rate **0.005**, L2 **0.001**
   on weight matrices, seed **42**, and checkpoints at **25, 50 and 100 epochs**.
   Biases are unpenalized. Initialize weights with normal standard deviation
   inverse square root of input width, and biases to zero. No encoder is adapted.

Select candidates using only the selection role. The criterion is normalized
partial ROC area for **rejecting unsupported cases** while the useful-rejection
rate spans 0–5%, averaged equally over human, negation, relation and temporal
strata. The area is divided by 0.05; a chance scorer has area 0.025 and a perfect
scorer has area 1. Ties at 12 decimals favor fewer parameters, then stronger L2
or earlier epoch. Save the best linear and best nonlinear head as ablations, and
select one overall. Do not refit on selection or calibration.

Train a question-only control using the selected architecture and its chosen
hyperparameters, without separately tuning that control. Context-only scores
are the constant 0.5 baseline: opposite labels on every identical context make
context-only information uninformative by construction. Report raw relevance
logits and the prior support-v1 head too. All choices are made before test scoring.

## Common calibration and operating points

Each method uses **one common threshold**, with no suite label, phenomenon
label, template ID or oracle available at inference. Recalibrate all baselines
on the same new calibration data for a fair comparison.

Accept when score >= threshold. Candidates are distinct calibration scores plus
the next representable float above the maximum. The conservative threshold
maximizes equal-stratum unsupported rejection subject to **at most 2% supported
rejection in every stratum**: human, negation, relation and temporal. There are
384 supported human calibration cases (at most seven errors) and 128 supported
cases per contrast phenomenon (at most two errors each). Ties prefer lower
mean useful rejection, then the lower threshold. This empirical constraint is
not a guarantee of a 2% test/deployment error rate.

The secondary balanced threshold maximizes equal-stratum balanced accuracy,
with the same tie rule. No threshold is tuned on test. No test-selected
combination of heads or fallback is introduced.

## Test reporting and uncertainty

Report confusion counts, useful rejection, unsupported rejection, accuracy,
coverage and AUROC for all fixed methods at both operating points. Keep the
human transfer and synthetic diagnostic suites separate. Show each contrast
phenomenon's results and matched question-only control to detect shortcuts.

Use 5,000 paired bootstrap replicates with seed 42. Resample human test articles
(32 clusters); resample synthetic **template families** (9 clusters), preserving
all four-way groups inside each cluster. Report intervals for both rejection
rates and balanced accuracy, and selected-minus-previous/raw/linear differences.
The synthetic interval is exploratory: only nine families, shared concepts and
related paraphrases remain, and instance-level resampling would overstate power.
No independent human label-error uncertainty is included.

Write the selected heads and thresholds before opening test feature shards.
Publish all decisions, development candidate scores, frozen inputs and results,
including negative outcomes or useful-rejection regressions. An improvement in
synthetic examples is insufficient to claim robust natural-language abstention,
and nothing here tests navigation or generated answers. Do not change API defaults
based on this experiment alone. After this run, the test is observed evidence
and cannot be reused as a fresh holdout for further changes.

## Reproduction and budget

Use [Dockerfile.probe](../Dockerfile.probe). No new dependencies or paid compute
are required. Prior cumulative estimated cloud spend is **$7.94877 of $10**;
this experiment allows **$0 new cloud spend**. Local electricity and hardware
amortization are unmeasured. Extraction stops at a two-hour local time budget.

```sh
docker build -f Dockerfile.probe -t arborvault-probe-cpu .
mkdir -p .cache/huggingface
docker run --rm -v "$PWD:/app" -v "$PWD/.cache/huggingface:/cache" \
  -e PYTHONPATH=/app/src -e HF_HOME=/cache -e OPENBLAS_NUM_THREADS=1 \
  arborvault-probe-cpu python scripts/run_support_v2.py extract \
  --features reports/support-v2/features
docker run --rm -v "$PWD:/app" -e PYTHONPATH=/app/src -e OPENBLAS_NUM_THREADS=1 \
  arborvault-probe-cpu python scripts/run_support_v2.py analyze \
  --features reports/support-v2/features --output reports/support-v2/results
```

Fresh output directories are required. For exact numerical reproduction, use
the published feature directory and write analysis to a new output location.
Tests check finite-difference gradients, nonlinear interaction learning, label
crossings, role and source separation, and per-stratum calibration constraints.
