# Support classifier v1: frozen protocol

## Question and scope

Does training on human-labeled, topically relevant but unsupported question/paragraph
pairs improve the context-acceptance classifier, relative to its previously
published head? This evaluates **paragraph support**, not navigation, retrieval
recall, final-answer quality or an IBM STAIR reproduction.

This protocol, configuration, dataset and analysis code are committed and published
before feature extraction on the held-out test. There is one specified analysis;
test labels cannot choose regularization, thresholds, architecture or reported
operating points. Previous manual/QASPER tests are development evidence and are
not used as new held-out results.

## Data

See the [dataset card](../datasets/support-v1/README.md). Fit uses 4,062 questions
on 128 articles, selection 768 on 32 articles, calibration 768 on 32 articles,
and test 1,120 on all 35 original SQuAD 2.0 dev articles. Each paragraph is paired
with one positive and one original human negative. A pre-outcome AI-assisted audit
of 50 fit questions excludes six ambiguous pairs without relabeling them.
All text is complete within the fixed 512-token model limit. The public dev split
is new to this ArborVault experiment; pretraining exposure is unknown.

## Encoder and training

Use `cross-encoder/ms-marco-MiniLM-L6-v2` at revision
`233902d25c440f23af6f7d6e94d2946bac0bee0a`. The encoder remains frozen.
The only newly trained parameters are a standardized 384-dimensional logistic
linear head and intercept. This is **not Qwen fine-tuning** and not end-to-end
cross-encoder fine-tuning.

The representation is the last-layer CLS vector, identical to the previous
acceptance probe's representation. CPU float32 eager attention, right padding,
batch size 16, four CPU threads and seed 42 are fixed. Vector elements and raw
relevance logits are rounded to seven decimals before fitting and scoring.
Raw logits and CLS features come from the same forward pass. Only question and
paragraph text enter the encoder; IDs, labels, answer spans, distractors, titles
and split metadata do not.

Reuse `answerability.fit_probe` without changing its objective: equal question
weights, logistic loss, L2 on standardized weights but not the intercept.
Fit L2 values 0.1, 1 and 10 on fit articles. Choose by article-macro AUROC on
selection articles; ties at 12 decimal places prefer larger L2. Do not refit
on selection or calibration. Apply the same procedure separately to the
question-only and context-only controls. Identical opposite-label context twins
make a context-only learner uninformative by construction.

## Comparisons and calibration

Report all five systems:

1. Newly trained question-plus-context probe.
2. Previous published probe with frozen coefficients and standardized features.
3. Original cross-encoder relevance logit, without new training.
4. Newly trained question-only control.
5. Newly trained context-only control.

All five calibrate thresholds on the **same new calibration articles**, including
the old probe. Thus the primary comparison isolates retraining the head rather
than combining retraining with a threshold-transfer disadvantage. The raw scorer
is a relevance baseline; it was not originally trained to verify answer support.

Accept if score >= threshold. Candidate thresholds are every distinct calibration
score plus the next representable float above the maximum. The conservative
operating point maximizes unsupported-context rejection subject to at most 2%
rejection of supported contexts on calibration. With 384 supported calibration
examples, this permits at most seven errors; it is an empirical constraint,
**not a guarantee on test or deployment**. The balanced point maximizes the
mean of supported acceptance and unsupported rejection. Ties prefer fewer
supported rejections, then the lower threshold. No thresholds are chosen on test.

## Evaluation and uncertainty

The primary comparison is balanced accuracy of new versus previous probe at the
balanced operating point. Report both operating points for all methods, including
supported contexts rejected, unsupported contexts accepted, coverage, unsupported
fraction among accepted contexts, overall and article-macro AUROC.

Use 5,000 paired article-cluster bootstrap replicates with seed 42, resampling the
35 test articles and retaining all paired questions inside each sampled article.
Report percentile 95% intervals for balanced accuracy and new-minus-previous and
new-minus-raw differences. These intervals describe variation across this sample;
topics can be related and residual annotation uncertainty is not modeled.
Secondary comparisons are exploratory, without multiple-comparison correction.

Freeze and write selected coefficients and thresholds before the analysis opens
the held-out feature shard. Save scores and decisions for every test question.
Do not relabel test errors or publish a test-selected winner as a production default.
The new classifier remains an explicit research artifact, separate from the API.

## Runtime, cost and reproduction

No cloud resources are allocated. Previous cumulative estimated spend is
**$7.94877 of $10**. Additional cloud spend for this experiment is fixed at zero.
Local electricity and hardware amortization are not measured. Feature extraction
has a two-hour local time limit checked between batches.

```sh
docker build -f Dockerfile.probe -t arborvault-probe-cpu .
mkdir -p .cache/huggingface
docker run --rm -v "$PWD:/app" -v "$PWD/.cache/huggingface:/cache" \
  -e PYTHONPATH=/app/src -e HF_HOME=/cache -e OPENBLAS_NUM_THREADS=1 \
  arborvault-probe-cpu python scripts/run_support_probe.py extract \
  --features reports/support-v1/features
docker run --rm -v "$PWD:/app" -e PYTHONPATH=/app/src -e OPENBLAS_NUM_THREADS=1 \
  arborvault-probe-cpu python scripts/run_support_probe.py analyze \
  --features reports/support-v1/features --output reports/support-v1/results
```

The commands use fresh output locations and refuse overwrites. For the exact
published analysis, use the committed feature directory instead of re-extraction.
`requirements.probe-cpu.txt` pins the additional CPU stack. Model downloads are
public and require network access, but no provider credentials. The ordinary
ArborVault test environment does not require downloading the model; an optional
real-encoder regression checks exact whole-pair encoding and agreement with the
old CLS feature path.
