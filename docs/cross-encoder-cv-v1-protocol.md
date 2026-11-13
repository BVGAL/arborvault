# Cross-encoder CV v1: frozen protocol

This retrospective experiment follows [candidate scoring v1](candidate-scoring-v1-results.md).
It evaluates the small scorer with five document-separated outer folds and treats
abstention as a separate query-level decision. It is independent research, not
an official or verified IBM STAIR reproduction. No production default changes.

## Data roles and leakage controls

Pool all 840 existing QASPER questions from historical train, development and test.
Every document is assigned to exactly one outer evaluation fold by hash order and
round-robin allocation, independently of answer labels. Questions from one document
remain together. Reject duplicate full-document contents under different IDs.
Each question obtains exactly one out-of-fold prediction from a newly initialized
model that did not train, select a checkpoint or calibrate on that document.

Within each outer fold, hash-order the remaining documents with a fold-specific
salt and cycle through 20 positions: 12 fit, two checkpoint selection, three gate
fit and three threshold calibration. These roles are mutually document-disjoint;
roughly 60/10/15/15 percent of the outer training documents. Report actual counts.
The final checkpoint is not retrained on calibration documents. This intentionally
uses less than four full outer folds for weight updates in exchange for separated
calibration. Require at least five questions of each class in every role.

Historical test figures remain unchanged, but the old test is **now reused inside
this CV experiment**. The corpus and recipe were previously inspected; this is not
a newly untouched benchmark, and CV does not undo prior research selection.
Labels and all annotators' evidence remain unchanged; AI audit proposals are excluded.

## Models, training and retrieval

Use the same pinned MiniLM cross-encoder, Qwen3-4B and E5-base-v2 revisions as
[scoring v1](candidate-scoring-v1-protocol.md). Initialize each of five cross-encoders
from the original retrieval-pretrained weights, not the earlier adapted checkpoint.
Use seed 42 for each fold, one epoch, AdamW LR 2e-5, no weight decay, gradient clip 1,
four accumulated questions and microbatches of four pairs. Retain all checkpoints
at steps 25, 50, 75, 100 and the final step when applicable. Select the trained
checkpoint by selection-set flat Recall@5, then evidence coverage/MRR, earliest tie.
Always include the unadapted cross-encoder reference, even if it wins selection.

Training preserves the existing positive-union/weak-negative recipe, including
branch examples and query-normalized binary loss; this is not new label creation.
Both neural scorers receive identical candidate text fitted to both tokenizers'
512-token limits. Flat retrieval uses the existing BM25 prefilter, at most 64
passages, then scorer ranking. Only those passages are inferred; no oracle candidate
is added. Return at most five passages and 512 words. Qwen is fixed NF4/BF16 base
with the independent yes-minus-no logit; no new Qwen fine-tuning is performed.

E5 embeds standard path-plus-passage text with query/passage prefixes, 512-token
truncation and normalized FP32 mean pooling. It searches all passages of the known
document and shares the final context budget. Dense indexing and cross-encoding
have different computational costs. Qwen, E5 and unadapted cross-encoder scores
are computed once without labels and reused across folds. Prior run scores are
not substituted. The experiment evaluates flat retrieval, not a new tree policy.

## Abstention controls

The gate accepts or rejects the **whole query**. When accepted, preserve its
complete ranked context; do not discard individual passages by a relevance threshold.
For each model and fold compare:

- Coverage-first: always return the retrieved context.
- Raw maximum score, with conservative and balanced thresholds.
- A learned answerability gate, with conservative and balanced thresholds.

Fit a small logistic model only on gate-fit documents. Features are highest score,
first/second score gap, mean top-five score, score standard deviation and query-term
overlap with the highest-scoring passage. Use gate-fit standardization, no class
reweighting, mean binary cross entropy plus .01/2 times squared coefficient norm;
do not penalize the intercept. Use a fixed Newton optimizer. Export coefficients,
scaling, iterations and training counts. Its probabilities are model estimates,
not guaranteed calibrated confidence or proof of answerability.

Choose thresholds on separate threshold-calibration documents, never outer labels.
Search all unique calibration confidence boundaries, plus always-keep/reject-all.
Conservative selection maximizes unknown abstentions subject to at most two points
of absolute recall loss and two points of extra false abstention on answerable
questions; then maximize recall and prefer fewer false abstentions/lower threshold.
Balanced selection maximizes (answerable keep rate + unknown rejection rate)/2,
then recall, then fewer false abstentions/lower threshold. It has **no recall-loss
guard** and may sacrifice substantial useful retrieval. Publish both operating
points without retrospectively selecting one from outer scores. Calibration guards
are empirical, not guarantees on outer folds or production traffic.

## Reporting and resource limits

Lock weights, gate and thresholds before computing each fold's outer metrics.
Publish fold-level and pooled out-of-fold results, Recall@5, evidence coverage,
MRR, answerable false abstentions, unknown rejection, balanced accuracy, retained
query coverage and conditional recall. Report answerability AUROC and unknown
average precision with its prevalence reference. "Unsupported return rate" means
returned unknown queries or answerable queries with zero annotated-evidence recall,
divided by retained queries; it is not a generated-answer error rate. Incomplete
evidence annotation can overstate that rate.

Use 2,000 paired document-bootstrap replicates for retrieval and abstention deltas,
including zero-effect comparisons. These conditional intervals resample saved
out-of-fold predictions, not training runs. They do not account fully for overlapping
training sets across folds, training-seed variance, prior selection, incomplete labels
or domain shift; they are not a guaranteed CI for the complete CV learning procedure.
Report all five folds and nominal intervals without multiple-comparison correction.
Any confidence/risk curves are descriptive, never used to retune on outer outcomes.

Use one GPU at at most USD 1.20/hour, a two-hour runner deadline and independent
three-hour allocation stop. Prior estimated cumulative spend is USD 4.50519; the
total envelope stays USD 10. Include setup, idle/download time and storage, retain
verified artifacts, then delete the pod. No automatic retry allocation, persistent
network volume, production promotion or model-hub upload.

```sh
pip install -c constraints.txt '.[dev]'
pip install -r requirements.training.txt
python scripts/run_cross_encoder_cv.py --output /path/to/new-cv
```

Use `--prepare-only` to inspect fold roles without GPU work. The config is
[`configs/cross-encoder-cv-v1.json`](../configs/cross-encoder-cv-v1.json).
Methodological references: [group-separated evaluation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
and [separate threshold tuning](https://scikit-learn.org/stable/modules/classification_threshold.html).
This implementation uses deterministic hash groups, not StratifiedGroupKFold.
