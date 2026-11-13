# Candidate scoring v1: frozen development experiment

This experiment follows the [checkpoint diagnosis](routing-v2-diagnostics.md).
It tests a specialized scalar scorer, not manual neuron editing or a larger model.
It is independent research, not an official or verified implementation of IBM STAIR.

## Data and uncertainty

The original QASPER v2 corpus, license, paragraph IDs and human labels are unchanged.
The 600 original training questions are partitioned by a fixed hash of document ID:
457 questions on 333 documents train both scorers; 143 questions on 93 other
documents calibrate policies. Calibration contains 29 unanswerable questions.
The 120 original development questions are evaluated after checkpoint selection.
The experiment runner does not load or evaluate the original test. Routine dataset
integrity tests read split records to check hashes, source mappings and document
separation; they produce no model predictions or performance-based selection.
The raw `test_not_opened` flags refer to the experiment runner, not those integrity
checks. Development has been inspected in prior work;
these results are exploratory, not a fresh held-out confirmation.

Training contains 6,563 candidate pairs, including 1,127 annotated positives.
All annotations contribute positive passages and their ancestors. The root is
excluded. For branches and passages separately, sample up to four BM25-hard and
two hash-selected unannotated candidates. Unannotated candidates are **weak
negatives**, not verified irrelevant passages. Positive mass is 1, weak-negative
mass is .25, normalized per question; human-agreed unanswerable questions have
negative mass 1. This heuristic does not eliminate label noise and is not an
unbiased positive-unlabeled estimator. The AI audit proposals are not new labels.

## Models and learning

- Original pinned Qwen3-4B, NF4/BF16, non-thinking. Its scalar is the next-token
  `yes` logit minus the `no` logit. No ID generation or fixed target cardinality.
- Pinned `cross-encoder/ms-marco-MiniLM-L6-v2` (about 22.7M parameters), a
  retrieval-pretrained scalar scorer. Its prior retrieval supervision differs
  from Qwen's; this is a practical baseline, not a controlled model-size study.

Both receive the same question/candidate texts, groups, labels, sample weights,
query order, one epoch and seed 42. Excerpts start at 1,400 characters and are
shortened identically until **both** tokenizers fit 512 tokens. Full questions
and paths are preserved; an oversized header causes failure, not silent truncation.
Binary cross-entropy is independent per candidate, balanced per question.

Qwen retains the preceding LoRA rank 8, alpha 16, q/v projections, dropout .05,
constant LR 5e-5, zero weight decay and gradient clip 1. The small cross-encoder
updates all weights at LR 2e-5. Four queries accumulate per update (115 updates),
with microbatches of four pairs. This changes format, examples and objective
together: results cannot isolate any one change's causal contribution.

Save checkpoints at steps 25, 50, 75, 100 and 115. A fixed 24-question probe
(20 answerable/four unknown, distinct calibration documents) measures actual
flat/tree retrieval at each checkpoint. Select the trained checkpoint with best
mean flat/tree Recall@5, then evidence coverage/MRR, then earliest step. Report
epoch zero even if it wins. The small probe and one training seed limit certainty.
No model is automatically promoted to production.

## Retrieval and fair accounting

Primary comparison: the **same scorer**, flat versus tree, maximum 64 scored
candidates per query, final k=5 and 512 context words. Secondary budgets: 32/128.
The tree uses beam width four; branches consume the same pair budget as passages.
When a frontier exceeds the remaining budget, BM25 prefilters it. Flat retrieval
uses BM25 to select at most the same number of passages. Ties use source order.
Both rank final passages by the scorer, not a separate lexical reranker.

Full independent candidate scores are cached offline to replay all policies.
Only scores of visited candidates affect tree choices; unvisited evidence cannot
rescue a path. Report actual exhaustive inference usage **separately** from logical
visited-candidate/token counts. Equal pair caps do not mean equal FLOPs across
model sizes, equal token counts, or a measured online latency. A flat/tree
difference includes frontier restrictions and branch overhead, not just structure.

Report coverage-first policies without a threshold and a separate abstention
policy. Calibrate thresholds only on the 143 calibration questions. Candidate
thresholds are predeclared quantiles of the maximum reached passage score.
Maximize unknown abstentions subject to at most two percentage points of recall
loss and at most two percentage points of additional empty answerable queries.
These are empirical calibration guards, not guarantees on new documents.

Reference retrievers: pinned E5-base-v2, BM25 (k1=1.2, b=.75, positive IDF), and
their rank fusion (RRF k=60), over all passages of the known document, sharing
the final context budget. E5 uses its required query/passage prefixes and 512-token
truncation; unlike the two scorers, it indexes stable full candidate views.
Baseline index/dot-product cost differs from cross-encoding; do not equate them.
Reference baselines have no abstention threshold; zero unknown abstentions are a
policy choice, not evidence that dense or lexical retrieval cannot abstain.

Report all query-level results, Recall@5, MRR, evidence coverage, unknown abstentions,
empty answerable counts, checkpoint curves and paired document-bootstrap intervals
(2,000 replicates). Those intervals do not cover seed/domain/label uncertainty.
No generated-answer or corpus-wide search claim follows from this pilot.

## Secondary replay amendment, before development inference

While Qwen training was still running, code review identified a policy confound:
the original tree beam discards already-scored leaves beyond four per frontier,
whereas flat ranking can return five. Preserve the primary protocol and checkpoint
selection unchanged, and report an additional `tree_scored` replay. It retains
all already-scored leaves for final top-five ranking and applies width four only
to branch expansion. Scored branches and leaves still share the 64-pair cap.
This changes the search policy, not weights, labels, inputs or the scoring cache.
Calibrate its separate threshold on the same calibration set only. Report its
paired difference from flat and from the primary tree; do not silently replace
the original tree result. No new model inference or training is needed for replay.

## Reproduction and resource limit

```sh
pip install -c constraints.txt '.[dev]'
pip install -r requirements.training.txt
python scripts/run_candidate_scoring.py --output /path/to/new-run
```

Use `--prepare-only` to validate the data partition without GPU dependencies.
The recipe and pinned model revisions are in
[`configs/candidate-scoring-v1.json`](../configs/candidate-scoring-v1.json).
Use one GPU, a 9,000-second process deadline, an independent three-hour allocation
stop deadline, and delete the pod after verified artifact download. The cumulative
authorized envelope remains USD 10, including USD 1.87386 previously estimated.
No persistent network volume, automatic production deployment, or model-hub upload.

Sources: [LoRA](https://arxiv.org/abs/2106.09685),
[cross-encoder model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2),
[E5 model card](https://huggingface.co/intfloat/e5-base-v2).
