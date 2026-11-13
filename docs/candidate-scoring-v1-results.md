# Candidate scoring v1: development results

**Candidate scoring is promising; this Qwen adaptation and the tree policy are not demonstrated upgrades.**
On the 100 answerable development questions, BM25 followed by Qwen base scoring
reaches **80.00% Recall@5**; its new LoRA reaches **78.67%**. The paired difference
is -1.33 points, with a document-bootstrap 95% interval of [-6.12, +3.27].
There is no measured reason to replace the base with this adapter. Qwen's MRR
does rise from .5968 to .6260: ranking the first hit and recovering all annotated
evidence are distinct objectives, and the adapter does not improve both.

The retrieval-pretrained 22.7M-parameter cross-encoder improves from **72.17% to
81.00%** with flat reranking, a difference of +8.83 points [+0.51, +17.18].
The standard-text E5 reference reaches **71.00%**; the adapted cross-encoder's
observed advantage is +10.00 points [+1.01, +18.57]. These are exploratory,
unadjusted intervals on one development sample and one training seed, not a
confirmed general victory over vector retrieval. Its 81% versus Qwen base's
80% also does not establish that the smaller model is better or equivalent.
It is a practical candidate for the next validation because it uses far fewer
parameters, but actual online latency still needs measurement.

The vector input control matters: E5 scores **64.33% with JSON views and 71.00%
with standard text**. Reporting only the former would overstate the advantage.
Qwen base versus standard-text E5 is +9.00 points with an interval crossing zero
[-1.26, +19.15]. BM25 and RRF are reported below even where fusion loses recall.

The primary tree loses to matched flat reranking for every scorer. For Qwen base,
recall is **67.58% versus 80.00%**. Retaining all already-scored leaves in the
disclosed secondary control lifts tree recall to **73.33%**, still 6.67 points
below flat [-12.50, -0.84]. For the adapted cross-encoder the corresponding values
are **67.17%, 70.33% and 81.00%**. The tree spends budget on branches and can prune
useful passages; this implementation supplies no evidence that hierarchy helps
in the current short, known-document regime. This is not a conclusion about all
hierarchical search or longer-document tasks.

The stage analysis locates concrete losses. Qwen base's flat candidate pool
contains 99.00% of the annotated evidence under the availability metric. The
primary tree scores a pool containing 84.83%, then retains 69.92% after pruning;
its final recall is 67.58%. The word budget accounts for only one further point
of loss. Search coverage and leaf retention matter more here than that final
context cut. These are availability summaries, not an optimal retriever proof.

Qwen's probe loss falls from 1.8212 to .5848 and probe pairwise accuracy rises
from 78.14% to 80.51%, yet its development retrieval does not improve. A small
probe's better score is insufficient evidence of generalization. This result
does not identify a unique cause such as exposure bias, insufficient capacity,
or overfitting; all require more targeted controls.

**Abstention remains unresolved.** All reported 64-pair primary and secondary
policies abstain on **0/20** unknown development questions, including calibrated
policies. Within the declared threshold grid, calibration found no nonzero
abstention operating point satisfying the recall/empty-answerable guards.
This is a negative result for this scoring-and-threshold recipe, not a claim
that abstention is impossible. Do not present improved recall as an improvement
on no-answer handling.

**Next decision:** keep production unchanged and pause further Qwen fine-tuning.
Validate the small adapted cross-encoder with document-separated folds or new
held-out documents, more unknown questions, and independently reviewed labels.
Measure end-to-end retrieval latency before choosing a serving model. A new
hierarchy experiment should preserve scored leaves and test long documents as
well as this regime, with the same scorer and explicit search budgets. Manual
neuron edits or sacrificing math/coding ability are not justified by these data;
neither general capability was measured.


## What was tested

This is a new training run from the original Qwen3-4B and a retrieval-pretrained
MiniLM cross-encoder, not continuation of the former ID-generation adapters.
Both learn independent candidate scores from the same 6,563 pairs (1,127 positives)
on 457 training questions. Existing annotators' evidence is unioned for training;
unannotated candidates receive weaker negative weight. AI audit proposals are excluded.
All original corpus labels and historical test scores remain unchanged.

143 questions on separate training-source documents calibrate policies. A fixed
24-question subset chooses checkpoints; all 120 development questions are scored
after those choices. Selected trained steps: Qwen **115**, cross-encoder
**100**. The base is always reported. See the
[frozen protocol](candidate-scoring-v1-protocol.md) for exact rules and limitations.

## Coverage-first comparison

The two neural policies share a maximum of 64 scored candidates, k=5 and a 512-word
returned context. Tree beam width is four, including branches in the pair budget;
flat retrieval reranks BM25 candidates. Both order final passages by the same model.
Thus improvement over the old ID-generating navigator can come from the new
selection policy; the base-versus-tuned comparisons **within this table** measure
the incremental effect of this new training recipe.

| Method | Recall@5 | Evidence coverage | MRR | Empty answerable | Unknown abstentions | Mean scored candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen-base/flat/64 | 80.00% | 77.50% | 0.5968 | 0/100 | 0/20 | 42.17 |
| qwen-base/tree/64 | 67.58% | 65.08% | 0.5692 | 0/100 | 0/20 | 30.15 |
| qwen-tuned/flat/64 | 78.67% | 73.67% | 0.6260 | 0/100 | 0/20 | 42.17 |
| qwen-tuned/tree/64 | 64.58% | 61.08% | 0.5387 | 0/100 | 0/20 | 32.96 |
| cross-encoder-base/flat/64 | 72.17% | 70.17% | 0.5485 | 0/100 | 0/20 | 42.17 |
| cross-encoder-base/tree/64 | 62.50% | 60.50% | 0.5075 | 0/100 | 0/20 | 29.62 |
| cross-encoder-tuned/flat/64 | 81.00% | 73.50% | 0.5853 | 0/100 | 0/20 | 42.17 |
| cross-encoder-tuned/tree/64 | 67.17% | 62.17% | 0.5200 | 0/100 | 0/20 | 33.51 |

Reference retrievers use all passages of the known document, with the same final
context budget. Dense indexing/dot products have different cost from cross-encoding.

| Method | Recall@5 | Evidence coverage | MRR | Empty answerable | Unknown abstentions | Mean scored candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense_e5_base | 64.33% | 62.00% | 0.4643 | 0/100 | 0/20 | all document passages |
| bm25 | 53.67% | 51.17% | 0.3850 | 0/100 | 0/20 | all document passages |
| hybrid_rrf | 65.67% | 62.83% | 0.4877 | 0/100 | 0/20 | all document passages |

An additional standard-text reference uses section paths plus original passage
text instead of JSON. It was specified before the E5 stage and does not change
the neural training or selection. Both representations remain visible:

| Method | Recall@5 | Evidence coverage | MRR | Empty answerable | Unknown abstentions | Mean scored candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense_e5_text | 71.00% | 68.50% | 0.4977 | 0/100 | 0/20 | all document passages |
| bm25_text | 51.17% | 48.67% | 0.3712 | 0/100 | 0/20 | all document passages |
| hybrid_rrf_text | 61.83% | 57.33% | 0.4572 | 0/100 | 0/20 | all document passages |

See the [standard-text reference](benchmarks/candidate-scoring-v1/reference.json)
and [timestamped specification](benchmarks/candidate-scoring-v1/reference-amendment.json).
Its paired comparisons are included below. Do not select a favorable input format
and discard the other result.

All percentages concern annotated paragraph evidence, not generated-answer accuracy.
The two neural models receive the same fitted candidate text, at most 512 tokens
under either tokenizer. E5 indexes stable full views with standard prefixes and its
own 512-token truncation. The small model has prior MS MARCO retrieval supervision;
this is not a clean experiment isolating model size or general intelligence.

## Abstention is a separate operating point

Each threshold is selected on the 143 calibration questions, with an empirical
two-point recall/empty-answerable guard. No development label chooses the threshold.
The guard need not hold on development. Twenty unanswerable development questions
are too few for a strong general claim about abstention.

| Method | Recall@5 | Evidence coverage | MRR | Empty answerable | Unknown abstentions | Mean scored candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen-base/flat/64-calibrated | 80.00% | 77.50% | 0.5968 | 0/100 | 0/20 | 42.17 |
| qwen-base/tree/64-calibrated | 67.58% | 65.08% | 0.5692 | 0/100 | 0/20 | 30.15 |
| qwen-tuned/flat/64-calibrated | 78.67% | 73.67% | 0.6260 | 0/100 | 0/20 | 42.17 |
| qwen-tuned/tree/64-calibrated | 64.58% | 61.08% | 0.5387 | 0/100 | 0/20 | 32.96 |
| cross-encoder-base/flat/64-calibrated | 72.17% | 70.17% | 0.5485 | 0/100 | 0/20 | 42.17 |
| cross-encoder-base/tree/64-calibrated | 62.50% | 60.50% | 0.5075 | 0/100 | 0/20 | 29.62 |
| cross-encoder-tuned/flat/64-calibrated | 81.00% | 73.50% | 0.5853 | 0/100 | 0/20 | 42.17 |
| cross-encoder-tuned/tree/64-calibrated | 67.17% | 62.17% | 0.5200 | 0/100 | 0/20 | 33.51 |

Reference baselines were not calibrated for abstention; their zero abstentions
reflect an operating policy, not an intrinsic inability to abstain.

## Secondary control: retain all scored leaves

Before development inference, code review identified that the primary tree's
width-four frontier discards scored leaves beyond four, while flat can return five.
The [protocol amendment](candidate-scoring-v1-protocol.md#secondary-replay-amendment-before-development-inference)
preserves primary results and checkpoint selection, and adds a separate replay:
retain all scored leaves for final ranking; width four restricts only branches.
The 64-pair budget, scorer, inputs and final context remain unchanged.

| Scorer | Retain all scored leaves: Recall@5 | Difference from flat | Paired 95% interval |
| --- | ---: | ---: | ---: |
| qwen-base | 73.33% | -6.67 points | [-12.50, -0.84] |
| qwen-tuned | 68.83% | -9.83 points | [-17.71, -1.81] |
| cross-encoder-base | 66.33% | -5.83 points | [-14.12, +2.31] |
| cross-encoder-tuned | 70.33% | -10.67 points | [-18.11, -3.14] |

This is a search-policy control, not an additional model training or inference run.
Its calibrated thresholds also use calibration only. Full results and intervals
are in the [secondary replay](benchmarks/candidate-scoring-v1/secondary-replay.json),
executed with source `2369e53fc060a5afc41840fe05f9044ca8942c14`. This variant was not the
checkpoint-selection criterion, and remains a disclosed secondary analysis.
The [timestamped amendment](benchmarks/candidate-scoring-v1/secondary-amendment.json)
records that checkpoint selection and development scoring had not yet occurred.

## Uncertainty and checkpoint curves

| Paired development comparison | Recall difference | Document-bootstrap 95% interval |
| --- | ---: | ---: |
| qwen/flat/tuning | -1.33 points | [-6.12, +3.27] |
| qwen/tree/tuning | -3.00 points | [-9.76, +3.67] |
| qwen-base/tree-minus-flat | -12.42 points | [-19.42, -5.74] |
| qwen-base/flat-minus-e5 | +15.67 points | [+5.14, +25.79] |
| qwen-tuned/tree-minus-flat | -14.08 points | [-21.14, -7.10] |
| qwen-tuned/flat-minus-e5 | +14.33 points | [+3.83, +24.39] |
| cross-encoder/flat/tuning | +8.83 points | [+0.51, +17.18] |
| cross-encoder/tree/tuning | +4.67 points | [-3.80, +12.85] |
| cross-encoder-base/tree-minus-flat | -9.67 points | [-18.14, -1.74] |
| cross-encoder-base/flat-minus-e5 | +7.83 points | [-0.98, +17.37] |
| cross-encoder-tuned/tree-minus-flat | -13.83 points | [-22.05, -5.71] |
| cross-encoder-tuned/flat-minus-e5 | +16.67 points | [+6.91, +26.84] |
| qwen-base/flat/64/minus-e5-text | +9.00 points | [-1.26, +19.15] |
| qwen-base/tree/64/minus-e5-text | -3.42 points | [-14.47, +7.43] |
| qwen-tuned/flat/64/minus-e5-text | +7.67 points | [-2.08, +17.17] |
| qwen-tuned/tree/64/minus-e5-text | -6.42 points | [-17.93, +4.50] |
| cross-encoder-base/flat/64/minus-e5-text | +1.17 points | [-7.82, +9.93] |
| cross-encoder-base/tree/64/minus-e5-text | -8.50 points | [-19.27, +2.04] |
| cross-encoder-tuned/flat/64/minus-e5-text | +10.00 points | [+1.01, +18.57] |
| cross-encoder-tuned/tree/64/minus-e5-text | -3.83 points | [-16.31, +8.08] |

Intervals use 2,000 paired document-bootstrap replicates on answerable questions.
They condition on this dataset and one training seed; they do not cover seed
variation, pretraining contamination, domain shift or incomplete annotation.
They are nominal intervals, without correction for the multiple comparisons;
they support exploratory diagnosis rather than a confirmatory significance claim.

| Model | Step | Probe mean flat/tree recall | Probe weighted BCE | Probe pairwise accuracy |
| --- | ---: | ---: | ---: | ---: |
| qwen | 0 | 58.75% | 1.8212 | 78.14% |
| qwen | 25 | 55.62% | 0.6531 | 76.52% |
| qwen | 50 | 51.25% | 0.5487 | 77.50% |
| qwen | 75 | 58.75% | 0.5842 | 77.83% |
| qwen | 100 | 63.75% | 0.5181 | 78.99% |
| qwen | 115 | 66.25% | 0.5848 | 80.51% |
| cross-encoder | 0 | 48.75% | 3.3692 | 69.55% |
| cross-encoder | 25 | 51.88% | 1.4317 | 70.96% |
| cross-encoder | 50 | 53.12% | 0.9623 | 69.86% |
| cross-encoder | 75 | 57.50% | 0.7308 | 72.01% |
| cross-encoder | 100 | 64.38% | 0.6754 | 73.86% |
| cross-encoder | 115 | 61.88% | 0.6535 | 75.18% |

Probe BCE and pairwise accuracy are postprocessed from frozen scores on the same
weighted positive/weak-negative sampling scheme. They were not used to choose the
checkpoint. The small checkpoint probe can be noisy; the latest step is not
automatically selected. More frequent checkpoints do not create more independent data.

## Where annotated evidence disappears

| Primary policy | Gold available among scored candidates | Gold retained before top-k | Recall before word budget | Final recall |
| --- | ---: | ---: | ---: | ---: |
| cross-encoder-base/flat/64 | 99.00% | 99.00% | 73.67% | 72.17% |
| cross-encoder-base/tree/64 | 74.67% | 63.75% | 62.50% | 62.50% |
| cross-encoder-tuned/flat/64 | 99.00% | 99.00% | 82.00% | 81.00% |
| cross-encoder-tuned/tree/64 | 77.33% | 67.42% | 67.42% | 67.17% |
| qwen-base/flat/64 | 99.00% | 99.00% | 81.00% | 80.00% |
| qwen-base/tree/64 | 84.83% | 69.92% | 68.58% | 67.58% |
| qwen-tuned/flat/64 | 99.00% | 99.00% | 79.83% | 78.67% |
| qwen-tuned/tree/64 | 81.33% | 65.92% | 64.58% | 64.58% |

These are descriptive availability bounds on the 100 answerable development
questions, not guaranteed attainable recall. Each stage uses the best valid
annotation alternative and ignores the constraints applied by later stages.
The figures locate pruning, ranking and context losses without attributing them
to a lack of general intelligence or changing any gold label.

## Reproducibility, cost and limits

The GPU executed source `e9a25ec7835d7919b8f65fc66134a2700690afd6`. All **85**
downloaded artifact hashes, checkpoint-selection rules, calibration rules and **32**
neural policy replays were verified offline against saved scores and frozen rules.
This is reproducibility checking, not independent human review. All reported evidence metrics
were recomputed against the original corpus. The GPU environment passed **22 tests**.
Optional tensor tests also run in that environment rather than the API image.

The experiment took **45.62 minutes** after
dependency setup and peaked at **4.68 GiB** of
allocated GPU memory. Exhaustive score-cache creation and its token counts are
reported separately from simulated visited-candidate budgets. These are not
production serving latency or FLOP benchmarks; smaller pair counts do not alone
prove faster service.

Estimated allocation-to-deletion GPU/storage cost: **$2.63**;
cumulative with preceding v2 work: **$4.51 / $10**.
The allocation lasted **175.24 minutes**, including setup, waiting time and artifact
transfer; the 45.62-minute experiment runtime is not the full billable duration.
This is a runtime/rate estimate, not a final invoice.
The pod was deleted and absence verified. No production model was replaced and
no model was uploaded to Hugging Face. Checkpoint weights are retained locally;
the public [hash inventory](benchmarks/candidate-scoring-v1/model-artifact-hashes.json)
records them without committing binaries to Git.

This remains exploratory development on short English NLP papers with the document
already known. It does not establish performance on long documents, multi-document
search, ACL filtering, mathematics, coding or general reasoning. The historical test
was not re-evaluated; routine integrity tests only checked dataset consistency.
Candidate scoring is an experimental module, not a new production API default.

Artifacts: [results per question](benchmarks/candidate-scoring-v1/results.json),
[verified summary](benchmarks/candidate-scoring-v1/verified-summary.json),
[locked checkpoint choices](benchmarks/candidate-scoring-v1/selection-before-development.json),
[training plan](benchmarks/candidate-scoring-v1/plan.json),
[environment](benchmarks/candidate-scoring-v1/environment.json),
[cost and cleanup](benchmarks/candidate-scoring-v1/run-cost.json),
[public checksums](benchmarks/candidate-scoring-v1/public-manifest.json).

Reproduce with the pinned recipe:

```sh
python scripts/run_candidate_scoring.py --output /path/to/new-run
python scripts/benchmark_e5_text.py --output /path/to/reference.json
python scripts/summarize_candidate_scoring.py /path/to/new-run --reference /path/to/reference.json --output /path/to/verified-summary.json
python scripts/replay_scored_leaves.py /path/to/new-run --reference /path/to/reference.json --code-revision YOUR_SOURCE_COMMIT --output /path/to/secondary-replay.json
```

The first two commands require the pinned GPU training environment described in
the protocol. The latter two replay and check saved artifacts on CPU. The full
verifier expects the checkpoint binaries as well as their hashes; the public
report directory intentionally omits those binaries and is not a complete model
download. No Hugging Face model release is claimed.
