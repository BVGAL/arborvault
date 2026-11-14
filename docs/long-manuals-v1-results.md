# Long manuals v1: longer documents and scoped abstention

2026-11-14. Inference-only test; no new fine-tuning or production promotion.

E5 has the highest observed primary evidence recall (**95.00%**), followed by
the pretrained cross-encoder (**94.17%**), the earlier QASPER-adapted checkpoint
(**92.50%**) and Qwen base (**91.67%**). These small flat-retrieval differences
do not establish broad superiority: there are only 120 single-evidence cases
from eleven source-family groups. The adapted cross-encoder shows **no transfer
gain** over its pretrained base on this corpus.

The stronger finding is the **candidate-access failure of the current tree
policy**. At 64 pairs, Qwen's tree retrieves evidence in only **34.17%**, versus
91.67% with the same scorer in flat retrieval. Its tree even scores an annotated
passage in only 34.17%: virtually all of this loss happens before final reranking.
The flat lexical shortlist contains evidence for **118/120** questions; the
tree spends about 29 of its 64 pairs on branches. At 128 pairs the tree improves
to 62.50% for Qwen and 63.33% for the pretrained cross-encoder, still far below
their corresponding flat results. Longer documents do not rescue this fixed
traversal policy. This diagnoses the implementation's access/expansion strategy,
not a general impossibility of useful document structure or a model's intelligence.

Abstention is much easier on these deliberately scoped synthetic negatives than
on QASPER. With calibration-only conservative thresholds, the pretrained
cross-encoder rejects **160/168** unknown cases and **1/120** answerable cases;
recall remains 94.17% because that rejected answerable case was already a retrieval
failure. Qwen rejects **157/168** with **zero** false abstentions and retains
91.67% recall. The adapted checkpoint rejects only **130/168** while falsely
rejecting **5/120**, reducing recall to **88.33%**; its calibration guard does not
transfer to the test. E5's conservative setting rejects 94/168 with no false
abstentions; its balanced setting rejects 160/168, loses two answerable cases
and retains 93.33% recall. Report both operating points, rather than treating
one threshold as the model's intrinsic abstention capability.

The next justified work is to improve and test candidate access and branch-budget
allocation, with the pretrained cross-encoder and E5 retained as strong controls.
Another navigation fine-tune is not supported by these results yet. Before using
this corpus to select training changes, commission independent annotation review
and reserve new evaluation data: the current test is now observed, synthetic,
single-domain and potentially incomplete in its alternative evidence labels.

## Dataset and scope

The frozen test contains **12 manuals, 120 answerable and 168 unanswerable query-document cases** (204 distinct question texts). Four other manuals (40 answerable, 56 unanswerable cases) calibrate thresholds. No document appears in both partitions. All 16 Python reference/manual pages are new to ArborVault experiments; model pretraining exposure is unknown.

Median document length is **12,923.5 words**, versus **3,416** in the old 624-paper QASPER pool; range 8,295–25,094. These documents are longer in distribution, not longer than every old paper. Trees contain 463–1,150 nodes, including section/API branches, and 4–7 levels. Native blocks differ in granularity from QASPER paragraphs.

Questions are **assistant-authored synthetic data, without independent human adjudication**. Every answerable question targets one annotated block, with exact duplicates allowed as alternatives. Recall@5 is therefore effectively evidence hit rate here. Unknown queries comprise 84 wrong-manual transfers plus 84 requests for unspecified performance/configuration/guarantees; they do not estimate production prevalence. Semantically valid alternative evidence may remain unannotated.

The correct document is supplied. Final context is at most five blocks and 512 words. Annotated test evidence has a median of **38 words**; 91/120 targets are at depth four or deeper, counting root as depth one. This is single-block factual lookup within long manuals, not long-form synthesis. See the [complete corpus profile](benchmarks/long-manuals-v1/corpus-profile.json). This evaluates retrieval and abstention, not answer generation, corpus-wide search or a verified STAIR reproduction. See the [dataset card](../datasets/long-manuals-v1/README.md) and [frozen protocol](long-manuals-v1-protocol.md).

**Disclosed pre-results correction:** the first partial Qwen attempt was stopped after 96 logged cases, before any score cache or metric was inspected. Generic leaf titles were replacing the paragraph previews expected by the existing tree-outline code. Titles were changed to the first sentence up to 110 characters, matching the prior QASPER parser, and every system was rerun from scratch. Question text, evidence, partitions, weights and policies did not change. The interrupted attempt is recorded in the [export record](benchmarks/long-manuals-v1/export-check.json) and included in the cost estimate.

## Coverage-first retrieval

| System | Flat 64 Recall@5 | Tree 64 | Flat 128 | Tree 128 | Flat 64 MRR | Flat 64 full evidence |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B base | 91.67% | 34.17% | 92.50% | 62.50% | 0.7582 | 91.67% |
| Cross-encoder base | 94.17% | 36.67% | 95.00% | 63.33% | 0.8415 | 94.17% |
| Cross-encoder adapted (step 100) | 92.50% | 33.33% | 93.33% | 55.00% | 0.8236 | 92.50% |
| E5-base-v2 (all leaves) | 95.00% | — | — | — | 0.8086 | 95.00% |
| BM25 (all leaves) | 87.50% | — | — | — | 0.7554 | 87.50% |

E5 and BM25 search all leaves; their value in the first column is not a 64-pair evaluation. For the neural rerankers, flat means BM25 prefilter then scoring; tree uses width four, charges branches to the same pair budget and retains every scored leaf. There is no lexical fallback. Qwen and cross-encoders share tokenizer-fitted candidate text; E5 uses its fixed native text/path representation. Equal final context does not imply equal compute.

### Paired descriptive uncertainty

| Left minus right | Recall difference | 95% paired family-bootstrap interval |
|---|---:|---:|
| cross-encoder-tuned/flat-64 − qwen-base/flat-64 | +0.83 pp | [-3.57, +6.36] pp |
| cross-encoder-tuned/flat-64 − e5-text | -2.50 pp | [-5.45, +0.00] pp |
| cross-encoder-tuned/flat-64 − cross-encoder-base/flat-64 | -1.67 pp | [-4.17, +0.00] pp |
| qwen-base/flat-64 − e5-text | -3.33 pp | [-10.00, +1.43] pp |
| qwen-base/tree_scored-64 − qwen-base/flat-64 | -57.50 pp | [-67.50, -46.36] pp |
| qwen-base/tree_scored-128 − qwen-base/flat-128 | -30.00 pp | [-39.17, -21.67] pp |
| cross-encoder-base/tree_scored-64 − cross-encoder-base/flat-64 | -57.50 pp | [-69.17, -46.43] pp |
| cross-encoder-base/tree_scored-128 − cross-encoder-base/flat-128 | -31.67 pp | [-44.55, -20.63] pp |
| cross-encoder-tuned/tree_scored-64 − cross-encoder-tuned/flat-64 | -59.17 pp | [-69.09, -50.00] pp |
| cross-encoder-tuned/tree_scored-128 − cross-encoder-tuned/flat-128 | -38.33 pp | [-45.45, -30.91] pp |

5,000 paired draws, seed 42; logging reference and cookbook are one family, leaving **11 groups**. These are descriptive intervals conditional on the corpus and fixed weights, without multiplicity or training-seed adjustment. Transfers couple question topics across source/destination documents, so residual dependence may make these intervals optimistic.

### Candidate access versus reranking

| System/policy | Evidence reached by scoring stage | Mean charged pairs | Mean branch pairs |
|---|---:|---:|---:|
| qwen-base/flat-64 | 98.33% | 64.0 | 0.0 |
| qwen-base/tree_scored-64 | 34.17% | 63.9 | 29.4 |
| qwen-base/flat-128 | 100.00% | 128.0 | 0.0 |
| qwen-base/tree_scored-128 | 65.00% | 115.6 | 45.6 |
| cross-encoder-base/flat-64 | 98.33% | 64.0 | 0.0 |
| cross-encoder-base/tree_scored-64 | 37.50% | 63.9 | 28.9 |
| cross-encoder-base/flat-128 | 100.00% | 128.0 | 0.0 |
| cross-encoder-base/tree_scored-128 | 64.17% | 114.9 | 43.2 |
| cross-encoder-tuned/flat-64 | 98.33% | 64.0 | 0.0 |
| cross-encoder-tuned/tree_scored-64 | 34.17% | 64.0 | 29.4 |
| cross-encoder-tuned/flat-128 | 100.00% | 128.0 | 0.0 |
| cross-encoder-tuned/tree_scored-128 | 55.83% | 124.2 | 49.5 |

These ceilings are diagnostic only: they count annotated evidence among candidates actually scored. A gap between this ceiling and final recall arises after candidate access; evidence absent from the candidate set cannot be recovered by reranking it.

## Query-level abstention

| System | Gate | Recall after rejection | Unknowns rejected / 168 | Answerable rejected / 120 | Query coverage |
|---|---|---:|---:|---:|---:|
| qwen-base/flat-64 | coverage | 91.67% | 0 | 0 | 100.00% |
| qwen-base/flat-64 | conservative | 91.67% | 157 | 0 | 45.49% |
| qwen-base/flat-64 | balanced | 90.83% | 166 | 2 | 41.67% |
| cross-encoder-base/flat-64 | coverage | 94.17% | 0 | 0 | 100.00% |
| cross-encoder-base/flat-64 | conservative | 94.17% | 160 | 1 | 44.10% |
| cross-encoder-base/flat-64 | balanced | 94.17% | 160 | 1 | 44.10% |
| cross-encoder-tuned/flat-64 | coverage | 92.50% | 0 | 0 | 100.00% |
| cross-encoder-tuned/flat-64 | conservative | 88.33% | 130 | 5 | 53.12% |
| cross-encoder-tuned/flat-64 | balanced | 82.50% | 160 | 13 | 39.93% |
| e5-text | coverage | 95.00% | 0 | 0 | 100.00% |
| e5-text | conservative | 95.00% | 94 | 0 | 67.36% |
| e5-text | balanced | 93.33% | 160 | 2 | 43.75% |
| bm25 | coverage | 87.50% | 0 | 0 | 100.00% |
| bm25 | conservative | 87.50% | 29 | 2 | 89.24% |
| bm25 | balanced | 80.83% | 132 | 16 | 48.61% |

Confidence is the maximum reached-passage score. Thresholds come only from the calibration manuals. Conservative selection permits at most two percentage points of calibration recall loss and additional false abstention; with 40 answerable calibration questions this permits no additional false rejection there. **This is not a guarantee on the test.** Balanced selection maximizes calibration balanced accuracy with no recall-loss guard. No context is filled after rejection.

### Unknown construction categories (conservative gate)

| System | Category | Rejected | Count |
|---|---|---:|---:|
| qwen-base/flat-64 | cross_document_transfer | 79 | 84 |
| qwen-base/flat-64 | unspecified_configuration_or_guarantee | 56 | 60 |
| qwen-base/flat-64 | unspecified_performance | 22 | 24 |
| cross-encoder-base/flat-64 | cross_document_transfer | 83 | 84 |
| cross-encoder-base/flat-64 | unspecified_configuration_or_guarantee | 53 | 60 |
| cross-encoder-base/flat-64 | unspecified_performance | 24 | 24 |
| cross-encoder-tuned/flat-64 | cross_document_transfer | 66 | 84 |
| cross-encoder-tuned/flat-64 | unspecified_configuration_or_guarantee | 42 | 60 |
| cross-encoder-tuned/flat-64 | unspecified_performance | 22 | 24 |
| e5-text | cross_document_transfer | 56 | 84 |
| e5-text | unspecified_configuration_or_guarantee | 29 | 60 |
| e5-text | unspecified_performance | 9 | 24 |
| bm25 | cross_document_transfer | 19 | 84 |
| bm25 | unspecified_configuration_or_guarantee | 8 | 60 |
| bm25 | unspecified_performance | 2 | 24 |

All 42 system/policy/gate outputs, individual predictions, calibration boundaries, discrimination metrics and absolute intervals are available in the [verified machine-readable summary](benchmarks/long-manuals-v1/verified-summary.json). Never treat rejection of wrong-manual transfers as equivalent to rejecting subtle unsupported claims within the right manual.

## Reproducibility, validation and cost

- Frozen inference source: `9524486bf2226b92dc48076e0aa6433caca28e2b`. The model and dataset locks were committed before inference; no weights or policy were changed after observing test scores.
- Adapted cross-encoder: the previously selected candidate-scoring-v1 step 100, verified against its published hashes; no CV fold was selected for favorable new results.
- Independently verified 9 original artifact files, replayed every traversal, checked pair/token budgets and recomputed all thresholds from calibration only.
- Full local suite: **112 passed, six optional dependency skips**; GPU scorer/retrieval tests: **26 passed**. GitHub checks are recorded against the published commits.
- Inference process: **37.26 minutes**. This includes model loading; allocation time also includes installation and transfer.
- New estimated cost: **$0.80613**; cumulative **$5.79872 / $10**. The pod was deleted and absence verified. This allocation-to-deletion compute/storage estimate is not a final provider invoice.
- Public artifacts contain no access tokens, account identifiers, pod addresses or personal project names. Original source notices/licenses are preserved. No model-hub upload or production deployment occurred.

The first offline replay exposed a reproducibility bug in legacy BM25 summation: query terms came from an unordered set. A separate [numerical audit](benchmarks/long-manuals-v1/numerical-audit.json) found differences at most 7.11e-15 and a tie-order permutation for one query. Terms now use deterministic ordering and accurate summation. **Every recorded candidate sequence, expansion and final neural ranking reproduces exactly with zero tolerance**; no model score, selected candidate or annotation was changed. The raw artifacts remain byte-for-byte intact. A subprocess regression test covers different Python hash seeds.

Raw scores and traces are in [the frozen experiment directory](benchmarks/long-manuals-v1/experiment/plan.json), with [original file hashes](benchmarks/long-manuals-v1/experiment/artifact-hashes.json), [measured usage](benchmarks/long-manuals-v1/experiment/usage.json) and [resource/cost record](benchmarks/long-manuals-v1/run-cost.json). Checkpoint binaries remain local; exact hashes and prior training code document provenance.

Reproduce offline analysis from the repository root:

```bash
python scripts/analyze_long_benchmark.py --experiment docs/benchmarks/long-manuals-v1/experiment --output /tmp/long-manuals-verified.json
```

After this run the test is observed. Any future training or policy decisions based on these results must treat it as development data and reserve a new independently annotated evaluation.
