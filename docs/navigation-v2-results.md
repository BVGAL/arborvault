# Navigation v2: Qwen, cross-encoder and Tacit

2026-11-14. Inference only; no new fine-tuning. All three models use the same search policies and charged pair limits.

The largest measured gain comes from **changing the search policy while keeping
the scorer fixed**. At 64 pairs, Qwen improves from **21/48 (43.75%)** with the
previous tree to **41/48 (85.42%)** with incremental navigation and retained
alternatives. Adding lexical descendant guidance reaches **45/48 (93.75%)**,
matching Qwen's flat retrieval and the full-leaf E5 reference on this test.
The cross-encoder and Tacit show the same large policy improvement.

The pretrained cross-encoder has the highest observed guided-tree recall:
**46/48 (95.83%)** at 64 pairs, versus Qwen's **45/48** and Tacit's **44/48
(91.67%)**. Its flat control reaches **47/48 (97.92%)**. The guided tree therefore
does not establish a structural advantage over flat retrieval: it closes most
of the earlier implementation gap. At 128 pairs the cross-encoder's guided tree
matches its flat control at 47/48; Qwen and Tacit remain at 45/48 and 44/48.

**Tacit-4B does not demonstrate a reliable advantage over its Qwen3-4B base in
this retrieval task.** The guided-64 difference is minus one case, or **−2.08
percentage points**, with a descriptive paired interval of **[−10.00, +5.00]**.
That interval is compatible with either direction. Tacit is an independently
published, self-distilled decision derivative of Qwen, not a model trained in
this experiment. Here both use the same prompt and readout; their full chat
templates and Yes/No token IDs were also checked for equality. See the
[prompt compatibility check](benchmarks/navigation-v2/prompt-check.json).

The small pretrained cross-encoder remains a useful practical reference. Its
actual all-policy scoring stage took about **97 seconds**, versus 1,592 for
Qwen and 1,490 for Tacit on this GPU. These are experiment-stage timings with
shared caches and different input-fitting overhead, not isolated serving
latencies or a tuned inference-engine benchmark.

For Qwen, annotated-evidence access rises from **22/48** in the old tree to
**42/48** in the incremental tree and **45/48** in the guided tree. Mean branch
spending falls from **34.73** to **22.02** pairs for guided-64, leaving nearly
42 passage evaluations. This supports the diagnosis that the earlier policy
often spent its allowance before reaching the relevant text. The incremental
policy combines retained alternatives, smaller expansion batches and a branch
cap; their individual effects are not separately identified here.

**Abstention remains unresolved.** With the conservative calibration-only gate,
guided Qwen rejects **26/48** unknown cases with no false rejections, the
cross-encoder rejects **30/48** with none, and Tacit rejects **30/48** while
rejecting one answerable case. Missing runtime/configuration questions are
particularly difficult: only **5/24**, **12/24** and **8/24**, respectively, are
rejected. Semantic relevance to a manual is insufficient to establish that it
contains a deployment-specific fact. More aggressive thresholds also discard
answerable cases; all tradeoffs are reported below. E5's balanced gate rejects
35/48 unknowns and one answerable case, retaining its 45/48 evidence recall
because that rejected case was already a retrieval miss.

A qualitative review of all three strict Qwen guided-64 misses found two
substantive distinctions worth testing next: raw versus buffered I/O and class
definitions versus instance state. The correct branches were scored but left
unexpanded; flat retrieval also misranks the gold passages. The third miss
returns plausible alternative paragraphs about private pickling dispatch tables,
which the frozen gold does not include. This suggests an annotation limitation,
not an independently adjudicated extra success. The
[three-case audit](benchmarks/navigation-v2/failure-audit.json) preserves all IDs;
**no frozen labels or primary metrics were changed after inference**.

The next supported step is further validation of incremental search, with the
pretrained cross-encoder and flat retrieval retained as controls, plus explicit
answerability work on missing-information negatives. The present evidence does
not justify adopting Tacit over Qwen or starting another navigation fine-tune.
It also does not test whether a learned controller using search history could
help: the models here score candidates, while code manages the search actions.


## Frozen data and comparison

Six new Python 3.13 manuals provide 48 answerable and 48 scoped-unanswerable test cases (72 distinct question texts). Four separate, reused calibration manuals provide 40 answerable and 56 unknown cases. Test documents contain 4,778 to 9,214 words, with maximum depth 5 to 6: hierarchical, but shorter than the previous long-manual distribution.

Annotations are assistant-authored and source-checked, without independent human adjudication. Evidence may have unannotated semantic alternatives. Half the negatives concern missing runtime/configuration information; half transfer questions from another test document. The correct document is supplied. Pretraining exposure is unknown. Five source-family groups support only exploratory uncertainty estimates.

Qwen and Tacit use identical native Tacit yes/no prompts, full-vocabulary Yes/No logit margins, non-thinking inference and NF4 weights. The cross-encoder receives the same fitted candidate text. Maximum input is 512 tokens for every model; final output is at most five passages and 512 words. Earlier Qwen figures used another prompt and corpus, so they are not a direct before/after control.

See the [frozen protocol](navigation-v2-protocol.md), [dataset card](../datasets/navigation-v2/README.md), and [configuration](../configs/navigation-v2.json).

## Evidence recall without an abstention gate

The columns separate the previous breadth-frontier tree, incremental navigation with retained alternatives, and lexical-assisted navigation. Guided navigation uses global descendant BM25 signals and query-matched excerpts; it is not pure tree navigation. All branch evaluations count against the same neural pair allowance. Global lexical work is additional and disclosed, not counted as a neural pair.

### 64 charged pairs per query

| Model | Flat BM25 shortlist | Previous tree | Incremental tree | Lexical-assisted tree |
|---|---:|---:|---:|---:|
| Qwen3-4B base | 93.75% | 43.75% | 85.42% | 93.75% |
| MiniLM cross-encoder base | 97.92% | 41.67% | 89.58% | 95.83% |
| Tacit-4B | 91.67% | 37.50% | 81.25% | 91.67% |

### 128 charged pairs per query

| Model | Flat BM25 shortlist | Previous tree | Incremental tree | Lexical-assisted tree |
|---|---:|---:|---:|---:|
| Qwen3-4B base | 93.75% | 70.83% | 83.33% | 93.75% |
| MiniLM cross-encoder base | 97.92% | 64.58% | 95.83% | 97.92% |
| Tacit-4B | 91.67% | 68.75% | 89.58% | 91.67% |

Full-leaf references, outside the equal neural-pair comparison: **E5 93.75%**, **BM25 91.67%**.

## Candidate access and budget use

Candidate access is the fraction of annotated evidence present among actually scored candidates, before final ranking. A high access ceiling with lower recall indicates ranking losses; a low ceiling indicates search/access losses. The branch allowance is a cap, not a guarantee that enough leaves are reachable.

| Model / policy | Pair cap | Evidence access | Mean branch pairs | Mean leaf pairs | Mean unused pairs | Mean input tokens |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B base / flat | 64 | 100.00% | 0.00 | 64.00 | 0.00 | 14850 |
| Qwen3-4B base / tree_scored | 64 | 45.83% | 34.73 | 29.06 | 0.21 | 16232 |
| Qwen3-4B base / tree_budgeted | 64 | 87.50% | 21.94 | 42.06 | 0.00 | 16303 |
| Qwen3-4B base / tree_guided | 64 | 93.75% | 22.02 | 41.98 | 0.00 | 16749 |
| MiniLM cross-encoder base / flat | 64 | 100.00% | 0.00 | 64.00 | 0.00 | 9710 |
| MiniLM cross-encoder base / tree_scored | 64 | 41.67% | 34.98 | 28.50 | 0.52 | 10974 |
| MiniLM cross-encoder base / tree_budgeted | 64 | 91.67% | 21.85 | 42.15 | 0.00 | 11372 |
| MiniLM cross-encoder base / tree_guided | 64 | 97.92% | 22.00 | 42.00 | 0.00 | 11986 |
| Tacit-4B / flat | 64 | 100.00% | 0.00 | 64.00 | 0.00 | 14850 |
| Tacit-4B / tree_scored | 64 | 37.50% | 35.71 | 28.00 | 0.29 | 16037 |
| Tacit-4B / tree_budgeted | 64 | 87.50% | 22.10 | 41.90 | 0.00 | 16346 |
| Tacit-4B / tree_guided | 64 | 95.83% | 22.06 | 41.94 | 0.00 | 16874 |
| Qwen3-4B base / flat | 128 | 100.00% | 0.00 | 128.00 | 0.00 | 28948 |
| Qwen3-4B base / tree_scored | 128 | 75.00% | 44.94 | 61.06 | 22.00 | 25562 |
| Qwen3-4B base / tree_budgeted | 128 | 89.58% | 42.79 | 85.21 | 0.00 | 31172 |
| Qwen3-4B base / tree_guided | 128 | 97.92% | 42.00 | 86.00 | 0.00 | 31941 |
| MiniLM cross-encoder base / flat | 128 | 100.00% | 0.00 | 128.00 | 0.00 | 18591 |
| MiniLM cross-encoder base / tree_scored | 128 | 66.67% | 43.21 | 59.33 | 25.46 | 16279 |
| MiniLM cross-encoder base / tree_budgeted | 128 | 97.92% | 42.62 | 85.38 | 0.00 | 21117 |
| MiniLM cross-encoder base / tree_guided | 128 | 100.00% | 43.25 | 84.75 | 0.00 | 22877 |
| Tacit-4B / flat | 128 | 100.00% | 0.00 | 128.00 | 0.00 | 28948 |
| Tacit-4B / tree_scored | 128 | 70.83% | 46.17 | 60.77 | 21.06 | 25593 |
| Tacit-4B / tree_budgeted | 128 | 97.92% | 43.48 | 84.52 | 0.00 | 31390 |
| Tacit-4B / tree_guided | 128 | 97.92% | 42.60 | 85.40 | 0.00 | 32383 |

## Abstention calibrated on separate documents

Thresholds use the maximum reached passage score. The conservative gate allows at most a two-point calibration recall/false-rejection loss; balanced thresholds optimize calibration balanced accuracy. Neither guarantees the same behavior on new documents. Scores are not asserted to be calibrated probabilities.

| Model / policy | Gate | Recall | Unknowns rejected / 48 | Answerable cases rejected / 48 |
|---|---|---:|---:|---:|
| Qwen3-4B base / flat-64 | conservative | 93.75% | 24 | 0 |
| Qwen3-4B base / flat-64 | balanced | 91.67% | 30 | 1 |
| Qwen3-4B base / tree_guided-64 | conservative | 93.75% | 26 | 0 |
| Qwen3-4B base / tree_guided-64 | balanced | 83.33% | 33 | 6 |
| MiniLM cross-encoder base / flat-64 | conservative | 97.92% | 30 | 0 |
| MiniLM cross-encoder base / flat-64 | balanced | 97.92% | 30 | 0 |
| MiniLM cross-encoder base / tree_guided-64 | conservative | 95.83% | 30 | 0 |
| MiniLM cross-encoder base / tree_guided-64 | balanced | 95.83% | 30 | 0 |
| Tacit-4B / flat-64 | conservative | 91.67% | 30 | 1 |
| Tacit-4B / flat-64 | balanced | 85.42% | 33 | 5 |
| Tacit-4B / tree_guided-64 | conservative | 91.67% | 30 | 1 |
| Tacit-4B / tree_guided-64 | balanced | 85.42% | 35 | 4 |
| E5 full-leaf | conservative | 93.75% | 16 | 0 |
| E5 full-leaf | balanced | 93.75% | 35 | 1 |

## Paired uncertainty

Percentage-point differences with 95% paired bootstrap intervals, resampling five document families 5,000 times (ZIP/TAR manuals grouped). These are descriptive intervals with few clusters, not strong evidence of broad superiority. All comparisons, including 128-pair results, are in the machine-readable summary.

| Left minus right | Difference | 95% interval |
|---|---:|---:|
| qwen-base/tree_guided-64/coverage minus qwen-base/tree_scored-64/coverage | +50.00 | [+25.00, +77.08] |
| qwen-base/tree_guided-64/coverage minus qwen-base/tree_budgeted-64/coverage | +8.33 | [+0.00, +18.75] |
| qwen-base/tree_guided-64/coverage minus qwen-base/flat-64/coverage | +0.00 | [+0.00, +0.00] |
| cross-encoder-base/tree_guided-64/coverage minus cross-encoder-base/tree_scored-64/coverage | +54.17 | [+40.00, +66.67] |
| cross-encoder-base/tree_guided-64/coverage minus cross-encoder-base/tree_budgeted-64/coverage | +6.25 | [+2.08, +10.42] |
| cross-encoder-base/tree_guided-64/coverage minus cross-encoder-base/flat-64/coverage | -2.08 | [-4.69, +0.00] |
| tacit-4b/tree_guided-64/coverage minus tacit-4b/tree_scored-64/coverage | +54.17 | [+35.00, +70.83] |
| tacit-4b/tree_guided-64/coverage minus tacit-4b/tree_budgeted-64/coverage | +10.42 | [+2.50, +16.07] |
| tacit-4b/tree_guided-64/coverage minus tacit-4b/flat-64/coverage | +0.00 | [+0.00, +0.00] |
| tacit-4b/flat-64/coverage minus qwen-base/flat-64/coverage | -2.08 | [-10.00, +5.00] |
| tacit-4b/tree_guided-64/coverage minus qwen-base/tree_guided-64/coverage | -2.08 | [-10.00, +5.00] |
| cross-encoder-base/tree_guided-64/coverage minus qwen-base/tree_guided-64/coverage | +2.08 | [-7.81, +15.00] |
| qwen-base/tree_guided-64/coverage minus e5-text/coverage | +0.00 | [-15.00, +9.38] |
| cross-encoder-base/tree_guided-64/coverage minus e5-text/coverage | +2.08 | [+0.00, +7.50] |
| tacit-4b/tree_guided-64/coverage minus e5-text/coverage | -2.08 | [-12.50, +7.81] |

## Compute, verification and reproducibility

| Model | Actual experiment seconds | Unique evaluated pairs | Actual processed input tokens |
|---|---:|---:|---:|
| Qwen3-4B base | 1592.0 | 55700 | 13642341 |
| MiniLM cross-encoder base | 97.0 | 54393 | 9439412 |
| Tacit-4B | 1490.0 | 55314 | 13614281 |

These timings cover all policies with shared score caches and input fitting; they are not isolated serving latencies or equal-FLOP comparisons. Guided branch representations have their own cache keys; leaves reuse identical scores. Every policy is independently charged for its selected pairs even when another policy has already computed them.

Allocated GPU/storage cost estimate: **USD 0.90686**, cumulative **USD 6.70558 / 10**. The estimate includes installation, model downloads, waiting and artifact transfer, from allocation request through verified deletion; it is not a final provider invoice. The single GPU was deleted and the experiment inventory is empty.

Inference source: `158f55524b57695171998cda7db3dc5c03ac33f0`. Analysis script SHA-256: `70a023df237c3612fcceac5cf398bacd8481518327a7d62c1f67346d3c496a2f`. All 9 original artifact hashes and all policy traces were checked. Incremental replay additionally checks ancestor access, unique charges and no unscored branch expansion. Full local suite: 123 passed, six optional skips.

Artifacts: [verified summary and query-level results](benchmarks/navigation-v2/verified-summary.json), [raw scores and traces](benchmarks/navigation-v2/experiment), [cost record](benchmarks/navigation-v2/run-cost.json), [public hashes](benchmarks/navigation-v2/public-manifest.json).

No production default is changed by this benchmark, and no new model weights are published. The learned-controller hypothesis remains untested: here the model scores candidates while ordinary code chooses expansion, stopping and budget allocation. This project is independent of IBM, TypeSafe and the Tacit authors; it is not a verified STAIR reproduction.
