# Expanded navigation and abstention validation

2026-11-14. Frozen transformer weights; a learned linear acceptance probe. All evaluation rules were fixed before test inference.

The larger test confirms the navigation repair, but **does not establish a benefit over flat retrieval or a successful conservative abstention improvement for the cross-encoder**. No production default is changed.

At 64 charged candidates, guided MiniLM reaches **95.42%** evidence recall, compared with **52.08%** for the previous tree and **89.17%** for the incremental tree without global descendant guidance. However, its flat version also reaches **95.42%**, with identical per-question recall outcomes. Qwen guided reaches **89.58%** versus **88.96%** flat; their paired difference is +0.63 points, with an exploratory 95% interval of [-1.67, +3.54]. E5 reaches **90.83%**, and BM25 **92.29%**. MiniLM's +4.58-point difference from E5 has an interval of [0.00, +8.33]; this is not firm evidence of general superiority. The strong lexical reference also reflects the explicit API names in this corpus.

The primary cross-encoder conservative comparison is exactly null: both raw relevance and raw-plus-context gates reject **157/240 unknowns** and **7/240 answerable cases**, retaining **92.92%** recall. The selected context threshold is `null`, meaning the additional filter is inactive. Raw relevance already rejects all **80/80 wrong-document transfers**, but only **35/80 runtime/input questions** and **42/80 invented-API questions**. The remaining failures are about evidence sufficiency within a related topic.

The predeclared balanced operating point offers a modest tradeoff: raw-plus-context rejects **205/240 unknowns** and **25/240 answerable cases**, versus **206/240** and **33/240** for balanced raw relevance. Recall increases from **84.58% to 86.25%**. The recall difference interval is [-1.96, +4.35] points; the false-rejection difference interval is [-6.25, +0.46]. Neither excludes no change. These figures are reported as a secondary result, not used to select a new deployment threshold. Question-only acceptance alone rejects only **61/240 unknowns** while discarding **21/240 answerable cases**; adding it to the raw conservative gate rejects 186 unknowns but 27 answerable cases. It is not an effective replacement.

The same conservative context gate raises Qwen's unknown rejection from **138 to 160/240**, with **zero false rejections** and unchanged **89.58%** recall in this sample. For E5 it raises rejection from **129 to 152/240**, while false rejections rise from **2 to 3/240** and recall falls from **90.00% to 89.58%**. The benefit is therefore retriever-dependent; it cannot be transferred to MiniLM by assumption.

## Why the conservative gate is inactive

With only 48 calibration positives, the two-percentage-point constraint permits zero extra document-level false rejections. The lowest positive context score is **0.4414** for `a-shutil-00048`, which asks what filename `shutil.copy` uses when copying into a directory. Its returned context misses the basename rule. All eleven unknown cases still accepted by raw calibration score above this example. Rejecting even one additional unknown therefore first rejects this answerable query, violating the frozen constraint even though its current context is unsupported and its evidence recall is already zero. This is an objective conflict between document answerability and returned-context sufficiency, not evidence that the correct response is to increase model size.

The constraint is not the whole explanation. A supported `sys` question scores only 0.6072, below several unknowns; the context-only conservative filter also underperforms raw relevance on test. Threshold relaxation alone would trade away valid evidence. The [calibration diagnostic](benchmarks/abstention-v3/calibration-diagnostic.json) contains the actual questions, scores and returned passages. No verifier truncation occurs in the low-score calibration examples discussed here.

## Error audit and next experiment

An assistant reviewed **50 deterministically selected failures plus five supplementary cases**, covering every false rejection. The 55 cases comprise **26 runtime/input failures**, **22 invented-API failures**, **six rejected contexts containing the full annotated evidence**, and **one retrieval miss followed by rejection**. Generic API descriptions, example values and defaults do not establish the state of a particular application. Exceptions or parameters of a related real API do not validate an invented symbol. All seven conservative false rejections come from the raw gate; the new conservative gate makes no additional decisions.

The calibration audit also flags a plausible unannotated alternative for `a-functools-00038`: retrieved text says an LRU cache reuses previously computed values, while the frozen gold passage describes avoiding expensive repeated calls. This needs independent adjudication. Labels and scores remain unchanged. The audit is qualitative, not independent human review or a population estimate; its selections and complete returned contexts are published.

A next training experiment should contrast **the same question with supporting and topically related unsupported contexts**, include incorrect API assumptions and cases where examples/defaults are insufficient, and retain a question-only control. Retrieval recovery and final abstention should be evaluated separately: withholding a bad context must still count as a failure to answer a document-answerable query, while being distinguished from rejecting good evidence. Thresholds need more calibration documents and independently reviewed labels. Any such changes require a fresh held-out test; this test is now observed. The current results do not justify another large-model fine-tuning run or a production switch.


## Test and limitations

20 new Python manuals contain 240 answerable and 240 unknown query-document cases (400 distinct question texts). Four new calibration manuals contain 48 of each. Historical probe fitting and selection use 18 and four separate old manuals. The correct document is supplied. This is a larger single-domain, mixed-length study, not multi-domain or corpus-wide search.

Annotations are assistant-authored synthetic cases without independent human adjudication. Exact duplicate evidence alternatives and three two-passage command-name labels are included; semantic alternatives may be incomplete. Unknown cases split evenly between unprovided runtime/input facts, undocumented invented API assumptions, and wrong-document transfers. Most positive cases ask a single API fact with explicit names; this does not establish multi-hop or multi-document reasoning. The 80 invented-API negatives reuse four templates, and runtime negatives contain recurrent application-state wording. Model pretraining exposure is unknown. Test labels were not used for fitting, model selection, threshold choice or navigation changes.

See the [frozen protocol](abstention-v3-protocol.md), [dataset card](../datasets/abstention-v3/README.md), and [configuration](../configs/abstention-v3.json).

## Retrieval at 64 charged candidates

Five passages and 512 words maximum. Branches count against the 64-pair allowance. Lexical-guided search uses global BM25 preprocessing, outside the neural pair count. E5 and BM25 inspect all leaves and are external references.

| Model | Flat | Previous tree | Incremental tree | Lexical-assisted tree |
|---|---:|---:|---:|---:|
| Qwen3-4B base | 88.96% | 46.88% | 85.00% | 89.58% |
| MiniLM cross-encoder base | 95.42% | 52.08% | 89.17% | 95.42% |

Recall is the mean fraction of gold passage IDs appearing in the final context, even if the last passage is shortened. Evidence coverage is the mean fraction of gold passages retained in full; it is not a binary complete-set success rate. Alternative evidence sets are accepted. Neither metric is generated-answer accuracy.

Full-leaf E5 recall: **90.83%**. BM25: **92.29%**.

## Abstention and evidence

Conservative thresholds permit at most two percentage points of extra calibration false rejection and recall loss. With 48 positives this permits zero additional false rejections. This is an empirical calibration constraint, not a guarantee on test documents. Balanced thresholds maximize calibration balanced accuracy. The joint gate requires both raw relevance and the learned probe to accept. Every learned gate adds one MiniLM forward pass, at most 512 tokens, after retrieval.

| Retriever | Acceptance gate | Recall | Fully retained evidence | Unknowns rejected / 240 | Answerable rejected / 240 |
|---|---|---:|---:|---:|---:|
| Cross-encoder guided | coverage | 95.42% | 95.42% | 0 | 0 |
| Cross-encoder guided | conservative | 92.92% | 92.92% | 157 | 7 |
| Cross-encoder guided | context-alone-conservative | 92.50% | 92.50% | 134 | 8 |
| Cross-encoder guided | context-joint-conservative | 92.92% | 92.92% | 157 | 7 |
| Cross-encoder guided | question_only-joint-conservative | 84.79% | 84.79% | 186 | 27 |
| Cross-encoder guided | question_only-alone-conservative | 86.88% | 86.88% | 61 | 21 |
| Cross-encoder guided | balanced | 84.58% | 84.58% | 206 | 33 |
| Cross-encoder guided | context-joint-balanced | 86.25% | 86.25% | 205 | 25 |
| Qwen guided | coverage | 89.58% | 89.58% | 0 | 0 |
| Qwen guided | conservative | 89.58% | 89.58% | 138 | 0 |
| Qwen guided | context-alone-conservative | 89.58% | 89.58% | 116 | 0 |
| Qwen guided | context-joint-conservative | 89.58% | 89.58% | 160 | 0 |
| Qwen guided | question_only-joint-conservative | 81.46% | 81.46% | 179 | 21 |
| Qwen guided | question_only-alone-conservative | 81.46% | 81.46% | 61 | 21 |
| Qwen guided | balanced | 88.33% | 88.33% | 174 | 6 |
| Qwen guided | context-joint-balanced | 84.58% | 84.58% | 201 | 17 |
| E5 full-leaf | coverage | 90.83% | 90.83% | 0 | 0 |
| E5 full-leaf | conservative | 90.00% | 90.00% | 129 | 2 |
| E5 full-leaf | context-alone-conservative | 90.42% | 90.42% | 114 | 1 |
| E5 full-leaf | context-joint-conservative | 89.58% | 89.58% | 152 | 3 |
| E5 full-leaf | question_only-joint-conservative | 82.71% | 82.71% | 160 | 22 |
| E5 full-leaf | question_only-alone-conservative | 83.12% | 83.12% | 61 | 21 |
| E5 full-leaf | balanced | 81.04% | 81.04% | 195 | 26 |
| E5 full-leaf | context-joint-balanced | 78.75% | 78.75% | 213 | 36 |

## Unknown categories

Each category contains 80 test cases.

| Retriever | Gate | Missing runtime/input | Undocumented API | Wrong document |
|---|---|---:|---:|---:|
| Cross-encoder | conservative | 35/80 | 42/80 | 80/80 |
| Cross-encoder | context-joint-conservative | 35/80 | 42/80 | 80/80 |
| Cross-encoder | question_only-joint-conservative | 64/80 | 42/80 | 80/80 |
| Qwen | conservative | 19/80 | 42/80 | 77/80 |
| Qwen | context-joint-conservative | 29/80 | 51/80 | 80/80 |
| Qwen | question_only-joint-conservative | 60/80 | 42/80 | 77/80 |
| E5 | conservative | 32/80 | 17/80 | 80/80 |
| E5 | context-joint-conservative | 39/80 | 33/80 | 80/80 |
| E5 | question_only-joint-conservative | 63/80 | 17/80 | 80/80 |

## Probe fitting and truncation

| Probe | Eligible historical fitting examples | Historical selection examples | Selected L2 |
|---|---:|---:|---:|
| context | 1121 | 272 | 0.1 |
| question_only | 383 | 94 | 0.1 |

fit: 104/1152 context examples are truncated to the verifier token allowance. Counts include multiple retrieval policies per question.

selection: 16/288 context examples are truncated to the verifier token allowance. Counts include multiple retrieval policies per question.

calibration: 11/480 context examples are truncated to the verifier token allowance. Counts include multiple retrieval policies per question.

test: 106/2400 context examples are truncated to the verifier token allowance. Counts include multiple retrieval policies per question.

Historical positives require complete gold evidence before verifier token truncation; this does not guarantee the retained token prefix still contains it. This probe is an empirical acceptance classifier, not an entailment proof.

## Paired uncertainty

5,000 paired bootstrap draws over nine related-manual families. Intervals remain exploratory; synthetic cases and related manuals are not independent natural user queries. Family resampling does not separately cluster question texts reused across families. Values are percentage-point differences.

| Comparison | Recall change [95% interval] | Unknown rejection change [95% interval] | False rejection change [95% interval] |
|---|---|---|---|
| qwen-base/tree_guided-64/coverage minus qwen-base/tree_scored-64/coverage | +42.71 [+33.80, +51.88] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| qwen-base/tree_guided-64/coverage minus qwen-base/tree_budgeted-64/coverage | +4.58 [+0.36, +11.25] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| qwen-base/tree_guided-64/coverage minus qwen-base/flat-64/coverage | +0.63 [-1.67, +3.54] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| cross-encoder-base/tree_guided-64/coverage minus cross-encoder-base/tree_scored-64/coverage | +43.33 [+34.38, +51.59] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| cross-encoder-base/tree_guided-64/coverage minus cross-encoder-base/tree_budgeted-64/coverage | +6.25 [+2.08, +13.24] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| cross-encoder-base/tree_guided-64/coverage minus cross-encoder-base/flat-64/coverage | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| cross-encoder-base/tree_guided-64/coverage minus qwen-base/tree_guided-64/coverage | +5.83 [+0.98, +9.78] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| qwen-base/tree_guided-64/coverage minus e5-text/coverage | -1.25 [-5.56, +3.95] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| cross-encoder-base/tree_guided-64/coverage minus e5-text/coverage | +4.58 [+0.00, +8.33] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| qwen-base/flat-64/context-joint-conservative minus qwen-base/flat-64/conservative | +0.00 [+0.00, +0.00] | +3.75 [+1.14, +7.35] | +0.00 [+0.00, +0.00] |
| qwen-base/flat-64/context-joint-conservative minus qwen-base/flat-64/question_only-joint-conservative | +7.92 [+3.70, +13.77] | -14.17 [-19.84, -8.77] | -8.75 [-15.74, -3.75] |
| qwen-base/tree_guided-64/context-joint-conservative minus qwen-base/tree_guided-64/conservative | +0.00 [+0.00, +0.00] | +9.17 [+6.16, +12.50] | +0.00 [+0.00, +0.00] |
| qwen-base/tree_guided-64/context-joint-conservative minus qwen-base/tree_guided-64/question_only-joint-conservative | +8.13 [+3.75, +14.06] | -7.92 [-13.60, -2.45] | -8.75 [-15.74, -3.75] |
| cross-encoder-base/flat-64/context-joint-conservative minus cross-encoder-base/flat-64/conservative | -0.42 [-1.25, +0.00] | +2.50 [+0.83, +4.17] | +0.42 [+0.00, +1.25] |
| cross-encoder-base/flat-64/context-joint-conservative minus cross-encoder-base/flat-64/question_only-joint-conservative | +7.71 [+2.92, +13.82] | -9.17 [-13.73, -5.30] | -7.92 [-14.17, -3.00] |
| cross-encoder-base/tree_guided-64/context-joint-conservative minus cross-encoder-base/tree_guided-64/conservative | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| cross-encoder-base/tree_guided-64/context-joint-conservative minus cross-encoder-base/tree_guided-64/question_only-joint-conservative | +8.13 [+3.75, +14.06] | -12.08 [-15.48, -8.82] | -8.33 [-14.35, -3.75] |
| e5-text/context-joint-conservative minus e5-text/conservative | -0.42 [-1.39, +0.00] | +9.58 [+6.67, +13.54] | +0.42 [+0.00, +1.39] |
| e5-text/context-joint-conservative minus e5-text/question_only-joint-conservative | +6.88 [+2.08, +12.73] | -3.33 [-7.92, +0.93] | -7.92 [-14.04, -2.92] |

## Execution recovery

Retrieval completed on the frozen source before a Transformers 5 API incompatibility interrupted verifier feature extraction. Repair `dfd52f62b3a6bf030c08a2fe18e09f68d96de5a6` replaces the removed `prepare_for_model` method with public pair tokenization (`only_second`). It passed 39 GPU-environment tests, including the pinned real tokenizer, exact question/context-prefix preservation and padding. Empty contexts use its native question-only encoding.

Only the feature stage was resumed, before inspecting new test metrics. Configuration, labels, classifier objective and threshold rules were unchanged. All three retrieval score files are byte-identical before and after recovery. Original run timings plus the resumed feature stage exclude repair waiting; the allocation-to-deletion cost includes it. See [recovery provenance](benchmarks/abstention-v3/experiment/feature-recovery.json), [validation](benchmarks/abstention-v3/validation.json), and the [tokenization API](https://huggingface.co/docs/transformers/main/en/pad_truncation).

## Reproducibility and cost

Inference source: `db5bb7362149d1511f8d67c4033d9292f30d3253`. All 10 original artifact hashes, corpus locks, exact traversal traces, ancestor access, unique charges and verifier input hashes were checked. Full local suite before inference: 129 passed, six optional skips. GPU setup tests and final CI are recorded with the published update.

Incremental estimated GPU and storage cost: **USD 1.24319**; cumulative **USD 7.94877 / 10**. The estimate counts allocation through verified deletion, including setup and artifact transfer, and is not a final provider invoice. The GPU and its temporary volume were deleted.

| Stage | Seconds | Actual neural pairs or forwards | Input tokens |
|---|---:|---:|---:|
| qwen-base | 2464.8 | 84755 | 21255588 |
| cross-encoder-base | 138.8 | 83101 | 14630634 |
| answerability_probe | 14.6 | 4546 | 1031891 |

Stage timings share caches and include different preparation work. They are not isolated serving latencies. The learned gate uses a ranked context prefix fitted to 512 tokens, which may omit lower-ranked evidence. No production retrieval default is changed.

Artifacts: [verified results](benchmarks/abstention-v3/verified-summary.json), [fitted probe coefficients and selection](benchmarks/abstention-v3/fitted-probes.json), [raw experiment](benchmarks/abstention-v3/experiment), [error audit](benchmarks/abstention-v3/failure-audit.json), [cost](benchmarks/abstention-v3/run-cost.json), [public hashes](benchmarks/abstention-v3/public-manifest.json).

Independent of IBM and the model and documentation authors; not an official or verified STAIR reproduction.
