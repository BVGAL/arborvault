# Long manuals v1: protocol frozen before inference

Date: 2026-11-14. Configuration: [`long-manuals-v1.json`](../configs/long-manuals-v1.json).
Dataset and qualification: [dataset card](../datasets/long-manuals-v1/README.md).

Question: do the previous candidate scorers transfer to longer, explicitly
hierarchical technical manuals, and can they abstain on more unsupported
queries? This is an inference-only, known-document comparison. No new training,
test-based checkpoint selection or production promotion is performed.

**Pre-results implementation correction:** a partial Qwen inference attempt was
stopped before inspecting score caches or metrics because the new parser used
generic leaf titles. Existing tree outlines expect paragraph previews. Leaf
titles were changed to the first sentence up to 110 characters, exactly as in
the QASPER parser; all systems rerun from scratch. Question text, evidence IDs,
partitions, weights, policy and thresholds rules remained fixed. Both source
locks remain in Git history and the discarded attempt's time is included in
cost. The corrected representation is the operative freeze.

## Fixed systems

- Qwen3-4B base, the existing pinned revision, non-thinking NF4/BF16; yes/no
  next-token log-odds score each candidate.
- MS MARCO MiniLM-L6 cross-encoder base, as a control.
- The same cross-encoder adapted on QASPER in candidate-scoring-v1, **step 100**,
  selected in that earlier experiment. Its exact four file hashes are fixed in
  the config and checked against the previously published artifact manifest.
  This is not the best-performing fold from the retrospective cross-validation.
- E5-base-v2, pinned revision and native query/passage prefixes; normalized mean
  pooling and cosine similarity over every leaf in the supplied document.
- BM25 is an additional inexpensive control using the same document blocks.

Qwen and both cross-encoders use identical question/candidate JSON, including
document title, ancestor path, branch/passage kind and excerpt. The excerpt is
shortened identically to fit both tokenizers within 512 tokens. E5 follows the
previous text-plus-path representation with its native 512-token truncation.
This is a system comparison; representations and total compute are not identical
between dense indexing and reranking.

## Fixed policies and budgets

The primary reranking policy is BM25 prefilter of 64 leaves followed by the
model scorer. The structural ablation uses the **same scorer**, a width-four
branch beam and a total budget of 64 scored candidates. Branches consume budget.
An oversized frontier is BM25-prefiltered to the remaining budget. All scored
leaves are retained for final ranking, including those outside the branch beam.
Both policies are also reported at 128 pairs as a prespecified secondary budget.
There is no lexical fallback filling missing tree results.

Inference scores only requested candidates, using a per-question cache shared
across policies and budgets. Therefore the measured experiment cost is the
union of visited pairs; it is not the cost of one independent policy call.
Per-policy charged pairs, visited nodes and expansions are saved for replay.
Final context is at most five leaves and 512 words, with the last included leaf
possibly truncated. Recall counts matching node IDs; full-evidence coverage also
requires that the complete annotated block fit in the context.

## Abstention without test tuning

For each fixed system/policy, confidence is the maximum score among reached
passages (a finite sentinel for no reached passages). A query gate accepts the
entire retrieved context or rejects it. It never fills an abstention by fallback.
Thresholds are selected using **only the four calibration documents**:

1. Conservative: maximize unknown rejection subject to at most two percentage
   points lost recall and additional false abstention on calibration.
2. Balanced: maximize balanced answerability accuracy, without a recall guard.

All distinct calibration score boundaries plus always-keep/reject-all are
considered; ties prefer recall, fewer false abstentions and lower thresholds.
With only 40 answerable calibration questions, the conservative false-abstention
guard permits no additional rejected answerable query. These are empirical
calibration rules, not guaranteed test bounds. Raw maximum was chosen before
this test because the previous CV did not support the learned five-feature gate.

## Reporting and uncertainty

Report coverage-first Recall@5, MRR, nDCG and full-evidence coverage; conservative
and balanced gates report unknown rejection, false abstention, retained recall,
coverage and conditional quality. Stratify unknowns by construction category.
No answer correctness or end-user usefulness claim follows from retrieval alone.

The fixed primary comparison is tuned cross-encoder flat-64 against Qwen
flat-64 and E5. Also report tuned-minus-base and tree-minus-flat at both budgets.
Use 5,000 paired cluster-bootstrap draws, seed 42, grouping by document except
that the logging reference and cookbook share one family. Eleven test families
and synthetic single-evidence questions limit inference. Intervals are
descriptive, conditional on this corpus and fixed weights; no multiplicity or
training-seed uncertainty adjustment. Transferred question topics couple source
and destination documents, so even family bootstrap intervals may understate
dependence; do not interpret narrow intervals as a population-level guarantee.

Save all score caches, traversals, calibration choices, file hashes, environment,
usage and a separately replayed summary. Labels never enter the model input or
navigation callback. The test is no longer untouched after this run; future
model selection on these outcomes must treat it as development data.

## Cost and reproduction

Prior cumulative estimated cost: **$4.99259 of $10**. One GPU, at most $1.20/hour,
two-hour inference deadline and independent three-hour allocation stop. At that
ceiling, compute plus 80 GB temporary storage is about $3.64, leaving cumulative
estimated spend below $8.64. Verify the actual quote before allocation and delete
the pod after verified artifact download. Estimates include allocation, setup,
waiting and transfer, and are not a final provider invoice.

Install `requirements.training.txt` in the documented GPU environment, obtain
the earlier local step-100 checkpoint matching the config hashes, then run:

```bash
python scripts/run_long_benchmark.py --checkpoint /path/to/cross-encoder-step-100 --output /path/to/long-v1
python scripts/analyze_long_benchmark.py --experiment /path/to/long-v1 --output /path/to/verified-summary.json
```

Checkpoint binaries are not bundled in Git; the published score caches permit
offline replay without them. The prior training runner and checkpoint hashes
document provenance, but exact GPU retraining reproducibility is not promised.
