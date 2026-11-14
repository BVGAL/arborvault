# Navigation v2: frozen comparison

This experiment compares Qwen3-4B, the pretrained MiniLM cross-encoder, and
[Tacit-4B](https://huggingface.co/morriszjm/Tacit-4B), an independent Apache-2.0
decision-model derivative of Qwen3-4B. It does not use Jev's hosted API or train
new weights. It is not an IBM implementation or a verified STAIR replication.

## What changes

All models run four policies at 64 and 128 charged question-candidate pairs:

1. `flat`: BM25 shortlist of leaves, then neural reranking.
2. `tree_scored`: the previous breadth-frontier policy, four expanded branches
   per level. Every scored leaf remains eligible.
3. `tree_budgeted`: incremental expansion, two branch evaluations and at most
   four passage evaluations per iteration. All alternatives remain available.
   The highest-scoring pending branch is expanded next. Branch evaluations use
   at most 3/8 of the total allowance (24/64 or 48/128); no unscored branch can
   be traversed for free. The cap can leave unused allowance when insufficient
   leaves are reachable, which is reported rather than silently filled.
4. `tree_guided`: the same incremental policy, with maximum descendant BM25
   relevance guiding unscored branch shortlisting and three query-matched
   descendant excerpts in each branch representation. This explicitly accesses
   the full leaf lexical index: it is **lexical-assisted hierarchical retrieval**,
   not pure navigation. No global leaf fallback is used.

All approaches already have a full-document BM25 index. Global lexical scoring
and the guided summaries are disclosed separately from neural pair accounting.
The guided variant changes both prioritization and branch representation; this
experiment cannot attribute its effect separately to those two ingredients.
Policy constants are fixed before inference, with no held-out parameter search.

The stopping rules and allocation are ordinary code. This tests decision models
as candidate/branch scorers, not a learned multi-action controller. Qwen and
Tacit use the **same native Tacit yes/no prompt**, next-token Yes/No logit margin,
non-thinking inference, NF4/double quantization and BF16 arithmetic. No reasoning
escalation or option rotations add uncharged evaluations. The two tokenizers
and the cross-encoder must all fit the same candidate text within 512 tokens;
only the excerpt may be shortened. The cross-encoder consumes query/candidate
pairs. Its parameter count and architecture differ, so equal pairs are not equal
FLOPs or latency. Prompt changes also mean new Qwen numbers cannot be interpreted
as a direct difference from the earlier yes/no experiment.

The Tacit adapter's native prompt was inspected at immutable revision
`af3fa1dc3cd98f937ab2ccedd4df8613fce5f1e5`. This repository implements that small
prompt format itself and loads standard model weights with `trust_remote_code=False`.
Remote adapter Python is not executed. Other model revisions remain pinned in
`configs/candidate-scoring-v1.json` and the referenced Qwen recipe.

## Data and separation

Four existing calibration manuals (96 cases: 40 answerable, 56 unknown) are reused.
Six new Python 3.13 manuals form the frozen test: io, pickle, inspect,
asyncio-eventloop, zipfile and tarfile. There are 48 answerable cases and 48
unknowns, including 24 within-test question transfers and 24 missing runtime or
configuration questions. These are 96 query-document cases and 72 distinct
question texts. Source HTML, license notices, evidence IDs and hashes are included.

These documents were not used in prior ArborVault experiments. Their 4,778–9,214
words and depth 5–6 make them hierarchical manuals, but they are **shorter than
the prior long-manual distribution**. Model pretraining exposure is unknown.
Annotations are assistant-authored and source-checked, not independently human
adjudicated. Single-block gold evidence may omit semantic alternatives. Negative
runtime questions are synthetic and relatively explicit. The document is known.
Results do not establish open-domain or corpus-wide retrieval performance.

Earlier failures informed the design; the earlier observed test is not reused as
a fresh holdout. Source/target exact overlap and document identity are checked;
this is not a semantic contamination audit. No inference or parameter selection
is performed on the new test before the source/config/data freeze.

## Measurement and acceptance

Report Recall@5 within a 512-word final context, candidate-evidence access,
branch/leaf evaluations, consumed tokens and unused allowance. Query-level
abstention uses maximum reached passage score, with conservative and balanced
thresholds fitted only on calibration documents. Raw logit margins are not
asserted to be calibrated probabilities. Report false rejection and unknown
rejection together with retrieval recall.

Include full-leaf BM25 and E5 as external references; their indexed retrieval is
not presented as equal neural pair cost. Replay every policy from recorded
scores, validate hashes and budgets, and bootstrap paired differences over five
source families (zipfile/tarfile combined), 5,000 draws. Six documents and 48
positives imply wide uncertainty; report null or negative outcomes as well.

Actual experiment compute/time is measured across all policies with score-cache
reuse. These times are not isolated per-policy serving latencies. No production
promotion is implied by one benchmark. GPU and storage must stay inside the
cumulative USD 10 cap, including the previous USD 5.79872 estimate. One GPU at
at most USD 1.20/hour, an independent three-hour allocation watchdog, and a
9,000-second inference deadline leave a cost reserve for setup and transfer.

```bash
python scripts/run_navigation_benchmark.py --output /workspace/navigation-v2
python scripts/analyze_navigation_benchmark.py --experiment /workspace/navigation-v2 --output results.json
```
