# Routing v2: proposed data and training protocol

Status: **proposed, not trained or measured**. This plan follows the
[v1 error audit](qwen3-error-analysis.md). Numerical targets below are engineering acceptance
criteria to freeze before running, not measured improvements or statistical guarantees.

## 1. Separate diagnosis, development and final testing

Keep `public-docs-v1` and `routing-bootstrap-v1` unchanged. Do not turn the six observed test
regressions into training targets and then advertise the old test as independent.

Create a new licensed public corpus with more document structures and domains. Split by
source document/version and semantic question family **before** writing paraphrases or
mining negatives. Near duplicates and translations stay together. Use three roles:

- Training: source-grounded routing examples, with evidence spans and ancestor paths.
- Development: disjoint documents/families, used for checkpoint and parameter selection.
- Final test: independently reviewed, frozen and unused until the protocol is fixed.

Report a known-corpus slice and an unseen-document/domain slice separately. A reusable
router should not depend on memorizing one tenant's corpus. Corpus-specific parametric
indexing, closer to STAIR, is a separate experiment with a different update/privacy model.

## 2. Build examples that match actual navigation

Start with a modest pilot of **500–1,000 distinct, evidence-grounded questions** across at
least 20 public documents, rather than multiplying a few templates. Convert these questions
into several thousand route decisions only after checking source and label quality. Reserve
at least 100 distinct questions each for development and final testing; expand these for
credible claims beyond a pilot.

Each decision record should retain: source revision/license, question family, query language,
tree depth, prompt/feature version, candidate source paths and order, local-ID mapping,
positive IDs, evidence spans, and review status. Never publish production tenant data.

Required coverage:

- Every level from document roots to evidence sections; at least half of positive route
  examples below the document root. Cover all target depths present in the pilot corpus.
- Confusable sibling headings, relevant topics in late branches, long documents and generic
  parents whose own text does not contain the answer.
- Candidate sets of varied sizes, including larger than eight, shuffled positions and one
  through four relevant IDs. Measure token lengths using the actual Qwen tokenizer.
- Semantically difficult negatives mined by lexical and dense similarity **within the
  training split**, reviewed to avoid treating unlabeled-but-relevant passages as negatives.
- Paired candidate-present/candidate-absent situations, wrong-branch recovery, and truly
  unknown questions. Distinguish “this subtree lacks evidence” from global unanswerability.
- English and Italian paraphrases of real evidence, with translations grouped in one split.
  Reduce procedural format fixtures to a small fraction, initially no more than 10%.

Use a stronger model only to propose questions/labels if needed. Source-span validation,
independent review and ambiguity checks determine acceptance; teacher output is not truth.
Validate counts, depth coverage and normalized/semantic duplicates before renting a GPU.

## 3. Expose the tree and retain diagnostic traces

Version a new routing prompt with explicit distinction between exploring a plausible branch
and returning evidence. Compare a compact, heading-first subtree outline with the current
240-character prefix. Keep bounded token budgets and measure pruning coverage. Train and
serve exactly the same candidate features and serialization.

For small corpora, also evaluate one-pass selection from the complete section outline;
for large corpora, use hierarchical selection with multiple branches and bounded recovery.
This tests whether repeated ancestor decisions, rather than model capacity, are the bottleneck.

The next evaluation must record, using stable public source IDs:

- Candidate paths, chosen IDs and retained gold-ancestor coverage at each depth.
- First depth where a relevant path is lost; empty selections and valid-but-wrong selections.
- Specific validation error category, separate from transport and context errors.
- Sections reached before ranking, rank of each relevant section and final word-budget loss.
- Whether an oracle-fed frontier changes the choice, separating local ranking from cascading
  earlier mistakes. Do not use the oracle at deployment or as a reported model baseline.

Constrain output to a JSON array with allowed IDs, unique items and the beam limit, or score
valid candidates directly. Verify that the chosen implementation enforces all constraints;
generic JSON mode alone is insufficient. Keep the existing rejection path as a final check.

## 4. Train conservatively and select by retrieval

Start again from the exact original base revision, not from the regressed v1 adapter. Keep
NF4/BF16, the shared non-thinking template, completion-only loss, q/v LoRA rank 8 and the
same inference budget fixed initially. Changing rank, target modules and dataset at once
would obscure which change helped.

First compare the improved representation **without fine-tuning** to its matching tuned
variant. Use a small preregistered learning-rate comparison, such as 2e-5 and 5e-5, against
the existing 1e-4 reference only if the runtime budget permits. Begin with at most one pass
over the new data, evaluate intermediate checkpoints, and allow a second pass only if
development retrieval is still improving. These values are hypotheses, not optimal settings.

Keep epoch zero as a selectable checkpoint. Save intermediate adapters instead of overwriting
the best token-loss checkpoint. Use development pure-navigation Recall@5 and evidence coverage
for selection, with MRR as a tie-breaker, while requiring:

- No observed invalid selections on the development set.
- No higher answerable empty-result rate than the matched base.
- No worse unknown-question abstention than the matched base.

If none qualifies, retain the base and report a failed adaptation. Log NLL by data source,
depth, language and positive/empty target, but do not use aggregate NLL as the quality metric.
Retain the best two development candidates for a stability check on another training seed
before touching the final test. A few-question gain should not be called statistically robust.

## 5. Isolate effects and keep the vector baseline

| Comparison | Purpose |
| --- | --- |
| Original base, old features | Historical anchor; frozen v1 measurement |
| Original base, new structural features | Information/representation effect |
| New data adapter, same new features | Incremental effect of training |
| Above with constrained selection | Format failure effect, separate from semantic recall |
| Above with lexical fallback | Recovery and false-positive tradeoff |
| Same-corpus MiniLM sections | Dense retrieval reference |
| MiniLM candidates plus structural reranking | Separately labeled hybrid product option |

Run every comparable variant with identical source sections, labels, top-k, returned-context
budget and base runtime. The hybrid is useful if immediate quality is the priority, but it
does not establish vector-free retrieval. Do not silently replace the primary pure-navigation
endpoint with a dense fallback and attribute the gain to the fine-tuned model.

First aim to exceed the matched base without increasing false rejections. Then measure the
gap to MiniLM on the **new** test; its old 91.67% is not a universal target transferable to
another corpus. Report paired per-query changes and uncertainty, with resampling by semantic
family/document rather than treating near-duplicate questions as independent observations.

Before execution, freeze data hashes, the experiment matrix, acceptance rules and the cloud
cost ceiling after a runtime estimate. No new paid run is part of this diagnostic update.
