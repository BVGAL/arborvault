# Why the first Qwen3 adapter regressed

Post-hoc diagnosis of the frozen 2026-11-13 experiment, using the original reports and
training bundle. No new GPU run, prompt change, retraining or relabeling was performed.
The [machine-readable audit](benchmarks/qwen3-error-analysis.json) contains all 36 answerable
queries, source hashes, dataset distributions and a label-informed routing diagnostic.

The strongest evidence points to **insufficient training coverage of hierarchical routing,
combined with premature rejection**. Overfitting to the small procedural task is a plausible
explanation, not an experimentally isolated cause. A different learning rate or larger LoRA
rank has not been shown to solve it.

## What actually changed

| Pure navigation outcome | Matched base | Adapter |
| --- | ---: | ---: |
| Recall@5, 36 answerable queries | 61.11% | 47.22% |
| All labeled sections retrieved | 21 | 16 |
| Some labeled sections retrieved | 2 | 2 |
| No returned evidence on answerable queries | 4 | 12 |
| Rejection on the very first routing call | 3 | 9 |
| Correct rejection of unknown queries | 8/8 | 8/8 |

The adapter loses six cases the base solved and gains one. Three losses are first-call
rejections, two return nothing after a second call, and one returns the right document but
misses its required section. All six losses are found by MiniLM. English recall falls from
60% to 50%; Italian falls from 66.67% to 33.33%, but the Italian subset has only six queries.

| Query | Observed regression | Missing section |
| --- | --- | --- |
| `fastapi-path-03`: why `/users/me` must precede `/users/{user_id}` | Rejects at root | Order matters |
| `httpx-timeouts-05`: timeout per chunk or whole response | Rejects at root | Fine tuning the configuration |
| `httpx-clients-02`: close a client without a `with` block | Rejects at root | Usage |
| `fastapi-body-04`: where JSON schemas appear | Empty after two calls | Automatic docs |
| `httpx-quickstart-06`: redirects and redirect history | Empty after two calls | Redirection and History |
| `httpx-clients-05`: common URL prefix | Returns other client sections | Other Client-only configuration options |

These are post-hoc diagnostic examples, **not new training examples**. Original reports did
not retain every candidate set, selection or raw invalid response. Except for first-call
rejection, the exact wrong branch cannot be reconstructed confidently from final results.

## The training set did not teach the deployed task

- **384/392 training rows (97.96%) are procedural**, using six topics, six candidates and
  short direct evidence. Removing random entity IDs leaves only **19 query templates**.
- The eight real training rows are six positive questions about **one HTTPX limits
  document**, plus two unknown questions. **All eight are depth zero**. There is no real
  training example of selecting a subsection or recovering after an incorrect branch.
- Validation has 64 procedural rows and 20 real routes from one other question family
  plus unknowns. All **17 procedural validation query templates also occur in training**.
  Exact strings and groups are disjoint, but this is not independent semantic coverage.
- Training has no three/four-ID targets, only six/eight-candidate inputs and at most 710
  tokens. The measured evaluation reaches 2,639 input tokens. This is a distribution shift,
  not evidence that the inference context was truncated.
- Empty targets are 50/392 (12.76%). The audit does **not** establish an excessive negative
  fraction as the cause. The more consequential distinction missing from the examples is
  “a branch plausibly leads to evidence” versus “this short excerpt directly answers.”

The training NLL falls from 0.225365 to 0.001985 to 0.000046. Validation NLL reaches its
minimum at epoch two, then rises. This supports concern about fitting repeated templates;
it does not show which epoch had the best retrieval. Only the selected adapter was retained,
and checkpoints were selected by token loss rather than independent retrieval quality.

## Routing hides useful structure

`routing_summaries()` assembles at most 1,200 characters, starting with the node title and
body prefix. `routing_messages()` then sends only the first **240 characters per candidate**.
Later child headings can disappear behind introductory text. Only **6 of 38 gold section
titles** occur literally in their root's transmitted excerpt; all six regressed section
titles are absent. Title absence does not prove semantic insufficiency, but it identifies
a concrete information bottleneck. An excerpt that says “Path Parameters” may never expose
“Order matters,” although that section exists below it.

MiniLM embeds each section's path and content and searches sections directly. It can reach
a relevant section without first accepting every ancestor. Its tokenizer also has a length
cap, so it should not be described as seeing every word without truncation.

An **oracle diagnostic** selects labeled ancestors and leaves the real navigation ranking,
k=5 and 512-word budget unchanged. It obtains **100% recall and evidence coverage** on this
small test. Traversal runs in memory; all candidate frontiers fit the normal limits. This
shows that the stored evidence and downstream budget permit recovery when routes are right.
It is not an LLM result, a deployable baseline, or a promise of achievable general performance.

## Format errors do not explain this regression

Each model rejects one unique query's invalid selection, repeated in navigation and auto:
base `fastapi-errors-04`, adapter `httpx-clients-04`. Both retain previously collected
evidence. The base still retrieves its gold section; the adapter misses a case the base
already missed. **None of the six newly lost cases has a selector failure.**

Both outputs terminate correctly and fail selection validation. The logs do not preserve
enough information to distinguish too many IDs, unknown IDs, wrong element types or
duplicates. Constraints can remove those format failures; they cannot explain or repair
the six observed semantic regressions on their own.

With lexical fallback, adapter recall rises to 69.44% versus base 66.67%, but MRR falls
from 0.6134 to 0.5856. Fallback is invoked for 23 rather than 20 queries. This is not evidence
that pure navigation improved. MiniLM remains at 91.67% recall.

## What to change next

The [v2 experiment plan](routing-v2-plan.md) prioritizes realistic full-depth routes, better
structure visibility and checkpoint selection by retrieval. It specifies new data splits,
instrumentation, controlled experiments and criteria for retaining the base when tuning
does not help. The v1 test remains a historical regression set; changes informed by this
analysis require a new, frozen test before generalization claims.

This also differs from [STAIR sections 4.2 and 5.2](https://arxiv.org/html/2609.03874v2):
STAIR learns corpus-linked question-to-section mappings, provides a complete table of
contents, uses constrained generation and selects checkpoints by development Recall@1.
Our first adapter learned a mostly procedural local-ID selector. The paper's results are
not a prediction for this experiment or a verified reproduction by ArborVault.

[TRL's SFT documentation](https://huggingface.co/docs/trl/sft_trainer) defines the objective
as token-level likelihood and gives approximately 1e-4 as a common adapter learning rate.
Thus, token-loss improvement is not a retrieval guarantee, and the current 1e-4 is not
inherently a bug. Lower rates are a controlled experiment after repairing the data.

## Reproduce the audit

From a repository checkout with its ordinary evaluation dependencies installed:

```sh
python scripts/analyze_qwen3.py --output reports/qwen3-error-analysis.json
pytest -q tests/test_error_analysis.py
```

The analysis reads immutable v1 data and reports. It does not call a model, access cloud
services, write training rows or modify the published measurements.
