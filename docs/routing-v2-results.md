# Routing v2: measured Qwen3-4B experiment

Executed on 2026-11-13. This is a new document-scoped QASPER retrieval pilot,
not a repeat of the old eight-document benchmark and not a verified STAIR reproduction.
The full experiment and the added decoding ablation are reported, including failed promotion.

## Outcome

Neither adaptation passed its development promotion gate: **retain the base**. The beam
checkpoint recovers much of the greedy regression, but it does not demonstrate an advantage
over its matched base. Development also did not justify preferring the beam base to greedy.
Test measurements below do not retroactively choose the checkpoint or policy.

## Separate test

100 answerable and 20 unanswerable questions from 96 papers, disjoint from training and
development. Same paragraph units, known-document scope, k=5 and 512-word returned context.
Recall and coverage average over answerable questions; abstention is reported separately.
All failures remain in the denominator. These are retrieval metrics, not generated-answer
accuracy or official QASPER evidence F1. MiniLM is the vector retrieval component of a RAG
baseline; this experiment does not score a final answer generator.

| Method | Recall@5 | MRR | Full paragraph coverage | Empty answerable | Unknown abstentions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Greedy: Base | 0.7058 | 0.6192 | 0.6858 | 0/100 | 1/20 |
| Greedy: Base + lexical | 0.7558 | 0.6307 | 0.7358 | 0/100 | 0/20 |
| Greedy: Fine-tuned (step 175) | 0.4025 | 0.4350 | 0.4025 | 29/100 | 14/20 |
| Greedy: Fine-tuned (step 175) + lexical | 0.6892 | 0.5502 | 0.6708 | 0/100 | 0/20 |
| Four hypotheses: Base | 0.7375 | 0.6275 | 0.7125 | 0/100 | 0/20 |
| Four hypotheses: Base + lexical | 0.8025 | 0.6313 | 0.7775 | 0/100 | 0/20 |
| Four hypotheses: Fine-tuned (step 88) | 0.6983 | 0.6142 | 0.6883 | 2/100 | 7/20 |
| Four hypotheses: Fine-tuned (step 88) + lexical | 0.7633 | 0.6197 | 0.7233 | 0/100 | 0/20 |
| MiniLM vector retrieval | 0.7442 | 0.5172 | 0.7142 | 0/100 | 0/20 |
| TF-IDF lexical | 0.5883 | 0.3413 | 0.5650 | 0/100 | 0/20 |

The greedy and beam policies selected different trained checkpoints on development (steps
175 and 88).
A cross-policy adapted test comparison therefore combines checkpoint and decoding changes;
only the matched base/adapted pair within each policy isolates adaptation. The development
comparison at step 88 holds weights fixed when diagnosing the effect of broader search.

Greedy adapted-minus-base recall: **-0.3033**; document-clustered 95%
bootstrap interval **[-0.3904, -0.2157]**;
2 questions improve and 38 regress.
With four hypotheses: **-0.0392**, interval
**[-0.1117, +0.0333]**;
8 improve and 11 regress.
Intervals use 2,000 paired resamples by document and describe this filtered pilot, not broad
generalization or uncertainty across training seeds.

Base plus lexical minus MiniLM recall is **+0.0117**,
with interval **[-0.0872,
+0.1159]**.
A small numerical lead whose interval crosses zero is not established superiority to RAG.
The beam base plus lexical ablation reaches **0.8025**;
its difference from MiniLM is **+0.0583**, interval
**[-0.0337,
+0.1559]**.
It is the highest numerical test row, but it was not selected as the default by inspecting
this test. The development navigation criterion did not prefer the beam base over greedy.

## What changed and what the traces show

- Human QASPER questions replace procedural templates: 600 train, 120 development, 120 test,
  with document/content-hash and exact normalized question separation. Labels are upstream
  human annotations, not a new independent review. English NLP papers only.
- 1,400 train decisions include 673 positive decisions below the first routing level,
  wrong-subtree negatives, shuffled candidate positions and tokenizer-aware feature fitting.
- Heading-first descendant outlines expose later sections. Document-scoped routing skips
  an unnecessary root rejection. JSON-prefix constraints enforce valid unique candidate IDs.
- Base and adapted inference share NF4 weights, tokenizer, prompt and decoding policy within
  each comparison. All four adapters come from the same fixed one-epoch, two-seed recipe.

On development, the greedy base selected four paths in 168/264 calls; seed 42 at step 88
selected one in 239/264. This was observed before test inspection and motivated the registered
four-hypothesis decoding ablation. It uses up to four beams (`min(4, candidate_count + 1)`)
and combines score-ranked constrained alternatives, capped
at four distinct IDs; a highest-scoring empty hypothesis still abstains.

The training audit finds **966 of 1,173 positive decisions (82.35%) with a single target**,
plus 227 empty targets. A plausible mechanism is that exact-sequence supervision teaches a
narrow evidence path while exploratory retrieval benefits from retaining uncertain branches.
This is an interpretation supported by selection-width traces, not a causal proof. The beam
ablation measures how much changing search breadth can recover without changing weights.
Single-target labels are not inherently wrong: only 12/100 answerable development questions
require multiple paragraphs, and an oracle choosing one paragraph would reach Recall@5 0.9325.
The observed loss therefore also involves choosing the wrong path. Wider search protects
against routing uncertainty; it does not by itself establish stronger semantic ranking.

| Test policy | Mean paragraphs returned | First route keeps any gold ancestor | First route rejects answerable | Median seconds | p95 seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Greedy: base | 3.42 | 85/100 | 0/100 | 1.71 | 2.65 |
| Greedy: tuned | 0.80 | 60/100 | 13/100 | 0.70 | 1.27 |
| Four hypotheses: base | 3.63 | 91/100 | 0/100 | 4.12 | 6.96 |
| Four hypotheses: tuned | 3.31 | 85/100 | 2/100 | 3.29 | 5.53 |

Latency is one sequential GPU pass, without repeated warm/cold trials. Four generation beams
require more inference work. Returned-hypothesis token totals include EOS and shared prefixes;
they are not a hardware-compute count. Vector latency was not measured by this pilot runner.

## Development selection

Primary seed 42 selected step **175** under greedy decoding and step
**88** under the four-hypothesis policy. Seed 43 checks the same selected fraction
and cannot replace seed 42. Eligibility requires no selector failures, no increase in empty
answerable results and no loss of unknown abstention versus the matched base. Ranking uses
recall, coverage and MRR. If none is eligible, the best trained candidate is reported and
the base is retained. Aggregate training token loss is not the selection metric.

| Development variant | Recall@5 | Empty answerable | Unknown abstentions |
| --- | ---: | ---: | ---: |
| Greedy: base | 0.6092 | 2/100 | 0/20 |
| Greedy: seed-42-step-175 | 0.3208 | 23/100 | 15/20 |
| Greedy: seed-42-step-88 | 0.3158 | 5/100 | 5/20 |
| Greedy: seed-43-step-175 | 0.3525 | 12/100 | 9/20 |
| Greedy: seed-43-step-88 | 0.3558 | 6/100 | 4/20 |
| Four hypotheses: base | 0.6050 | 2/100 | 0/20 |
| Four hypotheses: seed-42-step-175 | 0.4642 | 24/100 | 14/20 |
| Four hypotheses: seed-42-step-88 | 0.5975 | 2/100 | 0/20 |
| Four hypotheses: seed-43-step-88 | 0.5658 | 5/100 | 4/20 |

The greedy protocol was frozen at `e342dc3`; the beam policy was publicly registered at
`3358c7c` while seed 43 was still training. Initial beam code `ae43ecb` exhausted GPU memory
after 92 base-development questions. The completed retry uses `d4ba75f`: attention batch rows
are evaluated sequentially with unchanged precision, masks, weights, context and routing policy.
All 15 runtime checks passed, including numerical parity, and the previously failing document
completed with a 12.40 GiB peak. The incomplete attempt is preserved and disclosed in the
[incident record](benchmarks/routing-v2/beams/previous-attempt.json) and
[memory preflight](benchmarks/routing-v2/beams/memory-preflight.json).
The greedy test ran automatically, but its scores
were kept unread until the separate beam development selection was locked. Both policies
are published regardless of their test outcomes.

## Cost, artifacts and reproduction

One temporary RTX 4090 at $0.89/hour, a three-hour independent stop deadline and no shared
production volumes. Allocation-to-deletion time: **107.3 minutes**.
Estimated GPU plus storage cost: **$1.61**, below the authorized
$10. This is a runtime/rate estimate, not a final invoice. Pod deletion was verified; no
experiment GPU or volume was left allocated. No Hugging Face upload was performed.

- [Protocol and commands](routing-v2-experiment.md), [dataset card](../datasets/qasper-routing-v2/README.md).
- [Comparison and uncertainty](benchmarks/routing-v2/comparison.json),
  [dataset routing audit](benchmarks/routing-v2/dataset-routing-audit.json).
- [Training provenance](benchmarks/routing-v2/training-provenance.json),
  [greedy selection](benchmarks/routing-v2/selection-before-test.json),
  [beam selection](benchmarks/routing-v2/beams/selection-before-test.json).
- [Base test](benchmarks/routing-v2/base-test.json),
  [adapted test](benchmarks/routing-v2/tuned-test.json),
  [beam base](benchmarks/routing-v2/beams/base-test.json),
  [beam adapted](benchmarks/routing-v2/beams/tuned-test.json),
  [vector/lexical test](benchmarks/routing-v2/baselines-test.json).
- [Published-file checksums](benchmarks/routing-v2/public-manifest.json),
  [runtime](benchmarks/routing-v2/runtime-environment.json),
  [adapter checks](benchmarks/routing-v2/adapter-checks.json),
  [cost record](benchmarks/routing-v2/run-cost.json).

Summarize downloaded full artifact directories with:

```sh
python scripts/summarize_routing_v2.py /path/to/experiment-v2 --beams /path/to/beam-v2 --output comparison.json
```

The v2 engine is a reusable Python experiment component. Existing FastAPI endpoints retain
their established behavior and authorization layer; callers must supply an authorized document
tree. The pilot does not validate global document discovery, Italian, OCR, multi-tenant scale
or stronger dense/hybrid/reranked baselines. Public data may occur in model pretraining.

## Next experiment, not a claimed improvement

Use this test only for reporting; choose future settings on development and reserve a fresh
test for the next iteration. Separate exploratory branch ranking from final evidence selection.
Train per-candidate relevance/ranking or a set-based objective, so alternative useful routes
are not penalized merely for differing from one serialized target. Balance genuinely
multi-branch examples, review difficult negatives and include recovery frontiers produced by
the base on training documents. In particular, train on mixed frontiers containing several
competing branches, rather than only oracle descendants plus isolated empty subtrees. The
current oracle-path exporter does not reproduce every frontier reached during exploration.
Calibrate breadth and abstention on development, compare
against the unchanged base and vector baseline, and retain the two-seed promotion guard.
