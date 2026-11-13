# Routing v2: checkpoint diagnostics and evidence audit

The adapted model loses evidence **even when given the correct navigation state**.
This rules out an explanation based solely on mistakes accumulated during an
autonomous trajectory. The adapter learns a narrower, more precise selection policy:
step 88 improves conditional token prediction and precision, but substantially lowers
recall. Step 175 becomes more reluctant to select anything. No new fine-tuning was
performed and the base remains the deployment recommendation.

These are **post-hoc development diagnostics**, not a new held-out benchmark or a
causal proof that one training mechanism explains the whole regression. The
[original test results](routing-v2-results.md) and scores are unchanged.

## Controlled comparison

Original Qwen3-4B revision, NF4 double quantization/BF16 compute, non-thinking template,
context 4096, greedy constrained JSON, width at most four, and original seed-42 LoRA
weights at steps 88/175. No optimizer, parameter update, retuning or test inspection
was introduced. The 120 development questions cover 102 documents: 100 answerable
and 20 unanswerable. They generate 244 canonical oracle states: 224 positive and
20 initial empty-target states. Every variant receives exactly the same states.

Each state is evaluated with correct output prefixes and with a fresh full decision
that sees no target. Oracle states follow the first annotation; candidates remain
in runtime order. Alternative annotations are retained, and union-based precision
is also included in the machine-readable results. Training used shuffled order.

| Model | Dev token NLL ↓ | Conditional choice accuracy | Oracle-state precision | Oracle-state recall | Oracle-state F1 | Mean selected | Empty positive states |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 3.4095 | 61.85% | 49.18% | 85.64% | 57.47% | 2.70 | 2/224 |
| LoRA step 88 | 0.4404 | 77.00% | 62.87% | 60.49% | 60.66% | 1.07 | 7/224 |
| LoRA step 175 | 0.4613 | 76.45% | 58.18% | 56.44% | 56.42% | 0.94 | 33/224 |

Token NLL is held-out completion-token-weighted cross-entropy, including EOS;
it is not the historical cumulative training loss. Conditional choice accuracy
excludes forced tokens and the final EOS decision (Qwen accepts equivalent EOS IDs).
It still scores an ordered canonical serialization, **not semantic set accuracy**.
Free-decision precision/recall/F1 are macro averages over the 224 positive states;
an empty prediction on a positive state has precision and recall zero.

Step 88's higher local F1 is only a numerical increase: the query-balanced F1
bootstrap interval includes zero. Its precision increase and recall decrease have
intervals that exclude zero. Raw token accuracy, exact-set match, alternative-aware
precision and per-depth metrics are in the full report.

## Local decisions versus autonomous retrieval

| Model | Initial oracle recall | Deeper oracle recall | Autonomous Recall@5 | Empty answerable | Unknown abstentions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 78.00% | 91.80% | 60.25% | 2/100 | 0/20 |
| LoRA step 88 | 50.00% | 68.95% | 31.58% | 5/100 | 5/20 |
| LoRA step 175 | 47.00% | 64.05% | 32.08% | 23/100 | 15/20 |

The first decision is especially informative: it occurs before any earlier routing
mistake. Its recall falls from 78% to 50% and 47%. Deeper oracle states also lose
recall despite being placed on the annotated path. Missing exposure to erroneous
states can still contribute, but cannot explain the entire deficit.

The autonomous percentages are paragraph retrieval over 100 answerable questions,
k=5 and a 512-word context. Oracle percentages measure local candidate selection,
with a different denominator and annotation scope. **Do not subtract these columns
to estimate exposure bias.** The replay also records lexical fallback, but it is
not part of this pure-navigation diagnosis.

| Model | All annotated alternatives lost at first decision | Lost later | Never entirely lost |
| --- | ---: | ---: | ---: |
| Base | 18 | 18 | 64 |
| LoRA step 88 | 46 | 19 | 35 |
| LoRA step 175 | 49 | 15 | 36 |

This trace analysis retains evidence already collected and respects alternative
annotations. Pruning is irreversible in the current navigator, which has no return
action. The counts locate failures; they do not prove why a wrong choice occurred.

For uncertainty, average each question's positive oracle decisions first and then
resample its document (2,000 paired bootstrap samples, fixed seed; 87 documents
containing answerable questions). This avoids giving deeper questions more weight.

- LoRA step 88 minus base, query-balanced oracle recall: **-24.89 points**, document-bootstrap 95% interval **[-30.08, -20.10]**.
- LoRA step 175 minus base, query-balanced oracle recall: **-29.38 points**, document-bootstrap 95% interval **[-35.14, -23.88]**.

The intervals condition on this dataset and primary training seed. They do not
include training-seed variance, new-domain uncertainty or uncertainty in the labels.

## What changed between checkpoints

Held-out token NLL changes from 0.4404 at step 88 to 0.4613 at step 175, while the
historical cumulative training NLL decreased. Positive-state empty selections rise
from 7 to 33; mean breadth falls from 1.07 to 0.94. This is consistent with later
overfitting and/or increasingly conservative decisions. Two checkpoints cannot
separate these mechanisms or identify an optimal training step.

Under correct output prefixes, premature list closure is observed at 3 token
positions for base, 25 for step 88 and 53 for step 175. This diagnostic counts a
locally preferred legal token that would close a shorter array; it is not the count
of failed autonomous queries. It complements the observed breadth/empty-selection
change rather than serving as a causal proof.

The current objective rewards a specific short action string. The desired retrieval
objective values keeping useful routes available within a final evidence budget.
Improving the former does not guarantee improving the latter. Other plausible
contributors remain incomplete evidence labels, candidate ordering, excerpt limits,
uncertainty calibration and mismatched mixed-branch frontiers.

## Audit of 50 failures and five controls

The frozen sample contains 26 regressions, 21 shared failures and three partial
improvements, with one question per document. Five all-model successes were added
as controls. The reviewer read 273 full annotated/pooled retrieved paragraphs with
model identities masked, and inspected extra surrounding source text in ambiguous
cases. All 55 source mappings match the original QASPER development file exactly.

**This is an AI-assisted qualitative audit, not independent human annotation.** The
reviewer knew aggregate results; masking does not establish complete blindness.
Confidence labels are reviewer judgments, not calibrated probabilities. Full
paragraph review is separated from checks on the actual returned 512-word context.

| Assessment of gold evidence in the 50 failure cases | Cases |
| --- | ---: |
| Appears sufficient for the question | 29 |
| Question/annotation scope ambiguous | 10 |
| Possibly redundant required paragraphs | 5 |
| Possibly incomplete or mismatched evidence | 6 |

These categories overlap with the separate alternative-evidence flag; 29 sufficient
gold sets do **not** imply 29 unquestionably valid retrieval penalties.

- Twelve failure cases have proposed alternative evidence sets, eight with high
  reviewer confidence. Some require multiple passages together.
- In six cases, a high-confidence alternative set is fully present in the actual
  context of at least one variant with incomplete annotated recall: **A14, A32,
  A37, A41, A43, A44**. This is conservative full-paragraph visibility, not proof
  that every case is a human-confirmed false negative.
- Examples: A14 retrieves the method for adding negation to LAMA; A32 retrieves
  the bilingual-transfer procedure; A43 retrieves a complete language list that
  answers the Arabic-membership question. These facts occur outside the selected
  annotated paragraphs.
- A19 already retrieves an annotated paragraph naming all three NER models, while
  the gold sets require additional model descriptions. This is redundancy rather
  than an unannotated alternative.
- A23 and A53 expose missing formula representations; A39 exposes upstream layout
  noise. Exact source mapping does not guarantee a complete, clean representation.

All original gold sets and scores remain unchanged. This failure-enriched sample
cannot estimate corpus-wide annotation error or the fraction of the overall
fine-tuning regression explained by labels. Human adjudication is required before
using the proposed alternatives as new supervised labels.

See [all 55 judgments, source IDs and per-model context checks](benchmarks/routing-v2-diagnostics/audit-cases.json)
and the [frozen selection](benchmarks/routing-v2-diagnostics/audit-selection.json).

## Execution, costs and reproducibility

The original RTX4090 had no available capacity. Runpod allocated one A40 at
$0.59/hour. Initial decisions compared with the old RTX4090 run differed in 4/120
base cases, 0/120 step-88 cases and 1/120 step-175 cases. Therefore all three fixed
autonomous policies were replayed on the A40, using the same quantized base tensors
within each process. Initial-state agreement with that replay: **Base 120/120, LoRA step 88 120/120, LoRA step 175 120/120**.
Hardware/library execution can affect borderline choices; historical reports were
preserved and the audit sample still refers to the original reports.

Oracle diagnostics took 11.75 minutes and peaked
at 6.43 GiB of allocated GPU memory. Oracle runner source:
`e6c14e74904d3572b5e936687b0e45590b02b0c4`. Replay source: `00328f48d7a641a3b475e69d7698dbcbb853cee9`.
Both use the pinned training environment. Postprocessing additionally excludes EOS
from the conditional choice metric; raw inference outputs are unchanged.

Estimated incremental GPU plus storage cost: **$0.26**;
cumulative with the preceding v2 allocation: **$1.87 / $10**.
This is allocation-to-deletion runtime/rate accounting, not a final provider invoice.
The pod was deleted and absence verified after downloading/checking all artifacts.
No new adapter or Hugging Face model was published.

The application suite passed 85 tests (five optional Torch checks skipped in the
API image); 13 GPU-runtime checks passed, including teacher-forced loss/grammar
validation. The diagnostic-only postprocessing tests were also run after changes.

Reproduce inference with the original adapters:

```sh
python scripts/diagnose_routing_v2.py --adapters /path/to/adapters --output /path/to/oracle
python scripts/replay_routing_v2.py --adapters /path/to/adapters --output /path/to/navigation
python scripts/prepare_routing_audit.py --original-dev /path/to/qasper-dev-v0.3.json --output /path/to/audit
python scripts/summarize_routing_diagnostics.py --diagnostics /path/to/oracle --replay /path/to/navigation --audit /path/to/audit --judgments docs/benchmarks/routing-v2-diagnostics/audit-judgments.json --output /path/to/comparison.json
```

Artifact links: [comparison](benchmarks/routing-v2-diagnostics/comparison.json),
[oracle manifest](benchmarks/routing-v2-diagnostics/oracle/manifest.json),
[navigation manifest](benchmarks/routing-v2-diagnostics/navigation/manifest.json),
[cost/cleanup](benchmarks/routing-v2-diagnostics/run-cost.json),
[public checksums](benchmarks/routing-v2-diagnostics/public-manifest.json),
[protocol](routing-v2-diagnostic-protocol.md).

## Implication for the next experiment

First adjudicate the evidence issues and compare candidate scoring with a small
cross-encoder under the same context budget, with breadth and abstention calibrated
on development. Add more frequent checkpoints and monitor local recall alongside
token loss. Then test recovery actions and on-policy relabeling as a separate
ablation. These diagnostics do not justify starting another fine-tuning run yet,
nor demonstrate a general advantage over vector retrieval.
