# Routing v2 post-hoc diagnostic protocol

This is an inference-only investigation of the existing seed-42 Qwen3-4B base,
step-88 and step-175 adapters. No new fine-tuning, deployment promotion or fresh
test claim is part of this work. The already-inspected v2 development set is used.
The original prompt, precision, input fitting, candidate order and greedy grammar
are unchanged. All three variants share one quantized base in one process.

## Three conditions

1. **Correct output prefixes:** completion-only full-vocabulary token NLL and
   conditional token accuracy, plus grammar-constrained accuracy at positions
   with more than one legal continuation. Forced punctuation positions must not
   inflate choice accuracy. The canonical ordered target uses the first evidence
   annotation, matching training; these are not semantic set metrics.
2. **Free decisions on oracle states:** traverse ancestors of the first annotated
   evidence set, retain every candidate at each reached frontier, and generate a
   fresh decision without supplying target tokens. Record selected-set precision,
   recall, F1, exact match, width and empty selections. Preserve the union of all
   annotations as secondary metadata. Candidate order matches inference, not the
   shuffled training exporter. Report initial and deeper decisions separately.
3. **Autonomous navigation:** reuse the complete, hashed v2 greedy development
   traces for the same three variants. Analyze the first irreversible evidence
   pruning, mixed/no-evidence frontiers and context-budget loss. Reuse avoids
   changing the original observation and duplicate inference costs.

Oracle-state counts are decision-weighted; retrieval is question-weighted.
Their percentage difference is not a causal estimate of exposure bias. Compare
variants on identical states and aggregate paired query changes by document.
Known annotation incompleteness, canonical output order, prompt information loss,
decoding, local errors and distribution shift remain competing explanations.
Unknown questions contribute one initial empty-target state, reported separately.

## Evidence audit

Freeze a deterministic, stratified sample of 50 answerable development questions
with Recall@5 below one for at least one of the three greedy variants. Strata:
regression relative to base; partial improvement; shared failure. Sample by a fixed
SHA256 order within strata, keeping one question per document where feasible.
Add five all-model successes as controls. Freeze IDs before reading passages.

The review packet masks model identities and pools retrieved passages in shuffled
order. It includes the question, annotated evidence, full retrieved paragraph
texts and source paths. Inspect surrounding original source text when ambiguous.
Check mapping against the original QASPER dev file. Classify insufficient labels,
valid unannotated alternatives, true retrieval failures and unresolved cases;
record the exact supporting passage IDs and a rationale per case.

This is an **AI-assisted qualitative review**, not independent human annotation.
The reviewer knows aggregate experiment results, and masking cannot guarantee
complete blindness. Do not silently replace gold labels or change official scores.
Reviewing failure-enriched cases cannot estimate corpus-wide annotation error.
No model-generated answer is treated as evidence; judgments must reference text.
Potential alternative evidence remains provisional until human adjudication.

## Execution and provenance

Run `python scripts/diagnose_routing_v2.py --adapters /path/to/adapters
--output /path/to/new-diagnostics` in `requirements.training.txt` environment.
The runner records adapter/data/code hashes and per-state results. It has no
optimizer and never writes model weights. Runtime failures are surfaced and
partial outputs preserved. GPU resources are removed after artifact verification.
Any paid execution remains within the previously authorized total budget of $10,
including the approximately $1.61 prior v2 allocation.

### Hardware comparability amendment

RTX4090 capacity was unavailable. The allocated A40 runs the same pinned software,
NF4/BF16 precision and original weights. After base diagnostics completed, 4/120
initial selections differed from archived RTX4090 traces. Before interpreting
cross-condition differences, replay all three fixed greedy development policies
on the A40 using `scripts/replay_routing_v2.py`. Preserve original traces and the
audit sample unchanged. This is a comparability check, not a policy adjustment;
no new model selection or held-out test is performed.
