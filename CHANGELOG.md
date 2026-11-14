# Changelog

## Unreleased

- Fix verifier pair tokenization for Transformers 5. Add pinned real-tokenizer
  regression checks and a feature-only recovery command with source and artifact
  provenance. All 39 targeted GPU-environment tests pass; retrieval outputs remain
  byte-identical across the compatibility repair.

- Validate the frozen 64-pair navigation policies on 20 unseen manuals with 480
  test cases, plus four disjoint calibration manuals; compare Qwen base, MiniLM
  cross-encoder and full-leaf E5 with paired family-bootstrap intervals.
- Add a frozen-encoder linear acceptance probe, historical document-separated
  fitting/selection, joint conservative gating and a question-only shortcut
  control. Publish every operating point and model without test-selected tuning.
- Verify all artifact and corpus hashes, exact traversals and verifier input
  contexts. Complete 129 local tests (six optional skips) and delete the Runpod
  allocation. Cumulative estimated spend: USD 7.94877/10.
  See the [expanded validation report](docs/abstention-v3-results.md).

- Add experimental incremental navigation with retained alternatives, an explicit
  branch allowance, and a separately identified lexical-assisted variant.
- Compare Qwen base, pretrained cross-encoder and Tacit-4B using matched 64/128
  pair budgets, shared candidate text and identical Qwen/Tacit decision prompts.
  Freeze six new manuals with 48 answerable and 48 unknown test cases, plus four
  separate calibration manuals. Publish all four policy ablations and E5/BM25 controls.
- Verify raw hashes, exact traversals, ancestor access, candidate coverage,
  calibration-only abstention and five-family paired bootstrap intervals.
  Report synthetic-label and small-sample limitations without production promotion.
- Complete 123 local tests (six optional skips), 31 GPU-environment tests and
  delete the single Runpod allocation. Cumulative estimated cost:
  USD 6.70558 / 10. See [results](docs/navigation-v2-results.md).

- Add a separately frozen 16-manual corpus: 12 test documents with 120 answerable
  and 168 scoped-unanswerable cases; four document-disjoint calibration manuals.
  Preserve licensed source snapshots, annotation rationales and explicit synthetic-data limits.
- Compare Qwen base, a fixed prior adapted cross-encoder, its pretrained base and E5.
  Primary flat recall: 0.9167, 0.9250, 0.9417, 0.9500, respectively. Publish matched
  tree/flat runs at 64/128 pairs, calibrated abstention, all traces and descriptive uncertainty.
- Restore meaningful paragraph previews before inspecting outcomes; disclose the interrupted
  attempt, rerun every system and include all allocated time in cost. Add a regression check.
- Verify all artifact hashes and replay calculations, delete the GPU allocation, and report
  cumulative estimated spend of USD 5.79872 within the USD 10 budget.
  Full local suite: 112 passed, six optional skips; GPU checks: 26 passed.
- Make BM25 summation deterministic across Python hash seeds after detecting machine-precision
  tie-order drift in offline replay. Reproduce every original neural traversal exactly;
  preserve all scores and annotations and publish the numerical audit.

- Complete five-fold document CV over all 840 existing QASPER questions, with separate
  inner roles for weight updates, checkpoint choice, gate fitting and threshold tuning.
  Explicitly disclose retrospective reuse of historical test questions; retain old reports.
- Report pooled coverage-first recall: Qwen base 0.7144, adapted cross-encoder 0.6708,
  cross-encoder base 0.6580, E5 0.5851. The training gain's paired interval includes zero.
- Add query-level abstention with raw-score and learned gates, conservative and balanced
  thresholds, full false-abstention accounting, discrimination and calibration diagnostics.
  The adapted cross-encoder's simple conservative gate rejects 15/140 unknowns and 17/700
  answerable questions; its learned gate performs worse at the observed tradeoffs.
- Verify all 154 downloaded artifacts and 20 out-of-fold policy reports, retain checkpoints
  locally, delete the Runpod allocation and publish an estimated cumulative USD 4.99 spend.
  No production promotion or model-hub upload; full local suite: 101 passed, six optional skips.

- Add an experimental candidate scorer with matched flat/tree retrieval, independent
  yes/no scores, weak-negative supervision and a small cross-encoder comparison.
  Train both on 457 document-separated questions, calibrate on 143 and evaluate
  on the existing 120-question development split without re-evaluating the old test.
- Publish all checkpoint curves, per-question results, paired uncertainty, E5/BM25/RRF
  references and the disclosed scored-leaf retention control. Flat development recall:
  Qwen base 0.8000, new LoRA 0.7867; cross-encoder base 0.7217, adapted 0.8100;
  standard-text E5 0.7100. Keep the production default; no claim of general superiority.
- Record failed abstention calibration, full artifact verification and Runpod deletion.
  Retain checkpoint weights locally with public hashes; no model-hub release.

- Diagnose base and existing seed-42 checkpoints on 244 development oracle states each,
  with teacher-forced token metrics, free decisions and matched autonomous replay.
  Publish a masked AI-assisted audit of 50 failures plus five controls, full context checks,
  verified artifacts and resource cleanup; no new fine-tuning or changed test scores.

- Publish the separate v2 test: greedy base 0.7058, adapted
  0.4025; with up to four hypotheses, base 0.7375, adapted
  0.6983. Base plus lexical 0.7558; MiniLM 0.7442.
  Neither adaptation passes the development promotion gate; preserve the base recommendation.
- Report the beam-plus-lexical ablations too: base 0.8025,
  adapted 0.7633. The base's numerical lead over MiniLM
  has an uncertainty interval crossing zero and does not establish general superiority.
- Disclose the incomplete beam attempt and retry with memory-bounded SDPA scheduling, preserving
  weights, context, precision and decoding policy. Verify and retain artifacts before pod deletion.
- Add a separate routing-v2 research pilot: 600 human QASPER training questions, 120 development
  and 120 test questions, with disjoint documents, evidence mappings and source/license hashes.
- Add heading-first tree outlines, shared tokenizer-aware features, constrained valid-ID
  selection, per-depth diagnostics and a development-triggered four-hypothesis decoding ablation.
- Train two Qwen3-4B LoRA seeds for one epoch, retain four checkpoints and select using development
  retrieval with abstention/error guards. Keep the base when adaptation fails the promotion gate.
- Add paired document-bootstrap summaries, artifact validation and 78 passing application tests;
  four optional PyTorch checks also pass in the training environment (15 runtime checks total).
- Audit the Qwen3 regression per query: six losses/one gain, 3-to-9 first-call rejections,
  and no real subsection-routing examples in training. Add a reproducible offline diagnostic,
  gold-ancestor ceiling check, and an initial v2 dataset/training plan, subsequently implemented as a separate pilot.
- Correct the failure description: navigation retains evidence collected before an invalid
  selection; the 4/12 empty answerable results are separate valid no-route decisions.
- Train Qwen3-4B QLoRA on one Runpod RTX 4090: 392 train/84 validation examples,
  three epochs and 147 optimizer steps; select epoch two by validation loss only.
- Publish matched NF4 base/adapter retrieval runs with unchanged corpus and MiniLM weights.
  Test Recall@5: base 0.6111, adapted 0.4722; with lexical fallback 0.6667 and 0.6944;
  MiniLM 0.9167. Two invalid selections per model keep the comparison exploratory.
- Retain the initial and repeated reports. Correct multi-EOS validation and add failure-stage
  diagnostics; repeated retrieval scores are identical, with no retraining or test-driven tuning.
- Add pinned optional training dependencies, adapter/checkpoint hashes and portable provenance.
- Verify 64 regression tests (one optional Torch test skipped in the API environment) and
  all five Qwen runtime checks in the training environment.
- Add local Qwen3-4B configuration, explicit thinking/context controls and model provenance checks.
- Add a disposable-database benchmark runner and publish all development/test query results.
- Measure 270 local Ollama baseline calls across both splits without provider/JSON errors,
  using unmodified weights.
- Record the generic-tag Thinking-2507 mismatch and pin the original `qwen3:4b-q4_K_M` artifact.
- Test Recall@5: Qwen navigation 0.5278, Qwen + lexical 0.6389, MiniLM dense 0.9167.
- Add local controls and provenance privacy regression coverage.

## 0.3.0 — 2026-11-13

- Add optional page OCR with PDFium/Tesseract in isolated parser workers.
- Make upload/text/page/node/resource budgets configurable; raise conservative defaults.
- Enforce document read/write/delete ACLs across current and historical API surfaces and RLS.
- Fetch navigation candidates lazily; remove the 10,000-node automatic lexical handoff.
- Add tree cursor pagination, routing-summary migration and scoped ACL administration.
- Add native Ollama routing with request/output caps and usage accounting.
- Publish reproducible routing bootstrap data, an optional attention-LoRA recipe and paired evaluation checks.
- Keep public test labels out of training; distinguish cloud/runtime differences and draft annotations.
- No trained model, paid cloud benchmark result or verified STAIR reproduction is included.

## 0.2.0 — 2026-11-13

- Add checksum-verified, transactional migrations adopting v0.1 without losing its current tree.
- Retain immutable document revisions, expose history and historical trees, and paginate listings.
- Use fresh IDs after deletion/re-ingestion and consistent snapshots during retrieval.
- Enforce PostgreSQL RLS with a separate restricted runtime database role.
- Add explicit read/write/delete permissions, hashed file-backed credentials, hot revocation and request quotas.
- Isolate parsing in disposable resource-limited workers and an internal no-egress Compose service.
- Add backup/restore verification, upgrade guidance, digest-pinned base images and versioned build tooling.
- Add eight licensed public source documents and 60 draft annotated evaluation queries with dev/test splits.
- Add a PostgreSQL-backed evaluator with matched section units/context budgets, evidence coverage and BEIR-style export.
- Expand regression coverage for migrations, authorization, history, isolation, parser failure and dataset integrity.

## 0.1.0 — 2026-11-13

Initial independent structured-retrieval MVP with FastAPI, PostgreSQL, parsing, navigation,
lexical fallback, tenant namespaces, synthetic evaluation and Apache-2.0 licensing.
