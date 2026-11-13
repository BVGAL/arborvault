# Changelog

## Unreleased

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
