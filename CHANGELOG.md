# Changelog

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
