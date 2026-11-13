# Changelog

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
