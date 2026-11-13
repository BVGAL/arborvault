# Initial validation — 2026-11-13

Validation of the initial MVP used Linux containers with Python 3.12 and PostgreSQL 17.12.

- 17 tests passed, including real PostgreSQL persistence, concurrent upserts, transactional
  failure handling, tenant/namespace isolation, deletion and scoped foreign-key enforcement.
- Markdown hierarchy/fenced code, text continuation, HTML, DOCX and a generated text PDF
  were exercised, together with malformed inputs, request size and NUL rejection.
- API credentials fail closed; unknown model IDs and selector errors use lexical fallback.
- OpenAPI describes the upload body and typed document/tree/query responses.
- Docker Compose built and started the migration, database and non-root API successfully.
  `/readyz` returned ready; a real HTTP upload/query retrieved the synthetic backup passage.
- Ruff checks and Git whitespace checks passed.
- The [dense retrieval smoke benchmark](benchmarks/synthetic-dense.json) completed with
  local MiniLM embeddings. See [evaluation limitations](evaluation.md).
- `pip-audit` reported no known vulnerabilities in the final installed test environment
  after upgrading pip. The local ArborVault package is not covered by an advisory database;
  dependency scanning is not a security audit of application code or container OS packages.
- A tracked-file secret-pattern scan and manual provenance review found no live credentials,
  personal paths, private agent code or company data in the publication set. Local `.env`
  and downloaded model weights are excluded from Git and source archives.

External model routing and downstream agent deployments were not tested. Source parsing
and retrieval still have the limits described in the README. The CI workflow repeats unit
and PostgreSQL integration checks; benchmark quality claims require a larger held-out dataset.
