# Architecture — v0.3

```mermaid
flowchart LR
  A[Any AI agent] -->|HTTP + scoped credential| API[FastAPI v1]
  API --> AUTH[Credential hashes + operation permissions]
  API --> P[Private parser service]
  P --> W[Disposable worker: time + resource limits]
  W --> T[Sections and original extracted text]
  T --> API
  API --> DB[(PostgreSQL: documents / revisions / nodes)]
  API --> R[Navigation + lexical fallback]
  R -->|Repeatable-read snapshot + tenant scope| DB
  R --> C[Passages + immutable revision citations]
  C --> A
  M[Administrator migration job] --> DB
```

`domain.py` defines records/store interfaces; `storage.py` implements PostgreSQL; `memory.py`
is explicitly test-only. `migrate.py` applies numbered SQL migrations atomically with an
advisory lock and SHA-256 checksums. It can adopt the existing v0.1 schema without changing
document IDs or losing the currently stored tree. `schema.sql` is retained solely as a
v0.1 upgrade-test fixture. Schema rollback is by backup restore, not destructive down scripts.

The logical ingestion key is `(tenant, namespace, source_key)`. A source-specific advisory
lock serializes upsert/delete/ACL operations. Each new document lifecycle has a fresh UUID;
updates preserve that ID and append a revision. Every revision has its own metadata and
nodes. Scoped foreign keys enforce parent/revision/document consistency. The runtime role
cannot update historical rows. A full DELETE cascades through all history; recreating a
source uses another UUID and cannot reassign an old citation to new evidence.

Read and write transactions set tenant, namespace, authenticated subject and operation using
transaction-local PostgreSQL settings. RLS policies on documents, revisions and nodes also
evaluate the current document ACL; the ACL table is tenant/namespace scoped. Missing context denies reads.
The API checks its database role is neither owner, superuser nor BYPASSRLS. SQL predicates
remain explicit as a second application-level check. The scope-setting connection is a
trusted service capability: RLS does not protect against a fully compromised API process
or a trusted administrator deliberately choosing another scope. Never expose SQL credentials
to tenants. The pool resets transaction-local context on commit/rollback.

Retrieval uses one repeatable-read, read-only transaction for navigation and lexical fallback,
so a concurrent ingestion cannot mix revisions within one response. ACL changes apply to
subsequent snapshots. Routing summaries are computed on ingestion. Each traversal fetches
only the root/child candidate window and selected original passages. Wide frontiers are
lexically shortlisted in PostgreSQL to 160 candidates; this bounds transfer and model input,
but ranking still scans eligible sibling summaries in the database. Local title/summary
matching remains the default. The optional model selector
can replace candidate selection; it cannot emit arbitrary executable operations. Unknown
IDs and model failures are validated and handled. PostgreSQL lexical scores are actual
`ts_rank_cd` values, not the test store's token-overlap approximation.

Compose runs parser HTTP separately on an internal network with no egress, credentials,
published port or persistent storage. Each parse creates a disposable subprocess with a
minimal environment, temporary directory, a configurable wall deadline and Linux CPU/address
space/file-size limits. At most two workers run per parser process. The container adds
non-root execution, read-only filesystem, dropped capabilities and PID/memory limits.
Without `PARSER_URL`, a local disposable subprocess is used; it lacks container network and
filesystem isolation and is intended for development or trusted inputs.

The HTTP layer enforces explicit `read`, `write` and `delete` permissions, reloads hashed
credential files per request, bounds request bodies, returns cursor pages and adds request
IDs. Request logs contain route templates, status and duration, not queries, documents,
credentials or source metadata. Quotas are per credential and API process; use a gateway
for aggregate quotas across replicas. The API's readiness check verifies schema availability.

`settings.py` owns deployment parsing budgets. `ocr.py` renders PDF pages with PDFium and
runs Tesseract only inside parser workers in Compose. Worker timeout terminates the whole
process group, including OCR subprocesses. OCR locators keep page numbers and add `ocr=true`.
The parser container includes English/Italian packs and no model-provider credentials.

`ollama.py` maps candidate IDs to request-local numeric labels, validates model output and
records call/token usage. `training_data.py` exports the same prompt contract with oracle
routes. `llm_eval.py` extends the PostgreSQL benchmark and rejects incompatible paired
comparisons. These modules train/evaluate a candidate selector, not corpus knowledge stored
inside model parameters as in STAIR.
