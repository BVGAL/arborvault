# ArborVault

[Automated checks](.github/workflows/ci.yml)

**Structure-aware knowledge retrieval and document memory for any AI agent.**

ArborVault stores document hierarchies in PostgreSQL and returns source-backed sections
through a small HTTP API. Use it as an experimental retrieval component in an agent or
RAG pipeline. The framework-neutral API supports reusable integrations with any AI agent.

**Status: working research MVP, not production hardened.** It is not a drop-in Qdrant
replacement and does not establish that structured retrieval outperforms dense retrieval.
See the [initial validation record](docs/validation.md) for what was actually tested.

## Relationship to STAIR

Inspired by the use of document structure in [STAIR, Kumar et al. (2026)](https://arxiv.org/abs/2609.03874v2).
STAIR trains a model to retrieve table-of-contents entries using corpus knowledge in its
parameters. ArborVault instead persists a mutable tree and navigates it at query time,
with a deterministic local selector or optional external model. No fine-tuning is required.
This is an **independent project, not affiliated with or endorsed by IBM, not an official
STAIR implementation, and not a verified reproduction**. Paper results do not describe
this repository. No paper code, SearchTome books, or benchmark data are bundled.

## Quick start

Requires Docker with Compose v2. Run from this repository:

```sh
python scripts/configure.py
docker compose up --build -d
curl http://localhost:8000/readyz
```

The configuration script generates local credentials in ignored `.env`; it never prints
their values. Copy the `demo` bearer key from `.env` into your client environment as
`ARBORVAULT_API_KEY`. The API binds only to `127.0.0.1`; PostgreSQL has no published port.
Interactive API documentation: <http://localhost:8000/docs>.

```sh
curl -X POST 'http://localhost:8000/v1/namespaces/demo/documents?filename=guide.md&source_key=guide&title=Demo%20guide' \
  -H "Authorization: Bearer $ARBORVAULT_API_KEY" \
  -H 'Content-Type: application/octet-stream' --data-binary @examples/guide.md

curl 'http://localhost:8000/v1/namespaces/demo/query' \
  -H "Authorization: Bearer $ARBORVAULT_API_KEY" -H 'Content-Type: application/json' \
  -d '{"query":"How long are backups retained?","k":3}'
```

These are POSIX shell examples. On Windows, use `curl.exe` and the appropriate environment
variable syntax, or run `python examples/agent_client.py` after setting the key.

## Included

- FastAPI, versioned HTTP routes, generated OpenAPI, bearer authentication.
- Tenant derived from the credential, namespace allowlists, scoped reads and writes.
- Markdown heading trees (including fenced-code handling), UTF-8 text, HTML headings,
  DOCX headings/tables, PDF page extraction. PDFs are page trees, not inferred chapters;
  scanned PDFs need external OCR. DOCX tables follow paragraphs and lose original placement.
- PostgreSQL adjacency tree, materialized title paths, scoped foreign keys, GIN full-text
  index, atomic upsert/revision replacement and idempotent unchanged uploads.
- Bounded beam navigation, optional model selector, validated node IDs, lexical fallback,
  query trace, source locator, digest and revision with every result.
- Runnable comparative evaluation, synthetic data, tests and PostgreSQL CI.
- Apache-2.0 license for original code, documentation and synthetic fixtures.

## API

Every data route starts with `/v1/namespaces/{namespace}` and needs `Authorization: Bearer …`.
Clients cannot choose their tenant in a request body or a header.

| Method and suffix | Behavior |
| --- | --- |
| `POST /documents` | Raw file body; query fields `filename`, `source_key`, `title`, optional `source_uri` |
| `GET /documents` | List current documents (MVP limit: 1,000) |
| `GET /documents/{id}/tree` | Current tree with content, IDs and source locators |
| `DELETE /documents/{id}` | Delete current document and its tree |
| `POST /query` | `{ "query": "…", "k": 5, "mode": "auto" }` |

Modes: `auto` navigates then fills with lexical hits on route failure/insufficient results;
`navigation` disables fallback; `lexical` is an ablation. Scores are ranking heuristics,
not calibrated probabilities; scores from different methods are not directly comparable.
An empty response means no hit was found, not a proof that the corpus cannot answer.
No answer generator is included: agents receive untrusted source text and citations.

Uploads sharing a `(tenant, namespace, source_key)` update the same document. Unchanged
bytes and metadata keep the revision; changes replace the tree atomically. Node IDs are
revision-specific. Historical content is not retained: consumers must keep returned text
and digests when they need durable evidence. Source URIs are metadata and are never fetched.

## Optional model routing

Local title/excerpt matching is the default and has limited synonym understanding.
Set `SELECTOR_URL` to an OpenAI-compatible **base URL ending in `/v1`**, `SELECTOR_MODEL`,
and `SELECTOR_API_KEY` in the server environment. The adapter calls `/chat/completions`.
Queries, titles and short excerpts will be sent to that provider; this is explicitly opt-in.
Responses must contain only a JSON array of supplied node IDs. Unknown IDs, invalid JSON,
timeouts and request failures trigger lexical fallback in `auto` mode.
This adapter is not a fine-tuned STAIR model. Use a trusted provider endpoint.

## Development and evaluation

```sh
python -m venv .venv
# Activate .venv for your shell
pip install -c constraints.txt -e '.[dev]'
pytest
ruff check .
python -m arborvault.evaluate --output reports/smoke.json
# Actual dense embedding baseline (downloads model weights on first run):
pip install -c constraints.txt -e '.[eval]'
python -m arborvault.evaluate --dense --output reports/dense.json
```

Without `--dense`, the vector baseline uses TF-IDF cosine and is explicitly labeled as
lexical, not semantic embeddings. With `--dense`, FastEmbed runs MiniLM embeddings and
exact cosine search over fixed-size chunks. This compares the retrieval stage of vector
RAG, not end-to-end answer quality or Qdrant deployment performance. See
[evaluation](docs/evaluation.md) for metrics, corpus alignment, controls and limitations.
`constraints.txt` records the Linux package versions used for the initial test/evaluation
run; it is a version constraint file, not a hash-verified lockfile. Update it deliberately
and rerun tests/audit when upgrading dependencies.

Set `TEST_DATABASE_URL` to an **isolated disposable PostgreSQL database** to run integration
tests as well. CI supplies one automatically. Those tests create and remove their own
random tenant data. Schema initialization is a separate command:
`python -m arborvault.storage` using `DATABASE_URL`. Compose runs it before starting the API.

To run every test in isolated Linux containers, including PostgreSQL:
`docker compose -f compose.test.yaml up --build --abort-on-container-exit --exit-code-from tests`.
Then `docker compose -f compose.test.yaml down` removes the test containers.

## Architecture and limits

See [architecture](docs/architecture.md), [security](SECURITY.md) and
[Agent integration](docs/agent-integration.md). The memory store exists only for tests/evaluation;
the deployed server requires PostgreSQL and valid credentials and fails closed otherwise.

MVP limits: 2 MB request body, 1 million extracted characters, 2,000 nodes/document,
200 PDF pages, 10,000 nodes loaded for navigation, 160 candidates/level, depth 12,
beam width 4. Larger corpora fall back to PostgreSQL lexical retrieval. No streaming
ingestion, OCR, user-level document ACLs, RLS, production quotas, history, automatic
memory consolidation, cross-document reasoning, or training pipeline is included.
Resource limits reduce risk but do not make parsers a security boundary; isolate ingestion
workers before accepting hostile public uploads. Authentication grants read/write/delete
within allowed namespaces; split deployment credentials if different roles are required.

## Roadmap

1. Validate on independently reviewed agent workloads and publish retrieval/abstention measurements.
2. Fetch children lazily from PostgreSQL; add jobs, parser isolation, pagination and RLS.
3. Add hybrid dense retrieval only if held-out measurements justify it.
4. Explore a separately versioned learned ToC selector and SearchTome evaluation after
   checking dataset rights. Any reproduction claim requires its own evidence.

Contributions: see [CONTRIBUTING.md](CONTRIBUTING.md). License: [Apache-2.0](LICENSE).
