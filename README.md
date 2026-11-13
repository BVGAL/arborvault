# ArborVault

[Automated checks](.github/workflows/ci.yml)

**Structure-aware knowledge retrieval and document memory for any AI agent.**

ArborVault stores document hierarchies in PostgreSQL and returns source-backed sections
through a small HTTP API. Use it as an experimental retrieval component in an agent or
RAG pipeline. The framework-neutral API supports reusable integrations with any AI agent.

**Latest development experiment:** independent candidate scoring with BM25 reranking
reaches Recall@5 **0.8000** for Qwen base and **0.7867** for its new LoRA. A small
retrieval-pretrained cross-encoder improves from **0.7217** to **0.8100**; standard-text
E5 vector retrieval reaches **0.7100**. Tree traversal trails matched flat reranking,
and abstention remains unresolved. This is exploratory development on 100 answerable
and 20 unknown questions, with one training seed; no production model was replaced.
Read the [complete candidate-scoring results](docs/candidate-scoring-v1-results.md)
and [frozen protocol](docs/candidate-scoring-v1-protocol.md). These are **development**
scores and must not be compared directly with the historical test figures below.

**Historical routing v2 pilot:** on a separate QASPER test, base plus lexical retrieval
reaches Recall@5 **0.7558**, versus **0.7442**
for MiniLM vector retrieval; this small difference does not establish superiority.
Greedy navigation is **0.7058** for the base and **0.4025**
for the selected adapter. With up to four decoding hypotheses, the matched results are
**0.7375** and **0.6983**.
The additional beam-base-plus-lexical ablation reaches **0.8025**,
with uncertainty overlapping the dense baseline; it was not promoted based on this test.
Fine-tuning fails the development promotion rules; retain the base.

See the [complete results, uncertainty and failures](docs/routing-v2-results.md),
[fixed protocol](docs/routing-v2-experiment.md) and [dataset card](datasets/qasper-routing-v2/README.md).
This pilot retrieves paragraphs inside a known document; it does not measure corpus-wide
search or generated-answer quality. The [historical v1 experiment](docs/evaluation.md#qwen3-4b-qlora-on-runpod)
and its [error audit](docs/qwen3-error-analysis.md) remain available separately.

**Post-hoc diagnosis:** adapted routing already loses recall on correct oracle states.
Read the [base / step-88 / step-175 diagnostics and 50-failure evidence audit](docs/routing-v2-diagnostics.md).
The audit is AI-assisted and leaves the published labels and test scores unchanged.

**Status: v0.3 research preview; production qualification is still ongoing.** It is not a drop-in Qdrant
replacement and does not establish that structured retrieval outperforms dense retrieval.
See [v0.3 release notes](docs/releases/v0.3.0.md), [upgrade instructions](docs/upgrade-v03.md)
and the historical [v0.1 validation record](docs/validation.md).

## Relationship to STAIR

Inspired by the use of document structure in [STAIR, Kumar et al. (2026)](https://arxiv.org/abs/2609.03874v2).
STAIR trains a model to retrieve table-of-contents entries using corpus knowledge in its
parameters. ArborVault instead persists a mutable tree and navigates it at query time,
with a deterministic local selector or optional external model. No fine-tuning is required.
This is an **independent project, not affiliated with or endorsed by IBM, not an official
STAIR implementation, and not a verified reproduction**. Paper results do not describe
this repository. No paper code, SearchTome books, or SearchTome benchmark data are bundled.

## Quick start

Requires Docker with Compose v2. Run from this repository:

```sh
python scripts/configure.py
docker compose up --build -d
curl http://localhost:8000/readyz
```

The configuration script generates database passwords in ignored `.env`, API credential
hashes in ignored `.secrets/keys.json` and a bearer token in `.arborvault-demo-token`.
It never prints secret values. Load that token into your client environment as
`ARBORVAULT_API_KEY`. Existing v0.1 users must follow the upgrade guide before starting.
The API binds only to `127.0.0.1`; PostgreSQL and the parser have no published ports.
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
- Tenant derived from the credential, namespace allowlists, document ACLs with stable
  subjects, separate read/write/delete permissions, hot credential revocation and request quotas.
- Markdown heading trees (including fenced-code handling), UTF-8 text, HTML headings,
  DOCX headings/tables, PDF page extraction. PDFs are page trees, not inferred chapters;
  scanned pages support Tesseract OCR (`auto`, `always`, `off`). DOCX tables follow paragraphs and lose original placement.
- PostgreSQL adjacency tree, materialized title paths, scoped foreign keys, GIN full-text
  index, immutable retained revisions, atomic upserts and idempotent unchanged uploads.
- Numbered checksum-verified migrations, PostgreSQL Row-Level Security and a restricted
  runtime role; historical tree reads, cursor pagination and consistent retrieval snapshots.
- Disposable parser workers with deadlines/resource limits, in a separate no-egress
  container without database credentials when using Compose; backup/restore verification.
- Lazy PostgreSQL beam navigation, optional Ollama/OpenAI-compatible selector, validated node IDs, lexical fallback,
  query trace, source locator, digest and revision with every result.
- A frozen [public-document dataset](datasets/public-docs-v1/README.md), PostgreSQL-backed
  comparison, synthetic smoke fixtures, regression tests and PostgreSQL CI.
- A [routing training dataset](datasets/routing-bootstrap-v1/README.md), optional LoRA recipe,
  bounded Ollama evaluation and guarded base/fine-tuned [comparison workflow](docs/llm-evaluation.md).
- A separate [QASPER routing v2 dataset](datasets/qasper-routing-v2/README.md), heading-first
  Python navigation engine, constrained candidate selection, two-seed Qwen3 training and
  a [document-scoped experiment protocol](docs/routing-v2-experiment.md). This experimental
  engine does not automatically replace the existing HTTP API's navigation path.
- Apache-2.0 license for original code, documentation and synthetic fixtures.

## API

Every data route starts with `/v1/namespaces/{namespace}` and needs `Authorization: Bearer …`.
Clients cannot choose their tenant in a request body or a header.

| Method and suffix | Behavior |
| --- | --- |
| `POST /documents` | Raw file; `filename`, `source_key`, `title`; optional `source_uri`, `visibility=private`, `ocr_mode=auto` |
| `GET /documents?limit=100&cursor=…` | Current documents; follow `next_cursor` until null |
| `GET /documents/{id}/revisions` | Retained revisions, descending; supports limit/cursor |
| `GET /documents/{id}/tree?revision=1&limit=500` | Historical/current tree; follow `next_cursor` until null |
| `GET /documents/{id}/acl`, `PUT /documents/{id}/acl` | Owner or scoped `manage_acl` administrator; read/write/delete subject lists |
| `DELETE /documents/{id}` | Permanently delete document and every retained revision |
| `POST /query` | `{ "query": "…", "k": 5, "mode": "auto" }` |

Modes: `auto` navigates then fills with lexical hits on route failure/insufficient results;
`navigation` disables fallback; `lexical` is an ablation. Scores are ranking heuristics,
not calibrated probabilities; scores from different methods are not directly comparable.
An empty response means no hit was found, not a proof that the corpus cannot answer.
No answer generator is included: agents receive untrusted source text and citations.

Uploads sharing a `(tenant, namespace, source_key)` update the same document. Unchanged
bytes and metadata keep the revision; changes append a new revision atomically. Node IDs
are revision-specific and previous content remains readable until the document is deleted.
Deleting and re-ingesting a source creates a new document ID so old citations never silently
resolve to new content. Source URIs are metadata and are never fetched.

## Optional model routing

Local title/excerpt matching is the default and has limited synonym understanding.
For local Qwen3-4B without a cloud subscription, use the explicit `qwen3:4b-q4_K_M` tag and
`python scripts/configure_ollama.py --local --docker-client`. See the
[installation and reproducible benchmark guide](docs/local-ollama.md). The generic `qwen3:4b`
tag can identify a different variant, so published runs record the actual model digest.

For Ollama Cloud, run `python scripts/configure_ollama.py --model gpt-oss:20b` locally.
It prompts for the key without displaying it and writes only ignored `.env` settings.
Restart the API to opt into sending authorized candidate excerpts to the provider.
The native adapter uses `SELECTOR_PROVIDER=ollama` and `https://ollama.com/api`.
Call/output caps and provider failures are recorded by the evaluation runner. Cloud
inference and user-run fine-tuning are separate operations; no paid job starts during setup.

Alternatively set `SELECTOR_URL` to an OpenAI-compatible **base URL ending in `/v1`**, `SELECTOR_MODEL`,
and `SELECTOR_API_KEY` in the server environment. The adapter calls `/chat/completions`.
Queries, titles and short excerpts will be sent to that provider; this is explicitly opt-in.
Responses must contain only a JSON array of supplied node IDs. Unknown IDs, invalid JSON,
timeouts and request failures trigger lexical fallback in `auto` mode.
This adapter is not a fine-tuned STAIR model. Use a trusted provider endpoint.

## Development and evaluation

```sh
python -m venv .venv
# Activate .venv for your shell
pip install -c constraints.txt -e '.[dev,ocr]'
pytest
ruff check .
python -m arborvault.evaluate --output reports/smoke.json
python -m arborvault.dataset datasets/public-docs-v1
# Actual dense embedding baseline (downloads model weights on first run):
pip install -c constraints.txt -e '.[eval]'
python -m arborvault.evaluate --dense --output reports/dense.json
# With EVAL_DATABASE_URL pointing to a migrated disposable database's runtime role:
python -m arborvault.benchmark datasets/public-docs-v1 --split dev --dense --output reports/dev.json
```

In the synthetic runner, without `--dense`, the vector baseline uses TF-IDF cosine and is labeled as
lexical, not semantic embeddings. With `--dense`, FastEmbed runs MiniLM embeddings and
exact cosine search over fixed-size chunks. This compares the retrieval stage of vector
RAG, not end-to-end answer quality or Qdrant deployment performance. See
[evaluation](docs/evaluation.md) for metrics, corpus alignment, controls and limitations.
The public-docs runner uses the actual PostgreSQL implementation and identical section
units/context budgets for all methods. Its 60 questions are assistant-authored drafts
awaiting independent human review; see the dataset card before interpreting results.
`constraints.txt` records the Linux package versions used for the test/evaluation
run; it is a version constraint file, not a hash-verified lockfile. Update it deliberately
and rerun tests/audit when upgrading dependencies.

Set `TEST_DATABASE_URL` to an **isolated disposable PostgreSQL database** to run integration
tests as well. CI supplies one automatically. Those tests create and remove their own
random tenant data. Schema initialization is a separate command:
`python -m arborvault.migrate` using an administrator `DATABASE_URL`. Compose also provisions
the non-owner runtime role before starting the API. See [operations](docs/operations.md).

To run every test in isolated Linux containers, including PostgreSQL:
`docker compose -f compose.test.yaml up --build --abort-on-container-exit --exit-code-from tests`.
Then `docker compose -f compose.test.yaml down` removes the test containers.

## Architecture and limits

See [architecture](docs/architecture.md), [security](SECURITY.md) and
[Agent integration](docs/agent-integration.md). The memory store exists only for tests/evaluation;
the deployed server requires PostgreSQL and valid credentials and fails closed otherwise.

Default parsing budgets: 25 MB request body, 20 million extracted characters, 50,000 nodes
per document and 2,000 PDF pages. These are configurable deployment budgets, not throughput
guarantees; CPU, memory and time limits can reject a complex file earlier. OCR is installed
in the Compose parser; native installations also need Tesseract language packs.
Navigation loads up to 160 candidates per level, depth 12, beam width 4. Large sibling sets
are shortlisted by lexical routing-summary score in PostgreSQL; there is no 10,000-node
corpus cutoff. This shortlist can miss semantic matches. Default summaries are extractive,
not learned. There is no streaming ingestion, shared quota across replicas, automatic memory consolidation,
or cross-document answer reasoning. Local subprocess parsing is weaker isolation
than the provided Compose service. Retained history increases storage use; DELETE purges all
revisions. Use a gateway with TLS and shared quotas for a multi-instance deployment.

## Roadmap

1. Validate on independently reviewed agent workloads and publish retrieval/abstention measurements.
2. Add durable ingestion jobs, shared resource quotas and broader scale measurements.
3. Add hybrid dense retrieval only if held-out measurements justify it.
4. Expand independently reviewed routing data and unseen corpora before further Qwen3
   fine-tuning. The first measured adapter is experimental; weights are not hosted on Hugging Face.
5. Explore a separately versioned learned ToC selector and SearchTome evaluation after
   checking dataset rights. Any reproduction claim requires its own evidence.

Contributions: see [CONTRIBUTING.md](CONTRIBUTING.md). License: [Apache-2.0](LICENSE).
