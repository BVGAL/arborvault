# Public Docs v1 — draft retrieval evaluation set

**60 assistant-authored draft questions over eight real public documentation files.**
This is a reviewable starting dataset, not a human-certified benchmark. It is distinct
from the synthetic v0.1 smoke fixtures and from IBM's SearchTome.

| Property | Value |
| --- | --- |
| Corpus | FastAPI 0.115.0 tutorials; HTTPX 0.28.1 documentation |
| Source snapshot | Exact upstream commits and SHA-256 hashes in `manifest.json` |
| Questions | 48 answerable; 12 deliberately unanswerable |
| Splits | 16 development; 44 test; families grouped by source document |
| Languages | English and Italian questions; English source documents |
| Labels | Source file, heading line and supporting excerpt in `queries.jsonl` |
| Annotation status | Assistant draft; independent human review outstanding |

The corpus remains unchanged from the pinned upstream Markdown files. Upstream code
include directives and image references remain literal; referenced assets are not bundled
or fetched. Questions were written against available prose, not missing code snippets.
The corpus covers developer documentation only. It does not establish performance across
business, medical, legal or scientific knowledge, and it does not contain company data.

## Rights and provenance

FastAPI documents retain the MIT license and copyright of Sebastián Ramírez; HTTPX
documents retain the BSD-3-Clause license and copyright of Encode OSS Ltd. Complete
notices are in `licenses/`; paths and hashes are recorded per document. Original query
annotations and dataset tooling are Apache-2.0. Inclusion does not imply upstream endorsement.
Some upstream examples contain fictitious passwords or public author paths; these are
unmodified public documentation examples, not private ArborVault/user credentials.

## Validate and export

```sh
python -m arborvault.dataset datasets/public-docs-v1
python -m arborvault.dataset datasets/public-docs-v1 --export reports/public-docs-export
```

Validation checks corpus and license hashes, source revisions, unique query IDs, split
families, answerability and the existence of every labeled excerpt in its parsed section.
It cannot establish semantic completeness of relevance judgments. Export contains
`corpus.jsonl`, `queries.jsonl`, split-specific `qrels/*.tsv` and `labels.json`. Unanswerable
queries have no positive qrels; retain `labels.json` because some external evaluators
silently exclude queries without positive judgments.

## Run against PostgreSQL

Use a disposable, migrated database and the restricted runtime role. Set
`EVAL_DATABASE_URL` locally; do not put it in command history or reports.

```sh
python -m arborvault.benchmark datasets/public-docs-v1 --split dev --output reports/dev.json
python -m arborvault.benchmark datasets/public-docs-v1 --split test --dense --output reports/test.json
```

The runner creates a unique evaluation tenant and deletes only its own inserted documents
in a `finally` block. It uses the deployed PostgreSQL lexical implementation. All methods
receive the same parsed sections, top-k limit and a 512-whitespace-word context budget.
Dense comparison embeds those sections; it is explicitly not a fixed-chunk RAG comparison.
Model weights download only with `--dense`. The previous chunk-based smoke runner remains
available separately as `python -m arborvault.evaluate`.

Reports include section Recall@k/MRR/nDCG, evidence coverage after truncation, no-answer
abstention, per-query results, p50/p95 retrieval latency, corpus/annotation hash and model
artifact hashes. A relevant section truncated before its evidence can still count toward
section recall, but not evidence coverage. Whitespace words are an approximation rather
than model tokens. No answer generator, reranker or end-to-end answer judge is evaluated.
The small timing sample is diagnostic, not an infrastructure capacity measurement.

## Annotation review and freeze policy

1. Independently verify each question is answerable from the pinned sources or is absent.
2. Check the supporting excerpt and add every other relevant section, not just the first match.
3. Review Italian questions for faithful meaning and identify ambiguities.
4. Keep related paraphrases in one split. Test questions and labels must not drive tuning.
5. Mark reviewed records with reviewer identity/agreement only after actual human review.
6. If labels or corpus change, bump the dataset version and retain old report hashes.

The first v0.2 report is an exploratory baseline. Test results are public and have been
seen by the implementer; future tuning requires a fresh independently authored test set.
