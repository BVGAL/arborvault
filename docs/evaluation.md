# Evaluation protocol

## v0.3 regression and LLM workflow

The v0.3 rerun uses the same frozen corpus, annotations, MiniLM weights, k=5 and 512-word
budget. Test Recall@5 remains 0.5833 for local navigation/auto, 0.4444 for lexical and
0.9167 for dense sections. Raw [dev](benchmarks/public-docs-v03-dev.json) and
[test](benchmarks/public-docs-v03-test.json) reports retain all per-query outcomes.
The dataset SHA-256 is unchanged. Timings remain exploratory single-pass measurements;
ACL checks and lazy SQL traversal alter database work and are not a scalability proof.

The [LLM workflow](llm-evaluation.md) adds native Ollama routing, base/adapted comparisons
and separately split [training data](../datasets/routing-bootstrap-v1/README.md). This release
validates the protocol with mocked provider responses; no paid model result or fine-tuned
model is claimed. A future comparison must evaluate the same original base checkpoint and
its adapter, with matched serving conditions and an independently reviewed held-out set.

## v0.2 public documentation baseline

The [dataset card](../datasets/public-docs-v1/README.md) describes eight licensed, pinned
FastAPI/HTTPX source files, 85 extracted content sections and 60 draft annotated questions.
Development has 12 answerable/4 unanswerable questions; test has 36/8. English and Italian
questions use English documents. All documents are indexed for both question splits;
families of related questions stay in the same split. This is retrieval evaluation, not
unseen-document generalization. Independent annotation review remains outstanding.

The runner is `python -m arborvault.benchmark datasets/public-docs-v1 --split dev --dense`.
Set `EVAL_DATABASE_URL` to the restricted runtime role in a migrated disposable database.
The runner cleans up its unique evaluation tenant. Validate and export BEIR-style data with
`python -m arborvault.dataset datasets/public-docs-v1 --export reports/beir`.

Frozen first baseline, run 2026-11-13 with k=5 and a 512-whitespace-word context budget:

| Method | Dev Recall@5 | Test Recall@5 | Test MRR | Test evidence coverage | Test no-answer abstention |
| --- | ---: | ---: | ---: | ---: | ---: |
| Navigation | 0.8333 | 0.5833 | 0.5162 | 0.5833 | 0.1250 |
| Navigation + lexical fallback | 0.8333 | 0.5833 | 0.5162 | 0.5833 | 0.1250 |
| PostgreSQL lexical | 0.4167 | 0.4444 | 0.4167 | 0.4722 | 0.1250 |
| MiniLM dense sections | 0.9167 | 0.9167 | 0.7000 | 0.9444 | 0.0000 |

Raw [development](benchmarks/public-docs-v02-dev.json) and
[test](benchmarks/public-docs-v02-test.json) reports include per-query results, hashes,
versions and latency. Dataset/annotation SHA-256:
`a76183361bf0a7e730850d38bcc5b172ac7386efd56ea4306c2b163ab1c5c408`.

**The dense baseline retrieves more labeled evidence on this dataset.** Local navigation
and fallback do not establish an advantage over vector retrieval. Every method still
returns irrelevant context for most unanswerable questions. Priorities are independently
reviewed labels and better rejection/calibration, evaluated on new held-out queries.
No retrieval logic or threshold was tuned after looking at these test results.

All four methods return the same extracted section units, limited to k and then truncated
in ranked order to the word budget. Section recall/MRR/nDCG score retained section IDs;
evidence coverage checks whether labeled excerpts survive truncation. Coverage can exceed
section recall when the same supporting text also appears in another section. Multiple
positive qrels can be alternate evidence, so recall is not answer accuracy. The MiniLM
model primarily targets English and its default tokenizer truncates long section inputs.
This is a useful baseline, not the strongest possible multilingual/hybrid/RAG system.

Environment: Docker Desktop Linux x86_64, 16 logical CPUs and approximately 15.4 GiB allocated
RAM; Python 3.12.14, PostgreSQL 17, two ONNX inference threads, cached model weights. Package
versions and model checksum are in each report. Latency is one sequential pass with no
warm/cold separation or repeated-trial confidence intervals. PostgreSQL methods include
database I/O while dense ranking runs in process; timings are diagnostic and do not compare
Qdrant with PostgreSQL. No generated answers, paid routing, reranker or quality judge is used.

## Historical v0.1 synthetic smoke protocol

The first [dense smoke report](benchmarks/synthetic-dense.json) was run on 2026-11-13
in Linux containers on CPU, at k=3. All four methods reached recall 1.0 on the ten
answerable toy questions. Unanswerable abstention was 0.5 for the local structured/lexical
methods and 0.0 for the untuned dense baseline. These tiny, easy examples verify the
evaluation path; they do not establish practical quality or comparative superiority.

Run `python -m arborvault.evaluate` for navigation, navigation plus fallback, lexical
retrieval and TF-IDF cosine chunk retrieval. Add `--dense` to replace TF-IDF with actual
MiniLM dense embeddings through FastEmbed. This runs exact cosine ranking in memory;
no Qdrant instance or external inference API is needed. First run downloads model weights.
Use `--cache-dir` to retain them; SHA-256 hashes of ONNX/JSON model artifacts are recorded.

The built-in dataset is 3 original synthetic documents, 9 content sections and 12 questions
(10 answerable, 2 deliberately unanswerable). It is a smoke benchmark, not a representative
dataset or a statistically powered experiment. Do not tune on it and claim generalization.
No IBM results or SearchTome scores are claimed. Local navigation and lexical evaluation
use MemoryStore's token-overlap ranking; PostgreSQL full-text behavior is covered separately
by integration tests. They are not equivalent ranking implementations.

All methods see the same extracted content. The vector baseline chunks the flattened
documents at 256 characters with a 64-character overlap, independently of section boundaries.
Its labels are the sections overlapped by each chunk; top-k chunks can cover more than k
sections. Consequently recall is comparable as coverage, but MRR/nDCG depend on expanded
section order and context budgets differ. This protocol favors chunk coverage; it is not
a controlled claim that one retrieval unit is intrinsically better. Larger experiments
must equalize context tokens, vary chunk/overlap settings and include a section-embedding
ablation. The synthetic texts are too short to test chunking quality meaningfully.

Reports include Recall@k, MRR, nDCG, unanswerable abstention rate, per-query IDs/latency,
index build times, model name, artifact hashes, package versions, corpus hash and platform.
Dense retrieval has no tuned rejection threshold and usually returns candidates even on
unanswerable questions. This exposes a limitation instead of claiming perfect abstention.
No answer generation, paid model routing or answer-quality judge is run. Monetary API cost
is zero for the local paths; CPU, model download and storage costs are not monetized.

For a decision about replacing vector RAG:

1. Collect a licensed corpus and independently annotated queries from actual tasks. Keep
   customer data and permissions outside this public repository.
2. Freeze a test split, include paraphrases, cross-document questions, updates, multilingual
   material and no-answer cases. Tune only on development data.
3. Compare navigation, lexical, dense section search, dense chunk search, hybrid retrieval,
   reranking and optional model routing with matched context budgets.
4. Repeat warm/cold runs and report p50/p95 latency, ingestion/update time, index size,
   model token counts/cost and bootstrap uncertainty intervals.
5. Fix the same answer model/prompt for all retrievers; independently assess answer
   correctness, evidence support, citation validity and useful abstention.
6. Re-run authorization and deletion tests before integrating with a real agent.
