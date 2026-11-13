# Evaluation protocol

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
