# Evaluation protocol

## Qwen3-4B QLoRA on Runpod

Measured on 2026-11-13 using a single RTX 4090. Base and adapted inference share the same
NF4 base tensors, tokenizer, prompt, decoding and corpus. The adapter is enabled or disabled
in the same process. This **rerun base**, rather than the earlier Ollama Q4_K_M artifact, is
the relevant comparison. The frozen public test contains 36 answerable and 8 unanswerable
questions, with k=5 and a 512-word context budget. Dataset and MiniLM weight hashes are
unchanged from the previous experiment.

| Method | Test Recall@5 | MRR | Evidence coverage | Unknown abstention | Answerable empty results |
| --- | ---: | ---: | ---: | ---: | ---: |
| Local navigation | 0.5833 | 0.5162 | 0.5833 | 1/8 | 0/36 |
| PostgreSQL lexical | 0.4444 | 0.4167 | 0.4722 | 1/8 | 0/36 |
| MiniLM dense sections | **0.9167** | **0.7000** | **0.9444** | 0/8 | 0/36 |
| Qwen3 NF4 base navigation | 0.6111 | 0.5579 | 0.6111 | **8/8** | 4/36 |
| Qwen3 NF4 base + lexical | 0.6667 | 0.6134 | 0.6667 | 1/8 | 0/36 |
| Qwen3 QLoRA navigation | 0.4722 | 0.4167 | 0.4722 | **8/8** | 12/36 |
| Qwen3 QLoRA + lexical | 0.6944 | 0.5856 | 0.6944 | 1/8 | 0/36 |

**This adapter is not a demonstrated retrieval upgrade.** Pure navigation loses 13.89
percentage points of recall and returns no evidence for more answerable questions. With
fallback, recall rises by 2.78 points (one question-equivalent out of 36), but MRR and nDCG
fall. MiniLM remains substantially stronger at finding labeled evidence. Both model variants
reject all eight unknown questions in pure navigation; fallback fills seven with lexical hits.
Dense retrieval has no rejection threshold in this benchmark, so its 0/8 does not establish
that dense systems cannot abstain.

**Failures are included, not discarded.** Each model makes two invalid selections: the same
question once in `llm_navigation` and once in `llm_auto`. Base fails on `fastapi-errors-04`,
the adapter on `httpx-clients-04`. The recorded failure stage is `selection_validation`
with `ValueError`, after valid termination with token 151645. These are selection-contract
failures, not context truncation or transport errors. Navigation stops on the failed call and
retains evidence found in earlier steps; auto can add lexical fallback. In these two cases,
both models retain some evidence; the base still finds the labeled section and the adapter
does not. All 44 questions were attempted
in every method, but the reports deliberately retain `complete: false` for test and the
comparison is labeled **exploratory**. The 4/12 answerable empty results are valid no-route
decisions, separate from these selection failures. See the [post-hoc error analysis](qwen3-error-analysis.md).

The final evaluation uses 528 measured model calls: 64 base-dev, 56 adapted-dev, 222
base-test and 186 adapted-test. There are four failed calls, all on test. One synthetic warm-up
per variant/split is excluded. No generated answers are scored. Development recall is 0.9167
for base navigation and 0.8333 for adapted navigation, and 0.9167 for both with fallback;
dev is diagnostic because its questions partly appear in training/validation.

Raw final reports: [base dev](benchmarks/qwen3-nf4-base-dev.json),
[adapted dev](benchmarks/qwen3-nf4-tuned-dev.json),
[base test](benchmarks/qwen3-nf4-base-test.json),
[adapted test](benchmarks/qwen3-nf4-tuned-test.json),
[paired test comparison](benchmarks/qwen3-nf4-comparison-test.json).
The [training record](benchmarks/qwen3-routing-training.json),
[runtime inventory](benchmarks/qwen3-routing-environment.json),
[base artifact hashes](benchmarks/qwen3-routing-base-artifacts.json) and
[adapter checks](benchmarks/qwen3-routing-adapter-checks.json) support reproduction.

An initial evaluation was repeated after correcting the runner to accept both EOS tokens
declared by Qwen3, and adding failure-stage diagnostics. The recorded errors still occur in
selection validation; retrieval scores, abstention and call counts are identical. No weights,
labels, prompts, rankings or decoding settings were adjusted. Initial reports are retained:
[base dev](benchmarks/qwen3-initial-base-dev.json),
[adapted dev](benchmarks/qwen3-initial-tuned-dev.json),
[base test](benchmarks/qwen3-initial-base-test.json),
[adapted test](benchmarks/qwen3-initial-tuned-test.json).
Both passes together used 1,056 measured calls and eight excluded warm-ups.

Training code: `4494b3a6f770fc62c67c050d4e9572d812c95c2e`; corrected evaluation code is
recorded in [evaluation provenance](benchmarks/qwen3-routing-evaluation-provenance.json).
See the [recipe and reproduction guide](qwen3-fine-tuning.md). Training uses 392 examples
and 84 validation examples; epoch two is selected from three by validation loss only.
No test questions/qrels enter training. However, this small public test was measured before
training, labels remain assistant-authored drafts, and synthetic training templates repeat.
It is not a new blind or independently reviewed benchmark. Latency is exploratory: GPU
routing, CPU dense ranking, different database work, one sequential pass and fixed order.

## Local Qwen3-4B baseline

Measured on 2026-11-13, using original Qwen3-4B weights through Ollama **without fine-tuning**.
Every method uses the same frozen corpus, labels, section units, k=5 and 512-word context
budget. Development contains 16 questions; test contains 36 answerable and 8 unanswerable
questions. Code/settings were fixed at `52931bd9ad27767969caea59dd5d0cb5c62df4b5` before
development and kept unchanged for test. No prompt, threshold or ranking change followed
inspection of test results.

| Method | Dev Recall@5 | Test Recall@5 | Test MRR | Test evidence coverage | No-answer abstention | Test p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Local navigation | 0.8333 | 0.5833 | 0.5162 | 0.5833 | 1/8 | 35.1 |
| Local navigation + lexical | 0.8333 | 0.5833 | 0.5162 | 0.5833 | 1/8 | 33.6 |
| PostgreSQL lexical | 0.4167 | 0.4444 | 0.4167 | 0.4722 | 1/8 | 41.2 |
| MiniLM dense sections | 0.9167 | **0.9167** | **0.7000** | **0.9444** | 0/8 | 4.0 |
| Qwen3 navigation | 0.9167 | 0.5278 | 0.4745 | 0.5556 | **8/8** | 665.7 |
| Qwen3 navigation + lexical | 0.9167 | 0.6389 | 0.5856 | 0.6667 | 1/8 | 694.0 |

**Dense retrieval remains stronger on labeled evidence.** Qwen with lexical fallback raises
test Recall@5 from 0.5833 to 0.6389, but remains below MiniLM's 0.9167. Pure Qwen navigation
performs worse than the local selector on test despite matching dense recall on development.
This small development split did not predict the test result.

Abstention has a cost: Qwen navigation rejects all 8 unanswerable questions, but also returns
no context for **9 of 36 answerable questions**. Automatic fallback runs for 34 of 44 questions
and fills 7 of the 8 unanswerable cases with lexical hits, leaving only 1/8 correct abstentions.
The dense baseline has no rejection threshold and always returns top-k context; its 0/8 is
not evidence that dense systems cannot implement abstention. Develop any improved rejection
policy on new development data and evaluate it on new held-out questions.

Raw [development](benchmarks/public-docs-qwen3-4b-dev.json) and
[test](benchmarks/public-docs-qwen3-4b-test.json) reports include all methods and all queries.
Test used **206 model calls, zero provider/JSON failures**, 136,004 input tokens and 1,284
output tokens. Development used 64 calls, 38,682 input tokens and 346 output tokens, also
without failures. One synthetic warm-up per run is excluded from those counts. The largest
test prompt was 2,113 tokens, below the 8,192-token context. Artifact hashes stayed unchanged.

[Reproduction instructions](local-ollama.md) specify Ollama 0.40.16, explicit tag
`qwen3:4b-q4_K_M`, Q4_K_M, thinking off, temperature 0, seed 0 and output cap 256. The generic
`qwen3:4b` tag pointed to Thinking-2507 and was rejected during installation preflight.
No custom template or weight change was used in the benchmark. Ollama does not attest the
exact upstream Hugging Face commit; the manifest digest identifies the served artifact.

Timing includes PostgreSQL and HTTP/model inference for Qwen, while MiniLM ranks an in-memory
index on CPU. Qwen runs on an 8 GB GPU. Fixed order, prefix caches, one warm-up and one pass
prevent a controlled latency comparison. These are **retrieval-stage results**, not generated
answer scores or a Qdrant/PostgreSQL deployment comparison. Labels are assistant-authored
drafts and the corpus is tiny and public; general or statistical superiority is not established.
The fine-tuned model is measured separately above under a matched NF4 runtime.

## v0.3 regression and LLM workflow

The v0.3 rerun uses the same frozen corpus, annotations, MiniLM weights, k=5 and 512-word
budget. Test Recall@5 remains 0.5833 for local navigation/auto, 0.4444 for lexical and
0.9167 for dense sections. Raw [dev](benchmarks/public-docs-v03-dev.json) and
[test](benchmarks/public-docs-v03-test.json) reports retain all per-query outcomes.
The dataset SHA-256 is unchanged. Timings remain exploratory single-pass measurements;
ACL checks and lazy SQL traversal alter database work and are not a scalability proof.

The [LLM workflow](llm-evaluation.md) adds native Ollama routing, base/adapted comparisons
and separately split [training data](../datasets/routing-bootstrap-v1/README.md). The release
initially validated the protocol with mocked responses; the measurement above adds real local
inference on main. The subsequent Qwen3 experiment above evaluates the original base and
its adapter under matched serving conditions; independent label review remains outstanding.

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
