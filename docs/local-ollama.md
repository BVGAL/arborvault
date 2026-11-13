# Local Qwen3-4B evaluation

Ollama can serve the routing model locally without a Cloud subscription or API key.
The benchmark uses the original [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B),
quantized to Q4_K_M, with **no ArborVault fine-tuning**. Original weights remain under
Apache-2.0. This tests retrieval, not generated answer quality or a full Qdrant deployment.

## Install and identify the model

Install [Ollama for Windows](https://docs.ollama.com/windows), or use the official installer
for your platform, then open a fresh terminal:

```sh
ollama pull qwen3:4b-q4_K_M
ollama show qwen3:4b-q4_K_M
```

Use the explicit tag. On 2026-11-13 the generic `qwen3:4b` tag resolved to **Thinking-2507**,
which supports thinking only; it is a different checkpoint. Installation preflight caught
that difference before any benchmark questions were evaluated. The final run used the
original model, its stock template and these artifact identifiers:

- Ollama version: `0.40.16`.
- Model manifest digest: `2bfd38a7daaf4b1037efe517ccb73d1a3bbd4822cf89f1a82be1569050a114e0`.
- Downloaded model size reported by Ollama: 2,620,788,260 bytes.
- Quantization: `Q4_K_M`; parameter count: 4,022,468,096.
- Template SHA-256: `ae370d884f108d16e7cc8fd5259ebc5773a0afa6e078b11f4ed7e39a27e0dfc4`.

Tags can change. Compare the digest in `/api/tags` with the report before claiming an exact
rerun. The Ollama artifact identifies the served weights/template, but does not attest an
exact upstream Hugging Face commit. That field remains `unknown`, rather than inventing one.
Pin the upstream checkpoint and rerun the base under matched conditions when preparing the
future fine-tuned comparison.

## Connect the API

For a Python API running on the same host:

```sh
python scripts/configure_ollama.py --local --model qwen3:4b-q4_K_M
```

For the supplied Docker Desktop stack using host Ollama:

```sh
python scripts/configure_ollama.py --local --docker-client --model qwen3:4b-q4_K_M
docker compose up --build -d --wait
```

These commands save ignored local configuration, use no provider key, and set thinking
off, context 8,192 and maximum output 256. Docker Desktop accesses the host through
`host.docker.internal`; verify connectivity on your platform. Keep the unauthenticated
Ollama endpoint restricted to your machine/trusted container network.

## Reproduce the retrieval benchmark

The standalone runner creates a random disposable database and restricted role on the
test PostgreSQL service, then removes both. It does not use the application's database.
It verifies supported thinking controls, performs one synthetic JSON warm-up, checks the
applied context size, and records the model digest before and after the run.

PowerShell, from the repository root:

```powershell
docker compose -f compose.test.yaml build tests
docker compose -f compose.test.yaml up -d --wait db
docker compose -f compose.test.yaml run --rm `
  -v "${PWD}:/workspace" -w /workspace `
  -e PYTHONPATH=/workspace/src `
  -e "ARBORVAULT_CODE_REVISION=$(git rev-parse HEAD)" `
  tests python scripts/benchmark_local_ollama.py `
  --split dev --cache-dir /workspace/.cache/models `
  --output /workspace/reports/qwen3-4b-dev.json
```

After fixing the configuration, repeat with `--split test` and a distinct output path.
Use `docker compose -f compose.test.yaml down` to remove the test containers. Model downloads,
local reports and credentials are excluded from Git. A Linux/native runner can use
`--url http://127.0.0.1:11434/api` with `TEST_DATABASE_URL` pointing to a disposable test server.

Fixed evaluation settings:

| Setting | Value |
| --- | --- |
| Dataset | Public Docs v1, 8 documents, 85 content sections |
| Development/test | 16 / 44 questions, grouped by question family |
| Returned context | k=5, at most 512 whitespace words |
| Navigation | beam 4, depth 12, candidate window 160 |
| Model decoding | temperature 0, seed 0, thinking false |
| Model budgets | context 8,192; output 256; at most 1,056 calls |
| Vector baseline | all-MiniLM-L6-v2, identical sections, exact cosine search |

Greedy decoding was fixed for this routing experiment; it is not a claim of Qwen's best
general-purpose sampling settings. Reports include every query, provider usage, failures,
prompt/template hashes, model digest and code revision. The runner refuses a thinking-only
variant and marks possible context truncation as failure. It does not automatically retry
or silently replace malformed model responses with successful LLM results.

The model ran on an 8 GB NVIDIA GPU; Ollama reported all 4,001,072,414 loaded bytes in VRAM.
PostgreSQL and MiniLM ran in Linux containers on CPU. Timings are a single sequential pass
with a warm-up and uncleared prefix caches, so the faster second LLM method can benefit from
cache reuse. They are diagnostic, not a controlled hardware/performance comparison.

See [the measured results](evaluation.md#local-qwen3-4b-baseline) and the raw reports linked
there. Fine-tuning, independent annotation review, unseen corpora and answer generation
evaluation remain separate work.
