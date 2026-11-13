# Base model, fine-tuning and reproducible evaluation

The first measured local baseline uses [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
with the explicit Ollama tag `qwen3:4b-q4_K_M`; see [local setup and results](local-ollama.md).
The first Qwen3 routing adapter has now been trained and evaluated on Runpod; use the
[measured QLoRA recipe](qwen3-fine-tuning.md) and [matched results](evaluation.md#qwen3-4b-qlora-on-runpod).
It does not demonstrate an overall improvement, and its weights are not hosted on Hugging Face.
Pin the exact Hugging Face base commit before training.
An alternative cloud family is [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b),
also open-weight under Apache-2.0. The native cloud API uses `gpt-oss:20b`; check the current
[Ollama catalog](https://ollama.com/library/gpt-oss:20b-cloud) before running.

This is candidate-routing fine-tuning. It does not reproduce STAIR's corpus knowledge in
model parameters, its training procedure or its published scores.

## 1. Establish the unmodified baseline

Install ArborVault and optionally `.[eval]` for MiniLM. Provide `EVAL_DATABASE_URL` through
your secret manager/environment: it must identify a migrated, disposable PostgreSQL database
using the restricted runtime role. The runner inserts and removes a random evaluation tenant.
Do not point it at production. Then configure provider access locally:

For a subscription-free local run, follow [the Qwen3-4B guide](local-ollama.md).
For optional cloud evaluation:

```sh
python scripts/configure_ollama.py --model gpt-oss:20b
python -m arborvault.llm_eval run datasets/public-docs-v1 --variant base --split dev \
  --max-calls 192 --max-output-tokens 1024 --dense --output reports/base-dev.json
```

Use the frozen test split only after fixing prompts, hyperparameters and selection rules.
For 44 test queries, two LLM methods and depth 12, the worst-case request cap is 1,056.
Calls are sequential and have no automatic retries. Reported token usage comes from the
provider, including failed parsing of a completed response. `--max-calls` and the output
cap bound requests; **they are not a currency budget**. Review current pricing and set
account spending controls before execution. The server adapter has a process-lifetime
`SELECTOR_MAX_CALLS` cap, reset on restart; it is not a distributed billing limit.

Each report compares deterministic navigation, automatic fallback, PostgreSQL lexical
retrieval, optional MiniLM dense section retrieval, LLM navigation and LLM automatic fallback.
All use the same source sections, queries, k=5 and 512-word returned-context budget.
Metrics include recall, MRR, nDCG, evidence coverage, abstention, latency and provider usage.
Failures/budget exhaustion mark the LLM run incomplete, even if lexical fallback returns hits.
This evaluates retrieval, not generated answer quality.

Ollama Cloud currently lacks structured output constraints. The adapter requests a JSON
array in the prompt, checks local labels and rejects malformed, duplicate, foreign or
truncated selections. Authorized query/candidate excerpts leave the server. Provider URLs
are administrator configuration; redirects are not followed with credentials.

## 2. Validate and train personally

For Qwen3-4B, use `scripts/train_qwen3.py` and `scripts/benchmark_qwen3.py` as documented
in the [Qwen3 guide](qwen3-fine-tuning.md). The older `train_lora.py` example below remains
a separate optional recipe, principally for gpt-oss; it is not the measured Qwen3 QLoRA run.

The checked-in [bootstrap dataset](../datasets/routing-bootstrap-v1/README.md) contains
procedural examples plus oracle routes from the public **development** questions. The
44 public test questions and their relevance labels are excluded from training exports.
Public corpus documents may appear as candidates, as expected for retrieval over a known
corpus. This is not a held-out-document benchmark. Review and expand training labels first.

```sh
python -m arborvault.training_data datasets/public-docs-v1 --output reports/routing-data
python scripts/train_lora.py reports/routing-data --validate-only
```

The training script uses attention-only q/v LoRA, completion-only loss, deterministic seeds,
the base model's tokenizer/chat template, and immutable base revision. It refuses mismatched
hashes, overlapping question families, test-label leakage, output overwrite and silently
truncated examples. It saves the adapter and provenance locally; it never uploads to the Hub.

Install training packages in a separate Linux GPU environment:

```sh
pip install -e .
pip install 'torch>=2.8,<3' 'transformers>=4.55,<6' 'peft>=0.17,<1' \
  'trl>=0.20,<2' 'datasets>=4,<6' 'accelerate>=1.10,<2'
python scripts/train_lora.py datasets/routing-bootstrap-v1 \
  --revision YOUR_IMMUTABLE_BASE_COMMIT --output models/routing-adapter
```

These ranges are a starting recipe, not a tested GPU lockfile. Record resolved packages
and freeze your successful environment. The gpt-oss path dequantizes the base to BF16;
budget for a large GPU. The [official training example](https://developers.openai.com/cookbook/articles/gpt-oss/fine-tune-transfomers)
uses an 80 GB H100; inference memory requirements do not describe this training recipe.
Only data preparation/validation has been executed for this older gpt-oss recipe; a full
20B training run has not been performed. No hardware/time/quality guarantee is made.

## 3. Compare the same base and adapted model

Serving an adapter uploaded to Hugging Face is separate from uploading it. Ollama Cloud's
managed catalog does not automatically run arbitrary Hub adapters. Merge/export in a
compatible GPU environment and verify the chosen runtime's support for that architecture,
or serve both models yourself. Consult [Ollama import documentation](https://docs.ollama.com/import).
The GPT-OSS merge/GGUF conversion path is not validated by this repository.

For a controlled comparison, **rerun the original base alongside the adapted model on the
same runtime**, with identical precision, tokenizer, template, decoding, corpus, split and
context budget. A cloud baseline followed by local fine-tuned inference is exploratory
because provider revision, quantization and serving behavior may differ.

Run both variants using the same `--base-model`, immutable `--base-revision`, `--runtime-id`
and `--quantization`; set `--model` to the appropriate served checkpoint name. Keep all
settings fixed and publish artifact hashes and training provenance. Then:

```sh
python -m arborvault.llm_eval compare reports/base-test.json reports/tuned-test.json \
  --output reports/comparison.json
```

The comparison rejects a different base family/revision, prompt, query set, split or budget.
Unknown/different runtimes, cloud checkpoint provenance and incomplete calls require
`--allow-exploratory`, which labels the output accordingly. Runtime metadata is an operator
declaration, not cryptographic proof. The small draft test set does not establish statistical
significance; independent review and additional unseen corpora remain required.

## 4. Publish the trained model

After personally completing training and evaluation, create a Hugging Face model repository.
Upload the adapter/tokenizer, `training-provenance.json`, license notices and an honest model
card with exact base revision, training data hashes, measured results and limitations.
Use the local `hf auth login` flow; never place tokens in files intended for publication.
Retain the upstream MIT/BSD notices for dataset excerpts. Do not describe the model as IBM
official or a verified STAIR reproduction.

Only after the Hub repository exists, add its real URL and immutable revision to this
project's documentation with the matching comparison report. No placeholder model link
is advertised as a trained artifact.

API references: [Ollama Cloud](https://docs.ollama.com/cloud),
[native chat](https://docs.ollama.com/api/chat),
[TRL SFT](https://huggingface.co/docs/trl/sft_trainer),
[PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora).
