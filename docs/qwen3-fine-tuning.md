# Qwen3 routing fine-tuning experiment

This is a small candidate-routing QLoRA experiment, not a STAIR reproduction or
general-purpose knowledge fine-tuning. The initial recipe is fixed in
[`qwen3-routing-v1.json`](../configs/qwen3-routing-v1.json). The first run was completed
on 2026-11-13 using one Runpod RTX 4090. See [retrieval results and failures](evaluation.md#qwen3-4b-qlora-on-runpod):
the adapter is experimental and does not establish an overall improvement over the base.

## Measured training

Training updated 2,949,120 LoRA parameters in 147 optimizer steps over approximately
511 seconds, excluding environment setup. Base weights stayed frozen. Epoch two was selected
using validation loss; test results played no role in selecting the checkpoint.

| Epoch | Steps completed | Training completion-token NLL | Validation completion-token NLL |
| --- | ---: | ---: | ---: |
| 1 | 49 | 0.225365 | 0.129785 |
| **2 (selected)** | 98 | 0.001985 | **0.111580** |
| 3 | 147 | 0.000046 | 0.126937 |

Very low training loss does not imply retrieval quality: the repeated procedural templates
are easy to fit. All 144 adapter tensors are finite, and all 72 LoRA-B tensors are nonzero.
The adapter config pins the original base revision. Maximum training/validation lengths are
710/711 tokens, below the 2,048-token cap. [Provenance and adapter hashes](benchmarks/qwen3-routing-training.json)
and [runtime versions](benchmarks/qwen3-routing-environment.json) are public; weights have
not been uploaded to Hugging Face and no placeholder model link is advertised.

## Protocol fixed before the run

- Original `Qwen/Qwen3-4B`, immutable revision
  `1cfa9a7208912126459214e8b04321603b3df60c`.
- NF4 double quantization, BF16 computation, q/v LoRA rank 8 and alpha 16.
- 392 training and 84 validation examples from the unchanged routing-bootstrap-v1 bundle.
- Three epochs, batch size 1, accumulation 8, AdamW at a constant 1e-4, seed 42.
- JSON-completion-only causal loss, including EOS. The exact non-thinking inference
  prefix is used during training. Overlong examples are rejected, never truncated.
- Checkpoint selection uses minimum validation completion-token NLL after a full epoch.
  No public test result is used for selection or tuning.
- Same 44 public test questions, k=5 and 512-word returned-context budget as before.
  Dev is diagnostic: its questions partly appear in the training/validation bundle.

The public test set was measured in the earlier Ollama experiment. It is held out from
training gradients, but it is not a newly collected blind test set. Draft labels, a small
corpus and repeated synthetic templates limit conclusions. No generated answers are scored.

## Prepared environment

Use one CUDA GPU with BF16 support and preferably at least 24 GB VRAM. The optional
`Dockerfile.training` and `requirements.training.txt` keep training dependencies outside
the API image. Model weights and adapters are ignored by Git. Record the actual resolved
packages and image digest when executing. The recorded environment passed dependency checks;
the recipe remains experimental because of its small dataset and observed retrieval regression.

```sh
python scripts/train_qwen3.py --output reports/qwen3-check --validate-only
pip install -c constraints.txt '.[dev,eval]'
pip install -r requirements.training.txt
pip check
```

Provide `TEST_DATABASE_URL` for a disposable PostgreSQL administrator. The benchmark
creates a random database and restricted runtime role, enforces RLS, then removes both.
Never use a production database. Set `ARBORVAULT_CODE_REVISION` to the executed Git commit.

```sh
bash scripts/run_qwen3_experiment.sh /workspace/experiments/qwen3-routing-v1
```

The runner saves the selected adapter, tokenizer, per-epoch losses, training provenance,
full per-query base/adapted reports, comparison reports and artifact checksums. It never
uploads to Hugging Face. Retain source licenses if sharing a training bundle.

## Paired comparison

Base and fine-tuned inference run on the same NF4 model tensors in the same process,
with the LoRA adapter disabled or enabled. Both use the original tokenizer/template,
non-thinking greedy decoding, 8,192-token context and 256 output tokens. No merge or
re-quantization is involved. MiniLM, lexical and deterministic navigation baselines use
the same extracted sections. The earlier Ollama Q4_K_M results remain historical;
they are not the controlled base for this NF4 experiment.

## Runpod operation

Prepare code and dataset before renting a GPU. Confirm the current hourly quote, available
account balance, disk allocation and total authorized budget. Use one new, uniquely named
temporary pod and a dedicated SSH key. No account-wide Runpod or Hugging Face credential
is needed inside the pod for this public model. Do not attach production volumes.

Set a control-plane stop deadline independent of the training process; a training timeout
alone does not stop GPU billing. Monitor real progress and cost. Retrieve the adapter and
reports, verify every artifact checksum, then terminate that exact pod and verify removal.
Stopped pods can still incur storage charges. Never bulk-delete pods or volumes.
