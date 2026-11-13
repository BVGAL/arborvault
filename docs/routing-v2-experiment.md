# Routing v2 execution protocol

Frozen before paid execution on 2026-11-13. Results will be appended after the run; this
document does not yet claim a successful adapter or improvement.

The v2 Python navigation engine (`arborvault.routing_v2`) is document-scoped: callers supply
one authorized document tree. It exposes real subsection headings before paragraph previews,
distinguishes branches from evidence, and avoids an unnecessary model decision on the already
selected document root. Existing v1 API behavior and its historical measurements remain available.
This experiment does not measure document discovery across a corpus or PostgreSQL throughput.

## Fixed data and comparison

Use the [QASPER-derived pilot](../datasets/qasper-routing-v2/README.md): 600 training questions,
120 development and 120 test, with 426/102/96 disjoint documents. Upstream human annotations,
conversion rules, filtering and hashes are public. Training uses 1,400 route decisions from
the train split only. Test examples are not loaded by training or development evaluation.

The pilot is English-only NLP research papers. This replaces the original plan's proposed
new assistant-authored multi-domain questions with existing human annotations. It does not
claim Italian, multi-domain, OCR or official QASPER performance. Its numbers are not directly
comparable to the old eight-document benchmark's 91.67% MiniLM recall.

Every variant uses the same paragraph units, known-document scope, k=5 and 512-word context.
Report mean paragraph Recall@5, MRR, nDCG, full-paragraph evidence coverage, answerable empty
results and unknown abstention. Use the best matching alternative human evidence annotation.
Dense MiniLM and lexical TF-IDF search the same document paragraphs. Dense has no abstention
threshold. Navigation+lexical derives from the same navigation trace, avoiding duplicate calls.
All failed queries remain in the denominator. This is retrieval evaluation, not answer generation.

## Fixed training and selection

Recipe: [`qwen3-routing-v2.json`](../configs/qwen3-routing-v2.json).

This is supervised candidate selection using a frozen quantized base and trainable low-rank
updates, following [LoRA](https://arxiv.org/abs/2106.09685) and the quantized-base approach of
[QLoRA](https://arxiv.org/abs/2305.14314). The run updates 2,949,120 adapter parameters in the
attention query/value projections. Targets are annotated routing IDs plus EOS; the loss is
computed only on those completion tokens. Training does not optimize an answer generator or
a retrieval reward directly. The optimizer is ordinary AdamW, not a paged optimizer.

- Original Qwen/Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`.
- NF4 double quantization/BF16, q/v LoRA rank 8, alpha 16, dropout 0.05.
- One epoch per seed, seeds 42 and 43, AdamW 5e-5, accumulation 8, no weight decay.
- Completion-only loss, including EOS, non-thinking template, 4,096-token context,
  64-token generation cap. No silent sequence truncation.
- Save checkpoints at optimizer steps 88 and 175 for each seed. Development retrieval,
  rather than token loss, chooses a checkpoint from **primary seed 42**.
- An eligible checkpoint has no selector failures, no more answerable empty results and
  no fewer unknown abstentions than the matched development base. Rank eligible candidates
  by recall, evidence coverage and MRR, preferring the earlier step on a tie.
- Seed 43 measures stability at the same selected training fraction and cannot replace the
  primary candidate. Both seeds must clear the guards and beat the development base before
  promotion is considered. A single learning rate isolates the data/representation change and
  keeps two-seed confirmation within the budget; no rate sweep or second epoch is planned.
- Retain epoch zero as the deployment fallback. If no candidate clears the guards or beats
  the base, report the best trained candidate transparently and recommend retaining the base.
  Write `selection-before-test.json` before either test model evaluation.

Final base and adapted test inference share the same loaded NF4 base tensors, original
tokenizer, new routing prompt and output constraints. A canonical JSON-prefix grammar permits
only valid local IDs, at most four unique selections, including an empty array. Both variants
use this same constraint; its benefit cannot be attributed to fine-tuning alone.

Prompts retain all candidate IDs. A common tokenizer-aware fitter reduces excerpt budgets
through 9,000/6,000/3,000/1,000/0 characters and path-component prefixes through
110/80/48/32/16 characters until the input fits. Compression counts and per-call budgets are
recorded. Overlong candidate paths fail explicitly. The same fitting is used during training.
Trace output records source IDs/paths, selections, reached evidence and annotation-based route
coverage, with annotations inspected only **after** each inference.

## Cost and execution

Use one temporary RTX 4090, with the API quote checked immediately before allocation.
On 2026-11-13 the verified quote was $0.89/hour. Set an independent three-hour stop deadline
from allocation (GPU envelope $2.67 plus small storage charges), leaving margin below the
authorized $10 total. The experiment process has its own 9,000-second deadline. Neither is
a substitute for retrieving artifacts, terminating the specific pod and verifying removal.

Prepare dependencies using `requirements.training.txt`, then run in a dedicated CUDA environment:

```sh
python scripts/run_routing_v2.py --output /workspace/experiment-v2
```

The runner saves all four adapters, development/test per-query reports, seed stability,
the locked selection, provenance and checksums. It does not publish to Hugging Face.
The historical v1 dataset and result files are unchanged; this experiment is a new pilot.

## Development-triggered beam ablation (2026-11-13)

This extension was specified while seed 43 was still training, after inspecting development
results only. The frozen greedy run remains unchanged and all of its results will be reported.
The primary seed's half-epoch development recall fell from 60.92% to 31.58%. Its routing calls
usually selected one path (239/264 calls), whereas the base usually selected four (168/264).
This motivates a decoding ablation, not a claim that the training has improved retrieval.

`BeamQwenSelector` uses up to four constrained generation beams: `min(4, candidate_count + 1)`
(two for a single candidate, three for two candidates, four otherwise),
length penalty 1.0 and early stopping. Score-ranked valid JSON hypotheses are combined into
at most four distinct candidate IDs. If the highest-scoring hypothesis is empty, abstain.
No thresholds, weights, prompts, training examples or test questions change. The same policy
is applied to base and adapted models sharing one set of NF4 tensors. Beam inference adds
compute; record actual query latency and keep the original GPU stop deadline.

Run `scripts/benchmark_routing_beams.py --training /workspace/experiment-v2
--output /workspace/beam-v2` after the training process finishes. Evaluate the beam base and
both seed-42 checkpoints on development; select with the original guards, then confirm seed
43 at the same fraction. Save a separate locked selection before evaluating either beam test.
The greedy test may already exist on disk, but neither the beam runner nor its operator reads
its scores until the beam selection is frozen. Prefer the beam base to the greedy base only
if development quality improves and the same abstention/error guards pass. Report both
policies on the test regardless of outcome; never use test performance to pick a checkpoint.

Generation output and sequence scores follow the
[Transformers beam output API](https://huggingface.co/docs/transformers/v5.22.2/en/internal/generation_utils#transformers.generation.GenerateBeamDecoderOnlyOutput).

### Memory incident and execution-only correction

The initial beam run exhausted 24 GB GPU memory during base development, after 92 completed
questions and before adapter development or any beam test. Preserve that incomplete attempt;
it is not a completed benchmark. The retry schedules SDPA attention one beam batch row at a
time, retaining the standard causal/padding mask builder. Numerical parity tests cover prefill,
cached-style decoding and distinct per-beam masks. It changes memory scheduling, not weights,
activation precision, context length, candidate features, beam count or selection rules.
Both base and adapter use the same corrected execution. The retry starts development from
the beginning in a new output directory; the original three-hour cloud stop remains active.
The retry launcher also sets `PYTORCH_ALLOC_CONF=expandable_segments:True` to limit allocation
fragmentation. This affects memory allocation, not tensor precision or routing parameters.
Run the attention scheduler only in the dedicated inference process; its explicit opt-in
replaces that process's SDPA implementation and is not intended for shared training services.
