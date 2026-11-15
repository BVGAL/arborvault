# Support v3: adapting the reader while guarding temporal retention

Preregistered before fitting or scoring the new test. The target is better
abstention on negation and reversed roles, without sacrificing the temporal
progress in v2. This is a paragraph-support experiment, not a new end-to-end
retrieval, answer-generation or Qwen benchmark. No production default changes
are implied by this experiment. There is no IBM affiliation or verified STAIR
replication claim.

## Data and boundaries

See the [dataset card](../datasets/support-v3/README.md). There are 9,566 fit
examples, 1,632 selection examples, 1,120 calibration examples, and 2,688 test
examples. Fit replays exactly the v2 fit rows and adds 1,664 new controlled rows.
Human selection remains the established 768 examples. No previous calibration
or test rows enter fitting. New human calibration/test use 8/24 articles absent
from **every** support-v1 and support-v2 role. Full question/paragraph input must
fit 512 tokens; never truncate. Test contains 768 human questions and 1,920
controlled examples from **15 families, five per phenomenon**.

New contrast families include permission/absence of permission, exceptions to
prohibitions, active/passive transfers, borrowing and buying, and event years
versus reference-document years. Entities use readable fictional names plus
unique suffixes. The same question and same context each occur with both labels.
Only question and context enter inference. Metadata/oracles/spans are excluded.
The author has seen the templates to check labels, but no model outcomes when
freezing them. This is not an independently annotated natural-traffic benchmark.

## Trainable reader

Base model: [MS MARCO MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2),
revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`.
Use [LoRA](https://arxiv.org/abs/2106.09685) rank 8, alpha 16 on query, key, value,
attention output, intermediate and feed-forward output linear matrices in all
six blocks. Base encoder matrices, embeddings and layer norms stay frozen;
pooler and output classifier also train. LoRA B matrices start at zero, A uses
Kaiming uniform. Verify an exact frozen-parameter hash before and after training.
Keep native training dropout. No hand-written semantic parser is part of scoring.

One fixed seed 42, CPU float32 eager attention, four threads, batch 16. AdamW
LR 3e-4, weight decay .01, 90-step linear warmup then cosine decay to zero at
step 1,800, gradient clipping 1.0. No hyperparameter search or additional seeds.
Cycle equally through human, negation, relation, temporal strata. Sample complete
balanced groups without replacement **within each batch**, with replacement
between steps: eight human paragraph pairs or four contrast quartets per batch.
Each stratum has 450 updates / 7,200 row exposures. These are repeated draws,
not 28,800 distinct examples or full dataset epochs.

Objective is binary cross entropy plus 0.5 times a counterfactual ranking loss.
For controlled examples only, match the **same question** against supported and
unsupported contexts and minimize `softplus(1 - positive_logit + negative_logit)`.
No test or calibration labels enter optimization. Temporal replay/gradients
remain present throughout training; preservation is measured, not assumed.

Save checkpoints at steps 300, 600, 1,200 and 1,800. Select solely on selection
data using normalized partial ROC area for unsupported rejection while useful
rejection ranges from 0 to 5%. Require temporal partial area within .02 of the
v2 head and human partial area within .02 of v1. Among eligible checkpoints,
maximize the **minimum of negation and relation** partial areas, then the mean
of all four strata, then prefer the earlier checkpoint. If none pass retention,
report the best diagnostic candidate explicitly as ineligible for promotion.

## Comparators and gates

Preserve v1/v2 coefficients. Evaluate original relevance logits and a frozen
representation v3 head (32-unit tanh MLP, epochs 25/50/100, original v2 optimizer
and row-weighted objective). Apply the same selection guard to its checkpoints.
That head is a comparator, **not** a causal isolation of encoder adaptation:
sampling, loss and architecture differ from the adapter. All baselines read
the same complete inputs. Round frozen CLS vectors to seven decimals as in v2.

Calibrate one common threshold per method on calibration only: at most 2% of
useful examples rejected **within each stratum**, then maximize equally weighted
unsupported rejection; ties minimize useful loss then threshold. Secondary
balanced threshold maximizes equal-stratum balanced accuracy. Also report v2
with its unchanged original gates, separately from fair recalibration.
Write selected weights and all thresholds before computing test predictions.

Primary comparison: selected adapter versus recalibrated v2, with v1 as the
human-transfer reference. Report both error rates with denominators, AUROC,
counterfactual ordering, per-family metrics and all declared comparators.
Paired bootstrap: 5,000 replicates, seed 42, articles for human and template
families for controlled cases. Five families per phenomenon imply weak power
for generalization claims. Question-only/context-only input ablations must
remain chance on the exactly balanced controlled test.

The declared observed promotion gate requires all of:

- Development retention guards passed.
- Negation and relation each improve unsupported rejection by at least 10
  percentage points over v2 on the new test.
- Temporal unsupported rejection is at most 5 points below v2.
- Human unsupported rejection is at most 2 points below v1.
- Useful rejection stays at or below 2% in **each** of the four test strata.

Passing is an observed research result, not a confidence-bound guarantee or a
production certification. Failing keeps the prior reference and publishes the
negative outcome. Do not repair thresholds or choose another checkpoint after
test results. Previously observed v2 failures inform the design but are not
claimed as new held-out evidence.

## Reproduction, cost and scope

Use the pinned `Dockerfile.probe` / `requirements.probe-cpu.txt` environment.
Run `scripts/build_support_v3.py`, `scripts/run_support_v3.py train`, then
`scripts/run_support_v3.py evaluate`; each refuses to overwrite completed output.
The source file is the original public SQuAD `train-v2.0.json`, verified by hash.
See the artifact README for commands and independent re-inference verification.
All learned trainable tensors are published as plain JSON; downloaded base
weights remain upstream. No pickle execution, remote model code or paid API.

Maximum local runtime per command: 7,200 seconds. Additional cloud spend: **0**.
Prior cumulative estimate: **$7.94877**, total cap **$10**. Local electricity and
hardware amortization are unmeasured. Qwen is not being trained.
