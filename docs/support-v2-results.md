# Semantic abstention v2: temporal gains, failed generalization elsewhere

**The v2 classifier is not promoted.** It rejects substantially more incorrect
temporal references in controlled records, but fails the broader abstention goal.
On new human questions it rejects **33/512 unsupported paragraphs**, versus
**48/512** for the previous support-v1 head. On new negation wording it rejects
**50/192 useful paragraphs (26.04%)**, despite satisfying the 2% constraint in
every calibration stratum. Reversed sender/recipient relations remain near chance.

The selected model is a **32-unit tanh head at epoch 100**, with 12,353 trainable
coefficients over the frozen MiniLM representation. It was selected on development
data, before test scores. A separately reported linear head isolates adding data
without adding nonlinear capacity. No Qwen or encoder weights were trained, no
test-driven threshold repair was applied, and no production default was replaced.

## Fixed comparison

The [preregistered protocol](support-v2-protocol.md) and
[dataset](../datasets/support-v2/README.md) were published at
[`3a725b4`](https://github.com/BVGAL/arborvault/commit/3a725b49651cd7b41d37cd34da0bdc8195f02ea1)
before feature extraction and fitting. The prior test was not reused for training.

| Role | Human examples | Controlled examples | Total |
|---|---:|---:|---:|
| Fit | 4,062 | 3,840 | 7,902 |
| Select model/checkpoint | 768 | 768 | 1,536 |
| Calibrate gates | 768 | 768 | 1,536 |
| New test | 1,024 | 1,152 | 2,176 |

The new human test has **32 articles absent from all support-v1 roles**; another
24 new articles calibrate thresholds. The human source is the original public
SQuAD 2.0 training split, with original questions, full paragraphs and labels.
It is held out from this experiment, not necessarily absent from pretraining.

Controlled cases cross two questions with two contexts, balancing labels for
each identical question and each identical context. Both unit names stay in
reversed-relation examples, and both event/reference years stay in temporal
examples. No text is truncated. Test templates and entities are absent from
training roles, but the templates were authored and reviewed by the same
AI-assisted author before outcomes. They are not independently adjudicated
natural traffic. Only **nine test template families**, three per phenomenon,
support the 1,152 controlled instances.

All methods use one common threshold, without a category or template label at
inference. Previous coefficients are frozen, but thresholds are recalibrated
fairly on the same new calibration data. The conservative gate permits at most
2% useful rejection in **each** of human, negation, relation and temporal
calibration strata. The balanced gate is secondary and maximizes their equally
weighted balanced accuracy. Neither condition guarantees the same test behavior.

## Primary result: conservative abstention

Unsupported rejection should be **higher**, while useful rejection should be
**lower**. These two columns must be read together.

| Test group | Previous: unsupported rejected | V2 selected: unsupported rejected | Previous: useful rejected | V2 selected: useful rejected |
|---|---:|---:|---:|---:|
| Human questions | 48/512 (9.38%) | **33/512 (6.45%)** | 9/512 (1.76%) | 6/512 (1.17%) |
| Negation | 1/192 (0.52%) | 97/192 (50.52%) | 0/192 | **50/192 (26.04%)** |
| Reversed relations | 0/192 | **0/192** | 0/192 | 0/192 |
| Temporal references | 22/192 (11.46%) | **126/192 (65.62%)** | 0/192 | **0/192** |
| All controlled cases | 23/576 (3.99%) | 223/576 (38.72%) | 0/576 | 50/576 (8.68%) |

On human questions, the new head rejects **15 fewer unsupported cases**, while
rejecting three fewer useful cases. The unsupported-rejection difference is
**−2.93 percentage points**, paired article-bootstrap 95% interval
**[−5.86, −0.39]**. The useful-rejection difference is **−0.59 points**, interval
**[−1.76, +0.59]**. This is an operating-point tradeoff, not a comparison at
identical realized test error rates; the primary goal of increased human
unsupported rejection was not achieved.

The controlled aggregate gains **34.72 points** in unsupported rejection
(exploratory family-bootstrap interval **[+13.37, +58.85]**), but adds **8.68 points**
of useful rejection (interval **[0, +25.00]**). Resampling only nine families can
omit the failing family entirely; the interval touching zero does not erase the
observed 50 useful rejections. The negation failure is far above the nominal 2%
calibration target and prevents a robust-abstention claim.

## Architecture ablation and baselines

Each cell below is **unsupported rejected / useful rejected**, at the conservative
gate. Human denominators are 512 per class; controlled denominators are 576.

| Method | Human test | Controlled test | Human AUROC | Controlled AUROC |
|---|---:|---:|---:|---:|
| Selected nonlinear head | 33 / 6 | 223 / 50 | 0.7331 | 0.7821 |
| Retrained linear head, L2=0.1 | 23 / 4 | 137 / 4 | 0.7193 | 0.7500 |
| Previous support-v1 head | 48 / 9 | 23 / 0 | 0.7427 | 0.7000 |
| Original relevance logit | 17 / 6 | 0 / 0 | 0.7085 | 0.6240 |
| Matched question-only head | 31 / 27 | 0 / 0 | 0.5567 | 0.5000 |
| Constant context-only control | 0 / 0 | 0 / 0 | 0.5000 | 0.5000 |

The linear data-only ablation is less brittle than the selected nonlinear head
on controlled negations: **73/192 unsupported rejections and 4/192 useful
rejections**, versus 97 and 50. It rejects 64/192 temporal negatives without
useful losses, and still rejects no reversed-relation negatives. It also rejects
fewer human negatives than the previous head. This is a prespecified ablation,
**not a new winner chosen after inspecting test performance**.

Question-only scoring is exactly chance on controlled cases, as intended by
the crossed construction. It still shows question-style effects on the human
sample. A constant context-only classifier is uninformative because every
context is paired with both labels. These controls do not rule out all other
template or identifier shortcuts.

## Why the selected head failed

The following are **post-hoc diagnostics**, with no retraining, relabeling or
threshold changes. Full data are in
[diagnostics.json](benchmarks/support-v2/diagnostics.json), reproducible with
`python scripts/diagnose_support_v2.py`.

**Negation: score-scale transfer is unstable.** Calibration negations have
AUROC 1.0000 and zero useful rejections under the selected conservative gate.
New test negations have AUROC 0.7098 and 50 useful rejections. **47 of these 50
come from the “clearance / no clearance” wording family**, which loses 47/64
useful cases. Explicit statements granting clearance and explicit statements
denying clearance are both incorrectly rejected when they answer their matching
questions. Their scores fall below the common threshold of 0.12428.

Holding the question fixed and comparing its two contexts, the new head orders
the supported one above the unsupported one in **179/192 negation comparisons
(93.23%)**, versus **181/192 (94.27%)** for the previous head. Thus the larger
head did not improve this within-question ordering. Relative ranking can be
good while absolute scores shift enough across wording families to break a
shared gate. This is evidence of a calibration/generalization problem, not
proof that simply moving the threshold on this observed test would fix deployment.

**Reversed relations: little usable separation is learned.** The selected head's
relation AUROC is **0.5882 on fit**, **0.5044 on selection**, **0.4549 on calibration**
and **0.4891 on test**. Same-question counterfactual ordering on test is exactly
**50%**. The problem is visible before test, rather than being only a bad final
threshold. Both heads fail on these unfamiliar unit identifiers and direction
swaps. This does not prove the frozen encoder contains no relational information
or that another training objective cannot recover it; it shows that this tested
representation/head/data combination did not generalize.

**Temporal binding: stronger margins make more cases rejectable.** Temporal test
AUROC rises from **0.9118 to 0.9945** and conservative wrong-year rejection rises
from 22/192 to 126/192 without useful losses. However, **the selected, previous
and linear heads, and raw relevance, already order the two contexts correctly
for every temporal question**. The
new head improves margins and score comparability for the common threshold;
it has not demonstrated a newly acquired temporal-reasoning capability.

The development selection criterion averaged low-error rejection across four
strata. The chosen epoch-100 head scored **0.33036**, versus **0.30037** for the
best linear head. Strong temporal performance can compensate for an unsolved
relation stratum under this criterion. The full candidate schedule is preserved
in [fitted-models.json](benchmarks/support-v2/results/fitted-models.json).

An [AI-assisted illustrative audit](benchmarks/support-v2/error-audit.json)
includes ten fixed-hash-selected cases: negation failures, reversed relations,
correct temporal rejections, and human false accepts/rejects. Human failures
include accepting an altered action where the source describes preservation,
and rejecting an explicit answer when the original question contains a typo.
This is not independent human adjudication or an error-rate estimate. All
official labels and scores remain unchanged.

## Secondary balanced setting

Increasing rejection aggressively produces high useful losses:

| Suite and method | Unsupported rejected | Useful rejected | Accuracy |
|---|---:|---:|---:|
| Human, selected head | 386/512 | 212/512 (41.41%) | 66.99% |
| Human, previous head | 471/512 | 323/512 (63.09%) | 64.45% |
| Controlled, selected head | 525/576 | 308/576 (53.47%) | 68.84% |
| Controlled, previous head | 430/576 | 297/576 (51.56%) | 61.55% |

These accuracy increases are not a successful prudent gate. On reversed
relations the selected balanced gate rejects **181/192 useful cases** and
174/192 unsupported cases: mostly rejecting everything does not solve direction.
Raw relevance's common balanced threshold also rejects 506/512 useful human
cases. That partly reflects mismatched score scales across these different
components, not a claim that the underlying relevance ranker cannot retrieve.

Every method and phenomenon at both operating points is available in the
[complete numerical summary](benchmarks/support-v2/results/summary.json), with
all [individual decisions](benchmarks/support-v2/results/decisions.json).

## Decision and next evidence needed

Retain the previous head as the research reference and archive both newly
trained heads as **experimental, not promoted**. Neither the previous head nor
this experiment qualifies a production abstention system. The controlled
temporal gain is useful evidence; broad natural-language abstention remains weak.

Before another promotion attempt, expand the number of independently worded
permission/negation families and test leave-family-out calibration stability.
For relations, compare representations or encoder adaptation that explicitly
retain entity roles, with both meaningful names and unfamiliar identifiers.
Check relational separation on development before spending a new holdout.
Keep natural questions in the objective and use per-stratum development guardrails
so one easy phenomenon cannot mask a near-chance one. These are follow-up
hypotheses, not changes made using this test.

This test is now observed evidence and must be replaced for future tuning.
Synthetic records, related paraphrases, limited independent families, source
label ambiguity, English-only data and 50% unsupported prevalence restrict
generalization. This measures supplied-paragraph support, not navigation,
corpus-wide retrieval, generated answers or superiority over vector RAG.

## Reproduction, checks and costs

All weights, thresholds, source and feature hashes, selection scores, training
objectives and decisions are in the [artifact directory](benchmarks/support-v2/README.md).
A fresh rerun produced **byte-identical fitted models, decisions and summary**.
The verifier reconstructed every decision and interval and checked that each
conservative calibration stratum respected its 2% budget.

Focused tests pass **24/24**, including real tokenizer/encoder checks and an
independent finite-difference gradient check. The full local suite passes
139 tests with 14 environment-dependent skips. The
[preregistration CI](https://github.com/BVGAL/arborvault/actions/runs/43619445802)
passes **143 tests with 10 skips**, plus its separate Compose checks.

Local CPU extraction took **165.47 seconds** for 16,607 new forward passes,
reusing 12,075 previous fit/selection forwards by input/source hash. Fitting,
selection, calibration and analysis took about **9.45 seconds** including
container launch. These are workflow timings, not serving-latency benchmarks.

**New cloud spend: $0.00. Cumulative estimated spend: $7.94877 of $10.00.**
No paid compute was allocated. Local electricity and hardware amortization
are not measured; see the [cost record](benchmarks/support-v2/run-cost.json).
