# Expanded navigation and abstention validation

Frozen before inference on 2026-11-14. Independent ArborVault research; not an
official IBM implementation or a verified STAIR reproduction.

## Questions and scope

1. Does the incremental navigation improvement persist on more unseen manuals?
2. Can a small learned acceptance filter reduce unsupported returns from the
   pretrained cross-encoder without a large loss on answerable questions?
3. Does that filter use retrieved evidence, or primarily recognize question style?

The primary system is the **pretrained** MS MARCO MiniLM cross-encoder with
`tree_guided-64`. Controls are Qwen3-4B base, full-leaf E5, flat BM25-shortlisted
retrieval, the previous tree, and incremental navigation without descendant
lexical guidance. No transformer weights are fine-tuned and Tacit is not rerun.

## Partitions

| Role | Documents | Purpose |
|---|---:|---|
| Historical fit | 18 | Fit the acceptance probe using previously published contexts |
| Historical selection | 4 | Select regularization using document-macro AUROC |
| New calibration | 4 | Set acceptance thresholds |
| New test | 20 | Evaluate once, after all rules are fixed |

Historical fit combines the 12 test manuals from `long-manuals-v1` and six from
`navigation-v2`. Selection uses argparse, decimal, pathlib and unittest from
the latter study. Previous test data become development data explicitly; none
are counted as new validation. The new corpus has **480 test cases: 240
answerable and 240 unknown**, versus 96 in the previous navigation study.
Calibration has 48 answerable and 48 unknown cases. All four roles are disjoint
by document, and historical inputs are hash-locked.

Within each new manual, 12 questions are source-grounded and 12 are unknown:
four request unprovided runtime state or inputs, four assume an undocumented
invented API, and four concern an API in another manual. Cross-document
transfers repeat positive question texts **within the same split**: there are
400 distinct test question texts, not 480 independent questions.

The source is Python 3.13 documentation, with mixed document lengths. This is a
larger single-domain test, not broad domain generalization. The correct manual
is supplied. Questions and reference answers are assistant-authored and
source-checked; there is no independent human adjudication. Exact duplicate
passages are accepted as alternatives, but semantic alternatives may be missing.
Three pdb questions also require a separate command-name passage. Model
pretraining exposure is unknown. Exact overlap checks are not semantic
contamination detection.

## Retrieval and compute accounting

All four neural navigation policies receive **64 charged question-candidate
pairs**, including branches, and return at most five passages / 512 words.
Qwen and MiniLM receive the same fitted candidate text, at most 512 input tokens.
Qwen uses the native binary decision prompt from the previous navigation study,
with NF4 weights and no generated reasoning. Models and revisions are pinned in
the configuration and its scorer recipe.

`tree_guided` uses global descendant BM25 scores and three query-matched snippets.
It is lexical-assisted navigation. BM25 preprocessing is outside the neural-pair
allowance; full-leaf E5 and BM25 remain external references. The frozen search
policy is unchanged from navigation v2. Search caches can be shared between
experimental policies, but each traversal is independently charged and replayed.
Total experiment times are not isolated serving latencies or equal-FLOP costs.

## Acceptance probe

The frozen MiniLM cross-encoder encodes `(question, returned context)`. A logistic
linear classifier learns from its last-layer CLS representation (384 dimensions).
It sees only the returned 512-word context, further shortened to a ranked prefix
that fits together with the whole question into **512 tokens**. Truncation is
recorded. This can omit lower-ranked evidence and is a limitation of the verifier.

Historical examples come from Qwen base, MiniLM base and E5. A positive context is
eligible for fitting only when it contains a full annotated evidence set.
Unanswerable contexts are negative. An answerable retrieval miss is omitted,
not relabeled as a negative, to avoid manufacturing negatives from incomplete
annotations. Duplicate contexts do not increase a question's total fitting
weight. Feature scaling is fitted only on historical fit rows.

Three fixed L2 penalties (0.1, 1, 10) are compared on historical selection
documents; document-macro answerability AUROC selects one, with larger L2 breaking
ties. The selected probe is **not refitted** on selection, calibration or test
documents. The same procedure separately fits a **question-only probe** to expose
synthetic question-style shortcuts. Neither probe receives gold IDs, document
IDs, reference answers or negative-category labels as model input.

On the new calibration documents we compare:

- Raw maximum retrieval score, retaining the previous threshold baseline.
- Learned probe alone.
- Joint acceptance: both the raw conservative gate and the learned gate accept.
- Question-only versions of the last two controls.

Conservative thresholds maximize unknown rejection subject to no more than a
two-percentage-point recall loss and false-rejection increase on calibration,
measured **jointly against ungated retrieval**. With 48 calibration positives,
this permits zero extra false rejections. Balanced thresholds maximize balanced
accuracy. These are empirical calibration criteria, not guarantees under domain
shift. Unknown calibration prevalence is artificial (50%). All operating points
are reported; there is no test-selected threshold or model.

Each deployed learned gate adds **one MiniLM forward pass** of at most 512 tokens
after retrieval. It does not consume another search candidate. Hence the learned
system has up to 64 retrieval evaluations **plus one verification evaluation**;
it is not claimed to have the same total compute as the raw-score gate. Qwen and
E5 also receive the same optional gate for comparison. The question-only control
is a separate experiment, not an additional production step.

## Reporting and acceptance of claims

Report ungated evidence recall, full evidence coverage, unknown rejection, false
rejection, retained-query coverage, and per-category and per-document results.
Use 5,000 paired bootstrap draws over nine fixed related-manual families (listed
in the config); intervals with only nine clusters remain exploratory. Report
paired changes in recall and both rejection rates. Broad ranking of models is
not inferred from a few cases. A null or harmful gate result is published too.

The primary abstention comparison is the cross-encoder guided joint conservative
gate against its raw conservative gate. A useful improvement must be assessed
alongside false rejections and the question-only control. No production API
defaults change automatically as a result of this benchmark.

## Reproduction and budget

```sh
python scripts/run_navigation_benchmark.py --config configs/abstention-v3.json --output reports/abstention-v3/experiment
python scripts/analyze_navigation_benchmark.py --config configs/abstention-v3.json --dataset datasets/abstention-v3 --experiment reports/abstention-v3/experiment --output reports/abstention-v3/verified-summary.json
```

Use the pinned Linux/CUDA environment in `requirements.training.txt`. The analyzer
verifies corpus and artifact hashes, exact traversal replay, ancestor access,
unique charges, fitted-context hashes and document partitions. Fitted probe
coefficients are exported separately and contain no private data.

Prior estimated cumulative spend is **USD 6.70558 / 10**. This run permits one
GPU at at most USD 0.95/hour, an independent allocation deletion deadline of
2.5 hours, and an inference deadline of 7,800 seconds. At that maximum GPU rate,
compute plus the configured temporary storage leaves a margin below the remaining
USD 3.29442. Allocation, setup, downloads and export all count toward cost. Delete
the GPU and temporary volume after verified artifact download, then perform
analysis and reporting locally. Actual costs and cleanup evidence will be reported.
