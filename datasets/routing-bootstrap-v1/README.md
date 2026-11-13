# Routing bootstrap v1

Reproducible, deliberately small candidate-selection training data for ArborVault's shared
`routing-v1` prompt. It is not SearchTome and contains no private project or conversation data.

| Split | Route examples | Origin |
| --- | ---: | --- |
| train | 392 | 384 procedural examples + routes for 8 public dev questions |
| validation | 84 | 64 procedural examples + routes for 8 different public dev questions |
| synthetic_test | 64 | Procedural fixtures for format testing only |

The separate [public-docs-v1](../public-docs-v1/README.md) test split retains 44 questions
for external retrieval evaluation. None of its questions or qrels is used as a training
target. The known corpus can appear in routing prompts; document-level generalization is
not claimed. Public dev questions are partitioned by family: `httpx-limits` and unknown
families 01/02 train, `fastapi-query` and unknown families 03/04 validate.

Original procedural examples cover English/Italian queries, single/multiple branches,
abstention and shuffled candidate positions with request-local IDs. They reuse templates
across splits, so success on the synthetic test is not a credible real-world quality result.
Public routes use draft, assistant-authored relevance labels and their ancestor paths.
No model-generated hidden reasoning is included or required; targets are JSON ID lists.

Every JSONL record has `id`, `group`, `provenance` and three `messages`. Candidate order
is seeded per example. Hashes, source revisions, upstream licenses and public split IDs are
in `manifest.json`. Training exports retain the original source notices in `licenses/`.
Original procedural content is Apache-2.0; corpus excerpts retain MIT/BSD-3-Clause licenses.

Recreate and validate from the repository root:

```sh
python -m arborvault.training_data datasets/public-docs-v1 --output reports/routing-data
python scripts/train_lora.py reports/routing-data --validate-only
```

Review labels independently and add unseen domains before training a model intended for
release. The included size is suitable for exercising the workflow, not for a quality claim.
See [model evaluation](../../docs/llm-evaluation.md) for training and publishing steps.
