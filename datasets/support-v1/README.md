# Human-labeled paragraph support, v1

This is an attributed, article-separated subset of **SQuAD 2.0**, by Pranav
Rajpurkar, Robin Jia and Percy Liang (2018), *Know What You Don't Know:
Unanswerable Questions for SQuAD*.

- [Original dataset and authors](https://rajpurkar.github.io/SQuAD-explorer/)
- [Paper](https://arxiv.org/abs/1806.03822)
- [Dataset license: CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- [Full license text](LICENSE-DATA.txt)

The original data includes Wikipedia passages and crowdworker-written questions.
Unanswerable questions were deliberately written to resemble answerable questions.
They are human-authored adversarial benchmark questions, **not production traffic**.
The passages are historical source snapshots; their claims are not current advice.

## Task and construction

Given the complete paragraph and a question, decide whether that paragraph
contains the answer. Every retained paragraph has exactly one original positive
and one original negative question. The two questions see the identical paragraph.
Text, original question IDs, human labels and answer spans are unchanged.
`plausible_answers` are distractor annotations, never positive evidence or model inputs.

Articles, paragraphs and questions are ordered by fixed SHA-256 keys. Whole
question/paragraph pairs must fit the pinned MiniLM tokenizer's 512-token limit;
**no truncation or sentence deletion is used**. Positive offsets are validated.
We exclude exact normalized questions already present in earlier ArborVault
datasets, duplicate normalized questions, and passages repeated across articles.
Semantic overlap and encoder pretraining exposure are unknown.

| Role | Source split | Articles | Paragraphs | Supported | Unsupported |
|---|---|---:|---:|---:|---:|
| Fit | original train | 128 | 2,031 | 2,031 | 2,031 |
| Select regularization | original train | 32 | 384 | 384 | 384 |
| Calibrate thresholds | original train | 32 | 384 | 384 | 384 |
| Held-out test | original dev | 35 | 560 | 560 | 560 |

All four roles use disjoint articles and paragraphs. The original public dev split
is held out for this experiment, not a private or never-before-published benchmark.
No claim is made that a supplied paragraph's answerability equals document-level
answerability, retrieval recall, or generated-answer correctness.

## Label audit and remaining limitations

Before model outcomes, an **AI-assisted** audit inspected 25 hash-selected fit
pairs from 25 articles (50 questions). Six pairs were excluded from fit because
of ambiguous wording or plausible alternative support. The original labels were
not rewritten and removed pairs were not replenished. See the itemized
[audit](source-audit.json). Selection, calibration and test labels were untouched.

This is not independent human adjudication. The remaining data can contain noisy
labels, unnatural edits, lexical shortcuts and underspecified questions. The
negative prevalence is artificially 50%; deployment precision depends on the
actual prevalence. Results concern English Wikipedia paragraphs, not long manuals,
multilingual assistants or corpus-wide search. Question-only and context-only
controls measure some, but not all, shortcuts.

## License and reproducibility

This selected dataset and its annotations are distributed under **CC BY-SA 4.0**,
separately from ArborVault's Apache-2.0 implementation. Changes are selection,
article partitioning, audit exclusions, metadata and normalized hashes; original
question and context text is unmodified. Source URLs, SHA-256 checksums, exclusions
and historical question-file locks are in [manifest.json](manifest.json).

To rebuild into a new directory, from the repository in the probe environment:

```sh
python scripts/build_support_dataset.py --download --sources .cache/squad2 \
  --output reports/support-v1-rebuilt
```

The builder refuses to overwrite an existing dataset. Compare the regenerated
`examples.jsonl` SHA-256 with the manifest. The experiment protocol and runtime
commands are in [support-v1-protocol.md](../../docs/support-v1-protocol.md).
