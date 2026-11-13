# QASPER routing v2 pilot

Derived from **QASPER v0.3**, by Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan,
Noah A. Smith and Matt Gardner (2021), *A Dataset of Information-Seeking Questions and
Answers Anchored in Research Papers*. [Upstream dataset](https://huggingface.co/datasets/allenai/qasper),
[paper](https://arxiv.org/abs/2105.03011). Dataset content is **CC BY 4.0**, separate from
ArborVault's Apache-2.0 code license. Preserve this attribution and source notices.

| Split | Documents | Questions | Answerable | Unknown |
| --- | ---: | ---: | ---: | ---: |
| Train | 426 | 600 | 500 | 100 |
| Development | 102 | 120 | 100 | 20 |
| Test | 96 | 120 | 100 | 20 |

This is **known-document paragraph evidence retrieval**, not global search over all papers,
answer generation, or the official QASPER score. Every method is given the same document
scope and paragraph units. The new test has different documents from training/development
and from public-docs-v1. It is publicly available upstream and may have appeared in model
pretraining; it is not a claim of unknown pretraining data.

The upstream questions and evidence were annotated by people. This conversion is deterministic,
with exact evidence mapping checks and regression tests; it has not received a new independent
human audit. No model generated extra questions or selected examples by measured performance.

## Transformations and selection

- Preserve upstream train/dev/test paper partitions. Select fixed quotas in seeded SHA-256
  question order, excluding exact normalized question duplicates across and within splits.
- Build section/subsection paths from `:::` headings and paragraph leaf nodes. Normalize
  whitespace; keep full paragraph text. Leaf display titles are first-sentence prefixes.
- Retain only consistent answerability annotations. Exclude questions with figures/tables,
  unmapped textual evidence, or more than four gold paragraphs in an annotation.
- Bound the pilot to papers with at most 400 nodes and paragraphs at most 12,000 characters.
  These filters were fixed before model evaluation; exclusions are counted in the manifest.
- Preserve alternative human evidence sets. Score against the best matching annotation;
  never claim this custom Recall@5 metric equals the official evidence F1.
- Remove annotator/worker identifiers, demographic/background fields and generated answers.
  Keep public paper IDs, titles, source URLs, question IDs and evidence-node mappings.

All examples are English and from NLP papers. This pilot does not establish Italian quality,
unseen-domain generalization, PDF/OCR accuracy or corpus-wide document discovery. Semantic
question patterns can recur despite document and exact-text separation.

## Training export

`arborvault.qasper.training_routes()` reads **only the train split**. The 600 human questions
produce 1,400 decisions: 1,273 oracle-path decisions and 127 wrong-subtree negative decisions.
There are 1,173 positive decisions, of which 673 (57.37%) are below the first routing level.
Candidate sets vary up to 101; local IDs are shuffled deterministically. Wrong-subtree negatives
exclude ancestors of every annotated evidence alternative, not only the chosen training target.
There are no procedural query templates. “Wrong subtree” means no annotated evidence in that
candidate set; incompleteness of human labels remains a limitation.

Use the versioned routing-v2 prompt and tokenizer-aware feature fitting for both training
and inference. Record final tokenization hashes and compression counts at execution time.

## Reproduction and license

Download the original archives listed in `SOURCES.json`, verify SHA-256, and extract only the
three named QASPER JSON files to a local directory. Then run:

```sh
python -m arborvault.qasper /path/to/source-json /path/to/new-output
```

The exporter refuses an existing output directory. `manifest.json` records input and output
hashes, counts, selection policy and exclusions. Original text and annotations remain under
[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
Changes by ArborVault: deterministic selection, normalization, hierarchical node conversion,
metadata minimization and evidence mapping. Upstream source paper URLs remain in every document.
