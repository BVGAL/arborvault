# Semantic contrasts and fresh human transfer: support v2

This dataset targets negation, reversed relations and temporal binding while
retaining a separate human-authored transfer benchmark. The task is **whether
the supplied paragraph supports a non-null named answer to the question**.
A refuted premise does not supply the requested entity/reference. Explaining
that no such answer is given can be useful assistant behavior, but counts as
abstention in this support-classification task. No answer generation is evaluated.

## Human component

Human questions and Wikipedia paragraphs come from **SQuAD 2.0**, by Pranav
Rajpurkar, Robin Jia and Percy Liang (2018), *Know What You Don't Know:
Unanswerable Questions for SQuAD*.
[Original source and attribution](https://rajpurkar.github.io/SQuAD-explorer/),
[paper](https://arxiv.org/abs/1806.03822),
[license](../support-v1/LICENSE-DATA.txt): **CC BY-SA 4.0**.

We retain the 4,062 fit questions and 768 selection questions from support-v1.
All 56 human calibration/test articles are absent from **every** support-v1
role, including its old test. They are hash-selected from unused articles in
the original public SQuAD training split. Text, source question IDs and answer
spans are unchanged; selection, role assignments and metadata are new. Every
selected paragraph has one original positive and one original human negative.
No positive answer spans are deleted to invent negative labels.

These are crowdworker-written adversarial questions, not live application logs.
Their labels can be ambiguous. The source was public and may have appeared in
encoder pretraining. Novelty means unused in the earlier ArborVault study, not
unknown to every pretrained model. Exact normalized questions from the earlier
ArborVault question files are excluded from the new human partitions.

## Controlled component

Original fictional records, questions and templates use **Apache-2.0**. No
private projects, documents, staff names or business facts are used. Each
scenario crosses **two contexts with two questions**, producing four cases:
two supported and two unsupported. The identical question has both labels
across contexts; the identical context has both labels across questions.

- **Negation:** permission is granted or denied, with questions asking for
  the unit granted or denied permission. Negation is present in valid examples too.
- **Relation:** sender and recipient swap, while both unit names and the
  reference code remain in the text. The question asks for the code of a
  particular directed transfer.
- **Temporal:** the event year and a cited document's year swap. Both years
  remain present, so literal date overlap cannot determine the label.

Units and scenarios are unique across roles. Template families are disjoint:
15 fit, 6 selection, 6 calibration and 9 test families. Entities, oracle state,
variant numbers, source IDs, role and labels are metadata only, never encoder
inputs. The full question and paragraph fit 512 tokens without truncation.

| Role | Human questions | Human articles | Contrast questions | Contrast families |
|---|---:|---:|---:|---:|
| Fit | 4,062 | 128 | 3,840 | 15 |
| Selection | 768 | 32 | 768 | 6 |
| Calibration | 768 | 24 | 768 | 6 |
| Test | 1,024 | 32 | 1,152 | 9 |

Each component and phenomenon has equal positive/negative counts. The two test
components are reported separately; pooling them would hide domain-specific
behavior. Test contrasts have only **three wording families per phenomenon**,
so thousands of template instances do not mean thousands of independent tasks.

## Annotation quality and limits

The counterfactual labels follow the authored equality oracle (requested state
equals recorded state). Automated checks verify crossings, balanced labels per
question/context, exact positive spans, disjoint roles, templates and entities.
An **AI-assisted author review** inspected both contexts and questions of one
rendered scenario in each of the 36 families before model outcomes. See
[template-audit.json](template-audit.json). This is not independent human
adjudication; the author knew the test templates while designing them.

Short fictional records, hexadecimal unit identifiers, limited vocabularies,
shared semantic operators and balanced prevalence limit realism and independence.
Disjoint templates prevent exact wording reuse, but related paraphrases and
concepts remain shared. The separate human transfer test guards against
reporting template success as general answerability improvement. Both components
still test paragraph support, not long-document navigation or end-to-end RAG.

## Rebuild and licenses

The mixed `examples.jsonl` contains separately licensed records, identified by
each row's `license`. Human SQuAD derivatives retain CC BY-SA 4.0 attribution;
independently authored synthetic records use Apache-2.0. The manifest provides
source hashes, previous input locks, counts and provenance.

Using the CPU environment and the official hash-locked SQuAD source:

```sh
python scripts/build_support_v2.py --source .cache/squad2/train-v2.0.json \
  --output reports/support-v2-rebuilt
```

The builder refuses to overwrite existing data. The source URL and SHA-256 are
in [configuration](../../configs/support-v2.json). The earlier source-download
command in [support-v1](../support-v1/README.md) fetches the same official file.
Compare rebuilt `examples.jsonl` with this dataset's manifest before extraction.
