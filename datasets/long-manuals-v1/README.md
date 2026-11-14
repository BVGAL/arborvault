# Long manuals v1: frozen, exploratory retrieval test

Sixteen Python 3.13 documentation pages, downloaded on 2026-11-14, form a new
ArborVault corpus. The frozen bytes, their SHA-256 hashes and normalized trees
are included. This is **not a claim that these documents were absent from model
pretraining**. See [novelty audit](novelty-audit.json).

| Partition | Documents | Answerable | Unanswerable |
|---|---:|---:|---:|
| Calibration | 4 | 40 | 56 |
| Test | 12 | 120 | 168 |

Counts are query-document cases. Transfers reuse question wording with another
document: the test has **204 distinct question texts** among 288 cases; calibration
has 68 among 96. These are not 288 independent question texts.

Calibration uses argparse, decimal, pathlib and unittest. All other manuals are
test-only. No weights are trained on this corpus. Questions, evidence, corpus,
checkpoint choice and threshold-selection rules are frozen before inference.
Four calibration documents are a small sample: a calibration guard is not a
guarantee of test behavior.

The documents contain 8,295–25,094 whitespace-delimited words (median 12,923.5),
versus a median of 3,416 in the historical QASPER pool. They are longer **in
distribution**, not longer than every old paper. Each tree has 463–1,150 nodes
and 4–7 levels. Sections and API definitions form branches; paragraphs, code
blocks, table rows and list items form leaves. Blocks above 160 words are split
consecutively without overlap. These native documentation blocks are often
shorter than QASPER paragraphs; node counts are not directly comparable measures
of difficulty. No document is cut down to the previous 400-node pilot limit.

## Annotation provenance and limitations

All questions and reference answers are **assistant-authored synthetic data**.
The source evidence was checked during authoring, but there has been **no
independent human adjudication**. `positive.tsv` and `unknown.tsv` preserve the
authoring records. `questions.json` includes evidence IDs, reference answers,
source links, scope rationales and review status.

Each document has ten source-grounded questions, seven plausible requests for
unspecified performance, configuration or operational guarantees, and seven
questions transferred from another manual in the **same partition**. A transfer
has an answer elsewhere, but not in the supplied document. The source question
and supporting evidence elsewhere are recorded. These are scoped negatives,
not claims that the information is unknowable. Transfers intentionally repeat
question wording within a partition; no question crosses calibration/test.

Deployment-specific unknowns and wrong-manual transfers can be easier than
natural ambiguous user questions. Report their results separately. The 58.3%
unknown prevalence is deliberate and does not estimate production prevalence.
Answerable questions target a single supporting block; exact duplicate blocks
are accepted as alternatives. Semantically valid alternatives may remain
unannotated. This corpus is a transparent diagnostic benchmark, not an
independently validated leaderboard or a multi-hop question-answering test.

The logging reference and cookbook share a source family. Analysis joins them
for uncertainty estimates, leaving eleven test document-family groups. All
manuals share the same language ecosystem; do not generalize to all long
documents or infer a length-only effect from a simultaneous domain change.

## Scope and licensing

The correct document is supplied. Systems retrieve evidence under the same
top-five / 512-word context budget; answer generation is not evaluated.
HTML navigation and scripts are removed during normalization, text whitespace
is normalized, headings and API signatures become paths, and long blocks are
split. Source HTML remains unchanged.

Before inspecting any score or metric, a partial inference attempt was stopped:
generic leaf labels were found to remove the paragraph previews used by the
existing tree-outline implementation. Leaf titles now use the first sentence,
up to 110 characters, matching the QASPER parser. Source text, question wording,
labels, partitions, checkpoints and retrieval policies were unchanged. All
systems were rerun on the corrected representation; the interrupted attempt is
included in the cost record. Git history preserves the initial and corrected
locks. This was an implementation correction before results, not test-based tuning.

Python documentation is copyright Python Software Foundation and the other
holders identified in the sources. Its PSF-2.0 terms and historical notices are
preserved in [the full source license](sources/license.html); documentation code
examples are additionally available under 0BSD. See the official
[Python license](https://docs.python.org/3.13/license.html). Those third-party
terms govern the copied source content. ArborVault's new annotations and code
use Apache-2.0. No endorsement by Python, IBM or any model provider is implied.

The manifest lists each source URL and hash. `tests/test_long_benchmark.py`
reconstructs every tree from the frozen HTML and checks evidence, partitions,
counts and hashes. The inference protocol is in
[`docs/long-manuals-v1-protocol.md`](../../docs/long-manuals-v1-protocol.md).
