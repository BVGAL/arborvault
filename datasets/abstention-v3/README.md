# Abstention v3 corpus

24 Python 3.13 manuals, disjoint from earlier ArborVault manual experiments.
20 test documents contain 240 answerable and 240 unknown query-document cases;
four calibration documents contain 48 of each. Test questions have 400 distinct
texts because wrong-document negatives reuse positive questions within the split.

Manuals range from short references to a 28,778-word OS manual. This is a larger,
mixed-length, single-domain known-document retrieval test. It is not a corpus-wide
search or answer-generation evaluation, and model pretraining exposure is unknown.

Questions are assistant-authored, source-checked synthetic examples without
independent human adjudication. Unknown categories are unprovided runtime/input
facts, invented APIs absent from the supplied manual, and cross-document transfers.
These categories may have recognizable linguistic cues; the protocol includes a
question-only acceptance control. Single-block evidence labels may omit semantic
alternatives; exact duplicate alternatives and three paired command-name evidence
sets are included. Labels are frozen before inference.

Sources are snapshot HTML from `https://docs.python.org/3.13/library/`. Source
URLs, source hashes, hierarchy statistics and all corpus hashes are in
`manifest.json`; `novelty.json` scopes the exact overlap checks. `positive.tsv`
and `negative-seeds.json` expose the annotation inputs. See
[the frozen protocol](../../docs/abstention-v3-protocol.md).

Documentation is distributed under **PSF-2.0**, and documentation examples also
under **0BSD**; [the full notices](sources/license.html) accompany the snapshots.
Original ArborVault annotations are Apache-2.0. This project is independent of
the Python documentation authors, IBM, and the model authors.
