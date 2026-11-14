# Navigation v2 held-out manuals

Six new Python 3.13 manuals, 48 answerable and 48 unanswerable test cases; four
reused calibration manuals with 96 cases. This is an assistant-authored synthetic
benchmark, not an independently human-reviewed dataset. Known-document scope,
single-block evidence and unknown model pretraining exposure limit conclusions.

`manifest.json` locks source HTML, parsed trees, questions and annotations.
`positive.tsv` and `unknown.tsv` contain the newly authored items. Cross-document
negative transfers stay within the new test partition. `novelty.json` records
the exact overlap checks. Source documentation is PSF-2.0, examples additionally
0BSD; see `sources/license.html`. Original annotations are Apache-2.0.

See [the frozen protocol](../../docs/navigation-v2-protocol.md) for costs, model
revisions, policy definitions, calibration and limitations.
