# Support v3: negation, role reversal and temporal retention

This dataset supports the [frozen experiment protocol](../../docs/support-v3-protocol.md).
The label means the paragraph explicitly supplies a non-null answer satisfying
the question, including its polarity, roles and time. A useful refutation of a
false premise is not a positive named answer under this task definition.

| Role | Human | Negation | Reversed relations | Temporal | Total |
|---|---:|---:|---:|---:|---:|
| Fit | 4,062 | 1,792 | 2,048 | 1,664 | 9,566 |
| Selection | 768 | 288 | 288 | 288 | 1,632 |
| Calibration | 256 | 288 | 288 | 288 | 1,120 |
| Test | 768 | 640 | 640 | 640 | 2,688 |

Every role/phenomenon is label-balanced. Fit includes v2 fit replay and 13 new
families with 32 scenarios each; selection/calibration each have three new
families per controlled phenomenon and 24 scenarios per family; test has five
new families per phenomenon and 32 scenarios per family. A scenario has four
crossed rows: two questions, two contexts, two supported and two unsupported.
Same-question and same-context twins have opposite labels. Both years remain
in temporal contexts; both actors remain in reversed relations. Each role has
disjoint template families, full fictional entity identifiers, documents,
normalized questions and contexts. Repeated questions **within** a role are
allowed only with different scenarios; family-cluster uncertainty is required.

Human examples come from [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/),
the original training file, with unchanged questions, full paragraphs and answer
spans. Fit/selection reuse the established 128/32 articles. Calibration/test
select 8/24 unused articles by a deterministic hash, with 16 supported/unsupported
paragraph pairs per article. Every article previously present in any v1/v2 role
is excluded from new calibration/test. Questions in other repository benchmarks
are excluded by normalized hash. 512-token whole-pair limit, no truncation.
These sources can have occurred in upstream pretraining; held-out here does not
mean universally unseen. The new human sample is smaller than v2 and the source
pool is nearly exhausted; future independent tests need a different source.

The 46 new controlled families were authored and reviewed by the same
AI-assisted author before scoring, including both states/questions, role
inversion, negation scope, answer spans, distracting dates and grammar. The
review covers 184 rendered cases (one complete quartet per family). It is **not
independent human adjudication**. Fifteen test families are a diagnostic wording
test, not 1,920 independent natural requests. Broader wording, realistic context,
domain and language transfer remain limitations. All fictional records are in
English; no company or private-project data is used.

`manifest.json` records provenance hashes. Original SQuAD content/derivatives
are **CC BY-SA 4.0**, with [source license and attribution](../support-v1/README.md).
Original synthetic templates/records and builder code are **Apache-2.0**.
Question/answer spans and oracle metadata are retained for audit, never inference.

Rebuild with the pinned cached tokenizer:

```sh
python scripts/build_support_v3.py --source /sources/train-v2.0.json --output /work/rebuilt-support-v3
```

Do not treat already observed test scores as a fresh validation set for another
round of threshold tuning.
