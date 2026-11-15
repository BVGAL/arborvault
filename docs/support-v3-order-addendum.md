# Pre-test addendum: counterbalance entity order

This addendum was prepared **during training, after observing development
checkpoints 300 and 600, before scoring any primary test examples**. It is not
part of the initial preregistration. It does not change the frozen dataset,
training, checkpoints, selection rule or calibration. It adds a supplementary
stress test and an additional requirement before recommending this candidate.

Static inspection found a shortcut in the initial controlled relation test:
all five families phrase a supported question with the two names in the reverse
order to their appearance in the context. A rule that predicts support from
this ordering alone achieves **100%** there. Crossed labels prevent question-only
and context-only shortcuts, but do not prevent this joint-input shortcut.
This limitation must accompany any result on the original relation test.

The [additional dataset](../datasets/support-v3-order-check/README.md) keeps
the same 160 scenarios, contexts and labels, but adds 640 rows with semantically
equivalent questions that reverse the question's name order. The union has
1,280 rows, and agreement of name order is exactly label-balanced (50% accuracy
for either ordering rule). Each question and context alone remains balanced.
It is a targeted stress test of one known shortcut, not proof that all shortcuts
are removed. Question paraphrases and labels were checked by the same author;
no independent annotation is claimed. There are still only **five families**.

Run all already-declared models with the **unchanged final calibration gates**.
Do not select another checkpoint, tune thresholds, or train on these rows.
Report alternate wording separately and the balanced union, paired bootstrap by
family, and within-question positive/negative context ordering. The initial
primary promotion gate stays as published; a recommendation must additionally
reject at least 10 percentage points more unsupported cases than v2 on the
alternate wording while rejecting at most 2% of useful alternate cases.

Build and run with `scripts/support_v3_order_check.py build` and `evaluate`.
`docs/benchmarks/support-v3/order-addendum-inputs.json` locks the source, data and
this addendum before primary test scoring. Any differences between the original
and alternate test will be reported, not repaired after observation.
