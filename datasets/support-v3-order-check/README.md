# Support-v3 entity-order stress test

Supplement to [support-v3](../support-v3/README.md), described in the
[pre-test addendum](../../docs/support-v3-order-addendum.md). All 640 rows are
test-only: 320 supported, 320 unsupported. They reuse the 160 **held-out** relation
scenarios and five families of the original test, adding equivalent questions
with the opposite name ordering. Contexts, actors, answer spans and labels are
unchanged. This is paired evidence, not another 640 independent scenarios.

In the original relation test, support corresponds exactly to opposite name
orders between question/context. In the alternate test, support corresponds to
matching orders. The union counterbalances this feature; either ordering-only
rule has 50% accuracy there. This diagnoses one shortcut, not all possible ones.

The dataset was added during training after development checkpoints 300 and 600
had been seen, **before any primary or supplemental test predictions**. It never
enters training, checkpoint selection or calibration. It is openly reported as
an addendum, rather than retroactively included in the initial preregistration.

Original synthetic content, Apache-2.0. Same-author label/paraphrase review, no
independent human adjudication. Public numerical results are not production
guarantees. `manifest.json` records source hashes; build with
`python scripts/support_v3_order_check.py build --data /work/rebuilt-order-check`.
