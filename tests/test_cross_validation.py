import copy
import math

import pytest

from arborvault.cross_validation import (
    apply_gate,
    discriminative_metrics,
    features,
    fit_gate,
    fold_plan,
    gate_scores,
    operating_metrics,
    tune_threshold,
    validate_plan,
)
from arborvault.domain import Node
from arborvault.routing_v2 import Tree


def test_outer_and_inner_groups_keep_entire_documents_together():
    queries = [{"id": str(i), "document": str(i // 3)} for i in range(600)]
    plan = fold_plan(queries)
    assert plan == fold_plan(list(reversed(queries)))
    evaluated = [i for f in plan for i in f["roles"]["evaluation"]]
    assert len(evaluated) == len(set(evaluated)) == 600
    damaged = copy.deepcopy(plan)
    source = damaged[0]["roles"]["fit"]
    damaged[0]["roles"]["gate_fit"].append(source.pop())
    with pytest.raises(ValueError, match="Document leakage"):
        validate_plan(queries, damaged)


def row(identity, answerable=True, recall=1.0):
    return {
        "id": identity,
        "document": identity,
        "answerable": answerable,
        "predicted": ["a", "b"],
        "abstained": False,
        "recall": recall if answerable else None,
        "mrr": recall if answerable else None,
        "ndcg": recall if answerable else None,
        "evidence_coverage": recall if answerable else None,
    }


def test_query_gate_preserves_context_and_charges_false_abstentions():
    rows = [row("p"), row("u", False)]
    original = copy.deepcopy(rows)
    accepted = apply_gate(rows, [0.8, 0.1], 0.5)
    assert accepted[0]["predicted"] == ["a", "b"]
    assert accepted[1]["predicted"] == []
    report = operating_metrics(accepted)
    assert report["recall"] == 1
    assert report["unknown_abstentions"] == 1
    assert report["unsupported_return_rate"] == 0
    rejected = operating_metrics(apply_gate(rows, [0.2, 0.1], 0.5))
    assert rejected["recall"] == 0
    assert rejected["answerable_empty"] == 1
    assert rejected["unsupported_return_rate"] is None
    assert rows == original


def test_exhaustive_threshold_grid_can_reject_unknown_without_dropping_passages():
    rows = [row("p"), row("u", False)]
    tuned = tune_threshold(rows, [0.11, 0.10])
    assert tuned["conservative"]["unknown_abstentions"] == 1
    assert tuned["conservative"]["recall"] == 1
    tied = tune_threshold(rows, [0.1, 0.1])
    assert tied["conservative"]["unknown_abstentions"] == 0


def test_conservative_gate_does_not_hide_recall_loss_in_conditional_metric():
    rows = [row("a"), row("b"), row("u", False)]
    tuned = tune_threshold(rows, [0.1, 0.9, 0.2])
    assert tuned["conservative"]["answerable_empty"] == 0
    assert tuned["balanced"]["answerable_empty"] == 1
    assert tuned["balanced"]["recall"] == 0.5
    assert tuned["balanced"]["conditional_answerable_recall"] == 1


def test_logistic_gate_handles_constant_columns_and_restores_from_json():
    import json

    pytest.importorskip("numpy")
    x = [[float(i), 1, 0, 0, 0] for i in range(-10, 11) if i]
    labels = [v[0] > 0 for v in x]
    gate = fit_gate(x, labels)
    restored = json.loads(json.dumps(gate))
    prediction = gate_scores(restored, [[-5, 1, 0, 0, 0], [5, 1, 0, 0, 0]])
    assert prediction[0] < 0.2 < 0.8 < prediction[1]
    assert prediction[0] + prediction[1] == pytest.approx(1)
    assert gate["mean"][1] == 1 and gate["scale"][1] == 1
    assert all(math.isfinite(c) for c in gate["coefficients"])
    with pytest.raises(ValueError, match="both classes"):
        fit_gate(x, [True] * len(x))


def test_answerability_features_never_use_evidence_labels():
    tree = Tree(
        [
            Node("r", None, "doc", "", ["doc"], 0),
            Node("p", "r", "one", "alpha evidence", ["doc", "one"], 1),
            Node("n", "r", "two", "unrelated", ["doc", "two"], 2),
        ]
    )
    q = {"question": "alpha", "gold_sets": [["p"]], "answerable": True}
    entry = {"scores": {"p": 2.0, "n": 1.0}}
    first = features(tree, q, entry)
    q.update(gold_sets=[], answerable=False)
    assert features(tree, q, entry) == first
    assert first == [2.0, 1.0, 1.5, 0.5, 1.0]


def test_discrimination_metrics_account_for_ties_and_unknown_prevalence():
    rows = [row("a"), row("b"), row("u", False)]
    tied = discriminative_metrics(rows, [0.5, 0.5, 0.5])
    assert tied["answerability_auroc"] == 0.5
    assert tied["unknown_average_precision"] == pytest.approx(1 / 3)
    perfect = discriminative_metrics(rows, [0.8, 0.9, 0.1])
    assert perfect["answerability_auroc"] == perfect["unknown_average_precision"] == 1
