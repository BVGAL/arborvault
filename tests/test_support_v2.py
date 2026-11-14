import copy
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

from arborvault.support_data import normalized
from arborvault.support_probe import auc, encode_batch, model_input
from arborvault.support_v2_data import load, validate
from arborvault.support_v2_eval import fit_models
from arborvault.support_v2_models import (
    calibrate_common,
    mlp_checkpoints,
    mlp_objective_gradient,
    partial_rejection_auc,
    scores,
    select_score,
)


def dataset():
    return load(Path("datasets/support-v2"))[1]


def test_human_transfer_articles_and_questions_are_new_to_all_v1_roles():
    rows = dataset()
    old = [
        json.loads(line)
        for line in Path("datasets/support-v1/examples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    prior_docs = {r["document"] for r in old}
    prior_questions = {normalized(r["question"]) for r in old}
    fresh = [r for r in rows if r["suite"] == "human" and r["role"] in {"calibration", "test"}]
    assert len(rows) == 13150
    assert len(fresh) == 1792
    assert not {r["document"] for r in fresh} & prior_docs
    assert not {normalized(r["question"]) for r in fresh} & prior_questions
    assert len({r["document"] for r in fresh if r["role"] == "test"}) == 32


def test_counterfactuals_remove_single_input_shortcuts_and_keep_both_dates():
    rows = dataset()
    for row in rows:
        if row["suite"] != "contrast":
            continue
        if row["phenomenon"] == "relation":
            assert all(
                entity in row["context"] and entity in row["question"] for entity in row["entities"]
            )
        if row["phenomenon"] == "temporal":
            years = set(re.findall(r"\b20\d{2}\b", row["context"]))
            assert len(years) == 2
            assert set(re.findall(r"\b20\d{2}\b", row["question"])) < years
    group = next(r["group_id"] for r in rows if r["suite"] == "contrast")
    quad = [r for r in rows if r["group_id"] == group]
    assert sum(r["answerable"] for r in quad) == 2
    for field in ("question", "context"):
        for text in {r[field] for r in quad}:
            assert {r["answerable"] for r in quad if r[field] == text} == {True, False}


def test_corrupt_or_leaking_counterfactuals_fail_closed():
    rows = dataset()
    altered = copy.deepcopy(rows)
    first = next(r for r in altered if r["suite"] == "contrast")
    first["oracle"]["requested_state"] = 1 - first["oracle"]["requested_state"]
    with pytest.raises(ValueError, match="oracle"):
        validate(altered)
    altered = copy.deepcopy(rows)
    first = next(r for r in altered if r["suite"] == "contrast" and r["role"] == "test")
    first["template_family"] = "negation/00"
    with pytest.raises(ValueError, match="Template family"):
        validate(altered)


def test_partial_auc_operating_direction_and_ties():
    assert partial_rejection_auc([False, True], [0, 1]) == 1
    assert partial_rejection_auc([False, True], [1, 0]) == 0
    assert partial_rejection_auc([False, True], [0.5, 0.5]) == pytest.approx(0.025)
    with pytest.raises(ValueError, match="selection"):
        select_score([{"role": "test"}], [], {})
    with pytest.raises(ValueError, match="test data"):
        fit_models({}, {"fit": [], "selection": [], "calibration": [], "test": []}, {})


def test_mlp_gradient_matches_independent_finite_differences():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(5, 2))
    y = np.array([0, 1, 1, 0, 1])
    parameters = [
        rng.normal(size=(2, 3)) * 0.2,
        rng.normal(size=3) * 0.2,
        rng.normal(size=3) * 0.2,
        np.zeros(1),
    ]
    _, grad = mlp_objective_gradient(x, y, parameters, 0.01)
    eps = 1e-6
    for pi, param in enumerate(parameters):
        for index in np.ndindex(param.shape):
            plus = [p.copy() for p in parameters]
            minus = [p.copy() for p in parameters]
            plus[pi][index] += eps
            minus[pi][index] -= eps
            expected = (
                mlp_objective_gradient(x, y, plus, 0.01)[0]
                - mlp_objective_gradient(x, y, minus, 0.01)[0]
            ) / (2 * eps)
            assert grad[pi][index] == pytest.approx(expected, abs=1e-7)


def test_mlp_can_learn_interaction_and_question_twins_stay_at_chance():
    config = {
        "seed": 42,
        "mlp_hidden": 8,
        "mlp_learning_rate": 0.02,
        "mlp_l2": 0.001,
        "mlp_checkpoints": [400],
    }
    x = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float)
    y = [1, 0, 0, 1]
    model = mlp_checkpoints(x, y, config)[400]
    assert (np.array(scores(model, x)) >= 0.5).tolist() == list(map(bool, y))
    questions = np.array([[1, 2], [3, 4], [1, 2], [3, 4]], dtype=float)
    model = mlp_checkpoints(questions, y, config)[400]
    assert auc(y, scores(model, questions)) == 0.5


def test_common_gate_budget_applies_to_each_stratum():
    config = {"strata": ["human", "negation"], "conservative_max_false_rejection_rate": 0.02}
    rows = [
        {"role": "calibration", "phenomenon": s, "answerable": y}
        for s in config["strata"]
        for y in (True, True, False, False)
    ]
    result = calibrate_common(rows, [0.9, 0.8, 0.2, 0.4, 0.3, 0.7, 0.1, 0.6], config)
    assert result["conservative"]["threshold"] == 0.3
    assert all(
        s["false_rejection"] == 0 for s in result["conservative"]["calibration_strata"].values()
    )
    with pytest.raises(ValueError, match="calibration"):
        calibrate_common([dict(r, role="test") for r in rows], [0.5] * 8, config)


@pytest.mark.skipif(
    os.getenv("RUN_SUPPORT_ENCODER_TEST") != "1", reason="Requires pinned cached tokenizer"
)
def test_real_tokenizer_fits_all_training_contrast_families_without_truncation():
    from transformers import AutoTokenizer

    config = json.loads(Path("configs/support-v2.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"], revision=config["revision"], local_files_only=True
    )
    rows = [
        r
        for r in dataset()
        if r["suite"] == "contrast" and r["role"] == "fit" and r["group_id"].endswith("/000")
    ]
    encoded, _ = encode_batch(tokenizer, [model_input(r, "context") for r in rows], 512)
    assert len(rows) == 60
    assert [len(v["input_ids"]) for v in encoded] == [r["input_tokens"] for r in rows]
