import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from arborvault.support_data import digest, load, normalized, validate
from arborvault.support_probe import (
    auc,
    calibrate,
    encode_batch,
    fit_models,
    input_key,
    metrics,
    model_input,
    paired_intervals,
)


def corpus():
    return load(Path("datasets/support-v1"))[1]


def test_frozen_support_corpus_is_disjoint_paired_and_has_intact_spans():
    rows = corpus()
    assert len(rows) == 6718
    assert sum(r["role"] == "test" for r in rows) == 1120
    assert len({r["document"] for r in rows if r["role"] == "test"}) == 35
    assert len({r["pair_id"] for r in rows}) * 2 == len(rows)
    validate(rows)


def test_support_validation_rejects_hidden_leakage_and_corrupt_labels():
    rows = corpus()
    changed = copy.deepcopy(rows)
    changed[-1]["document"] = changed[0]["document"]
    with pytest.raises(ValueError, match="crosses"):
        validate(changed)
    changed = copy.deepcopy(rows)
    changed[1]["answers"] = [{"text": "fake", "answer_start": 0}]
    with pytest.raises(ValueError, match="label"):
        validate(changed)
    changed = copy.deepcopy(rows)
    changed[0]["context"] = "An altered paragraph."
    with pytest.raises(ValueError, match="hash"):
        validate(changed)


def test_model_inputs_exclude_all_label_and_source_metadata():
    row = corpus()[0]
    for mode in ("context", "question_only", "context_only"):
        original = input_key(model_input(row, mode))
        altered = dict(
            row,
            answers=[{"text": "leaked"}],
            answerable=not row["answerable"],
            plausible_answers=[{"text": "leaked"}],
            document="leaked",
            role="test",
        )
        assert input_key(model_input(altered, mode)) == original
    assert model_input(row, "context_only") == [row["context"]]
    assert row["pair_id"] == digest(normalized(row["context"]))


def test_auc_ties_and_direction_and_perfect_decisions():
    assert auc([False, True], [0.5, 0.5]) == 0.5
    assert auc([False, True, False, True], [1, 4, 2, 3]) == 1
    assert auc([False, True], [1, 0]) == 0
    rows = [{"answerable": y, "document": "a"} for y in [False, True]]
    result = metrics(rows, [0.1, 0.9], 0.5)
    assert result["balanced_accuracy"] == result["coverage"] * 2 == 1
    assert result["unsupported_fraction_among_accepted"] == 0


def test_calibration_budget_and_test_is_not_a_calibration_source():
    rows = [{"role": "calibration", "answerable": y} for y in [True, True, False, False]]
    gates = calibrate(rows, [0.4, 0.9, 0.3, 0.5], 0.02)
    assert gates["conservative"]["threshold"] == 0.4
    assert gates["conservative"]["calibration_false_rejection"] == 0
    # Balanced ties prefer retaining supported contexts.
    assert gates["balanced"]["threshold"] == 0.4
    with pytest.raises(ValueError, match="calibration"):
        calibrate([dict(r, role="test") for r in rows], [0.4, 0.9, 0.3, 0.5], 0.02)
    with pytest.raises(ValueError, match="test data"):
        fit_models({}, {"fit": [], "selection": [], "calibration": [], "test": []}, {})


def test_context_only_identical_twins_cannot_learn_label():
    from arborvault.answerability import fit_probe, probe_scores

    x = [[1, 2], [1, 2], [3, -1], [3, -1], [-2, 5], [-2, 5]]
    y = [True, False] * 3
    fitted = fit_probe(x, y, list("abcdef"), l2=0.1)
    assert np.allclose(probe_scores(fitted, x), 0.5)
    assert auc(y, probe_scores(fitted, x)) == 0.5


def test_paired_bootstrap_keeps_article_pairs_together():
    rows = [{"document": d, "answerable": y} for d in ["a", "b"] for y in [True, False]]
    scores = {"context": [1, 0, 1, 0], "previous": [0, 1, 0, 1], "raw": [1, 0, 1, 0]}
    thresholds = {
        name: {point: {"threshold": 0.5} for point in ("balanced", "conservative")}
        for name in scores
    }
    result = paired_intervals(rows, scores, thresholds, seed=42, replicates=100)
    assert result["articles"] == 2
    assert result["operating_points"]["balanced"]["new_minus_previous"] == {
        "estimate": 1.0,
        "ci95": [1.0, 1.0],
    }


@pytest.mark.skipif(
    os.getenv("RUN_SUPPORT_ENCODER_TEST") != "1", reason="Requires cached real CPU encoder"
)
def test_real_encoder_whole_pair_and_same_cls_as_previous_probe():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    config = json.loads(Path("configs/support-v1.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"], revision=config["revision"], local_files_only=True
    )
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            config["model"],
            revision=config["revision"],
            local_files_only=True,
            attn_implementation="eager",
        )
        .cpu()
        .eval()
    )
    torch.set_num_threads(2)
    rows = [r for r in corpus() if r["role"] == "fit"][:2]
    encoded, batch = encode_batch(tokenizer, [model_input(r, "context") for r in rows], 512)
    for r, value in zip(rows, encoded, strict=True):
        assert len(value["input_ids"]) == r["input_tokens"]
        assert value["input_ids"] == tokenizer(r["question"], r["context"])["input_ids"]
    with torch.inference_mode():
        full = model(**batch, output_hidden_states=True)
        old = model.base_model(**batch).last_hidden_state[:, 0]
    assert torch.equal(full.hidden_states[-1][:, 0], old)
    assert full.logits.shape == (2, 1)
    with pytest.raises(ValueError, match="truncation is forbidden"):
        encode_batch(tokenizer, [["Question?", "long " * 1000]], 512)
