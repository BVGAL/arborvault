import copy
import json
from pathlib import Path

import numpy as np
import pytest

from arborvault.support_data import normalized
from arborvault.support_v2_data import load, validate
from arborvault.support_v3_eval import counterfactual_ordering
from arborvault.support_v3_training import choice, choose


def dataset():
    return load(Path("datasets/support-v3"))[1]


def test_new_human_articles_and_questions_exclude_all_previous_roles():
    rows = dataset()
    prior = []
    for version in ("v1", "v2"):
        prior.extend(
            json.loads(line)
            for line in Path(f"datasets/support-{version}/examples.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    fresh = [r for r in rows if r["suite"] == "human" and r["role"] in {"calibration", "test"}]
    assert len(rows) == 15006
    assert len(fresh) == 1024
    assert not {r["document"] for r in fresh} & {r["document"] for r in prior}
    assert not {normalized(r["question"]) for r in fresh} & {
        normalized(r["question"]) for r in prior
    }
    assert not {normalized(r["context"]) for r in fresh} & {normalized(r["context"]) for r in prior}
    assert len({r["document"] for r in fresh if r["role"] == "test"}) == 24


def test_dataset_replays_only_old_fit_and_human_selection():
    rows = dataset()
    old = load(Path("datasets/support-v2"))[1]
    expected = {
        r["id"]: r
        for r in old
        if r["role"] == "fit" or (r["role"] == "selection" and r["suite"] == "human")
    }
    for row in rows:
        if row["id"] in expected:
            assert row == expected.pop(row["id"])
    assert expected == {}
    forbidden = {r["id"] for r in old if r["role"] in {"calibration", "test"}}
    assert not {r["id"] for r in rows} & forbidden
    test = [r for r in rows if r["role"] == "test" and r["suite"] == "contrast"]
    assert len(test) == 1920
    assert len({r["template_family"] for r in test}) == 15


def test_label_balance_and_corrupt_group_detection():
    rows = dataset()
    assert all(r["input_tokens"] <= 512 for r in rows)
    changed = copy.deepcopy(rows)
    row = next(r for r in changed if r["id"].startswith("v3/"))
    row["oracle"]["state"] = 1 - row["oracle"]["state"]
    with pytest.raises(ValueError, match="oracle"):
        validate(changed)


def test_retention_guard_overrides_large_aggregate_score():
    a = dict(eligible=False, target_minimum=0.99, mean_partial_rejection_auc=0.99, step=10)
    b = dict(eligible=True, target_minimum=0.30, mean_partial_rejection_auc=0.40, step=20)
    assert choose([a, b]) == b
    assert choose([b, dict(b, step=30)]) == b
    with pytest.raises(ValueError, match="selection"):
        choice([{"role": "test"}], [], {}, 1, {})


def test_counterfactual_pair_direction_and_ties():
    rows = [r for r in dataset() if r["id"].startswith("v3/relation/test/00/000/")]
    result = counterfactual_ordering(rows, {"perfect": [1, 0, 0, 1], "constant": [0] * 4})
    assert result["perfect"]["accuracy_with_half_ties"] == 1
    assert result["constant"]["accuracy_with_half_ties"] == 0.5


def test_lora_roundtrip_base_frozen_and_ranking_gradients():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import BertConfig, BertForSequenceClassification

    from arborvault.support_v3_adapter import (
        attach,
        frozen_hash,
        objective,
        restore,
        trainable_state,
    )

    config = json.loads(Path("configs/support-v3.json").read_text())
    torch.manual_seed(7)
    model = BertForSequenceClassification(
        BertConfig(
            vocab_size=20,
            hidden_size=24,
            intermediate_size=48,
            num_attention_heads=4,
            num_hidden_layers=6,
            num_labels=1,
        )
    ).eval()
    tensors = {"input_ids": torch.tensor([[1, 2, 3], [1, 4, 5]])}
    original = model(**tensors).logits.detach()
    attach(model, config)
    assert torch.equal(model(**tensors).logits, original)  # zero-initialized update
    before = frozen_hash(model)
    initial = trainable_state(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    model(**tensors).logits.square().mean().backward()
    optimizer.step()
    assert frozen_hash(model) == before
    assert any(
        name.endswith(".b") and torch.count_nonzero(p) for name, p in model.named_parameters()
    )
    restore(model, initial)
    assert torch.equal(model(**tensors).logits, original)
    with pytest.raises(ValueError, match="keys"):
        restore(model, {})
    rows = [r for r in dataset() if r["id"].startswith("v3/relation/fit/00/000/")]
    good = torch.tensor([2.0, -2.0, -2.0, 2.0], requires_grad=True)
    bad = -good.detach().clone().requires_grad_(True)
    loss, _, _ = objective(good, rows, config)
    assert loss < objective(bad, rows, config)[0]
    loss.backward()
    assert good.grad.tolist()[0] < 0 and good.grad.tolist()[1] > 0


def test_sampler_is_reproducible_balanced_and_refuses_test():
    pytest.importorskip("torch")
    from arborvault.support_v3_adapter import group_batches, lr_multiplier, ranking_pairs

    config = json.loads(Path("configs/support-v3.json").read_text())
    rows = [r for r in dataset() if r["role"] == "fit"]
    short = dict(config, checkpoints=[8])
    batches = list(group_batches(rows, short))
    assert batches == list(group_batches(rows, short))
    for stratum, ix in batches:
        assert len(ix) == len(set(ix)) == 16
        batch = [rows[i] for i in ix]
        assert sum(r["answerable"] for r in batch) == 8
        assert len(ranking_pairs(batch)) == (0 if stratum == "human" else 8)
    assert np.isclose(lr_multiplier(config["warmup_steps"], config), 1)
    assert np.isclose(lr_multiplier(max(config["checkpoints"]), config), 0)
    with pytest.raises(ValueError, match="fit"):
        list(group_batches([{"role": "test"}], config))
