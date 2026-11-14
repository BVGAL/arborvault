import json
from pathlib import Path

import numpy as np
import pytest

from arborvault.answerability import fit_evidence, fit_probe, probe_scores, validate_partitions
from arborvault.answerability_analysis import joint_gate, tune_joint
from arborvault.long_documents import load_corpus


def test_duplicate_contexts_do_not_overweight_a_training_question():
    x = [[-2, 1], [-1, 0], [1, 0], [2, 1]]
    y = [0, 0, 1, 1]
    first = fit_probe(x, y, ["a", "b", "c", "d"], l2=0.1)
    second = fit_probe(x + x[:1] * 5, y + y[:1] * 5, ["a", "b", "c", "d"] + ["a"] * 5, l2=0.1)
    assert np.allclose(probe_scores(first, x), probe_scores(second, x), atol=1e-7)
    assert probe_scores(first, [[-3, 0]])[0] < 0.1
    assert probe_scores(first, [[3, 0]])[0] > 0.9


def test_document_partitions_fail_closed():
    history = [{"document": "old1", "role": "fit"}, {"document": "old2", "role": "selection"}]
    queries = [{"document": "new1", "split": "calibration"}, {"document": "new2", "split": "test"}]
    validate_partitions(history, queries)
    with pytest.raises(ValueError, match="crosses"):
        validate_partitions(history, queries + [{"document": "old1", "split": "test"}])


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def num_special_tokens_to_add(self, pair):
        return 3

    def prepare_for_model(self, ids, pair_ids, **kwargs):
        return {"input_ids": ["CLS", *ids, "SEP", *pair_ids, "SEP"]}


def test_verifier_preserves_question_and_never_sees_unreturned_text():
    fitted, shortened = fit_evidence(WordTokenizer(), "which value", "one two three four", 8)
    assert fitted["input_ids"] == ["CLS", "which", "value", "SEP", "one", "two", "three", "SEP"]
    assert shortened
    with pytest.raises(ValueError, match="Question exceeds"):
        fit_evidence(WordTokenizer(), "one two three four five", "context", 8)


def test_joint_calibration_bounds_total_false_rejections():
    rows, raw, learned = [], [], []
    for i in range(150):
        positive = i < 100
        rows.append(
            dict(
                id=str(i),
                answerable=positive,
                predicted=["p"],
                abstained=False,
                recall=float(positive),
                mrr=float(positive),
                ndcg=float(positive),
                evidence_coverage=float(positive),
            )
        )
        raw.append(0 if i < 2 or i >= 125 else 1)
        learned.append(0 if 2 <= i < 4 or i >= 100 else 1)
    chosen = tune_joint(rows, raw, learned)
    result = joint_gate(
        rows, raw, learned, chosen["raw_threshold"], chosen["conservative"]["threshold"]
    )
    assert sum(r["abstained"] for r in result if r["answerable"]) <= 2
    assert all(r["abstained"] for r in result[125:])


def test_frozen_expanded_corpus_is_disjoint_and_balanced():
    manifest, trees, queries = load_corpus(Path("datasets/abstention-v3"))
    assert manifest["counts"]["test"]["answerable"] == 240
    assert manifest["counts"]["test"]["unknown"] == 240
    old = set()
    for source in ("long-manuals-v1", "navigation-v2"):
        old.update(
            d["id"]
            for d in json.loads(
                (Path("datasets") / source / "documents.json").read_text(encoding="utf-8")
            )
        )
    assert not old & set(trees)
    assert len(trees) == 24
    for q in queries:
        if q["category"] == "undocumented_api_assumption":
            body = " ".join(n.content for n in trees[q["document"]].nodes.values()).lower()
            assert q["audit"]["absent_api"].rsplit(".", 1)[-1].lower() not in body


def test_probe_rejects_nonfinite_features():
    with pytest.raises(ValueError, match="Invalid"):
        fit_probe([[float("nan")], [1]], [0, 1], ["a", "b"], l2=0.1)
