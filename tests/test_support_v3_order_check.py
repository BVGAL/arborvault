import json
from collections import defaultdict
from pathlib import Path

from arborvault.support_data import file_hash


def test_order_addendum_counterbalances_shortcut_without_relabeling():
    original = {
        r["id"]: r
        for r in map(
            json.loads,
            Path("datasets/support-v3/examples.jsonl").read_text(encoding="utf-8").splitlines(),
        )
    }
    path = Path("datasets/support-v3-order-check/examples.jsonl")
    alternate = list(map(json.loads, path.read_text(encoding="utf-8").splitlines()))
    manifest = json.loads(path.with_name("manifest.json").read_text(encoding="utf-8"))
    assert file_hash(path) == manifest["examples_sha256"]
    assert (
        file_hash(Path("datasets/support-v3/examples.jsonl"))
        == manifest["original_examples_sha256"]
    )
    assert len(alternate) == 640
    counts, paired = defaultdict(list), []
    for row in alternate:
        old = original[row["id"].removesuffix("/order-alternate")]
        assert old["role"] == row["role"] == "test"
        for key in ("context", "answerable", "answers", "group_id", "entities", "template_family"):
            assert row[key] == old[key]
        assert row["question"] != old["question"]
        assert row["input_tokens"] <= 512
        paired.extend([old, row])
        counts[(row["group_id"], row["question"])].append(row["answerable"])
    assert all(sorted(labels) == [False, True] for labels in counts.values())
    cells = defaultdict(list)
    for row in paired:
        a, b = row["entities"]
        context_order = row["context"].index(a) < row["context"].index(b)
        question_order = row["question"].index(a) < row["question"].index(b)
        cells[(row["template_family"], context_order == question_order)].append(row["answerable"])
    assert len(cells) == 10
    assert all(sum(labels) == len(labels) / 2 for labels in cells.values())
