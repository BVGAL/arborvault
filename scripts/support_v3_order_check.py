"""Pre-test addendum: counterbalance entity order without changing model/gates."""

import argparse
import json
from pathlib import Path

import numpy as np

from arborvault.support_contrasts import ASSETS
from arborvault.support_data import file_hash
from arborvault.support_probe import metrics, write_json
from arborvault.support_v2_data import load, validate
from arborvault.support_v3_eval import counterfactual_ordering, paired_bootstrap
from arborvault.support_v3_training import baseline_scores, infer_selected, read, runtime

QUESTIONS = (
    "Which document identifies {sender} as the origin of the {asset} delivery accepted by {receiver}?",
    "Which ledger reference records {sender}'s stock supplying {asset} to {receiver}'s stock?",
    "Which mailing receipt names {receiver} as the addressee of {asset} mailed by {sender}?",
    "Which rental identifies {sender} as owner and {receiver} as renter of {asset}?",
    "Which movement identifier lists {sender} as outgoing unit and {receiver} as incoming unit for {asset}?",
)


def entity_order_matches(row):
    a, b = row["entities"]
    return (row["context"].index(a) < row["context"].index(b)) == (
        row["question"].index(a) < row["question"].index(b)
    )


def build(config, output):
    from transformers import AutoTokenizer

    if output.exists():
        raise ValueError("Refusing to overwrite order-check data")
    _, rows = load(Path(config["dataset"]))
    original = [r for r in rows if r["role"] == "test" and r["phenomenon"] == "relation"]
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"], revision=config["revision"], local_files_only=True
    )
    alternate = []
    for row in original:
        index = int(row["template_family"].rsplit("/", 1)[1])
        a, b = row["entities"]
        requested = row["question_variant"]
        assets = [asset for asset in ASSETS if asset in row["context"].lower()]
        if len(assets) != 1:
            raise ValueError("Ambiguous synthetic asset")
        question = QUESTIONS[index].format(
            sender=[a, b][requested], receiver=[b, a][requested], asset=assets[0]
        )
        new = dict(row, id=row["id"] + "/order-alternate", question=question)
        new["input_tokens"] = len(
            tokenizer(question, row["context"], truncation=False)["input_ids"]
        )
        alternate.append(new)
    # Reuse complete validator with unchanged non-test roles; alternate replaces
    # original relation test rows, never appears in fitting or selection.
    validate([r for r in rows if r not in original] + alternate)
    assert all(entity_order_matches(r) != r["answerable"] for r in original)
    assert all(entity_order_matches(r) == r["answerable"] for r in alternate)
    combined = original + alternate
    shortcut = sum(entity_order_matches(r) == r["answerable"] for r in combined) / len(combined)
    assert shortcut == 0.5
    output.mkdir(parents=True)
    path = output / "examples.jsonl"
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in alternate),
        encoding="utf-8",
        newline="\n",
    )
    write_json(
        output / "manifest.json",
        dict(
            examples_sha256=file_hash(path),
            original_examples_sha256=file_hash(Path(config["dataset"]) / "examples.jsonl"),
            examples=len(alternate),
            original_examples=len(original),
            combined_order_only_accuracy=shortcut,
            added_before_primary_test_scoring=True,
            development_checkpoints_already_observed=[300, 600],
            license="Apache-2.0",
            independent_adjudication=False,
        ),
    )
    return alternate


def evaluate(config, data, root):
    from arborvault.support_v3_adapter import predict

    manifest = read(data / "manifest.json")
    assert file_hash(data / "examples.jsonl") == manifest["examples_sha256"]
    assert (
        file_hash(Path(config["dataset"]) / "examples.jsonl")
        == manifest["original_examples_sha256"]
    )
    rows = [
        json.loads(x) for x in (data / "examples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    base, tokenizer = runtime(config)
    raw, vectors = predict(base, tokenizer, rows, config, lambda: None, features=True)
    values = baseline_scores(read(root / "training/baselines.json")["models"], vectors, raw)
    selected, _ = infer_selected(config, root / "training")
    predicted, _ = predict(selected, tokenizer, rows, config, lambda: None)
    values["selected"] = predicted.tolist()
    values["v2_original_gate"] = values["v2"]
    thresholds = read(root / "results/calibration.json")["thresholds"]
    primary = [d for d in read(root / "results/decisions.json") if d["phenomenon"] == "relation"]
    _, all_rows = load(Path(config["dataset"]))
    original = [r for r in all_rows if r["role"] == "test" and r["phenomenon"] == "relation"]
    assert [d["id"] for d in primary] == [r["id"] for r in original]
    combined_values = {name: [d["scores"][name] for d in primary] + v for name, v in values.items()}
    summaries = {}
    for label, examples, val in [
        ("alternate_order", rows, values),
        ("order_balanced_union", original + rows, combined_values),
    ]:
        summaries[label] = {
            "metrics": {
                name: {
                    point: metrics(examples, val[name], spec["threshold"])
                    for point, spec in gates.items()
                }
                for name, gates in thresholds.items()
            },
            "counterfactual_ordering": counterfactual_ordering(examples, val),
            "bootstrap": paired_bootstrap(examples, val, thresholds, config),
        }
    current = summaries["alternate_order"]["metrics"]["selected"]["conservative"]
    old = summaries["alternate_order"]["metrics"]["v2"]["conservative"]
    checks = {
        "alternate_useful_rejection_at_most_2pct": current["supported_rejected"]
        / current["supported"]
        <= 0.02,
        "alternate_unsupported_rejection_improves_10pp": current["unsupported_rejection"]
        - old["unsupported_rejection"]
        >= 0.10,
    }
    result = dict(
        summaries=summaries,
        additional_research_checks=checks,
        additional_checks_passed=all(checks.values()),
        prediction_ids=[r["id"] for r in rows],
        scores=values,
        thresholds_changed=False,
        provenance={
            "data_sha256": manifest["examples_sha256"],
            "calibration_sha256": file_hash(root / "results/calibration.json"),
            "adapter_sha256": file_hash(root / "training/adapter.json"),
        },
        original_order_only_accuracy=1.0,
        combined_order_only_accuracy=0.5,
        caution="Supplementary test preregistered during training after two development checkpoints; not in the initial frozen protocol. Five family clusters; same author, no independent adjudication.",
    )
    # The 100% original shortcut predicts support when name order differs.
    assert np.isfinite(np.array(list(values.values()))).all()
    write_json(root / "order-check-results.json", result)
    print(json.dumps(checks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "evaluate"])
    parser.add_argument("--data", type=Path, default=Path("datasets/support-v3-order-check"))
    args = parser.parse_args()
    config = read(Path("configs/support-v3.json"))
    if args.action == "build":
        rows = build(config, args.data)
        print(
            json.dumps({"new_rows": len(rows), "label_balance": sum(r["answerable"] for r in rows)})
        )
    else:
        evaluate(config, args.data, Path("docs/benchmarks/support-v3"))


if __name__ == "__main__":
    main()
