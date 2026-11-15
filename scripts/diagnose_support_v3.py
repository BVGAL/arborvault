"""Post-hoc family breakdown and deterministic illustrative audit, no tuning."""

import json
from pathlib import Path

from arborvault.support_data import digest
from arborvault.support_probe import metrics, write_json
from arborvault.support_v2_data import load
from arborvault.support_v3_eval import counterfactual_ordering
from arborvault.support_v3_training import read


def main():
    root = Path("docs/benchmarks/support-v3")
    _, rows = load(Path("datasets/support-v3"))
    test = [r for r in rows if r["role"] == "test"]
    decisions = read(root / "results/decisions.json")
    if [r["id"] for r in test] != [d["id"] for d in decisions]:
        raise ValueError("Test order differs")
    thresholds = read(root / "results/calibration.json")["thresholds"]
    families = {}
    for family in sorted({r["template_family"] for r in test if r["suite"] == "contrast"}):
        indices = [i for i, r in enumerate(test) if r.get("template_family") == family]
        subset = [test[i] for i in indices]
        values = {name: [decisions[i]["scores"][name] for i in indices] for name in thresholds}
        families[family] = {
            "metrics": {
                name: metrics(subset, v, thresholds[name]["conservative"]["threshold"])
                for name, v in values.items()
            },
            "counterfactual_ordering": counterfactual_ordering(subset, values),
        }
    audit = []
    for phenomenon in ("human", "negation", "relation", "temporal"):
        for label in (True, False):
            for error in (True, False):
                pool = [
                    i
                    for i, r in enumerate(test)
                    if r["phenomenon"] == phenomenon
                    and r["answerable"] == label
                    and (decisions[i]["accepted"]["selected"]["conservative"] != label) == error
                ]
                for index in sorted(
                    pool, key=lambda i: digest("support-v3-audit/" + test[i]["id"])
                )[:2]:
                    row, decision = test[index], decisions[index]
                    audit.append(
                        {
                            "id": row["id"],
                            "phenomenon": phenomenon,
                            "error": error,
                            "question": row["question"],
                            "context": row["context"],
                            "supported": label,
                            "answers": row["answers"],
                            "template_family": row.get("template_family"),
                            "scores": decision["scores"],
                            "accepted": decision["accepted"],
                        }
                    )
    calibration = read(root / "results/calibration.json")
    row_by_id = {r["id"]: r for r in rows}
    lowest = sorted(
        [
            (value, row_by_id[identifier])
            for identifier, value in zip(
                calibration["ids"], calibration["scores"]["selected"], strict=True
            )
            if row_by_id[identifier]["answerable"]
        ],
        key=lambda item: item[0],
    )[:6]
    write_json(
        root / "diagnostics.json",
        {
            "post_hoc": True,
            "no_model_or_threshold_change": True,
            "families": families,
            "selected_conservative_threshold": calibration["thresholds"]["selected"][
                "conservative"
            ]["threshold"],
            "lowest_supported_calibration_scores": [
                {
                    "id": row["id"],
                    "phenomenon": row["phenomenon"],
                    "template_family": row.get("template_family"),
                    "question": row["question"],
                    "context": row["context"],
                    "score": value,
                }
                for value, row in lowest
            ],
        },
    )
    write_json(
        root / "error-audit.json",
        {
            "selection": "Two hash-selected illustrations per phenomenon x support label x error/success cell when available; not a prevalence sample or independent human adjudication",
            "labels_changed": False,
            "examples": audit,
        },
    )
    print(json.dumps({"families": len(families), "illustrative_examples": len(audit)}))


if __name__ == "__main__":
    main()
