"""Post-hoc frozen-head diagnostics; no model or threshold changes."""

import json
from pathlib import Path

import numpy as np

from arborvault.support_probe import auc, write_json
from arborvault.support_v2_data import load
from arborvault.support_v2_eval import score_models
from arborvault.support_v2_features import base_cache, read_role

config = json.loads(Path("configs/support-v2.json").read_text())
root = Path("docs/benchmarks/support-v2")
fitted = json.loads((root / "results/fitted-models.json").read_text())
_, all_rows = load(Path(config["dataset"]))
base, _ = base_cache(config)
development = {}
family = {}
pair_order = {}
for role in ["fit", "selection", "calibration", "test"]:
    rows, features = read_role(root / "features", all_rows, role, base)
    values = score_models(fitted["models"], features)
    development[role] = {}
    for phenomenon in config["strata"]:
        indices = [i for i, r in enumerate(rows) if r["phenomenon"] == phenomenon]
        labels = np.array([rows[i]["answerable"] for i in indices])
        development[role][phenomenon] = {}
        for name in ["selected", "previous", "linear", "raw"]:
            v = np.array([values[name][i] for i in indices])
            accept = v >= fitted["thresholds"][name]["conservative"]["threshold"]
            development[role][phenomenon][name] = {
                "auroc": auc(labels, v),
                "unsupported_rejected": int((~labels & ~accept).sum()),
                "supported_rejected": int((labels & ~accept).sum()),
                "unsupported": int((~labels).sum()),
                "supported": int(labels.sum()),
            }
    if role != "test":
        continue
    for name in sorted({r["template_family"] for r in rows if r["suite"] == "contrast"}):
        indices = [i for i, r in enumerate(rows) if r.get("template_family") == name]
        labels = np.array([rows[i]["answerable"] for i in indices])
        family[name] = {}
        for method in ["selected", "previous", "linear", "raw"]:
            v = np.array([values[method][i] for i in indices])
            accept = v >= fitted["thresholds"][method]["conservative"]["threshold"]
            family[name][method] = {
                "auroc": auc(labels, v),
                "unsupported_rejected": int((~labels & ~accept).sum()),
                "supported_rejected": int((labels & ~accept).sum()),
                "unsupported": int((~labels).sum()),
                "supported": int(labels.sum()),
            }
    for phenomenon in ["negation", "relation", "temporal"]:
        groups = {r["group_id"] for r in rows if r["phenomenon"] == phenomenon}
        pair_order[phenomenon] = {}
        for method in ["selected", "previous", "linear", "raw"]:
            margins = []
            for group in sorted(groups):
                indices = [i for i, r in enumerate(rows) if r["group_id"] == group]
                for qv in [0, 1]:
                    ix = [i for i in indices if rows[i]["question_variant"] == qv]
                    pos = next(i for i in ix if rows[i]["answerable"])
                    neg = next(i for i in ix if not rows[i]["answerable"])
                    margins.append(values[method][pos] - values[method][neg])
            pair_order[phenomenon][method] = {
                "questions": len(margins),
                "correct_ordering_fraction": float(
                    np.mean([1 if x > 0 else 0.5 if x == 0 else 0 for x in margins])
                ),
                "mean_score_margin": float(np.mean(margins)),
            }
result = {
    "scope": "Post-hoc diagnosis only; no head, epoch, threshold, label or ranking changed",
    "by_partition": development,
    "test_template_families": family,
    "same_question_counterfactual_ordering": pair_order,
}
write_json(root / "diagnostics.json", result)
print(json.dumps(result, indent=2))
