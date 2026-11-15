"""Common gates and a separate test, with paired article/family uncertainty."""

import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .support_data import file_hash
from .support_probe import metrics, write_json
from .support_v2_data import load
from .support_v2_models import calibrate_common
from .support_v3_training import baseline_scores, infer_selected, read, runtime


def paired_bootstrap(rows, values, thresholds, config):
    key = "document" if rows[0]["suite"] == "human" else "template_family"
    units = sorted({r[key] for r in rows})
    ix = [[i for i, r in enumerate(rows) if r[key] == unit] for unit in units]
    y = np.array([r["answerable"] for r in rows])
    rng = np.random.default_rng(config["bootstrap_seed"])
    sample = rng.integers(len(units), size=(config["bootstrap_replicates"], len(units)))
    distributions, observed = {}, {}
    for name, score in values.items():
        reject = np.asarray(score) < thresholds[name]["conservative"]["threshold"]
        counts = np.array(
            [[sum(reject[j] & ~y[j]), sum(reject[j] & y[j]), sum(~y[j]), sum(y[j])] for j in ix]
        )
        total, drawn = counts.sum(axis=0), counts[sample].sum(axis=1)
        observed[name] = {
            "unsupported_rejection": float(total[0] / total[2]),
            "supported_rejection": float(total[1] / total[3]),
        }
        distributions[name] = {
            "unsupported_rejection": drawn[:, 0] / drawn[:, 2],
            "supported_rejection": drawn[:, 1] / drawn[:, 3],
        }
    return dict(
        unit=key,
        clusters=len(units),
        replicates=config["bootstrap_replicates"],
        method_intervals={
            name: {k: np.quantile(v, [0.025, 0.975]).tolist() for k, v in d.items()}
            for name, d in distributions.items()
        },
        selected_minus={
            name: {
                k: {
                    "estimate": observed["selected"][k] - observed[name][k],
                    "ci95": np.quantile(
                        distributions["selected"][k] - distributions[name][k], [0.025, 0.975]
                    ).tolist(),
                }
                for k in observed[name]
            }
            for name in ("v1", "v2", "frozen_v3", "raw")
        },
    )


def counterfactual_ordering(rows, values):
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        if r["suite"] == "contrast":
            groups[(r["group_id"], r["question"])].append(i)
    result = {}
    for name, score in values.items():
        wins, ties = 0, 0
        for pair in groups.values():
            positive = next(i for i in pair if rows[i]["answerable"])
            negative = next(i for i in pair if not rows[i]["answerable"])
            wins += score[positive] > score[negative]
            ties += score[positive] == score[negative]
        result[name] = {
            "pairs": len(groups),
            "wins": int(wins),
            "ties": int(ties),
            "accuracy_with_half_ties": float((wins + 0.5 * ties) / len(groups)) if groups else None,
        }
    return result


def summarize(config, rows, values, thresholds, selection_eligible):
    if {r["role"] for r in rows} != {"test"}:
        raise ValueError("Only test rows may enter final evaluation")
    cohorts, intervals, ordering = {}, {}, {}
    for cohort in ("human", "contrast", "negation", "relation", "temporal"):
        indices = [
            i for i, r in enumerate(rows) if r["suite"] == cohort or r["phenomenon"] == cohort
        ]
        subset = [rows[i] for i in indices]
        scores = {name: [v[i] for i in indices] for name, v in values.items()}
        cohorts[cohort] = {
            name: {
                point: metrics(subset, scores[name], spec["threshold"])
                for point, spec in gates.items()
            }
            for name, gates in thresholds.items()
        }
        intervals[cohort] = paired_bootstrap(subset, scores, thresholds, config)
        if cohort != "human":
            ordering[cohort] = counterfactual_ordering(subset, scores)
    checks = {"development_retention_guards": selection_eligible}
    for phenomenon in config["strata"]:
        current = cohorts[phenomenon]["selected"]["conservative"]
        checks[phenomenon + "_useful_rejection_at_most_2pct"] = (
            current["supported_rejected"] / current["supported"] <= 0.02
        )
        baseline = cohorts[phenomenon]["v1" if phenomenon == "human" else "v2"]["conservative"]
        tolerance = {"human": -0.02, "temporal": -0.05, "negation": 0.10, "relation": 0.10}[
            phenomenon
        ]
        checks[phenomenon + "_unsupported_rejection_target"] = (
            current["unsupported_rejection"] - baseline["unsupported_rejection"] >= tolerance
        )
    return dict(
        cohorts=cohorts,
        bootstrap=intervals,
        counterfactual_ordering=ordering,
        observed_promotion_checks=checks,
        observed_promotion_passed=all(checks.values()),
        caution="One seed and 5 template families per phenomenon. Gates are observed research criteria, not a production certification. Human source is public SQuAD, not live traffic.",
    )


def evaluate(config, config_path, training_dir, output, deadline):
    from .support_v3_adapter import predict

    if output.exists():
        raise ValueError("Refusing to overwrite evaluation")
    output.mkdir(parents=True)
    started = time.monotonic()
    dataset, rows = load(Path(config["dataset"]))
    training = read(training_dir / "training.json")
    if (
        training["config_sha256"] != file_hash(config_path)
        or training["examples_sha256"] != dataset["examples_sha256"]
    ):
        raise ValueError("Training inputs changed")
    calibration = [r for r in rows if r["role"] == "calibration"]
    baselines = read(training_dir / "baselines.json")
    base, tokenizer = runtime(config)
    raw, vectors = predict(base, tokenizer, calibration, config, deadline, features=True)
    calibration_values = baseline_scores(baselines["models"], vectors, raw)
    selected, _ = infer_selected(config, training_dir)
    selected_values, _ = predict(selected, tokenizer, calibration, config, deadline)
    calibration_values["selected"] = selected_values.tolist()
    thresholds = {
        name: calibrate_common(calibration, value, config)
        for name, value in calibration_values.items()
    }
    # An unchanged v2 operating point is also reported, separate from fair recalibration.
    thresholds["v2_original_gate"] = baselines["original_v2_gates"]
    calibration_values["v2_original_gate"] = calibration_values["v2"]
    artifact = dict(
        ids=[r["id"] for r in calibration],
        scores=calibration_values,
        thresholds=thresholds,
        adapter_sha256=file_hash(training_dir / "adapter.json"),
        training_sha256=file_hash(training_dir / "training.json"),
    )
    write_json(output / "calibration.json", artifact)
    # Only after persisting the model and every threshold do we score the new test.
    test = [r for r in rows if r["role"] == "test"]
    raw, vectors = predict(base, tokenizer, test, config, deadline, features=True)
    values = baseline_scores(baselines["models"], vectors, raw)
    selected_values, _ = predict(selected, tokenizer, test, config, deadline)
    values["selected"] = selected_values.tolist()
    values["v2_original_gate"] = values["v2"]
    summary = summarize(config, test, values, thresholds, training["selected"]["eligible"])
    decisions = [
        dict(
            id=r["id"],
            supported=r["answerable"],
            phenomenon=r["phenomenon"],
            document=r["document"],
            template_family=r.get("template_family"),
            scores={name: float(v[i]) for name, v in values.items()},
            accepted={
                name: {
                    point: bool(values[name][i] >= spec["threshold"])
                    for point, spec in gates.items()
                }
                for name, gates in thresholds.items()
            },
        )
        for i, r in enumerate(test)
    ]
    write_json(output / "decisions.json", decisions, compact=True)
    # Input-only ablations: duplicates have opposite labels; any deterministic
    # scorer must be chance on controlled cases. This checks actual inference.
    controls = {}
    contrasts = [r for r in test if r["suite"] == "contrast"]
    from .support_probe import auc

    for mode in ("question_only", "context_only"):
        predictions, _ = predict(selected, tokenizer, contrasts, config, deadline, mode=mode)
        controls[mode] = dict(
            ids=[r["id"] for r in contrasts],
            scores=predictions.tolist(),
            auroc=auc([r["answerable"] for r in contrasts], predictions),
        )
    write_json(output / "input-controls.json", controls, compact=True)
    summary["input_control_aurocs"] = {name: value["auroc"] for name, value in controls.items()}
    summary["provenance"] = {
        "config_sha256": file_hash(config_path),
        "examples_sha256": dataset["examples_sha256"],
        "calibration_sha256": file_hash(output / "calibration.json"),
        "adapter_sha256": file_hash(training_dir / "adapter.json"),
    }
    write_json(output / "summary.json", summary)
    write_json(
        output / "runtime.json",
        {"seconds": time.monotonic() - started, "device": "cpu", "new_cloud_spend_usd": 0},
    )
    return summary
