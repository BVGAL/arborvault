"""Common abstention gates, untouched human transfer and template-held-out diagnostics."""

import json
from pathlib import Path

import numpy as np

from .support_data import file_hash
from .support_probe import metrics, write_json
from .support_v2_data import load
from .support_v2_features import base_cache, read_role
from .support_v2_models import calibrate_common, fit_candidates, linear, mlp_checkpoints, scores


def fit_models(config, rows, features):
    if set(rows) != {"fit", "selection", "calibration"} or set(features) != set(rows):
        raise ValueError("Fitting must not receive test data")
    models, selection = fit_candidates(
        rows["fit"],
        features["fit"]["context"],
        rows["selection"],
        features["selection"]["context"],
        config,
    )
    previous = Path(config["previous_probe"])
    if file_hash(previous) != config["previous_probe_sha256"]:
        raise ValueError("Previous classifier changed")
    old = json.loads(previous.read_text(encoding="utf-8"))["fitted_probes"]["context"]
    models["previous"] = {"kind": "linear", "parameters": 385, "probe": old}
    chosen = models["selected"]
    y = [r["answerable"] for r in rows["fit"]]
    if chosen["kind"] == "linear":
        control = linear(
            features["fit"]["question_only"],
            y,
            [r["id"] for r in rows["fit"]],
            chosen["probe"]["l2"],
        )
    else:
        control_config = dict(config, mlp_checkpoints=[chosen["epoch"]])
        control = mlp_checkpoints(features["fit"]["question_only"], y, control_config)[
            chosen["epoch"]
        ]
    models["question_only"] = control
    models["context_only"] = {"kind": "constant", "value": 0.5, "parameters": 0}
    calibration_scores = score_models(models, features["calibration"])
    thresholds = {
        name: calibrate_common(rows["calibration"], values, config)
        for name, values in calibration_scores.items()
    }
    return {
        "models": models,
        "selection": selection,
        "thresholds": thresholds,
        "question_control": "Same architecture and chosen hyperparameters as selected head; question-only input",
        "context_control": "Constant chance, because each context has opposite labels by construction",
        "test_used_for_fitting": False,
        "previous_coefficients_refitted": False,
    }


def score_models(models, features):
    result = {
        name: scores(model, features["question_only" if name == "question_only" else "context"])
        for name, model in models.items()
    }
    result["raw"] = features["raw"]
    return result


def bootstrap(rows, all_scores, thresholds, config, suite):
    group_key = "document" if suite == "human" else "template_family"
    subset = [(i, r) for i, r in enumerate(rows) if r["suite"] == suite]
    units = sorted({r[group_key] for _, r in subset})
    indices = [[i for i, r in subset if r[group_key] == unit] for unit in units]
    y = np.array([r["answerable"] for r in rows], dtype=bool)
    rng = np.random.default_rng(config["bootstrap_seed"])
    sampled = rng.integers(0, len(units), size=(config["bootstrap_replicates"], len(units)))
    result = {}
    for point in ("conservative", "balanced"):
        distributions, observed = {}, {}
        for name, values in all_scores.items():
            accept = np.asarray(values) >= thresholds[name][point]["threshold"]
            counts = np.array(
                [
                    [
                        np.sum(~y[ix] & ~accept[ix]),
                        np.sum(y[ix] & ~accept[ix]),
                        np.sum(~y[ix]),
                        np.sum(y[ix]),
                    ]
                    for ix in indices
                ]
            )
            total = counts.sum(axis=0)
            drawn = counts[sampled].sum(axis=1)
            distributions[name] = {
                "unsupported_rejection": drawn[:, 0] / drawn[:, 2],
                "supported_rejection": drawn[:, 1] / drawn[:, 3],
            }
            distributions[name]["balanced_accuracy"] = (
                1
                - distributions[name]["supported_rejection"]
                + distributions[name]["unsupported_rejection"]
            ) / 2
            observed[name] = {
                "unsupported_rejection": float(total[0] / total[2]),
                "supported_rejection": float(total[1] / total[3]),
                "balanced_accuracy": float((1 - total[1] / total[3] + total[0] / total[2]) / 2),
            }
        result[point] = {
            "method_intervals": {
                name: {
                    metric: np.quantile(v, [0.025, 0.975]).tolist() for metric, v in dist.items()
                }
                for name, dist in distributions.items()
            },
            "selected_minus": {
                baseline: {
                    metric: {
                        "estimate": observed["selected"][metric] - observed[baseline][metric],
                        "ci95": np.quantile(
                            distributions["selected"][metric] - distributions[baseline][metric],
                            [0.025, 0.975],
                        ).tolist(),
                    }
                    for metric in observed["selected"]
                }
                for baseline in ("previous", "raw", "linear")
            },
        }
    return {
        "unit": group_key,
        "clusters": len(units),
        "replicates": config["bootstrap_replicates"],
        "seed": config["bootstrap_seed"],
        "operating_points": result,
    }


def evaluate_test(config, rows, values, fitted):
    if {r["role"] for r in rows} != {"test"}:
        raise ValueError("Evaluation requires test rows")
    thresholds = fitted["thresholds"]
    result = {}
    for cohort in ("human", "contrast", "negation", "relation", "temporal"):
        indices = [
            i for i, r in enumerate(rows) if r["suite"] == cohort or r["phenomenon"] == cohort
        ]
        subset = [rows[i] for i in indices]
        result[cohort] = {
            name: {
                point: metrics(subset, [v[i] for i in indices], spec["threshold"])
                for point, spec in thresholds[name].items()
            }
            for name, v in values.items()
        }
    decisions = [
        {
            "id": r["id"],
            "document": r["document"],
            "suite": r["suite"],
            "phenomenon": r["phenomenon"],
            "template_family": r.get("template_family"),
            "supported": r["answerable"],
            "scores": {name: float(v[i]) for name, v in values.items()},
            "accepted": {
                name: {
                    point: bool(v[i] >= spec["threshold"])
                    for point, spec in thresholds[name].items()
                }
                for name, v in values.items()
            },
        }
        for i, r in enumerate(rows)
    ]
    summary = {
        "selected_candidate": fitted["selection"]["selected_candidate"],
        "cohorts": result,
        "bootstrap": {
            suite: bootstrap(rows, values, thresholds, config, suite)
            for suite in ("human", "contrast")
        },
    }
    return summary, decisions


def analyze(config, feature_dir, output, config_path):
    if output.exists():
        raise ValueError("Refusing to overwrite a completed analysis")
    _, all_rows = load(Path(config["dataset"]))
    manifest = json.loads((feature_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["config_sha256"] != file_hash(config_path) or manifest[
        "dataset_manifest_sha256"
    ] != file_hash(Path(config["dataset"]) / "manifest.json"):
        raise ValueError("Dataset or configuration changed after extraction")
    cache, sources = base_cache(config)
    if sources != manifest["sources"]:
        raise ValueError("Reused feature sources changed")
    rows, features = {}, {}
    for role in ("fit", "selection", "calibration"):
        rows[role], features[role] = read_role(feature_dir, all_rows, role, cache)
    fitted = fit_models(config, rows, features)
    output.mkdir(parents=True)
    write_json(output / "fitted-models.json", fitted)
    print(json.dumps(fitted["selection"]), flush=True)
    # Freeze chosen model and gates before opening any test features.
    test, test_features = read_role(feature_dir, all_rows, "test", cache)
    values = score_models(fitted["models"], test_features)
    summary, decisions = evaluate_test(config, test, values, fitted)
    summary["provenance"] = {
        "config_sha256": file_hash(config_path),
        "feature_manifest_sha256": file_hash(feature_dir / "manifest.json"),
        "fitted_models_sha256": file_hash(output / "fitted-models.json"),
    }
    write_json(output / "decisions.json", decisions, compact=True)
    write_json(output / "summary.json", summary)
    return summary
