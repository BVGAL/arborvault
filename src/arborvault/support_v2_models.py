"""Small support heads and development-only selection for low-error abstention."""

import copy

import numpy as np

from .answerability import fit_probe, probe_scores


def partial_rejection_auc(labels, scores, limit=0.05):
    """Normalized ROC area: reject unsupported examples at useful-rejection <= limit."""
    y, s = np.asarray(labels, dtype=bool), np.asarray(scores, dtype=float)
    if len(y) != len(s) or not np.isfinite(s).all() or not y.any() or y.all() or not 0 < limit <= 1:
        raise ValueError("Invalid partial ROC inputs")
    order = np.argsort(s, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(s[order])) + 1, len(y)]
    fp = np.r_[0.0, np.cumsum(y[order])[ends - 1] / y.sum()]
    tp = np.r_[0.0, np.cumsum(~y[order])[ends - 1] / (~y).sum()]
    before = fp < limit
    x, values = np.r_[fp[before], limit], np.r_[tp[before], np.interp(limit, fp, tp)]
    return float(np.trapezoid(values, x) / limit)


def select_score(rows, scores, config):
    if {r["role"] for r in rows} != {"selection"}:
        raise ValueError("Candidate choice requires selection data only")
    result = {}
    for stratum in config["strata"]:
        indices = [i for i, r in enumerate(rows) if r["phenomenon"] == stratum]
        result[stratum] = partial_rejection_auc(
            [rows[i]["answerable"] for i in indices],
            [scores[i] for i in indices],
            config["selection_low_fpr_limit"],
        )
    return {"strata": result, "mean_partial_rejection_auc": float(np.mean(list(result.values())))}


def linear(x, y, ids, l2):
    return {"kind": "linear", "parameters": len(x[0]) + 1, "probe": fit_probe(x, y, ids, l2=l2)}


def scores(model, x):
    x = np.asarray(x, dtype=float)
    if model["kind"] == "linear":
        return probe_scores(model["probe"], x)
    if model["kind"] == "constant":
        return [model["value"]] * len(x)
    if (
        model["kind"] != "mlp"
        or x.ndim != 2
        or x.shape[1] != len(model["mean"])
        or not np.isfinite(x).all()
    ):
        raise ValueError("Invalid support-head input")
    z = (x - model["mean"]) / model["scale"]
    h = np.tanh(z @ np.asarray(model["w1"]) + model["b1"])
    logits = h @ np.asarray(model["w2"]) + model["b2"]
    return (1 / (1 + np.exp(-np.clip(logits, -700, 700)))).tolist()


def mlp_objective_gradient(x, y, parameters, l2):
    w1, b1, w2, b2 = parameters
    hidden = np.tanh(x @ w1 + b1)
    logits = hidden @ w2 + b2
    p = 1 / (1 + np.exp(-np.clip(logits, -700, 700)))
    residual = (p - y) / len(y)
    delta = residual[:, None] * w2 * (1 - hidden**2)
    loss = np.mean(np.logaddexp(0, logits) - y * logits) + 0.5 * l2 * (
        np.sum(w1 * w1) + np.sum(w2 * w2)
    )
    gradients = [
        x.T @ delta + l2 * w1,
        delta.sum(axis=0),
        hidden.T @ residual + l2 * w2,
        np.array([residual.sum()]),
    ]
    return float(loss), gradients


def mlp_checkpoints(x, y, config):
    """Full-batch float64 Adam; checkpoints fixed before inspecting test outcomes."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 2 or len(x) != len(y) or set(y) != {0.0, 1.0} or not np.isfinite(x).all():
        raise ValueError("Invalid MLP training data")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = (x - mean) / scale
    rng = np.random.default_rng(config["seed"])
    hidden = config["mlp_hidden"]
    params = [
        rng.normal(0, 1 / np.sqrt(x.shape[1]), (x.shape[1], hidden)),
        np.zeros(hidden),
        rng.normal(0, 1 / np.sqrt(hidden), hidden),
        np.zeros(1),
    ]
    first, second = [np.zeros_like(v) for v in params], [np.zeros_like(v) for v in params]
    checkpoints, history = {}, []
    for epoch in range(1, max(config["mlp_checkpoints"]) + 1):
        loss, gradients = mlp_objective_gradient(z, y, params, config["mlp_l2"])
        if not np.isfinite(loss) or any(not np.isfinite(g).all() for g in gradients):
            raise ValueError("Non-finite MLP optimization")
        for i, gradient in enumerate(gradients):
            first[i] = 0.9 * first[i] + 0.1 * gradient
            second[i] = 0.999 * second[i] + 0.001 * gradient**2
            params[i] -= (
                config["mlp_learning_rate"]
                * (first[i] / (1 - 0.9**epoch))
                / (np.sqrt(second[i] / (1 - 0.999**epoch)) + 1e-8)
            )
        if epoch % 5 == 0 or epoch == 1:
            current_loss, _ = mlp_objective_gradient(z, y, params, config["mlp_l2"])
            history.append({"epoch": epoch, "training_objective_after_update": current_loss})
        if epoch in config["mlp_checkpoints"]:
            checkpoints[epoch] = {
                "kind": "mlp",
                "parameters": sum(v.size for v in params),
                "mean": mean.tolist(),
                "scale": scale.tolist(),
                **{
                    key: v.tolist() for key, v in zip(("w1", "b1", "w2", "b2"), params, strict=True)
                },
                "epoch": epoch,
                "hidden": hidden,
                "l2": config["mlp_l2"],
                "learning_rate": config["mlp_learning_rate"],
                "seed": config["seed"],
                "training_examples": len(y),
                "history": copy.deepcopy(history),
            }
    return checkpoints


def fit_candidates(fit_rows, fit_x, selection_rows, selection_x, config):
    if {r["role"] for r in fit_rows} != {"fit"} or {r["role"] for r in selection_rows} != {
        "selection"
    }:
        raise ValueError("Fitting and candidate selection require their own partitions")
    y = [r["answerable"] for r in fit_rows]
    candidates = []
    for l2 in config["linear_l2_candidates"]:
        model = linear(fit_x, y, [r["id"] for r in fit_rows], l2)
        candidates.append(
            {
                "id": f"linear-l2-{l2}",
                "kind": "linear",
                "model": model,
                "selection": select_score(selection_rows, scores(model, selection_x), config),
            }
        )
    for epoch, model in mlp_checkpoints(fit_x, y, config).items():
        candidates.append(
            {
                "id": f"mlp-epoch-{epoch}",
                "kind": "mlp",
                "model": model,
                "selection": select_score(selection_rows, scores(model, selection_x), config),
            }
        )

    def key(c):
        model = c["model"]
        regularization = model["probe"]["l2"] if model["kind"] == "linear" else model["l2"]
        return (
            round(c["selection"]["mean_partial_rejection_auc"], 12),
            -model["parameters"],
            regularization,
            -model.get("epoch", 0),
        )

    selected = max(candidates, key=key)
    models = {
        kind: max((c for c in candidates if c["kind"] == kind), key=key)["model"]
        for kind in ("linear", "mlp")
    }
    models["selected"] = selected["model"]
    summaries = [{k: v for k, v in c.items() if k != "model"} for c in candidates]
    return models, {"selected_candidate": selected["id"], "candidates": summaries}


def calibrate_common(rows, values, config):
    if {r["role"] for r in rows} != {"calibration"}:
        raise ValueError("Gate fitting requires calibration data only")
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite calibration scores")
    thresholds = np.r_[np.unique(values), np.nextafter(max(values), np.inf)]
    accepted = values[:, None] >= thresholds
    statistics = {}
    for stratum in config["strata"]:
        pos = np.array([r["phenomenon"] == stratum and r["answerable"] for r in rows])
        neg = np.array([r["phenomenon"] == stratum and not r["answerable"] for r in rows])
        if not pos.any() or not neg.any():
            raise ValueError("Calibration stratum lacks either label")
        false, reject = (~accepted[pos]).mean(axis=0), (~accepted[neg]).mean(axis=0)
        statistics[stratum] = {
            "false_rejection": false,
            "unsupported_rejection": reject,
            "balanced_accuracy": (1 - false + reject) / 2,
        }
    false = np.mean([s["false_rejection"] for s in statistics.values()], axis=0)
    reject = np.mean([s["unsupported_rejection"] for s in statistics.values()], axis=0)
    balanced = (1 - false + reject) / 2
    allowed = np.all(
        [
            s["false_rejection"] <= config["conservative_max_false_rejection_rate"]
            for s in statistics.values()
        ],
        axis=0,
    )
    conservative = max(
        np.flatnonzero(allowed), key=lambda i: (reject[i], -false[i], -thresholds[i])
    )
    balanced_index = max(
        range(len(thresholds)), key=lambda i: (balanced[i], -false[i], -thresholds[i])
    )
    return {
        point: {
            "threshold": float(thresholds[i]),
            "calibration_strata": {
                name: {key: float(array[i]) for key, array in stats.items()}
                for name, stats in statistics.items()
            },
            "macro_balanced_accuracy": float(balanced[i]),
        }
        for point, i in (("conservative", conservative), ("balanced", balanced_index))
    }
