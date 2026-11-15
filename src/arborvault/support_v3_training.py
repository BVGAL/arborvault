"""CPU adapter training with development-only retention guards and checkpoints."""

import json
import time
from pathlib import Path

import numpy as np

from .support_data import file_hash
from .support_probe import model_input, write_json
from .support_v2_data import load
from .support_v2_models import mlp_checkpoints, scores, select_score


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def runtime(config):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"], revision=config["revision"], local_files_only=True
    )
    tokenizer.padding_side = "right"
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            config["model"],
            revision=config["revision"],
            local_files_only=True,
            use_safetensors=True,
            trust_remote_code=False,
            attn_implementation="eager",
        )
        .cpu()
        .eval()
    )
    return model, tokenizer


def old_models(config):
    path = Path(config["previous_models"])
    if file_hash(path) != config["previous_models_sha256"]:
        raise ValueError("Previous models changed")
    old = read(path)
    return {"v1": old["models"]["previous"], "v2": old["models"]["selected"]}, old["thresholds"][
        "selected"
    ]


def baseline_scores(models, vectors, raw):
    return {**{name: scores(model, vectors) for name, model in models.items()}, "raw": raw.tolist()}


def choice(rows, values, references, step, config):
    stats = select_score(rows, values, config)
    strata = stats["strata"]
    tolerance = config["selection_guard_tolerance"]
    guards = {
        "temporal": strata["temporal"] >= references["v2"]["strata"]["temporal"] - tolerance,
        "human": strata["human"] >= references["v1"]["strata"]["human"] - tolerance,
    }
    return {
        "step": step,
        **stats,
        "retention_guards": guards,
        "eligible": all(guards.values()),
        "target_minimum": min(strata["negation"], strata["relation"]),
    }


def choose(candidates):
    if not candidates:
        raise ValueError("No development candidates")
    return max(
        candidates,
        key=lambda c: (
            c["eligible"],
            round(c["target_minimum"], 12),
            round(c["mean_partial_rejection_auc"], 12),
            -c["step"],
        ),
    )


def train(config, config_path, output, work, deadline):
    import torch
    import transformers

    from .support_v3_adapter import (
        attach,
        frozen_hash,
        group_batches,
        lr_multiplier,
        objective,
        predict,
        trainable_state,
    )

    if output.exists():
        raise ValueError("Refusing to overwrite training output")
    output.mkdir(parents=True)
    work.mkdir(parents=True, exist_ok=True)
    (work / "checkpoints").mkdir()
    started = time.monotonic()
    dataset, all_rows = load(Path(config["dataset"]))
    fit = [r for r in all_rows if r["role"] == "fit"]
    selection = [r for r in all_rows if r["role"] == "selection"]
    # Test and calibration never enter the optimization or checkpoint choice.
    model, tokenizer = runtime(config)
    models, original_gates = old_models(config)
    _, fit_vectors = predict(model, tokenizer, fit, config, deadline, features=True)
    dev_raw, dev_vectors = predict(model, tokenizer, selection, config, deadline, features=True)
    values = baseline_scores(models, dev_vectors, dev_raw)
    references = {name: select_score(selection, score, config) for name, score in values.items()}
    heads = mlp_checkpoints(fit_vectors, [r["answerable"] for r in fit], config)
    head_candidates = [
        choice(selection, scores(head, dev_vectors), references, epoch, config)
        for epoch, head in heads.items()
    ]
    best_head = choose(head_candidates)
    models["frozen_v3"] = heads[best_head["step"]]
    values["frozen_v3"] = scores(models["frozen_v3"], dev_vectors)
    write_json(
        output / "baselines.json",
        {
            "models": models,
            "original_v2_gates": original_gates,
            "frozen_v3_candidates": head_candidates,
            "frozen_v3_selected": best_head,
        },
    )
    write_json(
        output / "selection-baselines.json",
        {"ids": [r["id"] for r in selection], "scores": values, "metrics": references},
        compact=True,
    )
    del fit_vectors, dev_vectors

    torch.manual_seed(config["seed"])
    targets = attach(model, config)
    frozen_before = frozen_hash(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    encoded = [tokenizer(*model_input(r, "context"), truncation=False) for r in fit]
    if any(len(x["input_ids"]) > config["max_length"] for x in encoded):
        raise ValueError("Training would truncate an input")
    history, candidates, exposure, dev_scores = [], [], {s: 0 for s in config["strata"]}, {}
    training_start = time.monotonic()
    for step, (stratum, indices) in enumerate(group_batches(fit, config), 1):
        deadline()
        model.train()
        batch = [fit[i] for i in indices]
        inputs = tokenizer.pad([encoded[i] for i in indices], padding=True, return_tensors="pt")
        lr = config["learning_rate"] * lr_multiplier(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        logits = model(**inputs).logits[:, 0]
        loss, bce, ranking = objective(logits, batch, config)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable, config["gradient_clip"], error_if_nonfinite=True
        )
        optimizer.step()
        exposure[stratum] += len(batch)
        if step == 1 or step % 25 == 0:
            record = dict(
                step=step,
                stratum=stratum,
                loss=loss.item(),
                bce=bce.item(),
                ranking=ranking.item(),
                lr=lr,
                grad_norm=float(grad_norm),
                seconds=time.monotonic() - training_start,
            )
            history.append(record)
            write_json(output / "training-history.json", history)
            print(json.dumps(record), flush=True)
        if step in config["checkpoints"]:
            path = work / "checkpoints" / f"step-{step}.json"
            write_json(path, trainable_state(model), compact=True)
            predictions, _ = predict(model, tokenizer, selection, config, deadline)
            candidate = choice(selection, predictions, references, step, config)
            candidate["weights_sha256"] = file_hash(path)
            candidates.append(candidate)
            dev_scores[str(step)] = predictions.tolist()
            write_json(
                output / "selection.json",
                {
                    "candidates": candidates,
                    "ids": [r["id"] for r in selection],
                    "scores": dev_scores,
                },
            )
            print(json.dumps({"checkpoint": candidate}), flush=True)
    best = choose(candidates)
    weights = work / "checkpoints" / f"step-{best['step']}.json"
    (output / "adapter.json").write_bytes(weights.read_bytes())
    frozen_after = frozen_hash(model)
    if frozen_before != frozen_after:
        raise ValueError("Frozen base encoder weights changed")
    result = dict(
        config_sha256=file_hash(config_path),
        dataset_manifest_sha256=file_hash(Path(config["dataset"]) / "manifest.json"),
        examples_sha256=dataset["examples_sha256"],
        selected=best,
        selected_adapter_sha256=file_hash(output / "adapter.json"),
        trainable_parameters=sum(p.numel() for p in trainable),
        frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
        frozen_before=frozen_before,
        frozen_after=frozen_after,
        targets=targets,
        fit_ids=[r["id"] for r in fit],
        selection_ids=[r["id"] for r in selection],
        training_exposures=exposure,
        optimizer_steps=max(config["checkpoints"]),
        seed=config["seed"],
        torch=torch.__version__,
        transformers=transformers.__version__,
        numpy=np.__version__,
        device="cpu",
        seconds=time.monotonic() - started,
        calibration_or_test_used_for_training=False,
        limitations="One seed. Frozen-head comparator differs in objective/sampling; not an isolated causal ablation of encoder adaptation.",
    )
    write_json(output / "training.json", result)
    return result


def infer_selected(config, training_dir):
    from .support_v3_adapter import attach, restore

    model, tokenizer = runtime(config)
    attach(model, config)
    manifest = read(training_dir / "training.json")
    if file_hash(training_dir / "adapter.json") != manifest["selected_adapter_sha256"]:
        raise ValueError("Adapter hash mismatch")
    restore(model, read(training_dir / "adapter.json"))
    return model, tokenizer
