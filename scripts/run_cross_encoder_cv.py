"""Frozen five-fold document CV; no outer labels select weights or abstention rules."""

import argparse
import gc
import json
import math
import os
import random
import time
from importlib.metadata import version
from pathlib import Path

from arborvault.candidate_models import CrossEncoderScorer, QwenScorer, fit_text
from arborvault.candidate_scoring import (
    candidate_text,
    training_groups,
    views_by_id,
    weighted_binary_loss,
)
from arborvault.cross_validation import (
    candidates,
    features,
    fingerprint,
    fit_gate,
    fold_plan,
    gate_scores,
    pool,
    retrieval_rows,
    tune_threshold,
)
from arborvault.qwen3 import file_hash, load_model, load_recipe, packages
from arborvault.routing_eval import summarize


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cross-encoder-cv-v1.json"))
    parser.add_argument("--dataset", type=Path, default=Path("datasets/qasper-routing-v2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    fixed = {
        "outer_folds": 5,
        "inner_document_proportions": [0.6, 0.1, 0.15, 0.15],
        "epochs": 1,
        "candidate_budget": 64,
        "max_length": 512,
        "gate_l2": 0.01,
        "calibration_recall_loss_guard": 0.02,
        "calibration_false_abstention_guard": 0.02,
    }
    if any(config[k] != v for k, v in fixed.items()):
        raise ValueError("Unsupported change to frozen CV protocol")
    source = json.loads(Path(config["scorer_config"]).read_text())
    trees, queries, historical = pool(args.dataset)
    by_id = {q["id"]: q for q in queries}
    folds = fold_plan(queries)
    for fold in folds:
        fold["counts"] = {
            role: {
                "questions": len(ids),
                "documents": len({by_id[i]["document"] for i in ids}),
                "unknown": sum(not by_id[i]["answerable"] for i in ids),
            }
            for role, ids in fold["roles"].items()
        }
        if any(
            c["unknown"] < 5 or c["questions"] - c["unknown"] < 5 for c in fold["counts"].values()
        ):
            raise ValueError("Insufficient classes in an inner or outer partition")
    plan = {
        "code_revision": os.getenv("ARBORVAULT_CODE_REVISION", "unknown"),
        "config": config,
        "scorer_config": source,
        "config_sha256": file_hash(args.config),
        "dataset_sha256": file_hash(args.dataset / "manifest.json"),
        "folds": folds,
        "questions": len(queries),
        "documents": len(trees),
        "answerable": sum(q["answerable"] for q in queries),
        "historical_split": historical,
        "exact_document_content_duplicates": 0,
        "scope": "Retrospective pooled train/dev/test CV; historical test is no longer untouched for this experiment",
    }
    save(args.output / "plan.json", plan)
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "questions": len(queries),
                    "documents": len(trees),
                    "fold_counts": [f["counts"] for f in folds],
                }
            )
        )
        return
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, set_seed

    started = time.monotonic()
    torch.set_num_threads(4)
    set_seed(config["seed"])

    def deadline():
        if time.monotonic() - started > config["max_seconds"]:
            raise TimeoutError("CV process deadline reached")

    def log(**event):
        event["elapsed_seconds"] = round(time.monotonic() - started, 3)
        with (args.output / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)

    recipe = load_recipe(Path(source["qwen_recipe"]))
    qt = AutoTokenizer.from_pretrained(
        recipe["base_model"], revision=recipe["base_revision"], trust_remote_code=False
    )
    ct = AutoTokenizer.from_pretrained(
        source["cross_encoder"], revision=source["cross_encoder_revision"], trust_remote_code=False
    )
    save(
        args.output / "environment.json",
        {
            "packages": {**packages(), "numpy": version("numpy")},
            "gpu": torch.cuda.get_device_name(),
            "qwen_revision": recipe["base_revision"],
        },
    )
    fitted, chosen_ids, fitting = {}, {}, []
    for q in queries:
        deadline()
        tree = trees[q["document"]]
        fitted[q["id"]] = {}
        chosen_ids[q["id"]] = candidates(tree, q["question"])
        for identity, view in views_by_id(tree).items():
            text, meta = fit_text(q["question"], candidate_text(view), qt, ct)
            fitted[q["id"]][identity] = text
            fitting.append(meta)
    save(
        args.output / "input-fitting.json",
        {
            "pairs": len(fitting),
            "shortened": sum(m["excerpt_characters_removed"] > 0 for m in fitting),
            "max_qwen_tokens": max(m["qwen_tokens"] for m in fitting),
            "max_cross_encoder_tokens": max(m["cross_encoder_tokens"] for m in fitting),
            "fitted_sha256": fingerprint(fitted),
            "candidate_ids_sha256": fingerprint(chosen_ids),
        },
    )

    def infer(scorer, subset, label):
        cache = {}
        begin = time.monotonic()
        for number, q in enumerate(subset, 1):
            deadline()
            ids = chosen_ids[q["id"]]
            values, lengths = scorer.score(
                [q["question"]] * len(ids), [fitted[q["id"]][i] for i in ids]
            )
            if not all(math.isfinite(v) for v in values):
                raise ValueError("Non-finite inference score")
            cache[q["id"]] = {
                "scores": dict(zip(ids, values, strict=True)),
                "tokens": dict(zip(ids, lengths, strict=True)),
            }
            if number % 25 == 0 or number == len(subset):
                log(stage="inference", variant=label, questions=number, total=len(subset))
        return cache, {
            "elapsed_seconds": time.monotonic() - begin,
            "pairs": sum(len(v["scores"]) for v in cache.values()),
            "tokens": sum(sum(v["tokens"].values()) for v in cache.values()),
        }

    # Fixed base references are computed once, without labels, and shared across folds.
    model, qt = load_model(recipe)
    qwen, usage = infer(QwenScorer(model, qt, 8), queries, "qwen-base")
    save(args.output / "scores/qwen-base.json", qwen)
    usages = {"qwen-base": usage}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    ce = AutoModelForSequenceClassification.from_pretrained(
        source["cross_encoder"], revision=source["cross_encoder_revision"], trust_remote_code=False
    ).to("cuda")
    base, usage = infer(CrossEncoderScorer(ce, ct), queries, "cross-encoder-base")
    save(args.output / "scores/cross-encoder-base.json", base)
    usages["cross-encoder-base"] = usage
    del ce
    gc.collect()
    torch.cuda.empty_cache()

    et = AutoTokenizer.from_pretrained(
        source["embedding_model"], revision=source["embedding_revision"], trust_remote_code=False
    )
    encoder = (
        AutoModel.from_pretrained(
            source["embedding_model"],
            revision=source["embedding_revision"],
            trust_remote_code=False,
        )
        .to("cuda")
        .eval()
    )
    e5_started = time.monotonic()

    def embed(texts, model=encoder):
        result = []
        with torch.inference_mode():
            for offset in range(0, len(texts), 32):
                deadline()
                inputs = et(
                    texts[offset : offset + 32],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to("cuda")
                hidden = model(**inputs).last_hidden_state.masked_fill(
                    ~inputs["attention_mask"][..., None].bool(), 0
                )
                result.append(
                    torch.nn.functional.normalize(
                        hidden.sum(1) / inputs["attention_mask"].sum(1)[..., None], dim=1
                    )
                )
        return torch.cat(result)

    indexed, e5 = {}, {}
    for number, q in enumerate(queries, 1):
        tree = trees[q["document"]]
        if q["document"] not in indexed:
            leaves = [n for n in tree.nodes.values() if not tree.children[n.id]]
            vectors = embed(
                ["passage: " + " / ".join(n.path[1:]) + "\n" + n.content for n in leaves]
            )
            indexed[q["document"]] = ([n.id for n in leaves], vectors)
        ids, vectors = indexed[q["document"]]
        values = (vectors @ embed(["query: " + q["question"]])[0]).tolist()
        e5[q["id"]] = {"scores": dict(zip(ids, values, strict=True))}
        if number % 50 == 0:
            log(stage="inference", variant="e5-text", questions=number, total=len(queries))
    save(args.output / "scores/e5-text.json", e5)
    usages["e5-text"] = {
        "elapsed_seconds": time.monotonic() - e5_started,
        "indexed_documents": len(indexed),
        "scored_passages": sum(len(v["scores"]) for v in e5.values()),
    }
    del embed, encoder, indexed, vectors
    gc.collect()
    torch.cuda.empty_cache()
    save(args.output / "reference-usage.json", usages)

    for fold in folds:
        number = fold["fold"]
        directory = args.output / f"fold-{number}"
        roles = {role: [by_id[i] for i in ids] for role, ids in fold["roles"].items()}
        groups = training_groups(trees, roles["fit"])
        for group in groups:
            for item in group["items"]:
                item["text"] = fitted[group["id"]][item["node_id"]]
        save(
            directory / "training-plan.json",
            {
                "groups_sha256": fingerprint(groups),
                "queries": len(groups),
                "pairs": sum(len(g["items"]) for g in groups),
                "seed": config["seed"],
                "fit_ids": fold["roles"]["fit"],
            },
        )
        set_seed(config["seed"])
        model = AutoModelForSequenceClassification.from_pretrained(
            source["cross_encoder"],
            revision=source["cross_encoder_revision"],
            trust_remote_code=False,
        ).to("cuda")
        scorer = CrossEncoderScorer(model, ct)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["learning_rate"], weight_decay=0
        )
        order = list(range(len(groups)))
        random.Random(config["seed"]).shuffle(order)
        interval, accumulation = (
            config["checkpoint_interval"],
            config["gradient_accumulation_queries"],
        )
        total = math.ceil(len(order) / accumulation)
        history, loss_sum = [], 0.0

        def checkpoint(step, cache, selection_queries=roles["selection"], target_dir=directory):
            report = summarize(retrieval_rows(trees, selection_queries, cache))
            quality = [report[k] for k in ("recall", "evidence_coverage", "mrr")]
            save(target_dir / f"probe/step-{step}.json", cache)
            return {"step": step, "quality": quality}

        history.append(checkpoint(0, {i: base[i] for i in fold["roles"]["selection"]}))
        for step, offset in enumerate(range(0, len(order), accumulation), 1):
            deadline()
            batch = order[offset : offset + accumulation]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for index in batch:
                group = groups[index]
                for start in range(0, len(group["items"]), config["micro_batch_size"]):
                    items = group["items"][start : start + config["micro_batch_size"]]
                    logits, _ = scorer.logits(
                        [group["question"]] * len(items), [i["text"] for i in items]
                    )
                    loss = weighted_binary_loss(
                        logits,
                        torch.tensor([i["label"] for i in items], device="cuda"),
                        torch.tensor([i["weight"] for i in items], device="cuda"),
                    )
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite training loss")
                    loss_sum += loss.detach().item()
                    (loss / len(batch)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            if step % 10 == 0:
                log(stage="training", fold=number, step=step, steps=total)
            if step % interval == 0 or step == total:
                target = directory / f"checkpoints/step-{step}"
                model.save_pretrained(target, safe_serialization=True)
                ct.save_pretrained(target)
                cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all()
                cache, _ = infer(scorer, roles["selection"], f"fold-{number}-probe-{step}")
                record = checkpoint(step, cache)
                record["cumulative_query_bce"] = loss_sum / min(offset + accumulation, len(groups))
                history.append(record)
                save(directory / "training.json", history)
                torch.set_rng_state(cpu)
                torch.cuda.set_rng_state_all(cuda)
        chosen = max(history[1:], key=lambda r: (r["quality"], -r["step"]))
        save(
            directory / "selection.json",
            {
                "selected": chosen,
                "history": history,
                "rule": "Flat selection-set recall, then evidence coverage/MRR, earliest tie; best trained checkpoint reported alongside base",
            },
        )
        del scorer, model, optimizer, logits, loss
        gc.collect()
        torch.cuda.empty_cache()
        model = AutoModelForSequenceClassification.from_pretrained(
            directory / f"checkpoints/step-{chosen['step']}", trust_remote_code=False
        ).to("cuda")
        scorer = CrossEncoderScorer(model, ct)
        inner = roles["gate_fit"] + roles["threshold"]
        trained, inner_usage = infer(scorer, inner, f"fold-{number}-calibration")
        methods = {
            "cross-encoder-tuned": trained,
            "cross-encoder-base": base,
            "qwen-base": qwen,
            "e5-text": e5,
        }
        locks = {}
        for name, cache in methods.items():
            gate_x = [features(trees[q["document"]], q, cache[q["id"]]) for q in roles["gate_fit"]]
            gate = fit_gate(gate_x, [q["answerable"] for q in roles["gate_fit"]])
            threshold_x = [
                features(trees[q["document"]], q, cache[q["id"]]) for q in roles["threshold"]
            ]
            rows = retrieval_rows(trees, roles["threshold"], cache)
            locks[name] = {
                "gate": gate,
                "max_score": tune_threshold(rows, [x[0] for x in threshold_x]),
                "learned_gate": tune_threshold(rows, gate_scores(gate, threshold_x)),
            }
        # Save all choices before evaluating this fold's held-out questions.
        save(
            directory / "locked-before-evaluation.json",
            {
                "selection": chosen,
                "policies": locks,
                "evaluation_metrics_not_used": True,
                "evaluation_ids": fold["roles"]["evaluation"],
            },
        )
        evaluation, outer_usage = infer(scorer, roles["evaluation"], f"fold-{number}-evaluation")
        trained.update(evaluation)
        save(directory / "scores.json", trained)
        save(
            directory / "usage.json",
            {"inner_inference": inner_usage, "outer_inference": outer_usage},
        )
        del model, scorer
        gc.collect()
        torch.cuda.empty_cache()
        log(stage="fold_complete", fold=number, selected_step=chosen["step"])
    save(
        args.output / "complete.json",
        {
            "complete": True,
            "folds": 5,
            "elapsed_seconds": time.monotonic() - started,
            "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
            "outer_performance_not_used_for_selection": True,
        },
    )
    log(stage="complete")
    save(
        args.output / "artifact-manifest.json",
        {
            p.relative_to(args.output).as_posix(): file_hash(p)
            for p in sorted(args.output.rglob("*"))
            if p.is_file() and p.name != "artifact-manifest.json"
        },
    )


if __name__ == "__main__":
    main()
