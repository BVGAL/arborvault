"""Train two candidate scorers on identical pairs and evaluate a frozen development pilot."""

import argparse
import gc
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

from arborvault.candidate_models import CrossEncoderScorer, QwenScorer, fit_text
from arborvault.candidate_scoring import (
    BM25,
    calibrate,
    candidate_text,
    evaluate_scores,
    partition,
    probe_quality,
    probe_queries,
    training_groups,
    views_by_id,
    weighted_binary_loss,
)
from arborvault.qasper import load_split
from arborvault.qwen3 import file_hash, load_model, load_recipe, packages
from arborvault.routing_eval import context, score, summarize


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/candidate-scoring-v1.json"))
    parser.add_argument("--dataset", type=Path, default=Path("datasets/qasper-routing-v2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    fixed = {
        "epochs": 1,
        "weak_negative_mass": 0.25,
        "primary_pair_budget": 64,
        "secondary_pair_budgets": [32, 128],
        "tree_width": 4,
        "k": 5,
        "context_words": 512,
        "max_length": 512,
    }
    if any(config.get(k) != v for k, v in fixed.items()):
        raise ValueError("This version implements only the frozen scoring-v1 protocol")
    _, trees, train = load_split(args.dataset, "train")
    fit, calibration = partition(train)
    probe = probe_queries(calibration)
    groups = training_groups(trees, fit)
    plan = {
        "scope": "Development pilot, known-document retrieval; historical test remains unopened",
        "code_revision": os.getenv("ARBORVAULT_CODE_REVISION", "unknown"),
        "config": config,
        "config_sha256": file_hash(args.config),
        "dataset_manifest_sha256": file_hash(args.dataset / "manifest.json"),
        "training_query_ids": [q["id"] for q in fit],
        "calibration_query_ids": [q["id"] for q in calibration],
        "checkpoint_probe_query_ids": [q["id"] for q in probe],
        "training_groups_sha256": hashlib.sha256(
            json.dumps(groups, sort_keys=True).encode()
        ).hexdigest(),
        "training_pairs": sum(len(g["items"]) for g in groups),
        "positive_pairs": sum(i["label"] for g in groups for i in g["items"]),
        "training_documents": len({q["document"] for q in fit}),
        "calibration_documents": len({q["document"] for q in calibration}),
        "calibration_unknown": sum(not q["answerable"] for q in calibration),
        "steps": math.ceil(len(groups) / config["gradient_accumulation_queries"]),
        "labels": "Union of existing human annotations only; audit proposals are not training labels",
        "weak_negative_note": "Unannotated does not mean irrelevant; mass .25 is a heuristic, not unbiased PU learning",
        "cache_note": "Exhaustive offline score generation; replay accounts for logical candidate/token budgets, not online latency",
    }
    save(args.output / "plan.json", plan)
    if args.prepare_only:
        print(json.dumps({k: v for k, v in plan.items() if not k.endswith("_ids")}))
        return
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, set_seed

    started = time.monotonic()

    def deadline():
        if time.monotonic() - started > config["max_seconds"]:
            raise TimeoutError("Experiment deadline reached")

    def log(event):
        event["elapsed_seconds"] = round(time.monotonic() - started, 3)
        with (args.output / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)

    torch.set_num_threads(4)
    set_seed(config["seed"])
    recipe = load_recipe(Path(config["qwen_recipe"]))
    qt = AutoTokenizer.from_pretrained(
        recipe["base_model"], revision=recipe["base_revision"], trust_remote_code=False
    )
    ct = AutoTokenizer.from_pretrained(
        config["cross_encoder"], revision=config["cross_encoder_revision"], trust_remote_code=False
    )
    save(
        args.output / "environment.json",
        {
            "packages": packages(),
            "gpu": torch.cuda.get_device_name(),
            "base_revision": recipe["base_revision"],
        },
    )
    fitted, fitting = {}, []
    for q in train:
        if q not in calibration:
            continue
        rows = []
        for identity, view in views_by_id(trees[q["document"]]).items():
            text, meta = fit_text(q["question"], candidate_text(view), qt, ct, config["max_length"])
            rows.append({"node_id": identity, "text": text})
            fitting.append(meta)
        fitted[q["id"]] = rows
    for group in groups:
        for item in group["items"]:
            item["text"], meta = fit_text(
                group["question"], item["text"], qt, ct, config["max_length"]
            )
            fitting.append(meta)
    save(
        args.output / "input-fitting.json",
        {
            "pairs": len(fitting),
            "shortened": sum(m["excerpt_characters_removed"] > 0 for m in fitting),
            "max_qwen_tokens": max(m["qwen_tokens"] for m in fitting),
            "max_cross_encoder_tokens": max(m["cross_encoder_tokens"] for m in fitting),
            "fitted_training_groups_sha256": hashlib.sha256(
                json.dumps(groups, sort_keys=True).encode()
            ).hexdigest(),
        },
    )

    def infer(scorer, queries, label):
        cache = {}
        begin = time.monotonic()
        for number, q in enumerate(queries, 1):
            deadline()
            items = fitted[q["id"]]
            values, lengths = scorer.score([q["question"]] * len(items), [i["text"] for i in items])
            if not all(math.isfinite(s) for s in values):
                raise ValueError("Non-finite model score")
            cache[q["id"]] = {
                "scores": dict(zip([i["node_id"] for i in items], values, strict=True)),
                "tokens": dict(zip([i["node_id"] for i in items], lengths, strict=True)),
            }
            if number % 25 == 0 or number == len(queries):
                log(
                    {
                        "stage": "inference",
                        "variant": label,
                        "questions": number,
                        "total": len(queries),
                    }
                )
        return cache, {
            "elapsed_seconds": time.monotonic() - begin,
            "actual_scored_pairs": sum(len(v["scores"]) for v in cache.values()),
            "actual_input_tokens": sum(sum(v["tokens"].values()) for v in cache.values()),
        }

    def train_scorer(scorer, label, lr):
        baseline, _ = infer(scorer, probe, label + "-base-probe")
        candidates = [{"step": 0, "quality": probe_quality(trees, probe, baseline)}]
        save(args.output / "probe" / (label + "-base.json"), baseline)
        optimizer = torch.optim.AdamW(
            [p for p in scorer.model.parameters() if p.requires_grad], lr=lr, weight_decay=0
        )
        order = list(range(len(groups)))
        random.Random(config["seed"]).shuffle(order)
        accumulation = config["gradient_accumulation_queries"]
        running = 0.0
        set_seed(config["seed"])
        for step, offset in enumerate(range(0, len(order), accumulation), 1):
            deadline()
            batch = order[offset : offset + accumulation]
            scorer.model.train()
            optimizer.zero_grad(set_to_none=True)
            for index in batch:
                group = groups[index]
                for start in range(0, len(group["items"]), config["micro_batch_size"]):
                    items = group["items"][start : start + config["micro_batch_size"]]
                    logits, _ = scorer.logits(
                        [group["question"]] * len(items), [i["text"] for i in items]
                    )
                    labels = torch.tensor([i["label"] for i in items], device=logits.device)
                    weights = torch.tensor([i["weight"] for i in items], device=logits.device)
                    loss = weighted_binary_loss(logits, labels, weights)
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite training loss")
                    running += loss.detach().item()
                    (loss / len(batch)).backward()
            torch.nn.utils.clip_grad_norm_(scorer.model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            if step % 5 == 0:
                log(
                    {
                        "stage": "training",
                        "variant": label,
                        "step": step,
                        "steps": plan["steps"],
                        "cumulative_query_bce": running / min(offset + accumulation, len(groups)),
                    }
                )
            if step % config["checkpoint_interval"] == 0 or step == plan["steps"]:
                target = args.output / "checkpoints" / f"{label}-step-{step}"
                scorer.model.save_pretrained(target, safe_serialization=True)
                scorer.tokenizer.save_pretrained(target)
                cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all()
                cache, usage = infer(scorer, probe, label + f"-step-{step}-probe")
                record = {
                    "step": step,
                    "quality": probe_quality(trees, probe, cache),
                    "checkpoint": target.relative_to(args.output).as_posix(),
                    "cumulative_query_bce": running / min(offset + accumulation, len(groups)),
                    "probe_usage": usage,
                }
                candidates.append(record)
                save(args.output / "probe" / f"{label}-step-{step}.json", cache)
                save(args.output / (label + "-training.json"), candidates)
                log({"stage": "checkpoint", "variant": label, **record})
                torch.set_rng_state(cpu)
                torch.cuda.set_rng_state_all(cuda)
        # Report the best trained candidate even when epoch zero is preferable.
        chosen = max(candidates[1:], key=lambda r: (r["quality"], -r["step"]))
        result = {
            "selected_trained": chosen,
            "base_probe": candidates[0],
            "all_checkpoints": candidates,
            "beats_base_on_probe": chosen["quality"] > candidates[0]["quality"],
            "selection_rule": "Mean flat/tree Recall@5, then coverage/MRR on 24 calibration probe questions; earliest step breaks ties",
        }
        save(args.output / (label + "-selection.json"), result)
        del optimizer, loss, logits
        return result

    # Both train with identical query groups, labels, weights and order.
    ce = AutoModelForSequenceClassification.from_pretrained(
        config["cross_encoder"], revision=config["cross_encoder_revision"], trust_remote_code=False
    ).to("cuda")
    ce_selection = train_scorer(
        CrossEncoderScorer(ce, ct, config["cross_encoder_batch_size"]),
        "cross-encoder",
        config["cross_encoder_learning_rate"],
    )
    del ce
    gc.collect()
    torch.cuda.empty_cache()
    model, qt = load_model(recipe)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=recipe["lora_r"],
            lora_alpha=recipe["lora_alpha"],
            lora_dropout=recipe["lora_dropout"],
            target_modules=recipe["target_modules"],
            task_type="CAUSAL_LM",
            bias="none",
        ),
    )
    q_selection = train_scorer(
        QwenScorer(model, qt, config["qwen_batch_size"]), "qwen", config["qwen_learning_rate"]
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    save(
        args.output / "selection-before-development.json",
        {
            "qwen": q_selection,
            "cross_encoder": ce_selection,
            "development_not_evaluated": True,
            "test_not_opened": True,
        },
    )

    # Development labels are loaded only after both checkpoint choices are fixed.
    _, dev_trees, dev = load_split(args.dataset, "dev")
    if set(trees) & set(dev_trees):
        raise ValueError("Development documents overlap training/calibration")
    for q in dev:
        fitted[q["id"]] = [
            {
                "node_id": i,
                "text": fit_text(q["question"], candidate_text(v), qt, ct, config["max_length"])[0],
            }
            for i, v in views_by_id(dev_trees[q["document"]]).items()
        ]
    methods, usages, selections = {}, {}, {}

    def assess(scorer, label):
        cal, usage = infer(scorer, calibration, label + "-calibration")
        save(args.output / "scores" / (label + "-calibration.json"), cal)
        policy = {mode: calibrate(trees, calibration, cal, mode=mode) for mode in ("flat", "tree")}
        selections[label] = policy
        save(args.output / "calibration" / (label + ".json"), policy)
        cache, dev_usage = infer(scorer, dev, label + "-development")
        save(args.output / "scores" / (label + "-dev.json"), cache)
        usages[label] = {"calibration": usage, "development": dev_usage}
        for mode in ("flat", "tree"):
            for budget in (32, 64, 128):
                methods[f"{label}/{mode}/{budget}"] = evaluate_scores(
                    dev_trees, dev, cache, mode=mode, budget=budget
                )
            methods[f"{label}/{mode}/64-calibrated"] = evaluate_scores(
                dev_trees, dev, cache, mode=mode, threshold=policy[mode]["chosen"]["threshold"]
            )

    for label, source, revision in (
        ("cross-encoder-base", config["cross_encoder"], config["cross_encoder_revision"]),
        (
            "cross-encoder-tuned",
            str(args.output / ce_selection["selected_trained"]["checkpoint"]),
            None,
        ),
    ):
        ce = AutoModelForSequenceClassification.from_pretrained(
            source, revision=revision, trust_remote_code=False
        ).to("cuda")
        assess(CrossEncoderScorer(ce, ct, config["cross_encoder_batch_size"]), label)
        del ce
        gc.collect()
        torch.cuda.empty_cache()
    model, qt = load_model(recipe)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = PeftModel.from_pretrained(
        model, args.output / q_selection["selected_trained"]["checkpoint"], is_trainable=False
    )
    scorer = QwenScorer(model, qt, config["qwen_batch_size"])
    with model.disable_adapter():
        assess(scorer, "qwen-base")
    assess(scorer, "qwen-tuned")
    del scorer, model
    gc.collect()
    torch.cuda.empty_cache()

    # Stronger dense baseline and lexical/hybrid references; no new supervision.
    et = AutoTokenizer.from_pretrained(
        config["embedding_model"], revision=config["embedding_revision"]
    )
    em = (
        AutoModel.from_pretrained(
            config["embedding_model"],
            revision=config["embedding_revision"],
            trust_remote_code=False,
        )
        .to("cuda")
        .eval()
    )

    def embed(texts):
        vectors = []
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
                h = em(**inputs).last_hidden_state.masked_fill(
                    ~inputs["attention_mask"][..., None].bool(), 0
                )
                vectors.append(
                    torch.nn.functional.normalize(
                        h.sum(1) / inputs["attention_mask"].sum(1)[..., None], dim=1
                    )
                )
        return torch.cat(vectors)

    baseline_rows = {"dense_e5_base": [], "bm25": [], "hybrid_rrf": []}
    document_vectors = {}
    for q in dev:
        tree = dev_trees[q["document"]]
        leaves = [n for n in tree.nodes.values() if not tree.children[n.id]]
        if q["document"] not in document_vectors:
            # Fitting depends on question length, so use stable full views for dense indexing.
            texts = [candidate_text(v) for v in tree.views(leaves)]
            document_vectors[q["document"]] = (embed(["passage: " + t for t in texts]), BM25(texts))
        vectors, lexical = document_vectors[q["document"]]
        dense = (vectors @ embed(["query: " + q["question"]])[0]).tolist()
        lex = lexical.scores(q["question"])
        orders = {
            "dense_e5_base": sorted(range(len(leaves)), key=lambda i: (-dense[i], i)),
            "bm25": sorted(range(len(leaves)), key=lambda i: (-lex[i], i)),
        }
        fusion = [0.0] * len(leaves)
        for ranking in orders.values():
            for rank, i in enumerate(ranking, 1):
                fusion[i] += 1 / (60 + rank)
        orders["hybrid_rrf"] = sorted(range(len(leaves)), key=lambda i: (-fusion[i], i))
        for label, ranking in orders.items():
            selected = context([leaves[i] for i in ranking])
            baseline_rows[label].append(
                {
                    "id": q["id"],
                    "document": q["document"],
                    "answerable": q["answerable"],
                    "predicted": [n.id for n, _ in selected],
                    **score(selected, q),
                }
            )
    methods.update({name: summarize(rows) for name, rows in baseline_rows.items()})
    from summarize_routing_v2 import clustered_delta

    comparisons = {}
    for family in ("qwen", "cross-encoder"):
        for mode in ("flat", "tree"):
            comparisons[f"{family}/{mode}/tuning"] = clustered_delta(
                methods[f"{family}-base/{mode}/64"]["queries"],
                methods[f"{family}-tuned/{mode}/64"]["queries"],
            )
        for variant in ("base", "tuned"):
            name = family + "-" + variant
            comparisons[name + "/tree-minus-flat"] = clustered_delta(
                methods[name + "/flat/64"]["queries"], methods[name + "/tree/64"]["queries"]
            )
            comparisons[name + "/flat-minus-e5"] = clustered_delta(
                methods["dense_e5_base"]["queries"], methods[name + "/flat/64"]["queries"]
            )
    save(
        args.output / "results.json",
        {
            "methods": methods,
            "paired_document_bootstrap": comparisons,
            "calibration": selections,
            "usage": usages,
            "scope": plan["scope"],
            "development_queries": len(dev),
        },
    )
    save(
        args.output / "complete.json",
        {
            "complete": True,
            "elapsed_seconds": time.monotonic() - started,
            "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
            "test_not_opened": True,
            "new_training_seeds": [config["seed"]],
        },
    )
    log({"stage": "complete"})
    save(
        args.output / "artifact-manifest.json",
        {
            p.relative_to(args.output).as_posix(): file_hash(p)
            for p in sorted(args.output.rglob("*"))
            if p.is_file()
        },
    )


if __name__ == "__main__":
    main()
