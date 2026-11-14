"""Frozen inference-only navigation ablation, with equal charged pair budgets."""

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path

from arborvault.budget_navigation import ContentGuidance, budgeted_traverse
from arborvault.candidate_models import CrossEncoderScorer
from arborvault.decision_scoring import DecisionScorer, fit_decision_text
from arborvault.long_benchmark import DocumentIndex, traverse
from arborvault.long_documents import load_corpus
from arborvault.qwen3 import file_hash, load_model, load_recipe, packages


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/navigation-v2.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest, trees, queries = load_corpus(Path(config["dataset"]))
    source = json.loads(Path(config["scorer_config"]).read_text(encoding="utf-8"))
    if config.get("answerability_history"):
        from arborvault.answerability import historical_examples, validate_partitions

        history, locks = historical_examples(config["answerability_history"])
        validate_partitions(history, queries)
        if locks != config["history_locks"]:
            raise ValueError("Historical inputs changed before inference")
    args.output.mkdir(parents=True, exist_ok=False)
    save(
        args.output / "plan.json",
        {
            "source_revision": os.getenv("ARBORVAULT_CODE_REVISION", "unknown"),
            "config": config,
            "scorer_config": source,
            "config_sha256": file_hash(args.config),
            "dataset_manifest_sha256": file_hash(Path(config["dataset"]) / "manifest.json"),
            "dataset_counts": manifest["counts"],
            "calibration_ids": [q["id"] for q in queries if q["split"] == "calibration"],
            "test_ids": [q["id"] for q in queries if q["split"] == "test"],
        },
    )
    if args.prepare_only:
        print(json.dumps(manifest["counts"]))
        return
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, set_seed

    started = time.monotonic()
    torch.set_num_threads(4)
    set_seed(config["seed"])
    indexes = {d: DocumentIndex(t) for d, t in trees.items()}
    lexical = {q["id"]: indexes[q["document"]].scores(q["question"]) for q in queries}
    guidance = {q["id"]: ContentGuidance(indexes[q["document"]], lexical[q["id"]]) for q in queries}
    save(
        args.output / "environment.json",
        {"packages": packages(), "gpu": torch.cuda.get_device_name()},
    )

    def deadline():
        if time.monotonic() - started > config["max_seconds"]:
            raise TimeoutError("Inference deadline reached")

    def log(**event):
        event["elapsed_seconds"] = round(time.monotonic() - started, 3)
        with (args.output / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)

    recipe = load_recipe(Path(source["qwen_recipe"]))
    qt = AutoTokenizer.from_pretrained(
        recipe["base_model"], revision=recipe["base_revision"], trust_remote_code=False
    )
    models = config.get("models", ["qwen-base", "tacit-4b", "cross-encoder-base"])
    tt = (
        AutoTokenizer.from_pretrained(
            config["tacit_model"], revision=config["tacit_revision"], trust_remote_code=False
        )
        if "tacit-4b" in models
        else qt
    )
    ct = AutoTokenizer.from_pretrained(
        source["cross_encoder"], revision=source["cross_encoder_revision"]
    )
    fitted_cache, usage = {}, {}

    def infer(scorer, label):
        begin, count, tokens, shortened = time.monotonic(), 0, 0, 0
        cache = {}
        for number, q in enumerate(queries, 1):
            index, guide = indexes[q["document"]], guidance[q["id"]]
            entry = {"scores": {}, "tokens": {}, "policies": {}}

            def get_scores(ids, guided=False, index=index, guide=guide, entry=entry, q=q):
                nonlocal count, tokens, shortened
                deadline()
                keys = [
                    ("guided" if guided and index.tree.children[i] else "static") + "|" + i
                    for i in ids
                ]
                missing = [
                    (i, key) for i, key in zip(ids, keys, strict=True) if key not in entry["scores"]
                ]
                if missing:
                    fitted = []
                    for i, key in missing:
                        cache_key = (q["id"], key)
                        if cache_key not in fitted_cache:
                            raw = guide.texts[i] if guided else index.texts[i]
                            fitted_cache[cache_key] = fit_decision_text(
                                q["question"], raw, [qt, tt], ct
                            )
                        fitted.append(fitted_cache[cache_key])
                    values, lengths = scorer.score(
                        [q["question"]] * len(missing), [t for t, _ in fitted]
                    )
                    if len(values) != len(missing) or not all(math.isfinite(s) for s in values):
                        raise ValueError("Invalid model scores")
                    entry["scores"].update(zip([key for _, key in missing], values, strict=True))
                    entry["tokens"].update(zip([key for _, key in missing], lengths, strict=True))
                    count += len(missing)
                    tokens += sum(lengths)
                    shortened += sum(m["excerpt_characters_removed"] > 0 for _, m in fitted)
                return {i: entry["scores"][key] for i, key in zip(ids, keys, strict=True)}

            for budget in config["budgets"]:
                for mode in config["modes"]:
                    if mode in {"flat", "tree_scored"}:
                        policy = traverse(
                            index, lexical[q["id"]], get_scores, mode=mode, budget=budget, width=4
                        )
                    else:
                        guided = mode == "tree_guided"
                        policy = budgeted_traverse(
                            index,
                            guide.priorities if guided else lexical[q["id"]],
                            lambda ids, guided=guided: get_scores(ids, guided),
                            budget=budget,
                            branch_batch=config["branch_batch"],
                            passage_batch=config["passage_batch"],
                        )
                    entry["policies"][f"{mode}-{budget}"] = policy
            cache[q["id"]] = entry
            with (args.output / f"{label}.partial.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"id": q["id"], **entry}) + "\n")
            if number % 8 == 0 or number == len(queries):
                log(variant=label, questions=number, total=len(queries), unique_pairs=count)
        save(args.output / "scores" / f"{label}.json", cache)
        (args.output / f"{label}.partial.jsonl").unlink()
        usage[label] = {
            "seconds": time.monotonic() - begin,
            "unique_pairs": count,
            "tokens": tokens,
            "shortened_pairs": shortened,
            "timing_scope": "All policies with shared score caches; not isolated per-policy latency",
        }
        save(args.output / "usage.json", usage)

    recipes = [("qwen-base", recipe)]
    if "tacit-4b" in models:
        recipes.append(
            (
                "tacit-4b",
                {
                    **recipe,
                    "base_model": config["tacit_model"],
                    "base_revision": config["tacit_revision"],
                },
            )
        )
    for label, current in recipes:
        log(stage="load_model", variant=label)
        model, tokenizer = load_model(current)
        infer(DecisionScorer(model, tokenizer, config["qwen_batch_size"]), label)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    model = AutoModelForSequenceClassification.from_pretrained(
        source["cross_encoder"], revision=source["cross_encoder_revision"]
    ).to("cuda")
    infer(CrossEncoderScorer(model, ct), "cross-encoder-base")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Full-leaf E5 is an external reference, not charged as an equal pair-budget policy.
    et = AutoTokenizer.from_pretrained(
        source["embedding_model"], revision=source["embedding_revision"]
    )
    encoder = (
        AutoModel.from_pretrained(source["embedding_model"], revision=source["embedding_revision"])
        .to("cuda")
        .eval()
    )

    def embed(texts):
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
                hidden = encoder(**inputs).last_hidden_state.masked_fill(
                    ~inputs["attention_mask"][..., None].bool(), 0
                )
                result.append(
                    torch.nn.functional.normalize(
                        hidden.sum(1) / inputs["attention_mask"].sum(1)[..., None], dim=1
                    )
                )
        return torch.cat(result)

    begin, vectors, e5 = time.monotonic(), {}, {}
    for q in queries:
        d, index = q["document"], indexes[q["document"]]
        if d not in vectors:
            vectors[d] = embed(
                [
                    "passage: "
                    + " / ".join(trees[d].nodes[i].path[1:])
                    + "\n"
                    + trees[d].nodes[i].content
                    for i in index.leaves
                ]
            )
        values = (vectors[d] @ embed(["query: " + q["question"]])[0]).tolist()
        e5[q["id"]] = {"scores": dict(zip(index.leaves, values, strict=True))}
    save(args.output / "scores/e5-text.json", e5)
    usage["e5-text"] = {
        "seconds": time.monotonic() - begin,
        "indexed_passages": sum(len(i.leaves) for i in indexes.values()),
    }
    save(args.output / "usage.json", usage)
    if config.get("answerability_history"):
        from arborvault.answerability import extract_probe_features

        del encoder
        gc.collect()
        torch.cuda.empty_cache()
        log(stage="probe_features")
        begin = time.monotonic()
        model = AutoModelForSequenceClassification.from_pretrained(
            source["cross_encoder"], revision=source["cross_encoder_revision"]
        ).to("cuda")
        usage["answerability_probe"] = {
            **extract_probe_features(config, args.output, trees, queries, model, ct, deadline),
            "seconds": time.monotonic() - begin,
            "extra_forwards_per_deployed_gate": 1,
        }
        save(args.output / "usage.json", usage)
    save(args.output / "runtime.json", {"complete": True, "seconds": time.monotonic() - started})
    log(stage="complete")
    save(
        args.output / "artifact-hashes.json",
        {
            p.relative_to(args.output).as_posix(): file_hash(p)
            for p in sorted(args.output.rglob("*"))
            if p.is_file()
        },
    )


if __name__ == "__main__":
    main()
