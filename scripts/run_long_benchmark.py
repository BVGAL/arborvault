"""Frozen inference-only long-manual benchmark, with document-separated calibration."""

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path

from arborvault.candidate_models import CrossEncoderScorer, QwenScorer, fit_text
from arborvault.long_benchmark import DocumentIndex, traverse
from arborvault.long_documents import load_corpus
from arborvault.qwen3 import file_hash, load_model, load_recipe, packages


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/long-manuals-v1.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["budgets"] != [64, 128] or config["width"] != 4:
        raise ValueError("Unsupported deviation from frozen policy")
    manifest, trees, queries = load_corpus(Path(config["dataset"]))
    source = json.loads(Path(config["scorer_config"]).read_text())
    for name, expected in config["checkpoint_files"].items():
        if file_hash(args.checkpoint / name) != expected:
            raise ValueError("Unexpected cross-encoder checkpoint")
    args.output.mkdir(parents=True, exist_ok=False)
    save(
        args.output / "plan.json",
        {
            "source_revision": os.getenv("ARBORVAULT_CODE_REVISION", "unknown"),
            "config": config,
            "scorer_config": source,
            "dataset_manifest_sha256": file_hash(Path(config["dataset"]) / "manifest.json"),
            "dataset_counts": manifest["counts"],
            "config_sha256": file_hash(args.config),
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
    save(
        args.output / "environment.json",
        {"packages": packages(), "gpu": torch.cuda.get_device_name()},
    )

    def deadline():
        if time.monotonic() - started > config["max_seconds"]:
            raise TimeoutError("Inference budget deadline reached")

    def log(**event):
        event["elapsed_seconds"] = round(time.monotonic() - started, 3)
        with (args.output / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)

    recipe = load_recipe(Path(source["qwen_recipe"]))
    qt = AutoTokenizer.from_pretrained(recipe["base_model"], revision=recipe["base_revision"])
    ct = AutoTokenizer.from_pretrained(
        source["cross_encoder"], revision=source["cross_encoder_revision"]
    )
    usage = {}

    def infer(scorer, label):
        begin, count, tokens, shortened = time.monotonic(), 0, 0, 0
        cache = {}
        for number, q in enumerate(queries, 1):
            index = indexes[q["document"]]
            entry = {"scores": {}, "tokens": {}, "policies": {}}

            def get_scores(ids, entry=entry, q=q, index=index):
                nonlocal count, tokens, shortened
                deadline()
                missing = [i for i in ids if i not in entry["scores"]]
                if missing:
                    fitted = [fit_text(q["question"], index.texts[i], qt, ct) for i in missing]
                    scores, lengths = scorer.score(
                        [q["question"]] * len(missing), [t for t, _ in fitted]
                    )
                    if not all(math.isfinite(s) for s in scores):
                        raise ValueError("Non-finite score")
                    entry["scores"].update(zip(missing, scores, strict=True))
                    entry["tokens"].update(zip(missing, lengths, strict=True))
                    count += len(missing)
                    tokens += sum(lengths)
                    shortened += sum(m["excerpt_characters_removed"] > 0 for _, m in fitted)
                return {i: entry["scores"][i] for i in ids}

            for budget in config["budgets"]:
                for mode in ("flat", "tree_scored"):
                    entry["policies"][f"{mode}-{budget}"] = traverse(
                        index,
                        lexical[q["id"]],
                        get_scores,
                        mode=mode,
                        budget=budget,
                        width=config["width"],
                    )
            cache[q["id"]] = entry
            if number % 16 == 0 or number == len(queries):
                log(variant=label, questions=number, total=len(queries), unique_pairs=count)
        save(args.output / "scores" / f"{label}.json", cache)
        usage[label] = {
            "seconds": time.monotonic() - begin,
            "unique_pairs": count,
            "tokens": tokens,
            "shortened_pairs": shortened,
        }
        save(args.output / "usage.json", usage)

    model, qt = load_model(recipe)
    infer(QwenScorer(model, qt, 8), "qwen-base")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    for label, model_path, revision in [
        ("cross-encoder-base", source["cross_encoder"], source["cross_encoder_revision"]),
        ("cross-encoder-tuned", str(args.checkpoint), None),
    ]:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, revision=revision
        ).to("cuda")
        infer(CrossEncoderScorer(model, ct), label)
        del model
        gc.collect()
        torch.cuda.empty_cache()

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

    begin, e5, vectors = time.monotonic(), {}, {}
    for number, q in enumerate(queries, 1):
        d = q["document"]
        index = indexes[d]
        if d not in vectors:
            texts = [
                "passage: "
                + " / ".join(trees[d].nodes[i].path[1:])
                + "\n"
                + trees[d].nodes[i].content
                for i in index.leaves
            ]
            vectors[d] = embed(texts)
        scores = (vectors[d] @ embed(["query: " + q["question"]])[0]).tolist()
        e5[q["id"]] = {"scores": dict(zip(index.leaves, scores, strict=True))}
        if number % 32 == 0:
            log(variant="e5-text", questions=number, total=len(queries))
    save(args.output / "scores/e5-text.json", e5)
    usage["e5-text"] = {
        "seconds": time.monotonic() - begin,
        "indexed_passages": sum(len(i.leaves) for i in indexes.values()),
    }
    save(args.output / "usage.json", usage)
    save(args.output / "runtime.json", {"complete": True, "seconds": time.monotonic() - started})
    log(stage="complete")
    files = {
        p.relative_to(args.output).as_posix(): file_hash(p)
        for p in sorted(args.output.rglob("*"))
        if p.is_file()
    }
    save(args.output / "artifact-hashes.json", files)


if __name__ == "__main__":
    main()
