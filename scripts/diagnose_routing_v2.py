"""Diagnose the existing seed-42 adapters on development, without training."""

import argparse
import hashlib
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

from arborvault.qasper import load_split
from arborvault.qwen3 import file_hash, load_model, load_recipe, packages
from arborvault.routing_diagnostics import diagnose_state, oracle_states, summarize_states
from arborvault.routing_v2 import QwenSelector


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/qasper-routing-v2"))
    parser.add_argument("--recipe", type=Path, default=Path("configs/qwen3-routing-v2.json"))
    parser.add_argument("--adapters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=5400)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory; preserve prior runs")
    args.output.mkdir(parents=True)
    recipe = load_recipe(args.recipe)
    _, trees, queries = load_split(args.dataset, "dev")
    states = [s for q in queries for s in oracle_states(trees[q["document"]], q)]
    by_query = {q["id"]: q for q in queries}
    started = time.monotonic()
    plan = {
        "scope": "Post-hoc development diagnostics; no training, selection or new test claims",
        "seed": 42,
        "variants": ["base", "step-88", "step-175"],
        "base_revision": recipe["base_revision"],
        "recipe_sha256": file_hash(args.recipe),
        "dataset_sha256": file_hash(args.dataset / "manifest.json"),
        "oracle_states_sha256": hashlib.sha256(
            json.dumps(states, sort_keys=True).encode()
        ).hexdigest(),
        "states": len(states),
        "queries": len(queries),
        "code_revision": os.getenv("ARBORVAULT_CODE_REVISION", "unknown"),
        "adapter_hashes": {
            str(step): file_hash(
                args.adapters / f"seed-42-step-{step}" / "adapter_model.safetensors"
            )
            for step in (88, 175)
        },
    }
    save(args.output / "plan.json", plan)
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import set_seed

    set_seed(0)
    model, tokenizer = load_model(recipe)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = PeftModel.from_pretrained(
        model, args.adapters / "seed-42-step-88", adapter_name="step-88", is_trainable=False
    )
    model.load_adapter(
        args.adapters / "seed-42-step-175", adapter_name="step-175", is_trainable=False
    )
    model.eval()
    save(
        args.output / "environment.json",
        {"packages": packages(), "gpu": torch.cuda.get_device_name()},
    )
    for variant in plan["variants"]:
        if variant != "base":
            model.set_adapter(variant)
        model.eval()
        rows = []
        selector = QwenSelector(model, tokenizer)
        with model.disable_adapter() if variant == "base" else nullcontext():
            with (args.output / (variant + "-states.jsonl")).open("w", encoding="utf-8") as stream:
                for state in states:
                    if time.monotonic() - started > args.max_seconds:
                        raise TimeoutError("Diagnostic deadline exceeded")
                    row = diagnose_state(
                        model,
                        tokenizer,
                        selector,
                        trees[state["document"]],
                        by_query[state["query_id"]]["question"],
                        state,
                    )
                    rows.append(row)
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    print(
                        json.dumps({"variant": variant, "states": len(rows), "total": len(states)}),
                        flush=True,
                    )
        save(args.output / (variant + "-summary.json"), summarize_states(rows))
    save(
        args.output / "complete.json",
        {
            "complete": True,
            "elapsed_seconds": time.monotonic() - started,
            "shared_base_tensors": True,
            "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        },
    )
    save(
        args.output / "manifest.json",
        {p.name: file_hash(p) for p in sorted(args.output.iterdir()) if p.is_file()},
    )


if __name__ == "__main__":
    main()
