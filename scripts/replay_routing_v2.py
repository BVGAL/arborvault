"""Replay the three fixed greedy development policies on the diagnostic GPU."""

import argparse
import os
import time
from contextlib import nullcontext
from pathlib import Path

from diagnose_routing_v2 import save

from arborvault.qwen3 import file_hash, load_model, load_recipe, packages
from arborvault.routing_eval import evaluate
from arborvault.routing_v2 import QwenSelector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import set_seed

    recipe_path = Path("configs/qwen3-routing-v2.json")
    dataset = Path("datasets/qasper-routing-v2")
    recipe = load_recipe(recipe_path)
    set_seed(0)
    model, tokenizer = load_model(recipe)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = PeftModel.from_pretrained(
        model, args.adapters / "seed-42-step-88", adapter_name="step-88", is_trainable=False
    )
    model.load_adapter(
        args.adapters / "seed-42-step-175", adapter_name="step-175", is_trainable=False
    )
    started = time.monotonic()
    save(
        args.output / "environment.json",
        {"gpu": torch.cuda.get_device_name(), "packages": packages()},
    )
    save(
        args.output / "plan.json",
        {
            "reason": "A40 oracle-state initial decisions differ from archived RTX4090 decisions; replay all three fixed policies on the same GPU as diagnostics",
            "split": "dev",
            "training": False,
            "policy_changes": False,
            "recipe_sha256": file_hash(recipe_path),
            "dataset_sha256": file_hash(dataset / "manifest.json"),
            "base_revision": recipe["base_revision"],
            "code_revision": os.getenv("ARBORVAULT_CODE_REVISION", "unknown"),
            "adapter_hashes": {
                str(s): file_hash(args.adapters / f"seed-42-step-{s}" / "adapter_model.safetensors")
                for s in (88, 175)
            },
        },
    )
    for variant in ("base", "step-88", "step-175"):
        if variant != "base":
            model.set_adapter(variant)
        model.eval()

        def progress(event):
            if time.monotonic() - started > 3600:
                raise TimeoutError("Replay deadline exceeded")
            print(variant, event["query_id"], flush=True)

        with model.disable_adapter() if variant == "base" else nullcontext():
            report = evaluate(dataset, "dev", QwenSelector(model, tokenizer), progress=progress)
        if not report["complete"]:
            raise ValueError("Replay contained incomplete navigation")
        save(args.output / (variant + ".json"), report)
    save(
        args.output / "complete.json",
        {
            "complete": True,
            "elapsed_seconds": time.monotonic() - started,
            "shared_base_tensors": True,
        },
    )
    save(
        args.output / "manifest.json",
        {p.name: file_hash(p) for p in sorted(args.output.iterdir()) if p.is_file()},
    )


if __name__ == "__main__":
    main()
