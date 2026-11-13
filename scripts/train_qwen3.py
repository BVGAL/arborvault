"""Run the fixed Qwen3 routing QLoRA experiment. Saves locally; never uploads to a Hub."""

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

from arborvault.qwen3 import (
    completion_loss,
    encode_completion,
    file_hash,
    load_model,
    load_recipe,
    packages,
)
from arborvault.training_data import load_bundle


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=Path("configs/qwen3-routing-v1.json"))
    parser.add_argument("--dataset", type=Path, default=Path("datasets/routing-bootstrap-v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--time-limit-seconds", type=int, default=7200)
    args = parser.parse_args()
    recipe = load_recipe(args.recipe)
    manifest, rows = load_bundle(args.dataset)
    if args.validate_only:
        print(json.dumps({"validated": True, "counts": manifest["counts"], "recipe": recipe}))
        return
    if args.time_limit_seconds < 60:
        parser.error("Time limit must be at least 60 seconds")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be empty; preserve previous experiments")
    args.output.mkdir(parents=True, exist_ok=True)

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import set_seed

    set_seed(recipe["seed"])
    # No public test questions/qrels are read by this program.
    write_json(args.output / "experiment-plan.json", recipe)
    start = time.monotonic()

    def deadline():
        if time.monotonic() - start > args.time_limit_seconds:
            raise TimeoutError("Training time limit reached; experiment is incomplete")

    model, tokenizer = load_model(recipe)
    encoded = {
        split: [
            encode_completion(tokenizer, row["messages"], recipe["max_length"])
            for row in rows[split]
        ]
        for split in ("train", "validation")
    }
    model = prepare_model_for_kbit_training(
        model, gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=recipe["lora_r"],
            lora_alpha=recipe["lora_alpha"],
            lora_dropout=recipe["lora_dropout"],
            target_modules=recipe["target_modules"],
        ),
    )
    model.config.use_cache = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"]
    )
    accumulation = recipe["gradient_accumulation_steps"]
    best_loss, best_epoch, step, history = math.inf, None, 0, []
    stream = (args.output / "progress.jsonl").open("w", encoding="utf-8")

    def log(value):
        stream.write(json.dumps(value) + "\n")
        stream.flush()
        print(json.dumps(value), flush=True)

    try:
        log({"stage": "loaded", "trainable_parameters": sum(p.numel() for p in trainable)})
        for epoch in range(1, recipe["epochs"] + 1):
            order = list(range(len(encoded["train"])))
            random.Random(recipe["seed"] + epoch).shuffle(order)
            model.train()
            total_loss, total_tokens = 0.0, 0
            for offset in range(0, len(order), accumulation):
                batch = order[offset : offset + accumulation]
                optimizer.zero_grad(set_to_none=True)
                for index in batch:
                    deadline()
                    loss, count = completion_loss(model, encoded["train"][index])
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite training loss")
                    total_loss += loss.detach().item() * count
                    total_tokens += count
                    (loss / len(batch)).backward()
                torch.nn.utils.clip_grad_norm_(
                    trainable, recipe["max_grad_norm"], error_if_nonfinite=True
                )
                optimizer.step()
                step += 1
                if step % 5 == 0:
                    log(
                        {
                            "stage": "training",
                            "epoch": epoch,
                            "step": step,
                            "nll": total_loss / total_tokens,
                        }
                    )
            model.eval()
            val_loss, val_tokens = 0.0, 0
            with torch.inference_mode():
                for example in encoded["validation"]:
                    deadline()
                    loss, count = completion_loss(model, example)
                    val_loss += loss.item() * count
                    val_tokens += count
            validation = val_loss / val_tokens
            if not math.isfinite(validation):
                raise ValueError("Non-finite validation loss")
            record = {
                "epoch": epoch,
                "step": step,
                "training_completion_token_nll": total_loss / total_tokens,
                "validation_completion_token_nll": validation,
            }
            history.append(record)
            if validation < best_loss:
                best_loss, best_epoch = validation, epoch
                model.save_pretrained(args.output / "adapter", safe_serialization=True)
                tokenizer.save_pretrained(args.output / "adapter")
            write_json(args.output / "epoch-metrics.json", history)
            log({"stage": "validated", **record, "selected_epoch": best_epoch})

        metadata = {
            "complete": True,
            "recipe": recipe,
            "recipe_sha256": file_hash(args.recipe),
            "base_model": recipe["base_model"],
            "base_revision": recipe["base_revision"],
            "prompt_sha256": manifest["prompt_sha256"],
            "dataset_sha256": manifest["sha256"],
            "public_dataset_sha256": manifest["public_dataset_sha256"],
            "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
            "selected_epoch": best_epoch,
            "best_validation_completion_token_nll": best_loss,
            "epochs": history,
            "optimizer_steps": step,
            "trainable_parameters": sum(p.numel() for p in trainable),
            "maximum_training_tokens": max(len(row["input_ids"]) for row in encoded["train"]),
            "maximum_validation_tokens": max(
                len(row["input_ids"]) for row in encoded["validation"]
            ),
            "elapsed_seconds": time.monotonic() - start,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "gpu": torch.cuda.get_device_name(),
            "packages": packages(),
            "adapter_sha256": {
                path.name: file_hash(path)
                for path in sorted((args.output / "adapter").iterdir())
                if path.is_file()
            },
            "selection_scope": "Only training/validation examples; public test excluded",
            "retrieval_quality": "Not established by training loss; requires held-out retrieval evaluation",
        }
        write_json(args.output / "training-provenance.json", metadata)
        log({"stage": "complete", "selected_epoch": best_epoch, "optimizer_steps": step})
    finally:
        stream.close()


if __name__ == "__main__":
    main()
