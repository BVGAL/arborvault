"""Fixed Qwen3 routing v2 experiment: two seeds, dev selection, then one frozen test pass."""

import argparse
import gc
import hashlib
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

from arborvault.qasper import load_split, training_routes
from arborvault.qwen3 import (
    completion_loss,
    encode_completion,
    file_hash,
    load_model,
    load_recipe,
    packages,
)
from arborvault.routing_eval import baseline, evaluate
from arborvault.routing_v2 import PROMPT_SHA256, QwenSelector, fit_messages


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def quality(report):
    m = report["methods"]["navigation"]
    return m["recall"], m["evidence_coverage"], m["mrr"]


def qualifies(report, base):
    m, b = (r["methods"]["navigation"] for r in (report, base))
    return (
        report["complete"]
        and m["selector_errors"] == 0
        and m["answerable_empty"] <= b["answerable_empty"]
        and m["unknown_abstentions"] >= b["unknown_abstentions"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/qasper-routing-v2"))
    parser.add_argument("--recipe", type=Path, default=Path("configs/qwen3-routing-v2.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/routing-v2"))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    recipe = load_recipe(args.recipe)
    manifest, trees, _ = load_split(args.dataset, "train")
    routes = training_routes(args.dataset)
    if args.validate_only:
        print(json.dumps({"validated": True, "routes": len(routes), "splits": manifest["splits"]}))
        return
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be empty; preserve earlier experiments")
    args.output.mkdir(parents=True, exist_ok=True)

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import set_seed

    started = time.monotonic()
    stream = (args.output / "progress.jsonl").open("w", encoding="utf-8")

    def deadline():
        if time.monotonic() - started > recipe["max_total_seconds"]:
            raise TimeoutError("Experiment deadline reached; cloud watchdog is independent")

    def log(event):
        deadline()
        stream.write(json.dumps(event) + "\n")
        stream.flush()
        print(json.dumps(event), flush=True)

    def assess(model, tokenizer, split, tag, disabled=False):
        model.eval()
        set_seed(0)
        with model.disable_adapter() if disabled else nullcontext():
            selector = QwenSelector(
                model,
                tokenizer,
                num_ctx=recipe["evaluation_num_ctx"],
                max_output_tokens=recipe["evaluation_max_output_tokens"],
            )
            report = evaluate(
                args.dataset,
                split,
                selector,
                progress=lambda event: log({"stage": "evaluation", "tag": tag, **event}),
            )
        report.update(
            variant=tag,
            base_revision=recipe["base_revision"],
            prompt_sha256=PROMPT_SHA256,
            recipe_sha256=file_hash(args.recipe),
            code_revision=os.getenv("ARBORVAULT_CODE_REVISION", "unknown"),
        )
        save(args.output / "evaluation" / (tag + ".json"), report)
        log(
            {
                "stage": "evaluation_complete",
                "tag": tag,
                "quality": quality(report),
                "empty": report["methods"]["navigation"]["answerable_empty"],
                "unknown_abstentions": report["methods"]["navigation"]["unknown_abstentions"],
            }
        )
        return report

    metadata = {
        "recipe": recipe,
        "recipe_sha256": file_hash(args.recipe),
        "dataset_manifest_sha256": file_hash(args.dataset / "manifest.json"),
        "routing_prompt_sha256": PROMPT_SHA256,
        "packages": packages(),
        "code_revision": os.getenv("ARBORVAULT_CODE_REVISION", "unknown"),
        "selection_uses": "development only; test not loaded until adapter choice is saved",
        "checkpoints": [],
        "training_routes_sha256": hashlib.sha256(
            json.dumps(routes, sort_keys=True).encode()
        ).hexdigest(),
    }
    save(args.output / "preregistered-plan.json", metadata)
    baseline_dev, primary_candidates, confirmation_candidates = None, [], []
    encoded = None
    for seed in (recipe["seed"], recipe["confirmation_seed"]):
        deadline()
        set_seed(seed)
        model, tokenizer = load_model(recipe)
        model = prepare_model_for_kbit_training(
            model, gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                revision=recipe["base_revision"],
                r=recipe["lora_r"],
                lora_alpha=recipe["lora_alpha"],
                lora_dropout=recipe["lora_dropout"],
                target_modules=recipe["target_modules"],
            ),
        )
        model.config.use_cache = False
        if encoded is None:
            encoded, budgets = [], {}
            for route in routes:
                tree = trees[route["document"]]
                candidates = tree.views([tree.nodes[i] for i in route["candidate_ids"]])
                conversation, _, _, budget = fit_messages(
                    tokenizer,
                    route["query"],
                    candidates,
                    4,
                    recipe["evaluation_num_ctx"] - recipe["evaluation_max_output_tokens"],
                )
                target = route["messages"][-1]
                encoded.append(
                    encode_completion(tokenizer, conversation + [target], recipe["max_length"])
                )
                budgets[str(budget)] = budgets.get(str(budget), 0) + 1
            metadata.update(
                training_examples=len(encoded),
                maximum_training_tokens=max(len(r["input_ids"]) for r in encoded),
                excerpt_budget_histogram=budgets,
                chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                gpu=torch.cuda.get_device_name(),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                encoded_training_sha256=hashlib.sha256(
                    json.dumps(encoded, sort_keys=True).encode()
                ).hexdigest(),
            )
            save(args.output / "tokenization-preflight.json", metadata)
            log(
                {
                    "stage": "preflight",
                    "routes": len(encoded),
                    "max_tokens": metadata["maximum_training_tokens"],
                    "budgets": budgets,
                }
            )
            baseline_dev = assess(model, tokenizer, "dev", "base-dev", disabled=True)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable, lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"]
        )
        order = list(range(len(encoded)))
        random.Random(seed).shuffle(order)
        accumulation = recipe["gradient_accumulation_steps"]
        steps = math.ceil(len(order) / accumulation)
        checkpoint_steps = {
            math.ceil(steps * fraction): fraction for fraction in recipe["checkpoints"]
        }
        total_loss, total_tokens = 0.0, 0
        model.train()
        set_seed(seed)
        for step, offset in enumerate(range(0, len(order), accumulation), 1):
            deadline()
            batch = order[offset : offset + accumulation]
            optimizer.zero_grad(set_to_none=True)
            for index in batch:
                loss, count = completion_loss(model, encoded[index])
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                total_loss += loss.detach().item() * count
                total_tokens += count
                (loss / len(batch)).backward()
            torch.nn.utils.clip_grad_norm_(
                trainable, recipe["max_grad_norm"], error_if_nonfinite=True
            )
            optimizer.step()
            if step % 5 == 0:
                log(
                    {
                        "stage": "training",
                        "seed": seed,
                        "step": step,
                        "steps": steps,
                        "completion_token_nll": total_loss / total_tokens,
                    }
                )
            if step in checkpoint_steps:
                name = f"seed-{seed}-step-{step}"
                adapter = args.output / "adapters" / name
                model.save_pretrained(adapter, safe_serialization=True)
                tokenizer.save_pretrained(adapter)
                cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
                report = assess(model, tokenizer, "dev", name + "-dev")
                candidate = {
                    "seed": seed,
                    "step": step,
                    "fraction": checkpoint_steps[step],
                    "adapter": "adapters/" + name,
                    "quality": quality(report),
                    "qualifies": qualifies(report, baseline_dev),
                    "completion_token_nll": total_loss / total_tokens,
                    "adapter_sha256": file_hash(adapter / "adapter_model.safetensors"),
                }
                (primary_candidates if seed == recipe["seed"] else confirmation_candidates).append(
                    candidate
                )
                metadata["checkpoints"].append(candidate)
                save(args.output / "training-progress.json", metadata)
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state_all(cuda_rng)
                model.train()
        del loss, optimizer, trainable, model
        gc.collect()
        torch.cuda.empty_cache()
    eligible = [c for c in primary_candidates if c["qualifies"]]
    chosen = max(eligible or primary_candidates, key=lambda c: (c["quality"], -c["step"]))
    confirmation = next(c for c in confirmation_candidates if c["fraction"] == chosen["fraction"])
    promotion = (
        chosen["qualifies"]
        and tuple(chosen["quality"]) > quality(baseline_dev)
        and confirmation["qualifies"]
        and tuple(confirmation["quality"]) > quality(baseline_dev)
    )
    selection = {
        "chosen_primary_adapter": chosen,
        "confirmation_same_fraction": confirmation,
        "promote_over_base": promotion,
        "deployment_recommendation": "candidate_pending_test" if promotion else "retain_base",
        "test_not_yet_evaluated": True,
    }
    save(args.output / "selection-before-test.json", selection)
    log({"stage": "selection_locked", **selection})
    model, tokenizer = load_model(recipe)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = PeftModel.from_pretrained(model, args.output / chosen["adapter"], is_trainable=False)
    # Final base/adapter comparison shares the exact same NF4 tensors in this process.
    base_test = assess(model, tokenizer, "test", "base-test", disabled=True)
    tuned_test = assess(model, tokenizer, "test", "tuned-test")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    for split in ("dev", "test"):
        log({"stage": "dense_baseline", "split": split})
        save(
            args.output / "evaluation" / ("baselines-" + split + ".json"),
            baseline(args.dataset, split, args.cache_dir),
        )
    metadata.update(
        complete=True,
        elapsed_seconds=time.monotonic() - started,
        selection=selection,
        base_test_quality=quality(base_test),
        tuned_test_quality=quality(tuned_test),
        test_shared_base_tensors=True,
        test_complete=base_test["complete"] and tuned_test["complete"],
    )
    save(args.output / "training-provenance.json", metadata)
    hashes = {
        p.relative_to(args.output).as_posix(): file_hash(p)
        for p in sorted(args.output.rglob("*"))
        if p.is_file() and p.name != "artifact-manifest.json"
    }
    stream.close()
    # progress.jsonl is closed before its final checksum is recorded.
    hashes["progress.jsonl"] = file_hash(args.output / "progress.jsonl")
    save(args.output / "artifact-manifest.json", hashes)
    print(
        json.dumps({"stage": "complete", "elapsed_seconds": metadata["elapsed_seconds"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
