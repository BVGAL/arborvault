"""Frozen beam ablation; select on development before reading any test score."""

import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

from run_routing_v2 import qualifies, quality, save

from arborvault.qwen3 import file_hash, load_model, load_recipe, packages
from arborvault.routing_beams import POLICY, POLICY_SHA256, BeamQwenSelector
from arborvault.routing_eval import evaluate
from arborvault.routing_v2 import PROMPT_SHA256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/qasper-routing-v2"))
    parser.add_argument("--recipe", type=Path, default=Path("configs/qwen3-routing-v2.json"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be empty; preserve earlier experiments")
    recipe = load_recipe(args.recipe)
    # Training progress contains no test metrics; deliberately do not read final provenance.
    training = json.loads((args.training / "training-progress.json").read_text())
    if training["recipe_sha256"] != file_hash(args.recipe):
        raise ValueError("Training recipe mismatch")
    if training["dataset_manifest_sha256"] != file_hash(args.dataset / "manifest.json"):
        raise ValueError("Training dataset mismatch")
    candidates = training["checkpoints"]
    if len(candidates) != 4:
        raise ValueError("Four fixed checkpoints required")
    for c in candidates:
        if (
            file_hash(args.training / c["adapter"] / "adapter_model.safetensors")
            != c["adapter_sha256"]
        ):
            raise ValueError("Adapter checksum mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    stream = (args.output / "progress.jsonl").open("w", encoding="utf-8")

    def log(event):
        if time.monotonic() - started > 3600:
            raise TimeoutError("Beam ablation deadline exceeded")
        stream.write(json.dumps(event) + "\n")
        stream.flush()
        print(json.dumps(event), flush=True)

    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import set_seed

    model, tokenizer = load_model(recipe)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    for index, c in enumerate(candidates):
        name = f"seed-{c['seed']}-step-{c['step']}"
        if index == 0:
            model = PeftModel.from_pretrained(
                model, args.training / c["adapter"], adapter_name=name, is_trainable=False
            )
        else:
            model.load_adapter(args.training / c["adapter"], adapter_name=name, is_trainable=False)
    model.eval()
    metadata = {
        "recipe_sha256": file_hash(args.recipe),
        "dataset_manifest_sha256": file_hash(args.dataset / "manifest.json"),
        "code_revision": os.getenv("ARBORVAULT_CODE_REVISION", "unknown"),
        "training_code_revision": training["code_revision"],
        "policy": POLICY,
        "policy_sha256": POLICY_SHA256,
        "packages": packages(),
        "gpu": torch.cuda.get_device_name(),
        "all_evaluations_shared_base_tensors": True,
    }
    save(args.output / "preregistered-plan.json", metadata)

    def assess(split, tag, candidate=None):
        set_seed(0)
        if candidate:
            model.set_adapter(f"seed-{candidate['seed']}-step-{candidate['step']}")
        with nullcontext() if candidate else model.disable_adapter():
            selector = BeamQwenSelector(
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
            code_revision=metadata["code_revision"],
            decoding_policy_sha256=POLICY_SHA256,
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

    base = assess("dev", "base-dev")
    primary = []
    for c in candidates:
        if c["seed"] != recipe["seed"]:
            continue
        report = assess("dev", f"seed-{c['seed']}-step-{c['step']}-dev", c)
        primary.append({**c, "quality": quality(report), "qualifies": qualifies(report, base)})
    eligible = [c for c in primary if c["qualifies"]]
    chosen = max(eligible or primary, key=lambda c: (c["quality"], -c["step"]))
    confirmation = next(
        c
        for c in candidates
        if c["seed"] == recipe["confirmation_seed"] and c["fraction"] == chosen["fraction"]
    )
    report = assess(
        "dev", f"seed-{confirmation['seed']}-step-{confirmation['step']}-dev", confirmation
    )
    confirmation = {
        **confirmation,
        "quality": quality(report),
        "qualifies": qualifies(report, base),
    }
    promotion = (
        chosen["qualifies"]
        and tuple(chosen["quality"]) > quality(base)
        and confirmation["qualifies"]
        and tuple(confirmation["quality"]) > quality(base)
    )
    # Policy preference is also development-only; guards apply against the greedy base.
    greedy_base = json.loads((args.training / "evaluation/base-dev.json").read_text())
    prefer_beams = qualifies(base, greedy_base) and quality(base) > quality(greedy_base)
    selection = {
        "chosen_primary_adapter": chosen,
        "confirmation_same_fraction": confirmation,
        "promote_over_base": promotion,
        "deployment_recommendation": "candidate_pending_test" if promotion else "retain_base",
        "prefer_beam_base_over_greedy_base": prefer_beams,
        "test_not_yet_evaluated": True,
        "test_sealing": "Greedy test may exist on disk; this runner has not read it",
    }
    save(args.output / "selection-before-test.json", selection)
    log({"stage": "selection_locked", **selection})
    base_test = assess("test", "base-test")
    tuned_test = assess("test", "tuned-test", chosen)
    metadata.update(
        complete=True,
        selection=selection,
        elapsed_seconds=time.monotonic() - started,
        test_complete=base_test["complete"] and tuned_test["complete"],
    )
    save(args.output / "beam-provenance.json", metadata)
    stream.close()
    save(
        args.output / "artifact-manifest.json",
        {
            p.relative_to(args.output).as_posix(): file_hash(p)
            for p in sorted(args.output.rglob("*"))
            if p.is_file() and p.name != "artifact-manifest.json"
        },
    )
    print(
        json.dumps({"stage": "complete", "elapsed_seconds": metadata["elapsed_seconds"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
