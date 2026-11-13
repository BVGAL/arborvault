"""Optional attention-only LoRA training. No training, downloads or Hub writes in validate-only mode."""

import argparse
import hashlib
import json
import re
from importlib.metadata import version
from pathlib import Path

from arborvault.training_data import load_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--revision", help="Immutable 40-character model commit SHA")
    parser.add_argument("--output", type=Path, default=Path("models/routing-adapter"))
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    manifest, splits = load_bundle(args.dataset)
    if args.validate_only:
        print(
            json.dumps(
                {"validation": "passed", "counts": manifest["counts"], "training_executed": False}
            )
        )
        return
    if not args.revision or not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("Pin --revision to the exact base checkpoint before training")
    if args.max_length < 1 or args.epochs <= 0 or args.learning_rate <= 0:
        parser.error("Training settings must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty; keep experiments separate")
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
    from trl import SFTConfig, SFTTrainer

    from datasets import Dataset

    if not torch.cuda.is_available():
        parser.error("This training recipe requires a CUDA GPU; validate-only works without one")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=False
    )
    if not tokenizer.chat_template:
        parser.error("The base tokenizer must supply its chat template")
    for split in ("train", "validation"):
        for row in splits[split]:
            if len(tokenizer.apply_chat_template(row["messages"])) > args.max_length:
                parser.error(
                    "An example exceeds max-length; increase it instead of truncating labels"
                )
    options = (
        {"quantization_config": Mxfp4Config(dequantize=True)}
        if args.model == "openai/gpt-oss-20b"
        else {}
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        **options,
    )
    model.config.use_cache = False
    datasets = {
        split: Dataset.from_list(
            [
                {"prompt": r["messages"][:-1], "completion": r["messages"][-1:]}
                for r in splits[split]
            ]
        )
        for split in ("train", "validation")
    }
    config = SFTConfig(
        output_dir=str(args.output),
        seed=42,
        data_seed=42,
        max_length=args.max_length,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        bf16=True,
        completion_only_loss=True,
        packing=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        push_to_hub=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        peft_config=LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
        ),
    )
    training = trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    metadata = {
        "base_model": args.model,
        "base_revision": args.revision,
        "prompt_sha256": manifest["prompt_sha256"],
        "dataset_sha256": manifest["sha256"],
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "recipe": "attention-qv-lora-v1",
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "seed": 42,
        "training_metrics": training.metrics,
        "packages": {
            name: version(name) for name in ("torch", "transformers", "trl", "peft", "datasets")
        },
        "retrieval_quality": "not evaluated by the training loss",
    }
    (args.output / "training-provenance.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print("Adapter saved locally. Evaluate it before publishing; no Hub upload was performed.")


if __name__ == "__main__":
    main()
