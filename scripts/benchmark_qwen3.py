"""Compare base and trained Qwen3 in one NF4 runtime against the unchanged retrieval baselines."""

import argparse
import hashlib
import json
import os
import secrets
from contextlib import nullcontext
from pathlib import Path
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from arborvault.benchmark import run
from arborvault.llm_eval import compare
from arborvault.migrate import migrate, provision_runtime_role
from arborvault.ollama import PROMPT_SHA256, PROMPT_VERSION
from arborvault.qwen3 import TransformersSelector, file_hash, load_model, load_recipe, packages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=Path("configs/qwen3-routing-v1.json"))
    parser.add_argument("--dataset", type=Path, default=Path("datasets/public-docs-v1"))
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", default=".cache/models")
    args = parser.parse_args()
    recipe = load_recipe(args.recipe)
    provenance = json.loads((args.training_output / "training-provenance.json").read_text())
    if not provenance["complete"] or provenance["recipe_sha256"] != file_hash(args.recipe):
        parser.error("Training must be complete and match the fixed recipe")
    if provenance["prompt_sha256"] != PROMPT_SHA256:
        parser.error("Training used a different routing prompt")
    adapter = args.training_output / "adapter"
    for name, expected in provenance["adapter_sha256"].items():
        if Path(name).name != name or file_hash(adapter / name) != expected:
            parser.error("Adapter artifact hash mismatch")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty")
    if "TEST_DATABASE_URL" not in os.environ:
        parser.error(
            "Set TEST_DATABASE_URL to an isolated PostgreSQL server with create permission"
        )
    args.output.mkdir(parents=True, exist_ok=True)

    import torch
    from peft import PeftModel
    from transformers import set_seed

    set_seed(0)
    model, tokenizer = load_model(recipe)
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    template_hash = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
    if template_hash != provenance["chat_template_sha256"]:
        raise ValueError("Chat template differs from training")
    # Both variants use the SAME quantized base tensors. No merge or re-quantization.
    environment = packages()
    runtime_id = "transformers-" + environment["transformers"] + "/torch-" + environment["torch"]
    admin = os.environ["TEST_DATABASE_URL"]
    database, role = "av_qwen_" + uuid4().hex[:16], "av_role_" + uuid4().hex[:16]
    options = conninfo_to_dict(admin)
    reports = {}
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    try:
        options["dbname"] = database
        temporary_admin = make_conninfo(**options)
        migrate(temporary_admin)
        password = secrets.token_urlsafe(32)
        provision_runtime_role(temporary_admin, password, role)
        options.update(user=role, password=password)
        for variant in ("base", "fine-tuned"):
            context = model.disable_adapter() if variant == "base" else nullcontext()
            with context:
                # Same synthetic warm-up, excluded from both usage counters and measured queries.
                warmup = TransformersSelector(model, tokenizer)
                warmup.select("Synthetic installation check with no available evidence", [], 4)
                selector = TransformersSelector(
                    model,
                    tokenizer,
                    num_ctx=recipe["evaluation_num_ctx"],
                    max_output_tokens=recipe["evaluation_max_output_tokens"],
                )
                with (args.output / (variant + ".progress.jsonl")).open("w") as stream:

                    def progress(method, identity, row, variant=variant):
                        event = {
                            "variant": variant,
                            "method": method,
                            "query_id": identity,
                            "provider_usage": row["provider_usage"],
                            "fallback_reason": row["fallback_reason"],
                        }
                        stream.write(json.dumps(event) + "\n")
                        stream.flush()
                        print(json.dumps(event), flush=True)

                    report = run(
                        args.dataset,
                        make_conninfo(**options),
                        split=args.split,
                        dense=True,
                        cache_dir=args.cache_dir,
                        selector=selector,
                        progress=progress,
                    )
            if report["dataset_sha256"] != provenance["public_dataset_sha256"]:
                raise ValueError("Public evaluation corpus changed since training")
            report["experiment"] = {
                "variant": variant,
                "base_model": recipe["base_model"],
                "base_revision": recipe["base_revision"],
                "runtime_id": runtime_id,
                "provider": "transformers-local",
                "quantization": recipe["quantization"],
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": PROMPT_SHA256,
                "temperature": 0,
                "seed": 0,
                "think": False,
                "num_ctx": recipe["evaluation_num_ctx"],
                "max_output_tokens": recipe["evaluation_max_output_tokens"],
                "max_calls": 1056,
            }
            report["served_artifact"] = {
                "template_sha256": template_hash,
                "parameters_sha256": hashlib.sha256(
                    json.dumps(recipe, sort_keys=True).encode()
                ).hexdigest(),
                "adapter_enabled": variant == "fine-tuned",
                "adapter_sha256": provenance["adapter_sha256"]["adapter_model.safetensors"],
                "shared_base_tensors": True,
            }
            report["training_provenance_sha256"] = file_hash(
                args.training_output / "training-provenance.json"
            )
            report["runtime_packages"] = environment
            report["gpu"] = torch.cuda.get_device_name()
            report["code_revision"] = os.getenv("ARBORVAULT_CODE_REVISION", "unknown")
            report["provider_usage"] = selector.usage()
            report["selector_failure_events"] = [e for e in selector.events if not e["ok"]]
            report["generation_eos_token_ids"] = model.generation_config.eos_token_id
            report["provider_max_prompt_tokens"] = max(
                (e["prompt_tokens"] for e in selector.events), default=0
            )
            report["complete"] = selector.usage()["failures"] == 0 and not any(
                row["fallback_reason"] in {"selector_error", "invalid_selection"}
                for method in ("llm_navigation", "llm_auto")
                for row in report["methods"][method]["queries"]
            )
            report["timing_protocol"] = (
                "One sequential pass, base before adapter; one excluded synthetic warm-up per variant; GPU routing versus CPU MiniLM. Latency is exploratory."
            )
            report["limitations"] += [
                "The earlier Ollama Q4_K_M run is not the controlled base for this NF4 experiment.",
                "Development questions partly appear in training/validation; dev is diagnostic only.",
                "This public test set was measured previously; it is held out from gradients, not a new blind benchmark.",
            ]
            (args.output / (variant + ".json")).write_text(json.dumps(report, indent=2) + "\n")
            reports[variant] = report
        result = compare(
            reports["base"],
            reports["fine-tuned"],
            exploratory=not all(r["complete"] for r in reports.values()),
        )
        (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"stage": "complete", "comparison": result}), flush=True)
    finally:
        with psycopg.connect(admin, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database)))
            conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))


if __name__ == "__main__":
    main()
