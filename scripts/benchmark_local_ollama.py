"""Evaluate an installed Ollama model against matched baselines in a disposable PostgreSQL DB."""

import argparse
import json
import os
import secrets
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

import httpx
import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from arborvault.benchmark import run
from arborvault.llm_eval import local_model_provenance
from arborvault.migrate import migrate, provision_runtime_role
from arborvault.ollama import PROMPT_SHA256, PROMPT_VERSION, OllamaSelector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/public-docs-v1"))
    parser.add_argument("--url", default="http://host.docker.internal:11434/api")
    parser.add_argument("--model", default="qwen3:4b-q4_K_M")
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", default=".cache/models")
    args = parser.parse_args()
    if "TEST_DATABASE_URL" not in os.environ:
        parser.error(
            "Set TEST_DATABASE_URL to an isolated test server with database-creation permission"
        )
    # The settings below are fixed before reading any test outcomes.
    selector = OllamaSelector(
        args.url,
        args.model,
        think=False,
        num_ctx=8192,
        max_output_tokens=256,
        max_calls=1056,
        timeout=120,
    )
    provenance = local_model_provenance(args.url, args.model)
    thinking = provenance.get("thinking")
    if thinking and False not in thinking.get("values", []):
        parser.error(
            "This installed variant does not support non-thinking mode; check its explicit tag"
        )
    warmup = httpx.post(
        args.url.rstrip("/") + "/chat",
        timeout=120,
        follow_redirects=False,
        json={
            "model": args.model,
            "stream": False,
            "think": False,
            "messages": [
                {
                    "role": "user",
                    "content": "Return only the JSON array [] for installation verification.",
                }
            ],
            "options": {"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": 256},
        },
    )
    warmup.raise_for_status()
    body = warmup.json()
    if body.get("done_reason") == "length" or json.loads(body["message"]["content"]) != []:
        raise RuntimeError("Non-thinking JSON warm-up failed; evaluation was not started")
    loaded = httpx.get(args.url.rstrip("/") + "/ps", timeout=10)
    loaded.raise_for_status()
    model = next(r for r in loaded.json()["models"] if r["name"] == args.model)
    if model.get("context_length") != 8192:
        raise RuntimeError("Server did not apply the requested context length")
    loaded_metadata = {k: model.get(k) for k in ("size", "size_vram", "context_length", "runner")}
    admin = os.environ["TEST_DATABASE_URL"]
    database = "arborvault_eval_" + uuid4().hex[:16]
    role = database + "_app"
    options = conninfo_to_dict(admin)
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    try:
        options["dbname"] = database
        temporary_admin = make_conninfo(**options)
        migrate(temporary_admin)
        password = secrets.token_urlsafe(32)
        provision_runtime_role(temporary_admin, password, role)
        options.update(user=role, password=password)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.with_suffix(".progress.jsonl").open("w", encoding="utf-8") as stream:

            def progress(method, identity, row):
                # No source text, credentials, machine paths or model reasoning are logged.
                event = {
                    "method": method,
                    "query_id": identity,
                    "latency_ms": row["latency_ms"],
                    "fallback_reason": row["fallback_reason"],
                    "provider_usage": row["provider_usage"],
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
        report["experiment"] = {
            "variant": "base",
            "base_model": args.base_model,
            "base_revision": "unknown",
            "served_model": args.model,
            "runtime_id": "ollama-" + provenance["ollama_version"],
            "provider": "ollama",
            "quantization": provenance["details"]["quantization_level"],
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": PROMPT_SHA256,
            "temperature": 0,
            "seed": 0,
            "think": False,
            "num_ctx": 8192,
            "max_output_tokens": 256,
            "max_calls": 1056,
        }
        report["served_artifact"] = provenance
        report["code_revision"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            if (Path.cwd() / ".git").exists() and shutil.which("git")
            else os.getenv("ARBORVAULT_CODE_REVISION", "unknown")
        )
        report["artifact_unchanged"] = provenance == local_model_provenance(args.url, args.model)
        report["loaded_model"] = loaded_metadata
        report["provider_usage"] = selector.usage()
        report["provider_max_prompt_tokens"] = max(
            (e["prompt_tokens"] for e in selector.events), default=0
        )
        report["complete"] = report["artifact_unchanged"] and not any(
            row["fallback_reason"] in {"selector_error", "invalid_selection"}
            for method in ("llm_navigation", "llm_auto")
            for row in report["methods"][method]["queries"]
        )
        report["timing_protocol"] = (
            "One sequential pass, fixed method/query order. One synthetic Ollama warm-up excluded. GPU routing versus CPU MiniLM. Prefix caches are not cleared; timings are exploratory."
        )
        report["limitations"] += [
            "The base checkpoint is identified by Ollama artifact digest; exact upstream Hugging Face revision is not attested.",
            "Non-thinking greedy decoding is fixed for reproducibility, not a claim of the model's optimal settings.",
            "No fine-tuning or generated-answer scoring was performed.",
        ]
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "complete": report["complete"],
                    "provider_usage": report["provider_usage"],
                    "metrics": {
                        name: {k: v for k, v in data.items() if k != "queries"}
                        for name, data in report["methods"].items()
                    },
                }
            ),
            flush=True,
        )
    finally:
        with psycopg.connect(admin, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database)))
            conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))


if __name__ == "__main__":
    main()
