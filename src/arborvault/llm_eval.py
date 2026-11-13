"""Run a bounded Ollama retrieval evaluation, then compare base and adapted checkpoints."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import httpx

from .benchmark import run
from .ollama import PROMPT_SHA256, OllamaSelector, thinking_value


def local_model_provenance(url, model, client=None):
    """Record a portable allowlist; Ollama's full Modelfile can contain private local paths."""
    client = client or httpx
    base = url.rstrip("/")
    version_response = client.get(base + "/version", timeout=10, follow_redirects=False)
    version_response.raise_for_status()
    tags = client.get(base + "/tags", timeout=10, follow_redirects=False)
    tags.raise_for_status()
    tag = next(
        (r for r in tags.json()["models"] if r["name"] == model or r.get("model") == model), None
    )
    if tag is None:
        raise ValueError("Requested model is not installed locally")
    show = client.post(base + "/show", json={"model": model}, timeout=10, follow_redirects=False)
    show.raise_for_status()
    metadata = show.json()
    return {
        "ollama_version": version_response.json()["version"],
        "model_digest": tag["digest"],
        "model_bytes": tag["size"],
        "details": {
            key: metadata.get("details", {}).get(key)
            for key in ("format", "family", "parameter_size", "quantization_level")
        },
        "template_sha256": hashlib.sha256(metadata.get("template", "").encode()).hexdigest(),
        "parameters_sha256": hashlib.sha256(metadata.get("parameters", "").encode()).hexdigest(),
        "capabilities": metadata.get("capabilities", []),
        "thinking": metadata.get("thinking"),
        "template_source": metadata.get("template_source"),
        "model_identity": {
            key: metadata.get("model_info", {}).get(key)
            for key in (
                "general.basename",
                "general.finetune",
                "general.version",
                "general.parameter_count",
            )
        },
        "base_revision_note": "Ollama catalog digest identifies the served artifact; its exact upstream Hugging Face commit is not attested.",
    }


def load_local_settings(path):
    settings = dict(os.environ)
    if path and path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition("=")
            if separator and name in {"SELECTOR_URL", "SELECTOR_MODEL", "SELECTOR_API_KEY"}:
                settings.setdefault(name, value.strip())
    return settings


def compare(base, tuned, exploratory=False):
    if base["experiment"]["variant"] != "base" or tuned["experiment"]["variant"] != "fine-tuned":
        raise ValueError("Supply base first and fine-tuned second")
    required = ["dataset_sha256", "split", "k", "word_budget"]
    experiment = [
        "base_model",
        "base_revision",
        "prompt_sha256",
        "max_output_tokens",
        "temperature",
        "seed",
    ]
    for key in required:
        if base[key] != tuned[key]:
            raise ValueError("Incompatible evaluation setting: " + key)
    for key in experiment:
        if base["experiment"][key] != tuned["experiment"][key]:
            raise ValueError("Incompatible model setting: " + key)
    for key in ("think", "num_ctx"):
        if base["experiment"].get(key) != tuned["experiment"].get(key):
            raise ValueError("Incompatible model setting: " + key)
    a_provenance, b_provenance = base.get("served_artifact", {}), tuned.get("served_artifact", {})
    for key in ("template_sha256", "parameters_sha256"):
        if a_provenance.get(key) != b_provenance.get(key):
            raise ValueError("Incompatible serving metadata: " + key)
    differences = []
    for key in ("runtime_id", "quantization", "provider"):
        a, b = base["experiment"][key], tuned["experiment"][key]
        if a != b or a == "unknown":
            differences.append(key)
    if base["experiment"]["base_revision"] == "unknown":
        differences.append("base_revision")
    if any(r["experiment"]["provider"] == "ollama-cloud" for r in (base, tuned)):
        differences.append("cloud_checkpoint_not_independently_verified")
    if any(not r.get("complete") or r["provider_usage"]["failures"] for r in (base, tuned)):
        differences.append("incomplete_or_failed_provider_calls")
    if differences and not exploratory:
        raise ValueError("Controlled comparison unavailable: " + ", ".join(differences))
    deltas = {}
    for method in ("llm_navigation", "llm_auto"):
        a, b = base["methods"][method], tuned["methods"][method]
        if [r["id"] for r in a["queries"]] != [r["id"] for r in b["queries"]]:
            raise ValueError("Query set or order differs")
        deltas[method] = {
            key: b[key] - a[key] for key in ("recall_at_k", "mrr", "ndcg", "evidence_coverage")
        }
    return {
        "comparison": "exploratory" if differences else "paired_with_declared_provenance",
        "limitations": differences
        + [
            "Checkpoint provenance and runtime metadata are operator declarations.",
            "Small draft labels do not establish statistical significance.",
        ],
        "delta_fine_tuned_minus_base": deltas,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("directory", type=Path)
    run_parser.add_argument("--env-file", type=Path, default=Path(".env"))
    run_parser.add_argument("--model")
    run_parser.add_argument("--url", help="Ollama API base URL, including /api")
    run_parser.add_argument("--variant", choices=["base", "fine-tuned"], required=True)
    run_parser.add_argument("--base-model", default="openai/gpt-oss-20b")
    run_parser.add_argument("--base-revision", default="unknown")
    run_parser.add_argument("--runtime-id", default="unknown")
    run_parser.add_argument("--quantization", default="unknown")
    run_parser.add_argument("--split", choices=["dev", "test"], default="dev")
    run_parser.add_argument("--max-calls", type=int, required=True)
    run_parser.add_argument("--max-output-tokens", type=int, default=1024)
    run_parser.add_argument(
        "--thinking", choices=["default", "on", "off", "low", "medium", "high"], default="default"
    )
    run_parser.add_argument("--num-ctx", type=int)
    run_parser.add_argument("--cache-dir", default=".cache/models")
    run_parser.add_argument(
        "--inspect-local",
        action="store_true",
        help="Record and check the installed Ollama artifact digest",
    )
    run_parser.add_argument("--dense", action="store_true")
    run_parser.add_argument("--output", type=Path, required=True)
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("base", type=Path)
    compare_parser.add_argument("fine_tuned", type=Path)
    compare_parser.add_argument("--allow-exploratory", action="store_true")
    compare_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "compare":
        report = compare(
            json.loads(args.base.read_text()),
            json.loads(args.fine_tuned.read_text()),
            args.allow_exploratory,
        )
    else:
        settings = load_local_settings(args.env_file)
        url = args.url or settings.get("SELECTOR_URL", "https://ollama.com/api")
        model = args.model or settings.get("SELECTOR_MODEL", "gpt-oss:20b")
        key = settings.get("SELECTOR_API_KEY", "")
        if "ollama.com" in url and not key:
            parser.error("Configure Ollama access locally with scripts/configure_ollama.py")
        if args.base_revision != "unknown" and not re.fullmatch(
            r"[0-9a-f]{40}", args.base_revision
        ):
            parser.error("Base revision must be an immutable Hugging Face commit SHA")
        if "EVAL_DATABASE_URL" not in settings:
            parser.error("Set EVAL_DATABASE_URL to a disposable migrated database runtime role")
        selector = OllamaSelector(
            url,
            model,
            key,
            max_calls=args.max_calls,
            max_output_tokens=args.max_output_tokens,
            think=thinking_value(args.thinking),
            num_ctx=args.num_ctx,
        )
        provenance = local_model_provenance(url, model) if args.inspect_local else None
        report = run(
            args.directory,
            settings["EVAL_DATABASE_URL"],
            split=args.split,
            dense=args.dense,
            selector=selector,
            cache_dir=args.cache_dir,
        )
        report["experiment"] = {
            "variant": args.variant,
            "served_model": model,
            "base_model": args.base_model,
            "base_revision": args.base_revision,
            "runtime_id": args.runtime_id,
            "quantization": args.quantization,
            "provider": "ollama-cloud"
            if "ollama.com" in url or model.endswith("-cloud")
            else "ollama",
            "prompt_sha256": PROMPT_SHA256,
            "max_output_tokens": args.max_output_tokens,
            "temperature": 0,
            "seed": 0,
            "max_calls": args.max_calls,
            "think": thinking_value(args.thinking),
            "num_ctx": args.num_ctx,
        }
        report["provider_usage"] = selector.usage()
        if provenance:
            report["served_artifact"] = provenance
            report["artifact_unchanged"] = provenance == local_model_provenance(url, model)
        report["complete"] = not any(
            q["fallback_reason"] in {"selector_error", "invalid_selection"}
            for method in ("llm_navigation", "llm_auto")
            for q in report["methods"][method]["queries"]
        )
        report["complete"] = report["complete"] and report.get("artifact_unchanged", True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k in {"complete", "provider_usage", "comparison", "delta_fine_tuned_minus_base"}
            }
        )
    )


if __name__ == "__main__":
    main()
