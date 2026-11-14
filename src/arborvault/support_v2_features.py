"""Whole-pair frozen features, reusing only the old fit/selection cache by hash."""

import json
import math
import time
from pathlib import Path

import numpy as np

from .support_data import file_hash
from .support_probe import MODES, ROLES, encode_batch, input_key, model_input, write_json
from .support_v2_data import load


def base_cache(config):
    path = Path(config["cached_features"])
    if file_hash(path / "manifest.json") != config["cached_feature_manifest_sha256"]:
        raise ValueError("Original feature manifest changed")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    for key in ("model", "revision", "feature_decimals"):
        if manifest[key] != config[key]:
            raise ValueError("Old feature representation does not match")
    result, sources = {}, {}
    for role in ("fit", "selection"):
        file = path / (role + ".json")
        if file_hash(file) != manifest["roles"][role]["sha256"]:
            raise ValueError("Original feature shard changed")
        shard = json.loads(file.read_text(encoding="utf-8"))
        sources[role] = {"path": file.as_posix(), "sha256": file_hash(file)}
        for key, value in shard["features"].items():
            result[key] = {"source_role": role, **value}
    return result, sources


def extract(config, output, config_path, deadline):
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if output.exists():
        raise ValueError("Refusing to overwrite feature extraction")
    dataset, rows = load(Path(config["dataset"]))
    cache, sources = base_cache(config)
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
    tokenizer.padding_side = "right"
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            config["model"],
            revision=config["revision"],
            attn_implementation="eager",
            use_safetensors=True,
            trust_remote_code=False,
        )
        .cpu()
        .eval()
    )
    output.mkdir(parents=True)
    manifest = {
        "config_sha256": file_hash(config_path),
        "dataset_manifest_sha256": file_hash(Path(config["dataset"]) / "manifest.json"),
        "examples_sha256": dataset["examples_sha256"],
        "model": config["model"],
        "revision": config["revision"],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "cpu_threads": config["cpu_threads"],
        "feature_decimals": config["feature_decimals"],
        "representation": "Frozen last-layer CLS, float32 eager attention, whole question/context; 384 dimensions",
        "sources": sources,
        "roles": {},
    }
    for role in ROLES:
        start, records, jobs, reused = time.monotonic(), {}, {}, {}
        for row in rows:
            if row["role"] != role:
                continue
            records[row["id"]] = {}
            for mode in MODES:
                parts = model_input(row, mode)
                key = input_key(parts)
                records[row["id"]][mode] = key
                if key in cache:
                    reused[key] = cache[key]["source_role"]
                else:
                    jobs[key] = parts
        features, items = {}, list(jobs.items())
        with torch.inference_mode():
            for offset in range(0, len(items), config["batch_size"]):
                deadline()
                batch = items[offset : offset + config["batch_size"]]
                encoded, tensors = encode_batch(
                    tokenizer, [parts for _, parts in batch], config["max_length"]
                )
                result = model(**tensors, output_hidden_states=True)
                vectors, logits = (
                    result.hidden_states[-1][:, 0].float().tolist(),
                    result.logits[:, 0].float().tolist(),
                )
                for (key, _), vector, logit, tokens in zip(
                    batch, vectors, logits, encoded, strict=True
                ):
                    if len(vector) != 384 or not all(map(math.isfinite, vector + [logit])):
                        raise ValueError("Invalid encoder features")
                    features[key] = {
                        "vector": [round(v, config["feature_decimals"]) for v in vector],
                        "logit": round(logit, config["feature_decimals"]),
                        "tokens": len(tokens["input_ids"]),
                    }
                if offset % (16 * config["batch_size"]) == 0:
                    print(
                        json.dumps(
                            {
                                "role": role,
                                "completed": len(features),
                                "new_forwards": len(items),
                                "reused": len(reused),
                                "seconds": round(time.monotonic() - start, 2),
                            }
                        ),
                        flush=True,
                    )
        path = output / (role + ".json")
        write_json(path, {"records": records, "features": features, "reused": reused}, compact=True)
        manifest["roles"][role] = {
            "sha256": file_hash(path),
            "examples": len(records),
            "new_forwards": len(features),
            "reused": len(reused),
            "seconds": time.monotonic() - start,
        }
        write_json(output / "manifest.json", manifest)
        print(json.dumps({"finished": role, **manifest["roles"][role]}), flush=True)
    return manifest


def read_role(directory, rows, role, cache):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    path = directory / (role + ".json")
    if file_hash(path) != manifest["roles"][role]["sha256"]:
        raise ValueError("Feature shard hash mismatch")
    shard = json.loads(path.read_text(encoding="utf-8"))
    subset = [r for r in rows if r["role"] == role]
    if set(shard["records"]) != {r["id"] for r in subset}:
        raise ValueError("Feature partition IDs differ")
    result = {}
    for mode in MODES:
        vectors, logits = [], []
        for row in subset:
            key = shard["records"][row["id"]][mode]
            if key != input_key(model_input(row, mode)):
                raise ValueError("Input provenance mismatch")
            if key in shard["reused"]:
                feature = cache[key]
                if shard["reused"][key] != feature["source_role"]:
                    raise ValueError("Reused feature provenance mismatch")
            else:
                feature = shard["features"][key]
            if mode == "context" and feature["tokens"] != row["input_tokens"]:
                raise ValueError("Question/context was truncated")
            vectors.append(feature["vector"])
            logits.append(feature["logit"])
        result[mode] = np.asarray(vectors, dtype=float)
        if mode == "context":
            result["raw"] = logits
    return subset, result
