"""PostgreSQL-backed retrieval evaluation on identical source sections and a shared word budget."""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import time
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from .dataset import load_dataset
from .evaluate import DenseIndex, metrics
from .parsing import build_nodes
from .retrieval import TitleSelector, retrieve
from .storage import PostgresStore


def budgeted(rows, k, word_budget):
    selected, remaining = [], word_budget
    for doc, node, score, method in rows[:k]:
        words = list(re.finditer(r"\S+", node.content))
        if not words or remaining <= 0:
            break
        count = min(remaining, len(words))
        text = node.content[: words[count - 1].end()]
        selected.append((doc, node, score, method, text))
        remaining -= count
    return selected


def run(
    directory,
    url,
    split="dev",
    dense=False,
    k=5,
    word_budget=512,
    cache_dir=None,
    selector=None,
    progress=None,
):
    manifest, documents, queries, _, dataset_hash = load_dataset(directory)
    queries = [q for q in queries if q["split"] == split]
    store = PostgresStore(url, enforce_runtime_role=True)
    tenant, namespace = "eval-" + uuid4().hex, "public-docs-v1"
    created, report = [], {}
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    try:
        for key, doc in documents.items():
            record, _ = store.upsert(
                tenant,
                namespace,
                key,
                doc["title"],
                doc["source_url"],
                doc["sha256"],
                lambda i, r, sections=doc["sections"], title=doc["title"]: build_nodes(
                    sections, i, r, title
                ),
            )
            created.append(record.id)
        pairs = sorted(
            [(d, n) for d, n in store.nodes(tenant, namespace) if n.content],
            key=lambda pair: (pair[0].source_key, pair[1].ordinal),
        )
        start = time.perf_counter()
        index = (
            DenseIndex(
                [" / ".join(n.path) + "\n" + n.content for _, n in pairs], model_name, cache_dir
            )
            if dense
            else None
        )
        build_ms = (time.perf_counter() - start) * 1000
        methods = ["navigation", "auto", "lexical"] + (["dense_sections"] if dense else [])
        if selector is not None:
            methods += ["llm_navigation", "llm_auto"]
        for method in methods:
            rows = []
            for query in queries:
                start = time.perf_counter()
                reason = None
                before = selector.usage() if selector is not None else {}
                if method == "dense_sections":
                    hits = [(*pairs[i], 0.0, "dense") for i in index.search(query["question"])[:k]]
                    fallback = False
                else:
                    hits, _, fallback, reason = retrieve(
                        store,
                        tenant,
                        namespace,
                        query["question"],
                        k,
                        selector if method.startswith("llm_") else TitleSelector(),
                        method.removeprefix("llm_"),
                    )
                selected = budgeted(hits, k, word_budget)
                elapsed = (time.perf_counter() - start) * 1000
                # Stable source IDs in reports; no ephemeral tenant IDs or database URL.
                predicted = [f"{d.source_key}:{n.ordinal}" for d, n, _, _, _ in selected]
                gold = {
                    f"{d.source_key}:{n.ordinal}"
                    for ref in query["relevant"]
                    for d, n in pairs
                    if d.source_key == ref["document"]
                    and n.locator.get("line", 0) == ref["section_line"]
                    and ref["evidence"] in n.content
                }
                evidence_coverage = sum(
                    any(
                        d.source_key == ref["document"] and ref["evidence"] in text
                        for d, _, _, _, text in selected
                    )
                    for ref in query["relevant"]
                ) / max(1, len(query["relevant"]))
                rows.append(
                    {
                        "id": query["id"],
                        "language": query["language"],
                        "category": query["category"],
                        "gold": sorted(gold),
                        "predicted": predicted,
                        "latency_ms": elapsed,
                        "fallback": fallback,
                        "fallback_reason": reason,
                        "provider_usage": {
                            key: value - before[key] for key, value in selector.usage().items()
                        }
                        if selector is not None
                        else {},
                        "returned_words": sum(len(text.split()) for *_, text in selected),
                        "evidence_coverage": evidence_coverage,
                        **metrics(predicted, gold, k),
                    }
                )
                if progress is not None:
                    progress(method, query["id"], rows[-1])
            answerable = [r for r in rows if r["gold"]]
            unknown = [r for r in rows if not r["gold"]]
            latencies = sorted(r["latency_ms"] for r in rows)
            report[method] = {
                "recall_at_k": statistics.mean(r["recall"] for r in answerable),
                "mrr": statistics.mean(r["mrr"] for r in answerable),
                "ndcg": statistics.mean(r["ndcg"] for r in answerable),
                "evidence_coverage": statistics.mean(r["evidence_coverage"] for r in answerable),
                "unanswerable_abstention": statistics.mean(r["abstained"] for r in unknown)
                if unknown
                else None,
                "latency_p50_ms": statistics.median(latencies),
                "latency_p95_ms": latencies[max(0, math.ceil(0.95 * len(latencies)) - 1)],
                "queries": rows,
            }
        model_hashes = {}
        if dense and cache_dir:
            for path in sorted(Path(cache_dir).rglob("*.onnx")):
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                model_hashes[str(path.relative_to(cache_dir))] = digest.hexdigest()
        return {
            "dataset": manifest["id"],
            "dataset_sha256": dataset_hash,
            "split": split,
            "annotation_status": "assistant_draft_needs_independent_review",
            "backend": "PostgreSQL",
            "k": k,
            "word_budget": word_budget,
            "unit": "same extracted sections for all methods",
            "model": model_name if dense else None,
            "model_sha256": model_hashes,
            "python": platform.python_version(),
            "arborvault": version("arborvault"),
            "packages": {
                name: version(name)
                for name in (["psycopg", "numpy", "fastembed"] if dense else ["psycopg"])
            },
            "documents": len(documents),
            "content_sections": len(pairs),
            "timing_protocol": "single sequential pass; PostgreSQL paths include database I/O, dense index is in memory",
            "vector_build_ms": build_ms,
            "methods": report,
            "limitations": [
                "Single small developer-documentation corpus",
                "Model-authored draft qrels",
                "Budget counts whitespace words, not model tokens",
                "No answer generation evaluated",
                "MiniLM is primarily English; its default tokenizer truncates long section inputs",
                "Latency is exploratory, not a controlled database performance comparison",
                "Public test set; do not tune on reported test results",
            ],
        }
    finally:
        for identity in created:
            store.delete(tenant, namespace, identity)
        store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--k", type=int, choices=range(1, 21), default=5)
    parser.add_argument("--word-budget", type=int, default=512)
    parser.add_argument("--cache-dir", default=".cache/models")
    parser.add_argument("--output", type=Path, default=Path("reports/public-docs.json"))
    args = parser.parse_args()
    if args.word_budget < 1:
        parser.error("Word budget must be positive")
    report = run(
        args.directory,
        os.environ["EVAL_DATABASE_URL"],
        args.split,
        args.dense,
        args.k,
        args.word_budget,
        args.cache_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "split": args.split,
                "methods": {
                    name: {k: v for k, v in values.items() if k != "queries"}
                    for name, values in report["methods"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
