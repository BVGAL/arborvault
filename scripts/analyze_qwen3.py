"""Offline diagnosis of frozen Qwen3 reports; never exports test labels for training."""

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def normalized_template(query):
    # The v1 procedural generator appends a random ten-character hex identifier.
    return " ".join(re.sub(r"\b[0-9a-f]{10}\b", "<entity>", query).casefold().split())


def training_audit(directory):
    audit, templates = {}, {}
    for split in ("train", "validation", "synthetic_test"):
        rows = jsonl(directory / f"{split}.jsonl")
        origins, counts, sizes, depths, positive_documents = (Counter() for _ in range(5))
        public, procedural_templates = [], set()
        for row in rows:
            payload = json.loads(row["messages"][1]["content"])
            target = json.loads(row["messages"][2]["content"])
            candidates = {c["id"]: c for c in payload["candidates"]}
            origins[row["provenance"]] += 1
            counts[str(len(candidates))] += 1
            sizes[str(len(target))] += 1
            if row["provenance"] == "procedural-original-v1":
                procedural_templates.add(normalized_template(payload["query"]))
            else:
                depth = int(row["id"].rsplit("/", 1)[1])
                depths[str(depth)] += 1
                for label in target:
                    positive_documents[candidates[label]["path"][0]] += 1
                public.append({"id": row["id"], "depth": depth, "targets": len(target)})
        templates[split] = procedural_templates
        audit[split] = {
            "examples": len(rows),
            "origins": dict(origins),
            "candidate_count_histogram": dict(counts),
            "target_count_histogram": dict(sizes),
            "public_depth_histogram": dict(depths),
            "public_positive_documents": dict(positive_documents),
            "public_routes": public,
            "unique_procedural_query_templates": len(procedural_templates),
        }
    audit["train_validation_shared_query_templates"] = len(
        templates["train"] & templates["validation"]
    )
    audit["template_overlap_definition"] = "Exact query after replacing procedural entity IDs"
    return audit


def outcome(row):
    if row["recall"] == 1:
        return "full_recall"
    if row["recall"] > 0:
        return "partial_recall"
    if row["provider_usage"].get("failures", 0):
        return "selector_failure"
    if row["abstained"]:
        if row["fallback_reason"] == "no_route" and row["provider_usage"]["calls"] == 1:
            return "root_rejection"
        return "empty_after_deeper_calls"
    gold_documents = {x.rsplit(":", 1)[0] for x in row["gold"]}
    if any(x.rsplit(":", 1)[0] in gold_documents for x in row["predicted"]):
        return "miss_with_correct_document_in_results"
    return "miss_with_other_documents_only"


def paired_audit(base, tuned, queries):
    if any(base[k] != tuned[k] for k in ("dataset_sha256", "split", "k", "word_budget")):
        raise ValueError("Reports use different evaluation inputs")
    b = {r["id"]: r for r in base["methods"]["llm_navigation"]["queries"]}
    t = {r["id"]: r for r in tuned["methods"]["llm_navigation"]["queries"]}
    dense = {r["id"]: r for r in base["methods"]["dense_sections"]["queries"]}
    if set(b) != set(t) or set(b) != set(dense):
        raise ValueError("Reports have different query IDs")
    for identity in b:
        if b[identity]["gold"] != t[identity]["gold"]:
            raise ValueError("Reports have different relevance labels")
    answerable = [identity for identity, row in b.items() if row["gold"]]
    result = {"answerable": len(answerable), "unknown": len(b) - len(answerable)}
    for variant, rows in (("base", b), ("tuned", t)):
        result[variant] = {
            "outcomes": dict(Counter(outcome(rows[i]) for i in answerable)),
            "empty_answerable": sum(rows[i]["abstained"] for i in answerable),
            "unknown_abstentions": sum(r["abstained"] for r in rows.values() if not r["gold"]),
            "selector_failure_queries": [
                i for i, r in rows.items() if r["provider_usage"].get("failures", 0)
            ],
            "selector_failure_with_retained_evidence": [
                i
                for i, r in rows.items()
                if r["provider_usage"].get("failures", 0) and r["predicted"]
            ],
        }
    result["lost"] = [i for i in answerable if t[i]["recall"] < b[i]["recall"]]
    result["gained"] = [i for i in answerable if t[i]["recall"] > b[i]["recall"]]
    result["dense_success_tuned_miss"] = [
        i for i in answerable if dense[i]["recall"] == 1 and t[i]["recall"] < 1
    ]
    result["root_rejection_lost"] = [i for i in result["lost"] if outcome(t[i]) == "root_rejection"]
    result["by_group"], result["by_language"] = {}, {}
    for field in ("group", "language"):
        groups = defaultdict(list)
        for identity in answerable:
            groups[queries[identity][field]].append(identity)
        for group, ids in sorted(groups.items()):
            result["by_" + field][group] = {
                "questions": len(ids),
                **{
                    name: statistics.mean(rows[i]["recall"] for i in ids)
                    for name, rows in (("base", b), ("tuned", t), ("dense", dense))
                },
            }
    result["queries"] = [
        {
            "id": i,
            "question": queries[i]["question"],
            "gold": b[i]["gold"],
            "base_recall": b[i]["recall"],
            "tuned_recall": t[i]["recall"],
            "dense_recall": dense[i]["recall"],
            "base_outcome": outcome(b[i]),
            "tuned_outcome": outcome(t[i]),
            "base_predicted": b[i]["predicted"],
            "tuned_predicted": t[i]["predicted"],
            "base_calls": b[i]["provider_usage"]["calls"],
            "tuned_calls": t[i]["provider_usage"]["calls"],
        }
        for i in answerable
    ]
    return result


def oracle_audit(indexed, queries, k, word_budget):
    """Label-informed routing diagnostic; never a deployable baseline or training export."""
    from arborvault.benchmark import budgeted
    from arborvault.evaluate import metrics
    from arborvault.retrieval import _retrieve, routing_summaries

    nodes = {
        n.id: (SimpleNamespace(source_key=key), n) for key, doc in indexed.items() for n in doc
    }
    summaries = routing_summaries([n for _, n in nodes.values()])

    class Store:
        def children(self, tenant, namespace, parents, query, limit):
            candidates = [
                (d, replace(n, content=summaries[n.id]))
                for d, n in nodes.values()
                if n.parent_id in ([None] if parents is None else parents)
            ]
            # Every frontier is below the shortlist limit; oracle decisions ignore ordering.
            assert len(candidates) < limit
            return candidates

        def passages(self, tenant, namespace, picked):
            return sorted([nodes[i] for i in picked], key=lambda p: (p[0].source_key, p[1].ordinal))

    class Oracle:
        def select(self, query, candidates, width):
            picked = [n.id for n in candidates if n.id in ancestors]
            assert len(picked) <= width
            return picked

    result = []
    for query in queries:
        if not query["relevant"]:
            continue
        gold_ids = {
            n.id
            for ref in query["relevant"]
            for n in indexed[ref["document"]]
            if n.locator.get("line", 0) == ref["section_line"] and ref["evidence"] in n.content
        }
        ancestors = set(gold_ids)
        visibility = []
        for identity in gold_ids:
            d, gold_node = nodes[identity]
            root, depth = gold_node, 0
            while root.parent_id:
                root = nodes[root.parent_id][1]
                ancestors.add(root.id)
                depth += 1
            visibility.append(
                {
                    "source": f"{d.source_key}:{gold_node.ordinal}",
                    "title": gold_node.title,
                    "depth": depth,
                    "title_in_root_excerpt": gold_node.title.casefold()
                    in summaries[root.id][:240].casefold(),
                    "root_excerpt": summaries[root.id][:240],
                }
            )
        hits, trace, _, _ = _retrieve(
            Store(), "diagnostic", "public", query["question"], k, Oracle(), "navigation"
        )
        selected = budgeted(hits, k, word_budget)
        predicted = [f"{d.source_key}:{n.ordinal}" for d, n, *_ in selected]
        gold = {f"{nodes[i][0].source_key}:{nodes[i][1].ordinal}" for i in gold_ids}
        coverage = sum(
            any(
                d.source_key == ref["document"] and ref["evidence"] in text
                for d, _, _, _, text in selected
            )
            for ref in query["relevant"]
        ) / len(query["relevant"])
        result.append(
            {
                "id": query["id"],
                "predicted": predicted,
                "gold": sorted(gold),
                **metrics(predicted, gold, k),
                "evidence_coverage": coverage,
                "calls": len(trace),
                "gold_routes": visibility,
            }
        )
    return {
        "definition": "Gold ancestor oracle with unchanged navigation ranking, k and word budget; in-memory traversal, no model calls",
        "recall_at_k": statistics.mean(r["recall"] for r in result),
        "evidence_coverage": statistics.mean(r["evidence_coverage"] for r in result),
        "gold_depth_histogram": dict(
            Counter(str(g["depth"]) for r in result for g in r["gold_routes"])
        ),
        "gold_titles_visible_in_root_excerpt": sum(
            g["title_in_root_excerpt"] for r in result for g in r["gold_routes"]
        ),
        "queries": result,
    }


def main():
    from arborvault.dataset import load_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("reports/qwen3-error-analysis.json"))
    args = parser.parse_args()
    root = args.root
    base_path = root / "docs/benchmarks/qwen3-nf4-base-test.json"
    tuned_path = root / "docs/benchmarks/qwen3-nf4-tuned-test.json"
    base, tuned = read(base_path), read(tuned_path)
    _, _, queries, indexed, digest = load_dataset(root / "datasets/public-docs-v1")
    if base["dataset_sha256"] != digest:
        raise ValueError("Dataset differs from measured report")
    query_map = {q["id"]: q for q in queries}
    report = {
        "scope": "Post-hoc diagnostic of v1; test must not become a training source",
        "input_sha256": {
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                base_path,
                tuned_path,
                root / "datasets/routing-bootstrap-v1/train.jsonl",
                root / "datasets/routing-bootstrap-v1/validation.jsonl",
            ]
        },
        "training": training_audit(root / "datasets/routing-bootstrap-v1"),
        "paired": paired_audit(base, tuned, query_map),
        "oracle": oracle_audit(
            indexed, [q for q in queries if q["split"] == "test"], base["k"], base["word_budget"]
        ),
        "limits": [
            "No original candidate/selection trace or raw invalid output was retained.",
            "Only first-call no_route proves root rejection; nonempty misses do not prove the exact failed depth.",
            "Heading visibility is a literal diagnostic, not a semantic sufficiency test.",
            "Oracle uses test labels and is diagnostic only; it is not a model measurement.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "training": report["training"],
                "paired": {k: v for k, v in report["paired"].items() if k != "queries"},
                "oracle": {k: v for k, v in report["oracle"].items() if k != "queries"},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
