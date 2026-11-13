"""Freeze a reproducible 50-failure / five-control, model-masked evidence audit."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from arborvault.qasper import load_split, normalize
from arborvault.qwen3 import file_hash

SALT = "arborvault-routing-v2-audit-20261113"


def order(value):
    return hashlib.sha256((SALT + value).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/qasper-routing-v2"))
    parser.add_argument("--reports", type=Path, default=Path("docs/benchmarks/routing-v2"))
    parser.add_argument("--original-dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest, trees, queries = load_split(args.dataset, "dev")
    if file_hash(args.original_dev) != manifest["source_sha256"]["qasper-dev-v0.3.json"]:
        raise ValueError("Original source checksum mismatch")
    original = json.loads(args.original_dev.read_text(encoding="utf-8"))
    names = ["base-dev", "seed-42-step-88-dev", "seed-42-step-175-dev"]
    reports = [json.loads((args.reports / (n + ".json")).read_text()) for n in names]
    indexed = [{q["id"]: q for q in r["methods"]["navigation"]["queries"]} for r in reports]
    strata = defaultdict(list)
    for query in queries:
        if not query["answerable"]:
            continue
        recalls = [r[query["id"]]["recall"] for r in indexed]
        if min(recalls) == 1:
            stratum = "control"
        elif recalls[0] > min(recalls[1:]):
            stratum = "regression"
        elif max(recalls[1:]) > recalls[0]:
            stratum = "partial_gain"
        else:
            stratum = "shared_failure"
        strata[stratum].append(query)
    quotas = {"partial_gain": 3, "regression": 26, "shared_failure": 21, "control": 5}
    selected, used = [], set()
    for stratum, quota in quotas.items():
        count = 0
        for q in sorted(strata[stratum], key=lambda q: order(q["id"])):
            if q["document"] in used:
                continue
            used.add(q["document"])
            selected.append((stratum, q))
            count += 1
            if count == quota:
                break
        if count != quota:
            raise ValueError("Insufficient distinct documents in " + stratum)
    packet, key = [], []
    for number, (stratum, query) in enumerate(
        sorted(selected, key=lambda item: order(item[1]["id"])), 1
    ):
        case = f"A{number:02d}"
        tree = trees[query["document"]]
        predictions = {i for report in indexed for i in report[query["id"]]["predicted"]}
        gold = {i for group in query["gold_sets"] for i in group}
        paper = original[query["document"]]
        source_texts = {normalize(p) for s in paper["full_text"] for p in s["paragraphs"]}
        if not all(normalize(tree.nodes[i].content) in source_texts for i in gold | predictions):
            raise ValueError("Paragraph is not present in original source")
        source_qa = next(q for q in paper["qas"] if q["question_id"] == query["id"])
        source_evidence = {
            normalize(e) for a in source_qa["answers"] for e in a["answer"]["evidence"]
        }
        if {normalize(tree.nodes[i].content) for i in gold} != source_evidence:
            raise ValueError("Gold paragraph mapping differs from original annotations")
        packet.append(
            {
                "case": case,
                "query_id": query["id"],
                "document": query["document"],
                "title": tree.root.title,
                "question": query["question"],
                "gold_sets": query["gold_sets"],
                "mapping_verified": True,
                "paragraphs": [
                    {
                        "id": i,
                        "path": tree.nodes[i].path[1:],
                        "text": tree.nodes[i].content,
                        "annotated": i in gold,
                        "retrieved_by_any": i in predictions,
                    }
                    for i in sorted(gold | predictions, key=order)
                ],
            }
        )
        key.append(
            {
                "case": case,
                "stratum": stratum,
                "query_id": query["id"],
                "document": query["document"],
                "results": {
                    n: {k: r[query["id"]][k] for k in ("recall", "predicted", "reason")}
                    for n, r in zip(names, indexed, strict=True)
                },
            }
        )
    protocol = {
        "salt": SALT,
        "split": "dev",
        "pool": {k: len(v) for k, v in strata.items()},
        "quotas": quotas,
        "selected_counts": dict(Counter(k["stratum"] for k in key)),
        "distinct_documents": len(used),
        "reviewer": "AI-assisted; no independent human adjudication",
        "source_sha256": file_hash(args.original_dev),
        "input_report_hashes": {n: file_hash(args.reports / (n + ".json")) for n in names},
        "packet_sha256": hashlib.sha256(json.dumps(packet, sort_keys=True).encode()).hexdigest(),
    }
    for name, value in (
        ("packet.json", packet),
        ("selection.json", key),
        ("protocol.json", protocol),
    ):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "cases": len(packet),
                "paragraphs": sum(len(p["paragraphs"]) for p in packet),
                "counts": protocol["selected_counts"],
                "packet_sha256": protocol["packet_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
