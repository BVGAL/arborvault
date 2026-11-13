"""Validate and export the frozen public-document evaluation set."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from .parsing import build_nodes, parse
from .storage import document_id


def within(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Dataset path escapes its directory")
    return path


def load_dataset(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    documents, indexed = {}, {}
    for doc in manifest["documents"]:
        identity = doc["id"]
        if identity in documents or not re.fullmatch(r"[a-z0-9-]+", identity):
            raise ValueError("Invalid or duplicate document ID")
        if not re.fullmatch(r"[0-9a-f]{40}", doc["commit"]):
            raise ValueError("Source commit must be immutable")
        raw = within(root, doc["path"]).read_bytes()
        license_bytes = within(root, doc["license_path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != doc["sha256"]:
            raise ValueError(f"Corpus hash mismatch: {identity}")
        if hashlib.sha256(license_bytes).hexdigest() != doc["license_sha256"]:
            raise ValueError(f"License hash mismatch: {identity}")
        sections = parse(raw, doc["path"])
        nodes = build_nodes(
            sections, document_id("dataset", "public-docs-v1", identity), 1, doc["title"]
        )
        indexed[identity] = nodes
        documents[identity] = {**doc, "raw": raw, "sections": sections}
    queries = [
        json.loads(line)
        for line in (root / "queries.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    seen, groups, missing = set(), {}, []
    for query in queries:
        if query["id"] in seen or not query["question"].strip():
            raise ValueError("Invalid or duplicate query")
        seen.add(query["id"])
        if query["split"] not in {"dev", "test"}:
            raise ValueError("Invalid split")
        group = query["group"]
        if group in groups and groups[group] != query["split"]:
            raise ValueError("Question family appears in multiple splits")
        groups[group] = query["split"]
        if bool(query["relevant"]) == (query["category"] == "unanswerable"):
            raise ValueError("Answerability label conflicts with qrels")
        for ref in query["relevant"]:
            matches = [
                n
                for n in indexed.get(ref["document"], [])
                if n.locator.get("line", 0) == ref["section_line"] and ref["evidence"] in n.content
            ]
            if not matches:
                missing.append(f"{query['id']} at {ref['document']}:{ref['section_line']}")
    if missing:
        raise ValueError("Evidence not found for " + "; ".join(missing))
    digest = hashlib.sha256(
        (root / "manifest.json").read_bytes() + (root / "queries.jsonl").read_bytes()
    ).hexdigest()
    return manifest, documents, queries, indexed, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--export", type=Path, help="Export section corpus, queries and qrels in BEIR-style files"
    )
    args = parser.parse_args()
    manifest, documents, queries, indexed, digest = load_dataset(args.directory)
    if args.export:
        args.export.mkdir(parents=True, exist_ok=True)
        corpus = [
            {
                "_id": n.id,
                "title": " / ".join(n.path),
                "text": n.content,
                "metadata": {"source_document": doc, "line": n.locator.get("line", 0)},
            }
            for doc, nodes in indexed.items()
            for n in nodes
            if n.content
        ]
        (args.export / "corpus.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in corpus), encoding="utf-8"
        )
        (args.export / "queries.jsonl").write_text(
            "".join(json.dumps({"_id": q["id"], "text": q["question"]}) + "\n" for q in queries),
            encoding="utf-8",
        )
        (args.export / "qrels").mkdir(exist_ok=True)
        for split in ("dev", "test"):
            lines = ["query-id\tcorpus-id\tscore\n"]
            for query in queries:
                if query["split"] == split:
                    relevant = {
                        n.id
                        for ref in query["relevant"]
                        for n in indexed[ref["document"]]
                        if n.locator.get("line", 0) == ref["section_line"]
                        and ref["evidence"] in n.content
                    }
                    lines.extend(f"{query['id']}\t{node_id}\t1\n" for node_id in sorted(relevant))
            (args.export / "qrels" / f"{split}.tsv").write_text("".join(lines), encoding="utf-8")
        (args.export / "labels.json").write_text(json.dumps(queries, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "dataset": manifest["id"],
                "documents": len(documents),
                "queries": len(queries),
                "dev": sum(q["split"] == "dev" for q in queries),
                "test": sum(q["split"] == "test" for q in queries),
                "sha256": digest,
            }
        )
    )


if __name__ == "__main__":
    main()
