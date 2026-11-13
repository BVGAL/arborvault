"""Deterministic, document-disjoint QASPER evidence-routing pilot (CC BY 4.0 data)."""

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .domain import Node
from .retrieval import overlap
from .routing_v2 import PROMPT_SHA256, Tree, messages

DATASET = "qasper-routing-v2"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalize(text):
    return " ".join(text.split())


def paper_nodes(identity, paper):
    nodes, headings, paragraphs = [], {}, {}

    def add(parent, title, content, kind):
        node = Node(
            f"{identity}:{len(nodes)}",
            parent.id if parent else None,
            title,
            content,
            (parent.path if parent else []) + [title],
            len(nodes),
            {"kind": kind},
        )
        nodes.append(node)
        return node

    root = add(None, paper["title"], "", "document")
    for section in paper["full_text"]:
        names = [normalize(s) for s in (section["section_name"] or "").split(":::") if s.strip()]
        names = names or ["Untitled section"]
        parent = root
        for i, name in enumerate(names):
            key = tuple(names[: i + 1])
            if key not in headings:
                headings[key] = add(parent, name, "", "section")
            parent = headings[key]
        for paragraph in section["paragraphs"]:
            text = normalize(paragraph)
            if not text:
                continue
            title = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0][:110]
            node = add(parent, title, text, "paragraph")
            paragraphs.setdefault(text, []).append(node.id)
    if len(nodes) > 400 or not paragraphs or any(len(n.content) > 12000 for n in nodes):
        raise ValueError("Paper exceeds pilot structural bounds")
    return nodes, paragraphs


def question_record(identity, qa, paragraphs):
    answers = [entry["answer"] for entry in qa["answers"]]
    if not answers or len({a["unanswerable"] for a in answers}) != 1:
        raise ValueError("Missing or disagreeing answerability")
    alternatives = []
    if not answers[0]["unanswerable"]:
        for answer in answers:
            evidence = [normalize(e) for e in answer["evidence"]]
            if not evidence or any(
                e.startswith("FLOAT SELECTED") or e not in paragraphs for e in evidence
            ):
                raise ValueError("Non-textual or unmapped evidence")
            gold = sorted({i for e in evidence for i in paragraphs[e]})
            if len(gold) > 4:
                raise ValueError("More than four evidence paragraphs in an annotation")
            alternatives.append(gold)
        alternatives = sorted({tuple(g) for g in alternatives})
    return {
        "id": qa["question_id"],
        "document": identity,
        "question": normalize(qa["question"]),
        "language": "en",
        "gold_sets": [list(g) for g in alternatives],
        "answerable": bool(alternatives),
        "annotation": "QASPER human annotations",
    }


def build(source, output):
    output.mkdir(parents=True, exist_ok=False)
    seen_papers, seen_questions, seen_texts = set(), set(), set()
    manifest = {
        "id": DATASET,
        "version": 2,
        "license": "CC-BY-4.0",
        "upstream": "allenai/qasper v0.3",
        "scope": "Known-document evidence retrieval",
        "prompt_sha256": PROMPT_SHA256,
        "files": {},
        "splits": {},
        "source_sha256": {},
        "selection": "SHA256 order, fixed 500/100 answerable and 100/20 unknown quotas for train/evaluation; no model scores",
    }
    for split, positive, negative in (("train", 500, 100), ("dev", 100, 20), ("test", 100, 20)):
        source_path = source / f"qasper-{split}-v0.3.json"
        raw = source_path.read_bytes()
        manifest["source_sha256"][source_path.name] = digest(raw)
        original = json.loads(raw)
        pool, documents, rejected = [], {}, Counter()
        for identity, paper in original.items():
            try:
                nodes, paragraphs = paper_nodes(identity, paper)
            except ValueError as exc:
                rejected[str(exc)] += len(paper["qas"])
                continue
            content_digest = digest("\n".join(n.content for n in nodes).encode())
            if identity in seen_papers or content_digest in seen_texts:
                rejected["Document overlaps previous split"] += len(paper["qas"])
                continue
            documents[identity] = {
                "id": identity,
                "title": paper["title"],
                "source_url": "https://arxiv.org/abs/" + identity,
                "content_sha256": content_digest,
                "nodes": [asdict(n) for n in nodes],
            }
            for qa in paper["qas"]:
                try:
                    record = question_record(identity, qa, paragraphs)
                except ValueError as exc:
                    rejected[str(exc)] += 1
                    continue
                if normalize(record["question"]).casefold() in seen_questions:
                    rejected["Exact normalized question overlaps previous split"] += 1
                    continue
                pool.append(record)
        selected, current_questions = [], set()
        for answerable, quota in ((True, positive), (False, negative)):
            candidates = sorted(
                [q for q in pool if q["answerable"] == answerable],
                key=lambda q: digest(("arborvault-v2/" + q["id"]).encode()),
            )
            selected_class = []
            for q in candidates:
                canonical = q["question"].casefold()
                if canonical not in current_questions:
                    selected_class.append(q)
                    current_questions.add(canonical)
                if len(selected_class) == quota:
                    break
            if len(selected_class) != quota:
                raise ValueError(
                    f"Insufficient {split} examples: {answerable} {len(selected_class)}"
                )
            selected.extend(selected_class)
        selected.sort(key=lambda q: q["id"])
        doc_ids = {q["document"] for q in selected}
        selected_documents = [documents[i] for i in sorted(doc_ids)]
        for suffix, rows in (("documents", selected_documents), ("questions", selected)):
            filename = f"{split}-{suffix}.jsonl"
            payload = "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
            ).encode()
            (output / filename).write_bytes(payload)
            manifest["files"][filename] = digest(payload)
        manifest["splits"][split] = {
            "documents": len(doc_ids),
            "questions": len(selected),
            "answerable": positive,
            "unknown": negative,
            "eligible_pool": len(pool),
            "exclusions": dict(rejected),
        }
        seen_papers.update(doc_ids)
        seen_texts.update(documents[i]["content_sha256"] for i in doc_ids)
        seen_questions.update(current_questions)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


def load_split(directory, split):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    values = []
    for suffix in ("documents", "questions"):
        filename = f"{split}-{suffix}.jsonl"
        raw = (directory / filename).read_bytes()
        if digest(raw) != manifest["files"][filename]:
            raise ValueError("QASPER bundle checksum mismatch")
        values.append([json.loads(line) for line in raw.splitlines()])
    documents, queries = values
    trees = {d["id"]: Tree([Node(**n) for n in d["nodes"]]) for d in documents}
    for query in queries:
        tree = trees[query["document"]]
        if any(
            i not in tree.nodes or not tree.nodes[i].content for g in query["gold_sets"] for i in g
        ):
            raise ValueError("Unmapped evidence in exported dataset")
    return manifest, trees, queries


def training_routes(directory):
    _, trees, queries = load_split(directory, "train")
    rows = []
    for query in queries:
        tree = trees[query["document"]]
        gold = set(query["gold_sets"][0]) if query["gold_sets"] else set()
        ancestors = tree.ancestors(gold)
        frontier = tree.children[tree.root.id]
        depth = 0
        while frontier:
            identity = query["id"] + "/" + str(depth)
            frontier = list(frontier)
            random.Random(identity).shuffle(frontier)
            views = tree.views(frontier)
            conversation, labels = messages(query["question"], views, 4)
            target = [label for label, node in labels.items() if node in ancestors]
            if len(target) > 4:
                raise ValueError("Oracle exceeds beam")
            rows.append(
                {
                    "id": identity,
                    "query_id": query["id"],
                    "document": query["document"],
                    "depth": depth,
                    "kind": "oracle_route",
                    "query": query["question"],
                    "candidate_ids": [n.id for n in frontier],
                    "target_ids": target,
                    "messages": conversation
                    + [{"role": "assistant", "content": json.dumps(target, separators=(",", ":"))}],
                }
            )
            picked = [labels[label] for label in target]
            frontier = [child for i in picked for child in tree.children[i]]
            depth += 1
        # One difficult wrong-branch example on a fixed 1/4 of answerable training questions.
        if gold and int(digest(query["id"].encode())[:8], 16) % 4 == 0:
            all_ancestors = tree.ancestors({i for group in query["gold_sets"] for i in group})
            wrong = [
                n for n in tree.nodes.values() if n.id not in all_ancestors and tree.children[n.id]
            ]
            wrong.sort(key=lambda n: (-overlap(query["question"], " ".join(n.path)), n.ordinal))
            if wrong:
                candidates = tree.children[wrong[0].id]
                conversation, _ = messages(query["question"], tree.views(candidates), 4)
                rows.append(
                    {
                        "id": query["id"] + "/negative",
                        "query_id": query["id"],
                        "document": query["document"],
                        "depth": len(wrong[0].path) - 1,
                        "kind": "wrong_subtree",
                        "query": query["question"],
                        "candidate_ids": [n.id for n in candidates],
                        "target_ids": [],
                        "messages": conversation + [{"role": "assistant", "content": "[]"}],
                    }
                )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = build(args.source, args.output)
    print(json.dumps(result["splits"], indent=2))


if __name__ == "__main__":
    main()
