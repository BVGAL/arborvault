"""Export reproducible routing examples without using public test questions as training labels."""

import argparse
import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path

from .dataset import load_dataset, within
from .domain import Node
from .ollama import PROMPT_SHA256, PROMPT_VERSION, routing_messages
from .retrieval import routing_summaries


def example(query, candidates, wanted, identity, group, provenance):
    candidates = list(candidates)
    random.Random(identity).shuffle(candidates)
    messages, labels = routing_messages(query, candidates, 4)
    selected = [label for label, node_id in labels.items() if node_id in wanted]
    if len(selected) > 4:
        raise ValueError("Oracle route exceeds beam width")
    return {
        "id": identity,
        "group": group,
        "provenance": provenance,
        "messages": messages + [{"role": "assistant", "content": json.dumps(selected)}],
    }


def public_examples(directory):
    _, _, queries, indexed, digest = load_dataset(directory)
    nodes = [node for document in indexed.values() for node in document]
    by_id = {node.id: node for node in nodes}
    summaries = routing_summaries(nodes)
    result = {"train": [], "validation": []}
    query_ids = {"train": [], "validation": [], "held_out_test": []}
    for query in queries:
        if query["split"] == "test":
            query_ids["held_out_test"].append(query["id"])
            continue
        split = (
            "train"
            if query["group"] in {"httpx-limits", "unanswerable-01", "unanswerable-02"}
            else "validation"
        )
        query_ids[split].append(query["id"])
        relevant = {
            node.id
            for ref in query["relevant"]
            for node in indexed[ref["document"]]
            if node.locator.get("line", 0) == ref["section_line"]
            and ref["evidence"] in node.content
        }
        ancestors = set(relevant)
        for identity in relevant:
            parent = by_id[identity].parent_id
            while parent:
                ancestors.add(parent)
                parent = by_id[parent].parent_id
        frontier = [node for node in nodes if node.parent_id is None]
        depth = 0
        while frontier:
            wanted = {node.id for node in frontier} & ancestors
            result[split].append(
                example(
                    query["question"],
                    [replace(n, content=summaries[n.id]) for n in frontier],
                    wanted,
                    query["id"] + "/" + str(depth),
                    query["group"],
                    "public-docs-v1/dev",
                )
            )
            frontier = [node for node in nodes if node.parent_id in wanted]
            depth += 1
    return result, query_ids, digest


def synthetic_examples():
    """Procedural format-training fixtures, not evidence of general retrieval quality."""
    topics = [
        (
            "Retention",
            "Backup copies expire after {n} days.",
            "How long are backups kept?",
            "Quanto durano i backup?",
        ),
        (
            "Access",
            "Access tokens expire after {n} hours.",
            "When does an access token expire?",
            "Quando scade un token di accesso?",
        ),
        (
            "Capacity",
            "The connection pool permits {n} concurrent connections.",
            "What is the connection pool limit?",
            "Qual è il limite di connessioni?",
        ),
        (
            "Recovery",
            "Recovery jobs retry at most {n} times.",
            "How many times is recovery retried?",
            "Quanti tentativi di ripristino sono previsti?",
        ),
        (
            "Timeouts",
            "A request times out after {n} seconds.",
            "How long can a request wait?",
            "Qual è il tempo massimo di una richiesta?",
        ),
        (
            "Exports",
            "Exports are available for {n} days.",
            "How long can an export be downloaded?",
            "Per quanti giorni si può scaricare un export?",
        ),
    ]
    result = {"train": [], "validation": [], "synthetic_test": []}
    for split, count in (("train", 384), ("validation", 64), ("synthetic_test", 64)):
        for i in range(count):
            identity = f"synthetic/{split}/{i:04d}"
            rng = random.Random(identity)
            target = rng.randrange(len(topics))
            entity = hashlib.sha256(identity.encode()).hexdigest()[:10]
            candidates = [
                Node(
                    str(j), None, title, sentence.format(n=rng.randrange(5, 90)), [entity, title], j
                )
                for j, (title, sentence, _, _) in enumerate(topics)
            ]
            query = topics[target][2 + i % 2] + " " + entity
            wanted = {str(target)}
            if i % 8 == 0:
                query, wanted = "What is the orbital speed of " + entity + "?", set()
            elif i % 8 == 1:
                second = (target + 1) % len(topics)
                query += " " + topics[second][2 + i % 2]
                wanted.add(str(second))
            result[split].append(
                example(query, candidates, wanted, identity, identity, "procedural-original-v1")
            )
    return result


def export(directory, output):
    public, ids, dataset_hash = public_examples(directory)
    rows = synthetic_examples()
    for split in public:
        rows[split].extend(public[split])
    groups = [{r["group"] for r in examples} for examples in rows.values()]
    if any(a & b for i, a in enumerate(groups) for b in groups[i + 1 :]):
        raise ValueError("Question family leakage across splits")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sources = json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
    (output / "licenses").mkdir(exist_ok=True)
    for doc in sources["documents"]:
        (output / "licenses" / Path(doc["license_path"]).name).write_bytes(
            within(Path(directory), doc["license_path"]).read_bytes()
        )
    hashes = {}
    for split, examples in rows.items():
        raw = "".join(
            json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in examples
        ).encode()
        (output / (split + ".jsonl")).write_bytes(raw)
        hashes[split] = hashlib.sha256(raw).hexdigest()
    manifest = {
        "id": "routing-bootstrap-v1",
        "annotation_status": "procedural_and_assistant_draft",
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": PROMPT_SHA256,
        "public_dataset_sha256": dataset_hash,
        "source_documents": sources["documents"],
        "public_query_ids": ids,
        "counts": {k: len(v) for k, v in rows.items()},
        "sha256": hashes,
        "limitations": [
            "Procedural templates repeat across synthetic splits; use public test for external evaluation.",
            "Only 16 public development questions, split by family into 8 training and 8 validation questions.",
            "All corpus documents can appear as routing candidates; test questions and test relevance labels are never exported for training.",
            "Bootstrap data requires independent review and expansion before quality claims.",
        ],
        "licenses": [
            "Apache-2.0 for original procedural examples",
            "MIT and BSD-3-Clause for public corpus excerpts; retain source notices",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_bundle(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["prompt_sha256"] != PROMPT_SHA256:
        raise ValueError("Dataset uses a different routing prompt")
    rows, groups, questions = {}, set(), set()
    held_out = set(manifest["public_query_ids"]["held_out_test"])
    for split in ("train", "validation", "synthetic_test"):
        raw = (directory / (split + ".jsonl")).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest["sha256"][split]:
            raise ValueError("Dataset checksum mismatch: " + split)
        rows[split] = [json.loads(line) for line in raw.splitlines()]
        if len(rows[split]) != manifest["counts"][split]:
            raise ValueError("Dataset count mismatch")
        current_groups = {r["group"] for r in rows[split]}
        current_questions = set()
        for row in rows[split]:
            if row["id"].split("/")[0] in held_out:
                raise ValueError("Public test question leaked into training bundle")
            messages = row["messages"]
            if [m["role"] for m in messages] != ["system", "user", "assistant"]:
                raise ValueError("Invalid conversation roles")
            payload = json.loads(messages[1]["content"])
            from .ollama import SYSTEM, decode_selection

            if messages[0]["content"] != SYSTEM.format(width=4):
                raise ValueError("Unexpected system prompt")
            decode_selection(
                messages[2]["content"], {c["id"]: c["id"] for c in payload["candidates"]}, 4
            )
            current_questions.add(" ".join(payload["query"].casefold().split()))
        if groups & current_groups or questions & current_questions:
            raise ValueError("Training and evaluation splits overlap")
        groups.update(current_groups)
        questions.update(current_questions)
    return manifest, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/routing-data"))
    args = parser.parse_args()
    manifest = export(args.directory, args.output)
    print(
        json.dumps(
            {"dataset": manifest["id"], "counts": manifest["counts"], "sha256": manifest["sha256"]}
        )
    )


if __name__ == "__main__":
    main()
