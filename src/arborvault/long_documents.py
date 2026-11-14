"""Research HTML corpus: preserve sections and API definitions, with bounded leaves."""

import hashlib
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag


def parse_manual(html, identity, url, max_words=160):
    """Normalize readable body blocks once; retain source anchors and hierarchy.

    Navigation, scripts and heading permalinks are excluded. Long blocks are split
    consecutively, without overlap. API definitions form branches below sections.
    This corpus parser is deliberately separate from the production ingest API.
    """
    soup = BeautifulSoup(html, "html.parser")
    body = soup.select_one('div.body[role="main"]') or soup.select_one("main")
    if body is None:
        raise ValueError("No main document body")
    for unwanted in body.select("script, style, a.headerlink, nav"):
        unwanted.decompose()
    title = body.find("h1").get_text(" ", strip=True)
    nodes = []

    def add(parent, heading, content, anchor, kind):
        node_id = f"{identity}-{len(nodes):05d}"
        path = (parent["path"] if parent else []) + [heading]
        item = {
            "id": node_id,
            "parent_id": parent["id"] if parent else None,
            "title": heading,
            "content": content,
            "path": path,
            "ordinal": len(nodes),
            "locator": {"url": url, "anchor": anchor, "kind": kind},
        }
        nodes.append(item)
        return item

    root = add(None, title, "", "", "document")

    def leaf(element, parent, anchor):
        text = " ".join(element.get_text(" ", strip=True).split())
        words = text.split()
        for offset in range(0, len(words), max_words):
            excerpt = " ".join(words[offset : offset + max_words])
            add(
                parent,
                re.split(r"(?<=[.!?])\s", excerpt, maxsplit=1)[0][:110],
                excerpt,
                element.get("id", anchor),
                element.name,
            )

    def walk(element, parent, anchor=""):
        if not isinstance(element, Tag):
            return
        anchor = element.get("id", anchor)
        if element.name == "section":
            heading = element.find(["h1", "h2", "h3", "h4", "h5", "h6"], recursive=False)
            if heading is not None and heading.name != "h1":
                parent = add(parent, heading.get_text(" ", strip=True), "", anchor, "section")
        if element.name == "dl" and "py" in element.get("class", []):
            signatures = element.find_all("dt", recursive=False)
            name = " | ".join(t.get_text(" ", strip=True) for t in signatures)
            parent = add(
                parent, name, "", signatures[0].get("id", anchor) if signatures else anchor, "api"
            )
        if element.name in {"p", "pre", "tr", "dt"}:
            leaf(element, parent, anchor)
            return
        if element.name in {"li", "dd"} and not element.find(
            ["p", "pre", "dl", "ul", "ol", "table"]
        ):
            leaf(element, parent, anchor)
            return
        if element.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            return
        for child in element.children:
            walk(child, parent, anchor)

    walk(body, root)
    parents = {n["parent_id"] for n in nodes}
    # Empty headings cannot masquerade as passages. Keep ancestors of real text.
    keep = {n["id"] for n in nodes if n["content"]}
    by_id = {n["id"]: n for n in nodes}
    for identity in list(keep):
        current = by_id[identity]
        while current["parent_id"]:
            keep.add(current["parent_id"])
            current = by_id[current["parent_id"]]
    nodes = [n for n in nodes if n["id"] in keep]
    return {
        "id": root["id"].rsplit("-", 1)[0],
        "title": title,
        "url": url,
        "nodes": nodes,
        "stats": {
            "nodes": len(nodes),
            "passages": sum(bool(n["content"]) for n in nodes),
            "words": sum(len(n["content"].split()) for n in nodes),
            "max_depth": max(len(n["path"]) for n in nodes),
            "branches": sum(n["id"] in parents for n in nodes),
        },
    }


def load_corpus(directory):
    from .domain import Node
    from .routing_v2 import Tree

    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        if hashlib.sha256((directory / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Corpus hash mismatch: {relative}")
    docs = json.loads((directory / "documents.json").read_text())
    queries = json.loads((directory / "questions.json").read_text())
    trees = {d["id"]: Tree([Node(**n) for n in d["nodes"]]) for d in docs}
    roles = {q["document"]: q["split"] for q in queries}
    if any(roles[q["document"]] != q["split"] for q in queries):
        raise ValueError("Document leakage")
    for q in queries:
        tree = trees[q["document"]]
        if bool(q["gold_sets"]) != q["answerable"]:
            raise ValueError("Invalid answerability")
        if any(tree.children[i] or not tree.nodes[i].content for g in q["gold_sets"] for i in g):
            raise ValueError("Evidence must refer to passages")
    return manifest, trees, queries
