"""Matched document-scoped evaluation; all methods share paragraph units and context budget."""

import hashlib
import json
import statistics
import time
from pathlib import Path

from .evaluate import TfidfIndex, metrics
from .qasper import load_split
from .routing_v2 import navigate


def context(nodes, k=5, words=512):
    result = []
    for node in nodes[:k]:
        if words <= 0:
            break
        tokens = node.content.split()
        excerpt = " ".join(tokens[:words])
        words -= min(words, len(tokens))
        if excerpt:
            result.append((node, excerpt))
    return result


def score(selected, query, k=5):
    predicted = [n.id for n, _ in selected]
    if not query["gold_sets"]:
        return {**metrics(predicted, [], k), "evidence_coverage": None}
    # Alternative human evidence annotations are valid independently, as in QASPER scoring.
    scores = []
    for gold in query["gold_sets"]:
        full = {n.id for n, text in selected if text == " ".join(n.content.split())}
        scores.append(
            {**metrics(predicted, gold, k), "evidence_coverage": len(full & set(gold)) / len(gold)}
        )
    return max(scores, key=lambda s: (s["recall"], s["mrr"], s["evidence_coverage"]))


def summarize(rows):
    answerable = [r for r in rows if r["answerable"]]
    unknown = [r for r in rows if not r["answerable"]]
    return {
        **{
            name: statistics.mean(r[name] for r in answerable)
            for name in ("recall", "mrr", "ndcg", "evidence_coverage")
        },
        "answerable_empty": sum(r["abstained"] for r in answerable),
        "unknown_abstentions": sum(r["abstained"] for r in unknown),
        "answerable_count": len(answerable),
        "unknown_count": len(unknown),
        "selector_errors": sum(r.get("reason") == "selector_error" for r in rows),
        "queries": rows,
    }


def baseline(dataset, split, cache_dir):
    import numpy as np
    from fastembed import TextEmbedding

    manifest, trees, queries = load_split(dataset, split)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = TextEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir=str(cache_dir / "models"),
        threads=4,
    )
    methods = {"dense_minilm": [], "lexical_tfidf": []}
    by_document = {}
    for query in queries:
        by_document.setdefault(query["document"], []).append(query)
    for document, group in by_document.items():
        passages = [n for n in trees[document].nodes.values() if n.content]
        texts = [" / ".join(n.path[1:]) + "\n" + n.content for n in passages]
        content_hash = hashlib.sha256(json.dumps(texts).encode()).hexdigest()
        cache_file = cache_dir / (content_hash + ".npy")
        if cache_file.exists():
            vectors = np.load(cache_file, allow_pickle=False)
        else:
            vectors = np.array(list(model.embed(texts)))
            np.save(cache_file, vectors, allow_pickle=False)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        lexical = TfidfIndex(texts)
        for query in group:
            q = np.array(next(iter(model.embed([query["question"]]))))
            q /= max(np.linalg.norm(q), 1e-12)
            rankings = {
                "dense_minilm": np.argsort(-(vectors @ q), kind="stable").tolist(),
                "lexical_tfidf": lexical.search(query["question"]),
            }
            for method, ranking in rankings.items():
                selected = context([passages[i] for i in ranking])
                methods[method].append(
                    {
                        "id": query["id"],
                        "document": document,
                        "answerable": query["answerable"],
                        "predicted": [n.id for n, _ in selected],
                        **score(selected, query),
                    }
                )
    hashes = {
        p.relative_to(cache_dir / "models").as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (cache_dir / "models").rglob("*.onnx")
    }
    return {
        "dataset": manifest["id"],
        "split": split,
        "model_sha256": hashes,
        "methods": {
            m: summarize(sorted(rows, key=lambda r: r["id"])) for m, rows in methods.items()
        },
    }


def evaluate(dataset, split, selector, *, progress=None):
    manifest, trees, queries = load_split(dataset, split)
    rows = {"navigation": [], "navigation_lexical": []}
    lexical = {}
    for identity, tree in trees.items():
        passages = [n for n in tree.nodes.values() if n.content]
        lexical[identity] = (
            passages,
            TfidfIndex([" / ".join(n.path[1:]) + "\n" + n.content for n in passages]),
        )
    for query in queries:
        tree = trees[query["document"]]
        start = time.perf_counter()
        before = len(selector.events)
        nodes, trace, reason = navigate(tree, query["question"], selector)
        events = selector.events[before:]
        alternatives = [tree.ancestors(gold) for gold in query["gold_sets"]]
        for step in trace:
            # Diagnostic only: annotations are inspected after inference, never passed to selector.
            step["gold_ancestor_candidates"] = [
                sorted(set(step["candidates"]) & group) for group in alternatives
            ]
            step["gold_ancestor_selected"] = [
                sorted(set(step["selected"]) & group) for group in alternatives
            ]
        passage_nodes, index = lexical[query["document"]]
        fallback = reason is not None or len(nodes) < 5
        auto = nodes[:]
        if fallback:
            already = {n.id for n in auto}
            auto.extend(
                passage_nodes[i]
                for i in index.search(query["question"])
                if passage_nodes[i].id not in already
            )
        for method, hits in (("navigation", nodes), ("navigation_lexical", auto)):
            selected = context(hits)
            row = {
                "id": query["id"],
                "document": query["document"],
                "answerable": query["answerable"],
                "gold_sets": query["gold_sets"],
                "predicted": [n.id for n, _ in selected],
                "reason": reason,
                "fallback": method == "navigation_lexical" and fallback,
                "reached_before_topk_and_word_budget": [n.id for n in nodes],
                "returned_words": sum(len(text.split()) for _, text in selected),
                "trace": trace if method == "navigation" else [],
                "model_events": events if method == "navigation" else [],
                "latency_ms": (time.perf_counter() - start) * 1000,
                **score(selected, query),
            }
            rows[method].append(row)
        if progress:
            progress({"query_id": query["id"], "calls": len(events), "reason": reason})
    return {
        "dataset": manifest["id"],
        "dataset_manifest_sha256": hashlib.sha256(
            (dataset / "manifest.json").read_bytes()
        ).hexdigest(),
        "split": split,
        "scope": "Known-document paragraph evidence retrieval",
        "k": 5,
        "word_budget": 512,
        "complete": all(
            r["reason"] not in {"selector_error", "candidate_limit", "depth_limit"}
            for r in rows["navigation"]
        ),
        "methods": {m: summarize(r) for m, r in rows.items()},
        "usage": {
            "calls": len(selector.events),
            "failures": sum(not e["ok"] for e in selector.events),
            "prompt_tokens": sum(e.get("prompt_tokens", 0) for e in selector.events),
            "output_tokens": sum(e.get("output_tokens", 0) for e in selector.events),
        },
    }
