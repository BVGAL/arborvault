"""Small synthetic retrieval benchmark; not evidence of general superiority."""

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from .parsing import build_nodes, parse
from .retrieval import TitleSelector, retrieve, tokens
from .storage import MemoryStore

CORPUS = {
    "operations": """# Operations
## Backups
Daily backups are retained for thirty days. Restore drills run monthly.
## Access renewal
Temporary access expires after twenty four hours. A project maintainer can renew it.
## Incident response
During an outage the incident coordinator assigns a severity and opens a status log.
""",
    "gardening": """# Gardening
## Seed storage
Store dry seeds in a cool dark cupboard. Label each packet with the harvest year.
## Irrigation
Water young plants early in the morning. Moist soil reduces transplant stress.
## Compost
Mix green plant scraps with dry leaves. Turn the compost weekly to add air.
""",
    "astronomy": """# Astronomy
## Lunar observation
The moon is easiest to observe along the terminator, where shadows reveal crater relief.
## Telescope care
Keep optical lenses covered when not in use. Avoid touching the glass surface.
## Meteor showers
Watch meteor showers from a dark location away from city lights. No telescope is needed.
""",
}
QUESTIONS = [
    ("How long are backups retained?", "operations", "Backups"),
    ("Who renews temporary access?", "operations", "Access renewal"),
    ("What does the incident coordinator do during an outage?", "operations", "Incident response"),
    ("Where should dry seeds be stored?", "gardening", "Seed storage"),
    ("When should young plants be watered?", "gardening", "Irrigation"),
    ("What materials should I mix for compost?", "gardening", "Compost"),
    ("Where are shadows helpful for observing moon craters?", "astronomy", "Lunar observation"),
    ("How do I protect telescope lenses?", "astronomy", "Telescope care"),
    ("Do I need a telescope to see meteor showers?", "astronomy", "Meteor showers"),
    ("How can I keep the optical glass clean and safe?", "astronomy", "Telescope care"),
    ("What is the price of a train ticket to Mars?", None, None),
    ("quasarxyz zzzunknown", None, None),
]


def metrics(predicted, relevant, k):
    # Deduplicate before scoring so repeated chunks cannot inflate relevance.
    predicted = list(dict.fromkeys(predicted))[:k]
    relevant = set(relevant)
    if not relevant:
        return {"recall": None, "mrr": None, "ndcg": None, "abstained": not predicted}
    flags = [p in relevant for p in predicted]
    dcg = sum(1 / math.log2(i + 2) for i, hit in enumerate(flags) if hit)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return {
        "recall": sum(flags) / len(relevant),
        "mrr": next((1 / (i + 1) for i, hit in enumerate(flags) if hit), 0),
        "ndcg": dcg / ideal,
        "abstained": not predicted,
    }


class TfidfIndex:
    def __init__(self, texts):
        self.docs = [Counter(tokens(t)) for t in texts]
        self.idf = {
            t: math.log((1 + len(texts)) / (1 + sum(t in d for d in self.docs))) + 1
            for d in self.docs
            for t in d
        }
        self.vectors = [self.vector(d) for d in self.docs]

    def vector(self, counts):
        values = {t: c * self.idf.get(t, 0) for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in values.values())) or 1
        return {t: v / norm for t, v in values.items()}

    def search(self, query):
        q = self.vector(Counter(tokens(query)))
        scores = [
            (i, sum(v * d.get(t, 0) for t, v in q.items())) for i, d in enumerate(self.vectors)
        ]
        return [i for i, s in sorted(scores, key=lambda x: (-x[1], x[0])) if s > 0]


class DenseIndex:
    def __init__(self, texts, model, cache_dir):
        import numpy as np
        from fastembed import TextEmbedding

        self.np = np
        self.model = TextEmbedding(model_name=model, cache_dir=cache_dir, threads=2)
        self.embeddings = np.array(list(self.model.embed(texts)))
        self.embeddings /= np.maximum(np.linalg.norm(self.embeddings, axis=1, keepdims=True), 1e-12)

    def search(self, query):
        np = self.np
        q = np.array(list(self.model.embed([query])))[0]
        q /= max(np.linalg.norm(q), 1e-12)
        # No tuned abstention threshold; all candidates are returned in score order.
        return np.argsort(-(self.embeddings @ q), kind="stable").tolist()


def run(dense=False, k=3, model="sentence-transformers/all-MiniLM-L6-v2", cache_dir=None):
    started = time.perf_counter()
    store = MemoryStore()
    for key, text in CORPUS.items():
        sections = parse(text.encode(), "fixture.md")
        store.upsert(
            "evaluation",
            "synthetic",
            key,
            key,
            f"urn:synthetic:{key}",
            hashlib.sha256(text.encode()).hexdigest(),
            lambda identity, revision, s=sections, t=key: build_nodes(s, identity, revision, t),
        )
    pairs = [(d, n) for d, n in store.nodes("evaluation", "synthetic") if n.content]
    structure_build_ms = (time.perf_counter() - started) * 1000
    gold = {(d.source_key, n.title): n.id for d, n in pairs}
    # Chunk full extracted document text without respecting section boundaries. Each
    # chunk maps to every gold section it overlaps, for section-level comparison.
    chunks, mappings = [], []
    for doc in store.documents("evaluation", "synthetic"):
        full, spans = "", []
        for d, n in pairs:
            if d.id == doc.id:
                begin = len(full)
                full += n.title + "\n" + n.content + "\n"
                spans.append((begin, len(full), n.id))
        for offset in range(0, len(full), 192):
            chunks.append(full[offset : offset + 256])
            mappings.append([i for a, b, i in spans if a < offset + 256 and b > offset])
    index_start = time.perf_counter()
    vector = DenseIndex(chunks, model, cache_dir) if dense else TfidfIndex(chunks)
    index_ms = (time.perf_counter() - index_start) * 1000
    methods = {"navigation": "navigation", "navigation_fallback": "auto", "lexical": "lexical"}
    vector_name = "dense_minilm_chunks" if dense else "tfidf_cosine_chunks"
    methods[vector_name] = None
    results = {}
    for name, mode in methods.items():
        rows = []
        for query, document, title in QUESTIONS:
            before = time.perf_counter()
            if mode:
                hits, _, fallback, _ = retrieve(
                    store, "evaluation", "synthetic", query, k, TitleSelector(), mode
                )
                predicted = [n.id for _, n, _, _ in hits]
                units = len(predicted)
            else:
                chunk_ids = vector.search(query)[:k]
                predicted = list(dict.fromkeys(i for index in chunk_ids for i in mappings[index]))
                units, fallback = len(chunk_ids), False
            elapsed = (time.perf_counter() - before) * 1000
            relevant = [gold[document, title]] if document else []
            # Vector k counts chunks; all overlapped sections count as retrieved. This
            # is intentionally favorable to chunks, reported with their context budget.
            score = metrics(predicted, relevant, max(k, len(predicted)))
            rows.append(
                {
                    "query": query,
                    "gold": relevant,
                    "predicted": predicted,
                    "latency_ms": elapsed,
                    "units": units,
                    "fallback": fallback,
                    **score,
                }
            )
        answerable = [r for r in rows if r["gold"]]
        unanswerable = [r for r in rows if not r["gold"]]
        results[name] = {
            "recall_at_k": statistics.mean(r["recall"] for r in answerable),
            "mrr": statistics.mean(r["mrr"] for r in answerable),
            "ndcg": statistics.mean(r["ndcg"] for r in answerable),
            "unanswerable_abstention": statistics.mean(r["abstained"] for r in unanswerable),
            "latency_p50_ms": statistics.median(r["latency_ms"] for r in rows),
            "latency_max_ms": max(r["latency_ms"] for r in rows),
            "queries": rows,
        }
    model_files = {}
    if dense and cache_dir:
        for path in sorted(Path(cache_dir).rglob("*")):
            if path.is_file() and path.suffix in {".onnx", ".json"}:
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                model_files[str(path.relative_to(cache_dir))] = digest.hexdigest()
    return {
        "warning": "Tiny synthetic smoke benchmark; not SearchTome or evidence of superiority.",
        "k": k,
        "queries": len(QUESTIONS),
        "sections": len(pairs),
        "chunks": len(chunks),
        "chunk_characters": 256,
        "chunk_overlap": 64,
        "backend": "MemoryStore",
        "corpus_sha256": hashlib.sha256(json.dumps(CORPUS, sort_keys=True).encode()).hexdigest(),
        "model": model if dense else None,
        "model_files_sha256": model_files,
        "python": platform.python_version(),
        "platform": platform.system(),
        "packages": {
            p: version(p)
            for p in (["arborvault", "fastembed", "numpy"] if dense else ["arborvault"])
        },
        "structure_build_ms": structure_build_ms,
        "vector_index_build_ms": index_ms,
        "methods": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--k", type=int, default=3, choices=range(1, 21))
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--cache-dir", default=".cache/models")
    parser.add_argument("--output", default="reports/smoke.json")
    args = parser.parse_args()
    report = run(args.dense, args.k, args.model, args.cache_dir)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                name: {k: v for k, v in row.items() if k != "queries"}
                for name, row in report["methods"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
