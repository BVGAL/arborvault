"""Test-only implementation of the same revision and pagination semantics."""

import copy
import threading
from contextlib import contextmanager
from uuid import uuid4

from .domain import Document
from .retrieval import overlap


class MemoryStore:
    def __init__(self):
        self.data = {}
        self.lock = threading.RLock()

    def upsert(self, tenant, namespace, source_key, title, source_uri, digest, builder):
        with self.lock:
            identity = next(
                (
                    i
                    for (t, n, i), h in self.data.items()
                    if (t, n) == (tenant, namespace) and h[-1][0].source_key == source_key
                ),
                str(uuid4()),
            )
            key = tenant, namespace, identity
            history = self.data.get(key, [])
            if history and history[-1][0].digest == digest:
                return copy.deepcopy(history[-1][0]), False
            revision = history[-1][0].revision + 1 if history else 1
            doc = Document(identity, source_key, title, source_uri, revision, digest)
            nodes = builder(identity, revision)
            self.data.setdefault(key, []).append((doc, nodes))
            return copy.deepcopy(doc), True

    def documents(self, tenant, namespace, limit=100, after=None):
        with self.lock:
            rows = [
                history[-1][0]
                for (t, n, i), history in sorted(self.data.items())
                if (t, n) == (tenant, namespace) and (after is None or i > after)
            ]
            return copy.deepcopy(rows[:limit])

    def revisions(self, tenant, namespace, document_id, limit=100, before=None):
        with self.lock:
            history = self.data.get((tenant, namespace, document_id), [])
            return copy.deepcopy(
                [d for d, _ in reversed(history) if before is None or d.revision < before][:limit]
            )

    def nodes(self, tenant, namespace, document_id=None, revision=None):
        with self.lock:
            result = []
            for (t, n, i), history in sorted(self.data.items()):
                if (t, n) != (tenant, namespace) or (document_id is not None and i != document_id):
                    continue
                matches = (
                    [row for row in history if row[0].revision == revision]
                    if revision
                    else history[-1:]
                )
                for doc, nodes in matches:
                    result.extend((doc, node) for node in nodes)
            return copy.deepcopy(result[:10001])

    def lexical(self, tenant, namespace, query, k):
        scored = [
            (d, n, overlap(query, n.title + " " + n.content))
            for d, n in self.nodes(tenant, namespace)
            if n.content
        ]
        return sorted(
            [row for row in scored if row[2] > 0],
            key=lambda row: (-row[2], row[0].source_key, row[1].ordinal),
        )[:k]

    def delete(self, tenant, namespace, document_id):
        with self.lock:
            return self.data.pop((tenant, namespace, document_id), None) is not None

    @contextmanager
    def snapshot(self, tenant, namespace):
        view = MemoryStore()
        with self.lock:
            view.data = copy.deepcopy(
                {key: value for key, value in self.data.items() if key[:2] == (tenant, namespace)}
            )
        yield view

    def ping(self):
        return True

    def close(self):
        pass
