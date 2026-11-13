"""Test-only implementation of the same revision and pagination semantics."""

import copy
import threading
from contextlib import contextmanager
from dataclasses import replace
from uuid import uuid4

from .access import allowed, initial_acl
from .domain import Document
from .retrieval import overlap, routing_summaries


class MemoryStore:
    def __init__(self):
        self.data = {}
        self.acls = {}
        self.routing = {}
        self.lock = threading.RLock()

    def upsert(
        self,
        tenant,
        namespace,
        source_key,
        title,
        source_uri,
        digest,
        builder,
        visibility="namespace",
    ):
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
            if history and not allowed(self.acls[key], "write"):
                raise PermissionError("Document unavailable")
            if history and history[-1][0].digest == digest:
                return copy.deepcopy(history[-1][0]), False
            revision = history[-1][0].revision + 1 if history else 1
            doc = Document(identity, source_key, title, source_uri, revision, digest)
            nodes = builder(identity, revision)
            self.routing[key] = routing_summaries(nodes)
            self.acls.setdefault(key, initial_acl(visibility))
            self.data.setdefault(key, []).append((doc, nodes))
            return copy.deepcopy(doc), True

    def documents(self, tenant, namespace, limit=100, after=None):
        with self.lock:
            rows = [
                history[-1][0]
                for (t, n, i), history in sorted(self.data.items())
                if (t, n) == (tenant, namespace)
                and (after is None or i > after)
                and allowed(self.acls[t, n, i])
            ]
            return copy.deepcopy(rows[:limit])

    def revisions(self, tenant, namespace, document_id, limit=100, before=None):
        with self.lock:
            acl = self.acls.get((tenant, namespace, document_id))
            if acl is None or not allowed(acl):
                return []
            history = self.data.get((tenant, namespace, document_id), [])
            return copy.deepcopy(
                [d for d, _ in reversed(history) if before is None or d.revision < before][:limit]
            )

    def nodes(self, tenant, namespace, document_id=None, revision=None, limit=10001, after=-1):
        with self.lock:
            result = []
            for (t, n, i), history in sorted(self.data.items()):
                if (t, n) != (tenant, namespace) or (document_id is not None and i != document_id):
                    continue
                if not allowed(self.acls[t, n, i]):
                    continue
                matches = (
                    [row for row in history if row[0].revision == revision]
                    if revision
                    else history[-1:]
                )
                for doc, nodes in matches:
                    result.extend((doc, node) for node in nodes if node.ordinal > after)
            return copy.deepcopy(result[:limit])

    def children(self, tenant, namespace, parents, query, limit):
        candidates = []
        with self.lock:
            for key, history in self.data.items():
                if key[:2] != (tenant, namespace) or not allowed(self.acls[key]):
                    continue
                doc, nodes = history[-1]
                for node in nodes:
                    if (parents is None and node.parent_id is None) or (
                        parents is not None and node.parent_id in parents
                    ):
                        candidates.append((doc, replace(node, content=self.routing[key][node.id])))
        candidates.sort(
            key=lambda p: (-overlap(query, p[1].content), p[0].source_key, p[1].ordinal)
        )
        return copy.deepcopy(candidates[:limit])

    def passages(self, tenant, namespace, identities):
        with self.lock:
            return copy.deepcopy(
                [
                    (doc, node)
                    for key, history in self.data.items()
                    if key[:2] == (tenant, namespace) and allowed(self.acls[key])
                    for doc, nodes in history[-1:]
                    for node in nodes
                    if node.id in identities
                ]
            )

    def lexical(self, tenant, namespace, query, k):
        scored = [
            (d, n, overlap(query, n.title + " " + n.content))
            for d, n in self.nodes(tenant, namespace, limit=1_000_000)
            if n.content
        ]
        return sorted(
            [row for row in scored if row[2] > 0],
            key=lambda row: (-row[2], row[0].source_key, row[1].ordinal),
        )[:k]

    def delete(self, tenant, namespace, document_id):
        with self.lock:
            key = tenant, namespace, document_id
            if key not in self.acls or not allowed(self.acls[key], "delete"):
                return False
            self.acls.pop(key)
            self.routing.pop(key)
            return self.data.pop(key, None) is not None

    def get_acl(self, tenant, namespace, document_id):
        with self.lock:
            acl = self.acls.get((tenant, namespace, document_id))
            return copy.deepcopy(acl) if acl and allowed(acl, "manage") else None

    def set_acl(self, tenant, namespace, document_id, acl):
        with self.lock:
            old = self.acls.get((tenant, namespace, document_id))
            if old is None or not allowed(old, "manage"):
                return False
            old.update(copy.deepcopy(acl))
            return True

    @contextmanager
    def snapshot(self, tenant, namespace):
        view = MemoryStore()
        with self.lock:
            view.data = copy.deepcopy(
                {
                    key: value
                    for key, value in self.data.items()
                    if key[:2] == (tenant, namespace) and allowed(self.acls[key])
                }
            )
            view.acls = copy.deepcopy({key: self.acls[key] for key in view.data})
            view.routing = copy.deepcopy({key: self.routing[key] for key in view.data})
        yield view

    def ping(self):
        return True

    def close(self):
        pass
