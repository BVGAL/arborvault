from dataclasses import asdict, dataclass, field
from typing import Protocol


@dataclass
class Node:
    id: str
    parent_id: str | None
    title: str
    content: str
    path: list[str]
    ordinal: int
    locator: dict = field(default_factory=dict)

    def as_dict(self):
        return asdict(self)


@dataclass
class Document:
    id: str
    source_key: str
    title: str
    source_uri: str | None
    revision: int
    digest: str


class Store(Protocol):
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
    ): ...
    def documents(self, tenant, namespace, limit=100, after=None): ...
    def revisions(self, tenant, namespace, document_id, limit=100, before=None): ...
    def nodes(self, tenant, namespace, document_id=None, revision=None, limit=10001, after=-1): ...
    def children(self, tenant, namespace, parents, query, limit): ...
    def passages(self, tenant, namespace, identities): ...
    def get_acl(self, tenant, namespace, document_id): ...
    def set_acl(self, tenant, namespace, document_id, acl): ...
    def snapshot(self, tenant, namespace): ...
    def lexical(self, tenant, namespace, query, k): ...
    def delete(self, tenant, namespace, document_id): ...
    def ping(self): ...
    def close(self): ...
