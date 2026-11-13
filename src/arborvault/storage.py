import copy
import threading
from importlib.resources import files
from uuid import NAMESPACE_URL, uuid5

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from .domain import Document, Node
from .retrieval import overlap, tokens


def document_id(tenant, namespace, source_key):
    import json

    return str(uuid5(NAMESPACE_URL, json.dumps([tenant, namespace, source_key])))


class MemoryStore:
    """Explicit test/demo backend, never selected implicitly."""

    def __init__(self):
        self.data = {}
        self.lock = threading.RLock()

    def upsert(self, tenant, namespace, source_key, title, source_uri, digest, builder):
        with self.lock:
            identity = document_id(tenant, namespace, source_key)
            key = tenant, namespace, identity
            old = self.data.get(key)
            if old and old[0].digest == digest:
                return copy.deepcopy(old[0]), False
            revision = old[0].revision + 1 if old else 1
            doc = Document(identity, source_key, title, source_uri, revision, digest)
            nodes = builder(identity, revision)
            self.data[key] = doc, nodes
            return copy.deepcopy(doc), True

    def documents(self, tenant, namespace):
        with self.lock:
            return copy.deepcopy(
                [d for (t, n, _), (d, _) in self.data.items() if (t, n) == (tenant, namespace)]
            )

    def nodes(self, tenant, namespace, document_id=None):
        with self.lock:
            return copy.deepcopy(
                [
                    (d, node)
                    for (t, n, i), (d, nodes) in self.data.items()
                    if (t, n) == (tenant, namespace) and (document_id is None or i == document_id)
                    for node in nodes
                ]
            )

    def lexical(self, tenant, namespace, query, k):
        scored = [
            (d, n, overlap(query, n.title + " " + n.content))
            for d, n in self.nodes(tenant, namespace)
            if n.content
        ]
        return sorted([row for row in scored if row[2] > 0], key=lambda row: (-row[2], row[1].id))[
            :k
        ]

    def delete(self, tenant, namespace, document_id):
        with self.lock:
            return self.data.pop((tenant, namespace, document_id), None) is not None

    def ping(self):
        return True

    def close(self):
        pass


class PostgresStore:
    def __init__(self, url):
        self.pool = ConnectionPool(
            url, min_size=1, max_size=8, kwargs={"row_factory": dict_row}, timeout=10, open=True
        )
        self.pool.wait(timeout=15)

    def migrate(self):
        with self.pool.connection() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(738152931)")
            conn.execute(files("arborvault").joinpath("schema.sql").read_text())

    def upsert(self, tenant, namespace, source_key, title, source_uri, digest, builder):
        identity = document_id(tenant, namespace, source_key)
        scope = (tenant, namespace, identity)
        with self.pool.connection() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (identity,))
            old = conn.execute(
                "SELECT * FROM documents WHERE tenant=%s AND namespace=%s AND id=%s", scope
            ).fetchone()
            if old and old["digest"] == digest:
                return self._doc(old), False
            revision = old["revision"] + 1 if old else 1
            nodes = builder(identity, revision)
            conn.execute(
                "INSERT INTO documents VALUES (%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (tenant,namespace,id) DO UPDATE SET title=EXCLUDED.title, "
                "source_uri=EXCLUDED.source_uri,revision=EXCLUDED.revision,digest=EXCLUDED.digest",
                (*scope, source_key, title, source_uri, revision, digest),
            )
            conn.execute(
                "DELETE FROM nodes WHERE tenant=%s AND namespace=%s AND document_id=%s", scope
            )
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO nodes (tenant,namespace,document_id,id,parent_id,title,"
                    "content,path,ordinal,locator) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    [
                        (
                            *scope,
                            n.id,
                            n.parent_id,
                            n.title,
                            n.content,
                            Jsonb(n.path),
                            n.ordinal,
                            Jsonb(n.locator),
                        )
                        for n in nodes
                    ],
                )
        return Document(identity, source_key, title, source_uri, revision, digest), True

    @staticmethod
    def _doc(row):
        return Document(
            str(row["id"]),
            row["source_key"],
            row["title"],
            row["source_uri"],
            row["revision"],
            row["digest"],
        )

    @staticmethod
    def _pair(row):
        doc = Document(**row["doc"])
        doc.id = str(doc.id)
        node = Node(
            str(row["id"]),
            str(row["parent_id"]) if row["parent_id"] else None,
            row["title"],
            row["content"],
            row["path"],
            row["ordinal"],
            row["locator"],
        )
        return doc, node

    def documents(self, tenant, namespace):
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE tenant=%s AND namespace=%s ORDER BY id LIMIT 1000",
                (tenant, namespace),
            ).fetchall()
        return [self._doc(row) for row in rows]

    _join = """SELECT n.*, jsonb_build_object('id',d.id,'source_key',d.source_key,
        'title',d.title,'source_uri',d.source_uri,'revision',d.revision,'digest',d.digest) AS doc
        FROM nodes n JOIN documents d ON (n.tenant,n.namespace,n.document_id) =
        (d.tenant,d.namespace,d.id) WHERE n.tenant=%s AND n.namespace=%s """

    def nodes(self, tenant, namespace, document_id=None):
        query = self._join
        params = [tenant, namespace]
        if document_id:
            query += "AND n.document_id=%s "
            params.append(document_id)
        with self.pool.connection() as conn:
            rows = conn.execute(
                query + "ORDER BY n.document_id,n.ordinal LIMIT 10001", params
            ).fetchall()
        return [self._pair(r) for r in rows]

    def lexical(self, tenant, namespace, query, k):
        # Build OR terms from alphanumeric tokens, then bind them as a value.
        terms = " OR ".join(tokens(query))
        if not terms:
            return []
        sql = self._join + "AND n.content<>'' AND n.search @@ websearch_to_tsquery('simple',%s) "
        sql += "ORDER BY ts_rank_cd(n.search,websearch_to_tsquery('simple',%s)) DESC,n.id LIMIT %s"
        with self.pool.connection() as conn:
            rows = conn.execute(sql, (tenant, namespace, terms, terms, k)).fetchall()
        return [(*self._pair(r), overlap(query, r["title"] + " " + r["content"])) for r in rows]

    def delete(self, tenant, namespace, document_id):
        with self.pool.connection() as conn:
            return (
                conn.execute(
                    "DELETE FROM documents WHERE tenant=%s AND namespace=%s AND id=%s",
                    (tenant, namespace, document_id),
                ).rowcount
                > 0
            )

    def ping(self):
        with self.pool.connection() as conn:
            conn.execute("SELECT 1")
        return True

    def close(self):
        self.pool.close()


def main():
    import os

    store = PostgresStore(os.environ["DATABASE_URL"])
    try:
        store.migrate()
        print("Schema v1 ready")
    finally:
        store.close()


if __name__ == "__main__":
    main()
