from contextlib import contextmanager
from dataclasses import replace
from uuid import NAMESPACE_URL, uuid4, uuid5

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from .access import CURRENT_PRINCIPAL, initial_acl
from .domain import Document, Node
from .memory import MemoryStore as MemoryStore
from .migrate import SCHEMA_VERSION, migrate
from .retrieval import routing_summaries, tokens


def document_id(tenant, namespace, source_key):
    import json

    return str(uuid5(NAMESPACE_URL, json.dumps([tenant, namespace, source_key])))


class PostgresStore:
    def __init__(self, url, enforce_runtime_role=False):
        self.url = url
        self.pool = ConnectionPool(
            url, min_size=1, max_size=8, timeout=10, open=True, kwargs={"row_factory": dict_row}
        )
        self.pool.wait(timeout=15)
        if enforce_runtime_role:
            try:
                with self.pool.connection() as conn:
                    role = conn.execute(
                        "SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user"
                    ).fetchone()
                    owns = conn.execute(
                        "SELECT 1 FROM pg_class WHERE relname IN "
                        "('documents','nodes','document_revisions','document_access') "
                        "AND relowner=(SELECT oid FROM pg_roles WHERE rolname=current_user)"
                    ).fetchone()
                    if role["rolsuper"] or role["rolbypassrls"] or owns:
                        raise RuntimeError(
                            "API requires a non-owner, non-superuser, NOBYPASSRLS role"
                        )
                self.ping()
            except Exception:
                self.close()
                raise

    def migrate(self):
        return migrate(self.url)

    @contextmanager
    def scoped(self, tenant, namespace, snapshot=False, operation="read"):
        with self.pool.connection() as conn:
            if snapshot:
                conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            conn.execute(
                "SELECT set_config('arborvault.tenant',%s,true), "
                "set_config('arborvault.namespace',%s,true), "
                "set_config('statement_timeout','30000',true), "
                "set_config('arborvault.principal',%s,true), "
                "set_config('arborvault.operation',%s,true), "
                "set_config('arborvault.manage_acl',%s,true)",
                (
                    tenant,
                    namespace,
                    CURRENT_PRINCIPAL.get().subject,
                    operation,
                    str(CURRENT_PRINCIPAL.get().manage_acl).lower(),
                ),
            )
            yield conn

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
        lock_key = document_id(tenant, namespace, source_key)
        with self.scoped(tenant, namespace, operation="write") as conn:
            conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (lock_key,))
            old = conn.execute(
                "SELECT * FROM documents WHERE tenant=%s AND namespace=%s AND source_key=%s "
                "AND arborvault_document_allowed(tenant,namespace,id)",
                (tenant, namespace, source_key),
            ).fetchone()
            if old and old["digest"] == digest:
                return self._doc(old), False
            revision = old["revision"] + 1 if old else 1
            identity = str(old["id"]) if old else str(uuid4())
            scope = tenant, namespace, identity
            nodes = builder(identity, revision)
            routing = routing_summaries(nodes)
            if old:
                conn.execute(
                    "UPDATE documents SET title=%s,source_uri=%s,revision=%s,digest=%s "
                    "WHERE tenant=%s AND namespace=%s AND id=%s "
                    "AND arborvault_document_allowed(tenant,namespace,id)",
                    (title, source_uri, revision, digest, *scope),
                )
            else:
                conn.execute(
                    "INSERT INTO documents VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                    (*scope, source_key, title, source_uri, revision, digest),
                )
            if old is None:
                acl = initial_acl(visibility)
                conn.execute(
                    "INSERT INTO document_access VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    (*scope, acl["owner"], acl["read"], acl["write"], acl["delete"]),
                )
            conn.execute(
                "INSERT INTO document_revisions "
                "(tenant,namespace,document_id,revision,title,source_uri,digest) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                (*scope, revision, title, source_uri, digest),
            )
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO nodes (tenant,namespace,document_id,revision,id,parent_id,"
                    "title,content,path,ordinal,locator,routing_text) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    [
                        (
                            *scope,
                            revision,
                            n.id,
                            n.parent_id,
                            n.title,
                            n.content,
                            Jsonb(n.path),
                            n.ordinal,
                            Jsonb(n.locator),
                            routing[n.id],
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

    def documents(self, tenant, namespace, limit=100, after=None):
        with self.scoped(tenant, namespace) as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE tenant=%s AND namespace=%s "
                "AND arborvault_document_allowed(tenant,namespace,id) "
                "AND (%s::uuid IS NULL OR id>%s::uuid) ORDER BY id LIMIT %s",
                (tenant, namespace, after, after, limit),
            ).fetchall()
        return [self._doc(row) for row in rows]

    def revisions(self, tenant, namespace, document_id, limit=100, before=None):
        with self.scoped(tenant, namespace) as conn:
            rows = conn.execute(
                "SELECT r.*,r.document_id AS id,d.source_key FROM document_revisions r "
                "JOIN documents d ON (r.tenant,r.namespace,r.document_id)=(d.tenant,d.namespace,d.id) "
                "WHERE r.tenant=%s AND r.namespace=%s AND r.document_id=%s "
                "AND arborvault_document_allowed(d.tenant,d.namespace,d.id) "
                "AND (%s::integer IS NULL OR r.revision<%s) ORDER BY r.revision DESC LIMIT %s",
                (tenant, namespace, document_id, before, before, limit),
            ).fetchall()
        return [self._doc(row) for row in rows]

    _join = """SELECT n.*, jsonb_build_object('id',d.id,'source_key',d.source_key,
        'title',r.title,'source_uri',r.source_uri,'revision',r.revision,'digest',r.digest) AS doc
        FROM nodes n JOIN document_revisions r USING (tenant,namespace,document_id,revision)
        JOIN documents d ON (n.tenant,n.namespace,n.document_id)=(d.tenant,d.namespace,d.id)
        WHERE n.tenant=%s AND n.namespace=%s
        AND arborvault_document_allowed(d.tenant,d.namespace,d.id) """

    def _nodes(
        self, conn, tenant, namespace, document_id=None, revision=None, limit=10001, after=-1
    ):
        query, params = self._join, [tenant, namespace]
        if document_id:
            query += "AND n.document_id=%s "
            params.append(document_id)
        if revision is not None:
            query += "AND n.revision=%s "
            params.append(revision)
        else:
            query += "AND n.revision=d.revision "
        query += "AND n.ordinal>%s "
        params.extend([after, limit])
        return [
            self._pair(row)
            for row in conn.execute(
                query + "ORDER BY n.document_id,n.ordinal LIMIT %s", params
            ).fetchall()
        ]

    def nodes(self, tenant, namespace, document_id=None, revision=None, limit=10001, after=-1):
        with self.scoped(tenant, namespace) as conn:
            return self._nodes(conn, tenant, namespace, document_id, revision, limit, after)

    def _children(self, conn, tenant, namespace, parents, query, limit):
        statement = self._join.replace(
            "SELECT n.*,",
            "SELECT n.id,n.parent_id,n.title,n.path,n.ordinal,n.locator,n.routing_text,n.routing_text AS content,",
        )
        statement += "AND n.revision=d.revision "
        params = [tenant, namespace]
        if parents is None:
            statement += "AND n.parent_id IS NULL "
        else:
            statement += "AND n.parent_id=ANY(%s::uuid[]) "
            params.append(parents)
        terms = " OR ".join(tokens(query))
        statement += "ORDER BY ts_rank_cd(to_tsvector('simple',n.routing_text), "
        statement += "websearch_to_tsquery('simple',%s)) DESC,d.source_key,n.ordinal LIMIT %s"
        rows = conn.execute(statement, (*params, terms, limit)).fetchall()
        return [
            (d, replace(n, content=row["routing_text"]))
            for row in rows
            for d, n in [self._pair(row)]
        ]

    def _passages(self, conn, tenant, namespace, identities):
        rows = conn.execute(
            self._join
            + "AND n.revision=d.revision AND n.id=ANY(%s::uuid[]) ORDER BY d.source_key,n.ordinal",
            (tenant, namespace, identities),
        ).fetchall()
        return [self._pair(row) for row in rows]

    def _lexical(self, conn, tenant, namespace, query, k):
        terms = " OR ".join(tokens(query))
        if not terms:
            return []
        query_sql = self._join.replace(
            "SELECT n.*,",
            "SELECT n.*, ts_rank_cd(n.search,websearch_to_tsquery('simple',%s)) AS rank,",
        )
        query_sql += "AND n.revision=d.revision AND n.content<>'' "
        query_sql += "AND n.search @@ websearch_to_tsquery('simple',%s) ORDER BY rank DESC,d.source_key,n.ordinal LIMIT %s"
        rows = conn.execute(query_sql, (terms, tenant, namespace, terms, k)).fetchall()
        return [(*self._pair(row), float(row["rank"])) for row in rows]

    def lexical(self, tenant, namespace, query, k):
        with self.scoped(tenant, namespace) as conn:
            return self._lexical(conn, tenant, namespace, query, k)

    @contextmanager
    def snapshot(self, tenant, namespace):
        with self.scoped(tenant, namespace, snapshot=True) as conn:
            yield _Snapshot(self, conn)

    def delete(self, tenant, namespace, identity):
        with self.scoped(tenant, namespace, operation="delete") as conn:
            row = conn.execute(
                "SELECT source_key FROM documents WHERE tenant=%s AND namespace=%s AND id=%s "
                "AND arborvault_document_allowed(tenant,namespace,id)",
                (tenant, namespace, identity),
            ).fetchone()
            if row is None:
                return False
            lock_key = document_id(tenant, namespace, row["source_key"])
            conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (lock_key,))
            return (
                conn.execute(
                    "DELETE FROM documents WHERE tenant=%s AND namespace=%s AND id=%s "
                    "AND arborvault_document_allowed(tenant,namespace,id)",
                    (tenant, namespace, identity),
                ).rowcount
                > 0
            )

    def get_acl(self, tenant, namespace, identity):
        with self.scoped(tenant, namespace, operation="manage") as conn:
            row = conn.execute(
                "SELECT owner,readers,writers,deleters FROM document_access "
                "WHERE tenant=%s AND namespace=%s AND document_id=%s "
                "AND arborvault_document_allowed(tenant,namespace,document_id)",
                (tenant, namespace, identity),
            ).fetchone()
        return (
            {
                "owner": row["owner"],
                "read": row["readers"],
                "write": row["writers"],
                "delete": row["deleters"],
            }
            if row
            else None
        )

    def set_acl(self, tenant, namespace, identity, acl):
        with self.scoped(tenant, namespace, operation="manage") as conn:
            row = conn.execute(
                "SELECT source_key FROM documents WHERE tenant=%s AND namespace=%s AND id=%s "
                "AND arborvault_document_allowed(tenant,namespace,id)",
                (tenant, namespace, identity),
            ).fetchone()
            if row is None:
                return False
            conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
                (document_id(tenant, namespace, row["source_key"]),),
            )
            return (
                conn.execute(
                    "UPDATE document_access SET readers=%s,writers=%s,deleters=%s "
                    "WHERE tenant=%s AND namespace=%s AND document_id=%s "
                    "AND arborvault_document_allowed(tenant,namespace,document_id)",
                    (acl["read"], acl["write"], acl["delete"], tenant, namespace, identity),
                ).rowcount
                > 0
            )

    def ping(self):
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT max(version) AS version,count(*) AS count FROM schema_migrations"
            ).fetchone()
            if row["version"] != SCHEMA_VERSION or row["count"] != SCHEMA_VERSION:
                raise RuntimeError("Schema migration required")
            conn.execute("SELECT 1 FROM documents LIMIT 0")
        return True

    def close(self):
        self.pool.close()


class _Snapshot:
    def __init__(self, store, conn):
        self.store, self.conn = store, conn

    def nodes(self, *args):
        return self.store._nodes(self.conn, *args)

    def lexical(self, *args):
        return self.store._lexical(self.conn, *args)

    def children(self, *args):
        return self.store._children(self.conn, *args)

    def passages(self, *args):
        return self.store._passages(self.conn, *args)


if __name__ == "__main__":
    from .migrate import main

    main()
