import os
import secrets
from importlib.resources import files
from pathlib import Path
from uuid import uuid4

import httpx
import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from arborvault.access import CURRENT_PRINCIPAL, AuthorizedStore, Principal
from arborvault.benchmark import run as benchmark
from arborvault.migrate import migrate, provision_runtime_role
from arborvault.ollama import OllamaSelector
from arborvault.parsing import Section, build_nodes, parse
from arborvault.retrieval import TitleSelector, retrieve, routing_summaries
from arborvault.storage import PostgresStore, document_id

pytestmark = pytest.mark.postgres


@pytest.fixture
def isolated_database():
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is not configured")
    name = "arborvault_test_" + uuid4().hex
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    options = conninfo_to_dict(url)
    options["dbname"] = name
    try:
        yield make_conninfo(**options)
    finally:
        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))


def test_upgrade_preserves_legacy_current_revision_and_checksums(isolated_database):
    url = isolated_database
    identity = document_id("legacy", "docs", "guide")
    with psycopg.connect(url) as conn:
        conn.execute(files("arborvault").joinpath("schema.sql").read_text())
        conn.execute(
            "INSERT INTO documents VALUES ('legacy','docs',%s,'guide','Guide',NULL,3,'digest')",
            (identity,),
        )
        conn.execute(
            "INSERT INTO nodes (tenant,namespace,document_id,id,title,content,path,ordinal,locator) "
            "VALUES ('legacy','docs',%s,%s,'Guide','Original v0.1 evidence','[]',0,'{}')",
            (identity, str(uuid4())),
        )
    assert migrate(url) == [1, 2, 3, 4]
    assert migrate(url) == []
    store = PostgresStore(url)
    try:
        rows = store.nodes("legacy", "docs", identity, revision=3)
        assert rows[0][1].content == "Original v0.1 evidence"
        assert store.revisions("legacy", "docs", identity)[0].revision == 3
        with store.snapshot("legacy", "docs") as snapshot:
            assert (
                snapshot.children("legacy", "docs", None, "evidence", 10)[0][1].content
                == routing_summaries([rows[0][1]])[rows[0][1].id]
            )
    finally:
        store.close()
    with psycopg.connect(url) as conn:
        conn.execute("UPDATE schema_migrations SET checksum='tampered' WHERE version=1")
    with pytest.raises(RuntimeError, match="checksum"):
        migrate(url)


def test_rls_history_snapshot_and_runtime_role(isolated_database):
    url = isolated_database
    migrate(url)
    password = secrets.token_urlsafe(32)
    role = "arborvault_test_app_" + uuid4().hex[:12]
    provision_runtime_role(url, password, role)
    options = conninfo_to_dict(url)
    options.update(user=role, password=password)
    store = PostgresStore(make_conninfo(**options), enforce_runtime_role=True)

    def put(tenant, namespace, text):
        return store.upsert(
            tenant,
            namespace,
            "guide",
            "Guide",
            None,
            text,
            lambda i, r: build_nodes(parse(text.encode(), "a.txt"), i, r, "Guide"),
        )[0]

    try:
        first = put("a", "one", "backups old evidence")
        put("b", "one", "confidential other tenant")
        put("a", "two", "confidential other namespace")
        with store.scoped("a", "one") as conn:
            assert conn.execute("SELECT count(*) AS count FROM documents").fetchone()["count"] == 1
            assert conn.execute("SELECT count(*) AS count FROM nodes").fetchone()["count"] == 1
        with store.pool.connection() as conn:
            assert conn.execute("SELECT count(*) AS count FROM documents").fetchone()["count"] == 0
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            with store.scoped("a", "one") as conn:
                conn.execute(
                    "INSERT INTO documents VALUES ('b','one',%s,'bad','Bad',NULL,1,'x')",
                    (str(uuid4()),),
                )
        with store.snapshot("a", "one") as snapshot:
            assert snapshot.nodes("a", "one")[0][0].revision == 1
            put("a", "one", "backups new evidence")
            assert snapshot.lexical("a", "one", "backups", 1)[0][0].revision == 1
        assert store.nodes("a", "one")[0][0].revision == 2
        assert store.nodes("a", "one", first.id, 1)[0][1].content == "backups old evidence"
        assert store.revisions("b", "one", first.id) == []
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            with store.scoped("a", "one") as conn:
                conn.execute("UPDATE nodes SET content='changed'")
        assert store.delete("a", "one", first.id)
        assert store.revisions("a", "one", first.id) == []
        recreated = put("a", "one", "brand new evidence")
        assert recreated.id != first.id
        assert store.nodes("a", "one", first.id, 1) == []
        with pytest.raises(RuntimeError, match="non-owner"):
            PostgresStore(url, enforce_runtime_role=True)
    finally:
        store.close()
        with psycopg.connect(url) as conn:
            conn.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)))
            conn.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))


def test_document_acl_enforced_by_postgres_and_large_navigation(isolated_database):
    url = isolated_database
    migrate(url)
    role, password = "arborvault_acl_" + uuid4().hex[:12], secrets.token_urlsafe(32)
    provision_runtime_role(url, password, role)
    options = conninfo_to_dict(url)
    options.update(user=role, password=password)
    store = PostgresStore(make_conninfo(**options), enforce_runtime_role=True)
    owner = AuthorizedStore(store, Principal("owner"))
    reader = AuthorizedStore(store, Principal("reader"))
    admin = AuthorizedStore(store, Principal("admin", True))
    try:
        sections = [
            Section(1, "Topic " + str(i), "quartzunique" if i == 11000 else "ordinary", {})
            for i in range(11001)
        ]
        doc, _ = owner.upsert(
            "a",
            "docs",
            "large",
            "Large",
            None,
            "v1",
            lambda i, r: build_nodes(sections, i, r, "Large"),
            visibility="private",
        )
        assert reader.documents("a", "docs") == []
        assert reader.revisions("a", "docs", doc.id) == []
        assert reader.nodes("a", "docs", doc.id) == []
        assert reader.lexical("a", "docs", "ordinary", 5) == []
        assert retrieve(reader, "a", "docs", "Large quartzunique", 1, TitleSelector())[0] == []
        assert not reader.delete("a", "docs", doc.id)
        assert not reader.set_acl("a", "docs", doc.id, {"read": ["*"], "write": [], "delete": []})
        token = CURRENT_PRINCIPAL.set(Principal("reader"))
        try:
            with store.scoped("a", "docs") as conn:
                for table in ("documents", "nodes", "document_revisions"):
                    assert (
                        conn.execute(
                            sql.SQL("SELECT count(*) AS count FROM {}").format(
                                sql.Identifier(table)
                            )
                        ).fetchone()["count"]
                        == 0
                    )
        finally:
            CURRENT_PRINCIPAL.reset(token)
        # Grant read only: historical reads are visible, writes and deletes remain denied.
        assert owner.set_acl("a", "docs", doc.id, {"read": ["reader"], "write": [], "delete": []})
        hits, trace, fallback, reason = retrieve(
            reader, "a", "docs", "Large quartzunique", 1, TitleSelector(), "navigation"
        )
        assert hits[0][1].content == "quartzunique"
        assert any(item["shortlisted"] for item in trace)
        assert not fallback and reason is None
        with pytest.raises(psycopg.errors.UniqueViolation):
            reader.upsert("a", "docs", "large", "Overwrite", None, "v2", lambda *_: [])
        assert not reader.delete("a", "docs", doc.id)
        assert admin.nodes("a", "docs", doc.id) == []
        assert admin.set_acl("a", "docs", doc.id, {"read": [], "write": ["reader"], "delete": []})
        updated, _ = reader.upsert(
            "a",
            "docs",
            "large",
            "Large",
            None,
            "v2",
            lambda i, r: build_nodes([Section(0, "Title", "new evidence", {})], i, r, "Large"),
        )
        assert updated.revision == 2
        assert reader.nodes("a", "docs", doc.id, revision=1) == []
        assert len(owner.nodes("a", "docs", doc.id, revision=1, limit=12000)) > 10000
        assert owner.delete("a", "docs", doc.id)
        with httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    json={
                        "done": True,
                        "done_reason": "stop",
                        "message": {"content": "[]"},
                        "prompt_eval_count": 20,
                        "eval_count": 2,
                    },
                )
            )
        ) as client:
            selector = OllamaSelector("https://ollama.com/api", "synthetic-mock", client=client)
            report = benchmark(
                Path(__file__).resolve().parents[1] / "datasets/public-docs-v1",
                make_conninfo(**options),
                selector=selector,
            )
            assert len(report["methods"]["llm_navigation"]["queries"]) == 16
            assert report["methods"]["llm_navigation"]["recall_at_k"] == 0
            usage = [
                r["provider_usage"]["calls"]
                for name in ("llm_navigation", "llm_auto")
                for r in report["methods"][name]["queries"]
            ]
            assert sum(usage) == selector.usage()["calls"] == 32
            assert selector.usage()["failures"] == 0
    finally:
        store.close()
        with psycopg.connect(url) as conn:
            conn.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)))
            conn.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))
