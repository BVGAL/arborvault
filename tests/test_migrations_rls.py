import os
import secrets
from importlib.resources import files
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from arborvault.migrate import migrate, provision_runtime_role
from arborvault.parsing import build_nodes, parse
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
    assert migrate(url) == [1, 2, 3]
    assert migrate(url) == []
    store = PostgresStore(url)
    try:
        rows = store.nodes("legacy", "docs", identity, revision=3)
        assert rows[0][1].content == "Original v0.1 evidence"
        assert store.revisions("legacy", "docs", identity)[0].revision == 3
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
