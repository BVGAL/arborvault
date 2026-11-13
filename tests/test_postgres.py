import os
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient

from arborvault.api import create_app
from arborvault.parsing import build_nodes, parse
from arborvault.storage import PostgresStore

pytestmark = pytest.mark.postgres


@pytest.fixture
def store():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is not configured")
    store = PostgresStore(url)
    store.migrate()
    yield store
    store.close()


def test_postgres_atomic_revisions_scoping_and_foreign_keys(store):
    tenant, other = str(uuid4()), str(uuid4())
    sections = parse(b"# Guide\n## Backups\nRetain backups for thirty days.", "a.md")

    def put(i):
        return store.upsert(
            tenant,
            "one",
            "source",
            "Guide",
            None,
            str(i),
            lambda identity, rev: build_nodes(sections, identity, rev, "Guide"),
        )

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(put, range(8)))
        doc = store.documents(tenant, "one")[0]
        assert doc.revision == 8
        assert store.nodes(other, "one") == []
        assert store.nodes(tenant, "two") == []
        assert store.lexical(other, "one", "backups", 3) == []
        assert store.lexical(tenant, "one", "How long are backups retained?", 3)
        assert not store.delete(other, "one", doc.id)
        rows = store.nodes(tenant, "one", doc.id)
        root = next(n for _, n in rows if n.parent_id is None)
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            with store.pool.connection() as conn:
                conn.execute(
                    "UPDATE nodes SET parent_id=%s WHERE tenant=%s AND namespace=%s AND id=%s",
                    (str(uuid4()), tenant, "one", root.id),
                )

        def broken(*args):
            raise ValueError("synthetic failure")

        with pytest.raises(ValueError):
            store.upsert(tenant, "one", "source", "Guide", None, "broken", broken)
        assert store.documents(tenant, "one")[0].revision == 8
        key = "synthetic-postgres-key-000000000"
        with TestClient(
            create_app(store, [{"key": key, "tenant": tenant, "namespaces": ["one"]}])
        ) as client:
            response = client.post(
                "/v1/namespaces/one/query",
                json={"query": "backups"},
                headers={"Authorization": "Bearer " + key},
            )
            assert response.status_code == 200
            assert response.json()["hits"][0]["revision"] == 8
        assert store.delete(tenant, "one", doc.id)
        assert store.nodes(tenant, "one") == []
    finally:
        for doc in store.documents(tenant, "one"):
            store.delete(tenant, "one", doc.id)
