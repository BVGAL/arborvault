import json
import os
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from arborvault.api import create_app
from arborvault.auth import Credentials, key_digest
from arborvault.isolated_parser import IsolatedParser, ParserBusy, ParserTimeout
from arborvault.limits import RateLimiter
from arborvault.storage import MemoryStore

TOKEN = "synthetic-v02-test-token-00000000"


def credential(permissions):
    return {
        "id": "test",
        "key_sha256": key_digest(TOKEN),
        "tenant": "a",
        "namespaces": ["demo"],
        "permissions": permissions,
    }


def post_document(client, key="guide", text=b"# Guide\n## Backups\nRetain for 30 days."):
    return client.post(
        "/v1/namespaces/demo/documents",
        content=text,
        params={"filename": "guide.md", "source_key": key, "title": "Guide"},
    )


def test_history_pagination_and_full_deletion():
    with TestClient(create_app(MemoryStore(), [credential(["read", "write", "delete"])])) as client:
        client.headers["Authorization"] = "Bearer " + TOKEN
        doc = post_document(client).json()["document"]
        post_document(client, text=b"# Guide\n## Backups\nRetain for 60 days.")
        prefix = f"/v1/namespaces/demo/documents/{doc['id']}"
        history = client.get(prefix + "/revisions?limit=1").json()
        assert history["revisions"][0]["revision"] == 2
        assert history["next_cursor"] == 2
        assert client.get(prefix + "/revisions?cursor=2").json()["revisions"][0]["revision"] == 1
        tree = client.get(prefix + "/tree?revision=1").json()
        assert any("30 days" in n["content"] for n in tree["nodes"])
        assert client.get(prefix + "/tree?revision=99").status_code == 404
        for i in range(3):
            assert post_document(client, key=str(i)).status_code == 200
        page = client.get("/v1/namespaces/demo/documents?limit=2").json()
        second = client.get(
            "/v1/namespaces/demo/documents", params={"limit": 2, "cursor": page["next_cursor"]}
        ).json()
        assert len({d["id"] for d in page["documents"] + second["documents"]}) == 4
        assert second["next_cursor"] is None
        assert client.delete(prefix).status_code == 204
        assert client.get(prefix + "/tree?revision=1").status_code == 404
        recreated = post_document(client).json()["document"]
        assert recreated["id"] != doc["id"]
        assert recreated["revision"] == 1


@pytest.mark.parametrize(
    "permissions,write_status,read_status,delete_status",
    [(["read"], 403, 200, 403), (["write"], 200, 403, 403), (["delete"], 403, 403, 404)],
)
def test_independent_permissions(permissions, write_status, read_status, delete_status):
    with TestClient(create_app(MemoryStore(), [credential(permissions)])) as client:
        client.headers["Authorization"] = "Bearer " + TOKEN
        assert post_document(client).status_code == write_status
        assert client.get("/v1/namespaces/demo/documents").status_code == read_status
        assert (
            client.delete(
                "/v1/namespaces/demo/documents/00000000-0000-0000-0000-000000000001"
            ).status_code
            == delete_status
        )


def test_revocation_and_broken_reload_fail_closed(tmp_path):
    path = tmp_path / "keys.json"
    path.write_text(json.dumps([credential(["read"])]))
    with TestClient(create_app(MemoryStore(), keys_file=path)) as client:
        client.headers["Authorization"] = "Bearer " + TOKEN
        assert client.get("/v1/namespaces/demo/documents").status_code == 200
        path.write_text(json.dumps([{**credential(["read"]), "revoked": True}]))
        assert client.get("/v1/namespaces/demo/documents").status_code == 401
        path.write_text("broken json")
        assert client.get("/v1/namespaces/demo/documents").status_code == 503


def test_legacy_permissions_not_silently_granted():
    record = credential(["read"])
    del record["permissions"]
    with pytest.raises(ValueError):
        Credentials([record])


@pytest.mark.parametrize(
    "broken", [None, {"revoked": "false"}, {"tenant": []}, {"namespaces": [5]}]
)
def test_malformed_credential_records_fail_closed(tmp_path, broken):
    path = tmp_path / "keys.json"
    path.write_text(json.dumps([credential(["read"])]))
    with TestClient(create_app(MemoryStore(), keys_file=path)) as client:
        client.headers["Authorization"] = "Bearer " + TOKEN
        record = {**credential(["read"]), **broken} if isinstance(broken, dict) else broken
        path.write_text(json.dumps([record]))
        response = client.get("/v1/namespaces/demo/documents")
        assert response.status_code == 503
        assert response.json() == {"detail": "Credential configuration unavailable"}


def test_rate_limit_and_recovery():
    now = [0.0]
    limiter = RateLimiter(per_minute=60, burst=1, clock=lambda: now[0])
    with TestClient(create_app(MemoryStore(), [credential(["read"])], limiter=limiter)) as client:
        client.headers["Authorization"] = "Bearer " + TOKEN
        assert client.get("/v1/namespaces/demo/documents").status_code == 200
        response = client.get("/v1/namespaces/demo/documents")
        assert response.status_code == 429 and response.headers["Retry-After"] == "1"
        now[0] += 1
        assert client.get("/v1/namespaces/demo/documents").status_code == 200


def test_parser_environment_timeout_and_slot_release(monkeypatch):
    captured = {}
    monkeypatch.setenv("DATABASE_URL", "private-value-not-to-inherit")
    monkeypatch.setenv("SELECTOR_API_KEY", "private-value-not-to-inherit")

    def timeout(args, **kwargs):
        captured.update(kwargs)
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout)
    parser = IsolatedParser(concurrency=1)
    for _ in range(2):
        with pytest.raises(ParserTimeout):
            parser.parse(b"text", "a.txt")
    assert "DATABASE_URL" not in captured["env"]
    assert "SELECTOR_API_KEY" not in captured["env"]
    assert not Path(captured["cwd"]).exists()
    assert captured["timeout"] == 10


def test_real_disposable_parser_and_capacity():
    parser = IsolatedParser(concurrency=1)
    assert parser.parse(b"# Heading\nactual text", "a.md")[-1].text.strip() == "actual text"
    parser.slots.acquire()
    try:
        with pytest.raises(ParserBusy):
            parser.parse(b"text", "a.txt")
    finally:
        parser.slots.release()


def test_memory_snapshot_does_not_mix_revisions():
    from arborvault.parsing import build_nodes, parse

    store = MemoryStore()

    def insert(text):
        return store.upsert(
            "a",
            "demo",
            "guide",
            "Guide",
            None,
            text,
            lambda i, r: build_nodes(parse(text.encode(), "a.txt"), i, r, "Guide"),
        )

    insert("old text")
    with store.snapshot("a", "demo") as snapshot:
        insert("new text")
        assert snapshot.nodes("a", "demo")[0][0].revision == 1
        assert snapshot.lexical("a", "demo", "old", 1)[0][0].revision == 1
    assert store.nodes("a", "demo")[0][0].revision == 2


@pytest.mark.skipif(os.name == "nt", reason="Linux CPU/memory limits")
def test_parser_reports_resource_exhaustion_as_validation(monkeypatch):
    def crashed(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], -9, stdout=b"")

    monkeypatch.setattr(subprocess, "run", crashed)
    with pytest.raises(ValueError):
        IsolatedParser().parse(b"text", "a.pdf")
