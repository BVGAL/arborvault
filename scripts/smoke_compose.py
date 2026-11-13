"""Exercise a configured local Compose stack; create and delete only a unique test document."""

import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from uuid import uuid4

from backup import restore_check, run

ROOT = Path(__file__).resolve().parents[1]


def main():
    token = os.getenv("ARBORVAULT_API_KEY") or (ROOT / ".arborvault-demo-token").read_text().strip()
    base = os.getenv("ARBORVAULT_URL", "http://127.0.0.1:8000").rstrip("/")
    namespace = os.getenv("ARBORVAULT_NAMESPACE", "demo")
    prefix = f"/v1/namespaces/{namespace}"
    source = "compose-smoke-" + uuid4().hex

    def request(path, method="GET", data=None):
        headers = {"Authorization": "Bearer " + token}
        if isinstance(data, dict):
            data = json.dumps(data).encode()
            headers["Content-Type"] = "application/json"
        elif data is not None:
            headers["Content-Type"] = "application/octet-stream"
        req = urllib.request.Request(base + path, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=30) as response:
            body = response.read()
            return json.loads(body) if body else None

    assert request("/readyz")["status"] == "ready"
    identity = None
    try:
        endpoint = (
            prefix
            + "/documents?"
            + urllib.parse.urlencode(
                {"filename": "smoke.md", "source_key": source, "title": "Synthetic restore guide"}
            )
        )
        first = request(endpoint, "POST", b"# Backups\nRetention is seven days.\n")
        identity = first["document"]["id"]
        assert first["changed"] and first["document"]["revision"] == 1
        same = request(endpoint, "POST", b"# Backups\nRetention is seven days.\n")
        assert not same["changed"]
        second = request(endpoint, "POST", b"# Backups\nRetention is fourteen days.\n")
        assert second["document"]["id"] == identity and second["document"]["revision"] == 2
        path = prefix + "/documents/" + identity
        historical = request(path + "/tree?revision=1")
        assert "seven days" in " ".join(n["content"] for n in historical["nodes"])
        page = request(path + "/revisions?limit=1")
        assert page["revisions"][0]["revision"] == 2 and page["next_cursor"] == 2
        assert request(path + "/revisions?cursor=2")["revisions"][0]["revision"] == 1
        hits = request(prefix + "/query", "POST", {"query": "fourteen", "mode": "lexical"})
        assert any(h["document_id"] == identity and h["revision"] == 2 for h in hits["hits"])
        with tempfile.TemporaryDirectory(prefix="arborvault-drill-") as temporary:
            backup = Path(temporary) / "test.dump"
            backup.write_bytes(
                run(
                    "pg_dump",
                    "-U",
                    "arborvault",
                    "-d",
                    "arborvault",
                    "--format=custom",
                    "--no-owner",
                    "--no-acl",
                )
            )
            backup.chmod(0o600)
            restored = restore_check(backup)
            assert restored["revisions"] >= 2 and restored["schema_version"] == 3
        print(json.dumps({"compose_smoke": "passed", "restore": restored}))
    finally:
        if identity:
            request(prefix + "/documents/" + identity, "DELETE")
    try:
        request(prefix + "/documents/" + identity + "/tree?revision=1")
    except urllib.error.HTTPError as error:
        assert error.code == 404
    else:
        raise AssertionError("Deleted history is still accessible")


if __name__ == "__main__":
    main()
