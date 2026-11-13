"""Generate private local configuration or explicitly upgrade an existing v0.1 setup."""

import argparse
import hashlib
import json
import os
import secrets
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upgrade", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path, keyfile = root / ".env", root / ".secrets" / "keys.json"
    if keyfile.exists():
        raise SystemExit("v0.2 credentials already exist; left unchanged")
    if path.exists() and not args.upgrade:
        raise SystemExit(".env exists. Back up the database, then use --upgrade for v0.1.")
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    values = dict(
        line.split("=", 1) for line in old.splitlines() if "=" in line and not line.startswith("#")
    )
    records = json.loads(values.get("ARBORVAULT_KEYS", "[]").strip("'"))
    if not records:
        records = [{"key": secrets.token_urlsafe(32), "tenant": "demo", "namespaces": ["demo"]}]
    raw = records[0]["key"]
    hashed = [
        {
            "id": f"initial-{i}",
            "key_sha256": hashlib.sha256(record["key"].encode()).hexdigest(),
            "tenant": record["tenant"],
            "namespaces": record["namespaces"],
            "permissions": record.get("permissions", ["read", "write", "delete"]),
        }
        for i, record in enumerate(records)
    ]
    keyfile.parent.mkdir(mode=0o755, exist_ok=True)
    keyfile.write_text(json.dumps(hashed, indent=2) + "\n", encoding="utf-8")
    keyfile.chmod(0o644)  # High-entropy hashes, read by unprivileged container UID.
    token_file = root / ".arborvault-demo-token"
    with token_file.open("x", encoding="utf-8") as stream:
        stream.write(raw)
    token_file.chmod(0o600)
    if old:
        backup = root / ".env.v0.1-backup"
        with backup.open("x", encoding="utf-8") as stream:
            stream.write(old)
        backup.chmod(0o600)
    lines = [
        line
        for line in old.splitlines()
        if not line.startswith(("ARBORVAULT_KEYS=", "APP_DATABASE_PASSWORD="))
    ]
    if "POSTGRES_PASSWORD" not in values:
        lines.append("POSTGRES_PASSWORD=" + secrets.token_hex(24))
    lines.append("APP_DATABASE_PASSWORD=" + secrets.token_hex(24))
    temporary = path.with_name(".env.new")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)
    print("Configuration ready. Bearer token: .arborvault-demo-token (never commit).")
    print("Start with docker compose up --build -d. Credentials are not printed.")


if __name__ == "__main__":
    main()
