"""Explicit operation permissions and hot-reloaded credential revocation."""

import hashlib
import hmac
import json
import re
from pathlib import Path

PERMISSIONS = {"read", "write", "delete"}
IDENTIFIER = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}\Z")


def key_digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


class Credentials:
    def __init__(self, records=None, filename=None):
        self.records = records
        self.filename = Path(filename) if filename else None
        self.load()

    def load(self):
        # Reopen on every request so replacing the mounted directory's file revokes
        # immediately across API workers. Malformed config fails closed.
        records = (
            json.loads(self.filename.read_text(encoding="utf-8")) if self.filename else self.records
        )
        if not isinstance(records, list) or not records:
            raise ValueError("Configure at least one scoped credential")
        validated, seen = [], set()
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("revoked", False), bool):
                raise ValueError("Invalid credential record")
            digest = record.get("key_sha256")
            if "key" in record:
                if not isinstance(record["key"], str) or len(record["key"]) < 24:
                    raise ValueError("Credential is too short")
                digest = key_digest(record["key"])
            namespaces, permissions = record.get("namespaces"), record.get("permissions")
            if (
                not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or digest in seen
                or not isinstance(record.get("tenant"), str)
                or not IDENTIFIER.fullmatch(record.get("tenant", ""))
                or not isinstance(namespaces, list)
                or not namespaces
                or not all(
                    isinstance(n, str) and (n == "*" or IDENTIFIER.fullmatch(n)) for n in namespaces
                )
                or not isinstance(permissions, list)
                or not permissions
                or not all(isinstance(p, str) for p in permissions)
                or not set(permissions) <= PERMISSIONS
            ):
                raise ValueError("Invalid credential scope; explicit permissions are required")
            seen.add(digest)
            validated.append({**record, "key_sha256": digest})
        return validated

    def authenticate(self, token):
        digest = key_digest(token)
        for record in self.load():
            if hmac.compare_digest(digest, record["key_sha256"]) and not record.get(
                "revoked", False
            ):
                return record
        return None
