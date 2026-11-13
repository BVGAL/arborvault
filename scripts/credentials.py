"""Issue or revoke credentials in a local file; API processes reload it on each request."""

import argparse
import json
import os
import secrets
from pathlib import Path

from arborvault.auth import Credentials, key_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["issue", "revoke"])
    parser.add_argument("--id", required=True)
    parser.add_argument("--file", type=Path, default=Path(".secrets/keys.json"))
    parser.add_argument("--tenant")
    parser.add_argument("--namespaces", nargs="+")
    parser.add_argument(
        "--permissions", nargs="+", choices=["read", "write", "delete", "manage_acl"]
    )
    parser.add_argument("--subject", help="Stable ACL identity; defaults to credential ID")
    parser.add_argument("--token-file", type=Path)
    args = parser.parse_args()
    records = json.loads(args.file.read_text())
    matches = [r for r in records if r.get("id") == args.id]
    if args.action == "revoke":
        if len(matches) != 1:
            parser.error("Credential ID must identify exactly one credential")
        matches[0]["revoked"] = True
    else:
        if matches or not all([args.tenant, args.namespaces, args.permissions, args.token_file]):
            parser.error("Issue needs a new ID, tenant, namespaces, permissions and token-file")
        token = secrets.token_urlsafe(32)
        records.append(
            {
                "id": args.id,
                "subject": args.subject or args.id,
                "key_sha256": key_digest(token),
                "tenant": args.tenant,
                "namespaces": args.namespaces,
                "permissions": args.permissions,
            }
        )
        Credentials(records)
        with args.token_file.open("x", encoding="utf-8") as stream:
            stream.write(token)
        args.token_file.chmod(0o600)
    Credentials(records)
    temporary = args.file.with_suffix(".new")
    temporary.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    temporary.chmod(0o644)
    os.replace(temporary, args.file)
    print("Credential configuration updated; secret values were not printed.")


if __name__ == "__main__":
    main()
