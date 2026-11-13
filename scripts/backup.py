"""Local Compose backup and non-destructive restore verification. Requires Python and Docker."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def run(*command, data=None):
    result = subprocess.run(
        ["docker", "compose", "exec", "-T", "db", *command],
        cwd=ROOT,
        input=data,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError("Database maintenance command failed; inspect container health")
    return result.stdout


def restore_check(path):
    # Always restore into a new disposable database. Never overwrite a live database.
    database = "arborvault_restore_" + uuid4().hex
    run("createdb", "-U", "arborvault", database)
    try:
        run(
            "pg_restore",
            "-U",
            "arborvault",
            "--dbname",
            database,
            "--no-owner",
            "--no-acl",
            "--exit-on-error",
            data=path.read_bytes(),
        )
        result = run(
            "psql",
            "-U",
            "arborvault",
            "-d",
            database,
            "-At",
            "-c",
            "SELECT json_build_object('documents',(SELECT count(*) FROM documents),"
            "'revisions',(SELECT count(*) FROM document_revisions),"
            "'nodes',(SELECT count(*) FROM nodes),"
            "'schema_version',(SELECT max(version) FROM schema_migrations))",
        )
        return json.loads(result)
    finally:
        run("dropdb", "-U", "arborvault", database)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["backup", "verify"])
    parser.add_argument("file", type=Path)
    args = parser.parse_args()
    if args.action == "backup":
        if args.file.exists():
            parser.error("Refusing to overwrite an existing backup")
        data = run(
            "pg_dump",
            "-U",
            "arborvault",
            "-d",
            "arborvault",
            "--format=custom",
            "--no-owner",
            "--no-acl",
        )
        args.file.parent.mkdir(parents=True, exist_ok=True)
        with args.file.open("xb") as stream:
            stream.write(data)
        args.file.chmod(0o600)
        print(json.dumps({"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}))
    else:
        print(json.dumps(restore_check(args.file)))


if __name__ == "__main__":
    main()
