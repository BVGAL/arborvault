"""Forward-only, checksum-verified migrations, including adoption of the v0.1 schema."""

import argparse
import hashlib
import os
from importlib.resources import files

import psycopg
from psycopg import sql

SCHEMA_VERSION = 4


def migrate(url):
    applied = []
    with psycopg.connect(url) as conn:
        conn.execute("SELECT pg_advisory_xact_lock(738152931)")
        conn.execute("""CREATE TABLE IF NOT EXISTS schema_migrations (
            version integer PRIMARY KEY, name text NOT NULL, checksum text NOT NULL,
            applied_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
        known = dict(conn.execute("SELECT version,checksum FROM schema_migrations").fetchall())
        if any(version > SCHEMA_VERSION for version in known):
            raise RuntimeError("Database schema is newer than this application")
        for path in sorted(
            files("arborvault").joinpath("migrations").iterdir(), key=lambda p: p.name
        ):
            if not path.name.endswith(".sql"):
                continue
            version = int(path.name.split("_", 1)[0])
            content = path.read_text(encoding="utf-8").replace("\r\n", "\n")
            digest = hashlib.sha256(content.encode()).hexdigest()
            if version in known:
                if known[version] != digest:
                    raise RuntimeError(f"Migration {version} checksum mismatch")
                continue
            conn.execute(content)
            conn.execute(
                "INSERT INTO schema_migrations(version,name,checksum) VALUES (%s,%s,%s)",
                (version, path.name, digest),
            )
            applied.append(version)
    return applied


def provision_runtime_role(url, password, role="arborvault_app"):
    """Administrator-only provisioning; never called by the API process."""
    if len(password) < 24:
        raise ValueError("Runtime database password must be at least 24 characters")
    with psycopg.connect(url) as conn:
        conn.execute("SELECT pg_advisory_xact_lock(738152932)")
        if not conn.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (role,)).fetchone():
            conn.execute(sql.SQL("CREATE ROLE {} LOGIN").format(sql.Identifier(role)))
        # Fixed role name or safely quoted explicit admin parameter. No request data.
        conn.execute(
            sql.SQL(
                "ALTER ROLE {} NOSUPERUSER NOCREATEDB NOCREATEROLE "
                "NOREPLICATION NOBYPASSRLS PASSWORD {}"
            ).format(sql.Identifier(role), sql.Literal(password))
        )
        conn.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(sql.Identifier(role)))
        conn.execute(
            sql.SQL("GRANT SELECT,INSERT,UPDATE,DELETE ON documents TO {}").format(
                sql.Identifier(role)
            )
        )
        conn.execute(
            sql.SQL("GRANT SELECT,INSERT,DELETE ON document_revisions,nodes TO {}").format(
                sql.Identifier(role)
            )
        )
        conn.execute(
            sql.SQL("GRANT SELECT ON schema_migrations TO {}").format(sql.Identifier(role))
        )
        conn.execute(
            sql.SQL("GRANT SELECT,INSERT,UPDATE,DELETE ON document_access TO {}").format(
                sql.Identifier(role)
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provision-role", action="store_true")
    args = parser.parse_args()
    url = os.environ["DATABASE_URL"]
    versions = migrate(url)
    if args.provision_role:
        provision_runtime_role(url, os.environ["APP_DATABASE_PASSWORD"])
    print(f"Schema {SCHEMA_VERSION} ready; applied migrations: {versions}")


if __name__ == "__main__":
    main()
