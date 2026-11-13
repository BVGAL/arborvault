# Upgrade from v0.1 to v0.2

Changes requiring attention: explicit credential permissions; a separate restricted database
role; retained history; an additional parser service. Back up before migration. Migration is
forward-only and transactional; it preserves the revision currently stored in v0.1 but cannot
recover older revisions already discarded by v0.1.

1. Stop writes: `docker compose stop api`. Retain the database volume.
2. Check out v0.2, then run `python scripts/configure.py --upgrade`. It preserves the existing
   database password, makes a private `.env.v0.1-backup`, adds an independent runtime database
   password and converts API keys to hashes with explicit existing read/write/delete capability.
   The first token is saved separately for the local example. It does not print credentials.
3. Run `docker compose up -d db`, then `python scripts/backup.py backup backups/pre-v02.dump`.
   This is a PostgreSQL custom-format backup; protect it like the source documents.
4. Run `docker compose up --build -d`. The migration job adopts v0.1, creates revision storage,
   backfills current content, enables RLS and grants only the required runtime privileges.
5. Verify `/readyz`, read an existing document, ingest a new revision, and fetch
   `/documents/{id}/tree?revision=<old revision>`. Check query citations and client permissions.
6. Make a new v0.2 backup and run `python scripts/backup.py verify backups/post-v02.dump`.
   Verification restores into a freshly named temporary database and removes only that database.

For non-Compose deployments, run `python -m arborvault.migrate` with administrator
`DATABASE_URL`; provision a non-owner NOBYPASSRLS API login using the migration CLI's
`--provision-role` and `APP_DATABASE_PASSWORD`, or equivalent restricted grants. Run the
API with that login. New credential records require a nonempty `permissions` list.

## Rollback and identity

Do not run v0.1 against a migrated v0.2 database: v0.1 does not preserve the new invariants.
To roll back, stop writes, create a separate empty database, restore the pre-upgrade dump,
and point v0.1 at that database with the saved v0.1 configuration. Keep the migrated volume
until verification is complete. Changes made after the backup need explicit reconciliation.

Document IDs for existing sources are preserved. New lifecycles use new UUIDs; never derive
IDs client-side. A DELETE purges all revisions and re-ingestion creates a different ID.
Listing responses add `next_cursor`; clients should follow it rather than assume a complete
list. The `/v1` route family remains, with additive history routes and explicit authorization.
