# Operating v0.2

## Startup and health

Use `python scripts/configure.py` once for a new local installation, then
`docker compose up --build -d`. The database and parser have internal-only networks. The
API exposes loopback port 8000. `/healthz` means the process is alive; `/readyz` verifies
the expected database schema. Ingestion returns 503 when parser capacity is unavailable
and 504 when parsing exceeds its deadline; retry with backoff and the same source key.
Upserts are idempotent for identical content and metadata.

## Keys and quotas

After installing the package, issue a read-only credential locally:

```sh
python scripts/credentials.py issue --id agent-read --tenant example-agent --namespaces knowledge \
  --permissions read --token-file .arborvault-agent-token
python scripts/credentials.py revoke --id agent-read
```

The configuration file is replaced atomically. Compose mounts its directory so replacements
are visible without restarting API workers. Token files are not mounted into the API/parser.
Use `write` for ingest-only clients and `delete` only for clients allowed to purge all history.
Concurrent administrators must serialize edits to the local key file; it is not a distributed
credential management system. Changing environment-backed `ARBORVAULT_KEYS` requires restart.

Defaults are 60 requests/minute with burst 60 per credential per API process. Set
`RATE_PER_MINUTE` and `RATE_BURST` for your deployment. Replicas have independent buckets;
enforce aggregate quotas at your gateway. Parsers allow two concurrent processes and
bound each to ten seconds wall time, five seconds CPU and 384 MiB virtual address space
on Linux. These limits intentionally reject some complex documents; adjust only with testing.

## Backup and recovery

```sh
python scripts/backup.py backup backups/arborvault.dump
python scripts/backup.py verify backups/arborvault.dump
```

The backup command refuses to overwrite a file and never prints its contents. Restore
verification uses the running PostgreSQL container's matching-version tools and a fresh
temporary database, never the active database. It reports table counts and schema version.
For full recovery, restore a trusted dump into a separately created database, rerun role
provisioning (dumps omit roles/ACLs), validate historical reads, and switch application
configuration only after checks. Back up credential configuration separately through your
secret-management process. Test recovery regularly; volume persistence alone is not backup.

## Distribution and updates

Application version is 0.2.0. Compose base images are pinned by digest and dependency
versions constrained in `constraints.txt`; the wheel build backend is pinned too. These
pins improve repeatability but are not a promise of permanently vulnerability-free images
or a complete hash-locked supply chain. Refresh pins deliberately, audit and rerun CI.

Release archives/wheels must be built from a clean tagged commit. Never include `.env`,
`.secrets`, token files, dumps, uploaded documents, model caches or local reports. Docker
build context exclusions cover those files. `scripts/scan_public.py` checks tracked content.
Do not treat the local synthetic/draft benchmark as production acceptance criteria.
