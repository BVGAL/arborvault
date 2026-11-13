# Upgrade to v0.3

Back up and verify the v0.2 database before upgrading. Keep the old image and dump for
rollback. Schema migration 4 adds document ACLs and backfills routing summaries for all
retained revisions; its duration depends on stored history. Stop API writers during migration.

```sh
python scripts/backup.py backup backups/before-v03.dump
python scripts/backup.py verify backups/before-v03.dump
docker compose stop api
docker compose up --build -d --wait
python scripts/smoke_compose.py
```

Compose runs the migration with administrator credentials, grants the new ACL table to
the restricted runtime role, and starts the API only afterward. Do not edit old migration
files or bypass checksum checks. Restore the pre-upgrade dump into a separate database
to roll back; v0.2 cannot run against schema 4.

## Authorization migration

Existing documents retain namespace-wide read/write/delete access (`["*"]`) and an empty
owner, because their original uploader identity was not recorded. Migration does not invent
an owner or silently promote existing credentials. New documents default to namespace-wide
visibility for compatibility; pass `visibility=private` to restrict them to their creator.

Issue a scoped ACL administrator explicitly when legacy documents need restrictions:

```sh
python scripts/credentials.py issue --id acl-admin --subject acl-admin --tenant demo \
  --namespaces demo --permissions manage_acl --token-file .arborvault-acl-admin-token
```

Use that credential to PUT `/v1/namespaces/demo/documents/{id}/acl`, for example
`{"read":["reader"],"write":["ingester"],"delete":["retention-job"]}`.
This permission manages ACLs but does not independently authorize reading private contents.
The creator remains owner and can always manage their document. Owners still require
appropriate credential-level permissions to invoke read/write/delete API routes.

An optional credential `subject` defaults to its stable `id` (or its digest for legacy
records without an ID). Retain the same subject when rotating keys, and revoke the old key.
Never change subjects accidentally: ACL membership references them. Wildcards apply only
inside the authenticated tenant/namespace. Current ACLs govern all retained revisions.

## Client and parser changes

- Tree reads now return at most 500 nodes by default (maximum page size 2,000).
  Follow `next_cursor`; pin `revision` across pages when reading a concurrently updated tree.
- Parsing defaults are 25 MB, 20 million characters, 2,000 PDF pages and 50,000 nodes.
  These remain resource budgets; they do not guarantee every file of that size completes.
- OCR defaults to `auto`; it runs on pages without extractable text. Use `always` for
  defective text layers or `off` for native text only. Compose includes English/Italian.
- A parser-version/OCR fingerprint is now part of the ingest digest. The first re-upload
  of a v0.2 source creates one new revision, even if its raw bytes are identical.
- Lazy routing may change ranking/tie order. Historical v0.2 measurements are not a v0.3
  performance guarantee. Broad sibling sets use lexical shortlisting before selection.
- Training and provider evaluation are optional. Existing deployments retain local routing
  until a provider URL is configured. Model weights and credentials are never in the image.
