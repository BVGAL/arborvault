# Security

ArborVault v0.3 is a research preview, not a completed production security assessment.
Use the supplied Compose topology or equivalent controls. Keep TLS, network admission,
shared quotas and monitoring at your gateway for any shared deployment.

## Identity and authorization

Bearer tokens must be generated randomly with at least 24 characters. The default setup
stores only SHA-256 token hashes in `.secrets/keys.json`; the bootstrap plaintext token is
kept separately in `.arborvault-demo-token`. Hashing here assumes high-entropy generated
tokens, not human passwords. Tokens map to a tenant, namespace allowlist and explicit
`read`, `write`, `delete` permissions. Missing permissions fail closed. File-backed revocation
takes effect on the next request in every API process. Environment-backed credentials are
also supported but require process restart after changes. See `scripts/credentials.py`.

RLS independently filters documents, revisions and nodes. API startup rejects a database
owner, superuser or BYPASSRLS role. The administrator migration job is separate from the
API. Database service credentials remain trusted; a compromised API can set another tenant
context, so this is defense against application mistakes, not isolation from a compromised
service. Document ACLs filter all API read surfaces, retrieval candidates and historical
revisions before provider calls. Subjects are taken only from authenticated credential
records; never accept a caller-supplied subject header. Shared credentials still share authority.
Owners can manage their own ACLs; `manage_acl` allows management only within the credential's
tenant/namespaces and does not independently grant content access. ACL wildcards mean all
authenticated subjects in that namespace, never other tenants. Existing documents migrate
with namespace-wide ACLs for compatibility; use the upgrade guide to restrict them explicitly.
An in-flight repeatable-read query retains its starting ACL snapshot. Revocation affects
subsequent requests; it cannot recall excerpts already sent to a provider.

## Document processing and providers

The Compose parser has no database/provider credentials, no external network, no published
port and no persistent volume. Resource-limited disposable subprocesses handle parsing;
the API handles busy/time-out responses without falling back to parsing in-process. Local
subprocess mode does not provide a container boundary and must not be exposed to hostile
public uploads. Resource limits do not prove absence of parser vulnerabilities; keep
dependencies current and apply host/container security maintenance.

Source URIs are metadata and are never fetched. The optional model selector sends queries,
titles and excerpts to the configured provider; leave it disabled without authorization.
Validating model-selected IDs does not guarantee relevant retrieval or prevent every prompt
injection. Agents must treat returned text as evidence, never tool instructions.

## Retention and disclosure

Updates retain historical extracted content. Deleting a document purges every stored
revision, but not copies in your backups or downstream agent logs. Define retention for
those systems separately. Backups contain document data and require private storage.
Never put secrets in source metadata or document content intended for public sharing.
Default request access logs are disabled; application logs use route templates and request IDs.

`.env`, `.secrets`, plaintext token files, databases, dumps, model caches and reports are
excluded from source archives. The public evaluation corpus has explicit upstream licenses
and includes only public documentation; its example passwords are illustrative upstream text.

Report vulnerabilities through GitHub private vulnerability reporting when enabled. If that
channel is unavailable, request a private contact without posting exploit details or live
credentials. The repository's scanners are complementary checks, not a substitute for review.
