# Security

This is a research MVP. Keep it on a trusted network. The default Compose setup publishes
only a loopback API port, uses random local credentials, runs the API without root, and
does not expose PostgreSQL. Put TLS, rate limits, identity management, quotas and monitoring
in front of any shared deployment. Do not publish the development Compose service directly.

Credentials live in environment variables; `.env`, private keys, logs, reports and data are
excluded from Git. Never put secrets in source URIs, source keys or document titles because
these are returned as citation metadata. API keys authorize all operations in their allowed
namespaces. Revocation means removing a key and restarting the service. Namespace is a
partition within a tenant; this version has no document-level ACLs or database RLS.

File parsers run in the API process. Size and expansion limits are not sufficient protection
against every malicious PDF or DOCX. Use isolated worker processes with CPU/time/memory
limits and malware screening before accepting untrusted public uploads. No external URI
fetching, arbitrary execution or model-selected tool execution is implemented.

The optional model selector sends text to the configured provider. Leave it unset unless
that transfer is authorized. Model output is validated against the current candidate IDs;
this prevents invented IDs from being accepted, not prompt injection or incorrect retrieval.

Report vulnerabilities privately through GitHub's private vulnerability reporting when
enabled. If unavailable, open an issue requesting a private contact without disclosing
exploit details or sensitive data. Never attach live credentials or customer documents.
