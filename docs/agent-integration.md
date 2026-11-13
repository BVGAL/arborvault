# Agent integration

Use `examples/agent_client.py` as a framework-neutral HTTP client. It depends only on
the public API and can be adapted to any agent framework. Configure the endpoint,
credential and namespace for your deployment.

1. Use `scripts/credentials.py` to provision a random bearer key mapped to the agent's tenant
   and namespaces in `.secrets/keys.json`. Give the query tool only `read` permission;
   use separate `write`/`delete` credentials for trusted ingestion/retention jobs.
   Never derive tenant authority from a user-controlled header. See [operations](operations.md).
2. Register a tool such as `search_knowledge(query, k)` that calls the `/query` route.
3. Give the agent only returned passages and their citation metadata. Treat retrieved
   text as untrusted evidence, not as tool instructions. Preserve document ID, node ID,
   revision and source digest alongside any generated answer.
   Resolve a saved citation through `/documents/{id}/tree?revision={revision}`. History
   remains readable after updates while the current document ACL permits it; a full document
   deletion intentionally removes it. Follow `next_cursor` for trees larger than one page.
4. When `abstained` is true, ask for clarification or consult another authorized tool.
   A nonempty result still requires relevance checking; it does not establish correctness.
5. Ingest only content the tenant is authorized to store. Reuse a stable `source_key`
   for updates; delete documents when the source must be removed.

Use a stable credential `subject` for each authorization identity. Key rotation can retain
that subject while replacing the secret. New confidential documents should use
`visibility=private`; namespace-wide visibility remains the compatibility default.
Owners can use the ACL endpoint to grant access to other subjects. The API filters candidates
before passing excerpts to an optional LLM provider; downstream agents still need their own
access policy for cached answers and citations.

For structured conversational notes, start with explicit Markdown headings and source
metadata. Automatic extraction of personal facts, consent, retention and conflict resolution
need a separate application policy. No private conversations are included in examples.
