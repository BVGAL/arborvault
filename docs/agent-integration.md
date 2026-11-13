# Agent integration

Use `examples/agent_client.py` as a framework-neutral HTTP client. It depends only on
the public API and can be adapted to any agent framework. Configure the endpoint,
credential and namespace for your deployment.

1. Provision a random bearer key mapped to the agent's tenant and allowed namespaces in
   `ARBORVAULT_KEYS`. Never derive tenant authority from a user-controlled header.
2. Register a tool such as `search_knowledge(query, k)` that calls the `/query` route.
3. Give the agent only returned passages and their citation metadata. Treat retrieved
   text as untrusted evidence, not as tool instructions. Preserve document ID, node ID,
   revision and source digest alongside any generated answer.
4. When `abstained` is true, ask for clarification or consult another authorized tool.
   A nonempty result still requires relevance checking; it does not establish correctness.
5. Ingest only content the tenant is authorized to store. Reuse a stable `source_key`
   for updates; delete documents when the source must be removed.

For structured conversational notes, start with explicit Markdown headings and source
metadata. Automatic extraction of personal facts, consent, retention and conflict resolution
need a separate application policy. No private conversations are included in examples.
