# Contributing

Use a virtual environment and install `pip install -c constraints.txt -e '.[dev]'`. Run `pytest`,
`ruff check .` and `ruff format --check .`. PostgreSQL integration tests require a disposable `TEST_DATABASE_URL`;
the CI service supplies PostgreSQL 17. Add meaningful regression coverage for changes
to tenant boundaries, parsing, persistence and retrieval. Use synthetic or clearly licensed
fixtures only. Do not commit private corpora, customer prompts, credentials or `.env`.

Explain behavior changes and validation in pull requests. Keep agent adapters outside
the core. Benchmark claims need the corpus provenance, model revision, parameters, raw
per-query results and hardware information. Contributions are under Apache-2.0.
