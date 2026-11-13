"""Framework-neutral HTTP client for integrating an AI agent with ArborVault."""

import os
from pathlib import Path

import httpx


class KnowledgeClient:
    def __init__(self, base_url, api_key, namespace):
        from urllib.parse import quote

        self.client = httpx.Client(
            base_url=base_url, timeout=30, headers={"Authorization": f"Bearer {api_key}"}
        )
        self.prefix = f"/v1/namespaces/{quote(namespace, safe='')}"

    def ingest(self, source_key, title, path):
        path = Path(path)
        response = self.client.post(
            self.prefix + "/documents",
            content=path.read_bytes(),
            params={"source_key": source_key, "title": title, "filename": path.name},
        )
        response.raise_for_status()
        return response.json()

    def retrieve(self, query, k=5):
        response = self.client.post(self.prefix + "/query", json={"query": query, "k": k})
        response.raise_for_status()
        return response.json()

    def close(self):
        self.client.close()


if __name__ == "__main__":
    client = KnowledgeClient(
        os.getenv("ARBORVAULT_URL", "http://localhost:8000"),
        os.environ["ARBORVAULT_API_KEY"],
        "demo",
    )
    try:
        client.ingest("guide", "Demo guide", Path(__file__).with_name("guide.md"))
        result = client.retrieve("backups retained", 3)
        for hit in result["hits"]:
            print(hit["path"], hit["text"], hit["citation"])
    finally:
        client.close()
