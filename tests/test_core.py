import io
from concurrent.futures import ThreadPoolExecutor

import pytest
from docx import Document as Docx
from fastapi.testclient import TestClient

from arborvault.api import create_app
from arborvault.evaluate import metrics, run
from arborvault.parsing import MAX_BYTES, build_nodes, parse
from arborvault.retrieval import TitleSelector, navigate
from arborvault.storage import MemoryStore

KEY_A = "synthetic-test-credential-a-000000"
KEY_B = "synthetic-test-credential-b-000000"
KEYS = [
    {
        "key": KEY_A,
        "tenant": "a",
        "namespaces": ["demo", "second"],
        "permissions": ["read", "write", "delete"],
    },
    {
        "key": KEY_B,
        "tenant": "b",
        "namespaces": ["demo"],
        "permissions": ["read", "write", "delete"],
    },
]
TEXT = b"# Guide\n## Backups\nBackups last 30 days.\n## Access\nAccess lasts 24 hours."


@pytest.fixture
def client():
    with TestClient(create_app(MemoryStore(), KEYS)) as client:
        client.headers["Authorization"] = "Bearer " + KEY_A
        yield client


def upload(client, text=TEXT, namespace="demo", source="guide"):
    return client.post(
        f"/v1/namespaces/{namespace}/documents",
        content=text,
        params={"filename": "guide.md", "source_key": source, "title": "Guide"},
    )


def test_ingest_idempotence_update_and_citation(client):
    first = upload(client).json()
    assert first["changed"] is True
    assert upload(client).json()["changed"] is False
    result = client.post("/v1/namespaces/demo/query", json={"query": "backups", "k": 1}).json()
    assert result["hits"][0]["text"] == "Backups last 30 days."
    assert result["hits"][0]["citation"]["locator"]["line"] == 2
    old_id = result["hits"][0]["node_id"]
    updated = upload(client, TEXT.replace(b"30", b"60")).json()["document"]
    assert updated["id"] == first["document"]["id"]
    assert updated["revision"] == 2
    result = client.post("/v1/namespaces/demo/query", json={"query": "backups", "k": 1}).json()
    assert "60" in result["hits"][0]["text"]
    assert old_id != result["hits"][0]["node_id"]


def test_tenant_and_namespace_isolation(client):
    identity = upload(client).json()["document"]["id"]
    client.headers["Authorization"] = "Bearer " + KEY_B
    assert client.get("/v1/namespaces/demo/documents").json()["documents"] == []
    assert client.get(f"/v1/namespaces/demo/documents/{identity}/tree").status_code == 404
    assert client.delete(f"/v1/namespaces/demo/documents/{identity}").status_code == 404
    assert client.post("/v1/namespaces/demo/query", json={"query": "backups"}).json()["hits"] == []
    assert client.get("/v1/namespaces/second/documents").status_code == 403
    client.headers["Authorization"] = "Bearer " + KEY_A
    assert client.get("/v1/namespaces/second/documents").json()["documents"] == []
    assert client.delete(f"/v1/namespaces/demo/documents/{identity}").status_code == 204
    assert client.get(f"/v1/namespaces/demo/documents/{identity}/tree").status_code == 404


def test_auth_validation_and_size(client):
    assert client.post("/v1/namespaces/demo/query", json={"query": " "}).status_code == 422
    assert (
        client.post("/v1/namespaces/demo/query", json={"query": "x", "tenant": "b"}).status_code
        == 422
    )
    assert upload(client, b"x" * (MAX_BYTES + 1)).status_code == 413
    assert upload(client, b"\xff").status_code == 422
    assert upload(client, b"").status_code == 422
    client.headers.clear()
    assert client.get("/v1/namespaces/demo/documents").status_code == 401


def test_selector_failure_and_invalid_ids_fall_back():
    class Broken:
        def select(self, *args):
            return ["foreign-node"]

    with TestClient(create_app(MemoryStore(), KEYS, Broken())) as client:
        client.headers["Authorization"] = "Bearer " + KEY_A
        upload(client)
        result = client.post("/v1/namespaces/demo/query", json={"query": "backups"}).json()
        assert result["fallback_used"] is True
        assert result["fallback_reason"] == "invalid_selection"
        assert result["hits"][0]["method"] == "lexical"
        assert client.post("/v1/namespaces/demo/query", json={"query": "zzzzxyz"}).json()[
            "abstained"
        ]


def test_markdown_hierarchy_fences_and_duplicate_headings():
    text = b"# A\nintro\n```python\n# not heading\n```\n### B\nbody\n## B\nsecond"
    sections = parse(text, "a.md")
    assert [s.title for s in sections] == ["Preamble", "A", "B", "B"]
    nodes = build_nodes(sections, "00000000-0000-0000-0000-000000000001", 1, "Doc")
    assert len({n.id for n in nodes}) == len(nodes)
    assert nodes[-1].parent_id == nodes[1].id
    assert "not heading" in nodes[1].content


def test_text_overflow_preserved():
    text = "a" * 14000
    nodes = build_nodes(
        parse(text.encode(), "a.txt"), "00000000-0000-0000-0000-000000000001", 1, "Doc"
    )
    assert "".join(n.content for n in nodes) == text
    assert all(len(n.content) <= 4000 for n in nodes)


def test_html_and_docx():
    sections = parse(b"<h1>Guide</h1><p>real content</p><script>bad()</script>", "a.html")
    assert "real content" in sections[-1].text
    assert "bad" not in sections[-1].text
    document = Docx()
    document.add_heading("Guide", 1)
    document.add_paragraph("Paragraph content")
    buffer = io.BytesIO()
    document.save(buffer)
    sections = parse(buffer.getvalue(), "a.docx")
    assert sections[-1].title == "Guide"
    assert "Paragraph content" in sections[-1].text


@pytest.mark.parametrize("filename", ["a.pdf", "a.docx", "a.exe"])
def test_malformed_uploads(client, filename):
    response = client.post(
        "/v1/namespaces/demo/documents",
        content=b"garbage",
        params={"filename": filename, "source_key": "x", "title": "x"},
    )
    assert response.status_code == 422


def test_transaction_builder_failure_and_concurrent_upserts():
    store = MemoryStore()

    def write(i):
        return store.upsert("a", "b", "same", "t", None, str(i), lambda *_: [])

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(write, range(20)))
    assert store.documents("a", "b")[0].revision == 20

    def broken(*args):
        raise ValueError("bad parse")

    with pytest.raises(ValueError):
        store.upsert("a", "b", "same", "t", None, "new", broken)
    assert store.documents("a", "b")[0].revision == 20


def test_limits_and_metrics():
    sections = parse(TEXT, "a.md")
    nodes = build_nodes(sections, "00000000-0000-0000-0000-000000000001", 1, "Guide")
    _, _, reason = navigate("Guide", nodes, TitleSelector(), max_depth=1)
    assert reason == "depth_limit"
    assert metrics(["a", "a", "b"], ["b"], 2)["mrr"] == 0.5
    assert metrics([], [], 3)["abstained"]
    report = run()
    assert len(report["methods"]) == 4
    assert report["queries"] == 12


def test_fail_closed_without_credentials():
    with pytest.raises(RuntimeError), TestClient(create_app(MemoryStore(), [])):
        pass


def test_openapi_contract_and_nul_rejection(client):
    schema = client.get("/openapi.json").json()
    upload_schema = schema["paths"]["/v1/namespaces/{namespace}/documents"]["post"]
    assert "application/octet-stream" in upload_schema["requestBody"]["content"]
    assert "QueryResponse" in schema["components"]["schemas"]
    assert upload(client, b"text\x00bad").status_code == 422


def test_valid_pdf_and_page_locator():
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(300, 300)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 200 Td (Synthetic PDF content) Tj ET")
    page[NameObject("/Contents")] = stream
    buffer = io.BytesIO()
    writer.write(buffer)
    sections = parse(buffer.getvalue(), "test.pdf")
    assert sections[0].locator == {"page": 1}
    assert "Synthetic PDF content" in sections[0].text


def test_model_error_fallback(client):
    class Failure:
        def select(self, *args):
            raise ValueError("malformed model JSON")

    client.app.state.selector = Failure()
    upload(client)
    result = client.post("/v1/namespaces/demo/query", json={"query": "backups"}).json()
    assert result["fallback_reason"] == "selector_error"
    assert result["hits"][0]["method"] == "lexical"
