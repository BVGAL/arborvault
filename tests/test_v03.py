import io
import json
import shutil
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from arborvault.api import create_app
from arborvault.domain import Node
from arborvault.isolated_parser import IsolatedParser
from arborvault.memory import MemoryStore
from arborvault.ollama import OllamaSelector, decode_selection
from arborvault.parsing import Section, build_nodes, parse
from arborvault.retrieval import TitleSelector, retrieve
from arborvault.settings import ParsingLimits


def test_document_acl_all_surfaces_and_credential_rotation():
    keys = [
        {
            "id": name,
            "subject": subject,
            "key": "synthetic-v03-credential-" + name,
            "tenant": "test",
            "namespaces": ["docs"],
            "permissions": ["read", "write", "delete"] + extra,
        }
        for name, subject, extra in [
            ("owner", "owner", []),
            ("reader", "reader", []),
            ("rotated", "reader", []),
            ("admin", "admin", ["manage_acl"]),
        ]
    ]
    store = MemoryStore()
    with TestClient(create_app(store, keys)) as client:

        def actor(name):
            client.headers["Authorization"] = "Bearer synthetic-v03-credential-" + name

        def upload(text, **params):
            return client.post(
                "/v1/namespaces/docs/documents",
                content=text,
                params={
                    "filename": "note.txt",
                    "source_key": "private",
                    "title": "Hidden title",
                    "visibility": "private",
                    **params,
                },
            )

        actor("owner")
        document = upload(b"classified evidence version one").json()["document"]
        upload(b"classified evidence version two")
        prefix = "/v1/namespaces/docs/documents/" + document["id"]
        actor("reader")
        assert client.get("/v1/namespaces/docs/documents").json()["documents"] == []
        for path in ("/tree", "/tree?revision=1", "/revisions", "/acl"):
            assert client.get(prefix + path).status_code == 404
        for mode in ("auto", "navigation", "lexical"):
            data = client.post(
                "/v1/namespaces/docs/query", json={"query": "classified", "mode": mode}
            ).json()
            assert data["hits"] == [] and data["trace"] == []
        assert upload(b"overwrite").status_code == 404
        assert client.delete(prefix).status_code == 404
        assert (
            client.put(prefix + "/acl", json={"read": ["*"], "write": [], "delete": []}).status_code
            == 404
        )
        actor("owner")
        assert (
            client.put(
                prefix + "/acl", json={"read": ["reader"], "write": [], "delete": []}
            ).status_code
            == 204
        )
        for name in ("reader", "rotated"):
            actor(name)
            assert client.get(prefix + "/tree?revision=1").status_code == 200
            assert upload(b"forbidden").status_code == 404
        actor("admin")
        # Managing ACLs is distinct from reading private contents.
        assert client.get(prefix + "/tree").status_code == 404
        assert (
            client.put(
                prefix + "/acl", json={"read": [], "write": ["reader"], "delete": []}
            ).status_code
            == 204
        )
        actor("reader")
        assert client.get(prefix + "/tree").status_code == 404
        assert upload(b"writer can create next revision", visibility="namespace").status_code == 200
        assert client.get(prefix + "/tree").status_code == 404  # update does not reset ACL
        actor("owner")
        assert client.get(prefix + "/acl").json()["owner"] == "owner"
        assert client.delete(prefix).status_code == 204


def test_navigation_beyond_ten_thousand_nodes_is_lazy():
    store = MemoryStore()
    sections = [
        Section(1, "Topic " + str(i), "quartzunique" if i == 11000 else "ordinary", {})
        for i in range(11001)
    ]
    doc, _ = store.upsert(
        "t", "n", "large", "Large", None, "v1", lambda i, r: build_nodes(sections, i, r, "Large")
    )
    assert len(store.data["t", "n", doc.id][0][1]) > 10000

    class Guard:
        def select(self, query, candidates, width):
            assert len(candidates) <= 160
            return TitleSelector().select(query, candidates, width)

    hits, trace, fallback, reason = retrieve(
        store, "t", "n", "Large quartzunique", 1, Guard(), "navigation"
    )
    assert hits[0][1].content == "quartzunique"
    assert any(t["shortlisted"] for t in trace)
    assert not fallback and reason != "corpus_limit"
    with pytest.raises(ValueError):
        build_nodes(
            sections, str(uuid4()), 1, "Large", limits=replace(ParsingLimits(), max_nodes=2000)
        )


def test_tree_pagination_preserves_every_node():
    key = {
        "id": "test",
        "key": "synthetic-v03-tree-pagination",
        "tenant": "test",
        "namespaces": ["docs"],
        "permissions": ["read", "write"],
    }
    with TestClient(create_app(MemoryStore(), [key])) as client:
        client.headers["Authorization"] = "Bearer " + key["key"]
        result = client.post(
            "/v1/namespaces/docs/documents",
            content=b"# Root\n## A\none\n## B\ntwo\n## C\nthree",
            params={"filename": "x.md", "source_key": "x", "title": "X"},
        ).json()
        prefix = "/v1/namespaces/docs/documents/" + result["document"]["id"] + "/tree"
        first = client.get(prefix, params={"limit": 2}).json()
        second = client.get(prefix, params={"limit": 2, "cursor": first["next_cursor"]}).json()
        assert len({n["id"] for n in first["nodes"] + second["nodes"]}) == 4
        third = client.get(prefix, params={"limit": 2, "cursor": second["next_cursor"]}).json()
        assert len({n["id"] for n in first["nodes"] + second["nodes"] + third["nodes"]}) == 5
        assert third["next_cursor"] is None


def test_real_ocr_and_more_than_200_pdf_pages():
    pytest.importorskip("pypdfium2")
    if not shutil.which("tesseract"):
        pytest.skip("Install Tesseract for the OCR integration test")
    from PIL import Image, ImageDraw, ImageFont
    from pypdf import PdfReader, PdfWriter

    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = (
        ImageFont.truetype(str(font_path), 44)
        if font_path.exists()
        else ImageFont.load_default(size=44)
    )
    image = Image.new("RGB", (1400, 350), "white")
    ImageDraw.Draw(image).text(
        (50, 80), "Backups expire after thirty days.", fill="black", font=font
    )
    buffer = io.BytesIO()
    image.save(buffer, "PDF", resolution=150)
    scanned = buffer.getvalue()
    sections = IsolatedParser().parse(scanned, "scanned.pdf", "auto")
    assert "thirty days" in sections[0].text.lower()
    assert sections[0].locator == {"page": 1, "ocr": True}
    with pytest.raises(ValueError):
        parse(scanned, "scanned.pdf", ocr_mode="off")
    writer = PdfWriter()
    for _ in range(200):
        writer.add_blank_page(100, 100)
    writer.add_page(PdfReader(io.BytesIO(scanned)).pages[0])
    large = io.BytesIO()
    writer.write(large)
    # Native text on the final page avoids running OCR on 200 intentionally blank pages.
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    page = writer.pages[-1]
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
    )
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 10 50 Td (Evidence on page 201) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    large = io.BytesIO()
    writer.write(large)
    result = parse(large.getvalue(), "large.pdf", ocr_mode="off")
    assert result[-1].locator == {"page": 201}
    assert "201" in result[-1].text
    with pytest.raises(ValueError):
        parse(large.getvalue(), "large.pdf", limits=replace(ParsingLimits(), max_pdf_pages=200))


def test_ollama_protocol_validation_usage_and_budget():
    captured = []

    def respond(request):
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "done": True,
                "done_reason": "stop",
                "message": {"content": '["0"]'},
                "prompt_eval_count": 50,
                "eval_count": 5,
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        selector = OllamaSelector(
            "https://ollama.com/api",
            "gpt-oss:20b",
            "synthetic-test-only",
            max_calls=1,
            client=client,
        )
        node = Node("persistent-id", None, "Backups", "Evidence", ["Backups"], 0)
        assert selector.select("backups", [node], 4) == [node.id]
        assert "persistent-id" not in json.dumps(captured)
        assert "format" not in captured[0] and captured[0]["stream"] is False
        with pytest.raises(ValueError, match="budget"):
            selector.select("backups", [node], 4)
        assert selector.usage() == {
            "calls": 1,
            "failures": 0,
            "prompt_tokens": 50,
            "output_tokens": 5,
        }


@pytest.mark.parametrize("content", ['["999"]', '["0","0"]', "{}", "[0]", "```json\n[]\n```"])
def test_model_cannot_inject_unknown_or_malformed_selections(content):
    with pytest.raises(ValueError):
        decode_selection(content, {"0": "known"}, 4)


@pytest.mark.parametrize("body", [{"done": False}, {"done": True, "done_reason": "length"}])
def test_truncated_provider_results_are_failures(body):
    body["message"] = {"content": "[]"}
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))
    ) as client:
        selector = OllamaSelector("https://ollama.com/api", "model", client=client)
        with pytest.raises(ValueError):
            selector.select("query", [], 4)
        assert selector.usage()["failures"] == 1


@pytest.mark.parametrize("body", [[], None, {"done": True, "message": []}])
def test_invalid_provider_envelope_fails_closed(body):
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=json.dumps(body)))
    ) as client:
        selector = OllamaSelector("https://ollama.com/api", "model", client=client)
        with pytest.raises(ValueError):
            selector.select("query", [], 4)
        assert selector.usage()["failures"] == 1
