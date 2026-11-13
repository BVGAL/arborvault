import hashlib
import hmac
import json
import os
import re
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Literal
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from .domain import Document, Node
from .parsing import MAX_BYTES, build_nodes, parse
from .retrieval import ChatSelector, TitleSelector, retrieve
from .storage import PostgresStore


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)
    mode: Literal["auto", "navigation", "lexical"] = "auto"


class IngestResponse(BaseModel):
    document: Document
    changed: bool


class DocumentsResponse(BaseModel):
    documents: list[Document]


class TreeResponse(BaseModel):
    document: Document
    nodes: list[Node]


class Citation(BaseModel):
    source_key: str
    source_uri: str | None
    locator: dict
    digest: str


class Hit(BaseModel):
    document_id: str
    revision: int
    node_id: str
    title: str
    path: list[str]
    text: str
    score: float
    method: Literal["navigation", "lexical"]
    citation: Citation


class QueryResponse(BaseModel):
    hits: list[Hit]
    trace: list[dict]
    fallback_used: bool
    fallback_reason: str | None
    abstained: bool


def create_app(store=None, keys=None, selector=None):
    @asynccontextmanager
    async def lifespan(app):
        configured = (
            keys if keys is not None else json.loads(os.environ.get("ARBORVAULT_KEYS", "[]"))
        )
        if not configured or any(
            len(k.get("key", "")) < 24 or not k.get("tenant") or not k.get("namespaces")
            for k in configured
        ):
            raise RuntimeError("Configure tenant credentials with keys of at least 24 characters")
        if len({k["key"] for k in configured}) != len(configured):
            raise RuntimeError("Keys must be unique")
        app.state.keys = configured
        app.state.store = store or PostgresStore(os.environ["DATABASE_URL"])
        app.state.selector = selector or TitleSelector()
        if selector is None and os.getenv("SELECTOR_URL"):
            app.state.selector = ChatSelector(
                os.environ["SELECTOR_URL"],
                os.environ["SELECTOR_MODEL"],
                os.environ.get("SELECTOR_API_KEY", ""),
            )
        yield
        if store is None:
            app.state.store.close()

    app = FastAPI(
        title="ArborVault",
        version="0.1.0",
        lifespan=lifespan,
        description="Independent structure-aware retrieval. Not affiliated with IBM.",
    )

    # Limit the full body before JSON or file parsing, including chunked requests.
    @app.middleware("http")
    async def body_limit(request, call_next):
        from starlette.responses import JSONResponse

        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_BYTES:
                return JSONResponse({"detail": "Request body too large"}, status_code=413)
            chunks.append(chunk)
        request._body = b"".join(chunks)
        return await call_next(request)

    def scope(namespace: str, request: Request, authorization: str = Header(default="")):
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", namespace):
            raise HTTPException(422, "Invalid namespace")
        supplied = (
            authorization.removeprefix("Bearer ") if authorization.startswith("Bearer ") else ""
        )
        for credential in request.app.state.keys:
            if hmac.compare_digest(supplied.encode(), credential["key"].encode()):
                if (
                    namespace not in credential["namespaces"]
                    and "*" not in credential["namespaces"]
                ):
                    raise HTTPException(403, "Namespace forbidden")
                return credential["tenant"], namespace
        raise HTTPException(
            401, "Invalid bearer credential", headers={"WWW-Authenticate": "Bearer"}
        )

    @app.get("/healthz")
    def health():
        return {"status": "ok"}

    @app.get("/readyz")
    def ready(request: Request):
        try:
            request.app.state.store.ping()
        except Exception:
            raise HTTPException(503, "Storage unavailable") from None
        return {"status": "ready"}

    @app.post(
        "/v1/namespaces/{namespace}/documents",
        response_model=IngestResponse,
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/octet-stream": {"schema": {"type": "string", "format": "binary"}}
                },
            }
        },
    )
    async def ingest(
        request: Request,
        filename: str = Query(max_length=255),
        source_key: str = Query(min_length=1, max_length=300),
        title: str = Query(min_length=1, max_length=500),
        source_uri: str | None = Query(default=None, max_length=2000),
        context=Depends(scope),
    ):
        data = await request.body()
        if source_uri and not source_uri.startswith(("https://", "http://", "urn:")):
            raise HTTPException(422, "Source URI must use https, http or urn")
        try:
            sections = await run_in_threadpool(parse, data, filename)
            digest = hashlib.sha256(
                data + json.dumps([filename, title, source_uri]).encode()
            ).hexdigest()
            doc, changed = await run_in_threadpool(
                request.app.state.store.upsert,
                *context,
                source_key,
                title,
                source_uri,
                digest,
                lambda i, v: build_nodes(sections, i, v, title),
            )
        except (ValueError, UnicodeError):
            raise HTTPException(
                422, "Unsupported, empty, malformed or oversized document"
            ) from None
        except zip_errors():
            raise HTTPException(422, "Malformed document") from None
        return {"document": asdict(doc), "changed": changed}

    @app.get("/v1/namespaces/{namespace}/documents", response_model=DocumentsResponse)
    def documents(request: Request, context=Depends(scope)):
        return {"documents": [asdict(d) for d in request.app.state.store.documents(*context)]}

    @app.get("/v1/namespaces/{namespace}/documents/{document_id}/tree", response_model=TreeResponse)
    def tree(document_id: UUID, request: Request, context=Depends(scope)):
        rows = request.app.state.store.nodes(*context, str(document_id))
        if not rows:
            raise HTTPException(404, "Document not found")
        return {"document": asdict(rows[0][0]), "nodes": [n.as_dict() for _, n in rows]}

    @app.delete("/v1/namespaces/{namespace}/documents/{document_id}", status_code=204)
    def delete(document_id: UUID, request: Request, context=Depends(scope)):
        if not request.app.state.store.delete(*context, str(document_id)):
            raise HTTPException(404, "Document not found")

    @app.post("/v1/namespaces/{namespace}/query", response_model=QueryResponse)
    def query(body: QueryRequest, request: Request, context=Depends(scope)):
        hits, trace, fallback, reason = retrieve(
            request.app.state.store,
            *context,
            body.query,
            body.k,
            request.app.state.selector,
            body.mode,
        )
        return {
            "hits": [
                {
                    "document_id": d.id,
                    "revision": d.revision,
                    "node_id": n.id,
                    "title": n.title,
                    "path": n.path,
                    "text": n.content,
                    "score": score,
                    "method": method,
                    "citation": {
                        "source_key": d.source_key,
                        "source_uri": d.source_uri,
                        "locator": n.locator,
                        "digest": d.digest,
                    },
                }
                for d, n, score, method in hits
            ],
            "trace": trace,
            "fallback_used": fallback,
            "fallback_reason": reason,
            "abstained": not hits,
        }

    return app


def zip_errors():
    from zipfile import BadZipFile

    from lxml.etree import XMLSyntaxError
    from pypdf.errors import PdfReadError

    return (BadZipFile, PdfReadError, XMLSyntaxError, KeyError)


app = create_app()
