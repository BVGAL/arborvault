import hashlib
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Literal
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from psycopg.errors import UniqueViolation
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool

from . import __version__
from .access import AuthorizedStore, Principal, Scope
from .auth import Credentials
from .domain import Document, Node
from .isolated_parser import IsolatedParser, ParserBusy, ParserTimeout, RemoteParser
from .limits import RateLimiter
from .ollama import OllamaSelector, thinking_value
from .parsing import MAX_BYTES, build_nodes
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
    next_cursor: str | None = None


class RevisionsResponse(BaseModel):
    revisions: list[Document]
    next_cursor: int | None = None


class TreeResponse(BaseModel):
    document: Document
    nodes: list[Node]
    next_cursor: int | None = None


class DocumentACL(BaseModel):
    model_config = ConfigDict(extra="forbid")
    read: list[str] = Field(max_length=100)
    write: list[str] = Field(max_length=100)
    delete: list[str] = Field(max_length=100)

    @field_validator("read", "write", "delete")
    @classmethod
    def valid_members(cls, values):
        if any(v != "*" and not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", v) for v in values):
            raise ValueError("ACL members must be credential subjects or *")
        return list(dict.fromkeys(values))


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


def create_app(store=None, keys=None, selector=None, parser=None, keys_file=None, limiter=None):
    @asynccontextmanager
    async def lifespan(app):
        configured = (
            keys if keys is not None else json.loads(os.environ.get("ARBORVAULT_KEYS", "[]"))
        )
        try:
            app.state.credentials = Credentials(
                configured, keys_file or os.getenv("ARBORVAULT_KEYS_FILE")
            )
        except (ValueError, OSError, TypeError):
            raise RuntimeError(
                "Invalid credential configuration; explicit permissions required"
            ) from None
        app.state.store = store or PostgresStore(
            os.environ["DATABASE_URL"], enforce_runtime_role=True
        )
        app.state.parser = parser or (
            RemoteParser(os.environ["PARSER_URL"]) if os.getenv("PARSER_URL") else IsolatedParser()
        )
        app.state.limiter = limiter or RateLimiter(
            int(os.getenv("RATE_PER_MINUTE", "60")), int(os.getenv("RATE_BURST", "60"))
        )
        app.state.selector = selector or TitleSelector()
        if selector is None and os.getenv("SELECTOR_URL"):
            provider = os.getenv("SELECTOR_PROVIDER", "openai-compatible")
            if provider not in {"ollama", "openai-compatible"}:
                raise RuntimeError("Unsupported selector provider")
            adapter = OllamaSelector if provider == "ollama" else ChatSelector
            options = (
                {
                    "max_calls": int(os.getenv("SELECTOR_MAX_CALLS", "1000")),
                    "max_output_tokens": int(os.getenv("SELECTOR_MAX_OUTPUT_TOKENS", "1024")),
                    "think": thinking_value(os.getenv("SELECTOR_THINK", "default")),
                    "num_ctx": int(os.environ["SELECTOR_NUM_CTX"])
                    if os.getenv("SELECTOR_NUM_CTX")
                    else None,
                }
                if provider == "ollama"
                else {}
            )
            app.state.selector = adapter(
                os.environ["SELECTOR_URL"],
                os.environ["SELECTOR_MODEL"],
                os.environ.get("SELECTOR_API_KEY", ""),
                **options,
            )
        yield
        if store is None:
            app.state.store.close()

    app = FastAPI(
        title="ArborVault",
        version=__version__,
        lifespan=lifespan,
        description="Independent structure-aware retrieval. Not affiliated with IBM.",
    )

    # Limit the full body before JSON or file parsing, including chunked requests.
    @app.middleware("http")
    async def body_limit(request, call_next):
        from starlette.responses import JSONResponse

        started = time.perf_counter()
        request_id = str(uuid4())
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_BYTES:
                return JSONResponse({"detail": "Request body too large"}, status_code=413)
            chunks.append(chunk)
        request._body = b"".join(chunks)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        route = request.scope.get("route")
        logging.getLogger("uvicorn.error.arborvault.requests").info(
            json.dumps(
                {
                    "request_id": request_id,
                    "method": request.method,
                    "route": getattr(route, "path", "unmatched"),
                    "status": response.status_code,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                }
            )
        )
        return response

    def scope(namespace: str, request: Request, authorization: str = Header(default="")):
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", namespace):
            raise HTTPException(422, "Invalid namespace")
        supplied = (
            authorization.removeprefix("Bearer ") if authorization.startswith("Bearer ") else ""
        )
        try:
            credential = request.app.state.credentials.authenticate(supplied)
        except (ValueError, OSError, TypeError):
            raise HTTPException(503, "Credential configuration unavailable") from None
        if credential is None:
            raise HTTPException(
                401, "Invalid bearer credential", headers={"WWW-Authenticate": "Bearer"}
            )
        if namespace not in credential["namespaces"] and "*" not in credential["namespaces"]:
            raise HTTPException(403, "Namespace forbidden")
        permission = (
            "delete"
            if request.method == "DELETE"
            else (
                "write"
                if request.method == "POST" and request.url.path.endswith("/documents")
                else "read"
            )
        )
        if request.url.path.endswith("/acl"):
            permission = "write"
        if permission not in credential["permissions"] and not (
            request.url.path.endswith("/acl") and "manage_acl" in credential["permissions"]
        ):
            raise HTTPException(403, f"Missing {permission} permission")
        if not request.app.state.limiter.allow(credential["key_sha256"]):
            raise HTTPException(429, "Request quota exceeded", headers={"Retry-After": "1"})
        principal = Principal(credential["subject"], "manage_acl" in credential["permissions"])
        return Scope(
            credential["tenant"], namespace, AuthorizedStore(request.app.state.store, principal)
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
        visibility: Literal["namespace", "private"] = "namespace",
        ocr_mode: Literal["off", "auto", "always"] | None = None,
        context=Depends(scope),
    ):
        data = await request.body()
        if any("\x00" in value for value in (filename, source_key, title, source_uri or "")):
            raise HTTPException(422, "Invalid metadata")
        if source_uri and not source_uri.startswith(("https://", "http://", "urn:")):
            raise HTTPException(422, "Source URI must use https, http or urn")
        try:
            effective_ocr = ocr_mode or os.getenv("OCR_MODE", "auto")
            sections = await run_in_threadpool(
                request.app.state.parser.parse, data, filename, effective_ocr
            )
            digest = hashlib.sha256(
                data
                + json.dumps(
                    [
                        filename,
                        title,
                        source_uri,
                        "parser-v3",
                        effective_ocr,
                        os.getenv("OCR_LANGUAGES", "eng+ita"),
                    ]
                ).encode()
            ).hexdigest()
            doc, changed = await run_in_threadpool(
                context.store.upsert,
                *context,
                source_key,
                title,
                source_uri,
                digest,
                lambda i, v: build_nodes(sections, i, v, title),
                visibility=visibility,
            )
        except (PermissionError, UniqueViolation):
            raise HTTPException(404, "Document unavailable") from None
        except (ValueError, UnicodeError):
            raise HTTPException(
                422, "Unsupported, empty, malformed or oversized document"
            ) from None
        except ParserBusy:
            raise HTTPException(
                503, "Parser unavailable or busy", headers={"Retry-After": "1"}
            ) from None
        except ParserTimeout:
            raise HTTPException(504, "Parser deadline exceeded") from None
        return {"document": asdict(doc), "changed": changed}

    @app.get("/v1/namespaces/{namespace}/documents", response_model=DocumentsResponse)
    def documents(
        request: Request,
        limit: int = Query(default=100, ge=1, le=1000),
        cursor: UUID | None = None,
        context=Depends(scope),
    ):
        rows = context.store.documents(*context, limit + 1, str(cursor) if cursor else None)
        return {
            "documents": [asdict(d) for d in rows[:limit]],
            "next_cursor": rows[limit - 1].id if len(rows) > limit else None,
        }

    @app.get(
        "/v1/namespaces/{namespace}/documents/{document_id}/revisions",
        response_model=RevisionsResponse,
    )
    def revisions(
        document_id: UUID,
        request: Request,
        limit: int = Query(default=100, ge=1, le=1000),
        cursor: int | None = Query(default=None, ge=1),
        context=Depends(scope),
    ):
        rows = context.store.revisions(*context, str(document_id), limit + 1, cursor)
        if not rows and cursor is None:
            raise HTTPException(404, "Document not found")
        return {
            "revisions": [asdict(d) for d in rows[:limit]],
            "next_cursor": rows[limit - 1].revision if len(rows) > limit else None,
        }

    @app.get("/v1/namespaces/{namespace}/documents/{document_id}/tree", response_model=TreeResponse)
    def tree(
        document_id: UUID,
        request: Request,
        revision: int | None = Query(default=None, ge=1),
        limit: int = Query(default=500, ge=1, le=2000),
        cursor: int = Query(default=-1, ge=-1),
        context=Depends(scope),
    ):
        rows = context.store.nodes(*context, str(document_id), revision, limit + 1, cursor)
        if not rows:
            raise HTTPException(404, "Document not found")
        return {
            "document": asdict(rows[0][0]),
            "nodes": [n.as_dict() for _, n in rows[:limit]],
            "next_cursor": rows[limit - 1][1].ordinal if len(rows) > limit else None,
        }

    @app.get("/v1/namespaces/{namespace}/documents/{document_id}/acl")
    def get_acl(document_id: UUID, context=Depends(scope)):
        acl = context.store.get_acl(*context, str(document_id))
        if acl is None:
            raise HTTPException(404, "Document unavailable")
        return acl

    @app.put("/v1/namespaces/{namespace}/documents/{document_id}/acl", status_code=204)
    def put_acl(document_id: UUID, body: DocumentACL, context=Depends(scope)):
        if not context.store.set_acl(*context, str(document_id), body.model_dump()):
            raise HTTPException(404, "Document unavailable")

    @app.delete("/v1/namespaces/{namespace}/documents/{document_id}", status_code=204)
    def delete(document_id: UUID, request: Request, context=Depends(scope)):
        if not context.store.delete(*context, str(document_id)):
            raise HTTPException(404, "Document not found")

    @app.post("/v1/namespaces/{namespace}/query", response_model=QueryResponse)
    def query(body: QueryRequest, request: Request, context=Depends(scope)):
        hits, trace, fallback, reason = retrieve(
            context.store,
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


app = create_app()
