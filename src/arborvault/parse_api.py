"""Internal-only parser service. Compose provides no egress or credentials."""

from dataclasses import asdict
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

from .isolated_parser import IsolatedParser, ParserBusy, ParserTimeout
from .parsing import MAX_BYTES

app = FastAPI(title="ArborVault internal parser", docs_url=None, openapi_url=None)
parser = IsolatedParser()


@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.post("/parse")
async def parse(
    request: Request,
    filename: str = Query(max_length=255),
    ocr_mode: Literal["off", "auto", "always"] | None = None,
):
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "Document too large")
    try:
        sections = await run_in_threadpool(parser.parse, bytes(data), filename, ocr_mode)
    except ValueError:
        raise HTTPException(422, "Malformed or unsupported document") from None
    except ParserBusy:
        raise HTTPException(503, "Parser busy", headers={"Retry-After": "1"}) from None
    except ParserTimeout:
        raise HTTPException(504, "Parser deadline exceeded") from None
    return {"sections": [asdict(s) for s in sections]}
