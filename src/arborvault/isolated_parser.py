import base64
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import httpx

from .parsing import MAX_BYTES, Section


class ParserBusy(Exception):
    pass


class ParserTimeout(Exception):
    pass


class IsolatedParser:
    def __init__(self, timeout=10, concurrency=2):
        self.timeout = timeout
        self.slots = threading.BoundedSemaphore(concurrency)

    def parse(self, data, filename):
        if not data or len(data) > MAX_BYTES:
            raise ValueError("Invalid document size")
        if not self.slots.acquire(blocking=False):
            raise ParserBusy()
        try:
            payload = json.dumps(
                {"data": base64.b64encode(data).decode(), "filename": filename}
            ).encode()
            environment = {
                k: os.environ[k] for k in ("PATH", "SYSTEMROOT", "WINDIR") if k in os.environ
            }
            environment["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            with tempfile.TemporaryDirectory(prefix="arborvault-parse-") as directory:
                result = subprocess.run(
                    [sys.executable, "-m", "arborvault.parser_worker"],
                    input=payload,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=self.timeout,
                    cwd=directory,
                    env=environment,
                    check=False,
                )
            if result.returncode != 0 or len(result.stdout) > 8_000_000:
                raise ValueError("Parser resource limit or failure")
            decoded = json.loads(result.stdout)
            if "error" in decoded:
                raise ValueError(decoded["error"])
            return [Section(**s) for s in decoded["sections"]]
        except subprocess.TimeoutExpired:
            raise ParserTimeout() from None
        finally:
            self.slots.release()


class RemoteParser:
    def __init__(self, url):
        self.url = url.rstrip("/")

    def parse(self, data, filename):
        try:
            response = httpx.post(
                self.url + "/parse",
                params={"filename": filename},
                content=data,
                headers={"Content-Type": "application/octet-stream"},
                timeout=15,
            )
            if response.status_code == 503:
                raise ParserBusy()
            if response.status_code == 504:
                raise ParserTimeout()
            if response.status_code == 422:
                raise ValueError("Malformed or unsupported document")
            response.raise_for_status()
            return [Section(**s) for s in response.json()["sections"]]
        except httpx.TimeoutException:
            raise ParserTimeout() from None
        except httpx.HTTPError:
            raise ParserBusy() from None
