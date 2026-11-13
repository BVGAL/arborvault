import base64
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import httpx

from .parsing import Section
from .settings import ParsingLimits


class ParserBusy(Exception):
    pass


class ParserTimeout(Exception):
    pass


def run_worker(command, *, input, timeout, cwd, env):
    with subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=cwd,
        env=env,
        start_new_session=os.name != "nt",
    ) as process:
        try:
            output, _ = process.communicate(input, timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True
                )
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.kill()
            process.communicate()
            raise
        return subprocess.CompletedProcess(command, process.returncode, stdout=output)


class IsolatedParser:
    def __init__(self, timeout=None, concurrency=2, limits=None):
        self.limits = limits or ParsingLimits.from_env()
        self.timeout = timeout or self.limits.wall_seconds
        self.slots = threading.BoundedSemaphore(concurrency)

    def parse(self, data, filename, ocr_mode=None):
        if not data or len(data) > self.limits.max_bytes:
            raise ValueError("Invalid document size")
        if not self.slots.acquire(blocking=False):
            raise ParserBusy()
        try:
            payload = json.dumps(
                {
                    "data": base64.b64encode(data).decode(),
                    "filename": filename,
                    "ocr_mode": ocr_mode,
                }
            ).encode()
            environment = {
                k: os.environ[k] for k in ("PATH", "SYSTEMROOT", "WINDIR") if k in os.environ
            }
            environment["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            environment.update(self.limits.environment())
            environment["OCR_MODE"] = os.getenv("OCR_MODE", "auto")
            environment["OCR_LANGUAGES"] = os.getenv("OCR_LANGUAGES", "eng+ita")
            environment["OMP_THREAD_LIMIT"] = "1"
            with tempfile.TemporaryDirectory(prefix="arborvault-parse-") as directory:
                result = run_worker(
                    [sys.executable, "-m", "arborvault.parser_worker"],
                    input=payload,
                    timeout=self.timeout,
                    cwd=directory,
                    env=environment,
                )
            if result.returncode != 0 or len(result.stdout) > self.limits.output_bytes:
                raise ValueError("Parser resource limit or failure")
            decoded = json.loads(result.stdout)
            if decoded.get("timeout"):
                raise ParserTimeout()
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

    def parse(self, data, filename, ocr_mode=None):
        try:
            response = httpx.post(
                self.url + "/parse",
                params={"filename": filename, **({"ocr_mode": ocr_mode} if ocr_mode else {})},
                content=data,
                headers={"Content-Type": "application/octet-stream"},
                timeout=ParsingLimits.from_env().wall_seconds + 15,
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
