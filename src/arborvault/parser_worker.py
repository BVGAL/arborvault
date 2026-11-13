"""One disposable parser process: JSON in/out, no database/provider credentials."""

import base64
import json
import subprocess
import sys
from dataclasses import asdict

from .settings import ParsingLimits


def main():
    limits = ParsingLimits.from_env()
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds))
        memory = limits.memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        file_size = max(limits.max_bytes * 2, limits.ocr_max_pixels * 4)
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_size, file_size))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    except ImportError:
        pass  # Windows: wall timeout still applies; use Compose for resource limits.
    from .parsing import parse

    try:
        payload = json.loads(sys.stdin.buffer.read(limits.max_bytes * 2 + 4096))
        sections = parse(
            base64.b64decode(payload["data"], validate=True),
            payload["filename"],
            limits=limits,
            ocr_mode=payload.get("ocr_mode"),
        )
        output = {"sections": [asdict(s) for s in sections]}
    except subprocess.TimeoutExpired:
        output = {"error": "OCR page deadline exceeded", "timeout": True}
    except Exception:
        output = {"error": "Unsupported, malformed or resource-intensive document"}
    sys.stdout.write(json.dumps(output))


if __name__ == "__main__":
    main()
