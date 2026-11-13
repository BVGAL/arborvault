"""One disposable parser process: JSON in/out, no database/provider credentials."""

import base64
import json
import sys
from dataclasses import asdict


def main():
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
        resource.setrlimit(resource.RLIMIT_AS, (384 * 1024 * 1024, 384 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024 * 1024, 8 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    except ImportError:
        pass  # Windows: wall timeout still applies; use Compose for resource limits.
    from .parsing import parse

    try:
        payload = json.loads(sys.stdin.buffer.read(2_800_000))
        sections = parse(base64.b64decode(payload["data"], validate=True), payload["filename"])
        output = {"sections": [asdict(s) for s in sections]}
    except Exception:
        output = {"error": "Unsupported, malformed or resource-intensive document"}
    sys.stdout.write(json.dumps(output))


if __name__ == "__main__":
    main()
