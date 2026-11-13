"""Generate local-only credentials without printing or overwriting existing secrets."""

import json
import secrets
from pathlib import Path


def main():
    path = Path(__file__).resolve().parents[1] / ".env"
    credentials = [{"key": secrets.token_urlsafe(32), "tenant": "demo", "namespaces": ["demo"]}]
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(f"POSTGRES_PASSWORD={secrets.token_hex(24)}\n")
            stream.write("ARBORVAULT_KEYS='" + json.dumps(credentials) + "'\n")
        path.chmod(0o600)
    except FileExistsError:
        raise SystemExit(".env already exists; left unchanged") from None
    print("Created local .env. Keep it private. Run docker compose up --build -d.")


if __name__ == "__main__":
    main()
