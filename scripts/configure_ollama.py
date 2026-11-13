"""Configure local Ollama or Ollama Cloud without printing or committing secrets."""

import argparse
import getpass
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--local", action="store_true", help="Use local Ollama without any API key")
    parser.add_argument(
        "--docker-client",
        action="store_true",
        help="Use host.docker.internal for a Docker API client",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.docker_client and not args.local:
        parser.error("--docker-client requires --local")
    model = (args.model or ("qwen3:4b-q4_K_M" if args.local else "gpt-oss:20b")).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,199}", model):
        parser.error("Supply a model name from the current Ollama Cloud catalog")
    key = "" if args.local else getpass.getpass("Ollama API key (hidden): ").strip()
    if not args.local and not re.fullmatch(r"[A-Za-z0-9_.:/+=-]{16,}", key):
        parser.error("Supply the API token without whitespace or quotes")
    path = root / ".env"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    settings = {
        "SELECTOR_PROVIDER": "ollama",
        "SELECTOR_URL": (
            "http://host.docker.internal:11434/api"
            if args.docker_client
            else "http://127.0.0.1:11434/api"
        )
        if args.local
        else "https://ollama.com/api",
        "SELECTOR_MODEL": model,
        "SELECTOR_API_KEY": key,
        "SELECTOR_THINK": "off" if args.local else "default",
        "SELECTOR_NUM_CTX": "8192" if args.local else "",
        "SELECTOR_MAX_OUTPUT_TOKENS": "256" if args.local else "1024",
    }
    lines = [line for line in existing.splitlines() if line.split("=", 1)[0] not in settings]
    lines.extend(f"{name}={value}" for name, value in settings.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o600)
    print("Local Ollama configuration saved. No key value was printed. Restart the API to apply.")


if __name__ == "__main__":
    main()
