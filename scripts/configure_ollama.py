"""Configure Ollama Cloud locally without printing or committing its API key."""

import argparse
import getpass
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-oss:20b")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    model = args.model.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,199}", model):
        parser.error("Supply a model name from the current Ollama Cloud catalog")
    key = getpass.getpass("Ollama API key (hidden): ").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.:/+=-]{16,}", key):
        parser.error("Supply the API token without whitespace or quotes")
    path = root / ".env"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    settings = {
        "SELECTOR_PROVIDER": "ollama",
        "SELECTOR_URL": "https://ollama.com/api",
        "SELECTOR_MODEL": model,
        "SELECTOR_API_KEY": key,
    }
    lines = [line for line in existing.splitlines() if line.split("=", 1)[0] not in settings]
    lines.extend(f"{name}={value}" for name, value in settings.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o600)
    print("Local Ollama configuration saved. No key value was printed. Restart the API to apply.")


if __name__ == "__main__":
    main()
