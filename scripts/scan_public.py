"""Scan tracked files (or an explicit source allowlist) without printing secret values."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "github_token": re.compile(r"(?:github_pat_|gh[pousr]_)[A-Za-z0-9_]{20,}"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "aws_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "provider_key": re.compile(r"sk-[A-Za-z0-9_-]{30,}"),
    "personal_windows_path": re.compile(r"[A-Z]:[\\/]Users[\\/][^\s]+"),
}


def main():
    tracked = (
        subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
        .stdout.decode()
        .split("\0")
    )
    if not any(tracked):
        raise SystemExit("No tracked files: stage the intended publication first")
    findings = []
    for name in filter(None, tracked):
        path = Path(name)
        if path.name.startswith(".env") and path.name != ".env.example":
            findings.append((name, "environment_file"))
        if path.suffix in {".pem", ".key", ".p12", ".db", ".dump"}:
            findings.append((name, "private_material"))
        if ".secrets" in path.parts or path.name.startswith(".arborvault-"):
            findings.append((name, "credential_material"))
        text = (ROOT / path).read_text(encoding="utf-8")
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                findings.append((name, label))
    if findings:
        for name, label in findings:
            print(f"BLOCKED {name}: {label}")
        raise SystemExit(1)
    print(f"Public-content scan passed: {len(list(filter(None, tracked)))} tracked text files")
    print("Heuristic scan only; manual provenance review is also required.")


if __name__ == "__main__":
    main()
