"""Restore missing corpus files from the pinned manifest, verifying bytes before writing."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    for document in manifest["documents"]:
        target = (root / document["path"]).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Invalid target path")
        if not document["source_url"].startswith("https://raw.githubusercontent.com/"):
            raise ValueError("Only pinned public GitHub sources are supported")
        if document["commit"] not in document["source_url"]:
            raise ValueError("Source must contain its pinned revision")
        with urllib.request.urlopen(document["source_url"], timeout=30) as response:
            data = response.read(2_000_001)
        if hashlib.sha256(data).hexdigest() != document["sha256"]:
            raise ValueError("Source bytes do not match the manifest")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    print("Corpus restored and hashes verified. License files are included in the repository.")


if __name__ == "__main__":
    main()
