"""Build the document-disjoint, human-labeled context support benchmark."""

import argparse
import json
from pathlib import Path

from arborvault.support_data import build, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/support-v1.json"))
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument(
        "--download", action="store_true", help="Download hash-locked public sources"
    )
    parser.add_argument("--output", type=Path, default=Path("datasets/support-v1"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.download:
        import httpx

        args.sources.mkdir(parents=True, exist_ok=True)
        with httpx.Client(timeout=120, follow_redirects=True) as client:
            for name, spec in config["sources"].items():
                path = args.sources / name
                if path.exists() and file_hash(path) == spec["sha256"]:
                    continue
                response = client.get(spec["url"])
                response.raise_for_status()
                path.write_bytes(response.content)
                if file_hash(path) != spec["sha256"]:
                    raise ValueError("Downloaded source hash mismatch: " + name)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
    result = build(config, args.sources, args.output, tokenizer, Path("datasets"))
    print(json.dumps(result["counts"]))


if __name__ == "__main__":
    main()
