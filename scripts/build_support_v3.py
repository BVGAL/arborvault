"""Build new support-v3 partitions from the original public source."""

import argparse
import json
from pathlib import Path

from arborvault.support_v3_data import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/support-v3.json"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("datasets/support-v3"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"], revision=config["revision"], local_files_only=True
    )
    result = build(config, args.source, args.output, tokenizer)
    print(json.dumps(result["counts"]), flush=True)


if __name__ == "__main__":
    main()
