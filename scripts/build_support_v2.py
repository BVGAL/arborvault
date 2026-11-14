"""Build fresh human transfer articles and semantic counterfactual groups."""

import argparse
import json
from pathlib import Path

from arborvault.support_v2_data import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/support-v2.json"))
    parser.add_argument("--source", type=Path, required=True, help="Original SQuAD train-v2.0.json")
    parser.add_argument("--output", type=Path, default=Path("datasets/support-v2"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["config_path"] = args.config.as_posix()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
    result = build(config, args.source, args.output, tokenizer)
    print(json.dumps(result["counts"]))


if __name__ == "__main__":
    main()
