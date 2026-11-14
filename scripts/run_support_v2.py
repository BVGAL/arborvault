"""Extract frozen features or fit/evaluate the preregistered abstention study."""

import argparse
import json
import time
from pathlib import Path

from arborvault.support_v2_eval import analyze
from arborvault.support_v2_features import extract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["extract", "analyze"])
    parser.add_argument("--config", type=Path, default=Path("configs/support-v2.json"))
    parser.add_argument(
        "--features", type=Path, default=Path("docs/benchmarks/support-v2/features")
    )
    parser.add_argument("--output", type=Path, default=Path("docs/benchmarks/support-v2/results"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    start = time.monotonic()

    def deadline():
        if time.monotonic() - start > config["maximum_local_seconds"]:
            raise TimeoutError("Local extraction time limit exceeded")

    if args.action == "extract":
        result = extract(config, args.features, args.config, deadline)
    else:
        result = analyze(config, args.features, args.output, args.config)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
