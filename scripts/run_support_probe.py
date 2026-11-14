"""Extract CPU features or reproduce the frozen support-classifier analysis."""

import argparse
import json
import time
from pathlib import Path

from arborvault.support_probe import analyze, extract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["extract", "analyze"])
    parser.add_argument("--config", type=Path, default=Path("configs/support-v1.json"))
    parser.add_argument(
        "--features", type=Path, default=Path("docs/benchmarks/support-v1/features")
    )
    parser.add_argument("--output", type=Path, default=Path("docs/benchmarks/support-v1/results"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    started = time.monotonic()

    def deadline():
        if time.monotonic() - started > config["maximum_local_seconds"]:
            raise TimeoutError("Local experiment time limit exceeded")

    if args.action == "extract":
        result = extract(config, args.features, args.config, deadline=deadline)
    else:
        result = analyze(config, args.features, args.output, args.config)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
