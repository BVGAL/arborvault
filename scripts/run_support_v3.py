"""Train or evaluate the local support-v3 adapter experiment."""

import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "evaluate"])
    parser.add_argument("--config", type=Path, default=Path("configs/support-v3.json"))
    parser.add_argument(
        "--training", type=Path, default=Path("docs/benchmarks/support-v3/training")
    )
    parser.add_argument("--output", type=Path, default=Path("docs/benchmarks/support-v3/results"))
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    start = time.monotonic()

    def deadline():
        if time.monotonic() - start > config["maximum_local_seconds"]:
            raise TimeoutError("Local experiment time limit exceeded")

    if args.action == "train":
        from arborvault.support_v3_training import train

        result = train(config, args.config, args.training, args.work, deadline)
        print(
            json.dumps({k: result[k] for k in ("selected", "trainable_parameters", "seconds")}),
            flush=True,
        )
    else:
        from arborvault.support_v3_eval import evaluate

        result = evaluate(config, args.config, args.training, args.output, deadline)
        print(json.dumps(result["observed_promotion_checks"]), flush=True)


if __name__ == "__main__":
    main()
