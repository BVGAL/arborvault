#!/usr/bin/env bash
# Run inside a prepared Linux GPU environment with a disposable PostgreSQL server.
set -euo pipefail
: "${TEST_DATABASE_URL:?Set a disposable PostgreSQL administrator URL}"
experiment_dir="${1:?Supply a new experiment output directory}"
if [[ -e "$experiment_dir" ]]; then
  echo "Refusing to overwrite an existing experiment" >&2
  exit 2
fi
mkdir -p "$experiment_dir"
export HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
python scripts/train_qwen3.py --output "$experiment_dir/training" --time-limit-seconds 7200
python scripts/benchmark_qwen3.py --training-output "$experiment_dir/training" \
  --split dev --output "$experiment_dir/dev"
python scripts/benchmark_qwen3.py --training-output "$experiment_dir/training" \
  --split test --output "$experiment_dir/test"
python - "$experiment_dir" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
hashes = {}
for path in sorted(root.rglob('*')):
    if path.is_file():
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
(root / 'artifact-manifest.json').write_text(json.dumps(hashes, indent=2) + '\n')
print(json.dumps({'stage': 'experiment_complete', 'files': len(hashes)}))
PY
# Finishing this process does NOT stop cloud billing. Retrieve and verify artifacts,
# then terminate only this experiment's pod using the authenticated control plane.
