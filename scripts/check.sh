#!/usr/bin/env bash
# Development check: lint (if ruff is installed), tests, and examples.
set -euo pipefail
cd "$(dirname "$0")/.."

if python3 -m ruff --version >/dev/null 2>&1; then
    python3 -m ruff check src tests examples benchmarks
fi

python3 -m pytest tests/ -q
python3 examples/regression_basic.py
python3 examples/binary_classification_basic.py
python3 examples/dataset_api_basic.py
python3 examples/stacking_lightgbm.py  # self-skips without lightgbm

echo "All checks passed."
