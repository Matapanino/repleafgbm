#!/usr/bin/env bash
# Development check: lint (if ruff is installed), tests, and examples.
set -euo pipefail
cd "$(dirname "$0")/.."

# torch and lightgbm each bundle their own OpenMP runtime; loading both into
# one process can deadlock parallel regions on macOS. Single-threaded OpenMP
# avoids it, and the test suite/examples are small enough not to care.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

if python3 -m ruff --version >/dev/null 2>&1; then
    python3 -m ruff check src tests examples benchmarks
fi

python3 -m pytest tests/ -q
python3 examples/regression_basic.py
python3 examples/binary_classification_basic.py
python3 examples/dataset_api_basic.py
python3 examples/stacking_lightgbm.py  # self-skips without lightgbm

echo "All checks passed."
