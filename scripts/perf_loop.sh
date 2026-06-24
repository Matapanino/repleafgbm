#!/usr/bin/env bash
# Local driver for the CUDA/GPU overnight optimization loop.
#
# Thin wrapper around benchmarks/cuda_overnight_loop.py that pins the stable
# single-thread environment (OMP_NUM_THREADS=1 avoids the torch+lightgbm libomp
# deadlock and steadies timing) and the src layout. GPU (cuda backend) work runs
# on Colab via scripts/colab_gpu_test.sh — this driver covers the numpy/rust
# locally-measurable surface plus the orchestrator dry-run.
#
# Usage:
#   bash scripts/perf_loop.sh --quick                 # plumbing dry-run (numpy+rust)
#   bash scripts/perf_loop.sh --mode matrix --reps 5 --tasks regression --sizes small
#   bash scripts/perf_loop.sh --mode ab --task multioutput --size large \
#       --variant-a rust --variant-b rust            # (env-* flags for A/B configs)
#
# All arguments are forwarded verbatim to the orchestrator.
set -euo pipefail

cd "$(dirname "$0")/.."

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONPATH="src:${PYTHONPATH:-}"

exec python3 -m benchmarks.cuda_overnight_loop "$@"
