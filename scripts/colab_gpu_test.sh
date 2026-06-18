#!/usr/bin/env bash
# Build + parity-test the CUDA backend on a Google Colab GPU VM.
#
# The dev machine (macOS) has no NVIDIA GPU and CI has no GPU runner, so the
# `split_backend="cuda"` path is validated here: provision a GPU VM via the
# Colab CLI, upload the current working tree, run tests/test_cuda_backend.py on
# the GPU, pull back a markdown report, then tear the VM down.
#
# Requires the Colab CLI (https://github.com/googlecolab/google-colab-cli):
#   uv tool install google-colab-cli   # or: pip install google-colab-cli
#
# Usage:
#   bash scripts/colab_gpu_test.sh [--gpu T4|L4|A100] [--session NAME] [--keep]
#
#   --gpu      GPU type to request (default: T4 — enough for correctness).
#   --session  Colab session name (default: rlgbm-gpu).
#   --keep     Leave the VM running afterwards (for iterating); default stops it.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="T4"
SESSION="rlgbm-gpu"
KEEP=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        --session) SESSION="$2"; shift 2 ;;
        --keep) KEEP=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if ! command -v colab >/dev/null 2>&1; then
    echo "error: the 'colab' CLI is not installed." >&2
    echo "  uv tool install google-colab-cli   # or: pip install google-colab-cli" >&2
    exit 1
fi

DATE="$(date +%F)"
REPORT_OUT="experiments/results/${DATE}-cuda-parity.md"
SUITE_OUT="experiments/results/${DATE}-gpu-backend-suite.md"
BENCH_OUT="artifacts/gpu_bench/${DATE}-${GPU}/cases.jsonl"
TARBALL="$(mktemp -t rlgbm-XXXXXX).tar.gz"
cleanup_local() { rm -f "$TARBALL"; }
trap cleanup_local EXIT

echo ">> packing working tree -> $TARBALL"
# Working tree (not just committed files) so in-progress CUDA work is tested.
tar --exclude='.git' --exclude='**/__pycache__' --exclude='*.egg-info' \
    --exclude='target' --exclude='build' --exclude='dist' --exclude='.pytest_cache' \
    -czf "$TARBALL" .

echo ">> provisioning $GPU VM (session: $SESSION)"
colab new -s "$SESSION" --gpu "$GPU"

stop_vm() { [[ "$KEEP" -eq 0 ]] && colab stop -s "$SESSION" || true; }
trap 'cleanup_local; stop_vm' EXIT

echo ">> uploading working tree"
colab upload -s "$SESSION" "$TARBALL" /content/rlgbm.tar.gz

echo ">> running CUDA parity tests + benchmark on the GPU"
# --timeout is an *idle* reply timeout (default 30s); the gpu_profile matrix has
# single fits (e.g. multiclass numpy) that run silently for ~45s, so a small
# timeout aborts the exec after the remote already finished. Give it headroom.
colab exec -s "$SESSION" --timeout 1800 -f scripts/colab_remote_test.py

echo ">> downloading report -> $REPORT_OUT"
mkdir -p experiments/results
colab download -s "$SESSION" /content/cuda_parity_report.md "$REPORT_OUT"

echo ">> downloading backend-suite report -> $SUITE_OUT"
# Best-effort: only the newer driver writes the backend-comparison suite.
colab download -s "$SESSION" /content/gpu_backend_suite.md "$SUITE_OUT" || \
    echo "   (no backend-suite report produced; skipping)"

echo ">> downloading gpu_profile transfer counters -> $BENCH_OUT"
mkdir -p "$(dirname "$BENCH_OUT")"
# Best-effort: the JSONL only exists if the gpu_profile smoke ran (newer driver).
colab download -s "$SESSION" /content/gpu_bench/cases.jsonl "$BENCH_OUT" || \
    echo "   (no gpu_bench JSONL produced; skipping)"

echo ">> done. report at $REPORT_OUT"
if [[ "$KEEP" -eq 1 ]]; then
    echo ">> VM left running (session: $SESSION); 'colab stop -s $SESSION' when done."
fi
