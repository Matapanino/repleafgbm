#!/usr/bin/env bash
# Sweep the CUDA adaptive-scan threshold (REPLEAFGBM_CUDA_SCAN_MIN_CELLS) on a
# Google Colab GPU VM and pull back the crossover report + raw JSONL.
#
# Companion to scripts/colab_gpu_test.sh (parity loop). Same provisioning, upload,
# and teardown plumbing, but it execs scripts/colab_scan_sweep.py — which runs the
# cuda-only threshold sweep across the scan-dominated workloads — instead of the
# parity driver. The default scan threshold is unchanged; this only measures it.
#
# Requires the Colab CLI (https://github.com/googlecolab/google-colab-cli):
#   uv tool install google-colab-cli   # or: pip install google-colab-cli
#
# Usage:
#   bash scripts/colab_scan_sweep.sh [--gpu T4|L4|A100] [--session NAME] [--keep]
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="T4"
SESSION="rlgbm-scan-sweep"
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
REPORT_OUT="experiments/results/${DATE}-cuda-scan-sweep.md"
SWEEP_OUT="artifacts/gpu_bench/${DATE}-${GPU}/scan_sweep.jsonl"
TARBALL="$(mktemp -t rlgbm-XXXXXX).tar.gz"
cleanup_local() { rm -f "$TARBALL"; }
trap cleanup_local EXIT

echo ">> packing working tree -> $TARBALL"
tar --exclude='.git' --exclude='**/__pycache__' --exclude='*.egg-info' \
    --exclude='target' --exclude='build' --exclude='dist' --exclude='.pytest_cache' \
    -czf "$TARBALL" .

echo ">> provisioning $GPU VM (session: $SESSION)"
colab new -s "$SESSION" --gpu "$GPU"

stop_vm() { [[ "$KEEP" -eq 0 ]] && colab stop -s "$SESSION" || true; }
trap 'cleanup_local; stop_vm' EXIT

echo ">> uploading working tree"
colab upload -s "$SESSION" "$TARBALL" /content/rlgbm.tar.gz

echo ">> running CUDA scan-threshold sweep on the GPU"
# --timeout is the idle reply timeout; the multiclass sweep has ~20s fits x5
# thresholds that run silently, so give it ample headroom.
colab exec -s "$SESSION" --timeout 1800 -f scripts/colab_scan_sweep.py

echo ">> downloading crossover report -> $REPORT_OUT"
mkdir -p experiments/results
colab download -s "$SESSION" /content/scan_sweep_report.md "$REPORT_OUT"

echo ">> downloading raw sweep JSONL -> $SWEEP_OUT"
mkdir -p "$(dirname "$SWEEP_OUT")"
colab download -s "$SESSION" /content/gpu_bench/scan_sweep.jsonl "$SWEEP_OUT" || \
    echo "   (no sweep JSONL produced; skipping)"

echo ">> done. report at $REPORT_OUT"
if [[ "$KEEP" -eq 1 ]]; then
    echo ">> VM left running (session: $SESSION); 'colab stop -s $SESSION' when done."
fi
