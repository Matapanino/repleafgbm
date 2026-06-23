#!/usr/bin/env bash
# Interleaved A/B confirmation of the CUDA scan-threshold crossover on a Colab GPU
# VM: times 32768 (on-device scan at 200f) vs 131072 (host scan) back-to-back over
# several reps and pulls back the paired-diff report + raw JSONL.
#
# Companion to scripts/colab_scan_sweep.sh — same provisioning/upload/teardown, but
# it execs scripts/colab_scan_ab.py. Confirms (or refutes) the sweep's ~5-11% host
# edge on wide shapes before any default change. The default is unchanged here.
#
# Usage:
#   bash scripts/colab_scan_ab.sh [--gpu T4|L4|A100] [--session NAME] [--keep]
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="T4"
SESSION="rlgbm-scan-ab"
KEEP=0
# Hard wall-clock cap (s) on the remote exec. colab exec can hang for hours if the
# kernel websocket drops ("Connection was lost") because it does not always honour
# its own --timeout; an external watchdog kills it so the run can't wedge. The A/B
# itself is ~6 min, so the default leaves generous headroom. Override via env.
EXEC_TIMEOUT="${EXEC_TIMEOUT:-900}"
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
REPORT_OUT="experiments/results/${DATE}-cuda-scan-ab.md"
AB_OUT="artifacts/gpu_bench/${DATE}-${GPU}/scan_ab.jsonl"
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

echo ">> running CUDA scan-threshold A/B on the GPU (watchdog: ${EXEC_TIMEOUT}s)"
# Run exec in the background under an external watchdog: if the websocket drops
# and the CLI hangs, the watchdog SIGKILLs it so the run fails fast instead of
# wedging for hours. The EXIT trap still stops the VM either way.
colab exec -s "$SESSION" --timeout 600 -f scripts/colab_scan_ab.py &
exec_pid=$!
( sleep "$EXEC_TIMEOUT"; kill -KILL "$exec_pid" 2>/dev/null ) &
wd_pid=$!
exec_rc=0
wait "$exec_pid" || exec_rc=$?
kill "$wd_pid" 2>/dev/null || true
wait "$wd_pid" 2>/dev/null || true

if [[ "$exec_rc" -ne 0 ]]; then
    echo ">> colab exec failed or timed out (rc=$exec_rc); skipping downloads." >&2
    exit "$exec_rc"   # EXIT trap still stops the VM
fi

echo ">> downloading A/B report -> $REPORT_OUT"
mkdir -p experiments/results
colab download -s "$SESSION" /content/scan_ab_report.md "$REPORT_OUT"

echo ">> downloading raw A/B JSONL -> $AB_OUT"
mkdir -p "$(dirname "$AB_OUT")"
colab download -s "$SESSION" /content/gpu_bench/scan_ab.jsonl "$AB_OUT" || \
    echo "   (no A/B JSONL produced; skipping)"

echo ">> done. report at $REPORT_OUT"
if [[ "$KEEP" -eq 1 ]]; then
    echo ">> VM left running (session: $SESSION); 'colab stop -s $SESSION' when done."
fi
