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
#                                  [--include-untracked]
#
#   --gpu      GPU type to request (default: T4 — enough for correctness).
#   --session  Colab session name (default: rlgbm-gpu).
#   --keep     Leave the VM running afterwards (for iterating); default stops it.
#   --include-untracked
#              Pack the live working tree (in-progress, uncommitted work) instead
#              of `git archive HEAD`. The default refuses uncommitted tracked
#              changes; this opt-in packs them, minus a strict exclude list, and
#              still refuses secret-like files.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="T4"
SESSION="rlgbm-gpu"
KEEP=0
INCLUDE_UNTRACKED=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        --session) SESSION="$2"; shift 2 ;;
        --keep) KEEP=1; shift ;;
        --include-untracked) INCLUDE_UNTRACKED=1; shift ;;
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
# --- release-safe upload tarball ---------------------------------------------
# Default: archive the COMMITTED tree (git archive HEAD); untracked files never
# ride along and uncommitted tracked changes are refused, so secrets / caches /
# build artifacts / private notes cannot leak into the upload. --include-untracked
# packs the live working tree instead (for in-progress work), minus the excludes
# below and with a hard refusal on secret-like files.
TAR_EXCLUDES=(
    --exclude='./.git' --exclude='**/.git'
    --exclude='**/.claude'
    --exclude='**/.env' --exclude='**/.env.*'
    --exclude='**/*.pem' --exclude='**/*.key' --exclude='**/id_rsa*'
    --exclude='**/.pypirc' --exclude='**/.npmrc'
    --exclude='**/__pycache__' --exclude='*.egg-info'
    --exclude='**/.pytest_cache' --exclude='**/.mypy_cache' --exclude='**/.ruff_cache'
    --exclude='**/.coverage'
    --exclude='./artifacts' --exclude='**/catboost_info' --exclude='./site'
    --exclude='./dist' --exclude='./build' --exclude='./target' --exclude='**/target'
    --exclude='./docs/paper' --exclude='./docs/gpu-research'
    --exclude='**/*.jsonl'                  # benchmark result ledgers
    --exclude='**/next-session-prompt.md'   # private runbook (assistant prompt)
)

guard_no_secrets() {
    local hits
    hits="$(find . -path ./.git -prune -o \( -name '.env' -o -name '.env.*' \
        -o -name '*.pem' -o -name '*.key' -o -name 'id_rsa*' \
        -o -name '.pypirc' -o -name '.npmrc' \) -print 2>/dev/null)"
    if [[ -n "$hits" ]]; then
        echo "error: refusing to pack -- secret-like files are present:" >&2
        echo "$hits" >&2
        exit 1
    fi
}

make_tarball() {  # $1 = output path
    if [[ "$INCLUDE_UNTRACKED" -eq 1 ]]; then
        echo ">> packing live working tree (--include-untracked) -> $1"
        guard_no_secrets
        tar "${TAR_EXCLUDES[@]}" -czf "$1" .
        return
    fi
    if ! git diff --quiet HEAD; then
        echo "error: tracked files differ from HEAD -- 'git archive HEAD' would omit your changes." >&2
        echo "  commit/stash them, or re-run with --include-untracked to pack the live tree." >&2
        git status --short >&2
        exit 1
    fi
    if [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
        echo ">> note: untracked files are NOT included (use --include-untracked to add them):"
        git ls-files --others --exclude-standard
    fi
    echo ">> archiving committed tree (git archive HEAD) -> $1"
    git archive --format=tar.gz -o "$1" HEAD
}

TARBALL="$(mktemp -t rlgbm-XXXXXX).tar.gz"
cleanup_local() { rm -f "$TARBALL"; }
trap cleanup_local EXIT

make_tarball "$TARBALL"

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
