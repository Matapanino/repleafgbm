#!/usr/bin/env bash
# Run the fair leaderboard (benchmarks/leaderboard.py) on a Google Colab CPU VM,
# resumably. The full Grinsztajn run (all models x ~40 trials x >=10 seeds) is
# hours of CPU; the ledger makes it survive the 30s idle timeout and disconnects.
#
# Resume model: each run downloads the JSONL ledger back to LOCAL_LEDGER; a
# re-invocation re-uploads it so completed (dataset, model, seed) cells are
# skipped. Re-run this script until the leaderboard report is complete.
#
# Requires the Colab CLI (https://github.com/googlecolab/google-colab-cli):
#   uv tool install google-colab-cli   # or: pip install google-colab-cli
#
# Usage:
#   bash scripts/colab_cpu_bench.sh [--suite NAME] [--seeds N] [--n-trials N]
#       [--quick] [--session NAME] [--keep] [-- <extra leaderboard args>]
set -euo pipefail
cd "$(dirname "$0")/.."

SUITE="grinsztajn_num_reg"
SEEDS=10
TRIALS=40
SESSION="rlgbm-cpu"
KEEP=0
QUICK=0
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --suite) SUITE="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --n-trials) TRIALS="$2"; shift 2 ;;
        --session) SESSION="$2"; shift 2 ;;
        --quick) QUICK=1; shift ;;
        --keep) KEEP=1; shift ;;
        --) shift; EXTRA=("$@"); break ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if ! command -v colab >/dev/null 2>&1; then
    echo "error: the 'colab' CLI is not installed." >&2
    echo "  uv tool install google-colab-cli   # or: pip install google-colab-cli" >&2
    exit 1
fi

DATE="$(date +%F)"
REPORT_OUT="experiments/results/${DATE}-leaderboard-colab.md"
LOCAL_LEDGER="benchmarks/results/leaderboard_${SUITE}.jsonl"
TARBALL="$(mktemp -t rlgbm-XXXXXX).tar.gz"
ARGV_FILE="$(mktemp -t rlgbm-argv-XXXXXX)"
cleanup_local() { rm -f "$TARBALL" "$ARGV_FILE"; }
trap cleanup_local EXIT

# Leaderboard argv (one per line) read by the remote driver. Output + ledger are
# pinned to /content paths so the driver can hand them back for download/resume.
{
    echo "--suite"; echo "$SUITE"
    echo "--seeds"; echo "$SEEDS"
    echo "--n-trials"; echo "$TRIALS"
    [[ "$QUICK" -eq 1 ]] && echo "--quick"
    echo "--out"; echo "/content/leaderboard.md"
    echo "--ledger"; echo "/content/ledger.jsonl"
    for a in "${EXTRA[@]:-}"; do [[ -n "$a" ]] && echo "$a"; done
} > "$ARGV_FILE"

echo ">> packing working tree -> $TARBALL"
tar --exclude='.git' --exclude='**/__pycache__' --exclude='*.egg-info' \
    --exclude='target' --exclude='build' --exclude='dist' --exclude='.pytest_cache' \
    -czf "$TARBALL" .

echo ">> provisioning CPU VM (session: $SESSION)"
colab new -s "$SESSION"
stop_vm() { [[ "$KEEP" -eq 0 ]] && colab stop -s "$SESSION" || true; }
trap 'cleanup_local; stop_vm' EXIT

echo ">> uploading working tree + argv"
colab upload -s "$SESSION" "$TARBALL" /content/rlgbm.tar.gz
colab upload -s "$SESSION" "$ARGV_FILE" /content/bench_argv.txt

# Resume: hand the VM the ledger from a previous run, if we have one.
if [[ -f "$LOCAL_LEDGER" ]]; then
    echo ">> resuming from $LOCAL_LEDGER"
    colab upload -s "$SESSION" "$LOCAL_LEDGER" /content/ledger_in.jsonl
fi

echo ">> running leaderboard (OMP_NUM_THREADS=1) — idle timeout 1800s"
colab exec -s "$SESSION" --timeout 1800 -f scripts/colab_remote_bench.py

echo ">> downloading ledger -> $LOCAL_LEDGER (for the next resume)"
mkdir -p "$(dirname "$LOCAL_LEDGER")"
colab download -s "$SESSION" /content/ledger.jsonl "$LOCAL_LEDGER" || \
    echo "   (no ledger produced; skipping)"

echo ">> downloading report -> $REPORT_OUT"
mkdir -p experiments/results
colab download -s "$SESSION" /content/leaderboard.md "$REPORT_OUT"
for task in regression binary multiclass; do
    colab download -s "$SESSION" "/content/leaderboard-cd-${task}.png" \
        "experiments/results/${DATE}-leaderboard-cd-${task}.png" 2>/dev/null || true
done

echo ">> done. report at $REPORT_OUT"
echo "   Re-run this script to resume any unfinished cells (ledger is reused)."
if [[ "$KEEP" -eq 1 ]]; then
    echo ">> VM left running (session: $SESSION); 'colab stop -s $SESSION' when done."
fi
