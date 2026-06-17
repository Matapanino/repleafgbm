#!/usr/bin/env bash
# Run the trainable-embeddings benchmark on a Google Colab VM.
#
# Compares the encoder families (fixed identity/plr/periodic/cross, the
# pretrained-then-frozen torch_periodic/torch_plr/torch_periodic_plr/torch_mlp,
# and optional external GBMs) across regression/binary/multiclass. Pretraining
# runs on CPU (fit-time only); the VM provides scale, isolation, and a
# torch-equipped reproducible environment. Mirrors scripts/colab_gpu_test.sh.
#
# Requires the Colab CLI (https://github.com/googlecolab/google-colab-cli):
#   uv tool install google-colab-cli   # or: pip install google-colab-cli
#
# Usage:
#   bash scripts/colab_trainable_embeddings.sh [--gpu T4|L4|A100] [--session NAME] [--keep]
#
#   --gpu      VM accelerator to request (default: T4; pretraining is CPU, so a
#              CPU VM works too — GPU mainly buys faster CPU/more RAM here).
#   --session  Colab session name (default: rlgbm-emb).
#   --keep     Leave the VM running afterwards (for iterating); default stops it.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="T4"
SESSION="rlgbm-emb"
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

OUT_DIR="artifacts"
TARBALL="$(mktemp -t rlgbm-XXXXXX).tar.gz"
RESULTS="$(mktemp -t rlgbm-te-XXXXXX).tar.gz"
cleanup_local() { rm -f "$TARBALL" "$RESULTS"; }
trap cleanup_local EXIT

echo ">> packing working tree -> $TARBALL"
tar --exclude='.git' --exclude='**/__pycache__' --exclude='*.egg-info' \
    --exclude='target' --exclude='build' --exclude='dist' --exclude='.pytest_cache' \
    --exclude='artifacts' \
    -czf "$TARBALL" .

echo ">> provisioning $GPU VM (session: $SESSION)"
colab new -s "$SESSION" --gpu "$GPU"

stop_vm() { [[ "$KEEP" -eq 0 ]] && colab stop -s "$SESSION" || true; }
trap 'cleanup_local; stop_vm' EXIT

echo ">> uploading working tree"
colab upload -s "$SESSION" "$TARBALL" /content/rlgbm.tar.gz

echo ">> running trainable-embeddings benchmark on the VM"
colab exec -s "$SESSION" -f scripts/colab_trainable_embeddings.py

echo ">> downloading artifacts -> $OUT_DIR/trainable_embeddings/"
mkdir -p "$OUT_DIR"
colab download -s "$SESSION" /content/te_results.tar.gz "$RESULTS"
tar -xzf "$RESULTS" -C "$OUT_DIR"

echo ">> done. metrics.jsonl / summary.md / env.json in $OUT_DIR/trainable_embeddings/"
echo "   (copy summary.md to experiments/results/<date>-trainable-embeddings.md to commit it)"
if [[ "$KEEP" -eq 1 ]]; then
    echo ">> VM left running (session: $SESSION); 'colab stop -s $SESSION' when done."
fi
