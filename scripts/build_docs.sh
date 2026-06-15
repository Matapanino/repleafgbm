#!/usr/bin/env bash
# Build the public API reference with pdoc (docs/ holds the prose; pdoc renders
# the API from docstrings). Output is generated, not committed.
#
#   pip install -e ".[docs]"   # or: pip install pdoc
#   bash scripts/build_docs.sh [output_dir]
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-site/api}"

# Allow running from a source checkout without an editable install.
if ! python3 -c "import repleafgbm" >/dev/null 2>&1; then
    export PYTHONPATH="src:${PYTHONPATH:-}"
fi

echo "Building API reference into $OUT ..."
python3 -m pdoc -o "$OUT" repleafgbm
echo "API reference written to $OUT/index.html"
