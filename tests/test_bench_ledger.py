"""Unit tests for benchmarks/ledger.py (resumable checkpointing).

Verifies write->reopen persistence, the skip-completed resume path, corrupt
trailing-line tolerance, and that the provenance header is written exactly once.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable so ``benchmarks`` resolves.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.ledger import Ledger, cell_key  # noqa: E402

_META = {"run_id": "test", "git_sha": "abc", "packages": {}}


def test_write_reopen_persists(tmp_path):
    p = tmp_path / "ledger.jsonl"
    led = Ledger(p, provenance=_META)
    led.record("grin", "california", "lightgbm", 0, {"rmse": 0.45})
    assert led.done("grin", "california", "lightgbm", 0)
    assert not led.done("grin", "california", "xgboost", 0)

    # A fresh handle on the same file sees the prior cell and its payload.
    reopened = Ledger(p)
    assert reopened.done("grin", "california", "lightgbm", 0)
    assert reopened.get("grin", "california", "lightgbm", 0) == {"rmse": 0.45}
    assert len(reopened) == 1


def test_resume_only_computes_missing(tmp_path):
    p = tmp_path / "ledger.jsonl"
    cells = [("d1", "m1"), ("d1", "m2"), ("d2", "m1"), ("d2", "m2")]

    led = Ledger(p, provenance=_META)
    for dataset, model in cells[:2]:
        led.record("s", dataset, model, 0, {"ok": True})

    # Restart: a new run skips completed cells, computes only the rest.
    resumed = Ledger(p)
    computed = []
    for dataset, model in cells:
        if resumed.done("s", dataset, model, 0):
            continue
        computed.append((dataset, model))
        resumed.record("s", dataset, model, 0, {"ok": True})
    assert computed == cells[2:]
    assert len(Ledger(p)) == 4


def test_corrupt_trailing_line_tolerated(tmp_path):
    p = tmp_path / "ledger.jsonl"
    led = Ledger(p, provenance=_META)
    led.record("s", "d", "m", 0, {"v": 1})
    # Simulate a process killed mid-write: a truncated final line.
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"key": "s|d|m|1|eval", "payl')

    reopened = Ledger(p)
    assert reopened.done("s", "d", "m", 0)          # the good record survived
    assert not reopened.done("s", "d", "m", 1)      # the truncated one is ignored
    assert len(reopened) == 1


def test_meta_written_once(tmp_path):
    p = tmp_path / "ledger.jsonl"
    Ledger(p, provenance=_META).record("s", "d", "m", 0, {})
    reopened = Ledger(p)  # must NOT append a second _meta line
    assert reopened.provenance == _META
    meta_lines = [ln for ln in p.read_text().splitlines() if '"_meta"' in ln]
    assert len(meta_lines) == 1


def test_cell_key_is_deterministic():
    assert cell_key("s", "d", "m", 3) == "s|d|m|3|eval"
    assert cell_key("s", "d", "m", 3, "hpo") == "s|d|m|3|hpo"
