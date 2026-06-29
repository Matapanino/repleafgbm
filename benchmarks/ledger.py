"""Resumable JSONL checkpoint ledger for long benchmark runs.

A fair-leaderboard run is a grid of *cells* — ``(suite, dataset, model, seed)``
each doing its own HPO + test evaluation. On Colab CPU such a run takes hours and
must survive the 30s idle timeout and disconnects. The ledger appends **one JSON
line per completed cell**; on restart, completed cells are skipped (``done``) and
their results are read back for aggregation (``records``) without recompute.

The first line of a fresh file is a ``_meta`` provenance record (git SHA + dirty
flag, package versions, ``OMP_NUM_THREADS``, a UTC ``run_id``) mirroring
``gpu_profile.py``, so a published leaderboard ties to the environment that made
it. A half-written trailing line from a killed process is tolerated on load.

Lives under ``benchmarks/`` only; never imported by the library (``src/``).
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Packages whose versions are pinned in the provenance header.
_PROVENANCE_PKGS = (
    "numpy", "pandas", "scipy", "scikit-learn", "repleafgbm",
    "optuna", "lightgbm", "xgboost", "catboost", "matplotlib",
)


def cell_key(suite: str, dataset: str, model: str, seed: int, stage: str = "eval") -> str:
    """Deterministic key string for one grid cell."""
    return f"{suite}|{dataset}|{model}|{seed}|{stage}"


def _git_state() -> tuple[str | None, bool | None]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or None
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return sha, bool(dirty)
    except Exception:  # pragma: no cover - git absent / not a repo
        return None, None


def default_provenance() -> dict[str, Any]:
    """Capture git + environment + package versions for the ledger header."""
    import importlib.metadata as md

    def ver(dist: str) -> str:
        try:
            return md.version(dist)
        except md.PackageNotFoundError:
            return "(not installed)"

    sha, dirty = _git_state()
    return {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "git_sha": sha,
        "git_dirty": dirty,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "packages": {p: ver(p) for p in _PROVENANCE_PKGS},
    }


class Ledger:
    """Append-only, resumable record of completed benchmark cells.

    Parameters
    ----------
    path:
        JSONL file. Created (with a ``_meta`` header) if absent.
    write_meta:
        Write the provenance header on first creation (default ``True``).
    provenance:
        Override the captured provenance (mainly for tests).
    """

    def __init__(self, path, write_meta: bool = True, provenance: dict | None = None):
        self.path = Path(path)
        self._records: dict[str, dict[str, Any]] = {}
        self._meta: dict[str, Any] | None = None
        fresh = (not self.path.exists()) or self.path.stat().st_size == 0
        if fresh and write_meta:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._meta = provenance if provenance is not None else default_provenance()
            self._append_raw({"_meta": self._meta})
        self._load()

    # -- persistence ----------------------------------------------------------
    def _append_raw(self, obj: dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # Tolerate a half-written trailing line from a killed run.
                    continue
                if "_meta" in obj:
                    self._meta = obj["_meta"]
                    continue
                key = obj.get("key")
                if key is not None:
                    self._records[key] = obj

    # -- query ----------------------------------------------------------------
    @property
    def provenance(self) -> dict[str, Any] | None:
        return self._meta

    def done(self, suite, dataset, model, seed, stage: str = "eval") -> bool:
        return cell_key(suite, dataset, model, seed, stage) in self._records

    def get(self, suite, dataset, model, seed, stage: str = "eval") -> dict | None:
        rec = self._records.get(cell_key(suite, dataset, model, seed, stage))
        return rec["payload"] if rec is not None else None

    def records(self) -> list[dict[str, Any]]:
        """All completed cells as full records (key fields + ``payload``)."""
        return list(self._records.values())

    def __len__(self) -> int:
        return len(self._records)

    # -- mutation -------------------------------------------------------------
    def record(self, suite, dataset, model, seed, payload: dict, stage: str = "eval") -> None:
        """Append one completed cell and remember it in-process."""
        key = cell_key(suite, dataset, model, seed, stage)
        obj = {
            "key": key,
            "suite": suite,
            "dataset": dataset,
            "model": model,
            "seed": int(seed),
            "stage": stage,
            "ts": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        self._append_raw(obj)
        self._records[key] = obj
