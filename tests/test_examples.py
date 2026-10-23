"""Smoke tests for executable repository examples."""

import os
import subprocess
import sys
from pathlib import Path


def test_matgbm_leaf_experts_example_exits_zero():
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    source_root = str(root / "src")
    environment["PYTHONPATH"] = source_root + os.pathsep + environment.get("PYTHONPATH", "")
    environment["OMP_NUM_THREADS"] = "1"

    completed = subprocess.run(
        [sys.executable, "examples/matgbm_leaf_experts.py"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
