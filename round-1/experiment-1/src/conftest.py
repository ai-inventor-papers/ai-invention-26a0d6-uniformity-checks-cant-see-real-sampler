"""Make the ``reservoir`` package importable when pytest collects tests/.

Pytest does not add the repo root to sys.path without this (importmode=prepend
only inserts the tests directory).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))