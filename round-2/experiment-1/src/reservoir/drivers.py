"""Phase drivers: null calibration, power study, bug battery.

Each driver writes results/*.json intermediates as it completes
(crash-resilient) and returns the merged evidence dict that assemble.py turns
into the schema-valid method_out.json.
"""
from __future__ import annotations

import resource
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from .config import (RESULTS, NullCell, power_needs_null_cell, select_bugs,
                     select_null, select_power)
from .harness import (_dump_json, run_bug_cell, run_null_cell, run_power_cell,
                      run_parallel)


@dataclass
class _CellPayload:
    """Picklable worker payload: a phase cell plus its null-threshold record.

    Closures cannot cross a spawn-context ProcessPoolExecutor boundary, so
    power/bug workers take a payload object instead of a lambda binding the
    thresholds dict (the third-scale power run only worked because a single
    cell runs in-process).
    """
    cell: object
    null: dict

    def key(self) -> str:
        return getattr(self.cell, "key", lambda: str(self.cell))()


def _power_worker(payload: _CellPayload) -> dict:
    """Top-level power worker: run_power_cell(cell, thresholds)."""
    return run_power_cell(payload.cell, payload.null)


def _bug_worker(payload: _CellPayload) -> dict:
    """Top-level bug-battery worker: run_bug_cell(cell, thresholds)."""
    return run_bug_cell(payload.cell, payload.null)


def _worker_limits() -> None:
    """Defensive per-process memory cap (raise MemoryError, never OOM-kill)."""
    try:
        resource.setrlimit(resource.RLIMIT_AS, (14 << 30, 14 << 30))  # 14 GB virtual
    except (ValueError, OSError):
        pass


def run_null_phase(scale: str, workers: int) -> dict:
    cells = select_null(scale)
    logger.info(f"null phase: {len(cells)} cells at scale={scale}, workers={workers}")
    results = run_parallel(cells, run_null_cell, workers, "null")

    table = {"scale": scale, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "cells": results}
    _dump_json(RESULTS / "null_calibration.json", table)
    n_done = sum(1 for r in results if not r.get("skipped"))
    logger.info(f"null phase done: {n_done}/{len(results)} cells")

    # thresholds lookup for power/bugbattery phases
    thresholds: dict[str, dict] = {}
    for r in results:
        if not r.get("skipped"):
            thresholds[r["cell"]] = r
    _dump_json(RESULTS / "thresholds.json", thresholds)
    return table


def load_null_thresholds() -> dict[str, dict]:
    p = RESULTS / "thresholds.json"
    if not p.exists():
        raise FileNotFoundError(
            "thresholds.json missing -- run the null phase first (method.py null)")
    import json
    return json.loads(p.read_text())


def _ensure_null_thresholds(needed: dict[str, NullCell], workers: int,
                            thresholds: dict[str, dict]) -> None:
    """Run any needed null cells that were not yet computed (F2-safe)."""
    missing = {k: v for k, v in needed.items() if k not in thresholds}
    if not missing:
        return
    logger.info(f"auto-running {len(missing)} missing null cells: {sorted(missing)}")
    results = run_parallel(list(missing.values()), run_null_cell, workers, "null_support")
    for r in results:
        if not r.get("skipped"):
            thresholds[r["cell"]] = r
    _dump_json(RESULTS / "thresholds.json", thresholds)


def run_power_phase(scale: str, workers: int) -> dict:
    cells = select_power(scale)
    thresholds = load_null_thresholds()
    needed: dict[str, NullCell] = {}
    for c in cells:
        nc = power_needs_null_cell(c.n, c.p, c.m)
        if nc is None:
            raise RuntimeError(f"no matching null cell for power cell {c.key()}")
        needed.setdefault(nc.key(), nc)
    _ensure_null_thresholds(needed, workers, thresholds)

    payloads = [_CellPayload(cell=c, null=thresholds[power_needs_null_cell(c.n, c.p, c.m).key()])
                for c in cells]
    logger.info(f"power phase: {len(cells)} cells at scale={scale}, workers={workers}")
    results = run_parallel(payloads, _power_worker, workers, "power")
    table = {"scale": scale, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "cells": results}
    _dump_json(RESULTS / "power_surfaces.json", table)
    n_done = sum(1 for r in results if not r.get("skipped"))
    logger.info(f"power phase done: {n_done}/{len(results)} cells")
    return table


def run_bug_phase(scale: str, workers: int) -> dict:
    cells = select_bugs(scale)
    thresholds = load_null_thresholds()
    needed: dict[str, NullCell] = {}
    for c in cells:
        nc = _null_cell_for(c.n, c.k, c.m)
        if nc is None:
            raise RuntimeError(f"no matching null cell for bug cell {c.key()}")
        needed.setdefault(nc.key(), nc)
    _ensure_null_thresholds(needed, workers, thresholds)

    payloads = [_CellPayload(cell=c, null=thresholds[_null_cell_for(c.n, c.k, c.m).key()])
                for c in cells]
    logger.info(f"bug battery: {len(cells)} cells at scale={scale}, workers={workers}")
    results = run_parallel(payloads, _bug_worker, workers, "bug")
    table = {"scale": scale, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "cells": results}
    _dump_json(RESULTS / "bug_battery.json", table)
    n_done = sum(1 for r in results if not r.get("skipped"))
    logger.info(f"bug battery done: {n_done}/{len(results)} cells")
    return table


def _null_cell_for(n: int, k: int, m: int) -> NullCell | None:
    from .config import NULL_CELLS
    for c in NULL_CELLS:
        if c.n == n and c.k == k and c.m == m:
            return c
    return None