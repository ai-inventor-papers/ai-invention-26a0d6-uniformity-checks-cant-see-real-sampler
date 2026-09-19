#!/usr/bin/env python3
"""Smoke test: the power/bug workers must cross a spawn ProcessPoolExecutor.

Regression test for the closure-pickling bug that silently forced single-cell
runs: run_power_phase/run_bug_phase previously wrapped run_power_cell in an
enclosing-scope closure, which fails to pickle under the spawn start method
whenever a phase has >= 2 cells (the third-scale power run only worked
because it had 1 cell and ran in-process).  This test submits two tiny
payloads through the same machinery to force the pickling path.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reservoir.config import PowerCell, BugCell
from reservoir.drivers import _power_worker, _bug_worker, _CellPayload


def main() -> None:
    # --- power worker through a real spawn pool (2 cells -> pickling) ---
    null_300 = {"thresholds": {
        "maxdev": {"alpha_0.05": 3.0, "alpha_0.01": 4.0},
        "chi2": {"alpha_0.05": 300.0, "alpha_0.01": 320.0},
        "trend_slope": {"alpha_0.05": 2.0, "alpha_0.01": 2.6},
        "energy": {"alpha_0.05": 1400.0, "alpha_0.01": 1600.0}},
        "trend_z_reference": {"mu": 0.0, "sd": 1.0}}
    cells = [
        PowerCell(n=150, p=0.5, m=200, n_fine=200, n_coarse=120),
        PowerCell(n=150, p=0.05, m=200, n_fine=200, n_coarse=120),
    ]
    payloads = [_CellPayload(cell=c, null=null_300) for c in cells]

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
        futs = [pool.submit(_power_worker, p) for p in payloads]
        outs = [f.result(timeout=600) for f in futs]
    assert all("families" in o and len(o["families"]) == 5 for o in outs), \
        "power worker output incomplete"
    print("POWER WORKER SPAWN PICKLING: OK", [o["n"] for o in outs])

    # --- bug worker through a real spawn pool ---
    bug_cells = [BugCell(n=60, k=30, m=200, n_reps=100),
                 BugCell(n=80, k=40, m=200, n_reps=100)]
    null_bug = {"thresholds": {
        "maxdev": {"alpha_0.05": 3.0, "alpha_0.01": 4.0},
        "chi2": {"alpha_0.05": 60.0, "alpha_0.01": 70.0},
        "trend_slope": {"alpha_0.05": 2.0, "alpha_0.01": 2.6},
        "energy": {"alpha_0.05": 300.0, "alpha_0.01": 400.0}},
        "trend_z_reference": {"mu": 0.0, "sd": 1.0}}
    bug_payloads = [_CellPayload(cell=c, null=null_bug) for c in bug_cells]
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
        futs = [pool.submit(_bug_worker, p) for p in bug_payloads]
        outs = [f.result(timeout=600) for f in futs]
    assert all("bugs" in o and len(o["bugs"]) == 4 for o in outs), \
        "bug worker output incomplete"
    print("BUG WORKER SPAWN PICKLING: OK", [o["n"] for o in outs])
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()