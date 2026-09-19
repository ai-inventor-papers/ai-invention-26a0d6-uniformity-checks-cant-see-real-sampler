#!/usr/bin/env python3
"""Reservoir-sampling uniformity protocol -- single CLI entrypoint.

Subcommands (each writes results/*.json intermediates as it completes):
  sanity      -- adversarial gates before scaling (duality, Algorithm R vs
                 priority, null self-consistency, mechanism/spike-blind-band)
  null        -- null calibration (a) + C1 benchmark (b) + C3 variance (c)
  power       -- detection power (M) over five bias families, blind bands,
                 gap-vs-n, spike inversion
  bugbattery  -- four faithful buggy implementations through the same harness
  report      -- assemble schema-valid method_out.json + README.md
  demo        -- quick user-facing run of the core task (n=300 mini)

Iteration-2 CONFIRM pipeline (trial-budget law):
  confirm anchors      -- phase 1: extract iteration-1 anchors, pre-register
  confirm null         -- phase 2: null calibration at m in {2000,5000,10000}
  confirm fwer         -- phase 3: FWER of the corrected 3-test protocol
  confirm power        -- phase 4: half-power measurement (+ G4 anchor gate)
  confirm law          -- phase 5: P1/P2/P3 tests and verdict
  confirm bugs         -- phase 6: blind bug battery in trials-to-90% units
  confirm pract        -- phase 7: practitioner maxdev null at m = 10^6
  confirm secondary    -- phase 8: n/p-invariance arms
  confirm gates        -- phase 9: G1-G5 sanity gates
  confirm report       -- phase 10: schema-valid method_out.json + README
  confirm all          -- phases 1-10 sequentially, skipping existing checkpoints

Usage:
  uv run method.py sanity
  uv run method.py null --scale full --workers 4
  uv run method.py confirm all --workers 6
  uv run method.py confirm power --workers 6 --m 2000,5000,10000
"""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from reservoir import __version__  # noqa: E402
from reservoir.assemble import assemble, write_readme  # noqa: E402
from reservoir.config import LOGS, RESULTS, child_rng  # noqa: E402
from reservoir.drivers import (load_null_thresholds, run_bug_phase,  # noqa: E402
                               run_null_phase, run_power_phase)
from reservoir.samplers import algorithm_R_counts, priority_counts_matrix, validate_count_rows  # noqa: E402
from reservoir.sanity import run_all_gates  # noqa: E402
from reservoir.stats import compute_stats, empirical_quantiles  # noqa: E402

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOGS / "run.log"), rotation="50 MB", level="DEBUG")


def _set_memory_limits() -> None:
    """Container has 29 GB; cap virtual memory so we raise MemoryError, never
    get OOM-killed.  Workers set their own tighter cap in _worker_limits."""
    try:
        resource.setrlimit(resource.RLIMIT_AS, (22 << 30, 22 << 30))
    except (ValueError, OSError):
        logger.warning("could not set RLIMIT_AS")


def _workers_arg(n: int | None) -> int:
    if n and n > 0:
        return n
    try:  # cgroup v2 cpu quota
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return max(1, min(4, -(-int(parts[0]) // int(parts[1]))))
    except (OSError, ValueError):
        pass
    return 4


def _demo() -> None:
    """Core task demo: sample k of n uniformly, verify empirically, report
    max deviation from expected frequency (also exercises the pipeline)."""
    n, k, m, n_reps = 300, 150, 500, 2000
    rng = child_rng("demo")
    counts = priority_counts_matrix(n, k, m, n_reps, rng)
    validate_count_rows(counts, m, k)
    mu = m * k / n
    stats = compute_stats(counts, mu)
    qs = empirical_quantiles(stats["maxdev"], (0.5, 0.95, 0.99, 0.999))
    counts_r = algorithm_R_counts(n, k, m, n_reps, child_rng("demo_algoR"), variant="correct")
    validate_count_rows(counts_r, m, k)
    mu_r = m * k / n
    qs_r = empirical_quantiles(compute_stats(counts_r, mu_r)["maxdev"], (0.5, 0.95, 0.99))
    out = {
        "n": n, "k": k, "m": m, "n_reps": n_reps, "mu": mu,
        "maxdev_quantiles_priority": {str(q): v for q, v in qs.items()},
        "maxdev_quantiles_algorithmR": {str(q): v for q, v in qs_r.items()},
        "duality_maxdev_check": float(stats["maxdev"].max() <= max(qs.values()) + 1e-9),
        "verdict": "uniform: maxdev stays within the expected band",
    }
    (RESULTS / "demo.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=2))


def main() -> None:
    import argparse

    _set_memory_limits()
    RESULTS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(description="Reservoir-sampling uniformity protocol")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("sanity", help="adversarial gates before scaling")
    sub.add_parser("demo", help="quick core-task demo")

    for name in ("null", "power", "bugbattery"):
        p = sub.add_parser(name, help=f"run the {name} phase")
        p.add_argument("--scale", choices=("mini", "third", "full"), default="full")
        p.add_argument("--workers", type=int, default=None)

    sub.add_parser("report", help="assemble method_out.json + README.md")

    # ---- iteration-2 confirm pipeline ----
    cp = sub.add_parser("confirm", help="iteration-2 confirm phases")
    csub = cp.add_subparsers(dest="confirm_command", required=True)
    csub.add_parser("anchors", help="phase 1: anchors + pre-registration")
    for cname in ("null", "fwer"):
        p = csub.add_parser(cname, help=f"run confirm phase {cname}")
        p.add_argument("--workers", type=int, default=None)
        p.add_argument("--m", type=str, default=None,
                       help="comma-separated trial budgets (default: all)")
        p.add_argument("--force", action="store_true")
    p = csub.add_parser("power", help="phase 4: half-power measurement")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--m", type=str, default="2000,5000,10000",
                   help="comma-separated trial budgets")
    p.add_argument("--family", type=str, default=None,
                   help="comma-separated families (default: both)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-resume", action="store_true",
                   help="ignore previously cached grid rows for a cell")
    csub.add_parser("law", help="phase 5: P1/P2/P3 + verdict")
    p = csub.add_parser("bugs", help="phase 6: blind bug battery")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p = csub.add_parser("pract", help="phase 7: practitioner null")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--cell-b", action="store_true", help="include cell B (n=3000)")
    p.add_argument("--force", action="store_true")
    p = csub.add_parser("secondary", help="phase 8: n/p-invariance arms")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--arms", type=str, default="1", help="comma-separated arm ids")
    p.add_argument("--force", action="store_true")
    csub.add_parser("gates", help="phase 9: G1-G5 sanity gates")
    csub.add_parser("report", help="phase 10: assemble method_out.json + README")
    p = csub.add_parser("all", help="phases 1-10 sequentially")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--with-cell-b", action="store_true")
    p.add_argument("--arms", type=str, default="1")

    args = parser.parse_args()

    if args.command == "sanity":
        t0 = time.time()
        gates = run_all_gates()
        gates["wall_s"] = round(time.time() - t0, 2)
        (RESULTS / "sanity_gates.json").write_text(json.dumps(gates, indent=1))
        logger.info(f"sanity gates passed in {gates['wall_s']}s -> ready to scale")
        return

    if args.command == "demo":
        _demo()
        return

    if args.command == "report":
        assemble(require_complete=False)
        summary = _readme_summary()
        write_readme(summary)
        logger.info("report written: method_out.json + README.md")
        return

    workers = _workers_arg(getattr(args, "workers", None))
    if args.command == "null":
        run_null_phase(args.scale, workers)
    elif args.command == "power":
        run_power_phase(args.scale, workers)
    elif args.command == "bugbattery":
        run_bug_phase(args.scale, workers)
    elif args.command == "confirm":
        _run_confirm(args, workers)


def _run_confirm(args, workers: int) -> None:
    """Iteration-2 confirm pipeline dispatcher (crash-resumable checkpoints)."""
    import csv

    from reservoir.anchors import run_phase1
    from reservoir.assemble2 import assemble2, write_readme2
    from reservoir.bugs2 import run_bug_phase as run_bug2
    from reservoir.confirm_null import run_fwer_phase, run_null_phase as run_null2
    from reservoir.confirm_power import run_power_phase as run_power2
    from reservoir.law import run_law_tests
    from reservoir.pract import run_pract_phase
    from reservoir.sanity2 import run_all_confirm_gates
    from reservoir.secondary import run_secondary_phase

    cmd = args.confirm_command
    timings = LOGS / "timings.csv"

    def _record(phase: str, wall_s: float) -> None:
        with timings.open("a", newline="") as fh:
            csv.writer(fh).writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), phase,
                                     round(wall_s, 2)])

    def _timed(phase: str, fn) -> None:
        t0 = time.time()
        logger.info(f"=== CONFIRM {phase} START ===")
        fn()
        _record(phase, time.time() - t0)
        logger.info(f"=== CONFIRM {phase} DONE ({time.time() - t0:.1f}s) ===")

    if cmd == "anchors":
        _timed("anchors", lambda: run_phase1())
    elif cmd == "null":
        m_list = tuple(int(x) for x in args.m.split(",") if x.strip()) if args.m else None
        _timed("null", lambda: run_null2(workers, m_list=m_list, force=args.force))
    elif cmd == "fwer":
        m_list = tuple(int(x) for x in args.m.split(",") if x.strip()) if args.m else None
        _timed("fwer", lambda: run_fwer_phase(workers, m_list=m_list, force=args.force))
    elif cmd == "power":
        m_list = tuple(int(x) for x in args.m.split(",") if x.strip())
        fam_list = tuple(x.strip() for x in args.family.split(",") if x.strip()) \
            if args.family else None
        _timed("power", lambda: run_power2(
            workers, m_targets=m_list, force=args.force,
            families=fam_list or None, resume=not args.no_resume))
    elif cmd == "law":
        law_path = RESULTS / "half_power_law.json"
        if not law_path.exists():
            raise FileNotFoundError(
                "half_power_law.json missing -- run 'confirm power' first")
        law_json = json.loads(law_path.read_text())
        _timed("law", lambda: run_law_tests(law_json, force=True))
    elif cmd == "bugs":
        _timed("bugs", lambda: run_bug2(workers, force=args.force))
    elif cmd == "pract":
        _timed("pract", lambda: run_pract_phase(workers, with_cell_b=args.cell_b,
                                                force=args.force))
    elif cmd == "secondary":
        arms = tuple(int(x) for x in args.arms.split(",") if x.strip())
        _timed("secondary", lambda: run_secondary_phase(workers, arms=arms,
                                                        force=args.force))
    elif cmd == "gates":
        _timed("gates", lambda: run_all_confirm_gates())
    elif cmd == "report":
        _timed("report", lambda: (assemble2(), write_readme2()))
    elif cmd == "all":
        seq = [
            ("anchors", "anchors", lambda: run_phase1()),
            ("null", "null", lambda: run_null2(workers)),
            ("fwer", "fwer", lambda: run_fwer_phase(workers)),
            ("power", "power", lambda: run_power2(workers)),
            ("law", "law", lambda: run_law_tests(
                json.loads((RESULTS / "half_power_law.json").read_text()), force=True)),
            ("bugs", "bugs", lambda: run_bug2(workers)),
            ("pract", "pract", lambda: run_pract_phase(workers,
                                                       with_cell_b=args.with_cell_b)),
            ("secondary", "secondary", lambda: run_secondary_phase(
                workers, arms=tuple(int(x) for x in args.arms.split(",")
                                    if x.strip()))),
            ("gates", "gates", lambda: run_all_confirm_gates()),
            ("report", "report", lambda: (assemble2(), write_readme2())),
        ]
        for phase, label, fn in seq:
            _timed(label, fn)


def _readme_summary() -> dict:
    null_t = json.loads((RESULTS / "null_calibration.json").read_text())
    power_t = json.loads((RESULTS / "power_surfaces.json").read_text())
    bug_t = json.loads((RESULTS / "bug_battery.json").read_text())
    return {
        "null_cells": [r for r in null_t["cells"] if not r.get("skipped")],
        "null_total": len(null_t["cells"]),
        "null_skipped": sum(1 for r in null_t["cells"] if r.get("skipped")),
        "power_cells": [r for r in power_t["cells"] if not r.get("skipped")],
        "power_total": len(power_t["cells"]),
        "power_skipped": sum(1 for r in power_t["cells"] if r.get("skipped")),
        "bug_cells": [r for r in bug_t["cells"] if not r.get("skipped")],
    }


if __name__ == "__main__":
    main()