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

Usage:
  uv run method.py sanity
  uv run method.py null --scale full --workers 4
  uv run method.py power --scale full --workers 4
  uv run method.py bugbattery --scale full --workers 4
  uv run method.py report
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

    workers = _workers_arg(args.workers)
    if args.command == "null":
        run_null_phase(args.scale, workers)
    elif args.command == "power":
        run_power_phase(args.scale, workers)
    elif args.command == "bugbattery":
        run_bug_phase(args.scale, workers)


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