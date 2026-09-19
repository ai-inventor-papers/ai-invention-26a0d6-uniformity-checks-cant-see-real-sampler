"""PHASE 7 -- practitioner deliverable: empirical maxdev null at m = 10^6.

Cell A (mandatory): n=1000, k=50, m=1_000_000, N=300 null replicates of the
correct priority sampler; 5 tour replicates each of the priority reservoir and
sequential Algorithm R.
Cell B (optional): n=3000, k=150, m=1_000_000, N=200 null replicates.

Deliverables per cell:
  * empirical alpha=0.05 maxdev null threshold (simulation);
  * analytic independent-Binomial benchmark quantiles (restricted x-grid per
    fallback F8: the max quantile lives near mu + O(sqrt(m p q log n))) and
    the relative error vs simulation;
  * per-sampler tour verdicts: maxdev observed vs threshold, ratio, PASS/FAIL;
  * the 10^6-trial budget itself is the practitioner-scale claim: expected
    maxdev magnitude ~ sqrt(2 p m log n) ~ 830 at n=1000.
Checkpoint: results/practitioner_null.json.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np
from loguru import logger
from scipy.stats import binom

from .benchmarks import extremes_floor
from .config import RESULTS, child_rng
from .confirm import N_PRACT_A, N_PRACT_B, N_PRACT_SAMP
from .harness import _dump_json, run_parallel, _safe_run
from .samplers import algorithm_R_counts, priority_counts_matrix, validate_count_rows
from .stats import compute_stats

M_BIG = 1_000_000
CELLS = (("A", 1000, 50, N_PRACT_A), ("B", 3000, 150, N_PRACT_B))
MAXDEV_TAUS = (0.95, 0.99)


def ib_benchmark_restricted(n_pos: int, m: int, k: int,
                            taus: tuple[float, ...] = MAXDEV_TAUS) -> dict[float, float]:
    """Independent-Binomial maxdev quantiles on a restricted x-grid (F8).

    P(max_i c_i <= x) = F(x)^n with c_i iid Binom(m, p).  The max quantile
    lives within a few sd of mu + sqrt(2 p m log n), so a dense grid over
    [mu-12sd, mu+12sd] covers it; CDF extrapolates to 0 / 1 outside.
    """
    p = k / n_pos
    mu = m * p
    sd = np.sqrt(m * p * (1.0 - p))
    margin = 12.0
    lo = max(0, int(np.floor(mu - margin * sd)))
    hi = min(m, int(np.ceil(mu + margin * sd)))
    x = np.arange(lo, hi + 1, dtype=np.float64)
    cdf = binom.cdf(x, m, p) ** n_pos
    out: dict[float, float] = {}
    for t in taus:
        if t >= 1.0:
            out[float(t)] = float(m - mu)
            continue
        idx = int(np.searchsorted(cdf, t, side="left"))
        idx = min(max(idx, 0), len(x) - 1)
        out[float(t)] = float(x[idx] - mu)
    return out


@dataclass
class _NullRepPayload:
    cell: str
    n: int
    k: int
    m: int
    rep_i: int
    null_N: int  # replicate index range base (seed uniquifier)


def _null_rep_worker(p: _NullRepPayload) -> dict:
    rng = child_rng(f"pract_null_{p.cell}_{p.rep_i}")
    counts = priority_counts_matrix(p.n, p.k, p.m, 1, rng)
    mu = p.m * p.k / p.n
    maxdev = float(compute_stats(counts, mu)["maxdev"][0])
    return {"cell": p.cell, "rep_i": p.rep_i, "maxdev": maxdev}


@dataclass
class _TourPayload:
    cell: str
    n: int
    k: int
    m: int
    sampler: str  # "priority" | "algorithm_R"
    rep_i: int


def _tour_worker(p: _TourPayload) -> dict:
    rng = child_rng(f"pract_tour_{p.cell}_{p.sampler}_{p.rep_i}")
    if p.sampler == "priority":
        counts = priority_counts_matrix(p.n, p.k, p.m, 1, rng)
    else:
        counts = algorithm_R_counts(p.n, p.k, p.m, 1, rng, variant="correct")
    mu = p.m * p.k / p.n
    maxdev = float(compute_stats(counts, mu)["maxdev"][0])
    return {"cell": p.cell, "sampler": p.sampler, "rep_i": p.rep_i, "maxdev": maxdev}


def run_cell(cell: str, n: int, k: int, null_N: int, workers: int) -> dict:
    null_jobs = [_NullRepPayload(cell, n, k, M_BIG, i, null_N) for i in range(null_N)]
    null_rows = [r for r in run_parallel(null_jobs, _null_rep_worker, workers,
                                         "pract_null") if not r.get("skipped")]
    maxdevs = np.array([r["maxdev"] for r in null_rows], dtype=np.float64)
    thr_alpha = float(np.quantile(maxdevs, 0.95))
    threshold_rec = {
        "threshold_alpha_0.05": thr_alpha,
        "n_null_reps": len(null_rows),
        "empirical_maxdev_mean": float(maxdevs.mean()),
        "empirical_q95": thr_alpha,
        "floor": extremes_floor(n, M_BIG, k),
        "expected_magnitude_note": f"~ sqrt(2 p m log n) = "
                                   f"{extremes_floor(n, M_BIG, k)['A_floor']:.0f}",
    }
    bench = ib_benchmark_restricted(n, M_BIG, k)
    threshold_rec["ib_benchmark_q95"] = bench[0.95]
    threshold_rec["rel_error_sim_vs_ib"] = \
        (thr_alpha - bench[0.95]) / bench[0.95] if bench[0.95] > 0 else None

    tours = []
    for sampler in ("priority", "algorithm_R"):
        jobs = [_TourPayload(cell, n, k, M_BIG, sampler, i) for i in range(N_PRACT_SAMP)]
        rows = [r for r in run_parallel(jobs, _tour_worker, workers, "pract_tour")
                if not r.get("skipped")]
        dv = np.array([r["maxdev"] for r in rows], dtype=np.float64)
        tours.append({
            "sampler": sampler, "n_tour_reps": len(rows),
            "per_rep_maxdev": [float(v) for v in dv],
            "mean_maxdev": float(dv.mean()), "max_maxdev": float(dv.max()),
            "ratio_maxdev_over_threshold": float(dv.mean() / thr_alpha),
            "verdict": "PASS" if float(dv.mean()) <= thr_alpha else "FAIL",
        })
        logger.info(f"[pract] {cell}/{sampler}: mean maxdev={dv.mean():.1f} "
                    f"vs threshold {thr_alpha:.1f} -> "
                    f"{'PASS' if dv.mean() <= thr_alpha else 'FAIL'}")
    return {"cell": cell, "n": n, "k": k, "m": M_BIG,
            "null_threshold": threshold_rec, "tours": tours}


def run_pract_phase(workers: int, with_cell_b: bool = True, force: bool = False) -> dict:
    out_path = RESULTS / "practitioner_null.json"
    if out_path.exists() and not force:
        logger.info(f"practitioner_null checkpoint exists ({out_path.name}); loading")
        return json.loads(out_path.read_text())
    cells_out = []
    for cell, n, k, null_N in CELLS:
        if cell == "B" and not with_cell_b:
            continue
        cells_out.append(run_cell(cell, n, k, null_N, workers))
    table = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "m": M_BIG, "cells": cells_out,
             "note": ("10^6-trial budget is the practitioner-scale claim; pass "
                      "verdict vs the simulated alpha=5% threshold and the "
                      "independent-Binomial benchmark for reference")}
    _dump_json(out_path, table)
    return table


if __name__ == "__main__":  # pragma: no cover
    logger.remove()
    logger.add(lambda _: None, level="INFO")