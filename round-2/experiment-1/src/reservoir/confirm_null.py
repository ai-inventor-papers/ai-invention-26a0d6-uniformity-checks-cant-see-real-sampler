"""PHASES 2-3 -- confirm-cell null calibration and corrected-protocol FWER.

PHASE 2 (deliverable: thresholds_confirm.json):
  per m in {2000, 5000, 10000} at n=3000, p=0.05:
    * N=2500 correct-priority null replicates (seed 'confirm_null_<m>')
    * the four statistics; trend standardized with the null mean/sd
    * empirical quantiles (0.5/0.95/0.99/0.999) and per-stat thresholds at
      alphas (0.05, alpha'=1-0.95^(1/3), alpha/3, 0.01)
    * C1 maxdev benchmark quantiles + relative errors

PHASE 3 (deliverable: fwer_check.json):
  per m: N=3000 INDEPENDENT null replicates (fresh seed 'fwer_<m>'); joint
  rejection of the 3-test protocol {maxdev, chi2, trend|z|} at per-test alpha'
  (Sidak) -> FWER_sidak, and at alpha/3 -> FWER_bonf; marginal per-stat rates.
  energy is measured but NOT part of the protocol.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass

import numpy as np
from loguru import logger

from .benchmarks import maxdev_benchmark_quantiles
from .config import QC_QUANTILES, RESULTS, child_rng
from .confirm import (ALPHA_KEYS, BONF_ALPHA, CONFIRM_ALPHAS, CONFIRM_K, CONFIRM_M,
                      CONFIRM_N, CONFIRM_P, N_FWER_REPS, N_NULL_REPS, SIDAK_ALPHA,
                      alpha_label, err_se)
from .harness import _dump_json, run_parallel, _safe_run
from .samplers import priority_counts_matrix, validate_count_rows
from .stats import compute_stats, empirical_quantiles, standardize_slopes

STATS = ("maxdev", "chi2", "trend_slope", "energy")
PROTOCOL_STATS = ("maxdev", "chi2", "trend_slope")  # energy measured, not used


# ---------------------------------------------------------------------------
# Phase 2 -- null calibration
# ---------------------------------------------------------------------------
NULL_CHUNKS = 5          # parallel chunks per m (each chunk: N_NULL_REPS/5 reps)
FWER_CHUNKS = 6          # parallel chunks per m for the phase-3 FWER batch

def _null_chunk_worker(args: tuple[int, int, int]) -> dict:
    """(m, part, chunk_reps) -> four statistic arrays for that chunk.

    Seeds are derived per (m, part): worker-count independent, so the null
    batch reproduces exactly whatever worker pool it runs on.
    """
    m, part, chunk_reps = args
    n, p = CONFIRM_N, CONFIRM_P
    k = int(p * n)
    mu = m * p
    rng = child_rng(f"confirm_null_{m}_part{part}")
    counts = priority_counts_matrix(n, k, m, chunk_reps, rng)
    validate_count_rows(counts, m, k)
    stats = compute_stats(counts, mu)
    return {"maxdev": stats["maxdev"], "chi2": stats["chi2"],
            "trend_slope": stats["trend_slope"], "energy": stats["energy"],
            "chunk_reps": chunk_reps}


def _gather_null_stats(m: int, n_reps: int, chunks: int) -> dict[str, np.ndarray]:
    """Full-batch statistic arrays for one m, computed in parallel chunks."""
    chunk_reps = -(-n_reps // chunks)          # ceiling so we cover n_reps
    args = [(m, part, chunk_reps) for part in range(chunks)]
    rows = [r for r in run_parallel(args, _null_chunk_worker, max(2, min(chunks, 8)),
                                    "confirm_null_chunk") if not r.get("skipped")]
    if not rows:
        raise RuntimeError(f"all null chunks failed for m={m}")
    n_got = sum(r["chunk_reps"] for r in rows)
    out = {s: np.concatenate([r[s] for r in rows]) for s in STATS}
    return out, n_got


def run_confirm_null_cell(m: int, n_reps: int = N_NULL_REPS) -> dict:
    n, p = CONFIRM_N, CONFIRM_P
    k = int(p * n)
    mu = m * p
    samples, n_got = _gather_null_stats(m, n_reps, NULL_CHUNKS)

    trend_ref = {"mu": float(np.mean(samples["trend_slope"])),
                 "sd": float(np.std(samples["trend_slope"], ddof=1))}
    z = standardize_slopes(samples["trend_slope"], trend_ref["mu"], trend_ref["sd"])
    stat_samples = {"maxdev": samples["maxdev"], "chi2": samples["chi2"],
                    "trend_slope": np.abs(z), "energy": samples["energy"]}

    quantiles = {name: empirical_quantiles(stat_samples[name], QC_QUANTILES)
                 for name in STATS}
    thresholds: dict[str, dict[str, float]] = {}
    for name in STATS:
        thresholds[name] = {alpha_label(a): float(np.quantile(stat_samples[name], 1.0 - a))
                            for a in CONFIRM_ALPHAS}

    c1 = {"benchmark_Fn_quantiles": {f"q{t:g}": maxdev_benchmark_quantiles(n, m, k)[t]
                                     for t in maxdev_benchmark_quantiles(n, m, k)},
          "relative_error_sim_vs_bench": {}, "benchmark_note": None}
    bench = maxdev_benchmark_quantiles(n, m, k)
    c1["benchmark_Fn_quantiles"] = {f"q{t:g}": v for t, v in bench.items()}
    c1["relative_error_sim_vs_bench"] = {
        f"q{t:g}": ((quantiles["maxdev"][t] - bench[t]) / bench[t]) if bench[t] > 0 else None
        for t in bench}
    c1["benchmark_note"] = ("F(x)^n tracks the max-side only; the within-trial "
                            "negative dependence shrinks the max, so sim is "
                            "typically slightly below the benchmark; context row "
                            "for the null-law accuracy, not a deliverable.")

    return {"cell": f"confirm_null_m{m}", "n": n, "k": k, "p": p, "m": m, "mu": mu,
            "n_reps": n_got, "chunked": True, "n_chunks": NULL_CHUNKS,
            "seed_scheme": f"confirm_null_{m}_part{{i}}",
            "trend_z_reference": trend_ref,
            "stat_quantiles": quantiles, "thresholds": thresholds, "c1": c1}


def run_null_phase(workers: int, m_list: tuple[int, ...] | None = None,
                   force: bool = False) -> dict:
    if m_list is None:
        m_list = CONFIRM_M
    out_path = RESULTS / "null_calibration_confirm.json"
    thr_path = RESULTS / "thresholds_confirm.json"
    if out_path.exists() and not force:
        logger.info(f"null_confirm checkpoint exists ({out_path.name}); loading")
        table = json.loads(out_path.read_text())
        return table
    results = run_parallel([m for m in m_list], run_confirm_null_cell, workers,
                           "confirm_null")
    table = {"scale": "confirm", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "cells": results}
    _dump_json(out_path, table)
    thresholds = {r["cell"]: r for r in results if not r.get("skipped")}
    _dump_json(thr_path, thresholds)
    for r in results:
        if not r.get("skipped"):
            logger.info(f"  m={r['m']}: maxdev thr a0.05={r['thresholds']['maxdev']['0.05']:g}, "
                        f"chi2 thr a0.05={r['thresholds']['chi2']['0.05']:g}, "
                        f"sidak maxdev thr={r['thresholds']['maxdev']['sidak']:g}")
    return table


def load_confirm_thresholds() -> dict[str, dict]:
    p = RESULTS / "thresholds_confirm.json"
    if not p.exists():
        raise FileNotFoundError("thresholds_confirm.json missing -- run confirm null first")
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# Phase 3 -- FWER of the corrected 3-test protocol
# ---------------------------------------------------------------------------
def _fwer_chunk_worker(args: tuple[int, int, int]) -> dict:
    """(m, part, chunk_reps) -> statistic arrays of an INDEPENDENT null batch.

    Seeds are fresh per (m, part) ('fwer_<m>_part<part>'), independent of the
    phase-2 calibration batch ('confirm_null_<m>_part<part>').
    """
    m, part, chunk_reps = args
    n, p = CONFIRM_N, CONFIRM_P
    k = int(p * n)
    mu = m * p
    rng = child_rng(f"fwer_{m}_part{part}")
    counts = priority_counts_matrix(n, k, m, chunk_reps, rng)
    validate_count_rows(counts, m, k)
    stats = compute_stats(counts, mu)
    return {"maxdev": stats["maxdev"], "chi2": stats["chi2"],
            "trend_slope": stats["trend_slope"], "energy": stats["energy"],
            "chunk_reps": chunk_reps}


def _fwer_worker(m: int) -> dict:
    n, p = CONFIRM_N, CONFIRM_P
    k = int(p * n)
    mu = m * p
    null = load_confirm_thresholds()[f"confirm_null_m{m}"]
    thr = null["thresholds"]
    trend_ref = null["trend_z_reference"]
    chunk_reps = -(-N_FWER_REPS // FWER_CHUNKS)
    args = [(m, part, chunk_reps) for part in range(FWER_CHUNKS)]
    rows = [r for r in run_parallel(args, _fwer_chunk_worker,
                                    max(2, min(FWER_CHUNKS, 8)), "confirm_fwer_chunk")
            if not r.get("skipped")]
    if not rows:
        raise RuntimeError(f"all fwer chunks failed for m={m}")
    n_tot = sum(r["chunk_reps"] for r in rows)
    stats = {s: np.concatenate([r[s] for r in rows]) for s in STATS}
    z = standardize_slopes(stats["trend_slope"], trend_ref["mu"], trend_ref["sd"])
    arrs = {"maxdev": stats["maxdev"], "chi2": stats["chi2"],
            "trend_slope": np.abs(z), "energy": stats["energy"]}

    def joint(alpha: float) -> tuple[float, dict[str, float]]:
        flags = {s: arrs[s] > thr[s][alpha_label(alpha)] for s in PROTOCOL_STATS}
        any3 = np.logical_or(np.logical_or(flags["maxdev"], flags["chi2"]), flags["trend_slope"])
        marg = {s: float(flags[s].mean()) for s in PROTOCOL_STATS}
        marg["energy"] = float(np.mean(arrs["energy"] > thr["energy"][alpha_label(alpha)]))
        return float(any3.mean()), marg

    fwer_sidak, marg_sidak = joint(SIDAK_ALPHA)
    fwer_bonf, marg_bonf = joint(BONF_ALPHA)
    se = err_se(n_tot)
    return {"m": m, "N": n_tot, "chunked": True, "n_chunks": FWER_CHUNKS,
            "seed_scheme": f"fwer_{m}_part{{i}}",
            "fwer_sidak": fwer_sidak, "fwer_bonf": fwer_bonf,
            "sidak_to_0.05_ratio": fwer_sidak / 0.05,
            "bonf_to_0.05_ratio": fwer_bonf / 0.05,
            "binomial_se": se,
            "sidak_within_2se": abs(fwer_sidak - 0.05) <= 2 * se,
            "bonf_within_2se": abs(fwer_bonf - 0.05) <= 2 * se,
            "marginal_sidak": marg_sidak, "marginal_bonf": marg_bonf,
            "marginal_sidak_note": ("sanity: each marginal ~= alpha' = 0.01695 "
                                    "within binomial noise sqrt(alpha'*(1-alpha')/3000)"),
            }


def run_fwer_phase(workers: int, m_list: tuple[int, ...] | None = None,
                   force: bool = False) -> dict:
    if m_list is None:
        m_list = CONFIRM_M
    out_path = RESULTS / "fwer_check.json"
    if out_path.exists() and not force:
        logger.info(f"fwer checkpoint exists ({out_path.name}); loading")
        return json.loads(out_path.read_text())
    results = run_parallel([m for m in m_list], _fwer_worker, workers, "confirm_fwer")
    table = {"scale": "confirm", "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "protocol": "3-test {maxdev, chi2, trend|z|}, energy measured but not used",
             "cells": results}
    _dump_json(out_path, table)
    for r in results:
        if not r.get("skipped"):
            logger.info(f"  m={r['m']}: FWER_sidak={r['fwer_sidak']:.4f} "
                        f"(ratio {r['sidak_to_0.05_ratio']:.3f}), "
                        f"FWER_bonf={r['fwer_bonf']:.4f}")
    return table


if __name__ == "__main__":  # pragma: no cover
    import sys
    from loguru import logger as _l
    _l.remove(); _l.add(sys.stdout, level="INFO")