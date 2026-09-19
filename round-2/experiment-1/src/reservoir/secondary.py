"""PHASE 8 -- secondary arms: n- and p-invariance of the design law.

Arm 1 (mandatory): n=10000, p=0.05 (k=500), linear_trend, m in {2000, 5000}.
  null thresholds (N=1500, alpha=0.05) + power grid for maxdev and chi2
  (6 mults around the m-invariant predictions from the n=10000 iteration-1
  anchor, n_fine=800) -> kappa_hat(m) vs derived 322x headline.

Arm 2 (optional): n=3000, p=0.5 (k=1500), linear_trend, m in {2000, 5000}
  (same recipe; kappa_hat vs the iteration-1 n=3000 p=0.5 derived value).

The law says kappa = 1/(A_acc_over_delta_floor)^2 is m- and n-invariant in
floor-relative units, so measuring at NEW budgets confirms the invariance.
Checkpoint: results/secondary_arms.json.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass

import numpy as np
from loguru import logger

from .anchors import STAT_DISPLAY
from .benchmarks import extremes_floor
from .config import RESULTS, child_rng, family_shape
from .confirm import alpha_label
from .harness import (_amplitude_for_delta, _calibrate_amplitude, _delta_of_counts,
                      _dump_json, run_parallel)
from .samplers import priority_counts_matrix, validate_count_rows
from .stats import compute_stats, empirical_quantiles, power_from_flags, rejection_flags, \
    standardize_slopes

ARMS = (
    # Pre-registered PHASE 8 recipe restored on 2026-09-19 (a stale-session
    # trim to m=2000-only / n_null=450 / n_fine=400 was reverted).
    {"arm": 1, "n": 10000, "p": 0.05, "m_list": (2000, 5000), "family": "linear_trend",
     "n_null": 1500, "n_fine": 800, "anchor_cell": "power_n10000_p0.05_m1000"},
    {"arm": 2, "n": 3000, "p": 0.5, "m_list": (2000, 5000), "family": "linear_trend",
     "n_null": 1500, "n_fine": 800, "anchor_cell": "power_n3000_p0.5_m2000"},
)
MULT_STEPS = (0.4, 1.0, 1.7)
MDEV_STEPS = (0.5, 1.0, 1.7)
DEDUPE = 0.05
SEC_NULL_CHUNKS = 3     # null batches chunked for pool parallelism


def _null_chunk(cell: tuple) -> dict:
    """One parallel null chunk.  run_parallel maps worker_fn(cell) with a
    SINGLE argument, so the (arm, m, chunk_reps, part) tuple is unpacked here
    (same convention as confirm_null._null_chunk_worker; a 4-positional
    signature made every chunk fail with TypeError on the spawn pool)."""
    arm, m, chunk_reps, part = cell
    n, k = arm["n"], int(arm["p"] * arm["n"])
    mu = m * arm["p"]
    rng = child_rng(f"sec_null_{arm['arm']}_{m}_part{part}")
    counts = priority_counts_matrix(n, k, m, chunk_reps, rng)
    validate_count_rows(counts, m, k)
    stats = compute_stats(counts, mu)
    return {"maxdev": stats["maxdev"], "chi2": stats["chi2"],
            "trend_slope": stats["trend_slope"], "energy": stats["energy"]}


def _null_for(arm: dict, m: int, workers: int = 4) -> dict:
    """Null batch for one (arm, m): threshold + trend reference, computed in
    SEC_NULL_CHUNKS parallel chunks (seeds per chunk -> worker-count
    independent)."""
    stats = {"maxdev": [], "chi2": [], "trend_slope": [], "energy": []}
    args = [(arm, m, -(-arm["n_null"] // SEC_NULL_CHUNKS), part)
            for part in range(SEC_NULL_CHUNKS)]
    rows = [r for r in run_parallel(args, _null_chunk, workers, "sec_null")
            if not r.get("skipped")]
    if not rows:
        raise RuntimeError(f"all sec-null chunks failed for arm {arm['arm']} m={m}")
    arrs_raw = {s: np.concatenate([r[s] for r in rows]) for s in stats}
    trend_ref = {"mu": float(np.mean(arrs_raw["trend_slope"])),
                 "sd": float(np.std(arrs_raw["trend_slope"], ddof=1))}
    z = standardize_slopes(arrs_raw["trend_slope"], trend_ref["mu"], trend_ref["sd"])
    arrs = {"maxdev": arrs_raw["maxdev"], "chi2": arrs_raw["chi2"],
            "trend_slope": np.abs(z), "energy": arrs_raw["energy"]}
    thresholds = {s: float(np.quantile(arrs[s], 0.95)) for s in arrs}  # alpha=0.05
    return {"thresholds": thresholds, "trend_ref": trend_ref,
            "stat_quantiles": {s: empirical_quantiles(arrs[s], (0.5, 0.95, 0.99, 0.999))
                               for s in arrs}}


@dataclass
class _SecPowerPayload:
    arm: int
    n: int
    k: int
    m: int
    family: str
    mult: float
    a: float
    n_fine: int
    thresholds: dict
    trend_ref: dict

    def key(self) -> str:
        return f"sec_power_{self.arm}_{self.m}_{self.mult:.6g}"


def _sec_power_worker(p: _SecPowerPayload) -> dict:
    p0 = p.k / p.n
    mu = p.m * p0
    rng = child_rng(p.key())
    counts = priority_counts_matrix(p.n, p.k, p.m, p.n_fine, rng,
                                    shape=family_shape(p.family, p.n), a=p.a)
    validate_count_rows(counts, p.m, p.k)
    stats = compute_stats(counts, mu)
    z = standardize_slopes(stats["trend_slope"], p.trend_ref["mu"], p.trend_ref["sd"])
    arrs = {"maxdev": stats["maxdev"], "chi2": stats["chi2"],
            "trend_slope": np.abs(z), "energy": stats["energy"]}
    th = {s: (0.0, p.thresholds[s]) for s in arrs}
    powers = power_from_flags(rejection_flags(arrs, th))
    return {"arm": p.arm, "m": p.m, "mult": p.mult, "a": p.a, "powers": powers,
            "achieved_delta": _delta_of_counts(counts[:100], p.m, p0)}


def _crossing(rows: list[dict], stat: str) -> float | None:
    xs = np.array([r["mult"] for r in rows], dtype=np.float64)
    ys = np.array([r["powers"][stat] for r in rows], dtype=np.float64)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    for i in range(len(xs) - 1):
        if (ys[i] - 0.5) * (ys[i + 1] - 0.5) <= 0 and ys[i] != ys[i + 1]:
            t = (0.5 - ys[i]) / (ys[i + 1] - ys[i])
            return float(np.exp(np.log(xs[i]) + t * (np.log(xs[i + 1]) - np.log(xs[i]))))
    return None


def run_arm(arm: dict, anchor: dict, workers: int) -> dict:
    n, k, p0 = arm["n"], int(arm["p"] * arm["n"]), arm["p"]
    family = arm["family"]
    Aacc_mult = anchor["A_acc_over_delta_floor"]
    mdev_mult = anchor["mdev_mult_anchor"] or anchor["anchors"]["maxdev"]["mult"]
    kappa_derived = anchor["kappa_derived"]
    calib = _calibrate_amplitude(n, k, 2000, p0, family_shape(family, n),
                                 child_rng(f"sec_calib_{arm['arm']}"))
    per_m = {}
    for m in arm["m_list"]:
        null = _null_for(arm, m, workers=workers)
        delta_floor = extremes_floor(n, m, k)["delta_floor"]
        acc = [Aacc_mult * s for s in MULT_STEPS]
        mdev = [mdev_mult * s for s in MDEV_STEPS]
        mults = sorted(set(round(v, 10) for v in acc + mdev))
        kept = [v for i, v in enumerate(mults)
                if i == 0 or v / mults[i - 1] > 1.0 + DEDUPE]
        jobs = []
        for mt in kept:
            a_t = _amplitude_for_delta(mt * delta_floor, *calib)
            if a_t < 1e-6:
                continue
            jobs.append(_SecPowerPayload(arm=arm["arm"], n=n, k=k, m=m, family=family,
                                         mult=mt, a=a_t, n_fine=arm["n_fine"],
                                         thresholds=null["thresholds"],
                                         trend_ref=null["trend_ref"]))
        rows = [r for r in run_parallel(jobs, _sec_power_worker, workers,
                                        "sec_power") if not r.get("skipped")]
        chi_mult = _crossing(rows, "chi2")
        mdev_mult_meas = _crossing(rows, "maxdev")
        per_m[m] = {
            "m": m, "delta_floor": delta_floor,
            "kappa_hat": (1.0 / chi_mult ** 2) if chi_mult else None,
            "chi2_half_mult": chi_mult, "maxdev_half_mult": mdev_mult_meas,
            "rows": rows, "null_quantiles": null["stat_quantiles"],
        }
        logger.info(f"[sec arm{arm['arm']}] m={m}: chi2 mult={chi_mult}, "
                    f"kappa_hat={per_m[m]['kappa_hat']}")
    return {"arm": arm["arm"], "n": n, "p": p0, "family": family,
            "kappa_derived": kappa_derived,
            "per_m": {str(m): {k2: v for k2, v in per_m[m].items() if k2 != "rows"}
                      for m in arm["m_list"]},
            "power_rows": {str(m): per_m[m]["rows"] for m in arm["m_list"]},
            "kappa_hat": {str(m): per_m[m]["kappa_hat"] for m in arm["m_list"]},
            "m_invariance_spread": None,
            "note": ("law says kappa is m-invariant; kappa_hat on the two budgets "
                     "vs kappa_derived from iteration-1")}


def run_secondary_phase(workers: int, arms: tuple[int, ...] = (1,),
                        force: bool = False) -> dict:
    out_path = RESULTS / "secondary_arms.json"
    if out_path.exists() and not force:
        logger.info(f"secondary_arms checkpoint exists ({out_path.name}); loading")
        return json.loads(out_path.read_text())
    anchors = json.loads((RESULTS / "iter1_anchors.json").read_text())["cells"]
    cells = []
    for arm_cfg in ARMS:
        if arm_cfg["arm"] not in arms:
            continue
        anch = anchors[arm_cfg["anchor_cell"]]["linear_trend"]
        cells.append(run_arm(arm_cfg, anch, workers))
    table = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "arms": cells}
    _dump_json(out_path, table)
    return table


if __name__ == "__main__":  # pragma: no cover
    logger.remove()
    logger.add(lambda _: None, level="INFO")