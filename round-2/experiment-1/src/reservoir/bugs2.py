"""PHASE 6 -- blind bug battery scored in trials-to-90%-detection units.

Cell n=3000, k=150 (p=0.05); m in {5000, 10000}; n_reps=800; variants =
BUG_VARIANTS (recency_slot, drop_oldest, modulo_slot, float_threshold).

Two PROTOCOLS scored on the SAME replicates:
  (a) maxdev-only at alpha=0.05 (the mandated report);
  (b) 3-test protocol {maxdev, chi2, trend|z|} with joint rejection at the
      per-test Sidak level alpha' (FWER ~5%).

Trials-to-90%-detection m90 per (variant, protocol):
  * power(m=10000) >= 0.9  -> linear-in-probit interpolation between the
    m=5000 and m=10000 powers (probit vs log10 m);
  * power(m=5000)  >= 0.9  -> '<=5000' (with the power value);
  * otherwise -> probit(power) ~ a + b*log10(m) extrapolation, flagged
    'EXTRAPOLATED'.
float_threshold is the negative control (power ~= alpha; sanity gate G5).
Checkpoint: results/bug_battery_trials.json.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np
from loguru import logger
from scipy.stats import norm

from .config import BUG_VARIANTS, RESULTS, child_rng
from .confirm import (BONF_ALPHA, CONFIRM_K, CONFIRM_N, CONFIRM_P, N_BUG_REPS,
                      SIDAK_ALPHA, alpha_label)
from .confirm_null import load_confirm_thresholds
from .harness import _dump_json, run_parallel
from .samplers import algorithm_R_counts, validate_count_rows
from .stats import compute_stats, standardize_slopes

BUG2_M = (5000, 10000)
STATS = ("maxdev", "chi2", "trend_slope", "energy")


@dataclass
class _BugPayload:
    variant: str
    m: int
    thresholds: dict
    trend_ref: dict

    def key(self) -> str:
        return f"bug2_{self.variant}_{self.m}"


def _bug_worker(p: _BugPayload) -> dict:
    n, k = CONFIRM_N, CONFIRM_K
    mu = p.m * CONFIRM_P
    rng = child_rng(p.key())
    counts = algorithm_R_counts(n, k, p.m, N_BUG_REPS, rng, variant=p.variant)
    validate_count_rows(counts, p.m, k)
    stats = compute_stats(counts, mu)
    z = standardize_slopes(stats["trend_slope"], p.trend_ref["mu"], p.trend_ref["sd"])
    arrs = {"maxdev": stats["maxdev"], "chi2": stats["chi2"],
            "trend_slope": np.abs(z), "energy": stats["energy"]}

    # protocol (a): maxdev only at alpha=0.05
    pa = float(np.mean(arrs["maxdev"] > p.thresholds["maxdev"]["0.05"]))
    # protocol (b): joint rejection over {maxdev, chi2, trend|z|} at alpha'
    th = {s: p.thresholds[s]["sidak"] for s in ("maxdev", "chi2", "trend_slope")}
    joint = (arrs["maxdev"] > th["maxdev"]) | (arrs["chi2"] > th["chi2"]) | \
            (arrs["trend_slope"] > th["trend_slope"])
    pb = float(np.mean(joint))
    return {"variant": p.variant, "m": p.m, "n_reps": N_BUG_REPS,
            "power_maxdev_only_alpha05": pa, "power_3test_sidak": pb,
            "per_stat_power_at_sidak": {s: float(np.mean(arrs[s] > p.thresholds[s]["sidak"]))
                                        for s in STATS}}


def m90_from_two_points(m_lo: int, p_lo: float, m_hi: int, p_hi: float,
                        target: float = 0.9) -> dict:
    """Linear-in-probit interpolation between two budgets (log10 m scale)."""
    z_lo, z_hi = norm.ppf(min(max(p_lo, 1e-9), 1 - 1e-9)), norm.ppf(min(max(p_hi, 1e-9), 1 - 1e-9))
    z_t = norm.ppf(target)
    if abs(z_hi - z_lo) < 1e-12:
        return {"m90": float(m_hi), "method": "probit_interp_flat"}
    log10_m90 = np.log10(m_lo) + (z_t - z_lo) / (z_hi - z_lo) * (np.log10(m_hi) - np.log10(m_lo))
    return {"m90": float(10 ** log10_m90), "method": "probit_interp"}


def m90_two_point(variant: str, powers: dict[int, float]) -> dict:
    """m90 per the pre-registered rule from powers at m=5000 and m=10000.

    Order per plan PHASE 6.3: power(m=5000) >= 0.9 -> '<=5000' (both
    protocols saturate for gross bugs); elif power(m=10000) >= 0.9 ->
    probit interpolation between the two budgets; else probit extrapolation.
    """
    p5, p10 = powers.get(5000, 0.0), powers.get(10000, 0.0)
    if p5 >= 0.9:
        return {"m90": "<=5000", "method": "saturated_at_m5000",
                "power_at_m5000": p5, "power_at_m10000": p10}
    if p10 >= 0.9:
        interp = m90_from_two_points(5000, p5, 10000, p10)
        return {"m90": interp["m90"], "method": interp["method"],
                "power_at_m5000": p5, "power_at_m10000": p10}
    # extrapolation: probit(power) ~ a + b*log10(m)
    z5, z10 = norm.ppf(min(max(p5, 1e-9), 1 - 1e-9)), norm.ppf(min(max(p10, 1e-9), 1 - 1e-9))
    b = (z10 - z5) / (np.log10(10000) - np.log10(5000))
    log10_m90 = np.log10(5000) + (norm.ppf(0.9) - z5) / b if abs(b) > 1e-12 else np.nan
    return {"m90": float(10 ** log10_m90) if np.isfinite(log10_m90) else None,
            "method": "EXTRAPOLATED", "slope_probit_per_log10m": float(b),
            "power_at_m5000": p5, "power_at_m10000": p10}


def run_bug_phase(workers: int, m_list: tuple[int, ...] = BUG2_M,
                  force: bool = False) -> dict:
    out_path = RESULTS / "bug_battery_trials.json"
    if out_path.exists() and not force:
        logger.info(f"bug_battery_trials checkpoint exists ({out_path.name}); loading")
        return json.loads(out_path.read_text())

    thresholds = load_confirm_thresholds()
    payloads, per_m_null = [], {}
    for m in m_list:
        null = thresholds[f"confirm_null_m{m}"]
        per_m_null[m] = null
        for variant in BUG_VARIANTS:
            payloads.append(_BugPayload(variant=variant, m=m,
                                        thresholds=null["thresholds"],
                                        trend_ref=null["trend_z_reference"]))
    results = run_parallel(payloads, _bug_worker, workers, "confirm_bugs")

    by_var: dict[str, dict[int, dict]] = {}
    for r in results:
        if not r.get("skipped"):
            by_var.setdefault(r["variant"], {})[r["m"]] = r

    cells = []
    for variant in BUG_VARIANTS:
        rec = {"variant": variant}
        if variant not in by_var or len(by_var[variant]) < len(m_list):
            cells.append({"variant": variant, "skipped": True})
            continue
        for protocol, key_fn, power_key in (
                ("maxdev_only", lambda r: r["power_maxdev_only_alpha05"], "maxdev_only"),
                ("3test", lambda r: r["power_3test_sidak"], "3test")):
            powers = {m: key_fn(by_var[variant][m]) for m in m_list}
            m90 = m90_two_point(variant, powers)
            rec[f"m90_{protocol}"] = m90
            rec[f"power_{protocol}_per_m"] = powers
        # ratio: how much sooner the 3-test protocol detects vs maxdev-only
        a = rec["m90_maxdev_only"]; b = rec["m90_3test"]
        if isinstance(a, dict) and isinstance(b, dict):
            av = a["m90"]; bv = b["m90"]
            if isinstance(av, str) and isinstance(bv, str):
                rec["trials_ratio"] = "tie"
            elif isinstance(av, str):
                rec["trials_ratio"] = "maxdev_saturates_at_5000"
            elif isinstance(bv, str):
                rec["trials_ratio"] = "3test_saturates_at_5000"
            elif av and bv:
                rec["trials_ratio"] = av / bv
            else:
                rec["trials_ratio"] = None
        cells.append(rec)
        logger.info(f"[bugs2] {variant}: m90(maxdev-only)={rec['m90_maxdev_only']['m90']}, "
                    f"m90(3test)={rec['m90_3test']['m90']}, ratio={rec.get('trials_ratio')}")

    table = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "cell": "n=3000 k=150 p=0.05", "m_list": list(m_list),
             "n_reps": N_BUG_REPS, "protocols": {
                 "maxdev_only": "maxdev > alpha=0.05 threshold",
                 "3test": "any of {maxdev, chi2, trend|z|} > per-test Sidak threshold"},
             "negative_control_note": ("float_threshold is the inert off-by-epsilon "
                                       "bug; powers must sit at the protocol alpha"),
             "variants": cells}
    _dump_json(out_path, table)
    return table


if __name__ == "__main__":  # pragma: no cover
    logger.remove()
    logger.add(lambda _: None, level="INFO")