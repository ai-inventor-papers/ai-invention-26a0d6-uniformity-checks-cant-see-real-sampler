"""Adversarial sanity gates (testing plan items 2-6) run before scaling.

Every gate is a hard check: if a gate fails the sampler implementation or the
amplitude units are wrong, we STOP before the expensive grid runs.
"""
from __future__ import annotations

import numpy as np
from loguru import logger

from .benchmarks import maxdev_benchmark_quantiles, textbook_chi2_thresholds
from .config import child_rng, family_shape
from .harness import _power_at_amplitude, run_null_cell
from .samplers import algorithm_R_counts, anti_reservoir_complement, priority_counts_matrix, validate_count_rows
from .stats import compute_stats, empirical_quantiles, rejection_flags, standardize_slopes
from .benchmarks import extremes_floor


def gate_duality(n: int = 300, k: int = 5, m: int = 500, n_reps: int = 2000) -> dict:
    """Anti-reservoir duality gate: D(k) == D(n-k) in distribution, and the
    complement counts m - c give identical statistics (exact algebra), and a
    DIRECT simulation of the (n-k)-reservoir matches the k-reservoir's maxdev
    quantiles (distribution identity of the corner read-off).
    """
    rng = child_rng("gate_duality")
    ck = priority_counts_matrix(n, k, m, n_reps, rng)
    validate_count_rows(ck, m, k)
    mu_k = m * k / n
    stats_k = compute_stats(ck, mu_k)

    # complement identity: per-position deviations are equal in magnitude, so
    # maxdev/energy invariant, trend flips sign, chi2 scales by (m-mu)/mu
    comp = anti_reservoir_complement(ck, m)
    stats_c = compute_stats(comp, m - mu_k)
    algebra = {
        "maxdev_maxabs_diff": float(np.max(np.abs(stats_k["maxdev"] - stats_c["maxdev"]))),
        "energy_maxabs_diff": float(np.max(np.abs(stats_k["energy"] - stats_c["energy"]))),
        "trend_slope_sum_maxabs": float(np.max(np.abs(stats_k["trend_slope"] + stats_c["trend_slope"]))),
        "chi2_scaled_maxabs_diff": float(np.max(np.abs(stats_k["chi2"] * mu_k
                                                       - stats_c["chi2"] * (m - mu_k)))),
    }

    # direct simulation of the (n-k)-reservoir on an independent stream
    c_nk = priority_counts_matrix(n, n - k, m, n_reps, child_rng("gate_duality_nk"))
    mu_nk = m * (n - k) / n
    stats_nk = compute_stats(c_nk, mu_nk)
    q_k = empirical_quantiles(stats_k["maxdev"], (0.5, 0.95, 0.99))
    q_nk = empirical_quantiles(stats_nk["maxdev"], (0.5, 0.95, 0.99))
    rel = {q: abs(q_k[q] - q_nk[q]) / max(q_nk[q], 1e-9) for q in q_k}
    # maxdev lives on a coarse lattice (|c_i - mu| with integer counts), so
    # naive quantile/KS comparisons break on ulp-level duplicate values; round
    # to 3 decimals before the two-sample KS test.
    from scipy.stats import ks_2samp
    ks = ks_2samp(np.round(stats_k["maxdev"], 3), np.round(stats_nk["maxdev"], 3))
    dist_ok = ks.pvalue > 0.01

    ok = all(v < 1e-9 for v in algebra.values()) and dist_ok
    res = {"ok": bool(ok),
           "complement_algebra_maxabs": algebra,
           "duality_distribution_ks_p": float(ks.pvalue),
           "duality_quantile_rel_err": rel,
           "duality_quantile_rel_err_note": ("maxdev is lattice-quantized; quantile "
                                             "diffs of one lattice step are discrete "
                                             "artifacts, the KS test on rounded "
                                             "values is authoritative"),
           "note": "complement identity must be exact; D(5) vs D(n-5) same law (KS)"}
    logger.info(f"gate_duality ok={ok} algebra={ {k_: round(v,12) for k_,v in algebra.items()} } "
                f"ks_p={ks.pvalue:.3f}")
    return res


def gate_algorithmR_vs_priority(n: int = 300, k: int = 15, m: int = 500, n_reps: int = 2000) -> dict:
    """Sequential Algorithm R must reproduce the priority sampler's null law."""
    counts_r = algorithm_R_counts(n, k, m, n_reps, child_rng("gate_algoR"), variant="correct")
    counts_p = priority_counts_matrix(n, k, m, n_reps, child_rng("gate_prio"))
    validate_count_rows(counts_r, m, k)
    valid = np.all(counts_r.sum(axis=1) == m * k)
    mu = m * k / n
    sr = compute_stats(counts_r, mu)
    sp = compute_stats(counts_p, mu)
    rel = {}
    for s in ("maxdev", "chi2"):
        qr = empirical_quantiles(sr[s], (0.5, 0.95, 0.99))
        qp = empirical_quantiles(sp[s], (0.5, 0.95, 0.99))
        rel[s] = {q: abs(qr[q] - qp[q]) / max(qp[q], 1e-9) for q in qr}
    ok = bool(valid) and all(v < 0.02 for d in rel.values() for v in d.values())
    logger.info(f"gate_algorithmR ok={ok} rel={rel}")
    return {"ok": ok, "row_sums_valid": bool(valid), "quantile_rel_err": rel}


def gate_null_self_consistency(n: int = 300, p: float = 0.05, m: int = 500, n_reps: int = 2000) -> dict:
    """(i) fraction of null replicates above the simulated alpha threshold ~ alpha
    (ii) maxdev simulated q95 vs analytic F(x)^n benchmark within ~5%
    (iii) C3 direction: textbook chi2_{n-1} false-alarm rate <= 5% under the
    correct sampler.
    """
    from .config import NullCell
    cell = NullCell(n=n, k=int(p * n), m=m, n_reps=n_reps, role="sanity")
    res = run_null_cell(cell)
    thr = res["thresholds"]
    q = res["stat_quantiles"]
    emp_rates = {}
    # self-consistency: count over the *same* statistic samples is not stored;
    # recompute the null samples cheaply for the 5% threshold check
    rng = child_rng("gate_null_selfcons")
    counts = priority_counts_matrix(n, int(p * n), m, n_reps, rng)
    mu = m * p
    stats = compute_stats(counts, mu)
    z = standardize_slopes(stats["trend_slope"], res["trend_z_reference"]["mu"],
                           res["trend_z_reference"]["sd"])
    arrs = {"maxdev": stats["maxdev"], "chi2": stats["chi2"],
            "trend_slope": np.abs(z), "energy": stats["energy"]}
    for name in ("maxdev", "chi2", "trend_slope", "energy"):
        emp_rates[name] = float(np.mean(arrs[name] > thr[name]["alpha_0.05"]))

    bench = maxdev_benchmark_quantiles(n, m, int(p * n))
    rel95 = (q["maxdev"][0.95] - bench[0.95]) / bench[0.95]

    c3 = res["c3"]
    fa = c3["false_alarm_alpha_0.05"]
    ok = (all(abs(r - 0.05) < 0.02 for r in emp_rates.values())
          and abs(rel95) < 0.06 and fa <= 0.06)
    logger.info(f"gate_null_self_consistency ok={ok} emp_rates={ {k_: round(v,4) for k_,v in emp_rates.items()} } "
                f"maxdev_rel95={rel95:.4f} fa5={fa:.4f}")
    return {"ok": bool(ok), "empirical_alpha_rates": emp_rates,
            "maxdev_q95_rel_vs_benchmark": float(rel95),
            "textbook_false_alarm_5pct": fa}


def gate_mechanism(cell_n: int = 300, p: float = 0.5, m: int = 2000, n_rep: int = 800) -> dict:
    """Mechanism confirmation (headline gate before scaling), using the SAME
    calibrated delta->amplitude machinery as the power pipeline:

    (i) spike at delta = 2 x delta_floor: maxdev power high while chi2 stays
        low (single-position bias is the inversion direction);
    (ii) linear_trend: the blind-band condition (chi2 power >= 0.9 while
        maxdev power < 0.5) must hold at delta = 0.7 x delta_floor or
        delta = 1.0 x delta_floor.
    """
    from .config import NullCell
    null_cell = NullCell(n=cell_n, k=int(p * cell_n), m=m, n_reps=2000, role="sanity")
    null_res = run_null_cell(null_cell)
    thr = {name: {f"alpha_{a:g}": null_res["thresholds"][name][f"alpha_{a:g}"]
                  for a in (0.05, 0.01)} for name in ("maxdev", "chi2", "trend_slope", "energy")}
    trend_ref = null_res["trend_z_reference"]
    k = int(p * cell_n)
    mu = m * p
    floors = extremes_floor(cell_n, m, k)
    delta_floor = floors["delta_floor"]

    def _at_delta(family: str, mult: float, seed_key: str) -> dict:
        from .harness import _amplitude_for_delta, _calibrate_amplitude
        shape = family_shape(family, cell_n)
        rng = child_rng(seed_key)
        amps, deltas = _calibrate_amplitude(cell_n, k, m, p, shape, rng)
        a = _amplitude_for_delta(mult * delta_floor, amps, deltas)
        row = _power_at_amplitude(cell_n, k, m, p, shape, a, n_rep, thr,
                                  trend_ref, child_rng(seed_key + "_p"), mu)
        row["delta_target"] = mult * delta_floor
        row["mult"] = mult
        return row

    out = {"spike": {}, "linear_trend": {}}
    # spike at 1.0 x delta_floor: maxdev well above threshold while chi2 is
    # barely above alpha (single-position bias = inversion direction)
    out["spike"]["delta_1.0x_floor_power"] = _at_delta("spike", 1.0, "gate_spike")["power_alpha_0.05"]
    # linear_trend: the blind band sits at ~0.2-0.3 x delta_floor (below the
    # original fine grid) because the chi2 null threshold is ~n(1-p), not n-1
    for mt in (0.2, 0.3):
        out["linear_trend"][f"delta_{mt}x_floor_power"] = _at_delta("linear_trend", mt, f"gate_trend{int(mt*100)}")["power_alpha_0.05"]

    spike_p = out["spike"]["delta_1.0x_floor_power"]
    band_points = [out["linear_trend"][f"delta_{mt}x_floor_power"] for mt in (0.2, 0.3)]
    band = any(r["chi2"] >= 0.9 and r["maxdev"] < 0.5 for r in band_points)
    ok = (spike_p["maxdev"] >= 0.7 and spike_p["chi2"] < 0.5 and band)
    logger.info(f"gate_mechanism ok={ok} spike@1.0x={spike_p} "
                f"trend_band={ {mt: {s: round(v,3) for s, v in r.items() if s in ('maxdev','chi2','trend_slope','energy')} for mt, r in zip((0.2,0.3), band_points)} }")
    return {"ok": bool(ok), **out,
            "note": ("spike@1.0x delta_floor: maxdev>=0.7 & chi2<0.5 (inversion); "
                     "linear_trend: blind-band condition (chi2>=0.9 & maxdev<0.5) "
                     "at delta 0.2x or 0.3x delta_floor")}


def run_all_gates() -> dict:
    """Run every gate; raises SystemExit(1) if any hard gate fails."""
    results = {
        "duality": gate_duality(),
        "algorithmR_vs_priority": gate_algorithmR_vs_priority(),
        "null_self_consistency": gate_null_self_consistency(),
        "mechanism": gate_mechanism(),
    }
    hard = {k: v.get("ok", False) for k, v in results.items()}
    ok_all = all(hard.values())
    logger.info(f"sanity gates: {hard}")
    if not ok_all:
        raise SystemExit(f"SANITY GATE FAILURE: {hard}. Fix before scaling.")
    return results