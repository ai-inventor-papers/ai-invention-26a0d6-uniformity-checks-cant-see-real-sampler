"""Shared execution layer: null cells, thresholds, and the power pipeline.

Parallelism: cells are independent, so the drivers fan cells out to a
ProcessPoolExecutor using the SPAWN start method (fork deadlocks with loguru).
Seeds are deterministic per cell key, so results do not depend on worker
count or scheduling order.
"""
from __future__ import annotations

import gc
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from loguru import logger

from .config import (BENCH_ALPHAS, BUG_VARIANTS, CALIB_AMPS, CALIB_REPS,
                     COARSE_DELTAS, COARSE_EXTEND, FAMILIES, FINE_DELTAS,
                     FINE_DELTAS_BIG, GAP_FAMILIES, MAX_COARSE_MULT,
                     POWER_ALPHAS, QC_QUANTILES, RESULTS, child_rng,
                     family_shape)
from .benchmarks import (c3_reference, extremes_floor,
                         maxdev_benchmark_quantiles, textbook_chi2_thresholds)
from .samplers import (algorithm_R_counts, anti_reservoir_complement,
                       priority_counts_matrix, validate_count_rows)
from .stats import (empirical_quantiles, power_from_flags, rejection_flags,
                    compute_stats, standardize_slopes)

STAT_NAMES = ("maxdev", "chi2", "trend_slope", "energy")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))
    logger.debug(f"wrote {path}")


def run_parallel(cells, worker_fn, workers: int, phase: str) -> list[dict]:
    """Map worker_fn over cells with a spawn-context pool; F2 resilience.

    worker_fn(cell) -> dict.  Failures are logged, recorded as skipped cells
    (so method_out.json still validates), and do not abort the run.
    """
    results: list[dict] = []
    if workers <= 1 or len(cells) <= 1:
        for cell in cells:
            results.append(_safe_run(worker_fn, cell, phase))
        return results
    import multiprocessing as mp

    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(_safe_run, worker_fn, cell, phase): cell for cell in cells}
        for fut in futures:
            cell = futures[fut]
            try:
                res = fut.result()
            except Exception:
                logger.exception(f"worker crashed for cell {cell}")
                res = {"cell": getattr(cell, "key", lambda: str(cell))(), "skipped": True}
            results.append(res)
    return results


def _safe_run(worker_fn, cell, phase: str) -> dict:
    key = getattr(cell, "key", lambda: str(cell))()
    t0 = time.time()
    try:
        res = worker_fn(cell)
        res.setdefault("cell", key)
        res["wall_s"] = round(time.time() - t0, 2)
        res["skipped"] = False
        logger.info(f"[{phase}] {key} done in {res['wall_s']}s")
        return res
    except MemoryError:
        logger.error(f"[{phase}] {key} MemoryError; recorded as skipped (F2)")
    except Exception:
        logger.exception(f"[{phase}] {key} failed; recorded as skipped (F2)")
    return {"cell": key, "skipped": True, "wall_s": round(time.time() - t0, 2)}


# ---------------------------------------------------------------------------
# Null cell pipeline (PHASE 3: calibration a, C1 benchmark b, C3 variance c)
# ---------------------------------------------------------------------------
def run_null_cell(cell) -> dict:
    """Simulate the null, compute the four statistics, and derive:
    empirical quantiles + alpha thresholds per statistic; C1 benchmark
    quantiles and relative errors for maxdev; C3 chi2 variance ratio and the
    CORRECT-sampler false-alarm rate at the textbook chi2_{n-1} thresholds.
    """
    n, k, m, n_reps = cell.n, cell.k, cell.m, cell.n_reps
    p = k / n
    mu = m * p
    rng = child_rng(cell.key())

    # corner cells: simulate the k=5 reservoir only; the k=n-5 corner is the
    # anti-reservoir complement (exact), used for the benchmark comparison.
    counts = priority_counts_matrix(n, k, m, n_reps, rng)
    validate_count_rows(counts, m, k)
    stats = compute_stats(counts, mu)

    # trend standardization reference (null mean/sd of the raw slope)
    trend_ref = {"mu": float(np.mean(stats["trend_slope"])),
                 "sd": float(np.std(stats["trend_slope"], ddof=1))}
    z = standardize_slopes(stats["trend_slope"], trend_ref["mu"], trend_ref["sd"])
    stat_samples = {"maxdev": stats["maxdev"], "chi2": stats["chi2"],
                    "trend_slope": np.abs(z), "energy": stats["energy"]}

    # (a) empirical quantiles and alpha thresholds
    thresholds: dict[str, dict[str, float]] = {}
    quantiles: dict[str, dict[str, float]] = {}
    for name in STAT_NAMES:
        quantiles[name] = empirical_quantiles(stat_samples[name], QC_QUANTILES)
        thresholds[name] = {f"alpha_{a:g}": float(np.quantile(stat_samples[name], 1.0 - a))
                            for a in POWER_ALPHAS}
        thresholds[name]["alpha_0.001"] = float(np.quantile(stat_samples[name], 0.999))

    # (b) C1 benchmark for maxdev
    bench = maxdev_benchmark_quantiles(n, m, k)
    sim = quantiles["maxdev"]
    c1 = {
        "benchmark_Fn_quantiles": {f"q{t:g}": bench[t] for t in bench},
        "relative_error_sim_vs_bench": {
            f"q{t:g}": (sim[t] - bench[t]) / bench[t] if bench[t] > 0 else None
            for t in bench
        },
        "benchmark_note": ("F(x)^n is an approximation: it tracks the max-side only "
                           "(maxdev also has a min-side event, |mu - min c_i|, which "
                           "pushes maxdev's quantiles up) and ignores the negative "
                           "within-trial dependence (which shrinks the max and pushes "
                           "them down). sim vs bench therefore agree within a few "
                           "percent but the sign is not guaranteed."),
    }
    # duality consistency: the complement (n-k)-reservoir counts are m - c; the
    # per-position deviations are identical in magnitude, so
    #   maxdev, energy, |trend| are invariant, and chi2 scales by mu_dual/mu
    # (chi2's denominator is mu, which differs between the k and n-k corners).
    comp = anti_reservoir_complement(counts, m)
    mu_dual = m - mu
    stats_comp = compute_stats(comp, mu_dual)
    c1["duality"] = {
        "maxdev_maxabs_diff": float(np.max(np.abs(stats["maxdev"] - stats_comp["maxdev"]))),
        "energy_maxabs_diff": float(np.max(np.abs(stats["energy"] - stats_comp["energy"]))),
        "trend_slope_sum_maxabs": float(np.max(np.abs(stats["trend_slope"] + stats_comp["trend_slope"]))),
        "chi2_scaled_maxabs_diff": float(np.max(np.abs(stats["chi2"] * mu - stats_comp["chi2"] * mu_dual))),
    }
    c1["duality_ok"] = all(v < 1e-9 for v in c1["duality"].values())

    # (c) C3 chi2 variance + textbook false-alarm rate
    var_true = float(np.var(stats["chi2"], ddof=1))
    var_closed, c3_src = c3_reference(n, m, k, rng)
    mean_chi2 = float(np.mean(stats["chi2"]))
    n_chi2 = n - 1
    c3 = {
        "chi2_sample_var": var_true,
        "product_binomial_var": var_closed,
        "c3_source": c3_src,
        "variance_ratio_true_vs_product_binomial": var_true / var_closed,
        "chi2_df_var": 2.0 * n_chi2,
        "variance_ratio_true_vs_chi2df": var_true / (2.0 * n_chi2),
        "chi2_sample_mean": mean_chi2,
        "chi2_theory_mean_under_independence": float(n * (1.0 - p)),
        "textbook_chi2_95": textbook_chi2_thresholds(n)[0.05],
    }
    for a in BENCH_ALPHAS:
        thr = textbook_chi2_thresholds(n, (a,))[a]
        c3[f"false_alarm_alpha_{a:g}"] = float(np.mean(stats["chi2"] > thr))

    # extreme-corner read-off: simulate k = 5 only; the k = n - 5 corner comes
    # from the anti-reservoir duality (complement counts; maxdev/energy/|trend|
    # identical samples; chi2 sample scaled by mu/mu_dual -- exact).
    dual_corner = None
    if cell.role == "corner" and (n - k) > 0:
        mu_dual = m * (n - k) / n
        # complement counts c' = m - c have deviations (c' - mu_dual) =
        # -(c - mu), so chi2' = sum dev^2 / mu_dual = chi2 * mu / mu_dual
        # (mu_dual > mu for corner k=5, so chi2' < chi2).
        scale = mu / mu_dual
        bench_dual = maxdev_benchmark_quantiles(n, m, n - k)
        sim_dual = quantiles["maxdev"]
        dual_corner = {
            "k_dual": n - k, "p_dual": (n - k) / n, "mu_dual": mu_dual,
            "maxdev_quantiles": quantiles["maxdev"],
            "chi2_quantiles_scaled": {f"q{q:g}": quantiles["chi2"][q] * scale
                                      for q in QC_QUANTILES},
            "energy_quantiles": quantiles["energy"],
            "trend_abs_z_quantiles": quantiles["trend_slope"],
            "thresholds": {
                "alpha_0.05": {s: (thresholds[s]["alpha_0.05"] if s != "chi2"
                                   else thresholds["chi2"]["alpha_0.05"] * scale)
                               for s in STAT_NAMES},
                "alpha_0.01": {s: (thresholds[s]["alpha_0.01"] if s != "chi2"
                                   else thresholds["chi2"]["alpha_0.01"] * scale)
                               for s in STAT_NAMES},
            },
            "benchmark_maxdev_Fn": bench_dual,
            "benchmark_maxdev_rel_error": {
                f"q{t:g}": ((sim_dual[t] - bench_dual[t]) / bench_dual[t])
                if bench_dual[t] > 0 else None for t in bench_dual
            },
            "read_off_note": ("k=n-5 corner read exactly from the k=5 simulation "
                              "via the anti-reservoir complement duality D(n-5)=D(5)"),
        }

    return {
        "cell": cell.key(), "n": n, "k": k, "m": m, "p": p, "mu": mu,
        "n_reps": n_reps, "role": cell.role,
        "trend_z_reference": trend_ref,
        "stat_quantiles": quantiles,
        "thresholds": thresholds,
        "c1": c1,
        "c3": c3,
        "dual_corner": dual_corner,
    }


# ---------------------------------------------------------------------------
# Power pipeline (PHASE 4: measure M)
# ---------------------------------------------------------------------------
def _delta_of_counts(counts: np.ndarray, m: int, p: float) -> float:
    """max_i |p_i - p| / p from pooled replicate counts p_i = sum c_i/(N*m)."""
    n = counts.shape[1]
    phat = counts.sum(axis=0) / (counts.shape[0] * m)
    return float(np.max(np.abs(phat - p)) / p)


def _calibrate_amplitude(n: int, k: int, m: int, p: float, shape: np.ndarray,
                         rng: np.random.Generator) -> tuple[list[float], list[float]]:
    """Empirical delta(a) map at a few calibration amplitudes (plan: ~linear)."""
    amps, deltas = [], []
    for a in CALIB_AMPS:
        counts = priority_counts_matrix(n, k, m, CALIB_REPS, rng, shape=shape, a=a)
        deltas.append(_delta_of_counts(counts, m, p))
        amps.append(a)
    return amps, deltas


def _amplitude_for_delta(target_d: float, amps: list[float], deltas: list[float]) -> float:
    """Invert the empirical delta(a) map via linear interpolation."""
    order = np.argsort(deltas)
    ds = np.array(deltas)[order]
    as_ = np.array(amps)[order]
    if target_d <= ds[0]:
        return float(as_[0] * target_d / max(ds[0], 1e-12))
    if target_d >= ds[-1]:
        # linear extrapolation from the last two points
        slope = (as_[-1] - as_[-2]) / max(ds[-1] - ds[-2], 1e-12)
        return float(as_[-1] + slope * (target_d - ds[-1]))
    return float(np.interp(target_d, ds, as_))


def _power_at_amplitude(n: int, k: int, m: int, p: float, shape: np.ndarray, a: float,
                        n_rep: int, thresholds: dict, trend_ref: dict,
                        rng: np.random.Generator, mu: float) -> dict:
    """Power of all four statistics at one amplitude; also achieved delta."""
    counts = priority_counts_matrix(n, k, m, n_rep, rng, shape=shape, a=a)
    validate_count_rows(counts, m, k)
    stats = compute_stats(counts, mu)
    z = standardize_slopes(stats["trend_slope"], trend_ref["mu"], trend_ref["sd"])
    stat_arrs = {"maxdev": stats["maxdev"], "chi2": stats["chi2"],
                 "trend_slope": np.abs(z), "energy": stats["energy"]}
    out = {"a": a, "delta": _delta_of_counts(counts[: min(100, n_rep)], m, p)}
    for alph in POWER_ALPHAS:
        th = {name: (0.0, thresholds[name][f"alpha_{alph:g}"]) for name in STAT_NAMES}
        flags = rejection_flags(stat_arrs, th)
        out[f"power_alpha_{alph:g}"] = power_from_flags(flags)
    return out


def run_power_cell(cell, null_res: dict) -> dict:
    """Full power pipeline for one (n, p, m) cell: calibration -> coarse pass
    (A_acc + blind-band scan) -> fine pass at smart amplitudes -> read-offs.
    """
    n, p, m = cell.n, cell.p, cell.m
    k = int(p * n)
    mu = m * p
    thresholds = {name: {f"alpha_{a:g}": null_res["thresholds"][name][f"alpha_{a:g}"]
                         for a in POWER_ALPHAS} for name in STAT_NAMES}
    trend_ref = null_res["trend_z_reference"]
    floors = extremes_floor(n, m, k)
    delta_floor = floors["delta_floor"]

    rng = child_rng(cell.key())
    families_out: dict[str, dict] = {}
    blind_bands: list[dict] = []
    for family in FAMILIES:
        shape = family_shape(family, n)
        fam_key = f"{cell.key()}_{family}"
        fam_rng = child_rng(fam_key)
        cal_amps, cal_deltas = _calibrate_amplitude(n, k, m, p, shape, fam_rng)

        # --- coarse pass (N = n_coarse) ---
        coarse_mults = list(COARSE_DELTAS)
        coarse = []
        max_mult = max(coarse_mults)
        guard = 0
        while max_mult <= MAX_COARSE_MULT and guard <= len(COARSE_EXTEND):
            # only the newly appended mult is fresh; earlier ones were already
            # computed in a previous loop iteration (avoid duplicate work)
            new_mults = coarse_mults if guard == 0 else [coarse_mults[-1]]
            for mt in new_mults:
                d_t = delta_floor * mt
                a_t = _amplitude_for_delta(d_t, cal_amps, cal_deltas)
                if a_t < 1e-6:
                    continue
                row = _power_at_amplitude(n, k, m, p, shape, a_t, cell.n_coarse,
                                          thresholds, trend_ref, fam_rng, mu)
                row["delta_target"] = d_t
                row["mult"] = d_t / delta_floor
                coarse.append(row)
            # adaptive extension until chi2 power crosses 0.5 (gap trend)
            chi2_pows = [r["power_alpha_0.05"]["chi2"] for r in coarse]
            if max(chi2_pows) >= 0.5 or max_mult >= MAX_COARSE_MULT:
                break
            max_mult = COARSE_EXTEND[guard]
            coarse_mults.append(COARSE_EXTEND[guard])
            guard += 1

        # A_acc: delta where chi2 power crosses 0.5 (linear interpolation)
        a_acc = _find_crossing(coarse, "chi2")

        # --- fine pass at smart amplitude points ---
        # at n = 1e4 a reduced 5-point sweep is used (F1 trim, see config)
        fine_deltas = FINE_DELTAS_BIG if n >= 10000 else FINE_DELTAS
        fine_mults = sorted(set(list(fine_deltas) + ([a_acc / delta_floor] if a_acc else [])))
        fine = []
        for mult in fine_mults:
            if mult < 0.05:
                continue
            d_t = delta_floor * mult
            a_t = _amplitude_for_delta(d_t, cal_amps, cal_deltas)
            if a_t < 1e-6:
                continue
            row = _power_at_amplitude(n, k, m, p, shape, a_t, cell.n_fine,
                                      thresholds, trend_ref, fam_rng, mu)
            row["delta_target"] = d_t
            row["mult"] = d_t / delta_floor
            fine.append(row)

        # --- A_acc fallback: downward log-bisection (gap families only) ---
        # At small n the chi2 crossing sits below the coarse grid's lowest
        # point (mult 0.5 x floor), so _find_crossing returns None and the
        # gap-vs-n table would be empty.  For the gap families we bracket the
        # crossing on a log-mult grid and bisect twice at coarse N.
        a_acc_method = "coarse_interp"
        a_acc_probe: list[dict] = []
        if family in GAP_FAMILIES and a_acc is None:
            def _probe(mt: float) -> dict | None:
                d_t = delta_floor * mt
                a_t = _amplitude_for_delta(d_t, cal_amps, cal_deltas)
                if a_t < 1e-6:
                    return None
                row = _power_at_amplitude(n, k, m, p, shape, a_t, cell.n_coarse,
                                          thresholds, trend_ref, fam_rng, mu)
                row["delta_target"] = d_t
                row["mult"] = mt
                return row

            mults = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0]
            probes: list[dict] = []
            lo_row: dict | None = None
            for mt in mults:
                row = _probe(mt)
                if row is None:
                    continue
                probes.append(row)
                pw = row["power_alpha_0.05"]["chi2"]
                if pw < 0.5:
                    lo_row = row
                elif lo_row is not None:
                    hi_row = row
                    for _ in range(2):  # log-mult bisection, coarse N
                        mid = float(np.sqrt(lo_row["mult"] * hi_row["mult"]))
                        mrow = _probe(mid)
                        if mrow is None:
                            break
                        probes.append(mrow)
                        if mrow["power_alpha_0.05"]["chi2"] < 0.5:
                            lo_row = mrow
                        else:
                            hi_row = mrow
                    ll, lh = np.log(lo_row["mult"]), np.log(hi_row["mult"])
                    pl = lo_row["power_alpha_0.05"]["chi2"]
                    ph = hi_row["power_alpha_0.05"]["chi2"]
                    t = (0.5 - pl) / max(ph - pl, 1e-12)
                    a_acc = float(delta_floor * np.exp(ll + t * (lh - ll)))
                    a_acc_method = "probe_bisection"
                    break
                else:
                    break  # saturated already at the smallest mult: no crossing
            a_acc_probe = probes

        points = coarse + fine
        # blind band: maxdev power < 0.5 while max(chi2, trend, energy) >= 0.95
        bb = _blind_band(points)
        if bb is not None:
            blind_bands.append({"cell": cell.key(), "family": family, **bb})

        families_out[family] = {
            "delta_floor": delta_floor,
            "A_floor": floors["A_floor"],
            "A_acc_delta": a_acc,
            "A_acc_over_delta_floor": (a_acc / delta_floor) if a_acc else None,
            "A_acc_method": a_acc_method,
            "A_acc_probe_pass": a_acc_probe,
            "coarse_pass": coarse,
            "fine_pass": fine,
            "calibration": {"amps": cal_amps, "deltas": cal_deltas},
        }
        gc.collect()

    # read-offs
    gap = {}
    for family in GAP_FAMILIES:
        fam = families_out.get(family)
        if fam and fam["A_acc_over_delta_floor"] is not None:
            gap[family] = {
                "n": n, "p": p, "m": m,
                "A_acc_over_delta_floor": fam["A_acc_over_delta_floor"],
                "trials_gap_ratio_sq": fam["A_acc_over_delta_floor"] ** 2,
            }
    spike = families_out.get("spike")
    spike_inversion = None
    if spike:
        rows = [{"mult": r["mult"], "delta": r["delta"],
                 "maxdev_power": r["power_alpha_0.05"]["maxdev"],
                 "chi2_power": r["power_alpha_0.05"]["chi2"]} for r in spike["fine_pass"]]
        spike_inversion = {"cell": cell.key(),
                           "note": "expect maxdev >= chi2 (single-position bias)",
                           "rows": rows}

    return {
        "cell": cell.key(), "n": n, "p": p, "m": m, "k": k,
        "delta_floor": delta_floor, "A_floor": floors["A_floor"],
        "families": families_out, "blind_bands": blind_bands,
        "gap_vs_n": gap, "spike_inversion": spike_inversion,
    }


def _find_crossing(points: list[dict], stat: str) -> float | None:
    """Delta at which the 5% power of ``stat`` crosses 0.5 (linear interp)."""
    xs = [r["delta_target"] for r in points]
    ys = [r["power_alpha_0.05"][stat] for r in points]
    if not xs:
        return None
    order = np.argsort(xs)
    xs = np.array(xs)[order]
    ys = np.array(ys)[order]
    for i in range(len(xs) - 1):
        if (ys[i] - 0.5) * (ys[i + 1] - 0.5) <= 0 and ys[i] != ys[i + 1]:
            t = (0.5 - ys[i]) / (ys[i + 1] - ys[i])
            return float(xs[i] + t * (xs[i + 1] - xs[i]))
    return None


def _blind_band(points: list[dict]) -> dict | None:
    """Contiguous delta-interval where maxdev power < 0.5 at 5% while
    max(chi2, trend, energy) power >= 0.95 at 5%; midpoint powers recorded.
    """
    rows = sorted(points, key=lambda r: r["delta_target"])
    band: list[dict] = []
    for r in rows:
        p05 = r["power_alpha_0.05"]
        other = max(p05["chi2"], p05["trend_slope"], p05["energy"])
        if p05["maxdev"] < 0.5 and other >= 0.95:
            band.append(r)
    if not band:
        return None
    lo = band[0]["delta_target"]
    hi = band[-1]["delta_target"]
    mid = band[len(band) // 2]["power_alpha_0.05"]
    return {
        "delta_lo": lo, "delta_hi": hi, "width": hi - lo,
        "n_band_points": len(band),
        "midpoint_power_alpha_0.05": mid,
    }


# ---------------------------------------------------------------------------
# Bug battery (PHASE 5)
# ---------------------------------------------------------------------------
def run_bug_cell(cell, null_res: dict) -> dict:
    """Per-bug power of all four statistics using the Phase-3 null thresholds."""
    n, k, m, n_reps = cell.n, cell.k, cell.m, cell.n_reps
    p = k / n
    mu = m * p
    thresholds = {name: {f"alpha_{a:g}": null_res["thresholds"][name][f"alpha_{a:g}"]
                             for a in POWER_ALPHAS} for name in STAT_NAMES}
    trend_ref = null_res["trend_z_reference"]
    bugs_out: dict[str, dict] = {}
    for variant in BUG_VARIANTS:
        rng = child_rng(f"{cell.key()}_{variant}")
        counts = algorithm_R_counts(n, k, m, n_reps, rng, variant=variant)
        validate_count_rows(counts, m, k)
        stats = compute_stats(counts, mu)
        z = standardize_slopes(stats["trend_slope"], trend_ref["mu"], trend_ref["sd"])
        stat_arrs = {"maxdev": stats["maxdev"], "chi2": stats["chi2"],
                     "trend_slope": np.abs(z), "energy": stats["energy"]}
        powers = {}
        for alph in POWER_ALPHAS:
            th = {name: (0.0, thresholds[name][f"alpha_{alph:g}"]) for name in STAT_NAMES}
            powers[f"power_alpha_{alph:g}"] = power_from_flags(rejection_flags(stat_arrs, th))
        # bias profile: mean relative deviation per position (family linkage)
        dev_profile = (counts.mean(axis=0) - mu) / mu
        bugs_out[variant] = {"powers": powers,
                             "max_abs_bias": float(np.max(np.abs(dev_profile))),
                             "profile": dev_profile.tolist()}
    # family linkage: correlation of each bug's deviation profile with the
    # five family shape vectors (a signed, mean-centered Pearson r)
    linkage = {}
    centered = {f: family_shape(f, n) for f in FAMILIES}
    for variant, rec in bugs_out.items():
        prof = np.asarray(rec["profile"])
        corrs = {}
        for f, shp in centered.items():
            a, b = prof - prof.mean(), shp - shp.mean()
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            corrs[f] = float(a @ b / denom) if denom > 0 else 0.0
        linkage[variant] = corrs
    return {"cell": cell.key(), "n": n, "k": k, "m": m, "bugs": bugs_out,
            "family_linkage_corr": linkage}