"""PHASE 9 -- confirm-iteration sanity gates (G1-G5).

G1 anti-reservoir duality D(k) = D(n-k): same-run complement counts m-c must
   give maxdev/energy identical, chi2*mu invariant, slope sign flip
   (max-abs-diff < 1e-9) -- n=3000, k=150, m=500, N=2500.
G2 sampler equivalence: Algorithm R correct vs priority -- empirical maxdev
   0.95-quantiles within 5% (n=3000, m=2000, N=1500).
G3 null self-consistency at alpha: rejection rate of the PHASE-2 null batch at
   its own alpha=0.05 threshold ~= alpha within +-3 SE (per m).
G4 reproducibility anchor: m=2000 half-power vs iteration-1 anchors within 10%
   (computed in confirm_power.run_power_phase; reported here from the
   sanity_g4_anchor.json checkpoint).
G5 negative control: float_threshold power ~= protocol alpha within +-3 SE
   (read from bug_battery_trials.json).
Checkpoint: results/sanity_gates.json.  Any FAIL -> SystemExit.
"""
from __future__ import annotations

import json
import time

import numpy as np
from loguru import logger

from .config import RESULTS, child_rng
from .confirm import CONFIRM_FAMILIES, CONFIRM_M, G2_Q95_TOL, G4_ANCHOR_TOL, err_se
from .harness import run_parallel
from .samplers import (algorithm_R_counts, anti_reservoir_complement,
                       priority_counts_matrix, validate_count_rows)
from .stats import compute_stats, empirical_quantiles, standardize_slopes

G3_CHECK_REPS = 500      # independent null-check batch size per m (G3; SE band
                         # +-2.9pp, adequate against the +-3 SE consistency)


def gate_duality(n: int = 3000, k: int = 150, m: int = 500, n_reps: int = 2500) -> dict:
    rng = child_rng("sanity2_duality")
    ck = priority_counts_matrix(n, k, m, n_reps, rng)
    mu_k = m * k / n
    sk = compute_stats(ck, mu_k)
    comp = anti_reservoir_complement(ck, m)
    sc = compute_stats(comp, m - mu_k)
    algebra = {
        "maxdev_maxabs_diff": float(np.max(np.abs(sk["maxdev"] - sc["maxdev"]))),
        "energy_maxabs_diff": float(np.max(np.abs(sk["energy"] - sc["energy"]))),
        "trend_slope_sum_maxabs": float(np.max(np.abs(sk["trend_slope"] + sc["trend_slope"]))),
        "chi2_scaled_maxabs_diff": float(np.max(np.abs(sk["chi2"] * mu_k
                                                       - sc["chi2"] * (m - mu_k)))),
    }
    ok = all(v < 1e-9 for v in algebra.values())
    logger.info(f"[G1] duality ok={ok} algebra={ {k_: round(v, 12) for k_, v in algebra.items()} }")
    return {"ok": bool(ok), "complement_algebra_maxabs": algebra}


def gate_algorithmR_vs_priority(n: int = 3000, k: int = 150, m: int = 2000,
                                n_reps: int = 1500) -> dict:
    cr = algorithm_R_counts(n, k, m, n_reps, child_rng("sanity2_algoR"), variant="correct")
    cp = priority_counts_matrix(n, k, m, n_reps, child_rng("sanity2_prio"))
    validate_count_rows(cr, m, k)
    mu = m * k / n
    qr = empirical_quantiles(compute_stats(cr, mu)["maxdev"], (0.5, 0.95, 0.99))
    qp = empirical_quantiles(compute_stats(cp, mu)["maxdev"], (0.5, 0.95, 0.99))
    rel95 = abs(qr[0.95] - qp[0.95]) / qp[0.95]
    ok = bool(rel95 <= G2_Q95_TOL)
    logger.info(f"[G2] Algorithm R vs priority ok={ok} rel95={rel95:.4f}")
    return {"ok": ok, "maxdev_q95_rel_err": float(rel95), "q95_algoR": qr[0.95],
            "q95_priority": qp[0.95]}


def _g3_batch(m: int) -> dict:
    """Independent null batch (fresh seed) tested at the stored alpha=0.05
    thresholds; runs as a pool job so the three m-batches are parallel."""
    th = json.loads((RESULTS / "thresholds_confirm.json").read_text())
    nulls = json.loads((RESULTS / "null_calibration_confirm.json").read_text())
    r = next(x for x in nulls["cells"] if not x.get("skipped") and x["m"] == m)
    n, k, p = r["n"], r["k"], r["p"]
    se = err_se(G3_CHECK_REPS) * 3
    rng = child_rng(f"sanity2_nullcheck_{m}")
    counts = priority_counts_matrix(n, k, m, G3_CHECK_REPS, rng)
    mu = m * p
    arrs = compute_stats(counts, mu)
    z = standardize_slopes(arrs["trend_slope"], r["trend_z_reference"]["mu"],
                           r["trend_z_reference"]["sd"])
    samples = {"maxdev": arrs["maxdev"], "chi2": arrs["chi2"],
               "trend_slope": np.abs(z), "energy": arrs["energy"]}
    emp = {s: float(np.mean(samples[s] > th[f"confirm_null_m{m}"]["thresholds"][s]["0.05"]))
           for s in samples}
    ok = all(abs(v - 0.05) <= se for v in emp.values())
    logger.info(f"[G3] m={m} ok={ok} rates={ {k_: round(v, 4) for k_, v in emp.items()} }")
    return {"m": m, "n_check_reps": G3_CHECK_REPS, "se3": se,
            "empirical_alpha_rates": emp, "ok": bool(ok)}


def gate_null_self_consistency() -> dict:
    nulls = json.loads((RESULTS / "null_calibration_confirm.json").read_text())
    res = [r for r in run_parallel([x["m"] for x in nulls["cells"]
                                    if not x.get("skipped")],
                                   _g3_batch, 3, "confirm_g3") if not r.get("skipped")]
    ok_all = bool(res) and all(r["ok"] for r in res)
    return {"ok": bool(ok_all), "per_m": res}


def gate_g4() -> dict:
    p = RESULTS / "sanity_g4_anchor.json"
    if not p.exists():
        return {"ok": False, "note": "sanity_g4_anchor.json missing (run power phase)"}
    g4 = json.loads(p.read_text())
    logger.info(f"[G4] reproducibility anchor ok={g4.get('ok')}")
    return g4


def gate_g5() -> dict:
    p = RESULTS / "bug_battery_trials.json"
    if not p.exists():
        return {"ok": False, "note": "bug_battery_trials.json missing (run bugs2)"}
    data = json.loads(p.read_text())
    se = 3 * err_se(800)
    out = []
    for v in data["variants"]:
        if v.get("variant") != "float_threshold" or v.get("skipped"):
            continue
        # JSON round-trip strings the m keys; restore ints
        pa = {int(k): val for k, val in v["power_maxdev_only_per_m"].items()}
        pb = {int(k): val for k, val in v["power_3test_per_m"].items()}
        fwer = json.loads((RESULTS / "fwer_check.json").read_text())
        ref = {r["m"]: r["fwer_sidak"] for r in fwer["cells"] if not r.get("skipped")}
        rows = []
        for m in sorted(pa):
            ok_a = abs(pa[m] - 0.05) <= se
            ok_b = abs(pb[m] - ref.get(m, 0.05)) <= se
            rows.append({"m": m, "power_maxdev_only": pa[m], "power_3test": pb[m],
                         "ref_fwer_sidak": ref.get(m), "ok_maxdev_only": ok_a,
                         "ok_3test": ok_b})
        ok = all(r["ok_maxdev_only"] and r["ok_3test"] for r in rows)
        out.append({"variant": "float_threshold", "ok": ok, "rows": rows})
        logger.info(f"[G5] float_threshold ok={ok} rows={rows}")
    ok_all = bool(out) and all(r["ok"] for r in out)
    return {"ok": ok_all, "check": out}


_GATE_FNS = {
    "G1_duality": gate_duality,
    "G2_algorithmR_vs_priority": gate_algorithmR_vs_priority,
    "G3_null_self_consistency": gate_null_self_consistency,
    "G4_repro_anchor_m2000": gate_g4,
    "G5_negative_control": gate_g5,
}


def run_all_confirm_gates(use_cache: bool = True) -> dict:
    """Run G1-G5; each gate's record is cached to results/sanity_gate_<name>.json
    so expensive gates (G2, G3) can be spread across foreground invocations."""
    gates: dict = {}
    for name, fn in _GATE_FNS.items():
        cache_p = RESULTS / f"sanity_gate_{name}.json"
        if use_cache and cache_p.exists():
            gates[name] = json.loads(cache_p.read_text())
            logger.info(f"[gates] {name} loaded from cache")
            continue
        gates[name] = fn()
        cache_p.write_text(json.dumps(gates[name], indent=1))
    gates["wall_s"] = 0.0
    ok_all = all(g.get("ok", False) for g in gates.values()
                 if isinstance(g, dict) and "ok" in g)
    logger.info(f"confirm sanity gates: { {k: g.get('ok') for k, g in gates.items() if isinstance(g, dict)} }")
    if not ok_all:
        raise SystemExit(f"CONFIRM SANITY GATE FAILURE: {gates}")
    (RESULTS / "sanity_gates.json").write_text(json.dumps(gates, indent=1))
    return gates


if __name__ == "__main__":  # pragma: no cover
    logger.remove()
    logger.add(lambda _: None, level="INFO")