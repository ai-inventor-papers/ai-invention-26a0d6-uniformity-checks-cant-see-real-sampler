"""PHASE 5 -- pre-registered law tests (P1/P2/P3) and verdict.

Reads results/half_power_law.json (crossings per (m, family, stat, alpha)) and
results/iter1_anchors.json (kappa_derived) and computes, verbatim per
results/pre_registration.json:

  P1 (maxdev m-invariance in floor-relative units): max pairwise relative
     spread of maxdev mult over m in {2000, 5000, 10000} <= 15%.
  P2 (accumulated m^(-1/2)): LS slope of log(A_half_delta_chi2) vs log(m) in
     [-0.55, -0.45]; trend and energy slopes reported likewise.
  P3 (kappa m-invariance + reproduction): kappa_hat(m) = 1/mult_chi2(m)^2;
     CONFIRMED iff |kappa_hat - kappa_derived|/kappa_derived <= 0.25 for every
     m in {5000, 10000} AND spread of kappa_hat over {2000,5000,10000} <= 25%.

Verdict per family (CONFIRMED / PARTIALLY_CONFIRMED / MEASURED_LAW), overall =
worst over the two families.  Written into results/half_power_law.json under
'verdict'.
"""
from __future__ import annotations

import json
import time

import numpy as np
from loguru import logger

from .config import RESULTS
from .confirm import (CONFIRM_FAMILIES, CONFIRM_M, P1_MAX_SPREAD_TOL,
                      P2_SLOPE_RANGE, P3_KAPPA_REL_TOL, P3_KAPPA_SPREAD_TOL,
                      VERDICT_ORDER)
from .anchors import STAT_DISPLAY

SPREAD_STATS = "maxdev"
KAPPA_STAT_ALPHA = "0.05"


def pairwise_spread(values: list[float]) -> float:
    """max pairwise relative spread: (max-min)/mean (requires len >= 2)."""
    if len(values) < 2:
        return 0.0
    v = np.asarray(values, dtype=np.float64)
    return float((v.max() - v.min()) / v.mean())


def ls_slope(xs: list[float], ys: list[float]) -> float:
    x = np.log(np.asarray(xs, dtype=np.float64))
    y = np.log(np.asarray(ys, dtype=np.float64))
    return float(np.polyfit(x, y, 1)[0])


def run_law_tests(half_power: dict, force: bool = False) -> dict:
    out_path = RESULTS / "half_power_law.json"
    if out_path.exists() and not force:
        law = json.loads(out_path.read_text())
        if law.get("verdict"):
            logger.info("half_power_law.json already has a verdict; loading")
            return law
        half_power = law

    anchors = json.loads((RESULTS / "iter1_anchors.json").read_text())
    # NOTE: cells are keyed by (m, family) -- two families share the same m,
    # so an m-only key would silently alias families (both families reading
    # the exp_recency cell was observed as a spurious PARTIALLY_CONFIRMED).
    moms = {(r.get("m"), r.get("family")): r for r in half_power["cells"]
            if not r.get("skipped")}
    families_out: dict[str, dict] = {}

    for fam in CONFIRM_FAMILIES:
        fam_cells = {m: moms[(m, fam)] for m in CONFIRM_M if (m, fam) in moms}
        if len(fam_cells) < 3:
            logger.error(f"[law] family {fam}: only {len(fam_cells)} budgets present")
            families_out[fam] = {"skipped": True, "missing_m": [m for m in CONFIRM_M
                                                                if m not in fam_cells]}
            continue
        m_list = sorted(fam_cells)
        # ---- P1: maxdev mult spread ----
        mdev_mults = [fam_cells[m]["crossings"]["maxdev"]["0.05"].get("mult")
                      for m in m_list]
        if any(v is None for v in mdev_mults):
            p1 = {"pass": False, "note": "maxdev crossing missing at some m"}
            p1_values = [{"m": m, "mult": v} for m, v in zip(m_list, mdev_mults)]
        else:
            spread = pairwise_spread(mdev_mults)  # type: ignore[arg-type]
            p1 = {"pass": bool(spread <= P1_MAX_SPREAD_TOL), "spread": spread,
                  "tolerance": P1_MAX_SPREAD_TOL}
            p1_values = [{"m": m, "mult": v} for m, v in zip(m_list, mdev_mults)]

        # ---- P2: log-log slope of A_half_delta vs m ----
        p2 = {}
        for s in ("chi2", "trend_slope", "energy"):
            deltas = [fam_cells[m]["crossings"][s]["0.05"].get("A_half_delta")
                      for m in m_list]
            if any(v is None or v <= 0 for v in deltas):
                p2[s] = {"slope": None, "pass": False, "note": "crossing missing"}
                continue
            slope = ls_slope(m_list, deltas)  # type: ignore[arg-type]
            p2[s] = {"slope": slope,
                     "pass": bool(P2_SLOPE_RANGE[0] <= slope <= P2_SLOPE_RANGE[1])
                     if s == "chi2" else bool(P2_SLOPE_RANGE[0] <= slope <= P2_SLOPE_RANGE[1]),
                     "range": list(P2_SLOPE_RANGE), "m": m_list,
                     "A_half_deltas": deltas}

        # ---- P3: kappa_hat reproduction and m-invariance ----
        kappas = [fam_cells[m].get("kappa_hat") for m in m_list]
        kappa_derived = anchors["cells"]["power_n3000_p0.05_m2000"][fam]["kappa_derived"]
        if any(v is None for v in kappas) or kappa_derived is None:
            p3 = {"pass": False, "note": "kappa_hat missing (chi2 crossing absent)"}
        else:
            rel_errs = {m: abs(kappas[i] - kappa_derived) / kappa_derived
                        for i, m in enumerate(m_list)}  # type: ignore[arg-type]
            kappa_spread = pairwise_spread(kappas)  # type: ignore[arg-type]
            repro_ok = all(rel_errs[m] <= P3_KAPPA_REL_TOL for m in (5000, 10000)
                           if m in rel_errs)
            invar_ok = kappa_spread <= P3_KAPPA_SPREAD_TOL
            p3 = {"kappa_hat_per_m": {str(m): kappas[i] for i, m in enumerate(m_list)},
                  "kappa_derived": kappa_derived,
                  "rel_err_per_m": {str(m): rel_errs[m] for m in m_list},
                  "kappa_spread": kappa_spread,
                  "reproduction_ok_5000_10000": repro_ok,
                  "invariance_ok": invar_ok,
                  "pass": bool(repro_ok and invar_ok),
                  "tolerances": {"kappa_rel_tol": P3_KAPPA_REL_TOL,
                                 "kappa_spread_tol": P3_KAPPA_SPREAD_TOL}}

        # ---- verdict per family ----
        p1_ok, p2_ok = p1.get("pass", False), p2.get("chi2", {}).get("pass", False)
        if p3.get("pass"):
            verdict = "CONFIRMED"
        elif p1_ok and p2_ok:
            factor = float(np.median([kappas[i] / kappa_derived for i, m in enumerate(m_list)
                                      if kappas[i] is not None and kappa_derived]))
            verdict = "PARTIALLY_CONFIRMED"
        else:
            verdict = "MEASURED_LAW"
        factor = None
        if p1_ok and p2_ok and not p3.get("pass") and kappa_derived:
            factor = float(np.median([kappas[i] / kappa_derived
                                      for i, m in enumerate(m_list)
                                      if kappas[i] is not None]))

        families_out[fam] = {"P1_maxdev_m_invariance": p1,
                             "P1_maxdev_values": p1_values,
                             "P2_accumulated_m_minus_half": p2,
                             "P3_kappa_reproduction": p3,
                             "verdict": verdict,
                             "constant_shift_factor": factor,
                             "shift_note": ("median over m of kappa_hat/kappa_derived") 
                             if factor is not None else None}
        logger.info(f"[law] {fam}: P1={p1.get('pass')} P2(chi2)={p2.get('chi2', {}).get('pass')} "
                    f"P3={p3.get('pass')} -> {verdict}" + 
                    (f" factor={factor:.3f}" if factor else ""))

    # ---- overall verdict: worst across families ----
    fam_verdicts = [families_out[f]["verdict"] for f in CONFIRM_FAMILIES
                    if "verdict" in families_out[f]]
    if len(fam_verdicts) < len(CONFIRM_FAMILIES):
        overall = "MEASURED_LAW"
        overall_note = "some family data missing; overall downgraded to MEASURED_LAW"
    else:
        overall = max(fam_verdicts, key=VERDICT_ORDER.index)
        overall_note = "lexicographic worst across families"

    verdict_block = {
        "overall_verdict": overall,
        "per_family": {f: families_out[f].get("verdict") for f in CONFIRM_FAMILIES},
        "overall_note": overall_note,
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pre_registration_pointer": "results/pre_registration.json (verbatim rule)",
    }
    half_power["verdict"] = verdict_block
    half_power["families"] = families_out
    out_path.write_text(json.dumps(half_power, indent=1))
    logger.info(f"[law] OVERALL VERDICT: {overall}")
    return half_power


if __name__ == "__main__":  # pragma: no cover
    logger.remove()
    logger.add(lambda _: None, level="INFO")