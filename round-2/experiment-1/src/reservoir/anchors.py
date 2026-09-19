"""PHASE 1 -- iteration-1 anchor extraction and pre-registration (no simulation).

Reads the iteration-1 results (READ-ONLY; never modified) and writes:
  results/iter1_anchors.json   -- small audit-trail extract of the numbers the
                                  confirm run needs
  results/pre_registration.json -- verbatim P1/P2/P3 statements, tolerances,
                                  kappa_derived table, anchor table, verdict rule

Anchor conventions:
  * maxdev: linear interpolation of the power=0.5 crossing on delta_target over
    the fine+coarse pass rows of the iteration-1 power cell (well bracketed).
  * chi2:   the headline iteration-1 A_acc_delta (the established chi2
    half-power delta; probe-bisection at coarse N=300).
  * energy: crossing interpolated from the probe pass rows when bracketed,
    else an upper bound (power >= 0.5 at the smallest probed mult).
  * trend:  in iteration-1 the |z| trend statistic saturates (power ~1.0) at
    the smallest probed mult 0.05 x delta_floor -- the iteration-1 anchor is an
    UPPER BOUND (crossing <= that delta).  G4 treats trend with the pre-stated
    slack.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger

from .config import RESULTS
from .confirm import (ALPHA_KEYS, CONFIRM_FAMILIES, EXTEND_MAX_STEPS, G4_ANCHOR_TOL,
                      G4_TREND_SLACK, P1_MAX_SPREAD_TOL, P2_SLOPE_RANGE,
                      P3_KAPPA_REL_TOL, P3_KAPPA_SPREAD_TOL, VERDICT_ORDER)

# read-only iteration-1 results location
ITER1_DIR = Path("/ai-inventor/aii_data/runs/run_yRBWOr6EQPIx/"
                 "3_invention_loop/iter_1/gen_art/gen_art_experiment_1/results")

STATS = ("maxdev", "chi2", "trend_slope", "energy")
# readable stat names for the anchor table
STAT_DISPLAY = {"maxdev": "maxdev", "chi2": "chi2", "trend_slope": "trend|z|",
                "energy": "subwin_energy"}


def _load_iter1(name: str) -> dict:
    p = ITER1_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"iteration-1 results missing: {p} (F1 fallback "
                                f"to published numbers is handled by the caller)")
    return json.loads(p.read_text())


def _crossing_delta(rows: list[dict], stat: str) -> float | None:
    """Linear interpolation of the power=0.5 crossing on delta_target (delta scale)."""
    if not rows:
        return None
    xs = np.array([r["delta_target"] for r in rows], dtype=np.float64)
    ys = np.array([r["power_alpha_0.05"][stat] for r in rows], dtype=np.float64)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    # dedupe identical deltas (keep the later, finer measurement)
    keep = np.concatenate([[True], xs[1:] - xs[:-1] > 1e-15])
    xs, ys = xs[keep], ys[keep]
    for i in range(len(xs) - 1):
        if (ys[i] - 0.5) * (ys[i + 1] - 0.5) <= 0 and ys[i] != ys[i + 1]:
            t = (0.5 - ys[i]) / (ys[i + 1] - ys[i])
            return float(xs[i] + t * (xs[i + 1] - xs[i]))
    return None


def _crossing_mult(rows: list[dict], stat: str) -> float | None:
    """Crossing on log(mult) scale (the phase-4 convention)."""
    if not rows:
        return None
    xs = np.array([r["mult"] for r in rows], dtype=np.float64)
    ys = np.array([r["power_alpha_0.05"][stat] for r in rows], dtype=np.float64)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    keep = np.concatenate([[True], xs[1:] - xs[:-1] > 1e-15])
    xs, ys = xs[keep], ys[keep]
    for i in range(len(xs) - 1):
        if (ys[i] - 0.5) * (ys[i + 1] - 0.5) <= 0 and ys[i] != ys[i + 1]:
            t = (0.5 - ys[i]) / (ys[i + 1] - ys[i])
            return float(np.exp(np.log(xs[i]) + t * (np.log(xs[i + 1]) - np.log(xs[i]))))
    return None


def _anchor_for_stat(rows_by_src: dict[str, list[dict]], stat: str,
                     a_acc_delta: float | None, delta_floor: float) -> dict:
    """Best available anchor for one statistic from the iteration-1 rows.

    rows_by_src: {"fine_pass": [...], "coarse_pass": [...], "probe_pass": [...]}
    Returns {"value_delta", "mult", "kind"} with kind in
    {"fine_interp", "coarse_interp", "probe_interp", "a_acc", "upper_bound"}.
    """
    if stat == "chi2" and a_acc_delta is not None:
        return {"value_delta": a_acc_delta, "mult": a_acc_delta / delta_floor,
                "kind": "a_acc"}
    combined = list(rows_by_src.get("fine_pass", [])) + \
        list(rows_by_src.get("coarse_pass", []))
    cr = _crossing_delta(combined, stat)
    if cr is not None and any(r["mult"] < 0.5 for r in combined):
        kind = "fine_interp" if rows_by_src.get("fine_pass") else "coarse_interp"
        return {"value_delta": cr, "mult": cr / delta_floor, "kind": kind}
    probes = list(rows_by_src.get("probe_pass", []))
    pr = _crossing_delta(probes, stat) if probes else None
    if pr is not None:
        return {"value_delta": pr, "mult": pr / delta_floor, "kind": "probe_interp"}
    if probes:
        # not bracketed in the probe pass: power >= 0.5 already at the smallest
        # probed mult -> upper bound at that mult
        pmin = min(probes, key=lambda r: r["mult"])
        return {"value_delta": pmin["delta_target"], "mult": pmin["mult"],
                "kind": "upper_bound"}
    return {"value_delta": None, "mult": None, "kind": "unbracketed"}


def extract_anchors() -> dict:
    """Read iteration-1 results and return the anchor extract."""
    ps = _load_iter1("power_surfaces.json")
    cells = {c["cell"]: c for c in ps["cells"]}

    def cell_anchor(cell_key: str, family: str) -> dict | None:
        c = cells.get(cell_key)
        if c is None or family not in c.get("families", {}):
            logger.warning(f"iter-1 cell {cell_key} family {family} missing")
            return None
        f = c["families"][family]
        df = f["delta_floor"]
        acc = f.get("A_acc_over_delta_floor")
        a_acc_delta = f.get("A_acc_delta")
        rows = {"fine_pass": f["fine_pass"], "coarse_pass": f["coarse_pass"],
                "probe_pass": f.get("A_acc_probe_pass", [])}
        stats = {s: _anchor_for_stat(rows, s, a_acc_delta, df) for s in STATS}
        mdev_anchor = stats["maxdev"]
        return {
            "family": family, "cell": cell_key,
            "delta_floor": df, "A_floor": f["A_floor"],
            "A_acc_over_delta_floor": acc,
            "kappa_derived": (1.0 / acc ** 2) if acc else None,
            "A_acc_delta": a_acc_delta,
            "mdev_mult_anchor": mdev_anchor.get("mult"),
            "anchors": {
                s: {"value_delta": a["value_delta"], "mult": a["mult"], "kind": a["kind"]}
                for s, a in stats.items()
            },
            "n_fine_iter1": c.get("families", {}).get(family, {}).get
            ("fine_pass", [{}])[0].get("n_rep") if f.get("fine_pass") else None,
            "probe_pass_rows": rows["probe_pass"],
        }

    out = {"iter1_dir": str(ITER1_DIR),
           "generated_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
           "read_only_note": ("extract from iteration-1 results; the source "
                              "files are NOT modified by this iteration"),
           "cells": {}}
    out["cells"]["power_n3000_p0.05_m2000"] = {
        fam: cell_anchor("power_n3000_p0.05_m2000", fam) for fam in CONFIRM_FAMILIES}
    # secondary-arm comparison cells
    out["cells"]["power_n10000_p0.05_m1000"] = {
        "linear_trend": cell_anchor("power_n10000_p0.05_m1000", "linear_trend")}
    out["cells"]["power_n300_p0.05_m2000"] = {
        "linear_trend": cell_anchor("power_n300_p0.05_m2000", "linear_trend")}
    out["cells"]["power_n3000_p0.5_m2000"] = {
        "linear_trend": cell_anchor("power_n3000_p0.5_m2000", "linear_trend")}

    # null thresholds used for power/bug phases in iteration-1 (continuity)
    th = _load_iter1("thresholds.json")
    t = th.get("null_n3000_k150_m2000")
    if t is not None:
        out["null_reference"] = {
            "cell": "null_n3000_k150_m2000", "n_reps": t["n_reps"], "mu": t["mu"],
            "thresholds_alpha_0.05": {s: t["thresholds"][s]["alpha_0.05"]
                                      for s in STATS},
            "trend_z_reference": t["trend_z_reference"],
            "stat_quantiles": {s: {str(q): v for q, v in
                                   t["stat_quantiles"][s].items()} for s in STATS},
        }
    return out


def pre_registration(anchors: dict) -> dict:
    """Verbatim pre-registration: hypotheses, tolerances, anchors, verdict rule."""
    cells = anchors["cells"]
    kappa_table = {}
    anchor_table = {}
    for fam in CONFIRM_FAMILIES:
        rec = cells["power_n3000_p0.05_m2000"][fam]
        kappa_table[fam] = rec["kappa_derived"]
        anchor_table[fam] = {s: a for s, a in rec["anchors"].items()}
    sec = cells["power_n10000_p0.05_m1000"]["linear_trend"]
    sec300 = cells["power_n300_p0.05_m2000"]["linear_trend"]
    sec50 = cells["power_n3000_p0.5_m2000"]["linear_trend"]
    return {
        "title": "Trial-budget law for reservoir uniformity checks: confirm-by-measurement",
        "cell": "n=3000, p=0.05 (k=150), m in {2000, 5000, 10000}",
        "families": list(CONFIRM_FAMILIES),
        "hypotheses": {
            "P1": ("maxdev half-power is m-invariant in floor-relative units: "
                   "max pairwise relative spread of A_half_over_delta_floor(maxdev) "
                   f"over m in {{2000,5000,10000}} <= {P1_MAX_SPREAD_TOL:.0%}"),
            "P2": ("accumulated detectors' half-power scales as m^(-1/2): LS slope "
                   "of log(A_half_delta_chi2) vs log(m) in "
                   f"{P2_SLOPE_RANGE} (trend and energy reported likewise)"),
            "P3": (f"kappa_hat(m) = (delta_floor(m)/A_half_delta_chi2(m))^2 is "
                   "m-invariant and reproduces the derived table: "
                   f"|kappa_hat(m)-kappa_derived|/kappa_derived <= {P3_KAPPA_REL_TOL:.0%} "
                   f"for every m in {{5000,10000}} AND spread of kappa_hat over "
                   f"{{2000,5000,10000}} <= {P3_KAPPA_SPREAD_TOL:.0%}"),
        },
        "tolerances": {
            "P1_max_spread_tol": P1_MAX_SPREAD_TOL,
            "P2_slope_range": list(P2_SLOPE_RANGE),
            "P3_kappa_rel_tol": P3_KAPPA_REL_TOL,
            "P3_kappa_spread_tol": P3_KAPPA_SPREAD_TOL,
            "G4_anchor_tol": G4_ANCHOR_TOL,
            "G4_trend_slack": G4_TREND_SLACK,
        },
        "kappa_derived": kappa_table,
        "kappa_derived_headlines": {
            "n=3000 p=0.05 (confirm cell)": kappa_table,
            "n=10000 p=0.05 m=1000 (322x headline)": sec["kappa_derived"],
            "n=300 p=0.05 m=2000 (32x value)": sec300["kappa_derived"],
            "n=3000 p=0.5 m=2000 (arm-2 value)": sec50["kappa_derived"],
        },
        "anchors_m2000": anchor_table,
        "anchor_kinds": {
            fam: {s: a["kind"] for s, a in anchor_table[fam].items()}
            for fam in CONFIRM_FAMILIES
        },
        "anchor_kind_note": ("maxdev/chi2: bracketed in iteration-1 (fine_interp / "
                             "a_acc); energy: coarse probe interpolation; trend: "
                             "UPPER BOUND (iteration-1 power saturated at the "
                             "smallest probed mult 0.05 x delta_floor) -- G4 applies "
                             f"the {G4_TREND_SLACK:.0%} slack to trend."),
        "verdict_rule": (
            "Per family at alpha=0.05: "
            "P1 PASS iff max pairwise relative spread of maxdev A_half_over_delta_floor "
            f"over m in {{2000,5000,10000}} <= {P1_MAX_SPREAD_TOL:.0%}. "
            "P2 PASS iff LS slope of log(A_half_delta_chi2) vs log(m) in "
            f"{P2_SLOPE_RANGE}. "
            "P3: kappa_hat(m) = (delta_floor(m)/A_half_delta_chi2(m))^2; "
            "CONFIRMED iff |kappa_hat(m)-kappa_derived|/kappa_derived <= "
            f"{P3_KAPPA_REL_TOL:.0%} for every m in {{5000,10000}} AND max pairwise "
            f"relative spread of kappa_hat over {{2000,5000,10000}} <= {P3_KAPPA_SPREAD_TOL:.0%}. "
            "PARTIALLY_CONFIRMED iff P1 and P2 PASS but the multiplier shifts by a "
            "roughly constant factor (factor = median over m of kappa_hat/kappa_derived, "
            "reported). Otherwise MEASURED_LAW: the fitted exponent and intercept over "
            "the three budgets become the finding. Overall verdict = worst across the "
            f"two families in the order {VERDICT_ORDER}. "
            "The direction of the blind band is NOT re-litigated: it is established "
            "by iteration-1."
        ),
        "blind_band_direction_note": (
            "Established by iteration-1: maxdev under-resolves diffuse biases "
            "(chi2/trend/energy saturate at ~0.2-0.35 x delta_floor while maxdev "
            "crosses at ~0.31-0.37 x delta_floor at n=3000). Not re-tested here."),
    }


def run_phase1() -> dict:
    anchors = extract_anchors()
    reg = pre_registration(anchors)
    (RESULTS / "iter1_anchors.json").write_text(json.dumps(anchors, indent=1))
    (RESULTS / "pre_registration.json").write_text(json.dumps(reg, indent=1))
    for fam in CONFIRM_FAMILIES:
        a = anchors["cells"]["power_n3000_p0.05_m2000"][fam]
        logger.info(f"phase1 {fam}: kappa_derived={a['kappa_derived']:.4g} "
                    f"mdev_mult_anchor={a['mdev_mult_anchor']:.4g} "
                    f"Aacc_mult={a['A_acc_over_delta_floor']:.4g}")
        for s in STATS:
            anch = a["anchors"][s]
            logger.info(f"  anchor[{s}] kind={anch['kind']:>13} "
                        f"delta={anch['value_delta']} mult={anch['mult']}")
    logger.info("wrote results/iter1_anchors.json + results/pre_registration.json")
    return {"anchors": anchors, "pre_registration": reg}


if __name__ == "__main__":  # pragma: no cover
    from loguru import logger as _l
    import sys
    _l.remove(); _l.add(sys.stdout, level="INFO")
    run_phase1()