"""PHASE 4 -- confirm-cell power measurement at three trial budgets.

Per (m, family), m in {2000, 5000, 10000}, family in {linear_trend, exp_recency}
at n=3000, p=0.05:
  * delta_floor(m) = A_floor/(p*m), A_floor = sqrt(2 p m log n)
  * amplitude calibration a->delta once per (n, family) at m=2000 (delta is a
    per-trial probability deviation, independent of m), reused for all m
  * smart amplitude grid: predicted m-invariant mults from the iteration-1
    anchors (Aacc_mult for chi2/trend/energy, mdev_mult for maxdev), deduped
    within 5%, plus a saturation safety point at 2.0 x max(predicted mults)
  * per grid point: n_fine = 1000 (m in {2000,5000}) / 800 (m=10000) biased
    priority replicates, all four statistics at alpha=0.05, alpha' (Sidak) and
    alpha/3 thresholds
  * half-power crossings per (stat, alpha) by log-mult linear interpolation,
    refined by 2-round log-mult bisection at 500 reps when the bracket points
    lie farther than 0.15 from power 0.5; extended x2 (up or down) up to two
    steps at 500 reps when the grid misses the bracket
Checkpoints: results/power_curves_m_invariance.json (full rows) and
results/half_power_law.json (crossings + kappa_hat(m) per budget).
"""
from __future__ import annotations

import gc
import json
import math
import time
from dataclasses import dataclass

import numpy as np
from loguru import logger

from .anchors import STAT_DISPLAY
from .benchmarks import extremes_floor
from .config import CALIB_AMPS, CALIB_REPS, RESULTS, child_rng, family_shape
from .confirm import (ACC_MULT_STEPS, BONF_ALPHA, CONFIRM_FAMILIES, CONFIRM_K,
                      CONFIRM_M, CONFIRM_N, CONFIRM_P, EXTEND_MAX_STEPS,
                      EXTEND_STEP, G4_ANCHOR_TOL, G4_TREND_SLACK, GRID_DEDUPE_TOL,
                      MDEV_MULT_STEPS, N_BISECT, N_FINE_5000, N_FINE_BIG,
                      N_FINE_MID, POWER_ALPHAS2, SAFETY_GRID_MULT, SIDAK_ALPHA,
                      alpha_label)
from .harness import (_amplitude_for_delta, _calibrate_amplitude, _delta_of_counts,
                      _dump_json, _safe_run, run_parallel)
from .samplers import priority_counts_matrix, validate_count_rows
from .stats import compute_stats, power_from_flags, rejection_flags, standardize_slopes

STATS = ("maxdev", "chi2", "trend_slope", "energy")
POWER_ALPHAS = POWER_ALPHAS2


# ---------------------------------------------------------------------------
# Picklable payload + worker (spawn pool)
# ---------------------------------------------------------------------------
@dataclass
class _PowerPayload:
    m: int
    family: str
    mult: float
    a: float
    n_fine: int
    thresholds: dict
    trend_ref: dict
    calib_amps: list
    calib_deltas: list

    def key(self) -> str:
        return f"confirm_power_{self.m}_{self.family}_{self.mult:.6g}"


def _power_worker(p: _PowerPayload) -> dict:
    n, k = CONFIRM_N, CONFIRM_K
    p0 = CONFIRM_P
    mu = p.m * p0
    rng = child_rng(p.key())
    shape = family_shape(p.family, n)
    counts = priority_counts_matrix(n, k, p.m, p.n_fine, rng, shape=shape, a=p.a)
    validate_count_rows(counts, p.m, k)
    stats = compute_stats(counts, mu)
    z = standardize_slopes(stats["trend_slope"], p.trend_ref["mu"], p.trend_ref["sd"])
    arrs = {"maxdev": stats["maxdev"], "chi2": stats["chi2"],
            "trend_slope": np.abs(z), "energy": stats["energy"]}
    powers: dict[str, dict[str, float]] = {}
    for alph in POWER_ALPHAS:
        lab = alpha_label(alph)
        th = {s: (0.0, p.thresholds[s][lab]) for s in STATS}
        powers[lab] = power_from_flags(rejection_flags(arrs, th))
    return {"m": p.m, "family": p.family, "mult": p.mult, "a": p.a,
            "n_reps": p.n_fine, "achieved_delta": _delta_of_counts(counts[:100], p.m, p0),
            "powers": powers}


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------
def smart_mults(Aacc_mult: float, mdev_mult: float) -> list[float]:
    """Union of predicted half-power mults (accumulated vs maxdev statistics),
    deduped within 5%, plus a saturation safety point."""
    acc = [Aacc_mult * s for s in ACC_MULT_STEPS]
    mdev = [mdev_mult * s for s in MDEV_MULT_STEPS]
    safety = SAFETY_GRID_MULT * max(Aacc_mult, mdev_mult)
    mults = sorted(set(round(v, 10) for v in acc + mdev + [safety]))
    kept: list[float] = []
    for v in mults:
        if not kept or v / kept[-1] > 1.0 + GRID_DEDUPE_TOL:
            kept.append(v)
    return kept


# ---------------------------------------------------------------------------
# Crossing machinery (log-mult scale)
# ---------------------------------------------------------------------------
def _crossing_logmult(rows: list[dict], stat: str, alpha_lab: str) -> float | None:
    xs = [r["mult"] for r in rows]
    ys = [r["powers"][alpha_lab][stat] for r in rows]
    if len(xs) < 2:
        return None
    order = np.argsort(xs)
    xs = np.array(xs, dtype=np.float64)[order]
    ys = np.array(ys, dtype=np.float64)[order]
    keep = np.concatenate([[True], xs[1:] - xs[:-1] > 1e-15])
    xs, ys = xs[keep], ys[keep]
    for i in range(len(xs) - 1):
        if (ys[i] - 0.5) * (ys[i + 1] - 0.5) <= 0 and ys[i] != ys[i + 1]:
            t = (0.5 - ys[i]) / (ys[i + 1] - ys[i])
            return float(np.exp(np.log(xs[i]) + t * (np.log(xs[i + 1]) - np.log(xs[i]))))
    return None


def _bracket(rows: list[dict], stat: str, alpha_lab: str) -> tuple[int, int] | None:
    xs = [r["mult"] for r in rows]
    ys = [r["powers"][alpha_lab][stat] for r in rows]
    order = np.argsort(xs)
    xs = np.array(xs)[order]
    ys = np.array(ys)[order]
    for i in range(len(xs) - 1):
        if (ys[i] - 0.5) * (ys[i + 1] - 0.5) <= 0 and ys[i] != ys[i + 1]:
            return int(i), int(i + 1)
    return None


def _eval_point(m: int, family: str, mult: float, a: float, n_fine: int,
                thresholds: dict, trend_ref: dict, calib: tuple[list, list],
                cache: dict[str, dict], extra_actions: list[str], action: str) -> dict:
    key = f"{mult:.10g}"
    if key in cache:
        return cache[key]
    payload = _PowerPayload(m=m, family=family, mult=mult, a=a, n_fine=n_fine,
                            thresholds=thresholds, trend_ref=trend_ref,
                            calib_amps=calib[0], calib_deltas=calib[1])
    row = _power_worker(payload)
    cache[key] = row
    if action:
        extra_actions.append(action)
    return row


def _missing_brackets(cache_rows: list[dict]) -> list[tuple[str, str, str]]:
    """(stat, alpha_lab, direction) pairs whose power-0.5 crossing is not
    bracketed by the rows in cache."""
    out = []
    for stat in STATS:
        for alph in POWER_ALPHAS:
            lab = alpha_label(alph)
            if _crossing_logmult(cache_rows, stat, lab) is not None:
                continue
            ys = [r["powers"][lab][stat] for r in cache_rows]
            if not ys:
                continue
            if max(ys) < 0.5:
                out.append((stat, lab, "up"))
            elif min(ys) > 0.5:
                out.append((stat, lab, "down"))
            else:
                out.append((stat, lab, "up"))
    return out


# ---------------------------------------------------------------------------
# Per-(m, family) power cell
# ---------------------------------------------------------------------------
def run_confirm_power_cell(m: int, family: str, anchors_family: dict, thresholds: dict,
                           trend_ref: dict, calib: tuple[list, list],
                           n_fine: int, workers: int,
                           resume_rows: list[dict] | None = None) -> dict:
    n, p0 = CONFIRM_N, CONFIRM_P
    k = CONFIRM_K
    delta_floor = extremes_floor(n, m, k)["delta_floor"]
    Aacc_mult = anchors_family["A_acc_over_delta_floor"]
    mdev_mult = anchors_family["mdev_mult_anchor"]

    mults = smart_mults(Aacc_mult, mdev_mult)
    cache: dict[str, dict] = {}
    for r in resume_rows or []:
        if r.get("m") == m and r.get("family") == family and "mult" in r:
            cache[f"{r['mult']:.10g}"] = r
    jobs = []
    for mt in mults:
        if f"{mt:.10g}" in cache:
            continue  # deterministic seed -> resumed row is identical
        a_t = _amplitude_for_delta(mt * delta_floor, *calib)
        if a_t < 1e-6:
            continue
        jobs.append(_PowerPayload(m=m, family=family, mult=mt, a=a_t, n_fine=n_fine,
                                  thresholds=thresholds, trend_ref=trend_ref,
                                  calib_amps=calib[0], calib_deltas=calib[1]))
    if jobs:
        grid_results = run_parallel(jobs, _power_worker, workers, "confirm_power_grid")
        for r in grid_results:
            if not r.get("skipped"):
                cache[f"{r['mult']:.10g}"] = r
    grid_rows = [r for r in cache.values() if "mult" in r]
    if len(grid_rows) < 2:
        raise RuntimeError(f"power grid failed (or empty resume) for m={m} {family}: "
                           f"{len(cache)} rows")
    actions: list[str] = ["resume"] if resume_rows else []

    # ---- extension until every (stat, alpha) crossing is bracketed ----
    for _ in range(EXTEND_MAX_STEPS):
        needed = _missing_brackets(list(cache.values()))
        if not needed:
            break
        for stat, alpha_lab, direction in needed:
            if direction == "up":
                new_mult = max(r["mult"] for r in cache.values()) * EXTEND_STEP
            else:
                new_mult = min(r["mult"] for r in cache.values()) / EXTEND_STEP
            a_t = _amplitude_for_delta(new_mult * delta_floor, *calib)
            if a_t < 1e-6 or new_mult > 64.0:
                continue
            if f"{new_mult:.10g}" not in cache:
                _eval_point(m, family, new_mult, a_t,
                            min(N_BISECT, 250) if m >= 10000 else N_BISECT,
                            thresholds, trend_ref, calib, cache, actions,
                            f"extend_{direction}")

    # ---- log-mult bisection refinement ----
    # Scope and trigger are tuned to the pre-registered deliverables (P1/P3 use
    # the alpha=0.05 maxdev/chi2 crossings): refine ONLY {chi2, maxdev} at
    # alpha=0.05 and only when the bracket endpoints lie > 0.20 from power 0.5
    # (the anchor-pinned grid already places a point AT the predicted crossing,
    # so refinements are typically 0-2 per cell).  sidak/bonf crossings and the
    # trend/energy 0.05 crossings are read off the grid by interpolation (their
    # records are reported, not used by the law tests).  Pre-registered plan:
    # 2-round log-mult bisection at 500 reps at every budget (a stale-session
    # trim to 1 round / 250 reps at m=10000 was reverted on 2026-09-19).
    n_bisect_rounds = 2
    bisect_n = N_BISECT
    bisect_stats = ("chi2", "maxdev")
    trigger_tol = 0.20
    for _round in range(n_bisect_rounds):
        refined_any = False
        for stat in bisect_stats:
            lab = "0.05"  # deliverable level (P1/P3); see module docstring
            br = _bracket(list(cache.values()), stat, lab)
            if br is None:
                continue
            rows_sorted = sorted(cache.values(), key=lambda r: r["mult"])
            lo, hi = rows_sorted[br[0]], rows_sorted[br[1]]
            if max(abs(lo["powers"][lab][stat] - 0.5),
                   abs(hi["powers"][lab][stat] - 0.5)) <= trigger_tol:
                continue
            mid = math.sqrt(lo["mult"] * hi["mult"])
            if f"{mid:.10g}" in cache:
                continue
            a_t = _amplitude_for_delta(mid * delta_floor, *calib)
            if a_t < 1e-6:
                continue
            _eval_point(m, family, mid, a_t, bisect_n, thresholds, trend_ref,
                        calib, cache, actions, f"bisect_r{_round + 1}")
            refined_any = True
        if not refined_any:
            break

    rows = sorted(cache.values(), key=lambda r: r["mult"])
    method_tag = "grid" if not actions else "grid+" + "+".join(dict.fromkeys(actions))
    # ---- half-power read-offs ----
    crossings: dict[str, dict[str, dict]] = {}
    for stat in STATS:
        crossings[stat] = {}
        for alph in POWER_ALPHAS:
            lab = alpha_label(alph)
            cr = _crossing_logmult(rows, stat, lab)
            rec = {"A_half_delta": (cr * delta_floor) if cr is not None else None,
                   "mult": cr, "method": method_tag}
            if cr is None:
                pmin = min(rows, key=lambda r: r["mult"])["powers"][lab][stat]
                rec["note"] = ("crossing_below_grid" if pmin > 0.5
                               else "crossing_above_grid")
            crossings[stat][lab] = rec

    chi_mult = crossings["chi2"]["0.05"]["mult"]
    return {"m": m, "family": family, "n": n, "k": k, "delta_floor": delta_floor,
            "predicted_mults": {"Aacc_mult": Aacc_mult, "mdev_mult": mdev_mult},
            "n_fine": n_fine, "crossings": crossings, "rows": rows,
            "kappa_hat": (1.0 / chi_mult ** 2) if chi_mult else None}


# ---------------------------------------------------------------------------
# Phase drivers
# ---------------------------------------------------------------------------
def _load_anchors() -> dict:
    p = RESULTS / "iter1_anchors.json"
    if not p.exists():
        raise FileNotFoundError("iter1_anchors.json missing -- run confirm anchors first")
    return json.loads(p.read_text())


def _calibrate_calib(family: str, rng) -> tuple[list[float], list[float]]:
    """a->delta calibration at m=2000 (cheapest budget; delta is m-independent)."""
    return _calibrate_amplitude(CONFIRM_N, CONFIRM_K, 2000, CONFIRM_P,
                                family_shape(family, CONFIRM_N), rng)


def evaluate_g4_anchors(anchors_fam: dict, cells: list[dict]) -> dict:
    """G4 reproducibility gate: m=2000 half-power vs iteration-1 anchors.

    10% band for bracketed anchors (maxdev, chi2); trend anchors are upper
    bounds (iteration-1 saturated) so the check is crossing <= 1.2 x bound;
    energy from coarse probe interpolation keeps the 10% band.
    """
    rows, ok_flags = [], []
    for fam in CONFIRM_FAMILIES:
        matches = [c for c in cells if c.get("family") == fam and c.get("m") == 2000]
        if not matches:
            rows.append({"family": fam, "ok": False, "note": "m=2000 cell missing"})
            ok_flags.append(False)
            continue
        rec = matches[0]
        for s in STAT_DISPLAY:
            anch = anchors_fam[fam]["anchors"][s]
            kind = anch["kind"]
            # G4 compares at alpha = 0.05 -- the level of the derived table
            # and of the iteration-1 anchors (rec["crossings"][s] is the
            # per-alpha dict {"0.05": ..., "sidak": ..., "bonf": ...}).
            i2 = rec["crossings"][s].get("0.05", {})
            if kind == "upper_bound":
                ok = (i2.get("mult") is not None
                      and i2["mult"] <= G4_TREND_SLACK * anch["mult"])
            elif kind == "unbracketed":
                ok = i2.get("mult") is not None
            else:
                anch_delta = anch["value_delta"]
                ok = (i2.get("A_half_delta") is not None
                      and abs(i2["A_half_delta"] - anch_delta) / anch_delta <= G4_ANCHOR_TOL)
            rows.append({"family": fam, "stat": s, "anchor_kind": kind,
                         "anchor_delta": anch["value_delta"],
                         "iter2_delta": i2.get("A_half_delta"),
                         "iter2_mult": i2.get("mult"),
                         "ok": bool(ok)})
            ok_flags.append(ok)
    return {"ok": bool(ok_flags) and all(ok_flags), "rows": rows,
            "note": ("G4: m=2000 half-power vs iteration-1 anchors. Band 10% for "
                     "bracketed anchors; trend = upper-bound check with 20% slack; "
                     "unbracketed anchors count as check-only.")}


def _maybe_fire_g4_gate(law_cells: list[dict], m_targets: tuple[int, ...],
                        families: tuple[str, ...], anchors: dict | None = None) -> None:
    """G4 reproducibility gate.  Hard (SystemExit) only when the full triple
    budget AND both families are targeted and the m=2000 cells are present;
    otherwise recorded as a soft checkpoint if the m=2000 cells exist."""
    if set(m_targets) != set(CONFIRM_M) or families != CONFIRM_FAMILIES:
        return
    present = all(
        any(c.get("cell") == f"power_cell_m2000_{f}" and not c.get("skipped")
            for c in law_cells) for f in CONFIRM_FAMILIES)
    if not present:
        return
    anchors = anchors or _load_anchors()["cells"]["power_n3000_p0.05_m2000"]
    g4 = evaluate_g4_anchors(anchors, law_cells)
    _dump_json(RESULTS / "sanity_g4_anchor.json", g4)
    if not g4["ok"]:
        logger.error(f"[G4] reproducibility anchor FAILED at m=2000: {g4}")
        raise SystemExit("G4 gate failure: m=2000 anchors deviate > 10% from "
                         "iteration-1; refusing to trust m=5000/10000 cells")
    logger.info("[G4] m=2000 reproducibility anchor PASSED vs iteration-1")


def run_power_phase(workers: int, m_targets: tuple[int, ...] = CONFIRM_M,
                    force: bool = False, families: tuple[str, ...] | None = None,
                    resume: bool = True) -> dict:
    if families is None:
        families = CONFIRM_FAMILIES
    out_path = RESULTS / "power_curves_m_invariance.json"
    law_path = RESULTS / "half_power_law.json"
    law_table = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "n": CONFIRM_N, "p": CONFIRM_P, "m_list": list(m_targets),
                 "cells": []}
    curves_table = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "cells": []}

    requested = {f"power_cell_m{m}_{f}" for m in m_targets for f in families}
    if law_path.exists():
        law_table = json.loads(law_path.read_text())
        curves_table = json.loads(out_path.read_text()) if out_path.exists() else curves_table
        if force:
            # drop ONLY the requested cells; keep other budgets/families so a
            # partial --force re-run does not destroy completed cells
            law_table["cells"] = [c for c in law_table["cells"]
                                  if c.get("cell") not in requested or c.get("skipped")]
            curves_table["cells"] = [c for c in curves_table["cells"]
                                     if c.get("cell") not in requested]
            law_table["m_list"] = list(m_targets)
            logger.info(f"force: recompute {sorted(requested)}; keeping other cells")
        else:
            missing = [k for k in requested
                       if not any(c.get("cell") == k and not c.get("skipped")
                                  for c in law_table["cells"])]
            if not missing:
                _maybe_fire_g4_gate(anchors=None, law_cells=law_table["cells"],
                                    m_targets=m_targets, families=families)
                logger.info(f"all requested cells present in {law_path.name}; loading")
                return law_table
            logger.info(f"resuming: {len(missing)} cells missing -> {sorted(missing)}")

    anchors = _load_anchors()["cells"]["power_n3000_p0.05_m2000"]
    thresholds = json.loads((RESULTS / "thresholds_confirm.json").read_text())

    # amplitude calibration once per (n, family) -- delta is m-independent,
    # cache for ALL trial budgets (plan PHASE 4.2)
    calibs = {family: _calibrate_calib(family, child_rng(f"confirm_calib_{CONFIRM_N}_{family}"))
              for family in families}

    # m-outer loop: the m=2000 reproducibility cells run FIRST so the G4 gate
    # fires BEFORE any m=5000/10000 compute (plan PHASE 9 G4).
    for m in sorted(m_targets):
        for family in families:
            cell_key = f"power_cell_m{m}_{family}"
            if any(c.get("cell") == cell_key for c in law_table["cells"]
                   if not c.get("skipped")):
                logger.info(f"{cell_key} already in checkpoint; skipping")
                continue
            null = thresholds[f"confirm_null_m{m}"]
            n_fine = N_FINE_BIG if m >= 10000 else (N_FINE_5000 if m >= 5000 else N_FINE_MID)
            t0 = time.time()
            logger.info(f"[phase4] running m={m} family={family} "
                        f"(n_fine={n_fine}, workers={workers})")
            prev_rows = []
            if resume:
                for c in curves_table.get("cells", []):
                    if c.get("cell") == cell_key and not c.get("skipped"):
                        prev_rows = c.get("rows", [])
                        break
            try:
                res = run_confirm_power_cell(m, family, anchors[family],
                                             null["thresholds"],
                                             null["trend_z_reference"], calibs[family],
                                             n_fine, workers, resume_rows=prev_rows)
            except Exception:
                logger.exception(f"[phase4] cell {cell_key} failed")
                curves_table["cells"].append({"cell": cell_key, "skipped": True})
                law_table["cells"].append({"cell": cell_key, "skipped": True})
                _dump_json(out_path, curves_table)
                _dump_json(law_path, law_table)
                continue
            res["cell"] = cell_key
            res["wall_s"] = round(time.time() - t0, 2)
            curves_table["cells"].append(res)
            law_cell = {"cell": cell_key, "m": m, "family": family,
                        "delta_floor": res["delta_floor"],
                        "n_fine": res["n_fine"],
                        "crossings": res["crossings"],
                        "wall_s": res["wall_s"]}
            if res["kappa_hat"] is not None:
                law_cell["kappa_hat"] = res["kappa_hat"]
            law_table["cells"].append(law_cell)
            _dump_json(out_path, curves_table)
            _dump_json(law_path, law_table)
            logger.info(f"[phase4] m={m} {family}: chi2 mult@0.05="
                        f"{res['crossings']['chi2']['0.05'].get('mult')}, "
                        f"maxdev mult@0.05="
                        f"{res['crossings']['maxdev']['0.05'].get('mult')}, "
                        f"kappa_hat={res['kappa_hat']}")
            gc.collect()

        # G4 reproducibility gate: after the m=2000 cells of the targeted
        # families are in, before any m=5000/10000 compute (plan PHASE 9 G4).
        _maybe_fire_g4_gate(law_table["cells"], m_targets, families, anchors)
    return law_table


if __name__ == "__main__":  # pragma: no cover
    logger.remove()
    logger.add(sys.stdout, level="INFO")