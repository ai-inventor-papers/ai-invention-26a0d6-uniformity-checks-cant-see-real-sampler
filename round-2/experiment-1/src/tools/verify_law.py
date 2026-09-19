#!/usr/bin/env python3
"""T5 -- independent post-hoc verification of the iteration-2 confirm evidence.

Recomputes P1/P2/P3, the verdict, the FWER/negative-control read-offs and the
practitioner IB-benchmark comparison DIRECTLY from the raw checkpoint records,
without importing the reservoir library (guards against assembly-stage
transcription bugs).  Mirrors iteration-1's tools/verify_evidence.py.

USAGE:  uv run python tools/verify_law.py [results_dir]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

RESULTS = Path(sys.argv[1] if len(sys.argv) > 1 else "results").resolve()

SIDAK = 1.0 - 0.95 ** (1.0 / 3.0)
BONF = 0.05 / 3.0
M_LIST = (2000, 5000, 10000)
FAMILIES = ("linear_trend", "exp_recency")
STATS = ("maxdev", "chi2", "trend_slope", "energy")
P1_TOL, P2_RANGE, P3_TOL = 0.15, (-0.55, -0.45), 0.25
P3_SPREAD_TOL = 0.25


def pairwise_spread(values: list[float]) -> float:
    v = sorted(values)
    return (v[-1] - v[0]) / (sum(v) / len(v)) if len(v) >= 2 else 0.0


def ls_slope(xs, ys) -> float:
    n = len(xs)
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx, my = sum(lx) / n, sum(ly) / n
    num = sum((lx[i] - mx) * (ly[i] - my) for i in range(n))
    den = sum((lx[i] - mx) ** 2 for i in range(n))
    return num / den


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def main() -> int:
    ok = True
    report: list[str] = []

    def check(cond: bool, msg: str) -> None:
        nonlocal ok
        report.append(("PASS " if cond else "FAIL ") + msg)
        ok = ok and cond

    # ---- 1. power law: recompute P1/P2/P3 from raw crossings ----
    law = load("half_power_law.json")
    anchors = load("iter1_anchors.json")
    cells = {c["cell"]: c for c in law["cells"] if not c.get("skipped")}
    for fam in FAMILIES:
        fam_cells = {m: cells[f"power_cell_m{m}_{fam}"] for m in M_LIST}
        # P1: maxdev mult spread over the three budgets
        mdev = [fam_cells[m]["crossings"]["maxdev"]["0.05"]["mult"] for m in M_LIST]
        check(all(v is not None for v in mdev), f"P1 {fam}: maxdev crossings present")
        if all(v is not None for v in mdev):
            sp = pairwise_spread(mdev)
            check(sp <= P1_TOL, f"P1 {fam}: maxdev mult spread {sp:.4f} <= {P1_TOL}")
        # P2: LS slope of log(A_half_delta) vs log(m) for chi2
        chi = [fam_cells[m]["crossings"]["chi2"]["0.05"]["A_half_delta"] for m in M_LIST]
        if all(v and v > 0 for v in chi):
            sl = ls_slope(list(M_LIST), chi)
            check(P2_RANGE[0] <= sl <= P2_RANGE[1],
                  f"P2 {fam}: chi2 log-log slope {sl:.4f} in {P2_RANGE}")
        # P3: kappa_hat = (delta_floor/A_half_delta_chi2)^2 vs kappa_derived
        kd = anchors["cells"]["power_n3000_p0.05_m2000"][fam]["kappa_derived"]
        kh = {m: (fam_cells[m]["delta_floor"] / chi[i]) ** 2 if chi[i] else None
              for i, m in enumerate(M_LIST)}
        if all(v for v in kh.values()):
            rel = {m: abs(kh[m] - kd) / kd for m in M_LIST}
            check(all(rel[m] <= P3_TOL for m in (5000, 10000)),
                  f"P3 {fam}: kappa_hat reproduces kappa_derived ({kd:.1f}) within "
                  f"{P3_TOL:.0%} at m in {{5000,10000}}: "
                  f"{ {m: round(rel[m], 3) for m in M_LIST} }")
            # cross-check the recorded kappa_hat values
            rec_kh = [fam_cells[m].get("kappa_hat") for m in M_LIST]
            check(all(abs(kh[m] - rec_kh[i]) / kh[m] < 1e-9
                      for i, m in enumerate(M_LIST) if rec_kh[i] is not None),
                  f"P3 {fam}: recorded kappa_hat matches recomputation")
            check(pairwise_spread([kh[m] for m in M_LIST]) <= P3_SPREAD_TOL,
                  f"P3 {fam}: kappa_hat m-invariance spread "
                  f"{pairwise_spread([kh[m] for m in M_LIST]):.4f} <= {P3_SPREAD_TOL}")
    # verdict sanity: recompute worst-across-families
    fam_v = {f: law["verdict"]["per_family"][f] for f in FAMILIES}
    order = {"CONFIRMED": 0, "PARTIALLY_CONFIRMED": 1, "MEASURED_LAW": 2}
    worst = max(fam_v.values(), key=order.get) if fam_v else "MEASURED_LAW"
    check(law["verdict"]["overall_verdict"] == worst,
          f"overall verdict {law['verdict']['overall_verdict']} == worst-of-families {worst}")

    # ---- 2. FWER: recompute Sidak FWER ~= 0.05 within 2SE ----
    fwer = load("fwer_check.json")
    for r in fwer["cells"]:
        if r.get("skipped"):
            continue
        n = r["N"]
        se = math.sqrt(0.05 * 0.95 / n)
        check(abs(r["fwer_sidak"] - 0.05) <= 2 * se,
              f"FWER m={r['m']}: sidak {r['fwer_sidak']:.4f} within 2SE of 0.05")

    # ---- 3. negative control: float_threshold power ~= protocol alpha ----
    bugs = load("bug_battery_trials.json")
    ft = next(v for v in bugs["variants"] if v.get("variant") == "float_threshold"
              and not v.get("skipped"))
    se5 = math.sqrt(0.05 * 0.95 / bugs["n_reps"])
    for m in bugs["m_list"]:
        pa = ft["power_maxdev_only_per_m"][str(m)]
        check(abs(pa - 0.05) <= 3 * se5,
              f"G5 m={m}: float_threshold maxdev-only power {pa:.4f} within 3SE of 0.05")

    # ---- 4. practitioner: sim threshold vs independent-Binomial benchmark ----
    pract = load("practitioner_null.json")
    for cell in pract["cells"]:
        thr = cell["null_threshold"]["threshold_alpha_0.05"]
        ib = cell["null_threshold"]["ib_benchmark_q95"]
        rel_err = abs(thr - ib) / ib if ib else None
        check(rel_err is not None and rel_err <= 0.10,
              f"pract cell {cell['cell']}: sim threshold {thr:.1f} vs IB {ib:.1f} "
              f"rel err {rel_err:.2%} <= 10%")

    print("\n".join(report))
    print(f"\nT5 VERDICT: {'ALL CHECKS PASS' if ok else 'CHECKS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())