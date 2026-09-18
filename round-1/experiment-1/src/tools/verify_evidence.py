#!/usr/bin/env python3
"""Post-hoc evidence verification (run after the power phase completes).

Independent read-offs for the three scientific claims:
  M  -- the blind band exists: maxdev power < 0.5 while max(chi2, trend,
        energy) power >= 0.95 at 5% alpha, for diffuse bias families;
        and the gap (A_acc / delta_floor) widens with n.
  C1 -- simulated maxdev null quantiles track the independent-Binomial
        benchmark F(x)^n (rel error reported per cell).
  C3 -- chi2 null variance under the choice-based null vs the product-
        Binomial prediction; textbook chi2_{n-1} false alarms.
Also prints the bug-battery linkage table (bug -> abstract family).

Usage: .venv/bin/python tools/verify_evidence.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    res = ROOT / "results"

    power = json.loads((res / "power_surfaces.json").read_text())
    cells = [c for c in power.get("cells", []) if not c.get("skipped")]
    print(f"== POWER: {len(cells)}/{len(power.get('cells', []))} cells ==")
    if not cells:
        print("no power cells -> run `uv run method.py power --scale full` first")
        return 1

    all_spike_ok, n_blind, n_gap = True, 0, 0
    for c in sorted(cells, key=lambda r: (r["n"], r["p"])):
        fams = c.get("families", {})
        aacc = {f: fams[f].get("A_acc_over_delta_floor") for f in fams}
        print(f"\ncell n={c['n']} p={c['p']:g} m={c['m']} delta_floor={c['delta_floor']:.4f}")
        print("  A_acc/delta_floor:", {k: (round(v, 3) if v else None) for k, v in aacc.items()})
        bb = c.get("blind_bands") or []
        for b in bb:
            print(f"  BLIND BAND {b['cell']} {b['family']}: delta=[{b['delta_lo']:.4f},{b['delta_hi']:.4f}] "
                  f"width={b['width']:.4f} midpoint5%={ {k: round(v, 2) for k, v in b['midpoint_power_alpha_0.05'].items()} }")
        n_blind += len(bb)
        sp = c.get("spike_inversion")
        if sp:
            rows = sp.get("rows", [])
            bad = [r for r in rows if r["maxdev_power"] + 0.05 < r["chi2_power"]]
            ok = len(bad) == 0 and len(rows) > 0
            all_spike_ok &= ok
            med = rows[len(rows) // 2] if rows else {}
            print(f"  spike inversion ok={ok} (maxdev>=chi2 everywhere): "
                  f"mid row delta={med.get('delta')} maxdev_pow={med.get('maxdev_power')} chi2_pow={med.get('chi2_power')}")
        gap = c.get("gap_vs_n") or {}
        if gap:
            n_gap += 1
            print("  gap_vs_n rows:", {f: round(g["A_acc_over_delta_floor"], 3) for f, g in gap.items()})

    print(f"\n== hive read-offs: blind bands found in {n_blind} (cell,family) pairs; "
          f"spike inversion holds at all powered cells: {all_spike_ok}; gap rows: {n_gap} ==")

    bug = json.loads((res / "bug_battery.json").read_text())
    print("\n== BUG BATTERY ==")
    for c in bug.get("cells", []):
        if c.get("skipped"):
            continue
        print(f"cell n={c['n']} k={c['k']} m={c['m']}")
        link = c.get("family_linkage_corr", {})
        for v, corrs in link.items():
            best = max(corrs, key=lambda f: abs(corrs[f]))
            print(f"  {v:<18} -> best family {best:<12} r={corrs[best]:+.2f}  "
                  f"maxdev5%={c['bugs'][v]['powers']['power_alpha_0.05']['maxdev']:.2f} "
                  f"chi2_5%={c['bugs'][v]['powers']['power_alpha_0.05']['chi2']:.2f}")

    null = json.loads((res / "null_calibration.json").read_text())
    nc = [r for r in null.get("cells", []) if not r.get("skipped")]
    print(f"\n== NULL (C1/C3): {len(nc)} cells ==")
    for r in sorted(nc, key=lambda x: (x["n"], x["k"]))[:12]:
        c1 = r["c1"]; c3 = r["c3"]
        print(f"  {r['cell']:<26} C1rel95={100*c1['relative_error_sim_vs_bench']['q0.95']:6.1f}% "
              f"C3ratio={c3['variance_ratio_true_vs_product_binomial']:.3f} FA5={c3['false_alarm_alpha_0.05']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())