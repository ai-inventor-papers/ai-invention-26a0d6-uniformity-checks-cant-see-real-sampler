"""PHASE 10 -- schema-valid method_out.json assembly + README for iteration 2.

Datasets (one per deliverable):
  * reservoir_uniformity_law_confirm  -- per (m, family) half-power read-offs
  * reservoir_uniformity_fwer         -- per m protocol FWER
  * reservoir_uniformity_bug_trials   -- per (variant, m) trials-to-90%
  * reservoir_uniformity_practitioner_null -- per sampler x cell tour verdicts
  * reservoir_uniformity_secondary_arms    -- per arm kappa reproduction
All predict_* values are strings (schema patternProperties type string).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from loguru import logger

from .config import RESULTS, ROOT, SEED
from .confirm import (BONF_ALPHA, CONFIRM_FAMILIES, CONFIRM_K, CONFIRM_M,
                      CONFIRM_N, CONFIRM_P, SIDAK_ALPHA, _SCALE)
from .anchors import STAT_DISPLAY


def _load(name: str) -> dict:
    p = RESULTS / name
    if not p.exists():
        raise FileNotFoundError(f"missing results/{name} -- run the confirm phases first")
    return json.loads(p.read_text())


def _fmt(v) -> str:
    if v is None:
        return "None"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


# ---------------------------------------------------------------------------
# Dataset 1: law confirm
# ---------------------------------------------------------------------------
def _law_examples(law: dict, anchors: dict) -> list[dict]:
    ex, cells = [], law.get("cells", [])
    for c in sorted([x for x in cells if not x.get("skipped")], key=lambda x: (x["m"], x["family"])):
        fam = c["family"]
        m = c["m"]
        out = {"m": m, "family": fam, "delta_floor": c["delta_floor"],
               "half_power_delta_per_stat": {STAT_DISPLAY[s]:
                                             c["crossings"][s]["0.05"].get("A_half_delta")
                                             for s in STAT_DISPLAY},
               "mult_per_stat_alpha05": {STAT_DISPLAY[s]:
                                         c["crossings"][s]["0.05"].get("mult")
                                         for s in STAT_DISPLAY},
               "mult_per_stat_sidak": {STAT_DISPLAY[s]:
                                       c["crossings"][s]["sidak"].get("mult")
                                       for s in STAT_DISPLAY},
               "kappa_hat": c.get("kappa_hat"),
               "method": c["crossings"]["chi2"]["0.05"].get("method")}
        ex.append({
            "input": (f"Confirm trial-budget law, cell n={CONFIRM_N}, p={CONFIRM_P}, "
                      f"m={m}, family={fam}: measure the half-power amplitude of "
                      "maxdev, chi2, trend|z| and subwindow energy at alpha=0.05 "
                      "and the Sidak level, and report kappa_hat(m)."),
            "output": json.dumps(out, separators=(",", ":")),
            "metadata_cell": c["cell"], "metadata_family": fam,
            "metadata_m": str(m), "metadata_alpha": "0.05_and_sidak",
            "predict_kappa_hat_" + fam.replace("_", ""): _fmt(c.get("kappa_hat")),
            "predict_maxdev_mult": _fmt(c["crossings"]["maxdev"]["0.05"].get("mult")),
            "predict_chi2_mult": _fmt(c["crossings"]["chi2"]["0.05"].get("mult")),
            "predict_trend_mult": _fmt(c["crossings"]["trend_slope"]["0.05"].get("mult")),
            "predict_energy_mult": _fmt(c["crossings"]["energy"]["0.05"].get("mult")),
            "predict_law_verdict": law.get("verdict", {}).get("overall_verdict", "MEASURED_LAW"),
            "predict_baseline_kappa_derived_" + fam.replace("_", ""):
                _fmt(anchors["kappa_derived_" + fam] if False else
                     _kappa_derived(anchors, fam)),
        })
    return ex


def _kappa_derived(anchors: dict, fam: str) -> float | None:
    return anchors["cells"]["power_n3000_p0.05_m2000"][fam]["kappa_derived"]


# ---------------------------------------------------------------------------
# Dataset 2: FWER
# ---------------------------------------------------------------------------
def _fwer_examples(fwer: dict) -> list[dict]:
    ex = []
    for r in sorted([x for x in fwer["cells"] if not x.get("skipped")], key=lambda x: x["m"]):
        ex.append({
            "input": (f"Empirical FWER of the corrected 3-test protocol "
                      "{maxdev, chi2, trend|z|} at n=3000, p=0.05, m={r['m']}, "
                      "N={r['N']} null replicates of a correct priority sampler."),
            "output": json.dumps({"fwer_sidak": r["fwer_sidak"], "fwer_bonf": r["fwer_bonf"],
                                  "marginal_sidak": r["marginal_sidak"], "N": r["N"],
                                  "binomial_se": r["binomial_se"]}, separators=(",", ":")),
            "metadata_cell": f"fwer_m{r['m']}", "metadata_m": str(r["m"]),
            "metadata_N": str(r["N"]),
            "predict_fwer_sidak": _fmt(r["fwer_sidak"]),
            "predict_fwer_bonf": _fmt(r["fwer_bonf"]),
            "predict_baseline_alpha_divided_by_3": _fmt(BONF_ALPHA),
            "predict_sidak_alpha": _fmt(SIDAK_ALPHA),
        })
    return ex


# ---------------------------------------------------------------------------
# Dataset 3: bug trials
# ---------------------------------------------------------------------------
def _bug_examples(bugs: dict) -> list[dict]:
    ex = []
    for v in bugs["variants"]:
        if v.get("skipped"):
            continue
        for m in bugs["m_list"]:
            # JSON round-trip strings the per-m dict keys; look up by str(m)
            p_a = v.get("power_maxdev_only_per_m", {}).get(str(m))
            p_b = v.get("power_3test_per_m", {}).get(str(m))
            ex.append({
                "input": (f"Blind bug battery, n=3000, p=0.05, variant="
                          f"{v['variant']}, m={m}: detection power of the "
                          "maxdev-only protocol (alpha=0.05) and the 3-test "
                          "protocol (per-test Sidak), trials-to-90% units."),
                "output": json.dumps({"variant": v["variant"], "m": m,
                                      "power_maxdev_only": p_a, "power_3test": p_b,
                                      "m90_maxdev_only": v["m90_maxdev_only"],
                                      "m90_3test": v["m90_3test"],
                                      "trials_ratio": v.get("trials_ratio")},
                                     separators=(",", ":")),
                "metadata_variant": v["variant"], "metadata_m": str(m),
                "predict_trials_to_90_maxdev_only": _fmt(v["m90_maxdev_only"]["m90"]),
                "predict_trials_to_90_3test": _fmt(v["m90_3test"]["m90"]),
                "predict_trials_ratio": _fmt(v.get("trials_ratio")),
                "predict_power_maxdev_only_m" + str(m): _fmt(p_a),
                "predict_power_3test_m" + str(m): _fmt(p_b),
                "predict_baseline_tie_flag": str(
                    v["m90_maxdev_only"]["m90"] == v["m90_3test"]["m90"]),
            })
    return ex


# ---------------------------------------------------------------------------
# Dataset 4: practitioner null
# ---------------------------------------------------------------------------
def _pract_examples(pract: dict) -> list[dict]:
    ex = []
    for cell in pract["cells"]:
        thr = cell["null_threshold"]["threshold_alpha_0.05"]
        ib = cell["null_threshold"]["ib_benchmark_q95"]
        for t in cell["tours"]:
            ex.append({
                "input": (f"Practitioner-scale uniformity check: n={cell['n']}, "
                          f"k={cell['k']}, m={cell['m']} trials with the "
                          f"{t['sampler']} sampler; report maxdev vs the "
                          "simulated alpha=5% null threshold."),
                "output": json.dumps({"maxdev_mean": t["mean_maxdev"],
                                      "maxdev_max": t["max_maxdev"],
                                      "per_rep": t["per_rep_maxdev"],
                                      "verdict": t["verdict"]}, separators=(",", ":")),
                "metadata_sampler": t["sampler"], "metadata_cell": cell["cell"],
                "metadata_n": str(cell["n"]), "metadata_m": str(cell["m"]),
                "predict_maxdev_observed": _fmt(t["mean_maxdev"]),
                "predict_maxdev_threshold_alpha05": _fmt(thr),
                "predict_ib_benchmark_q95": _fmt(ib),
                "predict_verdict": t["verdict"],
                "predict_expected_magnitude": _fmt(
                    cell["null_threshold"]["expected_magnitude_note"].split("=")[-1].strip()),
            })
    return ex


# ---------------------------------------------------------------------------
# Dataset 5: secondary arms
# ---------------------------------------------------------------------------
def _secondary_examples(sec: dict) -> list[dict]:
    ex = []
    for arm in sec.get("arms", []):
        for m, kh in arm["kappa_hat"].items():
            ex.append({
                "input": (f"Secondary arm {arm['arm']}: n={arm['n']}, p={arm['p']}, "
                          f"family={arm['family']}, m={m}: reproduce kappa_derived "
                          "at a new trial budget."),
                "output": json.dumps({"m": int(m), "kappa_hat": kh,
                                      "kappa_derived": arm["kappa_derived"],
                                      "chi2_half_mult":
                                          arm["per_m"][m]["chi2_half_mult"]},
                                     separators=(",", ":")),
                "metadata_arm": str(arm["arm"]), "metadata_m": m,
                "metadata_n": str(arm["n"]), "metadata_p": str(arm["p"]),
                "predict_kappa_hat": _fmt(kh),
                "predict_kappa_derived": _fmt(arm["kappa_derived"]),
            })
    return ex


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def assemble2() -> Path:
    law = _load("half_power_law.json")
    anchors = _load("iter1_anchors.json")
    fwer = _load("fwer_check.json")
    bugs = _load("bug_battery_trials.json")
    pract = _load("practitioner_null.json")
    sec = _load("secondary_arms.json") if (RESULTS / "secondary_arms.json").exists() \
        else {"arms": []}

    datasets = [
        {"dataset": "reservoir_uniformity_law_confirm",
         "examples": _law_examples(law, anchors)},
        {"dataset": "reservoir_uniformity_fwer",
         "examples": _fwer_examples(fwer)},
        {"dataset": "reservoir_uniformity_bug_trials",
         "examples": _bug_examples(bugs)},
        {"dataset": "reservoir_uniformity_practitioner_null",
         "examples": _pract_examples(pract)},
        {"dataset": "reservoir_uniformity_secondary_arms",
         "examples": _secondary_examples(sec)},
    ]
    if not datasets[-1]["examples"]:
        datasets.pop()

    method_out = {
        "metadata": {
            "method_name": "reservoir_uniformity_law_confirm",
            "version": "0.2.0",
            "seed": SEED,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "description": ("Confirm-by-measurement of the trials-to-detection "
                            "design law for reservoir-sampling uniformity checks: "
                            "half-power amplitudes of the four statistics at "
                            "m in {2000,5000,10000}, the corrected 3-test protocol "
                            "FWER, blind bug-battery trials-to-90%, practitioner "
                            "maxdev null at 10^6 trials, and n/p-invariance "
                            "secondary arms. Ports the iteration-1 harness "
                            "unchanged (identical statistic definitions, "
                            "threshold conventions and seeds)."),
            "scale": _SCALE,
            "grid_summary": {
                "law_confirm_cells": len([c for c in law.get("cells", [])
                                          if not c.get("skipped")]),
                "fwer_cells": len([c for c in fwer["cells"] if not c.get("skipped")]),
                "bug_variants": len([v for v in bugs["variants"] if not v.get("skipped")]),
                "practitioner_cells": len(pract["cells"]),
                "secondary_arms": len(sec.get("arms", [])),
            },
            "pre_registration_pointer": "results/pre_registration.json",
            "verdict": law.get("verdict", {}).get("overall_verdict", "MEASURED_LAW"),
            "statistics": [
                "maxdev = max_i |c_i - mu| (mandated report)",
                "chi2 = sum_i (c_i - mu)^2 / mu",
                "trend_slope: OLS slope of c_i on position, |z|-standardized",
                "subwin_energy: max sliding-window sum of squared devs"],
        },
        "datasets": datasets,
    }
    out_path = ROOT / "method_out.json"
    out_path.write_text(json.dumps(method_out, indent=1))
    logger.info(f"wrote {out_path} with {sum(len(d['examples']) for d in datasets)} examples")
    return out_path


def write_readme2() -> Path:
    law = _load("half_power_law.json")
    verdict = law.get("verdict", {}).get("overall_verdict", "MEASURED_LAW")
    txt = f"""# Reservoir uniformity protocol -- iteration 2 (law confirm)

Confirm-by-measurement of the trials-to-detection design law from iteration 1.
Confirm cell n=3000, p=0.05 (k=150); trial budgets m in {{2000, 5000, 10000}};
families {{linear_trend, exp_recency}}.

**Overall verdict: {verdict}** (pre-registered rule in `results/pre_registration.json`).

## Results -> paper mapping

| results/*.json | Paper section |
|---|---|
| `half_power_law.json` (incl. `verdict`, P1/P2/P3) | Methods / Design-law; verdict |
| `power_curves_m_invariance.json` | Figure (m-collapse: power-vs-mult per budget) |
| `fwer_check.json` | Corrected protocol (Sidak FWER ~5%) |
| `bug_battery_trials.json` | Blind battery (trials-to-90% per protocol) |
| `practitioner_null.json` | Sampler deliverable (reviewer MINOR #4; m=10^6) |
| `secondary_arms.json` | n/p-invariance (322x headline etc.) |
| `null_calibration_confirm.json`, `thresholds_confirm.json` | Null calibration |
| `sanity_gates.json`, `sanity_g4_anchor.json` | Reproducibility / gates |
| `iter1_anchors.json`, `pre_registration.json` | Audit trail / pre-registration |

## Quick tour experiments

    uv run method.py demo                                  # core-task demo
    uv run method.py confirm anchors                       # phase 1 (no sim)
    uv run method.py confirm null --workers 6              # phase 2
    uv run method.py confirm fwer --workers 6              # phase 3
    uv run method.py confirm power --workers 6             # phase 4 (+G4 gate)
    uv run method.py confirm law                           # phase 5 (verdict)
    uv run method.py confirm bugs --workers 6              # phase 6
    uv run method.py confirm pract --workers 6             # phase 7
    uv run method.py confirm secondary --workers 6 --arms 1,2   # phase 8
    uv run method.py confirm gates                         # phase 9
    uv run method.py confirm report                        # phase 10

Scaling ladder: `RESERVOIR_CONFIRM_SCALE=mini|third|full uv run method.py ...`.

## Re-running everything

    bash run_confirm.sh            # full confirm pipeline (background-safe)
    uv run python tools/verify_law.py   # independent T5 verification
"""
    p = ROOT / "README.md"
    p.write_text(txt)
    return p


if __name__ == "__main__":  # pragma: no cover
    logger.remove()
    logger.add(lambda _: None, level="INFO")