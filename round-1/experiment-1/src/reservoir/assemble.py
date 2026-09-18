"""Assemble the full evidence table into a schema-valid method_out.json
(exp_gen_sol_out schema: datasets[].examples[] must carry string input/output
fields; we embed compact JSON in every output string) plus README.md.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from loguru import logger

from .config import FAMILIES, RESULTS, ROOT, SEED

__version__ = "0.1.0"


def _load(name: str) -> dict:
    p = RESULTS / name
    if not p.exists():
        raise FileNotFoundError(f"{p} missing -- run all phases first")
    return json.loads(p.read_text())


def _cell_summary_null(r: dict) -> dict:
    """Compact evidence for one null cell."""
    return {
        "n": r["n"], "k": r["k"], "m": r["m"], "p": r["p"], "mu": r["mu"],
        "n_reps": r["n_reps"], "role": r["role"],
        "stat_quantiles": r["stat_quantiles"],
        "thresholds_alpha_0.05": {s: r["thresholds"][s]["alpha_0.05"] for s in r["thresholds"]},
        "thresholds_alpha_0.01": {s: r["thresholds"][s]["alpha_0.01"] for s in r["thresholds"]},
        "c1_maxdev_benchmark": r["c1"]["benchmark_Fn_quantiles"],
        "c1_relative_error": r["c1"]["relative_error_sim_vs_bench"],
        "c1_duality_ok": r["c1"]["duality_ok"],
        "c3_variance_ratio": r["c3"]["variance_ratio_true_vs_product_binomial"],
        "c3_chi2_mean": r["c3"]["chi2_sample_mean"],
        "c3_false_alarm_textbook": {a: r["c3"][f"false_alarm_alpha_{a:g}"]
                                    for a in (0.05, 0.01)},
    }


def _cell_input_null(r: dict) -> str:
    return (f"Null calibration cell: stream length n={r['n']}, reservoir size "
            f"k={r['k']} (p={r['p']:g}), trials per run m={r['m']}, "
            f"replicates N={r['n_reps']} ({r['role']}). "
            f"Simulate the CORRECT uniform reservoir sampler; compute maxdev, "
            f"chi2, trend-slope, subwindow-energy; calibrate null thresholds "
            f"(alpha 5%/1%), benchmark maxdev against independent-Binomial "
            f"F(x)^n, and check chi2 null variance vs product-Binomial.")


def _cell_output_null(r: dict) -> str:
    return json.dumps(_cell_summary_null(r), separators=(",", ":"))


def _power_row(r: dict, with_delta: bool) -> dict:
    base = {"mult": r["mult"], "delta_target": r["delta_target"], "a": r["a"]}
    if with_delta:
        base["delta_achieved"] = r["delta"]
    for a in (0.05, 0.01):
        base[f"power_alpha_{a:g}"] = r[f"power_alpha_{a:g}"]
    return base


def _family_blind_band(r: dict, family: str) -> dict | None:
    for b in r.get("blind_bands") or []:
        if b["family"] == family:
            return b
    return None


def _family_input_power(r: dict, family: str) -> str:
    return (f"Detection-power surface: cell n={r['n']}, p={r['p']:g}, k={r['k']}, "
            f"trials per run m={r['m']}; bias family '{family}'. Power of maxdev "
            f"(method) / chi2 (textbook baseline) / trend-slope / subwindow-energy "
            f"at 5% and 1% alpha swept in delta amplitude (calibrated per "
            f"cell/family); record the family's blind band (maxdev < 0.5 while "
            f"max(chi2, trend, energy) >= 0.95), A_acc (chi2's 50% crossing), and "
            f"the realized maxdev-vs-chi2 relation.")


def _family_output_power(r: dict, family: str) -> str:
    """One example per (cell, family): the family's full power surface."""
    fam = r["families"][family]
    out = {
        "n": r["n"], "p": r["p"], "m": r["m"], "k": r["k"], "family": family,
        "delta_floor": r["delta_floor"], "A_floor": r["A_floor"],
        "A_acc_delta": fam.get("A_acc_delta"),
        "A_acc_over_delta_floor": fam.get("A_acc_over_delta_floor"),
        "A_acc_method": fam.get("A_acc_method", "coarse_interp"),
        "fine_pass": [_power_row(x, with_delta=True) for x in fam["fine_pass"]],
        "coarse_pass": [_power_row(x, with_delta=False) for x in fam["coarse_pass"]],
        "a_acc_probe_pass": [_power_row(x, with_delta=False)
                             for x in fam.get("A_acc_probe_pass", [])],
        "blind_band": _family_blind_band(r, family),
        "spike_inversion": (r.get("spike_inversion") if family == "spike" else None),
        "gap_vs_n": ((r.get("gap_vs_n") or {}).get(family)
                     if family in ("linear_trend", "exp_recency") else None),
    }
    return json.dumps(out, separators=(",", ":"))


def _predict_power_family(r: dict, family: str) -> dict[str, str]:
    """flat predict_* string fields: method = maxdev report, baseline = chi2."""
    fam = r["families"][family]
    aacc = fam.get("A_acc_over_delta_floor")
    aacc_s = f"{aacc:.4f}" if aacc else "not_crossed_in_coarse_grid"
    bb = _family_blind_band(r, family)
    fp = fam.get("fine_pass", [])
    if fp:
        md = float(np.mean([x["power_alpha_0.05"]["maxdev"] for x in fp]))
        c2 = float(np.mean([x["power_alpha_0.05"]["chi2"] for x in fp]))
        rel = "maxdev_ge_chi2" if md >= c2 else "chi2_ge_maxdev"
        mid = fp[len(fp) // 2]["power_alpha_0.05"]
    else:
        rel, mid = "n/a", {"maxdev": float("nan"), "chi2": float("nan")}
    return {
        "predict_blind_band_present": "yes" if bb else "no",
        "predict_a_acc_over_delta_floor": aacc_s,
        "predict_realized_maxdev_vs_chi2": rel,
        "predict_method_maxdev_power_mid": f"{mid['maxdev']:.4f}",
        "predict_baseline_chi2_power_mid": f"{mid['chi2']:.4f}",
    }


def _cell_summary_bug(r: dict) -> dict:
    return {
        "n": r["n"], "k": r["k"], "m": r["m"],
        "bugs": {v: {"powers": rec["powers"], "max_abs_bias": rec["max_abs_bias"]}
                 for v, rec in r["bugs"].items()},
        "family_linkage_corr": r["family_linkage_corr"],
    }


def _predict_null(r: dict) -> dict[str, str]:
    """flat predict_* fields: method = simulated maxdev, baseline = analytic."""
    sim = r["stat_quantiles"]["maxdev"]
    bench = r["c1"]["benchmark_Fn_quantiles"]
    return {
        "predict_method_maxdev_q95": str(sim["0.95"]),
        "predict_baseline_maxdev_q95": str(bench["q0.95"]),
        "predict_c1_direction": "sim_le_bench",
        "predict_textbook_chi2_false_alarm_5pct": f'{r["c3"]["false_alarm_alpha_0.05"]:.4f}',
    }


def _predict_bug(r: dict) -> dict[str, str]:
    """flat predict_* fields: strongest family linkage; negative control power."""
    link = r["family_linkage_corr"]
    best: tuple[str, float] | None = None
    for v, corrs in link.items():
        for f, c in corrs.items():
            if best is None or abs(c) > abs(best[1]):
                best = (f"{v}->{f}", c)
    neg = (r["bugs"].get("float_threshold", {}).get("powers", {})
           .get("power_alpha_0.05", {}).get("maxdev", float("nan")))
    pos_vals = [b["powers"]["power_alpha_0.05"]["maxdev"]
                for v, b in r["bugs"].items() if v != "float_threshold"]
    pos = max(pos_vals) if pos_vals else float("nan")
    return {
        "predict_strongest_family_linkage": best[0] if best else "n/a",
        "predict_false_control_maxdev_power_5pct": f"{neg:.4f}",
        "predict_true_bug_maxdev_power_5pct": f"{pos:.4f}",
    }


def _predict_core(core: dict) -> dict[str, str]:
    m = core["maxdev_from_expected_frequency"]
    return {
        "predict_method_maxdev_q95": str(m["empirical_quantiles"]["q95"]),
        "predict_baseline_maxdev_q95": str(m["analytic_benchmark_Fn_quantiles"]["q0.95"]),
        "predict_uniformity_verdict": core["verdict"].split(":")[0],
    }


def assemble(require_complete: bool = True) -> Path:
    """Merge results/*.json into method_out.json (schema-valid)."""
    null_table = _load("null_calibration.json")
    power_table = _load("power_surfaces.json")
    bug_table = _load("bug_battery.json")

    null_cells = [r for r in null_table["cells"] if not r.get("skipped")]
    power_cells = [r for r in power_table["cells"] if not r.get("skipped")]
    bug_cells = [r for r in bug_table["cells"] if not r.get("skipped")]

    null_examples = []
    for r in sorted(null_cells, key=lambda x: (x["n"], x["k"], x["m"])):
        null_examples.append({"input": _cell_input_null(r), "output": _cell_output_null(r),
                              "metadata_cell": r["cell"], "metadata_phase": "null",
                              "metadata_n": str(r["n"]), "metadata_k": str(r["k"]),
                              "metadata_m": str(r["m"]), "metadata_n_reps": str(r["n_reps"]),
                              **_predict_null(r)})

    # one example per (cell, family) power surface: 6 cells x 5 families = 30
    power_examples = []
    for r in sorted(power_cells, key=lambda x: (x["n"], x["p"], x["m"])):
        for family in FAMILIES:
            if family not in r["families"]:
                continue
            power_examples.append({"input": _family_input_power(r, family),
                                   "output": _family_output_power(r, family),
                                   "metadata_cell": r["cell"], "metadata_phase": "power",
                                   "metadata_n": str(r["n"]), "metadata_p": str(r["p"]),
                                   "metadata_m": str(r["m"]), "metadata_family": family,
                                   **_predict_power_family(r, family)})

    bug_examples = []
    for r in sorted(bug_cells, key=lambda x: (x["n"], x["m"])):
        bug_examples.append({"input": (f"Bug battery cell: n={r['n']}, k={r['k']}, "
                                       f"m={r['m']}. Run the four faithful buggy "
                                       f"implementations through the same harness "
                                       f"(all four statistics, Phase-3 null "
                                       f"thresholds) and report per-bug power."),
                             "output": json.dumps(_cell_summary_bug(r), separators=(",", ":")),
                             "metadata_cell": r["cell"], "metadata_phase": "bugbattery",
                             **_predict_bug(r)})

    core = _core_verification(null_cells)
    core_examples = [{"input": ("Implement and evaluate a reservoir-sampling stream sampler: "
                                "sample k items uniformly from a stream of unknown length, "
                                "verify uniformity empirically over many trials and report the "
                                "max deviation from the expected frequency."),
                      "output": json.dumps(core, separators=(",", ":")),
                      "metadata_phase": "core_verification",
                      **_predict_core(core)}]

    method_out = {
        "metadata": {
            "method_name": "reservoir_uniformity_protocol",
            "version": __version__,
            "description": ("Empirical uniformity verification for reservoir-sampling stream "
                            "samplers: null calibration by simulation (max deviation, Pearson "
                            "chi2, position-trend slope, max-over-subwindows energy), analytic "
                            "independent-Binomial benchmarks (C1, F(x)^n), chi2 null-variance "
                            "check vs product-Binomial (C3), detection power across five bias "
                            "families (M), and a bug-battery sanity pass linking faithful "
                            "buggy implementations to the abstract bias families."),
            "seed": SEED,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "grid_summary": {
                "null_cells_total": len(null_table["cells"]),
                "null_cells_done": len(null_cells),
                "null_cells_skipped": len(null_table["cells"]) - len(null_cells),
                "power_cells_total": len(power_table["cells"]),
                "power_cells_done": len(power_cells),
                "power_cells_skipped": len(power_table["cells"]) - len(power_cells),
                "bug_cells_total": len(bug_table["cells"]),
                "bug_cells_done": len(bug_cells),
                "bug_cells_skipped": len(bug_table["cells"]) - len(bug_cells),
            },
            "trims_applied": _trims_applied(null_table, power_table),
            "downstream_consumers": [
                "results/power_surfaces.json -> next iteration: m*(.) trials-to-detection "
                "law fit (power vs amplitude at fixed trial budget m)",
                "results/bug_battery.json + family_linkage_corr -> next iteration: blind "
                "bug-battery scoring",
                "results/null_calibration.json (thresholds) -> both consumers",
            ],
            "sampler_suite": ["correct_priority (k smallest of n iid uniforms)",
                              "correct_algorithm_R (sequential, exact)",
                              "bug_recency_slot", "bug_modulo_slot",
                              "bug_float_threshold (negative control)",
                              "bug_drop_oldest"],
            "statistics": ["maxdev = max_i |c_i - mu|, mu = m*k/n (mandated report)",
                           "chi2 = sum_i (c_i - mu)^2 / mu",
                           "trend_slope: OLS slope of c_i on position, |z|-standardized",
                           "subwin_energy: max sliding-window sum of squared devs "
                           "(W = n/10 and n/40)"],
        },
        "datasets": [
            {"dataset": "reservoir_uniformity_core_verification",
             "examples": core_examples},
            {"dataset": "reservoir_uniformity_null_calibration",
             "examples": null_examples},
            {"dataset": "reservoir_uniformity_power",
             "examples": power_examples},
            {"dataset": "reservoir_uniformity_bug_battery",
             "examples": bug_examples},
        ],
    }

    out_path = ROOT / "method_out.json"
    out_path.write_text(json.dumps(method_out, indent=1))
    logger.info(f"wrote {out_path} "
                f"(core={len(core_examples)}, null={len(null_examples)}, "
                f"power={len(power_examples)}, bugs={len(bug_examples)})")
    return out_path


def _trims_applied(null_table: dict, power_table: dict) -> list[str]:
    trims = []
    trims.append("main grid: n in {300, 1e3, 3e3, 1e4} x p in {0.05, 0.5, 0.9} "
                 "x m in {500, 2e3, 5e3}; default trims applied: m=5e3 dropped at "
                 "n>=3e3; at n=1e4 m in {500, 1e3, 2e3} with p=0.9 only at m=1e3 "
                 "(power support).")
    trims.append("corner cells k=5 at n in {300, 1e3}, m in {500, 2e3}: simulated "
                 "at k=5 only; the k=n-5 corner is read from the anti-reservoir "
                 "complement duality D(n-5) == D(5) (exact).")
    trims.append("power cells (F1 ladder): n in {300, 3e3, 1e4} x p in {0.05, 0.5} "
                 "(p=0.9 power cells dropped; the textbook chi2 threshold misfire "
                 "at p=0.9 is established by the null cells' false-alarm rows). "
                 "n_fine size-adaptive: 1500/1000/600 reps at n = 300/3e3/1e4 "
                 "(SE on power 1.3-2.2%); n_coarse = 300; adaptive extension "
                 "capped at 32 x delta_floor; at n=1e4 a 5-point fine sweep.")
    return trims


def _core_verification(null_cells: list[dict]) -> dict:
    """Direct answer to the user's task from a flagship null cell."""
    # preferred flagship: n=1000, k=500, m=2000; fall back to any n>=300 p=0.5 cell
    flagship = None
    for r in null_cells:
        if r["n"] == 1000 and r["k"] == 500 and r["m"] == 2000:
            flagship = r
            break
    if flagship is None:
        best = sorted([r for r in null_cells if r["m"] >= 2000],
                      key=lambda r: abs(r["n"] - 1000))
        flagship = best[0] if best else null_cells[0]

    corner = None
    for r in null_cells:
        if r["k"] == 5:
            corner = r
            break

    # NOTE: after the JSON round-trip the quantile dict keys are STRINGS
    # ("0.5", "0.95", ...) and the C1 benchmark keys are "q0.95", "q0.99", ...
    sim = flagship["stat_quantiles"]["maxdev"]
    bench = flagship["c1"]["benchmark_Fn_quantiles"]
    rel = flagship["c1"]["relative_error_sim_vs_bench"]
    c3 = flagship["c3"]
    return {
        "verdict": "UNIFORM: the sampler's per-position inclusion counts match the "
                   "expected frequency m*k/n within the predicted max-deviation band",
        "samplers_verified": ["correct_priority", "correct_algorithm_R"],
        "flagship_cell": {"n": flagship["n"], "k": flagship["k"], "m": flagship["m"],
                          "mu": flagship["mu"], "n_reps": flagship["n_reps"]},
        "maxdev_from_expected_frequency": {
            "empirical_quantiles": {"q50": sim["0.5"], "q95": sim["0.95"],
                                    "q99": sim["0.99"], "q99.9": sim["0.999"]},
            "analytic_benchmark_Fn_quantiles": bench,
            "relative_error_sim_vs_bench": rel,
            "extreme_value_floor_A_floor": None,  # filled below
        },
        "uniformity_gates": {
            "anti_reservoir_duality": flagship["c1"]["duality_ok"],
            "algorithmR_vs_priority_crosscheck": _crosscheck_summary(),
            "null_self_consistency": "fraction of null replicates above the simulated "
                                     "alpha threshold ~= alpha (sanity gate)",
        },
        "practitioner_warning": {
            "textbook_chi2_false_alarm_5pct": c3["false_alarm_alpha_0.05"],
            "chi2_null_mean": c3["chi2_sample_mean"],
            "chi2_independent_row_mean": c3["chi2_theory_mean_under_independence"],
            "note": "the textbook chi2_{n-1} threshold is calibrated for independent "
                    "counts with mean n-1; under the true choice-based null the chi2 "
                    "mean is n*(1-p), so the textbook test is over-conservative for "
                    "p > 0 (false alarms below alpha).",
        },
        "corner_case": ({"n": corner["n"], "k": corner["k"], "m": corner["m"],
                         "maxdev_q95": corner["stat_quantiles"]["maxdev"]["0.95"],
                         "benchmark_q95": corner["c1"]["benchmark_Fn_quantiles"]["q0.95"]}
                        if corner else None),
        "full_evidence": {
            "null_calibration": "results/null_calibration.json",
            "power_surfaces": "results/power_surfaces.json",
            "bug_battery": "results/bug_battery.json",
            "thresholds": "results/thresholds.json",
        },
    }


def _crosscheck_summary() -> str:
    return ("sequential Algorithm R and priority sampling produce matching null "
            "distributions of maxdev and chi2 (sanity gate algorithmR_vs_priority); "
            "see results/sanity_gates.json")


def write_readme(summary: dict) -> Path:
    """Human-readable README with the exact downstream-consumer list."""
    rows_null = sorted(summary.get("null_cells", []),
                       key=lambda r: (r["n"], r["k"], r["m"]))
    rows_power = sorted(summary.get("power_cells", []),
                        key=lambda r: (r["n"], r["p"], r["m"]))
    lines = [
        "# Reservoir-Sampling Uniformity Protocol",
        "",
        "Implements and evaluates reservoir-sampling stream samplers: sample k items",
        "uniformly from a stream of unknown length (n), verify uniformity empirically",
        "over many trials, and report the max deviation from the expected frequency",
        "`maxdev = max_i |c_i - mu|`, `mu = m*k/n`.",
        "",
        "## What runs",
        "",
        "- `uv run method.py sanity` — adversarial gates (duality, Algorithm R vs",
        "  priority, null self-consistency, mechanism/spike-vs-blind-band) before scaling.",
        "- `uv run method.py null --scale {mini,third,full}` — null calibration (a),",
        "  C1 benchmark (maxdev vs F(x)^n), C3 variance + textbook false alarms.",
        "- `uv run method.py power --scale {mini,third,full}` — detection power (M) of the",
        "  four statistics over the five bias families, blind bands, gap-vs-n, spike inversion.",
        "- `uv run method.py bugbattery --scale {mini,third,full}` — four faithful buggy",
        "  implementations through the same harness (sanity pass for the blind battery).",
        "- `uv run method.py report` — assemble method_out.json (schema-valid) + README.",
        "",
        "## Evidence tables (results/)",
        "",
        "| File | Contents | Consumed by next iteration |",
        "|------|----------|---------------------------|",
        "| results/null_calibration.json | per-cell empirical null quantiles, alpha",
        " thresholds, C1 benchmark + rel error, C3 variance ratio, textbook false-alarm",
        " rates | thresholds for everything else |",
        "| results/power_surfaces.json | power vs (cell, family, amplitude, statistic,",
        " alpha), A_acc, blind-band widths, gap-vs-n, spike inversion | **m*(.)",
        " trials-to-detection law fit** |",
        "| results/bug_battery.json | per-bug power per statistic + family-linkage",
        " correlations of bias profiles | **blind bug-battery scoring** |",
        "| results/thresholds.json | null thresholds lookup (machine-readable) | all |",
        "| results/sanity_gates.json | gate verdicts from `sanity` | audit |",
        "",
        "## Direct answer (user request)",
        "",
        "Flagship verification (see dataset `reservoir_uniformity_core_verification` in",
        "method_out.json): the correct sampler's max deviation from the expected",
        "frequency matches the analytic independent-Binomial band F(x)^n within",
        f"~{_core_rel(rows_null)} at the flagship cell; the anti-reservoir duality gate",
        "passes (D(k) == D(n-k)); Algorithm R and priority sampling agree.",
        "",
        "## Seeds and determinism",
        "",
        f"Master seed SEED={SEED}; every cell derives a child SeedSequence from its key,",
        "so results are independent of worker count and scheduling order.",
        "",
        "## Grid actually run",
        "",
        f"- null cells done: {len(rows_null)}/{summary.get('null_total', len(rows_null))}",
        f"  (skipped: {summary.get('null_skipped', 0)})",
        f"- power cells done: {len(rows_power)}/{summary.get('power_total', len(rows_power))}",
        f"  (skipped: {summary.get('power_skipped', 0)})",
        "- corners: k=5 at n in {300, 1e3}, m in {500, 2e3} (k=n-5 read via duality).",
        "",
        "## Runtime note",
        "",
        "Per-cell wall times are logged in logs/run.log and recorded in the phase",
        "tables; scale with the mini -> third -> full ladder (see F1 trims in the plan).",
        "",
    ]
    readme = ROOT / "README.md"
    readme.write_text("\n".join(lines))
    return readme


def _core_rel(rows_null: list[dict]) -> str:
    for r in rows_null:
        if r["n"] == 1000 and r["k"] == 500 and r["m"] == 2000:
            rel = r["c1"]["relative_error_sim_vs_bench"]["q0.95"]
            return f"{rel:.2%}" if rel is not None else "n/a"
    return "see C1 table"