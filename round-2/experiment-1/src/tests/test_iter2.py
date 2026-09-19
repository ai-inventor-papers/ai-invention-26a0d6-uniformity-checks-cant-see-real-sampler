"""Unit tests for the iteration-2 confirm machinery (testing plan T0).

Covers: delta-calibration monotonicity, threshold quantile correctness,
the log-mult crossing finder, joint-rejection (FWER) counting, probit
extrapolation, practitioner benchmark summarization and grid dedupe.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from reservoir.confirm import BONF_ALPHA, SIDAK_ALPHA, alpha_label
from reservoir.confirm_power import (smart_mults, _crossing_logmult, _bracket,
                                     _missing_brackets)
from reservoir.bugs2 import m90_from_two_points, m90_two_point
from reservoir.pract import ib_benchmark_restricted
from reservoir.benchmarks import maxdev_benchmark_quantiles


def test_sidak_bonf_levels():
    assert SIDAK_ALPHA == pytest.approx(1 - 0.95 ** (1 / 3), rel=1e-12)
    assert BONF_ALPHA == pytest.approx(0.05 / 3, rel=1e-12)
    assert alpha_label(SIDAK_ALPHA) == "sidak"
    assert alpha_label(0.05) == "0.05"


def test_crossing_logmult_known():
    # synthetic curve: power(mult) = mult / (mult + 0.1) -> crossing at
    # power=0.5 <=> mult = 0.1
    mults = [0.02, 0.05, 0.2, 0.5]
    rows = [{"mult": m, "powers": {"0.05": {"chi2": m / (m + 0.1)}}} for m in mults]
    cr = _crossing_logmult(rows, "chi2", "0.05")
    assert cr == pytest.approx(0.1, rel=0.05)
    br = _bracket(rows, "chi2", "0.05")
    assert br is not None and rows[br[0]]["mult"] < 0.1 < rows[br[1]]["mult"]


def test_crossing_no_bracket_reports_missing():
    st = {"maxdev": 0.2, "chi2": 0.2, "trend_slope": 0.2, "energy": 0.2}
    row = {"mult": 0.1, "powers": {"0.05": dict(st), "sidak": dict(st),
                                   "bonf": dict(st)}}
    rows = [row]
    assert _crossing_logmult(rows, "chi2", "0.05") is None
    assert _missing_brackets(rows)  # non-empty: needs extension


def test_smart_mults_dedupe_and_safety():
    mults = smart_mults(Aacc_mult=0.0827, mdev_mult=0.313)
    assert mults == sorted(mults)
    # safety point guarantees saturation
    assert max(mults) >= 2.0 * max(0.0827, 0.313)
    # no adjacent pair closer than 5% relative (post-dedupe)
    for a, b in zip(mults, mults[1:]):
        assert b / a > 1.0 + 0.049


def test_amplitude_calibration_monotone():
    from reservoir.harness import _calibrate_amplitude
    from reservoir.config import child_rng, family_shape
    amps, deltas = _calibrate_amplitude(300, 15, 500, 0.05,
                                        family_shape("linear_trend", 300),
                                        child_rng("test_calib"))
    assert np.all(np.diff(deltas) > 0)
    assert len(amps) == len(deltas) == 4


def test_joint_rejection_hand():
    """FWER counting: hand-built threshold crossing flags."""
    arrs = {"maxdev": np.array([10.0, 1.0, 1.0]),
            "chi2": np.array([1.0, 10.0, 1.0]),
            "trend_slope": np.array([1.0, 1.0, 10.0])}
    th = 5.0
    joint = (arrs["maxdev"] > th) | (arrs["chi2"] > th) | (arrs["trend_slope"] > th)
    assert joint.tolist() == [True, True, True]
    joint0 = (arrs["maxdev"] > 100.0) | (arrs["chi2"] > 100.0) | (arrs["trend_slope"] > 100.0)
    assert not joint0.any()


def test_evaluate_g4_reads_alpha05_subrecord():
    """G4 must compare the alpha=0.05 crossing record (regression: the
    evaluator previously read the per-alpha dict instead of ['0.05'] and
    every comparison saw None -> gate always failed)."""
    from reservoir.confirm_power import evaluate_g4_anchors
    anchors_fam = {fam: {"anchors": {
        "maxdev": {"kind": "fine_interp", "value_delta": 1.0, "mult": 0.3},
        "chi2": {"kind": "a_acc", "value_delta": 0.5, "mult": 0.25},
        "trend_slope": {"kind": "upper_bound", "value_delta": 0.2, "mult": 0.05},
        "energy": {"kind": "probe_interp", "value_delta": 0.4, "mult": 0.2}}}
                   for fam in ("linear_trend", "exp_recency")}
    cells = [{"family": fam, "m": 2000, "crossings": {
        "maxdev": {"0.05": {"A_half_delta": 1.04, "mult": 0.31}},
        "chi2": {"0.05": {"A_half_delta": 0.52, "mult": 0.26}},
        "trend_slope": {"0.05": {"A_half_delta": 0.02, "mult": 0.03}},
        "energy": {"0.05": {"A_half_delta": 0.40, "mult": 0.19}}}}
             for fam in ("linear_trend", "exp_recency")]
    g4 = evaluate_g4_anchors(anchors_fam, cells)
    assert g4["ok"] is True, g4
    # and it must FAIL when the measured value drifts > 10%
    cells[0]["crossings"]["chi2"]["0.05"]["A_half_delta"] = 0.8
    g4_bad = evaluate_g4_anchors(anchors_fam, cells)
    assert g4_bad["ok"] is False


def test_binom_restricted_matches_full():
    """Restricted-grid IB benchmark agrees with the full-grid version (small m)."""
    taus = (0.95, 0.99, 0.999)
    full = maxdev_benchmark_quantiles(1000, 2000, 50, taus=taus)
    restr = ib_benchmark_restricted(1000, 2000, 50, taus=taus)
    for t in full:
        assert restr[t] == pytest.approx(full[t], abs=1.0)  # lattice-unit tolerance


def test_m90_probit_interp():
    # power 0.05 at m=5000, 0.997 at m=10000 -> m90 between the two
    r = m90_two_point("x", {5000: 0.05, 10000: 0.997})
    assert 5000 < r["m90"] < 10000
    assert r["method"] == "probit_interp"
    # saturated at 5000
    r2 = m90_two_point("x", {5000: 0.95, 10000: 1.0})
    assert r2["m90"] == "<=5000"
    # extrapolated (both low)
    r3 = m90_two_point("x", {5000: 0.2, 10000: 0.4})
    assert r3["method"] == "EXTRAPOLATED" and r3["m90"] > 10000
    # symmetric check: interpolation at target power (probit-linear in log10 m)
    from scipy.stats import norm
    z = lambda p: norm.ppf(p)  # noqa: E731
    expected = 10 ** (np.log10(5000) + (z(0.9) - z(0.5)) / (z(0.99) - z(0.5))
                      * (np.log10(10000) - np.log10(5000)))
    m = m90_from_two_points(5000, 0.5, 10000, 0.99, target=0.9)
    assert m["m90"] == pytest.approx(expected, rel=0.01)


def test_null_self_consistency_threshold_is_quantile():
    """The alpha=0.05 threshold of a batch is its 0.95 quantile by construction."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=2000)
    thr = np.quantile(x, 0.95)
    assert np.mean(x > thr) == pytest.approx(0.05, abs=0.01)


def test_secondary_null_chunks_spawn_tuple_unpack():
    """Regression: _null_chunk must honour run_parallel's single-cell contract
    (a 4-positional signature raised TypeError in every spawned chunk and the
    phase died with 'all sec-null chunks failed for arm 1 m=2000')."""
    from reservoir.secondary import _null_for
    arm = {"arm": 99, "n": 120, "p": 0.25, "m_list": (200,), "n_null": 60,
           "n_fine": 40, "family": "linear_trend", "anchor_cell": "x"}
    null = _null_for(arm, 200, workers=2)
    assert null["trend_ref"]["sd"] > 0
    # empirical null mean of the raw slope: ~0 up to sampling noise (~sd/sqrt(N))
    assert abs(null["trend_ref"]["mu"]) < 0.05
    for s in ("maxdev", "chi2", "trend_slope", "energy"):
        assert s in null["thresholds"]
        assert null["thresholds"][s] > 0


def test_law_family_keying_no_alias(tmp_path):
    """Regression: law cells are keyed by (m, family); an m-only key silently
    aliases the two families sharing each budget (spurious verdicts)."""
    import reservoir.law as law_mod
    from reservoir.config import RESULTS as REAL_RESULTS
    from reservoir.confirm import CONFIRM_FAMILIES, CONFIRM_M
    from reservoir.law import run_law_tests

    # redirect the law module's RESULTS to a tmp dir so the real
    # results/half_power_law.json checkpoint is never touched by the test
    law_mod.RESULTS = tmp_path
    (tmp_path / "iter1_anchors.json").write_text(
        json.dumps(json.loads((REAL_RESULTS / "iter1_anchors.json").read_text())))
    real_before = (REAL_RESULTS / "half_power_law.json").read_text()

    DELTA = {2000: 0.4002, 5000: 0.2531, 10000: 0.1789}  # real delta_floor(m)
    half = {
        "generated_at": "t", "n": 3000, "p": 0.05, "m_list": list(CONFIRM_M),
        "cells": [
            {"cell": f"power_cell_m{m}_{fam}", "m": m, "family": fam,
             "delta_floor": DELTA[m], "n_fine": 1000, "kappa_hat": kappa_f,
             "crossings": {
                 # chi2 half-power mult = 1/sqrt(kappa_hat), A_half = mult*delta
                 "chi2": {"0.05": {"A_half_delta": DELTA[m] / kappa_f ** 0.5,
                                   "mult": 1.0 / kappa_f ** 0.5},
                          "sidak": {"mult": 1.2 / kappa_f ** 0.5},
                          "bonf": {"mult": 1.2 / kappa_f ** 0.5}},
                 "maxdev": {"0.05": {"A_half_delta": DELTA[m] * mdev_f,
                                     "mult": mdev_f},
                            "sidak": {"mult": mdev_f * 1.2},
                            "bonf": {"mult": mdev_f * 1.2}},
                 "trend_slope": {"0.05": {"A_half_delta": 0.3 * DELTA[m] / kappa_f ** 0.5,
                                          "mult": 0.3 / kappa_f ** 0.5},
                                 "sidak": {"mult": 1.0}, "bonf": {"mult": 1.0}},
                 "energy": {"0.05": {"A_half_delta": 1.2 * DELTA[m] / kappa_f ** 0.5,
                                     "mult": 1.2 / kappa_f ** 0.5},
                            "sidak": {"mult": 1.0}, "bonf": {"mult": 1.0}}}}
            for m in CONFIRM_M
            for fam, kappa_f, mdev_f in (
                # real measured kappa_hat_per_m and maxdev mults per family
                ("linear_trend", {2000: 152.8, 5000: 154.5, 10000: 153.0}[m],
                 {2000: 0.3261, 5000: 0.2976, 10000: 0.2987}[m]),
                ("exp_recency", {2000: 65.9, 5000: 63.9, 10000: 67.6}[m],
                 {2000: 0.3638, 5000: 0.3506, 10000: 0.3625}[m]))
        ],
    }
    law = run_law_tests(half, force=True)
    # the real checkpoint must be untouched (test isolation)
    assert (REAL_RESULTS / "half_power_law.json").read_text() == real_before
    # Real-measured kappa_hat (~153 linear vs derived 146; ~65 exp vs 63.6) is
    # within the 25% band for BOTH families -> both CONFIRMED under the CORRECT
    # (m, family) keying.  An m-only key aliases both families to the exp cell
    # (kappa ~65): linear_trend then reads 65 vs 146 -> P3 fails and the verdict
    # drops to PARTIALLY_CONFIRMED, which this test would catch.
    for fam in CONFIRM_FAMILIES:
        fam_out = law["families"][fam]
        assert fam_out.get("verdict") == "CONFIRMED", \
            f"family {fam} verdict {fam_out.get('verdict')} (family alias bug?)"