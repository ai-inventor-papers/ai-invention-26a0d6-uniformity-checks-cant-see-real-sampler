"""Iteration-2 confirm-phase configuration and pre-registration constants.

Confirm cell: n=3000, p=0.05 (k=150), families {linear_trend, exp_recency},
trial budgets m in {2000, 5000, 10000}.  Everything the law tests measure
against is defined HERE so that the verdict is computed from pre-stated
tolerances, never improvised after seeing the numbers (saved verbatim into
results/pre_registration.json by reservoir/anchors.py).
"""
from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Confirm cell geometry
# ---------------------------------------------------------------------------
CONFIRM_N = 3000
CONFIRM_P = 0.05
CONFIRM_K = 150
CONFIRM_M = (2000, 5000, 10000)
CONFIRM_FAMILIES = ("linear_trend", "exp_recency")

# Correction level for the 3-test protocol {maxdev, chi2, trend|z|}: per-test
# alpha' such that 1 - (1 - alpha')^3 = 0.05 (Sidak).  Plus the Bonferroni
# variant alpha/3.
SIDAK_ALPHA = 1.0 - 0.95 ** (1.0 / 3.0)          # ~0.016952
BONF_ALPHA = 0.05 / 3.0                            # ~0.016667
CONFIRM_ALPHAS = (0.05, SIDAK_ALPHA, BONF_ALPHA, 0.01)
POWER_ALPHAS2 = (0.05, SIDAK_ALPHA, BONF_ALPHA)   # power evaluated at these

# Replicate budgets (default = FULL scale; the RESERVOIR_CONFIRM_SCALE env var
# selects the aii-long-running-tasks ladder: mini -> third -> full, by scaling
# the constants below at import time).
N_NULL_REPS = 2500      # Phase 2 null calibration per m (SE <= 1%)
N_FWER_REPS = 3000      # Phase 3 FWER replicates per m (SE ~= 0.4%)
N_FINE_MID = 1000       # Phase 4 fine power reps at m = 2000 -- pre-registered
N_FINE_5000 = 1000      # Phase 4 fine power reps at m = 5000 -- pre-registered.
                        #   (A transient stale-session trim to 450 was reverted
                        #   on 2026-09-19; the plan mandates 1000.)
N_FINE_BIG = 800        # Phase 4 fine power reps at m = 10000 -- pre-registered
                        #   (stale-session trim to 350 reverted; plan mandates 800)
N_BISECT = 500          # coarse reps for bracket refinement / extension
N_BUG_REPS = 800        # Phase 6 bug battery reps per (variant, m)
N_SEC_NULL = 1500       # Phase 8 secondary-arm null reps per m (arm tables use
                        #   per-arm n_null; this is the generic default)
N_SEC_FINE = 800        # Phase 8 secondary-arm power reps
N_PRACT_A = 300         # Phase 7 null reps at n=1000 (cell A; SE ~1.4% at alpha)
N_PRACT_B = 200         # Phase 7 null reps at n=3000 (cell B; SE ~1.8%)
N_PRACT_SAMP = 5        # sampler tours per cell in Phase 7

import os as _os

_SCALE = _os.environ.get("RESERVOIR_CONFIRM_SCALE", "full").strip().lower()
if _SCALE == "mini":
    N_NULL_REPS, N_FWER_REPS = 300, 600
    N_FINE_MID, N_FINE_5000, N_FINE_BIG, N_BISECT = 300, 250, 200, 200
    N_BUG_REPS = 200
    N_SEC_NULL, N_SEC_FINE = 400, 250
    N_PRACT_A, N_PRACT_B, N_PRACT_SAMP = 30, 20, 2
elif _SCALE == "third":
    N_NULL_REPS, N_FWER_REPS = 800, 1200
    N_FINE_MID, N_FINE_5000, N_FINE_BIG, N_BISECT = 500, 400, 300, 300
    N_BUG_REPS = 400
    N_SEC_NULL, N_SEC_FINE = 700, 400
    N_PRACT_A, N_PRACT_B, N_PRACT_SAMP = 100, 60, 3

# Smart-grid multipliers (plan PHASE 4.3: ~7-9 points per cell; the union of
# the accumulated and maxdev lists is deduped within 5% plus one saturation
# safety point).  Every statistic's crossing stays bracketed: chi2/trend/
# energy live in [0.03, 0.19] mult units (covered by acc x {0.4,1.0,1.7} and
# mdev x 0.5), maxdev in [0.15, 0.7] (covered by mdev x {0.5,1.0,1.7,2.2} and
# the safety point); bisection refines each bracket to |power-0.5| <= 0.15.
ACC_MULT_STEPS = (0.4, 1.0, 1.7)
MDEV_MULT_STEPS = (0.5, 1.0, 1.7)
GRID_DEDUPE_TOL = 0.05  # merge points within 5% relative
SAFETY_GRID_MULT = 2.0  # guarantee saturation for the upper bracket
EXTEND_MAX_STEPS = 2    # amplitude extension steps (x2 per step)
EXTEND_STEP = 2.0

# Threshold-application alpha keys used in power rows
ALPHA_KEYS = {"0.05": 0.05, "sidak": SIDAK_ALPHA, "bonf": BONF_ALPHA}

# ---------------------------------------------------------------------------
# Pre-registered tolerances (verbatim into pre_registration.json)
# ---------------------------------------------------------------------------
P1_MAX_SPREAD_TOL = 0.15   # max pairwise rel. spread of maxdev mults <= 15%
P2_SLOPE_RANGE = (-0.55, -0.45)  # LS slope of log(A_half_delta) vs log(m)
P3_KAPPA_REL_TOL = 0.25    # |kappa_hat - kappa_derived|/kappa_derived <= 25%
P3_KAPPA_SPREAD_TOL = 0.25  # spread of kappa_hat over m in {2000..10000}
G4_ANCHOR_TOL = 0.10       # 10% reproducibility band at m=2000 (bracketed stats)
G4_TREND_SLACK = 1.2       # trend anchor is an upper bound; allow 20% slack
G2_Q95_TOL = 0.05          # Algorithm R vs priority maxdev q95 within 5%
G3_SE_N = 3                # null self-consistency band: +-3 SE
G5_SE_N = 3

# M90 interpolation (plan PHASE 6.3): probit(power) linear in log10(m)
M90_PROBITS = (0.5, 0.9)

# Verdict rule (verbatim; see pre_registration.json):
#   For each family at alpha=0.05:
#     P1 PASS iff max pairwise relative spread of maxdev A_half_over_delta_floor
#        over m in {2000,5000,10000} <= 15%.
#     P2 PASS iff LS slope of log(A_half_delta_chi2) vs log(m) in
#        [-0.55, -0.45] (trend and energy slopes reported likewise).
#     P3: kappa_hat(m) = (delta_floor(m)/A_half_delta_chi2(m))^2.
#        CONFIRMED iff |kappa_hat(m) - kappa_derived|/kappa_derived <= 0.25 for
#        every m in {5000, 10000} AND max pairwise relative spread of kappa_hat
#        over {2000, 5000, 10000} <= 25%.
#        PARTIALLY_CONFIRMED iff P1 and P2 PASS but the multiplier shifts by a
#        roughly constant factor (factor = median over m of kappa_hat/kappa_derived).
#        Otherwise MEASURED_LAW (fitted exponent and intercept become the law).
#   Overall verdict = worst (lexicographic CONFIRMED > PARTIALLY_CONFIRMED >
#   MEASURED_LAW) across the two families.

VERDICT_ORDER = ("CONFIRMED", "PARTIALLY_CONFIRMED", "MEASURED_LAW")

# ---------------------------------------------------------------------------
# Phase keys / helper
# ---------------------------------------------------------------------------
def alpha_label(alpha: float) -> str:
    for lab, a in ALPHA_KEYS.items():
        if abs(a - alpha) < 1e-12:
            return lab
    return f"alpha_{alpha:g}"


def err_se(n: int) -> float:
    """Binomial standard error of a ~0.05 rejection rate over n reps."""
    return math.sqrt(0.05 * 0.95 / n)