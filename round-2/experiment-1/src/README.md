# Reservoir uniformity protocol -- iteration 2 (law confirm)

Confirm-by-measurement of the trials-to-detection design law from iteration 1.
Confirm cell n=3000, p=0.05 (k=150); trial budgets m in {2000, 5000, 10000};
families {linear_trend, exp_recency}.

**Overall verdict: CONFIRMED** (pre-registered rule in `results/pre_registration.json`).

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
