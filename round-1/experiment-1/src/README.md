# Reservoir-Sampling Uniformity Protocol

Implements and evaluates reservoir-sampling stream samplers: sample k items
uniformly from a stream of unknown length (n), verify uniformity empirically
over many trials, and report the max deviation from the expected frequency
`maxdev = max_i |c_i - mu|`, `mu = m*k/n`.

## What runs

- `uv run method.py sanity` — adversarial gates (duality, Algorithm R vs
  priority, null self-consistency, mechanism/spike-vs-blind-band) before scaling.
- `uv run method.py null --scale {mini,third,full}` — null calibration (a),
  C1 benchmark (maxdev vs F(x)^n), C3 variance + textbook false alarms.
- `uv run method.py power --scale {mini,third,full}` — detection power (M) of the
  four statistics over the five bias families, blind bands, gap-vs-n, spike inversion.
- `uv run method.py bugbattery --scale {mini,third,full}` — four faithful buggy
  implementations through the same harness (sanity pass for the blind battery).
- `uv run method.py report` — assemble method_out.json (schema-valid) + README.

## Evidence tables (results/)

| File | Contents | Consumed by next iteration |
|------|----------|---------------------------|
| results/null_calibration.json | per-cell empirical null quantiles, alpha
 thresholds, C1 benchmark + rel error, C3 variance ratio, textbook false-alarm
 rates | thresholds for everything else |
| results/power_surfaces.json | power vs (cell, family, amplitude, statistic,
 alpha), A_acc, blind-band widths, gap-vs-n, spike inversion | **m*(.)
 trials-to-detection law fit** |
| results/bug_battery.json | per-bug power per statistic + family-linkage
 correlations of bias profiles | **blind bug-battery scoring** |
| results/thresholds.json | null thresholds lookup (machine-readable) | all |
| results/sanity_gates.json | gate verdicts from `sanity` | audit |

## Direct answer (user request)

Flagship verification (see dataset `reservoir_uniformity_core_verification` in
method_out.json): the correct sampler's max deviation from the expected
frequency matches the analytic independent-Binomial band F(x)^n within
~3.45% at the flagship cell; the anti-reservoir duality gate
passes (D(k) == D(n-k)); Algorithm R and priority sampling agree.

## Seeds and determinism

Master seed SEED=0; every cell derives a child SeedSequence from its key,
so results are independent of worker count and scheduling order.

## Grid actually run

- null cells done: 35/35
  (skipped: 0)
- power cells done: 6/6
  (skipped: 0)
- corners: k=5 at n in {300, 1e3}, m in {500, 2e3} (k=n-5 read via duality).

## Runtime note

Per-cell wall times are logged in logs/run.log and recorded in the phase
tables; scale with the mini -> third -> full ladder (see F1 trims in the plan).
