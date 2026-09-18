# Evaluation artifact: reviewer-required reconciliation tables

Pure data-processing evaluation that **rebuilds the three reviewer-required tables from the
stored iteration-1 evidence** (no new simulation). Every number in `eval_out.json` traces to a
row carrying a `source_artifact` id and a cell key, satisfying the acceptance check that any
number reprinted in the paper traces to a row in `eval_out.json`.

## What this artifact is

| Deliverable | Contents | Feeds paper section |
|-------------|----------|---------------------|
| `eval_out.json` | exp_eval_sol_out-schema-valid: `metadata` (conventions, source artifacts, per-table notes, table-level summaries, 15 self-consistency checks) + `metadata.tables.table1_corrected` (30 rows) + `metadata.tables.c2_inflation_pinned` (pinned cells + 7-profile flagship table + all 189 checkpoints) + `metadata.tables.null_law_validation` (35 rows); `metrics_agg` (16 headline aggregates); `datasets` (263 examples, one per row, `output` = full row JSON) | the three sections below |
| `eval_sol_out.json` | identical copy of `eval_out.json` (both exp_eval_sol_out-schema-valid); kept for iteration-1 naming compatibility | pipeline schema validation |
| `logs/run.log` | full run log incl. every self-consistency check | audit |
| `eval.py` | the reconstruction script (aii-python conventions: uv, loguru, pathlib, `main()`) | reproduction |

Run with: `uv run eval.py` (needs the two iteration-1 dependency workspaces at their original paths).

## Conventions (quoted verbatim in eval_out.json metadata)

- **Delta scale (one convention, two scalings).** `A_floor = sqrt(2*p*m*log(n))` (count units);
  `delta_floor = A_floor/(p*m)` (fraction-of-`mu` units, `mu = m*k/n`). TARGET delta in floor units =
  `probe.mult` (`= delta_target/delta_floor`); ACHIEVED delta in floor units = `probe.delta/delta_floor`.
  Any delta can be written in floor units (mult) or in TARGET delta fraction-of-mu units (mult × delta_floor).
  Columns are never mixed.
- **IB benchmark (main-screen C1 convention):** `maxdev q(t) = Q(t) - mu` with
  `Q(t) = min{x : binom.cdf(x; m, k/n)^n >= t}` for `t in {0.95, 0.99, 0.999}` (count units).
  Reused DEP1's `reservoir.benchmarks.maxdev_benchmark_quantiles` / `extremes_floor` via sys.path
  (identical C1 code path; fallback local implementation verified against stored values).
- **Blind band (alpha = 0.05):** maximal consecutive mult interval over the union of `A_acc_probe_pass`
  and `fine_pass` probes where maxdev power < 0.5 AND max(chi2, trend_slope, energy) >= 0.95.

## Table A -> paper "Experiments: power-atlas" (Table 1)

`tables.table1_corrected` — all 30 (cell,family) rows with `in_blind_band` (re-derived) **and**
`stored_blind_band` flags, TARGET mult `mult_lo/mult_hi`, ACHIEVED `achieved_delta_lo_floor /
achieved_delta_hi_floor` (probe delta / delta_floor), midpoint powers incl. `max_acc =
max(chi2, trend_slope, energy)`, and `chi2_below_0.95` (accumulation carried by trend/energy alone).

**Reconciliation findings the paper must reflect**
1. **Blind-pair count 21/30 vs 23/30.** The stored `blind_bands` were computed over coarse+fine probes
   only (21 pairs). The plan's union-based re-derivation adds 2 previously-unreported linear_trend pairs
   (power_n3000_p0.5_m2000 band mult [0.05,0.1]; power_n10000_p0.5_m1000 band mult [0.05,0.05]) and
   extends 11 of the 21 stored bands further down (trend_slope power = 1.0 carries the accumulation at the
   A_acc probes). No stored band is lost. Prose must cite a definition: "21/30 (stored coarse+fine)" or
   "23/30 (union re-derivation)" — both are auditable per row.
2. **Flagship (reviewer MINOR #1).** At (power_n3000_p0.05_m2000, linear_trend): `delta_floor =
   0.400159157527358`; probe with maxdev power closest to 0.47 is the TARGET-mult-0.300 probe,
   ACHIEVED delta 0.1483 -> `flagship_achieved_over_floor = 0.3706` (≈37% of the floor), maxdev power
   0.467 vs accumulated 1.0/1.0/1.0. The cited "0.080-0.120" band is in **TARGET delta fraction-of-mu
   units = floor mult 0.200-0.300** — same convention, different scaling (*not* comparable to the
   ACHIEVED 0.371 without dividing by delta_floor).

Prose to align (replacements in `flagship_note` can be quoted):

  > OLD: "maxdev power is 0.47 at 37% of the floor while the table band is 0.080-0.120"
  > NEW: "at the probe with TARGET mult 0.300 (floor units), whose realized ACHIEVED delta was
  > 0.1483 = 0.371 x delta_floor (37% of the floor), maxdev power was 0.467 while
  > max(chi2, trend, energy) power was 1.0/1.0/1.0; the blind band spans TARGET mult 0.200-0.300
  > (= TARGET delta 0.080-0.120 in fraction-of-mu units) and its union re-derivation spans 0.050-0.300"

## Table B -> paper "Per-value failure mode" section (reviewer MINOR #3)

`tables.c2_inflation_pinned` — scans all 189 on-grid C2 null checkpoints (195 checkpoint files
parsed; 6 alias duplicates consolidated; 1 off-grid stray `uniform n30 k15 m50` excluded and
documented). Per checkpoint: `thresh_PV_naive_maxdev`, `thresh_PP_maxdev`, analytic
`ib_pp_q95 = maxdev_benchmark_quantiles(n,m,k)[0.95]`, `pp_relative_factor = PV_naive/PP`,
`ib_relative_factor = PV_naive/ib_pp_q95`, `degenerate_k_gt_V`.

**Pinned inflation (exact replacement for "~2,372x"):** the maximum is at
**profile = zipf_2.0, n = 5000, k = 4500, m = 5000** (V_realized = 101, maxfreq_share = 0.589,
entropy_deficit = 0.805):
`thresh_PV_naive_maxdev = 218,336.33` vs `thresh_PP_maxdev = 92.05` ->

- **pp_relative_factor = 2371.93** (this is the quoted "~2,372x": 0.003% off, within the 5% keep-quote margin — the pinned measurement *is* the attribution);
- **ib_relative_factor = 2481.09** (PV_naive vs the analytic per-position F^n q95; same cell under
  both definitions — `same_cell_under_both_definitions = true`);
- **c2_vs_ib_ratio = 1.003** at that cell (C2-simulated per-value-IB ratio; across all 189 cells
  max = 35.9, mean = 5.5).

The reviewer's 'is it vs_ib or PV/PP?' ambiguity therefore closes definitively: the 2,372x is a
**reference-mis-specification effect of the naive per-value reference (PV_naive/PP)**, not an
IB-deviation effect. The cell is degenerate (`k = 4500 > V_realized = 101`); at degenerate cells the
naive reference k/V exceeds the physical per-value inclusion bound and PP-relative vs IB-relative
factors diverge (they coincide in value here only by a margin). Remember also: correct *adversarial*
monotone-bias amplitude (TARGET mult) vs realized amplitude (ACHIEVED) columns are present in every
`flagship_rows` row of this table for the 7 profiles at the pinned (n, k, m).

Prose to align:

  > OLD: "the PV_naive/PP maxdev null threshold inflates up to ~2400x with skew"
  > NEW: "under the naive per-value reference k/V, the maxdev null threshold inflates by 2,372x
  > (PV_naive/PP reading; pinned at zipf_2.0, n=5000, k=4500, m=5000: threshold 218,336 vs 92.1);
  > the IB-relative reading is 2,481x and the C2-simulated vs_ib reading is 1.003 (max 35.9 over the
  > grid), i.e. the inflation is a reference-mis-specification of the naive per-value space, not an
  > IB-deviation effect"

## Table C -> paper "Null-law validation / novelty" section (reviewer MAJOR #2)

`tables.null_law_validation` — all 35 null cells (none skipped), each with benchmark q95/q99/q999,
sim-vs-bench rel_err at the three levels, exact anti-reservoir duality maxabs diff (`maxdev_maxabs_diff`),
`duality_ok`, chi2 null variance ratios, and textbook-chi2 false alarms. Summaries in
`metadata.table_summaries.null_law_validation`:

| Quantity | Measured | Paper claim it evidences |
|----------|----------|--------------------------|
| rel_err p <= 0.5 | 0.0-5.0% (mean 1.8%), sim <= bench always | benchmark valid for interior k, first null-law validation of the fixed-sum repeated-trial maxdev statistic |
| rel_err p = 0.9 | 7.5-20.0% (mean 12.4%); caveat = F(x)^n min-side approximation | explicit caveat that the benchmark fails on the min side at high p |
| duality failures | 0 (maxabs diff 0.0 on every cell) | anti-reservoir duality D(k)=D(n-k) exact |
| chi2 var ratio vs product-Binomial | 0.860-1.066 | textbook independence idealization bounds |
| chi2 var ratio vs chi2_(n-1) | 0.184-1.129 | chi2 df idealization over-/under-dispersed |
| textbook chi2_(n-1) FA at alpha 0.05 (main grid) | 0.0-1.25% (p>=0.5: exactly 0.0 on all 17 cells) | textbook test strongly over-conservative for the choice-based null |
| textbook FA corner k=5 | 3.05-4.45% | small-expected-count regime approximates chi2_(n-1) better |

Cross-checks: `results/thresholds.json` matches `null_calibration.json` on all 35 cells
(stat_quantiles/thresholds/c1/c3 blocks); the analytic benchmark reproduces stored c1 entries exactly
on 3 sampled cells (incl. n=10000, p=0.9).

Prose to align (measured evidence for the reframed novelty claims):

  > "the textbook chi2_{n-1} uniformity test is over-conservative for the reservoir null: empirical
  > false-alarm rate 0.0-1.3% at alpha=0.05 on the main grid (exactly 0.0% at p >= 0.5); only the
  > extreme corners k=5 rise to 3.0-4.5% where expected counts are small"

## How to audit

1. `uv run eval.py` reruns the whole reconstruction (seconds, CPU only) and re-emits both JSONs.
2. `metadata.self_consistency_checks` — 15/15 pass (30 table-A rows; 21 stored / 23 re-derived
   blind pairs documented; chi2_below_0.95 flags present where stored midpoint chi2 < 0.95; flagship
   probe achieved/floor 0.371 ± 0.01 and delta_floor 0.4002; 189 on-grid C2 checkpoints; max-inflation
   cell pinned under both ratio definitions, quote-checked; 35 null cells none skipped; every row has
   a cell key).
3. Schema: `eval_out.json` and `eval_sol_out.json` (and all full/mini/preview variants) validate against
   `exp_eval_sol_out` (aii-json skill), PASSED: metrics_agg present, datasets-grouped examples use only
   `input`/`output`/`metadata_*`/`predict_*`/`eval_*` fields; the canonical structured rows live in
   `metadata.tables`.
4. Units: TARGET vs ACHIEVED, mult vs delta-fraction-of-mu, and the three inflation readings are
   explicitly labeled in every row where they could be confused (`flagship_*` fields,
   `pp_relative_factor`/`ib_relative_factor`/`c2_vs_ib_ratio`).

Source artifacts: `art_UF60msYdlAIi` (main screen, table A + C) and `art_uCeKcHDqIJMh` (C2 per-value
screen, table B). No simulation was run; all numbers were read at stored precision from these artifacts.