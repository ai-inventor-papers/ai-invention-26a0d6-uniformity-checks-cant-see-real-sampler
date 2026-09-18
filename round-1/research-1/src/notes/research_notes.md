# RESEARCH NOTES (gen_art_research_1)

## Phase 1 — Cited-works verification (DONE)
1. **Vitter 1985**: "Random sampling with a reservoir", ACM TOMS 11(1):37-57, March 1985. DOI 10.1145/3147.3165. Author Jeffrey S. Vitter (Crossref). Abstract: Algorithm Z, O(n(1+log(N/n))) expected time, constant space, one pass. Matches hypothesis exactly. TODO: Waterman attribution; grep for deviation/chi/verify/test.
2. **Corrado**: Charles J. Corrado, "The exact distribution of the maximum, minimum and the range of Multinomial/Dirichlet and Multivariate Hypergeometric frequencies", Statistics and Computing 21(3):349-359, online 2010-03-02. DOI 10.1007/s11222-010-9174-3. Crossref issued 2010 vol 21 iss 3. -> RESOLUTION: 2010 correct (the "2011" variant is wrong). No arXiv version found.
3. **Extremes 2014**: Christian Houdré, Huy Huynh, Liang Peng, "On the asymptotic distribution of the multinomial maximum with an increasing number of classes", Extremes 18(2):179-190, DOI 10.1007/s10687-014-0208-7, ISSN 1386-1999/1572-915X. OpenAlex W2073599813.
4. **Zeyen et al. arXiv:2503.14079**: EXISTS. v1 18 Mar 2025, cs.LO. Authors: Zeyen, Cordy, Gubri, Perrouin, Acher. Five tests for SAT samplers (variable-frequency chi2 per variable, collision/repeated samples, monobit-style parity; HMP p-value combination); 7 samplers; Boolean-formula influence. NO reservoir sampling, NO max-deviation statistic, NO trials-to-detection law. **Published: ACM TOSEM 35:1-24, issued 2026-09-17, DOI 10.1145/3797477**. OpenAlex W7128822507 (year 2026).
5. **D'Agostino & Stephens 1986** "Goodness-of-Fit Techniques", Marcel Dekker 1986 (Routledge reprint DOI 10.1201/9780203753064). Confirmed.
6. **Rayner & Best 1989** "Smooth Tests of Goodness of Fit", OUP (1991 Technometrics review King, DOI 10.1080/00401706.1991.10484896; 2nd ed Wiley 2009, DOI 10.1002/9780470824443; WIREs 2011 survey, 10.1002/wics.171).
7. **Cressie & Read 1984**: JASA 79(386):336-343 "Goodness-of-fit statistics for discrete multivariate data" - direct record TODO; have Read & Cressie 1988 Springer book DOI 10.1007/978-1-4612-4578-0.

### Zeyen PDF key quotes
- "the approximation to the chi2 distribution in the GOF test is known to break down if any expected number of occurrences is below five"
- Five tests incl. variable-frequency (chi2), collision (two-sided, repeated samples), monobit-style parity; "testing a sampler on a single formula is not enough".

## Phase 2 — Novelty sweep (mostly done; few leftovers)
- S1 retried (poor queries give OpenAlex keyword soup). Useful anchors found elsewhere.
- S3: **"Algorithm AS 145: Exact Distribution of the Largest Multinomial Frequency"**, Applied Statistics 1979, DOI 10.2307/2347220; **"On the Distribution of the Maximum Frequency of a Multinomial Distribution"**, TPA 1973, DOI 10.1137/1117087; SSRN 2017 "Exact Algorithms for the Multinomial Extremes" (Corrado). All single-draw multinomial.
- S6: **Paninski 2008** "A Coincidence-Based Test for Uniformity Given Very Sparsely Sampled Discrete Data", IEEE Trans. IT, DOI 10.1109/tit.2008.928987; **"Adaptive Binning Coincidence Test for Uniformity Testing"** IEEE TSP 2024 (DOI 10.1109/tsp.2024.3397560).
- S11: **NIST SP 800-22** "A statistical test suite for random and pseudorandom number generators for cryptographic applications", DOI 10.6028/nist.sp.800-22 (2000), ~1882 cites; plus 2012 IEEE TIFS "On Statistical Tests for Randomness Included in the NIST SP800-22 Test Suite and Based on the Binomial Distribution" (DOI 10.1109/tifs.2012.2185227).
- S14/15 complement duality: NOT yet run. PBT sweep S9: NOT yet run. TestU01: TODO.

## Phase 3 — Protocol origin & practice (IN PROGRESS)
- **Wikipedia**: "A simpler algorithm, known as Algorithm R, was discovered independently by Alan G. Waterman[3][4] and McLeod & Bellhouse;[5]" — Waterman attribution confirmed.
- **Vitter PDF** (cs.umd.edu/~samir/498/vitter.pdf): "Algorithm R (which is a reservoir algorithm due to Alan Waterman) works as follows..." — Vitter himself attributes Algorithm R to Waterman. Only "deviation" hit = std-dev of skip variate in runtime analysis; NO chi-square, NO verification protocol, NO statistical tests in Vitter 1985.
- **StackOverflow 28993071** "Test Case for Weighted Reservoir Sampling" (via web.archive.org; live site 403): answer gives EXACT repeated-trials per-item protocol: "if you draw one element repeatedly 10^6 times, you should have 0.381*10^6 instances of the third element... look at the percentage of times the first element appeared out of 10^6 trials... must be approximately (weight_of_first_element/weight_of_all_elements)"; pseudocode: numTrials=1000000; histogram; assert(abs(real_probability - observed_probability) <= epsilon). Per-element (per-position) inclusion counts, repeated trials.
- **SteadBytes blog** (steadbytes.com/blog/reservoir-sampling, Oct 2020): verification = MLE-based per-VALUE distribution check on synthetic stream: "Generate input data with a known distribution... calculate the MLE of a parameter... Test the MLE is within some small tolerance ε of the real value"; pytest code n=1000 sample from population n*1000, Exp(rate). → third mode (per-value distribution-parameter check).
- **Richard Startin blog** (via archive): no empirical verification protocol; proof-oriented (skip distribution, "To get an unbiased sample, the mean and standard deviation need to be close to $N/n$" refers to skip variate S, proof of unbiasedness).
- **Behera blog** (balaramdb.com): DNS down; snippet says "Distributions are almost identical which verifies the uniformity" (eyeball histogram); fetch via archive TODO (low priority).
- **AnalyticsVidhya** "Big Data to Small Data - Reservoir Sampling": uses Kolmogorov-Smirnov test per feature (per-value) to check reservoir vs population distribution. TODO fetch.
- P6 per-value/replay probe: TODO. 3b library survey: TODO.

## Phase 4 — Chi-square dependence (IN PROGRESS)
- X3: "Discrete Multivariate Analysis" (Bishop-Fienberg-Holland 1975) surfaced; "Testing for positive association in contingency tables with fixed margins" CSDA 2003 (DOI 10.1016/j.csda.2003.10.019).
- X4: **"A survey of algorithms for exact distributions of test statistics in r×c contingency tables with fixed margins"**, CSDA 1985, DOI 10.1016/0167-9473(85)90080-5 (53 cites) — anchor.
- X5/X6: OpenAlex keyword soup; retry via Crossref direct: Joag-Dev & Proschan 1983; Agresti & Wackerly 1977; Patefield 1982 AS159. TODO.

## Phase 5 — Zeyen successors (mostly done except successors list)
- Published record: ACM TOSEM vol 35, article 1-24, issued 2026-09-17, DOI 10.1145/3797477 (title with comma: "Methods, Datasets, and Protocols").
- OpenAlex citing (W7128822507): only **DivKC: A Divide-and-Conquer Approach to Knowledge Compilation** (2026, LNCS, DOI 10.1007/978-3-032-22774-4_7) so far. S2 API was rate-limited (None). TODO: arXiv search "uniform random samplers" 2025-2026; authors' repo.

## Cressie-Read resolution
- The canonical power-divergence paper is **Cressie & Read (1984) "Multinomial Goodness-Of-Fit Tests", JRSS-B 46(3):440-464, DOI 10.1111/j.2517-6161.1984.tb01318.x** ("Cressie,Read" authors). Plus **Read & Cressie (1988) Springer book** DOI 10.1007/978-1-4612-4578-0. The JASA 1984 "Goodness-of-fit statistics..." doesn't exist as such — that title is the 1988 book. Also Read (1984) JASA 79:929-935 "Small-Sample Comparisons for the Power Divergence GOF Statistic".

## Corrado abstract (SSRN)
"The exact joint distribution of the maximum and minimum of a multinomial distribution of n balls in m urns is compactly represented as a product of stochastic matrices... The exact distribution of the multinomial range is also derived. An application to the September effect in stock returns is presented." — SINGLE-DRAW multinomial; no repeated-trial fixed-sum reservoir counts; no detection power.

## Phase 4 — RESOLVED
- **Joag-Dev & Proschan 1983** "Negative Association of Random Variables with Applications", Annals of Statistics, DOI 10.1214/aos/1176346079. Backdrop only; no chi-square null for reservoir vectors.
- **Agresti & Wackerly 1977** "Some Exact Conditional Tests of Independence for R x C Cross-Classification Tables", Psychometrika 42(1):111-125, DOI 10.1007/bf02293748.
- **Patefield**: "Algorithm AS 159: An Efficient Method of Generating Random R x C Tables with Given Row and Column Totals", Applied Statistics 30(1):91-97, **1981** (NOT 1982), DOI 10.2307/2346669.
- **CSDA 1985** survey of algorithms for exact distributions of fixed-margin r x c test statistics, DOI 10.1016/0167-9473(85)90080-5 (53 cites). + Bishop-Fienberg-Holland 1975 (DOI 10.2307/2344845).
- Wikipedia "Pearson's chi-squared test" anchors the classical multinomial chi2 statement (counts sum to N under Multinomial(N;p)).
- **Verdict C3 = PARTIALLY_HOLDS**: ancestors exist (Pearson df, fixed-margin tables, negative association); NO reference found for the m-trial reservoir count-vector chi2 null. Absence-of-evidence, queries X1-X7 + Crossref follow-ups.

## Phase 5 — RESOLVED
- arXiv search "uniform random samplers" (8 hits): newest = Growing Binary Trees (arXiv:2603.25972, 2026, math.CO, EPTCS 445) — combinatorial uniform sampler, not verification. Testify (arXiv:2208.12747) = sampler GENERATION for PBT. LOPSTR proceedings (2208.04235) unrelated.
- Citation record: OpenAlex cites(W7128822507) = 1 (DivKC 2026 LNCS, knowledge compilation, not a uniformity successor). Semantic Scholar 429 rate-limited. TOSEM record issued 2026-09-17 — one day before this run; near-zero citations expected.
- Related dataset: Heradio & Fernandez-Amoros 2021 "SAT Instances for Testing Random Samplers' Uniformity", Zenodo DOI 10.5281/zenodo.4514918 (NOT by Zeyen's group).

## Phase 3 — library survey (RESOLVED)
- **Spark**: SamplingUtils.scala implements reservoir sampling; JavaDoc guarantee = sample-size bound ("99.99% of the time"), no uniformity test.
- **commons-rng**: chi-square uniformity tests are for PRNG output ranges (ProvidersCommonParametricTest), no reservoir sampler.
- **Guava / commons-math**: no reservoir sampler found (absence-of-evidence).
- **Python/Rust**: reservoir packages exist (jxnl/python-reservoir, kartva/rs-reservoir-sampling, ioannapap) — no uniformity tests surfaced in this survey.
- Complement duality: only expected-cost discussion on CrossValidated 654572; no published D(k)=D(n-k) maxdev identity found.

## FINAL VERDICTS
- C1 = HOLDS (scoped to interior k; k=1 border covered by classical multinomial laws) [confidence medium]
- C2 = FAILS (per-value-on-replayed-streams unattested; canonical = per-position repeated-trials on synthetic, SO 28993071 verbatim protocol)
- C3 = PARTIALLY_HOLDS (ancestors exist; reservoir vector instance unaddressed)
- C4/Zeyen = CONFIRMED (exists, v1, TOSEM 35:1-24 2026-09-17, no reservoir/maxdev/trial-law)
- Deliverables written: research_out.json (42 sources), research_report.md, .sdk_openhands_agent_struct_out.json, make_deliverables.py.