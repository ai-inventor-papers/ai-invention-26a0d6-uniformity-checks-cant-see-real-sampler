#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build research_out.json, structured_verdicts.json and
.sdk_openhands_agent_struct_out.json for the reservoir-verification
max-deviation prior-art research artifact.

All bibliographic/numeric facts below were verified against the cited URL/DOI
during this research sweep; see research_report.md for the methodology appendix.
research_out.json deliberately uses the SAME schema shape as the .sdk output
(answer = prose string with numbered citations, sources with index/url/title/
summary), per verifier requirements; the plan's structured q1-q4 verdicts are
preserved in structured_verdicts.json and reproduced inside the prose.
"""
import json, os, re

WS = "/ai-inventor/aii_data/runs/run_yRBWOr6EQPIx/3_invention_loop/iter_1/gen_art/gen_art_research_1"

# ---------------------------------------------------------------------------
# Master source list (index, title, url, summary, authors, year, venue, doi,
# passages = list of (quote, locator)).
# ---------------------------------------------------------------------------
S = []

def add(idx, title, url, summary, authors=None, year=None, venue=None,
        doi=None, passages=None):
    S.append(dict(index=idx, title=title, url=url, summary=summary,
                  authors=authors, year=year, venue=venue, doi_or_arxiv=doi,
                  supporting_passages=passages or []))

add(1, "Random sampling with a reservoir",
    "https://doi.org/10.1145/3147.3165",
    "Canonical reservoir-sampling paper (Algorithm R/Z). Crossref/ACM record pins title, author (Jeffrey S. Vitter), venue ACM Transactions on Mathematical Software 11(1):37-57, March 1985, DOI 10.1145/3147.3165. Contains marginal-uniformity correctness proofs and running-time analysis; no empirical verification protocol.",
    ["Jeffrey S. Vitter"], 1985, "ACM Transactions on Mathematical Software",
    "10.1145/3147.3165")

add(2, "Random Sampling with a Reservoir (scanned PDF on UMD mirror)",
    "https://www.cs.umd.edu/~samir/498/vitter.pdf",
    "Scanned copy of Vitter 1985. Grep for 'deviation|chi|verify|test|Waterman' shows: (a) the only 'deviation' hits concern the standard deviation of the skip variate in the runtime analysis; (b) the only 'test' hits concern an internal algorithmic equivalence test in Algorithm Z's skip computation; (c) NO chi-square / NO repeated-trial uniformity-verification protocol / NO max-deviation statistic. Attributes Algorithm R to Alan Waterman.",
    ["Jeffrey S. Vitter"], 1985, "ACM TOMS (mirrored PDF)", None,
    [("Algorithm R (which is is a reservoir algorithm due to Alan Waterman) works as follows: When the (t + 1)st record in the file is being processed, for t L n, the n candidates form a random sample of the first t records.",
      "p. 38, Algorithm R definition")])

add(3, "Reservoir sampling (Wikipedia)",
    "https://en.wikipedia.org/wiki/Reservoir_sampling",
    "Confirms the Waterman / McLeod-Bellhouse attribution of Algorithm R and the Vitter 1985 citation with the UMD PDF link. Per its lead section, 'A simpler algorithm, known as Algorithm R, was discovered independently by Alan G. Waterman and McLeod & Bellhouse' (paraphrase-qualified: Wikipedia's footnote markers make exact machine quotation of the passage fragile, so no supporting_passage is attached). No verification-protocol content (no repeated-trial statistical testing described).",
    None, None, None, None)

add(4, "The exact distribution of the maximum, minimum and the range of Multinomial/Dirichlet and Multivariate Hypergeometric frequencies",
    "https://doi.org/10.1007/s11222-010-9174-3",
    "Resolves the 2010-vs-2011 discrepancy: Crossref and OpenAlex both date the paper 2010, Springer journal 'Statistics and Computing' 21(3):349-359, author Charles J. Corrado, online 2010-03-02, DOI 10.1007/s11222-010-9174-3. SINGLE-DRAW cell maxima/minima/range (one sampling operation), not the repeated-trial fixed-sum k-subset count vector; no detection-power content.",
    ["Charles J. Corrado"], 2010, "Statistics and Computing",
    "10.1007/s11222-010-9174-3")

add(5, "SSRN version of Corrado 2010 (abstract)",
    "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=989082",
    "SSRN preprint record (posted August 19, 2010). Abstract (fetched in this research; SSRN blocks automated re-fetching, so the abstract sentence is described here rather than machine-quoted) confirms the object is the single-draw multinomial: 'n balls in m urns', with the exact max/min/range distributions compactly represented as products of stochastic matrices, plus an application to the September effect in stock returns. No repeated-trials verification, no power analysis.",
    ["Charles J. Corrado"], 2010, "SSRN Electronic Journal",
    "10.2139/ssrn.989082")

add(6, "On the asymptotic distribution of the multinomial maximum with an increasing number of classes",
    "https://doi.org/10.1007/s10687-014-0208-7",
    "Resolves the 'Extremes (2014)' citation: authors Christian Houdre, Huy Huynh, Liang Peng; journal Extremes 18(2):179-190 (2014), DOI 10.1007/s10687-014-0208-7, ISSN 1386-1999/1572-915X. Extreme-value asymptotics for the single-draw multinomial maximum as the number of classes grows; no repeated-trial fixed-sum statistic, no verification framing.",
    ["Christian Houdre", "Huy Huynh", "Liang Peng"], 2014, "Extremes",
    "10.1007/s10687-014-0208-7")

add(7, "Algorithm AS 145: Exact Distribution of the Largest Multinomial Frequency",
    "https://doi.org/10.2307/2347220",
    "Nearest neighbor for the null law: exact distribution of the single-draw largest multinomial frequency (Applied Statistics, 1979). Differs from the reservoir object: one draw, n cells/N balls; the repeated-trial k-subset fixed-sum counts are a different coupling except at the k=1 border.",
    None, 1979, "Applied Statistics", "10.2307/2347220")

add(8, "On the Distribution of the Maximum Frequency of a Multinomial Distribution",
    "https://doi.org/10.1137/1117087",
    "Further single-draw multinomial-maximum prior art (Theory of Probability & Its Applications 17(4):712-717, 1973). Same 'why different' note as AS 145.",
    None, 1973, "Theory of Probability & Its Applications", "10.1137/1117087")

add(9, "Multinomial Goodness-Of-Fit Tests (Cressie & Read power-divergence family)",
    "https://doi.org/10.1111/j.2517-6161.1984.tb01318.x",
    "The canonical 1984 Cressie-Read reference is this JRSS-B paper (46(3):440-464); there is no 1984 JASA paper with the book title. Introduces the power-divergence family of GOF statistics for multinomial data; power comparisons of chi-square vs competitors against smooth (diffuse) alternatives are settled classical ground.",
    ["Noel Cressie", "Timothy R. C. Read"], 1984,
    "Journal of the Royal Statistical Society Series B", "10.1111/j.2517-6161.1984.tb01318.x")

add(10, "Goodness-of-Fit Statistics for Discrete Multivariate Data (book)",
    "https://doi.org/10.1007/978-1-4612-4578-0",
    "Read & Cressie 1988 Springer monograph: book-length treatment of the power-divergence family for discrete multivariate data. Covers classical multinomial GOF; does not treat repeated-trial fixed-sum inclusion-count vectors from reservoir runs.",
    ["Timothy R. C. Read", "Noel Cressie"], 1988, "Springer Series in Statistics",
    "10.1007/978-1-4612-4578-0")

add(11, "Goodness-of-Fit Techniques",
    "https://doi.org/10.1201/9780203753064",
    "D'Agostino & Stephens 1986 (Marcel Dekker; Routledge reprint 2017, DOI 10.1201/9780203753064): landmark GoF treatise (EDF statistics, chi-square-based methods, power studies). Contains classical multinomial chi-square material (fixed-sum df correction) but no max-deviation statistic for repeated-trial fixed-sum count vectors.",
    ["Ralph B. D'Agostino", "Michael A. Stephens"], 1986,
    "Marcel Dekker / Routledge", "10.1201/9780203753064")

add(12, "Smooth Tests of Goodness of Fit (2nd ed.)",
    "https://doi.org/10.1002/9780470824443",
    "Rayner, Thas & Best 2009 Wiley 2nd edition of Rayner & Best 1989 (OUP). The smooth-test framework decomposes Pearson's chi-square into asymptotically independent components (first-order/trend-like vs higher-order) - the direct ancestor of any position-trend style test; no repeated-trial reservoir count-vector statistic.",
    ["J. C. W. Rayner", "O. Thas", "D. J. Best"], 2009, "Wiley",
    "10.1002/9780470824443")

add(13, "Smooth tests of goodness of fit (WIREs survey)",
    "https://doi.org/10.1002/wics.171",
    "Rayner, Thas & Best 2011 survey (WIREs Computational Statistics 3(5):397-406). Abstract captures the component decomposition of smooth-test statistics used as the statistical ancestor of the position-trend test in the paper's related work.",
    ["J. C. W. Rayner", "O. Thas", "D. J. Best"], 2011,
    "WIREs Computational Statistics", "10.1002/wics.171")

add(14, "A Coincidence-Based Test for Uniformity Given Very Sparsely Sampled Discrete Data",
    "https://doi.org/10.1109/tit.2008.928987",
    "Paninski 2008 (IEEE Trans. Information Theory): the sqrt(n)-efficient collision-based uniformity test for sparse categorical data - the information-theoretic ancestor cited for any accumulation-statistic (collision/fingerprint) arm of the verification battery.",
    ["Liam Paninski"], 2008, "IEEE Transactions on Information Theory",
    "10.1109/tit.2008.928987")

add(15, "Adaptive Binning Coincidence Test for Uniformity Testing",
    "https://doi.org/10.1109/tsp.2024.3397560",
    "2024 IEEE Trans. Signal Processing successor of the coincidence-test line; shows the distribution-testing community is still active on sparse-support uniformity testing. No reservoir-sampling, no max-deviation-for-fixed-sum-counts content.",
    None, 2024, "IEEE Transactions on Signal Processing",
    "10.1109/tsp.2024.3397560")

add(16, "A statistical test suite for random and pseudorandom number generators for cryptographic applications (NIST SP 800-22)",
    "https://doi.org/10.6028/nist.sp.800-22",
    "NIST SP 800-22 (2000): the canonical PRNG test battery. Tests target a single bitstream, not repeated-trial inclusion-count vectors; the battery is the closest 'protocol power atlas' analog but the object of testing differs.",
    None, 2000, "NIST Special Publication 800-22", "10.6028/nist.sp.800-22")

add(17, "On Statistical Tests for Randomness Included in the NIST SP800-22 Test Suite and Based on the Binomial Distribution",
    "https://doi.org/10.1109/tifs.2012.2185227",
    "2012 IEEE TIFS analysis of power and distributional properties of binomial-based tests inside SP 800-22 - an example of per-test power analysis for bitstream tests (again, not for repeated-trial inclusion counts).",
    None, 2012, "IEEE Transactions on Information Forensics and Security",
    "10.1109/tifs.2012.2185227")

add(18, "TestU01: A C library for empirical testing of random number generators",
    "https://doi.org/10.1145/1268776.1268777",
    "L'Ecuyer & Simard 2007 (ACM TOMS 33(4), article 22): TestU01 CRNG batteries; per-test power behavior against defect classes is studied there. Object is again a bitstream, not reservoir inclusion counts.",
    ["Pierre L'Ecuyer", "Richard Simard"], 2007, "ACM Transactions on Mathematical Software",
    "10.1145/1268776.1268777")

add(19, "Benchmarking the power of empirical tests for random number generators (thesis)",
    "https://doi.org/10.5353/th_b4150846",
    "HKU thesis by Xu benchmarking the detection power of empirical RNG tests - direct evidence that PRNG-battery power benchmarking exists as a genre, supporting the paper's framing of a 'power atlas' while keeping the reservoir object distinct.",
    ["Xu (HKU thesis)"], None, "University of Hong Kong", "10.5353/th_b4150846")

add(20, "Testing Uniform Random Samplers: Methods, Datasets and Protocols (arXiv abs)",
    "https://arxiv.org/abs/2503.14079",
    "arXiv:2503.14079 EXISTS: v1, submitted 18 Mar 2025, cs.LO (Logic in Computer Science). Authors: Olivier Zeyen, Maxime Cordy, Martin Gubri, Gilles Perrouin, Mathieu Acher. Five statistical tests for SAT-solution samplers; no reservoir sampling, no max-deviation statistic, no per-position count-vector treatment in the abstract.",
    ["Olivier Zeyen", "Maxime Cordy", "Martin Gubri", "Gilles Perrouin", "Mathieu Acher"],
    2025, "arXiv (cs.LO)", "arXiv:2503.14079",
    [("We propose a framework that contains five statistical tests which are suited to test uniform random samplers. Moreover, we demonstrate their use by testing seven samplers.",
      "Abstract")])

add(21, "Testing Uniform Random Samplers: Methods, Datasets, and Protocols (published, ACM TOSEM)",
    "https://doi.org/10.1145/3797477",
    "Journal record: published in ACM Transactions on Software Engineering and Methodology, vol. 35, pp. 1-24, issued 2026-09-17, DOI 10.1145/3797477 (title with comma variant). Published version appeared the day before this research ran (2026-09-18), so citation/successor counts are still tiny.",
    ["Olivier Zeyen", "Maxime Cordy", "Martin Gubri", "Gilles Perrouin", "Mathieu Acher"],
    2026, "ACM Transactions on Software Engineering and Methodology", "10.1145/3797477")

add(22, "Zeyen et al. full text (arXiv PDF)",
    "https://arxiv.org/pdf/2503.14079",
    "Full-text grep for 'reservoir|maxim|deviation|sample size|trials|power': zero hits for reservoir sampling or max-deviation; tests are variable-frequency chi-square per Boolean variable, a collision/repeated-samples test, and a monobit-style parity test; discusses required sample size only qualitatively per test - no trials-to-detection design law.",
    ["Olivier Zeyen", "Maxime Cordy", "Martin Gubri", "Gilles Perrouin", "Mathieu Acher"],
    2025, "arXiv (cs.LO)", "arXiv:2503.14079")

add(23, "DivKC: A Divide-and-Conquer Approach to Knowledge Compilation",
    "https://doi.org/10.1007/978-3-032-22774-4_7",
    "The only OpenAlex-indexed citing work of Zeyen et al. (2026, LNCS 32774). It cites the uniformity-testing suite but is about knowledge compilation; not a uniformity-verification successor, no trial-count laws, no reservoir content.",
    None, 2026, "Lecture Notes in Computer Science", "10.1007/978-3-032-22774-4_7")

add(24, "Growing Binary Trees (arXiv:2603.25972)",
    "https://arxiv.org/abs/2603.25972",
    "Newest (2026, math.CO, EPTCS 445) arXiv hit for 'uniform random samplers': a combinatorial uniform random sampler for binary trees with prescribed profile - generation, not uniformity verification; shows the phrase is used by the combinatorial random-generation community.",
    ["Olivier Bodini", "Antoine Genitrini", "Khaydar Nurligareev"], 2026, "arXiv (math.CO)",
    "arXiv:2603.25972")

add(25, "Automatic Synthesis of Random Generators for Numerically Constrained Algebraic Recursive Types",
    "https://arxiv.org/abs/2208.12747",
    "Testify framework (2022/2023): synthesizes uniform random samplers for constrained recursive types (Boltzmann sampling) for property-based testing. Confirms 'uniformity' concerns in PBT concern the GENERATED distribution of test inputs, not statistically verifying a streaming reservoir sampler; no design law for verification power.",
    ["Ghiles Ziat", "Vincent Botbol", "Matthieu Dien", "Arnaud Gotlieb", "Martin Pepin", "Catherine Dubois"],
    2022, "arXiv (cs.PL)", "arXiv:2208.12747")

add(26, "Test Case for Weighted Reservoir Sampling (Stack Overflow)",
    "https://web.archive.org/web/20160724165920/http://stackoverflow.com/questions/28993071/test-case-for-weighted-reservoir-sampling",
    "CANONICAL PROTOCOL ORIGIN (Stack Overflow folklore; the live page returns HTTP 403 to automated fetchers, so the timestamped Wayback Machine capture of 2016-07-24 is the verifiable URL and was fetched for this research): the accepted answer by Mike Koltsov (answered Sep 2015) prescribes running about 10^6 trials, counting per-element inclusions into a histogram, comparing observed vs expected probabilities, and asserting that the absolute difference stays within epsilon - i.e., a threshold on the maximum per-element deviation. This is exactly the 'run m trials, count inclusions per position/element, threshold the max deviation' protocol the hypothesis assumes, found verbatim in teaching folklore.",
    ["Mike Koltsov"], 2015, "Stack Overflow (archived)", None,
    [("Therefore, you can look at the percentage of times the first element appeared out of 106 trials.",
      "Accepted answer, second paragraph (Wayback capture 2016-07-24)"),
     ("assert(abs(real_probability - observed_probability) <= epsilon) # measuring absolute difference, but you can switch to relative difference",
      "Accepted answer, pseudo-code block")])

add(27, "Reservoir Sampling (SteadBytes blog)",
    "https://steadbytes.com/blog/reservoir-sampling/",
    "Per-VALUE verification mode: generate a synthetic stream with a known parametric distribution, sample, compute the MLE of the parameter from the sampled VALUES, assert within tolerance (pytest; n=1000, population n*1000, Exp(rate)). Shows the per-value-style check exists in blog practice - but on synthetic streams, not on replayed streams.",
    None, 2020, "Blog (SteadBytes)", None)

add(28, "Big Data to Small Data - Welcome to the World of Reservoir Sampling (Analytics Vidhya)",
    "https://www.analyticsvidhya.com/blog/2020/11/big-data-to-small-data-welcome-to-the-world-of-reservoir-sampling/",
    "Per-value verification on a static dataset: compares the sampled sub-population against the original data per feature (KS test for features; chi-square framing for categories). Supports the per-value mode existing in practice, but on the source population, not replayed streams.",
    None, 2020, "Blog (Analytics Vidhya)", None)

add(29, "Reservoir Sampling: Uniform Sampling of Streaming Data (Balaram Behera blog)",
    "https://balaramdb.com/2020/06/reservoir-sampling/",
    "Eyeball-histogram per-value verification: 'Distribution of Samples by Reservoir Sampling of Streaming Data - Notice that the distributions are almost identical which verifies the uniformity of the Reservoir Sampling in practice.' Site was DNS-dead during this sweep; snippet from the search index (flag: verified only via snippet).",
    None, 2020, "Blog", None)

add(30, "Reservoir Sampling (Richard Startin blog)",
    "https://richardstartin.github.io/posts/reservoir-sampling/",
    "Theory/derivation-focused (reproduces Vitter's skip-variate combinatorics); the 'mean and standard deviation need to be close to N/n' statement refers to the skip distribution in the proof of unbiasedness, not to an empirical verification protocol. No repeated-trial count verification.",
    None, 2020, "Blog", None)

add(31, "Reservoir Sampling (Amir Ziai, Medium)",
    "https://medium.com/@amirziai/reservoir-sampling-53f83b9410e3",
    "Repeated-trials interpretation of correctness: 'if we do this sampling many many times, the output we get is consistent with a uniform sampling' (per-output interpretation; page returns 403 - snippet only). Supports repeated-trials folklore; weaker than the SO protocol statement.",
    None, 2023, "Medium", None)

add(32, "Negative Association of Random Variables with Applications",
    "https://doi.org/10.1214/aos/1176346079",
    "Joag-Dev & Proschan 1983 (Annals of Statistics 11(1):286-295): the classical negative-association framework; sampling-without-replacement indicators are the canonical negatively associated family - the statistical backdrop for the fixed-sum negative dependence claim of C3. No chi-square null statement for repeated-trial count vectors.",
    ["Kumar Joag-Dev", "Frank Proschan"], 1983, "The Annals of Statistics",
    "10.1214/aos/1176346079")

add(33, "Some Exact Conditional Tests of Independence for R x C Cross-Classification Tables",
    "https://doi.org/10.1007/bf02293748",
    "Agresti & Wackerly 1977 (Psychometrika 42(1):111-125): exact conditional tests for contingency tables with fixed margins - the closest published treatment of chi-square-type inference under fixed-sum dependence, but for tables, not for reservoir inclusion-count vectors.",
    ["Alan Agresti", "Dennis Wackerly"], 1977, "Psychometrika", "10.1007/bf02293748")

add(34, "Algorithm AS 159: An Efficient Method of Generating Random R x C Tables with Given Row and Column Totals",
    "https://doi.org/10.2307/2346669",
    "Patefield 1981 (Applied Statistics 30(1):91-97; Crossref dates it 1981, not 1982): Monte Carlo generation of tables with fixed margins - the machinery behind conditional chi-square p-values under fixed sums; distinct from reservoir count vectors.",
    ["W. M. Patefield"], 1981, "Applied Statistics", "10.2307/2346669")

add(35, "A survey of algorithms for exact distributions of test statistics in r x c contingency tables with fixed margins",
    "https://doi.org/10.1016/0167-9473(85)90080-5",
    "1985 CSDA survey (53 citations) of exact-distribution algorithms for fixed-margin contingency tables; supports that the fixed-sum dependence problem is well-studied FOR TABLES but no treatment for repeated-trial k-subset count vectors was found.",
    None, 1985, "Computational Statistics & Data Analysis", "10.1016/0167-9473(85)90080-5")

add(36, "Pearson's chi-squared test (Wikipedia)",
    "https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test",
    "Standard statement of the classical multinomial chi-square null: counts O_i with sum N under Multinomial(N;p), statistic sum (O_i - N p_i)^2/(N p_i); the degrees-of-freedom reduction comes from the fixed sum. This is the textbook ancestor of the 'single-draw fixed-sum correction' that C3 distinguishes from the m-trial coupling.",
    None, None, "Wikipedia", None)

add(37, "Apache Spark SamplingUtils.scala",
    "https://github.com/apache/spark/blob/master/core/src/main/scala/org/apache/spark/util/random/SamplingUtils.scala",
    "Real-world library practice: Spark implements reservoir sampling (used for RDD sampling), and quantifies a probabilistic bound on SAMPLE SIZE achieved ('99.99% of the time' in the JavaDoc) - a design guarantee, not a uniformity verification. No chi-square/maxdev uniformity test found in this survey.",
    None, None, "GitHub (apache/spark)", None)

add(38, "Apache Commons RNG ProvidersCommonParametricTest.java",
    "https://github.com/apache/commons-rng/blob/master/commons-rng-simple/src/test/java/org/apache/commons/rng/simple/ProvidersCommonParametricTest.java",
    "Commons RNG's chi-square uniformity tests target the PRNG output distribution (nextInt/nextLong ranges), not a reservoir sampler. No reservoir sampler/test found in commons-rng or commons-math in this survey; no uniformity test for a reservoir.",
    None, None, "GitHub (apache/commons-rng)", None)

add(39, "SAT Instances for Testing Random Samplers' Uniformity (dataset)",
    "https://doi.org/10.5281/zenodo.4514918",
    "Related SAT-sampler-testing resource: Zenodo dataset (2021) by Heradio & Fernandez-Amoros with SAT instances for assessing samplers' uniformity - evidence of an active SAT-sampler-uniformity testing ecosystem adjacent to Zeyen et al.; no reservoir content.",
    ["Ruben Heradio", "David Fernandez-Amoros"], 2021, "Zenodo",
    "10.5281/zenodo.4514918")

add(40, "Apache Spark SamplingUtils JavaDoc (spark.apache.org)",
    "https://spark.apache.org/docs/latest/api/java/org/apache/spark/util/random/SamplingUtils.html",
    "Documents the reservoir sampling implementation 'that also returns the input size' and the sampling-rate bound 'guarantees a sample of size greater than or equal to sampleSizeLowerBound 99.99% of the time'; corroborates that library practice quantifies sample-size guarantees rather than uniformity-testing.",
    None, None, "Apache Spark JavaDoc", None)

add(41, "Discrete Multivariate Analysis: Theory and Practice",
    "https://doi.org/10.2307/2344845",
    "Bishop, Fienberg & Holland 1975: the classical contingency-table/multivariate-categorical-data reference (log-linear models, chi-square GOF). An ancestor of C3's fixed-margin conditional-testing literature; no reservoir count vectors.",
    None, 1975, "MIT Press (reviewed in JRSS-A)", "10.2307/2344845")

add(42, "Reservoir sampling vs Sampling excluding replacements (Cross Validated)",
    "https://stats.stackexchange.com/questions/654572/reservoir-sampling-vs-sampling-excluding-replacements",
    "2024 Cross Validated thread comparing expected PRNG-call counts of reservoir sampling vs sampling-excluding-replacements: the 'complement/duality' discussion in community practice concerns expected cost, not a max-deviation identity D(k)=D(n-k). Confirms the complement direction is discussed but the maxdev identity is not.",
    None, 2024, "Cross Validated (StackExchange)", None)

IDX = {s["index"] for s in S}
assert IDX == set(range(1, len(S) + 1)), "indices must be 1..N"

# ---------------------------------------------------------------------------
# Comprehensive answer prose citing ALL sources (verifier-independent check).
# ---------------------------------------------------------------------------
answer_prose = (
    "Q1 NOVELTY OF THE MAX-DEVIATION NULL LAW, ITS POWER, AND DESIGN LAWS. "
    "The originating reservoir-sampling paper by Vitter [1] (ACM TOMS 11(1):37-57, 1985) contains correctness proofs "
    "for marginal uniformity and running-time analysis, and its scanned full text [2] shows no chi-square, no "
    "repeated-trial verification protocol, and no max-deviation statistic - while crediting Algorithm R to Alan "
    "Waterman, as Wikipedia [3] also does. After roughly fifteen distinct scholarly and general-web queries (OpenAlex, "
    "Crossref, ddgs; including 2023-2026 software-engineering and property-based-testing angles, e.g. samplers for "
    "constrained types [25]), no published derivation was found of the null law, the detection power, or a "
    "trials-to-detection design law for max_i|c_i - m*k/n| with c the fixed-sum vector of m iid k-subset inclusion "
    "counts. The nearest neighbors are all different objects and must be cited as such: single-draw multinomial "
    "max/min/range laws (Corrado 2010 [4], with the SSRN preprint abstract [5]; Houdre, Huynh & Peng 2014 in Extremes "
    "18(2):179-190 [6]; Algorithm AS 145 for the largest multinomial frequency [7]; the 1973 maximum-frequency law "
    "[8]) - which coincide with the reservoir object only at the k=1 border; the classical goodness-of-fit family "
    "(Cressie & Read's power-divergence paper [9] and its book treatment [10]; D'Agostino & Stephens' Goodness-of-Fit "
    "Techniques [11]; Rayner & Best's smooth tests [12] with the WIREs survey describing the asymptotically "
    "independent component decomposition [13]); sqrt(n)-efficient collision-based uniformity tests from distribution "
    "testing (Paninski 2008 [14]; 2024 adaptive-binning successor [15]); and PRNG battery power analyses (NIST SP "
    "800-22 [16]; binomial-NIST-test power analysis [17]; TestU01 [18]; an RNG-test power-benchmark thesis [19]). "
    "VERDICT C1 = HOLDS but must be scoped: no prior art was found for the interior-k (2 <= k <= n-2) fixed-sum "
    "instance, while the k=1 border is covered by the classical multinomial maxima literature [6], [7], [8]; confidence "
    "medium because this is absence-of-evidence, not proof of non-existence. "
    "Q2 PROTOCOL ORIGIN AND PRACTICE. The 'run m trials, count per-position or per-element inclusions, threshold the "
    "maximum deviation' protocol is verbatim Stack Overflow folklore: the top-voted answer to 'Test Case for Weighted "
    "Reservoir Sampling' [26] prescribes 10^6 trials, a per-element inclusion histogram, a comparison of observed vs "
    "expected probabilities, and assert(|observed - expected| <= epsilon). Repeated-trials phrasing also appears in "
    "blog teaching [31], and per-value distribution-parameter verification is described by SteadBytes (MLE tolerance "
    "test on sampled values) [27], Analytics Vidhya (chi-square/KS per feature against the source population) [28], "
    "and Behera (eyeball histograms) [29]; theory-only write-ups with no verification protocol include Startin [30], "
    "while library practice shows no uniformity tests either: Spark's SamplingUtils implements reservoir sampling but "
    "only guarantees a sample-size bound 99.99% of the time [37], [40], and Commons RNG's chi-square tests target PRNG "
    "output ranges, not a reservoir sampler [38]. Critically, ZERO sources were found practicing per-value "
    "verification on replayed streams. VERDICT C2 = FAILS: the premise that per-value-on-replayed-streams is the "
    "dominant practice is unsupported; per-position repeated-trial verification on synthetic streams is the observed "
    "canonical mode and matches the hypothesis's design. "
    "Q3 CHI-SQUARE DEPENDENCE FOR FIXED-SUM COUNTS. The classical ancestors all exist: Pearson's multinomial chi-square "
    "with the fixed-sum degrees-of-freedom reduction [36]; exact conditional tests for fixed-margin contingency tables "
    "(Agresti & Wackerly 1977 [33]; Patefield's AS 159 [34]; the 1985 CSDA survey [35]; Bishop, Fienberg & Holland's "
    "Discrete Multivariate Analysis [41]); and negative association of sampling-without-replacement indicators "
    "(Joag-Dev & Proschan 1983) [32]. No reference was found that gives the chi-square null (distribution, variance, "
    "or quantiles) for the m-trial reservoir inclusion-count vector itself. VERDICT C3 = PARTIALLY_HOLDS: the study's "
    "null-calibration work fills an unaddressed instance of a well-documented problem family and must cite these "
    "ancestors. "
    "Q4 ZEYEN ET AL. STATUS AND SUCCESSORS. arXiv:2503.14079 exists exactly as cited - v1, submitted 18 March 2025, "
    "cs.LO, by Zeyen, Cordy, Gubri, Perrouin and Acher [20] - and was formally published in ACM Transactions on "
    "Software Engineering and Methodology 35:1-24, issued 2026-09-17, DOI 10.1145/3797477 [21]. Full-text grep of the "
    "paper [22] shows five SAT-sampler-specific tests (variable-frequency chi-square per Boolean variable, a "
    "collision/repeated-samples test, a monobit-style parity test) with only qualitative sample-size remarks and NO "
    "reservoir sampling, NO max-deviation statistic, and NO trials-to-detection law. Successors are nascent: the sole "
    "OpenAlex citation is DivKC 2026 (a knowledge-compilation paper, not a uniformity-verification successor) [23]; "
    "arXiv's newest 'uniform random samplers' hits are combinatorial random generation (Growing Binary Trees [24]) "
    "rather than verification; the adjacent SAT-sampler-testing dataset by Heradio & Fernandez-Amoros [39] and the "
    "complement/duality discussion on Cross Validated [42] complete the landscape. VERDICT C4 = CONFIRMED; the "
    "related-work paragraph should cite both the arXiv [20] and TOSEM [21] versions and note the SAT-constrained-"
    "hyperspace vs reservoir-stream gap as open. "
    "OVERALL: with C1 scoped to interior k, C2 reframed (per-position synthetic is canonical; per-value checks are "
    "secondary; replayed-stream per-value verification is unattested and worth the study itself measuring), and C3 "
    "cited back to its classical ancestors, the paper's novelty and framing claims are calibrated and defensible."
)

# Every source index must be cited at least once; every [N] must resolve.
all_idx = {s["index"] for s in S}
cited = set(int(x) for x in re.findall(r"\[(\d+)\]", answer_prose))
missing = sorted(all_idx - cited)
unknown = sorted(cited - all_idx)
assert not missing, f"uncited sources: {missing}"
assert not unknown, f"citation to unknown index: {unknown}"

# ---------------------------------------------------------------------------
# Structured verdicts (plan's q1-q4 shape) preserved as a separate artifact.
# ---------------------------------------------------------------------------
structured_verdicts = {
    "q1_novelty": {
        "maxdev_null_law_published": False,
        "power_analysis_published": False,
        "m_star_design_law_published": False,
        "nearest_neighbors": [
            {"title": "The exact distribution of the maximum, minimum and the range of Multinomial/Dirichlet and Multivariate Hypergeometric frequencies (Corrado 2010)",
             "why_different": "Single-draw multinomial max/min/range; coincides with the reservoir object only at the k=1 border.",
             "url": "https://doi.org/10.1007/s11222-010-9174-3"},
            {"title": "Algorithm AS 145: Exact Distribution of the Largest Multinomial Frequency (1979)",
             "why_different": "Single-draw largest cell frequency; no repeated-trial fixed-sum vector, no verification framing.",
             "url": "https://doi.org/10.2307/2347220"},
            {"title": "On the asymptotic distribution of the multinomial maximum with an increasing number of classes (Houdre, Huynh, Peng 2014, Extremes)",
             "why_different": "Single-draw k=1-style asymptotics with growing number of classes; no repeated-trials statistic, no power analysis.",
             "url": "https://doi.org/10.1007/s10687-014-0208-7"},
            {"title": "A Coincidence-Based Test for Uniformity (Paninski 2008)",
             "why_different": "sqrt(n)-efficient uniformity test with sample-complexity bounds; the information-theoretic ancestor of any accumulation-statistic arm, but not the reservoir k-subset maxdev statistic.",
             "url": "https://doi.org/10.1109/tit.2008.928987"},
            {"title": "NIST SP 800-22 / TestU01 (2000-2007)",
             "why_different": "PRNG test batteries with per-test power behavior - the closest 'protocol power atlas' analog; object is a single bitstream, not repeated-trial inclusion counts.",
             "url": "https://doi.org/10.1145/1268776.1268777"}
        ],
        "premise_C1": "HOLDS",
        "confidence": "medium",
        "verdict": ("No published derivation of the null law / power / trials-to-detection law for "
                    "max_i |c_i - m*k/n| with c the fixed-sum vector of m iid k-subset indicator sums was found "
                    "across ~15 distinct scholarly+general queries (OpenAlex, Crossref, ddgs) including 2023-2026 "
                    "SE/PBT venues. Caveats: the k=1 border reduces to the classical single-draw multinomial whose "
                    "largest-frequency/range laws ARE published (AS 145, Corrado, Houdre et al.), so the claim must "
                    "be scoped to interior k (2 <= k <= n-2); this is absence-of-evidence, not proof of "
                    "non-existence.")
    },
    "q2_protocol_origin": {
        "canonical_statement_found": True,
        "canonical_sources": [
            {"quote": ("Therefore, you can look at the percentage of times the first element appeared out of 10^6 "
                       "trials. This must be approximately (weight_of_first_element/weight_of_all_elements). ... "
                       "numTrials = 1000000 ... histogram[element]++ ... "
                       "assert(abs(real_probability - observed_probability) <= epsilon)"),
             "source": "Stack Overflow, 'Test Case for Weighted Reservoir Sampling' (top-voted answer)",
             "url": "https://stackoverflow.com/questions/28993071/test-case-for-weighted-reservoir-sampling",
             "type": "stackoverflow"}
        ],
        "practice_table_summary": ("Per-position/per-element repeated-trial inclusion-count verification "
                                   "(numTrials ~1e6, histogram, epsilon compare) in N=2 explicit sources (SO answer; "
                                   "Ziai Medium snippet) plus 2 theory-only lecture-note series with no verification "
                                   "content (Utah MCMD S7.2, Rice COMP441); per-value distribution-mode verification "
                                   "in N=3 blog sources (SteadBytes MLE/pytest; Analytics Vidhya chi-square/KS per "
                                   "feature; Behera eyeball histograms) - all on synthetic or static source data; "
                                   "ZERO sources for per-value verification on replayed streams."),
        "premise_C2": "FAILS",
        "verdict": ("The premise that real-world practice is 'per-value verification on replayed streams' is not "
                    "supported: every concrete protocol found counts per-element/per-position inclusions over many "
                    "trials on synthetic or static streams (the SO answer is verbatim the hypothesis's protocol), "
                    "per-value checks appear only as distribution comparisons against known source populations, and "
                    "no source of any kind replays one fixed stream across seeds for per-value statistics. The paper "
                    "should frame per-position-on-synthetic-streams as the canonical mode, per-value checks as a "
                    "secondary blog-level mode, and replayed-stream per-value verification as unattested (candidate "
                    "contribution of the study's own screening arm).")
    },
    "q3_chisq_dependence": {
        "fixed_sum_chisq_reference_found": False,
        "nearest_ancestors": [
            {"title": "Pearson's chi-squared test (multinomial null; fixed-sum df reduction)",
             "what_it_covers": "Classical single-draw multinomial correction; applies to Var(c_i)=m p (1-p) margins, not the m-trial k-subset coupling.",
             "url": "https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test"},
            {"title": "Some Exact Conditional Tests of Independence for R x C Cross-Classification Tables (Agresti & Wackerly 1977)",
             "what_it_covers": "Exact conditional tests for tables with fixed margins - closest published treatment of inference under fixed-sum dependence, but for contingency tables, not reservoir count vectors.",
             "url": "https://doi.org/10.1007/bf02293748"},
            {"title": "Algorithm AS 159 (Patefield 1981) / CSDA 1985 survey of fixed-margin exact tests",
             "what_it_covers": "Monte-Carlo/exact machinery for chi-square under fixed row/column sums; table setting only.",
             "url": "https://doi.org/10.2307/2346669"},
            {"title": "Negative Association of Random Variables with Applications (Joag-Dev & Proschan 1983)",
             "what_it_covers": "Sampling-without-replacement indicators are negatively associated - the mathematical backdrop for the fixed-sum negative dependence; no follow-up located quantifies the chi-square null for repeated-trial k-subset count vectors.",
             "url": "https://doi.org/10.1214/aos/1176346079"},
            {"title": "Smooth Tests of Goodness of Fit (Rayner, Thas, Best 2009/2011)",
             "what_it_covers": "Decomposes Pearson chi-square into asymptotically independent components (ancestor of position-trend tests); no fixed-sum inclusion-count correction.",
             "url": "https://doi.org/10.1002/wics.171"}
        ],
        "premise_C3": "PARTIALLY_HOLDS",
        "verdict": ("The classical/conditional/negative-association ancestors all exist and are well documented "
                    "(Pearson df correction; Agresti-Wackerly and Patefield for fixed-margin tables; "
                    "Joag-Dev-Proschan negative association), but NO reference was found that gives the chi-square "
                    "null/reference (distribution, variance, or quantiles) for the m-trial reservoir inclusion-count "
                    "vector - the exact object of alternate C3. The study's null-calibration work therefore quantifies "
                    "a well-motivated but apparently unaddressed instance; it must cite the ancestors and avoid "
                    "claiming the negative-association concept itself as novel.")
    },
    "q4_zeyen_status": {
        "arxiv_id": "2503.14079",
        "exists": True,
        "authors": ["Olivier Zeyen", "Maxime Cordy", "Martin Gubri", "Gilles Perrouin", "Mathieu Acher"],
        "versions": ("v1 only (submitted 18 Mar 2025); no revision; published as ACM TOSEM vol. 35, pp. 1-24, "
                     "issued 2026-09-17, DOI 10.1145/3797477"),
        "covers_reservoir_or_maxdev": False,
        "successors": [
            {"title": "DivKC: A Divide-and-Conquer Approach to Knowledge Compilation",
             "year": 2026, "url": "https://doi.org/10.1007/978-3-032-22774-4_7",
             "note": "Only OpenAlex-indexed citation so far; cites the suite but is not a uniformity-testing successor (no trial laws, no reservoir)."},
            {"title": "Growing Binary Trees (arXiv:2603.25972)",
             "year": 2026, "url": "https://arxiv.org/abs/2603.25972",
             "note": "Combinatorial uniform random sampler (binary trees); generation, not verification - top new hit for the phrase 'uniform random samplers'."},
            {"title": "Adaptive Binning Coincidence Test for Uniformity Testing (IEEE TSP 2024)",
             "year": 2024, "url": "https://doi.org/10.1109/tsp.2024.3397560",
             "note": "Parallel line in distribution testing; not SAT-sampler-specific."}
        ],
        "verdict": ("arXiv:2503.14079 exists exactly as cited (v1, 18 Mar 2025, cs.LO; Zeyen, Cordy, Gubri, "
                    "Perrouin, Acher) and was formally published in ACM TOSEM (35:1-24) on 2026-09-17, one day "
                    "before this research ran. Its five tests are SAT-sampler-specific (variable-frequency chi-square "
                    "per variable, collision, monobit-style parity); full-text grep found no reservoir sampling, no "
                    "max-deviation statistic, and no trials-to-detection design law (only qualitative per-test "
                    "sample-size remarks). Successor work is still nascent (1 citation); no 2025-2026 "
                    "uniformity-verification successor touching reservoirs was found.")
    }
}

# ---------------------------------------------------------------------------
# Single schema-shaped payload written to BOTH research_out.json and the .sdk
# output (identical findings per "Keep the findings identical" requirement).
# ---------------------------------------------------------------------------
payload = {
    "title": "Prior-art check for reservoir max-deviation verification",
    "layman_summary": (
        "Web research on who published the statistics behind the max-deviation report used to verify reservoir "
        "samplers, where that protocol comes from, chi-square dependence work, and a 2025 SAT-sampler test suite."
    ),
    "summary": (
        "Research gate for a reservoir-verification power study. It resolves four factual questions with verdicts: "
        "(a) C1 HOLDS (scoped) - no prior art found for the null law / power / trials-to-detection law of the "
        "fixed-sum repeated-trial max-deviation statistic; nearest neighbors (single-draw multinomial maxima: "
        "Corrado 2010, AS 145 1979, Extremes 2014; distribution-testing collision tests: Paninski 2008; PRNG "
        "batteries: NIST SP 800-22, TestU01) are all different objects, with the k=1 border covered by published "
        "multinomial laws, so the novelty claim must target interior k (2 <= k <= n-2); (b) C2 FAILS - the protocol "
        "is verbatim Stack Overflow folklore (10^6 trials, per-element inclusion histogram, epsilon assert) and "
        "observed practice is per-position on synthetic streams; per-value checks exist in blogs (MLE/KS/eyeball) "
        "but zero sources use per-value verification on replayed streams; (c) C3 PARTIALLY_HOLDS - classical "
        "ancestors (Pearson df correction, Agresti-Wackerly/Patefield fixed-margin tests, Joag-Dev-Proschan negative "
        "association) are well documented but no reference treats the reservoir count-vector chi-square null; "
        "(d) Zeyen et al. arXiv:2503.14079 exists as cited and was published in ACM TOSEM 35:1-24 on 2026-09-17 "
        "(DOI 10.1145/3797477); it covers SAT samplers only (five tests, no reservoir, no maxdev, no trial-count "
        "law); successors are nascent (1 citation). Citation-integrity resolutions: Corrado = 2010 (not 2011); "
        "Extremes 2014 = Houdre-Huynh-Peng; Cressie-Read 1984 = JRSS-B 46(3):440-464; Patefield = 1981; Vitter's "
        "own paper attributes Algorithm R to Alan Waterman. Deliverables: research_out.json (42 sources + "
        "structured_verdicts.json), research_report.md with related-work skeleton, risks, and follow-up questions "
        "(collision test as fifth statistic; m* law vs property-testing bounds; per-value replay arm)."
    ),
    "answer": answer_prose,
    "sources": [
        {
            "index": s["index"],
            "url": s["url"],
            "title": s["title"],
            "summary": s["summary"],
            "authors": s["authors"],
            "year": s["year"],
            "supporting_passages": [
                {"quote": q, "locator": loc} for (q, loc) in s["supporting_passages"]
            ],
        }
        for s in S
    ],
    "follow_up_questions": [
        ("Should the collision/fingerprint test from the distribution-testing literature (Paninski 2008; Batu et al.; "
         "Adaptive Binning Coincidence Test 2024) be added as a fifth statistic, given it is the sqrt(n)-efficient "
         "accumulation test against which the chi-square-based maxdev may lose power at sparse k?"),
        ("Should the study's trials-to-detection law m*(.) be compared against the property-testing sample-complexity "
         "bounds found in this sweep, so the paper claims the reservoir/fixed-sum instance as new while crediting the "
         "information-theoretic ancestors?"),
        ("Does the paper need the Waterman attribution of Algorithm R (Vitter 1985 p. 38 and Wikipedia) plus the "
         "verbatim Stack Overflow protocol quote as intro anchors for the claim that the protocol is folklore rather "
         "than textbook-canonical?"),
        ("Should the screening experiment add a per-value replay arm (same stream, many seeds, per-value statistics) "
         "to bound the C2 generality claim, given that no source practicing per-value-on-replayed-streams "
         "verification was found today?")
    ],
    "out_expected_files": {"output": "research_out.json"},
    "upload_ignore_regexes": [r"(^|/)\.aii_cost_ledger\.jsonl$"],
}

for fname in ("research_out.json", ".sdk_openhands_agent_struct_out.json"):
    with open(os.path.join(WS, fname), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

with open(os.path.join(WS, "structured_verdicts.json"), "w", encoding="utf-8") as f:
    json.dump(structured_verdicts, f, indent=2, ensure_ascii=False)

print("WROTE research_out.json, .sdk_openhands_agent_struct_out.json, structured_verdicts.json")
print("sources:", len(S), "| answer chars:", len(answer_prose), "| cited:", len(cited), "of", len(S))