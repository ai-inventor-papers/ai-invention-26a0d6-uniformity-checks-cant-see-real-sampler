"""Unit tests for the four protocol statistics and the sampler suite.

Hand-computed examples: c = [0, m, 0] with n = 3, k = 1 and other tiny
configurations including the plan's edge cases k = 1, k = n - 1, m = 1.
"""
from __future__ import annotations

import numpy as np
import pytest

from reservoir.samplers import (algorithm_R_counts, anti_reservoir_complement,
                                priority_counts_matrix, validate_count_rows)
from reservoir.stats import compute_stats, sliding_window_max
from reservoir.config import child_rng


def test_maxdev_hand():
    # n=3, k=1, m=3 -> mu = 1; c = [0, 3, 0]
    mu = 1.0
    s = compute_stats(np.array([[0.0, 3.0, 0.0]]), mu)
    assert s["maxdev"][0] == pytest.approx(2.0)
    assert s["chi2"][0] == pytest.approx(6.0)
    # slope: positions (0,1,2) centered (-1,0,1); cov with c-1 = +1 + 0 -1 = 0
    assert s["trend_slope"][0] == pytest.approx(0.0)
    # energy at W=2: dev2 = [1,4,1]; windows [1,4]=5 and [4,1]=5
    assert sliding_window_max(np.array([[1.0, 4.0, 1.0]]), 2)[0] == pytest.approx(5.0)


def test_edge_m1():
    # m = 1, k = 1, n = 3: mu = 1/3; c = [1, 0, 0]
    mu = 1.0 / 3.0
    s = compute_stats(np.array([[1.0, 0.0, 0.0]]), mu)
    assert s["maxdev"][0] == pytest.approx(2.0 / 3.0)
    assert s["chi2"][0] == pytest.approx(2.0)


def test_edge_kn1():
    # k = n - 1, m = 1, n = 3: mu = 2/3; c = [1, 1, 0]
    mu = 2.0 / 3.0
    s = compute_stats(np.array([[1.0, 1.0, 0.0]]), mu)
    assert s["maxdev"][0] == pytest.approx(2.0 / 3.0)
    assert s["chi2"][0] == pytest.approx(1.0)


def test_duality_algebra():
    # complement c -> m - c with mu -> m - mu: per-position deviations are
    # equal in magnitude; maxdev/energy invariant, trend flips sign, and chi2
    # scales by (m - mu)/mu (its denominator is mu)
    c = np.array([[6.0, 6.0, 6.0, 7.0, 5.0], [3.0, 2.0, 7.0, 1.0, 2.0]])
    m, k, n = 10, 3, 5
    mu = m * k / n
    s = compute_stats(c, mu)
    comp = anti_reservoir_complement(c.astype(np.int64), m).astype(np.float64)
    sc = compute_stats(comp, m - mu)
    assert np.allclose(s["maxdev"], sc["maxdev"], atol=1e-12)
    assert np.allclose(s["energy"], sc["energy"], atol=1e-12)
    assert np.allclose(s["trend_slope"], -sc["trend_slope"], atol=1e-12)
    assert np.allclose(s["chi2"] * mu, sc["chi2"] * (m - mu), atol=1e-12)


def test_sliding_window_energy_bruteforce():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(7, 50))
    got = sliding_window_max(x, 10)
    want = np.array([max(x[i, j:j + 10].sum() for j in range(50 - 10 + 1)) for i in range(7)])
    assert np.allclose(got, want)


def test_priority_row_sums_and_marginals():
    n, k, m, n_reps = 300, 15, 500, 2000
    counts = priority_counts_matrix(n, k, m, n_reps, child_rng("test_prio"))
    validate_count_rows(counts, m, k)
    p = k / n
    marg = counts.mean(axis=0) / m
    assert np.max(np.abs(marg - p)) < 0.01  # inclusion probability ~ k/n
    # mean per-column variance close to binomial marginal m p (1-p)
    # (mean over columns, not max: the max of 300 near-Gaussian estimates
    #  inflates the tail to ~3 sd by multiple comparisons)
    var = counts.var(axis=0, ddof=1)
    assert np.abs(var.mean() - m * p * (1 - p)) / (m * p * (1 - p)) < 0.05


def test_algorithmR_correct_is_uniform():
    n, k, m, n_reps = 60, 6, 400, 1500
    counts = algorithm_R_counts(n, k, m, n_reps, child_rng("test_algoR"), variant="correct")
    validate_count_rows(counts, m, k)
    p = k / n
    marg = counts.mean(axis=0) / m
    assert np.max(np.abs(marg - p)) < 0.025
    # single-trial law: reservoir must be a uniform k-subset -> column means k/n


def test_algorithmR_matches_priority_distribution():
    n, k, m, n_reps = 300, 15, 500, 2000
    c_r = algorithm_R_counts(n, k, m, n_reps, child_rng("test_r_vs_p_r"), variant="correct")
    c_p = priority_counts_matrix(n, k, m, n_reps, child_rng("test_r_vs_p_p"))
    mu = m * k / n
    qr = np.quantile(compute_stats(c_r, mu)["maxdev"], (0.5, 0.95, 0.99))
    qp = np.quantile(compute_stats(c_p, mu)["maxdev"], (0.5, 0.95, 0.99))
    assert np.allclose(qr, qp, rtol=0.02)


def test_bug_drop_oldest_keeps_last_k():
    n, k, m, n_reps = 100, 10, 50, 20
    counts = algorithm_R_counts(n, k, m, n_reps, child_rng("test_drop"), variant="drop_oldest")
    validate_count_rows(counts, m, k)
    # every trial keeps exactly the last k positions
    for i in np.arange(n):
        want = m if i >= n - k else 0
        assert np.all(counts[:, i] == want)


def test_bug_float_threshold_is_inert():
    # j < k - 1e-9 with integer j is mathematically identical to j < k
    n, k, m, n_reps = 200, 20, 300, 800
    c_bug = algorithm_R_counts(n, k, m, n_reps, child_rng("test_ft_bug"), variant="float_threshold")
    c_ok = algorithm_R_counts(n, k, m, n_reps, child_rng("test_ft_ok"), variant="correct")
    # distributions must match within Monte Carlo noise (the bug is inert)
    mu = m * k / n
    qb = np.quantile(compute_stats(c_bug, mu)["maxdev"], (0.5, 0.95))
    qo = np.quantile(compute_stats(c_ok, mu)["maxdev"], (0.5, 0.95))
    assert np.allclose(qb, qo, rtol=0.05)


def test_bug_recency_slot_biases_positions():
    n, k, m, n_reps = 200, 20, 300, 800
    counts = algorithm_R_counts(n, k, m, n_reps, child_rng("test_rs"), variant="recency_slot")
    validate_count_rows(counts, m, k)
    marg = counts.mean(axis=0) / m
    # the biased victim-slot selection must create a position-dependent bias:
    # |dev| from k/n must exceed ~5 null standard errors of a marginal estimate
    dev = marg - k / n
    null_sd = np.sqrt((k / n) * (1 - k / n) / (m * n_reps))
    assert np.max(np.abs(dev)) > 5 * null_sd


def test_bug_modulo_slot_faithful_victim_distribution():
    # the bug implements j = int(U * RAND_MAX) % k: slots 0..(32767 % k) get one
    # extra integer draw.  Verify the empirical victim-slot law analytic form.
    k = 20
    rng = np.random.default_rng(7)
    j = (rng.random(4_000_000) * 32768).astype(np.int64) % k
    probs = np.bincount(j, minlength=k) / j.size
    want = np.full(k, 1638 / 32768.0)
    want[: 32767 % k + 1] = 1639 / 32768.0
    # ~4e6 draws -> per-bin SE ~1.1e-4; allow 3 sd of the max of 20 bins
    assert np.allclose(probs, want, atol=3.5e-4)
    # and the sampler output is size-valid (output-size checks cannot reject it)
    n, m, n_reps = 200, 300, 100
    counts = algorithm_R_counts(n, k, m, n_reps, child_rng("test_ms"), variant="modulo_slot")
    validate_count_rows(counts, m, k)