"""Analytic benchmarks for the uniformity protocol (C1 and C3 checks).

All benchmarks are computed in-process so the simulated-vs-analytic deltas are
self-consistent:

* C1 -- independent-Binomial max  null quantile: for c_i iid Binom(m, p),
  P(max_i c_i <= x) = F(x)^n where F is the Binomial CDF.  Q(t) = min x such
  that F(x)^n >= t; the benchmark max-deviation quantile is Q(t) - mu.
  (Because {maxdev <= d} is a SUBSET of {max <= mu + d}, the simulated maxdev
  quantile is always <= this benchmark -- a deterministic direction check.)

* Textbook thresholds -- chi2.ppf(1 - alpha, n - 1), what practitioners apply
  to the per-position inclusion counts.

* C3 -- product-Binomial chi2 variance in closed form: for iid Binom(m, p)
  marginals, Var(chi2) = n * (mu4 - mu^2) / mu^2 with mu4 the fourth central
  moment of a Binomial(m, p) (closed form mu4 = m p q (1 + 3 p q (m - 2))).
  Cross-checked empirically with 5e4 cheap independent draws; if the closed
  form drifts > 0.5% from simulation the empirical value becomes the C3
  reference (fallback F3).

* Analytic anchors -- extreme-value floor A_floor = sqrt(2 p m log n) in
  count units and per-position amplitude delta_floor = A_floor / (p m), the
  scale at which maxdev starts to resolve a single-position bias.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binom, chi2

from .config import BENCH_ALPHAS, MAXDEV_TAUS


def binom_max_quantiles(n_pos: int, m: int, p: float, taus: tuple[float, ...] = MAXDEV_TAUS
                        ) -> dict[float, int]:
    """Q(t) = min x : binom.cdf(x; m, p)^n_pos >= t, for each t in taus.

    Vectorized over x = 0..m (O(m log m)); returns dict tau -> integer count
    quantile of the max-of-n_pos independent Binom(m, p) variables.
    """
    x = np.arange(m + 1, dtype=np.float64)
    cdf = binom.cdf(x, m, p)
    target = np.power(cdf, n_pos)  # monotone increasing in x, ends at 1
    out: dict[float, int] = {}
    for tau in taus:
        if tau >= 1.0:
            out[float(tau)] = m
            continue
        idx = int(np.searchsorted(target, tau, side="left"))
        out[float(tau)] = int(min(idx, m))
    return out


def maxdev_benchmark_quantiles(n_pos: int, m: int, k: int,
                               taus: tuple[float, ...] = MAXDEV_TAUS) -> dict[float, float]:
    """C1 benchmark: maxdev quantiles = Q(t) - mu under independence."""
    p = k / n_pos
    mu = m * p
    qs = binom_max_quantiles(n_pos, m, p, taus)
    return {float(t): float(q - mu) for t, q in qs.items()}


def textbook_chi2_thresholds(n_pos: int, alphas: tuple[float, ...] = BENCH_ALPHAS
                             ) -> dict[float, float]:
    """chi2.ppf(1 - alpha, df = n - 1) -- the textbook uniformity test."""
    return {float(a): float(chi2.ppf(1.0 - a, n_pos - 1)) for a in alphas}


def product_binomial_chi2_var(n_pos: int, m: int, k: int) -> float:
    """C3 closed-form chi2 variance under iid Binom(m, p) marginals."""
    p = k / n_pos
    q = 1.0 - p
    mu = m * p
    mu4 = m * p * q * (1.0 + 3.0 * p * q * (m - 2.0))  # 4th central moment
    return float(n_pos * (mu4 - mu * mu) / (mu * mu))


def empirical_independent_chi2_var(n_pos: int, m: int, k: int, n_samples: int = 50_000,
                                   rng: np.random.Generator | None = None) -> float:
    """Empirical Var(chi2) under iid Binom marginals (F3 cross-check source)."""
    if rng is None:
        rng = np.random.default_rng(12345)
    p = k / n_pos
    mu = m * p
    x = rng.binomial(m, p, size=(n_samples, n_pos)).astype(np.float64)
    chi2_vals = ((x - mu) ** 2).sum(axis=1) / mu
    return float(chi2_vals.var(ddof=1))


def c3_reference(n_pos: int, m: int, k: int, rng: np.random.Generator | None = None,
                 n_samples: int = 50_000) -> tuple[float, str]:
    """Closed form with the F3 fallback; returns (variance, source_label)."""
    closed = product_binomial_chi2_var(n_pos, m, k)
    emp = empirical_independent_chi2_var(n_pos, m, k, n_samples, rng)
    if abs(emp - closed) / max(closed, 1e-12) > 0.005:
        return emp, "empirical_independent_binomial (F3 fallback: closed form drifted)"
    return closed, "product_binomial_closed_form"


def extremes_floor(n_pos: int, m: int, k: int) -> dict[str, float]:
    """A_floor (count units) and delta_floor (per-position relative amplitude)."""
    p = k / n_pos
    a_floor = float(np.sqrt(2.0 * p * m * np.log(n_pos)))
    return {"A_floor": a_floor, "delta_floor": a_floor / (p * m)}