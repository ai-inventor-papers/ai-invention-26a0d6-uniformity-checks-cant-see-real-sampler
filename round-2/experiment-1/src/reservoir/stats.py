"""The four protocol statistics and their null-standardization helpers.

Given the per-run inclusion count vector c (length n, ``sum(c) == m*k``) with
expected per-position frequency ``mu = m*k/n``:

* ``maxdev(c)``      -- the mandated protocol report: max |c_i - mu|
* ``chi2(c)``        -- Pearson-style chi-square sum (c_i - mu)^2 / mu
* ``trend_slope(c)`` -- OLS slope of c_i on stream position i (raw units;
                        standardized to z by null mean/sd in the harness)
* ``subwin_energy``  -- max sliding-window sum of squared deviations at two
                        window scales (W = ceil(n/10) and W = ceil(n/40))

All functions are vectorized over replicate rows: input ``counts`` has shape
``(N_reps, n)``.  Bad cases (k = 1, k = n - 1, m = 1) are unit-tested in
tests/test_stats.py.
"""
from __future__ import annotations

import numpy as np

STAT_NAMES = ("maxdev", "chi2", "trend_slope", "energy")


def sliding_window_max(x2: np.ndarray, w: int) -> np.ndarray:
    """Max over sliding windows of length ``w`` of the row sums of ``x2``.

    ``x2`` has shape (N, n); returns (N,) maxima over all w-length windows
    (O(n) per row via np.cumsum).
    """
    n = x2.shape[1]
    if w >= n:
        return x2.sum(axis=1)
    cum = np.cumsum(x2, axis=1)
    # padded prefix sums: cum_pad[j] = sum of x2[:, :j]; window start j spans
    # positions j..j+w-1 -> cum_pad[j + w] - cum_pad[j], j = 0..n-w
    cum_pad = np.concatenate([np.zeros((x2.shape[0], 1), dtype=x2.dtype), cum], axis=1)
    window_sums = cum_pad[:, w:] - cum_pad[:, :-w]
    return window_sums.max(axis=1)


def compute_stats(counts: np.ndarray, mu: float, positions: np.ndarray | None = None,
                  window_fracs: tuple[float, ...] = (0.1, 0.025)) -> dict[str, np.ndarray]:
    """Compute all four statistics for a (N_reps, n) count matrix.

    Returns a dict mapping statistic name -> (N_reps,) float array.  The
    trend slope is returned in RAW units; callers standardize it with the
    null sample mean/sd (see harness.standardize_slopes).
    """
    counts = np.asarray(counts, dtype=np.float64)
    dev = counts - mu
    dev2 = dev * dev

    maxdev = np.max(np.abs(dev), axis=1)
    chi2 = dev2.sum(axis=1) / mu

    n = counts.shape[1]
    if positions is None:
        positions = np.arange(n, dtype=np.float64)
    else:
        positions = np.asarray(positions, dtype=np.float64)
    pos_cent = positions - positions.mean()
    denom = float(np.sum(pos_cent * pos_cent))
    slope = (dev @ pos_cent) / denom if denom > 0 else np.zeros(counts.shape[0])

    energy = sliding_window_max(dev2, max(2, int(np.ceil(n * window_fracs[0]))))
    if len(window_fracs) > 1:
        w2 = max(2, int(np.ceil(n * window_fracs[1])))
        if w2 != max(2, int(np.ceil(n * window_fracs[0]))):
            energy = np.maximum(energy, sliding_window_max(dev2, w2))

    return {"maxdev": maxdev, "chi2": chi2, "trend_slope": slope, "energy": energy}


def standardize_slopes(slopes: np.ndarray, mu_slope: float, sd_slope: float) -> np.ndarray:
    """z = (slope - null_mean) / null_sd; power tests use |z|."""
    if sd_slope <= 0:
        return np.zeros_like(slopes, dtype=np.float64)
    return (slopes - mu_slope) / sd_slope


def rejection_flags(stats: dict[str, np.ndarray], thresholds: dict[str, tuple[float, float]],
                    trend_key: str = "trend_slope") -> dict[str, np.ndarray]:
    """Boolean (N,) rejection flags per statistic under null thresholds.

    ``thresholds[stat]`` = (lo, hi).  For one-sided statistics (maxdev, chi2,
    energy) lo = 0 and rejection is ``stat > hi``; for the two-sided trend
    statistic rejection is ``abs(z) > hi`` (lo unused).
    """
    flags: dict[str, np.ndarray] = {}
    for name, arr in stats.items():
        lo, hi = thresholds[name]
        if name == trend_key:
            flags[name] = np.abs(arr) > hi
        else:
            flags[name] = arr > hi
    return flags


def power_from_flags(flags: dict[str, np.ndarray]) -> dict[str, float]:
    """Fraction of replicates rejected, per statistic (1 x len(stats) dict)."""
    return {name: float(fl.mean()) for name, fl in flags.items()}


def empirical_quantiles(samples: np.ndarray, qs: tuple[float, ...] = (0.5, 0.95, 0.99, 0.999)
                        ) -> dict[float, float]:
    """Empirical quantiles of a 1-D sample; keys are the q values."""
    qs_arr = np.asarray(qs, dtype=np.float64)
    vals = np.quantile(np.asarray(samples, dtype=np.float64), qs_arr)
    return {float(q): float(v) for q, v in zip(qs_arr, vals)}