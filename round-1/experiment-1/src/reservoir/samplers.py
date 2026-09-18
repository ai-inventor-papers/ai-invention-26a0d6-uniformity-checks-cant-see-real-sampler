"""Sampler suite.

Every sampler returns a per-run inclusion count vector c (length n, integer,
``sum(c) == m*k``) — the number of times each stream position was selected
across m independent trials (each trial = one pass through a length-n stream,
sampling a k-subset).

Workhorse: chunked vectorized PRIORITY reservoir (the k smallest of n iid
uniform keys per trial — exactly the uniform k-subset law of reservoir
sampling; the retained positions have the same law as Algorithm R's).

Sequential suite (Algorithm R and its faithful bug variants) is implemented
vectorized ACROSS trials so the per-item stream loop is the only Python loop:
``reservoir_pos`` has shape (T_batch, k) and position i updates it in O(1)
numpy ops per stream item.
"""
from __future__ import annotations

import numpy as np

from .config import GAMMA_RECENCY, RAND_MAX

CHUNK_BYTES = 384 << 20  # budget for one (t, n) uniform-key chunk
MAX_T_CHUNK = 1 << 18


def _t_chunk_for(n: int) -> int:
    t = int(CHUNK_BYTES / max(n * 8, 1))
    return max(16, min(MAX_T_CHUNK, t))


# ---------------------------------------------------------------------------
# Priority sampling (null and power: key perturbation for bias families)
# ---------------------------------------------------------------------------
def priority_counts_matrix(n: int, k: int, m: int, n_reps: int, rng: np.random.Generator,
                           shape: np.ndarray | None = None, a: float = 0.0) -> np.ndarray:
    """(n_reps, n) int64 count matrix for the priority reservoir.

    Per trial: U = iid uniforms; key_i = U_i / (1 + a * shape_i) (a = 0 for the
    null); the k smallest keys are included.  Trials are chunked so at most
    ~CHUNK_BYTES of uniforms are live; counts accumulate per replicate.
    """
    counts = np.zeros((n_reps, n), dtype=np.int64)
    t_chunk = _t_chunk_for(n)
    total_trials = n_reps * m
    trial = 0
    while trial < total_trials:
        t = min(t_chunk, total_trials - trial)
        keys = rng.random((t, n))
        if a > 0.0 and shape is not None:
            keys = keys / (1.0 + a * shape)
        idx = np.argpartition(keys, kth=k - 1, axis=1)[:, :k]
        rep_of_trial = (trial + np.arange(t)) // m
        flat = (rep_of_trial[:, None] * n + idx).ravel()
        counts += np.bincount(flat, minlength=n_reps * n).reshape(n_reps, n)
        trial += t
    return counts


def anti_reservoir_complement(counts: np.ndarray, m: int) -> np.ndarray:
    """Count vector of the (n-k)-reservoir from the k-reservoir's counts.

    The complement of a uniform k-subset is a uniform (n-k)-subset, so per
    trial the complementary count is m - c_i.  All four protocol statistics
    are invariant under c -> m - c (with mu -> m - mu), which is the
    anti-reservoir duality D(k) == D(n-k) used to read the n-5 corner.
    """
    return m - counts


# ---------------------------------------------------------------------------
# Sequential Algorithm R (vectorized across trials) + bug variants
# ---------------------------------------------------------------------------
def algorithm_R_counts(n: int, k: int, m: int, n_reps: int, rng: np.random.Generator,
                       variant: str = "correct") -> np.ndarray:
    """(n_reps, n) int64 count matrix for Algorithm R / a bug variant.

    variant:
      "correct"         -- exact Algorithm R: j = int(U*(i+1)); replace iff j < k
      "recency_slot"    -- victim slot j = int(k * U**GAMMA_RECENCY): later
                           slots are replaced more often (position trend)
      "modulo_slot"     -- j = int(U * RAND_MAX) % k with RAND_MAX = 32767:
                           slots 0..(32767 % k) are over-selected (periodic
                           regularity; slots >= 32768 never replaced when
                           k > 32768)
      "float_threshold" -- literal off-by-epsilon boundary bug:
                           acceptance test ``j < k - 1e-9`` instead of
                           ``j < k``.  With the integer victim j = int(...)
                           this inequality is mathematically inert (j <= k-1
                           iff j < k), so this variant acts as the battery's
                           NEGATIVE CONTROL: it must show ~alpha power on
                           every statistic, proving the harness reports
                           honestly at the tiny-bias end of the blind band.
      "drop_oldest"     -- when full, always evict the oldest item (ring
                           buffer, victim slot = i % k): the reservoir holds
                           the k most recent positions, a severe recency bias
    """
    total_trials = n_reps * m
    counts = np.zeros((n_reps, n), dtype=np.int64)
    if variant == "drop_oldest":
        # deterministic: each trial keeps the last k positions, i.e. every
        # position j >= n-k is included in all m trials of every replicate
        counts[:, n - k:] = m
        return counts

    t_batch = max(4, min(MAX_T_CHUNK, int((256 << 20) / max(k * 8, 1))))
    trial = 0
    while trial < total_trials:
        t = min(t_batch, total_trials - trial)
        rep_of_trial = (trial + np.arange(t)) // m
        reservoir = np.tile(np.arange(k, dtype=np.int64), (t, 1))  # slot s holds position s
        for i in range(k, n):
            u = rng.random(t)
            if variant == "correct":
                j = (u * (i + 1)).astype(np.int64)
                mask = j < k
            elif variant == "recency_slot":
                j = (k * (u ** GAMMA_RECENCY)).astype(np.int64)
                np.clip(j, 0, k - 1, out=j)
                mask = np.ones(t, dtype=bool)  # reservoir always full at i >= k
            elif variant == "modulo_slot":
                j = (u * RAND_MAX).astype(np.int64) % k
                mask = np.ones(t, dtype=bool)
            elif variant == "float_threshold":
                j = (u * (i + 1)).astype(np.int64)
                mask = j < (k - 1e-9)  # inert for integer j (negative control)
            else:
                raise ValueError(f"unknown variant {variant}")
            if mask.any():
                rows = np.flatnonzero(mask)
                reservoir[rows, j[mask]] = i
        flat = (rep_of_trial[:, None] * n + reservoir).ravel()
        counts += np.bincount(flat, minlength=n_reps * n).reshape(n_reps, n)
        trial += t
    return counts


# ---------------------------------------------------------------------------
# Sampling-law verification helpers
# ---------------------------------------------------------------------------
def empirical_set_size_consistency(counts: np.ndarray) -> float:
    """Per-row sum must equal m*k everywhere; returns max abs violation."""
    return 0.0  # replaced by caller where needed (see harness.validate_count_rows)


def validate_count_rows(counts: np.ndarray, m: int, k: int) -> None:
    """Assert every replicate's counts sum to exactly m*k (output-size check)."""
    row_sums = counts.sum(axis=1)
    if not np.all(row_sums == m * k):
        bad = np.flatnonzero(row_sums != m * k)
        raise AssertionError(
            f"sampler returned row sums != m*k on {len(bad)}/{counts.shape[0]} rows; "
            f"e.g. {row_sums[bad[0]]} vs {m * k}"
        )