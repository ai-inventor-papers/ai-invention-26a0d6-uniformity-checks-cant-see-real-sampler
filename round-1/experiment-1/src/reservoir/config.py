"""Grid definitions, seeds, and paths for the reservoir uniformity protocol.

All cells listed here are the FULL grid (plan PHASE 3/4/5).  The ``--scale``
flag of the CLI selects a prefix of each list (mini / third / full) so the
run can be staged per aii-long-running-tasks and trimmed via the F1 ladder.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"

SEED = 0  # master seed; every cell derives a deterministic child stream

# ---------------------------------------------------------------------------
# Sampling parameters
# ---------------------------------------------------------------------------
GAMMA_RECENCY = 2.0        # bug_recency_slot: victim j = int(k * U**gamma)
RAND_MAX = 32767           # bug_modulo_slot: classic broken C rand() RAND_MAX
CALIB_AMPS = (0.5, 1.0, 2.0, 4.0)      # a-values for delta(a) calibration
CALIB_REPS = 100                        # replicates per calibration point
COARSE_REPS = 300                       # coarse power pass replicates (F1 trim from 400)
COARSE_DELTAS = (0.5, 1.0, 2.0, 4.0, 8.0)  # multiples of delta_floor
COARSE_EXTEND = (16.0, 32.0, 64.0)      # adaptive extension steps (gap trend)
# smart amplitude points x delta_floor (densified at the low end: the
# empirical chi2 null threshold is n*(1-p) < n-1, so the blind band sits at
# ~0.2-0.35 x delta_floor for diffuse families -- below the original grid)
FINE_DELTAS = (0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)
# F1 trim: at n = 1e4 the fine pass uses a 5-point sweep (power SE ~2-2.5%).
FINE_DELTAS_BIG = (0.2, 0.5, 1.0, 2.0, 3.0)
POWER_ALPHAS = (0.05, 0.01)
MAX_COARSE_MULT = 32.0                  # stop extension at 32 x delta_floor (F1 trim)
DEDUPE_TOL = 0.05                       # merge fine amplitudes within 5%

MAXDEV_TAUS = (0.95, 0.99, 0.999)       # C1 benchmark quantile levels
BENCH_ALPHAS = (0.05, 0.01)             # textbook chi2 thresholds
QC_QUANTILES = (0.5, 0.95, 0.99, 0.999)  # empirical null quantiles to report

# Vision of windows for the subwindow-energy statistic (fractions of n)
ENERGY_WINDOW_FRACS = (0.1, 0.025)


@dataclass(frozen=True)
class NullCell:
    n: int
    k: int
    m: int
    n_reps: int
    role: str = "main"  # "main" | "corner" | "power_support"

    def key(self) -> str:
        return f"null_n{self.n}_k{self.k}_m{self.m}"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# FULL null grid (plan PHASE 3).  Default trims already applied:
#   * drop m = 5e3 at n >= 3e3
#   * at n = 1e4 keep m in {500, 1000, 2000} (m=1e3 added as power support for
#     the n=1e4 power cells, which use m=1e3)
#   * at n = 1e4 the p=0.9 cell is kept only at m=1e3 (power support)
# Corner cells simulate k=5 only; the k=n-5 corner is read off the
# anti-reservoir complement duality D(n-5) == D(5) (exact, see run_null).
# ---------------------------------------------------------------------------
_N_REPS_SMALL = 4000   # n <= 1e3
_N_REPS_LARGE = 2500   # n >= 3e3

def _build_main_cells() -> list[NullCell]:
    cells: list[NullCell] = []
    for n, m_list in ((300, (500, 2000, 5000)), (1000, (500, 2000, 5000)),
                      (3000, (500, 2000)), (10000, (500, 1000, 2000))):
        n_reps = _N_REPS_SMALL if n <= 1000 else _N_REPS_LARGE
        for p in (0.05, 0.5, 0.9):
            # at n = 1e4 the p = 0.9 cell exists only at m = 1e3 (power support)
            ms = (1000,) if (n == 10000 and p == 0.9) else m_list
            for m in ms:
                cells.append(NullCell(n=n, k=int(p * n), m=m, n_reps=n_reps))
    return cells


MAIN_CELLS: list[NullCell] = _build_main_cells()
for _name in ("n", "k", "p", "m"):  # ensure no leaked loop vars
    globals().pop(_name, None)

CORNER_CELLS: list[NullCell] = [
    NullCell(n=300, k=5, m=500, n_reps=_N_REPS_SMALL, role="corner"),
    NullCell(n=300, k=5, m=2000, n_reps=_N_REPS_SMALL, role="corner"),
    NullCell(n=1000, k=5, m=500, n_reps=_N_REPS_SMALL, role="corner"),
    NullCell(n=1000, k=5, m=2000, n_reps=_N_REPS_SMALL, role="corner"),
]

NULL_CELLS: list[NullCell] = MAIN_CELLS + CORNER_CELLS


# ---------------------------------------------------------------------------
# FULL power grid (plan PHASE 4, F1-trimmed): n in {300, 3e3, 1e4} x p in
# {0.05, 0.5}; p = 0.9 power cells dropped (the textbook-threshold misfire at
# p = 0.9 is already established by the null cells' false-alarm rows, and the
# argpartition at k ~ 0.9n is the most expensive config for the least new
# information).  m fixed: 2e3 for n <= 3e3, 1e3 for n = 1e4.  n_fine is
# size-adaptive: 1500 at n=300, 1000 at n=3000, 600 at n=1e4 (SE on power
# ~1.3-2.2%); n_coarse = 300 everywhere (F1 trim).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PowerCell:
    n: int
    p: float
    m: int
    n_fine: int
    n_coarse: int = COARSE_REPS

    def key(self) -> str:
        return f"power_n{self.n}_p{self.p:g}_m{self.m}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["k"] = int(self.p * self.n)
        return d


POWER_CELLS: list[PowerCell] = [
    PowerCell(n=300, p=0.05, m=2000, n_fine=1500),
    PowerCell(n=300, p=0.5, m=2000, n_fine=1500),
    PowerCell(n=3000, p=0.05, m=2000, n_fine=1000),
    PowerCell(n=3000, p=0.5, m=2000, n_fine=1000),
    PowerCell(n=10000, p=0.05, m=1000, n_fine=600),
    PowerCell(n=10000, p=0.5, m=1000, n_fine=600),
]

# ---------------------------------------------------------------------------
# Bias families: shape_i drives the priority-key perturbation
#   key_i = U_i / (1 + a * shape_i)   (larger shape -> smaller key ->
#   higher inclusion probability; monotone and faithful to implementations)
# ---------------------------------------------------------------------------
def family_shape(family: str, n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    if family == "linear_trend":
        return i / (n - 1)
    if family == "exp_recency":
        return np.exp(-3.0 * i / n)
    if family == "half_ramp":
        s = np.zeros(n)
        half = n // 2
        s[half:] = (i[half:] - half) / max(half, 1)
        return s
    if family == "spike":
        s = np.zeros(n)
        s[n // 2] = 1.0
        return s
    if family == "periodic":
        s = np.zeros(n)
        s[np.arange(n) % 7 == 3] = 1.0
        return s
    raise ValueError(f"unknown family {family}")


FAMILIES = ("linear_trend", "exp_recency", "half_ramp", "spike", "periodic")
GAP_FAMILIES = ("linear_trend", "exp_recency")  # gap-vs-n trend lines

# ---------------------------------------------------------------------------
# Bug battery (plan PHASE 5): three representative cells at p = 0.5, m = 2e3
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BugCell:
    n: int
    k: int
    m: int
    n_reps: int = 800

    def key(self) -> str:
        return f"bug_n{self.n}_k{self.k}_m{self.m}"

    def to_dict(self) -> dict:
        return asdict(self)


BUG_CELLS: list[BugCell] = [
    BugCell(n=300, k=150, m=2000, n_reps=800),
    BugCell(n=1000, k=500, m=2000, n_reps=800),
    BugCell(n=3000, k=1500, m=2000, n_reps=800),
]
BUG_VARIANTS = ("recency_slot", "modulo_slot", "float_threshold", "drop_oldest")

# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
def cell_ss(key: str, seed: int = SEED) -> np.random.SeedSequence:
    """Deterministic SeedSequence for a cell key (use as rng >> default_rng)."""
    h = hashlib.sha256(key.encode("utf-8")).digest()
    return np.random.SeedSequence([seed, int.from_bytes(h[:16], "big")])


def child_rng(key: str, seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(cell_ss(key, seed))


# ---------------------------------------------------------------------------
# Scale selectors (mini -> third -> full, per aii-long-running-tasks)
# ---------------------------------------------------------------------------
def _match_null(n: int, p: float, m: int) -> NullCell | None:
    k = int(p * n)
    for c in NULL_CELLS:
        if c.n == n and c.k == k and c.m == m:
            return c
    return None


def select_null(scale: str) -> list[NullCell]:
    if scale == "mini":
        want = [(300, 0.05, 500), (300, 0.5, 2000)]
        out = [c for (n, p, m) in want if (c := _match_null(n, p, m)) is not None]
        corner = _match_null(300, 5 / 300, 500)
        if corner is not None:
            out.append(corner)
        return out
    if scale == "third":
        want = [(n, p, m) for n in (300, 1000) for p in (0.05, 0.5) for m in (500, 2000)]
        out = [c for (n, p, m) in want if (c := _match_null(n, p, m)) is not None]
        for n in (300, 1000):
            for m in (500, 2000):
                c = _match_null(n, 5 / n, m)
                if c is not None:
                    out.append(c)
        return out
    if scale == "full":
        return NULL_CELLS
    raise ValueError(f"unknown scale {scale}")


def select_power(scale: str) -> list[PowerCell]:
    if scale == "mini":
        return [PowerCell(n=300, p=0.5, m=2000, n_fine=500, n_coarse=300)]
    if scale == "third":
        return [c for c in POWER_CELLS if c.n in (300, 3000) and c.p in (0.05, 0.5)]
    if scale == "full":
        return POWER_CELLS
    raise ValueError(f"unknown scale {scale}")


def select_bugs(scale: str) -> list[BugCell]:
    if scale == "mini":
        return [BugCell(n=300, k=150, m=2000, n_reps=300)]
    if scale == "third":
        return [BugCell(n=300, k=150, m=2000, n_reps=800)]
    if scale == "full":
        return BUG_CELLS
    raise ValueError(f"unknown scale {scale}")


def power_needs_null_cell(n: int, p: float, m: int) -> NullCell | None:
    """Return the null cell whose thresholds the (n, p, m) power cell uses."""
    k = int(p * n)
    for c in MAIN_CELLS:
        if c.n == n and c.k == k and c.m == m:
            return c
    return None