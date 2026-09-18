"""Reservoir-sampling uniformity protocol.

Implements and evaluates reservoir-sampling stream samplers: sample k items
uniformly from a stream of unknown length (n), verify uniformity empirically
over many trials, and report the max deviation from the expected frequency.
"""

__version__ = "0.1.0"