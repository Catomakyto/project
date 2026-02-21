from __future__ import annotations

import math
from typing import Iterable, List, Tuple

import numpy as np


def empirical_bernstein_lcb(samples: Iterable[float], delta: float = 0.05) -> Tuple[float, float, float]:
    """Empirical Bernstein lower confidence bound.

    Returns (mean, lcb, radius).
    """
    vals = np.asarray(list(samples), dtype=float)
    if vals.size == 0:
        return 0.0, float("-inf"), float("inf")

    mean = float(np.mean(vals))
    if vals.size == 1:
        return mean, mean, 0.0

    var = float(np.var(vals, ddof=1))
    r = float(np.max(vals) - np.min(vals))
    log_term = math.log(3.0 / max(delta, 1e-12))
    radius = math.sqrt(2.0 * var * log_term / vals.size) + 3.0 * r * log_term / vals.size
    return mean, mean - radius, radius
