from __future__ import annotations

from math import floor
from typing import Sequence


def linear_quantile(values: Sequence[float], p: float) -> float:
    xs = sorted(float(x) for x in values)
    if not xs:
        raise ValueError("empty sample")
    if len(xs) == 1:
        return xs[0]
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0,1]")
    h = (len(xs) - 1) * p
    i = int(floor(h))
    frac = h - i
    if i + 1 < len(xs):
        return xs[i] * (1.0 - frac) + xs[i + 1] * frac
    return xs[-1]
