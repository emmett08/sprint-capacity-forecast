from __future__ import annotations
import math
from abc import ABC, abstractmethod
from forecast.core.types import Number, ThreePointEstimate

class TriadCalibrator(ABC):
    """Applies bias and upper-tail inflation to a three-point triad."""
    @abstractmethod
    def correct(self, triad: ThreePointEstimate, alpha: Number, delta_b: Number) -> ThreePointEstimate:
        ...

class LogMultiplicativeTriadCalibrator(TriadCalibrator):
    """Multiplicative correction in the natural domain; additive on the log scale."""
    def correct(self, triad: ThreePointEstimate, alpha: Number, delta_b: Number) -> ThreePointEstimate:
        if delta_b < 0:
            raise ValueError("delta_b must be non-negative.")
        g = math.exp(alpha)
        gb = math.exp(alpha + delta_b)
        a2 = g * triad.optimistic
        m2 = g * triad.most_likely
        b2 = gb * triad.pessimistic
        corrected = ThreePointEstimate(a2, m2, b2)
        corrected.validate()
        return corrected
