from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List, Sequence

from forecast.bias import BiasPosterior
from forecast.core.types import BiasObservation, Number
from forecast.utils import linear_quantile


class UpperTailInflationCalibrator(ABC):
    """Calibrates delta_b to achieve target coverage for pessimistic bounds."""
    def fit(self, observations: Sequence[BiasObservation], bias: BiasPosterior, target_coverage: float) -> Number:
        return self._fit(observations, bias, target_coverage)
    @abstractmethod
    def _fit(self, observations: Sequence[BiasObservation], bias: BiasPosterior, target_coverage: float) -> Number:
        ...

# TODO validate implementation
class QuantileUpperTailInflation(UpperTailInflationCalibrator):
    """Log-domain inflation so that P(actual <= exp(alpha_i + delta_b) * b) ~= target_coverage."""
    def _fit(self, observations: Sequence[BiasObservation], bias: BiasPosterior, target_coverage: float) -> Number:
        if not 0.5 <= target_coverage < 1.0:
            raise ValueError("target_coverage must be in [0.5, 1).")
        ratios: List[Number] = []
        for obs in observations:
            if obs.optimistic_hours is None or obs.most_likely_hours is None or obs.pessimistic_hours is None:
                continue
            b = float(obs.pessimistic_hours)
            if b <= 0:
                continue
            alpha = bias.alpha_for(obs.engineer_id)
            ratios.append(obs.actual_hours / (math.exp(alpha) * b))
        if not ratios:
            raise ValueError("No historical PERT pessimistic bounds available for calibration.")
        q = linear_quantile(ratios, target_coverage)
        q = max(q, 1.0)
        return math.log(q)
