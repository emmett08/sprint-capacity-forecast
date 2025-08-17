from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

from forecast.core.types import Number, ThreePointEstimate


class PertDistribution(ABC):
    """PERT-like continuous distribution interface."""
    @abstractmethod
    def mean(self) -> Number: ...
    @abstractmethod
    def variance(self) -> Number: ...
    @abstractmethod
    def sample(self, n: int = 1, rng: Optional[random.Random] = None) -> List[Number]: ...

@dataclass(frozen=True)
class BetaPert(PertDistribution):
    a: Number
    m: Number
    b: Number
    shape: Number = 4.0
    def __post_init__(self) -> None:
        triad = ThreePointEstimate(self.a, self.m, self.b)
        triad.validate()
        if self.shape <= 0:
            raise ValueError("shape must be positive.")
    @property
    def _alpha_beta(self) -> Tuple[Number, Number]:
        span = self.b - self.a
        if span <= 0:
            raise ValueError("Require b > a.")
        alpha = 1.0 + self.shape * (self.m - self.a) / span
        beta = 1.0 + self.shape * (self.b - self.m) / span
        if alpha <= 0 or beta <= 0:
            raise ValueError("Derived Beta shape parameters must be positive.")
        return alpha, beta
    def mean(self) -> Number:
        return (self.a + self.shape * self.m + self.b) / (self.shape + 2.0)
    def variance(self) -> Number:
        alpha, beta = self._alpha_beta
        span = self.b - self.a
        v_beta = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1.0))
        return (span ** 2) * v_beta
    def sample(self, n: int = 1, rng: Optional[random.Random] = None) -> List[Number]:
        alpha, beta = self._alpha_beta
        r = rng or random
        u = [r.betavariate(alpha, beta) for _ in range(n)]
        return [self.a + (self.b - self.a) * ui for ui in u]

class PertFactory(ABC):
    @abstractmethod
    def from_triad(self, triad: ThreePointEstimate) -> PertDistribution: ...

class BetaPertFactory(PertFactory):
    def __init__(self, shape: Number = 4.0) -> None:
        self._shape = shape
    def from_triad(self, triad: ThreePointEstimate) -> PertDistribution:
        triad.validate()
        return BetaPert(a=triad.optimistic, m=triad.most_likely, b=triad.pessimistic, shape=self._shape)
