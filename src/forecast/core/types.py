from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

Number = float
EngineerId = str

@dataclass(frozen=True)
class ThreePointEstimate:
    optimistic: Number
    most_likely: Number
    pessimistic: Number
    def validate(self) -> None:
        if not (self.optimistic > 0 and self.most_likely > 0 and self.pessimistic > 0):
            raise ValueError("All PERT points must be strictly positive.")
        if not (self.optimistic <= self.most_likely <= self.pessimistic):
            raise ValueError("Require optimistic <= most_likely <= pessimistic.")
        if self.optimistic == self.pessimistic:
            raise ValueError("Degenerate PERT triad with zero range.")

@dataclass(frozen=True)
class BiasObservation:
    engineer_id: EngineerId
    modal_estimate_hours: Number
    actual_hours: Number
    
    # TODO: replace with Optional[ThreePointEstimate] <- rename to PERTTriad
    optimistic_hours: Optional[Number] = None
    most_likely_hours: Optional[Number] = None
    pessimistic_hours: Optional[Number] = None

    def eta(self) -> Number:
        if self.modal_estimate_hours <= 0 or self.actual_hours <= 0:
            raise ValueError("Modal and actual hours must be strictly positive.")
        return math.log(self.actual_hours / self.modal_estimate_hours)
    
    def has_pert(self) -> bool:
        return (
            self.optimistic_hours is not None
            and self.most_likely_hours is not None
            and self.pessimistic_hours is not None
        )
