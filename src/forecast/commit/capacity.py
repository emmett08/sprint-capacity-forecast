from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import NormalDist
from typing import Dict, Mapping, Optional, Sequence, Tuple

from forecast.bias import BiasPosterior
from forecast.core.types import EngineerId, Number, ThreePointEstimate
from forecast.pert import PertDistribution
from forecast.service import PertCalibrator


@dataclass(frozen=True)
class DailyInputs:
    h_by_engineer: Mapping[EngineerId, Number]
    q_by_engineer: Mapping[EngineerId, Number]
    triad_by_engineer: Mapping[EngineerId, ThreePointEstimate]
    corr_by_pair: Optional[Mapping[Tuple[EngineerId, EngineerId], Number]] = None


@dataclass(frozen=True)
class DailyMoments:
    mu_C: Number
    var_C: Number


class CommitCapacityEngine:
    def __init__(self, calibrator: PertCalibrator, bias: BiasPosterior, delta_b: Number) -> None:
        self._cal = calibrator
        self._bias = bias
        self._delta_b = float(delta_b)

    def _z_alpha(self, alpha: Number) -> Number:
        if not (0.0 < alpha < 0.5):
            raise ValueError("alpha must be in (0, 0.5)")
        return NormalDist().inv_cdf(1.0 - float(alpha))

    def _build_focus(self, engineer_id: EngineerId, triad: ThreePointEstimate) -> PertDistribution:
        return self._cal.build_distribution(triad, engineer_id, self._bias, self._delta_b)

    def _pair_corr(self, corr: Optional[Mapping[Tuple[EngineerId, EngineerId], Number]], i: EngineerId, j: EngineerId) -> Optional[Number]:
        if corr is None:
            return None
        if (i, j) in corr:
            return float(corr[(i, j)])
        if (j, i) in corr:
            return float(corr[(j, i)])
        return None

    def _daily_moments(self, x: DailyInputs) -> DailyMoments:
        ids_h = set(x.h_by_engineer.keys())
        ids_q = set(x.q_by_engineer.keys())
        ids_t = set(x.triad_by_engineer.keys())
        if ids_h != ids_q or ids_h != ids_t:
            raise ValueError("Engineer key sets for h, q, and triad must match")
        ids: Tuple[EngineerId, ...] = tuple(sorted(ids_h))
        h = {i: float(x.h_by_engineer[i]) for i in ids}
        q = {i: float(x.q_by_engineer[i]) for i in ids}
        for i in ids:
            if not (0.0 <= q[i] <= 1.0) or h[i] <= 0.0:
                raise ValueError("q in [0,1] and h>0 are required")
        dists: Dict[EngineerId, PertDistribution] = {i: self._build_focus(i, x.triad_by_engineer[i]) for i in ids}
        muF = {i: float(dists[i].mean()) for i in ids}
        varF = {i: float(dists[i].variance()) for i in ids}
        for i in ids:
            if muF[i] <= 0.0 or varF[i] < 0.0 or not isfinite(muF[i]) or not isfinite(varF[i]):
                raise ValueError("Invalid PERT-induced focus moments")
        w = {i: h[i] * q[i] for i in ids}
        mu_C = sum(w[i] * muF[i] for i in ids)
        term_cov = 0.0
        for a_idx, i in enumerate(ids):
            term_cov += (w[i] ** 2) * varF[i]
            for j in ids[a_idx + 1 :]:
                rho = self._pair_corr(x.corr_by_pair, i, j)
                if rho is None:
                    continue
                cov_ij = float(rho) * sqrt(varF[i]) * sqrt(varF[j])
                term_cov += 2.0 * w[i] * w[j] * cov_ij
        term_diag = sum((h[i] ** 2) * (q[i] * (1.0 - q[i]) * (muF[i] * muF[i] + varF[i])) for i in ids)
        var_C = term_cov + term_diag
        if var_C < 0.0:
            raise ValueError("Negative variance detected; check correlation inputs")
        return DailyMoments(mu_C=mu_C, var_C=var_C)

    def probability_of_success(
        self,
        days: Sequence[DailyInputs],
        t: int,
        workload: Number,
        achieved_to_date: Number,
    ) -> Number:
        """
        P( C_hat_T >= W | C_hat_t ), assuming independence across remaining days
        and normal aggregation: S_t ~ N(mu_rem, sigma_rem^2).

        Args:
            days: schedule inputs for days 1..T
            t: current day index in [0, T]; remaining days are t+1..T
            workload: required effective hours W
            achieved_to_date: realised cumulative effective hours C_hat_t

        Returns:
            Probability in [0,1].
        """
        T = len(days)
        if not (0 <= t <= T):
            raise ValueError("t must be in [0, T]")
        if workload < 0 or achieved_to_date < 0:
            raise ValueError("workload and achieved_to_date must be non-negative")

        mus, vars_ = self.daily_schedule_moments(days)
        mu_rem = sum(mus[s] for s in range(t, T))
        var_rem = sum(vars_[s] for s in range(t, T))
        R_t = float(workload) - float(achieved_to_date)

        if var_rem <= 0.0:
            return 1.0 if mu_rem >= R_t else 0.0

        z = (mu_rem - R_t) / sqrt(var_rem)
        return NormalDist().cdf(z)

    def daily_schedule_moments(self, days: Sequence[DailyInputs]) -> Tuple[Tuple[Number, ...], Tuple[Number, ...]]:
        ms: list[Number] = []
        vs: list[Number] = []
        for d in days:
            m = self._daily_moments(d)
            ms.append(m.mu_C)
            vs.append(m.var_C)
        return tuple(ms), tuple(vs)

    def commit_capacity(self, days: Sequence[DailyInputs], t: int, alpha: Number) -> Number:
        T = len(days)
        if not (0 <= t <= T):
            raise ValueError("t must be in [0, T]")
        if t == T:
            return 0.0
        mus, vars_ = self.daily_schedule_moments(days)
        mu_rem = sum(mus[s] for s in range(t, T))
        var_rem = sum(vars_[s] for s in range(t, T))
        z = self._z_alpha(alpha)
        x = mu_rem - z * sqrt(var_rem) if var_rem > 0.0 else mu_rem
        return x if x > 0.0 else 0.0
