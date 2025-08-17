from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from statistics import mean
from typing import Dict, List, Mapping, MutableMapping, Sequence

from forecast.core.types import BiasObservation, EngineerId, Number


@dataclass(frozen=True)
class BiasPosterior:
    """Posterior for engineer bias on the log-ratio scale."""
    mu: Number
    sigma2_within: Number
    tau2_between: Number
    alpha_by_engineer: Mapping[EngineerId, Number]
    def alpha_for(self, engineer_id: EngineerId) -> Number:
        return self.alpha_by_engineer.get(engineer_id, 0.0)

class BiasCalibrator(ABC):
    """Abstract calibrator producing a BiasPosterior from observations."""
    def fit(self, observations: Sequence[BiasObservation]) -> BiasPosterior:
        return self._fit(observations)
    @abstractmethod
    def _fit(self, observations: Sequence[BiasObservation]) -> BiasPosterior:
        ...

class EmpiricalBayesGaussianCalibrator(BiasCalibrator):
    """Pooled Gaussian EB shrinkage of per-engineer log-errors."""
    def _fit(self, observations: Sequence[BiasObservation]) -> BiasPosterior:
        if not observations:
            raise ValueError("At least one observation is required.")
        etas_by_engineer: MutableMapping[EngineerId, List[Number]] = {}
        for obs in observations:
            etas_by_engineer.setdefault(obs.engineer_id, []).append(obs.eta())
        engineer_ids = list(etas_by_engineer.keys())
        n_engineers = len(engineer_ids)
        n_total = sum(len(v) for v in etas_by_engineer.values())
        eta_all = [e for vs in etas_by_engineer.values() for e in vs]
        mu_hat = mean(eta_all)
        ss_within = 0.0
        for vs in etas_by_engineer.values():
            if len(vs) > 1:
                m_i = mean(vs)
                ss_within += sum((e - m_i) ** 2 for e in vs)
        dof_within = max(n_total - n_engineers, 1)
        sigma2_hat = ss_within / dof_within if ss_within > 0 else 0.0
        means = {i: mean(vs) for i, vs in etas_by_engineer.items()}
        bar_devs = [(m_i - mu_hat) for m_i in means.values()]
        var_bar = sum(d * d for d in bar_devs) / max(n_engineers - 1, 1)
        mean_inv_ni_sigma = mean(sigma2_hat / max(len(etas_by_engineer[i]), 1) for i in engineer_ids)
        tau2_hat = max(var_bar - mean_inv_ni_sigma, 0.0)
        alpha_map: Dict[EngineerId, Number] = {}
        for i in engineer_ids:
            n_i = len(etas_by_engineer[i])
            if tau2_hat == 0.0:
                kappa_i = 0.0
            else:
                kappa_i = (n_i * tau2_hat) / (n_i * tau2_hat + sigma2_hat)
            alpha_map[i] = kappa_i * (means[i] - mu_hat)
        return BiasPosterior(mu=mu_hat, sigma2_within=sigma2_hat, tau2_between=tau2_hat, alpha_by_engineer=alpha_map)
