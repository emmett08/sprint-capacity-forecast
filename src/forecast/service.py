from __future__ import annotations

from forecast.bias import BiasCalibrator, BiasPosterior, EmpiricalBayesGaussianCalibrator
from forecast.core.types import EngineerId, Number, ThreePointEstimate
from forecast.correction import LogMultiplicativeTriadCalibrator, TriadCalibrator
from forecast.inflation import QuantileUpperTailInflation, UpperTailInflationCalibrator
from forecast.pert import BetaPertFactory, PertDistribution, PertFactory


class PertCalibrator:
    """Facade for fitting bias, calibrating upper-tail inflation, and producing PERT distributions."""
    def __init__(
        self,
        bias_calibrator: BiasCalibrator,
        tail_inflation: UpperTailInflationCalibrator,
        triad_calibrator: TriadCalibrator,
        pert_factory: PertFactory,
    ) -> None:
        self._bias_calibrator = bias_calibrator
        self._tail_inflation = tail_inflation
        self._triad_calibrator = triad_calibrator
        self._pert_factory = pert_factory

    def fit_bias(self, observations):
        return self._bias_calibrator.fit(observations)
    
    def fit_delta_b(self, observations, bias: BiasPosterior, target_coverage: float) -> Number:
        return self._tail_inflation.fit(observations, bias, target_coverage)
    
    def build_distribution(
        self,
        triad: ThreePointEstimate,
        engineer_id: EngineerId,
        bias: BiasPosterior,
        delta_b: Number,
    ) -> PertDistribution:
        alpha = bias.alpha_for(engineer_id)
        corrected = self._triad_calibrator.correct(triad, alpha, delta_b)
        return self._pert_factory.from_triad(corrected)

def default_calibrator(shape: Number = 4.0) -> PertCalibrator:
    return PertCalibrator(
        bias_calibrator=EmpiricalBayesGaussianCalibrator(),
        tail_inflation=QuantileUpperTailInflation(),
        triad_calibrator=LogMultiplicativeTriadCalibrator(),
        pert_factory=BetaPertFactory(shape=shape),
    )
