from .bias import BiasCalibrator, BiasPosterior, EmpiricalBayesGaussianCalibrator
from .commit import CommitCapacityEngine, DailyInputs, DailyMoments
from .core.types import BiasObservation, EngineerId, Number, ThreePointEstimate
from .correction import LogMultiplicativeTriadCalibrator, TriadCalibrator
from .inflation import QuantileUpperTailInflation, UpperTailInflationCalibrator
from .pert import BetaPert, BetaPertFactory, PertDistribution, PertFactory
from .service import PertCalibrator, default_calibrator

# 0.1 aliases for compatibility
BiasModel = BiasPosterior
class BiasEstimator(BiasCalibrator): ...
EmpiricalBayesGaussianBiasEstimator = EmpiricalBayesGaussianCalibrator
PessimisticInflationCalibrator = UpperTailInflationCalibrator
QuantilePessimisticInflation = QuantileUpperTailInflation
PertCorrector = TriadCalibrator
MultiplicativePertCorrector = LogMultiplicativeTriadCalibrator
BiasCorrectedPertService = PertCalibrator
default_bias_corrected_pert_service = default_calibrator

__all__ = [
    "ThreePointEstimate",
    "BiasObservation",
    "Number",
    "EngineerId",
    "BiasPosterior",
    "BiasCalibrator",
    "EmpiricalBayesGaussianCalibrator",
    "UpperTailInflationCalibrator",
    "QuantileUpperTailInflation",
    "PertDistribution",
    "BetaPert",
    "PertFactory",
    "BetaPertFactory",
    "TriadCalibrator",
    "LogMultiplicativeTriadCalibrator",
    "PertCalibrator",
    "default_calibrator",
    "BiasModel",
    "BiasEstimator",
    "EmpiricalBayesGaussianBiasEstimator",
    "PessimisticInflationCalibrator",
    "QuantilePessimisticInflation",
    "PertCorrector",
    "MultiplicativePertCorrector",
    "BiasCorrectedPertService",
    "default_bias_corrected_pert_service",
    "CommitCapacityEngine",
    "DailyInputs",
    "DailyMoments",
]

__version__ = "0.3.1"
