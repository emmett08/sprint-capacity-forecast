from __future__ import annotations

import math
import random
from typing import Mapping, Optional, List
import pytest

from forecast.bias.bias import BiasCalibrator
from forecast.commit import CommitCapacityEngine, DailyInputs, DailyMoments
from forecast.core.types import EngineerId, Number, ThreePointEstimate
from forecast.bias import BiasPosterior
from forecast.correction.correction import TriadCalibrator
from forecast.inflation.inflation import UpperTailInflationCalibrator
from forecast.pert import PertDistribution
from forecast.pert.pert import PertFactory
from forecast.service import PertCalibrator
from forecast import default_calibrator

class _StubDist(PertDistribution):
    def __init__(self, mean: float, var: float) -> None:
        self._m = float(mean); self._v = float(var)
    def mean(self) -> float: return self._m
    def variance(self) -> float: return self._v
    def sample(self, n: int = 1, rng: Optional[random.Random] = None) -> List[Number]:
        return [self._m] * n

class _StubCal:
    def __init__(
        self,
        m: Optional[Mapping[EngineerId, float]] = None,
        v: Optional[Mapping[EngineerId, float]] = None,
        *,
        mean_by_e: Optional[Mapping[EngineerId, float]] = None,
        var_by_e: Optional[Mapping[EngineerId, float]] = None,
    ) -> None:
        if mean_by_e is not None or var_by_e is not None:
            if mean_by_e is None or var_by_e is None:
                raise TypeError("Provide both mean_by_e and var_by_e, or both m and v.")
            m, v = mean_by_e, var_by_e
        if m is None or v is None:
            raise TypeError("Provide both m and v, or both mean_by_e and var_by_e.")
        self._m = {k: float(x) for k, x in m.items()}
        self._v = {k: float(x) for k, x in v.items()}

    def build_distribution(
        self,
        triad: ThreePointEstimate,
        engineer_id: EngineerId,
        bias: BiasPosterior,
        delta_b: Number,
    ) -> PertDistribution:
        triad.validate()
        _ = float(delta_b); _ = bias.alpha_for(engineer_id)
        return _StubDist(self._m[engineer_id], self._v[engineer_id])

class _DummyBiasCal(BiasCalibrator):
    def _fit(self, observations):
        return BiasPosterior(mu=0.0, sigma2_within=0.0, tau2_between=0.0, alpha_by_engineer={})

class _DummyInfl(UpperTailInflationCalibrator):
    def _fit(self, observations, bias, target_coverage):
        return 0.0

class _DummyTriad(TriadCalibrator):
    def correct(self, triad: ThreePointEstimate, alpha: Number, delta_b: Number) -> ThreePointEstimate:
        return triad

class _DummyPertFactory(PertFactory):
    def from_triad(self, triad: ThreePointEstimate) -> PertDistribution:
        raise NotImplementedError

class _StubAsPertCalibrator(PertCalibrator):
    def __init__(self, stub: _StubCal) -> None:
        super().__init__(
            bias_calibrator=_DummyBiasCal(),
            tail_inflation=_DummyInfl(),
            triad_calibrator=_DummyTriad(),
            pert_factory=_DummyPertFactory(),
        )
        self._stub = stub

    def build_distribution(
        self,
        triad: ThreePointEstimate,
        engineer_id: EngineerId,
        bias: BiasPosterior,
        delta_b: Number,
    ) -> PertDistribution:
        return self._stub.build_distribution(triad, engineer_id, bias, delta_b)
    
def test_daily_moments_independent_two_engineers() -> None:
    cal = _StubCal(mean_by_e={"e1": 1.0, "e2": 0.8}, var_by_e={"e1": 0.09, "e2": 0.16})
    bias = BiasPosterior(mu=0.0, sigma2_within=0.0, tau2_between=0.0, alpha_by_engineer={})
    eng = CommitCapacityEngine(calibrator=_StubAsPertCalibrator(cal), bias=bias, delta_b=0.0)

    x = DailyInputs(
        h_by_engineer={"e1": 6.0, "e2": 7.5},
        q_by_engineer={"e1": 0.95, "e2": 0.90},
        triad_by_engineer={"e1": ThreePointEstimate(0.9, 1.0, 1.1), "e2": ThreePointEstimate(0.7, 0.9, 1.2)},
        corr_by_pair=None,
    )
    m = eng._daily_moments(x)
    w1, w2 = 6.0 * 0.95, 7.5 * 0.90
    mu_expected = w1 * 1.0 + w2 * 0.8
    term_cov = (w1**2) * 0.09 + (w2**2) * 0.16
    v1 = 0.95 * (1 - 0.95) * (1.0**2 + 0.09)
    v2 = 0.90 * (1 - 0.90) * (0.8**2 + 0.16)
    term_diag = (6.0**2) * v1 + (7.5**2) * v2
    var_expected = term_cov + term_diag

    assert isinstance(m, DailyMoments)
    assert m.mu_C == pytest.approx(mu_expected, rel=1e-12)
    assert m.var_C == pytest.approx(var_expected, rel=1e-12)


def test_daily_moments_with_positive_correlation() -> None:
    cal = _StubCal(mean_by_e={"e1": 1.0, "e2": 1.2}, var_by_e={"e1": 0.25, "e2": 0.36})
    bias = BiasPosterior(0.0, 0.0, 0.0, {})
    eng = CommitCapacityEngine(calibrator=_StubAsPertCalibrator(cal), bias=bias, delta_b=0.0)

    x = DailyInputs(
        h_by_engineer={"e1": 5.0, "e2": 5.0},
        q_by_engineer={"e1": 1.0, "e2": 1.0},
        triad_by_engineer={"e1": ThreePointEstimate(0.9, 1.0, 1.1), "e2": ThreePointEstimate(0.9, 1.1, 1.3)},
        corr_by_pair={("e1", "e2"): 0.5},
    )
    m = eng._daily_moments(x)
    w1 = w2 = 5.0
    base = (w1**2) * 0.25 + (w2**2) * 0.36
    cov = 2.0 * w1 * w2 * (0.5 * math.sqrt(0.25) * math.sqrt(0.36))
    assert m.var_C == pytest.approx(base + cov, rel=1e-12)
    assert m.mu_C == pytest.approx(w1 * 1.0 + w2 * 1.2, rel=1e-12)


def test_corr_symmetry_lookup() -> None:
    cal = _StubCal(mean_by_e={"a": 1.0, "b": 1.0}, var_by_e={"a": 0.01, "b": 0.01})
    bias = BiasPosterior(0.0, 0.0, 0.0, {})
    eng = CommitCapacityEngine(calibrator=_StubAsPertCalibrator(cal), bias=bias, delta_b=0.0)

    x = DailyInputs(
        h_by_engineer={"a": 1.0, "b": 1.0},
        q_by_engineer={"a": 1.0, "b": 1.0},
        triad_by_engineer={"a": ThreePointEstimate(0.9, 1.0, 1.1), "b": ThreePointEstimate(0.9, 1.0, 1.1)},
        corr_by_pair={("b", "a"): 0.3},
    )
    m = eng._daily_moments(x)
    base = 0.01 + 0.01
    cov = 2.0 * 1.0 * 1.0 * (0.3 * math.sqrt(0.01) * math.sqrt(0.01))
    assert m.var_C == pytest.approx(base + cov, rel=1e-12)


def test_invalid_keys_raises() -> None:
    cal = _StubCal(mean_by_e={"e1": 1.0}, var_by_e={"e1": 0.01})
    bias = BiasPosterior(0.0, 0.0, 0.0, {})
    eng = CommitCapacityEngine(calibrator=_StubAsPertCalibrator(cal), bias=bias, delta_b=0.0)

    x = DailyInputs(
        h_by_engineer={"e1": 1.0},
        q_by_engineer={"e1": 1.0},
        triad_by_engineer={"e2": ThreePointEstimate(0.9, 1.0, 1.1)},
    )
    with pytest.raises(ValueError, match="Engineer key sets"):
        eng._daily_moments(x)


def test_invalid_h_or_q_raises() -> None:
    cal = _StubCal(mean_by_e={"e1": 1.0}, var_by_e={"e1": 0.01})
    bias = BiasPosterior(0.0, 0.0, 0.0, {})
    eng = CommitCapacityEngine(calibrator=_StubAsPertCalibrator(cal), bias=bias, delta_b=0.0)

    bad_q = DailyInputs(
        h_by_engineer={"e1": 1.0},
        q_by_engineer={"e1": 1.1},
        triad_by_engineer={"e1": ThreePointEstimate(0.9, 1.0, 1.1)},
    )
    with pytest.raises(ValueError, match="q in \\[0,1\\] and h>0"):
        eng._daily_moments(bad_q)

    bad_h = DailyInputs(
        h_by_engineer={"e1": 0.0},
        q_by_engineer={"e1": 1.0},
        triad_by_engineer={"e1": ThreePointEstimate(0.9, 1.0, 1.1)},
    )
    with pytest.raises(ValueError, match="q in \\[0,1\\] and h>0"):
        eng._daily_moments(bad_h)


def test_negative_variance_rejected_for_pathological_corr() -> None:
    cal = _StubCal(mean_by_e={"e1": 1.0, "e2": 1.0}, var_by_e={"e1": 1.0, "e2": 1.0})
    bias = BiasPosterior(0.0, 0.0, 0.0, {})
    eng = CommitCapacityEngine(calibrator=_StubAsPertCalibrator(cal), bias=bias, delta_b=0.0)

    x = DailyInputs(
        h_by_engineer={"e1": 10.0, "e2": 10.0},
        q_by_engineer={"e1": 1.0, "e2": 1.0},
        triad_by_engineer={"e1": ThreePointEstimate(0.9, 1.0, 1.1), "e2": ThreePointEstimate(0.9, 1.0, 1.1)},
        corr_by_pair={("e1", "e2"): -2.5},
    )
    with pytest.raises(ValueError, match="Negative variance"):
        eng._daily_moments(x)


def test_alpha_bounds_and_z_values() -> None:
    cal = _StubCal(mean_by_e={"e": 1.0}, var_by_e={"e": 0.0})
    bias = BiasPosterior(0.0, 0.0, 0.0, {})
    eng = CommitCapacityEngine(calibrator=_StubAsPertCalibrator(cal), bias=bias, delta_b=0.0)
    with pytest.raises(ValueError, match="alpha must be in"):
        eng._z_alpha(0.0)
    with pytest.raises(ValueError, match="alpha must be in"):
        eng._z_alpha(0.5)
    z_10 = eng._z_alpha(0.10)
    z_05 = eng._z_alpha(0.05)
    assert z_05 > z_10 > 0.0
    assert z_10 == pytest.approx(1.2815515655446004, rel=1e-12)
    assert z_05 == pytest.approx(1.6448536269514722, rel=1e-12)

def test_daily_schedule_and_commit_capacity_endpoints() -> None:
    cal = _StubCal(mean_by_e={"e1": 1.0}, var_by_e={"e1": 0.25})
    bias = BiasPosterior(0.0, 0.0, 0.0, {})
    eng = CommitCapacityEngine(calibrator=_StubAsPertCalibrator(cal), bias=bias, delta_b=0.0)

    days = [
        DailyInputs(
            h_by_engineer={"e1": 8.0},
            q_by_engineer={"e1": 1.0},
            triad_by_engineer={"e1": ThreePointEstimate(0.9, 1.0, 1.1)},
        ),
        DailyInputs(
            h_by_engineer={"e1": 6.0},
            q_by_engineer={"e1": 0.5},
            triad_by_engineer={"e1": ThreePointEstimate(0.9, 1.0, 1.1)},
        ),
    ]
    mus, vars_ = eng.daily_schedule_moments(days)
    assert mus == pytest.approx((8.0, 3.0), rel=1e-12)
    assert vars_ == pytest.approx((16.0, 13.5), rel=1e-12)

    c0 = eng.commit_capacity(days, t=0, alpha=0.10)
    mu_rem = sum(mus)
    sd_rem = math.sqrt(sum(vars_))
    expected = max(0.0, mu_rem - 1.2815515655446004 * sd_rem)
    assert c0 == pytest.approx(expected, rel=1e-12)

    c1 = eng.commit_capacity(days, t=1, alpha=0.10)
    expected1 = max(0.0, mus[1] - 1.2815515655446004 * math.sqrt(vars_[1]))
    assert c1 == pytest.approx(expected1, rel=1e-12)

    assert eng.commit_capacity(days, t=2, alpha=0.10) == 0.0


def test_t_bounds_in_commit_capacity() -> None:
    cal = _StubCal(mean_by_e={"e": 1.0}, var_by_e={"e": 0.0})
    bias = BiasPosterior(0.0, 0.0, 0.0, {})
    eng = CommitCapacityEngine(calibrator=_StubAsPertCalibrator(cal), bias=bias, delta_b=0.0)
    days = [
        DailyInputs(
            h_by_engineer={"e": 1.0},
            q_by_engineer={"e": 1.0},
            triad_by_engineer={"e": ThreePointEstimate(0.9, 1.0, 1.1)},
        )
    ]
    with pytest.raises(ValueError, match="t must be in \\[0, T\\]"):
        eng.commit_capacity(days, t=-1, alpha=0.1)
    with pytest.raises(ValueError, match="t must be in \\[0, T\\]"):
        eng.commit_capacity(days, t=2, alpha=0.1)


def test_commit_is_monotone_in_alpha() -> None:
    cal = _StubCal(mean_by_e={"e1": 1.0, "e2": 0.9}, var_by_e={"e1": 0.16, "e2": 0.09})
    bias = BiasPosterior(0.0, 0.0, 0.0, {})
    eng = CommitCapacityEngine(calibrator=_StubAsPertCalibrator(cal), bias=bias, delta_b=0.0)
    days = [
        DailyInputs(
            h_by_engineer={"e1": 6.0, "e2": 6.0},
            q_by_engineer={"e1": 0.9, "e2": 0.9},
            triad_by_engineer={"e1": ThreePointEstimate(0.9, 1.0, 1.2), "e2": ThreePointEstimate(0.85, 0.95, 1.15)},
        )
    ]
    c20 = eng.commit_capacity(days, t=0, alpha=0.20)
    c10 = eng.commit_capacity(days, t=0, alpha=0.10)
    c05 = eng.commit_capacity(days, t=0, alpha=0.05)
    assert c20 >= c10 >= c05


def test_integration_default_calibrator_rejects_degenerate_triad() -> None:
    cal = default_calibrator(shape=4.0)
    bias = BiasPosterior(mu=0.0, sigma2_within=0.0, tau2_between=0.0, alpha_by_engineer={})
    eng = CommitCapacityEngine(calibrator=cal, bias=bias, delta_b=0.0)

    x = DailyInputs(
        h_by_engineer={"e": 1.0},
        q_by_engineer={"e": 1.0},
        triad_by_engineer={"e": ThreePointEstimate(1.0, 1.0, 1.0)},
    )
    with pytest.raises(ValueError, match="Degenerate PERT triad"):
        eng._daily_moments(x)

def test_probability_of_success_matches_closed_form() -> None:
    # Day1: mu=8, var=16; Day2: mu=3, var=13.5 (from prior unit test)
    cal = _StubCal(m={"e1": 1.0}, v={"e1": 0.25})
    bias = BiasPosterior(0.0, 0.0, 0.0, {})
    eng = CommitCapacityEngine(calibrator=_StubAsPertCalibrator(cal), bias=bias, delta_b=0.0)

    days = [
        DailyInputs(
            h_by_engineer={"e1": 8.0},
            q_by_engineer={"e1": 1.0},
            triad_by_engineer={"e1": ThreePointEstimate(0.9, 1.0, 1.1)},
        ),
        DailyInputs(
            h_by_engineer={"e1": 6.0},
            q_by_engineer={"e1": 0.5},
            triad_by_engineer={"e1": ThreePointEstimate(0.9, 1.0, 1.1)},
        ),
    ]
    mus, vars_ = eng.daily_schedule_moments(days)
    assert mus == pytest.approx((8.0, 3.0))
    assert vars_ == pytest.approx((16.0, 13.5))

    # At t=0, achieved=0, choose W=10
    mu_rem = sum(mus)              # 11
    var_rem = sum(vars_)           # 29.5
    z = (mu_rem - 10.0) / math.sqrt(var_rem)
    from statistics import NormalDist
    expected = NormalDist().cdf(z)

    p = eng.probability_of_success(days, t=0, workload=10.0, achieved_to_date=0.0)
    assert p == pytest.approx(expected, rel=1e-12)

def test_probability_edge_zero_variance() -> None:
    # Zero variance remaining: deterministic outcome.
    cal = _StubCal(mean_by_e={"e": 1.0}, var_by_e={"e": 0.0})
    bias = BiasPosterior(0.0, 0.0, 0.0, {})
    eng = CommitCapacityEngine(calibrator=_StubAsPertCalibrator(cal), bias=bias, delta_b=0.0)

    days = [
        DailyInputs(
            h_by_engineer={"e": 5.0},
            q_by_engineer={"e": 1.0},
            triad_by_engineer={"e": ThreePointEstimate(0.9, 1.0, 1.1)},
        )
    ]
    # Remaining mean = 5, variance = 0
    assert eng.probability_of_success(days, t=0, workload=4.9, achieved_to_date=0.0) == 1.0
    assert eng.probability_of_success(days, t=0, workload=5.0, achieved_to_date=0.0) == 1.0
    assert eng.probability_of_success(days, t=0, workload=5.1, achieved_to_date=0.0) == 0.0

def test_bounds_and_inputs_validation_for_success_probability() -> None:
    cal = _StubCal(mean_by_e={"e": 1.0}, var_by_e={"e": 0.25})
    bias = BiasPosterior(0.0, 0.0, 0.0, {})
    eng = CommitCapacityEngine(calibrator=_StubAsPertCalibrator(cal), bias=bias, delta_b=0.0)
    days = [
        DailyInputs(
            h_by_engineer={"e": 1.0},
            q_by_engineer={"e": 1.0},
            triad_by_engineer={"e": ThreePointEstimate(0.9, 1.0, 1.1)},
        )
    ]
    with pytest.raises(ValueError, match="t must be in \\[0, T\\]"):
        eng.probability_of_success(days, t=-1, workload=1.0, achieved_to_date=0.0)
    with pytest.raises(ValueError, match="t must be in \\[0, T\\]"):
        eng.probability_of_success(days, t=2, workload=1.0, achieved_to_date=0.0)
    with pytest.raises(ValueError, match="must be non-negative"):
        eng.probability_of_success(days, t=0, workload=-1.0, achieved_to_date=0.0)
    with pytest.raises(ValueError, match="must be non-negative"):
        eng.probability_of_success(days, t=0, workload=1.0, achieved_to_date=-0.1)
