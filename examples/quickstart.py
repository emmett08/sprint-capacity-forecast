from __future__ import annotations

import math
from typing import Iterable

from forecast import (
    BiasObservation,
    CommitCapacityEngine,
    DailyInputs,
    ThreePointEstimate,
    default_calibrator,
)

obs = [
    BiasObservation("e1", modal_estimate_hours=6, actual_hours=9, optimistic_hours=4, most_likely_hours=6, pessimistic_hours=18),
    BiasObservation("e1", modal_estimate_hours=8, actual_hours=24, optimistic_hours=6, most_likely_hours=8, pessimistic_hours=14),
    BiasObservation("e2", modal_estimate_hours=3, actual_hours=2.5, optimistic_hours=2, most_likely_hours=3, pessimistic_hours=5),
    BiasObservation("e2", modal_estimate_hours=4, actual_hours=3.2),
    BiasObservation("e1", modal_estimate_hours=8.0, actual_hours=10.2, optimistic_hours=6.0, most_likely_hours=8.0, pessimistic_hours=12.0),
    BiasObservation("e2", modal_estimate_hours=6.0, actual_hours=5.1, optimistic_hours=5.0, most_likely_hours=6.0, pessimistic_hours=9.0),
]

cal = default_calibrator(shape=4.0)
bias = cal.fit_bias(obs)
delta_b = cal.fit_delta_b(obs, bias, target_coverage=0.9)

engine = CommitCapacityEngine(calibrator=cal, bias=bias, delta_b=delta_b)

triad = ThreePointEstimate(optimistic=5, most_likely=8, pessimistic=14)
dist = cal.build_distribution(triad, engineer_id="e1", bias=bias, delta_b=delta_b)


print("alpha_e1 =", bias.alpha_for("e1"))
print("alpha_e2 =", bias.alpha_for("e2"))
print("delta_b  =", delta_b)
print("mean     =", dist.mean())
print("var      =", dist.variance())
print("samples  =", dist.sample(n=5))


days = [
    DailyInputs(
        h_by_engineer={"e1": 6.0, "e2": 7.5},
        q_by_engineer={"e1": 0.95, "e2": 0.90},
        triad_by_engineer={"e1": ThreePointEstimate(0.8, 1.0, 1.3), "e2": ThreePointEstimate(0.7, 0.9, 1.2)},
        corr_by_pair={("e1", "e2"): 0.25},
    ),
    DailyInputs(
        h_by_engineer={"e1": 6.0, "e2": 7.5},
        q_by_engineer={"e1": 0.90, "e2": 0.85},
        triad_by_engineer={"e1": ThreePointEstimate(0.7, 0.95, 1.25), "e2": ThreePointEstimate(0.65, 0.9, 1.2)},
        corr_by_pair=None,
    ),
]

mus, vars_ = engine.daily_schedule_moments(days)
for t, (m, v) in enumerate(zip(mus, vars_), start=1):
    print(f"day {t}: mu_C={m:.6f} var_C={v:.6f} sd_C={math.sqrt(v):.6f}")

def report_commit(t_values: Iterable[int], alphas: Iterable[float]) -> None:
    """
    Prints the committed capacity for each combination of t and alpha values.

    Args:
        t_values (Iterable[int]): An iterable of integer values representing time periods or steps.
        alphas (Iterable[float]): An iterable of float values representing alpha parameters.

    Returns:
        None

    Side Effects:
        Prints the committed capacity for each (t, alpha) pair to the standard output.

    Note:
        Assumes that the variables `engine` and `days` are defined in the enclosing scope.
    """
    for t_val in t_values:
        for a in alphas:
            c = engine.commit_capacity(days=days, t=t_val, alpha=a)
            print(f"C_commit(t={t_val}, alpha={a:.3f}) = {c:.6f}")

def report_success(t_values: Iterable[int], workload: float, achieved: dict[int, float]) -> None:
    """
    Prints the probability of completing workload W by end of sprint, conditional on
    capacity achieved to date.

    Args:
        t_values: indices of days already completed
        workload: required workload W
        achieved: map t -> observed cumulative capacity C_hat_t
    """
    for t_val in t_values:
        a = achieved.get(t_val, 0.0)
        p = engine.probability_of_success(days=days, t=t_val, workload=workload, achieved_to_date=a)
        print(f"P_success(t={t_val}, achieved={a:.1f}, W={workload}) = {p:.6f}")


report_commit(t_values=[0, 1, 2], alphas=[0.10, 0.20, 0.05])

# Example: suppose workload is 20 effective hours total, and you observed nothing yet (t=0),
# then after day 1 you actually achieved 13.0 hours.
report_success(t_values=[0, 1], workload=20.0, achieved={0: 0.0, 1: 13.0})
