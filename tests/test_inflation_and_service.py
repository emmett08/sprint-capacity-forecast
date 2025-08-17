from forecast import (
    BiasObservation,
    EmpiricalBayesGaussianCalibrator,
    QuantileUpperTailInflation,
    default_calibrator,
)


def test_quantile_inflation():
    obs = [
        BiasObservation("a", 5.0, 6.0, 4.0, 5.0, 8.0),
        BiasObservation("a", 6.0, 6.5, 4.5, 6.0, 10.0),
        BiasObservation("b", 3.0, 2.9, 2.0, 3.0, 5.0),
    ]
    bias = EmpiricalBayesGaussianCalibrator().fit(obs)
    delta = QuantileUpperTailInflation().fit(obs, bias, 0.8)
    assert delta >= 0

def test_service_pipeline():
    obs = [
        BiasObservation("a", 5.0, 6.0, 4.0, 5.0, 8.0),
        BiasObservation("a", 6.0, 6.5, 4.5, 6.0, 10.0),
        BiasObservation("b", 3.0, 2.9, 2.0, 3.0, 5.0),
    ]
    svc = default_calibrator()
    bias = svc.fit_bias(obs)
    delta = svc.fit_delta_b(obs, bias, 0.9)
    assert delta >= 0
