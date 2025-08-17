from pytest import approx

from forecast import BiasObservation, EmpiricalBayesGaussianCalibrator


def test_bias_fit_basic():
    obs = [
        BiasObservation("a", 5.0, 6.0),
        BiasObservation("a", 5.0, 5.5),
        BiasObservation("b", 3.0, 2.7),
    ]
    post = EmpiricalBayesGaussianCalibrator().fit(obs)
    assert post.mu == approx(post.mu)
    assert post.sigma2_within >= 0
    assert post.tau2_between >= 0
