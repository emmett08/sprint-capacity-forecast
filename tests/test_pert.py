from forecast import BetaPertFactory, ThreePointEstimate


def test_beta_pert_factory():
    triad = ThreePointEstimate(1.0, 2.0, 5.0)
    dist = BetaPertFactory(shape=4.0).from_triad(triad)
    assert dist.mean() > 0
    assert dist.variance() > 0
