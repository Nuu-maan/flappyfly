import numpy as np

from flappyfly.readout import Readout


def test_ridge_recovers_linear_target():
    rng = np.random.default_rng(0)
    X = rng.poisson(2.0, (500, 20)).astype(np.float32)
    w_true = rng.normal(size=20)
    t = X @ w_true + 3
    ro = Readout(np.arange(20)).fit(X, t, l2=1e-3)
    pred = (X - ro.mean) / ro.std @ ro.w + ro.b
    assert np.corrcoef(pred, t)[0, 1] > 0.999
    assert ro.act(np.r_[X[0], np.zeros(5)]) == (t[0] > 0)
