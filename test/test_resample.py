import pytest
import numpy as np
import numpy.testing as npt

import qampy.core.special_fcts
from qampy.core import resample, utils


class TestRRcosresample(object):
    @pytest.mark.parametrize("N", np.arange(1, 6))
    def test_len(self, N):
        x = np.random.randn(1000) + 1.j
        xn = resample.rrcos_resample(x, fold=1, fnew=N, Ts=None, beta=0.1)
        assert xn.shape[0] == x.shape[0]*N

    @pytest.mark.parametrize("beta", np.linspace(0.1, 2, 20))
    def test_beta(self, beta):
        beta = 10**-beta
        N = 1000
        x = np.zeros(N, dtype=float)
        x[N//2] = 1
        xn = resample.rrcos_resample(x, fold=1, fnew=4, Ts=1, beta=beta)
        t = np.linspace(0, N, xn.shape[0], endpoint=False) - N//2
        b = qampy.core.special_fcts.rrcos_time(t, beta, 1)
        xn /= xn.max()
        b /= b.max()
        npt.assert_array_almost_equal(xn, b, decimal=1)
