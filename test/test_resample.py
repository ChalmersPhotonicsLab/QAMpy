import pytest
import numpy as np
import random
import numpy.testing as npt
import matplotlib.pylab as plt

from dsp import resample, filter, utils, modulation

class TestRRcosResample(object):
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
        b = utils.rrcos_time(t, beta, 1)
        xn /= xn.max()
        b /= b.max()
        npt.assert_array_almost_equal(xn, b, decimal=1)

    @pytest.mark.parametrize("fconv", [True, False])
    @pytest.mark.parametrize("ntaps", [4000, 4001, None])
    @pytest.mark.parametrize("os", np.arange(2, 4))
    def test_resample_err(self, fconv, ntaps, os):
        Q = modulation.QAMModulator(32)
        s, sym, bits = Q.generate_signal(2**16, None, ndim=1)
        s2 = resample.rrcos_resample(sym[0], 1, os, Ts=1, beta=0.1, fftconv=fconv, taps=ntaps, renormalise=True)
        s3 = resample.rrcos_resample(s2, os , 1, Ts=1, beta=0.1, fftconv=fconv, taps=ntaps, renormalise=True)
        npt.assert_almost_equal(sym[0,10:-10], s3[10:-10], 2) # need to eliminate edges because they introduce errors

    @pytest.mark.parametrize("fconv", [True, False])
    @pytest.mark.parametrize("ntaps", [4000, 4001, None])
    def test_resample_err(self, fconv, ntaps):
        Q = modulation.QAMModulator(32)
        s, sym, bits = Q.generate_signal(2**16, None, ndim=1)
        s2 = resample.rrcos_resample(sym[0], 1, 2, Ts=1, beta=0.1, fftconv=fconv, taps=ntaps, renormalise=True)
        ss2 = abs(np.fft.fftshift(np.fft.fft(s2)))**2
        assert np.mean(ss2[0:int(1/8*ss2.size)]) < np.mean(ss2[int(3/8*ss2.size):int(4/8*ss2.size)])/1000


