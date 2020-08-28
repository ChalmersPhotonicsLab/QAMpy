import pytest
import numpy as np
import numpy.testing as npt

from qampy.core import filter as cfilter
from qampy import signals, filtering


class TestMovingAvg(object):
    def test_lengths(self):
        for i in range(100, 131):
            x = np.arange(i)
            for j in range(5, 30):
                assert len(cfilter.moving_average(x, j)) == i - j + 1

    def test_numbers1(self):
        npt.assert_allclose(np.array([6,9,12]) / 3, cfilter.moving_average(np.arange(1, 6), 3))

    def test_numbers2(self):
        npt.assert_allclose(np.array([1.,1., 1.]), cfilter.moving_average(np.ones(5), 3))

    def test_numbers3(self):
        npt.assert_allclose(np.array([6,9,12,15]) / 3, cfilter.moving_average(np.arange(1, 7), 3))

class Test2dcapability(object):
    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_pre_filter(self, ndim):
        x = np.random.randn(ndim, 2**15) + 0.j
        y = cfilter.pre_filter(x, 0.01)
        assert x.shape == y.shape

    def test_pre_filter_1d(self):
        x = np.random.randn(2**15) + 0.j
        y = cfilter.pre_filter(x, 0.01)
        assert x.shape == y.shape

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    @pytest.mark.parametrize("ftype", ["gauss", "exp", "bessel", "butter"])
    @pytest.mark.parametrize("analog", [True, False])
    @pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
    def test_filter_signal_type(self, ndim, ftype, analog, dtype):
        x = np.random.randn(ndim, 2**15) + 0.j
        x = x.astype(dtype)
        y = cfilter.filter_signal(x, 2, 0.01, ftype=ftype, analog=analog)
        assert x.shape == y.shape
        assert x.dtype == y.dtype

    @pytest.mark.parametrize("ftype", ["gauss", "exp", "bessel", "butter"])
    @pytest.mark.parametrize("analog", [True, False])
    def test_filter_signal_1d(self, ftype, analog):
        x = np.random.randn(2**15) + 0.j
        y = cfilter.filter_signal(x, 2, 0.01, ftype=ftype, analog=analog)
        assert x.shape == y.shape

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_rrcos_pulseshaping(self, ndim):
        x = np.random.randn(ndim, 2**15) + 0.j
        y = cfilter.rrcos_pulseshaping(x, 2, 1, 0.2)
        assert x.shape == y.shape

    def test_rrcos_pulseshaping(self):
        x = np.random.randn(2**15) + 0.j
        y = cfilter.rrcos_pulseshaping(x, 2, 1, 0.2)
        assert x.shape == y.shape

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    @pytest.mark.parametrize("N", [3, 4, 10])
    def test_moving_avg(self, ndim, N):
        x = np.random.randn(ndim, 2**15) + 0.j
        y = cfilter.moving_average(x, N=N)
        assert x.shape[0] == y.shape[0]
        assert x.shape[1] - N + 1 == y.shape[1]

    @pytest.mark.parametrize("N", [3, 4, 10])
    def test_moving_avg_1d(self, N):
        x = np.random.randn(2**15) + 0.j
        y = cfilter.moving_average(x, N=N)
        assert x.shape[0] - N + 1 == y.shape[0]

class TestReturnObjectsAdv(object):
    s = signals.ResampledQAM(16, 2 ** 14, fs=2, nmodes=2)

    @pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
    def test_pre_filter(self, dtype):
        s = self.s.astype(dtype)
        s2 = cfilter.pre_filter(s, 0.01)
        assert s2.dtype is np.dtype(dtype)

    @pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
    def test_filter_signal(self, dtype):
        s = self.s.astype(dtype)
        s2 = cfilter.filter_signal(s, s.fs, 0.001)
        assert s2.dtype == np.dtype(dtype)

    @pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
    def test_filter_signal_analog(self, dtype):
        s = self.s.astype(dtype)
        s2 = cfilter.filter_signal(s, s.fs, 0.001, analog=True)
        assert s2.dtype is np.dtype(dtype)

    @pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
    def test_rrcos_pulseshaping(self, dtype):
        s = self.s.astype(dtype)
        s2 = cfilter.rrcos_pulseshaping(s, s.fs, 1 / s.fb, 0.1)
        assert s2.dtype is np.dtype(dtype)

    @pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
    def test_mvg_avg(self, dtype):
        s = self.s.astype(dtype)
        s2 = cfilter.moving_average(s)
        assert type(self.s) is type(s2)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    @pytest.mark.xfail(reason="core version does not preserve subclass")
    def test_pre_filter_attr(self, attr):
        s = self.s
        s2 = cfilter.pre_filter(s, 0.01)
        assert getattr(s, attr) is getattr(s2, attr)

    @pytest.mark.xfail(reason="core version does not preserve subclass")
    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_filter_signal_attr(self, attr):
        s = self.s
        s2 = cfilter.filter_signal(s, s.fs, 0.01)
        assert getattr(s, attr) is getattr(s2, attr)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    @pytest.mark.xfail(reason="core version does not preserve subclass")
    def test_filter_signal_analog_attr(self, attr):
        s = self.s
        s2 = cfilter.filter_signal(s, s.fs, 0.01, analog=True)
        assert getattr(s, attr) is getattr(s2, attr)

    @pytest.mark.xfail(reason="core version does not preserve subclass")
    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_rrcos_pulseshaping_attr(self, attr):
        s = self.s
        s2 = cfilter.rrcos_pulseshaping(s, s.fs, 1 / s.fb, 0.1)
        assert getattr(s, attr) is getattr(s2, attr)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    @pytest.mark.xfail(reason="core version does not preserve subclass")
    def test_mvg_avg_attr(self, attr):
        s = self.s
        s2 = cfilter.moving_average(s)
        assert getattr(s, attr) is getattr(s2, attr)

class TestReturnObjectsBasic(object):

    @pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
    def test_pre_filter(self, dtype):
        s = signals.ResampledQAM(16, 2 ** 14, fs=2, nmodes=2, dtype=dtype)
        s2 = filtering.pre_filter(s, 0.01)
        assert type(s) is type(s2)
        assert s.dtype is s2.dtype

    @pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
    def test_filter_signal(self, dtype):
        s = signals.ResampledQAM(16, 2 ** 14, fs=2, nmodes=2, dtype=dtype)
        s2 = filtering.filter_signal(s, 0.001)
        assert type(s) is type(s2)
        assert s.dtype is s2.dtype

    @pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
    def test_filter_signal_analog(self, dtype):
        s = signals.ResampledQAM(16, 2 ** 14, fs=2, nmodes=2, dtype=dtype)
        s2 = filtering.filter_signal_analog(s, 0.001)
        assert type(s) is type(s2)
        assert s.dtype is s2.dtype

    @pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
    def test_rrcos_pulseshaping(self, dtype):
        s = signals.ResampledQAM(16, 2 ** 14, fs=2, nmodes=2, dtype=dtype)
        s2 = filtering.rrcos_pulseshaping(s, 0.1)
        assert type(s) is type(s2)
        assert s.dtype is s2.dtype

    @pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
    def test_mvg_avg(self, dtype):
        s = signals.ResampledQAM(16, 2 ** 14, fs=2, nmodes=2, dtype=dtype)
        s2 = filtering.moving_average(s)
        assert type(s) is type(s2)
        assert s.dtype is s2.dtype

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_pre_filter_attr(self, attr):
        s = signals.ResampledQAM(16, 2 ** 14, fs=2, nmodes=2)
        s2 = filtering.pre_filter(s, 0.01)
        assert getattr(s, attr) is getattr(s2, attr)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_filter_signal_attr(self, attr):
        s = signals.ResampledQAM(16, 2 ** 14, fs=2, nmodes=2)
        s2 = filtering.filter_signal(s, 0.01)
        assert getattr(s, attr) is getattr(s2, attr)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_filter_signal_analog_attr(self, attr):
        s = signals.ResampledQAM(16, 2 ** 14, fs=2, nmodes=2)
        s2 = filtering.filter_signal_analog(s, 0.01)
        assert getattr(s, attr) is getattr(s2, attr)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_rrcos_pulseshaping_attr(self, attr):
        s = signals.ResampledQAM(16, 2 ** 14, fs=2, nmodes=2)
        s2 = filtering.rrcos_pulseshaping(s, 0.1)
        assert getattr(s, attr) is getattr(s2, attr)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_mvg_avg_attr(self, attr):
        s = signals.ResampledQAM(16, 2 ** 14, fs=2, nmodes=2)
        s2 = filtering.moving_average(s)
        assert getattr(s, attr) is getattr(s2, attr)
