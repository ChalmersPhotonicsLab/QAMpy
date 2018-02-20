import pytest
import numpy as np
import numpy.testing as npt

from dsp.core import filter as cfilter
from dsp import modulation, filtering


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
    def test_filter_signal(self, ndim):
        x = np.random.randn(ndim, 2**15) + 0.j
        y = cfilter.filter_signal(x, 2, 0.01)
        assert x.shape == y.shape

    def test_filter_signal_1d(self):
        x = np.random.randn(2**15) + 0.j
        y = cfilter.filter_signal(x, 2, 0.01)
        assert x.shape == y.shape

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    @pytest.mark.parametrize("ftype", ["gauss", "exp", "bessel", "butter"])
    def test_filter_signal_analog(self, ndim, ftype):
        x = np.random.randn(ndim, 2**15) + 0.j
        y = cfilter.filter_signal_analog(x, 2, 0.01, ftype=ftype)
        assert x.shape == y.shape

    def test_filter_signal_analog_1d(self):
        x = np.random.randn(2**15) + 0.j
        y = cfilter.filter_signal_analog(x, 2, 0.01)
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
    s = modulation.ResampledQAM(16, 2 ** 14, fs=2, nmodes=2)

    @pytest.mark.xfail(reason="core version does not preserve subclass")
    def test_pre_filter(self):
        s2 = cfilter.pre_filter(self.s, 0.01)
        assert type(self.s) is type(s2)

    @pytest.mark.xfail(reason="core version does not preserve subclass")
    def test_filter_signal(self):
        s2 = cfilter.filter_signal(self.s, self.s.fs, 0.001)
        assert type(self.s) is type(s2)

    def test_filter_signal_analog(self):
        s2 = cfilter.filter_signal_analog(self.s, self.s.fs, 0.001)
        assert type(self.s) is type(s2)

    @pytest.mark.xfail(reason="core version does not preserve subclass")
    def test_rrcos_pulseshaping(self):
        s2 = cfilter.rrcos_pulseshaping(self.s, self.s.fs, 1 / self.s.fb, 0.1)
        assert type(self.s) is type(s2)

    def test_mvg_avg(self):
        s2 = cfilter.moving_average(self.s)
        assert type(self.s) is type(s2)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    @pytest.mark.xfail(reason="core version does not preserve subclass")
    def test_pre_filter_attr(self, attr):
        s2 = cfilter.pre_filter(self.s, 0.01)
        assert getattr(self.s, attr) is getattr(s2, attr)

    @pytest.mark.xfail(reason="core version does not preserve subclass")
    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_filter_signal_attr(self, attr):
        s2 = cfilter.filter_signal(self.s, self.s.fs, 0.01)
        assert getattr(self.s, attr) is getattr(s2, attr)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_filter_signal_analog_attr(self, attr):
        s2 = cfilter.filter_signal_analog(self.s, self.s.fs, 0.01)
        assert getattr(self.s, attr) is getattr(s2, attr)

    @pytest.mark.xfail(reason="core version does not preserve subclass")
    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_rrcos_pulseshaping_attr(self, attr):
        s2 = cfilter.rrcos_pulseshaping(self.s, self.s.fs, 1 / self.s.fb, 0.1)
        assert getattr(self.s, attr) is getattr(s2, attr)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_mvg_avg_attr(self, attr):
        s2 = cfilter.moving_average(self.s)
        assert getattr(self.s, attr) is getattr(s2, attr)

class TestReturnObjectsBasic(object):
    s = modulation.ResampledQAM(16, 2 ** 14, fs=2, nmodes=2)

    def test_pre_filter(self):
        s2 = filtering.pre_filter(self.s, 0.01)
        assert type(self.s) is type(s2)

    @pytest.mark.xfail(reason="filter_signal is not exported to basic")
    def test_filter_signal(self):
        s2 = filtering.filter_signal(self.s, self.s.fs, 0.001)
        assert type(self.s) is type(s2)

    def test_filter_signal_analog(self):
        s2 = filtering.filter_signal_analog(self.s, 0.001)
        assert type(self.s) is type(s2)

    def test_rrcos_pulseshaping(self):
        s2 = filtering.rrcos_pulseshaping(self.s, 0.1)
        assert type(self.s) is type(s2)

    def test_mvg_avg(self):
        s2 = filtering.moving_average(self.s)
        assert type(self.s) is type(s2)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_pre_filter_attr(self, attr):
        s2 = filtering.pre_filter(self.s, 0.01)
        assert getattr(self.s, attr) is getattr(s2, attr)

    @pytest.mark.xfail(reason="filter_signal is not exported to basic")
    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_filter_signal_attr(self, attr):
        s2 = filtering.filter_signal(self.s, 0.01)
        assert getattr(self.s, attr) is getattr(s2, attr)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_filter_signal_analog_attr(self, attr):
        s2 = filtering.filter_signal_analog(self.s, 0.01)
        assert getattr(self.s, attr) is getattr(s2, attr)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_rrcos_pulseshaping_attr(self, attr):
        s2 = filtering.rrcos_pulseshaping(self.s, 0.1)
        assert getattr(self.s, attr) is getattr(s2, attr)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_mvg_avg_attr(self, attr):
        s2 = filtering.moving_average(self.s)
        assert getattr(self.s, attr) is getattr(s2, attr)
