import pytest
import numpy as np
import numpy.testing as npt

from dsp.core import filter
from dsp import modulation


class TestMovingAvg(object):
    def test_lengths(self):
        for i in range(100, 131):
            x = np.arange(i)
            for j in range(5, 30):
                assert len(filter.moving_average(x, j)) == i - j + 1

    def test_numbers1(self):
        npt.assert_allclose(np.array([6,9,12]) / 3, filter.moving_average(np.arange(1, 6), 3))

    def test_numbers2(self):
        npt.assert_allclose(np.array([1.,1., 1.]), filter.moving_average(np.ones(5), 3))

    def test_numbers3(self):
        npt.assert_allclose(np.array([6,9,12,15]) / 3, filter.moving_average(np.arange(1, 7), 3))

class Test2dcapability(object):
    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_pre_filter(self, ndim):
        x = np.random.randn(ndim, 2**15) + 0.j
        y = filter.pre_filter(x, 0.01)
        assert x.shape == y.shape

    def test_pre_filter_1d(self):
        x = np.random.randn(2**15) + 0.j
        y = filter.pre_filter(x, 0.01)
        assert x.shape == y.shape

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_filter_signal(self, ndim):
        x = np.random.randn(ndim, 2**15) + 0.j
        y = filter.filter_signal(x, 2, 0.01)
        assert x.shape == y.shape

    def test_filter_signal_1d(self):
        x = np.random.randn(2**15) + 0.j
        y = filter.filter_signal(x, 2, 0.01)
        assert x.shape == y.shape

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    @pytest.mark.parametrize("ftype", ["gauss", "exp", "bessel", "butter"])
    def test_filter_signal_analog(self, ndim, ftype):
        x = np.random.randn(ndim, 2**15) + 0.j
        y = filter.filter_signal_analog(x, 2, 0.01, ftype=ftype)
        assert x.shape == y.shape

    def test_filter_signal_analog_1d(self):
        x = np.random.randn(2**15) + 0.j
        y = filter.filter_signal_analog(x, 2, 0.01)
        assert x.shape == y.shape

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_rrcos_pulseshaping(self, ndim):
        x = np.random.randn(ndim, 2**15) + 0.j
        y = filter.rrcos_pulseshaping(x, 2, 1, 0.2)
        assert x.shape == y.shape

    def test_rrcos_pulseshaping(self):
        x = np.random.randn(2**15) + 0.j
        y = filter.rrcos_pulseshaping(x, 2, 1, 0.2)
        assert x.shape == y.shape

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_moving_avg(self, ndim):
        x = np.random.randn(ndim, 2**15) + 0.j
        y = filter.moving_average(x, N=3)
        assert x.shape[0] == y.shape[0]

    def test_moving_avg_1d(self):
        x = np.random.randn(2**15) + 0.j
        y = filter.moving_average(x)
        assert x.shape == y.shape

class TestReturnObjects(object):
    s = modulation.ResampledQAM(16, 2 ** 14, fs=2, nmodes=2)

    def test_pre_filter(self):
        s2 = filter.pre_filter(self.s, 0.01)
        assert type(self.s) is type(s2)

    def test_filter_signal(self):
        s2 = filter.filter_signal(self.s, self.s.fs, 0.001)
        assert type(self.s) is type(s2)

    def test_filter_signal_analog(self):
        s2 = filter.filter_signal_analog(self.s, self.s.fs, 0.001)
        assert type(self.s) is type(s2)

    def test_rrcos_pulseshaping(self):
        s2 = filter.rrcos_pulseshaping(self.s, self.s.fs, 1 / self.s.fb, 0.1)
        assert type(self.s) is type(s2)

    def test_mvg_avg(self):
        s2 = filter.moving_average(self.s)
        assert type(self.s) is type(s2)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_pre_filter_attr(self, attr):
        s2 = filter.pre_filter(self.s, 0.01)
        assert getattr(self.s, attr) is getattr(s2, attr)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_filter_signal_attr(self, attr):
        s2 = filter.filter_signal(self.s, self.s.fs, 0.01)
        assert getattr(self.s, attr) is getattr(s2, attr)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_filter_signal_analog_attr(self, attr):
        s2 = filter.filter_signal_analog(self.s, self.s.fs, 0.01)
        assert getattr(self.s, attr) is getattr(s2, attr)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_rrcos_pulseshaping_attr(self, attr):
        s2 = filter.rrcos_pulseshaping(self.s, self.s.fs, 1 / self.s.fb, 0.1)
        assert getattr(self.s, attr) is getattr(s2, attr)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_mvg_avg_attr(self, attr):
        s2 = filter.moving_average(self.s)
        assert getattr(self.s, attr) is getattr(s2, attr)
