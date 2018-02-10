import pytest
import numpy as np
import numpy.testing as npt

from dsp import filter


class TestMovingAvg(object):
    def test_lengths(self):
        for i in range(100, 131):
            x = np.arange(i)
            for j in range(5, 30):
                assert len(filter.moving_average(x, j)) == i-j+1

    def test_numbers1(self):
        npt.assert_allclose(np.array([6,9,12])/3, filter.moving_average(np.arange(1,6), 3))

    def test_numbers2(self):
        npt.assert_allclose(np.array([1.,1., 1.]), filter.moving_average(np.ones(5), 3))

    def test_numbers3(self):
        npt.assert_allclose(np.array([6,9,12,15])/3, filter.moving_average(np.arange(1,7), 3))

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

