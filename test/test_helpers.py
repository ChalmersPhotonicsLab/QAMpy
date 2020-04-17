import pytest
import numpy as np
import numpy.testing as npt
from qampy import signals, helpers

class TestReturnObject(object):
    def test_cabssquared(self):
        s = signals.SignalQAMGrayCoded(64, 2 ** 12)
        s2 = helpers.cabssquared(s)
        assert type(s) is type(s2)

    def test_dB2lin(self):
        s = signals.SignalQAMGrayCoded(64, 2 ** 12)
        s2 = helpers.dB2lin(s)
        assert type(s) is type(s2)

    def test_lin2dB(self):
        s = signals.SignalQAMGrayCoded(64, 2 ** 12)
        s2 = helpers.lin2dB(s)
        assert type(s) is type(s2)

    def test_normalise_and_center(self):
        s = signals.SignalQAMGrayCoded(64, 2 ** 12)
        s2 = helpers.normalise_and_center(s)
        assert type(s) is type(s2)

    def test_dump_edges(self):
        s = signals.SignalQAMGrayCoded(64, 2 ** 12)
        s2 = helpers.dump_edges(s, 100)
        assert type(s) is type(s2)

    def test_rescale_signal(self):
        s = signals.SignalQAMGrayCoded(64, 2 ** 12)
        s2 = helpers.rescale_signal(s, 100)
        assert type(s) is type(s2)


class Test2DCap(object):
    @pytest.mark.parametrize("nmodes", [1, 2, 3])
    def test_cabssquared(self, nmodes):
        s = signals.SignalQAMGrayCoded(64, 2 ** 12, nmodes=nmodes)
        s2 = helpers.cabssquared(s)
        assert s.shape == s2.shape

    @pytest.mark.parametrize("nmodes", [1, 2, 3])
    def test_dB2lin(self, nmodes):
        s = signals.SignalQAMGrayCoded(64, 2 ** 12, nmodes=nmodes)
        s2 = helpers.dB2lin(s)
        assert s.shape == s2.shape

    @pytest.mark.parametrize("nmodes", [1, 2, 3])
    def test_lin2dB(self, nmodes):
        s = signals.SignalQAMGrayCoded(64, 2 ** 12, nmodes=nmodes)
        s2 = helpers.lin2dB(s)
        assert s.shape == s2.shape

    @pytest.mark.parametrize("nmodes", [1, 2, 3])
    def test_normalise_and_center(self, nmodes):
        s = signals.SignalQAMGrayCoded(64, 2 ** 12, nmodes=nmodes)
        s2 = helpers.normalise_and_center(s)
        assert s.shape == s2.shape

    @pytest.mark.parametrize("nmodes", [1, 2, 3])
    def test_dump_edges(self, nmodes):
        s = signals.SignalQAMGrayCoded(64, 2 ** 12, nmodes=nmodes)
        s2 = helpers.dump_edges(s, 100)
        assert s.shape == (s2.shape[0], s2.shape[1]+200)

    @pytest.mark.parametrize("nmodes", [1, 2, 3])
    def test_rescale_signal(self, nmodes):
        s = signals.SignalQAMGrayCoded(64, 2 ** 12, nmodes=nmodes)
        s2 = helpers.rescale_signal(s, 100)
        assert s.shape == (s2.shape[0], s2.shape[1])

@pytest.mark.parametrize("nmodes", [1, 2, 3])
def test_normalise(nmodes):
    s = signals.SignalQAMGrayCoded(64, 2 ** 12, nmodes=nmodes)
    s2 = helpers.normalise_and_center(s)
    s3 = np.zeros(s.shape, dtype=s.dtype)
    for i in range(nmodes):
        s3[i] = helpers.normalise_and_center(s[i])
    npt.assert_array_almost_equal(s2, s3)

class TestRescaleFunction(object):

    @pytest.mark.parametrize("nmodes", [1, 2, 3])
    def test_rescale_signal_real(self, nmodes):
        s = signals.SignalQAMGrayCoded(64, 2 ** 12, nmodes=nmodes)
        s /= abs(s.max())
        scale = np.random.randint(1, 4, nmodes)
        for i in range(nmodes):
            s[i].real *= scale[i]
        s2 = helpers.rescale_signal(s)
        for i in range(nmodes):
            assert np.isclose(abs(s2[i].real).max(), 1)
            assert np.isclose(abs(s2[i].imag).max(), 1/scale[i])

    @pytest.mark.parametrize("nmodes", [1, 2, 3])
    def test_rescale_signal_imag(self, nmodes):
        s = signals.SignalQAMGrayCoded(64, 2 ** 12, nmodes=nmodes)
        s /= abs(s.max())
        scale = np.random.randint(1, 4, nmodes)
        for i in range(nmodes):
            s[i].imag *= scale[i]
        s2 = helpers.rescale_signal(s)
        for i in range(nmodes):
            assert np.isclose(abs(s2[i].imag).max(), 1)
            assert np.isclose(abs(s2[i].real).max(), 1/scale[i])

    @pytest.mark.parametrize("nmodes", [1, 2, 3])
    def test_rescale_signal_altern(self, nmodes):
        s = signals.SignalQAMGrayCoded(64, 2 ** 12, nmodes=nmodes)
        s /= abs(s.max())
        scale = np.random.randint(1, 4, nmodes)
        for i in range(nmodes):
            if i%2:
                s[i].imag *= scale[i]
            else:
                s[i].real *= scale[i]
        s2 = helpers.rescale_signal(s)
        for i in range(nmodes):
            if i%2:
                assert np.isclose(abs(s2[i].imag).max(), 1)
                assert np.isclose(abs(s2[i].real).max(), 1/scale[i])
            else:
                assert np.isclose(abs(s2[i].real).max(), 1)
                assert np.isclose(abs(s2[i].imag).max(), 1/scale[i])
                
    @pytest.mark.parametrize("nmodes", [1, 2, 3])
    def test_rescale_signal_ndscale(self, nmodes):
        s = signals.SignalQAMGrayCoded(64, 2 ** 12, nmodes=nmodes)
        s /= abs(s.max())
        scale = np.random.randint(1, 4, nmodes)
        s2 = helpers.rescale_signal(s, scale)
        for i in range(nmodes):
            assert np.isclose(abs(s2[i].imag).max(), scale[i])
            assert np.isclose(abs(s2[i].real).max(), scale[i])

        
