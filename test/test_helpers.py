import pytest
import numpy as np
import numpy.testing as npt
from dsp import modulation, helpers

class TestReturnObject(object):
    def test_cabssquared(self):
        s = modulation.SignalQAMGrayCoded(64, 2**12)
        s2 = helpers.cabssquared(s)
        assert type(s) is type(s2)

    def test_dB2lin(self):
        s = modulation.SignalQAMGrayCoded(64, 2**12)
        s2 = helpers.dB2lin(s)
        assert type(s) is type(s2)

    def test_lin2dB(self):
        s = modulation.SignalQAMGrayCoded(64, 2**12)
        s2 = helpers.lin2dB(s)
        assert type(s) is type(s2)

    def test_normalise_and_center(self):
        s = modulation.SignalQAMGrayCoded(64, 2**12)
        s2 = helpers.normalise_and_center(s)
        assert type(s) is type(s2)

    def test_dump_edges(self):
        s = modulation.SignalQAMGrayCoded(64, 2**12)
        s2 = helpers.dump_edges(s, 100)
        assert type(s) is type(s2)

class Test2DCap(object):
    @pytest.mark.parametrize("nmodes", [1, 2, 3])
    def test_cabssquared(self, nmodes):
        s = modulation.SignalQAMGrayCoded(64, 2**12, nmodes=nmodes)
        s2 = helpers.cabssquared(s)
        assert s.shape == s2.shape

    @pytest.mark.parametrize("nmodes", [1, 2, 3])
    def test_dB2lin(self, nmodes):
        s = modulation.SignalQAMGrayCoded(64, 2**12, nmodes=nmodes)
        s2 = helpers.dB2lin(s)
        assert s.shape == s2.shape

    @pytest.mark.parametrize("nmodes", [1, 2, 3])
    def test_lin2dB(self, nmodes):
        s = modulation.SignalQAMGrayCoded(64, 2**12, nmodes=nmodes)
        s2 = helpers.lin2dB(s)
        assert s.shape == s2.shape

    @pytest.mark.parametrize("nmodes", [1, 2, 3])
    def test_normalise_and_center(self, nmodes):
        s = modulation.SignalQAMGrayCoded(64, 2**12, nmodes=nmodes)
        s2 = helpers.normalise_and_center(s)
        assert s.shape == s2.shape

    @pytest.mark.parametrize("nmodes", [1, 2, 3])
    def test_dump_edges(self, nmodes):
        s = modulation.SignalQAMGrayCoded(64, 2**12, nmodes=nmodes)
        s2 = helpers.dump_edges(s, 100)
        assert s.shape == (s2.shape[0], s2.shape[1]+200)

