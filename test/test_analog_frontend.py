import pytest

from dsp.core import analog_frontend as canalog
from dsp import modulation

class TestMultiDim(object):
    @pytest.mark.parametrize("ndim", [1,2,3])
    def test_comp_IQ(self, ndim):
        s = modulation.SignalQAMGrayCoded(64, 2**12, nmodes=ndim)
        s2 = canalog.comp_IQ_inbalance(s)
        assert s.shape == s2.shape

    #TODO: currently rf delay fails
    @pytest.mark.parametrize("ndim", [1,2,3])
    def test_comp_rf_delay(self, ndim):
        s = modulation.SignalQAMGrayCoded(64, 2**12, nmodes=ndim)
        s2 = canalog.comp_rf_delay(s, 0.001, s.fs)
        assert s.shape == s2.shape

class TestReturnObjects(object):
    s = modulation.ResampledQAM(16, 2 ** 14, fs=2, nmodes=2)

    def test_comp_IQ(self):
        s2 = canalog.comp_IQ_inbalance(self.s)
        assert type(self.s) is type(s2)

    #TODO: currently rf delay fails
    def test_comp_rf_delay(self):
        s2 = canalog.comp_rf_delay(self.s, 0.001, self.s.fs)
        assert type(self.s) is type(s2)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_comp_IQ_attr(self, attr):
        s2 = canalog.comp_IQ_inbalance(self.s)
        assert getattr(self.s, attr) is getattr(s2, attr)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_comp_rf_delay_attr(self, attr):
        s2 = canalog.comp_rf_delay(self.s, 0.001, self.s.fs)
        assert getattr(self.s, attr) is getattr(s2, attr)