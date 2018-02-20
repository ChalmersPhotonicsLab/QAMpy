import pytest

from dsp.core import analog_frontend as canalog
from dsp import modulation, analog_frontend

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

    @pytest.mark.xfail(reason="core function does not preserve object type")
    def test_comp_rf_delay(self):
        s2 = canalog.comp_rf_delay(self.s, 0.001, self.s.fs)
        assert type(self.s) is type(s2)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_comp_IQ_attr(self, attr):
        s2 = canalog.comp_IQ_inbalance(self.s)
        assert getattr(self.s, attr) is getattr(s2, attr)

    @pytest.mark.xfail(reason="core function does not preserve object type")
    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_comp_rf_delay_attr(self, attr):
        s2 = canalog.comp_rf_delay(self.s, 0.001, self.s.fs)
        assert getattr(self.s, attr) is getattr(s2, attr)

    def test_comp_IQ2(self):
        s2 = analog_frontend.comp_IQ_inbalance(self.s)
        assert type(self.s) is type(s2)

    def test_comp_rf_delay2(self):
        s2 = analog_frontend.comp_rf_delay(self.s, 0.001)
        assert type(self.s) is type(s2)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_comp_IQ_attr2(self, attr):
        s2 = analog_frontend.comp_IQ_inbalance(self.s)
        assert getattr(self.s, attr) is getattr(s2, attr)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_comp_rf_delay_attr2(self, attr):
        s2 = analog_frontend.comp_rf_delay(self.s, 0.001)
        assert getattr(self.s, attr) is getattr(s2, attr)