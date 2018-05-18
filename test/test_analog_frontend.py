import pytest

from qampy.core import analog_frontend as canalog
from qampy import signals, analog_frontend

class TestMultiDim(object):
    @pytest.mark.parametrize("ndim", [1,2,3])
    def test_comp_IQ(self, ndim):
        s = signals.SignalQAMGrayCoded(64, 2 ** 12, nmodes=ndim)
        s2 = canalog.comp_IQ_inbalance(s)
        assert s.shape == s2.shape

    @pytest.mark.parametrize("ndim", [1,2,3])
    def test_comp_rf_delay(self, ndim):
        s = signals.SignalQAMGrayCoded(64, 2 ** 12, nmodes=ndim)
        s2 = canalog.comp_rf_delay(s, 0.001, s.fs)
        assert s.shape == s2.shape

    @pytest.mark.parametrize("ndim", [1,2,3])
    def test_orthonormalize_signal(self, ndim):
        s = signals.SignalQAMGrayCoded(64, 2 ** 12, nmodes=ndim)
        s2 = canalog.orthonormalize_signal(s, int(s.fs//s.fb))
        assert s.shape == s2.shape

class TestReturnObjects(object):
    s = signals.ResampledQAM(16, 2 ** 14, fs=2, nmodes=2)

    def test_comp_IQ(self):
        s2 = canalog.comp_IQ_inbalance(self.s)
        assert type(self.s) is type(s2)

    def test_orthonormalize_signal(self):
        s2 = canalog.orthonormalize_signal(self.s, os=2)
        assert type(self.s) is type(s2)

    @pytest.mark.xfail(reason="core function does not preserve object type")
    def test_comp_rf_delay(self):
        s2 = canalog.comp_rf_delay(self.s, 0.001, self.s.fs)
        assert type(self.s) is type(s2)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_comp_IQ_attr(self, attr):
        s2 = canalog.comp_IQ_inbalance(self.s)
        assert getattr(self.s, attr) is getattr(s2, attr)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_orthonormalize_signal_attr(self, attr):
        s2 = canalog.orthonormalize_signal(self.s, int(self.s.fs/self.s.fb))
        assert getattr(self.s, attr) is getattr(s2, attr)

    @pytest.mark.xfail(reason="core function does not preserve object type")
    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_comp_rf_delay_attr(self, attr):
        s2 = canalog.comp_rf_delay(self.s, 0.001, self.s.fs)
        assert getattr(self.s, attr) is getattr(s2, attr)

    def test_comp_IQ2(self):
        s2 = analog_frontend.comp_IQ_inbalance(self.s)
        assert type(self.s) is type(s2)

    def test_orthonormalize_signal2(self):
        s2 = analog_frontend.orthonormalize_signal(self.s)
        assert type(self.s) is type(s2)

    def test_comp_rf_delay2(self):
        s2 = analog_frontend.comp_rf_delay(self.s, 0.001)
        assert type(self.s) is type(s2)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_comp_IQ_attr2(self, attr):
        s2 = analog_frontend.comp_IQ_inbalance(self.s)
        assert getattr(self.s, attr) is getattr(s2, attr)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_orthonormalize_signal_attr2(self, attr):
        s2 = analog_frontend.orthonormalize_signal(self.s)
        assert getattr(self.s, attr) is getattr(s2, attr)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_comp_rf_delay_attr2(self, attr):
        s2 = analog_frontend.comp_rf_delay(self.s, 0.001)
        assert getattr(self.s, attr) is getattr(s2, attr)