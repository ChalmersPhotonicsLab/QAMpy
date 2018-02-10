import pytest

from dsp import modulation, analog_frontend


class TestReturnObjects(object):
    s = modulation.ResampledQAM(16, 2**14, fs=2, nmodes=2)

    def test_comp_IQ(self):
        s2 = analog_frontend.comp_IQ_inbalance(self.s)
        assert type(self.s) is type(s2)

    def test_comp_rf_delay(self):
        s2 = analog_frontend.comp_rf_delay(self.s, 0.001, self.s.fs)
        assert type(self.s) is type(s2)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_comp_IQ_attr(self, attr):
        s2 = analog_frontend.comp_IQ_inbalance(self.s)
        assert getattr(self.s, attr) is getattr(s2, attr)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_comp_rf_delay_attr(self, attr):
        s2 = analog_frontend.comp_rf_delay(self.s, 0.001, self.s.fs)
        assert getattr(self.s, attr) is getattr(s2, attr)