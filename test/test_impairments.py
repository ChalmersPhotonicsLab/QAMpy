import pytest
import numpy as np
import numpy.testing as npt

from dsp import modulation, impairments, theory, helpers


class TestReturnObjects(object):
    s = modulation.ResampledQAM(16, 2 ** 14, fs=2, nmodes=2)

    def test_rotate_field(self):
        s2 = impairments.rotate_field(self.s, np.pi / 3)
        assert type(self.s) is type(s2)

    def test_apply_PMD(self):
        s2 = impairments.apply_PMD_to_field(self.s, np.pi / 3, 1e-3)
        assert type(self.s) is type(s2)

    def test_apply_phase_noise(self):
        s2 = impairments.apply_phase_noise(self.s, 1e-3)
        assert type(self.s) is type(s2)

    def test_change_snr(self):
        s2 = impairments.change_snr(self.s, 30)
        assert type(self.s) is type(s2)

    def test_add_carrier_offset(self):
        s2 = impairments.add_carrier_offset(self.s, 1e-3)
        assert type(self.s) is type(s2)

    def test_simulate_transmission(self):
        s2 = impairments.simulate_transmission(self.s, snr=20, freq_off=1e-4, lwdth=1e-4,
                                               dgd=1e-2)
        assert type(self.s) is type(s2)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_add_awgn(self, attr):
        s2 = impairments.add_awgn(self.s, 0.01)
        assert getattr(self.s, attr) is getattr(s2, attr)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_rotate_field_attr(self, attr):
        s2 = impairments.rotate_field(self.s, np.pi / 3)
        assert getattr(self.s, attr) is getattr(s2, attr)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_apply_PMD_attr(self, attr):
        s2 = impairments.apply_PMD_to_field(self.s, np.pi / 3, 1e-3)
        assert getattr(self.s, attr) is getattr(s2, attr)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_apply_phase_noise_attr(self, attr):
        s2 = impairments.apply_phase_noise(self.s, 1e-3)
        assert getattr(self.s, attr) is getattr(s2, attr)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_add_awgn_attr(self, attr):
        s2 = impairments.add_awgn(self.s, 0.01)
        assert getattr(self.s, attr) is getattr(s2, attr)

@pytest.mark.parametrize("M", [4, 16, 64, 128])
@pytest.mark.parametrize("snr", [0, 2, 4])
def test_snrvstheory_ser(M, snr):
    off = {4: 6, 16: 13, 64: 18, 128: 20}
    s = modulation.SignalQAMGrayCoded(M, 2**16)
    ss = impairments.change_snr(s, snr+off[M])
    ser = ss.cal_ser()
    ser_t = theory.ser_vs_es_over_n0_qam(10**((snr+off[M])/10), M)
    npt.assert_allclose(ser, ser_t, rtol=0.3)

