import pytest
import numpy as np
import numpy.testing as npt

from qampy import signals, impairments
from qampy.core import impairments as cimpairments

class TestReturnDtype(object):
    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_rotate_field(self, dtype):
        s = signals.ResampledQAM(16, 2**14, fs=2, nmodes=2, dtype=dtype)
        s2 = impairments.rotate_field(s, np.pi / 3)
        assert np.dtype(dtype) is s2.dtype

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_apply_PMD(self, dtype):
        s = signals.ResampledQAM(16, 2**14, fs=2, nmodes=2, dtype=dtype)
        s2 = impairments.apply_PMD(s, np.pi / 3, 1e-3)
        assert np.dtype(dtype) is s2.dtype

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_apply_phase_noise(self, dtype):
        s = signals.ResampledQAM(16, 2**14, fs=2, nmodes=2, dtype=dtype)
        s2 = impairments.apply_phase_noise(s, 1e-3)
        assert np.dtype(dtype) is s2.dtype

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_change_snr(self, dtype):
        s = signals.ResampledQAM(16, 2**14, fs=2, nmodes=2, dtype=dtype)
        s2 = impairments.change_snr(s, 30)
        assert np.dtype(dtype) is s2.dtype

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_add_carrier_offset(self, dtype):
        s = signals.ResampledQAM(16, 2**14, fs=2, nmodes=2, dtype=dtype)
        s2 = impairments.add_carrier_offset(s, 1e-3)
        assert np.dtype(dtype) is s2.dtype

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_sim_tx_response(self, dtype):
        s = signals.SignalQAMGrayCoded(16, 2**14,  nmodes=2, dtype=dtype, fb=20e9)
        s = s.resample(s.fb*2, beta=0.1)
        s2 = impairments.sim_tx_response(s)
        assert np.dtype(dtype) is s2.dtype

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_sim_DAC_response(self, dtype):
        s = signals.SignalQAMGrayCoded(16, 2**14,  nmodes=2, dtype=dtype, fb=20e9)
        s = s.resample(s.fb*2, beta=0.1)
        s2 = impairments.sim_DAC_response(s)
        assert np.dtype(dtype) is s2.dtype

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_sim_mod_response(self, dtype):
        s = signals.SignalQAMGrayCoded(16, 2**14,  nmodes=2, dtype=dtype, fb=20e9)
        s = s.resample(s.fb*2, beta=0.1)
        s2 = impairments.sim_mod_response(s)
        assert np.dtype(dtype) is s2.dtype
        
class TestReturnObjects(object):
    s = signals.ResampledQAM(16, 2 ** 14, fs=2, nmodes=2)

    def test_rotate_field(self):
        s2 = impairments.rotate_field(self.s, np.pi / 3)
        assert type(self.s) is type(s2)

    def test_apply_PMD(self):
        s2 = impairments.apply_PMD(self.s, np.pi / 3, 1e-3)
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
        
    def test_sim_tx_response(self):
        s = signals.SignalQAMGrayCoded(64, 2**15, fb=20e9, nmodes=1)
        s2 = s.resample(s.fb*2, beta=0.1)
        s3 = impairments.sim_tx_response(s2)
        assert type(s) is type(s2)

    def test_sim_DAC_response(self):
        s = signals.SignalQAMGrayCoded(64, 2**15, fb=20e9, nmodes=1)
        s2 = s.resample(s.fb*2, beta=0.1)
        s3 = impairments.sim_DAC_response(s2)
        assert type(s) is type(s2)

    def test_sim_mod_response(self):
        s = signals.SignalQAMGrayCoded(64, 2**15, fb=20e9, nmodes=1)
        s2 = s.resample(s.fb*2, beta=0.1)
        s3 = impairments.sim_mod_response(s2)
        assert type(s) is type(s2)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_sim_tx_response(self, attr):
        s2 = impairments.sim_tx_response(self.s, dac_params={"cutoff":0.9})
        assert getattr(self.s, attr) is getattr(s2, attr)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_sim_DAC_response(self, attr):
        s2 = impairments.sim_DAC_response(self.s, cutoff=0.9)
        assert getattr(self.s, attr) is getattr(s2, attr)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_sim_mod_response(self, attr):
        s2 = impairments.sim_mod_response(self.s)
        assert getattr(self.s, attr) is getattr(s2, attr)

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
        s2 = impairments.apply_PMD(self.s, np.pi / 3, 1e-3)
        assert getattr(self.s, attr) is getattr(s2, attr)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_apply_phase_noise_attr(self, attr):
        s2 = impairments.apply_phase_noise(self.s, 1e-3)
        assert getattr(self.s, attr) is getattr(s2, attr)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_add_awgn_attr(self, attr):
        s2 = impairments.add_awgn(self.s, 0.01)
        assert getattr(self.s, attr) is getattr(s2, attr)


class TestCoreReturnDtype(object):
    
    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_rotate_field(self, dtype):
        s = signals.SignalQAMGrayCoded(16, 2**16, fb=10e9, nmodes=2, dtype=dtype)
        s2 = cimpairments.rotate_field(s, np.random.random(1)[0])
        assert s.dtype == s2.dtype

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_apply_PMD_to_field(self, dtype):
        s = signals.SignalQAMGrayCoded(16, 2**16, fb=10e9, nmodes=2, dtype=dtype)
        s2 = cimpairments.apply_PMD_to_field(s, 0.4, 120e-12, s.fs)
        assert s.dtype == s2.dtype

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_apply_phase_noise(self, dtype):
        s = signals.SignalQAMGrayCoded(16, 2**16, fb=10e9, nmodes=2, dtype=dtype)
        s2 = cimpairments.apply_phase_noise(s, 100e3, s.fs)
        assert s.dtype == s2.dtype
        
    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_add_awgn(self, dtype):
        s = signals.SignalQAMGrayCoded(16, 2**16, fb=10e9, nmodes=2, dtype=dtype)
        s2 = cimpairments.add_awgn(s, 0.03)
        assert s.dtype == s2.dtype

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_change_snr(self, dtype):
        s = signals.SignalQAMGrayCoded(16, 2**16, fb=10e9, nmodes=2, dtype=dtype)
        s2 = cimpairments.change_snr(s, 20, s.fb, s.fb)
        assert s.dtype == s2.dtype
        
    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_add_carrier_offset(self, dtype):
        s = signals.SignalQAMGrayCoded(16, 2**16, fb=10e9, nmodes=2, dtype=dtype)
        s2 = cimpairments.add_carrier_offset(s, 1e8, s.fb)
        assert s.dtype == s2.dtype

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_add_modal_delay(self, dtype):
        s = signals.SignalQAMGrayCoded(16, 2**16, fb=10e9, nmodes=2, dtype=dtype)
        s2 = cimpairments.add_modal_delay(s, [0,1000])
        assert s.dtype == s2.dtype
        
    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_simulate_transmission(self, dtype):
        s = signals.SignalQAMGrayCoded(16, 2**16, fb=10e9, nmodes=2, dtype=dtype)
        s2 = cimpairments.simulate_transmission(s, s.fb, s.fb, 20, 10e6, 100e3, 100e-12, modal_delay=[0,100])
        assert s.dtype == s2.dtype
     
    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_quantize_signal(self, dtype):
        s = signals.SignalQAMGrayCoded(16, 2**16, fb=10e9, nmodes=2, dtype=dtype)
        s2 = cimpairments.quantize_signal(s)
        assert s.dtype == s2.dtype
        
    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    @pytest.mark.parametrize("rescale_in", [True, False])
    @pytest.mark.parametrize("rescale_out", [True, False])
    def test_quantize_signal_New(self, dtype, rescale_in, rescale_out):
        s = signals.SignalQAMGrayCoded(16, 2**16, fb=10e9, nmodes=2, dtype=dtype)
        s2 = cimpairments.quantize_signal_New(s, rescale_in=rescale_in, rescale_out=rescale_out)
        assert s.dtype == s2.dtype

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_modulator_response(self, dtype):
        s = signals.SignalQAMGrayCoded(16, 2**16, fb=10e9, nmodes=2, dtype=dtype)
        s2 = cimpairments.modulator_response(s)
        assert s.dtype == s2.dtype
        
    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    @pytest.mark.parametrize("dac_r", [{"cutoff": 18e9}, {}])
    def test_sim_DAC_response(self, dtype, dac_r):
        s = signals.SignalQAMGrayCoded(16, 2**16, fb=40e9, nmodes=2, dtype=dtype)
        s2 = cimpairments.sim_DAC_response(s, s.fb, **dac_r )
        assert s.dtype == s2.dtype

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_apply_DAC_filter(self, dtype):
        #TODO test the loaded filter as well
        s = signals.SignalQAMGrayCoded(16, 2**16, fb=40e9, nmodes=2, dtype=dtype)
        s2 = cimpairments.apply_DAC_filter(s, s.fb)
        assert s.dtype == s2.dtype

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_apply_enob_as_awgn(self, dtype):
        s = signals.SignalQAMGrayCoded(16, 2**16, fb=40e9, nmodes=2, dtype=dtype)
        s2 = cimpairments.apply_enob_as_awgn(s, 5)
        assert s.dtype == s2.dtype
     
    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_sim_tx_response(self, dtype):
        s = signals.SignalQAMGrayCoded(16, 2**16, fb=40e9, nmodes=2, dtype=dtype)
        s2 = cimpairments.sim_tx_response(s, s.fb)
        assert s.dtype == s2.dtype

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_ideal_amplifier_response(self, dtype):
        s = signals.SignalQAMGrayCoded(16, 2**16, fb=40e9, nmodes=2, dtype=dtype)
        s2 = cimpairments.ideal_amplifier_response(s, 2)
        assert s.dtype == s2.dtype
        
    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_add_dispersion(self, dtype):
        s = signals.SignalQAMGrayCoded(16, 2**16, fb=40e9, nmodes=2, dtype=dtype)
        s2 = cimpairments.add_dispersion(s, s.fb, 20e-24, 1000)
        assert s.dtype == s2.dtype

class TestCoreFcts(object):
    @pytest.mark.parametrize("z", np.linspace(0.2, 2, 4))
    def test_dispersion(self, z):
        def broadening(z, b2, t0):
            return np.sqrt(1+(b2*z/t0**2)**2)
        def cal_fwhm(x, y):
            N = 10**5
            xn = np.linspace(x[0], x[-1], N)
            yn = np.interp(xn, x, y)
            m = np.max(yn)
            i0,iend = np.where(yn>(m/2))[0][[0,-1]]
            return xn[iend] - xn[i0]
        t = np.linspace(-40, 40, 2**11, endpoint=False) * 1e-12
        fs = 1/(t[1]-t[0])
        t0 = 5e-12
        D = 20e-17
        C = 2.99792458e8
        wl = 1550e-9
        #b2 = D*2*np.pi/wl**2
        b2 = wl**2/(2*np.pi*C) * D
        sig = np.exp(-t**2/2/t0**2).reshape(1,-1)+0j
        Ld = t0**2/b2
        sigo = cimpairments.add_dispersion(sig, fs, D, Ld*z )
        fwhm = cal_fwhm(t/t0, abs(sigo[0])**2)
        t1 = fwhm/(2*np.sqrt(np.log(2)))
        t11 = broadening(z*Ld, b2, t0)
        npt.assert_allclose(t1, t11, atol=2e-4)





