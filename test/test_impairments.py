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





