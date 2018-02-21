import pytest

import numpy as np
import numpy.testing as npt

from dsp import modulation, helpers, phaserec, equalisation, impairments


@pytest.mark.parametrize("lw", np.linspace(10, 1000, 4))
@pytest.mark.parametrize("M", [4, 16, 32, 64])
def test_phaserec_bps(lw, M):
    fb = 40.e9
    os = 1
    fs = os*fb
    N = 2**17
    snr = 30
    shiftN = np.random.randint(-N/2, N/2, 1)
    s = modulation.SignalQAMGrayCoded(M, N, fb=fb)
    s = s.resample(fs, beta=0.1, renormalise=True)
    s = impairments.change_snr(s, snr)
    s = np.roll(s, shiftN, axis=1)
    pp = impairments.apply_phase_noise(s, lw)
    recoverd, ph1 = phaserec.bps(pp, M , s.coded_symbols, 14, method='pyx')
    recoverd = helpers.dump_edges(recoverd, 20)
    ser = recoverd.cal_ser()
    npt.assert_allclose(ser, 0)

@pytest.mark.parametrize("lw", np.linspace(10, 1000, 4))
@pytest.mark.parametrize("M", [4, 16, 32, 64])
def test_phaserec_bps_2stage(lw, M):
    fb = 40.e9
    os = 1
    fs = os*fb
    N = 2**17
    snr = 30
    shiftN = np.random.randint(-N/2, N/2, 1)
    s = modulation.SignalQAMGrayCoded(M, N, fb=fb)
    s = s.resample(fs, beta=0.1, renormalise=True)
    s = impairments.change_snr(s, snr)
    s = np.roll(s, shiftN, axis=1)
    pp = impairments.apply_phase_noise(s, lw)
    recoverd, ph1 = phaserec.bps_twostage(pp, max(4, M/2), s.coded_symbols, 14, method='pyx')
    recoverd = helpers.dump_edges(recoverd, 20)
    ser = recoverd.cal_ser()
    npt.assert_allclose(ser, 0)

class TestCMA(object):
    @pytest.mark.parametrize("method", ["cma", "mcma"])
    @pytest.mark.parametrize("phi", np.linspace(4.3, 8, 5))
    def test_pol_rot(self, method, phi):
        phi = np.pi/phi
        fb = 40.e9
        os = 2
        fs = os*fb
        N = 2**16
        snr = 15
        beta = 0.1
        mu = 0.1e-1
        M = 4
        s = modulation.SignalQAMGrayCoded(M, N, nmodes=2, fb=fb)
        s = s.resample(fs, beta=0.1, renormalise=True)
        s = impairments.rotate_field(s, phi)
        wxy, err = equalisation.equalise_signal(s, mu, M, Ntaps=3, method=method, adaptive_stepsize=True)
        sout = equalisation.apply_filter(s, wxy)
        ser = sout.cal_ser()
        npt.assert_allclose(ser, 0)

    @pytest.mark.parametrize("method", ["cma", "mcma"])
    @pytest.mark.parametrize("phi", np.linspace(4.3, 7.5, 4)) # at the moment I get failures for phi > 6
    @pytest.mark.parametrize("dgd", np.linspace(10, 300, 4)*1e-12)
    def test_pmd(self, method, phi, dgd):
        theta = np.pi/phi
        fb = 40.e9
        os = 2
        fs = os*fb
        N = 2**16
        snr = 15
        beta = 0.1
        mu = 4e-5
        M = 4
        ntaps = 21
        s = modulation.SignalQAMGrayCoded(M, N, nmodes=2, fb=fb)
        s = s.resample(fs, beta=0.1, renormalise=True)
        s = impairments.apply_PMD_to_field(s, theta, dgd)
        wxy, err = equalisation.equalise_signal(s, mu, M, Ntaps=ntaps, method=method, adaptive_stepsize=False)
        sout = equalisation.apply_filter(s, wxy)
        ser = sout.cal_ser()
        npt.assert_allclose(ser, 0)

    @pytest.mark.parametrize("method", ["cma", "mcma"])
    @pytest.mark.parametrize("phi", np.linspace(4.3, 7.5, 3)) # at the moment I get failures for phi > 6
    @pytest.mark.parametrize("dgd", np.linspace(10, 300, 3)*1e-12)
    @pytest.mark.parametrize("lw", np.linspace(10e3, 1000e3, 4))
    def test_pmd_phase(self, method, phi, dgd, lw):
        theta = np.pi/phi
        fb = 40.e9
        os = 2
        fs = os*fb
        N = 2**16
        snr = 15
        beta = 0.1
        mu = 4e-5
        M = 4
        ntaps = 21
        s = modulation.SignalQAMGrayCoded(M, N, nmodes=2, fb=fb)
        s = s.resample(fs, beta=0.1, renormalise=True)
        s = impairments.apply_phase_noise(s, lw)
        s = impairments.apply_PMD_to_field(s, theta, dgd)
        wxy, err = equalisation.equalise_signal(s, mu, M, Ntaps=ntaps, method=method, adaptive_stepsize=False)
        sout = equalisation.apply_filter(s, wxy)
        sout, ph = phaserec.viterbiviterbi(sout,11)
        sout = helpers.dump_edges(sout, 20)
        ser = sout.cal_ser()
        npt.assert_allclose(ser, 0)


