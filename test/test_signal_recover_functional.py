import pytest

import numpy as np
import numpy.testing as npt

from qampy import signals, helpers, phaserec, equalisation, impairments, core

import matplotlib.pylab as plt

@pytest.mark.parametrize("lw", np.linspace(10, 1000, 4))
@pytest.mark.parametrize("M", [4, 16, 32, 64])
def test_phaserec_bps(lw, M):
    fb = 40.e9
    os = 1
    fs = os*fb
    N = 2**17
    snr = 30
    shiftN = np.random.randint(-N/2, N/2, 1)
    s = signals.SignalQAMGrayCoded(M, N, fb=fb)
    s = s.resample(fs, beta=0.1, renormalise=True)
    s = impairments.change_snr(s, snr)
    s = np.roll(s, shiftN, axis=1)
    pp = impairments.apply_phase_noise(s, lw)
    recoverd, ph1 = phaserec.bps(pp, M , 14, method='pyt')
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
    s = signals.SignalQAMGrayCoded(M, N, fb=fb)
    s = s.resample(fs, beta=0.1, renormalise=True)
    s = impairments.change_snr(s, snr)
    s = np.roll(s, shiftN, axis=1)
    pp = impairments.apply_phase_noise(s, lw)
    recoverd, ph1 = phaserec.bps_twostage(pp, max(4, M//2), 14, method='pyt')
    recoverd = helpers.dump_edges(recoverd, 20)
    ser = recoverd.cal_ser()
    npt.assert_allclose(ser, 0)

class TestDualMode(object):
    @pytest.mark.parametrize("method1", ["cma", "mcma"])
    @pytest.mark.parametrize("method2", ["sbd", "mddma"])
    @pytest.mark.parametrize("phi", np.linspace(4.3, 8, 5))
    def test_pol_rot(self, method1, method2, phi):
        phi = np.pi/phi
        fb = 40.e9
        os = 2
        fs = os*fb
        N = 2**16
        beta = 0.9
        mu1 = 0.1e-2
        mu2 = 0.1e-2
        M = 32
        s = signals.SignalQAMGrayCoded(M, N, nmodes=2, fb=fb)
        s = s.resample(fs, beta=beta, renormalise=True)
        s = impairments.rotate_field(s, phi)
        sout, wxy, err = equalisation.dual_mode_equalisation(s, (mu1, mu2), Ntaps=5, Niter=(3,3), methods=(method1, method2),
                                                             adaptive_stepsize=(True, True))
        sout = helpers.normalise_and_center(sout)
        ser = sout.cal_ser()
        #plt.plot(sout[0].real, sout[0].imag, 'r.')
        #plt.plot(sout[1].real, sout[1].imag, 'b.')
        #plt.show()
        if ser.mean() > 0.5:
            ser = sout[::-1].cal_ser()
        npt.assert_allclose(ser, 0)

    @pytest.mark.parametrize("method1", ["cma", "mcma"])
    @pytest.mark.parametrize("method2", ["sbd", "mddma"])
    def test_pmd(self, method1, method2):
        theta = np.pi/5
        dgd = 120e-12
        fb = 40.e9
        os = 2
        fs = os*fb
        N = 2**16
        snr = 15
        beta = 0.9
        mu1 = 4e-4
        mu2 = 4e-4
        M = 32
        ntaps = 21
        s = signals.SignalQAMGrayCoded(M, N, nmodes=2, fb=fb)
        s = s.resample(fs, beta=beta, renormalise=True)
        s = impairments.apply_PMD(s, theta, dgd)
        sout, wxy, err = equalisation.dual_mode_equalisation(s, (mu1, mu2), Ntaps=ntaps, methods=(method1, method2),
                                                             adaptive_stepsize=(True, True))
        sout = helpers.normalise_and_center(sout)
        ser = sout.cal_ser()
        if ser.mean() > 0.4:
            ser = sout[::-1].cal_ser()
        npt.assert_allclose(ser, 0, atol=1.01*2/N) # can tolerate 1-2 errors

    ##TODO: figure out the failures for small angles
    @pytest.mark.xfail
    @pytest.mark.parametrize("method1", ["cma", "mcma"])
    @pytest.mark.parametrize("method2", ["sbd", "mddma"])
    def test_pmd_fails(self, method1, method2):
        theta = np.pi/5
        dgd = 200e-12
        fb = 40.e9
        os = 2
        fs = os*fb
        N = 2**16
        snr = 15
        beta = 0.9
        mu1 = 4e-3
        mu2 = 4e-3
        M = 32
        ntaps = 21
        s = signals.SignalQAMGrayCoded(M, N, nmodes=2, fb=fb)
        s = s.resample(fs, beta=beta, renormalise=True)
        s = impairments.apply_PMD(s, theta, dgd)
        sout, wxy, err = equalisation.dual_mode_equalisation(s, (mu1, mu2), Niter=(3,3), Ntaps=ntaps, methods=(method1, method2),
                                                             adaptive_stepsize=(True, True))
        sout = helpers.normalise_and_center(sout)
        ser = sout.cal_ser()
        npt.assert_allclose(ser, 0, atol=1.01*4/N)

    @pytest.mark.parametrize("method1", ["cma", "mcma"])
    @pytest.mark.parametrize("method2", ["sbd", "mddma"])
    @pytest.mark.parametrize("lw", np.linspace(10e3, 100e3, 2))
    def test_pmd_phase(self, method1, method2, lw):
        theta = np.pi/4.5
        dgd = 100e-12
        fb = 40.e9
        os = 2
        fs = os*fb
        N = 2**16
        snr = 15
        beta = 0.9
        mu1 = 2e-3
        if method2 == "mddma":
            mu2 = 1.0e-3
        else:
            mu2 = 2e-3
        M = 32
        ntaps = 21
        s = signals.SignalQAMGrayCoded(M, N, nmodes=2, fb=fb)
        s = s.resample(fs, beta=beta, renormalise=True)
        s = impairments.apply_phase_noise(s, lw)
        s = impairments.apply_PMD(s, theta, dgd)
        sout, wxy, err = equalisation.dual_mode_equalisation(s, (mu1, mu2), Ntaps=ntaps, methods=(method1, method2),
                                                             adaptive_stepsize=(True, True))
        sout, ph = phaserec.bps(sout, M, 21)
        sout = helpers.normalise_and_center(sout)
        sout = helpers.dump_edges(sout, 50)
        ser = sout.cal_ser()
        if ser.mean() > 0.4:
            ser = sout[::-1].cal_ser()
        npt.assert_allclose(ser, 0, atol=1.01*3/N)# Three wrong symbols is ok


class TestLMS(object):
    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    @pytest.mark.parametrize("method", ["sbd", "mddma", "dd", "dd_real", "dd_data_real", "sbd_data", "rde", "mrde"])
    def test_method(self, dtype, method):
        fb = 40.e9
        os = 2
        fs = os*fb
        N = 2**13
        beta = 0.1
        mu = 0.2e-2
        M = 16
        taps = 13
        s = signals.SignalQAMGrayCoded(M, N, nmodes=2, fb=fb, dtype=dtype)
        s = s.resample(fs, beta=beta, renormalise=True)
        #s = impairments.change_snr(s, 20)
        #wxy, err = equalisation.equalise_signal(s, mu, Ntaps=taps, method=method, adaptive_stepsize=True)
        wxy, err = equalisation.equalise_signal(s, mu, Niter=3, Ntaps=taps, method=method, adaptive_stepsize=True)
        sout = equalisation.apply_filter(s, wxy)
        ser = sout.cal_ser()
        #plt.plot(sout[0].real, sout[0].imag, 'r.')
        #plt.plot(sout[1].real, sout[1].imag, 'b.')
        #plt.show()
        npt.assert_allclose(ser, 0, atol=3./N)
        assert np.dtype(dtype) is sout.dtype


class TestCMA(object):
    @pytest.mark.parametrize("method", ["cma", "mcma", "cma_real"])
    @pytest.mark.parametrize("phi", np.linspace(4.3, 8, 5))
    def test_pol_rot(self, method, phi):
        phi = np.pi/phi
        fb = 40.e9
        os = 2
        fs = os*fb
        N = 2**16
        beta = 0.1
        mu = 0.1e-2
        M = 4
        ntaps = 5
        s = signals.SignalQAMGrayCoded(M, N, nmodes=2, fb=fb)
        s = s.resample(fs, beta=beta, renormalise=True)
        s = impairments.rotate_field(s, phi)
        wxy, err = equalisation.equalise_signal(s, mu, Niter=3, Ntaps=ntaps, method=method,
                                                adaptive_stepsize=True , avoid_cma_sing=False)
        sout = equalisation.apply_filter(s, wxy)
        #plt.plot(sout[0].real, sout[0].imag, '.r')
        #plt.show()
        ser = sout.cal_ser()
        #if ser.mean() > 0.5:
           #ser = sout[::-1].cal_ser
        npt.assert_allclose(ser, 0)

    @pytest.mark.parametrize("method", ["cma", "mcma", "cma_real"])
    @pytest.mark.parametrize("phi", np.linspace(4.3, 5.8, 3))
    @pytest.mark.parametrize("dgd", np.linspace(10, 300, 4)*1e-12)
    def test_pmd(self, method, phi, dgd):
        theta = np.pi/phi
        fb = 40.e9
        os = 2
        fs = os*fb
        N = 2**16
        snr = 15
        beta = 0.1
        mu = 4e-3
        M = 4
        ntaps = 21
        s = signals.SignalQAMGrayCoded(M, N, nmodes=2, fb=fb)
        s = s.resample(fs, beta=beta, renormalise=True)
        s = impairments.apply_PMD(s, theta, dgd)
        wxy, err = equalisation.equalise_signal(s, mu, Niter=5, Ntaps=ntaps, method=method,
                                                adaptive_stepsize=True, avoid_cma_sing=False)
        sout = equalisation.apply_filter(s, wxy)
        sout = helpers.normalise_and_center(sout)
        ser = sout.cal_ser()
        npt.assert_allclose(ser, 0)

    @pytest.mark.parametrize("method", ["cma", "mcma"])
    @pytest.mark.parametrize("dgd", [100e-12, 200e-12])
    def test_pmd_2(self, method, dgd):
        phi = 6.5
        theta = np.pi/phi
        fb = 40.e9
        os = 2
        fs = os*fb
        N = 2**16
        snr = 15
        beta = 0.1
        mu = 0.9e-3
        M = 4
        ntaps = 7
        s = signals.SignalQAMGrayCoded(M, N, nmodes=2, fb=fb)
        s = s.resample(fs, beta=beta, renormalise=True)
        s = impairments.apply_PMD(s, theta, dgd)
        wxy, err = equalisation.equalise_signal(s, mu, Ntaps=ntaps, method=method,
                                                adaptive_stepsize=True, avoid_cma_sing=True)
        sout = equalisation.apply_filter(s, wxy)
        sout = helpers.normalise_and_center(sout)
        ser = sout.cal_ser()
        npt.assert_allclose(ser, 0)

    @pytest.mark.parametrize("method", ["cma", "mcma"])
    @pytest.mark.parametrize("phi", np.linspace(7.5, 8.5, 2))
    @pytest.mark.parametrize("dgd", np.linspace(200, 300, 2)*1e-12)
    def test_pmd_2(self, method, phi, dgd):
        theta = np.pi/phi
        fb = 40.e9
        os = 2
        fs = os*fb
        N = 2**16
        snr = 15
        beta = 0.6
        mu = 0.1e-4
        M = 4
        ntaps = 21
        s = signals.SignalQAMGrayCoded(M, N, nmodes=2, fb=fb)
        s = s.resample(fs, beta=beta, renormalise=True)
        s = impairments.apply_PMD(s, theta, dgd)
        wxy, err = equalisation.equalise_signal(s, mu, Ntaps=ntaps, Niter=3, method=method, adaptive_stepsize=False)
        sout = equalisation.apply_filter(s, wxy)
        sout = helpers.normalise_and_center(sout)
        ser = sout.cal_ser()
        npt.assert_allclose(ser, 0, atol=1.01*4/N)

    @pytest.mark.parametrize("method", ["cma", "mcma"])
    @pytest.mark.parametrize("phi", np.linspace(4.3, 5.7, 2))
    @pytest.mark.parametrize("dgd", np.linspace(10, 300, 3)*1e-12)
    @pytest.mark.parametrize("lw", np.linspace(10e3, 1000e3, 3))
    def test_pmd_phase(self, method, phi, dgd, lw):
        theta = np.pi/phi
        fb = 40.e9
        os = 2
        fs = os*fb
        N = 2**16
        snr = 15
        beta = 0.3
        mu = 1e-2
        M = 4
        ntaps = 21
        s = signals.SignalQAMGrayCoded(M, N, nmodes=2, fb=fb)
        s = s.resample(fs, beta=beta, renormalise=True)
        s = impairments.apply_phase_noise(s, lw)
        s = impairments.apply_PMD(s, theta, dgd)
        wxy, err = equalisation.equalise_signal(s, mu, Niter=3, Ntaps=ntaps, method=method, adaptive_stepsize=True)
        sout = equalisation.apply_filter(s, wxy)
        sout, ph = phaserec.viterbiviterbi(sout,11)
        sout = helpers.normalise_and_center(sout)
        sout = helpers.dump_edges(sout, 30)
        ser = sout.cal_ser()
        npt.assert_allclose(ser, 0, atol=1.01*3/N)# Three wrong symbols is ok

    @pytest.mark.xfail
    @pytest.mark.parametrize("method", ["cma", "mcma"])
    @pytest.mark.parametrize("phi", np.linspace(5.9, 7.5, 2))
    @pytest.mark.parametrize("dgd", [300e-12])
    @pytest.mark.parametrize("lw", np.linspace(10e3, 1000e3, 2))
    def test_pmd_phase_fails(self, method, phi, dgd, lw):
        theta = np.pi/phi
        fb = 40.e9
        os = 2
        fs = os*fb
        N = 2**16
        snr = 15
        beta = 0.3
        mu = 2e-4
        M = 4
        ntaps = 15
        s = signals.SignalQAMGrayCoded(M, N, nmodes=2, fb=fb)
        s = s.resample(fs, beta=beta, renormalise=True)
        s = impairments.apply_phase_noise(s, lw)
        s = impairments.apply_PMD(s, theta, dgd)
        wxy, err = equalisation.equalise_signal(s, mu, Ntaps=ntaps, method=method, adaptive_stepsize=False)
        sout = equalisation.apply_filter(s, wxy)
        sout, ph = phaserec.viterbiviterbi(sout,11)
        sout = helpers.normalise_and_center(sout)
        sout = helpers.dump_edges(sout, 20)
        ser = sout.cal_ser()
        npt.assert_allclose(ser, 0)



