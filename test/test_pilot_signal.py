import pytest
import numpy as np
import numpy.testing as npt
from qampy import signals, equalisation, impairments, core, phaserec


class TestPilotSignalRecovery(object):
    @pytest.mark.parametrize("theta", np.linspace(0.1, 1, 5))
    def test_recovery_with_rotation(self, theta):
        snr = 30.
        ntaps = 17
        sig = signals.SignalWithPilots(64, 2**16, 1024, 32, nframes=3, nmodes=2, fb=24e9)
        sig2 = sig.resample(2*sig.fb, beta=0.1, renormalise=True)
        sig3 = impairments.change_snr(sig2, snr)
        sig3 = core.impairments.rotate_field(sig3, np.pi*theta)
        sig4 = sig3[:, 20000:]
        sig4.sync2frame(Ntaps=ntaps)
        s1, s2 = equalisation.pilot_equalizer(sig4, [1e-3, 1e-3], ntaps, True)
        ser = s2.cal_ser(synced=False)
        snr_m = s2.est_snr(synced=False)
        snr_db = 10*np.log10(np.mean(snr_m))
        assert np.mean(ser) < 1e-4
        #npt.assert_almost_equal(snr, snr_db)

    @pytest.mark.parametrize("theta", np.linspace(0.1, 1, 5))
    @pytest.mark.parametrize("dgd", np.linspace(100e-12, 1000e-12, 5))
    def test_recovery_with_pmd(self, theta, dgd):
        snr = 30.
        ntaps = 17
        sig = signals.SignalWithPilots(64, 2**16, 1024, 32, nframes=3, nmodes=2, fb=24e9)
        sig2 = sig.resample(2*sig.fb, beta=0.1, renormalise=True)
        sig3 = impairments.change_snr(sig2, snr)
        sig3 = impairments.apply_PMD(sig3, theta, dgd)
        #sig3 = core.impairments.rotate_field(sig3, np.pi*theta)
        sig4 = sig3[:, 20000:]
        sig4.sync2frame(Ntaps=ntaps, Niter=10)
        s1, s2 = equalisation.pilot_equalizer(sig4, [1e-3, 1e-3], ntaps, True, adaptive_stepsize=True)
        ser = s2.cal_ser(synced=False)
        snr_m = s2.est_snr(synced=False)
        snr_db = 10*np.log10(np.mean(snr_m))
        assert np.mean(ser) < 1e-4
        #npt.assert_almost_equal(snr, snr_db)

    def test_swap_pols(self):
        snr = 30.
        sig = signals.SignalWithPilots(64, 2**16, 1024, 32, nframes=3, nmodes=2, fb=24e9)
        sig2 = sig.resample(2*sig.fb, beta=0.1, renormalise=True)
        sig2 = impairments.change_snr(sig2, snr)
        sig3 = sig2[::-1]
        sig4 = sig3[:, 20000:]
        sig4.sync2frame()
        s1, s2 = equalisation.pilot_equalizer(sig4, [1e-3, 1e-3], 17, True, adaptive_stepsize=True)
        ser = s2.cal_ser(synced=True)
        assert np.mean(ser) < 1e-4

    @pytest.mark.parametrize("lw", 10**(np.linspace(3, 4.5, 3)))
    def test_cpe(self,lw):
        snr = 37
        ntaps = 17
        sig = signals.SignalWithPilots(64, 2**16, 1024, 32, nframes=3, nmodes=2, fb=24e9)
        sig2 = sig.resample(2*sig.fb, beta=0.01, renormalise=True)
        sig2 = impairments.apply_phase_noise(sig2, 100e3)
        sig3 = impairments.change_snr(sig2, snr)
        sig4 = sig3[:, 20000:]
        sig4.sync2frame()
        s1, s2 = equalisation.pilot_equalizer(sig4, [1e-3, 1e-3], ntaps, True, adaptive_stepsize=True)
        d, ph = phaserec.pilot_cpe(s2, nframes=1)
        assert np.mean(d.cal_ber()) < 1e-6
#
