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
        sig4.sync2frame(corr_coarse_foe=False)
        s1, s2 = equalisation.pilot_equalizer(sig4, [5e-3, 5e-3], ntaps, True, adaptive_stepsize=True, foe_comp=False)
        ph = phaserec.find_pilot_const_phase(s2.extract_pilots()[:,:s2._pilot_seq_len], s2.pilot_seq)
        s2 = phaserec.correct_pilot_const_phase(s2, ph)
        ser = s2.cal_ser()
        assert np.mean(ser) < 1e-4

    @pytest.mark.parametrize("theta", np.linspace(0.1, 1, 5))
    @pytest.mark.parametrize("dgd", np.linspace(100e-12, 400e-12, 4))
    def test_recovery_with_pmd(self, theta, dgd):
        snr = 30.
        ntaps = 41
        sig = signals.SignalWithPilots(64, 2**16, 1024, 32, nframes=3, nmodes=2, fb=24e9)
        sig2 = sig.resample(2*sig.fb, beta=0.1, renormalise=True)
        sig3 = impairments.change_snr(sig2, snr)
        sig3 = impairments.apply_PMD(sig3, theta*np.pi, dgd)
        sig4 = sig3[:, 20000:]
        sig4.sync2frame(corr_coarse_foe=False)
        s1, s2 = equalisation.pilot_equalizer(sig4, [5e-3, 5e-3], ntaps, True, adaptive_stepsize=True, foe_comp=False)
        ph = phaserec.find_pilot_const_phase(s2.extract_pilots()[:,:s2._pilot_seq_len], s2.pilot_seq)
        s2 = phaserec.correct_pilot_const_phase(s2, ph)
        ser = s2.cal_ser(synced=False)
        snr_m = s2.est_snr(synced=False)
        snr_db = 10*np.log10(np.mean(snr_m))
        assert np.mean(ser) < 1e-4

    def test_swap_pols(self):
        snr = 30.
        sig = signals.SignalWithPilots(64, 2**16, 1024, 32, nframes=3, nmodes=2, fb=24e9)
        sig2 = sig.resample(2*sig.fb, beta=0.1, renormalise=True)
        sig2 = impairments.change_snr(sig2, snr)
        sig3 = sig2[::-1]
        sig4 = sig3[:, 20000:]
        sig4.sync2frame(corr_coarse_foe=False)
        s1, s2 = equalisation.pilot_equalizer(sig4, [1e-3, 1e-3], 17, True, adaptive_stepsize=True, foe_comp=False)
        ph = phaserec.find_pilot_const_phase(s2.extract_pilots()[:,:s2._pilot_seq_len], s2.pilot_seq)
        s2 = phaserec.correct_pilot_const_phase(s2, ph)
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
        sig4.sync2frame(corr_coarse_foe=False)
        s1, s2 = equalisation.pilot_equalizer(sig4, [1e-3, 1e-3], ntaps, True, adaptive_stepsize=True, foe_comp=False)
        d, ph = phaserec.pilot_cpe(s2, nframes=1)
        assert np.mean(d.cal_ber()) < 1e-5
#
