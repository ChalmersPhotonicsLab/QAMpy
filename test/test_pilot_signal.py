import pytest
import numpy as np
import numpy.testing as npt
from qampy import signals, equalisation, impairments, core, phaserec, theory, helpers


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
        s1, s2 = equalisation.pilot_equaliser(sig4, [5e-3, 5e-3], ntaps, True, adaptive_stepsize=True, foe_comp=False)
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
        s1, s2 = equalisation.pilot_equaliser(sig4, [5e-3, 5e-3], ntaps, True, adaptive_stepsize=True, foe_comp=False)
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
        s1, s2 = equalisation.pilot_equaliser(sig4, [1e-3, 1e-3], 17, True, adaptive_stepsize=True, foe_comp=False)
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
        s1, s2 = equalisation.pilot_equaliser(sig4, [1e-3, 1e-3], ntaps, True, adaptive_stepsize=True, foe_comp=False)
        d, ph = phaserec.pilot_cpe(s2, nframes=1)
        assert np.mean(d.cal_ber()) < 1e-5
        
    @pytest.mark.parametrize("fo", [20e3, 100e3, 1e4])
    def test_freq_offset(self, fo):
        snr = 37
        ntaps = 17
        sig = signals.SignalWithPilots(64, 2**16, 1024, 32, nframes=3, nmodes=2, fb=24e9)
        sig2 = sig.resample(2*sig.fb, beta=0.01, renormalise=True)
        sig3 = impairments.simulate_transmission(sig, snr, freq_off=fo)
        sig4 = helpers.normalise_and_center(sig3)
        sig4.sync2frame(corr_coarse_foe=False)
        s1, s2 = equalisation.pilot_equalizer(sig4, [1e-3, 1e-3], ntaps, True, adaptive_stepsize=True, foe_comp=True)
        d, ph = phaserec.pilot_cpe(s2, nframes=1)
        assert np.mean(d.cal_ber()) < 1e-5
#
class TestSignalGeneration(object):
    @pytest.mark.parametrize("nmodes", [1,2])
    def test_from_numpy(self, nmodes):
       c1 = theory.cal_symbols_qam(4)
       cdat = theory.cal_symbols_qam(128)
       c1 = c1/np.sqrt(np.mean(abs(c1)**2))
       cdat = cdat/np.sqrt(np.mean(abs(cdat)**2))
       ph = np.random.choice(c1, 1024).reshape(1,-1)
       dat = np.random.choice(cdat, 2**16*nmodes).reshape(nmodes,-1)
       ss = signals.SignalWithPilots.from_symbol_array(dat, 2**16, 1024, 0, pilots=ph, payload_kwargs={"M":128})
       assert ss.pilots.M == 4
       assert ss.pilots.size == 1024*nmodes
       assert ss.size == 2**16*nmodes
       assert ss.symbols.M == 128

    def test_from_numpy_isframe(self):
       c1 = theory.cal_symbols_qam(4)
       cdat = theory.cal_symbols_qam(128)
       c1 = c1/np.sqrt(np.mean(abs(c1)**2))
       cdat = cdat/np.sqrt(np.mean(abs(cdat)**2))
       ph = np.random.choice(c1, 1024).reshape(1,-1)
       dat = np.random.choice(cdat, 2**16).reshape(1,-1)
       s = np.hstack([ph,dat])
       ss = signals.SignalWithPilots.from_symbol_array(dat, 2**16, 1024, 0,  payload_kwargs={"M":128}, payload_is_frame=True)
       assert ss.pilots.M == 4
       assert ss.pilots.size == 1024
       assert ss.size == 2**16
       assert ss.symbols.M == 128

    @pytest.mark.parametrize("nmodes", [1,2])
    def test_from_signal_object(self, nmodes):
        ph = signals.SignalQAMGrayCoded(4, 1024, nmodes=1)
        dat = signals.SignalQAMGrayCoded(128, 2**16, nmodes=nmodes)
        ss = signals.SignalWithPilots.from_symbol_array(dat, 2**16, 1024, 0,  pilots=ph)
        assert ss.pilots.M == 4
        assert ss.pilots.size == 1024*nmodes
        assert ss.size == 2**16*nmodes
        assert ss.symbols.M == 128
    
    @pytest.mark.parametrize("nmodes", [1,2])
    @pytest.mark.parametrize("method", ["est_snr", "cal_gmi", "cal_ber", "cal_ser"])
    @pytest.mark.parametrize("nframes", np.arange(1,4))
    def test_nframes_calculation(selfself, nmodes, method, nframes):
        ss = signals.SignalWithPilots(64, 2**16, 1024, 32, nframes=3, nmodes=nmodes)
        s2 = impairments.change_snr(ss, 20)
        m = getattr(s2, method)(nframes=nframes)
        print(m)

    @pytest.mark.parametrize("nlen",[1, 2.2, 5.5])
    def test_recreate_nframes(self, nlen):
        ss = signals.SignalWithPilots(64, 2**16, 1024, 32, nframes=3)
        oo = np.copy(ss)
        oo1 = np.tile(oo, (1, int((nlen*ss.frame_len)//ss.frame_len)))
        oo2 = np.hstack([oo1, oo[:, :oo.shape[-1]*int(nlen%ss.frame_len)] ])
        out = ss.recreate_from_np_array(oo2)
    
    def test_too_short_frame(self):
        with pytest.raises(AssertionError):
            ss = signals.SignalWithPilots(64, 2**16, 1024, 32, nframes=3)
            oo = np.copy(ss)
            out = ss.recreate_from_np_array(oo[:,:int(ss.frame_len/2)])
        