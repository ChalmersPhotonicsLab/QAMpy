import pytest
import numpy as np
import numpy.testing as npt
from qampy import signals, equalisation, impairments, core, phaserec, theory, helpers
from qampy.core.equalisation import DATA_AIDED, REAL_VALUED


class TestPilotSignalRecovery(object):
    @pytest.mark.parametrize("method", REAL_VALUED)
    def test_realvalued_framesync(self, method):
        # currently we do not support real-valued equalisers for frame-sync
        sig = signals.SignalWithPilots(64, 2**16, 1024, 32, nframes=2, nmodes=2, fb=20e9)
        s2 = sig.resample(sig.fb*2, beta=0.1)
        with pytest.raises(ValueError):
            s2.sync2frame(method=method)
            
    @pytest.mark.parametrize("method", DATA_AIDED)
    def test_data_aided_framesync(self, method):
        # currently we do not support real-valued equalisers for frame-sync
        sig = signals.SignalWithPilots(64, 2**16, 1024, 32, nframes=2, nmodes=2, fb=20e9)
        s2 = sig.resample(sig.fb*2, beta=0.1)
        with pytest.raises(ValueError):
            s2.sync2frame(method=method)
            
    @pytest.mark.parametrize("method", REAL_VALUED)
    def test_data_mix_realandcomplexmethods(self, method):
        # currently we do not support to mix real-valued and complex equaliser methods
        sig = signals.SignalWithPilots(64, 2**16, 1024, 32, nframes=2, nmodes=2, fb=20e9)
        s2 = sig.resample(sig.fb*2, beta=0.1)
        s3 = impairments.simulate_transmission(s2, 25, modal_delay=[10000,10000])
        s3.sync2frame()
        with pytest.raises(ValueError):
            equalisation.pilot_equaliser(s3, (1e-3, 1e-3), 11, methods=(method, "sbd"))
        
    @pytest.mark.parametrize("snr", [10, 15, 20, 25])
    def test_framesync_noshift(self, snr):
        sig = signals.SignalWithPilots(64, 2**16, 1024, 32, nframes=2, nmodes=2, fb=20e9)
        s2 = sig.resample(sig.fb*2, beta=0.1)
        s3 = impairments.simulate_transmission(s2, snr)
        assert s3.sync2frame()
        
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
    @pytest.mark.parametrize("mode_offset", [0, 3333])
    def test_coarse_freq_offset(self, fo, mode_offset):
        snr = 37
        ntaps = 19
        sig = signals.SignalWithPilots(64, 2**16, 1024, 32, nframes=3, nmodes=2, fb=24e9)
        sig2 = sig.resample(2*sig.fb, beta=0.01, renormalise=True)
        sig3 = impairments.simulate_transmission(sig2, snr, freq_off=fo, modal_delay=[0, mode_offset])
        sig4 = helpers.normalise_and_center(sig3)
        sig4 = sig4[:, 2000:]
        sig4.sync2frame(corr_coarse_foe=True)
        s1, s2 = equalisation.pilot_equaliser(sig4, [1e-3, 1e-3], ntaps, True, adaptive_stepsize=True, foe_comp=True)
        d, ph = phaserec.pilot_cpe(s2, nframes=1)
        assert np.mean(d.cal_ber()) < 1e-5
        
    @pytest.mark.parametrize("fo", np.arange(5, 10))
    @pytest.mark.parametrize("apply_coarse", [False, True])
    def test_eqn_freq_offset(self, fo, apply_coarse):
        snr = 37
        ntaps = 19
        sig = signals.SignalWithPilots(64, 2**16, 1024, 32, nframes=3, nmodes=2, fb=24e9)
        sig2 = sig.resample(2*sig.fb, beta=0.01, renormalise=True)
        sig3 = impairments.simulate_transmission(sig2, snr, freq_off=10**fo)
        sig4 = helpers.normalise_and_center(sig3)
        sig4 = sig4[:, 2000:]
        sig4.sync2frame()
        if apply_coarse:
            sig4.corr_foe()
        s1, s2 = equalisation.pilot_equaliser(sig4, [1e-3, 1e-3], ntaps, True, adaptive_stepsize=True, foe_comp=True)
        d, ph = phaserec.pilot_cpe(s2, nframes=1)
        assert np.mean(d.cal_ber()) < 1e-5
            
    @pytest.mark.parametrize("frames", [[0], [0,1], [2], [0,2]])
    @pytest.mark.parametrize("modal_delay", [0, 500])
    def test_apply_filter_frames(self, frames, modal_delay):
        Ntaps = 45
        s = signals.SignalWithPilots(64, 2**16, 1024, 32, nframes=4, nmodes=2, fb=24e9)
        s2 = s.resample(2*s.fb, beta=0.1, renormalise=True)
        s3 = impairments.simulate_transmission(s2, 30, modal_delay=[2000, 2000+modal_delay])
        s3.sync2frame(Ntaps=Ntaps-14*2)
        wx = equalisation.pilot_equaliser(s3, 1e-3, Ntaps, apply=False, foe_comp=False)
        sout = equalisation.apply_filter(s3, wx, frames=frames)
        assert sout.shape[-1] == s.frame_len*len(frames)
        for ber in sout.cal_ber():
            assert ber < 1e-3


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
    @pytest.mark.parametrize("frames", np.arange(0,3))
    def test_nframes_calculation(self, nmodes, method, frames):
        ss = signals.SignalWithPilots(64, 2**16, 1024, 32, nframes=3, nmodes=nmodes)
        s2 = impairments.change_snr(ss, 20)
        m = getattr(s2, method)(frames=frames)
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

class TestSignalQuality(object):
    @pytest.mark.parametrize("frame", [0,1,2])
    def test_cal_ser_frames(self, frame):
        sig = signals.SignalWithPilots(64, 10**4+1000, 1000, 0, nframes=3, nmodes=1)
        sig[0, 1010+frame*sig.frame_len] *= 1j
        for i in range(3):
            ser = sig.cal_ser(frames=[i])
            if i == frame:
                assert np.isclose(ser, 1/10**4)
            else:
                assert np.isclose(ser, 0)
        ser  = sig.cal_ser(frames=[0,1,2])
        assert np.isclose(ser, 1/(3*10**4))

    @pytest.mark.parametrize("frame", [0,1,2])
    def test_cal_ber_frames(self, frame):
        M = 64
        sig = signals.SignalWithPilots(M, 10**4+1000, 1000, 0, nframes=3, nmodes=1)
        dd = np.diff(np.unique(sig.coded_symbols.real)).min()
        sig[0, 1010+frame*sig.frame_len] += dd*0.8
        for i in range(3):
            ber = sig.cal_ber(frames=[i])
            if i == frame:
                assert np.isclose(ber, 1/(np.log2(M)*10**4))
            else:
                assert np.isclose(ber, 0)
        ber  = sig.cal_ber(frames=[0,1,2])
        assert np.isclose(ber, 1/(3*np.log2(M)*10**4))

    @pytest.mark.parametrize("frame", [0,1,2])
    def test_cal_evm_frames(self, frame):
        M = 64
        sig = signals.SignalWithPilots(M, 10**4+1000, 1000, 0, nframes=3, nmodes=1)
        dd = np.diff(np.unique(sig.coded_symbols.real)).min()
        sig[0, 1010+frame*sig.frame_len] += dd*0.6
        for i in range(3):
            evm = sig.cal_evm(frames=[i])
            if i == frame:
                assert np.isclose(evm, 0.00185164) # calculated before
            else:
                assert np.isclose(evm, 0)
        evm  = sig.cal_evm(frames=[0,1,2])
        assert np.isclose(evm, 0.00106904)

    @pytest.mark.parametrize("frame", [0,1,2])
    def test_cal_gmi_frames(self, frame):
        M = 64
        sig = signals.SignalWithPilots(M, 10**4+1000, 1000, 0, nframes=3, nmodes=1)
        sig2 = impairments.change_snr(sig[:,:sig.frame_len], 25)
        sig[0, frame*sig.frame_len:(frame+1)*sig.frame_len] = sig2[:]
        for i in range(3):
            gmi = sig.cal_gmi(frames=[i])[0]
            if i == frame:
                assert np.isclose(gmi, sig2.cal_gmi()[0])
            else:
                assert np.isclose(gmi, np.log2(M))

    @pytest.mark.parametrize("frame", [0,1,2])
    def test_est_snr_frames(self, frame):
        M = 64
        snr_in = 25.
        sig = signals.SignalWithPilots(M, 10**4+1000, 1000, 0, nframes=3, nmodes=1)
        sig2 = impairments.change_snr(sig[:,:sig.frame_len], snr_in)
        sig[0, frame*sig.frame_len:(frame+1)*sig.frame_len] = sig2[:]
        for i in range(3):
            snr = sig.est_snr(frames=[i])
            if i == frame:
                assert np.isclose(snr, 10**(snr_in/10), rtol=0.03)
            else:
                assert np.all(snr > 10e15)
