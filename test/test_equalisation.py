import pytest
import numpy as np
import numpy.testing as npt

from qampy import signals, equalisation, impairments
from qampy.core import impairments as cimpairments
from qampy.core import equalisation as cequalisation


class TestReturnObject(object):
    s = signals.ResampledQAM(16, 2 ** 16, fb=20e9, fs=40e9, nmodes=2)
    os = 2

    def test_apply_filter_basic(self):
        s2 = cimpairments.simulate_transmission(self.s, self.s.fb, self.s.fs, snr=20, dgd=100e-12)
        wx, err = equalisation.equalise_signal(s2, 1e-3, Ntaps=11)
        s3 = equalisation.apply_filter(s2, wx)
        assert type(s3) is type(self.s)

    def test_eq_applykw(self):
        s2 = cimpairments.simulate_transmission(self.s, self.s.fb, self.s.fs, snr=20, dgd=100e-12)
        s3, wx, err = equalisation.equalise_signal(s2, 1e-3, Ntaps=11, apply=True)
        assert type(s3) is type(self.s)

    def test_eq_applykw_dual(self):
        s2 = cimpairments.simulate_transmission(self.s, self.s.fb, self.s.fs, snr=20, dgd=100e-12)
        s3, wx, err = equalisation.dual_mode_equalisation(s2, (1e-3, 1e-3), 11, apply=True)
        assert type(s3) is type(self.s)

    @pytest.mark.xfail(reason="The core equalisation functions are not expected to preserve subclasses")
    def test_apply_filter_adv(self):
        s2 = cimpairments.simulate_transmission(self.s, self.s.fb, self.s.fs, snr=20, dgd=100e-12)
        wx, err = cequalisation.equalise_signal(s2, self.os, 1e-3, s2.M, Ntaps=11)
        s3 = equalisation.apply_filter(s2, self.os, wx)
        assert type(s3) is type(self.s)

class TestEqualisation(object):

    @pytest.mark.parametrize("N", [1,2,3])
    def test_nd_dualmode(self, N):
        import numpy as np
        s = signals.ResampledQAM(16, 2 ** 16, fb=20e9, fs=40e9, nmodes=N)
        s2 = impairments.change_snr(s, 25)
        E, wx, err = equalisation.dual_mode_equalisation(s2, (1e-3, 1e-3), 11, apply=True, adaptive_stepsize=(True,True))
        assert np.mean(E.cal_ber() < 1e-3)


class TestEqualiseSignalParameters(object):
    @pytest.mark.parametrize( ("modes", "sigmodes"),
                              [ (None, 1),
                                (None, 2),
                                (np.arange(2), 2),
                                (np.arange(2), 3),
                                pytest.param(np.arange(2), 1, marks=pytest.mark.xfail(raises=AssertionError))
                                ])
    def test_selected_modes(self, modes, sigmodes):
        sig = signals.SignalQAMGrayCoded(4, 2**15, nmodes=sigmodes)
        sig = impairments.change_snr(sig, 15)
        sig = sig.resample(sig.fb*2, beta=0.1)
        E, wx, e = cequalisation.equalise_signal(sig, sig.os, 1e-3, sig.M, Ntaps=10, modes=modes, apply=True)
        if modes is None:
            modes = np.arange(sigmodes)
        E = sig.recreate_from_np_array(E)
        ser = np.mean(E.cal_ser())
        assert ser < 1e-5

    @pytest.mark.parametrize(("sigmodes", "symbolsmodes",),
                             [ (1, None),
                               (2, None),
                               (1, 0),
                               (2, 0),
                               (1, 1),
                               (2, 2),
                               pytest.param(3, 2, marks=pytest.mark.xfail(raises=ValueError))])
    def test_symbols(self, sigmodes, symbolsmodes):
        selected_modes = np.arange(sigmodes)
        sig = signals.SignalQAMGrayCoded(4, 2**15, nmodes=sigmodes)
        sig = impairments.change_snr(sig, 15)
        sig = sig.resample(sig.fb*2, beta=0.1)
        if symbolsmodes is not None:
            symbols = sig.coded_symbols
            if symbolsmodes > 0:
                symbols = np.tile(symbols, (symbolsmodes, 1))
        else:
            symbols = None
        E, wx, e  = cequalisation.equalise_signal(sig, sig.os, 1e-3, sig.M, Ntaps=10, symbols=None, apply=True,
                                                  modes=selected_modes)
        E = sig.recreate_from_np_array(E)
        ser = np.mean(E.cal_ser())
        assert ser < 1e-5
            
    def test_dual_mode_64qam(self):
        sig = signals.SignalQAMGrayCoded(64, 10**5, nmodes=2)
        sig = impairments.change_snr(sig, 30)
        sig = sig.resample(sig.fb*2, beta=0.1)
        E, wx, e = equalisation.dual_mode_equalisation(sig, (1e-3, 1e-3), 19, adaptive_stepsize=(True, True))
        ser = np.mean(E.cal_ser())
        assert ser < 1e-5
            
    def test_single_mode_64(self):
        sig = signals.SignalQAMGrayCoded(64, 10**5, nmodes=2)
        sig = impairments.change_snr(sig, 30)
        sig = sig.resample(sig.fb*2, beta=0.1)
        E, wx, e = equalisation.equalise_signal(sig,1e-3, Ntaps=19, adaptive_stepsize=True, apply=True)
        serx,sery = E.cal_ser()
        assert serx < 1e-4
        assert sery < 1e-4

    @pytest.mark.parametrize("modes", [[0],[1], np.arange(2)])
    def test_data_aided(self,  modes):
        from qampy import helpers
        ntaps = 21
        sig = signals.SignalQAMGrayCoded(64, 10**5, nmodes=2, fb=25e9)
        sig2 = sig.resample(2*sig.fb, beta=0.02)
        sig2 = helpers.normalise_and_center(sig2)
        sig2 = np.roll(sig2, ntaps//2)
        sig3 = impairments.simulate_transmission(sig2, dgd=15e-12, theta=np.pi/3., snr=35)
        sig3 = helpers.normalise_and_center(sig3)
        sigout, wxy, err = equalisation.equalise_signal(sig3, 1e-3, Ntaps=ntaps, adaptive_stepsize=True,
                                                symbols=sig3.symbols, apply=True, method="sbd_data", TrSyms=10000, modes=modes)
        sigout = helpers.normalise_and_center(sigout)
        gmi = np.mean(sigout.cal_gmi(llr_minmax=True)[0])
        assert gmi > 5.9
        
    @pytest.mark.parametrize("rollframe", [True, False])
    @pytest.mark.parametrize("modal_delay", [(2000,2000), (3000, 2000)])
    @pytest.mark.parametrize("ddmethod", ["sbd", "sbd_data"])
    def test_pilot_based(self, rollframe, modal_delay, ddmethod):
        from qampy import phaserec
        mysig = signals.SignalWithPilots(64,2**16,2**10,32,nmodes=2,Mpilots=4,nframes=3,fb=24e9)
        mysig2 = mysig.resample(mysig.fb*2,beta=0.01)
        mysig3 = impairments.simulate_transmission(mysig2,snr=25,dgd=10e-12, freq_off=00e6,lwdth=000e3,roll_frame_sync=rollframe, modal_delay=modal_delay)
        mysig3.sync2frame()
        mysig3.corr_foe()
        wxy, eq_sig = equalisation.pilot_equaliser(mysig3, (1e-3, 1e-3), 45, foe_comp=False, methods=("cma", ddmethod))
        cpe_sig, ph = phaserec.pilot_cpe(eq_sig,N=5,use_seq=False) 
        gmi = np.mean(cpe_sig.cal_gmi()[0])
        assert gmi > 5.5
        
        

                        