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
    @pytest.mark.parametrize( ("selected_modes", "sigmodes"),
                              [ (None, 1),
                                (None, 2),
                                (np.arange(2), 2),
                                (np.arange(2), 3),
                                pytest.param(np.arange(2), 1, marks=pytest.mark.xfail(raises=AssertionError))
                                ])
    def test_selected_modes(self, selected_modes, sigmodes):
        sig = signals.SignalQAMGrayCoded(4, 2**15, nmodes=sigmodes)
        sig = impairments.change_snr(sig, 15)
        sig = sig.resample(sig.fb*2, beta=0.1)
        E, wx, e = cequalisation.equalise_signal(sig, sig.os, 1e-3, sig.M, Ntaps=10, selected_modes=selected_modes, apply=True)
        if selected_modes is None:
            selected_modes = np.arange(sigmodes)
        E = sig.recreate_from_np_array(E[selected_modes])
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
                                                  nmodes=selected_modes)
        E = sig.recreate_from_np_array(E)
        ser = np.mean(E.cal_ser())
        assert ser < 1e-5
            
    def test_dual_mode_64qam(self):
        sig = signals.SignalQAMGrayCoded(64, 10**5, nmodes=2)
        sig = impairments.change_snr(sig, 30)
        sig = sig.resample(sig.fb*2, beta=0.1)
        E, wx, e = equalisation.dual_mode_equalisation(sig, (1e-3, 1e-3), 19, adaptive_stepsize=(True, True))
        ser = np.mean(E)
        assert ser < 1e-5
            
    def test_single_mode_64(self):
        sig = signals.SignalQAMGrayCoded(64, 10**5, nmodes=2)
        sig = impairments.change_snr(sig, 30)
        sig = sig.resample(sig.fb*2, beta=0.1)
        E, wx, e = equalisation.equalise_signal(sig,1e-3, Ntaps=19, adaptive_stepsize=True, apply=True)
        ser = np.mean(E.cal_ser())
        assert ser < 1e-5
        
    @pytest.mark.parametrize("ndim", [0, 1, np.arange(2)])
    @pytest.mark.parametrize("nmodes", [1,2])
    def test_data_aided(self, ndim, nmodes):
        from qampy import helpers
        ntaps = 21
        sig = signals.SignalQAMGrayCoded(64, 10**5, nmodes=2, fb=25e9)
        sig2 = sig.resample(2*sig.fb, beta=0.02)
        sig2 = helpers.normalise_and_center(sig2)
        sig2 = np.roll(sig2, ntaps//2)
        sig3 = impairments.simulate_transmission(sig2, dgd=15e-12, theta=np.pi/3., snr=35)
        sig3 = helpers.normalise_and_center(sig3)
        if nmodes == 2:
            N = np.arange(2)
        else:
            N = ndim
        sigout, wxy, err = equalisation.equalise_signal(sig3, 1e-3, Ntaps=ntaps, adaptive_stepsize=True,
                                                symbols=sig3.symbols[N], apply=True, method="sbd_data", TrSyms=10000, nmodes=nmodes)
        sigout = helpers.normalise_and_center(sigout)
        gmi = np.mean(sigout.cal_gmi(llr_minmax=True)[0])
        assert gmi > 5.9
        
        

                        