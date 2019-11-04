import pytest
import numpy as np
import numpy.testing as npt

from qampy import signals, equalisation
from qampy.core import impairments
from qampy.core import equalisation as cequalisation


class TestReturnObject(object):
    s = signals.ResampledQAM(16, 2 ** 16, fb=20e9, fs=40e9, nmodes=2)
    os = 2

    def test_apply_filter_basic(self):
        s2 = impairments.simulate_transmission(self.s, self.s.fb, self.s.fs, snr=20, dgd=100e-12)
        wx, err = equalisation.equalise_signal(s2, 1e-3, Ntaps=11)
        s3 = equalisation.apply_filter(s2, wx)
        assert type(s3) is type(self.s)

    def test_eq_applykw(self):
        s2 = impairments.simulate_transmission(self.s, self.s.fb, self.s.fs, snr=20, dgd=100e-12)
        s3, wx, err = equalisation.equalise_signal(s2, 1e-3, Ntaps=11, apply=True)
        assert type(s3) is type(self.s)

    def test_eq_applykw_dual(self):
        s2 = impairments.simulate_transmission(self.s, self.s.fb, self.s.fs, snr=20, dgd=100e-12)
        s3, wx, err = equalisation.dual_mode_equalisation(s2, (1e-3, 1e-3), 11, apply=True)
        assert type(s3) is type(self.s)

    @pytest.mark.xfail(reason="The core equalisation functions are not expected to preserve subclasses")
    def test_apply_filter_adv(self):
        s2 = impairments.simulate_transmission(self.s, self.s.fb, self.s.fs, snr=20, dgd=100e-12)
        wx, err = cequalisation.equalise_signal(s2, self.os, 1e-3, s2.M, Ntaps=11)
        s3 = equalisation.apply_filter(s2, self.os, wx)
        assert type(s3) is type(self.s)

class TestEqualisation(object):

    @pytest.mark.parametrize("N", [1,2,3])
    def test_nd_dualmode(self, N):
        import numpy as np
        from qampy import impairments
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
        sig = signals.SignalQAMGrayCoded(4, 2**10, nmodes=sigmodes)
        wx, e = cequalisation.equalise_signal(sig, sig.os, 1e-3, sig.M, Ntaps=10, selected_modes=selected_modes)

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
        sig = signals.SignalQAMGrayCoded(4, 2**10, nmodes=sigmodes)
        if symbolsmodes is not None:
            symbols = sig.coded_symbols
            if symbolsmodes > 0:
                symbols = np.tile(symbols, (symbolsmodes, 1))
        else:
            symbols = None
        wx, e  = cequalisation.equalise_signal(sig, sig.os, 1e-3, sig.M, Ntaps=10, symbols=None, selected_modes=selected_modes)
            
            
            