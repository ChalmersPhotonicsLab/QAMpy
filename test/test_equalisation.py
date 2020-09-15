import pytest
import numpy as np
import numpy.testing as npt

from qampy import signals, equalisation, impairments, helpers
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
            
    @pytest.mark.parametrize("M", [4, 64])
    @pytest.mark.parametrize("nmodes", [1, 2, 4])
    @pytest.mark.parametrize("rmodes", [None, 0, -1])
    @pytest.mark.parametrize("method", cequalisation.TRAINING_FCTS)
    def test_single_mode(self, M, nmodes, rmodes, method):
        Ntaps=19
        sig = signals.SignalQAMGrayCoded(M, 10**5, nmodes=nmodes)
        sig = impairments.change_snr(sig, 30)
        sig = sig.resample(sig.fb*2, beta=0.1)
        if method in cequalisation.DATA_AIDED:
            sig = np.roll(sig, Ntaps//2, axis=-1)
        if rmodes is None:
            modes = None
        else:
            if nmodes == 1 and rmodes == -1:
                modes = np.array([0])
            else:
                modes = np.arange(nmodes+rmodes)
            if modes.size > 1:
                np.random.shuffle(modes)
        E, wx, e = equalisation.equalise_signal(sig, 0.5e-2, Niter=3, Ntaps=Ntaps, adaptive_stepsize=True, apply=True, modes=modes)
        ser =  E.cal_ser()
        if rmodes is None:
            assert ser.size == nmodes
        else:
            assert ser.size == modes.size
        assert np.all(ser < 1e-4)

    @pytest.mark.parametrize("modes", [[0],[1], np.arange(2)])
    @pytest.mark.parametrize("method", cequalisation.DATA_AIDED)
    @pytest.mark.parametrize("ps_sym", [True, False])
    def test_data_aided(self,  modes, method, ps_sym):
        from qampy import helpers
        ntaps = 21
        sig = signals.SignalQAMGrayCoded(64, 10**5, nmodes=2, fb=25e9)
        sig2 = sig.resample(2*sig.fb, beta=0.02)
        sig2 = helpers.normalise_and_center(sig2)
        sig2 = np.roll(sig2, ntaps//2)
        sig3 = impairments.simulate_transmission(sig2, dgd=150e-12, theta=np.pi/3., snr=35)
        sig3 = helpers.normalise_and_center(sig3)
        if ps_sym:
            symbs = sig3.symbols
        else:
            symbs = None
        sigout, wxy, err = equalisation.equalise_signal(sig3, 1e-3, Ntaps=ntaps, adaptive_stepsize=True,
                                                symbols=symbs, apply=True, method=method, TrSyms=20000, modes=modes)
        sigout = helpers.normalise_and_center(sigout)
        gmi = np.mean(sigout.cal_gmi(llr_minmax=True)[0])
        assert gmi > 5.9
        
    @pytest.mark.parametrize("rollframe", [True, False])
    @pytest.mark.parametrize("modal_delay", [(2000,2000), (3000, 2000)])
    @pytest.mark.parametrize("method", [("cma", "sbd"), ("cma", "sbd_data"), ("cma_real", "dd_data_real"),
                                        ("cma_real", "dd_real")])
    def test_pilot_based(self, rollframe, modal_delay, method):
        from qampy import phaserec
        mysig = signals.SignalWithPilots(64,2**16,2**10,32,nmodes=2,Mpilots=4,nframes=3,fb=24e9)
        mysig2 = mysig.resample(mysig.fb*2,beta=0.01)
        mysig3 = impairments.simulate_transmission(mysig2,snr=25,dgd=10e-12, freq_off=100e6,lwdth=100e3,roll_frame_sync=rollframe, modal_delay=modal_delay)
        mysig3.sync2frame()
        mysig3.corr_foe()
        wxy, eq_sig = equalisation.pilot_equaliser(mysig3, (1e-3, 1e-3), 45, foe_comp=False, methods=method)
        cpe_sig, ph = phaserec.pilot_cpe(eq_sig,N=5,use_seq=False)
        gmi = np.mean(cpe_sig.cal_gmi()[0])
        assert gmi > 5.5

    @pytest.mark.parametrize("method", ["cma_real", "dd_real", "dd_data_real" ])
    def test_real_valued_single_mode(self, method):
        s = signals.SignalQAMGrayCoded(4, 10**5, nmodes=1, fb=25e9)
        s2 = s.resample(s.fb*2, beta=0.1)
        s2 = impairments.sim_mod_response(s*0.2, dcbias=1.1)
        s3 = helpers.normalise_and_center(s2)
        s4 = impairments.change_snr(s3, 15)
        s5, wx, err = equalisation.equalise_signal(s4, 1e-3, Ntaps=17, method=method, adaptive_stepsize=True, apply=True)
        assert s5.cal_ser() < 1e5
        
    @pytest.mark.parametrize("frames", [[0], [0,1], [0,1,2]])
    def test_pilot_based_nframe_len(self, frames ):
        mysig = signals.SignalWithPilots(64,2**16,2**10,32,nmodes=2,Mpilots=4,nframes=4,fb=24e9)
        mysig2 = mysig.resample(mysig.fb*2,beta=0.01)
        mysig3 = impairments.simulate_transmission(mysig2,snr=25)
        mysig3.sync2frame()
        wxy, eq_sig,_ = equalisation.pilot_equaliser_nframes(mysig3, (1e-3, 1e-3), 45, foe_comp=False, frames=frames)
        assert eq_sig.shape[-1] == eq_sig.frame_len*len(frames)
        
    @pytest.mark.parametrize("frames", [[0], [0,1], [0,1,2]])
    def test_pilot_based_nframe_ber(self, frames ):
        mysig = signals.SignalWithPilots(64,2**16,2**10,32,nmodes=2,Mpilots=4,nframes=4,fb=24e9)
        mysig2 = mysig.resample(mysig.fb*2,beta=0.1)
        mysig3 = impairments.simulate_transmission(mysig2,snr=25, modal_delay=(4000, 4000))
        mysig3.sync2frame()
        wxy, eq_sig,_ = equalisation.pilot_equaliser_nframes(mysig3, (1e-3, 1e-3), 45, foe_comp=False, frames=frames)
        assert np.all(eq_sig.cal_ber(frames=frames) < 5e-3)
         

class TestUtilities(object):
    @pytest.mark.parametrize("method", ["sbd", "mddma", "dd"])
    @pytest.mark.parametrize("M", [4, 16])
    @pytest.mark.parametrize("nmodes", [1, 3])
    @pytest.mark.parametrize("pass_syms", [True, False])
    def test_reshape_symbols_DD_cmplx(self, method, M, nmodes, pass_syms):
        from qampy import theory
        from qampy.core.equalisation.equalisation import _reshape_symbols
        syms = theory.cal_symbols_qam(M)/np.sqrt(theory.cal_scaling_factor_qam(M))
        if pass_syms:
            s = _reshape_symbols(syms, method, M, np.complex128, nmodes)
        else:
            s = _reshape_symbols(None, method, M, np.complex128, nmodes)
        assert s.shape[0] == nmodes
        assert s.shape[1] == syms.shape[0]
        for i in range(nmodes):
            npt.assert_allclose(s[i], syms)
                
    @pytest.mark.parametrize("M", [4, 16])
    @pytest.mark.parametrize("nmodes", [1, 3])
    @pytest.mark.parametrize("pass_syms", [None, "cmplx", 'float'])
    def test_reshape_symbols_DD_real(self, M, nmodes, pass_syms):
        from qampy import theory
        from qampy.core.equalisation.equalisation import _reshape_symbols
        syms = theory.cal_symbols_qam(M)/np.sqrt(theory.cal_scaling_factor_qam(M))
        if pass_syms is None:
            s = _reshape_symbols(None, "dd_real", M, np.float64, nmodes*2)
        elif pass_syms is "cmplx":
            s = _reshape_symbols(syms, "dd_real", M, np.float64, nmodes*2)
        elif pass_syms is "float":
            s = _reshape_symbols(np.array([syms.real, syms.imag]), "dd_real", M, np.float64, nmodes*2)
        assert s.shape[0] == nmodes*2
        assert s.shape[1] == syms.shape[0]
        for i in range(nmodes*2):
            if i < nmodes:
                npt.assert_allclose(s[i], syms.real)
            else:
                npt.assert_allclose(s[i], syms.imag)

    @pytest.mark.parametrize("method", ["sbd_data", "dd_data_real"])
    @pytest.mark.parametrize("M", [4, 16])
    @pytest.mark.parametrize("nmodes", [1, 3])
    def test_reshape_symbols_data(self, method, M, nmodes):
        from qampy.core.equalisation.equalisation import _reshape_symbols
        sig = signals.SignalQAMGrayCoded(M, 1000, nmodes=nmodes)
        syms = sig.symbols
        if method is "dd_data_real":
            s = _reshape_symbols(syms, method, M, np.float64, nmodes*2)
            assert s.shape[0] == nmodes*2
            assert s.shape[1] == syms.shape[1]
            for i in range(nmodes*2):
                if i < nmodes:
                    npt.assert_allclose(s[i], syms[i].real)
                else:
                    npt.assert_allclose(s[i], syms[i%nmodes].imag)
        else:
            s = _reshape_symbols(syms, method, M, np.complex128, nmodes)
            assert s.shape[0] == nmodes
            assert s.shape[1] == syms.shape[1]
            for i in range(nmodes):
                npt.assert_allclose(s[i], syms[i])

    @pytest.mark.parametrize("method", ["cma", "mcma"])
    @pytest.mark.parametrize("M", [4, 16])
    @pytest.mark.parametrize("nmodes", [1, 3])
    @pytest.mark.parametrize("pass_syms", [True, False])
    def test_reshape_symbols_cma(self, method, M, nmodes, pass_syms):
        from qampy.core.equalisation.equalisation import _reshape_symbols, generate_symbols_for_eq
        syms = generate_symbols_for_eq(method, M, np.complex128)
        if pass_syms:
            s = _reshape_symbols(syms, method, M, np.complex128, nmodes)
        else:
            s = _reshape_symbols(None, method, M, np.complex128, nmodes)
        assert s.shape[0] == nmodes
        assert s.shape[1] == syms.shape[0]
        for i in range(nmodes):
            npt.assert_allclose(s[i], syms[0])
            
    @pytest.mark.parametrize("M", [4, 16])
    @pytest.mark.parametrize("nmodes", [1, 3])
    @pytest.mark.parametrize("pass_syms", [True, False])
    def test_reshape_symbols_cma_real(self,  M, nmodes, pass_syms):
        from qampy.core.equalisation.equalisation import _reshape_symbols, generate_symbols_for_eq
        method = "cma_real"
        syms = generate_symbols_for_eq(method, M, np.float64)
        from qampy.core.equalisation.equalisation import _reshape_symbols, generate_symbols_for_eq
        if pass_syms:
            s = _reshape_symbols(None, "cma_real", M, np.float64, nmodes*2)
        else:
            s = _reshape_symbols(np.array([syms.real, syms.imag]), "cma_real", M, np.float64, nmodes*2)
        assert s.shape[0] == 2*nmodes
        assert s.shape[1] == 1
        for i in range(nmodes*2):
            if i < nmodes:
                npt.assert_allclose(s[i], syms[0])
            else:
                npt.assert_allclose(s[i], syms[1])

                        