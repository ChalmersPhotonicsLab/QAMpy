import pytest
import numpy as np
import numpy.testing as npt

from qampy import signals, impairments, theory, helpers
from qampy.core import signal_quality


def _flip_symbols(sig, idx, d, mode=0):
    for i in idx:
        if np.random.randint(0, 2):
            if sig[mode,i].real > 0:
                sig[mode,i] -= d
            else:
                sig[mode,i] += d
        else:
            if sig[mode,i].imag > 0:
                sig[mode,i] -= 1.j * d
            else:
                sig[mode,i] += 1.j * d
    return sig


class TestDtypePreserve(object):
    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_generate_bitmap_mtx(self, dtype):
        s =signals.SignalQAMGrayCoded(32, 2**16, dtype=dtype)
        o = signal_quality.generate_bitmapping_mtx(s.coded_symbols, s.demodulate(s.coded_symbols), s.M, dtype=dtype)
        assert np.dtype(dtype) is o.dtype

class TestNoResampling(object):
    @pytest.mark.parametrize("M", [4, 16, 64, 128])
    @pytest.mark.parametrize("snr", [0, 2, 4])
    @pytest.mark.parametrize("synced", [True, False])
    def test_vstheory_ser(self, M, snr, synced):
        s = signals.SignalQAMGrayCoded(M, 2 ** 16)
        off = {4: 6, 16: 13, 64: 18, 128: 20}
        ss = impairments.change_snr(s, snr+off[M])
        ser = ss.cal_ser(synced=synced)
        ser_t = theory.ser_vs_es_over_n0_qam(10**((snr+off[M])/10), M)
        npt.assert_allclose(ser, ser_t, rtol=0.3)

    @pytest.mark.parametrize("M", [4, 16, 64, 128])
    @pytest.mark.parametrize("snr", [0, 2, 4])
    @pytest.mark.parametrize("synced", [True, False])
    def test_vstheory_ber(self, M, snr, synced):
        s = signals.SignalQAMGrayCoded(M, 2 ** 16)
        off = {4: 6, 16: 13, 64: 18, 128: 20}
        ss = impairments.change_snr(s, snr+off[M])
        ber = ss.cal_ber(synced=synced)
        ber_t = theory.ber_vs_es_over_n0_qam(10**((snr+off[M])/10), M)
        npt.assert_allclose(ber, ber_t, rtol=0.15)

    @pytest.mark.parametrize("M", [4, 16, 64, 128])
    @pytest.mark.parametrize("snr", [0, 2, 4])
    @pytest.mark.parametrize("synced", [True, False])
    def test_vstheory_evm(self, M, snr, synced):
        s = signals.SignalQAMGrayCoded(M, 2 ** 16)
        off = {4: 6, 16: 13, 64: 18, 128: 20}
        snr = snr+off[M]
        ss = impairments.change_snr(s, snr)
        evm = helpers.lin2dB(ss.cal_evm(synced=synced)**2)
        npt.assert_allclose(snr, -evm, rtol=0.01)

class TestWithResampling(object):
    @pytest.mark.parametrize("M", [4, 16, 64, 128])
    @pytest.mark.parametrize("snr", [0, 2, 4])
    @pytest.mark.parametrize("os", [2, 3])
    def test_vstheory_ser(self, M, snr, os):
        s = signals.SignalQAMGrayCoded(M, 2 ** 16)
        off = {4: 6, 16: 13, 64: 18, 128: 20}
        s = s.resample(os, beta=0.1, renormalise=True)
        ss = impairments.change_snr(s, snr+off[M])
        ss = ss.resample(1, beta=0.1, renormalise=True)
        ser = ss.cal_ser()
        ser_t = theory.ser_vs_es_over_n0_qam(10**((snr+off[M])/10), M)
        npt.assert_allclose(ser, ser_t, rtol=0.3)

    @pytest.mark.parametrize("M", [4, 16, 64, 128])
    @pytest.mark.parametrize("snr", [0, 2, 4])
    @pytest.mark.parametrize("os", [2, 3])
    def test_vstheory_ber(self, M, snr, os):
        s = signals.SignalQAMGrayCoded(M, 2 ** 16)
        off = {4: 6, 16: 13, 64: 18, 128: 20}
        s = s.resample(os, beta=0.1, renormalise=True)
        ss = impairments.change_snr(s, snr+off[M])
        ss = ss.resample(1, beta=0.1, renormalise=True)
        ber = ss.cal_ber()
        ber_t = theory.ber_vs_es_over_n0_qam(10**((snr+off[M])/10), M)
        npt.assert_allclose(ber, ber_t, rtol=0.18)

    # error is significantly higher for M=4 so ommitting
    @pytest.mark.parametrize("M", [16, 64, 128])
    @pytest.mark.parametrize("snr", [0, 2, 4])
    @pytest.mark.parametrize("os", [2, 3])
    def test_vstheory_evm(self, M, snr, os):
        s = signals.SignalQAMGrayCoded(M, 2 ** 16)
        off = {4: 6, 16: 13, 64: 18, 128: 20}
        snr = snr+off[M]
        s = s.resample(os, beta=0.1, renormalise=True)
        ss = impairments.change_snr(s, snr)
        ss = ss.resample(1, beta=0.1, renormalise=True)
        evm = helpers.lin2dB(ss.cal_evm() ** 2)
        npt.assert_allclose(snr, -evm, rtol=0.1)

class TestVerboseReturn(object):
    @pytest.mark.parametrize("M", [32, 64])
    def test_verbose_ser(self, M):
        s = signals.SignalQAMGrayCoded(M, 2**12)
        snr = 30
        s = impairments.change_snr(s, snr)
        d = abs(np.unique(np.diff(s.coded_symbols.real)))
        dmin = d[np.where(d>0)].min()
        s2 = _flip_symbols(s, [100], dmin)
        ser, errs, syms = s2.cal_ser(synced=True, verbose=True)
        assert errs[0,100] != 0
        assert np.count_nonzero(errs) == 1

    @pytest.mark.parametrize("M", [32, 64])
    def test_verbose_ber(self, M):
        s = signals.SignalQAMGrayCoded(M, 2**12)
        snr = 30
        #s = impairments.change_snr(s, snr)
        d = abs(np.unique(np.diff(s.coded_symbols.real)))
        dmin = d[np.where(d>0)].min()
        s2 = _flip_symbols(s, [100], dmin)
        ber, errs, bits = s2.cal_ber(synced=True, verbose=True)
        assert abs(np.argmax(errs) - 100*np.log2(M)) < np.log2(M)
        assert ber[0] != 0
        assert np.count_nonzero(errs) == 1

    @pytest.mark.parametrize("M", [32, 64])
    def test_verbose_ser_syms(self, M):
        s = signals.SignalQAMGrayCoded(M, 2**12)
        snr = 30
        s = impairments.change_snr(s, snr)
        d = abs(np.unique(np.diff(s.coded_symbols.real)))
        dmin = d[np.where(d>0)].min()
        s2 = _flip_symbols(s, [100], dmin)
        ser, errs, syms = s2.cal_ser(synced=True, verbose=True)
        assert syms.shape == s2.shape

    @pytest.mark.parametrize("M", [32, 64])
    def test_verbose_ber_syms(self, M):
        s = signals.SignalQAMGrayCoded(M, 2**12)
        snr = 30
        s = impairments.change_snr(s, snr)
        d = abs(np.unique(np.diff(s.coded_symbols.real)))
        dmin = d[np.where(d>0)].min()
        s2 = _flip_symbols(s, [100], dmin)
        ser, errs, syms = s2.cal_ber(synced=True, verbose=True)
        assert syms.shape == s2.bits.shape





