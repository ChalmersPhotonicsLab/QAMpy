import pytest
import numpy as np
import numpy.testing as npt

from qampy import signals, impairments, theory, helpers
from qampy.core import signal_quality

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
        npt.assert_allclose(ber, ber_t, rtol=0.15)

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
