import pytest
import numpy as np
import numpy.testing as npt

from qampy import signals
from qampy.core import ber_functions, impairments

#TODO: We should check that all the syncing works when we have only symbols from a relatively short PRBS pattern that
#      repeats
class TestFindSequenceOffset(object):
    s = signals.SignalQAMGrayCoded(16, 3 * 10 ** 4, nmodes=1)
    d = np.diff(np.unique(s.symbols.real)).min()

    @pytest.mark.parametrize("shiftN", [np.random.randint(l*3*10**4//4+1, (l+1)*3*10**4//4) for l in range(4)])
    @pytest.mark.parametrize("snr", [5, 20, 40])
    @pytest.mark.parametrize("N1", [None, 4000])
    def test_shift(self, shiftN, snr, N1):
        sig = impairments.change_snr(self.s, self.s.fb, self.s.fs, snr)
        sig = sig[0]
        syms = sig.symbols[0]
        sig2 = np.roll(sig, shift=shiftN)
        offset = ber_functions.find_sequence_offset(syms[:N1], sig2)
        assert (offset == shiftN)

    @pytest.mark.parametrize("shiftN", [np.random.randint(l*(2**15-1)//2+1, (l+1)*(2**15-1)//2) for l in range(4)]+[48630])
    @pytest.mark.parametrize("N1", [None, 4000])
    def test_data(self, shiftN, N1):
        sig = self.s[0]
        syms = self.s.symbols[0]
        sig2 = np.roll(sig, shift=shiftN)
        offset = ber_functions.find_sequence_offset(syms[:N1], sig2)
        sig2 = np.roll(sig2, -offset)
        npt.assert_allclose(syms[:N1], sig2[:N1], atol=self.d/4)

    @pytest.mark.parametrize("shiftN", [100, 1000, 5001])
    @pytest.mark.parametrize("N1", [None, 4000])
    def test_ints_shift(self, shiftN, N1):
        b = np.random.randint(0,2, 2**16)
        b2 = np.roll(b, shift=shiftN)
        offset = ber_functions.find_sequence_offset(b[:N1], b2)
        assert (shiftN == offset)


class TestFindSequenceOffsetComplex(object):
    s = signals.SignalQAMGrayCoded(16, 2 * (2 ** 15 - 1), nmodes=1)
    d = np.diff(np.unique(s.symbols.real)).min()

    @pytest.mark.parametrize("shiftN", [np.random.randint(l*(2**15-1)//2+1, (l+1)*(2**15-1)//2) for l in range(4)]+[48630])
    @pytest.mark.parametrize("i", range(4))
    @pytest.mark.parametrize("snr", [5, 20, 40])
    @pytest.mark.parametrize("N1", [None, 4000])
    def test_shift(self, shiftN, i, snr, N1):
        sig = self.s[0]
        syms = self.s.symbols[0]
        sig2 = np.roll(sig, shift=shiftN)
        offset, syms2, ii = ber_functions.find_sequence_offset_complex(syms[:N1]*1j**i, sig2)
        assert (offset == shiftN)

    @pytest.mark.parametrize("shiftN", [np.random.randint(l*(2**15-1)//2+1, (l+1)*(2**15-1)//2) for l in range(4)]+[48630])
    @pytest.mark.parametrize("i", range(4))
    @pytest.mark.parametrize("N1", [None, 4000])
    def test_data(self, shiftN, i, N1):
        sig = self.s[0]
        syms = self.s.symbols[0]
        sig2 = np.roll(sig, shift=shiftN)
        offset, syms2, ii = ber_functions.find_sequence_offset_complex(syms[:N1]*1j**i, sig2)
        sig2 = np.roll(sig2, -offset)
        npt.assert_allclose(syms2, sig2[:N1], atol=self.d/4)

    @pytest.mark.parametrize("shiftN", [np.random.randint(l*(2**15-1)//2+1, (l+1)*(2**15-1)//2) for l in range(4)]+[48630])
    @pytest.mark.parametrize("i", range(4))
    def test_rotation(self, shiftN, i):
        sig = self.s[0]
        syms = self.s.symbols[0]
        sig2 = np.roll(sig, shift=shiftN)
        offset, syms2, ii = ber_functions.find_sequence_offset_complex(syms*1j**i, sig2)
        assert (4-ii)%4 == i

class TestSyncAndAdjust(object):
    s = signals.SignalQAMGrayCoded(16, 3 * 10 ** 4, nmodes=1)
    d = np.diff(np.unique(s.symbols.real)).min()

    @pytest.mark.parametrize("N1, N2, adjust", [(None, 1000, "tx"),(None, 1000, "rx" ), (1000, None, "tx"), (1000, None, "rx") ])
    def test_length(self, N1, N2, adjust):
        sig = self.s[0]
        N = self.s.shape[1]
        syms = self.s.symbols[0]
        tx, rx = ber_functions.sync_and_adjust(sig[:N1], sig[:N2], adjust=adjust)
        if N1 is None and adjust == "tx":
            assert (tx.shape[0] == N2) and (rx.shape[0] == N2)
        elif N2 is None and adjust == "rx":
            assert (tx.shape[0] == N1) and (rx.shape[0] == N1)
        else:
            assert (tx.shape[0] == N) and (rx.shape[0] == N)

    @pytest.mark.parametrize("N0", [(None, 1000), (1000, None)])
    @pytest.mark.parametrize("shiftN", [0, 43, 150, 800])
    @pytest.mark.parametrize("adjust", ['rx', 'tx'])
    def test_length_with_shift(self, N0, shiftN, adjust):
        sig = self.s[0]
        N = self.s.shape[1]
        N1, N2 = N0
        sign = np.roll(sig, shiftN)
        tx, rx = ber_functions.sync_and_adjust(sig[:N1], sign[:N2], adjust=adjust)
        if N1 is None and adjust is "tx":
            assert (tx.shape[0] == N2) and (rx.shape[0] == N2)
        elif N2 is None and adjust is "rx":
            assert (tx.shape[0] == N1) and (rx.shape[0] == N1)
        else:
            assert (tx.shape[0] == N) and (rx.shape[0] == N)

    @pytest.mark.parametrize("shiftN", [np.random.randint(l*(2**15-1)//2+1, (l+1)*(2**15-1)//2) for l in range(4)]+[48630])
    @pytest.mark.parametrize("adjust", ["tx", "rx"])
    def test_flip(self, adjust, shiftN):
        Np = 2**15-1
        N = 2*Np
        s = signals.SignalQAMGrayCoded(16, N, bitclass=signals.PRBSBits)
        sig = s[0]
        syms = s.symbols[0]
        syms2 = np.roll(syms, shift=shiftN)
        tx, rx = ber_functions.sync_and_adjust(syms, syms2, adjust=adjust)
        npt.assert_allclose(tx, rx)
        tx, rx = ber_functions.sync_and_adjust(syms2, syms, adjust=adjust)
        npt.assert_allclose(tx, rx)

    @pytest.mark.parametrize("adjust", ["tx", "rx"])
    @pytest.mark.parametrize("tx_i, rx_i", list(zip(list(range(1,4)) + 3*[0], 3*[0]+list(range(1,4)))))
    @pytest.mark.parametrize("N1, N2", [(None, 2**15-1), (2**15-1, None)])
    @pytest.mark.parametrize("shiftN", [np.random.randint(l*(2**15-1)//2+1, (l+1)*(2**15-1)//2) for l in range(4)] + [48630])
    def test_rotated_and_diff_length(self, adjust, tx_i, rx_i, N1, N2, shiftN):
        Np = 2**15-1
        N = 2*Np
        s = signals.SignalQAMGrayCoded(16, N, bitclass=signals.PRBSBits)
        sig = s[0]
        syms = s.symbols[0]
        syms2 = np.roll(syms, shift=shiftN)
        tx, rx = ber_functions.sync_and_adjust(syms[:N1]*1j**tx_i, syms2[:N2]*1j**rx_i, adjust=adjust)
        npt.assert_allclose(tx, rx)

class TestAdjustDataLength(object):
    s = signals.SignalQAMGrayCoded(16, 3 * 10 ** 4, nmodes=1)
    d = np.diff(np.unique(s.symbols.real)).min()

    @pytest.mark.parametrize("N1, N2", [(None, 1000), (1000, None)])
    @pytest.mark.parametrize("method", ["truncate", "extend", None])
    @pytest.mark.parametrize("offset", [0, 300])
    def test_length(self, N1, N2, method, offset):
        sig = self.s[0]
        N = self.s.shape[1]
        syms = self.s.symbols[0]
        tx, rx = ber_functions.adjust_data_length(sig[:N1], sig[:N2], method=method, offset=offset)
        if method is "truncate":
            N_ex = N1 or N2
        if method is "extend":
            N_ex = N
        if method is None:
            if N1:
                N_ex = N
            else:
                N_ex = N2
        assert (tx.shape[0] == N_ex) and (rx.shape[0] == N_ex)

