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
        assert (offset == -shiftN) or ((3*10**4 - offset )== shiftN)

    @pytest.mark.parametrize("shiftN", [np.random.randint(l*(2**15-1)//2+1, (l+1)*(2**15-1)//2) for l in range(4)]+[48630])
    @pytest.mark.parametrize("N1", [None, 4000])
    def test_data(self, shiftN, N1):
        sig = self.s[0]
        syms = self.s.symbols[0]
        sig2 = np.roll(sig, shift=shiftN)
        offset = ber_functions.find_sequence_offset(syms[:N1], sig2)
        sig2 = np.roll(sig2, offset)
        npt.assert_allclose(syms[:N1], sig2[:N1], atol=self.d/4)

    @pytest.mark.parametrize("shiftN", [100, 1000, 5001])
    @pytest.mark.parametrize("N1", [None, 4000])
    def test_ints_shift(self, shiftN, N1):
        b = np.random.randint(0,2, 2**16)
        b2 = np.roll(b, shift=shiftN)
        offset = ber_functions.find_sequence_offset(b[:N1], b2)
        assert (shiftN == -offset)

    @pytest.mark.parametrize("shiftN", [100, 1000, 5001])
    @pytest.mark.parametrize("N1", [None, 4000])
    def test_ints_shift2(self, shiftN, N1):
        b = np.random.randint(0,2, 2**16)
        b2 = np.roll(b, shift=shiftN)
        offset = ber_functions.find_sequence_offset(b, b2[:N1])
        assert (shiftN == -offset) or (shiftN+offset == 2**16)

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
        offset, syms2, ii, acm = ber_functions.find_sequence_offset_complex(syms[:N1] * 1j ** i, sig2)
        assert (shiftN == -offset) or (shiftN+offset == 2*(2**15-1))

    @pytest.mark.parametrize("shiftN", [np.random.randint(l*(2**15-1)//2+1, (l+1)*(2**15-1)//2) for l in range(4)]+[48630])
    @pytest.mark.parametrize("i", range(4))
    @pytest.mark.parametrize("N1", [None, 4000])
    def test_data(self, shiftN, i, N1):
        sig = self.s[0]
        syms = self.s.symbols[0]*1j**i
        sig2 = np.roll(sig, shift=shiftN)
        offset, syms2, ii, acm = ber_functions.find_sequence_offset_complex(syms[:N1], sig2)
        sig2 = np.roll(syms2, offset)
        npt.assert_allclose(syms[:N1], sig2[:N1], atol=self.d/4)

    @pytest.mark.parametrize("shiftN", [np.random.randint(l*(2**15-1)//2+1, (l+1)*(2**15-1)//2) for l in range(4)]+[48630])
    @pytest.mark.parametrize("i", range(4))
    def test_rotation(self, shiftN, i):
        sig = self.s[0]
        syms = self.s.symbols[0]
        sig2 = np.roll(sig, shift=shiftN)
        offset, syms2, ii, acm = ber_functions.find_sequence_offset_complex(syms, sig2 * 1j ** i)
        assert (4-ii)%4 == i

class TestSyncAndAdjust(object):
    s = signals.SignalQAMGrayCoded(16, 3 * 10 ** 4, nmodes=1)
    d = np.diff(np.unique(s.symbols.real)).min()

    @pytest.mark.parametrize("N1, N2, adjust", [(None, 1000, "tx"),(None, 1000, "rx" ), (1000, None, "tx"), (1000, None, "rx") ])
    def test_length(self, N1, N2, adjust):
        sig = self.s[0]
        N = self.s.shape[1]
        syms = self.s.symbols[0]
        (tx, rx), acm = ber_functions.sync_and_adjust(sig[:N1], sig[:N2], adjust=adjust)
        if N1 is None and adjust == "tx":
            assert (tx.shape[0] == N2) and (rx.shape[0] == N2)
        elif N2 is None and adjust == "rx":
            assert (tx.shape[0] == N1) and (rx.shape[0] == N1)
        else:
            assert (tx.shape[0] == N) and (rx.shape[0] == N)


    @pytest.mark.parametrize(
        ("rx_longer", "adjust"),
        [
            (True, "tx"),
            pytest.param(False, "tx", marks=pytest.mark.xfail(reason="to short array throws off offset find")),
            pytest.param(True, "rx", marks=pytest.mark.xfail(reason="to short array throws off offset find")),
                (False, 'rx'),
            (True, "tx"),
            (None, "rx"),
            (None, "tx"),
        ]
    )
    def test_slices(self, rx_longer, adjust):
        x = np.arange(1000.)
        xx = np.tile(x, 3)
        y = xx[110:1000 + 3 * 110]
        ym = xx[110:1000 - 3 * 110]
        y_equal = xx[110:1000 + 1 * 110]
        if rx_longer is None:
            tx = x
            rx = y_equal
        else:
            if adjust == "tx":
                if rx_longer:
                    rx = y
                    tx = x
                else:
                    rx = ym
                    tx = x
            elif adjust == "rx":
                if rx_longer:
                    rx = x
                    tx = ym
                else:
                    rx = x
                    tx = y
        (tx, rx), acm = ber_functions.sync_and_adjust(tx, rx, adjust=adjust)
        npt.assert_array_almost_equal(tx, rx)

    @pytest.mark.parametrize("N", [234, 1000, 2001])
    @pytest.mark.parametrize("rx_longer", [True, False, None])
    def test_slices_data_tx(self, N, rx_longer):
        s = signals.SignalQAMGrayCoded(4, 2**16)[0]
        ss = np.tile(s, 3)
        if rx_longer is None:
            y = ss[N:2**16+N]
        elif rx_longer:
            y = ss[N:2**16+2*N]
        else:
            y = ss[N:2**16-2*N]
        (tx, rx), acm = ber_functions.sync_and_adjust(s, y, adjust="tx")
        npt.assert_array_almost_equal(tx,rx)

    @pytest.mark.parametrize("N", [234, 1000, 2001])
    @pytest.mark.parametrize("tx_longer", [True, False, None])
    def test_slices_data_rx(self, N, tx_longer):
        s = signals.SignalQAMGrayCoded(4, 2**16)[0]
        ss = np.tile(s, 3)
        if tx_longer is None:
            y = ss[N:2**16+N]
        elif tx_longer:
            y = ss[N:2**16+2*N]
        else:
            y = ss[N:2**16-2*N]
        (tx, rx), acm = ber_functions.sync_and_adjust(y, s, adjust="rx")
        npt.assert_array_almost_equal(tx,rx)


    @pytest.mark.parametrize("rx_longer", [True, False, None])
    @pytest.mark.parametrize("adjust", ['tx', 'rx'])
    def test_slices_length(self, rx_longer, adjust):
        x = np.arange(1000.)
        xx = np.tile(x, 3)
        y = xx[11:1000+3*11]
        y_equal = xx[11:1000+1*11]
        if rx_longer is None:
            tx = x
            rx = y_equal
        else:
            if adjust == "tx":
                if rx_longer:
                    rx = y
                    tx = x
                else:
                    rx = x
                    tx = y
            elif adjust == "rx":
                if rx_longer:
                    rx = y
                    tx = x
                else:
                    rx = x
                    tx = y
        (tx, rx), acm = ber_functions.sync_and_adjust(tx, rx, adjust=adjust)
        assert tx.shape == rx.shape

    @pytest.mark.parametrize("N0", [(None, 1000), (1000, None)])
    @pytest.mark.parametrize("shiftN", [0, 43, 150, 800])
    @pytest.mark.parametrize("adjust", ['rx', 'tx'])
    def test_length_with_shift(self, N0, shiftN, adjust):
        sig = self.s[0]
        N = self.s.shape[1]
        N1, N2 = N0
        sign = np.roll(sig, shiftN)
        (tx, rx), acm = ber_functions.sync_and_adjust(sig[:N1], sign[:N2], adjust=adjust)
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
        (tx, rx), acm = ber_functions.sync_and_adjust(syms, syms2, adjust=adjust)
        npt.assert_allclose(tx, rx)
        (tx, rx), acm = ber_functions.sync_and_adjust(syms2, syms, adjust=adjust)
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
        (tx, rx), acm = ber_functions.sync_and_adjust(syms[:N1]*1j**tx_i, syms2[:N2]*1j**rx_i, adjust=adjust)
        npt.assert_allclose(tx, rx)

    @pytest.mark.parametrize("N", [12342])
    @pytest.mark.parametrize("plus", [True, False])
    def test_ser_with_random_slice(self, N, plus):
        s = signals.SignalQAMGrayCoded(4, 2**17)
        ss = np.tile(s, 4)
        if plus:
            s2 = ss[0,N:2**17+3*N]
        else:
            s2 = ss[0,N:2**17-3*N]
        npt.assert_allclose(s2.cal_ser(),0)


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

