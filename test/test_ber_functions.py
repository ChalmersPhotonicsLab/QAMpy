import pytest
import numpy as np
import numpy.testing as npt

from dsp import modulation, ber_functions


class TestSynchronize(object):
    Q = modulation.QAMModulator(16)
    d = np.diff(np.unique(Q.symbols.real)).min()

    @pytest.mark.parametrize("shiftN", [np.random.randint(l*3*10**4//4+1, (l+1)*3*10**4//4) for l in range(4)])
    @pytest.mark.parametrize("snr", [5, 20, 40])
    @pytest.mark.parametrize("N1", [None, 4000])
    def test_find_sequence_offset_shift(self, shiftN, snr, N1):
        N = 3*10**4 # needs to be larger than PRBS pattern
        sig, syms, bits = self.Q.generate_signal(N, snr, dual_pol=False)
        sig2 = np.roll(sig, shift=shiftN)
        offset = ber_functions.find_sequence_offset(syms[:N1], sig2)
        assert (offset == shiftN)

    @pytest.mark.parametrize("shiftN", [np.random.randint(l*(2**15-1)//2+1, (l+1)*(2**15-1)//2) for l in range(4)]+[48630])
    @pytest.mark.parametrize("N1", [None, 4000])
    def test_find_sequence_offset_data(self, shiftN, N1):
        N = 2*(2**15-1)
        sig, syms, bits = self.Q.generate_signal(N, None, dual_pol=False, beta=0.01)
        sig2 = np.roll(sig, shift=shiftN)
        offset = ber_functions.find_sequence_offset(syms[:N1], sig2)
        sig2 = np.roll(sig2, -offset)
        npt.assert_allclose(syms[:N1], sig2[:N1], atol=self.d/4)

    @pytest.mark.parametrize("shiftN", [np.random.randint(l*(2**15-1)//2+1, (l+1)*(2**15-1)//2) for l in range(4)]+[48630])
    @pytest.mark.parametrize("i", range(4))
    @pytest.mark.parametrize("snr", [5, 20, 40])
    @pytest.mark.parametrize("N1", [None, 4000])
    def test_find_sequence_offset_complex_shift(self, shiftN, i, snr, N1):
        N = 2*(2**15-1)
        sig, syms, bits = self.Q.generate_signal(N, None, dual_pol=False, beta=0.01, PRBS=False)
        sig2 = np.roll(sig, shift=shiftN)
        offset, syms2, ii = ber_functions.find_sequence_offset_complex(syms[:N1]*1j**i, sig2)
        assert (offset == shiftN)

    @pytest.mark.parametrize("shiftN", [np.random.randint(l*(2**15-1)//2+1, (l+1)*(2**15-1)//2) for l in range(4)]+[48630])
    @pytest.mark.parametrize("i", range(4))
    @pytest.mark.parametrize("N1", [None, 4000])
    def test_find_sequence_offset_complex_data(self, shiftN, i, N1):
        N = 2*(2**15-1)
        sig, syms, bits = self.Q.generate_signal(N, None, dual_pol=False, beta=0.01)
        sig2 = np.roll(sig, shift=shiftN)
        offset, syms2, ii = ber_functions.find_sequence_offset_complex(syms[:N1]*1j**i, sig2)
        sig2 = np.roll(sig2, -offset)
        npt.assert_allclose(syms2, sig2[:N1], atol=self.d/4)

    @pytest.mark.parametrize("shiftN", [np.random.randint(l*(2**15-1)//2+1, (l+1)*(2**15-1)//2) for l in range(4)]+[48630])
    @pytest.mark.parametrize("i", range(4))
    def test_find_sequence_offset_complex_test_rotation(self, shiftN, i):
        N = 2*(2**15-1)
        sig, syms, bits = self.Q.generate_signal(N, None, dual_pol=False, beta=0.01)
        sig2 = np.roll(sig, shift=shiftN)
        offset, syms2, ii = ber_functions.find_sequence_offset_complex(syms*1j**i, sig2)
        assert (4-ii)%4 == i

    @pytest.mark.parametrize("N1, N2, adjust", [(None, 1000, "tx"),(None, 1000, "rx" ), (1000, None, "tx"), (1000, None, "rx") ])
    def test_sync_adjust_test_length(self, N1, N2, adjust):
        N = 3*10**4 # needs to be larger than PRBS pattern
        sig, syms, bits = self.Q.generate_signal(N, None, dual_pol=False, beta=0.01, PRBS=False)
        tx, rx = ber_functions.sync_and_adjust(sig[:N1], sig[:N2], adjust=adjust)
        if N1 is None and adjust is "tx":
            assert (tx.shape[0] == N2) and (rx.shape[0] == N2)
        elif N2 is None and adjust is "rx":
            assert (tx.shape[0] == N1) and (rx.shape[0] == N1)
        else:
            assert (tx.shape[0] == N) and (rx.shape[0] == N)

    @pytest.mark.parametrize("shiftN", [np.random.randint(l*(2**15-1)//2+1, (l+1)*(2**15-1)//2) for l in range(4)]+[48630])
    @pytest.mark.parametrize("adjust", ["tx", "rx"])
    def test_sync_adjust_offset_same_length(self, adjust, shiftN):
        Np = 2**15-1
        N = 2*Np
        sig, syms, bits = self.Q.generate_signal(N, None, dual_pol=False, beta=0.01, PRBS=True)
        #shiftN = np.random.randint(1, N)
        syms2 = np.roll(syms, shift=shiftN)
        tx, rx = ber_functions.sync_and_adjust(syms, syms2, adjust=adjust)
        npt.assert_allclose(tx, rx)
        tx, rx = ber_functions.sync_and_adjust(syms2, syms, adjust=adjust)
        npt.assert_allclose(tx, rx)

    @pytest.mark.parametrize("adjust", ["tx", "rx"])
    @pytest.mark.parametrize("tx_i, rx_i", list(zip(list(range(1,4)) + 3*[0], 3*[0]+list(range(1,4)))))
    @pytest.mark.parametrize("N1, N2", [(None, 2**15-1), (2**15-1, None)])
    @pytest.mark.parametrize("shiftN", [np.random.randint(l*(2**15-1)//2+1, (l+1)*(2**15-1)//2) for l in range(4)] + [48630])
    def test_sync_adjust_offset_rotated(self, adjust, tx_i, rx_i, N1, N2, shiftN):
        Np = 2**15-1
        N = 2*Np
        sig, syms, bits = self.Q.generate_signal(N, None, dual_pol=False, beta=0.01, PRBS=True)
        syms2 = np.roll(syms, shift=shiftN)
        tx, rx = ber_functions.sync_and_adjust(syms[:N1]*1j**tx_i, syms2[:N2]*1j**rx_i, adjust=adjust)
        npt.assert_allclose(tx, rx)

    @pytest.mark.parametrize("N1, N2", [(None, 1000), (1000, None)])
    @pytest.mark.parametrize("method", ["truncate", "extend", None])
    def test_adjust_data_length(self, N1, N2, method):
        N = 3*10**4 # needs to be larger than PRBS pattern
        sig, syms, bits = self.Q.generate_signal(N, None, dual_pol=False, beta=0.01, PRBS=False)
        tx, rx = ber_functions.adjust_data_length(sig[:N1], sig[:N2], method=method)
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

