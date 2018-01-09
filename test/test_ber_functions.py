import pytest
import numpy as np
import numpy.testing as npt
import matplotlib.pylab as plt

from dsp import modulation, ber_functions


class TestSynchronize(object):
    Q = modulation.QAMModulator(16)
    d = np.diff(np.unique(Q.symbols.real)).min()

    @pytest.mark.parametrize("shiftN", [np.random.randint(l*3*10**4//4+1, (l+1)*3*10**4//4) for l in range(4)])
    def test_find_sequence_offset_same_length_shift(self, shiftN):
        N = 3*10**4 # needs to be larger than PRBS pattern
        sig, syms, bits = self.Q.generate_signal(N, None, dual_pol=False)
        sig2 = np.roll(sig, shift=shiftN)
        offset = ber_functions.find_sequence_offset(syms, sig2)
        print(offset, shiftN)
        assert (offset == shiftN) or (N-offset == shiftN)

    @pytest.mark.parametrize("shiftN", [np.random.randint(l*3*10**4//4+1, (l+1)*3*10**4//4) for l in range(4)])
    def test_find_sequence_offset_same_length_data(self, shiftN):
        N = 3*10**4 # needs to be larger than PRBS pattern
        sig, syms, bits = self.Q.generate_signal(N, None, dual_pol=False, beta=0.01)
        sig2 = np.roll(sig, shift=shiftN)
        offset = ber_functions.find_sequence_offset(syms, sig2)
        syms2 = np.roll(syms, offset)
        print(offset, shiftN)
        npt.assert_allclose(syms2, sig2, atol=self.d/4)

    @pytest.mark.parametrize("shiftN", [np.random.randint(l*3*10**4//8+1, (l+1)*3*10**4//8) for l in range(8)]+[28924])
    def test_find_sequence_offset_same_length_shift_with_errors(self, shiftN):
        N = 3*10**4 # needs to be larger than PRBS pattern
        sig, syms, bits = self.Q.generate_signal(N, 15, dual_pol=False, beta=0.01)
        sig2 = np.roll(sig, shift=shiftN)
        sign = self.Q.quantize(sig2)
        offset = ber_functions.find_sequence_offset(syms, sign)
        print(offset, shiftN)
        assert (offset == shiftN) #or (N-offset == shiftN)

    @pytest.mark.parametrize("shiftN", [np.random.randint(l*3*10**4//8+1, (l+1)*3*10**4//8) for l in range(8)]+[28924])
    def test_find_sequence_offset_diff_length_shift(self, shiftN):
        N = 3*10**4 # needs to be larger than PRBS pattern
        sig, syms, bits = self.Q.generate_signal(N, None, dual_pol=False, beta=0.01, PRBS=False)
        sig2 = np.roll(sig, shift=shiftN)
        offset = ber_functions.find_sequence_offset(syms[:4000], sig2)
        print(offset, shiftN)
        assert (offset == shiftN)# or (N-offset == shiftN)

    @pytest.mark.parametrize("shiftN", [np.random.randint(l*3*10**4//8+1, (l+1)*3*10**4//8) for l in range(8)]+[28924])
    def test_find_sequence_offset_diff_length_data(self, shiftN):
        N = 3*10**4 # needs to be larger than PRBS pattern
        sig, syms, bits = self.Q.generate_signal(N, None, dual_pol=False, beta=0.01, PRBS=False)
        sig2 = np.roll(sig, shift=shiftN)
        offset = ber_functions.find_sequence_offset(syms[:4000], sig2)
        sig2 = np.roll(sig2, -offset)
        npt.assert_allclose(syms[:4000], sig2[:4000], atol=self.d/4)

    @pytest.mark.parametrize("N1, N2, adjust", [(None, 1000, "tx"),(None, 1000, "rx" ), (1000, None, "tx"), (1000, None, "rx") ])
    def test_sync_adjust_length(self, N1, N2, adjust):
        N = 3*10**4 # needs to be larger than PRBS pattern
        sig, syms, bits = self.Q.generate_signal(N, None, dual_pol=False, beta=0.01, PRBS=False)
        #N1 = 1000
        tx, rx = ber_functions.sync_and_adjust(sig[:N1], sig[:N2], adjust=adjust)
        if N1 is None and adjust is "tx":
            assert (tx.shape[0] == N2) and (rx.shape[0] == N2)
        elif N2 is None and adjust is "rx":
            assert (tx.shape[0] == N1) and (rx.shape[0] == N1)
        else:
            assert (tx.shape[0] == N) and (rx.shape[0] == N)

    @pytest.mark.parametrize("adjust", ["tx", "rx"])
    def test_sync_adjust_offset_same_length(self, adjust):
        N = 3*10**4 # needs to be larger than PRBS pattern
        sig, syms, bits = self.Q.generate_signal(N, None, dual_pol=False, beta=0.01, PRBS=False)
        shiftN = np.random.randint(1, N)
        syms2 = np.roll(syms, shift=shiftN)
        tx, rx = ber_functions.sync_and_adjust(syms, syms2, adjust=adjust)
        npt.assert_allclose(tx, rx)
        tx, rx = ber_functions.sync_and_adjust(syms2, syms, adjust=adjust)
        npt.assert_allclose(tx, rx)

    @pytest.mark.parametrize("adjust", ["tx", "rx"])
    def test_sync_adjust_offset_diff_length(self, adjust):
        Np = 2**15-1
        N = 2*Np
        sig, syms, bits = self.Q.generate_signal(N, None, dual_pol=False, beta=0.01, PRBS=True)
        shiftN = 0#np.random.randint(1, Np)
        syms2 = np.roll(syms, shift=shiftN)
        tx, rx = ber_functions.sync_and_adjust(syms[:Np], syms2, adjust=adjust)
        npt.assert_allclose(tx, rx)
        tx, rx = ber_functions.sync_and_adjust(syms, syms2[:Np], adjust=adjust)
        npt.assert_allclose(tx, rx)

    @pytest.mark.parametrize("adjust", ["tx", "rx"])
    @pytest.mark.parametrize("tx_i, rx_i", list(zip(list(range(1,4)) + 3*[0], 3*[0]+list(range(1,4)))))
    def test_sync_adjust_offset_rotated(self, adjust, tx_i, rx_i):
        #N = 3*10**4 # needs to be larger than PRBS pattern
        Np = 2**15-1
        N = 2*Np
        sig, syms, bits = self.Q.generate_signal(N, None, dual_pol=False, beta=0.01, PRBS=True)
        shiftN = np.random.randint(1, N)
        syms2 = np.roll(syms, shift=shiftN)
        print(shiftN)
        tx, rx = ber_functions.sync_and_adjust(syms*1j**tx_i, syms2*1j**rx_i, adjust=adjust)
        npt.assert_allclose(tx, rx)
        #for i in range(4):
            #tx, rx = ber_functions.sync_and_adjust(syms*1j**i, syms2*1.j**0, adjust=adjust)
            #npt.assert_allclose(tx, rx)
            #tx, rx = ber_functions.sync_and_adjust(syms, syms2*1j**i, adjust=adjust)
            #npt.assert_allclose(tx, rx)
            #tx, rx = ber_functions.sync_and_adjust(syms*1j**i, syms2, adjust="tx")
            #npt.assert_allclose(tx, rx)
            #tx, rx = ber_functions.sync_and_adjust(syms, syms2*1j**i, adjust="rx")
            #npt.assert_allclose(tx, rx)

    def test_sync_adjust_offset_diff_length_rotated(self):
        #N = 3*10**4 # needs to be larger than PRBS pattern
        Np = 2**15-1
        N = 2*Np
        sig, syms, bits = self.Q.generate_signal(N, None, dual_pol=False, beta=0.01, PRBS=True)
        shiftN = np.random.randint(1, N)
        syms2 = np.roll(syms, shift=shiftN)
        for i in range(0,4):
            tx, rx = ber_functions.sync_and_adjust(syms[:Np]*1j**i, syms2, adjust="rx")
            npt.assert_allclose(tx, rx)
            tx, rx = ber_functions.sync_and_adjust(syms*1j**i, syms2[:Np], adjust="rx")
            npt.assert_allclose(tx, rx)
            tx, rx = ber_functions.sync_and_adjust(syms, syms2[:Np]*1j**i, adjust="rx")
            npt.assert_allclose(tx, rx)
            tx, rx = ber_functions.sync_and_adjust(syms*1j**i, syms2[:Np], adjust="rx")
            npt.assert_allclose(tx, rx)
            tx, rx = ber_functions.sync_and_adjust(syms[:Np]*1j**i, syms2, adjust="tx")
            npt.assert_allclose(tx, rx)
            tx, rx = ber_functions.sync_and_adjust(syms*1j**i, syms2[:Np], adjust="tx")
            npt.assert_allclose(tx, rx)
            tx, rx = ber_functions.sync_and_adjust(syms, syms2[:Np]*1j**i, adjust="tx")
            npt.assert_allclose(tx, rx)
            tx, rx = ber_functions.sync_and_adjust(syms*1j**i, syms2[:Np], adjust="tx")
            npt.assert_allclose(tx, rx)

    def test_adjust_data_length(self):
        N = 3*10**4 # needs to be larger than PRBS pattern
        N1 = 1000
        sig, syms, bits = self.Q.generate_signal(N, None, dual_pol=False, beta=0.01, PRBS=False)
        tx, rx = ber_functions.adjust_data_length(sig, sig[:N1], method="truncate")
        assert (tx.shape[0] == N1) and (rx.shape[0] == N1)
        tx, rx = ber_functions.adjust_data_length(sig, sig[:N1], method="extend")
        assert (tx.shape[0] == N) and (rx.shape[0] == N)
        tx, rx = ber_functions.adjust_data_length(sig, sig[:N1], method=None)
        assert (tx.shape[0] == N1) and (rx.shape[0] == N1)
        tx, rx = ber_functions.adjust_data_length(sig[:N1], sig, method="truncate")
        assert (tx.shape[0] == N1) and (rx.shape[0] == N1)
        tx, rx = ber_functions.adjust_data_length(sig[:N1], sig, method="extend")
        assert (tx.shape[0] == N) and (rx.shape[0] == N)
        tx, rx = ber_functions.adjust_data_length(sig[:N1], sig, method=None)
        assert (tx.shape[0] == N) and (rx.shape[0] == N)






