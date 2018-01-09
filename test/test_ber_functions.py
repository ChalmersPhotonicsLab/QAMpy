import numpy as np
import numpy.testing as npt
import matplotlib.pylab as plt

from dsp import modulation, ber_functions

class TestSynchronize(object):
    Q = modulation.QAMModulator(16)
    d = np.diff(np.unique(Q.symbols.real)).min()
    def test_find_sequence_offset_same_length_shift(self):
        N = 3*10**4 # needs to be larger than PRBS pattern
        sig, syms, bits = self.Q.generate_signal(N, None, dual_pol=False)
        for l in range(0,4):
            shiftN = np.random.randint(l*N//4+1, (l+1)*N//4)
            sig2 = np.roll(sig, shift=shiftN)
            offset = ber_functions.find_sequence_offset(syms, sig2)
            print(offset, shiftN)
            assert (offset == shiftN) or (N-offset == shiftN)

    def test_find_sequence_offset_same_length_data(self):
        N = 3*10**4 # needs to be larger than PRBS pattern
        sig, syms, bits = self.Q.generate_signal(N, None, dual_pol=False, beta=0.01)
        for l in range(0,4):
            shiftN = np.random.randint(l*N//4+1, (l+1)*N//4)
            sig2 = np.roll(sig, shift=shiftN)
            offset = ber_functions.find_sequence_offset(syms, sig2)
            syms2 = np.roll(syms, offset)
            print(offset, shiftN)
            npt.assert_allclose(syms2, sig2, atol=self.d/4)

    def test_find_sequence_offset_same_length_shift_with_errors(self):
        N = 3*10**4 # needs to be larger than PRBS pattern
        sig, syms, bits = self.Q.generate_signal(N, 15, dual_pol=False, beta=0.01)
        for l in range(0,4):
            shiftN = np.random.randint(l*N//4+1, (l+1)*N//4)
            sig2 = np.roll(sig, shift=shiftN)
            sign = self.Q.quantize(sig2)
            offset = ber_functions.find_sequence_offset(syms, sign)
            print(offset, shiftN)
            assert (offset == shiftN) or (N-offset == shiftN)

    def test_find_sequence_offset_diff_length_shift(self):
        N = 3*10**4 # needs to be larger than PRBS pattern
        sig, syms, bits = self.Q.generate_signal(N, None, dual_pol=False, beta=0.01, PRBS=False)
        for l in range(0,4):
            shiftN = np.random.randint(l*N//4+1, (l+1)*N//4)
            sig2 = np.roll(sig, shift=shiftN)
            offset = ber_functions.find_sequence_offset(syms[:4000], sig2)
            print(offset, shiftN)
            assert (offset == shiftN) or (N-offset == shiftN)

    def test_sync_adjust_length(self):
        N = 3*10**4 # needs to be larger than PRBS pattern
        sig, syms, bits = self.Q.generate_signal(N, None, dual_pol=False, beta=0.01, PRBS=False)
        N1 = 1000
        tx, rx = ber_functions.sync_and_adjust(sig, sig[:N1], adjust="tx")
        assert (tx.shape[0] == N1) and (rx.shape[0] == N1)
        tx, rx = ber_functions.sync_and_adjust(sig, sig[:N1], adjust="rx")
        assert (tx.shape[0] == N) and (rx.shape[0] == N)
        tx, rx = ber_functions.sync_and_adjust(sig[:N1], sig, adjust="tx")
        assert (tx.shape[0] == N) and (rx.shape[0] == N)
        tx, rx = ber_functions.sync_and_adjust(sig[:N1], sig, adjust="rx")
        assert (tx.shape[0] == N1) and (rx.shape[0] == N1)

    def test_sync_adjust_offset_same_length(self):
        N = 3*10**4 # needs to be larger than PRBS pattern
        sig, syms, bits = self.Q.generate_signal(N, None, dual_pol=False, beta=0.01, PRBS=False)
        shiftN = np.random.randint(1, N)
        syms2 = np.roll(syms, shift=shiftN)
        tx, rx = ber_functions.sync_and_adjust(syms, syms2, adjust="rx")
        npt.assert_allclose(tx, rx)
        tx, rx = ber_functions.sync_and_adjust(syms, syms2, adjust="tx")
        npt.assert_allclose(tx, rx)
        tx, rx = ber_functions.sync_and_adjust(syms2, syms, adjust="tx")
        npt.assert_allclose(tx, rx)
        tx, rx = ber_functions.sync_and_adjust(syms2, syms, adjust="rx")
        npt.assert_allclose(tx, rx)

    def test_sync_adjust_offset_diff_length(self):
        Np = 2**15-1
        N = 2*Np
        sig, syms, bits = self.Q.generate_signal(N, None, dual_pol=False, beta=0.01, PRBS=True)
        shiftN = 0#np.random.randint(1, Np)
        syms2 = np.roll(syms, shift=shiftN)
        tx, rx = ber_functions.sync_and_adjust(syms[:Np], syms2, adjust="tx")
        npt.assert_allclose(tx, rx)
        tx, rx = ber_functions.sync_and_adjust(syms[:Np], syms2, adjust="rx")
        npt.assert_allclose(tx, rx)
        tx, rx = ber_functions.sync_and_adjust(syms, syms2[:Np], adjust="tx")
        npt.assert_allclose(tx, rx)
        tx, rx = ber_functions.sync_and_adjust(syms, syms2[:Np], adjust="rx")
        npt.assert_allclose(tx, rx)

    def test_sync_adjust_offset_rotated(self):
        #N = 3*10**4 # needs to be larger than PRBS pattern
        Np = 2**15-1
        N = 2*Np
        sig, syms, bits = self.Q.generate_signal(N, None, dual_pol=False, beta=0.01, PRBS=True)
        shiftN = np.random.randint(1, N)
        syms2 = np.roll(syms, shift=shiftN)
        for i in range(4):
            tx, rx = ber_functions.sync_and_adjust(syms*1j**i, syms2, adjust="rx")
            npt.assert_allclose(tx, rx)
            tx, rx = ber_functions.sync_and_adjust(syms, syms2*1j**i, adjust="rx")
            npt.assert_allclose(tx, rx)
            tx, rx = ber_functions.sync_and_adjust(syms*1j**i, syms2, adjust="tx")
            npt.assert_allclose(tx, rx)
            tx, rx = ber_functions.sync_and_adjust(syms, syms2*1j**i, adjust="rx")
            npt.assert_allclose(tx, rx)

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






