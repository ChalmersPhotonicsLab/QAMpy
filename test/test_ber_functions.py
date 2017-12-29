import numpy as np
import numpy.testing as npt
import matplotlib.pylab as plt

from dsp import modulation, ber_functions

class TestSynchronize(object):
    Q = modulation.QAMModulator(16)
    d = np.diff(np.unique(Q.symbols.real)).min()
    def test_tx2rx_xcorr_same_length_shift(self):
        N = 3*10**4 # needs to be larger than PRBS pattern
        sig, syms, bits = self.Q.generate_signal(N, None, dual_pol=False)
        for l in range(0,4):
            shiftN = np.random.randint(l*N//4+1, (l+1)*N//4)
            sig2 = np.roll(sig, shift=shiftN)
            sign, offset, ac = ber_functions.sync_tx2rx_xcorr(syms, sig2)
            print(offset, shiftN)
            assert (offset == shiftN) or (N-offset == shiftN)

    def test_tx2rx_xcorr_same_length_data(self):
        N = 3*10**4 # needs to be larger than PRBS pattern
        sig, syms, bits = self.Q.generate_signal(N, None, dual_pol=False, beta=0.01)
        for l in range(0,4):
            shiftN = np.random.randint(l*N//4+1, (l+1)*N//4)
            sig2 = np.roll(sig, shift=shiftN)
            syms2, offset, ac = ber_functions.sync_tx2rx_xcorr(syms, sig2)
            print(offset, shiftN)
            npt.assert_allclose(syms2, sig2, atol=self.d/4)

    def test_tx2rx_xcorr_same_length_shift_with_errors(self):
        N = 3*10**4 # needs to be larger than PRBS pattern
        sig, syms, bits = self.Q.generate_signal(N, 15, dual_pol=False, beta=0.01)
        for l in range(0,4):
            shiftN = np.random.randint(l*N//4+1, (l+1)*N//4)
            sig2 = np.roll(sig, shift=shiftN)
            sign = self.Q.quantize(sig2)
            syms2, offset, ac = ber_functions.sync_tx2rx_xcorr(syms, sign)
            print(offset, shiftN)
            assert (offset == shiftN) or (N-offset == shiftN)

    def test_tx2rx_xcorr_diff_length_shift(self):
        N = 3*10**4 # needs to be larger than PRBS pattern
        sig, syms, bits = self.Q.generate_signal(N, None, dual_pol=False, beta=0.01, PRBS=False)
        for l in range(0,4):
            shiftN = np.random.randint(l*N//4+1, (l+1)*N//4)
            sig2 = np.roll(sig, shift=shiftN)
            syms2, offset, ac = ber_functions.sync_tx2rx_xcorr(syms[:4000], sig2)
            print(offset, shiftN)
            assert (offset == shiftN) or (N-offset == shiftN)

