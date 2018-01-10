import pytest
import numpy as np
import random
import numpy.testing as npt
import matplotlib.pylab as plt

from dsp import modulation

def _flip_symbols(sig, idx, d):
    for i in idx:
        if np.random.randint(0,2):
            if sig[i].real > 0:
                sig[i] -= d
            else:
                sig[i] += d
        else:
            if sig[i].imag > 0:
                sig[i] -= 1.j*d
            else:
                sig[i] += 1.j*d
    return sig

class TestModulatorAttr(object):
    Q = modulation.QAMModulator(16)
    d = np.diff(np.unique(Q.symbols.real)).min()

    def test_quantize_symbols_set(self):
        """Check if all quantized symbols are from Q.symbols"""
        sig, sym, bits = self.Q.generate_signal(2 ** 10, 30, beta=0.01, ndim=1)
        sym_demod = self.Q.quantize(sig[0])
        out = np.in1d(sym_demod, self.Q.symbols)
        assert np.alltrue(out)

    def test_quantize_symbols_correct(self):
        """Check if all quantized symbols are from Q.symbols"""
        sig, sym, bits = self.Q.generate_signal(2 ** 10, None, beta=0.01, ndim=1)
        sym_demod = self.Q.quantize(sig[0])
        npt.assert_allclose(sym_demod, sym[0])

    def test_decode_correct(self):
        sig, sym, bits = self.Q.generate_signal(2 ** 10, None, beta=0.01, ndim=1)
        bb = self.Q.decode(sym[0])
        npt.assert_array_equal(bb, bits[0])

    def test_gen_signal(self):
        pass

    @pytest.mark.parametrize("Nerrors", range(5))
    @pytest.mark.parametrize("shiftN", np.random.randint(0,2**10, size=10))
    def test_ser(self, Nerrors, shiftN):
        sig, sym, bits = self.Q.generate_signal(2 ** 10, None, beta=0.01, ndim=1)
        #sig = sig[0]
        #sym = sym[0]
        idx = random.sample(range(sig.shape[1]), Nerrors)
        _flip_symbols(sig[0], idx, self.d)
        sig = np.roll(sig, shift=shiftN)
        ser = self.Q.cal_ser(sig, symbols_tx=sym)
        npt.assert_almost_equal(ser, Nerrors/sig.shape[1])

    @pytest.mark.parametrize("Nerrors", range(5))
    @pytest.mark.parametrize("shiftN", np.random.randint(0,2**10, size=10))
    def test_ber(self, Nerrors, shiftN):
        sig, sym, bits = self.Q.generate_signal(2 ** 10, None, beta=0.01, ndim=1)
        sig = sig[0]
        bits = bits[0]
        idx = random.sample(range(sig.shape[0]), Nerrors)
        _flip_symbols(sig, idx, self.d)
        sig = np.roll(sig, shift=shiftN)
        ber = self.Q.cal_ber(sig, bits_tx=bits)
        npt.assert_almost_equal(ber, Nerrors/(sig.shape[0]*self.Q.Nbits))








