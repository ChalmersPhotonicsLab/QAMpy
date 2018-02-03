import pytest
import numpy as np
import random
import numpy.testing as npt
import matplotlib.pylab as plt

from dsp import modulation, signal_quality

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

class TestBits(object):
    @pytest.mark.parametrize("ctype", [modulation.PRBSBits, modulation.RandomBits])
    @pytest.mark.parametrize("N", [2**10, 2**14, 2**18])
    @pytest.mark.parametrize("nmodes", np.arange(1, 4))
    def testshape(self, ctype, N, nmodes):
        b = ctype(N, nmodes=nmodes)
        assert b.shape == (nmodes, N)

    @pytest.mark.parametrize("ctype", [modulation.PRBSBits, modulation.RandomBits])
    def testtype(self, ctype):
        c = ctype(100, nmodes=1)
        assert c.dtype == np.bool

    @pytest.mark.parametrize("ctype", [modulation.PRBSBits, modulation.RandomBits])
    def testdist(self, ctype):
        c = ctype(10**6)
        ones = np.count_nonzero(c)
        assert (ones-10**6/2)<1000

    def testprbspreserveattr(self):
        c = modulation.PRBSBits(1000)
        cc = np.roll(c, 100)
        assert cc._seed == c._seed
        assert cc._order == c._order

    def testrandpreserveattr(self):
        c = modulation.RandomBits(1000)
        cc = np.roll(c, 100)
        assert cc._seed == c._seed
        assert cc._rand_state == c._rand_state


class TestQAMSymbolsGray(object):
    @pytest.mark.parametrize("N", [np.random.randint(1,2**20)])
    @pytest.mark.parametrize("nmodes", np.arange(1, 4))
    def test_shape(self, N, nmodes):
        s = modulation.QAMSymbolsGrayCoded(16, N, nmodes)
        assert s.shape == (nmodes, N)

    @pytest.mark.parametrize("M", [2**i for i in range(2, 8)])
    def testavgpow(self, M):
        s = modulation.QAMSymbolsGrayCoded(M, 2**18)
        p = (abs(s)**2).mean()
        npt.assert_almost_equal(p, 1, 2)

    @pytest.mark.parametrize("M", [2**i for i in range(2, 8)])
    def test_symbols(self, M):
        s = modulation.QAMSymbolsGrayCoded(M, 1000, nmodes=1)
        si = np.unique(s)
        thsyms = modulation.theory.cal_symbols_qam(M)/np.sqrt(modulation.theory.cal_scaling_factor_qam(M))
        d = np.min(abs(s[0,:, np.newaxis]-thsyms), axis=1)
        assert si.shape[0] == M
        npt.assert_array_almost_equal(d, 0)

    @pytest.mark.parametrize("M", [2**i for i in range(2, 8)])
    @pytest.mark.parametrize("prbsseed", np.arange(1,10))
    def testbits(self, M, prbsseed):
        s = modulation.QAMSymbolsGrayCoded(M, 1000, nmodes=1, seed=[prbsseed])
        npt.assert_array_almost_equal(s.demodulate(s), s.bits)

    @pytest.mark.parametrize("M", [2**i for i in range(2, 8)])
    @pytest.mark.parametrize("prbsseed", np.arange(1,10))
    def testbits2(self, M, prbsseed):
        N = 1000
        s = modulation.QAMSymbolsGrayCoded(M, N, nmodes=1, seed=[prbsseed])
        bitsq = modulation.make_prbs_extXOR(s.bits._order[0], N*np.log2(M), prbsseed)
        npt.assert_array_almost_equal(s.demodulate(s)[0], bitsq)

    @pytest.mark.parametrize("M", [2**i for i in range(2, 8)])
    def testfromarray_order(self, M):
        a = np.random.choice(modulation.theory.cal_symbols_qam(M), 1000)
        s = modulation.QAMSymbolsGrayCoded.from_symbol_array(a)
        assert np.unique(s).shape[0] is M

    @pytest.mark.parametrize("M", [2**i for i in range(2, 8)])
    def testfromarray_avgpow(self, M):
        a = np.random.choice(modulation.theory.cal_symbols_qam(M), 1000)
        s = modulation.QAMSymbolsGrayCoded.from_symbol_array(a)
        npt.assert_almost_equal((abs(s)**2).mean(), (abs(s)**2).mean())

    @pytest.mark.parametrize("N", [1024, 12423, 100000, 2**18])
    @pytest.mark.parametrize("M", [2**i for i in range(2, 8)])
    def testfrombits_len(self,  N, M):
        b = modulation.make_prbs_extXOR(15, N)
        s = modulation.QAMSymbolsGrayCoded.from_bit_array(b, M)
        nbit = int(np.log2(M))
        nbitlen = N//nbit
        assert s.shape[1] == nbitlen

    @pytest.mark.parametrize("N", [1024, 12423, 100000, 2**18])
    @pytest.mark.parametrize("M", [2**i for i in range(2, 8)])
    def testfrombits_bits(self,  N, M):
        b = modulation.make_prbs_extXOR(15, N)
        s = modulation.QAMSymbolsGrayCoded.from_bit_array(b, M)
        nbit = int(np.log2(M))
        nbitlen = N//nbit
        npt.assert_almost_equal(s.demodulate(s)[0], b[:nbitlen*nbit])

    @pytest.mark.parametrize("attr", ["_M", "_bits", "_encoding", "_bitmap_mtx",
                                      "_fb", "_code", "_coded_symbols"])
    def test_preserveattr(self, attr):
        s1 = modulation.QAMSymbolsGrayCoded(16, 1000)
        s2 = s1+10
        a1 = getattr(s1, attr)
        a2 = getattr(s2, attr)
        if isinstance(a1, np.ndarray):
            npt.assert_array_almost_equal(a1, a2)
        else:
            assert a1 == a2

    def test_symbolpreserve(self):
        s1 = modulation.QAMSymbolsGrayCoded(16, 1000)
        s2 = s1 + 10
        npt.assert_array_almost_equal(s1, s2.symbols)


class TestPilotSignal(object):
    @pytest.mark.parametrize("N", [2**18, 2**12, 2**14])
    @pytest.mark.parametrize("nmodes", range(1, 4))
    def testshape(self, N, nmodes):
        s = modulation.SignalWithPilots(128, N, 256, 32, 1, nmodes=nmodes )
        assert s.shape[1] == N and s.shape[0] == nmodes

    @pytest.mark.parametrize("N", [1, 123, 256, 534])
    def testseqlen(self, N):
        QPSK = modulation.QAMModulator(4)
        s = modulation.SignalWithPilots(128, 2**18, N, 0, 1 )
        dist = abs(s[0, :N, np.newaxis] - QPSK.symbols)
        npt.assert_array_almost_equal(np.min(dist, axis=1), 0)

    @pytest.mark.parametrize("N", [1, 2, 32, 64, 128])
    def testphpilots(self, N):
        QPSK = modulation.QAMModulator(4)
        s = modulation.SignalWithPilots(128, 2**18, 0, N, 1 )
        dist = abs(s[0, ::N, np.newaxis] - QPSK.symbols)
        npt.assert_array_almost_equal(np.min(dist, axis=1), 0)


class TestTDHybridsSymbols(object):
    @pytest.mark.parametrize("M1", [4, 16, 32, 64, 128, 256])
    @pytest.mark.parametrize("M2", [4, 16, 32, 64, 128, 256])
    def test_dist(self, M1, M2):
        s = modulation.TDHQAMSymbols((M1, M2), 1000, fr=0.5)
        d1_r = np.min(np.diff(np.unique(s._symbols_M1.real)))
        d2_r = np.min(np.diff(np.unique(s._symbols_M2.real)))
        d1_i = np.min(np.diff(np.unique(s._symbols_M1.imag)))
        d2_i = np.min(np.diff(np.unique(s._symbols_M2.imag)))
        npt.assert_approx_equal(d1_r, d2_r)
        npt.assert_approx_equal(d1_i, d2_i)

    @pytest.mark.parametrize("r1", np.arange(1, 10))
    @pytest.mark.parametrize("r2", np.arange(1, 10))
    def test_ratio(self, r1, r2):
        import math
        if math.gcd(r1, r2) > 1:
            assert True
            return
        r = r1+r2
        o = modulation.TDHQAMSymbols((16,4),1000, fr=r2/r)
        for i in range(r):
            s = o[0, i::r]
            if i%r < r1:
                d = np.min(abs(s[:, np.newaxis]-o._symbols_M1), axis=1)
                npt.assert_array_almost_equal(d, 0)
            else:
                d = np.min(abs(s[:, np.newaxis]-o._symbols_M2), axis=1)
                npt.assert_array_almost_equal(d, 0)

    def testclass(self):
        s = modulation.TDHQAMSymbols((16,4), 1000)
        type(s._symbols_M1 ) is modulation.QAMSymbolsGrayCoded
        type(s._symbols_M2 ) is modulation.QAMSymbolsGrayCoded

    @pytest.mark.parametrize("r1", np.arange(1, 3))
    @pytest.mark.parametrize("r2", np.arange(1, 3))
    def test_from_arrays_shape(self, r1, r2):
        import math
        if math.gcd(r1, r2) > 1:
            assert True
            return
        r = r1+r2
        s1 = modulation.QAMSymbolsGrayCoded(16, 1000*r1)
        s2 = modulation.QAMSymbolsGrayCoded(4, 1000*r2)
        o = modulation.TDHQAMSymbols.from_symbol_arrays(s1, s2, r2/r)
        assert o.shape == (1, 1000*(r1+r2))

    @pytest.mark.parametrize("r1", np.arange(1, 3))
    @pytest.mark.parametrize("r2", np.arange(1, 3))
    def test_from_arrays_ratio(self, r1, r2):
        import math
        if math.gcd(r1, r2) > 1:
            assert True
            return
        r = r1+r2
        s1 = modulation.QAMSymbolsGrayCoded(16, 1000*r1)
        s2 = modulation.QAMSymbolsGrayCoded(4, 1000*r2)
        o = modulation.TDHQAMSymbols.from_symbol_arrays(s1, s2, r2/r)
        o2 = modulation.TDHQAMSymbols((16, 4), 1000*(r1+r2), fr=r2/r)
        npt.assert_array_almost_equal(o, o2)


class TestModulatorAttr(object):
    Q = modulation.QAMModulator(16)
    d = np.diff(np.unique(Q.symbols.real)).min()

    def test_quantize_symbols_set(self):
        """Check if all quantized symbols are from Q.symbols"""
        sig, sym, bits = self.Q.generate_signal(2 ** 10, 30, beta=0.01, ndim=1)
        sym_demod = self.Q.quantize(sig)
        out = np.in1d(sym_demod.flatten(), self.Q.symbols.flatten())
        assert np.alltrue(out)

    def test_quantize_symbols_correct(self):
        """Check if all quantized symbols are from Q.symbols"""
        sig, sym, bits = self.Q.generate_signal(2 ** 10, None, beta=0.01, ndim=1)
        sym_demod = self.Q.quantize(sig)
        npt.assert_allclose(sym_demod, sym)

    def test_decode_correct(self):
        sig, sym, bits = self.Q.generate_signal(2 ** 10, None, beta=0.01, ndim=1)
        bb = self.Q.decode(sym[0])
        npt.assert_array_equal(bb, bits[0])

    def test_gen_signal(self):
        pass

    @pytest.mark.parametrize("Nerrors", range(5))
    @pytest.mark.parametrize("shiftN", np.random.randint(0,2**10, size=10))
    @pytest.mark.parametrize("ndims", range(1,4))
    def test_ser(self, Nerrors, shiftN, ndims):
        sig, sym, bits = self.Q.generate_signal(2 ** 10, None, beta=0.01, ndim=ndims)
        for i in range(ndims):
            idx = random.sample(range(sig.shape[1]), Nerrors)
            _flip_symbols(sig[i], idx, self.d)
        sig = np.roll(sig, shift=shiftN, axis=-1)
        ser = self.Q.cal_ser(sig)
        # note that for the npt.assert_... functions if the desired is scalar, all array values are checked
        # against the scalar hence it passes if ser is 1- or multi-dim
        npt.assert_almost_equal(ser, Nerrors/sig.shape[1])

    @pytest.mark.parametrize("Nerrors", range(5))
    @pytest.mark.parametrize("shiftN", np.random.randint(0,2**10, size=10))
    @pytest.mark.parametrize("ndims", range(1,3))
    def test_ber(self, Nerrors, shiftN, ndims):
        sig, sym, bits = self.Q.generate_signal(2 ** 10, None, beta=0.01, ndim=ndims)
        for i in range(ndims):
            idx = random.sample(range(sig.shape[1]), Nerrors)
            _flip_symbols(sig[i], idx, self.d)
            sig[i] = np.roll(sig[i], shift=(shiftN+i*np.random.randint(0, 100)))
        ber = self.Q.cal_ber(sig)
        npt.assert_almost_equal(ber, Nerrors/(sig.shape[1]*self.Q.Nbits))

    @pytest.mark.parametrize("snr", [10, 15, 20])
    @pytest.mark.parametrize("ndims", range(2,3))
    def test_evm(self, snr, ndims):
        sig, sym, bits = self.Q.generate_signal(2 ** 15, snr, beta=0.01, ndim=ndims)
        evm = self.Q.cal_evm(sig, blind=False)
        npt.assert_almost_equal(-10*np.log10(evm**2), snr, decimal=0)

    @pytest.mark.parametrize("snr", [10, 15, 20])
    def test_est_snr(self, snr):
        sig, sym, bits = self.Q.generate_signal(2 ** 15, snr, beta=0.01, ndim=1)
        e_snr = self.Q.est_snr(sig)
        npt.assert_almost_equal(10*np.log10(e_snr), snr, decimal=1)

class TestPilotModulator(object):
    Q = modulation.PilotModulator(128)

    @pytest.mark.parametrize("N", [2**18, 2**12, 2**14])
    @pytest.mark.parametrize("ndims", range(1, 4))
    def testshape(self, N, ndims):
        s, d, p = self.Q.generate_signal(N, 256, 32, ndims)
        assert s.shape[1] == N and s.shape[0] == ndims

    @pytest.mark.parametrize("N", [1, 123, 256, 534])
    def testseqlen(self, N):
        QPSK = modulation.QAMModulator(4)
        s, d, p = self.Q.generate_signal(2**16, N, 0, 1)
        dist = abs(s[0, :N, np.newaxis] - QPSK.symbols)
        npt.assert_array_almost_equal(np.min(dist, axis=1), 0)

    @pytest.mark.parametrize("N", [1, 2, 32, 64, 128])
    def testphpilots(self, N):
        QPSK = modulation.QAMModulator(4)
        s, d, p = self.Q.generate_signal(2**16, 0, N, 1)
        dist = abs(s[0, ::N, np.newaxis] - QPSK.symbols)
        npt.assert_array_almost_equal(np.min(dist, axis=1), 0)


class TestTDHybrid(object):
    @pytest.mark.parametrize("M1", [4, 16, 32, 64, 128, 256])
    @pytest.mark.parametrize("M2", [4, 16, 32, 64, 128, 256])
    def test_dist(self, M1, M2):
        hm = modulation.TDHQAMModulator(M1, M2, 0.5, power_method="dist")
        d1_r = np.min(np.diff(np.unique(hm.mod_M1.symbols.real)))
        d2_r = np.min(np.diff(np.unique(hm.mod_M2.symbols.real)))
        d1_i = np.min(np.diff(np.unique(hm.mod_M1.symbols.imag)))
        d2_i = np.min(np.diff(np.unique(hm.mod_M2.symbols.imag)))
        npt.assert_approx_equal(d1_r, d2_r)
        npt.assert_approx_equal(d1_i, d2_i)

    @pytest.mark.parametrize("r1", np.arange(1, 10))
    @pytest.mark.parametrize("r2", np.arange(1, 10))
    def test_ratio(self, r1, r2):
        import math
        if math.gcd(r1, r2) > 1:
            assert True
            return
        r = r1+r2
        hm = modulation.TDHQAMModulator(16, 4, r2/r)
        o = hm.generate_signal(1000)
        for i in range(r):
            s = o[0, i::r]
            if i%r < r1:
                d = np.min(abs(s[:, np.newaxis]-hm.mod_M1.symbols), axis=1)
                npt.assert_array_almost_equal(d, 0)
            else:
                d = np.min(abs(s[:, np.newaxis]-hm.mod_M2.symbols), axis=1)
                npt.assert_array_almost_equal(d, 0)





@pytest.mark.parametrize("M", [16, 32, 64, 128, 256])
@pytest.mark.parametrize("ndims", range(1,3))
def test_gmi_cal(M, ndims):
    Q = modulation.QAMModulator(M)
    sig, syms, bits = Q.generate_signal(2**15, None, ndim=ndims)
    gmi = Q.cal_gmi(sig)[0]
    npt.assert_almost_equal(gmi, np.log2(M), decimal=1)









