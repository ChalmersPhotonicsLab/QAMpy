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
        s = modulation.QAMSymbolsGrayCoded(M, N, nmodes=1, seed=[prbsseed], bitclass=modulation.PRBSBits)
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

    def test_symbols_implace_op(self):
        s = modulation.QAMSymbolsGrayCoded(4, 2**12)
        avg1 = (abs(s)**2).mean()
        s += 5
        avg2 = (abs(s.symbols)**2).mean()
        npt.assert_array_almost_equal(avg1, avg2)

class TestPilotSignal(object):
    @pytest.mark.parametrize("N", [2**18, 2**12, 2**14])
    @pytest.mark.parametrize("nmodes", range(1, 4))
    def testshape(self, N, nmodes):
        s = modulation.SignalWithPilots(128, N, 256, 32, 1, nmodes=nmodes )
        assert s.shape[1] == N and s.shape[0] == nmodes

    @pytest.mark.parametrize("N", [1, 123, 256, 534])
    def testseqlen(self, N):
        QPSK = modulation.QAMSymbolsGrayCoded(4, 200)
        s = modulation.SignalWithPilots(128, 2**18, N, 0, 1 )
        dist = abs(s[0, :N, np.newaxis] - QPSK.coded_symbols)
        npt.assert_array_almost_equal(np.min(dist, axis=1), 0)

    @pytest.mark.parametrize("N", [1, 2, 32, 64, 128])
    def testphpilots(self, N):
        QPSK = modulation.QAMSymbolsGrayCoded(4, 200)
        s = modulation.SignalWithPilots(128, 2**18, 0, N, 1 )
        dist = abs(s[0, ::N, np.newaxis] - QPSK.coded_symbols)
        npt.assert_array_almost_equal(np.min(dist, axis=1), 0)

    @pytest.mark.parametrize("Nseq", [2, 32, 64, 128])
    @pytest.mark.parametrize("ph_i", [2, 32, 64, 128])
    def test_symbols(self, Nseq, ph_i):
        ph_fr = 50
        s = modulation.SignalWithPilots(128, ph_fr*ph_i+Nseq , Nseq, ph_i, 1)
        idx, idx_d, idx_p = modulation.SignalWithPilots._cal_pilot_idx(ph_fr*ph_i+Nseq, Nseq, ph_i)
        npt.assert_array_almost_equal(s[:,idx_d], s.symbols)

    @pytest.mark.parametrize("N", np.arange(2, 5))
    def testframes_shape(self, N):
        s = modulation.SignalWithPilots(128, 2**16, 128, 32, nframes=N)
        assert 2**16*N == s.shape[1]

    @pytest.mark.parametrize("N", np.arange(2, 5))
    def testframes_data(self, N):
        flen = 2**16
        s = modulation.SignalWithPilots(128, flen, 128, 32, nframes=N)
        for i in range(1,N):
            npt.assert_array_almost_equal(s[:,:flen], s[:, i*flen:(i+1)*flen])

    @pytest.mark.parametrize("N", [1, 123, 256, 534])
    def test_from_data_seqlen(self, N):
        QPSK = modulation.QAMSymbolsGrayCoded(4, 200)
        data = modulation.QAMSymbolsGrayCoded(128, 2**12)
        s = modulation.SignalWithPilots.from_data_array(data, 2**12, N, 0, 1)
        dist = abs(s[0, :N, np.newaxis] - QPSK.coded_symbols)
        npt.assert_array_almost_equal(np.min(dist, axis=1), 0)

    @pytest.mark.parametrize("N", [1, 2, 32, 64, 128])
    def test_from_data_phpilots(self, N):
        QPSK = modulation.QAMSymbolsGrayCoded(4, 200)
        data = modulation.QAMSymbolsGrayCoded(128, 2**12)
        s = modulation.SignalWithPilots.from_data_array(data, 2**12, 0, N, 1)
        dist = abs(s[0, ::N, np.newaxis] - QPSK.coded_symbols)
        npt.assert_array_almost_equal(np.min(dist, axis=1), 0)

    def test_from_data_symbols(self):
        data = modulation.QAMSymbolsGrayCoded(128, 2**12)
        s = modulation.SignalWithPilots.from_data_array(data, 2**12, 128, 16, 1)
        npt.assert_array_almost_equal(data[:, :s.symbols.shape[1]], s.symbols)

    def test_from_data_symbols2(self):
        data = modulation.QAMSymbolsGrayCoded(128, 2**12)
        idx, idx_d, idx_p = modulation.SignalWithPilots._cal_pilot_idx(2**12, 128, 16)
        s = modulation.SignalWithPilots.from_data_array(data, 2**12, 128, 16, 1)
        npt.assert_array_almost_equal(s[:,idx_d], data[:, :np.count_nonzero(idx_d)])

    @pytest.mark.parametrize("p_ins", [0, 1, 16, 32])
    def test_from_data_symbols3(self, p_ins):
        data = modulation.QAMSymbolsGrayCoded(128, 2**12-128)
        s = modulation.SignalWithPilots.from_data_array(data, 2**12, 128, p_ins, 1)
        assert s.shape[1] == 2**12

    @pytest.mark.parametrize("os", np.arange(2,5))
    def test_resample(self, os):
        N = 2**12-128
        Nn = os*N
        s = modulation.QAMSymbolsGrayCoded(128, N)
        sn = modulation.SignalWithPilots.from_data_array(s, 2**12, 128, None, 1)
        si = sn.resample(2, beta=0.2, renormalise=False)
        assert si.shape[0] == sn.shape[0]
        assert si.shape[1] == 2*sn.shape[1]
        npt.assert_array_almost_equal(s ,si.symbols)


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
        s1 = modulation.QAMSymbolsGrayCoded(16, 1000*r1, seed=[1,2])
        s2 = modulation.QAMSymbolsGrayCoded(4, 1000*r2, seed=[1,2])
        o = modulation.TDHQAMSymbols.from_symbol_arrays(s1, s2, r2/r)
        o2 = modulation.TDHQAMSymbols((16, 4), 1000*(r1+r2), fr=r2/r, seed=[1,2] )
        npt.assert_array_almost_equal(o, o2)

    @pytest.mark.parametrize("attr", ["fb", "M", "f_M1", "f_M2", "f_M", "fr"])
    def test_preserveattr(self, attr):
        s1 = modulation.TDHQAMSymbols((16, 4), 1000, 0.5)
        s2 = s1+10
        a1 = getattr(s1, attr)
        a2 = getattr(s2, attr)
        if isinstance(a1, np.ndarray):
            npt.assert_array_almost_equal(a1, a2)
        else:
            assert a1 == a2

class TestSignal(object):
    @pytest.mark.parametrize("attr", ["fb", "M", "fs", "symbols"])
    def test_preserveattr(self, attr):
        s1 = modulation.Signal(16, 1000, fs=2)
        s2 = s1+10
        a1 = getattr(s1, attr)
        a2 = getattr(s2, attr)
        if isinstance(a1, np.ndarray):
            npt.assert_array_almost_equal(a1, a2)
        else:
            assert a1 == a2

    @pytest.mark.parametrize("os", np.arange(2,5))
    @pytest.mark.parametrize("nmodes", np.arange(1,3))
    def test_samplerate(self, os, nmodes):
        N = 1000
        Nn = os*N
        s = modulation.Signal(16, N, fs=os, nmodes=nmodes)
        assert s.shape == (nmodes, Nn)

    @pytest.mark.parametrize("os", np.arange(2,5))
    def test_resample(self, os):
        N = 1000
        Nn = os*N
        s = modulation.QAMSymbolsGrayCoded(128, N)
        sn = modulation.Signal.from_symbol_array(s, fs=os, beta=0.2, renormalise=False)
        si = sn.resample(1, beta=0.2, renormalise=False)
        si /= abs(si).max()
        s /= abs(s).max()
        npt.assert_array_almost_equal(s, si)

    def test_symbols_implace_op(self):
        s = modulation.Signal(4, 2 ** 12)
        avg1 = (abs(s.symbols)**2).mean()
        s += 5
        avg2 = (abs(s.symbols)**2).mean()
        npt.assert_array_almost_equal(avg1, avg2)

    def test_symbolinherit(self):
        N = 1000
        s = modulation.QAMSymbolsGrayCoded(128, N)
        sn = modulation.Signal.from_symbol_array(s, fs=2, beta=0.2, renormalise=False)
        assert sn.symbols is s

    def test_symbolinherit2(self):
        N = 1000
        s = modulation.QAMSymbolsGrayCoded(128, N)
        sn = modulation.Signal.from_symbol_array(s, fs=2, beta=0.2, renormalise=False)
        sn2 = sn+2
        assert sn2.symbols is s

    def test_symbolinherit3(self):
        N = 1000
        s = modulation.Signal(16, N, fs=2)
        sn = s.resample(1, beta=0.2)
        assert sn.symbols is s.symbols

    @pytest.mark.parametrize("attr", ["M", "bits", "_encoding", "_bitmap_mtx",
                                      "fb", "_code", "coded_symbols"])
    def test_symbol_attr(self, attr):
        s = modulation.Signal(16, 2000, fs=2)
        a = getattr(s, attr)
        assert a is not None


class TestSignalQualityOnSignal(object):

    @pytest.mark.parametrize("nmodes", np.arange(1, 4))
    def test_ser_shape(self, nmodes):
        s = modulation.Signal(16, 2 ** 16, nmodes=nmodes)
        ser = s.cal_ser()
        assert ser.shape[0] == nmodes

    def test_ser_value(self):
        s = modulation.Signal(16, 2 ** 16)
        ser = s.cal_ser()
        assert ser[0] == 0

    @pytest.mark.parametrize("nmodes", np.arange(1, 4))
    def test_evm_shape(self, nmodes):
        s = modulation.Signal(16, 2 ** 16, nmodes=nmodes)
        evm = s.cal_evm()
        assert evm.shape[0] == nmodes

    def test_evm_value(self):
        s = modulation.Signal(16, 2 ** 16)
        evm = s.cal_evm()
        assert evm[0] < 1e-4

    @pytest.mark.parametrize("nmodes", np.arange(1, 4))
    def test_ber_shape(self, nmodes):
        s = modulation.Signal(16, 2 ** 16, nmodes=nmodes)
        ber = s.cal_ber()
        assert ber.shape[0] == nmodes

    def test_ber_value(self):
        s = modulation.Signal(16, 2 ** 16)
        s += 0.05*(np.random.randn(2**16) + 1.j*np.random.randn(2**16))
        ber = s.cal_ber()
        assert ber[0] == 0

    @pytest.mark.parametrize("nmodes", np.arange(1, 4))
    def test_est_snr_shape(self, nmodes):
        s = modulation.Signal(16, 2 ** 16, nmodes=nmodes)
        snr = s.est_snr()
        assert snr.shape[0] == nmodes

    def test_est_snr_value(self):
        s = modulation.Signal(16, 2 ** 16)
        snr = s.est_snr()
        assert snr[0] > 1e25

    @pytest.mark.parametrize("nmodes", np.arange(1, 4))
    def test_cal_gmi_shape(self, nmodes):
        s = modulation.Signal(16, 2 ** 16, nmodes=nmodes)
        gmi, gmi_pb = s.cal_gmi()
        assert gmi.shape[0] == nmodes

    @pytest.mark.parametrize("M", [16, 128, 256])
    def test_cal_gmi_value(self, M):
        s = modulation.Signal(M, 2 ** 16)
        s += 0.05*(np.random.randn(2**16) + 1.j*np.random.randn(2**16))
        nbits = np.log2(M)
        gmi, gmi_pb = s.cal_gmi()
        npt.assert_almost_equal(gmi[0], nbits)



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









