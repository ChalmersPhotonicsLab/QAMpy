import pytest
import numpy as np
import numpy.testing as npt

from dsp import signals, theory


def _flip_symbols(sig, idx, d, mode=0):
    for i in idx:
        if np.random.randint(0, 2):
            if sig[mode,i].real > 0:
                sig[mode,i] -= d
            else:
                sig[mode,i] += d
        else:
            if sig[mode,i].imag > 0:
                sig[mode,i] -= 1.j * d
            else:
                sig[mode,i] += 1.j * d
    return sig


class TestBits(object):
    @pytest.mark.parametrize("ctype", [signals.PRBSBits, signals.RandomBits])
    @pytest.mark.parametrize("N", [2 ** 10, 2 ** 14, 2 ** 18])
    @pytest.mark.parametrize("nmodes", np.arange(1, 4))
    def testshape(self, ctype, N, nmodes):
        b = ctype(N, nmodes=nmodes)
        assert b.shape == (nmodes, N)

    @pytest.mark.parametrize("ctype", [signals.PRBSBits, signals.RandomBits])
    def testtype(self, ctype):
        c = ctype(100, nmodes=1)
        assert c.dtype == np.bool

    @pytest.mark.parametrize("ctype", [signals.PRBSBits, signals.RandomBits])
    def testdist(self, ctype):
        c = ctype(10 ** 6)
        ones = np.count_nonzero(c)
        assert (ones - 10 ** 6 / 2) < 1000

    def testprbspreserveattr(self):
        c = signals.PRBSBits(1000)
        cc = np.roll(c, 100)
        assert cc._seed == c._seed
        assert cc._order == c._order

    def testrandpreserveattr(self):
        c = signals.RandomBits(1000)
        cc = np.roll(c, 100)
        assert cc._seed == c._seed
        assert cc._rand_state == c._rand_state


class TestQAMSymbolsGray(object):
    @pytest.mark.parametrize("N", [np.random.randint(1, 2 ** 20)])
    @pytest.mark.parametrize("nmodes", np.arange(1, 4))
    def test_shape(self, N, nmodes):
        s = signals.SignalQAMGrayCoded(16, N, nmodes)
        assert s.shape == (nmodes, N)

    @pytest.mark.parametrize("M", [2 ** i for i in range(2, 8)])
    def testavgpow(self, M):
        s = signals.SignalQAMGrayCoded(M, 2 ** 18)
        p = (abs(s) ** 2).mean()
        npt.assert_almost_equal(p, 1, 2)

    @pytest.mark.parametrize("M", [2 ** i for i in range(2, 8)])
    def test_symbols(self, M):
        s = signals.SignalQAMGrayCoded(M, 1000, nmodes=1)
        si = np.unique(s)
        thsyms = theory.cal_symbols_qam(M) / np.sqrt(theory.cal_scaling_factor_qam(M))
        d = np.min(abs(s[0, :, np.newaxis] - thsyms), axis=1)
        assert si.shape[0] == M
        npt.assert_array_almost_equal(d, 0)

    @pytest.mark.parametrize("M", [2 ** i for i in range(2, 8)])
    @pytest.mark.parametrize("prbsseed", np.arange(1, 10))
    def testbits(self, M, prbsseed):
        s = signals.SignalQAMGrayCoded(M, 1000, nmodes=1, seed=[prbsseed])
        npt.assert_array_almost_equal(s.demodulate(s), s.bits)

    @pytest.mark.parametrize("M", [2 ** i for i in range(2, 8)])
    @pytest.mark.parametrize("prbsseed", np.arange(1, 10))
    def testbits2(self, M, prbsseed):
        N = 1000
        s = signals.SignalQAMGrayCoded(M, N, nmodes=1, seed=[prbsseed], bitclass=signals.PRBSBits)
        bitsq = signals.make_prbs_extXOR(s.bits._order[0], N * np.log2(M), prbsseed)
        npt.assert_array_almost_equal(s.demodulate(s)[0], bitsq)

    @pytest.mark.parametrize("M", [2 ** i for i in range(2, 8)])
    def testfromarray_order(self, M):
        a = np.random.choice(theory.cal_symbols_qam(M), 1000)
        s = signals.SignalQAMGrayCoded.from_symbol_array(a)
        assert np.unique(s).shape[0] is M

    @pytest.mark.parametrize("M", [2 ** i for i in range(2, 8)])
    def testfromarray_avgpow(self, M):
        a = np.random.choice(theory.cal_symbols_qam(M), 1000)
        s = signals.SignalQAMGrayCoded.from_symbol_array(a)
        npt.assert_almost_equal((abs(s) ** 2).mean(), (abs(s) ** 2).mean())

    @pytest.mark.parametrize("N", [1024, 12423, 100000, 2 ** 18])
    @pytest.mark.parametrize("M", [2 ** i for i in range(2, 8)])
    def testfrombits_len(self, N, M):
        b = signals.make_prbs_extXOR(15, N)
        s = signals.SignalQAMGrayCoded.from_bit_array(b, M)
        nbit = int(np.log2(M))
        nbitlen = N // nbit
        assert s.shape[1] == nbitlen

    @pytest.mark.parametrize("N", [1024, 12423, 100000, 2 ** 18])
    @pytest.mark.parametrize("M", [2 ** i for i in range(2, 8)])
    def testfrombits_bits(self, N, M):
        b = signals.make_prbs_extXOR(15, N)
        s = signals.SignalQAMGrayCoded.from_bit_array(b, M)
        nbit = int(np.log2(M))
        nbitlen = N // nbit
        npt.assert_almost_equal(s.demodulate(s)[0], b[:nbitlen * nbit])

    @pytest.mark.parametrize("attr", ["_M", "_bits", "_encoding", "_bitmap_mtx",
                                      "_fb", "_code", "_coded_symbols"])
    def test_preserveattr(self, attr):
        s1 = signals.SignalQAMGrayCoded(16, 1000)
        s2 = s1 + 10
        a1 = getattr(s1, attr)
        a2 = getattr(s2, attr)
        if isinstance(a1, np.ndarray):
            npt.assert_array_almost_equal(a1, a2)
        else:
            assert a1 == a2

    def test_symbolpreserve(self):
        s1 = signals.SignalQAMGrayCoded(16, 1000)
        s2 = s1 + 10
        npt.assert_array_almost_equal(s1, s2.symbols)

    def test_symbols_implace_op(self):
        s = signals.SignalQAMGrayCoded(4, 2 ** 12)
        avg1 = (abs(s) ** 2).mean()
        s += 5
        avg2 = (abs(s.symbols) ** 2).mean()
        npt.assert_array_almost_equal(avg1, avg2)

    @pytest.mark.parametrize("os", np.arange(2, 5))
    @pytest.mark.parametrize("nmodes", np.arange(1, 3))
    def test_samplerate(self, os, nmodes):
        N = 1000
        Nn = os * N
        s = signals.SignalQAMGrayCoded(16, N, nmodes=nmodes)
        s = s.resample(os, beta=0.2)
        assert s.shape == (nmodes, Nn)

    @pytest.mark.parametrize("os", np.arange(2, 5))
    @pytest.mark.parametrize("fftconv", [True, False])
    def test_resample(self, os, fftconv):
        N = 1000
        Nn = os * N
        s = signals.SignalQAMGrayCoded(128, N)
        sn = s.resample(os, beta=0.2, renormalise=False, fftconv=fftconv)
        assert sn.fs == os
        assert s.fb ==1
        assert sn.fb == 1

    @pytest.mark.parametrize("fftconv", [True, False])
    @pytest.mark.parametrize("os", np.arange(2, 5))
    @pytest.mark.parametrize("ntaps", [4000, 4001, None]) # taps should hit all filter cases
    def test_resample2(self, fftconv, os, ntaps):
        N = 2**16
        Nn = os * N
        s = signals.SignalQAMGrayCoded(128, N)
        sn = s.resample(os, beta=0.2, renormalise=True, taps=ntaps, fftconv=fftconv)
        si = sn.resample(1, beta=0.2, renormalise=True, taps=ntaps, fftconv=fftconv)
        npt.assert_array_almost_equal(s[10:-10], si[10:-10], 2) # need to not ignore edges due to errors there

    @pytest.mark.parametrize("fftconv", [True, False])
    @pytest.mark.parametrize("ntaps", [4000, 4001, None])
    def test_resample_filter(self, fftconv, ntaps):
        # test to check if filtering is actually working
        # as I had bug where resample did not filter
        N = 2**16
        os = 2
        s = signals.SignalQAMGrayCoded(128, N)
        sn = s.resample(os, beta=0.2, renormalise=True, taps=ntaps, fftconv=fftconv)
        sp = abs(np.fft.fftshift(np.fft.fft(sn[0])))**2
        assert sp[2*N//8] < sp.max()/1000

    @pytest.mark.parametrize("fftconv", [True, False])
    @pytest.mark.parametrize("ntaps", [4000, 4001, None])
    def test_resample_filter2(self, fftconv, ntaps):
        # test to check if filtering is actually working
        # as I had bug where resample did not filter
        N = 2**16
        os = 2
        s = signals.SignalQAMGrayCoded(128, N)
        sn = s.resample(os, beta=0.2, renormalise=True, taps=ntaps, fftconv=fftconv)
        sp = abs(np.fft.fftshift(np.fft.fft(sn[0])))**2
        assert np.mean(sp[0:int(1/8*sp.size)]) < np.mean(sp[int(3/8*sp.size):int(1/2*sp.size)])/1000

    @pytest.mark.parametrize("M", [4, 16, 32, 64])
    def test_scale(self, M):
        N = 1000
        s = signals.SignalQAMGrayCoded(M, N, nmodes=1)
        p = np.mean(abs(s.coded_symbols)**2)
        npt.assert_almost_equal(p, 1)

    def test_recreate_from_np_array(self):
        N = 1000
        N2 = 2*N
        s = signals.SignalQAMGrayCoded(128, N)
        arr = np.arange(1000, dtype=np.complex128)
        s2 = s.recreate_from_np_array(arr)
        assert type(s) is type(s2)

    def test_recreate_from_np_array_attr(self):
        N = 1000
        N2 = 2*N
        s = signals.SignalQAMGrayCoded(128, N)
        arr = np.arange(1000, dtype=np.complex128)
        s2 = s.recreate_from_np_array(arr)
        for attr in s._inheritattr_:
            assert getattr(s, attr) is getattr(s2, attr)

    def test_recreate_from_np_array_attr2(self):
        N = 1000
        N2 = 2*N
        s = signals.SignalQAMGrayCoded(128, N)
        arr = np.arange(1000, dtype=np.complex128)
        s2 = s.recreate_from_np_array(arr)
        for attr in s._inheritbase_:
            assert getattr(s, attr) is getattr(s2, attr)

    @pytest.mark.parametrize("attr", [("_fs", 2), ("fs", 2), ("symbols", np.arange(10)), ("bla", "a")])
    def test_recreate_from_np_array_attr3(self, attr):
        N = 1000
        N2 = 2*N
        s = signals.SignalQAMGrayCoded(128, N)
        arr = np.arange(1000, dtype=np.complex128)
        s2 = s.recreate_from_np_array(arr, **dict([attr]))
        a = getattr(s2, attr[0])
        assert a is attr[1]

    @pytest.mark.parametrize("attr", [("_fs", 2), ("_symbols", np.arange(10))])
    def test_recreate_from_np_array_attr4(self, attr):
        N = 1000
        N2 = 2*N
        s = signals.SignalQAMGrayCoded(128, N)
        arr = np.arange(1000, dtype=np.complex128)
        s2 = s.recreate_from_np_array(arr, **dict([attr]))
        a = getattr(s2, attr[0].strip("_"))
        assert a is attr[1]

    def test_recreate_from_np_array_attr5(self):
        N = 1000
        N2 = 2*N
        s = signals.SignalQAMGrayCoded(128, N)
        arr = np.arange(1000, dtype=np.complex128)
        s2 = s.recreate_from_np_array(arr, fs=4)
        a = s2.fs
        assert a is 4

    @pytest.mark.parametrize("factor", [0.5, 1, 2])
    def test_recreate_from_np_array_shape(self, factor):
        N = 1000
        N2 = factor*N
        s = signals.SignalQAMGrayCoded(128, N)
        arr = np.arange(N2, dtype=np.complex128)
        s2 = s.recreate_from_np_array(arr)
        assert s2.shape[0] == N2

    @pytest.mark.parametrize("nmodes", [1, 2, 3])
    @pytest.mark.parametrize("ndims", [1, 2, 3])
    def test_recreate_from_np_array_shape2(self, nmodes, ndims):
        N = 1000
        s = signals.SignalQAMGrayCoded(128, N, nmodes=nmodes)
        arr = np.arange(ndims*N, dtype=np.complex128).reshape(ndims, N)
        s2 = s.recreate_from_np_array(arr)
        assert s2.shape == (ndims, N)

    @pytest.mark.parametrize("dtype", [np.float32, np.int, np.complex64, np.complex128])
    def test_recreate_from_np_array_dtype(self, dtype):
        N = 1000
        N2 = 2*N
        s = signals.SignalQAMGrayCoded(128, N)
        arr = np.arange(1000, dtype=dtype)
        s2 = s.recreate_from_np_array(arr)
        assert type(s) is type(s2)
        assert s2.dtype is np.dtype(dtype)




class TestPilotSignal(object):
    @pytest.mark.parametrize("N", [2 ** 18, 2 ** 12, 2 ** 14])
    @pytest.mark.parametrize("nmodes", range(1, 4))
    def testshape(self, N, nmodes):
        s = signals.SignalWithPilots(128, N, 256, 32, 1, nmodes=nmodes)
        assert s.shape[1] == N and s.shape[0] == nmodes

    @pytest.mark.parametrize("N", [1, 123, 256, 534])
    def testseqlen(self, N):
        QPSK = signals.SignalQAMGrayCoded(4, 200)
        s = signals.SignalWithPilots(128, 2 ** 18, N, 0, 1)
        dist = abs(s[0, :N, np.newaxis] - QPSK.coded_symbols)
        npt.assert_array_almost_equal(np.min(dist, axis=1), 0)

    @pytest.mark.parametrize("N", [1, 2, 32, 64, 128])
    def testphpilots(self, N):
        QPSK = signals.SignalQAMGrayCoded(4, 200)
        s = signals.SignalWithPilots(128, 2 ** 18, 0, N, 1)
        dist = abs(s[0, ::N, np.newaxis] - QPSK.coded_symbols)
        npt.assert_array_almost_equal(np.min(dist, axis=1), 0)

    @pytest.mark.parametrize("Nseq", [2, 32, 64, 128])
    @pytest.mark.parametrize("ph_i", [2, 32, 64, 128])
    def test_symbols(self, Nseq, ph_i):
        ph_fr = 50
        s = signals.SignalWithPilots(128, ph_fr * ph_i + Nseq, Nseq, ph_i, 1)
        idx, idx_d, idx_p = signals.SignalWithPilots._cal_pilot_idx(ph_fr * ph_i + Nseq, Nseq, ph_i)
        npt.assert_array_almost_equal(s[:, idx_d], s.symbols)

    @pytest.mark.parametrize("N", np.arange(2, 5))
    def testframes_shape(self, N):
        s = signals.SignalWithPilots(128, 2 ** 16, 128, 32, nframes=N)
        assert 2 ** 16 * N == s.shape[1]

    @pytest.mark.parametrize("N", np.arange(2, 5))
    def testframes_data(self, N):
        flen = 2 ** 16
        s = signals.SignalWithPilots(128, flen, 128, 32, nframes=N)
        for i in range(1, N):
            npt.assert_array_almost_equal(s[:, :flen], s[:, i * flen:(i + 1) * flen])

    @pytest.mark.parametrize("N", [1, 123, 256, 534])
    def test_from_data_seqlen(self, N):
        QPSK = signals.SignalQAMGrayCoded(4, 200)
        data = signals.SignalQAMGrayCoded(128, 2 ** 12)
        s = signals.SignalWithPilots.from_data_array(data, 2 ** 12, N, 0, 1)
        dist = abs(s[0, :N, np.newaxis] - QPSK.coded_symbols)
        npt.assert_array_almost_equal(np.min(dist, axis=1), 0)

    @pytest.mark.parametrize("N", [1, 2, 32, 64, 128])
    def test_from_data_phpilots(self, N):
        QPSK = signals.SignalQAMGrayCoded(4, 200)
        data = signals.SignalQAMGrayCoded(128, 2 ** 12)
        s = signals.SignalWithPilots.from_data_array(data, 2 ** 12, 0, N, 1)
        dist = abs(s[0, ::N, np.newaxis] - QPSK.coded_symbols)
        npt.assert_array_almost_equal(np.min(dist, axis=1), 0)

    def test_from_data_symbols(self):
        data = signals.SignalQAMGrayCoded(128, 2 ** 12)
        s = signals.SignalWithPilots.from_data_array(data, 2 ** 12, 128, 16, 1)
        npt.assert_array_almost_equal(data[:, :s.symbols.shape[1]], s.symbols)

    def test_from_data_symbols2(self):
        data = signals.SignalQAMGrayCoded(128, 2 ** 12)
        idx, idx_d, idx_p = signals.SignalWithPilots._cal_pilot_idx(2 ** 12, 128, 16)
        s = signals.SignalWithPilots.from_data_array(data, 2 ** 12, 128, 16, 1)
        npt.assert_array_almost_equal(s[:, idx_d], data[:, :np.count_nonzero(idx_d)])

    @pytest.mark.parametrize("p_ins", [0, 1, 16, 32])
    def test_from_data_symbols3(self, p_ins):
        data = signals.SignalQAMGrayCoded(128, 2 ** 12 - 128)
        s = signals.SignalWithPilots.from_data_array(data, 2 ** 12, 128, p_ins, 1)
        assert s.shape[1] == 2 ** 12

    @pytest.mark.parametrize("os", np.arange(2, 5))
    def test_resample(self, os):
        N = 2 ** 12 - 128
        Nn = os * N
        s = signals.SignalQAMGrayCoded(128, N)
        sn = signals.SignalWithPilots.from_data_array(s, 2 ** 12, 128, None, 1)
        si = sn.resample(2, beta=0.2, renormalise=False)
        assert si.shape[0] == sn.shape[0]
        assert si.shape[1] == 2 * sn.shape[1]
        npt.assert_array_almost_equal(s, si.symbols)

    @pytest.mark.parametrize("Nseq", [64, 128])
    @pytest.mark.parametrize("ph_i", [2, 32, 64])
    @pytest.mark.parametrize("nframes", [1, 2, 3, 4])
    def test_get_data(self, Nseq, ph_i, nframes):
        N = 2 ** 16
        s = signals.SignalWithPilots(64, N, Nseq, ph_i, nframes=nframes)
        npt.assert_array_almost_equal(s.get_data(), np.tile(s.symbols, nframes))

    def test_symbol_inherit(self):
        s = signals.SignalQAMGrayCoded(128, 2 ** 16, nmodes=2)
        sp = signals.SignalWithPilots.from_data_array(s, 2 ** 16, 256, 32)
        npt.assert_array_almost_equal(sp.symbols, sp.get_data())

    def test_symbol_inherit_shape(self):
        s = signals.SignalQAMGrayCoded(128, 2 ** 16, nmodes=2)
        sp = signals.SignalWithPilots.from_data_array(s, 2 ** 16, 256, 32)
        N = 2**16-256
        ph = N//32
        NN = N-ph
        assert sp.symbols.shape[1] == NN

    def test_symbol_inherit2(self):
        sp = signals.SignalWithPilots(128, 2 ** 16, 256, 32, nmodes=2)
        npt.assert_array_almost_equal(sp.symbols, sp.get_data())

    def test_symbol_inherit_shape2(self):
        sp = signals.SignalWithPilots(128, 2 ** 16, 256, 32, nmodes=2)
        N = 2**16-256
        ph = N//32
        NN = N-ph
        assert sp.symbols.shape[1] == NN


class TestTDHybridsSymbols(object):
    @pytest.mark.parametrize("M1", [4, 16, 32, 64, 128, 256])
    @pytest.mark.parametrize("M2", [4, 16, 32, 64, 128, 256])
    def test_dist(self, M1, M2):
        s = signals.TDHQAMSymbols((M1, M2), 1000, fr=0.5)
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
        r = r1 + r2
        o = signals.TDHQAMSymbols((16, 4), 1000, fr=r2 / r)
        for i in range(r):
            s = o[0, i::r]
            if i % r < r1:
                d = np.min(abs(s[:, np.newaxis] - o._symbols_M1), axis=1)
                npt.assert_array_almost_equal(d, 0)
            else:
                d = np.min(abs(s[:, np.newaxis] - o._symbols_M2), axis=1)
                npt.assert_array_almost_equal(d, 0)

    def testclass(self):
        s = signals.TDHQAMSymbols((16, 4), 1000)
        type(s._symbols_M1) is signals.SignalQAMGrayCoded
        type(s._symbols_M2) is signals.SignalQAMGrayCoded

    @pytest.mark.parametrize("r1", np.arange(1, 3))
    @pytest.mark.parametrize("r2", np.arange(1, 3))
    def test_from_arrays_shape(self, r1, r2):
        import math
        if math.gcd(r1, r2) > 1:
            assert True
            return
        r = r1 + r2
        s1 = signals.SignalQAMGrayCoded(16, 1000 * r1)
        s2 = signals.SignalQAMGrayCoded(4, 1000 * r2)
        o = signals.TDHQAMSymbols.from_symbol_arrays(s1, s2, r2 / r)
        assert o.shape == (1, 1000 * (r1 + r2))

    @pytest.mark.parametrize("r1", np.arange(1, 3))
    @pytest.mark.parametrize("r2", np.arange(1, 3))
    def test_from_arrays_ratio(self, r1, r2):
        import math
        if math.gcd(r1, r2) > 1:
            assert True
            return
        r = r1 + r2
        s1 = signals.SignalQAMGrayCoded(16, 1000 * r1, seed=[1, 2])
        s2 = signals.SignalQAMGrayCoded(4, 1000 * r2, seed=[1, 2])
        o = signals.TDHQAMSymbols.from_symbol_arrays(s1, s2, r2 / r)
        o2 = signals.TDHQAMSymbols((16, 4), 1000 * (r1 + r2), fr=r2 / r, seed=[1, 2])
        npt.assert_array_almost_equal(o, o2)

    @pytest.mark.parametrize("attr", ["fb", "M", "f_M1", "f_M2", "f_M", "fr"])
    def test_preserveattr(self, attr):
        s1 = signals.TDHQAMSymbols((16, 4), 1000, 0.5)
        s2 = s1 + 10
        a1 = getattr(s1, attr)
        a2 = getattr(s2, attr)
        if isinstance(a1, np.ndarray):
            npt.assert_array_almost_equal(a1, a2)
        else:
            assert a1 == a2


class TestSignal(object):
    @pytest.mark.parametrize("attr", ["fb", "M", "fs", "symbols"])
    def test_preserveattr(self, attr):
        s1 = signals.ResampledQAM(16, 1000, fs=2)
        s2 = s1 + 10
        a1 = getattr(s1, attr)
        a2 = getattr(s2, attr)
        if isinstance(a1, np.ndarray):
            npt.assert_array_almost_equal(a1, a2)
        else:
            assert a1 == a2

    @pytest.mark.parametrize("os", np.arange(2, 5))
    @pytest.mark.parametrize("nmodes", np.arange(1, 3))
    def test_samplerate(self, os, nmodes):
        N = 1000
        Nn = os * N
        s = signals.ResampledQAM(16, N, fs=os, nmodes=nmodes)
        assert s.shape == (nmodes, Nn)

    @pytest.mark.parametrize("os", np.arange(2, 5))
    def test_resample(self, os):
        N = 1000
        Nn = os * N
        s = signals.SignalQAMGrayCoded(128, N)
        sn = signals.ResampledQAM.from_symbol_array(s, fs=os, beta=0.2, renormalise=True)
        si = sn.resample(1, beta=0.2, renormalise=True)
        #si /= abs(si).max()
        #s /= abs(s).max()
        npt.assert_array_almost_equal(s[10:-10], si[10:-10])

    @pytest.mark.parametrize("os", np.arange(2, 5))
    def test_from_symbol_array(self, os):
        N = 1000
        Nn = os * N
        s = signals.SignalQAMGrayCoded(128, N)
        sn = signals.ResampledQAM.from_symbol_array(s, fs=os, beta=0.2, renormalise=False)
        assert sn.shape[1] == s.shape[1] * os

    def test_symbols_implace_op(self):
        s = signals.ResampledQAM(4, 2 ** 12)
        avg1 = (abs(s.symbols) ** 2).mean()
        s += 5
        avg2 = (abs(s.symbols) ** 2).mean()
        npt.assert_array_almost_equal(avg1, avg2)

    def test_symbolinherit(self):
        N = 1000
        s = signals.SignalQAMGrayCoded(128, N)
        sn = signals.ResampledQAM.from_symbol_array(s, fs=2, beta=0.2, renormalise=False)
        npt.assert_array_almost_equal(s, sn.symbols)
        #assert sn.symbols is s

    def test_symbolinherit2(self):
        N = 1000
        s = signals.SignalQAMGrayCoded(128, N)
        sn = signals.ResampledQAM.from_symbol_array(s, fs=2, beta=0.2, renormalise=False)
        sn2 = sn + 2
        #assert sn2.symbols is s
        npt.assert_array_almost_equal(s, sn2.symbols)

    def test_symbolinherit3(self):
        N = 1000
        s = signals.ResampledQAM(16, N, fs=2)
        sn = s.resample(1, beta=0.2)
        npt.assert_array_almost_equal(s.symbols, sn.symbols)
        #assert sn.symbols is s.symbols

    @pytest.mark.parametrize("attr", ["M", "bits", "_encoding", "_bitmap_mtx",
                                      "fb", "_code", "coded_symbols"])
    def test_symbol_attr(self, attr):
        s = signals.ResampledQAM(16, 2000, fs=2)
        a = getattr(s, attr)
        assert a is not None

class TestSignalQualityCorrectnes(object):
    @pytest.mark.parametrize("err_syms", np.arange(1, 100, 10))
    def test_ser_calculation(self, err_syms):
        N = 1000
        s = signals.SignalQAMGrayCoded(64, N, nmodes=1)
        d = np.diff(np.unique(s.coded_symbols.real)).min()
        ii = np.arange(1,err_syms+1)
        s2 = _flip_symbols(s, ii, d)
        ser = s.cal_ser()
        npt.assert_almost_equal(ser.flatten(), err_syms/N)

    @pytest.mark.parametrize("err_syms", np.arange(1, 100, 10))
    def test_ser_calculation2(self, err_syms):
        N = 1000
        s = signals.SignalQAMGrayCoded(64, N, nmodes=1)
        d = np.diff(np.unique(s.coded_symbols.real)).min()
        ii = np.arange(1,err_syms+1)
        s2 = _flip_symbols(s, ii, 2*d)
        ser = s.cal_ser()
        npt.assert_almost_equal(ser.flatten(), err_syms/N)

    @pytest.mark.parametrize("err_syms", np.arange(1, 100, 10))
    def test_ber_calculation(self, err_syms):
        N = 1000
        M = 64
        s = signals.SignalQAMGrayCoded(M, N, nmodes=1)
        d = np.diff(np.unique(s.coded_symbols.real)).min()
        ii = np.arange(1,err_syms+1)
        s2 = _flip_symbols(s, ii, d)
        ber = s.cal_ber()
        npt.assert_almost_equal(ber.flatten(), err_syms/(N*np.log2(M)))


class TestSignalQualityOnSignal(object):

    @pytest.mark.parametrize("nmodes", np.arange(1, 4))
    def test_ser_shape(self, nmodes):
        s = signals.ResampledQAM(16, 2 ** 16, nmodes=nmodes)
        ser = s.cal_ser()
        assert ser.shape[0] == nmodes

    def test_ser_value(self):
        s = signals.ResampledQAM(16, 2 ** 16)
        ser = s.cal_ser()
        assert ser[0] == 0

    @pytest.mark.parametrize("nmodes", np.arange(1, 4))
    def test_evm_shape(self, nmodes):
        s = signals.ResampledQAM(16, 2 ** 16, nmodes=nmodes)
        evm = s.cal_evm()
        assert evm.shape[0] == nmodes

    def test_evm_value(self):
        s = signals.ResampledQAM(16, 2 ** 16)
        evm = s.cal_evm()
        assert evm[0] < 1e-4

    @pytest.mark.parametrize("nmodes", np.arange(1, 4))
    def test_ber_shape(self, nmodes):
        s = signals.ResampledQAM(16, 2 ** 16, nmodes=nmodes)
        ber = s.cal_ber()
        assert ber.shape[0] == nmodes

    def test_ber_value(self):
        s = signals.ResampledQAM(16, 2 ** 16)
        s += 0.05 * (np.random.randn(2 ** 16) + 1.j * np.random.randn(2 ** 16))
        ber = s.cal_ber()
        assert ber[0] == 0

    @pytest.mark.parametrize("nmodes", np.arange(1, 4))
    def test_est_snr_shape(self, nmodes):
        s = signals.ResampledQAM(16, 2 ** 16, nmodes=nmodes)
        snr = s.est_snr()
        assert snr.shape[0] == nmodes

    def test_est_snr_value(self):
        s = signals.ResampledQAM(16, 2 ** 16)
        snr = s.est_snr()
        assert snr[0] > 1e25

    @pytest.mark.parametrize("nmodes", np.arange(1, 4))
    def test_cal_gmi_shape(self, nmodes):
        s = signals.ResampledQAM(16, 2 ** 16, nmodes=nmodes)
        gmi, gmi_pb = s.cal_gmi()
        assert gmi.shape[0] == nmodes

    @pytest.mark.parametrize("M", [16, 128, 256])
    def test_cal_gmi_value(self, M):
        s = signals.ResampledQAM(M, 2 ** 16)
        s += 0.0004 * (np.random.randn(2 ** 16) + 1.j * np.random.randn(2 ** 16))
        nbits = np.log2(M)
        gmi, gmi_pb = s.cal_gmi()
        npt.assert_almost_equal(gmi[0], nbits)


class TestPilotSignalQualityOnSignal(object):

    @pytest.mark.parametrize("nmodes", np.arange(1, 4))
    def test_ser_shape(self, nmodes):
        s = signals.SignalWithPilots(16, 2 ** 16, 128, 32, nmodes=nmodes)
        ser = s.cal_ser()
        assert ser.shape[0] == nmodes

    @pytest.mark.parametrize("nframes", np.arange(1, 4))
    def test_ser_value(self, nframes):
        s = signals.SignalWithPilots(16, 2 ** 16, 128, 32, nmodes=1, nframes=nframes)
        ser = s.cal_ser()
        assert ser[0] == 0

    @pytest.mark.parametrize("nmodes", np.arange(1, 4))
    def test_evm_shape(self, nmodes):
        s = signals.SignalWithPilots(16, 2 ** 16, 128, 32, nmodes=nmodes)
        evm = s.cal_evm()
        assert evm.shape[0] == nmodes

    @pytest.mark.parametrize("nframes", np.arange(1, 4))
    def test_evm_value(self, nframes):
        s = signals.SignalWithPilots(16, 2 ** 16, 128, 32, nmodes=1, nframes=nframes)
        evm = s.cal_evm()
        assert evm[0] < 1e-4

    @pytest.mark.parametrize("nmodes", np.arange(1, 4))
    def test_ber_shape(self, nmodes):
        s = signals.SignalWithPilots(16, 2 ** 16, 128, 32, nmodes=nmodes)
        ber = s.cal_ber()
        assert ber.shape[0] == nmodes

    @pytest.mark.parametrize("nframes", np.arange(1, 4))
    def test_ber_value(self, nframes):
        s = signals.SignalWithPilots(16, 2 ** 16, 128, 32, nmodes=1, nframes=nframes)
        s += 0.05 * (np.random.randn(2 ** 16*nframes) + 1.j * np.random.randn(2 ** 16*nframes))
        ber = s.cal_ber()
        assert ber[0] == 0

    @pytest.mark.parametrize("nmodes", np.arange(1, 4))
    def test_est_snr_shape(self, nmodes):
        s = signals.SignalWithPilots(16, 2 ** 16, 128, 32, nmodes=nmodes)
        snr = s.est_snr()
        assert snr.shape[0] == nmodes

    @pytest.mark.parametrize("nframes", np.arange(1, 4))
    def test_est_snr_value(self, nframes):
        s = signals.SignalWithPilots(16, 2 ** 16, 128, 32, nmodes=1, nframes=nframes)
        snr = s.est_snr()
        assert snr[0] > 1e25

    @pytest.mark.parametrize("nmodes", np.arange(1, 4))
    def test_cal_gmi_shape(self, nmodes):
        s = signals.SignalWithPilots(16, 2 ** 16, 128, 32, nmodes=nmodes)
        gmi, gmi_pb = s.cal_gmi()
        assert gmi.shape[0] == nmodes

    @pytest.mark.parametrize("M", [16, 128, 256])
    @pytest.mark.parametrize("nframes", np.arange(1, 4))
    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_cal_gmi_value(self, M, nframes, dtype):
        s = signals.SignalWithPilots(M, 2 ** 16, 128, 32, nmodes=1, nframes=nframes, dtype=dtype)
        s += (0.0004 * (np.random.randn(2 ** 16*nframes) + 1.j * np.random.randn(2 ** 16*nframes))).astype(dtype)
        nbits = np.log2(M)
        gmi, gmi_pb = s.cal_gmi()
        npt.assert_almost_equal(gmi[0], nbits)

class TestSymbolOnlySignal(object):
    @pytest.mark.parametrize("nmodes", np.arange(1, 4))
    @pytest.mark.parametrize("N", [2**i for i in np.arange(10, 14)])
    def test_shape(self, nmodes, N):
        syms = theory.cal_symbols_qam(32)
        s = signals.SymbolOnlySignal(32, N, syms, nmodes=nmodes)
        assert s.shape == (nmodes, N)

    def test_syms_from_set(self):
        syms = theory.cal_symbols_qam(32)
        s = signals.SymbolOnlySignal(32, 20, syms, nmodes=1)
        d = np.min(abs(s[0, :, np.newaxis] - syms), axis=1)
        npt.assert_almost_equal(0, d)

    def test_from_array_class(self):
        syms = theory.cal_symbols_qam(32)
        symsN = np.random.choice(syms, (1, 1000))
        s = signals.SymbolOnlySignal(32, 20, syms, nmodes=1)
        s2 = signals.SymbolOnlySignal.from_symbol_array(symsN)
        assert type(s) is type(s2)

    @pytest.mark.parametrize("M", [16, 32, 64, 128, 256])
    def test_from_array_coded_symbols(self, M):
        syms = theory.cal_symbols_qam(M)
        symsN = np.random.choice(syms, (1, 1000))
        s = signals.SymbolOnlySignal.from_symbol_array(symsN)
        d = np.min(abs(s.coded_symbols[:, np.newaxis] - syms), axis=1)
        npt.assert_almost_equal(0, d)

    @pytest.mark.parametrize("M", [16, 32, 64, 128, 256])
    def test_from_array_coded_symbols2(self, M):
        syms = theory.cal_symbols_qam(M)
        symsN = np.random.choice(syms, (1, 4000))
        s = signals.SymbolOnlySignal.from_symbol_array(symsN)
        assert s.coded_symbols.size == M

    def test_from_array_coded_symbols3(self):
        syms = theory.cal_symbols_qam(32)
        symsN = np.random.choice(syms, (1, 1000))
        s = signals.SymbolOnlySignal.from_symbol_array(symsN, coded_symbols=syms)
        npt.assert_array_almost_equal(s.coded_symbols, syms)

class TestDtype(object):
    @pytest.mark.parametrize("dt", [np.complex64, np.complex128])
    def test_SignalQAMGray_dtype(self, dt):
        s = signals.SignalQAMGrayCoded(32, 2**10, dtype=dt)
        assert s.dtype is np.dtype(dt)

    @pytest.mark.parametrize("dt", [np.complex64, np.complex128])
    def test_from_symbol_array(self, dt):
        s = signals.SignalQAMGrayCoded(32, 2**10, dtype=dt)
        ss = signals.SignalQAMGrayCoded.from_symbol_array(s)
        assert ss.dtype is s.dtype


