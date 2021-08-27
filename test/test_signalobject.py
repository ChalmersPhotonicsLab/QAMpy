import pytest
import sys
import numpy as np
import numpy.testing as npt

from qampy import signals, theory, impairments, equalisation, core


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
    @pytest.mark.parametrize("attr", ["M", "fs", "fb", "bits", "coded_symbols", "_encoding",
                                      "_bitmap_mtx", "_code", "Nbits"])
    def test_attr_present(self, attr):
        s = signals.ResampledQAM(128, 2**12)
        assert getattr(s, attr) is not None

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
        s = signals.SignalQAMGrayCoded.from_symbol_array(a, M=M)
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

    @pytest.mark.parametrize("fs", [160e9, 80e9])
    #@pytest.mark.parametrize("fftconv", [True, False])
    @pytest.mark.parametrize("fftconv", [True, False])
    @pytest.mark.parametrize("beta", [None, 0.2])
    def test_resample_freq(self, fs, fftconv, beta):
        N = 2**18
        s = signals.SignalQAMGrayCoded(128, N, fb=35e9)
        sn = s.resample(fs, beta=beta, renormalise=False, fftconv=fftconv)
        npt.assert_allclose(sn.fs, fs)

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

    def test_pickle(self):
        import tempfile
        import pickle
        dr = tempfile.TemporaryDirectory()
        fp = open(dr.name+"/data.pic", "wb")
        s = signals.SignalQAMGrayCoded(128, 2**12, fb=10e9, nmodes=2)
        pickle.dump(s, fp)
        fp.close()
        fp2 = open(dr.name+"/data.pic", "rb")
        s2 = pickle.load(fp2)
        fp2.close()
        dr.cleanup()
        npt.assert_array_almost_equal(s,s2)
        npt.assert_array_almost_equal(s.symbols,s2.symbols)
        npt.assert_array_almost_equal(s.bits, s2.bits)
        npt.assert_array_almost_equal(s.coded_symbols, s2.coded_symbols)
        assert s.fb == s2.fb
        assert s.fs == s2.fs
        assert s.M == s2.M

    @pytest.mark.parametrize("Nmodes", [2, 3, 4, 6])
    @pytest.mark.parametrize("M", [ 4, 16])
    def test_sync_reorder_modes(self, Nmodes, M):
        sig = signals.SignalQAMGrayCoded(M, 2**16, nmodes=Nmodes)
        sig = impairments.change_snr(sig, M+10)
        idx = np.arange(Nmodes)
        np.random.shuffle(idx)
        s2 = sig[idx,:]
        ser = s2.cal_ser()
        npt.assert_array_almost_equal(ser, 0)



class TestResampledQAM(object):
    @pytest.mark.parametrize("attr", ["M", "fs", "fb", "bits", "coded_symbols", "_encoding",
                                      "_bitmap_mtx", "_code", "Nbits"])
    def test_attr_present(self, attr):
        s = signals.ResampledQAM(128, 2**12)
        assert getattr(s, attr) is not None


class TestPilotSignal(object):

    @pytest.mark.parametrize("attr", ["M", "fs", "fb", "pilots", "symbols", "pilot_seq",
                                      "ph_pilots", "nframes", "frame_len", "pilot_scale"])
    def test_attr_present(self, attr):
        s = signals.SignalWithPilots(128, 2**12, 256, 32, 1, nmodes=1)
        assert getattr(s, attr) is not None

    @pytest.mark.parametrize("attr", ["M", "fs", "fb", "pilots", "symbols", "pilot_seq",
                                      "ph_pilots", "nframes", "frame_len", "pilot_scale"])
    def test_attr_present_from_data(self, attr):
        si = signals.SignalQAMGrayCoded(128, 2**12)
        s = signals.SignalWithPilots.from_symbol_array(si, 2**12, 256, 32)
        assert getattr(s, attr) is not None

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
        s = signals.SignalWithPilots.from_symbol_array(data, 2 ** 12, N, 0)
        dist = abs(s[0, :N, np.newaxis] - QPSK.coded_symbols)
        npt.assert_array_almost_equal(np.min(dist, axis=1), 0)

    @pytest.mark.parametrize("N", [1, 2, 32, 64, 128])
    def test_from_data_phpilots(self, N):
        QPSK = signals.SignalQAMGrayCoded(4, 200)
        data = signals.SignalQAMGrayCoded(128, 2 ** 12)
        s = signals.SignalWithPilots.from_symbol_array(data, 2 ** 12, 0, N)
        dist = abs(s[0, ::N, np.newaxis] - QPSK.coded_symbols)
        npt.assert_array_almost_equal(np.min(dist, axis=1), 0)

    def test_from_data_symbols(self):
        data = signals.SignalQAMGrayCoded(128, 2 ** 12)
        s = signals.SignalWithPilots.from_symbol_array(data, 2 ** 12, 128, 16)
        npt.assert_array_almost_equal(data[:, :s.symbols.shape[1]], s.symbols)

    def test_from_data_symbols2(self):
        data = signals.SignalQAMGrayCoded(128, 2 ** 12)
        idx, idx_d, idx_p = signals.SignalWithPilots._cal_pilot_idx(2 ** 12, 128, 16)
        s = signals.SignalWithPilots.from_symbol_array(data, 2 ** 12, 128, 16)
        npt.assert_array_almost_equal(s[:, idx_d], data[:, :np.count_nonzero(idx_d)])

    @pytest.mark.parametrize("p_ins", [0, 1, 16, 32])
    def test_from_data_symbols3(self, p_ins):
        data = signals.SignalQAMGrayCoded(128, 2 ** 12 - 128)
        s = signals.SignalWithPilots.from_symbol_array(data, 2 ** 12, 128, p_ins)
        assert s.shape[1] == 2 ** 12

    @pytest.mark.parametrize("os", np.arange(2, 5))
    def test_resample(self, os):
        N = 2 ** 12 - 128
        Nn = os * N
        s = signals.SignalQAMGrayCoded(128, N)
        sn = signals.SignalWithPilots.from_symbol_array(s, 2 ** 12, 128, None)
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
        sp = signals.SignalWithPilots.from_symbol_array(s, 2 ** 16, 256, 32)
        npt.assert_array_almost_equal(sp.symbols, sp.get_data())

    def test_symbol_inherit_shape(self):
        s = signals.SignalQAMGrayCoded(128, 2 ** 16, nmodes=2)
        sp = signals.SignalWithPilots.from_symbol_array(s, 2 ** 16, 256, 32)
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

    def test_pickle(self):
        import tempfile
        import pickle
        dr = tempfile.TemporaryDirectory()
        fp = open(dr.name+"/data.pic", "wb")
        s = signals.SignalWithPilots(128, 2**16, 256, 32)
        pickle.dump(s, fp)
        fp.close()
        fp2 = open(dr.name+"/data.pic", "rb")
        s2 = pickle.load(fp2)
        fp2.close()
        dr.cleanup()
        npt.assert_array_almost_equal(s,s2)
        npt.assert_array_almost_equal(s.symbols,s2.symbols)
        npt.assert_array_almost_equal(s.pilots, s2.pilots)
        npt.assert_array_almost_equal(s.coded_symbols, s2.coded_symbols)
        npt.assert_array_almost_equal(s.pilot_seq, s2.pilot_seq)
        npt.assert_array_almost_equal(s.ph_pilots, s2.ph_pilots)
        npt.assert_array_almost_equal(s.get_data(), s2.get_data())
        assert s.fb == s2.fb
        assert s.fs == s2.fs
        assert s.M == s2.M
        assert s.pilot_scale == s2.pilot_scale
        assert s.nframes == s2.nframes

class TestTDHybridsSymbols(object):
    @pytest.mark.parametrize("attr", ["M", "fs", "fb", "symbols_M1", "symbols_M2",
                                      "fr", "f_M", "f_M1", "f_M2"])
    def test_attr_present(self, attr):
        s = signals.TDHQAMSymbols((64, 128), 2**12)
        assert getattr(s, attr) is not None

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
    @pytest.mark.skipif(sys.version_info < (3,5),reason="requires python3.5 or above")
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
    @pytest.mark.skipif(sys.version_info < (3,5),reason="requires python3.5 or above")
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
    @pytest.mark.skipif(sys.version_info < (3,5),reason="requires python3.5 or above")
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
    @pytest.mark.parametrize("shift", np.random.randint(-200, 200, 10))
    def test_ser_calculation(self, err_syms, shift):
        N = 1000
        s = signals.SignalQAMGrayCoded(64, N, nmodes=1)
        s = np.roll(s, shift, axis=-1)
        d = np.diff(np.unique(s.coded_symbols.real))
        dmin = d[np.where(d>0)].min()
        ii = np.arange(1,err_syms+1)
        s2 = _flip_symbols(s, ii, dmin)
        ser = s.cal_ser()
        npt.assert_almost_equal(ser.flatten(), err_syms/N)

    @pytest.mark.parametrize("err_syms", np.arange(1, 100, 10))
    def test_ser_calculation2(self, err_syms):
        N = 1000
        s = signals.SignalQAMGrayCoded(64, N, nmodes=1)
        d = np.diff(np.unique(s.coded_symbols.real))
        dmin = d[np.where(d>0)].min()
        ii = np.arange(1,err_syms+1)
        s2 = _flip_symbols(s, ii, 2*dmin)
        ser = s.cal_ser()
        npt.assert_almost_equal(ser.flatten(), err_syms/N)
        
    @pytest.mark.parametrize("M", [4,32,64,256])
    @pytest.mark.parametrize("shift", np.random.randint(-1000,1000, 2))
    @pytest.mark.parametrize("snr", np.linspace(-20, 40, 20))
    def test_ser_vs_theory(self, shift, M, snr):
        from qampy import theory, impairments
        s = signals.SignalQAMGrayCoded(M, 10**5, nmodes=1)
        ser_t = theory.ser_vs_es_over_n0_qam(10**(snr/10), M)
        if ser_t < 1e-4:
            assert True
            return
        s2 = impairments.change_snr(s, snr)
        s2 = np.roll(s2, shift, axis=-1)
        ser = s2.cal_ser()
        npt.assert_allclose(ser, ser_t, rtol=0.4)

    @pytest.mark.parametrize("err_syms", np.arange(1, 100, 10))
    def test_ber_calculation(self, err_syms):
        N = 1000
        M = 64
        s = signals.SignalQAMGrayCoded(M, N, nmodes=1)
        d = np.diff(np.unique(s.coded_symbols.real))
        dmin = d[np.where(d>0)].min()
        ii = np.arange(1,err_syms+1)
        s2 = _flip_symbols(s, ii, dmin)
        ber = s.cal_ber()
        npt.assert_almost_equal(ber.flatten(), err_syms/(N*np.log2(M)))

class TestPilotCalcs(object):
    @pytest.mark.parametrize("ntaps", range(17, 20))
    @pytest.mark.parametrize("shift", np.arange(0, 2)+3000)
    def test_sync_frame(self, ntaps, shift):
        s = signals.SignalWithPilots(128, 2**16, 1024, 32, nframes=3)
        s = s.resample(2*s.fs, beta=0.1)
        s = impairments.change_snr(s, 30)
        s = np.roll(s, shift, axis=-1)
        s.sync2frame(Ntaps=ntaps, adaptive_stepsize=True, mu=1e-2, method="cma", Niter=10)
        shf = shift-ntaps//2
        npt.assert_equal(s.shiftfctrs[0], shf - shf%2)

    @pytest.mark.parametrize("ntaps", range(17, 21))
    @pytest.mark.parametrize("shift", np.arange(0, 2)+3000)
    def test_sync_frame_equalise(self, ntaps, shift):
        s = signals.SignalWithPilots(128, 2**16, 512, 32, nframes=3, nmodes=1)
        s = s.resample(2*s.fs, beta=0.1)
        s = impairments.change_snr(s, 40)
        s = np.roll(s, shift, axis=-1)
        wx1, ret = s.sync2frame(Ntaps=ntaps, returntaps=True, adaptive_stepsize=True, mu=1e-2, method="cma", Niter=10)
        ss = np.roll(s, -s.shiftfctrs[0], axis=-1)
        so = core.equalisation.apply_filter(ss, ss.os, wx1)
        npt.assert_array_equal(np.sign(so[0,:512].imag), np.sign(s.pilot_seq[0].imag))
        npt.assert_array_equal(np.sign(so[0,:512].real), np.sign(s.pilot_seq[0].real))

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

    @pytest.mark.parametrize("nmodes", np.arange(1,5))       #sub_vars[:,i] = np.var(err_out[:,int(-step/os+Ntaps):])arametrize("nmodes", np.arange(1, 4))
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

    @pytest.mark.parametrize("dim", [0,1])
    def test_gmi_single_d(self, dim):
        s = signals.SignalQAMGrayCoded(4, 2**16, nmodes=2)
        gmi = s[dim].cal_gmi()[0]
        npt.assert_almost_equal(gmi, 2)
        
    def test_gmi_single_d_synced(self):
        s = signals.SignalQAMGrayCoded(4, 2**16, nmodes=2)
        gmi = s[0].cal_gmi(synced=True)[0]
        npt.assert_almost_equal(gmi, 2)

    @pytest.mark.parametrize("method", ["cal_ser", "cal_ber", "cal_evm"])
    @pytest.mark.parametrize("dim", [0,1])
    def test_err_single_de(self, method, dim):
        s = signals.SignalQAMGrayCoded(4, 2**16, nmodes=2)
        err = getattr(s[dim], method)()
        assert err < 1e-5
        
    @pytest.mark.parametrize("method", ["cal_ser", "cal_ber", "cal_evm"])
    def test_err_single_d_synced(self, method):
        s = signals.SignalQAMGrayCoded(4, 2**16, nmodes=2)
        err = getattr(s[0], method)(synced=True)
        assert err < 1e-5
 

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

    def test_for_noncontiguous_symbols(self):
        # test for bug #41
        s = signals.SignalQAMGrayCoded(4, 2**10, nmodes=4)
        s2 = s.reshape(1,-1,4)
        ss = signals.SymbolOnlySignal.from_symbol_array(s2[0,:,4:1:-1].T)

class TestDtype(object):
    @pytest.mark.parametrize("dt", [np.complex64, np.complex128])
    def test_SignalQAMGray_dtype(self, dt):
        s = signals.SignalQAMGrayCoded(32, 2**10, dtype=dt)
        assert s.dtype is np.dtype(dt)
        assert np.dtype(dt) is s.symbols.dtype
        assert np.dtype(dt) is s.coded_symbols.dtype

    @pytest.mark.parametrize("dt", [np.complex64, np.complex128])
    def test_from_symbol_array(self, dt):
        s = signals.SignalQAMGrayCoded(32, 2**10, dtype=dt)
        ss = signals.SignalQAMGrayCoded.from_symbol_array(s)
        assert ss.dtype is s.dtype
        assert np.dtype(dt) is s.symbols.dtype
        assert np.dtype(dt) is s.coded_symbols.dtype

    @pytest.mark.parametrize("dt", [np.complex64, np.complex128])
    def testfrombits_bits(self, dt):
        N = 2**12
        M = 16
        b = signals.make_prbs_extXOR(15, N)
        s = signals.SignalQAMGrayCoded.from_bit_array(b, M, dtype=dt)
        assert np.dtype(dt) is s.dtype
        assert np.dtype(dt) is s.symbols.dtype
        assert np.dtype(dt) is s.coded_symbols.dtype

    @pytest.mark.parametrize("dt", [np.complex64, np.complex128])
    def testquantize(self, dt):
        s = signals.SignalQAMGrayCoded(32, 2**10, dtype=dt)
        s2 = impairments.change_snr(s, 30)
        sn = s2.make_decision()
        assert  np.dtype(dt) is sn.dtype
        assert np.dtype(dt) is sn.symbols.dtype
        assert np.dtype(dt) is s.coded_symbols.dtype

    @pytest.mark.parametrize("dt", [np.complex64, np.complex128])
    @pytest.mark.parametrize("N", [0, 40])
    def test_sync_adj1(self, dt, N):
        s = signals.SignalQAMGrayCoded(32, 2**10, dtype=dt)
        s2 = np.roll(s, 102, axis=-1)
        s2 = s2[:,:-N or None]
        tx, rx = s2._sync_and_adjust(s2, s2.symbols)
        assert tx.dtype is np.dtype(dt)
        assert rx.dtype is np.dtype(dt)

    @pytest.mark.parametrize("dt", [np.complex64, np.complex128])
    def test_resample(self, dt):
        s = signals.SignalQAMGrayCoded(32, 2**10, dtype=dt)
        s2 = s.resample(2, beta=0.2, renormalise=True)
        assert np.dtype(dt) is s2.dtype
        assert np.dtype(dt) is s2.symbols.dtype
        assert np.dtype(dt) is s.coded_symbols.dtype

    @pytest.mark.parametrize("dt", [np.complex64, np.complex128])
    def test_tdhqam(self, dt):
        s = signals.TDHQAMSymbols((64, 128), 2**12, dtype=dt)
        assert np.dtype(dt) is s.dtype
        assert np.dtype(dt) is s._symbols_M1.dtype
        assert np.dtype(dt) is s._symbols_M2.dtype

    @pytest.mark.parametrize("dt", [np.complex64, np.complex128])
    def test_tdhqam_from_symbol(self, dt):
        s1 = signals.SignalQAMGrayCoded(64, 2**12, dtype=dt)
        s2 = signals.SignalQAMGrayCoded(32, 2**12, dtype=dt)
        s = signals.TDHQAMSymbols.from_symbol_arrays(s1, s2, 0.5)
        assert np.dtype(dt) is s.dtype

    @pytest.mark.parametrize("dt", [np.complex64, np.complex128])
    @pytest.mark.parametrize("nframes", [1, 2])
    def test_pilot_signal(self, dt, nframes):
        s = signals.SignalWithPilots(32,2**12, 256, 32, nframes=nframes, dtype=dt )
        assert np.dtype(dt) is s.dtype
        assert np.dtype(dt) is s.symbols.dtype
        assert np.dtype(dt) is s.pilots.dtype

    @pytest.mark.parametrize("dt", [np.complex64, np.complex128])
    @pytest.mark.parametrize("nframes", [1, 2])
    def test_pilot_signal_from_data(self, dt, nframes):
        s1 = signals.SignalQAMGrayCoded(32, 2**12, dtype=dt)
        s = signals.SignalWithPilots.from_symbol_array(s1, 2**12, 256, 32, nframes=nframes)
        assert np.dtype(dt) is s.dtype
        assert np.dtype(dt) is s.symbols.dtype
        assert np.dtype(dt) is s.pilots.dtype

    @pytest.mark.parametrize("dt", [np.complex64, np.complex128])
    def test_pilot_signal(self, dt):
        s = signals.SignalWithPilots(32,2**12, 256, 32, dtype=dt )
        s3 = s.get_data()
        assert np.dtype(dt) is s3.dtype
        assert np.dtype(dt) is s3.symbols.dtype













