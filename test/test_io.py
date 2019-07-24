import pytest
import numpy as np
import numpy.testing as npt
import os
import tempfile
from scipy.io import loadmat, savemat

from qampy import io, signals


class Testsave(object):
    @pytest.mark.parametrize("lvl", np.arange(1,6))
    def test_file_exits(self, lvl):
        tmpdir = tempfile.mkdtemp()
        fn = os.path.join(tmpdir, "tmp")
        sig = signals.SignalQAMGrayCoded(4, 2**12, nmodes=1)
        io.save_signal(fn, sig, lvl)
        assert os.path.isfile(fn)

    @pytest.mark.parametrize("nmodes", np.arange(1,6))
    @pytest.mark.parametrize("lvl", np.arange(1,6))
    def test_compare_to_load(self, nmodes, lvl):
        tmpdir = tempfile.mkdtemp()
        fn = os.path.join(tmpdir, "tmp")
        sig = signals.SignalQAMGrayCoded(4, 2**12, nmodes=1)
        io.save_signal(fn, sig, lvl)
        sigld = io.load_signal(fn)
        npt.assert_array_almost_equal(sig, sigld)

    @pytest.mark.parametrize("lvl", np.arange(1,6))
    def test_compare_to_load(self, lvl):
        tmpdir = tempfile.mkdtemp()
        fn = os.path.join(tmpdir, "tmp")
        sig = signals.SignalQAMGrayCoded(4, 2**12, nmodes=1)
        io.save_signal(fn, sig, lvl)
        sigld = io.load_signal(fn)
        for attr in ['fb', 'M', "fs"]:
            assert getattr(sig, attr) == getattr(sigld, attr)

class TestMatIO(object):
    @pytest.mark.parametrize("nmodes", np.arange(1,4))
    def test_load_single_key(self, nmodes):
        sig = signals.SignalQAMGrayCoded(16, 2**16, nmodes, fb=20e9)
        tmpdir = tempfile.mkdtemp()
        fn = os.path.join(tmpdir, "tmp")
        savemat(fn, {"sig":sig.symbols})
        sigout = io.load_symbols_from_matlab_file(fn, sig.M, (("sig",),), fb=sig.fb, normalise=False)
        assert sig.fb == sigout.fb
        assert sig.M == sigout.M
        npt.assert_almost_equal(sig, sigout)

    @pytest.mark.parametrize("nmodes", np.arange(1,4))
    def test_load_single_key_dim2cmplx(self, nmodes):
        sig = signals.SignalQAMGrayCoded(16, 2**16, nmodes, fb=20e9)
        tmpdir = tempfile.mkdtemp()
        dat = []
        for i in range(nmodes):
            dat.append(sig.symbols[i].real)
            dat.append(sig.symbols[i].imag)
        portmap = np.arange(2*nmodes)
        portmap = portmap.reshape(-1, 2)
        fn = os.path.join(tmpdir, "tmp")
        savemat(fn, {"sig":dat})
        sigout = io.load_symbols_from_matlab_file(fn, sig.M, (("sig",),), fb=sig.fb,
                                                  normalise=False, dim2cmplx=True, portmap=portmap)
        assert sig.fb == sigout.fb
        assert sig.M == sigout.M
        npt.assert_almost_equal(sig, sigout)

    @pytest.mark.parametrize("nmodes", np.arange(1,4))
    def test_load_key_per_dim(self, nmodes):
        sig = signals.SignalQAMGrayCoded(16, 2**16, nmodes, fb=20e9)
        tmpdir = tempfile.mkdtemp()
        fn = os.path.join(tmpdir, "tmp")
        dat = {}
        keys = []
        for i in range(nmodes):
            dat["sig_{}".format(i)] = sig.symbols[i]
            keys.append(("sig_{}".format(i),))
        savemat(fn, dat)
        sigout = io.load_symbols_from_matlab_file(fn, sig.M, keys, fb=sig.fb, normalise=False)
        assert sig.fb == sigout.fb
        assert sig.M == sigout.M
        assert sig.shape == sigout.shape
        npt.assert_almost_equal(sig, sigout)

    @pytest.mark.parametrize("nmodes", np.arange(1,4))
    def test_load_real_imag(self, nmodes):
        sig = signals.SignalQAMGrayCoded(16, 2**16, nmodes, fb=20e9)
        tmpdir = tempfile.mkdtemp()
        fn = os.path.join(tmpdir, "tmp")
        savemat(fn, {"sig_r":sig.symbols.real, "sig_i":sig.symbols.imag})
        sigout = io.load_symbols_from_matlab_file(fn, sig.M, (("sig_r","sig_i"),), fb=sig.fb, normalise=False)
        assert sig.fb == sigout.fb
        assert sig.M == sigout.M
        npt.assert_almost_equal(sig, sigout)

    @pytest.mark.parametrize("nmodes", np.arange(1,4))
    def test_load_key_per_dim_real_imag(self, nmodes):
        sig = signals.SignalQAMGrayCoded(16, 2**16, nmodes, fb=20e9)
        tmpdir = tempfile.mkdtemp()
        fn = os.path.join(tmpdir, "tmp")
        dat = {}
        keys = []
        for i in range(nmodes):
            dat["sig_{}_r".format(i)] = sig.symbols[i].real
            dat["sig_{}_i".format(i)] = sig.symbols[i].imag
            keys.append(("sig_{}_r".format(i),"sig_{}_i".format(i)))
        savemat(fn, dat)
        sigout = io.load_symbols_from_matlab_file(fn, sig.M, keys, fb=sig.fb, normalise=False)
        assert sig.fb == sigout.fb
        assert sig.M == sigout.M
        assert sig.shape == sigout.shape
        npt.assert_almost_equal(sig, sigout)

    @pytest.mark.parametrize("nmodes", np.arange(1,4))
    def test_create_signal_from_matlab(self, nmodes):
        sig = signals.SignalQAMGrayCoded(16, 2**16, nmodes, fb=20e9)
        sig2 = sig.resample(2*sig.fb, beta=0.1)
        tmpdir = tempfile.mkdtemp()
        fn = os.path.join(tmpdir, "tmp")
        savemat(fn, {"data":sig2})
        sigout = io.create_signal_from_matlab(sig, fn, 2*sig.fs, (("data",),))
        assert sig2.fs == sigout.fs
        assert sig2.M == sigout.M
        assert sig2.shape == sigout.shape
        npt.assert_almost_equal(sig2, sigout)
