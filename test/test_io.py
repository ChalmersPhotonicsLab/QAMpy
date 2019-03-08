import pytest
import numpy as np
import numpy.testing as npt
import os
import tempfile

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
