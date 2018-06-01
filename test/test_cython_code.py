import pytest
import numpy as np
import numpy.testing as npt
from qampy import signals, impairments
from qampy.core.equalisation import cython_equalisation


class TestQuantize(object):
    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_dtype(self, dtype):
        s = signals.SignalQAMGrayCoded(32, 2**12, dtype=dtype)
        s = impairments.change_snr(s, 23).astype(dtype)
        o = cython_equalisation.make_decision(s[0], s.coded_symbols)
        assert o.dtype is np.dtype(dtype)

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_dtype(self, dtype):
        s = signals.SignalQAMGrayCoded(32, 2**12, dtype=dtype)
        s = impairments.change_snr(s, 22).astype(dtype)
        o = cython_equalisation.make_decision(s[0], s.coded_symbols)
        xx = abs(s.symbols[0] - o)
        npt.assert_array_almost_equal(xx, 0)

