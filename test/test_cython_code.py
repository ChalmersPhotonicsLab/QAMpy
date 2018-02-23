import pytest
import numpy as np
import numpy.testing as npt
from dsp import signals, impairments
from dsp.core.equalisation import equaliser_cython


class TestQuantize(object):
    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128, np.complex256])
    def test_dtype(self, dtype):
        s = signals.SignalQAMGrayCoded(32, 2**12, dtype=dtype)
        s = impairments.change_snr(s, 23).astype(dtype)
        o = equaliser_cython.quantize(s[0], s.coded_symbols)
        assert o.dtype is np.dtype(dtype)

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128, np.complex256])
    def test_dtype(self, dtype):
        s = signals.SignalQAMGrayCoded(32, 2**12, dtype=dtype)
        s = impairments.change_snr(s, 23).astype(dtype)
        o = equaliser_cython.quantize(s[0], s.coded_symbols)
        npt.assert_array_almost_equal(s.symbols[0], o)

