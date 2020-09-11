import pytest
import numpy as np
import numpy.testing as npt
from qampy import signals, impairments
from qampy.core.equalisation import pythran_equalisation


class TestQuantize(object):
    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_dtype(self, dtype):
        s = signals.SignalQAMGrayCoded(32, 2**12, dtype=dtype)
        s = impairments.change_snr(s, 23)
        o, d, i = pythran_equalisation.make_decision(s[0], s.coded_symbols)
        assert o.dtype is np.dtype(dtype)

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_decision(self, dtype):
        s = signals.SignalQAMGrayCoded(32, 2**12, dtype=dtype)
        s = impairments.change_snr(s, 30).astype(dtype)
        o, d, i = pythran_equalisation.make_decision(s[0], s.coded_symbols)
        xx = abs(s.symbols[0] - o)
        npt.assert_array_almost_equal(xx, 0)

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_decision_symbols(self, dtype):
        s = signals.SignalQAMGrayCoded(32, 2**12, dtype=dtype)
        o, d, i = pythran_equalisation.make_decision(s[0], s.coded_symbols)
        npt.assert_array_almost_equal(o, s.coded_symbols[i])
        npt.assert_array_almost_equal(s.symbols[0], s.coded_symbols[i])
        
    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    @pytest.mark.parametrize("dist", np.linspace(0, 0.3, 4))
    def test_decision_distance(self, dtype, dist):
        s = signals.SignalQAMGrayCoded(4, 2**12, dtype=dtype)
        s += dist
        o, d, i = pythran_equalisation.make_decision(s[0], s.coded_symbols)
        npt.assert_array_almost_equal(dist, d)
         

