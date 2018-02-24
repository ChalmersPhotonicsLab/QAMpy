import pytest
import numpy as np
import numpy.testing as npt

from dsp import signals
from dsp.core import resample
from dsp.core.equalisation import equaliser_cython


@pytest.mark.parametrize("arr", [np.random.randn(N)+1.j*np.random.randn(N)
                                 for N in [2**16, 10**6+1 ]])
@pytest.mark.parametrize("taps", [100, 1000, 4000])
@pytest.mark.parametrize("fconv", [True, False])
def test_resampling_benchmark(arr, taps, fconv, benchmark):
    benchmark.group = "resample N-%d taps-%d"%(arr.size, taps)
    s = benchmark(resample.rrcos_resample, arr, 1, 3, 1, beta=0.1, taps=taps, fftconv=fconv )

@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_quantize_precision(dtype, benchmark):
    s = signals.SignalQAMGrayCoded(128, 2**20, dtype=dtype)
    o = benchmark(equaliser_cython.quantize, s[0], s.coded_symbols)
    npt.assert_array_almost_equal(s.symbols[0], o)







