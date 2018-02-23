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

@pytest.mark.parametrize("dtype", [np.complex64, np.complex128, np.complex256])
def test_quantize_precision(dtype, benchmark):
    s = signals.SignalQAMGrayCoded(128, 2**20, dtype=dtype)
    o = benchmark(equaliser_cython.quantize, s[0], s.coded_symbols)
    npt.assert_array_almost_equal(s.symbols[0], o)

@pytest.mark.parametrize("dtype", [np.complex64, np.complex128, np.complex256])
@pytest.mark.parametrize("method", [equaliser_cython.quantize])#, equaliser_cython.quantize3])
def test_quantize_precision_2(dtype, method, benchmark):
    s = signals.SignalQAMGrayCoded(128, 2**20, dtype=dtype)
    benchmark.group = "quantize precision-%d"%(8*np.dtype(dtype).itemsize)
    o = benchmark(method, s[0], s.coded_symbols)
    npt.assert_array_almost_equal(s.symbols[0], o)

@pytest.mark.parametrize("method", [equaliser_cython.quantize, equaliser_cython.quantize2])
def test_quantize_precision_3(method, benchmark):
    s = signals.SignalQAMGrayCoded(128, 2**20)
    o = benchmark(method, s[0], s.coded_symbols)
    npt.assert_array_almost_equal(s.symbols[0], o)

@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("method", [equaliser_cython.testapplyfilter1, equaliser_cython.testapplyfilter2])
def test_apply_filter(dtype, method, benchmark):
    s = signals.SignalQAMGrayCoded(128, 2**20, nmodes=2, dtype=dtype)
    wx = np.random.randn(2, 2**16) + 1.j*np.random.randn(2, 2**16)
    o = benchmark(method, s, 2**16, wx.astype(dtype))

@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_apply_filter2(dtype):
    s = signals.SignalQAMGrayCoded(128, 2**20, nmodes=2, dtype=dtype)
    wx = (np.random.randn(2, 2**16) + 1.j*np.random.randn(2, 2**16))*0.1
    o1 = equaliser_cython.testapplyfilter1(s, 2**16, wx.astype(dtype))
    o2 = equaliser_cython.testapplyfilter2(s, 2**16, wx.astype(dtype))
    npt.assert_array_almost_equal(o1, o2)



def test_update_filter():
    s = signals.SignalQAMGrayCoded(128, 2**20)
    wx = np.random.randn(2, 10) + 1.j*np.random.randn(2, 10)
    wx2 = wx.copy()
    wx0 = wx.copy()
    equaliser_cython.testupdate2(s, 10, 1e-1, 1.1+1.1j, wx)
    equaliser_cython.testupdate3(s, 10, 1e-1, 1.1+1.1j, wx2)
    npt.assert_array_almost_equal(wx, wx2)
    wxx = np.mean(abs(wx-wx0))
    wxx2 = np.mean(abs(wx2-wx0))
    assert wxx > 0.
    assert wxx2 > 0.

@pytest.mark.parametrize("method", [equaliser_cython.testupdate1, equaliser_cython.testupdate2, equaliser_cython.testupdate3])
def test_update_filter2(method, benchmark):
    s = signals.SignalQAMGrayCoded(128, 2**20)
    wx = np.random.randn(2, 2**16) + 1.j*np.random.randn(2, 2**16)
    wx0 = wx.copy()
    benchmark(method, s, 2**16, 1e-1, 1.1+1.1j, wx)
    wxx = np.mean(abs(wx-wx0))
    assert wxx > 0.




