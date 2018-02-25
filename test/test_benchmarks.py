import pytest
import numpy as np
import numpy.testing as npt

from dsp import signals, impairments, equalisation, helpers, phaserec
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

@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_equalisation_prec(dtype, benchmark):
    fb = 40.e9
    os = 2
    fs = os*fb
    N = 10**5
    mu = np.array(4e-4, dtype=np.float32)
    theta = np.pi/5.45
    theta2 = np.pi/4
    t_pmd = 75e-12
    M = 4
    ntaps=40
    snr =  14
    sig = signals.SignalQAMGrayCoded(M, N, fb=fb, nmodes=2)
    S = sig.resample(fs, renormalise=True, beta=0.1)
    S = impairments.apply_phase_noise(S, 100e3)
    S = impairments.change_snr(S, snr)
    SS = impairments.apply_PMD_to_field(S, theta, t_pmd)
    wxy, err = benchmark(equalisation.equalise_signal, SS, mu, Ntaps=ntaps, method="mcma", adaptive_step=True)
    E = equalisation.apply_filter(SS,  wxy)
    E = helpers.normalise_and_center(E)
    E, ph = phaserec.viterbiviterbi(E, 11)
    E = helpers.dump_edges(E, 20)
    ser = E.cal_ser().mean()
    npt.assert_allclose(0, ser, atol=3e-5)






