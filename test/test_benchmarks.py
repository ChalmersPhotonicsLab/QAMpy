import pytest
import numpy as np
import numpy.testing as npt

import matplotlib.pylab as plt
from qampy import signals, impairments, equalisation, helpers, phaserec
from qampy.core import resample
from qampy.core.equalisation import cython_equalisation
from qampy.core import equalisation as cequalisation



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
    o = benchmark(cython_equalisation.make_decision, s[0], s.coded_symbols)
    npt.assert_array_almost_equal(s.symbols[0], o)

@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_equalisation_prec(dtype, benchmark):
    fb = 40.e9
    os = 2
    fs = os*fb
    N = 10**5
    #mu = np.float32(4e-4)
    mu = 4e-4
    theta = np.pi/5.45
    theta2 = np.pi/4
    t_pmd = 75e-12
    M = 4
    ntaps=40
    snr =  14
    sig = signals.SignalQAMGrayCoded(M, N, fb=fb, nmodes=2, dtype=dtype)
    S = sig.resample(fs, renormalise=True, beta=0.1)
    S = impairments.apply_phase_noise(S, 100e3)
    S = impairments.change_snr(S, snr)
    SS = impairments.apply_PMD_to_field(S, theta, t_pmd)
    wxy, err = benchmark(equalisation.equalise_signal, SS, mu, Ntaps=ntaps, method="mcma", adaptive_stepsize=True)
    E = equalisation.apply_filter(SS,  wxy)
    E = helpers.normalise_and_center(E)
    E, ph = phaserec.viterbiviterbi(E, 11)
    E = helpers.dump_edges(E, 20)
    ser = E.cal_ser().mean()
    npt.assert_allclose(0, ser, atol=3e-5)

@pytest.mark.parametrize("dtype", [ np.complex64, np.complex128])
#@pytest.m    npt.assert_allclose(0, ser, atol=3e-5)ark.parametrize("method", [cequalisation.apply_filter, cython_equalisation.apply_filter_signal, cython_equalisation.apply_filter_singal2 ])
def test_apply_filter(dtype, benchmark):
    fb = 40.e9
    os = 2
    fs = os*fb
    N = 10**5
    mu = 4e-4
    theta = np.pi/5.45
    theta2 = np.pi/4
    t_pmd = 75e-12
    M = 4
    ntaps=40
    snr =  14
    sig = signals.SignalQAMGrayCoded(M, N, fb=fb, nmodes=2, dtype=dtype)
    S = sig.resample(fs, renormalise=True, beta=0.1)
    S = impairments.change_snr(S, snr)
    SS = impairments.apply_PMD_to_field(S, theta, t_pmd)
    wxy, err = equalisation.equalise_signal(SS, mu, Ntaps=ntaps, method="mcma", adaptive_step=True)
    E1 = equalisation.apply_filter(SS,  wxy, method="pyx")
    E2 = equalisation.apply_filter(SS, wxy, method="py")
    E2 = E1.recreate_from_np_array(E2)
    E1 = helpers.normalise_and_center(E1)
    E2 = helpers.normalise_and_center(E2)
    E1, ph = phaserec.viterbiviterbi(E1, 11)
    E2, ph = phaserec.viterbiviterbi(E2, 11)
    E1 = helpers.dump_edges(E1, 20)
    E2 = helpers.dump_edges(E2, 20)
    ser1 = E1.cal_ser().mean()
    ser2 = E2.cal_ser().mean()
    npt.assert_allclose(0, ser1, atol=3e-5)
    npt.assert_allclose(0, ser2, atol=3e-5)




@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("method", ["py", "pyx"])
def test_apply_filter_benchmark(dtype, method, benchmark):
    benchmark.group = "apply filter "+str(dtype)
    fb = 40.e9
    os = 2
    fs = os*fb
    N = 2**17
    mu = 4e-4
    theta = np.pi/5.45
    theta2 = np.pi/4
    t_pmd = 75e-12
    M = 4
    ntaps=40
    snr =  14
    sig = signals.SignalQAMGrayCoded(M, N, fb=fb, nmodes=2, dtype=dtype)
    S = sig.resample(fs, renormalise=True, beta=0.1)
    S = impairments.change_snr(S, snr)
    SS = impairments.apply_PMD_to_field(S, theta, t_pmd)
    wxy, err = equalisation.equalise_signal(SS, mu, Ntaps=ntaps, method="mcma", adaptive_step=True)
    benchmark(equalisation.apply_filter, SS,  wxy, method)

