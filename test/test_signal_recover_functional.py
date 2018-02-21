import pytest

import numpy as np
import numpy.testing as npt

from dsp import modulation, helpers, phaserec, equalisation, impairments

@pytest.mark.parametrize("lw", np.linspace(10, 1000, 4))
@pytest.mark.parametrize("M", [4, 16, 32, 64])
def test_phaserec_bps(lw, M):
    fb = 40.e9
    os = 1
    fs = os*fb
    N = 2**17
    snr = 30
    shiftN = np.random.randint(-N/2, N/2, 1)
    s = modulation.SignalQAMGrayCoded(M, N, fb=fb)
    s = s.resample(fs, beta=0.1, renormalise=True)
    s = impairments.change_snr(s, snr)
    s = np.roll(s, shiftN, axis=1)
    pp = impairments.apply_phase_noise(s, lw)
    recoverd, ph1 = phaserec.bps(pp, M , s.coded_symbols, 14, method='pyx')
    recoverd = helpers.dump_edges(recoverd, 20)
    ser = recoverd.cal_ser()
    npt.assert_allclose(ser, 0)

@pytest.mark.parametrize("lw", np.linspace(10, 1000, 4))
@pytest.mark.parametrize("M", [4, 16, 32, 64])
def test_phaserec_bps_2stage(lw, M):
    fb = 40.e9
    os = 1
    fs = os*fb
    N = 2**17
    snr = 30
    shiftN = np.random.randint(-N/2, N/2, 1)
    s = modulation.SignalQAMGrayCoded(M, N, fb=fb)
    s = s.resample(fs, beta=0.1, renormalise=True)
    s = impairments.change_snr(s, snr)
    s = np.roll(s, shiftN, axis=1)
    pp = impairments.apply_phase_noise(s, lw)
    recoverd, ph1 = phaserec.bps_twostage(pp, max(4, M/2), s.coded_symbols, 14, method='pyx')
    recoverd = helpers.dump_edges(recoverd, 20)
    ser = recoverd.cal_ser()
    npt.assert_allclose(ser, 0)




