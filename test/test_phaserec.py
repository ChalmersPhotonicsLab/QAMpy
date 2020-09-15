import pytest
import numpy as np
import numpy.testing as npt
from qampy import signals, impairments, phaserec
from qampy.core import phaserecovery as cphaserecovery
import matplotlib.pylab as plt



class TestReturnObject(object):
    @pytest.mark.parametrize("ndim", np.arange(1, 4))
    def test_viterbi(self, ndim):
        s = signals.SignalQAMGrayCoded(4, 2 ** 16, fb=20e9, nmodes=ndim)
        s2, ph = phaserec.viterbiviterbi(s, 10)
        assert type(s2) is type(s)

    @pytest.mark.parametrize("ndim", np.arange(1, 4))
    def test_bps(self, ndim):
        s = signals.SignalQAMGrayCoded(32, 2 ** 16, fb=20e9, nmodes=ndim)
        s2, ph = phaserec.bps(s, 32, 10)
        assert type(s2) is type(s)

    @pytest.mark.parametrize("ndim", np.arange(1, 4))
    def test_bps_twostage(self, ndim):
        s = signals.SignalQAMGrayCoded(32, 2 ** 16, fb=20e9, nmodes=ndim)
        s2, ph = phaserec.bps_twostage(s, 32, 10)
        assert type(s2) is type(s)

    @pytest.mark.parametrize("ndim", np.arange(1, 4))
    def test_phase_partition_16qam(self, ndim):
        s = signals.SignalQAMGrayCoded(16, 2 ** 16, fb=20e9, nmodes=ndim)
        s2, ph = phaserec.phase_partition_16qam(s, 10)
        assert type(s2) is type(s)

    @pytest.mark.parametrize("ndim", np.arange(1, 4))
    def test_comp_freq_offset(self, ndim):
        s = signals.SignalQAMGrayCoded(16, 2 ** 16, fb=20e9, nmodes=ndim)
        s2 = phaserec.comp_freq_offset(s, np.ones(ndim) * 1e6)
        assert type(s2) is type(s)

class Test2DCapability(object):

    @pytest.mark.parametrize("ndim", np.arange(1, 4))
    def test_viterbi(self, ndim):
        s = signals.SignalQAMGrayCoded(4, 2 ** 16, fb=20e9, nmodes=ndim)
        s2, ph = cphaserecovery.viterbiviterbi(s, 10, 4)
        assert ndim == s2.shape[0]

    @pytest.mark.parametrize("ndim", np.arange(1, 4))
    def test_bps(self, ndim):
        s = signals.SignalQAMGrayCoded(32, 2 ** 16, fb=20e9, nmodes=ndim)
        s2, ph = cphaserecovery.bps(s, 32 , s.coded_symbols, 10)
        assert ndim == s2.shape[0]

    @pytest.mark.parametrize("ndim", np.arange(1, 4))
    def test_bps_twostage(self, ndim):
        s = signals.SignalQAMGrayCoded(32, 2 ** 16, fb=20e9, nmodes=ndim)
        s2, ph = cphaserecovery.bps_twostage(s, 32 , s.coded_symbols, 10)
        assert ndim == s2.shape[0]

    @pytest.mark.parametrize("ndim", np.arange(1, 4))
    def test_phase_partition_16qam(self, ndim):
        s = signals.SignalQAMGrayCoded(16, 2 ** 16, fb=20e9, nmodes=ndim)
        s2, ph = cphaserecovery.phase_partition_16qam(s, 10)
        assert ndim == s2.shape[0]

    @pytest.mark.parametrize("ndim", np.arange(1, 4))
    def test_find_freq_offset(self, ndim):
        s = signals.SignalQAMGrayCoded(16, 2 ** 16, fb=20e9, nmodes=ndim)
        fo = cphaserecovery.find_freq_offset(s)
        assert ndim == fo.shape[0]

    @pytest.mark.parametrize("ndim", np.arange(1, 4))
    def test_comp_freq_offset(self, ndim):
        s = signals.SignalQAMGrayCoded(16, 2 ** 16, fb=20e9, nmodes=ndim)
        ph = np.ones(ndim)*1e6
        s2 = cphaserecovery.comp_freq_offset(s, ph)
        assert ndim == s2.shape[0]

    def test_viterbi_1d(self):
        s = signals.SignalQAMGrayCoded(4, 2 ** 16, fb=20e9, nmodes=1)
        s2, ph = cphaserecovery.viterbiviterbi(s.flatten(), 10, 4)
        assert 2**16 == s2.shape[0]

    def test_bps_1d(self):
        s = signals.SignalQAMGrayCoded(32, 2 ** 16, fb=20e9, nmodes=1)
        s2, ph = cphaserecovery.bps(s.flatten(), 32 , s.coded_symbols, 10)
        assert  2**16 == s2.shape[0]

    def test_bps_twostage(self):
        s = signals.SignalQAMGrayCoded(32, 2 ** 16, fb=20e9, nmodes=1)
        s2, ph = cphaserecovery.bps_twostage(s.flatten(), 32 , s.coded_symbols, 10)
        assert 2**16 == s2.shape[0]

    def test_phase_partition_16qam(self):
        s = signals.SignalQAMGrayCoded(16, 2 ** 16, fb=20e9, nmodes=1)
        s2, ph = cphaserecovery.phase_partition_16qam(s.flatten(), 10)
        assert 2**16 == s2.shape[0]

    def test_comp_freq_offset(self):
        s = signals.SignalQAMGrayCoded(16, 2 ** 16, fb=20e9, nmodes=1)
        ph = np.ones(1)*1e6
        s2 = cphaserecovery.comp_freq_offset(s.flatten(), ph)
        assert 2**16 == s2.shape[0]

class TestDtype(object):
    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_bps(self, dtype):
        s = signals.SignalQAMGrayCoded(32, 2**12, dtype=dtype)
        s *= np.exp(1.j*np.pi/3)
        s2, ph = phaserec.bps(s, 32, 10, method="pyt")
        assert s2.dtype is np.dtype(dtype)
        assert ph.dtype.itemsize is np.dtype(dtype).itemsize//2

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_bps_two_stage(self, dtype):
        s = signals.SignalQAMGrayCoded(32, 2**12, dtype=dtype)
        s *= np.exp(1.j*np.pi/3)
        s2, ph = phaserec.bps_twostage(s, 32, 10, method="pyt")
        assert s2.dtype is np.dtype(dtype)
        assert ph.dtype.itemsize is np.dtype(dtype).itemsize//2


class TestCorrect(object):
    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    @pytest.mark.parametrize("angle", np.linspace(0.1, np.pi/4.1, 8))
    def test_bps(self, dtype, angle):
        s = signals.SignalQAMGrayCoded(32, 2**12, dtype=dtype)
        s3 = s*np.exp(1.j*angle)
        s2, ph = phaserec.bps(s3, 32 , 11, method="pyt")
        o = ph[0][20:-20]+angle
        ser = s2[:,20:-20].cal_ser()
        npt.assert_allclose(0, ser)
        npt.assert_allclose(0, o, atol=np.pi/4/32)

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    @pytest.mark.parametrize("angle", np.linspace(0.1, np.pi/4.1, 8))
    def test_bps_two_stage(self, dtype, angle):
        s = signals.SignalQAMGrayCoded(32, 2**12, dtype=dtype)
        s3 = s*np.exp(1.j*angle)
        s2, ph = phaserec.bps_twostage(s3, 32//2 , 11, method="pyt")
        o = ph[0][25:-25]+angle
        ser = s2[:,25:-25].cal_ser()
        npt.assert_allclose(0, ser)
        npt.assert_allclose(0, o, atol=np.pi/4/32)

