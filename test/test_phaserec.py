import pytest
import numpy as np
from dsp import modulation, phaserecovery, impairments

class TestReturnObject(object):
    @pytest.mark.parametrize("ndim", np.arange(1, 4))
    def test_viterbi(self, ndim):
        s = modulation.SignalQAMGrayCoded(4, 2**16, fb=20e9,  nmodes=ndim)
        s2 = phaserecovery.viterbiviterbi(s, 10, 4)
        assert type(s2) is type(s)

    @pytest.mark.parametrize("ndim", np.arange(1, 4))
    def test_bps(self, ndim):
        s = modulation.SignalQAMGrayCoded(32, 2**16, fb=20e9,  nmodes=ndim)
        s2, ph = phaserecovery.bps(s, 32 , s.coded_symbols, 10)
        assert type(s2) is type(s)

    @pytest.mark.parametrize("ndim", np.arange(1, 4))
    def test_bps_twostage(self, ndim):
        s = modulation.SignalQAMGrayCoded(32, 2**16, fb=20e9,  nmodes=ndim)
        s2,ph = phaserecovery.bps_twostage(s, 32 , s.coded_symbols, 10)
        assert type(s2) is type(s)

    @pytest.mark.parametrize("ndim", np.arange(1, 4))
    def test_phase_partition_16qam(self, ndim):
        s = modulation.SignalQAMGrayCoded(16, 2**16, fb=20e9,  nmodes=ndim)
        s2 = phaserecovery.phase_partition_16qam(s, 10)
        assert type(s2) is type(s)

    @pytest.mark.parametrize("ndim", np.arange(1, 4))
    def test_comp_freq_offset(self, ndim):
        s = modulation.SignalQAMGrayCoded(16, 2**16, fb=20e9,  nmodes=ndim)
        s2 = phaserecovery.comp_freq_offset(s, np.ones(ndim)*1e6 )
        assert type(s2) is type(s)

class Test2DCapability(object):

    @pytest.mark.parametrize("ndim", np.arange(1, 4))
    def test_viterbi(self, ndim):
        s = modulation.SignalQAMGrayCoded(4, 2**16, fb=20e9,  nmodes=ndim)
        s2, ph = phaserecovery.viterbiviterbi(s, 10, 4)
        assert ndim == s2.shape[0]

    @pytest.mark.parametrize("ndim", np.arange(1, 4))
    def test_bps(self, ndim):
        s = modulation.SignalQAMGrayCoded(32, 2**16, fb=20e9,  nmodes=ndim)
        s2, ph = phaserecovery.bps(s, 32 , s.coded_symbols, 10)
        assert ndim == s2.shape[0]

    @pytest.mark.parametrize("ndim", np.arange(1, 4))
    def test_bps_twostage(self, ndim):
        s = modulation.SignalQAMGrayCoded(32, 2**16, fb=20e9,  nmodes=ndim)
        s2, ph = phaserecovery.bps_twostage(s, 32 , s.coded_symbols, 10)
        assert ndim == s2.shape[0]

    @pytest.mark.parametrize("ndim", np.arange(1, 4))
    def test_phase_partition_16qam(self, ndim):
        s = modulation.SignalQAMGrayCoded(16, 2**16, fb=20e9,  nmodes=ndim)
        s2, ph = phaserecovery.phase_partition_16qam(s, 10)
        assert ndim == s2.shape[0]

    @pytest.mark.parametrize("ndim", np.arange(1, 4))
    def test_find_freq_offset(self, ndim):
        s = modulation.SignalQAMGrayCoded(16, 2**16, fb=20e9,  nmodes=ndim)
        fo = phaserecovery.find_freq_offset(s)
        assert ndim == fo.shape[0]

    @pytest.mark.parametrize("ndim", np.arange(1, 4))
    def test_comp_freq_offset(self, ndim):
        s = modulation.SignalQAMGrayCoded(16, 2**16, fb=20e9,  nmodes=ndim)
        ph = np.ones(ndim)*1e6
        s2 = phaserecovery.comp_freq_offset(s, ph)
        assert ndim == s2.shape[0]

    def test_viterbi_1d(self):
        s = modulation.SignalQAMGrayCoded(4, 2**16, fb=20e9,  nmodes=1)
        s2, ph = phaserecovery.viterbiviterbi(s.flatten(), 10, 4)
        assert 2**16 == s2.shape[0]

    def test_bps_1d(self):
        s = modulation.SignalQAMGrayCoded(32, 2**16, fb=20e9,  nmodes=1)
        s2, ph = phaserecovery.bps(s.flatten(), 32 , s.coded_symbols, 10)
        assert  2**16 == s2.shape[0]

    def test_bps_twostage(self):
        s = modulation.SignalQAMGrayCoded(32, 2**16, fb=20e9,  nmodes=1)
        s2, ph = phaserecovery.bps_twostage(s.flatten(), 32 , s.coded_symbols, 10)
        assert 2**16 == s2.shape[0]

    def test_phase_partition_16qam(self):
        s = modulation.SignalQAMGrayCoded(16, 2**16, fb=20e9,  nmodes=1)
        s2, ph = phaserecovery.phase_partition_16qam(s.flatten(), 10)
        assert 2**16 == s2.shape[0]

    def test_comp_freq_offset(self):
        s = modulation.SignalQAMGrayCoded(16, 2**16, fb=20e9,  nmodes=1)
        ph = np.ones(1)*1e6
        s2 = phaserecovery.comp_freq_offset(s.flatten(), ph)
        assert 2**16 == s2.shape[0]



