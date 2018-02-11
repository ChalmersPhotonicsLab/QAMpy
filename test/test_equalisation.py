import pytest
import numpy.testing as npt

from dsp import modulation, equalisation
from dsp.core import impairments
from dsp.core import equalisation as cequalisation


class TestReturnObject(object):
    s = modulation.ResampledQAM(16, 2 ** 16, fb=20e9, fs=40e9, nmodes=2)
    os = 2

    def test_apply_filter_basic(self):
        s2 = impairments.simulate_transmission(self.s, self.s.fb, self.s.fs, snr=20, dgd=100e-12)
        wx, err = equalisation.equalise_signal(s2, 1e-3, s2.M, Ntaps=11)
        s3 = equalisation.apply_filter(s2, wx)
        assert type(s3) is type(self.s)


    @pytest.mark.xfail(reason="The core equalisation functions are not expected to preserve subclasses")
    def test_apply_filter_adv(self):
        s2 = impairments.simulate_transmission(self.s, self.s.fb, self.s.fs, snr=20, dgd=100e-12)
        wx, err = cequalisation.equalise_signal(s2, self.os, 1e-3, s2.M, Ntaps=11)
        s3 = equalisation.apply_filter(s2, self.os, wx)
        assert type(s3) is type(self.s)


