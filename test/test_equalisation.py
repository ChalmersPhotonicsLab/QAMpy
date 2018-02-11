import pytest
import numpy.testing as npt

from dsp import modulation
from dsp.core import impairments, equalisation


class TestReturnObject(object):
    s = modulation.ResampledQAM(16, 2 ** 16, fb=20e9, fs=40e9, nmodes=2)
    os = 2

    def test_apply_filter(self):
        s2 = impairments.simulate_transmission(self.s, self.s.fb, self.s.fs, snr=20, dgd=100e-12)
        wx, err = equalisation.equalise_signal(s2, self.os, 1e-3, s2.M, Ntaps=11)
        s3 = equalisation.apply_filter(s2, self.os, wx)
        assert type(s3) is type(self.s)


