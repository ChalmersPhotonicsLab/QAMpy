import pytest
import numpy as np
import random
import numpy.testing as npt
import matplotlib.pylab as plt

from dsp import modulation, impairments


class TestReturnObjects(object):
    s = modulation.ResampledQAM(16, 2**14, fs=2, nmodes=2)

    def test_rotate_field(self):
        s2 = impairments.rotate_field(self.s, np.pi/3)
        assert type(self.s) is type(s2)

    def test_apply_PMD(self):
        s2 = impairments.apply_PMD_to_field(self.s, np.pi/3, 1e-3, self.s.fs)
        assert type(self.s) is type(s2)

    def test_apply_phase_noise(self):
        s2 = impairments.apply_phase_noise(self.s, 1e-3, self.s.fs)
        assert type(self.s) is type(s2)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_add_awgn(self, attr):
        s2 = impairments.add_awgn(self.s, 0.01)
        assert getattr(self.s, attr) is getattr(s2, attr)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_rotate_field_attr(self, attr):
        s2 = impairments.rotate_field(self.s, np.pi/3)
        assert getattr(self.s, attr) is getattr(s2, attr)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_apply_PMD_attr(self, attr):
        s2 = impairments.apply_PMD_to_field(self.s, np.pi/3, 1e-3, self.s.fs)
        assert getattr(self.s, attr) is getattr(s2, attr)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_apply_phase_noise_attr(self, attr):
        s2 = impairments.apply_phase_noise(self.s, 1e-3, self.s.fs)
        assert getattr(self.s, attr) is getattr(s2, attr)

    @pytest.mark.parametrize("attr", ["fs", "symbols", "fb"])
    def test_add_awgn_attr(self, attr):
        s2 = impairments.add_awgn(self.s, 0.01)
        assert getattr(self.s, attr) is getattr(s2, attr)
