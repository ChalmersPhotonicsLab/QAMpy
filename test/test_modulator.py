import numpy as np
import numpy.testing as npt
import matplotlib.pylab as plt

from dsp import modulation


class TestModulatorAttr(object):
    Q = modulation.QAMModulator(16)
    def test_SER(self):
        pass
