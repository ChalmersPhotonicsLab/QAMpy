import numpy as np
from qampy import signals
from qampy.core import pilotbased_receiver, pilotbased_transmitter

class TestFrameSync(object):
    def test_single_frame(self):
        s = signals.SignalWithPilots(256, 2**16, 256, 32, nmodes=2, nframes=3, fb=20e9)
        sig_tx = signal.resample(2*s.fb*2, beta=0.1)
        sig_tx = pilotbased_transmitter.sim_tx(sig_tx, snr=15,