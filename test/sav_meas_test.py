#import cProfile
import numpy as np
import matplotlib.pylab as plt
from dsp import equalisation, modulation, utils, phaserecovery, IO
from timeit import default_timer as timer
import arrayfire as af



fb = 40.
os = 2
fs = os*fb
N = 3*10**5
M = 64
QAM = modulation.QAMModulator(M)
snrs = [10, 20, 30]
#lw_LO = [100e3]

#single pol
h5_sp = IO.tb.open_file("sav_meas_test_single_pol.h5", "w", "singlepol")
IO.create_meas_group(h5_sp, "single polarisation generated signal", (1,2*N))
IO.create_input_group(h5_sp, "input symbols", N, None)
IO.create_parameter_group(h5_sp, "Measurement parameters")
id_m = 0
for snr in snrs:
    X, symbolsX, bitsX = QAM.generateSignal(N, snr, baudrate=fb, samplingrate=fs, PRBS=True)
    IO.save_osc_meas(h5_sp, X, id_m, osnr=snr, wl=1550, samplingrate=fs, symbolrate=fb, MQAM=M)
    id_m += 1
h5_sp.close()

h5_dp = IO.tb.open_file("sav_meas_test_dual_pol.h5", "w", "singlepol")
IO.create_meas_group(h5_dp, "dual polarisation generated signal", (2,2*N))
IO.create_input_group(h5_dp, "input symbols", None, bitsX.shape)
IO.create_parameter_group(h5_dp, "Measurement parameters")
id_m = 0
for snr in snrs:
    X, symbolsX, bitsX = QAM.generateSignal(N, snr, baudrate=fb, samplingrate=fs, PRBS=True)
    Y, symbolsX, bitsX = QAM.generateSignal(N, snr, baudrate=fb, samplingrate=fs, PRBS=True)
    IO.save_inputs(h5_dp, id_m, bits=bitsX)
    E = np.vstack([X, Y])
    IO.save_osc_meas(h5_dp, E, id_m, osnr=snr, wl=1550, samplingrate=fs, symbolrate=fb, MQAM=M)
    id_m += 1
h5_dp.close()

