#import cProfile
import numpy as np
from qampy.core import io
from qampy import signals
import os as os_mod



fb = 40.
os = 2
fs = os*fb
N = 3*10**5
M = 64
QAM = signals.QAMModulator(M)
snrs = [10, 20, 30]
#lw_LO = [100e3]

hdfn1 ="/tmp/sav_meas_test_single_pol.h5" 
hdfn2 ="/tmp/sav_meas_test_dual_pol.h5" 
if os_mod.path.exists(hdfn1):
    os_mod.remove(hdfn1)

#single pol
h5_sp = io.create_h5_meas_file(hdfn1, "single polarisation test save")
for snr in snrs:
    X, symbolsX, bitsX = QAM.generateSignal(N, snr, baudrate=fb, samplingrate=fs, PRBS=True)
    id_m = io.save_osc_meas(h5_sp, X, osnr=snr, wl=1550, samplingrate=fs, symbolrate=fb, MQAM=M)
    io.save_inputs(h5_sp, id_m, symbols=symbolsX)
h5_sp.close()

if os_mod.path.exists(hdfn2):
    os_mod.remove(hdfn2)

h5_dp = io.create_h5_meas_file(hdfn2, "dual polarisation test save")
for snr in snrs:
    X, symbolsX, bitsX = QAM.generateSignal(N, snr, baudrate=fb, samplingrate=fs, PRBS=True)
    Y, symbolsX, bitsX = QAM.generateSignal(N, snr, baudrate=fb, samplingrate=fs, PRBS=True)
    E = np.vstack([X, Y])
    bits = np.vstack([bitsX, bitsX])
    id_m = io.save_osc_meas(h5_dp, E, osnr=snr, wl=1550, samplingrate=fs, symbolrate=fb, MQAM=M)
    io.save_inputs(h5_dp, id_m, bits=bits)
h5_dp.close()
