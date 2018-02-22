#import cProfile
import numpy as np
from dsp.core import impairments, phaserecovery
from dsp import signals

fb = 40.e9
os = 1
fs = os*fb
N = 10**5
M = 64
QAM = signals.QAMModulator(M)
snr = 40
lw_LO = np.linspace(10e1, 1000e1, 10)
#lw_LO = [100e3]
sers = []

X, symbolsX, bitsX = QAM.generate_signal(N, snr, baudrate=fb, samplingrate=fs, PRBS=False , dual_pol=False)
#symbolsX = np.random.randn(2**15-1) + 1.j*np.random.randn(2**15-1)

for lw in lw_LO:
    shiftN = np.random.randint(-N/2, N/2, 1)
    pp = impairments.phase_noise(X.shape[0], lw, fs)
    XX = X*np.exp(1.j*pp)
    print("shiftN: %d"%shiftN)
    recoverd,ph= phaserecovery.bps(XX, 64, QAM.symbols, 5)
    ser = QAM.cal_ser(X, symbol_tx=np.roll(symbolsX, shiftN))
    print("SER=%f"%ser)
    sers.append(ser)

#plt.plot(lw_LO, sers)


