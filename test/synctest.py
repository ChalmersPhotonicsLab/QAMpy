#import cProfile
import numpy as np
import matplotlib.pylab as plt
from dsp import equalisation, modulation, utils, phaserecovery



fb = 40.e9
os = 1
fs = os*fb
N = 10**5
M = 64
QAM = modulation.QAMModulator(M)
snr = 40
lw_LO = np.linspace(10e1, 1000e1, 10)
#lw_LO = [100e3]
sers = []

X, symbolsX, bitsX = QAM.generateSignal(N, snr, baudrate=fb, samplingrate=fs, PRBS=False )
#symbolsX = np.random.randn(2**15-1) + 1.j*np.random.randn(2**15-1)

for lw in lw_LO:
    shiftN = np.random.randint(-N/2, N/2, 1)
    pp = utils.phase_noise(X.shape[0], lw, fs)
    XX = X*np.exp(1.j*pp)
    print("shiftN: %d"%shiftN)
    recoverd,ph= phaserecovery.blindphasesearch(XX, 64, QAM.symbols, 5)
    ser,s,d = QAM.calculate_SER(X, symbol_tx=np.roll(symbolsX, shiftN))
    print("SER=%f"%ser)
    sers.append(ser)

#plt.plot(lw_LO, sers)


