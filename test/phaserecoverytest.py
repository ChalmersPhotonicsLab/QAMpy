#import cProfile
import numpy as np
import matplotlib.pylab as plt
from dsp import equalisation, modulation, utils, phaserecovery



fb = 40.e9
os = 1
fs = os*fb
N = 2*10**5
M = 64
QAM = modulation.QAMModulator(M)
snr = 23
lw_LO = np.linspace(10e1, 1000e1, 2)
#lw_LO = [100e3]
sers = []

X, symbolsX, bitsX = QAM.generateSignal(N, snr, baudrate=fb, samplingrate=fs)

for lw in lw_LO:
    shiftN = np.random.randint(-N/2, N/2, 1)
    pp = utils.phase_noise(X.shape[0], lw, fs)
    XX = X*np.exp(1.j*pp)
    print("shiftN: %d"%shiftN)
    #recoverd,ph= phaserecovery.blindphasesearch_af(XX, 64, QAM.symbols, 14)
    ser,s,d = QAM.calculate_SER(X, symbol_tx=np.roll(symbolsX[:2**15-1], shiftN), N1=2**15, N2=800)
    print(ser)
    sers.append(ser)

#plt.plot(lw_LO, sers)
plt.show()


