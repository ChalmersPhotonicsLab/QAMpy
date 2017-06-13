#import cProfile
import numpy as np
import matplotlib.pylab as plt
from dsp import equalisation, modulation, utils, phaserecovery
from timeit import default_timer as timer
import arrayfire as af



fb = 40.e9
os = 1
fs = os*fb
N = 3*10**5
M = 64
QAM = modulation.QAMModulator(M)
snr = 30
lw_LO = np.linspace(10e1, 1000e1, 4)
#lw_LO = [100e3]
sers = []


X, symbolsX, bitsX = QAM.generate_signal(N, snr, baudrate=fb, samplingrate=fs, PRBS=True)

for lw in lw_LO:
    shiftN = np.random.randint(-N/2, N/2, 1)
    xtest = np.roll(symbolsX[:(2**15-1)], shiftN)
    pp = utils.phase_noise(X.shape[0], lw, fs)
    XX = X*np.exp(1.j*pp)
    t1 = timer()
    recoverd_af,ph= phaserecovery.blindphasesearch(XX, 64, QAM.symbols, 14, method="af")
    t2 = timer()
    recoverd_pyx,ph= phaserecovery.blindphasesearch(XX, 64, QAM.symbols, 14, method="cython")
    t3 = timer()
    recoverd_2s,ph= phaserecovery.blindphasesearch_twostage(XX, 28, QAM.symbols, 14, method='cython')
    t4 = timer()
    ser,s,d = QAM.calc_SER(recoverd_af, symbol_tx=xtest)
    ser2,s,d = QAM.calc_SER(recoverd_pyx, symbol_tx=xtest)
    ser3,s,d = QAM.calc_SER(recoverd_2s, symbol_tx=xtest)
    print("1 stage af ser=%g"%ser)
    print("1 stage pyx ser=%g"%ser2)
    print("2 stage pyx ser=%g"%ser3)
    print("time af %.1f"%abs(t2-t1))
    print("time pyx %.1f"%abs(t3-t2))
    print("time 2 stage %.1f"%abs(t4-t3))
    sers.append(ser)

#plt.plot(lw_LO, sers)
plt.show()


