import numpy as np
import matplotlib.pylab as plt
from dsp import  phaserecovery, modulation, utils


fb = 40.e9
os = 2
fs = os*fb
N = 10**5
theta = np.pi/2.45
M = 4
snr = 14
carrier_f = 1e6
QAM = modulation.QAMModulator(M)
X, Sx, Bx = QAM.generate_signal(N, snr, baudrate=fb, samplingrate=fs , carrier_df=carrier_f, dual_pol=False)



E = phaserecovery.viterbiviterbi(X[::2], 9, 4)

evm1 = QAM.cal_evm(E)
evm2 = QAM.cal_evm(X)

plt.figure()
plt.subplot(121)
plt.title('Recovered')
plt.plot(E.real, E.imag, 'ro', label=r"$EVM=%.1f\%%$"%(100*evm1))
plt.legend()
plt.subplot(122)
plt.title('Original')
plt.plot(X.real, X.imag, 'go', label=r"$EVM_y=%.1f\%%$"%(100*evm2))
plt.legend()
plt.show()
