import numpy as np
import matplotlib.pylab as plt
from dsp import signals, phaserecovery
from dsp.signal_quality import cal_blind_evm


fb = 40.e9
os = 2
fs = os*fb
N = 2**18
theta = np.pi/2.45
carrier_f = 1e6

X, XIdata, XQdata = signals.generateRandomQPSKData(N, 14, baudrate=fb, samplingrate=fs, carrier_df=carrier_f)

Y = X * np.exp(-1.j*2*np.pi*np.arange(len(X))*carrier_f/fs)
Y = Y[::2]

E = phaserecovery.viterbiviterbi_qpsk(9, X[::2])

evm1 = cal_blind_evm(E, 4)
evm2 = cal_blind_evm(Y, 4)

plt.figure()
plt.subplot(121)
plt.title('Recovered')
plt.plot(E.real, E.imag, 'ro', label=r"$EVM=%.1f\%%$"%(100*evm1))
plt.legend()
plt.subplot(122)
plt.title('Original')
plt.plot(Y.real, Y.imag, 'go', label=r"$EVM_y=%.1f\%%$"%(100*evm2))
plt.legend()
plt.show()
