import numpy as np
import matplotlib.pylab as plt
from dsp import  equalisation, utils, modulation


fb = 40.e9
os = 2
fs = os*fb
N = 10**5
mu = 4e-4
theta = np.pi/2.45
theta2 = np.pi/4
t_pmd = 75e-12
M = 4
ntaps=40

QAM = modulation.QAMModulator(M)

X, sx, bx = QAM.generate_signal(N, 14, baudrate=fb, samplingrate=fs)
Y, sy, by = QAM.generate_signal(N, 14, baudrate=fb, samplingrate=fs)
omega = 2*np.pi*np.linspace(-fs/2, fs/2, N*os, endpoint=False)

S = np.vstack([X,Y])
SS = utils.apply_PMD_to_field(S, theta, t_pmd, omega)
wxy, err = equalisation.equalise_signal(SS, os, mu, M, Ntaps=ntaps, method="cma")
wxy_m, err_m = equalisation.equalise_signal(SS, os, mu, M, Ntaps=ntaps, method="mcma")
E = equalisation.apply_filter(SS, os, wxy)
E_m = equalisation.apply_filter(SS, os, wxy_m)
print(E.shape)
print(SS.shape)


evmX = QAM.cal_evm(X[::2])
evmY = QAM.cal_evm(Y[::2])
evmEx = QAM.cal_evm(E[0])
evmEy = QAM.cal_evm(E[1])
evmEx_m = QAM.cal_evm(E_m[0])
evmEy_m = QAM.cal_evm(E_m[1])
#sys.exit()
plt.figure()
plt.subplot(131)
plt.title('Recovered CMA')
plt.plot(E[0].real, E[0].imag, 'ro', label=r"$EVM_x=%.1f\%%$"%(100*evmEx))
plt.plot(E[1].real, E[1].imag, 'go' ,label=r"$EVM_y=%.1f\%%$"%(evmEy*100))
plt.legend()
plt.subplot(132)
plt.title('Recovered MCMA')
plt.plot(E_m[0].real, E_m[0].imag, 'ro', label=r"$EVM_x=%.1f\%%$"%(100*evmEx_m))
plt.plot(E_m[1].real, E_m[1].imag, 'go' ,label=r"$EVM_y=%.1f\%%$"%(evmEy_m*100))
plt.legend()
plt.subplot(133)
plt.title('Original')
plt.plot(X[::2].real, X[::2].imag, 'ro', label=r"$EVM_x=%.1f\%%$"%(100*evmX))
plt.plot(Y[::2].real, Y[::2].imag, 'go', label=r"$EVM_y=%.1f\%%$"%(100*evmY))
plt.legend()

plt.figure()
plt.subplot(221)
plt.title('Taps CMA')
plt.plot(wxy[0][0,:], 'r')
plt.plot(wxy[0][1,:], '--r')
plt.plot(wxy[1][0,:], 'g')
plt.plot(wxy[1][1,:], '--g')
plt.subplot(222)
plt.title('error CMA')
plt.plot(abs(err[0]), color='r')
plt.plot(abs(err[1]), color='g')
plt.subplot(223)
plt.title('Taps MCMA')
plt.plot(wxy_m[0][0,:], 'r')
plt.plot(wxy_m[0][1,:], '--r')
plt.plot(wxy_m[1][0,:], 'g')
plt.plot(wxy_m[1][1,:], '--g')
plt.subplot(224)
plt.title('error MCMA')
plt.plot(abs(err_m[0]), color='r')
plt.plot(abs(err_m[1]), color='g')
plt.show()



