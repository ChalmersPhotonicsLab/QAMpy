import numpy as np
import matplotlib.pylab as plt
from dsp import equalisation, modulation, impairments

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
snr =  12

sig = modulation.SignalQAMGrayCoded(M, N, fb=fb, nmodes=2)
S = sig.resample(fs, renormalise=True, beta=0.2)
S = impairments.change_snr(S, snr)

SS = impairments.apply_PMD_to_field(S, theta, t_pmd)
wxy, err = equalisation.equalise_signal(SS, mu, M, Ntaps=ntaps, method="cma")
wxy_m, err_m = equalisation.equalise_signal(SS, mu, M, Ntaps=ntaps, method="mcma")
E = equalisation.apply_filter(SS,  wxy)
E_m = equalisation.apply_filter(SS, wxy_m)
print(E.shape)
print(SS.shape)

evm = E.cal_evm()
evm_m = E_m.cal_evm()
evm0 = S[:, ::2].cal_evm()
#sys.exit()
plt.figure()
plt.subplot(131)
plt.title('Recovered CMA')
plt.plot(E[0].real, E[0].imag, 'ro', label=r"$EVM_x=%.1f\%%$"%(100*evm[0]))
plt.plot(E[1].real, E[1].imag, 'go' ,label=r"$EVM_y=%.1f\%%$"%(evm[1]*100))
plt.legend()
plt.subplot(132)
plt.title('Recovered MCMA')
plt.plot(E_m[0].real, E_m[0].imag, 'ro', label=r"$EVM_x=%.1f\%%$"%(100*evm_m[0]))
plt.plot(E_m[1].real, E_m[1].imag, 'go' ,label=r"$EVM_y=%.1f\%%$"%(evm_m[1]*100))
plt.legend()
plt.subplot(133)
plt.title('Original')
plt.plot(S[0,::2].real, S[0,::2].imag, 'ro', label=r"$EVM_x=%.1f\%%$"%(100*evm0[0]))
plt.plot(S[1,::2].real, S[1,::2].imag, 'go', label=r"$EVM_y=%.1f\%%$"%(100*evm0[1]))
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



