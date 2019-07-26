import numpy as np
import matplotlib.pylab as plt
from qampy import equalisation, signals, impairments, helpers

fb = 40.e9
os = 2
fs = os*fb
N = 2**18
theta = np.pi/2.35
M = 16
snr = 24
muCMA = 1e-3
muRDE = 0.5e-3
ntaps = 30
t_pmd = 50e-12

sig = signals.ResampledQAM(M, N, nmodes=2, fb=fb, fs=fs, resamplekwargs={"beta":0.01, "renormalise":True})
sig = impairments.change_snr(sig, snr)
SS = impairments.apply_PMD(sig, theta, t_pmd)

E_s, wxy_s, (err_s, err_rde_s) = equalisation.dual_mode_equalisation(SS, (muCMA, muRDE), ntaps,
                                                                            methods=("mcma", "mrde"))
E_s = helpers.normalise_and_center(E_s)
evm = sig[:, ::2].cal_evm()
evmE_s = E_s.cal_evm()
gmiE = E_s.cal_gmi()
print(gmiE)


plt.figure()
plt.subplot(221)
plt.hexbin(E_s[0].real, E_s[0].imag)
plt.text(0.999, 0.9, r"$EVM_x={:.1f}\%$".format(100*evmE_s[0]), color='w', horizontalalignment="right", fontsize=14)
plt.subplot(222)
plt.title('Recovered MCMA/MRDE')
plt.hexbin(E_s[1].real, E_s[1].imag)
plt.text(0.999, 0.9, r"$EVM_y={:.1f}\%$".format(100*evmE_s[1]), color='w', horizontalalignment="right", fontsize=14)
plt.subplot(223)
plt.title('Original')
plt.hexbin(sig[0,::2].real, sig[0,::2].imag)
plt.text(0.999, 0.9, r"$EVM_x={:.1f}\%$".format(100*evm[0]), color='w', horizontalalignment="right", fontsize=14)
plt.subplot(224)
plt.hexbin(sig[1,::2].real, sig[1,::2].imag)
plt.text(0.999, 0.9, r"$EVM_y={:.1f}\%$".format(100*evm[1]), color='w', horizontalalignment="right", fontsize=14)

plt.figure()
plt.subplot(221)
plt.title('MCMA/MRDE Taps X-Pol')
plt.plot(wxy_s[0, 0,:], 'r')
plt.plot(wxy_s[0, 1,:], '--r')
plt.subplot(223)
plt.title('MCMA/MRDE Taps Y-Pol')
plt.plot(wxy_s[1, 0,:], 'g')
plt.plot(wxy_s[1, 1,:], '--g')
plt.subplot(222)
plt.title('MCMA/MRDE error mcma')
plt.plot(abs(err_s[0]), color='r')
plt.plot(abs(err_s[1]), color='g')
plt.subplot(224)
plt.title('MCMA/MRDE error mrde')
plt.plot(abs(err_rde_s[0]), color='r')
plt.plot(abs(err_rde_s[1]), color='g')
plt.show()

