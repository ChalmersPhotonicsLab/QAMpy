import numpy as np
import matplotlib.pylab as plt
from qampy import equalisation, signals, impairments, helpers
from qampy.core.equalisation.pythran_equalisation import apply_filter_to_signal
from timeit import default_timer as timer

fb = 40.e9
os = 2
fs = os*fb
N = 2**17
theta = 1* np.pi/3
theta2 = np.pi/4.8
M = 64
snr = 25
muCMA = 0.19e-2
muRDE = 0.19e-2
ntaps = 13
t_pmd = 30.e-12
Ncma = None
Nrde = None

sig = signals.ResampledQAM(M, N, nmodes=2, fb=fb, fs=fs, resamplekwargs={"beta":0.01, "renormalise":True})
sig = impairments.change_snr(sig, snr)

SS = impairments.apply_PMD(sig, theta, t_pmd)
t1 = timer()
E_s, wxy_s, (err_s, err_rde_s) = equalisation.dual_mode_equalisation(SS, (muCMA, muRDE), ntaps, TrSyms=(Ncma, Nrde),
                                                                     methods=("mcma", "mddma"),
                                                                     adaptive_stepsize=(True, True))
#E_s, wxy_s, err_s = equalisation.equalise_signal(SS, muCMA, Ntaps=ntaps, TrSyms=Ncma, method="cma", adaptive_stepsize=True, apply=True)
t2 = timer()
print("eqn time: {}".format(t2-t1))


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
plt.title('Recovered MCMA/SBD')
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
plt.title('MCMA/SBD Taps X-Pol')
plt.plot(wxy_s[0, 0,:], 'r')
plt.plot(wxy_s[0, 1,:], '--r')
plt.subplot(223)
plt.title('MCMA/SBD Taps Y-Pol')
plt.plot(wxy_s[1, 0,:], 'g')
plt.plot(wxy_s[1, 1,:], '--g')
plt.subplot(222)
plt.title('MCMA/SBD error mcma')
plt.plot(abs(err_s[0]), color='r')
plt.plot(abs(err_s[1]), color='g')
plt.subplot(224)
plt.title('MCMA/SBD error sbd')
plt.plot(abs(err_rde_s[0]), color='r')
plt.plot(abs(err_rde_s[1]), color='g')
plt.show()


