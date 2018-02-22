import numpy as np
import matplotlib.pylab as plt
from dsp import equalisation, signals, impairments, helpers
#from dsp.core import impairments

fb = 40.e9
os = 2
fs = os*fb
N = 2**17
theta = 1* np.pi/4.5
theta2 = np.pi/4.3
M = 64
snr = 25
muCMA = 0.19e-2
muRDE = 0.19e-2
ntaps = 15
t_pmd = 100.e-12
Ncma = None
Nrde = None

sig = signals.ResampledQAM(M, N, nmodes=2, fb=fb, fs=fs, resamplekwargs={"beta":0.01, "renormalise":True})
sig = impairments.change_snr(sig, snr)

SS = impairments.apply_PMD_to_field(sig, theta, t_pmd)

E_m, wxy_m, (err_m, err_rde_m) = equalisation.dual_mode_equalisation(SS, (muCMA, muRDE), ntaps, TrSyms=(Ncma, Nrde),
                                                                     methods=("mcma", "mrde"),
                                                                     adaptive_stepsize=(True, True))
E_s, wxy_s, (err_s, err_rde_s) = equalisation.dual_mode_equalisation(SS, (muCMA, muRDE), ntaps, TrSyms=(Ncma, Nrde),
                                                                     methods=("mcma", "sbd"),
                                                                     adaptive_stepsize=(True, True))
E, wxy, (err, err_rde) = equalisation.dual_mode_equalisation(SS, (muCMA, muRDE), ntaps, TrSyms=(Ncma, Nrde),
                                                             methods=("mcma", "mddma"), adaptive_stepsize=(True, True))


E = helpers.normalise_and_center(E)
E_s = helpers.normalise_and_center(E_s)
E_m = helpers.normalise_and_center(E_m)
evm = sig[:,::2].cal_evm()
evmE = E.cal_evm()
evmE_m = E_m.cal_evm()
evmE_s = E_s.cal_evm()
gmiE = E.cal_gmi()
print(gmiE)


plt.figure()
plt.subplot(241)
plt.title('Recovered MCMA/MDDMA')
plt.hexbin(E[0].real, E[0].imag,  label=r"$EVM_x=%.1f\%%$"%(evmE[0]*100))
plt.legend()
plt.subplot(242)
plt.hexbin(E[1].real, E[1].imag,  label=r"$EVM_y=%.1f\%%$"%(100*evmE[1]))
plt.legend()
plt.subplot(243)
plt.title('Recovered MCMA/MRDE')
plt.hexbin(E_m[0].real, E_m[0].imag, label=r"$EVM_x=%.1f\%%$"%(evmE_m[0]*100))
plt.legend()
plt.subplot(244)
plt.hexbin(E_m[1].real, E_m[1].imag,  label=r"$EVM_y=%.1f\%%$"%(100*evmE_m[1]))
plt.legend()
plt.subplot(245)
plt.title('Recovered MCMA/SBD')
plt.hexbin(E_s[0].real, E_s[0].imag, label=r"$EVM_x=%.1f\%%$"%(evmE_s[0]*100))
plt.legend()
plt.subplot(246)
plt.hexbin(E_s[1].real, E_s[1].imag,  label=r"$EVM_y=%.1f\%%$"%(100*evmE_s[1]))
plt.legend()
plt.subplot(247)
plt.title('Original')
plt.hexbin(sig[0,::2].real, sig[0,::2].imag, label=r"$EVM_x=%.1f\%%$"%(100*evm[0]))
plt.legend()
plt.subplot(248)
plt.hexbin(sig[1,::2].real, sig[1,::2].imag,  label=r"$EVM_y=%.1f\%%$"%(100*evm[1]))
plt.legend()

plt.figure()
plt.subplot(331)
plt.title('CMA/MDDMA Taps')
plt.plot(wxy[0,0,:], 'r')
plt.plot(wxy[0,1,:], '--r')
plt.plot(wxy[1,0,:], 'g')
plt.plot(wxy[1, 1,:], '--g')
plt.subplot(332)
plt.title('CMA/MDDMA error cma')
plt.plot(abs(err[0]), color='r')
plt.plot(abs(err[1])- 10, color='g')
plt.subplot(333)
plt.title('CMA/MDDMA error MDDMA')
plt.plot(abs(err_rde[0]), color='r')
plt.plot(abs(err_rde[1])-10, color='g')
plt.subplot(334)
plt.title('MCMA/MRDE Taps')
plt.plot(wxy_m[0,0,:], 'r')
plt.plot(wxy_m[0,1,:], '--r')
plt.plot(wxy_m[1,0,:], 'g')
plt.plot(wxy_m[1,1,:], '--g')
plt.subplot(335)
plt.title('MCMA/MRDE error cma')
plt.plot(abs(err_m[0]), color='r')
plt.plot(abs(err_m[1])- 10, color='g')
plt.subplot(336)
plt.title('MCMA/MRDE error rde')
plt.plot(abs(err_rde_m[0]), color='r')
plt.plot(abs(err_rde_m[1])-10, color='g')
plt.subplot(337)
plt.title('MCMA/SBD Taps')
plt.plot(wxy_s[0, 0,:], 'r')
plt.plot(wxy_s[0, 1,:], '--r')
plt.plot(wxy_s[1, 0,:], 'g')
plt.plot(wxy_s[1, 1,:], '--g')
plt.subplot(338)
plt.title('MCMA/SBD error cma')
plt.plot(abs(err_s[0]), color='r')
plt.plot(abs(err_s[1])- 10, color='g')
plt.subplot(339)
plt.title('MCMA/SBD error rde')
plt.plot(abs(err_rde_s[0]), color='r')
plt.plot(abs(err_rde_s[1])-10, color='g')
plt.show()


