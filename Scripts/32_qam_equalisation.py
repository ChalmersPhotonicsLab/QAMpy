import numpy as np
import matplotlib.pylab as plt
from qampy import equalisation, signals, impairments, helpers



fb = 40.e9
os = 2
fs = os*fb
N = 10**6
theta = np.pi/4.6
M = 32
snr = 25
muCMA = 1e-3
muRDE = 1.e-3
ntaps = 11
t_pmd = 20.e-12
#Ncma = N//4//os -int(1.5*ntaps)
Ncma = 10000
Nrde = N//2//os -int(1.5*ntaps)

#S, symbols, bits = QAM.generate_signal(N, snr,  baudrate=fb, samplingrate=fs, PRBSorder=(15,23))
sig = signals.SignalQAMGrayCoded(M, N, fb=fb, nmodes=2)
S  = sig.resample(fs, beta=0.1, renormalise=True)
S = impairments.change_snr(S, snr)

SS = impairments.apply_PMD(S, theta, t_pmd)

E , wx, (err, err_rde) = equalisation.dual_mode_equalisation(SS, (muCMA, muRDE), ntaps, methods=("mcma", "sbd"),
                                                             adaptive=(True, True))
E = helpers.normalise_and_center(E)
evm = E.cal_evm()
evm_s = S[:, ::2].cal_evm()

#sys.exit()
plt.figure()
plt.subplot(121)
plt.title('Recovered MCMA/MDDMA')
plt.plot(E[0].real, E[0].imag, 'r.' ,label=r"$EVM_x=%.1f\%%$"%(evm[0]*100))
plt.plot(E[1].real, E[1].imag, 'g.', label=r"$EVM_y=%.1f\%%$"%(100*evm[1]))
plt.legend()
plt.subplot(122)
plt.title('Original')
plt.plot(S[0,::2].real, S[0,::2].imag, 'r.', label=r"$EVM_x=%.1f\%%$"%(100*evm_s[0]))
plt.plot(S[1,::2].real, S[1,::2].imag, 'g.', label=r"$EVM_y=%.1f\%%$"%(100*evm_s[1]))
plt.legend()

plt.figure()
plt.subplot(221)
plt.title('Taps')
plt.plot(wx[0,0,:], 'r')
plt.plot(wx[0,1,:], '--r')
plt.plot(wx[1,0,:], 'g')
plt.plot(wx[1, 1,:], '--g')
plt.subplot(223)
plt.title('CMA error')
plt.plot(abs(err[0]), color='r')
plt.plot(abs(err[1])- 10, color='g')
plt.subplot(223)
plt.title('SBD error')
plt.plot(abs(err_rde[0]), color='r')
plt.plot(abs(err_rde[1])-10, color='g')
plt.show()


