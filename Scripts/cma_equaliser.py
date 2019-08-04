import numpy as np
import matplotlib.pylab as plt
from qampy import equalisation, signals, impairments, helpers, phaserec

fb = 40.e9
os = 2
fs = os*fb
N = 4*10**5
mu = 4e-4
theta = np.pi/5.45
theta2 = np.pi/2.1
t_pmd = 75e-12
M = 4
ntaps=40
snr =  14

sig = signals.SignalQAMGrayCoded(M, N, fb=fb, nmodes=2, dtype=np.complex64)
S = sig.resample(fs, renormalise=True, beta=0.1)
S = impairments.apply_phase_noise(S, 100e3)
S = impairments.change_snr(S, snr)

SS = impairments.apply_PMD(S, theta, t_pmd)
wxy, err = equalisation.equalise_signal(SS, mu, Ntaps=ntaps, TrSyms=None, method="cma", adaptive_step=True)
wxy_m, err_m = equalisation.equalise_signal(SS, mu,  TrSyms=None,Ntaps=ntaps, method="mcma", adaptive_step=True)
E = equalisation.apply_filter(SS,  wxy)
E_m = equalisation.apply_filter(SS, wxy_m)
E = helpers.normalise_and_center(E)
E_m = helpers.normalise_and_center(E_m)
E, ph = phaserec.viterbiviterbi(E, 11)
E_m, ph = phaserec.viterbiviterbi(E_m, 11)
E = helpers.dump_edges(E, 20)
E_m = helpers.dump_edges(E_m, 20)


# note that because of the noise we get sync failures doing SER
gmi = E.cal_gmi()[0]
gmi_m = E_m.cal_gmi()[0]
gmi0 = S[:, ::2].cal_gmi()[0]

plt.figure()
plt.subplot(221)
plt.title('Recovered CMA X')
plt.hexbin(E[0].real, E[0].imag)
plt.text(0.999, 0.9, r"$GMI_x={:.1f}$".format(gmi[0]), color='w', horizontalalignment="right", fontsize=14)
plt.subplot(222)
plt.title('Recovered CMA Y')
plt.hexbin(E[1].real, E[1].imag)
plt.text(0.999, 0.9, r"$GMI_y={:.1f}$".format(gmi[1]), color='w', horizontalalignment="right", fontsize=14)
plt.subplot(223)
plt.title('Recovered MCMA X')
plt.hexbin(E_m[0].real, E_m[0].imag)
plt.text(0.999, 0.9, r"$GMi_x={:.1f}$".format(gmi_m[0]), color='w', horizontalalignment="right", fontsize=14)
plt.subplot(224)
plt.title('Recovered MCMA Y')
plt.hexbin(E_m[1].real, E_m[1].imag)
plt.text(0.999, 0.9, r"$GMI_y={:.1f}$".format(gmi_m[1]), color='w', horizontalalignment="right", fontsize=14)


fig, ax = plt.subplots(2,2)
plt.title('Taps CMA')
ax[0, 0].plot(wxy[0][0,:], 'r')
plt.axis([0, ntaps, -1, 1])
ax[0, 1].plot(wxy[0][1,:], '--r')
plt.axis([0, ntaps, -1, 1])
ax[1, 0].plot(wxy[1][0,:], 'g')
plt.axis([0, ntaps, -1, 1])
ax[1, 1].plot(wxy[1][1,:], '--g')
plt.axis([0, ntaps, -1, 1])

fig, ax = plt.subplots(2,2)
plt.title('Taps MCMA')
ax[0, 0 ].plot(wxy_m[0][0,:], 'r')
plt.axis([0, ntaps, -1, 1])
ax[0, 1].plot(wxy_m[0][1,:], '--r')
plt.axis([0, ntaps, -1, 1])
ax[1, 0].plot(wxy_m[1][0,:], 'g')
plt.axis([0, ntaps, -1, 1])
ax[1, 1].plot(wxy_m[1][1,:], '--g')
plt.axis([0, ntaps, -1, 1])
plt.show()



