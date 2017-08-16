import numpy as np
import matplotlib.pyplot as plt
from dsp import theory, ber_functions, modulation, utils, equalisation, filter
from scipy.signal import fftconvolve
import sys



"""
Check the calculation of EVM, BER, Q vs theoretical symbol error rate compare against _[1]


References
----------
...[1] Shafik, R. (2006). On the extended relationships among EVM, BER and SNR as performance metrics. In Conference on Electrical and Computer Engineering (p. 408). Retrieved from http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=4178493
"""

snr = np.linspace(5, 30, 8)
snrf = np.linspace(5, 30, 500)
evmf = np.linspace(-30, 0, 500)
N = 2**16
Mqams = [ 4, 16, 64, 128]

plt.figure()
ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(223)
ax4 = plt.subplot(224)

ax1.set_title("BER vs SNR")
ax2.set_title("SER vs SNR")
ax3.set_title("BER vs EVM")
ax4.set_title("EVM vs SNR")
ax1.set_xlabel('SNR [dB]')
ax1.set_ylabel('BER [dB]')
ax1.set_yscale('log')
ax1.set_xlim(0,30)
ax1.set_ylim(1e-5,1)
ax2.set_xlabel('SNR [dB]')
ax2.set_ylabel('SER [dB]')
ax2.set_yscale('log')
ax2.set_xlim(0,30)
ax2.set_ylim(1e-5,1)
ax3.set_xlabel('EVM [dB]')
ax3.set_ylabel('BER [dB]')
ax3.set_yscale('log')
ax3.set_xlim(-30,0)
ax3.set_ylim(1e-6,1)
ax4.set_xlabel('SNR [dB]')
ax4.set_ylabel('EVM [dB]')
ax4.set_xlim(0, 30)
ax4.set_ylim(-30, 0)

c = ['b', 'r', 'g', 'c', 'k']
s = ['o', '<', 's', '+', 'd']
j = 0
fb = 10e9
os = 2
fs = os*fb
ntaps = 13
beta = 0.01
for M in Mqams:
    print("%d-QAM"%M)
    ser = np.zeros(snr.shape)
    ber = np.zeros(snr.shape)
    evm1 = np.zeros(snr.shape)
    evm_known = np.zeros(snr.shape)
    q_known = np.zeros(snr.shape)
    tt = []
    ox = []

    i = 0
    for sr in snr:
        print("SNR = %2f.0 dB"%sr)
        modulator = modulation.QAMModulator(M)
        signal, syms, bits = modulator.generate_signal(N, sr, samplingrate=fs, baudrate=fb, beta=beta, dual_pol=False)
        signalx = np.atleast_2d(filter.rrcos_pulseshaping(signal, fs, 1/fb, beta))
        signalafter = np.atleast_2d(signalx[0,::2])
        evm1[i] = modulator.cal_evm(signalafter[0])
        evm_known[i] = modulator.cal_evm(signalafter[0], syms)
        # check to see that we can recovery timing delay
        signalafter = np.roll(signalafter * 1.j**np.random.randint(0,4), np.random.randint(4, 3000))
        ser[i] = modulator.cal_ser(signalafter[0], symbol_tx=syms)
        ber[i] = modulator.cal_ber(signalafter[0], bits_tx=bits)[0]
        i += 1
    ax1.plot(snrf, theory.ber_vs_es_over_n0_qam(10**(snrf/10), M), color=c[j], label="%d-QAM theory"%M)
    ax1.plot(snr, ber, color=c[j], marker=s[j], lw=0, label="%d-QAM"%M)
    ax2.plot(snrf, theory.ser_vs_es_over_n0_qam(10**(snrf/10), M), color=c[j], label="%d-QAM theory"%M)
    ax2.plot(snr, ser, color=c[j], marker=s[j], lw=0, label="%d-QAM"%M)
    ax3.plot(evmf, theory.ber_vs_evm_qam(evmf, M), color=c[j], label="%d-QAM theory"%M)
    ax3.plot(utils.lin2dB(evm1**2), ber, color=c[j], marker=s[j], lw=0, label="%d-QAM"%M)
    # illustrate the difference between a blind and non-blind EVM
    ax3.plot(utils.lin2dB(evm_known**2), ber, color=c[j], marker='*', lw=0, label="%d-QAM non-blind"%M)
    ax4.plot(snr, utils.lin2dB(evm1**2), color=c[j], marker=s[j], lw=0, label="%d-QAM"%M)
    ax4.plot(snr, utils.lin2dB(evm_known**2), color=c[j], marker='*', lw=0, label="%d-QAM non-blind"%M)
    #ax4.plot(snr, utils.lin2dB(evm2), color=c[j], marker='*', lw=0, label="%d-QAM signalq"%M)
    j += 1
ax1.legend()
ax2.legend()
#ax3.legend()
ax4.legend()
plt.show()


