import numpy as np
import matplotlib.pyplot as plt
from dsp import theory, ber_functions, signal_quality, modulation, utils
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
N = 10**5
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



c = ['b', 'r', 'g', 'c']
s = ['o', '<', 's', '+']
j = 0
for M in Mqams:
    print("%d-QAM"%M)
    ser = np.zeros(snr.shape)
    ber = np.zeros(snr.shape)
    evm1 = np.zeros(snr.shape)
    #evm2 = np.zeros(snr.shape)
    evm_known = np.zeros(snr.shape)
    i = 0
    for sr in snr:
        print("SNR = %2f.0 dB"%sr)
        modulator = modulation.QAMModulator(M)
        signal, syms, bits = modulator.generateSignal(N, sr)
        evm1[i] = modulator.cal_EVM(signal)
        #evm2[i] = signal_quality.cal_blind_evm(signal, M)
        evm_known[i] = modulator.cal_EVM(signal, syms)
        # check to see that we can recovery timing delay
        signal = np.roll(signal * 1.j**np.random.randint(0,4), np.random.randint(4, 3000))
        ser[i] = modulator.calculate_SER(signal, symbol_tx=syms)[0]
        ber[i] = modulator.cal_BER(signal, bits)[0]
        i += 1
    ax1.plot(snrf, theory.MQAM_BERvsEsN0(10**(snrf/10), M), color=c[j], label="%d-QAM theory"%M)
    ax1.plot(snr, ber, color=c[j], marker=s[j], lw=0, label="%d-QAM"%M)
    ax2.plot(snrf, theory.MQAM_SERvsEsN0(10**(snrf/10), M), color=c[j], label="%d-QAM theory"%M)
    ax2.plot(snr, ser, color=c[j], marker=s[j], lw=0, label="%d-QAM"%M)
    ax3.plot(evmf, theory.MQAM_BERvsEVM(evmf, M), color=c[j], label="%d-QAM theory"%M)
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

