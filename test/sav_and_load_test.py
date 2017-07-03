import numpy as np
import matplotlib.pyplot as plt
from dsp import theory, ber_functions, modulation, utils, equalisation, IO
from scipy.signal import fftconvolve
import sys



"""
Check the calculation of EVM, BER, Q vs theoretical symbol error rate compare against _[1]


References
----------
...[1] Shafik, R. (2006). On the extended relationships among EVM, BER and SNR as performance metrics. In Conference on Electrical and Computer Engineering (p. 408). Retrieved from http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=4178493
"""

snr = np.linspace(5, 30, 4)
wls = [1550, 1552, 1554]
N = 2**16
M = 16
fb = 10
os = 2
fs = os*fb
mu = 1e-3
astep = True
method=("mcma","")
ntaps = 13
beta = 0.1
h5f = IO.tb.open_file("sav_recall.h5", "w", "multi wl and OSN msr")
IO.create_meas_group(h5f, "single polarisation generated signal", (1,N))
IO.create_input_group(h5f, "input symbols", N, int(N*np.log2(M)))
IO.create_parameter_group(h5f, "Measurement parameters")
IO.create_recvd_data_group(h5f, "Received data", 

id = 0
for wl in wls:
    for sr in snr:
        modulator = modulation.QAMModulator(M)
        signal, syms, bits = modulator.generateSignal(N, sr, samplingrate=fs, baudrate=fb, beta=beta)
        IO.save_osc_meas(h5f, signal, id, osnr=sr, wl=wl, samplingrate=fs, symbolrate=fb, MQAM=M)
        IO.save_inputs(h5f, id, symbols=syms, bits=bits)
        signal = np.atleast_2d(signal)
        signalx = np.atleast_2d(utils.rrcos_pulseshaping(signal[0], fs, 1/fb, beta))
        wx, er =  equalisation.equalise_signal(signalx, os, mu, M, Ntaps=ntaps, adaptive_step=astep, method=method[0])
        signalafter = equalisation.apply_filter(signalx, os, wx )
        evm_known[i] = modulator.cal_EVM(signalafter[0], syms)
        # check to see that we can recovery timing delay
        #signalafter = np.roll(signalafter * 1.j**np.random.randint(0,4), np.random.randint(4, 3000))
        ser[i] = modulator.calculate_SER(signalafter[0], symbol_tx=syms)[0]
        ber[i] = modulator.cal_BER(signalafter[0], bits)[0]

        id += 1

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


