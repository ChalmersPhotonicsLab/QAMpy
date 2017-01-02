import numpy as np
import matplotlib.pyplot as plt
from dsp import signals, theory, ber_functions, signal_quality


""" Check the calculation of EVM, BER, Q vs theoretical symbol error rate"""

snr = np.arange(5, 17, 1)
ser = []
ber = []
evm = []
snr2 = []
for sr in snr:
    print(sr)
    data_rx, dataI, dataQ = signals.generateRandomQPSKData(10**6, sr)
    data_tx = 2*(dataI+1.j*dataQ-0.5-0.5j)
    ser.append(signals.cal_ser_qpsk(data_rx, data_tx))
    try:
        ber.append(ber_functions.cal_BER_QPSK_prbs(data_rx,15, 23)[0])
    except:
        ber.append(1)
    snr2.append(signal_quality.cal_SNR_QAM(data_rx, 4))
    evm.append(signal_quality.cal_blind_evm(data_rx, 4))
evm = np.array(evm)
ber = np.array(ber)
snr2 = np.array(snr2)
plt.figure()
ax = plt.subplot(311)
plt.title(r"BER vs SNR")
plt.plot(snr, ber, 'bo', label="calculated BER")
plt.plot(snr, theory.MPSK_SERvsEsN0(10**(snr/10.), 4)/2, label='symbol error rate/2 theory')
#plt.plot(snr, 10*np.log10(ser), 'or',label='calculation')
plt.xlabel('SNR [dB]')
plt.ylabel('BER [dB]')
plt.legend()
ax.set_yscale('log')
ax2 = plt.subplot(312)
plt.title(r"EVM vs BER")
plt.plot(ber, evm, 'b', label="EVM")
#plt.plot(snr, 10*np.log10(ser), 'or',label='calculation')
plt.xlabel('EVM [dB]')
plt.ylabel('BER [dB]')
ax2.set_yscale('log')
plt.legend()
ax3 = plt.subplot(313)
plt.title(r"$EVM^2$ vs SNR")
plt.plot(snr, 10*np.log10(evm**2), 'ro', label=r"$EVM^2")
plt.plot(snr, -snr,  label=r"SNR")
plt.xlabel('SNR [dB]')
plt.ylabel('$EVM^2$ [dB]')
plt.legend()
plt.show()

