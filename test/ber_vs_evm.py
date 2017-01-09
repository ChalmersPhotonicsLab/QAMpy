import numpy as np
import matplotlib.pyplot as plt
from dsp import signals, theory, ber_functions, signal_quality, modulation


""" Check the calculation of EVM, BER, Q vs theoretical symbol error rate"""

snr = np.arange(5, 17, 1)
ser = []
ber = []
ser_class = []
ber_class = []
evm = []
snr2 = []
modl = modulation.QAMModulator(4)
N = 10**6


for sr in snr:
    print(sr)
    data_rx, dataI, dataQ = signals.generateRandomQPSKData(N, sr)
    data_tx = 2*(dataI+1.j*dataQ-0.5-0.5j)
    ser.append(signal_quality.cal_ser_QAM(data_rx, data_tx, 4))
    data_rx2, symbols2, bits2 = modl.generateSignal(N, sr)
    ser_class.append(modl.calculate_SER(data_rx2, symbol_tx=symbols2))
    try:
        ber.append(ber_functions.cal_BER_QPSK_prbs(data_rx, 15, 23)[0])
    except:
        ber.append(1)
    try:
        ber_class.append(modl.cal_BER(data_rx2, bits_tx=bits2)[0])
    except:
        ber_class.append(1)
    snr2.append(signal_quality.cal_SNR_QAM(data_rx, 4))
    evm.append(signal_quality.cal_blind_evm(data_rx, 4))
evm = np.array(evm)
ber = np.array(ber)
snr2 = np.array(snr2)
ser_class = np.array(ser_class)
ber_class = np.array(ber_class)
plt.figure()
ax = plt.subplot(411)
plt.title(r"BER vs SNR")
plt.plot(snr, ber, 'bo', label="calculated BER")
plt.plot(snr, ber_class, 'ro', label="calculated BER from class")
plt.plot(snr, theory.MPSK_SERvsEsN0(10**(snr/10.), 4)/2, label='symbol error rate/2 theory')
#plt.plot(snr, 10*np.log10(ser), 'or',label='calculation')
plt.xlabel('SNR [dB]')
plt.ylabel('BER [dB]')
plt.legend()
ax.set_yscale('log')
ax2 = plt.subplot(412)
plt.title(r"EVM vs BER")
plt.plot(ber, evm, 'b', label="EVM")
#plt.plot(snr, 10*np.log10(ser), 'or',label='calculation')
plt.xlabel('EVM [dB]')
plt.ylabel('BER [dB]')
ax2.set_yscale('log')
plt.legend()
ax3 = plt.subplot(413)
plt.title(r"$EVM^2$ vs SNR")
plt.plot(snr, 10*np.log10(evm**2), 'ro', label=r"$EVM^2")
plt.plot(snr, -snr,  label=r"SNR")
plt.plot(snr, -10*np.log10(snr2), 'go', label="SNR calculated")
plt.xlabel('SNR [dB]')
plt.ylabel('$EVM^2$ [dB]')
plt.legend()
ax4 = plt.subplot(414)
plt.title(r"SER vs SNR")
plt.plot(snr, ser, 'bo', label="calculated SER")
plt.plot(snr, ser_class, 'ro', label="calculated SER from class")
plt.plot(snr, theory.MPSK_SERvsEsN0(10**(snr/10.), 4), label='symbol error rate theory')
#plt.plot(snr, 10*np.log10(ser), 'or',label='calculation')
plt.xlabel('SNR [dB]')
plt.ylabel('SER [dB]')
plt.legend()
ax4.set_yscale('log')
plt.show()

