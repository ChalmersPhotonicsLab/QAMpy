import numpy as np
import matplotlib.pyplot as plt
from dsp import signals, theory, signal_quality


""" Check the symbol rate of a QAM signal against the theoretical symbol rate"""

M = 16
snr = np.arange(2, 15, 1)
ser = []
for sr in snr:
    data_rx, symbols, bits = signals.generateRandomQPSKData(10**6, sr)
    ser.append(signal_quality.cal_ser_QAM(data_rx, symbols, M))
plt.figure()
plt.plot(snr, 10*np.log10(theory.MQAM_SERvsEsN0(10**(snr/10.), M)), label='theory')
plt.plot(snr, 10*np.log10(ser), 'or',label='calculation')
plt.xlabel('SNR [dB]')
plt.ylabel('SER [dB]')
plt.legend()
plt.show()

