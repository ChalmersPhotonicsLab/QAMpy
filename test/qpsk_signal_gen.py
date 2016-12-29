import numpy as np
import matplotlib.pyplot as plt
from dsp import signals, theory


""" Check the symbol rate of a QPSK signal against the theoretical symbol rate"""

snr = np.arange(2, 15, 1)
ser = []
for sr in snr:
    data_rx, dataI, dataQ = signals.generateRandomQPSKData(10**6, sr)
    data_tx = 2*(dataI+1.j*dataQ-0.5-0.5j)
    ser.append(signals.cal_ser_qpsk(data_rx, data_tx))
plt.figure()
plt.plot(snr, 10*np.log10(theory.MPSK_SERvsEsN0(10**(snr/10.), 4)), label='theory')
plt.plot(snr, 10*np.log10(ser), 'or',label='calculation')
plt.xlabel('SNR [dB]')
plt.ylabel('SER [dB]')
plt.legend()
plt.show()

