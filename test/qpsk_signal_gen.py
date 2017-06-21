import numpy as np
import matplotlib.pyplot as plt
from dsp import  theory, signal_quality, modulation


""" Check the symbol rate of a QPSK signal against the theoretical symbol rate"""

snr = np.arange(2, 15, 1)
ser = []
ser2 = []
QAM = modulation.QAMModulator(4)
N = 10**5

for sr in snr:
    data_rx, data_tx, bits2 = QAM.generate_signal(N, sr, IQsep=True)
    ser.append(signal_quality.cal_ser_QAM(data_rx, data_tx, 4))
plt.figure()
plt.plot(snr, 10*np.log10(theory.ser_vs_es_over_n0_mpsk(10**(snr/10.), 4)), label='theory')
plt.plot(snr, 10*np.log10(ser), 'or',label='calculation')
plt.xlabel('SNR [dB]')
plt.ylabel('SER [dB]')
plt.legend()
plt.show()

