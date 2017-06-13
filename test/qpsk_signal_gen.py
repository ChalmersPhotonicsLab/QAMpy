import numpy as np
import matplotlib.pyplot as plt
from dsp import signals, theory, signal_quality, modulation


""" Check the symbol rate of a QPSK signal against the theoretical symbol rate"""

snr = np.arange(2, 15, 1)
ser = []
ser2 = []
QAM = modulation.QAMModulator(4)
N = 10**5

for sr in snr:
    data_rx, dataI, dataQ = signals.generateRandomQPSKData(N, sr)
    data_rx2, symbols2, bits2 = QAM.generate_signal(N, sr, IQsep=True)
    data_tx = 2*(dataI+1.j*dataQ-0.5-0.5j)
    ser.append(signal_quality.cal_ser_QAM(data_rx, data_tx, 4))
    ser2.append(QAM.cal_SER(data_rx2, symbol_tx=symbols2))
plt.figure()
plt.plot(snr, 10*np.log10(theory.ser_vs_esn0_mpsk(10**(snr/10.), 4)), label='theory')
plt.plot(snr, 10*np.log10(ser), 'or',label='calculation')
plt.plot(snr, 10*np.log10(ser2), 'ob',label='class calculation')
plt.xlabel('SNR [dB]')
plt.ylabel('SER [dB]')
plt.legend()
plt.show()

