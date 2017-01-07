import numpy as np
import matplotlib.pyplot as plt
from dsp import signals, theory, modulation


""" Check the symbol rate of a QAM signal against the theoretical symbol rate"""

M = 32
snr = np.arange(5, 25, 1)
N = 10**5
ser = []

modl = modulation.QAMModulator(M)
for sr in snr:
    data_rx, symbols, bits = signals.generate_MQAM_data_signal(N, sr, modl)
    ser.append(np.count_nonzero(modl.quantize(data_rx)[0]-symbols)/N)
plt.figure()
plt.plot(snr, 10*np.log10(theory.MQAM_SERvsEsN0(10**(snr/10.), M)), label='theory')
plt.plot(snr, 10*np.log10(ser), 'or',label='calculation')
plt.xlabel('SNR [dB]')
plt.ylabel('SER [dB]')
plt.legend()
plt.show()

