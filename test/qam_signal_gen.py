import numpy as np
import matplotlib.pyplot as plt
from dsp import signals, theory, modulation


""" Check the symbol rate of a QAM signal against the theoretical symbol rate"""

M = 32
snr = np.arange(5, 25, 1)
N = 10**5
ser = []
ser2 = []

modl = modulation.QAMModulator(M)
for sr in snr:
    data_rx, symbols, bits = modl.generateSignal(N, sr)
    ser.append(modl.calculate_SER(data_rx, symbol_tx=symbols)[0])
    ser2.append(modl.calculate_SER(data_rx, bits_tx=bits)[0])

ser = 10*np.log10(np.array(ser))
ser2 = 10*np.log10(np.array(ser2))
theory_ser = 10*np.log10(modl.theoretical_SER(10**(snr/10)))
assert np.allclose(ser, ser2), "SER calculated from symbol is different to ser calculated from bits"
#assert np.allclose(ser, theory_ser, atol=1), "SER calculated from symbol is different to theoretical ser"
plt.figure()
plt.plot(snr, theory_ser, label='theory')
plt.plot(snr, ser, 'or',label='calculation')
plt.xlabel('SNR [dB]')
plt.ylabel('SER [dB]')
plt.legend()
plt.show()

