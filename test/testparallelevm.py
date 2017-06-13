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

M = 128
N = 10**6
snr = 30
modulator = modulation.QAMModulator(M)
signal, syms, bits = modulator.generate_signal(N, snr)
evm = modulator.cal_EVM(signal)
print(evm)
