from dsp import signals

"""
Check the calculation of EVM, BER, Q vs theoretical symbol error rate compare against _[1]


References
----------
...[1] Shafik, R. (2006). On the extended relationships among EVM, BER and SNR as performance metrics. In Conference on Electrical and Computer Engineering (p. 408). Retrieved from http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=4178493
"""

M = 128
N = 10**6
snr = 30
modulator = signals.QAMModulator(M)
signal, syms, bits = modulator.generate_signal(N, snr, dual_pol=False)
evm = modulator.cal_evm(signal)
print(evm)
