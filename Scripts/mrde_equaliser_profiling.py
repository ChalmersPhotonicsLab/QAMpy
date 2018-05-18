import numpy as np
from qampy import equalisation, signals
from qampy.core import impairments
from timeit import default_timer as timer




fb = 40.e9
os = 2
fs = os*fb
N = 10**6
theta = np.pi/2.35
M = 16
QAM = signals.QAMModulator(16)
snr = 24
muCMA = 3e-4
muRDE = 3e-4
ntaps = 30
trsyms = N//os//2-(ntaps+5) # use full width for training
methods = ("mcma", "sbd")

S, symbols, bits = QAM.generate_signal(N, snr,  baudrate=fb, samplingrate=fs, PRBSorder=(15,23))

t_pmd = 50e-12

SS = impairments.apply_PMD_to_field(S, theta, t_pmd, fs)

t1 = timer()
E, (wx, wy), (err, err_rde) = equalisation.dual_mode_equalisation(SS, os, M, ntaps, methods=methods, beta=4)
t2 = timer()
print("EQ time: %.1f"%(t2-t1))


evmX = QAM.cal_evm(S[0, ::2])
evmY = QAM.cal_evm(S[1, ::2])
evmEx = QAM.cal_evm(E[0])
evmEy = QAM.cal_evm(E[1])
print(evmEx)
print(evmY)
#sys.exit()
