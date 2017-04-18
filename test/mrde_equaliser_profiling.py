import numpy as np
import matplotlib.pylab as plt
from dsp import signals, equalisation, modulation, utils



fb = 40.e9
os = 2
fs = os*fb
N = 10**6
theta = np.pi/2.35
M = 16
QAM = modulation.QAMModulator(16)
snr = 24
muCMA = 3e-4
muRDE = 3e-4
ntaps = 30
trsyms = N//os//2-(ntaps+5) # use full width for training
methods = ("mcma", "mrde")

X, Xsymbols, Xbits = QAM.generateSignal(N, snr,  baudrate=fb, samplingrate=fs, PRBSorder=15)
Y, Ysymbols, Ybits = QAM.generateSignal(N, snr, baudrate=fb, samplingrate=fs, PRBSorder=23)

omega = 2*np.pi*np.linspace(-fs/2, fs/2, N*os, endpoint=False)
t_pmd = 50e-12

SS = utils.apply_PMD_to_field(np.vstack([X,Y]), theta, t_pmd, omega)

E, (wx, wy), (err, err_rde) = equalisation.dual_mode_equalisation(SS, os, (muCMA, muRDE), M, ntaps, methods=methods)


evmX = QAM.cal_EVM(X[::2])
evmY = QAM.cal_EVM(Y[::2])
evmEx = QAM.cal_EVM(E[0])
evmEy = QAM.cal_EVM(E[1])
#sys.exit()
