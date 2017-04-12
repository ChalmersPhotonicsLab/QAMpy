import numpy as np
import matplotlib.pylab as plt
from dsp import signals, equalisation, modulation, utils, phaserecovery
from scipy.io import loadmat


os = 2
M = 64
QAM = modulation.QAMModulator(M)
snr = 35
muCMA = 4e-3
muRDE = 2e-3
Ncma = 100000
Nrde = 300000
ntaps = 33
niter = 4

Dat = loadmat('OSNRLoading_1544p32_Att_2_OSNR_37_3.mat')

X = Dat['CH1'] + 1.j * Dat['CH2']
Y = Dat['CH3'] + 1.j * Dat['CH4']
X = X.flatten()
Y = Y.flatten()
#X = X[len(X)//2:]
#Y = Y[len(Y)//2:]

X = utils.pre_filter(X, 2*3.9)
Y = utils.pre_filter(Y, 2*3.9)
X = utils.resample(2.5, 2, X)
Y = utils.resample(2.5, 2, Y)
X = utils.comp_IQbalance(X)
Y = utils.comp_IQbalance(Y)
SS = np.vstack([X[5000:-5000],Y[5000:-5000]])


E, wxy, err_both = equalisation.dual_mode_equalisation(SS, os, (muCMA, muRDE), M, ntaps, Niter=(1,1), methods=("mcma", "sbd"), adaptive_stepsize=(True,True) )
print("X pol phase")
Ex, phx = phaserecovery.blindphasesearch(E[0], 32, QAM.symbols, 8)
print("X pol phase done")
print("Y pol phase")
Ey, phy = phaserecovery.blindphasesearch(E[1], 32, QAM.symbols, 8)
print("Y pol phase done")
Ec = np.vstack([Ex,Ey])

evmX = QAM.cal_EVM(X[::2])
evmY = QAM.cal_EVM(Y[::2])
evmEx = QAM.cal_EVM(E[0])
evmEy = QAM.cal_EVM(E[1])
evmEx_c = QAM.cal_EVM(Ec[0])
evmEy_c = QAM.cal_EVM(Ec[1])
#sys.exit()
plt.figure()
plt.subplot(221)
plt.title(r'After Phaserecovery $EVM_x=%.1f\%%$'%(evmEx_c*100))
plt.hexbin(Ec[0].real, Ec[0].imag)
plt.subplot(222)
plt.title(r'After Phaserecovery $EVM_y=%.1f\%%$'%(evmEy_c*100))
plt.hexbin(Ec[1].real, Ec[1].imag)
plt.subplot(223)
plt.title(r'Before Phaserecovery $EVM_x=%.1f\%%$'%(evmEx*100))
plt.hexbin(E[0].real, E[0].imag)
plt.subplot(224)
plt.title(r'Before Phaserecovery $EVM_y=%.1f\%%$'%(evmEy*100))
plt.hexbin(E[1].real, E[1].imag)
plt.show()
