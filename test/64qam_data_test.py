import numpy as np
import matplotlib.pylab as plt
from dsp import  equalisation, modulation, utils, phaserecovery, signal_quality
from scipy.io import loadmat


os = 2
M = 64
QAM = modulation.QAMModulator(M)
snr = 35
muCMA = 1e-5
muRDE = 5e-5
Ncma = 100000
Nrde = 300000
ntaps = 33
niter = 4

#Dat = loadmat('data/OSNRLoading_1544p32_Att_2_OSNR_37_3.mat')
Dat = loadmat('data/SH_20Gbaud64QAM_test1.mat')
X = Dat['CH1'] + 1.j * Dat['CH2']
Y = Dat['CH3'] + 1.j * Dat['CH4']
X = X.flatten()
Y = Y.flatten()
#X = X[len(X)//2:]
#Y = Y[len(Y)//2:]

#X = utils.pre_filter(X, 2*3.9)
#Y = utils.pre_filter(Y, 2*3.9)
#X = utils.resample(X, 2.5, 2)
#Y = utils.resample(Y, 2.5, 2)
X = utils.rrcos_resample_zeroins(X, 2.5, 2, beta=0.05, Ts=1)
Y = utils.rrcos_resample_zeroins(Y, 2.5, 2, beta=0.05, Ts=1)
X = utils.comp_IQbalance(X)
Y = utils.comp_IQbalance(Y)
print(X.shape)
print(Y.shape)
SS = np.vstack([X[5000:-5000],Y[5000:-5000]])

SS = SS[:,:int(2e5)]

E, wxy, err_both = equalisation.dual_mode_equalisation(SS, os, (muCMA, muRDE), M, ntaps, Niter=(5,5), methods=("mcma", "sbd"), adaptive_stepsize=(True,True) )

X = signal_quality.normalise_sig(E[0,:],M)[1]
Y = signal_quality.normalise_sig(E[1,:],M)[1]    
E = np.vstack([X,Y])

foe = phaserecovery.find_freq_offset(E,fft_size = 2**10)


E = phaserecovery.comp_freq_offset(E,foe)

#Ec = E[:,2e4:-2e4]
wx, err_both = equalisation.equalise_signal(E, 1, muRDE, M,Niter=4, Ntaps=ntaps, method="sbd" , adaptive_stepsize=False)
Ec = equalisation.apply_filter(E, 1, wx)
E = Ec


print("X pol phase")
Ex, phx = phaserecovery.bps(E[0], 32, QAM.symbols, 8)
print("X pol phase done")
print("Y pol phase")
Ey, phy = phaserecovery.bps(E[1], 32, QAM.symbols, 8)
print("Y pol phase done")
Ec = np.vstack([Ex,Ey])

evmX = QAM.cal_evm(X[::2])
evmY = QAM.cal_evm(Y[::2])
evmEx = QAM.cal_evm(E[0])
evmEy = QAM.cal_evm(E[1])
#Ec[0] = signal_quality.normalise_sig(Ec[0], M)[1]
#Ec[1] = signal_quality.normalise_sig(Ec[1], M)[1]
evmEx_c = QAM.cal_evm(Ec[0])
evmEy_c = QAM.cal_evm(Ec[1])
print(evmEy_c)
print(evmEx_c)
#sys.exit()
plt.figure()
plt.subplot(221)
#plt.title(r'After Phaserecovery $EVM_x=%.1f\%%$'%(evmEx_c*100))
plt.hexbin(Ec[0].real, Ec[0].imag)
plt.subplot(222)
#plt.title(r'After Phaserecovery $EVM_y=%.1f\%%$'%(evmEy_c*100))
plt.hexbin(Ec[1].real, Ec[1].imag)
plt.subplot(223)
#plt.title(r'Before Phaserecovery $EVM_x=%.1f\%%$'%(evmEx*100))
plt.hexbin(E[0].real, E[0].imag)
plt.subplot(224)
#plt.title(r'Before Phaserecovery $EVM_y=%.1f\%%$'%(evmEy*100))
plt.hexbin(E[1].real, E[1].imag)
plt.show()
