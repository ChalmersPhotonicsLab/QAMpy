import numpy as np
import matplotlib.pylab as plt
from dsp import signals, equalisation, modulation, utils
from dsp.signal_quality import cal_blind_evm
from scipy.io import loadmat



def H_PMD(theta, t, omega): #see Ip and Kahn JLT 25, 2033 (2007)
    """"""
    h1 = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    h2 = np.array([[np.exp(1.j*omega*t/2), np.zeros(len(omega))],[np.zeros(len(omega)), np.exp(-1.j*omega*t/2)]])
    h3 = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    H = np.einsum('ij,jkl->ikl', h1, h2)
    H = np.einsum('ijl,jk->ikl', H, h3)
    return H

def rotate_field(theta, field):
    h = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.dot(h, field)

def applyPMD(field, H):
    Sf = np.fft.fftshift(np.fft.fft(np.fft.fftshift(field, axes=1),axis=1), axes=1)
    SSf = np.einsum('ijk,ik -> ik',H , Sf)
    SS = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(SSf, axes=1),axis=1), axes=1)
    return SS

def compIQbalance(signal):
    signal -= np.mean(signal)
    I = signal.real
    Q = signal.imag

    # phase balance
    mon_signal = np.sum(I*Q)/np.sum(I**2)
    phase_inbalance = np.arcsin(-mon_signal)
    Q_balcd = (Q + np.sin(phase_inbalance)*I)/np.cos(phase_inbalance)
    am_bal = np.sum(I**2)/np.sum(Q_balcd**2)
    Q_comp = Q_balcd * np.sqrt(am_bal)
    print("Phase imbalance: %g"%phase_inbalance)
    print("Amplitude imbalance: %g"%am_bal)
    return I + 1.j * Q_comp


def pre_filter(signal, bw):
    N = len(signal)
    h = np.zeros(N, dtype=np.float64)
    h[int(N/bw):-int(N/bw)] = 1
    s = np.fft.ifft(np.fft.ifftshift(np.fft.fftshift(np.fft.fft(signal))*h))
    return s

os = 2
M = 64
QAM = modulation.QAMModulator(M)
snr = 35
muCMA = 5e-3
muRDE = 2e-3
ntaps = 33
#Ncma = N//4//os -int(1.5*ntaps)
Ncma = 300000
Nrde = 300000

Dat = loadmat('OSNRLoading_1544p32_Att_2_OSNR_37_3.mat')

X = Dat['CH1'] + 1.j * Dat['CH2']
Y = Dat['CH3'] + 1.j * Dat['CH4']
X = X.flatten()
Y = Y.flatten()

X = pre_filter(X, 3.75)
Y = pre_filter(Y, 3.75)
X = utils.resample(2.5, 2, X)
Y = utils.resample(2.5, 2, Y)
X = compIQbalance(X)
Y = compIQbalance(Y)
SS = np.vstack([X,Y])


#E_m, wx_m, wy_m, err_m, err_rde_m = equalisation.FS_MCMA_MRDE(SS, Ncma, Nrde, ntaps, os, muCMA, muRDE, M)
#E_m, wx_m, wy_m, err_m, err_rde_m, = equalisation.joint_MCMA_MDDMA(SS, Ncma, Nrde, ntaps, os, muCMA, muRDE, M)
#E_s, wx_s, wy_s, err_s, err_rde_s = equalisation.FS_MCMA_SBD(SS, Ncma, Nrde, ntaps, os, muCMA, muRDE, M)
(wx, wy), err_cma = equalisation.MCMA_LMS(SS, 2, 2, muCMA, M, 33)
(wxdd, wydd), err_dd = equalisation.MDDMA_LMS(SS, 2, 2, muRDE, M, (wx,wy))
#E, wx, wy, err, err_rde = equalisation.FS_MCMA_MDDMA(SS, Ncma, Nrde, ntaps, os, muCMA, muRDE, M)
#E, wx, wy, err, err_rde = equalisation.FS_CMA_RDE(SS, Ncma, Nrde, ntaps, os, muCMA, muRDE, M)
E = equalisation.apply_filter(SS, wxdd, wydd, 33, 2 )
Ec = equalisation.apply_filter(SS, wx, wy, 33, 2)

evmX = cal_blind_evm(X[::2], M)
evmY = cal_blind_evm(Y[::2], M)
evmEx = cal_blind_evm(E[0], M)
evmEy = cal_blind_evm(E[1], M)
evmEx_c = cal_blind_evm(Ec[0], M)
evmEy_c = cal_blind_evm(Ec[1], M)
#evmEx_m = cal_blind_evm(E_m[0], M)
#evmEy_m = cal_blind_evm(E_m[1], M)
#evmEx_s = cal_blind_evm(E_s[0], M)
#evmEy_s = cal_blind_evm(E_s[1], M)

#sys.exit()
plt.figure()
plt.subplot(221)
plt.title(r'Recovered MCMA $EVM_x=%.1f\%%$'%(evmEx_c*100))
plt.hexbin(Ec[0].real, Ec[0].imag)
plt.subplot(222)
plt.title(r'Recovered MCMA $EVM_y=%.1f\%%$'%(evmEy_c*100))
plt.hexbin(Ec[1].real, Ec[1].imag)
plt.subplot(223)
plt.title(r'Recovered SBD $EVM_x=%.1f\%%$'%(evmEx*100))
plt.hexbin(E[0].real, E[0].imag)
plt.subplot(224)
plt.title(r'Recovered SBD $EVM_y=%.1f\%%$'%(evmEy*100))
plt.hexbin(E[1].real, E[1].imag)
plt.show()
sys.exit()
#plt.plot(E[0].real, E[0].imag, 'r.' ,label=r"$EVM_x=%.1f\%%$"%(evmEx*100))
#plt.plot(E[1].real, E[1].imag, 'g.', label=r"$EVM_y=%.1f\%%$"%(100*evmEy))

plt.subplot(212)
plt.plot(Ec[0].real, Ec[0].imag, 'r.' ,label=r"$EVMCMA_x=%.1f\%%$"%(evmEx_c*100))
plt.plot(Ec[1].real, Ec[1].imag, 'g.', label=r"$EVMCMA_y=%.1f\%%$"%(100*evmEy_c))
plt.legend()
plt.show()
sys.exit()
plt.title('Recovered MCMA/MRDE')
plt.plot(E_m[1].real, E_m[1].imag, 'g.', label=r"$EVM_y=%.1f\%%$"%(100*evmEy_m))
plt.plot(E_m[0].real, E_m[0].imag, 'r.' ,label=r"$EVM_x=%.1f\%%$"%(evmEx_m*100))
plt.legend()
plt.subplot(223)
plt.title('Recovered MCMA/MDDMA')
plt.plot(E_s[1].real, E_s[1].imag, 'g.', label=r"$EVM_y=%.1f\%%$"%(100*evmEy_s))
plt.plot(E_s[0].real, E_s[0].imag, 'r.' ,label=r"$EVM_x=%.1f\%%$"%(evmEx_s*100))
plt.legend()
plt.subplot(224)
plt.title('Original')
plt.plot(X[::2].real, X[::2].imag, 'r.', label=r"$EVM_x=%.1f\%%$"%(100*evmX))
plt.plot(Y[::2].real, Y[::2].imag, 'g.', label=r"$EVM_y=%.1f\%%$"%(100*evmY))
plt.legend()

plt.figure()
plt.subplot(331)
plt.title('CMA/MDDMA Taps')
plt.plot(wx[0,:], 'r')
plt.plot(wx[1,:], '--r')
plt.plot(wy[0,:], 'g')
plt.plot(wy[1,:], '--g')
plt.subplot(332)
plt.title('CMA/MDDMA error cma')
plt.plot(abs(err[0]), color='r')
plt.plot(abs(err[1])- 10, color='g')
plt.subplot(333)
plt.title('CMA/MDDMA error MDDMA')
plt.plot(abs(err_rde[0]), color='r')
plt.plot(abs(err_rde[1])-10, color='g')
plt.subplot(334)
plt.title('MCMA/MRDE Taps')
plt.plot(wx_m[0,:], 'r')
plt.plot(wx_m[1,:], '--r')
plt.plot(wy_m[0,:], 'g')
plt.plot(wy_m[1,:], '--g')
plt.subplot(335)
plt.title('MCMA/MRDE error cma')
plt.plot(abs(err_m[0]), color='r')
plt.plot(abs(err_m[1])- 10, color='g')
plt.subplot(336)
plt.title('MCMA/MRDE error rde')
plt.plot(abs(err_rde_m[0]), color='r')
plt.plot(abs(err_rde_m[1])-10, color='g')
plt.subplot(337)
plt.title('MCMA/SBD Taps')
plt.plot(wx_s[0,:], 'r')
plt.plot(wx_s[1,:], '--r')
plt.plot(wy_s[0,:], 'g')
plt.plot(wy_s[1,:], '--g')
plt.subplot(338)
plt.title('MCMA/SBD error cma')
plt.plot(abs(err_s[0]), color='r')
plt.plot(abs(err_s[1])- 10, color='g')
plt.subplot(339)
plt.title('MCMA/SBD error rde')
plt.plot(abs(err_rde_s[0]), color='r')
plt.plot(abs(err_rde_s[1])-10, color='g')
plt.show()


