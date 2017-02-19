import numpy as np
import matplotlib.pylab as plt
from dsp import signals, equalisation, modulation, utils, phaserecovery
from dsp.signal_quality import cal_blind_evm
from scipy.io import loadmat
from scipy.signal import lfilter, freqz, butter

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=5)
    y = lfilter(b, a, data)
    return y

def running_mean(data, N):
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[N:] - cumsum[:-N])/N

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
muCMA = 4e-3
muRDE = 0.3e-2
ntaps = 33
niter = 1

Dat = loadmat('OSNRLoading_1546p31_Att_1_OSNR_37_1.mat')
X = Dat['CH1'] + 1.j * Dat['CH2']
Y = Dat['CH3'] + 1.j * Dat['CH4']
X = X.flatten()
Y = Y.flatten()

X = pre_filter(X, 3.75)
Y = pre_filter(Y, 3.75)
X = utils.resample(2.5, 2, X)
Y = utils.resample(2.5, 2, Y)
#print(X.shape)
X = compIQbalance(X)
Y = compIQbalance(Y)
D = 16.91*1e-6
L = 80.e3
#X = equalisation.CDcomp(X, 40e9, 0, L, D, 1545e-9)
#Y = equalisation.CDcomp(Y, 40e9, 0, L, D, 1545e-9)
SS = np.vstack([X[1000:-1000],Y[1000:-1000]])



#E_m, wx_m, wy_m, err_m, err_rde_m = equalisation.FS_MCMA_MRDE(SS, Ncma, Nrde, ntaps, os, muCMA, muRDE, M)
#E_m, wx_m, wy_m, err_m, err_rde_m, = equalisation.joint_MCMA_MDDMA(SS, Ncma, Nrde, ntaps, os, muCMA, muRDE, M)
#E_s, wx_s, wy_s, err_s, err_rde_s = equalisation.FS_MCMA_SBD(SS, Ncma, Nrde, ntaps, os, muCMA, muRDE, M)
#(wx, wy), err_cma = equalisation.CMA_LMS(SS, 1, os, muCMA, M, ntaps)
#(wx, wy), err_cma, SS = equalisation.MCMA_LMS(SS, niter, os, muCMA, M, (wx,wy))
#Ec = equalisation.apply_filter(SS, wx, wy, ntaps, os)
#(wxdd, wydd), err_dd, SS = equalisation.SBD_LMS(SS, niter, os, muRDE, M, (wx,wy))
#E, (wx,wy), (err_cma, err_dd) = equalisation.dual_mode_equalisation(SS, os, (muCMA, muRDE), M, ntaps, Niter=(3,3), methods=("mcma_adaptive", "sbd_adaptive"))
#E, wx, wy, err, err_rde = equalisation.FS_MCMA_MDDMA(SS, Ncma, Nrde, ntaps, os, muCMA, muRDE, M)
#E, wx, wy, err, err_rde = equalisation.FS_CMA_RDE(SS, Ncma, Nrde, ntaps, os, muCMA, muRDE, M)
#Ex,Ey = equalisation.apply_filter(SS, wxdd, wydd, ntaps, os )
#Ec = equalisation.apply_filter(SS, wxc, wyc, ntaps, os)
#Ex = Ex*np.exp(-1.j*2*np.pi*0.003)
#E = [Ex,Ey]
E, wxy, err_both = equalisation.dual_mode_equalisation(SS, os, (muCMA, muRDE), M, ntaps, Niter=(1,1), methods=("mcma", "sbd"), adaptive_stepsize=(True,True) )
print("X pol phase")
Ex = phaserecovery.blindphasesearch_af(E[0], 64, QAM.symbols, 8)
print("X pol phase done")
print("Y pol phase")
Ey = phaserecovery.blindphasesearch_af(E[1], 64, QAM.symbols, 8)
print("Y pol phase done")
E = np.vstack([Ex,Ey])
evmX = cal_blind_evm(X[::2], M)
evmY = cal_blind_evm(Y[::2], M)
evmEx = cal_blind_evm(E[0], M)
evmEy = cal_blind_evm(E[1], M)
evmEx_c = QAM.cal_EVM(E[0])
evmEy_c = QAM.cal_EVM(E[1])
print("EVM x 1 = %.1f"%(100*evmEx_c))
print("EVM x 2 = %.1f"%(100*evmEx))
#evmEx_c = cal_blind_evm(Ec[0], M)
#evmEy_c = cal_blind_evm(Ec[1], M)
#evmEx_m = cal_blind_evm(E_m[0], M)
#evmEy_m = cal_blind_evm(E_m[1], M)
#evmEx_s = cal_blind_evm(E_s[0], M)
#evmEy_s = cal_blind_evm(E_s[1], M)
Ec = E

#sys.exit()
plt.figure()
plt.subplot(221)
plt.title(r'Recovered MCMA $EVM_x=%.1f\%%$'%(evmEx_c*100))
#plt.hexbin(Ec[0].real, Ec[0].imag)
plt.plot(Ec[0][::100].real, Ec[0][::100].imag, '.')
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
plt.figure()
plt.subplot(211)
plt.title("MCMA error")
plt.plot(10*np.log10(running_mean(abs(err_cma[0,:])**2, 500)), label="X-polarisation")
plt.plot(10*np.log10(running_mean(abs(err_cma[1,:])**2, 500)), label="Y-polarisation")
plt.legend()
plt.subplot(212)
plt.title("DD error")
plt.plot(10*np.log10(running_mean(abs(err_dd[0,:])**2,500)), label="X-polarisation")
plt.plot(10*np.log10(running_mean(abs(err_dd[1,:])**2, 500)), label="Y-polarisation")
plt.legend()
plt.show()


