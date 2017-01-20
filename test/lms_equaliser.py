import numpy as np
import matplotlib.pylab as plt
from dsp import signals, equalisation, modulation
from dsp.signal_quality import cal_blind_evm



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

fb = 40.e9
os = 2
fs = os*fb
N = 10**6
theta = np.pi/2.35
M = 64
QAM = modulation.QAMModulator(M)
snr = 25
muCMA = 1e-3
muRDE = 1.e-3
ntaps = 11
Ncma = 3*N//4//os -int(1.5*ntaps)
Nrde = N//4//os -int(1.5*ntaps)

X, Xsymbols, Xbits = QAM.generateSignal(N, snr,  baudrate=fb, samplingrate=fs, PRBSorder=15)
Y, Ysymbols, Ybits = QAM.generateSignal(N, snr, baudrate=fb, samplingrate=fs, PRBSorder=23)

omega = 2*np.pi*np.linspace(-fs/2, fs/2, N*os, endpoint=False)
t_pmd = 50.e-12

H = H_PMD(theta, t_pmd, omega)
H2 = H_PMD(-theta, -t_pmd, omega)

S = np.vstack([X,Y])

SS = applyPMD(S, H)
#SS = rotate_field(np.pi*2.4/1.3, SS)

#E, wx, wy, err, err_rde = equalisation.FS_MCMA_MRDE_general(SS, len(SS[0])//os//2 - 31, len(SS[0])//os//2 - 31, 30, 2, 0.001, 0.0003, 16)
#E_m, wx_m, wy_m, err_m, err_rde_m = equalisation.FS_MCMA_MRDE(SS, Ncma, Nrde, ntaps, os, muCMA, muRDE, M)
#E_s, wx_s, wy_s, err_s, err_rde_s = equalisation.FS_MCMA_SBD(SS, Ncma, Nrde, ntaps, os, muCMA, muRDE, M)
#E, wx, wy, err, err_rde = equalisation.FS_MCMA_MDDMA(SS, Ncma, Nrde, ntaps, os, muCMA, muRDE, M)
#E, wx, wy, err, err_rde = equalisation.FS_CMA_RDE(SS, Ncma, Nrde, ntaps, os, muCMA, muRDE, M)


(wx,wy), err_cma = equalisation.MCMA_LMS(SS, 5, os, muCMA, M, ntaps)
(wx,wy), err_sbd = equalisation.SBD_LMS(SS, 5, os, muRDE, M, (wx,wy))

Ex, Ey = equalisation.apply_filter(SS, wx, wy, ntaps, os)
print("equalised")

E = np.vstack([Ex,Ey])

evmX = cal_blind_evm(X[::2], M)
evmY = cal_blind_evm(Y[::2], M)
evmEx = cal_blind_evm(E[0], M)
evmEy = cal_blind_evm(E[1], M)

#sys.exit()
plt.figure()
plt.subplot(211)
plt.title('Recovered MCMA/SBD')
plt.plot(E[0,::100].real, E[0,::100].imag, 'r.' ,label=r"$EVM_x=%.1f\%%$"%(evmEx*100))
plt.plot(E[1,::100].real, E[1,::100].imag, 'g.', label=r"$EVM_y=%.1f\%%$"%(100*evmEy))
plt.legend()
plt.subplot(212)
plt.title('Original')
plt.plot(X[::200].real, X[::200].imag, 'r.', label=r"$EVM_x=%.1f\%%$"%(100*evmX))
plt.plot(Y[::200].real, Y[::200].imag, 'g.', label=r"$EVM_y=%.1f\%%$"%(100*evmY))
plt.legend()
plt.show()
#plt.figure()
#plt.subplot(331)
#plt.title('CMA/MDDMA Taps')
#plt.plot(wx[0,:], 'r')
#plt.plot(wx[1,:], '--r')
#plt.plot(wy[0,:], 'g')
#plt.plot(wy[1,:], '--g')
#plt.subplot(332)
#plt.title('CMA/MDDMA error cma')
#plt.plot(abs(err[0]), color='r')
#plt.plot(abs(err[1])- 10, color='g')
