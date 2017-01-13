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

fb = 40.e9
os = 2
fs = os*fb
N = 2**18
theta = np.pi/2.35
M = 16
QAM = modulation.QAMModulator(16)
snr = 24
muCMA = 3e-4
muRDE = 3e-4

X, Xsymbols, Xbits = QAM.generateSignal(N, snr,  baudrate=fb, samplingrate=fs, PRBSorder=15)
Y, Ysymbols, Ybits = QAM.generateSignal(N, snr, baudrate=fb, samplingrate=fs, PRBSorder=23)

omega = 2*np.pi*np.linspace(-fs/2, fs/2, N*os, endpoint=False)
t_pmd = 50e-12

H = H_PMD(theta, t_pmd, omega)

S = np.vstack([X,Y])
Sf = np.fft.fftshift(np.fft.fft(np.fft.fftshift(S, axes=1),axis=1), axes=1)
SSf = np.einsum('ijk,ik -> ik',H , Sf)
SS = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(SSf, axes=1),axis=1), axes=1)

#E, wx, wy, err, err_rde = equalisation.FS_MCMA_MRDE_general(SS, len(SS[0])//os//2 - 31, len(SS[0])//os//2 - 31, 30, 2, 0.001, 0.0003, 16)
E_m, wx_m, wy_m, err_m, err_rde_m = equalisation.FS_MCMA_MRDE(SS, 30000, 30000, 30, os, muCMA, muRDE, M)
E, wx, wy, err, err_rde = equalisation.FS_CMA_RDE(SS, 30000, 30000, 30, os, muCMA, muRDE, M)


evmX = cal_blind_evm(X[::2], M)
evmY = cal_blind_evm(Y[::2], M)
evmEx = cal_blind_evm(E[0], M)
evmEy = cal_blind_evm(E[1], M)
evmEx_m = cal_blind_evm(E_m[0], M)
evmEy_m = cal_blind_evm(E_m[1], M)
#sys.exit()
