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

X, Xsymbols, Xbits = QAM.generateSignal(N, snr,  baudrate=fb, samplingrate=fs, PRBSorder=15)
Y, Ysymbols, Ybits = QAM.generateSignal(N, snr, baudrate=fb, samplingrate=fs, PRBSorder=23)

omega = 2*np.pi*np.linspace(-fs/2, fs/2, N*os, endpoint=False)
t_pmd = 50e-12

H = H_PMD(theta, t_pmd, omega)

S = np.vstack([X,Y])
Sf = np.fft.fftshift(np.fft.fft(np.fft.fftshift(S, axes=1),axis=1), axes=1)
SSf = np.einsum('ijk,ik -> ik',H , Sf)
SS = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(SSf, axes=1),axis=1), axes=1)

E, wx, wy, err, err_rde = equalisation.FS_CMA_RDE_16QAM(SS, 10000, 40000, 30, 2, 0.0001, 0.0003)


evmX = cal_blind_evm(X[::2], M)
evmY = cal_blind_evm(Y[::2], M)
evmEx = cal_blind_evm(E[0], M)
evmEy = cal_blind_evm(E[1], M)
#sys.exit()
plt.figure()
plt.subplot(131)
plt.title('Recovered')
plt.plot(E[0].real, E[0].imag, 'r.' ,label=r"$EVM_x=%.1f\%%$"%(evmEx*100))
plt.legend()
plt.subplot(132)
plt.plot(E[1].real, E[1].imag, 'g.', label=r"$EVM_y=%.1f\%%$"%(100*evmEy))
plt.legend()
plt.subplot(133)
plt.title('Original')
plt.plot(X[::2].real, X[::2].imag, 'ro', label=r"$EVM_x=%.1f\%%$"%(100*evmX))
plt.plot(Y[::2].real, Y[::2].imag, 'go', label=r"$EVM_y=%.1f\%%$"%(100*evmY))
plt.legend()

plt.figure()
plt.subplot(311)
plt.title('Taps')
plt.plot(wx[0,:], 'r')
plt.plot(wx[1,:], '--r')
plt.plot(wy[0,:], 'g')
plt.plot(wy[1,:], '--g')
plt.subplot(312)
plt.title('error cma')
plt.plot(err[0], color='r')
plt.plot(err[1]- 10, color='g')
plt.subplot(313)
plt.title('error rde')
plt.plot(err_rde[0], color='r')
plt.plot(err_rde[1]-10, color='g')
plt.show()


