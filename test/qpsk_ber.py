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

fb = 40.e9
os = 2
fs = os*fb
N = 2**18
theta = np.pi/2.45
theta2 = np.pi/4
QAM = modulation.QAMModulator(4)
snr = 12

X, symbolsX, bitsX = QAM.generateSignal(N, snr, PRBSorder=15, baudrate=fb, samplingrate=fs)
Y, symbolsY, bitsY = QAM.generateSignal(N, snr, PRBSorder=23, baudrate=fb, samplingrate=fs)

omega = 2*np.pi*np.linspace(-fs/2, fs/2, os*N, endpoint=False)
t_pmd = 75e-12

H = H_PMD(theta, t_pmd, omega)

S = np.vstack([X,Y])
Sf = np.fft.fftshift(np.fft.fft(np.fft.fftshift(S, axes=1),axis=1), axes=1)
SSf = np.einsum('ijk,ik -> ik',H , Sf)
SS = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(SSf, axes=1),axis=1), axes=1)

E, wx, wy, err = equalisation.FS_CMA(SS, 10000, 40, os, 0.1)

E = E[:,1000:-1000]

try:
    berx = (QAM.cal_BER(E[0], bits_tx=bitsX, Lsync=50))
except:
    berx = (QAM.cal_BER(E[1], bits_tx=bitsX, Lsync=50))
try:
    bery = (QAM.cal_BER(E[1], bits_tx=bitsY, Lsync=50))
except:
    bery = (QAM.cal_BER(E[0], bits_tx=bitsY, Lsync=50))

print("X BER %f dB"%(10*np.log10(berx[0])))
print("Y BER %f dB"%(10*np.log10(bery[0])))
evmX = cal_blind_evm(X[::os], 4)
evmY = cal_blind_evm(Y[::os], 4)
evmEx = cal_blind_evm(E[0], 4)
evmEy = cal_blind_evm(E[1], 4)

#sys.exit()
plt.figure()
plt.subplot(121)
plt.title('Recovered')
plt.plot(E[0].real, E[0].imag, 'ro', label=r"$EVM_x=%.1f\%%$"%(100*evmEx))
plt.plot(E[1].real, E[1].imag, 'go' ,label=r"$EVM_y=%.1f\%%$"%(evmEy*100))
plt.legend()
plt.subplot(122)
plt.title('Original')
plt.plot(X[::2].real, X[::2].imag, 'ro', label=r"$EVM_x=%.1f\%%$"%(100*evmX))
plt.plot(Y[::2].real, Y[::2].imag, 'go', label=r"$EVM_y=%.1f\%%$"%(100*evmY))
plt.legend()

plt.figure()
plt.subplot(211)
plt.title('Taps')
plt.plot(wx[0,:], 'r')
plt.plot(wx[1,:], '--r')
plt.plot(wy[0,:], 'g')
plt.plot(wy[1,:], '--g')
plt.subplot(212)
plt.title('error')
plt.plot(err[0], color='r')
plt.plot(err[1], color='g')
plt.show()


