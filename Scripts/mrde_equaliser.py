import numpy as np
import matplotlib.pylab as plt
from dsp import equalisation, modulation
from dsp.core import impairments

fb = 40.e9
os = 2
fs = os*fb
N = 2**18
theta = np.pi/2.35
M = 16
QAM = modulation.QAMModulator(16)
snr = 24
muCMA = 1e-3
muRDE = 0.5e-3
ntaps = 30
t_pmd = 50e-12
Ncma = N//6//os -int(1.5*ntaps)
Nrde = 5*N//6//os -int(1.5*ntaps)

S, symbols, bits = QAM.generate_signal(N, snr, baudrate=fb, samplingrate=fs, PRBSorder=(15,23))
SS = impairments.apply_PMD_to_field(S, theta, t_pmd, fs)

E_m, (wx_m, wy_m), (err_m, err_rde_m) = equalisation.dual_mode_equalisation(SS, os, (muCMA, muRDE), M, ntaps, TrSyms=(Ncma, Nrde), methods=("mcma","mrde" ))


evmX = QAM.cal_evm(S[0, ::2])
evmY = QAM.cal_evm(S[1, ::2])
evmEx_m = QAM.cal_evm(E_m[0])
evmEy_m = QAM.cal_evm(E_m[1])
#sys.exit()
plt.figure()
plt.subplot(121)
plt.title('Recovered MCMA/MRDE')
plt.plot(E_m[0].real, E_m[0].imag, 'r.' ,label=r"$EVM_x=%.1f\%%$"%(evmEx_m*100))
plt.plot(E_m[1].real, E_m[1].imag, 'g.', label=r"$EVM_y=%.1f\%%$"%(100*evmEy_m))
plt.legend()
plt.subplot(122)
plt.title('Original')
plt.plot(S[0,::2].real, S[0,::2].imag, 'ro', label=r"$EVM_x=%.1f\%%$"%(100*evmX))
plt.plot(S[1,::2].real, S[1,::2].imag, 'go', label=r"$EVM_y=%.1f\%%$"%(100*evmY))
plt.legend()

plt.figure()
plt.subplot(311)
plt.title('MCMA/MRDE Taps')
plt.plot(wx_m[0,:], 'r')
plt.plot(wx_m[1,:], '--r')
plt.plot(wy_m[0,:], 'g')
plt.plot(wy_m[1,:], '--g')
plt.subplot(312)
plt.title('MCMA/MRDE error cma')
plt.plot(abs(err_m[0]), color='r')
plt.plot(abs(err_m[1])- 10, color='g')
plt.subplot(313)
plt.title('MCMA/MRDE error rde')
plt.plot(abs(err_rde_m[0]), color='r')
plt.plot(abs(err_rde_m[1])-10, color='g')
plt.show()


