import numpy as np
from matplotlib import pylab as plt
from qampy import signals, impairments, equalisation, phaserec

mysig = signals.SignalWithPilots(64,2**16,2**10,32,nmodes=2,Mpilots=4,nframes=3,fb=24e9)
mysig2 = mysig.resample(mysig.fb*2,beta=0.01)
mysig3 = impairments.simulate_transmission(mysig2,snr=20,dgd=10e-12, freq_off=00e6,lwdth=000e3,roll_frame_sync=True, modal_delay=[2000, 3000])
mysig3.sync2frame()
print(mysig3.shiftfctrs)
mysig3.corr_foe()
wxy, eq_sig = equalisation.pilot_equaliser(mysig3, (1e-3, 1e-3), 45, foe_comp=False, methods=("cma", "sbd_data"))
cpe_sig, ph = phaserec.pilot_cpe(eq_sig,N=5,use_seq=False)
#cpe_sig = eq_sig
print(cpe_sig.cal_gmi())
plt.figure()
plt.subplot(121)
plt.hist2d(cpe_sig[0].real, cpe_sig[0].imag, bins=200)
#plt.plot(wxy[0][0].real, '-k')
#plt.plot(wxy[0][1].real, '--k')
#plt.plot(wxy[0][0].imag, '-c')
#plt.plot(wxy[0][1].imag, '--c')
plt.subplot(122)
plt.hist2d(cpe_sig[1].real, cpe_sig[1].imag, bins=200)
#plt.plot(wxy[1][0].real, '-k')
#plt.plot(wxy[1][1].real, '--k')
#plt.plot(wxy[1][0].imag, '-c')
#plt.plot(wxy[1][1].imag, '--c')
plt.show()
