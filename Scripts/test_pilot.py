from qampy import signals, equalisation, impairments, core, phaserec
import numpy as np
import matplotlib.pylab as plt
ntaps = 17
snr = 30
dgd = 100e-12
theta = 3.7
sig = signals.SignalWithPilots(64, 2**16, 1024, 32, nframes=3, nmodes=2, fb=24e9)
#sig2 = impairments.change_snr(sig, snr)
#sig3 = sig2.resample(2*sig2.fb, beta=0.01, renormalise=True)
sig2 = sig.resample(2*sig.fb, beta=0.01, renormalise=True)
sig2 = impairments.apply_phase_noise(sig2, 100e3)
sig3 = impairments.change_snr(sig2, snr)
sig3 = core.impairments.rotate_field(sig3, np.pi/0.1)
#sig4 = sig3[::-1, 20000:]
#sig4[0,:] = sig3[1,20000:]
#sig4[1,:] = sig3[0,20000:]
sig4 = sig3[:, 20000:]
#sig4 = impairments.apply_PMD(sig4, theta, dgd)
#sig4 = impairments.apply_PMD(sig4, theta, dgd)
#sig4[0,:] = sig3[1, 20000:]
#sig4[1,:] = sig3[0, 20000:]

sig4.sync2frame(Ntaps=ntaps)
wx, s1 = equalisation.pilot_equaliser(sig4, [1e-3, 1e-3], ntaps, True, adaptive_stepsize=True)
s2, ph = phaserec.pilot_cpe(s1, nframes=1)
gmi = s2.cal_gmi()
evm = s2.cal_evm()
ber = s2.cal_ber()
ser = s2.cal_ser()
ksnr = s2.est_snr()
print("gmi {}".format(gmi))
print("ber {}".format(ber))
print("evm {}".format(evm))
print("snr {}".format(snr))
print("ser {}".format(ser))
plt.figure()
plt.subplot(121)
plt.title("Without CPE")
plt.hexbin(s1[0].real, s1[0].imag)
plt.subplot(122)
plt.title("With CPE")
plt.hexbin(s2[0].real, s2[0].imag)
plt.show()
sn = signals.SignalWithPilots.from_data_array(sig.symbols[0].reshape(1,-1), )
