from qampy import signals, equalisation, impairments, core
import numpy as np
import matplotlib.pylab as plt
ntaps = 23
snr = 20
dgd = 80e-12
theta = 3.7
sig = signals.SignalWithPilots(64, 2**16, 1024, 32, nframes=3, nmodes=2, fb=24e9)
sig2 = sig.resample(2*sig.fb, beta=0.1, renormalise=True)
sig3 = impairments.change_snr(sig2, snr)
#sig3 = core.impairments.rotate_field(sig3, np.pi/)
sig3 = impairments.apply_PMD(sig3, theta, dgd)
sig4 = sig3[:, 20000:]
sig4.sync2frame(Ntaps=ntaps, Niter=10)
s1, s2 = equalisation.pilot_equalizer(sig4, [1e-3, 1e-3], ntaps, True)
gmi = s2.cal_gmi()
evm = s2.cal_evm()
ber = s2.cal_ser()
ser = s2.cal_ber()
snr = s2.est_snr()
print("gmi {}".format(gmi))
print("ber {}".format(ber))
print("evm {}".format(evm))
print("snr {}".format(snr))
print("ser {}".format(ser))
plt.hexbin(s2[0].real, s2[0].imag)
plt.show()
