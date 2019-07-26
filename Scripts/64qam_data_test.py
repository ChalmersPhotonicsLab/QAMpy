import numpy as np
import matplotlib.pylab as plt

from qampy import equalisation, analog_frontend, signals, phaserec, io, helpers


os = 2
M = 64
snr = 35
muCMA = 6e-4
muRDE = 6e-4
Ncma = 100000
Nrde = 300000
ntaps = 51
niter = 1
symbs = io.load_symbols_from_matlab_file("data/20GBaud_SRRC0P05_64QAM_PRBS15.mat", 64, (("X_Symbs",),),
                                         fb=20e9, normalise=True, fake_polmux=True)
sig = io.create_signal_from_matlab(symbs, "data/OSNRLoading_IdealRxLaser_1544p91_Att_1_OSNR_38_1.mat", 50e9,
                                   (("CH1","CH2"),("CH3","CH4" )))
sig = sig[:,:10**6]
sig = helpers.normalise_and_center(sig)
sig = sig.resample(2*symbs.fb, beta=0.05, renormalise=True)
#sig = analog_frontend.comp_IQ_inbalance(sig)

E, wxy, err_both = equalisation.dual_mode_equalisation(sig, (muCMA, muRDE), ntaps, Niter=(niter, niter), methods=("mcma", "sbd"),
                                                       adaptive_stepsize=(True, True))
#E, wxy, err = equalisation.equalise_signal(sig, muCMA, Ntaps=33, apply=True)

E = helpers.normalise_and_center(E)
gmi = E.cal_gmi()
print(gmi)
#sys.exit()
plt.figure()
plt.subplot(211)
plt.hexbin(E[0].real, E[0].imag)
plt.subplot(212)
plt.hexbin(E[1].real, E[1].imag)
plt.show()
