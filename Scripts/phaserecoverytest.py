import numpy as np
from qampy import impairments, phaserec
from qampy import signals, helpers



fb = 40.e9
os = 1
fs = os*fb
N = 3*10**5
M = 64
snr = 30
lw_LO = np.linspace(10e1, 1000e1, 4)
#lw_LO = [100e3]
sers = []

for lw in lw_LO:
    shiftN = np.random.randint(-N/2, N/2, 1)
    s = signals.SignalQAMGrayCoded(M, N, fb=fb)
    s = s.resample(fs, beta=0.1, renormaise=True)
    s = impairments.change_snr(s, snr)
    s = np.roll(s, shiftN, axis=1)
    pp = impairments.apply_phase_noise(s, lw)
    recoverd, ph1 = phaserec.bps_twostage(pp, 28,  14, method='pyx')
    recoverd_2, ph2 = phaserec.bps(pp, 64,  14, method='pyx')
    recoverd = helpers.dump_edges(recoverd, 20)
    recoverd_2 = helpers.dump_edges(recoverd_2, 20)
    ser = recoverd.cal_ser()
    ser2 = recoverd_2.cal_ser()
    print("1 stage pyx ser=%g"%ser)
    print("2 stage pyx ser=%g"%ser2)


