from __future__ import division, print_function
import numpy as np
from . segmentaxis import segment_axis

def viterbiviterbi_gen(N, E, M):
    """Viterbi-Viterbi blind phase recovery for an M-PSK signal"""
    E = E.flatten()
    L = len(E)
    phi = np.angle(E)
    E_raised = np.exp(1.j*phi)**M
    sa = segment_axis(E_raised, 2*N, 2*N-1)
    phase_est = np.sum(sa[:L-2*N], axis=1)
    phase_est = np.unwrap(np.angle(phase_est))
    E = E[N:L-N]
    #if M == 4: # QPSK needs pi/4 shift
    # need a shift by pi/M for constellation points to not be on the axis
    phase_est = phase_est - np.pi
    return E*np.exp(-1.j*phase_est/M)

def viterbiviterbi_qpsk(N, E):
    """Viterbi-Viterbi blind phase recovery for QPSK signal"""
    return viterbiviterbi_gen(N, E, 4)

def viterbiviterbi_bpsk(N, E):
    """Viterbi-Viterbi for BPSK"""
    return viterbiviterbi_gen(N, E, 2)

