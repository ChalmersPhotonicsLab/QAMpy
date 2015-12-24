from __future__ import division, print_function
import numpy as np
from . segmentaxis import segment_axis

SYMBOLS_16QAM =  np.array([1+1.j, 1-1.j, -1+1.j, -1-1.j, 1+3.j, 1-3.j, -1-3.j,
        3+1.j, 3-1.j, -3+1.j, -3,-1.j, 3+3.j, 3-3.j, -3.+3.j, -3-3.j])


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

def __findmax_16QAM(rk, ci, vk):
    mkk = np.real(rk*np.conj(ci)*np.conj(vk)-abs(ci)**2/2)
    pk = np.argmax(mkk)
    return ci[pk]

def ML_phase_16QAM(X, Y, pix, piy, cfactor):
    """ML_phase_16QAM phase recovery using pilots for starting the estimator
    on a dual-pol 16 QAM signal.

    Parameters:
        X:      input signal X polarisation
        Y:      input signal Y polarisation
        pix :   the known data signal pilot (X polarisation)
        piy:    the known data signal pilot (Y polarisation)
        cfactor:cfactor length of pilots

    Return:
        RecoveredX, RecoveredY recovered X and Y polarisation signals
    """
    N = len(X)
    pilotX = np.zeros(N, dtype=np.complex)
    pilotY = np.zeros(N, dtype=np.complex)
    pilotX[:cfactor] = pix
    pilotY[:cfactor] = piy
    pcoeX = np.zeros(N, dtype=np.complex)
    pcoeY = np.zeros(N, dtype=np.complex)
    pcoeX[:cfactor] = np.angle(np.conj(pilotX[:cfactor])*X[:cfactor])
    pcoeY[:cfactor] = np.angle(np.conj(pilotY[:cfactor])*Y[:cfactor])
    for k in range(cfactor, N):
        pcoeX[k] = np.angle(np.sum(np.conj(pilotX[k-cfactor:k])*
            X[k-cfactor:k]))
        pcoeY[k] = np.angle(np.sum(np.conj(pilotY[k-cfactor:k])*
            Y[k-cfactor:k]))
        pilotX[k] = findmax_16QAM(X[k], SYMBOLS_16QAM,\
                    np.sum(np.conj(pilotX[k-cfactor:k])*X[k-cfactor:k])/\
                    np.sum(np.abs(pilotX[k-cfactor:k])**2))
        pilotY[k] = __findmax_16QAM(Y[k], SYMBOLS_16QAM,
                    np.sum(np.conj(pilotY[k-cfactor:k])*Y[k-cfactor:k])/\
                    np.sum(np.abs(pilotY[k-cfactor:k])**2))
    return X*np.exp(-1.j*pcoeX), Y*np.exp(-1.j*pcoeY)

