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
        pilotX[k] = __findmax_16QAM(X[k], SYMBOLS_16QAM,\
                    np.sum(np.conj(pilotX[k-cfactor:k])*X[k-cfactor:k])/\
                    np.sum(np.abs(pilotX[k-cfactor:k])**2))
        pilotY[k] = __findmax_16QAM(Y[k], SYMBOLS_16QAM,
                    np.sum(np.conj(pilotY[k-cfactor:k])*Y[k-cfactor:k])/\
                    np.sum(np.abs(pilotY[k-cfactor:k])**2))
    return X*np.exp(-1.j*pcoeX), Y*np.exp(-1.j*pcoeY)

def calS0(E, gamma):
    N = len(E)
    r2 = np.sum(abs(E)**2)/N
    r4 = np.sum(abs(E)**4)/N
    S1 = 1-2*r2**2/r4-np.sqrt((2-gamma)*(2*r2**4/r4**2-r2**2/r4))
    S2 = gamma*r2**2/r4-1
    return r2/(1+S2/S1)


def partition_16QAM(E):
    S0 = calS0(E, 1.32)
    inner = (np.sqrt(S0/5)+np.sqrt(S0))/2.
    outer = (np.sqrt(9*S0/5)+np.sqrt(S0))/2.
    Ea = abs(E)
    class1_mask = (Ea< inner)|(Ea > outer)
    class2_mask = ~class1_mask
    return class1_mask, class2_mask


def ff_Phase_recovery_16QAM(E, Nangles, Nsymbols):
    phi = np.linspace(0, np.pi, Nangles)
    N = len(E)
    d = (abs(E[:, np.newaxis, np.newaxis]*np.exp(1.j*phi)[:, np.newaxis] -
             SYMBOLS_16QAM)**2).min(axis=2)
    phinew = np.zeros(N-Nsymbols, dtype=np.float)
    for k in range(Nsymbols, N-Nsymbols):
        phinew[k] = phi[np.sum(d[k-Nsymbols:k+Nsymbols],axis=0).argmin()]
    return E[Nsymbols:]*np.exp(1.j*phinew)


def QPSK_partition_phase_16QAM(Nblock, E):
    dphi = np.pi/4+np.arctan(1/3)
    L = len(E)
    # partition QPSK signal into qpsk constellation and non-qpsk const
    c1_m, c2_m = partition_16QAM(E)
    Sx = np.zeros(len(E), dtype=np.complex128)
    Sx[c2_m] = (E[c2_m]*np.exp(1.j*dphi))**4
    So = np.zeros(len(E), dtype=np.complex128)
    So[c2_m] = (E[c2_m]*np.exp(1.j*-dphi))**4
    S1 = np.zeros(len(E), dtype=np.complex128)
    S1[c1_m] = (E[c1_m])**4
    E_n = np.zeros(len(E), dtype=np.complex128)
    phi_est = np.zeros(len(E), dtype=np.float64)
    for i in range(0, L, Nblock):
        S1_sum = np.sum(S1[i:i+Nblock])
        Sx_tmp = np.min([S1_sum-Sx[i:i+Nblock],S1_sum-So[i:i+Nblock]],
                axis=0)[c2_m[i:i+Nblock]]
        phi_est[i:i+Nblock] = np.angle(S1_sum+Sx_tmp.sum())
    phi_est = np.unwrap(phi_est)/4-np.pi/4
    return (E*np.exp(-1.j*phi_est))[:(L//Nblock)*Nblock]

