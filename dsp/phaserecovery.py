import numpy as np


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
    if M == 4: # QPSK needs pi/4 shift
        phase_est = phase_est - np.pi
    return E*np.exp(-1.j*phase_est/M)

def viterbiviterbi(N, E):
    return viterbiviterbi_gen(N, E, 4)

def viterbiviterbi_BPSK(N, E):
    return viterbiviterbi_gen(N, E, 2)

def viterbiviterbi_ne(N, E):
    E = E.flatten()
    L = len(E)
    phi = np.angle(E)
    E_raised = ne.evaluate('exp(1.j*phi)**4')
    sa = segment_axis(E_raised, 2*N, 2*N-1)
    phase_est = np.sum(sa[:L-2*N], axis=1)
    phase_est = np.unwrap(np.angle(phase_est))
    E = E[N:L-N]
    phase_est = phase_est - np.pi    # shifts by pi/4 to make it 4 QAM
    return ne.evaluate('E*exp(-1.j*(phase_est/4.))')

