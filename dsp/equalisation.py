from __future__ import division
import numpy as np
import scipy.signal as scisig
import numexpr as ne

from . segmentaxis import segment_axis
from . import mathfcts


class DataSyncError(Exception):
    pass


def FS_CMA_training_python(TrSyms, Ntaps, os, mu, E, wx):
    err = np.zeros(TrSyms, dtype=np.float)
    for i in xrange(0, TrSyms):
        X = E[:, i*os:i*os+Ntaps]
        Xest = np.sum(wx*X)
        err[i] = abs(Xest)-1
        wx -= mu*err[i]*Xest*np.conj(X)
    return err, wx

def FS_RDE_training_python(TrCMA, TrRDE, Ntaps, os, muRDE, E, wx, part, code):
    err = np.zeros(TrRDE, dtype=np.float)
    for i in xrange(TrCMA, TrCMA+TrRDE):
        X = E[:, i*os:i*os+Ntaps]
        Xest = np.sum(wx*X)
        Ssq = abs(Xest)**2
        S_DD = quantize1(Ssq, part, code)
        err[i-TrCMA] = S_DD-Ssq
        wx += muRDE*err[i-TrCMA]*Xest*np.conj(X)
    return err, wx

try:
    from dsp_cython import FS_RDE_training
except:
    #use python code if cython code is not available
    FS_RDE_training = FS_RDE_training_python

try:
    from dsp_cython import FS_CMA_training
except:
    #use python code if cython code is not available
    FS_CMA_training = FS_CMA_training_python

try:
    from dsp_cython import lfsr_ext
except:
    from mathfcts import lfsr_ext

try:
    from dsp_cython import lfsr_int
except:
    from mathfcts import lfsr_int

SYMBOLS_16QAM =  np.array([1+1.j, 1-1.j, -1+1.j, -1-1.j, 1+3.j, 1-3.j, -1-3.j,
        3+1.j, 3-1.j, -3+1.j, -3,-1.j, 3+3.j, 3-3.j, -3.+3.j, -3-3.j])

def calS0(E, gamma):
    N = len(E)
    r2 = np.sum(abs(E)**2)/N
    r4 = np.sum(abs(E)**4)/N
    S1 = 1-2*r2**2/r4-np.sqrt((2-gamma)*(2*r2**4/r4**2-r2**2/r4))
    S2 = gamma*r2**2/r4-1
    return r2/(1+S2/S1)

def findmax_16QAM(rk, ci, vk):
    mkk = np.real(rk*np.conj(ci)*np.conj(vk)-abs(ci)**2/2)
    pk = np.argmax(mkk)
    return ci[pk]

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
        pilotY[k] = findmax_16QAM(Y[k], SYMBOLS_16QAM,
                    np.sum(np.conj(pilotY[k-cfactor:k])*Y[k-cfactor:k])/\
                    np.sum(np.abs(pilotY[k-cfactor:k])**2))
    return X*np.exp(-1.j*pcoeX), Y*np.exp(-1.j*pcoeY)

def viterbiviterbi(N, E):
    return viterbiviterbi_gen(N, E, 4)

def viterbiviterbi_BPSK(N, E):
    return viterbiviterbi_gen(N, E, 2)

def viterbiviterbi_gen(N, E, M):
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


def viterbiviterbi_ne(N, E):
    E = E.flatten()
    L = len(E)
    E_raised = ne.evaluate('exp(1.j*phi)**4')
    sa = segment_axis(E_raised, 2*N, 2*N-1)
    phase_est = np.sum(sa[:L-2*N], axis=1)
    phase_est = np.unwrap(np.angle(phase_est))
    E = E[N:L-N]
    phase_est = phase_est - np.pi    # shifts by pi/4 to make it 4 QAM
    return ne.evaluate('E*exp(-1.j*(phase_est/4.))')

def resample(Fold, Fnew, E, window=None):
    ''' resamples the signal from Fold to Fnew'''
    E = E.flatten()
    L = len(E)
    num = Fnew/Fold*L
    if window is None:
        E = scisig.resample(E, num)
    else:
        E = scisig.resample(E, num, window=window)
    return E

def FS_CMA(TrSyms, Ntaps, os, mu, Ex, Ey):
    '''performs PMD equalization using CMA algorithm
    taps for X initialised to [0001000]
    taps for Y initialised to orthogonal polarization of X pol taps'''
    Ex = Ex.flatten()
    Ey = Ey.flatten()
    # if can't have more training samples than field
    assert TrSyms*os < len(Ex)-Ntaps, "More training samples than"\
                                    " overall samples"
    L = len(Ex)
    mu = mu/Ntaps
    E = np.vstack([Ex, Ey])
    # scale signal
    P = np.mean(mathfcts.cabssquared(E))
    E = E/np.sqrt(P)
    err = np.zeros((2, TrSyms), dtype='float')
    # ** training for X polarisation **
    wx = np.zeros((2, Ntaps), dtype=np.complex128)
    wx[1, Ntaps//2] = 1
    # run CMA
    err[0,:], wx = FS_CMA_training(TrSyms, Ntaps, os, mu, E, wx)
    # ** training for y polarisation **
    wy = np.zeros((2, Ntaps), dtype=np.complex128)
    # initialising the taps to be ortthogonal to the x polarisation
    wy[1, :] = wx[0, ::-1]
    wy[0, :] = -wx[1, ::-1]
    wy = np.conj(wy)
    # centering the taps
    wR = np.abs(wx)
    wXmax = np.where(np.max(wR) == wR)
    wR = np.abs(wy)
    wYmax = np.where(np.max(wR) == wR)
    delay = abs(wYmax[1]-wXmax[1])
    pad = np.zeros((2, delay), dtype=np.complex128)
    if delay > 0:
        wy = wy[:, delay:]
        wy = np.hstack([wy, pad])
    elif delay < 0:
        wy = wy[:, 0:Ntaps-delay-1]
        wy = np.hstack([pad, wy])
    # run CMA
    err[1,:], wy = FS_CMA_training(TrSyms, Ntaps, os, mu, E, wy)
    # equalise data points. Reuse samples used for channel estimation
    syms = L//2-Ntaps//os-1
    X = segment_axis(E, Ntaps, Ntaps-os, axis=1)
    EestX = np.sum(wx[:, np.newaxis, :]*X, axis=(0, 2))[:syms]
    EestY = np.sum(wy[:, np.newaxis, :]*X, axis=(0, 2))[:syms]
    dump = 1000
    EestX = EestX[dump:-dump]
    EestY = EestY[dump:-dump]
    return EestX, EestY, wx, wy, err

# quantization function
def quantize(signal, partitions, codebook):
    quanta = []
    for datum in signal:
        index = 0
        while index < len(partitions) and datum > partitions[index]:
            index += 1
        quanta.append(codebook[index])
    return quanta

# quantization for single value
def quantize1(signal, partitions, codebook):
    index = 0
    while index < len(partitions) and signal > partitions[index]:
        index += 1
    quanta = codebook[index]
    return quanta

def FS_CMA_RDE_16QAM(TrCMA, TrRDE, Ntaps, os, muCMA, muRDE, Ex, Ey):
    '''performs PMD equalization using CMA algorithm
    taps for X initialised to [0001000]
    taps for Y initialised to orthogonal polarization of X pol taps'''
    Ex = Ex.flatten()
    Ey = Ey.flatten()
    L = len(Ex)
    muCMA = muCMA/Ntaps
    muRDE = muRDE/Ntaps
    # if can't have more training samples than field
    assert (TrCMA+TrRDE)*os < L-Ntaps, "More training samples than overall samples"
    # constellation properties
    R = 13.2
    code = np.array([2., 10., 18.])
    part = np.array([5.24, 13.71])
    E = np.vstack([Ex, Ey])
    # scale signal
    P = np.mean(mathfcts.cabssquared(E))
    E = E/np.sqrt(P)
    err_cma = np.zeros((2, TrCMA), dtype='float')
    err_rde = np.zeros((2, TrRDE), dtype='float')
    # ** training for X polarisation **
    wx = np.zeros((2, Ntaps), dtype=np.complex128)
    wx[1, Ntaps//2] = 1
    # find taps with CMA
    err_cma[0,:], wx = FS_CMA_training(TrCMA, Ntaps, os, muCMA, E, wx)
    # scale taps for RDE
    wx = np.sqrt(R)*wx
    # use refine taps with RDE
    err_rde[0,:], wx = FS_RDE_training(TrCMA, TrRDE, Ntaps, os, muRDE, E, wx,
            part, code)
    # ** training for y polarisation **
    wy = np.zeros((2, Ntaps), dtype=np.complex128)
    # initialising the taps to be orthogonal to the x polarisation
    wy[1, :] = wx[0, ::-1]
    wy[0, :] = -wx[1, ::-1]
    wy = np.conj(wy)/np.sqrt(R)
    # centering the taps
    wR = np.abs(wx)
    wXmax = np.where(np.max(wR) == wR)
    wR = np.abs(wy)
    wYmax = np.where(np.max(wR) == wR)
    delay = abs(wYmax[1]-wXmax[1])
    pad = np.zeros((2, delay), dtype=np.complex128)
    if delay > 0:
        wy = wy[:, delay:]
        wy = np.hstack([wy, pad])
    elif delay < 0:
        wy = wy[:, 0:Ntaps-delay-1]
        wy = np.hstack([pad, wy])
    # find taps with CMA
    err_cma[1,:], wy = FS_CMA_training(TrCMA, Ntaps, os, muCMA, E, wy)
    # scale taps for RDE
    wy = np.sqrt(R)*wy
    # use refine taps with RDE
    err_rde[1, :], wy = FS_RDE_training(TrCMA, TrRDE, Ntaps, os, muRDE, E, wy,
            part, code)
    # equalise data points. Reuse samples used for channel estimation
    syms = L//2-Ntaps//os-1
    X = segment_axis(E, Ntaps, Ntaps-os, axis=1)
    EestX = np.sum(wx[:, np.newaxis, :]*X, axis=(0, 2))[:syms]
    EestY = np.sum(wy[:, np.newaxis, :]*X, axis=(0, 2))[:syms]
    dump = 1000
    EestX = EestX[dump:-dump]
    EestY = EestY[dump:-dump]
    return EestX, EestY, wx, wy, err_cma, err_rde

def CDcomp(fs, N, L, D, sig, wl):
    '''Compensate for CD for a single pol signal using overlap add
    setting N=0 assumes cyclic boundary conditions and uses a single FFT/IFFT
    All units are SI'''
    sig = sig.flatten()
    samp = len(sig)
    #wl *= 1e-9
    #L = L*1.e3
    c = 2.99792458e8
    #D = D*1.e-6
    if N == 0:
        N = samp
#    H = np.zeros(N,dtype='complex')
    H = np.arange(0, N)+1j*np.zeros(N, dtype='float')
    H -= N//2
    H *= H
    H *= np.pi*D*wl**2*L*fs**2/(c*N**2)
    H = np.exp(-1j*H)
    #H1 = H
    H = np.fft.fftshift(H)
    if N == samp:
        sigEQ = np.fft.fft(sig)
        sigEQ *= H
        sigEQ = np.fft.ifft(sigEQ)
    else:
        n = N//2
        zp = N//4
        B = samp//n
        sigB = np.zeros(N, dtype=np.complex128)
        sigEQ = np.zeros(n*(B+1), dtype=np.complex128)
        sB = np.zeros((B, N), dtype=np.complex128)
        for i in xrange(0, B):
            sigB = np.zeros(N, dtype=np.complex128)
            sigB[zp:-zp] = sig[i*n:i*n+n]
            sigB = np.fft.fft(sigB)
            sigB *= H
            sigB = np.fft.ifft(sigB)
            sB[i, :] = sigB
            sigEQ[i*n:i*n+n+2*zp] = sigEQ[i*n:i*n+n+2*zp]+sigB
        sigEQ = sigEQ[zp:-zp]
    return sigEQ

