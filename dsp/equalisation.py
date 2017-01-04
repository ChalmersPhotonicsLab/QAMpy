from __future__ import division
import pyximport; pyximport.install()
import numpy as np
import scipy.signal as scisig
import numexpr as ne

from . segmentaxis import segment_axis
from . import mathfcts


def FS_CMA_training_python(TrSyms, Ntaps, os, mu, E, wx):
    """
    Training of the CMA algorithm to determine the equaliser taps.

    Parameters
    ----------
    TrSyms : int
       number of symbols to use for training needs to be less than len(Ex)

    Ntaps   : int
       number of equaliser taps

    os      : int
       oversampling factor

    mu      : float
       step size parameter

    E      : array_like
       dual polarisation signal field

    wx     : array_like
       initial equaliser taps

    Returns
    -------

    err       : array_like
       CMA estimation error for x and y polarisation

    wx        : array_like
       equaliser taps
    """
    err = np.zeros(TrSyms, dtype=np.float)
    for i in range(0, TrSyms):
        X = E[:, i*os:i*os+Ntaps]
        Xest = np.sum(wx*X)
        err[i] = abs(Xest)-1
        wx -= mu*err[i]*Xest*np.conj(X)
    return err, wx

def FS_RDE_training_python(TrCMA, TrRDE, Ntaps, os, muRDE, E, wx, part, code):
    """
    Training of the RDE algorithm to determine the equaliser taps.

    Parameters
    ----------
    TrCMA : int
       number of symbols to use for training the initial CMA needs to be less than len(Ex)

    TrRDE : int
       number of symbols to use for training the radius directed equaliser, needs to be less than len(Ex)

    Ntaps   : int
       number of equaliser taps

    os      : int
       oversampling factor

    muRDE   : float
       step size parameter

    E       : array_like
       dual polarisation signal field

    wx     : array_like
       initial equaliser taps

    part    : array_like
       partitioning vector defining the boundaries between the different QAM constellation rings

    code    : array_like
       code vector defining the signal powers for the different QAM constellation rings

    Returns
    -------

    err       : array_like
       CMA estimation error for x and y polarisation

    wx    : array_like
       equaliser taps
    """
    err = np.zeros(TrRDE, dtype=np.float)
    for i in range(TrCMA, TrCMA+TrRDE):
        X = E[:, i*os:i*os+Ntaps]
        Xest = np.sum(wx*X)
        Ssq = abs(Xest)**2
        S_DD = partition_value(Ssq, part, code)
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




def FS_CMA(TrSyms, Ntaps, os, mu, Ex, Ey):
    """
    Equalisation of PMD and residual dispersion for an QPSK signal based on the Fractionally spaced (FS) Constant Modulus Algorithm (CMA)
    The taps for the X polarisation are initialised to [0001000] and the Y polarisation is initialised orthogonally.

    Parameters
    ----------
    TrSyms : int
       number of symbols to use for training needs to be less than len(Ex)

    Ntaps   : int
       number of equaliser taps

    os      : int
       oversampling factor

    mu      : float
       step size parameter

    Ex, Ey     : array_like
       x and y polarisation of the signal field

    Returns
    -------

    EestX, EestY : array_like
       equalised x and y polarisation of the field

    wx, wy    : array_like
       equaliser taps for the x and y polarisation

    err       : array_like
       CMA estimation error for x and y polarisation
    """
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
    wXmaxidx = np.unravel_index(np.argmax(abs(wx)), wx.shape)
    wYmaxidx = np.unravel_index(np.argmax(abs(wy)), wy.shape)
    delay = abs(wYmaxidx[1]-wXmaxidx[1])
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

def partition_signal(signal, partitions, codebook):
    """
    Partition a signal according to their power

    Parameters
    ----------
    signal : array_like
        input signal to partition

    partitions : array_like
        partition vector defining the boundaries between the different blocks

    code   : array_like
        the values to partition tolerance

    Returns
    -------
    quanta : array_like
        partitioned quanta
    """
    quanta = []
    for datum in signal:
        index = 0
        while index < len(partitions) and datum > partitions[index]:
            index += 1
        quanta.append(codebook[index])
    return quanta

def partition_value(signal, partitions, codebook):
    """
    Partition a value according to their power

    Parameters
    ----------
    signal : float
        input value to partition

    partitions : array_like
        partition vector defining the boundaries between the different blocks

    code   : array_like
        the values to partition tolerance

    Returns
    -------
    quanta : float
        partitioned quanta
    """
    index = 0
    while index < len(partitions) and signal > partitions[index]:
        index += 1
    quanta = codebook[index]
    return quanta

def FS_CMA_RDE_16QAM(TrCMA, TrRDE, Ntaps, os, muCMA, muRDE, Ex, Ey):
    """
    Equalisation of PMD and residual dispersion of a 16 QAM signal based on a radius directed equalisation (RDE)
    fractionally spaced Constant Modulus Algorithm (FS-CMA)
    The taps for the X polarisation are initialised to [0001000] and the Y polarisation is initialised orthogonally.

    Parameters
    ----------
    TrCMA : int
       number of symbols to use for training the initial CMA needs to be less than len(Ex)

    TrRDE : int
       number of symbols to use for training the radius directed equaliser, needs to be less than len(Ex)

    Ntaps   : int
       number of equaliser taps

    os      : int
       oversampling factor

    muCMA      : float
       step size parameter for the CMA algorithm

    muRDE      : float
       step size parameter for the RDE algorithm

    Ex, Ey     : array_like
       x and y polarisation of the signal field

    Returns
    -------

    EestX, EestY : array_like
       equalised x and y polarisation of the field

    wx, wy    : array_like
       equaliser taps for the x and y polarisation

    err_cma, err_rde  : array_like
       CMA and RDE estimation error for x and y polarisation
    """
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
    """
    Static chromatic dispersion compensation of a single polarisation signal using overlap-add.
    All units are assumed to be SI.

    Parameters
    ----------
    fs   :  float
       sampling rate

    N    :  int
       block size (N=0, assumes cyclic boundary conditions and uses a single FFT/IFFT)

    L    :  float
       length of the compensated fibre 

    D    :  float
       dispersion 

    sig  : array_like
       single polarisation signal

    wl   : float
       center wavelength

    Returns
    -------

    sigEQ : array_like
       compensated signal
    """
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
        for i in range(0, B):
            sigB = np.zeros(N, dtype=np.complex128)
            sigB[zp:-zp] = sig[i*n:i*n+n]
            sigB = np.fft.fft(sigB)
            sigB *= H
            sigB = np.fft.ifft(sigB)
            sB[i, :] = sigB
            sigEQ[i*n:i*n+n+2*zp] = sigEQ[i*n:i*n+n+2*zp]+sigB
        sigEQ = sigEQ[zp:-zp]
    return sigEQ
