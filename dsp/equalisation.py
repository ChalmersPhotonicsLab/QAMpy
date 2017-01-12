from __future__ import division
import pyximport
pyximport.install()
import numpy as np
import scipy.signal as scisig
import numexpr as ne

from .segmentaxis import segment_axis
from . import utils
from .modulation import calculate_MQAM_symbols, calculate_MQAM_scaling_factor


def FS_MCMA_training_python(E, TrSyms, Ntaps, os, mu, wx):
    """
    Training of the Modified CMA algorithm to determine the equaliser taps. Details in _[1]. Assumes a normalised signal.

    Parameters
    ----------
    E       : array_like
       dual polarisation signal field

    TrSyms : int
       number of symbols to use needs to be less than len(Ex)

    Ntaps   : int
       number of equaliser taps

    os      : int
       oversampling factor

    mu   : float
       step size parameter

    wx     : array_like
       initial equaliser taps
    
    Returns
    -------

    err       : array_like
       estimation error for x and y polarisation

    wx    : array_like
       equaliser taps

    Notes
    -----
    ..[1] Oh, K. N., & Chin, Y. O. (1995). Modified constant modulus algorithm: blind equalization and carrier phase recovery algorithm. Proceedings IEEE International Conference on Communications ICC ’95, 1, 498–502. http://doi.org/10.1109/ICC.1995.525219
    """
    err = np.zeros(TrSyms, dtype=np.complex)
    for i in range(TrSyms):
        X = E[:, i * os:i * os + Ntaps]
        Xest = np.sum(wx * X)
        err[i] = (np.abs(Xest.real)**2 -0.5) * Xest.real + (np.abs(Xest.imag)**2 - 0.5)*Xest.imag*1.j
        wx -= mu * err[i] * np.conj(X)
    return abs(err), wx


def FS_CMA_training_python(E, TrSyms, Ntaps, os, mu, wx):
    """
    Training of the CMA algorithm to determine the equaliser taps.

    Parameters
    ----------
    E      : array_like
       dual polarisation signal field

    TrSyms : int
       number of symbols to use for training needs to be less than len(Ex)

    Ntaps   : int
       number of equaliser taps

    os      : int
       oversampling factor

    mu      : float
       step size parameter

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
        X = E[:, i * os:i * os + Ntaps]
        Xest = np.sum(wx * X)
        err[i] = abs(Xest) - 1
        wx -= mu * err[i] * Xest * np.conj(X)
    return err, wx

def FS_MRDE_training_python(E, TrRDE, Ntaps, os, muRDE, wx, part, code):
    """
    Training of the Modified RDE algorithm to determine the equaliser taps. Details in _[1]. Assumes a normalised signal.

    Parameters
    ----------
    E       : array_like
       dual polarisation signal field

    TrRDE : int
       number of symbols to use for training the radius directed equaliser, needs to be less than len(Ex)

    Ntaps   : int
       number of equaliser taps

    os      : int
       oversampling factor

    muRDE   : float
       step size parameter

    wx     : array_like
       initial equaliser taps

    part    : array_like (complex)
       partitioning vector defining the boundaries between the different real and imaginary QAM constellation "rings"

    code    : array_like
       code vector defining the signal powers for the different QAM constellation rings

    Returns
    -------

    err       : array_like
       RDE estimation error for x and y polarisation

    wx    : array_like
       equaliser taps

    Notes
    -----
    ..[1] Oh, K. N., & Chin, Y. O. (1995). Modified constant modulus algorithm: blind equalization and carrier phase recovery algorithm. Proceedings IEEE International Conference on Communications ICC ’95, 1, 498–502. http://doi.org/10.1109/ICC.1995.525219
    """
    err = np.zeros(TrRDE, dtype=np.complex)
    for i in range(TrRDE):
        X = E[:, i * os:i * os + Ntaps]
        Xest = np.sum(wx * X)
        Ssq = Xest.real**2 + 1.j * Xest.imag**2
        R = partition_value(Ssq.real, part.real, code.real) + partition_value(Ssq.imag, part.imag, code.imag)
        err[i] = (Ssq.real - R.real)*Xest.real + (Ssq.imag - R.imag)*1.j*Xest.imag
        wx -= muRDE * err[i] * np.conj(X)
    return np.abs(err), wx


def FS_RDE_training_python(E, TrRDE, Ntaps, os, muRDE, wx, part, code):
    """
    Training of the RDE algorithm to determine the equaliser taps.

    Parameters
    ----------
    E       : array_like
       dual polarisation signal field

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
    for i in range(TrRDE):
        X = E[:, i * os:i * os + Ntaps]
        Xest = np.sum(wx * X)
        Ssq = abs(Xest)**2
        S_DD = partition_value(Ssq, part, code)
        err[i] = Ssq - S_DD #- Ssq
        wx -= muRDE * err[i] * Xest * np.conj(X)
    return err, wx


try:
    from .dsp_cython import FS_RDE_training
except:
    Warning("can not use cython RDE training")
    #use python code if cython code is not available
    FS_RDE_training = FS_RDE_training_python

try:
    from .dsp_cython import FS_CMA_training
except:
    Warning("can not use cython CMA training")
    #use python code if cython code is not available
    FS_CMA_training = FS_CMA_training_python

try:
    from .dsp_cython import FS_MRDE_training
except:
    Warning("can not use cython MRDE training")
    #use python code if cython code is not available
    FS_RDE_training = FS_MRDE_training_python

try:
    from .dsp_cython import FS_MCMA_training
except:
    Warning("can not use cython MCMA training")
    #use python code if cython code is not available
    FS_MCMA_training = FS_MCMA_training_python


def FS_MCMA(E, TrSyms, Ntaps, os, mu):
    """
    Equalisation of PMD and residual dispersion for an QPSK signal based on the Fractionally spaced (FS) Modified Constant Modulus Algorithm (CMA), see Oh and Chin _[1] for details.
    The taps for the X polarisation are initialised to [0001000] and the Y polarisation is initialised orthogonally.

    Parameters
    ----------
    E    : array_like
       x and y polarisation of the signal field (2D complex array first dim is the polarisation)

    TrSyms : int
       number of symbols to use for training needs to be less than len(Ex)

    Ntaps   : int
       number of equaliser taps

    os      : int
       oversampling factor

    mu      : float
       step size parameter

    Returns
    -------

    E     : array_like
       equalised x and y polarisation of the field (2D array first dimension is polarisation)

    wx, wy    : array_like
       equaliser taps for the x and y polarisation

    err       : array_like
       CMA estimation error for x and y polarisation

    References
    ----------
    ..[1] Oh, K. N., & Chin, Y. O. (1995). Modified constant modulus algorithm: blind equalization and carrier phase recovery algorithm. Proceedings IEEE International Conference on Communications ICC ’95, 1, 498–502. http://doi.org/10.1109/ICC.1995.525219
    """
    # if can't have more training samples than field
    L = E.shape[1]
    assert TrSyms*os < L - Ntaps, "More training samples than"\
                                    " overall samples"
    mu = mu / Ntaps
    # scale signal
    P = np.mean(utils.cabssquared(E))
    E = E / np.sqrt(P)
    err = np.zeros((2, TrSyms), dtype=np.complex128)
    # ** training for X polarisation **
    wx = np.zeros((2, Ntaps), dtype=np.complex128)
    wx[1, Ntaps // 2] = 1
    # run CMA
    err[0, :], wx = FS_MCMA_training(E, TrSyms, Ntaps, os, mu, wx)
    # ** training for y polarisation **
    wy = _init_orthogonaltaps(wx)
    # run CMA
    err[1, :], wy = FS_MCMA_training(E, TrSyms, Ntaps, os, mu, wy)
    # equalise data points. Reuse samples used for channel estimation
    X = segment_axis(E, Ntaps, Ntaps - os, axis=1)
    EestX = np.sum(wx[:, np.newaxis, :] * X, axis=(0, 2))
    EestY = np.sum(wy[:, np.newaxis, :] * X, axis=(0, 2))
    return np.vstack([EestX, EestY]), wx, wy, err

def FS_CMA(E, TrSyms, Ntaps, os, mu):
    """
    Equalisation of PMD and residual dispersion for an QPSK signal based on the Fractionally spaced (FS) Constant Modulus Algorithm (CMA)
    The taps for the X polarisation are initialised to [0001000] and the Y polarisation is initialised orthogonally.

    Parameters
    ----------
    E    : array_like
       x and y polarisation of the signal field (2D complex array first dim is the polarisation)

    TrSyms : int
       number of symbols to use for training needs to be less than len(Ex)

    Ntaps   : int
       number of equaliser taps

    os      : int
       oversampling factor

    mu      : float
       step size parameter

    Returns
    -------

    E     : array_like
       equalised x and y polarisation of the field (2D array first dimension is polarisation)

    wx, wy    : array_like
       equaliser taps for the x and y polarisation

    err       : array_like
       CMA estimation error for x and y polarisation
    """
    # if can't have more training samples than field
    L = E.shape[1]
    assert TrSyms*os < L - Ntaps, "More training samples than"\
                                    " overall samples"
    mu = mu / Ntaps
    # scale signal
    P = np.mean(utils.cabssquared(E))
    E = E / np.sqrt(P)
    err = np.zeros((2, TrSyms), dtype='float')
    # ** training for X polarisation **
    wx = np.zeros((2, Ntaps), dtype=np.complex128)
    wx[1, Ntaps // 2] = 1
    # run CMA
    err[0, :], wx = FS_CMA_training(E, TrSyms, Ntaps, os, mu, wx)
    # ** training for y polarisation **
    wy = _init_orthogonaltaps(wx)
    # run CMA
    err[1, :], wy = FS_CMA_training(E, TrSyms, Ntaps, os, mu, wy)
    # equalise data points. Reuse samples used for channel estimation
    X = segment_axis(E, Ntaps, Ntaps - os, axis=1)
    EestX = np.sum(wx[:, np.newaxis, :] * X, axis=(0, 2))
    EestY = np.sum(wy[:, np.newaxis, :] * X, axis=(0, 2))
    return np.vstack([EestX, EestY]), wx, wy, err

def _init_orthogonaltaps(wx):
    wy = np.zeros(wx.shape, dtype=np.complex128)
    # initialising the taps to be ortthogonal to the x polarisation
    #wy = -np.conj(wx)[::-1,::-1]
    wy = wx[::-1,::-1]
    # centering the taps
    wXmaxidx = np.unravel_index(np.argmax(abs(wx)), wx.shape)
    wYmaxidx = np.unravel_index(np.argmax(abs(wy)), wy.shape)
    delay = abs(wYmaxidx[0] - wXmaxidx[0])
    pad = np.zeros((2, delay), dtype=np.complex128)
    if delay > 0:
        wy = wy[:, delay:]
        wy = np.hstack([wy, pad])
    elif delay < 0:
        wy = wy[:, 0:Ntaps - delay - 1]
        wy = np.hstack([pad, wy])
    return wy

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

def FS_MCMA_MRDE_general(E, TrCMA, TrRDE, Ntaps, os, muCMA, muRDE, M):
    """
    Equalisation of PMD and residual dispersion of a M QAM signal based on a modified radius directed equalisation (RDE)
    fractionally spaced Constant Modulus Algorithm (FS-CMA). This equaliser is a dual mode equaliser, which performs intial convergence using the MCMA algorithm before switching to the MRDE.
    The taps for the X polarisation are initialised to [0001000] and the Y polarisation is initialised orthogonally. Some details can be found in _[1]

    Parameters
    ----------
    E    : array_like
       x and y polarisation of the signal field (2D complex array first dim is the polarisation)

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

    Returns
    -------

    E: array_like
       equalised x and y polarisation of the field (2D array first dimension is polarisation)

    wx, wy    : array_like
       equaliser taps for the x and y polarisation

    err_cma, err_rde  : array_like
       CMA and RDE estimation error for x and y polarisation

    References
    ----------
    ...[1] A FAMILY OF ALGORITHMS FOR BLIND EQUALIZATION OF QAM SIGNALS, Filho, Silva, Miranda
    """
    L = E.shape[1]
    muCMA = muCMA
    muRDE = muRDE
    # if can't have more training samples than field
    assert (TrCMA + TrRDE
            ) * os < L - Ntaps, "More training samples than overall samples"
    # constellation properties
    R = 13.2
    code = np.array([2., 10., 18.])
    part = np.array([5.24, 13.71])
    # scale signal
    P = np.mean(utils.cabssquared(E))
    E = E / np.sqrt(P)
    err_cma = np.zeros((2, TrCMA), dtype='float')
    err_rde = np.zeros((2, TrRDE), dtype='float')
    # ** training for X polarisation **
    wx = np.zeros((2, Ntaps), dtype=np.complex128)
    wx[1, Ntaps // 2] = 1
    # find taps with CMA
    err_cma[0, :], wx = FS_CMA_training(E, TrCMA, Ntaps, os, muCMA, wx)
    # scale taps for RDE
    #wx = np.sqrt(R) * wx
    # use refine taps with RDE
    err_rde[0, :], wx = FS_RDE_training(E[:,TrCMA:], TrRDE, Ntaps, os, muRDE, wx,
                                        part, code)
    # ** training for y polarisation **
    wy = _init_orthogonaltaps(wx)/np.sqrt(R)
    # find taps with CMA
    err_cma[1, :], wy = FS_CMA_training(E, TrCMA, Ntaps, os, muCMA, wy)
    # scale taps for RDE
    #wy = np.sqrt(R) * wy
    # use refine taps with RDE
    err_rde[1, :], wy = FS_RDE_training(E[:,TrCMA:], TrRDE, Ntaps, os, muRDE, wy,
                                        part, code)
    # equalise data points. Reuse samples used for channel estimation
    X = segment_axis(E, Ntaps, Ntaps - os, axis=1)
    EestX = np.sum(wx[:, np.newaxis, :] * X, axis=(0, 2))
    EestY = np.sum(wy[:, np.newaxis, :] * X, axis=(0, 2))
    return np.vstack([EestX, EestY]), wx, wy, err_cma, err_rde

def generate_partition_codes_complex(M):
    """
    Generate complex partitions and codes for M-QAM for MRDE based on the real and imaginary radii of the different symbols. The partitions define the boundaries between the different codes. This is used to determine on which real/imaginary radius a signal symbol should lie on. The real and imaginary parts should be used for parititioning the real and imaginary parts of the signal in MRDE.

    Parameters
    ----------
    M       : int
        M-QAM order

    Returns
    -------
    parts   : array_like
        the boundaries between the different codes for parititioning
    codes   : array_like
        the nearest symbol radius 
    """
    syms = calculate_MQAM_symbols(M)
    scale = calculate_MQAM_scaling_factor(M)
    syms /= scale
    syms_r = np.unique(abs(syms.real)**4/abs(syms.real)**2)
    syms_i = np.unique(abs(syms.imag)**4/abs(syms.imag)**2)
    codes = syms_r + 1.j * syms_i
    part_r = syms_r[:-1] + np.diff(syms_r)/2
    part_i = syms_i[:-1] + np.diff(syms_i)/2
    parts = part_r + 1.j*part_i
    return parts, codes

def generate_partition_codes_radius(M):
    """
    Generate partitions and codes for M-QAM for RDE based on the radius of the different symbols. The partitions define the boundaries between the different codes. This is used to determine on which radius a signal symbol should lie.

    Parameters
    ----------
    M       : int
        M-QAM order

    Returns
    -------
    parts   : array_like
        the boundaries between the different codes for parititioning
    codes   : array_like
        the nearest symbol radius 
    """
    syms = calculate_MQAM_symbols(M)
    scale = calculate_MQAM_scaling_factor(M)
    syms /= scale
    codes = np.unique(abs(syms)**4/abs(syms)**2)
    parts = codes[:-1] + np.diff(codes)/2
    return parts, codes


def FS_CMA_RDE_16QAM(E, TrCMA, TrRDE, Ntaps, os, muCMA, muRDE):
    """
    Equalisation of PMD and residual dispersion of a 16 QAM signal based on a radius directed equalisation (RDE)
    fractionally spaced Constant Modulus Algorithm (FS-CMA)
    The taps for the X polarisation are initialised to [0001000] and the Y polarisation is initialised orthogonally.

    Parameters
    ----------
    E    : array_like
       x and y polarisation of the signal field (2D complex array first dim is the polarisation)

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

    Returns
    -------

    E: array_like
       equalised x and y polarisation of the field (2D array first dimension is polarisation)

    wx, wy    : array_like
       equaliser taps for the x and y polarisation

    err_cma, err_rde  : array_like
       CMA and RDE estimation error for x and y polarisation
    """
    L = E.shape[1]
    muCMA = muCMA
    muRDE = muRDE
    # if can't have more training samples than field
    assert (TrCMA + TrRDE
            ) * os < L - Ntaps, "More training samples than overall samples"
    # constellation properties
    R = 13.2
    code = np.array([2., 10., 18.])
    part = np.array([5.24, 13.71])
    # scale signal
    P = np.mean(utils.cabssquared(E))
    E = E / np.sqrt(P)
    err_cma = np.zeros((2, TrCMA), dtype='float')
    err_rde = np.zeros((2, TrRDE), dtype='float')
    # ** training for X polarisation **
    wx = np.zeros((2, Ntaps), dtype=np.complex128)
    wx[1, Ntaps // 2] = 1
    # find taps with CMA
    err_cma[0, :], wx = FS_CMA_training(E, TrCMA, Ntaps, os, muCMA, wx)
    # scale taps for RDE
    #wx = np.sqrt(R) * wx
    # use refine taps with RDE
    err_rde[0, :], wx = FS_RDE_training(E[:,TrCMA:], TrRDE, Ntaps, os, muRDE, wx,
                                        part, code)
    # ** training for y polarisation **
    wy = _init_orthogonaltaps(wx)/np.sqrt(R)
    # find taps with CMA
    err_cma[1, :], wy = FS_CMA_training(E, TrCMA, Ntaps, os, muCMA, wy)
    # scale taps for RDE
    #wy = np.sqrt(R) * wy
    # use refine taps with RDE
    err_rde[1, :], wy = FS_RDE_training(E[:,TrCMA:], TrRDE, Ntaps, os, muRDE, wy,
                                        part, code)
    # equalise data points. Reuse samples used for channel estimation
    X = segment_axis(E, Ntaps, Ntaps - os, axis=1)
    EestX = np.sum(wx[:, np.newaxis, :] * X, axis=(0, 2))
    EestY = np.sum(wy[:, np.newaxis, :] * X, axis=(0, 2))
    return np.vstack([EestX, EestY]), wx, wy, err_cma, err_rde


def CDcomp(E, fs, N, L, D, wl):
    """
    Static chromatic dispersion compensation of a single polarisation signal using overlap-add.
    All units are assumed to be SI.

    Parameters
    ----------
    E  : array_like
       single polarisation signal

    fs   :  float
       sampling rate

    N    :  int
       block size (N=0, assumes cyclic boundary conditions and uses a single FFT/IFFT)

    L    :  float
       length of the compensated fibre

    D    :  float
       dispersion

    wl   : float
       center wavelength

    Returns
    -------

    sigEQ : array_like
       compensated signal
    """
    E = E.flatten()
    samp = len(E)
    #wl *= 1e-9
    #L = L*1.e3
    c = 2.99792458e8
    #D = D*1.e-6
    if N == 0:
        N = samp

#    H = np.zeros(N,dtype='complex')
    H = np.arange(0, N) + 1j * np.zeros(N, dtype='float')
    H -= N // 2
    H *= H
    H *= np.pi * D * wl**2 * L * fs**2 / (c * N**2)
    H = np.exp(-1j * H)
    #H1 = H
    H = np.fft.fftshift(H)
    if N == samp:
        sigEQ = np.fft.fft(E)
        sigEQ *= H
        sigEQ = np.fft.ifft(sigEQ)
    else:
        n = N // 2
        zp = N // 4
        B = samp // n
        sigB = np.zeros(N, dtype=np.complex128)
        sigEQ = np.zeros(n * (B + 1), dtype=np.complex128)
        sB = np.zeros((B, N), dtype=np.complex128)
        for i in range(0, B):
            sigB = np.zeros(N, dtype=np.complex128)
            sigB[zp:-zp] = E[i * n:i * n + n]
            sigB = np.fft.fft(sigB)
            sigB *= H
            sigB = np.fft.ifft(sigB)
            sB[i, :] = sigB
            sigEQ[i * n:i * n + n + 2 * zp] = sigEQ[i * n:i * n + n + 2 *
                                                    zp] + sigB
        sigEQ = sigEQ[zp:-zp]
    return sigEQ
