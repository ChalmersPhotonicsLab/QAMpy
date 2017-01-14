from __future__ import division
import pyximport
pyximport.install()
import numpy as np
import scipy.signal as scisig
import numexpr as ne

from ..segmentaxis import segment_axis
from .. import utils
from ..modulation import calculate_MQAM_symbols, calculate_MQAM_scaling_factor


try:
    from .equaliser_cython import FS_RDE_training, FS_CMA_training, FS_MRDE_training, FS_MCMA_training
except:
    #use python code if cython code is not available
    Warning("can not use cython training functions")
from .training_python import FS_RDE_training, FS_CMA_training, FS_MRDE_training, FS_MCMA_training

def _apply_filter(E, wx, wy, Ntaps, os):
    # equalise data points. Reuse samples used for channel estimation
    # this seems significantly faster than the previous method using a segment axis
    X1 = segment_axis(E[0], Ntaps, Ntaps-os)
    X2 = segment_axis(E[1], Ntaps, Ntaps-os)
    X = np.hstack([X1,X2])
    ww = np.vstack([wx.flatten(), wy.flatten()])
    Eest = np.dot(X, ww.transpose())
    return Eest[:,0],  Eest[:,1]


def _calculate_Rconstant(M):
    syms = calculate_MQAM_symbols(M)
    scale = calculate_MQAM_scaling_factor(M)
    syms /= np.sqrt(scale)
    return np.mean(abs(syms)**4)/np.mean(abs(syms)**2)

def _calculate_Rconstant_complex(M):
    syms = calculate_MQAM_symbols(M)
    scale = calculate_MQAM_scaling_factor(M)
    syms /= np.sqrt(scale)
    return np.mean(syms.real**4)/np.mean(syms.real**2) + 1.j * np.mean(syms.imag**4)/np.mean(syms.imag**2)

def _init_taps(Ntaps):
    wx = np.zeros((2, Ntaps), dtype=np.complex128)
    wx[0, Ntaps // 2] = 1
    return wx

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
    syms /= np.sqrt(scale)
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
    syms /= np.sqrt(scale)
    codes = np.unique(abs(syms)**4/abs(syms)**2)
    parts = codes[:-1] + np.diff(codes)/2
    return parts, codes

def FS_MCMA(E, TrSyms, Ntaps, os, mu, M):
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

    M       : integer
       QAM order 

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
    R = _calculate_Rconstant_complex(M)
    # scale signal
    E = utils.normalise_and_center(E)
    err = np.zeros((2, TrSyms), dtype=np.complex128)
    # ** training for X polarisation **
    wx = _init_taps(Ntaps)
    # run CMA
    err[0, :], wx = FS_MCMA_training(E, TrSyms, Ntaps, os, mu, wx, R)
    # ** training for y polarisation **
    wy = _init_orthogonaltaps(wx)
    # run CMA
    err[1, :], wy = FS_MCMA_training(E, TrSyms, Ntaps, os, mu, wy, R)
    # equalise data points. Reuse samples used for channel estimation
    EestX, EestY = _apply_filter(E, wx, wy, Ntaps, os)
    return np.vstack([EestX, EestY]), wx, wy, err

def FS_CMA(E, TrSyms, Ntaps, os, mu, M):
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

    M       : integer
       QAM order 

    Returns
    -------
    E = utils.normalise_and_center(E)

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
    # scale signal
    E = utils.normalise_and_center(E)
    R = _calculate_Rconstant(M)
    err = np.zeros((2, TrSyms), dtype=np.complex128)
    # ** training for X polarisation **
    wx = _init_taps(Ntaps)
    # run CMA
    err[0, :], wx = FS_CMA_training(E, TrSyms, Ntaps, os, mu, wx, R)
    # ** training for y polarisation **
    wy = _init_orthogonaltaps(wx)
    # run CMA
    err[1, :], wy = FS_CMA_training(E, TrSyms, Ntaps, os, mu, wy, R)
    # equalise data points. Reuse samples used for channel estimation
    EestX, EestY = _apply_filter(E, wx, wy, Ntaps, os)
    return np.vstack([EestX, EestY]), wx, wy, err

def FS_CMA_RDE(E, TrCMA, TrRDE, Ntaps, os, muCMA, muRDE, M):
    """
    Equalisation of PMD and residual dispersion of a M-QAM signal based on a radius directed equalisation (RDE)
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

    M       : integer
       QAM order

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
    # if can't have more training samples than field
    assert (TrCMA + TrRDE
            ) * os < L - Ntaps, "More training samples than overall samples"
    # constellation properties
    R = _calculate_Rconstant(M)
    part, code = generate_partition_codes_radius(M)
    # scale signal
    E = utils.normalise_and_center(E)
    err_cma = np.zeros((2, TrCMA), dtype=np.complex128)
    err_rde = np.zeros((2, TrRDE), dtype=np.complex128)
    # ** training for X polarisation **
    wx = _init_taps(Ntaps)
    # find taps with CMA
    err_cma[0, :], wx = FS_CMA_training(E, TrCMA, Ntaps, os, muCMA, wx, R)
    # use refine taps with RDE
    err_rde[0, :], wx = FS_RDE_training(E[:,TrCMA:], TrRDE, Ntaps, os, muRDE, wx,
                                        part, code)
    # ** training for y polarisation **
    wy = _init_orthogonaltaps(wx)#/np.sqrt(R)
    # find taps with CMA
    err_cma[1, :], wy = FS_CMA_training(E, TrCMA, Ntaps, os, muCMA, wy, R)
    # use refine taps with RDE
    err_rde[1, :], wy = FS_RDE_training(E[:,TrCMA:], TrRDE, Ntaps, os, muRDE, wy,
                                        part, code)
    # equalise data points. Reuse samples used for channel estimation
    EestX, EestY = _apply_filter(E, wx, wy, Ntaps, os)
    return np.vstack([EestX, EestY]), wx, wy, err_cma, err_rde

def FS_MCMA_MRDE(E, TrCMA, TrRDE, Ntaps, os, muCMA, muRDE, M):
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

    M     : int
       M-QAM order

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
    # if can't have more training samples than field
    assert (TrCMA + TrRDE
            ) * os < L - Ntaps, "More training samples than overall samples"
    # constellation properties
    part, code = generate_partition_codes_complex(16)
    R = _calculate_Rconstant_complex(M)
    # scale signal
    E = utils.normalise_and_center(E)
    # initialise error vectors
    err_cma = np.zeros((2, TrCMA), dtype=np.complex128)
    err_rde = np.zeros((2, TrRDE), dtype=np.complex128)
    # ** training for X polarisation **
    wx = _init_taps(Ntaps)
    # find taps with CMA
    err_cma[0, :], wx = FS_MCMA_training(E, TrCMA, Ntaps, os, muCMA, wx, R)
    # refine taps with RDE
    err_rde[0, :], wx = FS_MRDE_training(E[:,TrCMA:], TrRDE, Ntaps, os, muRDE, wx,
                                        part, code)
    # ** training for y polarisation **
    wy = _init_orthogonaltaps(wx)
    # find taps with CMA
    err_cma[1, :], wy = FS_MCMA_training(E, TrCMA, Ntaps, os, muCMA, wy, R)
    # refine taps with RDE
    err_rde[1, :], wy = FS_MRDE_training(E[:,TrCMA:], TrRDE, Ntaps, os, muRDE, wy,
                                        part, code)
    # equalise data points. Reuse samples used for channel estimation
    EestX, EestY = _apply_filter(E, wx, wy, Ntaps, os)
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
