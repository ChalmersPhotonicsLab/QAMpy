# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np

import qampy.helpers
from .. import utils
from ...theory import cal_symbols_qam, cal_scaling_factor_qam
from ..segmentaxis import segment_axis

#TODO: update documentation with all references

"""
    Equalisation functions the equaliser update functions provided are:

No decision based:
-----------------
Constant Modulus Algorithm (CMA) after _[1]
Radius Directed Error (RDE) after _[2]
Modfied Constant Modulus Algorithm (MCMA) after _[3]
Modified Radius Directed Error (MRDA) after _[7]
Constellation Matched Error Algorithm (CME) after _[5]
Square Contour Algorithm (SCA)  after _[6]

Decision Directed
-----------------
Symbol Based Decision (SBD) after _[7]
Modified Decision Directed Modulus Algorithm (MDDMA) after _[8]

Adaptive Step Size Algorithms
-----------------------------
based on the step size adoption in _[9]  it is possible to use an adaptive step for all equalisers using the adaptive_stepsize keyword parameter

References
----------
...[3] Oh, K. N., & Chin, Y. O. (1995). Modified constant modulus algorithm: blind equalization and carrier phase recovery algorithm. Proceedings IEEE International Conference on Communications ICC ’95, 1, 498–502. http://doi.org/10.1109/ICC.1995.525219
...[5] He, L., Amin, M. G., Reed, C., & Malkemes, R. C. (2004). A Hybrid Adaptive Blind Equalization Algorithm for QAM Signals in Wireless Communications, 52(7), 2058–2069.
...[6] Sheikh, S. A., & Fan, P. (2008). New blind equalization techniques based on improved square contour algorithm ✩, 18, 680–693. http://doi.org/10.1016/j.dsp.2007.09.001
...[7] Filho, M., Silva, M. T. M., & Miranda, M. D. (2008). A FAMILY OF ALGORITHMS FOR BLIND EQUALIZATION OF QAM SIGNALS. In 2011 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 6–9).
...[8] Fernandes, C. A. R., Favier, G., & Mota, J. C. M. (2007). Decision directed adaptive blind equalization based on the constant modulus algorithm. Signal, Image and Video Processing, 1(4), 333–346. http://doi.org/10.1007/s11760-007-0027-2
...[9] D. Ashmawy, K. Banovic, E. Abdel-Raheem, M. Youssif, H. Mansour, and M. Mohanna, “Joint MCMA and DD blind equalization algorithm with variable-step size,” Proc. 2009 IEEE Int. Conf. Electro/Information Technol. EIT 2009, no. 1, pp. 174–177, 2009.

"""

#TODO: include selection for either numba or cython code
try:
    from .cython_errorfcts import ErrorFctMCMA, ErrorFctMRDE, ErrorFctSBD, ErrorFctMDDMA, ErrorFctDD,\
        ErrorFctCMA, ErrorFctRDE, ErrorFctSCA, ErrorFctCME
    from .cython_equalisation import train_eq, ErrorFct
    from .cython_equalisation import apply_filter_to_signal as apply_filter_pyx
except:
    ##use python code if cython code is not available
    raise Warning("can not use cython training functions")
    from .numba_equalisation import ErrorFctMCMA, ErrorFctMRDE, ErrorFctSBD, ErrorFctMDDMA, ErrorFctDD,\
        ErrorFctCMA, ErrorFctRDE, ErrorFctSCA, ErrorFctCME, train_eq

TRAINING_FCTS = ["cma", "mcma",
                 "rde", "mrde",
                 "sbd", "mddma",
                 "sca", "cme",
                 "dd"]
def _select_errorfct(method, M, symbols, dtype, **kwargs):
    #TODO: investigate if it makes sense to include the calculations of constants inside the methods
    if method in ["mcma"]:
        return ErrorFctMCMA(_cal_Rconstant_complex(M))
    elif method in ["cma"]:
        return ErrorFctCMA(_cal_Rconstant(M))
    elif method in ["rde"]:
        p, c = generate_partition_codes_radius(M)
        return ErrorFctRDE(p, c)
    elif method in ["mrde"]:
        p, c = generate_partition_codes_complex(M)
        return ErrorFctMRDE(p, c)
    elif method in ["sca"]:
        return ErrorFctSCA(_cal_Rsca(M))
    elif method in ["cme"]:
        if symbols is None:
            syms = cal_symbols_qam(M)/np.sqrt(cal_scaling_factor_qam(M))
            syms = syms.astype(dtype)
        else:
            syms = symbols
        d = np.min(abs(np.diff(np.unique(syms.real)))) # should be fixed to consider different spacing between real and imag
        R = _cal_Rconstant(M)
        r2 = np.mean(abs(syms)**2)
        r4 = np.mean(abs(syms)**4)
        A = r4/r2
        bb = np.max(abs(syms*abs(syms)**2-A))
        try:
            beta = kwargs['beta']
        except:
            r2 = np.mean(abs(syms)**2)
            r4 = np.mean(abs(syms)**4)
            A = r4/r2
            beta = np.max(abs(syms*abs(syms)**2-A))/2
        return ErrorFctCME(R, d, beta)
    elif method in ['sbd']:
        if symbols is None:
            return ErrorFctSBD((cal_symbols_qam(M) / np.sqrt(cal_scaling_factor_qam(M))).astype(dtype))
        else:
            return ErrorFctSBD(symbols)
    elif method in ['mddma']:
        return ErrorFctMDDMA((cal_symbols_qam(M) / np.sqrt(cal_scaling_factor_qam(M))).astype(dtype))
    elif method in ['dd']:
        return ErrorFctDD((cal_symbols_qam(M) / np.sqrt(cal_scaling_factor_qam(M))).astype(dtype))
    else:
        raise ValueError("%s is unknown method"%method)

def apply_filter(E, os, wxy, method="pyx"):
    """
    Apply the equaliser filter taps to the input signal.

    Parameters
    ----------

    E      : array_like
        input signal to be equalised

    os     : int
        oversampling factor

    wxy    : tuple(array_like, array_like,optional)
        filter taps for the x and y polarisation

    method : string
        use python ("py") based or cython ("pyx") based function

    Returns
    -------

    Eest   : array_like
        equalised signal
    """
    if method == "py":
        return apply_filter_py(E, os, wxy)
    elif method == "pyx":
        return apply_filter_pyx(E, os, wxy)
    else:
        raise NotImplementedError("Only py and pyx methods are implemented")

def apply_filter_py(E, os, wxy):
    """
    Apply the equaliser filter taps to the input signal.

    Parameters
    ----------

    E      : array_like
        input signal to be equalised

    os     : int
        oversampling factor

    wxy    : tuple(array_like, array_like,optional)
        filter taps for the x and y polarisation

    Returns
    -------

    Eest   : array_like
        equalised signal
    """
    # equalise data points. Reuse samples used for channel estimation
    # this seems significantly faster than the previous method using a segment axis
    E = np.atleast_2d(E)
    pols = E.shape[0]
    Ntaps = wxy[0].shape[1]
    X1 = segment_axis(E[0], Ntaps, Ntaps-os)
    X = X1
    ww = wxy[0].flatten()

    # Case for a butterfly-configured EQ
    if pols > 1:
        for pol in range(1,pols):
            X_P = segment_axis(E[pol], Ntaps,Ntaps-os)
            X = np.hstack([X,X_P])
            ww = np.vstack([ww,wxy[pol].flatten()])

        # Compute the output
        Eest = np.dot(X,ww.transpose())

        # Extract the error (per butterfly entry)
        Eest_tmp = Eest[:,0]
        for pol in range(1,pols):
            Eest_tmp = np.vstack([Eest_tmp,Eest[:,pol]])
        Eest = Eest_tmp

    # Single mode EQ
    else:
        Eest = np.dot(X, ww.transpose())
        Eest = np.atleast_2d(Eest)

    return Eest

def _cal_Rdash(syms):
     return (abs(syms.real + syms.imag) + abs(syms.real - syms.imag)) * (np.sign(syms.real + syms.imag) + np.sign(syms.real-syms.imag) + 1.j*(np.sign(syms.real+syms.imag) - np.sign(syms.real-syms.imag)))*syms.conj()

def _cal_Rsca(M):
    syms = cal_symbols_qam(M)
    syms /= np.sqrt(cal_scaling_factor_qam(M))
    Rd = _cal_Rdash(syms)
    return np.mean((abs(syms.real + syms.imag) + abs(syms.real - syms.imag))**2 * Rd)/(4*np.mean(Rd))

def _cal_Rconstant(M):
    syms = cal_symbols_qam(M)
    scale = cal_scaling_factor_qam(M)
    syms /= np.sqrt(scale)
    return np.mean(abs(syms)**4)/np.mean(abs(syms)**2)

def _cal_Rconstant_complex(M):
    syms = cal_symbols_qam(M)
    scale = cal_scaling_factor_qam(M)
    syms /= np.sqrt(scale)
    return np.mean(syms.real**4)/np.mean(syms.real**2) + 1.j * np.mean(syms.imag**4)/np.mean(syms.imag**2)

def _init_taps(Ntaps, pols):
    wx = np.zeros((pols, Ntaps), dtype=np.complex128)
    wx[0, Ntaps // 2] = 1
    return wx

def orthogonalizetaps(wx):
    """
    Return taps orthogonal to the input taps. This only works for dual-polarization signals
    and follows the technique described in _[1] to avoid the CMA pol-demux singularity.

    Parameters
    ----------
    wx : array_like
        X-pol taps

    Returns
    -------
    wy : array_like
        Y-pol taps orthogonal to X-pol

    References
    ----------
    ..[1] L. Liu, et al. “Initial Tap Setup of Constant Modulus Algorithm for Polarization De-Multiplexing in
    Optical Coherent Receivers,” in Optical Fiber Communication Conference and National Fiber Optic Engineers Conference
    (2009), paper OMT2, 2009, p. OMT2.
    """
    Ntaps = wx.shape[1]
    wy = np.zeros(wx.shape, dtype=np.complex128)
    # initialising the taps to be opposite orthogonal to the x polarisation (note that we do not want fully orthogonal
    wy = np.conj(wx[::-1,::-1])
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
    syms = cal_symbols_qam(M)
    scale = cal_scaling_factor_qam(M)
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
    syms = cal_symbols_qam(M)
    scale = cal_scaling_factor_qam(M)
    syms /= np.sqrt(scale)
    codes = np.unique(abs(syms)**4/abs(syms)**2)
    parts = codes[:-1] + np.diff(codes)/2
    return parts, codes

def _lms_init(E, os, wxy, Ntaps, TrSyms, Niter):
    E = np.atleast_2d(E)
    pols = E.shape[0]
    L = E.shape[1]
    # scale signal
    E = qampy.helpers.normalise_and_center(E)
    if wxy is None:
        # Allocate matrix and set first taps
        wxy = np.zeros((pols,pols,Ntaps), dtype=np.complex128)
        wxy[0] = _init_taps(Ntaps, pols)

        # Add orthogonal taps to all other modes
        if pols > 1:
            for pol in range(1,pols):
                wxy[pol] = np.roll(wxy[0],pol,axis=0)

    else:
        wxy = np.asarray(wxy)
        if pols > 1:
            Ntaps = wxy[0].shape[1]
        else:
            try:
                wxy = wxy.flatten()
                Ntaps = len(wxy)
                wxy = np.asarray([wxy.copy(),])
            except:
                Ntaps = len(wxy[0])
    if TrSyms is None:
        TrSyms = int(L//os//Ntaps-1)*int(Ntaps)
    err = np.zeros((pols, Niter * TrSyms ), dtype=np.complex128)
    # the copy below is important because otherwise the array will not be contiguous, which will cause issues in
    # the C functions
    Eout = E[:, :(TrSyms-1)*os+Ntaps].copy()
    return Eout, wxy, TrSyms, Ntaps, err, pols

def dual_mode_equalisation(E, os, mu, M, Ntaps, TrSyms=(None,None), Niter=(1,1), methods=("mcma", "sbd"), adaptive_stepsize=(False, False), symbols=None,  avoid_cma_sing=(True, False), **kwargs):
    """
    Blind equalisation of PMD and residual dispersion, with a dual mode approach. Typically this is done using a CMA type initial equaliser for pre-convergence and a decision directed equaliser as a second to improve MSE. 


    Parameters
    ----------
    E    : array_like
        single or dual polarisation signal field (2D complex array first dim is the polarisation)

    os      : int
        oversampling factor

    mu      : tuple(float, float)
        step size parameter for the first and second equaliser method

    M       : integer
        QAM order

    Ntaps   : int
        number of filter taps. Either this or wxy need to be given. If given taps are initialised as [00100]

    TrSyms  : tuple(int,int) optional
        number of symbols to use for filter estimation for each equaliser mode. Default is (None, None) which means use all symbols in both equaliser.

    Niter   : tuple(int, int), optional
        number of iterations for each equaliser. Default is one single iteration for both

    method  : tuple(string,string), optional
        equaliser method for the first and second mode has to be one of cma, rde, mrde, mcma, sbd, mddma, sca, dd_adaptive, sbd_adaptive, mcma_adaptive

    adaptive_stepsize : tuple(bool, bool)
        whether to adapt the step size upon training for each of the equaliser modes

    symbols : array_like
        array of coded symbols to decide on for dd-based equalisation functions


    Returns
    -------

    E         : array_like
        equalised signal X and Y polarisation

    (wx, wy)  : tuple(array_like, array_like)
       equaliser taps for the x and y polarisation

    (err1, err2)       : tuple(array_like, array_like)
       estimation error for x and y polarisation for each equaliser mode

    """
    wxy, err1 = equalise_signal(E, os, mu[0], M, Ntaps=Ntaps, TrSyms=TrSyms[0], Niter=Niter[0], method=methods[0], adaptive_stepsize=adaptive_stepsize[0], symbols=symbols, avoid_cma_sing=avoid_cma_sing[0] **kwargs)
    wxy2, err2 = equalise_signal(E, os, mu[1], M, wxy=wxy, TrSyms=TrSyms[1], Niter=Niter[1], method=methods[1], adaptive_stepsize=adaptive_stepsize[1],  symbols=symbols, avoid_cma_sing=avoid_cma_sing[1], **kwargs)
    Eest = apply_filter(E, os, wxy2)
    return Eest, wxy2, (err1, err2)

def equalise_signal(E, os, mu, M, wxy=None, Ntaps=None, TrSyms=None, Niter=1, method="mcma", adaptive_stepsize=False,  symbols=None, avoid_cma_sing=True, **kwargs):
    """
    Blind equalisation of PMD and residual dispersion, using a chosen equalisation method. The method can be any of the keys in the TRAINING_FCTS dictionary. 
    
    Parameters
    ----------
    E    : array_like
        single or dual polarisation signal field (2D complex array first dim is the polarisation)

    os      : int
        oversampling factor

    mu      : float
        step size parameter

    M       : integer
        QAM order

    wxy     : array_like optional
        the wx and wy filter taps. Either this or Ntaps has to be given.

    Ntaps   : int
        number of filter taps. Either this or wxy need to be given. If given taps are initialised as [00100]

    TrSyms  : int, optional
        number of symbols to use for filter estimation. Default is None which means use all symbols.

    Niter   : int, optional
        number of iterations. Default is one single iteration

    method  : string, optional
        equaliser method has to be one of cma, rde, mrde, mcma, sbd, mddma, sca, dd_adaptive, sbd_adaptive, mcma_adaptive

    adaptive_stepsize : bool, optional
        whether to use an adaptive stepsize or a fixed
    symbols : array_like
        array of coded symbols to decide on for dd-based equalisation functions
    avoid_cma_sing : bool
        avoid the CMA polarization demux singularity by orthogonallizing taps after first pol convergence

    Returns
    -------

    (wx, wy)    : tuple(array_like, array_like)
       equaliser taps for the x and y polarisation

    err       : array_like
       estimation error for x and y polarisation

    """
    method = method.lower()
    eqfct = _select_errorfct(method, M, symbols, E.dtype, **kwargs)
    # scale signal
    E, wxy, TrSyms, Ntaps, err, pols = _lms_init(E, os, wxy, Ntaps, TrSyms, Niter)
    wxy = wxy.astype(E.dtype)
    for l in range(pols):
        for i in range(Niter):
            #err[l, i * TrSyms:(i+1)*TrSyms], wxy[l] = eqfct(E, TrSyms, Ntaps, os, mu, wxy[l],  adaptive=adaptive_stepsize)
            err[l, i * TrSyms:(i+1)*TrSyms], wxy[l] = train_eq(E, TrSyms, os, mu, wxy[l], eqfct, adaptive=adaptive_stepsize)
        if (l < 1) and avoid_cma_sing:
            wxy[l+1] = orthogonalizetaps(wxy[l])

    return wxy, err



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

    c = 2.99792458e8
    if N == 0:
        N = samp

#    H = np.zeros(N,dtype='complex')
    #H = np.arange(0, N) + 1j * np.zeros(N, dtype='float')
    #H -= N // 2
    #H *= 2*np.pi
    #H *= fs
    #H *= H
    #H *= D * wl**2 * L / (c * N**2)

    omega = np.pi * fs * np.linspace(-1,1,N,dtype = complex)
    beta2 = D * wl**2 / (c * 2 * np.pi)

    H = np.exp(-.5j * omega**2 * beta2 * L )
    #H1 = H
    #H = np.fft.fftshift(H)
    if N == samp:
        sigEQ = np.fft.fftshift(np.fft.fft(E))
        sigEQ *= H
        sigEQ = np.fft.ifft(np.fft.ifftshift(sigEQ))
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
    return sigEQ, H
