# -*- coding: utf-8 -*-
#  This file is part of QAMpy.
#
#  QAMpy is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  Foobar is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with QAMpy.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright 2018 Jochen Schröder, Mikael Mazur

#TODO: update documentation with all references

"""
    Equalisa tion functions the equaliser update functions provided are:

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

from __future__ import division
import numpy as np

import qampy.helpers
from qampy.theory import cal_symbols_qam, cal_scaling_factor_qam
from qampy.core.segmentaxis import segment_axis
from qampy.core.equalisation import pythran_equalisation

TRAINING_FCTS = ["cma", "mcma",
                 "rde", "mrde",
                 "sbd", "mddma",
                 "dd"]

def _select_errorfct(method, M, symbols, dtype, **kwargs):
    #TODO: investigate if it makes sense to include the calculations of constants inside the methods
    if method in ["cma"]:
        return (_cal_Rconstant(M) + 0j).astype(dtype)
    if method in ["mcma"]:
        return _cal_Rconstant_complex(M).astype(dtype)
    if method in ["rde"]:
        p = generate_partition_codes_radius(M)
        return (p+0j).astype(dtype)
    if method in ["mrde"]:
        p = generate_partition_codes_complex(M)
        return  p.astype(dtype)
    if method in ["sbd"]:
        if symbols is None:
            symbols = (cal_symbols_qam(M)/np.sqrt(cal_scaling_factor_qam(M))).astype(dtype)
        return symbols
    if method in ["mddma"]:
        if symbols is None:
            symbols = (cal_symbols_qam(M)/np.sqrt(cal_scaling_factor_qam(M))).astype(dtype)
        return  symbols
    if method in ["dd"]:
        if symbols is None:
            symbols = (cal_symbols_qam(M)/np.sqrt(cal_scaling_factor_qam(M))).astype(dtype)
        return symbols
    raise ValueError("%s is unknown method"%method)

def apply_filter(E, os, wxy, method="pyt"):
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
        use python ("py") based or pythran ("pyt") based function

    Returns
    -------

    Eest   : array_like
        equalised signal
    """
    if method == "py":
        return apply_filter_py(E, os, wxy)
    elif method == "pyt":
        return pythran_equalisation.apply_filter_to_signal(E.copy(), os, wxy.copy())
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
    # we could use gaussian initialisation, but no big advantage
    #itaps = np.arange(0, Ntaps)-Ntaps//2
    #width = Ntaps//8
    #wx[0] = np.exp(-itaps**2/2/width**2)
    #wx[0] = wx[0]/np.sqrt(np.sum(abs(wx[0])**2))
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
    return np.hstack([parts, codes])

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
    return np.hstack([parts,codes])

def _lms_init(E, os, wxy, Ntaps, TrSyms, mu):
    E = np.atleast_2d(E)
    pols = E.shape[0]
    L = E.shape[1]
    # scale signal
    E = qampy.helpers.normalise_and_center(E)
    if wxy is None:
        # Allocate matrix and set first taps
        wxy = np.zeros((pols,pols,Ntaps), dtype=E.dtype)
        wxy[0] = _init_taps(Ntaps, pols)
        # Add orthogonal taps to all other modes
        if pols > 1:
            for pol in range(1,pols):
                wxy[pol] = np.roll(wxy[0],pol,axis=0)
    else:
        wxy = np.asarray(wxy, dtype=E.dtype)
        Ntaps = wxy[0].shape[1]
    if TrSyms is None:
        TrSyms = int(L//os//Ntaps-1)*int(Ntaps)
    if E.dtype.type == np.complex64:
        mu = np.float32(mu)
    # the copy below is important because otherwise the array will not be contiguous, which will cause issues in
    # the C functions
    Eout = E[:, :(TrSyms-1)*os+Ntaps].copy()
    return Eout, wxy, TrSyms, Ntaps, mu, pols

def dual_mode_equalisation(E, os, mu, M, Ntaps, TrSyms=(None,None), Niter=(1,1), methods=("mcma", "sbd"), adaptive_stepsize=(False, False), symbols=None,  avoid_cma_sing=(False, False), selected_modes = None, apply=True, **kwargs):
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

    symbols : array_like, optional
        tuple of symbol arrays. If only a single array is given they are used for both dimensions

    apply: Bool, optional
        whether to apply the filter taps and return the equalised signal

    Returns
    -------
    E         : array_like
        equalised signal X and Y polarisation

    (wx, wy)  : tuple(array_like, array_like)
       equaliser taps for the x and y polarisation

    (err1, err2)       : tuple(array_like, array_like)
       estimation error for x and y polarisation for each equaliser mode

    if apply is False do not return E
    """
    if len(symbols) == 1 or symbols.ndim == 1:
        symbols = [symbols, symbols]
    wxy, err1 = equalise_signal(E, os, mu[0], M, Ntaps=Ntaps, TrSyms=TrSyms[0], Niter=Niter[0], method=methods[0], adaptive_stepsize=adaptive_stepsize[0], symbols=symbols[0], avoid_cma_sing=avoid_cma_sing[0], selected_modes = None,**kwargs)
    wxy2, err2 = equalise_signal(E, os, mu[1], M, wxy=wxy, TrSyms=TrSyms[1], Niter=Niter[1], method=methods[1], adaptive_stepsize=adaptive_stepsize[1],  symbols=symbols[0], avoid_cma_sing=avoid_cma_sing[1],selected_modes = None, **kwargs)
    if apply:
        Eest = apply_filter(E, os, wxy2)
        return Eest, wxy2, (err1, err2)
    else:
        return wxy2, (err1, err2)

def equalise_signal(E, os, mu, M, wxy=None, Ntaps=None, TrSyms=None, Niter=1, method="mcma", adaptive_stepsize=False,  symbols=None, avoid_cma_sing=False, apply=False, selected_modes = None, **kwargs):
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
    apply: Bool, optional
        whether to apply the filter taps and return the equalised signal

    Returns
    -------
    if apply:
        E : array_like
        equalised signal

    (wx, wy)    : tuple(array_like, array_like)
       equaliser taps for the x and y polarisation

    err       : array_like
       estimation error for x and y polarisation

    """
    method = method.lower()
    E, wxy, TrSyms, Ntaps, mu, pols = _lms_init(E, os, wxy, Ntaps, TrSyms, mu)
    wxy = wxy.astype(E.dtype)
    symbols = _select_errorfct(method, M, symbols, E.dtype, **kwargs)
    err, wxy, mu = pythran_equalisation.train_equaliser(E, TrSyms, Niter, os, mu, wxy, adaptive_stepsize, np.atleast_2d(symbols), method)
    if apply:
        # TODO: The below is suboptimal because we should really only apply to the selected modes for efficiency
        Eest = apply_filter(E, os, wxy)
        return np.squeeze(Eest[selected_modes]), np.squeeze(wxy[selected_modes]), err
    else:
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
