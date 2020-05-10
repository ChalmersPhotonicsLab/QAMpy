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
Radius Directed Error (RDE) after _[1]
Modfied Constant Modulus Algorithm (MCMA) after _[2]
Modified Radius Directed Error (MRDA) after _[3]
Decision-directed LMS (DD) after _[1]

Decision Directed
-----------------
Symbol Based Decision (SBD) after _[3]
Modified Decision Directed Modulus Algorithm (MDDMA) after _[4]

Adaptive Step Size Algorithms
-----------------------------
based on the step size adoption in _[5]  it is possible to use an adaptive step for all equalisers using the adaptive_stepsize keyword parameter

Real Valued
-----------
There are also real-valued CMA and DD as well as a data-aided DD algorithm

Data aided
----------
In addition there is a data aided SBD algorithm and real_valued DD algorithm

References
----------
...[1] M. S. Faruk and S. J. Savory, ‘Digital Signal Processing for Coherent Transceivers Employing Multilevel Formats’, Journal of Lightwave Technology, vol. 35, no. 5, pp. 1125–1141, Mar. 2017, doi: 10.1109/JLT.2017.2662319.
...[2] Oh, K. N., & Chin, Y. O. (1995). Modified constant modulus algorithm: blind equalization and carrier phase recovery algorithm. Proceedings IEEE International Conference on Communications ICC ’95, 1, 498–502. http://doi.org/10.1109/ICC.1995.525219
...[3] Filho, M., Silva, M. T. M., & Miranda, M. D. (2008). A FAMILY OF ALGORITHMS FOR BLIND EQUALIZATION OF QAM SIGNALS. In 2011 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 6–9).
...[8] Fernandes, C. A. R., Favier, G., & Mota, J. C. M. (2007). Decision directed adaptive blind equalization based on the constant modulus algorithm. Signal, Image and Video Processing, 1(4), 333–346. http://doi.org/10.1007/s11760-007-0027-2
...[5] D. Ashmawy, K. Banovic, E. Abdel-Raheem, M. Youssif, H. Mansour, and M. Mohanna, “Joint MCMA and DD blind equalization algorithm with variable-step size,” Proc. 2009 IEEE Int. Conf. Electro/Information Technol. EIT 2009, no. 1, pp. 174–177, 2009.

"""

from __future__ import division
import numpy as np

import qampy.helpers
from qampy.theory import cal_symbols_qam, cal_scaling_factor_qam
from qampy.core.segmentaxis import segment_axis
from qampy.core.equalisation import pythran_equalisation

DECISION_BASED = ["sbd", "mddma", "dd", "sbd_data", "dd_real", "dd_data_real"]
NONDECISION_BASED = ["cma", "mcma", "rde", "mrde", "cma_real"]
REAL_VALUED = ["cma_real", "dd_real", "dd_data_real"]
DATA_AIDED = ["dd_data_real", "sbd_data"]

TRAINING_FCTS =  DECISION_BASED + NONDECISION_BASED

def generate_symbols_for_eq(method, M, dtype):
    #TODO: investigate if it makes sense to include the calculations of constants inside the methods
    if method in ["cma"]:
        return np.atleast_2d(_cal_Rconstant(M) + 0j).astype(dtype)
    if method in ["mcma"]:
        return np.atleast_2d(_cal_Rconstant_complex(M)).astype(dtype)
    if method in ["rde"]:
        p = generate_partition_codes_radius(M)
        return np.atleast_2d(p+0j).astype(dtype)
    if method in ["mrde"]:
        p = generate_partition_codes_complex(M)
        return  np.atleast_2d(p).astype(dtype)
    if method in ["sbd"]:
        symbols = np.atleast_2d(cal_symbols_qam(M)/np.sqrt(cal_scaling_factor_qam(M))).astype(dtype)
        return symbols
    if method in ["mddma"]:
        symbols = np.atleast_2d(cal_symbols_qam(M)/np.sqrt(cal_scaling_factor_qam(M))).astype(dtype)
        return  symbols
    if method in ["dd"]:
        symbols = np.atleast_2d(cal_symbols_qam(M)/np.sqrt(cal_scaling_factor_qam(M))).astype(dtype)
        return symbols
    if method in ["cma_real"]:
        return np.repeat([np.atleast_1d(_cal_Rconstant_complex(M).real.astype(dtype))], 2, axis=0)
    if method in ["dd_real"]:
        symbols = cal_symbols_qam(M)/np.sqrt(cal_scaling_factor_qam(M))
        symbols_out = np.vstack([symbols.real, symbols.imag]).astype(dtype)
        return symbols_out
    if method in ["dd_data_real", "sbd_data"]:
        raise ValueError("%s is a data-aided method and needs the symbols to be passed"%method)
    raise ValueError("%s is unknown method"%method)

def apply_filter(E, os, wxy, method="pyt", modes=None):
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
    
    modes : array_like, optional
        mode numbers over which to apply the filters

    Returns
    -------

    Eest   : array_like
        equalised signal
    """
    E = np.copy(E) # pythran requires non-reshaped arrays, copy to make sure they are
    wxy = np.copy(wxy)
    if modes is None:
        modes = np.arange(wxy.shape[0])
    else:
        modes = np.copy(np.atleast_1d(modes))
    nmodes = modes.shape[0]
    if method == "py":
        return apply_filter_py(E, os, wxy)
    elif method == "pyt":
        if np.iscomplexobj(E) and np.iscomplexobj(wxy):
            return pythran_equalisation.apply_filter_to_signal(E, os, wxy, modes)
        elif np.iscomplexobj(E):
            E  = _convert_sig_to_real(E)
        Etmp = pythran_equalisation.apply_filter_to_signal(E, os, wxy, modes)
        if E.itemsize == 8:
            return _convert_sig_to_cmplx(Etmp, nmodes, np.complex128(1j))
        elif E.itemsize == 4:
            return _convert_sig_to_cmplx(Etmp, nmodes, np.complex64(1j))
        else:
            raise ValueError("The field has an unknown data type")
    else:
        raise NotImplementedError("Only py and pythran methods are implemented")

def apply_filter_py(E, os, wxy, modes=None):
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
    
    modes : array_like, optional
        mode numbers over which to apply the filters

    Returns
    -------

    Eest   : array_like
        equalised signal
    """
    # equalise data points. Reuse samples used for channel estimation
    # this seems significantly faster than the previous method using a segment axis
    # TODO something fails still in this function
    E = np.atleast_2d(E)
    Ntaps = wxy.shape[-1]
    nmodes_max = wxy.shape[0]
    if modes is None:
        modes = np.arange(nmodes_max)
        nmodes = nmodes_max
    else:
        modes = np.atleast_1d(modes)
        assert np.max(modes) < nmodes_max, "largest mode number is larger than shape of signal"
        nmodes = modes.sizeX = X1
    X = segment_axis(E[modes[0]], Ntaps, Ntaps-os)
    ww = wxy[0].flatten()
    # Case for a butterfly-configured EQ
    if nmodes_max > 1:
        for pol in range(1, nmodes):
            X_P = segment_axis(E[modes[pol]], Ntaps,Ntaps-os)
            X = np.hstack([X,X_P])
            ww = np.vstack([ww,wxy[modes[pol]].flatten()])

        # Compute the output
        Eest = np.dot(X,ww.transpose())

        # Extract the error (per butterfly entry)
        Eest_tmp = Eest[:,0]
        for pol in range(1, nmodes):
            Eest_tmp = np.vstack([Eest_tmp,Eest[:,modes[pol]]])
        Eest = Eest_tmp

    # Single mode EQ
    else:
        Eest = np.dot(X, ww.transpose())
        Eest = np.atleast_2d(Eest)

    return Eest

def _convert_sig_to_real(E):
    Etmp = np.zeros((2*E.shape[0], E.shape[1]), dtype=E.real.dtype)
    Etmp[:E.shape[0]] = E.real
    Etmp[E.shape[0]:] = E.imag
    return np.ascontiguousarray(Etmp)

def _convert_sig_to_cmplx(E, modes, Im=np.complex128(1j)):
    return E[:modes//2,:] + Im * E[modes//2:,:]

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

def _cal_training_symbol_len(os, ntaps, L):
    return int(L//os//ntaps-1)*int(ntaps)

def _init_taps(Ntaps, nmodes, nmodes2, dtype):
    wxy = np.zeros((nmodes, nmodes2, Ntaps), dtype=dtype)
    for i in range(nmodes):
        wxy[i, i, Ntaps // 2] = 1
    # we could use gaussian initialisation, but no big advantage
    #itaps = np.arange(0, Ntaps)-Ntaps//2
    #width = Ntaps//8
    #wx[0] = np.exp(-itaps**2/2/width**2)
    #wx[0] = wx[0]/np.sqrt(np.sum(abs(wx[0])**2))
    return wxy

def _lms_init(E, os, wxy, Ntaps, TrSyms, mu):
    E = np.atleast_2d(E)
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

def dual_mode_equalisation(E, os, mu, M, wxy=None, Ntaps=None, TrSyms=(None,None), Niter=(1,1), methods=("mcma", "sbd"), adaptive_stepsize=(False, False), symbols=None,  modes=None, apply=True, **kwargs):
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
        array of symbol arrays. If only a single array is given they are used for both dimensions (default=None, use the 
        symbols generated for the QAM format)
    
    modes : array_like, optional
        array or list  of modes to  equalise over (default=None  equalise over all modes of the input signal)

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
    symbols = np.atleast_1d(symbols)
    if symbols.ndim < 3:
        symbols = np.tile(symbols, (2,1,1))
    wxy, err1 = equalise_signal(E, os, mu[0], M, wxy=wxy, Ntaps=Ntaps, TrSyms=TrSyms[0], Niter=Niter[0], method=methods[0], adaptive_stepsize=adaptive_stepsize[0], symbols=symbols[0],  modes=modes,**kwargs)
    wxy2, err2 = equalise_signal(E, os, mu[1], M, wxy=wxy, TrSyms=TrSyms[1], Niter=Niter[1], method=methods[1], adaptive_stepsize=adaptive_stepsize[1],  symbols=symbols[1],  modes=modes, **kwargs)
    if apply:
        Eest = apply_filter(E, os, wxy2, modes=modes)
        return Eest, wxy2, (err1, err2)
    else:
        return wxy2, (err1, err2)

def equalise_signal(E, os, mu, M, wxy=None, Ntaps=None, TrSyms=None, Niter=1, method="mcma", adaptive_stepsize=False,  symbols=None,  modes=None, apply=False,  **kwargs):
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
        equaliser method has to be one of cma, mcma, rde, mrde, sbd, sbd_data, mddma, dd

    adaptive_stepsize : bool, optional
        whether to use an adaptive stepsize or a fixed
        
    symbols : array_like, optional
        array of coded symbols to decide on for dd-based equalisation functions (default=None, generate symbols for this
        QAM format)
        
    modes: array_like, optional
        array or list  of modes to  equalise over (default=None  equalise over all modes of the input signal)

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
    dtype_c = E.dtype
    if method in REAL_VALUED:
        E = _convert_sig_to_real(E)
    else:
        E = np.copy(E) #  copy to make pythran happy
    mu = E.real.dtype.type(mu)
    nmodes = E.shape[0]
    if modes is None:
        modes = np.arange(nmodes)
    else:
        if method in REAL_VALUED:
            modes = np.atleast_1d(modes)
            modes = np.hstack([modes, modes+nmodes//2])
        else:
            modes = np.atleast_1d(modes)
        assert np.max(modes) < nmodes, "largest mode number is larger than shape of signal"
    if wxy is None:
        wxy = _init_taps(Ntaps, nmodes, nmodes, E.dtype)
    else:
        wxy = np.ascontiguousarray(wxy, dtype=E.dtype)
        Ntaps = wxy.shape[-1]
        assert wxy.ndim == 3, "wxy needs to be three dimensional"
        assert wxy.shape[:2] == (nmodes, nmodes), "The first 2 dimensions of wxy need to be the same shape as E"
    if TrSyms is None:
        TrSyms = _cal_training_symbol_len(os, Ntaps, E.shape[-1])
    symbols = _reshape_symbols(symbols, method, M, E.dtype, nmodes)
    if method in REAL_VALUED:
        err, wxy, mu = pythran_equalisation.train_equaliser_realvalued(E, TrSyms, Niter, os, mu, wxy, modes, adaptive_stepsize, symbols.copy(), method[:-5]) # copies are needed because pythran has problems with reshaped arrays
    else:
        err, wxy, mu = pythran_equalisation.train_equaliser(E, TrSyms, Niter, os, mu, wxy, modes, adaptive_stepsize, symbols.copy(), method) # copies are needed because pythran has problems with reshaped arrays
    if apply:
        # TODO: The below is suboptimal because we should really only apply to the selected modes for efficiency
        Eest = apply_filter(E, os, wxy, modes=modes)
        if method in REAL_VALUED:
            return Eest, wxy, err
        else:
            return Eest, wxy, err
    else:
        return wxy, err
    
def _reshape_symbols(symbols, method, M, dtype, nmodes):
    if symbols is None or method in NONDECISION_BASED: # This code currently prevents passing "symbol arrays for RDE or CMA algorithms
        symbols = generate_symbols_for_eq(method, M, dtype)
    if method not in REAL_VALUED:
        if symbols.ndim == 1 or symbols.shape[0] == 1:
            symbols = np.tile(symbols, (nmodes, 1))
        elif symbols.shape[0] != nmodes:
            raise ValueError("Symbols array is shape {} but signal has {} modes, symbols must be 1d or of shape (1, N) or ({}, N)".format(symbols.shape, nmodes, nmodes))
        symbols = symbols.astype(dtype)
        symbols = np.atleast_2d(symbols)
    else:
        if np.iscomplexobj(symbols):
            if symbols.ndim == 1 or symbols.shape[0] == 1:
                symbols = np.repeat([symbols.real, symbols.imag], nmodes//2, axis=0).squeeze()
                symbols = symbols.reshape(nmodes, -1)
            elif symbols.shape[0] == nmodes//2:
                symbols = np.vstack([symbols.real, symbols.imag])
            else:
                raise ValueError("Symbols array is  complex and has {} modes, but needs to either have one mode or the same modes as the signal ({})".format(symbols.shape[0], nmodes//2))
        else:
            if symbols.shape[0] == 2 and nmodes > 2:
                symbols = np.repeat([symbols[0], symbols[1]], nmodes//2, axis=0).squeeze()
                symbols = symbols.reshape(nmodes, -1)
            elif symbols.shape[0] != nmodes:
                raise ValueError("Symbols array is shape {} but signal has {} modes, symbols must be 1d or of shape (1, N) or ({}, N)".format(symbols.shape, nmodes, nmodes))
        symbols = symbols.astype(dtype)
    return symbols

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
