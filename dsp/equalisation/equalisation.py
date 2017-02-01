from __future__ import division
import pyximport
pyximport.install()
import numpy as np

from .. import utils
from ..modulation import calculate_MQAM_symbols, calculate_MQAM_scaling_factor
from ..segmentaxis import segment_axis


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
MCMA with adaptive step size after _[9]
SBD with adaptive step size  after _[10]
Decision Directed (DD) with adaptive step size after _[9]

References
----------
...[3] Oh, K. N., & Chin, Y. O. (1995). Modified constant modulus algorithm: blind equalization and carrier phase recovery algorithm. Proceedings IEEE International Conference on Communications ICC ’95, 1, 498–502. http://doi.org/10.1109/ICC.1995.525219
...[5] He, L., Amin, M. G., Reed, C., & Malkemes, R. C. (2004). A Hybrid Adaptive Blind Equalization Algorithm for QAM Signals in Wireless Communications, 52(7), 2058–2069.
...[6] Sheikh, S. A., & Fan, P. (2008). New blind equalization techniques based on improved square contour algorithm ✩, 18, 680–693. http://doi.org/10.1016/j.dsp.2007.09.001
...[7] Filho, M., Silva, M. T. M., & Miranda, M. D. (2008). A FAMILY OF ALGORITHMS FOR BLIND EQUALIZATION OF QAM SIGNALS. In 2011 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 6–9).
...[8] Fernandes, C. A. R., Favier, G., & Mota, J. C. M. (2007). Decision directed adaptive blind equalization based on the constant modulus algorithm. Signal, Image and Video Processing, 1(4), 333–346. http://doi.org/10.1007/s11760-007-0027-2
...[9] D. Ashmawy, K. Banovic, E. Abdel-Raheem, M. Youssif, H. Mansour, and M. Mohanna, “Joint MCMA and DD blind equalization algorithm with variable-step size,” Proc. 2009 IEEE Int. Conf. Electro/Information Technol. EIT 2009, no. 1, pp. 174–177, 2009.

"""

try:
    from .equaliser_cython import FS_RDE, FS_CMA, FS_MRDE, FS_MCMA, SBD, MDDMA, MCMA_adaptive, SBD_adaptive
except:
    #use python code if cython code is not available
    raise Warning("can not use cython training functions")
    from .training_python import FS_RDE, FS_CMA, FS_MRDE, FS_MCMA, SBD, MDDMA, FS_MCMA_adpative, FS_MCMA_adaptive2
from .training_python import FS_SCA, FS_CME

TRAINING_FCTS = {"cma": FS_CMA, "mcma": FS_MCMA,
                 "rde": FS_RDE, "mrde": FS_MRDE,
                 "sbd": SBD, "mddma": MDDMA,
                 "mcma_adaptive": MCMA_adaptive, "sbd_adaptive": SBD_adaptive,
                 "sca": FS_SCA, "cme": FS_CME}


def _init_args(method, M):
    if method in ["mcma", "mcma_adaptive"]:
        return _calculate_Rconstant_complex(M),
    elif method in ["cma"]:
        return _calculate_Rconstant(M),
    elif method in ["rde"]:
        return generate_partition_codes_radius(M),
    elif method in ["mrde"]:
        return generate_partition_codes_complex(M),
    elif method in ["sca"]:
        return _calculate_Rsca(M),
    else:
        return calculate_MQAM_symbols(M)/np.sqrt(calculate_MQAM_scaling_factor(M)),


def apply_filter(E, os, wxy):
    # equalise data points. Reuse samples used for channel estimation
    # this seems significantly faster than the previous method using a segment axis
    wx = wxy[0]
    wy = wxy[1]
    Ntaps = wx.shape[1]
    X1 = segment_axis(E[0], Ntaps, Ntaps-os)
    X2 = segment_axis(E[1], Ntaps, Ntaps-os)
    X = np.hstack([X1,X2])
    ww = np.vstack([wx.flatten(), wy.flatten()])
    Eest = np.dot(X, ww.transpose())
    return np.vstack([Eest[:,0],  Eest[:,1]])

def _calculate_Rdash(syms):
     return (abs(syms.real + syms.imag) + abs(syms.real - syms.imag)) * (sign(syms.real + syms.imag) + sign(syms.real-syms.imag) + 1.j*(sign(syms.real+syms.imag) - sign(syms.real-syms.imag)))*syms.conj()

def _calculate_Rsca(M):
    syms = calculate_MQAM_symbols(M)
    syms /= np.sqrt(calculate_MQAM_scaling_factor(M))
    Rd = _calculate_Rdash(syms)
    return np.mean((abs(syms.real + syms.imag) + abs(syms.real - syms.imag))**2 * Rd)/(4*np.mean(Rd))

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
    wy = -np.conj(wx[::-1,::-1])
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

def _lms_init(E, os, wxy, Ntaps, Niter):
    L = E.shape[1]
    # scale signal
    E = utils.normalise_and_center(E)
    if wxy is None:
        wx = _init_taps(Ntaps)
        wy = _init_orthogonaltaps(wx)
        empty_taps = True
    else:
        wx = wxy[0]
        wy = wxy[1]
        Ntaps = wx.shape[1]
        empty_taps = False
    TrSyms = int(L//os//Ntaps)*int(Ntaps)
    err = np.zeros((2, Niter * TrSyms ), dtype=np.complex128)
    return E[:,:TrSyms*os], wx, wy, TrSyms, Ntaps, err

def dual_mode_equalisation(E, os, mu, M, Ntaps, TrSyms=(None,None), Niter=(1,1), methods=("mcma", "sbd")):
    """
    Blind equalisation of PMD and residual dispersion, with a dual mode approach. Typically this is done using a CMA type initial equaliser for pre-convergence and a decision directed equaliser as a second to improve MSE. 


    Parameters
    ----------
    E    : array_like
        x and y polarisation of the signal field (2D complex array first dim is the polarisation)

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

    Returns
    -------

    E         : array_like
        equalised signal X and Y polarisation

    (wx, wy)  : tuple(array_like, array_like)
       equaliser taps for the x and y polarisation

    (err1, err2)       : tuple(array_like, array_like)
       estimation error for x and y polarisation for each equaliser mode

    """
    (wx, wy), err1 = equalise_signal(E, os, mu[0], M, Ntaps=Ntaps, TrSyms=TrSyms[0], Niter=Niter[0], method=methods[0])
    (wx2, wy2), err2 = equalise_signal(E, os, mu[0], M, wxy=(wx,wy), TrSyms=TrSyms[0], Niter=Niter[0], method=methods[0])
    EestX, EestY = apply_filter(E, os, (wx2, wy2))
    return np.vstack([EestX, EestY]), (wx2, wy2), (err1, err2)

def equalise_signal(E, os, mu, M, wxy=None, Ntaps=None, TrSyms=None, Niter=1, method="mcma_adaptive"):
    """
    Blind equalisation of PMD and residual dispersion, using a chosen equalisation method. The method can be any of the keys in the TRAINING_FCTS dictionary. 
    
    Parameters
    ----------
    E    : array_like
        x and y polarisation of the signal field (2D complex array first dim is the polarisation)

    os      : int
        oversampling factor

    mu      : float
        step size parameter

    M       : integer
        QAM order

    wxy     : tuple(array_like, array_like), optional
        tuple of the wx and wy filter taps. Either this or Ntaps has to be given.

    Ntaps   : int
        number of filter taps. Either this or wxy need to be given. If given taps are initialised as [00100]

    TrSyms  : int, optional
        number of symbols to use for filter estimation. Default is None which means use all symbols.

    Niter   : int, optional
        number of iterations. Default is one single iteration

    method  : string
        equaliser method has to be one of cma, rde, mrde, mcma, sbd, mddma, sca, dd_adaptive, sbd_adaptive, mcma_adaptive

    Returns
    -------

    (wx, wy)    : tuple(array_like, array_like)
       equaliser taps for the x and y polarisation

    err       : array_like
       estimation error for x and y polarisation

    """
    method = method.lower()
    training_fct = TRAINING_FCTS[method]
    args = _init_args(method, M)
    # scale signal
    E, wx, wy, TrSyms, Ntaps, err = _lms_init(E, os, wxy, Ntaps, Niter)
    for i in range(Niter):
        print("LMS iteration %d"%i)
        # run CMA
        err[0, i * TrSyms:(i+1)*TrSyms], wx = training_fct(E, TrSyms, Ntaps, os, mu, wx, *args)
        # run CMA
        err[1, i*TrSyms:(i+1)*TrSyms], wy = training_fct(E, TrSyms, Ntaps, os, mu, wy, *args)
    return (wx,wy), err


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
