from __future__ import division
import numpy as np

from .. import utils
from ..theory import cal_qam_symbols, cal_qam_scaling_factor
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

try:
    from .equaliser_cython import FS_RDE, FS_CMA, FS_MRDE, FS_MCMA, FS_SBD, FS_MDDMA, FS_DD
except:
    #use python code if cython code is not available
    raise Warning("can not use cython training functions")
    from .training_python import FS_RDE, FS_CMA, FS_MRDE, FS_MCMA, FS_SBD, FS_MDDMA
from .training_python import FS_SCA, FS_CME

TRAINING_FCTS = {"cma": FS_CMA, "mcma": FS_MCMA,
                 "rde": FS_RDE, "mrde": FS_MRDE,
                 "sbd": FS_SBD, "mddma": FS_MDDMA,
                 "sca": FS_SCA, "cme": FS_CME,
                 "dd": FS_DD}


def _init_args(method, M, **kwargs):
    if method in ["mcma"]:
        return _cal_Rconstant_complex(M),
    elif method in ["cma"]:
        return _cal_Rconstant(M),
    elif method in ["rde"]:
        return generate_partition_codes_radius(M)
    elif method in ["mrde"]:
        return generate_partition_codes_complex(M)
    elif method in ["sca"]:
        return _cal_Rsca(M),
    elif method in ["cme"]:
        syms = cal_qam_symbols(M)/np.sqrt(cal_qam_scaling_factor(M))
        d = np.min(abs(np.diff(syms.real))) # should be fixed to consider different spacing between real and imag
        R = _cal_Rconstant(M)
        beta = kwargs['beta']
        return (R, d, beta)
    else:
        return cal_qam_symbols(M)/np.sqrt(cal_qam_scaling_factor(M)),


def apply_filter(E, os, wxy):
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
    if pols == 2:
        X2 = segment_axis(E[1], Ntaps, Ntaps-os)
        X = np.hstack([X1,X2])
        ww = np.vstack([wxy[0].flatten(), wxy[1].flatten()])
        Eest = np.dot(X, ww.transpose())
        return np.vstack([Eest[:,0],  Eest[:,1]])
    else:
        X = X1
        ww = wxy[0].flatten()
        Eest = np.dot(X, ww.transpose())
        return np.atleast_2d(Eest)

def _cal_Rdash(syms):
     return (abs(syms.real + syms.imag) + abs(syms.real - syms.imag)) * (np.sign(syms.real + syms.imag) + np.sign(syms.real-syms.imag) + 1.j*(np.sign(syms.real+syms.imag) - np.sign(syms.real-syms.imag)))*syms.conj()

def _cal_Rsca(M):
    syms = cal_qam_symbols(M)
    syms /= np.sqrt(cal_qam_scaling_factor(M))
    Rd = _cal_Rdash(syms)
    return np.mean((abs(syms.real + syms.imag) + abs(syms.real - syms.imag))**2 * Rd)/(4*np.mean(Rd))

def _cal_Rconstant(M):
    syms = cal_qam_symbols(M)
    scale = cal_qam_scaling_factor(M)
    syms /= np.sqrt(scale)
    return np.mean(abs(syms)**4)/np.mean(abs(syms)**2)

def _cal_Rconstant_complex(M):
    syms = cal_qam_symbols(M)
    scale = cal_qam_scaling_factor(M)
    syms /= np.sqrt(scale)
    return np.mean(syms.real**4)/np.mean(syms.real**2) + 1.j * np.mean(syms.imag**4)/np.mean(syms.imag**2)

def _init_taps(Ntaps, pols):
    wx = np.zeros((pols, Ntaps), dtype=np.complex128)
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
    syms = cal_qam_symbols(M)
    scale = cal_qam_scaling_factor(M)
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
    syms = cal_qam_symbols(M)
    scale = cal_qam_scaling_factor(M)
    syms /= np.sqrt(scale)
    codes = np.unique(abs(syms)**4/abs(syms)**2)
    parts = codes[:-1] + np.diff(codes)/2
    return parts, codes

def _lms_init(E, os, wxy, Ntaps, TrSyms, Niter):
    E = np.atleast_2d(E)
    pols = E.shape[0]
    L = E.shape[1]
    # scale signal
    E = utils.normalise_and_center(E)
    if wxy is None:
        wxy = [_init_taps(Ntaps, pols),]
        if pols == 2:
            wy = _init_orthogonaltaps(wxy[0])
            wxy = [wxy[0],wy]
    else:
        if pols == 2:
            Ntaps = wxy[0].shape[1]
        else:
            try:
                wxy = wxy.flatten()
                Ntaps = len(wxy)
                wxy = [wxy.copy(),]
            except:
                Ntaps = len(wxy[0])
    if not TrSyms:
        TrSyms = int(L//os//Ntaps-1)*int(Ntaps)
    err = np.zeros((pols, Niter * TrSyms ), dtype=np.complex128)
    # the copy below is important because otherwise the array will not be contiguous, which will cause issues in
    # the C functions
    Eout = E[:, :(TrSyms-1)*os+Ntaps].copy()
    return Eout, wxy, TrSyms, Ntaps, err, pols

def dual_mode_equalisation(E, os, mu, M, Ntaps, TrSyms=(None,None), Niter=(1,1), methods=("mcma", "sbd"), adaptive_stepsize=(False, False), **kwargs):
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

    Returns
    -------

    E         : array_like
        equalised signal X and Y polarisation

    (wx, wy)  : tuple(array_like, array_like)
       equaliser taps for the x and y polarisation

    (err1, err2)       : tuple(array_like, array_like)
       estimation error for x and y polarisation for each equaliser mode

    """
    wxy, err1 = equalise_signal(E, os, mu[0], M, Ntaps=Ntaps, TrSyms=TrSyms[0], Niter=Niter[0], method=methods[0], adaptive_stepsize=adaptive_stepsize[0], **kwargs)
    wxy2, err2 = equalise_signal(E, os, mu[1], M, wxy=wxy, TrSyms=TrSyms[1], Niter=Niter[1], method=methods[1], adaptive_stepsize=adaptive_stepsize[1], **kwargs)
    Eest = apply_filter(E, os, wxy2)
    return Eest, wxy2, (err1, err2)

def equalise_signal(E, os, mu, M, wxy=None, Ntaps=None, TrSyms=None, Niter=1, method="mcma", adaptive_stepsize=False, **kwargs):
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

    wxy     : tuple(array_like, array_like), optional
        tuple of the wx and wy filter taps. Either this or Ntaps has to be given.

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

    Returns
    -------

    (wx, wy)    : tuple(array_like, array_like)
       equaliser taps for the x and y polarisation

    err       : array_like
       estimation error for x and y polarisation

    """
    method = method.lower()
    training_fct = TRAINING_FCTS[method]
    args = _init_args(method, M, **kwargs)
    # scale signal
    E, wxy, TrSyms, Ntaps, err, pols = _lms_init(E, os, wxy, Ntaps, TrSyms, Niter)
    for i in range(Niter):
        print("LMS iteration %d"%i)
        for l in range(pols):
            err[l, i * TrSyms:(i+1)*TrSyms], wxy[l] = training_fct(E, TrSyms, Ntaps, os, mu, wxy[l], *args, adaptive=adaptive_stepsize)
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
