from __future__ import division, print_function
import numpy as np
from .utils import cabssquared
from .theory import  cal_symbols_qam, cal_scaling_factor_qam
from .equalisation import quantize as _quantize_pyx
from . import modulation, ber_functions
from .dsp_cython import soft_l_value_demapper
import numba

try:
    import arrayfire as af
except ImportError:
    af = None

def quantize(signal, symbols, method="pyx", **kwargs):
    """
    Quantize signal array onto symbols.

    Parameters
    ----------
    signal : array_like
        input signal array
    symbols : array_like
        array of symbols to quantize onto
    method : string, optional
        what method to use ('af' for arrayfire or 'pyx' for python)
    kwargs
        keyword arguments passed to pyx or af functions

    Returns
    -------
    out : array_like
        array of quantized symbols

    """
    if method == "pyx":
        return _quantize_pyx(signal, symbols, **kwargs)
    elif method == "af":
        if af == None:
            raise RuntimeError("Arrayfire was not imported so cannot use this method for quantization")
        return _quantize_af(signal, symbols, **kwargs)
    else:
        raise ValueError("method '%s' unknown has to be either 'pyx' or 'af'"%(method))


def _quantize_af(signal, symbols, precision=16):
    """
    Quantize signal array  onto symbols. Arrayfire function.

    Parameters
    ----------
    signal : array_like
        input signal array
    symbols : array_like
        array of symbols to quantize onto
    precision : int, optional
        bit precision either 16 for complex128 or 8 for complex 64

    Returns
    -------
    out : array_like
        array of quantized symbols

    """
    global  NMAX
    if precision == 16:
        prec_dtype = np.complex128
    elif precision == 8:
        prec_dtype = np.complex64
    else:
        raise ValueError("Precision has to be either 16 for double complex or 8 for single complex")
    Nmax = NMAX//len(symbols.flatten())//16
    L = signal.flatten().shape[0]
    sig = af.np_to_af_array(signal.flatten().astype(prec_dtype))
    sym = af.transpose(af.np_to_af_array(symbols.flatten().astype(prec_dtype)))
    tmp = af.constant(0, L, dtype=af.Dtype.c64)
    if L < Nmax:
        v, idx = af.imin(af.abs(af.broadcast(lambda x,y: x-y, sig,sym)), dim=1)
        tmp = af.transpose(sym)[idx]
    else:
        steps = L//Nmax
        rem = L%Nmax
        for i in range(steps):
            v, idx = af.imin(af.abs(af.broadcast(lambda x,y: x-y, sig[i*Nmax:(i+1)*Nmax],sym)), dim=1)
            tmp[i*Nmax:(i+1)*Nmax] = af.transpose(sym)[idx]
        v, idx = af.imin(af.abs(af.broadcast(lambda x,y: x-y, sig[steps*Nmax:],sym)), dim=1)
        tmp[steps*Nmax:] = af.transpose(sym)[idx]
    return np.array(tmp)


def norm_to_s0(sig, M):
    """
    Normalise signal to signal power calculated according to _[1]

    Parameters:
    ----------
    sig : array_like
        signal to me normalised
    M   : integer
        QAM order of the signal

    Returns
    -------
    sig_out : array_like
        normalised signal
    """
    norm = np.sqrt(cal_s0(sig, M))
    return sig / norm


def _cal_evm_blind(sig, M):
    """Blind calculation of the linear Error Vector Magnitude for an M-QAM
    signal. Does not consider Symbol errors.

    Parameters
    ----------
    sig : array_like
        input signal
    M : int
       QAM order

    Returns
    -------
    evm : float
        Error Vector Magnitude
        """
    ideal = cal_symbols_qam(M).flatten()
    Pi = norm_to_s0(ideal, M)
    Pm = norm_to_s0(sig, M)
    evm = np.mean(np.min(abs(Pm[:,np.newaxis].real-Pi.real)**2 +\
            abs(Pm[:,np.newaxis].imag-Pi.imag)**2, axis=1))
    evm /= np.mean(abs(Pi)**2)
    return np.sqrt(evm)


def cal_evm(sig, M, known=None):
    """Calculation of the linear Error Vector Magnitude for an M-QAM
    signal.

    Parameters
    ----------
    sig : array_like
        input signal
    M : int
       QAM order
    known : array_like
        the error-free symbols

    Returns
    -------
    evm : float
        Error Vector Magnitude
    """
    if known is None:
        return _cal_evm_blind(sig, M)
    else:
        Pi = norm_to_s0(known, M)
        Ps = norm_to_s0(sig, M)
        evm = np.mean(abs(Pi.real - Ps.real)**2 + \
                  abs(Pi.imag - Ps.imag)**2)
        evm /= np.mean(abs(Pi)**2)
        return np.sqrt(evm)


def cal_snr_qam(E, M):
    """Calculate the signal to noise ratio SNR according to formula given in
    _[1]

    Parameters:
    ----------
    E   : array_like
      input field
    M:  : int
      order of the QAM constallation

    Returns:
    -------
    S0/N: : float
        linear SNR estimate

    References:
    ----------
    ...[1] Gao and Tepedelenlioglu in IEEE Trans in Signal Processing Vol 53,
    pg 865 (2005).

    """
    gamma = _cal_gamma(M)
    r2 = np.mean(abs(E)**2)
    r4 = np.mean(abs(E)**4)
    S1 = 1 - 2 * r2**2 / r4 - np.sqrt(
        (2 - gamma) * (2 * r2**4 / r4**2 - r2**2 / r4))
    S2 = gamma * r2**2 / r4 - 1
    return S1 / S2


def _cal_gamma(M):
    """Calculate the gamma factor for SNR estimation."""
    A = abs(cal_symbols_qam(M)) / np.sqrt(cal_scaling_factor_qam(M))
    uniq, counts = np.unique(A, return_counts=True)
    return np.sum(uniq**4 * counts / M)


def cal_s0(E, M):
    """Calculate the signal power S0 according to formula given in
    Gao and Tepedelenlioglu in IEEE Trans in Signal Processing Vol 53,
    pg 865 (2005).

    Parameters:
    ----------
    E   : array_like
      input field
    M:  : int

    Returns:
    -------
    S0   : float
       signal power estimate
    """
    N = len(E)
    gamma = _cal_gamma(M)
    r2 = np.mean(abs(E)**2)
    r4 = np.mean(abs(E)**4)
    S1 = 1 - 2 * r2**2 / r4 - np.sqrt(
        (2 - gamma) * (2 * r2**4 / r4**2 - r2**2 / r4))
    S2 = gamma * r2**2 / r4 - 1
    # S0 = r2/(1+S2/S1) because r2=S0+N and S1/S2=S0/N
    return r2 / (1 + S2 / S1)


def cal_snr_blind_qpsk(E):
    """
    Calculates the SNR of a QPSK signal based on the variance of the constellation
    assmuing no symbol errors"""
    E4 = -E**4
    Eref = E4**(1. / 4)
    #P = np.mean(abs(Eref**2))
    P = np.mean(cabssquared(Eref))
    var = np.var(Eref)
    SNR = 10 * np.log10(P / var)
    return SNR


def cal_ser_qam(data_rx, symbol_tx, M, method="pyx"):
    """
    Calculate the symbol error rate

    Parameters
    ----------

    data_rx : array_like
        received signal
    symbols_tx : array_like
            original symbols
    M       : int
        QAM order
    method : string, option
        method to use for quantization (either `af` for arrayfire or `pyx` for cython)

    Returns
    -------
    SER : float
        Symbol error rate estimate
    """
    data_demod = quantize(data_rx, M, method=method)
    return np.count_nonzero(data_demod - symbol_tx) / len(data_rx)

def generate_bitmapping_mtx(mod):
    out_mtx = np.reshape(mod.decode(mod.gray_coded_symbols),(mod.M, mod.bits))
    symbs = mod.gray_coded_symbols
    bit_map = np.zeros([mod.bits, int(mod.M/2),2], dtype=complex)

    for bit in range(mod.bits):
        bit_map[bit,:,0] = symbs[~out_mtx[:,bit]]
        bit_map[bit,:,1] = symbs[out_mtx[:,bit]]
    return symbs, bit_map

def calc_gmi(rx_symbs, tx_symbs, M):
    rx_symbs = np.atleast_2d(rx_symbs)
    tx_symbs = np.atleast_2d(tx_symbs)
    mod = modulation.QAMModulator(M)
    symbs, bit_map = generate_bitmapping_mtx(mod)
    num_bits = int(np.log2(M))
    GMI = np.zeros(rx_symbs.shape[0])
    GMI_per_bit = np.zeros((rx_symbs.shape[0],num_bits))
    SNR_est = np.zeros(rx_symbs.shape[0])
    # For every mode present, calculate GMI based on SD-demapping
    for mode in range(rx_symbs.shape[0]):
        # GMI Calc
        rx_symbs[mode] = rx_symbs[mode] / np.sqrt(np.mean(np.abs(rx_symbs[mode]) ** 2))
        tx, rx = ber_functions.sync_and_adjust(tx_symbs[mode],rx_symbs[mode])
        snr = estimate_snr(rx, tx, symbs)[0]
        SNR_est[mode] = snr
        l_values = soft_l_value_demapper(rx,M,10**(snr/10),bit_map)
        bits = mod.decode(mod.quantize(tx)).astype(np.int)
        # GMI per bit
        for bit in range(num_bits):
            GMI_per_bit[mode, bit] = 1 - np.mean(np.log2(1+np.exp(((-1)**bits[bit::num_bits])*l_values[bit::num_bits])))
        # Sum GMI
        GMI[mode] = np.sum(GMI_per_bit[mode])
    return GMI, GMI_per_bit, SNR_est

def estimate_snr(rx_symbs, tx_symbs,symbs):
    N = symbs.shape[0]
    rx_symbs = rx_symbs / np.sqrt(np.mean(np.abs(rx_symbs)**2))
    Px = np.zeros(N)
    N0 = 0
    mus = np.zeros(N, dtype = complex)
    sigmas = np.zeros(N, dtype= complex)
    in_pow = 0
    for ind in range(N):
        sel_symbs = rx_symbs[np.bool_(tx_symbs == symbs[ind])]
        Px[ind] = sel_symbs.shape[0] / rx_symbs.shape[0]
        mus[ind] = np.mean(sel_symbs)
        sigmas[ind] = np.std(sel_symbs)

        N0 += np.abs(sigmas[ind])**2*Px[ind]
        in_pow += np.abs(mus[ind])**2*Px[ind]
    SNR = 10*np.log10(in_pow/N0)
    return SNR, Px, mus, in_pow, N0
