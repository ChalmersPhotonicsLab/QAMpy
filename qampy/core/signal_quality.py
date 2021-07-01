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
"""
Functions for calculating various signal quality metrics
"""

from __future__ import division, print_function
import numpy as np
from qampy.helpers import cabssquared
from qampy.theory import  cal_symbols_qam, cal_scaling_factor_qam
from qampy.core.equalisation.pythran_equalisation import make_decision as _decision_pyt
from qampy.core.pythran_dsp import soft_l_value_demapper, estimate_snr
from qampy.core.pythran_dsp import soft_l_value_demapper_minmax
from qampy.core.pythran_dsp import cal_mi_mc, cal_mi_mc_fast

try:
    import arrayfire as af
except ImportError:
    af = None

def _soft_l_value_demapper_af(rx_symbs, M, snr, bits_map):
    num_bits = int(np.log2(M))
    N = rx_symbs.shape[0]
    k = bits_map.shape[1]
    sig = af.np_to_af_array(rx_symbs)
    bit_mtx = af.moddims(af.np_to_af_array(bits_map), 1, num_bits, k, 2)
    tmp = af.sum(af.broadcast(lambda x,y: af.exp(-snr*af.abs(x-y)**2), bit_mtx, sig), dim=2)
    lvl = af.log(tmp[:,:,:,1]) - af.log(tmp[:,:,:,0])
    return np.array(lvl)

def make_decision(signal, symbols, method="pyt", **kwargs):
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
    if method == "pyt":
        return _decision_pyt(signal, symbols, **kwargs)
    elif method == "af":
        if af == None:
            raise RuntimeError("Arrayfire was not imported so cannot use this method for quantization")
        return _decision_af(signal, symbols, **kwargs)
    else:
        raise ValueError("method '%s' unknown has to be either 'pyx' or 'af'"%(method))


def _decision_af(signal, symbols, precision=16):
    """
    Make symbol decisions on  signal array  onto symbols. Arrayfire function.

    Parameters
    ----------
    signal : array_like
        input signal array
    symbols : array_like
        array of symbols to decide onto
    precision : int, optional
        bit precision either 16 for complex128 or 8 for complex 64

    Returns
    -------
    out : array_like
        array of decided symbols

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
        method to use for decision making (either `af` for arrayfire or `pyx` for cython)

    Returns
    -------
    SER : float
        Symbol error rate estimate
    """
    data_demod = make_decision(data_rx, M, method=method)
    return np.count_nonzero(data_demod - symbol_tx) / len(data_rx)

def generate_bitmapping_mtx(coded_symbs, coded_bits, M, dtype=np.complex128):
    num_bits = int(np.log2(M))
    out_mtx = np.reshape(coded_bits, (M, num_bits))
    bit_map = np.zeros([num_bits, int(M/2),2], dtype=dtype)
    for bit in range(num_bits):
        bit_map[bit,:,0] = coded_symbs[~out_mtx[:,bit]]
        bit_map[bit,:,1] = coded_symbs[out_mtx[:,bit]]
    return bit_map

def cal_mi(signal, symbols_tx, alphabet, N0, fast=True):
    """
    Calculate the mutual information for a given (noisy) signal array and the
    transmitted symbol array.

    Parameters
    ----------
    signal : array_like
        The signal after transmission to calculate the MI for
    symbols_tx : array_like
        The original symbols that were transmitted
    alphabet: array_like
        The symbol alphabet
    N0 : float
        The noise strength of the signal in linear units
    fast : bool
        Use fast calculation

    Returns
    -------
        mi: float
            The calculated mutual information
    """
    nmodes = signal.shape[0]
    mi = np.zeros(nmodes, dtype=np.float64)
    if fast:
        return cal_mi_mc_fast(signal, symbols_tx, alphabet, N0)
    else:
        noise = signal-symbols_tx
        return cal_mi_mc(noise, alphabet, N0)