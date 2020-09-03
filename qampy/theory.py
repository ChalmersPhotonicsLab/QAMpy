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

from __future__ import division
import numpy as np
from scipy.special import erfc

from qampy.core.special_fcts import q_function
from qampy.core import pythran_dsp
from qampy.core.utils import bin2gray
from qampy.helpers import dB2lin


# All the formulas below are taken from dsplog.com

def ser_vs_es_over_n0_qam(snr, M):
    """Calculate the symbol error rate (SER) of an M-QAM signal as a function
    of Es/N0 (Symbol energy over noise energy, given in linear units. Works
    only correctly for M > 4"""
    return 2*(1-1/np.sqrt(M))*erfc(np.sqrt(3*snr/(2*(M-1)))) -\
            (1-2/np.sqrt(M)+1/M)*erfc(np.sqrt(3*snr/(2*(M-1))))**2

def ber_vs_evm_qam(evm_dB, M):
    """Calculate the bit-error-rate for a M-QAM signal as a function of EVM. Taken from _[1]. Note that here we miss the square in the definition to match the plots given in the paper.

    Parameters
    ----------
    evm_dB     : array_like
          Error-Vector Magnitude in dB

    M          : integer
          order of M-QAM

    Returns
    -------

    ber        : array_like
          bit error rate in  linear units

    References
    ----------
    ...[1] Shafik, R. (2006). On the extended relationships among EVM, BER and SNR as performance metrics. In Conference on Electrical and Computer Engineering (p. 408). Retrieved from http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=4178493

    Note
    ----
    The EVM in dB is defined as $EVM(dB) = 10 \log_{10}{P_{error}/P_{reference}}$ (see e.g. Wikipedia) this is different to the percentage EVM which is defined as the RMS value i.e. $EVM(\%)=\sqrt{1/N \sum^N_{j=1} [(I_j-I_{j,0})^2 + (Q_j -Q_{j,0})^2]} $ so there is a difference of a power 2. So to get the same plot as in _[1] we need to enter the log of EVM^2
    """
    L = np.sqrt(M)
    evm = dB2lin(evm_dB)
    ber = 2 * (1-1/L) / np.log2(L) * q_function(np.sqrt(3 * np.log2(L) / (L ** 2 - 1) * (2 / (evm * np.log2(M)))))
    return ber


def ber_vs_es_over_n0_qam(snr, M):
    """
    Bit-error-rate vs signal to noise ratio after formula in _[1].

    Parameters
    ----------

    snr   : array_like
        Signal to noise ratio in linear units

    M     : integer
        Order of M-QAM 

    Returns
    -------

    ber   : array_like
        theoretical bit-error-rate

    References
    ----------
    ...[3] Shafik, R. (2006). On the extended relationships among EVM, BER and SNR as performance metrics. In Conference on Electrical and Computer Engineering (p. 408). Retrieved from http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=4178493
    """
    L = np.sqrt(M)
    ber = 2 * (1-1/L) / np.log2(L) * q_function(np.sqrt(3 * np.log2(L) / (L ** 2 - 1) * (2 * snr / np.log2(M))))
    return ber

def ser_vs_es_over_n0_psk(snr, M):
    """Calculate the symbol error rate (SER) of an M-PSK signal as a function
    of Es/N0 (Symbol energy over noise energy, given in linear units"""
    return erfc(np.sqrt(snr) * np.sin(np.pi / M))


def ser_vs_es_over_n0_4pam(snr):
    """Calculate the symbol error rate (SER) of an 4-PAM signal as a function
    of Es/N0 (Symbol energy over noise energy, given in linear units"""
    return 0.75 * erfc(np.sqrt(snr / 5))


def cal_symbols_qam(M):
    """
    Generate the symbols on the constellation diagram for M-QAM
    """
    if np.log2(M) % 2 > 0.5:
        return cal_symbols_cross_qam(M)
    else:
        return cal_symbols_square_qam(M)

def cal_scaling_factor_qam(M):
    """
    Calculate the scaling factor for normalising MQAM symbols to 1 average Power
    """
    bits = np.log2(M)
    if not bits % 2:
        scale = 2 / 3 * (M - 1)
    else:
        symbols = cal_symbols_qam(M)
        scale = (abs(symbols)**2).mean()
    return scale

def cal_symbols_square_qam(M):
    """
    Generate the symbols on the constellation diagram for square M-QAM
    """
    qam = np.mgrid[-(2 * np.sqrt(M) / 2 - 1):2 * np.sqrt(
        M) / 2 - 1:1.j * np.sqrt(M), -(2 * np.sqrt(M) / 2 - 1):2 * np.sqrt(M) /
                   2 - 1:1.j * np.sqrt(M)]
    return (qam[0] + 1.j * qam[1]).flatten()


def cal_symbols_cross_qam(M):
    """
    Generate the symbols on the constellation diagram for non-square (cross) M-QAM
    """
    N = (np.log2(M) - 1) / 2
    s = 2**(N - 1)
    rect = np.mgrid[-(2**(N + 1) - 1):2**(N + 1) - 1:1.j * 2**(N + 1), -(
        2**N - 1):2**N - 1:1.j * 2**N]
    qam = rect[0] + 1.j * rect[1]
    idx1 = np.where((abs(qam.real) > 3 * s) & (abs(qam.imag) > s))
    idx2 = np.where((abs(qam.real) > 3 * s) & (abs(qam.imag) <= s))
    qam[idx1] = np.sign(qam[idx1].real) * (
        abs(qam[idx1].real) - 2 * s) + 1.j * (np.sign(qam[idx1].imag) *
                                              (4 * s - abs(qam[idx1].imag)))
    qam[idx2] = np.sign(qam[idx2].real) * (
        4 * s - abs(qam[idx2].real)) + 1.j * (np.sign(qam[idx2].imag) *
                                              (abs(qam[idx2].imag) + 2 * s))
    return qam.flatten()


def gray_code_qam(M):
    """
    Generate gray code map for M-QAM constellations
    """
    Nbits = int(np.log2(M))
    if Nbits % 2 == 0:
        N = Nbits // 2
        idx = np.mgrid[0:2**N:1, 0:2**N:1]
    else:
        N = (Nbits - 1) // 2
        idx = np.mgrid[0:2**(N + 1):1, 0:2**N:1]
    gidx = bin2gray(idx)
    return ((gidx[0] << N) | gidx[1]).flatten()

def cal_ps_probablts(symbols, nu):
    """
    Calculate probabilities for probabilistic constellation shaping
    of symbols

    Parameters
    ----------
    symbols : array_like
        coded input symbols
    nu : float
        shaping factor

    Returns
    -------
    symbs : array_like
        real part of the coded symbols
    px  : array_like
        corresponding probabilities
    """

    symbs = np.unique(symbols.real)
    px = np.zeros(symbs.shape[0])
    div_factor = 0
    for ind in range(px.shape[0]):
        div_factor += np.exp(-nu * np.abs(symbs[ind]) ** 2)
    for ind in range(px.shape[0]):
        px[ind] = np.exp(-nu * np.abs(symbs[ind]) ** 2) / div_factor
    return symbs, px

def generate_ps_symbols(N, symbs, px, normalize=True):
    """
    Generate a set of probabilistically shaped symbols

    Parameters
    ----------
    N : int
        length of the symbol array
    symbs : array_like
        the real part of the symbols to pick
    px : array_like
        the corresponding probabilities
    normalize : bool,optional
        whether to normalise the output

    Returns
    -------
    mod_symbols: array_like
        set of probabilistically shaped symbols
    """
    mod_symbs = np.random.choice(symbs, N, p=px) + \
                1j * np.random.choice(symbs, N, p=px)
    if normalize:
        mod_symbs = utils.normalise_and_center(mod_symbs)
    return mod_symbs

def hybrid_qam_ber_vs_esn0(snr, pr, fr, M1, M2):
    """
    Calculate  bit error rate as function of SNR for time-domain hybrid QAM (according to _[1]).

    Parameters
    ----------
    snr : float
        signal ot noise ratio in dB
    pr  : float
        ratio of average signal power modulated with first QAM order over avg power of second QAM order
    fr  : float
        format ratio, fraction of symbols with modulation M2 of the overall frame.
    M1 : integer
        First QAM format
    M2 : integer
        Second QAM format

    Returns
    -------
    ber : float
        theoretical bit error rate for the given TD hybrid QAM

    References
    ----------
    ..[1] Curri et al. "Time-division hybrid modulation formats: Tx operation strategies and countermeasures to
        nonlinear propagation", OFC 2014
    """
    snr = 10**(snr/10)
    bps1 = np.log2(M1)
    bps2 = np.log2(M2)
    return 1/((1-fr)*bps1+fr*bps2)*((1-fr)*bps1*theory.MQAM_BERvsEsN0(snr/((1-fr)+fr*pr), M1) + fr*bps2*theory.MQAM_BERvsEsN0(pr*snr/((1-fr)+fr*pr), M2))

def cal_gmi(M, snr, N=10**3):
    """
    Calculate the soft-decision GMI for a given modulation format
    based on a Monte-Carlo method. Assumes a gray-coded square QAM format.

    Parameters
    ----------
    M : int
        QAM order
    snr : float or array_like
        Signal-to-noise-ratio in dB where to calculate the gmi
    N : int, optional
        Number of noise realisations for the Monte-Carlo simulation

    Returns
    -------
    GMI
    """
    from qampy.signals import SignalQAMGrayCoded
    from qampy.core.pythran_dsp import cal_gmi_mc
    snr = np.atleast_1d(snr)
    s = SignalQAMGrayCoded(M, 1000, nmodes=1)
    btx = s._bitmap_mtx
    syms = s.coded_symbols
    snr_lin = 10**(snr/10)
    gmi = np.zeros_like(snr_lin)
    for i in range(snr.size):
        gmi[i] = cal_gmi_mc(syms, snr_lin[i], N, btx)
    return gmi

def sim_mi_mc(symbols, snr, N):
    """
    Perform Monte-Carlo simulations of AWGN Mutual Information.

    Parameters
    ----------
    symbols : array_like
        symbol alphabet to calculate the MI for
    snr : float
        The signal to noise ratio that should be simulated (in dB)
    N : int
        The size of the noise array to average over

    Returns
    -------
        mi : float
            Mutual information of the symbol alphabet at the given SNR
    """
    symbols /= np.sqrt(np.mean(abs(symbols)**2))
    N0 = 10**(-snr/10)
    sigma = np.sqrt(N0/2)
    noise = (np.random.randn(N) + 1j*np.random.randn(N))*sigma
    return pythran_dsp.cal_mi_mc(noise, symbols, N0)