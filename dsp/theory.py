from __future__ import division
import numpy as np
from scipy.special import erfc
from .utils import bin2gray, dB2lin

# All the formulas below are taken from dsplog.com

def q_function(x):
    """The Q function is the tail probability of the standard normal distribution see _[1,2] for a definition and its relation to the erfc. In _[3] it is called the Gaussian co-error function.

    References
    ----------
    ...[1] https://en.wikipedia.org/wiki/Q-function
    ...[2] https://en.wikipedia.org/wiki/Error_function#Integral_of_error_function_with_Gaussian_density_function
    ...[3] Shafik, R. (2006). On the extended relationships among EVM, BER and SNR as performance metrics. In Conference on Electrical and Computer Engineering (p. 408). Retrieved from http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=4178493
    """
    return 0.5*erfc(x/np.sqrt(2))

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
    ber = 2*(1-1/L)/np.log2(L)*q_function(np.sqrt(3*np.log2(L)/(L**2-1)*(2/(evm*np.log2(M)))))
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
    ber = 2*(1-1/L)/np.log2(L)*q_function(np.sqrt(3*np.log2(L)/(L**2-1)*(2*snr/np.log2(M))))
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

def prob_shaping_propabilites(symbols, nu):
        symbs = np.unique(symbols.real)
        px = np.zeros(symbs.shape[0])
        div_factor = 0
        for ind in range(px.shape[0]):
            div_factor += np.exp(-nu * np.abs(symbs[ind]) ** 2)
        for ind in range(px.shape[0]):
            px[ind] = np.exp(-nu * np.abs(symbs[ind]) ** 2) / div_factor
        return symbs, px

def generate_shaped_symbols(N, symbs, px, normalize=True):
        mod_symbs = np.random.choice(symbs, N, p=px) + \
                    1j * np.random.choice(symbs, N, p=px)
        if normalize:
            mod_symbs = utils.normalise_and_center(mod_symbs)
        return mod_symbs
