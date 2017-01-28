from __future__ import division
import numpy as np
from scipy.special import erfc
from .utils import bin2gray, dB2lin

# All the formulas below are taken from dsplog.com

def Q_function(x):
    """The Q function is the tail probability of the standard normal distribution see _[1,2] for a definition and its relation to the erfc. In _[3] it is called the Gaussian co-error function.

    References
    ----------
    ...[1] https://en.wikipedia.org/wiki/Q-function
    ...[2] https://en.wikipedia.org/wiki/Error_function#Integral_of_error_function_with_Gaussian_density_function
    ...[3] Shafik, R. (2006). On the extended relationships among EVM, BER and SNR as performance metrics. In Conference on Electrical and Computer Engineering (p. 408). Retrieved from http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=4178493
    """
    return 0.5*erfc(x/np.sqrt(2))

def MQAM_SERvsEsN0(snr, M):
    """Calculate the symbol error rate (SER) of an M-QAM signal as a function
    of Es/N0 (Symbol energy over noise energy, given in linear units. Works
    only correctly for M > 4"""
    return 2*(1-1/np.sqrt(M))*erfc(np.sqrt(3*snr/(2*(M-1)))) -\
            (1-2/np.sqrt(M)+1/M)*erfc(np.sqrt(3*snr/(2*(M-1))))**2

def MQAM_BERvsEVM(evm_dB, M):
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
    ber = 2*(1-1/L)/np.log2(L)*Q_function(np.sqrt(3*np.log2(L)/(L**2-1)*(2/(evm*np.log2(M)))))
    return ber


def MQAM_BERvsEsN0(snr, M):
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
    ber = 2*(1-1/L)/np.log2(L)*Q_function(np.sqrt(3*np.log2(L)/(L**2-1)*(2*snr/np.log2(M))))
    return ber

def MPSK_SERvsEsN0(snr, M):
    """Calculate the symbol error rate (SER) of an M-PSK signal as a function
    of Es/N0 (Symbol energy over noise energy, given in linear units"""
    return erfc(np.sqrt(snr) * np.sin(np.pi / M))


def FourPAM_SERvsEsN0(snr):
    """Calculate the symbol error rate (SER) of an 4-PAM signal as a function
    of Es/N0 (Symbol energy over noise energy, given in linear units"""
    return 0.75 * erfc(np.sqrt(snr / 5))


def MQAMScalingFactor(M):
    """Calculate the factor for scaling the average energy to 1"""
    return 2 / 3 * (M - 1)
