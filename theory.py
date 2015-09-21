from __future__ import division
import numpy as np
from scipy.special import erfc


def MQAM_SERvsEsN0(snr, M):
    """Calculate the symbol error rate (SER) of an M-QAM signal as a function
    of Es/N0 (Symbol energy over noise energy, given in linear units. Works
    only correctly for M > 4"""
    return 2*(1-1/np.sqrt(M))*erfc(np.sqrt(3*snr/(2*(M-1)))) -\
            (1-2/np.sqrt(M)+1/M)*erfc(np.sqrt(3*snr/(2*(M-1))))**2


def MPSK_SERvsEsN0(snr, M):
    """Calculate the symbol error rate (SER) of an M-PSK signal as a function
    of Es/N0 (Symbol energy over noise energy, given in linear units"""
    return erfc(np.sqrt(snr)*np.sin(np.pi/M))


def FourPAM_SERvsEsN0(snr):
    """Calculate the symbol error rate (SER) of an 4-PAM signal as a function
    of Es/N0 (Symbol energy over noise energy, given in linear units"""
    return 0.75*erfc(np.sqrt(snr/5))


def MQAMScalingFactor(M):
    """Calculate the factor for scaling the average energy to 1"""
    return 2/3*(M-1)


def CalculateMQAMSymbols(M):
    """Genarate the symbols on the constallation diagram for M-QAM"""
    qam = np.mgrid[-(2*np.sqrt(M)/2-1):2*np.sqrt(M)/2-1:1.j*np.sqrt(M),
            -(2*np.sqrt(M)/2-1):2*np.sqrt(M)/2-1:1.j*np.sqrt(M)]
    return qam[0]+1.j*qam[1]

