from __future__ import division
import numpy as np
from scipy.special import erfc
from .mathfcts import bin2gray

# All the formulas below are taken from dsplog.com


def MQAM_SERvsEsN0(snr, M):
    """Calculate the symbol error rate (SER) of an M-QAM signal as a function
    of Es/N0 (Symbol energy over noise energy, given in linear units. Works
    only correctly for M > 4"""
    return 2*(1-1/np.sqrt(M))*erfc(np.sqrt(3*snr/(2*(M-1)))) -\
            (1-2/np.sqrt(M)+1/M)*erfc(np.sqrt(3*snr/(2*(M-1))))**2


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


def CalculateMQAMSymbols(M):
    """
    Generate the symbols on the constellation diagram for M-QAM
    """
    if np.log2(M)%2 > 0.5:
        return CalculateCrossQAMSymbols(M)
    else:
        return CalculateSquareQAMSymbols(M)

def CalculateSquareQAMSymbols(M):
    """
    Generate the symbols on the constellation diagram for square M-QAM
    """
    qam = np.mgrid[-(2 * np.sqrt(M) / 2 - 1):2 * np.sqrt(
        M) / 2 - 1:1.j * np.sqrt(M), -(2 * np.sqrt(M) / 2 - 1):2 * np.sqrt(M) /
                   2 - 1:1.j * np.sqrt(M)]
    return (qam[0] + 1.j * qam[1]).flatten()

def CalculateCrossQAMSymbols(M):
    """
    Generate the symbols on the constellation diagram for non-square (cross) M-QAM
    """
    N = (np.log2(M)-1)/2
    s = 2**(N-1)
    rect = np.mgrid[
        -(2**(N+1) - 1) : 2**(N+1) - 1: 1.j*2**(N+1),
        -(2**N - 1) : 2**N - 1: 1.j*2**N
    ]
    qam = rect[0] + 1.j*rect[1]
    idx1 = np.where((abs(qam.real) > 3*s)&(abs(qam.imag) > s))
    idx2 = np.where((abs(qam.real) > 3*s)&(abs(qam.imag) <= s))
    qam[idx1] = np.sign(qam[idx1].real)*(abs(qam[idx1].real)-2*s) + 1.j*(np.sign(qam[idx1].imag)*(4*s-abs(qam[idx1].imag)))
    qam[idx2] = np.sign(qam[idx2].real)*(4*s-abs(qam[idx2].real)) + 1.j*(np.sign(qam[idx2].imag)*(abs(qam[idx2].imag)+2*s))
    return qam.flatten()

def gray_code_for_qam(M):
    """
    Generate gray code map for M-QAM constellations
    """
    Nbits = int(np.log2(M))
    if Nbits%2 == 0:
        N = Nbits//2
        idx = np.mgrid[
            0 : 2**N : 1,
            0 : 2**N : 1
        ]
    else:
        N = (Nbits - 1)//2 
        idx = np.mgrid[
            0 : 2**(N+1): 1,
            0 : 2**N: 1
        ]
    gidx = bin2gray(idx)
    return (gidx[0] << N)| gidx[1] 
