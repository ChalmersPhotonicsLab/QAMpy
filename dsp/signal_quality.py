from __future__ import division, print_function
import numpy as np
from . import  mathfcts
from . theory import CalculateMQAMSymbols, MQAMScalingFactor

def normalise_sig(sig, M):
    """Normalise signal to average power"""
    norm = np.sqrt(calS0(sig, M))
    return 1/norm, sig/norm

def cal_blind_evm(sig, M):
    """Blind calculation of the linear Error Vector Magnitude for an M-QAM
    signal. Does not consider Symbol errors."""
    ideal = CalculateMQAMSymbols(M).flatten()
    Ai, Pi = normalise_sig(ideal, M)
    Am, Pm = normalise_sig(sig, M)
    evm = np.mean(np.min(abs(Pm[:,np.newaxis].real-Pi.real)**2 +\
            abs(Pm[:,np.newaxis].imag-Pi.imag)**2, axis=1))
    evm /= np.mean(abs(Pi)**2)
    return np.sqrt(evm)

def cal_evm_known_data(sig, ideal, M):
    """Calculation of the linear Error Vector Magnitude for a signal against
    the ideal signal"""
    Ai, Pi = normalise_sig(ideal, M)
    As, Ps = normalise_sig(sig, M)
    evm = np.mean(abs(Pi.real - Ps.real)**2 + \
                  abs(Pi.imag - Ps.imag)**2)
    evm /= np.mean(abs(Pi)**2)
    return np.sqrt(evm)

def cal_SNR_QAM(E, M):
    """Calculate the signal to noise ratio SNR according to formula given in
    Gao and Tepedelenlioglu in IEEE Trans in Signal Processing Vol 53,
    pg 865 (2005).

    Parameters:
        E:      input field
        M:      order of the QAM constallation

    Returns:
        S0/N: SNR estimate
    """
    gamma = _cal_gamma(M)
    r2 = np.mean(abs(E)**2)
    r4 = np.mean(abs(E)**4)
    S1 = 1-2*r2**2/r4-np.sqrt((2-gamma)*(2*r2**4/r4**2-r2**2/r4))
    S2 = gamma*r2**2/r4-1
    return S1/S2

def _cal_gamma(M):
    """Calculate the gamma factor for SNR estimation."""
    A = abs(CalculateMQAMSymbols(M))/np.sqrt(MQAMScalingFactor(M))
    uniq, counts = np.unique(A, return_counts=True)
    return np.sum(uniq**4*counts/M)

def cal_Q_16QAM(E):
    """Calculate the signal to noise ratio SNR according to formula given in
    Gao and Tepedelenlioglu in IEEE Trans in Signal Processing Vol 53,
    pg 865 (2005).

    Parameters:
        E:      input field

    Returns:
        S0/N: SNR estimate
    """
    return cal_SNR_QAM(E, 16)

def calS0(E, M):
    """Calculate the signal power S0 according to formula given in
    Gao and Tepedelenlioglu in IEEE Trans in Signal Processing Vol 53,
    pg 865 (2005).

    Parameters:
        E:      input field
        M:      order of the QAM constellation

    Returns:
        S0: signal power estimate
    """
    N = len(E)
    gamma = _cal_gamma(M)
    r2 = np.mean(abs(E)**2)
    r4 = np.mean(abs(E)**4)
    S1 = 1-2*r2**2/r4-np.sqrt((2-gamma)*(2*r2**4/r4**2-r2**2/r4))
    S2 = gamma*r2**2/r4-1
    # S0 = r2/(1+S2/S1) because r2=S0+N and S1/S2=S0/N
    return r2/(1+S2/S1)

def SNR_QPSK_blind(E):
    ''' calculates the SNR from the constellation assmuing no symbol errors'''
    E4 = -E**4
    Eref = E4**(1./4)
    #P = np.mean(abs(Eref**2))
    P = np.mean(mathfcts.cabssquared(Eref))
    var = np.var(Eref)
    SNR = 10*np.log10(P/var)
    return SNR

def cal_ser_QAM(data_rx, data_tx, M):
    """
    Calculate the symbol error rate

    Parameters
    ----------

    data_rx : array_like
            received signal
    data_tx : array_like
            original signal
    M       : int
            QAM order
    """
    data_demod = QAMdemod(M, data_rx)[0]
    return np.count_nonzero(data_demod-data_tx)/len(data_rx)

