import numpy as np
from theory import CalculateMQAMSymbols

def normalise_sig(sig):
    """Normalise signal to average power"""
    norm = np.sqrt(np.mean(abs(sig)**2))
    return 1/norm, sig/norm

def cal_blind_evm(sig, M):
    """Blind calculation of the linear Error Vector Magnitude for an M-QAM
    signal. Does not consider Symbol errors."""
    ideal = CalculateMQAMSymbols(M).flatten()
    Ai, Pi = normalise_sig(ideal)
    Am, Pm = normalise_sig(sig)
    evm = np.mean(np.min(abs(Pm[:,np.newaxis].real-Pi.real)**2 +\
            abs(Pm[:,np.newaxis].imag-Pi.imag)**2, axis=1))
    evm /= np.mean(abs(Pi)**2)
    return np.sqrt(evm)

def cal_evm_known_data(sig, ideal):
    """Calculation of the linear Error Vector Magnitude for a signal against
    the ideal signal"""
    Ai, Pi = normalize_sig(ideal)
    As, Ps = normalize_sig(sig)
    evm = np.mean(abs(Pi.real - Ps.real)**2 + \
                  abs(Pi.imag - Ps.imag)**2)
    evm /= np.mean(abs(Pi)**2)
    return np.sqrt(evm)

def cal_Q_16QAM(E, gamma=1.32):
    """Calculate the signal power S0 according to formula given in
    Gao and Tepedelenlioglu in IEEE Trans in Signal Processing Vol 53,
    pg 865 (2005).

    Parameters:
        E:      input field
        gamma:  constant dependent on modulation format [default=1.32 for 16 QAM]

    Returns:
        S0/N: OSNR estimate
    """
    N = len(E)
    r2 = np.sum(abs(E)**2)/N
    r4 = np.sum(abs(E)**4)/N
    S1 = 1-2*r2**2/r4-np.sqrt((2-gamma)*(2*r2**4/r4**2-r2**2/r4))
    S2 = gamma*r2**2/r4-1
    return S1/S2

def calS0(E, gamma):
    N = len(E)
    r2 = np.sum(abs(E)**2)/N
    r4 = np.sum(abs(E)**4)/N
    S1 = 1-2*r2**2/r4-np.sqrt((2-gamma)*(2*r2**4/r4**2-r2**2/r4))
    S2 = gamma*r2**2/r4-1
    return r2/(1+S2/S1)
