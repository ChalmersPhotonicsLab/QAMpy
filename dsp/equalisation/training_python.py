import numpy as np


def partition_signal(signal, partitions, codebook):
    """
    Partition a signal according to their power

    Parameters
    ----------
    signal : array_like
        input signal to partition

    partitions : array_like
        partition vector defining the boundaries between the different blocks

    code   : array_like
        the values to partition tolerance

    Returns
    -------
    quanta : array_like
        partitioned quanta
    """
    quanta = []
    for datum in signal:
        index = 0
        while index < len(partitions) and datum > partitions[index]:
            index += 1
        quanta.append(codebook[index])
    return quanta

def partition_value(signal, partitions, codebook):
    """
    Partition a value according to their power

    Parameters
    ----------
    signal : float
        input value to partition

    partitions : array_like
        partition vector defining the boundaries between the different blocks

    code   : array_like
        the values to partition tolerance

    Returns
    -------
    quanta : float
        partitioned quanta
    """
    index = 0
    while index < len(partitions) and signal > partitions[index]:
        index += 1
    quanta = codebook[index]
    return quanta

def FS_MCMA_training(E, TrSyms, Ntaps, os, mu, wx):
    """
    Training of the Modified CMA algorithm to determine the equaliser taps. Details in _[1]. Assumes a normalised signal.

    Parameters
    ----------
    E       : array_like
       dual polarisation signal field

    TrSyms : int
       number of symbols to use needs to be less than len(Ex)

    Ntaps   : int
       number of equaliser taps

    os      : int
       oversampling factor

    mu   : float
       step size parameter

    wx     : array_like
       initial equaliser taps
    
    Returns
    -------

    err       : array_like
       estimation error for x and y polarisation

    wx    : array_like
       equaliser taps

    Notes
    -----
    ..[1] Oh, K. N., & Chin, Y. O. (1995). Modified constant modulus algorithm: blind equalization and carrier phase recovery algorithm. Proceedings IEEE International Conference on Communications ICC ’95, 1, 498–502. http://doi.org/10.1109/ICC.1995.525219
    """
    err = np.zeros(TrSyms, dtype=np.complex)
    for i in range(TrSyms):
        X = E[:, i * os:i * os + Ntaps]
        Xest = np.sum(wx * X)
        err[i] = (np.abs(Xest.real)**2 -0.5) * Xest.real + (np.abs(Xest.imag)**2 - 0.5)*Xest.imag*1.j
        wx -= mu * err[i] * np.conj(X)
    return err, wx

def FS_CMA_training(E, TrSyms, Ntaps, os, mu, wx):
    """
    Training of the CMA algorithm to determine the equaliser taps.

    Parameters
    ----------
    E      : array_like
       dual polarisation signal field

    TrSyms : int
       number of symbols to use for training needs to be less than len(Ex)

    Ntaps   : int
       number of equaliser taps

    os      : int
       oversampling factor

    mu      : float
       step size parameter

    wx     : array_like
       initial equaliser taps

    Returns
    -------

    err       : array_like
       CMA estimation error for x and y polarisation

    wx        : array_like
       equaliser taps
    """
    err = np.zeros(TrSyms, dtype=np.float)
    for i in range(0, TrSyms):
        X = E[:, i * os:i * os + Ntaps]
        Xest = np.sum(wx * X)
        err[i] = abs(Xest) - 1
        wx -= mu * err[i] * Xest * np.conj(X)
    return err, wx

def FS_MRDE_training(E, TrRDE, Ntaps, os, muRDE, wx, part, code):
    """
    Training of the Modified RDE algorithm to determine the equaliser taps. Details in _[1]. Assumes a normalised signal.

    Parameters
    ----------
    E       : array_like
       dual polarisation signal field

    TrRDE : int
       number of symbols to use for training the radius directed equaliser, needs to be less than len(Ex)

    Ntaps   : int
       number of equaliser taps

    os      : int
       oversampling factor

    muRDE   : float
       step size parameter

    wx     : array_like
       initial equaliser taps

    part    : array_like (complex)
       partitioning vector defining the boundaries between the different real and imaginary QAM constellation "rings"

    code    : array_like
       code vector defining the signal powers for the different QAM constellation rings

    Returns
    -------

    err       : array_like
       RDE estimation error for x and y polarisation

    wx    : array_like
       equaliser taps

    Notes
    -----
    ..[1] Oh, K. N., & Chin, Y. O. (1995). Modified constant modulus algorithm: blind equalization and carrier phase recovery algorithm. Proceedings IEEE International Conference on Communications ICC ’95, 1, 498–502. http://doi.org/10.1109/ICC.1995.525219
    """
    err = np.zeros(TrRDE, dtype=np.complex)
    for i in range(TrRDE):
        X = E[:, i * os:i * os + Ntaps]
        Xest = np.sum(wx * X)
        Ssq = Xest.real**2 + 1.j * Xest.imag**2
        R = partition_value(Ssq.real, part.real, code.real) + partition_value(Ssq.imag, part.imag, code.imag)*1.j
        err[i] = (Ssq.real - R.real)*Xest.real + (Ssq.imag - R.imag)*1.j*Xest.imag
        wx -= muRDE * err[i] * np.conj(X)
    return err, wx

def FS_RDE_training(E, TrRDE, Ntaps, os, muRDE, wx, part, code):
    """
    Training of the RDE algorithm to determine the equaliser taps.

    Parameters
    ----------
    E       : array_like
       dual polarisation signal field

    TrCMA : int
       number of symbols to use for training the initial CMA needs to be less than len(Ex)

    TrRDE : int
       number of symbols to use for training the radius directed equaliser, needs to be less than len(Ex)

    Ntaps   : int
       number of equaliser taps

    os      : int
       oversampling factor

    muRDE   : float
       step size parameter

    wx     : array_like
       initial equaliser taps

    part    : array_like
       partitioning vector defining the boundaries between the different QAM constellation rings

    code    : array_like
       code vector defining the signal powers for the different QAM constellation rings

    Returns
    -------

    err       : array_like
       CMA estimation error for x and y polarisation

    wx    : array_like
       equaliser taps
    """
    err = np.zeros(TrRDE, dtype=np.float)
    for i in range(TrRDE):
        X = E[:, i * os:i * os + Ntaps]
        Xest = np.sum(wx * X)
        Ssq = abs(Xest)**2
        S_DD = partition_value(Ssq, part, code)
        err[i] = Ssq - S_DD #- Ssq
        wx -= muRDE * err[i] * Xest * np.conj(X)
    return err, wx
