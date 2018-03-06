import numpy as np
import numba
import locale
locale.setlocale(locale.LC_NUMERIC, 'C')

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

@numba.jit(nopython=True)
def adapt_step(mu, err_p, err):
    if err.real*err_p.real > 0 and err.imag*err_p.imag >  0:
        lm = 0
    else:
        lm = 1
    mu = mu/(1+lm*mu*(err.real*err.real + err.imag*err.imag))
    return mu

@numba.jit(nopython=True)
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

@numba.jit(nopython=True)
def sum(x):
    y = x.flatten()
    __sum = 0.j
    for i in range(y.size):
        __sum += y[i]
    return __sum


@numba.jit(nopython=True)
def train_eq(E, TrSyms, os, mu, wx, errfct  adaptive=False):
    """
    Training of the equaliser

    Parameters
    ----------
    E       : array_like
        dual polarisation signal field
    TrSyms : int
        number of symbols to use for training the radius directed equaliser, needs to be less than len(Ex)
    os      : int
        oversampling factor
    mu   : float
        step size parameter
    wx     : array_like
        initial equaliser taps
    errfct : fct
        the equaliser errorfct

    Returns
    -------
    err       : array_like
        estimation error for x and y polarisation
    wx    : array_like
        equaliser taps
    """
    Ntaps = wx.shape[1]
    err = np.zeros(TrSyms, dtype=np.complex128)
    for i in range(TrSyms):
        X = E[:, i * os:i * os + Ntaps]
        Xest = sum(wx * X)
        err[i] = errfct(Xest)
        wx -= mu * err[i] * np.conj(X)
        if adaptive and i > 0:
            mu = adapt_step(mu, err[i], err[i-1])
    return err, wx

def ErrorFctCMA(R):
    """
    Create Constant Modulus Algorithm (CMA) error

    Parameters
    ----------
    R : float
        radius for the CMA algorithm

    Returns
    -------
    train_eq: function
        equaliser training function

    """
    @numba.jit(nopython=True)
    def cma_fct(Xest):
        return (abs(Xest)**2 - R)*Xest
    return cma_fct

def ErrorFctMCMA(R):
    """
    Create Modified Constant Modulus Algorithm error after _[1]

    Parameters
    ----------
    R : complex
        Complex radius for the CMA algorithm

    Returns
    -------
    train_eq: function
        equaliser training function

    References
    -----
    ..[1] Oh, K. N., & Chin, Y. O. (1995). Modified constant modulus algorithm: blind equalization and carrier phase recovery algorithm. Proceedings IEEE International Conference on Communications ICC ’95, 1, 498–502. http://doi.org/10.1109/ICC.1995.525219
    """
    @numba.jit(nopython=True)
    def mcma_fct(Xest):
        return (Xest.real**2 - R.real) * Xest.real + (Xest.imag**2 - R.imag)*Xest.imag*1.j
    return mcma_fct

def ErrorFctRDE(partition, codebook):
    """
    Create Radius Directed Error after _[1]

    Parameters
    ----------
    partition    : array_like, float
       partitioning vector defining the boundaries between the different QAM constellation rings

    codebook    : array_like, float
       code vector defining the signal powers for the different QAM constellation rings

    Returns
    -------
    train_eq: function
        equaliser training function

    References
    -----
    """
    @numba.jit(nopython=True)
    def rde_fct(Xest):
        Ssq = abs(Xest)**2
        S_DD = partition_value(Ssq, part, code)
        return (Ssq - S_DD)*Xest
    return rde_fct

def ErrorFctMRDE(partition, codebook):
    """
    Create Modified Radius Directed Error after _[1]

    Parameters
    ----------
    partition    : array_like, complex
       partitioning vector defining the boundaries between the different QAM constellation rings

    codebook    : array_like, complex
       code vector defining the signal powers for the different QAM constellation rings

    Returns
    -------
    train_eq: function
        equaliser training function

    References
    -----
    """
    @numba.jit(nopython=True)
    def mrde_fct(Xest):
        Ssq = Xest.real**2 + 1.j * Xest.imag**2
        R = partition_value(Ssq.real, partion.real, codebook.real) + partition_value(Ssq.imag, partition.imag, codebook.imag)*1.j
        return (Ssq.real - R.real)*Xest.real + (Ssq.imag - R.imag)*1.j*Xest.imag
    return mrde_fct

def ErrorFctSBD(symbols):
    """
    Symbol Based Decision (SBD) training function after _[1]. This is a DD error function. This does not implement the neighbor weigthing detailed further in _[1].

    Parameters
    ----------
    symbols    : array_like, complex
        the symbols of the QAM format being recovered

    Returns
    -------
    train_eq: function
        equaliser training function

    References
    -----
    ...[1] Filho, M., Silva, M. T. M., & Miranda, M. D. (2008). A FAMILY OF ALGORITHMS FOR BLIND EQUALIZATION OF QAM SIGNALS. In 2011 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 6–9).
    """
    @numba.jit(nopython=True)
    def sbd_fct(Xest):
        R = symbols[np.argmin(np.abs(Xest-symbols))]
        return (Xest.real - R.real)*abs(R.real) + (Xest.imag - R.imag)*1.j*abs(R.imag)
    return sbd_fct

def ErrorFctMDDMA(symbols):
    """
    Modified Decision Directed Modulus Algorithm (MDDMA) error after _[1].
    This is a DD error function.

    Parameters
    ----------
    symbols    : array_like, complex
        the symbols of the QAM format being recovered

    Returns
    -------
    train_eq: function
        equaliser training function

    References
    -----
    ...[1] Fernandes, C. A. R., Favier, G., & Mota, J. C. M. (2007). Decision directed adaptive blind equalization based on the constant modulus algorithm. Signal, Image and Video Processing, 1(4), 333–346. http://doi.org/10.1007/s11760-007-0027-2
    """
    @numba.jit(nopython=True)
    def mddma_fct(Xest):
        R = symbols[np.argmin(np.abs(Xest-symbols))]
        return (Xest.real**2 - R.real**2)*Xest.real + (Xest.imag**2 - R.imag**2)*1.j*Xest.imag
    return mddma_fct

def ErrorFctDD(symbols):
    """
    Decision Directed error

    Parameters
    ----------
    R : complex
        Complex radius for the CMA algorithm

    Returns
    -------
    train_eq: function
        equaliser training function

    References
    -----
    """
    @numba.jit(nopython=True)
    def dd_fct(Xest):
        R = symbols[np.argmin(np.abs(Xest-symbols))]
        return Xest - R
    return dd_fct

def ErrorFctCME(R, d, beta):
    """
    Constellation matched error algorithm after _[1]

    Parameters
    ----------
    R : float
        radius for the CMA algorithm
    d : float
        distance between constellation points along one dimension
    beta: float
        ratio between CMA and sin part of the algorithm

    Returns
    -------
    train_eq: function
        equaliser training function

    References
    -----
    ...[1] He, L., Amin, M. G., Reed, C., & Malkemes, R. C. (2004). A Hybrid Adaptive Blind Equalization Algorithm for QAM Signals in Wireless Communications, 52(7), 2058–2069.
    """
    @numba.jit(nopython=True)
    def cme_fct(Xest):
        err = (abs(Xest)**2 - R)*Xest
        err +=  beta * np.pi/(2*d) * (np.sin(Xest.real*np.pi/d) + 1.j * np.sin(Xest.imag*np.pi/d))
        return err
    return cme_fct

def ErrorFctSCA(R):
    """
    Create Square Contour Algorithm error after _[1]

    Parameters
    ----------
    R : float
        radius for the algorithm

    Returns
    -------
    train_eq: function
        equaliser training function

    References
    -----
    ...[1] Sheikh, S. A., & Fan, P. (2008). New blind equalization techniques based on improved square contour algorithm ✩, 18, 680–693. http://doi.org/10.1016/j.qampy.2007.09.001
    """
    @numba.jit(nopython=True)
    def sca_fct(Xest):
        if abs(Xest.real) >= abs(Xest.imag):
            A = 1
            if abs(Xest.real) == abs(Xest.imag):
                B = 1
            else:
                B = 0
        else:
            A = 0
            B = 1
        return 4*Xest.real*(4*Xest.real**2 - 4*R**2)*A + 1.j*4*Xest.imag*(4*Xest.imag**2 - 4*R**2)*B
    return sca_fct
