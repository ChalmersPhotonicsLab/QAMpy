import numpy as np

def partition_value_complex(signal, partitions, codebook):
# without this function compilation on windows fails
    L = partitions.shape[0]
    r = 0+0j
    for i in range(L):
        if signal.real <= partitions[i].real:
            r = codebook[i].real + 1j*r.imag
        if signal.imag <= partitions[i].imag:
            r = r.real + 1j*codebook[i].imag
    return r

def partition_value(signal, partitions, codebook):
    L = partitions.shape[0]
    index = 0
    while index < L and signal > partitions[index]:
        index += 1
    return codebook[index]

def adapt_step(mu, err_p, err):
    if err.real*err_p.real > 0 and err.imag*err_p.imag > 0:
        return mu
    else:
        return mu/(1+mu*(err.real*err.real + err.imag*err.imag))

def apply_filter(E, wx):
    pols = E.shape[0]
    Ntaps = wx.shape[1]
    Xest = 0+0j
    for k in range(pols):
        for i in range(Ntaps):
            Xest += E[k, i]*np.conj(wx[k,i])
    return Xest

#pythran export apply_filter_to_signal(complex128[][], int, complex128[][][])
#pythran export apply_filter_to_signal(complex64[][], int, complex64[][][])
def apply_filter_to_signal(E, os, wx):
    """
    Apply a filter to a signal 
    
    Parameters
    ----------
    E : array_like
        Signal to filter 
    os : int
        oversampling factor
    wx : array_like
        filter taps

    Returns
    -------
    output : array_like
        filtered and downsampled signal
    """
    Ntaps = wx.shape[2]
    L = E.shape[1]
    modes = E.shape[0]
    N = (L-Ntaps+os)//os
    output  = np.zeros((modes, N), dtype=E.dtype)
    #omp parallel for
    for idx in range(modes*N): # manual collapse of loop
        j = idx//N
        i = idx%N
        Xest = apply_filter(E[:, i*os:i*os+Ntaps], wx[j])
        output[j, i] = Xest
    return output

#pythran export mcma_equaliser(complex128[][], int, int, int, float64, complex128[][][],
    # bool, complex128)
#pythran export mcma_equaliser(complex64[][], int, int, int, float32, complex64[][][],
# bool, complex64)
def mcma_equaliser(E, TrSyms, Niter, os, mu, wx, adaptive, R):
    """
    Train filter taps using the modified constant modulus algorithm (MCMA) (see equalisation.py for references)
    
    Parameters
    ----------
    E : array_like
        input signal 
    TrSyms : int
        number of training symbols to use
    Niter : int
        interations over the training symbols
    os : int
        oversampling factor
    mu : float
        equaliser step size 
    wx : array_like
        filter taps 
    adaptive : bool
        whether to perform an adaptive stepsize
    R : complex
        complex radius for the constant modulus

    Returns
    -------
    err : array_like
        error vector
    wx  : array_like
        filter taps
    mu  : float
        last step size
    """
    err, wx, mu = cma_like(E, TrSyms, Niter, os, mu, wx, adaptive, R, mcma_error)
    return err, wx, mu

#pythran export cma_equaliser(complex128[][], int, int, int, float64, complex128[][][],
    # bool, complex128)
#pythran export cma_equaliser(complex64[][], int, int, int, float32, complex64[][][],
# bool, complex64)
def cma_equaliser(E, TrSyms, Niter, os, mu, wx, adaptive, R):
    """
    Train filter taps using the constant modulus algorithm (CMA) (see equalisation.py for references)
    
    Parameters
    ----------
    E : array_like
        input signal 
    TrSyms : int
        number of training symbols to use
    Niter : int
        interations over the training symbols
    os : int
        oversampling factor
    mu : float
        equaliser step size 
    wx : array_like
        filter taps 
    adaptive : bool
        whether to perform an adaptive stepsize
    R : complex
        complex radius for the constant modulus

    Returns
    -------
    err : array_like
        error vector
    wx  : array_like
        filter taps
    mu  : float
        last step size
    """    
    err, wx, mu = cma_like(E, TrSyms, Niter, os, mu, wx, adaptive, R, cma_error)
    return err, wx, mu

#pythran export rde_equaliser(complex128[][], int, int, int, float64, complex128[][][],
    # bool, complex128[], complex128[])
#pythran export rde_equaliser(complex64[][], int, int, int, float32, complex64[][][],
    # bool, complex64[], complex64[])
def rde_equaliser(E, TrSyms, Niter, os, mu, wx, adaptive, partition, codebook):
    """
    Train filter taps using the radius direced (RDE) algorithm (see equalisation.py for references)
    
    Parameters
    ----------
    E : array_like
        input signal 
    TrSyms : int
        number of training symbols to use
    Niter : int
        interations over the training symbols
    os : int
        oversampling factor
    mu : float
        equaliser step size 
    wx : array_like
        filter taps 
    adaptive : bool
        whether to perform an adaptive stepsize
    partition : array_like
        complex partitioning borders for the radius decision
    codebook : array_like
        complex output radii for the radius decision

    Returns
    -------
    err : array_like
        error vector
    wx  : array_like
        filter taps
    mu  : float
        last step size
    """ 
    err, wx, mu = rde_like(E, TrSyms, Niter, os, mu, wx, adaptive, partition, codebook, rde_error)
    return err, wx, mu

#pythran export mrde_equaliser(complex128[][], int, int, int, float64, complex128[][][],
    # bool, complex128[], complex128[])
#pythran export mrde_equaliser(complex64[][], int, int, int, float32, complex64[][][],
    # bool, complex64[], complex64[])
def mrde_equaliser(E, TrSyms, Niter, os, mu, wx, adaptive, partition, codebook):
    """
    Train filter taps using the modified radius directed (MRDE) algorithm (see equalisation.py for references)
    
    Parameters
    ----------
    E : array_like
        input signal 
    TrSyms : int
        number of training symbols to use
    Niter : int
        interations over the training symbols
    os : int
        oversampling factor
    mu : float
        equaliser step size 
    wx : array_like
        filter taps 
    adaptive : bool
        whether to perform an adaptive stepsize
    partition : array_like
        complex partitioning borders for the radius decision
    codebook : array_like
        complex output radii for the radius decision

    Returns
    -------
    err : array_like
        error vector
    wx  : array_like
        filter taps
    mu  : float
        last step size
    """ 
    err, wx, mu = rde_like(E, TrSyms, Niter, os, mu, wx, adaptive, partition, codebook, mrde_error)
    return err, wx, mu

#pythran export sbd_equaliser(complex128[][], int, int, int, float64, complex128[][][],
    # bool, complex128[])
#pythran export sbd_equaliser(complex64[][], int, int, int, float32, complex64[][][],
# bool, complex64[])
def sbd_equaliser(E, TrSyms, Niter, os, mu, wx, adaptive, symbs):
    """
    Train filter taps using the symbol based decision (SBD) algorithm (see equalisation.py for references)
    
    Parameters
    ----------
    E : array_like
        input signal 
    TrSyms : int
        number of training symbols to use
    Niter : int
        interations over the training symbols
    os : int
        oversampling factor
    mu : float
        equaliser step size 
    wx : array_like
        filter taps 
    adaptive : bool
        whether to perform an adaptive stepsize
    symbs : array_like
        complex symbol alphabet

    Returns
    -------
    err : array_like
        error vector
    wx  : array_like
        filter taps
    mu  : float
        last step size
    """   
    err, wx, mu = dd_like(E, TrSyms, Niter, os, mu, wx, adaptive, symbs, sbd_error)
    return err, wx, mu

#pythran export mddma_equaliser(complex128[][], int, int, int, float64, complex128[][][],
    # bool, complex128[])
#pythran export mddma_equaliser(complex64[][], int, int, int, float32, complex64[][][],
# bool, complex64[])
def mddma_equaliser(E, TrSyms, Niter, os, mu, wx, adaptive, symbs):
    """
    Train filter taps using the modified decsions directed modulus (MDDMA) algorithm (see equalisation.py for references)
    
    Parameters
    ----------
    E : array_like
        input signal 
    TrSyms : int
        number of training symbols to use
    Niter : int
        interations over the training symbols
    os : int
        oversampling factor
    mu : float
        equaliser step size 
    wx : array_like
        filter taps 
    adaptive : bool
        whether to perform an adaptive stepsize
    symbs : array_like
        complex symbol alphabet

    Returns
    -------
    err : array_like
        error vector
    wx  : array_like
        filter taps
    mu  : float
        last step size
    """    
    err, wx, mu = dd_like(E, TrSyms, Niter, os, mu, wx, adaptive, symbs, mddma_error)
    return err, wx, mu

#pythran export ddlms_equaliser(complex128[][], int, int, int, float64, complex128[][][],
# bool, complex128[])
#pythran export ddlms_equaliser(complex64[][], int, int, int, float32, complex64[][][],
# bool, complex64[])
def ddlms_equaliser(E, TrSyms, Niter, os, mu, wx, adaptive, symbs):
    """
    Train filter taps using the decision directed least mean (DDLMS) square algorithm (see equalisation.py for references)
    
    Parameters
    ----------
    E : array_like
        input signal 
    TrSyms : int
        number of training symbols to use
    Niter : int
        interations over the training symbols
    os : int
        oversampling factor
    mu : float
        equaliser step size 
    wx : array_like
        filter taps 
    adaptive : bool
        whether to perform an adaptive stepsize
    symbs : array_like
        complex symbol alphabet

    Returns
    -------
    err : array_like
        error vector
    wx  : array_like
        filter taps
    mu  : float
        last step size
    """    
    err, wx, mu = dd_like(E, TrSyms, Niter, os, mu, wx, adaptive, symbs, ddlms_error)
    return err, wx, mu

def cma_like(E, TrSyms, Niter, os, mu, wx, adaptive, R, errfct):
    Ntaps = wx.shape[-1]
    pols = wx.shape[0]
    err = np.zeros((pols, TrSyms*Niter), dtype=E.dtype)
    #omp parallel for
    for pol in range(pols):
        for it in range(Niter):
            for i in range(TrSyms):
                X = E[:, i * os:i * os + Ntaps]
                Xest = apply_filter(X,  wx[pol])
                err[pol, it*Niter+i] = errfct(Xest, R)
                wx[pol] += mu * np.conj(err[pol, it*Niter+i]) * X
                if adaptive and i > 0:
                    mu = adapt_step(mu, err[pol, it*Niter+i], err[pol, it*Niter+i-1])
    return err, wx, mu

def rde_like(E, TrSyms, Niter, os, mu, wx, adaptive, partition, codebook, errfct):
    Ntaps = wx.shape[-1]
    pols = wx.shape[0]
    err = np.zeros((pols, TrSyms*Niter), dtype=E.dtype)
    #omp parallel for
    for pol in range(pols):
        for it in range(Niter):
            for i in range(TrSyms):
                X = E[:, i * os:i * os + Ntaps]
                Xest = apply_filter(X,  wx[pol])
                err[pol, it*Niter+i] = errfct(Xest, partition, codebook)
                wx[pol] += mu * np.conj(err[pol, it*Niter+i]) * X
                if adaptive and i > 0:
                    mu = adapt_step(mu, err[pol, it*Niter+i], err[pol, it*Niter+i-1])
    return err, wx, mu

def dd_like(E, TrSyms, Niter, os, mu, wx, adaptive, symbols, errfct):
    Ntaps = wx.shape[-1]
    pols = wx.shape[0]
    err = np.zeros((pols, TrSyms*Niter), dtype=E.dtype)
    #omp parallel for
    for pol in range(pols):
        for it in range(Niter):
            for i in range(TrSyms):
                X = E[:, i * os:i * os + Ntaps]
                Xest = apply_filter(X,  wx[pol])
                err[pol, it*Niter+i], dist, symb = errfct(Xest, symbols)
                wx[pol] += mu * np.conj(err[pol, it*Niter+i]) * X
                if adaptive and i > 0:
                    mu = adapt_step(mu, err[pol, it*Niter+i], err[pol, it*Niter+i-1])
    return err, wx, mu

######################################################
# Error functions
######################################################
def cma_error(Xest, R):
    return (R.real - abs(Xest)**2)*Xest

def mcma_error(Xest, R):
    return (R.real - Xest.real**2)*Xest.real + (R.imag - Xest.imag**2)*Xest.imag*1.j

def rde_error(Xest, partition, codebook):
    sq = abs(Xest)**2
    r = partition_value(sq, partition.real, codebook.real)
    return Xest*(r-sq)

def mrde_error(Xest, partition, codebook):
    sq = Xest.real**2 + 1j*Xest.imag
    r = partition_value(sq.real, partition.real, codebook.real) + 1j * partition_value(sq.imag, partition.imag, codebook.imag)
    return (r.real - sq.real)*Xest.real + 1j*(r.imag - sq.imag)*Xest.imag

def sbd_error(Xest, symbs):
    symbol, dist = det_symbol(Xest, symbs)
    return (symbol.real - Xest.real)*abs(symbol.real) + (symbol.imag - Xest.imag)*1.j*abs(symbol.imag), symbol, dist

def mddma_error(Xest, symbs):
    symbol, dist = det_symbol(Xest, symbs)
    return (symbol.real**2 - Xest.real**2)*Xest.real + 1.j*(symbol.imag**2 - Xest.imag**2)*Xest.imag, symbol, dist

def ddlms_error(Xest, symbs):
    symbol, dist = det_symbol(Xest, symbs)
    return symbol - Xest, symbol, dist

def det_symbol_argmin(X, symbs): # this version is about 1.5 times slower than the one below
    dist = np.abs(X-symbs)
    idx = np.argmin(dist)
    return symbs[idx], dist[idx]

#pythran export det_symbol(complex128, complex128[])
#pythran export det_symbol(complex64, complex64[])
def det_symbol(X, symbs):
    """
    Decision operator function
    
    Parameters
    ----------
    X : complex
        sample to make a decision on
    symbs: array_like
        symbol alphabet
        
    Returns
    -------
    s : complex
        symbol from symbs that was decided on
    d : float
        distance of X from s
    """
    d0 = 1000.
    s = 1.+1.j
    for j in range(symbs.shape[0]):
        d = abs(X-symbs[j])**2
        if d < d0:
            d0 = d
            s = symbs[j]
    return s, d0

#pythran export det_symbol_parallel(complex128, complex128[])
#pythran export det_symbol_parallel(complex64, complex64[])
def det_symbol_parallel(X, symbs): # this version can be much faster if not in a tight loop
    """
    Decision operator function. This is the parallel version which can be much faster if not run in a
    tight loop.

    Parameters
    ----------
    X : complex
        sample to make a decision on
    symbs: array_like
        symbol alphabet

    Returns
    -------
    s : complex
        symbol from symbs that was decided on
    d : float
         distance of X from s
    """
    d0 = 1000.
    s = 1.+1.j
    d0_priv = d0
    s_priv = s
    ##omp parallel for
    for j in range(symbs.shape[0]):
        d = abs(X-symbs[j])**2
        if d < d0_priv:
            d0_priv = d
            s_priv = symbs[j]
    #omp critical
    if d0_priv < d0:
        d0 = d0_priv
        s = s_priv
    return s, d0

#pythran export make_decision(complex128[], complex128[])
#pythran export make_decision(complex64[], complex64[])
def make_decision(E, symbols):
    """
    Apply decision operator to input signal
    
    Parameters
    ----------
    E : array_like
        input signal to decide on
    symbols : array_like
        symbol alphabet

    Returns
    -------
    det_symbs : array_like
        decided symbols decision
    """
    L = E.shape[0]
    M = symbols.shape[0]
    det_symbs = np.zeros_like(E)
    #omp parallel for
    for i in range(L):
        det_symbs[i] = det_symbol(E[i], symbols)[0]
    return det_symbs

