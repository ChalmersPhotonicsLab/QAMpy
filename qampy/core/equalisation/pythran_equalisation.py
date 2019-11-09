import numpy as np

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

#pythran export apply_filter_to_signal(complex128[:,:], int, complex128[:,:,:])
#pythran export apply_filter_to_signal(complex64[:,:], int, complex64[:,:,:])
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
    modes = wx.shape[0]
    N = (L-Ntaps+os)//os
    output  = np.zeros((modes, N), dtype=E.dtype)
    #omp parallel for
    for idx in range(modes*N): # manual collapse of loop
        j = idx//N
        i = idx%N
        Xest = apply_filter(E[:, i*os:i*os+Ntaps], wx[j])
        output[j, i] = Xest
    return output

#pythran export train_equaliser(complex128[][], int, int, int, float64, complex128[][][], int[], bool, complex128[][], str)
def train_equaliser(E, TrSyms, Niter, os, mu, wx, modes, adaptive, symbols,  method):
    if method == "mcma":
        errorfct = mcma_error
    elif method == "cma":
        errorfct = cma_error
    elif method == "sbd":
        errorfct = sbd_error
    elif method == "rde":
        errorfct = rde_error
    elif method == "mrde":
        errorfct = mrde_error
    elif method == "mddma":
        errorfct = mddma_error
    elif method == "ddlms":
        errorfct = ddlms_error
    elif method == "sbd_data":
        errorfct = sbd_data_error
    else:
        raise ValueError("Unknown method %s"%method)
    nmodes = E.shape[0]
    assert symbols.shape[0] == nmodes, "symbols must be at least size of modes"
    assert wx.shape[0] == nmodes, "wx needs to have at least as many dimensions as the maximum mode"
    ntaps = wx.shape[-1]
    err = np.zeros((nmodes, TrSyms*Niter), dtype=E.dtype)
    #omp parallel for
    for mode in modes:
        for it in range(Niter):
            for i in range(TrSyms):
                X = E[:, i * os:i * os + ntaps]
                Xest = apply_filter(X,  wx[mode])
                err[mode, it*Niter+i], symb, d = errorfct(Xest, symbols[mode], i)
                wx[mode] += mu * np.conj(err[mode, it*Niter+i]) * X
                if adaptive and i > 0:
                    mu = adapt_step(mu, err[mode, it*Niter+i], err[mode, it*Niter+i-1])
    return err, wx, mu

######################################################
# Error functions
######################################################
def cma_error(Xest, s1, i):
    d = s1[0].real - abs(Xest)**2
    return d*Xest, Xest, d

def mcma_error(Xest, s1, i):
    dr = (s1[0].real - Xest.real**2)
    di = (s1[0].imag - Xest.imag**2)
    return dr*Xest.real + di*Xest.imag*1.j, Xest, dr + di

def rde_error(Xest, symbs, i):
    partition, codebook = np.split(symbs, 2)
    sq = abs(Xest)**2
    r = partition_value(sq, partition.real, codebook.real)
    return Xest*(r-sq), r+0j, sq

def mrde_error(Xest, symbs, i):
    partition, codebook = np.split(symbs, 2)
    sq = Xest.real**2 + 1j*Xest.imag**2
    r = partition_value(sq.real, partition.real, codebook.real) + 1j * partition_value(sq.imag, partition.imag, codebook.imag)
    return (r.real - sq.real)*Xest.real + 1j*(r.imag - sq.imag)*Xest.imag, r, abs(sq)

def sbd_error(Xest, symbs, i):
    symbol, dist = det_symbol(Xest, symbs)
    return (symbol.real - Xest.real)*abs(symbol.real) + (symbol.imag - Xest.imag)*1.j*abs(symbol.imag), symbol, dist

def sbd_data_error(Xest, symbs, i):
    symbol = symbs[i]
    dist = symbol - Xest
    return dist.real*abs(symbol.real) + dist.imag*1.j*abs(symbol.imag), symbol, abs(dist)

def mddma_error(Xest, symbs, i):
    symbol, dist = det_symbol(Xest, symbs)
    return (symbol.real**2 - Xest.real**2)*Xest.real + 1.j*(symbol.imag**2 - Xest.imag**2)*Xest.imag, symbol, dist

def ddlms_error(Xest, symbs, i):
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
