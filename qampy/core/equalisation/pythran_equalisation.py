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

def adapt_step_real(mu, err_p, err):
    if err*err_p > 0:
        return mu
    else:
        return mu/(1+mu*(err*err))

def apply_filter(E, wx):
    pols = E.shape[0]
    Ntaps = wx.shape[1]
    Xest = E.dtype.type(0)
    for k in range(pols):
        for i in range(Ntaps):
            Xest += E[k, i]*np.conj(wx[k,i])
    return Xest

#pythran export apply_filter_to_signal(float64[:,:], int, float64[:,:,:], int[:] or None)
#pythran export apply_filter_to_signal(float32[:,:], int, float32[:,:,:], int[:] or None)
#pythran export apply_filter_to_signal(complex128[:,:], int, complex128[:,:,:], int[:] or None)
#pythran export apply_filter_to_signal(complex64[:,:], int, complex64[:,:,:], int[:] or None)
def apply_filter_to_signal(E, os, wx, modes=None):
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
    modes : array_like, optional
        mode numbers over which to apply the filters
        

    Returns
    -------
    output : array_like
        filtered and downsampled signal
    """
    assert os > 0, "oversampling factor must be larger than 0"
    nmodes_max = wx.shape[0]
    Ntaps = wx.shape[-1]
    if modes is None:
        modes = np.arange(nmodes_max)
        nmodes = nmodes_max
    else:
        modes = np.atleast_1d(modes)
        assert np.max(modes) < nmodes_max, "largest mode number is larger than shape of signal"
        nmodes = modes.size
    L = E.shape[1]
    N = (L-Ntaps+1)//os
    output  = np.zeros((nmodes, N), dtype=E.dtype)
    #omp parallel for collapse(2)
    for j in range(nmodes):
        for i in range(N):
            Xest = apply_filter(E[:, i*os:i*os+Ntaps], wx[modes[j]])
            output[j, i] = Xest
    return output

#pythran export train_equaliser_realvalued(float32[][], int, int, int, float32, float32[][][], int[], bool, float32[][], str)
#pythran export train_equaliser_realvalued(float64[][], int, int, int, float64, float64[][][], int[], bool, float64[][], str)
def train_equaliser_realvalued(E, TrSyms, Niter, os, mu, wx, modes, adaptive, symbols,  method):
    if method == "cma":
        errorfct = cma_error_real
    elif method == "dd":
        errorfct = dd_error_real
    elif method == "dd_data":
        errorfct = dd_data_error_real
    else:
        raise ValueError("Unknown method %s"%method)
    nmodes = E.shape[0]
    ntaps = wx.shape[-1]
    assert symbols.shape[0] == nmodes, "symbols must be at least size of modes"
    assert wx.shape[0] == nmodes, "wx needs to have at least as many dimensions as the maximum mode"
    assert E.shape[1] > TrSyms*os+ntaps, "Field must be longer than the number of training symbols"
    assert modes.max() < nmodes, "Maximum mode number must not be higher than number of modes"
    err = np.zeros((nmodes, TrSyms*Niter), dtype=E.dtype)
    #omp parallel for
    for mode in modes:
        for it in range(Niter):
            for i in range(TrSyms):
                X = E[:, i * os:i * os + ntaps]
                Xest = apply_filter(X,  wx[mode])
                err[mode, it*TrSyms+i] = errorfct(Xest, symbols[mode], i)
                wx[mode] += mu * err[mode, it*TrSyms+i] * X
                if adaptive and i > 0:
                    mu = adapt_step_real(mu, err[mode, it*TrSyms+i], err[mode, it*TrSyms+i-1])
    return err, wx, mu

def cma_error_real(Xest, s1, i):
    d = s1[0] - abs(Xest)**2
    return d*Xest

def dd_error_real(Xest, symbs, i):
    symbol, dist, i = det_symbol_argmin(Xest, symbs)
    return (symbol - Xest)*abs(symbol) 

def dd_data_error_real(Xest, symbs, i):
    symbol = symbs[i]
    dist = symbol - Xest
    return dist*abs(symbol)


#pythran export train_equaliser(complex128[][], int, int, int, float64, complex128[][][], int[], bool, complex128[][], str)
#pythran export train_equaliser(complex64[][], int, int, int, float32, complex64[][][], int[], bool, complex64[][], str)
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
    elif method == "dd":
        errorfct = ddlms_error
    elif method == "sbd_data":
        errorfct = sbd_data_error
    else:
        raise ValueError("Unknown method %s"%method)
    nmodes = E.shape[0]
    ntaps = wx.shape[-1]
    assert symbols.shape[0] == nmodes, "symbols must be at least size of modes"
    assert wx.shape[0] == nmodes, "wx needs to have at least as many dimensions as the maximum mode"
    assert E.shape[1] > TrSyms*os+ntaps, "Field must be longer than the number of training symbols"
    assert modes.max() < nmodes, "Maximum mode number must not be higher than number of modes"
    err = np.zeros((nmodes, TrSyms*Niter), dtype=E.dtype)
    #omp parallel for
    for mode in modes:
        for it in range(Niter):
            for i in range(TrSyms):
                X = E[:, i * os:i * os + ntaps]
                Xest = apply_filter(X,  wx[mode])
                err[mode, it*TrSyms+i] = errorfct(Xest, symbols[mode], i)
                wx[mode] += mu * np.conj(err[mode, it*TrSyms+i]) * X
                if adaptive and i > 0:
                    mu = adapt_step(mu, err[mode, it*TrSyms+i], err[mode, it*TrSyms+i-1])
    return err, wx, mu

######################################################
# Error functions
######################################################
def cma_error(Xest, s1, i):
    d = s1[0].real - abs(Xest)**2
    return d*Xest

def mcma_error(Xest, s1, i):
    J = Xest.dtype.type(1j)
    dr = (s1[0].real - Xest.real**2)
    di = (s1[0].imag - Xest.imag**2)
    return dr*Xest.real + di*Xest.imag*J

def rde_error(Xest, symbs, i):
    partition, codebook = np.split(symbs, 2)
    sq = abs(Xest)**2
    r = partition_value(sq, partition.real, codebook.real)
    return Xest*(r-sq)

def mrde_error(Xest, symbs, i):
    J = Xest.dtype.type(1j)
    partition, codebook = np.split(symbs, 2)
    sq = Xest.real**2 + J*Xest.imag**2
    r = partition_value(sq.real, partition.real, codebook.real) + J * partition_value(sq.imag, partition.imag, codebook.imag)
    return (r.real - sq.real)*Xest.real + J*(r.imag - sq.imag)*Xest.imag

def sbd_error(Xest, symbs, i):
    J = Xest.dtype.type(1j)
    symbol, dist = det_symbol(Xest, symbs)
    return (symbol.real - Xest.real)*abs(symbol.real) + (symbol.imag - Xest.imag)*J*abs(symbol.imag)

def sbd_data_error(Xest, symbs, i):
    J = Xest.dtype.type(1j)
    symbol = symbs[i]
    dist = symbol - Xest
    return dist.real*abs(symbol.real) + dist.imag*J*abs(symbol.imag)

def mddma_error(Xest, symbs, i):
    J = Xest.dtype.type(1j)
    symbol, dist = det_symbol(Xest, symbs)
    return (symbol.real**2 - Xest.real**2)*Xest.real + J*(symbol.imag**2 - Xest.imag**2)*Xest.imag

def ddlms_error(Xest, symbs, i):
    symbol, dist = det_symbol(Xest, symbs)
    return symbol - Xest

def det_symbol_argmin(X, symbs): # this version is about 1.5 times slower than the one below
    dist = np.abs(X-symbs)
    idx = np.argmin(dist)
    return symbs[idx], dist[idx], idx

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
    s = symbs.dtype.type(1)
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
    s = symbs.dtype.type(1)
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
    # TODO we might want to make this work for ND symbols (would require to use a manual collapse)
    L = E.shape[0]
    M = symbols.shape[0]
    det_symbs = np.zeros_like(E)
    dist = np.zeros(E.shape, dtype=E.real.dtype)
    idx = np.zeros(E.shape, dtype=np.int32)
    #omp parallel for
    for i in range(L):
        s, d, ix = det_symbol_argmin(E[i], symbols)
        dist[i] = d
        det_symbs[i] = s
        idx[i] = ix
    return det_symbs, dist, idx
