import numpy as np

def partition_value(signal, partitions, codebook):
    L = partitions.shape[0]
    index = 0
    while index < L and signal > partitions[index]:
        index += 1
    return codebook[index]

def cabsq(x):
    return x.real**2 + x.imag**2

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
    err, wx, mu = cma_like(E, TrSyms, Niter, os, mu, wx, adaptive, R, mcma_error)
    return err, wx, mu

#pythran export cma_equaliser(complex128[][], int, int, int, float64, complex128[][][],
    # bool, complex128)
#pythran export cma_equaliser(complex64[][], int, int, int, float32, complex64[][][],
# bool, complex64)
def cma_equaliser(E, TrSyms, Niter, os, mu, wx, adaptive, R):
    err, wx, mu = cma_like(E, TrSyms, Niter, os, mu, wx, adaptive, R, cma_error)
    return err, wx, mu

#pythran export sbd_equaliser(complex128[][], int, int, int, float64, complex128[][][],
    # bool, complex128[])
#pythran export sbd_equaliser(complex64[][], int, int, int, float32, complex64[][][],
# bool, complex64[])
def sbd_equaliser(E, TrSyms, Niter, os, mu, wx, adaptive, symbs):
    err, wx, mu = dd_like(E, TrSyms, Niter, os, mu, wx, adaptive, symbs, sbd_error)
    return err, wx, mu

#pythran export mddma_equaliser(complex128[][], int, int, int, float64, complex128[][][],
    # bool, complex128[])
#pythran export mddma_equaliser(complex64[][], int, int, int, float32, complex64[][][],
# bool, complex64[])
def mddma_equaliser(E, TrSyms, Niter, os, mu, wx, adaptive, symbs):
    err, wx, mu = dd_like(E, TrSyms, Niter, os, mu, wx, adaptive, symbs, mddma_error)
    return err, wx, mu

#pythran export ddlms_equaliser(complex128[][], int, int, int, float64, complex128[][][],
# bool, complex128[])
#pythran export ddlms_equaliser(complex64[][], int, int, int, float32, complex64[][][],
# bool, complex64[])
def ddlms_equaliser(E, TrSyms, Niter, os, mu, wx, adaptive, symbs):
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
    d0 = 1000.
    s = 1.+1.j
    for j in range(symbs.shape[0]):
        d = cabsq(X-symbs[j])
        if d < d0:
            d0 = d
            s = symbs[j]
    return s, d0

#pythran export det_symbol_parallel(complex128, complex128[])
#pythran export det_symbol_parallel(complex64, complex64[])
def det_symbol_parallel(X, symbs): # this version can be much faster if not in a tight loop
    d0 = 1000.
    s = 1.+1.j
    d0_priv = d0
    s_priv = s
    ##omp parallel for
    for j in range(symbs.shape[0]):
        d = cabsq(X-symbs[j])
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
    L = E.shape[0]
    M = symbols.shape[0]
    det_symbs = np.zeros_like(E)
    #omp parallel for
    for i in range(L):
        det_symbs[i] = det_symbol(E[i], symbols)[0]
    return det_symbs

