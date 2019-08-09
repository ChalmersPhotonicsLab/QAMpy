import numpy as np


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

#pythran export train_eq(complex128[][], int, int, float64,
    # complex128[][], str,
    # (complex128, complex128[]), bool)
#pythran export train_eq(complex64[][], int, int, float32,
        # complex64[][], str,
        # (complex64, complex64[]), bool)
def train_eq(E, TrSyms, os, mu, wx, method,  errfctprs, adaptive):
    Ntaps = wx.shape[1]
    pols = wx.shape[0]
    R, symbs = errfctprs
    if method == "cma":
        errfct = cma_error
    elif method == "mcma":
        errfct = mcma_error
    elif method == "sbd":
        errfct = sbd_error
    elif method == "mddma":
        errfct = mddma_error
    elif method == "dd":
        errfct = dd_error
    else:
        raise ValueError("%s does not correspond to an error function")
    err = np.zeros(TrSyms, dtype=E.dtype)
    for i in range(TrSyms):
        X = E[:, i * os:i * os + Ntaps]
        Xest = apply_filter(X,  wx)
        err[i], R = errfct(Xest, R, symbs)
        wx += mu * np.conj(err[i]) * X
        if adaptive and i > 0:
            mu = adapt_step(mu, err[i], err[i-1])
    return err, wx, mu

# Error functions

def cma_error(Xest, R, symbs):
    return (R.real - abs(Xest)**2)*Xest, R+0j

def mcma_error(Xest, R, symbs):
    return (R.real - Xest.real**2)*Xest.real + (R.imag - Xest.imag**2)*Xest.imag*1.j, R

def sbd_error(Xest, R, symbs):
    #R,d = symbs[np.argmin(np.abs(Xest-symbs))]
    R, d = det_symbol(Xest, symbs)
    return (R.real - Xest.real)*abs(R.real) + (R.imag - Xest.imag)*1.j*abs(R.imag), R

def mddma_error(Xest, R, symbs):
    R, d = det_symbol(Xest, symbs)
    return (R.real**2 - Xest.real**2)*Xest.real + 1.j*(R.imag**2 - Xest.imag**2)*Xest.imag, R

def dd_error(Xest, R, symbs):
    R, d = det_symbol(Xest, symbs)
    return R - Xest, R

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

