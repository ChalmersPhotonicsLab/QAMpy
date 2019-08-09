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
    # complex128[][], int,
    # (complex128, complex128[]), bool)
#pythran export train_eq(complex64[][], int, int, float32,
        # complex64[][], int,
        # (complex64, complex64[]), bool)
def train_eq(E, TrSyms, os, mu, wx, errN,  errfctprs, adaptive):
    Ntaps = wx.shape[1]
    pols = wx.shape[0]
    R, symbs = errfctprs
    if errN == 1:
        errfct = cma_error
    elif errN == 2:
        errfct = mcma_error
    elif errN == 3:
        errfct = sbd_error
    elif errN == 4:
        errfct = mddma_error
    elif errN == 5:
        errfct = dd_error
    else:
        raise ValueError("%d does not correspond to an error function (valid values: 1=5)")
    err = np.zeros(TrSyms, dtype=E.dtype)
    for i in range(TrSyms):
        X = E[:, i * os:i * os + Ntaps]
        Xest = apply_filter(X,  wx)
        err[i] = errfct(Xest, R, symbs)
        wx += mu * np.conj(err[i]) * X
        if adaptive and i > 0:
            mu = adapt_step(mu, err[i], err[i-1])
    return err, wx, mu

# Error functions

def cma_error(Xest, R, symbs):
    return (R.real - abs(Xest)**2)*Xest

def mcma_error(Xest, R, symbs):
    return (R.real - Xest.real**2)*Xest.real + (R.imag - Xest.imag**2)*Xest.imag*1.j

def sbd_error(Xest, R, symbs):
    #R,d = symbs[np.argmin(np.abs(Xest-symbs))]
    R, d = det_symbol(Xest, symbs)
    return (R.real - Xest.real)*abs(R.real) + (R.imag - Xest.imag)*1.j*abs(R.imag)

def mddma_error(Xest, R, symbs):
    R, d = det_symbol(Xest, symbs)
    return (R.real**2 - Xest.real**2)*Xest.real + 1.j*(R.imag**2 - Xest.imag**2)*Xest.imag

def dd_error(Xest, R, symbs):
    R, d = det_symbol(Xest, symbs)
    return R - Xest

#def det_symbol(X, symbs):
    #dist = np.abs(X-symbs)
    #idx = np.argmin(dist)
    #return symbs[idx], dist[idx]

#we need to use this version because of #1133 which made argmin significantly slower
#pythran export det_symbol(complex128, complex128[])
#pythran export det_symbol(complex64, complex64[])
def det_symbol(X, symbs):
    d0 = 1000.
    s = 1.+1.j
    d0_priv = d0
    s_priv = s
    #omp parallel for
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

