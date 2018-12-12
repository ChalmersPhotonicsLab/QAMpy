# -*- coding: utf-8 -*-
#  This file is part of QAMpy.
#
#  QAMpy is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  Foobar is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with QAMpy.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright 2018 Jochen Schröder, Mikael Mazur

# cython: profile=False, boundscheck=False, wraparound=False
from __future__ import division
import numpy as np
from cython.parallel import prange
cimport cython
from cpython cimport bool
cimport numpy as np
cimport scipy.linalg.cython_blas as scblas
from ccomplex cimport *

cdef class ErrorFct:
    cpdef double complex calc_error(self, double complex Xest):
        return 0
    cpdef float complex calc_errorf(self, float complex Xest):
        return 0

cdef complexing cconj(complexing x) nogil:
    if complexing is complex64_t:
        return conjf(x)
    else:
        return conj(x)

cdef complexing ccreal(complexing x) nogil:
    if complexing is complex64_t:
        return crealf(x)
    else:
        return creal(x)

cdef complexing ccimag(complexing x) nogil:
    if complexing is complex64_t:
        return cimagf(x)
    else:
        return cimag(x)

cdef complexing det_symbol(complexing[:] syms, int M, complexing value, cython.floating *dists) nogil:
    cdef complexing symbol = 0
    cdef cython.floating dist0
    cdef cython.floating dist
    dist0 = 10000.
    for j in range(M):
        dist = (syms[j].real - value.real)**2 + (syms[j].imag - value.imag)**2
        if dist < dist0:
            symbol = syms[j]
            dist0 = dist
    dists[0] = dist0
    return symbol

def make_decision(complexing[:] E, complexing[:] symbols):
    """
    Quantize signal to symbols, based on closest distance.

    Parameters
    ----------
    sig     : array_like
        input signal field, 1D array of complex values
    symbols : array_like
        symbol alphabet to quantize to (1D array, dtype=complex)

    Returns:
    sigsyms : array_like
        array of detected symbols
    """
    cdef int L = E.shape[0]
    cdef int M = symbols.shape[0]
    cdef int i, j, k
    cdef double distd
    cdef float distf
    cdef complexing[:] det_syms
    cdef complexing out_sym

    if complexing is complex64_t:
        det_syms = np.zeros(L, dtype=np.complex64)
        for i in prange(L, nogil=True, schedule='static'):
            out_sym = det_symbol(symbols, M, E[i], &distf)
            det_syms[i] = out_sym
        return det_syms
    else:
        det_syms = np.zeros(L, dtype=np.complex128)
        for i in prange(L, nogil=True, schedule='static'):
            out_sym = det_symbol(symbols, M, E[i], &distd)
            det_syms[i] = out_sym
        return det_syms

cdef partition_value(cython.floating signal,
                     double[:] partitions,
                     double[:] codebook):
                    #np.ndarray[ndim=1, dtype=np.float64_t] partitions,
                    #np.ndarray[ndim=1, dtype=np.float64_t] codebook):
    cdef unsigned int index = 0
    cdef unsigned int L = partitions.shape[0]
    while index < L and signal > partitions[index]:
        index += 1
    return codebook[index]

cdef cython.floating adapt_step(cython.floating mu, complexing err_p, complexing err):
    if err.real*err_p.real > 0 and err.imag*err_p.imag >  0:
        lm = 0
    else:
        lm = 1
    mu = mu/(1+lm*mu*(err.real*err.real + err.imag*err.imag))
    return mu

cdef complexing apply_filter(complexing[:,:] E, int Ntaps, complexing[:,:] wx, unsigned int pols) nogil:
    cdef int j, k
    cdef complexing Xest=0
    j = 1
    for k in range(0,pols):
        if complexing is complex64_t:
            Xest += scblas.cdotc(<int *> &Ntaps, &wx[k,0], &j, &E[k,0], &j)
        else:
            Xest += scblas.zdotc(<int *> &Ntaps, &wx[k,0], &j, &E[k,0], &j)
    return Xest

def apply_filter_to_signal(complexing[:,:] E, int os, complexing[:,:,:] wx):
    """
    Apply the equaliser filter taps to the input signal.

    Parameters
    ----------

    E      : array_like
        input signal to be equalised

    os     : int
        oversampling factor

    wxy    : tuple(array_like, array_like,optional)
        filter taps for the x and y polarisation

    Returns
    -------

    Eest   : array_like
        equalised signal
    """
    cdef complexing[:, :] output
    cdef int modes = E.shape[0]
    cdef int L = E.shape[1]
    cdef int Ntaps = wx.shape[2]
    cdef int i,j, N,idx
    cdef complexing Xest = 0
    N = (L - Ntaps + os)//os
    output = np.zeros((modes, N), dtype="c%d"%E.itemsize)
    for idx in prange(N*modes, nogil=True, schedule="static"):
        j = idx//N
        i = idx%N
        Xest = apply_filter(E[:, i*os:i*os+Ntaps], Ntaps, wx[j], modes)
        output[j, i] = Xest
    return np.array(output)

cdef void update_filter(complexing[:,:] E, int Ntaps, cython.floating mu, complexing err,
                        complexing[:,:] wx, int modes) nogil:
    cdef int i,j
    for k in range(modes):
        for j in range(Ntaps):
                wx[k, j] += mu * cconj(err) * E[k, j]

def train_eq(complexing[:,:] E,
                    int TrSyms,
                    unsigned int os,
                    cython.floating mu,
                    complexing[:,:] wx,
                    ErrorFct errfct,
                    bool adaptive=False):
    """
    Generate the filter taps by training the equaliser.

    Parameters
    ----------
    E : array_like
        signal to be equalised
    TrSyms : int
        number of training symbols to use
    os : int
        oversampling ratio
    mu : float
        tap update stepsize
    wx : array_like
        equaliser taps
    errfct : ErrorFct
        the equaliser error function to use
    adaptive : bool
        whether to use an adaptive step size

    Returns
    -------
    err : array_like
        error
    wxy : array_like
        adjusted taps
    """
    cdef complexing[:] err
    cdef unsigned int i, j, k
    cdef unsigned int pols = E.shape[0]
    cdef unsigned int L = E.shape[1]
    cdef unsigned int Ntaps = wx.shape[1]
    cdef complexing Xest
    err = np.zeros(TrSyms, dtype="c%d"%E.itemsize)
    for i in range(0, TrSyms):
        Xest = apply_filter(E[:, i*os:], Ntaps, wx, pols)
        if complexing is complex64_t:
            err[i] = errfct.calc_errorf(Xest)
            # this does make a significant difference
            update_filter(E[:, i*os:], Ntaps, <float> mu, err[i], wx, pols)
        else:
            err[i] = errfct.calc_error(Xest)
            update_filter(E[:, i*os:], Ntaps, mu, err[i], wx, pols)
        if adaptive and i > 0:
            mu = adapt_step(mu, err[i-1], err[i])
    return err, wx, mu
