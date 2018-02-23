# cython: profile=False, boundscheck=False, wraparound=False
from __future__ import division
import numpy as np
from cython.parallel import prange
cimport cython
from cpython cimport bool
cimport numpy as np
cimport scipy.linalg.cython_blas as scblas
from ccomplex cimport *
from cmath import sin

cdef complexing cconj(complexing x) nogil:
    if complexing is complex64_t:
        return conjf(x)
    else:
        return conj(x)

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

def quantize(complexing[:] E, complexing[:] symbols):
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
    cdef unsigned int L = E.shape[0]
    cdef unsigned int M = symbols.shape[0]
    cdef int i, j, k
    cdef double distd
    cdef float distf
    cdef np.ndarray[ndim=1, dtype=complexing] det_syms
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
                     np.float64_t[:] partitions,
                     np.float64_t[:] codebook):
                    #np.ndarray[ndim=1, dtype=np.float64_t] partitions,
                    #np.ndarray[ndim=1, dtype=np.float64_t] codebook):
    cdef unsigned int index = 0
    cdef unsigned int L = len(partitions)
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

cdef class ErrorFct:
    cpdef double complex calc_error(self, double complex Xest)  except *:
        return 0
    #def __call__(self, double complex Xest):
        #return self.calc_error(Xest)
    def __call__(self, np.ndarray[ndim=2, dtype=double complex] E,
                    int TrSyms,
                    int Ntaps,
                    unsigned int os,
                    double mu,
                    np.ndarray[ndim=2, dtype=double complex] wx,
                    bool adaptive=False):
        return train_eq(E, TrSyms, Ntaps, os, mu, wx, self, adaptive)

cdef double complex I=1.j

cdef class ErrorFctMCMA(ErrorFct):
    cdef double R_real
    cdef double R_imag
    def __init__(self, double complex R):
        self.R_real = R.real
        self.R_imag = R.imag
    cpdef double complex calc_error(self, double complex Xest)  except *:
        return (creal(Xest)**2 - self.R_real)*creal(Xest) + I*(cimag(Xest)**2 -self.R_imag)*cimag(Xest)

cdef class ErrorFctCMA(ErrorFct):
    cdef double R
    def __init__(self, double R):
        self.R = R
    cpdef double complex calc_error(self, double complex Xest)  except *:
        return (creal(Xest)**2 + cimag(Xest)**2 - self.R)*Xest

cdef class ErrorFctRDE(ErrorFct):
    cdef np.float64_t[:] partition
    cdef np.float64_t[:] codebook
    def __init__(self, np.ndarray[ndim=1, dtype=np.float64_t] partition,
                                              np.ndarray[ndim=1, dtype=np.float64_t] codebook):
        self.partition = partition
        self.codebook = codebook
    cpdef double complex calc_error(self, double complex Xest)  except *:
        cdef double Ssq
        cdef double S_DD
        Ssq = Xest.real**2 + Xest.imag**2
        S_DD = partition_value(Ssq, self.partition, self.codebook)
        return Xest*(Ssq - S_DD)

cdef class ErrorFctMRDE(ErrorFct):
    cdef np.float64_t[:] partition_real
    cdef np.float64_t[:] partition_imag
    cdef np.float64_t[:] codebook_real
    cdef np.float64_t[:] codebook_imag
    def __init__(self, np.ndarray[ndim=1, dtype=np.complex128_t] partition,
                                              np.ndarray[ndim=1, dtype=np.complex128_t] codebook):
        self.partition_real = partition.real
        self.partition_imag = partition.imag
        self.codebook_real = codebook.real
        self.codebook_imag = codebook.imag
    cpdef double complex calc_error(self, double complex Xest)  except *:
        cdef double complex Ssq
        cdef double complex S_DD
        Ssq = creal(Xest)**2 + 1.j * cimag(Xest)**2
        S_DD = partition_value(Ssq.real, self.partition_real, self.codebook_real) + 1.j * partition_value(Ssq.imag, self.partition_imag, self.codebook_imag)
        return (Ssq.real - S_DD.real)*Xest.real + 1.j*(Ssq.imag - S_DD.imag)*Xest.imag

cdef class ErrorFctGenericDD(ErrorFct):
    cdef double complex[:] symbols
    cdef double dist
    cdef int N
    def __init__(self, np.ndarray[ndim=1, dtype=double complex] symbols):
        self.symbols = symbols
        self.N = symbols.shape[0]

cdef class ErrorFctSBD(ErrorFctGenericDD):
    cpdef double complex calc_error(self, double complex Xest)  except *:
        cdef double complex R
        R = det_symbol(self.symbols, self.N, Xest, &self.dist)
        return (Xest.real - R.real)*abs(R.real) + 1.j*(Xest.imag - R.imag)*abs(R.imag)

cdef class ErrorFctMDDMA(ErrorFctGenericDD):
    cpdef double complex calc_error(self, double complex Xest)  except *:
        cdef double complex R
        R = det_symbol(self.symbols, self.N, Xest, &self.dist)
        return (Xest.real**2 - R.real**2)*Xest.real + 1.j*(Xest.imag**2 - R.imag**2)*Xest.imag

cdef class ErrorFctDD(ErrorFctGenericDD):
    cpdef double complex calc_error(self, double complex Xest)  except *:
        cdef double complex R
        R = det_symbol(self.symbols, self.N, Xest, &self.dist)
        return Xest - R

cdef class ErrorFctSCA(ErrorFct):
    cdef double complex R
    def __init__(self, double complex R):
        self.R = R
    cpdef double complex calc_error(self, double complex Xest) except *:
        cdef int A
        cdef int B
        if abs(Xest.real) >= abs(Xest.imag):
            A = 1
            if abs(Xest.real) == abs(Xest.imag):
                B = 1
            else:
                B = 0
        else:
            A = 0
            B = 1
        return 4*Xest.real*(4*Xest.real**2 - 4*self.R**2)*A + 1.j*4*Xest.imag*(4*Xest.imag**2 - 4*self.R**2)*B

cdef class ErrorFctCME(ErrorFct):
    cdef double beta
    cdef double d
    cdef double R
    def __init__(self, double R, double d, double beta):
        self.beta = beta
        self.d = d
        self.R = R
    cpdef double complex calc_error(self, double complex Xest) except *:
        return (abs(Xest)**2 - self.R)*Xest + self.beta * np.pi/(2*self.d) * (sin(Xest.real*np.pi/self.d) + 1.j * sin(Xest.imag*np.pi/self.d))

cdef complexing apply_filter(complexing[:,:] E, int Ntaps, complexing[:,:] wx, unsigned int pols) nogil:
    cdef int j, k
    cdef complexing Xest=0
    j = 1
    for k in range(0,pols):
        if complexing is complex64_t:
            Xest += scblas.cdotu(<int *> &Ntaps, &E[k,0], &j, &wx[k,0], &j)
        else:
            Xest += scblas.zdotu(<int *> &Ntaps, &E[k,0], &j, &wx[k,0], &j)
    return Xest

cdef void update_filter(complexing[:,:] E, int Ntaps, cython.floating mu, complexing err,
                        complexing[:,:] wx, int modes) nogil:
    cdef int i,j
    for k in range(modes):
        for j in range(Ntaps):
                wx[k, j] -= mu * err * cconj(E[k, j])

def train_eq(double complex[:,:] E, #np.ndarray[ndim=2, dtype=double complex] E,
                    int TrSyms,
                    int Ntaps,
                    unsigned int os,
                    double mu,
                    #np.ndarray[ndim=2, dtype=double complex] wx,
                    double complex[:,:] wx,
                    ErrorFct errfct,
                    bool adaptive=False):
    cdef np.ndarray[ndim=1, dtype=double complex] err = np.zeros(TrSyms, dtype=np.complex128)
    cdef unsigned int i, j, k
    cdef unsigned int pols = E.shape[0]
    cdef unsigned int L = E.shape[1]
    cdef double complex Xest

    for i in range(0, TrSyms):
        Xest = apply_filter(E[:, i*os:], Ntaps, wx, pols)
        err[i] = errfct.calc_error(Xest)
        update_filter(E[:, i*os:], Ntaps, mu, err[i], wx, pols)
        if adaptive and i > 0:
            mu = adapt_step(mu, err[i-1], err[i])
    return err, wx
