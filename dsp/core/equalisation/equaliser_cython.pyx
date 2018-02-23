# cython: profile=False, boundscheck=False, wraparound=False
from __future__ import division
import numpy as np
from cython.view cimport array as cvarray
from cython.parallel import prange
cimport cython
from cpython cimport bool
cimport numpy as np
cimport scipy.linalg.cython_blas as scblas


ctypedef double complex complex128_t
ctypedef float complex complex64_t
ctypedef long double complex complex256_t

ctypedef long double float128_t

ctypedef fused complex_all:
    float complex
    double complex
    long double complex

ctypedef fused floating_all:
    float
    double
    long double

cdef extern from "math.h":
    double sin(double)

cdef extern from "complex.h":
    double complex conj(double complex)

cdef extern from "complex.h":
    float complex conjf(float complex)

cdef extern from "complex.h":
    long double complex conjl(long double complex)

cdef extern from "complex.h":
    double creal(double complex)

cdef extern from "complex.h" nogil:
    double cabs(double complex)

cdef extern from "complex.h" nogil:
    long double cabsl(long double complex)

cdef extern from "complex.h" nogil:
    float cabsf(float complex)

cdef extern from "complex.h":
    double cimag(double complex)

cdef extern from "equaliserC.h":
    double complex apply_filter(double complex *E, unsigned int Ntaps, double complex *wx, unsigned int pols, unsigned int L)

cdef extern from "equaliserC.h":
    double complex mcma_error(double complex Xest, void *args)

cdef extern from "equaliserC.h" nogil:
    double complex det_symbol(double complex *syms, unsigned int M, double complex value, double *distout)

cdef extern from "equaliserC.h":
    void update_filter(double complex *E, unsigned int Ntaps, double mu, double complex err, double complex *wx, unsigned int pols, unsigned int L)


cdef complex_all det_symbol2(complex_all[:] syms, int M, complex_all value, floating_all *dists) nogil:
    cdef complex_all symbol = 0
    cdef floating_all dist0
    cdef floating_all dist
    dist0 = 10000.
    for j in range(M):
        dist = (syms[j].real - value.real)**2 + (syms[j].imag - value.imag)**2
        if dist < dist0:
            symbol = syms[j]
            dist0 = dist
    return symbol

cdef double complex det_symbol_d(double complex[:] syms, int M, double complex value, double *dists) nogil:
    cdef double complex symbol = 0
    cdef double dist0
    cdef double dist
    dist0 = 10000.
    for j in range(M):
        dist = cabs(syms[j]-value)
        if dist < dist0:
            symbol = syms[j]
            dist0 = dist
    return symbol

cdef long double complex det_symbol_l(long double complex[:] syms, int M, long double complex value, long double *dists) nogil:
    cdef long double complex symbol = 0
    cdef long double dist0
    cdef long double dist
    dist0 = 10000.
    for j in range(M):
        dist = cabsl(syms[j]-value)
        if dist < dist0:
            symbol = syms[j]
            dist0 = dist
    return symbol

cdef float complex det_symbol_f(float complex[:] syms, int M, float complex value, float *dists) nogil:
    cdef float complex symbol = 0
    cdef float dist0
    cdef float dist
    dist0 = 10000.
    for j in range(M):
        dist = cabsf(syms[j]-value)
        if dist < dist0:
            symbol = syms[j]
            dist0 = dist
    return symbol

def quantize(complex_all[:] E, complex_all[:] symbols):
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
    cdef long double distl
    cdef np.ndarray[ndim=1, dtype=complex_all] det_syms
    cdef complex_all out_sym

    if complex_all is complex256_t:
        det_syms = np.zeros(L, dtype=np.complex256)
        for i in prange(L, nogil=True, schedule='static'):
            out_sym = det_symbol2(symbols, M, E[i], &distl)
            det_syms[i] = out_sym
        return det_syms
    elif complex_all is complex64_t:
        det_syms = np.zeros(L, dtype=np.complex64)
        for i in prange(L, nogil=True, schedule='static'):
            out_sym = det_symbol2(symbols, M, E[i], &distf)
            det_syms[i] = out_sym
        return det_syms
    else:
        det_syms = np.zeros(L, dtype=np.complex128)
        for i in prange(L, nogil=True, schedule='static'):
            out_sym = det_symbol2(symbols, M, E[i], &distd)
            det_syms[i] = out_sym
        return det_syms

def quantize2(np.ndarray[ndim=1, dtype=np.complex128_t] E, np.ndarray[ndim=1, dtype=np.complex128_t] symbols):
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
    cdef double dists
    cdef int i
    cdef np.ndarray[ndim =1, dtype=np.complex128_t] det_syms = np.zeros(L, dtype=np.complex128)
    for i in prange(L, nogil=True, schedule='static', num_threads=8):
        det_syms[i] = det_symbol(<double complex *>symbols.data, M, E[i], &dists)
    return det_syms

cdef partition_value(double signal,
                     np.float64_t[:] partitions,
                     np.float64_t[:] codebook):
                    #np.ndarray[ndim=1, dtype=np.float64_t] partitions,
                    #np.ndarray[ndim=1, dtype=np.float64_t] codebook):
    cdef unsigned int index = 0
    cdef unsigned int L = len(partitions)
    while index < L and signal > partitions[index]:
        index += 1
    return codebook[index]

cdef double adapt_step(double mu, double complex err_p, double complex err):
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
        R = det_symbol(&self.symbols[0], self.N, Xest, &self.dist)
        return (Xest.real - R.real)*abs(R.real) + 1.j*(Xest.imag - R.imag)*abs(R.imag)

cdef class ErrorFctMDDMA(ErrorFctGenericDD):
    cpdef double complex calc_error(self, double complex Xest)  except *:
        cdef double complex R
        R = det_symbol(&self.symbols[0], self.N, Xest, &self.dist)
        return (Xest.real**2 - R.real**2)*Xest.real + 1.j*(Xest.imag**2 - R.imag**2)*Xest.imag

cdef class ErrorFctDD(ErrorFctGenericDD):
    cpdef double complex calc_error(self, double complex Xest)  except *:
        cdef double complex R
        R = det_symbol(&self.symbols[0], self.N, Xest, &self.dist)
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

cdef complex_all apply_filter2(complex_all[:,:] E, int Ntaps, complex_all[:,:] wx, int modes) nogil:
    cdef int j,k
    cdef complex_all Xest
    Xest = 0
    for k in range(modes):
        for j in range(Ntaps):
            Xest += wx[k,j] * E[k, j]
    return Xest

cdef complex_all apply_filter3(complex_all[:,:] E, int Ntaps, complex_all[:,:] wx, unsigned int pols):
    cdef int j, k
    cdef complex_all Xest=0
    j = 1
    for k in range(0,pols):
        if complex_all is complex64_t:
            Xest += scblas.cdotu(<int *> &Ntaps, &E[k,0], &j, &wx[k,0], &j)
        if complex_all is complex128_t:
            Xest += scblas.zdotu(<int *> &Ntaps, &E[k,0], &j, &wx[k,0], &j)
    return Xest

def testapplyfilter1(complex_all[:,:] E, int taps, complex_all[:,:] wx):
    cdef complex_all Xest = 0
    cdef int pols = E.shape[0]
    Xest = apply_filter2(E, taps, wx, pols)
    return Xest

def testapplyfilter2(complex_all[:,:] E, int taps, complex_all[:,:] wx):
    cdef complex_all Xest = 0
    cdef int pols = E.shape[0]
    Xest = apply_filter3(E, taps, wx, pols)
    return Xest


cdef void update_filter2(complex_all[:,:] E, int Ntaps, floating_all mu, complex_all err,
                        complex_all[:,:] wx, int modes):
    cdef int i,j
    for k in range(modes):
        for j in range(Ntaps):
            if complex_all is complex64_t:
                wx[k, j] -= mu * err * conjf(E[k, j])
            elif complex_all is complex256_t:
                wx[k, j] -= mu * err * conjl(E[k, j])
            else:
                wx[k, j] -= mu * err * conj(E[k, j])

cdef void update_filter3(complex_all[:,:] E, int Ntaps, floating_all mu, complex_all err,
                        complex_all[:,:] wx, int modes):
    cdef int i,j
    for k in range(modes):
        for j in range(Ntaps):
                wx[k, j] -= mu * err * (E[k, j].real - E[k,j].imag)

def testupdate1(double complex[:,:] E, int Ntaps, double mu, double complex err,
                double complex[:,:] wx):
    cdef int L = E.shape[1]
    cdef int pols = E.shape[0]
    update_filter(&E[0,0], Ntaps, mu, err, &wx[0,0], pols, L)

def testupdate2(double complex[:,:] E, int Ntaps, double mu, double complex err,
                double complex[:,:] wx):
    cdef int pols = E.shape[0]
    update_filter2(E, Ntaps, mu, err, wx, pols)

def testupdate3(double complex[:,:] E, int Ntaps, double mu, double complex err,
                double complex[:,:] wx):
    cdef int pols = E.shape[0]
    update_filter3(E, Ntaps, mu, err, wx, pols)

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
        Xest = apply_filter3(E[:, i*os:], Ntaps, wx, pols)
        err[i] = errfct.calc_error(Xest)
        #update_filter(&E[0,i*os], Ntaps, mu, err[i], &wx[0,0], pols, L)
        update_filter2(E[:, i*os:], Ntaps, mu, err[i], wx, pols)
        if adaptive and i > 0:
            mu = adapt_step(mu, err[i-1], err[i])
    return err, wx
