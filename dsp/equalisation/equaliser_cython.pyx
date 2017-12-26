# cython: profile=True, boundscheck=False, wraparound=False
from __future__ import division
import numpy as np
from cython.view cimport array as cvarray
from cython.parallel import prange
cimport cython
from cpython cimport bool
cimport numpy as np

cdef extern from "complex.h":
    double complex conj(double complex)

cdef extern from "equaliserC.h":
    double complex apply_filter(double complex *E, unsigned int Ntaps, double complex *wx, unsigned int pols, unsigned int L)

cdef extern from "equaliserC.h" nogil:
    double complex det_symbol(double complex *syms, unsigned int M, double complex value, double *distout)

cdef extern from "equaliserC.h":
    void update_filter(double complex *E, unsigned int Ntaps, double mu, double complex err, double complex *wx, unsigned int pols, unsigned int L)


def quantize(np.ndarray[ndim=1, dtype=np.complex128_t] E, np.ndarray[ndim=1, dtype=np.complex128_t] symbols):
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

def mrde_err(a, b):
    cdef np.float64_t[:] partition_real = a.real
    cdef np.float64_t[:] partition_imag = a.imag
    cdef np.float64_t[:] codebook_real = b.real
    cdef np.float64_t[:] codebook_imag = b.imag
    #cdef complex double[:] codebook = a
    def mrde_fct(double complex Xest):
        cdef double complex Ssq
        cdef double complex S_DD
        #pp = np.asarray(partition)
        #cc = np.asarray(codebook)
        Ssq = Xest.real**2 + 1.j * Xest.imag**2
        S_DD = partition_value(Ssq.real, partition_real, codebook_real) + 1.j * partition_value(Ssq.imag, partition_imag, codebook_imag)
        return (Ssq.real - S_DD.real)*Xest.real + 1.j*(Ssq.imag - S_DD.imag)*Xest.imag
    return mrde_fct

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

def FS_CMA(np.ndarray[ndim=2, dtype=np.complex128_t] E,
                    int TrSyms,
                    int Ntaps,
                    unsigned int os,
                    double mu,
                    np.ndarray[ndim=2, dtype=np.complex128_t] wx,
                    double R,
                    bool adaptive=False):
    cdef np.ndarray[ndim=1, dtype=np.complex128_t] err = np.zeros(TrSyms, dtype=np.complex128)
    cdef unsigned int i, j, k
    cdef unsigned int pols = E.shape[0]
    cdef unsigned int L = E.shape[1]
    cdef np.complex128_t Xest
    for i in range(0, TrSyms):
        Xest = apply_filter(&E[0, i*os], Ntaps, &wx[0,0], pols, L)
        err[i] = Xest*(Xest.real**2+Xest.imag**2 - R)
        update_filter(&E[0,i*os], Ntaps, mu, err[i], &wx[0,0], pols, L)
        if adaptive and i > 0:
            mu = adapt_step(mu, err[i-1], err[i])
    return err, wx

cdef class ErrFct:
    cpdef double complex calc_error(self, double complex Xest)  except *:
        return 0

cdef class MCMAErr(ErrFct):
    cdef double complex R
    def __cinit__(self, double complex R):
        self.R = R
    def __init__(self, *args, **kwargs):
        pass
    cpdef double complex calc_error(self, double complex Xest)  except *:
        return (Xest.real**2 - self.R.real)*Xest.real + 1.j*(Xest.imag**2 -self.R.imag)*Xest.imag

cdef class MRDEErr(ErrFct):
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
        Ssq = Xest.real**2 + 1.j * Xest.imag**2
        S_DD = partition_value(Ssq.real, self.partition_real, self.codebook_real) + 1.j * partition_value(Ssq.imag, self.partition_imag, self.codebook_imag)
        return (Ssq.real - S_DD.real)*Xest.real + 1.j*(Ssq.imag - S_DD.imag)*Xest.imag


def mcma_err(double complex R):
    def mcma_fct(double complex Xest):
        return (Xest.real**2 - R.real)*Xest.real + 1.j*(Xest.imag**2 -R.imag)*Xest.imag
    return mcma_fct

ctypedef double complex (*cfct) (double complex)

cdef double complex R
cdef mcma_fct2(double complex Xest):
    global R
    return (Xest.real**2 - R.real)*Xest.real + 1.j*(Xest.imag**2 -R.imag)*Xest.imag

def select_err(method, args):
    if method == "mcma":
        global R
        R = args['R']
        #return mcma_err(args['R'])
        #return [mcma_fct2]
        return MCMAErr(args['R'])
    elif method == "mrde":
        #return mrde_err(args['partition'], args['codebook'])
        return MRDEErr(args['partition'], args['codebook'])
    else:
        return mcma_err(args['R'])

def generic_eq(np.ndarray[ndim=2, dtype=np.complex128_t] E,
                    int TrSyms,
                    int Ntaps,
                    unsigned int os,
                    double mu,
                    np.ndarray[ndim=2, dtype=np.complex128_t] wx,
                    ErrFct errfct,
                    #object errfct,
                    bool adaptive=False):
    cdef np.ndarray[ndim=1, dtype=np.complex128_t] err = np.zeros(TrSyms, dtype=np.complex128)
    cdef unsigned int i, j, k
    cdef unsigned int pols = E.shape[0]
    cdef unsigned int L = E.shape[1]
    cdef np.complex128_t Xest

    for i in range(0, TrSyms):
        Xest = apply_filter(&E[0, i*os], Ntaps, &wx[0,0], pols, L)
        err[i] = errfct.calc_error(Xest)
        #err[i] = errfct2(Xest)
        update_filter(&E[0,i*os], Ntaps, mu, err[i], &wx[0,0], pols, L)
        if adaptive and i > 0:
            mu = adapt_step(mu, err[i-1], err[i])
    return err, wx



def FS_MCMA(np.ndarray[ndim=2, dtype=np.complex128_t] E,
                     int TrSyms,
                     int Ntaps,
                     unsigned int os,
                     double mu,
                     np.ndarray[ndim=2, dtype=np.complex128_t] wx,
                     np.complex128_t R,
                    bool adaptive=False):
    cdef np.ndarray[ndim=1, dtype=np.complex128_t] err = np.zeros(TrSyms, dtype=np.complex128)
    cdef unsigned int i, j, k
    cdef unsigned int pols = E.shape[0]
    cdef unsigned int L = E.shape[1]
    cdef np.complex128_t Xest
    for i in range(0, TrSyms):
        Xest = apply_filter(&E[0, i*os], Ntaps, &wx[0,0], pols, L)
        err[i] = (Xest.real**2 - R.real)*Xest.real + 1.j*(Xest.imag**2 - R.imag)*Xest.imag
        update_filter(&E[0,i*os], Ntaps, mu, err[i], &wx[0,0], pols, L)
        if adaptive and i > 0:
            mu = adapt_step(mu, err[i-1], err[i])
    return err, wx

def FS_RDE(np.ndarray[ndim=2, dtype=np.complex128_t] E,
                    int TrSyms,
                    int Ntaps,
                    unsigned int os,
                    double mu,
                    np.ndarray[ndim=2, dtype=np.complex128_t] wx,
                    np.ndarray[ndim=1, dtype=np.float64_t] partition,
                    np.ndarray[ndim=1, dtype=np.float64_t] codebook,
                    bool adaptive=False):
    cdef np.ndarray[ndim=1, dtype=np.complex128_t] err = np.zeros(TrSyms, dtype=np.complex128)
    cdef unsigned int i, j, k
    cdef unsigned int pols = E.shape[0]
    cdef unsigned int L = E.shape[1]
    cdef double complex Xest
    cdef double Ssq, S_DD
    for i in range(TrSyms):
        Xest = apply_filter(&E[0, i*os], Ntaps, &wx[0,0], pols, L)
        Ssq = Xest.real**2 + Xest.imag**2
        S_DD = partition_value(Ssq, partition, codebook)
        err[i] = Xest*(Ssq - S_DD)
        update_filter(&E[0,i*os], Ntaps, mu, err[i], &wx[0,0], pols, L)
        if adaptive and i > 0:
            mu = adapt_step(mu, err[i-1], err[i])
    return err, wx

def FS_MRDE(np.ndarray[ndim=2, dtype=np.complex128_t] E,
                     int TrSyms, int Ntaps, unsigned int os,
                     double mu,
                     np.ndarray[ndim=2, dtype=np.complex128_t] wx,
                     np.ndarray[ndim=1, dtype=np.complex128_t] partition,
                     np.ndarray[ndim=1, dtype=np.complex128_t] codebook,
                    bool adaptive=False):
    cdef np.ndarray[ndim=1, dtype=np.complex128_t] err = np.zeros(TrSyms, dtype=np.complex128)
    cdef unsigned int i, j, k
    cdef unsigned int pols = E.shape[0]
    cdef unsigned int L = E.shape[1]
    cdef np.complex128_t Xest, Ssq, S_DD
    for i in range(TrSyms):
        Xest = apply_filter(&E[0, i*os], Ntaps, &wx[0,0], pols, L)
        Ssq = Xest.real**2 + 1.j * Xest.imag**2
        S_DD = partition_value(Ssq.real, partition.real, codebook.real) + 1.j * partition_value(Ssq.imag, partition.imag, codebook.imag)
        err[i] = (Ssq.real - S_DD.real)*Xest.real + 1.j*(Ssq.imag - S_DD.imag)*Xest.imag
        update_filter(&E[0,i*os], Ntaps, mu, err[i], &wx[0,0], pols, L)
        if adaptive and i > 0:
            mu = adapt_step(mu, err[i-1], err[i])
    return err, wx

def FS_SBD(np.ndarray[ndim=2, dtype=np.complex128_t] E,
                     int TrSyms, int Ntaps, unsigned int os,
                     double mu,
                     np.ndarray[ndim=2, dtype=np.complex128_t] wx,
                     np.ndarray[ndim=1, dtype=np.complex128_t] symbols,
                     bool adaptive=False):
    cdef np.ndarray[ndim=1, dtype=np.complex128_t] err = np.zeros(TrSyms, dtype=np.complex128)
    cdef unsigned int i, j, k, N
    cdef double complex Xest, R
    cdef unsigned int pols = E.shape[0]
    cdef unsigned int M = len(symbols)
    cdef double lm, dist
    cdef unsigned int L = E.shape[1]
    for i in range(TrSyms):
        Xest = apply_filter(&E[0, i*os], Ntaps, &wx[0,0], pols, L)
        R = det_symbol(&symbols[0], M, Xest, &dist)
        err[i] = (Xest.real - R.real)*abs(R.real) + 1.j*(Xest.imag - R.imag)*abs(R.imag)
        update_filter(&E[0,i*os], Ntaps, mu, err[i], &wx[0,0], pols, L)
        if adaptive and i > 0:
            mu = adapt_step(mu, err[i-1], err[i])
    return err, wx

def FS_MDDMA(np.ndarray[ndim=2, dtype=np.complex128_t] E,
                     int TrSyms, int Ntaps, unsigned int os,
                     double mu,
                     np.ndarray[ndim=2, dtype=np.complex128_t] wx,
                     np.ndarray[ndim=1, dtype=np.complex128_t] symbols,
                     bool adaptive=False):
    cdef np.ndarray[ndim=1, dtype=np.complex128_t] err = np.zeros(TrSyms, dtype=np.complex128)
    cdef unsigned int i, j, k
    cdef np.complex128_t Xest, R
    cdef unsigned int M = len(symbols)
    cdef unsigned int pols = E.shape[0]
    cdef unsigned int L = E.shape[1]
    cdef double dist
    for i in range(TrSyms):
        Xest = apply_filter(&E[0, i*os], Ntaps, &wx[0,0], pols, L)
        R = det_symbol(&symbols[0], M, Xest, &dist)
        err[i] = (Xest.real**2 - R.real**2)*Xest.real + 1.j*(Xest.imag**2 - R.imag**2)*Xest.imag
        update_filter(&E[0,i*os], Ntaps, mu, err[i], &wx[0,0], pols, L)
        if adaptive and i > 0:
            mu = adapt_step(mu, err[i-1], err[i])
    return err, wx

def FS_DD(np.ndarray[ndim=2, dtype=np.complex128_t] E,
                     int TrSyms, int Ntaps, unsigned int os,
                     double mu,
                     np.ndarray[ndim=2, dtype=np.complex128_t] wx,
                     np.ndarray[ndim=1, dtype=np.complex128_t] symbols,
                     bool adaptive=False):
    cdef np.ndarray[ndim=1, dtype=np.complex128_t] err = np.zeros(TrSyms, dtype=np.complex128)
    cdef unsigned int i, j, k
    cdef np.complex128_t Xest, R
    cdef unsigned int M = len(symbols)
    cdef unsigned int pols = E.shape[0]
    cdef unsigned int L = E.shape[1]
    cdef double dist
    for i in range(TrSyms):
        Xest = apply_filter(&E[0, i*os], Ntaps, &wx[0,0], pols, L)
        R = det_symbol(&symbols[0], M, Xest, &dist)
        err[i] = (Xest - R)
        update_filter(&E[0,i*os], Ntaps, mu, err[i], &wx[0,0], pols, L)
        if adaptive and i > 0:
            mu = adapt_step(mu, err[i-1], err[i])
    return err, wx
