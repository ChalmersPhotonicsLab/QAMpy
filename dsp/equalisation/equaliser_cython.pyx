# cython: profile=True, boundscheck=False, wraparound=False
from __future__ import division
import numpy as np
cimport cython
cimport numpy as np

def partition_value(double signal,
                    np.ndarray[ndim=1, dtype=np.float64_t] partitions,
                    np.ndarray[ndim=1, dtype=np.float64_t] codebook):
    cdef unsigned int index = 0
    cdef unsigned int L = len(partitions)
    while index < L and signal > partitions[index]:
        index += 1
    return codebook[index]

def FS_CMA_training(np.ndarray[ndim=2, dtype=np.complex128_t] E,
                    int TrSyms,
                    int Ntaps,
                    unsigned int os,
                    double mu,
                    np.ndarray[ndim=2, dtype=np.complex128_t] wx,
                    double R):
    cdef np.ndarray[ndim=1, dtype=np.complex128_t] err = np.zeros(TrSyms, dtype=np.complex128)
    cdef np.ndarray[ndim=2, dtype=np.complex128_t] X = np.zeros([2,Ntaps], dtype=np.complex128)
    cdef unsigned int i, j, k
    cdef np.complex128_t Xest
    for i in range(0, TrSyms):
       Xest = 0.j
       for j in range(Ntaps):
           for k in range(2):
               X[<unsigned int> k, <unsigned int> j] = E[<unsigned int> k,
                                                         <unsigned int> i*os+j]
               Xest +=  wx[<unsigned int> k,<unsigned int> j]*X[<unsigned int>
                                                                k,<unsigned int> j]
               err[<unsigned int> i] = (Xest.real**2+Xest.imag**2 - R)*Xest
       for j in range(Ntaps):
           for k in range(2):
               wx[<unsigned int> k,<unsigned int> j] = wx[<unsigned int>
                                                          k,<unsigned int> j]-mu*err[<unsigned int>
                                                                                     i]*X[<unsigned int> k,
                                                                                          <unsigned int> j].conjugate()
    return err, wx

def FS_MCMA_training(np.ndarray[ndim=2, dtype=np.complex128_t] E,
                     int TrSyms,
                     int Ntaps,
                     unsigned int os,
                     double mu,
                     np.ndarray[ndim=2, dtype=np.complex128_t] wx,
                     np.complex128_t R):
    cdef np.ndarray[ndim=1, dtype=np.complex128_t] err = np.zeros(TrSyms, dtype=np.complex128)
    cdef np.ndarray[ndim=2, dtype=np.complex128_t] X = np.zeros([2,Ntaps], dtype=np.complex128)
    cdef unsigned int i, j, k
    cdef np.complex128_t Xest
    for i in range(0, TrSyms):
       Xest = 0.j
       for j in range(Ntaps):
           for k in range(2):
               X[<unsigned int> k, <unsigned int> j] = E[<unsigned int> k,
                       <unsigned int> i*os+j]
               Xest +=  wx[<unsigned int> k,<unsigned int> j]*X[<unsigned int>
                       k,<unsigned int> j]
       err[<unsigned int> i] = (Xest.real**2 - R.real)*Xest.real + 1.j*(Xest.imag**2 - R.imag)*Xest.imag
       for j in range(Ntaps):
           for k in range(2):
               wx[<unsigned int> k,<unsigned int> j] = wx[<unsigned int>
                       k,<unsigned int> j]-mu*err[<unsigned int>
                               i]*X[<unsigned int> k,
                                   <unsigned int> j].conjugate()
    return err, wx

def FS_RDE_training(np.ndarray[ndim=2, dtype=np.complex128_t] E,
                    int TrSyms,
                    int Ntaps,
                    unsigned int os,
                    double mu,
                    np.ndarray[ndim=2, dtype=np.complex128_t] wx,
                    np.ndarray[ndim=1, dtype=np.float64_t] partition,
                    np.ndarray[ndim=1, dtype=np.float64_t] codebook):
    cdef np.ndarray[ndim=1, dtype=np.complex128_t] err = np.zeros(TrSyms, dtype=np.complex128)
    cdef np.ndarray[ndim=2, dtype=np.complex128_t] X = np.zeros([2,Ntaps], dtype=np.complex128)
    cdef unsigned int i, j, k
    cdef np.complex128_t Xest
    cdef np.float64_t Ssq, S_DD
    for i in range(TrSyms):
       Xest = 0.j
       for j in range(Ntaps):
           for k in range(2):
               X[<unsigned int> k, <unsigned int> j] = E[<unsigned int> k,
                       <unsigned int> i*os+j]
               Xest += wx[<unsigned int> k,<unsigned int> j]*X[<unsigned int>
                       k,<unsigned int> j]
       Ssq = Xest.real**2 + Xest.imag**2
       S_DD = partition_value(Ssq, partition, codebook)
       err[<unsigned int> i] = (Ssq - S_DD)*Xest
       for j in range(Ntaps):
           for k in range(2):
               wx[<unsigned int> k,<unsigned int> j] = wx[<unsigned int>
                       k,<unsigned int> j] - mu * err[<unsigned int>
                               i]*X[<unsigned int> k,
                                   <unsigned int> j].conjugate()
    return err, wx

def FS_MRDE_training(np.ndarray[ndim=2, dtype=np.complex128_t] E,
                     int TrSyms, int Ntaps, unsigned int os,
                     double mu,
                     np.ndarray[ndim=2, dtype=np.complex128_t] wx,
                     np.ndarray[ndim=1, dtype=np.complex128_t] partition,
                     np.ndarray[ndim=1, dtype=np.complex128_t] codebook):
    cdef np.ndarray[ndim=1, dtype=np.complex128_t] err = np.zeros(TrSyms, dtype=np.complex128)
    cdef np.ndarray[ndim=2, dtype=np.complex128_t] X = np.zeros([2,Ntaps], dtype=np.complex128)
    cdef unsigned int i, j, k
    cdef np.complex128_t Xest, Ssq, S_DD
    for i in range(TrSyms):
       Xest = 0.j
       for j in range(Ntaps):
           for k in range(2):
               X[<unsigned int> k, <unsigned int> j] = E[<unsigned int> k,
                       <unsigned int> i*os+j]
               Xest += wx[<unsigned int> k,<unsigned int> j]*X[<unsigned int>
                       k,<unsigned int> j]
       Ssq = Xest.real**2 + 1.j * Xest.imag**2
       S_DD = partition_value(Ssq.real, partition.real, codebook.real) + 1.j * partition_value(Ssq.imag, partition.imag, codebook.imag)
       err[<unsigned int> i] = (Ssq.real - S_DD.real)*Xest.real + 1.j*(Ssq.imag - S_DD.imag)*Xest.imag
       for j in range(Ntaps):
           for k in range(2):
               wx[<unsigned int> k,<unsigned int> j] = wx[<unsigned int>
                       k,<unsigned int> j] - mu*err[<unsigned int>
                               i]*X[<unsigned int> k,
                                   <unsigned int> j].conjugate()
       #update_filter(X, Ntaps, mu, err[<unsigned int> i], wx)
    return err, wx

cdef inline np.ndarray[ndim=2, dtype=np.complex128_t] update_filter(np.ndarray[ndim=2, dtype=np.complex128_t] X,
                                                             int Ntaps,
                                                             double mu,
                                                             np.complex128_t err,
                                                             np.ndarray[ndim=2, dtype=np.complex128_t] wx):
    cdef unsigned int i, j, k
    for j in range(Ntaps):
        for k in range(2):
            wx[<unsigned int> k,<unsigned int> j] = wx[<unsigned int>
                                                       k,<unsigned int> j] - mu*err*X[<unsigned int> k,
                                                                                            <unsigned int> j].conjugate()
    return wx
