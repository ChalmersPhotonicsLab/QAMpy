from __future__ import division
import numpy as np
from segmentaxis import segment_axis
cimport cython
cimport numpy as np


def FS_CMA_training(np.ndarray[ndim=2, dtype=np.complex128_t] E,
    int TrSyms, int Ntaps, unsigned int os, double mu,  np.ndarray[ndim=2, dtype=np.complex128_t] wx):
    cdef np.ndarray[ndim=1, dtype=np.float64_t] err = np.zeros(TrSyms, dtype=np.float64)
    cdef np.ndarray[ndim=2, dtype=np.complex128_t] X = np.zeros([2,Ntaps], dtype=np.complex128)
    cdef unsigned int i, j, k
    cdef np.complex128_t Xest
    for i in range(0, TrSyms):
       Xest = 0.j
       for j in range(Ntaps):
           for k in range(2):
               X[<unsigned int> k, <unsigned int> j] = E[<unsigned int> k,
                       <unsigned int> i*os+j]
               Xest = Xest + wx[<unsigned int> k,<unsigned int> j]*X[<unsigned int>
                       k,<unsigned int> j]
       err[<unsigned int> i] = abs(Xest)-1
       for j in range(Ntaps):
           for k in range(2):
               wx[<unsigned int> k,<unsigned int> j] = wx[<unsigned int>
                       k,<unsigned int> j]-mu*err[<unsigned int>
                               i]*Xest*X[<unsigned int> k,
                                   <unsigned int> j].conjugate()
    return err, wx

def partition_value(double signal, np.ndarray[ndim=1, dtype=np.float64_t] partitions,
        np.ndarray[ndim=1, dtype=np.float64_t] codebook):
    cdef unsigned int index = 0
    cdef unsigned int L = len(partitions)
    while index < L and signal > partitions[index]:
        index += 1
    return codebook[index]

def FS_RDE_training(np.ndarray[ndim=2, dtype=np.complex128_t] E,
        int TrCMA, int TrRDE, int Ntaps, unsigned int os,
        double muRDE,
        np.ndarray[ndim=2, dtype=np.complex128_t] wx, 
        np.ndarray[ndim=1, dtype=np.float64_t] partition, 
        np.ndarray[ndim=1, dtype=np.float64_t] codebook):
    cdef np.ndarray[ndim=1, dtype=np.float64_t] err = np.zeros(TrRDE, dtype=np.float64)
    cdef np.ndarray[ndim=2, dtype=np.complex128_t] X = np.zeros([2,Ntaps], dtype=np.complex128)
    cdef unsigned int i, j, k
    cdef np.complex128_t Xest
    cdef np.float64_t Ssq, S_DD
    for i in range(TrCMA, TrRDE):
       Xest = 0.j
       for j in range(Ntaps):
           for k in range(2):
               X[<unsigned int> k, <unsigned int> j] = E[<unsigned int> k,
                       <unsigned int> i*os+j]
               Xest = Xest + wx[<unsigned int> k,<unsigned int> j]*X[<unsigned int>
                       k,<unsigned int> j]
       Ssq = abs(Xest)**2
       S_DD = partition_value(Ssq, partition, codebook)
       err[<unsigned int> i - TrCMA] = S_DD-Ssq
       for j in range(Ntaps):
           for k in range(2):
               wx[<unsigned int> k,<unsigned int> j] = wx[<unsigned int>
                       k,<unsigned int> j]+muRDE*err[<unsigned int>
                               i]*Xest*X[<unsigned int> k,
                                   <unsigned int> j].conjugate()
    return err, wx
 

def lfsr_ext(int seed, taps, int nbits):
    """A Fibonacci or external XOR linear feedback shift register.

    Parameters:
        seed  -- binary number denoting the state registers
        taps  -- list of registers that are input to the XOR
        nbits -- number of registers

    yields (xor, state) where xor is the output of the registers and state is
    the register state at every step
    """
    cdef int sr, t
    cdef int xor
    sr = seed
    while 1:
        xor = 0
        for t in taps:
            if (sr & (1<<(nbits-t))) != 0:
                xor ^= 1
        sr = (xor << nbits-1) + (sr >> 1)
        yield xor, sr

def lfsr_int(int seed, int mask):
    """A linear feedback shift register, using Galois or internal XOR implementation,
    the seed is a binary number with the bit length N. The mask determines
    the polynomial and has length N+1, also the first and last bit of the
    mask need to be 1.
    Returns a generator which yields the bit and the register state"""
    cdef int state = seed
    cdef int nbits = mask.bit_length()-1
    cdef int xor
    while True:
        state = (state << 1)
        xor = state >> nbits
        #the modulus operation on has an effect if the last bit is 1
        if xor != 0:
            state ^= mask #this performs the modulus operation
        yield xor, state
 
