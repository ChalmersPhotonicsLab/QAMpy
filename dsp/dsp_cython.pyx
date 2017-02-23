from __future__ import division
import numpy as np
cimport cython
from cpython cimport bool
cimport numpy as np


def gen_unwrap(double[::1] seq, double max_diff, double period):
    cdef int i
    cdef double diff
    cdef long nperiods = 0
    cdef np.ndarray[ndim=1, dtype=double] new_array = np.zeros(seq.shape[0], dtype=np.float)
    new_array[0] = seq[0]
    for i in range(1, seq.shape[0]):
        diff = seq[i]-seq[i-1]
        if diff > period:
            nperiods -= 1
        elif diff > max_diff:
            new_array[i] = seq[i-1]
            continue
        elif diff < -period:
            nperiods += 1
        elif diff < -max_diff:
            new_array[i] = seq[i-1]
            continue
        new_array[i] = seq[i] + period * nperiods
    return new_array

def prbs_ext(np.int64_t seed, taps, int nbits, int N):
    cdef int t
    cdef np.int64_t xor, sr
    cdef np.ndarray[ndim=1, dtype=np.uint8_t] out = np.zeros(N, dtype=np.uint8)
    sr = seed
    for i in range(N):
        xor = 0
        for t in taps:
            if (sr & (1<<(nbits-t))) != 0:
                xor ^= 1
        sr = (xor << nbits-1) + (sr >> 1)
        out[i] = xor
    return out

def lfsr_ext(np.int64_t seed, taps, int nbits):
    """A Fibonacci or external XOR linear feedback shift register.

    Parameters:
        seed  -- binary number denoting the state registers
        taps  -- list of registers that are input to the XOR
        nbits -- number of registers

    yields (xor, state) where xor is the output of the registers and state is
    the register state at every step
    """
    cdef int  t
    cdef np.int64_t xor, sr
    sr = seed
    while 1:
        xor = 0
        for t in taps:
            if (sr & (1<<(nbits-t))) != 0:
                xor ^= 1
        sr = (xor << nbits-1) + (sr >> 1)
        yield xor, sr

def prbs_int(np.int64_t seed, np.int64_t mask, int N):
    cdef np.ndarray[ndim=1, dtype=np.uint8_t] out = np.zeros(N, dtype=np.uint8)
    cdef np.int64_t state = seed
    cdef int nbits = mask.bit_length()-1
    cdef np.int64_t xor
    for i in range(N):
        state = (state << 1)
        xor = state >> nbits
        if xor != 0:
            state ^= mask #this performs the modulus operation
        out[i] = xor
    return out

def lfsr_int(np.int64_t seed, np.int64_t mask):
    """A linear feedback shift register, using Galois or internal XOR implementation,
    the seed is a binary number with the bit length N. The mask determines
    the polynomial and has length N+1, also the first and last bit of the
    mask need to be 1.
    Returns a generator which yields the bit and the register state"""
    cdef np.int64_t state = seed
    cdef int nbits = mask.bit_length()-1
    cdef np.int64_t xor
    while True:
        state = (state << 1)
        xor = state >> nbits
        #the modulus operation on has an effect if the last bit is 1
        if xor != 0:
            state ^= mask #this performs the modulus operation
        yield xor, state
