# cython: profile=False, boundscheck=False, wraparound=False
from __future__ import division
import numpy as np
from cython.parallel import prange
cimport cython
cimport numpy as np
from ccomplex cimport *
from .equalisation cimport cython_equalisation
from cmath cimport exp, log, pow


cdef double cabssq(cython_equalisation.complexing x) nogil:
    if cython_equalisation.complexing is cython_equalisation.complex64_t:
        return cimagf(x)*cimagf(x) + crealf(x)*crealf(x)
    else:
        return cimag(x)*cimag(x) + creal(x)*creal(x)

def unwrap_discont(double[::1] seq, double max_diff, double period):
    cdef int i
    cdef double diff
    cdef long nperiods = 0
    cdef np.ndarray[ndim=1, dtype=double] new_array = np.zeros(seq.shape[0], dtype=np.float)
    new_array[0] = seq[0]
    for i in range(1, seq.shape[0]):
        diff = seq[i]-new_array[i-1]
        if diff > period:
            nperiods -= 1
        elif diff > max_diff:
            new_array[i] = new_array[i-1]
            continue
        elif diff < -period:
            nperiods += 1
        elif diff < -max_diff:
            new_array[i] = new_array[i-1]
            continue
        new_array[i] = seq[i] + period * nperiods
    return new_array

def bps(cython_equalisation.complexing[:] E, cython.floating[:,:] testangles, cython_equalisation.complexing[:] symbols, int N):
    cdef ssize_t i, j, ph_idx
    cdef int L = E.shape[0]
    cdef int p = testangles.shape[0]
    cdef int M = symbols.shape[0]
    cdef int Ntestangles = testangles.shape[1]
    cdef float x1 = 1
    cdef double x2 = 1
    cdef np.ndarray[ndim=1, dtype=ssize_t] idx
    cdef np.ndarray[ndim=2, dtype=cython_equalisation.complexing] comp_angles
    cdef cython.floating[:,:] dists
    cdef cython.floating dtmp = 0.
    cdef cython_equalisation.complexing s = 0
    cdef cython_equalisation.complexing tmp = 0
    cdtype = "c%d"%E.itemsize
    fdtype = "f%d"%testangles.itemsize
    comp_angles = np.zeros((p, Ntestangles), dtype=cdtype)
    comp_angles[:,:] = np.exp(1.j*np.array(testangles[:,:]))
    dists = np.zeros((L, Ntestangles), dtype=fdtype)+100.
    for i in prange(L, schedule='static', nogil=True):
        if testangles.shape[0] > 1:
            ph_idx = i
        else:
            ph_idx = 0
        for j in range(Ntestangles):
            tmp = E[i] * comp_angles[ph_idx, j]
            s = cython_equalisation.det_symbol(symbols, M, tmp, &dtmp)
            if dtmp < dists[i, j]:
                dists[i, j] = dtmp
    return np.array(select_angle_index(dists, 2*N))

cpdef ssize_t[:] select_angle_index(cython.floating[:,:] x, int N):
    cdef cython.floating[:,:] csum
    cdef ssize_t[:] idx
    cdef ssize_t i,k, L, M
    cdef cython.floating dmin, dtmp
    L = x.shape[0]
    M = x.shape[1]
    csum = np.zeros((L,M), dtype="f%d"%x.itemsize)
    idx = np.zeros(L, dtype=np.intp)
    for i in range(1, L-1):
        dmin = 1000.
        if i < N:
            for k in range(M):
                csum[i,k] = csum[i,k]+x[i,k]
        else:
            for k in range(M):
                csum[i,k] = csum[i-1,k]+x[i,k]
                dtmp = csum[i,k]  - csum[i-N,k]
                if dtmp < dmin:
                    idx[i-N//2] = k
                    dmin = dtmp
    return idx

cpdef prbs_ext(np.int64_t seed, taps, int nbits, int N):
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

cpdef prbs_int(np.int64_t seed, np.int64_t mask, int N):
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

cpdef soft_l_value_demapper(cython_equalisation.complexing[:] rx_symbs, int M, double snr, cython_equalisation.complexing[:,:,:] bits_map):
    cdef int num_bits = int(np.log2(M))
    cdef double[:] L_values = np.zeros(rx_symbs.shape[0]*num_bits)
    cdef int mode, bit, symb, l
    cdef int N = rx_symbs.shape[0]
    cdef int k = bits_map.shape[1]
    cdef double tmp = 0
    cdef double tmp2 = 0

    for bit in range(num_bits):
        for symb in prange(N, schedule='static', nogil=True):
            tmp = 0
            tmp2 = 0
            for l in range(k):
                tmp = tmp + exp(-snr*cabssq(bits_map[bit,l,1] - rx_symbs[symb]))
                tmp2 = tmp2 + exp(-snr*cabssq(bits_map[bit,l,0] - rx_symbs[symb]))
            L_values[symb*num_bits + bit] = log(tmp) - log(tmp2)
    return L_values

