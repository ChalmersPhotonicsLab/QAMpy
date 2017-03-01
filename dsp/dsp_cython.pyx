# cython: profile=True, boundscheck=False, wraparound=False
from __future__ import division
import numpy as np
from cython.parallel import prange
cimport cython
from cpython cimport bool
cimport numpy as np

cdef extern from "complex.h":
    double complex conj(double complex)

cdef extern from "equaliserC.h" nogil:
    double complex det_symbol(double complex *syms, unsigned int M, double complex value, double *distout)

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

def bps3(np.ndarray[ndim=1, dtype=np.complex128_t] E, int Mtestangles, np.ndarray[ndim=1, dtype=np.complex128_t] symbols, int N):
    cdef unsigned int i, j, k, M, l, o
    cdef np.ndarray[ndim=1, dtype=np.float64_t] angles
    cdef np.ndarray[ndim=1, dtype=np.uint32_t] idx
    cdef np.ndarray[ndim=1, dtype=np.float64_t] angles_out
    cdef np.ndarray[ndim=2, dtype=np.float64_t] dists
    #cdef np.ndarray[ndim=2, dtype=np.float64_t] distn
    cdef np.ndarray[ndim=1, dtype=np.complex128_t] pp, En
    cdef double dtmp, dmin,
    cdef np.complex128_t dd, s
    angles = np.linspace(-np.pi/4, np.pi/4, Mtestangles, endpoint=False)
    pp = np.exp(1.j*angles)
    L = E.shape[0]
    dists = np.zeros((L, Mtestangles))+100.
    #distn = np.zeros(L, Mtestangeles)
    angles_out = np.zeros(E.shape[0], dtype=np.float64)
    idx = np.zeros(L, dtype=np.uint32)
    M = symbols.shape[0]
    for i in prange(L, schedule='static', num_threads=8, nogil=True):
        #for j in prange(Mtestangles):
        for j in range(Mtestangles):
            #if i < N:
            s = det_symbol(&symbols[0], M, E[i]*pp[j], &dtmp)
            if dtmp < dists[<unsigned int>i,<unsigned int>j]:
                dists[<unsigned int>i, <unsigned int>j] = dtmp
            #else:
            #dists = np.roll(dists, -1, 0)
                #o = N - i%N
                #for k in range(M):
                    #dd = E[<unsigned int>i]*pp[<unsigned int>j] - symbols[k]
                    #dtmp = dd.real*dd.real + dd.imag*dd.imag
                    #if dtmp < dists[<unsigned int>o,<unsigned int>j]:
                        #dists[<unsigned int>o, <unsigned int>j] = dtmp
                #for l in range(N):
                    #distn[j] += dists[<unsigned int>l, <unsigned int>j]
                #if distn[j] < dmin:
                    #idx[<unsigned int>i-N//2] = j
                    #dmin = distn[<unsigned int>j]
                #angles_out[<unsigned int>i-N//2] = angles[idx[<unsigned int>i-N//2]]
    idx = avg_win3(dists, 2*N)
    #distn = avg_win2(dists, 2*N)
    #idx[N:-N] = distn.argmin(axis=1)
    #idx[:] = distn.argmin(axis=1)
    angles_out = angles[idx]
    angles_out = np.unwrap(angles_out*4, discont=np.pi)/4
    En = E*np.exp(1.j*angles_out)
    return En, angles_out, idx

def avg_win2(np.ndarray[ndim=2, dtype=np.float64_t] x, int N):
    cdef np.ndarray[ndim=2, dtype=np.float64_t] output, csum
    cdef int i,k, L, M
    L = x.shape[0]
    M = x.shape[1]
    csum = np.zeros((L,M))
    output = np.zeros((L,M))
    for i in range(1, L-1):
        if i < N:
            for k in range(x.shape[1]):
                csum[i-1,k] = csum[i-1,k]+x[i,k]
        if i >= N:
            for k in range(x.shape[1]):
                csum[i,k] = csum[i-1,k]+x[i,k]
                output[i,k] = csum[i,k]  - csum[i-N,k]
    return output

def avg_win3(np.ndarray[ndim=2, dtype=np.float64_t] x, int N):
    cdef np.ndarray[ndim=2, dtype=np.float64_t] csum
    cdef np.ndarray[ndim=1, dtype=np.uint32_t] idx
    cdef int i,k, L, M
    cdef double dmin, dtmp
    L = x.shape[0]
    M = x.shape[1]
    csum = np.zeros((L,M))
    idx = np.zeros(L, dtype=np.uint32)
    output = np.zeros((L,M))
    for i in range(1, L-1):
        dmin = 1000.
        if i < N:
            for k in range(x.shape[1]):
                csum[i-1,k] = csum[i-1,k]+x[i,k]
        if i >= N:
            for k in range(x.shape[1]):
                csum[i,k] = csum[i-1,k]+x[i,k]
                dtmp = csum[i,k]  - csum[i-N,k]
                if dtmp < dmin:
                    idx[i-N//2] = k
                    dmin = dtmp
    return idx



def bps(np.ndarray[ndim=1, dtype=np.complex128_t] E, int Mtestangles, np.ndarray[ndim=1, dtype=np.complex128_t] symbols, int N):
    cdef unsigned int i, j, k, M, l, o
    cdef np.ndarray[ndim=1, dtype=np.float64_t] angles
    cdef np.ndarray[ndim=1, dtype=np.uint32_t] idx
    cdef np.ndarray[ndim=1, dtype=np.float64_t] angles_out
    cdef np.ndarray[ndim=2, dtype=np.float64_t] dists
    cdef np.ndarray[ndim=1, dtype=np.float64_t] distn
    cdef np.ndarray[ndim=1, dtype=np.complex128_t] pp, En
    cdef double dtmp, dmin
    cdef np.complex128_t dd
    dmin = 1000.
    angles = np.linspace(-np.pi/4, np.pi/4, Mtestangles, endpoint=False)
    pp = np.exp(1.j*angles)
    dists = np.zeros((N, Mtestangles))+100.
    distn = np.zeros(Mtestangles)
    angles_out = np.zeros(E.shape[0], dtype=np.float64)
    idx = np.zeros(E.shape[0], dtype=np.uint32)
    L = E.shape[0]
    M = symbols.shape[0]
    for i in range(L):
        #for j in prange(Mtestangles):
        for j in range(Mtestangles):
            if i < N:
                for k in range(M):
                    dd = E[<unsigned int> i]*pp[<unsigned int> j] - symbols[<unsigned int> k]
                    dtmp = dd.real*dd.real + dd.imag*dd.imag
                    if dtmp < dists[<unsigned int>i,<unsigned int>j]:
                        dists[<unsigned int>i, <unsigned int>j] = dtmp
            else:
            #dists = np.roll(dists, -1, 0)
                print('where')
                o = N - i%N
                for k in range(M):
                    dd = E[<unsigned int>i]*pp[<unsigned int>j] - symbols[k]
                    dtmp = dd.real*dd.real + dd.imag*dd.imag
                    if dtmp < dists[<unsigned int>o,<unsigned int>j]:
                        dists[<unsigned int>o, <unsigned int>j] = dtmp
                for l in range(N):
                    distn[j] += dists[<unsigned int>l, <unsigned int>j]
                if distn[j] < dmin:
                    idx[<unsigned int>i-N//2] = j
                    dmin = distn[<unsigned int>j]
                angles_out[<unsigned int>i-N//2] = angles[idx[<unsigned int>i-N//2]]
    angles_out = np.unwrap(angles_out*4, discont=np.pi)/4
    En = E*np.exp(1.j*angles_out)
    return En, angles_out

def avg_win(x, N, axis=0):
    cs = np.cumsum(x, axis=axis)
    return cs[N:]-cs[:-N]

# def bps2(np.ndarray[ndim=1, dtype=np.complex128_t] E, int Mtestangles, np.ndarray[ndim=1, dtype=np.complex128_t] symbols, int N):
#     cdef unsigned int i, j, k, M, l, o, offset
#     cdef np.ndarray[ndim=1, dtype=np.float64_t] angles
#     cdef np.ndarray[ndim=1, dtype=np.uint32_t] idx
#     cdef np.ndarray[ndim=1, dtype=np.float64_t] angles_out
#     cdef np.ndarray[ndim=2, dtype=np.float64_t] dists
#     cdef np.ndarray[ndim=1, dtype=np.float64_t] distn
#     cdef np.ndarray[ndim=1, dtype=np.complex128_t] pp, En,EE
#     cdef double dtmp, dmin
#     cdef np.complex128_t dd
#     dmin = 1000.
#     angles = np.linspace(-np.pi/4, np.pi/4, Mtestangles, endpoint=False)
#     pp = np.exp(1.j*angles)
#     dists = np.zeros((E.shape[0], Mtestangles))+100.
#     distn = np.zeros(Mtestangles)
#     angles_out = np.zeros(E.shape[0], dtype=np.float64)
#     idx = np.zeros(E.shape[0], dtype=np.uint32)
#     L = E.shape[0]
#     M = symbols.shape[0]
#     nthreads = 8
#     ll = L//nthreads
#     for i in prange(8, nogil=True, num_threads=nthreads):
#         if i == 0:
#             offset = i*ll
#             endoff = (i+1)*ll + N
#         else:
#             offset = i*ll - N
#             endoff = (i+1)*ll + N
#         for l in range(ll):
#             for j in range(Mtestangles):
#                 for k in range(M):
#                     dmin = abs(E[i*ll+])
#                 dists[i*ll+(offset-N), k]
#         dist = np.min(abs(E[i*ll-offset:(i+1)*ll+offset, :, np.newaxis] - symbols)**2, axis=2)
#         idx[i*ll:(i+1)*ll] = avg_win(EE, N).argmin(axis=0)
#     angles_out = np.unwrap(angles[idx]*4)/4
#    return E*np.exp(1.j*angles_out), angles_out

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
