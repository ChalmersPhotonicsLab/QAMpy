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
cimport numpy as np
from ccomplex cimport *
from qampy.core.equalisation cimport cython_equalisation
from qampy.core.equalisation.cmath cimport exp, log, pow, log2


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
        dtmp = 100. # it is important to assign to the variable, othewise it will not
                    # be discovered as private by the compiler (passing the pointer  only does
                    # not work)
        if p > 1:
            ph_idx = i
        else:
            ph_idx = 0
        for j in range(Ntestangles):
            tmp = E[i] * comp_angles[ph_idx, j]
            s = cython_equalisation.det_symbol(symbols, M, tmp, &dtmp)
            if dtmp < dists[i, j]:
                dists[i, j] = dtmp
    return np.array(select_angle_index(dists, 2*N))

cpdef int[:] select_angle_index(cython.floating[:,:] x, int N):
    cdef cython.floating[:,:] csum
    cdef int[:] idx
    cdef int i,k, L, M
    cdef cython.floating dmin, dtmp
    L = x.shape[0]
    M = x.shape[1]
    csum = np.zeros((L,M), dtype="f%d"%x.itemsize)
    idx = np.zeros(L, dtype=np.intc)
    for i in range(1, L):
        dmin = 1000.
        if i < N:
            for k in range(M):
                csum[i,k] = csum[i-1,k]+x[i,k]
        else:
            for k in range(M):
                csum[i,k] = csum[i-1,k]+x[i,k]
                dtmp = csum[i,k]  - csum[i-N,k]
                if dtmp < dmin:
                    idx[i-N//2] = k
                    dmin = dtmp
    return idx

cpdef select_angles(cython.floating[:,:] angles, cython.integral[:] idx):
    cdef cython.floating[:] angles_out
    cdef int i, L
    if angles.shape[0] > 1:
        L = angles.shape[0]
        angles_out = np.zeros(L, dtype="f%d"%angles.itemsize)
        for i in prange(L, schedule='static', nogil=True):
            angles_out[i] = angles[i, idx[i]]
    else:
        L = idx.shape[0]
        angles_out = np.zeros(L, dtype="f%d"%angles.itemsize)
        for i in prange(L, schedule='static', nogil=True):
            angles_out[i] = angles[0, idx[i] ]
    return np.array(angles_out)

cpdef double[:] soft_l_value_demapper(cython_equalisation.complexing[:] rx_symbs, int M, double snr, cython_equalisation.complexing[:,:,:] bits_map):
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

cpdef double[:] soft_l_value_demapper_minmax(cython_equalisation.complexing[:] rx_symbs, int M, double snr, cython_equalisation.complexing[:,:,:] bits_map):
    cdef int num_bits = int(np.log2(M))
    cdef double[:] L_values = np.zeros(rx_symbs.shape[0]*num_bits)
    cdef int mode, bit, symb, l
    cdef int N = rx_symbs.shape[0]
    cdef int k = bits_map.shape[1]
    cdef double tmp = 10000
    cdef double tmp2 = 10000
    cdef double tmp3 = 10000
    cdef double tmp4 = 10000
    for bit in prange(num_bits, schedule="static",nogil=True):
        #for symb in prange(N, schedule='static', nogil=True):
        for symb in range(N):
            tmp = 10000
            tmp2 = 10000
            for l in range(k):
                tmp3 = cabssq(bits_map[bit,l,1] - rx_symbs[symb])
                if tmp3 < tmp:
                    tmp = tmp3
                tmp4 = cabssq(bits_map[bit,l,0] - rx_symbs[symb])
                if tmp4 < tmp2:
                    tmp2 = tmp4
            L_values[symb*num_bits + bit] = snr*(tmp2-tmp)
    return L_values

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

cpdef cal_gmi_mc(double complex[:] symbols, double snr, int ns, double complex[:,:,:] bit_map):
    cdef int M = symbols.size
    cdef int nbits = int(np.log2(M))
    cdef double complex[:] z
    cdef double gmi = 0
    cdef int i, j, b, l
    cdef double complex sym
    z = np.sqrt(1/snr)*(np.random.randn(ns) + 1j*np.random.randn(ns))/np.sqrt(2)
    for k in range(nbits):
        for b in range(2):
            for sym in bit_map[k,:,b]:
                for l in range(ns):
                    nom = cal_exp_sum(sym, symbols, z[l], snr)
                    denom = cal_exp_sum(sym, bit_map[k,:,b], z[l], snr)
                    gmi += log2(nom/denom)/ns
    return nbits - gmi/M

cdef double cal_exp_sum(double complex sym, double complex[:] syms, double complex z, double sigma):
    cdef int i
    cdef double out = 0
    cdef N = syms.size
    for i in range(N):
        out += exp(-sigma*(2*creal(z*(sym-syms[i])) + cabs(sym-syms[i])**2))
    return out

