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

"""
Various convenient utility functions.
"""

from __future__ import division, print_function

import numpy as np


def factorial(n):
    """The factorial of n, i.e. n!"""
    if n == 0: return 1
    return n * factorial(n - 1)


def linspacestep(start, step, N):
    """
    Create an array of given length for a given start and step
    value.

    Parameters
    ----------
    start : float
        first value to start with
    step : float
        size of the step
    N : int
        number of steps

    Returns
    -------
    out : array_like
        array of length N from start to start+N*step (not included)
    """
    return np.arange(start, start + N * step, step=step)


def lfsr_int(seed, mask):
    """
    A linear feedback shift register, using Galois or internal XOR
    implementation.

    Parameters
    ----------
    seed : int
        an integer representing the list of bits as the starting point of the
        register. Length N
    mask : int
        Determines the polynomial of the shift register (length N+1). The
        first and last bit of the mask must be 1.

    Yields
    ------
    xor : int
        output bit of the register
    state : int
        state of the register
    """
    state = seed
    nbits = mask.bit_length() - 1
    while True:
        state = (state << 1)
        xor = state >> nbits
        #the modulus operation on has an effect if the last bit is 1
        if xor != 0:
            state ^= mask  #this performs the modulus operation
        yield xor, state

def lfsr_ext(seed, taps, nbits):
    """A Fibonacci or external XOR linear feedback shift register.

    Parameters
    ----------
    seed : int
        binary number denoting the input state registers
    taps  : list
        list of registers that are input to the XOR (length 2)
    nbits : int
        number of registers

    Yields
    ------
    xor : int
        output bit of the registers
    state : int
        state of the register
    """
    sr = seed
    while 1:
        xor = 0
        for t in taps:
            if (sr & (1 << (nbits - t))) != 0:
                xor ^= 1
        sr = (xor << nbits - 1) + (sr >> 1)
        yield xor, sr

def bool2bin(x):
    """
    Convert an array of boolean values into a binary number. If the input
    array is not a array of booleans it will be converted.
    """
    assert len(x) < 64, "array must not be longer than 63"
    x = np.asarray(x, dtype=bool)
    y = 0
    for i, j in enumerate(x):
        y += j << i
    return y


def find_offset(sequence, data):
    """
    Find index where binary sequence occurs fist in the binary data array

    Parameters
    ----------
    sequence : array_like
        sequence to search for inside the data
    data : array_like
        data array in which to find the sequence

    It is required that len(data) > sequence

    Returns
    -------
    idx : int
        index where sequence first occurs in data
    """
    assert len(data) > len(sequence), "data has to be longer than sequence"
    if not data.dtype == sequence.dtype:
        raise Warning("""data and sequence are not the same dtype, converting
        data to dtype of sequence""")
        data = data.astype(sequence.dtype)
    # using this string conversion method is much faster than array methods,
    # however it only finds the first occurence
    return data.tostring().index(sequence.tostring()) // data.itemsize


def rolling_window(data, size, wrap=False):
    """
    Reshapes a 1D array into a 2D array with overlapping frames. Stops when the
    last value of data is reached.

    Parameters
    ----------
    data : array_like
        Data array to segment
    size : int
        The frame size

    Returns
    -------
    out : array_like
        output segmented 2D array


    Examples
    >>> utils.rolling_window(np.arange(10), 3)
    array([[0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6],
            [5, 6, 7],
            [6, 7, 8],
            [7, 8, 9]])
    """
    if wrap:
        dt = size - 1
        shape = data.shape[:-1] + (data.shape[-1], size)
        data = np.hstack([data, data[:dt]])
    else:
        shape = data.shape[:-1] + (data.shape[-1] - size + 1, size)
    strides = data.strides + (data.strides[-1], )
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def bin2gray(value):
    """
    Convert a binary value to an gray coded value see _[1]. This also works for arrays.
    ..[1] https://en.wikipedia.org/wiki/Gray_code#Constructing_an_n-bit_Gray_code
    """
    return value^(value >> 1)

def convert_iqtosinglebitstream(idat, qdat, nbits):
    """
    Interleave a two bitstreams into a single bitstream with nbits per symbol. This can be used to create a combined PRBS signal from 2 PRBS sequences for I and Q channel. If nbits is odd we will use nbits//2 + 1 bits from the first stream and nbits//2 from the second.

    Parameters
    ----------
    idat    : array_like
        input data stream (1D array of booleans)
    qdat    : array_like
        input data stream (1D array of booleans)
    nbits   : int
        number of bits per symbol that we want after interleaving

    Returns
    -------
    output   : array_like
        interleaved bit stream
    """
    if nbits%2:
        N = [nbits//2+1, nbits//2]
    else:
        N = [nbits//2, nbits//2]
    idat_n = idat[:len(idat)-(len(idat)%N[0])]
    idat_n = idat_n.reshape(N[0], len(idat_n)/N[0])
    qdat_n = qdat[:len(qdat)-(len(qdat)%N[1])]
    qdat_n = qdat_n.reshape(N[1], len(qdat_n)/N[1])
    l = min(len(idat_n[0]), len(qdat_n[0]))
    return np.hstack([idat_n[:l], qdat_n[:l]]).flatten()
