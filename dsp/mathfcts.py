from __future__ import division, print_function
import numpy as np

""" a number of mathematical convenience functions"""

def cabssquared(x):
    """Calculate the absolute squared of a complex number"""
    return x.real**2 + x.imag**2

def tanh(x,p):
    """Calculate the hyperbolic tangent t = p[0]*tanh((x-p[1])/p[2])
    """
    return p[0]*tanh((x-p[1])/p[2])

def gauss(x, p):
    """Calculate the Gaussian function g = p[0]*exp(-((x-p[1])/p[2])**2/2)
    """
    return p[0]*np.exp(- ((x - p[1])/p[2])**2 /2.)

def supergauss(x, p):
    """
    Calculate the Supergaussian functions g =
    p[0]*exp(-((x-p[1])/p[2])**(2*p[3])/2)
    """
    return p[0]*np.exp(- ((x-p[1])/p[2])**(2*p[3])/2.)

def sech(x, p):
    """Calculate the sech function s = p[0]/cosh((x-p[1])/p[2])
    """
    return p[0]/np.cosh((x-p[1])/p[2])

def factorial(n):
    """The factorial of n, i.e. n!"""
    if n == 0: return 1
    return n * factorial(n-1)

def linspacestep(start, step, N):
    """Return an array with N values starting at start and increasing the next value by step"""
    return np.arange(start,start+N*step, step=step)

def lfsr_int(seed, mask):
    """A linear feedback shift register, using Galois or internal XOR implementation,
    the seed is a binary number with the bit length N. The mask determines
    the polynomial and has length N+1, also the first and last bit of the
    mask need to be 1.
    Returns a generator which yields the bit and the register state"""
    state = seed
    nbits = mask.bit_length()-1
    while True:
        state = (state << 1)
        xor = state >> nbits
        #the modulus operation on has an effect if the last bit is 1
        if xor != 0:
            state ^= mask #this performs the modulus operation
        yield xor, state

def lfsr_ext(seed, taps, nbits):
    """A Fibonacci or external XOR linear feedback shift register.

    Parameters:
        seed  -- binary number denoting the state registers
        taps  -- list of registers that are input to the XOR
        nbits -- number of registers

    yields (xor, state) where xor is the output of the registers and state is
    the register state at every step
    """
    sr = seed
    while 1:
        xor = 0
        for t in taps:
            if (sr & (1<<(nbits-t))) != 0:
                xor ^= 1
        sr = (xor << nbits-1) + (sr >> 1)
        yield xor, sr

def bool2bin(x):
    """Convert an array of boolean values into a binary number. If the input
    array is not a array of booleans it will be converted."""
    assert len(x)<64, "array must not be longer than 63"
    x = np.asarray(x, dtype=bool)
    y = 0
    for i, j in enumerate(x):
        y += j<<i
    return y

def find_offset(sequence, data):
    """Find index where sequence occurs in the data array

    Parameters:
        sequence: sequence to find in data [array]
        data: the data array in which to find the sequence [array]

    It is required that len(data) > sequence

    returns index where sequence starts within data
    """
    assert len(data)>len(sequence), "data has to be longer than sequence"
    if not data.dtype==sequence.dtype:
        raise Warning("""data and sequence are not the same dtype, converting
        data to dtype of sequence""")
        data = data.astype(sequence.dtype)
    # using this string conversion method is much faster than array methods,
    # however it only finds the first occurence
    return data.tostring().index(sequence.tostring())//data.itemsize

def rolling_window(data, size):
     shape = data.shape[:-1] + (data.shape[-1] - size + 1, size)
     strides = data.strides + (data. strides[-1],)
     return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

def resample(Fold, Fnew, E, window=None):
    ''' resamples the signal from Fold to Fnew'''
    E = E.flatten()
    L = len(E)
    num = Fnew/Fold*L
    if window is None:
        E = scisig.resample(E, num)
    else:
        E = scisig.resample(E, num, window=window)
    return E



