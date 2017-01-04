from __future__ import division, print_function
import pyximport
pyximport.install()
import numpy as np
from . import mathfcts

try:
    from .dsp_cython import lfsr_ext
except:
    from .mathfcts import lfsr_ext
    print("can not import cython build module")

try:
    from .dsp_cython import lfsr_int
except:
    print("can not import cython build module")
    from .mathfcts import lfsr_int


def make_prbs_extXOR(order, nbits, seed=None):
    """
    Create Pseudo Random Bit Sequence using
    Linear Feedback Shift Register with a Fibonacci or external XOR
    implementation.

    Parameters
    ----------:
    order : int
        Order of the bit sequence (must be one of 7, 15, 23, 31)
    nbits : int
        The length or number of bits in the sequence
    seed : int, optional
        Seed for the LFSR, None corresponds to all bits one (default is None)

    Returns
    -------
    prbs : array_like
        Array of nbits, dtype=bool, len=nbits
    """
    assert order in [7, 15, 23, 31], """Only orders 7, 15, 23, 31 are
    implemented"""
    tapdict = {7: [7, 6], 15: [15, 14], 23: [23, 18], 31: [31, 28]}
    if seed is None:
        seed = mathfcts.bool2bin(np.ones(order))
    else:
        try:
            seed = mathfcts.bool2bin(seed)
        except TypeError:
            seed = seed
    out = np.zeros(nbits, dtype=bool)
    lfsr = lfsr_ext(seed, tapdict[order], order)
    for i in range(nbits):
        out[i] = next(lfsr)[0]
    return out


def make_prbs_intXOR(order, nbits, seed=None):
    """Create Pseudo Random Bit Sequence using a Linear Feedback
    Shift Register with Galois internal XOR implementation.

    Parameters
    ----------:
    order : int
        Order of the bit sequence (must be one of 7, 15, 23, 31)
    nbits : int
        The length or number of bits in the sequence
    seed : int, optional
        Seed for the LFSR, None corresponds to all bits one (default is None)

    Returns
    -------
    prbs : array_like
        Array of nbits, dtype=bool, len=nbits
    """
    assert order in [7, 15, 23, 31], """Only orders 7, 15, 23, 31 are
    implemented"""
    masks = {
        7: 2**7 + 2**6 + 1,
        15: 2**15 + 2**14 + 1,
        23: 2**23 + 2**18 + 1,
        31: 2**31 + 2**28 + 1
    }
    if seed is None:
        seed = mathfcts.bool2bin(np.ones(order))
    else:
        try:
            seed = mathfcts.bool2bin(seed)
        except TypeError:
            seed = seed
    out = np.empty(nbits, dtype=bool)
    lfsr = lfsr_int(seed, masks[order])
    for i in range(nbits):
        out[i] = next(lfsr)[0]
    return out
