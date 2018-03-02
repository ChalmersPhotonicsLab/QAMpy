from __future__ import division, print_function
import numpy as np
from . import utils

try:
    pass
    from .dsp_cython import lfsr_ext, prbs_ext
except:
    from .utils import lfsr_ext
    print("can not import cython build module")

try:
    from .dsp_cython import lfsr_int, prbs_int
except:
    print("can not import cython build module")
    from .utils import lfsr_int

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
        seed = utils.bool2bin(np.ones(order))
    else:
        try:
            seed = utils.bool2bin(seed)
        except TypeError:
            seed = seed
    out = prbs_ext(seed, tapdict[order], order, nbits)
    return out.astype(bool)


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
        seed = utils.bool2bin(np.ones(order))
    else:
        try:
            seed = utils.bool2bin(seed)
        except TypeError:
            seed = seed
    out = prbs_int(seed, masks[order], nbits)
    return out.astype(bool)

