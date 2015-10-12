import numpy as np
from . import mathfunctions

def make_prbs_extXOR(order, nbits, seed=None):
    """Create Pseudo Random Bit Sequence using a Linear Feedback
    Shift Register.

    Parameters:
        order: the order of the sequence [can be 7, 15, 23, 31]
        nbits: the number of bits in the sequence
        seed: seed for the LFSR [default:None] None corresponds to all ones

    Returns array of length nbits dtype=bool
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
    for i in xrange(nbits):
        out[i] = lfsr.next()[0]
    return out

def make_prbs_intXOR(order, nbits, seed=None):
    """Create Pseudo Random Bit Sequence using a Linear Feedback
    Shift Register with internal XOR.

    Parameters:
        order: the order of the sequence [can be 7, 15, 23, 31]
        nbits: the number of bits in the sequence
        seed: seed for the LFSR [default:None] None corresponds to all ones

    Returns array of length nbits dtype=bool
    """
    assert order in [7, 15, 23, 31], """Only orders 7, 15, 23, 31 are
    implemented"""
    masks = {7: 2**7+2**6+1, 15: 2**15+2**14+1, 23:
            2**23+2**18+1, 31: 2**31+2**28+1}
    if seed is None:
        seed = mathfcts.bool2bin(np.ones(order))
    else:
        try:
            seed = mathfcts.bool2bin(seed)
        except TypeError:
            seed = seed
    out = np.empty(nbits, dtype=bool)
    lfsr = lfsr_int(seed, masks[order])
    for i in xrange(nbits):
        out[i] = lfsr.next()[0]
    return out

