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
Functions for fast creation of pseudo random bit sequences  (PRBS) patterns
"""

from __future__ import division, print_function
import numpy as np
from qampy.core import utils
from qampy.core.pythran_dsp import prbs_int, prbs_ext

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
    assert order in [7, 15, 23, 31], """Only orders 7, 15, 23, 31 are implemented"""
    nbits = int(nbits) # need to make sure it is integer for pythran
    tapdict = {7: [7, 6], 15: [15, 14], 23: [23, 18], 31: [31, 28]}
    if seed is None:
        seed = utils.bool2bin(np.ones(order))
    else:
        try:
            seed = utils.bool2bin(seed)
        except TypeError:
            seed = seed
    out = prbs_ext(seed, np.array(tapdict[order]), order, nbits)
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

