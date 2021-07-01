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
IO helper functions
"""

import zlib
import pickle
import numpy as np
from scipy.io import loadmat


def save_signal(fn, signal, lvl=4):
    """
    Save a signal object using zlib compression

    Parameters
    ----------
    fn : basestring
        filename
    signal : SignalObject
        the signal to save
    lvl : int, optional
        the compression to use for zlib
    """
    with  open(fn, "wb") as fp:
        sc = zlib.compress(pickle.dumps(signal, protocol=pickle.HIGHEST_PROTOCOL), level=lvl)
        fp.write(sc)

def load_signal(fn):
    """
    Load a signal object from a zlib compressed pickle file.

    Parameters
    ----------
    fn : basestring
        filename of the file

    Returns
    -------
    sig : SignalObject
        The loaded signal object

    """
    with open(fn, "rb") as fp:
        s = zlib.decompress(fp.read())
        obj = pickle.loads(s)
    return obj

def ndarray_from_matlab(fn, keys, transpose=False, dim2cmplx=False, portmap=[[0,1], [2,3]]):
    """
    Load a signal from matlab and put in the correct numpy array format

    Parameters
    ----------
    fn: basestring
        filename of the matlab file
    keys: list or tuple
        Nested list of keys (array names) in the matlab file. Depending on how the data is structured the keys can be in
        one of the following formats:
            If the symbols are given as a multi-dimensional array of complex numbers:
                                                        [[ keyname ]]
            If the symbols are given as multi-dimensional real arrays pairs for real and imaginary part:
                                                        [[ key_real, key_imag ]]
            If the different symbols of modes are saved in different complex arrays:
                                                        [[key_mode1], ..., [key_modeN]]
            If the symbols are given as pairs of real arrays for each mode:
                                                        [[key_mode1_real, key_mode1_imag], ... [key_modeN_real, key_modeN_imag]]
    transpose: boolean, optional
        Whether to transpose the matlab arrays
    dim2cmplx: boolean, optional
        Whether one of the dimensions is of the matlab arrays indicates real and imaginary parts.
        This is common for data from a realtime oscilloscope
    portmap: list, optional
        The mapping of dimension to mode and real and imaginary (or quadrature and in-phase) parts.
        only used when dim2cmplx is True.

    Returns
    -------
    symbs : ndarray
        numpy array in the correct format for a signal object

    """
    mat_dict = loadmat(fn)
    if len(keys) == 1:
        if len(keys[0]) == 2:
            symbs = mat_dict[keys[0][0]] + 1j*mat_dict[keys[0][1]]
        elif len(keys[0]) == 1:
            symbs = mat_dict[keys[0][0]]
        else:
            raise ValueError("Keys is in the wrong format, see documentation for correct format")
        if transpose:
            symbs = np.transpose(symbs)
    else:
        for i in range(len(keys)):
            if len(keys[0]) == 2:
                out = mat_dict[keys[i][0]].flatten() + 1j*mat_dict[keys[i][1]].flatten()
            elif len(keys[0]) == 1:
                out = mat_dict[keys[i][0]].flatten()
            else:
                raise ValueError("Keys is in the wrong format, see documentation for correct format")
            if i > 0:
                symbs = np.vstack([ symbs, out])
            else:
                symbs = out
    if dim2cmplx:
        out = []
        for i in range(len(portmap)):
            out.append(symbs[portmap[i][0]] + 1j*symbs[portmap[i][1]])
        symbs = np.array(out)
    return symbs
