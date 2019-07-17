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

from qampy.core.io import load_signal, save_signal
from qampy import signals
from qampy import helpers
from scipy.io import loadmat


def load_symbols_from_matlab_file(fn, M, keys, fb=10e9, transpose=False, fake_polmux=False, fake_pm_delay=0,
                                  normalise=True):
    """
    Create a signal object from matlab file.

    Parameters
    ----------
    fn : basestring
        Matlab filename
    M : int
        QAM order
    keys: list
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
    nmodes: int, optional
        Number of modes
    fb: float, optional
        Symbol rate
    transpose: boolean, optional
        Whether to transpose the matlab arrays
    fake_polmux: boolean, optional
        Whether the signal uses fake polmux, thus the single dimension matlab symbol array should be duplicated
    fake_pm_delay: int, optional
        Number of symbols to delay the second dimension in the fake pol-mux
    normalise : boolean, optional
        Normalise the symbols to an average power of 1 or not. Note that if you do not normalise you need to be very
        careful with your signal metrix

    Returns
    -------

    sig : signal_object
        Signal generated from the loaded symbols
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
                symbs = mat_dict[keys[0][0]] + 1j*mat_dict[keys[0][1]]
            elif len(keys[0]) == 1:
                symbs = mat_dict[keys[0][0]]
            else:
                raise ValueError("Keys is in the wrong format, see documentation for correct format")
            if i > 0:
                out = np.vstack([ out, symbs])
            else:
                out = symbs
    if fake_polmux:
        symbs = np.vstack([np.roll(symbs, fake_pm_delay ), symbs])
    if normalise:
        symbs = helpers.normalise_and_center(symbs)
    return signals.SignalQAMGrayCoded.from_symbol_array(symbs, M, fb)
