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
# Copyright 2018 Jochen Schröder, Zonglong

import numpy as np

def clipper(sig, clipping_level):
    """
    Clip the signal out of the range (-clipping_level, clipping_level).
    """
    sig_clip_re = np.sign(sig.real) * np.minimum(abs(sig.real), clipping_level*np.ones((1, sig.shape[1])))
    sig_clip_im = np.sign(sig.imag) * np.minimum(abs(sig.imag), clipping_level*np.ones((1, sig.shape[1])))

    return sig_clip_re + 1j* sig_clip_im

def modulator_arsin(sig, vpi_i, vpi_q):
    """
    Use arcsin() function to compensate modulator nonlinear sin() response.
    Input signal range should be (-1,1), which is required by the arcsin() function.
    """
    sig_out_re = 2 * vpi_i / np.pi * np.arcsin(sig.real)
    sig__out_im = 2 * vpi_q / np.pi * np.arcsin(sig.imag)
    sig_out = sig_out_re + 1j*sig__out_im

    return sig_out

def DAC_freq_comp():
    """
    Compensate frequency response of digital-to-analog converter(DAC).
    """

    return