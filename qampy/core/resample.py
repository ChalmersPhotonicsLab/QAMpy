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
Signal resampling functions with different filter methods
"""

import fractions
import numpy as np
from scipy import signal as scisig

from qampy.helpers import normalise_and_center
from qampy.core.special_fcts import rrcos_freq, rrcos_time
from qampy.core.filter import rrcos_pulseshaping


def _resamplingfactors(fold, fnew):
    ratn = fractions.Fraction(fnew/fold).limit_denominator()
    return ratn.numerator, ratn.denominator


def resample_poly(signal, fold, fnew, window=None, renormalise=False):
    """
    Resamples a signal from an old frequency to a new. Preserves the whole data
    but adjusts the length of the array in the process.

    Parameters
    ----------
    signal: array_like
        signal to be resampled
    fold : float
        Sampling frequency of the signal
    fnew : float
        New desired sampling frequency.
    window : array_like, optional
        sampling windowing function
    renormalise : bool, optional
        whether to renormalise and recenter the signal to a power of 1.

    Returns
    -------
    out : array_like
        resampled signal of length fnew/fold*len(signal)

    """
    signal = signal.flatten()
    L = len(signal)
    up, down = _resamplingfactors(fold, fnew)
    if window is None:
        sig_new = scisig.resample_poly(signal, up, down)
    else:
        sig_new = scisig.resample_poly(signal, up, down, window=window)
    if renormalise:
        p = np.mean(abs(signal)**2)
        sig_new = normalise_and_center(sig_new)*np.sqrt(p)
    return sig_new

def rrcos_resample(signal, fold, fnew, Ts=None, beta=None, taps=4001, renormalise=False, fftconv=True):
    """
    Resample a signal using a root raised cosine filter. This performs pulse shaping and resampling a the same time.
    The resampling is done by scipy.signal.resample_poly. This function can be quite slow.

    Parameters
    ----------
    signal   : array_like
        input time domain signal
    fold     : float
        sampling frequency of the input signal
    fnew     : float
        desired sampling frequency
    Ts       : float, optional
        time width of the RRCOS filter (default:None makes this 1/fold)
    beta     : float, optional
        filter roll off factor between (0,1] (default:None will use the default filter in poly_resample)
    taps : int, optional
        taps of the interpolation filter if taps is None we filter by zeroinsertion upsampling and multipling
        with the full length rrcos frequency response in the spectral domain
    fftconv : bool, optional
        filter using zero insertion upsampling/decimation and filtering using fftconvolve. I often faster for
        long filters and power of two signal lengths.

    Returns
    -------
    sig_out  : array_like
        resampled output signal

    """
    if beta is None:
        return resample_poly(signal, fold, fnew)
    assert 0 < beta <= 1, "beta needs to be in interval (0,1]"
    if Ts is None:
        Ts = 1/fold
    up, down = _resamplingfactors(fold, fnew)
    fup = up*fold
    # for very large tap lengths it is better to filter with fftconvolve
    # it could be better to use default values here, but this seems to depend on pc
    if fftconv or taps is None:
        sig_new = np.zeros((up*signal.size,), dtype=signal.dtype)
        sig_new[::up] = signal
        sig_new = rrcos_pulseshaping(sig_new, fup, Ts, beta, taps)
        sig_new = sig_new[::down]
    else:
        t = np.linspace(0, taps, taps, endpoint=False)
        t -= t[(t.size-1)//2]
        t /= fup
        nqf = rrcos_time(t, beta, Ts)
        nqf /= nqf.max()
        sig_new = scisig.resample_poly(signal, up, down, window=nqf)
    if renormalise:
        p = np.mean(abs(signal)**2)
        sig_new = normalise_and_center(sig_new)*np.sqrt(p)
    return sig_new