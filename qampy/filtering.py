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

from qampy import core
from qampy.core.filter import moving_average
__doc__  = core.filter.__doc__

def filter_signal(signal, cutoff, ftype="bessel", order=2, analog=False):
    """
    Apply a filter to a signal for simulating e.g. electrical bandwidth limitation

    Parameters
    ----------

    signal  : SignalObject
        input signal
    cutoff  : float
        3 dB cutoff frequency of the filter
    ftype   : string, optional
        filter type can be either a bessel, butter, exp or gauss filter (default=bessel)
    order   : int, optional
        order of the filter (default = 2)
    analog : bool, optional
        whether to use analog filtering with lsim, much slower, but could give sligthly more accurate
        results for bessel and butter filters does not have an effect for other filters  (default=False)


    Returns
    -------
    signalout : SignalObject
        filtered output signal
    """
    return signal.recreate_from_np_array(core.filter.filter_signal(signal, signal.fs, cutoff, ftype=ftype, order=order, analog=analog))

def filter_signal_analog(signal, cutoff, ftype="bessel", order=2):
    """
    Legacy function will be deprecated
    Apply an analog filter to a signal for simulating e.g. electrical bandwidth limitation
    
    
    Parameters
    ----------

    signal  : SignalObject
        input signal
    cutoff  : float
        3 dB cutoff frequency of the filter
    ftype   : string, optional
        filter type can be either a bessel, butter, exp or gauss filter (default=bessel)
    order   : int, optional
        order of the filter (default = 2)

    Returns
    -------
    signalout : SignalObject
        filtered output signal    
    """
    return filter_signal(signal, cutoff, ftype=ftype, order=order, analog=True)

def pre_filter(signal, bw):
    """
    Low-pass pre-filter signal with square shape filter

    Parameters
    ----------

    signal : SignalObject
        single polarization signal

    bw     : float
        bandwidth of the rejected part, given as fraction of overall length

    Returns
    -------
    signal_out : SignalObject
        filtered signal
    """
    sig_out = core.filter.pre_filter(signal, bw)
    return signal.recreate_from_np_array(sig_out)

def rrcos_pulseshaping(sig, beta, T=None):
    """
    Root-raised cosine filter applied in the spectral domain.

    Parameters
    ----------
    sig    : SignalObject
        input time distribution of the signal
    beta  : float
        filter roll-off factor needs to be in range [0, 1]
    T     : float, optional
        width of the filter (default: None, use the inverse of the signals symbol rate)

    Returns
    -------
    sign_out : SignalObject
        filtered signal in time domain
    """
    if T is None:
        T = sig.fb
    sig_out = core.filter.rrcos_pulseshaping(sig, sig.fs, T, beta)
    return sig.recreate_from_np_array(sig_out)

