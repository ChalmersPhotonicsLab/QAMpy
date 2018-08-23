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
from qampy.core.phaserecovery import bps, bps_twostage, phase_partition_16qam


def find_freq_offset(sig, average_over_modes = False, fft_size = 4096):
    """
    Find the frequency offset by searching in the spectrum of the signal
    raised to 4. Doing so eliminates the modulation for QPSK but the method also
    works for higher order M-QAM.

    Parameters
    ----------
        sig : SignalObject
            signal array with N modes
        average_over_modes : bool
            Using the field in all modes for estimation
        fft_size: array
            Size of FFT used to estimate. Should be power of 2, otherwise the
            next higher power of 2 will be used.

    Returns
    -------
        freq_offset : int
            found frequency offset
    """
    return core.phaserecovery.find_freq_offset(sig, sig.os, average_over_modes=average_over_modes,
                                               fft_size=fft_size)

def comp_freq_offset(sig, freq_offset):
    """
    Compensate for frequency offset in signal

    Parameters
    ----------
        sig : array_line
            signal array with N modes
        freq_offset: array_like
            frequency offset to compensate for if 1D apply to all modes

    Returns
    -------
        comp_signal : array with N modes
            input signal with removed frequency offset

    """
    arr = core.phaserecovery.comp_freq_offset(sig, freq_offset, sig.os)
    return sig.recreate_from_np_array(arr)

def viterbiviterbi(E, N):
    """
    Viterbi-Viterbi blind phase recovery for an M-PSK signal

    Parameters
    ----------
    E : array_like
        the electric field of the signal
    N : int
        block length of samples to average over

    Returns
    -------
    Eout : array_like
        Field with compensated phases
    """
    return core.phaserecovery.viterbiviterbi(E, N, E.M)

