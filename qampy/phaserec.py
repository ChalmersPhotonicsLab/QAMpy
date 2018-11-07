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
from qampy.core.phaserecovery import bps_twostage, phase_partition_16qam

def bps_twostage(E, Mtestangles, N, B=4, **kwargs):
    """
    Perform a blind phase search phase recovery using two stages after _[1]

    Parameters
    ----------

    E           : SignalObject
        input signal 

    Mtestangles : int
        number of initial test angles to try

    symbols     : array_like
        the symbols of the modulation format

    N           : int
        block length to use for averaging

    B           : int, optional
        number of second stage test angles

    **kwargs    :
        keyword arguments to be passed to the core function

    Returns
    -------
    Eout    : SignalObject
        phase compensated field
    ph      : array_like
        unwrapped angle from phase recovery

    References
    ----------
    ..[1] Qunbi Zhuge and Chen Chen and David V. Plant, Low Computation Complexity Two-Stage Feedforward Carrier Recovery Algorithm for M-QAM, Optical Fiber Communication Conference (OFC, 2011)
    """
    return core.phaserecovery.bps_twostage(E, Mtestangles, E.coded_symbols, N, B=B, **kwargs)

def bps(E, Mtestangles, N, **kwargs):
    """
    Perform a blind phase search after _[1]

    Parameters
    ----------

    E           : SignalObject
        input signal

    Mtestangles : int
        number of test angles to try

    N           : int
        block length to use for averaging

    **kwargs    :
        keyword arguments to be passed to the core function

    Returns
    -------
    Eout    : SignalObject
        signal with compensated phase
    ph      : array_like
        unwrapped angle from phase recovery

    References
    ----------
    ..[1] Timo Pfau et al, Hardware-Efficient Coherent Digital Receiver Concept With Feedforward Carrier Recovery for M-QAM Constellations, Journal of Lightwave Technology 27, pp 989-999 (2009)
    """
    return core.phaserecovery.bps(E, Mtestangles, E.coded_symbols, N, **kwargs)

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

def pilot_cpe(signal, N=1, pilot_rat=1, max_blocks=None, nframes=1, use_seq=False):
    """
    Pilot based Carrier Phase Estimation

    Parameters
    ----------
    signal : PilotSignalObject
        the signal to perform phase estimation on
    N : int (optional)
        length of the averaging filter for phase estimation
    pilot_rat : int (optional)
        use every nth pilot
    max_blocks : int (optional)
        maximum number of blocks to process
    nframes: int (optional)
        how many frames to process, the output data will be truncated to min(signal.shape[-1], signal.frame_len*nframes)
    use_seq : bool (optional)
        use the pilot sequence for CPE as well
    Returns
    -------
    signal : PilotSignalObject
        phase compensated signal
    trace : array_like
        phase trace
    """
    if use_seq:
        seq_len = signal._pilot_seq_len
        idx = signal._idx_pil
        pilots = signal.pilots
    else:
        seq_len = None
        idx = signal._idx_pil[signal._pilot_seq_len:]
        pilots = signal.ph_pilots
    data, phase = core.pilotbased_receiver.pilot_based_cpe_new(signal, pilots, idx, signal.frame_len, seq_len=seq_len,
                                                           max_num_blocks=max_blocks, use_pilot_ratio=pilot_rat, num_average=N,
                                                            nframes=nframes)
    return signal.recreate_from_np_array(data), phase


