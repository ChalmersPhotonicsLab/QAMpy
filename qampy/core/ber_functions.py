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
BER and bit and symbol stream calculation methods.
"""
from __future__ import division, print_function
import numpy as np
from scipy.signal import fftconvolve
from qampy.core import utils


#TODO: refactor to use remove all unneeded functions

class DataSyncError(Exception):
    pass

def find_sequence_offset(x, y, show_cc=False):
    """
    Find the time shift between two input signals that might contain errors
    might contain errors, using cross-correlation between data_rx and data_tx.
    Calculates np.fftconvolve(data_rx, data_tx, 'same'). This calculates how data_rx
    has to be shifted to align with data_tx

    Parameters
    ----------

    x : array_like
            The first signal

    y : array_like
        second signal, this is the signal that will need to be shifted

    show_cc : bool, optional
        if true return the calculated crosscorrelation

    Returns
    -------
    offset index : int
        the index to shift y in order to align both signals
    crosscorrelation: array_like, optional
        the "full" crosscorrelation between x and y
    """
    # needed to convert bools to integers
    X = 1.*x
    Y = 1.*y
    N_X = X.shape[0]
    N_Y = Y.shape[0]
    if np.issubdtype(Y.dtype , np.complexfloating):
        ac = fftconvolve(X, Y.conj()[::-1], 'full')
    else:
        ac = fftconvolve(X, Y[::-1], 'full')
    idx = abs(ac).argmax()-(N_Y-1) # this is necessary to find the correct position (size of full is N_X+N_Y-1)
    if show_cc is True:
        return idx, ac
    else:
        return idx

def find_sequence_offset_complex(x, y):
    """
    Find the offset of one sequence in the other even if both sequences are complex.

    Parameters
    ----------
    x : array_like
        transmitted data sequence
    y : array_like
        received data sequence

    Returns
    -------
    idx : integer
        offset index
    y : array_like
        y array possibly rotated to correct 1.j**i for complex arrays
    ii : integer
        power for complex rotation angle 1.j**ii
    """
    acm = 0.
    if not np.iscomplexobj(x) and not np.iscomplexobj(y):
        idx, acm = find_sequence_offset(x, y, show_cc=True)
        return idx, y, 0, acm
    for i in range(4):
        rx = y * 1.j ** i
        idx, ac = find_sequence_offset(x, rx, show_cc=True)
        act = ac.real.max()
        if act > acm:
            ii = i
            ix = idx
            acm = act
    return ix, y * 1.j ** ii, ii, acm

def sync_and_adjust(data_tx, data_rx, adjust="tx"):
    """
    Synchronize and adjust length of received and transmitted data sequence. When the length
    differs between sequences the sequence length will be adjusted based on the adjust parameter
    and the length of the sequences. If the to adjusting sequence is shorter than the other sequence,
    we will assume that the pattern is repetitive and therefore pad the sequence. If it is longer than
    the other sequence we will truncate after adjusting the offset. Note that if sequences are complex we will
    rotate around the complex plane to remove abiguities.

    Parameters
    ----------
    data_tx : array_like
        transmitted symbol or bit sequence
    data_rx : array_like
        received symbol sequence can be noisy
    adjust : string, optional
        parameter that determines which data sequence to adjust. If "tx" truncate or extend data_tx
        if "rx" truncate or extend data_rx

    Returns
    -------
    tx : array_like
       (possibly adjusted) tx data
    rx : array_like
       (possibly adjusted) rx data
    """
    N_tx = data_tx.shape[0]
    N_rx = data_rx.shape[0]
    assert adjust == "tx" or adjust == "rx", "adjust need to be either 'tx' or 'rx'"
    if N_tx > N_rx:
        if adjust == "tx":
            offset, tx, ii, acm = find_sequence_offset_complex(data_rx, data_tx)
            tx = np.roll(tx, offset)
            return adjust_data_length(tx, data_rx, method="truncate"), acm
        elif adjust == "rx":
            offset, rx, ii, acm = find_sequence_offset_complex(data_tx, data_rx)
            tx, rx = adjust_data_length(data_tx, rx, method="extend", offset=offset)
            return (tx, rx), acm
    elif N_tx < N_rx:
        if adjust == "tx":
            offset, tx, ii, acm = find_sequence_offset_complex(data_rx, data_tx)
            tx, rx = adjust_data_length(tx, data_rx, method="extend", offset=offset)
            return (tx, rx), acm
        elif adjust == "rx":
            offset, rx, ii, acm  = find_sequence_offset_complex(data_tx, data_rx)
            rx = np.roll(rx, offset)
            return adjust_data_length(data_tx, rx, method="truncate"), acm
    else:
        if adjust == "tx":
            offset, tx, ii, acm = find_sequence_offset_complex(data_rx, data_tx)
            return (np.roll(tx, offset), data_rx), acm
        elif adjust == "rx":
            offset, rx, ii, acm = find_sequence_offset_complex(data_tx, data_rx)
            return (data_tx, np.roll(rx, offset)), acm

def sync_rx2tx(data_tx, data_rx, Lsync, imax=200):
    """Sync the received data sequence to the transmitted data, which
    might contain errors. Starts to with data_rx[:Lsync] if it does not find
    the offset it will iterate through data[i*Lsync:Lsync*(i+1)] until offset is found
    or imax is reached.

    Parameters
    ----------
    data_tx : array_like
            the known input data sequence.
    data_rx : array_like
        the received data sequence which might contain errors.
    Lsync : int
        the number of elements to use for syncing.
    imax : imax, optional
        maximum number of tries before giving up (the default is 200).

    Returns
    -------
    offset index : int
        the index where data_rx starts in data_rx
    data_rx_sync : array_like
        data_rx which is synchronized to data_tx

    Raises
    ------
    DataSyncError
        If no position can be found.
    """
    for i in np.arange(imax)*Lsync:
        try:
            sequence = data_rx[i:i + Lsync]
            idx_offs = utils.find_offset(sequence, data_tx)
            idx_offs = idx_offs - i
            data_rx_synced = np.roll(data_rx, idx_offs)
            return idx_offs, data_rx_synced
        except ValueError:
            pass
    raise DataSyncError("maximum iterations exceeded")

def sync_tx2rx(data_tx, data_rx, Lsync, imax=200):
    """Sync the transmitted data sequence to the received data, which
    might contain errors. Starts to with data_rx[:Lsync] if it does not find
    the offset it will iterate through data[i:Lsync+i] until offset is found
    or imax is reached.

    Parameters
    ----------
    data_tx : array_like
            the known input data sequence.
    data_rx : array_like
        the received data sequence which might contain errors.
    Lsync : int
        the number of elements to use for syncing.
    imax : imax, optional
        maximum number of tries before giving up (the default is 200).

    Returns
    -------
    offset index : int
        the index where data_rx starts in data_tx
    data_tx_sync : array_like
        data_tx which is synchronized to data_rx

    Raises
    ------
    DataSyncError
        If no position can be found.
    """
    for i in np.arange(imax)*Lsync:
        try:
            sequence = data_rx[i:i + Lsync]
            idx_offs = utils.find_offset(sequence, data_tx)
            idx_offs = idx_offs - i
            data_tx_synced = np.roll(data_tx, -idx_offs)
            return idx_offs, data_tx_synced
        except ValueError:
            pass
    raise DataSyncError("maximum iterations exceeded")

def adjust_data_length(data_tx, data_rx, method=None, offset=0):
    """Adjust the length of data_tx to match data_rx, either by truncation
    or repeating the data.

    Parameters
    ----------
    data_tx, data_rx : array_like
        known input data sequence, received data sequence

    method : string, optional
        method to use for adjusting the length. This can be either None, "extend" or "truncate".
        Description:
            "extend"   - pad the short array with its data from the beginning. This assumes that the data is periodic
            "truncate" - cut the shorter array to the length of the longer one
            None       - (default) either truncate or extend data_tx

    offset : int, optional
       offset where the start of the to extended array sits in the reference array


    Returns
    -------
    data_tx_new, data_rx_new : array_like
        adjusted data sequences
    """
    if method is None:
        if len(data_tx) > len(data_rx):
            return data_tx[:len(data_rx)], data_rx
        elif len(data_tx) < len(data_rx):
            if offset == 0:
                data_tx = _adjust_to(data_tx, data_rx.shape[0])
            else:
                data_tx1 = _adjust_to(data_tx, offset, back=False)
                data_tx2 = _adjust_to(data_tx, data_rx.shape[0]-offset)
                data_tx = np.hstack([data_tx1, data_tx2])
            return data_tx, data_rx
        else:
            return data_tx, data_rx
    elif method == "truncate":
        if len(data_tx) > len(data_rx):
            return data_tx[:len(data_rx)], data_rx
        elif len(data_tx) < len(data_rx):
            return data_tx, data_rx[:len(data_tx)]
        else:
            return data_tx, data_rx
    elif method == "extend":
        if len(data_tx) > len(data_rx):
            if offset == 0:
                data_rx = _adjust_to(data_rx, data_tx.shape[0])
            else:
                data_rx1 = _adjust_to(data_rx, offset, back=False)
                data_rx2 = _adjust_to(data_rx, data_tx.shape[0]-data_rx1.shape[0])
                data_rx = np.hstack([data_rx1, data_rx2])
            return data_tx, data_rx
        elif len(data_tx) < len(data_rx):
            if offset == 0:
                data_tx = _adjust_to(data_tx, data_rx.shape[0])
            else:
                data_tx1 = _adjust_to(data_tx, offset, back=False)
                data_tx2 = _adjust_to(data_tx, data_rx.shape[0]-data_tx1.shape[0])
                data_tx = np.hstack([data_tx1, data_tx2])
            return data_tx, data_rx
        else:
            return data_tx, data_rx

def _adjust_to(data, N, back=True):
    L = data.shape[0]
    K = N//L
    rem = N%L
    try:
        tmp = np.hstack([data for i in range(K)])
    except ValueError:
        tmp = np.array([], dtype=data.dtype)
    if back:
        data = np.hstack([tmp, data[:rem]])
    else:
        data = np.hstack([data[-rem:], tmp])
    return data

def cal_ber_syncd(data_rx, data_tx, threshold=0.2):
    """Calculate the bit-error rate (BER) between two synchronised binary data
    signals in linear units.

    Parameters
    ----------
    data_tx : array_like
            the known input data sequence.
    data_rx : array_like
        the received data signal stream
    threshold : float, optional
       threshold BER value. If calculated BER is larger than the threshold, an
       error is return as this likely indicates a wrong sync (default is 0.2).

    Returns
    -------
    ber : float
        bit-error rate in linear units
    errs : int
        number of counted errors.
    N : int
        length of data_tx

    Raises
    ------
    ValueError
        if ber>threshold, as this indicates a sync error.
    """
    errs = np.count_nonzero(data_rx != data_tx)
    N = len(data_tx)
    ber = errs / N
    if ber > threshold:
        raise ValueError("BER is over %.1f, this is probably a wrong sync" %
                         threshold)
    return ber, errs, N


def cal_ber_nosyncd(data_rx, data_tx):
    """
    Calculate the BER between a received bit stream and a known
    bit sequence which is not synchronised. If data_tx is shorter than data_rx it is assumed
    that data_rx is repetitive. This function automatically inverts the data if
    it fails to sync.

    Parameters
    ----------
    data_tx : array_like
        the known input data sequence.
    data_rx : array_like
        the received data sequence which might contain errors.
    Lsync : int
        the number of elements to use for syncing.
    imax : imax, optional
        maximum number of tries before giving up (the default is 200).

    Returns
    -------
    ber : float
        bit error rate in linear units
    errs : int
        number of counted errors
    N : int
        length of data
    """
    try:
        idx = find_sequence_offset(data_tx, data_rx)
    except DataSyncError:
        # if we cannot sync try to use inverted data
        idx = find_sequence_offset(~data_tx, data_rx)
    data_tx_sync = adjust_data_length(data_tx_sync, data_rx)
    #TODO this still returns a slightly smaller value, as if there would be
    # one less error, maybe this happens in the adjust_data_length
    return cal_ber_syncd(data_rx, data_tx_sync)
