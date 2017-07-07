from __future__ import division, print_function
import numpy as np
from scipy.signal import fftconvolve
from . import utils, prbs
from . import theory


class DataSyncError(Exception):
    pass

def sync_rx2tx_xcorr(data_tx, data_rx, N1, N2):
    """
    Sync the received data sequence to the transmitted data, which
    might contain errors, using cross-correlation between data_rx and data_tx.
    Calculates np.correlate(data_rx[:N1], data_tx[:N2], 'full').

    Parameters
    ----------

    data_tx : array_like
            the known input data sequence.

    data_rx : array_like
        the received data sequence which might contain errors.

    N1 : int
        the length of elements of the longer array input to correlate. This should be long enough so that the subsequence we use for searching is present, a good value is the number of bits in the PRBS.

    N2 : int, optional
        the length of the subsequence from data_tx to use to search for in data_rx

    Returns
    -------
    offset index : int
        the index where data_rx starts in data_rx
    data_rx_sync : array_like
        data_rx which is synchronized to data_tx

    """
    # this one gives a slightly higher BER need to investigate

    # needed to convert bools to integers
    tx = 1.*data_tx
    rx = 1.*data_rx
    ac = np.correlate(rx[:N1], tx[:N2], 'full')
    idx = abs(ac).argmax() - len(ac)//2 + (N1-N2)//2
    return np.roll(data_rx, idx), idx

def sync_tx2rx_xcorr(data_tx, data_rx):
    """
    Sync the transmitted data sequence to the received data, which
    might contain errors, using cross-correlation between data_rx and data_tx.
    Calculates np.fftconvolve(data_rx, data_tx, 'same'). This assumes that data_tx is at least once inside data_rx and repetitive>

    Parameters
    ----------

    data_tx : array_like
            the known input data sequence.

    data_rx : array_like
        the received data sequence which might contain errors.


    Returns
    -------
    data_rx_sync : array_like
        data_tx which is synchronized to data_rx
    offset index : int
        the index where data_rx starts in data_rx
    ac           : array_like
        the correlation between the two sequences
    """
    # needed to convert bools to integers
    tx = 1.*data_tx
    rx = 1.*data_rx
    if tx.dtype==np.complex128:
        N2 = len(tx)
        ac = fftconvolve(np.angle(rx), np.angle(tx)[::-1], 'same')
    else:
        N2 = len(tx)
        ac = fftconvolve(rx, tt[::-1], 'same')
    # I still don't quite get why the following works, an alternative would be to zero pad the shorter array
    # and the offset would be argmax()-len(ac)//2 however this does take significantly longer
    idx = abs(ac).argmax() - 5*N2//2
    return np.roll(data_tx, idx), idx, ac

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

def adjust_data_length(data_tx, data_rx, method=None):
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

    Returns
    -------
    data_tx_new, data_rx_new : array_like
        adjusted data sequences
    """
    if method is None:
        if len(data_tx) > len(data_rx):
            return data_tx[:len(data_rx)], data_rx
        elif len(data_tx) < len(data_rx):
            data_tx = _extend_by(data_tx, data_rx.shape[0]-data_tx.shape[0])
            return data_tx, data_rx
        else:
            return data_tx, data_rx
    elif method is "truncate":
        if len(data_tx) > len(data_rx):
            return data_tx[:len(data_rx)], data_rx
        elif len(data_tx) < len(data_rx):
            return data_tx, data_rx[:len(data_tx)]
        else:
            return data_tx, data_rx
    elif method is "extend":
        if len(data_tx) > len(data_rx):
            data_rx = _extend_by(data_rx, data_tx.shape[0]-data_rx.shape[0])
            return data_tx, data__rx
        elif len(data_tx) < len(data_rx):
            data_tx = _extend_by(data_tx, data_rx.shape[0]-data_tx.shape[0])
            return data_tx, data_rx
        else:
            return data_tx, data_rx

def _extend_by(data, N):
    L = data.shape[0]
    K = N//L
    rem = N%L
    data = np.hstack([data for i in range(K+1)])
    data = np.hstack([data, data[:rem]])
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


def cal_ber_nosync(data_rx, data_tx, Lsync, imax=200):
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
        idx, data_tx_sync = sync_Tx2Rx(data_tx, data_rx, Lsync, imax)
    except DataSyncError:
        # if we cannot sync try to use inverted data
        idx, data_tx_sync = sync_Tx2Rx(-data_tx, data_rx, Lsync, imax)
    data_tx_sync = adjust_data_length(data_tx_sync, data_rx)
    #TODO this still returns a slightly smaller value, as if there would be
    # one less error, maybe this happens in the adjust_data_length
    return cal_ber_syncd(data_rx, data_tx_sync)


def quantize_qam(sig, M):
    """Quantize a QAM signal assuming Grey coding where possible
    using maximum likelyhood. Calculates distance to ideal points.
    Only works for vectors (1D array), not for M-D arrays

    Parameters
    ----------
    sig : array_like
        signal data
    M : int
        QAM order (currently has to be 4)

    Returns
    -------
    sym : array_like
        symbols after demodulation (complex numbers)
    idx : array_like
        indices of the data items in the constellation diagram
    """
    L = len(sig)
    sym = np.zeros(L, dtype=np.complex128)
    data = np.zeros(L, dtype='int')
    cons = theory.cal_symbols_qam(M).flatten()
    scal = theory.cal_scaling_factor_qam(M)
    P = np.mean(utils.cabssquared(sig))
    sig = sig / np.sqrt(P)
    idx = abs(sig[:, np.newaxis] - cons).argmin(axis=1)
    sym = cons[idx]
    return sym, idx
