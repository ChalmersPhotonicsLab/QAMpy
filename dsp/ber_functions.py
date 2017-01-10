from __future__ import division, print_function
import numpy as np
from . import utils, prbs
from . import theory
from . import modulation


class DataSyncError(Exception):
    pass


def sync_Tx2Rx(data_tx, data_rx, Lsync, imax=200):
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
    for i in range(imax):
        try:
            sequence = data_rx[i:i + Lsync]
            idx_offs = utils.find_offset(sequence, data_tx)
            idx_offs = idx_offs - i
            data_tx_synced = np.roll(data_tx, -idx_offs)
            return idx_offs, data_tx_synced
        except ValueError:
            pass
    raise DataSyncError("maximum iterations exceeded")

def sync_PRBS2Rx(data_rx, order, Lsync, imax=200):
    """
    Synchronise a PRBS sequence to a possibly noisy received data stream.
    This is different to the general sync code, because in general the data
    array will be much shorter than the prbs sequence, which takes a long
    time to compute for lengths of > 2**23-1.

    Parameters
    ----------
    data_rx : array_like
        the received data signal stream
    order : int
        the order of the PRBS sequence.
    Lsync : int
        the number of bits to use to test for equality.
    imax : int, optional
        the maximum number of iterations to test with (default is 200).

    Returns
    -------
    prbs_tx_sync : array_like
        prbs sequence that it synchronized to the data stream

    Raises
    ------
    DataSyncError
        If no position can be found.
    """
    for i in range(imax):
        if i + Lsync > len(data_rx):
            break
        datablock = data_rx[i:i + Lsync]
        prbsseq = prbs.make_prbs_extXOR(order, Lsync - order,
                                        datablock[:order])
        if np.all(datablock[order:] == prbsseq):
            prbsval = np.hstack([
                datablock, prbs.make_prbs_extXOR(
                    order,
                    len(data_rx) - i - len(datablock), datablock[-order:])
            ])
            return i, prbsval
    raise DataSyncError("maximum iterations exceeded")


def adjust_data_length(data_tx, data_rx):
    """Adjust the length of data_tx to match data_rx, either by truncation
    or repeating the data.

    Parameters
    ----------
    data_tx, data_rx : array_like
        known input data sequence, received data sequence

    Returns
    -------
    data_tx_new : array_like
        input data sequence truncated or repeated to the same length as data_rx
    """
    if len(data_tx) > len(data_rx):
        return data_tx[:len(data_rx)]
    elif len(data_tx) < len(data_rx):
        data_tx = np.hstack([data_tx, data_tx[:len(data_rx)-len(data_tx)]])
        return data_tx
    else:
        return data_tx


def _cal_BER_only(data_rx, data_tx, threshold=0.2):
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


# TODO: the parameters of this function should really be changed to
# (data_rx, data_tx, Lsync, order=None, imax=200)
def cal_BER(data_rx, Lsync, order=None, data_tx=None, imax=200):
    """Calculate the BER between an received binary data stream and a given bit
    sequence or a PRBS. If data_tx is shorter than data_rx it is assumed that
    data_rx is repetitive.

    Parameters:
    ----------
    data_rx : array_like
        received binary data stream.
    Lsync : int
        number of bits to use for synchronisation
    order : int, optional
        order of PRBS if no data_tx is given (default:None)
    data_tx : array_like
        known input bit sequence (if none assume PRBS (default:None))
    imax : int, optional
        number of iterations to try for sync (default is 200).

    Returns
    -------
    ber : float
        linear bit error rate
    """
    if data_tx is None and order is not None:
        return cal_BER_PRBS(data_rx.astype(np.bool), order, Lsync, imax)
    elif order is None and data_tx is not None:
        return cal_BER_known_seq(
            data_rx.astype(np.bool), data_tx.astype(np.bool), Lsync, imax)
    else:
        raise ValueError("data_tx and order must not both be None")


def cal_BER_PRBS(data_rx, order, Lsync, imax=200):
    """Calculate the BER between data_tx and a PRBS signal with a given
    order. This function automatically tries the inverted data if it fails
    to sync.

    Parameters
    ----------
    data_rx: array_like
        measured receiver bit stream
    order : int
        order of the PRBS
    Lsync : int
        number of bits to use for synchronisation.
    imax : int, optional
        number of iterations to try for sync (default is 200).

    Returns
    -------
    ber : float
        bit error rate in linear units
    errs : int
        number of counted errors
    N : int
        length of data
    """
    inverted = False
    try:
        idx, prbsval = sync_PRBS2Rx(data_rx, order, Lsync, imax)
    except DataSyncError:
        inverted = True
        # if we cannot sync try to use inverted data
        data_rx = -data_rx
        idx, prbsval = sync_PRBS2Rx(data_rx, order, Lsync, imax)
    return _cal_BER_only(data_rx[idx:], prbsval), inverted


def cal_BER_known_seq(data_rx, data_tx, Lsync, imax=200):
    """Calculate the BER between a received bit stream and a known
    bit sequence. If data_tx is shorter than data_rx it is assumed
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
    return _cal_BER_only(data_rx, data_tx_sync)



def cal_BER_QPSK_prbs(data_rx, order_I, order_Q, Lsync=None, imax=200):
    """Calculate the BER for a QPSK signal the I and Q channels can either have
    the same or different orders.

    Parameters:
    ----------
    data_rx : array_like
        received bit stream
    order_I : int
        PRBS order of the in-phase component
    order_Q : int
        PRBS order of the quadrature component.
    Lsync : int, optional
        the length of bits to use for syncing, if None use twice the length of
        the longer order (default=None)
    imax : int, optional
        maximum number of tries for syncing before giving up (default=200)

    Returns
    -------
    ber : float
        bit error rate in linear units (np.nan if failed to sync)
    errs : int
        number of counted errors (np.nan if failed to sync)
    N : int
        length of data (np.nan if failed to sync)
    """
    #TODO implement offset parameter
    data_demod = QAMquantize(data_rx, 4)[0]
    data_I = (1 + data_demod.real).astype(np.bool)
    data_Q = (1 + data_demod.imag).astype(np.bool)
    if Lsync is None:
        Lsync = 2 * max(order_I, order_Q)
    try:
        (ber_I, err_I, N_I), inverted = cal_BER_PRBS(data_I, order_I, Lsync,
                                                     imax)
    except DataSyncError:
        tmp = order_I
        order_I = order_Q
        order_Q = tmp
        try:
            (ber_I, err_I, N_I), inverted = cal_BER_PRBS(data_I, order_I,
                                                         Lsync, imax)
        except DataSyncError:
            raise Warning("Could not sync PRBS to data")
            return np.nan, np.nan, np.nan
    if inverted:
        data_Q = -data_Q
    try:
        (ber_Q, err_Q, N_Q), inverted = cal_BER_PRBS(data_Q, order_Q, Lsync,
                                                     imax)
    except DataSyncError:
        raise Warning("Could not sync PRBS to data")
        return np.nan, np.nan, np.nan

    return (ber_I + ber_Q) / 2., err_Q + err_I, N_Q + N_I


def QAMquantize(sig, M):
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
    cons = modulation.calculate_MQAM_symbols(M).flatten()
    scal = theory.MQAMScalingFactor(M)
    P = np.mean(utils.cabssquared(sig))
    sig = sig / np.sqrt(P)
    idx = abs(sig[:, np.newaxis] - cons).argmin(axis=1)
    sym = cons[idx]
    return sym, idx
