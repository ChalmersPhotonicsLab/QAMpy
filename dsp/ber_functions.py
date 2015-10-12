import numpy as np

class DataSyncError(Exception):
    pass

def sync_Tx2Rx(data_tx, data_rx, Lsync, imax=200):
    """Sync the transmitted data sequence to the received data, which
    might contain errors. Starts to with data_rx[:Lsync] if it does not find
    the offset it will iterate through data[i:Lsync+i] until offset is found
    or imax is reached.

    Parameters:
        data_tx:    the known input data sequence [np.array]
        data_rx:    the received data sequence which might contain errors [np.array]
        Lsync:      the number of elements to use for syncing [int]
        imax:       maximum number of tries before giving up [int]

    returns offset index, data_tx_sync which is synchronized to data_rx
    """
    for i in xrange(imax):
        try:
            sequence = data_rx[i:i+Lsync]
            idx_offs = mathfcts.find_offset(sequence, data_tx)
            idx_offs = idx_offs - i
            data_tx_synced = np.roll(data_tx, -idx_offs)
            return idx_offs, data_tx_synced
        except ValueError:
            pass
    raise DataSyncError("maximum iterations exceeded")

def sync_PRBS2Rx(data_rx, order, Lsync, imax=200):
    """
    Synchronise a prbs sequence to a possibly noisy received data stream.
    This is different to the general sync code, because in general the data
    array will be much shorter than the prbs sequence, which takes a long
    time to compute for lengths of > 2**23-1.

    Parameters:
        data_rx:    the received data signal [np.ndarray]
        order:      the order of the prbs sequence [int]
        Lsync:      the length of the bits to test for equality [int]
        imax:       the maximum number of iterations to test with [int]

    returns prbs sequence that it synchronized to the data stream
    """
    for i in range(imax):
        if i+Lsync > len(data_rx):
            break
        datablock = data_rx[i:i+Lsync]
        prbsseq = make_prbs_extXOR(order, Lsync-order, datablock[:order])
        if np.all(datablock[order:] == prbsseq):
            prbs = np.hstack([datablock,
                make_prbs_extXOR(order,
                    len(data_rx)-i-len(datablock),
                    datablock[-order:])])
            return i, prbs
    raise DataSyncError("maximum iterations exceeded")

def adjust_data_length(data_tx, data_rx):
    """Adjust the length of data_tx to match data_rx, either by truncation
    or repeating the data"""
    if len(data_tx) > len(data_rx):
        return data_tx[:len(data_rx)]
    elif len(data_tx) < len(data_rx):
        for i in range(len(data_rx)//len(data_tx)):
            data_tx = np.hstack([data_tx, data_tx])
        data_tx = np.hstack([data_tx,
            data_tx[len(data_rx)%len(data_tx)]])
        return data_tx
    else:
        return data_tx

 def _cal_BER_only(data_rx, data_tx, threshold=0.2):
    """Calculate the BER between two synchronised binary data signals"""
    errs = np.count_nonzero(data_rx-data_tx)
    N = len(data_tx)
    ber = errs/N
    if ber > threshold:
        raise ValueError("BER is over %.1f, this is probably a wrong sync"%threshold)
    return ber, errs, N

def cal_BER(data_rx, Lsync, order=None, data_tx=None, imax=200):
    """Calculate the BER between data_tx and data_rx. If data_tx is shorter
    than data_rx it is assumed that data_rx is repetitive.

    Parameters:
    data_rx:    measured receiver data [np.ndarray]
    Lsync:      number of bits to use for synchronisation [np.int]
    order:      order of PRBS if no data_tx is given [default:None]
    data_tx:    known input data if none assume PRBS [default:None]
    imax:       number of iterations to try for sync [np.int]

    returns ber
    """
    if data_tx is None and order is not None:
        return cal_BER_PRBS(data_rx.astype(np.bool), order, Lsync, imax)
    elif order is None and data_tx is not None:
        return cal_BER_known_seq(data_rx.astype(np.bool),
                data_tx.astype(np.bool), Lsync, imax)
    else:
        raise ValueError("data_tx and order must not both be None")

def cal_BER_PRBS(data_rx, order, Lsync, imax=200):
    """Calculate the BER between data_tx and a prbs signal with a given
    order.

    Parameters:
    data_rx:    measured receiver data [np.ndarray]
    order:      order of the PRBS [np.int]
    Lsync:      number of bits to use for synchronisation [np.int]
    imax:       number of iterations to try for sync [np.int]

    returns ber, number of errors, length of data
    """
    inverted = False
    try:
        idx, prbs = sync_PRBS2Rx(data_rx, order, Lsync, imax)
    except DataSyncError:
        inverted = True
        # if we cannot sync try to use inverted data
        data_rx = -data_rx
        idx, prbs = sync_PRBS2Rx(data_rx, order, Lsync, imax)
    return _cal_BER_only(data_rx[idx:], prbs), inverted

def cal_BER_known_seq(data_rx, data_tx, Lsync, imax=200):
    """Calculate the BER between data_tx and data_rx. If data_tx is shorter
    than data_rx it is assumed that data_rx is repetitive.

    Parameters:
    data_rx:    measured receiver data [np.ndarray]
    data_tx:    known input data [np.ndarray]
    Lsync:      number of bits to use for synchronisation [np.int]
    imax:       number of iterations to try for sync [np.int]

    returns ber, number of errors, length of data
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
    data_rx:    measured receiver data [np.ndarray(dtype=np.complex)]
    order_I:    the PRBS order of the in-phase component [int]
    order_Q:    the PRBS order of the quadrature component [int]
    Lsync:      the length of bits to use for syncing, if None use double
                the longer order [int, default=None]
    imax:       maximum number of tries for syncing before giving up [int,
                default=200]

    Returns:
        BER, No. of errors, length of data
    """
    #TODO implement offset parameter
    data_demod = QAMdemod(4, data_rx)[0]
    data_I = (1+data_demod.real).astype(np.bool)
    data_Q = (1+data_demod.imag).astype(np.bool)
    if Lsync is None:
        Lsync = 2*max(order_I,order_Q)
    try:
        (ber_I, err_I, N_I), inverted = cal_BER_PRBS(data_I, order_I, Lsync, imax)
    except DataSyncError:
        tmp = order_I
        order_I = order_Q
        order_Q = tmp
        try:
            (ber_I, err_I, N_I), inverted = cal_BER_PRBS(data_I, order_I, Lsync, imax)
        except DataSyncError:
            raise Warning("Could not sync PRBS to data")
            return np.nan, np.nan, np.nan
    if inverted:
        data_Q = -data_Q
    try:
        (ber_Q, err_Q, N_Q), inverted = cal_BER_PRBS(data_Q, order_Q, Lsync, imax)
    except DataSyncError:
        raise Warning("Could not sync PRBS to data")
        return np.nan, np.nan, np.nan

    return (ber_I+ber_Q)/2., err_Q+err_I, N_Q+N_I
