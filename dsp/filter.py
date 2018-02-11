from . import core
from .core.filter import moving_average

def filter_signal_analog(signal, cutoff, ftype="bessel", order=2):
    """
    Apply an analog filter to a signal for simulating e.g. electrical bandwidth limitation

    Parameters
    ----------

    signal  : SignalObject
        input signal
    cutoff  : float
        3 dB cutoff frequency of the filter
    ftype   : string, optional
        filter type can be either a bessel, butter, exp or gauss filter (default=bessel)
    order   : int
        order of the filter

    Returns
    -------
    signalout : SignalObject
        filtered output signal
    """
    return core.filter.filter_signal_analog(signal, signal.fs, cutoff, ftype="bessel", order=2)

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

