from . import core
from .core.analog_frontend import comp_IQ_inbalance


def comp_rf_delay(signal, delay):
    """
    Adds a delay to the signal in frequency domain. Can be
    used to compensate for impairments such as RF cables of different length
    between the optical hybrid and ADC.

    Parameters
    ----------
        signal : signalobject
            Real-valued input signal
        delay : float
            Delay  in s

    Returns
    -------
        sig_out : signalobject
            Signal after compensating for delay
    """
    comp = core.analog_frontend.comp_rf_delay(signal, delay, signal.fs)
    return signal.recreate_from_np_array(comp)

def orthonormalize_signal(signal):
    """
    Orthogonalizing signal using the Gram-Schmidt process _[1].

    Parameters
    ----------
    E : signalobject
       input signal

    Returns
    -------
    E_out : signalobject
        orthonormalized signal

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process for more
       detailed description.
    """
    os = int(signal.fs/signal.fb)
    return core.analog_frontend.orthonormalize_signal(signal, os)


