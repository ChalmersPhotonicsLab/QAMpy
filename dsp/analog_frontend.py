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



