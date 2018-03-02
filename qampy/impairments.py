import numpy as np
from . import core
from .core.impairments import rotate_field, add_awgn

def apply_PMD(field, theta, t_dgd):
    """
    Apply PMD to a given input field

    Parameters
    ----------

    field : SignalObject
        input dual polarisation optical field (first axis is polarisation)

    theta : float
        angle of the principle axis to the observed axis

    t_dgd : float
        differential group delay between the polarisation axes

    Returns
    -------
    out  : SignalObject
       new dual polarisation field with PMD
    """
    return core.impairments.apply_PMD_to_field(field, theta, t_dgd, field.fs)

def apply_phase_noise(signal, df):
    """
    Add phase noise from local oscillators, based on a Wiener noise process.

    Parameters
    ----------

    signal  : array_like
        single polarisation signal

    df : float
        combined linewidth of local oscillators in the system

    Returns
    -------
    out : array_like
       output signal with phase noise

    """
    return core.impairments.apply_phase_noise(signal, df, signal.fs)

def change_snr(sig, snr):
    """
    Change the SNR of a signal assuming that the input signal is noiseless

    Parameters
    ----------
    sig : array_like
        the signal to change
    snr : float
        the desired signal to noise ratio in dB

    Returns
    -------
    sig : array_like
        output signal with given SNR
    """
    return core.impairments.change_snr(sig, snr, sig.fb, sig.fs)

def add_carrier_offset(sig, fo):
    """
    Add frequency offset to signal

    Parameters
    ----------
    sig : array_like
        signal input array
    df : float
        frequency offset

    Returns
    -------
    signal : array_like
        signal with added offset
    """
    return core.impairments.add_carrier_offset(sig, fo, sig.fs)

def simulate_transmission(sig, snr=None, freq_off=None, lwdth=None, dgd=None, theta=np.pi/3.731):
    """
    Convenience function to simulate impairments on signal at once

    Parameters
    ----------
    sig : array_like
        input signal
    snr : flaat, optional
        desired signal-to-noise ratio of the signal. (default: None, don't change SNR)
    freq_off : float, optional
        apply a carrier offset to signal (default: None, don't apply offset)
    lwdth : float
        linewidth of the transmitter and LO lasers (default: None, infinite linewidth)
    dgd : float
        first-order PMD (differential group delay) (default: None, do not apply PMD)
    theta : float
        rotation angle to principle states of polarization

    Returns
    -------
    signal : array_like
        signal with transmission impairments applied
    """
    if lwdth is not None:
        sig = apply_phase_noise(sig, lwdth)
    if freq_off is not None:
        sig = add_carrier_offset(sig, freq_off)
    if snr is not None:
        sig = change_snr(sig, snr)
    if dgd is not None:
        sig = apply_PMD(sig, theta, dgd)
    return sig

