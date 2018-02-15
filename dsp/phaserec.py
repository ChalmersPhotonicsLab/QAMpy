from . import core
from .core.phaserecovery import viterbiviterbi, bps, bps_twostage, phase_partition_16qam


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
    os = int(sig.fs/sig.fb)
    return core.phaserecovery.find_freq_offset(sig, os, average_over_modes=average_over_modes,
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
    arr = core.phaserecovery.comp_freq_offset(sig, freq_offset, sig.fs)
    return sig.recreate_from_np_array(arr)
