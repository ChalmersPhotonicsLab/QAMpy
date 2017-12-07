import fractions
import numpy as np
from scipy import signal as scisig

from dsp.utils import normalise_and_center, rrcos_time, rrcos_freq


def _resamplingfactors(fold, fnew):
    ratn = fractions.Fraction(fnew/fold).limit_denominator()
    return ratn.numerator, ratn.denominator


def resample_poly(signal, fold, fnew, window=None, renormalise=False):
    """
    Resamples a signal from an old frequency to a new. Preserves the whole data
    but adjusts the length of the array in the process.

    Parameters
    ----------
    signal: array_like
        signal to be resampled
    fold : float
        Sampling frequency of the signal
    fnew : float
        New desired sampling frequency.
    window : array_like, optional
        sampling windowing function
    renormalise : bool, optional
        whether to renormalise and recenter the signal to a power of 1.

    Returns
    -------
    out : array_like
        resampled signal of length fnew/fold*len(signal)

    """
    signal = signal.flatten()
    L = len(signal)
    up, down = _resamplingfactors(fold, fnew)
    if window is None:
        signal = scisig.resample_poly(signal, up, down)
    else:
        signal = scisig.resample_poly(signal, up, down, window=window)
    if renormalise:
        signal = normalise_and_center(signal)
    return signal


def rrcos_resample_zeroins(signal, fold, fnew, Ts=None, beta=0., renormalise=False):
    """
    Resample a signal using a root raised cosine filter. This performs pulse shaping and resampling a the same time.
    The resampling is done by upsampling using zeroinsertion and down sampling by decimation.

    Parameters
    ----------
    signal   : array_like
        input time domain signal
    fold     : float
        sampling frequency of the input signal
    fnew     : float
        desired sampling frequency
    Ts       : float, optional
        time width of the RRCOS filter (default:None makes this 1/fold)
    beta     : float, optional
        filter roll off factor between [0,1]
    renormalise : bool, optional
        whether to renormalise and recenter the signal to a power of 1.

    Returns
    -------
    sig_out  : array_like
        resampled output signal

    """
    if Ts is None:
        Ts = 1/fold
    N = signal.shape[0]
    up, down = _resamplingfactors(fold, fnew)
    sig_new = np.zeros(N*up, signal.dtype)
    sig_new[::up] = signal
    # below is a way to apply the filter in the time domain (somewhat) there is a shift by 1 point which can make a difference for the equaliser
    #tup = np.linspace(-N*up/2, N*up/2, N*up, endpoint=False)/(fold*up)
    #nqf = rrcos_time(tup, beta, Ts)
    #nqf /= nqf.max()
    #sig_new = scisig.fftconvolve(sig_new, nqf, 'same')
    sig_new = rrcos_pulseshaping(sig_new, up*fold, Ts, beta)
    if renormalise:
        sig_new = normalise_and_center(sig_new[::down])
    else:
        sig_new = sig_new[::down]
    return sig_new


def rrcos_resample_poly(signal, fold, fnew, Ts=None, beta=None, taps=4000, renormalise=False, tdomain_filter=False):
    """
    Resample a signal using a root raised cosine filter. This performs pulse shaping and resampling a the same time.
    The resampling is done by scipy.signal.resample_poly. This function can be quite slow.

    Parameters
    ----------
    signal   : array_like
        input time domain signal
    fold     : float
        sampling frequency of the input signal
    fnew     : float
        desired sampling frequency
    Ts       : float, optional
        time width of the RRCOS filter (default:None makes this 1/fold)
    beta     : float, optional
        filter roll off factor between [0,1] (default:None will use the default filter in poly_resample)
    taps : float, optional
        taps of the interpolation filter
    tdomain_filter: bool, optional
        whether to calculate the RRCOS filter in the time or frequency domain (with fft). Frequency domain seems to work slightly better

    Returns
    -------
    sig_out  : array_like
        resampled output signal

    """
    if beta is None:
        return resample_poly(signal, fold, fnew)
    if Ts is None:
        Ts = 1/fold
    up, down = _resamplingfactors(fold, fnew)
    fup = up*fold
    if tdomain_filter:
        t = np.linspace(-taps/2, taps/2, taps, endpoint=False)/fup
        nqf = rrcos_time(t, beta, Ts)
    else:
        f = np.linspace(-1/2, 1/2, taps, endpoint=False)*fup
        nqf = np.fft.fftshift(np.fft.fft(rrcos_freq(f, beta, Ts)))
    nqf /= nqf.max()
    sig_new = scisig.resample_poly(signal, up, down, window=nqf)
    if renormalise:
        sig_new = normalise_and_center(sig_new)
    return sig_new