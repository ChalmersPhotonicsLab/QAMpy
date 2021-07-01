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
Filter functions for matched filtering, pulse-shaping and impairment simulation.
"""

import numpy as np
from qampy.core.special_fcts import rrcos_freq, rrcos_time
import scipy.signal as scisig
import scipy.fft as scifft

def pre_filter(signal, bw):
    """
    Low-pass pre-filter signal with square shape filter

    Parameters
    ----------

    signal : array_like
        single polarization signal

    bw     : float
        bandwidth of the rejected part, given as fraction of overall length
    """
    sig = np.atleast_2d(signal)
    N = sig.shape
    h = np.zeros(N, dtype=sig.real.dtype)
    h[:,int(N[1]/(bw/2)):-int(N[1]/(bw/2))] = 1
    s = scifft.ifft(scifft.ifftshift(scifft.fftshift(scifft.fft(sig, axis=-1), axes=-1)*h, axes=-1), axis=-1)
    if signal.ndim < 2:
        return s.flatten()
    else:
        return s

def pre_filter_wdm(signal, bw, os,center_freq = 0):
    """

    Ideal LP filter for selecting part of the spectrum. Uses FFT

    Parameters
    ----------
    signal : array-like
        Input signal
    bw : float
        Filter bandwidth, normalized units
    os : float
        Oversampling factor
    center_freq : float
        center frequency, normalized units. Default is DC centered operation

    Returns
    -------
    s : array-like
        Output signal after filtering

    """
    # Prepare signal
    N = len(signal)
    h = np.zeros(N, dtype=sig.real.dtype)
    freq_axis = scifft.fftfreq(N, 1 / os)

    # Create filter window
    idx = np.where(abs(freq_axis-center_freq) < bw / 2)
    h[idx] = 1
    
    # Filter and output
    s = (scifft.ifft(scifft.fft(signal) * h))
    return s

def filter_signal(signal, fs, cutoff, ftype="bessel", order=2, analog=False):
    """
    Apply an analog filter to a signal for simulating e.g. electrical bandwidth limitation

    Parameters
    ----------

    signal  : array_like
        input signal array
    fs      : float
        sampling frequency of the input signal
    cutoff  : float
        3 dB cutoff frequency of the filter
    ftype   : string, optional
        filter type can be either a bessel, butter, exp or gauss filter (default=bessel)
    order   : int
        order of the filter

    Returns
    -------
    signalout : array_like
        filtered output signal
    """
    sig = np.atleast_2d(signal)
    if ftype == "gauss":
        f = np.linspace(-fs/2, fs/2, sig.shape[1], endpoint=False, dtype=sig.dtype)
        w = cutoff/(2*np.sqrt(2*np.log(2))) # might need to add a factor of 2 here do we want FWHM or HWHM?
        g = np.exp(-f**2/(2*w**2))
        fsignal = scifft.fftshift(scifft.fft(scifft.fftshift(sig, axes=-1), axis=-1), axes=-1) * g
        if signal.ndim == 1:
            return scifft.fftshift(scifft.ifft(scifft.fftshift(fsignal))).flatten()
        else:
            return scifft.fftshift(scifft.ifft(scifft.fftshift(fsignal)))
    if ftype == "exp":
        f = np.linspace(-fs/2, fs/2, sig.shape[1], endpoint=False, dtype=sig.dtype)
        w = cutoff/(np.sqrt(2*np.log(2)**2)) # might need to add a factor of 2 here do we want FWHM or HWHM?
        g = np.exp(-np.sqrt((f**2/(2*w**2))))
        g /= g.max()
        fsignal = scifft.fftshift(scifft.fft(scifft.fftshift(signal))) * g
        if signal.ndim == 1:
            return scifft.fftshift(scifft.ifft(scifft.fftshift(fsignal))).flatten()
        else:
            return scifft.fftshift(scifft.ifft(scifft.fftshift(fsignal)))
    Wn = cutoff*2*np.pi if analog else cutoff
    frmt = "ba" if analog else "sos"
    fs_in = None if analog else fs
    if ftype == "bessel":
        system = scisig.bessel(order, Wn,  'low', norm='mag', analog=analog, output=frmt, fs=fs_in)
    elif ftype == "butter":
        system = scisig.butter(order, Wn, 'low',  analog=analog, output=frmt, fs=fs_in)
    if analog:
        t = np.arange(0, sig.shape[1])*1/fs
        sig2 = np.zeros_like(sig)
        for i in range(sig.shape[0]):
            to, yo, xo = scisig.lsim(system, sig[i], t)
            sig2[i] = yo.astype(sig.dtype)
    else:
        sig2 = scisig.sosfilt(system.astype(sig.dtype), sig, axis=-1)
    if signal.ndim == 1:
        return sig2.flatten()
    else:
        return sig2

def _rrcos_pulseshaping_freq(sig, fs, T, beta):
    """
    Root-raised cosine filter in the spectral domain by multiplying the fft of the signal with the
    frequency response of the rrcos filter.

    Parameters
    ----------
    sig    : array_like
        input time distribution of the signal
    fs    : float
        sampling frequency of the signal
    T     : float
        width of the filter (typically this is the symbol period)
    beta  : float
        filter roll-off factor needs to be in range [0, 1]

    Returns
    -------
    sign_out : array_like
        filtered signal in time domain
    """
    f = scifft.fftfreq(sig.shape[0])*fs
    nyq_fil = rrcos_freq(f, beta, T)
    nyq_fil /= nyq_fil.max()
    sig_f = scifft.fft(sig)
    sig_out = scifft.ifft(sig_f*nyq_fil)
    return sig_out

def rrcos_pulseshaping(sig, fs, T, beta, taps=1001):
    """
    Root-raised cosine filter applied in the time domain using fftconvolve.

    Parameters
    ----------
    sig    : array_like
        input time distribution of the signal
    fs    : float
        sampling frequency of the signal
    T     : float
        width of the filter (typically this is the symbol period)
    beta  : float
        filter roll-off factor needs to be in range [0, 1]
    taps : int, optional
        number of filter taps (if None do a spectral domain filter)

    Returns
    -------
    sign_out : array_like
        filtered signal in time domain
    """
    if taps is None:
        return _rrcos_pulseshaping_freq(sig, fs, T, beta)
    t = np.linspace(0, taps, taps, endpoint=False)
    t -= t[(t.size-1)//2]
    t /= fs
    nqt = rrcos_time(t, beta, T)
    nqt /= nqt.max()
    if sig.ndim > 1:
        sig_out = np.zeros_like(sig)
        for i in range(sig.shape[0]):
            sig_out[i] = scisig.fftconvolve(sig[i], nqt, 'same')
    else:
        sig_out = scisig.fftconvolve(sig, nqt, 'same')
    return sig_out


def moving_average(sig, N=3):
    """
    Moving average of signal

    Parameters
    ----------

    sig : array_like
        Signal for moving average
    N: number of averaging samples

    Returns
    -------

    mvg : array_like
        Average signal of length len(sig)-n+1
    """
    sign = np.atleast_2d(sig)
    ret = np.cumsum(np.insert(sign, 0,0, axis=-1), dtype=sig.dtype, axis=-1)
    if sig.ndim == 1:
        return ((ret[:, N:] - ret[:,:-N])/N).flatten()
    else:
        return (ret[:, N:] - ret[:,:-N])/N
