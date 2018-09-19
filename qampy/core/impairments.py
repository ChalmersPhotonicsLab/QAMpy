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
import numpy as np


def H_PMD(theta, t_dgd, omega):
    """
    Calculate the response for PMD applied to the signal (see e.g. _[1])

    Parameters
    ----------

    theta : float
        angle of the principle axis to the observed axis

    t_dgd : float
        differential group delay between the polarisation axes

    omega : array_like
        angular frequency of the light field

    Returns
    -------
    H : matrix
        response matrix for applying dgd

    References
    ----------
    .. [1] Ip and Kahn JLT 25, 2033 (2007)
    """
    h1 = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    h2 = np.array([[np.exp(1.j*omega*t_dgd/2), np.zeros(len(omega))],[np.zeros(len(omega)), np.exp(-1.j*omega*t_dgd/2)]])
    h3 = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    H = np.einsum('ij,jkl->ikl', h1, h2)
    H = np.einsum('ijl,jk->ikl', H, h3)
    return H

def rotate_field(field, theta):
    """
    Rotate a dual polarisation field by the given angle

    Parameters
    ----------

    field : array_like
        input dual polarisation optical field (first axis is polarisation)

    theta : float
        angle to rotate by

    Returns
    -------
    rotated_field : array_like
        new rotated field
    """
    h = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype="f%d"%(field.itemsize//2))
    return np.dot(h, field)

def _applyPMD(field, H):
    Sf = np.fft.fftshift(np.fft.fft(np.fft.fftshift(field, axes=1),axis=1), axes=1)
    SSf = np.einsum('ijk,ik -> ik',H , Sf)
    SS = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(SSf, axes=1),axis=1), axes=1)
    try:
        return field.recreate_from_np_array(SS.astype(field.dtype))
    except:
        return SS.astype(field.dtype)

def apply_PMD_to_field(field, theta, t_dgd, fs):
    """
    Apply PMD to a given input field

    Parameters
    ----------

    field : array_like
        input dual polarisation optical field (first axis is polarisation)

    theta : float
        angle of the principle axis to the observed axis

    t_dgd : float
        differential group delay between the polarisation axes

    fs : float
        sampling rate of the field

    Returns
    -------
    out  : array_like
       new dual polarisation field with PMD
    """
    omega = 2*np.pi*np.linspace(-fs/2, fs/2, field.shape[1], endpoint=False)
    H = H_PMD(theta, t_dgd, omega)
    return _applyPMD(field, H)

def phase_noise(sz, df, fs):
    """
    Calculate phase noise from local oscillators, based on a Wiener noise process with a variance given by :math:`\sigma^2=2\pi df/fs`

    Parameters
    ----------

    sz  : tuple
        size of the phase noise array

    df : float
        combined linewidth of local oscillators in the system

    fs : float
        sampling frequency of the signal

    Returns
    -------
    phase : array_like
       randomly varying phase term

    """
    var = 2*np.pi*df/fs
    f = np.random.normal(scale=np.sqrt(var), size=sz)
    return np.cumsum(f, axis=1)

#TODO: make multi-dim phase noise configurable
def apply_phase_noise(signal, df, fs):
    """
    Add phase noise from local oscillators, based on a Wiener noise process.

    Parameters
    ----------

    signal  : array_like
        single polarisation signal

    df : float
        combined linewidth of local oscillators in the system

    fs : float
        sampling frequency of the signal

    Returns
    -------
    out : array_like
       output signal with phase noise

    """
    N = signal.shape
    ph = phase_noise(N, df, fs)
    return signal*np.exp(1.j*ph).astype(signal.dtype)

def add_awgn(sig, strgth):
    """
    Add additive white Gaussian noise to a signal.

    Parameters
    ----------
    sig    : array_like
        signal input array can be 1d or 2d, if 2d noise will be added to every dimension
    strgth : float
        the strength of the noise to be added to each dimension

    Returns
    -------
    sigout : array_like
       output signal with added noise

    """
    return sig + (strgth * (np.random.randn(*sig.shape) + 1.j*np.random.randn(*sig.shape))/np.sqrt(2)).astype(sig.dtype) # sqrt(2) because of var vs std

#TODO: we should check that this is correct both when the signal is oversampled or not
def change_snr(sig, snr, fb, fs):
    """
    Change the SNR of a signal assuming that the input signal is noiseless

    Parameters
    ----------
    sig : array_like
        the signal to change
    snr : float
        the desired signal to noise ratio in dB
    fb  : float
        the symbol rate
    fs  : float
        the sampling rate

    Returns
    -------
    sig : array_like
        output signal with given SNR
    """
    os = fs/fb
    p = np.mean(abs(sig)**2)
    n = 10 ** (-snr / 20) * np.sqrt(os)
    return add_awgn(sig, p*n)

def add_carrier_offset(sig, fo, fs):
    """
    Add frequency offset to signal

    Parameters
    ----------
    sig : array_like
        signal input array
    df : float
        frequency offset
    fs : float
        sampling rate

    Returns
    -------
    signal : array_like
        signal with added offset
    """
    sign = np.atleast_2d(sig)
    if sig.ndim == 1:
        return  (sign * np.exp(2.j * np.pi * np.arange(sign.shape[1], dtype=sign.dtype) * fo / fs)).flatten()
    else:
        return  sign * np.exp(2.j * np.pi * np.arange(sign.shape[1], dtype=sign.dtype) * fo / fs)

def add_modal_delay(sig, delay):
    """
    Add a modal delay of n-symbols to modes of signal, e.g. to simulate a fake-pol mux system.

    Parameters
    ----------
    sig : array_like
        input signal
    delay : array_like
        array of delays given in number of samples

    Returns
    -------
    sig_out : array_like
        output signal where each mode has been shifted by the appropriate delay
    """
    nmodes = sig.shape[0]
    delay = np.asarray(delay)
    assert delay.shape[0] == sig.shape[0], "Delay array must have the same length as number of modes of signal "
    sig_out = sig.copy()
    for i in range(nmodes):
        sig_out[i] = np.roll(sig_out[i], delay[i], axis=-1)
    return sig_out


def simulate_transmission(sig, fb, fs, snr=None, freq_off=None, lwdth=None, dgd=None, theta=np.pi/3.731, modal_delay=None):
    """
    Convenience function to simulate impairments on signal at once

    Parameters
    ----------
    sig : array_like
        input signal
    fb  : flaot
        symbol rate of the signal
    fs  : float
        sampling rate of the signal
    snr : flaat, optional
        desired signal-to-noise ratio of the signal. (default: None, don't change SNR)
    freq_off : float, optional
        apply a carrier offset to signal (default: None, don't apply offset)
    lwdth : float, optional
        linewidth of the transmitter and LO lasers (default: None, infinite linewidth)
    dgd : float, optional
        first-order PMD (differential group delay) (default: None, do not apply PMD)
    theta : float, optional
        rotation angle to principle states of polarization
    modal_delay : array_like, optional
        add a delay given in N samples to the signal (default: None, do not add delay)
    Returns
    -------
    signal : array_like
        signal with transmission impairments applied
    """
    if lwdth is not None:
        sig = apply_phase_noise(sig, lwdth, fs)
    if freq_off is not None:
        sig = add_carrier_offset(sig, freq_off, fs)
    if snr is not None:
        sig = change_snr(sig, snr, fb, fs)
    if modal_delay is not None:
        sig = add_modal_delay(sig, delay)
    if dgd is not None:
        sig = apply_PMD_to_field(sig, theta, dgd, fs)
    return sig

