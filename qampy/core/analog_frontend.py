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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DSP-based compensation functions to overcome analog impairments
prior to the ADC in the receiver. 
"""

import numpy as np


def comp_IQ_inbalance(signal):
    """
    Compensate for imbalance between I and Q from an optical hybrid. Takes I 
    as the real part and orthogonalize Q with respect to it. 
    """
    
    # Center signal around a mean of 0
    signal -= np.mean(signal)
    I = signal.real
    Q = signal.imag

    # phase balance
    mon_signal = np.sum(I*Q)/np.sum(I**2)
    phase_inbalance = np.arcsin(-mon_signal)
    Q_balcd = (Q + np.sin(phase_inbalance)*I)/np.cos(phase_inbalance)
    
    # Amplidue imbalance
    amp_inbalance = np.sum(I**2)/np.sum(Q_balcd**2)

    # Build output
    comp_singal = I + 1.j * (Q_balcd * np.sqrt(amp_inbalance))
    
    return comp_singal

def comp_rf_delay(signal, delay, sampling_rate=50e9 ):
    """
    Adds a delay of X picoseconds to the signal in frequency domain. Can be 
    used to compensate for impairments such as RF cables of different length 
    between the optical hybrid and ADC. 

    Parameters
    ----------
    signal : array_like
        Real-valued input signal
    delay : float
        Delay  in s
    sampling_rate : scalar, optional
        ADC sampling rate

    Returns
    -------
    sig_out : array_like
        Signal after compensating for delay
    
    """

    sig = np.atleast_2d(signal)
    # Frequency base vector
    freqVector = np.fft.fftfreq(sig.shape[1], sampling_rate/2)
    
    # Phase-delayed version
    sig_out = np.empty_like(sig)
    sig_out = np.fft.ifft(np.exp(-1j*2*np.pi*delay*freqVector)*\
                          np.fft.fft(sig, axis=1))
    # Real part of output
    if signal.ndim > 1:
        return sig_out.real
    else:
        return sig_out.real.flatten()


def orthonormalize_signal(E, os=1):
    """
    Orthogonalizing signal using the Gram-Schmidt process _[1].

    Parameters
    ----------
    E : array_like
       input signal
    os : int, optional
        oversampling ratio of the signal

    Returns
    -------
    E_out : array_like
        orthonormalized signal

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process for more
       detailed description.
    """

    E = np.atleast_2d(E)
    E_out = np.empty_like(E)
    for l in range(E.shape[0]):
        # Center
        real_out = E[l,:].real - E[l,:].real.mean()
        tmp_imag = E[l,:].imag - E[l,:].imag.mean()

        # Calculate scalar products
        mean_pow_inphase = np.mean(real_out**2)
        mean_pow_quadphase = np.mean(tmp_imag**2)
        mean_pow_imb = np.mean(real_out*tmp_imag)

        # Output, Imag orthogonal to Real part of signal
        sig_out = real_out / np.sqrt(mean_pow_inphase) +\
                                    1j*(tmp_imag - mean_pow_imb * real_out / mean_pow_inphase) / np.sqrt(mean_pow_quadphase)
        # Final total normalization to ensure IQ-power equals 1
        E_out[l,:] = sig_out - np.mean(sig_out[::os])
        E_out[l,:] = E_out[l,:] / np.sqrt(np.mean(np.abs(E_out[l,::os])**2))

    return E_out