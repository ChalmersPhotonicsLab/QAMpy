#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File to contain DSP-based compensation functions to overcome analog impairment
prior to the ADC in the receiver. 

Created on Thu May 25 17:22:49 2017

@author: mazurm
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


#TODO: this should be fixed for multi-dim
def comp_rf_delay(signal, delay, sampling_rate = 50e9 ):
    """
    Adds a delay of X picoseconds to the signal in frequency domain. Can be 
    used to compensate for impairments such as RF cables of different length 
    between the optical hybrid and ADC. 

    Parameters
    ----------
        signal : array_like
            Real-valued input signal
        delay : float
            Delay in ps
        sampling_rate : scalar, optional
            ADC sampling rate

    Returns
    -------
        sig_out : array_like
            Signal after compensating for delay
    
    """
    
    # Frequency base vector
    freqVector = np.fft.fftfreq(signal.size, sampling_rate/2)
    
    # Phase-dealyed version
    sig_out = np.fft.ifft(np.exp(-1j*2*np.pi*delay*1e-12*freqVector)*\
                          np.fft.fft(signal))
    #TODO: the factor 1e-12 should really be cleaned up

    # Real part of output
    return sig_out.real
