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
Functions for the simulation of transmission and transceiver impairments.
"""
import numpy as np
import warnings
from qampy.helpers import normalise_and_center, rescale_signal
from qampy.core.filter import filter_signal
from scipy import interpolate, fft
from qampy.core.digital_pre_compensation import clipper

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
    h2 = np.array([[np.exp(-1.j*omega*t_dgd/2), np.zeros(len(omega))],[np.zeros(len(omega)), np.exp(1.j*omega*t_dgd/2)]])
    #h3 = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    #h3 = h1.T
    h3 =np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])
    H = np.einsum('ij,jkl->ikl', h1, h2)
    #H = np.einsum('ijl,jk->ikl', H, h3)
    return H, h3

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

def _applyPMD_einsum(field, H, h3):
    Sf = fft.fftshift(fft.fft(fft.ifftshift(field, axes=1),axis=1), axes=1)
    SSf = np.einsum('ijk,ik -> ik',H , Sf)
    SS = fft.fftshift(fft.ifft(fft.ifftshift(SSf, axes=1),axis=1), axes=1)
    SS = np.dot(h3, SS)
    try:
        return field.recreate_from_np_array(SS.astype(field.dtype))
    except:
        return SS.astype(field.dtype)

def _applyPMD_dot(field, theta, t_dgd, omega):
    Sf = fft.fftshift(fft.fft(fft.ifftshift(field, axes=1),axis=1), axes=1)
    Sff = rotate_field(Sf, theta)
    h2 = np.array([np.exp(-1.j*omega*t_dgd/2),  np.exp(1.j*omega*t_dgd/2)])
    Sn = Sff*h2
    Sf2 = rotate_field(Sn, -theta)
    SS = fft.fftshift(fft.ifft(fft.ifftshift(Sf2, axes=1), axis=1), axes=1)
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
    return _applyPMD_dot(field, theta, t_dgd, omega)

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
    if len(f.shape) > 1:
        return np.cumsum(f, axis=1)
    else:
        return np.cumsum(f)

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
    return add_awgn(sig, np.sqrt(p)*n)

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


def simulate_transmission(sig, fb, fs, snr=None, freq_off=None, lwdth=None, dgd=None, theta=np.pi/3.731, modal_delay=None, roll_frame_sync=False):
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

    if roll_frame_sync:
        if not (sig.nframes > 1):
            warnings.warn("Only single frame present, discontinuity introduced")
        sig = np.roll(sig,sig.pilots.shape[1],axis=-1)
    if lwdth is not None:
        sig = apply_phase_noise(sig, lwdth, fs)
    if freq_off is not None:
        sig = add_carrier_offset(sig, freq_off, fs)
    if snr is not None:
        sig = change_snr(sig, snr, fb, fs)
    if modal_delay is not None:
        sig = add_modal_delay(sig, modal_delay)
    if dgd is not None:
        sig = apply_PMD_to_field(sig, theta, dgd, fs)
    return sig

def quantize_signal(sig, nbits=6, rescale=True, re_normalize=True):
    """
    Function so simulate limited resultion using DACs and ADCs

    Parameters:
        sig:            Input signal, numpy array
        nbits:          Quantization resultion
        rescale:        Rescale to range p/m 1, default True
        re_normalize:   Renormalize signal after quantization

    Returns:
        sig_out:        Output quantized waveform

    """
    # Create a 2D signal
    sig = np.atleast_2d(sig)
    npols=sig.shape[0]

    # Rescale to
    if rescale:
        for pol in range(npols):
            sig[pol] /= np.abs(sig[pol]).max()


    levels = np.linspace(-1,1,2**(nbits))

    sig_out = np.zeros(sig.shape, dtype=sig.dtype)
    for pol in range(npols):
        sig_quant_re = levels[np.digitize(sig[pol].real,levels[:-1],right=False)]
        sig_quant_im = levels[np.digitize(sig[pol].imag,levels[:-1],right=False)]
        sig_out[pol] = sig_quant_re + 1j*sig_quant_im

    if not np.iscomplexobj(sig):
        sig_out = sig_out.real

    if re_normalize:
        sig_out = normalise_and_center(sig_out)

    return sig_out

def quantize_signal_New(sig_in, nbits=6, rescale_in=True, rescale_out=True):
    """
    Function so simulate limited resultion using DACs and ADCs, limit quantization error to (-delta/2,delta/2) and set
        decision threshold at mid-point between two quantization levels.

    Parameters:
        sig_in:            Input signal, numpy array, notice: input signal should be rescale to (-1,1)
        nbits:          Quantization resolution
        rescale_in:        Rescale input signal to (-1,1)
        rescale_out:       Rescale output signal to (-input_max_swing,input_max_swing)
    Returns:
        sig_out:        Output quantized waveform

    """
    # 2**nbits interval within (-1,1), output swing is (-1+delta/2,1-delta/2)
    # Create a 2D signal
    sig_in = np.atleast_2d(sig_in)
    npols = sig_in.shape[0]

    # Rescale to
    sig = np.zeros((npols, sig_in.shape[1]), dtype=sig_in.dtype)
    if rescale_in:
        sig = rescale_signal(sig_in, swing=1)

    # Clipping exist if signal range is larger than (-1,1)
    swing = 2
    delta = swing/2**nbits
    levels_out = np.linspace(-1+delta/2, 1-delta/2, 2**nbits)
    levels_dec = levels_out + delta/2

    sig_out = np.zeros(sig.shape, dtype=sig_in.dtype)
    for pol in range(npols):
        sig_quant_re = levels_out[np.digitize(sig[pol].real, levels_dec[:-1], right=False)]
        sig_quant_im = levels_out[np.digitize(sig[pol].imag, levels_dec[:-1], right=False)]
        sig_out[pol] = sig_quant_re + 1j * sig_quant_im

    if not np.iscomplexobj(sig):
        sig_out = sig_out.real

    if rescale_out:
        max_swing = np.maximum(abs(sig_in.real).max(), abs(sig_in.imag).max())
        sig_out = sig_out * max_swing

    return sig_in.recreate_from_np_array(sig_out)

def modulator_response(rfsig, dcbias=1, gfactr=1, cfactr=0, dcbias_out=0.5, gfactr_out=1):
    """
    Function so simulate IQ modulator response.

    Parameters
    ----------
    rfsig:  array_like
        complex version of the I (real part) and Q (imaginary part) of the signal
    dcsig:  complex or float, optional
            DC bias for I (real) and Q (imaginary) channel. If dcsig is real use the same DC bias for I and Q
    vpi:   complex or float, optional
            Vpi of the MZM (zero-power point) in I (real)  and Q (imaginary) channel. If vpi is real use the  same Vpi
            for both.
    gfactr: complex or float, optional
            Split imbalance and path dependent loss of I (real) and Q (imaginary) MZM. 
            An ideal MZM with infinite extinction ratio has gfactor=1. If gfactr is real use the same value for both I
            and Q.
    cfactr:  complex or float, optional
           Chirp factors of I (real) and (Q) channel MZMs, caused by the asymmetry in the electrode design of the MZM. 
           cfactr = 0 for ideal MZM.
    dcbias_out : float, optional
            DCBias of the outer MZM
    gfactr_out : float, optional
            gain factor of the outer MZM

    Returns
    -------
    e_out: array_like
            Output signal of IQ modulator. (i.e. Here assume that input laser power is 0 dBm)

    """

    if not np.iscomplex(dcbias):
        dcbias = dcbias + 1j*dcbias
    if not np.iscomplex(gfactr):
        gfactr = gfactr + 1j*gfactr
    if not np.iscomplex(cfactr):
        cfactr = cfactr + 1j*cfactr
    volt = rfsig.real + dcbias.real + 1j * (rfsig.imag + dcbias.imag)
    # Use the minus sign (-) to modulate lower level RF signal to corresponding Low-level optical field, if V_bias = Vpi
    e_i = -(np.exp(1j * np.pi * volt.real * (1 + cfactr.real) / 2) +
            gfactr.real * np.exp(-1j * np.pi * volt.real * (1 - cfactr.real) / 2)) / (1 + gfactr.real)
    e_q = -(np.exp(1j * np.pi * volt.imag * (1 + cfactr.imag) / 2) +
            gfactr.imag * np.exp(-1j * np.pi * volt.imag * (1 - cfactr.imag) / 2)) / (1 + gfactr.imag)
    e_out = np.exp(1j * np.pi / 4) * (e_i * np.exp(-1j * np.pi * dcbias_out/2 ) +
                                      gfactr_out * e_q * np.exp(1j * np.pi * dcbias_out / 2)) / (1 + gfactr_out)
    return e_out

def er_to_g(ext_rat):
    """

    Parameters
    ----------
    ext_rat:

    Returns
    -------

    """
    g = (10**(ext_rat/20)-1)/(10**(ext_rat/20)+1)
    return g

def sim_DAC_response(sig, fs, enob=5, clip_rat=1, quant_bits=0, **dac_params):
    """
    Function to simulate DAC response, including quantization noise (ENOB) and frequency response.
    
    Parameters
    ----------
    sig:  array_like
        Input signal 
    fs: float
        Sampling frequency of the signal
    enob: float, optional
        Effective number of bits of the DAC (i.e. 6 bits.) modelled as AWGN. If enob=0 only quantize. 
        If both enob and quant_bits are given, quantize first and then add enob noise.
    clip_rat: float, optional
        Ratio of signal left after clipping. (i.e. clip_rat=0.8 means 20% of the signal is clipped) (default 1: no clipping)
    quant_bits: float, optional
        Number of bits in the quantizer, only applied if not =0. (Default: don't qpply quantization)
    dac_params: dict, optional
        Parameters for the DAC response check apply_DAC_filter for the keyword parameters. If this is 
        empty than do not apply the DAC response
    
    Returns
    -------
    filter_sig:  array_like
        Quantized, clipped and filtered output signal
    """
    if np.isclose(clip_rat, 1):
        sig_clip = sig
    else:
        sig_res = rescale_signal(sig, 1/clip_rat)
        sig_clip = clipper(sig_res, 1)
    if not np.isclose(quant_bits, 0):
        sig_clip = quantize_signal_New(sig_clip, nbits=quant_bits, rescale_in=True, rescale_out=True)
    if not np.isclose(enob, 0):
        sig_clip = apply_enob_as_awgn(sig_clip, enob)
    if dac_params:
        filter_sig = apply_DAC_filter(sig_clip, fs, **dac_params)
    else:
        filter_sig = sig_clip
    return filter_sig

def apply_DAC_filter(sig, fs, cutoff=18e9, fn=None, ch=1):
    """
    Apply the frequency response filter of the DAC. This function
    uses either a 2nd order Bessel filter or a measured frequency response
    loaded from a file.

    Parameters
    ----------
    sig : array_like
        signal to be filtered. Can be real or complex
    fs : float
        sampling rate of the signal
    cutoff : float, optional
        Cutoff frequency used by only by Bessel filter
    fn : string, optional
        filename of a experimentally measured response, if None use a Bessel
        filter approximation
    ch : int, optional
        channel number of the measured response to use
    Returns
    -------
    filter_sig : array_like
        filtered signal
    """
    # filtering was split into real and imaginary before but that should not be necessary
    if fn is None:
        filter_sig = filter_signal(sig, fs, cutoff, ftype="bessel", order=2)
    else:
        H_dac = load_dac_response(fn, fs, sig.shape[-1], ch=ch).astype(sig) # should check if response is real
        sigf = fft.fft(sig)
        filter_sig = fft.ifft(sigf * H_dac)
    return filter_sig

def apply_enob_as_awgn(sig, enob, verbose=False):
    """
    Add noise from limited ENOB as modelled as AWGN to signal.
    
    Parameters
    ----------
    sig : array_like
        input signal
    enob : float
        Effective Number of Bits of the ADC or DAC
    verbose : bool, optional
        Wether to return additional information
    Returns
    -------
    if verbose is True:
        sig_enob_noise, snr_enob
    else:
        sig_enob_noise
        
    sig_enob_noise: array_like
        signal with added noise
    snr_enob: float
        SNR corresponding to the ENOB in dB
    """
    powsig_mean = np.mean(abs(sig)**2) 
    if np.iscomplexobj(sig):
        x_max = np.maximum(abs(sig.real).max(), abs(sig.imag).max())    # maximum amplitude in real or imag part
    else:
        x_max = abs(sig).max()           # maximum amplitude of the signal
    delta = x_max / 2**(enob-1)
    pownoise_mean = delta ** 2 / 12
    sig_enob_noise = add_awgn(sig, np.sqrt(2*pownoise_mean))  # add two-time noise power to complex signal
    snr_enob = 10*np.log10(powsig_mean/2/pownoise_mean)  # Use half of the signal power to calculate snr
    if verbose:
        return sig_enob_noise, snr_enob
    else:
        return sig_enob_noise

def load_dac_response(fn, fs, N, ch=1):
    """
    Load the measured dac response and adjust it to target sampling frequency.

    Parameters
    ---------- 
    fn : string
        filename of the resposne to be loaded
    fs: float
        sampling rate
    N : int
        length of the output vector
    ch: int, optional
        Which of the DAC channels to load the response for
    
    Returns
    -------
    dacf_interp : array_like
        frequency response of the DAC for channel ch with vector length N
    """
    npzfile = np.load(fn)
    dac_f = npzfile['dac_res_ch%d' % ch]
    # Copy spectrum to the negative frequency side
    dacf_complex = np.atleast_2d(dac_f[:, 1] * np.exp(1j * dac_f[:, 2]))
    dacf = np.concatenate((np.fliplr(np.conj(dacf_complex[:, 1:])), dacf_complex), axis=1)
    dac_freq = np.concatenate((np.fliplr(-np.atleast_2d(dac_f[1:, 0])), np.atleast_2d(dac_f[:, 0])), axis=1)
    freq_sig_fft = fft.fftfreq(N)*fs
    # Interpolate the dac response, do zero-padding if fs/2 > 32 GHz
    polyfit = interpolate.interp1d(dac_freq.flatten(), dacf.flatten(), kind='linear', bounds_error=False, fill_value=dac_f[320, 1])
    dacf_interp = polyfit(freq_sig_fft)
    dacf_interp = np.atleast_2d(dacf_interp)
    return dacf_interp

def sim_tx_response(sig, fs, enob=6, tgt_v=1, clip_rat=1, quant_bits=0, dac_params={"cutoff":18e9, "fn": None, "ch":None}, **mod_prms):
    """
    Simulate a realistic transmitter possibly including quantization, noise due to limited ENOB, 
    and DAC frequency response

    Parameters
    ----------
    sig: array_like
        Input signal used for transmission
    fs: float
        Sampling frequency of signal
    enob: float, optional
        efficient number of bits for DAC. If enob=0 only use quantizer. Unit: bits
    tgt_v : float, optional
        target Voltage in fraction of Vpi
    clip_rat: float, optional
        Ratio of signal left after clipping. (i.e. clip_rat=0.8 means 20% of the signal is clipped) (default 1: no clipping)
    quant_bits: float, optional
        Number of bits in the quantizer, only applied if not =0. (Default: don't qpply quantization)    
    dac_params: dict, optional
        parameters to pass to the DAC filter
    mod_prms: dict, optional
        parameters to pass to the modulator (see modulator response for details)
    
    Returns
    -------
    e_out: array_like
        Signal with TX impairments
    """
    # Apply signal to DAC model
    sig_dac_out = sim_DAC_response(sig, fs, enob,  clip_rat=clip_rat, quant_bits=quant_bits, **dac_params)
    # Amplify the signal to target voltage(V)
    sig_amp = ideal_amplifier_response(sig_dac_out, tgt_v)
    e_out = modulator_response(sig_amp, **mod_prms)
    return e_out

def ideal_amplifier_response(sig, out_volt):
    """
    Simulate a ideal amplifier, which just scale RF signal to out_volt.
    Parameters
    ----------
    sig
    out_volt

    Returns
    -------

    """
    current_volt = max(abs(sig.real).max(), abs(sig.imag).max())
    return sig / current_volt * out_volt

def add_dispersion(sig, fs, D, L, wl0=1550e-9):
    """
    Add dispersion to signal.

    Parameters
    ----------
    sig : array_like
        input signal
    fs : flaot
        sampling frequency of the signal (in SI units)
    D : float
        Dispersion factor in s/m/m
    L : float
        Length of the dispersion in m
    wl0 : float,optional
        center wavelength of the signal

    Returns
    -------
    sig_out : array_like
        dispersed signal
    """
    C = 2.99792458e8
    N = sig.shape[-1]
    omega = fft.fftfreq(N, 1/fs)*np.pi*2
    beta2 = D * wl0**2 / (C*np.pi*2)
    H = np.exp(-0.5j * omega**2 * beta2 * L).astype(sig.dtype)
    sff = fft.fft(fft.ifftshift(sig, axes=-1), axis=-1)
    sig_out = fft.fftshift(fft.ifft(sff*H))
    return sig_out

