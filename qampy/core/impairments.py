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
import warnings
from ..helpers import normalise_and_center, rescale_signal
from ..filtering import filter_signal_analog

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
    Sf = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(field, axes=1),axis=1), axes=1)
    SSf = np.einsum('ijk,ik -> ik',H , Sf)
    SS = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(SSf, axes=1),axis=1), axes=1)
    SS = np.dot(h3, SS)
    try:
        return field.recreate_from_np_array(SS.astype(field.dtype))
    except:
        return SS.astype(field.dtype)

def _applyPMD_dot(field, theta, t_dgd, omega):
    Sf = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(field, axes=1),axis=1), axes=1)
    Sff = rotate_field(Sf, theta)
    h2 = np.array([np.exp(-1.j*omega*t_dgd/2),  np.exp(1.j*omega*t_dgd/2)])
    Sn = Sff*h2
    Sf2 = rotate_field(Sn, -theta)
    SS = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Sf2, axes=1), axis=1), axes=1)
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
        sig = add_modal_delay(sig, delay)
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

    sig_out = np.zeros(sig.shape,dtype="complex")
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
    sig = np.zeros((npols,sig_in.shape[1]), dtype=sig_in.dtype)
    if rescale_in:
        for pol in range(npols):
            # notice: different pol may have different scale factor which cause power different between x and y -pol.
            sig[pol] = rescale_signal(sig_in[pol], swing=1)

    # Clipping exist if signal range is larger than (-1,1)
    swing = 2
    delta = swing/2**nbits
    levels_out = np.linspace(-1+delta/2, 1-delta/2, 2**nbits)
    levels_dec = levels_out + delta/2

    sig_out = np.zeros(sig.shape, dtype="complex")
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

def modulator_response(rfsig_i,rfsig_q,dcsig_i=3.5,dcsig_q=3.5,dcsig_p=3.5/2,vpi_i=3.5,vpi_q=3.5,vpi_p=3.5,gi=1,gq=1,gp=1,ai=0,aq=0):
    """
    Function so simulate IQ modulator response.

    Parameters
    ----------
    rfsig_i:  array_like
            RF input signal to I channel
    rfsig_q:  array_like
            RF input signal to Q channel
    dcsig_i:  float
            DC bias signal to I channel
    dcsig_q:  float
            DC bias signal to Q channel
    dcsig_p:  float
            DC bias signal to outer MZM used to control the phase difference of I anc Q signal
            Normally is set to vpi_p/2, which correspond to 90 degree
    vpi_i: float
            Vpi of the MZM (zero-power point) in I channel
    vpi_q: float
            Vpi of the MZM (zero-power point) in Q channel
    vpi_p: float
            Vpi of the outer MZM (zero-power point) used to control phase difference.
    gi: float
            Account for split imbalance and path dependent losses of I MZM. i.e. gi=1 for ideal MZM with infinite extinction ratio
    gq: float
            Account for split imbalance and path dependent losses of Q MZM
    gp: float
            Account for split imbalance and path dependent losses of Q MZM
    ai: float
            Chirp factors of I channel MZM, caused by the asymmetry in the electrode design of the MZM. i.e. ai = 0 for ideal MZM
    aq: float
            Chirp factors of Q channel MZM, caused by the asymmetry in the electrode design of the MZM

    Returns
    -------
    e_out: array_like
            Output signal of IQ modulator. (i.e. Here assume that input laser power is 0 dBm)

    """

    volt_i = rfsig_i + dcsig_i
    volt_q = rfsig_q + dcsig_q
    volt_p = dcsig_p
    # Use the minus sign (-) to modulate lower level RF signal to corresponding Low-level optical field, if V_bias = Vpi
    e_i = -(np.exp(1j*np.pi*volt_i*(1+ai)/(2*vpi_i)) + gi*np.exp(-1j*np.pi*volt_i*(1-ai)/(2*vpi_i)))/(1+gi)
    e_q = -(np.exp(1j*np.pi*volt_q*(1+aq)/(2*vpi_q)) + gq*np.exp(-1j*np.pi*volt_q*(1-aq)/(2*vpi_q)))/(1+gq)
    e_out = np.exp(1j*np.pi/4)*(e_i*np.exp(-1j*np.pi*volt_p/(2*vpi_p)) + gp*e_q*np.exp(1j*np.pi*volt_p/(2*vpi_p)))/(1+gp)
    return e_out

def er_to_g(ext_rat):
    """

    Parameters
    ----------
    ext_rat

    Returns
    -------

    """
    g = (10**(ext_rat/20)-1)/(10**(ext_rat/20)+1)
    return g

def DAC_response(sig, enob, cutoff, quantizer_model=True):
    """
    Function to simulate DAC response, including quantization noise (ENOB) and frequency response.
    Parameters
    ----------
    sig:              Input signal, signal object.
    enob:             Efficient number of bits      (i.e. 6 bits.)
    cutoff:           3-dB cutoff frequency of DAC. (i.e. 16 GHz.)
    quantizer_model:  if quantizer_model='true', use quantizer model to simulate quantization noise.
                      if quantizer_model='False', use AWGN model to simulate quantization noise.
    out_volt:         Targeted output amplitude of the RF signal.

    Returns
    -------
    filter_sig:     Quantized and filtered output signal
    snr_enob:       signal-to-noise-ratio induced by ENOB.
    """
    powsig_mean = (abs(sig) ** 2).mean()  # mean power of the real signal

    # Apply dac model to real signal
    if not np.iscomplexobj(sig):
        if quantizer_model:
            sig_enob_noise = quantize_signal_New(sig, nbits=enob, rescale_in=True, rescale_out=True)
            pownoise_mean = (abs(sig_enob_noise-sig)**2).mean()
            snr_enob = 10*np.log10(powsig_mean/pownoise_mean)  # unit:dB
        else:
            # Add AWGN noise due to ENOB
            x_max = abs(sig).max()           # maximum amplitude of the signal
            delta = x_max / 2**(enob-1)
            pownoise_mean = delta ** 2 / 12
            sig_enob_noise = add_awgn(sig, np.sqrt(pownoise_mean))
            snr_enob = 10*np.log10(powsig_mean/pownoise_mean)  # unit:dB

        # Apply 2-order bessel filter to simulate frequency response of DAC
        filter_sig = filter_signal_analog(sig_enob_noise, cutoff, ftype="bessel", order=2)

    # Apply dac model to complex signal
    else:
        if quantizer_model:
            sig_enob_noise = quantize_signal_New(sig, nbits=enob, rescale_in=True, rescale_out=True)
            pownoise_mean = (abs(sig_enob_noise-sig)**2).mean()  # include noise in real part and imag part
            snr_enob = 10*np.log10(powsig_mean/pownoise_mean)  # unit:dB
        else:
            # Add AWGN noise due to ENOB

            x_max = np.maximum(abs(sig.real).max(), abs(sig.imag).max())    # maximum amplitude in real or imag part
            delta = x_max / 2**(enob-1)
            pownoise_mean = delta ** 2 / 12
            sig_enob_noise = add_awgn(sig, np.sqrt(2*pownoise_mean))  # add two-time noise power to complex signal
            snr_enob = 10*np.log10(powsig_mean/2/pownoise_mean)  # Use half of the signal power to calculate snr

        # Apply 2-order bessel filter to simulate frequency response of DAC
        filter_sig_re = filter_signal_analog(sig_enob_noise.real, cutoff, ftype="bessel", order=2)
        filter_sig_im = filter_signal_analog(sig_enob_noise.imag, cutoff, ftype="bessel", order=2)
        filter_sig = filter_sig_re + 1j* filter_sig_im

    return filter_sig, sig_enob_noise, snr_enob

def Simulate_transmitter_response(sig, enob=6, cutoff=16e9, target_voltage=3.5, power_in=0):
    """

    Parameters
    ----------
    sig: array_like
            Input signal used for transmission
    enob: float
            efficient number of bits for DAC. UnitL bits
    cutoff: float
            3-dB cut-off frequency for DAC. Unit: GHz
    power_in: float
            Laser power input to IQ modulator. Default is set to 0 dBm.
    Returns
    -------

    """
    # Apply signal to DAC model
    [sig_dac_out, sig_enob_noise, snr_enob] = DAC_response(sig, enob, cutoff, quantizer_model=False)

    # Amplify the signal to target voltage(V)
    sig_amp = ideal_amplifier_response(sig_dac_out, target_voltage)

    # Input quantized signal to IQ modulator
    rfsig_i = sig_amp.real
    rfsig_q = sig_amp.imag

    e_out = modulator_response(rfsig_i, rfsig_q, dcsig_i=3.5, dcsig_q=3.5, dcsig_p=3.5 / 2, vpi_i=3.5, vpi_q=3.5, vpi_p=3.5,
                       gi=1, gq=1, gp=1, ai=0, aq=0)
    power_out = 10 * np.log10( abs(e_out*np.conj(e_out)).mean() * (10 ** (power_in / 10)))

    # return e_out, power_out, snr_enob_i, snr_enob_q
    return e_out

def ideal_amplifier_response(sig,out_volt):
    """
    Simulate a ideal amplifier, which just scale RF signal to out_volt.
    Parameters
    ----------
    sig
    out_volt

    Returns
    -------

    """
    current_volt = abs(sig).max()
    return sig / current_volt * out_volt
