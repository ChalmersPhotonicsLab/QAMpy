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
from qampy.helpers import normalise_and_center, rescale_signal
from qampy.core.filter import filter_signal
from scipy import interpolate
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
    sig = np.zeros((npols, sig_in.shape[1]), dtype=sig_in.dtype)
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

def modulator_response(rfsig, dcbias=3.5, vpi=3.5, gfactr=1, cfactr=0, prms_outer=(3.5/2, 3.5, 1)):
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
    prms_outer: array_like, optional
            DCBias, Vpi and gain factor of the outer MZM.

    Returns
    -------
    e_out: array_like
            Output signal of IQ modulator. (i.e. Here assume that input laser power is 0 dBm)

    """

    if not np.iscomplex(dcbias):
        dcbias = dcbias + 1j*dcbias
    if not np.iscomplex(vpi):
        vpi = vpi + 1j*vpi
    if not np.iscomplex(gfactr):
        gfactr = gfactr + 1j*gfactr
    if not np.iscomplex(cfactr):
        cfactr = cfactr + 1j*cfactr
    volt = rfsig.real + dcbias.real + 1j * (rfsig.imag + dcbias.imag)

    dcbias_outer, vpi_outer, gfactr_outer = prms_outer
    # Use the minus sign (-) to modulate lower level RF signal to corresponding Low-level optical field, if V_bias = Vpi
    e_i = -(np.exp(1j * np.pi * volt.real * (1 + cfactr.real) / (2 * vpi.real)) +
            gfactr.real * np.exp(-1j * np.pi * volt.real * (1 - cfactr.real) / (2 * vpi.real))) / (1 + gfactr.real)
    e_q = -(np.exp(1j * np.pi * volt.imag * (1 + cfactr.imag) / (2 * vpi.imag)) +
            gfactr.imag * np.exp(-1j * np.pi * volt.imag * (1 - cfactr.imag) / (2 * vpi.imag))) / (1 + gfactr.imag)
    e_out = np.exp(1j * np.pi / 4) * (e_i * np.exp(-1j * np.pi * dcbias_outer / (2 * vpi_outer)) +
                                      gfactr_outer * e_q * np.exp(1j * np.pi * dcbias_outer / (2 * vpi_outer))) / (1 + gfactr_outer)
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

def sim_DAC_response(sig, fs, enob, cutoff, quantizer_model=True, clipping=False, clip_rat=1, quan_and_enob=False):
    """
    Function to simulate DAC response, including quantization noise (ENOB) and frequency response.
    Parameters
    ----------
    sig:  array_like
        Input signal
    fs: float
        Sampling frequency of the signal
    enob: float
        Effective number of bits of the DAC     (i.e. 6 bits.)
    cutoff: float
        3-dB cutoff frequency of DAC. (i.e. 16 GHz.)
    quantizer_model:  Bool, optional
        if quantizer_model='True', use quantizer model to simulate quantization noise, else
        if quantizer_model='False', use AWGN model to simulate quantization noise.
    clipping: Bool, optional
        if clipping='True', clips the signal based on clip_rat, else
        if clipping='False', no clipping operation.
    clip_rat: float, optional
        Ratio of signal left after clipping. (i.e. clip_rat=0.8 means 20% of the signal is clipping)
    Returns
    -------
    filter_sig:  array_like
        Quantized and filtered output signal
    snr_enob_noise: array_like
        noise caused by the limited ENOB
    snr_enob:    float
        signal-to-noise-ratio induced by ENOB.
    """

    if clipping:
        sig_res = rescale_signal(sig, 1/clip_rat)
        sig_clip = clipper(sig_res, 1)
    else:
        sig_clip = sig

    if quan_and_enob:
        sig_clip = quantize_signal_New(sig_clip, nbits=8, rescale_in=True, rescale_out=True)

    powsig_mean = (abs(sig_clip) ** 2).mean()  # mean power of the real signal
    # Apply dac model to real signal
    if not np.iscomplexobj(sig_clip):
        if quantizer_model:
            sig_enob_noise = quantize_signal_New(sig_clip, nbits=enob, rescale_in=True, rescale_out=True)
            pownoise_mean = (abs(sig_enob_noise-sig_clip)**2).mean()
            snr_enob = 10*np.log10(powsig_mean/pownoise_mean)  # unit:dB
        else:
            # Add AWGN noise due to ENOB
            x_max = abs(sig_clip).max()           # maximum amplitude of the signal
            delta = x_max / 2**(enob-1)
            pownoise_mean = delta ** 2 / 12
            sig_enob_noise = add_awgn(sig_clip, np.sqrt(pownoise_mean))
            snr_enob = 10*np.log10(powsig_mean/pownoise_mean)  # unit:dB
        # Apply 2-order bessel filter to simulate frequency response of DAC
        filter_sig = filter_signal(sig_enob_noise, fs, cutoff, ftype="bessel", order=2)
    # Apply dac model to complex signal
    else:
        if quantizer_model:
            sig_enob_noise = quantize_signal_New(sig_clip, nbits=enob, rescale_in=True, rescale_out=True)
            pownoise_mean = (abs(sig_enob_noise-sig_clip)**2).mean()  # include noise in real part and imag part
            snr_enob = 10*np.log10(powsig_mean/pownoise_mean)  # unit:dB
        else:
            # Add AWGN noise due to ENOB
            x_max = np.maximum(abs(sig_clip.real).max(), abs(sig_clip.imag).max())    # maximum amplitude in real or imag part
            delta = x_max / 2**(enob-1)
            pownoise_mean = delta ** 2 / 12
            sig_enob_noise = add_awgn(sig_clip, np.sqrt(2*pownoise_mean))  # add two-time noise power to complex signal
            snr_enob = 10*np.log10(powsig_mean/2/pownoise_mean)  # Use half of the signal power to calculate snr
        # Apply 2-order bessel filter to simulate frequency response of DAC
        filter_sig_re = filter_signal(sig_enob_noise.real, fs, cutoff, ftype="bessel", order=2)
        filter_sig_im = filter_signal(sig_enob_noise.imag, fs, cutoff, ftype="bessel", order=2)
        filter_sig = filter_sig_re + 1j * filter_sig_im
    return filter_sig, sig_enob_noise, snr_enob

def sim_AWG_dac_response(sig, enob, quantizer_model=True, ch=1, dac_volt=0.4):

    sig = np.atleast_2d(sig)
    powsig_mean = (abs(sig) ** 2).mean()  # mean power of the real signal
    # Load dac response
    dacf = load_dac_response(sig.fs, sig.shape[1], ch=ch, dac_volt=dac_volt)
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
        # Apply AWG dac response to simulate frequency response of DAC
        sigf = np.fft.fft(sig_enob_noise)
        filter_sig = np.fft.ifft(sigf * dacf)
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
        sigf_re = np.fft.fft(sig_enob_noise.real)
        sigf_im = np.fft.fft(sig_enob_noise.imag)
        filter_sig_re = np.fft.ifft(sigf_re * dacf)
        filter_sig_im = np.fft.ifft(sigf_im * dacf)
        filter_sig = filter_sig_re + 1j * filter_sig_im
    return filter_sig, sig_enob_noise, snr_enob, dacf

def load_dac_response(fs, freq_len, ch = 1, dac_volt = 0.40):
    """
    Load the measured dac response and adjust it to target sampling frequency.
    """
    main_path = 'C:/Users/Zonglong/PycharmProjects/EXP received data/Tx impairment/DAC frequency response/'
    fn = main_path + ('dac_frequency_response_vpp_%.2f'%dac_volt).replace('.', 'p') + '.npz'
    npzfile = np.load(fn)
    dac_f = npzfile['dac_res_ch%d' % ch]

    # Copy spectrum to the negative frequency side
    dacf_complex = np.atleast_2d(dac_f[:, 1] * np.exp(1j * dac_f[:, 2]))
    dacf = np.concatenate((np.fliplr(np.conj(dacf_complex[:, 1:])), dacf_complex), axis=1)
    dac_freq = np.concatenate((np.fliplr(-np.atleast_2d(dac_f[1:, 0])), np.atleast_2d(dac_f[:, 0])), axis=1)

    freq_sig_fft = np.fft.fftfreq(freq_len) * fs
    # Interpolate the dac response, do zero-padding if fs/2 > 32 GHz
    polyfit = interpolate.interp1d(dac_freq.flatten(), dacf.flatten(), kind='linear', bounds_error=False, fill_value=dac_f[320, 1])
    dacf_interp = polyfit(freq_sig_fft)
    dacf_interp = np.atleast_2d(dacf_interp)
    return dacf_interp


def sim_tx_response(sig, fs, enob=6, cutoff=16e9, tgt_v=3.5, p_in=0, clipping=False, clip_rat=1, quan_and_enob=False, **mod_prms):
    """

    Parameters
    ----------
    sig: array_like
        Input signal used for transmission
    fs: float
        Sampling frequency of signal
    enob: float, optional
        efficient number of bits for DAC. UnitL bits
    cutoff: float, optional
        3-dB cut-off frequency for DAC. Unit: GHz
    tgt_v : float, optional
        target Voltage
    p_in: float, optional
        Laser power input to IQ modulator in dBm. Default is set to 0 dBm.
    clipping: bool, optional
        Operate clipping in DAC or not. Default is False.
    Returns
    -------
    e_out: array_like
        Signal with TX impairments

    """
    # Apply signal to DAC model
    sig_dac_out, sig_enob_noise, snr_enob = sim_DAC_response(sig, fs, enob, cutoff, quantizer_model=False, clipping=clipping, clip_rat=clip_rat, quan_and_enob=quan_and_enob)
    # Amplify the signal to target voltage(V)
    sig_amp = ideal_amplifier_response(sig_dac_out, tgt_v)
    e_out = modulator_response(sig_amp, **mod_prms)
    power_out = 10 * np.log10(abs(e_out*np.conj(e_out)).mean() * (10 ** (p_in / 10)))
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
    omega = np.fft.fftfreq(N, 1/fs)*np.pi*2
    beta2 = D * wl0**2 / (C*np.pi*2)
    H = np.exp(-0.5j * omega**2 * beta2 * L)
    sff = np.fft.fft(np.fft.ifftshift(sig, axes=-1), axis=-1)
    sig_out = np.fft.fftshift(np.fft.ifft(sff*H))
    return sig_out

