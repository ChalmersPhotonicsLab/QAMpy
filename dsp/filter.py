import numpy as np
from utils import rrcos_freq
import scipy.signal as scisig

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
    N = len(signal)
    h = np.zeros(N, dtype=np.float64)
    h[int(N/(bw/2)):-int(N/(bw/2))] = 1
    s = np.fft.ifft(np.fft.ifftshift(np.fft.fftshift(np.fft.fft(signal))*h))
    return s

def filter_signal(signal, fs, cutoff, ftype="bessel", order=2):
    nyq = 0.5*fs
    cutoff_norm = cutoff/nyq
    b, a = scisig.bessel(order, cutoff_norm, 'low', norm='mag', analog=False)
    y = scisig.lfilter(b, a, signal)
    return y

def filter_signal_analog(signal, fs, cutoff, ftype="bessel", order=2):
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
    if ftype == "gauss":
        f = np.linspace(-fs/2, fs/2, signal.shape[0], endpoint=False)
        w = cutoff/(2*np.sqrt(2*np.log(2))) # might need to add a factor of 2 here do we want FWHM or HWHM?
        g = np.exp(-f**2/(2*w**2))
        fsignal = np.fft.fftshift(np.fft.fft(np.fft.fftshift(signal))) * g
        return np.fft.fftshift(np.fft.ifft(np.fft.fftshift(fsignal)))
    if ftype == "exp":
        f = np.linspace(-fs/2, fs/2, signal.shape[0], endpoint=False)
        w = cutoff/(np.sqrt(2*np.log(2)**2)) # might need to add a factor of 2 here do we want FWHM or HWHM?
        g = np.exp(-np.sqrt((f**2/(2*w**2))))
        g /= g.max()
        fsignal = np.fft.fftshift(np.fft.fft(np.fft.fftshift(signal))) * g
        return np.fft.fftshift(np.fft.ifft(np.fft.fftshift(fsignal)))
    if ftype == "bessel":
        system = scisig.bessel(order, cutoff*2*np.pi, 'low', norm='mag', analog=True)
    elif ftype == "butter":
        system = scisig.butter(order, cutoff*2*np.pi, 'low', norm='mag', analog=True)
    t = np.arange(0, signal.shape[0])*1/fs
    to, yo, xo = scisig.lsim(system, signal, t)
    return yo

def rrcos_pulseshaping(sig, fs, T, beta):
    """
    Root-raised cosine filter applied in the spectral domain.

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
    f = np.linspace(-fs/2, fs/2, sig.shape[0], endpoint=False)
    nyq_fil = rrcos_freq(f, beta, T))
    nyq_fil /= nyq_fil.max()
    sig_f = np.fft.fftshift(np.fft.fft(np.fft.fftshift(sig)))
    sig_out = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(sig_f*nyq_fil)))
    return sig_out