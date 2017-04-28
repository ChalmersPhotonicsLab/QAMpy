from __future__ import division, print_function
import numpy as np
import scipy.signal as scisig
import fractions
""" a number of convenience functions"""


def cabssquared(x):
    """Calculate the absolute squared of a complex number"""
    return x.real**2 + x.imag**2

def dB2lin(x):
    """
    Convert input from dB(m) units to linear units
    """
    return 10**(x/10)

def lin2dB(x):
    """
    Convert input from linear units to dB(m)
    """
    return 10*np.log10(x)

def ttanh(x, A, x0, w):
    """
    Calculate the hyperbolic tangent with a given amplitude, zero offset and
    width.

    Parameters
    ----------
    x : array_like
        Input array variable
    A : float
        Amplitude
    x0 : float
        Zero-offset
    w : float
        Width

    Returns
    -------
    array_like
        calculated array
    """
    return A * tanh((x - x0) / w)


def gauss(x, A, x0, w):
    """
    Calculate the Gaussian function with a given amplitude, zero offset and
    width.

    Parameters
    ----------
    x : array_like
        Input array variable
    A : float
        Amplitude
    x0 : float
        Zero offset
    w : float
        Width

    Returns
    -------
    array_like
        calculated array
    """
    return A * np.exp(-((x - x0) / w)**2 / 2.)


def supergauss(x, A, x0, w, o):
    """
    Calculate the Supergaussian functions with a given amplitude,
    zero offset, width and order.

    Parameters
    ----------
    x : array_like
        Input array variable
    A : float
        Amplitude
    x0 : float
        Zero offset
    w : float
        Width
    o : float
        order of the supergaussian

    Returns
    -------
    array_like
        calculated array
    """
    return A * np.exp(-((x - x0) / w)**(2 * o) / 2.)

def normalise_and_center(E):
    """
    Normalise and center the input field, by calculating the mean power for each polarisation separate and dividing by its square-root
    """
    if E.ndim > 1:
        for i in range(E.shape[0]):
            E[i] -= np.mean(E[i])
            P = np.sqrt(np.mean(cabssquared(E[i])))
            E[i] /= P
    else:
        E = E.real - np.mean(E.real) + 1.j * (E.imag-np.mean(E.imag))
        P = np.sqrt(np.mean(cabssquared(E)))
        E /= P
    return E

def sech(x, A, x0, w):
    """
    Calculate the hyperbolic secant function with a given
    amplitude, zero offset and width.

    Parameters
    ----------
    x : array_like
        Input array variable
    A : float
        Amplitude
    x0 : float
        Zero offset
    w : float
        Width

    Returns
    -------
    array_like
        calculated array
    """
    return A / np.cosh((x - x0) / w)


def factorial(n):
    """The factorial of n, i.e. n!"""
    if n == 0: return 1
    return n * factorial(n - 1)


def linspacestep(start, step, N):
    """
    Create an array of given length for a given start and step
    value.

    Parameters
    ----------
    start : float
        first value to start with
    step : float
        size of the step
    N : int
        number of steps

    Returns
    -------
    out : array_like
        array of length N from start to start+N*step (not included)
    """
    return np.arange(start, start + N * step, step=step)

def dump_edges(E, N):
    """
    Remove N samples from the front and end of the input field.
    """
    return E[N:-N]

def lfsr_int(seed, mask):
    """
    A linear feedback shift register, using Galois or internal XOR
    implementation.

    Parameters
    ----------
    seed : int
        an integer representing the list of bits as the starting point of the
        register. Length N
    mask : int
        Determines the polynomial of the shift register (length N+1). The
        first and last bit of the mask must be 1.

    Yields
    ------
    xor : int
        output bit of the register
    state : int
        state of the register
    """
    state = seed
    nbits = mask.bit_length() - 1
    while True:
        state = (state << 1)
        xor = state >> nbits
        #the modulus operation on has an effect if the last bit is 1
        if xor != 0:
            state ^= mask  #this performs the modulus operation
        yield xor, state

def lfsr_ext(seed, taps, nbits):
    """A Fibonacci or external XOR linear feedback shift register.

    Parameters
    ----------
    seed : int
        binary number denoting the input state registers
    taps  : list
        list of registers that are input to the XOR (length 2)
    nbits : int
        number of registers

    Yields
    ------
    xor : int
        output bit of the registers
    state : int
        state of the register
    """
    sr = seed
    while 1:
        xor = 0
        for t in taps:
            if (sr & (1 << (nbits - t))) != 0:
                xor ^= 1
        sr = (xor << nbits - 1) + (sr >> 1)
        yield xor, sr

def bool2bin(x):
    """
    Convert an array of boolean values into a binary number. If the input
    array is not a array of booleans it will be converted.
    """
    assert len(x) < 64, "array must not be longer than 63"
    x = np.asarray(x, dtype=bool)
    y = 0
    for i, j in enumerate(x):
        y += j << i
    return y


def find_offset(sequence, data):
    """
    Find index where binary sequence occurs fist in the binary data array

    Parameters
    ----------
    sequence : array_like
        sequence to search for inside the data
    data : array_like
        data array in which to find the sequence

    It is required that len(data) > sequence

    Returns
    -------
    idx : int
        index where sequence first occurs in data
    """
    assert len(data) > len(sequence), "data has to be longer than sequence"
    if not data.dtype == sequence.dtype:
        raise Warning("""data and sequence are not the same dtype, converting
        data to dtype of sequence""")
        data = data.astype(sequence.dtype)
    # using this string conversion method is much faster than array methods,
    # however it only finds the first occurence
    return data.tostring().index(sequence.tostring()) // data.itemsize


def rolling_window(data, size):
    """
    Reshapes a 1D array into a 2D array with overlapping frames. Stops when the
    last value of data is reached.

    Parameters
    ----------
    data : array_like
        Data array to segment
    size : int
        The frame size

    Returns
    -------
    out : array_like
        output segmented 2D array


    Examples
    >>> utils.rolling_window(np.arange(10), 3)
    array([[0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6],
            [5, 6, 7],
            [6, 7, 8],
            [7, 8, 9]])
    """
    shape = data.shape[:-1] + (data.shape[-1] - size + 1, size)
    strides = data.strides + (data.strides[-1], )
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

def _resamplingfactors(fold, fnew):
    ratn = fractions.Fraction(fnew/fold).limit_denominator()
    return ratn.numerator, ratn.denominator

def resample(signal, fold, fnew, window=None):
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
        sig_new = normalise_and_center(sig_new)
    return sig_new[::down]

def rrcos_resample_poly(signal, fold, fnew, Ts=None, beta=None, discardfactor=1e-3):
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
    discardfactor : float, optional
        discard filter elements below this threshold, this causes the filter to be significantly shorter
        and thus speeds up the function significantly

    Returns
    -------
    sig_out  : array_like
        resampled output signal

    """
    if beta is None:
        return resample(signal, fold, fnew)
    if Ts is None:
        Ts = 1/fold
    else:
        ratn = fractions.Fraction(fnew/fold).limit_denominator()
        fup = ratn.numerator*fold
        Nup = signal.shape[0]*ratn.numerator
        t = np.linspace(-Nup/2, Nup/2, Nup, endpoint=False)*1/fup
        nqf = rrcos_time(t, beta, Ts)
        nqf /= nqf.max()
        nqf = nqf[np.where(abs(nqf)>discardfactor)]
        return resample(signal, fold, fnew, window=nqf)

def rcos_time(t, beta, T):
    """Time response of a raised cosine filter with a given roll-off factor and width """
    return np.sinc(t / T) * np.cos(t / T * np.pi * beta) / (1 - 4 *
                                                            (beta * t / T)**2)

def rcos_freq(f, beta, T):
    """Frequency response of a raised cosine filter with a given roll-off factor and width """
    rc = np.zeros(f.shape[0], dtype=f.dtype)
    rc[np.where(np.abs(f) <= (1 - beta) / (2 * T))] = T
    idx = np.where((np.abs(f) > (1 - beta) / (2 * T)) & (np.abs(f) <= (
        1 + beta) / (2 * T)))
    rc[idx] = T / 2 * (1 + np.cos(np.pi * T / beta *
                                                     (np.abs(f[idx]) - (1 - beta) /
                                                      (2 * T))))
    return rc

def rrcos_freq(f, beta, T):
    """Frequency transfer function of the square-root-raised cosine filter with a given roll-off factor and time width/sampling period after _[1]

    Parameters
    ----------

    f   : array_like
        frequency vector
    beta : float
        roll-off factor needs to be between 0 and 1 (0 corresponds to a sinc pulse, square spectrum)

    T   : float
        symbol period

    Returns
    -------
    y   : array_like
       filter response

    References
    ----------
    ..[1] B.P. Lathi, Z. Ding Modern Digital and Analog Communication Systems
    """
    return np.sqrt(rcos_freq(f, beta, T))

def rrcos_time(t, beta, T):
    """Time impulse response of the square-root-raised cosine filter with a given roll-off factor and time width/sampling period after _[1]
    This implementation differs by a factor 2 from the previous.

    Parameters
    ----------

    t   : array_like
        time vector
    beta : float
        roll-off factor needs to be between 0 and 1 (0 corresponds to a sinc pulse, square spectrum)

    T   : float
        symbol period

    Returns
    -------
    y   : array_like
       filter response

    References
    ----------
    ..[1] https://en.wikipedia.org/wiki/Root-raised-cosine_filter
    """
    rrcos = 1/T*((np.sin(np.pi*t/T*(1-beta)) +  4*beta*t/T*np.cos(np.pi*t/T*(1+beta)))/(np.pi*t/T*(1-(4*beta*t/T)**2)))
    eps = abs(t[0]-t[1])/4
    idx1 = np.where(abs(t)<eps)
    rrcos[idx1] = 1/T*(1+beta*(4/np.pi-1))
    idx2 = np.where(abs(abs(t)-abs(T/(4*beta)))<eps)
    rrcos[idx2] = beta/(T*np.sqrt(2))*((1+2/np.pi)*np.sin(np.pi/(4*beta))+(1-2/np.pi)*np.cos(np.pi/(4*beta)))
    return rrcos

def bin2gray(value):
    """
    Convert a binary value to an gray coded value see _[1]. This also works for arrays.
    ..[1] https://en.wikipedia.org/wiki/Gray_code#Constructing_an_n-bit_Gray_code
    """
    return value^(value >> 1)

def convert_iqtosinglebitstream(idat, qdat, nbits):
    """
    Interleave a two bitstreams into a single bitstream with nbits per symbol. This can be used to create a combined PRBS signal from 2 PRBS sequences for I and Q channel. If nbits is odd we will use nbits//2 + 1 bits from the first stream and nbits//2 from the second.

    Parameters
    ----------
    idat    : array_like
        input data stream (1D array of booleans)
    qdat    : array_like
        input data stream (1D array of booleans)
    nbits   : int
        number of bits per symbol that we want after interleaving

    Returns
    -------
    output   : array_like
        interleaved bit stream
    """
    if nbits%2:
        N = [nbits//2+1, nbits//2]
    else:
        N = [nbits//2, nbits//2]
    idat_n = idat[:len(idat)-(len(idat)%N[0])]
    idat_n = idat_n.reshape(N[0], len(idat_n)/N[0])
    qdat_n = qdat[:len(qdat)-(len(qdat)%N[1])]
    qdat_n = qdat_n.reshape(N[1], len(qdat_n)/N[1])
    l = min(len(idat_n[0]), len(qdat_n[0]))
    return np.hstack([idat_n[:l], qdat_n[:l]]).flatten()

def H_PMD(theta, t_dgd, omega): #see Ip and Kahn JLT 25, 2033 (2007)
    """
    Calculate the response for PMD applied to the signal

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
    h = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.dot(h, field)

def _applyPMD(field, H):
    Sf = np.fft.fftshift(np.fft.fft(np.fft.fftshift(field, axes=1),axis=1), axes=1)
    SSf = np.einsum('ijk,ik -> ik',H , Sf)
    SS = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(SSf, axes=1),axis=1), axes=1)
    return SS

def apply_PMD_to_field(field, theta, t_dgd, omega):
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

    omega : array_like
        angular frequency of the light field

    Returns
    -------
    out  : array_like
       new dual polarisation field with PMD
    """
    H = H_PMD(theta, t_dgd, omega)
    return _applyPMD(field, H)

def phase_noise(N, df, fs):
    """
    Calculate phase noise from local oscillators, based on a Wiener noise process.

    Parameters
    ----------

    N  : integer
        length of the phase noise vector

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
    f = np.random.normal(scale=np.sqrt(var), size=N)
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
    N = signal.shape[0]
    ph = phase_noise(N, df, fs)
    return signal*np.exp(1.j*ph)

def comp_IQbalance(signal):
    """
    Compensate IQ imbalance of a signal
    """
    signal -= np.mean(signal)
    I = signal.real
    Q = signal.imag

    # phase balance
    mon_signal = np.sum(I*Q)/np.sum(I**2)
    phase_inbalance = np.arcsin(-mon_signal)
    Q_balcd = (Q + np.sin(phase_inbalance)*I)/np.cos(phase_inbalance)
    am_bal = np.sum(I**2)/np.sum(Q_balcd**2)
    Q_comp = Q_balcd * np.sqrt(am_bal)
    return I + 1.j * Q_comp

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
    nyq_fil = np.sqrt(rcos_freq(f, beta, T))
    nyq_fil /= nyq_fil.max()
    sig_f = np.fft.fftshift(np.fft.fft(np.fft.fftshift(sig)))
    sig_out = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(sig_f*nyq_fil)))
    return sig_out

