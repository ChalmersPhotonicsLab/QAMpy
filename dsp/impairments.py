import numpy as np


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
    omega = 2*np.pi*np.linspace(-fs/2, fs/2, field.shape[1], endpoints=False)
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
    sigout = np.zeros(sig.shape, dtype=sig.dtype)
    if sig.ndim > 1:
        N = sig.shape[1]
        for i in range(sig.shape[0]):
            sigout[i] = sig[i] + strgth * (np.random.randn(N) + 1.j*np.random.randn(N))/np.sqrt(2) # sqrt(2) because of var vs std
    else:
        N = sig.shape[0]
        sigout = sig + strgth * (np.random.randn(N) + 1.j*np.random.randn(N))/np.sqrt(2) # sqrt(2) because of var vs std
    return sigout
