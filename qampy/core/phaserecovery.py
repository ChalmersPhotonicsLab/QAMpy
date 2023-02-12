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
Functions for carrier offset and phase estimation and correction functions
"""

from __future__ import division, print_function
import numpy as np
from qampy.core.segmentaxis import segment_axis
from qampy.core.signal_quality import cal_s0
#from qampy.core.dsp_cython import bps as _bps_idx_pyx
from qampy.core.pythran_dsp import select_angles
from qampy.core.pythran_dsp import bps as _bps_idx_pyt
#from qampy.core.dsp_cython import select_angles

from qampy.core.filter import moving_average
try:
    import arrayfire as af
except ImportError:
    af = None

NMAX = 4*1024**3

def viterbiviterbi(E, N, M):
    """
    Viterbi-Viterbi blind phase recovery for an M-PSK signal

    Parameters
    ----------
    E : array_like
        the electric field of the signal
    N : int
        number of samples to average over
    M : int
        order of the M-PSK

    Returns
    -------
    Eout : array_like
        Field with compensated phases
    """
    E2d = np.atleast_2d(E)
    Eout = np.zeros_like(E2d)
    L = E2d.shape[1]
    for i in range(E2d.shape[0]):
        Et = E2d[i]
        L = len(Et)
        phi = np.angle(Et)
        E_raised = np.exp(1.j * phi)**M
        sa = segment_axis(E_raised, N, N - 1)
        phase_est = np.sum(sa, axis=1)
        phase_est = np.unwrap(np.angle(phase_est))
        phase_est = (phase_est - np.pi)/M
        if N % 2:
            Eout[i, (N - 1) // 2:L - (N - 1) // 2] = E2d[i, (N - 1) // 2:L - (N - 1) // 2] * np.exp(-1.j*phase_est)
        else:
            Eout[i, N // 2 - 1:L - (N // 2)] = E2d[i, N // 2 - 1:L - (N // 2)] * np.exp(-1.j*phase_est)
    #if M == 4: # QPSK needs pi/4 shift
    # need a shift by pi/M for constellation points to not be on the axis
    if E.ndim == 1:
        return  Eout.flatten(), phase_est.flatten()
    else:
        return Eout, phase_est

def _bps_idx_py(E, angles, symbols, N):
    """
    Blind phase search using Python. This is slow compared to the cython and arrayfire methods and should not be used.
    """
    EE = E[:,np.newaxis]*np.exp(1.j*angles)
    idx = np.zeros(E.shape[0], dtype=np.int32)
    dist = (abs(EE[:, :, np.newaxis]-symbols)**2).min(axis=2)
    csum = np.cumsum(dist, axis=0)
    mvg = csum[2*N:]-csum[:-2*N]
    idx[N:-N] = mvg.argmin(1)
    return idx

def bps(E, Mtestangles, symbols, N, method="pyt", **kwargs):
    """
    Perform a blind phase search after _[1]

    Parameters
    ----------

    E           : array_like
        input signal (single polarisation)

    Mtestangles : int
        number of test angles to try

    symbols     : array_like
        the symbols of the modulation format

    N           : int
        block length to use for averaging

    method      : string, optional
        implementation method to use has to be "af" for arrayfire (uses OpenCL) or "pyx" for a cython-OpenMP based parallel search or "py" for slow python search

    **kwargs    :
        arguments to be passed to the search function

    Returns
    -------
    Eout    : array_like
        signal with compensated phase
    ph      : array_like
        unwrapped angle from phase recovery

    References
    ----------
    ..[1] Timo Pfau et al, Hardware-Efficient Coherent Digital Receiver Concept With Feedforward Carrier Recovery for M-QAM Constellations, Journal of Lightwave Technology 27, pp 989-999 (2009)
    """
    if method.lower() == "pyx":
        bps_fct = _bps_idx_pyx
    elif method.lower() == "af":
        if af == None:
            raise RuntimeError("Arrayfire was not imported so cannot use it")
        bps_fct = _bps_idx_af
    elif method.lower() == "py":
        bps_fct = _bps_idx_py
    elif method.lower() == "pyt":
        bps_fct = _bps_idx_pyt
    else:
        raise ValueError("Method needs to be 'pyx', 'py' or 'af'")
    if E.dtype is np.dtype(np.complex64):
        dtype = np.float32
    else:
        dtype = np.float64
    angles = np.linspace(-np.pi/4, np.pi/4, Mtestangles, endpoint=False, dtype=dtype).reshape(1,-1)
    Ew = np.atleast_2d(E).astype(E.dtype)
    ph = []
    for i in range(Ew.shape[0]):
        idx =  bps_fct(Ew[i], angles, symbols, N)
        ph.append(select_angles(np.copy(angles), idx.astype(int)))
    ph = np.asarray(ph, dtype=dtype)
    # ignore the phases outside the averaging window
    # better to use normal unwrap instead of fancy tricks
    #ph[N:-N] = unwrap_discont(ph[N:-N], 10*np.pi/2/Mtestangles, np.pi/2)
    ph[:, N:-N] = np.unwrap(ph[:, N:-N]*4)/4
    if E.ndim == 1:
        return (Ew*np.exp(1.j*ph)).flatten(), ph.flatten()
    else:
        return Ew*np.exp(1.j*ph), ph

def _movavg_af(X, N, axis=0):
    """
    Calculate moving average over N samples using arrayfire
    """
    cs = af.accum(X, dim=axis)
    return cs[N:] - cs[:-N]

def _bps_idx_af(E, angles, symbols, N):
    global NMAX
    prec_dtype = E.dtype
    Ntestangles = angles.shape[1]
    Nmax = NMAX//Ntestangles//symbols.shape[0]//16
    L = E.shape[0]
    EE = E[:,np.newaxis]*np.exp(1.j*angles)
    syms  = af.np_to_af_array(symbols.astype(prec_dtype).reshape(1,1,-1))
    idxnd = np.zeros(L, dtype=np.int32)
    if L <= Nmax+N:
        Eaf = af.np_to_af_array(EE.astype(prec_dtype))
        tmp = af.min(af.abs(af.broadcast(lambda x,y: x-y, Eaf[0:L,:], syms))**2, dim=2)
        cs = _movavg_af(tmp, 2*N, axis=0)
        val, idx = af.imin(cs, dim=1)
        idxnd[N:-N] = np.array(idx)
    else:
        K = L//Nmax
        R = L%Nmax
        if R < N:
            R = R+Nmax
            K -= 1
        Eaf = af.np_to_af_array(EE[0:Nmax+N].astype(prec_dtype))
        tmp = af.min(af.abs(af.broadcast(lambda x,y: x-y, Eaf, syms))**2, dim=2)
        tt = np.array(tmp)
        cs = _movavg_af(tmp, 2*N, axis=0)
        val, idx = af.imin(cs, dim=1)
        idxnd[N:Nmax] = np.array(idx)
        for i in range(1,K):
            Eaf = af.np_to_af_array(EE[i*Nmax-N:(i+1)*Nmax+N].astype(np.complex128))
            tmp = af.min(af.abs(af.broadcast(lambda x,y: x-y, Eaf, syms))**2, dim=2)
            cs = _movavg_af(tmp, 2*N, axis=0)
            val, idx = af.imin(cs, dim=1)
            idxnd[i*Nmax:(i+1)*Nmax] = np.array(idx)
        Eaf = af.np_to_af_array(EE[K*Nmax-N:K*Nmax+R].astype(np.complex128))
        tmp = af.min(af.abs(af.broadcast(lambda x,y: x-y, Eaf, syms))**2, dim=2)
        cs = _movavg_af(tmp, 2*N, axis=0)
        val, idx = af.imin(cs, dim=1)
        idxnd[K*Nmax:-N] = np.array(idx)
    return idxnd

def bps_af(E, testangles, symbols, N, **kwargs):
    """
    Arrayfire based blind phase search. See bps for parameters
    """
    return bps(E, testangles, symbols, N, method="af", **kwargs)


def bps_pyx(E, testangles, symbols, N, **kwargs):
    """
    Cython based blind phase search. See bps for parameters
    """
    return bps(E, testangles, symbols, N, method="pyx", **kwargs)


def bps_twostage(E, Mtestangles, symbols, N , B=4, method="pyt", **kwargs):
    """
    Perform a blind phase search phase recovery using two stages after _[1]

    Parameters
    ----------

    E           : array_like
        input signal (single polarisation)

    Mtestangles : int
        number of initial test angles to try

    symbols     : array_like
        the symbols of the modulation format

    N           : int
        block length to use for averaging

    B           : int, optional
        number of second stage test angles

    method      : string, optional
        implementation method to use has to be "af" for arrayfire (uses OpenCL) or
        "pyx" for a Cython-OpenMP based parallel search, "pyt" for pythran  or "py" for a slower python based search

    **kwargs    :
        arguments to be passed to the search function

    Returns
    -------
    Eout    : array_like
        phase compensated field
    ph      : array_like
        unwrapped angle from phase recovery

    References
    ----------
    ..[1] Qunbi Zhuge and Chen Chen and David V. Plant, Low Computation Complexity Two-Stage Feedforward Carrier Recovery Algorithm for M-QAM, Optical Fiber Communication Conference (OFC, 2011)
    """
    if method.lower() == "pyx":
        bps_fct = _bps_idx_pyx
    elif method.lower() == "af":
        bps_fct = _bps_idx_af
    elif method.lower() == "pyt":
        bps_fct = _bps_idx_pyt
    elif method.lower() == "py":
        bps_fct = _bps_idx_py
    else:
        raise ValueError("Method needs to be 'pyx', 'py' or 'af'")
    angles = np.linspace(-np.pi/4, np.pi/4, Mtestangles, endpoint=False, dtype=E.real.dtype).reshape(1,-1)
    Ew = np.atleast_2d(E)
    ph_out = []
    for i in range(Ew.shape[0]):
        idx =  bps_fct(np.copy(Ew[i]), angles, symbols, N, **kwargs)
        ph = select_angles(np.copy(angles), idx)
        b = np.linspace(-B/2, B/2, B)
        phn = (ph[:,np.newaxis] + b[np.newaxis,:]/(B*Mtestangles)*np.pi/2).astype(E.real.dtype)
        idx2 = bps_fct(np.copy(Ew[i]), phn, symbols, N, **kwargs)
        phf = select_angles(np.copy(phn), idx2)
        ph_out.append(np.unwrap(phf*4, discont=np.pi*4/4)/4)
    ph_out = np.asarray(ph_out, dtype=E.real.dtype)
    En = Ew*np.exp(1.j*ph_out)
    if E.ndim == 1:
        return En.flatten(), ph_out.flatten()
    else:
        return En, ph_out



def partition_16qam(E):
    r"""
    Partition a 16-QAM signal into the inner and outer circles.

    Separates a 16-QAM signal into the inner and outer rings, which have
    different phase orientations. Detailed in _[1].

    Parameters
    ----------
        E : array_like
            electric field of the signal

    Returns
    -------
        class1_mask : array_like
            A mask designating the class 1 symbols which are the smallest and
            largest rings.
        class2_mask : array_like
            A mask designating the class 2 symbols which lie on the middle ring

    References
    ----------
    .. [1] R. Muller and D. D. A. Mello, “Phase-offset estimation for
       joint-polarization phase-recovery in DP-16-QAM systems,” Photonics
       Technol. Lett. …, vol. 22, no. 20, pp. 1515–1517, 2010.
    """

    S0 = cal_s0(E, 1.32)
    inner = (np.sqrt(S0 / 5) + np.sqrt(S0)) / 2.
    outer = (np.sqrt(9 * S0 / 5) + np.sqrt(S0)) / 2.
    Ea = abs(E)
    class1_mask = (Ea < inner) | (Ea > outer)
    class2_mask = ~class1_mask
    return class1_mask, class2_mask


def phase_partition_16qam(E, Nblock):
    r"""16-QAM blind phase recovery using QPSK partitioning.

    A blind phase estimator for 16-QAM signals based on partitioning the signal
    into 3 rings, which are then phase estimated using traditional V-V phase
    estimation after Fatadin et al _[1].

    Parameters
    ----------
        E : array_like
            electric field of the signal
        Nblock : int
            number of samples in an averaging block

    Returns
    -------
        E_rec : array_like
            electric field of the signal with recovered phase.
        phase : array_like
            recovered phase array

    References
    ----------
    .. [1] I. Fatadin, D. Ives, and S. Savory, “Laser linewidth tolerance
       for 16-QAM coherent optical systems using QPSK partitioning,”
       Photonics Technol. Lett. IEEE, vol. 22, no. 9, pp. 631–633, May 2010.

    """
    E2d = np.atleast_2d(E)
    dphi = np.pi / 4 + np.arctan(1 / 3)
    modes, L = E2d.shape
    ph_out = []
    for j in range(modes):
        # partition QPSK signal into qpsk constellation and non-qpsk const
        c1_m, c2_m = partition_16qam(E2d[j])
        Sx = np.zeros(L, dtype=np.complex128)
        Sx[c2_m] = (E2d[j, c2_m] * np.exp(1.j * dphi))**4
        So = np.zeros(L, dtype=np.complex128)
        So[c2_m] = (E2d[j, c2_m] * np.exp(1.j * -dphi))**4
        S1 = np.zeros(L, dtype=np.complex128)
        S1[c1_m] = (E2d[j, c1_m])**4
        E_n = np.zeros(L, dtype=np.complex128)
        phi_est = np.zeros(L, dtype=np.float64)
        for i in range(0, L, Nblock):
            S1_sum = np.sum(S1[i:i + Nblock])
            Sx_tmp = np.min(
                [S1_sum - Sx[i:i + Nblock], S1_sum - So[i:i + Nblock]],
                axis=0)[c2_m[i:i + Nblock]]
            phi_est[i:i + Nblock] = np.angle(S1_sum + Sx_tmp.sum())
        ph_out.append(np.unwrap(phi_est) / 4 - np.pi / 4)
    phi_out = np.asarray(ph_out)
    if E.ndim == 1:
        return (E2d * np.exp(-1.j * phi_est)).flatten(), phi_out.flatten()
    else:
        return E2d * np.exp(-1.j * phi_est), phi_out


def find_freq_offset(sig, os=1, average_over_modes = True, fft_size = 2**16):
    """
    Find the frequency offset by searching in the spectrum of the signal
    raised to 4. Doing so eliminates the modulation for QPSK but the method also
    works for higher order M-QAM.

    Parameters
    ----------
        sig : array_line
            signal array with N modes
        os: int
            oversampling ratio (Samples per symbols in sig)
        average_over_modes : bool
            Using the field in all modes for estimation

        fft_size: array
            Size of FFT used to estimate. Should be power of 2, otherwise the
            next higher power of 2 will be used.

    Returns
    -------
        freq_offset : int
            found frequency offset

    """
    if not((np.log2(fft_size)%2 == 0) | (np.log2(fft_size)%2 == 1)):
        fft_size = 2**(int(np.ceil(np.log2(fft_size))))

    # Fix number of stuff
    sig = np.atleast_2d(sig)
    npols, L = sig.shape

    # Find offset for all modes
    freq_sig = np.zeros([npols,fft_size])
    for l in range(npols):
        freq_sig[l,:] = np.abs(np.fft.fft(sig[l,:]**4,fft_size))**2

    # Extract corresponding FO
    freq_offset = np.zeros([npols,1])
    freq_vector = np.fft.fftfreq(fft_size,1/os)/4
    for k in range(npols):
        max_freq_bin = np.argmax(np.abs(freq_sig[k,:]))
        freq_offset[k,0] = freq_vector[max_freq_bin]


    if average_over_modes:
        freq_offset = np.mean(freq_offset)*np.ones(freq_offset.shape)

    return freq_offset

def comp_freq_offset(sig, freq_offset, os=1 ):
    """
    Compensate for frequency offset in signal

    Parameters
    ----------
        sig : array_line
            signal array with N modes
        freq_offset: array_like
            frequency offset to compensate for if 1D apply to all modes
        os: int
            oversampling ratio (Samples per symbols in sig)


    Returns
    -------
        comp_signal : array with N modes
            input signal with removed frequency offset

    """
    # Fix number of stuff
    ndim = sig.ndim
    sig = np.atleast_2d(sig)
    #freq_offset = np.atleast_2d(freq_offset)
    npols = sig.shape[0]

    # Output Vector
    comp_signal = np.zeros([npols, np.shape(sig)[1]], dtype=sig.dtype)

    # Fix output
    sig_len = len(sig[0,:])
    time_vec = np.arange(1,sig_len + 1,dtype = float)
    for l in range(npols):
        lin_phase = 2 * np.pi * time_vec * freq_offset[l] /  os
        comp_signal[l,:] = sig[l,:] * np.exp(-1j * lin_phase)
    if ndim == 1:
        return comp_signal.flatten()
    else:
        return comp_signal
