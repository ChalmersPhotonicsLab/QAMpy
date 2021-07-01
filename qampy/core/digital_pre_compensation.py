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
# Copyright 2018 Jochen Schröder, Zonglong He

"""
Digital pre-distortion and compensation of transceiver impairments
"""

import numpy as np
from qampy.core.special_fcts import rrcos_freq
from qampy.core.utils import rolling_window
from qampy.core import pythran_dsp
from scipy import signal,interpolate


def clipper(sig, clipping_level):
    """
    Clip signal to the range (-clipping_level, clipping_level).
    """
    sig_2d = np.atleast_2d(sig)
    sig_clip_re = np.sign(sig.real) * np.minimum(abs(sig_2d.real), clipping_level*np.ones((1, sig_2d.shape[1])))
    sig_clip_im = np.sign(sig.imag) * np.minimum(abs(sig_2d.imag), clipping_level*np.ones((1, sig_2d.shape[1])))

    return sig_clip_re + 1j* sig_clip_im

def comp_mod_sin(sig, vpi=1.14):
    """
    Use arcsin() function to compensate modulator nonlinear sin() response.

    Parameters
    ----------
    sig: array_like
        Complex input signal should be in range (-1,1)
    vpi : complex or float, optional
        Vpi of the modulator if a float both Vpi of real and imaginary part are assumed to be the same
    """
    if not np.iscomplexobj(vpi):
        vpi = vpi + 1j*vpi
    sig_out_re = 2 * vpi.real * np.arcsin(sig.real)
    sig__out_im = 2 * vpi.imag * np.arcsin(sig.imag)
    sig_out = sig_out_re + 1j*sig__out_im
    return sig_out

def comp_dac_resp(dpe_fb, sim_len, rrc_beta, PAPR=9, prms_dac=(16e9, 2, 'sos', 6), os=2):
    """
    Compensate frequency response of a simulated digital-to-analog converter(DAC).

    Parameters
    ----------

    dpe_fb : int
        symbol rate to compensate for
    sim_len : int
        length of the oversampled signal array
    rrc_beta : float
        root-raised cosine roll-off factor of the simulated signal
    PAPR: int (optional)
        peak to average power ratio of the signal
    prms_dac: tuple(float, int, str, int)
        DAC filer parameters for calculating the filter response using scipy.signal
    os: int (optional)
        oversampling factor of the signal
    """
    dpe_fs = dpe_fb * os
    # Derive RRC filter frequency response np.sqrt(n_f)
    T_rrc = 1/dpe_fb
    fre_rrc = np.fft.fftfreq(sim_len) * dpe_fs
    rrc_f = rrcos_freq(fre_rrc, rrc_beta, T_rrc)
    rrc_f /= rrc_f.max()
    n_f = rrc_f ** 2

    # Derive bessel filter (DAC) frequency response d_f
    cutoff, order, frmt, enob = prms_dac
    system_dig = signal.bessel(order, cutoff, 'low', analog=False, output=frmt, norm='mag', fs=dpe_fs)
    # w_bes=np.linspace(0,fs-fs/worN,worN)
    w_bes, d_f = signal.sosfreqz(system_dig, worN=sim_len, whole=True, fs=dpe_fs)

    # Calculate dpe filter p_f
    df = dpe_fs/sim_len
    alpha = 10 ** (PAPR / 10) / (6 * dpe_fb * 2 ** (2 * enob)) * np.sum(abs(d_f) ** 2 * n_f * df)
    p_f = n_f * np.conj(d_f) / (n_f * abs(d_f) ** 2 + alpha)
    return p_f

def find_sym_patterns(sig, ref_sym, N, ret_ptrns=False):
    """
    Find and index patterns elements of length N.
    
    Parameters
    ----------
    sig : array_like
        Array where to look for patterns
    ref_sym : array_like
        Reference elements/symbols which we look for
    N  : int
        pattern length
    ret_ptrns : bool, optional
        Also return the patterns of elements

    Returns
    -------
    pattern_idx : array_like
        index array of patterns in the signal
    if ret_ptrns:
        sym_ptrns : array_like
            Array of possible patterns 
    """
    M = ref_sym.size
    L = int(M**N)
    idx = np.arange(L).reshape(N*[M])
    sig_idx = np.argmin(abs(sig.reshape(1,-1)-ref_sym.reshape(-1,1)), axis=0)
    sig_rwin = rolling_window(sig_idx, N, wrap=True)
    pattern_idx = idx[tuple(sig_rwin.T)]
    if ret_ptrns:
        pidx = np.array(np.unravel_index(np.arange(L), N*[M])).T
        return pattern_idx, ref_sym[pidx]
    else:
        return idx[tuple(sig_rwin.T)]

def cal_lut(tx_sig, rx_sig, ref_sym, mem_len=3, idx_data=None, real_ptrns=True):
    """
    Calculate a lookup table for precompensation of pattern based errors. This can 
    be considered a simplified Volterra filter. The function works on 1D signals only, if 
    more dimenions (modes) are desired one has to loop.
    
    Parameters
    ----------
    tx_sig : array_like
        The transmitted signal, needs to be 1D
    rx_sig : array_like
        The received signal, needs to be 1D
    ref_sym : array_like
        The symbol alphabet as complex numbers
    mem_len : int, optional
        Length of the pattern compensation
    idx_data : array_like, optional
        index array if calculation on only a subset of symbols is desired. (Default is operate on full array). 
        IMPORTANT: if operating on the full array, we will use the symbols from the end of the array for the beginning 
        pattern. So like the first mem_len//2 symbols should be skipped
    real_ptrns : bool, optional 
        whether to operate on the complex patterns or on the in-phase and quadrature components. Operating on the
        complex patterns increases LUT size significantly (default: operate on in-phase and quadrature separately)

    Returns
    -------
    ea : array_like
        average error LUT. 
    idx_I: array_like
        index map of symbol to pattern in LUT, in-phase component
    idx_Q : array_like
        index map of symbol to pattern in LUT, quadrature component, if real_ptrns is False, this will be the same as idx_I
    """
    assert tx_sig.ndim == 1 and rx_sig.ndim == 1, "Ony 1d signals are supported, loop if you need more dimensions"
    assert tx_sig.shape == rx_sig.shape, "Tx and Rx signal need to have the same shape"
    if idx_data is None:
        idx_data = np.ones(tx_sig.shape[-1], dtype=np.bool)
    err = (tx_sig - rx_sig).flatten()
    idx = np.nonzero(idx_data)[0] - mem_len//2
    if real_ptrns:
        ref_sym_I = np.unique(ref_sym.real)
        ref_sym_Q = np.unique(ref_sym.imag)
        M = ref_sym_I.size
        N = int(M**mem_len)
        idx_I = find_sym_patterns(tx_sig.real, ref_sym_I, mem_len)
        idx_Q = find_sym_patterns(tx_sig.imag, ref_sym_Q, mem_len)
        idx_I = np.copy(idx_I[idx]) # copy to avoid pythran errors
        idx_Q = np.copy(idx_Q[idx])
        ea = pythran_dsp.cal_lut_avg(np.copy(err[idx_data]), idx_I, idx_Q, N)
        return ea, idx_I, idx_Q
    else:
        ref_sym_c = np.unique(ref_sym)
        M = ref_sym.size
        N = int(M**mem_len)
        idx_c = find_sym_patterns(tx_sig, ref_sym_c, mem_len)
        L =np.count_nonzero(idx_data)
        idx_c = np.copy(idx_c[idx])
        ea = pythran_dsp.cal_lut_avg(np.copy(err[idx_data]), idx_c, idx_c, N)
        return ea, idx_c, idx_c
