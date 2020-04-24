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
# Copyright 2018 Jochen Schröder, Zonglong

import numpy as np
from qampy.core.special_fcts import rrcos_freq
from scipy import signal,interpolate

def clipper(sig, clipping_level):
    """
    Clip signal to the range (-clipping_level, clipping_level).
    """
    sig_2d = np.atleast_2d(sig)
    sig_clip_re = np.sign(sig.real) * np.minimum(abs(sig_2d.real), clipping_level*np.ones((1, sig_2d.shape[1])))
    sig_clip_im = np.sign(sig.imag) * np.minimum(abs(sig_2d.imag), clipping_level*np.ones((1, sig_2d.shape[1])))

    return sig_clip_re + 1j* sig_clip_im

def comp_mod_sin(sig, vpi=3.5):
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
    sig_out_re = 2 * vpi.real / np.pi * np.arcsin(sig.real)
    sig__out_im = 2 * vpi.imag / np.pi * np.arcsin(sig.imag)
    sig_out = sig_out_re + 1j*sig__out_im
    return sig_out

def comp_dac_resp(dpe_fb, sim_len, rrc_beta, PAPR=9, prms_dac=(16e9, 2, 'sos', 6)):
    """
    Compensate frequency response of simulated digital-to-analog converter(DAC).
    """
    dpe_fs = dpe_fb * 2
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

