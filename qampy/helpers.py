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
Convenient helper functions
"""
from __future__ import division, print_function
import numpy as np


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


def normalise_and_center(E):
    """
    Normalise and center the input field, by calculating the mean power for each polarisation separate and dividing by its square-root
    """
    if E.ndim > 1:
        E = E - np.mean(E, axis=-1)[:, np.newaxis]
        P = np.sqrt(np.mean(cabssquared(E), axis=-1))
        E /= P[:, np.newaxis]
    else:
        E = E.real - np.mean(E.real) + 1.j * (E.imag-np.mean(E.imag))
        P = np.sqrt(np.mean(cabssquared(E)))
        E /= P
    return E

def normalise_and_center_pil(sig, idx_pil):
    """
    Normalise and center the input field only based on pilot symbols, by calculating the mean power for each polarisation separate and dividing by its square-root
    """
    sig_pil = sig[:, idx_pil]
    if sig.ndim > 1:
        ct_fac = - np.mean(sig_pil, axis=-1)[:, np.newaxis]
        sig_pil = sig_pil + ct_fac
        pil_p = np.sqrt(np.mean(abs(sig_pil)**2, axis=-1))
        sig_out = (sig + ct_fac) / pil_p[:, np.newaxis]
    else:
        ct_fac = -(np.mean(sig_pil.real) + 1.j * np.mean(sig_pil.imag))
        sig_pil = sig_pil + ct_fac
        pil_p = np.sqrt(np.mean(abs(sig_pil)**2))
        sig_out = (sig + ct_fac) / pil_p
    return sig_out


def dump_edges(E, N):
    """
    Remove N samples from the front and end of the input field.
    """
    if E.ndim > 1:
        return E[:,N:-N]
    else:
        return E[N:-N]

def set_mid_point(E, mid_pos=0):
    """
    Move the (1-pol) signal's mid-position to given value
    """
    if np.iscomplexobj(E):
        ori_mid_pos = (E.real.max() + E.real.min())/2 + 1j*(E.imag.max() + E.imag.min())/2
        return E - ori_mid_pos + mid_pos
    if not np.iscomplexobj(E):
        ori_mid_pos = (E.max() + E.min())/2
        return E - ori_mid_pos + mid_pos


def rescale_signal(E, swing=1):
    """
    Rescale the (1-pol) signal to (-swing, swing).
    """
    swing = np.atleast_1d(swing)
    if np.iscomplexobj(E):
        scale_factor = np.maximum(np.max(abs(E.real), axis=-1), np.max(abs(E.imag), axis=-1))
        return E / scale_factor[:, np.newaxis] * swing[:,np.newaxis]
    else:
        scale_factor = np.max(abs(E), axis=-1)
        return E / scale_factor[:, np.newaxis] * swing[:,np.newaxis]

def set_mid_and_resale(E,mid_pos=0,swing=1):
    """
    Change (1-pol) signal mid-position to given value and rescale the real signal to (-swing, swing).
    """
    sig_out = set_mid_point(E, mid_pos)
    sig_out = rescale_signal(sig_out, swing)

    return sig_out

def get_center_shift_fac(E):
    """
    Obtain shift factor (x_shift, y_shift) that is used to center the signal.
    """
    if E.ndim > 1:
        shift_fac = - np.mean(E, axis=-1)[:, np.newaxis]
    else:
        shift_fac = -(np.mean(E.real) + 1.j * np.mean(E.imag))
    return shift_fac

def find_pilot_idx(nframe=2, frame_len = 2 ** 16, os_rate=2, pilot_seq_len=1024, pilot_ins_rat=32):
    """
    find pilot index for object with both pilot sequence (at the beginning) and phase pilot.
    """
    idx_os = np.arange(frame_len * nframe * os_rate)
    idx_pil_seq = idx_os < 0  # generate array with All-False element
    idx_pil_ph = idx_os < 0
    for i in range(nframe):
        idx_temp = frame_len * os_rate * i
        idx_pil_seq = idx_pil_seq | ((idx_temp <= idx_os) & (idx_os < idx_temp + pilot_seq_len * os_rate))
        idx_pil_ph = idx_pil_ph | ((((idx_os - pilot_seq_len * os_rate - idx_temp) % (pilot_ins_rat * os_rate) == 0) | (
                    (idx_os - pilot_seq_len * os_rate - idx_temp) % (pilot_ins_rat * os_rate) == 1)) & (
                                               idx_os - pilot_seq_len * os_rate - idx_temp >= 0) & (
                                               idx_os < idx_temp + frame_len * os_rate))
    idx_pil = idx_pil_seq | idx_pil_ph
    idx_data = ~idx_pil
    return idx_pil
