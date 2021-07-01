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

from . import core
from qampy.core.analog_frontend import comp_IQ_inbalance
from qampy.core.analog_frontend import __doc__


def comp_rf_delay(signal, delay):
    """
    Adds a delay to the signal in frequency domain. Can be
    used to compensate for impairments such as RF cables of different length
    between the optical hybrid and ADC.

    Parameters
    ----------
        signal : signalobject
            Real-valued input signal
        delay : float
            Delay  in s

    Returns
    -------
        sig_out : signalobject
            Signal after compensating for delay
    """
    comp = core.analog_frontend.comp_rf_delay(signal, delay, signal.fs)
    return signal.recreate_from_np_array(comp)

def orthonormalize_signal(signal):
    """
    Orthogonalizing signal using the Gram-Schmidt process _[1].

    Parameters
    ----------
    E : signalobject
       input signal

    Returns
    -------
    E_out : signalobject
        orthonormalized signal

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process for more
       detailed description.
    """
    return core.analog_frontend.orthonormalize_signal(signal, signal.os)


