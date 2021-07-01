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
Convenience module for calculation of special mathematical functions useful for
communications.
"""

from __future__ import division, print_function

import numpy as np
from scipy.special import erfc


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
    return A * np.tanh((x - x0) / w)


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
    with np.errstate(divide='ignore', invalid='ignore'):
        rrcos = 1/T*((np.sin(np.pi*t/T*(1-beta)) +  4*beta*t/T*np.cos(np.pi*t/T*(1+beta)))/(np.pi*t/T*(1-(4*beta*t/T)**2)))
        eps = abs(t[0]-t[1])/4
        idx1 = np.where(abs(t)<eps)
        rrcos[idx1] = 1/T*(1+beta*(4/np.pi-1))
        idx2 = np.where(abs(abs(t)-abs(T/(4*beta)))<eps)
        rrcos[idx2] = beta/(T*np.sqrt(2))*((1+2/np.pi)*np.sin(np.pi/(4*beta))+(1-2/np.pi)*np.cos(np.pi/(4*beta)))
    return rrcos


def q_function(x):
    """The Q function is the tail probability of the standard normal distribution see _[1,2] for a definition and its relation to the erfc. In _[3] it is called the Gaussian co-error function.

    References
    ----------
    ...[1] https://en.wikipedia.org/wiki/Q-function
    ...[2] https://en.wikipedia.org/wiki/Error_function#Integral_of_error_function_with_Gaussian_density_function
    ...[3] Shafik, R. (2006). On the extended relationships among EVM, BER and SNR as performance metrics. In Conference on Electrical and Computer Engineering (p. 408). Retrieved from http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=4178493
    """
    return 0.5*erfc(x/np.sqrt(2))