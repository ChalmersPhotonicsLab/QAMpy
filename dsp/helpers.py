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
        for i in range(E.shape[0]):
            E[i] -= np.mean(E[i])
            P = np.sqrt(np.mean(cabssquared(E[i])))
            E[i] /= P
    else:
        E = E.real - np.mean(E.real) + 1.j * (E.imag-np.mean(E.imag))
        P = np.sqrt(np.mean(cabssquared(E)))
        E /= P
    return E


def dump_edges(E, N):
    """
    Remove N samples from the front and end of the input field.
    """
    return E[N:-N]