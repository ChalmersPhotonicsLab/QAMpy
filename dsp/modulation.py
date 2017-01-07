from __future__ import division
import numpy as np
from bitarray import bitarray
from .utils import bin2gray, cabssquared
from . import theory




def quantize(sig, symbols):
    """
    Quantize signal to symbols, based on closest distance.

    Parameters
    ----------
    sig     : array_like
        input signal field, 1D array of complex values
    symbols : array_like
        symbol alphabet to quantize to (1D array, dtype=complex)

    Returns:
    sigsyms : array_like
        array of detected symbols
    idx     : array_like
        array of indices to the symbols in the symbol array
    """
    P = np.mean(cabssquared(sig))
    sig /= np.sqrt(P)
    idx = abs(sig[:, np.newaxis] - symbols).argmin(axis=1)
    sigsyms = symbols[idx]
    return sigsyms, idx


def calculate_MQAM_symbols(M):
    """
    Generate the symbols on the constellation diagram for M-QAM
    """
    if np.log2(M)%2 > 0.5:
        return calculate_cross_QAM_symbols(M)
    else:
        return calculate_square_QAM_symbols(M)

def calculate_square_QAM_symbols(M):
    """
    Generate the symbols on the constellation diagram for square M-QAM
    """
    qam = np.mgrid[-(2 * np.sqrt(M) / 2 - 1):2 * np.sqrt(
        M) / 2 - 1:1.j * np.sqrt(M), -(2 * np.sqrt(M) / 2 - 1):2 * np.sqrt(M) /
                   2 - 1:1.j * np.sqrt(M)]
    return (qam[0] + 1.j * qam[1]).flatten()

def calculate_cross_QAM_symbols(M):
    """
    Generate the symbols on the constellation diagram for non-square (cross) M-QAM
    """
    N = (np.log2(M)-1)/2
    s = 2**(N-1)
    rect = np.mgrid[
        -(2**(N+1) - 1) : 2**(N+1) - 1: 1.j*2**(N+1),
        -(2**N - 1) : 2**N - 1: 1.j*2**N
    ]
    qam = rect[0] + 1.j*rect[1]
    idx1 = np.where((abs(qam.real) > 3*s)&(abs(qam.imag) > s))
    idx2 = np.where((abs(qam.real) > 3*s)&(abs(qam.imag) <= s))
    qam[idx1] = np.sign(qam[idx1].real)*(abs(qam[idx1].real)-2*s) + 1.j*(np.sign(qam[idx1].imag)*(4*s-abs(qam[idx1].imag)))
    qam[idx2] = np.sign(qam[idx2].real)*(4*s-abs(qam[idx2].real)) + 1.j*(np.sign(qam[idx2].imag)*(abs(qam[idx2].imag)+2*s))
    return qam.flatten()

def gray_code_for_qam(M):
    """
    Generate gray code map for M-QAM constellations
    """
    Nbits = int(np.log2(M))
    if Nbits%2 == 0:
        N = Nbits//2
        idx = np.mgrid[
            0 : 2**N : 1,
            0 : 2**N : 1
        ]
    else:
        N = (Nbits - 1)//2
        idx = np.mgrid[
            0 : 2**(N+1): 1,
            0 : 2**N: 1
        ]
    gidx = bin2gray(idx)
    return ((gidx[0] << N)| gidx[1] ).flatten()


class QAMModulator(object):
    """
    A modulator object for modulating and demodulating rectangular QAM signals (without carrier frequency and equalisation). Currently only Gray coding of bits is supported.
    """

    def __init__(self, M, coding=None):
        """
        Initialise QAM modulator

        Parameters
        ----------
        M        :  int
            number of constellation points, indicates QAM order

        coding   : string, optional
           coding method currently only the default gray coding is supported

        """
        self.M = M
        self.symbols = calculate_MQAM_symbols(M)
        if not self.bits%2:
            self._scale = theory.MQAMScalingFactor(self.M)
        else:
            self._scale = (abs(self.symbols)**2).mean()
        self.coding = None
        self._graycode = gray_code_for_qam(M)
        self.gray_coded_symbols = self.symbols[self._graycode]
        bformat = "0%db"%self.bits
        self._encoding = dict([(self.symbols[i].tobytes(), bitarray(format(self._graycode[i], bformat))) for i in range(len(self._graycode))])

    @property
    def bits(self):
        """
        Number of bits per symbol
        """
        return int(np.log2(self.M))

    def modulate(self, data):
        """
        Modulate a bit sequence into QAM symbols

        Parameters
        ----------
        data     : array_like
           1D array of bits represented as bools. If the len(data)%self.M != 0 then we only encode up to the nearest divisor

        Returns
        -------
        outdata  : array_like
            1D array of complex symbol values. Normalised to energy of 1
        """
        rem  = len(data)%self.bits
        if rem > 0:
            data = data[:-rem]
        datab = bitarray()
        datab.pack(data.tobytes())
        # the below is not really the fastest method but easy encoding/decoding is possible
        return np.fromstring(b''.join(datab.decode(self._encoding)), dtype=np.complex128)/np.sqrt(self._scale)

    def decode(self, symbols):
        """
        Decode array of input symbols to bits according to the coding of the modulator.

        Parameters
        ----------
        symbols   : array_like
            1D array of complex input symbols

        Returns
        -------
        outbits   : array_like
            1D array of booleans representing bits
        """
        bt = bitarray()
        bt.encode(self._encoding, symbols.view(dtype="S16"))
        return np.fromstring(bt.unpack(), dtype=np.bool)

    def quantize(self, signal):
        """
        Make symbol decisions based on the input field. Decision is made based on difference from constellation points

        Parameters
        ----------
        signal   : array_like
            1D array of the input signal

        Returns
        -------
        symbols  : array_like
            1d array of the detected symbols
        idx      : array_like
            1D array of indices into QAMmodulator.symbols
        """
        return quantize(signal, self.symbols/np.sqrt(self._scale))
