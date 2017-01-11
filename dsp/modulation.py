from __future__ import division
import numpy as np
from bitarray import bitarray
from .utils import bin2gray, cabssquared, convert_iqtosinglebitstream, resample
from .prbs import make_prbs_extXOR
from . import theory
from . import ber_functions


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
    if np.log2(M) % 2 > 0.5:
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
    N = (np.log2(M) - 1) / 2
    s = 2**(N - 1)
    rect = np.mgrid[-(2**(N + 1) - 1):2**(N + 1) - 1:1.j * 2**(N + 1), -(
        2**N - 1):2**N - 1:1.j * 2**N]
    qam = rect[0] + 1.j * rect[1]
    idx1 = np.where((abs(qam.real) > 3 * s) & (abs(qam.imag) > s))
    idx2 = np.where((abs(qam.real) > 3 * s) & (abs(qam.imag) <= s))
    qam[idx1] = np.sign(qam[idx1].real) * (
        abs(qam[idx1].real) - 2 * s) + 1.j * (np.sign(qam[idx1].imag) *
                                              (4 * s - abs(qam[idx1].imag)))
    qam[idx2] = np.sign(qam[idx2].real) * (
        4 * s - abs(qam[idx2].real)) + 1.j * (np.sign(qam[idx2].imag) *
                                              (abs(qam[idx2].imag) + 2 * s))
    return qam.flatten()


def gray_code_for_qam(M):
    """
    Generate gray code map for M-QAM constellations
    """
    Nbits = int(np.log2(M))
    if Nbits % 2 == 0:
        N = Nbits // 2
        idx = np.mgrid[0:2**N:1, 0:2**N:1]
    else:
        N = (Nbits - 1) // 2
        idx = np.mgrid[0:2**(N + 1):1, 0:2**N:1]
    gidx = bin2gray(idx)
    return ((gidx[0] << N) | gidx[1]).flatten()


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
        if not self.bits % 2:
            self._scale = theory.MQAMScalingFactor(self.M)
        else:
            self._scale = (abs(self.symbols)**2).mean()
        self.symbols /= np.sqrt(self._scale)
        self.coding = None
        self._graycode = gray_code_for_qam(M)
        self.gray_coded_symbols = self.symbols[self._graycode]
        bformat = "0%db" % self.bits
        self._encoding = dict([(self.symbols[i].tobytes(),
                                bitarray(format(self._graycode[i], bformat)))
                               for i in range(len(self._graycode))])

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
        rem = len(data) % self.bits
        if rem > 0:
            data = data[:-rem]
        datab = bitarray()
        datab.pack(data.tobytes())
        # the below is not really the fastest method but easy encoding/decoding is possible
        return np.fromstring(
            b''.join(datab.decode(self._encoding)),
            dtype=np.complex128) 

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
        return np.fromstring(bt.unpack(zero=b'\x00', one=b'\x01'), dtype=np.bool)

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
        return quantize(signal, self.symbols)

    def generateSignal(self,
                       N,
                       snr,
                       carrier_df=0,
                       baudrate=1,
                       samplingrate=1,
                       IQsep=False,
                       PRBS=True,
                       PRBSorder=15,
                       PRBSseed=None):
        """
        Generate a M-QAM data signal array

        Parameters:
        ----------
        N :  int
            number of symbols to be generated.
        snr: number
            Signal to Noise Ratio (Es/N0) in logarithmic units
        carrier_df: number, optional
            carrier frequency offset, relative to the overall window, if not given it
            is set to 0 (baseband modulation)
        baudrate:  number, optional
            symbol rate of the signal. This should be the real symbol rate, used
            for calculating the oversampling factor. If not given it is 1.
        samplingrate: number, optional
            the rate at which the signal is sampled. Together with the baudrate
            this is used for calculating the oversampling factor. Note that if samplingrate is
            different to baudrate the length of the returned array is not N. (Default is 1.)
        IQsep  : bool, optional
            Whether to generate two independent data streams for I and Q (Default is False)
        PRBS: bool, optional
            If True the bits are generated as standard PRBS sequences, if False
            random bits are generated using numpy.random.randint.
            (Default is True)
        PRBSOrder: int or tuple of bool, optional
            The PRBS order i.e. the length 2**order of the PRBS to use
            If IQsep is True this needs to be a tuple of two ints for the I and Q sequence generation.
            (Default is 15)
        PRBSseed: array_like, optional
            Seed to the PRBS generator needs to be a 1D array of booleans of length order.
          . (Default=None, which corresponds to a seed of all 1's)
        """
        if IQsep:
            if np.isscalar(PRBSorder):
                if np.isscalar(PRBSseed) or PRBSseed is None:
                    bitsq = twostreamPRBS(N, self.bits, PRBS)
                else:
                    bitsq = twostreamPRBS(N, self.bits, PRBS, PRBSseed=PRBSseed)
            else:
                bitsq = twostreamPRBS(N, self.bits, PRBS, PRBSseed=PRBSseed, PRBSorder=PRBSorder)
        else:
            Nbits = N * self.bits
            if PRBS == True:
                bitsq = make_prbs_extXOR(PRBSorder, Nbits, PRBSseed)
            else:
                bitsq = np.random.randint(0, high=2, size=Nbits).astype(np.bool)
        symbols = self.modulate(bitsq)
        noise = (np.random.randn(N) + 1.j * np.random.randn(N)) / np.sqrt(
            2)  # sqrt(2) because N/2 = sigma
        outdata = symbols + noise * 10**(-snr / 20)  #the 20 here is so we don't have to take the sqrt
        outdata = resample(baudrate, samplingrate, outdata)
        return outdata * np.exp(2.j * np.pi * np.arange(len(outdata)) *
                                carrier_df / samplingrate), symbols, bitsq

    def theoretical_SER(self, snr):
        """
        Return the theoretical SER for this modulation format for the given SNR (in linear units)
        """
        return theory.MQAM_SERvsEsN0(snr, self.M)

    def calculate_SER(self, signal_rx, bits_tx=None, symbol_tx=None):
        """
        Calculate the symbol error rate of the signal. This function does not do any synchronization and assumes that signal and transmitted data start at the same symbol. 

        Parameters
        ----------
        signal_rx  : array_like
            Received signal (1D complex array)
        bits_tx    : array_like, optional
            bitstream at the transmitter for comparison against signal. If set to None symbol_tx needs to be given (Default is None)
        symbol_tx  : array_like, optional
            symbols at the transmitter for comparison against signal. If set to None bits_tx needs to be given (Default is None)

        Returns
        -------
        SER   : float
            symbol error rate
        """
        assert symbol_tx is not None or bits_tx is not None, "data_tx or symbol_tx must be given"
        if symbol_tx is None:
            symbol_tx = self.modulate(bits_tx)
        data_demod = self.quantize(signal_rx)[0]
        return np.count_nonzero(data_demod - symbol_tx)/len(signal_rx)

    def cal_BER(self, signal_rx, bits_tx=None, PRBS=None, Lsync=100, imax=5):
        assert bits_tx is not None or PRBS is not None, "either bits_tx or PRBS needs to be given"
        bits_demod = self.decode(self.quantize(signal_rx)[0])
        if bits_tx is None:
            if len(PRBS)>1:
                bits_tx = make_prbs_extXOR(len(bits_demod), PRBS[0], PRBS[1])
            else:
                bits_tx = make_prbs_extXOR(len(bits_demod), PRBS[0])
        else:
            bits_tx = ber_functions.adjust_data_length(bits_tx, bits_demod)
        i = 0
        while i < 5:
            if i == 4:
                raise ber_functions.DataSyncError("could not sync signal to sequence")
            try:
                idx, tx_synced = ber_functions.sync_Tx2Rx(bits_tx, bits_demod, Lsync, imax)
                break
            except:
                signal_rx *= 1.j  # rotate by 90 degrees
                bits_demod = self.decode(self.quantize(signal_rx)[0])
                bits_demod = ber_functions.adjust_data_length(bits_demod, bits_tx)
            i += 1
        return ber_functions._cal_BER_only(tx_synced, bits_demod)



def twostreamPRBS(Nsyms, bits, PRBS=True, PRBSorder=(15, 23), PRBSseed=(None,None)):
    """
    Generate a PRBS from two independent PRBS generators.

    Parameters
    ----------
    Nsyms   : int
        the number of symbols at the output
    bits    : ints
        the bits per symbol
    PRBS    : bool, optional
        whether to use PRBS signal generation or np.random.randint (Default True use PRBS)
    PRBSorder : tuple(int, int), optional
        The PRBS order for each generated sequence (Default is (15,23))
    PRBSseed  : tuple(int, int), optional
        The seed for each PRBS

    Returns
    -------
    bitseq  : array_like
        1D array of booleans that are the interleaved PRBS sequence
    """
    if bits % 2:
        Nbits = (Nsyms * (bits // 2 + 1), Nsyms * bits // 2)
    else:
        Nbits = (Nsyms * bits // 2, Nsyms * bits // 2)
    if PRBS:
        bitsq1 = make_prbs_extXOR(PRBSorder[0], Nbits[0], PRBSseed[0])
        bitsq2 = make_prbs_extXOR(PRBSorder[0], Nbits[1], PRBSseed[1])
    else:
        bitsq1 = np.random.randint(0, high=2, size=Nbits[0]).astype(np.bool)
        bitsq2 = np.random.randint(0, high=2, size=Nbits[1]).astype(np.bool)
    return convert_iqtosinglebitstream(bitsq1, bitsq2, bits)
