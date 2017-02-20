from __future__ import division
import numpy as np
import arrayfire as af
from bitarray import bitarray
from .utils import bin2gray, cabssquared, convert_iqtosinglebitstream, resample, normalise_and_center, bool2bin
from .prbs import make_prbs_extXOR
#from .equalisation import quantize
from .theory import MQAMScalingFactor, calculate_MQAM_symbols, calculate_MQAM_scaling_factor, gray_code_for_qam
from . import theory
from . import ber_functions
from .phaserecovery import NMAX


def quantize(signal, symbols):
    global  NMAX
    Nmax = NMAX//len(symbols.flatten())//16
    L = signal.flatten().shape[0]
    sig = af.np_to_af_array(signal.flatten())
    sym = af.transpose(af.np_to_af_array(symbols.flatten()))
    tmp = af.constant(0, L, dtype=af.Dtype.c64)
    if L < Nmax:
        v, idx = af.imin(af.abs(af.broadcast(lambda x,y: x-y, sig,sym)), dim=1)
        tmp = af.transpose(sym)[idx]
    else:
        steps = L//Nmax
        rem = L%Nmax
        for i in range(steps):
            v, idx = af.imin(af.abs(af.broadcast(lambda x,y: x-y, sig[i*Nmax:(i+1)*Nmax],sym)), dim=1)
            tmp[i*Nmax:(i+1)*Nmax] = af.transpose(sym)[idx]
        v, idx = af.imin(af.abs(af.broadcast(lambda x,y: x-y, sig[steps*Nmax:],sym)), dim=1)
        tmp[steps*Nmax:] = af.transpose(sym)[idx]
    return np.array(tmp)


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
        self._scale = calculate_MQAM_scaling_factor(M)
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
        signal = normalise_and_center(signal)
        return quantize(normalise_and_center(signal), self.symbols)

    def cal_EVM(self, signal, syms=None):
        """
        Calculate the Error Vector Magnitude of the input signal either blindly or against a known symbol sequence, after _[1]. The EVM here is normalised to the average symbol power, not the peak as in some other definitions.

        Parameters
        ----------

        signal    : array_like
            input signal to measure the EVM offset

        syms      : array_like, optional
            known symbol sequence. If this is None, the signal is quantized into its symbols and the EVM is calculated blindly. For low SNRs this will underestimate the real EVM, because detection errors are not counted.

        Returns
        -------

        evm       : array_like
            RMS EVM

        References
        ----------
        ...[1] Shafik, R. (2006). On the extended relationships among EVM, BER and SNR as performance metrics. In Conference on Electrical and Computer Engineering (p. 408). Retrieved from http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=4178493


        Note
        ----

        The RMS EVM differs from the EVM in dB by a square factor, see the different definitions e.g. on wikipedia.
        """
        if syms == None:
            syms = self.quantize(signal)
        return np.sqrt(np.mean(cabssquared(syms-signal)))#/np.mean(abs(self.symbols)**2))

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
        noise = (np.random.randn(N) + 1.j * np.random.randn(N)) / np.sqrt(2)  # sqrt(2) because N/2 = sigma
        outdata = symbols + noise * 10**(-snr / 20)  #the 20 here is so we don't have to take the sqrt
        outdata = resample(baudrate, samplingrate, outdata)
        return outdata * np.exp(2.j * np.pi * np.arange(len(outdata)) *
                                carrier_df / samplingrate), symbols, bitsq

    def theoretical_SER(self, snr):
        """
        Return the theoretical SER for this modulation format for the given SNR (in linear units)
        """
        return theory.MQAM_SERvsEsN0(snr, self.M)

    def calculate_SER(self, signal_rx, symbol_tx=None, bits_tx=None, synced=False, N1=2**15, N2=8000):
        """
        Calculate the symbol error rate of the signal. This function does not do any synchronization and assumes that signal and transmitted data start at the same symbol. 

        Parameters
        ----------
        signal_rx  : array_like
            Received signal (1D complex array)

        symbol_tx  : array_like, optional
            symbols at the transmitter for comparison against signal. If set to None bits_tx needs to be given (Default is None)
        bits_tx    : array_like, optional
            bitstream at the transmitter for comparison against signal. If set to None symbol_tx needs to be given (Default is None)

        synced    : bool, optional
            whether signal_tx and symbol_tx are synchronised.

        N1        : integer, optional
            length of the rx signal to use for the crosscorrelation sync. A good value is the length of PRBS order of the transmitted signal

        N2         : integer, optional
            subsequence to use for searching the offset. This should not be too small otherwise there will be high BERs, 1/6 of the PRBS length seems to work quite well. 


        Returns
        -------
        SER   : float
            symbol error rate
        """
        assert symbol_tx is not None or bits_tx is not None, "data_tx or symbol_tx must be given"
        if symbol_tx is None:
            symbol_tx = self.modulate(bits_tx)
        data_demod = self.quantize(signal_rx)
        if not synced:
            symbol_tx = self._sync_symbol2signal(symbol_tx, data_demod, N1, N2)
        return np.count_nonzero(data_demod - symbol_tx)/len(signal_rx)

    def _sync_symbol2signal(self, syms_tx, syms_demod, N1, N2):
        acm = 0.
        for i in range(4):
            syms_tx = syms_tx*1.j**i
            s_sync, idx, ac = ber_functions.sync_Tx2Rx_Xcorr(syms_tx, syms_demod, N1, N2)
            act = abs(ac).max()
            if act > acm:
                s_tx_sync = s_sync
                ix = idx
                acm = act
        return s_tx_sync

    def cal_BER(self, signal_rx, bits_tx=None, syms_tx=None, PRBS=(15,bool2bin(np.ones(15))), N1=2**15, N2=8000):
        """
        Calculate the bit-error-rate for the given signal, against either a PRBS sequence or a given bit sequence.

        Parameters
        ----------

        signal_rx    : array_like
            received signal to demodulate and calculate BER of

        bits_tx      : array_like, optional
            transmitted bit sequence to compare against. (default is None, which either PRBS or syms_tx has to be given)

        syms_tx      : array_like, optional
            transmitted bit sequence to compare against. (default is None, which means bits_tx or PRBS has to be given)

        PRBS         : tuple(int, int), optional
            tuple of PRBS order and seed, the order has to be integer 7, 15, 23, 31 and the seed has to be None or a binary array of length of the PRBS order. If the seed is None it will be initialised to all bits one.

        N1        : integer, optional
            length of the rx signal to use for the crosscorrelation sync. A good value is the length of PRBS order of the transmitted signal

        N2         : integer, optional
            subsequence to use for searching the offset. This should not be too small otherwise there will be high BERs, 1/6 of the PRBS length seems to work quite well. 

        Returns
        -------

        ber          : float
            bit-error-rate in linear units

        errs         : integer
            number of detected errors

        N            : integer
            length of input sequence
        """
        assert bits_tx is not None or PRBS is not None, "either bits_tx or PRBS needs to be given"
        syms_demod = self.quantize(signal_rx)
        if syms_tx is None:
            if bits_tx is None:
                bits_tx = make_prbs_extXOR( PRBS[0], len(syms_demod)*self.bits, seed=PRBS[1])
            syms_tx = self.modulate(bits_tx)
            syms_tx = ber_functions.adjust_data_length(syms_tx, syms_demod)
        s_tx_sync = self._sync_symbol2signal(syms_tx, syms_demod, N1, N2)
        bits_demod = self.decode(syms_demod)
        tx_synced = self.decode(s_tx_sync)
        return ber_functions._cal_BER_only(tx_synced, bits_demod, threshold=0.8)


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
