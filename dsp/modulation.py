from __future__ import division
import numpy as np
from bitarray import bitarray

from . import resample
from . import theory
from . import ber_functions
from . import utils
from . import impairments
from .prbs import make_prbs_extXOR
from .signal_quality import quantize, generate_bitmapping_mtx

class QAMModulator(object):
    """
    A modulator object for modulating and demodulating rectangular QAM signals (without carrier frequency and equalisation). Currently only Gray coding of bits is supported.
    """

    def __init__(self, M, scaling_factor=None, coding=None):
        """
        Initialise QAM modulator

        Parameters
        ----------
        M        :  int
            number of constellation points, indicates QAM order

        scaling_factor: float, optional
            scaling factor to scale QAM symbols, if not given the symbols will
            scaled to an average power of 1
        coding   : string, optional
           coding method currently only the default gray coding is supported

        """
        self.M = M
        self.symbols = theory.cal_symbols_qam(M)
        if not scaling_factor:
            self._scale = theory.cal_scaling_factor_qam(M)
        else:
            self._scale = scaling_factor
        self.symbols /= np.sqrt(self._scale)
        self.coding = None
        self._graycode = theory.gray_code_qam(M)
        self.gray_coded_symbols = self.symbols[self._graycode]
        bformat = "0%db" % self.Nbits
        self._encoding = dict([(self.symbols[i].tobytes(),
                                bitarray(format(self._graycode[i], bformat)))
                               for i in range(len(self._graycode))])
        self.bitmap_mtx = generate_bitmapping_mtx(self.gray_coded_symbols, self.decode(self.gray_coded_symbols), self.M)

    @property
    def Nbits(self):
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
        return quantize(utils.normalise_and_center(signal), self.symbols)

    def generate_signal(self,
                      N,
                       snr,
                       carrier_df=0,
                       lw_LO = 0,
                       baudrate=1,
                       samplingrate=1,
                       PRBS=True,
                       PRBSorder=(15, 23),
                       PRBSseed=(None, None),
                       beta=0.1,
                        resample_noise=False, dual_pol=True):
        """
        Generate a M-QAM data signal array

        Parameters:
        ----------
        N :  int
            number of symbols to be generated.
        snr: number
            Signal to Noise Ratio (Es/N0) in logarithmic units, if None do not add noise
        carrier_df: number, optional
            carrier frequency offset, relative to the overall window, if not given it
            is set to 0 (baseband modulation)
        lw_LO : float, optional
            linewidth in frequency of the local oscillator for emulation of phase noise (default is 0, no phase noise)
        baudrate:  number, optional
            symbol rate of the signal. This should be the real symbol rate, used
            for calculating the oversampling factor. If not given it is 1.
        samplingrate: number, optional
            the rate at which the signal is sampled. Together with the baudrate
            this is used for calculating the oversampling factor. Note that if samplingrate is
            different to baudrate the length of the returned array is not N. (Default is 1.)
        PRBS: bool or tuple, optional
            If True the bits are generated as standard PRBS sequences, if False
            random bits are generated using numpy.random.randint. If the signal is dual_pol this needs
            be a tuple for each pol.
            (Default is True)
        PRBSOrder: int or tuple of bool, optional
            The PRBS order i.e. the length 2**order of the PRBS to use
            If dual_pol is True this needs to be a tuple of two ints for the polarisations
            (Default is 15)
        PRBSseed: array_like, optional
            Seed to the PRBS generator needs to be a 1D array of booleans of length order. If dual pol this needs to be a 2D array
          . (Default=None, which corresponds to a seed of all 1's)
        beta   : float, optional
            roll-off factor for the root raised cosine pulseshaping filter, value needs to be in range [0,1]
        resample_noise : bool
            whether to add the noise before resampling or after (default: False add noise after resampling)
        """
        out = []
        syms = []
        bits = []
        for i in range(2):
            Nbits = N * self.bits
            if PRBS == True:
                bitsq = make_prbs_extXOR(PRBSorder[i], Nbits, PRBSseed[i])
            else:
                bitsq = np.random.randint(0, high=2, size=Nbits).astype(np.bool)
            symbols = self.modulate(bitsq)
            if resample_noise:
                if snr is not None:
                    outdata = impairments.add_awgn(symbols, 10**(-snr/20))
                outdata = resample.resample_poly(baudrate, samplingrate, outdata)
            else:
                os = samplingrate/baudrate
                outdata = resample.rrcos_resample_zeroins(symbols, baudrate, samplingrate, beta=beta, Ts=1 / baudrate, renormalise=True)
                if snr is not None:
                    outdata = impairments.add_awgn(outdata, 10**(-snr/20)*np.sqrt(os))
            outdata *= np.exp(2.j * np.pi * np.arange(len(outdata)) * carrier_df / samplingrate)
            # not 100% clear if we should apply before or after resampling
            if lw_LO:
                outdata = impairments.apply_phase_noise(outdata, lw_LO, samplingrate)
            if dual_pol:
                out.append(outdata)
                syms.append(symbols)
                bits.append(bitsq)
            else:
                return outdata, symbols, bitsq
        return np.array(out), np.array(syms), np.array(bits)

    def cal_ser(self, signal_rx, symbols_tx=None, bits_tx=None, synced=False):
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


        Returns
        -------
        SER   : float
            symbol error rate
        """
        assert symbols_tx is not None or bits_tx is not None, "data_tx or symbol_tx must be given"
        if symbols_tx is None:
            symbols_tx = self.modulate(bits_tx)
        data_demod = self.quantize(signal_rx)
        if not synced:
            symbols_tx, data_demod = ber_functions.sync_and_adjust(symbols_tx, data_demod)
        return np.count_nonzero(data_demod - symbols_tx)/len(data_demod)

    def cal_ber(self, signal_rx, symbols_tx=None, bits_tx=None, synced=False, PRBS=(15, utils.bool2bin(np.ones(15)))):
        """
        Calculate the bit-error-rate for the given signal, against either a PRBS sequence or a given bit sequence.

        Parameters
        ----------

        signal_rx    : array_like
            received signal to demodulate and calculate BER of

        bits_tx      : array_like, optional
            transmitted bit sequence to compare against. (default is None, which either PRBS or syms_tx has to be given)

        symbols_tx      : array_like, optional
            transmitted bit sequence to compare against. (default is None, which means bits_tx or PRBS has to be given)

        synced    : bool, optional
            whether signal_tx and symbol_tx are synchronised.

        PRBS         : tuple(int, int), optional
            tuple of PRBS order and seed, the order has to be integer 7, 15, 23, 31 and the seed has to be None or a binary array of length of the PRBS order. If the seed is None it will be initialised to all bits one.


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
        if symbols_tx is None:
            if bits_tx is None:
                bits_tx = make_prbs_extXOR( PRBS[0], len(syms_demod)*self.bits, seed=PRBS[1])
            symbols_tx = self.modulate(bits_tx)
        if not synced:
            s_tx_sync, syms_demod = ber_functions.sync_and_adjust(symbols_tx, syms_demod)
        bits_demod = self.decode(syms_demod)
        tx_synced = self.decode(s_tx_sync)
        return ber_functions.cal_ber_syncd(tx_synced, bits_demod, threshold=0.8)[0]

    def cal_evm(self, signal_rx, symbols_tx=None):
        """
        Calculate the Error Vector Magnitude of the input signal either blindly or against a known symbol sequence, after _[1]. The EVM here is normalised to the average symbol power, not the peak as in some other definitions.

        Parameters
        ----------

        signal_rx    : array_like
            input signal to measure the EVM offset

        symbols_tx      : array_like, optional
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
        if symbols_tx is None:
            symbols_tx = self.quantize(signal_rx)
        else:
            symbols_tx, signal_ad = ber_functions.sync_and_adjust(symbols_tx, signal_rx)
        return np.sqrt(np.mean(utils.cabssquared(symbols_tx - signal_rx)))#/np.mean(abs(self.symbols)**2))

