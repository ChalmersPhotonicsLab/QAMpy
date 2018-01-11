from __future__ import division
import numpy as np
from bitarray import bitarray

from . import resample
from . import theory
from . import ber_functions
from . import utils
from . import impairments
from .prbs import make_prbs_extXOR
from .signal_quality import quantize, generate_bitmapping_mtx, estimate_snr, soft_l_value_demapper

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

    def _sync_and_adjust(self, tx, rx):
        tx_out = []
        rx_out = []
        for i in range(tx.shape[0]):
            t, r = ber_functions.sync_and_adjust(tx[i], rx[i])
            tx_out.append(t)
            rx_out.append(r)
        return np.array(tx_out), np.array(rx_out)

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
        rem = len(data) % self.Nbits
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
            array of complex input symbols

        Note
        ----
        Unlike the other functions this function does not always return a 2D array.

        Returns
        -------
        outbits   : array_like
            array of booleans representing bits with same number of dimensions as symbols
        """
        if symbols.ndim is 1:
            bt = bitarray()
            bt.encode(self._encoding, symbols.view(dtype="S16"))
            return np.fromstring(bt.unpack(zero=b'\x00', one=b'\x01'), dtype=np.bool)
        bits = []
        for i in range(symbols.shape[0]):
            bt = bitarray()
            bt.encode(self._encoding, symbols[i].view(dtype="S16"))
            bits.append(np.fromstring(bt.unpack(zero=b'\x00', one=b'\x01'), dtype=np.bool) )
        return np.array(bits)


    def quantize(self, signal):
        """
        Make symbol decisions based on the input field. Decision is made based on difference from constellation points

        Parameters
        ----------
        signal   : array_like
            2D array of the input signal

        Returns
        -------
        symbols  : array_like
            2d array of the detected symbols
        """
        signal = np.atleast_2d(signal)
        outsyms = np.zeros_like(signal)
        for i in range(signal.shape[0]):
            outsyms[i] = quantize(utils.normalise_and_center(signal[i]), self.symbols)
        return outsyms

    def generate_signal(self, N, snr, carrier_df=0, lw_LO=0, baudrate=1, samplingrate=1, PRBS=True, PRBSorder=(15, 23),
                        PRBSseed=(None, None), beta=0.1, resample_noise=False, ndim=2):
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
        ndim : interger
            number of dimensions of the generated signal (default=2 dual polarization signal)

        Returns
        -------
        signal : array_like
            the calculated signal array (2D)
        syms  : array_like
            the transmitted symbols (2D)
        bits : array_like
            the transmitted bits (2D)

        """
        out = []
        syms = []
        bits = []
        for i in range(ndim):
            Nbits = N * self.Nbits
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
            out.append(outdata)
            syms.append(symbols)
            bits.append(bitsq)
        self.symbols_tx = np.array(syms)
        self.bits_tx = np.array(bits)
        return np.array(out), self.symbols_tx, self.bits_tx

    def cal_ser(self, signal_rx, symbols_tx=None, bits_tx=None, synced=False):
        """
        Calculate the symbol error rate of the received signal.Currently does not check
        for correct polarization.

        Parameters
        ----------
        signal_rx  : array_like
            Received signal (2D complex array)
        symbols_tx  : array_like, optional
            symbols at the transmitter for comparison against signal.
        bits_tx    : array_like, optional
            bitstream at the transmitter for comparison against signal.
        synced    : bool, optional
            whether signal_tx and symbol_tx are synchronised.

        Note
        ----
        If neither symbols_tx or bits_tx are given use self.symbols_tx

        Returns
        -------
        SER   : array_like
            symbol error rate per dimension
        """
        signal_rx = np.atleast_2d(signal_rx)
        ndim = signal_rx.shape[0]
        if symbols_tx is None:
            if bits_tx is None:
                symbols_tx = self.symbols_tx
            else:
                symbols_tx = np.zeros_like(signal_rx)
                bits_tx = np.atleast_2d(bits_tx)
                for i in range(ndim):
                    symbols_tx[i] = self.modulate(bits_tx[i])
        else:
            symbols_tx = np.atleast_2d(symbols_tx)
        data_demod = self.quantize(signal_rx)
        if not synced:
            symbols_tx, data_demod = self._sync_and_adjust(symbols_tx, data_demod)
        errs = np.count_nonzero(data_demod - symbols_tx, axis=-1)
        return errs/data_demod.shape[1]

    def cal_ber(self, signal_rx, symbols_tx=None, bits_tx=None, synced=False):
        """
        Calculate the bit-error-rate for the received signal compared to transmitted symbols or bits. Currently does not check
        for correct polarization.

        Parameters
        ----------
        signal_rx    : array_like
            received signal to demodulate and calculate BER of
        bits_tx      : array_like, optional
            transmitted bit sequence to compare against.
        symbols_tx      : array_like, optional
            transmitted bit sequence to compare against.
        synced    : bool, optional
            whether signal_tx and symbol_tx are synchronised.

        Note
        ----
        If neither bits_tx or symbols_tx are given, we use the self.symbols_tx


        Returns
        -------
        ber          :  array_like
            bit-error-rate in linear units per dimension
        """
        signal_rx = np.atleast_2d(signal_rx)
        ndim = signal_rx.shape[0]
        syms_demod = self.quantize(signal_rx)
        if symbols_tx is None:
            if bits_tx is None:
                symbols_tx = self.symbols_tx
            else:
                symbols_tx = []
                bits_tx = np.atleast_2d(bits_tx)
                for i in range(ndim):
                    symbols_tx.append(self.modulate(bits_tx[i]))
                symbols_tx = np.array(symbols_tx)
        else:
            symbols_tx = np.atleast_2d(symbols_tx)
        if not synced:
            symbols_tx, syms_demod = self._sync_and_adjust(symbols_tx, syms_demod)
        bits_demod = self.decode(syms_demod)
        tx_synced = self.decode(symbols_tx)
        errs = np.count_nonzero(tx_synced - bits_demod, axis=-1)
        return errs/bits_demod.shape[1]

    def cal_evm(self, signal_rx, blind=False, symbols_tx=None):
        """
        Calculate the Error Vector Magnitude of the input signal either blindly or against a known symbol sequence, after _[1].
        The EVM here is normalised to the average symbol power, not the peak as in some other definitions. Currently does not check
        for correct polarization.

        Parameters
        ----------
        signal_rx    : array_like
            input signal to measure the EVM offset
        blind : bool, optional
            calculate the blind EVM (signal is quantized, without taking into account symbol errors). For low SNRs this
            will underestimate the real EVM, because detection errors are not counted.
        symbols_tx      : array_like, optional
            known symbol sequence. If this is None self.symbols_tx will be used unless blind is True.

        Returns
        -------
        evm       : array_like
            RMS EVM per dimension

        References
        ----------
        ...[1] Shafik, R. "On the extended relationships among EVM, BER and SNR as performance metrics". In Conference on Electrical and Computer Engineering (p. 408) (2006).
         Retrieved from http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=4178493


        Note
        ----
        The to calculate the EVM in dB from the RMS EVM we need calculate 10 log10(EVM**2). This differs from some defintions
        of EVM, e.g. on wikipedia.
        """
        signal_rx = np.atleast_2d(signal_rx)
        if blind:
            symbols_tx = self.quantize(signal_rx)
        else:
            if symbols_tx is None:
                symbols_tx = self.symbols_tx
            else:
                symbols_tx = np.atleast_2d(symbols_tx)
            symbols_tx, signal_ad = self._sync_and_adjust(symbols_tx, signal_rx)
        return np.sqrt(np.mean(utils.cabssquared(symbols_tx - signal_rx), axis=-1))#/np.mean(abs(self.symbols)**2))

    def est_snr(self, signal_rx, symbols_tx=None, synced=False):
        """
        Estimate the SNR of a given input signal, using known symbols.

        Parameters
        ----------
        signal_rx : array_like
            input signal
        symbols_tx : array_like, optional
            known transmitted symbols (default: None means that self.symbols_tx are used)
        synced : bool, optional
            whether the signal and symbols are synchronized already

        Returns
        -------
        snr: array_like
            snr estimate per dimension
        """
        signal_rx = np.atleast_2d(signal_rx)
        ndims = signal_rx.shape[0]
        if symbols_tx is None:
            symbols_tx = self.symbols_tx
        else:
            symbols_tx = np.atleast_2d(symbols_tx)
        if not synced:
            symbols_tx, signal_rx = self._sync_and_adjust(symbols_tx, signal_rx)
        snr = np.zeros(ndims, dtype=np.float64)
        for i in range(ndims):
            snr[i] = estimate_snr(signal_rx[i], symbols_tx[i], self.gray_coded_symbols)
        return snr

    def cal_gmi(self, signal_rx, symbols_tx=None):
        """
        Calculate the generalized mutual information for the received signal.

        Parameters
        ----------
        signal_rx : array_like
            equalised input signal
        symbols_tx : array_like
            transmitted symbols (default:None use self.symbols_tx of the modulator)

        Returns
        -------
        gmi : array_like
            generalized mutual information per mode
        gmi_per_bit : array_like
            generalized mutual information per transmitted bit per mode
        """
        signal_rx = np.atleast_2d(signal_rx)
        if symbols_tx is None:
            symbols_tx = self.symbols_tx
        else:
            symbols_tx = np.atleast_2d(symbols_tx)
        ndims = signal_rx.shape[0]
        GMI = np.zeros(ndims, dtype=np.float64)
        GMI_per_bit = np.zeros((ndims, self.Nbits), dtype=np.float64)
        mm = np.sqrt(np.mean(np.abs(signal_rx)**2, axis=-1))
        signal_rx = signal_rx/mm[:,np.newaxis]
        tx, rx = self._sync_and_adjust(symbols_tx, signal_rx)
        snr = self.est_snr(rx, symbols_tx=tx, synced=True)
        bits = self.decode(self.quantize(tx)).astype(np.int)
        # For every mode present, calculate GMI based on SD-demapping
        for mode in range(ndims):
            l_values = soft_l_value_demapper(rx[mode], self.M, snr[mode], self.bitmap_mtx)
            # GMI per bit
            for bit in range(self.Nbits):
                GMI_per_bit[mode, bit] = 1 - np.mean(np.log2(1+np.exp(((-1)**bits[mode, bit::self.Nbits])*l_values[bit::self.Nbits])))
            GMI[mode] = np.sum(GMI_per_bit[mode])
        return GMI, GMI_per_bit


