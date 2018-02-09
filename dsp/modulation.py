from __future__ import division
import numpy as np
import abc
import fractions
import warnings
from bitarray import bitarray

from . import resample
from . import theory
from . import ber_functions
from . import utils
from . import impairments
from . import pilotbased_receiver
from .prbs import make_prbs_extXOR
from .signal_quality import quantize, generate_bitmapping_mtx, estimate_snr, soft_l_value_demapper


class SymbolBase(np.ndarray):
    __metaclass__ = abc.ABCMeta
    _inheritattr_ = [] #list of attributes names that should be inherited

    @classmethod
    @abc.abstractmethod
    def _demodulate(cls, symbols):
        """
        Demodulate an array of input symbols to bits according

        Parameters
        ----------
        symbols   : array_like
            array of complex input symbols
        mapping   : array_like
            mapping between symbols and bits
        kwargs
            other arguments to use

        Returns
        -------
        bits   : array_like
            array of booleans representing bits with same number of dimensions as symbols
        """

    @classmethod
    @abc.abstractmethod
    def _modulate(cls, bits):
        """
        Modulate a bit sequence into symbols

        Parameters
        ----------
        bits     : array_like
           1D array of bits represented as bools. If the len(data)%self.M != 0 then we only encode up to the nearest divisor

        Returns
        -------
        outdata  : array_like
            1D array of complex symbol values. Normalised to energy of 1
        """

    def __array_finalize__(self, obj):
        if obj is None: return
        for attr in self._inheritattr_:
            setattr(self, attr, getattr(obj, attr, None))
        if hasattr(obj, "_symbols"):
            s =  getattr(obj, "_symbols")
            if s is None:
                self._symbols = obj
            else:
                self._symbols = obj._symbols

class RandomBits(np.ndarray):
    def __new__(cls, N, nmodes=1, seed=None):
        R = np.random.RandomState(seed)
        bitsq = R.randint(0, high=2, size=(nmodes,N)).astype(np.bool)
        obj = bitsq.view(cls)
        obj._rand_state = R
        obj._seed = seed
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self._seed = getattr(obj, "_seed", None)
        self._rand_state = getattr(obj, "_rand_state", None)

class PRBSBits(np.ndarray):
    def __new__(cls, N, nmodes=1, seed=[None, None], order=[15, 23]):
        if len(order) < nmodes:
            warnings.warn("PRBSorder is not given for all dimensions, picking random orders and seeds")
            order_n = []
            seed_n = []
            orders = [15, 23]
            for i in range(nmodes):
                try:
                   order_n.append(order[i])
                   seed_n.append(seed[i])
                except IndexError:
                    o = np.random.choice(orders)
                    order_n.append(o)
                    s = np.random.randint(0, 2**o)
                    seed_n.append(s)
                order = order_n
                seed = seed_n
        bits = np.empty((nmodes, N), dtype=np.bool)
        for i in range(nmodes):
            bits[i][:] = make_prbs_extXOR(order[i], N, seed[i])
        obj = bits.view(cls)
        obj._order = order
        obj._seed = seed
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self._seed = getattr(obj, "_seed", None)
        self._order = getattr(obj, "_order", None)


class QAMSymbolsGrayCoded(SymbolBase):
    _inheritattr_ = ["_M", "_symbols", "_bits", "_encoding", "_bitmap_mtx", "_fb", "_code",
                     "_coded_symbols"]

    @staticmethod
    def _demodulate( symbols, encoding):
        """
        Decode array of input symbols to bits according to the coding of the modulator.

        Parameters
        ----------
        symbols   : array_like
            array of complex input symbols
        encoding  : array_like
            mapping between symbols and bits

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
            bt.encode(encoding, symbols.view(dtype="S16"))
            return np.fromstring(bt.unpack(zero=b'\x00', one=b'\x01'), dtype=np.bool)
        bits = []
        for i in range(symbols.shape[0]):
            bt = bitarray()
            bt.encode(encoding, symbols[i].view(dtype="S16"))
            bits.append(np.fromstring(bt.unpack(zero=b'\x00', one=b'\x01'), dtype=np.bool) )
        return np.array(bits)

    @staticmethod
    def _modulate(data, encoding, M, dtype=np.complex128):
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
        data = np.atleast_2d(data)
        nmodes = data.shape[0]
        bitspsym = int(np.log2(M))
        Nsym = data.shape[1]//bitspsym
        out = np.empty((nmodes,Nsym), dtype=dtype)
        N = data.shape[1] - data.shape[1]%bitspsym
        for i in range(nmodes):
            datab = bitarray()
            datab.pack(data[i, :N].tobytes())
            # the below is not really the fastest method but easy encoding/decoding is possible
            out[i,:] = np.fromstring(b''.join(datab.decode(encoding)), dtype=dtype)
        return out

    @classmethod
    def from_symbol_array(cls, symbs, M=None, fb=1):
        symbs = np.atleast_2d(symbs)
        if M is None:
            warnings.warn("no M given, estimating how mnay unique symbols are in array, this can cause errors")
            M = np.unique(symbs).shape[0]
        scale = np.sqrt(theory.cal_scaling_factor_qam(M))/np.sqrt((abs(np.unique(symbs))**2).mean())
        coded_symbols, graycode, encoding, bitmap_mtx = cls._generate_mapping(M, scale)
        out = np.empty_like(symbs)
        for i in range(symbs.shape[0]):
            out[i] = quantize(symbs[i], coded_symbols)
        bits = cls._demodulate(out, encoding)
        obj = np.asarray(out).view(cls)
        obj._M = M
        obj._fb = fb
        obj._bits = bits
        obj._encoding = encoding
        obj._code = graycode
        obj._coded_symbols = coded_symbols
        obj._symbols = None
        return obj

    @classmethod
    def from_bit_array(cls, bits, M, fb=1):
        arr = np.atleast_2d(bits)
        nbits = int(np.log2(M))
        if arr.shape[1]%nbits > 0:
            warnings.warn("Length of bits not divisible by log2(M) truncating")
            len = arr.shape[1]//nbits*nbits
            arr = arr[:, :len]
        scale = np.sqrt(theory.cal_scaling_factor_qam(M))
        coded_symbols, graycode, encoding, bitmap_mtx = cls._generate_mapping(M, scale)
        #out = []
        #for i in range(arr.shape[0]):
        #    out.append( cls._modulate(arr[i], encoding, M))
        out = cls._modulate(arr, encoding, M)
        #out = np.asarray(out)
        obj = np.asarray(out).view(cls)
        obj._M = M
        obj._fb = fb
        obj._bits = bits
        obj._encoding = encoding
        obj._code = graycode
        obj._coded_symbols = coded_symbols
        obj._symbols = None
        return obj

    @classmethod
    def _generate_mapping(cls, M, scale):
        Nbits = np.log2(M)
        symbols = theory.cal_symbols_qam(M)
        # check if this gives the correct mapping
        symbols /= scale
        _graycode = theory.gray_code_qam(M)
        coded_symbols = symbols[_graycode]
        bformat = "0%db" % Nbits
        encoding = dict([(symbols[i].tobytes(),
                                bitarray(format(_graycode[i], bformat)))
                               for i in range(len(_graycode))])
        bitmap_mtx = generate_bitmapping_mtx(coded_symbols, cls._demodulate(coded_symbols, encoding), M)
        return coded_symbols, _graycode, encoding, bitmap_mtx


    # using Randombits as default class because they are slightly faster
    def __new__(cls, M, N, nmodes=1, fb=1, bitclass=RandomBits, **kwargs):
        scale = np.sqrt(theory.cal_scaling_factor_qam(M))
        coded_symbols, _graycode, encoding, bitmap_mtx = cls._generate_mapping(M, scale)
        Nbits = int(N * np.log2(M))
        bits = bitclass(Nbits, nmodes=nmodes, **kwargs)
        obj = cls._modulate(bits, encoding, M)
        obj = obj.view(cls)
        obj._bitmap_mtx = bitmap_mtx
        obj._encoding = encoding
        obj._coded_symbols = coded_symbols
        obj._M = M
        obj._fb = fb
        obj._code = _graycode
        obj._bits = bits
        obj._symbols = None
        return obj

    @property
    def symbols(self):
        return self._symbols

    @property
    def coded_symbols(self):
        return self._coded_symbols

    @property
    def bits(self):
        return self._bits

    @property
    def M(self):
        return self._M

    @property
    def fb(self):
        return self._fb

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
        return self._modulate(data, self._encoding, self.M)

    def demodulate(self, symbols):
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
             for i in range(signal.shape[0]):
            outsyms[i] = quantize(utils.normalise_and_center(signal[i]), self.coded_symbols)       array of booleans representing bits with same number of dimensions as symbols
        """
        return self._demodulate(symbols, self._encoding)

class SignalQualityMixing(object):

    def _signal_present(self, signal):
        if signal is None:
            return self
        else:
            return np.atleast_2d(signal)

    def quantize(self, signal=None):
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
        signal = self._signal_present(signal)
        outsyms = np.zeros_like(signal)
        for i in range(signal.shape[0]):
            outsyms[i] = quantize(utils.normalise_and_center(signal[i]), self.coded_symbols)
        return outsyms

    def _sync_and_adjust(self, tx, rx):
        tx_out = []
        rx_out = []
        for i in range(tx.shape[0]):
            t, r = ber_functions.sync_and_adjust(tx[i], rx[i])
            tx_out.append(t)
            rx_out.append(r)
        return np.array(tx_out), np.array(rx_out)


    def cal_ser(self, signal_rx=None, synced=False):
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
        signal_rx = self._signal_present(signal_rx)
        nmodes = signal_rx.shape[0]
        data_demod = self.quantize(signal_rx)
        if not synced:
            symbols_tx, data_demod = self._sync_and_adjust(self.symbols, data_demod)
        else:
            symbols_tx = self.symbols
        errs = np.count_nonzero(data_demod - symbols_tx, axis=-1)
        return np.asarray(errs)/data_demod.shape[1]

    def cal_ber(self, signal_rx=None, synced=False):
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
        signal_rx = self._signal_present(signal_rx)
        nmodes = signal_rx.shape[0]
        syms_demod = self.quantize(signal_rx)
        if not synced:
            symbols_tx, syms_demod = self._sync_and_adjust(self.symbols, syms_demod)
        else:
            symbols_tx = self.symbols
        #TODO: need to rename decode to demodulate
        bits_demod = self.demodulate(syms_demod)
        tx_synced = self.demodulate(symbols_tx)
        errs = np.count_nonzero(tx_synced - bits_demod, axis=-1)
        return np.asarray(errs)/bits_demod.shape[1]

    def cal_evm(self, signal_rx=None, blind=False):
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
        signal_rx = self._signal_present(signal_rx)
        nmodes = signal_rx.shape[0]
        symbols_tx, signal_ad = self._sync_and_adjust(self.symbols, signal_rx)
        return np.asarray(np.sqrt(np.mean(utils.cabssquared(symbols_tx - signal_rx), axis=-1)))#/np.mean(abs(self.symbols)**2))

    def est_snr(self, signal_rx=None, synced=False):
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
        signal_rx = self._signal_present(signal_rx)
        nmodes = signal_rx.shape[0]
        if not synced:
            symbols_tx, signal_rx = self._sync_and_adjust(self.symbols, signal_rx)
        else:
            symbols_tx = self.symbols
        snr = np.zeros(nmodes, dtype=np.float64)
        for i in range(nmodes):
            snr[i] = estimate_snr(signal_rx[i], symbols_tx[i], self.coded_symbols)
        return np.asarray(snr)

    def cal_gmi(self, signal_rx=None):
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
        signal_rx = self._signal_present(signal_rx)
        symbols_tx = self.symbols
        nmodes = signal_rx.shape[0]
        GMI = np.zeros(nmodes, dtype=np.float64)
        GMI_per_bit = np.zeros((nmodes, self.Nbits), dtype=np.float64)
        mm = np.sqrt(np.mean(np.abs(signal_rx)**2, axis=-1))
        signal_rx = signal_rx/mm[:,np.newaxis]
        tx, rx = self._sync_and_adjust(symbols_tx, signal_rx)
        snr = self.est_snr(rx, synced=True)
        bits = self.demodulate(self.quantize(tx)).astype(np.int)
        # For every mode present, calculate GMI based on SD-demapping
        for mode in range(nmodes):
            l_values = soft_l_value_demapper(rx[mode], self.M, snr[mode], self._bitmap_mtx)
            # GMI per bit
            for bit in range(self.Nbits):
                GMI_per_bit[mode, bit] = 1 - np.mean(np.log2(1+np.exp(((-1)**bits[mode, bit::self.Nbits])*l_values[bit::self.Nbits])))
            GMI[mode] = np.sum(GMI_per_bit[mode])
        return GMI, GMI_per_bit


class Signal(SymbolBase, SignalQualityMixing):
    _inheritattr_ = ["_bits", "_symbols", "_fs", "_fb"]
    def __new__(cls, M, N, fb=1, fs=1, nmodes=1, symbolclass=QAMSymbolsGrayCoded, dtype=np.complex128,
                classkwargs={}, resamplekwargs={"beta": 0.1}):
        obj = symbolclass(M,N, fb=fb, nmodes=nmodes, **classkwargs)
        os = fs/fb
        #TODO: check if we are not wasting memory here
        if not np.isclose(os, 1):
            onew = cls._resample_array(obj, fs, **resamplekwargs)
        else:
            onew = obj.copy().view(cls)
            onew._symbols = obj
            onew._fs = fs
        return onew

    @classmethod
    def _resample_array(cls, obj, fs, **kwargs):
        if hasattr(obj, "fs"):
            fold = obj.fs
            fb = obj.fb
        else:
            fb = obj.fb
            fold = fb
        os = fs/fold
        onew = np.empty((obj.shape[0], int(os*obj.shape[1])), dtype=obj.dtype)
        for i in range(obj.shape[0]):
            onew[i,:] = resample.rrcos_resample_zeroins(obj[i], fold, fs, Ts=1/fb, **kwargs)
        onew = np.asarray(onew).view(cls)
        syms = getattr(obj, "_symbols", None)
        if syms is None:
            onew._symbols = obj
        else:
            onew._symbols = obj._symbols
        onew._fs = fs
        return onew

    @property
    def fs(self):
        return self._fs

    @property
    def M(self):
        return self._symbols.M

    @property
    def fb(self):
        return self._symbols._fb

    @property
    def symbols(self):
        return self._symbols

    @classmethod
    def from_symbol_array(cls, array, fs, **kwargs):
        os = fs/array.fb
        if not np.isclose(os, 1):
            onew = cls._resample_array(array, fs, **kwargs)
        else:
            onew = array.view(cls)
            onew._symbols = array
            onew._fs = fs
        return onew

    def resample(self, fnew, **kwargs):
        return self._resample_array(self, fnew, **kwargs)

    def __getattr__(self, attr):
        return getattr(self._symbols, attr)


class TDHQAMSymbols(SymbolBase, SignalQualityMixing):
    _inheritattr_ = ["_M", "_symbols_M1", "_symbols_M2", "_fb", "_fr", "_symbols"]
    @staticmethod
    def _cal_fractions(fr):
        ratn = fractions.Fraction(fr).limit_denominator()
        f_M2 = ratn.numerator
        f_M = ratn.denominator
        f_M1 = f_M - f_M2 
        return f_M, f_M1, f_M2

    @staticmethod
    def _cal_symbol_idx(N, f_M, f_M1):
        idx = np.arange(N)
        idx1 = idx%f_M < f_M1
        idx2 = idx%f_M >= f_M1
        return idx, idx1, idx2

    def __new__(cls, M, N, fr=0.5, power_method="dist", snr=None, nmodes=1, fb=1,
                M1class=QAMSymbolsGrayCoded, M2class=QAMSymbolsGrayCoded, **kwargs):
        """
        Time-domain hybrid QAM (TDHQAM) modulator with two QAM-orders.

        Parameters
        ----------
        M1 : integer
            QAM order of the first part.
        M2 : integer
            QAM order of the second part
        fr : float
            fraction of the second format of the overall frame length
        power_method : string, optional
            method to calculate the power ratio of the different orders, currently on "dist" is implemented
        snr : float
            Design signal-to-noise ratio needed when using BER for calculation of the power ratio, currently does nothing
        """
        M1 = M[0]
        M2 = M[1]
        f_M, f_M1, f_M2 = cls._cal_fractions(fr)
        frms = N//f_M
        if N%f_M > 0:
            N = f_M * frms
            warnings.warn("length of overall pattern not divisable by number of frames, truncating to %d symbols"%N)
        N1 = frms * f_M1
        N2 = frms * f_M2
        out = np.zeros((nmodes, N), dtype=np.complex128)
        syms1 = M1class( M1, N1, nmodes=nmodes, fb=fb, **kwargs)
        syms2 = M2class( M2, N2, nmodes=nmodes, fb=fb, **kwargs)
        scale = cls.calculate_power_ratio(syms1.coded_symbols, syms2.coded_symbols, power_method)
        syms2 /= np.sqrt(scale)
        idx, idx1, idx2 = cls._cal_symbol_idx(N, f_M, f_M1)
        out[:,idx1] = syms1
        out[:,idx2] = syms2
        obj = out.view(cls)
        obj._symbols_M1 = syms1
        obj._symbols_M2 = syms2
        obj._fr = fr
        obj._fb = fb
        obj._M = M
        obj._power_method = power_method
        return obj

    @property
    def f_M(self):
        fM, fM1, fM2 = self._cal_fractions(self.fr)
        return fM
    
    @property
    def f_M1(self):
        fM, fM1, fM2 = self._cal_fractions(self.fr)
        return fM1
    
    @property
    def f_M2(self):
        fM, fM1, fM2 = self._cal_fractions(self.fr)
        return fM2

    @property
    def M(self):
        return (self._symbols_M1.M, self._symbols_M2.M)

    @property
    def fr(self):
        return self._fr

    @property
    def fb(self):
        return self._fb

    @classmethod
    def from_symbol_arrays(cls, syms_M1, syms_M2, fr, power_method="dist"):
        assert syms_M1.ndim == 2 and syms_M2.ndim == 2, "input needs to have two dimensions"
        assert syms_M1.shape[0] == syms_M2.shape[0], "Number of modes must be the same"
        f_M, f_M1, f_M2 = cls._cal_fractions(fr)
        scale = cls.calculate_power_ratio(syms_M1.coded_symbols, syms_M2.coded_symbols, power_method)
        syms_M2 /= np.sqrt(scale)
        N1 = syms_M1.shape[1]
        N2 = syms_M2.shape[1] 
        N = N1 + N2
        nframes = N//f_M
        if (nframes * f_M1 > N1) or (nframes * f_M2 > N2):
            warnings.warn("Need to truncate input arrays as ratio is not possible otherwise")
            nframes = min(N1//f_M1, N2//f_M2)
        N = nframes * f_M
        out = np.zeros((syms_M1.shape[0], N), dtype=syms_M1.dtype)
        idx, idx1, idx2 = cls._cal_symbol_idx(N, f_M, f_M1)
        out[:,idx1] = syms_M1
        out[:,idx2] = syms_M2
        obj = out.view(cls)
        obj._symbols_M1 = syms_M1
        obj._symbols_M2 = syms_M2
        obj._fr = fr
        obj._fb = syms_M1.fb
        obj._power_method = power_method
        return obj

    @staticmethod
    def calculate_power_ratio(M1symbols, M2symbols, method="dist"):
        if method is "dist":
            d1 = np.min(abs(np.diff(np.unique(M1symbols))))
            d2 = np.min(abs(np.diff(np.unique(M2symbols))))
            scf = (d2/d1)**2
            return scf
        else:
            raise NotImplementedError("Only 'dist' method is currently implemented")

    def _divide_signal_frame(self, signal):
        idx = np.arange(signal.shape[1])
        idx1 = idx[idx%self.f_M < self.f_M1]
        idx2 = idx[idx%self.f_M >= self.f_M1]
        syms1 = np.zeros((signal.shape[0], idx1.shape[0]), dtype=signal.dtype)
        syms2 = np.zeros((signal.shape[0], idx2.shape[0]), dtype=signal.dtype)
        if self.M[0] > self.M[1]:
            idx_m = idx1
        else:
            idx_m = idx2
        if self._power_method == "dist":
            idx_max = []
            for i in range(signal.shape[0]):
                imax = 0
                pmax = -10
                for j in range(self._frame_len):
                    pmax_n = np.mean(abs(signal[i, (idx_m+j)%idx.max()]))
                    if pmax_n > pmax:
                        imax = j
                        pmax = pmax_n
                syms1[i,:] = signal[i, (idx1+imax)%idx.max()]
                syms2[i,:] = signal[i, (idx2+imax)%idx.max()]
            return self._symbols_M1.from_symbol_array(syms1, fb=self.fb, M=self.M[0]), \
                   self._symbols_M2.from_symbol_array(syms2, fb=self.fb, M=self.M[1])
        else:
            raise NotImplementedError("currently only 'dist' method is implemented")

    def _demodulate(self):
        raise NotImplementedError("Use demodulation of subclasses")

    def _modulate(self):
        raise NotImplementedError("Use modulation of subclasses")

class SignalWithPilots(Signal,SignalQualityMixing):
    _inheritattr_ = ["_pilots", "_symbols", "_frame_len", "_pilot_seq_len", "_nframes"]

    @staticmethod
    def _cal_pilot_idx(frame_len, pilot_seq_len, pilot_ins_rat):
        idx = np.arange(frame_len)
        idx_pil_seq = idx < pilot_seq_len
        if pilot_ins_rat == 0 or pilot_ins_rat is None:
            idx_pil = idx_pil_seq
        else:
            if (frame_len - pilot_seq_len)%pilot_ins_rat != 0:
                raise ValueError("Frame without pilot sequence divided by pilot rate needs to be an integer")
            N_ph_frames = (frame_len - pilot_seq_len)//pilot_ins_rat
            idx_ph_pil = ((idx-pilot_seq_len)%pilot_ins_rat != 0) & (idx-pilot_seq_len >0)
            idx_pil = ~idx_ph_pil #^ idx_pil_seq
        idx_dat = ~idx_pil
        return idx, idx_dat, idx_pil

    def __new__(cls, M, frame_len, pilot_seq_len, pilot_ins_rat, nframes=1, scale_pilots=1,
                dataclass=QAMSymbolsGrayCoded, nmodes=1, **kwargs):
        out_symbs = np.empty((nmodes, frame_len), dtype=np.complex128)
        idx, idx_dat, idx_pil = cls._cal_pilot_idx(frame_len, pilot_seq_len, pilot_ins_rat)
        pilots = QAMSymbolsGrayCoded(4, np.count_nonzero(idx_pil), nmodes=nmodes, **kwargs)*scale_pilots
        # Note that currently the phase pilots start one symbol after the sequence
        # TODO: we should probably fix this
        out_symbs[:, idx_pil] = pilots
        symbs = dataclass(M, np.count_nonzero(idx_dat), nmodes=nmodes, **kwargs)
        out_symbs[:, idx_dat] = symbs
        out_symbs = np.tile(out_symbs, nframes)
        obj = out_symbs.view(cls)
        obj._frame_len = frame_len
        obj._pilot_seq_len = pilot_seq_len
        obj._pilot_ins_rat = pilot_ins_rat
        obj._nframes = nframes
        obj._symbols = symbs
        obj._pilots = pilots
        obj._idx_dat = idx_dat
        return obj

    @classmethod
    def from_data_array(cls, data, frame_len, pilot_seq_len, pilot_ins_rat, nframes=1, scale_pilots=1, **pilot_kwargs):
        nmodes, N = data.shape
        idx, idx_dat, idx_pil = cls._cal_pilot_idx(frame_len, pilot_seq_len, pilot_ins_rat)
        assert np.count_nonzero(idx_dat) <= N, "data frame is to short for the given frame length"
        if np.count_nonzero(idx_dat) > N:
            warnings.warn("Data for frame is shorter than length of data array, truncating")
        out_symbs = np.empty((nmodes, frame_len), dtype=data.dtype)
        Ndat = np.count_nonzero(idx_dat)
        pilots = QAMSymbolsGrayCoded(4, np.count_nonzero(idx_pil), nmodes=nmodes, **pilot_kwargs)/np.sqrt(scale_pilots)
        out_symbs[:, idx_pil] = pilots
        out_symbs[:, idx_dat] = data[:, :Ndat]
        out_symbs = np.tile(out_symbs, nframes)
        obj = out_symbs.view(cls)
        obj._frame_len = frame_len
        obj._pilot_seq_len = pilot_seq_len
        obj._pilot_ins_rat = pilot_ins_rat
        obj._nframes = nframes
        obj._symbols = data[:, :Ndat]
        obj._pilots = pilots
        obj._idx_dat = idx_dat
        return obj

    @property
    def pilot_seq(self):
        return self._pilots[:self._pilot_seq_len]

    @property
    def ph_pilots(self):
        return self._pilots[self._pilot_seq_len::self._pilot_ins_rat]

    @property
    def pilots(self):
        return self._pilots

    @property
    def nframes(self):
        return self._nframes

    @property
    def frame_len(self):
        return self._frame_len

    def get_data(self, shift_factors=None):
        if shift_factors is None:
            idx = np.tile(self._idx_dat, self.nframes)
            return self[:,idx]
        idxn = np.tile(self._idx_dat, self.nframes)
        idx_o = []
        for sf in shift_factors:
            idx_o.append(np.roll(idxn, sf))
        idx_o = np.array(idx_o)
        return self[idx_o].reshape(shift_factors.shape[0], idxn.shape[0])

    def __getattr__(self, attr):
        return getattr(self._symbols, attr)

    def cal_ser(self, signal_rx=None, synced=False):
        if signal_rx is None:
            signal_rx = self.get_data()
        return super().cal_ser(signal_rx, synced)

    def cal_ber(self, signal_rx=None, synced=False):
        if signal_rx is None:
            signal_rx = self.get_data()
        return super().cal_ber(signal_rx, synced)

    def cal_evm(self, signal_rx=None, blind=False):
        if signal_rx is None:
            signal_rx = self.get_data()
        return super().cal_evm(signal_rx, blind)

    def cal_gmi(self, signal_rx=None):
        if signal_rx is None:
            signal_rx = self.get_data()
        return super().cal_gmi(signal_rx)

    def est_snr(self, signal_rx=None, blind=False):
        if signal_rx is None:
            signal_rx = self.get_data()
        return super().est_snr(signal_rx, synced)







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
            outsyms[i] = quantize(utils.normalise_and_center(signal[i]), self.gray_coded_symbols)
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
        if PRBS:
            if len(PRBSorder) < ndim:
                warnings.warn("PRBSorder is not given for all dimensions, picking random orders and seeds")
                PRBSorder_n = []
                PRBSseed_n = []
                orders = [15, 23]
                for i in range(ndim):
                    try:
                        PRBSorder_n.append(PRBSorder[i])
                        PRBSseed_n.append(PRBSseed[i])
                    except IndexError:
                        o = np.random.choice(orders)
                        PRBSorder_n.append(o)
                        s = np.random.randint(0, 2**o)
                        PRBSseed_n.append(s)
                PRBSorder = PRBSorder_n
                PRBSseed = PRBSseed_n
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


class PilotModulator(object):
    def __init__(self, M):
        self.mod_data = QAMModulator(M)
        self.mod_pilot = QAMModulator(4)

    def generate_signal(self, frame_len, pilot_seq_len, pilot_ins_rat, ndim, **kwargs):
        """
        Generate a pilot based symbol sequence

        Parameters
        ----------
        frame_len : integer
            length of the data frame without pilot sequence
        pilot_seq_len : integer
            length of the pilot sequence at the beginning of the frame
        pilot_ins_rat : integer
            phase pilot insertion ratio, if 0 or None do not insert phase pilots
        ndim : integer
            number of dimensions of the signal
        kwargs
            keyword arguments to pass to QAMmodulator.generate_signal

        Returns
        -------
        symbols : array_like
            data frame with pilots
        data : array_like
            data symbols
        pilots : array_like
            pilot sequence
        """
        if pilot_ins_rat is 0 or pilot_ins_rat is None:
            N_ph_frames = 0
        else:
            if (frame_len - pilot_seq_len)%pilot_ins_rat != 0:
                raise ValueError("Frame without pilot sequence divided by pilot rate needs to be an integer")
            N_ph_frames = (frame_len - pilot_seq_len)//pilot_ins_rat
        N_pilots = pilot_seq_len + N_ph_frames
        out_symbs = np.zeros([ndim, frame_len], dtype=complex)
        self.mod_pilot.generate_signal(N_pilots, None, ndim=ndim, **kwargs)
        out_symbs[:, :pilot_seq_len] = self.mod_pilot.symbols_tx[:, :pilot_seq_len]
        if N_ph_frames:
            if not pilot_ins_rat == 1:
                self.mod_data.generate_signal(N_ph_frames * (pilot_ins_rat -1), None, ndim=ndim, **kwargs)
                # Note that currently the phase pilots start one symbol after the sequence
                # TODO: we should probably fix this
                out_symbs[:, pilot_seq_len::pilot_ins_rat] = self.mod_pilot.symbols_tx[:, pilot_seq_len:]
                for j in range(N_pilots):
                    out_symbs[:, pilot_seq_len + j * pilot_ins_rat + 1:
                          pilot_seq_len + (j + 1) * pilot_ins_rat] = \
                        self.mod_data.symbols_tx[:, j * (pilot_ins_rat - 1):(j + 1) * (pilot_ins_rat - 1)]
            else:
                out_symbs[:, pilot_seq_len:] = self.mod_pilot.symbols_tx[:, pilot_seq_len:]
        else:
            self.mod_data.generate_signal(frame_len-pilot_seq_len, None, ndim=ndim, **kwargs)
            out_symbs[:, pilot_seq_len:] = self.mod_data.symbols_tx[:,:]
        self.symbols_tx = out_symbs
        self.pilot_ins_rat = pilot_ins_rat
        self.pilot_seq_len = pilot_seq_len
        if pilot_ins_rat == 1:
            return out_symbs, np.array([], dtype=complex), self.mod_pilot.symbols_tx
        return out_symbs, self.mod_data.symbols_tx, self.mod_pilot.symbols_tx

    @property
    def pilot_seq(self):
        return self.mod_pilot.symbols_tx[:, :self.pilot_seq_len]

    @property
    def ph_pilots(self):
        return self.mod_pilot.symbols_tx[:, self.pilot_seq_len::self.pilot_ins_rat]


class TDHQAMModulator(object):
    def __init__(self, M1, M2, fr, power_method="dist", snr=None):
        """
        Time-domain hybrid QAM (TDHQAM) modulator with two QAM-orders.

        Parameters
        ----------
        M1 : integer
            QAM order of the first part.
        M2 : integer
            QAM order of the second part
        fr : float
            fraction of the second format of the overall frame length
        power_method : string, optional
            method to calculate the power ratio of the different orders, currently on "dist" is implemented
        snr : float
            Design signal-to-noise ratio needed when using BER for calculation of the power ratio, currently does nothing
        """
        if power_method is "ber":
            assert snr is not None, "snr needs to be given to calculate the power ratio based on ber"
        self.M1 = M1
        self.M2 = M2
        self.fr = fr
        self.mod_M1 = QAMModulator(M1)
        if power_method is "dist":
            d1 = np.min(abs(np.diff(np.unique(self.mod_M1.symbols))))
            M2symbols = theory.cal_symbols_qam(M2)
            d2 = np.min(abs(np.diff(np.unique(M2symbols))))
            scf = (d2/d1)**2
            self.mod_M2 = QAMModulator(M2, scaling_factor=scf)
        else:
            raise NotImplementedError("no other methods are implemented yet")

    def generate_signal(self, N, ndim=1, **kwargs):
        """
        Generate a hybrid qam signal

        Parameters
        ----------
        N : integer
            length of the signal
        ndim : integer, optional
            number of dimensions (modes, polarizations)
        kwargs
            arguments to pass to the modulator signal generations

        Returns
        -------
        out : array_like
            hybrid qam signal
        """
        ratn = fractions.Fraction(self.fr).limit_denominator()
        f_M2 = ratn.numerator
        f_M = ratn.denominator
        f_M1 = f_M - f_M2
        frms = N//f_M
        frms_rem = N%f_M
        N1 = frms * f_M1
        N2 = frms * f_M2
        out = np.zeros((ndim, N), dtype=np.complex128)
        if frms_rem:
            mi1 = min(f_M1, frms_rem)
            N1 += mi1
            frms_rem -= mi1
            if frms_rem > 0:
                assert frms_rem < f_M2, "remaining symbols should be less than symbol 2 frames"
                N2 += frms_rem
        sig1, sym1, bits1, = self.mod_M1.generate_signal(N1, None, ndim=ndim, **kwargs)
        sig2, sym2, bits2, = self.mod_M2.generate_signal(N2, None, ndim=ndim, **kwargs)
        idx = np.arange(N)
        idx1 = idx%f_M < f_M1
        idx2 = idx%f_M >= f_M1
        out[:,idx1] = sym1
        out[:,idx2] = sym2
        self.symbols_tx = out
        return out




