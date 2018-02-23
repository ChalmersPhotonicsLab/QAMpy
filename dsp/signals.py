from __future__ import division
import numpy as np
import abc
import fractions
import warnings
from bitarray import bitarray

import dsp.helpers
from .core import resample, utils
from dsp import theory
from .core import ber_functions
from .core.prbs import make_prbs_extXOR
from .core.signal_quality import quantize, generate_bitmapping_mtx, estimate_snr, soft_l_value_demapper



class RandomBits(np.ndarray):
    def __new__(cls, N, nmodes=1, seed=None):
        R = np.random.RandomState(seed)
        bitsq = R.randint(0, high=2, size=(nmodes, N)).astype(np.bool)
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
                    s = np.random.randint(0, 2 ** o)
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

class SignalBase(np.ndarray):
    __metaclass__ = abc.ABCMeta
    _inheritbase_ = ["_fs", "_fb", "_M"]
    _inheritattr_ = []  # list of attributes names that should be inherited
    __array_priority__ = 1

    @staticmethod
    def _copy_inherits(objold, objnew):
        for attr in objold._inheritbase_:
            setattr(objnew, attr, getattr(objold, attr))
        for attr in objold._inheritattr_:
            setattr(objnew, attr, getattr(objold, attr))

    def __array_finalize__(self, obj):
        if obj is None: return
        for attr in self._inheritbase_:
            setattr(self, attr, getattr(obj, attr, None))
        for attr in self._inheritattr_:
            setattr(self, attr, getattr(obj, attr, None))
        if hasattr(obj, "_symbols"):
            s = getattr(obj, "_symbols")
            if s is None:
                self._symbols = obj
            else:
                self._symbols = obj._symbols

    def _signal_present(self, signal):
        if signal is None:
            return self
        else:
            return np.atleast_2d(signal)

    def recreate_from_np_array(self, arr, **kwargs):
        obj = arr.view(self.__class__)
        self._copy_inherits(self, obj)
        for k, v in kwargs.items():
            if "_"+k in self._inheritattr_:
                k = "_" + k
            if "_"+k in self._inheritbase_:
                k = "_" + k
            setattr(obj, k, v)
        return obj

    @classmethod
    def _resample_array(cls, arr, fnew, fold, fb, **kwargs):
        os = fnew / fold
        if np.isclose(os, 1):
            return arr.copy().view(cls)
        onew = np.empty((arr.shape[0], int(os * arr.shape[1])), dtype=arr.dtype)
        for i in range(arr.shape[0]):
            onew[i, :] = resample.rrcos_resample(arr[i], fold, fnew, Ts=1 / fb, **kwargs)
        onew = np.asarray(onew).view(cls)
        cls._copy_inherits(arr, onew)
        return onew

    def resample(self, fnew, **kwargs):
        out = self._resample_array(self, fnew, self.fs, self.fb, **kwargs)
        out._symbols = self._symbols.copy()
        out._fs = fnew
        return out

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
        return np.asarray(errs) / data_demod.shape[1]

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
        # TODO: need to rename decode to demodulate
        bits_demod = self.demodulate(syms_demod)
        tx_synced = self.demodulate(symbols_tx)
        errs = np.count_nonzero(tx_synced ^ bits_demod, axis=-1)
        return np.asarray(errs) / bits_demod.shape[1]

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
        return np.asarray(
            np.sqrt(np.mean(dsp.helpers.cabssquared(symbols_tx - signal_rx), axis=-1)))  # /np.mean(abs(self.symbols)**2))

    def est_snr(self, signal_rx=None, synced=False, symbols_tx=None):
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
        if symbols_tx is None:
            symbols_tx = self.symbols
        if not synced:
            symbols_tx, signal_rx = self._sync_and_adjust(symbols_tx, signal_rx)
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
        mm = np.sqrt(np.mean(np.abs(signal_rx) ** 2, axis=-1))
        signal_rx = signal_rx / mm[:, np.newaxis]
        tx, rx = self._sync_and_adjust(symbols_tx, signal_rx)
        snr = self.est_snr(rx, synced=True, symbols_tx=tx)
        bits = self.demodulate(self.quantize(tx)).astype(np.int)
        # For every mode present, calculate GMI based on SD-demapping
        for mode in range(nmodes):
            l_values = soft_l_value_demapper(rx[mode], self.M, snr[mode], self._bitmap_mtx)
            # GMI per bit
            for bit in range(self.Nbits):
                GMI_per_bit[mode, bit] = 1 - np.mean(
                    np.log2(1 + np.exp(((-1) ** bits[mode, bit::self.Nbits]) * l_values[bit::self.Nbits])))
            GMI[mode] = np.sum(GMI_per_bit[mode])
        return GMI, GMI_per_bit

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

class SignalQAMGrayCoded(SignalBase):
    _inheritattr_ = ["_symbols", "_bits", "_encoding", "_bitmap_mtx",  "_code",
                     "_coded_symbols" ]

    @staticmethod
    def _demodulate(symbols, encoding):
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
            bt.encode(encoding, symbols)
            return np.fromstring(bt.unpack(zero=b'\x00', one=b'\x01'), dtype=np.bool)
        bits = []
        for i in range(symbols.shape[0]):
            bt = bitarray()
            bt.encode(encoding, symbols[i])
            bits.append(np.fromstring(bt.unpack(zero=b'\x00', one=b'\x01'), dtype=np.bool))
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
        Nsym = data.shape[1] // bitspsym
        out = np.empty((nmodes, Nsym), dtype=dtype)
        N = data.shape[1] - data.shape[1] % bitspsym
        for i in range(nmodes):
            datab = bitarray()
            datab.pack(data[i, :N].tobytes())
            # the below is not really the fastest method but easy encoding/decoding is possible
            out[i, :] = np.fromstring(b''.join(datab.decode(encoding)), dtype=dtype)
        return out

    @classmethod
    def from_symbol_array(cls, symbs, M=None, fb=1, dtype=None):
        symbs = np.atleast_2d(symbs)
        if M is None:
            warnings.warn("no M given, estimating how mnay unique symbols are in array, this can cause errors")
            M = np.unique(symbs).shape[0]
        if dtype is None:
            dtype = symbs.dtype
        scale = np.sqrt(theory.cal_scaling_factor_qam(M)) / np.sqrt((abs(np.unique(symbs)) ** 2).mean())
        coded_symbols, graycode, encoding, bitmap_mtx = cls._generate_mapping(M, scale, dtype=dtype)
        out = np.empty_like(symbs).astype(dtype)
        for i in range(symbs.shape[0]):
            out[i] = quantize(symbs[i], coded_symbols)
        bits = cls._demodulate(out, encoding)
        obj = np.asarray(out).view(cls)
        obj._M = M
        obj._fb = fb
        obj._fs = fb
        obj._bits = bits
        obj._encoding = encoding
        obj._code = graycode
        obj._coded_symbols = coded_symbols
        obj._symbols = obj.copy()
        return obj

    @classmethod
    def from_bit_array(cls, bits, M, fb=1, dtype=np.complex128):
        arr = np.atleast_2d(bits)
        nbits = int(np.log2(M))
        if arr.shape[1] % nbits > 0:
            warnings.warn("Length of bits not divisible by log2(M) truncating")
            len = arr.shape[1] // nbits * nbits
            arr = arr[:, :len]
        scale = np.sqrt(theory.cal_scaling_factor_qam(M))
        coded_symbols, graycode, encoding, bitmap_mtx = cls._generate_mapping(M, scale, dtype=dtype)
        # out = []
        # for i in range(arr.shape[0]):
        #    out.append( cls._modulate(arr[i], encoding, M))
        out = cls._modulate(arr, encoding, M, dtype)
        # out = np.asarray(out)
        obj = np.asarray(out).view(cls)
        obj._M = M
        obj._fb = fb
        obj._fs = fb
        obj._bits = bits
        obj._encoding = encoding
        obj._code = graycode
        obj._coded_symbols = coded_symbols
        obj._symbols = obj.copy()
        return obj

    @classmethod
    def _generate_mapping(cls, M, scale, dtype=np.complex128):
        Nbits = np.log2(M)
        symbols = theory.cal_symbols_qam(M).astype(dtype)
        # check if this gives the correct mapping
        symbols /= scale
        _graycode = theory.gray_code_qam(M)
        coded_symbols = symbols[_graycode]
        bformat = "0%db" % Nbits
        encoding = dict([(symbols[i],
                          bitarray(format(_graycode[i], bformat)))
                         for i in range(len(_graycode))])
        bitmap_mtx = generate_bitmapping_mtx(coded_symbols, cls._demodulate(coded_symbols, encoding), M)
        return coded_symbols, _graycode, encoding, bitmap_mtx

    # using Randombits as default class because they are slightly faster
    def __new__(cls, M, N, nmodes=1, fb=1, bitclass=RandomBits, dtype=np.complex128, **kwargs):
        scale = np.sqrt(theory.cal_scaling_factor_qam(M))
        coded_symbols, _graycode, encoding, bitmap_mtx = cls._generate_mapping(M, scale, dtype=dtype)
        Nbits = int(N * np.log2(M))
        bits = bitclass(Nbits, nmodes=nmodes, **kwargs)
        obj = cls._modulate(bits, encoding, M, dtype=dtype)
        obj = obj.view(cls)
        obj._bitmap_mtx = bitmap_mtx
        obj._encoding = encoding
        obj._coded_symbols = coded_symbols
        obj._M = M
        obj._fb = fb
        obj._fs = fb
        obj._code = _graycode
        obj._bits = bits
        obj._symbols = obj.copy()
        return obj

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
            outsyms[i] = quantize(signal[i], self.coded_symbols)
        return outsyms

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
    def fs(self):
        return self._fs

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
        return self._modulate(data, self._encoding, self.M, dtype=self.dtype)

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


class SymbolOnlySignal(SignalQAMGrayCoded):
    _inheritattr_ = ["_symbols",  "_coded_symbols" ]

    def __new__(cls, M, N, symbols, nmodes=1, fb=1):
        coded_symbols = symbols
        obj = np.random.choice(symbols, (nmodes, N))
        obj = obj.view(cls)
        obj._coded_symbols = coded_symbols
        obj._M = M
        obj._fb = fb
        obj._fs = fb
        obj._symbols = obj.copy()
        return obj

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
            outsyms[i] = quantize(signal[i], self.coded_symbols)
        return outsyms

    @classmethod
    def from_symbol_array(cls, symbs, coded_symbols=None, fb=1):
        symbs = np.atleast_2d(symbs)
        if coded_symbols is None:
            coded_symbols = np.unique(symbs).flatten()
        # not sure if this is really necessary, but avoids numerical error issues
        out = np.empty_like(symbs)
        for i in range(symbs.shape[0]):
            out[i] = quantize(symbs[i], coded_symbols)
        obj = np.asarray(out).view(cls)
        M = coded_symbols.size
        obj._M = M
        obj._coded_symbols = coded_symbols
        obj._fb = fb
        obj._fs = fb
        obj._symbols = obj.copy()
        return obj

    @staticmethod
    def _demodulate(symbols, encoding):
        raise NotImplementedError("SymbolOnlySignal class does not have bits")

    @staticmethod
    def _modulate(data, encoding, M, dtype=np.complex128):
        raise NotImplementedError("SymbolOnlySignal class does not have bits")

    def demodulate(self, symbols):
        raise NotImplementedError("SymbolOnlySignal class does not have bits")

    def modulate(self, data):
        raise NotImplementedError("SymbolOnlySignal class does not have bits")

    @classmethod
    def from_bit_array(cls, bits, M, fb=1):
        raise NotImplementedError("SymbolOnlySignal class does not have bits")

    def cal_gmi(self, signal_rx=None):
        raise NotImplementedError("SymbolOnlySignal class does not have bits gmi calculation not possible")

    def cal_ber(self, signal_rx=None):
        raise NotImplementedError("SymbolOnlySignal class does not have bits ber calculation not possible")

    def est_snr(self, signal_rx=None):
        raise NotImplementedError("SymbolOnlySignal class does not have bits snr estimation not possible")

class ResampledQAM(SignalQAMGrayCoded):

    def __new__(cls, M, N, fb=1, fs=1, resamplekwargs={"beta": 0.1}, **kwargs):
        obj = super().__new__(cls, M, N, fb=fb, **kwargs)
        # TODO: check if we are not wasting memory here
        onew = cls._resample_array(obj, fs, fb, fb, **resamplekwargs)
        onew._fs = fs
        return onew

    @classmethod
    def from_symbol_array(cls, array, fs, **kwargs):
        onew = cls._resample_array(array, fs, array.fs, array.fb, **kwargs)
        onew._fs = fs
        return onew


#TODO: Currently Signal Quality functions do not work for TDHQAMSymbols
class TDHQAMSymbols(SignalBase):
    _inheritattr_ = ["_symbols_M1", "_symbols_M2", "_fr", "_symbols" ]

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
        idx1 = idx % f_M < f_M1
        idx2 = idx % f_M >= f_M1
        return idx, idx1, idx2

    def __new__(cls, M, N, fr=0.5, power_method="dist", snr=None, nmodes=1, fb=1,
                M1class=SignalQAMGrayCoded, M2class=SignalQAMGrayCoded, **kwargs):
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
        frms = N // f_M
        if N % f_M > 0:
            N = f_M * frms
            warnings.warn("length of overall pattern not divisable by number of frames, truncating to %d symbols" % N)
        N1 = frms * f_M1
        N2 = frms * f_M2
        out = np.zeros((nmodes, N), dtype=np.complex128)
        syms1 = M1class(M1, N1, nmodes=nmodes, fb=fb, **kwargs)
        syms2 = M2class(M2, N2, nmodes=nmodes, fb=fb, **kwargs)
        scale = cls.calculate_power_ratio(syms1.coded_symbols, syms2.coded_symbols, power_method)
        syms2 /= np.sqrt(scale)
        idx, idx1, idx2 = cls._cal_symbol_idx(N, f_M, f_M1)
        out[:, idx1] = syms1
        out[:, idx2] = syms2
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
        nframes = N // f_M
        if (nframes * f_M1 > N1) or (nframes * f_M2 > N2):
            warnings.warn("Need to truncate input arrays as ratio is not possible otherwise")
            nframes = min(N1 // f_M1, N2 // f_M2)
        N = nframes * f_M
        out = np.zeros((syms_M1.shape[0], N), dtype=syms_M1.dtype)
        idx, idx1, idx2 = cls._cal_symbol_idx(N, f_M, f_M1)
        out[:, idx1] = syms_M1
        out[:, idx2] = syms_M2
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
            scf = (d2 / d1) ** 2
            return scf
        else:
            raise NotImplementedError("Only 'dist' method is currently implemented")

    def _divide_signal_frame(self, signal):
        idx = np.arange(signal.shape[1])
        idx1 = idx[idx % self.f_M < self.f_M1]
        idx2 = idx[idx % self.f_M >= self.f_M1]
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
                    pmax_n = np.mean(abs(signal[i, (idx_m + j) % idx.max()]))
                    if pmax_n > pmax:
                        imax = j
                        pmax = pmax_n
                syms1[i, :] = signal[i, (idx1 + imax) % idx.max()]
                syms2[i, :] = signal[i, (idx2 + imax) % idx.max()]
            return self._symbols_M1.from_symbol_array(syms1, fb=self.fb, M=self.M[0]), \
                   self._symbols_M2.from_symbol_array(syms2, fb=self.fb, M=self.M[1])
        else:
            raise NotImplementedError("currently only 'dist' method is implemented")

    def _demodulate(self):
        raise NotImplementedError("Use demodulation of subclasses")

    def _modulate(self):
        raise NotImplementedError("Use modulation of subclasses")


class SignalWithPilots(SignalBase):
    _inheritattr_ = ["_pilots", "_symbols", "_frame_len", "_pilot_seq_len", "_nframes",
                     "_idx_dat"]

    @staticmethod
    def _cal_pilot_idx(frame_len, pilot_seq_len, pilot_ins_rat):
        idx = np.arange(frame_len)
        idx_pil_seq = idx < pilot_seq_len
        if pilot_ins_rat == 0 or pilot_ins_rat is None:
            idx_pil = idx_pil_seq
        else:
            if (frame_len - pilot_seq_len) % pilot_ins_rat != 0:
                raise ValueError("Frame without pilot sequence divided by pilot rate needs to be an integer")
            N_ph_frames = (frame_len - pilot_seq_len) // pilot_ins_rat
            idx_ph_pil = ((idx - pilot_seq_len) % pilot_ins_rat != 0) & (idx - pilot_seq_len > 0)
            idx_pil = ~idx_ph_pil  # ^ idx_pil_seq
        idx_dat = ~idx_pil
        return idx, idx_dat, idx_pil

    def __new__(cls, M, frame_len, pilot_seq_len, pilot_ins_rat, nframes=1, scale_pilots=1, fs=1,
                dataclass=SignalQAMGrayCoded, nmodes=1, resamplekw={"beta":0.1}, **kwargs):
        out_symbs = np.empty((nmodes, frame_len), dtype=np.complex128)
        idx, idx_dat, idx_pil = cls._cal_pilot_idx(frame_len, pilot_seq_len, pilot_ins_rat)
        pilots = SignalQAMGrayCoded(4, np.count_nonzero(idx_pil), nmodes=nmodes, **kwargs) * scale_pilots
        # Note that currently the phase pilots start one symbol after the sequence
        # TODO: we should probably fix this
        out_symbs[:, idx_pil] = pilots
        symbs = dataclass(M, np.count_nonzero(idx_dat), nmodes=nmodes, **kwargs)
        out_symbs[:, idx_dat] = symbs
        out_symbs = np.tile(out_symbs, nframes)
        fb = symbs.fb
        obj = cls._resample_array(out_symbs, fs, fold=fb, fb=fb, **resamplekw)
        #obj = out_symbs.view(cls)
        obj._fs = fs
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
        pilots = SignalQAMGrayCoded(4, np.count_nonzero(idx_pil), nmodes=nmodes, **pilot_kwargs) / np.sqrt(
            scale_pilots)
        out_symbs[:, idx_pil] = pilots
        out_symbs[:, idx_dat] = data[:, :Ndat]
        out_symbs = np.tile(out_symbs, nframes)
        obj = out_symbs.view(cls)
        obj._fs = data.fb
        obj._frame_len = frame_len
        obj._pilot_seq_len = pilot_seq_len
        obj._pilot_ins_rat = pilot_ins_rat
        obj._nframes = nframes
        obj._symbols = data[:, :Ndat].copy()
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
    def symbols(self):
        return self._symbols

    @property
    def nframes(self):
        return self._nframes

    @property
    def frame_len(self):
        return self._frame_len

    def get_data(self, shift_factors=None):
        if shift_factors is None:
            idx = np.tile(self._idx_dat, self.nframes)
            return self[:, idx]
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

    def est_snr(self, signal_rx=None, synced=False, symbols_tx=None):
        if signal_rx is None:
            signal_rx = self.get_data()
        return super().est_snr(signal_rx, synced, symbols_tx)
