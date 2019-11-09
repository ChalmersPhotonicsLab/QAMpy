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

import numpy as np
from qampy import core
from qampy.core import equalisation, pilotbased_receiver
from qampy import phaserec
__doc__= equalisation.equalisation.__doc__

def apply_filter(sig, wxy, method="pyt"):
    """
    Apply the equaliser filter taps to the input signal.

    Parameters
    ----------

    sig      : SignalObject
        input signal to be equalised

    wxy    : tuple(array_like, array_like,optional)
        filter taps for the x and y polarisation

    method : basestring
        which apply filter method to use (pyx=cython, py=python)

    Returns
    -------

    sig_out   : SignalObject
        equalised signal
    """
    sig_out = core.equalisation.apply_filter(sig, sig.os, wxy, method=method)
    return sig.recreate_from_np_array(sig_out, fs=sig.fb)

def equalise_signal(sig, mu, wxy=None, Ntaps=None, TrSyms=None, Niter=1, method="mcma", adaptive_stepsize=False,
                    symbols=None, modes=None, apply=False, **kwargs):
    """
    Blind equalisation of PMD and residual dispersion, using a chosen equalisation method. The method can be any of the keys in the TRAINING_FCTS dictionary.

    Parameters
    ----------
    sig    : SignalObject
        single or dual polarisation signal field (2D complex array first dim is the polarisation)

    mu      : float
        step size parameter

    wxy     : array_like, optional
        the wx and wy filter taps. Either this or Ntaps has to be given.

    Ntaps   : int
        number of filter taps. Either this or wxy need to be given. If given taps are initialised as [00100]

    TrSyms  : int, optional
        number of symbols to use for filter estimation. Default is None which means use all symbols.

    Niter   : int, optional
        number of iterations. Default is one single iteration

    method  : string, optional
        equaliser method has to be one of cma, rde, mrde, mcma, sbd, mddma, sca, dd_adaptive, sbd_adaptive, mcma_adaptive

    adaptive_stepsize : bool, optional
        whether to use an adaptive stepsize or a fixed

    modes: array_like, optional
        array or list  of modes to  equalise over (default=None  equalise over all modes of the input signal)

    apply: Bool, optional
        whether to apply the filter taps and return the equalised signal

    Returns
    -------
    if apply:
        sig_out   : SignalObject
            equalised signal X and Y polarisation

    (wx, wy)    : tuple(array_like, array_like)
       equaliser taps for the x and y polarisation

    err       : array_like
       estimation error for x and y polarisation
    """
#    if method == "data" and symbols is None:
#        raise ValueError("DataAided equalization requires reference symbols!")
#    elif method == "data":
#        syms = symbols
#    else:
    if symbols is None:
        try:
            symbols = sig.coded_symbols
        except AttributeError:
            symbols = None
        
    if apply:
        sig_out, wxy, err = core.equalisation.equalise_signal(sig, sig.os, mu, sig.M, wxy=wxy, Ntaps=Ntaps, TrSyms=TrSyms, Niter=Niter, method=method,
                                                 adaptive_stepsize=adaptive_stepsize,  symbols=symbols,
                                                 avoid_cma_sing=avoid_cma_sing, apply=True, 
                                                              modes=modes, **kwargs)
        return sig.recreate_from_np_array(sig_out, fs=sig.fb), wxy, err
    else:
        return core.equalisation.equalise_signal(sig, sig.os, mu, sig.M, wxy=wxy, Ntaps=Ntaps, TrSyms=TrSyms, Niter=Niter, method=method,
                                adaptive_stepsize=adaptive_stepsize,  symbols=symbols, modes=modes,
                                             avoid_cma_sing=avoid_cma_sing, apply=False, **kwargs)

def dual_mode_equalisation(sig, mu, Ntaps, TrSyms=(None, None), Niter=(1, 1), methods=("mcma", "sbd"),
                           adaptive_stepsize=(False, False), symbols=None, modes=None, apply=True, **kwargs):
    """
    Blind equalisation of PMD and residual dispersion, with a dual mode approach. Typically this is done using a CMA type initial equaliser for pre-convergence and a decision directed equaliser as a second to improve MSE.


    Parameters
    ----------
    sig    : SignalObject
        single or dual polarisation signal field (2D subclass of SignalBase)

    mu      : tuple(float, float)
        step size parameter for the first and second equaliser method

    Ntaps   : int
        number of filter taps. Either this or wxy need to be given. If given taps are initialised as [00100]

    TrSyms  : tuple(int,int) optional
        number of symbols to use for filter estimation for each equaliser mode. Default is (None, None) which means use all symbols in both equaliser.

    Niter   : tuple(int, int), optional
        number of iterations for each equaliser. Default is one single iteration for both

    method  : tuple(string,string), optional
        equaliser method for the first and second mode has to be one of cma, rde, mrde, mcma, sbd, mddma, sca, dd_adaptive, sbd_adaptive, mcma_adaptive

    adaptive_stepsize : tuple(bool, bool), optional
        whether to adapt the step size upon training for each of the equaliser modes
        
    modes: array_like, optional
        array or list  of modes to  equalise over (default=None  equalise over all modes of the input signal)

    apply: Bool, optional
        whether to apply the filter taps and return the equalised signal


    Returns
    -------

    if apply:
        sig_out   : SignalObject
            equalised signal X and Y polarisation

    (wx, wy)  : tuple(array_like, array_like)
       equaliser taps for the x and y polarisation

    (err1, err2)       : tuple(array_like, array_like)
       estimation error for x and y polarisation for each equaliser mode

    if apply is False do not return sig_out

    """
    
    if symbols is None:
        try:
            symbols = sig.coded_symbols
        except AttributeError:
            symbols = None
    if apply:
        sig_out, wx, err = core.equalisation.dual_mode_equalisation(sig, sig.os, mu, sig.M, Ntaps, TrSyms=TrSyms, methods=methods,
                                                       adaptive_stepsize=adaptive_stepsize, symbols=symbols, Niter=Niter,
                                                                    modes=modes, apply=True,**kwargs)
        return sig.recreate_from_np_array(sig_out, fs=sig.fb), wx, err
    else:
        return core.equalisation.dual_mode_equalisation(sig, sig.os, mu, sig.M, Ntaps, TrSyms=TrSyms, methods=methods,
                                                        Niter=Niter, 
                                                       adaptive_stepsize=adaptive_stepsize, symbols=syms,
                                                                modes=modes, apply=False,**kwargs)



def pilot_equalizer(signal, mu, Ntaps, apply=True, foe_comp=True, verbose=False, **eqkwargs):
    """
    Pilot based equalisation 
    
    Parameters
    ----------
    signal : SignalObject
        Pilot-based signal object, has to be synced already
    mu : float
        Step size parameter
    Ntaps : int
        Number of equaliser taps
    apply : bool, optional
        Apply the filter to the signal
    foe_comp : bool, optional
        Do frequency offset compensation
    verbose : bool, optional
        Return verbose output
    **eqkwargs 
        Dictionary of values to pass to the equaliser functions

    Returns
    -------
    taps : array_like
        filter taps
    if apply also return
        out_sig : SignalObject
            equalised signal
    if verbose  also return
        foe_all : array_like
            estimated  frequency offset
        ntaps : tuple
            Tuple of equaliser and synchronization taps
    """

    if signal.shiftfctrs is None:
        raise ValueError("Stupid student, sync first")
    else:
        eq_shiftfctrs = np.array(signal.shiftfctrs,dtype=int)

    if (abs(Ntaps-signal.synctaps) % signal.os) != 0:
        raise ValueError("Tap difference need to be an integer of the oversampling")
    elif Ntaps != signal.synctaps:
        eq_shiftfctrs -= (Ntaps - signal.synctaps)//2


    taps_all, foe_all = pilotbased_receiver.equalize_pilot_sequence(signal, signal.pilot_seq, eq_shiftfctrs, os=signal.os, mu=mu,
                                                                    foe_comp=foe_comp, Ntaps = Ntaps, **eqkwargs)
    if foe_comp:
        out_sig = phaserec.comp_freq_offset(signal, foe_all)
    else:
        out_sig = signal
    if apply:
        if np.unique(signal.shiftfctrs).shape[0] > 1:
            eq_mode_sig = []
            for l in range(signal.shape[0]):
                eq_mode_sig.append(core.equalisation.apply_filter(out_sig[:,eq_shiftfctrs[l]:int(eq_shiftfctrs[l]+signal.frame_len*signal.os + Ntaps - 1)], signal.os, taps_all[l][None,:,:]))
            eq_mode_sig = signal.recreate_from_np_array(np.squeeze(np.array(eq_mode_sig)),fs=signal.fb)
        else:
            eq_mode_sig = apply_filter(out_sig[:,eq_shiftfctrs[0]:int(eq_shiftfctrs[0]+signal.frame_len*signal.os + Ntaps - 1)], np.array(taps_all))
        if verbose:
            return taps_all, eq_mode_sig, foe_all, (Ntaps, signal.synctaps)
        else:
            return taps_all, eq_mode_sig
    else:
        if verbose:
            taps_all, foe_all, (Ntaps, signal.synctaps)
        else:
            return taps_all
