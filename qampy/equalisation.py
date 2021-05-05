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

def _apply_to_pilotsignal(sig, wxy, frames):
    Ntaps = wxy.shape[-1]
    shiftfctrs = sig.shiftfctrs
    if Ntaps != sig.synctaps:
        shiftfctrs = shiftfctrs - (Ntaps - sig.synctaps)//2
    if np.min(shiftfctrs) < 0:
        shiftfctrs += sig.os*sig.frame_len
    assert shiftfctrs.max() + sig.os*sig.frame_len*(max(frames)+1) < sig.shape[-1] - (Ntaps - 1), \
        "Trying to equalise frame {}, but signal is not long enough".format(max(frames))
    if np.all(np.diff(frames) == 1):
        nframes = frames[-1] - frames[0] + 1
        if np.unique(shiftfctrs).shape[0] > 1:
            modes = np.arange(wxy.shape[0]).reshape(-1, sig.shape[0]).T #needed for real valued eqn
            eq_mode_sig = []
            for mode in modes:
                idx_0 = shiftfctrs[mode[0]] + frames[0]*sig.os*sig.frame_len
                idx_end = idx_0 + nframes*sig.frame_len*sig.os + Ntaps -1
                eq_mode_sig.append(core.equalisation.apply_filter(
                    sig[:, idx_0:idx_end], sig.os, wxy, modes=mode))
            return sig.recreate_from_np_array(np.squeeze(np.array(eq_mode_sig)), fs=sig.fb)
        else:
            idx_0 = shiftfctrs[0] + frames[0]*sig.os*sig.frame_len
            idx_end = idx_0 + nframes*sig.frame_len*sig.os + Ntaps -1
            return sig.recreate_from_np_array(core.equalisation.apply_filter(
                                    sig[:,idx_0:idx_end], sig.os, wxy), fs=sig.fb)
    else:
        if np.unique(shiftfctrs).shape[0] > 1:
            modes = np.arange(wxy.shape[0]).reshape(-1, sig.shape[0]).T #needed for real valued eqn
            all_mode_sig = []
            for frame in frames:
                eq_mode_sig = []
                for mode in modes:
                    idx_0 = shiftfctrs[mode[0]] + frame*sig.os*sig.frame_len
                    idx_end = idx_0 + sig.frame_len*sig.os + Ntaps -1
                    eq_mode_sig.append(core.equalisation.apply_filter(
                        sig[:, idx_0:idx_end], sig.os, wxy, modes=mode))
                all_mode_sig.append(np.squeeze(np.array(eq_mode_sig)))
            return sig.recreate_from_np_array(np.hstack(all_mode_sig), fs=sig.fb)
        else:
            all_mode_sig = []
            for frame in frames:
                idx_0 = shiftfctrs[0] + frame*sig.os*sig.frame_len
                idx_end = idx_0 + sig.frame_len*sig.os + Ntaps -1 
                all_mode_sig.append(core.equalisation.apply_filter(
                    sig[:,idx_0:idx_end], sig.os, wxy))
            return sig.recreate_from_np_array(np.hstack(all_mode_sig), fs=sig.fb)
   
def apply_filter(sig, wxy, method="pyt", frames=[0]):
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
        
    frames : list(int), optional
        if signal object is a pilot signal the frames to be equalised. If False,
        empty or None, do not perform a frame sync and apply filter to the whole signal.

    Returns
    -------

    sig_out   : SignalObject
        equalised signal
    """
    if hasattr(sig, "pilots") and frames: # pilot equaliser needs to be applied to the frame
        return _apply_to_pilotsignal(sig, wxy, frames)
    else:
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
        if method in core.equalisation.DATA_AIDED:
            symbols = sig.symbols
        else:
            try:
                symbols = sig.coded_symbols
            except AttributeError:
                symbols = None
    if apply:
        sig_out, wxy, err = core.equalisation.equalise_signal(sig, sig.os, mu, sig.M, wxy=wxy, Ntaps=Ntaps, TrSyms=TrSyms, Niter=Niter, method=method,
                                                 adaptive_stepsize=adaptive_stepsize,  symbols=symbols,
                                                 apply=True, modes=modes, **kwargs)
        return sig.recreate_from_np_array(sig_out, fs=sig.fb), wxy, err
    else:
        return core.equalisation.equalise_signal(sig, sig.os, mu, sig.M, wxy=wxy, Ntaps=Ntaps, TrSyms=TrSyms, Niter=Niter, method=method,
                                adaptive_stepsize=adaptive_stepsize,  symbols=symbols, modes=modes,
                                             apply=False, **kwargs)

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
        for method in methods:
            if method in core.equalisation.DATA_AIDED:
                symbols = sig.symbols
            else:
                try:
                    symbols = sig.coded_symbols
                except AttributeError:
                    symbols = None
    if apply:
        sig_out, wx, err = core.equalisation.dual_mode_equalisation(sig, sig.os, mu, sig.M, Ntaps=Ntaps, TrSyms=TrSyms, methods=methods,
                                                       adaptive_stepsize=adaptive_stepsize, symbols=symbols, Niter=Niter,
                                                                    modes=modes, apply=True,**kwargs)
        return sig.recreate_from_np_array(sig_out, fs=sig.fb), wx, err
    else:
        return core.equalisation.dual_mode_equalisation(sig, sig.os, mu, sig.M, Ntaps=Ntaps, TrSyms=TrSyms, methods=methods,
                                                        Niter=Niter, 
                                                       adaptive_stepsize=adaptive_stepsize, symbols=symbols,
                                                                modes=modes, apply=False,**kwargs)



def pilot_equaliser(signal, mu, Ntaps, apply=True, foe_comp=True, wxinit=None, frame=0, verbose=False, **eqkwargs):
    """
    Pilot based equalisation on a single frame

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
    wxinit : array_like, optional
        Initialisation taps for the equaliser. If this is given Ntaps will not be used
    frame : int, optional
        Which frame to equalise, frame numbers start at 0
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
    mu = np.atleast_1d(mu)
    if len(mu) == 1: # use the same mu for both equaliser steps
        mu = np.repeat(mu, 2)
    if wxinit is not None: # if we have init taps this determines the number of taps
        Ntaps = wxinit.shape[-1]
    if (abs(Ntaps-signal.synctaps) % 2) != 0:
        raise ValueError("Tap difference need to be an integer of the oversampling")
    elif Ntaps != signal.synctaps:
        eq_shiftfctrs = eq_shiftfctrs - (Ntaps - signal.synctaps)//2 + signal.os*signal.frame_len*frame
    assert signal.shape[-1] - eq_shiftfctrs.max() > signal.frame_len*signal.os, "You are trying to equalise an incomplete frame which does not work"
    
    taps_all, foe_all = pilotbased_receiver.equalize_pilot_sequence(signal, signal.pilot_seq, eq_shiftfctrs, os=signal.os, mu=mu,
                                                                    foe_comp=foe_comp, Ntaps = Ntaps, wxinit=wxinit, **eqkwargs)
    if foe_comp:
        out_sig = phaserec.comp_freq_offset(signal, foe_all)
    else:
        out_sig = signal
    if apply:
        eq_mode_sig = apply_filter(out_sig, taps_all, frames=[frame])
        if verbose:
            return taps_all, eq_mode_sig, foe_all, (Ntaps, signal.synctaps)
        else:
            return taps_all, eq_mode_sig
    else:
        if verbose:
            taps_all, foe_all, (Ntaps, signal.synctaps)
        else:
            return taps_all
        
def pilot_equaliser_nframes(signal, mu, Ntaps, apply=True, foe_comp=True, frames=[0], wxinit=None, verbose=True, **eqkwargs):
    """
    Pilot based equalisation  over multiple frames

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
    frames : array_like, optional
        List of frames to process the first frame is frame 0
    **eqkwargs
        Dictionary of values to pass to the equaliser functions

    Returns
    -------
    taps : array_like
        filter taps for each frame
    if apply also return
        out_sig : SignalObject
            equalised signal per frame
    if verbose  also return
        foe_all : array_like
            estimated  frequency offset per frame
        ntaps : tuple
            Tuple of equaliser and synchronization taps per frame
    """        
    if signal.shiftfctrs is None:
        raise ValueError("Stupid student, sync first")
    if frames is None:
        nframes = (signal.shape[-1] - np.max(signal.shiftfctrs))//(signal.os*signal.frame_len)
        frames = np.arange(nframes)
    frames = np.atleast_1d(frames)
    nframes = np.max(frames)
    assert signal.shape[-1] - (np.max(signal.shiftfctrs) + nframes*signal.frame_len*signal.os) > signal.frame_len*signal.os, "The last frame must be complete for equalisation"
    if wxinit is not None: # if we have init taps this determines the number of taps
        Ntaps = wxinit.shape[-1]
    rets= []
    for i in frames:
        ret = pilot_equaliser(signal, mu, Ntaps, apply=apply, foe_comp=foe_comp, wxinit=wxinit, verbose=verbose, frame=i, **eqkwargs)
        if i == 0:
            wxinit = ret[0]
        rets.append(ret)
    out = tuple(zip(*rets)) # return lists for taps, signals, foe ...
    if apply:
        # if we applied the arrays we want to return a single signal object
        sout = np.array(np.hstack(out[1])) # need to convert to array first to avoid an infinite recursion
        sout = signal.recreate_from_np_array(sout, fs=signal.fb)
        return out[0], sout, out[2:]
    else:
        return out
