from . import core

def apply_filter(sig, wxy, method="pyx"):
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
    os = int(sig.fs/sig.fb)
    sig_out = core.equalisation.apply_filter(sig, os, wxy, method=method)
    return sig.recreate_from_np_array(sig_out, fs=sig.fb)

def equalise_signal(sig, mu, wxy=None, Ntaps=None, TrSyms=None, Niter=1, method="mcma", adaptive_stepsize=False,
                    avoid_cma_sing=True, **kwargs):
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

    Returns
    -------

    (wx, wy)    : tuple(array_like, array_like)
       equaliser taps for the x and y polarisation

    err       : array_like
       estimation error for x and y polarisation
    """
    os = int(sig.fs/sig.fb)
    try:
        syms = sig.coded_symbols
    except AttributeError:
        syms = None
    return core.equalisation.equalise_signal(sig, os, mu, sig.M, wxy=wxy, Ntaps=Ntaps, TrSyms=TrSyms, Niter=Niter, method=method,
                                adaptive_stepsize=adaptive_stepsize,  symbols=syms,
                                             avoid_cma_sing=avoid_cma_sing, **kwargs)

def dual_mode_equalisation(sig, mu, Ntaps, TrSyms=(None, None), Niter=(1, 1), methods=("mcma", "sbd"),
                           adaptive_stepsize=(False, False), avoid_cma_sing=(True, False), **kwargs):
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

    adaptive_stepsize : tuple(bool, bool)
        whether to adapt the step size upon training for each of the equaliser modes

    Returns
    -------

    sig_out   : SignalObject
        equalised signal X and Y polarisation

    (wx, wy)  : tuple(array_like, array_like)
       equaliser taps for the x and y polarisation

    (err1, err2)       : tuple(array_like, array_like)
       estimation error for x and y polarisation for each equaliser mode
    """
    os = int(sig.fs/sig.fb)
    try:
        syms = sig.coded_symbols
    except AttributeError:
        syms = None
    sig_out, wx, err = core.equalisation.dual_mode_equalisation(sig, os, mu, sig.M, Ntaps, TrSyms=TrSyms, methods=methods,
                                                       adaptive_stepsize=adaptive_stepsize, symbols=syms,
                                                                avoid_cma_sing=avoid_cma_sing, **kwargs)
    return sig.recreate_from_np_array(sig_out, fs=sig.fb), wx, err




