from __future__ import division, print_function
import numpy as np

def FS_CMA_training_2(TrSyms, Ntaps, os, mu, E, wx):
    err = np.zeros(TrSyms, dtype=np.float64)
    X = np.zeros((TrSyms,2), dtype=np.complex128)
    for i in range(0, int(TrSyms)):
        X = E[:, i*os:i*os+Ntaps]
        Xest = np.sum(wx*X)
        err[i] = abs(Xest)-1
        wx = wx-mu*err[i]*Xest*np.conj(X)
    return err, wx


