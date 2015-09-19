from __future__ import division
import numpy as np
#from .dsp import resample

# local imports
from pyOMAqt.dsp import resample, make_prbs_extXOR


def generateRandomQPSKData(N, snr, carrier_f, baudrate,
                           samplingrate, PRBS=True, orderI=15, orderQ=23, offset=None):
    # TODO: the SNR calculation is not correct yet
    if PRBS == True:
        seedI = np.random.randint(0, high=2, size=orderI)
        seedQ = np.random.randint(0, high=2, size=orderQ)
        dataI = make_prbs_extXOR(np.int(orderI), np.int(N), seedI)
        dataQ = make_prbs_extXOR(np.int(orderQ), np.int(N), seedQ)
    else:
        dataI = np.random.randint(0, high=2, size=N)
        dataQ = np.random.randint(0, high=2, size=N)
    data = 2*(dataI+1.j*dataQ-0.5-0.5j)
    # the below SNR calculations are still somewhat off, need to check this
    Es = np.sum(abs(data)**2)/len(data)
    #Eb = Es/2.
    data = data/np.sqrt(Es)
    sigma = np.sqrt(1/(2*snr))
    data += np.random.randn(len(data))*sigma*np.exp(1.j*np.pi*np.random.randn(len(data)))
    data = resample(baudrate, samplingrate, data)
    return data*np.exp(1.j*np.arange(len(data))*carrier_f)
