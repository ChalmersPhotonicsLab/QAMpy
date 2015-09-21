from __future__ import division
import numpy as np
from scipy.special import erfc

# local imports
from dsp import resample, make_prbs_extXOR


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
    data /= np.sqrt(2) # normalise Energy per symbol to 1
    noise = (np.random.randn(N)+1.j*np.random.randn(N))/np.sqrt(2) # sqrt(2) because N/2 = sigma
    outdata = data+noise*10**(-snr/20) #the 20 here is so we don't have to take the sqrt
    outdata = resample(baudrate, samplingrate, outdata)
    return outdata*np.exp(1.j*np.arange(len(data))*carrier_f), data
