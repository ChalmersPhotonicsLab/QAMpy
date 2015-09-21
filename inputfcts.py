from __future__ import division
import numpy as np
from scipy.special import erfc

# local imports
from dsp import resample, make_prbs_extXOR


def generateRandomQPSKData(N, snr, carrier_f=0, baudrate=1,
                           samplingrate=1, PRBS=True, orderI=15, orderQ=23, offset=None):
    """Generate a QPSK data signal array

    Parameters:
    ----------
        N :  int
           Length of the array
        snr: number
           Signal to Noise Ratio (Es/N0) in logarithmic units
        carrier_f: number
           carrier frequency, relative to the overall window, if not given it
           is set to 0 (baseband modulation)
        baudrate:  number
           symbol rate of the signal. This should be the real symbol rate, used
           for calculating the oversampling factor. If not given it is 1.
        samplingrate: number
           the rate at which the signal is sampled. Together with the baudrate
           this is used for calculating the oversampling factor. Default is 1.
        PRBS: bool
           If True the bits are generated as standard PRBS sequences, if False
           random bits are generated using numpy.random.randint.
           Default is True
        orderI: int
           The PRBS order i.e. the length 2**orderI of the PRBS to use
           for the in-phase channel. Default is 15.
        orderQ: int
           The PRBS order i.e. the length 2**orderI of the PRBS to use
           for the quadrature channel. Default is 23..
        offset:

    Returns
    -------
    signal, dataI, dataQ
        signal: ndarray
            Signal with noise of length N
        dataI: ndarray
            data array used for the in-phase channel
        dataQ: ndarray
            data array used for the quadrature channel
    """
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
    return outdata*np.exp(1.j*np.arange(len(data))*carrier_f), dataI, dataQ
