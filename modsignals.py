from scipy.signal import sawtooth
import numpy as np
#from mathfcts import gauss


def create_random_RZOOK(t, reprate, t0):
    stmp = sawtooth(2*np.pi*reprate*t)
    field = np.sqrt(gauss(stmp,[1,0,t0])) + 0.j
    rtmp = np.random.random_integers(0,1,size=np.ceil((t.max()-t.min())*reprate))
    for i in range(len(rtmp)):
        field[np.where((abs(abs(t)-(i+1)/reprate)<=1)&(abs(t)-(i+1)/reprate<0))]*=rtmp[i]
    return field

def delayinterferometer(E, f, delay):
    sE = np.fft.fftshift(np.fft.fft(np.fft.fftshift(E)))
    sE *= np.exp(-1.j*(2*np.pi*f)*delay)
    EE = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(sE))) + E
    return EE

def create_random_NRZOOK(t, reprate):
    field = np.ones(len(t),dtype='complex')
    rtmp = np.random.random_integers(0,1,
            size=np.ceil((t.max()-t.min())*reprate))
    for i in range(len(rtmp)):
        field[np.where((abs(abs(t)-(i+1)/reprate)<=1)&(abs(t)-
            (i+1)/reprate<0))] *=rtmp[i]
    return field

def create_random_NRZDPSK(t, reprate):
    field = np.ones(len(t),dtype='complex')
    rtmp = np.random.random_integers(0,1,
            size=np.ceil((t.max()-t.min())*reprate))
    for i in range(len(rtmp)):
        field[np.where((abs(abs(t)-(i+1)/reprate)<=1)&(abs(t)-
            (i+1)/reprate<0))] *= np.exp(1.j*np.pi*rtmp[i])
    return field

def create_NRZQPSK(t, Nsymbols, f_carrier=0., rdn_seed=None):
    if len(t)%Nsymbols:
        raise Error("Length of t must be a multiple of the number of"
                " symbols")
    if len(t)/Nsymbols < 2:
        raise Error("We need at least two samples per symbol")
    if rdn_seed is not None:
        np.random.seed(rdn_seed)
    samplesperbit = len(t)/Nsymbols
    data = np.sign(np.random.randn(2*Nsymbols,1))
    dI = data[::2]
    dQ = data[1::2]
    bits_I = (dI>0)*1
    bits_Q = (dQ>0)*1
    data_I = np.dot(dI, np.ones((1,samplesperbit))).flatten()
    data_Q = np.dot(dQ, np.ones((1,samplesperbit))).flatten()
    return (data_I+1.j*data_Q)*np.exp(1.j*2*np.pi*f_carrier*t)/np.sqrt(2),(bits_I, bits_Q)

def calculate_qam_symbols(M):
    """Calculates the symbols of M-QAM"""
    x = np.linspace(-(2*np.sqrt(N)/2-1),2*np.sqrt(N)/2-1, np.sqrt(N))
    qam = np.mgrid[-(2*np.sqrt(N)/2-1):2*np.sqrt(N)/2-1:1.j*np.sqrt(N),
            -(2*np.sqrt(N)/2-1):2*np.sqrt(N)/2-1:1.j*np.sqrt(N)]
    return qam[0]+1.j*qam[1]



