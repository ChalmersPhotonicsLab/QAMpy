from scipy.signal import sawtooth
import numpy as np
from mathfcts import gauss


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
 
def create_NRZQPSK(t, reprate, f_car=2.e3, N=10., fs=16.e3):
    Fn=fs/2.
    Ts=1./fs
    T=1./N
    td = np.arange(0,(N*T), Ts)
    data = np.transpose(np.sign(np.random.randn(N,1)))
    data1 = np.ones((T/Ts,1))*data
    data2 = data1[:]
    return data, data1, data2
