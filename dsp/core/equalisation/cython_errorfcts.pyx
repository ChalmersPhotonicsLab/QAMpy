cimport cython
cimport numpy as np
from cmath cimport *
from ccomplex cimport *
from .cython_equalisation cimport complexing, det_symbol, complex64_t, \
    complex128_t, ErrorFct, partition_value
import numpy as np

#cdef class ErrorFct:
    #cpdef double complex calc_error(self, double complex Xest):
        #return 0
    #cpdef float complex calc_errorf(self, float complex Xest):
        #return 0

cdef class ErrorFctGenericDD_d(ErrorFct): #TODO: need to figure out how to change this one
    cdef double complex[:] symbols
    cdef public double dist
    cdef int N
    def __init__(self, double complex[:] symbols):
        self.symbols = symbols
        self.N = symbols.shape[0]

cdef class ErrorFctGenericDD_f(ErrorFct): #TODO: need to figure out how to change this one
    cdef public float complex[:] symbols
    cdef public float dist
    cdef int N
    def __init__(self, float complex[:] symbols):
        self.symbols = symbols
        self.N = symbols.shape[0]

cdef class ErrorFctSBD_d(ErrorFctGenericDD_d):
    cpdef double complex calc_error(self, double complex Xest):
        cdef double complex R
        R = det_symbol(self.symbols, self.N, Xest, &self.dist)
        return (Xest.real - R.real)*abs(R.real) + 1.j*(Xest.imag - R.imag)*abs(R.imag)

cdef class ErrorFctSBD_f(ErrorFctGenericDD_f):
    cpdef float complex calc_errorf(self, float complex Xest):
        cdef float complex R
        R = det_symbol(self.symbols, self.N, Xest, &self.dist)
        #return (Xest.real - R.real)*cabsf(R.real) + 1.j*(Xest.imag - R.imag)*acbs(R.imag)
        return (crealf(Xest) - crealf(R))*cabsf(R.real) + 1.j*(cimagf(Xest) - cimagf(R))*cabsf(R.imag)

cpdef ErrorFctSBD(complexing[:] symbols): # this is needed to work around bug with fused types and special functions in cython
    if complexing is complex64_t:
        return ErrorFctSBD_f(symbols)
    else:
        return ErrorFctSBD_d(symbols)

cdef class ErrorFctMDDMA_d(ErrorFctGenericDD_d):
    cpdef double complex calc_error(self, double complex Xest):
        cdef double complex R
        R = det_symbol(self.symbols, self.N, Xest, &self.dist)
        return (Xest.real**2 - R.real**2)*Xest.real + 1.j*(Xest.imag**2 - R.imag**2)*Xest.imag

cdef class ErrorFctMDDMA_f(ErrorFctGenericDD_f):
    cpdef float complex calc_errorf(self, float complex Xest):
        cdef float complex R
        R = det_symbol(self.symbols, self.N, Xest, &self.dist)
        return (Xest.real**2 - R.real**2)*Xest.real + 1.j*(Xest.imag**2 - R.imag**2)*Xest.imag

cpdef ErrorFctMDDMA(complexing[:] symbols): # this is needed to work around bug with fused types and special functions in cython
    if complexing is complex64_t:
        return ErrorFctMDDMA_f(symbols)
    else:
        return ErrorFctMDDMA_d(symbols)

cdef class ErrorFctDD_d(ErrorFctGenericDD_d):
    cpdef double complex calc_error(self, double complex Xest):
        cdef double complex R
        R = det_symbol(self.symbols, self.N, Xest, &self.dist)
        return Xest - R

cdef class ErrorFctDD_f(ErrorFctGenericDD_f):
    cpdef float complex calc_errorf(self, float complex Xest):
        cdef float complex R
        R = det_symbol(self.symbols, self.N, Xest, &self.dist)
        return Xest - R

cpdef ErrorFctDD(complexing[:] symbols): # this is needed to work around bug with fused types and special functions in cython
    if complexing is complex64_t:
        return ErrorFctDD_f(symbols)
    else:
        return ErrorFctDD_d(symbols)

cdef class ErrorFctMCMA(ErrorFct):
    cdef double R_real
    cdef double R_imag
    def __init__(self, double complex R):
        self.R_real = R.real
        self.R_imag = R.imag
    cpdef double complex calc_error(self, double complex Xest):
        return (creal(Xest)**2 - self.R_real)*creal(Xest) + 1j*(cimag(Xest)**2 -self.R_imag)*cimag(Xest)
    cpdef float complex calc_errorf(self, float complex Xest):
        return (crealf(Xest)**2 - self.R_real)*crealf(Xest) + 1j*(cimagf(Xest)**2 -self.R_imag)*cimagf(Xest)


cdef class ErrorFctCMA(ErrorFct):
    cdef double R
    def __init__(self, double R):
        self.R = R
    cpdef double complex calc_error(self, double complex Xest):
        return (creal(Xest)**2 + cimag(Xest)**2 - self.R)*Xest
    cpdef float complex calc_errorf(self, float complex Xest):
        return (crealf(Xest)**2 + cimagf(Xest)**2 - self.R)*Xest

cdef class ErrorFctRDE(ErrorFct):
    cdef double[:] partition
    cdef double[:] codebook
    def __init__(self, double[:] partition, double[:] codebook):
        self.partition = partition
        self.codebook = codebook
    cpdef double complex calc_error(self, double complex Xest):
        cdef double Ssq
        cdef double S_DD
        Ssq = Xest.real**2 + Xest.imag**2
        S_DD = partition_value(Ssq, self.partition, self.codebook)
        return Xest*(Ssq - S_DD)

cdef class ErrorFctMRDE(ErrorFct):
    cdef double[:] partition_real
    cdef double[:] partition_imag
    cdef double[:] codebook_real
    cdef double[:] codebook_imag
    def __init__(self, np.ndarray[ndim=1, dtype=double complex] partition,
                                              np.ndarray[ndim=1, dtype=double complex] codebook):
        self.partition_real = partition.real
        self.partition_imag = partition.imag
        self.codebook_real = codebook.real
        self.codebook_imag = codebook.imag
    cpdef double complex calc_error(self, double complex Xest):
        cdef double complex Ssq
        cdef double complex S_DD
        Ssq = creal(Xest)**2 + 1.j * cimag(Xest)**2
        S_DD = partition_value(Ssq.real, self.partition_real, self.codebook_real) + 1.j * partition_value(Ssq.imag, self.partition_imag, self.codebook_imag)
        return (Ssq.real - S_DD.real)*Xest.real + 1.j*(Ssq.imag - S_DD.imag)*Xest.imag


cdef class ErrorFctSCA(ErrorFct):
    cdef double complex R
    def __init__(self, double complex R):
        self.R = R
    cpdef double complex calc_error(self, double complex Xest):
        cdef int A
        cdef int B
        if abs(Xest.real) >= abs(Xest.imag):
            A = 1
            if abs(Xest.real) == abs(Xest.imag):
                B = 1
            else:
                B = 0
        else:
            A = 0
            B = 1
        return 4*Xest.real*(4*Xest.real**2 - 4*self.R**2)*A + 1.j*4*Xest.imag*(4*Xest.imag**2 - 4*self.R**2)*B

    cpdef float complex calc_errorf(self, float complex Xest):
        cdef int A
        cdef int B
        if abs(Xest.real) >= abs(Xest.imag):
            A = 1
            if abs(Xest.real) == abs(Xest.imag):
                B = 1
            else:
                B = 0
        else:
            A = 0
            B = 1
        return 4*Xest.real*(4*Xest.real**2 - 4*self.R**2)*A + 1.j*4*Xest.imag*(4*Xest.imag**2 - 4*self.R**2)*B

cdef class ErrorFctCME(ErrorFct):
    cdef double beta
    cdef double d
    cdef double R
    def __init__(self, double R, double d, double beta):
        self.beta = beta
        self.d = d
        self.R = R
    cpdef double complex calc_error(self, double complex Xest):
        return (abs(Xest)**2 - self.R)*Xest + self.beta * np.pi/(2*self.d) * (sin(Xest.real*np.pi/self.d) + 1.j * sin(Xest.imag*np.pi/self.d))

    cpdef float complex calc_errorf(self, float complex Xest):
        return (abs(Xest)**2 - self.R)*Xest + self.beta * np.pi/(2*self.d) * (sinf(Xest.real*np.pi/self.d) + 1.j * sin(Xest.imag*np.pi/self.d))
