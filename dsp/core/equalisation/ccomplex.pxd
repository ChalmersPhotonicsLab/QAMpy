cdef extern from "complex.h" nogil:
    double complex conj(double complex)

cdef extern from "complex.h" nogil:
    float complex conjf(float complex)

cdef extern from "complex.h" nogil:
    double cabs(double complex)

cdef extern from "complex.h" nogil:
    float cabsf(float complex)

cdef extern from "complex.h" nogil:
    double cimag(double complex)

cdef extern from "complex.h" nogil:
    float cimagf(float complex)

cdef extern from "complex.h" nogil:
    double creal(double complex)

cdef extern from "complex.h" nogil:
    float crealf(float complex)

cdef  extern from "complex.h" nogil:
    float complex I
