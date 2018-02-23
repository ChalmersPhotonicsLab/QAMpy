cimport cython
ctypedef double complex complex128_t
ctypedef float complex complex64_t
ctypedef long double float128_t

ctypedef fused complexing:
    float complex
    double complex

cdef extern from "math.h":
    double sin(double)

cdef complexing det_symbol(complexing[:] syms, int M, complexing value, cython.floating *dists) nogil