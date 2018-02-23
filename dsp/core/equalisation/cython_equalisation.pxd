ctypedef double complex complex128_t
ctypedef float complex complex64_t
ctypedef long double float128_t

ctypedef fused complexing:
    float complex
    double complex

cdef extern from "math.h":
    double sin(double)

