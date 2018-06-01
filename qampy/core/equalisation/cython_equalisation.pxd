cimport cython
ctypedef double complex complex128_t
ctypedef float complex complex64_t
ctypedef long double float128_t

ctypedef fused complexing:
    float complex
    double complex

cdef class ErrorFct:
    cpdef double complex calc_error(self, double complex Xest)
    cpdef float complex calc_errorf(self, float complex Xest)
cdef partition_value(cython.floating signal, double[:] partitions, double[:] codebook)
cdef complexing det_symbol(complexing[:] syms, int M, complexing value, cython.floating *dists) nogil
