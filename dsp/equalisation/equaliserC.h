#include <complex.h>

double complex det_symbol(double complex *syms, unsigned int M, double complex value, double *distout);

double complex apply_filter(double complex *E, unsigned int Ntaps, double complex *wx, unsigned int pols, unsigned int L);

void update_filter(double complex *E, unsigned int Ntaps,  double mu, double complex err, double complex *wx, unsigned int pols, unsigned int L);
