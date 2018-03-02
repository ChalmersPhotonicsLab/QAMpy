#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

double complex det_symbol(double complex *syms, unsigned int M, double complex value, double *distout){
  int i;
  double complex det_sym;
  double dist, dist0;
  dist0 = 10000.;
  det_sym = 0.;
    for (i=0; i<M; i++){
      dist = cabs(syms[i]-value);
      if (dist < dist0){
        det_sym = syms[i];
        dist0 = dist;
      }
    }
    *distout = dist0;
    return det_sym;
 }

double complex apply_filter(double complex *E, unsigned int Ntaps, double complex *wx, unsigned int pols, unsigned int L){

  int j, k;
  double complex Xest;
  Xest = 0.;
  for (k=0; k<pols;k++){
    for (j=0; j<Ntaps;j++){
      Xest = Xest + *(wx+j+k*Ntaps) * *(E + j + k*L);
    }
  }
  return Xest;
}

void update_filter(double complex *E, unsigned int Ntaps,  double mu, double complex err, double complex *wx, unsigned int pols, unsigned int L){
  int j, k;
  for (k=0; k<pols;k++){
    for (j=0; j<Ntaps;j++){
      *(wx + j + k*Ntaps) = *(wx + j + k*Ntaps) - mu * err * conj(*(E+j+k*L));
    }
  }
}
