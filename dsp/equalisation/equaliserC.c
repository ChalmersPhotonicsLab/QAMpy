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


void SBD_C(double complex *E, int TrSyms, int Ntaps,  unsigned int os, double mu, double complex *wx, double complex *symbols, unsigned int M, double complex *err, unsigned int pols){

  int i, j, k;
  double complex Xest, dsymb;
  for (i=0; i<TrSyms; i++){
      Xest = 0;
      for (j=0; j<Ntaps; j++) {
        for (k=0; k<pols; k++) {
          Xest = Xest + (wx+j*2)[k]*(E+(i*os+j)*pols)[k];
        }
      }
      //dsymb = det_symbol(symbols, M, Xest);
      err[i] = (creal(Xest) - creal(dsymb))*abs(creal(dsymb)) + I*(cimag(Xest) - cimag(dsymb)) * abs(cimag(dsymb));
      for(j=0; j<Ntaps; j++) {
        for(k=0; k<pols; k++) {
          (wx+j*pols)[k] = (wx+j*pols)[k]-mu*err[i]*Xest*conj((E+(i*os+j)*pols)[k]);
        }
      }
    }
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
