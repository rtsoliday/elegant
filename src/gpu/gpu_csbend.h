#ifndef GPU_CSBEND_H
#define GPU_CSBEND_H

#include "gpu_base.h"

#ifdef __cplusplus
extern "C" {
#endif

long gpu_track_through_csbend(long n_part, void *csbend, double p_error,
                              double Po, double **accepted, double z_start,
                              double *sigmaDelta2, char *rootname,
                              void *maxamp, void *apContour,
                              void *apFileData, long iSlice, void *eptr);
long gpu_track_through_csbendCSR(long n_part, void *csbend, double p_error,
                                 double Po, double **accepted,
                                 double z_start, double z_end, void *charge,
                                 char *rootname, void *maxamp,
                                 void *apContour, void *apFileData,
                                 void *eptr);
long gpu_track_through_driftCSR(long np, void *csrDrift, double Po,
                                double **accepted, double zStart,
                                double revolutionLength, void *charge,
                                char *rootname);
long gpu_csr_csbend_wake_available(long nParticles, long nBins);
long gpu_compute_csbend_csr_wake(double *dGamma, double *T1, double *T2,
                                 const double *ctHist,
                                 const double *ctHistDeriv,
                                 const double *denom,
                                 long nParticles, long nBins,
                                 double CSRConstant,
                                 double dsSlice,
                                 double slippageLength13,
                                 double dct,
                                 long steadyState,
                                 long trapazoidIntegration,
                                 long diSlippage,
                                 long diSlippage4,
                                 long copyComponentArrays,
                                 long copyDGammaArray);
long gpu_csr_csbend_resident_available(void *csbend, long nParticles,
                                       long nBins);
long gpu_csr_csbend_resident_begin(double *beta0, long nParticles);
long gpu_track_csbend_csr_enter_simple(long nParticles, double pCentral,
                                       double coordinateSign,
                                       long edge1Effect, double e1,
                                       double psi1, double rhoActual);
long gpu_copy_csbend_csr_beta0(double *beta0, long nParticles);
long gpu_prepare_csbend_csr_histogram_device(double *lower, double *upper,
                                             double *binSize, long *bins,
                                             double expansionFactor,
                                             long nParticles, double Po);
long gpu_compute_csbend_csr_histogram_device(double *ctHist,
                                             long nParticles, long nBins,
                                             double ctLower, double dct);
long gpu_apply_csbend_csr_kick_device(long nParticles, long nBins,
                                      double ctLower, double dct,
                                      double Po, double rho0);
long gpu_track_csbend_csr_body_slice(void *csbend, long nParticles,
                                     double sliceLength, double rho0,
                                     double rhoActual, double Po);
long gpu_track_csbend_csr_finalize_simple(long nParticles, double pCentral,
                                          double coordinateSign,
                                          long edge2Effect, double e2,
                                          double psi2, double rhoActual);
long gpu_copy_csbend_csr_dgamma(double *dGamma, long nBins);
void gpu_clear_csr_wake_cpu_shadow(void);
long gpu_copy_csr_wake_cpu_shadow(double *dGamma, long nBins);
void gpu_exactDrift(long np, double length);
void gpu_addCorrectorRadiationKick();

#ifdef __cplusplus
}
#endif

#endif
