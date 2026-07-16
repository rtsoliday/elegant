#ifndef GPU_POLYNOMIAL_SERIES_H
#define GPU_POLYNOMIAL_SERIES_H

#ifdef __cplusplus
extern "C" {
#endif

long gpu_polynomial_series_tracking(long nParticles, void *polynomialSeries,
                                    double pError, double pCentral,
                                    double **accepted, double zStart);

#ifdef __cplusplus
}
#endif

#endif
