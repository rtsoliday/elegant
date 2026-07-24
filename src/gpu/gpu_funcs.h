#ifndef GPU_FUNCS_H
#define GPU_FUNCS_H

#include "gpu_base.h"

#ifdef __cplusplus
extern "C" {
#endif

void gpu_offset_beam(long nToTrack, MALIGN *offset, double P_central);
void gpu_do_match_energy(long np, double *P_central, long change_beam);
void gpu_set_central_momentum(long np, double P_new, double *P_central);
void gpu_center_beam(CENTER *center, long np, long iPass, double p0);
void gpu_collect_trajectory_data(double *centroid, long n_part);
void gpu_compute_centroids(double *centroid, long n_part);
void gpu_matr_element_tracking(VMATRIX *M, MATR *matr, long np, double z);
void gpu_ematrix_element_tracking(VMATRIX *M, EMATRIX *matr, long np, double z, double *P_central);
long gpu_track_through_exact_corrector(long nParticles, void *element,
                                       double pCentral, double **accepted,
                                       double zStart);
long gpu_track_through_taper_aperture(long nParticles, void *element,
                                      double pCentral, double **accepted,
                                      double zStart);
long gpu_track_through_speedbump(long nParticles, void *element,
                                 double pCentral, double **accepted,
                                 double zStart);

#ifdef __cplusplus
}
#endif

#endif
