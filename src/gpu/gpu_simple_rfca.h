#ifndef GPU_SIMPLE_RFCA_H
#define GPU_SIMPLE_RFCA_H

#include "gpu_base.h"

#ifdef __cplusplus
extern "C" {
#endif

double gpu_findFiducialTime(long np, double s0, double sOffset,
                            double p0, unsigned long mode);
void gpu_ramped_rf_cavity(long np, double P_central, double length,
                          double volt, double omega, double phase,
                          long driftOnly);
/* MODRF: RAMPRF's loop with the phase from t - tAve */
void gpu_modulated_rf_cavity(long np, double P_central, double length, double volt, double omega,
                             double phase, double tAve, long driftOnly);
/* each particle's s/(c beta_from_delta(p0, delta)), on the host in particle order */
const double *gpu_rf_particle_times(long np, double pCentral);
long gpu_trackRfCavityWithWakes(long np, RFCA *rfca, double **accepted,
                                double *P_central, double zEnd, long iPass,
                                RUN *run, CHARGE *charge, WAKE *wake,
                                TRWAKE *trwake, LSCKICK *LSCKick,
                                long wakesAtEnd);
long gpu_track_rfcw_lsc_kick_only(double **part, long np, LSCKICK *lsc,
                                  double Po, CHARGE *charge,
                                  double lengthScale,
                                  double dgammaOverGamma);
long gpu_track_through_rfcw(long np, RFCW *rfcw, double **accepted,
                            double *P_central, double zEnd,
                            RUN *run, long iPass, CHARGE *charge);

#ifdef __cplusplus
}
#endif

#endif
