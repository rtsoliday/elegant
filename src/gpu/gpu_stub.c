#include "mdb.h"
#include "../track.h"
#include "gpu_base.h"
#include "fftpackC.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#pragma weak compute_centroids
#pragma weak accumulate_beam_sums
#pragma weak limit_amplitudes
#pragma weak elimit_amplitudes
#pragma weak rectangular_collimator
#pragma weak elliptical_collimator
#pragma weak beam_scraper
#pragma weak removeInvalidParticles
#pragma weak imposeApertureData
#pragma weak trackRfCavityWithWakes
#pragma weak track_through_rfcw
#pragma weak findFiducialTime
#pragma weak do_match_energy
#pragma weak trackKickMap
#pragma weak trackUndulatorKickMap
#pragma weak initializeKickMap
#pragma weak initializeUndulatorKickMap
#pragma weak AddWigglerRadiationIntegrals
#pragma weak get_phase_reference
#pragma weak unused_phase_reference
#pragma weak set_phase_reference
#pragma weak getFiducializationPidRange
#pragma weak determineOpenSideCode
#pragma weak interpolateApertureData
#pragma weak convertLocalCoordinatesToGlobal
#pragma weak multipole_tracking
#pragma weak multipole_tracking2
#pragma weak determineDQCorReferenceFrameTilt
#pragma weak computeQuadSteeringStrengths
#pragma weak rotate_xy
#pragma weak multipoleKicksDone
#pragma weak track_through_csbend
#pragma weak track_through_ccbend
#ifdef GPU_VERIFY
#pragma weak gpu_verify_csrcsbend_cpu_body_slice
#endif
#pragma weak computeCSBENDFieldCoefficients
#pragma weak Fx_xy
#pragma weak Fy_xy
#pragma weak expansionOrder1
#pragma weak hasSkew
#pragma weak hasNormal
#pragma weak set_up_wake
#pragma weak set_up_trwake
#pragma weak set_up_impedance
#pragma weak set_up_cwake
#pragma weak track_through_lscdrift
#pragma weak computeTimeCoordinatesOnly
#pragma weak binTimeDistribution
#pragma weak binTransverseTimeDistribution
#pragma weak convolveArrays
#pragma weak applyLongitudinalWakeKicks
#pragma weak addLSCKick
#pragma weak rms_emittance
#pragma weak printWarningForTracking
#pragma weak SavitzkyGolaySmooth
#pragma weak rotateBeamCoordinatesForMisalignment
#pragma weak initialize_polynomialSeries
#pragma weak polynomialSeries_tracking
#pragma weak bmapxyz_field_setup

extern unsigned long multipoleKicksDone;
extern double **Fx_xy;
extern double **Fy_xy;
extern long expansionOrder1;
extern long hasSkew;
extern long hasNormal;
extern void set_up_wake(WAKE *wakeData, RUN *run, long pass, long particles,
                        CHARGE *charge);
extern void set_up_trwake(TRWAKE *wakeData, RUN *run, long pass,
                          long particles, CHARGE *charge);
extern void set_up_impedance(IMPEDANCE *impedanceData, RUN *run, long pass,
                             long particles, CHARGE *charge);
extern void set_up_cwake(CWAKE *wakeData, RUN *run, long pass,
                         long particles, CHARGE *charge);
extern void computeTimeCoordinatesOnly(double *time, double Po,
                                       double **part, long np);
extern long binTimeDistribution(double *Itime, long *pbin, double tmin,
                                double dt, long nb, double *time,
                                double **part, double Po, long np);
extern long binTransverseTimeDistribution(double **posItime, double *pz,
                                          long *pbin, double tmin,
                                          double dt, long nb, double *time,
                                          double **part, double Po, long np,
                                          double dx, double dy, long xPower,
                                          long yPower);
extern void convolveArrays(double *output, long outputs, double *a1, long n1,
                           double *a2, long n2, long di2);
extern void applyLongitudinalWakeKicks(double **part, double *time,
                                       long *pbin, long np, double Po,
                                       double *Vtime, long nb, double tmin,
                                       double dt, long interpolate);
extern double rms_emittance(double **coord, long i1, long i2, long n,
                            double *s11Return, double *s12Return,
                            double *s22Return, double *c1Return,
                            double *c2Return);
extern void printWarningForTracking(char *text, char *detail);
extern long SavitzkyGolaySmooth(double *data, long rows, long order,
                                long nLeft, long nRight,
                                long derivativeOrder);
extern void rotateBeamCoordinatesForMisalignment(double **part, long np,
                                                 double angle);
extern void initialize_polynomialSeries(POLYNOMIALSERIES *polynomialSeries);
extern void bmapxyz_field_setup(BMAPXYZ *bmapxyz);
extern long polynomialSeries_tracking(
  double **particle, long nPart, POLYNOMIALSERIES *polynomialSeries,
  double pError, double pCentral, double **accepted, double zStart);
extern long trackRfCavityWithWakes(double **part, long np, RFCA *rfca,
                                   double **accepted, double *P_central,
                                   double zEnd, long iPass, RUN *run,
                                   CHARGE *charge, WAKE *wake,
                                   TRWAKE *trwake, LSCKICK *LSCKick,
                                   long wakesAtEnd);
extern double findFiducialTime(double **part, long np, double s0,
                               double sOffset, double p0,
                               unsigned long mode);
extern void do_match_energy(double **coord, long np, double *P_central,
                            long change_beam);
extern long trackKickMap(double **particle, double **accepted, long nParticles,
                         double pRef, KICKMAP *map, double zStart,
                         double *sigmaDelta2);
extern long trackUndulatorKickMap(double **particle, double **accepted,
                                  long nParticles, double pRef,
                                  UKICKMAP *map, double zStart);
extern void initializeKickMap(KICKMAP *map);
extern void initializeUndulatorKickMap(UKICKMAP *map);
extern void AddWigglerRadiationIntegrals(double length, long periods,
                                         double radius, double eta,
                                         double etap, double beta,
                                         double alpha, double *I1,
                                         double *I2, double *I3,
                                         double *I4, double *I5);
extern long get_phase_reference(double *phase, long phase_ref_number);
extern void computeCSBENDFieldCoefficients(double *b, double *c,
                                           double h1, long nonlinear,
                                           long expansionOrder);
extern long track_through_csbend(double **part, long n_part, CSBEND *csbend,
                                 double p_error, double Po,
                                 double **accepted, double z_start,
                                 double *sigmaDelta2, char *rootname,
                                 MAXAMP *maxamp, APCONTOUR *apContour,
                                 APERTURE_DATA *apFileData, long iSlice,
                                 ELEMENT_LIST *eptr);
#ifdef GPU_VERIFY
extern long gpu_verify_csrcsbend_cpu_body_slice(double *coord,
                                                CSRCSBEND *csbend,
                                                double beta0,
                                                double sliceLength,
                                                double rho0,
                                                double rhoActual,
                                                double Po);
#endif
extern long multipole_tracking(double **particle, long n_part, MULT *multipole,
                               double p_error, double Po, double **accepted,
                               double z_start);
extern long track_through_csbendCSR_cuda_resident_entry(
  double **part, long n_part, CSRCSBEND *csbend, double p_error,
  double Po, double **accepted, double z_start, double z_end,
  CHARGE *charge, char *rootname, MAXAMP *maxamp, APCONTOUR *apContour,
  APERTURE_DATA *apFileData, ELEMENT_LIST *eptr) __attribute__((weak));

extern int gpuCudaRuntimeGetDeviceCount(int *count);
extern int gpuCudaRuntimeSetDevice(int device);
extern const char *gpuCudaRuntimeGetErrorString(int code);
extern int gpuCudaRuntimeGetDeviceName(int device, char *name, unsigned long nameSize);
extern int gpuCudaMallocDouble(void **ptr, unsigned long count);
extern int gpuCudaMallocBytes(void **ptr, unsigned long bytes);
extern int gpuCudaFree(void *ptr);
extern int gpuCudaCopyHostToDevice(void *dst, const void *src, unsigned long count, float *milliseconds);
extern int gpuCudaCopyDeviceToHost(void *dst, const void *src, unsigned long count, float *milliseconds);
extern int gpuCudaCopyDeviceBytesToHost(void *dst, const void *src, unsigned long bytes, float *milliseconds);
extern int gpuCudaStableScatterRows(void *coord, void *scratchCoord,
                                    const void *prefix, long nParticles,
                                    int stride, long survivors,
                                    float *milliseconds);
extern int gpuCudaTrackParticles(void *coord, long nParticles, int stride,
                                 const GPU_MATRIX_DATA *matrix, float *milliseconds);
extern int gpuCudaExactDrift(void *coord, long nParticles, int stride,
                             double length, float *milliseconds);
extern int gpuCudaLinearDrift(void *coord, long nParticles, int stride,
                              double length, float *milliseconds);
extern int gpuCudaOffsetBeam(void *coord, long nParticles, int stride,
                             double dx, double dxp, double dy, double dyp,
                             double dz, double dt, double dp, double de,
                             double pCentral, long startPID, long endPID,
                             int allParticles, double cMks, float *milliseconds);
extern int gpuCudaApplyBatchedMomentumSearch(
  void *coord, long nParticles, int stride, const void *searchData,
  long searchParticles, long target, long pass, long firePass,
  void *history, void *historyCount, long turns,
  double dx, double dy, double pCentral, double cMks,
  float *milliseconds);
extern int gpuCudaClearBatchedSearchHistory(void *history,
                                            unsigned long historyCount,
                                            void *turnCount,
                                            unsigned long particles);
extern int gpuCudaSetCentralMomentum(void *coord, long nParticles, int stride,
                                     double oldP, double newP, float *milliseconds);
extern int gpuCudaMatchEnergy(void *coord, long nParticles, int stride,
                              double oldP, double averageP, int changeBeam,
                              float *milliseconds);
extern int gpuCudaMatchEnergyAndAverage(void *coord, long nParticles,
                                        int stride, double oldP,
                                        int changeBeam,
                                        GPU_BEAM_SUM_DATA *result,
                                        void *deviceScratch,
                                        float *milliseconds);
extern unsigned long gpuCudaMatchEnergyScratchBytes(void);
extern int gpuCudaCenteredBeamSums(void *coord, long nParticles, int stride,
                                   double pCentral, double cMks,
                                   const double *centroid,
                                   GPU_BEAM_SUM_DATA *result,
                                   float *milliseconds);
extern int gpuCudaLscStatistics(void *coord, long nParticles, int stride,
                                double pCentral, double cMks,
                                GPU_BEAM_SUM_DATA *result,
                                void *deviceResultScratch,
                                float *milliseconds);
extern int gpuCudaLscTransverseSums(void *coord, long nParticles, int stride,
                                    double xCentroid, double yCentroid,
                                    GPU_BEAM_SUM_DATA *result,
                                    void *deviceResultScratch,
                                    float *milliseconds);
extern int gpuCudaRfcaThinKick(void *coord, long nParticles, int stride,
                               double pCentral, double volt, double omega,
                               double phase, double cMks,
                               float *milliseconds);
extern int gpuCudaRfdfTrack(void *coord, long nParticles, int stride,
                            const GPU_RFDF_DATA *data,
                            int particleIdIndex, float *milliseconds);
extern int gpuCudaSreffectsTrack(void *coord, long nParticles, int stride,
                                 const GPU_SREFFECTS_DATA *data,
                                 float *milliseconds);
extern int gpuCudaBggexpTrack(void *coord, long nParticles, int stride,
                              const GPU_BGGEXP_DATA *data,
                              float *milliseconds);
extern void gpuCudaBggexpRelease(void);
extern int gpuCudaCwigglerTrack(void *coord, long nParticles, int stride,
                                const GPU_CWIGGLER_DATA *data,
                                float *milliseconds);
extern int gpuCudaFtableTrack(void *coord, long nParticles, int stride,
                              const GPU_FTABLE_DATA *data,
                              float *milliseconds);
extern void gpuCudaFtableRelease(void);
extern int gpuCudaBmxyzTrack(void *coord, long nParticles, int stride,
                             const GPU_BMXYZ_DATA *data, long *failedCount,
                             float *milliseconds);
extern void gpuCudaBmxyzRelease(void);
extern int gpuCudaRfcwRfOnlyMatrix(void *coord, long nParticles, int stride,
                                   double pCentral, double length,
                                   double volt, double omega, double phase,
                                   int end1Focus, int end2Focus,
                                   double dx, double dy, double cMks,
                                   float *milliseconds);
extern int gpuCudaRfcwRfOnlyMatrixChecked(
  void *coord, long nParticles, int stride, double pCentral, double length,
  double volt, double omega, double phase, int end1Focus, int end2Focus,
  double dx, double dy, double cMks, void *deviceLostCount,
  long *lostCount, float *milliseconds);
extern int gpuCudaRfcwKickInitial(void *coord, void *inverseF,
                                  long nParticles, int stride,
                                  double pCentral, double length,
                                  double volt, double omega, double phase,
                                  int end1Focus, double cMks,
                                  float *milliseconds);
extern int gpuCudaRfcwKickFinal(void *coord, const void *inverseF,
                                long nParticles, int stride,
                                double length, int end2Focus,
                                float *milliseconds);
extern int gpuCudaRfcwDgammaOverGammaSums(void *coord, long nParticles,
                                          int stride, double pCentral,
                                          double length, double volt,
                                          double omega, double phase,
                                          double cMks,
                                          GPU_BEAM_SUM_DATA *result,
                                          float *milliseconds);
extern int gpuCudaSubtractCoordinate(void *coord, long nParticles, int stride,
                                     int index, double value,
                                     float *milliseconds);
extern int gpuCudaKickMapTrackChecked(void *coord, long nParticles, int stride,
                                      const GPU_KICKMAP_DATA *data,
                                      const double *xpFactor,
                                      const double *ypFactor, long points,
                                      long *lostCount, float *milliseconds);
extern int gpuCudaKickMapTrackStableCompact(
  void *coord, void *scratchCoord, void *prefix, long nParticles, int stride,
  const GPU_KICKMAP_DATA *data, const double *xpFactor,
  const double *ypFactor, long points, double zStart, double pRef,
  long *remaining, float *milliseconds);
extern int gpuCudaCenterBeam(void *coord, long nParticles, int stride,
                             unsigned int coordinateMask, const double *offset,
                             int doTime, double pCentral, double timeOffset,
                             double cMks, float *milliseconds);
extern int gpuCudaCentroidSums(void *coord, long nParticles, int stride,
                               GPU_BEAM_SUM_DATA *result, float *milliseconds);
extern int gpuCudaTimeSums(void *coord, long nParticles, int stride,
                           double pCentral, double cMks,
                           GPU_BEAM_SUM_DATA *result, float *milliseconds);
extern int gpuCudaSelectedTimeSums(void *coord, long nParticles, int stride,
                                   double pCentral, double cMks,
                                   int bunchColumn, long selectedBunch,
                                   GPU_BEAM_SUM_DATA *result,
                                   float *milliseconds);
extern int gpuCudaFiducialTimeSums(void *coord, long nParticles, int stride,
                                   double pCentral, double sOffset,
                                   double cMks, int particleIdColumn,
                                   long startPID, long endPID,
                                   GPU_BEAM_SUM_DATA *result,
                                   float *milliseconds);
extern int gpuCudaFiducialPmaximum(void *coord, long nParticles, int stride,
                                   double pCentral, double sOffset,
                                   double cMks, int particleIdColumn,
                                   long startPID, long endPID,
                                   GPU_BEAM_SUM_DATA *result,
                                   float *milliseconds);
extern int gpuCudaFiducialFirst(void *coord, long nParticles, int stride,
                                double pCentral, double sOffset,
                                double cMks, int particleIdColumn,
                                long startPID, long endPID,
                                GPU_BEAM_SUM_DATA *result,
                                float *milliseconds);
extern int gpuCudaCentroidTimeSums(void *coord, long nParticles, int stride,
                                   double pCentral, double cMks,
                                   GPU_BEAM_SUM_DATA *result, float *milliseconds);
extern int gpuCudaBeamSums(void *coord, long nParticles, int stride,
                           double pCentral, double cMks,
                           GPU_BEAM_SUM_DATA *result, float *milliseconds);
extern unsigned long gpuCudaBeamSums2ScratchBytes(void);
extern int gpuCudaBeamSums2(void *coord, long nParticles, int stride,
                            double pCentral, double cMks,
                            GPU_BEAM_SUM_DATA *result,
                            GPU_BEAM_SUM_DATA *centeredResult,
                            void *deviceScratch, float *milliseconds);
extern int gpuCudaLongMinMax(void *coord, long nParticles, int stride,
                             int coordinateIndex,
                             GPU_LONG_MIN_MAX_DATA *result,
                             float *milliseconds);
extern int gpuCudaSortedBunchRanges(
  void *coord, long nParticles, int stride, int coordinateIndex,
  long minBunch, long nBuckets, long *start, long *count, long *sorted,
  float *milliseconds);
extern int gpuCudaDoubleMinMax(void *coord, long nParticles, int stride,
                               int coordinateIndex,
                               GPU_DOUBLE_MIN_MAX_DATA *result,
                               void *deviceResult,
                               float *milliseconds);
extern int gpuCudaLimitAmplitudeLossCount(void *coord, long nParticles, int stride,
                                          double xmax, double ymax,
                                          long *lostCount, float *milliseconds);
extern int gpuCudaLimitAmplitudesStableCompact(void *coord, void *scratchCoord,
                                               void *prefix, long nParticles,
                                               int stride, double xmax,
                                               double ymax, double z,
                                               double pCentral,
                                               long extrapolateZ,
                                               long *remaining,
                                               float *milliseconds);
extern int gpuCudaELimitAmplitudeLossCount(void *coord, long nParticles, int stride,
                                           double xmax, double ymax,
                                           long exponent, long yExponent,
                                           long *lostCount, float *milliseconds);
extern int gpuCudaELimitAmplitudesStableCompact(void *coord, void *scratchCoord,
                                                void *prefix, long nParticles,
                                                int stride, double xmax,
                                                double ymax, long exponent,
                                                long yExponent, double z,
                                                double pCentral,
                                                long extrapolateZ,
                                                long *remaining,
                                                float *milliseconds);
extern int gpuCudaRemoveInvalidParticlesLossCount(void *coord, long nParticles,
                                                  int stride, long *lostCount,
                                                  float *milliseconds);
extern int gpuCudaRemoveInvalidParticlesStableCompact(void *coord,
                                                      void *scratchCoord,
                                                      void *prefix,
                                                      long nParticles,
                                                      int stride, double z,
                                                      double pCentral,
                                                      long *remaining,
                                                      float *milliseconds);
extern int gpuCudaRectangularCollimatorLossCount(void *coord, long nParticles, int stride,
                                                 double xmax, double ymax,
                                                 double xCenter, double yCenter,
                                                 double length, long openCode, long *lostCount,
                                                 float *milliseconds);
extern int gpuCudaRectangularCollimatorStableCompact(void *coord, void *scratchCoord,
                                                     void *prefix, long nParticles,
                                                     int stride, double xmax,
                                                     double ymax, double xCenter,
                                                     double yCenter, double length,
                                                     long openCode,
                                                     double z, double pCentral,
                                                     long *remaining,
                                                     float *milliseconds);
extern int gpuCudaEllipticalCollimatorLossCount(void *coord, long nParticles, int stride,
                                                double xmax, double ymax,
                                                double xCenter, double yCenter,
                                                long exponent, long yExponent,
                                                double length,
                                                long openCode, long *lostCount,
                                                float *milliseconds);
extern int gpuCudaEllipticalCollimatorStableCompact(void *coord, void *scratchCoord,
                                                    void *prefix, long nParticles,
                                                    int stride, double xmax,
                                                    double ymax, double xCenter,
                                                    double yCenter, long exponent,
                                                    long yExponent,
                                                    double length, long openCode,
                                                    double z, double pCentral,
                                                    long *remaining,
                                                    float *milliseconds);
extern int gpuCudaScraperLossCount(void *coord, long nParticles, int stride,
                                   int plane, double center, double position,
                                   int sideSign, double length, long *lostCount,
                                   float *milliseconds);
extern int gpuCudaScraperStableCompact(void *coord, void *scratchCoord,
                                       void *prefix, long nParticles,
                                       int stride, int plane, double center,
                                       double position, int sideSign,
                                       double length, double z,
                                       double pCentral, long *remaining,
                                       float *milliseconds);
extern int gpuCudaApertureDataLossCount(void *coord, long nParticles,
                                        int stride, double xCenter,
                                        double yCenter, double xSize,
                                        double ySize, long *lostCount,
                                        float *milliseconds);
extern int gpuCudaApertureDataStableCompact(void *coord, void *scratchCoord,
                                            void *prefix, long nParticles,
                                            int stride, double xCenter,
                                            double yCenter, double xSize,
                                            double ySize, double z,
                                            double pCentral, long *remaining,
                                            float *milliseconds);
extern int gpuCudaMultipoleTrack(void *coord, long nParticles, int stride,
                                 const GPU_MULTIPOLE_DATA *multipole,
                                 int writeOutput, long *lostCount,
                                 float *milliseconds);
extern int gpuCudaMultipoleTrackChecked(void *coord, long nParticles, int stride,
                                        const GPU_MULTIPOLE_DATA *multipole,
                                        long *lostCount, float *milliseconds);
extern int gpuCudaMultipoleTrackStableCompact(void *coord, void *scratchCoord,
                                              void *prefix, long nParticles,
                                              int stride,
                                              const GPU_MULTIPOLE_DATA *multipole,
                                              long *remaining,
                                              float *milliseconds);
extern int gpuCudaCsbendTrackChecked(void *coord, long nParticles, int stride,
                                     const GPU_CSBEND_DATA *csbend,
                                     long *lostCount, float *milliseconds);
extern int gpuCudaCsbendTrackStableCompact(void *coord, void *scratchCoord,
                                           void *prefix, long nParticles,
                                           int stride,
                                           const GPU_CSBEND_DATA *csbend,
                                           long *remaining,
                                           float *milliseconds);
extern int gpuCudaCcbendTrackChecked(void *coord, long nParticles, int stride,
                                     const GPU_CCBEND_DATA *ccbend,
                                     long *lostCount, float *milliseconds);
extern int gpuCudaCsrCsbendBodySliceChecked(void *coord, long nParticles,
                                            int stride,
                                            const GPU_CSBEND_DATA *csbend,
                                            const double *beta0,
                                            void *backup,
                                            void *deviceLostCount,
                                            long *lostCount,
                                            float *milliseconds);
extern int gpuCudaCsrCsbendEnterSimpleChecked(void *coord, long nParticles,
                                              int stride, double pCentral,
                                              double coordinateSign,
                                              int edge1Effect,
                                              double e1,
                                              double psi1,
                                              double rhoActual,
                                              double *beta0,
                                              void *backup,
                                              void *deviceLostCount,
                                              long *lostCount,
                                              float *milliseconds);
extern int gpuCudaCsrCsbendFinalizeSimpleChecked(void *coord, long nParticles,
                                                 int stride, double pCentral,
                                                 double coordinateSign,
                                                 int edge2Effect,
                                                 double e2,
                                                 double psi2,
                                                 double rhoActual,
                                                 void *backup,
                                                 void *deviceLostCount,
                                                 long *lostCount,
                                                 float *milliseconds);
extern int gpuCudaWakeLongitudinalTrack(void *coord, long nParticles,
                                        int stride,
                                        const GPU_WAKE_LONGITUDINAL_DATA *wake,
                                        const double *wakeTable,
                                        long *binnedCount,
                                        double *itimeReturn,
                                        double *vtimeReturn,
                                        float *milliseconds);
extern int gpuCudaWakeLongitudinalHistogram(void *coord, long nParticles,
                                            int stride,
                                            const GPU_WAKE_LONGITUDINAL_DATA *wake,
                                            long *binnedCount,
                                            double *itimeReturn,
                                            float *milliseconds);
extern int gpuCudaWakeLongitudinalTrackFromHistogram(
  void *coord, long nParticles, int stride,
  const GPU_WAKE_LONGITUDINAL_DATA *wake, const double *wakeTable,
  const double *itimeInput, long *binnedCount,
  double *vtimeReturn, float *milliseconds);
extern int gpuCudaTrwakeTrack(void *coord, long nParticles, int stride,
                              const GPU_TRWAKE_DATA *wake,
                              const double *wakeTableX,
                              const double *wakeTableY,
                              long *binnedCount,
                              double *posItimeXReturn,
                              double *posItimeYReturn,
                              double *vtimeXReturn,
                              double *vtimeYReturn,
                              float *milliseconds);
extern int gpuCudaTrwakeHistogram(void *coord, long nParticles, int stride,
                                  const GPU_TRWAKE_DATA *wake,
                                  long *binnedCount,
                                  double *posItimeXReturn,
                                  double *posItimeYReturn,
                                  float *milliseconds);
extern int gpuCudaTrwakeTrackFromHistogram(
  void *coord, long nParticles, int stride, const GPU_TRWAKE_DATA *wake,
  const double *wakeTableX, const double *wakeTableY,
  const double *posItimeXInput, const double *posItimeYInput,
  long *binnedCount, double *vtimeXReturn, double *vtimeYReturn,
  float *milliseconds);
extern int gpuCudaCombinedWakeTrack(
  void *coord, long nParticles, int stride,
  const GPU_COMBINED_WAKE_DATA *wake,
  const double *const *tables, long *binnedCount,
  double *histogramReturn, double *voltageReturn,
  float *milliseconds);
extern void gpuCudaCombinedWakeRelease(void);
extern int gpuCudaRfmodeHistogram(
  void *coord, long nParticles, int stride, const GPU_RFMODE_DATA *data,
  unsigned long long *histogramReturn, long *binnedCount,
  float *milliseconds);
extern int gpuCudaRfmodeTimeCoordinates(
  void *coord, long nParticles, int stride, double pCentral, double cMks,
  double *timeReturn, float *kernelMilliseconds,
  float *transferMilliseconds);
extern int gpuCudaRfmodeApplyKicks(
  void *coord, long nParticles, int stride, const GPU_RFMODE_DATA *data,
  const double *voltage, float *milliseconds);
extern void gpuCudaRfmodeRelease(void);
extern int gpuCudaLscBin(void *coord, long nParticles, int stride,
                         const GPU_LSC_DATA *lsc, long *binnedCount,
                         double *itimeReturn, void *deviceItimeScratch,
                         void *deviceBinnedCountScratch,
                         float *milliseconds);
extern int gpuCudaLscApplyKickAndDrift(void *coord, long nParticles,
                                       int stride,
                                       const GPU_LSC_DATA *lsc,
                                       const double *vtime,
                                       void *deviceVtimeScratch,
                                       float *milliseconds);
extern int gpuCudaScmultLinearKick(void *coord, long nParticles, int stride,
                                   const GPU_SCMULT_LINEAR_DATA *data,
                                   float *milliseconds);
extern int gpuCudaScmultNonlinearKick(void *coord, long nParticles, int stride,
                                      const GPU_SCMULT_LINEAR_DATA *data,
                                      float *milliseconds);
extern int gpuCudaPolynomialSeriesTrack(
  void *coord, long nParticles, int stride,
  const GPU_POLYNOMIAL_SERIES_DATA *data, const double *coefficient,
  const int32_t *exponent, const void *owner, long *invalidCount,
  float *milliseconds);
extern void gpuCudaPolynomialSeriesRelease(void);
extern int gpuCudaCsrCsbendWake(const double *ctHist,
                                const double *ctHistDeriv,
                                const double *denom,
                                double *T1,
                                double *T2,
                                double *dGamma,
                                long nBins,
                                double CSRConstant,
                                double dsSlice,
                                double slippageLength13,
                                double dct,
                                long steadyState,
                                long trapazoidIntegration,
                                long diSlippage,
                                long diSlippage4,
                                float *milliseconds);
extern int gpuCudaCsrCsbendKickInPlace(void *coord,
                                       long nParticles,
                                       int stride,
                                       const double *dGamma,
                                       long nBins,
                                       double ctLower,
                                       double dct,
                                       double Po,
                                       double rho0,
                                       float *milliseconds);
extern int gpuCudaCsrHistogram(void *coord,
                               long nParticles,
                               int stride,
                               long coordinateIndex,
                               double lower,
                               double binSize,
                               long bins,
                               double *deviceHist,
                               double *histReturn,
                               float *milliseconds);

typedef struct GPU_CSR_SCRATCH {
  void *ctHist;
  void *wakeInput;
  void *denom;
  void *T1;
  void *T2;
  void *dGamma;
  void *kickDp;
  void *bodyBackup;
  void *bodyLostCount;
  void *rangeResult;
  double *hostWakeInput;
  double *cpuDGamma;
  long capacity;
  long kickDpCapacity;
  long bodyBackupCapacity;
  long bodyBackupStride;
  long hostWakeInputCapacity;
  long cpuDGammaCapacity;
  long cpuDGammaBins;
  long cpuDGammaValid;
  long dGammaBins;
  long dGammaValid;
  long denomBins;
  long denomValid;
  double denomDct;
} GPU_CSR_SCRATCH;

typedef struct GPU_APERTURE_SCRATCH {
  void *coord;
  void *prefix;
  long *hostPrefix;
  double *hostAccepted;
  long capacity;
  long prefixCapacity;
  long stride;
  long hostPrefixCapacity;
  long hostAcceptedCapacity;
} GPU_APERTURE_SCRATCH;

typedef struct GPU_ACCEPTED_BUFFER {
  void *coord;
  void *scratch;
  double **hostAccepted;
  void *hostBase;
  long capacity;
  long scratchCapacity;
  long stride;
  long nParticles;
  long deviceCurrent;
  long hostCurrent;
} GPU_ACCEPTED_BUFFER;

typedef struct GPU_KICKMAP_CACHE {
  void *xpFactor;
  void *ypFactor;
  const double *hostXpFactor;
  const double *hostYpFactor;
  long points;
} GPU_KICKMAP_CACHE;

typedef struct GPU_RFCW_KICK_SCRATCH {
  void *inverseF;
  long capacity;
} GPU_RFCW_KICK_SCRATCH;

typedef struct GPU_LSC_SCRATCH {
  void *result;
  void *itime;
  void *binnedCount;
  void *vtime;
  long binsCapacity;
} GPU_LSC_SCRATCH;

typedef struct GPU_BEAM_SUMS_SCRATCH {
  void *data;
} GPU_BEAM_SUMS_SCRATCH;

typedef struct GPU_RFCA_SCRATCH {
  void *lostCount;
  void *matchEnergy;
} GPU_RFCA_SCRATCH;

typedef struct GPU_BUNCH_RANGE_CACHE {
  const void *deviceCoord;
  long particles;
  long stride;
  long coordinateIndex;
  long minBunch;
  long maxBunch;
  long nBuckets;
  long *start;
  long *count;
  long valid;
} GPU_BUNCH_RANGE_CACHE;

static GPU_BASE gpuBase;
/* Used for small deterministic CPU confirmation tracks after a GPU-batched
 * search.  The surrounding do_tracking call still initializes normal GPU
 * bookkeeping, but no element is dispatched to CUDA while this is set. */
static long gpuTrackingSuppressed = 0;
static GPU_CSR_SCRATCH gpuCsrScratch;
static GPU_APERTURE_SCRATCH gpuApertureScratch;
static GPU_ACCEPTED_BUFFER gpuAcceptedBuffer;
static GPU_KICKMAP_CACHE gpuKickMapCache;
static GPU_RFCW_KICK_SCRATCH gpuRfcwKickScratch;
static GPU_LSC_SCRATCH gpuLscScratch;
static GPU_BEAM_SUMS_SCRATCH gpuBeamSumsScratch;
static GPU_RFCA_SCRATCH gpuRfcaScratch;
static GPU_BUNCH_RANGE_CACHE gpuBunchRangeCache;
typedef struct GPU_POLYNOMIAL_SERIES_CACHE {
  POLYNOMIALSERIES *owner;
  double *coefficient;
  int32_t *exponent;
  long totalTerms;
} GPU_POLYNOMIAL_SERIES_CACHE;
static GPU_POLYNOMIAL_SERIES_CACHE gpuPolynomialSeriesCache;
typedef struct GPU_BATCHED_SEARCH_SCRATCH {
  double *deviceData;
  double *deviceHistory;
  double *deviceHistoryCount;
  double *hostData;
  double *hostHistory;
  double *hostHistoryCount;
  long capacity;
  long historyCapacity;
  long particles;
  long turns;
  long firePass;
  long configured;
  long uploaded;
} GPU_BATCHED_SEARCH_SCRATCH;
static GPU_BATCHED_SEARCH_SCRATCH gpuBatchedSearchScratch;
static long gpuVerbose = 0;
static long gpuEnableExactDrift = 0;
static long gpuExactDriftExplicit = 0;
static long gpuEnableApertureParallelCompaction = 0;
static long gpuApertureParallelCompactionExplicit = 0;
static long gpuApertureParallelCompactionVerifyDisabled = 0;
static long gpuApertureParallelCompactionOrderDisabled = 0;
static long gpuEnableApertureAcceptedDevice = 0;
static long gpuEnableMagnetLossCompaction = 0;
static long gpuMagnetLossCompactionExplicit = 0;
static long gpuEnableCsbendDrift = 0;
static long gpuCsbendDriftExplicit = 0;
static long gpuDeviceIslandHasCsbend = 0;
static long gpuEnableCsrTracking = 0;
static long gpuCsrTrackingExplicit = 0;
static long gpuEnableCsrResident = 0;
static long gpuCsrResidentExplicit = 0;
static long gpuEnableScmult = 0;
static long gpuScmultExplicit = 0;
static long gpuEnableWakeTrackingDrift = 0;
static long gpuWakeTrackingDriftExplicit = 0;
static long gpuEnableCombinedWake = 0;
static long gpuCombinedWakeExplicit = 0;
static long gpuEnableCombinedWakeMultibunch = 0;
static long gpuEnableCombinedWakeFft = 0;
static long gpuEnableBatchedTuneTracking = 0;
static long gpuBatchedTuneMinParticles = 32;
static long gpuEnableBatchedSearchTracking = 0;
static long gpuBatchedSearchMinParticles = 32;
static long gpuEnablePolynomialSeries = 0;
static long gpuEnableRfdf = 0;
static long gpuRfdfMinParticles = 64;
static long gpuEnableSreffects = 0;
static long gpuSreffectsMinParticles = 64;
static long gpuEnableBggexp = 0;
static long gpuBggexpMinParticles = 64;
static long gpuEnableCwiggler = 0;
static long gpuCwigglerMinParticles = 64;
static long gpuEnableFtable = 0;
static long gpuFtableMinParticles = 64;
static long gpuEnableBmxyz = 0;
static long gpuBmxyzMinParticles = 64;
static long gpuEnableCcbend = 0;
static long gpuCcbendMinParticles = 64;
static long gpuEnableRfmode = 0;
static long gpuEnableFrfmode = 0;
static long gpuRfmodeMinParticles = 8192;
static long gpuEnableLscTracking = 0;
static long gpuLscTrackingExplicit = 0;
static long gpuEnableRfcwTrackingDrift = 0;
static long gpuRfcwTrackingDriftExplicit = 0;
#ifdef GPU_VERIFY
static long gpuCpuVerificationActive = 0;
#endif
static long gpuEnableRfcaChangeP0Drift = 0;
static long gpuRfcaChangeP0DriftExplicit = 0;
static long gpuRunAlwaysChangeP0 = 0;
static long gpuAvoidShortGpuIslands = 1;
static long gpuShortGpuIslandMaxElements = 4;
static long gpuMatrixDriftMinParticlesExplicit = 0;
static long gpuHelperMinParticlesExplicit = 0;
static long gpuMagnetMinParticlesExplicit = 0;
static long gpuWakeMinParticlesExplicit = 0;
static long gpuOutputDriftReductionMinParticlesExplicit = 0;
static long unsupportedReported = 0;
static double gpuWallStart = 0;
static double gpuPendingExactDriftLength = 0;
static long gpuPendingExactDriftCount = 0;
static long gpuPendingExactDriftParticles = 0;
static double gpuSavedCentroid[7];
static long gpuSavedCentroidValid = 0;
static BEAM_SUMS gpuSavedSums;
static BEAM_SUMS2 gpuSavedSums2;
static SPIN_SUMS gpuSavedSpinSums;
static long gpuSavedSumsValid = 0;
static unsigned long long *gpuRfmodeHostHistogram;
static long gpuRfmodeHostHistogramCapacity;
static double *gpuRfmodeHostTime;
static long gpuRfmodeHostTimeCapacity;

static long gpuPackMatrix(GPU_MATRIX_DATA *packed, VMATRIX *M);
static long gpuMatrixSupported(VMATRIX *M);
static void gpuReleaseKickMapCache(void);
static void gpuEnsureRfcwKickScratch(long nParticles);
static void gpuEnsureLscScratch(long bins);
static void gpuEnsureBeamSumsScratch(void);
static void gpuEnsureRfcaScratch(void);
static void gpuReleasePolynomialSeriesCache(void);
static void gpuRfcwApplyCoordinateOffset(long np, int index, double value,
                                         const char *operation);
static long gpuPassiveElementSupported(ELEMENT_LIST *eptr, long nParticles);
static long gpuElementEligible(ELEMENT_LIST *eptr, long nParticles);
static long gpuShouldUseCpuForShortGpuIsland(ELEMENT_LIST *eptr, long nParticles);
static void gpuFlushPendingExactDrift(const char *reason);
static long gpuCsrCsbendDeviceEntrySupported(ELEMENT_LIST *eptr, long nParticles);
static long gpuMultipoleElementSupported(ELEMENT_LIST *eptr);
static long gpuCsbendElementSupported(ELEMENT_LIST *eptr);
static long gpuCcbendElementSupported(ELEMENT_LIST *eptr);
static long gpuWakeElementSupported(ELEMENT_LIST *eptr);
static long gpuTrwakeElementSupported(ELEMENT_LIST *eptr);
static long gpuCombinedWakeElementSupported(ELEMENT_LIST *eptr);
static long gpuLscElementSupported(ELEMENT_LIST *eptr);
long gpu_csr_csbend_resident_available(void *csbend0, long nParticles,
                                       long nBins);
static void gpuWakeTrackingWarning(const char *text, const char *detail);
static void gpuSmoothWakeHistogram(double *data, long nb, long order,
                                   long halfWidth, const char *elementType,
                                   const char *inputFile);
static double gpuLscVarianceFromSums(const GPU_BEAM_SUM_DATA *sums, long coordinate);
static void gpuDisplaySyncTimings(void);
#ifdef GPU_VERIFY
static void gpuCompareWakeArray(const char *label, const char *arrayName,
                                const double *cpu, const double *gpu, long n);
static long gpuComputeCsrHistogramCpu(double *hist, double **part,
                                      long nParticles, long nBins,
                                      double ctLower, double dct);
#endif
void gpu_compute_centroids(double *centroid, long n_part);

static double wallSeconds(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec / 1000000.0;
}

static void gpuRecordWallSeconds(void) {
  double now;

  if (gpuWallStart <= 0)
    return;
  now = wallSeconds();
  if (now >= gpuWallStart)
    gpuBase.gpuWallSeconds += now - gpuWallStart;
  gpuWallStart = 0;
}

static long gpuEnvFlag(const char *name) {
  const char *value = getenv(name);

  if (!value)
    return 0;
  return strcmp(value, "1") == 0 || strcmp(value, "yes") == 0 ||
         strcmp(value, "true") == 0 || strcmp(value, "on") == 0 ||
         strcmp(value, "YES") == 0 || strcmp(value, "TRUE") == 0 ||
         strcmp(value, "ON") == 0;
}

static long gpuEnvLong(const char *name, long defaultValue) {
  char *endptr = NULL;
  const char *value = getenv(name);
  long result;

  if (!value || !*value)
    return defaultValue;
  result = strtol(value, &endptr, 10);
  if (endptr == value)
    return defaultValue;
  return result;
}

static double gpuEnvDouble(const char *name, double defaultValue) {
  char *endptr = NULL;
  const char *value = getenv(name);
  double result;

  if (!value || !*value)
    return defaultValue;
  result = strtod(value, &endptr);
  if (endptr == value)
    return defaultValue;
  return result;
}

static long gpuEnvSet(const char *name) {
  const char *value = getenv(name);

  return value && *value;
}

static long gpuEnvSetEither(const char *name, const char *legacyName) {
  return gpuEnvSet(name) || gpuEnvSet(legacyName);
}

static long gpuEnvLongEither(const char *name, const char *legacyName,
                             long defaultValue) {
  if (gpuEnvSet(name))
    return gpuEnvLong(name, defaultValue);
  return gpuEnvLong(legacyName, defaultValue);
}

static long gpuReasonContains(const char *reason, const char *fragment) {
  return reason && fragment && strstr(reason, fragment) != NULL;
}

static const char *gpuElementTypeName(ELEMENT_LIST *eptr) {
  if (!eptr || eptr->type < 0 || eptr->type >= N_TYPES || !entity_name[eptr->type])
    return "unknown";
  return entity_name[eptr->type];
}

static void gpuForceCpuForElement(ELEMENT_LIST *eptr, const char *reason) {
  char buffer[512];

  if (!reason)
    reason = "CPU element after CUDA element";
  if (!eptr) {
    snprintf(buffer, sizeof(buffer), "%s", reason);
  } else {
    snprintf(buffer, sizeof(buffer), "%s: %s %s#%ld",
             reason, gpuElementTypeName(eptr), eptr->name ? eptr->name : "?",
             eptr->occurence);
  }
  forceParticlesToCpu(buffer);
}

static void gpuRecordSyncRequest(const char *reason, long copied, long readOnly) {
  gpuBase.gpuSyncRequestCount++;
  if (copied)
    gpuBase.gpuSyncCopyCount++;
  if (readOnly)
    gpuBase.gpuSyncReadOnlyCount++;
  else
    gpuBase.gpuSyncMutableCount++;

  if (!reason || !*reason) {
    gpuBase.gpuSyncOtherCount++;
    return;
  }

  if (gpuReasonContains(reason, "gpuBaseDealloc")) {
    gpuBase.gpuSyncDeallocCount++;
  } else if (gpuReasonContains(reason, "VERIFY") ||
             gpuReasonContains(reason, "verify") ||
             gpuReasonContains(reason, "verification") ||
             gpuReasonContains(reason, "compareGpuCpu")) {
    gpuBase.gpuSyncVerificationCount++;
  } else if (gpuReasonContains(reason, "MPI") ||
             gpuReasonContains(reason, "scatter") ||
             gpuReasonContains(reason, "gather") ||
             gpuReasonContains(reason, "rank")) {
    gpuBase.gpuSyncMpiCount++;
  } else if (gpuReasonContains(reason, "output") ||
             gpuReasonContains(reason, "Output") ||
             gpuReasonContains(reason, "watch") ||
             gpuReasonContains(reason, "WATCH") ||
             gpuReasonContains(reason, "diagnostic") ||
             gpuReasonContains(reason, "histogram") ||
             gpuReasonContains(reason, "fitpoint") ||
             gpuReasonContains(reason, "dump_")) {
    gpuBase.gpuSyncOutputCount++;
  } else if (gpuReasonContains(reason, "loss fallback") ||
             gpuReasonContains(reason, "aperture") ||
             gpuReasonContains(reason, "Aperture") ||
             gpuReasonContains(reason, "collimator") ||
             gpuReasonContains(reason, "Collimator") ||
             gpuReasonContains(reason, "scraper") ||
             gpuReasonContains(reason, "limit_amplitudes") ||
             gpuReasonContains(reason, "elimit_amplitudes") ||
             gpuReasonContains(reason, "stopTrackingParticleLimit")) {
    gpuBase.gpuSyncApertureLossCount++;
  } else if (gpuReasonContains(reason, "wake") ||
             gpuReasonContains(reason, "Wake") ||
             gpuReasonContains(reason, "CSR") ||
             gpuReasonContains(reason, "csr") ||
             gpuReasonContains(reason, "LSC") ||
             gpuReasonContains(reason, "lsc") ||
             gpuReasonContains(reason, "SCMULT") ||
             gpuReasonContains(reason, "space charge") ||
             gpuReasonContains(reason, "collective")) {
    gpuBase.gpuSyncCollectiveCount++;
  } else if (gpuReasonContains(reason, "accumulate_beam_sums") ||
             gpuReasonContains(reason, "compute_centroids") ||
             gpuReasonContains(reason, "centroid") ||
             gpuReasonContains(reason, "reduction") ||
             gpuReasonContains(reason, "beam_sums")) {
    gpuBase.gpuSyncReductionCount++;
  } else if (gpuReasonContains(reason, "CPU element after CUDA element") ||
             gpuReasonContains(reason, "unsupported matrix") ||
             gpuReasonContains(reason, "fallback")) {
    gpuBase.gpuSyncCpuElementCount++;
  } else {
    gpuBase.gpuSyncOtherCount++;
  }
}

static const char *gpuMode(void) {
  const char *mode = getenv("ELEGANT_GPU_MODE");

  if (!mode || !*mode)
    return ELEGANT_GPU_DEFAULT_MODE;
  return mode;
}

static long gpuReferenceOutputUsesCpuHelpers(void) {
  return gpuBase.orderSensitiveOutputNeeded && !gpuHelperMinParticlesExplicit;
}

void gpuDescribeUsageSettings(char *buffer, unsigned long bufferSize) {
  const char *modeEnv = getenv("ELEGANT_GPU_MODE");
  const char *minParticlesEnv = getenv("ELEGANT_GPU_MIN_PARTICLES");
  const char *mode = gpuMode();
  long minParticles = gpuEnvLong("ELEGANT_GPU_MIN_PARTICLES",
                                 ELEGANT_GPU_DEFAULT_MIN_PARTICLES);
  char modeDetail[64];
  char statusBuffer[512];
  char minParticlesDetail[64];
  int deviceCount = 0;
  int status;

  if (!buffer || !bufferSize)
    return;

  if (modeEnv && *modeEnv)
    snprintf(modeDetail, sizeof(modeDetail), " (default %s)",
             ELEGANT_GPU_DEFAULT_MODE);
  else
    snprintf(modeDetail, sizeof(modeDetail), " (default)");
  if (minParticlesEnv && *minParticlesEnv)
    snprintf(minParticlesDetail, sizeof(minParticlesDetail), " (default %ld)",
             ELEGANT_GPU_DEFAULT_MIN_PARTICLES);
  else
    snprintf(minParticlesDetail, sizeof(minParticlesDetail), " (default)");

  statusBuffer[0] = 0;
  if (strcmp(mode, "off") == 0) {
    snprintf(statusBuffer, sizeof(statusBuffer),
             "CPU fallback (CUDA disabled by ELEGANT_GPU_MODE=off)");
  } else if (strcmp(mode, "auto") != 0 && strcmp(mode, "required") != 0) {
    snprintf(statusBuffer, sizeof(statusBuffer),
             "invalid ELEGANT_GPU_MODE; valid values are off, auto, required (run will fail)");
  } else {
    status = gpuCudaRuntimeGetDeviceCount(&deviceCount);
    if (status != 0 || deviceCount <= 0) {
      if (strcmp(mode, "required") == 0)
        snprintf(statusBuffer, sizeof(statusBuffer),
                 "%s (required mode will fail)",
                 status == 0 ? "no CUDA devices found" :
                 gpuCudaRuntimeGetErrorString(status));
      else
        snprintf(statusBuffer, sizeof(statusBuffer),
                 "CPU fallback (%s)",
                 status == 0 ? "no CUDA devices found" :
                 gpuCudaRuntimeGetErrorString(status));
    } else {
      long activeDevice;
      if (gpuEnvSet("ELEGANT_GPU_DEVICE")) {
        activeDevice = gpuEnvLong("ELEGANT_GPU_DEVICE", 0);
      } else {
        activeDevice = 0;
#if USE_MPI
        if (n_processors > 1) {
          if (isSlave && distributedBeam)
            activeDevice = (myid - 1) % deviceCount;
          else
            activeDevice = myid % deviceCount;
        }
#endif
      }
      if (activeDevice < 0 || activeDevice >= deviceCount) {
        if (strcmp(mode, "required") == 0)
          snprintf(statusBuffer, sizeof(statusBuffer),
                   "ELEGANT_GPU_DEVICE=%ld is outside the available range 0-%d (required mode will fail)",
                   activeDevice, deviceCount - 1);
        else
          snprintf(statusBuffer, sizeof(statusBuffer),
                   "CPU fallback (ELEGANT_GPU_DEVICE=%ld is outside the available range 0-%d)",
                   activeDevice, deviceCount - 1);
      } else {
        char deviceName[256] = "";
        status = gpuCudaRuntimeGetDeviceName((int)activeDevice, deviceName,
                                             sizeof(deviceName));
        snprintf(statusBuffer, sizeof(statusBuffer),
                 "CUDA device %ld%s%s",
                 activeDevice,
                 status == 0 ? ": " : "",
                 status == 0 ? deviceName : " detected");
      }
    }
  }

  snprintf(buffer, bufferSize,
           "CUDA settings: ELEGANT_GPU_MODE=%s%s; "
           "ELEGANT_GPU_MIN_PARTICLES=%ld%s; %s.",
           mode,
           modeDetail,
           minParticles,
           minParticlesDetail,
           statusBuffer);
}

static const char *gpuExactDriftStatus(void) {
  if (gpuEnableExactDrift)
    return gpuExactDriftExplicit ? "; exact tracking explicitly enabled" :
                                   "; exact tracking enabled by default";
  return "; exact tracking explicitly disabled";
}

static const char *gpuMatrixTrackingStatus(void) {
  if (!gpuBase.orderSensitiveOutputNeeded)
    return "";
  return gpuMatrixDriftMinParticlesExplicit ?
         "; matrix tracking follows explicit matrix-drift threshold" :
         "; matrix tracking uses CPU for order-sensitive output by default";
}

static const char *gpuHelperOutputStatus(void) {
  if (!gpuBase.orderSensitiveOutputNeeded)
    return "";
  return gpuHelperMinParticlesExplicit ?
         "; output helpers follow explicit helper threshold" :
         "; output helpers use CPU by default";
}

static const char *gpuMagnetTrackingStatus(void) {
  if (!gpuBase.orderSensitiveOutputNeeded)
    return "";
  return gpuMagnetMinParticlesExplicit ?
         "; magnet tracking follows explicit magnet threshold" :
         "; magnet tracking uses CPU for order-sensitive output by default";
}

static const char *gpuCsrResidentStatus(void) {
  if (gpuBase.orderSensitiveOutputNeeded &&
      !gpuCsrTrackingExplicit && !gpuCsrResidentExplicit)
    return "; CSR resident drift uses CPU for order-sensitive output by default";
  if (gpuEnableCsrResident)
    return gpuCsrResidentExplicit ?
           "; CSR resident drift explicitly enabled" :
           "; CSR resident drift enabled by default";
  if (gpuCsrResidentExplicit)
    return gpuEnvFlag("ELEGANT_GPU_ENABLE_CSR_RESIDENT_DRIFT") ?
           "; CSR resident drift requested but CSR drift tracking disabled" :
           "; CSR resident drift explicitly disabled";
  return "";
}

static const char *gpuCsrTrackingStatus(void) {
  if (gpuBase.orderSensitiveOutputNeeded &&
      !gpuCsrTrackingExplicit && !gpuCsrResidentExplicit)
    return "; CSR drift tracking uses CPU for order-sensitive output by default";
  if (gpuEnableCsrTracking)
    return gpuCsrTrackingExplicit ?
           "; CSR drift tracking explicitly enabled" :
           "; CSR drift tracking enabled by default";
  return gpuCsrTrackingExplicit ? "; CSR drift tracking explicitly disabled" :
                                  "";
}

static const char *gpuScmultStatus(void) {
  if (gpuEnableScmult)
    return gpuScmultExplicit ? "; SCMULT explicitly enabled" : "; SCMULT enabled";
  return gpuScmultExplicit ? "; SCMULT explicitly disabled" : "";
}

static const char *gpuWakeTrackingStatus(void) {
  if (gpuEnableWakeTrackingDrift)
    return "; wake tracking drift paths explicitly enabled";
  return gpuWakeTrackingDriftExplicit ?
         "; wake tracking drift paths explicitly disabled" :
         "; wake tracking drift paths disabled by default";
}

static const char *gpuCombinedWakeStatus(void) {
  if (gpuBase.orderSensitiveOutputNeeded && !gpuCombinedWakeExplicit)
    return "; IMPEDANCE/CWAKE uses CPU for order-sensitive output by default";
  if (gpuEnableCombinedWake)
    return gpuEnableCombinedWakeMultibunch ?
           (gpuEnableCombinedWakeFft ?
            (gpuCombinedWakeExplicit ?
             "; IMPEDANCE/CWAKE explicitly enabled; multibunch and CWAKE FFT enabled" :
             "; IMPEDANCE/CWAKE enabled by default; multibunch and CWAKE FFT enabled") :
            (gpuCombinedWakeExplicit ?
             "; IMPEDANCE/CWAKE explicitly enabled; multibunch enabled" :
             "; IMPEDANCE/CWAKE enabled by default; multibunch enabled")) :
           (gpuEnableCombinedWakeFft ?
            (gpuCombinedWakeExplicit ?
             "; IMPEDANCE/CWAKE explicitly enabled; CWAKE FFT enabled" :
             "; IMPEDANCE/CWAKE enabled by default; CWAKE FFT enabled") :
            (gpuCombinedWakeExplicit ?
             "; IMPEDANCE/CWAKE explicitly enabled" :
             "; IMPEDANCE/CWAKE enabled by default"));
  return gpuCombinedWakeExplicit ?
         "; IMPEDANCE/CWAKE explicitly disabled" :
         "";
}

static const char *gpuLscTrackingStatus(void) {
  if (!gpuEnableLscTracking)
    return gpuLscTrackingExplicit ? "; LSC tracking explicitly disabled" :
                                    "; LSC tracking disabled";
  return gpuLscTrackingExplicit ? "; LSC tracking explicitly enabled" :
                                  "; LSC tracking enabled by default";
}

static const char *gpuRfcwTrackingDriftStatus(void) {
  if (gpuEnableRfcwTrackingDrift)
    return "; RFCW tracking drift paths explicitly enabled";
  return gpuRfcwTrackingDriftExplicit ?
         "; RFCW tracking drift paths explicitly disabled" :
         "; RFCW tracking drift paths disabled by default";
}

static const char *gpuRfcaChangeP0DriftStatus(void) {
  if (gpuEnableRfcaChangeP0Drift)
    return "; RF cavity change-p0 drift paths explicitly enabled";
  return gpuRfcaChangeP0DriftExplicit ?
         "; RF cavity change-p0 drift paths explicitly disabled" :
         "; RF cavity change-p0 drift paths disabled by default";
}

static const char *gpuReductionOutputStatus(void) {
  if (!gpuBase.reductionOutputNeeded)
    return "";
  return gpuOutputDriftReductionMinParticlesExplicit ?
         "; output reductions follow explicit output-drift reduction threshold" :
         "; output reductions use CPU by default";
}

static const char *gpuApertureParallelCompactionStatus(void) {
  if (gpuApertureParallelCompactionVerifyDisabled)
    return "; aperture parallel compaction disabled in GPU_VERIFY";
  if (gpuApertureParallelCompactionOrderDisabled)
    return "; aperture parallel compaction disabled for order-sensitive output";
  if (gpuEnableApertureParallelCompaction)
    return gpuApertureParallelCompactionExplicit ?
           "; aperture parallel compaction explicitly enabled" :
           "; aperture parallel compaction enabled for no-loss-output runs";
  return gpuApertureParallelCompactionExplicit ?
         "; aperture parallel compaction explicitly disabled" : "";
}

static const char *gpuMagnetLossCompactionStatus(void) {
  if (gpuEnableMagnetLossCompaction)
    return gpuMagnetLossCompactionExplicit ?
           "; magnet loss compaction explicitly enabled" :
           "; magnet loss compaction enabled for no-loss-output runs";
  return gpuMagnetLossCompactionExplicit ?
         "; magnet loss compaction explicitly disabled" : "";
}

static const char *gpuCsbendDriftStatus(void) {
  if (gpuEnableCsbendDrift)
    return gpuCsbendDriftExplicit ?
           "; CSBEND drift paths explicitly enabled" :
           "; CSBEND drift paths enabled by default";
  return gpuCsbendDriftExplicit ? "; CSBEND drift paths explicitly disabled" :
                                  "";
}

static void gpuRequiredFailure(const char *message) {
  fprintf(stderr, "elegant CUDA support requested in required mode: %s\n", message);
  exit(1);
}

static void gpuFatalStatus(const char *operation, int status) {
  fprintf(stderr, "elegant CUDA: %s failed: %s\n", operation,
          gpuCudaRuntimeGetErrorString(status));
  exit(1);
}

static void gpuUnsupported(const char *functionName) {
  if (!unsupportedReported) {
    fprintf(stderr,
            "elegant CUDA backend reached unsupported GPU function %s. "
            "Phase 2 currently enables verified matrix, helper, and common reduction kernels.\n",
            functionName);
    unsupportedReported = 1;
  }
  exit(1);
}

static long gpuSimpleMatrixElement(ELEMENT_LIST *eptr) {
  if (!eptr)
    return 0;
  if ((entity_description[eptr->type].flags & MATRIX_TRACKING) &&
      (entity_description[eptr->type].flags & HAS_MATRIX))
    return 1;
  switch (eptr->type) {
  case T_ROTATE:
  case T_SOLE:
    return !spinCoordOffset;
  case T_HCOR:
    return eptr->p_elem && !((HCOR *)eptr->p_elem)->synchRad &&
           !((HCOR *)eptr->p_elem)->isr;
  case T_VCOR:
    return eptr->p_elem && !((VCOR *)eptr->p_elem)->synchRad &&
           !((VCOR *)eptr->p_elem)->isr;
  case T_HVCOR:
    return eptr->p_elem && !((HVCOR *)eptr->p_elem)->synchRad &&
           !((HVCOR *)eptr->p_elem)->isr;
  default:
    break;
  }
  return 0;
}

static long gpuSpecialMatrixElementUsesCpuAfterTrack(ELEMENT_LIST *eptr) {
  if (!eptr)
    return 1;
  switch (eptr->type) {
  case T_ROTATE:
  case T_SOLE:
    return spinCoordOffset != 0;
  case T_HCOR:
    return !eptr->p_elem || ((HCOR *)eptr->p_elem)->synchRad ||
           ((HCOR *)eptr->p_elem)->isr;
  case T_VCOR:
    return !eptr->p_elem || ((VCOR *)eptr->p_elem)->synchRad ||
           ((VCOR *)eptr->p_elem)->isr;
  case T_HVCOR:
    return !eptr->p_elem || ((HVCOR *)eptr->p_elem)->synchRad ||
           ((HVCOR *)eptr->p_elem)->isr;
  default:
    break;
  }
  return 0;
}

static long gpuHelperElementSupported(ELEMENT_LIST *eptr) {
  if (!eptr)
    return 0;
  if (gpuReferenceOutputUsesCpuHelpers())
    return 0;
  switch (eptr->type) {
  case T_MALIGN:
  case T_ENERGY:
  case T_MATR:
  case T_EMATRIX:
    return 1;
  case T_CENTER:
    return 0;
  default:
    break;
  }
  return 0;
}

static long gpuApertureElementSupported(ELEMENT_LIST *eptr) {
  if (!eptr)
    return 0;
  switch (eptr->type) {
  case T_RCOL:
  case T_ECOL:
  case T_SCRAPER:
    return 1;
  default:
    break;
  }
  return 0;
}

static long gpuRfcwStringPresent(const char *value) {
  return value && value[0];
}

static unsigned long gpuRfcwFiducialMode(const char *mode) {
  if (!mode)
    return FID_MODE_TMEAN;
  if (strcmp(mode, "light") == 0)
    return FID_MODE_LIGHT;
  if (strcmp(mode, "tmean") == 0)
    return FID_MODE_TMEAN;
  if (strcmp(mode, "first") == 0)
    return FID_MODE_FIRST;
  if (strcmp(mode, "pmaximum") == 0)
    return FID_MODE_PMAX;
  return 0;
}

static long gpuFiducialPidRange(unsigned long mode, long *startPID,
                                long *endPID) {
  if (!startPID || !endPID)
    return 0;
  *startPID = *endPID = -1;
  if (getFiducializationPidRange)
    return getFiducializationPidRange(mode, startPID, endPID);
  return 1;
}

static long gpuFiducialModeSupported(unsigned long mode) {
  long startPID = -1, endPID = -1;

  if (!mode)
    return 0;
  if (!gpuFiducialPidRange(mode, &startPID, &endPID))
    return 0;
  if (mode & (FID_MODE_LIGHT | FID_MODE_TMEAN | FID_MODE_FIRST |
              FID_MODE_PMAX))
    return 1;
  return 0;
}

static long gpuRfcwRfOnlyElementSupported(ELEMENT_LIST *eptr) {
  RFCW *rfcw;
  double phase = 0;

#if USE_MPI
  if (distributedBeam)
    return 0;
#endif
  if (gpuReferenceOutputUsesCpuHelpers())
    return 0;
  if (!eptr || eptr->type != T_RFCW || !eptr->p_elem)
    return 0;
  rfcw = (RFCW *)eptr->p_elem;
  if (rfcw->cellLength <= 0 || rfcw->length <= 0)
    return 0;
  if (rfcw->Q != 0 || rfcw->change_t || rfcw->linearize ||
      rfcw->backtrack)
    return 0;
  if ((rfcw->change_p0 || gpuRunAlwaysChangeP0) && !gpuEnableRfcaChangeP0Drift)
    return 0;
  if (gpuRfcwStringPresent(rfcw->bodyFocusModel))
    return 0;
  if (rfcw->nKicks > 0)
    return 0;
  if (rfcw->doLSC)
    return 0;
  if (gpuRfcwStringPresent(rfcw->wakeFile) ||
      gpuRfcwStringPresent(rfcw->zWakeFile) ||
      gpuRfcwStringPresent(rfcw->trWakeFile) ||
      gpuRfcwStringPresent(rfcw->tColumn) ||
      gpuRfcwStringPresent(rfcw->WxColumn) ||
      gpuRfcwStringPresent(rfcw->WyColumn) ||
      gpuRfcwStringPresent(rfcw->WzColumn))
    return 0;
  if (rfcw->initialized && rfcw->rfca.fiducial_seen &&
      rfcw->rfca.phase_reference != 0 && get_phase_reference &&
      get_phase_reference(&phase, rfcw->rfca.phase_reference) ==
        REF_PHASE_RETURNED)
    return 1;
  if (!get_phase_reference || !set_phase_reference || !unused_phase_reference)
    return 0;
  if (rfcw->tReference != -1)
    return 1;
  if (gpuFiducialModeSupported(gpuRfcwFiducialMode(rfcw->fiducial)))
    return 1;
  return 0;
}

static long gpuRfcwRfOnlyKickElementSupported(ELEMENT_LIST *eptr) {
  RFCW *rfcw;
  double phase = 0;

#if USE_MPI
  if (distributedBeam)
    return 0;
#endif
  if (gpuReferenceOutputUsesCpuHelpers())
    return 0;
  if (!eptr || eptr->type != T_RFCW || !eptr->p_elem)
    return 0;
  rfcw = (RFCW *)eptr->p_elem;
  if (rfcw->cellLength <= 0 || rfcw->length <= 0)
    return 0;
  if (rfcw->Q != 0 || rfcw->change_t || rfcw->linearize ||
      rfcw->backtrack)
    return 0;
  if ((rfcw->change_p0 || gpuRunAlwaysChangeP0) && !gpuEnableRfcaChangeP0Drift)
    return 0;
  if (gpuRfcwStringPresent(rfcw->bodyFocusModel))
    return 0;
  if (rfcw->nKicks < 1)
    return 0;
  if (rfcw->doLSC)
    return 0;
  if (gpuRfcwStringPresent(rfcw->wakeFile) ||
      gpuRfcwStringPresent(rfcw->zWakeFile) ||
      gpuRfcwStringPresent(rfcw->trWakeFile) ||
      gpuRfcwStringPresent(rfcw->tColumn) ||
      gpuRfcwStringPresent(rfcw->WxColumn) ||
      gpuRfcwStringPresent(rfcw->WyColumn) ||
      gpuRfcwStringPresent(rfcw->WzColumn))
    return 0;
  if (rfcw->initialized && rfcw->rfca.fiducial_seen &&
      rfcw->rfca.phase_reference != 0 && get_phase_reference &&
      get_phase_reference(&phase, rfcw->rfca.phase_reference) ==
        REF_PHASE_RETURNED)
    return 1;
  if (!get_phase_reference || !set_phase_reference || !unused_phase_reference)
    return 0;
  if (rfcw->tReference != -1)
    return 1;
  if (gpuFiducialModeSupported(gpuRfcwFiducialMode(rfcw->fiducial)))
    return 1;
  return 0;
}

static long gpuRfcwWakeActive(RFCW *rfcw, long *hasWake, long *hasTrwake) {
  long zWake = 0, trWake = 0;

  if (!rfcw)
    return 0;
  zWake = rfcw->includeZWake && gpuRfcwStringPresent(rfcw->WzColumn) &&
          (gpuRfcwStringPresent(rfcw->zWakeFile) ||
           gpuRfcwStringPresent(rfcw->wakeFile));
  trWake = rfcw->includeTrWake &&
           (gpuRfcwStringPresent(rfcw->WxColumn) ||
            gpuRfcwStringPresent(rfcw->WyColumn)) &&
           (gpuRfcwStringPresent(rfcw->trWakeFile) ||
            gpuRfcwStringPresent(rfcw->wakeFile));
  if (hasWake)
    *hasWake = zWake;
  if (hasTrwake)
    *hasTrwake = trWake;
  return zWake || trWake;
}

static long gpuRfcwLscDataSupported(RFCW *rfcw) {
  if (!rfcw)
    return 0;
  if (!rfcw->doLSC)
    return 1;
  if (rfcw->LSCBins < 2 || (rfcw->LSCBins % 2))
    return 0;
  if (rfcw->LSCInterpolate != 0 && rfcw->LSCInterpolate != 1)
    return 0;
  if (rfcw->LSCLowFrequencyCutoff0 >= 0 &&
      rfcw->LSCLowFrequencyCutoff1 < rfcw->LSCLowFrequencyCutoff0)
    return 0;
  if (rfcw->LSCHighFrequencyCutoff0 > 0 &&
      rfcw->LSCHighFrequencyCutoff1 <= rfcw->LSCHighFrequencyCutoff0)
    return 0;
  if (rfcw->LSCRadiusFactor == 0)
    return 0;
  return 1;
}

static long gpuRfcwLscKickDataSupported(LSCKICK *lsc) {
  if (!lsc)
    return 0;
  if (lsc->bins < 2 || (lsc->bins % 2))
    return 0;
  if (lsc->interpolate != 0 && lsc->interpolate != 1)
    return 0;
  if (lsc->lowFrequencyCutoff0 >= 0 &&
      lsc->lowFrequencyCutoff1 < lsc->lowFrequencyCutoff0)
    return 0;
  if (lsc->highFrequencyCutoff0 > 0 &&
      lsc->highFrequencyCutoff1 <= lsc->highFrequencyCutoff0)
    return 0;
  if (lsc->radiusFactor == 0)
    return 0;
  return 1;
}

static long gpuRfcwMatrixWakeElementSupported(ELEMENT_LIST *eptr) {
  RFCW *rfcw;
  double phase = 0;
  long hasWake = 0, hasTrwake = 0;

#if USE_MPI
  if (distributedBeam)
    return 0;
#endif
  if (!eptr || eptr->type != T_RFCW || !eptr->p_elem)
    return 0;
  rfcw = (RFCW *)eptr->p_elem;
  if (rfcw->cellLength <= 0 || rfcw->length <= 0)
    return 0;
  if (rfcw->Q != 0 || rfcw->change_t || rfcw->linearize ||
      rfcw->backtrack)
    return 0;
  if ((rfcw->change_p0 || gpuRunAlwaysChangeP0) && !gpuEnableRfcaChangeP0Drift)
    return 0;
  if (gpuRfcwStringPresent(rfcw->bodyFocusModel))
    return 0;
  if (rfcw->nKicks > 0)
    return 0;
  if (!gpuRfcwLscDataSupported(rfcw))
    return 0;
  if (rfcw->doLSC &&
      (!gpuEnableLscTracking || !gpuEnableRfcwTrackingDrift))
    return 0;
  gpuRfcwWakeActive(rfcw, &hasWake, &hasTrwake);
  if (!rfcw->doLSC && !hasWake && !hasTrwake)
    return 0;
  if ((hasWake || hasTrwake) && !gpuEnableWakeTrackingDrift)
    return 0;
  if (rfcw->n_bins != 0 && rfcw->n_bins < 2)
    return 0;
  if (rfcw->interpolate != 0 && rfcw->interpolate != 1)
    return 0;
  if (rfcw->initialized && rfcw->rfca.fiducial_seen &&
      rfcw->rfca.phase_reference != 0 && get_phase_reference &&
      get_phase_reference(&phase, rfcw->rfca.phase_reference) ==
        REF_PHASE_RETURNED)
    return 1;
  if (!get_phase_reference || !set_phase_reference || !unused_phase_reference)
    return 0;
  if (rfcw->tReference != -1)
    return 1;
  if (gpuFiducialModeSupported(gpuRfcwFiducialMode(rfcw->fiducial)))
    return 1;
  return 0;
}

static long gpuRfcwKickWakeElementSupported(ELEMENT_LIST *eptr) {
  RFCW *rfcw;
  double phase = 0;
  long hasWake = 0, hasTrwake = 0;

#if USE_MPI
  if (distributedBeam)
    return 0;
#endif
  if (!eptr || eptr->type != T_RFCW || !eptr->p_elem)
    return 0;
  rfcw = (RFCW *)eptr->p_elem;
  if (rfcw->cellLength <= 0 || rfcw->length <= 0)
    return 0;
  if (rfcw->Q != 0 || rfcw->change_t || rfcw->linearize ||
      rfcw->backtrack)
    return 0;
  if ((rfcw->change_p0 || gpuRunAlwaysChangeP0) && !gpuEnableRfcaChangeP0Drift)
    return 0;
  if (gpuRfcwStringPresent(rfcw->bodyFocusModel))
    return 0;
  if (rfcw->nKicks < 1)
    return 0;
  if (!gpuRfcwLscDataSupported(rfcw))
    return 0;
  if (rfcw->doLSC &&
      (!gpuEnableLscTracking || !gpuEnableRfcwTrackingDrift))
    return 0;
  gpuRfcwWakeActive(rfcw, &hasWake, &hasTrwake);
  if (!rfcw->doLSC && !hasWake && !hasTrwake)
    return 0;
  if ((hasWake || hasTrwake) && !gpuEnableWakeTrackingDrift)
    return 0;
  if (rfcw->n_bins != 0 && rfcw->n_bins < 2)
    return 0;
  if (rfcw->interpolate != 0 && rfcw->interpolate != 1)
    return 0;
  if (rfcw->initialized && rfcw->rfca.fiducial_seen &&
      rfcw->rfca.phase_reference != 0 && get_phase_reference &&
      get_phase_reference(&phase, rfcw->rfca.phase_reference) ==
        REF_PHASE_RETURNED)
    return 1;
  if (!get_phase_reference || !set_phase_reference || !unused_phase_reference)
    return 0;
  if (rfcw->tReference != -1)
    return 1;
  if (gpuFiducialModeSupported(gpuRfcwFiducialMode(rfcw->fiducial)))
    return 1;
  return 0;
}

static long gpuRfcaRemoveInvalidOnlyElementSupported(ELEMENT_LIST *eptr) {
  RFCA *rfca;

  if (!eptr || eptr->type != T_RFCA || !eptr->p_elem)
    return 0;
  rfca = (RFCA *)eptr->p_elem;
  if (rfca->length != 0 || rfca->volt == 0 || rfca->freq != 0 ||
      rfca->phase != 0 || rfca->Q != 0)
    return 0;
  if (rfca->change_p0 || rfca->change_t || rfca->end1Focus ||
      rfca->end2Focus || rfca->standingWave || rfca->linearize ||
      rfca->lockPhase || rfca->dx != 0 || rfca->dy != 0 ||
      rfca->backtrack)
    return 0;
  if (rfca->nKicks != 1)
    return 0;
  return 1;
}

static long gpuRfcaThinKickElementSupported(ELEMENT_LIST *eptr) {
  RFCA *rfca;

  if (gpuReferenceOutputUsesCpuHelpers())
    return 0;
  if (!eptr || eptr->type != T_RFCA || !eptr->p_elem)
    return 0;
  rfca = (RFCA *)eptr->p_elem;
  if (rfca->length != 0 || rfca->Q != 0)
    return 0;
  if (rfca->volt == 0)
    return 0;
  if ((rfca->change_p0 || gpuRunAlwaysChangeP0) && !gpuEnableRfcaChangeP0Drift)
    return 0;
  if (rfca->change_t || rfca->end1Focus || rfca->end2Focus ||
      rfca->linearize || rfca->lockPhase || rfca->backtrack)
    return 0;
  if (rfca->standingWave && rfca->nKicks != 1 && rfca->tReference == -1)
    return 0;
  return 1;
}

static long gpuRfcaRfOnlyMatrixElementSupported(ELEMENT_LIST *eptr) {
  RFCA *rfca;

#if USE_MPI
  if (distributedBeam)
    return 0;
#endif
  if (gpuReferenceOutputUsesCpuHelpers())
    return 0;
  if (!eptr || eptr->type != T_RFCA || !eptr->p_elem)
    return 0;
  rfca = (RFCA *)eptr->p_elem;
  if (rfca->length <= 0 || rfca->volt == 0 || rfca->Q != 0)
    return 0;
  if ((rfca->change_p0 || gpuRunAlwaysChangeP0) && !gpuEnableRfcaChangeP0Drift)
    return 0;
  if (rfca->change_t || rfca->linearize || rfca->lockPhase ||
      rfca->backtrack)
    return 0;
  if (gpuRfcwStringPresent(rfca->bodyFocusModel))
    return 0;
  if (rfca->nKicks > 0)
    return 0;
  if (!get_phase_reference || !set_phase_reference || !unused_phase_reference)
    return 0;
  if (rfca->tReference != -1)
    return 1;
  if (gpuFiducialModeSupported(gpuRfcwFiducialMode(rfca->fiducial)))
    return 1;
  return 0;
}

static long gpuRfcaRfOnlyKickElementSupported(ELEMENT_LIST *eptr) {
  RFCA *rfca;

#if USE_MPI
  if (distributedBeam)
    return 0;
#endif
  if (gpuReferenceOutputUsesCpuHelpers())
    return 0;
  if (!eptr || eptr->type != T_RFCA || !eptr->p_elem)
    return 0;
  rfca = (RFCA *)eptr->p_elem;
  if (rfca->length <= 0 || rfca->volt == 0 || rfca->Q != 0)
    return 0;
  if ((rfca->change_p0 || gpuRunAlwaysChangeP0) && !gpuEnableRfcaChangeP0Drift)
    return 0;
  if (rfca->change_t || rfca->linearize || rfca->lockPhase ||
      rfca->backtrack)
    return 0;
  if (gpuRfcwStringPresent(rfca->bodyFocusModel))
    return 0;
  if (rfca->nKicks < 1)
    return 0;
  if (!get_phase_reference || !set_phase_reference || !unused_phase_reference)
    return 0;
  if (rfca->tReference != -1)
    return 1;
  if (gpuFiducialModeSupported(gpuRfcwFiducialMode(rfca->fiducial)))
    return 1;
  return 0;
}

static long gpuKickMapElementSupported(ELEMENT_LIST *eptr) {
#if USE_MPI
  if (distributedBeam)
    return 0;
#endif
  if (gpuBase.backtrack)
    return 0;
  if (!eptr || !eptr->p_elem)
    return 0;
  switch (eptr->type) {
  case T_KICKMAP: {
    KICKMAP *map = (KICKMAP *)eptr->p_elem;

    if (!map->inputFile || !*map->inputFile)
      return 0;
    if (map->nKicks < 1 || map->length == 0)
      return 0;
    if (map->isr)
      return 0;
    if (map->tilt || map->dx || map->dy || map->dz || map->yaw)
      return 0;
    return 1;
  }
  case T_UKICKMAP: {
    UKICKMAP *map = (UKICKMAP *)eptr->p_elem;

    if (gpuBase.orderSensitiveOutputNeeded && !map->synchRad)
      return 0;
    if (!map->inputFile || !*map->inputFile)
      return 0;
    if (map->nKicks < 1 || map->length == 0)
      return 0;
    if (map->isr)
      return 0;
    if (map->synchRad && map->nKicks != map->periods)
      return 0;
    if (map->tilt || map->dx || map->dy || map->dz || map->yaw)
      return 0;
    return 1;
  }
  default:
    break;
  }
  return 0;
}

static long gpuStringSet(const char *value) {
  return value && *value;
}

static long gpuMultipoleDataPresent(MULTIPOLE_DATA *data) {
  return data && data->initialized && data->orders > 0;
}

static long gpuMultipoleCommonSupported(long nSlices, long nKicks,
                                        short integrationOrder,
                                        short synchRad, short isr,
                                        double tilt, double pitch, double yaw,
                                        double dx, double dy, double dz,
                                        short malignMethod,
                                        short expandHamiltonian) {
  (void)expandHamiltonian;
  (void)synchRad;

  if (spinCoordOffset)
    return 0;
  if (isr)
    return 0;
  (void)tilt;
  (void)dx;
  (void)dy;
  (void)dz;
  if (pitch || yaw || malignMethod)
    return 0;
  if (integrationOrder != 2 && integrationOrder != 4 && integrationOrder != 6)
    return 0;
  if (nKicks <= 0 && nSlices <= 0)
    return 0;
  return 1;
}

static long gpuMultipoleElementSupported(ELEMENT_LIST *eptr) {
  if (gpuBase.backtrack)
    return 0;
  if (!eptr || !eptr->p_elem)
    return 0;

  switch (eptr->type) {
  case T_MULT: {
    MULT *multipole = (MULT *)eptr->p_elem;

    if (multipole->synch_rad)
      return 0;
    if (!gpuMultipoleCommonSupported(multipole->nSlices, 0, 2,
                                     multipole->synch_rad, 0,
                                     multipole->tilt, 0, 0,
                                     multipole->dx, multipole->dy,
                                     multipole->dz, 0,
                                     multipole->expandHamiltonian))
      return 0;
    if (multipole->order < 0 || multipole->order > 3)
      return 0;
    if (!isfinite(multipole->length) || !isfinite(multipole->KnL) ||
        !isfinite(multipole->tilt) || !isfinite(multipole->bore) ||
        !isfinite(multipole->BTipL) || !isfinite(multipole->dx) ||
        !isfinite(multipole->dy) || !isfinite(multipole->dz) ||
        !isfinite(multipole->factor))
      return 0;
    return 1;
  }
  case T_KQUAD: {
    KQUAD *kquad = (KQUAD *)eptr->p_elem;

    if (kquad->synch_rad && kquad->length < 1e-6)
      return 0;
    if (!gpuMultipoleCommonSupported(kquad->nSlices, kquad->n_kicks,
                                     kquad->integration_order,
                                     kquad->synch_rad, kquad->isr,
                                     kquad->tilt, kquad->pitch, kquad->yaw,
                                     kquad->dx, kquad->dy, kquad->dz,
                                     kquad->malignMethod,
                                     kquad->expandHamiltonian))
      return 0;
    if (kquad->edge1_effects || kquad->edge2_effects || kquad->radial)
      return 0;
    if (kquad->lEffective > 0 && kquad->lEffective != kquad->length)
      return 0;
    if (gpuStringSet(kquad->systematic_multipoles) ||
        gpuStringSet(kquad->edge_multipoles) ||
        gpuStringSet(kquad->random_multipoles) ||
        gpuStringSet(kquad->steering_multipoles))
      return 0;
    if (gpuMultipoleDataPresent(&kquad->systematicMultipoleData) ||
        gpuMultipoleDataPresent(&kquad->edgeMultipoleData) ||
        gpuMultipoleDataPresent(&kquad->randomMultipoleData) ||
        gpuMultipoleDataPresent(&kquad->totalMultipoleData) ||
        gpuMultipoleDataPresent(&kquad->steeringMultipoleData))
      return 0;
    return 1;
  }
  case T_KSEXT: {
    KSEXT *ksext = (KSEXT *)eptr->p_elem;

    if (ksext->synch_rad)
      return 0;
    if (!gpuMultipoleCommonSupported(ksext->nSlices, ksext->n_kicks,
                                     ksext->integration_order,
                                     ksext->synch_rad, ksext->isr,
                                     ksext->tilt, ksext->pitch, ksext->yaw,
                                     ksext->dx, ksext->dy, ksext->dz,
                                     ksext->malignMethod,
                                     ksext->expandHamiltonian))
      return 0;
    if (gpuStringSet(ksext->systematic_multipoles) ||
        gpuStringSet(ksext->edge_multipoles) ||
        gpuStringSet(ksext->random_multipoles) ||
        gpuStringSet(ksext->steering_multipoles))
      return 0;
    if (gpuMultipoleDataPresent(&ksext->systematicMultipoleData) ||
        gpuMultipoleDataPresent(&ksext->edgeMultipoleData) ||
        gpuMultipoleDataPresent(&ksext->randomMultipoleData) ||
        gpuMultipoleDataPresent(&ksext->totalMultipoleData) ||
        gpuMultipoleDataPresent(&ksext->steeringMultipoleData))
      return 0;
    return 1;
  }
  case T_KOCT: {
    KOCT *koct = (KOCT *)eptr->p_elem;

    if (koct->synch_rad)
      return 0;
    if (!gpuMultipoleCommonSupported(koct->nSlices, koct->n_kicks,
                                     koct->integration_order,
                                     koct->synch_rad, koct->isr,
                                     koct->tilt, koct->pitch, koct->yaw,
                                     koct->dx, koct->dy, koct->dz,
                                     koct->malignMethod,
                                     koct->expandHamiltonian))
      return 0;
    if (gpuStringSet(koct->systematic_multipoles) ||
        gpuStringSet(koct->random_multipoles))
      return 0;
    if (gpuMultipoleDataPresent(&koct->systematicMultipoleData) ||
        gpuMultipoleDataPresent(&koct->randomMultipoleData) ||
        gpuMultipoleDataPresent(&koct->totalMultipoleData))
      return 0;
    return 1;
  }
  case T_DQCOR: {
    DQCOR *dqcor = (DQCOR *)eptr->p_elem;

    if (dqcor->synch_rad)
      return 0;
    if (!gpuMultipoleCommonSupported(dqcor->nSlices, 0,
                                     dqcor->integration_order,
                                     dqcor->synch_rad, dqcor->isr,
                                     dqcor->tilt, dqcor->pitch, dqcor->yaw,
                                     dqcor->dx, dqcor->dy, dqcor->dz,
                                     dqcor->malignMethod,
                                     0))
      return 0;
    if (gpuStringSet(dqcor->dipoleSystematicMultipoles) ||
        gpuStringSet(dqcor->quadrupoleSystematicMultipoles) ||
        gpuStringSet(dqcor->quadrupoleRandomMultipoles))
      return 0;
    if (gpuMultipoleDataPresent(&dqcor->dipoleSystematicMultipoleData) ||
        gpuMultipoleDataPresent(&dqcor->quadrupoleSystematicMultipoleData) ||
        gpuMultipoleDataPresent(&dqcor->quadrupoleRandomMultipoleData) ||
        gpuMultipoleDataPresent(&dqcor->totalMultipoleData))
      return 0;
    return 1;
  }
  default:
    break;
  }

  return 0;
}

static long gpuCsbendCommonSupported(CSBEND *csbend) {
  double rho;

  if (!csbend)
    return 0;
  if (spinCoordOffset)
    return 0;
  if (csbend->isr ||
      csbend->distributionBasedRadiation ||
      gpuStringSet(csbend->photonOutputFile))
    return 0;
  if (csbend->referenceCorrection || csbend->fseCorrection)
    return 0;
  if (csbend->epitch || csbend->eyaw || csbend->malignMethod)
    return 0;
  if (csbend->angle == 0)
    return 0;
  rho = csbend->length / csbend->angle;
  if (fabs(rho) > 1e6)
    return 0;
  if (csbend->nSlices <= 0)
    return 0;
  if (csbend->integration_order != 2 && csbend->integration_order != 4 &&
      csbend->integration_order != 6)
    return 0;
  if (csbend->expansionOrder > 10)
    return 0;
  return 1;
}

static long gpuCsbendElementSupported(ELEMENT_LIST *eptr) {
  if (!gpuEnableCsbendDrift)
    return 0;
  if (gpuBase.backtrack)
    return 0;
  if (!eptr || !eptr->p_elem || eptr->type != T_CSBEND)
    return 0;
  return gpuCsbendCommonSupported((CSBEND *)eptr->p_elem);
}

static long gpuCcbendCommonSupported(CCBEND *ccbend) {
  if (!ccbend || !gpuEnableCcbend || spinCoordOffset)
    return 0;
  if (ccbend->optimized != 1 || ccbend->length <= 0 || ccbend->angle <= 0 ||
      ccbend->nSlices <= 0 || ccbend->integration_order != 2 ||
      ccbend->fringeModel != -1)
    return 0;
  if (ccbend->synch_rad || ccbend->isr ||
      ccbend->distributionBasedRadiation || ccbend->synchRadInOrdinaryMatrix)
    return 0;
  if (ccbend->tilt || ccbend->yaw || ccbend->extraTilt ||
      ccbend->dx || ccbend->dy || ccbend->dz || ccbend->eTilt ||
      ccbend->ePitch || ccbend->eYaw || ccbend->malignMethod ||
      ccbend->edgeFlip)
    return 0;
  if (ccbend->K3 || ccbend->K4 || ccbend->K5 || ccbend->K6 ||
      ccbend->K7 || ccbend->K8)
    return 0;
  if (gpuStringSet(ccbend->systematic_multipoles) ||
      gpuStringSet(ccbend->edge_multipoles) ||
      gpuStringSet(ccbend->edge1_multipoles) ||
      gpuStringSet(ccbend->edge2_multipoles) ||
      gpuStringSet(ccbend->random_multipoles) ||
      gpuStringSet(ccbend->centroidOutputFile))
    return 0;
  if (gpuMultipoleDataPresent(&ccbend->systematicMultipoleData) ||
      gpuMultipoleDataPresent(&ccbend->edge1MultipoleData) ||
      gpuMultipoleDataPresent(&ccbend->edge2MultipoleData) ||
      gpuMultipoleDataPresent(&ccbend->randomMultipoleData) ||
      gpuMultipoleDataPresent(&ccbend->totalMultipoleData))
    return 0;
  if (ccbend->centroidsRequested || ccbend->SDDScen ||
      (ccbend->referenceCorrection & ~3) ||
      !isfinite(ccbend->KnDelta) || ccbend->KnDelta == 1 ||
      !isfinite(cos(ccbend->angle / 2)) || cos(ccbend->angle / 2) == 0)
    return 0;
  return 1;
}

static long gpuCcbendElementSupported(ELEMENT_LIST *eptr) {
  if (gpuBase.backtrack || !eptr || !eptr->p_elem ||
      eptr->type != T_CCBEND)
    return 0;
  return gpuCcbendCommonSupported((CCBEND *)eptr->p_elem);
}

static long gpuWakeDataSupported(WAKE *wake) {
  if (!wake)
    return 0;
  if (wake->n_bins != 0 && wake->n_bins < 2)
    return 0;
  if (wake->interpolate != 0 && wake->interpolate != 1)
    return 0;
  return 1;
}

static long gpuWakeElementSupported(ELEMENT_LIST *eptr) {
  if (!gpuEnableWakeTrackingDrift)
    return 0;
  if (!eptr || !eptr->p_elem || eptr->type != T_WAKE)
    return 0;
  return gpuWakeDataSupported((WAKE *)eptr->p_elem);
}

static long gpuTrwakeDataSupported(TRWAKE *wake) {
  if (!wake)
    return 0;
  if (wake->n_bins != 0 && wake->n_bins < 2)
    return 0;
  if (wake->tilt && spinCoordOffset)
    return 0;
  if (wake->interpolate != 0 && wake->interpolate != 1)
    return 0;
  if (wake->xDriveExponent < 0 || wake->yDriveExponent < 0 ||
      wake->xProbeExponent < 0 || wake->yProbeExponent < 0)
    return 0;
  return 1;
}

static long gpuTrwakeElementSupported(ELEMENT_LIST *eptr) {
  if (!gpuEnableWakeTrackingDrift)
    return 0;
  if (!eptr || !eptr->p_elem || eptr->type != T_TRWAKE)
    return 0;
  return gpuTrwakeDataSupported((TRWAKE *)eptr->p_elem);
}

static long gpuCombinedWakeElementSupported(ELEMENT_LIST *eptr) {
  if (!gpuEnableCombinedWake || !eptr || !eptr->p_elem)
    return 0;
  if (gpuBase.orderSensitiveOutputNeeded && !gpuCombinedWakeExplicit)
    return 0;
#if USE_MPI
  if (distributedBeam)
    return 0;
#endif
  if (eptr->type == T_IMPEDANCE) {
    IMPEDANCE *imp = (IMPEDANCE *)eptr->p_elem;
    if (imp->smoothing || imp->area_weight || imp->reverseTimeOrder ||
        gpuStringSet(imp->wakes))
      return 0;
    if (imp->interpolate != 0 && imp->interpolate != 1)
      return 0;
    return 1;
  }
  if (eptr->type == T_CWAKE) {
    CWAKE *wake = (CWAKE *)eptr->p_elem;
    if (wake->smoothing || wake->tilt || gpuBase.backtrack)
      return 0;
    if (wake->n_bins != 0 && wake->n_bins < 2)
      return 0;
    if (wake->interpolate != 0 && wake->interpolate != 1)
      return 0;
    return 1;
  }
  return 0;
}

static long gpuPolynomialSeriesElementSupported(ELEMENT_LIST *eptr) {
  if (!gpuEnablePolynomialSeries || gpuBase.backtrack)
    return 0;
  if (!eptr || eptr->type != T_POLYNOMIALSERIES || !eptr->p_elem)
    return 0;
  return 1;
}

static long gpuRfdfElementSupported(ELEMENT_LIST *eptr) {
  RFDF *rfdf;

  if (!gpuEnableRfdf || gpuBase.backtrack)
    return 0;
  if (!eptr || eptr->type != T_RFDF || !eptr->p_elem)
    return 0;
  rfdf = (RFDF *)eptr->p_elem;
  if (rfdf->frequency == 0 || rfdf->voltage == 0 ||
      rfdf->voltageNoise ||
      rfdf->phaseNoise || rfdf->groupVoltageNoise ||
      rfdf->groupPhaseNoise || rfdf->voltageNoiseGroup ||
      rfdf->phaseNoiseGroup)
    return 0;
  if (rfdf->dx || rfdf->dy || rfdf->dz || rfdf->tilt)
    return 0;
  return 1;
}

static long gpuSreffectsElementSupported(ELEMENT_LIST *eptr) {
#if USE_MPI
  (void)eptr;
  return 0;
#else
  SREFFECTS *sreffects;

  if (!gpuEnableSreffects || gpuBase.backtrack)
    return 0;
  if (!eptr || eptr->type != T_SREFFECTS || !eptr->p_elem)
    return 0;
  sreffects = (SREFFECTS *)eptr->p_elem;
  if (sreffects->qExcite || sreffects->DdeltaRef > 0)
    return 0;
  return 1;
#endif
}

static long gpuRfmodeElementSupported(ELEMENT_LIST *eptr) {
#if USE_MPI
  (void)eptr;
  return 0;
#else
  long noise;
  RFMODE *rfmode;
  FRFMODE *frfmode;

  if (!eptr || !eptr->p_elem || gpuBase.backtrack)
    return 0;
  if (eptr->type == T_RFMODE) {
    if (!gpuEnableRfmode)
      return 0;
    rfmode = (RFMODE *)eptr->p_elem;
    if (rfmode->binless || rfmode->n_bins < 2 || rfmode->bin_size <= 0 ||
        rfmode->bunchedBeamMode != 1 || rfmode->driveFrequency ||
        gpuStringSet(rfmode->record) ||
        gpuStringSet(rfmode->feedbackRecordFile) ||
        gpuStringSet(rfmode->fwaveform) || gpuStringSet(rfmode->Qwaveform) ||
        rfmode->adjustmentFraction ||
        (rfmode->interpolate != 0 && rfmode->interpolate != 1))
      return 0;
    for (noise = 0; noise < 8; noise++)
      if (gpuStringSet(rfmode->noiseData[noise]))
        return 0;
    return 1;
  }
  if (eptr->type == T_FRFMODE) {
    if (!gpuEnableFrfmode)
      return 0;
    frfmode = (FRFMODE *)eptr->p_elem;
    if (frfmode->n_bins < 2 || frfmode->bin_size <= 0 ||
        frfmode->bunchedBeamMode != 1 ||
        gpuStringSet(frfmode->outputFile))
      return 0;
    return 1;
  }
  return 0;
#endif
}

static long gpuBggexpElementSupported(ELEMENT_LIST *eptr) {
  BGGEXP *bgg;

  if (!gpuEnableBggexp || gpuBase.backtrack)
    return 0;
  if (!eptr || eptr->type != T_BGGEXP || !eptr->p_elem)
    return 0;
  bgg = (BGGEXP *)eptr->p_elem;
  if (bgg->symplectic || bgg->synchRad || bgg->isr ||
      bgg->particleOutputFile || bgg->isBend ||
      bgg->dx || bgg->dy || bgg->dz || bgg->tilt ||
      bgg->zInterval != 1)
    return 0;
  if (bgg->skewFilename && bgg->skewFilename[0])
    return 0;
  if ((!bgg->filename || !bgg->filename[0]) &&
      (!bgg->normalFilename || !bgg->normalFilename[0]))
    return 0;
  if (bgg->filename && bgg->filename[0] &&
      bgg->normalFilename && bgg->normalFilename[0])
    return 0;
  return 1;
}

static long gpuCwigglerElementSupported(ELEMENT_LIST *eptr) {
  CWIGGLER *cwiggler;
  long hasHorizontal, hasVertical;

  if (!gpuEnableCwiggler || gpuBase.backtrack)
    return 0;
  if (!eptr || eptr->type != T_CWIGGLER || !eptr->p_elem)
    return 0;
  cwiggler = (CWIGGLER *)eptr->p_elem;
  if (!cwiggler->sinusoidal || cwiggler->isr ||
      cwiggler->tgu || cwiggler->BySplitPole || cwiggler->BxSplitPole ||
      (cwiggler->ByFile && cwiggler->ByFile[0]) ||
      (cwiggler->BxFile && cwiggler->BxFile[0]) ||
      (cwiggler->fieldOutput && cwiggler->fieldOutput[0]) ||
      cwiggler->dx || cwiggler->dy || cwiggler->dz || cwiggler->tilt ||
      cwiggler->BConstant[0] || cwiggler->BConstant[1] ||
      cwiggler->gap > 0 || cwiggler->length <= 0 ||
      cwiggler->periods <= 0 || cwiggler->stepsPerPeriod <= 0 ||
      cwiggler->stepsPerPeriod % 4 ||
      (cwiggler->integrationOrder != 2 && cwiggler->integrationOrder != 4) ||
      cwiggler->poleFactor[0] != 1 || cwiggler->poleFactor[1] != 1 ||
      cwiggler->poleFactor[2] != 1)
    return 0;
  hasHorizontal = !cwiggler->vertical || cwiggler->helical;
  hasVertical = cwiggler->vertical || cwiggler->helical;
  if ((hasHorizontal && !cwiggler->BMax && !cwiggler->ByMax) ||
      (hasVertical && !cwiggler->BMax && !cwiggler->BxMax))
    return 0;
  return 1;
}

static long gpuFtableElementSupported(ELEMENT_LIST *eptr) {
  FTABLE *ftable;
  ntuple *table[3];
  long field, dimension;
  long tableLength = 1;

  if (!gpuEnableFtable || gpuBase.backtrack)
    return 0;
  if (!eptr || eptr->type != T_FTABLE || !eptr->p_elem)
    return 0;
  ftable = (FTABLE *)eptr->p_elem;
  if (!ftable->initialized || ftable->verbose || ftable->nKicks != 1 ||
      ftable->length <= 0 || !isfinite(ftable->factor) ||
      !isfinite(ftable->threshold) || ftable->threshold < 0 ||
      ftable->angle || ftable->e1 || ftable->e2 ||
      ftable->l1 || ftable->l2 || ftable->tilt ||
      ftable->dx || ftable->dy || ftable->dz)
    return 0;
  table[0] = ftable->Bx;
  table[1] = ftable->By;
  table[2] = ftable->Bz;
  for (field = 0; field < 3; field++) {
    if (!table[field] || table[field]->nD != 3 || !table[field]->value ||
        !table[field]->xmin || !table[field]->xmax ||
        !table[field]->dx || !table[field]->xbins)
      return 0;
  }
  for (dimension = 0; dimension < 3; dimension++) {
    if (table[0]->xbins[dimension] < 2 ||
        table[0]->dx[dimension] <= 0 ||
        table[0]->xmax[dimension] <= table[0]->xmin[dimension])
      return 0;
    if (tableLength > LONG_MAX / table[0]->xbins[dimension])
      return 0;
    tableLength *= table[0]->xbins[dimension];
    for (field = 1; field < 3; field++) {
      if (table[field]->xbins[dimension] != table[0]->xbins[dimension] ||
          table[field]->xmin[dimension] != table[0]->xmin[dimension] ||
          table[field]->xmax[dimension] != table[0]->xmax[dimension] ||
          table[field]->dx[dimension] != table[0]->dx[dimension])
        return 0;
    }
  }
  for (field = 0; field < 3; field++) {
    if (table[field]->length != tableLength)
      return 0;
  }
  return 1;
}

static long gpuBmxyzElementSupported(ELEMENT_LIST *eptr) {
#if USE_MPI
  (void)eptr;
  return 0;
#else
  BMAPXYZ *bmxyz;
  BMAPXYZ_DATA *data;
  long tableLength;

  if (!gpuEnableBmxyz || gpuBase.backtrack)
    return 0;
  if (!eptr || eptr->type != T_BMAPXYZ || !eptr->p_elem)
    return 0;
  bmxyz = (BMAPXYZ *)eptr->p_elem;
  if (!bmxyz->filename || !bmxyz->filename[0] ||
      !bmxyz->method || strcmp(bmxyz->method, "non-adaptive runge-kutta") != 0 ||
      !isfinite(bmxyz->accuracy) || bmxyz->accuracy < 1e-6 ||
      bmxyz->accuracy > 0.1 || bmxyz->synchRad || bmxyz->checkFields ||
      bmxyz->injectAtZero || bmxyz->singlePrecision || bmxyz->discardMap ||
      bmxyz->verbosity || bmxyz->xyInterpolationOrder != 1 ||
      bmxyz->xyGridExcess || bmxyz->dxError || bmxyz->dyError ||
      bmxyz->dzError || bmxyz->tilt ||
      (bmxyz->particleOutputFile && bmxyz->particleOutputFile[0]) ||
      (bmxyz->apContourElement && bmxyz->apContourElement[0]) ||
      bmxyz->BFactor[0] != 1 || bmxyz->BFactor[1] != 1 ||
      bmxyz->BFactor[2] != 1 || bmxyz->BInside[0] ||
      bmxyz->BInside[1] || bmxyz->BInside[2])
    return 0;
  if (!bmxyz->data && !bmapxyz_field_setup)
    return 0;
  if (!bmxyz->data)
    bmapxyz_field_setup(bmxyz);
  data = bmxyz->data;
  if (!data || !data->Fx || !data->Fy || !data->Fz ||
      data->nx < 2 || data->ny < 2 || data->nz < 2 ||
      data->dx <= 0 || data->dy <= 0 || data->dz <= 0 ||
      bmxyz->fieldLength <= 0 || bmxyz->length != bmxyz->fieldLength ||
      data->magnetSymmetry[0] || data->magnetSymmetry[1] ||
      data->magnetSymmetry[2])
    return 0;
  tableLength = data->nx;
  if (tableLength > LONG_MAX / data->ny)
    return 0;
  tableLength *= data->ny;
  if (tableLength > LONG_MAX / data->nz)
    return 0;
  tableLength *= data->nz;
  if (tableLength != data->points)
    return 0;
  return 1;
#endif
}

static long gpuLscDataSupported(LSCDRIFT *lsc) {
  if (!lsc)
    return 0;
  if (!lsc->lsc)
    return 0;
  if (lsc->bins < 2 || (lsc->bins % 2))
    return 0;
  if (lsc->interpolate != 0 && lsc->interpolate != 1)
    return 0;
  if (lsc->length < 0 && !lsc->backtrack)
    return 0;
  if (lsc->length == 0 && lsc->lEffective <= 0 && !lsc->autoLEffective)
    return 0;
  if (lsc->lowFrequencyCutoff0 >= 0 &&
      lsc->lowFrequencyCutoff1 < lsc->lowFrequencyCutoff0)
    return 0;
  if (lsc->highFrequencyCutoff0 > 0 &&
      lsc->highFrequencyCutoff1 <= lsc->highFrequencyCutoff0)
    return 0;
  if (lsc->radiusFactor == 0)
    return 0;
  return 1;
}

static long gpuLscElementSupported(ELEMENT_LIST *eptr) {
  LSCDRIFT *lsc;

  if (!gpuEnableLscTracking)
    return 0;
  if (!eptr || !eptr->p_elem || eptr->type != T_LSCDRIFT)
    return 0;
  if (gpuBase.orderSensitiveOutputNeeded)
    return 0;
  lsc = (LSCDRIFT *)eptr->p_elem;
  if (gpuBase.backtrack || lsc->backtrack)
    return 0;
  return gpuLscDataSupported(lsc);
}

static long gpuExactDriftElement(long type) {
  return type == T_EDRIFT;
}

static long gpuCsrDriftNoOpElementSupported(ELEMENT_LIST *eptr) {
  CSRDRIFT *csrDrift;

  if (!gpuEnableCsrResident)
    return 0;
  if (gpuBase.orderSensitiveOutputNeeded &&
      !gpuCsrTrackingExplicit && !gpuCsrResidentExplicit)
    return 0;
  if (!eptr || eptr->type != T_CSRDRIFT || !eptr->p_elem)
    return 0;
  csrDrift = (CSRDRIFT *)eptr->p_elem;
  return !csrDrift->csr && !csrDrift->LSCBins;
}

static long gpuParticleCountMeetsThreshold(long nParticles, long minParticles) {
  return minParticles <= 0 || nParticles >= minParticles;
}

static long gpuParticleCountAllowed(long nParticles, long minParticles) {
  return gpuBase.requiredMode ||
         gpuParticleCountMeetsThreshold(nParticles, minParticles);
}

static long gpuMatrixParticleCountAllowed(long nParticles) {
  if (gpuBase.orderSensitiveOutputNeeded && !gpuMatrixDriftMinParticlesExplicit)
    return 0;
  return gpuParticleCountAllowed(nParticles, gpuBase.matrixMinParticles);
}

static long gpuMagnetParticleCountAllowed(long nParticles) {
  if (gpuBase.orderSensitiveOutputNeeded && !gpuMagnetMinParticlesExplicit)
    return 0;
  return gpuParticleCountAllowed(nParticles, gpuBase.magnetMinParticles);
}

static long gpuCsbendDriftActive(void) {
  return gpuEnableCsbendDrift &&
         gpuDeviceIslandHasCsbend;
}

static long gpuCsbendDriftMatrixParticleCountAllowed(long nParticles) {
  if (gpuBase.orderSensitiveOutputNeeded &&
      !gpuMatrixDriftMinParticlesExplicit &&
      gpuCsbendDriftActive())
    return gpuParticleCountAllowed(nParticles, gpuBase.matrixMinParticles);
  return gpuMatrixParticleCountAllowed(nParticles);
}

static long gpuExactDriftParticleCountAllowed(long nParticles) {
  if (!gpuEnableExactDrift)
    return 0;
  if (gpuBase.orderSensitiveOutputNeeded && !gpuExactDriftExplicit)
    return gpuParticleCountMeetsThreshold(nParticles,
                                          gpuBase.exactDriftMinParticles);
  return gpuParticleCountAllowed(nParticles, gpuBase.exactDriftMinParticles);
}

static long gpuCsbendDriftExactDriftParticleCountAllowed(long nParticles) {
  if (gpuBase.orderSensitiveOutputNeeded &&
      !gpuExactDriftExplicit &&
      gpuCsbendDriftActive())
    return gpuParticleCountAllowed(nParticles,
                                   gpuBase.exactDriftMinParticles);
  return gpuExactDriftParticleCountAllowed(nParticles);
}

static long gpuWakeParticleCountAllowed(long nParticles) {
  if (trajectoryTracking)
    return 0;
  if (gpuBase.orderSensitiveOutputNeeded && !gpuWakeMinParticlesExplicit)
    return gpuParticleCountMeetsThreshold(nParticles, gpuBase.wakeMinParticles);
  return gpuParticleCountAllowed(nParticles, gpuBase.wakeMinParticles);
}

static long gpuScmultAllowed(long nParticles) {
  if (!gpuEnableScmult)
    return 0;
  if (!gpuBase.initialized || gpuBase.activeDevice < 0 || nParticles <= 0)
    return 0;
  if (!gpuParticleCountAllowed(nParticles, gpuBase.scmultMinParticles))
    return 0;
  if (!gpuBase.coord || !gpuBase.coord[0])
    return 0;
  return 1;
}

static long gpuHelperElementNeedsReduction(ELEMENT_LIST *eptr) {
  long ic;
  CENTER *center;

  if (!eptr || !eptr->p_elem)
    return 0;
  switch (eptr->type) {
  case T_ENERGY:
    return 1;
  case T_MATR:
    return !((MATR *)eptr->p_elem)->fiducialSeen;
  case T_EMATRIX:
    return !((EMATRIX *)eptr->p_elem)->fiducialSeen;
  case T_CENTER:
    center = (CENTER *)eptr->p_elem;
    for (ic = 0; ic < 7; ic++) {
      if (center->doCoord[ic] && !center->deltaSet[ic])
        return 1;
    }
    return 0;
  default:
    break;
  }
  return 0;
}

static long gpuPassiveElementSupported(ELEMENT_LIST *eptr, long nParticles) {
  if (!eptr)
    return 0;
  switch (eptr->type) {
  case T_CHARGE:
  case T_FLOORELEMENT:
  case T_MARK:
  case T_MAXAMP:
  case T_RECIRC:
  case T_TSCATTER:
  case T_TRCOUNT:
  case T_WATCH:
    return 1;
  case T_SCMULT:
    return gpuScmultAllowed(nParticles);
  default:
    break;
  }
  return 0;
}

static long gpuElementEligible(ELEMENT_LIST *eptr, long nParticles) {
  if (gpuTrackingSuppressed || !gpuBase.initialized ||
      gpuBase.activeDevice < 0 || !eptr || nParticles <= 0)
    return 0;
  if (gpuSpecialMatrixElementUsesCpuAfterTrack(eptr))
    return 0;
  if (gpuHelperElementSupported(eptr)) {
    if (!gpuParticleCountAllowed(nParticles, gpuBase.helperMinParticles))
      return 0;
    if (gpuHelperElementNeedsReduction(eptr) &&
        !gpuParticleCountAllowed(nParticles, gpuBase.reductionMinParticles))
      return 0;
    return 1;
  }
  if (gpuApertureElementSupported(eptr))
    return gpuParticleCountAllowed(nParticles, gpuBase.apertureMinParticles);
  if (gpuRfcaRemoveInvalidOnlyElementSupported(eptr))
    return gpuParticleCountAllowed(nParticles, gpuBase.apertureMinParticles);
  if (gpuRfcaThinKickElementSupported(eptr))
    return gpuParticleCountAllowed(nParticles, gpuBase.helperMinParticles);
  if (gpuRfcaRfOnlyMatrixElementSupported(eptr))
    return gpuParticleCountAllowed(nParticles, gpuBase.helperMinParticles);
  if (gpuRfcaRfOnlyKickElementSupported(eptr))
    return gpuParticleCountAllowed(nParticles, gpuBase.helperMinParticles);
  if (gpuRfcwRfOnlyElementSupported(eptr))
    return gpuParticleCountAllowed(nParticles, gpuBase.helperMinParticles);
  if (gpuRfcwRfOnlyKickElementSupported(eptr))
    return gpuParticleCountAllowed(nParticles, gpuBase.helperMinParticles);
  if (gpuRfcwMatrixWakeElementSupported(eptr))
    return gpuWakeParticleCountAllowed(nParticles);
  if (gpuRfcwKickWakeElementSupported(eptr))
    return gpuWakeParticleCountAllowed(nParticles);
  if (gpuKickMapElementSupported(eptr))
    return gpuMagnetParticleCountAllowed(nParticles);
  if (gpuCcbendElementSupported(eptr))
    return gpuParticleCountMeetsThreshold(nParticles,
                                          gpuCcbendMinParticles);
  if (gpuCsbendElementSupported(eptr))
    return gpuMagnetParticleCountAllowed(nParticles);
  if (gpuMultipoleElementSupported(eptr))
    return gpuMagnetParticleCountAllowed(nParticles);
  if (gpuPolynomialSeriesElementSupported(eptr))
    return gpuParticleCountMeetsThreshold(nParticles,
                                          gpuBase.magnetMinParticles);
  if (gpuRfdfElementSupported(eptr))
    return gpuParticleCountMeetsThreshold(nParticles,
                                          gpuRfdfMinParticles);
  if (gpuSreffectsElementSupported(eptr))
    return gpuParticleCountMeetsThreshold(nParticles,
                                          gpuSreffectsMinParticles);
  if (gpuBggexpElementSupported(eptr))
    return gpuParticleCountMeetsThreshold(nParticles,
                                          gpuBggexpMinParticles);
  if (gpuCwigglerElementSupported(eptr))
    return gpuParticleCountMeetsThreshold(nParticles,
                                          gpuCwigglerMinParticles);
  if (gpuFtableElementSupported(eptr))
    return gpuParticleCountMeetsThreshold(nParticles,
                                          gpuFtableMinParticles);
  if (gpuBmxyzElementSupported(eptr))
    return gpuParticleCountMeetsThreshold(nParticles,
                                          gpuBmxyzMinParticles);
  if (gpuRfmodeElementSupported(eptr))
    return gpuParticleCountMeetsThreshold(nParticles,
                                          gpuRfmodeMinParticles);
  if (gpuWakeElementSupported(eptr))
    return gpuWakeParticleCountAllowed(nParticles);
  if (gpuTrwakeElementSupported(eptr))
    return gpuWakeParticleCountAllowed(nParticles);
  if (gpuCombinedWakeElementSupported(eptr))
    return gpuWakeParticleCountAllowed(nParticles);
  if (gpuLscElementSupported(eptr))
    return gpuParticleCountAllowed(nParticles, gpuBase.lscMinParticles);
  if (gpuCsrCsbendDeviceEntrySupported(eptr, nParticles))
    return gpuParticleCountAllowed(nParticles, gpuBase.csrMinParticles);
  if (gpuCsrDriftNoOpElementSupported(eptr))
    return gpuParticleCountAllowed(nParticles, gpuBase.csrMinParticles);
  if (gpuSimpleMatrixElement(eptr))
    return gpuCsbendDriftMatrixParticleCountAllowed(nParticles);
  if (gpuExactDriftElement(eptr->type))
    return gpuCsbendDriftExactDriftParticleCountAllowed(nParticles);
  return 0;
}

void gpuSetTrackingSuppressed(long suppressed) {
  gpuTrackingSuppressed = suppressed ? 1 : 0;
}

static long gpuCurrentElementEligible(void) {
  return gpuElementEligible((ELEMENT_LIST *)gpuBase.element, gpuBase.nParticles);
}

static long gpuResidentCsrCsbendFollowsIsland(ELEMENT_LIST *eptr,
                                              long nParticles) {
  CSRCSBEND *csbend;

  if (!gpuEnableCsrResident || !eptr || eptr->type != T_CSRCSBEND ||
      !eptr->p_elem)
    return 0;
  csbend = (CSRCSBEND *)eptr->p_elem;
  return gpu_csr_csbend_resident_available(csbend, nParticles,
                                           csbend->bins);
}

static long gpuCsrCsbendDeviceEntrySupported(ELEMENT_LIST *eptr,
                                             long nParticles) {
  CSRCSBEND *csbend;

  if (!gpuEnableCsrResident || !gpuBase.deviceCurrent ||
      !eptr || eptr->type != T_CSRCSBEND || !eptr->p_elem || nParticles <= 0)
    return 0;
  csbend = (CSRCSBEND *)eptr->p_elem;
  if (csbend->angle == 0)
    return 0;
  if (csbend->dx || csbend->dy || csbend->dz || csbend->etilt)
    return 0;
  if (fabs(sin(csbend->tilt + (csbend->angle < 0 ? PI : 0))) > 1e-14)
    return 0;
  if (csbend->edge_order > 1)
    return 0;
  if ((csbend->edgeFlags & BEND_EDGE1_EFFECTS) &&
      csbend->edge_effects[csbend->e1Index] != 0 &&
      csbend->edge_effects[csbend->e1Index] != 1)
    return 0;
  return gpu_csr_csbend_resident_available(csbend, nParticles,
                                           csbend->bins);
}

static long gpuShouldUseCpuForShortGpuIsland(ELEMENT_LIST *eptr, long nParticles) {
  ELEMENT_LIST *next;
  long elements = 0;

  if (!gpuAvoidShortGpuIslands || gpuShortGpuIslandMaxElements <= 0)
    return 0;
  if (!gpuBase.hostCurrent || !eptr || nParticles <= 0)
    return 0;
  if (!gpuSimpleMatrixElement(eptr))
    return 0;

  for (next = eptr; next; next = next->succ) {
    if (gpuPassiveElementSupported(next, nParticles))
      continue;
    if (gpuSimpleMatrixElement(next) && gpuElementEligible(next, nParticles)) {
      if (++elements > gpuShortGpuIslandMaxElements)
        return 0;
      continue;
    }
    if (gpuResidentCsrCsbendFollowsIsland(next, nParticles))
      return 0;
    if (next == eptr)
      return 0;
    return !gpuElementEligible(next, nParticles);
  }
  return 0;
}

static void gpuRecordMilliseconds(double *seconds, float milliseconds) {
  if (milliseconds >= 0)
    *seconds += milliseconds / 1000.0;
}

static void gpuReleaseDeviceBuffer(void) {
  if (gpuBase.deviceCoord) {
    int status = gpuCudaFree(gpuBase.deviceCoord);
    if (status != 0)
      gpuFatalStatus("cudaFree(coord)", status);
  }
  gpuBase.deviceCoord = NULL;
  gpuBase.deviceCapacity = 0;
  gpuBase.deviceStride = 0;
  gpuBase.deviceCurrent = 0;
  gpuDeviceIslandHasCsbend = 0;
  gpuPendingExactDriftLength = 0;
  gpuPendingExactDriftCount = 0;
  gpuPendingExactDriftParticles = 0;
}

static void gpuReleaseAcceptedBuffer(void) {
  int status;

  if (gpuAcceptedBuffer.coord) {
    status = gpuCudaFree(gpuAcceptedBuffer.coord);
    if (status != 0)
      gpuFatalStatus("cudaFree(accepted coord)", status);
  }
  if (gpuAcceptedBuffer.scratch) {
    status = gpuCudaFree(gpuAcceptedBuffer.scratch);
    if (status != 0)
      gpuFatalStatus("cudaFree(accepted scratch)", status);
  }
  memset(&gpuAcceptedBuffer, 0, sizeof(gpuAcceptedBuffer));
}

static void gpuReleaseKickMapCache(void) {
  int status;

  if (gpuKickMapCache.xpFactor) {
    status = gpuCudaFree(gpuKickMapCache.xpFactor);
    if (status != 0)
      gpuFatalStatus("cudaFree(KICKMAP xpFactor cache)", status);
  }
  if (gpuKickMapCache.ypFactor) {
    status = gpuCudaFree(gpuKickMapCache.ypFactor);
    if (status != 0)
      gpuFatalStatus("cudaFree(KICKMAP ypFactor cache)", status);
  }
  memset(&gpuKickMapCache, 0, sizeof(gpuKickMapCache));
}

static void gpuReleaseRfcwKickScratch(void) {
  int status;

  if (gpuRfcwKickScratch.inverseF) {
    status = gpuCudaFree(gpuRfcwKickScratch.inverseF);
    if (status != 0)
      gpuFatalStatus("cudaFree(RFCW kick inverseF scratch)", status);
  }
  memset(&gpuRfcwKickScratch, 0, sizeof(gpuRfcwKickScratch));
}

static void gpuReleaseLscScratch(void) {
  int status;

  if (gpuLscScratch.result) {
    status = gpuCudaFree(gpuLscScratch.result);
    if (status != 0)
      gpuFatalStatus("cudaFree(LSC reduction scratch)", status);
  }
  if (gpuLscScratch.itime) {
    status = gpuCudaFree(gpuLscScratch.itime);
    if (status != 0)
      gpuFatalStatus("cudaFree(LSC histogram scratch)", status);
  }
  if (gpuLscScratch.binnedCount) {
    status = gpuCudaFree(gpuLscScratch.binnedCount);
    if (status != 0)
      gpuFatalStatus("cudaFree(LSC bin-count scratch)", status);
  }
  if (gpuLscScratch.vtime) {
    status = gpuCudaFree(gpuLscScratch.vtime);
    if (status != 0)
      gpuFatalStatus("cudaFree(LSC voltage scratch)", status);
  }
  memset(&gpuLscScratch, 0, sizeof(gpuLscScratch));
}

static void gpuReleaseBeamSumsScratch(void) {
  int status;

  if (gpuBeamSumsScratch.data) {
    status = gpuCudaFree(gpuBeamSumsScratch.data);
    if (status != 0)
      gpuFatalStatus("cudaFree(beam-sums reduction scratch)", status);
  }
  memset(&gpuBeamSumsScratch, 0, sizeof(gpuBeamSumsScratch));
}

static void gpuReleaseRfcaScratch(void) {
  int status;

  if (gpuRfcaScratch.lostCount) {
    status = gpuCudaFree(gpuRfcaScratch.lostCount);
    if (status != 0)
      gpuFatalStatus("cudaFree(RFCA loss-count scratch)", status);
  }
  if (gpuRfcaScratch.matchEnergy) {
    status = gpuCudaFree(gpuRfcaScratch.matchEnergy);
    if (status != 0)
      gpuFatalStatus("cudaFree(RFCA match-energy scratch)", status);
  }
  memset(&gpuRfcaScratch, 0, sizeof(gpuRfcaScratch));
}

static void gpuReleasePolynomialSeriesCache(void) {
  free(gpuPolynomialSeriesCache.coefficient);
  free(gpuPolynomialSeriesCache.exponent);
  memset(&gpuPolynomialSeriesCache, 0, sizeof(gpuPolynomialSeriesCache));
}

static void gpuReleaseBatchedSearchScratch(void) {
  int status;

  if (gpuBatchedSearchScratch.configured &&
      gpuBatchedSearchScratch.uploaded) {
    float milliseconds = 0;
    unsigned long historyValues =
      (unsigned long)gpuBatchedSearchScratch.particles * 5UL *
      (unsigned long)gpuBatchedSearchScratch.turns;
    if (gpuBatchedSearchScratch.hostHistory && historyValues) {
      status = gpuCudaCopyDeviceToHost(gpuBatchedSearchScratch.hostHistory,
                                       gpuBatchedSearchScratch.deviceHistory,
                                       historyValues, &milliseconds);
      if (status != 0)
        gpuFatalStatus("cudaMemcpy(batched search history device to host)",
                       status);
      gpuRecordMilliseconds(&gpuBase.gpuTransferToHostSeconds, milliseconds);
    }
    if (gpuBatchedSearchScratch.hostHistoryCount &&
        gpuBatchedSearchScratch.particles > 0) {
      milliseconds = 0;
      status = gpuCudaCopyDeviceToHost(
        gpuBatchedSearchScratch.hostHistoryCount,
        gpuBatchedSearchScratch.deviceHistoryCount,
        (unsigned long)gpuBatchedSearchScratch.particles, &milliseconds);
      if (status != 0)
        gpuFatalStatus("cudaMemcpy(batched search turn counts device to host)",
                       status);
      gpuRecordMilliseconds(&gpuBase.gpuTransferToHostSeconds, milliseconds);
    }
  }
  if (gpuBatchedSearchScratch.deviceData) {
    status = gpuCudaFree(gpuBatchedSearchScratch.deviceData);
    if (status != 0)
      gpuFatalStatus("cudaFree(batched search data)", status);
  }
  if (gpuBatchedSearchScratch.deviceHistory) {
    status = gpuCudaFree(gpuBatchedSearchScratch.deviceHistory);
    if (status != 0)
      gpuFatalStatus("cudaFree(batched search history)", status);
  }
  if (gpuBatchedSearchScratch.deviceHistoryCount) {
    status = gpuCudaFree(gpuBatchedSearchScratch.deviceHistoryCount);
    if (status != 0)
      gpuFatalStatus("cudaFree(batched search turn counts)", status);
  }
  free(gpuBatchedSearchScratch.hostData);
  memset(&gpuBatchedSearchScratch, 0, sizeof(gpuBatchedSearchScratch));
}

static void gpuReleaseApertureScratch(void) {
  int status;

  if (gpuApertureScratch.coord) {
    status = gpuCudaFree(gpuApertureScratch.coord);
    if (status != 0)
      gpuFatalStatus("cudaFree(aperture coord scratch)", status);
  }
  if (gpuApertureScratch.prefix) {
    status = gpuCudaFree(gpuApertureScratch.prefix);
    if (status != 0)
      gpuFatalStatus("cudaFree(aperture prefix scratch)", status);
  }
  if (gpuApertureScratch.hostPrefix)
    free(gpuApertureScratch.hostPrefix);
  if (gpuApertureScratch.hostAccepted)
    free(gpuApertureScratch.hostAccepted);
  memset(&gpuApertureScratch, 0, sizeof(gpuApertureScratch));
}

static void gpuReleaseCsrWakeScratch(void) {
  int status;

  if (gpuCsrScratch.ctHist) {
    status = gpuCudaFree(gpuCsrScratch.ctHist);
    if (status != 0)
      gpuFatalStatus("cudaFree(CSR ctHist)", status);
  }
  if (gpuCsrScratch.wakeInput) {
    status = gpuCudaFree(gpuCsrScratch.wakeInput);
    if (status != 0)
      gpuFatalStatus("cudaFree(CSR wake input)", status);
  }
  if (gpuCsrScratch.denom) {
    status = gpuCudaFree(gpuCsrScratch.denom);
    if (status != 0)
      gpuFatalStatus("cudaFree(CSR denom)", status);
  }
  if (gpuCsrScratch.T1) {
    status = gpuCudaFree(gpuCsrScratch.T1);
    if (status != 0)
      gpuFatalStatus("cudaFree(CSR T1)", status);
  }
  if (gpuCsrScratch.T2) {
    status = gpuCudaFree(gpuCsrScratch.T2);
    if (status != 0)
      gpuFatalStatus("cudaFree(CSR T2)", status);
  }
  if (gpuCsrScratch.dGamma) {
    status = gpuCudaFree(gpuCsrScratch.dGamma);
    if (status != 0)
      gpuFatalStatus("cudaFree(CSR dGamma)", status);
  }
  gpuCsrScratch.ctHist = NULL;
  gpuCsrScratch.wakeInput = NULL;
  gpuCsrScratch.denom = NULL;
  gpuCsrScratch.T1 = NULL;
  gpuCsrScratch.T2 = NULL;
  gpuCsrScratch.dGamma = NULL;
  gpuCsrScratch.capacity = 0;
  gpuCsrScratch.dGammaValid = 0;
  gpuCsrScratch.dGammaBins = 0;
  gpuCsrScratch.denomValid = 0;
  gpuCsrScratch.denomBins = 0;
  gpuCsrScratch.denomDct = 0;
}

static void gpuReleaseCsrScratch(void) {
  int status;

  gpuReleaseCsrWakeScratch();
  if (gpuCsrScratch.kickDp) {
    status = gpuCudaFree(gpuCsrScratch.kickDp);
    if (status != 0)
      gpuFatalStatus("cudaFree(CSR kick dp)", status);
  }
  if (gpuCsrScratch.bodyBackup) {
    status = gpuCudaFree(gpuCsrScratch.bodyBackup);
    if (status != 0)
      gpuFatalStatus("cudaFree(CSR body backup)", status);
  }
  if (gpuCsrScratch.bodyLostCount) {
    status = gpuCudaFree(gpuCsrScratch.bodyLostCount);
    if (status != 0)
      gpuFatalStatus("cudaFree(CSR body lost count)", status);
  }
  if (gpuCsrScratch.rangeResult) {
    status = gpuCudaFree(gpuCsrScratch.rangeResult);
    if (status != 0)
      gpuFatalStatus("cudaFree(CSR range result)", status);
  }
  if (gpuCsrScratch.hostWakeInput)
    free(gpuCsrScratch.hostWakeInput);
  if (gpuCsrScratch.cpuDGamma)
    free(gpuCsrScratch.cpuDGamma);
  memset(&gpuCsrScratch, 0, sizeof(gpuCsrScratch));
}

static void gpuEnsureRfcwKickScratch(long nParticles) {
  int status;

  if (nParticles <= 0)
    return;
  if (gpuRfcwKickScratch.inverseF &&
      gpuRfcwKickScratch.capacity >= nParticles)
    return;
  gpuReleaseRfcwKickScratch();
  status = gpuCudaMallocDouble(&gpuRfcwKickScratch.inverseF,
                               (unsigned long)nParticles);
  if (status != 0)
    gpuFatalStatus("cudaMalloc(RFCW kick inverseF scratch)", status);
  gpuRfcwKickScratch.capacity = nParticles;
}

static void gpuEnsureLscScratch(long bins) {
  int status;

  if (bins <= 0)
    return;
  if (gpuLscScratch.result && gpuLscScratch.itime &&
      gpuLscScratch.binnedCount && gpuLscScratch.vtime &&
      gpuLscScratch.binsCapacity >= bins)
    return;
  gpuReleaseLscScratch();
  status = gpuCudaMallocBytes(&gpuLscScratch.result,
                              (unsigned long)sizeof(GPU_BEAM_SUM_DATA));
  if (status != 0)
    gpuFatalStatus("cudaMalloc(LSC reduction scratch)", status);
  status = gpuCudaMallocDouble(&gpuLscScratch.itime, (unsigned long)bins);
  if (status != 0)
    gpuFatalStatus("cudaMalloc(LSC histogram scratch)", status);
  status = gpuCudaMallocBytes(&gpuLscScratch.binnedCount,
                              (unsigned long)sizeof(unsigned long long));
  if (status != 0)
    gpuFatalStatus("cudaMalloc(LSC bin-count scratch)", status);
  status = gpuCudaMallocDouble(&gpuLscScratch.vtime,
                               (unsigned long)(bins + 1));
  if (status != 0)
    gpuFatalStatus("cudaMalloc(LSC voltage scratch)", status);
  gpuLscScratch.binsCapacity = bins;
}

static void gpuEnsureBeamSumsScratch(void) {
  int status;

  if (gpuBeamSumsScratch.data)
    return;
  status = gpuCudaMallocBytes(&gpuBeamSumsScratch.data,
                              gpuCudaBeamSums2ScratchBytes());
  if (status != 0)
    gpuFatalStatus("cudaMalloc(beam-sums reduction scratch)", status);
}

static void gpuEnsureRfcaScratch(void) {
  int status;

  if (!gpuRfcaScratch.lostCount) {
    status = gpuCudaMallocBytes(&gpuRfcaScratch.lostCount, sizeof(long));
    if (status != 0)
      gpuFatalStatus("cudaMalloc(RFCA loss-count scratch)", status);
  }
  if (!gpuRfcaScratch.matchEnergy) {
    status = gpuCudaMallocBytes(&gpuRfcaScratch.matchEnergy,
                                gpuCudaMatchEnergyScratchBytes());
    if (status != 0)
      gpuFatalStatus("cudaMalloc(RFCA match-energy scratch)", status);
  }
}

#ifndef GPU_VERIFY
static void gpuEnsureApertureScratch(long nParticles) {
  unsigned long count;
  int status;

  if (nParticles <= 0)
    return;
  if (gpuApertureScratch.coord &&
      gpuApertureScratch.prefix &&
      gpuApertureScratch.capacity >= nParticles &&
      gpuApertureScratch.prefixCapacity >= nParticles &&
      gpuApertureScratch.stride == gpuBase.deviceStride)
    return;
  gpuReleaseApertureScratch();
  count = (unsigned long)nParticles * (unsigned long)gpuBase.deviceStride;
  status = gpuCudaMallocDouble(&gpuApertureScratch.coord, count);
  if (status != 0)
    gpuFatalStatus("cudaMalloc(aperture coord scratch)", status);
  status = gpuCudaMallocBytes(&gpuApertureScratch.prefix,
                              (unsigned long)nParticles * sizeof(long));
  if (status != 0)
    gpuFatalStatus("cudaMalloc(aperture prefix scratch)", status);
  gpuApertureScratch.capacity = nParticles;
  gpuApertureScratch.prefixCapacity = nParticles;
  gpuApertureScratch.stride = gpuBase.deviceStride;
}

static void gpuPromoteApertureScratchCoord(void) {
  void *oldCoord;
  long oldCapacity, oldStride;

  if (!gpuBase.deviceCoord || !gpuApertureScratch.coord)
    gpuRequiredFailure("missing CUDA aperture coordinate buffer for promotion");
  oldCoord = gpuBase.deviceCoord;
  oldCapacity = gpuBase.deviceCapacity;
  oldStride = gpuBase.deviceStride;

  /* Keep the compacted coordinate buffer resident; the prefix scratch has
   * independent capacity and remains paired with gpuApertureScratch.prefix. */
  gpuBase.deviceCoord = gpuApertureScratch.coord;
  gpuBase.deviceCapacity = gpuApertureScratch.capacity;
  gpuBase.deviceStride = gpuApertureScratch.stride;

  gpuApertureScratch.coord = oldCoord;
  gpuApertureScratch.capacity = oldCapacity;
  gpuApertureScratch.stride = oldStride;
}

static long *gpuEnsureApertureHostPrefix(long nParticles) {
  if (nParticles <= 0)
    return NULL;
  if (gpuApertureScratch.hostPrefixCapacity >= nParticles)
    return gpuApertureScratch.hostPrefix;
  gpuApertureScratch.hostPrefix =
    SDDS_Realloc(gpuApertureScratch.hostPrefix,
                 (size_t)nParticles * sizeof(*gpuApertureScratch.hostPrefix));
  if (!gpuApertureScratch.hostPrefix)
    gpuRequiredFailure("memory allocation failure for aperture host prefix scratch");
  gpuApertureScratch.hostPrefixCapacity = nParticles;
  return gpuApertureScratch.hostPrefix;
}

static double *gpuEnsureApertureHostAccepted(long nParticles, long stride) {
  long needed;

  if (nParticles <= 0 || stride <= 0)
    return NULL;
  needed = nParticles * stride;
  if (gpuApertureScratch.hostAcceptedCapacity >= needed)
    return gpuApertureScratch.hostAccepted;
  gpuApertureScratch.hostAccepted =
    SDDS_Realloc(gpuApertureScratch.hostAccepted,
                 (size_t)needed * sizeof(*gpuApertureScratch.hostAccepted));
  if (!gpuApertureScratch.hostAccepted)
    gpuRequiredFailure("memory allocation failure for aperture accepted scratch");
  gpuApertureScratch.hostAcceptedCapacity = needed;
  return gpuApertureScratch.hostAccepted;
}
#endif

static void gpuEnsureCsrScratch(long nBins) {
  unsigned long count;
  int status;

  if (nBins <= 0)
    return;
  if (gpuCsrScratch.capacity >= nBins)
    return;
  gpuReleaseCsrWakeScratch();
  count = (unsigned long)nBins;
  status = gpuCudaMallocDouble(&gpuCsrScratch.ctHist, count);
  if (status != 0)
    gpuFatalStatus("cudaMalloc(CSR ctHist)", status);
  status = gpuCudaMallocDouble(&gpuCsrScratch.wakeInput, 2 * count);
  if (status != 0)
    gpuFatalStatus("cudaMalloc(CSR wake input)", status);
  status = gpuCudaMallocDouble(&gpuCsrScratch.denom, count);
  if (status != 0)
    gpuFatalStatus("cudaMalloc(CSR denom)", status);
  status = gpuCudaMallocDouble(&gpuCsrScratch.T1, count);
  if (status != 0)
    gpuFatalStatus("cudaMalloc(CSR T1)", status);
  status = gpuCudaMallocDouble(&gpuCsrScratch.T2, count);
  if (status != 0)
    gpuFatalStatus("cudaMalloc(CSR T2)", status);
  status = gpuCudaMallocDouble(&gpuCsrScratch.dGamma, count);
  if (status != 0)
    gpuFatalStatus("cudaMalloc(CSR dGamma)", status);
  gpuCsrScratch.capacity = nBins;
  gpuCsrScratch.dGammaValid = 0;
  gpuCsrScratch.dGammaBins = 0;
  gpuCsrScratch.denomValid = 0;
  gpuCsrScratch.denomBins = 0;
  gpuCsrScratch.denomDct = 0;
}

static void gpuUploadCsrWakeInputs(const double *ctHist,
                                   const double *ctHistDeriv,
                                   long nBins,
                                   const double **deviceCtHist,
                                   const double **deviceCtHistDeriv) {
  double *buffer;
  float milliseconds = 0;
  unsigned long count, totalCount;
  int status;

  if (!ctHist || !ctHistDeriv || !deviceCtHist || !deviceCtHistDeriv ||
      nBins <= 0)
    gpuRequiredFailure("invalid CUDA CSR wake input upload request");
  count = (unsigned long)nBins;
  totalCount = 2 * count;
  if (gpuCsrScratch.hostWakeInputCapacity < (long)totalCount) {
    buffer = realloc(gpuCsrScratch.hostWakeInput,
                     (size_t)totalCount * sizeof(*gpuCsrScratch.hostWakeInput));
    if (!buffer)
      gpuRequiredFailure("unable to allocate CUDA CSR wake input scratch");
    gpuCsrScratch.hostWakeInput = buffer;
    gpuCsrScratch.hostWakeInputCapacity = (long)totalCount;
  }
  memcpy(gpuCsrScratch.hostWakeInput, ctHist,
         (size_t)count * sizeof(*gpuCsrScratch.hostWakeInput));
  memcpy(gpuCsrScratch.hostWakeInput + count, ctHistDeriv,
         (size_t)count * sizeof(*gpuCsrScratch.hostWakeInput));
  status = gpuCudaCopyHostToDevice(gpuCsrScratch.wakeInput,
                                   gpuCsrScratch.hostWakeInput,
                                   totalCount, &milliseconds);
  if (status != 0)
    gpuFatalStatus("cudaMemcpy(CSR wake inputs host to device)", status);
  gpuRecordMilliseconds(&gpuBase.gpuTransferToDeviceSeconds, milliseconds);
  *deviceCtHist = (const double *)gpuCsrScratch.wakeInput;
  *deviceCtHistDeriv = ((const double *)gpuCsrScratch.wakeInput) + count;
}

static void gpuUploadCsrDenomIfNeeded(const double *denom, long nBins,
                                      double dct) {
  float milliseconds = 0;
  unsigned long count;
  int status;

  if (!denom || nBins <= 0)
    gpuRequiredFailure("invalid CUDA CSR denominator upload request");
  if (gpuCsrScratch.denomValid &&
      gpuCsrScratch.denomBins == nBins &&
      gpuCsrScratch.denomDct == dct)
    return;
  count = (unsigned long)nBins;
  status = gpuCudaCopyHostToDevice(gpuCsrScratch.denom, denom, count,
                                   &milliseconds);
  if (status != 0)
    gpuFatalStatus("cudaMemcpy(CSR denom host to device)", status);
  gpuRecordMilliseconds(&gpuBase.gpuTransferToDeviceSeconds, milliseconds);
  gpuCsrScratch.denomValid = 1;
  gpuCsrScratch.denomBins = nBins;
  gpuCsrScratch.denomDct = dct;
}

static void gpuEnsureCsrKickDpScratch(long nParticles) {
  unsigned long count;
  int status;

  if (nParticles <= 0)
    return;
  if (gpuCsrScratch.kickDp && gpuCsrScratch.kickDpCapacity >= nParticles)
    return;
  if (gpuCsrScratch.kickDp) {
    status = gpuCudaFree(gpuCsrScratch.kickDp);
    if (status != 0)
      gpuFatalStatus("cudaFree(CSR kick dp)", status);
    gpuCsrScratch.kickDp = NULL;
    gpuCsrScratch.kickDpCapacity = 0;
  }
  count = (unsigned long)nParticles;
  status = gpuCudaMallocDouble(&gpuCsrScratch.kickDp, count);
  if (status != 0)
    gpuFatalStatus("cudaMalloc(CSR kick dp)", status);
  gpuCsrScratch.kickDpCapacity = nParticles;
}

static void gpuEnsureCsrBodyScratch(long nParticles, long stride) {
  unsigned long count;
  int status;

  if (stride <= 0)
    gpuRequiredFailure("invalid CUDA CSR body scratch stride");
  if (nParticles <= 0)
    return;
  if (gpuCsrScratch.bodyBackup &&
      gpuCsrScratch.bodyLostCount &&
      gpuCsrScratch.bodyBackupCapacity >= nParticles &&
      gpuCsrScratch.bodyBackupStride == stride)
    return;
  if (gpuCsrScratch.bodyBackup) {
    status = gpuCudaFree(gpuCsrScratch.bodyBackup);
    if (status != 0)
      gpuFatalStatus("cudaFree(CSR body backup)", status);
    gpuCsrScratch.bodyBackup = NULL;
    gpuCsrScratch.bodyBackupCapacity = 0;
    gpuCsrScratch.bodyBackupStride = 0;
  }
  if (gpuCsrScratch.bodyLostCount) {
    status = gpuCudaFree(gpuCsrScratch.bodyLostCount);
    if (status != 0)
      gpuFatalStatus("cudaFree(CSR body lost count)", status);
    gpuCsrScratch.bodyLostCount = NULL;
  }
  count = (unsigned long)nParticles * (unsigned long)stride;
  status = gpuCudaMallocDouble(&gpuCsrScratch.bodyBackup, count);
  if (status != 0)
    gpuFatalStatus("cudaMalloc(CSR body backup)", status);
  status = gpuCudaMallocBytes(&gpuCsrScratch.bodyLostCount,
                              (unsigned long)sizeof(unsigned long long));
  if (status != 0)
    gpuFatalStatus("cudaMalloc(CSR body lost count)", status);
  gpuCsrScratch.bodyBackupCapacity = nParticles;
  gpuCsrScratch.bodyBackupStride = stride;
}

void gpu_clear_csr_wake_cpu_shadow(void) {
  gpuCsrScratch.cpuDGammaValid = 0;
  gpuCsrScratch.cpuDGammaBins = 0;
}

long gpu_copy_csr_wake_cpu_shadow(double *dGamma, long nBins) {
  if (!dGamma || nBins <= 0 || !gpuCsrScratch.cpuDGamma ||
      !gpuCsrScratch.cpuDGammaValid || gpuCsrScratch.cpuDGammaBins != nBins)
    return 0;
  memcpy(dGamma, gpuCsrScratch.cpuDGamma, (size_t)nBins * sizeof(*dGamma));
  return 1;
}

#ifdef GPU_VERIFY
static void gpuStoreCsrWakeCpuShadow(const double *dGamma, long nBins) {
  double *buffer;

  if (!dGamma || nBins <= 0) {
    gpu_clear_csr_wake_cpu_shadow();
    return;
  }
  if (gpuCsrScratch.cpuDGammaCapacity < nBins) {
    buffer = realloc(gpuCsrScratch.cpuDGamma,
                     (size_t)nBins * sizeof(*gpuCsrScratch.cpuDGamma));
    if (!buffer)
      gpuRequiredFailure("unable to allocate CUDA CSRCSBEND CPU-shadow dGamma buffer");
    gpuCsrScratch.cpuDGamma = buffer;
    gpuCsrScratch.cpuDGammaCapacity = nBins;
  }
  memcpy(gpuCsrScratch.cpuDGamma, dGamma, (size_t)nBins * sizeof(*dGamma));
  gpuCsrScratch.cpuDGammaBins = nBins;
  gpuCsrScratch.cpuDGammaValid = 1;
}
#endif

static void gpuEnsureDeviceBuffer(long nParticles) {
  long stride = totalPropertiesPerParticle;
  unsigned long count;
  int status;

  if (stride < 7)
    gpuRequiredFailure("particle stride is smaller than the six phase-space coordinates plus particle ID");
  if (nParticles <= 0)
    return;
  if (gpuBase.deviceCoord && gpuBase.deviceCapacity >= nParticles && gpuBase.deviceStride == stride)
    return;
  gpuReleaseDeviceBuffer();
  free(gpuBunchRangeCache.start);
  free(gpuBunchRangeCache.count);
  memset(&gpuBunchRangeCache, 0, sizeof(gpuBunchRangeCache));
  count = (unsigned long)nParticles * (unsigned long)stride;
  status = gpuCudaMallocDouble(&gpuBase.deviceCoord, count);
  if (status != 0)
    gpuFatalStatus("cudaMalloc(coord)", status);
  gpuBase.deviceCapacity = nParticles;
  gpuBase.deviceStride = stride;
}

static void gpuNoticeHostBaseChange(void) {
  void *hostBase = gpuBase.coord ? (void *)gpuBase.coord[0] : NULL;

  if (hostBase != gpuBase.hostCoordBase) {
    gpuBase.hostCoordBase = hostBase;
    gpuBase.deviceCurrent = 0;
    gpuBase.hostCurrent = 1;
    gpuDeviceIslandHasCsbend = 0;
  }
}

static void gpuCopyHostToDeviceInternal(long nParticles, long flushPending) {
  float milliseconds = 0;
  unsigned long count;
  int status;

  if (nParticles <= 0)
    return;
  gpuNoticeHostBaseChange();
  gpuEnsureDeviceBuffer(nParticles);
  if (gpuBase.deviceCurrent) {
    if (flushPending)
      gpuFlushPendingExactDrift("pending exactDrift before CUDA coordinate use");
    return;
  }
  count = (unsigned long)nParticles * (unsigned long)gpuBase.deviceStride;
  status = gpuCudaCopyHostToDevice(gpuBase.deviceCoord, gpuBase.coord[0], count, &milliseconds);
  if (status != 0)
    gpuFatalStatus("cudaMemcpy(host to device coord)", status);
  gpuRecordMilliseconds(&gpuBase.gpuTransferToDeviceSeconds, milliseconds);
  gpuBase.deviceCurrent = 1;
  gpuBase.hostCurrent = 1;
  gpuBase.nParticles = nParticles;
}

static void gpuCopyHostToDevice(long nParticles) {
  gpuCopyHostToDeviceInternal(nParticles, 1);
}

static long gpuCopyDeviceToHost(long nParticles) {
  float milliseconds = 0;
  unsigned long count;
  int status;

  gpuFlushPendingExactDrift("pending exactDrift before CPU synchronization");
  if (!gpuBase.deviceCoord || !gpuBase.deviceCurrent || gpuBase.hostCurrent || nParticles <= 0)
    return 0;
  if (!gpuBase.coord || !gpuBase.coord[0])
    gpuRequiredFailure("host particle array is unavailable for CUDA synchronization");
  count = (unsigned long)nParticles * (unsigned long)gpuBase.deviceStride;
  status = gpuCudaCopyDeviceToHost(gpuBase.coord[0], gpuBase.deviceCoord, count, &milliseconds);
  if (status != 0)
    gpuFatalStatus("cudaMemcpy(device to host coord)", status);
  gpuRecordMilliseconds(&gpuBase.gpuTransferToHostSeconds, milliseconds);
  gpuBase.hostCurrent = 1;
  return 1;
}

#ifndef GPU_VERIFY
static void gpuAcceptedNoticeHostBaseChange(double **accepted) {
  void *hostBase = accepted ? (void *)accepted[0] : NULL;

  if (gpuAcceptedBuffer.hostAccepted == accepted &&
      gpuAcceptedBuffer.hostBase == hostBase)
    return;
  gpuReleaseAcceptedBuffer();
  gpuAcceptedBuffer.hostAccepted = accepted;
  gpuAcceptedBuffer.hostBase = hostBase;
  gpuAcceptedBuffer.hostCurrent = accepted && hostBase;
}
#endif

static long gpuCopyAcceptedDeviceToHost(long nParticles, const char *reason) {
  float milliseconds = 0;
  unsigned long count;
  int status;

  (void)reason;
  if (!gpuAcceptedBuffer.coord || !gpuAcceptedBuffer.deviceCurrent ||
      gpuAcceptedBuffer.hostCurrent)
    return 0;
  if (!gpuAcceptedBuffer.hostAccepted || !gpuAcceptedBuffer.hostAccepted[0])
    gpuRequiredFailure("host accepted array is unavailable for CUDA synchronization");
  if (nParticles <= 0)
    nParticles = gpuAcceptedBuffer.nParticles;
  if (nParticles <= 0)
    return 0;
  if (nParticles > gpuAcceptedBuffer.capacity)
    gpuRequiredFailure("invalid CUDA accepted synchronization range");
  count = (unsigned long)nParticles * (unsigned long)gpuAcceptedBuffer.stride;
  status = gpuCudaCopyDeviceToHost(gpuAcceptedBuffer.hostAccepted[0],
                                   gpuAcceptedBuffer.coord, count,
                                   &milliseconds);
  if (status != 0)
    gpuFatalStatus("cudaMemcpy(device accepted to host)", status);
  gpuRecordMilliseconds(&gpuBase.gpuTransferToHostSeconds, milliseconds);
  gpuAcceptedBuffer.hostCurrent = 1;
  gpuAcceptedBuffer.nParticles = nParticles;
  return 1;
}

#ifndef GPU_VERIFY
static void gpuEnsureAcceptedDevice(double **accepted, long nParticles,
                                    long stride) {
  float milliseconds = 0;
  unsigned long count;
  int status;

  if (!accepted || !accepted[0] || nParticles <= 0)
    return;
  if (stride <= 0)
    gpuRequiredFailure("invalid CUDA accepted buffer stride");
  gpuAcceptedNoticeHostBaseChange(accepted);
  if (!gpuAcceptedBuffer.coord || !gpuAcceptedBuffer.scratch ||
      gpuAcceptedBuffer.capacity < nParticles ||
      gpuAcceptedBuffer.scratchCapacity < nParticles ||
      gpuAcceptedBuffer.stride != stride) {
    if (gpuAcceptedBuffer.deviceCurrent && !gpuAcceptedBuffer.hostCurrent)
      gpuCopyAcceptedDeviceToHost(gpuAcceptedBuffer.nParticles,
                                  "CUDA accepted buffer resize");
    if (gpuAcceptedBuffer.coord) {
      status = gpuCudaFree(gpuAcceptedBuffer.coord);
      if (status != 0)
        gpuFatalStatus("cudaFree(accepted coord)", status);
      gpuAcceptedBuffer.coord = NULL;
    }
    if (gpuAcceptedBuffer.scratch) {
      status = gpuCudaFree(gpuAcceptedBuffer.scratch);
      if (status != 0)
        gpuFatalStatus("cudaFree(accepted scratch)", status);
      gpuAcceptedBuffer.scratch = NULL;
    }
    count = (unsigned long)nParticles * (unsigned long)stride;
    status = gpuCudaMallocDouble(&gpuAcceptedBuffer.coord, count);
    if (status != 0)
      gpuFatalStatus("cudaMalloc(accepted coord)", status);
    status = gpuCudaMallocDouble(&gpuAcceptedBuffer.scratch, count);
    if (status != 0)
      gpuFatalStatus("cudaMalloc(accepted scratch)", status);
    gpuAcceptedBuffer.capacity = nParticles;
    gpuAcceptedBuffer.scratchCapacity = nParticles;
    gpuAcceptedBuffer.stride = stride;
    gpuAcceptedBuffer.deviceCurrent = 0;
    gpuAcceptedBuffer.hostCurrent = 1;
  }
  if (gpuAcceptedBuffer.deviceCurrent)
    return;
  if (!gpuAcceptedBuffer.hostCurrent)
    gpuRequiredFailure("CUDA accepted host copy is stale before device upload");
  count = (unsigned long)nParticles * (unsigned long)stride;
  status = gpuCudaCopyHostToDevice(gpuAcceptedBuffer.coord, accepted[0],
                                   count, &milliseconds);
  if (status != 0)
    gpuFatalStatus("cudaMemcpy(host accepted to device)", status);
  gpuRecordMilliseconds(&gpuBase.gpuTransferToDeviceSeconds, milliseconds);
  gpuAcceptedBuffer.deviceCurrent = 1;
  gpuAcceptedBuffer.hostCurrent = 1;
  gpuAcceptedBuffer.nParticles = nParticles;
}
#endif

#ifndef GPU_VERIFY
static long gpuCopyDeviceRowsToHost(long startParticle, long rowCount,
                                    const char *reason) {
  double *deviceStart;
  float milliseconds = 0;
  unsigned long count;
  int status;

  if (rowCount <= 0)
    return 0;
  if (!gpuBase.deviceCoord || !gpuBase.deviceCurrent)
    return 0;
  if (!gpuBase.coord || !gpuBase.coord[0])
    gpuRequiredFailure("host particle array is unavailable for CUDA row synchronization");
  if (startParticle < 0 || startParticle + rowCount > gpuBase.deviceCapacity)
    gpuRequiredFailure("invalid CUDA particle row synchronization range");
  deviceStart = ((double *)gpuBase.deviceCoord) +
                (unsigned long)startParticle * (unsigned long)gpuBase.deviceStride;
  count = (unsigned long)rowCount * (unsigned long)gpuBase.deviceStride;
  status = gpuCudaCopyDeviceToHost(gpuBase.coord[startParticle],
                                   deviceStart, count, &milliseconds);
  if (status != 0)
    gpuFatalStatus("cudaMemcpy(device row range to host coord)", status);
  gpuRecordMilliseconds(&gpuBase.gpuTransferToHostSeconds, milliseconds);
  gpuRecordSyncRequest(reason, 1, 1);
  if (gpuVerbose && reason)
    fprintf(stderr, "elegant CUDA: read-only CPU row synchronization requested by %s (%ld row%s).\n",
            reason, rowCount, rowCount == 1 ? "" : "s");
  return 1;
}

static void gpuCopyAperturePrefixToHost(long nParticles, const char *reason) {
  long *hostPrefix;
  float milliseconds = 0;
  int status;

  if (nParticles <= 0)
    return;
  if (!gpuApertureScratch.prefix)
    gpuRequiredFailure("missing CUDA aperture prefix scratch");
  hostPrefix = gpuEnsureApertureHostPrefix(nParticles);
  status = gpuCudaCopyDeviceBytesToHost(hostPrefix, gpuApertureScratch.prefix,
                                        (unsigned long)nParticles * sizeof(*hostPrefix),
                                        &milliseconds);
  if (status != 0)
    gpuFatalStatus("cudaMemcpy(aperture prefix to host)", status);
  gpuRecordMilliseconds(&gpuBase.gpuTransferToHostSeconds, milliseconds);
  gpuRecordSyncRequest(reason, 1, 1);
}

static long gpuAperturePrefixSurvives(const long *prefix, long ip,
                                      long nParticles, long survivors) {
  if (ip + 1 < nParticles)
    return prefix[ip + 1] > prefix[ip];
  return survivors > prefix[ip];
}

static void gpuStablePartitionAccepted(double **accepted, long nParticles,
                                       long survivors) {
  long *prefix;
  double *scratch;
  long stride = totalPropertiesPerParticle;
  long ip, destination;

  if (!accepted || nParticles <= 0)
    return;
  prefix = gpuApertureScratch.hostPrefix;
  scratch = gpuEnsureApertureHostAccepted(nParticles, stride);
  if (!prefix || !scratch)
    gpuRequiredFailure("missing host aperture scratch for accepted partition");
  for (ip = 0; ip < nParticles; ip++) {
    if (gpuAperturePrefixSurvives(prefix, ip, nParticles, survivors))
      destination = prefix[ip];
    else
      destination = survivors + (ip - prefix[ip]);
    memcpy(scratch + destination * stride, accepted[ip],
           (size_t)stride * sizeof(*scratch));
  }
  memcpy(accepted[0], scratch, (size_t)nParticles * (size_t)stride * sizeof(**accepted));
}

static long gpuStablePartitionAcceptedOnDevice(double **accepted,
                                               long nParticles,
                                               long survivors,
                                               const char *reason) {
  void *oldCoord;
  long stride = totalPropertiesPerParticle;
  float milliseconds = 0;
  int status;

  if (!gpuEnableApertureAcceptedDevice || !accepted || nParticles <= 0)
    return 0;
  if (!gpuApertureScratch.prefix)
    gpuRequiredFailure("missing CUDA aperture prefix scratch for accepted partition");
  gpuEnsureAcceptedDevice(accepted, nParticles, stride);
  if (!gpuAcceptedBuffer.coord || !gpuAcceptedBuffer.scratch)
    gpuRequiredFailure("missing CUDA accepted buffers for stable partition");
  status = gpuCudaStableScatterRows(gpuAcceptedBuffer.coord,
                                    gpuAcceptedBuffer.scratch,
                                    gpuApertureScratch.prefix,
                                    nParticles, (int)stride, survivors,
                                    &milliseconds);
  if (status != 0)
    gpuFatalStatus(reason ? reason : "accepted CUDA stable compaction", status);
  gpuRecordMilliseconds(&gpuBase.gpuKernelSeconds, milliseconds);

  oldCoord = gpuAcceptedBuffer.coord;
  gpuAcceptedBuffer.coord = gpuAcceptedBuffer.scratch;
  gpuAcceptedBuffer.scratch = oldCoord;
  gpuAcceptedBuffer.nParticles = survivors;
  gpuAcceptedBuffer.deviceCurrent = 1;
  gpuAcceptedBuffer.hostCurrent = 0;
  return 1;
}

static long gpuApertureLossRowsNeedHost(void) {
  return gpuBase.lossOutputNeeded;
}

#ifndef GPU_VERIFY
static void gpuMarkDeviceChanged(long nParticles);
static void gpuRecordMagnetKernel(float milliseconds);

static long gpuMagnetLossCompactionAllowed(void) {
  return gpuEnableMagnetLossCompaction &&
         !gpuApertureLossRowsNeedHost() && globalLossCoordOffset <= 0;
}

static long gpuMultipoleStableCompact(GPU_MULTIPOLE_DATA *data,
                                      long nParticles, double **accepted,
                                      const char *name) {
  long remaining = nParticles;
  float milliseconds = 0;
  int status;

  if (nParticles <= 0)
    return nParticles;
  gpuEnsureApertureScratch(nParticles);
  status = gpuCudaMultipoleTrackStableCompact(
    gpuBase.deviceCoord, gpuApertureScratch.coord, gpuApertureScratch.prefix,
    nParticles, (int)gpuBase.deviceStride, data, &remaining, &milliseconds);
  if (status != 0)
    gpuFatalStatus(name ? name : "multipole stable magnet loss compaction",
                   status);
  gpuRecordMagnetKernel(milliseconds);
  if (accepted) {
    if (!gpuStablePartitionAcceptedOnDevice(
          accepted, nParticles, remaining,
          "multipole CUDA accepted stable compaction")) {
      gpuCopyAperturePrefixToHost(
        nParticles, "multipole CUDA stable compaction accepted map");
      gpuStablePartitionAccepted(accepted, nParticles, remaining);
    }
  }
  gpuPromoteApertureScratchCoord();
  gpuMarkDeviceChanged(remaining);
  gpuRecordWallSeconds();
  return remaining;
}

static long gpuCsbendStableCompact(GPU_CSBEND_DATA *data, long nParticles,
                                   double **accepted, const char *name) {
  long remaining = nParticles;
  float milliseconds = 0;
  int status;

  if (nParticles <= 0)
    return nParticles;
  gpuEnsureApertureScratch(nParticles);
  status = gpuCudaCsbendTrackStableCompact(
    gpuBase.deviceCoord, gpuApertureScratch.coord, gpuApertureScratch.prefix,
    nParticles, (int)gpuBase.deviceStride, data, &remaining, &milliseconds);
  if (status != 0)
    gpuFatalStatus(name ? name : "CSBEND stable magnet loss compaction",
                   status);
  gpuRecordMagnetKernel(milliseconds);
  if (accepted) {
    if (!gpuStablePartitionAcceptedOnDevice(
          accepted, nParticles, remaining,
          "CSBEND CUDA accepted stable compaction")) {
      gpuCopyAperturePrefixToHost(
        nParticles, "CSBEND CUDA stable compaction accepted map");
      gpuStablePartitionAccepted(accepted, nParticles, remaining);
    }
  }
  gpuPromoteApertureScratchCoord();
  gpuMarkDeviceChanged(remaining);
  gpuRecordWallSeconds();
  return remaining;
}

static long gpuKickMapStableCompact(GPU_KICKMAP_DATA *data, long nParticles,
                                    double **accepted, double zStart,
                                    double pRef, const double *xpFactor,
                                    const double *ypFactor, long points,
                                    const char *name) {
  long remaining = nParticles;
  float milliseconds = 0;
  int status;

  if (nParticles <= 0)
    return nParticles;
  gpuEnsureApertureScratch(nParticles);
  status = gpuCudaKickMapTrackStableCompact(
    gpuBase.deviceCoord, gpuApertureScratch.coord, gpuApertureScratch.prefix,
    nParticles, (int)gpuBase.deviceStride, data, xpFactor, ypFactor, points,
    zStart, pRef, &remaining, &milliseconds);
  if (status != 0)
    gpuFatalStatus(name ? name : "KICKMAP stable map-loss compaction", status);
  gpuRecordMagnetKernel(milliseconds);
  if (accepted) {
    if (!gpuStablePartitionAcceptedOnDevice(
          accepted, nParticles, remaining,
          "KICKMAP CUDA accepted stable compaction")) {
      gpuCopyAperturePrefixToHost(
        nParticles, "KICKMAP CUDA stable compaction accepted map");
      gpuStablePartitionAccepted(accepted, nParticles, remaining);
    }
  }
  gpuPromoteApertureScratchCoord();
  gpuMarkDeviceChanged(remaining);
  gpuRecordWallSeconds();
  return remaining;
}
#endif

static void gpuCopyApertureLossRowsToHost(long firstLost, long nParticles,
                                          const char *reason) {
  if (!gpuApertureLossRowsNeedHost())
    return;
  gpuCopyDeviceRowsToHost(firstLost, nParticles - firstLost, reason);
}

static void gpuSetGlobalLossCoordinates(long firstLost, long nParticles,
                                        double z, ELEMENT_LIST *eptr,
                                        long elliptical) {
  long ip;

  if (globalLossCoordOffset <= 0)
    return;
  if (!eptr)
    gpuRequiredFailure("missing element pointer for CUDA aperture global loss coordinates");
  if (!convertLocalCoordinatesToGlobal)
    gpuRequiredFailure("global loss-coordinate conversion is unavailable");
  for (ip = firstLost; ip < nParticles; ip++) {
    double X, Y, Z, theta;
    double dz = gpuBase.coord[ip][4] - z;
    short mode = GLOBAL_LOCAL_MODE_DZPM;

    if (elliptical &&
        (!isfinite(gpuBase.coord[ip][0]) || !isfinite(gpuBase.coord[ip][2]))) {
      mode = GLOBAL_LOCAL_MODE_END;
      dz = 0;
    }
    convertLocalCoordinatesToGlobal(&Z, &X, &Y, &theta, mode,
                                    gpuBase.coord[ip], eptr, dz, 0, 0);
    gpuBase.coord[ip][globalLossCoordOffset + 0] = X;
    gpuBase.coord[ip][globalLossCoordOffset + 1] = Z;
    gpuBase.coord[ip][globalLossCoordOffset + 2] = theta;
  }
}

static void gpuSetGlobalLossCoordinatesDZ(long firstLost, long nParticles,
                                          double z, ELEMENT_LIST *eptr) {
  long ip;

  if (globalLossCoordOffset <= 0)
    return;
  if (!eptr)
    gpuRequiredFailure("missing element pointer for CUDA aperture global loss coordinates");
  if (!convertLocalCoordinatesToGlobal)
    gpuRequiredFailure("global loss-coordinate conversion is unavailable");
  for (ip = firstLost; ip < nParticles; ip++) {
    double X, Y, Z, theta;
    double dz = gpuBase.coord[ip][4] - z;

    convertLocalCoordinatesToGlobal(&Z, &X, &Y, &theta,
                                    GLOBAL_LOCAL_MODE_DZ,
                                    gpuBase.coord[ip], eptr, dz, 0, 0);
    gpuBase.coord[ip][globalLossCoordOffset + 0] = X;
    gpuBase.coord[ip][globalLossCoordOffset + 1] = Z;
    gpuBase.coord[ip][globalLossCoordOffset + 2] = theta;
  }
}

static void gpuSetRectangularCollimatorGlobalLossCoordinates(
  long firstLost, long nParticles, double z, double length,
  ELEMENT_LIST *eptr, long openCode) {
  long ip;

  if (globalLossCoordOffset <= 0)
    return;
  if (!eptr)
    gpuRequiredFailure("missing element pointer for CUDA aperture global loss coordinates");
  if (!convertLocalCoordinatesToGlobal)
    gpuRequiredFailure("global loss-coordinate conversion is unavailable");
  for (ip = firstLost; ip < nParticles; ip++) {
    double X, Y, Z, theta;
    double dz = gpuBase.coord[ip][4] - z;

    if (openCode && length > 0) {
      double exitZ = z + length;
      double tolerance = 1e-12 * (1 + fabs(exitZ));

      dz = 0;
      if ((!isfinite(gpuBase.coord[ip][0]) || !isfinite(gpuBase.coord[ip][2])) &&
          fabs(gpuBase.coord[ip][4] - exitZ) <= tolerance)
        dz = length;
    }
    convertLocalCoordinatesToGlobal(&Z, &X, &Y, &theta,
                                    GLOBAL_LOCAL_MODE_DZ,
                                    gpuBase.coord[ip], eptr, dz, 0, 0);
    gpuBase.coord[ip][globalLossCoordOffset + 0] = X;
    gpuBase.coord[ip][globalLossCoordOffset + 1] = Z;
    gpuBase.coord[ip][globalLossCoordOffset + 2] = theta;
  }
}

static void gpuSetScraperGlobalLossCoordinates(long firstLost, long nParticles,
                                               double z, double length,
                                               ELEMENT_LIST *eptr) {
  long ip;

  if (globalLossCoordOffset <= 0)
    return;
  if (!eptr)
    gpuRequiredFailure("missing element pointer for CUDA scraper global loss coordinates");
  if (!convertLocalCoordinatesToGlobal)
    gpuRequiredFailure("global loss-coordinate conversion is unavailable");
  for (ip = firstLost; ip < nParticles; ip++) {
    double X, Y, Z, theta;
    double dz = 0;

    if (length > 0) {
      double exitZ = z + length;
      double tolerance = 1e-12 * (1 + fabs(exitZ));

      if (fabs(gpuBase.coord[ip][4] - z) > tolerance)
        dz = exitZ - gpuBase.coord[ip][4];
    }
    convertLocalCoordinatesToGlobal(&Z, &X, &Y, &theta,
                                    GLOBAL_LOCAL_MODE_DZ,
                                    gpuBase.coord[ip], eptr, dz, 0, 0);
    gpuBase.coord[ip][globalLossCoordOffset + 0] = X;
    gpuBase.coord[ip][globalLossCoordOffset + 1] = Z;
    gpuBase.coord[ip][globalLossCoordOffset + 2] = theta;
  }
}
#endif

static void gpuMarkHostWillChange(void) {
  gpuFlushPendingExactDrift("pending exactDrift before host coordinate mutation");
  gpuBase.elementOnGpu = 0;
  gpuBase.hostCurrent = 1;
  gpuBase.deviceCurrent = 0;
  gpuDeviceIslandHasCsbend = 0;
  if (gpuAcceptedBuffer.hostAccepted) {
    gpuAcceptedBuffer.hostCurrent = 1;
    gpuAcceptedBuffer.deviceCurrent = 0;
  }
}

static void gpuMarkDeviceChanged(long nParticles) {
  gpuBase.deviceCurrent = 1;
  gpuBase.hostCurrent = 0;
  gpuBase.nParticles = nParticles;
  if (gpuBase.element && ((ELEMENT_LIST *)gpuBase.element)->type == T_CSBEND)
    gpuDeviceIslandHasCsbend = 1;
}

static void gpuFlushPendingExactDrift(const char *reason) {
  float milliseconds = 0;
  int status;
  long nParticles;
  long startedWallTimer = 0;

  if (!gpuPendingExactDriftCount)
    return;
  nParticles = gpuPendingExactDriftParticles;
  if (!gpuBase.deviceCoord || !gpuBase.deviceCurrent || nParticles <= 0)
    gpuRequiredFailure(reason ? reason : "pending exactDrift lost device coordinates");

  if (gpuWallStart <= 0) {
    gpuWallStart = wallSeconds();
    startedWallTimer = 1;
  }
  status = gpuCudaExactDrift(gpuBase.deviceCoord, nParticles,
                             (int)gpuBase.deviceStride,
                             gpuPendingExactDriftLength, &milliseconds);
  if (status != 0)
    gpuFatalStatus("coalesced exactDrift kernel", status);
  gpuRecordMilliseconds(&gpuBase.gpuKernelSeconds, milliseconds);
  gpuBase.gpuExactDriftCount++;
  gpuPendingExactDriftLength = 0;
  gpuPendingExactDriftCount = 0;
  gpuPendingExactDriftParticles = 0;
  gpuMarkDeviceChanged(nParticles);
  if (startedWallTimer)
    gpuRecordWallSeconds();
}

static void gpuRecordHelperKernel(float milliseconds) {
  gpuRecordMilliseconds(&gpuBase.gpuKernelSeconds, milliseconds);
  gpuBase.gpuHelperCount++;
}

static void gpuRecordReductionKernel(float milliseconds) {
  gpuRecordMilliseconds(&gpuBase.gpuKernelSeconds, milliseconds);
  gpuBase.gpuReductionCount++;
}

static void gpuRecordApertureKernel(float milliseconds) {
  gpuRecordMilliseconds(&gpuBase.gpuKernelSeconds, milliseconds);
  gpuBase.gpuApertureCount++;
}

static void gpuRecordMagnetKernel(float milliseconds) {
  gpuRecordMilliseconds(&gpuBase.gpuKernelSeconds, milliseconds);
  gpuBase.gpuMagnetCount++;
}

static void gpuRecordWakeKernel(float milliseconds) {
  gpuRecordMilliseconds(&gpuBase.gpuKernelSeconds, milliseconds);
  gpuBase.gpuWakeCount++;
}

static void gpuRecordLscKernel(float milliseconds) {
  gpuRecordMilliseconds(&gpuBase.gpuKernelSeconds, milliseconds);
  gpuBase.gpuLscCount++;
}

static void gpuRecordCsrKernel(float milliseconds) {
  gpuRecordMilliseconds(&gpuBase.gpuKernelSeconds, milliseconds);
  gpuBase.gpuCsrCount++;
}

static void gpuRecordScmultKernel(float milliseconds) {
  gpuRecordMilliseconds(&gpuBase.gpuKernelSeconds, milliseconds);
  gpuBase.gpuScmultCount++;
}

static long gpuUpperTriangularIndex(long i, long j) {
  return i * 7 - i * (i - 1) / 2 + (j - i);
}

static double gpuCompareAbsTolerance(void) {
  return gpuEnvDouble("ELEGANT_GPU_REDUCTION_COMPARE_ABS",
                      gpuEnvDouble("ELEGANT_GPU_COMPARE_ABS", 1e-10));
}

static double gpuCompareRelTolerance(void) {
  return gpuEnvDouble("ELEGANT_GPU_REDUCTION_COMPARE_REL",
                      gpuEnvDouble("ELEGANT_GPU_COMPARE_REL", 1e-10));
}

static long gpuValuesClose(double cpu, double gpu, double absTol, double relTol,
                           double *absDiffReturn, double *relDiffReturn) {
  double absDiff = fabs(cpu - gpu);
  double scale = fmax(fabs(cpu), fabs(gpu));
  double relDiff = scale > DBL_MIN ? absDiff / scale : absDiff;

  if ((isnan(cpu) && isnan(gpu)) || cpu == gpu) {
    if (absDiffReturn)
      *absDiffReturn = 0;
    if (relDiffReturn)
      *relDiffReturn = 0;
    return 1;
  }
  if (absDiffReturn)
    *absDiffReturn = absDiff;
  if (relDiffReturn)
    *relDiffReturn = relDiff;
  if (!isfinite(cpu) || !isfinite(gpu))
    return 0;
  return absDiff <= absTol || relDiff <= relTol;
}

static void gpuCopyBeamSumsShallow(BEAM_SUMS *target, const BEAM_SUMS *source) {
  memcpy(target->centroid, source->centroid, sizeof(target->centroid));
  target->n_part = source->n_part;
  target->z = source->z;
  target->p0 = source->p0;
  target->charge = source->charge;
  target->pass = source->pass;
  if (source->beamSums2) {
    memcpy(&gpuSavedSums2, source->beamSums2, sizeof(gpuSavedSums2));
    target->beamSums2 = &gpuSavedSums2;
  } else {
    target->beamSums2 = NULL;
  }
  if (source->spinSums) {
    memcpy(&gpuSavedSpinSums, source->spinSums, sizeof(gpuSavedSpinSums));
    target->spinSums = &gpuSavedSpinSums;
  } else {
    target->spinSums = NULL;
  }
}

static void gpuCentroidSumsOnDevice(GPU_BEAM_SUM_DATA *result, long nParticles, const char *operation) {
  float milliseconds = 0;
  int status;

  if (!result)
    gpuRequiredFailure("NULL result pointer for CUDA centroid reduction");
  memset(result, 0, sizeof(*result));
  if (nParticles <= 0)
    return;
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaCentroidSums(gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride,
                               result, &milliseconds);
  if (status != 0)
    gpuFatalStatus(operation, status);
  gpuRecordReductionKernel(milliseconds);
}

static double gpuCoordinateSumOnDevice(long nParticles, long coordinate) {
  GPU_BEAM_SUM_DATA result;

  if (nParticles <= 0)
    return 0;
  if (coordinate < 0 || coordinate > 6)
    gpuRequiredFailure("invalid coordinate requested for CUDA coordinate reduction");
  gpuCentroidSumsOnDevice(&result, nParticles, "coordinate reduction kernel");
  return result.centroidSum[coordinate];
}

static double gpuTimeSumOnDevice(long nParticles, double pCentral) {
  GPU_BEAM_SUM_DATA result;
  float milliseconds = 0;
  int status;

  if (nParticles <= 0)
    return 0;
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaTimeSums(gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride,
                           pCentral, c_mks, &result, &milliseconds);
  if (status != 0)
    gpuFatalStatus("time-coordinate reduction kernel", status);
  gpuRecordReductionKernel(milliseconds);
  return result.centroidSum[6];
}

static void gpuCentroidTimeSumsOnDevice(GPU_BEAM_SUM_DATA *result, long nParticles, double pCentral,
                                        const char *operation) {
  float milliseconds = 0;
  int status;

  if (!result)
    gpuRequiredFailure("NULL result pointer for CUDA centroid/time reduction");
  memset(result, 0, sizeof(*result));
  if (nParticles <= 0)
    return;
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaCentroidTimeSums(gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride,
                                   pCentral, c_mks, result, &milliseconds);
  if (status != 0)
    gpuFatalStatus(operation, status);
  gpuRecordReductionKernel(milliseconds);
}

static void gpuApplyCenterOffsets(long nParticles, unsigned int coordinateMask, const double offset[6],
                                  long doTime, double pCentral, double timeOffset) {
  float milliseconds = 0;
  int status;

  if (nParticles <= 0)
    return;
  if (!coordinateMask && (!doTime || timeOffset == 0))
    return;
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaCenterBeam(gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride,
                             coordinateMask, offset, (int)(doTime && timeOffset != 0),
                             pCentral, timeOffset, c_mks, &milliseconds);
  if (status != 0)
    gpuFatalStatus("center beam kernel", status);
  gpuRecordHelperKernel(milliseconds);
  gpuMarkDeviceChanged(nParticles);
}

static void gpuLaunchTrackParticlesWithSReference(VMATRIX *matrix, long nParticles,
                                                  long useSReference, double sReference) {
  GPU_MATRIX_DATA packed;
  float milliseconds = 0;
  int status;

  if (nParticles <= 0)
    return;
  if (!gpuPackMatrix(&packed, matrix))
    gpuRequiredFailure("invalid matrix supplied to CUDA track_particles");
  packed.useSReference = useSReference ? 1 : 0;
  packed.sReference = useSReference ? sReference : 0;
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaTrackParticles(gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride,
                                 &packed, &milliseconds);
  if (status != 0)
    gpuFatalStatus("matrix tracking kernel", status);
  gpuRecordMilliseconds(&gpuBase.gpuKernelSeconds, milliseconds);
  gpuMarkDeviceChanged(nParticles);
  gpuBase.gpuTrackParticleCount++;
}

static void gpuLaunchTrackParticles(VMATRIX *matrix, long nParticles) {
  gpuLaunchTrackParticlesWithSReference(matrix, nParticles, 0, 0);
}

static long gpuPackMatrix(GPU_MATRIX_DATA *packed, VMATRIX *M) {
  long i, j, k, l;
  long offset;

  if (!packed || !gpuMatrixSupported(M))
    return 0;

  memset(packed, 0, sizeof(*packed));
  packed->order = (int)M->order;
  for (i = 0; i < 6; i++) {
    packed->C[i] = M->C[i];
    for (j = 0; j < 6; j++)
      packed->R[i * 6 + j] = M->R[i][j];
  }
  if (M->order >= 2) {
    for (i = 0; i < 6; i++)
      for (j = 0; j < 6; j++)
        for (k = 0; k <= j; k++)
          packed->T[i * 21 + j * (j + 1) / 2 + k] = M->T[i][j][k];
  }
  if (M->order >= 3) {
    for (i = 0; i < 6; i++) {
      offset = i * 56;
      for (j = 0; j < 6; j++)
        for (k = 0; k <= j; k++)
          for (l = 0; l <= k; l++)
            packed->Q[offset + j * (j + 1) * (j + 2) / 6 + k * (k + 1) / 2 + l] =
              M->Q[i][j][k][l];
    }
  }
  return 1;
}

static long gpuMatrixSupported(VMATRIX *M) {
  if (!M || M->order < 1 || M->order > 3)
    return 0;
  if (!M->C || !M->R)
    return 0;
  if (M->order >= 2 && !M->T)
    return 0;
  if (M->order >= 3 && !M->Q)
    return 0;
  return 1;
}

long gpu_matrix_supported(void *matrix) {
  if (!gpuCsbendDriftMatrixParticleCountAllowed(gpuBase.nParticles))
    return 0;
  return gpuMatrixSupported((VMATRIX *)matrix);
}

long gpu_reductions_enabled(long nParticles) {
  if (!gpuBase.initialized || gpuBase.activeDevice < 0 || nParticles <= 0)
    return 0;
#if USE_MPI
  if (distributedBeam)
    return 0;
#endif
  if (gpuBase.reductionOutputNeeded && !gpuOutputDriftReductionMinParticlesExplicit)
    return 0;
  return gpuParticleCountAllowed(nParticles, gpuBase.reductionMinParticles);
}

GPU_BASE *getGpuBase(void) {
  return &gpuBase;
}

void gpuBaseInit(double **coord, long nOriginal, double **accepted, double **lostPart,
                 long isMaster, long lossOutputNeeded,
                 long orderSensitiveOutputNeeded, long reductionOutputNeeded,
                 long alwaysChangeP0, long backtrack) {
  int deviceCount = 0;
  int status;
  long activeDevice;
  const char *mode = gpuMode();

  memset(&gpuBase, 0, sizeof(gpuBase));
#ifdef GPU_VERIFY
  gpuCpuVerificationActive = 0;
#endif
  gpuPendingExactDriftLength = 0;
  gpuPendingExactDriftCount = 0;
  gpuPendingExactDriftParticles = 0;
  gpuBase.coord = coord;
  gpuBase.accepted = accepted;
  gpuBase.lostPart = lostPart;
  gpuBase.nOriginal = nOriginal;
  gpuBase.nParticles = nOriginal;
  gpuBase.isMaster = isMaster;
  gpuBase.lossOutputNeeded = lossOutputNeeded ? 1 : 0;
  gpuBase.orderSensitiveOutputNeeded = orderSensitiveOutputNeeded ? 1 : 0;
  gpuBase.reductionOutputNeeded = reductionOutputNeeded ? 1 : 0;
  gpuBase.backtrack = backtrack ? 1 : 0;
  gpuDeviceIslandHasCsbend = 0;
  gpuRunAlwaysChangeP0 = alwaysChangeP0 ? 1 : 0;
  gpuBase.activeDevice = -1;
  gpuBase.minParticles = gpuEnvLong("ELEGANT_GPU_MIN_PARTICLES",
                                    ELEGANT_GPU_DEFAULT_MIN_PARTICLES);
  gpuMatrixDriftMinParticlesExplicit =
    gpuEnvSetEither("ELEGANT_GPU_MIN_MATRIX_DRIFT_PARTICLES",
                    "ELEGANT_GPU_MIN_MATRIX_PARTICLES");
  gpuBase.matrixMinParticles =
    gpuEnvLongEither("ELEGANT_GPU_MIN_MATRIX_DRIFT_PARTICLES",
                     "ELEGANT_GPU_MIN_MATRIX_PARTICLES",
                     gpuBase.minParticles);
  gpuHelperMinParticlesExplicit = gpuEnvSet("ELEGANT_GPU_MIN_HELPER_PARTICLES");
  gpuBase.helperMinParticles = gpuEnvLong("ELEGANT_GPU_MIN_HELPER_PARTICLES", gpuBase.minParticles);
  gpuBase.exactDriftMinParticles = gpuEnvLong("ELEGANT_GPU_MIN_EXACT_PARTICLES", gpuBase.minParticles);
  gpuOutputDriftReductionMinParticlesExplicit =
    gpuEnvSetEither("ELEGANT_GPU_MIN_OUTPUT_DRIFT_REDUCTION_PARTICLES",
                    "ELEGANT_GPU_MIN_REDUCTION_PARTICLES");
  gpuBase.reductionMinParticles =
    gpuEnvLongEither("ELEGANT_GPU_MIN_OUTPUT_DRIFT_REDUCTION_PARTICLES",
                     "ELEGANT_GPU_MIN_REDUCTION_PARTICLES",
                     gpuBase.minParticles);
  gpuBase.apertureMinParticles = gpuEnvLong("ELEGANT_GPU_MIN_APERTURE_PARTICLES", gpuBase.minParticles);
  gpuMagnetMinParticlesExplicit = gpuEnvSet("ELEGANT_GPU_MIN_MAGNET_PARTICLES");
  gpuBase.magnetMinParticles = gpuEnvLong("ELEGANT_GPU_MIN_MAGNET_PARTICLES", gpuBase.minParticles);
  gpuWakeMinParticlesExplicit = gpuEnvSet("ELEGANT_GPU_MIN_WAKE_PARTICLES");
  gpuBase.wakeMinParticles = gpuEnvLong("ELEGANT_GPU_MIN_WAKE_PARTICLES", gpuBase.minParticles);
  gpuBase.lscMinParticles = gpuEnvLong("ELEGANT_GPU_MIN_LSC_PARTICLES", gpuBase.wakeMinParticles);
  gpuBase.csrMinParticles = gpuEnvLong("ELEGANT_GPU_MIN_CSR_PARTICLES", gpuBase.minParticles);
  gpuBase.csrMinBins = gpuEnvLong("ELEGANT_GPU_MIN_CSR_BINS", 1024);
  gpuBase.scmultMinParticles = gpuEnvLong("ELEGANT_GPU_MIN_SCMULT_PARTICLES", gpuBase.minParticles);
  gpuBase.verifyMode = gpuEnvFlag("ELEGANT_GPU_VERIFY");
  gpuBase.requiredMode = strcmp(mode, "required") == 0;
  gpuBase.hostCurrent = 1;
  gpuBase.hostCoordBase = coord ? (void *)coord[0] : NULL;
  gpuVerbose = gpuEnvFlag("ELEGANT_GPU_VERBOSE");
  gpuExactDriftExplicit = gpuEnvSet("ELEGANT_GPU_ENABLE_EXACT");
  gpuEnableExactDrift = !gpuExactDriftExplicit ||
                        gpuEnvFlag("ELEGANT_GPU_ENABLE_EXACT");
  gpuApertureParallelCompactionExplicit =
    gpuEnvSet("ELEGANT_GPU_ENABLE_APERTURE_PARALLEL_COMPACTION");
  gpuApertureParallelCompactionVerifyDisabled = 0;
  gpuApertureParallelCompactionOrderDisabled = 0;
  gpuEnableApertureParallelCompaction =
    gpuApertureParallelCompactionExplicit ?
    gpuEnvFlag("ELEGANT_GPU_ENABLE_APERTURE_PARALLEL_COMPACTION") :
    !gpuBase.lossOutputNeeded;
#ifdef GPU_VERIFY
  if (gpuEnableApertureParallelCompaction) {
    gpuEnableApertureParallelCompaction = 0;
    gpuApertureParallelCompactionVerifyDisabled = 1;
  }
#endif
  if (gpuEnableApertureParallelCompaction &&
      gpuBase.orderSensitiveOutputNeeded &&
      !gpuApertureParallelCompactionExplicit) {
    gpuEnableApertureParallelCompaction = 0;
    gpuApertureParallelCompactionOrderDisabled = 1;
  }
  gpuMagnetLossCompactionExplicit =
    gpuEnvSet("ELEGANT_GPU_ENABLE_MAGNET_LOSS_COMPACTION");
  gpuEnableMagnetLossCompaction =
    gpuMagnetLossCompactionExplicit ?
    gpuEnvFlag("ELEGANT_GPU_ENABLE_MAGNET_LOSS_COMPACTION") :
    !gpuBase.lossOutputNeeded;
  gpuCsbendDriftExplicit = gpuEnvSet("ELEGANT_GPU_ENABLE_CSBEND_DRIFT");
  gpuEnableCsbendDrift = !gpuCsbendDriftExplicit ||
                         gpuEnvFlag("ELEGANT_GPU_ENABLE_CSBEND_DRIFT");
  gpuEnableApertureAcceptedDevice =
    (gpuEnableApertureParallelCompaction || gpuEnableMagnetLossCompaction) &&
    (!gpuEnvSet("ELEGANT_GPU_ENABLE_APERTURE_ACCEPTED_DEVICE") ||
     gpuEnvFlag("ELEGANT_GPU_ENABLE_APERTURE_ACCEPTED_DEVICE"));
  gpuCsrTrackingExplicit = gpuEnvSet("ELEGANT_GPU_ENABLE_CSR_DRIFT");
  gpuCsrResidentExplicit = gpuEnvSet("ELEGANT_GPU_ENABLE_CSR_RESIDENT_DRIFT");
  gpuEnableCsrTracking =
    !gpuCsrTrackingExplicit || gpuEnvFlag("ELEGANT_GPU_ENABLE_CSR_DRIFT");
  gpuEnableCsrResident =
    gpuEnableCsrTracking &&
    (!gpuCsrResidentExplicit ||
     gpuEnvFlag("ELEGANT_GPU_ENABLE_CSR_RESIDENT_DRIFT"));
  gpuScmultExplicit = gpuEnvSet("ELEGANT_GPU_ENABLE_SCMULT");
  gpuEnableScmult = !gpuScmultExplicit ||
                    gpuEnvFlag("ELEGANT_GPU_ENABLE_SCMULT");
  gpuWakeTrackingDriftExplicit =
    gpuEnvSet("ELEGANT_GPU_ENABLE_WAKE_TRACKING_DRIFT");
  gpuEnableWakeTrackingDrift =
    gpuWakeTrackingDriftExplicit &&
    gpuEnvFlag("ELEGANT_GPU_ENABLE_WAKE_TRACKING_DRIFT");
  gpuCombinedWakeExplicit =
    gpuEnvSet("ELEGANT_GPU_ENABLE_COMBINED_WAKE");
  gpuEnableCombinedWake =
    !gpuCombinedWakeExplicit ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_COMBINED_WAKE");
  gpuEnableCombinedWakeMultibunch =
    gpuEnableCombinedWake &&
    (!gpuEnvSet("ELEGANT_GPU_ENABLE_COMBINED_WAKE_MULTIBUNCH") ||
     gpuEnvFlag("ELEGANT_GPU_ENABLE_COMBINED_WAKE_MULTIBUNCH"));
  gpuEnableCombinedWakeFft =
    gpuEnableCombinedWake &&
    (!gpuEnvSet("ELEGANT_GPU_ENABLE_COMBINED_WAKE_FFT") ||
     gpuEnvFlag("ELEGANT_GPU_ENABLE_COMBINED_WAKE_FFT"));
  gpuEnableBatchedTuneTracking =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_BATCHED_TUNE_TRACKING") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_BATCHED_TUNE_TRACKING");
  gpuBatchedTuneMinParticles =
    gpuEnvLong("ELEGANT_GPU_MIN_BATCHED_TUNE_PARTICLES", 32);
  if (gpuBatchedTuneMinParticles < 1)
    gpuBatchedTuneMinParticles = 1;
  gpuEnableBatchedSearchTracking =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_BATCHED_SEARCH_TRACKING") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_BATCHED_SEARCH_TRACKING");
  gpuBatchedSearchMinParticles =
    gpuEnvLong("ELEGANT_GPU_MIN_BATCHED_SEARCH_PARTICLES", 32);
  if (gpuBatchedSearchMinParticles < 1)
    gpuBatchedSearchMinParticles = 1;
  gpuEnablePolynomialSeries =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_POLYNOMIAL_SERIES") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_POLYNOMIAL_SERIES");
  gpuEnableRfdf =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_RFDF") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_RFDF");
  gpuRfdfMinParticles =
    gpuEnvLong("ELEGANT_GPU_MIN_RFDF_PARTICLES", 64);
  if (gpuRfdfMinParticles < 1)
    gpuRfdfMinParticles = 1;
  gpuEnableSreffects =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_SREFFECTS") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_SREFFECTS");
  gpuSreffectsMinParticles =
    gpuEnvLong("ELEGANT_GPU_MIN_SREFFECTS_PARTICLES", 64);
  if (gpuSreffectsMinParticles < 1)
    gpuSreffectsMinParticles = 1;
  gpuEnableBggexp =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_BGGEXP") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_BGGEXP");
  gpuBggexpMinParticles =
    gpuEnvLong("ELEGANT_GPU_MIN_BGGEXP_PARTICLES", 64);
  if (gpuBggexpMinParticles < 1)
    gpuBggexpMinParticles = 1;
  gpuEnableCwiggler =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_CWIGGLER") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_CWIGGLER");
  gpuCwigglerMinParticles =
    gpuEnvLong("ELEGANT_GPU_MIN_CWIGGLER_PARTICLES", 64);
  if (gpuCwigglerMinParticles < 1)
    gpuCwigglerMinParticles = 1;
  gpuEnableFtable =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_FTABLE") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_FTABLE");
  gpuFtableMinParticles =
    gpuEnvLong("ELEGANT_GPU_MIN_FTABLE_PARTICLES", 64);
  if (gpuFtableMinParticles < 1)
    gpuFtableMinParticles = 1;
  gpuEnableBmxyz =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_BMXYZ") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_BMXYZ");
  gpuBmxyzMinParticles =
    gpuEnvLong("ELEGANT_GPU_MIN_BMXYZ_PARTICLES", 64);
  if (gpuBmxyzMinParticles < 1)
    gpuBmxyzMinParticles = 1;
  gpuEnableCcbend =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_CCBEND") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_CCBEND");
  gpuCcbendMinParticles =
    gpuEnvLong("ELEGANT_GPU_MIN_CCBEND_PARTICLES", 64);
  if (gpuCcbendMinParticles < 1)
    gpuCcbendMinParticles = 1;
  gpuEnableRfmode =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_RFMODE") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_RFMODE");
  gpuEnableFrfmode =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_FRFMODE") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_FRFMODE");
  gpuRfmodeMinParticles =
    gpuEnvLong("ELEGANT_GPU_MIN_RFMODE_PARTICLES", 8192);
  if (gpuRfmodeMinParticles < 1)
    gpuRfmodeMinParticles = 1;
  gpuLscTrackingExplicit = gpuEnvSet("ELEGANT_GPU_ENABLE_LSC");
  gpuEnableLscTracking = !gpuLscTrackingExplicit ||
                         gpuEnvFlag("ELEGANT_GPU_ENABLE_LSC");
  gpuRfcwTrackingDriftExplicit =
    gpuEnvSet("ELEGANT_GPU_ENABLE_RFCW_TRACKING_DRIFT");
  gpuEnableRfcwTrackingDrift =
    gpuRfcwTrackingDriftExplicit &&
    gpuEnvFlag("ELEGANT_GPU_ENABLE_RFCW_TRACKING_DRIFT");
  gpuRfcaChangeP0DriftExplicit =
    gpuEnvSet("ELEGANT_GPU_ENABLE_RFCA_CHANGE_P0_DRIFT");
  gpuEnableRfcaChangeP0Drift =
    gpuRfcaChangeP0DriftExplicit &&
    gpuEnvFlag("ELEGANT_GPU_ENABLE_RFCA_CHANGE_P0_DRIFT");
#if USE_MPI
  if (!gpuCsbendDriftExplicit)
    gpuEnableCsbendDrift = 0;
  if (!gpuCsrTrackingExplicit) {
    gpuEnableCsrTracking = 0;
    gpuEnableCsrResident = 0;
  } else if (!gpuCsrResidentExplicit) {
    gpuEnableCsrResident = 0;
  }
  if (!gpuScmultExplicit)
    gpuEnableScmult = 0;
  if (!gpuCombinedWakeExplicit) {
    gpuEnableCombinedWake = 0;
    gpuEnableCombinedWakeMultibunch = 0;
    gpuEnableCombinedWakeFft = 0;
  } else {
    if (!gpuEnvSet("ELEGANT_GPU_ENABLE_COMBINED_WAKE_MULTIBUNCH"))
      gpuEnableCombinedWakeMultibunch = 0;
    if (!gpuEnvSet("ELEGANT_GPU_ENABLE_COMBINED_WAKE_FFT"))
      gpuEnableCombinedWakeFft = 0;
  }
  if (!gpuRfcaChangeP0DriftExplicit)
    gpuEnableRfcaChangeP0Drift = 0;
  gpuEnableBatchedTuneTracking = 0;
  gpuEnableBatchedSearchTracking = 0;
  gpuEnablePolynomialSeries = 0;
  gpuEnableRfdf = 0;
  gpuEnableSreffects = 0;
  gpuEnableBggexp = 0;
  gpuEnableCwiggler = 0;
  gpuEnableFtable = 0;
  gpuEnableBmxyz = 0;
  gpuEnableRfmode = 0;
  gpuEnableFrfmode = 0;
#endif
  gpuAvoidShortGpuIslands = !gpuEnvSet("ELEGANT_GPU_AVOID_SHORT_GPU_ISLANDS") ||
                            gpuEnvFlag("ELEGANT_GPU_AVOID_SHORT_GPU_ISLANDS");
  gpuShortGpuIslandMaxElements = gpuEnvLong("ELEGANT_GPU_SHORT_GPU_ISLAND_MAX_ELEMENTS", 4);

  if (strcmp(mode, "off") == 0) {
    gpuBase.initialized = 1;
    return;
  }

  if (strcmp(mode, "auto") != 0 && strcmp(mode, "required") != 0)
    gpuRequiredFailure("ELEGANT_GPU_MODE must be off, auto, or required");

  status = gpuCudaRuntimeGetDeviceCount(&deviceCount);
  gpuBase.deviceCount = deviceCount;
  if (status != 0 || deviceCount <= 0) {
    if (gpuBase.requiredMode)
      gpuRequiredFailure(status == 0 ? "no CUDA devices found" : gpuCudaRuntimeGetErrorString(status));
    if (gpuVerbose)
      fprintf(stderr, "elegant CUDA: no usable CUDA device found, using CPU fallback.\n");
    gpuBase.initialized = 1;
    return;
  }

#if USE_MPI
  if (isMaster && distributedBeam && !partOnMaster) {
    if (gpuVerbose)
      fprintf(stderr,
              "elegant CUDA: MPI master rank %d has no local tracking particles; using CPU staging only.\n",
              myid);
    gpuBase.initialized = 1;
    return;
  }
#endif

  if (gpuEnvSet("ELEGANT_GPU_DEVICE")) {
    activeDevice = gpuEnvLong("ELEGANT_GPU_DEVICE", 0);
  } else {
    activeDevice = 0;
#if USE_MPI
    if (n_processors > 1) {
      if (isSlave && distributedBeam)
        activeDevice = (myid - 1) % deviceCount;
      else
        activeDevice = myid % deviceCount;
    }
#endif
  }

  gpuBase.activeDevice = activeDevice;
  if (gpuBase.activeDevice < 0 || gpuBase.activeDevice >= deviceCount) {
    if (gpuBase.requiredMode)
      gpuRequiredFailure("ELEGANT_GPU_DEVICE is outside the available device range");
    gpuBase.activeDevice = -1;
    gpuBase.initialized = 1;
    return;
  }

  status = gpuCudaRuntimeSetDevice((int)gpuBase.activeDevice);
  if (status != 0) {
    if (gpuBase.requiredMode)
      gpuRequiredFailure(gpuCudaRuntimeGetErrorString(status));
    gpuBase.activeDevice = -1;
    gpuBase.initialized = 1;
    return;
  }

  if (gpuVerbose) {
    char deviceName[256] = "";
    status = gpuCudaRuntimeGetDeviceName((int)gpuBase.activeDevice, deviceName, sizeof(deviceName));
#if USE_MPI
    fprintf(stderr,
              "elegant CUDA: selected device %ld%s%s on MPI rank %d; thresholds matrix=%ld helper=%ld reduction=%ld aperture=%ld magnet=%ld wake=%ld lsc=%ld csr=%ld particles csrBins=%ld scmult=%ld exactDrift=%ld particles%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s.\n",
            gpuBase.activeDevice,
            status == 0 ? " " : "",
            status == 0 ? deviceName : "",
            myid,
            gpuBase.matrixMinParticles,
            gpuBase.helperMinParticles,
            gpuBase.reductionMinParticles,
            gpuBase.apertureMinParticles,
            gpuBase.magnetMinParticles,
            gpuBase.wakeMinParticles,
            gpuBase.lscMinParticles,
            gpuBase.csrMinParticles,
            gpuBase.csrMinBins,
            gpuBase.scmultMinParticles,
            gpuBase.exactDriftMinParticles,
            gpuExactDriftStatus(),
            gpuMatrixTrackingStatus(),
            gpuHelperOutputStatus(),
            gpuReductionOutputStatus(),
            gpuApertureParallelCompactionStatus(),
            gpuEnableApertureAcceptedDevice ? "; accepted-device compaction enabled" : "",
            gpuMagnetTrackingStatus(),
            gpuMagnetLossCompactionStatus(),
            gpuCsbendDriftStatus(),
            gpuCsrTrackingStatus(),
            gpuCsrResidentStatus(),
            gpuScmultStatus(),
            gpuWakeTrackingStatus(),
            gpuCombinedWakeStatus(),
            gpuLscTrackingStatus(),
            gpuRfcwTrackingDriftStatus(),
            gpuRfcaChangeP0DriftStatus());
#else
    fprintf(stderr,
            "elegant CUDA: selected device %ld%s%s; thresholds matrix=%ld helper=%ld reduction=%ld aperture=%ld magnet=%ld wake=%ld lsc=%ld csr=%ld particles csrBins=%ld scmult=%ld exactDrift=%ld particles%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s.\n",
            gpuBase.activeDevice,
            status == 0 ? " " : "",
            status == 0 ? deviceName : "",
            gpuBase.matrixMinParticles,
            gpuBase.helperMinParticles,
            gpuBase.reductionMinParticles,
            gpuBase.apertureMinParticles,
            gpuBase.magnetMinParticles,
            gpuBase.wakeMinParticles,
            gpuBase.lscMinParticles,
            gpuBase.csrMinParticles,
            gpuBase.csrMinBins,
            gpuBase.scmultMinParticles,
            gpuBase.exactDriftMinParticles,
            gpuExactDriftStatus(),
            gpuMatrixTrackingStatus(),
            gpuHelperOutputStatus(),
            gpuReductionOutputStatus(),
            gpuApertureParallelCompactionStatus(),
            gpuEnableApertureAcceptedDevice ? "; accepted-device compaction enabled" : "",
            gpuMagnetTrackingStatus(),
            gpuMagnetLossCompactionStatus(),
            gpuCsbendDriftStatus(),
            gpuCsrTrackingStatus(),
            gpuCsrResidentStatus(),
            gpuScmultStatus(),
            gpuWakeTrackingStatus(),
            gpuCombinedWakeStatus(),
            gpuLscTrackingStatus(),
            gpuRfcwTrackingDriftStatus(),
            gpuRfcaChangeP0DriftStatus());
#endif
    if (gpuEnableFtable)
      fprintf(stderr,
              "elegant CUDA: fixed-step FTABLE enabled at %ld particles.\n",
              gpuFtableMinParticles);
    if (gpuEnableBmxyz)
      fprintf(stderr,
              "elegant CUDA: fixed-step BMXYZ enabled at %ld particles.\n",
              gpuBmxyzMinParticles);
    if (gpuEnableCcbend)
      fprintf(stderr,
              "elegant CUDA: deterministic order-2 CCBEND enabled at %ld particles.\n",
              gpuCcbendMinParticles);
    if (gpuEnableRfmode || gpuEnableFrfmode)
      fprintf(stderr,
              "elegant CUDA: resident RFMODE=%ld FRFMODE=%ld at %ld particles.\n",
              gpuEnableRfmode, gpuEnableFrfmode,
              gpuRfmodeMinParticles);
    if (gpuEnableSreffects)
      fprintf(stderr,
              "elegant CUDA: deterministic SREFFECTS enabled at %ld particles.\n",
              gpuSreffectsMinParticles);
  }

  gpuBase.initialized = 1;
}

void gpuDisableForRun(const char *reason) {
  if (!gpuBase.initialized || gpuBase.activeDevice < 0)
    return;
  if (gpuBase.requiredMode)
    gpuRequiredFailure(reason ? reason : "CUDA disabled for this run");
  if (gpuVerbose)
    fprintf(stderr, "elegant CUDA: %s; using CPU fallback.\n",
            reason ? reason : "CUDA disabled for this run");
  gpuReleaseDeviceBuffer();
  gpuBase.activeDevice = -1;
  gpuBase.elementOnGpu = 0;
  gpuBase.deviceCurrent = 0;
  gpuBase.hostCurrent = 1;
}

void gpuBaseDealloc(void) {
  forceParticlesToCpu("gpuBaseDealloc");
  if (gpuVerbose || gpuBase.gpuElementCount)
    gpuDisplaySyncTimings();
  gpuReleaseDeviceBuffer();
  gpuReleaseAcceptedBuffer();
  gpuReleaseKickMapCache();
  gpuReleaseRfcwKickScratch();
  gpuReleaseLscScratch();
  gpuReleaseBeamSumsScratch();
  gpuReleaseRfcaScratch();
  gpuReleasePolynomialSeriesCache();
  gpuReleaseBatchedSearchScratch();
  gpuCudaPolynomialSeriesRelease();
  gpuCudaBggexpRelease();
  gpuCudaFtableRelease();
  gpuCudaBmxyzRelease();
  gpuCudaRfmodeRelease();
  gpuReleaseApertureScratch();
  gpuReleaseCsrScratch();
  gpuCudaCombinedWakeRelease();
  free(gpuBunchRangeCache.start);
  free(gpuBunchRangeCache.count);
  memset(&gpuBunchRangeCache, 0, sizeof(gpuBunchRangeCache));
  free(gpuRfmodeHostHistogram);
  gpuRfmodeHostHistogram = NULL;
  gpuRfmodeHostHistogramCapacity = 0;
  free(gpuRfmodeHostTime);
  gpuRfmodeHostTime = NULL;
  gpuRfmodeHostTimeCapacity = 0;
  gpuBase.elementOnGpu = 0;
  gpuBase.element = NULL;
}

void setElementGpuData(void *eptr0, long nParticles) {
  ELEMENT_LIST *eptr = (ELEMENT_LIST *)eptr0;
  long eligible = 0;

  gpuBase.element = eptr0;
  gpuBase.nParticles = nParticles;
  gpuBase.elementOnGpu = 0;

  if (!gpuBase.initialized || gpuBase.activeDevice < 0 || !eptr || nParticles <= 0) {
    if (gpuBase.deviceCurrent && !gpuBase.hostCurrent)
      gpuForceCpuForElement(eptr, "CPU element after CUDA element");
    gpuMarkHostWillChange();
    return;
  }

  eligible = gpuCurrentElementEligible();

  if (eligible) {
    if (gpuShouldUseCpuForShortGpuIsland(eptr, nParticles)) {
      gpuBase.gpuShortGpuIslandCpuCount++;
      gpuMarkHostWillChange();
      return;
    }
    gpuBase.elementOnGpu = 1;
    gpuBase.gpuElementCount++;
    return;
  }

  if (gpuBase.deviceCurrent && gpuPassiveElementSupported(eptr, nParticles)) {
    gpuBase.elementOnGpu = 1;
    gpuBase.gpuPassiveElementCount++;
    return;
  }

  if (gpuBase.deviceCurrent && !gpuBase.hostCurrent)
    gpuForceCpuForElement(eptr, "CPU element after CUDA element");
  gpuMarkHostWillChange();
}

long getElementOnGpu(void) {
  return gpuBase.elementOnGpu;
}

long gpu_batched_tune_tracking_enabled(long particles) {
#if USE_MPI
  (void)particles;
  return 0;
#else
  return gpuEnableBatchedTuneTracking &&
         particles >= gpuBatchedTuneMinParticles;
#endif
}

long gpu_batched_tune_beamline_supported(void *beamline0) {
#if USE_MPI
  (void)beamline0;
  return 0;
#else
  LINE_LIST *beamline = (LINE_LIST *)beamline0;
  ELEMENT_LIST *eptr;
  long trackedElements = 0;

  if (!gpuEnableBatchedTuneTracking || !beamline)
    return 0;
  for (eptr = beamline->elem; eptr; eptr = eptr->succ) {
    if (eptr->ignore)
      continue;
    switch (eptr->type) {
    case T_MARK:
    case T_RECIRC:
      break;
    case T_POLYNOMIALSERIES:
      if (!gpuPolynomialSeriesElementSupported(eptr))
        return 0;
      trackedElements++;
      break;
    default:
      return 0;
    }
  }
  return trackedElements > 0;
#endif
}

long gpu_batched_search_tracking_enabled(long particles) {
#if USE_MPI
  (void)particles;
  return 0;
#else
  long enabled = gpuBase.initialized ? gpuEnableBatchedSearchTracking :
    (!gpuEnvSet("ELEGANT_GPU_ENABLE_BATCHED_SEARCH_TRACKING") ||
     gpuEnvFlag("ELEGANT_GPU_ENABLE_BATCHED_SEARCH_TRACKING"));
  long threshold = gpuBase.initialized ? gpuBatchedSearchMinParticles :
    gpuEnvLong("ELEGANT_GPU_MIN_BATCHED_SEARCH_PARTICLES", 32);
  if (threshold < 1)
    threshold = 1;
  return enabled && particles >= threshold;
#endif
}

long gpu_batched_search_beamline_supported(void *beamline0) {
#if USE_MPI
  (void)beamline0;
  return 0;
#else
  LINE_LIST *beamline = (LINE_LIST *)beamline0;
  ELEMENT_LIST *eptr;

  if (!beamline)
    return 0;
  for (eptr = beamline->elem; eptr; eptr = eptr->succ) {
    unsigned long flags;
    if (eptr->ignore)
      continue;
    /* Search boundaries and turn-by-turn tunes amplify otherwise small GPU
     * differences.  Keep the batched search on the validated deterministic
     * subset until damping and high-order bend maps have dedicated search
     * baselines.  The elements themselves may still use their normal CUDA
     * paths outside a batched search. */
    if (eptr->type == T_CSBEND) {
      CSBEND *csbend = (CSBEND *)eptr->p_elem;
      if (!csbend || csbend->synch_rad || csbend->k3 || csbend->k4 ||
          csbend->k5 || csbend->k6 || csbend->k7 || csbend->k8)
        return 0;
    } else if (eptr->type == T_KQUAD) {
      KQUAD *kquad = (KQUAD *)eptr->p_elem;
      if (!kquad || kquad->synch_rad)
        return 0;
    }
    flags = entity_description[eptr->type].flags;
    if (flags & (COLLECTIVE_EFFECTS | UNIPROCESSOR))
      return 0;
    /* Reuse the actual CUDA option guards, rather than element metadata
     * alone.  This excludes ISR/noise and unsupported option combinations
     * whose point-by-point CPU execution may depend on RNG ordering. */
    if (gpuElementEligible(eptr, LONG_MAX / 4) ||
        gpuPassiveElementSupported(eptr, LONG_MAX / 4))
      continue;
    switch (eptr->type) {
    case T_MARK:
    case T_RECIRC:
    case T_WATCH:
      break;
    default:
      return 0;
    }
  }
  return 1;
#endif
}

void gpu_configure_batched_momentum_search(const double *deltaById,
                                           const long *targetById,
                                           long particles, long turns,
                                           long firePass,
                                           double *history,
                                           double *historyCount) {
  long ip;

  gpuReleaseBatchedSearchScratch();
  if (!deltaById || !targetById || particles <= 0 || turns <= 0 ||
      !history || !historyCount)
    return;
  gpuBatchedSearchScratch.hostData =
    (double *)malloc((size_t)(2 * particles) *
                     sizeof(*gpuBatchedSearchScratch.hostData));
  if (!gpuBatchedSearchScratch.hostData)
    gpuRequiredFailure("unable to allocate batched search host data");
  for (ip = 0; ip < particles; ip++) {
    gpuBatchedSearchScratch.hostData[2 * ip] = targetById[ip];
    gpuBatchedSearchScratch.hostData[2 * ip + 1] = deltaById[ip];
  }
  memset(history, 0,
         (size_t)particles * 5 * (size_t)turns * sizeof(*history));
  memset(historyCount, 0, (size_t)particles * sizeof(*historyCount));
  gpuBatchedSearchScratch.hostHistory = history;
  gpuBatchedSearchScratch.hostHistoryCount = historyCount;
  gpuBatchedSearchScratch.particles = particles;
  gpuBatchedSearchScratch.turns = turns;
  gpuBatchedSearchScratch.firePass = firePass;
  gpuBatchedSearchScratch.configured = 1;
}

long gpu_apply_batched_momentum_search(long particles, long pass,
                                       long target, double dx, double dy,
                                       double pCentral) {
#if USE_MPI || defined(GPU_VERIFY)
  (void)particles;
  (void)pass;
  (void)target;
  (void)dx;
  (void)dy;
  (void)pCentral;
  return 0;
#else
  float milliseconds = 0;
  unsigned long historyValues;
  int status;

  if (!gpuBatchedSearchScratch.configured ||
      !gpuBase.initialized || gpuBase.activeDevice < 0 || particles <= 0)
    return 0;
  if (!gpuBatchedSearchScratch.uploaded) {
    historyValues =
      (unsigned long)gpuBatchedSearchScratch.particles * 5UL *
      (unsigned long)gpuBatchedSearchScratch.turns;
    status = gpuCudaMallocDouble((void **)&gpuBatchedSearchScratch.deviceData,
                                 (unsigned long)(2 * gpuBatchedSearchScratch.particles));
    if (status != 0)
      gpuFatalStatus("cudaMalloc(batched search data)", status);
    status = gpuCudaMallocDouble((void **)&gpuBatchedSearchScratch.deviceHistory,
                                 historyValues);
    if (status != 0)
      gpuFatalStatus("cudaMalloc(batched search history)", status);
    status = gpuCudaMallocDouble((void **)&gpuBatchedSearchScratch.deviceHistoryCount,
                                 (unsigned long)gpuBatchedSearchScratch.particles);
    if (status != 0)
      gpuFatalStatus("cudaMalloc(batched search turn counts)", status);
    status = gpuCudaCopyHostToDevice(
      gpuBatchedSearchScratch.deviceData,
      gpuBatchedSearchScratch.hostData,
      (unsigned long)(2 * gpuBatchedSearchScratch.particles), &milliseconds);
    if (status != 0)
      gpuFatalStatus("cudaMemcpy(batched search data host to device)", status);
    gpuRecordMilliseconds(&gpuBase.gpuTransferToDeviceSeconds, milliseconds);
    status = gpuCudaClearBatchedSearchHistory(
      gpuBatchedSearchScratch.deviceHistory, historyValues,
      gpuBatchedSearchScratch.deviceHistoryCount,
      (unsigned long)gpuBatchedSearchScratch.particles);
    if (status != 0)
      gpuFatalStatus("cudaMemset(batched search history)", status);
    gpuBatchedSearchScratch.uploaded = 1;
  }
  gpuCopyHostToDevice(particles);
  milliseconds = 0;
  status = gpuCudaApplyBatchedMomentumSearch(
    gpuBase.deviceCoord, particles, (int)gpuBase.deviceStride,
    gpuBatchedSearchScratch.deviceData,
    gpuBatchedSearchScratch.particles, target, pass,
    gpuBatchedSearchScratch.firePass,
    gpuBatchedSearchScratch.deviceHistory,
    gpuBatchedSearchScratch.deviceHistoryCount,
    gpuBatchedSearchScratch.turns, dx, dy, pCentral, c_mks,
    &milliseconds);
  if (status != 0)
    gpuFatalStatus("batched momentum-search offset/history kernel", status);
  gpuRecordHelperKernel(milliseconds);
  gpuMarkDeviceChanged(particles);
  gpuRecordWallSeconds();
  return 1;
#endif
}

void gpu_clear_batched_momentum_search(void) {
  gpuReleaseBatchedSearchScratch();
}

void gpuSetCpuParticleArray(double **coord, long nParticles) {
  if (!gpuBase.initialized)
    return;

  gpuMarkHostWillChange();
  gpuBase.coord = coord;
  gpuBase.hostCoordBase = coord ? (void *)coord[0] : NULL;
  gpuBase.nParticles = nParticles;
}

static void gpuDisplaySyncTimings(void) {
  if (!gpuBase.gpuSyncRequestCount)
    return;
  fprintf(stderr,
          "elegant CUDA: sync requests=%ld copies=%ld readOnly=%ld mutable=%ld output=%ld cpuElement=%ld apertureLoss=%ld mpi=%ld verification=%ld collective=%ld reductions=%ld dealloc=%ld other=%ld\n",
          gpuBase.gpuSyncRequestCount,
          gpuBase.gpuSyncCopyCount,
          gpuBase.gpuSyncReadOnlyCount,
          gpuBase.gpuSyncMutableCount,
          gpuBase.gpuSyncOutputCount,
          gpuBase.gpuSyncCpuElementCount,
          gpuBase.gpuSyncApertureLossCount,
          gpuBase.gpuSyncMpiCount,
          gpuBase.gpuSyncVerificationCount,
          gpuBase.gpuSyncCollectiveCount,
          gpuBase.gpuSyncReductionCount,
          gpuBase.gpuSyncDeallocCount,
          gpuBase.gpuSyncOtherCount);
}

static double **syncParticlesToCpu(const char *reason, long readOnly) {
  long copied = gpuCopyDeviceToHost(gpuBase.nParticles);
  long acceptedCopied = 0;
  if (!readOnly)
    acceptedCopied = gpuCopyAcceptedDeviceToHost(gpuBase.nParticles, reason);
  gpuRecordSyncRequest(reason, copied || acceptedCopied, readOnly);
  if (!readOnly)
    gpuMarkHostWillChange();
  if (gpuVerbose && reason)
    fprintf(stderr, "elegant CUDA: %sCPU synchronization requested by %s.\n",
            readOnly ? "read-only " : "", reason);
  return gpuBase.coord;
}

double **forceParticlesToCpu(const char *reason) {
  return syncParticlesToCpu(reason, 0);
}

double **copyParticlesToCpuReadOnly(const char *reason) {
  return syncParticlesToCpu(reason, 1);
}

void startGpuTimer(void) {
#ifdef GPU_VERIFY
  gpuCpuVerificationActive = 0;
#endif
  gpuWallStart = wallSeconds();
}

void startCpuTimer(void) {
  gpuBase.elementOnGpu = 0;
#ifdef GPU_VERIFY
  gpuCpuVerificationActive = 1;
#endif
}

void displayTimings(void) {
  if (!gpuVerbose && !gpuBase.gpuElementCount)
    return;
  fprintf(stderr,
          "elegant CUDA: elements=%ld passive=%ld matrix=%ld exactDrift=%ld linearDrift=%ld helpers=%ld reductions=%ld apertures=%ld magnets=%ld wakes=%ld lsc=%ld csr=%ld scmult=%ld wall=%.6fs kernel=%.6fs h2d=%.6fs d2h=%.6fs\n",
          gpuBase.gpuElementCount,
          gpuBase.gpuPassiveElementCount,
          gpuBase.gpuTrackParticleCount,
          gpuBase.gpuExactDriftCount,
          gpuBase.gpuLinearDriftCount,
          gpuBase.gpuHelperCount,
          gpuBase.gpuReductionCount,
          gpuBase.gpuApertureCount,
          gpuBase.gpuMagnetCount,
          gpuBase.gpuWakeCount,
          gpuBase.gpuLscCount,
          gpuBase.gpuCsrCount,
          gpuBase.gpuScmultCount,
          gpuBase.gpuWallSeconds,
          gpuBase.gpuKernelSeconds,
          gpuBase.gpuTransferToDeviceSeconds,
          gpuBase.gpuTransferToHostSeconds);
  if (gpuBase.gpuStandaloneLscCount ||
      gpuBase.gpuRfcwLscKickOnlyCount ||
      gpuBase.gpuRfcwLscFullCount)
    fprintf(stderr,
            "elegant CUDA: lsc detail standalone=%ld rfcwKickOnly=%ld rfcwFull=%ld\n",
            gpuBase.gpuStandaloneLscCount,
            gpuBase.gpuRfcwLscKickOnlyCount,
            gpuBase.gpuRfcwLscFullCount);
  if (gpuBase.gpuShortGpuIslandCpuCount)
    fprintf(stderr,
            "elegant CUDA: short GPU island CPU skips=%ld maxElements=%ld\n",
            gpuBase.gpuShortGpuIslandCpuCount,
            gpuShortGpuIslandMaxElements);
  gpuDisplaySyncTimings();
}

void compareGpuCpu(long nParticles, const char *label) {
  double absTol = gpuEnvDouble("ELEGANT_GPU_COMPARE_ABS", 1e-12);
  double defaultRelTol =
    label && strcmp(label, "field_table_tracking") == 0 ? 1e-10 : 1e-12;
  double relTol = gpuEnvDouble("ELEGANT_GPU_COMPARE_REL", defaultRelTol);
  double maxAbs = 0, maxRel = 0;
  double *gpuCopy;
  unsigned long count;
  float milliseconds = 0;
  int status;
  long ip, ic, stride, mismatches = 0;

  if (!gpuBase.deviceCoord || !gpuBase.deviceCurrent || nParticles <= 0)
    return;
  stride = gpuBase.deviceStride;
  count = (unsigned long)nParticles * (unsigned long)stride;
  gpuCopy = (double *)malloc(count * sizeof(*gpuCopy));
  if (!gpuCopy)
    gpuRequiredFailure("unable to allocate CUDA verification buffer");
  status = gpuCudaCopyDeviceToHost(gpuCopy, gpuBase.deviceCoord, count, &milliseconds);
  if (status != 0)
    gpuFatalStatus("cudaMemcpy(device to verify host)", status);
  gpuRecordMilliseconds(&gpuBase.gpuTransferToHostSeconds, milliseconds);

  for (ip = 0; ip < nParticles; ip++) {
    for (ic = 0; ic < stride; ic++) {
      double cpu = gpuBase.coord[ip][ic];
      double gpu = gpuCopy[ip * stride + ic];
      double absDiff = fabs(cpu - gpu);
      double scale = fmax(fabs(cpu), fabs(gpu));
      double relDiff = scale > DBL_MIN ? absDiff / scale : absDiff;
      if (absDiff > maxAbs)
        maxAbs = absDiff;
      if (relDiff > maxRel)
        maxRel = relDiff;
      if (!(absDiff <= absTol || relDiff <= relTol)) {
        if (mismatches < 10)
          fprintf(stderr,
                  "elegant CUDA VERIFY mismatch %s particle=%ld coord=%ld cpu=%.17e gpu=%.17e abs=%.3e rel=%.3e\n",
                  label ? label : "unknown", ip, ic, cpu, gpu, absDiff, relDiff);
        mismatches++;
      }
    }
  }
  free(gpuCopy);

  if (mismatches) {
    fprintf(stderr,
            "elegant CUDA VERIFY failed for %s: %ld mismatches, maxAbs=%.3e, maxRel=%.3e, absTol=%.3e, relTol=%.3e\n",
            label ? label : "unknown", mismatches, maxAbs, maxRel, absTol, relTol);
    exit(1);
  }
  if (gpuVerbose)
    fprintf(stderr, "elegant CUDA VERIFY passed for %s: maxAbs=%.3e maxRel=%.3e\n",
            label ? label : "unknown", maxAbs, maxRel);

  gpuBase.deviceCurrent = 0;
  gpuBase.hostCurrent = 1;
#ifdef GPU_VERIFY
  gpuCpuVerificationActive = 0;
#endif
}

void copyReductionArrays(double *centroid, double *sigma) {
  (void)sigma;
  if (centroid) {
    memcpy(gpuSavedCentroid, centroid, 6 * sizeof(*centroid));
    gpuSavedCentroidValid = 1;
  }
}

void compareReductionArrays(double *centroid, double *sigma, void *sums, const char *label) {
  (void)sigma;
  {
    double absTol = gpuCompareAbsTolerance();
    double relTol = gpuCompareRelTolerance();
    double maxAbs = 0, maxRel = 0;
    long mismatches = 0;
    long i, j;

    if (centroid && gpuSavedCentroidValid) {
      for (i = 0; i < 6; i++) {
        double absDiff, relDiff;
        if (!gpuValuesClose(centroid[i], gpuSavedCentroid[i], absTol, relTol, &absDiff, &relDiff)) {
          if (mismatches < 10)
            fprintf(stderr,
                    "elegant CUDA VERIFY reduction mismatch %s centroid[%ld] cpu=%.17e gpu=%.17e abs=%.3e rel=%.3e\n",
                    label ? label : "unknown", i, centroid[i], gpuSavedCentroid[i], absDiff, relDiff);
          mismatches++;
        }
        if (absDiff > maxAbs)
          maxAbs = absDiff;
        if (relDiff > maxRel)
          maxRel = relDiff;
      }
      gpuSavedCentroidValid = 0;
    }

    if (sums && gpuSavedSumsValid) {
      BEAM_SUMS *cpu = (BEAM_SUMS *)sums;
      BEAM_SUMS *gpu = &gpuSavedSums;
      if (cpu->n_part != gpu->n_part) {
        fprintf(stderr, "elegant CUDA VERIFY reduction mismatch %s n_part cpu=%ld gpu=%ld\n",
                label ? label : "unknown", cpu->n_part, gpu->n_part);
        mismatches++;
      }
      for (i = 0; i < 7; i++) {
        double absDiff, relDiff;
        if (!gpuValuesClose(cpu->centroid[i], gpu->centroid[i], absTol, relTol, &absDiff, &relDiff)) {
          if (mismatches < 10)
            fprintf(stderr,
                    "elegant CUDA VERIFY reduction mismatch %s centroid[%ld] cpu=%.17e gpu=%.17e abs=%.3e rel=%.3e\n",
                    label ? label : "unknown", i, cpu->centroid[i], gpu->centroid[i], absDiff, relDiff);
          mismatches++;
        }
        if (absDiff > maxAbs)
          maxAbs = absDiff;
        if (relDiff > maxRel)
          maxRel = relDiff;
      }
      if (cpu->beamSums2 && gpu->beamSums2) {
        for (i = 0; i < 7; i++) {
          double absDiff, relDiff;
          if (!gpuValuesClose(cpu->beamSums2->maxabs[i], gpu->beamSums2->maxabs[i], absTol, relTol, &absDiff, &relDiff) && mismatches++ < 10)
            fprintf(stderr, "elegant CUDA VERIFY reduction mismatch %s maxabs[%ld] cpu=%.17e gpu=%.17e abs=%.3e rel=%.3e\n",
                    label ? label : "unknown", i, cpu->beamSums2->maxabs[i], gpu->beamSums2->maxabs[i], absDiff, relDiff);
          if (absDiff > maxAbs)
            maxAbs = absDiff;
          if (relDiff > maxRel)
            maxRel = relDiff;
          if (!gpuValuesClose(cpu->beamSums2->max[i], gpu->beamSums2->max[i], absTol, relTol, &absDiff, &relDiff) && mismatches++ < 10)
            fprintf(stderr, "elegant CUDA VERIFY reduction mismatch %s max[%ld] cpu=%.17e gpu=%.17e abs=%.3e rel=%.3e\n",
                    label ? label : "unknown", i, cpu->beamSums2->max[i], gpu->beamSums2->max[i], absDiff, relDiff);
          if (absDiff > maxAbs)
            maxAbs = absDiff;
          if (relDiff > maxRel)
            maxRel = relDiff;
          if (!gpuValuesClose(cpu->beamSums2->min[i], gpu->beamSums2->min[i], absTol, relTol, &absDiff, &relDiff) && mismatches++ < 10)
            fprintf(stderr, "elegant CUDA VERIFY reduction mismatch %s min[%ld] cpu=%.17e gpu=%.17e abs=%.3e rel=%.3e\n",
                    label ? label : "unknown", i, cpu->beamSums2->min[i], gpu->beamSums2->min[i], absDiff, relDiff);
          if (absDiff > maxAbs)
            maxAbs = absDiff;
          if (relDiff > maxRel)
            maxRel = relDiff;
        }
        for (i = 0; i < 7; i++) {
          for (j = 0; j < 7; j++) {
            double absDiff, relDiff;
            if (!gpuValuesClose(cpu->beamSums2->sigma[i][j], gpu->beamSums2->sigma[i][j], absTol, relTol, &absDiff, &relDiff)) {
              if (mismatches < 10)
                fprintf(stderr,
                        "elegant CUDA VERIFY reduction mismatch %s sigma[%ld][%ld] cpu=%.17e gpu=%.17e abs=%.3e rel=%.3e\n",
                        label ? label : "unknown", i, j, cpu->beamSums2->sigma[i][j], gpu->beamSums2->sigma[i][j], absDiff, relDiff);
              mismatches++;
            }
            if (absDiff > maxAbs)
              maxAbs = absDiff;
            if (relDiff > maxRel)
              maxRel = relDiff;
          }
        }
      }
      gpuSavedSumsValid = 0;
    }

    if (mismatches) {
      fprintf(stderr,
              "elegant CUDA VERIFY reduction failed for %s: %ld mismatches, maxAbs=%.3e, maxRel=%.3e, absTol=%.3e, relTol=%.3e\n",
              label ? label : "unknown", mismatches, maxAbs, maxRel, absTol, relTol);
      exit(1);
    }
    if (gpuVerbose)
      fprintf(stderr, "elegant CUDA VERIFY reduction passed for %s: maxAbs=%.3e maxRel=%.3e\n",
              label ? label : "unknown", maxAbs, maxRel);
  }
  gpuBase.elementOnGpu = gpuCurrentElementEligible();
}

void *getGpuBeamSums(void *sums) {
  if (!sums)
    return NULL;
  gpuCopyBeamSumsShallow(&gpuSavedSums, (BEAM_SUMS *)sums);
  gpuSavedSumsValid = 1;
  return &gpuSavedSums;
}

void sortByPID(long nParticles) {
  (void)nParticles;
  gpuUnsupported("sortByPID");
}

void gpu_track_particles(void *matrix, long nParticles) {
  gpuLaunchTrackParticles((VMATRIX *)matrix, nParticles);
  gpuRecordWallSeconds();
}

void gpu_exactDrift(long nParticles, double length) {
  float milliseconds = 0;
  int status;

  if (nParticles <= 0)
    return;
  gpuCopyHostToDeviceInternal(nParticles, 0);
#ifndef GPU_VERIFY
  if (!gpuBase.verifyMode && !gpuBase.orderSensitiveOutputNeeded) {
    if (gpuPendingExactDriftCount &&
        gpuPendingExactDriftParticles != nParticles)
      gpuFlushPendingExactDrift("pending exactDrift particle-count change");
    gpuPendingExactDriftLength += length;
    gpuPendingExactDriftCount++;
    gpuPendingExactDriftParticles = nParticles;
    gpuBase.deviceCurrent = 1;
    gpuBase.hostCurrent = 0;
    gpuBase.nParticles = nParticles;
    gpuRecordWallSeconds();
    return;
  }
#endif
  status = gpuCudaExactDrift(gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride,
                             length, &milliseconds);
  if (status != 0)
    gpuFatalStatus("exactDrift kernel", status);
  gpuRecordWallSeconds();
  gpuRecordMilliseconds(&gpuBase.gpuKernelSeconds, milliseconds);
  gpuBase.deviceCurrent = 1;
  gpuBase.hostCurrent = 0;
  gpuBase.nParticles = nParticles;
  gpuBase.gpuExactDriftCount++;
}

static void gpu_linearDrift(long nParticles, double length) {
  float milliseconds = 0;
  int status;

  if (nParticles <= 0)
    return;
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaLinearDrift(gpuBase.deviceCoord, nParticles,
                              (int)gpuBase.deviceStride, length,
                              &milliseconds);
  if (status != 0)
    gpuFatalStatus("linear drift kernel", status);
  gpuRecordWallSeconds();
  gpuRecordMilliseconds(&gpuBase.gpuKernelSeconds, milliseconds);
  gpuBase.deviceCurrent = 1;
  gpuBase.hostCurrent = 0;
  gpuBase.nParticles = nParticles;
  gpuBase.gpuLinearDriftCount++;
}

void gpu_offset_beam(long nToTrack, MALIGN *offset, double P_central) {
  float milliseconds = 0;
  int status;
  long allParticles;

  if (!offset || nToTrack <= 0)
    return;
  if (offset->startPID >= 0 && offset->startPID > offset->endPID)
    gpuRequiredFailure("MALIGN startPID is greater than endPID");
  if ((offset->endPID >= 0 && offset->startPID < 0) ||
      (offset->startPID >= 0 && offset->endPID < 0))
    gpuRequiredFailure("MALIGN has invalid startPID/endPID range");

  allParticles = (offset->startPID == -1) && (offset->endPID == -1);
  gpuCopyHostToDevice(nToTrack);
  status = gpuCudaOffsetBeam(gpuBase.deviceCoord, nToTrack, (int)gpuBase.deviceStride,
                             offset->dx, offset->dxp, offset->dy, offset->dyp,
                             offset->dz, offset->dt, offset->dp, offset->de,
                             P_central, offset->startPID, offset->endPID,
                             (int)allParticles, c_mks, &milliseconds);
  if (status != 0)
    gpuFatalStatus("offset_beam kernel", status);
  gpuRecordHelperKernel(milliseconds);
  gpuMarkDeviceChanged(nToTrack);
  gpuRecordWallSeconds();
}

static long gpuMatchEnergyParticlesRemain(long np) {
#if USE_MPI
  if (distributedBeam && parallelStatus == trueParallel) {
    long npLocal = isMaster ? 0 : np;
    long npTotal = 0;
    MPI_Allreduce(&npLocal, &npTotal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    return npTotal > 0;
  }
#endif
  return np > 0;
}

void gpu_do_match_energy(long np, double *P_central, long change_beam) {
#if USE_MPI
  double **coord;
#endif
  GPU_BEAM_SUM_DATA result;
  double oldP, averageP;
  float milliseconds = 0;
  int status;
  long beamChanged;

  if (!P_central)
    gpuRequiredFailure("NULL central momentum pointer in gpu_do_match_energy");
  if (!gpuMatchEnergyParticlesRemain(np)) {
    gpuRecordWallSeconds();
    return;
  }

#if USE_MPI
  if (distributedBeam) {
    if (!do_match_energy)
      gpuRequiredFailure("CPU match-energy routine is unavailable");
    coord = forceParticlesToCpu("match energy CPU reference");
    do_match_energy(coord, np, P_central, change_beam);
    gpuRecordWallSeconds();
    return;
  }
#endif

  oldP = *P_central;
  gpuCopyHostToDevice(np);
  gpuEnsureRfcaScratch();
  status = gpuCudaMatchEnergyAndAverage(gpuBase.deviceCoord, np,
                                        (int)gpuBase.deviceStride,
                                        oldP, (int)change_beam,
                                        &result, gpuRfcaScratch.matchEnergy,
                                        &milliseconds);
  if (status != 0)
    gpuFatalStatus("match energy reduction/update kernel", status);
  gpuRecordHelperKernel(milliseconds);
  if (result.count <= 0) {
    gpuRecordWallSeconds();
    return;
  }
  averageP = result.centroidSum[5];
  beamChanged = change_beam ||
                oldP == 0 || fabs(averageP - oldP) / fabs(oldP) > 1e-14;

  if (beamChanged)
    gpuMarkDeviceChanged(np);
  if (!change_beam && beamChanged)
    *P_central = averageP;
  gpuRecordWallSeconds();
}

void gpu_set_central_momentum(long np, double P_new, double *P_central) {
  float milliseconds = 0;
  double oldP;
  int status;

  if (!P_central)
    gpuRequiredFailure("NULL central momentum pointer in gpu_set_central_momentum");
  oldP = *P_central;
  if (np <= 0) {
    *P_central = P_new;
    return;
  }
  if (oldP != P_new) {
    gpuCopyHostToDevice(np);
    status = gpuCudaSetCentralMomentum(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride,
                                       oldP, P_new, &milliseconds);
    if (status != 0)
      gpuFatalStatus("set central momentum kernel", status);
    gpuRecordHelperKernel(milliseconds);
    gpuMarkDeviceChanged(np);
  }
  *P_central = P_new;
  gpuRecordWallSeconds();
}

void gpu_center_beam(CENTER *center, long np, long iPass, double p0) {
  GPU_BEAM_SUM_DATA coordinateSums;
  double coordinateOffset[6] = {0, 0, 0, 0, 0, 0};
  double timeOffset = 0;
  unsigned int coordinateMask = 0;
  long coordinateSumsValid = 0;
  long timeSumsValid = 0;
  long doTimeCenter = 0;
  long needCoordinateSums = 0;
  long needTimeSum = 0;
  long ic;

  if (!center)
    return;
  if (center->onPass >= 0 && iPass != center->onPass)
    return;
  if (np > 0) {
    for (ic = 0; ic < 6; ic++) {
      if (center->doCoord[ic] && !center->deltaSet[ic]) {
        needCoordinateSums = 1;
        break;
      }
    }
    needTimeSum = center->doCoord[6] && !center->deltaSet[6];
    if (needCoordinateSums && needTimeSum) {
      gpuCentroidTimeSumsOnDevice(&coordinateSums, np, p0, "center coordinate/time reduction kernel");
      coordinateSumsValid = 1;
      timeSumsValid = 1;
    }
  }
  for (ic = 0; ic < 6; ic++) {
    double offset;
    if (!center->doCoord[ic])
      continue;
    if (!center->deltaSet[ic]) {
      if (np > 0) {
        if (!coordinateSumsValid) {
          gpuCentroidSumsOnDevice(&coordinateSums, np, "center coordinate reduction kernel");
          coordinateSumsValid = 1;
        }
        offset = coordinateSums.centroidSum[ic] / np;
      } else {
        offset = 0;
      }
      center->delta[ic] = offset;
      if (center->onceOnly)
        center->deltaSet[ic] = 1;
    } else {
      offset = center->delta[ic];
    }
    if (offset != 0) {
      coordinateOffset[ic] = offset;
      coordinateMask |= 1u << ic;
    }
  }
  if (center->doCoord[6]) {
    double offset;
    ic = 6;
    if (!center->deltaSet[ic]) {
      if (np > 0) {
        if (timeSumsValid)
          offset = coordinateSums.centroidSum[6] / np;
        else
          offset = gpuTimeSumOnDevice(np, p0) / np;
      } else {
        offset = 0;
      }
      center->delta[ic] = offset;
      if (center->onceOnly)
        center->deltaSet[ic] = 1;
    } else {
      offset = center->delta[ic];
    }
    if (offset != 0) {
      timeOffset = offset;
      doTimeCenter = 1;
    }
  }
  gpuApplyCenterOffsets(np, coordinateMask, coordinateOffset, doTimeCenter, p0, timeOffset);
  gpuRecordWallSeconds();
}

void gpu_collect_trajectory_data(double *centroid, long n_part) {
  gpu_compute_centroids(centroid, n_part);
}
void gpu_matr_element_tracking(VMATRIX *M, MATR *matr, long np, double z) {
  (void)z;

  if (np <= 0)
    return;
  if (!matr) {
    gpuLaunchTrackParticles(M, np);
  } else {
    if (!matr->fiducialSeen) {
      matr->sReference = gpuCoordinateSumOnDevice(np, 4) / np;
      matr->fiducialSeen = 1;
    }
    gpuLaunchTrackParticlesWithSReference(M, np, 1, matr->sReference);
  }
  gpuRecordWallSeconds();
}
void gpu_ematrix_element_tracking(VMATRIX *M, EMATRIX *matr, long np, double z, double *P_central) {
  (void)z;

  if (np <= 0)
    return;
  if (!matr) {
    gpuLaunchTrackParticles(M, np);
  } else {
    if (!matr->fiducialSeen) {
      matr->sReference = gpuCoordinateSumOnDevice(np, 4) / np;
      matr->fiducialSeen = 1;
    }
    gpuLaunchTrackParticlesWithSReference(M, np, 1, matr->sReference);
    if (matr->deltaP) {
      if (!P_central)
        gpuRequiredFailure("NULL central momentum pointer in gpu_ematrix_element_tracking");
      *P_central += matr->deltaP;
    }
  }
  gpuRecordWallSeconds();
}

static long gpuRectangularCollimatorOnCpu(RCOL *rcol, long np, double **accepted,
                                          double z, double Po, ELEMENT_LIST *eptr,
                                          const char *reason) {
  double **coord = forceParticlesToCpu(reason);
  long remaining;

  if (!rectangular_collimator)
    gpuRequiredFailure("CPU rectangular_collimator fallback is unavailable");
  gpuBase.elementOnGpu = 0;
  remaining = rectangular_collimator(coord, rcol, np, accepted, z, Po, eptr);
  gpuMarkHostWillChange();
  gpuRecordWallSeconds();
  return remaining;
}

static long gpuRectangularCollimatorLossCount(RCOL *rcol, long np, double length,
                                              long openCode) {
  long lostCount = 0;
  float milliseconds = 0;
  int status;

  if (np <= 0)
    return 0;
  gpuCopyHostToDevice(np);
  status = gpuCudaRectangularCollimatorLossCount(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride,
                                                 rcol->x_max, rcol->y_max, rcol->dx, rcol->dy,
                                                 length, openCode, &lostCount, &milliseconds);
  if (status != 0)
    gpuFatalStatus("rectangular collimator aperture predicate kernel", status);
  gpuRecordApertureKernel(milliseconds);
  return lostCount;
}

#ifndef GPU_VERIFY
static long gpuRectangularCollimatorStableCompact(RCOL *rcol, long np,
                                                  double **accepted,
                                                  double z, double Po,
                                                  ELEMENT_LIST *eptr,
                                                  long openCode) {
  long remaining = np;
  float milliseconds = 0;
  int status;

  if (np <= 0)
    return np;
  if (rcol->length > 0 && !gpuExactDriftParticleCountAllowed(np))
    return gpuRectangularCollimatorOnCpu(rcol, np, accepted, z, Po, eptr,
                                         "rectangular_collimator stable length drift below CUDA exact gate");
  gpuCopyHostToDevice(np);
  gpuEnsureApertureScratch(np);
  status = gpuCudaRectangularCollimatorStableCompact(
    gpuBase.deviceCoord, gpuApertureScratch.coord, gpuApertureScratch.prefix,
    np, (int)gpuBase.deviceStride, rcol->x_max, rcol->y_max,
    rcol->dx, rcol->dy, rcol->length, openCode, z, Po, &remaining, &milliseconds);
  if (status != 0)
    gpuFatalStatus("rectangular_collimator stable aperture compaction", status);
  gpuRecordApertureKernel(milliseconds);
  if (accepted) {
    if (!gpuStablePartitionAcceptedOnDevice(
          accepted, np, remaining,
          "rectangular_collimator CUDA accepted stable compaction")) {
      gpuCopyAperturePrefixToHost(np, "rectangular_collimator CUDA stable compaction accepted map");
      gpuStablePartitionAccepted(accepted, np, remaining);
    }
  }
  gpuPromoteApertureScratchCoord();
  gpuCopyApertureLossRowsToHost(
    remaining, np, "rectangular_collimator CUDA stable compaction loss tail output");
  if (gpuApertureLossRowsNeedHost())
    gpuSetRectangularCollimatorGlobalLossCoordinates(remaining, np, z, rcol->length,
                                                    eptr, openCode);
  gpuMarkDeviceChanged(remaining);
  gpuRecordWallSeconds();
  return remaining;
}
#endif

long gpu_rectangular_collimator(void *rcol0, long np, double **accepted,
                                double z, double Po, void *eptr) {
  RCOL *rcol = (RCOL *)rcol0;
  long openCode;

  if (np <= 0)
    return np;
  if (!rcol)
    gpuRequiredFailure("NULL RCOL pointer");
  if (!determineOpenSideCode && rcol->openSide && *rcol->openSide)
    gpuRequiredFailure("OPEN_SIDE decoding is unavailable");
  openCode = determineOpenSideCode ? determineOpenSideCode(rcol->openSide) : 0;
  if (rcol->invert)
    return gpuRectangularCollimatorOnCpu(rcol, np, accepted, z, Po, (ELEMENT_LIST *)eptr,
                                         "rectangular_collimator inverted aperture");
  if (rcol->x_max > 0 || rcol->y_max > 0) {
    if (gpuRectangularCollimatorLossCount(rcol, np, 0, openCode)) {
#ifndef GPU_VERIFY
      if (gpuEnableApertureParallelCompaction && (globalLossCoordOffset <= 0 || eptr))
        return gpuRectangularCollimatorStableCompact(rcol, np, accepted, z, Po,
                                                     (ELEMENT_LIST *)eptr, openCode);
#endif
      return gpuRectangularCollimatorOnCpu(rcol, np, accepted, z, Po, (ELEMENT_LIST *)eptr,
                                           "rectangular_collimator entrance loss fallback");
    }
    if (rcol->length > 0 && gpuRectangularCollimatorLossCount(rcol, np, rcol->length, openCode)) {
#ifndef GPU_VERIFY
      if (gpuEnableApertureParallelCompaction && (globalLossCoordOffset <= 0 || eptr))
        return gpuRectangularCollimatorStableCompact(rcol, np, accepted, z, Po,
                                                     (ELEMENT_LIST *)eptr, openCode);
#endif
      return gpuRectangularCollimatorOnCpu(rcol, np, accepted, z, Po, (ELEMENT_LIST *)eptr,
                                           "rectangular_collimator exit loss fallback");
    }
  }
  if (rcol->length > 0) {
    if (!gpuExactDriftParticleCountAllowed(np))
      return gpuRectangularCollimatorOnCpu(rcol, np, accepted, z, Po, (ELEMENT_LIST *)eptr,
                                           "rectangular_collimator length drift below CUDA exact gate");
    gpu_exactDrift(np, rcol->length);
  }
  gpuRecordWallSeconds();
  return np;
}

static long gpuLimitAmplitudesOnCpu(double xmax, double ymax, long np, double **accepted,
                                    double z, double Po, long extrapolate_z, long openCode,
                                    ELEMENT_LIST *eptr, const char *reason) {
  double **coord = forceParticlesToCpu(reason);
  long remaining;

  if (!limit_amplitudes)
    gpuRequiredFailure("CPU limit_amplitudes fallback is unavailable");
  gpuBase.elementOnGpu = 0;
  remaining = limit_amplitudes(coord, xmax, ymax, np, accepted, z, Po,
                               extrapolate_z, openCode, eptr);
  gpuMarkHostWillChange();
  gpuRecordWallSeconds();
  return remaining;
}

static long gpuLimitAmplitudeLossCount(double xmax, double ymax, long np) {
  long lostCount = 0;
  float milliseconds = 0;
  int status;

  if (np <= 0)
    return 0;
  gpuCopyHostToDevice(np);
  status = gpuCudaLimitAmplitudeLossCount(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride,
                                          xmax, ymax, &lostCount, &milliseconds);
  if (status != 0)
    gpuFatalStatus("limit_amplitudes aperture predicate kernel", status);
  gpuRecordApertureKernel(milliseconds);
  return lostCount;
}

#ifndef GPU_VERIFY
static long gpuLimitAmplitudesStableCompact(double xmax, double ymax, long np,
                                            double z, double Po,
                                            long extrapolate_z,
                                            double **accepted,
                                            ELEMENT_LIST *eptr) {
  long remaining = np;
  float milliseconds = 0;
  int status;

  if (np <= 0)
    return np;
  gpuCopyHostToDevice(np);
  gpuEnsureApertureScratch(np);
  status = gpuCudaLimitAmplitudesStableCompact(
    gpuBase.deviceCoord, gpuApertureScratch.coord, gpuApertureScratch.prefix,
    np, (int)gpuBase.deviceStride, xmax, ymax, z, Po, extrapolate_z,
    &remaining, &milliseconds);
  if (status != 0)
    gpuFatalStatus("limit_amplitudes stable aperture compaction", status);
  gpuRecordApertureKernel(milliseconds);
  if (accepted) {
    if (!gpuStablePartitionAcceptedOnDevice(
          accepted, np, remaining,
          "limit_amplitudes CUDA accepted stable compaction")) {
      gpuCopyAperturePrefixToHost(np, "limit_amplitudes CUDA stable compaction accepted map");
      gpuStablePartitionAccepted(accepted, np, remaining);
    }
  }
  gpuPromoteApertureScratchCoord();
  gpuCopyApertureLossRowsToHost(
    remaining, np, "limit_amplitudes CUDA stable compaction loss tail output");
  if (gpuApertureLossRowsNeedHost())
    gpuSetGlobalLossCoordinates(remaining, np, z, eptr, 0);
  gpuMarkDeviceChanged(remaining);
  gpuRecordWallSeconds();
  return remaining;
}
#endif

long gpu_limit_amplitudes(double xmax, double ymax, long np, double **accepted,
                          double z, double Po, long extrapolate_z, long openCode,
                          void *eptr) {
  long lostCount;

  if (np <= 0 || (xmax < 0 && ymax < 0) || (xmax == DBL_MAX && ymax == DBL_MAX))
    return np;
  if (openCode)
    return gpuLimitAmplitudesOnCpu(xmax, ymax, np, accepted, z, Po, extrapolate_z,
                                   openCode, (ELEMENT_LIST *)eptr,
                                   "limit_amplitudes open-side aperture");
  if (!gpuParticleCountAllowed(np, gpuBase.apertureMinParticles))
    return gpuLimitAmplitudesOnCpu(xmax, ymax, np, accepted, z, Po, extrapolate_z,
                                   openCode, (ELEMENT_LIST *)eptr,
                                   "limit_amplitudes below CUDA aperture threshold");
  lostCount = gpuLimitAmplitudeLossCount(xmax, ymax, np);
  if (!lostCount) {
    gpuRecordWallSeconds();
    return np;
  }
#ifndef GPU_VERIFY
  if (gpuEnableApertureParallelCompaction && (globalLossCoordOffset <= 0 || eptr))
    return gpuLimitAmplitudesStableCompact(xmax, ymax, np, z, Po,
                                           extrapolate_z, accepted,
                                           (ELEMENT_LIST *)eptr);
#endif
  return gpuLimitAmplitudesOnCpu(xmax, ymax, np, accepted, z, Po, extrapolate_z,
                                 openCode, (ELEMENT_LIST *)eptr,
                                 "limit_amplitudes particle loss fallback");
}
static long gpuRemoveInvalidParticlesOnCpu(long np, double **accepted,
                                           double z, double Po,
                                           const char *reason) {
  double **coord = forceParticlesToCpu(reason);
  long remaining;

  if (!removeInvalidParticles)
    gpuRequiredFailure("CPU removeInvalidParticles fallback is unavailable");
  gpuBase.elementOnGpu = 0;
  remaining = removeInvalidParticles(coord, np, accepted, z, Po);
  gpuMarkHostWillChange();
  gpuRecordWallSeconds();
  return remaining;
}

static long gpuRemoveInvalidParticlesLossCount(long np) {
  long lostCount = 0;
  float milliseconds = 0;
  int status;

  if (np <= 0)
    return 0;
  gpuCopyHostToDevice(np);
  status = gpuCudaRemoveInvalidParticlesLossCount(gpuBase.deviceCoord, np,
                                                  (int)gpuBase.deviceStride,
                                                  &lostCount, &milliseconds);
  if (status != 0)
    gpuFatalStatus("removeInvalidParticles predicate kernel", status);
  gpuRecordApertureKernel(milliseconds);
  return lostCount;
}

#ifndef GPU_VERIFY
static long gpuRemoveInvalidParticlesStableCompact(long np,
                                                   double **accepted,
                                                   double z, double Po) {
  long remaining = np;
  float milliseconds = 0;
  int status;

  if (np <= 0)
    return np;
  gpuCopyHostToDevice(np);
  gpuEnsureApertureScratch(np);
  status = gpuCudaRemoveInvalidParticlesStableCompact(
    gpuBase.deviceCoord, gpuApertureScratch.coord, gpuApertureScratch.prefix,
    np, (int)gpuBase.deviceStride, z, Po, &remaining, &milliseconds);
  if (status != 0)
    gpuFatalStatus("removeInvalidParticles stable compaction", status);
  gpuRecordApertureKernel(milliseconds);
  if (accepted) {
    if (!gpuStablePartitionAcceptedOnDevice(
          accepted, np, remaining,
          "removeInvalidParticles CUDA accepted stable compaction")) {
      gpuCopyAperturePrefixToHost(np, "removeInvalidParticles CUDA stable compaction accepted map");
      gpuStablePartitionAccepted(accepted, np, remaining);
    }
  }
  gpuPromoteApertureScratchCoord();
  gpuCopyApertureLossRowsToHost(
    remaining, np, "removeInvalidParticles CUDA stable compaction loss tail output");
  gpuMarkDeviceChanged(remaining);
  gpuRecordWallSeconds();
  return remaining;
}
#endif

long gpu_removeInvalidParticles(long np, double **accepted, double z, double Po) {
  long lostCount;

  if (np <= 0)
    return np;
  if (!gpuParticleCountAllowed(np, gpuBase.apertureMinParticles))
    return gpuRemoveInvalidParticlesOnCpu(np, accepted, z, Po,
                                          "removeInvalidParticles below CUDA aperture threshold");
  lostCount = gpuRemoveInvalidParticlesLossCount(np);
  if (!lostCount) {
    gpuRecordWallSeconds();
    return np;
  }
#ifndef GPU_VERIFY
  if (gpuEnableApertureParallelCompaction)
    return gpuRemoveInvalidParticlesStableCompact(np, accepted, z, Po);
#endif
  return gpuRemoveInvalidParticlesOnCpu(np, accepted, z, Po,
                                        "removeInvalidParticles particle loss fallback");
}

static long gpuEllipticalCollimatorOnCpu(ECOL *ecol, long np, double **accepted,
                                         double z, double Po, ELEMENT_LIST *eptr,
                                         const char *reason) {
  double **coord = forceParticlesToCpu(reason);
  long remaining;

  if (!elliptical_collimator)
    gpuRequiredFailure("CPU elliptical_collimator fallback is unavailable");
  gpuBase.elementOnGpu = 0;
  remaining = elliptical_collimator(coord, ecol, np, accepted, z, Po, eptr);
  gpuMarkHostWillChange();
  gpuRecordWallSeconds();
  return remaining;
}

static long gpuEllipticalCollimatorLossCount(ECOL *ecol, long np, double length,
                                             long exponent, long yExponent, long openCode) {
  long lostCount = 0;
  float milliseconds = 0;
  int status;

  if (np <= 0)
    return 0;
  gpuCopyHostToDevice(np);
  status = gpuCudaEllipticalCollimatorLossCount(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride,
                                                ecol->x_max, ecol->y_max, ecol->dx, ecol->dy,
                                                exponent, yExponent, length, openCode,
                                                &lostCount, &milliseconds);
  if (status != 0)
    gpuFatalStatus("elliptical collimator aperture predicate kernel", status);
  gpuRecordApertureKernel(milliseconds);
  return lostCount;
}

#ifndef GPU_VERIFY
static long gpuEllipticalCollimatorStableCompact(ECOL *ecol, long np,
                                                 double **accepted,
                                                 double z, double Po,
                                                 ELEMENT_LIST *eptr,
                                                 long exponent, long yExponent,
                                                 long openCode) {
  long remaining = np;
  float milliseconds = 0;
  int status;

  if (np <= 0)
    return np;
  if (ecol->length > 0 && !gpuExactDriftParticleCountAllowed(np))
    return gpuEllipticalCollimatorOnCpu(ecol, np, accepted, z, Po, eptr,
                                        "elliptical_collimator stable length drift below CUDA exact gate");
  gpuCopyHostToDevice(np);
  gpuEnsureApertureScratch(np);
  status = gpuCudaEllipticalCollimatorStableCompact(
    gpuBase.deviceCoord, gpuApertureScratch.coord, gpuApertureScratch.prefix,
    np, (int)gpuBase.deviceStride, ecol->x_max, ecol->y_max,
    ecol->dx, ecol->dy, exponent, yExponent, ecol->length, openCode, z, Po,
    &remaining, &milliseconds);
  if (status != 0)
    gpuFatalStatus("elliptical_collimator stable aperture compaction", status);
  gpuRecordApertureKernel(milliseconds);
  if (accepted) {
    if (!gpuStablePartitionAcceptedOnDevice(
          accepted, np, remaining,
          "elliptical_collimator CUDA accepted stable compaction")) {
      gpuCopyAperturePrefixToHost(np, "elliptical_collimator CUDA stable compaction accepted map");
      gpuStablePartitionAccepted(accepted, np, remaining);
    }
  }
  gpuPromoteApertureScratchCoord();
  gpuCopyApertureLossRowsToHost(
    remaining, np, "elliptical_collimator CUDA stable compaction loss tail output");
  if (gpuApertureLossRowsNeedHost())
    gpuSetGlobalLossCoordinatesDZ(remaining, np, z, eptr);
  gpuMarkDeviceChanged(remaining);
  gpuRecordWallSeconds();
  return remaining;
}
#endif

long gpu_elliptical_collimator(void *ecol0, long np, double **accepted,
                               double z, double Po, void *eptr) {
  ECOL *ecol = (ECOL *)ecol0;
  long openCode, ye;

  if (np <= 0)
    return np;
  if (!ecol)
    gpuRequiredFailure("NULL ECOL pointer");
  ye = ecol->yExponent ? ecol->yExponent : ecol->exponent;
  if (!determineOpenSideCode && ecol->openSide && *ecol->openSide)
    gpuRequiredFailure("OPEN_SIDE decoding is unavailable");
  openCode = determineOpenSideCode ? determineOpenSideCode(ecol->openSide) : 0;
  if (ecol->invert || ecol->exponent < 2 || ecol->exponent % 2 ||
      ye < 2 || ye % 2 || ecol->x_max <= 0 || ecol->y_max <= 0)
    return gpuEllipticalCollimatorOnCpu(ecol, np, accepted, z, Po, (ELEMENT_LIST *)eptr,
                                        "elliptical_collimator unsupported option");
  if (gpuEllipticalCollimatorLossCount(ecol, np, 0, ecol->exponent, ye, openCode)) {
#ifndef GPU_VERIFY
    if (gpuEnableApertureParallelCompaction && (globalLossCoordOffset <= 0 || eptr))
      return gpuEllipticalCollimatorStableCompact(ecol, np, accepted, z, Po,
                                                  (ELEMENT_LIST *)eptr,
                                                  ecol->exponent, ye, openCode);
#endif
    return gpuEllipticalCollimatorOnCpu(ecol, np, accepted, z, Po, (ELEMENT_LIST *)eptr,
                                        "elliptical_collimator entrance loss fallback");
  }
  if (ecol->length > 0 &&
      gpuEllipticalCollimatorLossCount(ecol, np, ecol->length,
                                       ecol->exponent, ye, openCode)) {
#ifndef GPU_VERIFY
    if (gpuEnableApertureParallelCompaction && (globalLossCoordOffset <= 0 || eptr))
      return gpuEllipticalCollimatorStableCompact(ecol, np, accepted, z, Po,
                                                  (ELEMENT_LIST *)eptr,
                                                  ecol->exponent, ye, openCode);
#endif
    return gpuEllipticalCollimatorOnCpu(ecol, np, accepted, z, Po, (ELEMENT_LIST *)eptr,
                                        "elliptical_collimator exit loss fallback");
  }
  if (ecol->length > 0) {
    if (!gpuExactDriftParticleCountAllowed(np))
      return gpuEllipticalCollimatorOnCpu(ecol, np, accepted, z, Po, (ELEMENT_LIST *)eptr,
                                          "elliptical_collimator length drift below CUDA exact gate");
    gpu_exactDrift(np, ecol->length);
  }
  gpuRecordWallSeconds();
  return np;
}

static long gpuELimitAmplitudesOnCpu(double xmax, double ymax, long np, double **accepted,
                                     double z, double Po, long extrapolate_z, long openCode,
                                     long exponent, long yExponent, ELEMENT_LIST *eptr,
                                     const char *reason) {
  double **coord = forceParticlesToCpu(reason);
  long remaining;

  if (!elimit_amplitudes)
    gpuRequiredFailure("CPU elimit_amplitudes fallback is unavailable");
  gpuBase.elementOnGpu = 0;
  remaining = elimit_amplitudes(coord, xmax, ymax, np, accepted, z, Po,
                                extrapolate_z, openCode, exponent, yExponent, eptr);
  gpuMarkHostWillChange();
  gpuRecordWallSeconds();
  return remaining;
}

static long gpuELimitAmplitudeLossCount(double xmax, double ymax, long np,
                                        long exponent, long yExponent) {
  long lostCount = 0;
  float milliseconds = 0;
  int status;

  if (np <= 0)
    return 0;
  gpuCopyHostToDevice(np);
  status = gpuCudaELimitAmplitudeLossCount(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride,
                                           xmax, ymax, exponent, yExponent,
                                           &lostCount, &milliseconds);
  if (status != 0)
    gpuFatalStatus("elimit_amplitudes aperture predicate kernel", status);
  gpuRecordApertureKernel(milliseconds);
  return lostCount;
}

#ifndef GPU_VERIFY
static long gpuELimitAmplitudesStableCompact(double xmax, double ymax, long np,
                                             double z, double Po,
                                             long extrapolate_z,
                                             long exponent,
                                             long yExponent,
                                             double **accepted,
                                             ELEMENT_LIST *eptr) {
  long remaining = np;
  float milliseconds = 0;
  int status;

  if (np <= 0)
    return np;
  gpuCopyHostToDevice(np);
  gpuEnsureApertureScratch(np);
  status = gpuCudaELimitAmplitudesStableCompact(
    gpuBase.deviceCoord, gpuApertureScratch.coord, gpuApertureScratch.prefix,
    np, (int)gpuBase.deviceStride, xmax, ymax, exponent, yExponent, z, Po,
    extrapolate_z, &remaining, &milliseconds);
  if (status != 0)
    gpuFatalStatus("elimit_amplitudes stable aperture compaction", status);
  gpuRecordApertureKernel(milliseconds);
  if (accepted) {
    if (!gpuStablePartitionAcceptedOnDevice(
          accepted, np, remaining,
          "elimit_amplitudes CUDA accepted stable compaction")) {
      gpuCopyAperturePrefixToHost(np, "elimit_amplitudes CUDA stable compaction accepted map");
      gpuStablePartitionAccepted(accepted, np, remaining);
    }
  }
  gpuPromoteApertureScratchCoord();
  gpuCopyApertureLossRowsToHost(
    remaining, np, "elimit_amplitudes CUDA stable compaction loss tail output");
  if (gpuApertureLossRowsNeedHost())
    gpuSetGlobalLossCoordinates(remaining, np, z, eptr, 1);
  gpuMarkDeviceChanged(remaining);
  gpuRecordWallSeconds();
  return remaining;
}
#endif

long gpu_elimit_amplitudes(double xmax, double ymax, long np, double **accepted,
                           double z, double Po, long extrapolate_z,
                           long openCode, long exponent, long yExponent, void *eptr) {
  long lostCount;
  long xe = exponent;
  long ye = yExponent ? yExponent : exponent;

  if (np <= 0 || (xmax == DBL_MAX && ymax == DBL_MAX))
    return np;
  if ((xe < 2 || xe % 2) || (ye < 2 || ye % 2))
    return gpuELimitAmplitudesOnCpu(xmax, ymax, np, accepted, z, Po, extrapolate_z,
                                    openCode, exponent, yExponent, (ELEMENT_LIST *)eptr,
                                    "elimit_amplitudes invalid exponent");
  if (xmax <= 0 || ymax <= 0) {
    if (xmax > 0 || ymax > 0)
      return gpu_limit_amplitudes(xmax, ymax, np, accepted, z, Po, extrapolate_z,
                                  openCode, eptr);
    return np;
  }
  if (openCode)
    return gpuELimitAmplitudesOnCpu(xmax, ymax, np, accepted, z, Po, extrapolate_z,
                                    openCode, exponent, yExponent, (ELEMENT_LIST *)eptr,
                                    "elimit_amplitudes open-side aperture");
  if (!gpuParticleCountAllowed(np, gpuBase.apertureMinParticles))
    return gpuELimitAmplitudesOnCpu(xmax, ymax, np, accepted, z, Po, extrapolate_z,
                                    openCode, exponent, yExponent, (ELEMENT_LIST *)eptr,
                                    "elimit_amplitudes below CUDA aperture threshold");
  lostCount = gpuELimitAmplitudeLossCount(xmax, ymax, np, xe, ye);
  if (!lostCount) {
    gpuRecordWallSeconds();
    return np;
  }
#ifndef GPU_VERIFY
  if (gpuEnableApertureParallelCompaction && (globalLossCoordOffset <= 0 || eptr))
    return gpuELimitAmplitudesStableCompact(xmax, ymax, np, z, Po,
                                            extrapolate_z, xe, ye, accepted,
                                            (ELEMENT_LIST *)eptr);
#endif
  return gpuELimitAmplitudesOnCpu(xmax, ymax, np, accepted, z, Po, extrapolate_z,
                                  openCode, exponent, yExponent, (ELEMENT_LIST *)eptr,
                                  "elimit_amplitudes particle loss fallback");
}
static long gpuBeamScraperOnCpu(SCRAPER *scraper, long np, double **accepted,
                                double z, double Po, ELEMENT_LIST *eptr,
                                const char *reason) {
  double **coord = forceParticlesToCpu(reason);
  long remaining;

  if (!beam_scraper)
    gpuRequiredFailure("CPU beam_scraper fallback is unavailable");
  gpuBase.elementOnGpu = 0;
  remaining = beam_scraper(coord, scraper, np, accepted, z, Po, eptr);
  gpuMarkHostWillChange();
  gpuRecordWallSeconds();
  return remaining;
}

static long gpuScraperLossCount(SCRAPER *scraper, long np, long plane,
                                long sideSign, double length) {
  long lostCount = 0;
  float milliseconds = 0;
  int status;

  if (np <= 0)
    return 0;
  gpuCopyHostToDevice(np);
  status = gpuCudaScraperLossCount(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride,
                                   (int)plane,
                                   plane == 0 ? scraper->dx : scraper->dy,
                                   scraper->position, (int)sideSign, length,
                                   &lostCount, &milliseconds);
  if (status != 0)
    gpuFatalStatus("scraper aperture predicate kernel", status);
  gpuRecordApertureKernel(milliseconds);
  return lostCount;
}

#ifndef GPU_VERIFY
static long gpuScraperStableCompact(SCRAPER *scraper, long np,
                                    double **accepted, double z,
                                    double Po, long plane,
                                    long sideSign, ELEMENT_LIST *eptr) {
  long remaining = np;
  float milliseconds = 0;
  int status;

  if (np <= 0)
    return np;
  if (scraper->length > 0 && !gpuExactDriftParticleCountAllowed(np))
    return gpuBeamScraperOnCpu(scraper, np, accepted, z, Po, eptr,
                               "beam_scraper stable length drift below CUDA exact gate");
  gpuCopyHostToDevice(np);
  gpuEnsureApertureScratch(np);
  status = gpuCudaScraperStableCompact(
    gpuBase.deviceCoord, gpuApertureScratch.coord, gpuApertureScratch.prefix,
    np, (int)gpuBase.deviceStride, (int)plane,
    plane == 0 ? scraper->dx : scraper->dy, scraper->position,
    (int)sideSign, scraper->length, z, Po, &remaining, &milliseconds);
  if (status != 0)
    gpuFatalStatus("beam_scraper stable aperture compaction", status);
  gpuRecordApertureKernel(milliseconds);
  if (accepted) {
    if (!gpuStablePartitionAcceptedOnDevice(
          accepted, np, remaining,
          "beam_scraper CUDA accepted stable compaction")) {
      gpuCopyAperturePrefixToHost(np, "beam_scraper CUDA stable compaction accepted map");
      gpuStablePartitionAccepted(accepted, np, remaining);
    }
  }
  gpuPromoteApertureScratchCoord();
  gpuCopyApertureLossRowsToHost(
    remaining, np, "beam_scraper CUDA stable compaction loss tail output");
  if (gpuApertureLossRowsNeedHost())
    gpuSetScraperGlobalLossCoordinates(remaining, np, z, scraper->length, eptr);
  gpuMarkDeviceChanged(remaining);
  gpuRecordWallSeconds();
  return remaining;
}
#endif

long gpu_beam_scraper(void *scraper0, long np, double **accepted,
                      double z, double Po, void *eptr) {
  SCRAPER *scraper = (SCRAPER *)scraper0;
  long direction, plane, sideSign;
  long dflag[2] = {0, 0};

  if (np <= 0)
    return np;
  if (!scraper)
    gpuRequiredFailure("NULL SCRAPER pointer");
  direction = interpretScraperDirection(scraper->insert_from, scraper->oldDirection);
  scraper->direction = direction;
  if (direction & DIRECTION_X) {
    plane = 0;
    dflag[0] = direction & DIRECTION_PLUS_X ? 1 : 0;
    dflag[1] = direction & DIRECTION_MINUS_X ? 1 : 0;
  } else if (direction & DIRECTION_Y) {
    plane = 2;
    dflag[0] = direction & DIRECTION_PLUS_Y ? 1 : 0;
    dflag[1] = direction & DIRECTION_MINUS_Y ? 1 : 0;
  } else {
    if (scraper->length) {
      if (!gpuExactDriftParticleCountAllowed(np))
        return gpuBeamScraperOnCpu(scraper, np, accepted, z, Po, (ELEMENT_LIST *)eptr,
                                   "beam_scraper length drift below CUDA exact gate");
      gpu_exactDrift(np, scraper->length);
    }
    gpuRecordWallSeconds();
    return np;
  }

  if (scraper->length && (scraper->Xo || scraper->Z))
    return gpuBeamScraperOnCpu(scraper, np, accepted, z, Po, (ELEMENT_LIST *)eptr,
                               "beam_scraper material interaction");
  if (dflag[0] && dflag[1]) {
    RCOL rcol;

    if (scraper->position <= 0)
      return gpuBeamScraperOnCpu(scraper, np, accepted, z, Po, (ELEMENT_LIST *)eptr,
                                 "beam_scraper two-sided all-loss aperture");
    rcol.length = scraper->length;
    rcol.x_max = plane == 0 ? scraper->position : 0;
    rcol.y_max = plane == 2 ? scraper->position : 0;
    rcol.dx = scraper->dx;
    rcol.dy = scraper->dy;
    rcol.openSide = NULL;
    rcol.invert = 0;
    return gpu_rectangular_collimator(&rcol, np, accepted, z, Po, eptr);
  }
  if (!dflag[0] && !dflag[1])
    return gpuBeamScraperOnCpu(scraper, np, accepted, z, Po, (ELEMENT_LIST *)eptr,
                               "beam_scraper unsupported insertion direction");

  sideSign = dflag[0] ? 1 : -1;
  if (gpuScraperLossCount(scraper, np, plane, sideSign, 0)) {
#ifndef GPU_VERIFY
    if (gpuEnableApertureParallelCompaction && (globalLossCoordOffset <= 0 || eptr))
      return gpuScraperStableCompact(scraper, np, accepted, z, Po, plane, sideSign,
                                     (ELEMENT_LIST *)eptr);
#endif
    return gpuBeamScraperOnCpu(scraper, np, accepted, z, Po, (ELEMENT_LIST *)eptr,
                               "beam_scraper entrance loss fallback");
  }
  if (scraper->length > 0 &&
      gpuScraperLossCount(scraper, np, plane, sideSign, scraper->length)) {
#ifndef GPU_VERIFY
    if (gpuEnableApertureParallelCompaction && (globalLossCoordOffset <= 0 || eptr))
      return gpuScraperStableCompact(scraper, np, accepted, z, Po, plane, sideSign,
                                     (ELEMENT_LIST *)eptr);
#endif
    return gpuBeamScraperOnCpu(scraper, np, accepted, z, Po, (ELEMENT_LIST *)eptr,
                               "beam_scraper exit loss fallback");
  }
  if (scraper->length > 0) {
    if (!gpuExactDriftParticleCountAllowed(np))
      return gpuBeamScraperOnCpu(scraper, np, accepted, z, Po, (ELEMENT_LIST *)eptr,
                                 "beam_scraper length drift below CUDA exact gate");
    gpu_exactDrift(np, scraper->length);
  }
  gpuRecordWallSeconds();
  return np;
}
static long gpuApertureDataLossCount(long np, double xCenter,
                                     double yCenter, double xSize,
                                     double ySize) {
  long lostCount = 0;
  float milliseconds = 0;
  int status;

  if (np <= 0)
    return 0;
  gpuCopyHostToDevice(np);
  status = gpuCudaApertureDataLossCount(gpuBase.deviceCoord, np,
                                        (int)gpuBase.deviceStride,
                                        xCenter, yCenter, xSize, ySize,
                                        &lostCount, &milliseconds);
  if (status != 0)
    gpuFatalStatus("aperture_data aperture predicate kernel", status);
  gpuRecordApertureKernel(milliseconds);
  return lostCount;
}

static long gpuApertureDataOnCpu(long np, double **accepted, double z,
                                 double Po, APERTURE_DATA *apData,
                                 ELEMENT_LIST *eptr, const char *reason) {
  double **coord = forceParticlesToCpu(reason);
  long remaining;

  if (!imposeApertureData)
    gpuRequiredFailure("CPU imposeApertureData fallback is unavailable");
  gpuBase.elementOnGpu = 0;
  remaining = imposeApertureData(coord, np, accepted, z, Po, apData, eptr);
  gpuMarkHostWillChange();
  gpuRecordWallSeconds();
  return remaining;
}

#ifndef GPU_VERIFY
static long gpuApertureDataStableCompact(long np, double **accepted,
                                         double z, double Po,
                                         double xCenter, double yCenter,
                                         double xSize, double ySize) {
  long remaining = np;
  float milliseconds = 0;
  int status;

  if (np <= 0)
    return np;
  gpuCopyHostToDevice(np);
  gpuEnsureApertureScratch(np);
  status = gpuCudaApertureDataStableCompact(
    gpuBase.deviceCoord, gpuApertureScratch.coord, gpuApertureScratch.prefix,
    np, (int)gpuBase.deviceStride, xCenter, yCenter, xSize, ySize,
    z, Po, &remaining, &milliseconds);
  if (status != 0)
    gpuFatalStatus("aperture_data stable aperture compaction", status);
  gpuRecordApertureKernel(milliseconds);
  if (accepted) {
    if (!gpuStablePartitionAcceptedOnDevice(
          accepted, np, remaining,
          "aperture_data CUDA accepted stable compaction")) {
      gpuCopyAperturePrefixToHost(np, "aperture_data CUDA stable compaction accepted map");
      gpuStablePartitionAccepted(accepted, np, remaining);
    }
  }
  gpuPromoteApertureScratchCoord();
  gpuCopyApertureLossRowsToHost(
    remaining, np, "aperture_data CUDA stable compaction loss tail output");
  gpuMarkDeviceChanged(remaining);
  gpuRecordWallSeconds();
  return remaining;
}
#endif

long gpu_imposeApertureData(long np, double **accepted, double z, double Po,
                            APERTURE_DATA *apData, ELEMENT_LIST *eptr) {
  double xCenter = 0, yCenter = 0, xSize = 0, ySize = 0;

  if (np <= 0)
    return np;
  if (!apData)
    gpuRequiredFailure("NULL aperture data pointer");
  if (!interpolateApertureData)
    return gpuApertureDataOnCpu(np, accepted, z, Po, apData, eptr,
                                "aperture_data interpolation fallback");
  if (!interpolateApertureData(z, apData, &xCenter, &yCenter, &xSize, &ySize)) {
    gpuRecordWallSeconds();
    return np;
  }
  if (!gpuParticleCountAllowed(np, gpuBase.apertureMinParticles))
    return gpuApertureDataOnCpu(np, accepted, z, Po, apData, eptr,
                                "aperture_data below CUDA aperture threshold");
  if (!gpuApertureDataLossCount(np, xCenter, yCenter, xSize, ySize)) {
    gpuRecordWallSeconds();
    return np;
  }
#ifndef GPU_VERIFY
  if (gpuEnableApertureParallelCompaction)
    return gpuApertureDataStableCompact(np, accepted, z, Po,
                                        xCenter, yCenter, xSize, ySize);
#endif
  return gpuApertureDataOnCpu(np, accepted, z, Po, apData, eptr,
                              "aperture_data particle loss fallback");
}
void gpu_compute_centroids(double *centroid, long n_part) {
  GPU_BEAM_SUM_DATA result;
  float milliseconds = 0;
  int status;
  long i;

  if (!centroid)
    return;
  for (i = 0; i < 6; i++)
    centroid[i] = 0;
  if (n_part <= 0)
    return;

  gpuCopyHostToDevice(n_part);
  status = gpuCudaCentroidSums(gpuBase.deviceCoord, n_part, (int)gpuBase.deviceStride,
                               &result, &milliseconds);
  if (status != 0)
    gpuFatalStatus("centroid reduction kernel", status);
  gpuRecordReductionKernel(milliseconds);
  for (i = 0; i < 6; i++)
    centroid[i] = result.centroidSum[i] / n_part;
  gpuRecordWallSeconds();
}

static long gpuBeamSumsSupported(BEAM_SUMS *sums, double *timeValue,
                                 double tMin, double tMax,
                                 long startPID, long endPID,
                                 unsigned long flags) {
  if (!sums)
    return 0;
  if (timeValue)
    return 0;
  if (tMin < tMax)
    return 0;
  if (!(startPID < 0 && endPID < 0))
    return 0;
  if (exactNormalizedEmittance || (flags & BEAM_SUMS_EXACTEMIT))
    return 0;
  if (sums->spinSums || spinCoordOffset)
    return 0;
  return 1;
}

static void gpuAccumulateBeamSumsOnCpu(void *sums, long n_part, double p_central, double mp_charge,
                                       double *timeValue, double tMin, double tMax,
                                       long startPID, long endPID, unsigned long flags) {
  long restoreElementOnGpu = gpuBase.elementOnGpu;
  forceParticlesToCpu("accumulate_beam_sums");
  if (accumulate_beam_sums)
    accumulate_beam_sums((BEAM_SUMS *)sums, gpuBase.coord, n_part, p_central, mp_charge,
                         timeValue, tMin, tMax, startPID, endPID, flags);
  else
    gpuUnsupported("gpu_accumulate_beam_sums");
  gpuBase.elementOnGpu = restoreElementOnGpu;
}

void gpu_accumulate_beam_sums(void *sums, long n_part, double p_central, double mp_charge,
                              double *timeValue, double tMin, double tMax,
                              long startPID, long endPID, unsigned long flags) {
  GPU_BEAM_SUM_DATA result, centeredResult;
  BEAM_SUMS *beamSums = (BEAM_SUMS *)sums;
  float milliseconds = 0;
  long oldCount, newCount, i, j, sparseAllowed;
  int status;
  static short sparse[7][7] = {
    {1, 1, 0, 0, 0, 1, 0},
    {0, 1, 0, 0, 0, 1, 0},
    {0, 0, 1, 1, 0, 1, 0},
    {0, 0, 0, 1, 0, 1, 0},
    {0, 0, 0, 0, 1, 1, 1},
    {0, 0, 0, 0, 0, 1, 0},
    {0, 0, 0, 0, 0, 0, 1}};
  double centroid[7];

  if (beamSums && beamSums->beamSums2 &&
      !gpuOutputDriftReductionMinParticlesExplicit) {
    gpuAccumulateBeamSumsOnCpu(sums, n_part, p_central, mp_charge,
                               timeValue, tMin, tMax, startPID, endPID, flags);
    return;
  }

  if (!gpuBeamSumsSupported(beamSums, timeValue, tMin, tMax, startPID, endPID, flags)) {
    gpuAccumulateBeamSumsOnCpu(sums, n_part, p_central, mp_charge,
                               timeValue, tMin, tMax, startPID, endPID, flags);
    return;
  }

  if (!beamSums->n_part)
    beamSums->p0 = p_central;
  if (n_part <= 0) {
    beamSums->charge = 0;
    return;
  }

  gpuCopyHostToDevice(n_part);
  if (beamSums->beamSums2) {
    gpuEnsureBeamSumsScratch();
    status = gpuCudaBeamSums2(gpuBase.deviceCoord, n_part,
                              (int)gpuBase.deviceStride,
                              p_central, c_mks, &result, &centeredResult,
                              gpuBeamSumsScratch.data, &milliseconds);
  } else {
    status = gpuCudaBeamSums(gpuBase.deviceCoord, n_part,
                             (int)gpuBase.deviceStride,
                             p_central, c_mks, &result, &milliseconds);
  }
  if (status != 0)
    gpuFatalStatus("compound beam sums reduction kernel", status);
  gpuRecordReductionKernel(milliseconds);

  if (result.count <= 0) {
    beamSums->charge = 0;
    gpuRecordWallSeconds();
    return;
  }

  oldCount = beamSums->n_part;
  newCount = oldCount + result.count;
  for (i = 0; i < 7; i++) {
    centroid[i] = result.centroidSum[i] / result.count;
    beamSums->centroid[i] = (beamSums->centroid[i] * oldCount + result.centroidSum[i]) / newCount;
  }

  if (beamSums->beamSums2) {
    if (centeredResult.count != result.count)
      gpuRequiredFailure("centered beam sums particle count mismatch");
  }
  gpuRecordWallSeconds();

  if (!(flags & BEAM_SUMS_NOMINMAX) && beamSums->beamSums2) {
    for (i = 0; i < 7; i++) {
      double minValue = result.min[i];
      double maxValue = result.max[i];
      double maxAbs = result.maxabs[i];
      if (i == 4 || i == 6) {
        minValue -= centroid[i];
        maxValue -= centroid[i];
        maxAbs = fmax(fabs(minValue), fabs(maxValue));
      }
      if (maxAbs > beamSums->beamSums2->maxabs[i])
        beamSums->beamSums2->maxabs[i] = maxAbs;
      if (maxValue > beamSums->beamSums2->max[i])
        beamSums->beamSums2->max[i] = maxValue;
      if (minValue < beamSums->beamSums2->min[i])
        beamSums->beamSums2->min[i] = minValue;
    }
  }

  if (beamSums->beamSums2) {
    for (i = 0; i < 7; i++) {
      for (j = i; j < 7; j++) {
        double Sij;
        sparseAllowed = !(flags & BEAM_SUMS_SPARSE) || sparse[i][j];
        if (!sparseAllowed) {
          beamSums->beamSums2->sigma[j][i] = beamSums->beamSums2->sigma[i][j] = 0;
          beamSums->beamSums2->sigman[j][i] = beamSums->beamSums2->sigman[i][j] = 0;
          continue;
        }
        Sij = centeredResult.productSum[gpuUpperTriangularIndex(i, j)];
        if (i == j && Sij < 0)
          Sij = 0;
        beamSums->beamSums2->sigma[j][i] =
          (beamSums->beamSums2->sigma[i][j] =
             (beamSums->beamSums2->sigma[i][j] * oldCount + Sij) / newCount);
      }
    }
  }

  beamSums->charge = mp_charge * result.count;
  beamSums->n_part = newCount;
}
long gpu_csr_csbend_wake_available(long nParticles, long nBins) {
  if (!gpuEnableCsrTracking)
    return 0;
  if (gpuBase.orderSensitiveOutputNeeded &&
      !gpuCsrTrackingExplicit && !gpuCsrResidentExplicit)
    return 0;
  if (!gpuBase.initialized || gpuBase.activeDevice < 0)
    return 0;
  if (nParticles <= 1 || nBins < 2)
    return 0;
  if (!gpuParticleCountAllowed(nParticles, gpuBase.csrMinParticles))
    return 0;
  if (!gpuBase.requiredMode && gpuBase.csrMinBins > 0 &&
      nBins < gpuBase.csrMinBins)
    return 0;
  return 1;
}

static long gpuPackCsrCsbendBodyTracking(GPU_CSBEND_DATA *data,
                                          CSRCSBEND *csbend,
                                          double sliceLength,
                                          double localRho0,
                                          double localRhoActual) {
  static GPU_CSBEND_DATA cachedData;
  static CSRCSBEND *cachedCsbend = NULL;
  static double cachedB[9], cachedC[9];
  static double cachedSliceLength = 0;
  static double cachedRho0 = 0;
  static double cachedRhoActual = 0;
  static double cachedCoordLimit = 0;
  static double cachedSlopeLimit = 0;
  static long cachedValid = 0;
  double h;

  if (!data || !csbend || !computeCSBENDFieldCoefficients ||
      !(&Fx_xy) || !(&Fy_xy) || !(&expansionOrder1) ||
      !(&hasSkew) || !(&hasNormal))
    return 0;
  if (sliceLength == 0 || localRho0 == 0 || localRhoActual == 0)
    return 0;
  if (csbend->integration_order != 2 &&
      csbend->integration_order != 4 &&
      csbend->integration_order != 6)
    return 0;

  if (cachedValid && cachedCsbend == csbend &&
      cachedSliceLength == sliceLength &&
      cachedRho0 == localRho0 &&
      cachedRhoActual == localRhoActual &&
      cachedCoordLimit == coordLimit &&
      cachedSlopeLimit == slopeLimit &&
      memcmp(cachedB, csbend->b, sizeof(cachedB)) == 0 &&
      memcmp(cachedC, csbend->c, sizeof(cachedC)) == 0) {
    *data = cachedData;
    return 1;
  }

  memset(data, 0, sizeof(*data));
  h = 1 / localRho0;
  computeCSBENDFieldCoefficients(csbend->b, csbend->c, h,
                                 csbend->nonlinear,
                                 csbend->expansionOrder);
  if (!Fx_xy || !Fy_xy || !*Fx_xy || !*Fy_xy)
    return 0;
  if (expansionOrder1 < 1 || expansionOrder1 > 11)
    return 0;

  data->nSlices = 1;
  data->integrationOrder = csbend->integration_order;
  data->expandHamiltonian = 0;
  data->hasSkew = hasSkew ? 1 : 0;
  data->hasNormal = hasNormal ? 1 : 0;
  data->expansionOrder1 = (int)expansionOrder1;
  data->length = sliceLength;
  data->rho0 = localRho0;
  data->rhoActual = localRhoActual;
  data->cosTilt = 1;
  data->sinTilt = 0;
  data->coordLimit = coordLimit;
  data->slopeLimit = slopeLimit;
  memcpy(data->Fx, *Fx_xy, sizeof(data->Fx));
  memcpy(data->Fy, *Fy_xy, sizeof(data->Fy));
  cachedData = *data;
  cachedCsbend = csbend;
  memcpy(cachedB, csbend->b, sizeof(cachedB));
  memcpy(cachedC, csbend->c, sizeof(cachedC));
  cachedSliceLength = sliceLength;
  cachedRho0 = localRho0;
  cachedRhoActual = localRhoActual;
  cachedCoordLimit = coordLimit;
  cachedSlopeLimit = slopeLimit;
  cachedValid = 1;
  return 1;
}

long gpu_csr_csbend_resident_available(void *csbend0, long nParticles,
                                       long nBins) {
  CSRCSBEND *csbend = (CSRCSBEND *)csbend0;

  if (!gpuEnableCsrResident)
    return 0;
  if (!csbend || !csbend->csr || csbend->backtrack || csbend->useMatrix ||
      csbend->synch_rad || csbend->isr || csbend->binOnce ||
      csbend->integratedGreensFunction || csbend->wffValues)
    return 0;
  if (csbend->wakeFilterFile && strlen(csbend->wakeFilterFile))
    return 0;
  if (csbend->particleOutputFile && strlen(csbend->particleOutputFile))
    return 0;
  if (csbend->derbenevCriterionMode &&
      strcmp(csbend->derbenevCriterionMode, "disable") != 0)
    return 0;
  return gpu_csr_csbend_wake_available(nParticles, nBins);
}

long gpu_csr_csbend_resident_begin(double *beta0, long nParticles) {
  float milliseconds = 0;
  unsigned long count;
  int status;
  long startedWallTimer = 0;

  if (!gpuEnableCsrResident || !beta0 || nParticles <= 1)
    return 0;
  if (gpuWallStart <= 0) {
    gpuWallStart = wallSeconds();
    startedWallTimer = 1;
  }

  gpuCopyHostToDevice(nParticles);
  gpuEnsureCsrKickDpScratch(nParticles);
  count = (unsigned long)nParticles;
  status = gpuCudaCopyHostToDevice(gpuCsrScratch.kickDp, beta0, count,
                                   &milliseconds);
  if (status != 0)
    gpuFatalStatus("cudaMemcpy(CSRCSBEND beta0 host to device)", status);
  gpuRecordMilliseconds(&gpuBase.gpuTransferToDeviceSeconds, milliseconds);

  if (startedWallTimer)
    gpuRecordWallSeconds();
  return 1;
}

long gpu_prepare_csbend_csr_histogram_device(double *lower, double *upper,
                                             double *binSize, long *bins,
                                             double expansionFactor,
                                             long nParticles, double Po) {
  GPU_DOUBLE_MIN_MAX_DATA range;
  float milliseconds = 0;
  int status;
  long startedWallTimer = 0;
  (void)Po;

  if (!gpuEnableCsrResident || !gpuBase.deviceCurrent ||
      !lower || !upper || !binSize || !bins || nParticles <= 1)
    return -1;
  if (*binSize <= 0 && *bins < 1)
    return -1;
  if (*binSize > 0 && *bins > 1)
    return -2;
  if (!gpuParticleCountAllowed(nParticles, gpuBase.csrMinParticles))
    return -1;

  if (gpuWallStart <= 0) {
    gpuWallStart = wallSeconds();
    startedWallTimer = 1;
  }
  if (!gpuCsrScratch.rangeResult) {
    status = gpuCudaMallocBytes(&gpuCsrScratch.rangeResult,
                                (unsigned long)sizeof(range));
    if (status != 0)
      gpuFatalStatus("cudaMalloc(CSR ct-range result)", status);
  }

  memset(&range, 0, sizeof(range));
  status = gpuCudaDoubleMinMax(gpuBase.deviceCoord, nParticles,
                               (int)gpuBase.deviceStride, 4,
                               &range, gpuCsrScratch.rangeResult,
                               &milliseconds);
  if (status != 0)
    gpuFatalStatus("CSRCSBEND resident ct-range reduction kernel", status);
  gpuRecordReductionKernel(milliseconds);

  if (range.count <= 0 || range.min == DBL_MAX || range.max == -DBL_MAX) {
    *lower = -sqrt(DBL_MAX);
    *upper = sqrt(DBL_MAX);
    *binSize = 0;
    *bins = 10;
    if (startedWallTimer)
      gpuRecordWallSeconds();
    return 0;
  }

  *lower = range.min;
  *upper = range.max;
  if (expansionFactor > 1) {
    double center = (*lower + *upper) / 2;
    double expandedRange = (*upper - *lower) * expansionFactor;
    *lower = center - expandedRange / 2;
    *upper = center + expandedRange / 2;
  }
  if (*binSize > 0)
    *bins = (*upper - *lower) / (*binSize);
  if (*bins < 2) {
    if (startedWallTimer)
      gpuRecordWallSeconds();
    return -1;
  }
  if (!gpuBase.requiredMode && gpuBase.csrMinBins > 0 &&
      *bins < gpuBase.csrMinBins) {
    if (startedWallTimer)
      gpuRecordWallSeconds();
    return -1;
  }
  *binSize = (*upper - *lower) / (*bins);

  if (startedWallTimer)
    gpuRecordWallSeconds();
  return range.count;
}

long gpu_compute_csbend_csr_histogram_device(double *ctHist,
                                             long nParticles, long nBins,
                                             double ctLower, double dct) {
  long nBinned = 0;
  long iBin;
  float milliseconds = 0;
  int status;
  long startedWallTimer = 0;
#ifdef GPU_VERIFY
  double *cpuHist = NULL;
  long cpuBinned;
#endif

  if (!gpuEnableCsrResident || !gpuBase.deviceCurrent ||
      !gpu_csr_csbend_wake_available(nParticles, nBins))
    return -1;
  if (!ctHist || nParticles <= 1 || nBins < 2 || dct <= 0)
    return -1;

  if (gpuWallStart <= 0) {
    gpuWallStart = wallSeconds();
    startedWallTimer = 1;
  }

  gpuEnsureCsrScratch(nBins);
  status = gpuCudaCsrHistogram(gpuBase.deviceCoord, nParticles,
                               (int)gpuBase.deviceStride, 4,
                               ctLower, dct, nBins,
                               (double *)gpuCsrScratch.ctHist, ctHist,
                               &milliseconds);
  if (status != 0)
    gpuFatalStatus("CSRCSBEND resident histogram CUDA kernel", status);
  gpuRecordCsrKernel(milliseconds);

  for (iBin = 0; iBin < nBins; iBin++)
    nBinned += (long)ctHist[iBin];

#ifdef GPU_VERIFY
  if (gpuBase.verifyMode) {
    double **coord = copyParticlesToCpuReadOnly("CSRCSBEND resident histogram verification");
    cpuHist = malloc(nBins * sizeof(*cpuHist));
    if (!cpuHist)
      gpuRequiredFailure("unable to allocate CUDA CSRCSBEND resident histogram verification buffer");
    cpuBinned = gpuComputeCsrHistogramCpu(cpuHist, coord, nParticles,
                                          nBins, ctLower, dct);
    if (cpuBinned != nBinned) {
      fprintf(stderr,
              "elegant CUDA VERIFY CSRCSBEND resident histogram bin-count mismatch cpu=%ld gpu=%ld\n",
              cpuBinned, nBinned);
      exit(1);
    }
    gpuCompareWakeArray("track_through_csbendCSR", "resident ctHist",
                        cpuHist, ctHist, nBins);
    free(cpuHist);
  }
#endif

  if (startedWallTimer)
    gpuRecordWallSeconds();
  return nBinned;
}

long gpu_apply_csbend_csr_kick_device(long nParticles, long nBins,
                                      double ctLower, double dct,
                                      double Po, double localRho0) {
  float milliseconds = 0;
  int status;
  long startedWallTimer = 0;

  if (!gpuEnableCsrResident || !gpuBase.deviceCurrent ||
      !gpu_csr_csbend_wake_available(nParticles, nBins))
    return 0;
  if (nParticles <= 0 || nBins < 2 || dct <= 0 || !Po || !localRho0)
    return 0;
  if (!gpuCsrScratch.dGamma || !gpuCsrScratch.dGammaValid ||
      gpuCsrScratch.dGammaBins != nBins)
    return 0;

  if (gpuWallStart <= 0) {
    gpuWallStart = wallSeconds();
    startedWallTimer = 1;
  }

  status = gpuCudaCsrCsbendKickInPlace(gpuBase.deviceCoord, nParticles,
                                       (int)gpuBase.deviceStride,
                                       (const double *)gpuCsrScratch.dGamma,
                                       nBins, ctLower, dct, Po, localRho0,
                                       &milliseconds);
  if (status != 0)
    gpuFatalStatus("CSRCSBEND resident kick CUDA kernel", status);
  gpuRecordCsrKernel(milliseconds);
  gpuMarkDeviceChanged(nParticles);

  if (startedWallTimer)
    gpuRecordWallSeconds();
  return 1;
}

long gpu_track_csbend_csr_body_slice(void *csbend0, long nParticles,
                                     double sliceLength, double localRho0,
                                     double localRhoActual, double Po) {
  CSRCSBEND *csbend = (CSRCSBEND *)csbend0;
  GPU_CSBEND_DATA data;
  long lostCount = 0;
  float milliseconds = 0;
  int status;
  long startedWallTimer = 0;
#ifdef GPU_VERIFY
  double *cpuStorage = NULL, **cpuCoord = NULL, *cpuBeta0 = NULL;
  double *gpuStorage = NULL;
  long stride = 0, ip, ic;
  unsigned long count = 0;
#endif

  if (!gpuEnableCsrResident || !gpuBase.deviceCurrent ||
      !csbend || nParticles <= 0 || !gpuCsrScratch.kickDp)
    return 0;
  if (!gpuPackCsrCsbendBodyTracking(&data, csbend, sliceLength,
                                    localRho0, localRhoActual))
    return 0;

  if (gpuWallStart <= 0) {
    gpuWallStart = wallSeconds();
    startedWallTimer = 1;
  }
#ifdef GPU_VERIFY
  if (gpuBase.verifyMode) {
    double **coord =
      copyParticlesToCpuReadOnly("CSRCSBEND resident body verification input");

    stride = gpuBase.deviceStride;
    count = (unsigned long)nParticles * (unsigned long)stride;
    cpuStorage = (double *)malloc(count * sizeof(*cpuStorage));
    gpuStorage = (double *)malloc(count * sizeof(*gpuStorage));
    cpuCoord = (double **)malloc(nParticles * sizeof(*cpuCoord));
    cpuBeta0 = (double *)malloc(nParticles * sizeof(*cpuBeta0));
    if (!cpuStorage || !gpuStorage || !cpuCoord || !cpuBeta0)
      gpuRequiredFailure("unable to allocate CUDA CSRCSBEND resident body verification buffers");
    for (ip = 0; ip < nParticles; ip++) {
      cpuCoord[ip] = cpuStorage + ip * stride;
      memcpy(cpuCoord[ip], coord[ip], (size_t)stride * sizeof(**cpuCoord));
    }
    status = gpuCudaCopyDeviceToHost(cpuBeta0, gpuCsrScratch.kickDp,
                                     (unsigned long)nParticles, &milliseconds);
    if (status != 0)
      gpuFatalStatus("cudaMemcpy(CSRCSBEND beta0 device to verify host)",
                     status);
    gpuRecordMilliseconds(&gpuBase.gpuTransferToHostSeconds, milliseconds);
    if (!gpu_verify_csrcsbend_cpu_body_slice)
      gpuRequiredFailure("CSRCSBEND resident body verification needs CPU CSBEND routines");

    for (ip = 0; ip < nParticles; ip++) {
      if (!gpu_verify_csrcsbend_cpu_body_slice(cpuCoord[ip], csbend,
                                               cpuBeta0[ip], sliceLength,
                                               localRho0, localRhoActual,
                                               Po))
        gpuRequiredFailure("CSRCSBEND resident body CPU-shadow particle loss");
    }
  }
#endif
  gpuEnsureCsrBodyScratch(nParticles, gpuBase.deviceStride);
  status = gpuCudaCsrCsbendBodySliceChecked(gpuBase.deviceCoord, nParticles,
                                            (int)gpuBase.deviceStride,
                                            &data,
                                            (const double *)gpuCsrScratch.kickDp,
                                            gpuCsrScratch.bodyBackup,
                                            gpuCsrScratch.bodyLostCount,
                                            &lostCount, &milliseconds);
  if (status != 0)
    gpuFatalStatus("CSRCSBEND resident body CUDA checked kernel", status);
  gpuRecordMagnetKernel(milliseconds);
#ifdef GPU_VERIFY
  if (gpuBase.verifyMode && !lostCount) {
    double absTol = gpuCompareAbsTolerance();
    double relTol = gpuCompareRelTolerance();
    double maxAbs = 0, maxRel = 0;
    long mismatches = 0;

    status = gpuCudaCopyDeviceToHost(gpuStorage, gpuBase.deviceCoord, count,
                                     &milliseconds);
    if (status != 0)
      gpuFatalStatus("cudaMemcpy(CSRCSBEND body device to verify host)",
                     status);
    gpuRecordMilliseconds(&gpuBase.gpuTransferToHostSeconds, milliseconds);
    for (ip = 0; ip < nParticles; ip++) {
      for (ic = 0; ic < 6 && ic < stride; ic++) {
        double absDiff, relDiff;
        double cpu = cpuCoord[ip][ic];
        double gpu = gpuStorage[ip * stride + ic];
        if (!gpuValuesClose(cpu, gpu, absTol, relTol, &absDiff, &relDiff)) {
          if (mismatches < 10)
            fprintf(stderr,
                    "elegant CUDA VERIFY mismatch CSRCSBEND resident body particle=%ld coord=%ld cpu=%.17e gpu=%.17e abs=%.3e rel=%.3e\n",
                    ip, ic, cpu, gpu, absDiff, relDiff);
          mismatches++;
        }
        if (absDiff > maxAbs)
          maxAbs = absDiff;
        if (relDiff > maxRel)
          maxRel = relDiff;
      }
    }
    if (mismatches) {
      fprintf(stderr,
              "elegant CUDA VERIFY failed for CSRCSBEND resident body: %ld mismatches, maxAbs=%.3e, maxRel=%.3e, absTol=%.3e, relTol=%.3e\n",
              mismatches, maxAbs, maxRel, absTol, relTol);
      exit(1);
    }
    if (gpuVerbose)
      fprintf(stderr,
              "elegant CUDA VERIFY passed for CSRCSBEND resident body: maxAbs=%.3e maxRel=%.3e\n",
              maxAbs, maxRel);
  }
#endif
  if (lostCount) {
#ifdef GPU_VERIFY
    free(cpuStorage);
    free(cpuCoord);
    free(cpuBeta0);
    free(gpuStorage);
#endif
    if (startedWallTimer)
      gpuRecordWallSeconds();
    return 0;
  }
  gpuMarkDeviceChanged(nParticles);

#ifdef GPU_VERIFY
  free(cpuStorage);
  free(cpuCoord);
  free(cpuBeta0);
  free(gpuStorage);
#endif
  if (startedWallTimer)
    gpuRecordWallSeconds();
  return 1;
}

#ifdef GPU_VERIFY
static long gpuVerifyCsrCsbendEnterSimpleCpu(double *coord, double *beta0,
                                             double pCentral,
                                             double coordinateSign,
                                             long edge1Effect, double e1,
                                             double psi1,
                                             double rhoActual) {
  double p0, beta;

  if (!coord || !beta0)
    return 0;
  p0 = pCentral * (1 + coord[5]);
  if (!isfinite(p0))
    return 0;
  beta = p0 / sqrt(p0 * p0 + 1);
  if (!isfinite(beta) || beta == 0)
    return 0;
  if (coordinateSign == -1) {
    coord[0] = -coord[0];
    coord[1] = -coord[1];
    coord[2] = -coord[2];
    coord[3] = -coord[3];
  }
  coord[4] /= beta;
  *beta0 = beta;
  if (edge1Effect) {
    double rho = (1 + coord[5]) * rhoActual;
    if (!isfinite(rho) || rho == 0)
      return 0;
    {
      double delta_xp = tan(e1) / rho * coord[0];
      coord[1] += delta_xp;
    }
    coord[3] -= tan(e1 - psi1 / (1 + coord[5])) / rho * coord[2];
  }
  return 1;
}

static long gpuVerifyCsrCsbendFinalizeSimpleCpu(double *coord,
                                                double pCentral,
                                                double coordinateSign,
                                                long edge2Effect, double e2,
                                                double psi2,
                                                double rhoActual) {
  double p1, beta1;

  if (!coord)
    return 0;
  p1 = pCentral * (1 + coord[5]);
  if (!isfinite(p1) || p1 <= 0)
    return 0;
  beta1 = p1 / sqrt(p1 * p1 + 1);
  coord[4] = coord[4] * beta1;
  if (edge2Effect) {
    double rho = (1 + coord[5]) * rhoActual;
    if (!isfinite(rho) || rho == 0)
      return 0;
    {
      double delta_xp = tan(e2) / rho * coord[0];
      coord[1] += delta_xp;
    }
    coord[3] -= tan(e2 - psi2 / (1 + coord[5])) / rho * coord[2];
  }
  if (coordinateSign == -1) {
    coord[0] = -coord[0];
    coord[1] = -coord[1];
    coord[2] = -coord[2];
    coord[3] = -coord[3];
  }
  return 1;
}

static void gpuVerifyCsrCsbendCompareCoords(const char *label,
                                            double **cpuCoord,
                                            const double *gpuStorage,
                                            long nParticles,
                                            long stride) {
  double absTol = gpuCompareAbsTolerance();
  double relTol = gpuCompareRelTolerance();
  double maxAbs = 0, maxRel = 0;
  long ip, ic, mismatches = 0;

  for (ip = 0; ip < nParticles; ip++) {
    for (ic = 0; ic < 6 && ic < stride; ic++) {
      double absDiff, relDiff;
      double cpu = cpuCoord[ip][ic];
      double gpu = gpuStorage[ip * stride + ic];
      if (!gpuValuesClose(cpu, gpu, absTol, relTol, &absDiff, &relDiff)) {
        if (mismatches < 10)
          fprintf(stderr,
                  "elegant CUDA VERIFY mismatch %s particle=%ld coord=%ld cpu=%.17e gpu=%.17e abs=%.3e rel=%.3e\n",
                  label, ip, ic, cpu, gpu, absDiff, relDiff);
        mismatches++;
      }
      if (absDiff > maxAbs)
        maxAbs = absDiff;
      if (relDiff > maxRel)
        maxRel = relDiff;
    }
  }
  if (mismatches) {
    fprintf(stderr,
            "elegant CUDA VERIFY failed for %s: %ld mismatches, maxAbs=%.3e, maxRel=%.3e, absTol=%.3e, relTol=%.3e\n",
            label, mismatches, maxAbs, maxRel, absTol, relTol);
    exit(1);
  }
  if (gpuVerbose)
    fprintf(stderr,
            "elegant CUDA VERIFY passed for %s: maxAbs=%.3e maxRel=%.3e\n",
            label, maxAbs, maxRel);
}

static void gpuVerifyCsrCsbendCompareBeta0(const double *cpuBeta0,
                                           const double *gpuBeta0,
                                           long nParticles) {
  double absTol = gpuCompareAbsTolerance();
  double relTol = gpuCompareRelTolerance();
  double maxAbs = 0, maxRel = 0;
  long ip, mismatches = 0;

  for (ip = 0; ip < nParticles; ip++) {
    double absDiff, relDiff;
    double cpu = cpuBeta0[ip];
    double gpu = gpuBeta0[ip];
    if (!gpuValuesClose(cpu, gpu, absTol, relTol, &absDiff, &relDiff)) {
      if (mismatches < 10)
        fprintf(stderr,
                "elegant CUDA VERIFY mismatch CSRCSBEND resident entry beta0 particle=%ld cpu=%.17e gpu=%.17e abs=%.3e rel=%.3e\n",
                ip, cpu, gpu, absDiff, relDiff);
      mismatches++;
    }
    if (absDiff > maxAbs)
      maxAbs = absDiff;
    if (relDiff > maxRel)
      maxRel = relDiff;
  }
  if (mismatches) {
    fprintf(stderr,
            "elegant CUDA VERIFY failed for CSRCSBEND resident entry beta0: %ld mismatches, maxAbs=%.3e, maxRel=%.3e, absTol=%.3e, relTol=%.3e\n",
            mismatches, maxAbs, maxRel, absTol, relTol);
    exit(1);
  }
  if (gpuVerbose)
    fprintf(stderr,
            "elegant CUDA VERIFY passed for CSRCSBEND resident entry beta0: maxAbs=%.3e maxRel=%.3e\n",
            maxAbs, maxRel);
}
#endif

long gpu_track_csbend_csr_enter_simple(long nParticles, double pCentral,
                                       double coordinateSign,
                                       long edge1Effect, double e1,
                                       double psi1, double rhoActual) {
  long lostCount = 0;
  float milliseconds = 0;
  int status;
  long startedWallTimer = 0;
#ifdef GPU_VERIFY
  double *cpuStorage = NULL, **cpuCoord = NULL;
  double *gpuStorage = NULL, *cpuBeta0 = NULL, *gpuBeta0 = NULL;
  long stride = 0, ip;
  unsigned long count = 0;
#endif

  if (!gpuEnableCsrResident || !gpuBase.deviceCurrent ||
      nParticles <= 0 || !pCentral ||
      (coordinateSign != 1 && coordinateSign != -1) ||
      (edge1Effect && rhoActual == 0))
    return 0;

  if (gpuWallStart <= 0) {
    gpuWallStart = wallSeconds();
    startedWallTimer = 1;
  }
#ifdef GPU_VERIFY
  if (gpuBase.verifyMode) {
    double **coord =
      copyParticlesToCpuReadOnly("CSRCSBEND resident entry verification input");

    stride = gpuBase.deviceStride;
    count = (unsigned long)nParticles * (unsigned long)stride;
    cpuStorage = (double *)malloc(count * sizeof(*cpuStorage));
    gpuStorage = (double *)malloc(count * sizeof(*gpuStorage));
    cpuCoord = (double **)malloc(nParticles * sizeof(*cpuCoord));
    cpuBeta0 = (double *)malloc(nParticles * sizeof(*cpuBeta0));
    gpuBeta0 = (double *)malloc(nParticles * sizeof(*gpuBeta0));
    if (!cpuStorage || !gpuStorage || !cpuCoord || !cpuBeta0 || !gpuBeta0)
      gpuRequiredFailure("unable to allocate CUDA CSRCSBEND entry verification buffers");
    for (ip = 0; ip < nParticles; ip++) {
      cpuCoord[ip] = cpuStorage + ip * stride;
      memcpy(cpuCoord[ip], coord[ip], (size_t)stride * sizeof(**cpuCoord));
      if (!gpuVerifyCsrCsbendEnterSimpleCpu(cpuCoord[ip], cpuBeta0 + ip,
                                            pCentral, coordinateSign,
                                            edge1Effect, e1, psi1,
                                            rhoActual))
        gpuRequiredFailure("CSRCSBEND resident entry CPU-shadow particle loss");
    }
  }
#endif
  gpuEnsureCsrKickDpScratch(nParticles);
  gpuEnsureCsrBodyScratch(nParticles, gpuBase.deviceStride);
  status = gpuCudaCsrCsbendEnterSimpleChecked(
    gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride,
    pCentral, coordinateSign, edge1Effect ? 1 : 0, e1, psi1, rhoActual,
    (double *)gpuCsrScratch.kickDp,
    gpuCsrScratch.bodyBackup, gpuCsrScratch.bodyLostCount,
    &lostCount, &milliseconds);
  if (status != 0)
    gpuFatalStatus("CSRCSBEND resident initial-coordinate CUDA checked kernel", status);
  gpuRecordMagnetKernel(milliseconds);
  if (lostCount) {
#ifdef GPU_VERIFY
    free(cpuStorage);
    free(cpuCoord);
    free(gpuStorage);
    free(cpuBeta0);
    free(gpuBeta0);
#endif
    if (startedWallTimer)
      gpuRecordWallSeconds();
    return 0;
  }
#ifdef GPU_VERIFY
  if (gpuBase.verifyMode) {
    status = gpuCudaCopyDeviceToHost(gpuStorage, gpuBase.deviceCoord, count,
                                     &milliseconds);
    if (status != 0)
      gpuFatalStatus("cudaMemcpy(CSRCSBEND entry device to verify host)",
                     status);
    gpuRecordMilliseconds(&gpuBase.gpuTransferToHostSeconds, milliseconds);
    status = gpuCudaCopyDeviceToHost(gpuBeta0, gpuCsrScratch.kickDp,
                                     (unsigned long)nParticles, &milliseconds);
    if (status != 0)
      gpuFatalStatus("cudaMemcpy(CSRCSBEND entry beta0 device to verify host)",
                     status);
    gpuRecordMilliseconds(&gpuBase.gpuTransferToHostSeconds, milliseconds);
    gpuVerifyCsrCsbendCompareCoords("CSRCSBEND resident entry",
                                    cpuCoord, gpuStorage, nParticles, stride);
    gpuVerifyCsrCsbendCompareBeta0(cpuBeta0, gpuBeta0, nParticles);
  }
#endif
  gpuMarkDeviceChanged(nParticles);

#ifdef GPU_VERIFY
  free(cpuStorage);
  free(cpuCoord);
  free(gpuStorage);
  free(cpuBeta0);
  free(gpuBeta0);
#endif
  if (startedWallTimer)
    gpuRecordWallSeconds();
  return 1;
}

long gpu_track_csbend_csr_finalize_simple(long nParticles, double pCentral,
                                          double coordinateSign,
                                          long edge2Effect, double e2,
                                          double psi2, double rhoActual) {
  long lostCount = 0;
  float milliseconds = 0;
  int status;
  long startedWallTimer = 0;
#ifdef GPU_VERIFY
  double *cpuStorage = NULL, **cpuCoord = NULL;
  double *gpuStorage = NULL;
  long stride = 0, ip;
  unsigned long count = 0;
#endif

  if (!gpuEnableCsrResident || !gpuBase.deviceCurrent ||
      nParticles <= 0 || !pCentral)
    return 0;

  if (gpuWallStart <= 0) {
    gpuWallStart = wallSeconds();
    startedWallTimer = 1;
  }
#ifdef GPU_VERIFY
  if (gpuBase.verifyMode) {
    double **coord =
      copyParticlesToCpuReadOnly("CSRCSBEND resident finalize verification input");

    stride = gpuBase.deviceStride;
    count = (unsigned long)nParticles * (unsigned long)stride;
    cpuStorage = (double *)malloc(count * sizeof(*cpuStorage));
    gpuStorage = (double *)malloc(count * sizeof(*gpuStorage));
    cpuCoord = (double **)malloc(nParticles * sizeof(*cpuCoord));
    if (!cpuStorage || !gpuStorage || !cpuCoord)
      gpuRequiredFailure("unable to allocate CUDA CSRCSBEND finalize verification buffers");
    for (ip = 0; ip < nParticles; ip++) {
      cpuCoord[ip] = cpuStorage + ip * stride;
      memcpy(cpuCoord[ip], coord[ip], (size_t)stride * sizeof(**cpuCoord));
      if (!gpuVerifyCsrCsbendFinalizeSimpleCpu(cpuCoord[ip], pCentral,
                                               coordinateSign,
                                               edge2Effect, e2, psi2,
                                               rhoActual))
        gpuRequiredFailure("CSRCSBEND resident finalize CPU-shadow particle loss");
    }
  }
#endif
  gpuEnsureCsrBodyScratch(nParticles, gpuBase.deviceStride);
  status = gpuCudaCsrCsbendFinalizeSimpleChecked(
    gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride,
    pCentral, coordinateSign, edge2Effect ? 1 : 0, e2, psi2, rhoActual,
    gpuCsrScratch.bodyBackup,
    gpuCsrScratch.bodyLostCount, &lostCount, &milliseconds);
  if (status != 0)
    gpuFatalStatus("CSRCSBEND resident final-coordinate CUDA checked kernel", status);
  gpuRecordMagnetKernel(milliseconds);
  if (lostCount) {
#ifdef GPU_VERIFY
    free(cpuStorage);
    free(cpuCoord);
    free(gpuStorage);
#endif
    if (startedWallTimer)
      gpuRecordWallSeconds();
    return 0;
  }
#ifdef GPU_VERIFY
  if (gpuBase.verifyMode) {
    status = gpuCudaCopyDeviceToHost(gpuStorage, gpuBase.deviceCoord, count,
                                     &milliseconds);
    if (status != 0)
      gpuFatalStatus("cudaMemcpy(CSRCSBEND finalize device to verify host)",
                     status);
    gpuRecordMilliseconds(&gpuBase.gpuTransferToHostSeconds, milliseconds);
    gpuVerifyCsrCsbendCompareCoords("CSRCSBEND resident finalize",
                                    cpuCoord, gpuStorage, nParticles, stride);
  }
#endif
  gpuMarkDeviceChanged(nParticles);

#ifdef GPU_VERIFY
  free(cpuStorage);
  free(cpuCoord);
  free(gpuStorage);
#endif
  if (startedWallTimer)
    gpuRecordWallSeconds();
  return 1;
}

long gpu_copy_csbend_csr_beta0(double *beta0, long nParticles) {
  float milliseconds = 0;
  unsigned long count;
  int status;

  if (!beta0 || nParticles <= 0 || !gpuCsrScratch.kickDp)
    return 0;
  count = (unsigned long)nParticles;
  status = gpuCudaCopyDeviceToHost(beta0, gpuCsrScratch.kickDp, count,
                                   &milliseconds);
  if (status != 0)
    gpuFatalStatus("cudaMemcpy(CSRCSBEND beta0 device to host)", status);
  gpuRecordMilliseconds(&gpuBase.gpuTransferToHostSeconds, milliseconds);
  return 1;
}

#ifdef GPU_VERIFY
static long gpuComputeCsrHistogramCpu(double *hist, double **part,
                                      long nParticles, long nBins,
                                      double ctLower, double dct) {
  long iParticle, iBin, nBinned;

  for (iBin = 0; iBin < nBins; iBin++)
    hist[iBin] = 0;
  for (iParticle = nBinned = 0; iParticle < nParticles; iParticle++) {
    double value = part[iParticle][4];
    if (isinf(value) || isnan(value))
      continue;
    iBin = (value - ctLower) / dct;
    if (iBin < 0 || iBin > nBins - 1)
      continue;
    hist[iBin] += 1;
    nBinned++;
  }
  return nBinned;
}
#endif

#ifdef GPU_VERIFY
static void gpuComputeCsbendCsrWakeCpu(double *dGamma, double *T1, double *T2,
                                       const double *ctHist,
                                       const double *ctHistDeriv,
                                       const double *denom,
                                       long nBins,
                                       double CSRConstant,
                                       double dsSlice,
                                       double slippageLength13,
                                       double dct,
                                       long steadyState,
                                       long trapazoidIntegration,
                                       long diSlippage,
                                       long diSlippage4) {
  long iBin, iBinBehind;

  for (iBin = 0; iBin < nBins; iBin++) {
    double term1 = 0, term2 = 0;
    long count = 0;
    T1[iBin] = T2[iBin] = 0;
    if (CSRConstant) {
      if (steadyState) {
        if (!trapazoidIntegration) {
          for (iBinBehind = iBin + 1; iBinBehind < nBins; iBinBehind++)
            T1[iBin] += ctHistDeriv[iBinBehind] / denom[iBinBehind - iBin];
        } else {
          if ((iBinBehind = iBin + 1) < nBins)
            term1 = ctHistDeriv[iBinBehind] / denom[iBinBehind - iBin];
          for (count = 0, iBinBehind = iBin + 1; iBinBehind < nBins;
               iBinBehind++, count++)
            T1[iBin] +=
              (term2 = ctHistDeriv[iBinBehind] / denom[iBinBehind - iBin]);
          if ((iBin + 1) < nBins)
            T1[iBin] +=
              0.3 * sqr(denom[1]) *
              (2 * ctHistDeriv[iBin + 1] + 3 * ctHistDeriv[iBin]) / dct;
          if (count > 1)
            T1[iBin] -= (term1 + term2) / 2;
        }
      } else {
        if (!trapazoidIntegration) {
          for (iBinBehind = iBin + 1;
               iBinBehind <= (iBin + diSlippage) && iBinBehind < nBins;
               iBinBehind++)
            T1[iBin] += ctHistDeriv[iBinBehind] / denom[iBinBehind - iBin];
        } else {
          if ((iBinBehind = iBin + 1) < nBins &&
              iBinBehind <= (iBin + diSlippage))
            term1 = ctHistDeriv[iBinBehind] / denom[iBinBehind - iBin] / 2;
          for (count = 0, iBinBehind = iBin + 1;
               iBinBehind <= (iBin + diSlippage) && iBinBehind < nBins;
               count++, iBinBehind++)
            T1[iBin] +=
              (term2 = ctHistDeriv[iBinBehind] / denom[iBinBehind - iBin]);
          if (diSlippage > 0 && (iBin + 1) < nBins)
            T1[iBin] +=
              0.3 * sqr(denom[1]) *
              (2 * ctHistDeriv[iBin + 1] + 3 * ctHistDeriv[iBin]) / dct;
          if (count > 1)
            T1[iBin] -= (term1 + term2) / 2;
        }
        if ((iBin + diSlippage) < nBins)
          T2[iBin] += ctHist[iBin + diSlippage];
        if ((iBin + diSlippage4) < nBins)
          T2[iBin] -= ctHist[iBin + diSlippage4];
      }
      T1[iBin] *= CSRConstant * dsSlice;
      T2[iBin] *= -CSRConstant * dsSlice / slippageLength13;
    }
    dGamma[iBin] = T1[iBin] + T2[iBin];
  }
}
#endif

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
                                 long copyDGammaArray) {
  float milliseconds = 0;
  unsigned long count;
  int status;
  long ok = 0;
  long startedWallTimer = 0;
  long copyWakeComponents = copyComponentArrays;
  long copyDGamma = copyDGammaArray;
  const double *deviceCtHist = NULL;
  const double *deviceCtHistDeriv = NULL;

  if (!gpu_csr_csbend_wake_available(nParticles, nBins))
    return 0;
  if (!dGamma || !T1 || !T2 || !ctHist || !ctHistDeriv || !denom)
    return 0;
  if (!CSRConstant || dsSlice == 0 || dct <= 0 || slippageLength13 == 0)
    return 0;
  if ((steadyState != 0 && steadyState != 1) ||
      (trapazoidIntegration != 0 && trapazoidIntegration != 1))
    return 0;

  if (gpuWallStart <= 0) {
    gpuWallStart = wallSeconds();
    startedWallTimer = 1;
  }

  count = (unsigned long)nBins;
  gpuEnsureCsrScratch(nBins);

  gpuUploadCsrWakeInputs(ctHist, ctHistDeriv, nBins,
                         &deviceCtHist, &deviceCtHistDeriv);
  gpuUploadCsrDenomIfNeeded(denom, nBins, dct);

  status = gpuCudaCsrCsbendWake(deviceCtHist,
                                deviceCtHistDeriv,
                                (const double *)gpuCsrScratch.denom,
                                (double *)gpuCsrScratch.T1,
                                (double *)gpuCsrScratch.T2,
                                (double *)gpuCsrScratch.dGamma,
                                nBins, CSRConstant, dsSlice,
                                slippageLength13, dct,
                                steadyState, trapazoidIntegration,
                                diSlippage, diSlippage4,
                                &milliseconds);
  if (status != 0)
    gpuFatalStatus("CSRCSBEND wake CUDA kernel", status);
  gpuRecordCsrKernel(milliseconds);
  gpuCsrScratch.dGammaValid = 1;
  gpuCsrScratch.dGammaBins = nBins;

#ifdef GPU_VERIFY
  if (gpuBase.verifyMode) {
    copyWakeComponents = 1;
    copyDGamma = 1;
  }
#endif
  if (copyWakeComponents) {
    status = gpuCudaCopyDeviceToHost(T1, gpuCsrScratch.T1, count, &milliseconds);
    if (status != 0)
      gpuFatalStatus("cudaMemcpy(CSR T1 device to host)", status);
    gpuRecordMilliseconds(&gpuBase.gpuTransferToHostSeconds, milliseconds);
    status = gpuCudaCopyDeviceToHost(T2, gpuCsrScratch.T2, count, &milliseconds);
    if (status != 0)
      gpuFatalStatus("cudaMemcpy(CSR T2 device to host)", status);
    gpuRecordMilliseconds(&gpuBase.gpuTransferToHostSeconds, milliseconds);
  }
  if (copyDGamma) {
    status = gpuCudaCopyDeviceToHost(dGamma, gpuCsrScratch.dGamma, count,
                                     &milliseconds);
    if (status != 0)
      gpuFatalStatus("cudaMemcpy(CSR dGamma device to host)", status);
    gpuRecordMilliseconds(&gpuBase.gpuTransferToHostSeconds, milliseconds);
  }
#ifdef GPU_VERIFY
  if (gpuBase.verifyMode) {
    double *cpuT1 = malloc(nBins * sizeof(*cpuT1));
    double *cpuT2 = malloc(nBins * sizeof(*cpuT2));
    double *cpuDGamma = malloc(nBins * sizeof(*cpuDGamma));
    if (!cpuT1 || !cpuT2 || !cpuDGamma)
      gpuRequiredFailure("unable to allocate CUDA CSRCSBEND verification buffers");
    gpuComputeCsbendCsrWakeCpu(cpuDGamma, cpuT1, cpuT2, ctHist, ctHistDeriv,
                               denom, nBins, CSRConstant, dsSlice,
                               slippageLength13, dct, steadyState,
                               trapazoidIntegration, diSlippage,
                               diSlippage4);
    gpuCompareWakeArray("track_through_csbendCSR", "T1", cpuT1, T1,
                        nBins);
    gpuCompareWakeArray("track_through_csbendCSR", "T2", cpuT2, T2,
                        nBins);
    gpuCompareWakeArray("track_through_csbendCSR", "dGamma", cpuDGamma,
                        dGamma, nBins);
    gpuStoreCsrWakeCpuShadow(cpuDGamma, nBins);
    free(cpuT1);
    free(cpuT2);
    free(cpuDGamma);
  }
#endif
  ok = 1;

  if (startedWallTimer)
    gpuRecordWallSeconds();
  return ok;
}

long gpu_copy_csbend_csr_dgamma(double *dGamma, long nBins) {
  float milliseconds = 0;
  unsigned long count;
  int status;

  if (!dGamma || nBins <= 0 || !gpuCsrScratch.dGamma ||
      !gpuCsrScratch.dGammaValid || gpuCsrScratch.dGammaBins != nBins)
    return 0;
  count = (unsigned long)nBins;
  status = gpuCudaCopyDeviceToHost(dGamma, gpuCsrScratch.dGamma, count,
                                   &milliseconds);
  if (status != 0)
    gpuFatalStatus("cudaMemcpy(CSR dGamma device to host)", status);
  gpuRecordMilliseconds(&gpuBase.gpuTransferToHostSeconds, milliseconds);
  return 1;
}

long gpu_track_through_csbendCSR(long n_part, void *csbend, double p_error,
                                 double Po, double **accepted,
                                 double z_start, double z_end, void *charge,
                                 char *rootname, void *maxamp,
                                 void *apContour, void *apFileData,
                                 void *eptr) {
  if (!track_through_csbendCSR_cuda_resident_entry) {
    gpuUnsupported("gpu_track_through_csbendCSR");
    return 0;
  }
  return track_through_csbendCSR_cuda_resident_entry(
    gpuBase.coord, n_part, (CSRCSBEND *)csbend, p_error, Po, accepted,
    z_start, z_end, (CHARGE *)charge, rootname, (MAXAMP *)maxamp,
    (APCONTOUR *)apContour, (APERTURE_DATA *)apFileData,
    (ELEMENT_LIST *)eptr);
}
long gpu_track_through_driftCSR(long np, void *csrDrift0, double Po,
                                double **accepted, double zStart,
                                double revolutionLength, void *charge,
                                char *rootname) {
  CSRDRIFT *csrDrift = (CSRDRIFT *)csrDrift0;

  (void)Po;
  (void)accepted;
  (void)zStart;
  (void)revolutionLength;
  (void)charge;
  (void)rootname;

  if (!csrDrift || csrDrift->csr || csrDrift->LSCBins) {
    gpuUnsupported("gpu_track_through_driftCSR");
    return 0;
  }
  if (csrDrift->linearOptics)
    gpu_linearDrift(np, csrDrift->length);
  else
    gpu_exactDrift(np, csrDrift->length);
  return np;
}
void gpu_addCorrectorRadiationKick() { gpuUnsupported("gpu_addCorrectorRadiationKick"); }
long gpu_track_through_matter() { gpuUnsupported("gpu_track_through_matter"); return 0; }

long gpu_scmult_linear_supported(long nParticles, long nBuckets, long nonlinear,
                                  double sliceDuration, long horizontal,
                                  long vertical) {
  if (!gpuScmultAllowed(nParticles))
    return 0;
  if (nBuckets != 1 || nonlinear || sliceDuration > 0)
    return 0;
  if (!horizontal && !vertical)
    return 0;
  return 1;
}

long gpu_scmult_nonlinear_supported(long nParticles, long nBuckets,
                                     long nonlinear, double sliceDuration,
                                     long horizontal, long vertical) {
  if (!gpuScmultAllowed(nParticles))
    return 0;
  if (nBuckets != 1 || !nonlinear || sliceDuration > 0)
    return 0;
  if (!horizontal && !vertical)
    return 0;
  return 1;
}

long gpu_scmult_single_bunch_supported(long nParticles, long idSlotsPerBunch,
                                        long nonlinear, double sliceDuration,
                                        long horizontal, long vertical) {
  (void)idSlotsPerBunch;
  if (nonlinear)
    return gpu_scmult_nonlinear_supported(nParticles, 1, nonlinear,
                                          sliceDuration, horizontal, vertical);
  return gpu_scmult_linear_supported(nParticles, 1, 0, sliceDuration,
                                     horizontal, vertical);
}

long gpu_scmult_can_skip_cpu(long nParticles, long idSlotsPerBunch) {
  (void)idSlotsPerBunch;
  if (!gpuScmultAllowed(nParticles))
    return 0;
  if (!gpuBase.deviceCoord || !gpuBase.deviceCurrent)
    return 0;
  return 1;
}

long gpu_scmult_can_initialize_on_gpu(long nParticles) {
  return gpuScmultAllowed(nParticles);
}

long gpu_scmult_count_bunches(long nParticles, long idSlotsPerBunch,
                              long *nBuckets) {
  GPU_LONG_MIN_MAX_DATA result;
  float milliseconds = 0;
  int status;
  long startedWallTimer = 0;

  if (!nBuckets)
    gpuRequiredFailure("NULL SCMULT bunch-count output pointer in CUDA path");
  *nBuckets = 0;
  if (!gpuScmultAllowed(nParticles))
    return 0;
  if (idSlotsPerBunch <= 0) {
    *nBuckets = 1;
    return 1;
  }
  if (bunchIndex >= totalPropertiesPerParticle)
    return 0;

  if (gpuWallStart <= 0) {
    gpuWallStart = wallSeconds();
    startedWallTimer = 1;
  }
  memset(&result, 0, sizeof(result));
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaLongMinMax(gpuBase.deviceCoord, nParticles,
                             (int)gpuBase.deviceStride, bunchIndex,
                             &result, &milliseconds);
  if (status != 0)
    gpuFatalStatus("SCMULT bunch-index reduction kernel", status);
  gpuRecordReductionKernel(milliseconds);
  if (startedWallTimer)
    gpuRecordWallSeconds();

  if (result.count <= 0)
    return 1;
#if USE_MPI
  if (!partOnMaster) {
    long minGlobal, maxGlobal;
    MPI_Allreduce(&result.min, &minGlobal, 1, MPI_LONG, MPI_MIN, workers);
    MPI_Allreduce(&result.max, &maxGlobal, 1, MPI_LONG, MPI_MAX, workers);
    result.min = minGlobal;
    result.max = maxGlobal;
  }
#endif
  *nBuckets = result.max - result.min + 1;
  return 1;
}

long gpu_scmult_compute_centroid_sigma(long nParticles, double Po,
                                        double *center, double *sigma) {
  GPU_BEAM_SUM_DATA sums;
  double variance;
  float milliseconds = 0;
  int status;
  long i, coordinate;
  long startedWallTimer = 0;

  if (!center || !sigma)
    gpuRequiredFailure("NULL SCMULT moment output pointer in CUDA path");
  for (i = 0; i < 3; i++) {
    center[i] = 0;
    sigma[i] = 0;
  }
  if (!gpuScmultAllowed(nParticles))
    return 0;

  if (gpuWallStart <= 0) {
    gpuWallStart = wallSeconds();
    startedWallTimer = 1;
  }
  memset(&sums, 0, sizeof(sums));
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaBeamSums(gpuBase.deviceCoord, nParticles,
                           (int)gpuBase.deviceStride, Po, c_mks,
                           &sums, &milliseconds);
  if (status != 0)
    gpuFatalStatus("SCMULT centroid/rms reduction kernel", status);
  gpuRecordReductionKernel(milliseconds);
  if (startedWallTimer)
    gpuRecordWallSeconds();

  if (sums.count <= 0)
    return 0;
  for (i = 0; i < 3; i++) {
    coordinate = 2 * i;
    center[i] = sums.centroidSum[coordinate] / sums.count;
    variance = gpuLscVarianceFromSums(&sums, coordinate);
    sigma[i] = sqrt(variance);
  }
  return 1;
}

void gpu_track_through_scmult_linear(long nParticles, double charge, double c1,
                                     long horizontal, long vertical,
                                     long uniformDistribution,
                                     const double *center, const double *sigma,
                                     double dmux, double dmuy, double betax,
                                     double betay) {
  GPU_SCMULT_LINEAR_DATA data;
  float milliseconds = 0;
  int status;

  if (nParticles <= 0)
    return;
  if (!center || !sigma)
    gpuRequiredFailure("NULL SCMULT centroid/sigma pointer in CUDA path");
  if (sigma[2] == 0)
    gpuRequiredFailure("zero longitudinal beam size in CUDA SCMULT path");
  if (horizontal && betax == 0)
    gpuRequiredFailure("zero betax in CUDA SCMULT path");
  if (vertical && betay == 0)
    gpuRequiredFailure("zero betay in CUDA SCMULT path");

  memset(&data, 0, sizeof(data));
  data.horizontal = horizontal ? 1 : 0;
  data.vertical = vertical ? 1 : 0;
  data.uniformDistribution = uniformDistribution ? 1 : 0;
  data.charge = charge;
  data.c1 = c1;
  memcpy(data.center, center, 3 * sizeof(*center));
  memcpy(data.sigma, sigma, 3 * sizeof(*sigma));
  data.dmux = dmux;
  data.dmuy = dmuy;
  data.betax = betax;
  data.betay = betay;

  startGpuTimer();
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaScmultLinearKick(gpuBase.deviceCoord, nParticles,
                                   (int)gpuBase.deviceStride, &data,
                                   &milliseconds);
  if (status != 0)
    gpuFatalStatus("SCMULT linear kick kernel", status);
  gpuRecordScmultKernel(milliseconds);
  gpuBase.elementOnGpu = 1;
  gpuBase.gpuElementCount++;
  gpuMarkDeviceChanged(nParticles);
  gpuRecordWallSeconds();
}

void gpu_track_through_scmult_nonlinear(long nParticles, double charge,
                                        double c1, long horizontal,
                                        long vertical,
                                        long uniformDistribution,
                                        const double *center,
                                        const double *sigma, double dmux,
                                        double dmuy, double betax,
                                        double betay) {
  GPU_SCMULT_LINEAR_DATA data;
  float milliseconds = 0;
  int status;

  if (nParticles <= 0)
    return;
  if (!center || !sigma)
    gpuRequiredFailure("NULL SCMULT centroid/sigma pointer in nonlinear CUDA path");
  if (sigma[0] <= 0 || sigma[1] <= 0 || sigma[2] == 0)
    gpuRequiredFailure("invalid beam size in nonlinear CUDA SCMULT path");
  if (horizontal && betax == 0)
    gpuRequiredFailure("zero betax in nonlinear CUDA SCMULT path");
  if (vertical && betay == 0)
    gpuRequiredFailure("zero betay in nonlinear CUDA SCMULT path");

  memset(&data, 0, sizeof(data));
  data.horizontal = horizontal ? 1 : 0;
  data.vertical = vertical ? 1 : 0;
  data.uniformDistribution = uniformDistribution ? 1 : 0;
  data.charge = charge;
  data.c1 = c1;
  memcpy(data.center, center, 3 * sizeof(*center));
  memcpy(data.sigma, sigma, 3 * sizeof(*sigma));
  data.dmux = dmux;
  data.dmuy = dmuy;
  data.betax = betax;
  data.betay = betay;

  startGpuTimer();
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaScmultNonlinearKick(gpuBase.deviceCoord, nParticles,
                                      (int)gpuBase.deviceStride, &data,
                                      &milliseconds);
  if (status != 0)
    gpuFatalStatus("SCMULT nonlinear kick kernel", status);
  gpuRecordScmultKernel(milliseconds);
  gpuBase.elementOnGpu = 1;
  gpuBase.gpuElementCount++;
  gpuMarkDeviceChanged(nParticles);
  gpuRecordWallSeconds();
}

static void gpuPreparePolynomialSeriesCache(POLYNOMIALSERIES *polynomialSeries,
                                            GPU_POLYNOMIAL_SERIES_DATA *data) {
  long coordinate, term, totalTerms;

  if (!polynomialSeries || !data)
    gpuRequiredFailure("NULL POLYNOMIALSERIES CUDA setup pointer");
  if (!polynomialSeries->elementInitialized) {
    if (!initialize_polynomialSeries)
      gpuRequiredFailure("POLYNOMIALSERIES initialization routine unavailable");
    initialize_polynomialSeries(polynomialSeries);
  }
  memset(data, 0, sizeof(*data));
  totalTerms = 0;
  for (coordinate = 0; coordinate < 6; coordinate++) {
    data->coordinateOffset[coordinate] = totalTerms;
    totalTerms += polynomialSeries->coord[coordinate].terms;
  }
  data->coordinateOffset[6] = totalTerms;
  data->totalTerms = totalTerms;
  if (gpuPolynomialSeriesCache.owner == polynomialSeries &&
      gpuPolynomialSeriesCache.totalTerms == totalTerms)
    return;

  gpuReleasePolynomialSeriesCache();
  gpuPolynomialSeriesCache.coefficient =
    (double *)malloc(totalTerms * sizeof(*gpuPolynomialSeriesCache.coefficient));
  gpuPolynomialSeriesCache.exponent =
    (int32_t *)malloc(6 * totalTerms *
                      sizeof(*gpuPolynomialSeriesCache.exponent));
  if (!gpuPolynomialSeriesCache.coefficient ||
      !gpuPolynomialSeriesCache.exponent)
    gpuRequiredFailure("unable to allocate POLYNOMIALSERIES CUDA map cache");
  totalTerms = 0;
  for (coordinate = 0; coordinate < 6; coordinate++) {
    POLYNOMIALSERIES_DATA *map = &polynomialSeries->coord[coordinate];
    for (term = 0; term < map->terms; term++, totalTerms++) {
      gpuPolynomialSeriesCache.coefficient[totalTerms] =
        map->Coefficient[term];
      gpuPolynomialSeriesCache.exponent[6 * totalTerms + 0] = map->Ix[term];
      gpuPolynomialSeriesCache.exponent[6 * totalTerms + 1] = map->Iqx[term];
      gpuPolynomialSeriesCache.exponent[6 * totalTerms + 2] = map->Iy[term];
      gpuPolynomialSeriesCache.exponent[6 * totalTerms + 3] = map->Iqy[term];
      gpuPolynomialSeriesCache.exponent[6 * totalTerms + 4] = map->Is[term];
      gpuPolynomialSeriesCache.exponent[6 * totalTerms + 5] =
        map->Idelta[term];
    }
  }
  gpuPolynomialSeriesCache.owner = polynomialSeries;
  gpuPolynomialSeriesCache.totalTerms = totalTerms;
}

long gpu_polynomial_series_tracking(long nParticles, void *polynomialSeries0,
                                    double pError, double pCentral,
                                    double **accepted, double zStart) {
  POLYNOMIALSERIES *polynomialSeries =
    (POLYNOMIALSERIES *)polynomialSeries0;
  GPU_POLYNOMIAL_SERIES_DATA data;
  long invalidCount = 0;
  float milliseconds = 0;
  int status;

  if (nParticles <= 0)
    return 0;
  if (!polynomialSeries)
    gpuRequiredFailure("NULL POLYNOMIALSERIES pointer in CUDA path");
  if (!gpuEnablePolynomialSeries || gpuBase.backtrack ||
      !gpuParticleCountMeetsThreshold(nParticles,
                                      gpuBase.magnetMinParticles)) {
    double **coord =
      forceParticlesToCpu("POLYNOMIALSERIES option CPU fallback");
    return polynomialSeries_tracking(
      coord, nParticles, polynomialSeries, pError, pCentral, accepted, zStart);
  }

  gpuPreparePolynomialSeriesCache(polynomialSeries, &data);
  data.tilt = polynomialSeries->tilt;
  data.dx = polynomialSeries->dx;
  data.dy = polynomialSeries->dy;
  data.dz = polynomialSeries->dz;
  data.pCentral = pCentral;
  data.coordinateLimit = coordLimit;
  data.slopeLimit = slopeLimit;

  startGpuTimer();
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaPolynomialSeriesTrack(
    gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride, &data,
    gpuPolynomialSeriesCache.coefficient,
    gpuPolynomialSeriesCache.exponent, polynomialSeries, &invalidCount,
    &milliseconds);
  if (status != 0)
    gpuFatalStatus("POLYNOMIALSERIES tracking CUDA kernel", status);
  gpuRecordMagnetKernel(milliseconds);
  if (invalidCount) {
#ifdef GPU_VERIFY
    gpuRequiredFailure(
      "POLYNOMIALSERIES CUDA verification encountered particle loss");
#else
    double **coord =
      forceParticlesToCpu("POLYNOMIALSERIES particle loss fallback");
    return polynomialSeries_tracking(
      coord, nParticles, polynomialSeries, pError, pCentral, accepted, zStart);
#endif
  }
  gpuMarkDeviceChanged(nParticles);
  gpuRecordWallSeconds();
  return nParticles;
}

void gpu_apply_rfdf(long nParticles, const GPU_RFDF_DATA *data) {
  float milliseconds = 0;
  int status;

  if (nParticles <= 0)
    return;
  if (!data)
    gpuRequiredFailure("NULL RFDF CUDA tracking data");
  startGpuTimer();
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaRfdfTrack(gpuBase.deviceCoord, nParticles,
                            (int)gpuBase.deviceStride, data,
                            particleIDIndex, &milliseconds);
  if (status != 0)
    gpuFatalStatus("RFDF tracking CUDA kernel", status);
  gpuRecordHelperKernel(milliseconds);
  gpuMarkDeviceChanged(nParticles);
  gpuRecordWallSeconds();
}

void gpu_track_sreffects(long nParticles,
                         const GPU_SREFFECTS_DATA *data) {
  float milliseconds = 0;
  int status;

  if (nParticles <= 0)
    return;
  if (!data)
    gpuRequiredFailure("NULL SREFFECTS CUDA tracking data");
  startGpuTimer();
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaSreffectsTrack(gpuBase.deviceCoord, nParticles,
                                 (int)gpuBase.deviceStride, data,
                                 &milliseconds);
  if (status != 0)
    gpuFatalStatus("SREFFECTS tracking CUDA kernel", status);
  gpuRecordHelperKernel(milliseconds);
  gpuMarkDeviceChanged(nParticles);
  gpuRecordWallSeconds();
}

void gpu_track_bggexp(long nParticles, const GPU_BGGEXP_DATA *data) {
  float milliseconds = 0;
  int status;

  if (nParticles <= 0)
    return;
  if (!data)
    gpuRequiredFailure("NULL BGGEXP CUDA tracking data");
  startGpuTimer();
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaBggexpTrack(gpuBase.deviceCoord, nParticles,
                              (int)gpuBase.deviceStride, data,
                              &milliseconds);
  if (status != 0)
    gpuFatalStatus("BGGEXP tracking CUDA kernel", status);
  gpuRecordMagnetKernel(milliseconds);
  gpuMarkDeviceChanged(nParticles);
  gpuRecordWallSeconds();
}

void gpu_track_cwiggler(long nParticles, const GPU_CWIGGLER_DATA *data) {
  float milliseconds = 0;
  int status;

  if (nParticles <= 0)
    return;
  if (!data)
    gpuRequiredFailure("NULL CWIGGLER CUDA tracking data");
  startGpuTimer();
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaCwigglerTrack(gpuBase.deviceCoord, nParticles,
                                (int)gpuBase.deviceStride, data,
                                &milliseconds);
  if (status != 0)
    gpuFatalStatus("CWIGGLER tracking CUDA kernel", status);
  gpuRecordMagnetKernel(milliseconds);
  gpuMarkDeviceChanged(nParticles);
  gpuRecordWallSeconds();
}

void gpu_track_ftable(long nParticles, FTABLE *ftable, double pCentral) {
  GPU_FTABLE_DATA data;
  ntuple *table[3];
  float milliseconds = 0;
  int status;
  long field, dimension;

  if (nParticles <= 0)
    return;
  if (!ftable || !ftable->Bx || !ftable->By || !ftable->Bz)
    gpuRequiredFailure("NULL FTABLE CUDA tracking data");
  table[0] = ftable->Bx;
  table[1] = ftable->By;
  table[2] = ftable->Bz;
  memset(&data, 0, sizeof(data));
  data.tableOwner = ftable->Bx;
  for (field = 0; field < 3; field++)
    data.field[field] = table[field]->value;
  for (dimension = 0; dimension < 3; dimension++) {
    data.dimensions[dimension] = table[0]->xbins[dimension];
    data.minimum[dimension] = table[0]->xmin[dimension];
    data.maximum[dimension] = table[0]->xmax[dimension];
    data.spacing[dimension] = table[0]->dx[dimension];
  }
  data.nKicks = ftable->nKicks;
  data.length = ftable->length;
  data.factor = ftable->factor;
  data.threshold = ftable->threshold;
  data.pCentral = pCentral;
  data.eomc = -particleCharge / particleMass / c_mks;

  startGpuTimer();
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaFtableTrack(gpuBase.deviceCoord, nParticles,
                              (int)gpuBase.deviceStride, &data,
                              &milliseconds);
  if (status != 0)
    gpuFatalStatus("FTABLE tracking CUDA kernel", status);
  gpuRecordMagnetKernel(milliseconds);
  gpuMarkDeviceChanged(nParticles);
  gpuRecordWallSeconds();
}

long gpu_track_bmxyz(long nParticles, BMAPXYZ *bmxyz, double pCentral) {
  GPU_BMXYZ_DATA data;
  BMAPXYZ_DATA *map;
  float milliseconds = 0;
  int status;
  long failedCount = 0;

  if (nParticles <= 0)
    return 1;
  if (!bmxyz || !(map = bmxyz->data) ||
      !map->Fx || !map->Fy || !map->Fz)
    gpuRequiredFailure("NULL BMXYZ CUDA tracking data");
  memset(&data, 0, sizeof(data));
  data.tableOwner = map;
  data.dimensions[0] = map->nx;
  data.dimensions[1] = map->ny;
  data.dimensions[2] = map->nz;
  data.field[0] = map->Fz;
  data.field[1] = map->Fx;
  data.field[2] = map->Fy;
  data.minimum[0] = map->xmin;
  data.minimum[1] = map->ymin;
  data.minimum[2] = map->zmin;
  data.spacing[0] = map->dx;
  data.spacing[1] = map->dy;
  data.spacing[2] = map->dz;
  data.fieldLength = bmxyz->fieldLength;
  data.integrationAccuracy = bmxyz->accuracy;
  data.strengthFactor = bmxyz->strength * (1 + bmxyz->fse);
  data.fieldIsMagnetic = map->BGiven ? 1 : 0;
  data.fieldScale = -particleCharge * particleRelSign /
                    (particleMass * c_mks * pCentral);

  startGpuTimer();
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaBmxyzTrack(gpuBase.deviceCoord, nParticles,
                             (int)gpuBase.deviceStride, &data,
                             &failedCount, &milliseconds);
  if (status != 0)
    gpuFatalStatus("BMXYZ fixed-step tracking CUDA kernel", status);
  gpuRecordMagnetKernel(milliseconds);
  if (!failedCount)
    gpuMarkDeviceChanged(nParticles);
  gpuRecordWallSeconds();
  return failedCount == 0;
}

static double gpuLscVarianceFromSums(const GPU_BEAM_SUM_DATA *sums, long coordinate) {
  double variance;

  if (!sums || sums->count <= 0)
    return 0;
  variance = sums->productSum[gpuUpperTriangularIndex(coordinate, coordinate)] -
             sums->centroidSum[coordinate] * sums->centroidSum[coordinate] / sums->count;
  variance /= sums->count;
  return variance > 0 ? variance : 0;
}

static void gpuLscCenteredVariances(const GPU_BEAM_SUM_DATA *sums,
                                    long nParticles, double pCentral,
                                    double *S11, double *S33) {
  GPU_BEAM_SUM_DATA centeredSums;
  double xCentroid, yCentroid;
  float milliseconds = 0;
  int status;

  (void)pCentral;
  if (!S11 || !S33)
    gpuRequiredFailure("NULL LSCDRIFT variance output pointer");
  *S11 = *S33 = 0;
  if (!sums || sums->count <= 0 || nParticles <= 0)
    return;

  xCentroid = sums->centroidSum[0] / sums->count;
  yCentroid = sums->centroidSum[2] / sums->count;
  memset(&centeredSums, 0, sizeof(centeredSums));
  status = gpuCudaLscTransverseSums(gpuBase.deviceCoord, nParticles,
                                    (int)gpuBase.deviceStride,
                                    xCentroid, yCentroid, &centeredSums,
                                    gpuLscScratch.result, &milliseconds);
  if (status != 0)
    gpuFatalStatus("LSC parallel beam-size reduction kernel", status);
  gpuRecordReductionKernel(milliseconds);
  if (centeredSums.count <= 0)
    return;
  *S11 = centeredSums.productSum[gpuUpperTriangularIndex(0, 0)] /
         centeredSums.count;
  *S33 = centeredSums.productSum[gpuUpperTriangularIndex(2, 2)] /
         centeredSums.count;
  if (*S11 < 0)
    *S11 = 0;
  if (*S33 < 0)
    *S33 = 0;
}

static void gpuLscComputeVoltage(double *vtime, double *ifreq,
                                 const double *itime, long nb,
                                 double dt, double length,
                                 double beamRadius,
                                 double macroParticleCharge,
                                 long backtrack, double Po,
                                 double lowFrequencyCutoff0,
                                 double lowFrequencyCutoff1,
                                 double highFrequencyCutoff0,
                                 double highFrequencyCutoff1) {
  double *vfreq = vtime;
  double Z0 = sqrt(mu_o / epsilon_o);
  double df, dk, factor, a2, ZImag, k, a1;
  long ib, nfreq, iReal, iImag;

  if (!vtime || !ifreq || !itime || nb < 2 || dt <= 0 || beamRadius == 0)
    gpuRequiredFailure("invalid LSCDRIFT voltage calculation inputs");

  memset(ifreq, 0, 2 * (nb + 1) * sizeof(*ifreq));
  memset(vtime, 0, 2 * (nb + 1) * sizeof(*vtime));
  memcpy(ifreq, itime, 2 * nb * sizeof(*ifreq));

  realFFT(ifreq, nb, 0);
  nfreq = nb / 2 + 1;
  if (highFrequencyCutoff0 > 0) {
    long i, i1, i2;
    double dfraction, fraction;

    i1 = highFrequencyCutoff0 * nfreq;
    if (i1 < 1)
      i1 = 1;
    i2 = highFrequencyCutoff1 * nfreq;
    if (i2 >= nfreq)
      i2 = nfreq - 1;
    dfraction = i1 == i2 ? 0 : 1. / (i2 - i1);
    fraction = 1;
    for (i = i1; i < i2; i++) {
      ifreq[2 * i - 1] *= fraction;
      ifreq[2 * i] *= fraction;
      if ((fraction -= dfraction) < 0)
        fraction = 0;
    }
    for (; i < nfreq - 1; i++) {
      ifreq[2 * i - 1] = 0;
      ifreq[2 * i] = 0;
    }
    ifreq[nb - 1] = 0;
  }
  if (lowFrequencyCutoff0 >= 0) {
    long i, i1, i2;
    double dfraction, fraction;

    i1 = lowFrequencyCutoff0 * nfreq;
    if (i1 < 1)
      i1 = 1;
    if (i1 >= nfreq)
      i1 = nfreq - 1;
    i2 = lowFrequencyCutoff1 * nfreq;
    if (i2 < i1)
      i2 = i1;
    if (i2 >= nfreq)
      i2 = nfreq - 1;
    dfraction = i1 == i2 ? 0 : 1. / (i2 - i1);
    fraction = 0;
    ifreq[0] = 0;
    for (i = 1; i < i1; i++) {
      ifreq[2 * i - 1] = 0;
      ifreq[2 * i] = 0;
    }
    for (i = i1; i < i2; i++) {
      ifreq[2 * i - 1] *= fraction;
      ifreq[2 * i] *= fraction;
      if ((fraction += dfraction) > 1)
        fraction = 1;
    }
  }
  df = 1. / (dt * nb);
  dk = df * PIx2 / c_mks;
  factor = macroParticleCharge / dt;
  if (backtrack)
    factor *= -1;
  a2 = Z0 / (PI * sqr(beamRadius)) * length;

  vfreq[0] = 0;
  if (nb % 2 == 0)
    vfreq[nb - 1] = 0;
  for (ib = 1; ib < nfreq - 1; ib++) {
    k = ib * dk;
    a1 = k * beamRadius / Po;
    ZImag = a2 / k * (1 - a1 * dbesk1(a1));
    iImag = (iReal = 2 * ib - 1) + 1;
    vfreq[iReal] = ifreq[iImag] * ZImag * factor;
    vfreq[iImag] = -ifreq[iReal] * ZImag * factor;
  }

  realFFT(vfreq, nb, INVERSE_FFT);
  vtime[nb] = 0;
}

#ifdef GPU_VERIFY
static double gpuLscArrayMaxAbsDiff(const double *a, const double *b,
                                    long n, long *indexReturn,
                                    long *mismatchReturn) {
  double maxAbs = 0;
  long i, index = -1, mismatches = 0;

  if (!a || !b || n <= 0) {
    if (indexReturn)
      *indexReturn = -1;
    if (mismatchReturn)
      *mismatchReturn = 0;
    return 0;
  }
  for (i = 0; i < n; i++) {
    double absDiff = fabs(a[i] - b[i]);
    if (absDiff != 0)
      mismatches++;
    if (absDiff > maxAbs) {
      maxAbs = absDiff;
      index = i;
    }
  }
  if (indexReturn)
    *indexReturn = index;
  if (mismatchReturn)
    *mismatchReturn = mismatches;
  return maxAbs;
}

static void gpuLscCompareCpuShadow(double **cpuCoord, double *gpuCopy,
                                   long np, long stride,
                                   const char *label) {
  double absTol = gpuEnvDouble("ELEGANT_GPU_COMPARE_ABS", 1e-12);
  double relTol = gpuEnvDouble("ELEGANT_GPU_COMPARE_REL", 1e-12);
  double maxAbs = 0, maxRel = 0;
  unsigned long count;
  float milliseconds = 0;
  int status;
  long ip, ic, mismatches = 0;

  if (!cpuCoord || !gpuCopy || np <= 0 || stride <= 0)
    return;
  count = (unsigned long)np * (unsigned long)stride;
  status = gpuCudaCopyDeviceToHost(gpuCopy, gpuBase.deviceCoord, count,
                                   &milliseconds);
  if (status != 0)
    gpuFatalStatus("cudaMemcpy(LSCDRIFT verify device to host)", status);
  gpuRecordMilliseconds(&gpuBase.gpuTransferToHostSeconds, milliseconds);

  for (ip = 0; ip < np; ip++) {
    for (ic = 0; ic < stride; ic++) {
      double cpu = cpuCoord[ip][ic];
      double gpu = gpuCopy[ip * stride + ic];
      double absDiff = fabs(cpu - gpu);
      double scale = fmax(fabs(cpu), fabs(gpu));
      double relDiff = scale > DBL_MIN ? absDiff / scale : absDiff;
      if (absDiff > maxAbs)
        maxAbs = absDiff;
      if (relDiff > maxRel)
        maxRel = relDiff;
      if (!(absDiff <= absTol || relDiff <= relTol)) {
        if (mismatches < 10)
          fprintf(stderr,
                  "elegant CUDA VERIFY mismatch %s particle=%ld coord=%ld cpu=%.17e gpu=%.17e abs=%.3e rel=%.3e\n",
                  label ? label : "track_through_lscdrift", ip, ic, cpu, gpu,
                  absDiff, relDiff);
        mismatches++;
      }
    }
  }

  if (mismatches) {
    fprintf(stderr,
            "elegant CUDA VERIFY failed for %s: %ld mismatches, maxAbs=%.3e, maxRel=%.3e, absTol=%.3e, relTol=%.3e\n",
            label ? label : "track_through_lscdrift", mismatches, maxAbs,
            maxRel, absTol, relTol);
    exit(1);
  }
  if (gpuVerbose)
    fprintf(stderr,
            "elegant CUDA VERIFY passed for %s: maxAbs=%.3e maxRel=%.3e\n",
            label ? label : "track_through_lscdrift", maxAbs, maxRel);
}

static void gpuLscAdvanceCpuShadow(double **part, double *time, long *pbin,
                                   long np, double Po, double *vtime,
                                   long nb, double tmin, double dt,
                                   long interpolate, long doDrift,
                                   double length, long backtrack) {
  long ip;
  double sign = backtrack ? -1 : 1;

  if (!applyLongitudinalWakeKicks)
    gpuRequiredFailure("LSCDRIFT verification needs applyLongitudinalWakeKicks");
  applyLongitudinalWakeKicks(part, time, pbin, np, Po, vtime,
                             nb, tmin, dt, interpolate);
  if (!doDrift)
    return;
  for (ip = 0; ip < np; ip++) {
    part[ip][4] += length * sqrt(1 + sqr(part[ip][1]) + sqr(part[ip][3])) * sign;
    part[ip][0] += length * part[ip][1] * sign;
    part[ip][2] += length * part[ip][3] * sign;
  }
}
#endif

void gpu_track_through_lscdrift(long np0, void *lsc0, double Po, void *charge0) {
  LSCDRIFT *lsc = (LSCDRIFT *)lsc0;
  CHARGE *charge = (CHARGE *)charge0;
  static double *Itime = NULL;
  static double *Ifreq = NULL;
  static double *Vtime = NULL;
  static long maxBins = 0;
  GPU_BEAM_SUM_DATA sums;
  GPU_LSC_DATA data;
  double lengthLeft, length, Imin, Imax, kSC, S11, S33, beamRadius;
  long nb, binnedCount = 0;
  short kickMode = 0;
  float milliseconds = 0;
  int status;
  char warningBuffer[1024];
#ifdef GPU_VERIFY
  double *cpuStorage = NULL, **cpuCoord = NULL;
  double *gpuVerifyStorage = NULL;
  double *cpuTime = NULL, *cpuItime = NULL, *cpuIfreq = NULL, *cpuVtime = NULL;
  long *cpuPbin = NULL;
  long stride, ip, cpuBinned;
  unsigned long count;
#endif

  if (np0 <= 0)
    return;
  if (!lsc)
    gpuRequiredFailure("NULL LSCDRIFT pointer in CUDA path");
  if (gpuBase.backtrack || lsc->backtrack) {
    double **coord = forceParticlesToCpu("LSCDRIFT backtracking CPU reference");

    if (!track_through_lscdrift)
      gpuRequiredFailure("CPU LSCDRIFT routine is unavailable");
    gpuBase.elementOnGpu = 0;
    track_through_lscdrift(coord, np0, lsc, Po, charge);
    gpuMarkHostWillChange();
    gpuRecordWallSeconds();
    return;
  }
  if (!gpuLscDataSupported(lsc))
    gpuRequiredFailure("unsupported LSCDRIFT options reached CUDA path");
  if (!charge)
    bombElegant("No charge defined for LSC.  Insert a CHARGE element in the beamline.", NULL);
  gpuBase.gpuStandaloneLscCount++;

  nb = lsc->bins;
  gpuEnsureLscScratch(nb);
  if (nb > maxBins) {
    maxBins = nb;
    Itime = trealloc(Itime, 2 * sizeof(*Itime) * (maxBins + 1));
    Ifreq = trealloc(Ifreq, 2 * sizeof(*Ifreq) * (maxBins + 1));
    Vtime = trealloc(Vtime, 2 * sizeof(*Vtime) * (maxBins + 1));
  }

  lengthLeft = fabs(lsc->length);
  if (lengthLeft == 0) {
    lengthLeft = fabs(lsc->lEffective);
    kickMode = 1;
  }
  gpuCopyHostToDevice(np0);

#ifdef GPU_VERIFY
  stride = gpuBase.deviceStride;
  count = (unsigned long)np0 * (unsigned long)stride;
  cpuStorage = (double *)malloc(count * sizeof(*cpuStorage));
  cpuCoord = (double **)malloc(np0 * sizeof(*cpuCoord));
  cpuTime = (double *)malloc(np0 * sizeof(*cpuTime));
  cpuPbin = (long *)malloc(np0 * sizeof(*cpuPbin));
  cpuItime = (double *)malloc(2 * (nb + 1) * sizeof(*cpuItime));
  cpuIfreq = (double *)malloc(2 * (nb + 1) * sizeof(*cpuIfreq));
  cpuVtime = (double *)malloc(2 * (nb + 1) * sizeof(*cpuVtime));
  gpuVerifyStorage = (double *)malloc(count * sizeof(*gpuVerifyStorage));
  if (!cpuStorage || !cpuCoord || !cpuTime || !cpuPbin ||
      !cpuItime || !cpuIfreq || !cpuVtime || !gpuVerifyStorage)
    gpuRequiredFailure("unable to allocate CUDA LSCDRIFT verification buffers");
  memcpy(cpuStorage, gpuBase.coord[0], count * sizeof(*cpuStorage));
  for (ip = 0; ip < np0; ip++)
    cpuCoord[ip] = cpuStorage + ip * stride;
#endif

  while (lengthLeft > 0) {
    memset(&sums, 0, sizeof(sums));
    milliseconds = 0;
    status = gpuCudaLscStatistics(gpuBase.deviceCoord, np0,
                                  (int)gpuBase.deviceStride, Po, c_mks,
                                  &sums, gpuLscScratch.result,
                                  &milliseconds);
    if (status != 0)
      gpuFatalStatus("LSCDRIFT time-coordinate reduction kernel", status);
    gpuRecordReductionKernel(milliseconds);

    data.bins = nb;
    data.interpolate = lsc->interpolate ? 1 : 0;
    data.doDrift = kickMode ? 0 : 1;
    data.backtrack = lsc->backtrack ? 1 : 0;
    data.tmin = sums.min[6];
    data.dt = (sums.max[6] - sums.min[6]) / (nb - 3);
    data.pCentral = Po;
    data.particleMassMV = particleMassMV;
    data.particleRelSign = particleRelSign;
    data.cMks = c_mks;
    if (data.dt <= 0)
      gpuRequiredFailure("non-positive LSCDRIFT time-bin spacing in CUDA path");

    memset(Itime, 0, 2 * sizeof(*Itime) * (nb + 1));
    milliseconds = 0;
    status = gpuCudaLscBin(gpuBase.deviceCoord, np0,
                           (int)gpuBase.deviceStride, &data,
                           &binnedCount, Itime, gpuLscScratch.itime,
                           gpuLscScratch.binnedCount, &milliseconds);
    if (status != 0)
      gpuFatalStatus("LSCDRIFT CUDA binning kernel", status);
    gpuRecordLscKernel(milliseconds);

    if (binnedCount != np0) {
      snprintf(warningBuffer, sizeof(warningBuffer),
               "Only %ld of %ld particles were binned. This shouldn't happen.",
               binnedCount, np0);
      gpuWakeTrackingWarning("Some particles were not binned in LSCDRIFT.",
                             warningBuffer);
    }
    if (lsc->smoothing)
      gpuSmoothWakeHistogram(Itime, nb, lsc->SGOrder, lsc->SGHalfWidth,
                             "LSCDRIFT", NULL);

    find_min_max(&Imin, &Imax, Itime, nb);
    Imax *= charge->macroParticleCharge / data.dt;
    gpuLscCenteredVariances(&sums, np0, Po, &S11, &S33);
    beamRadius = (sqrt(S11) + sqrt(S33)) / 2 * lsc->radiusFactor;
    if (beamRadius == 0) {
      fprintf(stderr, "Error: beam radius is zero in CUDA LSCDRIFT: S11=%le, S33=%le, RADIUS_FACTOR=%le\n",
              S11, S33, lsc->radiusFactor);
      exit(1);
    }

    if (Imax <= 0) {
      length = lengthLeft;
    } else {
      kSC = 2 / beamRadius * sqrt(Imax / ipow3(Po) / 17045.0);
      length = 0.1 / kSC;
      if (length > lengthLeft || kickMode)
        length = lengthLeft;
    }
    if (length <= 0 || !isfinite(length))
      length = lengthLeft;

    gpuLscComputeVoltage(Vtime, Ifreq, Itime, nb, data.dt, length,
                         beamRadius, charge->macroParticleCharge,
                         lsc->backtrack, Po,
                         lsc->lowFrequencyCutoff0,
                         lsc->lowFrequencyCutoff1,
                         lsc->highFrequencyCutoff0,
                         lsc->highFrequencyCutoff1);

#ifdef GPU_VERIFY
    memset(cpuItime, 0, 2 * sizeof(*cpuItime) * (nb + 1));
    computeTimeCoordinatesOnly(cpuTime, Po, cpuCoord, np0);
    cpuBinned = binTimeDistribution(cpuItime, cpuPbin, data.tmin, data.dt,
                                    nb, cpuTime, cpuCoord, Po, np0);
    if (cpuBinned != binnedCount) {
      fprintf(stderr,
              "elegant CUDA VERIFY LSCDRIFT bin-count mismatch track_through_lscdrift cpu=%ld gpu=%ld\n",
              cpuBinned, binnedCount);
      exit(1);
    }
    if (lsc->smoothing)
      gpuSmoothWakeHistogram(cpuItime, nb, lsc->SGOrder, lsc->SGHalfWidth,
                             "LSCDRIFT", NULL);
    gpuLscComputeVoltage(cpuVtime, cpuIfreq, cpuItime, nb, data.dt, length,
                         beamRadius, charge->macroParticleCharge,
                         lsc->backtrack, Po,
                         lsc->lowFrequencyCutoff0,
                         lsc->lowFrequencyCutoff1,
                         lsc->highFrequencyCutoff0,
                         lsc->highFrequencyCutoff1);
    gpuCompareWakeArray("track_through_lscdrift", "Itime",
                        cpuItime, Itime, nb);
    gpuCompareWakeArray("track_through_lscdrift", "Vtime",
                        cpuVtime, Vtime, nb);
    gpuLscAdvanceCpuShadow(cpuCoord, cpuTime, cpuPbin, np0, Po, cpuVtime,
                           nb, data.tmin, data.dt, lsc->interpolate,
                           !kickMode, length, lsc->backtrack);
#endif

    data.length = length;
    milliseconds = 0;
    status = gpuCudaLscApplyKickAndDrift(gpuBase.deviceCoord, np0,
                                         (int)gpuBase.deviceStride,
                                         &data, Vtime, gpuLscScratch.vtime,
                                         &milliseconds);
    if (status != 0)
      gpuFatalStatus("LSCDRIFT CUDA kick/drift kernel", status);
    gpuRecordLscKernel(milliseconds);
#ifdef GPU_VERIFY
    gpuLscCompareCpuShadow(cpuCoord, gpuVerifyStorage, np0, stride,
                           "track_through_lscdrift");
#endif
    gpuMarkDeviceChanged(np0);
    lengthLeft -= length;
  }

#ifdef GPU_VERIFY
  free(cpuStorage);
  free(cpuCoord);
  free(gpuVerifyStorage);
  free(cpuTime);
  free(cpuItime);
  free(cpuIfreq);
  free(cpuVtime);
  free(cpuPbin);
#endif
  gpuRecordWallSeconds();
}

static long gpuResolveMultipoleSlices(long nSlices, long nKicks,
                                      short integrationOrder, long *resolvedSlices) {
  if (!resolvedSlices)
    return 0;
  if (nKicks <= 0) {
    if (nSlices <= 0)
      return 0;
    *resolvedSlices = nSlices;
    return 1;
  }
  if (integrationOrder > 2) {
    nSlices = ceil(nKicks / (1.0 * integrationOrder));
    if (nSlices < 1)
      nSlices = 1;
  } else {
    nSlices = nKicks;
  }
  *resolvedSlices = nSlices;
  return nSlices > 0;
}

static void gpuCsbendTiltSinCos(double ttilt, double *cosTilt, double *sinTilt);

static void gpuInitMultipoleData(GPU_MULTIPOLE_DATA *data, double Po,
                                 double drift, long nSlices,
                                 short integrationOrder,
                                 short expandHamiltonian) {
  memset(data, 0, sizeof(*data));
  data->Po = Po;
  data->drift = drift;
  data->nSlices = nSlices;
  data->integrationOrder = integrationOrder;
  data->expandHamiltonian = expandHamiltonian ? 1 : 0;
  data->initialSlopeRoundTrip = 1;
  data->radiationBlock = 1;
  data->coordLimit = coordLimit;
  data->slopeLimit = slopeLimit;
  data->cosTilt = 1;
}

static long gpuPackMultipoleTracking(GPU_MULTIPOLE_DATA *data,
                                     ELEMENT_LIST *elem, double Po) {
  long nSlices = 0;

  if (!data || !elem || !elem->p_elem || !gpuMultipoleElementSupported(elem))
    return 0;

  switch (elem->type) {
  case T_MULT: {
    MULT *multipole = (MULT *)elem->p_elem;

    if (!gpuResolveMultipoleSlices(multipole->nSlices, 0, 2, &nSlices))
      return 0;
    gpuInitMultipoleData(data, Po, multipole->length, nSlices, 2,
                         multipole->expandHamiltonian);
    data->initialSlopeRoundTrip = 0;
    data->order[0] = multipole->order;
    if (multipole->bore)
      data->KnL[0] = dfactorial(multipole->order) * multipole->BTipL /
                     ipow(multipole->bore, multipole->order) *
                     (particleCharge / (particleMass * c_mks * Po)) *
                     multipole->factor;
    else
      data->KnL[0] = multipole->KnL * multipole->factor;
    data->dx = multipole->dx;
    data->dy = multipole->dy;
    data->dz = multipole->dz;
    gpuCsbendTiltSinCos(multipole->tilt, &data->cosTilt, &data->sinTilt);
    return 1;
  }
  case T_KQUAD: {
    KQUAD *kquad = (KQUAD *)elem->p_elem;
    double lEffective = kquad->lEffective > 0 ? kquad->lEffective : kquad->length;

    if (!gpuResolveMultipoleSlices(kquad->nSlices, kquad->n_kicks,
                                   kquad->integration_order, &nSlices))
      return 0;
    gpuInitMultipoleData(data, Po, lEffective, nSlices,
                         kquad->integration_order, kquad->expandHamiltonian);
    data->order[0] = 1;
    if (kquad->bore)
      data->KnL[0] = kquad->B / kquad->bore *
                     (particleCharge / (particleMass * c_mks * Po)) *
                     lEffective * (1 + kquad->fse);
    else
      data->KnL[0] = kquad->k1 * lEffective * (1 + kquad->fse);
    data->xkick = kquad->xkick * kquad->xKickCalibration;
    data->ykick = kquad->ykick * kquad->yKickCalibration;
    data->dx = kquad->dx;
    data->dy = kquad->dy;
    data->dz = kquad->dz;
    if (kquad->synch_rad)
      data->radCoef =
        sqr(particleCharge) * pow3(Po) /
        (6 * PI * epsilon_o * sqr(c_mks) * particleMass);
    gpuCsbendTiltSinCos(kquad->tilt, &data->cosTilt, &data->sinTilt);
    return 1;
  }
  case T_KSEXT: {
    KSEXT *ksext = (KSEXT *)elem->p_elem;

    if (!gpuResolveMultipoleSlices(ksext->nSlices, ksext->n_kicks,
                                   ksext->integration_order, &nSlices))
      return 0;
    gpuInitMultipoleData(data, Po, ksext->length, nSlices,
                         ksext->integration_order, ksext->expandHamiltonian);
    data->order[0] = 2;
    if (ksext->bore)
      data->KnL[0] = 2 * ksext->B / sqr(ksext->bore) *
                     (particleCharge / (particleMass * c_mks * Po)) *
                     ksext->length * (1 + ksext->fse);
    else
      data->KnL[0] = ksext->k2 * ksext->length * (1 + ksext->fse);
    if (ksext->k1) {
      data->order[1] = 1;
      data->KnL[1] = ksext->k1 * ksext->length;
    }
    if (ksext->j1) {
      data->order[2] = 1;
      data->KnL[2] = ksext->j1 * ksext->length;
      data->skew[2] = 1;
    }
    data->xkick = ksext->xkick * ksext->xKickCalibration;
    data->ykick = ksext->ykick * ksext->yKickCalibration;
    data->dx = ksext->dx;
    data->dy = ksext->dy;
    data->dz = ksext->dz;
    gpuCsbendTiltSinCos(ksext->tilt, &data->cosTilt, &data->sinTilt);
    return 1;
  }
  case T_KOCT: {
    KOCT *koct = (KOCT *)elem->p_elem;

    if (!gpuResolveMultipoleSlices(koct->nSlices, koct->n_kicks,
                                   koct->integration_order, &nSlices))
      return 0;
    gpuInitMultipoleData(data, Po, koct->length, nSlices,
                         koct->integration_order, koct->expandHamiltonian);
    data->order[0] = 3;
    if (koct->bore)
      data->KnL[0] = 6 * koct->B / ipow3(koct->bore) *
                     (particleCharge / (particleMass * c_mks * Po)) *
                     koct->length * (1 + koct->fse);
    else
      data->KnL[0] = koct->k3 * koct->length * (1 + koct->fse);
    data->dx = koct->dx;
    data->dy = koct->dy;
    data->dz = koct->dz;
    gpuCsbendTiltSinCos(koct->tilt, &data->cosTilt, &data->sinTilt);
    return 1;
  }
  case T_DQCOR: {
    DQCOR *dqcor = (DQCOR *)elem->p_elem;
    double K0 = 0, J0 = 0;

    if (!determineDQCorReferenceFrameTilt || !computeQuadSteeringStrengths || !rotate_xy)
      return 0;
    if (!gpuResolveMultipoleSlices(dqcor->nSlices, 0,
                                   dqcor->integration_order, &nSlices))
      return 0;
    gpuInitMultipoleData(data, Po, dqcor->length, nSlices,
                         dqcor->integration_order, 0);
    data->order[0] = 1;
    data->KnL[0] = dqcor->k1 * dqcor->length * (1 + dqcor->fse);
    data->order[1] = 1;
    data->KnL[1] = dqcor->j1 * dqcor->length * (1 + dqcor->fse);
    data->skew[1] = 1;
    determineDQCorReferenceFrameTilt(dqcor);
    computeQuadSteeringStrengths(&K0, &J0,
                                 dqcor->xKickReferenceFrame * dqcor->xKickCalibration,
                                 dqcor->yKickReferenceFrame * dqcor->yKickCalibration,
                                 dqcor->length, dqcor->K1ReferenceFrame);
    data->xkick = -K0 * dqcor->length;
    data->ykick = -J0 * dqcor->length;
    rotate_xy(&data->xkick, &data->ykick, -dqcor->phiReferenceFrame);
    data->dx = dqcor->dx;
    data->dy = dqcor->dy;
    data->dz = dqcor->dz;
    gpuCsbendTiltSinCos(dqcor->tilt, &data->cosTilt, &data->sinTilt);
    return 1;
  }
  default:
    break;
  }

  return 0;
}

static long gpuMultipoleOnCpu(long n_part, ELEMENT_LIST *elem, double p_error,
                              double Po, double **accepted, double z_start,
                              MAXAMP *maxamp, APCONTOUR *apcontour,
                              APERTURE_DATA *apFileData, double *sigmaDelta2,
                              long iSlice, const char *reason) {
  double **coord = forceParticlesToCpu(reason);
  long remaining;

  if (!multipole_tracking2)
    gpuRequiredFailure("CPU multipole_tracking2 fallback is unavailable");
  gpuBase.elementOnGpu = 0;
  remaining = multipole_tracking2(coord, n_part, elem, p_error, Po, accepted,
                                  z_start, maxamp, apcontour, apFileData,
                                  sigmaDelta2, iSlice);
  gpuMarkHostWillChange();
  gpuRecordWallSeconds();
  return remaining;
}

long gpu_multipole_tracking2(long n_part, void *elem0, double p_error,
                             double Po, double **accepted, double z_start,
                             void *maxamp0, void *apcontour0, void *apFileData0,
                             double *sigmaDelta2, long iSlice) {
  ELEMENT_LIST *elem = (ELEMENT_LIST *)elem0;
  MAXAMP *maxamp = (MAXAMP *)maxamp0;
  APCONTOUR *apcontour = (APCONTOUR *)apcontour0;
  APERTURE_DATA *apFileData = (APERTURE_DATA *)apFileData0;
  GPU_MULTIPOLE_DATA data;
  long lostCount = 0;
  float milliseconds = 0;
  int status;

  if (n_part <= 0)
    return n_part;
  if (!elem)
    gpuRequiredFailure("NULL element pointer in gpu_multipole_tracking2");
  if (gpuBase.backtrack)
    return gpuMultipoleOnCpu(n_part, elem, p_error, Po, accepted, z_start,
                             maxamp, apcontour, apFileData, sigmaDelta2, iSlice,
                             "multipole_tracking2 backtracking CPU reference");
  if (iSlice >= 0)
    return gpuMultipoleOnCpu(n_part, elem, p_error, Po, accepted, z_start,
                             maxamp, apcontour, apFileData, sigmaDelta2, iSlice,
                             "multipole_tracking2 slice-by-slice fallback");
  if (maxamp || apcontour || (apFileData && apFileData->initialized))
    return gpuMultipoleOnCpu(n_part, elem, p_error, Po, accepted, z_start,
                             maxamp, apcontour, apFileData, sigmaDelta2, iSlice,
                             "multipole_tracking2 aperture fallback");
  if (sigmaDelta2)
    return gpuMultipoleOnCpu(n_part, elem, p_error, Po, accepted, z_start,
                             maxamp, apcontour, apFileData, sigmaDelta2, iSlice,
                             "multipole_tracking2 radiation sigma fallback");
  if (!gpuPackMultipoleTracking(&data, elem, Po))
    return gpuMultipoleOnCpu(n_part, elem, p_error, Po, accepted, z_start,
                             maxamp, apcontour, apFileData, sigmaDelta2, iSlice,
                             "multipole_tracking2 unsupported option");

  gpuCopyHostToDevice(n_part);
  status = gpuCudaMultipoleTrackChecked(gpuBase.deviceCoord, n_part,
                                        (int)gpuBase.deviceStride, &data,
                                        &lostCount, &milliseconds);
  if (status != 0)
    gpuFatalStatus("multipole tracking CUDA checked kernel", status);
  gpuRecordMagnetKernel(milliseconds);
  if (lostCount) {
#ifndef GPU_VERIFY
    if (gpuMagnetLossCompactionAllowed()) {
      long remaining = gpuMultipoleStableCompact(
        &data, n_part, accepted,
        "multipole_tracking2 stable magnet loss compaction");
      if (&multipoleKicksDone)
        multipoleKicksDone += n_part * data.nSlices * data.integrationOrder;
      return remaining;
    }
#endif
    return gpuMultipoleOnCpu(n_part, elem, p_error, Po, accepted, z_start,
                             maxamp, apcontour, apFileData, sigmaDelta2, iSlice,
                             "multipole_tracking2 particle loss fallback");
  }
  gpuMarkDeviceChanged(n_part);
  gpuRecordWallSeconds();
  if (&multipoleKicksDone)
    multipoleKicksDone += n_part * data.nSlices * data.integrationOrder;
  return n_part;
}

static long gpuMultOnCpu(long n_part, MULT *multipole, double p_error,
                         double Po, double **accepted, double z_start,
                         const char *reason) {
  double **coord = forceParticlesToCpu(reason);
  long remaining;

  if (!multipole_tracking)
    gpuRequiredFailure("CPU multipole_tracking fallback is unavailable");
  gpuBase.elementOnGpu = 0;
  remaining = multipole_tracking(coord, n_part, multipole, p_error, Po,
                                 accepted, z_start);
  gpuMarkHostWillChange();
  gpuRecordWallSeconds();
  return remaining;
}

long gpu_multipole_tracking(long n_part, void *multipole0, double p_error,
                            double Po, double **accepted, double z_start) {
  MULT *multipole = (MULT *)multipole0;
  ELEMENT_LIST elem;
  GPU_MULTIPOLE_DATA data;
  long lostCount = 0;
  float milliseconds = 0;
  int status;

  if (n_part <= 0)
    return n_part;
  if (!multipole)
    gpuRequiredFailure("NULL MULT pointer in gpu_multipole_tracking");
  if (gpuBase.backtrack)
    return gpuMultOnCpu(n_part, multipole, p_error, Po, accepted, z_start,
                        "multipole_tracking backtracking CPU reference");

  memset(&elem, 0, sizeof(elem));
  elem.type = T_MULT;
  elem.p_elem = (char *)multipole;
  if (!gpuPackMultipoleTracking(&data, &elem, Po))
    return gpuMultOnCpu(n_part, multipole, p_error, Po, accepted, z_start,
                        "multipole_tracking unsupported option");
  if (data.KnL[0] == 0) {
    if (data.drift == 0)
      return n_part;
    if (gpuExactDriftParticleCountAllowed(n_part)) {
      gpu_exactDrift(n_part, data.drift);
      return n_part;
    }
    return gpuMultOnCpu(n_part, multipole, p_error, Po, accepted, z_start,
                        "zero-strength MULT exactDrift CPU reference");
  }

  gpuCopyHostToDevice(n_part);
  status = gpuCudaMultipoleTrackChecked(gpuBase.deviceCoord, n_part,
                                        (int)gpuBase.deviceStride, &data,
                                        &lostCount, &milliseconds);
  if (status != 0)
    gpuFatalStatus("MULT tracking CUDA checked kernel", status);
  gpuRecordMagnetKernel(milliseconds);
  if (lostCount) {
#ifndef GPU_VERIFY
    if (gpuMagnetLossCompactionAllowed()) {
      long remaining = gpuMultipoleStableCompact(
        &data, n_part, accepted,
        "multipole_tracking stable magnet loss compaction");
      if (&multipoleKicksDone && data.KnL[0])
        multipoleKicksDone += n_part * data.nSlices * 4;
      return remaining;
    }
#endif
    return gpuMultOnCpu(n_part, multipole, p_error, Po, accepted, z_start,
                        "multipole_tracking particle loss fallback");
  }
  gpuMarkDeviceChanged(n_part);
  gpuRecordWallSeconds();
  if (&multipoleKicksDone && data.KnL[0])
    multipoleKicksDone += n_part * data.nSlices * 4;
  return n_part;
}

static long gpuPackCcbendTracking(GPU_CCBEND_DATA *data, CCBEND *ccbend) {
  double rho0, fse, denominator;
  long i;

  if (!data || !gpuCcbendCommonSupported(ccbend))
    return 0;
  memset(data, 0, sizeof(*data));
  rho0 = ccbend->length / ccbend->angle;
  data->chordLength = 2 * rho0 * sin(ccbend->angle / 2);
  denominator = 1 - ccbend->KnDelta;
  fse = ccbend->fse + ccbend->fseOffset;
  data->KnL[0] =
    (1 + fse + ccbend->fseDipole) / rho0 * data->chordLength -
    ccbend->xKick;
  data->KnL[1] =
    (1 + fse + ccbend->fseQuadrupole) * ccbend->K1 *
    data->chordLength / denominator;
  data->KnL[2] =
    (1 + fse) * ccbend->K2 * data->chordLength / denominator;
  data->nSlices = ccbend->nSlices;
  data->referenceCorrection = ccbend->referenceCorrection;
  data->angleHalf = ccbend->angle / 2;
  data->dxOffset = ccbend->dxOffset;
  data->xAdjust = ccbend->xAdjust;
  data->coordLimit = coordLimit;
  data->slopeLimit = slopeLimit;
  memcpy(data->referenceTrajectory, ccbend->referenceTrajectory,
         sizeof(data->referenceTrajectory));
  if (!isfinite(data->chordLength) || data->chordLength <= 0 ||
      !isfinite(data->angleHalf) || !isfinite(data->dxOffset) ||
      !isfinite(data->xAdjust) || !isfinite(data->coordLimit) ||
      !isfinite(data->slopeLimit))
    return 0;
  for (i = 0; i < 3; i++)
    if (!isfinite(data->KnL[i]))
      return 0;
  for (i = 0; i < 5; i++)
    if (!isfinite(data->referenceTrajectory[i]))
      return 0;
  return 1;
}

static long gpuCcbendOnCpu(long n_part, ELEMENT_LIST *eptr, CCBEND *ccbend,
                           double Po, double **accepted, double z_start,
                           double *sigmaDelta2, char *rootname,
                           MAXAMP *maxamp, APCONTOUR *apContour,
                           APERTURE_DATA *apFileData, long iPart,
                           long iFinalSlice, const char *reason) {
  double **coord = forceParticlesToCpu(reason);
  long remaining;

  if (!track_through_ccbend)
    gpuRequiredFailure("CPU track_through_ccbend fallback is unavailable");
  gpuBase.elementOnGpu = 0;
  remaining = track_through_ccbend(coord, n_part, eptr, ccbend, Po, accepted,
                                   z_start, sigmaDelta2, rootname, maxamp,
                                   apContour, apFileData, iPart, iFinalSlice);
  gpuMarkHostWillChange();
  gpuRecordWallSeconds();
  return remaining;
}

long gpu_track_through_ccbend(long n_part, void *eptr0, void *ccbend0,
                              double Po, double **accepted, double z_start,
                              double *sigmaDelta2, char *rootname,
                              void *maxamp0, void *apContour0,
                              void *apFileData0, long iPart,
                              long iFinalSlice) {
  ELEMENT_LIST *eptr = (ELEMENT_LIST *)eptr0;
  CCBEND *ccbend = (CCBEND *)ccbend0;
  MAXAMP *maxamp = (MAXAMP *)maxamp0;
  APCONTOUR *apContour = (APCONTOUR *)apContour0;
  APERTURE_DATA *apFileData = (APERTURE_DATA *)apFileData0;
  GPU_CCBEND_DATA data;
  long lostCount = 0;
  float milliseconds = 0;
  int status;

  if (n_part <= 0)
    return n_part;
  if (!ccbend)
    gpuRequiredFailure("NULL CCBEND pointer in gpu_track_through_ccbend");
  if (gpuBase.backtrack || iPart >= 0 || iFinalSlice > 0)
    return gpuCcbendOnCpu(n_part, eptr, ccbend, Po, accepted, z_start,
                          sigmaDelta2, rootname, maxamp, apContour,
                          apFileData, iPart, iFinalSlice,
                          "CCBEND partial/backtracking CPU fallback");
  if (maxamp || apContour || (apFileData && apFileData->initialized))
    return gpuCcbendOnCpu(n_part, eptr, ccbend, Po, accepted, z_start,
                          sigmaDelta2, rootname, maxamp, apContour,
                          apFileData, iPart, iFinalSlice,
                          "CCBEND aperture CPU fallback");
  if (sigmaDelta2)
    return gpuCcbendOnCpu(n_part, eptr, ccbend, Po, accepted, z_start,
                          sigmaDelta2, rootname, maxamp, apContour,
                          apFileData, iPart, iFinalSlice,
                          "CCBEND radiation-sigma CPU fallback");
  if (!gpuPackCcbendTracking(&data, ccbend))
    return gpuCcbendOnCpu(n_part, eptr, ccbend, Po, accepted, z_start,
                          sigmaDelta2, rootname, maxamp, apContour,
                          apFileData, iPart, iFinalSlice,
                          "CCBEND unsupported-option CPU fallback");

  gpuCopyHostToDevice(n_part);
  status = gpuCudaCcbendTrackChecked(gpuBase.deviceCoord, n_part,
                                     (int)gpuBase.deviceStride, &data,
                                     &lostCount, &milliseconds);
  if (status != 0)
    gpuFatalStatus("CCBEND tracking CUDA checked kernel", status);
  gpuRecordMagnetKernel(milliseconds);
  if (lostCount)
    return gpuCcbendOnCpu(n_part, eptr, ccbend, Po, accepted, z_start,
                          sigmaDelta2, rootname, maxamp, apContour,
                          apFileData, iPart, iFinalSlice,
                          "CCBEND particle-loss CPU fallback");
  gpuMarkDeviceChanged(n_part);
  gpuRecordWallSeconds();
  if (&multipoleKicksDone)
    multipoleKicksDone += n_part * data.nSlices;
  return n_part;
}

static long gpuCsbendEdgesSupported(CSBEND *csbend) {
  long flags;

  if (!csbend)
    return 0;
  flags = csbend->edgeFlags;
  if (!(flags & BEND_EDGE_DETERMINED))
    return 0;
  if (!(flags & BEND_EDGE_EFFECTS))
    return 1;
  if (flags & BEND_EDGE1_EFFECTS) {
    long index = csbend->e1Index;

    if (csbend->edge_effects[index] != 1)
      return 0;
  }
  if (flags & BEND_EDGE2_EFFECTS) {
    long index = csbend->e2Index;

    if (csbend->edge_effects[index] != 1)
      return 0;
  }
  return 1;
}

static void gpuCsbendTiltSinCos(double ttilt, double *cosTilt, double *sinTilt) {
  if (ttilt == 0) {
    *cosTilt = 1;
    *sinTilt = 0;
  } else if (fabs(fabs(ttilt) - PI) < 1e-12) {
    *cosTilt = -1;
    *sinTilt = 0;
  } else if (fabs(ttilt - PIo2) < 1e-12) {
    *cosTilt = 0;
    *sinTilt = 1;
  } else if (fabs(ttilt + PIo2) < 1e-12) {
    *cosTilt = 0;
    *sinTilt = -1;
  } else {
    *cosTilt = cos(ttilt);
    *sinTilt = sin(ttilt);
  }
}

static void gpuCsbendRotateCoordinatesForMisalignment(double *coord, double angle) {
  double x, xp, y, yp;
  double sin_a, cos_a;

  if (!angle || fabs(fabs(angle) - PIx2) < 1e-12)
    return;
  if (fabs(fabs(angle) - PI) < 1e-12) {
    cos_a = -1;
    sin_a = 0;
  } else if (fabs(angle - PIo2) < 1e-12) {
    cos_a = 0;
    sin_a = 1;
  } else if (fabs(angle + PIo2) < 1e-12) {
    cos_a = 0;
    sin_a = -1;
  } else {
    cos_a = cos(angle);
    sin_a = sin(angle);
  }

  x = coord[0];
  xp = coord[1];
  y = coord[2];
  yp = coord[3];
  coord[0] = x * cos_a + y * sin_a;
  coord[2] = -x * sin_a + y * cos_a;
  coord[1] = xp * cos_a + yp * sin_a;
  coord[3] = -xp * sin_a + yp * cos_a;
}

static void gpuComputeEtiltCentroidOffset(double *dcoordEtilt, double rho0,
                                          double angle, double etilt,
                                          double tilt) {
  double q1a, q2a, q3a;
  double q1b, q2b, q3b;
  double qp1, qp2, qp3;
  double dz, tan_alpha, k;
  long i;

  for (i = 0; i < 6; i++)
    dcoordEtilt[i] = 0;
  if (!etilt)
    return;

  etilt *= -1;

  q1a = (1 - cos(angle)) * rho0 * (cos(etilt) - 1);
  q2a = 0;
  q3a = (1 - cos(angle)) * rho0 * sin(etilt);
  qp1 = sin(angle) * cos(etilt);
  qp2 = cos(angle);
  k = sqrt(qp1 * qp1 + qp2 * qp2);
  qp1 /= k;
  qp2 /= k;
  qp3 = sin(angle) * sin(etilt) / k;
  tan_alpha = 1.0 / tan(angle) / cos(etilt);
  q1b = q1a * tan_alpha / (tan(angle) + tan_alpha);
  q2b = -q1b * tan(angle);
  dz = sqrt((q1b - q1a) * (q1b - q1a) +
            (q2b - q2a) * (q2b - q2a));
  q3b = q3a + qp3 * dz;

  dcoordEtilt[0] = sqrt(q1b * q1b + q2b * q2b);
  dcoordEtilt[1] = tan(atan(tan_alpha) - (PIo2 - angle));
  dcoordEtilt[2] = q3b;
  dcoordEtilt[3] = qp3;
  dcoordEtilt[4] = dz * sqrt(1 + qp3 * qp3);
  gpuCsbendRotateCoordinatesForMisalignment(dcoordEtilt, -tilt);
}

static long gpuPackCsbendTracking(GPU_CSBEND_DATA *data, CSBEND *csbend,
                                  double Po) {
  double b[9], c[9], f[8], g[8];
  double rho, h, fse, tilt, term, e1, e2, psi1, psi2, angle;
  long i;

  if (!data || !gpuCsbendCommonSupported(csbend) ||
      !gpuCsbendEdgesSupported(csbend))
    return 0;
  if (!computeCSBENDFieldCoefficients || !(&Fx_xy) || !(&Fy_xy) ||
      !(&expansionOrder1) || !(&hasSkew) || !(&hasNormal))
    return 0;

  memset(data, 0, sizeof(*data));
  memset(b, 0, sizeof(b));
  memset(c, 0, sizeof(c));

  rho = csbend->length / csbend->angle;
  e1 = csbend->e[csbend->e1Index];
  e2 = csbend->e[csbend->e2Index];
  if (csbend->use_bn) {
    b[1] = csbend->b1;
    b[2] = csbend->b2;
    b[3] = csbend->b3;
    b[4] = csbend->b4;
    b[5] = csbend->b5;
    b[6] = csbend->b6;
    b[7] = csbend->b7;
    b[8] = csbend->b8;
  } else {
    b[1] = csbend->k1 * rho;
    b[2] = csbend->k2 * rho;
    b[3] = csbend->k3 * rho;
    b[4] = csbend->k4 * rho;
    b[5] = csbend->k5 * rho;
    b[6] = csbend->k6 * rho;
    b[7] = csbend->k7 * rho;
    b[8] = csbend->k8 * rho;
  }

  if (csbend->xReference > 0) {
    f[0] = csbend->f1;
    f[1] = csbend->f2;
    f[2] = csbend->f3;
    f[3] = csbend->f4;
    f[4] = csbend->f5;
    f[5] = csbend->f6;
    f[6] = csbend->f7;
    f[7] = csbend->f8;
    g[0] = csbend->g1;
    g[1] = csbend->g2;
    g[2] = csbend->g3;
    g[3] = csbend->g4;
    g[4] = csbend->g5;
    g[5] = csbend->g6;
    g[6] = csbend->g7;
    g[7] = csbend->g8;
    term = 1 / csbend->xReference;
    for (i = 0; i < 8; i++) {
      b[i + 1] += f[i] * term;
      c[i + 1] += g[i] * term;
      term *= (i + 2) / csbend->xReference;
    }
  }

  b[1] *= (1 + csbend->fse + csbend->fseQuadrupole) /
          (1 + csbend->fse + csbend->fseDipole);
  c[1] *= (1 + csbend->fse + csbend->fseQuadrupole) /
          (1 + csbend->fse + csbend->fseDipole);
  b[2] *= (1 + csbend->fse) / (1 + csbend->fse + csbend->fseDipole);
  c[2] *= (1 + csbend->fse) / (1 + csbend->fse + csbend->fseDipole);
  b[0] = csbend->xKick / csbend->angle;
  c[0] = csbend->yKick / csbend->angle;

  tilt = csbend->tilt;
  if (csbend->angle < 0 && csbend->malignMethod == 0) {
    rho = csbend->length / (-csbend->angle);
    angle = -csbend->angle;
    e1 = -e1;
    e2 = -e2;
    tilt += PI;
    for (i = 1; i < 9; i += 2) {
      b[i] *= -1;
      c[i] *= -1;
    }
  }
  else {
    angle = csbend->angle;
  }

  h = 1 / rho;
  fse = csbend->fse + csbend->fseDipole;
  if (fabs(fse + 1) < 1e-12)
    fse = -1 + 1e-12;

  computeCSBENDFieldCoefficients(b, c, h, csbend->nonlinear,
                                 csbend->expansionOrder);
  if (!Fx_xy || !Fy_xy || !*Fx_xy || !*Fy_xy)
    return 0;
  if (expansionOrder1 < 1 || expansionOrder1 > 11)
    return 0;

  data->nSlices = csbend->nSlices;
  data->integrationOrder = csbend->integration_order;
  data->expandHamiltonian = csbend->expandHamiltonian;
  data->hasSkew = hasSkew ? 1 : 0;
  data->hasNormal = hasNormal ? 1 : 0;
  data->expansionOrder1 = (int)expansionOrder1;
  data->length = csbend->length;
  data->rho0 = rho;
  data->rhoActual = 1 / ((1 + fse) * h);
  data->Po = Po;
  if (csbend->synch_rad)
    data->radCoef =
      sqr(particleCharge) * pow3(Po) * sqr(1 + fse) /
      (6 * PI * epsilon_o * sqr(c_mks) * particleMass * sqr(rho));
  psi1 = 2 * csbend->hgap *
         (csbend->fint[csbend->e1Index] >= 0 ?
            csbend->fint[csbend->e1Index] : csbend->fintBoth) *
         SIGN(rho) / fabs(data->rhoActual) / cos(e1) *
         (1 + sqr(sin(e1)));
  psi2 = 2 * csbend->hgap *
         (csbend->fint[csbend->e2Index] >= 0 ?
            csbend->fint[csbend->e2Index] : csbend->fintBoth) *
         SIGN(rho) / fabs(data->rhoActual) / cos(e2) *
         (1 + sqr(sin(e2)));
  if (!isfinite(psi1) || !isfinite(psi2))
    return 0;
  if (csbend->length < 0) {
    psi1 *= -1;
    psi2 *= -1;
  }
  data->edge1 = (csbend->edgeFlags & BEND_EDGE1_EFFECTS) ? 1 : 0;
  data->edge2 = (csbend->edgeFlags & BEND_EDGE2_EFFECTS) ? 1 : 0;
  data->edgeOrder = csbend->edge_order;
  data->e1 = e1;
  data->e2 = e2;
  data->he1 = csbend->h[csbend->e1Index];
  data->he2 = csbend->h[csbend->e2Index];
  data->psi1 = psi1;
  data->psi2 = psi2;
  data->fieldIndex = -b[1] / h;
  data->edgeKickLimit1 = csbend->edge_kick_limit[csbend->e1Index];
  data->edgeKickLimit2 = csbend->edge_kick_limit[csbend->e2Index];
  if (csbend->kick_limit_scaling) {
    data->edgeKickLimit1 *= rho / data->rhoActual;
    data->edgeKickLimit2 *= rho / data->rhoActual;
  }
  data->coordLimit = coordLimit;
  data->slopeLimit = slopeLimit;
  if (csbend->dx || csbend->dy || csbend->dz || csbend->etilt) {
    double dcoordEtilt[6] = {0, 0, 0, 0, 0, 0};
    double etilt = csbend->etilt * csbend->etiltSign;

    data->hasMisalignment = 1;
    data->dxi = -csbend->dx;
    data->dyi = -csbend->dy;
    data->dzi = csbend->dz;
    data->dxf = csbend->dx * cos(csbend->angle) +
                csbend->dz * sin(csbend->angle);
    data->dyf = csbend->dy;
    data->dzf = csbend->dx * sin(csbend->angle) -
                csbend->dz * cos(csbend->angle);
    gpuComputeEtiltCentroidOffset(dcoordEtilt, rho, angle, etilt, tilt);
    for (i = 0; i < 5; i++)
      data->dcoordEtilt[i] = dcoordEtilt[i];
  }
  gpuCsbendTiltSinCos(tilt + csbend->etilt * csbend->etiltSign,
                      &data->cosTilt, &data->sinTilt);
  memcpy(data->Fx, *Fx_xy, sizeof(data->Fx));
  memcpy(data->Fy, *Fy_xy, sizeof(data->Fy));
  return 1;
}

static long gpuCsbendOnCpu(long n_part, CSBEND *csbend, double p_error,
                           double Po, double **accepted, double z_start,
                           double *sigmaDelta2, char *rootname,
                           MAXAMP *maxamp, APCONTOUR *apContour,
                           APERTURE_DATA *apFileData, long iSlice,
                           ELEMENT_LIST *eptr, const char *reason) {
  double **coord = forceParticlesToCpu(reason);
  long remaining;

  if (!track_through_csbend)
    gpuRequiredFailure("CPU track_through_csbend fallback is unavailable");
  gpuBase.elementOnGpu = 0;
  remaining = track_through_csbend(coord, n_part, csbend, p_error, Po,
                                   accepted, z_start, sigmaDelta2, rootname,
                                   maxamp, apContour, apFileData, iSlice, eptr);
  gpuMarkHostWillChange();
  gpuRecordWallSeconds();
  return remaining;
}

long gpu_track_through_csbend(long n_part, void *csbend0, double p_error,
                              double Po, double **accepted, double z_start,
                              double *sigmaDelta2, char *rootname,
                              void *maxamp0, void *apContour0,
                              void *apFileData0, long iSlice, void *eptr0) {
  CSBEND *csbend = (CSBEND *)csbend0;
  MAXAMP *maxamp = (MAXAMP *)maxamp0;
  APCONTOUR *apContour = (APCONTOUR *)apContour0;
  APERTURE_DATA *apFileData = (APERTURE_DATA *)apFileData0;
  ELEMENT_LIST *eptr = (ELEMENT_LIST *)eptr0;
  GPU_CSBEND_DATA data;
  long lostCount = 0;
  float milliseconds = 0;
  int status;

  if (n_part <= 0)
    return n_part;
  if (!csbend)
    gpuRequiredFailure("NULL CSBEND pointer in gpu_track_through_csbend");
  if (gpuBase.backtrack)
    return gpuCsbendOnCpu(n_part, csbend, p_error, Po, accepted, z_start,
                          sigmaDelta2, rootname, maxamp, apContour, apFileData,
                          iSlice, eptr, "track_through_csbend backtracking CPU reference");
  if (iSlice >= 0)
    return gpuCsbendOnCpu(n_part, csbend, p_error, Po, accepted, z_start,
                          sigmaDelta2, rootname, maxamp, apContour, apFileData,
                          iSlice, eptr, "track_through_csbend slice fallback");
  if (maxamp || apContour || (apFileData && apFileData->initialized))
    return gpuCsbendOnCpu(n_part, csbend, p_error, Po, accepted, z_start,
                          sigmaDelta2, rootname, maxamp, apContour, apFileData,
                          iSlice, eptr, "track_through_csbend aperture fallback");
  if (sigmaDelta2)
    return gpuCsbendOnCpu(n_part, csbend, p_error, Po, accepted, z_start,
                          sigmaDelta2, rootname, maxamp, apContour, apFileData,
                          iSlice, eptr, "track_through_csbend radiation sigma fallback");
  if (!gpuPackCsbendTracking(&data, csbend, Po))
    return gpuCsbendOnCpu(n_part, csbend, p_error, Po, accepted, z_start,
                          sigmaDelta2, rootname, maxamp, apContour, apFileData,
                          iSlice, eptr, "track_through_csbend unsupported option");

  gpuCopyHostToDevice(n_part);
  status = gpuCudaCsbendTrackChecked(gpuBase.deviceCoord, n_part,
                                     (int)gpuBase.deviceStride, &data,
                                     &lostCount, &milliseconds);
  if (status != 0)
    gpuFatalStatus("CSBEND tracking CUDA checked kernel", status);
  gpuRecordMagnetKernel(milliseconds);
  if (lostCount) {
#ifndef GPU_VERIFY
    if (gpuMagnetLossCompactionAllowed()) {
      long remaining = gpuCsbendStableCompact(
        &data, n_part, accepted,
        "track_through_csbend stable magnet loss compaction");
      gpuDeviceIslandHasCsbend = 1;
      if (&multipoleKicksDone)
        multipoleKicksDone +=
          n_part * data.nSlices * (data.integrationOrder == 4 ? 4 : 1);
      return remaining;
    }
#endif
    return gpuCsbendOnCpu(n_part, csbend, p_error, Po, accepted, z_start,
                          sigmaDelta2, rootname, maxamp, apContour, apFileData,
                          iSlice, eptr, "track_through_csbend particle loss fallback");
  }
  gpuMarkDeviceChanged(n_part);
  gpuDeviceIslandHasCsbend = 1;
  gpuRecordWallSeconds();
  if (&multipoleKicksDone)
    multipoleKicksDone +=
      n_part * data.nSlices * (data.integrationOrder == 4 ? 4 : 1);
  return n_part;
}

static long gpuKickMapArraysReady(long points, long nx, long ny,
                                  double dxg, double dyg,
                                  double *xpFactor, double *ypFactor) {
  if (points <= 0 || nx <= 1 || ny <= 1)
    return 0;
  if (points < nx * ny)
    return 0;
  if (dxg == 0 || dyg == 0)
    return 0;
  if (!xpFactor || !ypFactor)
    return 0;
  return 1;
}

static void gpuEnsureKickMapCache(const double *xpFactor, const double *ypFactor,
                                  long points) {
  float milliseconds = 0;
  int status;

  if (!xpFactor || !ypFactor || points <= 0)
    gpuRequiredFailure("invalid KICKMAP cache request");
  if (gpuKickMapCache.xpFactor && gpuKickMapCache.ypFactor &&
      gpuKickMapCache.hostXpFactor == xpFactor &&
      gpuKickMapCache.hostYpFactor == ypFactor &&
      gpuKickMapCache.points == points)
    return;

  gpuReleaseKickMapCache();
  status = gpuCudaMallocDouble(&gpuKickMapCache.xpFactor, points);
  if (status != 0)
    gpuFatalStatus("cudaMalloc(KICKMAP xpFactor cache)", status);
  status = gpuCudaMallocDouble(&gpuKickMapCache.ypFactor, points);
  if (status != 0)
    gpuFatalStatus("cudaMalloc(KICKMAP ypFactor cache)", status);

  milliseconds = 0;
  status = gpuCudaCopyHostToDevice(gpuKickMapCache.xpFactor, xpFactor, points,
                                   &milliseconds);
  if (status != 0)
    gpuFatalStatus("cudaMemcpy(KICKMAP xpFactor host to device)", status);
  gpuRecordMilliseconds(&gpuBase.gpuTransferToDeviceSeconds, milliseconds);

  milliseconds = 0;
  status = gpuCudaCopyHostToDevice(gpuKickMapCache.ypFactor, ypFactor, points,
                                   &milliseconds);
  if (status != 0)
    gpuFatalStatus("cudaMemcpy(KICKMAP ypFactor host to device)", status);
  gpuRecordMilliseconds(&gpuBase.gpuTransferToDeviceSeconds, milliseconds);

  gpuKickMapCache.hostXpFactor = xpFactor;
  gpuKickMapCache.hostYpFactor = ypFactor;
  gpuKickMapCache.points = points;
}

static long gpuPackKickMapTracking(GPU_KICKMAP_DATA *data, KICKMAP *map,
                                   double pRef) {
  long kickSign;

  if (!data || !map)
    return 0;
  if (!map->initialized) {
    if (!initializeKickMap)
      return 0;
    initializeKickMap(map);
  }
  if (!gpuKickMapArraysReady(map->points, map->nx, map->ny, map->dxg,
                             map->dyg, map->xpFactor, map->ypFactor))
    return 0;
  if (map->nKicks < 1 || map->length == 0 || map->isr ||
      map->tilt || map->dx || map->dy || map->dz || map->yaw)
    return 0;
  kickSign = map->flipSign ? -1 : 1;
  memset(data, 0, sizeof(*data));
  data->undulator = 0;
  data->nKicks = map->nKicks;
  data->nx = map->nx;
  data->ny = map->ny;
  data->length = map->length / map->nKicks;
  data->halfLength = data->length / 2.0;
  data->xmin = map->xmin;
  data->ymin = map->ymin;
  data->dxg = map->dxg;
  data->dyg = map->dyg;
  data->kickScale = map->factor * kickSign / map->nKicks;
  data->pRef = pRef;
  if (map->synchRad)
    data->radCoef =
      sqr(particleCharge) * pow3(pRef) /
      (6 * PI * epsilon_o * sqr(c_mks) * particleMass);
  return 1;
}

static long gpuPackUndulatorKickMapTracking(GPU_KICKMAP_DATA *data,
                                            UKICKMAP *map, double pRef) {
  long kickSign;
  double eomc, K;

  if (!data || !map || pRef == 0)
    return 0;
  if (!map->initialized) {
    if (!initializeUndulatorKickMap)
      return 0;
    initializeUndulatorKickMap(map);
  }
  if (!gpuKickMapArraysReady(map->points, map->nx, map->ny, map->dxg,
                             map->dyg, map->xpFactor, map->ypFactor))
    return 0;
  if (map->nKicks < 1 || map->length == 0 || map->isr ||
      map->tilt || map->dx || map->dy || map->dz || map->yaw)
    return 0;
  if (map->synchRad && map->nKicks != map->periods)
    return 0;
  if (map->Kreference && map->Kactual)
    return 0;
  kickSign = map->flipSign ? -1 : 1;
  eomc = particleCharge / particleMass / c_mks;
  memset(data, 0, sizeof(*data));
  data->undulator = 1;
  data->nKicks = map->nKicks;
  data->nx = map->nx;
  data->ny = map->ny;
  data->length = map->length / map->nKicks;
  data->halfLength = data->length / 2.0;
  data->xmin = map->xmin;
  data->ymin = map->ymin;
  data->dxg = map->dxg;
  data->dyg = map->dyg;
  data->kickScale = sqr(map->fieldFactor * eomc / pRef) * kickSign;
  if (!map->singlePeriodMap)
    data->kickScale /= map->nKicks;
  data->pRef = pRef;
  if (map->synchRad) {
    double I1 = 0, I2 = 0, I3 = 0, I4 = 0, I5 = 0;
    double radCoef = 2. / 3 * particleRadius * ipow3(pRef);

    K = map->Kreference ? map->Kreference * map->fieldFactor : map->Kactual;
    if (map->periods && K) {
      double radius;

      if (!AddWigglerRadiationIntegrals)
        return 0;
      radius =
        sqrt(sqr(pRef) + 1) * (map->length / map->periods) / (PIx2 * K);
      AddWigglerRadiationIntegrals(
        map->length / map->periods, 2, radius, 0.0, 0.0, 1.0, 0.0,
        &I1, &I2, &I3, &I4, &I5);
    }
    data->radiationKick = radCoef * I2;
  }
  return 1;
}

static long gpuKickMapOnCpu(double **particle, double **accepted,
                            long nParticles, double pRef, KICKMAP *map,
                            double zStart, double *sigmaDelta2,
                            const char *reason) {
  double **coord = forceParticlesToCpu(reason);
  long remaining;

  if (!trackKickMap)
    gpuRequiredFailure("CPU KICKMAP fallback is unavailable");
  gpuBase.elementOnGpu = 0;
  remaining = trackKickMap(coord, accepted, nParticles, pRef, map, zStart,
                           sigmaDelta2);
  gpuMarkHostWillChange();
  gpuRecordWallSeconds();
  (void)particle;
  return remaining;
}

static long gpuUndulatorKickMapOnCpu(double **particle, double **accepted,
                                     long nParticles, double pRef,
                                     UKICKMAP *map, double zStart,
                                     const char *reason) {
  double **coord = forceParticlesToCpu(reason);
  long remaining;

  if (!trackUndulatorKickMap)
    gpuRequiredFailure("CPU UKICKMAP fallback is unavailable");
  gpuBase.elementOnGpu = 0;
  remaining = trackUndulatorKickMap(coord, accepted, nParticles, pRef, map,
                                    zStart);
  gpuMarkHostWillChange();
  gpuRecordWallSeconds();
  (void)particle;
  return remaining;
}

long gpu_track_kickmap(double **particle, double **accepted, long nParticles,
                       double pRef, void *map0, double zStart,
                       double *sigmaDelta2) {
  KICKMAP *map = (KICKMAP *)map0;
  GPU_KICKMAP_DATA data;
  long lostCount = 0;
  float milliseconds = 0;
  int status;

  if (nParticles <= 0)
    return nParticles;
  if (!map)
    gpuRequiredFailure("NULL KICKMAP pointer in CUDA path");
  if (gpuBase.backtrack)
    return gpuKickMapOnCpu(particle, accepted, nParticles, pRef, map, zStart,
                           sigmaDelta2, "KICKMAP backtracking CPU reference");
  if (sigmaDelta2)
    return gpuKickMapOnCpu(particle, accepted, nParticles, pRef, map, zStart,
                           sigmaDelta2, "KICKMAP radiation sigma fallback");
  if (!gpuPackKickMapTracking(&data, map, pRef))
    return gpuKickMapOnCpu(particle, accepted, nParticles, pRef, map, zStart,
                           sigmaDelta2, "KICKMAP unsupported CUDA option");
  gpuEnsureKickMapCache(map->xpFactor, map->ypFactor, map->points);
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaKickMapTrackChecked(gpuBase.deviceCoord, nParticles,
                                      (int)gpuBase.deviceStride, &data,
                                      gpuKickMapCache.xpFactor,
                                      gpuKickMapCache.ypFactor,
                                      map->points, &lostCount,
                                      &milliseconds);
  if (status != 0)
    gpuFatalStatus("KICKMAP tracking CUDA checked kernel", status);
  gpuRecordMagnetKernel(milliseconds);
  if (lostCount) {
#ifndef GPU_VERIFY
    if (gpuMagnetLossCompactionAllowed())
      return gpuKickMapStableCompact(
        &data, nParticles, accepted, zStart, pRef, gpuKickMapCache.xpFactor,
        gpuKickMapCache.ypFactor, map->points,
        "KICKMAP stable map-loss compaction");
#endif
    return gpuKickMapOnCpu(particle, accepted, nParticles, pRef, map, zStart,
                           sigmaDelta2, "KICKMAP particle loss fallback");
  }
#ifdef GPU_VERIFY
  if (!trackKickMap)
    gpuRequiredFailure("CPU KICKMAP verification fallback is unavailable");
  if (trackKickMap(gpuBase.coord, accepted, nParticles, pRef, map, zStart, NULL) != nParticles) {
    fprintf(stderr,
            "elegant CUDA VERIFY KICKMAP CPU shadow lost particles after CUDA no-loss check\n");
    exit(1);
  }
  compareGpuCpu(nParticles, "trackKickMap");
  gpuRecordWallSeconds();
  return nParticles;
#endif
  gpuMarkDeviceChanged(nParticles);
  gpuRecordWallSeconds();
  return nParticles;
}

long gpu_track_undulator_kickmap(double **particle, double **accepted,
                                 long nParticles, double pRef, void *map0,
                                 double zStart) {
  UKICKMAP *map = (UKICKMAP *)map0;
  GPU_KICKMAP_DATA data;
  long lostCount = 0;
  float milliseconds = 0;
  int status;

  if (nParticles <= 0)
    return nParticles;
  if (!map)
    gpuRequiredFailure("NULL UKICKMAP pointer in CUDA path");
  if (gpuBase.backtrack)
    return gpuUndulatorKickMapOnCpu(particle, accepted, nParticles, pRef, map,
                                    zStart, "UKICKMAP backtracking CPU reference");
  if (!gpuPackUndulatorKickMapTracking(&data, map, pRef))
    return gpuUndulatorKickMapOnCpu(particle, accepted, nParticles, pRef, map,
                                    zStart, "UKICKMAP unsupported CUDA option");
  gpuEnsureKickMapCache(map->xpFactor, map->ypFactor, map->points);
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaKickMapTrackChecked(gpuBase.deviceCoord, nParticles,
                                      (int)gpuBase.deviceStride, &data,
                                      gpuKickMapCache.xpFactor,
                                      gpuKickMapCache.ypFactor,
                                      map->points, &lostCount,
                                      &milliseconds);
  if (status != 0)
    gpuFatalStatus("UKICKMAP tracking CUDA checked kernel", status);
  gpuRecordMagnetKernel(milliseconds);
  if (lostCount) {
#ifndef GPU_VERIFY
    if (gpuMagnetLossCompactionAllowed())
      return gpuKickMapStableCompact(
        &data, nParticles, accepted, zStart, pRef, gpuKickMapCache.xpFactor,
        gpuKickMapCache.ypFactor, map->points,
        "UKICKMAP stable map-loss compaction");
#endif
    return gpuUndulatorKickMapOnCpu(particle, accepted, nParticles, pRef, map,
                                    zStart, "UKICKMAP particle loss fallback");
  }
#ifdef GPU_VERIFY
  if (!trackUndulatorKickMap)
    gpuRequiredFailure("CPU UKICKMAP verification fallback is unavailable");
  if (trackUndulatorKickMap(gpuBase.coord, accepted, nParticles, pRef, map,
                            zStart) != nParticles) {
    fprintf(stderr,
            "elegant CUDA VERIFY UKICKMAP CPU shadow lost particles after CUDA no-loss check\n");
    exit(1);
  }
  compareGpuCpu(nParticles, "trackUndulatorKickMap");
  gpuRecordWallSeconds();
  return nParticles;
#endif
  gpuMarkDeviceChanged(nParticles);
  gpuRecordWallSeconds();
  return nParticles;
}

static void gpuWakeTrackingWarning(const char *text, const char *detail) {
  if (printWarningForTracking)
    printWarningForTracking((char *)text, (char *)detail);
  else if (gpuVerbose)
    fprintf(stderr, "elegant CUDA WAKE warning: %s%s%s\n",
            text ? text : "",
            detail ? " " : "",
            detail ? detail : "");
}

static long gpuWakeSmoothingActive(long smoothing, long nb, long halfWidth) {
  return smoothing && nb >= (2 * halfWidth + 1);
}

static void gpuSmoothWakeHistogram(double *data, long nb, long order,
                                   long halfWidth, const char *elementType,
                                   const char *inputFile) {
  if (!data || nb <= 0)
    return;
  if (!SavitzkyGolaySmooth)
    gpuRequiredFailure("Savitzky-Golay smoothing routine unavailable in CUDA wake path");
  if (!SavitzkyGolaySmooth(data, nb, order, halfWidth, halfWidth, 0)) {
    fprintf(stderr, "Problem with smoothing for %s element (file %s)\n",
            elementType ? elementType : "wake",
            inputFile ? inputFile : "(null)");
    fprintf(stderr, "Parameters: nbins=%ld, order=%ld, half-width=%ld\n",
            nb, order, halfWidth);
    exit(1);
  }
}

#ifdef GPU_VERIFY
static long gpuWakeVerificationSuppressed = 0;

static void gpuCompareWakeArrayWithTolerance(
  const char *label, const char *arrayName,
  const double *cpu, const double *gpu, long n,
  double defaultAbsTol, double defaultRelTol) {
  double absTol = gpuEnvDouble("ELEGANT_GPU_WAKE_COMPARE_ABS",
                               gpuEnvDouble("ELEGANT_GPU_COMPARE_ABS",
                                            defaultAbsTol));
  double relTol = gpuEnvDouble("ELEGANT_GPU_WAKE_COMPARE_REL",
                               gpuEnvDouble("ELEGANT_GPU_COMPARE_REL",
                                            defaultRelTol));
  double maxAbs = 0, maxRel = 0;
  long mismatches = 0, i;

  for (i = 0; i < n; i++) {
    double absDiff, relDiff;
    if (!gpuValuesClose(cpu[i], gpu[i], absTol, relTol, &absDiff, &relDiff)) {
      if (mismatches < 10)
        fprintf(stderr,
                "elegant CUDA VERIFY wake mismatch %s %s[%ld] cpu=%.17e gpu=%.17e abs=%.3e rel=%.3e\n",
                label ? label : "unknown", arrayName ? arrayName : "array",
                i, cpu[i], gpu[i], absDiff, relDiff);
      mismatches++;
    }
    if (absDiff > maxAbs)
      maxAbs = absDiff;
    if (relDiff > maxRel)
      maxRel = relDiff;
  }

  if (mismatches) {
    fprintf(stderr,
            "elegant CUDA VERIFY wake failed for %s %s: %ld mismatches, maxAbs=%.3e, maxRel=%.3e, absTol=%.3e, relTol=%.3e\n",
            label ? label : "unknown", arrayName ? arrayName : "array",
            mismatches, maxAbs, maxRel, absTol, relTol);
    exit(1);
  }
  if (gpuVerbose)
    fprintf(stderr,
            "elegant CUDA VERIFY wake passed for %s %s: maxAbs=%.3e maxRel=%.3e\n",
            label ? label : "unknown", arrayName ? arrayName : "array",
            maxAbs, maxRel);
}

static void gpuCompareWakeArray(const char *label, const char *arrayName,
                                const double *cpu, const double *gpu, long n) {
  gpuCompareWakeArrayWithTolerance(label, arrayName, cpu, gpu, n,
                                   1e-9, 1e-10);
}

static void gpuVerifyWakeLongitudinal(WAKE *wakeData, long np, double Po,
                                      double tmin, double dt, long nb,
                                      double factor, long gpuBinned,
                                      const double *gpuItime,
                                      const double *gpuVtime) {
  double *cpuTime, *cpuItime, *cpuVtime;
  long *cpuPbin;
  long cpuBinned, ib;

  if (!computeTimeCoordinatesOnly || !binTimeDistribution || !convolveArrays)
    return;
  cpuTime = (double *)malloc(sizeof(*cpuTime) * np);
  cpuItime = (double *)malloc(sizeof(*cpuItime) * nb);
  cpuVtime = (double *)malloc(sizeof(*cpuVtime) * nb);
  cpuPbin = (long *)malloc(sizeof(*cpuPbin) * np);
  if (!cpuTime || !cpuItime || !cpuVtime || !cpuPbin)
    gpuRequiredFailure("unable to allocate CUDA WAKE verification buffers");

  computeTimeCoordinatesOnly(cpuTime, Po, gpuBase.coord, np);
  cpuBinned = binTimeDistribution(cpuItime, cpuPbin, tmin, dt, nb,
                                  cpuTime, gpuBase.coord, Po, np);
  if (cpuBinned != gpuBinned) {
    fprintf(stderr,
            "elegant CUDA VERIFY wake bin-count mismatch track_through_wake cpu=%ld gpu=%ld\n",
            cpuBinned, gpuBinned);
    exit(1);
  }
  if (gpuWakeSmoothingActive(wakeData->smoothing, nb, wakeData->SGHalfWidth))
    gpuSmoothWakeHistogram(cpuItime, nb, wakeData->SGOrder,
                           wakeData->SGHalfWidth, "WAKE",
                           wakeData->inputFile);
  convolveArrays(cpuVtime, nb, cpuItime, nb, wakeData->W,
                 wakeData->wakePoints, wakeData->i0);
  for (ib = 0; ib < nb; ib++)
    cpuVtime[ib] *= factor;

  gpuCompareWakeArray("track_through_wake", "Itime", cpuItime, gpuItime, nb);
  gpuCompareWakeArray("track_through_wake", "Vtime", cpuVtime, gpuVtime, nb);

  free(cpuTime);
  free(cpuItime);
  free(cpuVtime);
  free(cpuPbin);
}
#endif

typedef struct GPU_BUNCHED_WAKE_PLAN {
  long action;
  long useBunchFilter;
  long minBunch;
  long maxBunch;
  long nBuckets;
  long firstRelativeBunch;
  long lastRelativeBunch;
  long selectedBunch;
  long selectedRelativeBunch;
  long selectedStart;
  long selectedCount;
  const long *bucketStart;
  const long *bucketCount;
} GPU_BUNCHED_WAKE_PLAN;

typedef struct GPU_BUNCHED_WAKE_PLAN_CACHE {
  const void *owner;
  long particles;
  long valid;
  GPU_BUNCHED_WAKE_PLAN plan;
} GPU_BUNCHED_WAKE_PLAN_CACHE;

static GPU_BUNCHED_WAKE_PLAN_CACHE gpuImpedancePlanCache;
static GPU_BUNCHED_WAKE_PLAN_CACHE gpuCwakePlanCache;

static long gpuBunchedWakePlan(long np, long bunchedBeamMode,
                               long startBunch, long endBunch,
                               CHARGE *charge,
                               const char *operation,
                               GPU_BUNCHED_WAKE_PLAN *plan) {
  GPU_LONG_MIN_MAX_DATA range;
  long firstBucket, lastBucket, sorted = 0;
  long *newStart, *newCount;
  float milliseconds = 0;
  int status;

  if (!plan)
    return GPU_BUNCHED_WAKE_UNSUPPORTED;
  memset(plan, 0, sizeof(*plan));
  plan->action = GPU_BUNCHED_WAKE_UNSUPPORTED;
  if (!bunchedBeamMode || np <= 0) {
    plan->nBuckets = 1;
    plan->firstRelativeBunch = 0;
    plan->lastRelativeBunch = 0;
    plan->action = GPU_BUNCHED_WAKE_TRACK;
    return GPU_BUNCHED_WAKE_TRACK;
  }
#if USE_MPI
  if (distributedBeam)
    return GPU_BUNCHED_WAKE_UNSUPPORTED;
#endif
  if (!charge || charge->idSlotsPerBunch <= 0) {
    plan->minBunch = plan->maxBunch = 0;
    plan->nBuckets = 1;
  } else {
    if (bunchIndex >= totalPropertiesPerParticle)
      return GPU_BUNCHED_WAKE_UNSUPPORTED;

    memset(&range, 0, sizeof(range));
    gpuCopyHostToDevice(np);
    status = gpuCudaLongMinMax(gpuBase.deviceCoord, np,
                               (int)gpuBase.deviceStride, bunchIndex,
                               &range, &milliseconds);
    if (status != 0)
      gpuFatalStatus("wake bunch-index reduction kernel", status);
    gpuRecordReductionKernel(milliseconds);
    if (range.count != np || range.max < range.min)
      return GPU_BUNCHED_WAKE_UNSUPPORTED;
    plan->minBunch = range.min;
    plan->maxBunch = range.max;
    plan->nBuckets = range.max - range.min + 1;
  }

  if (gpuEnableCombinedWakeMultibunch && plan->nBuckets > 1 &&
      operation &&
      (strcmp(operation, "IMPEDANCE") == 0 || strcmp(operation, "CWAKE") == 0)) {
    if (plan->nBuckets > np)
      return GPU_BUNCHED_WAKE_UNSUPPORTED;
    if (!gpuBunchRangeCache.valid ||
        gpuBunchRangeCache.deviceCoord != gpuBase.deviceCoord ||
        gpuBunchRangeCache.particles != np ||
        gpuBunchRangeCache.stride != gpuBase.deviceStride ||
        gpuBunchRangeCache.coordinateIndex != bunchIndex ||
        gpuBunchRangeCache.minBunch != plan->minBunch ||
        gpuBunchRangeCache.maxBunch != plan->maxBunch) {
      newStart = (long *)malloc(plan->nBuckets * sizeof(*newStart));
      newCount = (long *)malloc(plan->nBuckets * sizeof(*newCount));
      if (!newStart || !newCount) {
        free(newStart);
        free(newCount);
        return GPU_BUNCHED_WAKE_UNSUPPORTED;
      }
      free(gpuBunchRangeCache.start);
      free(gpuBunchRangeCache.count);
      memset(&gpuBunchRangeCache, 0, sizeof(gpuBunchRangeCache));
      gpuBunchRangeCache.start = newStart;
      gpuBunchRangeCache.count = newCount;
      milliseconds = 0;
      status = gpuCudaSortedBunchRanges(
        gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, bunchIndex,
        plan->minBunch, plan->nBuckets, gpuBunchRangeCache.start,
        gpuBunchRangeCache.count, &sorted, &milliseconds);
      if (status != 0)
        gpuFatalStatus("wake sorted bunch-range kernel", status);
      gpuRecordReductionKernel(milliseconds);
      if (!sorted)
        return GPU_BUNCHED_WAKE_UNSUPPORTED;
      gpuBunchRangeCache.deviceCoord = gpuBase.deviceCoord;
      gpuBunchRangeCache.particles = np;
      gpuBunchRangeCache.stride = gpuBase.deviceStride;
      gpuBunchRangeCache.coordinateIndex = bunchIndex;
      gpuBunchRangeCache.minBunch = plan->minBunch;
      gpuBunchRangeCache.maxBunch = plan->maxBunch;
      gpuBunchRangeCache.nBuckets = plan->nBuckets;
      gpuBunchRangeCache.valid = 1;
    }
    plan->bucketStart = gpuBunchRangeCache.start;
    plan->bucketCount = gpuBunchRangeCache.count;
  }

  firstBucket = startBunch >= 0 ? startBunch : 0;
  lastBucket = endBunch >= 0 ? endBunch : plan->nBuckets - 1;
  if (firstBucket > plan->nBuckets - 1 || lastBucket < firstBucket) {
    plan->action = GPU_BUNCHED_WAKE_SKIP;
    return GPU_BUNCHED_WAKE_SKIP;
  }
  if (lastBucket > plan->nBuckets - 1)
    lastBucket = plan->nBuckets - 1;
  plan->firstRelativeBunch = firstBucket;
  plan->lastRelativeBunch = lastBucket;
  plan->selectedRelativeBunch = firstBucket;
  plan->selectedBunch = plan->minBunch + firstBucket;
  plan->useBunchFilter = plan->nBuckets > 1;
  if (plan->bucketStart && plan->bucketCount) {
    plan->selectedStart = plan->bucketStart[firstBucket];
    plan->selectedCount = plan->bucketCount[firstBucket];
  } else {
    plan->selectedStart = 0;
    plan->selectedCount = np;
  }
  plan->action = GPU_BUNCHED_WAKE_TRACK;
  return GPU_BUNCHED_WAKE_TRACK;
}

long gpu_rfmode_single_bunch_supported(long np, long bunchedBeamMode,
                                        void *charge0) {
  GPU_BUNCHED_WAKE_PLAN plan;
  long action;

  if (np <= 0 || bunchedBeamMode != 1)
    return 0;
  startGpuTimer();
  action = gpuBunchedWakePlan(np, bunchedBeamMode, -1, -1,
                              (CHARGE *)charge0, "RFMODE", &plan);
  gpuRecordWallSeconds();
  return action == GPU_BUNCHED_WAKE_TRACK && plan.nBuckets == 1;
}

double gpu_rfmode_time_mean(long np, double pCentral) {
  double sum = 0;
  float kernelMilliseconds = 0, transferMilliseconds = 0;
  int status;
  long ip;

  if (np <= 0)
    return 0;
  if (!gpuRfmodeHostTime || gpuRfmodeHostTimeCapacity < np) {
    double *replacement = (double *)realloc(
      gpuRfmodeHostTime, np * sizeof(*replacement));
    if (!replacement)
      gpuRequiredFailure("unable to allocate RFMODE/FRFMODE host time buffer");
    gpuRfmodeHostTime = replacement;
    gpuRfmodeHostTimeCapacity = np;
  }
  startGpuTimer();
  gpuCopyHostToDevice(np);
  status = gpuCudaRfmodeTimeCoordinates(
    gpuBase.deviceCoord, np, (int)gpuBase.deviceStride,
    pCentral, c_mks, gpuRfmodeHostTime,
    &kernelMilliseconds, &transferMilliseconds);
  if (status != 0)
    gpuFatalStatus("RFMODE/FRFMODE time-coordinate CUDA kernel", status);
  gpuRecordReductionKernel(kernelMilliseconds);
  gpuRecordMilliseconds(&gpuBase.gpuTransferToHostSeconds,
                        transferMilliseconds);
  for (ip = 0; ip < np; ip++)
    sum += gpuRfmodeHostTime[ip];
  gpuRecordWallSeconds();
  return sum / np;
}

#ifdef GPU_VERIFY
static double gpuRfmodeCpuTime(const double *part, double pCentral) {
  double p = pCentral * (1 + part[5]);
  return part[4] * sqrt(p * p + 1) / (c_mks * p);
}

static void gpuRfmodeCpuAddEnergy(double *part, double timeOfFlight,
                                  double pCentral, double dgamma) {
  double gamma, gamma1, pRatio, p, p1, pz1, pz;

  p = pCentral * (1 + part[5]);
  gamma1 = (gamma = sqrt(p * p + 1)) + dgamma;
  if (gamma1 <= 1)
    gamma1 = 1 + 1e-7;
  p1 = sqrt(gamma1 * gamma1 - 1);
  part[5] = (p1 - pCentral) / pCentral;
  part[4] = timeOfFlight * c_mks * p1 / gamma1;
  pz = p / sqrt(1 + part[1] * part[1] + part[3] * part[3]);
  pz1 = sqrt(pz * pz + gamma1 * gamma1 - gamma * gamma);
  pRatio = pz / pz1;
  part[1] *= pRatio;
  part[3] *= pRatio;
}
#endif

long gpu_rfmode_histogram(long np, double pCentral,
                          long bins, double tmin, double dt,
                          long *histogram, long *firstBin,
                          long *lastBin) {
  GPU_RFMODE_DATA data;
  long ib, binnedCount = 0;
  float milliseconds = 0;
  int status;

  if (np <= 0 || bins < 2 || dt <= 0 || !histogram ||
      !firstBin || !lastBin)
    gpuRequiredFailure("invalid RFMODE/FRFMODE CUDA histogram request");
  if (!gpuRfmodeHostHistogram || gpuRfmodeHostHistogramCapacity < bins) {
    unsigned long long *replacement = (unsigned long long *)realloc(
      gpuRfmodeHostHistogram, bins * sizeof(*replacement));
    if (!replacement)
      gpuRequiredFailure("unable to allocate RFMODE/FRFMODE host histogram");
    gpuRfmodeHostHistogram = replacement;
    gpuRfmodeHostHistogramCapacity = bins;
  }
  memset(&data, 0, sizeof(data));
  data.bins = bins;
  data.tmin = tmin;
  data.dt = dt;
  data.pCentral = pCentral;
  data.cMks = c_mks;
  startGpuTimer();
  gpuCopyHostToDevice(np);
  status = gpuCudaRfmodeHistogram(
    gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, &data,
    gpuRfmodeHostHistogram, &binnedCount, &milliseconds);
  if (status != 0)
    gpuFatalStatus("RFMODE/FRFMODE histogram CUDA kernel", status);
  gpuRecordWakeKernel(milliseconds);
  *firstBin = bins;
  *lastBin = 0;
  for (ib = 0; ib < bins; ib++) {
    if (gpuRfmodeHostHistogram[ib] > (unsigned long long)LONG_MAX)
      gpuRequiredFailure("RFMODE/FRFMODE histogram count overflow");
    histogram[ib] = (long)gpuRfmodeHostHistogram[ib];
    if (histogram[ib]) {
      if (ib < *firstBin)
        *firstBin = ib;
      if (ib > *lastBin)
        *lastBin = ib;
    }
  }
#ifdef GPU_VERIFY
  if (gpuBase.verifyMode) {
    double **coord = copyParticlesToCpuReadOnly(
      "RFMODE/FRFMODE histogram verification input");
    long *cpuHistogram = (long *)calloc(bins, sizeof(*cpuHistogram));
    long cpuBinned = 0;
    if (!cpuHistogram)
      gpuRequiredFailure("unable to allocate RFMODE/FRFMODE verification histogram");
    for (long ip = 0; ip < np; ip++) {
      double time = gpuRfmodeCpuTime(coord[ip], pCentral);
      long bin = (long)((time - tmin) / dt);
      if (bin < 0 || bin >= bins)
        continue;
      cpuHistogram[bin]++;
      cpuBinned++;
    }
    for (ib = 0; ib < bins; ib++)
      if (cpuHistogram[ib] != histogram[ib]) {
        fprintf(stderr,
                "elegant CUDA VERIFY mismatch RFMODE/FRFMODE histogram bin=%ld cpu=%ld gpu=%ld\n",
                ib, cpuHistogram[ib], histogram[ib]);
        free(cpuHistogram);
        exit(1);
      }
    free(cpuHistogram);
    if (cpuBinned != binnedCount)
      gpuRequiredFailure("RFMODE/FRFMODE verification binned-count mismatch");
    if (gpuVerbose)
      fprintf(stderr,
              "elegant CUDA VERIFY passed for RFMODE/FRFMODE histogram: %ld particles in %ld bins\n",
              binnedCount, bins);
  }
#endif
  gpuRecordWallSeconds();
  return binnedCount;
}

void gpu_rfmode_apply_kicks(long np, double pCentral,
                            long bins, double tmin, double dt,
                            long firstBin, long lastBin,
                            long interpolate, long nCavities,
                            const double *voltage) {
  GPU_RFMODE_DATA data;
  float milliseconds = 0;
  int status;

  if (np <= 0 || bins < 2 || dt <= 0 || firstBin < 0 ||
      lastBin < firstBin || lastBin >= bins || !voltage)
    gpuRequiredFailure("invalid RFMODE/FRFMODE CUDA kick request");
  memset(&data, 0, sizeof(data));
  data.bins = bins;
  data.firstBin = firstBin;
  data.lastBin = lastBin;
  data.interpolate = interpolate ? 1 : 0;
  data.nCavities = nCavities;
  data.tmin = tmin;
  data.dt = dt;
  data.pCentral = pCentral;
  data.particleMassMV = particleMassMV;
  data.particleRelSign = particleRelSign;
  data.cMks = c_mks;
  startGpuTimer();
#ifdef GPU_VERIFY
  if (gpuBase.verifyMode) {
    double **coord = copyParticlesToCpuReadOnly(
      "RFMODE/FRFMODE kick verification input");
    for (long ip = 0; ip < np; ip++) {
      double time = gpuRfmodeCpuTime(coord[ip], pCentral);
      double value;
      long ib = (long)((time - tmin) / dt);
      if (ib < 0 || ib >= bins)
        continue;
      if (interpolate) {
        double dt1 = time - (tmin + dt * (ib + 0.5));
        long ib1, ib2;
        if (dt1 < 0) {
          ib1 = ib - 1;
          ib2 = ib;
        } else {
          ib1 = ib;
          ib2 = ib + 1;
        }
        if (ib2 > lastBin) {
          ib2--;
          ib1--;
        }
        if (ib1 < firstBin) {
          ib1++;
          ib2++;
        }
        dt1 = time - (tmin + dt * (ib1 + 0.5));
        value = voltage[ib1] +
          (voltage[ib2] - voltage[ib1]) / dt * dt1;
      } else {
        value = voltage[ib];
      }
      gpuRfmodeCpuAddEnergy(
        coord[ip], time, pCentral,
        nCavities * value / (1e6 * particleMassMV * particleRelSign));
    }
  }
#endif
  status = gpuCudaRfmodeApplyKicks(
    gpuBase.deviceCoord, np, (int)gpuBase.deviceStride,
    &data, voltage, &milliseconds);
  if (status != 0)
    gpuFatalStatus("RFMODE/FRFMODE kick CUDA kernel", status);
  gpuRecordWakeKernel(milliseconds);
  gpuMarkDeviceChanged(np);
#ifdef GPU_VERIFY
  if (gpuBase.verifyMode)
    compareGpuCpu(np, "resident RFMODE/FRFMODE kick");
#endif
  gpuRecordWallSeconds();
}

long gpu_wake_bunched_mode_action(long np, void *wakeData0, void *charge0) {
  WAKE *wakeData = (WAKE *)wakeData0;
  GPU_BUNCHED_WAKE_PLAN plan;
  long action;

  if (!wakeData)
    return GPU_BUNCHED_WAKE_UNSUPPORTED;
  action = gpuBunchedWakePlan(
    np, wakeData->bunchedBeamMode, wakeData->startBunch,
    wakeData->endBunch, (CHARGE *)charge0, "WAKE", &plan);
  if (wakeData->change_p0 && action == GPU_BUNCHED_WAKE_SKIP)
    return GPU_BUNCHED_WAKE_MATCH_ONLY;
  return action;
}

long gpu_wake_bunched_mode_supported(long np, void *wakeData0, void *charge0) {
  return gpu_wake_bunched_mode_action(np, wakeData0, charge0) !=
         GPU_BUNCHED_WAKE_UNSUPPORTED;
}

long gpu_trwake_bunched_mode_action(long np, void *wakeData0, void *charge0) {
  TRWAKE *wakeData = (TRWAKE *)wakeData0;
  GPU_BUNCHED_WAKE_PLAN plan;

  if (!wakeData)
    return GPU_BUNCHED_WAKE_UNSUPPORTED;
  return gpuBunchedWakePlan(
    np, wakeData->bunchedBeamMode, wakeData->startBunch,
    wakeData->endBunch, (CHARGE *)charge0, "TRWAKE", &plan);
}

long gpu_trwake_bunched_mode_supported(long np, void *wakeData0, void *charge0) {
  return gpu_trwake_bunched_mode_action(np, wakeData0, charge0) !=
         GPU_BUNCHED_WAKE_UNSUPPORTED;
}

long gpu_impedance_bunched_mode_action(long np, void *impedanceData0,
                                       void *charge0) {
  IMPEDANCE *impedanceData = (IMPEDANCE *)impedanceData0;
  GPU_BUNCHED_WAKE_PLAN plan;
  long action;

  if (!impedanceData)
    return GPU_BUNCHED_WAKE_UNSUPPORTED;
  gpuImpedancePlanCache.valid = 0;
  action = gpuBunchedWakePlan(
    np, impedanceData->bunchedBeamMode, impedanceData->startBunch,
    impedanceData->endBunch, (CHARGE *)charge0, "IMPEDANCE", &plan);
  /* Multiple bunches are enabled as a separately measured follow-on stage. */
  if (!gpuEnableCombinedWakeMultibunch &&
      action != GPU_BUNCHED_WAKE_UNSUPPORTED && plan.nBuckets > 1)
    return GPU_BUNCHED_WAKE_UNSUPPORTED;
  if (action != GPU_BUNCHED_WAKE_UNSUPPORTED) {
    gpuImpedancePlanCache.owner = impedanceData;
    gpuImpedancePlanCache.particles = np;
    gpuImpedancePlanCache.plan = plan;
    gpuImpedancePlanCache.valid = 1;
  }
  return action;
}

long gpu_cwake_bunched_mode_action(long np, void *wakeData0, void *charge0) {
  CWAKE *wakeData = (CWAKE *)wakeData0;
  GPU_BUNCHED_WAKE_PLAN plan;
  long action;

  if (!wakeData)
    return GPU_BUNCHED_WAKE_UNSUPPORTED;
  gpuCwakePlanCache.valid = 0;
  action = gpuBunchedWakePlan(
    np, wakeData->bunchedBeamMode, wakeData->startBunch,
    wakeData->endBunch, (CHARGE *)charge0, "CWAKE", &plan);
  /* Multiple bunches are enabled as a separately measured follow-on stage. */
  if (!gpuEnableCombinedWakeMultibunch &&
      action != GPU_BUNCHED_WAKE_UNSUPPORTED && plan.nBuckets > 1)
    return GPU_BUNCHED_WAKE_UNSUPPORTED;
  if (action != GPU_BUNCHED_WAKE_UNSUPPORTED) {
    gpuCwakePlanCache.owner = wakeData;
    gpuCwakePlanCache.particles = np;
    gpuCwakePlanCache.plan = plan;
    gpuCwakePlanCache.valid = 1;
  }
  if (wakeData->change_p0 && action == GPU_BUNCHED_WAKE_SKIP)
    return GPU_BUNCHED_WAKE_MATCH_ONLY;
  return action;
}

static long gpuWakeTimeSums(long np, double Po,
                            const GPU_BUNCHED_WAKE_PLAN *plan,
                            GPU_BEAM_SUM_DATA *sums,
                            const char *operation) {
  float milliseconds = 0;
  int status;

  memset(sums, 0, sizeof(*sums));
  if (plan && plan->useBunchFilter && plan->bucketStart &&
      plan->selectedCount > 0) {
    status = gpuCudaBeamSums(
      ((double *)gpuBase.deviceCoord) +
        plan->selectedStart * gpuBase.deviceStride,
      plan->selectedCount, (int)gpuBase.deviceStride,
      Po, c_mks, sums, &milliseconds);
  } else if (plan && plan->useBunchFilter) {
    status = gpuCudaSelectedTimeSums(gpuBase.deviceCoord, np,
                                     (int)gpuBase.deviceStride,
                                     Po, c_mks, bunchIndex,
                                     plan->selectedBunch,
                                     sums, &milliseconds);
  } else {
    status = gpuCudaBeamSums(gpuBase.deviceCoord, np,
                             (int)gpuBase.deviceStride,
                             Po, c_mks, sums, &milliseconds);
  }
  if (status != 0)
    gpuFatalStatus(operation, status);
  gpuRecordReductionKernel(milliseconds);
  return sums->count;
}

static const int gpuImpedanceDriver[IMPEDANCE_N_WAKES] = {0, 1, 2, 0, 0};
static const int gpuImpedanceKickPlane[IMPEDANCE_N_WAKES] = {0, 1, 2, 1, 2};
static const long gpuImpedanceProbeExponent[IMPEDANCE_N_WAKES] = {0, 0, 0, 1, 1};
static const int gpuCwakeDriver[CWAKE_N_WAKES] = {0, 1, 2, 0, 0, 0, 0};
static const int gpuCwakeKickPlane[CWAKE_N_WAKES] = {0, 1, 2, 1, 2, 1, 2};
static const long gpuCwakeProbeExponent[CWAKE_N_WAKES] = {0, 0, 0, 1, 1, 0, 0};

static double gpuImpedanceChannelFactor(const IMPEDANCE *imp, long channel) {
  switch (channel) {
  case IMPEDANCE_ZL:  return imp->zFactor;
  case IMPEDANCE_ZXD: return imp->xFactor;
  case IMPEDANCE_ZYD: return imp->yFactor;
  case IMPEDANCE_ZXQ: return imp->qxFactor;
  case IMPEDANCE_ZYQ: return imp->qyFactor;
  default:            return 1;
  }
}

static double gpuCwakeChannelFactor(const CWAKE *wake, long channel) {
  switch (channel) {
  case CWAKE_WZ: return wake->zFactor;
  case CWAKE_WX: return wake->xFactor;
  case CWAKE_WY: return wake->yFactor;
  case CWAKE_QX: return wake->qxFactor;
  case CWAKE_QY: return wake->qyFactor;
  case CWAKE_CX: return wake->cxFactor;
  case CWAKE_CY: return wake->cyFactor;
  default:       return 1;
  }
}

#ifdef GPU_VERIFY
static void gpuVerifyCombinedHistogram(
  const GPU_COMBINED_WAKE_DATA *data, double **coord, long np,
  const double *gpuHistogram, double **cpuHistogramReturn,
  long **cpuPbinReturn, double **cpuTimeReturn) {
  double *cpuHistogram, *cpuTime;
  long *cpuPbin;
  long ip, ib;

  cpuHistogram = (double *)calloc(3 * data->bins, sizeof(*cpuHistogram));
  cpuTime = (double *)malloc(np * sizeof(*cpuTime));
  cpuPbin = (long *)malloc(np * sizeof(*cpuPbin));
  if (!cpuHistogram || !cpuTime || !cpuPbin)
    gpuRequiredFailure("unable to allocate combined-wake verification buffers");
  computeTimeCoordinatesOnly(cpuTime, data->pCentral, coord, np);
  for (ip = 0; ip < np; ip++) {
    cpuPbin[ip] = -1;
    if (data->useBunchFilter &&
        (long)coord[ip][data->bunchIndexColumn] != data->selectedBunch)
      continue;
    ib = (long)((cpuTime[ip] - data->tmin) / data->dt + 0.5);
    if (ib < 0 || ib >= data->bins)
      continue;
    cpuPbin[ip] = ib;
    cpuHistogram[ib] += 1;
    cpuHistogram[data->bins + ib] += coord[ip][0] - data->offset[0];
    cpuHistogram[2 * data->bins + ib] += coord[ip][2] - data->offset[1];
  }
  gpuCompareWakeArray("combined wake", "Itime", cpuHistogram,
                      gpuHistogram, data->bins);
  gpuCompareWakeArray("combined wake", "xItime",
                      cpuHistogram + data->bins,
                      gpuHistogram + data->bins, data->bins);
  gpuCompareWakeArray("combined wake", "yItime",
                      cpuHistogram + 2 * data->bins,
                      gpuHistogram + 2 * data->bins, data->bins);
  *cpuHistogramReturn = cpuHistogram;
  *cpuPbinReturn = cpuPbin;
  *cpuTimeReturn = cpuTime;
}

static void gpuVerifyCwakeIntermediate(
  CWAKE *wake, const GPU_COMBINED_WAKE_DATA *data, double **coord, long np,
  const double *gpuHistogram, const double *gpuVoltage) {
  double *cpuHistogram, *cpuTime, *cpuVoltage;
  long *cpuPbin;

  gpuVerifyCombinedHistogram(data, coord, np, gpuHistogram, &cpuHistogram,
                             &cpuPbin, &cpuTime);
  cpuVoltage = (double *)malloc(data->bins * sizeof(*cpuVoltage));
  if (!cpuVoltage)
    gpuRequiredFailure("unable to allocate CWAKE voltage verification buffer");
  for (long channel = 0; channel < CWAKE_N_WAKES; channel++) {
    if (!wake->enabled[channel])
      continue;
    convolveArrays(cpuVoltage, data->bins,
                   cpuHistogram + data->driver[channel] * data->bins,
                   data->bins, wake->W[channel], wake->wakePoints, wake->i0);
    for (long ib = 0; ib < data->bins; ib++)
      cpuVoltage[ib] *= data->factor[channel];
    if (data->allowTimeFft)
      gpuCompareWakeArrayWithTolerance(
        "track_through_cwake FFT", "Vtime", cpuVoltage,
        gpuVoltage + channel * data->bins, data->bins, 1e-6, 1e-8);
    else
      gpuCompareWakeArray("track_through_cwake", "Vtime", cpuVoltage,
                          gpuVoltage + channel * data->bins, data->bins);
  }
  free(cpuVoltage);
  free(cpuHistogram);
  free(cpuPbin);
  free(cpuTime);
}

static void gpuApplyImpedanceMultiplication(
  const double *ifreq, const double *impedance, double *vfreq,
  long bins, double factor) {
  long frequencyPoints = bins / 2 + 1;
  vfreq[0] = ifreq[0] * impedance[0] * factor;
  if (!(bins % 2))
    vfreq[bins - 1] = ifreq[bins - 1] * impedance[bins - 1] * factor;
  for (long frequency = 1; frequency < frequencyPoints - 1; frequency++) {
    long realIndex = 2 * frequency - 1;
    long imagIndex = realIndex + 1;
    vfreq[realIndex] =
      (ifreq[realIndex] * impedance[realIndex] -
       ifreq[imagIndex] * impedance[imagIndex]) * factor;
    vfreq[imagIndex] =
      (ifreq[realIndex] * impedance[imagIndex] +
       ifreq[imagIndex] * impedance[realIndex]) * factor;
  }
}

static void gpuVerifyImpedanceIntermediate(
  IMPEDANCE *imp, const GPU_COMBINED_WAKE_DATA *data, double **coord, long np,
  const double *gpuHistogram, const double *gpuVoltage) {
  double *cpuHistogram, *cpuTime, *ifreq, *cpuVoltage;
  long *cpuPbin;

  gpuVerifyCombinedHistogram(data, coord, np, gpuHistogram, &cpuHistogram,
                             &cpuPbin, &cpuTime);
  ifreq = (double *)calloc(2 * data->bins, sizeof(*ifreq));
  cpuVoltage = (double *)calloc(2 * (data->bins + 1), sizeof(*cpuVoltage));
  if (!ifreq || !cpuVoltage)
    gpuRequiredFailure("unable to allocate IMPEDANCE voltage verification buffers");
  for (long channel = 0; channel < IMPEDANCE_N_WAKES; channel++) {
    if (!imp->enabled[channel])
      continue;
    memset(ifreq, 0, 2 * data->bins * sizeof(*ifreq));
    memcpy(ifreq, cpuHistogram + data->driver[channel] * data->bins,
           data->bins * sizeof(*ifreq));
    realFFT(ifreq, data->bins, 0);
    memset(cpuVoltage, 0, 2 * (data->bins + 1) * sizeof(*cpuVoltage));
    gpuApplyImpedanceMultiplication(ifreq, imp->Z[channel], cpuVoltage,
                                    data->bins, data->factor[channel]);
    realFFT(cpuVoltage, data->bins, INVERSE_FFT);
    gpuCompareWakeArray("track_through_impedance", "Vtime", cpuVoltage,
                        gpuVoltage + channel * data->bins, data->bins);
  }
  free(ifreq);
  free(cpuVoltage);
  free(cpuHistogram);
  free(cpuPbin);
  free(cpuTime);
}
#endif

static long gpuTrackImpedanceBucket(
  long np, IMPEDANCE *imp, double Po, double rampFactor,
  const GPU_BUNCHED_WAKE_PLAN *plan) {
  GPU_BEAM_SUM_DATA sums;
  GPU_COMBINED_WAKE_DATA data;
  const double *tables[GPU_COMBINED_WAKE_CHANNELS] = {NULL};
  long expectedParticles, binnedCount = 0, trackedParticles, startParticle;
  double *deviceCoord;
  double beamSpan, tmin, tmax;
  float milliseconds = 0;
  int status;
#ifdef GPU_VERIFY
  double *gpuHistogram = NULL, *gpuVoltage = NULL;
  double **hostCoord;
#endif

  startParticle = plan->useBunchFilter && plan->bucketStart ?
                  plan->selectedStart : 0;
  trackedParticles = plan->useBunchFilter && plan->bucketStart ?
                     plan->selectedCount : np;
  if (trackedParticles <= 0)
    return 0;
  deviceCoord = ((double *)gpuBase.deviceCoord) +
                startParticle * gpuBase.deviceStride;
#ifdef GPU_VERIFY
  hostCoord = gpuBase.coord + startParticle;
#endif
  expectedParticles = gpuWakeTimeSums(
    np, Po, plan, &sums, "IMPEDANCE time-coordinate reduction kernel");
  if (expectedParticles <= 0)
    return 0;
  tmin = sums.min[6];
  tmax = sums.max[6];
  tmin -= imp->bin_size;
  tmax += imp->bin_size;
  beamSpan = tmax - tmin;
  if (beamSpan * 2 > imp->n_bins * imp->bin_size && !imp->allowLongBeam) {
    fprintf(stderr,
            "IMPEDANCE: time span %21.15le s exceeds half the total time span %21.15le s.\n",
            beamSpan, imp->n_bins * imp->bin_size);
    exit(1);
  }

  memset(&data, 0, sizeof(data));
  data.bins = imp->n_bins;
  data.tablePoints = imp->n_bins;
  data.mode = GPU_COMBINED_WAKE_MODE_IMPEDANCE;
  data.interpolate = (int)imp->interpolate;
  data.useBunchFilter =
    (int)(plan->useBunchFilter && !plan->bucketStart);
  data.bunchIndexColumn = bunchIndex;
  data.selectedBunch = plan->selectedBunch;
  data.tmin = tmin;
  data.dt = imp->bin_size;
  data.pCentral = Po;
  data.offset[0] = imp->dx;
  data.offset[1] = imp->dy;
  data.particleMassMV = particleMassMV;
  data.particleRelSign = particleRelSign;
  data.cMks = c_mks;
  for (long channel = 0; channel < IMPEDANCE_N_WAKES; channel++) {
    data.enabled[channel] = imp->enabled[channel] ? 1 : 0;
    data.driver[channel] = gpuImpedanceDriver[channel];
    data.kickPlane[channel] = gpuImpedanceKickPlane[channel];
    data.probeExponent[channel] = gpuImpedanceProbeExponent[channel];
    data.factor[channel] =
      imp->macroParticleCharge * particleRelSign / imp->bin_size *
      imp->factor * gpuImpedanceChannelFactor(imp, channel) * rampFactor;
    tables[channel] = imp->Z[channel];
  }
#ifdef GPU_VERIFY
  gpuHistogram = (double *)malloc(3 * data.bins * sizeof(*gpuHistogram));
  gpuVoltage = (double *)malloc(GPU_COMBINED_WAKE_CHANNELS * data.bins *
                                sizeof(*gpuVoltage));
  if (!gpuHistogram || !gpuVoltage)
    gpuRequiredFailure("unable to allocate IMPEDANCE GPU verification returns");
#endif
  status = gpuCudaCombinedWakeTrack(
    deviceCoord, trackedParticles, (int)gpuBase.deviceStride, &data, tables,
    &binnedCount,
#ifdef GPU_VERIFY
    gpuHistogram, gpuVoltage,
#else
    NULL, NULL,
#endif
    &milliseconds);
  if (status != 0)
    gpuFatalStatus("IMPEDANCE combined CUDA kernel", status);
  gpuRecordWakeKernel(milliseconds);
  if (binnedCount != expectedParticles)
    gpuWakeTrackingWarning("Some particles were not binned in IMPEDANCE.",
                           "Reduce impedance frequency spacing or enable ALLOW_LONG_BEAM.");
#ifdef GPU_VERIFY
  gpuVerifyImpedanceIntermediate(imp, &data, hostCoord, trackedParticles,
                                 gpuHistogram, gpuVoltage);
  free(gpuHistogram);
  free(gpuVoltage);
#endif
  return 1;
}

static long gpuTrackCwakeBucket(
  long np, CWAKE *wake, double Po, double rampFactor,
  const GPU_BUNCHED_WAKE_PLAN *plan) {
  GPU_BEAM_SUM_DATA sums;
  GPU_COMBINED_WAKE_DATA data;
  const double *tables[GPU_COMBINED_WAKE_CHANNELS] = {NULL};
  long expectedParticles, binnedCount = 0, bins, trackedParticles, startParticle;
  double *deviceCoord;
  double beamSpan, wakeSpan, tmin, tmax;
  float milliseconds = 0;
  int status;
#ifdef GPU_VERIFY
  double *gpuHistogram = NULL, *gpuVoltage = NULL;
  double **hostCoord;
#endif

  startParticle = plan->useBunchFilter && plan->bucketStart ?
                  plan->selectedStart : 0;
  trackedParticles = plan->useBunchFilter && plan->bucketStart ?
                     plan->selectedCount : np;
  if (trackedParticles <= 0)
    return 0;
  deviceCoord = ((double *)gpuBase.deviceCoord) +
                startParticle * gpuBase.deviceStride;
#ifdef GPU_VERIFY
  hostCoord = gpuBase.coord + startParticle;
#endif
  expectedParticles = gpuWakeTimeSums(
    np, Po, plan, &sums, "CWAKE time-coordinate reduction kernel");
  if (expectedParticles <= 0)
    return 0;
  tmin = sums.min[6];
  tmax = sums.max[6];
  beamSpan = tmax - tmin;
  wakeSpan = wake->t[wake->wakePoints - 1] - wake->t[0];
  if (beamSpan > wakeSpan && !wake->allowLongBeam) {
    fprintf(stderr,
            "CWAKE: beam length %le s exceeds wake length %le s.\n",
            beamSpan, wakeSpan);
    exit(1);
  }
  if (wake->n_bins) {
    bins = wake->n_bins;
    tmin = sums.centroidSum[6] / sums.count - wake->dt * bins / 2.0;
  } else {
    bins = beamSpan / wake->dt + 3;
    tmin -= wake->dt;
  }
  if (bins <= 0)
    return 0;

  memset(&data, 0, sizeof(data));
  data.bins = bins;
  data.tablePoints = wake->wakePoints;
  data.i0 = wake->i0;
  data.mode = GPU_COMBINED_WAKE_MODE_TIME;
  data.interpolate = (int)wake->interpolate;
  data.allowTimeFft = (int)gpuEnableCombinedWakeFft;
  data.useBunchFilter =
    (int)(plan->useBunchFilter && !plan->bucketStart);
  data.bunchIndexColumn = bunchIndex;
  data.selectedBunch = plan->selectedBunch;
  data.tmin = tmin;
  data.dt = wake->dt;
  data.pCentral = Po;
  data.offset[0] = wake->dx;
  data.offset[1] = wake->dy;
  data.particleMassMV = particleMassMV;
  data.particleRelSign = particleRelSign;
  data.cMks = c_mks;
  for (long channel = 0; channel < CWAKE_N_WAKES; channel++) {
    data.enabled[channel] = wake->enabled[channel] ? 1 : 0;
    data.driver[channel] = gpuCwakeDriver[channel];
    data.kickPlane[channel] = gpuCwakeKickPlane[channel];
    data.probeExponent[channel] = gpuCwakeProbeExponent[channel];
    data.factor[channel] =
      wake->macroParticleCharge * particleRelSign * wake->factor *
      gpuCwakeChannelFactor(wake, channel) * rampFactor;
    tables[channel] = wake->W[channel];
  }
#ifdef GPU_VERIFY
  gpuHistogram = (double *)malloc(3 * bins * sizeof(*gpuHistogram));
  gpuVoltage = (double *)malloc(GPU_COMBINED_WAKE_CHANNELS * bins *
                                sizeof(*gpuVoltage));
  if (!gpuHistogram || !gpuVoltage)
    gpuRequiredFailure("unable to allocate CWAKE GPU verification returns");
#endif
  status = gpuCudaCombinedWakeTrack(
    deviceCoord, trackedParticles, (int)gpuBase.deviceStride, &data, tables,
    &binnedCount,
#ifdef GPU_VERIFY
    gpuHistogram, gpuVoltage,
#else
    NULL, NULL,
#endif
    &milliseconds);
  if (status != 0)
    gpuFatalStatus("CWAKE combined CUDA kernel", status);
  gpuRecordWakeKernel(milliseconds);
  if (binnedCount != expectedParticles)
    gpuWakeTrackingWarning("Some particles were not binned in CWAKE.",
                           "Set N_BINS=0 to invoke autoscaling.");
#ifdef GPU_VERIFY
  gpuVerifyCwakeIntermediate(wake, &data, hostCoord, trackedParticles,
                             gpuHistogram, gpuVoltage);
  free(gpuHistogram);
  free(gpuVoltage);
#endif
  return 1;
}

void gpu_track_through_impedance(long np, void *impedanceData0, double Po,
                                 void *run0, long iPass, void *charge0) {
  IMPEDANCE *imp = (IMPEDANCE *)impedanceData0;
  RUN *run = (RUN *)run0;
  CHARGE *charge = (CHARGE *)charge0;
  GPU_BUNCHED_WAKE_PLAN plan, bucketPlan;
  long adjustedPass, first, last, tracked = 0;
  double rampFactor;

  if (np <= 0 || !imp)
    return;
  adjustedPass = iPass - imp->startOnPass;
  if (adjustedPass < 0 || imp->factor == 0) {
    gpuRecordWallSeconds();
    return;
  }
  if (!set_up_impedance)
    gpuRequiredFailure("IMPEDANCE setup routine is unavailable");
  set_up_impedance(imp, run, adjustedPass, np, charge);
  if (imp->n_bins < 2)
    gpuRequiredFailure("IMPEDANCE data was not initialized for CUDA tracking");
  if (gpuImpedancePlanCache.valid &&
      gpuImpedancePlanCache.owner == imp &&
      gpuImpedancePlanCache.particles == np) {
    plan = gpuImpedancePlanCache.plan;
    gpuImpedancePlanCache.valid = 0;
  } else if (gpuBunchedWakePlan(np, imp->bunchedBeamMode, imp->startBunch,
                                imp->endBunch, charge, "IMPEDANCE", &plan) ==
             GPU_BUNCHED_WAKE_UNSUPPORTED) {
    gpuRequiredFailure("unsupported IMPEDANCE bunch layout reached CUDA path");
  }
  if (plan.action == GPU_BUNCHED_WAKE_SKIP) {
    gpuRecordWallSeconds();
    return;
  }
  rampFactor = imp->rampPasses <= 1 ||
               adjustedPass >= imp->rampPasses - 1 ?
               1 : (adjustedPass + 1.0) / imp->rampPasses;
  gpuCopyHostToDevice(np);
  first = plan.useBunchFilter ? plan.firstRelativeBunch : 0;
  last = plan.useBunchFilter ? plan.lastRelativeBunch : 0;
  for (long relative = first; relative <= last; relative++) {
    bucketPlan = plan;
    if (bucketPlan.useBunchFilter) {
      bucketPlan.selectedRelativeBunch = relative;
      bucketPlan.selectedBunch = bucketPlan.minBunch + relative;
      if (bucketPlan.bucketStart && bucketPlan.bucketCount) {
        bucketPlan.selectedStart = bucketPlan.bucketStart[relative];
        bucketPlan.selectedCount = bucketPlan.bucketCount[relative];
      }
    }
    if (gpuTrackImpedanceBucket(np, imp, Po, rampFactor, &bucketPlan) > 0)
      tracked++;
  }
  if (tracked)
    gpuMarkDeviceChanged(np);
  gpuRecordWallSeconds();
}

void gpu_track_through_cwake(long np, void *wakeData0, double *PoInput,
                             void *run0, long iPass, void *charge0) {
  CWAKE *wake = (CWAKE *)wakeData0;
  RUN *run = (RUN *)run0;
  CHARGE *charge = (CHARGE *)charge0;
  GPU_BUNCHED_WAKE_PLAN plan, bucketPlan;
  long first, last, tracked = 0;
  double rampFactor, Po;

  if (np <= 0 || !wake || !PoInput)
    return;
  if (!set_up_cwake)
    gpuRequiredFailure("CWAKE setup routine is unavailable");
  set_up_cwake(wake, run, iPass, np, charge);
  if (gpuCwakePlanCache.valid &&
      gpuCwakePlanCache.owner == wake &&
      gpuCwakePlanCache.particles == np) {
    plan = gpuCwakePlanCache.plan;
    gpuCwakePlanCache.valid = 0;
  } else if (gpuBunchedWakePlan(np, wake->bunchedBeamMode, wake->startBunch,
                                wake->endBunch, charge, "CWAKE", &plan) ==
             GPU_BUNCHED_WAKE_UNSUPPORTED) {
    gpuRequiredFailure("unsupported CWAKE bunch layout reached CUDA path");
  }
  if (plan.action == GPU_BUNCHED_WAKE_SKIP) {
    if (wake->change_p0)
      gpu_do_match_energy(np, PoInput, 0);
    gpuRecordWallSeconds();
    return;
  }
  rampFactor = wake->rampPasses <= 1 || iPass >= wake->rampPasses - 1 ?
               1 : (iPass + 1.0) / wake->rampPasses;
  Po = *PoInput;
  gpuCopyHostToDevice(np);
  first = plan.useBunchFilter ? plan.firstRelativeBunch : 0;
  last = plan.useBunchFilter ? plan.lastRelativeBunch : 0;
  for (long relative = first; relative <= last; relative++) {
    bucketPlan = plan;
    if (bucketPlan.useBunchFilter) {
      bucketPlan.selectedRelativeBunch = relative;
      bucketPlan.selectedBunch = bucketPlan.minBunch + relative;
      if (bucketPlan.bucketStart && bucketPlan.bucketCount) {
        bucketPlan.selectedStart = bucketPlan.bucketStart[relative];
        bucketPlan.selectedCount = bucketPlan.bucketCount[relative];
      }
    }
    if (gpuTrackCwakeBucket(np, wake, Po, rampFactor, &bucketPlan) > 0)
      tracked++;
  }
  if (tracked)
    gpuMarkDeviceChanged(np);
  if (wake->change_p0 && wake->enabled[CWAKE_WZ])
    gpu_do_match_energy(np, PoInput, 0);
  gpuRecordWallSeconds();
}

static long gpuTrackWakeBucket(long np0, WAKE *wakeData, double Po,
                               double factor,
                               const GPU_BUNCHED_WAKE_PLAN *bunchedPlan) {
  GPU_BEAM_SUM_DATA sums;
  GPU_WAKE_LONGITUDINAL_DATA data;
  double tmin, tmax, beamSpan, wakeSpan;
  long nb, binnedCount = 0, smoothingActive = 0, expectedParticles = 0;
  float milliseconds = 0;
  int status;
  double *smoothedItime = NULL;
#ifdef GPU_VERIFY
  double *gpuItime = NULL, *gpuVtime = NULL;
#endif
  char warningBuffer[1024];

  expectedParticles = gpuWakeTimeSums(np0, Po, bunchedPlan, &sums,
                                      "WAKE time-coordinate reduction kernel");
  if (expectedParticles <= 0)
    return 0;

  tmin = sums.min[6];
  tmax = sums.max[6];
  beamSpan = tmax - tmin;
  wakeSpan = wakeData->t[wakeData->wakePoints - 1] - wakeData->t[0];
  if (beamSpan > wakeSpan) {
    if (!wakeData->allowLongBeam) {
      fprintf(stderr, "Error: The beam is longer than the longitudinal wake function.\nThis may produce unphysical results.\n");
      fprintf(stderr, "The beam length is %le s, while the wake length is %le s\n",
              beamSpan, wakeSpan);
      exit(1);
    }
    snprintf(warningBuffer, sizeof(warningBuffer),
             "This may produce unphysical results. The beam length is %le s, while the wake length is %le s\n",
             beamSpan, wakeSpan);
    gpuWakeTrackingWarning("The beam is longer than the longitudinal wake function.",
                           warningBuffer);
  }

  if (expectedParticles > 1 && beamSpan < 20 * wakeData->dt)
    gpuWakeTrackingWarning("Beam shorter than 20 times the spacing of the wake points.",
                           "May produce poor results. Consider using finer time spacing in wake data.");

  if (wakeData->n_bins) {
    nb = wakeData->n_bins;
    tmin = sums.centroidSum[6] / sums.count - wakeData->dt * nb / 2.0;
  } else {
    nb = beamSpan / wakeData->dt + 3;
    tmin -= wakeData->dt;
  }
  if (nb <= 0) {
    gpuWakeTrackingWarning("Number of wake bins is 0 or negative.",
                           "Probably indicates an extremely long bunch. Wake ignored!");
    return -1;
  }

  memset(&data, 0, sizeof(data));
  data.bins = nb;
  data.wakePoints = wakeData->wakePoints;
  data.i0 = wakeData->i0;
  data.interpolate = (int)wakeData->interpolate;
  data.useBunchFilter = (int)bunchedPlan->useBunchFilter;
  data.bunchIndexColumn = bunchIndex;
  data.selectedBunch = bunchedPlan->selectedBunch;
  data.tmin = tmin;
  data.dt = wakeData->dt;
  data.pCentral = Po;
  data.factor = factor;
  data.particleMassMV = particleMassMV;
  data.particleRelSign = particleRelSign;
  data.cMks = c_mks;
  smoothingActive = gpuWakeSmoothingActive(wakeData->smoothing, nb,
                                           wakeData->SGHalfWidth);

#ifdef GPU_VERIFY
  gpuVtime = (double *)malloc(sizeof(*gpuVtime) * nb);
  if (!smoothingActive)
    gpuItime = (double *)malloc(sizeof(*gpuItime) * nb);
  if ((!smoothingActive && !gpuItime) || !gpuVtime)
    gpuRequiredFailure("unable to allocate CUDA WAKE verification result buffers");
#endif

  milliseconds = 0;
  if (smoothingActive) {
    long trackBinnedCount = 0;

    smoothedItime = (double *)malloc(sizeof(*smoothedItime) * nb);
    if (!smoothedItime)
      gpuRequiredFailure("unable to allocate CUDA WAKE smoothing histogram");
    status = gpuCudaWakeLongitudinalHistogram(gpuBase.deviceCoord, np0,
                                              (int)gpuBase.deviceStride,
                                              &data, &binnedCount,
                                              smoothedItime,
                                              &milliseconds);
    if (status != 0)
      gpuFatalStatus("WAKE longitudinal CUDA histogram kernel", status);
    gpuRecordWakeKernel(milliseconds);
    gpuSmoothWakeHistogram(smoothedItime, nb, wakeData->SGOrder,
                           wakeData->SGHalfWidth, "WAKE",
                           wakeData->inputFile);

    milliseconds = 0;
    status = gpuCudaWakeLongitudinalTrackFromHistogram(
      gpuBase.deviceCoord, np0, (int)gpuBase.deviceStride,
      &data, wakeData->W, smoothedItime, &trackBinnedCount,
#ifdef GPU_VERIFY
      gpuVtime,
#else
      NULL,
#endif
      &milliseconds);
    if (status != 0)
      gpuFatalStatus("WAKE longitudinal CUDA smoothed track kernel", status);
    if (trackBinnedCount != binnedCount)
      gpuRequiredFailure("WAKE smoothed CUDA rebin count changed before kicks");
  } else {
    status = gpuCudaWakeLongitudinalTrack(gpuBase.deviceCoord, np0,
                                          (int)gpuBase.deviceStride,
                                          &data, wakeData->W,
                                          &binnedCount,
#ifdef GPU_VERIFY
                                          gpuItime, gpuVtime,
#else
                                          NULL, NULL,
#endif
                                          &milliseconds);
    if (status != 0)
      gpuFatalStatus("WAKE longitudinal CUDA kernel", status);
  }
  if (status != 0)
    gpuFatalStatus("WAKE longitudinal CUDA kernel", status);
  gpuRecordWakeKernel(milliseconds);

  if (binnedCount != expectedParticles) {
    snprintf(warningBuffer, sizeof(warningBuffer),
             "Only %ld of %ld particles were binned. Consider setting n_bins=0 to invoke autoscaling.",
             binnedCount, expectedParticles);
    gpuWakeTrackingWarning("Some particles not binned in WAKE.", warningBuffer);
  }

#ifdef GPU_VERIFY
  if (!gpuWakeVerificationSuppressed && !bunchedPlan->useBunchFilter)
    gpuVerifyWakeLongitudinal(wakeData, np0, Po, tmin, wakeData->dt, nb,
                              factor, binnedCount,
                              smoothingActive ? smoothedItime : gpuItime,
                              gpuVtime);
  free(gpuItime);
  free(gpuVtime);
#endif
  free(smoothedItime);
  return 1;
}

void gpu_track_through_wake(long np0, void *wakeData0, double *PoInput,
                            void *run0, long i_pass, void *charge0) {
  WAKE *wakeData = (WAKE *)wakeData0;
  RUN *run = (RUN *)run0;
  CHARGE *charge = (CHARGE *)charge0;
  GPU_BUNCHED_WAKE_PLAN bunchedPlan, bucketPlan;
  double Po, rampFactor, factor;
  long relativeBunch, firstRelativeBunch, lastRelativeBunch;
  long trackedBuckets = 0, bucketStatus;

  if (np0 <= 0)
    return;
  if (!wakeData || !PoInput)
    gpuRequiredFailure("NULL pointer in gpu_track_through_wake");
  if (!gpuWakeDataSupported(wakeData))
    gpuRequiredFailure("unsupported WAKE options reached CUDA path");
  if (gpuBunchedWakePlan(np0, wakeData->bunchedBeamMode,
                         wakeData->startBunch, wakeData->endBunch,
                         charge, "WAKE", &bunchedPlan) ==
      GPU_BUNCHED_WAKE_UNSUPPORTED)
    gpuRequiredFailure("unsupported WAKE bunched-beam layout reached CUDA path");
  if (bunchedPlan.action == GPU_BUNCHED_WAKE_SKIP) {
    if (wakeData->change_p0)
      gpu_do_match_energy(np0, PoInput, 0);
    gpuRecordWallSeconds();
    return;
  }
  if (!set_up_wake)
    gpuRequiredFailure("WAKE setup routine is unavailable");

  set_up_wake(wakeData, run, i_pass, np0, charge);
  if (!wakeData->W || !wakeData->t || wakeData->wakePoints < 2 || wakeData->dt <= 0)
    gpuRequiredFailure("WAKE data was not initialized for CUDA tracking");

  Po = *PoInput;
  gpuCopyHostToDevice(np0);
  if (i_pass >= (wakeData->rampPasses - 1))
    rampFactor = 1;
  else
    rampFactor = (i_pass + 1.0) / wakeData->rampPasses;
  factor = wakeData->macroParticleCharge * particleRelSign *
           wakeData->factor * rampFactor;

  firstRelativeBunch = bunchedPlan.useBunchFilter ?
    bunchedPlan.firstRelativeBunch : 0;
  lastRelativeBunch = bunchedPlan.useBunchFilter ?
    bunchedPlan.lastRelativeBunch : 0;
  for (relativeBunch = firstRelativeBunch;
       relativeBunch <= lastRelativeBunch; relativeBunch++) {
    bucketPlan = bunchedPlan;
    if (bucketPlan.useBunchFilter) {
      bucketPlan.selectedRelativeBunch = relativeBunch;
      bucketPlan.selectedBunch = bucketPlan.minBunch + relativeBunch;
    }
    bucketStatus = gpuTrackWakeBucket(np0, wakeData, Po, factor,
                                      &bucketPlan);
    if (bucketStatus < 0) {
      gpuRecordWallSeconds();
      return;
    }
    if (bucketStatus > 0)
      trackedBuckets++;
  }

  if (trackedBuckets)
    gpuMarkDeviceChanged(np0);
  if (wakeData->change_p0)
    gpu_do_match_energy(np0, PoInput, 0);
  gpuRecordWallSeconds();
}

#ifdef GPU_VERIFY
static void gpuVerifyTrwake(TRWAKE *wakeData, long np, double Po,
                            double tmin, double dt, long nb,
                            const double factor[2], long gpuBinned,
                            const double *gpuPosItimeX,
                            const double *gpuPosItimeY,
                            const double *gpuVtimeX,
                            const double *gpuVtimeY) {
  double *cpuTime, *cpuPz, *cpuPosItime[2], *cpuVtime;
  double **cpuCoord = gpuBase.coord;
  long *cpuPbin;
  long cpuBinned, ib, ip, plane;

  if (!computeTimeCoordinatesOnly || !binTransverseTimeDistribution ||
      !convolveArrays)
    return;
  if (wakeData->tilt && !rotateBeamCoordinatesForMisalignment)
    gpuRequiredFailure("TRWAKE tilt verification needs rotateBeamCoordinatesForMisalignment");
  cpuTime = (double *)malloc(sizeof(*cpuTime) * np);
  cpuPz = (double *)malloc(sizeof(*cpuPz) * np);
  cpuPosItime[0] = (double *)malloc(sizeof(**cpuPosItime) * nb);
  cpuPosItime[1] = (double *)malloc(sizeof(**cpuPosItime) * nb);
  cpuVtime = (double *)malloc(sizeof(*cpuVtime) * nb);
  cpuPbin = (long *)malloc(sizeof(*cpuPbin) * np);
  if (!cpuTime || !cpuPz || !cpuPosItime[0] || !cpuPosItime[1] ||
      !cpuVtime || !cpuPbin)
    gpuRequiredFailure("unable to allocate CUDA TRWAKE verification buffers");
  if (wakeData->tilt) {
    cpuCoord = (double **)czarray_2d(sizeof(double), np,
                                     totalPropertiesPerParticle);
    if (!cpuCoord)
      gpuRequiredFailure("unable to allocate CUDA TRWAKE tilted verification copy");
    for (ip = 0; ip < np; ip++)
      memcpy(cpuCoord[ip], gpuBase.coord[ip],
             sizeof(double) * totalPropertiesPerParticle);
    rotateBeamCoordinatesForMisalignment(cpuCoord, np, wakeData->tilt);
  }

  computeTimeCoordinatesOnly(cpuTime, Po, gpuBase.coord, np);
  cpuBinned = binTransverseTimeDistribution(
    cpuPosItime, cpuPz, cpuPbin, tmin, dt, nb, cpuTime, cpuCoord, Po, np,
    wakeData->dx, wakeData->dy,
    wakeData->xDriveExponent, wakeData->yDriveExponent);
  if (cpuBinned != gpuBinned) {
    fprintf(stderr,
            "elegant CUDA VERIFY transverse wake bin-count mismatch track_through_trwake cpu=%ld gpu=%ld\n",
            cpuBinned, gpuBinned);
    exit(1);
  }

  for (plane = 0; plane < 2; plane++) {
    const double *gpuVtime = plane == 0 ? gpuVtimeX : gpuVtimeY;
    const double *gpuPosItime = plane == 0 ? gpuPosItimeX : gpuPosItimeY;
    const char *name = plane == 0 ? "VtimeX" : "VtimeY";
    const char *histName = plane == 0 ? "posItimeX" : "posItimeY";

    if (wakeData->W[plane] && factor[plane] &&
        gpuWakeSmoothingActive(wakeData->smoothing, nb,
                               wakeData->SGHalfWidth))
      gpuSmoothWakeHistogram(cpuPosItime[plane], nb, wakeData->SGOrder,
                             wakeData->SGHalfWidth, "TRWAKE",
                             wakeData->inputFile);
    gpuCompareWakeArray("track_through_trwake", histName,
                        cpuPosItime[plane], gpuPosItime, nb);
    if (!wakeData->W[plane] || !factor[plane])
      continue;
    convolveArrays(cpuVtime, nb, cpuPosItime[plane], nb,
                   wakeData->W[plane], wakeData->wakePoints, wakeData->i0);
    for (ib = 0; ib < nb; ib++)
      cpuVtime[ib] *= factor[plane];
    gpuCompareWakeArray("track_through_trwake", name, cpuVtime, gpuVtime, nb);
  }

  free(cpuTime);
  free(cpuPz);
  free(cpuPosItime[0]);
  free(cpuPosItime[1]);
  free(cpuVtime);
  free(cpuPbin);
  if (cpuCoord != gpuBase.coord)
    free_czarray_2d((void **)cpuCoord, np, totalPropertiesPerParticle);
}
#endif

static long gpuTrackTrwakeBucket(long np0, TRWAKE *wakeData, double Po,
                                 double rampFactor,
                                 const GPU_BUNCHED_WAKE_PLAN *bunchedPlan) {
  GPU_BEAM_SUM_DATA sums;
  GPU_TRWAKE_DATA data;
  double tmin, tmax, beamSpan, wakeSpan;
  double factor[2] = {0, 0};
  long nb, binnedCount = 0, smoothingActive = 0, expectedParticles = 0;
  float milliseconds = 0;
  int status;
  double *smoothedPosItimeX = NULL, *smoothedPosItimeY = NULL;
#ifdef GPU_VERIFY
  double *gpuPosItimeX = NULL, *gpuPosItimeY = NULL;
  double *gpuVtimeX = NULL, *gpuVtimeY = NULL;
#endif
  char warningBuffer[1024];

  expectedParticles = gpuWakeTimeSums(np0, Po, bunchedPlan, &sums,
                                      "TRWAKE time-coordinate reduction kernel");
  if (expectedParticles <= 0)
    return 0;

  tmin = sums.min[6];
  tmax = sums.max[6];
  beamSpan = tmax - tmin;
  wakeSpan = wakeData->t[wakeData->wakePoints - 1] - wakeData->t[0];
  if (beamSpan > wakeSpan) {
    fprintf(stderr, "The beam is longer than the transverse wake function.\nThis would produce unphysical results.\n");
    fprintf(stderr, "The beam length is %le s, while the wake length is %le s\n",
            beamSpan, wakeSpan);
    exit(1);
  }

  if (wakeData->n_bins) {
    nb = wakeData->n_bins;
    if (bunchedPlan->useBunchFilter)
      tmin = sums.centroidSum[6] / sums.count - wakeData->dt * nb / 2.0;
    else
      tmin = -wakeData->dt * nb / 2.0;
    tmax = tmin + wakeData->dt * (nb - 1);
  } else {
    nb = beamSpan / wakeData->dt + 3;
    tmin -= wakeData->dt;
    tmax += wakeData->dt;
  }
  if (tmin > tmax || nb <= 0) {
    fprintf(stderr, "Problem with time coordinates in TRWAKE.  Po=%le\n", Po);
    exit(1);
  }

  factor[0] = wakeData->macroParticleCharge * particleRelSign *
              wakeData->factor * wakeData->xfactor * rampFactor;
  factor[1] = wakeData->macroParticleCharge * particleRelSign *
              wakeData->factor * wakeData->yfactor * rampFactor;

  memset(&data, 0, sizeof(data));
  data.bins = nb;
  data.wakePoints = wakeData->wakePoints;
  data.i0 = wakeData->i0;
  data.interpolate = (int)wakeData->interpolate;
  data.useBunchFilter = (int)bunchedPlan->useBunchFilter;
  data.bunchIndexColumn = bunchIndex;
  data.selectedBunch = bunchedPlan->selectedBunch;
  data.hasWake[0] = wakeData->W[0] ? 1 : 0;
  data.hasWake[1] = wakeData->W[1] ? 1 : 0;
  data.driveExponent[0] = wakeData->xDriveExponent;
  data.driveExponent[1] = wakeData->yDriveExponent;
  data.probeExponent[0] = wakeData->xProbeExponent;
  data.probeExponent[1] = wakeData->yProbeExponent;
  data.tmin = tmin;
  data.dt = wakeData->dt;
  data.pCentral = Po;
  data.factor[0] = factor[0];
  data.factor[1] = factor[1];
  data.offset[0] = wakeData->dx;
  data.offset[1] = wakeData->dy;
  data.hasTilt = wakeData->tilt ? 1 : 0;
  gpuCsbendTiltSinCos(wakeData->tilt, &data.cosTilt, &data.sinTilt);
  data.particleMassMV = particleMassMV;
  data.particleRelSign = particleRelSign;
  data.cMks = c_mks;
  smoothingActive = gpuWakeSmoothingActive(wakeData->smoothing, nb,
                                           wakeData->SGHalfWidth);

#ifdef GPU_VERIFY
  if (!smoothingActive) {
    gpuPosItimeX = (double *)malloc(sizeof(*gpuPosItimeX) * nb);
    gpuPosItimeY = (double *)malloc(sizeof(*gpuPosItimeY) * nb);
  }
  gpuVtimeX = (double *)malloc(sizeof(*gpuVtimeX) * nb);
  gpuVtimeY = (double *)malloc(sizeof(*gpuVtimeY) * nb);
  if ((!smoothingActive && (!gpuPosItimeX || !gpuPosItimeY)) ||
      !gpuVtimeX || !gpuVtimeY)
    gpuRequiredFailure("unable to allocate CUDA TRWAKE verification result buffers");
#endif

  milliseconds = 0;
  if (smoothingActive) {
    long trackBinnedCount = 0;

    smoothedPosItimeX = (double *)malloc(sizeof(*smoothedPosItimeX) * nb);
    smoothedPosItimeY = (double *)malloc(sizeof(*smoothedPosItimeY) * nb);
    if (!smoothedPosItimeX || !smoothedPosItimeY)
      gpuRequiredFailure("unable to allocate CUDA TRWAKE smoothing histograms");
    status = gpuCudaTrwakeHistogram(gpuBase.deviceCoord, np0,
                                    (int)gpuBase.deviceStride, &data,
                                    &binnedCount,
                                    smoothedPosItimeX,
                                    smoothedPosItimeY,
                                    &milliseconds);
    if (status != 0)
      gpuFatalStatus("TRWAKE CUDA histogram kernel", status);
    gpuRecordWakeKernel(milliseconds);
    if (data.hasWake[0] && factor[0])
      gpuSmoothWakeHistogram(smoothedPosItimeX, nb, wakeData->SGOrder,
                             wakeData->SGHalfWidth, "TRWAKE",
                             wakeData->inputFile);
    if (data.hasWake[1] && factor[1])
      gpuSmoothWakeHistogram(smoothedPosItimeY, nb, wakeData->SGOrder,
                             wakeData->SGHalfWidth, "TRWAKE",
                             wakeData->inputFile);

    milliseconds = 0;
    status = gpuCudaTrwakeTrackFromHistogram(
      gpuBase.deviceCoord, np0, (int)gpuBase.deviceStride, &data,
      wakeData->W[0], wakeData->W[1],
      smoothedPosItimeX, smoothedPosItimeY,
      &trackBinnedCount,
#ifdef GPU_VERIFY
      gpuVtimeX, gpuVtimeY,
#else
      NULL, NULL,
#endif
      &milliseconds);
    if (status != 0)
      gpuFatalStatus("TRWAKE CUDA smoothed track kernel", status);
    if (trackBinnedCount != binnedCount)
      gpuRequiredFailure("TRWAKE smoothed CUDA rebin count changed before kicks");
  } else {
    status = gpuCudaTrwakeTrack(gpuBase.deviceCoord, np0,
                                (int)gpuBase.deviceStride, &data,
                                wakeData->W[0], wakeData->W[1],
                                &binnedCount,
#ifdef GPU_VERIFY
                                gpuPosItimeX, gpuPosItimeY,
                                gpuVtimeX, gpuVtimeY,
#else
                                NULL, NULL, NULL, NULL,
#endif
                                &milliseconds);
    if (status != 0)
      gpuFatalStatus("TRWAKE CUDA kernel", status);
  }
  if (status != 0)
    gpuFatalStatus("TRWAKE CUDA kernel", status);
  gpuRecordWakeKernel(milliseconds);

  if (binnedCount != expectedParticles) {
    snprintf(warningBuffer, sizeof(warningBuffer),
             "Only %ld of %ld particles were binned. Consider setting N_BINS=0 to invoke autoscaling.",
             binnedCount, expectedParticles);
    gpuWakeTrackingWarning("Some particles not binned in TRWAKE.",
                           warningBuffer);
  }

#ifdef GPU_VERIFY
  if (!gpuWakeVerificationSuppressed && !bunchedPlan->useBunchFilter)
    gpuVerifyTrwake(wakeData, np0, Po, tmin, wakeData->dt, nb, factor,
                    binnedCount,
                    smoothingActive ? smoothedPosItimeX : gpuPosItimeX,
                    smoothingActive ? smoothedPosItimeY : gpuPosItimeY,
                    gpuVtimeX, gpuVtimeY);
  free(gpuPosItimeX);
  free(gpuPosItimeY);
  free(gpuVtimeX);
  free(gpuVtimeY);
#endif
  free(smoothedPosItimeX);
  free(smoothedPosItimeY);
  return 1;
}

void gpu_track_through_trwake(long np0, void *wakeData0, double Po,
                              void *run0, long i_pass, void *charge0) {
  TRWAKE *wakeData = (TRWAKE *)wakeData0;
  RUN *run = (RUN *)run0;
  CHARGE *charge = (CHARGE *)charge0;
  GPU_BUNCHED_WAKE_PLAN bunchedPlan, bucketPlan;
  double rampFactor;
  long relativeBunch, firstRelativeBunch, lastRelativeBunch;
  long trackedBuckets = 0, bucketStatus;

  if (np0 <= 0)
    return;
  if (!wakeData)
    gpuRequiredFailure("NULL pointer in gpu_track_through_trwake");
  if (!gpuTrwakeDataSupported(wakeData))
    gpuRequiredFailure("unsupported TRWAKE options reached CUDA path");
  if (gpuBunchedWakePlan(np0, wakeData->bunchedBeamMode,
                         wakeData->startBunch, wakeData->endBunch,
                         charge, "TRWAKE", &bunchedPlan) ==
      GPU_BUNCHED_WAKE_UNSUPPORTED)
    gpuRequiredFailure("unsupported TRWAKE bunched-beam layout reached CUDA path");
  if (bunchedPlan.action == GPU_BUNCHED_WAKE_SKIP)
    return;
  if (!set_up_trwake)
    gpuRequiredFailure("TRWAKE setup routine is unavailable");

  set_up_trwake(wakeData, run, i_pass, np0, charge);
  if ((!wakeData->W[0] && !wakeData->W[1]) || !wakeData->t ||
      wakeData->wakePoints < 2 || wakeData->dt <= 0)
    gpuRequiredFailure("TRWAKE data was not initialized for CUDA tracking");

  if (i_pass >= (wakeData->rampPasses - 1))
    rampFactor = 1;
  else
    rampFactor = (i_pass + 1.0) / wakeData->rampPasses;

  gpuCopyHostToDevice(np0);
  firstRelativeBunch = bunchedPlan.useBunchFilter ?
    bunchedPlan.firstRelativeBunch : 0;
  lastRelativeBunch = bunchedPlan.useBunchFilter ?
    bunchedPlan.lastRelativeBunch : 0;
  for (relativeBunch = firstRelativeBunch;
       relativeBunch <= lastRelativeBunch; relativeBunch++) {
    bucketPlan = bunchedPlan;
    if (bucketPlan.useBunchFilter) {
      bucketPlan.selectedRelativeBunch = relativeBunch;
      bucketPlan.selectedBunch = bucketPlan.minBunch + relativeBunch;
    }
    bucketStatus = gpuTrackTrwakeBucket(np0, wakeData, Po, rampFactor,
                                        &bucketPlan);
    if (bucketStatus > 0)
      trackedBuckets++;
  }

  if (trackedBuckets)
    gpuMarkDeviceChanged(np0);
  gpuRecordWallSeconds();
}

static long gpuRfcaOnCpu(long np, RFCA *rfca, double **accepted,
                         double *P_central, double zEnd, long iPass,
                         RUN *run, CHARGE *charge, WAKE *wake,
                         TRWAKE *trwake, LSCKICK *LSCKick,
                         long wakesAtEnd, const char *reason) {
  double **coord = forceParticlesToCpu(reason);
  long remaining;

  if (!trackRfCavityWithWakes)
    gpuRequiredFailure("CPU RFCA fallback is unavailable");
  gpuBase.elementOnGpu = 0;
  remaining = trackRfCavityWithWakes(coord, np, rfca, accepted, P_central,
                                     zEnd, iPass, run, charge, wake, trwake,
                                     LSCKick, wakesAtEnd);
  gpuMarkHostWillChange();
  gpuRecordWallSeconds();
  return remaining;
}

double gpu_findFiducialTime(long np, double s0, double sOffset,
                            double p0, unsigned long mode);

static long gpuRfcaThinKickOnDevice(long np, RFCA *rfca, double **accepted,
                                    double *P_central, double zEnd,
                                    double phase) {
  double omega, volt;
  float milliseconds = 0;
  int status;
  long remaining;

  if (np <= 0)
    return np;
  if (!rfca || !P_central)
    gpuRequiredFailure("NULL RFCA thin-kick input in CUDA path");

  omega = PIx2 * rfca->freq;
  volt = rfca->volt / (1e6 * particleMassMV * particleRelSign);
  startGpuTimer();
  gpuCopyHostToDevice(np);
  status = gpuCudaRfcaThinKick(gpuBase.deviceCoord, np,
                               (int)gpuBase.deviceStride, *P_central,
                               volt, omega, phase, c_mks, &milliseconds);
  if (status != 0)
    gpuFatalStatus("RFCA thin kick kernel", status);
  gpuRecordHelperKernel(milliseconds);
  gpuMarkDeviceChanged(np);
  gpuRecordWallSeconds();
  remaining = gpu_removeInvalidParticles(np, accepted, zEnd, *P_central);
  if (rfca->change_p0)
    gpu_do_match_energy(remaining, P_central, 0);
  return remaining;
}

static long gpuRfcaRfOnlyMatrixOnDevice(long np, RFCA *rfca,
                                        double **accepted,
                                        double *P_central, double zEnd,
                                        double phase) {
  double omega, volt;
  float milliseconds = 0;
  int status;
  long lostCount = 0, remaining;

  if (np <= 0)
    return np;
  if (!rfca || !P_central)
    gpuRequiredFailure("NULL RFCA RF-only matrix input in CUDA path");

  omega = PIx2 * rfca->freq;
  volt = rfca->volt / (1e6 * particleMassMV * particleRelSign);
  startGpuTimer();
  gpuCopyHostToDevice(np);
  gpuEnsureRfcaScratch();
  status = gpuCudaRfcwRfOnlyMatrixChecked(
    gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, *P_central,
    rfca->length, volt, omega, phase, (int)rfca->end1Focus,
    (int)rfca->end2Focus, rfca->dx, rfca->dy, c_mks,
    gpuRfcaScratch.lostCount, &lostCount, &milliseconds);
  if (status != 0)
    gpuFatalStatus("RFCA RF-only matrix kernel", status);
  gpuRecordHelperKernel(milliseconds);
  gpuMarkDeviceChanged(np);
  gpuRecordWallSeconds();
  remaining = lostCount ?
                gpu_removeInvalidParticles(np, accepted, zEnd, *P_central) :
                np;
  if (rfca->change_p0)
    gpu_do_match_energy(remaining, P_central, 0);
  return remaining;
}

static long gpuRfcaRfOnlyKickOnDevice(long np, RFCA *rfca,
                                      double **accepted,
                                      double *P_central, double zEnd,
                                      double phase) {
  long ik, nKicks, remaining;
  double omega, volt, sectionLength, sectionVolt, dtLight;
  float milliseconds = 0;
  int status;

  if (np <= 0)
    return np;
  if (!rfca || !P_central)
    gpuRequiredFailure("NULL RFCA RF-only kick input in CUDA path");

  gpuEnsureRfcwKickScratch(np);

  nKicks = rfca->nKicks > 0 ? rfca->nKicks : 1;
  sectionLength = rfca->length / nKicks;
  omega = PIx2 * rfca->freq;
  volt = rfca->volt / (1e6 * particleMassMV * particleRelSign);
  sectionVolt = volt / nKicks;
  dtLight = sectionLength / c_mks;

  gpuRfcwApplyCoordinateOffset(np, 0, rfca->dx,
                               "RFCA CUDA local x-offset kernel");
  gpuRfcwApplyCoordinateOffset(np, 2, rfca->dy,
                               "RFCA CUDA local y-offset kernel");

  for (ik = 0; ik < nKicks; ik++) {
    double sectionPhase = rfca->standingWave ? phase : phase - ik * omega * dtLight;

    startGpuTimer();
    gpuCopyHostToDevice(np);
    status = gpuCudaRfcwKickInitial(gpuBase.deviceCoord,
                                    gpuRfcwKickScratch.inverseF, np,
                                    (int)gpuBase.deviceStride, *P_central,
                                    sectionLength, sectionVolt, omega,
                                    sectionPhase,
                                    (int)(rfca->end1Focus && ik == 0),
                                    c_mks, &milliseconds);
    if (status != 0)
      gpuFatalStatus("RFCA RF-only kick initial kernel", status);
    gpuRecordHelperKernel(milliseconds);
    gpuMarkDeviceChanged(np);
    gpuRecordWallSeconds();

    startGpuTimer();
    gpuCopyHostToDevice(np);
    status = gpuCudaRfcwKickFinal(gpuBase.deviceCoord,
                                  gpuRfcwKickScratch.inverseF, np,
                                  (int)gpuBase.deviceStride, sectionLength,
                                  (int)(rfca->end2Focus && ik == nKicks - 1),
                                  &milliseconds);
    if (status != 0)
      gpuFatalStatus("RFCA RF-only kick final kernel", status);
    gpuRecordHelperKernel(milliseconds);
    gpuMarkDeviceChanged(np);
    gpuRecordWallSeconds();
  }

  gpuRfcwApplyCoordinateOffset(np, 0, -rfca->dx,
                               "RFCA CUDA restore x-offset kernel");
  gpuRfcwApplyCoordinateOffset(np, 2, -rfca->dy,
                               "RFCA CUDA restore y-offset kernel");

  remaining = gpu_removeInvalidParticles(np, accepted, zEnd, *P_central);
  if (rfca->change_p0)
    gpu_do_match_energy(remaining, P_central, 0);
  return remaining;
}

static long gpuRfcaSetupPhaseOnDeviceWithOffset(RFCA *rfca, long np,
                                                double *P_central, double zEnd,
                                                double sOffset,
                                                double *phaseOut) {
  double phase = 0, t0 = 0, omega;
  long phaseStatus;

  if (!rfca || !P_central || !phaseOut)
    return 0;
  if (!get_phase_reference || !set_phase_reference || !unused_phase_reference)
    return 0;
  if (rfca->phase_reference == 0)
    rfca->phase_reference = unused_phase_reference();
  phaseStatus = get_phase_reference(&phase, rfca->phase_reference);
  switch (phaseStatus) {
  case REF_PHASE_RETURNED:
    break;
  case REF_PHASE_NOT_SET:
  case REF_PHASE_NONEXISTENT:
    if (!rfca->fiducial_seen) {
      unsigned long mode;

      mode = gpuRfcwFiducialMode(rfca->fiducial);
      if (!gpuFiducialModeSupported(mode))
        return 0;
      omega = PIx2 * rfca->freq;
      if (rfca->tReference != -1)
        t0 = rfca->tReference;
      else if (mode & FID_MODE_LIGHT)
        t0 = (zEnd - rfca->length + sOffset) / c_mks;
      else if (mode & (FID_MODE_TMEAN | FID_MODE_FIRST | FID_MODE_PMAX))
        t0 = gpu_findFiducialTime(np, zEnd - rfca->length,
                                  sOffset, *P_central, mode);
      else
        return 0;
      rfca->phase_fiducial = -omega * t0;
      rfca->fiducial_seen = 1;
    }
    set_phase_reference(rfca->phase_reference,
                        phase = rfca->phase_fiducial);
    break;
  default:
    return 0;
  }

  if (rfca->freq)
    rfca->t_fiducial = -rfca->phase_fiducial / (PIx2 * rfca->freq);
  else
    rfca->t_fiducial = 0;
  *phaseOut = phase + rfca->phase * PI / 180.0;
  return 1;
}

static long gpuRfcaSetupPhaseOnDevice(RFCA *rfca, long np,
                                      double *P_central, double zEnd,
                                      double *phaseOut) {
  if (!rfca)
    return 0;
  return gpuRfcaSetupPhaseOnDeviceWithOffset(rfca, np, P_central, zEnd,
                                             rfca->length / 2, phaseOut);
}

static void gpuRefreshRfcwRfca(RFCW *rfcw) {
  RFCA *rfca;

  if (!rfcw)
    gpuRequiredFailure("NULL RFCW input in CUDA refresh");
  rfca = &rfcw->rfca;
  rfca->length = rfcw->length;
  rfca->volt = rfcw->volt;
  rfca->phase = rfcw->phase;
  rfca->freq = rfcw->freq;
  rfca->Q = rfcw->Q;
  rfca->change_p0 = rfcw->change_p0;
  rfca->change_t = rfcw->change_t;
  rfca->end1Focus = rfcw->end1Focus;
  rfca->end2Focus = rfcw->end2Focus;
  rfca->standingWave = rfcw->standingWave;
  rfca->bodyFocusModel = rfcw->bodyFocusModel;
  rfca->nKicks = rfcw->length ? rfcw->nKicks : 1;
  rfca->dx = rfcw->dx;
  rfca->dy = rfcw->dy;
  rfca->tReference = rfcw->tReference;
  rfca->linearize = rfcw->linearize;
  rfca->lockPhase = 0;
  rfca->backtrack = rfcw->backtrack;
  if (!rfcw->initialized) {
    rfca->phase_reference = rfcw->phase_reference;
    rfca->fiducial = rfcw->fiducial;
    rfca->fiducial_seen = 0;
  } else if (rfca->phase_reference == 0)
    rfca->phase_reference = rfcw->phase_reference;
}

static void gpuPrepareRfcwWakeData(RFCW *rfcw) {
  if (!rfcw)
    gpuRequiredFailure("NULL RFCW input in CUDA wake refresh");
  if (!rfcw->initialized && gpuRfcwStringPresent(rfcw->wakeFile)) {
    if (gpuRfcwStringPresent(rfcw->trWakeFile) ||
        gpuRfcwStringPresent(rfcw->zWakeFile))
      gpuRequiredFailure("RFCW WAKEFILE conflicts with TRWAKEFILE/ZWAKEFILE in CUDA path");
    SDDS_CopyString(&rfcw->trWakeFile, rfcw->wakeFile);
    SDDS_CopyString(&rfcw->zWakeFile, rfcw->wakeFile);
  }

  rfcw->trwake.charge = 0;
  rfcw->trwake.bunchedBeamMode = 1;
  rfcw->trwake.startBunch = rfcw->trwake.endBunch = -1;
  rfcw->trwake.xfactor = rfcw->trwake.yfactor = rfcw->trwake.factor = 1;
  rfcw->trwake.n_bins = rfcw->n_bins;
  rfcw->trwake.interpolate = rfcw->interpolate;
  rfcw->trwake.smoothing = rfcw->smoothing;
  rfcw->trwake.SGHalfWidth = rfcw->SGHalfWidth;
  rfcw->trwake.SGOrder = rfcw->SGOrder;
  rfcw->trwake.dx = 0;
  rfcw->trwake.dy = 0;
  rfcw->trwake.acausalAllowed = rfcw->trwake.i0 = 0;
  rfcw->trwake.xDriveExponent = rfcw->trwake.yDriveExponent = 1;
  rfcw->trwake.xProbeExponent = rfcw->trwake.yProbeExponent = 0;
  if (!rfcw->initialized && rfcw->includeTrWake) {
    rfcw->trwake.initialized = 0;
    if (gpuRfcwStringPresent(rfcw->WxColumn) ||
        gpuRfcwStringPresent(rfcw->WyColumn)) {
      if (!gpuRfcwStringPresent(rfcw->trWakeFile))
        gpuRequiredFailure("RFCW CUDA transverse wake has no input file");
      SDDS_CopyString(&rfcw->trwake.inputFile, rfcw->trWakeFile);
      if (!gpuRfcwStringPresent(rfcw->tColumn))
        gpuRequiredFailure("RFCW CUDA transverse wake has no time column");
      SDDS_CopyString(&rfcw->trwake.tColumn, rfcw->tColumn);
      if (gpuRfcwStringPresent(rfcw->WxColumn))
        SDDS_CopyString(&rfcw->trwake.WxColumn, rfcw->WxColumn);
      if (gpuRfcwStringPresent(rfcw->WyColumn))
        SDDS_CopyString(&rfcw->trwake.WyColumn, rfcw->WyColumn);
    }
  }

  rfcw->wake.charge = 0;
  rfcw->wake.bunchedBeamMode = 1;
  rfcw->wake.startBunch = rfcw->wake.endBunch = -1;
  rfcw->wake.n_bins = rfcw->n_bins;
  rfcw->wake.acausalAllowed = rfcw->wake.i0 = 0;
  rfcw->wake.interpolate = rfcw->interpolate;
  rfcw->wake.smoothing = rfcw->smoothing;
  rfcw->wake.SGHalfWidth = rfcw->SGHalfWidth;
  rfcw->wake.SGOrder = rfcw->SGOrder;
  rfcw->wake.change_p0 = rfcw->change_p0;
  rfcw->wake.factor = 1;
  if (!rfcw->initialized && rfcw->includeZWake) {
    if (gpuRfcwStringPresent(rfcw->WzColumn)) {
      if (!gpuRfcwStringPresent(rfcw->zWakeFile))
        gpuRequiredFailure("RFCW CUDA longitudinal wake has no input file");
      SDDS_CopyString(&rfcw->wake.inputFile, rfcw->zWakeFile);
      if (!gpuRfcwStringPresent(rfcw->tColumn))
        gpuRequiredFailure("RFCW CUDA longitudinal wake has no time column");
      SDDS_CopyString(&rfcw->wake.tColumn, rfcw->tColumn);
      SDDS_CopyString(&rfcw->wake.WColumn, rfcw->WzColumn);
      rfcw->wake.initialized = 0;
    }
  }

  rfcw->LSCKick.bins = rfcw->LSCBins;
  rfcw->LSCKick.interpolate = rfcw->LSCInterpolate;
  rfcw->LSCKick.lowFrequencyCutoff0 = rfcw->LSCLowFrequencyCutoff0;
  rfcw->LSCKick.lowFrequencyCutoff1 = rfcw->LSCLowFrequencyCutoff1;
  rfcw->LSCKick.highFrequencyCutoff0 = rfcw->LSCHighFrequencyCutoff0;
  rfcw->LSCKick.highFrequencyCutoff1 = rfcw->LSCHighFrequencyCutoff1;
  rfcw->LSCKick.radiusFactor = rfcw->LSCRadiusFactor;
  rfcw->LSCKick.backtrack = 0;

  if (gpuRfcwStringPresent(rfcw->WzColumn) && rfcw->includeZWake)
    rfcw->wake.factor =
      rfcw->length / rfcw->cellLength /
      (rfcw->rfca.nKicks ? rfcw->rfca.nKicks : 1);
  if ((gpuRfcwStringPresent(rfcw->WxColumn) ||
       gpuRfcwStringPresent(rfcw->WyColumn)) &&
      rfcw->includeTrWake)
    rfcw->trwake.factor =
      rfcw->length / rfcw->cellLength /
      (rfcw->rfca.nKicks ? rfcw->rfca.nKicks : 1);
}

static long gpuRfcwOnCpu(long np, RFCW *rfcw, double **accepted,
                         double *P_central, double zEnd,
                         RUN *run, long iPass, CHARGE *charge,
                         const char *reason) {
  double **coord = forceParticlesToCpu(reason);
  long remaining;

  if (!track_through_rfcw)
    gpuRequiredFailure("CPU RFCW fallback is unavailable");
  gpuBase.elementOnGpu = 0;
  remaining = track_through_rfcw(coord, np, rfcw, accepted, P_central, zEnd,
                                 run, iPass, charge);
  gpuMarkHostWillChange();
  gpuRecordWallSeconds();
  return remaining;
}

static void gpuRfcwApplyCoordinateOffset(long np, int index, double value,
                                         const char *operation) {
  float milliseconds = 0;
  int status;

  if (np <= 0 || value == 0)
    return;
  startGpuTimer();
  gpuCopyHostToDevice(np);
  status = gpuCudaSubtractCoordinate(gpuBase.deviceCoord, np,
                                     (int)gpuBase.deviceStride, index,
                                     value, &milliseconds);
  if (status != 0)
    gpuFatalStatus(operation, status);
  gpuRecordHelperKernel(milliseconds);
  gpuMarkDeviceChanged(np);
  gpuRecordWallSeconds();
}

static void gpuRfcwRfOnlyMatrixCoreOnDevice(long np, RFCW *rfcw,
                                            double *P_central,
                                            double phase,
                                            double dx, double dy) {
  double omega, volt;
  float milliseconds = 0;
  int status;

  if (np <= 0)
    return;
  if (!rfcw || !P_central)
    gpuRequiredFailure("NULL RFCW RF-only input in CUDA path");

  omega = PIx2 * rfcw->freq;
  volt = rfcw->volt / (1e6 * particleMassMV * particleRelSign);
  startGpuTimer();
  gpuCopyHostToDevice(np);
  status = gpuCudaRfcwRfOnlyMatrix(gpuBase.deviceCoord, np,
                                   (int)gpuBase.deviceStride, *P_central,
                                   rfcw->length, volt, omega, phase,
                                   (int)rfcw->end1Focus,
                                   (int)rfcw->end2Focus, dx, dy,
                                   c_mks,
                                   &milliseconds);
  if (status != 0)
    gpuFatalStatus("RFCW RF-only matrix kernel", status);
  gpuRecordHelperKernel(milliseconds);
  gpuMarkDeviceChanged(np);
  gpuRecordWallSeconds();
}

static double gpuRfcwDgammaOverGammaAveOnDevice(long np, RFCW *rfcw,
                                                double pCentral,
                                                double length, double volt,
                                                double phase) {
  GPU_BEAM_SUM_DATA sums;
  double omega;
  float milliseconds = 0;
  int status;

  if (np <= 0)
    return 0;
  if (!rfcw)
    gpuRequiredFailure("NULL RFCW input for CUDA LSC energy-change reduction");

  omega = PIx2 * rfcw->freq;
  memset(&sums, 0, sizeof(sums));
  startGpuTimer();
  gpuCopyHostToDevice(np);
  status = gpuCudaRfcwDgammaOverGammaSums(
    gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, pCentral,
    length, volt, omega, phase, c_mks, &sums, &milliseconds);
  if (status != 0)
    gpuFatalStatus("RFCW CUDA LSC energy-change reduction", status);
  gpuRecordReductionKernel(milliseconds);
  gpuRecordWallSeconds();
  return sums.count > 0 ? sums.centroidSum[6] / sums.count : 0;
}

static void gpuTrackRfcwLscKickOnDevice(long np, LSCKICK *lsc, double Po,
                                        CHARGE *charge, double lengthScale,
                                        double dgammaOverGamma) {
  static double *Itime = NULL;
  static double *Ifreq = NULL;
  static double *Vtime = NULL;
  static long maxBins = 0;
  GPU_BEAM_SUM_DATA sums;
  GPU_LSC_DATA data;
  double Imin, Imax, S11, S33, beamRadius, lengthLimit, kSC;
  double absLengthScale;
  long nb, binnedCount = 0;
  float milliseconds = 0;
  int status;
  char warningBuffer[1024];
#ifdef GPU_VERIFY
  double *cpuStorage = NULL, **cpuCoord = NULL, *gpuVerifyStorage = NULL;
  double *cpuTime = NULL, *cpuItime = NULL, *cpuIfreq = NULL, *cpuVtime = NULL;
  long *cpuPbin = NULL;
  long stride, ip, cpuBinned;
  unsigned long count;
#endif

  if (np <= 0)
    return;
  if (!lsc)
    return;
  if (!charge)
    bombElegant("No charge defined for LSC.  Insert a CHARGE element in the beamline.", NULL);

  nb = lsc->bins;
  if (nb < 2 || (nb % 2))
    gpuRequiredFailure("unsupported RFCW LSC bin count reached CUDA path");
  gpuEnsureLscScratch(nb);
  if (nb > maxBins) {
    maxBins = nb;
    Itime = trealloc(Itime, 2 * sizeof(*Itime) * (maxBins + 1));
    Ifreq = trealloc(Ifreq, 2 * sizeof(*Ifreq) * (maxBins + 1));
    Vtime = trealloc(Vtime, 2 * sizeof(*Vtime) * (maxBins + 1));
  }

  startGpuTimer();
  gpuCopyHostToDevice(np);
#ifdef GPU_VERIFY
  stride = gpuBase.deviceStride;
  count = (unsigned long)np * (unsigned long)stride;
  cpuStorage = (double *)malloc(count * sizeof(*cpuStorage));
  cpuCoord = (double **)malloc(np * sizeof(*cpuCoord));
  gpuVerifyStorage = (double *)malloc(count * sizeof(*gpuVerifyStorage));
  cpuTime = (double *)malloc(np * sizeof(*cpuTime));
  cpuPbin = (long *)malloc(np * sizeof(*cpuPbin));
  cpuItime = (double *)malloc(2 * (nb + 1) * sizeof(*cpuItime));
  cpuIfreq = (double *)malloc(2 * (nb + 1) * sizeof(*cpuIfreq));
  cpuVtime = (double *)malloc(2 * (nb + 1) * sizeof(*cpuVtime));
  if (!cpuStorage || !cpuCoord || !gpuVerifyStorage || !cpuTime ||
      !cpuPbin || !cpuItime || !cpuIfreq || !cpuVtime)
    gpuRequiredFailure("unable to allocate CUDA RFCW LSC verification buffers");
  status = gpuCudaCopyDeviceToHost(cpuStorage, gpuBase.deviceCoord, count,
                                   &milliseconds);
  if (status != 0)
    gpuFatalStatus("cudaMemcpy(RFCW LSC verify device to host)", status);
  gpuRecordMilliseconds(&gpuBase.gpuTransferToHostSeconds, milliseconds);
  for (ip = 0; ip < np; ip++)
    cpuCoord[ip] = cpuStorage + ip * stride;
#endif
  memset(&sums, 0, sizeof(sums));
  milliseconds = 0;
  status = gpuCudaLscStatistics(gpuBase.deviceCoord, np,
                                (int)gpuBase.deviceStride, Po, c_mks,
                                &sums, gpuLscScratch.result,
                                &milliseconds);
  if (status != 0)
    gpuFatalStatus("RFCW LSC time-coordinate reduction kernel", status);
  gpuRecordReductionKernel(milliseconds);

  memset(&data, 0, sizeof(data));
  data.bins = nb;
  data.interpolate = lsc->interpolate ? 1 : 0;
  data.doDrift = 0;
  data.backtrack = lsc->backtrack ? 1 : 0;
  data.tmin = sums.min[6];
  data.dt = (sums.max[6] - sums.min[6]) / (nb - 3);
  data.pCentral = Po;
  data.particleMassMV = particleMassMV;
  data.particleRelSign = particleRelSign;
  data.cMks = c_mks;
  if (data.dt <= 0)
    gpuRequiredFailure("non-positive RFCW LSC time-bin spacing in CUDA path");

  memset(Itime, 0, 2 * sizeof(*Itime) * (nb + 1));
  milliseconds = 0;
  status = gpuCudaLscBin(gpuBase.deviceCoord, np,
                         (int)gpuBase.deviceStride, &data,
                         &binnedCount, Itime, gpuLscScratch.itime,
                         gpuLscScratch.binnedCount, &milliseconds);
  if (status != 0)
    gpuFatalStatus("RFCW LSC CUDA binning kernel", status);
  gpuRecordLscKernel(milliseconds);

  if (binnedCount != np) {
    snprintf(warningBuffer, sizeof(warningBuffer),
             "Only %ld of %ld particles were binned. This shouldn't happen.",
             binnedCount, np);
    gpuWakeTrackingWarning("Some particles were not binned in RFCW LSCKICK.",
                           warningBuffer);
  }

  find_min_max(&Imin, &Imax, Itime, nb);
  Imax *= charge->macroParticleCharge / data.dt;
  gpuLscCenteredVariances(&sums, np, Po, &S11, &S33);
  beamRadius = (sqrt(S11) + sqrt(S33)) / 2 * lsc->radiusFactor;
  if (beamRadius == 0) {
    fprintf(stderr, "Error: beam radius is zero in CUDA RFCW LSCKICK: S11=%le, S33=%le, RADIUS_FACTOR=%le\n",
            S11, S33, lsc->radiusFactor);
    exit(1);
  }

  absLengthScale = fabs(lengthScale);
  if (Imax > 0) {
    kSC = 2 / beamRadius * sqrt(Imax / ipow3(Po) / 17045.0);
    if (kSC > 0 && isfinite(kSC)) {
      lengthLimit = 1 / kSC;
      if (dgammaOverGamma) {
        double energyLengthLimit = fabs(absLengthScale / dgammaOverGamma);
        if (energyLengthLimit < lengthLimit)
          lengthLimit = energyLengthLimit;
      }
      lengthLimit /= 10;
      if (lengthLimit < absLengthScale) {
        fprintf(stderr, "Error: distance between LSC kicks for RFCW CUDA path is too large.\n");
        fprintf(stderr, "Suggest reducing distance between kicks by factor %e\n",
                absLengthScale / lengthLimit);
        exit(1);
      }
    }
  }

  gpuLscComputeVoltage(Vtime, Ifreq, Itime, nb, data.dt, absLengthScale,
                       beamRadius, charge->macroParticleCharge,
                       data.backtrack, Po,
                       lsc->lowFrequencyCutoff0,
                       lsc->lowFrequencyCutoff1,
                       lsc->highFrequencyCutoff0,
                       lsc->highFrequencyCutoff1);

#ifdef GPU_VERIFY
  if (gpuEnvFlag("ELEGANT_GPU_LSC_VERIFY_COMPONENTS")) {
    double cpuTmin, cpuTmax, cpuDt;
    double cpuImin, cpuImax, cpuS11, cpuS33, cpuBeamRadius;
    double itimeMaxAbs, vtimeMaxAbs;
    long itimeIndex, vtimeIndex, itimeMismatches, vtimeMismatches;

    if (!computeTimeCoordinatesOnly || !binTimeDistribution || !rms_emittance)
      gpuRequiredFailure("RFCW LSC component verification needs CPU helpers");
    memset(cpuItime, 0, 2 * sizeof(*cpuItime) * (nb + 1));
    computeTimeCoordinatesOnly(cpuTime, Po, cpuCoord, np);
    find_min_max(&cpuTmin, &cpuTmax, cpuTime, np);
    cpuDt = (cpuTmax - cpuTmin) / (nb - 3);
    cpuBinned = binTimeDistribution(cpuItime, cpuPbin, cpuTmin, cpuDt,
                                    nb, cpuTime, cpuCoord, Po, np);
    find_min_max(&cpuImin, &cpuImax, cpuItime, nb);
    cpuImax *= charge->macroParticleCharge / cpuDt;
    rms_emittance(cpuCoord, 0, 2, np, &cpuS11, NULL, &cpuS33, NULL, NULL);
    cpuBeamRadius = (sqrt(cpuS11) + sqrt(cpuS33)) / 2 * lsc->radiusFactor;
    gpuLscComputeVoltage(cpuVtime, cpuIfreq, cpuItime, nb, cpuDt,
                         absLengthScale, cpuBeamRadius,
                         charge->macroParticleCharge, data.backtrack, Po,
                         lsc->lowFrequencyCutoff0,
                         lsc->lowFrequencyCutoff1,
                         lsc->highFrequencyCutoff0,
                         lsc->highFrequencyCutoff1);
    itimeMaxAbs = gpuLscArrayMaxAbsDiff(cpuItime, Itime, nb, &itimeIndex,
                                        &itimeMismatches);
    vtimeMaxAbs = gpuLscArrayMaxAbsDiff(cpuVtime, Vtime, nb + 1, &vtimeIndex,
                                        &vtimeMismatches);
    fprintf(stderr,
            "elegant CUDA VERIFY LSC components: np=%ld nb=%ld cpuBinned=%ld gpuBinned=%ld\n",
            np, nb, cpuBinned, binnedCount);
    fprintf(stderr,
            "  tmin cpu=%.17e gpu=%.17e abs=%.3e  dt cpu=%.17e gpu=%.17e abs=%.3e\n",
            cpuTmin, data.tmin, fabs(cpuTmin - data.tmin),
            cpuDt, data.dt, fabs(cpuDt - data.dt));
    fprintf(stderr,
            "  S11 cpu=%.17e gpu=%.17e abs=%.3e  S33 cpu=%.17e gpu=%.17e abs=%.3e\n",
            cpuS11, S11, fabs(cpuS11 - S11),
            cpuS33, S33, fabs(cpuS33 - S33));
    fprintf(stderr,
            "  beamRadius cpu=%.17e gpu=%.17e abs=%.3e  Imax cpu=%.17e gpu=%.17e abs=%.3e\n",
            cpuBeamRadius, beamRadius, fabs(cpuBeamRadius - beamRadius),
            cpuImax, Imax, fabs(cpuImax - Imax));
    fprintf(stderr,
            "  Itime mismatches=%ld maxAbs=%.3e at %ld  Vtime mismatches=%ld maxAbs=%.3e at %ld\n",
            itimeMismatches, itimeMaxAbs, itimeIndex,
            vtimeMismatches, vtimeMaxAbs, vtimeIndex);
    if (vtimeIndex >= 0)
      fprintf(stderr,
              "  Vtime[%ld] cpu=%.17e gpu=%.17e\n",
              vtimeIndex, cpuVtime[vtimeIndex], Vtime[vtimeIndex]);
    if (gpuEnvFlag("ELEGANT_GPU_LSC_VERIFY_USE_CPU_VTIME")) {
      data.tmin = cpuTmin;
      data.dt = cpuDt;
      memcpy(Vtime, cpuVtime, 2 * (nb + 1) * sizeof(*Vtime));
      fprintf(stderr,
              "  using CPU LSC tmin/dt/Vtime for CUDA kick verification\n");
    }
  }
  if (!addLSCKick)
    gpuRequiredFailure("RFCW LSC verification needs addLSCKick");
  addLSCKick(cpuCoord, np, lsc, Po, charge, lengthScale, dgammaOverGamma);
#endif

  data.length = 0;
  milliseconds = 0;
  status = gpuCudaLscApplyKickAndDrift(gpuBase.deviceCoord, np,
                                       (int)gpuBase.deviceStride,
                                       &data, Vtime, gpuLscScratch.vtime,
                                       &milliseconds);
  if (status != 0)
    gpuFatalStatus("RFCW LSC CUDA kick kernel", status);
  gpuRecordLscKernel(milliseconds);
#ifdef GPU_VERIFY
  gpuLscCompareCpuShadow(cpuCoord, gpuVerifyStorage, np, stride,
                         "track_through_rfcw_lsc");
  free(cpuStorage);
  free(cpuCoord);
  free(gpuVerifyStorage);
  free(cpuTime);
  free(cpuPbin);
  free(cpuItime);
  free(cpuIfreq);
  free(cpuVtime);
#endif
  gpuMarkDeviceChanged(np);
  gpuRecordWallSeconds();
}

static long gpuRfcwLscKickOnlyAllowed(double **part, long np,
                                      LSCKICK *lsc) {
  ELEMENT_LIST *eptr;

  if (!gpuEnableLscTracking)
    return 0;
  if (!gpuBase.initialized || gpuBase.activeDevice < 0 || np <= 0)
    return 0;
  if (gpuBase.backtrack)
    return 0;
#if USE_MPI
  if (distributedBeam)
    return 0;
#endif
#ifdef GPU_VERIFY
  if (gpuCpuVerificationActive)
    return 0;
#endif
  if (!part || !part[0])
    return 0;
  if (!gpuParticleCountAllowed(np, gpuBase.lscMinParticles))
    return 0;
  if (!gpuRfcwLscKickDataSupported(lsc) || lsc->backtrack)
    return 0;
  eptr = (ELEMENT_LIST *)gpuBase.element;
  if (!eptr || eptr->type != T_RFCW)
    return 0;
  return 1;
}

long gpu_track_rfcw_lsc_kick_only(double **part, long np, LSCKICK *lsc,
                                  double Po, CHARGE *charge,
                                  double lengthScale,
                                  double dgammaOverGamma) {
  long copied;

  if (!gpuRfcwLscKickOnlyAllowed(part, np, lsc))
    return 0;
  if (!charge)
    return 0;

  gpuMarkHostWillChange();
  gpuBase.coord = part;
  gpuBase.hostCoordBase = part ? (void *)part[0] : NULL;
  gpuBase.nParticles = np;
  gpuBase.gpuRfcwLscKickOnlyCount++;
  gpuTrackRfcwLscKickOnDevice(np, lsc, Po, charge, lengthScale,
                              dgammaOverGamma);

  startGpuTimer();
  copied = gpuCopyDeviceToHost(np);
  gpuRecordWallSeconds();
  gpuRecordSyncRequest("RFCW LSC kick-only CPU handoff", copied, 0);
  gpuMarkHostWillChange();
  return 1;
}

static long gpuRfcwRfOnlyMatrixOnDevice(long np, RFCW *rfcw,
                                        double **accepted,
                                        double *P_central, double zEnd,
                                        double phase) {
  long remaining;

  if (np <= 0)
    return np;
  gpuRfcwRfOnlyMatrixCoreOnDevice(np, rfcw, P_central, phase,
                                  rfcw->dx, rfcw->dy);
  remaining = gpu_removeInvalidParticles(np, accepted, zEnd, *P_central);
  if (rfcw->change_p0)
    gpu_do_match_energy(remaining, P_central, 0);
  return remaining;
}

static long gpuRfcwMatrixWakeOnDevice(long np, RFCW *rfcw,
                                      double **accepted,
                                      double *P_central, double zEnd,
                                      double phase, RUN *run, long iPass,
                                      CHARGE *charge) {
  long hasWake = 0, hasTrwake = 0, remaining;
  double dgammaOverGammaAve = 0, volt;

  if (np <= 0)
    return np;
  if (!rfcw || !P_central)
    gpuRequiredFailure("NULL RFCW wake input in CUDA path");
  gpuRfcwWakeActive(rfcw, &hasWake, &hasTrwake);

  gpuRfcwApplyCoordinateOffset(np, 0, rfcw->dx,
                               "RFCW CUDA local x-offset kernel");
  gpuRfcwApplyCoordinateOffset(np, 2, rfcw->dy,
                               "RFCW CUDA local y-offset kernel");
  if (rfcw->doLSC) {
    volt = rfcw->volt / (1e6 * particleMassMV * particleRelSign);
    dgammaOverGammaAve =
      gpuRfcwDgammaOverGammaAveOnDevice(np, rfcw, *P_central,
                                        rfcw->length, volt, phase);
  }
  gpuRfcwRfOnlyMatrixCoreOnDevice(np, rfcw, P_central, phase, 0, 0);
  /* CPU matrix-method RFCW applies wakes after the positive-length matrix. */
  {
#ifdef GPU_VERIFY
    gpuWakeVerificationSuppressed++;
#endif
    if (hasWake)
      gpu_track_through_wake(np, &rfcw->wake, P_central, run, iPass, charge);
    if (hasTrwake)
      gpu_track_through_trwake(np, &rfcw->trwake, *P_central, run, iPass,
                               charge);
#ifdef GPU_VERIFY
    gpuWakeVerificationSuppressed--;
#endif
  }
  if (rfcw->doLSC) {
    gpuBase.gpuRfcwLscFullCount++;
    gpuTrackRfcwLscKickOnDevice(np, &rfcw->LSCKick, *P_central, charge,
                                rfcw->length, dgammaOverGammaAve);
  }
  gpuRfcwApplyCoordinateOffset(np, 0, -rfcw->dx,
                               "RFCW CUDA restore x-offset kernel");
  gpuRfcwApplyCoordinateOffset(np, 2, -rfcw->dy,
                               "RFCW CUDA restore y-offset kernel");

  remaining = gpu_removeInvalidParticles(np, accepted, zEnd, *P_central);
  if (rfcw->change_p0)
    gpu_do_match_energy(remaining, P_central, 0);
  return remaining;
}

static long gpuRfcwKickWakeOnDevice(long np, RFCW *rfcw,
                                    double **accepted,
                                    double *P_central, double zEnd,
                                    double phase, RUN *run, long iPass,
                                    CHARGE *charge) {
  long hasWake = 0, hasTrwake = 0, remaining;
  long ik, nKicks;
  double omega, volt, length, sectionLength, sectionVolt, dtLight;
  float milliseconds = 0;
  int status;

  if (np <= 0)
    return np;
  if (!rfcw || !P_central)
    gpuRequiredFailure("NULL RFCW kick-method wake input in CUDA path");
  gpuRfcwWakeActive(rfcw, &hasWake, &hasTrwake);
  gpuEnsureRfcwKickScratch(np);

  nKicks = rfcw->nKicks > 0 ? rfcw->nKicks : 1;
  length = rfcw->length;
  sectionLength = length / nKicks;
  omega = PIx2 * rfcw->freq;
  volt = rfcw->volt / (1e6 * particleMassMV * particleRelSign);
  sectionVolt = volt / nKicks;
  dtLight = sectionLength / c_mks;

  gpuRfcwApplyCoordinateOffset(np, 0, rfcw->dx,
                               "RFCW CUDA local x-offset kernel");
  gpuRfcwApplyCoordinateOffset(np, 2, rfcw->dy,
                               "RFCW CUDA local y-offset kernel");

  for (ik = 0; ik < nKicks; ik++) {
    double sectionPhase = rfcw->standingWave ? phase : phase - ik * omega * dtLight;
    double dgammaOverGammaAve = 0;

    if (rfcw->doLSC)
      dgammaOverGammaAve =
        gpuRfcwDgammaOverGammaAveOnDevice(np, rfcw, *P_central,
                                          sectionLength, sectionVolt,
                                          sectionPhase);

    startGpuTimer();
    gpuCopyHostToDevice(np);
    status = gpuCudaRfcwKickInitial(gpuBase.deviceCoord,
                                    gpuRfcwKickScratch.inverseF, np,
                                    (int)gpuBase.deviceStride, *P_central,
                                    sectionLength, sectionVolt, omega,
                                    sectionPhase,
                                    (int)(rfcw->end1Focus && ik == 0),
                                    c_mks, &milliseconds);
    if (status != 0)
      gpuFatalStatus("RFCW kick-method initial kernel", status);
    gpuRecordHelperKernel(milliseconds);
    gpuMarkDeviceChanged(np);
    gpuRecordWallSeconds();

    if (!rfcw->wakesAtEnd) {
#ifdef GPU_VERIFY
      gpuWakeVerificationSuppressed++;
#endif
      if (hasWake)
        gpu_track_through_wake(np, &rfcw->wake, P_central, run, iPass, charge);
      if (hasTrwake)
        gpu_track_through_trwake(np, &rfcw->trwake, *P_central, run, iPass,
                                 charge);
#ifdef GPU_VERIFY
      gpuWakeVerificationSuppressed--;
#endif
      if (rfcw->doLSC) {
        gpuBase.gpuRfcwLscFullCount++;
        gpuTrackRfcwLscKickOnDevice(np, &rfcw->LSCKick, *P_central, charge,
                                    sectionLength, dgammaOverGammaAve);
      }
    }

    startGpuTimer();
    gpuCopyHostToDevice(np);
    status = gpuCudaRfcwKickFinal(gpuBase.deviceCoord,
                                  gpuRfcwKickScratch.inverseF, np,
                                  (int)gpuBase.deviceStride, sectionLength,
                                  (int)(rfcw->end2Focus && ik == nKicks - 1),
                                  &milliseconds);
    if (status != 0)
      gpuFatalStatus("RFCW kick-method final kernel", status);
    gpuRecordHelperKernel(milliseconds);
    gpuMarkDeviceChanged(np);
    gpuRecordWallSeconds();

    if (rfcw->wakesAtEnd) {
#ifdef GPU_VERIFY
      gpuWakeVerificationSuppressed++;
#endif
      if (hasWake)
        gpu_track_through_wake(np, &rfcw->wake, P_central, run, iPass, charge);
      if (hasTrwake)
        gpu_track_through_trwake(np, &rfcw->trwake, *P_central, run, iPass,
                                 charge);
#ifdef GPU_VERIFY
      gpuWakeVerificationSuppressed--;
#endif
      if (rfcw->doLSC) {
        gpuBase.gpuRfcwLscFullCount++;
        gpuTrackRfcwLscKickOnDevice(np, &rfcw->LSCKick, *P_central, charge,
                                    sectionLength, dgammaOverGammaAve);
      }
    }
  }

  gpuRfcwApplyCoordinateOffset(np, 0, -rfcw->dx,
                               "RFCW CUDA restore x-offset kernel");
  gpuRfcwApplyCoordinateOffset(np, 2, -rfcw->dy,
                               "RFCW CUDA restore y-offset kernel");

  remaining = gpu_removeInvalidParticles(np, accepted, zEnd, *P_central);
  if (rfcw->change_p0)
    gpu_do_match_energy(remaining, P_central, 0);
  return remaining;
}

double gpu_findFiducialTime(long np, double s0, double sOffset,
                            double p0, unsigned long mode) {
  double tFid;
  GPU_BEAM_SUM_DATA result;
  float milliseconds = 0;
  int status = 0;
  long startPID = -1, endPID = -1;

  if (mode & FID_MODE_LIGHT)
    return (s0 + sOffset) / c_mks;

  if (np <= 0)
    gpuRequiredFailure("No available particle for RF cavity fiducialization");
  if (!gpuFiducialPidRange(mode, &startPID, &endPID))
    gpuRequiredFailure("invalid fiducial PID range for CUDA RF fiducialization");

  memset(&result, 0, sizeof(result));
  startGpuTimer();
  gpuCopyHostToDevice(np);
  if (mode & FID_MODE_TMEAN)
    status = gpuCudaFiducialTimeSums(
      gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, p0, sOffset,
      c_mks, particleIDIndex, startPID, endPID, &result, &milliseconds);
  else if (mode & FID_MODE_FIRST)
    status = gpuCudaFiducialFirst(
      gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, p0, sOffset,
      c_mks, particleIDIndex, startPID, endPID, &result, &milliseconds);
  else if (mode & FID_MODE_PMAX)
    status = gpuCudaFiducialPmaximum(
      gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, p0, sOffset,
      c_mks, particleIDIndex, startPID, endPID, &result, &milliseconds);
  else
    gpuRequiredFailure("invalid fiducial mode in CUDA RF fiducialization");
  if (status != 0)
    gpuFatalStatus("RF fiducialization reduction kernel", status);
  gpuRecordReductionKernel(milliseconds);
  gpuRecordWallSeconds();

  if (result.count <= 0)
    gpuRequiredFailure("No available particle for RF cavity fiducialization");
  if (mode & FID_MODE_TMEAN)
    tFid = result.centroidSum[6] / result.count;
  else
    tFid = result.centroidSum[6];
  return tFid;
}

static long gpuRfcwSetupPhaseOnDeviceWithOffset(RFCW *rfcw, long np,
                                                double *P_central,
                                                double zEnd,
                                                double sOffset,
                                                double *phaseOut) {
  RFCA *rfca;
  double phase = 0, t0 = 0, omega;
  long phaseStatus;

  if (!rfcw || !P_central || !phaseOut)
    return 0;
  if (!get_phase_reference || !set_phase_reference || !unused_phase_reference)
    return 0;
  rfca = &rfcw->rfca;
  if (rfca->phase_reference == 0)
    rfca->phase_reference = unused_phase_reference();
  phaseStatus = get_phase_reference(&phase, rfca->phase_reference);
  switch (phaseStatus) {
  case REF_PHASE_RETURNED:
    break;
  case REF_PHASE_NOT_SET:
  case REF_PHASE_NONEXISTENT:
    if (!rfca->fiducial_seen) {
      unsigned long mode;

      mode = gpuRfcwFiducialMode(rfca->fiducial);
      if (!gpuFiducialModeSupported(mode))
        return 0;
      omega = PIx2 * rfca->freq;
      if (rfca->tReference != -1)
        t0 = rfca->tReference;
      else if (mode & FID_MODE_LIGHT)
        t0 = (zEnd - rfca->length + sOffset) / c_mks;
      else if (mode & (FID_MODE_TMEAN | FID_MODE_FIRST | FID_MODE_PMAX))
        t0 = gpu_findFiducialTime(np, zEnd - rfca->length,
                                  sOffset, *P_central, mode);
      else
        return 0;
      rfca->phase_fiducial = -omega * t0;
      rfca->fiducial_seen = 1;
    }
    set_phase_reference(rfca->phase_reference,
                        phase = rfca->phase_fiducial);
    break;
  default:
    return 0;
  }

  if (rfca->freq)
    rfca->t_fiducial = -rfca->phase_fiducial / (PIx2 * rfca->freq);
  else
    rfca->t_fiducial = 0;
  *phaseOut = phase + rfca->phase * PI / 180.0;
  rfcw->initialized = 1;
  return 1;
}

static long gpuRfcwSetupPhaseOnDevice(RFCW *rfcw, long np,
                                      double *P_central, double zEnd,
                                      double *phaseOut) {
  if (!rfcw)
    return 0;
  return gpuRfcwSetupPhaseOnDeviceWithOffset(rfcw, np, P_central, zEnd,
                                             rfcw->rfca.length / 2,
                                             phaseOut);
}

long gpu_trackRfCavityWithWakes(long np, RFCA *rfca, double **accepted,
                                double *P_central, double zEnd, long iPass,
                                RUN *run, CHARGE *charge, WAKE *wake,
                                TRWAKE *trwake, LSCKICK *LSCKick,
                                long wakesAtEnd) {
  ELEMENT_LIST *eptr = (ELEMENT_LIST *)gpuBase.element;
  double phase;

  if (!rfca || !P_central)
    gpuRequiredFailure("NULL RFCA input in CUDA path");
  if (wake || trwake || LSCKick || run || charge || iPass || wakesAtEnd) {
    return gpuRfcaOnCpu(np, rfca, accepted, P_central, zEnd, iPass,
                        run, charge, wake, trwake, LSCKick, wakesAtEnd,
                        "RFCA unsupported CUDA option");
  }

  if (gpuRfcaThinKickElementSupported(eptr)) {
    if (!gpuRfcaSetupPhaseOnDevice(rfca, np, P_central, zEnd, &phase)) {
      return gpuRfcaOnCpu(np, rfca, accepted, P_central, zEnd, iPass,
                          run, charge, wake, trwake, LSCKick, wakesAtEnd,
                          "RFCA CUDA phase reference setup");
    }
    return gpuRfcaThinKickOnDevice(np, rfca, accepted, P_central, zEnd,
                                   phase);
  }

  if (gpuRfcaRfOnlyMatrixElementSupported(eptr)) {
    if (!gpuRfcaSetupPhaseOnDevice(rfca, np, P_central, zEnd, &phase)) {
      return gpuRfcaOnCpu(np, rfca, accepted, P_central, zEnd, iPass,
                          run, charge, wake, trwake, LSCKick, wakesAtEnd,
                          "RFCA CUDA phase reference setup");
    }
    return gpuRfcaRfOnlyMatrixOnDevice(np, rfca, accepted, P_central, zEnd,
                                       phase);
  }

  if (gpuRfcaRfOnlyKickElementSupported(eptr)) {
    long nKicks = rfca->nKicks > 0 ? rfca->nKicks : 1;
    double sectionOffset = rfca->length / nKicks / 2;

    if (!gpuRfcaSetupPhaseOnDeviceWithOffset(rfca, np, P_central, zEnd,
                                             sectionOffset, &phase)) {
      return gpuRfcaOnCpu(np, rfca, accepted, P_central, zEnd, iPass,
                          run, charge, wake, trwake, LSCKick, wakesAtEnd,
                          "RFCA CUDA phase reference setup");
    }
    return gpuRfcaRfOnlyKickOnDevice(np, rfca, accepted, P_central, zEnd,
                                     phase);
  }

  if (!gpuRfcaRemoveInvalidOnlyElementSupported(eptr)) {
    return gpuRfcaOnCpu(np, rfca, accepted, P_central, zEnd, iPass,
                        run, charge, wake, trwake, LSCKick, wakesAtEnd,
                        "RFCA unsupported CUDA option");
  }

  if (rfca->phase_reference == 0 && unused_phase_reference)
    rfca->phase_reference = unused_phase_reference();
  if (!rfca->fiducial_seen) {
    rfca->phase_fiducial = 0;
    rfca->fiducial_seen = 1;
  }
  rfca->t_fiducial = 0;
  if (rfca->phase_reference && set_phase_reference)
    set_phase_reference(rfca->phase_reference, 0);
  return gpu_removeInvalidParticles(np, accepted, zEnd, *P_central);
}
long gpu_track_through_rfcw(long np, RFCW *rfcw, double **accepted,
                            double *P_central, double zEnd,
                            RUN *run, long iPass, CHARGE *charge) {
  ELEMENT_LIST *eptr = (ELEMENT_LIST *)gpuBase.element;
  double phase = 0;

  if (!rfcw || !P_central)
    gpuRequiredFailure("NULL RFCW input in CUDA path");
  gpuRefreshRfcwRfca(rfcw);

  if (gpuRfcwRfOnlyElementSupported(eptr)) {
    if (!gpuRfcwSetupPhaseOnDevice(rfcw, np, P_central, zEnd, &phase))
      return gpuRfcwOnCpu(np, rfcw, accepted, P_central, zEnd, run, iPass,
                          charge, "RFCW CUDA phase reference setup");
    return gpuRfcwRfOnlyMatrixOnDevice(np, rfcw, accepted, P_central, zEnd,
                                       phase);
  }

  if (gpuRfcwRfOnlyKickElementSupported(eptr)) {
    long nKicks = rfcw->nKicks > 0 ? rfcw->nKicks : 1;
    double sectionOffset = rfcw->length / nKicks / 2;

    if (!gpuRfcwSetupPhaseOnDeviceWithOffset(rfcw, np, P_central, zEnd,
                                             sectionOffset, &phase))
      return gpuRfcwOnCpu(np, rfcw, accepted, P_central, zEnd, run, iPass,
                          charge, "RFCW CUDA phase reference setup");
    return gpuRfcwKickWakeOnDevice(np, rfcw, accepted, P_central, zEnd,
                                   phase, run, iPass, charge);
  }

  if (gpuRfcwMatrixWakeElementSupported(eptr)) {
    gpuPrepareRfcwWakeData(rfcw);
    if (!gpuWakeDataSupported(&rfcw->wake) ||
        !gpuTrwakeDataSupported(&rfcw->trwake))
      return gpuRfcwOnCpu(np, rfcw, accepted, P_central, zEnd, run, iPass,
                          charge, "RFCW unsupported CUDA wake option");
    if (!gpuRfcwSetupPhaseOnDevice(rfcw, np, P_central, zEnd, &phase))
      return gpuRfcwOnCpu(np, rfcw, accepted, P_central, zEnd, run, iPass,
                          charge, "RFCW CUDA phase reference setup");
    return gpuRfcwMatrixWakeOnDevice(np, rfcw, accepted, P_central, zEnd,
                                     phase, run, iPass, charge);
  }

  if (gpuRfcwKickWakeElementSupported(eptr)) {
    long nKicks = rfcw->nKicks > 0 ? rfcw->nKicks : 1;
    double sectionOffset = rfcw->length / nKicks / 2;

    gpuPrepareRfcwWakeData(rfcw);
    if (!gpuWakeDataSupported(&rfcw->wake) ||
        !gpuTrwakeDataSupported(&rfcw->trwake))
      return gpuRfcwOnCpu(np, rfcw, accepted, P_central, zEnd, run, iPass,
                          charge, "RFCW unsupported CUDA wake option");
    if (!gpuRfcwSetupPhaseOnDeviceWithOffset(rfcw, np, P_central, zEnd,
                                             sectionOffset, &phase))
      return gpuRfcwOnCpu(np, rfcw, accepted, P_central, zEnd, run, iPass,
                          charge, "RFCW CUDA phase reference setup");
    return gpuRfcwKickWakeOnDevice(np, rfcw, accepted, P_central, zEnd,
                                   phase, run, iPass, charge);
  }

  return gpuRfcwOnCpu(np, rfcw, accepted, P_central, zEnd, run, iPass,
                      charge, "RFCW unsupported CUDA option");
}
