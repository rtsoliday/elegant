#include "mdb.h"
#include "match_string.h"
#include "../track.h"
#include "../zibs.h"
#undef random_2
#include "gpu_base.h"
#include "gpu_cwiggler.h"
#include "gpu_tune.h"
#include "gpu_simple_rfca.h"
#include "gpu_lorentz.h"
#include "fftpackC.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <pthread.h>
#ifdef _OPENMP
#  include <omp.h>
#endif

#pragma weak compute_centroids
#pragma weak setupMultApertureData
#pragma weak ccbendSetupCentroidOutput
#pragma weak ccbendCentroidSumsEnabled
#pragma weak ccbendWriteCentroidOutput
#pragma weak lgbendSetupCentroidOutput
#pragma weak lgbendCentroidSumsEnabled
#pragma weak lgbendWriteCentroidOutput
#pragma weak convertLocalCoordinatesToGlobal
#pragma weak initializeApContour
#pragma weak accumulate_beam_sums
#pragma weak limit_amplitudes
#pragma weak elimit_amplitudes
#pragma weak rectangular_collimator
#pragma weak elliptical_collimator
#pragma weak beam_scraper
#pragma weak removeInvalidParticles
#pragma weak imposeApertureData
#pragma weak trackThroughApContour
#pragma weak trackRfCavityWithWakes
#pragma weak track_through_rfcw
#pragma weak findFiducialTime
#pragma weak setupLaserModulator
#pragma weak motionGpuSetup
#pragma weak do_match_energy
#pragma weak KahanPlus
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
#pragma weak readLGBendApertureData
#pragma weak convertLocalCoordinatesToGlobal
#pragma weak multipole_tracking
#pragma weak multipole_tracking2
#pragma weak fmultipole_tracking
#pragma weak initialize_fmultipole
#pragma weak expandHamiltonian
#pragma weak beta_from_delta
#pragma weak determineDQCorReferenceFrameTilt
#pragma weak computeQuadSteeringStrengths
#pragma weak rotate_xy
#pragma weak multipoleKicksDone
#pragma weak track_through_csbend
#pragma weak track_through_ccbend
#pragma weak track_through_lgbend
#pragma weak trackThroughExactCorrector
#pragma weak trackThroughTaperApCirc
#pragma weak trackThroughTaperApRectangular
#pragma weak trackThroughTaperApElliptical
#pragma weak obstructionDataPresent
#pragma weak getLinearizedMisalignmentMatrix
#pragma weak getExactMisalignmentFrame
#pragma weak setupKquadErrorMultipoles
#pragma weak setupKsextErrorMultipoles
#pragma weak setupKoctErrorMultipoles
#pragma weak setupDqcorErrorMultipoles
#pragma weak setupCcbendErrorMultipoles
#pragma weak optimizeCcbendForGpu
#pragma weak optimizeLgbendForGpu
#pragma weak multipoleExpansionCoefficients
#pragma weak track_through_speedbump
#ifdef GPU_VERIFY
#pragma weak gpu_verify_csrcsbend_cpu_body_slice
#pragma weak rms_longitudinal_emittance
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
/* zlongit.c and ztransverse.c don't declare these in track.h */
void set_up_zlongit(ZLONGIT *zlongit, RUN *run, long pass, long particles, CHARGE *charge,
                    double timeSpan);
void set_up_ztransverse(ZTRANSVERSE *ztransverse, RUN *run, long pass, long particles, CHARGE *charge,
                        double timeSpan);
#pragma weak set_up_zlongit
#pragma weak set_up_ztransverse
#pragma weak exitElegant
/* IBSCATTER's host routines (ibs_tracking.c, zibs.c, sliceAnalysis.c) */
void init_IBS(ELEMENT_LIST *element);
void zeroslice(long islice, IBSCATTER *IBS);
void forth_propagate_twiss(IBSCATTER *IBS, long islice, double betax0, double alphax0,
                           double betay0, double alphay0, RUN *run);
void slicebeam(double **coord, int64_t np, double *time, double Po, long nslice, long *index,
               long *count, double *dt);
long computeSliceParameters(double C[6], double S[6][6], double **part, long *index, long start,
                            long end, double Po);
#pragma weak slicebeam
#pragma weak computeSliceParameters
#pragma weak init_IBS
#pragma weak zeroslice
#pragma weak forth_propagate_twiss
#pragma weak IBSRate
#pragma weak correctedEmittance
#pragma weak set_up_lrwake
#pragma weak lrwakeBunchVoltages
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
#pragma weak bmapxy_field_setup
#pragma weak lorentz_setup

extern unsigned long multipoleKicksDone;
extern double **Fx_xy;
extern double **Fy_xy;
extern long expansionOrder1;
extern long hasSkew;
extern long hasNormal;
extern void readLGBendApertureData(LGBEND *lgbend);
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
extern void bmapxy_field_setup(BMAPXY *bmapxy);
extern void lorentz_setup(void *field, long field_type, double **part,
                          long np, double Po);
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
extern long trackThroughExactCorrector(double **part, long n_part,
                                       ELEMENT_LIST *eptr, double Po,
                                       double **accepted, double z_start,
                                       double *sigmaDelta2);
#ifdef GPU_VERIFY
extern long gpu_verify_csrcsbend_cpu_body_slice(double *coord,
                                                CSRCSBEND *csbend,
                                                double beta0,
                                                double sliceLength,
                                                double rho0,
                                                double rhoActual,
                                                double Po,
                                                MULT_APERTURE_DATA *apertureData);
#endif
extern long multipole_tracking(double **particle, long n_part, MULT *multipole,
                               double p_error, double Po, double **accepted,
                               double z_start);
extern void initialize_fmultipole(FMULT *multipole);
extern unsigned short expandHamiltonian;
extern long track_through_csbendCSR_cuda_resident_entry(
  double **part, long n_part, CSRCSBEND *csbend, double p_error,
  double Po, double **accepted, double z_start, double z_end,
  CHARGE *charge, char *rootname, MAXAMP *maxamp, APCONTOUR *apContour,
  APERTURE_DATA *apFileData, ELEMENT_LIST *eptr) __attribute__((weak));
extern long track_through_driftCSR_cuda_resident_entry(
  double **part, long np, CSRDRIFT *csrDrift, double Po, double **accepted,
  double zStart, double revolutionLength, CHARGE *charge, char *rootname) __attribute__((weak));
extern long csrDriftHasWakeToApply(CSRDRIFT *csrDrift, long np) __attribute__((weak));

extern int gpuCudaRuntimeGetDeviceCount(int *count);
extern int gpuCudaRuntimeSetDevice(int device);
extern int gpuCudaRuntimeProbeDeviceCode(void);
extern const char *gpuCudaRuntimeGetErrorString(int code);
extern int gpuCudaRuntimeGetDeviceName(int device, char *name, unsigned long nameSize);
extern int gpuCudaMallocDouble(void **ptr, unsigned long count);
extern int gpuCudaMallocBytes(void **ptr, unsigned long bytes);
extern int gpuCudaFree(void *ptr);
extern int gpuCudaCopyHostToDevice(void *dst, const void *src, unsigned long count, float *milliseconds);
extern int gpuCudaCopyHostBytesToDevice(void *dst, const void *src, unsigned long bytes);
extern int gpuCudaCopyDeviceToDevice(void *dst, const void *src, unsigned long count);
extern int gpuCudaIsrFillDraws(long count, unsigned long long firstState, double limit,
                               const void **draws, long *uniformsUsed, float *milliseconds);
extern int gpuCudaIsrUploadDraws(long count, const double *hostDraws, const void **draws,
                                 float *milliseconds);
extern int gpuCudaCopyDeviceToHost(void *dst, const void *src, unsigned long count, float *milliseconds);
extern int gpuCudaCopyDeviceBytesToHost(void *dst, const void *src, unsigned long bytes, float *milliseconds);
extern int gpuCudaStableScatterRows(void *coord, void *scratchCoord,
                                    const void *prefix, long nParticles,
                                    int stride, long survivors,
                                    float *milliseconds);
extern int gpuCudaLostRowsBeforeElement(const void *coord, void *scratchCoord,
                                        const void *destination, long nParticles,
                                        int stride, long survivors, double *hostRows,
                                        float *milliseconds);
extern int gpuCudaCompactTrackedRows(void *coord, void *scratchCoord, void *prefix,
                                     long nParticles, int stride, long *remaining,
                                     float *milliseconds);
extern int gpuCudaRestoreMagnetBackup(void *coord, long nParticles, int stride);
extern int gpuCudaTrackParticles(void *coord, long nParticles, int stride,
                                 const GPU_MATRIX_DATA *matrix, float *milliseconds);
extern int gpuCudaExactDrift(void *coord, long nParticles, int stride,
                             double length, float *milliseconds);
extern int gpuCudaLinearDrift(void *coord, long nParticles, int stride,
                              double length, float *milliseconds);
extern int gpuCudaOffsetBeam(void *coord, long nParticles, int stride,
                             double dx, double dxp, double dy, double dyp,
                             double dz, double dt, double dp, double de,
                             double pCentral, int64_t startPID, int64_t endPID,
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
extern int gpuCudaBatchedApertureSearch(
  const GPU_BATCHED_APERTURE_ELEMENT *element, long elements,
  long lines, long nx, long nSplits, double splitFraction, long passes,
  double pCentral, double xmax, double ymax, const double *orbit,
  const double *dxFactor, const double *dyFactor,
  double *xLimit, double *yLimit, double *xLost, double *yLost,
  double *sLost, long *lossPass, long *lossElement,
  double *lossX0, double *lossY0, long *originStable, float *milliseconds);
extern int gpuCudaTuneProgramTrack(
  const GPU_TUNE_PROGRAM *program, const double *coordinate, long particles,
  int stride, long turns, long turnOffset, double *endingCoordinate,
  GPU_TUNE_PARTICLE_STATUS *particleStatus, double *history,
  float *milliseconds, long *launches, long *transfers);
extern int gpuCudaSetCentralMomentum(void *coord, long nParticles, int stride,
                                     double oldP, double newP, float *milliseconds);
extern int gpuCudaParticleTimes(void *coord, long nParticles, int stride, double pCentral,
                                double cMks, void *deviceTimes, double *hostTimes,
                                float *milliseconds);
extern int gpuCudaRamppRescale(void *coord, long nParticles, int stride, double oldP,
                               double newP, float *milliseconds);
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
extern int gpuCudaCoordinateColumnToHost(const void *coord, long nParticles, int stride,
                                         int column, double *values, float *milliseconds);
extern int gpuCudaColumnSnapshotTake(const void *coord, long nParticles, int stride,
                                     int column, float *milliseconds);
extern int gpuCudaColumnSnapshotCompare(const void *coord, long nParticles, int stride,
                                        int column, int *same, float *milliseconds);
extern void gpuCudaColumnSnapshotRelease(void);
extern void gpuCudaReductionRelease(void);
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
                               double phase, double cMks, const void *referenceKick,
                               float *milliseconds);
extern int gpuCudaRfcaPhaseValues(void *coord, void *phaseValues,
                                  void *momentum, long nParticles, int stride,
                                  double pCentral, double length, double omega,
                                  double phase, double timeShift, double cMks,
                                  float *milliseconds);
extern int gpuCudaRamprf(void *coord, long nParticles, int stride,
                         double pCentral, double length, double volt,
                         double omega, double phase, double tOffset, double cMks,
                         int stage, void *phaseValues,
                         const void *referenceKick, long *badCount,
                         float *milliseconds);
extern int gpuCudaRfParticleTimes(const void *coord, long nParticles, int stride, double pCentral,
                                  double cMks, double *hostValues, float *milliseconds);
extern void gpuCudaRfParticleTimesRelease(void);
extern int gpuCudaSpinSums(void *coord, long nParticles, int stride, int spinOffset,
                           double centroidSum[3], double productSum[6], float *milliseconds);
extern int gpuCudaSliceTimes(void *coord, long nParticles, int stride, double Po, double cMks,
                             double *tMin, double *tMax, float *milliseconds);
extern int gpuCudaSliceMoments(void *coord, long nParticles, int stride, const double *bounds,
                               long slices, long *count, double *mean, double *secondSum,
                               float *milliseconds);
extern int gpuCudaTfeedbackKick(void *coord, long nParticles, int stride,
                                int pickupCoordinate, int longitudinal,
                                double kick, float *milliseconds);
extern int gpuCudaRfdfTrack(void *coord, long nParticles, int stride,
                            const GPU_RFDF_DATA *data,
                            int particleIdIndex, float *milliseconds);
extern int gpuCudaSreffectsTrack(void *coord, long nParticles, int stride,
                                 const GPU_SREFFECTS_DATA *data,
                                 float *milliseconds);
extern int gpuCudaIlmatrixPhases(void *coord, void *phase, long nParticles, int stride,
                                 const GPU_ILMATRIX_DATA *data, float *milliseconds);
extern int gpuCudaIlmatrixTrack(void *coord, void *scratchCoord, void *prefix, long nParticles,
                                int stride, const GPU_ILMATRIX_DATA *data, const void *sinCos,
                                long *remaining, long *divideByZero, float *milliseconds);
extern int gpuCudaBggexpTrack(void *coord, long nParticles, int stride,
                              const GPU_BGGEXP_DATA *data,
                              float *milliseconds);
extern void gpuCudaBggexpRelease(void);
extern int gpuCudaCwigglerTrack(void *coord, long nParticles, int stride,
                                const GPU_CWIGGLER_DATA *data,
                                float *milliseconds);
extern void gpuCudaCwigglerRelease(void);
extern int gpuCudaFtableTrack(void *coord, long nParticles, int stride,
                              const GPU_FTABLE_DATA *data,
                              float *milliseconds);
extern void gpuCudaFtableRelease(void);
extern int gpuCudaBmxyzTrack(void *coord, long nParticles, int stride,
                             const GPU_BMXYZ_DATA *data,
                             void *scratchCoord, void *prefix, long *remaining,
                             long *failedCount, float *milliseconds);
extern void gpuCudaBmxyzRelease(void);
extern int gpuCudaLorentzTrack(void *coord, long nParticles, int stride,
                               const GPU_LORENTZ_DATA *data,
                               long *failedCount, float *milliseconds);
extern void gpuCudaLorentzRelease(void);
extern int gpuCudaLsrmdltrBackup(const void *coord, long values);
extern int gpuCudaLsrmdltrRestore(void *coord, long firstValue, long values);
extern int gpuCudaLsrmdltrPass(void *coord, long nParticles, int stride,
                               const GPU_LSRMDLTR_DATA *data, const long long *offsets,
                               int *counts, double *xRange0, long *failedCount,
                               float *milliseconds);
extern void gpuCudaLsrmdltrRelease(void);
extern int gpuCudaMotionPass(void *coord, long nParticles, int stride, const GPU_MOTION_DATA *data,
                             const double *tables, long tableValues, unsigned char *lostHost,
                             long *lostCount, long *cpuCount, float *milliseconds);
extern void gpuCudaMotionRelease(void);
extern int gpuCudaMotionPermute(void *coord, long nParticles, int stride, const long long *permHost);
extern int gpuCudaBeamBeamKick(void *coord, long nParticles, int stride, const GPU_BEAMBEAM_DATA *data,
                               float *milliseconds);
extern int gpuCudaRfcwRfOnlyMatrix(void *coord, long nParticles, int stride,
                                   double pCentral, double length,
                                   double volt, double omega, double phase,
                                   int end1Focus, int end2Focus,
                                   double dx, double dy, double cMks,
                                   const void *referenceKick,
                                   const void *referenceR12,
                                   float *milliseconds);
extern int gpuCudaRfcwRfOnlyMatrixChecked(
  void *coord, long nParticles, int stride, double pCentral, double length,
  double volt, double omega, double phase, int end1Focus, int end2Focus,
  double dx, double dy, double cMks, const void *referenceKick,
  const void *referenceR12, void *deviceLostCount,
  long *lostCount, float *milliseconds);
extern int gpuCudaRfcwKickInitial(void *coord, void *inverseF,
                                  long nParticles, int stride,
                                  double pCentral, double length,
                                  double volt, double omega, double phase,
                                  double timeShift, int end1Focus,
                                  double cMks, const void *referenceKick,
                                  float *milliseconds);
extern int gpuCudaRfcwKickFinal(void *coord, const void *inverseF,
                                long nParticles, int stride,
                                double length, int end2Focus,
                                float *milliseconds);
extern int gpuCudaRfcwDgammaOverGammaSums(void *coord, long nParticles,
                                          int stride, double pCentral,
                                          double length, double volt,
                                          double omega, double phase,
                                          double timeShift, double cMks,
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
extern int gpuCudaCenterTimeRoundTrip(void *coord, long nParticles, int stride, double pCentral,
                                      double timeOffset, double cMks, float *milliseconds);
extern int gpuCudaTimeColumnToHost(const void *coord, long nParticles, int stride, double pCentral,
                                   double cMks, double *hostTimes, float *milliseconds);
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
                                   int64_t startPID, int64_t endPID,
                                   GPU_BEAM_SUM_DATA *result,
                                   float *milliseconds);
extern int gpuCudaFiducialValues(void *coord, long nParticles, int stride,
                                 double pCentral, double sOffset,
                                 double cMks, int particleIdColumn,
                                 int64_t startPID, int64_t endPID,
                                 double *hostValue,
                                 unsigned char *hostSelected,
                                 double *hostDelta, float *milliseconds);
extern int gpuCudaFiducialPmaximum(void *coord, long nParticles, int stride,
                                   double pCentral, double sOffset,
                                   double cMks, int particleIdColumn,
                                   int64_t startPID, int64_t endPID,
                                   GPU_BEAM_SUM_DATA *result,
                                   float *milliseconds);
extern int gpuCudaFiducialFirst(void *coord, long nParticles, int stride,
                                double pCentral, double sOffset,
                                double cMks, int particleIdColumn,
                                int64_t startPID, int64_t endPID,
                                GPU_BEAM_SUM_DATA *result,
                                float *milliseconds);
extern int gpuCudaCentroidTimeSums(void *coord, long nParticles, int stride,
                                   double pCentral, double cMks,
                                   GPU_BEAM_SUM_DATA *result, float *milliseconds);
extern int gpuCudaBeamSums(void *coord, long nParticles, int stride,
                           double pCentral, double cMks,
                           GPU_BEAM_SUM_DATA *result, float *milliseconds);
extern unsigned long gpuCudaBeamSums2ScratchBytes(long nParticles);
extern int gpuCudaBeamSums2(void *coord, long nParticles, int stride,
                            double pCentral, double cMks,
                            GPU_BEAM_SUM_DATA *result,
                            GPU_BEAM_SUM_DATA *centeredResult,
                            void *deviceScratch, unsigned int productMask,
                            float *milliseconds);
extern int gpuCudaHistogramRanges(
  void *coord, long nParticles, int stride, double pCentral, double cMks,
  int64_t startPID, int64_t endPID, unsigned int coordinateMask,
  GPU_HISTOGRAM_RANGE_DATA *result, float *milliseconds);
extern int gpuCudaHistogramBins(
  void *coord, long nParticles, int stride, double pCentral, double cMks,
  int64_t startPID, int64_t endPID, long bins, unsigned int coordinateMask,
  double timeOffset, const double *lower, const double *upper,
  unsigned long long *histogramReturn, float *milliseconds);
extern void gpuCudaHistogramRelease(void);
extern int gpuCudaLongMinMax(void *coord, long nParticles, int stride,
                             int coordinateIndex,
                             GPU_LONG_MIN_MAX_DATA *result,
                             float *milliseconds);
extern int gpuCudaLrwakeValues(const void *coord, long nParticles, int stride, double pCentral,
                               double cMks, int bunchColumn, double *hostPacked,
                               float *milliseconds);
extern int gpuCudaLrwakeKick(void *coord, long nParticles, int stride, const double *hostVoltage,
                             long nBunches, long ibMin, int bunchColumn, double factor,
                             double pCentral, double massMV, double cMks, float *milliseconds);
extern void gpuCudaLrwakeRelease(void);
extern int gpuCudaSortedBunchRanges(
  void *coord, long nParticles, int stride, int coordinateIndex,
  long minBunch, long nBuckets, long *start, long *count, long *sorted,
  float *milliseconds);
extern int gpuCudaGatherByBunch(void *coord, long nParticles, int stride, int coordinateIndex,
                                long minBunch, long nBuckets, void **gatheredReturn,
                                long *start, long *count, float *milliseconds);
extern int gpuCudaScatterByBunch(void *coord, long nParticles, int stride, float *milliseconds);
extern void gpuCudaBunchGatherRelease(void);
extern int gpuCudaIbsInflate(void *coord, long nParticles, int stride, long nslice,
                             const unsigned int *sliceOfRow, const double *parameters,
                             const double *eta, double pCentral, double cMks, double dtSlice,
                             double duration, float *milliseconds);
extern void gpuCudaIbsRelease(void);
extern int gpuCudaDoubleMinMax(void *coord, long nParticles, int stride,
                               int coordinateIndex,
                               GPU_DOUBLE_MIN_MAX_DATA *result,
                               void *deviceResult,
                               float *milliseconds);
extern int gpuCudaLimitAmplitudeLossCount(void *coord, long nParticles, int stride,
                                          double xmax, double ymax, int openCode,
                                          long *lostCount, float *milliseconds);
extern int gpuCudaLimitAmplitudesStableCompact(void *coord, void *scratchCoord,
                                               void *prefix, long nParticles,
                                               int stride, double xmax,
                                               double ymax, int openCode, double z,
                                               double pCentral,
                                               long extrapolateZ,
                                               long *remaining,
                                               float *milliseconds);
extern int gpuCudaELimitAmplitudeLossCount(void *coord, long nParticles, int stride,
                                           double xmax, double ymax,
                                           long exponent, long yExponent, int openCode,
                                           long *lostCount, float *milliseconds);
extern int gpuCudaELimitAmplitudesStableCompact(void *coord, void *scratchCoord,
                                                void *prefix, long nParticles,
                                                int stride, double xmax,
                                                double ymax, long exponent,
                                                long yExponent, int openCode, double z,
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
extern int gpuCudaMultipoleLoadCoefficients(const double *table);
extern int gpuCudaMultipoleTrackChecked(void *coord, long nParticles, int stride,
                                        const GPU_MULTIPOLE_DATA *multipole,
                                        void *survivorFlags,
                                        long *lostCount, float *milliseconds);
extern int gpuCudaMultipoleTrackStableCompact(void *coord, void *scratchCoord,
                                              void *prefix, long nParticles,
                                              int stride,
                                              const GPU_MULTIPOLE_DATA *multipole,
                                              long *remaining,
                                              float *milliseconds);
extern int gpuCudaExactCorrectorTrackStableCompact(
  void *coord, void *scratchCoord, void *prefix, long nParticles, int stride,
  const GPU_EXACT_CORRECTOR_DATA *corrector, long *remaining,
  long *usedScratch, float *milliseconds);
extern int gpuCudaApContourLossCount(void *coord, long nParticles, int stride,
                                     const GPU_APERTURE_CONTOUR_DATA *contour, double length,
                                     long *lostCount, float *milliseconds);
extern int gpuCudaApContourStableCompact(void *coord, void *scratchCoord, void *prefix,
                                         long nParticles, int stride,
                                         const GPU_APERTURE_CONTOUR_DATA *contour, double z,
                                         double pCentral, long *remaining, float *milliseconds);
extern int gpuCudaTaperApertureTrackStableCompact(
  void *coord, void *scratchCoord, void *prefix, long nParticles, int stride,
  const GPU_TAPER_APERTURE_DATA *taper, long *remaining,
  float *milliseconds, long *uncertainCount);
extern int gpuCudaSpeedbumpTrackStableCompact(
  void *coord, void *scratchCoord, void *prefix, long nParticles, int stride,
  const GPU_SPEEDBUMP_DATA *speedbump, long *remaining,
  float *milliseconds);
extern int gpuCudaCsbendTrackChecked(void *coord, long nParticles, int stride,
                                     const GPU_CSBEND_DATA *csbend,
                                     void *survivorFlags,
                                     long *lostCount, float *milliseconds);
extern int gpuCudaCsbendTrackStableCompact(void *coord, void *scratchCoord,
                                           void *prefix, long nParticles,
                                           int stride,
                                           const GPU_CSBEND_DATA *csbend,
                                           long *remaining,
                                           float *milliseconds);
extern int gpuCudaCcbendTrackChecked(void *coord, long nParticles, int stride,
                                     const GPU_CCBEND_DATA *ccbend,
                                     void *survivorFlags,
                                     long *lostCount, float *milliseconds);
extern int gpuCudaCcbendTrackStableCompact(
  void *coord, void *scratchCoord, void *prefix, long nParticles, int stride,
  const GPU_CCBEND_DATA *ccbend, long *remaining, float *milliseconds);
extern int gpuCudaLgbendTrackChecked(void *coord, long nParticles, int stride,
                                     const GPU_LGBEND_DATA *lgbend,
                                     const GPU_LGBEND_LOCAL_APERTURE_DATA
                                       *localAperture,
                                     long *lostCount, float *milliseconds);
extern int gpuCudaLgbendTrackStableCompact(
  void *coord, void *scratchCoord, void *prefix, long nParticles, int stride,
  const GPU_LGBEND_DATA *lgbend,
  const GPU_LGBEND_LOCAL_APERTURE_DATA *localAperture,
  long *remaining, float *milliseconds);
extern int gpuCudaCsrCsbendBodySliceChecked(void *coord, long nParticles,
                                            int stride,
                                            const GPU_CSBEND_DATA *csbend,
                                            double *beta0,
                                            void *backup,
                                            int keepLosses,
                                            long *destination,
                                            long *remaining,
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
extern int gpuCudaOrderedTimeSum(void *coord, long nParticles, int stride,
                                 double pCentral, double cMks,
                                 int bunchColumn, long selectedBunch,
                                 double *sum, long *count,
                                 float *milliseconds);
extern int gpuCudaTimeSpan(void *coord, long nParticles, int stride,
                           double pCentral, double cMks, int bunchColumn,
                           long selectedBunch, long *count, double *tmin,
                           double *tmax, double *tsum, float *milliseconds);
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
  GPU_COMBINED_WAKE_HOST_VOLTAGE hostVoltage, void *hostVoltageContext,
  float *milliseconds);
extern void gpuCudaCombinedWakeRelease(void);
extern int gpuCudaRfmodeHistogram(
  void *coord, long nParticles, int stride, const GPU_RFMODE_DATA *data,
  unsigned long long *histogramReturn, long *binnedCount,
  float *milliseconds);
extern int gpuCudaTrfmodeHistogram(
  void *coord, long nParticles, int stride, const GPU_RFMODE_DATA *data,
  unsigned long long *histogramReturn, double *xsumReturn,
  double *ysumReturn, long *binnedCount, float *milliseconds);
extern int gpuCudaRfmodeTimeCoordinates(
  void *coord, long nParticles, int stride, double pCentral, double cMks,
  double *timeReturn, float *kernelMilliseconds,
  float *transferMilliseconds);
extern int gpuCudaRfmodeApplyKicks(
  void *coord, long nParticles, int stride, const GPU_RFMODE_DATA *data,
  const double *voltage, float *milliseconds);
extern int gpuCudaTrfmodeApplyKicks(
  void *coord, long nParticles, int stride, const GPU_RFMODE_DATA *data,
  const double *voltageX, const double *voltageY,
  const double *voltageZ, float *milliseconds);
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
extern int gpuCudaScmultMoments(void *coord, long nParticles, int stride,
                                GPU_SCMULT_MOMENT_DATA *result,
                                float *milliseconds);
extern void gpuCudaScmultRelease(void);
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
                                double t1Factor,
                                double t2Factor,
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
extern int gpuCudaCsrCsbendOutputRoundTrip(void *coord,
                                           long nParticles,
                                           int stride,
                                           double cosTilt,
                                           double sinTilt,
                                           double cMks,
                                           float *milliseconds);
extern int gpuCudaCsrDriftAdvance(void *coord,
                                  long nParticles,
                                  int stride,
                                  double dz,
                                  double Po,
                                  long linearOptics,
                                  long toTime,
                                  float *milliseconds);
extern int gpuCudaCsrDriftKick(void *coord,
                               long nParticles,
                               int stride,
                               const double *dGamma,
                               long nBins,
                               long applyKick,
                               double ct0,
                               double dct,
                               double Po,
                               long useFactor,
                               double factor,
                               float *milliseconds);
extern int gpuCudaCsrDriftCountInRange(void *coord,
                                       long nParticles,
                                       int stride,
                                       double ct0,
                                       double dct,
                                       long nBins,
                                       long *count,
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
  double *cpuCt;
  long cpuCtCapacity;
  long cpuCtCount;
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
  long particleCapacity;
} GPU_BEAM_SUMS_SCRATCH;

typedef struct GPU_RFCA_SCRATCH {
  void *lostCount;
  void *matchEnergy;
  void *referenceKick;
  void *referenceAux;
  double *hostReferenceKick;
  double *hostReferenceAux;
  long referenceCapacity;
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
static long gpuMomentumBatchScope = 0;
/* OpenMP tracking is deliberately opt-in and is active only while a
 * loss-sensitive batched search is using the CPU fallback. */
static long gpuOmpTrackingThreads = 1;
static long gpuOmpTrackingScope = 0;
static GPU_OMP_TRACKING_WORKSPACE gpuOmpTrackingWorkspace;
static GPU_CSR_SCRATCH gpuCsrScratch;
static GPU_APERTURE_SCRATCH gpuApertureScratch;
static GPU_ACCEPTED_BUFFER gpuAcceptedBuffer;
static GPU_KICKMAP_CACHE gpuKickMapCache;
static GPU_RFCW_KICK_SCRATCH gpuRfcwKickScratch;
static GPU_LSC_SCRATCH gpuLscScratch;
static GPU_BEAM_SUMS_SCRATCH gpuBeamSumsScratch;
static GPU_RFCA_SCRATCH gpuRfcaScratch;
static unsigned long gpuRfcaReferenceCalls = 0;
static unsigned long gpuRfcaReferenceParticles = 0;
static unsigned long gpuIsrReferenceFills = 0;
static unsigned long gpuIsrReferenceDraws = 0;
static unsigned long gpuLscReferenceCalls = 0;
static unsigned long gpuMatchEnergyCpuOrderCalls = 0;
static unsigned long gpuMatchEnergySkippedCalls = 0;
static double *gpuMatchEnergyDelta = NULL;
static long gpuMatchEnergyDeltaCapacity = 0;
/* The beam momentum matching last found already matched: its momenta are kept on the device (see
 * gpuCudaColumnSnapshotTake()), with the particle count and central momentum they went with */
static long gpuMatchEnergyMatchedValid = 0;
static long gpuMatchEnergyMatchedParticles = 0;
static double gpuMatchEnergyMatchedP = 0;
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
static long gpuApertureParallelCompactionOrderDisabled = 0;
static long gpuEnableApertureAcceptedDevice = 0;
static long gpuEnableMagnetLossCompaction = 0;
/* ELEGANT_GPU_TEST_MAGNET_LOSS_DISAGREEMENT=1: gpuMagnetLossRowsFromCpu() acts as if the CPU kept a
 * particle the device lost, so tests reach the whole-element CPU fallback that follows */
static long gpuTestMagnetLossDisagreement = 0;
/* ELEGANT_GPU_ENABLE_SLICE_SUMS=0 brings the beam to the host for slice analysis instead */
static long gpuEnableSliceSums = 0;
static long gpuMagnetLossCompactionExplicit = 0;
static long gpuEnableCsbendDrift = 0;
static long gpuCsbendDriftExplicit = 0;
static long gpuDeviceIslandHasCsbend = 0;
static long gpuEnableCsrTracking = 0;
static long gpuCsrTrackingExplicit = 0;
static long gpuEnableCsrResident = 0;
static long gpuEnableCsrDriftElement = 1;
static long gpuCsrResidentExplicit = 0;
static long gpuEnableScmult = 0;
static long gpuScmultExplicit = 0;
static long gpuEnableWakeTrackingDrift = 0;
static long gpuWakeTrackingDriftExplicit = 0;
static long gpuEnableCombinedWake = 0;
static long gpuCombinedWakeExplicit = 0;
static long gpuEnableCombinedWakeMultibunch = 0;
static long gpuEnableCombinedWakeFft = 0;
/* ZLONGIT and ZTRANSVERSE on the combined wake's kernels; ELEGANT_GPU_ENABLE_ZIMPEDANCE=0 keeps them on
 * the CPU.  ELEGANT_GPU_ENABLE_ZIMPEDANCE_GATHER=0 sends a bunched beam whose rows aren't in bunch
 * order to the CPU instead of gathering it on the device. */
static long gpuEnableZimpedance = 0;
static long gpuEnableZimpedanceGather = 0;
/* IBSCATTER with SMOOTH=1 on the device; ELEGANT_GPU_ENABLE_IBSCATTER=0 keeps it on the CPU */
static long gpuEnableIbs = 0;
static long gpuEnableLrwake = 0;
static long gpuEnableCorrugated = 0;
static long gpuEnableTaperApe = 0;
static long gpuEnableCenter = 0;
static long gpuEnableIdlePass = 0;
static long gpuTrackingPass = -1, gpuTrackingPassOffset = 0;
static long gpuEnableBatchedTuneTracking = 0;
static long gpuBatchedTuneMinParticles = 32;
static long gpuEnableBatchedSearchTracking = 0;
static long gpuBatchedSearchMinParticles = 32;
static long gpuEnablePolynomialSeries = 0;
static long gpuEnableRfdf = 0;
static long gpuRfdfMinParticles = 64;
static long gpuEnableSreffects = 0;
static long gpuSreffectsMinParticles = 64;
static long gpuEnableIlmatrix = 0;
static long gpuIlmatrixMinParticles = 64;
/* The device's phase advances and the host's sines and cosines of them, for ILMATRIX */
static struct {
  void *phase;        /* 2 a particle, then 4 a particle for the sines and cosines */
  double *hostPhase, *hostSinCos;
  long capacity;
} gpuIlmatrixScratch;
static long gpuIlmatrixReferenceCalls = 0;
static long gpuEnableBggexp = 0;
static long gpuEnableBggexpSymplectic = 0;
static long gpuBggexpMinParticles = 64;
static long gpuEnableCwiggler = 0;
static long gpuCwigglerMinParticles = 64;
static long gpuEnableHistogram = 0;
static long gpuHistogramMinParticles = 64;
static long gpuEnableFtable = 0;
static long gpuFtableMinParticles = 64;
static long gpuEnableBmxyz = 0;
static long gpuBmxyzMinParticles = 64;
static long gpuEnableLorentz = 0;
static long gpuLorentzMinParticles = 64;
static long gpuEnableLsrmdltr = 0;
static long gpuLsrmdltrMinParticles = 64;
static long gpuEnableBeambeam = 0;
static long gpuEnableModrf = 0;
static long gpuEnableRftmez0 = 0;
static long gpuEnableTwla = 0;
static long gpuEnableTmcf = 0;
static long gpuEnableCepl = 0;
static long gpuEnableTwpl = 0;
static long gpuEnableTwmta = 0;
static long gpuEnableMapSolenoid = 0;
static long gpuMotionMinParticles = 64;
static long gpuEnableCcbend = 0;
static long gpuCcbendMinParticles = 64;
static long gpuEnableLgbend = 0;
static long gpuLgbendMinParticles = 512;
static long gpuKeepResident = 1;
static long gpuResidentMinParticles = 1000;
static long gpuEnableRfmode = 0;
static long gpuEnableFrfmode = 0;
static long gpuEnableTrfmode = 0;
static long gpuEnableFtrfmode = 0;
static long gpuRfmodeMinParticles = 8192;
static long gpuEnableTfeedback = 0;
static long gpuSingleBunchBeam = 1;
static long gpuAfterElementAperture = 0;
static long gpuTfeedbackMinParticles = 64;
/* The smallest particle-count threshold of any element path, and of the paths that use only the
 * thresholds in gpuBase, from gpuBaseInit() */
static long gpuMinEligibleParticles = 0;
static long gpuMinGeneralEligibleParticles = 0;
static long gpuEnableLscTracking = 0;
static long gpuLscTrackingExplicit = 0;
static long gpuEnableRfcwTrackingDrift = 0;
static long gpuRfcwTrackingDriftExplicit = 0;
#ifdef GPU_VERIFY
static long gpuCpuVerificationActive = 0;
#ifdef GPU_VERIFY
static long gpuVerifyElementTrackedOnGpu = 0; /* this element's CUDA path ran, then its CPU shadow */
#endif
#endif
static long gpuEnableRfcaChangeP0Drift = 0;
static long gpuRfcaChangeP0DriftExplicit = 0;
static long gpuRunAlwaysChangeP0 = 0;
static long gpuAvoidShortGpuIslands = 1;
static long gpuDeviceCodeCheckedDevice = -1;
static int gpuDeviceCodeStatus = 0;
static long gpuShortGpuIslandMaxElements = 4;
static long gpuShortGpuIslandByCost = 1;
static long gpuShortGpuIslandByCostCount = 0;
static long gpuMatrixDriftMinParticlesExplicit = 0;
static long gpuHelperMinParticlesExplicit = 0;
static long gpuMatchEnergyCpuOrderSetting = -1;
static long gpuMagnetMinParticlesExplicit = 0;
static long gpuWakeMinParticlesExplicit = 0;
static long gpuOutputDriftReductionMinParticlesExplicit = 0;
static long gpuEnableOutputBeamSums = 1;
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
static GPU_BEAM_SUM_DATA gpuLastBeamSumResult;
static long gpuLastBeamSumResultValid = 0;
static double gpuLastBeamSumSigma56 = 0;
static unsigned long gpuMatrixSparseTermTotal = 0;
static unsigned long gpuMatrixSparseTermHistogram[GPU_MATRIX_MAX_TERMS + 1];
static unsigned long long *gpuRfmodeHostHistogram;
static long gpuRfmodeHostHistogramCapacity;
static double *gpuRfmodeHostXsum;
static double *gpuRfmodeHostYsum;
static double *gpuRfmodeHostTime;
static long gpuRfmodeHostTimeCapacity;
static unsigned long long *gpuHistogramHostBins;
static long gpuHistogramHostBinCapacity;
static GPU_SCMULT_MOMENT_DATA gpuScmultMomentCache;
static long gpuScmultMomentCacheParticles;
static long gpuScmultMomentCacheValid;

static long gpuPackMatrix(GPU_MATRIX_DATA *packed, VMATRIX *M);
static long gpuMatrixSupported(VMATRIX *M);
static long gpuPackMultipoleTracking(GPU_MULTIPOLE_DATA *data,
                                     ELEMENT_LIST *elem, double Po);
static long gpuPackExactCorrector(GPU_EXACT_CORRECTOR_DATA *data,
                                  ELEMENT_LIST *elem, double pCentral,
                                  double zStart);
static long gpuPackMaxamp(GPU_APERTURE_LIMIT_DATA *data, MAXAMP *maxamp,
                          double cosTilt, double sinTilt);
static long gpuPackTaperAperture(GPU_TAPER_APERTURE_DATA *data,
                                 ELEMENT_LIST *element, double pCentral,
                                 double zStart);
static long gpuPackSpeedbump(GPU_SPEEDBUMP_DATA *data,
                             ELEMENT_LIST *element, double pCentral,
                             double zStart);
static long gpuPackCsbendTracking(GPU_CSBEND_DATA *data, CSBEND *csbend,
                                  double Po);
static long gpuPackCcbendTracking(GPU_CCBEND_DATA *data, ELEMENT_LIST *eptr,
                                  CCBEND *ccbend, double Po);
static long gpuPackLgbendTracking(
  GPU_LGBEND_DATA *data,
  GPU_LGBEND_LOCAL_APERTURE_DATA *localAperture,
  ELEMENT_LIST *eptr, LGBEND *lgbend, double Po);
static void gpuReleaseKickMapCache(void);
static void gpuEnsureRfcwKickScratch(long nParticles);
static void gpuEnsureLscScratch(long bins);
static void gpuEnsureBeamSumsScratch(long nParticles);
static void gpuEnsureRfcaScratch(void);
static void gpuReleasePolynomialSeriesCache(void);
static void gpuRfcwApplyCoordinateOffset(long np, int index, double value,
                                         const char *operation);
static long gpuPassiveElementSupported(ELEMENT_LIST *eptr, long nParticles);
static long gpuElementEligible(ELEMENT_LIST *eptr, long nParticles);
static long gpuShouldUseCpuForShortGpuIsland(ELEMENT_LIST *eptr, long nParticles);
#if !USE_MPI
static long gpuSmallBeamElementCost(ELEMENT_LIST *eptr, long sums, double *cpuNs, double *deviceUs);
#endif
static void gpuFlushPendingExactDrift(const char *reason);
static long gpuCsrCsbendDeviceEntrySupported(ELEMENT_LIST *eptr, long nParticles);
static long gpuMultipoleElementSupported(ELEMENT_LIST *eptr);
static long gpuExactCorrectorElementSupported(ELEMENT_LIST *eptr);
static long gpuCsbendElementSupported(ELEMENT_LIST *eptr);
static long gpuCcbendElementSupported(ELEMENT_LIST *eptr);
static long gpuLgbendElementSupported(ELEMENT_LIST *eptr);
static long gpuWakeElementSupported(ELEMENT_LIST *eptr);
static long gpuTrwakeElementSupported(ELEMENT_LIST *eptr);
static long gpuCombinedWakeElementSupported(ELEMENT_LIST *eptr);
static long gpuLscElementSupported(ELEMENT_LIST *eptr);
static long gpuHistogramElementSupported(ELEMENT_LIST *eptr);
static long gpuHelperElementNeedsReduction(ELEMENT_LIST *eptr);
static long gpuSelectDevice(void);
long gpu_csr_csbend_resident_available(void *csbend0, long nParticles,
                                       long nBins);
static void gpuWakeTrackingWarning(const char *text, const char *detail);
static void gpuSmoothWakeHistogram(double *data, long nb, long order,
                                   long halfWidth, const char *elementType,
                                   const char *inputFile);
static void gpuDisplaySyncTimings(void);
#ifdef GPU_VERIFY
static void gpuCompareWakeArray(const char *label, const char *arrayName,
                                const double *cpu, const double *gpu, long n);
static long gpuComputeCsrHistogramCpu(double *hist, double **part,
                                      long nParticles, long nBins,
                                      double ctLower, double dct);
#endif
void gpu_compute_centroids(double *centroid, long n_part);
/* Element types moved to the device on November 15, 2026 (gpu_ports.h, gpu_ports_host.h at the end) */
static long gpuPortElementEligible(ELEMENT_LIST *eptr, long nParticles, long *eligible);
static void gpuPortReadSettings(void);
static void gpuPortDescribeSettings(void);
/* Set by those types when a survivor's x or y may no longer be finite, which limit_amplitudes() removes */
static long gpuPortInvalidPending = 0;
static long gpuPortInvalidSurvivors(void);
#if !USE_MPI
static void gpuPortSmallBeamCost(ELEMENT_LIST *eptr, double *cpuNs, double *deviceUs);
#endif

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

/* gpuBaseInit() runs at every tracking call and reads over a hundred variables, and getenv() scans the
 * whole environment each time, which slowed decks that track one particle tens of thousands of times.
 * elegant never changes the environment, so each name is looked up once. Every name is a string constant,
 * so its address is the key; a name built in a buffer would need a lookup by its text instead. */
#define GPU_ENV_CACHE_SLOTS 512
static struct {
  const char *name;
  const char *value;
} gpuEnvCache[GPU_ENV_CACHE_SLOTS];

static const char *gpuGetenv(const char *name) {
  unsigned long slot = (unsigned long)(((uintptr_t)name >> 3) * 2654435761u % GPU_ENV_CACHE_SLOTS), i;

  for (i = 0; i < GPU_ENV_CACHE_SLOTS; i++, slot = (slot + 1) % GPU_ENV_CACHE_SLOTS) {
    if (gpuEnvCache[slot].name == name)
      return gpuEnvCache[slot].value;
    if (!gpuEnvCache[slot].name) {
      gpuEnvCache[slot].value = getenv(name);
      gpuEnvCache[slot].name = name;
      return gpuEnvCache[slot].value;
    }
  }
  return getenv(name);
}

static long gpuEnvFlag(const char *name) {
  const char *value = gpuGetenv(name);

  if (!value)
    return 0;
  return strcmp(value, "1") == 0 || strcmp(value, "yes") == 0 ||
         strcmp(value, "true") == 0 || strcmp(value, "on") == 0 ||
         strcmp(value, "YES") == 0 || strcmp(value, "TRUE") == 0 ||
         strcmp(value, "ON") == 0;
}

static long gpuEnvLong(const char *name, long defaultValue) {
  char *endptr = NULL;
  const char *value = gpuGetenv(name);
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
  const char *value = gpuGetenv(name);
  double result;

  if (!value || !*value)
    return defaultValue;
  result = strtod(value, &endptr);
  if (endptr == value)
    return defaultValue;
  return result;
}

static long gpuEnvSet(const char *name) {
  const char *value = gpuGetenv(name);

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

/* ELEGANT_GPU_SYNC_REASONS=1 counts copies by reason and element type, without the
 * element's name, and prints them with the sync totals */
#define GPU_SYNC_REASON_SLOTS 64
static long gpuSyncReasonsEnabled = -1;
static long gpuSyncReasonCount;
static char gpuSyncReasonText[GPU_SYNC_REASON_SLOTS][160];
static long gpuSyncReasonCopies[GPU_SYNC_REASON_SLOTS];

static void gpuRecordSyncReasonCopy(const char *reason) {
  char key[160], *hash, *space;
  long i;

  if (gpuSyncReasonsEnabled < 0)
    gpuSyncReasonsEnabled = getenv("ELEGANT_GPU_SYNC_REASONS") &&
                            atol(getenv("ELEGANT_GPU_SYNC_REASONS")) > 0;
  if (!gpuSyncReasonsEnabled)
    return;
  snprintf(key, sizeof(key), "%s", reason && *reason ? reason : "(none)");
  if ((hash = strchr(key, '#'))) {
    *hash = 0;
    if ((space = strrchr(key, ' ')))
      *space = 0;
  }
  for (i = 0; i < gpuSyncReasonCount; i++)
    if (strcmp(gpuSyncReasonText[i], key) == 0)
      break;
  if (i == gpuSyncReasonCount) {
    if (gpuSyncReasonCount == GPU_SYNC_REASON_SLOTS)
      i = GPU_SYNC_REASON_SLOTS - 1;
    else
      snprintf(gpuSyncReasonText[gpuSyncReasonCount++], sizeof(gpuSyncReasonText[0]), "%s", key);
  }
  gpuSyncReasonCopies[i]++;
}

static void gpuRecordSyncRequest(const char *reason, long copied, long readOnly) {
  gpuBase.gpuSyncRequestCount++;
  if (copied) {
    gpuBase.gpuSyncCopyCount++;
    gpuRecordSyncReasonCopy(reason);
  }
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
  const char *mode = gpuGetenv("ELEGANT_GPU_MODE");

  if (!mode || !*mode)
    return ELEGANT_GPU_DEFAULT_MODE;
  return mode;
}

static long gpuReferenceOutputUsesCpuHelpers(void) {
  return gpuBase.orderSensitiveOutputNeeded && !gpuHelperMinParticlesExplicit;
}

/* ELEGANT_GPU_MATCH_ENERGY_SNAPSHOT=0 makes momentum matching add the beam up on every call again */
static long gpuMatchEnergySnapshotEnabled(void) {
  return !gpuEnvSet("ELEGANT_GPU_MATCH_ENERGY_SNAPSHOT") ||
         gpuEnvFlag("ELEGANT_GPU_MATCH_ENERGY_SNAPSHOT");
}

/* With output in CPU order, momentum matching takes the CPU's sum: do_match_energy's Kahan sum in
 * particle order, of the one coordinate copied to the host.  The particles change on the device.
 * ELEGANT_GPU_MATCH_ENERGY_CPU_ORDER=1 or 0 sets it for every run. */
static long gpuMatchEnergyInCpuOrder(void) {
  if (!KahanPlus)
    return 0;
  if (gpuMatchEnergyCpuOrderSetting >= 0)
    return gpuMatchEnergyCpuOrderSetting;
  return gpuBase.orderSensitiveOutputNeeded;
}

/* CPU-compatible reference values avoid systematic phase shifts over many
 * turns. Particle tracking remains on CUDA. Explicit opt-out retains the
 * device-libm path for workloads with an appropriate numerical envelope. */
static long gpuRfcaReferenceMathEnabled(void) {
  if (gpuEnvSet("ELEGANT_GPU_RFCA_REFERENCE_MATH"))
    return gpuEnvFlag("ELEGANT_GPU_RFCA_REFERENCE_MATH");
  return gpuBase.orderSensitiveOutputNeeded;
}

/* With output in CPU order, WAKE and TRWAKE (alone or in RFCW) add the mean time for fixed bins and TRWAKE's
 * bin sums in particle order, as the CPU does, so their kicks match it exactly.  The host adds the mean from
 * times the device makes, and one device thread per bin adds that bin's particles after a stable sort by
 * bin.  ELEGANT_GPU_WAKE_CPU_ORDER=1 or 0 sets it for every run. */
static long gpuWakeCpuOrderEnabled(void) {
  if (gpuEnvSet("ELEGANT_GPU_WAKE_CPU_ORDER"))
    return gpuEnvFlag("ELEGANT_GPU_WAKE_CPU_ORDER");
  return gpuBase.orderSensitiveOutputNeeded;
}

/* The ISR Gaussian numbers made on the host with gauss_rn_lim(), so they match the CPU's exactly.
 * CUDA's log and sin differ from glibc's by an ulp now and then, which a ring with SREFFECTS turns
 * into a different beam after enough turns.  SREFFECTS does this by default with order-sensitive
 * output, since its kernel is otherwise exact (a deck of nothing but SREFFECTS went from 0.38 s to
 * 4.45 s on one host thread or 1.04 s on 8, against 7.4 s on the CPU).  The magnets' kernels differ
 * from the CPU anyway and draw far more (a ring with ISR on every magnet went from 0.90 s to 1.93 s
 * on 8 threads), so they keep the device's numbers.  ELEGANT_GPU_ISR_REFERENCE_MATH=0 or 1 sets it for every element and run, and
 * ELEGANT_GPU_ISR_REFERENCE_THREADS the host threads (default: see gpuIsrHostThreads()). */
static long gpuIsrReferenceMathEnabled(long defaultSetting) {
  if (gpuEnvSet("ELEGANT_GPU_ISR_REFERENCE_MATH"))
    return gpuEnvFlag("ELEGANT_GPU_ISR_REFERENCE_MATH");
  return defaultSetting;
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
  return gpuMatrixDriftMinParticlesExplicit ?
         "; matrix tracking follows explicit matrix-drift threshold" :
         "";
}

static const char *gpuHelperOutputStatus(void) {
  if (!gpuBase.orderSensitiveOutputNeeded)
    return "";
  return gpuHelperMinParticlesExplicit ?
         "; output helpers follow explicit helper threshold" :
         "; output helpers use CPU by default";
}

static const char *gpuMagnetTrackingStatus(void) {
  return gpuMagnetMinParticlesExplicit ?
         "; magnet tracking follows explicit magnet threshold" :
         "";
}

/* Until November 13, 2026 CSR went to the CPU whenever output had to be in the CPU's order, because
 * the device's CSR bend wasn't elegant's to the last bit: its wake grouped two products differently
 * and its edge kicks took the device's tan().  Both are fixed (see "CSR with output" in
 * PARITY-VALIDATION.md), so the rule is off.  ELEGANT_GPU_CSR_CPU_FOR_ORDERED_OUTPUT=1 brings it back,
 * and ELEGANT_GPU_ENABLE_CSR_DRIFT or ..._CSR_RESIDENT_DRIFT set either way still overrides it. */
static long gpuCsrCpuForOrderedOutput(void) {
  /* Until the CSBEND drift formula of elegant-CPU-bugs-and-fixes.md (Part 4) is committed, elegant's bend
   * body takes asin(), sin() and cos() for each particle and the device's can't be its bits, so the rule is
   * on by default; ELEGANT_GPU_CSR_CPU_FOR_ORDERED_OUTPUT=0 turns it off.  Restore the opt-in with that
   * change. */
  return gpuBase.orderSensitiveOutputNeeded &&
         !gpuCsrTrackingExplicit && !gpuCsrResidentExplicit &&
         (gpuEnvSet("ELEGANT_GPU_CSR_CPU_FOR_ORDERED_OUTPUT") ?
            gpuEnvFlag("ELEGANT_GPU_CSR_CPU_FOR_ORDERED_OUTPUT") : 1);
}

static const char *gpuCsrResidentStatus(void) {
  if (gpuCsrCpuForOrderedOutput())
    return "; CSR resident drift uses CPU for order-sensitive output on request";
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
  if (gpuCsrCpuForOrderedOutput())
    return "; CSR drift tracking uses CPU for order-sensitive output on request";
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
    return gpuWakeTrackingDriftExplicit ?
           "; wake tracking drift paths explicitly enabled" :
           "; wake tracking drift paths enabled by default";
  return gpuWakeTrackingDriftExplicit ?
         "; wake tracking drift paths explicitly disabled" :
         "; wake tracking drift paths disabled by default";
}

static const char *gpuCombinedWakeStatus(void) {
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
    return gpuRfcwTrackingDriftExplicit ?
           "; RFCW tracking drift paths explicitly enabled" :
           "; RFCW tracking drift paths enabled by default";
  return gpuRfcwTrackingDriftExplicit ?
         "; RFCW tracking drift paths explicitly disabled" :
         "; RFCW tracking drift paths disabled by default";
}

static const char *gpuRfcaChangeP0DriftStatus(void) {
  if (gpuEnableRfcaChangeP0Drift)
    return gpuRfcaChangeP0DriftExplicit ?
           "; RF cavity change-p0 drift paths explicitly enabled" :
           "; RF cavity change-p0 drift paths enabled by default";
  return gpuRfcaChangeP0DriftExplicit ?
         "; RF cavity change-p0 drift paths explicitly disabled" :
         "; RF cavity change-p0 drift paths disabled by default";
}

static const char *gpuReductionOutputStatus(void) {
  if (!gpuBase.reductionOutputNeeded)
    return "";
  return gpuOutputDriftReductionMinParticlesExplicit ?
         "; output reductions follow explicit output-drift reduction threshold" :
         gpuEnableOutputBeamSums ?
         "; beam sums per element use the device, other output reductions use the CPU" :
         "; output reductions use CPU by default";
}

static const char *gpuApertureParallelCompactionStatus(void) {
  if (gpuApertureParallelCompactionOrderDisabled)
    return "; aperture parallel compaction disabled for order-sensitive output";
  if (gpuEnableApertureParallelCompaction)
    return gpuApertureParallelCompactionExplicit ?
           "; aperture parallel compaction explicitly enabled" :
           "; aperture device compaction preserves CPU loss order";
  return gpuApertureParallelCompactionExplicit ?
         "; aperture parallel compaction explicitly disabled" : "";
}

static const char *gpuMagnetLossCompactionStatus(void) {
  if (gpuEnableMagnetLossCompaction)
    return gpuMagnetLossCompactionExplicit ?
           "; magnet loss compaction explicitly enabled" :
           "; magnet loss compaction enabled";
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

/* Helpers that match the CPU exactly, so output in CPU order doesn't need the CPU: CENTER, RAMPP, MALIGN, MATR
 * and EMATRIX, whose one reference-time sum uses the CPU's order, and ENERGY, whose matching to the
 * beam's average momentum uses the CPU's sum then. */
static long gpuHelperMatchesCpu(ELEMENT_LIST *eptr) {
  ENERGY *energy;

  switch (eptr->type) {
  case T_CENTER:
    /* gpu_center_beam(): each mean from one column copied to the host, in particle order.  Not in
     * gpu-Pelegant: center_beam()'s sums, and computeTimeCoordinates() even with a stored offset, are
     * collective there, and the master stays on the CPU. */
#if USE_MPI
    return 0;
#else
    return gpuEnableCenter && eptr->p_elem != NULL && KahanPlus;
#endif
  case T_RAMPP:
    return eptr->p_elem != NULL;
  case T_MALIGN:
  case T_MATR:
  case T_EMATRIX:
    return 1;
  case T_ENERGY:
    energy = (ENERGY *)eptr->p_elem;
    return energy && ((!energy->match_beamline && !energy->match_particles) || gpuMatchEnergyInCpuOrder());
  default:
    return 0;
  }
}

static long gpuHelperElementSupported(ELEMENT_LIST *eptr) {
  if (!eptr)
    return 0;
#if USE_MPI
  /* With a distributed beam, these averages are over all the workers on the CPU */
  if (distributedBeam && gpuHelperElementNeedsReduction(eptr))
    return 0;
#endif
  if (gpuHelperMatchesCpu(eptr))
    return 1;
  if (gpuReferenceOutputUsesCpuHelpers())
    return 0;
  switch (eptr->type) {
  case T_ENERGY:
    return 1;
  case T_CENTER:
    return 0;
  default:
    break;
  }
  return 0;
}

static long gpuCompileContourLogic(APCONTOUR *apContour, signed char **ops, long **opStart);

/* Spin goes through the CPU's rotations, and DZ moves particles by their slopes before the check */
/* HOLD_OFF only stops the element itself from checking, which do_tracking() handles.  The CPU's
 * roll of spins and coordinates by TILT and back, and its DZ round trip, change survivors only by
 * rounding */
static long gpuApContourSupported(APCONTOUR *apcontour) {
  if (!apcontour || gpuBase.backtrack)
    return 0;
  if (apcontour->cancel)
    return 1;
  if (!apcontour->initialized) {
    if (!initializeApContour)
      return 0;
    initializeApContour(apcontour);
  }
  if (apcontour->nContours <= 0 || (apcontour->hasLogic && apcontour->nContours == 1))
    return 0;
  /* A HOLD_OFF element only drifts, and the copies checked after later elements have no length */
  if (apcontour->length > 0 && !apcontour->holdOff &&
      (apcontour->nContours > 1 || apcontour->resolution <= 0))
    return 0;
  return !apcontour->hasLogic || gpuCompileContourLogic(apcontour, NULL, NULL);
}

static long gpuApertureElementSupported(ELEMENT_LIST *eptr) {
  SPEEDBUMP *speedbump;

  if (!eptr)
    return 0;
  switch (eptr->type) {
  case T_RCOL:
  case T_ECOL:
  case T_SCRAPER:
    return 1;
  case T_APCONTOUR:
    return gpuApContourSupported((APCONTOUR *)eptr->p_elem);
  case T_TAPERAPC:
    return eptr->p_elem && !spinCoordOffset &&
           ((TAPERAPC *)eptr->p_elem)->length >= 0;
  case T_TAPERAPR:
    /* Until CPU fix 8 is committed (elegant-CPU-bugs-and-fixes.md), the CPU's global loss coordinates for a
     * TAPERAPR are taken in the tilted and offset frame; with them wanted the CPU tracks.  Remove with that fix. */
    if (globalLossCoordOffset > 0)
      return 0;
    return eptr->p_elem && !spinCoordOffset &&
           ((TAPERAPR *)eptr->p_elem)->length >= 0;
  case T_TAPERAPE:
    return gpuEnableTaperApe && eptr->p_elem && !spinCoordOffset &&
           ((TAPERAPE *)eptr->p_elem)->length >= 0;
  case T_SPEEDBUMP:
    if (!eptr->p_elem || spinCoordOffset || gpuBase.backtrack)
      return 0;
    /* Until CPU fix 7 is committed, the CPU's global X for a SPEEDBUMP loss is taken before the particle is
     * moved to the hit point; with global loss coordinates wanted the CPU tracks.  Remove with that fix. */
    if (globalLossCoordOffset > 0)
      return 0;
    speedbump = (SPEEDBUMP *)eptr->p_elem;
    if (!speedbump->insertFrom ||
        (strcmp(speedbump->insertFrom, "x") != 0 &&
         strcmp(speedbump->insertFrom, "+x") != 0 &&
         strcmp(speedbump->insertFrom, "-x") != 0 &&
         strcmp(speedbump->insertFrom, "y") != 0 &&
         strcmp(speedbump->insertFrom, "+y") != 0 &&
         strcmp(speedbump->insertFrom, "-y") != 0))
      return 0;
    return isfinite(speedbump->length) && speedbump->length >= 0 &&
           isfinite(speedbump->chord) && speedbump->chord >= 0 &&
           isfinite(speedbump->height) && speedbump->height >= 0 &&
           isfinite(speedbump->dzCenter) &&
           isfinite(speedbump->position) && isfinite(speedbump->dx) &&
           isfinite(speedbump->dy) &&
           speedbump->length / 2 - speedbump->chord / 2 +
               speedbump->dzCenter >= 0 &&
           speedbump->length / 2 + speedbump->chord / 2 -
               speedbump->dzCenter >= 0;
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

static long gpuFiducialPidRange(unsigned long mode, int64_t *startPID,
                                int64_t *endPID) {
  if (!startPID || !endPID)
    return 0;
  *startPID = *endPID = -1;
  if (getFiducializationPidRange)
    return getFiducializationPidRange(mode, startPID, endPID);
  return 1;
}

static long gpuFiducialModeSupported(unsigned long mode) {
  int64_t startPID = -1, endPID = -1;

  if (!mode)
    return 0;
  if (!gpuFiducialPidRange(mode, &startPID, &endPID))
    return 0;
  if (mode & (FID_MODE_LIGHT | FID_MODE_TMEAN | FID_MODE_FIRST |
              FID_MODE_PMAX))
    return 1;
  return 0;
}

/* ramped_rf_cavity() finds the voltage, frequency, phase and fiducial on the host as the CPU does,
 * with the device's fiducial times */
static long gpuRamprfElementSupported(ELEMENT_LIST *eptr) {
  RAMPRF *ramprf;

#if USE_MPI
  if (distributedBeam)
    return 0;
#endif
  if (!eptr || eptr->type != T_RAMPRF || !eptr->p_elem)
    return 0;
  ramprf = (RAMPRF *)eptr->p_elem;
  if (ramprf->volt == 0 || ramprf->pwaveform || ramprf->fiducial_seen)
    return 1;
  return gpuFiducialModeSupported(gpuRfcwFiducialMode(ramprf->fiducial));
}

/* MODRF: the mean time and the record file stay on the host with the CPU's code (mod_rfca.c); the kicks are
 * RAMPRF's on the device */
static long gpuModrfElementSupported(ELEMENT_LIST *eptr) {
  MODRF *modrf;

#if USE_MPI
  if (distributedBeam)
    return 0;
#endif
  if (!gpuEnableModrf || !eptr || eptr->type != T_MODRF || !eptr->p_elem || !KahanPlus)
    return 0;
  modrf = (MODRF *)eptr->p_elem;
  if (modrf->volt == 0 || modrf->fiducial_seen)
    return 1;
  return gpuFiducialModeSupported(gpuRfcwFiducialMode(modrf->fiducial));
}

static long gpuRfcwRfOnlyElementSupported(ELEMENT_LIST *eptr) {
  RFCW *rfcw;
  double phase = 0;

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
  /* Until CPU fix 18 is committed, the transverse wake with N_BINS stays on the CPU (see
   * gpuTrwakeDataSupported()); decided here, before any device setup.  Remove with that fix. */
  if (hasTrwake && rfcw->n_bins)
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
  /* Until CPU fix 18 is committed, the transverse wake with N_BINS stays on the CPU (see
   * gpuTrwakeDataSupported()); decided here, before any device setup.  Remove with that fix. */
  if (hasTrwake && rfcw->n_bins)
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

/* With a distributed beam, trackRfCavityWithWakes() first sums the particle count over all ranks,
 * as on the master, which stays on the CPU, so every RFCA does too */
static long gpuRfcaRemoveInvalidOnlyElementSupported(ELEMENT_LIST *eptr) {
  RFCA *rfca;

#if USE_MPI
  if (distributedBeam)
    return 0;
#endif
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

static long gpuRfcaNoOpElementSupported(ELEMENT_LIST *eptr) {
  RFCA *rfca;

#if USE_MPI
  if (distributedBeam)
    return 0;
#endif
  if (!eptr || eptr->type != T_RFCA || !eptr->p_elem)
    return 0;
  rfca = (RFCA *)eptr->p_elem;
  /*
   * trackRfCavityWithWakes returns before phase setup or particle access for
   * this case.  Keeping it resident is therefore exact even when FREQ and
   * other inactive RF parameters are present.  Preserve the CPU error for the
   * one option combination checked before that early return.
   */
  return rfca->length == 0 && rfca->volt == 0 &&
         !(rfca->change_t && rfca->Q);
}

static long gpuRfcaThinKickElementSupported(ELEMENT_LIST *eptr) {
  RFCA *rfca;

#if USE_MPI
  if (distributedBeam)
    return 0;
#endif
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
    /* Until CPU fix 5 is committed (elegant-CPU-bugs-and-fixes.md), trackUndulatorKickMap() gives a radiation
     * kick to the particle swapped into a lost one's place, which the device doesn't; with SYNCH_RAD the CPU
     * tracks.  Remove with that fix. */
    if (map->synchRad)
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
  (void)isr;
  (void)tilt;
  (void)dx;
  (void)dy;
  (void)dz;
  /* The CPU's original method ignores pitch and yaw */
  (void)pitch;
  (void)yaw;
  if (malignMethod < 0 || malignMethod > 4)
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
  case T_FMULT: {
    FMULT *fmult = (FMULT *)eptr->p_elem;

    if (!gpuMultipoleCommonSupported(fmult->nSlices, fmult->n_kicks, 4,
                                     fmult->synch_rad, 0, fmult->tilt, 0, 0,
                                     fmult->dx, fmult->dy, fmult->dz, 0, 0))
      return 0;
    if (!fmult->filename || !isfinite(fmult->length) || !isfinite(fmult->tilt) ||
        !isfinite(fmult->dx) || !isfinite(fmult->dy) || !isfinite(fmult->dz) ||
        !isfinite(fmult->fse) || !isfinite(fmult->factor))
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
    if (kquad->edge1_effects < 0 || kquad->edge1_effects > 3 ||
        kquad->edge2_effects < 0 || kquad->edge2_effects > 3 ||
        kquad->radial)
      return 0;
    if (!isfinite(kquad->lEffective) ||
        !isfinite(kquad->edge1NonlinearFactor) ||
        !isfinite(kquad->edge2NonlinearFactor))
      return 0;
    return 1;
  }
  case T_KSEXT: {
    KSEXT *ksext = (KSEXT *)eptr->p_elem;

    if (ksext->synch_rad && ksext->length < 1e-6)
      return 0;
    if (!gpuMultipoleCommonSupported(ksext->nSlices, ksext->n_kicks,
                                     ksext->integration_order,
                                     ksext->synch_rad, ksext->isr,
                                     ksext->tilt, ksext->pitch, ksext->yaw,
                                     ksext->dx, ksext->dy, ksext->dz,
                                     ksext->malignMethod,
                                     ksext->expandHamiltonian))
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
    return 1;
  }
  case T_KQUSE: {
    KQUSE *kquse = (KQUSE *)eptr->p_elem;

    /* MATRIX_TRACKING uses the element's matrix, and the CPU warns as it turns off radiation in a
     * KQUSE shorter than 1e-6 m */
    if (kquse->matrixTracking || kquse->synch_rad || (kquse->isr && kquse->length < 1e-6))
      return 0;
    if (!gpuMultipoleCommonSupported(kquse->nSlices, kquse->n_kicks,
                                     kquse->integration_order,
                                     kquse->synch_rad, kquse->isr,
                                     kquse->tilt, 0, 0,
                                     kquse->dx, kquse->dy, kquse->dz, 0,
                                     kquse->expandHamiltonian))
      return 0;
    return 1;
  }
  default:
    break;
  }

  return 0;
}

static long gpuExactCorrectorElementSupported(ELEMENT_LIST *eptr) {
  short synchRad = 0, isr = 0;
  char *steeringMultipoles = NULL, *randomMultipoles = NULL;
  MULTIPOLE_DATA *steeringData = NULL, *randomData = NULL, *totalData = NULL;

  if (!eptr || !eptr->p_elem || gpuBase.backtrack || spinCoordOffset)
    return 0;
  switch (eptr->type) {
  case T_EHCOR: {
    EHCOR *corrector = (EHCOR *)eptr->p_elem;
    synchRad = corrector->synchRad;
    isr = corrector->isr;
    steeringMultipoles = corrector->steeringMultipoles;
    randomMultipoles = corrector->randomMultipoles;
    steeringData = &corrector->steeringMultipoleData;
    randomData = &corrector->randomMultipoleData;
    totalData = &corrector->totalMultipoleData;
    break;
  }
  case T_EVCOR: {
    EVCOR *corrector = (EVCOR *)eptr->p_elem;
    synchRad = corrector->synchRad;
    isr = corrector->isr;
    steeringMultipoles = corrector->steeringMultipoles;
    randomMultipoles = corrector->randomMultipoles;
    steeringData = &corrector->steeringMultipoleData;
    randomData = &corrector->randomMultipoleData;
    totalData = &corrector->totalMultipoleData;
    break;
  }
  case T_EHVCOR: {
    EHVCOR *corrector = (EHVCOR *)eptr->p_elem;
    synchRad = corrector->synchRad;
    isr = corrector->isr;
    steeringMultipoles = corrector->steeringMultipoles;
    randomMultipoles = corrector->randomMultipoles;
    steeringData = &corrector->steeringMultipoleData;
    randomData = &corrector->randomMultipoleData;
    totalData = &corrector->totalMultipoleData;
    break;
  }
  default:
    return 0;
  }
  if (synchRad || isr || gpuStringSet(steeringMultipoles) ||
      gpuStringSet(randomMultipoles))
    return 0;
  if (gpuMultipoleDataPresent(steeringData) ||
      gpuMultipoleDataPresent(randomData) ||
      gpuMultipoleDataPresent(totalData))
    return 0;
  return 1;
}

static long gpuCsbendCommonSupported(CSBEND *csbend) {
  double rho;

  if (!csbend)
    return 0;
  if (csbend->distributionBasedRadiation ||
      gpuStringSet(csbend->photonOutputFile))
    return 0;
  if (csbend->referenceCorrection || csbend->fseCorrection)
    return 0;
  if (csbend->malignMethod < 0 || csbend->malignMethod > 4)
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
  if (!ccbend || !gpuEnableCcbend)
    return 0;
  if (ccbend->optimized < 0 || ccbend->length <= 0 || ccbend->angle == 0 ||
      ccbend->nSlices <= 0 ||
      (ccbend->integration_order != 2 &&
       ccbend->integration_order != 4 &&
       ccbend->integration_order != 6) ||
      ccbend->fringeModel < -1 || ccbend->fringeModel > 1 ||
      ccbend->edgeOrder > 3)
    return 0;
  if (ccbend->malignMethod < 0 || ccbend->malignMethod > 4)
    return 0;
  if ((ccbend->referenceCorrection & ~3) ||
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

static long gpuLgbendCommonSupported(LGBEND *lgbend) {
  long i;

  /* Momentum-aperture batches use the checked CUDA path even with LGBEND
   * turned off: keeping the complete search ensemble resident avoids thousands
   * of CPU island synchronizations. */
  if (!lgbend || (!gpuEnableLgbend && !gpuMomentumBatchScope) ||
      gpuBase.backtrack)
    return 0;
  if (gpuStringSet(lgbend->apertureDataFile) &&
      !lgbend->localApertureData) {
    if (!readLGBendApertureData)
      return 0;
    readLGBendApertureData(lgbend);
  }
  if (!lgbend->initialized || lgbend->nSegments <= 0 ||
      lgbend->nSegments > GPU_LGBEND_MAX_SEGMENTS || lgbend->nSlices <= 0 ||
      (lgbend->localApertureData &&
       lgbend->nSlices > GPU_LGBEND_MAX_LOCAL_APERTURE_SLICES) ||
      (lgbend->integration_order != 2 &&
       lgbend->integration_order != 4 &&
       lgbend->integration_order != 6) ||
      lgbend->edgeOrder > 3 || lgbend->optimized < 0)
    return 0;
  if (!isfinite(lgbend->predrift) || !isfinite(lgbend->postdrift) ||
      !lgbend->segment || !lgbend->fseOpt || !lgbend->KnDelta)
    return 0;
  for (i = 0; i < lgbend->nSegments; i++) {
    LGBEND_SEGMENT *segment = lgbend->segment + i;
    double denominator;
    if (!isfinite(segment->length) || segment->length <= 0 ||
        !isfinite(segment->angle) || segment->angle <= 0 ||
        !isfinite(segment->entryAngle) || !isfinite(segment->exitAngle) ||
        !isfinite(segment->entryX) || !isfinite(segment->exitX) ||
        !isfinite(segment->K1) || !isfinite(segment->K2) ||
        !isfinite(lgbend->fseOpt[i]) || !isfinite(lgbend->KnDelta[i]) ||
        lgbend->KnDelta[i] == 1)
      return 0;
    denominator = sin(segment->entryAngle) +
                  sin(segment->angle - segment->entryAngle);
    if (!isfinite(denominator) || denominator == 0)
      return 0;
  }
  return 1;
}

static long gpuLgbendElementSupported(ELEMENT_LIST *eptr) {
  if (!eptr || !eptr->p_elem || eptr->type != T_LGBEND)
    return 0;
  /* Until CPU fix 6 is committed (elegant-CPU-bugs-and-fixes.md), integrate_kick_KnL() reports a particle
   * lost at a segment's entrance at the distance the particle before it left, where the device reports the
   * entrance; with loss rows wanted the CPU tracks.  Remove with that fix. */
  if (gpuBase.lossOutputNeeded || globalLossCoordOffset > 0)
    return 0;
  return gpuLgbendCommonSupported((LGBEND *)eptr->p_elem);
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

/* With a distributed beam, the CPU's WAKE and TRWAKE sum their histograms over the workers, and
 * the device's would hold only this worker's particles */
static long gpuWakeElementSupported(ELEMENT_LIST *eptr) {
  if (!gpuEnableWakeTrackingDrift)
    return 0;
#if USE_MPI
  if (distributedBeam)
    return 0;
#endif
  if (!eptr || !eptr->p_elem)
    return 0;
  /* CORGPIPE and CORGPLATES are exactDrift(), a WAKE they build and exactDrift(), all of which go to the
   * device by themselves; only the beam's s range for the table is new (gpu_beam_s_range()) */
  if (eptr->type == T_CORGPIPE) {
    CORGPIPE *pipe = (CORGPIPE *)eptr->p_elem;
    return gpuEnableCorrugated && (pipe->n_bins == 0 || pipe->n_bins >= 2) &&
           (pipe->interpolate == 0 || pipe->interpolate == 1);
  }
  if (eptr->type == T_CORGPLATES) {
    CORGPLATES *plates = (CORGPLATES *)eptr->p_elem;
    return gpuEnableCorrugated && (plates->interpolate == 0 || plates->interpolate == 1);
  }
  if (eptr->type != T_WAKE)
    return 0;
  return gpuWakeDataSupported((WAKE *)eptr->p_elem);
}

static long gpuTrwakeDataSupported(TRWAKE *wake) {
  if (!wake)
    return 0;
  if (wake->n_bins != 0 && wake->n_bins < 2)
    return 0;
  /* Until CPU fix 18 is committed (elegant-CPU-bugs-and-fixes.md), track_through_trwake() centers fixed
   * bins on t = 0 when it finds one bucket, where the device centers them on the bunch.  The CPU counts
   * buckets its own way, so TRWAKE (and an RFCW's) with N_BINS stays on the CPU.  Remove with that fix. */
  if (wake->n_bins)
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
#if USE_MPI
  if (distributedBeam)
    return 0;
#endif
  if (!eptr || !eptr->p_elem || eptr->type != T_TRWAKE)
    return 0;
  return gpuTrwakeDataSupported((TRWAKE *)eptr->p_elem);
}

/* Output in CPU order no longer sends IMPEDANCE and CWAKE to the CPU: with gpuWakeCpuOrderEnabled()
 * the device adds the histograms in particle order, a time-domain wake is convolved directly and an
 * impedance takes the host's FFTs, so the kicks are the CPU's (see gpuTrackImpedanceBucket()). */
static long gpuCombinedWakeElementSupported(ELEMENT_LIST *eptr) {
  if (!gpuEnableCombinedWake || !eptr || !eptr->p_elem)
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
  /* ZLONGIT and ZTRANSVERSE: the device bins and kicks, the host makes the voltages with the CPU's
   * routines (smoothing, realFFT) and writes WAKES.  AREA_WEIGHT and REVERSE_TIME_ORDER stay on the
   * CPU; so do drive exponents above 1 (an ipow histogram) and probe exponents outside 0 to 3 (the
   * device's power is ipow's to the last bit only there). */
  if (eptr->type == T_ZLONGIT) {
    ZLONGIT *zlongit = (ZLONGIT *)eptr->p_elem;
    if (!gpuEnableZimpedance || zlongit->area_weight || zlongit->reverseTimeOrder)
      return 0;
    if (zlongit->interpolate != 0 && zlongit->interpolate != 1)
      return 0;
    return 1;
  }
  if (eptr->type == T_ZTRANSVERSE) {
    ZTRANSVERSE *ztransverse = (ZTRANSVERSE *)eptr->p_elem;
    if (!gpuEnableZimpedance)
      return 0;
    if (ztransverse->interpolate != 0 && ztransverse->interpolate != 1)
      return 0;
    if (ztransverse->xDriveExponent > 1 || ztransverse->yDriveExponent > 1 ||
        ztransverse->xProbeExponent < 0 || ztransverse->xProbeExponent > 3 ||
        ztransverse->yProbeExponent < 0 || ztransverse->yProbeExponent > 3)
      return 0;
    return 1;
  }
  return 0;
}

/* IBSCATTER: the slices, their sums and the emittance inflation on the device, the growth rates on the
 * host with the CPU's routines.  SMOOTH=0 (random kicks in particle order) stays on the CPU, and so do
 * the settings the CPU stops on.  Not in gpu-Pelegant. */
/* LRWAKE in elegant (not gpu-Pelegant); the bunch sums, history and wakes stay the CPU's */
static long gpuLrwakeElementSupported(ELEMENT_LIST *eptr) {
#if USE_MPI
  (void)eptr;
  return 0;
#else
  return gpuEnableLrwake && eptr && eptr->type == T_LRWAKE && eptr->p_elem && set_up_lrwake &&
         lrwakeBunchVoltages;
#endif
}

static long gpuIbsElementSupported(ELEMENT_LIST *eptr) {
#if USE_MPI
  (void)eptr;
  return 0;
#else
  IBSCATTER *IBS;

  if (!gpuEnableIbs || !eptr || eptr->type != T_IBSCATTER || !eptr->p_elem)
    return 0;
  IBS = (IBSCATTER *)eptr->p_elem;
  if (!IBS->smooth || IBS->nslice < 1 || (IBS->bunchedBeamMode && !IBS->isRing))
    return 0;
  return 1;
#endif
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
  if (sreffects->DdeltaRef > 0)
    return 0;
  return 1;
#endif
}

/* ILMATRIX on the device.  Spin tracking, the messages VERBOSITY asks for and global loss coordinates
 * stay with the CPU. */
static long gpuIlmatrixElementSupported(ELEMENT_LIST *eptr) {
#if USE_MPI
  (void)eptr;
  return 0;
#else
  if (!gpuEnableIlmatrix || gpuBase.backtrack)
    return 0;
  if (!eptr || eptr->type != T_ILMATRIX || !eptr->p_elem)
    return 0;
  if (spinCoordOffset || globalLossCoordOffset > 0)
    return 0;
  return !((ILMATRIX *)eptr->p_elem)->verbosity;
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
  TRFMODE *trfmode;
  FTRFMODE *ftrfmode;

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
  if (eptr->type == T_TRFMODE) {
    if (!gpuEnableTrfmode)
      return 0;
    trfmode = (TRFMODE *)eptr->p_elem;
    if (trfmode->binless || trfmode->n_bins < 2 ||
        trfmode->bin_size <= 0 || trfmode->bunchedBeamMode != 1 ||
        gpuStringSet(trfmode->record) ||
        (trfmode->interpolate != 0 && trfmode->interpolate != 1))
      return 0;
    return 1;
  }
  if (eptr->type == T_FTRFMODE) {
    if (!gpuEnableFtrfmode)
      return 0;
    ftrfmode = (FTRFMODE *)eptr->p_elem;
    if (ftrfmode->n_bins < 2 || ftrfmode->bin_size <= 0 ||
        ftrfmode->bunchedBeamMode != 1 ||
        gpuStringSet(ftrfmode->outputFile))
      return 0;
    return 1;
  }
  return 0;
#endif
}

#if !USE_MPI
static long gpuTfeedbackPickupCoordinate(const TFBPICKUP *pickup) {
  if (!pickup || !pickup->plane)
    return -1;
  if (strcmp(pickup->plane, "x") == 0 || strcmp(pickup->plane, "X") == 0)
    return 0;
  if (strcmp(pickup->plane, "y") == 0 || strcmp(pickup->plane, "Y") == 0)
    return 2;
  if (strcmp(pickup->plane, "time") == 0 ||
      strcmp(pickup->plane, "TIME") == 0 ||
      strcmp(pickup->plane, "phase") == 0 ||
      strcmp(pickup->plane, "PHASE") == 0)
    return 4;
  if (strcmp(pickup->plane, "delta") == 0 ||
      strcmp(pickup->plane, "DELTA") == 0)
    return 5;
  return -1;
}

static TFBPICKUP *gpuTfeedbackLinkedPickup(ELEMENT_LIST *eptr,
                                            const char *id) {
  ELEMENT_LIST *cursor;

  if (!eptr || !id || !*id)
    return NULL;
  cursor = eptr;
  while (cursor->pred)
    cursor = cursor->pred;
  for (; cursor; cursor = cursor->succ) {
    if (cursor->type == T_TFBPICKUP && cursor->p_elem) {
      TFBPICKUP *pickup = (TFBPICKUP *)cursor->p_elem;
      if (pickup->ID && strcmp(pickup->ID, id) == 0)
        return pickup;
    }
  }
  return NULL;
}

/* With one bunch, bunched beam mode is the same as without it.  The pickup noise is drawn on the
 * host from random_3 after the device average, as on the CPU. */
static long gpuTfeedbackPickupSupported(const TFBPICKUP *pickup) {
  return pickup && (pickup->bunchedBeamMode == 0 || gpuSingleBunchBeam) &&
         gpuTfeedbackPickupCoordinate(pickup) >= 0;
}
#endif

static long gpuTfeedbackElementSupported(ELEMENT_LIST *eptr) {
#if USE_MPI
  (void)eptr;
  return 0;
#else
  if (!gpuEnableTfeedback || gpuBase.backtrack || !eptr || !eptr->p_elem)
    return 0;
  if (eptr->type == T_TFBPICKUP)
    return gpuTfeedbackPickupSupported((TFBPICKUP *)eptr->p_elem);
  if (eptr->type == T_TFBDRIVER) {
    TFBDRIVER *driver = (TFBDRIVER *)eptr->p_elem;
    TFBPICKUP *pickup = driver->pickup ? driver->pickup :
      gpuTfeedbackLinkedPickup(eptr, driver->ID);
    long coordinate;
    if (!gpuTfeedbackPickupSupported(pickup) ||
        (driver->bunchedBeamMode != 0 && !gpuSingleBunchBeam) || driver->frequency != 0 ||
        driver->RaOverQ > 0 || driver->QLoaded > 0 ||
        driver->gainChargeScale > 0)
      return 0;
    coordinate = gpuTfeedbackPickupCoordinate(pickup);
    if (!driver->longitudinal && coordinate != 0 && coordinate != 2)
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
  if (bgg->synchRad || bgg->isr || bgg->particleOutputFile ||
      bgg->dx || bgg->dy || bgg->dz || bgg->tilt || bgg->zInterval <= 0)
    return 0;
  /* SYMPLECTIC=1 also takes a bend's frame and Z_INTERVAL */
  if (bgg->symplectic ? !gpuEnableBggexpSymplectic
                      : (bgg->isBend || bgg->zInterval != 1))
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
  long hasHorizontal, hasVertical, hasByFile, hasBxFile;

  if (!gpuEnableCwiggler || gpuBase.backtrack)
    return 0;
  if (!eptr || eptr->type != T_CWIGGLER || !eptr->p_elem)
    return 0;
  cwiggler = (CWIGGLER *)eptr->p_elem;
  hasByFile = cwiggler->ByFile && cwiggler->ByFile[0];
  hasBxFile = cwiggler->BxFile && cwiggler->BxFile[0];
  if (cwiggler->length <= 0 ||
      cwiggler->periods <= 0 || cwiggler->stepsPerPeriod <= 0 ||
      cwiggler->stepsPerPeriod % 4 ||
      (cwiggler->integrationOrder != 2 && cwiggler->integrationOrder != 4))
    return 0;
  if (cwiggler->sinusoidal) {
    if (cwiggler->BySplitPole || cwiggler->BxSplitPole || hasByFile || hasBxFile)
      return 0;
    hasHorizontal = !cwiggler->vertical || cwiggler->helical;
    hasVertical = cwiggler->vertical || cwiggler->helical;
  } else {
    if ((!hasByFile && !hasBxFile) || cwiggler->vertical || cwiggler->helical)
      return 0;
    hasHorizontal = hasByFile;
    hasVertical = hasBxFile;
  }
  /* With GAP, the field comes from the undulator model in UpdateCWigglerFields() */
  if (cwiggler->gap <= 0 &&
      ((hasHorizontal && !cwiggler->BMax && !cwiggler->ByMax) ||
       (hasVertical && !cwiggler->BMax && !cwiggler->BxMax)))
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
  /* VERBOSE writes the phase space between kicks */
  if (!ftable->initialized || ftable->verbose || ftable->nKicks < 1 ||
      ftable->length <= 0 || !isfinite(ftable->factor) ||
      !isfinite(ftable->threshold) || ftable->threshold < 0)
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

static long gpuLorentzElementSupported(ELEMENT_LIST *eptr) {
#if USE_MPI
  (void)eptr;
  return 0;
#else
  if (!gpuEnableLorentz || gpuBase.backtrack || !eptr || !eptr->p_elem)
    return 0;
  switch (eptr->type) {
  case T_BMAPXY: {
    BMAPXY *bmapxy = (BMAPXY *)eptr->p_elem;
    if (!bmapxy->filename || !bmapxy->filename[0] ||
        bmapxy->FxRpn || bmapxy->FyRpn || !bmapxy->method ||
        strcmp(bmapxy->method, "non-adaptive runge-kutta") != 0 ||
        !isfinite(bmapxy->length) || bmapxy->length <= 0 ||
        !isfinite(bmapxy->accuracy) || bmapxy->accuracy < 1e-6 ||
        bmapxy->accuracy > 0.1 || !isfinite(bmapxy->strength))
      return 0;
    if (!bmapxy->points && !bmapxy_field_setup)
      return 0;
    if (!bmapxy->points)
      bmapxy_field_setup(bmapxy);
    if (!bmapxy->Fx || !bmapxy->Fy || bmapxy->nx < 2 ||
        bmapxy->ny < 2 || bmapxy->dx <= 0 || bmapxy->dy <= 0 ||
        bmapxy->nx > LONG_MAX / bmapxy->ny ||
        bmapxy->nx * bmapxy->ny != bmapxy->points)
      return 0;
    return 1;
  }
  case T_NIBEND: {
    NIBEND *nibend = (NIBEND *)eptr->p_elem;
    if (!nibend->method ||
        strcmp(nibend->method, "non-adaptive runge-kutta") != 0 ||
        !nibend->model || strcmp(nibend->model, "hard-edge") != 0 ||
        !isfinite(nibend->length) || nibend->length <= 0 ||
        !isfinite(nibend->angle) || nibend->angle <= 0 ||
        !isfinite(nibend->accuracy) || nibend->accuracy < 1e-6 ||
        nibend->accuracy > 0.1 || nibend->synch_rad ||
        nibend->adjustBoundary || nibend->adjustField ||
        nibend->fudgePathLength || nibend->tilt || nibend->etilt ||
        nibend->dx || nibend->dy || nibend->dz ||
        (nibend->fint && nibend->hgap) || !isfinite(nibend->fse) ||
        nibend->e1Index < 0 || nibend->e1Index > 1 ||
        nibend->e2Index < 0 || nibend->e2Index > 1)
      return 0;
    return isfinite(nibend->e[nibend->e1Index]) &&
           isfinite(nibend->e[nibend->e2Index]);
  }
  case T_NISEPT: {
    NISEPT *nisept = (NISEPT *)eptr->p_elem;
    if (!nisept->method ||
        strcmp(nisept->method, "non-adaptive runge-kutta") != 0 ||
        !nisept->model || strcmp(nisept->model, "linear") != 0 ||
        !isfinite(nisept->length) || nisept->length <= 0 ||
        !isfinite(nisept->angle) || nisept->angle <= 0 ||
        !isfinite(nisept->e1) || !isfinite(nisept->b1) ||
        !isfinite(nisept->q1_ref) || !isfinite(nisept->flen) ||
        nisept->flen < 0 || !isfinite(nisept->accuracy) ||
        nisept->accuracy < 1e-6 || nisept->accuracy > 0.1)
      return 0;
    return lorentz_setup != NULL;
  }
  default:
    return 0;
  }
#endif
}

/* LSRMDLTR as motion() tracks it with the non-adaptive Runge-Kutta integrator. The other integrators
 * stay on the CPU, as does anything motion() would reject, so that the CPU gives the message. */
static long gpuLsrmdltrElementSupported(ELEMENT_LIST *eptr) {
#if USE_MPI
  (void)eptr;
  return 0;
#else
  static char *gpuMotionMethods[4] = {(char *)"runge-kutta", (char *)"bulirsch-stoer",
                                      (char *)"non-adaptive runge-kutta", (char *)"modified midpoint"};
  static char *gpuLsrmdltrFieldExpansions[3] = {(char *)"ideal", (char *)"exact", (char *)"leading terms"};
  LSRMDLTR *lsr;
  long fieldCode;

  if (!gpuEnableLsrmdltr || gpuBase.backtrack || !eptr || eptr->type != T_LSRMDLTR ||
      !eptr->p_elem || !setupLaserModulator || !findFiducialTime || !KahanPlus)
    return 0;
  lsr = (LSRMDLTR *)eptr->p_elem;
  if (!lsr->method || match_string(lsr->method, gpuMotionMethods, 4, 0) != 2 ||
      !lsr->fieldExpansion ||
      (fieldCode = match_string(lsr->fieldExpansion, gpuLsrmdltrFieldExpansions, 3, 0)) < 0 ||
      (fieldCode != 1 && lsr->tguGradient != 0) || lsr->nSteps < 2 ||
      lsr->periods < 1 || !isfinite(lsr->length) || lsr->length <= 0 || !isfinite(lsr->Bu) ||
      lsr->laserM < 0 || lsr->laserM > 4 || lsr->laserN < 0 || lsr->laserN > 4)
    return 0;
  return 1;
#endif
}

/* RFTMEZ0, TWLA, MAPSOLENOID, TMCF, CEPL, TWPL and TWMTA as motion() tracks them with the non-adaptive Runge-Kutta
 * integrator; the other integrators stay on the CPU, as does anything motion() would reject */
static long gpuMotionElementSupported(ELEMENT_LIST *eptr) {
#if USE_MPI
  (void)eptr;
  return 0;
#else
  static char *gpuMotionMethods[4] = {(char *)"runge-kutta", (char *)"bulirsch-stoer",
                                      (char *)"non-adaptive runge-kutta", (char *)"modified midpoint"};
  char *method;
  long nSteps;

  if (gpuBase.backtrack || !eptr || !eptr->p_elem || !motionGpuSetup || !KahanPlus)
    return 0;
  switch (eptr->type) {
  case T_RFTMEZ0:
    if (!gpuEnableRftmez0)
      return 0;
    method = ((RFTMEZ0 *)eptr->p_elem)->method;
    nSteps = ((RFTMEZ0 *)eptr->p_elem)->n_steps;
    break;
  case T_TWLA:
    if (!gpuEnableTwla)
      return 0;
    method = ((TW_LINAC *)eptr->p_elem)->method;
    nSteps = ((TW_LINAC *)eptr->p_elem)->n_steps;
    break;
  case T_MAPSOLENOID:
    if (!gpuEnableMapSolenoid)
      return 0;
    method = ((MAP_SOLENOID *)eptr->p_elem)->method;
    nSteps = ((MAP_SOLENOID *)eptr->p_elem)->n_steps;
    break;
  case T_TMCF:
    if (!gpuEnableTmcf)
      return 0;
    method = ((TMCF_MODE *)eptr->p_elem)->method;
    nSteps = ((TMCF_MODE *)eptr->p_elem)->n_steps;
    break;
  case T_CEPL:
    if (!gpuEnableCepl)
      return 0;
    method = ((CE_PLATES *)eptr->p_elem)->method;
    nSteps = ((CE_PLATES *)eptr->p_elem)->n_steps;
    break;
  case T_TWPL:
    if (!gpuEnableTwpl)
      return 0;
    method = ((TW_PLATES *)eptr->p_elem)->method;
    nSteps = ((TW_PLATES *)eptr->p_elem)->n_steps;
    break;
  case T_TWMTA:
    if (!gpuEnableTwmta)
      return 0;
    method = ((TWMTA *)eptr->p_elem)->method;
    nSteps = ((TWMTA *)eptr->p_elem)->n_steps;
    break;
  default:
    return 0;
  }
  if (!method || match_string(method, gpuMotionMethods, 4, 0) != 2 || nSteps < 2)
    return 0;
  return 1;
#endif
}

/* BEAMBEAM: a kick a particle from a fixed opposing beam, no sums, losses or random numbers, so also in
 * gpu-Pelegant.  Settings the CPU would reject stay there for its message. */
static long gpuBeambeamElementSupported(ELEMENT_LIST *eptr) {
  BEAMBEAM *bb;

  if (!gpuEnableBeambeam || !eptr || eptr->type != T_BEAMBEAM || !eptr->p_elem)
    return 0;
  bb = (BEAMBEAM *)eptr->p_elem;
  if (!bb->distribution || match_string(bb->distribution, beamBeamDistributionOption, N_BEAM_BEAM_DISTRIBUTIONS, 0) < 0 ||
      !(bb->size[0] > 0) || !(bb->size[1] > 0) || !isfinite(bb->size[0]) || !isfinite(bb->size[1]) ||
      !isfinite(bb->charge) || !isfinite(bb->centroid[0]) || !isfinite(bb->centroid[1]))
    return 0;
  return 1;
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

#if USE_MPI
  /* The CUDA histogram/variance path is local to a rank. The CPU element
   * performs collective reductions, including master participation. Mixing
   * these paths deadlocks and would not give a global space-charge field. */
  if (distributedBeam)
    return 0;
#endif
  if (!gpuEnableLscTracking)
    return 0;
  if (!eptr || !eptr->p_elem || eptr->type != T_LSCDRIFT)
    return 0;
  /* Output in CPU order no longer sends LSCDRIFT to the CPU: the histogram is a count, the voltage is
   * made on the host with the CPU's FFT, and gpuLscCenteredVariances() then takes the beam size from
   * the CPU's sums, so the kicks are the CPU's (4,000,000 particles of lsc4 checked bit for bit) */
  lsc = (LSCDRIFT *)eptr->p_elem;
  if (gpuBase.backtrack || lsc->backtrack)
    return 0;
  return gpuLscDataSupported(lsc);
}

static long gpuHistogramElementSupported(ELEMENT_LIST *eptr) {
#if USE_MPI
  (void)eptr;
  return 0;
#else
  HISTOGRAM *histogram;

  if (!gpuEnableHistogram || !eptr || eptr->type != T_HISTOGRAM ||
      !eptr->p_elem)
    return 0;
  histogram = (HISTOGRAM *)eptr->p_elem;
  if (histogram->disable || histogram->bins < 2)
    return 0;
  return 1;
#endif
}

static long gpuExactDriftElement(long type) {
  return type == T_EDRIFT;
}

/* With a distributed beam, track_through_driftCSR() sums the particle count over all ranks before
 * it finds nothing to do */
static long gpuCsrDriftElementSupported(ELEMENT_LIST *eptr) {
  CSRDRIFT *csrDrift;

  if (!gpuEnableCsrResident)
    return 0;
#if USE_MPI
  if (distributedBeam)
    return 0;
#endif
  if (gpuCsrCpuForOrderedOutput())
    return 0;
  if (!eptr || eptr->type != T_CSRDRIFT || !eptr->p_elem)
    return 0;
  csrDrift = (CSRDRIFT *)eptr->p_elem;
  if (csrDrift->LSCBins)
    return 0;
  if (!csrDrift->csr)
    return 1;
#if USE_MPI
  /* csbend.c builds the device steps of a drift with CSR only without MPI */
  return 0;
#else
  /* Since November 13, 2026 a drift with CSR on stays on the device too: the host computes the wake
   * as before and the device does the loops over particles, in the CPU's arithmetic.
   * ELEGANT_GPU_ENABLE_CSR_DRIFT_ELEMENT=0 sends these drifts back to the CPU. */
  return gpuEnableCsrDriftElement && track_through_driftCSR_cuda_resident_entry &&
         csrDriftHasWakeToApply;
#endif
}

static long gpuParticleCountMeetsThreshold(long nParticles, long minParticles) {
  return minParticles <= 0 || nParticles >= minParticles;
}

static long gpuParticleCountAllowed(long nParticles, long minParticles) {
  return gpuBase.requiredMode ||
         gpuParticleCountMeetsThreshold(nParticles, minParticles);
}

/* A beam already on the device stays there below the general thresholds, down to
 * ELEGANT_GPU_MIN_RESIDENT_PARTICLES, because sending it back costs a copy each way.
 * Without this, a beam that losses took below 10,000 particles was copied around each
 * LGBEND, which kept it on the device.  ELEGANT_GPU_ENABLE_KEEP_RESIDENT=0 restores the
 * old rule. */
static long gpuResidentParticleCountAllowed(long nParticles, long minParticles) {
  if (gpuParticleCountAllowed(nParticles, minParticles))
    return 1;
  return gpuKeepResident && gpuBase.deviceCurrent &&
         gpuParticleCountMeetsThreshold(nParticles, gpuResidentMinParticles);
}

static long gpuMatrixParticleCountAllowed(long nParticles) {
  return gpuResidentParticleCountAllowed(nParticles, gpuBase.matrixMinParticles);
}

static long gpuMagnetParticleCountAllowed(long nParticles) {
  return gpuResidentParticleCountAllowed(nParticles, gpuBase.magnetMinParticles);
}

static long gpuCsbendDriftActive(void) {
  return gpuEnableCsbendDrift &&
         gpuDeviceIslandHasCsbend;
}

static long gpuCsbendDriftMatrixParticleCountAllowed(long nParticles) {
  if (gpuBase.orderSensitiveOutputNeeded &&
      !gpuMatrixDriftMinParticlesExplicit &&
      gpuCsbendDriftActive())
    return gpuResidentParticleCountAllowed(nParticles, gpuBase.matrixMinParticles);
  return gpuMatrixParticleCountAllowed(nParticles);
}

/* Device exact drifts match the CPU exactly, so output in CPU order doesn't send a beam on the
 * device back for them */
static long gpuExactDriftParticleCountAllowed(long nParticles) {
  if (!gpuEnableExactDrift)
    return 0;
  return gpuResidentParticleCountAllowed(nParticles, gpuBase.exactDriftMinParticles);
}

static long gpuCsbendDriftExactDriftParticleCountAllowed(long nParticles) {
  if (gpuBase.orderSensitiveOutputNeeded &&
      !gpuExactDriftExplicit &&
      gpuCsbendDriftActive())
    return gpuResidentParticleCountAllowed(nParticles,
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

/* With a distributed beam, the CPU's SCMULT finds the beam size over all the workers, and the
 * device's moments would be this worker's alone */
static long gpuScmultAllowed(long nParticles) {
  if (!gpuEnableScmult)
    return 0;
#if USE_MPI
  if (distributedBeam)
    return 0;
#endif
  if (!gpuBase.initialized || gpuBase.activeDevice < 0 || nParticles <= 0)
    return 0;
  if (!gpuParticleCountAllowed(nParticles, gpuBase.scmultMinParticles))
    return 0;
  if (!gpuBase.coord || !gpuBase.coord[0])
    return 0;
  return gpuSelectDevice();
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
  case T_MRADINTEGRALS: /* do_tracking() does nothing for RIMULT */
  case T_CHARGE:
  case T_FLOORELEMENT:
  case T_MARK:
  case T_MAXAMP:
  case T_RECIRC:
  case T_TSCATTER:
  case T_TRCOUNT:
  case T_WATCH:
    return 1;
  case T_RFCA:
    return gpuRfcaNoOpElementSupported(eptr);
  case T_SCMULT:
    return gpuScmultAllowed(nParticles);
  default:
    break;
  }
  return 0;
}

/* These options use random numbers in the CPU's particle order, per-particle records or spin,
 * which plain tracking handles and batched programs don't */
static long gpuElementNeedsPlainTracking(ELEMENT_LIST *eptr) {
  if (!eptr || !eptr->p_elem)
    return 0;
  switch (eptr->type) {
  case T_CCBEND: {
    CCBEND *ccbend = (CCBEND *)eptr->p_elem;
    return spinCoordOffset || ccbend->isr || gpuStringSet(ccbend->centroidOutputFile) ||
           ccbend->centroidsRequested || ccbend->SDDScen;
  }
  case T_LGBEND: {
    LGBEND *lgbend = (LGBEND *)eptr->p_elem;
    return spinCoordOffset || lgbend->isr || gpuStringSet(lgbend->centroidOutputFile) ||
           lgbend->centroidsRequested || lgbend->SDDScen;
  }
  case T_CWIGGLER: {
    CWIGGLER *cwiggler = (CWIGGLER *)eptr->p_elem;
    return cwiggler->isr || gpuStringSet(cwiggler->fieldOutput);
  }
  case T_CSBEND:
    return ((CSBEND *)eptr->p_elem)->isr;
  case T_KQUAD:
    return ((KQUAD *)eptr->p_elem)->isr;
  case T_KSEXT:
    return ((KSEXT *)eptr->p_elem)->isr;
  case T_KOCT:
    return ((KOCT *)eptr->p_elem)->isr;
  case T_DQCOR:
    return ((DQCOR *)eptr->p_elem)->isr;
  case T_KQUSE:
    return ((KQUSE *)eptr->p_elem)->isr;
  default:
    return 0;
  }
}

static long gpuElementEligible(ELEMENT_LIST *eptr, long nParticles) {
  if (gpuTrackingSuppressed || !gpuBase.initialized ||
      gpuBase.activeDevice < 0 || !eptr || nParticles <= 0)
    return 0;
  if (nParticles >= LONG_MAX / 4 && gpuElementNeedsPlainTracking(eptr))
    return 0;
  if (gpuSpecialMatrixElementUsesCpuAfterTrack(eptr))
    return 0;
  if (gpuHelperElementSupported(eptr)) {
    if (!gpuResidentParticleCountAllowed(nParticles, gpuBase.helperMinParticles))
      return 0;
    if (gpuHelperElementNeedsReduction(eptr) &&
        !gpuResidentParticleCountAllowed(nParticles, gpuBase.reductionMinParticles))
      return 0;
    return 1;
  }
  if (gpuApertureElementSupported(eptr))
    return gpuResidentParticleCountAllowed(nParticles, gpuBase.apertureMinParticles);
  if (gpuRfcaRemoveInvalidOnlyElementSupported(eptr))
    return gpuResidentParticleCountAllowed(nParticles, gpuBase.apertureMinParticles);
  if (gpuRfcaThinKickElementSupported(eptr))
    return gpuResidentParticleCountAllowed(nParticles, gpuBase.helperMinParticles);
  if (gpuRamprfElementSupported(eptr))
    return gpuResidentParticleCountAllowed(nParticles, gpuBase.helperMinParticles);
  if (gpuModrfElementSupported(eptr))
    return gpuResidentParticleCountAllowed(nParticles, gpuBase.helperMinParticles);
  if (gpuRfcaRfOnlyMatrixElementSupported(eptr))
    return gpuResidentParticleCountAllowed(nParticles, gpuBase.helperMinParticles);
  if (gpuRfcaRfOnlyKickElementSupported(eptr))
    return gpuResidentParticleCountAllowed(nParticles, gpuBase.helperMinParticles);
  if (gpuRfcwRfOnlyElementSupported(eptr))
    return gpuResidentParticleCountAllowed(nParticles, gpuBase.helperMinParticles);
  if (gpuRfcwRfOnlyKickElementSupported(eptr))
    return gpuResidentParticleCountAllowed(nParticles, gpuBase.helperMinParticles);
  if (gpuRfcwMatrixWakeElementSupported(eptr))
    return gpuWakeParticleCountAllowed(nParticles);
  if (gpuRfcwKickWakeElementSupported(eptr))
    return gpuWakeParticleCountAllowed(nParticles);
  if (gpuKickMapElementSupported(eptr))
    return gpuMagnetParticleCountAllowed(nParticles);
  if (gpuExactCorrectorElementSupported(eptr))
    return gpuMagnetParticleCountAllowed(nParticles);
  if (gpuCcbendElementSupported(eptr))
    return gpuParticleCountMeetsThreshold(nParticles,
                                          gpuCcbendMinParticles);
  if (gpuLgbendElementSupported(eptr))
    return gpuParticleCountMeetsThreshold(nParticles,
                                          gpuLgbendMinParticles);
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
    return gpuParticleCountMeetsThreshold(nParticles, gpuSreffectsMinParticles) &&
           gpuMagnetParticleCountAllowed(nParticles);
  if (gpuIlmatrixElementSupported(eptr))
    return gpuParticleCountMeetsThreshold(nParticles, gpuIlmatrixMinParticles) &&
           gpuMagnetParticleCountAllowed(nParticles);
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
  if (gpuLorentzElementSupported(eptr))
    return gpuParticleCountMeetsThreshold(nParticles,
                                          gpuLorentzMinParticles);
  if (gpuLsrmdltrElementSupported(eptr))
    return gpuParticleCountMeetsThreshold(nParticles,
                                          gpuLsrmdltrMinParticles);
  if (gpuBeambeamElementSupported(eptr))
    return gpuMagnetParticleCountAllowed(nParticles);
  if (gpuMotionElementSupported(eptr))
    return gpuParticleCountMeetsThreshold(nParticles, gpuMotionMinParticles);
  if (gpuRfmodeElementSupported(eptr))
    return gpuParticleCountMeetsThreshold(nParticles,
                                          gpuRfmodeMinParticles);
  if (gpuTfeedbackElementSupported(eptr))
    return gpuParticleCountMeetsThreshold(nParticles,
                                          gpuTfeedbackMinParticles);
  if (gpuWakeElementSupported(eptr))
    return gpuWakeParticleCountAllowed(nParticles);
  if (gpuTrwakeElementSupported(eptr))
    return gpuWakeParticleCountAllowed(nParticles);
  if (gpuCombinedWakeElementSupported(eptr))
    return gpuWakeParticleCountAllowed(nParticles);
  if (gpuIbsElementSupported(eptr))
    return gpuWakeParticleCountAllowed(nParticles);
  if (gpuLrwakeElementSupported(eptr))
    return gpuWakeParticleCountAllowed(nParticles);
  if (gpuLscElementSupported(eptr))
    return gpuParticleCountAllowed(nParticles, gpuBase.lscMinParticles);
  if (gpuHistogramElementSupported(eptr))
    return gpuParticleCountMeetsThreshold(nParticles,
                                          gpuHistogramMinParticles);
  if (gpuCsrCsbendDeviceEntrySupported(eptr, nParticles))
    return gpuParticleCountAllowed(nParticles, gpuBase.csrMinParticles);
  if (gpuCsrDriftElementSupported(eptr))
    return gpuParticleCountAllowed(nParticles, gpuBase.csrMinParticles);
  if (gpuSimpleMatrixElement(eptr))
    return gpuCsbendDriftMatrixParticleCountAllowed(nParticles);
  if (gpuExactDriftElement(eptr->type))
    return gpuCsbendDriftExactDriftParticleCountAllowed(nParticles);
  {
    long eligible;
    if (gpuPortElementEligible(eptr, nParticles, &eligible))
      return eligible;
  }
  return 0;
}

long gpuGetTrackingSuppressed(void) { return gpuTrackingSuppressed; }

/* index_bunch_assignments() finds one bunch when there are no ID slots, or when every particle has
 * the same bunch index.  Losses can't add a bunch. */
void gpuSetAfterElementAperture(long active) { gpuAfterElementAperture = active ? 1 : 0; }

void gpuSetBunchAssignments(double **coord, long np, long idSlotsPerBunch) {
  long ip;

  gpuSingleBunchBeam = 1;
  if (idSlotsPerBunch <= 0)
    return;
  if (!coord || bunchIndex >= totalPropertiesPerParticle) {
    gpuSingleBunchBeam = 0;
    return;
  }
  for (ip = 1; ip < np; ip++)
    if ((long)coord[ip][bunchIndex] != (long)coord[0][bunchIndex]) {
      gpuSingleBunchBeam = 0;
      return;
    }
}

void gpu_momentum_search_batch_scope(long active) {
  gpuMomentumBatchScope = active ? 1 : 0;
}

extern int gpuCudaMemoryAvailable(unsigned long long *bytes);
unsigned long gpu_momentum_search_batch_capacity(long turns, long stride) {
  unsigned long long available = 0, limit = 512ULL * 1024 * 1024, perParticle;
  long override;
  if (turns <= 0 || stride <= 0 || gpuBase.activeDevice < 0 || !gpuSelectDevice() ||
      gpuCudaMemoryAvailable(&available) != 0)
    return 0;
  if (available / 4 < limit)
    limit = available / 4;
  /* Host+device histories, coordinates, target/delta arrays, and bookkeeping.
   * Conservatively count both copies against the same allocation budget. */
  perParticle = 8ULL * (10ULL * turns + 4ULL * stride + 32);
  override = gpuEnvLong("ELEGANT_GPU_MOMENTUM_BATCH_PARTICLES", LONG_MAX);
  if (override < 1)
    return 0;
  return (unsigned long)MIN(limit / perParticle, (unsigned long long)override);
}

unsigned long gpu_momentum_ensemble_batch_capacity(long stride) {
  unsigned long long available = 0, limit = 512ULL * 1024 * 1024;
  unsigned long long perParticle;
  long override;

  if (stride <= 0 ||
      (unsigned long long)stride > (ULLONG_MAX - 16ULL) / 2ULL)
    return 0;
  override = gpuEnvLong("ELEGANT_GPU_MOMENTUM_BATCH_PARTICLES", 65536);
  if (override < 1)
    return 0;
  /* Coordinates are present on the host and, when CUDA is available, on the
   * device.  Leave ample room for loss-compaction and element scratch. */
  perParticle = 8ULL * (2ULL * (unsigned long long)stride + 16ULL);
  if (gpuBase.activeDevice >= 0 && gpuSelectDevice() && gpuCudaMemoryAvailable(&available) == 0) {
    if (available / 4 < limit)
      limit = available / 4;
    return (unsigned long)MIN(limit / perParticle,
                              (unsigned long long)override);
  }
  return (unsigned long)override;
}

/* Deliberately separate from the dynamic-aperture/tune admission policies.
 * Classical radiation and high-order CSBEND maps are allowed here only with
 * CPU boundary certification. Stochastic and shared mutable state is not. */
long gpu_momentum_search_beamline_supported(void *beamline0, const char **reason) {
  LINE_LIST *beamline = beamline0;
  ELEMENT_LIST *eptr;
  *reason = NULL;
  if (!gpuBase.initialized || gpuBase.activeDevice < 0 || gpuBase.backtrack || spinCoordOffset) {
    *reason = "CUDA unavailable, backtracking, or spin tracking";
    return 0;
  }
  for (eptr = beamline->elem; eptr; eptr = eptr->succ) {
    long supported = 0;
    const char *restriction = "unsupported state/options";
    if (eptr->ignore)
      continue;
    switch (eptr->type) {
    case T_MARK: case T_RECIRC: case T_MAXAMP:
      /* MAXAMP updates per-tracking-call aperture limits, not element state.
       * It is passive and deliberately has no standalone CUDA kernel. */
      supported = 1;
      break;
    case T_RFCA: {
      RFCA *rf = (RFCA *)eptr->p_elem;
      if (rf && !rf->change_p0 && !rf->change_t && rf->Q == 0)
        supported = gpuRfcaNoOpElementSupported(eptr) ||
                    gpuElementEligible(eptr, LONG_MAX / 4);
      break;
    }
    case T_LGBEND: {
      long saved = gpuEnableLgbend;
      /* A disabled CUDA LGBEND remains a deterministic batched CPU element. */
      gpuEnableLgbend = 1;
      supported = gpuLgbendElementSupported(eptr) && !gpuElementNeedsPlainTracking(eptr);
      gpuEnableLgbend = saved;
      break;
    }
    case T_MALIGN: {
      MALIGN *malign = (MALIGN *)eptr->p_elem;
      restriction = "particle-ID-dependent MALIGN";
      if (malign && malign->startPID == -1 && malign->endPID == -1)
        supported = gpuElementEligible(eptr, LONG_MAX / 4);
      break;
    }
    case T_SCRAPER: {
      SCRAPER *scraper = (SCRAPER *)eptr->p_elem;
      restriction = "SCRAPER material interaction changes random-number consumption";
      if (scraper && !(scraper->length && (scraper->Xo || scraper->Z)))
        supported = gpuElementEligible(eptr, LONG_MAX / 4);
      break;
    }
    case T_MONI: case T_HMON: case T_VMON: {
      /* Monitors track as matrices; turn-by-turn readings average the beam.
       * (Fast orbit feedback, the other averaging reader, runs only inside its
       * own command.) */
      short storeTurnByTurn = eptr->type == T_MONI ? ((MONI *)eptr->p_elem)->storeTurnByTurn :
        eptr->type == T_HMON ? ((HMON *)eptr->p_elem)->storeTurnByTurn :
        ((VMON *)eptr->p_elem)->storeTurnByTurn;
      restriction = "turn-by-turn monitor readings average the batch";
      if (!storeTurnByTurn)
        supported = gpuElementEligible(eptr, LONG_MAX / 4);
      break;
    }
    case T_DRIF: case T_EDRIFT: case T_CSBEND: case T_CCBEND:
    case T_QUAD: case T_SBEN: case T_RBEN: case T_SEXT:
    case T_KQUAD: case T_KSEXT: case T_KOCT: case T_MULT:
    case T_HCOR: case T_VCOR: case T_HVCOR:
    case T_EHCOR: case T_EVCOR: case T_EHVCOR:
    case T_RCOL: case T_ECOL:
    case T_TAPERAPC: case T_TAPERAPR: case T_SPEEDBUMP:
      supported = gpuElementEligible(eptr, LONG_MAX / 4);
      break;
    default:
      break;
    }
    if (!supported) {
      static char detail[256];
      snprintf(detail, sizeof(detail), "%s at %s#%ld (%s)",
               restriction, eptr->name, eptr->occurence, entity_name[eptr->type]);
      *reason = detail;
      return 0;
    }
  }
  return 1;
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
  /* With a fringe term (HGAP and FINT both set) the vertical edge kick takes a tangent for every
   * particle, and the device's tan() isn't glibc's to the last bit.  Output in the CPU's order has to
   * be elegant's, so the CPU does those edges then. */
  if (gpuBase.orderSensitiveOutputNeeded && csbend->hgap * csbend->fint != 0)
    return 0;
  return gpu_csr_csbend_resident_available(csbend, nParticles,
                                           csbend->bins);
}

/* A short island by its elements may still be worth the beam's trip to the device and back.
 *
 * The count rule in gpuShortGpuIslandAhead() looks at how many elements lie before the next one the CPU
 * must take, not at the beam.  But the copies cost about 10 ns a particle each way and a matrix element
 * on the CPU costs 16 ns a particle at first order and 60 ns at second, so from 10,000 particles every
 * island the rule covers was cheaper on the device: a drift, quadrupole, drift between two CPU-only
 * elements took the CPU 125 to 300 ns a particle and the device 8 to 16 ns with the copies, a lone KQUAD
 * of 8 slices 650 ns against 21 to 27 ns, a lone CSBEND 890 ns against 23 to 30 ns (island_probe.py,
 * November 13, 2026, 10,000 to 1,000,000 particles).  A line with a ZLONGIT at every third element ran
 * at elegant's speed for that reason, 4.85 s where it takes 1.23 s.
 *
 * So the island's cost is estimated for each side from the numbers gpuChooseSmallBeamThresholds() uses,
 * and the island goes to the device when the CPU's is GPU_SHORT_ISLAND_MARGIN times the device's or more.
 * The estimate uses the lattice and the particle count, never a clock.  The copies are taken at 10 ns a
 * particle each, twice what was measured, and the CPU's numbers are the low ends, so the estimate errs
 * toward leaving the island on the CPU: a lone first-order drift stays there (16 ns against 9 to 13 ns
 * measured).  ELEGANT_GPU_SHORT_GPU_ISLAND_BY_COST=0 goes back to the count alone.  gpu-Pelegant keeps
 * the count: these timings are the serial build's. */
#if !USE_MPI
#define GPU_SHORT_ISLAND_COPY_US 20.0
#define GPU_SHORT_ISLAND_COPY_US_PER_PARTICLE 0.01
#define GPU_SHORT_ISLAND_MARGIN 1.4
/* ns a particle more than a first-order matrix: a second- or third-order one costs the CPU 60 ns and up */
#define GPU_SHORT_ISLAND_HIGH_ORDER_NS 45.0

static void gpuShortGpuIslandAddCost(ELEMENT_LIST *eptr, double *cpuNs, double *deviceUs) {
  double elementNs = 0, elementUs = 0;

  gpuSmallBeamElementCost(eptr, gpuBase.reductionOutputNeeded ? 1 : 0, &elementNs, &elementUs);
  /* the matrix is made when the element is first tracked; until then its order isn't known and the
   * first-order cost stands */
  if (gpuSimpleMatrixElement(eptr) && eptr->matrix && eptr->matrix->order >= 2)
    elementNs += GPU_SHORT_ISLAND_HIGH_ORDER_NS;
  *cpuNs += elementNs;
  *deviceUs += elementUs;
}

static long gpuShortGpuIslandPaysOnDevice(double cpuNs, double deviceUs, long nParticles) {
  double cpuUs = cpuNs * 1e-3 * nParticles;
  double tripUs = 2 * (GPU_SHORT_ISLAND_COPY_US + GPU_SHORT_ISLAND_COPY_US_PER_PARTICLE * nParticles);

  return cpuUs >= GPU_SHORT_ISLAND_MARGIN * (deviceUs + tripUs);
}
#endif

/* hostCurrent: whether the host holds the beam, which a walk through the lattice ahead of tracking
 * (gpuChooseSmallBeamThresholds) supplies for itself.  byCost, if not NULL, is set to 1 when the island
 * is short by its elements and goes to the device because that costs less. */
static long gpuShortGpuIslandAhead(ELEMENT_LIST *eptr, long nParticles, long hostCurrent, long *byCost) {
  ELEMENT_LIST *next;
  long cost = 0;
  unsigned long flags;
#if !USE_MPI
  double cpuNs = 0, deviceUs = 0;
#endif

  if (byCost)
    *byCost = 0;

  if (!gpuAvoidShortGpuIslands || gpuShortGpuIslandMaxElements <= 0)
    return 0;
  if (!hostCurrent || !eptr || nParticles <= 0)
    return 0;
  flags = entity_description[eptr->type].flags;
  /* The MAXAMP check after the element sends the beam back if the check can't use the device, as
   * below the resident floor, so a magnet with a lower threshold of its own, such as CCBEND or
   * LGBEND, would make a round trip by itself */
  if (gpuAfterElementAperture &&
      (gpuSimpleMatrixElement(eptr) || (flags & IS_MAGNET) ||
       eptr->type == T_CCBEND || eptr->type == T_LGBEND) &&
      !gpuParticleCountAllowed(nParticles, gpuBase.apertureMinParticles) &&
      !(gpuKeepResident && gpuParticleCountMeetsThreshold(nParticles, gpuResidentMinParticles)))
    return 1;
  if (!gpuSimpleMatrixElement(eptr) && !(flags & IS_MAGNET) &&
      eptr->type != T_TAPERAPC && eptr->type != T_TAPERAPR && eptr->type != T_TAPERAPE &&
      eptr->type != T_SPEEDBUMP)
    return 0;

  for (next = eptr; next; next = next->succ) {
    if (gpuPassiveElementSupported(next, nParticles))
      continue;
    if (gpuElementEligible(next, nParticles)) {
      /*
       * A magnet is substantially more expensive than a matrix drift, so
       * count it twice.  This still avoids an upload for an isolated magnet
       * (including its adjacent drifts), but keeps a two-magnet island on the
       * device.  Counting every eligible element equally made short repeating
       * cells such as D-B-D-B fall back entirely to the CPU.
       */
      cost += (entity_description[next->type].flags & IS_MAGNET) ? 2 : 1;
      if (cost > gpuShortGpuIslandMaxElements)
        return 0;
#if !USE_MPI
      gpuShortGpuIslandAddCost(next, &cpuNs, &deviceUs);
#endif
      continue;
    }
    if (gpuResidentCsrCsbendFollowsIsland(next, nParticles))
      return 0;
    if (next == eptr)
      return 0;
#if !USE_MPI
    /* short by its elements: the device takes it all the same if that costs less */
    if (gpuShortGpuIslandByCost && gpuShortGpuIslandPaysOnDevice(cpuNs, deviceUs, nParticles)) {
      if (byCost)
        *byCost = 1;
      return 0;
    }
#endif
    return !gpuElementEligible(next, nParticles);
  }
  return 0;
}

static long gpuShouldUseCpuForShortGpuIsland(ELEMENT_LIST *eptr, long nParticles) {
  long byCost = 0;
  long useCpu = gpuShortGpuIslandAhead(eptr, nParticles, gpuBase.hostCurrent, &byCost);

  if (byCost)
    gpuShortGpuIslandByCostCount++;
  return useCpu;
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
  gpuScmultMomentCacheValid = 0;
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
  if (gpuRfcaScratch.referenceKick) {
    status = gpuCudaFree(gpuRfcaScratch.referenceKick);
    if (status != 0)
      gpuFatalStatus("cudaFree(RFCA reference-kick scratch)", status);
  }
  free(gpuRfcaScratch.hostReferenceKick);
  free(gpuRfcaScratch.hostReferenceAux);
  memset(&gpuRfcaScratch, 0, sizeof(gpuRfcaScratch));
}

static void gpuReleaseIlmatrixScratch(void) {
  if (gpuIlmatrixScratch.phase) {
    int status = gpuCudaFree(gpuIlmatrixScratch.phase);
    if (status != 0)
      gpuFatalStatus("cudaFree(ILMATRIX phase scratch)", status);
  }
  free(gpuIlmatrixScratch.hostPhase);
  free(gpuIlmatrixScratch.hostSinCos);
  memset(&gpuIlmatrixScratch, 0, sizeof(gpuIlmatrixScratch));
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
  free(gpuCsrScratch.cpuCt);
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

static void gpuEnsureBeamSumsScratch(long nParticles) {
  int status;

  if (gpuBeamSumsScratch.data &&
      gpuBeamSumsScratch.particleCapacity >= nParticles)
    return;
  gpuReleaseBeamSumsScratch();
  status = gpuCudaMallocBytes(&gpuBeamSumsScratch.data,
                              gpuCudaBeamSums2ScratchBytes(nParticles));
  if (status != 0)
    gpuFatalStatus("cudaMalloc(beam-sums reduction scratch)", status);
  gpuBeamSumsScratch.particleCapacity = nParticles;
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

static void gpuEnsureRfcaReferenceScratch(long nParticles) {
  int status;
  if (nParticles <= gpuRfcaScratch.referenceCapacity)
    return;
  if (gpuRfcaScratch.referenceKick) {
    status = gpuCudaFree(gpuRfcaScratch.referenceKick);
    if (status != 0)
      gpuFatalStatus("cudaFree(RFCA reference-kick scratch resize)", status);
    gpuRfcaScratch.referenceKick = NULL;
  }
  gpuRfcaScratch.hostReferenceKick = trealloc(
    gpuRfcaScratch.hostReferenceKick, sizeof(double) * nParticles);
  gpuRfcaScratch.hostReferenceAux = trealloc(
    gpuRfcaScratch.hostReferenceAux, sizeof(double) * nParticles);
  status = gpuCudaMallocBytes(&gpuRfcaScratch.referenceKick,
                               2 * sizeof(double) * nParticles);
  if (status != 0)
    gpuFatalStatus("cudaMalloc(RFCA reference-kick scratch)", status);
  gpuRfcaScratch.referenceAux = (double *)gpuRfcaScratch.referenceKick + nParticles;
  gpuRfcaScratch.referenceCapacity = nParticles;
}

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
                              (unsigned long)nParticles * 2UL * sizeof(long));
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
  gpuCsrScratch.cpuCtCount = 0;
}

long gpu_copy_csr_wake_cpu_shadow(double *dGamma, long nBins) {
  if (!dGamma || nBins <= 0 || !gpuCsrScratch.cpuDGamma ||
      !gpuCsrScratch.cpuDGammaValid || gpuCsrScratch.cpuDGammaBins != nBins)
    return 0;
  memcpy(dGamma, gpuCsrScratch.cpuDGamma, (size_t)nBins * sizeof(*dGamma));
  return 1;
}

#ifdef GPU_VERIFY
/* Keeps c*t from the CPU check of the latest body slice. The CSR kick doesn't change it. */
static void gpuStoreCsrCtCpuShadow(double **coord, long nParticles) {
  long ip;

  if (gpuCsrScratch.cpuCtCapacity < nParticles) {
    double *buffer = realloc(gpuCsrScratch.cpuCt,
                             (size_t)nParticles * sizeof(*gpuCsrScratch.cpuCt));
    if (!buffer)
      gpuRequiredFailure("unable to allocate CUDA CSRCSBEND CPU-shadow c*t buffer");
    gpuCsrScratch.cpuCt = buffer;
    gpuCsrScratch.cpuCtCapacity = nParticles;
  }
  for (ip = 0; ip < nParticles; ip++)
    gpuCsrScratch.cpuCt[ip] = coord[ip][4];
  gpuCsrScratch.cpuCtCount = nParticles;
}

/* The preparation for a following CSRDRIFT bins c*t. It gets its result from the CPU code in
 * either build, so the wake scalars can only be wrong if it's given the wrong coordinates. */
void gpu_verify_csr_drift_prep_coordinates(double **part, long nParticles) {
  double absTol = gpuEnvDouble("ELEGANT_GPU_COMPARE_ABS", 1e-12);
  double relTol = gpuEnvDouble("ELEGANT_GPU_COMPARE_REL", 1e-12);
  double maxAbs = 0, maxRel = 0;
  long ip, mismatches = 0;

  if (!gpuBase.verifyMode || !part || gpuCsrScratch.cpuCtCount != nParticles)
    return;
  for (ip = 0; ip < nParticles; ip++) {
    double cpu = gpuCsrScratch.cpuCt[ip];
    double absDiff = fabs(cpu - part[ip][4]);
    double scale = fmax(fabs(cpu), fabs(part[ip][4]));
    double relDiff = scale > DBL_MIN ? absDiff / scale : absDiff;

    if (absDiff > maxAbs)
      maxAbs = absDiff;
    if (relDiff > maxRel)
      maxRel = relDiff;
    if (!(absDiff <= absTol || relDiff <= relTol))
      mismatches++;
  }
  if (mismatches) {
    fprintf(stderr,
            "elegant CUDA VERIFY failed for CSRDRIFT preparation input: %ld mismatches, maxAbs=%.3e, maxRel=%.3e, absTol=%.3e, relTol=%.3e\n",
            mismatches, maxAbs, maxRel, absTol, relTol);
    exit(1);
  }
  if (gpuVerbose)
    fprintf(stderr, "elegant CUDA VERIFY passed for CSRDRIFT preparation input: maxAbs=%.3e maxRel=%.3e\n",
            maxAbs, maxRel);
}

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
    gpuScmultMomentCacheValid = 0;
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
  /* The lost particles' rows follow the survivors', moved as on the CPU */
  if (nParticles < gpuAcceptedBuffer.nParticles)
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

static void gpuStablePartitionAccepted(double **accepted, long nParticles,
                                       long survivors) {
  long *prefix;
  double *scratch;
  long stride = totalPropertiesPerParticle;
  long ip, destination;

  if (!accepted || nParticles <= 0)
    return;
  (void)survivors;
  prefix = gpuApertureScratch.hostPrefix;
  scratch = gpuEnsureApertureHostAccepted(nParticles, stride);
  if (!prefix || !scratch)
    gpuRequiredFailure("missing host aperture scratch for accepted partition");
  for (ip = 0; ip < nParticles; ip++) {
    destination = prefix[ip];
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

  /* Keep every row the device holds since the upload, so the lost rows go back to the host too:
   * the rows of particles lost before this partition move along with the partitioned ones */
  if (gpuAcceptedBuffer.nParticles < nParticles)
    gpuAcceptedBuffer.nParticles = nParticles;
  else if (gpuAcceptedBuffer.nParticles > nParticles) {
    status = gpuCudaCopyDeviceToDevice(
      (double *)gpuAcceptedBuffer.scratch + (size_t)nParticles * stride,
      (double *)gpuAcceptedBuffer.coord + (size_t)nParticles * stride,
      (unsigned long)(gpuAcceptedBuffer.nParticles - nParticles) * (unsigned long)stride);
    if (status != 0)
      gpuFatalStatus(reason ? reason : "accepted CUDA stable compaction tail", status);
  }
  oldCoord = gpuAcceptedBuffer.coord;
  gpuAcceptedBuffer.coord = gpuAcceptedBuffer.scratch;
  gpuAcceptedBuffer.scratch = oldCoord;
  gpuAcceptedBuffer.deviceCurrent = 1;
  gpuAcceptedBuffer.hostCurrent = 0;
  return 1;
}

static long gpuApertureLossRowsNeedHost(void) {
#ifdef GPU_VERIFY
  /* GPU_VERIFY's CPU shadow writes the host's loss rows itself */
  return 0;
#else
  return gpuBase.lossOutputNeeded;
#endif
}

/* GPU_VERIFY's CPU shadow moves the accepted rows itself */
static double **gpuApertureHostAccepted(double **accepted) {
#ifdef GPU_VERIFY
  (void)accepted;
  return NULL;
#else
  return accepted;
#endif
}

#ifndef GPU_VERIFY
static void gpuMarkDeviceChanged(long nParticles);
static void gpuRecordMagnetKernel(float milliseconds);

static long gpuMagnetLossCompactionAllowed(void) {
  return gpuEnableMagnetLossCompaction;
}

/* elegant's routine for a magnet, on rows of its own: returns the number it keeps */
typedef long (*GPU_MAGNET_LOSS_TRACK)(double **rows, long nRows, void *cookie);

static struct {
  double *rows, *placed, *saved;
  double **pointers;
  long capacity, stride;
} gpuMagnetLossRows;

/* A magnet's lost rows, when a losses file or global loss coordinates want them on the host.  The
 * device has tracked and compacted into the aperture scratch, with each particle's place in the
 * prefix, and gpuBase.deviceCoord still holds the rows from before the magnet.  elegant's own routine
 * tracks the lost particles, alone, so each row comes out as the CPU writes it, whatever the magnet
 * does to a lost row afterwards (exit fringe, end drift, misalignment, global coordinates).  The
 * lossPassIndex slot, which is only read once a loss is recorded, carries each row's place through
 * the routine's swaps.  The rows go to the host's and the device's tail in the CPU's order.
 * Returns 0, with nothing changed, if there is no routine or if it keeps a particle the device lost
 * (a particle within rounding of an aperture): the caller then leaves the whole element to the CPU. */
static long gpuMagnetLossRowsFromCpu(long nParticles, long remaining, GPU_MAGNET_LOSS_TRACK track,
                                     void *cookie, const char *reason, long rowsInBackup) {
  long nLost = nParticles - remaining, stride = gpuBase.deviceStride, i, kept, elementOnGpu;
  unsigned long kicksDone = 0;
  float milliseconds = 0;
  int status;

  if (nLost <= 0 || (!gpuApertureLossRowsNeedHost() && globalLossCoordOffset <= 0))
    return 1;
  if (!track)
    return 0;
  if (!gpuBase.coord || !gpuBase.coord[0])
    gpuRequiredFailure("host particle array is unavailable for a magnet's lost rows");
  if (gpuMagnetLossRows.capacity < nLost || gpuMagnetLossRows.stride != stride) {
    long capacity = nLost > 2 * gpuMagnetLossRows.capacity ? nLost : 2 * gpuMagnetLossRows.capacity;

    free(gpuMagnetLossRows.rows);
    free(gpuMagnetLossRows.placed);
    free(gpuMagnetLossRows.saved);
    free(gpuMagnetLossRows.pointers);
    gpuMagnetLossRows.rows = tmalloc(sizeof(double) * capacity * stride);
    gpuMagnetLossRows.placed = tmalloc(sizeof(double) * capacity * stride);
    gpuMagnetLossRows.saved = tmalloc(sizeof(double) * capacity);
    gpuMagnetLossRows.pointers = tmalloc(sizeof(double *) * capacity);
    gpuMagnetLossRows.capacity = capacity;
    gpuMagnetLossRows.stride = stride;
  }
  /* after gpuCudaCompactTrackedRows() the device beam is already tracked, and the rows from before
   * the magnet are in the checked kernel's backup */
  status = gpuCudaLostRowsBeforeElement(rowsInBackup ? NULL : gpuBase.deviceCoord,
                                        gpuApertureScratch.coord,
                                        gpuApertureScratch.prefix, nParticles, (int)stride,
                                        remaining, gpuMagnetLossRows.rows, &milliseconds);
  if (status != 0)
    gpuFatalStatus("a magnet's lost rows from before the element", status);
  gpuRecordMilliseconds(&gpuBase.gpuTransferToHostSeconds, milliseconds);
  for (i = 0; i < nLost; i++) {
    gpuMagnetLossRows.pointers[i] = gpuMagnetLossRows.rows + i * stride;
    gpuMagnetLossRows.saved[i] = gpuMagnetLossRows.pointers[i][lossPassIndex];
    gpuMagnetLossRows.pointers[i][lossPassIndex] = i;
  }
  if (&multipoleKicksDone)
    kicksDone = multipoleKicksDone;
  elementOnGpu = gpuBase.elementOnGpu;
  gpuBase.elementOnGpu = 0;
  kept = track(gpuMagnetLossRows.pointers, nLost, cookie);
  gpuBase.elementOnGpu = elementOnGpu;
  if (&multipoleKicksDone)
    multipoleKicksDone = kicksDone;
  if (kept || gpuTestMagnetLossDisagreement) {
    if (gpuVerbose)
      fprintf(stderr, "elegant CUDA: the CPU kept %ld of %ld particles lost on the device (%s).\n",
              kept, nLost, reason ? reason : "magnet loss");
    return 0;
  }
  for (i = 0; i < nLost; i++) {
    double *row = gpuMagnetLossRows.rows + i * stride;
    long place = (long)row[lossPassIndex];

    if (place < 0 || place >= nLost || row[lossPassIndex] != place ||
        !gpuMagnetLossRows.pointers[place])
      gpuRequiredFailure("a magnet's lost rows came back from the CPU routine without their places");
    gpuMagnetLossRows.pointers[place] = NULL;
    row[lossPassIndex] = gpuMagnetLossRows.saved[place];
    memcpy(gpuMagnetLossRows.placed + place * stride, row, sizeof(double) * stride);
  }
  memcpy(gpuBase.coord[remaining], gpuMagnetLossRows.placed, sizeof(double) * nLost * stride);
  status = gpuCudaCopyHostToDevice((double *)gpuApertureScratch.coord + remaining * stride,
                                   gpuMagnetLossRows.placed, (unsigned long)nLost * stride,
                                   &milliseconds);
  if (status != 0)
    gpuFatalStatus("a magnet's lost rows to the device", status);
  gpuRecordMilliseconds(&gpuBase.gpuTransferToDeviceSeconds, milliseconds);
  gpuRecordSyncRequest(reason, 1, 1);
  if (gpuVerbose && reason)
    fprintf(stderr, "elegant CUDA: lost rows written by the CPU routine for %s (%ld row%s).\n",
            reason, nLost, nLost == 1 ? "" : "s");
  return 1;
}

/* ELEGANT_GPU_ENABLE_ONE_PASS_MAGNET_LOSS=0: a magnet call that loses a particle is put back and
 * tracked twice more to compact it, as it was until November 13, 2026 */
static long gpuOnePassMagnetLossEnabled(void) {
  return !gpuEnvSet("ELEGANT_GPU_ENABLE_ONE_PASS_MAGNET_LOSS") ||
         gpuEnvFlag("ELEGANT_GPU_ENABLE_ONE_PASS_MAGNET_LOSS");
}

/* Where a magnet's checked kernel is to note which particles it kept, so a call that loses particles
 * needn't be tracked again: the aperture scratch's prefix, or NULL when the call is to be put back
 * as it was on a loss (compaction off, or the old two extra passes asked for) */
static void *gpuMagnetSurvivorFlags(long nParticles) {
  if (!gpuMagnetLossCompactionAllowed() || !gpuOnePassMagnetLossEnabled() || nParticles <= 0)
    return NULL;
  gpuEnsureApertureScratch(nParticles);
  return gpuApertureScratch.prefix;
}

/* A magnet's checked kernel has tracked the beam in place, lost particles, and left the survivor
 * flags of gpuMagnetSurvivorFlags().  The rows are moved, as tracked, into the CPU's order: one pass
 * through the magnet where gpuMultipoleStableCompact() and its like made three.  Returns the number
 * kept, or -1 with the device beam put back as it was before the magnet, if the element must go to
 * the CPU (see gpuMagnetLossRowsFromCpu()). */
static long gpuCompactTrackedMagnet(long nParticles, double **accepted, const char *name,
                                    GPU_MAGNET_LOSS_TRACK track, void *cookie) {
  long remaining = nParticles;
  float milliseconds = 0;
  int status;

  status = gpuCudaCompactTrackedRows(gpuBase.deviceCoord, gpuApertureScratch.coord,
                                     gpuApertureScratch.prefix, nParticles,
                                     (int)gpuBase.deviceStride, &remaining, &milliseconds);
  if (status != 0)
    gpuFatalStatus(name ? name : "magnet loss compaction of tracked rows", status);
  gpuRecordMagnetKernel(milliseconds);
  if (!gpuMagnetLossRowsFromCpu(nParticles, remaining, track, cookie, name, 1)) {
    status = gpuCudaRestoreMagnetBackup(gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride);
    if (status != 0)
      gpuFatalStatus("restoring the beam from before a magnet", status);
    return -1;
  }
  if (accepted) {
    if (!gpuStablePartitionAcceptedOnDevice(accepted, nParticles, remaining,
                                            "magnet CUDA accepted stable compaction")) {
      gpuCopyAperturePrefixToHost(nParticles, "magnet CUDA stable compaction accepted map");
      gpuStablePartitionAccepted(accepted, nParticles, remaining);
    }
  }
  gpuPromoteApertureScratchCoord();
  gpuMarkDeviceChanged(remaining);
  gpuRecordWallSeconds();
  return remaining;
}

static long gpuMultipoleStableCompact(GPU_MULTIPOLE_DATA *data,
                                      long nParticles, double **accepted,
                                      const char *name,
  GPU_MAGNET_LOSS_TRACK track, void *cookie) {
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
  if (!gpuMagnetLossRowsFromCpu(nParticles, remaining, track, cookie, name, 0))
    return -1;
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
                                   double **accepted, const char *name,
  GPU_MAGNET_LOSS_TRACK track, void *cookie) {
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
  if (!gpuMagnetLossRowsFromCpu(nParticles, remaining, track, cookie, name, 0))
    return -1;
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

static long gpuCcbendStableCompact(GPU_CCBEND_DATA *data, long nParticles,
                                   double **accepted, const char *name,
  GPU_MAGNET_LOSS_TRACK track, void *cookie) {
  long remaining = nParticles;
  float milliseconds = 0;
  int status;

  if (nParticles <= 0)
    return nParticles;
  gpuEnsureApertureScratch(nParticles);
  status = gpuCudaCcbendTrackStableCompact(
    gpuBase.deviceCoord, gpuApertureScratch.coord, gpuApertureScratch.prefix,
    nParticles, (int)gpuBase.deviceStride, data, &remaining, &milliseconds);
  if (status != 0)
    gpuFatalStatus(name ? name : "CCBEND stable magnet loss compaction",
                   status);
  gpuRecordMagnetKernel(milliseconds);
  if (!gpuMagnetLossRowsFromCpu(nParticles, remaining, track, cookie, name, 0))
    return -1;
  if (accepted) {
    if (!gpuStablePartitionAcceptedOnDevice(
          accepted, nParticles, remaining,
          "CCBEND CUDA accepted stable compaction")) {
      gpuCopyAperturePrefixToHost(
        nParticles, "CCBEND CUDA stable compaction accepted map");
      gpuStablePartitionAccepted(accepted, nParticles, remaining);
    }
  }
  gpuPromoteApertureScratchCoord();
  gpuMarkDeviceChanged(remaining);
  gpuRecordWallSeconds();
  return remaining;
}

static long gpuLgbendStableCompact(
  GPU_LGBEND_DATA *data,
  GPU_LGBEND_LOCAL_APERTURE_DATA *localAperture,
  long nParticles, double **accepted, const char *name,
  GPU_MAGNET_LOSS_TRACK track, void *cookie) {
  long remaining = nParticles;
  float milliseconds = 0;
  int status;

  if (nParticles <= 0)
    return nParticles;
  gpuEnsureApertureScratch(nParticles);
  status = gpuCudaLgbendTrackStableCompact(
    gpuBase.deviceCoord, gpuApertureScratch.coord, gpuApertureScratch.prefix,
    nParticles, (int)gpuBase.deviceStride, data,
    localAperture && (localAperture->present || localAperture->hasFileAperture) ? localAperture : NULL,
    &remaining, &milliseconds);
  if (status != 0)
    gpuFatalStatus(name ? name : "LGBEND stable magnet loss compaction",
                   status);
  gpuRecordMagnetKernel(milliseconds);
  if (!gpuMagnetLossRowsFromCpu(nParticles, remaining, track, cookie, name, 0))
    return -1;
  if (accepted) {
    if (!gpuStablePartitionAcceptedOnDevice(
          accepted, nParticles, remaining,
          "LGBEND CUDA accepted stable compaction")) {
      gpuCopyAperturePrefixToHost(
        nParticles, "LGBEND CUDA stable compaction accepted map");
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
                                    const char *name,
  GPU_MAGNET_LOSS_TRACK track, void *cookie) {
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
  if (!gpuMagnetLossRowsFromCpu(nParticles, remaining, track, cookie, name, 0))
    return -1;
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
#else
/* The verify build compares each lossy magnet call with the CPU through its own kernels */
static void *gpuMagnetSurvivorFlags(long nParticles) {
  (void)nParticles;
  return NULL;
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

static void gpuMarkHostWillChange(void) {
  gpuFlushPendingExactDrift("pending exactDrift before host coordinate mutation");
  gpuBase.elementOnGpu = 0;
  gpuBase.hostCurrent = 1;
  gpuBase.deviceCurrent = 0;
  gpuDeviceIslandHasCsbend = 0;
  gpuScmultMomentCacheValid = 0;
  if (gpuAcceptedBuffer.hostAccepted) {
    gpuAcceptedBuffer.hostCurrent = 1;
    gpuAcceptedBuffer.deviceCurrent = 0;
  }
}

static void gpuMarkDeviceChanged(long nParticles) {
  gpuBase.deviceCurrent = 1;
  gpuBase.hostCurrent = 0;
  gpuBase.nParticles = nParticles;
  gpuScmultMomentCacheValid = 0;
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

/* Time and calls of the helper and reduction kernels by the line that recorded them, printed with the
 * activity summary when ELEGANT_GPU_PROFILE_TIMING is set (the times are real only then) */
#define GPU_KERNEL_SITE_SLOTS 256
static struct {
  int line;
  char kind;
  unsigned long calls;
  double seconds;
} gpuKernelSite[GPU_KERNEL_SITE_SLOTS];

static void gpuRecordKernelSite(int line, char kind, float milliseconds) {
  long slot = (unsigned long)line % GPU_KERNEL_SITE_SLOTS, tries;

  for (tries = 0; tries < GPU_KERNEL_SITE_SLOTS; tries++, slot = (slot + 1) % GPU_KERNEL_SITE_SLOTS) {
    if (!gpuKernelSite[slot].line) {
      gpuKernelSite[slot].line = line;
      gpuKernelSite[slot].kind = kind;
    }
    if (gpuKernelSite[slot].line == line) {
      gpuKernelSite[slot].calls++;
      if (milliseconds > 0)
        gpuKernelSite[slot].seconds += milliseconds * 1e-3;
      return;
    }
  }
}

static void gpuReportKernelSites(void) {
  long slot;

  for (slot = 0; slot < GPU_KERNEL_SITE_SLOTS; slot++)
    if (gpuKernelSite[slot].line) {
      fprintf(stderr, "elegant CUDA: %s kernel site gpu_stub.c:%d calls=%lu seconds=%.6f\n",
              gpuKernelSite[slot].kind == 'r' ? "reduction" : (gpuKernelSite[slot].kind == 'i' ? "ISR numbers" : "helper"), gpuKernelSite[slot].line,
              gpuKernelSite[slot].calls, gpuKernelSite[slot].seconds);
      gpuKernelSite[slot].line = 0;
      gpuKernelSite[slot].calls = 0;
      gpuKernelSite[slot].seconds = 0;
    }
}

static void gpuRecordHelperKernelAt(float milliseconds, int line) {
  gpuRecordMilliseconds(&gpuBase.gpuKernelSeconds, milliseconds);
  gpuBase.gpuHelperCount++;
  gpuRecordKernelSite(line, 'h', milliseconds);
}

static void gpuRecordReductionKernelAt(float milliseconds, int line) {
  gpuRecordMilliseconds(&gpuBase.gpuKernelSeconds, milliseconds);
  gpuRecordMilliseconds(&gpuBase.gpuReductionSeconds, milliseconds);
  gpuBase.gpuReductionCount++;
  gpuRecordKernelSite(line, 'r', milliseconds);
}
#define gpuRecordHelperKernel(milliseconds) gpuRecordHelperKernelAt(milliseconds, __LINE__)
#define gpuRecordReductionKernel(milliseconds) gpuRecordReductionKernelAt(milliseconds, __LINE__)

static void gpuRecordApertureKernel(float milliseconds) {
  gpuRecordMilliseconds(&gpuBase.gpuKernelSeconds, milliseconds);
  gpuBase.gpuApertureCount++;
}

static void gpuRecordMagnetKernel(float milliseconds) {
  gpuRecordMilliseconds(&gpuBase.gpuKernelSeconds, milliseconds);
  gpuBase.gpuMagnetCount++;
}

#ifdef GPU_VERIFY
static long gpuVerifyMultipoleStableCompact(GPU_MULTIPOLE_DATA *data,
                                            long nParticles,
                                            const char *name) {
  long remaining = nParticles;
  float milliseconds = 0;
  int status;

  gpuEnsureApertureScratch(nParticles);
  status = gpuCudaMultipoleTrackStableCompact(
    gpuBase.deviceCoord, gpuApertureScratch.coord, gpuApertureScratch.prefix,
    nParticles, (int)gpuBase.deviceStride, data, &remaining, &milliseconds);
  if (status != 0)
    gpuFatalStatus(name ? name :
                   "multipole GPU-VERIFY stable magnet loss compaction",
                   status);
  gpuRecordMagnetKernel(milliseconds);
  gpuPromoteApertureScratchCoord();
  gpuMarkDeviceChanged(remaining);
  return remaining;
}

static long gpuVerifyCsbendStableCompact(GPU_CSBEND_DATA *data,
                                         long nParticles,
                                         const char *name) {
  long remaining = nParticles;
  float milliseconds = 0;
  int status;

  gpuEnsureApertureScratch(nParticles);
  status = gpuCudaCsbendTrackStableCompact(
    gpuBase.deviceCoord, gpuApertureScratch.coord, gpuApertureScratch.prefix,
    nParticles, (int)gpuBase.deviceStride, data, &remaining, &milliseconds);
  if (status != 0)
    gpuFatalStatus(name ? name :
                   "CSBEND GPU-VERIFY stable magnet loss compaction",
                   status);
  gpuRecordMagnetKernel(milliseconds);
  gpuPromoteApertureScratchCoord();
  gpuMarkDeviceChanged(remaining);
  return remaining;
}
#endif

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
  gpuRecordMilliseconds(&gpuBase.gpuScmultKickSeconds, milliseconds);
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

/* The CPU's sum, in particle order, for a reference set once that must match the CPU exactly */
static double gpuCoordinateSumInCpuOrder(long nParticles, long coordinate, const char *reason) {
  double **coord = copyParticlesToCpuReadOnly(reason);
  double sum = 0;
  long i;

  for (i = 0; i < nParticles; i++)
    sum += coord[i][coordinate];
  return sum;
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
  status = gpuCudaTrackParticles(gpuBase.deviceCoord, nParticles,
                                 (int)gpuBase.deviceStride, &packed,
                                 &milliseconds);
  if (status != 0)
    gpuFatalStatus("matrix tracking kernel", status);
  gpuRecordMilliseconds(&gpuBase.gpuKernelSeconds, milliseconds);
  gpuRecordMilliseconds(&gpuBase.gpuMatrixSeconds, milliseconds);
  gpuMarkDeviceChanged(nParticles);
  gpuBase.gpuTrackParticleCount++;
}

static void gpuLaunchTrackParticles(VMATRIX *matrix, long nParticles) {
  gpuLaunchTrackParticlesWithSReference(matrix, nParticles, 0, 0);
}

static long gpuPackMatrix(GPU_MATRIX_DATA *packed, VMATRIX *M) {
  long i, j, k, l, terms = 0;

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
  if (M->order == 2) {
    for (i = 0; i < 6; i++) {
      for (j = 0; j < 6; j++) {
        packed->secondOrderOffset[i * 6 + j] = (int)terms;
        for (k = j; k >= 0; k--) {
          if (M->T[i][j][k] != 0) {
            packed->term[terms].coefficient = M->T[i][j][k];
            packed->term[terms].degree = 2;
            packed->term[terms].j = (unsigned char)j;
            packed->term[terms].k = (unsigned char)k;
            terms++;
          }
        }
      }
    }
    packed->secondOrderOffset[36] = (int)terms;
    if (!terms) {
      packed->order = 1;
      for (i = 0; i < 6; i++) {
        packed->termOffset[i] = (int)terms;
        for (j = 5; j >= 0; j--) {
          if (M->R[i][j] != 0) {
            packed->term[terms].coefficient = M->R[i][j];
            packed->term[terms].degree = 1;
            packed->term[terms].j = (unsigned char)j;
            terms++;
          }
        }
      }
      packed->termOffset[6] = (int)terms;
    }
    gpuMatrixSparseTermTotal += (unsigned long)terms;
    gpuMatrixSparseTermHistogram[terms]++;
  }
  if (M->order == 1 || M->order >= 3) {
    for (i = 0; i < 6; i++) {
      packed->termOffset[i] = (int)terms;
      for (j = 5; j >= 0; j--) {
        if (M->R[i][j] != 0) {
          packed->term[terms].coefficient = M->R[i][j];
          packed->term[terms].degree = 1;
          packed->term[terms].j = (unsigned char)j;
          terms++;
        }
        if (M->order == 1)
          continue;
        for (k = j; k >= 0; k--) {
          if (M->T[i][j][k] != 0) {
            packed->term[terms].coefficient = M->T[i][j][k];
            packed->term[terms].degree = 2;
            packed->term[terms].j = (unsigned char)j;
            packed->term[terms].k = (unsigned char)k;
            terms++;
          }
          for (l = k; l >= 0; l--) {
            if (M->Q[i][j][k][l] != 0) {
              packed->term[terms].coefficient = M->Q[i][j][k][l];
              packed->term[terms].degree = 3;
              packed->term[terms].j = (unsigned char)j;
              packed->term[terms].k = (unsigned char)k;
              packed->term[terms].l = (unsigned char)l;
              terms++;
            }
          }
        }
      }
    }
    packed->termOffset[6] = (int)terms;
    if (terms > GPU_MATRIX_MAX_TERMS)
      return 0;
    gpuMatrixSparseTermTotal += (unsigned long)terms;
    gpuMatrixSparseTermHistogram[terms]++;
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
  return gpuResidentParticleCountAllowed(nParticles, gpuBase.reductionMinParticles) &&
         gpuSelectDevice();
}

/* Each beam sum on the CPU copies the whole beam back, which dominates a run with a
 * centroid or sigma file.  Those sums per element, and WATCH parameter sums, use the device, and differ from the
 * CPU's by rounding.  WATCH parameter sums do so without those files too, since they feed only the WATCH file.
 * Other sums stay on the CPU, because fit points and optimization terms steer on them.
 * ELEGANT_GPU_ENABLE_OUTPUT_BEAM_SUMS=0 restores the CPU. */
static long gpuOutputBeamSumsAllowed(unsigned long flags) {
  return gpuEnableOutputBeamSums && (flags & BEAM_SUMS_ELEMENT_OUTPUT) &&
         (gpuBase.reductionOutputNeeded || (flags & BEAM_SUMS_WATCH_OUTPUT));
}

long gpu_beam_sums_enabled(long nParticles, unsigned long flags) {
  if (!gpuOutputBeamSumsAllowed(flags))
    return gpu_reductions_enabled(nParticles);
  if (!gpuBase.initialized || gpuBase.activeDevice < 0 || nParticles <= 0)
    return 0;
#if USE_MPI
  if (distributedBeam)
    return 0;
#endif
  return gpuResidentParticleCountAllowed(nParticles, gpuBase.reductionMinParticles) &&
         gpuSelectDevice();
}

/* With ELEGANT_GPU_ENABLE_SLICE_SUMS=1, performSliceAnalysis() reads the beam where the device has the
 * only current copy, so it needn't come to the host */
long gpu_slice_moments_available(long nParticles) {
#if USE_MPI
  (void)nParticles;
  return 0;
#else
  if (!gpuEnableSliceSums || !gpuBase.initialized || gpuBase.activeDevice < 0 || nParticles <= 0 ||
      nParticles != gpuBase.nParticles)
    return 0;
  gpuFlushPendingExactDrift("pending exactDrift before slice analysis");
  return gpuBase.deviceCoord && gpuBase.deviceCurrent && !gpuBase.hostCurrent;
#endif
}

/* performSliceAnalysis()'s numbers from the device: the bunch's time span, then for the whole beam
 * (slice 0) and, with more than one slice, slices 1 to nSlices, the count, the means C[slice][6] and
 * the moments S[slice][6][6].  Each sum runs over the particles in order, as computeSliceMoments()
 * does, so they are the CPU's.  Returns 0 if the host must compute them. */
long gpu_slice_moments(long nParticles, double Po, long nSlices, double *tMinAll, double *tMaxAll,
                       long *count, double *C, double *S) {
  long slices = nSlices > 1 ? nSlices + 1 : 1;
  double *bounds, *second;
  float milliseconds = 0;
  int status;

  if (!gpu_slice_moments_available(nParticles))
    return 0;
  status = gpuCudaSliceTimes(gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride, Po, c_mks,
                             tMinAll, tMaxAll, &milliseconds);
  if (status != 0)
    gpuFatalStatus("slice analysis time coordinates", status);
  gpuRecordReductionKernel(milliseconds);
  bounds = (double *)malloc(2 * slices * sizeof(*bounds));
  second = (double *)malloc(21 * slices * sizeof(*second));
  if (!bounds || !second)
    gpuRequiredFailure("unable to allocate slice analysis buffers");
  /* Equal bounds take the whole beam, as computeSliceMoments(..., 0, 0) does */
  bounds[0] = bounds[1] = 0;
  for (long slice = 1; slice < slices; slice++) {
    bounds[2 * slice] = (slice - 1) * (*tMaxAll - *tMinAll) / nSlices + *tMinAll;
    bounds[2 * slice + 1] = bounds[2 * slice] + (*tMaxAll - *tMinAll) / nSlices;
  }
  status = gpuCudaSliceMoments(gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride, bounds, slices,
                               count, C, second, &milliseconds);
  if (status != 0)
    gpuFatalStatus("slice analysis moments", status);
  gpuRecordReductionKernel(milliseconds);
  for (long slice = 0; slice < slices; slice++) {
    long pair = 0;

    for (long j = 0; j < 6; j++)
      for (long k = 0; k <= j; k++, pair++)
        S[slice * 36 + j * 6 + k] = S[slice * 36 + k * 6 + j] =
          count[slice] ? second[slice * 21 + pair] / count[slice] : 0;
  }
  free(bounds);
  free(second);
  return 1;
}

long gpu_watch_parameters_supported(void *watch0, long nParticles) {
#if USE_MPI
  (void)watch0;
  (void)nParticles;
  return 0;
#else
  WATCH *watch = (WATCH *)watch0;

  if (!watch || watch->disable || !getElementOnGpu() ||
      !gpu_beam_sums_enabled(nParticles, BEAM_SUMS_ELEMENT_OUTPUT | BEAM_SUMS_WATCH_OUTPUT))
    return 0;
  if (!(watch->startPID < 0 && watch->endPID < 0))
    return 0;
  if (exactNormalizedEmittance)
    return 0;
  return 1;
#endif
}

long gpu_watch_parameter_sums(long nParticles, int64_t *count,
                              double *pSum, double *gammaSum,
                              double *sigma56) {
  if (!gpuLastBeamSumResultValid ||
      gpuLastBeamSumResult.count != nParticles ||
      !count || !pSum || !gammaSum || !sigma56)
    return 0;
  *count = gpuLastBeamSumResult.count;
  *pSum = gpuLastBeamSumResult.pSum;
  *gammaSum = gpuLastBeamSumResult.gammaSum;
  *sigma56 = gpuLastBeamSumSigma56;
  return 1;
}

#ifdef GPU_VERIFY
/* Compares squares, scaled by the product before the correlation is removed, because el is
 * the small remainder of two nearly equal numbers when t and delta are strongly correlated.
 * Each time carries rounding of about DBL_EPSILON * |t|, which far along a machine matters
 * relative to St, so the CPU's and GPU's sums can't agree better than that. */
void gpu_verify_watch_el(long nParticles, double pCentral, int64_t startPID, int64_t endPID,
                         double gpuEl, void *sums0) {
  BEAM_SUMS *sums = (BEAM_SUMS *)sums0;
  double tolerance = gpuEnvDouble("ELEGANT_GPU_WATCH_EL_COMPARE", 1e-12);
  double timeVariance = sums->beamSums2->sigma[6][6];
  double scale = timeVariance * sums->beamSums2->sigma[5][5];
  double timeRounding = timeVariance > 0 ? DBL_EPSILON * fabs(sums->centroid[6]) / sqrt(timeVariance) : 0;
  double allowed = (tolerance + timeRounding) * scale;
  double **coord;
  double cpuEl, difference;

  if (!gpuBase.verifyMode)
    return;
  coord = copyParticlesToCpuReadOnly("WATCH el verification input");
  cpuEl = rms_longitudinal_emittance(coord, nParticles, pCentral, startPID, endPID);
  difference = fabs(cpuEl * cpuEl - gpuEl * gpuEl);
  if (difference > allowed) {
    fprintf(stderr,
            "elegant CUDA VERIFY failed for WATCH el: cpu=%.17e gpu=%.17e difference of squares=%.3e, allowed=%.3e, timeRounding=%.3e\n",
            cpuEl, gpuEl, difference, allowed, timeRounding);
    exit(1);
  }
  if (gpuVerbose)
    fprintf(stderr, "elegant CUDA VERIFY passed for WATCH el: cpu=%.3e gpu=%.3e relative=%.3e timeRounding=%.3e\n",
            cpuEl, gpuEl, scale > 0 ? difference / scale : 0, timeRounding);
}

static double gpuHistogramCpuCoordinate(const double *part, long coordinate,
                                        double pCentral,
                                        double timeOffset) {
  double p, time;

  if (coordinate != 4 && coordinate != 6)
    return part[coordinate];
  p = pCentral * (1 + part[5]);
  time = part[4] / (c_mks * p / sqrt(p * p + 1));
  return coordinate == 6 ? time - timeOffset : time;
}

static long gpuHistogramCpuParticleSelected(const double *part,
                                            int64_t startPID, int64_t endPID) {
  return (startPID < 0 && endPID < 0) ||
         (part[6] >= startPID && part[6] <= endPID);
}
#endif

int64_t gpu_histogram_ranges(long nParticles, double pCentral,
                             int64_t startPID, int64_t endPID,
                             unsigned int coordinateMask,
                             double *minimum, double *maximum) {
  GPU_HISTOGRAM_RANGE_DATA result;
  float milliseconds = 0;
  int status;

  if (nParticles <= 0 || !(coordinateMask & 0x7fU) ||
      !minimum || !maximum)
    gpuRequiredFailure("invalid HISTOGRAM CUDA range request");
  startGpuTimer();
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaHistogramRanges(
    gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride,
    pCentral, c_mks, startPID, endPID, coordinateMask,
    &result, &milliseconds);
  if (status != 0)
    gpuFatalStatus("HISTOGRAM CUDA range kernels", status);
  gpuRecordReductionKernel(milliseconds);
  gpuBase.gpuHistogramCount++;
  memcpy(minimum, result.minimum, sizeof(result.minimum));
  memcpy(maximum, result.maximum, sizeof(result.maximum));
#ifdef GPU_VERIFY
  if (gpuBase.verifyMode) {
    double **coord = copyParticlesToCpuReadOnly(
      "HISTOGRAM range verification input");
    double cpuMinimum[7], cpuMaximum[7];
    double absTol = gpuEnvDouble("ELEGANT_GPU_HISTOGRAM_COMPARE_ABS", 1e-15);
    double relTol = gpuEnvDouble("ELEGANT_GPU_HISTOGRAM_COMPARE_REL", 1e-12);
    int64_t cpuCount = 0;
    long mismatches = 0;

    for (long icoord = 0; icoord < 7; icoord++) {
      cpuMinimum[icoord] = DBL_MAX;
      cpuMaximum[icoord] = -DBL_MAX;
    }
    for (long ip = 0; ip < nParticles; ip++) {
      if (!gpuHistogramCpuParticleSelected(coord[ip], startPID, endPID))
        continue;
      cpuCount++;
      for (long icoord = 0; icoord < 7; icoord++) {
        double value;

        if (!(coordinateMask & (1U << icoord)))
          continue;
        value = gpuHistogramCpuCoordinate(coord[ip], icoord, pCentral, 0);
        if (value < cpuMinimum[icoord])
          cpuMinimum[icoord] = value;
        if (value > cpuMaximum[icoord])
          cpuMaximum[icoord] = value;
      }
    }
    if (cpuCount != result.count) {
      fprintf(stderr,
              "elegant CUDA VERIFY mismatch HISTOGRAM selection count cpu=%" PRId64 " gpu=%" PRId64 "\n",
              cpuCount, result.count);
      mismatches++;
    }
    if (cpuCount) {
      for (long icoord = 0; icoord < 7; icoord++) {
        double absDiff, relDiff;

        if (!(coordinateMask & (1U << icoord)))
          continue;
        if (!gpuValuesClose(cpuMinimum[icoord], result.minimum[icoord],
                            absTol, relTol, &absDiff, &relDiff)) {
          fprintf(stderr,
                  "elegant CUDA VERIFY mismatch HISTOGRAM minimum coordinate=%ld cpu=%.17e gpu=%.17e abs=%.3e rel=%.3e\n",
                  icoord, cpuMinimum[icoord], result.minimum[icoord],
                  absDiff, relDiff);
          mismatches++;
        }
        if (!gpuValuesClose(cpuMaximum[icoord], result.maximum[icoord],
                            absTol, relTol, &absDiff, &relDiff)) {
          fprintf(stderr,
                  "elegant CUDA VERIFY mismatch HISTOGRAM maximum coordinate=%ld cpu=%.17e gpu=%.17e abs=%.3e rel=%.3e\n",
                  icoord, cpuMaximum[icoord], result.maximum[icoord],
                  absDiff, relDiff);
          mismatches++;
        }
      }
    }
    if (mismatches)
      gpuRequiredFailure("HISTOGRAM CUDA range verification failed");
    if (gpuVerbose)
      fprintf(stderr,
              "elegant CUDA VERIFY passed for HISTOGRAM ranges: %" PRId64 " selected particles\n",
              result.count);
  }
#endif
  gpuRecordWallSeconds();
  return result.count;
}

void gpu_histogram_bins(long nParticles, double pCentral,
                        int64_t startPID, int64_t endPID, long bins,
                        unsigned int coordinateMask, double timeOffset,
                        const double *lower, const double *upper,
                        double *histogram) {
  long values;
  float milliseconds = 0;
  int status;

  if (nParticles <= 0 || bins < 2 || bins > LONG_MAX / 7 ||
      !(coordinateMask & 0x7fU) || !lower || !upper || !histogram)
    gpuRequiredFailure("invalid HISTOGRAM CUDA bin request");
  values = 7 * bins;
  if (!gpuHistogramHostBins || gpuHistogramHostBinCapacity < values) {
    unsigned long long *replacement = (unsigned long long *)realloc(
      gpuHistogramHostBins, values * sizeof(*replacement));
    if (!replacement)
      gpuRequiredFailure("unable to allocate HISTOGRAM host bin buffer");
    gpuHistogramHostBins = replacement;
    gpuHistogramHostBinCapacity = values;
  }
  startGpuTimer();
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaHistogramBins(
    gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride,
    pCentral, c_mks, startPID, endPID, bins, coordinateMask,
    timeOffset, lower, upper, gpuHistogramHostBins, &milliseconds);
  if (status != 0)
    gpuFatalStatus("HISTOGRAM CUDA bin kernels", status);
  gpuRecordReductionKernel(milliseconds);
  for (long index = 0; index < values; index++) {
    if (gpuHistogramHostBins[index] > 9007199254740992ULL)
      gpuRequiredFailure("HISTOGRAM CUDA bin count exceeds exact double range");
    histogram[index] = (double)gpuHistogramHostBins[index];
  }
#ifdef GPU_VERIFY
  if (gpuBase.verifyMode) {
    double **coord = copyParticlesToCpuReadOnly(
      "HISTOGRAM bin verification input");
    unsigned long long *cpuHistogram = (unsigned long long *)calloc(
      values, sizeof(*cpuHistogram));
    long mismatches = 0;

    if (!cpuHistogram)
      gpuRequiredFailure("unable to allocate HISTOGRAM verification bins");
    for (long ip = 0; ip < nParticles; ip++) {
      if (!gpuHistogramCpuParticleSelected(coord[ip], startPID, endPID))
        continue;
      for (long icoord = 0; icoord < 7; icoord++) {
        double binSize, dbin, value;
        long bin;

        if (!(coordinateMask & (1U << icoord)))
          continue;
        value = gpuHistogramCpuCoordinate(coord[ip], icoord, pCentral,
                                          timeOffset);
        binSize = (upper[icoord] - lower[icoord]) / bins;
        bin = (dbin = (value - lower[icoord]) / binSize);
        if (dbin < 0 || bin < 0 || bin >= bins)
          continue;
        cpuHistogram[icoord * bins + bin]++;
      }
    }
    for (long index = 0; index < values; index++) {
      if (cpuHistogram[index] == gpuHistogramHostBins[index])
        continue;
      if (mismatches < 10)
        fprintf(stderr,
                "elegant CUDA VERIFY mismatch HISTOGRAM coordinate=%ld bin=%ld cpu=%llu gpu=%llu\n",
                index / bins, index % bins,
                cpuHistogram[index], gpuHistogramHostBins[index]);
      mismatches++;
    }
    free(cpuHistogram);
    if (mismatches)
      gpuRequiredFailure("HISTOGRAM CUDA bin verification failed");
    if (gpuVerbose)
      fprintf(stderr,
              "elegant CUDA VERIFY passed for HISTOGRAM bins: mask=0x%x with %ld bins\n",
              coordinateMask, bins);
  }
#endif
  gpuRecordWallSeconds();
}

GPU_BASE *getGpuBase(void) {
  return &gpuBase;
}

long gpuSetOmpTrackingThreads(long threads) {
  if (threads < 1)
    return 0;
#if !defined(_OPENMP)
  if (threads > 1)
    return 0;
#endif
  gpuOmpTrackingThreads = threads;
  return 1;
}

long gpuGetOmpTrackingThreads(void) {
  return gpuOmpTrackingThreads;
}

long gpuOmpTrackingRequested(long particles) {
  return gpuOmpTrackingThreads > 1 && particles >= 64;
}

/* A loop of a few operations a particle gains nothing from threads on a small beam, and starting the
 * OpenMP team for it leaves the other threads spinning afterward: with 100-particle buckets through
 * FRFMODE and FTRFMODE, eight threads used 489 s of CPU time in a 62 s run, and a 1,000-particle
 * ring's time coordinates, computed every pass, 141 s in 20 s.  At 100,000 particles such a loop is
 * about a millisecond of work for one thread.  ELEGANT_GPU_OMP_LIGHT_LOOP_MIN_PARTICLES changes the
 * limit; the tests set it to 64 to run these loops on their small beams. */
#define GPU_OMP_LIGHT_LOOP_MIN_PARTICLES 100000
long gpuOmpLightLoopRequested(long particles) {
  static long minimum = -1;

  if (gpuOmpTrackingThreads <= 1)
    return 0;
  if (minimum < 0) {
    minimum = gpuEnvLong("ELEGANT_GPU_OMP_LIGHT_LOOP_MIN_PARTICLES", GPU_OMP_LIGHT_LOOP_MIN_PARTICLES);
    if (minimum < 64)
      minimum = 64;
  }
  return particles >= minimum;
}

/* -ompThreads and OMP_NUM_THREADS mean the same: one count for the tracking loops, the host thread
 * pool, the tune analysis, the batched-search switch and (through the caller) OpenMP's own default
 * and MKL.  elegant.c calls this once the command line has been read. */
void gpuSetHostThreads(long threads, const char *source) {
  if (threads < 1)
    threads = 1;
  gpuOmpTrackingThreads = threads;
  if (threads > 1)
    fprintf(stderr, "gpu-elegant: using %ld host threads (%s).\n", threads, source);
}

long gpuOmpTrackingEnabled(long particles) {
  return (gpuOmpTrackingScope || gpuMomentumBatchScope) &&
         gpuOmpTrackingRequested(particles);
}

long gpuOmpTrackingScopeActive(void) {
  return gpuOmpTrackingScope || gpuMomentumBatchScope;
}

GPU_OMP_TRACKING_WORKSPACE *gpuGetOmpTrackingWorkspace(long particles) {
  void *memory;
  long allocationParticles = particles;

  if (particles <= gpuOmpTrackingWorkspace.capacity)
    return &gpuOmpTrackingWorkspace;
  if (allocationParticles < gpuOmpTrackingWorkspace.capacity)
    allocationParticles = gpuOmpTrackingWorkspace.capacity;
  memory = realloc(gpuOmpTrackingWorkspace.survived,
                   allocationParticles *
                     sizeof(*gpuOmpTrackingWorkspace.survived));
  if (!memory)
    bombElegant("memory allocation failure for OpenMP survivor flags", NULL);
  gpuOmpTrackingWorkspace.survived = (unsigned char *)memory;
  memory = realloc(gpuOmpTrackingWorkspace.lossOffset,
                   allocationParticles *
                     sizeof(*gpuOmpTrackingWorkspace.lossOffset));
  if (!memory)
    bombElegant("memory allocation failure for OpenMP loss offsets", NULL);
  gpuOmpTrackingWorkspace.lossOffset = (double *)memory;
  memory = realloc(gpuOmpTrackingWorkspace.auxiliary,
                   allocationParticles *
                     sizeof(*gpuOmpTrackingWorkspace.auxiliary));
  if (!memory)
    bombElegant("memory allocation failure for OpenMP tracking workspace",
                NULL);
  gpuOmpTrackingWorkspace.auxiliary = (double *)memory;
  gpuOmpTrackingWorkspace.capacity = allocationParticles;
  return &gpuOmpTrackingWorkspace;
}

/* Moves lost particles' rows aside as the serial loops' swaps do: a particle lost at i trades rows
 * with the last one not yet looked at, which is then looked at in its place, so the OpenMP paths
 * leave the CPU's order.  Row contents move, not row pointers, since CUDA synchronization addresses
 * particles by fixed row.  survived[] moves with them */
long gpuCompactParticlesInCpuOrder(double **particle, long particles, unsigned char *survived) {
  long i = 0, top = particles - 1;

  if (!particle || !survived || particles <= 0)
    return particles > 0 ? particles : 0;
  while (i <= top) {
    if (survived[i])
      i++;
    else {
      double row[MAX_PROPERTIES_PER_PARTICLE];
      unsigned char flag = survived[i];

      /* swapParticles(), which lives in elegant.c */
      memcpy(row, particle[i], totalPropertiesPerParticle * sizeof(*row));
      memcpy(particle[i], particle[top], totalPropertiesPerParticle * sizeof(*row));
      memcpy(particle[top], row, totalPropertiesPerParticle * sizeof(*row));
      survived[i] = survived[top];
      survived[top] = flag;
      top--;
    }
  }
  return top + 1;
}

/* gpuMinEligibleParticles and gpuMinGeneralEligibleParticles from the thresholds as they stand */
static void gpuUpdateMinEligibleParticles(void) {
  /* The first 10 are gpuBase's; the others belong to the types in gpuElementHasOwnThreshold() */
  long thresholds[] = {
    gpuBase.matrixMinParticles, gpuBase.helperMinParticles, gpuBase.reductionMinParticles,
    gpuBase.apertureMinParticles, gpuBase.magnetMinParticles, gpuBase.wakeMinParticles,
    gpuBase.lscMinParticles, gpuBase.csrMinParticles, gpuBase.scmultMinParticles,
    gpuBase.exactDriftMinParticles, gpuCcbendMinParticles, gpuLgbendMinParticles,
    gpuRfdfMinParticles, gpuSreffectsMinParticles, gpuBggexpMinParticles,
    gpuCwigglerMinParticles, gpuFtableMinParticles, gpuBmxyzMinParticles,
    gpuLorentzMinParticles, gpuRfmodeMinParticles, gpuTfeedbackMinParticles,
    gpuHistogramMinParticles, gpuIlmatrixMinParticles, gpuLsrmdltrMinParticles, gpuMotionMinParticles};
  long i;

  gpuMinEligibleParticles = thresholds[0];
  for (i = 1; i < (long)(sizeof(thresholds) / sizeof(thresholds[0])); i++) {
    if (thresholds[i] < gpuMinEligibleParticles)
      gpuMinEligibleParticles = thresholds[i];
    if (i == 9)
      gpuMinGeneralEligibleParticles = gpuMinEligibleParticles;
  }
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
  gpuRfcaReferenceCalls = gpuRfcaReferenceParticles = 0;
  gpuIlmatrixReferenceCalls = 0;
  gpuIsrReferenceFills = gpuIsrReferenceDraws = 0;
  gpuLscReferenceCalls = 0;
  gpuMatchEnergyCpuOrderCalls = gpuMatchEnergySkippedCalls = 0;
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
  gpuMatchEnergyCpuOrderSetting = gpuEnvSet("ELEGANT_GPU_MATCH_ENERGY_CPU_ORDER") ?
    gpuEnvFlag("ELEGANT_GPU_MATCH_ENERGY_CPU_ORDER") : -1;
  gpuBase.helperMinParticles = gpuEnvLong("ELEGANT_GPU_MIN_HELPER_PARTICLES", gpuBase.minParticles);
  gpuBase.exactDriftMinParticles = gpuEnvLong("ELEGANT_GPU_MIN_EXACT_PARTICLES", gpuBase.minParticles);
  gpuEnableOutputBeamSums =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_OUTPUT_BEAM_SUMS") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_OUTPUT_BEAM_SUMS");
  gpuOutputDriftReductionMinParticlesExplicit =
    gpuEnvSetEither("ELEGANT_GPU_MIN_OUTPUT_DRIFT_REDUCTION_PARTICLES",
                    "ELEGANT_GPU_MIN_REDUCTION_PARTICLES");
  gpuBase.reductionMinParticles =
    gpuEnvLongEither("ELEGANT_GPU_MIN_OUTPUT_DRIFT_REDUCTION_PARTICLES",
                     "ELEGANT_GPU_MIN_REDUCTION_PARTICLES",
                     gpuBase.minParticles);
  gpuBase.apertureMinParticles = gpuEnvLong("ELEGANT_GPU_MIN_APERTURE_PARTICLES", gpuBase.minParticles);
  gpuKeepResident = !gpuEnvSet("ELEGANT_GPU_ENABLE_KEEP_RESIDENT") ||
                    gpuEnvFlag("ELEGANT_GPU_ENABLE_KEEP_RESIDENT");
  gpuResidentMinParticles = gpuEnvLong("ELEGANT_GPU_MIN_RESIDENT_PARTICLES", 1000);
  gpuMagnetMinParticlesExplicit = gpuEnvSet("ELEGANT_GPU_MIN_MAGNET_PARTICLES");
  gpuBase.magnetMinParticles = gpuEnvLong("ELEGANT_GPU_MIN_MAGNET_PARTICLES", gpuBase.minParticles);
  gpuWakeMinParticlesExplicit = gpuEnvSet("ELEGANT_GPU_MIN_WAKE_PARTICLES");
  gpuBase.wakeMinParticles = gpuEnvLong("ELEGANT_GPU_MIN_WAKE_PARTICLES", gpuBase.minParticles);
  gpuBase.lscMinParticles = gpuEnvLong("ELEGANT_GPU_MIN_LSC_PARTICLES", gpuBase.wakeMinParticles);
  gpuBase.csrMinParticles = gpuEnvLong("ELEGANT_GPU_MIN_CSR_PARTICLES", gpuBase.minParticles);
  /* No floor on the number of CSR bins since November 13, 2026.  The floor of 1,024 dated from when
   * only the wake was on the device; with the whole bend there, a 512-bin bend is 8 times faster than
   * the CPU at 10,000 particles and 21 times at 1,000,000, with the same particles to the last bit.
   * ELEGANT_GPU_MIN_CSR_BINS=1024 brings the floor back. */
  gpuBase.csrMinBins = gpuEnvLong("ELEGANT_GPU_MIN_CSR_BINS", 0);
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
  gpuApertureParallelCompactionOrderDisabled = 0;
  gpuEnableApertureParallelCompaction =
    gpuApertureParallelCompactionExplicit ?
    gpuEnvFlag("ELEGANT_GPU_ENABLE_APERTURE_PARALLEL_COMPACTION") :
    1;
  /* Off by default since November 10, 2026: one device thread per sum took 5.6 ms for 10,000 particles
   * at every element, ten times the read-only copy and the CPU's sums, with identical output */
  gpuEnableSliceSums = gpuEnvSet("ELEGANT_GPU_ENABLE_SLICE_SUMS") &&
                       gpuEnvFlag("ELEGANT_GPU_ENABLE_SLICE_SUMS");
  gpuMagnetLossCompactionExplicit =
    gpuEnvSet("ELEGANT_GPU_ENABLE_MAGNET_LOSS_COMPACTION");
  /* On with a losses file too since November 13, 2026: gpuMagnetLossRowsFromCpu() writes the lost rows */
  gpuEnableMagnetLossCompaction =
    !gpuMagnetLossCompactionExplicit ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_MAGNET_LOSS_COMPACTION");
  gpuTestMagnetLossDisagreement = gpuEnvFlag("ELEGANT_GPU_TEST_MAGNET_LOSS_DISAGREEMENT");
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
  gpuEnableCsrDriftElement = !gpuEnvSet("ELEGANT_GPU_ENABLE_CSR_DRIFT_ELEMENT") ||
                             gpuEnvFlag("ELEGANT_GPU_ENABLE_CSR_DRIFT_ELEMENT");
  gpuScmultExplicit = gpuEnvSet("ELEGANT_GPU_ENABLE_SCMULT");
  gpuEnableScmult = !gpuScmultExplicit ||
                    gpuEnvFlag("ELEGANT_GPU_ENABLE_SCMULT");
  gpuWakeTrackingDriftExplicit =
    gpuEnvSet("ELEGANT_GPU_ENABLE_WAKE_TRACKING_DRIFT");
  gpuEnableWakeTrackingDrift =
    !gpuWakeTrackingDriftExplicit ||
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
  gpuEnableZimpedance =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_ZIMPEDANCE") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_ZIMPEDANCE");
  gpuEnableZimpedanceGather =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_ZIMPEDANCE_GATHER") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_ZIMPEDANCE_GATHER");
  gpuEnableIbs =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_IBSCATTER") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_IBSCATTER");
  gpuEnableLrwake =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_LRWAKE") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_LRWAKE");
  gpuEnableCorrugated =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_CORRUGATED") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_CORRUGATED");
  gpuEnableTaperApe =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_TAPERAPE") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_TAPERAPE");
  gpuEnableCenter =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_CENTER") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_CENTER");
  gpuEnableIdlePass =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_IDLE_PASS") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_IDLE_PASS");
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
  gpuEnableIlmatrix =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_ILMATRIX") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_ILMATRIX");
  gpuIlmatrixMinParticles =
    gpuEnvLong("ELEGANT_GPU_MIN_ILMATRIX_PARTICLES", 64);
  if (gpuIlmatrixMinParticles < 1)
    gpuIlmatrixMinParticles = 1;
  gpuEnableBggexp =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_BGGEXP") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_BGGEXP");
  gpuEnableBggexpSymplectic =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_BGGEXP_SYMPLECTIC") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_BGGEXP_SYMPLECTIC");
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
  gpuEnableHistogram =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_HISTOGRAM_DIAGNOSTICS") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_HISTOGRAM_DIAGNOSTICS");
  gpuHistogramMinParticles =
    gpuEnvLong("ELEGANT_GPU_MIN_HISTOGRAM_PARTICLES", 64);
  if (gpuHistogramMinParticles < 1)
    gpuHistogramMinParticles = 1;
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
  gpuEnableLorentz =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_LORENTZ") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_LORENTZ");
  gpuLorentzMinParticles =
    gpuEnvLong("ELEGANT_GPU_MIN_LORENTZ_PARTICLES", 64);
  if (gpuLorentzMinParticles < 1)
    gpuLorentzMinParticles = 1;
  gpuEnableLsrmdltr =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_LSRMDLTR") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_LSRMDLTR");
  gpuLsrmdltrMinParticles =
    gpuEnvLong("ELEGANT_GPU_MIN_LSRMDLTR_PARTICLES", 64);
  if (gpuLsrmdltrMinParticles < 1)
    gpuLsrmdltrMinParticles = 1;
  gpuEnableBeambeam =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_BEAMBEAM") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_BEAMBEAM");
  gpuEnableModrf =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_MODRF") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_MODRF");
  gpuEnableRftmez0 =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_RFTMEZ0") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_RFTMEZ0");
  gpuEnableTwla =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_TWLA") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_TWLA");
  gpuEnableMapSolenoid =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_MAPSOLENOID") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_MAPSOLENOID");
  gpuEnableTmcf = !gpuEnvSet("ELEGANT_GPU_ENABLE_TMCF") || gpuEnvFlag("ELEGANT_GPU_ENABLE_TMCF");
  gpuEnableCepl = !gpuEnvSet("ELEGANT_GPU_ENABLE_CEPL") || gpuEnvFlag("ELEGANT_GPU_ENABLE_CEPL");
  gpuEnableTwpl = !gpuEnvSet("ELEGANT_GPU_ENABLE_TWPL") || gpuEnvFlag("ELEGANT_GPU_ENABLE_TWPL");
  gpuEnableTwmta = !gpuEnvSet("ELEGANT_GPU_ENABLE_TWMTA") || gpuEnvFlag("ELEGANT_GPU_ENABLE_TWMTA");
  gpuPortReadSettings();
  gpuMotionMinParticles =
    gpuEnvLong("ELEGANT_GPU_MIN_MOTION_PARTICLES", 64);
  if (gpuMotionMinParticles < 1)
    gpuMotionMinParticles = 1;
  gpuEnableCcbend =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_CCBEND") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_CCBEND");
  gpuCcbendMinParticles =
    gpuEnvLong("ELEGANT_GPU_MIN_CCBEND_PARTICLES", 64);
  if (gpuCcbendMinParticles < 1)
    gpuCcbendMinParticles = 1;
  /* On an RTX PRO 4000 Blackwell, a ring with 160 LGBENDs runs 3 to 12 times
     faster with the kernel at 16384 particles or more, and slower below about
     512. An older test on an RTX 3060 showed no gain through 262144 particles,
     so ELEGANT_GPU_ENABLE_LGBEND=0 keeps LGBEND on the CPU. */
  gpuEnableLgbend =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_LGBEND") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_LGBEND");
  gpuLgbendMinParticles =
    gpuEnvLong("ELEGANT_GPU_MIN_LGBEND_PARTICLES", 512);
  if (gpuLgbendMinParticles < 1)
    gpuLgbendMinParticles = 1;
  gpuEnableRfmode =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_RFMODE") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_RFMODE");
  gpuEnableFrfmode =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_FRFMODE") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_FRFMODE");
  gpuEnableTrfmode =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_TRFMODE") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_TRFMODE");
  gpuEnableFtrfmode =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_FTRFMODE") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_FTRFMODE");
  gpuRfmodeMinParticles =
    gpuEnvLong("ELEGANT_GPU_MIN_RFMODE_PARTICLES", 8192);
  if (gpuRfmodeMinParticles < 1)
    gpuRfmodeMinParticles = 1;
  gpuEnableTfeedback =
    !gpuEnvSet("ELEGANT_GPU_ENABLE_TFEEDBACK") ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_TFEEDBACK");
  gpuTfeedbackMinParticles =
    gpuEnvLong("ELEGANT_GPU_MIN_TFEEDBACK_PARTICLES", 64);
  if (gpuTfeedbackMinParticles < 1)
    gpuTfeedbackMinParticles = 1;
  gpuLscTrackingExplicit = gpuEnvSet("ELEGANT_GPU_ENABLE_LSC");
  gpuEnableLscTracking = !gpuLscTrackingExplicit ||
                         gpuEnvFlag("ELEGANT_GPU_ENABLE_LSC");
  /* These two and ELEGANT_GPU_ENABLE_WAKE_TRACKING_DRIFT above are on by default since November 10,
   * 2026: with them on, every RFCW, WAKE, TRWAKE and momentum matching step gave the CPU's numbers
   * exactly in 56 elegantTestSet decks, and RFCA differed by one unit in the last place only where no
   * output needs the CPU's sin, log and sum order.  Setting a variable to 0 keeps those elements on
   * the CPU. */
  gpuRfcwTrackingDriftExplicit =
    gpuEnvSet("ELEGANT_GPU_ENABLE_RFCW_TRACKING_DRIFT");
  gpuEnableRfcwTrackingDrift =
    !gpuRfcwTrackingDriftExplicit ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_RFCW_TRACKING_DRIFT");
  gpuRfcaChangeP0DriftExplicit =
    gpuEnvSet("ELEGANT_GPU_ENABLE_RFCA_CHANGE_P0_DRIFT");
  gpuEnableRfcaChangeP0Drift =
    !gpuRfcaChangeP0DriftExplicit ||
    gpuEnvFlag("ELEGANT_GPU_ENABLE_RFCA_CHANGE_P0_DRIFT");
#if USE_MPI
  /* gpu-Pelegant keeps the wake, RFCW and (below) change-p0 paths off unless asked: the November 10
   * validation covered the serial build only */
  if (!gpuWakeTrackingDriftExplicit)
    gpuEnableWakeTrackingDrift = 0;
  if (!gpuRfcwTrackingDriftExplicit)
    gpuEnableRfcwTrackingDrift = 0;
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
  gpuEnableIlmatrix = 0;
  gpuEnableBggexp = 0;
  gpuEnableBggexpSymplectic = 0;
  gpuEnableCwiggler = 0;
  gpuEnableHistogram = 0;
  gpuEnableFtable = 0;
  gpuEnableBmxyz = 0;
  gpuEnableLorentz = 0;
  gpuEnableLsrmdltr = 0;
  gpuEnableBeambeam = 0;
  gpuEnableModrf = 0;
  gpuEnableRftmez0 = 0;
  gpuEnableTwla = 0;
  gpuEnableMapSolenoid = 0;
  gpuEnableTmcf = gpuEnableCepl = gpuEnableTwpl = gpuEnableTwmta = 0;
  gpuEnableLgbend = 0;
  gpuEnableRfmode = 0;
  gpuEnableFrfmode = 0;
  gpuEnableTrfmode = 0;
  gpuEnableFtrfmode = 0;
  gpuEnableTfeedback = 0;
#endif
  gpuAvoidShortGpuIslands = !gpuEnvSet("ELEGANT_GPU_AVOID_SHORT_GPU_ISLANDS") ||
                            gpuEnvFlag("ELEGANT_GPU_AVOID_SHORT_GPU_ISLANDS");
  gpuShortGpuIslandMaxElements = gpuEnvLong("ELEGANT_GPU_SHORT_GPU_ISLAND_MAX_ELEMENTS", 4);
  gpuShortGpuIslandByCost = !gpuEnvSet("ELEGANT_GPU_SHORT_GPU_ISLAND_BY_COST") ||
                            gpuEnvFlag("ELEGANT_GPU_SHORT_GPU_ISLAND_BY_COST");
  gpuShortGpuIslandByCostCount = 0;
  /* Apply the batch's dispatch policy once, not in every element's threshold
   * test. The next gpuBaseInit reloads the ordinary environment thresholds. */
  if (gpuMomentumBatchScope) {
    gpuBase.matrixMinParticles = gpuBase.helperMinParticles = 32;
    gpuBase.exactDriftMinParticles = gpuBase.reductionMinParticles = 32;
    gpuBase.apertureMinParticles = gpuBase.magnetMinParticles = 32;
    gpuCcbendMinParticles = gpuLgbendMinParticles = 32;
  }

  /* Scalar reference and boundary-replay tracking deliberately suppresses
   * CUDA.  Leave a valid host-only base for the generic tracking code, but
   * do not rediscover/select a device for every one-particle replay. */
  if (gpuTrackingSuppressed) {
    gpuBase.initialized = 1;
    return;
  }

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

  /* Device 0 is selected at its first use in auto mode, so a run that never puts work on the GPU
   * doesn't create a CUDA context */
  if ((gpuBase.requiredMode || gpuVerbose || gpuBase.activeDevice != 0) && !gpuSelectDevice()) {
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
    if (gpuKeepResident)
      fprintf(stderr,
              "elegant CUDA: a beam on the device stays there down to %ld particles.\n",
              gpuResidentMinParticles);
    if (gpuEnableFtable)
      fprintf(stderr,
              "elegant CUDA: fixed-step FTABLE enabled at %ld particles.\n",
              gpuFtableMinParticles);
    if (gpuEnableBmxyz)
      fprintf(stderr,
              "elegant CUDA: fixed-step BMXYZ enabled at %ld particles.\n",
              gpuBmxyzMinParticles);
    if (gpuEnableLorentz)
      fprintf(stderr,
              "elegant CUDA: fixed-step Lorentz-family tracking enabled at %ld particles.\n",
              gpuLorentzMinParticles);
    if (gpuEnableLsrmdltr)
      fprintf(stderr,
              "elegant CUDA: fixed-step LSRMDLTR enabled at %ld particles.\n",
              gpuLsrmdltrMinParticles);
    if (gpuEnableBeambeam)
      fprintf(stderr, "elegant CUDA: BEAMBEAM kicks enabled with the magnet threshold.\n");
    gpuPortDescribeSettings();
    if (gpuEnableRftmez0 || gpuEnableTwla || gpuEnableMapSolenoid || gpuEnableTmcf || gpuEnableCepl ||
        gpuEnableTwpl || gpuEnableTwmta)
      fprintf(stderr, "elegant CUDA: fixed-step%s%s%s%s%s%s%s enabled at %ld particles.\n",
              gpuEnableRftmez0 ? " RFTMEZ0" : "", gpuEnableTwla ? " TWLA" : "",
              gpuEnableMapSolenoid ? " MAPSOLENOID" : "", gpuEnableTmcf ? " TMCF" : "",
              gpuEnableCepl ? " CEPL" : "", gpuEnableTwpl ? " TWPL" : "", gpuEnableTwmta ? " TWMTA" : "",
              gpuMotionMinParticles);
    if (gpuEnableCcbend)
      fprintf(stderr,
              "elegant CUDA: deterministic order-2 CCBEND enabled at %ld particles.\n",
              gpuCcbendMinParticles);
    if (gpuEnableLgbend)
      fprintf(stderr,
              "elegant CUDA: deterministic order-2 LGBEND enabled at %ld particles.\n",
              gpuLgbendMinParticles);
    if (gpuEnableRfmode || gpuEnableFrfmode ||
        gpuEnableTrfmode || gpuEnableFtrfmode)
      fprintf(stderr,
              "elegant CUDA: resident RFMODE=%ld FRFMODE=%ld TRFMODE=%ld FTRFMODE=%ld at %ld particles.\n",
              gpuEnableRfmode, gpuEnableFrfmode,
              gpuEnableTrfmode, gpuEnableFtrfmode,
              gpuRfmodeMinParticles);
    if (gpuEnableTfeedback)
      fprintf(stderr,
              "elegant CUDA: deterministic TFBPICKUP/TFBDRIVER enabled at %ld particles.\n",
              gpuTfeedbackMinParticles);
    if (gpuEnableSreffects)
      fprintf(stderr,
              "elegant CUDA: deterministic SREFFECTS enabled at %ld particles.\n",
              gpuSreffectsMinParticles);
    if (gpuEnableIlmatrix)
      fprintf(stderr,
              "elegant CUDA: ILMATRIX enabled at %ld particles.\n",
              gpuIlmatrixMinParticles);
    if (gpuEnableHistogram)
      fprintf(stderr,
              "elegant CUDA: HISTOGRAM diagnostics enabled at %ld particles.\n",
              gpuHistogramMinParticles);
  }

  gpuUpdateMinEligibleParticles();

  gpuBase.initialized = 1;
}

/* Setting the device creates its CUDA context, which takes about 0.15 s. A binary holds kernels
 * only for the architectures it was built for, and that first shows at a kernel launch, so the
 * probe finds out while the run can still fall back to the CPU. Callers check this before they
 * put work on the device. */
static long gpuSelectDevice(void) {
  int status;

  if (gpuBase.activeDevice < 0)
    return 0;
  if (gpuDeviceCodeCheckedDevice == gpuBase.activeDevice) {
    if (gpuDeviceCodeStatus == 0)
      return 1;
    gpuBase.activeDevice = -1;
    return 0;
  }
  status = gpuCudaRuntimeSetDevice((int)gpuBase.activeDevice);
  if (status != 0) {
    if (gpuBase.requiredMode)
      gpuRequiredFailure(gpuCudaRuntimeGetErrorString(status));
    gpuBase.activeDevice = -1;
    return 0;
  }
  gpuDeviceCodeStatus = gpuCudaRuntimeProbeDeviceCode();
  gpuDeviceCodeCheckedDevice = gpuBase.activeDevice;
  if (gpuDeviceCodeStatus != 0) {
    char deviceName[256] = "";
    char message[512];

    gpuCudaRuntimeGetDeviceName((int)gpuBase.activeDevice, deviceName, sizeof(deviceName));
    snprintf(message, sizeof(message),
             "this binary has no usable kernels for CUDA device %ld %s (%s)",
             gpuBase.activeDevice, deviceName,
             gpuCudaRuntimeGetErrorString(gpuDeviceCodeStatus));
    if (gpuBase.requiredMode)
      gpuRequiredFailure(message);
    fprintf(stderr, "elegant CUDA: %s; using CPU fallback.\n", message);
    gpuBase.activeDevice = -1;
    return 0;
  }
  return 1;
}

/* Beams between the resident floor and the general thresholds.
 *
 * ELEGANT_GPU_MIN_PARTICLES (10,000) keeps a smaller beam on the CPU unless an element with a threshold
 * of its own takes it, and a beam that losses take below it doesn't return to the device once an
 * element has sent it to the host.  The device's cost for an element hardly depends on the beam size
 * here (10 us for a matrix, 0.39 ms for the CSBEND below) while the CPU's is so many ns a particle, so
 * where the two cross depends on the lattice: with 1,000 particles a CSBEND took 3.2 ms on the CPU and
 * a 32-kick KQUAD 1.6 ms against 0.22 ms (rfRef1, November 11, 2026), and rfRef1, ibScatter3 and
 * scRing1 ran 4 to 6 times faster with the thresholds at 1,000.  Lowering them for every run would
 * not do: of the 117 elegantTestSet decks it touches, 44 got slower, most by the 0.15 s a CUDA context
 * takes, and some because their beams then go back and forth to CPU-only elements (trackingInterrupt1
 * took 24.5 s for 6.3 s).
 *
 * So each tracking call gets an estimate.  gpuSmallBeamWalk() goes through the lattice as tracking
 * would with the thresholds at the floor, and adds up what the elements the device would take cost
 * each side, and the copies between host and device.  From those sums comes the particle count at
 * which the device's total is GPU_SMALL_BEAM_MARGIN times less than the CPU's; when the beam has that
 * many, and the savings so far would pay for a CUDA context, the general thresholds drop to that count
 * for this call.  The estimate uses the lattice, the particle count and the passes, never a clock, so
 * a run takes the same path every time.
 *
 * ELEGANT_GPU_ENABLE_SMALL_BEAMS=0 turns this off.  A run that sets ELEGANT_GPU_MIN_PARTICLES has made
 * its own choice and is left alone, as is any threshold set by name.  gpu-Pelegant keeps the general
 * thresholds: these timings are the serial build's.  ELEGANT_GPU_SMALL_BEAM_REPORT=1 prints each
 * call's sums and choice (2 adds the element types) without the CUDA context ELEGANT_GPU_VERBOSE
 * starts. */
#define GPU_SMALL_BEAM_START_SECONDS 0.3
#define GPU_SMALL_BEAM_MARGIN 1.4
/* us: one copy of the beam between host and device, and the setup and final copy of one tracking call */
#define GPU_SMALL_BEAM_COPY_US 20.0
#define GPU_SMALL_BEAM_COPY_US_PER_PARTICLE 0.01
#define GPU_SMALL_BEAM_CALL_US 250.0

#if !USE_MPI
static double gpuSmallBeamBankedSeconds = 0;

typedef struct {
  long min, matrix, helper, exactDrift, reduction, aperture, magnet, wake, lsc, csr, scmult;
} GPU_GENERAL_THRESHOLDS;

static void gpuGeneralThresholdsGet(GPU_GENERAL_THRESHOLDS *t) {
  t->min = gpuBase.minParticles;
  t->matrix = gpuBase.matrixMinParticles;
  t->helper = gpuBase.helperMinParticles;
  t->exactDrift = gpuBase.exactDriftMinParticles;
  t->reduction = gpuBase.reductionMinParticles;
  t->aperture = gpuBase.apertureMinParticles;
  t->magnet = gpuBase.magnetMinParticles;
  t->wake = gpuBase.wakeMinParticles;
  t->lsc = gpuBase.lscMinParticles;
  t->csr = gpuBase.csrMinParticles;
  t->scmult = gpuBase.scmultMinParticles;
}

static void gpuGeneralThresholdsSet(const GPU_GENERAL_THRESHOLDS *t) {
  gpuBase.minParticles = t->min;
  gpuBase.matrixMinParticles = t->matrix;
  gpuBase.helperMinParticles = t->helper;
  gpuBase.exactDriftMinParticles = t->exactDrift;
  gpuBase.reductionMinParticles = t->reduction;
  gpuBase.apertureMinParticles = t->aperture;
  gpuBase.magnetMinParticles = t->magnet;
  gpuBase.wakeMinParticles = t->wake;
  gpuBase.lscMinParticles = t->lsc;
  gpuBase.csrMinParticles = t->csr;
  gpuBase.scmultMinParticles = t->scmult;
  gpuUpdateMinEligibleParticles();
}

/* Every general threshold the run didn't set by name comes down to count; none goes up */
static void gpuGeneralThresholdsLower(long count) {
  if (gpuBase.minParticles > count)
    gpuBase.minParticles = count;
  if (!gpuMatrixDriftMinParticlesExplicit && gpuBase.matrixMinParticles > count)
    gpuBase.matrixMinParticles = count;
  if (!gpuHelperMinParticlesExplicit && gpuBase.helperMinParticles > count)
    gpuBase.helperMinParticles = count;
  if (!gpuEnvSet("ELEGANT_GPU_MIN_EXACT_PARTICLES") && gpuBase.exactDriftMinParticles > count)
    gpuBase.exactDriftMinParticles = count;
  if (!gpuOutputDriftReductionMinParticlesExplicit && gpuBase.reductionMinParticles > count)
    gpuBase.reductionMinParticles = count;
  if (!gpuEnvSet("ELEGANT_GPU_MIN_APERTURE_PARTICLES") && gpuBase.apertureMinParticles > count)
    gpuBase.apertureMinParticles = count;
  if (!gpuMagnetMinParticlesExplicit && gpuBase.magnetMinParticles > count)
    gpuBase.magnetMinParticles = count;
  if (!gpuWakeMinParticlesExplicit && gpuBase.wakeMinParticles > count)
    gpuBase.wakeMinParticles = count;
  if (!gpuEnvSet("ELEGANT_GPU_MIN_LSC_PARTICLES") && gpuBase.lscMinParticles > count)
    gpuBase.lscMinParticles = count;
  if (!gpuEnvSet("ELEGANT_GPU_MIN_CSR_PARTICLES") && gpuBase.csrMinParticles > count)
    gpuBase.csrMinParticles = count;
  if (!gpuEnvSet("ELEGANT_GPU_MIN_SCMULT_PARTICLES") && gpuBase.scmultMinParticles > count)
    gpuBase.scmultMinParticles = count;
  gpuUpdateMinEligibleParticles();
}

/* What one element costs the CPU, in ns a particle, and the device, in us, for a beam of a few thousand.
 * From show_element_timing on November 11, 2026 (rfRef1, ibScatter3, scRing1, multipoles3, centroid2,
 * feedback2, srEffects2, rcol1, bggexp1 and the APS-U ring).  A CSBEND slice cost the CPU 540 to 800 ns
 * a particle and a KQUAD or KSEXT kick 22 to 71 ns; the low ends are used, so the saving is not
 * overstated.  Returns the kicks or slices counted, 0 for an element with a flat cost. */
static long gpuSmallBeamElementCost(ELEMENT_LIST *eptr, long sums, double *cpuNs, double *deviceUs) {
  long kicks = 0;

  /* a matrix, drift, corrector, monitor or collimator; with centroid or sigma output each element's
   * sums cost both sides more */
  *cpuNs = sums ? 60 : 15;
  *deviceUs = sums ? 40 : 10;
  if (!eptr || !eptr->p_elem)
    return 0;
  switch (eptr->type) {
  case T_CSBEND:
    kicks = ((CSBEND *)eptr->p_elem)->nSlices;
    if (kicks < 1)
      kicks = 1;
    *cpuNs += 500.0 * kicks;
    *deviceUs += 100.0 * kicks;
    return kicks;
  case T_KQUAD:
    kicks = MAX(((KQUAD *)eptr->p_elem)->n_kicks, ((KQUAD *)eptr->p_elem)->nSlices);
    break;
  case T_KSEXT:
    kicks = MAX(((KSEXT *)eptr->p_elem)->n_kicks, ((KSEXT *)eptr->p_elem)->nSlices);
    break;
  case T_KOCT:
    kicks = MAX(((KOCT *)eptr->p_elem)->n_kicks, ((KOCT *)eptr->p_elem)->nSlices);
    break;
  case T_KQUSE:
    kicks = MAX(((KQUSE *)eptr->p_elem)->n_kicks, ((KQUSE *)eptr->p_elem)->nSlices);
    break;
  case T_DQCOR:
    /* DQCOR2, 100 slices at order 6: 44 ms on the CPU with 1,000 particles, under 5 ms on the device */
    kicks = ((DQCOR *)eptr->p_elem)->nSlices;
    if (kicks < 1)
      kicks = 1;
    *cpuNs += 300.0 * kicks;
    *deviceUs += 50.0 * kicks;
    return kicks;
  case T_CCBEND:
    *cpuNs += 300;
    *deviceUs += 100;
    return 1;
  case T_LGBEND:
    *cpuNs += 10000;
    *deviceUs += 850;
    return 1;
  case T_BGGEXP:
    /* bggexp1: 2.8 ms a particle on the CPU, 0.19 s on the device for any small beam */
    if (((BGGEXP *)eptr->p_elem)->symplectic && ((BGGEXP *)eptr->p_elem)->zInterval > 0) {
      /* about 4 potential evaluations a step, each twice a field evaluation, every Z_INTERVAL planes */
      *cpuNs += 6000000.0 / ((BGGEXP *)eptr->p_elem)->zInterval;
      *deviceUs += 800000.0 / ((BGGEXP *)eptr->p_elem)->zInterval;
      return 1;
    }
    *cpuNs += 1500000;
    *deviceUs += 200000;
    return 1;
  case T_RFCA:
  case T_RFCW:
    *cpuNs += 10;
    *deviceUs += 55;
    return 0;
  case T_SREFFECTS:
    /* srEffects2, 40 of them and 121 matrices a pass: 6.9 ms on the CPU, 3.9 ms on the device */
    *cpuNs += 95;
    *deviceUs += 55;
    return 0;
  case T_ILMATRIX:
    /* 57 ns a particle on the CPU at 100,000 particles (November 12, 2026); the device's cost is the
     * two kernels and, with output in CPU order, the copies for the host's sines and cosines */
    *cpuNs += 40;
    *deviceUs += 60;
    return 0;
  case T_TFBPICKUP:
  case T_TFBDRIVER:
    *cpuNs = 7;
    *deviceUs = 21;
    return 0;
  case T_IBSCATTER:
    /* ibs_tracking.c: about 75 ns a particle besides the growth rates, which both sides pay */
    *cpuNs += 60;
    *deviceUs += 400;
    return 0;
  case T_CORGPIPE:
  case T_CORGPLATES:
    /* two exact drifts around a WAKE: the drifts, the binning and the kicks on both sides */
    *cpuNs += 60;
    *deviceUs += 150;
    return 0;
  case T_LRWAKE:
    /* track_through_lrwake(): the time, sums and kicks, about 40 ns a particle; the device's values, the copy
     * to the host for the sums in particle order, and the kick */
    *cpuNs += 40;
    *deviceUs += 150;
    return 0;
  case T_RFTMEZ0:
  case T_TWLA:
  case T_MAPSOLENOID: {
    /* November 14, 2026, elegant: RFTMEZ0 6.7 us a particle at 50 steps (ria1), TWLA 74 us at 1,000 (twla1),
     * MAPSOLENOID 130 us at 1,000 (mapSolenoid1); the device a step at a time like LSRMDLTR */
    long steps = eptr->type == T_RFTMEZ0 ? ((RFTMEZ0 *)eptr->p_elem)->n_steps :
                 (eptr->type == T_TWLA ? ((TW_LINAC *)eptr->p_elem)->n_steps :
                  ((MAP_SOLENOID *)eptr->p_elem)->n_steps);
    *cpuNs += (eptr->type == T_TWLA ? 74.0 : 132.0) * steps;
    *deviceUs += 20.0 * steps;
    return 0;
  }
  case T_TMCF:
  case T_CEPL:
  case T_TWPL:
  case T_TWMTA: {
    /* feature table 12, elegant at 100 steps: TMCF 74 ns a particle a step, CEPL 52, TWPL 58, TWMTA 102 */
    long steps = eptr->type == T_TMCF ? ((TMCF_MODE *)eptr->p_elem)->n_steps :
                 (eptr->type == T_CEPL ? ((CE_PLATES *)eptr->p_elem)->n_steps :
                  (eptr->type == T_TWPL ? ((TW_PLATES *)eptr->p_elem)->n_steps : ((TWMTA *)eptr->p_elem)->n_steps));
    *cpuNs += (eptr->type == T_TMCF ? 74.0 : (eptr->type == T_CEPL ? 52.0 : (eptr->type == T_TWPL ? 58.0 : 102.0))) * steps;
    *deviceUs += 20.0 * steps;
    return 0;
  }
  case T_MODRF:
    /* November 14, 2026: elegant 25 ns a particle on the PAR ring's MODRF (mean time and kick); the device
     * its particle times to the host for the mean in particle order, then RAMPRF's kick */
    *cpuNs += 25;
    *deviceUs += 60;
    return 0;
  case T_BEAMBEAM:
    /* November 14, 2026: elegant 133 ns a particle on a collider ring's gaussian BEAMBEAM (Faddeeva::w
     * twice a particle); the device about like nonlinear SCMULT's kick */
    *cpuNs += 130;
    *deviceUs += 60;
    return 0;
  case T_LSRMDLTR:
    /* November 14, 2026, feature table's lsrmdltr row (100 steps, a laser): elegant 19.1 us a particle,
     * about 190 ns a particle a step; the device 4.9 ms a call at 1,000 particles, where a thread's
     * steps one after another set the time, about 49 us a step */
    *cpuNs += 190.0 * ((LSRMDLTR *)eptr->p_elem)->nSteps;
    *deviceUs += 49.0 * ((LSRMDLTR *)eptr->p_elem)->nSteps;
    return 0;
  case T_ZLONGIT:
  case T_ZTRANSVERSE:
    /* November 14, 2026: elegant 49 ns a particle on the feature table's ZLONGIT, 84 ns a particle a
     * bunch call of 10,000 in the collective ring; the device 0.6 to 0.7 ms a call at 10,000 particles
     * (the time span, sort, ordered histogram, the FFTs and the copies), little more at 100,000 */
    *cpuNs += 50;
    *deviceUs += 600;
    return 0;
  default:
    gpuPortSmallBeamCost(eptr, cpuNs, deviceUs);
    return 0;
  }
  if (kicks < 1)
    kicks = 1;
  *cpuNs += 25.0 * kicks;
  *deviceUs += 20 + 5.0 * kicks;
  return kicks;
}

typedef struct {
  double cpuNs, deviceUs;  /* one pass of the elements the device would take */
  long deviceElements, cpuElements, uploads, downloads;
} GPU_SMALL_BEAM_WALK;

/* What the last walks of a beamline found.  Asking each element whether the device can take it costs
 * far more than reading its type, and an optimizer may track a small beam thousands of times, so once
 * two walks in a row agree, a later call with the same elements, beam size and output needs reuses the
 * sums. */
static struct {
  void *beamline;
  unsigned long long signature;
  long outputNeeds, nParticles, laps, agreed;
  GPU_SMALL_BEAM_WALK walk;
} gpuSmallBeamCache;

/* One pass as tracking would make it with the general thresholds at the floor: which elements the
 * device takes, and where the beam changes sides.  first is the list tracking uses, which with
 * concat_order is the concatenated one.  With laps of 2 the second pass is the one counted,
 * as a ring's beam starts its later passes where the pass before left it.  types, if not NULL, has
 * 3*N_TYPES counts: device elements, their kicks, CPU elements. */
static void gpuSmallBeamWalk(ELEMENT_LIST *first, long nParticles, long laps, long sums,
                             GPU_SMALL_BEAM_WALK *walk, long *types) {
  ELEMENT_LIST *eptr;
  long lap, onDevice = 0;

  for (lap = 0; lap < laps; lap++) {
    memset(walk, 0, sizeof(*walk));
    if (types)
      memset(types, 0, 3 * N_TYPES * sizeof(*types));
    for (eptr = first; eptr; eptr = eptr->succ) {
      double cpuNs = 0, deviceUs = 0;
      long kicks;

      if (eptr->ignore || gpuPassiveElementSupported(eptr, nParticles))
        continue;
      if (!gpuElementEligible(eptr, nParticles)) {
        if (onDevice)
          walk->downloads++;
        onDevice = 0;
        walk->cpuElements++;
        if (types && eptr->type >= 0 && eptr->type < N_TYPES)
          types[2 * N_TYPES + eptr->type]++;
        continue;
      }
      if (!onDevice) {
        if (gpuShortGpuIslandAhead(eptr, nParticles, 1, NULL)) {
          walk->cpuElements++;
          if (types && eptr->type >= 0 && eptr->type < N_TYPES)
            types[2 * N_TYPES + eptr->type]++;
          continue;
        }
        walk->uploads++;
        onDevice = 1;
      }
      kicks = gpuSmallBeamElementCost(eptr, sums, &cpuNs, &deviceUs);
      walk->cpuNs += cpuNs;
      walk->deviceUs += deviceUs;
      walk->deviceElements++;
      if (types && eptr->type >= 0 && eptr->type < N_TYPES) {
        types[eptr->type]++;
        types[N_TYPES + eptr->type] += kicks;
      }
    }
  }
  /* a single pass ends with the beam's return to the host */
  if (laps == 1 && onDevice)
    walk->downloads++;
}

static void gpuSmallBeamReportTypes(const long *types) {
  long i, section;
  static const char *label[2] = {"device", "CPU"};

  for (section = 0; section < 2; section++) {
    const long *count = types + (section ? 2 * N_TYPES : 0);
    fprintf(stderr, "elegant CUDA small beam:   %s:", label[section]);
    for (i = 0; i < N_TYPES; i++)
      if (count[i]) {
        if (!section && types[N_TYPES + i])
          fprintf(stderr, " %s=%ld/%ld", entity_name[i], count[i], types[N_TYPES + i]);
        else
          fprintf(stderr, " %s=%ld", entity_name[i], count[i]);
      }
    fputc('\n', stderr);
  }
}

#endif

void gpuChooseSmallBeamThresholds(void *beamline0, void *firstElement, long nParticles, long nPasses) {
#if USE_MPI
  (void)beamline0;
  (void)firstElement;
  (void)nParticles;
  (void)nPasses;
#else
  ELEMENT_LIST *first = (ELEMENT_LIST *)firstElement, *eptr;
  GPU_GENERAL_THRESHOLDS standing;
  GPU_SMALL_BEAM_WALK walk;
  double cpuUsPerParticle, fixedUs, copyUsPerParticle, breakEven, callSeconds, cpuUs, deviceUs;
  long floorParticles = gpuResidentMinParticles, general = gpuBase.minParticles;
  long contextExists, outputNeeds, sums, laps, nWalk, lowered, report, walked = 0;
  unsigned long long signature = 1469598103934665603ULL;
  static long types[3 * N_TYPES];

  if (!gpuBase.initialized || gpuBase.activeDevice < 0 || gpuBase.requiredMode || !first ||
      nPasses < 1 || !gpuKeepResident || floorParticles < 1 || general <= floorParticles ||
      nParticles < floorParticles || gpuEnvSet("ELEGANT_GPU_MIN_PARTICLES") ||
      (gpuEnvSet("ELEGANT_GPU_ENABLE_SMALL_BEAMS") && !gpuEnvFlag("ELEGANT_GPU_ENABLE_SMALL_BEAMS")))
    return;
  report = gpuEnvLong("ELEGANT_GPU_SMALL_BEAM_REPORT", 0);
  /* a beam at or above the general thresholds uses the device already; what matters for it is
   * whether it comes back to the device after losses have taken it under them */
  nWalk = nParticles < general ? nParticles : general - 1;
  laps = nPasses > 1 ? 2 : 1;
  sums = gpuBase.reductionOutputNeeded ? 1 : 0;

  /* the elements' types, kicks and ignore flags, and what the run writes */
  for (eptr = first; eptr; eptr = eptr->succ) {
    double cpuNs = 0, elementUs = 0;

    signature = (signature ^ (unsigned long long)eptr->type) * 1099511628211ULL;
    signature = (signature ^ (unsigned long long)(eptr->ignore ? 1 : 0)) * 1099511628211ULL;
    if (!eptr->ignore)
      signature = (signature ^ (unsigned long long)gpuSmallBeamElementCost(eptr, sums, &cpuNs, &elementUs)) *
                  1099511628211ULL;
  }
  outputNeeds = gpuBase.orderSensitiveOutputNeeded + 2 * gpuBase.reductionOutputNeeded +
                4 * gpuBase.lossOutputNeeded + 8 * gpuBase.backtrack + 16 * gpuRunAlwaysChangeP0;
  if (gpuSmallBeamCache.beamline == beamline0 && gpuSmallBeamCache.signature == signature &&
      gpuSmallBeamCache.outputNeeds == outputNeeds && gpuSmallBeamCache.nParticles == nWalk &&
      gpuSmallBeamCache.laps == laps && gpuSmallBeamCache.agreed >= 2) {
    walk = gpuSmallBeamCache.walk;
  } else {
    gpuGeneralThresholdsGet(&standing);
    gpuGeneralThresholdsLower(floorParticles);
    gpuSmallBeamWalk(first, nWalk, laps, sums, &walk, report > 1 ? types : NULL);
    gpuGeneralThresholdsSet(&standing);
    walked = 1;
    if (gpuSmallBeamCache.beamline == beamline0 && gpuSmallBeamCache.signature == signature &&
        gpuSmallBeamCache.outputNeeds == outputNeeds && gpuSmallBeamCache.nParticles == nWalk &&
        gpuSmallBeamCache.laps == laps && !memcmp(&gpuSmallBeamCache.walk, &walk, sizeof(walk)))
      gpuSmallBeamCache.agreed++;
    else
      gpuSmallBeamCache.agreed = 1;
    gpuSmallBeamCache.beamline = beamline0;
    gpuSmallBeamCache.signature = signature;
    gpuSmallBeamCache.outputNeeds = outputNeeds;
    gpuSmallBeamCache.nParticles = nWalk;
    gpuSmallBeamCache.laps = laps;
    gpuSmallBeamCache.walk = walk;
  }

  /* us a pass: the CPU's cpuUsPerParticle*n, the device's fixedUs + copyUsPerParticle*n */
  cpuUsPerParticle = walk.cpuNs * 1e-3;
  fixedUs = walk.deviceUs + (walk.uploads + walk.downloads) * GPU_SMALL_BEAM_COPY_US +
            GPU_SMALL_BEAM_CALL_US / nPasses;
  copyUsPerParticle = (walk.uploads + walk.downloads) * GPU_SMALL_BEAM_COPY_US_PER_PARTICLE;
  cpuUs = cpuUsPerParticle * nWalk;
  deviceUs = fixedUs + copyUsPerParticle * nWalk;
  callSeconds = nPasses * (cpuUs - deviceUs) * 1e-6;
  breakEven = -1;
  if (walk.deviceElements && cpuUsPerParticle > GPU_SMALL_BEAM_MARGIN * copyUsPerParticle)
    breakEven = ceil(GPU_SMALL_BEAM_MARGIN * fixedUs /
                     (cpuUsPerParticle - GPU_SMALL_BEAM_MARGIN * copyUsPerParticle));
  contextExists = gpuDeviceCodeCheckedDevice == gpuBase.activeDevice && gpuDeviceCodeStatus == 0;

  if (report || gpuVerbose) {
    fprintf(stderr,
            "elegant CUDA small beam: %ld particles (%ld walked), %ld passes, context %ld, sums %ld: device "
            "takes %ld elements, CPU %ld, %ld uploads, %ld downloads; a pass costs the CPU %.1f us, the device "
            "%.1f us; break-even %.0f particles.\n",
            nParticles, nWalk, nPasses, contextExists, sums, walk.deviceElements, walk.cpuElements,
            walk.uploads, walk.downloads, cpuUs, deviceUs, breakEven);
    if (report > 1 && walked)
      gpuSmallBeamReportTypes(types);
  }
  if (breakEven < 0 || breakEven > nWalk) {
    if (report || gpuVerbose)
      fprintf(stderr, "elegant CUDA small beam: general thresholds stay at %ld.\n", general);
    return;
  }
  if (!contextExists && nParticles < general &&
      gpuSmallBeamBankedSeconds + callSeconds < GPU_SMALL_BEAM_START_SECONDS) {
    /* not yet worth a CUDA context; later calls like this one may make it so */
    gpuSmallBeamBankedSeconds += callSeconds;
    if (report || gpuVerbose)
      fprintf(stderr,
              "elegant CUDA small beam: general thresholds stay at %ld: an estimated %.3f s saved in this "
              "call, %.3f s in all so far, is less than the %.1f s a CUDA context is worth.\n",
              general, callSeconds, gpuSmallBeamBankedSeconds, GPU_SMALL_BEAM_START_SECONDS);
    return;
  }

  lowered = breakEven > floorParticles ? (long)breakEven : floorParticles;
  gpuGeneralThresholdsLower(lowered);
  if (report || gpuVerbose)
    fprintf(stderr,
            "elegant CUDA small beam: general thresholds lowered to %ld for this call (an estimated %.3f s "
            "saved over %ld passes).\n",
            lowered, callSeconds, nPasses);
#endif
}

/* A tracking call that can't use a device never gets one back, so the per-element calls would only mark
 * the host current again. They cost a one-particle beam tracked for many passes about 20% of its run. */
long gpuElementHooksNeeded(void) {
#ifdef GPU_VERIFY
  return 1;
#else
  return !gpuBase.initialized || gpuBase.activeDevice >= 0;
#endif
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
  gpuCudaReductionRelease();
  gpuReleaseRfcaScratch();
  gpuReleaseIlmatrixScratch();
  gpuReleasePolynomialSeriesCache();
  gpuReleaseBatchedSearchScratch();
  gpuCudaPolynomialSeriesRelease();
  gpuCudaBggexpRelease();
  gpuCudaCwigglerRelease();
  gpuCudaFtableRelease();
  gpuCudaBmxyzRelease();
  gpuCudaLorentzRelease();
  gpuCudaLsrmdltrRelease();
  gpuCudaMotionRelease();
  gpuCudaRfParticleTimesRelease();
  gpuCudaRfmodeRelease();
  gpuCudaHistogramRelease();
  gpuCudaScmultRelease();
  gpuReleaseApertureScratch();
  gpuReleaseCsrScratch();
  gpuCudaCombinedWakeRelease();
  gpuCudaBunchGatherRelease();
  gpuCudaIbsRelease();
  gpuCudaLrwakeRelease();
  free(gpuBunchRangeCache.start);
  free(gpuBunchRangeCache.count);
  memset(&gpuBunchRangeCache, 0, sizeof(gpuBunchRangeCache));
  free(gpuMatchEnergyDelta);
  gpuMatchEnergyDelta = NULL;
  gpuMatchEnergyDeltaCapacity = 0;
  gpuCudaColumnSnapshotRelease();
  gpuMatchEnergyMatchedValid = 0;
  free(gpuRfmodeHostHistogram);
  gpuRfmodeHostHistogram = NULL;
  gpuRfmodeHostHistogramCapacity = 0;
  free(gpuRfmodeHostXsum);
  gpuRfmodeHostXsum = NULL;
  free(gpuRfmodeHostYsum);
  gpuRfmodeHostYsum = NULL;
  free(gpuRfmodeHostTime);
  gpuRfmodeHostTime = NULL;
  gpuRfmodeHostTimeCapacity = 0;
  free(gpuHistogramHostBins);
  gpuHistogramHostBins = NULL;
  gpuHistogramHostBinCapacity = 0;
  gpuBase.elementOnGpu = 0;
  gpuBase.element = NULL;
}

#ifndef GPU_VERIFY
/* Types whose GPU paths use a threshold of their own rather than only gpuBase's. A type left off
 * this list would only stay on the CPU between the two thresholds. */
static long gpuElementHasOwnThreshold(long type) {
  switch (type) {
  case T_CCBEND:
  case T_LGBEND:
  case T_RFDF:
  case T_SREFFECTS:
  case T_ILMATRIX:
  case T_BGGEXP:
  case T_CWIGGLER:
  case T_FTABLE:
  case T_BMAPXYZ:
  case T_BMAPXY:
  case T_NIBEND:
  case T_NISEPT:
  case T_LSRMDLTR:
  case T_RFTMEZ0:
  case T_TWLA:
  case T_MAPSOLENOID:
  case T_TMCF:
  case T_CEPL:
  case T_TWPL:
  case T_TWMTA:
  case T_RFMODE:
  case T_FRFMODE:
  case T_TRFMODE:
  case T_FTRFMODE:
  case T_TFBPICKUP:
  case T_TFBDRIVER:
  case T_HISTOGRAM:
    return 1;
  default:
    return 0;
  }
}
#endif

void gpuSetTrackingPass(long iPass, long passOffset) {
  gpuTrackingPass = iPass;
  gpuTrackingPassOffset = passOffset;
}

/* Elements that leave the particles alone on this pass, which do_tracking() decides from the pass number:
 * BRANCH (it only picks the next element), TWISS applied once after its first pass or disabled, POLAR off its
 * ON_PASS, CENTER off its ON_PASS, SCATTER outside START_ON_PASS..END_ON_PASS.  Plain tracking only: the
 * batched programs neither branch nor know the pass. */
static long gpuIdlePassElement(ELEMENT_LIST *eptr) {
  if (!gpuEnableIdlePass || !eptr || !eptr->p_elem || gpuTrackingPass < 0)
    return 0;
  switch (eptr->type) {
  case T_BRANCH:
    return 1;
  case T_TWISSELEMENT: {
    TWISSELEMENT *twiss = (TWISSELEMENT *)eptr->p_elem;
    return twiss->disable || (twiss->applyOnce && gpuTrackingPass != gpuTrackingPassOffset);
  }
  case T_POLAR:
    return ((POLAR *)eptr->p_elem)->onPass != gpuTrackingPass;
  case T_CENTER: {
    CENTER *center = (CENTER *)eptr->p_elem;
    return center->onPass >= 0 && center->onPass != gpuTrackingPass;
  }
  case T_SCATTER: {
    SCATTER *scatter = (SCATTER *)eptr->p_elem;
    return gpuTrackingPass < scatter->startOnPass ||
           (scatter->endOnPass >= 0 && gpuTrackingPass > scatter->endOnPass);
  }
  default:
    return 0;
  }
}

void setElementGpuData(void *eptr0, long nParticles) {
  ELEMENT_LIST *eptr = (ELEMENT_LIST *)eptr0;
  long eligible = 0;

  gpuBase.element = eptr0;
  gpuBase.nParticles = nParticles;
  gpuBase.elementOnGpu = 0;
#ifdef GPU_VERIFY
  gpuVerifyElementTrackedOnGpu = 0;
#endif

  if (!gpuBase.initialized || gpuBase.activeDevice < 0 || !eptr || nParticles <= 0) {
    if (gpuBase.deviceCurrent && !gpuBase.hostCurrent)
      gpuForceCpuForElement(eptr, "CPU element after CUDA element");
    gpuMarkHostWillChange();
    return;
  }

#ifndef GPU_VERIFY
  /* A beam on the host below every threshold this element's paths use can't go on the device, so the
   * element checks needn't run. They cost searches that track one particle, or one batch of trials, at a
   * time 10% to 40% of their run. */
  if (!gpuBase.deviceCurrent && !gpuBase.requiredMode &&
      nParticles < (gpuElementHasOwnThreshold(eptr->type) ? gpuMinEligibleParticles :
                    gpuMinGeneralEligibleParticles)) {
    gpuMarkHostWillChange();
    return;
  }
#endif

  eligible = gpuCurrentElementEligible();

  if (eligible) {
    if (gpuShouldUseCpuForShortGpuIsland(eptr, nParticles)) {
      gpuBase.gpuShortGpuIslandCpuCount++;
      gpuMarkHostWillChange();
      return;
    }
    if (!gpuSelectDevice()) {
      gpuMarkHostWillChange();
      return;
    }
    gpuBase.elementOnGpu = 1;
    gpuBase.gpuElementCount++;
    return;
  }

#ifdef GPU_VERIFY
  /* compareGpuCpu leaves only the CPU's coordinates current, which would keep every WATCH on its
   * CPU path in this build, so its GPU reductions and el would never be checked. */
  if (gpuBase.verifyMode && eptr->type == T_WATCH && !gpuBase.deviceCurrent)
    gpuCopyHostToDevice(nParticles);
#endif
  if (gpuBase.deviceCurrent &&
      (gpuPassiveElementSupported(eptr, nParticles) || gpuIdlePassElement(eptr))) {
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

#if !USE_MPI
/* The batched-tune settings, which gpuBaseInit() reads at the first tracking call.  A tune
 * footprint's first batch can come before any tracking, so read them here until then. */
static long gpuBatchedTuneEnabledNow(void) {
  if (gpuBase.initialized)
    return gpuEnableBatchedTuneTracking;
  return !gpuEnvSet("ELEGANT_GPU_ENABLE_BATCHED_TUNE_TRACKING") ||
         gpuEnvFlag("ELEGANT_GPU_ENABLE_BATCHED_TUNE_TRACKING");
}

static long gpuBatchedTuneMinParticlesNow(void) {
  long particles;

  if (gpuBase.initialized)
    return gpuBatchedTuneMinParticles;
  particles = gpuEnvLong("ELEGANT_GPU_MIN_BATCHED_TUNE_PARTICLES", 32);
  return particles < 1 ? 1 : particles;
}
#endif

long gpu_batched_tune_tracking_enabled(long particles) {
#if USE_MPI
  (void)particles;
  return 0;
#else
  if (obstructionDataPresent && obstructionDataPresent())
    return 0;
  return gpuBatchedTuneEnabledNow() && particles >= gpuBatchedTuneMinParticlesNow();
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
      /* Until CPU fix 21 is committed (elegant-CPU-bugs-and-fixes.md), polynomialSeries_tracking() stops
       * the pass at a particle with too much transverse momentum, which freezes every particle after it
       * in a batch; elegant tracks a footprint's particles one at a time.  Remove with that fix. */
      return 0;
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

#if !USE_MPI
static void gpuTuneProgramRelease(GPU_TUNE_PROGRAM *program) {
  if (!program)
    return;
  free(program->op);
  free(program->drift);
  free(program->matrix);
  free(program->multipole);
  free(program->exactCorrector);
  free(program->csbend);
  free(program->ccbend);
  free(program->lgbend);
  free(program->lgbendLocalAperture);
  free(program->collimator);
  free(program->scraper);
  free(program->taperAperture);
  free(program->speedbump);
  free(program->malign);
  free(program->postAperture);
  memset(program, 0, sizeof(*program));
}

/* A monitor tracks as its drift matrix; one without length is a no-op. */
static double gpuTuneProgramMonitorLength(ELEMENT_LIST *eptr) {
  switch (eptr->type) {
  case T_MONI:
    return ((MONI *)eptr->p_elem)->length;
  case T_HMON:
    return ((HMON *)eptr->p_elem)->length;
  case T_VMON:
    return ((VMON *)eptr->p_elem)->length;
  default:
    return 0;
  }
}

/* An element the CPU tracks with its matrix alone, such as QUAD, SBEN or SEXT: the program applies
 * the same matrix in the same order of operations */
static long gpuTuneProgramMatrixTracked(ELEMENT_LIST *eptr) {
  unsigned long flags;

  if (!eptr || eptr->type < 0 || eptr->type >= N_TYPES)
    return 0;
  flags = entity_description[eptr->type].flags;
  return (flags & MATRIX_TRACKING) && (flags & HAS_MATRIX);
}

/* Whether the program's arithmetic for this element is the CPU's, bit for bit.  The verify build,
 * which compares every particle, found these exceptions on November 11, 2026: a MULT with length
 * (3e-14) and a TAPERAPR with TILT (2e-16, fixed November 14), and the elements below that take a function from the
 * math library for each particle, where the device's library and the host's can round differently.
 * Kick magnets were exact with integration of order 2, 4 and 6, every MALIGN_METHOD with and without
 * offsets, DX, DY, DZ, TILT, PITCH, YAW, FSE, steering, edge effects, LEFFECTIVE, multipole files and
 * EXPAND_HAMILTONIAN.  Three of those needed the kernel corrected the same day: a linear
 * MALIGN_METHOD with no misalignment, DZ, and LEFFECTIVE.  The matrix correctors, collimators,
 * scraper, SPEEDBUMP, TAPERAPC, MALIGN and every matrix were exact.  LGBEND is counted apart. */
static long gpuTuneProgramExactElement(ELEMENT_LIST *eptr) {
  if (!eptr || !eptr->p_elem)
    return 1;
  switch (eptr->type) {
  case T_MULT:
    return !((MULT *)eptr->p_elem)->length;
  /* TAPERAPR with TILT was 2e-16 off until November 14, 2026 (path length from the slopes rotated back) */
  /* An exact corrector with a kick takes atan(), sin(), asin(), cos() and tan() for each particle,
   * from the device's library where the CPU takes the host's.  At 0.2 mrad 20,000 particles came out
   * the same; at 20 mrad 124 values differed in the last place.  With no kick the CPU makes it a drift. */
  case T_EHCOR:
    return ((EHCOR *)eptr->p_elem)->kick * ((EHCOR *)eptr->p_elem)->calibration == 0;
  case T_EVCOR:
    return ((EVCOR *)eptr->p_elem)->kick * ((EVCOR *)eptr->p_elem)->calibration == 0;
  case T_EHVCOR: {
    EHVCOR *corrector = (EHVCOR *)eptr->p_elem;

    return corrector->xkick * corrector->xcalibration == 0 && corrector->ykick * corrector->ycalibration == 0;
  }
  case T_CSBEND: {
    /* Until the CSBEND drift formula (elegant-CPU-bugs-and-fixes.md, Part 4) is committed, elegant's body
     * takes asin(), sin() and cos() for each particle, which the device can't match bit for bit.  Remove
     * this return with that change. */
    if (eptr->type == T_CSBEND)
      return 0;
    /* The body is the CPU's since November 11, 2026, when both sides took the drift that needs no
     * trigonometric function for a particle (csbend.c: computeCsbendDriftSteps(); here:
     * gpuPackCsbendDriftSteps()), and the device took the CPU's division and multiplication by 1+delta
     * after every kick.  An edge is the CPU's where it takes no function from the math library for a
     * particle: no edge effects, K. Hwang's fringe (2), and the default edge (1) and the simple symplectic
     * kick (3) with no fringe angle, that is with HGAP*FINT zero.  With a fringe angle those two take
     * tan(e - psi/(1+delta)).  Models 4 and 5 take exp() and hyperbolic functions, and ISR takes pow(). */
    CSBEND *csbend = (CSBEND *)eptr->p_elem;
    long edge;

    if (csbend->isr)
      return 0;
    for (edge = 0; edge < 2; edge++) {
      long index = edge ? csbend->e2Index : csbend->e1Index;
      long effect = csbend->edge_effects[index];
      double fint = csbend->fint[index] >= 0 ? csbend->fint[index] : csbend->fintBoth;

      if (!(csbend->edgeFlags & (edge ? BEND_EDGE2_EFFECTS : BEND_EDGE1_EFFECTS)) || effect == 0 || effect == 2)
        continue;
      if ((effect == 1 || effect == 3) && 2 * csbend->hgap * fint == 0)
        continue;
      return 0;
    }
    return 1;
  }
  case T_CCBEND: {
    /* Exact in 59 configurations the same day, once the device repeated the CPU's per-slice momentum
     * round trip (gpuPackCcbendTracking()) and its order in the fringe's last product.  The FRINGEMODEL=1
     * fringe takes exp() from the device's library where the CPU takes it from the host's: with I1 zero
     * its argument is 1e-20, and with K5 zero or EDGE_ORDER under 2 it is zero, and both give 1.  Any
     * other argument isn't claimed, though 20,000 particles came out the same. */
    CCBEND *ccbend = (CCBEND *)eptr->p_elem;

    return !(ccbend->fringeModel == 1 &&
             (ccbend->fringeInt1[3] || ccbend->fringeInt2[3] ||
              (ccbend->edgeOrder >= 2 && (ccbend->fringeInt1[5] || ccbend->fringeInt2[5]))));
  }
  default:
    return 1;
  }
}

static long gpuTuneProgramCountElement(GPU_TUNE_PROGRAM *program,
                                       ELEMENT_LIST *eptr) {
  if (!program || !eptr || eptr->ignore)
    return 1;
  switch (eptr->type) {
  case T_MARK:
  case T_RECIRC:
    return 1;
  case T_HMON:
  case T_VMON:
  case T_MONI:
    if (!gpuTuneProgramMonitorLength(eptr))
      return 1;
    program->matrixCount++;
    break;
  case T_EDRIFT:
  case T_RFCA:
    program->driftCount++;
    break;
  case T_DRIF:
  case T_HCOR:
  case T_VCOR:
    program->matrixCount++;
    break;
  case T_MULT:
  case T_KQUAD:
  case T_KSEXT:
  case T_KOCT:
  case T_DQCOR:
    program->multipoleCount++;
    break;
  case T_EHCOR:
  case T_EVCOR:
  case T_EHVCOR:
    program->exactCorrectorCount++;
    break;
  case T_CSBEND:
    program->csbendCount++;
    break;
  case T_CCBEND:
    program->ccbendCount++;
    break;
  case T_LGBEND:
    program->lgbendCount++;
    program->lgbendLocalApertureCount++;
    break;
  case T_RCOL:
  case T_ECOL:
    program->collimatorCount++;
    break;
  case T_SCRAPER:
    program->scraperCount++;
    break;
  case T_TAPERAPC:
  case T_TAPERAPR:
    program->taperApertureCount++;
    break;
  case T_SPEEDBUMP:
    program->speedbumpCount++;
    break;
  case T_MALIGN:
    program->malignCount++;
    break;
  case T_MAXAMP:
  case T_WATCH:
  case T_HISTOGRAM:
  case T_CHARGE:
    break;
  default:
    if (!gpuTuneProgramMatrixTracked(eptr))
      return 0;
    program->matrixCount++;
    break;
  }
  program->opCount++;
  return 1;
}

static long gpuTuneProgramAllocate(GPU_TUNE_PROGRAM *program) {
#define GPU_TUNE_ALLOC(member, count)                                      \
  do {                                                                     \
    if ((count) > 0) {                                                     \
      program->member = calloc((size_t)(count), sizeof(*program->member)); \
      if (!program->member)                                                \
        goto failed;                                                       \
    }                                                                      \
  } while (0)

  if (!program || program->opCount <= 0)
    return 0;
  GPU_TUNE_ALLOC(op, program->opCount);
  GPU_TUNE_ALLOC(drift, program->driftCount);
  GPU_TUNE_ALLOC(matrix, program->matrixCount);
  GPU_TUNE_ALLOC(multipole, program->multipoleCount);
  GPU_TUNE_ALLOC(exactCorrector, program->exactCorrectorCount);
  GPU_TUNE_ALLOC(csbend, program->csbendCount);
  GPU_TUNE_ALLOC(ccbend, program->ccbendCount);
  GPU_TUNE_ALLOC(lgbend, program->lgbendCount);
  GPU_TUNE_ALLOC(lgbendLocalAperture,
                 program->lgbendLocalApertureCount);
  GPU_TUNE_ALLOC(collimator, program->collimatorCount);
  GPU_TUNE_ALLOC(scraper, program->scraperCount);
  GPU_TUNE_ALLOC(taperAperture, program->taperApertureCount);
  GPU_TUNE_ALLOC(speedbump, program->speedbumpCount);
  GPU_TUNE_ALLOC(malign, program->malignCount);
  program->postApertureCount = program->opCount;
  GPU_TUNE_ALLOC(postAperture, program->postApertureCount);
#undef GPU_TUNE_ALLOC
  return 1;

failed:
#undef GPU_TUNE_ALLOC
  gpuTuneProgramRelease(program);
  return 0;
}

static long gpuTuneProgramPackCollimator(GPU_TUNE_COLLIMATOR_DATA *data,
                                         ELEMENT_LIST *eptr) {
  if (!data || !eptr || !eptr->p_elem)
    return 0;
  memset(data, 0, sizeof(*data));
  if (eptr->type == T_RCOL) {
    RCOL *rcol = (RCOL *)eptr->p_elem;
    if (rcol->invert ||
        ((rcol->openSide && *rcol->openSide) && !determineOpenSideCode))
      return 0;
    data->length = rcol->length;
    data->xMax = rcol->x_max;
    data->yMax = rcol->y_max;
    data->xCenter = rcol->dx;
    data->yCenter = rcol->dy;
    data->openCode =
      determineOpenSideCode ? determineOpenSideCode(rcol->openSide) : 0;
  } else if (eptr->type == T_ECOL) {
    ECOL *ecol = (ECOL *)eptr->p_elem;
    long yExponent =
      ecol->yExponent ? ecol->yExponent : ecol->exponent;
    if (ecol->invert || ecol->x_max <= 0 || ecol->y_max <= 0 ||
        ecol->exponent < 2 || (ecol->exponent & 1) ||
        yExponent < 2 || (yExponent & 1) ||
        ((ecol->openSide && *ecol->openSide) && !determineOpenSideCode))
      return 0;
    data->length = ecol->length;
    data->xMax = ecol->x_max;
    data->yMax = ecol->y_max;
    data->xCenter = ecol->dx;
    data->yCenter = ecol->dy;
    data->xExponent = ecol->exponent;
    data->yExponent = yExponent;
    data->openCode =
      determineOpenSideCode ? determineOpenSideCode(ecol->openSide) : 0;
  } else {
    return 0;
  }
  return data->length >= 0 && isfinite(data->length) &&
         isfinite(data->xMax) && isfinite(data->yMax) &&
         isfinite(data->xCenter) && isfinite(data->yCenter);
}

static long gpuTuneProgramPackScraper(GPU_TUNE_SCRAPER_DATA *data,
                                      ELEMENT_LIST *eptr) {
  SCRAPER *scraper;
  unsigned long direction;

  if (!data || !eptr || eptr->type != T_SCRAPER || !eptr->p_elem)
    return 0;
  scraper = (SCRAPER *)eptr->p_elem;
  if (scraper->length < 0 ||
      (scraper->length && (scraper->Xo || scraper->Z)))
    return 0;
  direction =
    interpretScraperDirection(scraper->insert_from, scraper->oldDirection);
  memset(data, 0, sizeof(*data));
  data->length = scraper->length;
  data->position = scraper->position;
  if (direction & DIRECTION_X) {
    data->plane = 0;
    data->center = scraper->dx;
    if (direction & DIRECTION_PLUS_X)
      data->sideSign = 1;
    if (direction & DIRECTION_MINUS_X) {
      if (data->sideSign)
        data->secondSideSign = -1;
      else
        data->sideSign = -1;
    }
  } else if (direction & DIRECTION_Y) {
    data->plane = 2;
    data->center = scraper->dy;
    if (direction & DIRECTION_PLUS_Y)
      data->sideSign = 1;
    if (direction & DIRECTION_MINUS_Y) {
      if (data->sideSign)
        data->secondSideSign = -1;
      else
        data->sideSign = -1;
    }
  } else {
    return scraper->length == 0;
  }
  if (data->secondSideSign && data->position <= 0)
    return 0;
  return isfinite(data->length) && isfinite(data->center) &&
         isfinite(data->position);
}

static void gpuTuneProgramSetStickyFromTaper(MAXAMP *sticky,
                                             ELEMENT_LIST *eptr) {
  memset(sticky, 0, sizeof(*sticky));
  if (eptr->type == T_TAPERAPC) {
    TAPERAPC *taper = (TAPERAPC *)eptr->p_elem;
    sticky->x_max = sticky->y_max = taper->r[taper->e2Index];
    sticky->elliptical = 1;
    sticky->exponent = sticky->yExponent = 2;
  } else {
    TAPERAPR *taper = (TAPERAPR *)eptr->p_elem;
    sticky->x_max = taper->xmax[taper->e2Index];
    sticky->y_max = taper->ymax[taper->e2Index];
  }
}

static long gpuTuneProgramBuild(GPU_TUNE_PROGRAM *program,
                                LINE_LIST *beamline, double pCentral) {
  ELEMENT_LIST *eptr;
  MAXAMP stickyBuffer, *sticky = NULL;
  long opIndex = 0, driftIndex = 0, matrixIndex = 0;
  long multipoleIndex = 0, exactCorrectorIndex = 0, csbendIndex = 0;
  long ccbendIndex = 0, lgbendIndex = 0, collimatorIndex = 0;
  long scraperIndex = 0, taperIndex = 0, speedbumpIndex = 0, malignIndex = 0;
  long elementIndex = 0;

  if (!program || !beamline)
    return 0;
  memset(program, 0, sizeof(*program));
  for (eptr = beamline->elem; eptr; eptr = eptr->succ)
    if (!gpuTuneProgramCountElement(program, eptr))
      goto unsupported;
  if (!gpuTuneProgramAllocate(program))
    return 0;

  for (eptr = beamline->elem; eptr; eptr = eptr->succ, elementIndex++) {
    GPU_TUNE_PROGRAM_OP *op;
    unsigned long classFlags;

    if (eptr->ignore)
      continue;
    switch (eptr->type) {
    case T_MARK:
    case T_RECIRC:
      continue;
    case T_HMON:
    case T_VMON:
    case T_MONI:
      if (!gpuTuneProgramMonitorLength(eptr))
        continue;
      break;
    default:
      break;
    }
    if (opIndex >= program->opCount || gpuElementNeedsPlainTracking(eptr))
      goto unsupported;
    if (!gpuTuneProgramExactElement(eptr))
      program->inexactCount++;
    op = program->op + opIndex;
    op->dataIndex = -1;
    op->auxiliaryIndex = -1;
    op->postApertureIndex = -1;
    op->elementIndex = elementIndex;

    switch (eptr->type) {
    case T_EDRIFT:
      op->opcode = GPU_TUNE_OP_EXACT_DRIFT;
      op->dataIndex = driftIndex;
      program->drift[driftIndex++].length =
        ((EDRIFT *)eptr->p_elem)->length;
      break;
    case T_RFCA: {
      RFCA *rfca = (RFCA *)eptr->p_elem;
      if (!rfca || rfca->volt != 0 || rfca->change_p0 ||
          rfca->change_t || rfca->Q || rfca->backtrack)
        goto unsupported;
      op->opcode = rfca->length ? GPU_TUNE_OP_EXACT_DRIFT :
                                 GPU_TUNE_OP_NOP;
      op->dataIndex = driftIndex;
      program->drift[driftIndex++].length = rfca->length;
      break;
    }
    case T_DRIF:
    case T_HCOR:
    case T_VCOR:
    case T_HMON:
    case T_VMON:
    case T_MONI:
      if (!eptr->matrix ||
          !gpuPackMatrix(program->matrix + matrixIndex, eptr->matrix))
        goto unsupported;
      op->opcode = GPU_TUNE_OP_MATRIX;
      op->dataIndex = matrixIndex++;
      break;
    case T_MULT:
    case T_KQUAD:
    case T_KSEXT:
    case T_KOCT:
    case T_DQCOR:
      if (!gpuPackMultipoleTracking(
            program->multipole + multipoleIndex, eptr, pCentral) ||
          !gpuPackMaxamp(
            &program->multipole[multipoleIndex].aperture, sticky,
            program->multipole[multipoleIndex].cosTilt,
            program->multipole[multipoleIndex].sinTilt))
        goto unsupported;
      /* the program's multipole operation has no radiation: SYNCH_RAD stays with do_tracking() */
      if (program->multipole[multipoleIndex].radCoef)
        goto unsupported;
      op->opcode = GPU_TUNE_OP_MULTIPOLE;
      op->dataIndex = multipoleIndex++;
      break;
    case T_EHCOR:
    case T_EVCOR:
    case T_EHVCOR:
      if (!gpuPackExactCorrector(
            program->exactCorrector + exactCorrectorIndex, eptr, pCentral,
            eptr->beg_pos))
        goto unsupported;
      op->opcode = GPU_TUNE_OP_EXACT_CORRECTOR;
      op->dataIndex = exactCorrectorIndex++;
      break;
    case T_CSBEND:
      if (!gpuPackCsbendTracking(
            program->csbend + csbendIndex, (CSBEND *)eptr->p_elem,
            pCentral) ||
          !gpuPackMaxamp(&program->csbend[csbendIndex].aperture, sticky,
                         program->csbend[csbendIndex].cosTilt,
                         program->csbend[csbendIndex].sinTilt))
        goto unsupported;
      program->csbend[csbendIndex].lossZStart = eptr->beg_pos;
      op->opcode = GPU_TUNE_OP_CSBEND;
      op->dataIndex = csbendIndex++;
      break;
    case T_CCBEND:
      if (!gpuPackCcbendTracking(
            program->ccbend + ccbendIndex, eptr, (CCBEND *)eptr->p_elem,
            pCentral) ||
          !gpuPackMaxamp(&program->ccbend[ccbendIndex].aperture, sticky,
                         program->ccbend[ccbendIndex].cosTilt,
                         program->ccbend[ccbendIndex].sinTilt))
        goto unsupported;
      op->opcode = GPU_TUNE_OP_CCBEND;
      op->dataIndex = ccbendIndex++;
      break;
    case T_LGBEND:
      if (!gpuPackLgbendTracking(
            program->lgbend + lgbendIndex,
            program->lgbendLocalAperture + lgbendIndex,
            eptr, (LGBEND *)eptr->p_elem, pCentral) ||
          !gpuPackMaxamp(&program->lgbend[lgbendIndex].aperture, sticky,
                         program->lgbend[lgbendIndex].cosTilt,
                         program->lgbend[lgbendIndex].sinTilt))
        goto unsupported;
      op->opcode = GPU_TUNE_OP_LGBEND;
      op->dataIndex = lgbendIndex;
      op->auxiliaryIndex = lgbendIndex++;
      break;
    case T_RCOL:
    case T_ECOL:
      if (!gpuTuneProgramPackCollimator(
            program->collimator + collimatorIndex, eptr))
        goto unsupported;
      op->opcode = eptr->type == T_RCOL ? GPU_TUNE_OP_RCOL :
                                         GPU_TUNE_OP_ECOL;
      op->dataIndex = collimatorIndex++;
      break;
    case T_SCRAPER:
      if (!gpuTuneProgramPackScraper(
            program->scraper + scraperIndex, eptr))
        goto unsupported;
      op->opcode = GPU_TUNE_OP_SCRAPER;
      op->dataIndex = scraperIndex++;
      break;
    case T_TAPERAPC:
    case T_TAPERAPR:
      if (!gpuPackTaperAperture(
            program->taperAperture + taperIndex, eptr, pCentral,
            eptr->beg_pos))
        goto unsupported;
      op->opcode = GPU_TUNE_OP_TAPER_APERTURE;
      op->dataIndex = taperIndex++;
      if ((eptr->type == T_TAPERAPC &&
           ((TAPERAPC *)eptr->p_elem)->sticky) ||
          (eptr->type == T_TAPERAPR &&
           ((TAPERAPR *)eptr->p_elem)->sticky)) {
        gpuTuneProgramSetStickyFromTaper(&stickyBuffer, eptr);
        sticky = &stickyBuffer;
      }
      break;
    case T_SPEEDBUMP:
      if (!gpuPackSpeedbump(
            program->speedbump + speedbumpIndex, eptr, pCentral,
            eptr->beg_pos))
        goto unsupported;
      op->opcode = GPU_TUNE_OP_SPEEDBUMP;
      op->dataIndex = speedbumpIndex++;
      break;
    case T_MAXAMP:
      sticky = (MAXAMP *)eptr->p_elem;
      op->opcode = GPU_TUNE_OP_NOP;
      break;
    case T_WATCH:
    case T_HISTOGRAM:
    case T_CHARGE:
      /* tracking with TEST_PARTICLES writes no file here and changes no coordinate */
      op->opcode = GPU_TUNE_OP_NOP;
      break;
    case T_MALIGN: {
      MALIGN *malign = (MALIGN *)eptr->p_elem;
      GPU_TUNE_MALIGN_DATA *data = program->malign + malignIndex;

      if (!malign || malign->startPID != -1 || malign->endPID != -1)
        goto unsupported;
      data->dx = malign->dx;
      data->dxp = malign->dxp;
      data->dy = malign->dy;
      data->dyp = malign->dyp;
      data->dz = malign->dz;
      data->dt = malign->dt;
      data->dp = malign->dp;
      data->de = malign->de;
      data->pCentral = pCentral;
      data->cMks = c_mks;
      data->onPass = malign->on_pass;
      /* do_tracking() leaves it out of TEST_PARTICLES tracking, which this is */
      op->opcode = malign->excludeOrbit ? GPU_TUNE_OP_NOP : GPU_TUNE_OP_MALIGN;
      op->dataIndex = malignIndex++;
      break;
    }
    default:
      if (!gpuTuneProgramMatrixTracked(eptr) || !eptr->matrix ||
          !gpuPackMatrix(program->matrix + matrixIndex, eptr->matrix))
        goto unsupported;
      op->opcode = GPU_TUNE_OP_MATRIX;
      op->dataIndex = matrixIndex++;
      break;
    }

    classFlags = entity_description[eptr->type].flags;
    if (!(classFlags & NO_APERTURE) && sticky) {
      op->postApertureIndex = opIndex;
      if (!gpuPackMaxamp(
            program->postAperture + opIndex, sticky, 1, 0))
        goto unsupported;
    }
    opIndex++;
  }
  if (opIndex != program->opCount)
    goto unsupported;
  /*
   * Loss parity is certified for this deliberately narrow deterministic
   * subset.  Native histories remain disabled until the ULP diagnostic
   * certifies every opcode; hybrid output therefore remains CPU-authored.
   */
  program->lossCertified = 1;
  /* The drift, matrix, multipole, corrector and aperture operations repeat the CPU's arithmetic in
   * its order, but for the configurations gpuTuneProgramExactElement() lists, and
   * gpu_batched_tune_program_checked() stops a verify build where they don't.  So do CCBEND and
   * CSBEND.  LGBEND doesn't: its kernel was rearranged for speed. */
  program->historyCertified = !program->lgbendCount && !program->inexactCount;
  return 1;

unsupported:
  if (gpuVerbose && eptr)
    fprintf(stderr,
            "elegant CUDA: fused tune program rejected %s %s#%ld.\n",
            gpuElementTypeName(eptr), eptr->name ? eptr->name : "?",
            eptr->occurence);
  gpuTuneProgramRelease(program);
  return 0;
}
#endif

#if !USE_MPI
/* Whether an element takes a time or phase reference that elegant sets again at the start of every
 * do_tracking() call (reset_special_elements() and delete_phase_references(), which run for each
 * call unless run_control's reset_rf_for_each_step is 0; a KICKER's is reset regardless).
 * computeTunesFromTracking() calls do_tracking() once a turn with one particle, so such an element
 * takes its reference from that particle on that turn: an RFCA then gives every particle its nominal
 * phase on every turn.  A batch tracks all turns in one call and would set the reference once, from
 * all the particles, so a beamline with such an element can't be batched, except for the plain RFCA
 * described below. */
static long gpuElementReferenceResetEachCall(ELEMENT_LIST *eptr, ELEMENT_LIST *first,
                                             unsigned long fiducialFlag) {
  if (!eptr || !eptr->p_elem)
    return 0;
  if (eptr->type == T_KICKER)
    return 1;
  if (!(fiducialFlag & RESET_RF_FOR_EACH_STEP))
    return 0;
  switch (eptr->type) {
  case T_RFCA: {
    RFCA *rfca = (RFCA *)eptr->p_elem;
    ELEMENT_LIST *other;
    /* with no voltage and no time change the reference is set but never used */
    if (rfca->volt == 0 && !rfca->change_t && !rfca->Q)
      return 0;
    /* A plain cavity is tracked in the batch one particle at a time with its reference set again for
     * each (simple_rf_cavity(), under gpuTuneOwnRfReference()).  Not with CHANGE_T, which moves the
     * tracking clock of the whole call, nor with CHANGE_P0, which rematches the central momentum to
     * the one particle, nor when another cavity uses the same PHASE_REFERENCE number: elegant then
     * lets the first of them set the reference the others use on that turn. */
    if (rfca->change_t || rfca->change_p0 || 
        (gpuEnvSet("ELEGANT_GPU_ENABLE_TUNE_RFCA_REFERENCE") &&
         !gpuEnvFlag("ELEGANT_GPU_ENABLE_TUNE_RFCA_REFERENCE")))
      return 1;
    if (rfca->phase_reference)
      for (other = first; other; other = other->succ)
        if (other != eptr && other->type == T_RFCA && other->p_elem &&
            ((RFCA *)other->p_elem)->phase_reference == rfca->phase_reference)
          return 1;
    return 0;
  }
  case T_RFCW:
  case T_RFDF:
  case T_RFTM110:
  case T_MODRF:
  case T_RAMPRF:
  case T_TMCF:
  case T_RFTMEZ0:
  case T_CEPL:
  case T_TWPL:
  case T_TWLA:
  case T_TWMTA:
    return 1;
  case T_RFMODE:
    return ((RFMODE *)eptr->p_elem)->reset_for_each_step ? 1 : 0;
  case T_FRFMODE:
    return ((FRFMODE *)eptr->p_elem)->reset_for_each_step ? 1 : 0;
  case T_TRFMODE:
    return ((TRFMODE *)eptr->p_elem)->reset_for_each_step ? 1 : 0;
  case T_FTRFMODE:
    return ((FTRFMODE *)eptr->p_elem)->reset_for_each_step ? 1 : 0;
  default:
    return 0;
  }
}
#endif

long gpu_batched_frequency_map_beamline_supported(void *beamline0) {
#if USE_MPI
  (void)beamline0;
  return 0;
#else
  LINE_LIST *beamline = (LINE_LIST *)beamline0;
  ELEMENT_LIST *eptr;

  if (!gpuEnableBatchedTuneTracking || !beamline)
    return 0;
  for (eptr = beamline->elem; eptr; eptr = eptr->succ) {
    unsigned long flags;

    if (eptr->ignore)
      continue;
    /* Until CPU fix 21 is committed, a POLYNOMIALSERIES particle with too much transverse momentum stops
     * the pass for every particle after it in a batch (see gpu_batched_tune_beamline_supported()) */
    if (eptr->type == T_POLYNOMIALSERIES)
      return 0;
    flags = entity_description[eptr->type].flags;
    if (flags & (HAS_MPI_COLLECTIVE | UNIPROCESSOR)) {
      if (gpuVerbose)
        fprintf(stderr,
                "elegant CUDA: batched tune tracking disabled by "
                "%s %s#%ld flags.\n",
                gpuElementTypeName(eptr), eptr->name ? eptr->name : "?",
                eptr->occurence);
      return 0;
    }
    if (gpuElementReferenceResetEachCall(eptr, beamline->elem, beamline->fiducial_flag)) {
      if (gpuVerbose)
        fprintf(stderr,
                "elegant CUDA: batched tune tracking disabled by %s %s#%ld: "
                "elegant sets its reference again for each particle on each turn.\n",
                gpuElementTypeName(eptr), eptr->name ? eptr->name : "?",
                eptr->occurence);
      return 0;
    }

    /* Batched tune tracking records each particle after every turn.  Admit
     * the deterministic CUDA subset plus narrow deterministic CPU islands
     * handled below.  The normal CUDA option guards exclude ISR, random
     * multipoles, unsupported radiation modes, and other CPU-only physics. */
    if (gpuElementEligible(eptr, LONG_MAX / 4) ||
        gpuPassiveElementSupported(eptr, LONG_MAX / 4))
      continue;
    if (eptr->type == T_LGBEND && eptr->p_elem) {
      LGBEND *lgbend = (LGBEND *)eptr->p_elem;

      /*
       * LGBEND initialization can be deferred until first tracking.  Admit
       * the narrow deterministic aperture-data case here; the subsequent
       * CPU-tracking predicate rechecks the complete CUDA option guard after
       * loading the table.
       */
      if (gpuStringSet(lgbend->apertureDataFile) &&
          lgbend->nSlices > 0 &&
          lgbend->nSlices <= GPU_LGBEND_MAX_LOCAL_APERTURE_SLICES &&
          !lgbend->isr && !lgbend->distributionBasedRadiation &&
          !gpuStringSet(lgbend->centroidOutputFile) &&
          !lgbend->centroidsRequested && !lgbend->SDDScen)
        continue;
    }
    if (gpuVerbose)
      fprintf(stderr,
              "elegant CUDA: batched tune tracking disabled by unsupported "
              "%s %s#%ld.\n",
              gpuElementTypeName(eptr), eptr->name ? eptr->name : "?",
              eptr->occurence);
    return 0;
  }
  if (gpuVerbose)
    fprintf(stderr,
            "elegant CUDA: full beamline is eligible for batched tune "
            "tracking.\n");
  return 1;
#endif
}

long gpu_batched_frequency_map_cpu_tracking_required(void *beamline0) {
#if USE_MPI
  (void)beamline0;
  return 1;
#else
  LINE_LIST *beamline = (LINE_LIST *)beamline0;

  if (!beamline)
    return 1;
  /*
   * The fused backend is presently loss-certified but not history-certified.
   * Its loss-only prepass also cost more than it saved on the reduced
   * frequency-map gates.  Keep ordinary element dispatch suppressed so the
   * authoritative CPU arithmetic (and -ompThreads, when requested) supplies
   * the published histories.  GPU_VERIFY still exercises the fused prepass
   * below as a certification diagnostic.
  */
  if (gpuVerbose) {
#ifdef GPU_VERIFY
    fprintf(stderr,
            "elegant CUDA: frequency map selected GPU prepass + OpenMP "
            "certification backend; published histories remain "
            "CPU-authored.\n");
#else
    fprintf(stderr,
            "elegant CUDA: frequency map selected CPU/OpenMP backend; "
            "native GPU histories are not certified and the loss-only "
            "prepass is diagnostic-only.\n");
#endif
  }
  return 1;
#endif
}

long gpu_batched_tune_analysis_threads(void) {
#if USE_MPI
  return 1;
#else
  long threads = 1;

  if (gpuEnvSet("ELEGANT_GPU_BATCHED_TUNE_THREADS"))
    threads = gpuEnvLong("ELEGANT_GPU_BATCHED_TUNE_THREADS", 1);
  else {
#  ifdef _OPENMP
    threads = omp_get_max_threads();
#  endif
    if (gpuGetOmpTrackingThreads() > threads)
      threads = gpuGetOmpTrackingThreads();
  }
  if (threads < 1)
    threads = 1;
  return threads > 16 ? 16 : threads;
#endif
}

#if !USE_MPI
/* What the last fused program did, for the activity summary printed if its histories are used */
static struct {
  long ops, matrix, magnets, apertures, passes;
  float milliseconds;
} gpuTuneProgramLast;
#endif

#if !USE_MPI
/* gpuBaseInit() leaves no active device when a tracking call has the device suppressed, which is how
 * the frequency map's CPU tracking and the checks of the fused program run.  The program needs the
 * device anyway. */
static long gpuTuneProgramSelectDevice(void) {
  const char *mode = gpuMode();
  int deviceCount = 0;
  long device;

  if (mode && strcmp(mode, "off") == 0)
    return 0;
  if (gpuBase.activeDevice < 0) {
    if (gpuCudaRuntimeGetDeviceCount(&deviceCount) != 0 || deviceCount <= 0)
      return 0;
    device = gpuEnvSet("ELEGANT_GPU_DEVICE") ? gpuEnvLong("ELEGANT_GPU_DEVICE", 0) : 0;
    if (device < 0 || device >= deviceCount)
      return 0;
    gpuBase.deviceCount = deviceCount;
    gpuBase.activeDevice = device;
  }
  return gpuSelectDevice();
}
#endif

/* Whether a history may differ from the CPU's.  0, the default, takes the device's histories only
 * when they are the CPU's bit for bit; a positive value is the largest difference allowed, as a
 * fraction of the history's largest value. */
double gpu_batched_tune_history_tolerance(void) {
#if USE_MPI
  return 0;
#else
  const char *text = gpuGetenv("ELEGANT_GPU_TUNE_HISTORY_TOLERANCE");
  double value;

  if (!text || !*text || sscanf(text, "%lf", &value) != 1 || !(value > 0))
    return 0;
  return value;
#endif
}

#if !USE_MPI
/* The run settings under which the fused program may stand in for do_tracking(): the program has no
 * per-pass work but its elements' */
static long gpuTuneProgramRunSupported(LINE_LIST *beamline, RUN *run) {
  if (!beamline || !run || !gpuBatchedTuneEnabledNow() ||
      (gpuEnvSet("ELEGANT_GPU_ENABLE_TUNE_PROGRAM") && !gpuEnvFlag("ELEGANT_GPU_ENABLE_TUNE_PROGRAM")))
    return 0;
  if (run->always_change_p0 || run->backtrack || run->apertureData.initialized ||
      run->modulationData.nItems || run->rampData.nItems || beamline->links ||
      (beamline->elem_recirc && beamline->elem_recirc != beamline->elem) || spinCoordOffset ||
      (obstructionDataPresent && obstructionDataPresent()))
    return 0;
  return 1;
}
#endif

/* Whether a batch through this beamline can use the fused program's histories: the program takes
 * every element, and either its arithmetic is the CPU's or a tolerance is set.  &tune_footprint
 * batches such a beamline; before, it batched only POLYNOMIALSERIES maps. */
long gpu_batched_tune_program_beamline(void *beamline0, void *run0) {
#if USE_MPI
  (void)beamline0;
  (void)run0;
  return 0;
#else
  LINE_LIST *beamline = (LINE_LIST *)beamline0;
  RUN *run = (RUN *)run0;
  GPU_TUNE_PROGRAM program;
  long usable;

  if (!gpuTuneProgramRunSupported(beamline, run))
    return 0;
  memset(&program, 0, sizeof(program));
  if (!gpuTuneProgramBuild(&program, beamline, run->p_central))
    return 0;
  usable = program.lossCertified &&
           (program.historyCertified || gpu_batched_tune_history_tolerance() > 0);
  gpuTuneProgramRelease(&program);
  return usable;
#endif
}

/* Tracks a batch of particles for a tune history through the fused program: one kernel launch a
 * turn for the whole beamline, in place of one for each element.  history gets x, x', y and y' at
 * the start of each turn (4 blocks of particles * turns), ending the final coordinates, and alive 1
 * for a particle that survived every turn.
 *
 * Returns 0, and the caller tracks on the CPU, when the program can't take the beamline or the run,
 * or when it has an operation whose arithmetic isn't the CPU's (the bends: see exact below) and no
 * tolerance is set.  The verify build runs such a program anyway, to measure the difference.
 * The caller checks what comes back against the CPU before it uses it; *exact says whether a
 * difference would be a defect. */
long gpu_batched_tune_program_track(
  void *beamline0, void *run0, double pCentral, const double *coordinate,
  long particles, long stride, long turns, long turnOffset, double *history,
  double *ending, long *alive, long *exact) {
#if USE_MPI
  (void)beamline0;
  (void)run0;
  (void)pCentral;
  (void)coordinate;
  (void)particles;
  (void)stride;
  (void)turns;
  (void)turnOffset;
  (void)history;
  (void)ending;
  (void)alive;
  (void)exact;
  return 0;
#else
  LINE_LIST *beamline = (LINE_LIST *)beamline0;
  RUN *run = (RUN *)run0;
  GPU_TUNE_PROGRAM program;
  GPU_TUNE_PARTICLE_STATUS *status = NULL;
  float milliseconds = 0;
  long launches = 0, transfers = 0, count = 0, ip;
  int cudaStatus;

  if (exact)
    *exact = 0;
  if (!coordinate || particles <= 0 || stride < 6 || turns <= 1 || !history || !ending || !alive ||
      particles < gpuBatchedTuneMinParticlesNow() || !gpuTuneProgramRunSupported(beamline, run))
    return 0;
  memset(&program, 0, sizeof(program));
  if (!gpuTuneProgramBuild(&program, beamline, pCentral) || !program.lossCertified)
    return 0;
  if (exact)
    *exact = program.historyCertified;
#ifndef GPU_VERIFY
  if (!program.historyCertified && gpu_batched_tune_history_tolerance() <= 0) {
    if (gpuVerbose)
      fprintf(stderr,
              "elegant CUDA: tune histories stay on the CPU: the beamline has %ld LGBEND and %ld other "
              "elements whose device arithmetic isn't claimed to be the CPU's to the last digit.\n",
              program.lgbendCount, program.inexactCount);
    gpuTuneProgramRelease(&program);
    return 0;
  }
#endif
  if (!gpuTuneProgramSelectDevice()) {
    gpuTuneProgramRelease(&program);
    return 0;
  }
  status = (GPU_TUNE_PARTICLE_STATUS *)calloc((size_t)particles, sizeof(*status));
  if (!status) {
    gpuTuneProgramRelease(&program);
    bombElegant("memory allocation failure for the fused tune program", NULL);
  }

  startGpuTimer();
  cudaStatus = gpuCudaTuneProgramTrack(
    &program, coordinate, particles, (int)stride, turns, turnOffset,
    ending, status, history, &milliseconds, &launches, &transfers);
  if (cudaStatus != 0) {
    if (gpuBase.requiredMode)
      gpuFatalStatus("fused tune program", cudaStatus);
    if (gpuVerbose)
      fprintf(stderr, "elegant CUDA: fused tune program failed (%s); tracking on the CPU.\n",
              gpuCudaRuntimeGetErrorString(cudaStatus));
    free(status);
    gpuTuneProgramRelease(&program);
    return 0;
  }
  for (ip = 0; ip < particles; ip++) {
    alive[ip] = status[ip].alive ? 1 : 0;
    count += alive[ip];
  }
  gpuTuneProgramLast.ops = program.opCount;
  gpuTuneProgramLast.matrix = program.matrixCount;
  gpuTuneProgramLast.magnets = program.multipoleCount + program.exactCorrectorCount + program.csbendCount +
                               program.ccbendCount + program.lgbendCount;
  gpuTuneProgramLast.apertures = program.collimatorCount + program.scraperCount +
                                 program.taperApertureCount + program.speedbumpCount;
  gpuTuneProgramLast.passes = turns - 1;
  gpuTuneProgramLast.milliseconds = milliseconds;
  if (gpuVerbose)
    fprintf(stderr,
            "elegant CUDA: fused tune program tracked %ld particles for %ld turns: ops=%ld (drift=%ld "
            "matrix=%ld multipole=%ld exactCorrector=%ld CSBEND=%ld CCBEND=%ld LGBEND=%ld collimator=%ld "
            "scraper=%ld taper=%ld speedbump=%ld), launches=%ld, survivors=%ld, GPU=%.6gs.\n",
            particles, turns, program.opCount, program.driftCount, program.matrixCount,
            program.multipoleCount, program.exactCorrectorCount, program.csbendCount,
            program.ccbendCount, program.lgbendCount, program.collimatorCount, program.scraperCount,
            program.taperApertureCount, program.speedbumpCount, launches, count,
            milliseconds / 1000.0);
  free(status);
  gpuTuneProgramRelease(&program);
  return 1;
#endif
}

/* The caller's comparison of the program's results with the CPU's for the particles it checked.
 * lossMismatches: particles one side lost and the other kept.  Returns 1 if the program's results
 * may be used.  A verify build stops when a program whose arithmetic should be the CPU's differs
 * from it, or when the program kept a particle the CPU lost. */
long gpu_batched_tune_program_checked(long particles, long checked, long exact, long lossMismatches,
                                      long keptButLost, long differing, double maxDifference) {
#if USE_MPI
  (void)particles;
  (void)checked;
  (void)exact;
  (void)lossMismatches;
  (void)keptButLost;
  (void)differing;
  (void)maxDifference;
  return 0;
#else
  double tolerance = gpu_batched_tune_history_tolerance();
  long usable = !lossMismatches && (!differing || (tolerance > 0 && maxDifference <= tolerance));

  if (gpuVerbose || (!usable && exact))
    fprintf(stderr,
            "elegant CUDA: fused tune program checked against the CPU on %ld of %ld particles: %ld lost "
            "on one side only, %ld with a differing history or final coordinate, largest difference %.3e "
            "of the history's range; %s.\n",
            checked, particles, lossMismatches, differing, maxDifference,
#ifdef GPU_VERIFY
            "the CPU's are used, as in every verify build");
#else
            usable ? "its histories are used" : "the CPU tracks the rest");
#endif
#ifdef GPU_VERIFY
  if (gpuBase.verifyMode && ((exact && (differing || lossMismatches)) || keptButLost)) {
    fprintf(stderr,
            "elegant CUDA VERIFY failed for the fused tune program: %ld particles lost on one side only "
            "(%ld kept by the program and lost on the CPU), %ld differing, largest difference %.3e\n",
            lossMismatches, keptButLost, differing, maxDifference);
    exit(1);
  }
  /* the verify build publishes the CPU's histories, as it always has */
  usable = 0;
#endif
  if (usable) {
    /* the check's tracking call has cleared the counts since the program ran */
    gpuBase.gpuElementCount += gpuTuneProgramLast.ops * gpuTuneProgramLast.passes;
    gpuBase.gpuTrackParticleCount += gpuTuneProgramLast.matrix * gpuTuneProgramLast.passes;
    gpuBase.gpuMagnetCount += gpuTuneProgramLast.magnets * gpuTuneProgramLast.passes;
    gpuBase.gpuApertureCount += gpuTuneProgramLast.apertures * gpuTuneProgramLast.passes;
    gpuRecordHelperKernel(gpuTuneProgramLast.milliseconds);
    displayTimings();
    gpuBase.gpuElementCount = gpuBase.gpuTrackParticleCount = gpuBase.gpuMagnetCount = 0;
    gpuBase.gpuApertureCount = gpuBase.gpuHelperCount = 0;
    gpuBase.gpuKernelSeconds = 0;
  }
  return usable;
#endif
}

long gpu_batched_tune_cpu_scope(long cpuOnly) {
#if USE_MPI
  (void)cpuOnly;
  return 0;
#else
  long before = gpuTrackingSuppressed;

  gpuSetTrackingSuppressed(cpuOnly);
  return before;
#endif
}

static long gpuTuneOwnRfReferenceScope = 0;

long gpuTuneOwnRfReference(void) {
  return gpuTuneOwnRfReferenceScope;
}

void gpu_batched_tune_own_rf_reference(void *beamline0) {
  LINE_LIST *beamline = (LINE_LIST *)beamline0;

  gpuTuneOwnRfReferenceScope =
    beamline && (beamline->fiducial_flag & RESET_RF_FOR_EACH_STEP) ? 1 : 0;
}

void gpu_batched_tune_tracking_set_cpu_only(long cpuOnly) {
#if USE_MPI
  (void)cpuOnly;
#else
  gpuSetTrackingSuppressed(cpuOnly);
  gpuOmpTrackingScope = cpuOnly ? 1 : 0;
  if (cpuOnly && gpuVerbose && gpuOmpTrackingThreads > 1)
    fprintf(stderr,
            "elegant CUDA: loss-sensitive batched CPU tracking will use "
            "%ld OpenMP threads for supported magnet particle loops.\n",
            gpuOmpTrackingThreads);
#endif
}

void gpu_batched_tune_tracking_report(const char *operation, long particles,
                                      long turns, long intervals) {
#if USE_MPI
  (void)operation;
  (void)particles;
  (void)turns;
  (void)intervals;
#else
  if (gpuVerbose)
    fprintf(stderr,
            "elegant CUDA: batched %s tracking %ld particles for %ld "
            "interval%s of %ld turns.\n",
            operation ? operation : "tune", particles, intervals,
            intervals == 1 ? "" : "s", turns);
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
  if (obstructionDataPresent && obstructionDataPresent())
    return 0;
  if (threshold < 1)
    threshold = 1;
  return enabled && particles >= threshold;
#endif
}

static long gpuPackApertureContour(GPU_APERTURE_CONTOUR_DATA *data, APCONTOUR *apContour);
#if !USE_MPI
static void gpuPinApertureContours(long pin);
#endif

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
    /* High-order bends and radiation used to be refused here, until they had
     * search baselines: the *-csbend-high-order and *-radiation search cases
     * in src/gpu/feature-set. */
    flags = entity_description[eptr->type].flags;
    if (flags & (HAS_MPI_COLLECTIVE | UNIPROCESSOR))
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

#if !USE_MPI
/* Elements the device-resident aperture search runs, or 0 if it can't run this beamline */
static long gpuResidentApertureSearchElements(LINE_LIST *beamline) {
  ELEMENT_LIST *eptr;
  long elements = 0;

  /* This is intentionally narrower than ordinary batched-search support.
   * A single CUDA thread follows one independent trial through the complete
   * flattened lattice, so every accepted element must be free of evolving
   * host state and order-sensitive output. */
  for (eptr = beamline->elem; eptr; eptr = eptr->succ) {
    RCOL *rcol;

    if (eptr->ignore)
      continue;
    switch (eptr->type) {
    case T_MARK:
    case T_RECIRC:
      break;
    case T_DRIF:
      if (!eptr->p_elem || !eptr->matrix || !gpuMatrixSupported(eptr->matrix))
        return 0;
      elements++;
      break;
    case T_KQUAD:
    case T_KSEXT:
    case T_KOCT:
      if (!gpuMultipoleElementSupported(eptr))
        return 0;
      elements++;
      break;
    case T_RCOL:
      rcol = (RCOL *)eptr->p_elem;
      if (!rcol || rcol->length != 0 || rcol->invert ||
          (rcol->openSide && *rcol->openSide))
        return 0;
      elements++;
      break;
    case T_APCONTOUR: {
      /* With no length or DZ, a lost particle stays where it is, as the device's contour checks keep it */
      APCONTOUR *apcontour = (APCONTOUR *)eptr->p_elem;
      if (!apcontour || !gpuApContourSupported(apcontour) ||
          (!apcontour->cancel && (apcontour->length != 0 || apcontour->dz)))
        return 0;
      elements++;
      break;
    }
    default:
      return 0;
    }
  }
  return elements;
}
#endif

/* The batched line search tracks every trial on a line, where the scalar search stops at the first
 * loss, so it pays only when the device-resident search runs the whole beamline, when its batches are
 * big enough for elements to go on the device, or when OpenMP threads share them. On SPEAR with 231
 * to 1,681 trials and none of these, it took 13% to 34% longer than the scalar search, with the same
 * result. */
long gpu_batched_aperture_search_worthwhile(void *beamline0, long trials) {
#if USE_MPI
  (void)beamline0;
  (void)trials;
  return 0;
#else
  const char *mode = gpuMode();
  long required = gpuBase.initialized ? gpuBase.requiredMode : strcmp(mode, "required") == 0;
  long threshold = gpuBase.initialized ? gpuMinGeneralEligibleParticles :
    gpuEnvLong("ELEGANT_GPU_MIN_PARTICLES", ELEGANT_GPU_DEFAULT_MIN_PARTICLES);

#  ifdef _OPENMP
  if (omp_get_max_threads() > 1)
    return 1;
#  endif
  if ((gpuBase.initialized && gpuBase.activeDevice < 0) || strcmp(mode, "off") == 0)
    return 0;
  return required || trials >= threshold ||
         gpuResidentApertureSearchElements((LINE_LIST *)beamline0) > 0;
#endif
}

long gpu_run_batched_aperture_search(
  void *beamline0, double pCentral, long lines, long nx, long nSplits,
  double splitFraction, long nPasses, double xmax, double ymax,
  const double *orbit, const double *dxFactor, const double *dyFactor,
  double *xLimit, double *yLimit, double *xLost, double *yLost,
  double *sLost, long *lossPass, long *lossElement, double *lossX0,
  double *lossY0, long *originStable) {
#if USE_MPI
  (void)beamline0;
  (void)pCentral;
  (void)lines;
  (void)nx;
  (void)nSplits;
  (void)splitFraction;
  (void)nPasses;
  (void)xmax;
  (void)ymax;
  (void)orbit;
  (void)dxFactor;
  (void)dyFactor;
  (void)xLimit;
  (void)yLimit;
  (void)xLost;
  (void)yLost;
  (void)sLost;
  (void)lossPass;
  (void)lossElement;
  (void)lossX0;
  (void)lossY0;
  (void)originStable;
  return 0;
#else
  LINE_LIST *beamline = (LINE_LIST *)beamline0;
  ELEMENT_LIST *eptr;
  GPU_BATCHED_APERTURE_ELEMENT *program = NULL;
  APCONTOUR *stickyContour = NULL;
  long elements = 0, packed = 0, beamElementIndex = 0, firstStickyContour = -1;
  long matrixElements = 0, magnetElements = 0, apertureElements = 0;
  long logicalExecutions;
  float milliseconds = 0;
  int status;

  if (!beamline || !gpuBase.initialized || gpuBase.activeDevice < 0 ||
      lines <= 0 || nx < 2 || nSplits < 0 || nPasses <= 0 ||
      splitFraction <= 0 || splitFraction >= 1 || !orbit ||
      !dxFactor || !dyFactor || !xLimit || !yLimit || !xLost || !yLost ||
      !sLost || !lossPass || !lossElement || !lossX0 || !lossY0 || !originStable)
    return 0;

  elements = gpuResidentApertureSearchElements(beamline);
  if (elements <= 0 || !gpuSelectDevice())
    return 0;

  program = (GPU_BATCHED_APERTURE_ELEMENT *)calloc(
    (size_t)elements, sizeof(*program));
  if (!program)
    gpuRequiredFailure("unable to allocate device-resident aperture-search program");
  /* The sticky contour do_tracking() keeps, which magnets check inside and every element after;
   * its contours stay on the device while the program is packed */
  gpuPinApertureContours(1);
  for (eptr = beamline->elem; eptr; eptr = eptr->succ) {
    GPU_BATCHED_APERTURE_ELEMENT *item;

    if (eptr->ignore)
      continue;
    if (eptr->type == T_MARK || eptr->type == T_RECIRC) {
      beamElementIndex++;
      continue;
    }
    item = program + packed;
    item->elementIndex = beamElementIndex++;
    switch (eptr->type) {
    case T_DRIF:
      item->type = GPU_BATCHED_APERTURE_MATRIX_DRIFT;
      item->data.drift.length = ((DRIFT *)eptr->p_elem)->length;
      item->data.drift.order = eptr->matrix->order;
      matrixElements++;
      break;
    case T_KQUAD:
    case T_KSEXT:
    case T_KOCT:
      item->type = GPU_BATCHED_APERTURE_MULTIPOLE;
      /* No MAXAMP, but the contour check undoes the magnet's tilt through the aperture data */
      if (!gpuPackMultipoleTracking(&item->data.multipole, eptr, pCentral) ||
          item->data.multipole.radCoef != 0 ||
          !gpuPackMaxamp(&item->data.multipole.aperture, NULL, item->data.multipole.cosTilt,
                         item->data.multipole.sinTilt) ||
          !gpuPackApertureContour(&item->data.multipole.contour, stickyContour))
        goto unsupported;
      item->data.multipole.lossZStart = eptr->beg_pos;
      if (!stickyContour &&
          !item->data.multipole.hasLinearMisalignment &&
          !item->data.multipole.hasExactMisalignment &&
          !item->data.multipole.nEdgeTerms &&
          !item->data.multipole.nSteeringTerms &&
          !item->data.multipole.nBodyTerms &&
          !item->data.multipole.KnL[0] &&
          !item->data.multipole.KnL[1] &&
          !item->data.multipole.KnL[2] &&
          !item->data.multipole.xkick &&
          !item->data.multipole.ykick &&
          !item->data.multipole.dx && !item->data.multipole.dy &&
          !item->data.multipole.dz &&
          !item->data.multipole.sinTilt &&
          item->data.multipole.cosTilt == 1) {
        GPU_MULTIPOLE_DATA multipole = item->data.multipole;
        memset(&item->data, 0, sizeof(item->data));
        item->type = GPU_BATCHED_APERTURE_MULTIPOLE_DRIFT;
        item->data.drift.length = multipole.drift;
        item->data.drift.coordLimit = multipole.coordLimit;
        item->data.drift.slopeLimit = multipole.slopeLimit;
        item->data.drift.expandHamiltonian = multipole.expandHamiltonian;
      }
      magnetElements++;
      break;
    case T_RCOL: {
      RCOL *rcol = (RCOL *)eptr->p_elem;
      item->type = GPU_BATCHED_APERTURE_RCOL;
      item->data.rcol.xmax = rcol->x_max;
      item->data.rcol.ymax = rcol->y_max;
      item->data.rcol.xCenter = rcol->dx;
      item->data.rcol.yCenter = rcol->dy;
      item->data.rcol.sStart = eptr->beg_pos;
      apertureElements++;
      break;
    }
    case T_APCONTOUR: {
      /* do_tracking()'s rules: CANCEL, or HOLD_OFF without STICKY, ends the sticky contour */
      APCONTOUR *apcontour = (APCONTOUR *)eptr->p_elem;
      item->type = GPU_BATCHED_APERTURE_APCONTOUR;
      if (apcontour->cancel || (apcontour->holdOff && !apcontour->sticky)) {
        stickyContour = NULL;
      } else {
        item->data.apContour.check = !apcontour->holdOff;
        item->data.apContour.sStart = eptr->beg_pos;
        if (!gpuPackApertureContour(&item->data.apContour.contour, apcontour))
          goto unsupported;
        if (apcontour->sticky) {
          if (!stickyContour && firstStickyContour < 0)
            firstStickyContour = packed;
          stickyContour = apcontour;
        } else
          stickyContour = NULL;
      }
      apertureElements++;
      break;
    }
    default:
      goto unsupported;
    }
    if (stickyContour && !(entity_description[eptr->type].flags & NO_APERTURE) &&
        !(eptr->type == T_APCONTOUR && stickyContour->holdOff)) {
      item->postContourActive = 1;
      item->sEnd = eptr->end_pos;
      if (!gpuPackApertureContour(&item->postContour, stickyContour))
        goto unsupported;
    }
    packed++;
  }
  gpuPinApertureContours(0);
  /* do_tracking() keeps a sticky contour into the next pass, where elements before the one that set
   * it would check it too */
  if (stickyContour && firstStickyContour != 0)
    goto unsupported;
  if (packed != elements)
    goto unsupported;

  startGpuTimer();
  status = gpuCudaBatchedApertureSearch(
    program, elements, lines, nx, nSplits, splitFraction, nPasses,
    pCentral, xmax, ymax, orbit, dxFactor, dyFactor,
    xLimit, yLimit, xLost, yLost, sLost, lossPass, lossElement,
    lossX0, lossY0, originStable, &milliseconds);
  if (status != 0)
    gpuFatalStatus("device-resident batched aperture search", status);
  gpuRecordHelperKernel(milliseconds);
  logicalExecutions = nPasses * (nSplits + 1);
  gpuBase.gpuElementCount += elements * logicalExecutions;
  gpuBase.gpuTrackParticleCount += matrixElements * logicalExecutions;
  gpuBase.gpuMagnetCount += magnetElements * logicalExecutions;
  gpuBase.gpuApertureCount += apertureElements * logicalExecutions;
  gpuRecordWallSeconds();
  if (gpuVerbose)
    fprintf(stderr,
            "elegant CUDA: device-resident aperture search packed %ld elements, %ld lines, %ld refinements.\n",
            elements, lines, nSplits);
  displayTimings();
  free(program);
  return 1;

unsupported:
  gpuPinApertureContours(0);
  free(program);
  return 0;
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
  if (!deltaById || !targetById || particles <= 0 || turns < 0 ||
      ((history == NULL) != (historyCount == NULL)) ||
      (turns > 0 && (!history || !historyCount)) ||
      (size_t)particles > SIZE_MAX / (2 * sizeof(*gpuBatchedSearchScratch.hostData)) ||
      (turns > 0 &&
       ((size_t)turns > SIZE_MAX / (5 * sizeof(*history)) ||
        (size_t)particles > SIZE_MAX /
          ((size_t)turns * 5 * sizeof(*history)))))
    return;
  gpuBatchedSearchScratch.hostData =
    (double *)malloc((size_t)particles * 2 *
                     sizeof(*gpuBatchedSearchScratch.hostData));
  if (!gpuBatchedSearchScratch.hostData)
    gpuRequiredFailure("unable to allocate batched search host data");
  for (ip = 0; ip < particles; ip++) {
    gpuBatchedSearchScratch.hostData[2 * ip] = targetById[ip];
    gpuBatchedSearchScratch.hostData[2 * ip + 1] = deltaById[ip];
  }
  if (turns > 0) {
    memset(history, 0,
           (size_t)particles * 5 * (size_t)turns * sizeof(*history));
    memset(historyCount, 0, (size_t)particles * sizeof(*historyCount));
  }
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
      !gpuBase.initialized || gpuBase.activeDevice < 0 || particles <= 0 || !gpuSelectDevice())
    return 0;
  if (!gpuBatchedSearchScratch.uploaded) {
    if ((unsigned long)gpuBatchedSearchScratch.particles > ULONG_MAX / 2UL ||
        (gpuBatchedSearchScratch.turns > 0 &&
         ((unsigned long)gpuBatchedSearchScratch.turns > ULONG_MAX / 5UL ||
          (unsigned long)gpuBatchedSearchScratch.particles >
            ULONG_MAX / (5UL * (unsigned long)gpuBatchedSearchScratch.turns))))
      gpuRequiredFailure("batched momentum-search allocation overflow");
    historyValues = (unsigned long)gpuBatchedSearchScratch.particles * 5UL *
                    (unsigned long)gpuBatchedSearchScratch.turns;
    status = gpuCudaMallocDouble((void **)&gpuBatchedSearchScratch.deviceData,
                                 2UL * (unsigned long)gpuBatchedSearchScratch.particles);
    if (status != 0)
      gpuFatalStatus("cudaMalloc(batched search data)", status);
    if (historyValues) {
      status = gpuCudaMallocDouble((void **)&gpuBatchedSearchScratch.deviceHistory,
                                   historyValues);
      if (status != 0)
        gpuFatalStatus("cudaMalloc(batched search history)", status);
      status = gpuCudaMallocDouble((void **)&gpuBatchedSearchScratch.deviceHistoryCount,
                                   (unsigned long)gpuBatchedSearchScratch.particles);
      if (status != 0)
        gpuFatalStatus("cudaMalloc(batched search turn counts)", status);
    }
    status = gpuCudaCopyHostToDevice(
      gpuBatchedSearchScratch.deviceData,
      gpuBatchedSearchScratch.hostData,
      2UL * (unsigned long)gpuBatchedSearchScratch.particles, &milliseconds);
    if (status != 0)
      gpuFatalStatus("cudaMemcpy(batched search data host to device)", status);
    gpuRecordMilliseconds(&gpuBase.gpuTransferToDeviceSeconds, milliseconds);
    if (historyValues) {
      status = gpuCudaClearBatchedSearchHistory(
        gpuBatchedSearchScratch.deviceHistory, historyValues,
        gpuBatchedSearchScratch.deviceHistoryCount,
        (unsigned long)gpuBatchedSearchScratch.particles);
      if (status != 0)
        gpuFatalStatus("cudaMemset(batched search history)", status);
    }
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

extern int gpuCudaTuneHistoryRecord(const void *coord, long nParticles, int stride, int idColumn,
                                    long particles, long turns, long turn, void *history);
extern int gpuCudaMemsetZeroDouble(void *data, unsigned long count);

/* A batched tune history recorded on the device, one block of particles * turns values for each of
 * x, x', y and y'.  Allocated on the first recorded turn, inside do_tracking, so it uses the
 * device tracking selected.  Verify builds record on the CPU. */
static struct {
  long particles, turns;
  double *device;
  unsigned char *recorded;
} gpuTuneHistory;

void gpu_tune_history_configure(long particles, long turns) {
  gpu_tune_history_clear();
  if (particles <= 0 || turns <= 1)
    return;
  gpuTuneHistory.recorded = (unsigned char *)calloc(turns, sizeof(*gpuTuneHistory.recorded));
  if (!gpuTuneHistory.recorded)
    gpuRequiredFailure("unable to allocate the tune history turn flags");
  gpuTuneHistory.particles = particles;
  gpuTuneHistory.turns = turns;
}

long gpu_tune_history_record(long np, long turn) {
#ifdef GPU_VERIFY
  (void)np;
  (void)turn;
  return 0;
#else
  unsigned long values;
  int status;

  if (!gpuTuneHistory.recorded || turn <= 0 || turn >= gpuTuneHistory.turns || np <= 0 ||
      !gpuBase.initialized || gpuBase.activeDevice < 0 || !gpuBase.deviceCurrent)
    return 0;
  if (!gpuTuneHistory.device) {
    values = 4UL * (unsigned long)gpuTuneHistory.particles * (unsigned long)gpuTuneHistory.turns;
    status = gpuCudaMallocDouble((void **)&gpuTuneHistory.device, values);
    if (status != 0)
      gpuFatalStatus("cudaMalloc(tune history)", status);
    status = gpuCudaMemsetZeroDouble(gpuTuneHistory.device, values);
    if (status != 0)
      gpuFatalStatus("cudaMemset(tune history)", status);
  }
  status = gpuCudaTuneHistoryRecord(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, (int)particleIDIndex,
                                    gpuTuneHistory.particles, gpuTuneHistory.turns, turn,
                                    gpuTuneHistory.device);
  if (status != 0)
    gpuFatalStatus("tune history kernel", status);
  gpuTuneHistory.recorded[turn] = 1;
  return 1;
#endif
}

/* Copies the turns recorded on the device into the host history, laid out the same way */
void gpu_tune_history_fetch(double *history) {
  unsigned long block, values;
  double *host;
  long turn, id, c;
  float milliseconds = 0;
  int status;

  if (!gpuTuneHistory.device || !history)
    return;
  block = (unsigned long)gpuTuneHistory.particles * (unsigned long)gpuTuneHistory.turns;
  values = 4UL * block;
  host = (double *)malloc(values * sizeof(*host));
  if (!host)
    gpuRequiredFailure("unable to allocate the tune history copy");
  status = gpuCudaCopyDeviceToHost(host, gpuTuneHistory.device, values, &milliseconds);
  if (status != 0)
    gpuFatalStatus("tune history copy to host", status);
  gpuRecordMilliseconds(&gpuBase.gpuTransferToHostSeconds, milliseconds);
  for (turn = 1; turn < gpuTuneHistory.turns; turn++) {
    if (!gpuTuneHistory.recorded[turn])
      continue;
    for (c = 0; c < 4; c++)
      for (id = 0; id < gpuTuneHistory.particles; id++)
        history[c * block + id * gpuTuneHistory.turns + turn] =
          host[c * block + id * gpuTuneHistory.turns + turn];
  }
  free(host);
}

void gpu_tune_history_clear(void) {
  if (gpuTuneHistory.device)
    gpuCudaFree(gpuTuneHistory.device);
  free(gpuTuneHistory.recorded);
  memset(&gpuTuneHistory, 0, sizeof(gpuTuneHistory));
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
  for (long i = 0; i < gpuSyncReasonCount; i++)
    fprintf(stderr, "elegant CUDA: copies=%ld for %s\n", gpuSyncReasonCopies[i],
            gpuSyncReasonText[i]);
}

static double **syncParticlesToCpu(const char *reason, long readOnly) {
  long copied;
  long acceptedCopied = 0;
  if (!gpuBase.initialized || gpuBase.activeDevice < 0)
    return gpuBase.coord;
  copied = gpuCopyDeviceToHost(gpuBase.nParticles);
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
  gpuVerifyElementTrackedOnGpu = 1;
#endif
}

void displayTimings(void) {
  if (gpuBase.activeDevice < 0 && !gpuBase.gpuElementCount)
    return;
  if (!gpuVerbose && !gpuBase.gpuElementCount)
    return;
  fprintf(stderr,
          "elegant CUDA: elements=%ld passive=%ld matrix=%ld exactDrift=%ld linearDrift=%ld helpers=%ld reductions=%ld apertures=%ld magnets=%ld wakes=%ld lsc=%ld csr=%ld scmult=%ld histograms=%ld wall=%.6fs kernel=%.6fs h2d=%.6fs d2h=%.6fs\n",
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
          gpuBase.gpuHistogramCount,
          gpuBase.gpuWallSeconds,
          gpuBase.gpuKernelSeconds,
          gpuBase.gpuTransferToDeviceSeconds,
          gpuBase.gpuTransferToHostSeconds);
  if (gpuLscReferenceCalls)
    fprintf(stderr, "elegant CUDA: LSC reference variance calls=%lu; "
                    "CPU reference sum, CUDA particle tracking\n",
            gpuLscReferenceCalls);
  if (gpuMatchEnergyCpuOrderCalls || gpuMatchEnergySkippedCalls)
    fprintf(stderr, "elegant CUDA: momentum matching calls=%lu; "
                    "CPU-order sum, CUDA particle updates; %lu more skipped for an unchanged, matched beam\n",
            gpuMatchEnergyCpuOrderCalls, gpuMatchEnergySkippedCalls);
  if (gpuRfcaReferenceCalls)
    fprintf(stderr, "elegant CUDA: RF reference math calls=%lu particles=%lu; "
                    "CPU reference/sine, CUDA particle updates\n",
            gpuRfcaReferenceCalls, gpuRfcaReferenceParticles);
  if (gpuIlmatrixReferenceCalls)
    fprintf(stderr,
            "elegant CUDA: ILMATRIX reference math calls=%ld; CPU sin/cos, CUDA particle updates\n",
            gpuIlmatrixReferenceCalls);
  if (gpuIsrReferenceFills)
    fprintf(stderr, "elegant CUDA: ISR reference random numbers fills=%lu numbers=%lu; "
                    "CPU gauss_rn_lim, CUDA particle updates\n",
            gpuIsrReferenceFills, gpuIsrReferenceDraws);
  if (gpuEnvFlag("ELEGANT_GPU_PROFILE_TIMING"))
    gpuReportKernelSites();
  if (gpuBase.gpuTrackParticleCount || gpuBase.gpuReductionCount)
    fprintf(stderr,
            "elegant CUDA: kernel detail matrix=%.6fs reduction=%.6fs sparseMatrixTerms=%lu\n",
            gpuBase.gpuMatrixSeconds, gpuBase.gpuReductionSeconds,
            gpuMatrixSparseTermTotal);
  if (gpuMatrixSparseTermTotal) {
    long termCount;
    fprintf(stderr, "elegant CUDA: sparse matrix term histogram");
    for (termCount = 0; termCount <= GPU_MATRIX_MAX_TERMS; termCount++)
      if (gpuMatrixSparseTermHistogram[termCount])
        fprintf(stderr, " %ld:%lu", termCount,
                gpuMatrixSparseTermHistogram[termCount]);
    fputc('\n', stderr);
  }
  if (gpuBase.gpuStandaloneLscCount ||
      gpuBase.gpuRfcwLscKickOnlyCount ||
      gpuBase.gpuRfcwLscFullCount)
    fprintf(stderr,
            "elegant CUDA: lsc detail standalone=%ld rfcwKickOnly=%ld rfcwFull=%ld\n",
            gpuBase.gpuStandaloneLscCount,
            gpuBase.gpuRfcwLscKickOnlyCount,
            gpuBase.gpuRfcwLscFullCount);
  if (gpuBase.gpuScmultCount || gpuBase.gpuScmultMomentCount ||
      gpuBase.gpuScmultMomentCacheHitCount)
    fprintf(stderr,
            "elegant CUDA: scmult detail moments=%ld cacheHits=%ld momentKernel=%.6fs kickKernel=%.6fs\n",
            gpuBase.gpuScmultMomentCount,
            gpuBase.gpuScmultMomentCacheHitCount,
            gpuBase.gpuScmultMomentSeconds,
            gpuBase.gpuScmultKickSeconds);
  if (gpuBase.gpuShortGpuIslandCpuCount || gpuShortGpuIslandByCostCount)
    fprintf(stderr,
            "elegant CUDA: short GPU island CPU skips=%ld maxElements=%ld sentToDeviceByCost=%ld\n",
            gpuBase.gpuShortGpuIslandCpuCount,
            gpuShortGpuIslandMaxElements,
            gpuShortGpuIslandByCostCount);
  gpuDisplaySyncTimings();
}

void compareGpuCpu(long nParticles, const char *label) {
  ELEMENT_LIST *verifyElement = (ELEMENT_LIST *)gpuBase.element;
  const char *verifyElementName =
    verifyElement && verifyElement->name ? verifyElement->name : "(unknown)";
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
      double absDiff, scale, relDiff;
      /* both invalid (elegant removes the particle after the element), or the same infinity */
      if ((isnan(cpu) && isnan(gpu)) || cpu == gpu)
        continue;
      absDiff = fabs(cpu - gpu);
      scale = fmax(fabs(cpu), fabs(gpu));
      relDiff = scale > DBL_MIN ? absDiff / scale : absDiff;
      if (absDiff > maxAbs)
        maxAbs = absDiff;
      if (relDiff > maxRel)
        maxRel = relDiff;
      if (!(absDiff <= absTol || relDiff <= relTol)) {
        if (mismatches < 10)
          fprintf(stderr,
                  "elegant CUDA VERIFY mismatch %s element=%s particle=%ld coord=%ld cpu=%.17e gpu=%.17e abs=%.3e rel=%.3e\n",
                  label ? label : "unknown", verifyElementName,
                  ip, ic, cpu, gpu, absDiff, relDiff);
        mismatches++;
      }
    }
  }
  free(gpuCopy);

  if (mismatches) {
    fprintf(stderr,
            "elegant CUDA VERIFY failed for %s element=%s: %ld mismatches, maxAbs=%.3e, maxRel=%.3e, absTol=%.3e, relTol=%.3e\n",
            label ? label : "unknown", verifyElementName,
            mismatches, maxAbs, maxRel, absTol, relTol);
    exit(1);
  }
  if (gpuVerbose)
    fprintf(stderr, "elegant CUDA VERIFY passed for %s element=%s: maxAbs=%.3e maxRel=%.3e\n",
            label ? label : "unknown", verifyElementName, maxAbs, maxRel);

  gpuBase.deviceCurrent = 0;
  gpuBase.hostCurrent = 1;
#ifdef GPU_VERIFY
  gpuCpuVerificationActive = 0;
#endif
}

#ifdef GPU_VERIFY
static int gpuCompareRowsByParticleID(const void *a, const void *b) {
  double idA = (*(double *const *)a)[6], idB = (*(double *const *)b)[6];

  return idA < idB ? -1 : idA > idB;
}

static void compareGpuCpuSurvivors(long survivors, const char *label);

/* Compares the device's first `survivors` rows with the host rows that have the same particle IDs,
 * for kernels whose survivors come out in a different order from the CPU's.  The host's rows stay
 * where they are. */
static void compareGpuCpuRowsByParticleID(double **hostRows, long survivors, const char *label) {
  ELEMENT_LIST *verifyElement = (ELEMENT_LIST *)gpuBase.element;
  double **hostCoord = gpuBase.coord;
  double **sorted, **paired, *deviceRows;
  float milliseconds = 0;
  long ip;
  int status;

  if (survivors <= 0) {
    compareGpuCpuSurvivors(survivors, label);
    return;
  }
  sorted = (double **)malloc(survivors * sizeof(*sorted));
  paired = (double **)malloc(survivors * sizeof(*paired));
  deviceRows = (double *)malloc(survivors * (size_t)gpuBase.deviceStride * sizeof(*deviceRows));
  if (!sorted || !paired || !deviceRows)
    gpuRequiredFailure("unable to allocate CUDA verification buffers for pairing by particle ID");
  memcpy(sorted, hostRows, survivors * sizeof(*sorted));
  qsort(sorted, survivors, sizeof(*sorted), gpuCompareRowsByParticleID);
  status = gpuCudaCopyDeviceToHost(deviceRows, gpuBase.deviceCoord,
                                   (unsigned long)survivors * (unsigned long)gpuBase.deviceStride,
                                   &milliseconds);
  if (status != 0)
    gpuFatalStatus("cudaMemcpy(device to verify host by particle ID)", status);
  for (ip = 0; ip < survivors; ip++) {
    double *key = deviceRows + ip * gpuBase.deviceStride;
    double **match = (double **)bsearch(&key, sorted, survivors, sizeof(*sorted),
                                        gpuCompareRowsByParticleID);
    if (!match) {
      fprintf(stderr,
              "elegant CUDA VERIFY failed for %s element=%s: GPU survivor with particle ID %.0f isn't a CPU survivor\n",
              label ? label : "unknown",
              verifyElement && verifyElement->name ? verifyElement->name : "(unknown)", key[6]);
      exit(1);
    }
    paired[ip] = *match;
  }
  gpuBase.coord = paired;
  compareGpuCpuSurvivors(survivors, label);
  gpuBase.coord = hostCoord;
  free(deviceRows);
  free(paired);
  free(sorted);
}

/* After a loss only the survivors can be compared. Kernels leave a lost particle's s and p
 * unconverted, and its coordinates needn't match once it has been lost. */
static void compareGpuCpuSurvivors(long survivors, const char *label) {
  if (survivors > 0) {
    compareGpuCpu(survivors, label);
    return;
  }
  gpuBase.deviceCurrent = 0;
  gpuBase.hostCurrent = 1;
  gpuCpuVerificationActive = 0;
}

/* The CPU shadow leaves the element on the CPU, so without this the MAXAMP, aperture-data and
 * APCONTOUR checks after an element would never run on the device in this build.  A normal build
 * makes them there whenever the element did. */
void gpuVerifyResumeElementOnGpu(void) {
  if (gpuBase.verifyMode && gpuVerifyElementTrackedOnGpu && !gpuBase.elementOnGpu)
    gpuBase.elementOnGpu = 1;
}

/* An aperture after its CPU shadow: the survivor counts must agree, then the survivors' rows, paired
 * by particle ID where the device keeps them in another order.  Lost rows aren't compared, because
 * their global loss coordinates exist only on the host. */
void compareGpuCpuApertureResult(long gpuRemaining, long cpuRemaining, long nParticles,
                                 long pairByParticleID, const char *label) {
  ELEMENT_LIST *verifyElement = (ELEMENT_LIST *)gpuBase.element;

  if (gpuRemaining != cpuRemaining) {
    fprintf(stderr, "elegant CUDA VERIFY failed for %s element=%s: CPU survivors=%ld GPU survivors=%ld\n",
            label ? label : "unknown",
            verifyElement && verifyElement->name ? verifyElement->name : "(unknown)",
            cpuRemaining, gpuRemaining);
    exit(1);
  }
  if (gpuRemaining == nParticles)
    compareGpuCpu(nParticles, label);
  else if (pairByParticleID)
    compareGpuCpuRowsByParticleID(gpuBase.coord, gpuRemaining, label);
  else
    compareGpuCpuSurvivors(gpuRemaining, label);
}

/* A value the device computed once and the CPU then reuses, such as a fiducial time or MATR's
 * reference time, which the particle comparison can't check */
void compareGpuCpuValue(double gpu, double cpu, double relTol, const char *label) {
  ELEMENT_LIST *verifyElement = (ELEMENT_LIST *)gpuBase.element;
  double absDiff = fabs(cpu - gpu);
  double scale = fmax(fabs(cpu), fabs(gpu));

  if (!(cpu == gpu || absDiff <= relTol * scale)) {
    fprintf(stderr,
            "elegant CUDA VERIFY failed for %s element=%s: cpu=%.17e gpu=%.17e abs=%.3e rel=%.3e relTol=%.3e\n",
            label ? label : "unknown",
            verifyElement && verifyElement->name ? verifyElement->name : "(unknown)",
            cpu, gpu, absDiff, scale > 0 ? absDiff / scale : absDiff, relTol);
    exit(1);
  }
  if (gpuVerbose)
    fprintf(stderr, "elegant CUDA VERIFY passed for %s element=%s: abs=%.3e\n",
            label ? label : "unknown",
            verifyElement && verifyElement->name ? verifyElement->name : "(unknown)", absDiff);
}
#endif

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
      if (cpu->spinSums && gpu->spinSums) {
        for (i = 0; i < 3; i++) {
          double absDiff, relDiff;
          if (!gpuValuesClose(cpu->spinSums->centroid[i], gpu->spinSums->centroid[i], absTol, relTol, &absDiff, &relDiff) &&
              mismatches++ < 10)
            fprintf(stderr, "elegant CUDA VERIFY reduction mismatch %s spin centroid[%ld] cpu=%.17e gpu=%.17e abs=%.3e rel=%.3e\n",
                    label ? label : "unknown", i, cpu->spinSums->centroid[i], gpu->spinSums->centroid[i], absDiff, relDiff);
          if (absDiff > maxAbs)
            maxAbs = absDiff;
          if (relDiff > maxRel)
            maxRel = relDiff;
          for (j = 0; j < 3; j++) {
            if (!gpuValuesClose(cpu->spinSums->sigma[i][j], gpu->spinSums->sigma[i][j], absTol, relTol, &absDiff, &relDiff) &&
                mismatches++ < 10)
              fprintf(stderr, "elegant CUDA VERIFY reduction mismatch %s spin sigma[%ld][%ld] cpu=%.17e gpu=%.17e abs=%.3e rel=%.3e\n",
                      label ? label : "unknown", i, j, cpu->spinSums->sigma[i][j], gpu->spinSums->sigma[i][j], absDiff, relDiff);
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
  gpuBase.elementOnGpu = gpuCurrentElementEligible() ||
                         gpuPassiveElementSupported((ELEMENT_LIST *)gpuBase.element, gpuBase.nParticles);
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

/* do_match_energy's average: the same Kahan sum, in particle order */
static double gpuMatchEnergyAverageInCpuOrder(long np, double P_central) {
  double sum = 0, error = 0;
  float milliseconds = 0;
  long ip;
  int status;

  gpuMatchEnergyCpuOrderCalls++;
  if (gpuBase.hostCurrent) {
    for (ip = 0; ip < np; ip++)
      sum = KahanPlus(sum, P_central * (1 + gpuBase.coord[ip][5]), &error);
    return sum / np;
  }
  if (gpuMatchEnergyDeltaCapacity < np) {
    free(gpuMatchEnergyDelta);
    gpuMatchEnergyDelta = malloc(sizeof(*gpuMatchEnergyDelta) * np);
    if (!gpuMatchEnergyDelta)
      gpuRequiredFailure("unable to allocate the momentum matching buffer");
    gpuMatchEnergyDeltaCapacity = np;
  }
  status = gpuCudaCoordinateColumnToHost(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, 5,
                                         gpuMatchEnergyDelta, &milliseconds);
  if (status != 0)
    gpuFatalStatus("momentum column copy", status);
  gpuRecordHelperKernel(milliseconds);
  for (ip = 0; ip < np; ip++)
    sum = KahanPlus(sum, P_central * (1 + gpuMatchEnergyDelta[ip]), &error);
  return sum / np;
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

#if USE_MPI
  /* do_match_energy() makes the collective calls, as on the master, which stays on the CPU; a call of our
   * own first would pair the workers' calls with the wrong ones and hang the run */
  if (distributedBeam) {
    if (!do_match_energy)
      gpuRequiredFailure("CPU match-energy routine is unavailable");
    coord = forceParticlesToCpu("match energy CPU reference");
#  ifdef GPU_VERIFY
    gpuBase.elementOnGpu = 0;
#  endif
    do_match_energy(coord, np, P_central, change_beam);
    gpuRecordWallSeconds();
    return;
  }
#endif
  if (np <= 0) {
    gpuRecordWallSeconds();
    return;
  }

  oldP = *P_central;
  gpuCopyHostToDevice(np);
  if (gpuMatchEnergyInCpuOrder()) {
    /* With always_change_p0 this runs after every element, and most elements leave the momenta
     * alone.  The sum is one host thread's work in particle order (0.55 ms for 200,000 particles),
     * so when it last found this very beam matched, with this central momentum, it isn't made
     * again: the same numbers in the same order give the same sum and the same answer.  The check
     * is a comparison of bits on the device. */
    if (!change_beam && gpuMatchEnergyMatchedValid && gpuMatchEnergyMatchedParticles == np &&
        gpuMatchEnergyMatchedP == oldP) {
      int same = 0;

      status = gpuCudaColumnSnapshotCompare(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, 5,
                                            &same, &milliseconds);
      if (status != 0)
        gpuFatalStatus("momentum matching snapshot comparison", status);
      gpuRecordHelperKernel(milliseconds);
      if (same) {
        gpuMatchEnergySkippedCalls++;
        gpuRecordWallSeconds();
        return;
      }
    }
    gpuMatchEnergyMatchedValid = 0;
    averageP = gpuMatchEnergyAverageInCpuOrder(np, oldP);
    if (change_beam || fabs(averageP - oldP) / oldP > 1e-14) {
      status = gpuCudaMatchEnergy(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride,
                                  oldP, averageP, (int)change_beam, &milliseconds);
      if (status != 0)
        gpuFatalStatus("match energy update kernel", status);
      gpuRecordHelperKernel(milliseconds);
      gpuMarkDeviceChanged(np);
      if (!change_beam)
        *P_central = averageP;
    } else if (gpuMatchEnergySnapshotEnabled()) {
      status = gpuCudaColumnSnapshotTake(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, 5,
                                         &milliseconds);
      if (status != 0)
        gpuFatalStatus("momentum matching snapshot", status);
      gpuRecordHelperKernel(milliseconds);
      gpuMatchEnergyMatchedParticles = np;
      gpuMatchEnergyMatchedP = oldP;
      gpuMatchEnergyMatchedValid = 1;
    }
    gpuRecordWallSeconds();
    return;
  }
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

/* RAMPP needs the beam's mean time, which the CPU adds in particle order.  The device makes each
 * particle's time and only those come to the host, which adds them in the same order, so the
 * mean is the CPU's exactly. */
void gpu_particle_times(long np, double pCentral, double *times) {
  static void *deviceTimes = NULL;
  static long deviceTimesSize = 0;
  float milliseconds = 0;
  int status;

  if (np <= 0)
    return;
  if (np > deviceTimesSize) {
    if (deviceTimes)
      gpuCudaFree(deviceTimes);
    if (gpuCudaMallocDouble(&deviceTimes, np) != 0)
      gpuRequiredFailure("unable to allocate the RAMPP particle times");
    deviceTimesSize = np;
  }
  gpuCopyHostToDevice(np);
  status = gpuCudaParticleTimes(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, pCentral, c_mks,
                                deviceTimes, times, &milliseconds);
  if (status != 0)
    gpuFatalStatus("RAMPP particle time kernel", status);
  gpuRecordHelperKernel(milliseconds);
#ifdef GPU_VERIFY
  if (gpuBase.verifyMode && beta_from_delta) {
    double **coord = copyParticlesToCpuReadOnly("RAMPP particle time verification");
    long ip;

    for (ip = 0; ip < np; ip++) {
      double t = coord[ip][4] / (c_mks * beta_from_delta(pCentral, coord[ip][5]));
      if (!(t == times[ip]) && !(isnan(t) && isnan(times[ip]))) {
        fprintf(stderr, "elegant CUDA VERIFY failed for RAMPP particle times: particle %ld cpu=%.17e gpu=%.17e\n",
                ip, t, times[ip]);
        exit(1);
      }
    }
    if (gpuVerbose)
      fprintf(stderr, "elegant CUDA VERIFY passed for RAMPP particle times: %ld identical\n", np);
  }
#endif
  gpuRecordWallSeconds();
}

void gpu_rampp_rescale(long np, double oldP, double newP) {
  float milliseconds = 0;
  int status;

  if (np <= 0)
    return;
  gpuCopyHostToDevice(np);
  status = gpuCudaRamppRescale(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, oldP, newP,
                               &milliseconds);
  if (status != 0)
    gpuFatalStatus("RAMPP momentum kernel", status);
  gpuRecordHelperKernel(milliseconds);
  gpuMarkDeviceChanged(np);
  gpuRecordWallSeconds();
}

/* center_beam() with the beam on the device, to the bit: each coordinate's mean from that column copied to the
 * host and added in particle order, the offsets subtracted on the device, then the time coordinate's mean
 * from each particle's time (computeTimeCoordinates()'s Kahan sum, after the offsets, as on the CPU) and the
 * round trip through time on the device. */
void gpu_center_beam(CENTER *center, long np, long iPass, double p0) {
  static double *column = NULL;
  static long columnCapacity = 0;
  double coordinateOffset[6] = {0, 0, 0, 0, 0, 0};
  unsigned int coordinateMask = 0;
  float milliseconds = 0;
  long ic, ip;
  int status;

  if (!center)
    return;
  if (center->onPass >= 0 && iPass != center->onPass)
    return;
  if (np > 0) {
    gpuCopyHostToDevice(np);
    if (columnCapacity < np) {
      free(column);
      if (!(column = (double *)malloc(np * sizeof(*column))))
        gpuRequiredFailure("unable to allocate the CENTER column");
      columnCapacity = np;
    }
  }
  for (ic = 0; ic < 6; ic++) {
    if (!center->doCoord[ic])
      continue;
    if (!center->deltaSet[ic]) {
      double sum = 0;
      if (np > 0) {
        status = gpuCudaCoordinateColumnToHost(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, (int)ic,
                                               column, &milliseconds);
        if (status != 0)
          gpuFatalStatus("CENTER column copy", status);
        gpuRecordHelperKernel(milliseconds);
        for (ip = 0; ip < np; ip++)
          sum += column[ip];
      }
      center->delta[ic] = coordinateOffset[ic] = (np > 0 ? sum / np : 0);
      if (center->onceOnly)
        center->deltaSet[ic] = 1;
    } else
      coordinateOffset[ic] = center->delta[ic];
    coordinateMask |= 1u << ic;
  }
  if (np > 0 && coordinateMask) {
    status = gpuCudaCenterBeam(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, coordinateMask,
                               coordinateOffset, 0, p0, 0.0, c_mks, &milliseconds);
    if (status != 0)
      gpuFatalStatus("center beam kernel", status);
    gpuRecordHelperKernel(milliseconds);
  }
  if (center->doCoord[6]) {
    double tmean = 0, error = 0, offset;
    if (np > 0) {
      status = gpuCudaTimeColumnToHost(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, p0, c_mks, column,
                                       &milliseconds);
      if (status != 0)
        gpuFatalStatus("CENTER time column", status);
      gpuRecordHelperKernel(milliseconds);
      for (ip = 0; ip < np; ip++)
        tmean = KahanPlus(tmean, column[ip], &error);
    }
    offset = tmean / np;
    if (center->deltaSet[6])
      offset = center->delta[6];
    if (np > 0) {
      status = gpuCudaCenterTimeRoundTrip(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, p0, offset, c_mks,
                                          &milliseconds);
      if (status != 0)
        gpuFatalStatus("CENTER time kernel", status);
      gpuRecordHelperKernel(milliseconds);
    }
    if (center->onceOnly && !center->deltaSet[6]) {
      center->delta[6] = offset;
      center->deltaSet[6] = 1;
    }
  }
  if (np > 0 && (coordinateMask || center->doCoord[6]))
    gpuMarkDeviceChanged(np);
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
      matr->sReference = gpuCoordinateSumInCpuOrder(np, 4, "MATR reference time") / np;
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
      matr->sReference = gpuCoordinateSumInCpuOrder(np, 4, "EMATRIX reference time") / np;
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
      if (gpuEnableApertureParallelCompaction && (globalLossCoordOffset <= 0 || eptr))
        return gpuRectangularCollimatorStableCompact(rcol, np, gpuApertureHostAccepted(accepted), z, Po,
                                                     (ELEMENT_LIST *)eptr, openCode);
      return gpuRectangularCollimatorOnCpu(rcol, np, accepted, z, Po, (ELEMENT_LIST *)eptr,
                                           "rectangular_collimator entrance loss fallback");
    }
    if (rcol->length > 0 && gpuRectangularCollimatorLossCount(rcol, np, rcol->length, openCode)) {
      if (gpuEnableApertureParallelCompaction && (globalLossCoordOffset <= 0 || eptr))
        return gpuRectangularCollimatorStableCompact(rcol, np, gpuApertureHostAccepted(accepted), z, Po,
                                                     (ELEMENT_LIST *)eptr, openCode);
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

static long gpuLimitAmplitudeLossCount(double xmax, double ymax, long np, int openCode) {
  long lostCount = 0;
  float milliseconds = 0;
  int status;

  if (np <= 0)
    return 0;
  gpuCopyHostToDevice(np);
  status = gpuCudaLimitAmplitudeLossCount(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride,
                                          xmax, ymax, openCode, &lostCount, &milliseconds);
  if (status != 0)
    gpuFatalStatus("limit_amplitudes aperture predicate kernel", status);
  gpuRecordApertureKernel(milliseconds);
  return lostCount;
}

static long gpuLimitAmplitudesStableCompact(double xmax, double ymax, int openCode, long np,
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
    np, (int)gpuBase.deviceStride, xmax, ymax, openCode, z, Po, extrapolate_z && !openCode,
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

long gpu_limit_amplitudes(double xmax, double ymax, long np, double **accepted,
                          double z, double Po, long extrapolate_z, long openCode,
                          void *eptr) {
  long lostCount;

  if (np <= 0 || (xmax < 0 && ymax < 0))
    return np;
  if (xmax == DBL_MAX && ymax == DBL_MAX) {
    /* Only invalid coordinates can be lost here.  The device's kernels remove their own, except those of
     * the types moved on November 15, 2026, which count them and leave them to this call, as the CPU does */
    long pending = gpuPortInvalidPending;

    gpuPortInvalidPending = 0;
    if (!pending || !gpuPortInvalidSurvivors())
      return np;
  } else
    gpuPortInvalidPending = 0;
  if (!gpuResidentParticleCountAllowed(np, gpuBase.apertureMinParticles))
    return gpuLimitAmplitudesOnCpu(xmax, ymax, np, accepted, z, Po, extrapolate_z,
                                   openCode, (ELEMENT_LIST *)eptr,
                                   "limit_amplitudes below CUDA aperture threshold");
  lostCount = gpuLimitAmplitudeLossCount(xmax, ymax, np, (int)openCode);
  if (!lostCount) {
    gpuRecordWallSeconds();
    return np;
  }
  if (gpuEnableApertureParallelCompaction && (globalLossCoordOffset <= 0 || eptr))
    return gpuLimitAmplitudesStableCompact(xmax, ymax, (int)openCode, np, z, Po,
                                           extrapolate_z, gpuApertureHostAccepted(accepted),
                                           (ELEMENT_LIST *)eptr);
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

long gpu_removeInvalidParticles(long np, double **accepted, double z, double Po) {
  long lostCount;

  if (np <= 0)
    return np;
  if (!gpuResidentParticleCountAllowed(np, gpuBase.apertureMinParticles))
    return gpuRemoveInvalidParticlesOnCpu(np, accepted, z, Po,
                                          "removeInvalidParticles below CUDA aperture threshold");
  lostCount = gpuRemoveInvalidParticlesLossCount(np);
  if (!lostCount) {
    gpuRecordWallSeconds();
    return np;
  }
  if (gpuEnableApertureParallelCompaction)
    return gpuRemoveInvalidParticlesStableCompact(np, gpuApertureHostAccepted(accepted), z, Po);
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
    if (gpuEnableApertureParallelCompaction && (globalLossCoordOffset <= 0 || eptr))
      return gpuEllipticalCollimatorStableCompact(ecol, np, gpuApertureHostAccepted(accepted), z, Po,
                                                  (ELEMENT_LIST *)eptr,
                                                  ecol->exponent, ye, openCode);
    return gpuEllipticalCollimatorOnCpu(ecol, np, accepted, z, Po, (ELEMENT_LIST *)eptr,
                                        "elliptical_collimator entrance loss fallback");
  }
  if (ecol->length > 0 &&
      gpuEllipticalCollimatorLossCount(ecol, np, ecol->length,
                                       ecol->exponent, ye, openCode)) {
    if (gpuEnableApertureParallelCompaction && (globalLossCoordOffset <= 0 || eptr))
      return gpuEllipticalCollimatorStableCompact(ecol, np, gpuApertureHostAccepted(accepted), z, Po,
                                                  (ELEMENT_LIST *)eptr,
                                                  ecol->exponent, ye, openCode);
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
                                        long exponent, long yExponent, int openCode) {
  long lostCount = 0;
  float milliseconds = 0;
  int status;

  if (np <= 0)
    return 0;
  gpuCopyHostToDevice(np);
  status = gpuCudaELimitAmplitudeLossCount(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride,
                                           xmax, ymax, exponent, yExponent, openCode,
                                           &lostCount, &milliseconds);
  if (status != 0)
    gpuFatalStatus("elimit_amplitudes aperture predicate kernel", status);
  gpuRecordApertureKernel(milliseconds);
  return lostCount;
}

static long gpuELimitAmplitudesStableCompact(double xmax, double ymax, int openCode, long np,
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
    np, (int)gpuBase.deviceStride, xmax, ymax, exponent, yExponent, openCode, z, Po,
    extrapolate_z && !openCode, &remaining, &milliseconds);
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
  if (!gpuResidentParticleCountAllowed(np, gpuBase.apertureMinParticles))
    return gpuELimitAmplitudesOnCpu(xmax, ymax, np, accepted, z, Po, extrapolate_z,
                                    openCode, exponent, yExponent, (ELEMENT_LIST *)eptr,
                                    "elimit_amplitudes below CUDA aperture threshold");
  lostCount = gpuELimitAmplitudeLossCount(xmax, ymax, np, xe, ye, (int)openCode);
  if (!lostCount) {
    gpuRecordWallSeconds();
    return np;
  }
  if (gpuEnableApertureParallelCompaction && (globalLossCoordOffset <= 0 || eptr))
    return gpuELimitAmplitudesStableCompact(xmax, ymax, (int)openCode, np, z, Po,
                                            extrapolate_z, xe, ye, gpuApertureHostAccepted(accepted),
                                            (ELEMENT_LIST *)eptr);
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
    if (gpuEnableApertureParallelCompaction && (globalLossCoordOffset <= 0 || eptr))
      return gpuScraperStableCompact(scraper, np, gpuApertureHostAccepted(accepted), z, Po, plane, sideSign,
                                     (ELEMENT_LIST *)eptr);
    return gpuBeamScraperOnCpu(scraper, np, accepted, z, Po, (ELEMENT_LIST *)eptr,
                               "beam_scraper entrance loss fallback");
  }
  if (scraper->length > 0 &&
      gpuScraperLossCount(scraper, np, plane, sideSign, scraper->length)) {
    if (gpuEnableApertureParallelCompaction && (globalLossCoordOffset <= 0 || eptr))
      return gpuScraperStableCompact(scraper, np, gpuApertureHostAccepted(accepted), z, Po, plane, sideSign,
                                     (ELEMENT_LIST *)eptr);
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
  if (!gpuResidentParticleCountAllowed(np, gpuBase.apertureMinParticles))
    return gpuApertureDataOnCpu(np, accepted, z, Po, apData, eptr,
                                "aperture_data below CUDA aperture threshold");
  if (!gpuApertureDataLossCount(np, xCenter, yCenter, xSize, ySize)) {
    gpuRecordWallSeconds();
    return np;
  }
  if (gpuEnableApertureParallelCompaction)
    return gpuApertureDataStableCompact(np, gpuApertureHostAccepted(accepted), z, Po,
                                        xCenter, yCenter, xSize, ySize);
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
                                 int64_t startPID, int64_t endPID,
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
  if (spinCoordOffset && !sums->spinSums)
    return 0;
  return 1;
}

static void gpuAccumulateBeamSumsOnCpu(void *sums, long n_part, double p_central, double mp_charge,
                                       double *timeValue, double tMin, double tMax,
                                       int64_t startPID, int64_t endPID, unsigned long flags) {
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
                              int64_t startPID, int64_t endPID, unsigned long flags) {
  GPU_BEAM_SUM_DATA result, centeredResult;
  BEAM_SUMS *beamSums = (BEAM_SUMS *)sums;
  float milliseconds = 0;
  int64_t oldCount, newCount;
  long i, j, sparseAllowed;
  unsigned int productMask = 0;
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

  gpuLastBeamSumResultValid = 0;
  gpuLastBeamSumSigma56 = 0;

  if (beamSums && beamSums->beamSums2 &&
      !gpuOutputDriftReductionMinParticlesExplicit &&
      !gpuOutputBeamSumsAllowed(flags)) {
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
    if (flags & BEAM_SUMS_SPARSE) {
      for (i = 0; i < 7; i++)
        for (j = i; j < 7; j++)
          if (sparse[i][j])
            productMask |= 1U << gpuUpperTriangularIndex(i, j);
      /* For a WATCH's el.  sigma[5][6] itself stays 0, as on the CPU. */
      productMask |= 1U << gpuUpperTriangularIndex(5, 6);
    } else {
      productMask = 0x0fffffffU;
    }
    gpuEnsureBeamSumsScratch(n_part);
    status = gpuCudaBeamSums2(gpuBase.deviceCoord, n_part,
                              (int)gpuBase.deviceStride,
                              p_central, c_mks, &result, &centeredResult,
                              gpuBeamSumsScratch.data, productMask,
                              &milliseconds);
  } else {
    gpuEnsureBeamSumsScratch(n_part);
    status = gpuCudaBeamSums2(gpuBase.deviceCoord, n_part,
                              (int)gpuBase.deviceStride,
                              p_central, c_mks, &result, &centeredResult,
                              gpuBeamSumsScratch.data, 0U,
                              &milliseconds);
  }
  if (status != 0)
    gpuFatalStatus("compound beam sums reduction kernel", status);
  gpuRecordReductionKernel(milliseconds);
  gpuLastBeamSumResult = result;
  gpuLastBeamSumResultValid = 1;

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
  if (spinCoordOffset) {
    double spinCentroidSum[3], spinProductSum[6];
    long k;

    status = gpuCudaSpinSums(gpuBase.deviceCoord, n_part, (int)gpuBase.deviceStride, (int)spinCoordOffset,
                             spinCentroidSum, spinProductSum, &milliseconds);
    if (status != 0)
      gpuFatalStatus("spin sums reduction kernel", status);
    gpuRecordReductionKernel(milliseconds);
    for (i = 0; i < 3; i++)
      beamSums->spinSums->centroid[i] =
        (beamSums->spinSums->centroid[i] * oldCount + spinCentroidSum[i]) / newCount;
    for (i = k = 0; i < 3; i++)
      for (j = i; j < 3; j++, k++)
        beamSums->spinSums->sigma[j][i] =
          (beamSums->spinSums->sigma[i][j] =
             (beamSums->spinSums->sigma[i][j] * oldCount + spinProductSum[k]) / newCount);
  }

  if (beamSums->beamSums2) {
    if (centeredResult.count != result.count)
      gpuRequiredFailure("centered beam sums particle count mismatch");
    gpuLastBeamSumSigma56 =
      centeredResult.productSum[gpuUpperTriangularIndex(5, 6)] / result.count;
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
  if (gpuCsrCpuForOrderedOutput())
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
  return gpuSelectDevice();
}

/* The constants of integrate_csbend_ordn()'s drift, by the same arithmetic as computeCsbendDriftSteps()
 * in csbend.c: the slice length times the substep's fraction, over the bending radius.  The fractions are
 * that function's. */
static void gpuPackCsbendDriftSteps(GPU_CSBEND_DATA *data) {
#define GPU_CSBEND_BETA 1.25992104989487316477
  static const double driftFrac2[2] = {
    0.5, 0.5};
  static const double driftFrac4[4] = {
    0.5 / (2 - GPU_CSBEND_BETA), (1 - GPU_CSBEND_BETA) / (2 - GPU_CSBEND_BETA) / 2,
    (1 - GPU_CSBEND_BETA) / (2 - GPU_CSBEND_BETA) / 2, 0.5 / (2 - GPU_CSBEND_BETA)};
  static const double driftFrac6[8] = {
    0.39225680523878,
    0.5100434119184585,
    -0.47105338540975655,
    0.0687531682525181,
    0.0687531682525181,
    -0.47105338540975655,
    0.5100434119184585,
    0.39225680523878,
  };
#undef GPU_CSBEND_BETA
  const double *driftFrac = driftFrac2;
  long nSubsteps = 2, j;
  double ds = data->length / data->nSlices;

  if (data->integrationOrder == 4) {
    driftFrac = driftFrac4;
    nSubsteps = 4;
  } else if (data->integrationOrder == 6) {
    driftFrac = driftFrac6;
    nSubsteps = 8;
  }
  for (j = 0; j < nSubsteps; j++) {
    double angle = ds * driftFrac[j] / data->rho0;
    double sinHalf = sin(angle / 2);
    data->driftSin[j] = sin(angle);
    data->driftCos[j] = cos(angle);
    data->driftVers[j] = 2 * sinHalf * sinHalf;
  }
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
  gpuPackCsbendDriftSteps(data);
  data->cosTilt = 1;
  data->sinTilt = 0;
  memcpy(data->Fx, *Fx_xy, sizeof(data->Fx));
  memcpy(data->Fy, *Fy_xy, sizeof(data->Fy));
  cachedData = *data;
  cachedCsbend = csbend;
  memcpy(cachedB, csbend->b, sizeof(cachedB));
  memcpy(cachedC, csbend->c, sizeof(cachedC));
  cachedSliceLength = sliceLength;
  cachedRho0 = localRho0;
  cachedRhoActual = localRhoActual;
  cachedValid = 1;
  return 1;
}

long gpu_csr_csbend_resident_available(void *csbend0, long nParticles,
                                       long nBins) {
  CSRCSBEND *csbend = (CSRCSBEND *)csbend0;

#if USE_MPI
  /* csbend.c builds its CUDA CSR steps only without MPI, so the device entry would run the CPU's
   * steps on a host copy that may be stale */
  (void)csbend;
  (void)nParticles;
  (void)nBins;
  return 0;
#else
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
#endif
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

#ifdef GPU_VERIFY
  double *cpuDp = NULL, *gpuStorage = NULL;
  long ip, stride = gpuBase.deviceStride;
  unsigned long count = (unsigned long)nParticles * (unsigned long)stride;

  if (gpuBase.verifyMode) {
    double **coord = copyParticlesToCpuReadOnly("CSRCSBEND resident kick verification input");
    double *dGamma = (double *)malloc((size_t)nBins * sizeof(*dGamma));

    cpuDp = (double *)malloc((size_t)nParticles * sizeof(*cpuDp));
    gpuStorage = (double *)malloc(count * sizeof(*gpuStorage));
    if (!dGamma || !cpuDp || !gpuStorage)
      gpuRequiredFailure("unable to allocate CUDA CSRCSBEND kick verification buffers");
    status = gpuCudaCopyDeviceToHost(dGamma, gpuCsrScratch.dGamma, (unsigned long)nBins,
                                     &milliseconds);
    if (status != 0)
      gpuFatalStatus("cudaMemcpy(CSRCSBEND dGamma device to verify host)", status);
    gpuRecordMilliseconds(&gpuBase.gpuTransferToHostSeconds, milliseconds);
    for (ip = 0; ip < nParticles; ip++) {
      double f = (coord[ip][4] - ctLower) / dct;
      long bin = (long)f;

      f -= bin;
      cpuDp[ip] = coord[ip][5];
      if (bin >= 0 && bin < nBins - 1)
        cpuDp[ip] += ((1 - f) * dGamma[bin] + f * dGamma[bin + 1]) / Po *
                     (1 + coord[ip][0] / localRho0);
    }
    free(dGamma);
  }
#endif

  status = gpuCudaCsrCsbendKickInPlace(gpuBase.deviceCoord, nParticles,
                                       (int)gpuBase.deviceStride,
                                       (const double *)gpuCsrScratch.dGamma,
                                       nBins, ctLower, dct, Po, localRho0,
                                       &milliseconds);
  if (status != 0)
    gpuFatalStatus("CSRCSBEND resident kick CUDA kernel", status);
  gpuRecordCsrKernel(milliseconds);
#ifdef GPU_VERIFY
  if (gpuBase.verifyMode) {
    double absTol = gpuCompareAbsTolerance();
    double relTol = gpuCompareRelTolerance();
    double maxAbs = 0, maxRel = 0;
    long mismatches = 0;

    status = gpuCudaCopyDeviceToHost(gpuStorage, gpuBase.deviceCoord, count, &milliseconds);
    if (status != 0)
      gpuFatalStatus("cudaMemcpy(CSRCSBEND kick device to verify host)", status);
    gpuRecordMilliseconds(&gpuBase.gpuTransferToHostSeconds, milliseconds);
    for (ip = 0; ip < nParticles; ip++) {
      double gpu = gpuStorage[ip * stride + 5];
      double absDiff = fabs(cpuDp[ip] - gpu);
      double scale = fmax(fabs(cpuDp[ip]), fabs(gpu));
      double relDiff = scale > DBL_MIN ? absDiff / scale : absDiff;

      if (absDiff > maxAbs)
        maxAbs = absDiff;
      if (relDiff > maxRel)
        maxRel = relDiff;
      if (!(absDiff <= absTol || relDiff <= relTol))
        mismatches++;
    }
    free(cpuDp);
    free(gpuStorage);
    if (mismatches) {
      fprintf(stderr,
              "elegant CUDA VERIFY failed for CSRCSBEND resident kick: %ld mismatches, maxAbs=%.3e, maxRel=%.3e, absTol=%.3e, relTol=%.3e\n",
              mismatches, maxAbs, maxRel, absTol, relTol);
      exit(1);
    }
    if (gpuVerbose)
      fprintf(stderr, "elegant CUDA VERIFY passed for CSRCSBEND resident kick: maxAbs=%.3e maxRel=%.3e\n",
              maxAbs, maxRel);
  }
#endif
  gpuMarkDeviceChanged(nParticles);

  if (startedWallTimer)
    gpuRecordWallSeconds();
  return 1;
}

/* CSRCSBEND converts the beam out of its bend coordinates and back around slice analysis output
 * inside the bend, which changes the times by rounding. The host copy the output read has been
 * through that; this does the same to the device rows, so both hold elegant's numbers. */
long gpu_csr_csbend_output_round_trip(long nParticles, double cosTtilt, double sinTtilt) {
  float milliseconds = 0;
  int status;
  long startedWallTimer = 0;

  if (!gpuBase.deviceCurrent || nParticles <= 0)
    return 0;
  if (gpuWallStart <= 0) {
    gpuWallStart = wallSeconds();
    startedWallTimer = 1;
  }
  status = gpuCudaCsrCsbendOutputRoundTrip(gpuBase.deviceCoord, nParticles,
                                           (int)gpuBase.deviceStride,
                                           cosTtilt, sinTtilt, c_mks, &milliseconds);
  if (status != 0)
    gpuFatalStatus("CSRCSBEND output round trip CUDA kernel", status);
  gpuRecordCsrKernel(milliseconds);
  gpuMarkDeviceChanged(nParticles);
  if (startedWallTimer)
    gpuRecordWallSeconds();
  return 1;
}

/* checkMultAperture()'s checks as setupMultApertureData() left them: MAXAMP, an aperture_data file
 * where it was evaluated, and an APCONTOUR, used by CSRCSBEND body slices and BMXYZ */
static long gpuPackMultApertureData(GPU_APERTURE_LIMIT_DATA *aperture,
                                    GPU_APERTURE_CONTOUR_DATA *contour,
                                    MULT_APERTURE_DATA *apertureData) {
  memset(aperture, 0, sizeof(*aperture));
  memset(contour, 0, sizeof(*contour));
  aperture->cosReverseTilt = 1;
  if (!apertureData)
    return 1;
  if (apertureData->localAperture)
    return 0;
  aperture->xCen = apertureData->xCen;
  aperture->yCen = apertureData->yCen;
  aperture->xMax = apertureData->xMax;
  aperture->yMax = apertureData->yMax;
  aperture->elliptical = apertureData->elliptical ? 1 : 0;
  aperture->xExponent = apertureData->xExponent;
  aperture->yExponent = apertureData->yExponent;
  aperture->openSide = apertureData->openSide;
  aperture->cosReverseTilt = apertureData->reverseTiltCS[0];
  aperture->sinReverseTilt = apertureData->reverseTiltCS[1];
  aperture->present = aperture->xMax > 0 || aperture->yMax > 0;
  if (aperture->elliptical && aperture->xMax > 0 && aperture->yMax > 0 &&
      (aperture->xExponent <= 0 || aperture->yExponent <= 0))
    return 0;
  if (!isfinite(aperture->xMax) || !isfinite(aperture->yMax) ||
      !isfinite(aperture->xCen) || !isfinite(aperture->yCen))
    return 0;
  return gpuPackApertureContour(contour, apertureData->apContour);
}

long gpu_csr_csbend_apertures_supported(void *apertureData) {
  GPU_CSBEND_DATA data;

  return gpuPackMultApertureData(&data.aperture, &data.contour, (MULT_APERTURE_DATA *)apertureData);
}

/* With apertures, particles lost as the CPU loses them stay lost on the device, in the CPU's order.
 * A losses file gets their rows from the device and acceptance output moves with them.  Global loss
 * coordinates stay on the CPU, which stops at the first loss in a CSRCSBEND, and so do the verify
 * build's losses and acceptance, whose CPU shadow here keeps neither */
long gpu_track_csbend_csr_body_slice(void *csbend0, long nParticles,
                                     double sliceLength, double localRho0,
                                     double localRhoActual, double Po,
                                     void *apertureData0, double zSliceStart,
                                     double cosTtilt, double sinTtilt,
                                     double **accepted, long *nLeft) {
  CSRCSBEND *csbend = (CSRCSBEND *)csbend0;
  MULT_APERTURE_DATA *apertureData = (MULT_APERTURE_DATA *)apertureData0;
  GPU_CSBEND_DATA data;
  long lostCount = 0, remaining = nParticles;
  int keepLosses;
  float milliseconds = 0;
  int status;
  long startedWallTimer = 0;
#ifdef GPU_VERIFY
  double *cpuStorage = NULL, **cpuCoord = NULL, *cpuBeta0 = NULL;
  double *gpuStorage = NULL;
  long stride = 0, ip, ic, cpuLeft = nParticles;
  unsigned long count = 0;
#endif

  if (!gpuEnableCsrResident || !gpuBase.deviceCurrent ||
      !csbend || nParticles <= 0 || !gpuCsrScratch.kickDp)
    return 0;
  if (!gpuPackCsrCsbendBodyTracking(&data, csbend, sliceLength,
                                    localRho0, localRhoActual) ||
      !gpuPackMultApertureData(&data.aperture, &data.contour, apertureData))
    return 0;
  data.Po = Po;
  data.lossZStart = zSliceStart; /* where this slice starts, which a lost particle's distance adds to */
  data.lossCosTtilt = cosTtilt;
  data.lossSinTtilt = sinTtilt;
  keepLosses = globalLossCoordOffset <= 0;
#ifdef GPU_VERIFY
  keepLosses = keepLosses && !gpuBase.lossOutputNeeded && !accepted;
#endif
  /* Until CPU fixes 15 and 16 are committed (elegant-CPU-bugs-and-fixes.md), track_through_csbendCSR()
   * leaves beta0[] in place when it swaps a lost particle out, and keepLosses would move it with the particle.
   * A slice that loses a particle is restored and handed to the CPU, which does it and the rest of the bend
   * with elegant's own numbers.  Remove this line with those fixes. */
  keepLosses = 0;
  if (keepLosses)
    gpuEnsureApertureScratch(nParticles);

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

    /* track_through_csbendCSR()'s loop: a lost particle is swapped with the last one not yet tracked */
    for (ip = 0; ip < cpuLeft; ip++) {
      if (!gpu_verify_csrcsbend_cpu_body_slice(cpuCoord[ip], csbend,
                                               cpuBeta0[ip], sliceLength,
                                               localRho0, localRhoActual,
                                               Po, apertureData)) {
        double *row = cpuCoord[ip], beta = cpuBeta0[ip];

        cpuLeft--;
        cpuCoord[ip] = cpuCoord[cpuLeft];
        cpuCoord[cpuLeft] = row;
        cpuBeta0[ip] = cpuBeta0[cpuLeft];
        cpuBeta0[cpuLeft] = beta;
        ip--;
      }
    }
  }
#endif
  gpuEnsureCsrBodyScratch(nParticles, gpuBase.deviceStride);
  status = gpuCudaCsrCsbendBodySliceChecked(gpuBase.deviceCoord, nParticles,
                                            (int)gpuBase.deviceStride,
                                            &data,
                                            (double *)gpuCsrScratch.kickDp,
                                            gpuCsrScratch.bodyBackup,
                                            keepLosses,
                                            keepLosses ? (long *)gpuApertureScratch.prefix : NULL,
                                            &remaining, &lostCount, &milliseconds);
  if (status != 0)
    gpuFatalStatus("CSRCSBEND resident body CUDA checked kernel", status);
  gpuRecordMagnetKernel(milliseconds);
#ifdef GPU_VERIFY
  if (gpuBase.verifyMode && !lostCount && cpuLeft != remaining) {
    fprintf(stderr, "elegant CUDA VERIFY failed for CSRCSBEND resident body: CPU keeps %ld particles, GPU %ld\n",
            cpuLeft, remaining);
    exit(1);
  }
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
    for (ip = 0; ip < remaining; ip++) {
      for (ic = 0; ic < stride; ic++) {
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
    if (remaining < nParticles) {
      /* the losses moved beta0 with the particles, so it is the shadow's exactly */
      status = gpuCudaCopyDeviceToHost(gpuStorage, gpuCsrScratch.kickDp, (unsigned long)remaining,
                                       &milliseconds);
      if (status != 0)
        gpuFatalStatus("cudaMemcpy(CSRCSBEND beta0 device to verify host)", status);
      for (ip = 0; ip < remaining; ip++)
        if (gpuStorage[ip] != cpuBeta0[ip]) {
          fprintf(stderr, "elegant CUDA VERIFY failed for CSRCSBEND resident body beta0 particle=%ld cpu=%.17e gpu=%.17e\n",
                  ip, cpuBeta0[ip], gpuStorage[ip]);
          exit(1);
        }
    }
    if (gpuVerbose)
      fprintf(stderr,
              "elegant CUDA VERIFY passed for CSRCSBEND resident body: %ld of %ld kept, maxAbs=%.3e maxRel=%.3e\n",
              remaining, nParticles, maxAbs, maxRel);
    gpuStoreCsrCtCpuShadow(cpuCoord, remaining);
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
  if (remaining < nParticles) {
    /* the destinations are where the aperture helpers look for them */
    if (accepted &&
        !gpuStablePartitionAcceptedOnDevice(accepted, nParticles, remaining,
                                            "CSRCSBEND resident body accepted compaction")) {
      gpuCopyAperturePrefixToHost(nParticles, "CSRCSBEND resident body accepted map");
      gpuStablePartitionAccepted(accepted, nParticles, remaining);
    }
    gpuCopyApertureLossRowsToHost(remaining, nParticles, "CSRCSBEND resident body loss rows");
  }
  gpuMarkDeviceChanged(remaining);
  *nLeft = remaining;

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
  /* see gpuCsrCsbendDeviceEntrySupported(): a fringe term's tangent is taken for every particle */
  if (gpuBase.orderSensitiveOutputNeeded && edge1Effect && psi1 != 0)
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
  if (gpuBase.orderSensitiveOutputNeeded && edge2Effect && psi2 != 0)
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
                                 long copyDGammaArray,
                                 double t1Factor,
                                 double t2Factor) {
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
                                diSlippage, diSlippage4, t1Factor, t2Factor,
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
/* CSRDRIFT with CSR on.  track_through_driftCSR() and track_through_driftCSR_Stupakov() in csbend.c
 * keep everything that works on the wake's bins; these do their loops over particles on the device
 * rows, in the same arithmetic, so the particles are the CPU's to the last bit.  Each returns 0 when
 * it can't run, and csbend.c then brings the beam to the host and carries on there. */
#ifdef GPU_VERIFY
/* The beam before a step, columns 0 to 5 of each particle, for the CPU's arithmetic to be applied to */
static double *gpuCsrDriftVerifyRows(long nParticles, const char *reason) {
  double **coord = copyParticlesToCpuReadOnly(reason);
  double *rows = (double *)malloc((size_t)nParticles * 6 * sizeof(*rows));
  long ip, i;

  if (!rows)
    gpuRequiredFailure("unable to allocate CUDA CSRDRIFT verification rows");
  for (ip = 0; ip < nParticles; ip++)
    for (i = 0; i < 6; i++)
      rows[ip * 6 + i] = coord[ip][i];
  return rows;
}

static void gpuCsrDriftVerifyCompare(const char *what, double *expected, long nParticles) {
  long stride = gpuBase.deviceStride, ip, i, mismatches = 0;
  unsigned long count = (unsigned long)nParticles * (unsigned long)stride;
  double *storage = (double *)malloc(count * sizeof(*storage));
  double absTol = gpuCompareAbsTolerance(), relTol = gpuCompareRelTolerance();
  double maxAbs = 0, maxRel = 0;
  float milliseconds = 0;
  int status;

  if (!storage)
    gpuRequiredFailure("unable to allocate CUDA CSRDRIFT verification buffer");
  status = gpuCudaCopyDeviceToHost(storage, gpuBase.deviceCoord, count, &milliseconds);
  if (status != 0)
    gpuFatalStatus("cudaMemcpy(CSRDRIFT device to verify host)", status);
  gpuRecordMilliseconds(&gpuBase.gpuTransferToHostSeconds, milliseconds);
  for (ip = 0; ip < nParticles; ip++)
    for (i = 0; i < 6; i++) {
      double cpu = expected[ip * 6 + i], gpu = storage[ip * stride + i];
      double absDiff = fabs(cpu - gpu);
      double scale = fmax(fabs(cpu), fabs(gpu));
      double relDiff = scale > DBL_MIN ? absDiff / scale : absDiff;

      if (absDiff > maxAbs)
        maxAbs = absDiff;
      if (relDiff > maxRel)
        maxRel = relDiff;
      if (!(absDiff <= absTol || relDiff <= relTol))
        mismatches++;
    }
  free(storage);
  free(expected);
  if (mismatches) {
    fprintf(stderr,
            "elegant CUDA VERIFY failed for CSRDRIFT %s: %ld mismatches, maxAbs=%.3e, maxRel=%.3e, absTol=%.3e, relTol=%.3e\n",
            what, mismatches, maxAbs, maxRel, absTol, relTol);
    exit(1);
  }
  if (gpuVerbose)
    fprintf(stderr, "elegant CUDA VERIFY passed for CSRDRIFT %s: maxAbs=%.3e maxRel=%.3e\n",
            what, maxAbs, maxRel);
}
#endif

static long gpuCsrDriftDeviceReady(long nParticles) {
  if (!gpuEnableCsrResident || !gpuEnableCsrDriftElement || nParticles <= 1)
    return 0;
  if (!gpuBase.initialized || gpuBase.activeDevice < 0 || !gpuSelectDevice())
    return 0;
  gpuCopyHostToDevice(nParticles);
  return gpuBase.deviceCurrent;
}

/* x += xp*dz, y += yp*dz, and s += dz*sqrt(1 + xp^2 + yp^2) (or s += dz with LINEARIZE); with toTime
 * the result is divided by beta, which makes it c*t */
long gpu_csr_drift_advance(long nParticles, double dz, double Po, long linearOptics, long toTime) {
  float milliseconds = 0;
  int status;
  long startedWallTimer = 0;
#ifdef GPU_VERIFY
  double *expected = NULL;
#endif

  if (!gpuCsrDriftDeviceReady(nParticles))
    return 0;
  if (gpuWallStart <= 0) {
    gpuWallStart = wallSeconds();
    startedWallTimer = 1;
  }
#ifdef GPU_VERIFY
  if (gpuBase.verifyMode) {
    long ip;

    expected = gpuCsrDriftVerifyRows(nParticles, "CSRDRIFT advance verification input");
    for (ip = 0; ip < nParticles; ip++) {
      double *coord = expected + ip * 6;

      coord[0] += coord[1] * dz;
      coord[2] += coord[3] * dz;
      if (toTime) {
        double p = Po * (1 + coord[5]);
        double beta = p / sqrt(p * p + 1);

        if (linearOptics)
          coord[4] = (coord[4] + dz) / beta;
        else
          coord[4] = (coord[4] + dz * sqrt(1 + sqr(coord[1]) + sqr(coord[3]))) / beta;
      } else if (linearOptics)
        coord[4] += dz;
      else
        coord[4] += dz * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));
    }
  }
#endif
  status = gpuCudaCsrDriftAdvance(gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride,
                                  dz, Po, linearOptics ? 1 : 0, toTime ? 1 : 0, &milliseconds);
  if (status != 0)
    gpuFatalStatus("CSRDRIFT advance CUDA kernel", status);
  gpuRecordCsrKernel(milliseconds);
  gpuMarkDeviceChanged(nParticles);
#ifdef GPU_VERIFY
  if (expected)
    gpuCsrDriftVerifyCompare(toTime ? "advance to c*t" : "drift", expected, nParticles);
#endif
  if (startedWallTimer)
    gpuRecordWallSeconds();
  return 1;
}

/* The number of particles the kick loop finds inside the wake's bins, or -1 */
long gpu_csr_drift_count_in_range(long nParticles, double ct0, double dct, long nBins) {
  float milliseconds = 0;
  int status;
  long count = 0, startedWallTimer = 0;

  if (nBins < 2 || !(dct > 0) || !gpuCsrDriftDeviceReady(nParticles))
    return -1;
  if (gpuWallStart <= 0) {
    gpuWallStart = wallSeconds();
    startedWallTimer = 1;
  }
  status = gpuCudaCsrDriftCountInRange(gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride,
                                       ct0, dct, nBins, &count, &milliseconds);
  if (status != 0)
    gpuFatalStatus("CSRDRIFT bin range count", status);
  gpuRecordReductionKernel(milliseconds);
#ifdef GPU_VERIFY
  if (gpuBase.verifyMode) {
    double **coord = copyParticlesToCpuReadOnly("CSRDRIFT bin range verification input");
    long ip, cpuCount = 0;

    for (ip = 0; ip < nParticles; ip++) {
      long iBin = (coord[ip][4] - ct0) / dct;

      if (iBin >= 0 && iBin < nBins - 1)
        cpuCount++;
    }
    if (cpuCount != count) {
      fprintf(stderr, "elegant CUDA VERIFY failed for CSRDRIFT bin range count: cpu=%ld gpu=%ld\n",
              cpuCount, count);
      exit(1);
    }
  }
#endif
  if (startedWallTimer)
    gpuRecordWallSeconds();
  return count;
}

/* The wake's kick by linear interpolation in dGamma (times factor when useFactor is set), then c*t back
 * to s.  dGamma NULL is the CPU with no wake stored; applyKick 0 is the Stupakov loop with CSR off. */
long gpu_csr_drift_kick(long nParticles, const double *dGamma, long nBins, long applyKick,
                        double ct0, double dct, double Po, long useFactor, double factor) {
  float milliseconds = 0;
  int status;
  long startedWallTimer = 0;
  long kick = dGamma && applyKick;
#ifdef GPU_VERIFY
  double *expected = NULL;
#endif

  if (!Po || (kick && (nBins < 2 || !(dct > 0))) || !gpuCsrDriftDeviceReady(nParticles))
    return 0;
  if (gpuWallStart <= 0) {
    gpuWallStart = wallSeconds();
    startedWallTimer = 1;
  }
#ifdef GPU_VERIFY
  if (gpuBase.verifyMode) {
    long ip;

    expected = gpuCsrDriftVerifyRows(nParticles, "CSRDRIFT kick verification input");
    for (ip = 0; ip < nParticles; ip++) {
      double *coord = expected + ip * 6;
      double p, beta;

      if (kick) {
        double f;
        long iBin = (f = (coord[4] - ct0) / dct);

        f -= iBin;
        if (iBin >= 0 && iBin < nBins - 1) {
          if (useFactor)
            coord[5] += ((1 - f) * dGamma[iBin] + f * dGamma[iBin + 1]) / Po * factor;
          else
            coord[5] += ((1 - f) * dGamma[iBin] + f * dGamma[iBin + 1]) / Po;
        }
      }
      p = (1 + coord[5]) * Po;
      beta = p / sqrt(p * p + 1);
      coord[4] = beta * coord[4];
    }
  }
#endif
  if (kick) {
    gpuEnsureCsrScratch(nBins);
    status = gpuCudaCopyHostToDevice(gpuCsrScratch.dGamma, (double *)dGamma, (unsigned long)nBins,
                                     &milliseconds);
    if (status != 0)
      gpuFatalStatus("cudaMemcpy(CSRDRIFT dGamma host to device)", status);
    gpuRecordMilliseconds(&gpuBase.gpuTransferToDeviceSeconds, milliseconds);
    /* this is the drift's wake now, not the one the last CSRCSBEND kick left on the device */
    gpuCsrScratch.dGammaValid = 0;
    gpuCsrScratch.dGammaBins = 0;
  }
  status = gpuCudaCsrDriftKick(gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride,
                               kick ? (const double *)gpuCsrScratch.dGamma : NULL,
                               nBins, kick, ct0, dct, Po, useFactor ? 1 : 0, factor,
                               &milliseconds);
  if (status != 0)
    gpuFatalStatus("CSRDRIFT kick CUDA kernel", status);
  gpuRecordCsrKernel(milliseconds);
  gpuMarkDeviceChanged(nParticles);
#ifdef GPU_VERIFY
  if (expected)
    gpuCsrDriftVerifyCompare("kick", expected, nParticles);
#endif
  if (startedWallTimer)
    gpuRecordWallSeconds();
  return 1;
}

long gpu_track_through_driftCSR(long np, void *csrDrift0, double Po,
                                double **accepted, double zStart,
                                double revolutionLength, void *charge,
                                char *rootname) {
  CSRDRIFT *csrDrift = (CSRDRIFT *)csrDrift0;

  if (!csrDrift || csrDrift->LSCBins) {
    gpuUnsupported("gpu_track_through_driftCSR");
    return 0;
  }
#if !USE_MPI
  /* with a wake to apply, csbend.c's routine runs with its particle loops on the device */
  if (csrDrift->csr && track_through_driftCSR_cuda_resident_entry && csrDriftHasWakeToApply &&
      csrDriftHasWakeToApply(csrDrift, np))
    return track_through_driftCSR_cuda_resident_entry(gpuBase.coord, np, csrDrift, Po, accepted,
                                                      zStart, revolutionLength, (CHARGE *)charge,
                                                      rootname);
#endif
  (void)Po;
  (void)accepted;
  (void)zStart;
  (void)revolutionLength;
  (void)charge;
  (void)rootname;
  if (csrDrift->linearOptics)
    gpu_linearDrift(np, csrDrift->length);
  else
    gpu_exactDrift(np, csrDrift->length);
  return np;
}
void gpu_addCorrectorRadiationKick() { gpuUnsupported("gpu_addCorrectorRadiationKick"); }

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

long gpu_scmult_kick_supported(long nonlinear, const double *sigma,
                               long horizontal, long vertical,
                               double betax, double betay) {
  if (sigma[2] == 0)
    return 0;
  if (nonlinear && (sigma[0] <= 0 || sigma[1] <= 0))
    return 0;
  if ((horizontal && betax == 0) || (vertical && betay == 0))
    return 0;
  return 1;
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
  GPU_SCMULT_MOMENT_DATA sums;
  double variance;
  float milliseconds = 0;
  int status;
  long i;
  long startedWallTimer = 0;

  (void)Po;

  if (!center || !sigma)
    gpuRequiredFailure("NULL SCMULT moment output pointer in CUDA path");
  for (i = 0; i < 3; i++) {
    center[i] = 0;
    sigma[i] = 0;
  }
  if (!gpuScmultAllowed(nParticles))
    return 0;

  gpuCopyHostToDevice(nParticles);
  if (gpuScmultMomentCacheValid &&
      gpuScmultMomentCacheParticles == nParticles) {
    sums = gpuScmultMomentCache;
    gpuBase.gpuScmultMomentCacheHitCount++;
  } else {
    if (gpuWallStart <= 0) {
      gpuWallStart = wallSeconds();
      startedWallTimer = 1;
    }
    memset(&sums, 0, sizeof(sums));
    status = gpuCudaScmultMoments(gpuBase.deviceCoord, nParticles,
                                  (int)gpuBase.deviceStride, &sums,
                                  &milliseconds);
    if (status != 0)
      gpuFatalStatus("SCMULT centroid/rms reduction kernel", status);
    gpuRecordReductionKernel(milliseconds);
    gpuRecordMilliseconds(&gpuBase.gpuScmultMomentSeconds, milliseconds);
    gpuBase.gpuScmultMomentCount++;
    gpuScmultMomentCache = sums;
    gpuScmultMomentCacheParticles = nParticles;
    gpuScmultMomentCacheValid = 1;
    if (startedWallTimer)
      gpuRecordWallSeconds();
  }

  if (sums.count <= 0)
    return 0;
  for (i = 0; i < 3; i++) {
    center[i] = sums.reference[i] + sums.sum[i] / sums.count;
    variance = sums.squareSum[i] - sums.sum[i] * sums.sum[i] / sums.count;
    variance /= sums.count;
    sigma[i] = sqrt(variance > 0 ? variance : 0);
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
  data.longitudinalScale = c1 / sigma[2] * charge *
    (uniformDistribution ? sqrt(PI / 6.0) : 1.0);

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
  data.longitudinalScale = c1 / sigma[2] * charge *
    (uniformDistribution ? PI / 12.0 : sqrt(PI / 2.0));
  data.roundBeam = fabs(sigma[0] - sigma[1]) / sigma[0] < 1e-6;
  if (data.roundBeam) {
    data.roundSigma = (sigma[0] + sigma[1]) / 2;
  } else {
    double majorSigma = sigma[0];
    double minorSigma = sigma[1];
    double transverseScale;

    data.swapXY = majorSigma < minorSigma;
    if (data.swapXY) {
      double temporary = majorSigma;
      majorSigma = minorSigma;
      minorSigma = temporary;
    }
    data.inverseSd = 1 / sqrt(2.0 *
                              (majorSigma * majorSigma -
                               minorSigma * minorSigma));
    data.minorMajorRatio = minorSigma / majorSigma;
    data.majorMinorRatio = majorSigma / minorSigma;
    data.inverseTwoMajorSigma2 = 1 / (2 * majorSigma * majorSigma);
    data.inverseTwoMinorSigma2 = 1 / (2 * minorSigma * minorSigma);
    transverseScale = sqrt((sigma[0] + sigma[1]) /
                           fabs(sigma[0] - sigma[1]));
    data.kxScale = dmux * sigma[0] * transverseScale / betax;
    data.kyScale = dmuy * sigma[1] * transverseScale / betay;
  }

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
#ifdef GPU_VERIFY
    gpuBase.elementOnGpu = 0;
#endif
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
    double **coord =
      forceParticlesToCpu("POLYNOMIALSERIES particle loss fallback");
#ifdef GPU_VERIFY
    gpuBase.elementOnGpu = 0;
#endif
    return polynomialSeries_tracking(
      coord, nParticles, polynomialSeries, pError, pCentral, accepted, zStart);
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

/* ISR on the GPU uses the CPU's random numbers in the CPU's order, so results stay comparable and
 * a later element sees the same sequence.  The device generates the sequence from random_2's state,
 * and the state is moved past the numbers the element used when it commits.  If a kernel loses a
 * particle, the CPU redoes the element with the same numbers. */
#define GPU_DLARAN_MULTIPLIER 33952834046453ULL
#define GPU_DLARAN_MASK ((1ULL << 48) - 1)
#define GPU_DLARAN_SCALE 3.552713678800501e-15

static struct {
  unsigned long long state;    /* state * GPU_DLARAN_SCALE is the next number in the sequence */
  unsigned long long elementStart;
  unsigned long long elementSeed; /* random_2's state when the element started */
  double limit; /* the Gaussian cutoff for this element's numbers */
  long used;                   /* how many numbers this element has used */
  long hostMath;               /* numbers made on the host, see gpuIsrReferenceMathEnabled() */
} gpuIsrSequence;

/* Moves a dlaran state the given number of steps */
static unsigned long long gpuDlaranJump(unsigned long long state, unsigned long long steps) {
  unsigned long long power = GPU_DLARAN_MULTIPLIER;

  for (; steps; steps >>= 1) {
    if (steps & 1)
      state = (state * power) & GPU_DLARAN_MASK;
    power = (power * power) & GPU_DLARAN_MASK;
  }
  return state;
}

static void gpuIsrBegin(void) {
  gpuIsrSequence.elementSeed = random_2_get_state();
  gpuIsrSequence.state = gpuDlaranJump(gpuIsrSequence.elementSeed, 1);
  gpuIsrSequence.elementStart = gpuIsrSequence.state;
  gpuIsrSequence.limit = srGaussianLimit;
  gpuIsrSequence.used = 0;
  gpuIsrSequence.hostMath = gpuIsrReferenceMathEnabled(0);
}

/* The numbers go back to the sequence for the CPU to use */
static void gpuIsrAbandon(void) {
  gpuIsrSequence.state = gpuIsrSequence.elementStart;
}

static void gpuIsrCommit(void) {
#ifdef GPU_VERIFY
  /* The CPU now tracks with these same numbers, which moves random_2 itself */
  gpuIsrAbandon();
#else
  random_2_set_state(gpuDlaranJump(gpuIsrSequence.elementSeed, gpuIsrSequence.used));
#endif
}


/* The compiled gauss_rn() calls sincos(), which this file's headers don't declare */
extern void sincos(double x, double *sine, double *cosine);

/* gauss_rn()'s number from the pair of uniforms starting at state.  Each number takes exactly two
 * uniforms, so it depends only on where its pair starts and pairs can be made in any order. */
static double gpuIsrHostPairGaussian(unsigned long long state) {
  double u1 = state * GPU_DLARAN_SCALE;
  double u2 = ((state * GPU_DLARAN_MULTIPLIER) & GPU_DLARAN_MASK) * GPU_DLARAN_SCALE;
  double sine, cosine;

  sincos(PIx2 * u2, &sine, &cosine);
  return sqrt(-2 * log(u1)) * sine;
}

/* Host threads for the numbers the device needs from this host's libm (the ISR Gaussian numbers and
 * the RF cavities' sin and log with output in CPU order).  Each number depends on its own index
 * alone, so any count gives the same numbers.
 *
 * One, as elegant runs, unless the run asks for more with -ompThreads or OMP_NUM_THREADS: gpu-elegant
 * starts no thread of its own accord.  (ELEGANT_GPU_ISR_REFERENCE_THREADS, which the GPU test set
 * uses for its SREFFECTS case, still sets the count, as it did before these threads existed.)  The
 * threads sleep between jobs: OpenMP's would spin for a while after each one, which took 18 s of
 * other cores' time in a 14 s LCLS-II run.
 *
 * Waking a sleeping thread costs 0.5 to 1 ms here (idle cores sit in deep C-states), so a job goes to
 * the threads only when one thread would need about 2 ms or more for it: 10,000 particles' sines on
 * 8 threads made modulate4 take 31 s instead of 20.4 s, and 20,000 made LCLSII-1 take 3.8 s instead
 * of 2.9 s, while 200,000 saved 1.4 s of 12.2 s in the full LCLS-II run and 196,000 Gaussian numbers
 * a fill took SREFFECTS from 4.47 s to 1.34 s. */
#define GPU_HOST_THREAD_MIN_SINES 100000
#define GPU_HOST_THREAD_MIN_GAUSSIANS 50000

static long gpuHostThreads(void) {
  long threads = 1;

#ifdef _OPENMP
  threads = omp_get_max_threads();
  if (gpuGetOmpTrackingThreads() > threads)
    threads = gpuGetOmpTrackingThreads();
#endif
  threads = gpuEnvLong("ELEGANT_GPU_ISR_REFERENCE_THREADS", threads);
  return threads < 1 ? 1 : threads;
}

typedef void (*GPU_HOST_RANGE_FUNCTION)(long first, long last, void *context);

static struct {
  pthread_mutex_t lock;
  pthread_cond_t wake, done;
  long workers;    /* threads started, numbered from 1; the caller is number 0 */
  long generation; /* counts the jobs */
  long parts;      /* ranges in the current job: threads 0 to parts-1 take one each */
  long pending;    /* workers that haven't finished their range of the current job */
  long n;
  GPU_HOST_RANGE_FUNCTION function;
  void *context;
} gpuHostPool = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, PTHREAD_COND_INITIALIZER,
                 0, 0, 0, 0, 0, NULL, NULL};

static void gpuHostPoolRange(long part, long parts, long n, long *first, long *last) {
  long chunk = (n + parts - 1) / parts;

  *first = part * chunk < n ? part * chunk : n;
  *last = *first + chunk < n ? *first + chunk : n;
}

static void *gpuHostPoolWorker(void *argument) {
  long number = (long)argument, seen = 0;

  pthread_mutex_lock(&gpuHostPool.lock);
  for (;;) {
    while (gpuHostPool.generation == seen)
      pthread_cond_wait(&gpuHostPool.wake, &gpuHostPool.lock);
    seen = gpuHostPool.generation;
    if (number < gpuHostPool.parts) {
      GPU_HOST_RANGE_FUNCTION function = gpuHostPool.function;
      void *context = gpuHostPool.context;
      long first, last;

      gpuHostPoolRange(number, gpuHostPool.parts, gpuHostPool.n, &first, &last);
      pthread_mutex_unlock(&gpuHostPool.lock);
      if (first < last)
        function(first, last, context);
      pthread_mutex_lock(&gpuHostPool.lock);
      if (--gpuHostPool.pending == 0)
        pthread_cond_signal(&gpuHostPool.done);
    }
  }
  return NULL;
}

/* function(first, last, context) over 0 to n in up to `threads` ranges at once, one of them on the
 * calling thread; returns when all are done.  The ranges must not depend on one another. */
static void gpuHostParallelFor(long n, long threads, GPU_HOST_RANGE_FUNCTION function,
                               void *context) {
  long first, last;

  if (n <= 0)
    return;
  if (threads > n)
    threads = n;
  pthread_mutex_lock(&gpuHostPool.lock);
  while (gpuHostPool.workers < threads - 1) {
    pthread_t thread;

    if (pthread_create(&thread, NULL, gpuHostPoolWorker, (void *)(gpuHostPool.workers + 1)) != 0)
      break;
    pthread_detach(thread);
    gpuHostPool.workers++;
  }
  if (threads > gpuHostPool.workers + 1)
    threads = gpuHostPool.workers + 1;
  if (threads <= 1) {
    pthread_mutex_unlock(&gpuHostPool.lock);
    function(0, n, context);
    return;
  }
  gpuHostPool.function = function;
  gpuHostPool.context = context;
  gpuHostPool.n = n;
  gpuHostPool.parts = threads;
  gpuHostPool.pending = threads - 1;
  gpuHostPool.generation++;
  pthread_cond_broadcast(&gpuHostPool.wake);
  pthread_mutex_unlock(&gpuHostPool.lock);
  gpuHostPoolRange(0, threads, n, &first, &last);
  if (first < last)
    function(first, last, context);
  pthread_mutex_lock(&gpuHostPool.lock);
  while (gpuHostPool.pending)
    pthread_cond_wait(&gpuHostPool.done, &gpuHostPool.lock);
  /* a worker that wakes late must find no range of this job left to take */
  gpuHostPool.parts = 0;
  pthread_mutex_unlock(&gpuHostPool.lock);
}

static long gpuIsrHostThreads(void) {
  return gpuHostThreads();
}

/* Kept between calls, for the numbers made on the host */
static double *gpuIsrHostDraws;
static long gpuIsrHostDrawCapacity;

typedef struct {
  double *values;
  unsigned long long start, pairMultiplier;
} GPU_ISR_HOST_FILL;

/* values[first] to values[last-1]: each from its own pair of uniforms, reached by a jump */
static void gpuIsrHostFillRange(long first, long last, void *context) {
  GPU_ISR_HOST_FILL *fill = (GPU_ISR_HOST_FILL *)context;
  unsigned long long state = gpuDlaranJump(fill->start, 2 * (unsigned long long)first);
  long j;

  for (j = first; j < last; j++) {
    fill->values[j] = gpuIsrHostPairGaussian(state);
    state = (state * fill->pairMultiplier) & GPU_DLARAN_MASK;
  }
}

/* gauss_rn_lim(0, 1, limit, ...) count times from random_2's sequence at firstState, into
 * gpuIsrHostDraws.  Threads make one number per pair of uniforms; those beyond the limit are then
 * dropped in order, as gauss_rn_lim() draws again for them.  Returns how many uniforms were used. */
static long gpuIsrHostFill(long count, unsigned long long firstState, double limit) {
  const unsigned long long pairMultiplier =
    (GPU_DLARAN_MULTIPLIER * GPU_DLARAN_MULTIPLIER) & GPU_DLARAN_MASK;
  double accepted = limit > 0 ? erf(limit / M_SQRT2) : 1;
  long threads = gpuIsrHostThreads();
  long made = 0, pairs = 0;
  GPU_ISR_HOST_FILL fill;

  while (made < count) {
    long remaining = count - made, batch, i;
    double wanted = remaining / accepted * 1.01 + 64, *values;
    unsigned long long start = gpuDlaranJump(firstState, 2 * (unsigned long long)pairs);

    if (!(wanted <= 64.0 * remaining + 64))
      wanted = 64.0 * remaining + 64;
    batch = (long)wanted;
    if (gpuIsrHostDrawCapacity < made + batch) {
      double *grown = (double *)realloc(gpuIsrHostDraws, (made + batch) * sizeof(*grown));

      if (!grown)
        gpuRequiredFailure("unable to allocate the ISR random numbers");
      gpuIsrHostDraws = grown;
      gpuIsrHostDrawCapacity = made + batch;
    }
    values = gpuIsrHostDraws + made;
    fill.values = values;
    fill.start = start;
    fill.pairMultiplier = pairMultiplier;
    gpuHostParallelFor(batch, batch >= GPU_HOST_THREAD_MIN_GAUSSIANS ? threads : 1,
                       gpuIsrHostFillRange, &fill);
    /* In place, since a number only moves toward the start */
    for (i = 0; i < batch && made < count; i++) {
      if (limit > 0 && fabs(values[i]) > limit)
        continue;
      gpuIsrHostDraws[made++] = values[i];
    }
    pairs += i;
  }
  return 2 * pairs;
}

#ifdef GPU_VERIFY
static unsigned long long gpuIsrHostState;
static long gpuIsrHostUsed;

/* random_2's sequence from gpuIsrHostState, without moving random_2 itself */
static double gpuIsrHostUniform(long iseed) {
  double value = gpuIsrHostState * GPU_DLARAN_SCALE;

  (void)iseed;
  gpuIsrHostState = (gpuIsrHostState * GPU_DLARAN_MULTIPLIER) & GPU_DLARAN_MASK;
  gpuIsrHostUsed++;
  return value;
}

/* Compares the device's numbers with gauss_rn_lim() on this host's libm, exactly for those made on
 * the host */
static void gpuIsrCheckDraws(const void *draws, long count, unsigned long long firstState, long used) {
  double *device = (double *)malloc(count * sizeof(*device));
  double maxAbs = 0;
  float milliseconds = 0;
  long i;
  int status;

  if (!device)
    gpuRequiredFailure("unable to allocate the ISR random number check");
  status = gpuCudaCopyDeviceToHost(device, draws, count, &milliseconds);
  if (status != 0)
    gpuFatalStatus("cudaMemcpy(ISR random number check)", status);
  gpuIsrHostState = firstState;
  gpuIsrHostUsed = 0;
  for (i = 0; i < count; i++) {
    double absDiff = fabs(device[i] - gauss_rn_lim(0.0, 1.0, gpuIsrSequence.limit, gpuIsrHostUniform));
    if (!(absDiff <= maxAbs))
      maxAbs = absDiff;
  }
  if (gpuIsrHostUsed != used || !(maxAbs <= (gpuIsrSequence.hostMath ? 0 : 1e-13))) {
    fprintf(stderr, "elegant CUDA VERIFY failed for ISR random numbers: CPU used %ld uniforms, GPU %ld, maxAbs=%.3e\n",
            gpuIsrHostUsed, used, maxAbs);
    exit(1);
  }
  if (gpuVerbose)
    fprintf(stderr, "elegant CUDA VERIFY passed for ISR random numbers: maxAbs=%.3e\n", maxAbs);
  free(device);
}
#endif

static const double *gpuIsrFillDraws(long count) {
  const void *draws = NULL;
  unsigned long long firstState = gpuIsrSequence.state;
  long used = 0;
  float milliseconds = 0;
  int status;

  if (gpuIsrSequence.hostMath) {
    used = gpuIsrHostFill(count, firstState, gpuIsrSequence.limit);
    status = gpuCudaIsrUploadDraws(count, gpuIsrHostDraws, &draws, &milliseconds);
    if (status != 0)
      gpuFatalStatus("cudaMemcpy(ISR random numbers)", status);
    gpuRecordMilliseconds(&gpuBase.gpuTransferToDeviceSeconds, milliseconds);
    gpuIsrReferenceFills++;
    gpuIsrReferenceDraws += count;
  } else {
    status = gpuCudaIsrFillDraws(count, firstState, gpuIsrSequence.limit, &draws, &used, &milliseconds);
    if (status != 0)
      gpuFatalStatus("ISR random numbers CUDA kernel", status);
    gpuRecordMagnetKernel(milliseconds);
    gpuRecordKernelSite(__LINE__, 'i', milliseconds);
  }
#ifdef GPU_VERIFY
  if (gpuBase.verifyMode)
    gpuIsrCheckDraws(draws, count, firstState, used);
#endif
  gpuIsrSequence.used += used;
  gpuIsrSequence.state = gpuDlaranJump(gpuIsrSequence.state, used);
  return (const double *)draws;
}

static long gpuSymplecticKicksPerSlice(long integrationOrder) {
  return integrationOrder == 2 ? 1 : (integrationOrder == 4 ? 3 : 7);
}

typedef int (*GPU_BATCH_TRACK)(void *cookie, void *coord, long count, const double *draws,
                               double *records, long *lostCount, float *milliseconds);
typedef void (*GPU_BATCH_CONSUME)(void *cookie, const double *records, long count);

/* Random numbers and output records go through the device in batches of about 256 MB, because those
 * for one call can exceed device memory.  On a loss, *lostCount is set and the coordinates are back
 * as they were. */
static void gpuTrackInBatches(long nParticles, long drawsPerParticle, long recordsPerParticle,
                              GPU_BATCH_TRACK track, GPU_BATCH_CONSUME consume, void *cookie,
                              const char *name, long *lostCount, float *totalMilliseconds) {
  long valuesPerParticle = drawsPerParticle + recordsPerParticle;
  long stride = gpuBase.deviceStride, start, batch;
  double *hostRecords = NULL;
  void *deviceRecords = NULL;
  int status;

  *lostCount = 0;
  *totalMilliseconds = 0;
  batch = (long)(gpuEnvDouble("ELEGANT_GPU_RECORDED_BATCH_BYTES", 256.0 * 1024 * 1024) / sizeof(double));
  batch /= valuesPerParticle > 0 ? valuesPerParticle : 1;
  if (batch < 1)
    batch = 1;
  if (batch > nParticles)
    batch = nParticles;
  if (recordsPerParticle > 0) {
    hostRecords = (double *)malloc(batch * recordsPerParticle * sizeof(*hostRecords));
    if (!hostRecords || gpuCudaMallocDouble(&deviceRecords, batch * recordsPerParticle) != 0)
      gpuRequiredFailure("unable to allocate the tracking output records");
  }
  if (batch < nParticles) {
    gpuEnsureApertureScratch(nParticles);
    status = gpuCudaCopyDeviceToDevice(gpuApertureScratch.coord, gpuBase.deviceCoord, nParticles * stride);
    if (status != 0)
      gpuFatalStatus(name, status);
  }
  for (start = 0; start < nParticles; start += batch) {
    long count = nParticles - start;
    float milliseconds = 0;
    const double *draws = NULL;

    if (count > batch)
      count = batch;
    if (drawsPerParticle > 0)
      draws = gpuIsrFillDraws(count * drawsPerParticle);
    status = track(cookie, (double *)gpuBase.deviceCoord + start * stride, count, draws,
                   (double *)deviceRecords, lostCount, &milliseconds);
    if (status != 0)
      gpuFatalStatus(name, status);
    *totalMilliseconds += milliseconds;
    if (*lostCount) {
      if (start) {
        status = gpuCudaCopyDeviceToDevice(gpuBase.deviceCoord, gpuApertureScratch.coord, start * stride);
        if (status != 0)
          gpuFatalStatus(name, status);
      }
      break;
    }
    if (recordsPerParticle > 0) {
      status = gpuCudaCopyDeviceToHost(hostRecords, deviceRecords, count * recordsPerParticle, &milliseconds);
      if (status != 0)
        gpuFatalStatus(name, status);
      gpuRecordMilliseconds(&gpuBase.gpuTransferToHostSeconds, milliseconds);
      consume(cookie, hostRecords, count);
    }
  }
  free(hostRecords);
  if (deviceRecords)
    gpuCudaFree(deviceRecords);
}

static int gpuSreffectsBatchTrack(void *cookie, void *coord, long count, const double *draws,
                                  double *records, long *lostCount, float *milliseconds) {
  GPU_SREFFECTS_DATA *data = (GPU_SREFFECTS_DATA *)cookie;

  (void)records;
  *lostCount = 0;
  data->draws = draws;
  return gpuCudaSreffectsTrack(coord, count, (int)gpuBase.deviceStride, data, milliseconds);
}

/* The CPU draws three numbers per particle unless lossOnly, even with QEXCITATION=0 */
void gpu_track_sreffects(long nParticles,
                         const GPU_SREFFECTS_DATA *data) {
  GPU_SREFFECTS_DATA batchData;
  float milliseconds = 0;
  long lostCount = 0;
  int status;

  if (nParticles <= 0)
    return;
  if (!data)
    gpuRequiredFailure("NULL SREFFECTS CUDA tracking data");
  startGpuTimer();
  gpuCopyHostToDevice(nParticles);
  batchData = *data;
  if (data->lossOnly) {
    status = gpuCudaSreffectsTrack(gpuBase.deviceCoord, nParticles,
                                   (int)gpuBase.deviceStride, &batchData,
                                   &milliseconds);
    if (status != 0)
      gpuFatalStatus("SREFFECTS tracking CUDA kernel", status);
  } else {
    gpuIsrBegin();
    gpuIsrSequence.limit = data->cutoff;
    gpuIsrSequence.hostMath = gpuIsrReferenceMathEnabled(gpuBase.orderSensitiveOutputNeeded);
    gpuTrackInBatches(nParticles, 3, 0, gpuSreffectsBatchTrack, NULL, &batchData,
                      "SREFFECTS tracking CUDA kernel", &lostCount, &milliseconds);
    gpuIsrCommit();
  }
  gpuRecordHelperKernel(milliseconds);
  gpuMarkDeviceChanged(nParticles);
  gpuRecordWallSeconds();
}

/* With output in CPU order, ILMATRIX takes the sines and cosines of its phase advances from this host's
 * libm, as the CPU does: CUDA's differ from glibc's by a unit in the last place now and then, which a
 * ring turns into a different beam after enough turns.  The device makes the phase advances and does
 * the rest.  ELEGANT_GPU_ILMATRIX_REFERENCE_MATH=1 or 0 sets it for every run. */
static long gpuIlmatrixReferenceMathEnabled(void) {
  if (gpuEnvSet("ELEGANT_GPU_ILMATRIX_REFERENCE_MATH"))
    return gpuEnvFlag("ELEGANT_GPU_ILMATRIX_REFERENCE_MATH");
  return gpuBase.orderSensitiveOutputNeeded;
}

/* do_tracking.c calls sin() and cos() one after the other; kept apart here too, so that the compiler
 * can't make one sincos() call of them */
static __attribute__((noinline)) double gpuIlmatrixHostSin(double x) { return sin(x); }
static __attribute__((noinline)) double gpuIlmatrixHostCos(double x) { return cos(x); }

static void gpuIlmatrixReferenceRange(long first, long last, void *context) {
  long i;

  (void)context;
  for (i = first; i < last; i++) {
    gpuIlmatrixScratch.hostSinCos[2 * i] = gpuIlmatrixHostSin(gpuIlmatrixScratch.hostPhase[i]);
    gpuIlmatrixScratch.hostSinCos[2 * i + 1] = gpuIlmatrixHostCos(gpuIlmatrixScratch.hostPhase[i]);
  }
}

static void gpuEnsureIlmatrixScratch(long nParticles) {
  int status;

  if (nParticles <= gpuIlmatrixScratch.capacity)
    return;
  gpuReleaseIlmatrixScratch();
  gpuIlmatrixScratch.hostPhase = tmalloc(sizeof(double) * 2 * nParticles);
  gpuIlmatrixScratch.hostSinCos = tmalloc(sizeof(double) * 4 * nParticles);
  status = gpuCudaMallocBytes(&gpuIlmatrixScratch.phase, 6 * sizeof(double) * nParticles);
  if (status != 0)
    gpuFatalStatus("cudaMalloc(ILMATRIX phase scratch)", status);
  gpuIlmatrixScratch.capacity = nParticles;
}

/* The device's array of the host's sines and cosines, or NULL when the kernel takes its own */
static void *gpuIlmatrixReferenceSinCos(long np, const GPU_ILMATRIX_DATA *data) {
  void *deviceSinCos;
  float milliseconds = 0;
  int status;

  if (!gpuIlmatrixReferenceMathEnabled())
    return NULL;
  gpuEnsureIlmatrixScratch(np);
  deviceSinCos = (double *)gpuIlmatrixScratch.phase + 2 * np;
  status = gpuCudaIlmatrixPhases(gpuBase.deviceCoord, gpuIlmatrixScratch.phase, np,
                                 (int)gpuBase.deviceStride, data, &milliseconds);
  if (status != 0)
    gpuFatalStatus("ILMATRIX phase kernel", status);
  gpuRecordHelperKernel(milliseconds);
  status = gpuCudaCopyDeviceToHost(gpuIlmatrixScratch.hostPhase, gpuIlmatrixScratch.phase, 2 * np,
                                   &milliseconds);
  if (status != 0)
    gpuFatalStatus("ILMATRIX phase transfer", status);
  gpuRecordMilliseconds(&gpuBase.gpuTransferToHostSeconds, milliseconds);
  /* Each number depends on its own phase alone, so any number of host threads gives the same */
  gpuHostParallelFor(2 * np, 2 * np >= GPU_HOST_THREAD_MIN_SINES ? gpuHostThreads() : 1,
                     gpuIlmatrixReferenceRange, NULL);
  status = gpuCudaCopyHostToDevice(deviceSinCos, gpuIlmatrixScratch.hostSinCos, 4 * np, &milliseconds);
  if (status != 0)
    gpuFatalStatus("ILMATRIX sine and cosine transfer", status);
  gpuRecordMilliseconds(&gpuBase.gpuTransferToDeviceSeconds, milliseconds);
  gpuIlmatrixReferenceCalls++;
  return deviceSinCos;
}

/* Returns the number of particles left.  The lost ones follow them in the order the CPU's swaps leave. */
long gpu_track_ilmatrix(long nParticles, double **accepted, const GPU_ILMATRIX_DATA *data) {
  long remaining = nParticles, divideByZero = 0;
  float milliseconds = 0;
  void *sinCos;
  int status;

  if (nParticles <= 0)
    return nParticles;
  if (!data)
    gpuRequiredFailure("NULL ILMATRIX CUDA tracking data");
  startGpuTimer();
  gpuCopyHostToDevice(nParticles);
  gpuEnsureApertureScratch(nParticles);
  sinCos = gpuIlmatrixReferenceSinCos(nParticles, data);
  status = gpuCudaIlmatrixTrack(gpuBase.deviceCoord, gpuApertureScratch.coord, gpuApertureScratch.prefix,
                                nParticles, (int)gpuBase.deviceStride, data, sinCos,
                                &remaining, &divideByZero, &milliseconds);
  if (status != 0)
    gpuFatalStatus("ILMATRIX tracking CUDA kernel", status);
  gpuRecordHelperKernel(milliseconds);
  if (divideByZero)
    bombElegant("divided by zero in trackWithChromaticLinearMatrix", NULL);
  if (remaining < nParticles) {
    accepted = gpuApertureHostAccepted(accepted);
    if (accepted) {
      if (!gpuStablePartitionAcceptedOnDevice(accepted, nParticles, remaining,
                                              "ILMATRIX CUDA accepted compaction")) {
        gpuCopyAperturePrefixToHost(nParticles, "ILMATRIX CUDA compaction accepted map");
        gpuStablePartitionAccepted(accepted, nParticles, remaining);
      }
    }
    gpuPromoteApertureScratchCoord();
    gpuCopyApertureLossRowsToHost(remaining, nParticles, "ILMATRIX CUDA loss tail output");
  }
  gpuMarkDeviceChanged(remaining);
  gpuRecordWallSeconds();
  return remaining;
}

static int gpuCwigglerBatchTrack(void *cookie, void *coord, long count, const double *draws,
                                 double *records, long *lostCount, float *milliseconds) {
  GPU_CWIGGLER_DATA *data = (GPU_CWIGGLER_DATA *)cookie;

  *lostCount = 0;
  data->isrDraws = draws;
  data->fieldRecord = records;
  return gpuCudaCwigglerTrack(coord, count, (int)gpuBase.deviceStride, data, milliseconds);
}

typedef struct {
  GPU_CWIGGLER_DATA data; /* first, for gpuCwigglerBatchTrack */
  GPU_CWIGGLER_FIELD_CONSUMER consumer;
  void *cookie;
  long rows;
} GPU_CWIGGLER_BATCH;

static void gpuCwigglerBatchConsume(void *cookie, const double *records, long count) {
  GPU_CWIGGLER_BATCH *batch = (GPU_CWIGGLER_BATCH *)cookie;

  batch->consumer(batch->cookie, records, count, batch->rows);
}

void gpu_track_cwiggler(long nParticles, const GPU_CWIGGLER_DATA *data,
                        GPU_CWIGGLER_FIELD_CONSUMER fieldConsumer, void *cookie) {
  float milliseconds = 0;
  int status;

  if (nParticles <= 0)
    return;
  if (!data)
    gpuRequiredFailure("NULL CWIGGLER CUDA tracking data");
  startGpuTimer();
  gpuCopyHostToDevice(nParticles);
  if (data->isrCoef || fieldConsumer) {
    GPU_CWIGGLER_BATCH batch;
    long lostCount = 0;

    batch.data = *data;
    batch.consumer = fieldConsumer;
    batch.cookie = cookie;
    batch.rows = data->periods * data->stepsPerPeriod + 1;
    if (data->isrCoef)
      gpuIsrBegin();
    gpuTrackInBatches(nParticles, data->isrCoef ? data->isrDrawsPerParticle : 0,
                      fieldConsumer ? batch.rows * GPU_CWIGGLER_FIELD_RECORD_SIZE : 0,
                      gpuCwigglerBatchTrack, gpuCwigglerBatchConsume, &batch,
                      "CWIGGLER tracking CUDA kernel", &lostCount, &milliseconds);
    if (data->isrCoef)
      gpuIsrCommit();
  } else {
    status = gpuCudaCwigglerTrack(gpuBase.deviceCoord, nParticles,
                                  (int)gpuBase.deviceStride, data,
                                  &milliseconds);
    if (status != 0)
      gpuFatalStatus("CWIGGLER tracking CUDA kernel", status);
  }
  gpuRecordMagnetKernel(milliseconds);
  gpuMarkDeviceChanged(nParticles);
  gpuRecordWallSeconds();
}

static long gpuPortRotation(double angle, double *cosA, double *sinA);

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
  /* field_table_tracking()'s frame changes, with the sines and cosines its routines use */
  if (ftable->e1 || ftable->l1) {
    data.entrance = ftable->e1 ? 2 : 1;
    data.l1 = ftable->l1;
    if (ftable->e1) {
      data.sinE1 = sin(ftable->e1);
      data.cosE1 = cos(ftable->e1);
    }
  }
  if (ftable->e2 || ftable->l2) {
    data.exit = ftable->e2 ? 2 : 1;
    data.l2 = ftable->l2;
    if (ftable->e2) {
      data.sinE2 = sin(ftable->e2);
      data.cosE2 = cos(ftable->e2);
      data.dxExit = ftable->l2 * data.sinE2;
    }
  }
  data.doOffset = ftable->dx || ftable->dy || ftable->dz;
  data.dx = ftable->dx;
  data.dy = ftable->dy;
  data.dz = ftable->dz;
  data.doTilt = ftable->tilt && gpuPortRotation(ftable->tilt, &data.cosTilt, &data.sinTilt) &&
                gpuPortRotation(-ftable->tilt, &data.cosUntilt, &data.sinUntilt);

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

/* Returns the number of particles kept, or -1 to leave the element to the CPU.  Apertures are
 * checked as lorentz() checks them; global loss coordinates stay on the CPU, which leaves only an
 * obstruction's there */
long gpu_track_bmxyz(long nParticles, BMAPXYZ *bmxyz, double pCentral, MAXAMP *maxamp,
                     APCONTOUR *apcontour, APERTURE_DATA *apFileData, double zStart,
                     double **accepted, ELEMENT_LIST *eptr) {
  GPU_BMXYZ_DATA data;
  BMAPXYZ_DATA *map;
  float milliseconds = 0;
  int status;
  long failedCount = 0, remaining = nParticles;

  if (nParticles <= 0)
    return nParticles;
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
  if (maxamp || apcontour || (apFileData && apFileData->initialized)) {
    MULT_APERTURE_DATA apertureData;

    if (!setupMultApertureData || globalLossCoordOffset > 0)
      return -1;
    setupMultApertureData(&apertureData, 0.0, apcontour, maxamp, apFileData, NULL, zStart, eptr);
    if (!gpuPackMultApertureData(&data.aperture, &data.contour, &apertureData))
      return -1;
    data.apertures = data.aperture.present || data.contour.nContours;
    data.lossZStart = zStart;
    data.pCentral = pCentral;
  }

  startGpuTimer();
  gpuCopyHostToDevice(nParticles);
  if (data.apertures)
    gpuEnsureApertureScratch(nParticles);
  status = gpuCudaBmxyzTrack(gpuBase.deviceCoord, nParticles,
                             (int)gpuBase.deviceStride, &data,
                             data.apertures ? gpuApertureScratch.coord : NULL,
                             data.apertures ? gpuApertureScratch.prefix : NULL, &remaining,
                             &failedCount, &milliseconds);
  if (status != 0)
    gpuFatalStatus("BMXYZ fixed-step tracking CUDA kernel", status);
  gpuRecordMagnetKernel(milliseconds);
  if (failedCount) {
    gpuRecordWallSeconds();
    return -1;
  }
  if (remaining < nParticles) {
    accepted = gpuApertureHostAccepted(accepted);
    if (accepted &&
        !gpuStablePartitionAcceptedOnDevice(accepted, nParticles, remaining,
                                            "BMXYZ CUDA accepted compaction")) {
      gpuCopyAperturePrefixToHost(nParticles, "BMXYZ CUDA compaction accepted map");
      gpuStablePartitionAccepted(accepted, nParticles, remaining);
    }
    gpuPromoteApertureScratchCoord();
    gpuCopyApertureLossRowsToHost(remaining, nParticles, "BMXYZ CUDA compaction loss tail output");
  }
  gpuMarkDeviceChanged(remaining);
  gpuRecordWallSeconds();
  return remaining;
}

long gpu_track_lorentz(long nParticles, void *field, long fieldType,
                       double pCentral) {
  GPU_LORENTZ_DATA data;
  float milliseconds = 0;
  int status;
  long failedCount = 0;

  if (nParticles <= 0)
    return 1;
  if (!field)
    gpuRequiredFailure("NULL Lorentz-family CUDA tracking data");
  memset(&data, 0, sizeof(data));
  data.length = 0;
  switch (fieldType) {
  case T_BMAPXY: {
    BMAPXY *bmapxy = (BMAPXY *)field;
    data.type = GPU_LORENTZ_BMAPXY;
    data.tableOwner = bmapxy->Fx;
    data.nx = bmapxy->nx;
    data.ny = bmapxy->ny;
    data.field[0] = bmapxy->Fx;
    data.field[1] = bmapxy->Fy;
    data.xmin = bmapxy->xmin;
    data.ymin = bmapxy->ymin;
    data.dx = bmapxy->dx;
    data.dy = bmapxy->dy;
    data.length = bmapxy->length;
    data.integrationAccuracy = bmapxy->accuracy;
    data.strengthFactor = bmapxy->strength;
    data.fieldIsMagnetic = bmapxy->BGiven ? 1 : 0;
    data.fieldScale = -particleCharge * particleRelSign /
                      (particleMass * c_mks * pCentral);
    break;
  }
  case T_NIBEND: {
    NIBEND *nibend = (NIBEND *)field;
    double halfAngle, alpha1, alpha2;
    data.type = GPU_LORENTZ_NIBEND;
    data.length = nibend->length;
    data.integrationAccuracy = nibend->accuracy;
    data.angle = nibend->angle;
    data.e1 = nibend->e[nibend->e1Index];
    data.e2 = nibend->e[nibend->e2Index];
    data.rho = nibend->length / nibend->angle;
    data.strengthFactor = (1 + nibend->fse) / data.rho;
    halfAngle = data.angle / 2;
    alpha1 = halfAngle - data.e1;
    alpha2 = halfAngle - data.e2;
    data.entranceIntercept = data.fringeEntranceIntercept =
      -data.rho * (sin(halfAngle) - cos(halfAngle) * tan(alpha1));
    data.exitIntercept = data.fringeExitIntercept =
      data.rho * (sin(halfAngle) - cos(halfAngle) * tan(alpha2));
    data.entranceSlope = -tan(alpha1);
    data.exitSlope = tan(alpha2);
    data.cosAlpha1 = cos(alpha1);
    data.sinAlpha1 = sin(alpha1);
    data.cosAlpha2 = cos(alpha2);
    data.sinAlpha2 = sin(alpha2);
    break;
  }
  case T_NISEPT: {
    NISEPT *nisept = (NISEPT *)field;
    lorentz_setup(nisept, T_NISEPT, NULL, 0, pCentral);
    data.type = GPU_LORENTZ_NISEPT;
    data.length = nisept->length;
    data.integrationAccuracy = nisept->accuracy;
    data.angle = nisept->angle;
    data.e1 = nisept->e1;
    data.e2 = nisept->e2;
    data.rho = nisept->rho0;
    data.b1 = nisept->b1;
    data.q1Reference = nisept->q1_ref;
    data.q1Offset = nisept->q1_offset;
    data.fringeLength = nisept->flen;
    data.strengthFactor = (1 + nisept->fse_opt) / data.rho;
    data.entranceIntercept = -data.rho * sin(data.e1) - data.fringeLength / 2;
    data.fringeEntranceIntercept = data.entranceIntercept + data.fringeLength;
    data.exitIntercept = data.rho * sin(data.e2) + data.fringeLength / 2;
    data.fringeExitIntercept = data.exitIntercept - data.fringeLength;
    break;
  }
  default:
    gpuRequiredFailure("invalid Lorentz-family CUDA element type");
    return 0;
  }

  startGpuTimer();
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaLorentzTrack(gpuBase.deviceCoord, nParticles,
                               (int)gpuBase.deviceStride, &data,
                               &failedCount, &milliseconds);
  if (status != 0)
    gpuFatalStatus("fixed-step Lorentz-family CUDA kernel", status);
  gpuRecordMagnetKernel(milliseconds);
  if (!failedCount)
    gpuMarkDeviceChanged(nParticles);
  gpuRecordWallSeconds();
  return failedCount == 0;
}

/* LSRMDLTR with ISR.  motion() draws one number a Runge-Kutta step, particle after particle, and a
 * particle's steps depend on its path, so where its numbers start in the sequence depends on every
 * particle before it.  The counts hardly depend on the numbers themselves, so each particle is given
 * an offset from a guessed count, the counts it reports give the offsets the CPU would have used, and
 * the particles from the first one whose offset was wrong are tracked again until all agree (one
 * pass when the guess is right).  random_2 then moves by exactly the numbers used.  The table of
 * numbers is filled for a batch of particles at a time.  Returns the number of passes, or 0 when the
 * CPU must track (a loss, or offsets that would not settle). */
static long gpuLsrmdltrGuessSteps = -1, gpuLsrmdltrGuess = 0;

static long gpuLsrmdltrIsrTrack(GPU_LSRMDLTR_DATA *data, long nParticles, double *xRange0,
                                float *totalMilliseconds) {
  static long long *offsets = NULL;
  static int *counts = NULL;
  static long capacity = 0;
  long stride = gpuBase.deviceStride, start, batch, guess, passes = 0;
  double bytes = gpuEnvDouble("ELEGANT_GPU_LSRMDLTR_DRAW_BYTES", 1024.0 * 1024 * 1024);

  guess = gpuLsrmdltrGuessSteps == data->nSteps && gpuLsrmdltrGuess > 0 ?
            gpuLsrmdltrGuess : (long)(data->nSteps / 1.5) + 1;
  *totalMilliseconds = 0;
  gpuIsrBegin();
  /* motion.cc: gauss_rn_lim(0.0, 1.0, 3.0, random_2) */
  gpuIsrSequence.limit = 3.0;
  for (start = 0; start < nParticles; start += batch) {
    unsigned long long batchState = gpuIsrSequence.state;
    long batchUsed = gpuIsrSequence.used, first = 0, i, pass;
    long long drawCount, total = 0;

    batch = (long)(bytes / sizeof(double) / (guess + 2));
    if (batch < 1)
      batch = 1;
    if (batch > nParticles - start)
      batch = nParticles - start;
    if (capacity < batch) {
      free(offsets);
      free(counts);
      offsets = (long long *)malloc(batch * sizeof(*offsets));
      counts = (int *)malloc(batch * sizeof(*counts));
      if (!offsets || !counts)
        gpuRequiredFailure("unable to allocate the LSRMDLTR ISR offsets");
      capacity = batch;
    }
    for (i = 0; i < batch; i++)
      offsets[i] = (long long)i * guess;
    drawCount = (long long)batch * (guess + 2) + 64;
    data->draws = gpuIsrFillDraws(drawCount);
    data->drawCount = drawCount;
    for (pass = 0;; pass++) {
      long failedCount = 0, mismatch = -1, refill = 0;
      long long p;
      float milliseconds = 0;
      int status;

      if (pass) {
        status = gpuCudaLsrmdltrRestore(gpuBase.deviceCoord, (start + first) * stride, (batch - first) * stride);
        if (status != 0)
          gpuFatalStatus("LSRMDLTR coordinate restore", status);
      }
      status = gpuCudaLsrmdltrPass((double *)gpuBase.deviceCoord + (start + first) * stride, batch - first,
                                   (int)stride, data, offsets + first, counts + first,
                                   start == 0 && first == 0 ? xRange0 : NULL, &failedCount, &milliseconds);
      if (status != 0)
        gpuFatalStatus("LSRMDLTR CUDA kernel", status);
      *totalMilliseconds += milliseconds;
      passes++;
      if (failedCount || pass >= 20) {
        gpuIsrAbandon();
        return 0;
      }
      /* the CPU's offsets from the counts; those up to the first wrong one are right */
      p = offsets[first];
      for (i = first; i < batch; i++) {
        if (mismatch < 0 && offsets[i] != p)
          mismatch = i;
        if (mismatch < 0 && counts[i] < 0) {
          /* this particle's offset is right but the table ran out */
          mismatch = i;
          refill = 1;
        }
        if (mismatch >= 0)
          offsets[i] = p;
        p += counts[i] >= 0 ? counts[i] : guess;
      }
      if (mismatch < 0) {
        total = p;
        break;
      }
      if (refill || p + 64 > drawCount) {
        drawCount = p + (long long)(batch - mismatch) * 2 + 64;
        if (refill)
          drawCount += (long long)(batch - mismatch) * guess;
        gpuIsrSequence.state = batchState;
        gpuIsrSequence.used = batchUsed;
        data->draws = gpuIsrFillDraws(drawCount);
        data->drawCount = drawCount;
      }
      first = mismatch;
    }
    /* random_2 moves by exactly the numbers this batch used */
    gpuIsrSequence.state = batchState;
    gpuIsrSequence.used = batchUsed;
    if (total > 0)
      (void)gpuIsrFillDraws(total);
    guess = (long)((total + batch / 2) / batch);
  }
  if (gpuVerbose)
    fprintf(stderr, "elegant CUDA: LSRMDLTR ISR offsets settled in %ld passes over %ld particles\n",
            passes, nParticles);
  gpuLsrmdltrGuessSteps = data->nSteps;
  gpuLsrmdltrGuess = guess;
  data->draws = NULL;
  data->drawCount = 0;
  gpuIsrCommit();
  return passes;
}

long gpu_track_lsrmdltr(long nParticles, void *field, double pCentral) {
  static long xMotionCenterMemory = -1;
  LSRMDLTR *lsr = (LSRMDLTR *)field;
  GPU_LSRMDLTR_DATA data;
  double xRange0[2] = {1e300, -1e300}, kscale;
  float milliseconds = 0;
  long failedCount = 0;
  int status;

  if (nParticles <= 0)
    return 1;
  if (!lsr)
    gpuRequiredFailure("NULL LSRMDLTR CUDA tracking data");
  if (!setupLaserModulator || !findFiducialTime)
    return 0;
  /* set_up_derivatives()'s LSRMDLTR case: the same function, then t0 from the CPU's sum */
  setupLaserModulator(lsr, pCentral, 1);
  kscale = lsr->k;
  lsr->t0 = gpu_findFiducialTime(nParticles, 0, lsr->length / 2, pCentral, FID_MODE_TMEAN) +
            lsr->timeProfileOffset;
  memset(&data, 0, sizeof(data));
  data.length = lsr->length;
  data.Bu = lsr->Bu;
  data.tguGradient = lsr->tguGradient;
  data.tguCompFactor = lsr->tguCompFactor;
  data.ku = lsr->ku;
  data.k = kscale;
  data.omega = lsr->omega;
  data.Escale = lsr->Escale;
  data.Bscale = lsr->Bscale;
  data.Ef0Laser = lsr->Ef0Laser;
  data.laserW0 = lsr->laserW0;
  data.ZRayleigh = lsr->ZRayleigh;
  data.laserPhase = lsr->laserPhase;
  data.laserX0 = lsr->laserX0;
  data.laserY0 = lsr->laserY0;
  data.laserZ0 = lsr->laserZ0;
  data.laserTilt = lsr->laserTilt;
  data.poleFactor1 = lsr->poleFactor1;
  data.poleFactor2 = lsr->poleFactor2;
  data.poleFactor3 = lsr->poleFactor3;
  data.t0 = lsr->t0;
  data.Zend = lsr->length * kscale;
  data.Zcenter = lsr->length / 2 * kscale;
  data.tolerance = lsr->accuracy;
  /* motion()'s globals */
  data.pCentral = pCentral;
  data.gammaCentral = sqrt(sqr(pCentral) + 1);
  data.radCoef = sqr(particleCharge) * pow3(pCentral) / (6 * PI * epsilon_o * sqr(c_mks) * particleMass);
  data.cMks = c_mks;
  data.eMks = e_mks;
  data.meMks = me_mks;
  data.periods = lsr->periods;
  data.nSteps = lsr->nSteps;
  /* rk_odeint3_na() has no limit; this one only keeps a pathological particle from holding the device */
  data.maxLoops = 4 * lsr->nSteps + 1000;
  data.laserM = lsr->laserM;
  data.laserN = lsr->laserN;
  data.helical = lsr->helical;
  data.fieldCode = lsr->fieldCode;
  data.synchRad = lsr->synchRad;
  data.laserOn = lsr->Ef0Laser > 0 && lsr->laserW0 > 0;
  data.isr = lsr->isr != 0;
  data.isrCoef = particleRadius * sqrt(55.0 / (24 * sqrt(3.0)) * pow5(pCentral) * 137.0359895);
  if (lsr->timeValue) {
    data.timeValue = lsr->timeValue;
    data.amplitudeValue = lsr->amplitudeValue;
    data.tProfilePoints = lsr->tProfilePoints;
  }

  startGpuTimer();
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaLsrmdltrBackup(gpuBase.deviceCoord, nParticles * gpuBase.deviceStride);
  if (status != 0)
    gpuFatalStatus("LSRMDLTR coordinate backup", status);
  if (data.isr)
    failedCount = !gpuLsrmdltrIsrTrack(&data, nParticles, xRange0, &milliseconds);
  else {
    status = gpuCudaLsrmdltrPass(gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride, &data,
                                 NULL, NULL, xRange0, &failedCount, &milliseconds);
    if (status != 0)
      gpuFatalStatus("LSRMDLTR CUDA kernel", status);
  }
  if (failedCount) {
    status = gpuCudaLsrmdltrRestore(gpuBase.deviceCoord, 0, nParticles * gpuBase.deviceStride);
    if (status != 0)
      gpuFatalStatus("LSRMDLTR coordinate restore", status);
  }
  gpuRecordMagnetKernel(milliseconds);
  if (!failedCount) {
    gpuMarkDeviceChanged(nParticles);
    /* motion() stores the middle of the first particle's x excursion */
    if (xMotionCenterMemory < 0)
      xMotionCenterMemory = rpn_create_mem((char *)"xMotionCenter", 0);
    rpn_store((xRange0[1] + xRange0[0]) / 2, NULL, xMotionCenterMemory);
  }
  gpuRecordWallSeconds();
  return failedCount == 0;
}

/* motion()'s tables for the device, back to back; returns the offset + 1 of the array, which
 * gpuCudaMotionPass() turns into the device address */
static double *gpuMotionTableAppend(double **tables, long *capacity, long *values, const double *source,
                                    long count) {
  long offset = *values;

  if (!source || count <= 0)
    return NULL;
  if (*values + count > *capacity) {
    long newCapacity = 2 * (*values + count);
    double *memory = (double *)realloc(*tables, newCapacity * sizeof(**tables));
    if (!memory)
      gpuRequiredFailure("unable to allocate the motion() field tables");
    *tables = memory;
    *capacity = newCapacity;
  }
  memcpy(*tables + offset, source, count * sizeof(*source));
  *values += count;
  return (double *)(size_t)(offset + 1);
}

/* gpu_track_motion()'s way back to the CPU: the element as it was before motion()'s set-up */
static long gpuMotionCpuAfterAll(void *field, const void *elementBefore, long size) {
  memcpy(field, elementBefore, size);
  return -1;
}

/* RFTMEZ0, TWLA, MAPSOLENOID, TMCF, CEPL, TWPL and TWMTA on the device as motion() tracks them.  motionGpuSetup() makes motion()'s
 * set-up (fiducial selection from the host's copy of the beam when it is due); lost particles' rows hold
 * motion()'s records and are put in its order on the host.  Returns the particles left, or -1 when the CPU
 * must track (the particles are then those before the element).  *changeP0 says whether motion() would
 * match the central momentum afterwards. */
long gpu_track_motion(double **coord, double **accepted, long nParticles, void *field, long type,
                      double *pCentral, double zStart, long *changeP0) {
  static long xMotionCenterMemory = -1;
  static unsigned char *lost = NULL;
  static long lostCapacity = 0;
  static double *tables = NULL;
  static long tableCapacity = 0;
  static void *elementBefore = NULL;
  static long elementCapacity = 0;
  MOTION_GPU_SETUP setup;
  GPU_MOTION_DATA data;
  long size;
  double **part = coord;
  long tableValues = 0, lostCount = 0, cpuCount = 0, needFiducial = 0, nLeft, i;
  float milliseconds = 0;
  int status;

  *changeP0 = 0;
  if (nParticles <= 0)
    return nParticles;
  if (!field || !motionGpuSetup || coord != gpuBase.coord)
    return -1;
  if (type == T_RFTMEZ0)
    needFiducial = !((RFTMEZ0 *)field)->fiducial_part;
  else if (type == T_TWLA)
    needFiducial = !((TW_LINAC *)field)->fiducial_part;
  else if (type == T_TMCF)
    needFiducial = !((TMCF_MODE *)field)->fiducial_part;
  else if (type == T_CEPL)
    needFiducial = !((CE_PLATES *)field)->fiducial_part;
  else if (type == T_TWPL)
    needFiducial = !((TW_PLATES *)field)->fiducial_part;
  else if (type == T_TWMTA)
    needFiducial = !((TWMTA *)field)->fiducial_part;
  startGpuTimer();
  /* motion()'s set-up keeps state in the element (the fiducial particle, and TWLA's ponderomotive factor from
   * that particle's energy on the call that selects it); if the CPU must track after all, its motion() starts
   * from the element as it was */
  size = entity_description[type].structure_size;
  if (elementCapacity < size) {
    free(elementBefore);
    if (!(elementBefore = malloc(size)))
      gpuRequiredFailure("unable to allocate the motion() element copy");
    elementCapacity = size;
  }
  memcpy(elementBefore, field, size);
  if (needFiducial)
    part = copyParticlesToCpuReadOnly("motion() fiducial particle");
  if (!motionGpuSetup(part, nParticles, field, type, *pCentral, zStart, &setup) || setup.integratorCode != 2) {
    memcpy(field, elementBefore, size);
    gpuRecordWallSeconds();
    return -1;
  }
  memset(&data, 0, sizeof(data));
  data.kscale = setup.kscale;
  data.Zend = setup.Zend;
  data.tauStart = setup.tauStart;
  data.Xlimit = setup.Xlimit;
  data.Ylimit = setup.Ylimit;
  data.tolerance = setup.tolerance;
  data.pCentral = *pCentral;
  data.zStart = zStart;
  data.XapertureCenter = setup.XapertureCenter;
  data.YapertureCenter = setup.YapertureCenter;
  data.Xoffset = setup.Xoffset;
  data.Yoffset = setup.Yoffset;
  data.Xcenter = setup.Xcenter;
  data.Ycenter = setup.Ycenter;
  data.Zcenter = setup.Zcenter;
  data.Xcenter1 = setup.Xcenter1;
  data.Ycenter1 = setup.Ycenter1;
  data.Zcenter1 = setup.Zcenter1;
  memcpy(data.partRot, setup.partRot, sizeof(data.partRot));
  memcpy(data.fieldRot, setup.fieldRot, sizeof(data.fieldRot));
  memcpy(data.partRot1, setup.partRot1, sizeof(data.partRot1));
  memcpy(data.fieldRot1, setup.fieldRot1, sizeof(data.fieldRot1));
  data.nSteps = setup.nSteps;
  /* rk_odeint3_na() has no limit; this one only keeps a pathological particle from holding the device */
  data.maxLoops = 4 * setup.nSteps + 1000;
  data.radialLimit = setup.radialLimit ? 1 : 0;
  switch (type) {
  case T_RFTMEZ0: {
    RFTMEZ0 *rf = (RFTMEZ0 *)field;
    data.type = GPU_MOTION_RFTMEZ0;
    if (!rf->Ez || !rf->dEzdZ || rf->nz < 2 || !(rf->dZ > 0))
      return gpuMotionCpuAfterAll(field, elementBefore, size);
    data.dZ = rf->dZ;
    data.EzPeak = rf->Ez_peak;
    data.nz = rf->nz;
    data.radialOrder = rf->radial_order ? 1 : 0;
    data.BxStrayScaled = rf->BxStrayScaled;
    data.ByStrayScaled = rf->ByStrayScaled;
    data.Ez = gpuMotionTableAppend(&tables, &tableCapacity, &tableValues, rf->Ez, rf->nz);
    data.dEzdZ = gpuMotionTableAppend(&tables, &tableCapacity, &tableValues, rf->dEzdZ, rf->nz);
    if (rf->BzSol) {
      long ir;
      if (rf->nzSol < 2 || rf->nrSol < 1 || (rf->nrSol == 1 && !rf->dBzdZSol) || (rf->nrSol > 1 && (!rf->BrSol || rf->nrSol < 2)))
        return gpuMotionCpuAfterAll(field, elementBefore, size);
      data.hasSolenoid = 1;
      data.Z0Sol = rf->Z0Sol;
      data.dZSol = rf->dZSol;
      data.dRSol = rf->dRSol;
      data.nzSol = rf->nzSol;
      data.nrSol = rf->nrSol;
      data.solenoidFactor = rf->solenoidFactor;
      for (ir = 0; ir < rf->nrSol; ir++) {
        double *offset = gpuMotionTableAppend(&tables, &tableCapacity, &tableValues, rf->BzSol[ir], rf->nzSol);
        if (ir == 0)
          data.BzSol = offset;
      }
      if (rf->nrSol > 1)
        for (ir = 0; ir < rf->nrSol; ir++) {
          double *offset = gpuMotionTableAppend(&tables, &tableCapacity, &tableValues, rf->BrSol[ir], rf->nzSol);
          if (ir == 0)
            data.BrSol = offset;
        }
      else
        data.dBzdZSol = gpuMotionTableAppend(&tables, &tableCapacity, &tableValues, rf->dBzdZSol, rf->nzSol);
    }
    break;
  }
  case T_TWLA: {
    TW_LINAC *twla = (TW_LINAC *)field;
    data.type = GPU_MOTION_TWLA;
    data.alphaS = twla->alphaS;
    data.EzS = twla->EzS;
    data.betaWave = twla->beta_wave;
    data.ErS = twla->ErS;
    data.BphiS = twla->BphiS;
    data.BsolS = twla->BsolS;
    data.FrP = twla->FrP;
    break;
  }
  case T_TMCF: {
    TMCF_MODE *tmcf = (TMCF_MODE *)field;
    data.type = GPU_MOTION_TMCF;
    data.tmcfEr = tmcf->Er;
    data.tmcfEz = tmcf->Ez;
    data.tmcfBphi = tmcf->Bphi;
    break;
  }
  case T_CEPL: {
    CE_PLATES *cep = (CE_PLATES *)field;
    data.type = GPU_MOTION_CEPL;
    data.Escaled = cep->E_scaled;
    data.Estatic = cep->E_static;
    data.cosTilt = cep->cos_tilt;
    data.sinTilt = cep->sin_tilt;
    break;
  }
  case T_TWPL: {
    TW_PLATES *twp = (TW_PLATES *)field;
    data.type = GPU_MOTION_TWPL;
    data.Escaled = twp->E_scaled;
    data.Estatic = twp->E_static;
    data.cosTilt = twp->cos_tilt;
    data.sinTilt = twp->sin_tilt;
    break;
  }
  case T_TWMTA: {
    TWMTA *twmta = (TWMTA *)field;
    data.type = GPU_MOTION_TWMTA;
    data.betaWave = twmta->beta_wave;
    data.Kx = twmta->Kx;
    data.Ky = twmta->Ky;
    data.ExS = twmta->ExS;
    data.EyS = twmta->EyS;
    data.EzS = twmta->EzS;
    data.BxS = twmta->BxS;
    data.ByS = twmta->ByS;
    data.BsolS = twmta->BsolS;
    data.alphaS = twmta->alphaS;
    break;
  }
  case T_MAPSOLENOID: {
    MAP_SOLENOID *sol = (MAP_SOLENOID *)field;
    long ir;
    data.type = GPU_MOTION_MAPSOLENOID;
    if (!sol->Bz || !sol->Br || sol->nz < 2 || sol->nr < 2 || !(sol->dz > 0) || !(sol->dr > 0))
      return gpuMotionCpuAfterAll(field, elementBefore, size);
    data.dzMap = sol->dz;
    data.drMap = sol->dr;
    data.factor = sol->factor;
    data.zMap0 = sol->zMap0;
    data.zMap1 = sol->zMap1;
    data.BxUniformScaled = sol->BxUniformScaled;
    data.ByUniformScaled = sol->ByUniformScaled;
    data.hasUniform = sol->BxUniform || sol->ByUniform;
    data.nzMap = sol->nz;
    data.nrMap = sol->nr;
    for (ir = 0; ir < sol->nr; ir++) {
      double *offset = gpuMotionTableAppend(&tables, &tableCapacity, &tableValues, sol->Bz[ir], sol->nz);
      if (ir == 0)
        data.BzMap = offset;
    }
    for (ir = 0; ir < sol->nr; ir++) {
      double *offset = gpuMotionTableAppend(&tables, &tableCapacity, &tableValues, sol->Br[ir], sol->nz);
      if (ir == 0)
        data.BrMap = offset;
    }
    break;
  }
  default:
    return gpuMotionCpuAfterAll(field, elementBefore, size);
  }
  if (lostCapacity < nParticles) {
    free(lost);
    if (!(lost = (unsigned char *)malloc(nParticles)))
      gpuRequiredFailure("unable to allocate the motion() loss flags");
    lostCapacity = nParticles;
  }

  gpuCopyHostToDevice(nParticles);
  status = gpuCudaLsrmdltrBackup(gpuBase.deviceCoord, nParticles * gpuBase.deviceStride);
  if (status != 0)
    gpuFatalStatus("motion() coordinate backup", status);
  status = gpuCudaMotionPass(gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride, &data, tables, tableValues,
                             lost, &lostCount, &cpuCount, &milliseconds);
  if (status != 0)
    gpuFatalStatus("motion() CUDA kernel", status);
  gpuRecordMagnetKernel(milliseconds);
  if (cpuCount) {
    status = gpuCudaLsrmdltrRestore(gpuBase.deviceCoord, 0, nParticles * gpuBase.deviceStride);
    if (status != 0)
      gpuFatalStatus("motion() coordinate restore", status);
    if (gpuVerbose)
      fprintf(stderr, "elegant CUDA: %ld particles need motion() on the CPU\n", cpuCount);
    gpuRecordWallSeconds();
    return gpuMotionCpuAfterAll(field, elementBefore, size);
  }
  /* motion() stores the middle of the first particle's x excursion, which only LSRMDLTR records */
  if (xMotionCenterMemory < 0)
    xMotionCenterMemory = rpn_create_mem((char *)"xMotionCenter", 0);
  rpn_store((-1e300 + 1e300) / 2, NULL, xMotionCenterMemory);
  nLeft = nParticles;
  if (lostCount) {
    /* motion()'s swaps: a particle lost at i trades rows with the last one not yet looked at, which is then
     * looked at in its place.  The accepted rows would move too; that stays the CPU's. */
    static long long *perm = NULL;
    static long permCapacity = 0;
    long top = nParticles - 1;

    if (accepted) {
      status = gpuCudaLsrmdltrRestore(gpuBase.deviceCoord, 0, nParticles * gpuBase.deviceStride);
      if (status != 0)
        gpuFatalStatus("motion() coordinate restore", status);
      gpuRecordWallSeconds();
      return gpuMotionCpuAfterAll(field, elementBefore, size);
    }
    if (permCapacity < nParticles) {
      free(perm);
      if (!(perm = (long long *)malloc(nParticles * sizeof(*perm))))
        gpuRequiredFailure("unable to allocate the motion() loss order");
      permCapacity = nParticles;
    }
    for (i = 0; i < nParticles; i++)
      perm[i] = i;
    i = 0;
    while (i <= top) {
      if (!lost[i])
        i++;
      else {
        long long row = perm[i];
        unsigned char flag = lost[i];
        perm[i] = perm[top];
        perm[top] = row;
        lost[i] = lost[top];
        lost[top] = flag;
        top--;
      }
    }
    nLeft = top + 1;
    status = gpuCudaMotionPermute(gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride, perm);
    if (status != 0)
      gpuFatalStatus("motion() loss order", status);
  }
  gpuMarkDeviceChanged(nLeft);
  *changeP0 = setup.changeP0;
#ifndef GPU_VERIFY
  /* motion() matches the central momentum to all the particles it was given, those it lost included, before
   * their rows are recorded; the verify build does this after the CPU's motion() */
  if (setup.changeP0)
    gpu_do_match_energy(nParticles, pCentral, 0);
#endif
  if (lostCount)
    gpuCopyApertureLossRowsToHost(nLeft, nParticles, "motion() lost particles");
  gpuRecordWallSeconds();
  return nLeft;
}

/* BEAMBEAM's kicks on the device (applyBeamBeamKicks()); the constants are formed here as the CPU forms
 * them for each particle */
long gpu_track_beambeam(long nParticles, void *element, double P0) {
  BEAMBEAM *bb = (BEAMBEAM *)element;
  GPU_BEAMBEAM_DATA data;
  double ionCharge;
  float milliseconds = 0;
  int status;

  if (nParticles <= 0)
    return 1;
  if (!bb)
    gpuRequiredFailure("NULL BEAMBEAM CUDA tracking data");
  memset(&data, 0, sizeof(data));
  data.distribution = (int)match_string(bb->distribution, beamBeamDistributionOption, N_BEAM_BEAM_DISTRIBUTIONS, 0);
  if (data.distribution < 0)
    return 0;
  data.centroid[0] = bb->centroid[0];
  data.centroid[1] = bb->centroid[1];
  data.size[0] = bb->size[0];
  data.size[1] = bb->size[1];
  data.charge = bb->charge;
  data.P0 = P0;
  /* applyBeamBeamKicks() and gaussianBeamKick(..., bb->charge, particleMass,
   * particleCharge * particleRelSign / e_mks) */
  ionCharge = particleCharge * particleRelSign / e_mks;
  data.P0c = P0 * c_mks;
  data.mass = particleMass;
  data.inverseMass = 1.0 / particleMass;
  data.C1 = c_mks * bb->charge * re_mks * me_mks * ionCharge / e_mks;
  data.C1round = 2 * c_mks * bb->charge * re_mks * me_mks * ionCharge / e_mks;
  data.twoPi = 2 * PI;
  /* ellipsoidalBeamKick(part, P0, particleMass, -particleRelSign * particleCharge, ...) */
  data.pCharge = -particleRelSign * particleCharge;
  data.pMassCP0 = particleMass * c_mks * P0;
  data.fourPiEpsilon = 4 * PI * epsilon_o;
  data.fourCharge = 4 * bb->charge;
  data.eightChargeThirds = 8 * bb->charge / 3;
  data.eightCharge = 8 * bb->charge;
  data.cMks = c_mks;

  startGpuTimer();
  gpuCopyHostToDevice(nParticles);
  status = gpuCudaBeamBeamKick(gpuBase.deviceCoord, nParticles, (int)gpuBase.deviceStride, &data, &milliseconds);
  if (status != 0)
    gpuFatalStatus("BEAMBEAM CUDA kernel", status);
  gpuRecordMagnetKernel(milliseconds);
  gpuMarkDeviceChanged(nParticles);
  gpuRecordWallSeconds();
  return 1;
}

/* LSC sums that every kick depends on run in the CPU's particle order when output must match elegant */
static long gpuLscCpuOrderSums(void) {
  return gpuEnvSet("ELEGANT_GPU_LSC_REFERENCE_VARIANCE") ?
           gpuEnvFlag("ELEGANT_GPU_LSC_REFERENCE_VARIANCE") :
           gpuBase.orderSensitiveOutputNeeded;
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

  if (gpuLscCpuOrderSums() && rms_emittance) {
    /* Beam radius controls both impedance and adaptive step length. A parallel
     * sum changes this shared reference value and can seed coherent changes
     * in all particle kicks. Keep CPU summation order for reference outputs;
     * binning, drift and voltage kicks still run on CUDA.  Only x and y come to
     * the host, where rms_emittance()'s own sums run on them in particle order:
     * the whole beam used to be copied for each step (1.5 s of 3.3 s for three
     * LSCDRIFTs of 4,000,000 particles). */
    static double *x = NULL, *y = NULL;
    static long capacity = 0;
    double xc = 0, yc = 0, s11 = 0, s33 = 0;
    long ip;

    if (capacity < nParticles) {
      free(x);
      free(y);
      x = (double *)malloc(nParticles * sizeof(*x));
      y = (double *)malloc(nParticles * sizeof(*y));
      if (!x || !y)
        gpuRequiredFailure("unable to allocate the LSC beam-size columns");
      capacity = nParticles;
    }
    status = gpuCudaCoordinateColumnToHost(gpuBase.deviceCoord, nParticles,
                                           (int)gpuBase.deviceStride, 0, x, &milliseconds);
    if (status != 0)
      gpuFatalStatus("LSC beam-size x column copy", status);
    gpuRecordHelperKernel(milliseconds);
    status = gpuCudaCoordinateColumnToHost(gpuBase.deviceCoord, nParticles,
                                           (int)gpuBase.deviceStride, 2, y, &milliseconds);
    if (status != 0)
      gpuFatalStatus("LSC beam-size y column copy", status);
    gpuRecordHelperKernel(milliseconds);
    /* rms_emittance(coord, 0, 2, n, &S11, NULL, &S33, NULL, NULL) */
    for (ip = 0; ip < nParticles; ip++) {
      xc += x[ip];
      yc += y[ip];
    }
    xc /= nParticles;
    yc /= nParticles;
    for (ip = 0; ip < nParticles; ip++) {
      double dx = x[ip] - xc, dy = y[ip] - yc;
      s11 += sqr(dx);
      s33 += sqr(dy);
    }
    *S11 = s11 / nParticles;
    *S33 = s33 / nParticles;
    gpuLscReferenceCalls++;
    return;
  }

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
                                 long backtrack, double pBeam,
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
    a1 = k * beamRadius / pBeam;
    ZImag = a2 / k * (1 - a1 * dbesk1(a1));
    iImag = (iReal = 2 * ib - 1) + 1;
    vfreq[iReal] = ifreq[iImag] * ZImag * factor;
    vfreq[iImag] = -ifreq[iReal] * ZImag * factor;
  }

  realFFT(vfreq, nb, INVERSE_FFT);
  vtime[nb] = 0;
}

/* lscBeamMomentum() in lsc.c: Po times one plus the mean delta of the binned particles.  The binned
 * particles are all of them unless the binning warned, and then the statistics kernel's sum, made in a
 * fixed order, is the sum.  With output in the CPU's order, column 5 comes to the host and the CPU's sum
 * runs on it in particle order, as gpuLscCenteredVariances() does for the beam size.  If some particle
 * wasn't binned, the time column comes too, and the host leaves out the particles binTimeDistribution()
 * leaves out. */
static double gpuLscBeamMomentum(const GPU_BEAM_SUM_DATA *sums,
                                 const GPU_LSC_DATA *data, long nParticles,
                                 long binnedCount, double Po) {
  static double *delta = NULL, *s = NULL;
  static long capacity = 0;
  double deltaSum = 0;
  long ip, count = 0, allBinned;
  float milliseconds = 0;
  int status;

  if (binnedCount <= 0 || nParticles <= 0)
    return Po;
  allBinned = binnedCount == nParticles;
  if (allBinned && !gpuLscCpuOrderSums())
    return Po * (1 + sums->centroidSum[5] / binnedCount);

  if (capacity < nParticles) {
    free(delta);
    free(s);
    delta = (double *)malloc(nParticles * sizeof(*delta));
    s = (double *)malloc(nParticles * sizeof(*s));
    if (!delta || !s)
      gpuRequiredFailure("unable to allocate the LSC beam-momentum columns");
    capacity = nParticles;
  }
  status = gpuCudaCoordinateColumnToHost(gpuBase.deviceCoord, nParticles,
                                         (int)gpuBase.deviceStride, 5, delta, &milliseconds);
  if (status != 0)
    gpuFatalStatus("LSC beam-momentum delta column copy", status);
  gpuRecordHelperKernel(milliseconds);
  if (!allBinned) {
    status = gpuCudaCoordinateColumnToHost(gpuBase.deviceCoord, nParticles,
                                           (int)gpuBase.deviceStride, 4, s, &milliseconds);
    if (status != 0)
      gpuFatalStatus("LSC beam-momentum s column copy", status);
    gpuRecordHelperKernel(milliseconds);
  }
  for (ip = 0; ip < nParticles; ip++) {
    if (!allBinned) {
      /* computeTimeCoordinatesOnly() and binTimeDistribution() */
      double P = Po * (delta[ip] + 1);
      double time = s[ip] * sqrt(sqr(P) + 1) / (c_mks * P);
      long ib = (time - data->tmin) / data->dt + 0.5;

      if (ib < 0 || ib > data->bins - 1)
        continue;
    }
    deltaSum += delta[ip];
    count++;
  }
  if (!count)
    return Po;
  return Po * (1 + deltaSum / count);
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

/* Mirrors lscBeamMomentum() in lsc.c for the verification shadow. */
static double gpuLscCpuBeamMomentum(double **part, long *pbin, long np,
                                    double Po) {
  double deltaSum = 0;
  long ip, count = 0;

  for (ip = 0; ip < np; ip++) {
    if (pbin[ip] < 0)
      continue;
    deltaSum += part[ip][5];
    count++;
  }
  if (!count)
    return Po;
  return Po * (1 + deltaSum / count);
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
  double pBeam;
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
    if (data.dt <= 0) {
      /* The CPU code warns that it can't bin any particle and carries on, so use it for what's left */
      LSCDRIFT remainder = *lsc;
      double **coord = forceParticlesToCpu("LSCDRIFT zero time spread CPU fallback");

      if (!track_through_lscdrift)
        gpuRequiredFailure("CPU LSCDRIFT routine is unavailable");
      if (kickMode)
        remainder.lEffective = lengthLeft;
      else
        remainder.length = lengthLeft;
      gpuBase.elementOnGpu = 0;
      track_through_lscdrift(coord, np0, &remainder, Po, charge);
      gpuMarkHostWillChange();
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
      return;
    }

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

    pBeam = gpuLscBeamMomentum(&sums, &data, np0, binnedCount, Po);

    if (Imax <= 0) {
      length = lengthLeft;
    } else {
      kSC = 2 / beamRadius * sqrt(Imax / ipow3(pBeam) / 17045.0);
      length = 0.1 / kSC;
      if (length > lengthLeft || kickMode)
        length = lengthLeft;
    }
    if (length <= 0 || !isfinite(length))
      length = lengthLeft;

    gpuLscComputeVoltage(Vtime, Ifreq, Itime, nb, data.dt, length,
                         beamRadius, charge->macroParticleCharge,
                         lsc->backtrack, pBeam,
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
                         lsc->backtrack,
                         gpuLscCpuBeamMomentum(cpuCoord, cpuPbin, np0, Po),
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
static void gpuRotationTiltSinCos(double tilt, double *cosTilt, double *sinTilt);

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

static void gpuPackQuadPartialFringeMatrix(double matrix[8], double K1,
                                           const double fringeInt[5],
                                           long part) {
  double J1x, J2x, J3x, J1y, J2y, J3y;
  double K1sqr = sqr(K1);
  double expJ1x, expJ1y;

  if (part == 1) {
    J1x = K1 * fringeInt[1] -
          2 * K1sqr * fringeInt[3] / 3. -
          K1sqr * fringeInt[0] * fringeInt[2] / 2;
    J2x = K1 * fringeInt[2];
    J3x = K1sqr *
          (fringeInt[2] + fringeInt[4] +
           fringeInt[0] * fringeInt[1]);
    J1y = -K1 * fringeInt[1] -
          2 * K1sqr * fringeInt[3] / 3. -
          K1sqr * fringeInt[0] * fringeInt[2] / 2;
    J2y = -J2x;
    J3y = J3x;
  } else {
    J1x = K1 * fringeInt[1] +
          K1sqr * fringeInt[0] * fringeInt[2] / 2;
    J2x = K1 * fringeInt[2];
    J3x = K1sqr *
          (fringeInt[4] - fringeInt[0] * fringeInt[1]);
    J1y = -K1 * fringeInt[1] +
          K1sqr * fringeInt[0] * fringeInt[2] / 2;
    J2y = -J2x;
    J3y = J3x;
  }

  expJ1x = matrix[0] = exp(J1x);
  matrix[1] = J2x / expJ1x;
  matrix[2] = expJ1x * J3x;
  matrix[3] = (1 + J2x * J3x) / expJ1x;
  expJ1y = matrix[4] = exp(J1y);
  matrix[5] = J2y / expJ1y;
  matrix[6] = expJ1y * J3y;
  matrix[7] = (1 + J2y * J3y) / expJ1y;
}

static void gpuPackQuadFringeMatrices(double matrix1[8], double matrix2[8],
                                      double K1,
                                      const double fringeIntM[5],
                                      const double fringeIntP[5],
                                      long entrance) {
  double temporary[8];

  gpuPackQuadPartialFringeMatrix(matrix1, K1, fringeIntM, 1);
  gpuPackQuadPartialFringeMatrix(matrix2, K1, fringeIntP, 2);
  if (!entrance)
    return;

  SWAP_DOUBLE(matrix1[0], matrix1[3]);
  SWAP_DOUBLE(matrix1[4], matrix1[7]);
  SWAP_DOUBLE(matrix2[0], matrix2[3]);
  SWAP_DOUBLE(matrix2[4], matrix2[7]);
  memcpy(temporary, matrix1, sizeof(temporary));
  memcpy(matrix1, matrix2, sizeof(temporary));
  memcpy(matrix2, temporary, sizeof(temporary));
}

static long gpuPackLinearMisalignment(GPU_LINEAR_MISALIGNMENT *misalignment,
                                      long method, double dx, double dy,
                                      double dz, double pitch, double yaw,
                                      double roll, double angle, double length) {
  if (!getLinearizedMisalignmentMatrix)
    return 0;
  for (int face = 0; face < 2; face++)
    getLinearizedMisalignmentMatrix(method, dx, dy, dz, pitch, yaw, roll, angle,
                                    length, face + 1, misalignment->C[face],
                                    misalignment->R[face]);
  return 1;
}

static long gpuPackExactMisalignment(GPU_EXACT_MISALIGNMENT *misalignment,
                                     long method, double dx, double dy,
                                     double dz, double pitch, double yaw,
                                     double roll, double angle, double length,
                                     long entryActive, long exitActive) {
  double frame[EXACT_MISALIGNMENT_FRAME_SIZE];

  if (!getExactMisalignmentFrame ||
      sizeof(misalignment->face[0]) != sizeof(frame))
    return 0;
  misalignment->active[0] = entryActive;
  misalignment->active[1] = exitActive;
  for (int face = 0; face < 2; face++) {
    getExactMisalignmentFrame(method, dx, dy, dz, pitch, yaw, roll, angle,
                              length, face + 1, frame);
    memcpy(&misalignment->face[face], frame, sizeof(frame));
  }
  return 1;
}

static long gpuPackMultipoleMisalignment(GPU_MULTIPOLE_DATA *data, long method,
                                         double pitch, double yaw) {
  if (method == 3 || method == 4) {
    data->hasExactMisalignment = 1;
    return gpuPackExactMisalignment(&data->exactMisalignment, method, data->dx,
                                    data->dy, data->dz, pitch, yaw, 0, 0,
                                    data->drift, 1,
                                    data->dx || data->dy || data->dz || pitch || yaw ||
                                    data->sinTilt || data->cosTilt != 1);
  }
  if (method != 1 && method != 2)
    return 1;
  /* 1: transform at both faces.  2: at the entrance only, which is what multipole_tracking2() does
   * when no offset, pitch, yaw or tilt is set. */
  data->hasLinearMisalignment = data->dx || data->dy || data->dz || pitch || yaw ||
                                data->sinTilt || data->cosTilt != 1 ? 1 : 2;
  return gpuPackLinearMisalignment(&data->misalignment, method, data->dx,
                                   data->dy, data->dz, pitch, yaw, 0, 0,
                                   data->drift);
}

static void gpuLoadMultipoleCoefficients(void) {
  static long loaded = 0;
  static double table[GPU_MULTIPOLE_MAX_TERM_ORDER + 1][GPU_MULTIPOLE_MAX_TERM_ORDER + 1];
  int status;

  if (loaded)
    return;
  for (long order = 0; order <= GPU_MULTIPOLE_MAX_TERM_ORDER; order++) {
    double *coefficient = multipoleExpansionCoefficients(order);
    for (long i = 0; i <= order; i++)
      table[order][i] = coefficient[i];
  }
  status = gpuCudaMultipoleLoadCoefficients(&table[0][0]);
  if (status != 0)
    gpuFatalStatus("multipole coefficient table load", status);
  loaded = 1;
}

static long gpuAppendMultipoleTerms(GPU_MULTIPOLE_TERM *terms, long total,
                                    const MULTIPOLE_DATA *multData,
                                    int *nTerms, int *maxTermOrder) {
  if (!multData || !multData->initialized)
    return 1;
  for (long i = 0; i < multData->orders; i++) {
    for (int skew = 0; skew < 2; skew++) {
      const double *strength = skew ? multData->JnL : multData->KnL;
      GPU_MULTIPOLE_TERM *term;

      if (!strength || !strength[i])
        continue;
      if (total >= GPU_MULTIPOLE_MAX_TERMS ||
          multData->order[i] < 0 ||
          multData->order[i] > GPU_MULTIPOLE_MAX_TERM_ORDER)
        return 0;
      term = terms + total++;
      term->strength = strength[i];
      term->order = multData->order[i];
      term->skew = skew;
      (*nTerms)++;
      if (term->order > *maxTermOrder)
        *maxTermOrder = term->order;
    }
  }
  return 1;
}

static long gpuPackMultipoleTermList(GPU_MULTIPOLE_DATA *data,
                                     const MULTIPOLE_DATA *multData,
                                     int *nTerms) {
  return gpuAppendMultipoleTerms(data->term,
                                 data->nEdgeTerms + data->nSteeringTerms + data->nBodyTerms,
                                 multData, nTerms, &data->maxTermOrder);
}

static long gpuPackMultipoleTerms(GPU_MULTIPOLE_DATA *data,
                                  const MULTIPOLE_DATA *edge,
                                  const MULTIPOLE_DATA *steering,
                                  const MULTIPOLE_DATA *body) {
  if (!gpuPackMultipoleTermList(data, edge, &data->nEdgeTerms) ||
      !gpuPackMultipoleTermList(data, steering, &data->nSteeringTerms) ||
      !gpuPackMultipoleTermList(data, body, &data->nBodyTerms))
    return 0;
  if (data->nEdgeTerms + data->nSteeringTerms + data->nBodyTerms) {
    if (!multipoleExpansionCoefficients)
      return 0;
    gpuLoadMultipoleCoefficients();
  }
  return 1;
}

static long gpuPackMultipoleTrackingWithoutSpin(GPU_MULTIPOLE_DATA *data,
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
    gpuRotationTiltSinCos(multipole->tilt, &data->cosTilt, &data->sinTilt);
    return 1;
  }
  case T_FMULT: {
    /* fmultipole_tracking(): order 4, one slice per kick, all terms from the file scaled by
     * (1+FSE)*FACTOR, no aperture, and expandHamiltonian as the last magnet left it */
    FMULT *fmult = (FMULT *)elem->p_elem;
    long n = fmult->n_kicks > 0 ? fmult->n_kicks : fmult->nSlices;

    if (n <= 0 || !initialize_fmultipole)
      return 0;
    if (!fmult->multData.initialized)
      initialize_fmultipole(fmult);
    gpuInitMultipoleData(data, Po, fmult->length, n, 4,
                         &expandHamiltonian ? expandHamiltonian : 0);
    data->dx = fmult->dx;
    data->dy = fmult->dy;
    data->dz = fmult->dz;
    if (fmult->synch_rad)
      data->radCoef =
        sqr(particleCharge) * pow3(Po) /
        (6 * PI * epsilon_o * sqr(c_mks) * particleMass);
    gpuRotationTiltSinCos(fmult->tilt, &data->cosTilt, &data->sinTilt);
    if (!gpuPackMultipoleTerms(data, NULL, NULL, &fmult->multData))
      return 0;
    for (int i = 0; i < data->nBodyTerms; i++)
      data->term[i].strength *= (1 + fmult->fse) * fmult->factor;
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
    data->endDrift = (kquad->length - lEffective) / 2;
    data->k1 = kquad->k1;
    data->edge1Effects = kquad->edge1_effects;
    data->edge2Effects = kquad->edge2_effects;
    data->edge1Linear = kquad->edge1Linear;
    data->edge2Linear = kquad->edge2Linear;
    data->edge1NonlinearFactor = kquad->edge1NonlinearFactor;
    data->edge2NonlinearFactor = kquad->edge2NonlinearFactor;
    if (data->edge1Effects)
      gpuPackQuadFringeMatrices(data->edge1M1, data->edge1M2, kquad->k1,
                                kquad->fringeIntM, kquad->fringeIntP, 1);
    if (data->edge2Effects)
      gpuPackQuadFringeMatrices(data->edge2M1, data->edge2M2, kquad->k1,
                                kquad->fringeIntM, kquad->fringeIntP, 0);
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
    gpuRotationTiltSinCos(kquad->tilt, &data->cosTilt, &data->sinTilt);
    if (!gpuPackMultipoleMisalignment(data, kquad->malignMethod, kquad->pitch,
                                      kquad->yaw))
      return 0;
    if (!setupKquadErrorMultipoles)
      return 0;
    setupKquadErrorMultipoles(kquad, data->KnL[0]);
    return gpuPackMultipoleTerms(data, &kquad->edgeMultipoleData,
                                 &kquad->steeringMultipoleData,
                                 &kquad->totalMultipoleData);
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
    if (ksext->synch_rad)
      data->radCoef =
        sqr(particleCharge) * pow3(Po) /
        (6 * PI * epsilon_o * sqr(c_mks) * particleMass);
    gpuRotationTiltSinCos(ksext->tilt, &data->cosTilt, &data->sinTilt);
    if (!gpuPackMultipoleMisalignment(data, ksext->malignMethod, ksext->pitch,
                                      ksext->yaw))
      return 0;
    if (!setupKsextErrorMultipoles)
      return 0;
    setupKsextErrorMultipoles(ksext, data->KnL[0]);
    return gpuPackMultipoleTerms(data, &ksext->edgeMultipoleData,
                                 &ksext->steeringMultipoleData,
                                 &ksext->totalMultipoleData);
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
    gpuRotationTiltSinCos(koct->tilt, &data->cosTilt, &data->sinTilt);
    if (!gpuPackMultipoleMisalignment(data, koct->malignMethod, koct->pitch,
                                      koct->yaw))
      return 0;
    if (!setupKoctErrorMultipoles)
      return 0;
    setupKoctErrorMultipoles(koct, data->KnL[0]);
    return gpuPackMultipoleTerms(data, NULL, NULL, &koct->totalMultipoleData);
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
    gpuRotationTiltSinCos(dqcor->tilt, &data->cosTilt, &data->sinTilt);
    if (!gpuPackMultipoleMisalignment(data, dqcor->malignMethod, dqcor->pitch,
                                      dqcor->yaw))
      return 0;
    if (!setupDqcorErrorMultipoles)
      return 0;
    setupDqcorErrorMultipoles(dqcor, data->KnL[0]);
    return gpuPackMultipoleTerms(data, NULL, &dqcor->dipoleSystematicMultipoleData,
                                 &dqcor->totalMultipoleData);
  }
  case T_KQUSE: {
    /* multipole_tracking2(): a quadrupole with the sextupole as a second order, original
     * misalignment method, no fringes or multipole files */
    KQUSE *kquse = (KQUSE *)elem->p_elem;

    if (!gpuResolveMultipoleSlices(kquse->nSlices, kquse->n_kicks,
                                   kquse->integration_order, &nSlices))
      return 0;
    gpuInitMultipoleData(data, Po, kquse->length, nSlices,
                         kquse->integration_order, kquse->expandHamiltonian);
    data->order[0] = 1;
    data->KnL[0] = kquse->k1 * kquse->length * (1 + kquse->fse1);
    data->order[1] = 2;
    data->KnL[1] = kquse->k2 * kquse->length * (1 + kquse->fse2);
    data->dx = kquse->dx;
    data->dy = kquse->dy;
    data->dz = kquse->dz;
    gpuRotationTiltSinCos(kquse->tilt, &data->cosTilt, &data->sinTilt);
    return 1;
  }
  default:
    break;
  }

  return 0;
}

/* Spin follows multipole_tracking2(): rolled by the tilt with the coordinates, as every misalignment
 * method's rotateBeamCoordinatesForMisalignment() does, and turned after each kick and edge kick.
 * multipole_tracking() only rolls it for MULT. */
static long gpuPackMultipoleTracking(GPU_MULTIPOLE_DATA *data,
                                     ELEMENT_LIST *elem, double Po) {
  if (!gpuPackMultipoleTrackingWithoutSpin(data, elem, Po))
    return 0;
  if (!spinCoordOffset)
    return 1;
  if (elem->type == T_MULT)
    data->spinRollOnly = 1;
  else if (elem->type != T_KQUAD && elem->type != T_KSEXT && elem->type != T_KOCT &&
           elem->type != T_DQCOR && elem->type != T_FMULT && elem->type != T_KQUSE)
    return 0;
  else {
    /* Until CPU fixes 12 and 13 are committed (elegant-CPU-bugs-and-fixes.md), multipole_tracking2() turns
     * the spins of a tilted KQUAD, KSEXT, KOCT, KQUSE or DQCOR by the tilt twice, and a zero-length magnet's
     * kick makes every spin NaN.  The device does neither, so these stay on the CPU and give elegant's
     * numbers.  Remove this block with those fixes. */
    double tilt = 0, length = 0;

    switch (elem->type) {
    case T_KQUAD:
      tilt = ((KQUAD *)elem->p_elem)->tilt;
      length = ((KQUAD *)elem->p_elem)->length;
      break;
    case T_KSEXT:
      tilt = ((KSEXT *)elem->p_elem)->tilt;
      length = ((KSEXT *)elem->p_elem)->length;
      break;
    case T_KOCT:
      tilt = ((KOCT *)elem->p_elem)->tilt;
      length = ((KOCT *)elem->p_elem)->length;
      break;
    case T_KQUSE:
      tilt = ((KQUSE *)elem->p_elem)->tilt;
      length = ((KQUSE *)elem->p_elem)->length;
      break;
    case T_DQCOR:
      tilt = ((DQCOR *)elem->p_elem)->tilt;
      length = ((DQCOR *)elem->p_elem)->length;
      break;
    case T_FMULT:
      length = ((FMULT *)elem->p_elem)->length;
      break;
    }
    if (tilt || !length)
      return 0;
  }
  data->spinOffset = spinCoordOffset;
  data->spinRigidity = -Po * particleMass * c_mks / particleCharge * particleRelSign;
  data->spinCharge = -particleCharge * particleRelSign;
  data->spinMassC = particleMass * c_mks;
  data->spinAnomaly = particleAnomalousMagneticMoment;
  return 1;
}

static long gpuPackExactCorrector(GPU_EXACT_CORRECTOR_DATA *data,
                                  ELEMENT_LIST *elem, double pCentral,
                                  double zStart) {
  double tilt = 0;

  if (!data || !gpuExactCorrectorElementSupported(elem))
    return 0;
  memset(data, 0, sizeof(*data));
  data->pCentral = pCentral;
  data->zStart = zStart;
  switch (elem->type) {
  case T_EHCOR: {
    EHCOR *corrector = (EHCOR *)elem->p_elem;
    data->length = corrector->length;
    data->xkick = corrector->kick * corrector->calibration;
    data->dx = corrector->dx;
    data->dy = corrector->dy;
    data->dz = corrector->dz;
    tilt = corrector->tilt;
    break;
  }
  case T_EVCOR: {
    EVCOR *corrector = (EVCOR *)elem->p_elem;
    data->length = corrector->length;
    data->ykick = corrector->kick * corrector->calibration;
    data->dx = corrector->dx;
    data->dy = corrector->dy;
    data->dz = corrector->dz;
    tilt = corrector->tilt;
    break;
  }
  case T_EHVCOR: {
    EHVCOR *corrector = (EHVCOR *)elem->p_elem;
    data->length = corrector->length;
    data->xkick = corrector->xkick * corrector->xcalibration;
    data->ykick = corrector->ykick * corrector->ycalibration;
    data->dx = corrector->dx;
    data->dy = corrector->dy;
    data->dz = corrector->dz;
    tilt = corrector->tilt;
    break;
  }
  default:
    return 0;
  }
  {
    double tx = tan(data->xkick);
    double ty = tan(data->ykick);
    data->theta0 = atan(sqrt(tx * tx + ty * ty));
    data->rho0 =
      data->theta0 ? fabs(data->length) / sin(data->theta0) : 0;
    if (data->theta0)
      tilt += atan2(ty, tx);
    if (data->length < 0)
      tilt += PI;
  }
  gpuRotationTiltSinCos(tilt, &data->cosTilt, &data->sinTilt);
  return 1;
}

static long gpuPackMaxamp(GPU_APERTURE_LIMIT_DATA *data, MAXAMP *maxamp,
                          double cosTilt, double sinTilt) {
  if (!data)
    return 0;
  memset(data, 0, sizeof(*data));
  data->cosReverseTilt = cosTilt;
  data->sinReverseTilt = -sinTilt;
  if (!maxamp)
    return 1;
  if ((maxamp->openSide && *maxamp->openSide) && !determineOpenSideCode)
    return 0;
  data->xMax = maxamp->x_max;
  data->yMax = maxamp->y_max;
  data->elliptical = maxamp->elliptical ? 1 : 0;
  data->xExponent = maxamp->exponent;
  data->yExponent = maxamp->yExponent ?
    maxamp->yExponent : maxamp->exponent;
  data->openSide = determineOpenSideCode ?
    (int)determineOpenSideCode(maxamp->openSide) : 0;
  data->present = data->xMax > 0 || data->yMax > 0;
  if (data->elliptical && data->present &&
      (data->xExponent <= 0 || data->yExponent <= 0))
    return 0;
  return isfinite(data->xMax) && isfinite(data->yMax);
}

/* An aperture_data file is evaluated once at the magnet's center, and may override MAXAMP */
static long gpuPackFileAperture(GPU_APERTURE_LIMIT_DATA *data, MAXAMP *maxamp,
                                APERTURE_DATA *apFileData, double zPosition,
                                ELEMENT_LIST *eptr) {
  MULT_APERTURE_DATA apertureData;

  if (!apFileData || !apFileData->initialized)
    return 1;
  if (!setupMultApertureData)
    return 0;
  setupMultApertureData(&apertureData, 0, NULL, maxamp, apFileData, NULL, zPosition, eptr);
  data->xMax = apertureData.xMax;
  data->yMax = apertureData.yMax;
  data->xCen = apertureData.xCen;
  data->yCen = apertureData.yCen;
  data->elliptical = apertureData.elliptical ? 1 : 0;
  data->openSide = apertureData.openSide;
  data->present = data->xMax > 0 || data->yMax > 0;
  return isfinite(data->xMax) && isfinite(data->yMax) &&
         isfinite(data->xCen) && isfinite(data->yCen);
}

/* Turns each contour's Logic expression into stack operations for the kernel: &&, || and ! as
 * rpn() does them, and popl.  Returns 1 plus the number of operations, or 0 if the expressions need
 * anything else, or more than truth values on a short stack, which leaves the magnet on the CPU.
 * With ops NULL this only checks. */
static long gpuCompileContourLogic(APCONTOUR *apContour, signed char **ops, long **opStart) {
  long total = 0, depth = 1, ic, used = 0;

  if (ops) {
    for (ic = 0; ic < apContour->nContours; ic++)
      if (apContour->logic[ic])
        total += strlen(apContour->logic[ic]);
    *ops = (signed char *)malloc((total + 1) * sizeof(**ops));
    *opStart = (long *)malloc((apContour->nContours + 1) * sizeof(**opStart));
    if (!*ops || !*opStart)
      gpuRequiredFailure("unable to allocate APCONTOUR logic");
    (*opStart)[0] = 0;
  }
  for (ic = 0; ic < apContour->nContours; ic++) {
    const char *s = apContour->logic[ic];

    if (!s || ++depth > GPU_CONTOUR_LOGIC_MAX_STACK)
      return 0;
    while (*s) {
      signed char op = 0;
      long length = strcspn(s, " \t");

      if (length == 0) {
        s++;
        continue;
      }
      if (length == 2 && !strncmp(s, "&&", 2))
        op = GPU_CONTOUR_LOGIC_AND;
      else if (length == 2 && !strncmp(s, "||", 2))
        op = GPU_CONTOUR_LOGIC_OR;
      else if (length == 1 && *s == '!')
        op = GPU_CONTOUR_LOGIC_NOT;
      else if (length == 4 && !strncmp(s, "popl", 4))
        op = GPU_CONTOUR_LOGIC_POP;
      else
        return 0;
      if (op == GPU_CONTOUR_LOGIC_NOT ? depth < 1 : depth < 2)
        return 0;
      if (op != GPU_CONTOUR_LOGIC_NOT)
        depth--;
      if (ops)
        (*ops)[used] = op;
      used++;
      s += length;
    }
    if (ops)
      (*opStart)[ic + 1] = used;
  }
  return used + 1;
}

/* Uploaded contours, by their points (a sticky check passes a copy of the element), so that a
 * program packing several elements at once can keep each one's.  The least recently used is
 * replaced, except while gpuPinApertureContours() protects those used since */
#define GPU_CONTOUR_CACHE_SIZE 32
typedef struct GPU_CONTOUR_CACHE_ENTRY {
  const void *owner;
  void *x, *y, *start, *logic, *logicStart;
  unsigned long lastUse;
} GPU_CONTOUR_CACHE_ENTRY;
static GPU_CONTOUR_CACHE_ENTRY gpuContourCache[GPU_CONTOUR_CACHE_SIZE];
static unsigned long gpuContourCacheClock = 0, gpuContourCachePinned = 0;

#if !USE_MPI
/* Until the next call with 0, contours packed from now on stay on the device */
static void gpuPinApertureContours(long pin) {
  gpuContourCachePinned = pin ? ++gpuContourCacheClock : 0;
}
#endif

static GPU_CONTOUR_CACHE_ENTRY *gpuUploadApertureContour(APCONTOUR *apContour) {
  GPU_CONTOUR_CACHE_ENTRY *entry = NULL;
  float milliseconds = 0;
  long points = 0, i;
  double *x, *y;
  long *start;

  for (i = 0; i < GPU_CONTOUR_CACHE_SIZE; i++)
    if (gpuContourCache[i].owner == apContour->x) {
      gpuContourCache[i].lastUse = ++gpuContourCacheClock;
      return gpuContourCache + i;
    }
  for (i = 0; i < GPU_CONTOUR_CACHE_SIZE; i++)
    if ((!gpuContourCache[i].owner || gpuContourCache[i].lastUse < gpuContourCachePinned ||
         !gpuContourCachePinned) &&
        (!entry || !gpuContourCache[i].owner ||
         (entry->owner && gpuContourCache[i].lastUse < entry->lastUse)))
      entry = gpuContourCache + i;
  if (!entry)
    return NULL;
  if (entry->x)
    gpuCudaFree(entry->x);
  if (entry->y)
    gpuCudaFree(entry->y);
  if (entry->start)
    gpuCudaFree(entry->start);
  if (entry->logic)
    gpuCudaFree(entry->logic);
  if (entry->logicStart)
    gpuCudaFree(entry->logicStart);
  memset(entry, 0, sizeof(*entry));
  for (i = 0; i < apContour->nContours; i++)
    points += apContour->nPoints[i];
  x = (double *)malloc(points * sizeof(*x));
  y = (double *)malloc(points * sizeof(*y));
  start = (long *)malloc((apContour->nContours + 1) * sizeof(*start));
  if (!x || !y || !start)
    gpuRequiredFailure("unable to allocate APCONTOUR upload buffers");
  start[0] = 0;
  for (i = 0; i < apContour->nContours; i++) {
    memcpy(x + start[i], apContour->x[i], apContour->nPoints[i] * sizeof(*x));
    memcpy(y + start[i], apContour->y[i], apContour->nPoints[i] * sizeof(*y));
    start[i + 1] = start[i] + apContour->nPoints[i];
  }
  if (gpuCudaMallocDouble(&entry->x, points) != 0 ||
      gpuCudaMallocDouble(&entry->y, points) != 0 ||
      gpuCudaMallocBytes(&entry->start, (apContour->nContours + 1) * sizeof(*start)) != 0 ||
      gpuCudaCopyHostToDevice(entry->x, x, points, &milliseconds) != 0 ||
      gpuCudaCopyHostToDevice(entry->y, y, points, &milliseconds) != 0 ||
      gpuCudaCopyHostBytesToDevice(entry->start, start, (apContour->nContours + 1) * sizeof(*start)) != 0)
    gpuRequiredFailure("unable to upload APCONTOUR points");
  free(x);
  free(y);
  free(start);
  if (apContour->hasLogic) {
    signed char *ops = NULL;
    long *opStart = NULL;
    long nOps = gpuCompileContourLogic(apContour, &ops, &opStart) - 1;

    if (gpuCudaMallocBytes(&entry->logic, nOps + 1) != 0 ||
        gpuCudaMallocBytes(&entry->logicStart, (apContour->nContours + 1) * sizeof(*opStart)) != 0 ||
        gpuCudaCopyHostBytesToDevice(entry->logic, ops, nOps + 1) != 0 ||
        gpuCudaCopyHostBytesToDevice(entry->logicStart, opStart,
                                     (apContour->nContours + 1) * sizeof(*opStart)) != 0)
      gpuRequiredFailure("unable to upload APCONTOUR logic");
    free(ops);
    free(opStart);
  }
  entry->owner = apContour->x;
  entry->lastUse = ++gpuContourCacheClock;
  return entry;
}

static long gpuPackApertureContour(GPU_APERTURE_CONTOUR_DATA *data, APCONTOUR *apContour) {
  GPU_CONTOUR_CACHE_ENTRY *entry;

  memset(data, 0, sizeof(*data));
  if (!apContour)
    return 1;
  if (!apContour->initialized) {
    if (!initializeApContour)
      return 0;
    initializeApContour(apContour);
  }
  if (apContour->nContours <= 0)
    return 0;
  if (apContour->hasLogic && !gpuCompileContourLogic(apContour, NULL, NULL))
    return 0;
  if (!(entry = gpuUploadApertureContour(apContour)))
    return 0;
  data->logic = (const signed char *)entry->logic;
  data->logicStart = (const long *)entry->logicStart;
  data->x = (const double *)entry->x;
  data->y = (const double *)entry->y;
  data->start = (const long *)entry->start;
  data->nContours = apContour->nContours;
  data->invert = apContour->invert ? 1 : 0;
  data->dx = apContour->dx;
  data->dy = apContour->dy;
  data->dz = apContour->dz;
  gpuRotationTiltSinCos(apContour->tilt, &data->cosTilt, &data->sinTilt);
  data->xFactor = apContour->xFactor;
  data->yFactor = apContour->yFactor;
  return 1;
}

static long gpuApContourOnCpu(long np, double **accepted, double z, double Po,
                              APCONTOUR *apcontour, ELEMENT_LIST *eptr, const char *reason) {
  double **coord = forceParticlesToCpu(reason);
  long remaining;

  if (!trackThroughApContour)
    gpuRequiredFailure("CPU trackThroughApContour fallback is unavailable");
  gpuBase.elementOnGpu = 0;
  remaining = trackThroughApContour(coord, apcontour, np, accepted, z, Po, eptr);
  gpuMarkHostWillChange();
  gpuRecordWallSeconds();
  return remaining;
}

static long gpuApContourStableCompact(GPU_APERTURE_CONTOUR_DATA *contour, long np, double **accepted,
                                      double z, double Po) {
  long remaining = np;
  float milliseconds = 0;
  int status;

  gpuEnsureApertureScratch(np);
  status = gpuCudaApContourStableCompact(gpuBase.deviceCoord, gpuApertureScratch.coord,
                                         gpuApertureScratch.prefix, np, (int)gpuBase.deviceStride,
                                         contour, z, Po, &remaining, &milliseconds);
  if (status != 0)
    gpuFatalStatus("APCONTOUR stable aperture compaction", status);
  gpuRecordApertureKernel(milliseconds);
  if (accepted) {
    if (!gpuStablePartitionAcceptedOnDevice(accepted, np, remaining,
                                            "APCONTOUR CUDA accepted stable compaction")) {
      gpuCopyAperturePrefixToHost(np, "APCONTOUR CUDA stable compaction accepted map");
      gpuStablePartitionAccepted(accepted, np, remaining);
    }
  }
  gpuPromoteApertureScratchCoord();
  gpuCopyApertureLossRowsToHost(remaining, np, "APCONTOUR CUDA stable compaction loss tail output");
  gpuMarkDeviceChanged(remaining);
  gpuRecordWallSeconds();
  return remaining;
}

/* The APCONTOUR element, and a sticky contour's check after each element, which comes here with L=0 */
long gpu_track_through_apcontour(void *apcontour0, long np, double **accepted, double z, double Po,
                                 void *eptr0) {
  APCONTOUR *apcontour = (APCONTOUR *)apcontour0;
  ELEMENT_LIST *eptr = (ELEMENT_LIST *)eptr0;
  GPU_APERTURE_CONTOUR_DATA contour;
  long lostCount = 0;
  float milliseconds = 0;
  int status;

  if (np <= 0)
    return np;
  if (!gpuApContourSupported(apcontour) ||
      !gpuResidentParticleCountAllowed(np, gpuBase.apertureMinParticles) ||
      !gpuPackApertureContour(&contour, apcontour))
    return gpuApContourOnCpu(np, accepted, z, Po, apcontour, eptr, "APCONTOUR CPU fallback");
  gpuCopyHostToDevice(np);
  status = gpuCudaApContourLossCount(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, &contour,
                                     apcontour->length, &lostCount, &milliseconds);
  if (status != 0)
    gpuFatalStatus("APCONTOUR predicate kernel", status);
  gpuRecordApertureKernel(milliseconds);
  if (!lostCount) {
    if (apcontour->length > 0)
      gpu_exactDrift(np, apcontour->length);
    gpuRecordWallSeconds();
    return np;
  }
  /* With DZ the CPU also moves each lost particle's s back along its slope */
  if (gpuEnableApertureParallelCompaction && apcontour->length <= 0 && globalLossCoordOffset <= 0 &&
      !apcontour->dz)
    return gpuApContourStableCompact(&contour, np, gpuApertureHostAccepted(accepted), z, Po);
  return gpuApContourOnCpu(np, accepted, z, Po, apcontour, eptr, "APCONTOUR particle loss fallback");
}

static long gpuPackTaperAperture(GPU_TAPER_APERTURE_DATA *data,
                                 ELEMENT_LIST *element, double pCentral,
                                 double zStart) {
  if (!data || !element || !element->p_elem)
    return 0;
  memset(data, 0, sizeof(*data));
  data->pCentral = pCentral;
  data->zStart = zStart;
  data->cosTilt = 1;
  data->localLossColumn = -1;
  switch (element->type) {
  case T_TAPERAPC: {
    TAPERAPC *taper = (TAPERAPC *)element->p_elem;
    if (taper->length < 0)
      return 0;
    data->type = GPU_TAPER_APERTURE_CIRCULAR;
    data->length = taper->length;
    data->xStart = taper->r[taper->e1Index];
    data->xEnd = taper->r[taper->e2Index];
    data->dx = taper->dx;
    data->dy = taper->dy;
    break;
  }
  case T_TAPERAPE: {
    TAPERAPE *taper = (TAPERAPE *)element->p_elem;
    if (taper->length < 0 || !(taper->resolution > 0) || taper->xExponent < 1 || taper->yExponent < 1)
      return 0;
    data->type = GPU_TAPER_APERTURE_ELLIPTICAL;
    data->length = taper->length;
    data->xStart = taper->a[taper->e1Index];
    data->xEnd = taper->a[taper->e2Index];
    data->yStart = taper->b[taper->e1Index];
    data->yEnd = taper->b[taper->e2Index];
    /* trackThroughTaperApElliptical()'s slopes of the semi-axes */
    if (taper->length > 0) {
      data->dadz = (taper->a[taper->e2Index] - taper->a[taper->e1Index]) / taper->length;
      data->dbdz = (taper->b[taper->e2Index] - taper->b[taper->e1Index]) / taper->length;
    }
    data->resolution = taper->resolution;
    if (globalLossCoordOffset > 0 && globalLossCoordOffset + 2 < gpuBase.deviceStride)
      data->localLossColumn = (int)globalLossCoordOffset;
    data->xExponent = taper->xExponent;
    data->yExponent = taper->yExponent;
    data->dx = taper->dx;
    data->dy = taper->dy;
    gpuRotationTiltSinCos(taper->tilt, &data->cosTilt, &data->sinTilt);
    break;
  }
  case T_TAPERAPR: {
    TAPERAPR *taper = (TAPERAPR *)element->p_elem;
    if (taper->length < 0)
      return 0;
    data->type = GPU_TAPER_APERTURE_RECTANGULAR;
    data->length = taper->length;
    data->xStart = taper->xmax[taper->e1Index];
    data->xEnd = taper->xmax[taper->e2Index];
    data->yStart = taper->ymax[taper->e1Index];
    data->yEnd = taper->ymax[taper->e2Index];
    data->dx = taper->dx;
    data->dy = taper->dy;
    gpuRotationTiltSinCos(taper->tilt,
                        &data->cosTilt, &data->sinTilt);
    break;
  }
  default:
    return 0;
  }
  return isfinite(data->length) && isfinite(data->xStart) &&
         isfinite(data->xEnd) && isfinite(data->yStart) &&
         isfinite(data->yEnd) && isfinite(data->dx) &&
         isfinite(data->dy);
}

static long gpuTaperApertureOnCpu(long nParticles, ELEMENT_LIST *element,
                                  double pCentral, double **accepted,
                                  double zStart, const char *reason) {
  double **coord = forceParticlesToCpu(reason);
  long remaining;

  gpuBase.elementOnGpu = 0;
  if (!element || !element->p_elem)
    gpuRequiredFailure("NULL tapered aperture element");
  if (element->type == T_TAPERAPC)
    remaining = trackThroughTaperApCirc(
      coord, (TAPERAPC *)element->p_elem, nParticles, accepted,
      zStart, pCentral, element);
  else if (element->type == T_TAPERAPR)
    remaining = trackThroughTaperApRectangular(
      coord, (TAPERAPR *)element->p_elem, nParticles, accepted,
      zStart, pCentral, element);
  else if (element->type == T_TAPERAPE)
    remaining = trackThroughTaperApElliptical(
      coord, (TAPERAPE *)element->p_elem, nParticles, accepted,
      zStart, pCentral, element);
  else
    gpuRequiredFailure("invalid tapered aperture CPU fallback");
  gpuMarkHostWillChange();
  gpuRecordWallSeconds();
  return remaining;
}

long gpu_track_through_taper_aperture(long nParticles, void *element0,
                                      double pCentral, double **accepted,
                                      double zStart) {
  ELEMENT_LIST *element = (ELEMENT_LIST *)element0;
  GPU_TAPER_APERTURE_DATA data;
  long remaining = nParticles, uncertain = 0;
  float milliseconds = 0;
  int status;

  if (nParticles <= 0)
    return nParticles;
#ifdef GPU_VERIFY
  if (globalLossCoordOffset > 0)
    return gpuTaperApertureOnCpu(
      nParticles, element, pCentral, accepted, zStart,
      "tapered aperture global-loss GPU_VERIFY fallback");
#endif
  if (!gpuPackTaperAperture(&data, element, pCentral, zStart))
    return gpuTaperApertureOnCpu(
      nParticles, element, pCentral, accepted, zStart,
      "tapered aperture unsupported-option CPU fallback");
  gpuCopyHostToDevice(nParticles);
  gpuEnsureApertureScratch(nParticles);
  status = gpuCudaTaperApertureTrackStableCompact(
    gpuBase.deviceCoord, gpuApertureScratch.coord, gpuApertureScratch.prefix,
    nParticles, (int)gpuBase.deviceStride, &data, &remaining, &milliseconds,
    data.type == GPU_TAPER_APERTURE_ELLIPTICAL ? &uncertain : NULL);
  if (status != 0)
    gpuFatalStatus("tapered aperture CUDA stable tracking", status);
  gpuRecordApertureKernel(milliseconds);
  /* A TAPERAPE test within a hair of its boundary: the device's atan2, cos, sin and pow might not decide it
   * as the CPU's do.  The device's coordinates are untouched (the results are in the scratch copy). */
  if (uncertain)
    return gpuTaperApertureOnCpu(
      nParticles, element, pCentral, accepted, zStart,
      "TAPERAPE boundary test CPU fallback");
#ifndef GPU_VERIFY
  if (remaining != nParticles && accepted) {
    if (!gpuStablePartitionAcceptedOnDevice(
          accepted, nParticles, remaining,
          "tapered aperture CUDA accepted stable compaction")) {
      gpuCopyAperturePrefixToHost(
        nParticles, "tapered aperture CUDA accepted map");
      gpuStablePartitionAccepted(accepted, nParticles, remaining);
    }
  }
#endif
  gpuPromoteApertureScratchCoord();
#ifndef GPU_VERIFY
  if (remaining != nParticles) {
    gpuCopyApertureLossRowsToHost(
      remaining, nParticles,
      "tapered aperture CUDA stable compaction loss tail output");
    if (globalLossCoordOffset > 0 && element) {
      static double *local = NULL;
      static long localSize = 0;
      for (long ip = remaining; ip < nParticles; ip++) {
        double X, Y, Z, theta, *row = gpuBase.coord[ip];
        double dz = gpuBase.coord[ip][4] - zStart;
        if (data.localLossColumn >= 0) {
          /* trackThroughTaperApElliptical() converts in the element's frame: x, x' and y from the device */
          if (localSize < gpuBase.deviceStride) {
            free(local);
            if (!(local = (double *)malloc(gpuBase.deviceStride * sizeof(*local))))
              gpuRequiredFailure("unable to allocate the TAPERAPE loss row");
            localSize = gpuBase.deviceStride;
          }
          memcpy(local, gpuBase.coord[ip], gpuBase.deviceStride * sizeof(*local));
          local[0] = gpuBase.coord[ip][data.localLossColumn];
          local[1] = gpuBase.coord[ip][data.localLossColumn + 1];
          local[2] = gpuBase.coord[ip][data.localLossColumn + 2];
          row = local;
        }
        convertLocalCoordinatesToGlobal(
          &Z, &X, &Y, &theta, GLOBAL_LOCAL_MODE_DZ,
          row, element, dz, 0, 0);
        gpuBase.coord[ip][globalLossCoordOffset + 0] = X;
        gpuBase.coord[ip][globalLossCoordOffset + 1] = Z;
        gpuBase.coord[ip][globalLossCoordOffset + 2] = theta;
      }
    }
  }
#else
  {
    long cpuRemaining;
    startCpuTimer();
    if (element->type == T_TAPERAPC)
      cpuRemaining = trackThroughTaperApCirc(
        gpuBase.coord, (TAPERAPC *)element->p_elem, nParticles, accepted,
        zStart, pCentral, element);
    else if (element->type == T_TAPERAPE)
      cpuRemaining = trackThroughTaperApElliptical(
        gpuBase.coord, (TAPERAPE *)element->p_elem, nParticles, accepted,
        zStart, pCentral, element);
    else
      cpuRemaining = trackThroughTaperApRectangular(
        gpuBase.coord, (TAPERAPR *)element->p_elem, nParticles, accepted,
        zStart, pCentral, element);
    if (cpuRemaining != remaining) {
      fprintf(stderr,
              "elegant CUDA VERIFY failed for tapered aperture element=%s: CPU survivors=%ld GPU survivors=%ld\n",
              element && element->name ? element->name : "(unknown)",
              cpuRemaining, remaining);
      exit(1);
    }
    compareGpuCpu(nParticles, "tapered aperture stable tracking");
    gpuRecordWallSeconds();
    return cpuRemaining;
  }
#endif
  gpuMarkDeviceChanged(remaining);
  gpuRecordWallSeconds();
  return remaining;
}

static long gpuPackSpeedbump(GPU_SPEEDBUMP_DATA *data,
                             ELEMENT_LIST *element, double pCentral,
                             double zStart) {
  SPEEDBUMP *speedbump;
  const char *direction;

  if (!data || !element || element->type != T_SPEEDBUMP ||
      !gpuApertureElementSupported(element))
    return 0;
  speedbump = (SPEEDBUMP *)element->p_elem;
  direction = speedbump->insertFrom;
  memset(data, 0, sizeof(*data));
  data->length = speedbump->length;
  data->chord = speedbump->chord;
  data->dzCenter = speedbump->dzCenter;
  data->height = speedbump->height;
  data->position = speedbump->position;
  data->zStart = zStart;
  data->pCentral = pCentral;
  data->scraperConvention = speedbump->scraperConvention ? 1 : 0;
  if (strchr(direction, 'x')) {
    data->plane = 0;
    data->offset = speedbump->dx;
  } else {
    data->plane = 2;
    data->offset = speedbump->dy;
  }
  data->plusDirection = direction[0] != '-';
  /* An unsigned direction selects both sides, as in interpretScraperDirection. */
  data->minusDirection = direction[0] != '+';
  if (speedbump->chord && speedbump->height)
    data->radius =
      (sqr(speedbump->chord) + 4 * sqr(speedbump->height)) /
      (8 * speedbump->height);
  return 1;
}

static long gpuSpeedbumpOnCpu(long nParticles, ELEMENT_LIST *element,
                              double pCentral, double **accepted,
                              double zStart, const char *reason) {
  double **coord = forceParticlesToCpu(reason);
  long remaining;

  gpuBase.elementOnGpu = 0;
  if (!element || !element->p_elem || !track_through_speedbump)
    gpuRequiredFailure("SPEEDBUMP CPU fallback is unavailable");
  remaining = track_through_speedbump(
    coord, (SPEEDBUMP *)element->p_elem, nParticles, accepted,
    zStart, pCentral, element);
  gpuMarkHostWillChange();
  gpuRecordWallSeconds();
  return remaining;
}

long gpu_track_through_speedbump(long nParticles, void *element0,
                                 double pCentral, double **accepted,
                                 double zStart) {
  ELEMENT_LIST *element = (ELEMENT_LIST *)element0;
  GPU_SPEEDBUMP_DATA data;
  long remaining = nParticles;
  float milliseconds = 0;
  int status;

  if (nParticles <= 0)
    return nParticles;
#ifdef GPU_VERIFY
  if (globalLossCoordOffset > 0)
    return gpuSpeedbumpOnCpu(
      nParticles, element, pCentral, accepted, zStart,
      "SPEEDBUMP global-loss GPU_VERIFY fallback");
#endif
  if (!gpuPackSpeedbump(&data, element, pCentral, zStart))
    return gpuSpeedbumpOnCpu(
      nParticles, element, pCentral, accepted, zStart,
      "SPEEDBUMP unsupported-option CPU fallback");
  gpuCopyHostToDevice(nParticles);
  gpuEnsureApertureScratch(nParticles);
  status = gpuCudaSpeedbumpTrackStableCompact(
    gpuBase.deviceCoord, gpuApertureScratch.coord, gpuApertureScratch.prefix,
    nParticles, (int)gpuBase.deviceStride, &data, &remaining, &milliseconds);
  if (status != 0)
    gpuFatalStatus("SPEEDBUMP CUDA stable tracking", status);
  gpuRecordApertureKernel(milliseconds);
#ifndef GPU_VERIFY
  if (remaining != nParticles && accepted) {
    if (!gpuStablePartitionAcceptedOnDevice(
          accepted, nParticles, remaining,
          "SPEEDBUMP CUDA accepted stable compaction")) {
      gpuCopyAperturePrefixToHost(
        nParticles, "SPEEDBUMP CUDA accepted map");
      gpuStablePartitionAccepted(accepted, nParticles, remaining);
    }
  }
#endif
  gpuPromoteApertureScratchCoord();
#ifndef GPU_VERIFY
  if (remaining != nParticles) {
    gpuCopyApertureLossRowsToHost(
      remaining, nParticles,
      "SPEEDBUMP CUDA stable compaction loss tail output");
    if (globalLossCoordOffset > 0 && element) {
      for (long ip = remaining; ip < nParticles; ip++) {
        double X, Y, Z, theta;
        double dz = gpuBase.coord[ip][4] - zStart;
        convertLocalCoordinatesToGlobal(
          &Z, &X, &Y, &theta, GLOBAL_LOCAL_MODE_DZ,
          gpuBase.coord[ip], element, dz, 0, 0);
        gpuBase.coord[ip][globalLossCoordOffset + 0] = X;
        gpuBase.coord[ip][globalLossCoordOffset + 1] = Z;
        gpuBase.coord[ip][globalLossCoordOffset + 2] = theta;
      }
    }
  }
#else
  {
    startCpuTimer();
    long cpuRemaining = track_through_speedbump(
      gpuBase.coord, (SPEEDBUMP *)element->p_elem, nParticles, accepted,
      zStart, pCentral, element);
    if (cpuRemaining != remaining) {
      fprintf(stderr,
              "elegant CUDA VERIFY failed for SPEEDBUMP element=%s: CPU survivors=%ld GPU survivors=%ld\n",
              element && element->name ? element->name : "(unknown)",
              cpuRemaining, remaining);
      exit(1);
    }
    compareGpuCpu(nParticles, "SPEEDBUMP stable tracking");
    gpuRecordWallSeconds();
    return cpuRemaining;
  }
#endif
  gpuMarkDeviceChanged(remaining);
  gpuRecordWallSeconds();
  return remaining;
}

long gpu_track_through_exact_corrector(long nParticles, void *element0,
                                       double pCentral, double **accepted,
                                       double zStart) {
  ELEMENT_LIST *element = (ELEMENT_LIST *)element0;
  GPU_EXACT_CORRECTOR_DATA data;
  long remaining = nParticles;
  long usedScratch = 0;
  float milliseconds = 0;
  int status;

  if (nParticles <= 0)
    return nParticles;
  if (!gpuPackExactCorrector(&data, element, pCentral, zStart)) {
    double **coord = forceParticlesToCpu(
      "exact corrector unsupported-option CPU fallback");
    long result;
    if (!element)
      gpuRequiredFailure("NULL exact corrector element");
    result = trackThroughExactCorrector(
      coord, nParticles, element, pCentral, accepted, zStart, NULL);
#ifndef GPU_VERIFY
    gpuSetGlobalLossCoordinates(result, nParticles,
                                zStart + *((double *)element->p_elem), element, 0);
#endif
    gpuBase.elementOnGpu = 0;
    gpuMarkHostWillChange();
    gpuRecordWallSeconds();
    return result;
  }

  gpuCopyHostToDevice(nParticles);
  gpuEnsureApertureScratch(nParticles);
  status = gpuCudaExactCorrectorTrackStableCompact(
    gpuBase.deviceCoord, gpuApertureScratch.coord, gpuApertureScratch.prefix,
    nParticles, (int)gpuBase.deviceStride, &data, &remaining, &usedScratch,
    &milliseconds);
  if (status != 0)
    gpuFatalStatus("exact corrector CUDA stable tracking", status);
  gpuRecordMagnetKernel(milliseconds);
#ifndef GPU_VERIFY
  if (remaining != nParticles && accepted) {
    if (!gpuStablePartitionAcceptedOnDevice(
          accepted, nParticles, remaining,
          "exact corrector CUDA accepted stable compaction")) {
      gpuCopyAperturePrefixToHost(
        nParticles, "exact corrector CUDA accepted map");
      gpuStablePartitionAccepted(accepted, nParticles, remaining);
    }
  }
#endif
  if (usedScratch)
    gpuPromoteApertureScratchCoord();
#ifndef GPU_VERIFY
  if (remaining != nParticles) {
    gpuCopyApertureLossRowsToHost(
      remaining, nParticles,
      "exact corrector CUDA stable compaction loss tail output");
    if (gpuApertureLossRowsNeedHost())
      gpuSetGlobalLossCoordinates(remaining, nParticles, zStart + data.length,
                                  element, 0);
  }
#else
  {
    startCpuTimer();
    long cpuRemaining = trackThroughExactCorrector(
      gpuBase.coord, nParticles, element, pCentral, accepted, zStart, NULL);
    /* With undefined slopes the kernel drops a particle here, but the CPU leaves it in
     * place with NaN coordinates, for do_tracking to remove after the element. */
    long ip, ic, invalid = 0;
    double **valid = (double **)malloc((cpuRemaining + 1) * sizeof(*valid));

    if (!valid)
      gpuRequiredFailure("unable to allocate CUDA exact corrector verification buffer");
    for (ip = 0; ip < cpuRemaining; ip++) {
      for (ic = 0; ic < 6; ic++)
        if (!isfinite(gpuBase.coord[ip][ic]))
          break;
      if (ic < 6)
        invalid++;
      else
        valid[ip - invalid] = gpuBase.coord[ip];
    }
    if (cpuRemaining - invalid != remaining) {
      fprintf(stderr,
              "elegant CUDA VERIFY failed for exact corrector element=%s: CPU survivors=%ld, of which %ld invalid, GPU survivors=%ld\n",
              element && element->name ? element->name : "(unknown)",
              cpuRemaining, invalid, remaining);
      exit(1);
    }
    if (invalid)
      /* Dropping those particles reorders the kernel's rows */
      compareGpuCpuRowsByParticleID(valid, remaining, "exact corrector stable tracking");
    else
      compareGpuCpu(nParticles, "exact corrector stable tracking");
    free(valid);
    gpuRecordWallSeconds();
    return cpuRemaining;
  }
#endif
  gpuMarkDeviceChanged(remaining);
  gpuRecordWallSeconds();
  return remaining;
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

#ifndef GPU_VERIFY
/* elegant's own routines on a magnet's lost particles, for gpuMagnetLossRowsFromCpu() */
typedef struct {
  ELEMENT_LIST *elem;
  double p_error, Po, z_start;
  MAXAMP *maxamp;
  APCONTOUR *apcontour;
  APERTURE_DATA *apFileData;
  long iSlice;
} GPU_MULTIPOLE2_LOSS_ARGS;

static long gpuMultipole2LossTrack(double **rows, long nRows, void *cookie) {
  GPU_MULTIPOLE2_LOSS_ARGS *a = (GPU_MULTIPOLE2_LOSS_ARGS *)cookie;

  if (!multipole_tracking2)
    return nRows;
  return multipole_tracking2(rows, nRows, a->elem, a->p_error, a->Po, NULL, a->z_start, a->maxamp,
                             a->apcontour, a->apFileData, NULL, a->iSlice);
}

typedef struct {
  void *element;
  double p_error, Po, z_start;
} GPU_MULT_LOSS_ARGS;

static long gpuFmultLossTrack(double **rows, long nRows, void *cookie) {
  GPU_MULT_LOSS_ARGS *a = (GPU_MULT_LOSS_ARGS *)cookie;

  if (!fmultipole_tracking)
    return nRows;
  return fmultipole_tracking(rows, nRows, (FMULT *)a->element, a->p_error, a->Po, NULL, a->z_start);
}

static long gpuMultLossTrack(double **rows, long nRows, void *cookie) {
  GPU_MULT_LOSS_ARGS *a = (GPU_MULT_LOSS_ARGS *)cookie;

  if (!multipole_tracking)
    return nRows;
  return multipole_tracking(rows, nRows, (MULT *)a->element, a->p_error, a->Po, NULL, a->z_start);
}
#endif

static void gpuPackMultipoleIsr(GPU_MULTIPOLE_DATA *data, ELEMENT_LIST *elem, double Po, long nParticles) {
  short isr = 0, isr1Particle = 0;
  double length = 0;

  switch (elem->type) {
  case T_KQUAD:
    isr = ((KQUAD *)elem->p_elem)->isr;
    isr1Particle = ((KQUAD *)elem->p_elem)->isr1Particle;
    length = ((KQUAD *)elem->p_elem)->length;
    break;
  case T_KSEXT:
    isr = ((KSEXT *)elem->p_elem)->isr;
    isr1Particle = ((KSEXT *)elem->p_elem)->isr1Particle;
    length = ((KSEXT *)elem->p_elem)->length;
    break;
  case T_KOCT:
    isr = ((KOCT *)elem->p_elem)->isr;
    isr1Particle = ((KOCT *)elem->p_elem)->isr1Particle;
    length = ((KOCT *)elem->p_elem)->length;
    break;
  case T_DQCOR:
    isr = ((DQCOR *)elem->p_elem)->isr;
    isr1Particle = ((DQCOR *)elem->p_elem)->isr1Particle;
    length = ((DQCOR *)elem->p_elem)->length;
    break;
  case T_KQUSE:
    isr = ((KQUSE *)elem->p_elem)->isr;
    isr1Particle = ((KQUSE *)elem->p_elem)->isr1Particle;
    length = ((KQUSE *)elem->p_elem)->length;
    break;
  default:
    break;
  }
  if (isr && (isr1Particle || nParticles > 1) && length >= 1e-6) {
    data->isrCoef = particleRadius * sqrt(55.0 / (24 * sqrt(3)) * pow5(Po) * 137.0359895);
    data->isrDrawsPerParticle = data->nSlices * gpuSymplecticKicksPerSlice(data->integrationOrder);
  }
}

static int gpuMultipoleBatchTrack(void *cookie, void *coord, long count, const double *draws,
                                  double *records, long *lostCount, float *milliseconds) {
  GPU_MULTIPOLE_DATA *data = (GPU_MULTIPOLE_DATA *)cookie;

  (void)records;
  data->isrDraws = draws;
  return gpuCudaMultipoleTrackChecked(coord, count, (int)gpuBase.deviceStride, data, NULL, lostCount, milliseconds);
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
  void *survivorFlags = NULL;
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
  if (sigmaDelta2)
    return gpuMultipoleOnCpu(n_part, elem, p_error, Po, accepted, z_start,
                             maxamp, apcontour, apFileData, sigmaDelta2, iSlice,
                             "multipole_tracking2 radiation sigma fallback");
  if (!gpuPackMultipoleTracking(&data, elem, Po))
    return gpuMultipoleOnCpu(n_part, elem, p_error, Po, accepted, z_start,
                             maxamp, apcontour, apFileData, sigmaDelta2, iSlice,
                             "multipole_tracking2 unsupported option");
  /* multipole_tracking2() takes an aperture_data file at z_start plus half the effective length */
  if (!gpuPackMaxamp(&data.aperture, maxamp,
                     data.cosTilt, data.sinTilt) ||
      !gpuPackFileAperture(&data.aperture, maxamp, apFileData, z_start + data.drift / 2, elem))
    return gpuMultipoleOnCpu(n_part, elem, p_error, Po, accepted, z_start,
                             maxamp, apcontour, apFileData, sigmaDelta2, iSlice,
                             "multipole_tracking2 unsupported MAXAMP or aperture_data option");
  if (!gpuPackApertureContour(&data.contour, apcontour))
    return gpuMultipoleOnCpu(n_part, elem, p_error, Po, accepted, z_start,
                             maxamp, apcontour, apFileData, sigmaDelta2, iSlice,
                             "multipole_tracking2 unsupported APCONTOUR option");
  data.lossZStart = z_start;
  gpuPackMultipoleIsr(&data, elem, Po, n_part);

  gpuCopyHostToDevice(n_part);
  if (data.isrCoef) {
    gpuIsrBegin();
    gpuTrackInBatches(n_part, data.isrDrawsPerParticle, 0, gpuMultipoleBatchTrack, NULL, &data,
                      "multipole tracking CUDA checked kernel", &lostCount, &milliseconds);
    data.isrDraws = NULL;
  } else {
    survivorFlags = gpuMagnetSurvivorFlags(n_part);
    status = gpuCudaMultipoleTrackChecked(gpuBase.deviceCoord, n_part,
                                          (int)gpuBase.deviceStride, &data, survivorFlags,
                                          &lostCount, &milliseconds);
    if (status != 0)
      gpuFatalStatus("multipole tracking CUDA checked kernel", status);
  }
  gpuRecordMagnetKernel(milliseconds);
  if (lostCount) {
    if (data.isrCoef) {
      gpuIsrAbandon();
      return gpuMultipoleOnCpu(n_part, elem, p_error, Po, accepted, z_start,
                               maxamp, apcontour, apFileData, sigmaDelta2, iSlice,
                               "multipole_tracking2 ISR particle-loss fallback");
    }
#ifndef GPU_VERIFY
    if (gpuMagnetLossCompactionAllowed()) {
      GPU_MULTIPOLE2_LOSS_ARGS lossArgs = {elem, p_error, Po, z_start, maxamp, apcontour, apFileData,
                                           iSlice};
      long remaining = survivorFlags ?
        gpuCompactTrackedMagnet(n_part, accepted, "multipole_tracking2 stable magnet loss compaction",
                                gpuMultipole2LossTrack, &lossArgs) :
        gpuMultipoleStableCompact(
          &data, n_part, accepted,
          "multipole_tracking2 stable magnet loss compaction", gpuMultipole2LossTrack, &lossArgs);

      if (remaining >= 0) {
        if (&multipoleKicksDone)
          multipoleKicksDone += n_part * data.nSlices * data.integrationOrder;
        return remaining;
      }
    }
#else
    {
      long remaining = gpuVerifyMultipoleStableCompact(
        &data, n_part,
        "multipole_tracking2 GPU-VERIFY stable magnet loss compaction");
      long cpuRemaining;

      startCpuTimer();
      cpuRemaining = multipole_tracking2(
        gpuBase.coord, n_part, elem, p_error, Po, accepted, z_start, maxamp,
        apcontour, apFileData, sigmaDelta2, iSlice);
      if (cpuRemaining != remaining) {
        fprintf(stderr,
                "elegant CUDA VERIFY failed for multipole_tracking2 element=%s: CPU survivors=%ld GPU survivors=%ld\n",
                elem->name ? elem->name : "(unknown)",
                cpuRemaining, remaining);
        exit(1);
      }
      compareGpuCpuSurvivors(remaining, "multipole_tracking2 stable loss tracking");
      gpuRecordWallSeconds();
      return cpuRemaining;
    }
#endif
    return gpuMultipoleOnCpu(n_part, elem, p_error, Po, accepted, z_start,
                             maxamp, apcontour, apFileData, sigmaDelta2, iSlice,
                             "multipole_tracking2 particle loss fallback");
  }
  if (data.isrCoef)
    gpuIsrCommit();
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

static long gpuFmultOnCpu(long n_part, FMULT *fmult, double p_error,
                          double Po, double **accepted, double z_start,
                          const char *reason) {
  double **coord = forceParticlesToCpu(reason);
  long remaining;

  if (!fmultipole_tracking)
    gpuRequiredFailure("CPU fmultipole_tracking fallback is unavailable");
  gpuBase.elementOnGpu = 0;
  remaining = fmultipole_tracking(coord, n_part, fmult, p_error, Po, accepted, z_start);
  gpuMarkHostWillChange();
  gpuRecordWallSeconds();
  return remaining;
}

long gpu_fmultipole_tracking(long n_part, void *fmult0, double p_error,
                             double Po, double **accepted, double z_start) {
  FMULT *fmult = (FMULT *)fmult0;
  ELEMENT_LIST elem;
  GPU_MULTIPOLE_DATA data;
  void *survivorFlags = NULL;
  long lostCount = 0;
  float milliseconds = 0;
  int status;

  if (n_part <= 0)
    return n_part;
  if (!fmult)
    gpuRequiredFailure("NULL FMULT pointer in gpu_fmultipole_tracking");
  if (gpuBase.backtrack)
    return gpuFmultOnCpu(n_part, fmult, p_error, Po, accepted, z_start,
                         "fmultipole_tracking backtracking CPU reference");
  memset(&elem, 0, sizeof(elem));
  elem.type = T_FMULT;
  elem.p_elem = (char *)fmult;
  if (!gpuPackMultipoleTracking(&data, &elem, Po))
    return gpuFmultOnCpu(n_part, fmult, p_error, Po, accepted, z_start,
                         "fmultipole_tracking unsupported option");
  data.lossZStart = z_start;
  if (&multipoleKicksDone)
    multipoleKicksDone += n_part * fmult->multData.orders * data.nSlices * 4;

  gpuCopyHostToDevice(n_part);
  survivorFlags = gpuMagnetSurvivorFlags(n_part);
  status = gpuCudaMultipoleTrackChecked(gpuBase.deviceCoord, n_part,
                                        (int)gpuBase.deviceStride, &data, survivorFlags,
                                        &lostCount, &milliseconds);
  if (status != 0)
    gpuFatalStatus("FMULT tracking CUDA checked kernel", status);
  gpuRecordMagnetKernel(milliseconds);
  if (lostCount) {
#ifndef GPU_VERIFY
    if (gpuMagnetLossCompactionAllowed()) {
      GPU_MULT_LOSS_ARGS lossArgs = {fmult, p_error, Po, z_start};
      long remaining = survivorFlags ?
        gpuCompactTrackedMagnet(n_part, accepted, "fmultipole_tracking stable magnet loss compaction",
                                gpuFmultLossTrack, &lossArgs) :
        gpuMultipoleStableCompact(
          &data, n_part, accepted, "fmultipole_tracking stable magnet loss compaction",
          gpuFmultLossTrack, &lossArgs);

      if (remaining >= 0)
        return remaining;
    }
#endif
    return gpuFmultOnCpu(n_part, fmult, p_error, Po, accepted, z_start,
                         "fmultipole_tracking particle loss fallback");
  }
  gpuMarkDeviceChanged(n_part);
  gpuRecordWallSeconds();
  return n_part;
}

long gpu_multipole_tracking(long n_part, void *multipole0, double p_error,
                            double Po, double **accepted, double z_start) {
  MULT *multipole = (MULT *)multipole0;
  ELEMENT_LIST elem;
  GPU_MULTIPOLE_DATA data;
  void *survivorFlags = NULL;
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
  data.lossZStart = z_start;
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
  survivorFlags = gpuMagnetSurvivorFlags(n_part);
  status = gpuCudaMultipoleTrackChecked(gpuBase.deviceCoord, n_part,
                                        (int)gpuBase.deviceStride, &data, survivorFlags,
                                        &lostCount, &milliseconds);
  if (status != 0)
    gpuFatalStatus("MULT tracking CUDA checked kernel", status);
  gpuRecordMagnetKernel(milliseconds);
  if (lostCount) {
#ifndef GPU_VERIFY
    if (gpuMagnetLossCompactionAllowed()) {
      GPU_MULT_LOSS_ARGS lossArgs = {multipole, p_error, Po, z_start};
      long remaining = survivorFlags ?
        gpuCompactTrackedMagnet(n_part, accepted, "multipole_tracking stable magnet loss compaction",
                                gpuMultLossTrack, &lossArgs) :
        gpuMultipoleStableCompact(
          &data, n_part, accepted,
          "multipole_tracking stable magnet loss compaction", gpuMultLossTrack, &lossArgs);

      if (remaining >= 0) {
        if (&multipoleKicksDone && data.KnL[0])
          multipoleKicksDone += n_part * data.nSlices * 4;
        return remaining;
      }
    }
#else
    {
      long remaining = gpuVerifyMultipoleStableCompact(
        &data, n_part,
        "multipole_tracking GPU-VERIFY stable magnet loss compaction");
      long cpuRemaining;

      startCpuTimer();
      cpuRemaining = multipole_tracking(
        gpuBase.coord, n_part, multipole, p_error, Po, accepted, z_start);
      if (cpuRemaining != remaining) {
        fprintf(stderr,
                "elegant CUDA VERIFY failed for MULT: CPU survivors=%ld GPU survivors=%ld\n",
                cpuRemaining, remaining);
        exit(1);
      }
      compareGpuCpuSurvivors(remaining, "multipole_tracking stable loss tracking");
      gpuRecordWallSeconds();
      return cpuRemaining;
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

/* Same arithmetic as the fringeModel==0 branch of verticalRbendFringe() */
static void gpuPackCcbendEdgeKick(double kick[3], double alpha, double rho0,
                                  double K1, double K2, double gK, long order) {
  if (gK != 0)
    alpha -= gK / fabs(rho0) / cos(alpha) * (1 + sqr(sin(alpha)));
  kick[0] = sin(alpha) / rho0;
  kick[1] = order > 1 ? sin(alpha) * K1 : 0;
  kick[2] = order > 2 ? sin(alpha) * K2 / 2 : 0;
}

static long gpuPackCcbendTracking(GPU_CCBEND_DATA *data, ELEMENT_LIST *eptr,
                                  CCBEND *ccbend, double Po) {
  double rho0, fse, denominator, tilt, yaw, alpha1, alpha2;
  double dx, dy, dz;
  const double highK[6] = {ccbend->K3, ccbend->K4, ccbend->K5,
                           ccbend->K6, ccbend->K7, ccbend->K8};
  long i;

  if (!data || !gpuCcbendCommonSupported(ccbend))
    return 0;
  if (optimizeCcbendForGpu) {
    long elementOnGpu = gpuBase.elementOnGpu;

    gpuBase.elementOnGpu = 0;
    optimizeCcbendForGpu(eptr, ccbend, Po);
    gpuBase.elementOnGpu = elementOnGpu;
  } else if (!ccbend->optimized && (ccbend->optimizeFse || ccbend->optimizeDx)) {
    return 0;
  }
  memset(data, 0, sizeof(*data));
  rho0 = ccbend->length / ccbend->angle;
  data->chordLength = 2 * rho0 * sin(ccbend->angle / 2);
  denominator = 1 - ccbend->KnDelta;
  fse = ccbend->optimized ? ccbend->fse + ccbend->fseOffset : ccbend->fse;
  data->KnL[0] =
    (1 + fse + ccbend->fseDipole) / rho0 * data->chordLength -
    ccbend->xKick;
  data->KnL[1] =
    (1 + fse + ccbend->fseQuadrupole) * ccbend->K1 *
    data->chordLength / denominator;
  data->KnL[2] =
    (1 + fse) * ccbend->K2 * data->chordLength / denominator;
  for (i = 0; i < 6; i++) {
    double KnL = (1 + fse) * highK[i] * data->chordLength / denominator;

    if (ccbend->angle < 0 && !((i + 3) % 2))
      KnL = -KnL;
    if (!isfinite(KnL))
      return 0;
    if (KnL) {
      data->highKnOrder[data->nHighKn] = i + 3;
      data->highKnL[data->nHighKn++] = KnL;
      data->maxTermOrder = i + 3;
    }
  }
  if (data->nHighKn)
    gpuLoadMultipoleCoefficients();
  data->angleSign = 1;
  data->angleHalf = ccbend->angle / 2;
  data->rho0 = rho0;
  yaw = ccbend->yaw;
  tilt = ccbend->tilt;
  if (ccbend->angle < 0) {
    data->angleSign = -1;
    data->angleHalf = -data->angleHalf;
    data->rho0 = -data->rho0;
    data->KnL[0] = -data->KnL[0];
    data->KnL[2] = -data->KnL[2];
    yaw = -yaw;
    tilt += PI;
  }
  if (ccbend->edgeFlip)
    yaw = -yaw;
  data->hasYaw = yaw != 0;
  gpuRotationTiltSinCos(tilt, &data->cosTilt, &data->sinTilt);
  alpha1 = data->angleHalf - yaw;
  alpha2 = data->angleHalf + yaw;
  data->planeTan = tan(alpha1);
  data->planeCos = cos(alpha1);
  data->planeSin = sin(alpha1);
  data->exitPlaneTan = tan(alpha2);
  data->exitPlaneCos = cos(alpha2);
  data->exitPlaneSin = sin(alpha2);
  data->spinAngle1 = alpha1;
  data->spinAngle2 = alpha2;
  data->fringe1Tan = tan(alpha1);
  data->fringe1Sin = sin(alpha1);
  data->fringe1Sec = 1.0 / cos(alpha1);
  data->fringe2Tan = tan(-alpha2);
  data->fringe2Sin = sin(-alpha2);
  data->fringe2Sec = 1.0 / cos(-alpha2);
  dx = ccbend->dx;
  dy = ccbend->dy;
  dz = ccbend->dz * (ccbend->edgeFlip ? -1 : 1);
  if (tilt) {
    dx = ccbend->dx * cos(tilt) + ccbend->dy * sin(tilt);
    dy = -ccbend->dx * sin(tilt) + ccbend->dy * cos(tilt);
  }
  data->dx = dx;
  data->dy = dy;
  data->dz = dz;
  data->malignMethod = ccbend->malignMethod;
  data->entryMisaligned = dx || dy || dz || ccbend->eYaw || ccbend->ePitch;
  data->exitMisaligned = data->entryMisaligned || ccbend->eTilt;
  gpuRotationTiltSinCos(ccbend->eTilt, &data->cosEtilt, &data->sinEtilt);
  if (data->exitMisaligned && (data->malignMethod == 1 || data->malignMethod == 2) &&
      !gpuPackLinearMisalignment(&data->misalignment, data->malignMethod, dx, dy, dz,
                                 ccbend->ePitch, ccbend->eYaw, ccbend->eTilt, 0,
                                 data->chordLength))
    return 0;
  if (data->exitMisaligned && data->malignMethod >= 3 &&
      !gpuPackExactMisalignment(&data->exactMisalignment, data->malignMethod, dx, dy, dz,
                                ccbend->ePitch, ccbend->eYaw, ccbend->eTilt, 0,
                                data->chordLength, 1, 1))
    return 0;
  data->nSlices = ccbend->nSlices;
  data->integrationOrder = ccbend->integration_order;
  data->fringeModel = ccbend->fringeModel;
  data->edgeOrder = ccbend->edgeOrder;
  data->Po = Po;
  {
    /* track_through_ccbend() works out both radiation coefficients whatever the flags.  With ISR off it
     * negates its ISR coefficient and doesn't zero it, and it zeroes the two only for a magnet shorter
     * than 1e-6 m that has ISR or SYNCH_RAD.  So integrate_kick_KnL()'s radiation block runs after every
     * slice of an ordinary CCBEND: it divides qx and qy by 1+dp and multiplies them back, which moves a
     * momentum by a unit in the last place in a few particles in a thousand.  The device does the same. */
    double cpuRadCoef = 0;
    double cpuIsrCoef = particleRadius * sqrt(55.0 / (24 * sqrt(3)) * pow5(Po) * 137.0359895);

    if (ccbend->synch_rad)
      cpuRadCoef = sqr(particleCharge) * pow3(Po) / (6 * PI * epsilon_o * sqr(c_mks) * particleMass);
    if (ccbend->length < 1e-6 && (ccbend->isr || ccbend->synch_rad))
      cpuRadCoef = cpuIsrCoef = 0;
    data->radCoef = cpuRadCoef;
    data->momentumRoundTrip = cpuRadCoef || cpuIsrCoef;
  }
  if (ccbend->optimized) {
    data->referenceCorrection = ccbend->referenceCorrection;
    data->dxOffset = ccbend->dxOffset;
    data->xAdjust = ccbend->xAdjust;
  }
  data->coordLimit = coordLimit;
  data->slopeLimit = slopeLimit;
  memcpy(data->referenceTrajectory, ccbend->referenceTrajectory,
         sizeof(data->referenceTrajectory));
  memcpy(data->fringeInt1,
         ccbend->edgeFlip ? ccbend->fringeInt2 : ccbend->fringeInt1,
         sizeof(data->fringeInt1));
  memcpy(data->fringeInt2,
         ccbend->edgeFlip ? ccbend->fringeInt1 : ccbend->fringeInt2,
         sizeof(data->fringeInt2));
  if (ccbend->edgeFlip) {
    data->fringeInt1[0] *= -1;
    data->fringeInt1[3] *= -1;
    data->fringeInt1[5] *= -1;
    data->fringeInt2[0] *= -1;
    data->fringeInt2[3] *= -1;
    data->fringeInt2[5] *= -1;
  }
  if (data->fringeModel == 0) {
    double K1 = data->KnL[1] / data->chordLength;
    double K2 = data->KnL[2] / data->chordLength;
    double gK1 = 2 * (ccbend->edgeFlip ? ccbend->fint2 : ccbend->fint1) * ccbend->hgap;
    double gK2 = 2 * (ccbend->edgeFlip ? ccbend->fint1 : ccbend->fint2) * ccbend->hgap;

    gpuPackCcbendEdgeKick(data->edge1Kick, alpha1, data->rho0, K1, K2, gK1,
                          data->edgeOrder);
    gpuPackCcbendEdgeKick(data->edge2Kick, alpha2, data->rho0, K1, K2, gK2,
                          data->edgeOrder);
    for (i = 0; i < 3; i++)
      if (!isfinite(data->edge1Kick[i]) || !isfinite(data->edge2Kick[i]))
        return 0;
  }
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
  if (gpuStringSet(ccbend->systematic_multipoles) ||
      gpuStringSet(ccbend->edge_multipoles) ||
      gpuStringSet(ccbend->edge1_multipoles) ||
      gpuStringSet(ccbend->edge2_multipoles) ||
      gpuStringSet(ccbend->random_multipoles)) {
    /* The CPU reports a bad reference order or strength */
    if (!setupCcbendErrorMultipoles || !multipoleExpansionCoefficients ||
        ccbend->referenceOrder < 0 || ccbend->referenceOrder > 2 ||
        !data->KnL[ccbend->referenceOrder])
      return 0;
    setupCcbendErrorMultipoles(ccbend, NULL, data->KnL[ccbend->referenceOrder],
                               data->angleSign);
    if (!gpuAppendMultipoleTerms(data->term, 0, &ccbend->edge1MultipoleData,
                                 &data->nEdge1Terms, &data->maxTermOrder) ||
        !gpuAppendMultipoleTerms(data->term, data->nEdge1Terms,
                                 &ccbend->edge2MultipoleData,
                                 &data->nEdge2Terms, &data->maxTermOrder) ||
        !gpuAppendMultipoleTerms(data->term, data->nEdge1Terms + data->nEdge2Terms,
                                 &ccbend->totalMultipoleData,
                                 &data->nBodyTerms, &data->maxTermOrder))
      return 0;
    if (data->nEdge1Terms + data->nEdge2Terms + data->nBodyTerms)
      gpuLoadMultipoleCoefficients();
  }
  return 1;
}

/* The options batched programs refuse */
static void gpuPackBendPlainOptions(GPU_BEND_OPTIONS *options, ELEMENT_LIST *eptr, double Po,
                                    long nParticles, short isr, short isr1Particle, double length,
                                    double xReference, double cosExtraTilt, long nSlices,
                                    long recordSlices) {
  if (isr && (isr1Particle || nParticles > 1) && length >= 1e-6)
    options->isrCoef = particleRadius * sqrt(55.0 / (24 * sqrt(3)) * pow5(Po) * 137.0359895);
  if (spinCoordOffset) {
    options->spinOffset = spinCoordOffset;
    options->spinBrho = -particleMass * c_mks * Po / particleCharge;
    options->spinCharge = -particleCharge * particleRelSign;
    options->spinMassC = particleMass * c_mks;
    options->spinAnomaly = particleAnomalousMagneticMoment;
  }
  if (recordSlices) {
    double coord[MAX_PROPERTIES_PER_PARTICLE], Z, X, Y, theta;

    /* These hold for the whole magnet, and with dX and dZ zero they are the magnet's reference point */
    memset(coord, 0, sizeof(coord));
    coord[0] = xReference;
    convertLocalCoordinatesToGlobal(&Z, &X, &Y, &theta, GLOBAL_LOCAL_MODE_DZ, coord, eptr, 0.0, 0, nSlices);
    options->globalZ1 = Z;
    options->globalX1 = X;
    options->globalTheta1 = -theta;
    options->globalSin = sin(options->globalTheta1);
    options->globalCos = cos(options->globalTheta1);
    options->globalCosExtraTilt = cosExtraTilt;
    options->globalDxOffset = xReference;
  }
}

typedef struct {
  GPU_CCBEND_DATA *ccbend;
  GPU_LGBEND_DATA *lgbend;
  GPU_LGBEND_LOCAL_APERTURE_DATA *localAperture;
  GLOBAL_BEAM_SUMS *beamSums;
  long rows;
  long rowOffset; /* for an LGBEND run a segment at a time */
} GPU_BEND_BATCH;

/* Same order of summation as the CPU */
static void gpuBendBatchConsume(void *cookie, const double *records, long count) {
  GPU_BEND_BATCH *batch = (GPU_BEND_BATCH *)cookie;
  long ip, row, i;

  for (ip = 0; ip < count; ip++) {
    for (row = 0; row < batch->rows; row++, records += GPU_BEND_RECORD_SIZE) {
      for (i = 0; i < GPU_BEND_RECORD_SIZE; i++)
        batch->beamSums[batch->rowOffset + row].centroid[i] += records[i];
      batch->beamSums[batch->rowOffset + row].n_part += 1;
    }
  }
}

static int gpuCcbendBatchTrack(void *cookie, void *coord, long count, const double *draws,
                               double *records, long *lostCount, float *milliseconds) {
  GPU_BEND_BATCH *batch = (GPU_BEND_BATCH *)cookie;

  batch->ccbend->options.isrDraws = draws;
  batch->ccbend->options.sliceRecord = records;
  return gpuCudaCcbendTrackChecked(coord, count, (int)gpuBase.deviceStride, batch->ccbend,
                                   NULL, lostCount, milliseconds);
}

#ifndef GPU_VERIFY
typedef struct {
  ELEMENT_LIST *eptr;
  void *element;
  double Po, z_start;
  char *rootname;
  MAXAMP *maxamp;
  APCONTOUR *apContour;
  APERTURE_DATA *apFileData;
  long iPart, iFinalSlice;
} GPU_BEND_LOSS_ARGS;

static long gpuCcbendLossTrack(double **rows, long nRows, void *cookie) {
  GPU_BEND_LOSS_ARGS *a = (GPU_BEND_LOSS_ARGS *)cookie;

  if (!track_through_ccbend)
    return nRows;
  return track_through_ccbend(rows, nRows, a->eptr, (CCBEND *)a->element, a->Po, NULL, a->z_start,
                              NULL, a->rootname, a->maxamp, a->apContour, a->apFileData, a->iPart,
                              a->iFinalSlice);
}

static long gpuLgbendLossTrack(double **rows, long nRows, void *cookie) {
  GPU_BEND_LOSS_ARGS *a = (GPU_BEND_LOSS_ARGS *)cookie;

  if (!track_through_lgbend)
    return nRows;
  return track_through_lgbend(rows, nRows, a->eptr, (LGBEND *)a->element, a->Po, NULL, a->z_start,
                              NULL, a->rootname, a->maxamp, a->apContour, a->apFileData, a->iPart,
                              a->iFinalSlice);
}
#endif

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
  GLOBAL_BEAM_SUMS *beamSums = NULL;
  void *survivorFlags = NULL;
  long lostCount = 0, recordSlices = 0;
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
  if (sigmaDelta2)
    return gpuCcbendOnCpu(n_part, eptr, ccbend, Po, accepted, z_start,
                          sigmaDelta2, rootname, maxamp, apContour,
                          apFileData, iPart, iFinalSlice,
                          "CCBEND radiation-sigma CPU fallback");
  ccbendSetupCentroidOutput(ccbend);
  if (!gpuPackCcbendTracking(&data, eptr, ccbend, Po))
    return gpuCcbendOnCpu(n_part, eptr, ccbend, Po, accepted, z_start,
                          sigmaDelta2, rootname, maxamp, apContour,
                          apFileData, iPart, iFinalSlice,
                          "CCBEND unsupported-option CPU fallback");
  if (!gpuPackMaxamp(&data.aperture, maxamp,
                     data.cosTilt, data.sinTilt) ||
      !gpuPackFileAperture(&data.aperture, maxamp, apFileData,
                           z_start + data.chordLength / 2, eptr) ||
      !gpuPackApertureContour(&data.options.contour, apContour))
    return gpuCcbendOnCpu(n_part, eptr, ccbend, Po, accepted, z_start,
                          sigmaDelta2, rootname, maxamp, apContour,
                          apFileData, iPart, iFinalSlice,
                          "CCBEND aperture CPU fallback");
  /* Under GPU_VERIFY the CPU tracks these particles again and writes the file */
#ifndef GPU_VERIFY
  recordSlices = ccbendCentroidSumsEnabled(ccbend, iPart, iFinalSlice);
#endif
  ccbend->extraTilt = ccbend->angle < 0 ? PI : 0;
  gpuPackBendPlainOptions(&data.options, eptr, Po, n_part, ccbend->isr, ccbend->isr1Particle,
                          ccbend->length, ccbend->dxOffset, cos(ccbend->extraTilt), data.nSlices,
                          recordSlices);
  if (data.options.isrCoef)
    gpuIsrBegin();
  if (recordSlices) {
    beamSums = (GLOBAL_BEAM_SUMS *)calloc(data.nSlices + 1, sizeof(*beamSums));
    if (!beamSums)
      gpuRequiredFailure("unable to allocate CCBEND centroid sums");
  }

  gpuCopyHostToDevice(n_part);
  if (data.options.isrCoef || beamSums) {
    GPU_BEND_BATCH batch;

    memset(&batch, 0, sizeof(batch));
    batch.ccbend = &data;
    batch.beamSums = beamSums;
    batch.rows = data.nSlices + 1;
    gpuTrackInBatches(n_part, data.options.isrCoef ? data.nSlices : 0,
                      beamSums ? batch.rows * GPU_BEND_RECORD_SIZE : 0,
                      gpuCcbendBatchTrack, gpuBendBatchConsume, &batch,
                      "CCBEND tracking CUDA checked kernel", &lostCount, &milliseconds);
    data.options.isrDraws = NULL;
    data.options.sliceRecord = NULL;
  } else {
    /* no ISR and no slice records here, so a loss is compacted on the device */
    survivorFlags = gpuMagnetSurvivorFlags(n_part);
    status = gpuCudaCcbendTrackChecked(gpuBase.deviceCoord, n_part,
                                       (int)gpuBase.deviceStride, &data, survivorFlags,
                                       &lostCount, &milliseconds);
    if (status != 0)
      gpuFatalStatus("CCBEND tracking CUDA checked kernel", status);
  }
  gpuRecordMagnetKernel(milliseconds);
  if (lostCount) {
    free(beamSums);
    if (data.options.isrCoef) {
      gpuIsrAbandon();
      return gpuCcbendOnCpu(n_part, eptr, ccbend, Po, accepted, z_start,
                            sigmaDelta2, rootname, maxamp, apContour,
                            apFileData, iPart, iFinalSlice,
                            "CCBEND ISR particle-loss CPU fallback");
    }
#ifndef GPU_VERIFY
    if (gpuMagnetLossCompactionAllowed() && !recordSlices) {
      GPU_BEND_LOSS_ARGS lossArgs = {eptr, ccbend, Po, z_start, rootname, maxamp, apContour,
                                     apFileData, iPart, iFinalSlice};
      long remaining = survivorFlags ?
        gpuCompactTrackedMagnet(n_part, accepted, "CCBEND stable particle-loss compaction",
                                sigmaDelta2 ? NULL : gpuCcbendLossTrack, &lossArgs) :
        gpuCcbendStableCompact(&data, n_part, accepted,
                               "CCBEND stable particle-loss compaction",
                               sigmaDelta2 ? NULL : gpuCcbendLossTrack, &lossArgs);

      if (remaining >= 0) {
        if (&multipoleKicksDone)
          multipoleKicksDone += n_part * data.nSlices;
        return remaining;
      }
    }
#endif
    return gpuCcbendOnCpu(n_part, eptr, ccbend, Po, accepted, z_start,
                          sigmaDelta2, rootname, maxamp, apContour,
                          apFileData, iPart, iFinalSlice,
                          "CCBEND particle-loss CPU fallback");
  }
  if (data.options.isrCoef)
    gpuIsrCommit();
  if (beamSums) {
    ccbendWriteCentroidOutput(ccbend, eptr, beamSums, Po);
    free(beamSums);
  }
  gpuMarkDeviceChanged(n_part);
  gpuRecordWallSeconds();
  if (&multipoleKicksDone)
    multipoleKicksDone += n_part * data.nSlices;
  return n_part;
}

static long gpuPackLgbendTracking(
  GPU_LGBEND_DATA *data,
  GPU_LGBEND_LOCAL_APERTURE_DATA *localAperture,
  ELEMENT_LIST *eptr, LGBEND *lgbend, double Po) {
  long i, j;

  if (!data || !localAperture || !gpuLgbendCommonSupported(lgbend))
    return 0;
  if (optimizeLgbendForGpu) {
    long elementOnGpu = gpuBase.elementOnGpu;

    gpuBase.elementOnGpu = 0;
    optimizeLgbendForGpu(eptr, lgbend, Po);
    gpuBase.elementOnGpu = elementOnGpu;
  } else if (!lgbend->optimized && lgbend->optimizeFse && lgbend->angle != 0) {
    return 0;
  }
  memset(data, 0, sizeof(*data));
  memset(localAperture, 0, sizeof(*localAperture));
  data->nSegments = lgbend->nSegments;
  data->firstSegment = 0;
  data->lastSegment = lgbend->nSegments;
  data->nSlices = lgbend->nSlices;
  data->integrationOrder = lgbend->integration_order;
  data->edgeOrder = lgbend->edgeOrder;
  data->predrift = lgbend->predrift;
  data->postdrift = lgbend->postdrift;
  data->entryPosition = lgbend->segment[0].entryX;
  data->entryAngle = lgbend->segment[0].entryAngle;
  data->entryPlaneTan = tan(data->entryAngle);
  data->entryPlaneCos = cos(data->entryAngle);
  data->entryPlaneSin = sin(data->entryAngle);
  data->exitPosition = lgbend->segment[lgbend->nSegments - 1].exitX;
  data->exitAngle = lgbend->segment[lgbend->nSegments - 1].exitAngle;
  data->exitPlaneTan = tan(-data->exitAngle);
  data->exitPlaneCos = cos(-data->exitAngle);
  data->exitPlaneSin = sin(-data->exitAngle);
  gpuRotationTiltSinCos(lgbend->tilt, &data->cosTilt, &data->sinTilt);
  data->apertureDx = lgbend->dx;
  data->apertureDy = lgbend->dy;
  if (lgbend->tilt) {
    data->apertureDx = lgbend->dx * cos(lgbend->tilt) + lgbend->dy * sin(lgbend->tilt);
    data->apertureDy = -lgbend->dx * sin(lgbend->tilt) + lgbend->dy * cos(lgbend->tilt);
  }
  for (i = 0; i < lgbend->nSegments; i++)
    data->zOffset[i] = lgbend->predrift * cos(lgbend->segment[0].entryAngle) +
                       (i > 0 ? lgbend->segment[i - 1].zAccumulated : 0);
  data->misaligned = lgbend->dx || lgbend->dy || lgbend->dz || lgbend->eyaw ||
                     lgbend->epitch || lgbend->etilt;
  if (data->misaligned &&
      !gpuPackExactMisalignment(&data->exactMisalignment, 4, data->apertureDx, data->apertureDy,
                                lgbend->dz * (lgbend->wasFlipped ? -1 : 1),
                                lgbend->epitch, lgbend->eyaw, lgbend->etilt, 0,
                                lgbend->segment[lgbend->nSegments - 1].zAccumulated, 1, 1))
    return 0;
  data->Po = Po;
  if (lgbend->synch_rad)
    data->radCoef =
      sqr(particleCharge) * pow3(Po) /
      (6 * PI * epsilon_o * sqr(c_mks) * particleMass);
  data->coordLimit = coordLimit;
  data->slopeLimit = slopeLimit;
  for (i = 0; i < lgbend->nSegments; i++) {
    LGBEND_SEGMENT *segment = lgbend->segment + i;
    double rho0, denominator, scale;
    denominator = sin(segment->entryAngle) +
                  sin(segment->angle - segment->entryAngle);
    rho0 = segment->length / denominator;
    denominator = 1 - lgbend->KnDelta[i];
    scale = 1 + lgbend->fse + lgbend->fseOpt[i];
    data->segment[i].length = segment->length;
    data->segment[i].entryAngle = segment->entryAngle;
    data->segment[i].exitAngle = segment->exitAngle;
    data->segment[i].fringe1Tan = tan(segment->entryAngle);
    data->segment[i].fringe1Sin = sin(segment->entryAngle);
    data->segment[i].fringe1Sec = 1.0 / cos(segment->entryAngle);
    data->segment[i].fringe2Tan = tan(segment->exitAngle);
    data->segment[i].fringe2Sin = sin(segment->exitAngle);
    data->segment[i].fringe2Sec = 1.0 / cos(segment->exitAngle);
    data->segment[i].invRho = 1 / rho0;
    data->segment[i].KnL[0] = scale / rho0 * segment->length;
    data->segment[i].KnL[1] =
      scale * segment->K1 * segment->length / denominator;
    data->segment[i].KnL[2] =
      scale * segment->K2 * segment->length / denominator;
    data->segment[i].K1 =
      data->segment[i].KnL[1] / segment->length;
    data->segment[i].angleSign = 1;
    data->segment[i].has1 = segment->has1 ? 1 : 0;
    data->segment[i].has2 = segment->has2 ? 1 : 0;
    data->segment[i].fringe1[0] = segment->fringeInt1K0;
    data->segment[i].fringe1[1] = segment->fringeInt1I0;
    data->segment[i].fringe1[2] = segment->fringeInt1K2;
    data->segment[i].fringe1[3] = segment->fringeInt1I1;
    data->segment[i].fringe1[4] = segment->fringeInt1K4;
    data->segment[i].fringe1[5] = segment->fringeInt1K5;
    data->segment[i].fringe1[6] = segment->fringeInt1K6;
    data->segment[i].fringe1[7] = segment->fringeInt1K7;
    data->segment[i].fringe2[0] = segment->fringeInt2K0;
    data->segment[i].fringe2[1] = segment->fringeInt2I0;
    data->segment[i].fringe2[2] = segment->fringeInt2K2;
    data->segment[i].fringe2[3] = segment->fringeInt2I1;
    data->segment[i].fringe2[4] = segment->fringeInt2K4;
    data->segment[i].fringe2[5] = segment->fringeInt2K5;
    data->segment[i].fringe2[6] = segment->fringeInt2K6;
    data->segment[i].fringe2[7] = segment->fringeInt2K7;
    if (!isfinite(data->segment[i].length))
      return 0;
    for (j = 0; j < 3; j++)
      if (!isfinite(data->segment[i].KnL[j]))
        return 0;
  }
  if (lgbend->localApertureData) {
    double dZOffset0;

    if (!interpolateApertureData ||
        lgbend->nSlices > GPU_LGBEND_MAX_LOCAL_APERTURE_SLICES)
      return 0;
    localAperture->present = 1;
    localAperture->nSegments = lgbend->nSegments;
    localAperture->nSlices = lgbend->nSlices;
    dZOffset0 =
      lgbend->predrift * cos(lgbend->segment[0].entryAngle);
    for (i = 0; i < lgbend->nSegments; i++) {
      double dZOffset =
        dZOffset0 +
        (i > 0 ? lgbend->segment[i - 1].zAccumulated : 0);
      double drift = lgbend->segment[i].length / lgbend->nSlices;

      for (j = 0; j <= lgbend->nSlices; j++) {
        GPU_LGBEND_LOCAL_APERTURE_POINT *point =
          &localAperture->point[i][j];
        if (!interpolateApertureData(
              dZOffset + j * drift, lgbend->localApertureData,
              &point->xCenter, &point->yCenter,
              &point->xMax, &point->yMax))
          return 0;
        if (!isfinite(point->xCenter) || !isfinite(point->yCenter) ||
            !isfinite(point->xMax) || !isfinite(point->yMax) ||
            point->xMax < 0 || point->yMax < 0)
          return 0;
      }
    }
  }
  return isfinite(data->entryPosition) && isfinite(data->entryAngle) &&
         isfinite(data->exitPosition) && isfinite(data->exitAngle) &&
         isfinite(data->coordLimit) && isfinite(data->slopeLimit);
}

#define GPU_LGBEND_MAX_ISR_DRAWS (1L << 27)

static int gpuLgbendSegmentBatchTrack(void *cookie, void *coord, long count, const double *draws,
                                      double *records, long *lostCount, float *milliseconds) {
  GPU_BEND_BATCH *batch = (GPU_BEND_BATCH *)cookie;

  batch->lgbend->options.isrDraws = draws;
  batch->lgbend->options.isrParticles = 0;
  batch->lgbend->options.isrFirstParticle = 0;
  batch->lgbend->options.sliceRecord = records;
  return gpuCudaLgbendTrackChecked(coord, count, (int)gpuBase.deviceStride, batch->lgbend,
                                   batch->localAperture, lostCount, milliseconds);
}

/* The CPU's random numbers go by segment, then particle, so a table too large for one launch is
 * made one segment at a time, and in batches of particles within a segment. */
static void gpuLgbendTrackBySegment(GPU_LGBEND_DATA *data,
                                    GPU_LGBEND_LOCAL_APERTURE_DATA *localAperture,
                                    GLOBAL_BEAM_SUMS *beamSums, long nParticles,
                                    long *lostCount, float *totalMilliseconds) {
  static void *start = NULL;
  static unsigned long startCount = 0;
  unsigned long count = (unsigned long)nParticles * (unsigned long)gpuBase.deviceStride;
  GPU_BEND_BATCH batch;
  long i;
  int status;

  if (count > startCount) {
    if (start)
      gpuCudaFree(start);
    start = NULL;
    startCount = 0;
    if (gpuCudaMallocDouble(&start, count) != 0)
      gpuRequiredFailure("unable to allocate the LGBEND segment scratch");
    startCount = count;
  }
  status = gpuCudaCopyDeviceToDevice(start, gpuBase.deviceCoord, count);
  if (status != 0)
    gpuFatalStatus("LGBEND segment copy", status);
  memset(&batch, 0, sizeof(batch));
  batch.lgbend = data;
  batch.localAperture = localAperture;
  batch.beamSums = beamSums;
  batch.rows = beamSums ? data->nSlices + 1 : 0;
  *totalMilliseconds = 0;
  for (i = 0; i < data->nSegments; i++) {
    float milliseconds = 0;

    data->firstSegment = i;
    data->lastSegment = i + 1;
    batch.rowOffset = i * (data->nSlices + 1);
    gpuTrackInBatches(nParticles, data->nSlices, batch.rows * GPU_BEND_RECORD_SIZE,
                      gpuLgbendSegmentBatchTrack, gpuBendBatchConsume, &batch,
                      "LGBEND tracking CUDA checked kernel", lostCount, &milliseconds);
    *totalMilliseconds += milliseconds;
    if (*lostCount) {
      status = gpuCudaCopyDeviceToDevice(gpuBase.deviceCoord, start, count);
      if (status != 0)
        gpuFatalStatus("LGBEND segment restore", status);
      break;
    }
  }
  data->firstSegment = 0;
  data->lastSegment = data->nSegments;
  data->options.sliceRecord = NULL;
}

static int gpuLgbendBatchTrack(void *cookie, void *coord, long count, const double *draws,
                               double *records, long *lostCount, float *milliseconds) {
  GPU_BEND_BATCH *batch = (GPU_BEND_BATCH *)cookie;

  (void)draws;
  batch->lgbend->options.isrFirstParticle =
    ((double *)coord - (double *)gpuBase.deviceCoord) / gpuBase.deviceStride;
  batch->lgbend->options.sliceRecord = records;
  return gpuCudaLgbendTrackChecked(coord, count, (int)gpuBase.deviceStride, batch->lgbend,
                                   batch->localAperture, lostCount, milliseconds);
}

/* An aperture_data file is evaluated once at each segment's center, and may override MAXAMP */
static long gpuPackLgbendFileApertures(GPU_LGBEND_LOCAL_APERTURE_DATA *localAperture,
                                       GPU_LGBEND_DATA *data, LGBEND *lgbend, MAXAMP *maxamp,
                                       APERTURE_DATA *apFileData, double zStart,
                                       ELEMENT_LIST *eptr) {
  long i;

  if (!apFileData || !apFileData->initialized)
    return 1;
  localAperture->hasFileAperture = 1;
  localAperture->nSegments = lgbend->nSegments;
  localAperture->nSlices = lgbend->nSlices;
  for (i = 0; i < lgbend->nSegments; i++) {
    localAperture->fileAperture[i] = data->aperture;
    if (!gpuPackFileAperture(&localAperture->fileAperture[i], maxamp, apFileData,
                             zStart + lgbend->segment[i].arcLengthStart + lgbend->segment[i].arcLength / 2,
                             eptr))
      return 0;
  }
  return 1;
}

static long gpuLgbendOnCpu(long n_part, ELEMENT_LIST *eptr, LGBEND *lgbend,
                           double Po, double **accepted, double z_start,
                           double *sigmaDelta2, char *rootname,
                           MAXAMP *maxamp, APCONTOUR *apContour,
                           APERTURE_DATA *apFileData, long iPart,
                           long iFinalSlice, const char *reason) {
  double **coord = forceParticlesToCpu(reason);
  long remaining;

  if (!track_through_lgbend)
    gpuRequiredFailure("CPU track_through_lgbend fallback is unavailable");
  gpuBase.elementOnGpu = 0;
  remaining = track_through_lgbend(coord, n_part, eptr, lgbend, Po, accepted,
                                   z_start, sigmaDelta2, rootname, maxamp,
                                   apContour, apFileData, iPart, iFinalSlice);
  gpuMarkHostWillChange();
  gpuRecordWallSeconds();
  return remaining;
}

long gpu_track_through_lgbend(long n_part, void *eptr0, void *lgbend0,
                              double Po, double **accepted, double z_start,
                              double *sigmaDelta2, char *rootname,
                              void *maxamp0, void *apContour0,
                              void *apFileData0, long iPart,
                              long iFinalSlice) {
  ELEMENT_LIST *eptr = (ELEMENT_LIST *)eptr0;
  LGBEND *lgbend = (LGBEND *)lgbend0;
  MAXAMP *maxamp = (MAXAMP *)maxamp0;
  APCONTOUR *apContour = (APCONTOUR *)apContour0;
  APERTURE_DATA *apFileData = (APERTURE_DATA *)apFileData0;
  GPU_LGBEND_DATA data;
  GPU_LGBEND_LOCAL_APERTURE_DATA localAperture;
  GLOBAL_BEAM_SUMS *beamSums = NULL;
  long lostCount = 0, recordSlices = 0, isrBySegment = 0;
  float milliseconds = 0;
  int status;

  if (n_part <= 0)
    return n_part;
  if (!lgbend)
    gpuRequiredFailure("NULL LGBEND pointer in gpu_track_through_lgbend");
  if (gpuBase.backtrack || iPart >= 0 || iFinalSlice > 0)
    return gpuLgbendOnCpu(n_part, eptr, lgbend, Po, accepted, z_start,
                          sigmaDelta2, rootname, maxamp, apContour,
                          apFileData, iPart, iFinalSlice,
                          "LGBEND partial/backtracking CPU fallback");
  if (sigmaDelta2)
    return gpuLgbendOnCpu(n_part, eptr, lgbend, Po, accepted, z_start,
                          sigmaDelta2, rootname, maxamp, apContour,
                          apFileData, iPart, iFinalSlice,
                          "LGBEND radiation-sigma CPU fallback");
  lgbendSetupCentroidOutput(lgbend);
  if (!gpuPackLgbendTracking(&data, &localAperture, eptr, lgbend, Po))
    return gpuLgbendOnCpu(n_part, eptr, lgbend, Po, accepted, z_start,
                          sigmaDelta2, rootname, maxamp, apContour,
                          apFileData, iPart, iFinalSlice,
                          "LGBEND unsupported-option CPU fallback");
  if (!gpuPackMaxamp(&data.aperture, maxamp, data.cosTilt, data.sinTilt) ||
      !gpuPackLgbendFileApertures(&localAperture, &data, lgbend, maxamp, apFileData, z_start, eptr) ||
      !gpuPackApertureContour(&data.options.contour, apContour))
    return gpuLgbendOnCpu(n_part, eptr, lgbend, Po, accepted, z_start,
                          sigmaDelta2, rootname, maxamp, apContour,
                          apFileData, iPart, iFinalSlice,
                          "LGBEND aperture CPU fallback");
  /* Under GPU_VERIFY the CPU tracks these particles again and writes the file */
#ifndef GPU_VERIFY
  recordSlices = lgbendCentroidSumsEnabled(lgbend, iPart, iFinalSlice);
#endif
  gpuPackBendPlainOptions(&data.options, eptr, Po, n_part, lgbend->isr, lgbend->isr1Particle,
                          lgbend->length, lgbend->xEntry, 1.0, data.nSlices, recordSlices);
  /* The random numbers go by segment first, so a table too large for one launch runs by segment */
  isrBySegment = data.options.isrCoef &&
                 n_part * data.nSegments * data.nSlices >
                   gpuEnvLong("ELEGANT_GPU_LGBEND_ISR_TABLE_DRAWS", GPU_LGBEND_MAX_ISR_DRAWS);
  if (data.options.isrCoef)
    gpuIsrBegin();
  if (recordSlices) {
    beamSums = (GLOBAL_BEAM_SUMS *)calloc(data.nSegments * (data.nSlices + 1), sizeof(*beamSums));
    if (!beamSums)
      gpuRequiredFailure("unable to allocate LGBEND centroid sums");
  }

  gpuCopyHostToDevice(n_part);
  if (data.options.isrCoef && !isrBySegment) {
    data.options.isrDraws = gpuIsrFillDraws(n_part * data.nSegments * data.nSlices);
    data.options.isrParticles = n_part;
  }
  if (isrBySegment) {
    gpuLgbendTrackBySegment(&data, localAperture.present || localAperture.hasFileAperture ?
                                     &localAperture : NULL,
                            beamSums, n_part, &lostCount, &milliseconds);
  } else if (beamSums) {
    GPU_BEND_BATCH batch;

    memset(&batch, 0, sizeof(batch));
    batch.lgbend = &data;
    batch.localAperture = localAperture.present || localAperture.hasFileAperture ? &localAperture : NULL;
    batch.beamSums = beamSums;
    batch.rows = data.nSegments * (data.nSlices + 1);
    gpuTrackInBatches(n_part, 0, batch.rows * GPU_BEND_RECORD_SIZE,
                      gpuLgbendBatchTrack, gpuBendBatchConsume, &batch,
                      "LGBEND tracking CUDA checked kernel", &lostCount, &milliseconds);
    data.options.sliceRecord = NULL;
  } else {
    status = gpuCudaLgbendTrackChecked(gpuBase.deviceCoord, n_part,
                                       (int)gpuBase.deviceStride, &data,
                                       localAperture.present || localAperture.hasFileAperture ?
                                         &localAperture : NULL,
                                       &lostCount, &milliseconds);
    if (status != 0)
      gpuFatalStatus("LGBEND tracking CUDA checked kernel", status);
  }
  gpuRecordMagnetKernel(milliseconds);
  if (lostCount) {
    free(beamSums);
    if (data.options.isrCoef) {
      gpuIsrAbandon();
      return gpuLgbendOnCpu(n_part, eptr, lgbend, Po, accepted, z_start,
                            sigmaDelta2, rootname, maxamp, apContour,
                            apFileData, iPart, iFinalSlice,
                            "LGBEND ISR particle-loss CPU fallback");
    }
#ifndef GPU_VERIFY
    if (gpuMagnetLossCompactionAllowed() && !recordSlices) {
      GPU_BEND_LOSS_ARGS lossArgs = {eptr, lgbend, Po, z_start, rootname, maxamp, apContour,
                                     apFileData, iPart, iFinalSlice};
      long remaining =
        gpuLgbendStableCompact(&data, &localAperture, n_part, accepted,
                               "LGBEND stable particle-loss compaction",
                               sigmaDelta2 ? NULL : gpuLgbendLossTrack, &lossArgs);

      if (remaining >= 0) {
        if (&multipoleKicksDone)
          multipoleKicksDone +=
            n_part * data.nSlices * data.nSegments;
        return remaining;
      }
    }
#endif
    return gpuLgbendOnCpu(n_part, eptr, lgbend, Po, accepted, z_start,
                          sigmaDelta2, rootname, maxamp, apContour,
                          apFileData, iPart, iFinalSlice,
                          "LGBEND particle-loss CPU fallback");
  }
  if (data.options.isrCoef)
    gpuIsrCommit();
  if (beamSums) {
    lgbendWriteCentroidOutput(lgbend, eptr, beamSums, Po, data.nSegments);
    free(beamSums);
  }
  gpuMarkDeviceChanged(n_part);
  gpuRecordWallSeconds();
  if (&multipoleKicksDone)
    multipoleKicksDone += n_part * data.nSlices * data.nSegments;
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
  /* Models 1 to 5.  With HGAP=0 the CPU stops for model 2 at EDGE_ORDER>1, and model 4 divides by
   * the gap, so those stay there */
  if (flags & BEND_EDGE1_EFFECTS) {
    long effect = csbend->edge_effects[csbend->e1Index];

    if (effect < 1 || effect > 5 ||
        (csbend->hgap == 0 && (effect == 4 || (effect == 2 && csbend->edge_order > 1))))
      return 0;
  }
  if (flags & BEND_EDGE2_EFFECTS) {
    long effect = csbend->edge_effects[csbend->e2Index];

    if (effect < 1 || effect > 5 ||
        (csbend->hgap == 0 && (effect == 4 || (effect == 2 && csbend->edge_order > 1))))
      return 0;
  }
  return 1;
}

/* csbend.c's own sine and cosine of the tilt.  It has no case for a full turn: there the CPU's CSBEND
 * rotates by the sine of 2 pi in double precision, -2.4e-16. */
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

/* The sine and cosine rotateBeamCoordinatesForMisalignment() uses, for every element that calls it
 * (all but CSBEND).  It doesn't rotate for a full turn, which is what a CCBEND with a negative angle
 * and TILT=pi has. */
static void gpuRotationTiltSinCos(double tilt, double *cosTilt, double *sinTilt) {
  if (fabs(fabs(tilt) - PIx2) < 1e-12) {
    *cosTilt = 1;
    *sinTilt = 0;
    return;
  }
  gpuCsbendTiltSinCos(tilt, cosTilt, sinTilt);
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
  data->invRho0 = 1. / rho;
  data->rhoActual = 1 / ((1 + fse) * h);
  gpuPackCsbendDriftSteps(data);
  data->Po = Po;
  if (csbend->synch_rad)
    data->radCoef =
      sqr(particleCharge) * pow3(Po) * sqr(1 + fse) /
      (6 * PI * epsilon_o * sqr(c_mks) * particleMass * sqr(rho));
  {
    /* track_through_csbend() works out its ISR constant whatever the flags, and with ISR off it negates
     * the constant and doesn't zero it.  So the radiation block of the integrators runs after every kick
     * of an ordinary CSBEND: it divides QX and QY by 1+delta and multiplies them back, which moves a
     * momentum by a unit in the last place with a chance of about |delta|.  The device does the same.
     * (CSRCSBEND zeroes its constant, so its body has no such step.) */
    double cpuIsrConstant =
      particleRadius * sqrt(55.0 / (24 * sqrt(3)) * pow5(Po) * 137.0359895 / pow3(fabs(data->rhoActual)));

    data->momentumRoundTrip = csbend->synch_rad || cpuIsrConstant ? 1 : 0;
  }
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
  data->edgeEffect1 = csbend->edge_effects[csbend->e1Index];
  data->edgeEffect2 = csbend->edge_effects[csbend->e2Index];
  data->edgeOrder = csbend->edge_order;
  data->edgeFlip = csbend->edgeFlip ? 1 : 0;
  data->e1 = e1;
  data->e2 = e2;
  /*
   * Use the coefficients prepared above, not csbend->b.  The host tracker
   * fills csbend->b lazily, so reading it here made the resident fringe map
   * depend on whether some earlier matrix calculation happened to initialize
   * the element.
   */
  data->K1 = b[1] / rho;
  memcpy(data->fringeInt1, csbend->fringeInt[csbend->e1Index],
         sizeof(data->fringeInt1));
  memcpy(data->fringeInt2, csbend->fringeInt[csbend->e2Index],
         sizeof(data->fringeInt2));
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
  data->fringeGap = 2 * csbend->hgap;
  data->fint1 = csbend->fint[csbend->e1Index] >= 0 ? csbend->fint[csbend->e1Index] : csbend->fintBoth;
  data->fint2 = csbend->fint[csbend->e2Index] >= 0 ? csbend->fint[csbend->e2Index] : csbend->fintBoth;
  data->edgeTrig1[0] = 1. / cos(e1);
  data->edgeTrig1[1] = tan(e1);
  data->edgeTrig1[2] = sin(e1);
  data->edgeTrig1[3] = cos(e1);
  data->edgeTrig2[0] = 1. / cos(e2);
  data->edgeTrig2[1] = tan(e2);
  data->edgeTrig2[2] = sin(e2);
  data->edgeTrig2[3] = cos(e2);
  if (csbend->malignMethod == 1 || csbend->malignMethod == 2) {
    data->hasLinearMisalignment = 1;
    if (!gpuPackLinearMisalignment(&data->misalignment, csbend->malignMethod,
                                   csbend->dx, csbend->dy, csbend->dz,
                                   csbend->epitch, csbend->eyaw, csbend->etilt,
                                   angle, csbend->length))
      return 0;
  } else if (csbend->malignMethod == 3 || csbend->malignMethod == 4) {
    data->hasExactMisalignment = 1;
    if (!gpuPackExactMisalignment(&data->exactMisalignment, csbend->malignMethod,
                                  csbend->dx, csbend->dy, csbend->dz,
                                  csbend->epitch, csbend->eyaw, csbend->etilt,
                                  angle, csbend->length, 1, 1))
      return 0;
  } else if (csbend->dx || csbend->dy || csbend->dz || csbend->etilt) {
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
  gpuCsbendTiltSinCos(data->hasLinearMisalignment || data->hasExactMisalignment ?
                        tilt : tilt + csbend->etilt * csbend->etiltSign,
                      &data->cosTilt, &data->sinTilt);
  memcpy(data->Fx, *Fx_xy, sizeof(data->Fx));
  memcpy(data->Fy, *Fy_xy, sizeof(data->Fy));
  if (spinCoordOffset) {
    data->spinOffset = spinCoordOffset;
    data->spinFieldConversion = -1 / data->rho0 / (particleCharge / (particleMass * c_mks * Po));
    data->spinH = 1 / data->rho0;
    data->spinCharge = -particleCharge * particleRelSign;
    data->spinMassC = particleMass * c_mks;
    data->spinAnomaly = particleAnomalousMagneticMoment;
  }
  return 1;
}

#ifndef GPU_VERIFY
typedef struct {
  CSBEND *csbend;
  double p_error, Po, z_start;
  char *rootname;
  MAXAMP *maxamp;
  APCONTOUR *apContour;
  APERTURE_DATA *apFileData;
  long iSlice;
  ELEMENT_LIST *eptr;
} GPU_CSBEND_LOSS_ARGS;

static long gpuCsbendLossTrack(double **rows, long nRows, void *cookie) {
  GPU_CSBEND_LOSS_ARGS *a = (GPU_CSBEND_LOSS_ARGS *)cookie;

  if (!track_through_csbend)
    return nRows;
  return track_through_csbend(rows, nRows, a->csbend, a->p_error, a->Po, NULL, a->z_start, NULL,
                              a->rootname, a->maxamp, a->apContour, a->apFileData, a->iSlice,
                              a->eptr);
}
#endif

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

static int gpuCsbendBatchTrack(void *cookie, void *coord, long count, const double *draws,
                               double *records, long *lostCount, float *milliseconds) {
  GPU_CSBEND_DATA *data = (GPU_CSBEND_DATA *)cookie;

  (void)records;
  data->isrDraws = draws;
  return gpuCudaCsbendTrackChecked(coord, count, (int)gpuBase.deviceStride, data, NULL, lostCount, milliseconds);
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
  void *survivorFlags = NULL;
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
  if (sigmaDelta2)
    return gpuCsbendOnCpu(n_part, csbend, p_error, Po, accepted, z_start,
                          sigmaDelta2, rootname, maxamp, apContour, apFileData,
                          iSlice, eptr, "track_through_csbend radiation sigma fallback");
  if (!gpuPackCsbendTracking(&data, csbend, Po))
    return gpuCsbendOnCpu(n_part, csbend, p_error, Po, accepted, z_start,
                          sigmaDelta2, rootname, maxamp, apContour, apFileData,
                          iSlice, eptr, "track_through_csbend unsupported option");
  if (!gpuPackMaxamp(&data.aperture, maxamp,
                     data.cosTilt, data.sinTilt) ||
      !gpuPackFileAperture(&data.aperture, maxamp, apFileData, z_start + csbend->length / 2, eptr))
    return gpuCsbendOnCpu(n_part, csbend, p_error, Po, accepted, z_start,
                          sigmaDelta2, rootname, maxamp, apContour, apFileData,
                          iSlice, eptr, "track_through_csbend MAXAMP or aperture_data fallback");
  if (!gpuPackApertureContour(&data.contour, apContour))
    return gpuCsbendOnCpu(n_part, csbend, p_error, Po, accepted, z_start,
                          sigmaDelta2, rootname, maxamp, apContour, apFileData,
                          iSlice, eptr, "CSBEND unsupported APCONTOUR CPU fallback");
  data.lossZStart = z_start;
  if (csbend->isr && (csbend->isr1Particle || n_part > 1)) {
    data.isrCoef = particleRadius * sqrt(55.0 / (24 * sqrt(3)) * pow5(Po) * 137.0359895 / pow3(fabs(data.rhoActual)));
    data.isrDrawsPerParticle = data.nSlices * gpuSymplecticKicksPerSlice(data.integrationOrder);
  }

  gpuCopyHostToDevice(n_part);
  if (data.isrCoef) {
    gpuIsrBegin();
    gpuTrackInBatches(n_part, data.isrDrawsPerParticle, 0, gpuCsbendBatchTrack, NULL, &data,
                      "CSBEND tracking CUDA checked kernel", &lostCount, &milliseconds);
    data.isrDraws = NULL;
  } else {
    survivorFlags = gpuMagnetSurvivorFlags(n_part);
    status = gpuCudaCsbendTrackChecked(gpuBase.deviceCoord, n_part,
                                       (int)gpuBase.deviceStride, &data, survivorFlags,
                                       &lostCount, &milliseconds);
    if (status != 0)
      gpuFatalStatus("CSBEND tracking CUDA checked kernel", status);
  }
  gpuRecordMagnetKernel(milliseconds);
  if (lostCount) {
    if (data.isrCoef) {
      gpuIsrAbandon();
      return gpuCsbendOnCpu(n_part, csbend, p_error, Po, accepted, z_start,
                            sigmaDelta2, rootname, maxamp, apContour, apFileData,
                            iSlice, eptr, "track_through_csbend ISR particle-loss fallback");
    }
#ifndef GPU_VERIFY
    if (gpuMagnetLossCompactionAllowed()) {
      GPU_CSBEND_LOSS_ARGS lossArgs = {csbend, p_error, Po, z_start, rootname, maxamp, apContour,
                                       apFileData, iSlice, eptr};
      long remaining = survivorFlags ?
        gpuCompactTrackedMagnet(n_part, accepted, "track_through_csbend stable magnet loss compaction",
                                gpuCsbendLossTrack, &lossArgs) :
        gpuCsbendStableCompact(
          &data, n_part, accepted,
          "track_through_csbend stable magnet loss compaction", gpuCsbendLossTrack, &lossArgs);

      if (remaining >= 0) {
        gpuDeviceIslandHasCsbend = 1;
        if (&multipoleKicksDone)
          multipoleKicksDone +=
            n_part * data.nSlices * (data.integrationOrder == 4 ? 4 : 1);
        return remaining;
      }
    }
#else
    {
      long remaining = gpuVerifyCsbendStableCompact(
        &data, n_part,
        "track_through_csbend GPU-VERIFY stable magnet loss compaction");
      long cpuRemaining;

      startCpuTimer();
      cpuRemaining = track_through_csbend(
        gpuBase.coord, n_part, csbend, p_error, Po, accepted, z_start,
        sigmaDelta2, rootname, maxamp, apContour, apFileData, iSlice, eptr);
      /* The CPU keeps a particle whose coordinates turn non-finite at the exit, and the
       * limit_amplitudes() after each element drops it, where the kernel drops it itself */
      if (limit_amplitudes)
        cpuRemaining = limit_amplitudes(gpuBase.coord, DBL_MAX, DBL_MAX, cpuRemaining, accepted,
                                        z_start + csbend->length, Po, 0, 0, eptr);
      if (cpuRemaining != remaining) {
        fprintf(stderr,
                "elegant CUDA VERIFY failed for CSBEND element=%s: CPU survivors=%ld GPU survivors=%ld\n",
                eptr && eptr->name ? eptr->name : "(unknown)",
                cpuRemaining, remaining);
        exit(1);
      }
      compareGpuCpuSurvivors(remaining, "track_through_csbend stable loss tracking");
      gpuRecordWallSeconds();
      return cpuRemaining;
    }
#endif
    return gpuCsbendOnCpu(n_part, csbend, p_error, Po, accepted, z_start,
                          sigmaDelta2, rootname, maxamp, apContour, apFileData,
                          iSlice, eptr, "track_through_csbend particle loss fallback");
  }
  if (data.isrCoef)
    gpuIsrCommit();
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

#ifndef GPU_VERIFY
typedef struct {
  void *map;
  double pRef, zStart;
} GPU_KICKMAP_LOSS_ARGS;

static long gpuKickMapLossTrack(double **rows, long nRows, void *cookie) {
  GPU_KICKMAP_LOSS_ARGS *a = (GPU_KICKMAP_LOSS_ARGS *)cookie;

  if (!trackKickMap)
    return nRows;
  return trackKickMap(rows, NULL, nRows, a->pRef, (KICKMAP *)a->map, a->zStart, NULL);
}

static long gpuUndulatorKickMapLossTrack(double **rows, long nRows, void *cookie) {
  GPU_KICKMAP_LOSS_ARGS *a = (GPU_KICKMAP_LOSS_ARGS *)cookie;

  if (!trackUndulatorKickMap)
    return nRows;
  return trackUndulatorKickMap(rows, NULL, nRows, a->pRef, (UKICKMAP *)a->map, a->zStart);
}
#endif

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
    if (gpuMagnetLossCompactionAllowed()) {
      GPU_KICKMAP_LOSS_ARGS lossArgs = {map, pRef, zStart};
      long remaining = gpuKickMapStableCompact(
        &data, nParticles, accepted, zStart, pRef, gpuKickMapCache.xpFactor,
        gpuKickMapCache.ypFactor, map->points,
        "KICKMAP stable map-loss compaction", sigmaDelta2 ? NULL : gpuKickMapLossTrack, &lossArgs);

      if (remaining >= 0)
        return remaining;
    }
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
    if (gpuMagnetLossCompactionAllowed()) {
      GPU_KICKMAP_LOSS_ARGS lossArgs = {map, pRef, zStart};
      long remaining = gpuKickMapStableCompact(
        &data, nParticles, accepted, zStart, pRef, gpuKickMapCache.xpFactor,
        gpuKickMapCache.ypFactor, map->points,
        "UKICKMAP stable map-loss compaction", gpuUndulatorKickMapLossTrack, &lossArgs);

      if (remaining >= 0)
        return remaining;
    }
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
  /* rows not in bunch order: ZLONGIT and ZTRANSVERSE gather them by bunch first */
  long needsGather;
} GPU_BUNCHED_WAKE_PLAN;

typedef struct GPU_BUNCHED_WAKE_PLAN_CACHE {
  const void *owner;
  long particles;
  long valid;
  GPU_BUNCHED_WAKE_PLAN plan;
} GPU_BUNCHED_WAKE_PLAN_CACHE;

static GPU_BUNCHED_WAKE_PLAN_CACHE gpuImpedancePlanCache;
static GPU_BUNCHED_WAKE_PLAN_CACHE gpuCwakePlanCache;
static GPU_BUNCHED_WAKE_PLAN_CACHE gpuZlongitPlanCache;
static GPU_BUNCHED_WAKE_PLAN_CACHE gpuZtransversePlanCache;
static GPU_BUNCHED_WAKE_PLAN_CACHE gpuIbsPlanCache;

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
      (strcmp(operation, "IMPEDANCE") == 0 || strcmp(operation, "CWAKE") == 0 ||
       strcmp(operation, "ZLONGIT") == 0 || strcmp(operation, "ZTRANSVERSE") == 0 ||
       strcmp(operation, "IBSCATTER") == 0)) {
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
      if (!sorted) {
        /* Losses and injections leave the rows out of bunch order.  ZLONGIT and ZTRANSVERSE then
         * gather them by bunch on the device (gpuZimpedanceGather()); the others go to the CPU. */
        if (!gpuEnableZimpedanceGather ||
            (strcmp(operation, "ZLONGIT") != 0 && strcmp(operation, "ZTRANSVERSE") != 0 &&
             strcmp(operation, "IBSCATTER") != 0))
          return GPU_BUNCHED_WAKE_UNSUPPORTED;
        plan->needsGather = 1;
      } else {
        gpuBunchRangeCache.deviceCoord = gpuBase.deviceCoord;
        gpuBunchRangeCache.particles = np;
        gpuBunchRangeCache.stride = gpuBase.deviceStride;
        gpuBunchRangeCache.coordinateIndex = bunchIndex;
        gpuBunchRangeCache.minBunch = plan->minBunch;
        gpuBunchRangeCache.maxBunch = plan->maxBunch;
        gpuBunchRangeCache.nBuckets = plan->nBuckets;
        gpuBunchRangeCache.valid = 1;
      }
    }
    if (!plan->needsGather) {
      plan->bucketStart = gpuBunchRangeCache.start;
      plan->bucketCount = gpuBunchRangeCache.count;
    }
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

double gpu_tfeedback_pickup_average(long np, long coordinate) {
  double average;

  if (np <= 0)
    return 0;
  if (coordinate < 0 || coordinate > 5)
    gpuRequiredFailure("invalid TFBPICKUP CUDA coordinate");
  startGpuTimer();
  average = gpuCoordinateSumOnDevice(np, coordinate) / np;
#ifdef GPU_VERIFY
  if (gpuBase.verifyMode) {
    double **coord = copyParticlesToCpuReadOnly(
      "TFBPICKUP reduction verification input");
    double cpuAverage = 0, absDiff, relDiff;
    long ip;
    for (ip = 0; ip < np; ip++)
      cpuAverage += coord[ip][coordinate];
    cpuAverage /= np;
    if (!gpuValuesClose(
          cpuAverage, average,
          gpuEnvDouble("ELEGANT_GPU_TFEEDBACK_COMPARE_ABS", 1e-15),
          gpuEnvDouble("ELEGANT_GPU_TFEEDBACK_COMPARE_REL", 1e-9),
          &absDiff, &relDiff)) {
      fprintf(stderr,
              "elegant CUDA VERIFY mismatch for TFBPICKUP average: CPU=%.17g GPU=%.17g abs=%.3e rel=%.3e\n",
              cpuAverage, average, absDiff, relDiff);
      gpuRequiredFailure("TFBPICKUP CUDA reduction verification failed");
    }
  }
#endif
  gpuRecordWallSeconds();
  return average;
}

void gpu_tfeedback_apply_kick(long np, long pickupCoordinate,
                              long longitudinal, double kick) {
  float milliseconds = 0;
  int status;

  if (np <= 0 || kick == 0)
    return;
  if (!longitudinal && pickupCoordinate != 0 && pickupCoordinate != 2)
    gpuRequiredFailure("invalid transverse TFBDRIVER CUDA pickup plane");
  startGpuTimer();
  gpuCopyHostToDevice(np);
#ifdef GPU_VERIFY
  if (gpuBase.verifyMode) {
    double **coord = copyParticlesToCpuReadOnly(
      "TFBDRIVER kick verification input");
    long ip;
    for (ip = 0; ip < np; ip++) {
      if (longitudinal)
        coord[ip][5] += kick;
      else
        coord[ip][pickupCoordinate + 1] += kick / (1 + coord[ip][5]);
    }
  }
#endif
  status = gpuCudaTfeedbackKick(
    gpuBase.deviceCoord, np, (int)gpuBase.deviceStride,
    (int)pickupCoordinate, longitudinal ? 1 : 0, kick, &milliseconds);
  if (status != 0)
    gpuFatalStatus("TFBDRIVER kick CUDA kernel", status);
  gpuRecordHelperKernel(milliseconds);
  gpuMarkDeviceChanged(np);
#ifdef GPU_VERIFY
  if (gpuBase.verifyMode)
    compareGpuCpu(np, "deterministic TFBDRIVER kick");
#endif
  gpuRecordWallSeconds();
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

long gpu_trfmode_histogram(long np, double pCentral,
                           long bins, double tmin, double dt,
                           double dx, double dy,
                           double *xsum, double *ysum,
                           unsigned long *histogram,
                           long *firstBin, long *lastBin) {
  GPU_RFMODE_DATA data;
  long ib, binnedCount = 0;
  float milliseconds = 0;
  int status;

  if (np <= 0 || bins < 2 || dt <= 0 || !xsum || !ysum || !histogram ||
      !firstBin || !lastBin)
    gpuRequiredFailure("invalid TRFMODE/FTRFMODE CUDA histogram request");
  if (!gpuRfmodeHostHistogram || !gpuRfmodeHostXsum ||
      !gpuRfmodeHostYsum || gpuRfmodeHostHistogramCapacity < bins) {
    unsigned long long *newHistogram = (unsigned long long *)realloc(
      gpuRfmodeHostHistogram, bins * sizeof(*newHistogram));
    double *newXsum = (double *)realloc(
      gpuRfmodeHostXsum, bins * sizeof(*newXsum));
    double *newYsum = (double *)realloc(
      gpuRfmodeHostYsum, bins * sizeof(*newYsum));
    if (!newHistogram || !newXsum || !newYsum)
      gpuRequiredFailure("unable to allocate TRFMODE/FTRFMODE host histogram");
    gpuRfmodeHostHistogram = newHistogram;
    gpuRfmodeHostXsum = newXsum;
    gpuRfmodeHostYsum = newYsum;
    gpuRfmodeHostHistogramCapacity = bins;
  }
  memset(&data, 0, sizeof(data));
  data.bins = bins;
  data.tmin = tmin;
  data.dt = dt;
  data.pCentral = pCentral;
  data.cMks = c_mks;
  data.dx = dx;
  data.dy = dy;
  startGpuTimer();
  gpuCopyHostToDevice(np);
  status = gpuCudaTrfmodeHistogram(
    gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, &data,
    gpuRfmodeHostHistogram, gpuRfmodeHostXsum, gpuRfmodeHostYsum,
    &binnedCount, &milliseconds);
  if (status != 0)
    gpuFatalStatus("TRFMODE/FTRFMODE histogram CUDA kernel", status);
  gpuRecordWakeKernel(milliseconds);
  *firstBin = bins;
  *lastBin = -1;
  for (ib = 0; ib < bins; ib++) {
    if (gpuRfmodeHostHistogram[ib] > (unsigned long long)ULONG_MAX)
      gpuRequiredFailure("TRFMODE/FTRFMODE histogram count overflow");
    histogram[ib] = (unsigned long)gpuRfmodeHostHistogram[ib];
    xsum[ib] = gpuRfmodeHostXsum[ib];
    ysum[ib] = gpuRfmodeHostYsum[ib];
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
      "TRFMODE/FTRFMODE histogram verification input");
    double *cpuXsum = (double *)calloc(bins, sizeof(*cpuXsum));
    double *cpuYsum = (double *)calloc(bins, sizeof(*cpuYsum));
    unsigned long *cpuHistogram = (unsigned long *)calloc(
      bins, sizeof(*cpuHistogram));
    double absTol = gpuCompareAbsTolerance();
    double relTol = gpuCompareRelTolerance();
    long cpuBinned = 0;
    if (!cpuXsum || !cpuYsum || !cpuHistogram)
      gpuRequiredFailure("unable to allocate TRFMODE/FTRFMODE verification histogram");
    for (long ip = 0; ip < np; ip++) {
      double time = gpuRfmodeCpuTime(coord[ip], pCentral);
      long bin = (long)((time - tmin) / dt);
      if (bin < 0 || bin >= bins)
        continue;
      cpuHistogram[bin]++;
      cpuXsum[bin] += coord[ip][0] - dx;
      cpuYsum[bin] += coord[ip][2] - dy;
      cpuBinned++;
    }
    for (ib = 0; ib < bins; ib++) {
      double absDiff, relDiff;
      if (cpuHistogram[ib] != histogram[ib] ||
          !gpuValuesClose(cpuXsum[ib], xsum[ib], absTol, relTol,
                          &absDiff, &relDiff) ||
          !gpuValuesClose(cpuYsum[ib], ysum[ib], absTol, relTol,
                          &absDiff, &relDiff)) {
        fprintf(stderr,
                "elegant CUDA VERIFY mismatch TRFMODE/FTRFMODE histogram bin=%ld countCpu=%lu countGpu=%lu xCpu=%.17e xGpu=%.17e yCpu=%.17e yGpu=%.17e\n",
                ib, cpuHistogram[ib], histogram[ib], cpuXsum[ib], xsum[ib],
                cpuYsum[ib], ysum[ib]);
        exit(1);
      }
    }
    free(cpuHistogram);
    free(cpuXsum);
    free(cpuYsum);
    if (cpuBinned != binnedCount)
      gpuRequiredFailure("TRFMODE/FTRFMODE verification binned-count mismatch");
    if (gpuVerbose)
      fprintf(stderr,
              "elegant CUDA VERIFY passed for TRFMODE/FTRFMODE histogram: %ld particles in %ld bins\n",
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

void gpu_trfmode_apply_kicks(long np, double pCentral,
                             long bins, double tmin, double dt,
                             long firstBin, long lastBin,
                             long interpolate, long nCavities,
                             const double *voltageX,
                             const double *voltageY,
                             const double *voltageZ) {
  GPU_RFMODE_DATA data;
  float milliseconds = 0;
  int status;

  if (np <= 0 || bins < 2 || dt <= 0 || firstBin < 0 ||
      lastBin < firstBin || lastBin >= bins || !voltageX || !voltageY ||
      !voltageZ)
    gpuRequiredFailure("invalid TRFMODE/FTRFMODE CUDA kick request");
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
      "TRFMODE/FTRFMODE kick verification input");
    for (long ip = 0; ip < np; ip++) {
      double time = gpuRfmodeCpuTime(coord[ip], pCentral);
      double Vx, Vy, Vz, P, Px, Py, Pz;
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
        Vx = voltageX[ib1] + (voltageX[ib2] - voltageX[ib1]) / dt * dt1;
        Vy = voltageY[ib1] + (voltageY[ib2] - voltageY[ib1]) / dt * dt1;
        Vz = voltageZ[ib1] + (voltageZ[ib2] - voltageZ[ib1]) / dt * dt1;
      } else {
        Vx = voltageX[ib];
        Vy = voltageY[ib];
        Vz = voltageZ[ib];
      }
      P = pCentral * (1 + coord[ip][5]);
      Pz = P / sqrt(1 + sqr(coord[ip][1]) + sqr(coord[ip][3])) +
        nCavities * Vz / (1e6 * particleMassMV * particleRelSign);
      Px = coord[ip][1] * Pz +
        nCavities * Vx / (1e6 * particleMassMV * particleRelSign);
      Py = coord[ip][3] * Pz +
        nCavities * Vy / (1e6 * particleMassMV * particleRelSign);
      P = sqrt(Pz * Pz + Px * Px + Py * Py);
      coord[ip][1] = Px / Pz;
      coord[ip][3] = Py / Pz;
      coord[ip][5] = (P - pCentral) / pCentral;
      coord[ip][4] = time * c_mks * P / sqrt(P * P + 1);
    }
  }
#endif
  status = gpuCudaTrfmodeApplyKicks(
    gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, &data,
    voltageX, voltageY, voltageZ, &milliseconds);
  if (status != 0)
    gpuFatalStatus("TRFMODE/FTRFMODE kick CUDA kernel", status);
  gpuRecordWakeKernel(milliseconds);
  gpuMarkDeviceChanged(np);
#ifdef GPU_VERIFY
  if (gpuBase.verifyMode)
    compareGpuCpu(np, "resident TRFMODE/FTRFMODE kick");
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

/* ZLONGIT and ZTRANSVERSE choose their bunches as IMPEDANCE does: index_bunch_assignments() and the
 * START_BUNCH/END_BUNCH test on the bucket number */
static long gpuZimpedanceBunchedModeAction(long np, long bunchedBeamMode, long startBunch,
                                           long endBunch, void *charge0, const void *owner,
                                           const char *operation,
                                           GPU_BUNCHED_WAKE_PLAN_CACHE *cache) {
  GPU_BUNCHED_WAKE_PLAN plan;
  long action;

  cache->valid = 0;
  action = gpuBunchedWakePlan(np, bunchedBeamMode, startBunch, endBunch,
                              (CHARGE *)charge0, operation, &plan);
  if (!gpuEnableCombinedWakeMultibunch &&
      action != GPU_BUNCHED_WAKE_UNSUPPORTED && plan.nBuckets > 1)
    return GPU_BUNCHED_WAKE_UNSUPPORTED;
  if (action != GPU_BUNCHED_WAKE_UNSUPPORTED) {
    cache->owner = owner;
    cache->particles = np;
    cache->plan = plan;
    cache->valid = 1;
  }
  return action;
}

long gpu_zlongit_bunched_mode_action(long np, void *zlongit0, void *charge0) {
  ZLONGIT *zlongit = (ZLONGIT *)zlongit0;

  if (!zlongit)
    return GPU_BUNCHED_WAKE_UNSUPPORTED;
  return gpuZimpedanceBunchedModeAction(np, zlongit->bunchedBeamMode, zlongit->startBunch,
                                        zlongit->endBunch, charge0, zlongit, "ZLONGIT",
                                        &gpuZlongitPlanCache);
}

long gpu_ztransverse_bunched_mode_action(long np, void *ztransverse0, void *charge0) {
  ZTRANSVERSE *ztransverse = (ZTRANSVERSE *)ztransverse0;

  if (!ztransverse)
    return GPU_BUNCHED_WAKE_UNSUPPORTED;
  return gpuZimpedanceBunchedModeAction(np, ztransverse->bunchedBeamMode, ztransverse->startBunch,
                                        ztransverse->endBunch, charge0, ztransverse, "ZTRANSVERSE",
                                        &gpuZtransversePlanCache);
}

/* IBSCATTER works on every bunch (index_bunch_assignments() buckets, no bunch range) */
void gpu_beam_s_range(long np, double *sMin, double *sMax) {
  GPU_DOUBLE_MIN_MAX_DATA range;
  float milliseconds = 0;
  int status;

  *sMin = DBL_MAX;
  *sMax = -DBL_MAX;
  if (np <= 0)
    return;
  gpuCopyHostToDevice(np);
  memset(&range, 0, sizeof(range));
  status = gpuCudaDoubleMinMax(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, 4, &range, NULL,
                               &milliseconds);
  if (status != 0)
    gpuFatalStatus("s range reduction kernel", status);
  gpuRecordReductionKernel(milliseconds);
  if (range.count > 0) {
    *sMin = range.min;
    *sMax = range.max;
  }
#ifdef GPU_VERIFY
  if (gpuBase.verifyMode) {
    double **coord = copyParticlesToCpuReadOnly("GPU_VERIFY s range");
    double cpuMin = DBL_MAX, cpuMax = -DBL_MAX;
    long ip;
    for (ip = 0; ip < np; ip++) {
      if (coord[ip][4] > cpuMax)
        cpuMax = coord[ip][4];
      if (coord[ip][4] < cpuMin)
        cpuMin = coord[ip][4];
    }
    compareGpuCpuValue(*sMin, cpuMin, 0.0, "s range minimum");
    compareGpuCpuValue(*sMax, cpuMax, 0.0, "s range maximum");
  }
#endif
}

/* track_through_lrwake() with the beam on the device: index_bunch_assignments()'s bunch range from the device,
 * each particle's time, x, y and bunch copied to the host and added per bunch in particle order as the CPU
 * adds them, lrwakeBunchVoltages() (elegant's own) for the history and wakes, the kicks on the device.  The
 * particles are elegant's to the bit.  Returns 0 for the CPU to do the element (it gives the messages). */
long gpu_track_through_lrwake(long np, void *lrwake0, double P0, void *run0, long iPass, void *charge0,
                              long *lastNBunches) {
  static double *packed = NULL, *work = NULL;
  static long packedCapacity = 0, workCapacity = 0;
  LRWAKE *wake = (LRWAKE *)lrwake0;
  CHARGE *charge = (CHARGE *)charge0;
  long nBunches = 1, ibMin = 0, ib, ip, bunchColumn = -1;
  double *tBunch, *xBunch, *yBunch, *QBunch, *voltage, rampFactor;
  float milliseconds = 0;
  int status;
#ifdef GPU_VERIFY
  double *history[4];
  long ih;
#endif

  if (np <= 0 || !wake || !charge || !lastNBunches || trajectoryTracking || !set_up_lrwake ||
      !lrwakeBunchVoltages)
    return 0;
  startGpuTimer();
  gpuCopyHostToDevice(np);
  if (charge->idSlotsPerBunch > 0) {
    GPU_LONG_MIN_MAX_DATA range;

    if (bunchIndex >= totalPropertiesPerParticle || bunchIndex >= gpuBase.deviceStride)
      return 0;
    memset(&range, 0, sizeof(range));
    status = gpuCudaLongMinMax(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, bunchIndex, &range,
                               &milliseconds);
    if (status != 0)
      gpuFatalStatus("LRWAKE bunch-index reduction kernel", status);
    gpuRecordReductionKernel(milliseconds);
    if (range.count != np || range.max < range.min)
      return 0;
    ibMin = range.min;
    nBunches = range.max - range.min + 1;
    if (nBunches > 1)
      bunchColumn = bunchIndex;
  }
  /* index_bunch_assignments() stops the run if the number of bunches grew */
  if (*lastNBunches > 0 && nBunches > *lastNBunches)
    return 0;
  *lastNBunches = nBunches;
  set_up_lrwake(wake, (RUN *)run0, iPass, np, charge, nBunches);
  if (iPass >= (wake->rampPasses - 1))
    rampFactor = 1;
  else
    rampFactor = (iPass + 1.0) / wake->rampPasses;

  if (packedCapacity < 4 * np) {
    free(packed);
    if (!(packed = (double *)malloc(4 * np * sizeof(*packed))))
      gpuRequiredFailure("unable to allocate the LRWAKE particle values");
    packedCapacity = 4 * np;
  }
  if (workCapacity < 9 * nBunches) {
    free(work);
    if (!(work = (double *)malloc(9 * nBunches * sizeof(*work))))
      gpuRequiredFailure("unable to allocate the LRWAKE bunch sums");
    workCapacity = 9 * nBunches;
  }
  milliseconds = 0;
  status = gpuCudaLrwakeValues(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, P0, c_mks,
                               (int)bunchColumn, packed, &milliseconds);
  if (status != 0)
    gpuFatalStatus("LRWAKE particle values", status);
  gpuRecordReductionKernel(milliseconds);
  tBunch = work;
  xBunch = work + nBunches;
  yBunch = work + 2 * nBunches;
  QBunch = work + 3 * nBunches;
  voltage = work + 4 * nBunches;
  for (ib = 0; ib < 4 * nBunches; ib++)
    work[ib] = 0;
  /* track_through_lrwake()'s loop */
  for (ip = 0; ip < np; ip++) {
    const double *value = packed + 4 * ip;
    ib = nBunches == 1 ? 0 : (long)(value[3] - ibMin);
    if (ib < 0 || ib >= nBunches)
      gpuRequiredFailure("LRWAKE particle outside the bunch range");
    tBunch[ib] += value[0];
    QBunch[ib] += 1;
    xBunch[ib] += value[1];
    yBunch[ib] += value[2];
  }
#ifdef GPU_VERIFY
  /* The CPU redoes the element and must find the history as it was */
  history[0] = wake->tHistory;
  history[1] = wake->QHistory;
  history[2] = wake->xHistory;
  history[3] = wake->yHistory;
  wake->tHistory = (double *)malloc(sizeof(double) * wake->nHistory);
  wake->QHistory = (double *)malloc(sizeof(double) * wake->nHistory);
  wake->xHistory = (double *)malloc(sizeof(double) * wake->nHistory);
  wake->yHistory = (double *)malloc(sizeof(double) * wake->nHistory);
  for (ih = 0; ih < wake->nHistory; ih++) {
    wake->tHistory[ih] = history[0][ih];
    wake->QHistory[ih] = history[1][ih];
    wake->xHistory[ih] = history[2][ih];
    wake->yHistory[ih] = history[3][ih];
  }
#endif
  lrwakeBunchVoltages(wake, charge, nBunches, tBunch, xBunch, yBunch, QBunch, voltage, voltage + nBunches,
                      voltage + 2 * nBunches, voltage + 3 * nBunches, voltage + 4 * nBunches);
#ifdef GPU_VERIFY
  free(wake->tHistory);
  free(wake->QHistory);
  free(wake->xHistory);
  free(wake->yHistory);
  wake->tHistory = history[0];
  wake->QHistory = history[1];
  wake->xHistory = history[2];
  wake->yHistory = history[3];
#endif
  milliseconds = 0;
  status = gpuCudaLrwakeKick(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, voltage, nBunches, ibMin,
                             (int)bunchColumn, wake->factor * rampFactor, P0, particleMassMV, c_mks,
                             &milliseconds);
  if (status != 0)
    gpuFatalStatus("LRWAKE kick kernel", status);
  gpuRecordMagnetKernel(milliseconds);
  gpuMarkDeviceChanged(np);
  gpuRecordWallSeconds();
  return 1;
}

long gpu_ibs_bunched_mode_action(long np, void *ibs0, void *charge0) {
  IBSCATTER *IBS = (IBSCATTER *)ibs0;

  if (!IBS)
    return GPU_BUNCHED_WAKE_UNSUPPORTED;
  return gpuZimpedanceBunchedModeAction(np, IBS->bunchedBeamMode, -1, -1, charge0, IBS, "IBSCATTER",
                                        &gpuIbsPlanCache);
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

/* ELEGANT_GPU_ENABLE_WIDE_WAKE_TIME_SUM=0: a wake with a set number of bins and no output in the
 * CPU's order takes its mean time from the beam-sum kernels, as it did until November 13, 2026 */
static long gpuWideWakeTimeSumEnabled(void) {
  return !gpuEnvSet("ELEGANT_GPU_ENABLE_WIDE_WAKE_TIME_SUM") ||
         gpuEnvFlag("ELEGANT_GPU_ENABLE_WIDE_WAKE_TIME_SUM");
}

/* The count and the least and greatest time of the particles a wake acts on, in sums->count, min[6]
 * and max[6].  With needTimeSum the sum of the times is made too, in centroidSum[6]: a wake with a set
 * number of bins centers them on the mean time.  It is not added in the CPU's order (with output in
 * that order the callers use gpuWakeOrderedTimeMean() instead).  It comes from the same kernel as the
 * span, one particle per thread.  It used to come from the beam-sum kernels, which add every sum of
 * the beam on 64 threads: 41 ms for a million particles at every wake, against 0.1 ms. */
static long gpuWakeTimeSums(long np, double Po,
                            const GPU_BUNCHED_WAKE_PLAN *plan,
                            GPU_BEAM_SUM_DATA *sums, long needTimeSum,
                            const char *operation) {
  float milliseconds = 0;
  double tmin = 0, tmax = 0, tsum = 0;
  long count = 0;
  int status;

  memset(sums, 0, sizeof(*sums));
  if (needTimeSum && !gpuWideWakeTimeSumEnabled()) {
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
  if (plan && plan->useBunchFilter && plan->bucketStart &&
      plan->selectedCount > 0)
    status = gpuCudaTimeSpan(
      ((double *)gpuBase.deviceCoord) +
        plan->selectedStart * gpuBase.deviceStride,
      plan->selectedCount, (int)gpuBase.deviceStride, Po, c_mks, -1, 0,
      &count, &tmin, &tmax, &tsum, &milliseconds);
  else if (plan && plan->useBunchFilter)
    status = gpuCudaTimeSpan(gpuBase.deviceCoord, np,
                             (int)gpuBase.deviceStride, Po, c_mks,
                             (int)bunchIndex, plan->selectedBunch,
                             &count, &tmin, &tmax, &tsum, &milliseconds);
  else
    status = gpuCudaTimeSpan(gpuBase.deviceCoord, np,
                             (int)gpuBase.deviceStride, Po, c_mks, -1, 0,
                             &count, &tmin, &tmax, &tsum, &milliseconds);
  if (status != 0)
    gpuFatalStatus(operation, status);
  gpuRecordReductionKernel(milliseconds);
  sums->count = count;
  sums->min[6] = tmin;
  sums->max[6] = tmax;
  sums->centroidSum[6] = tsum;
  return sums->count;
}

/* The mean time of the particles gpuWakeTimeSums() selects, added in index order */
static double gpuWakeOrderedTimeMean(long np, double Po,
                                     const GPU_BUNCHED_WAKE_PLAN *plan,
                                     const char *operation) {
  float milliseconds = 0;
  double sum = 0;
  long count = 0;
  int status;

  if (plan && plan->useBunchFilter && plan->bucketStart &&
      plan->selectedCount > 0)
    status = gpuCudaOrderedTimeSum(
      ((double *)gpuBase.deviceCoord) +
        plan->selectedStart * gpuBase.deviceStride,
      plan->selectedCount, (int)gpuBase.deviceStride, Po, c_mks, -1, 0,
      &sum, &count, &milliseconds);
  else if (plan && plan->useBunchFilter)
    status = gpuCudaOrderedTimeSum(gpuBase.deviceCoord, np,
                                   (int)gpuBase.deviceStride, Po, c_mks,
                                   (int)bunchIndex, plan->selectedBunch,
                                   &sum, &count, &milliseconds);
  else
    status = gpuCudaOrderedTimeSum(gpuBase.deviceCoord, np,
                                   (int)gpuBase.deviceStride, Po, c_mks,
                                   -1, 0, &sum, &count, &milliseconds);
  if (status != 0)
    gpuFatalStatus(operation, status);
  gpuRecordReductionKernel(milliseconds);
  return count > 0 ? sum / count : 0;
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

/* V(f) = Z(f) * I(f) * factor in realFFT()'s packed layout, as applyImpedanceMultiplication() in
 * impedance.c does it */
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

typedef struct {
  IMPEDANCE *imp;
  const GPU_COMBINED_WAKE_DATA *data;
} GPU_IMPEDANCE_HOST_VOLTAGE;

/* track_through_impedance()'s voltages from the device's histograms: the driver's FFT, the impedance
 * and the inverse FFT, with the CPU's routines, so the device kicks with the CPU's numbers.  The
 * tables are short (the impedance file's points), so this costs little beside the binning and the
 * kicks, which stay on the device. */
static void gpuImpedanceHostVoltage(const double *histogram, double *voltage, void *context0) {
  static double *ifreq = NULL, *vtime = NULL;
  static long capacity = 0;
  GPU_IMPEDANCE_HOST_VOLTAGE *context = (GPU_IMPEDANCE_HOST_VOLTAGE *)context0;
  const GPU_COMBINED_WAKE_DATA *data = context->data;
  IMPEDANCE *imp = context->imp;
  long nb = data->bins, channel;

  if (capacity < nb) {
    free(ifreq);
    free(vtime);
    ifreq = (double *)malloc(2 * nb * sizeof(*ifreq));
    vtime = (double *)malloc(2 * (nb + 1) * sizeof(*vtime));
    if (!ifreq || !vtime)
      gpuRequiredFailure("unable to allocate the IMPEDANCE host FFT buffers");
    capacity = nb;
  }
  for (channel = 0; channel < IMPEDANCE_N_WAKES; channel++) {
    if (!imp->enabled[channel])
      continue;
    memset(ifreq, 0, 2 * nb * sizeof(*ifreq));
    memcpy(ifreq, histogram + data->driver[channel] * nb, nb * sizeof(*ifreq));
    realFFT(ifreq, nb, 0);
    memset(vtime, 0, 2 * (nb + 1) * sizeof(*vtime));
    gpuApplyImpedanceMultiplication(ifreq, imp->Z[channel], vtime, nb, data->factor[channel]);
    realFFT(vtime, nb, INVERSE_FFT);
    memcpy(voltage + channel * nb, vtime, nb * sizeof(*vtime));
  }
}

#ifdef GPU_VERIFY
/* In CPU order the device's arrays must be the CPU's, to the last bit */
static void gpuCompareCombinedWakeArray(const GPU_COMBINED_WAKE_DATA *data, const char *label,
                                        const char *arrayName, const double *cpu,
                                        const double *gpu, long n) {
  if (data->cpuOrder)
    gpuCompareWakeArrayWithTolerance(label, arrayName, cpu, gpu, n, 0, 0);
  else
    gpuCompareWakeArray(label, arrayName, cpu, gpu, n);
}

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
  gpuCompareCombinedWakeArray(data, "combined wake", "Itime", cpuHistogram,
                              gpuHistogram, data->bins);
  gpuCompareCombinedWakeArray(data, "combined wake", "xItime",
                              cpuHistogram + data->bins,
                              gpuHistogram + data->bins, data->bins);
  gpuCompareCombinedWakeArray(data, "combined wake", "yItime",
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
    if (data->allowTimeFft && !data->cpuOrder)
      gpuCompareWakeArrayWithTolerance(
        "track_through_cwake FFT", "Vtime", cpuVoltage,
        gpuVoltage + channel * data->bins, data->bins, 1e-6, 1e-8);
    else
      gpuCompareCombinedWakeArray(data, "track_through_cwake", "Vtime", cpuVoltage,
                                  gpuVoltage + channel * data->bins, data->bins);
  }
  free(cpuVoltage);
  free(cpuHistogram);
  free(cpuPbin);
  free(cpuTime);
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
    gpuCompareCombinedWakeArray(data, "track_through_impedance", "Vtime", cpuVoltage,
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
  GPU_IMPEDANCE_HOST_VOLTAGE hostVoltage;
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
    np, Po, plan, &sums, 0, "IMPEDANCE time-coordinate reduction kernel");
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
  data.cpuOrder = gpuWakeCpuOrderEnabled() ? 1 : 0;
  hostVoltage.imp = imp;
  hostVoltage.data = &data;
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
    data.cpuOrder ? gpuImpedanceHostVoltage : NULL, &hostVoltage,
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
  long cpuOrder = gpuWakeCpuOrderEnabled();
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
    np, Po, plan, &sums, wake->n_bins != 0 && !cpuOrder,
    "CWAKE time-coordinate reduction kernel");
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
    /* fixed bins sit on the mean time, which the CPU adds in particle order */
    if (cpuOrder)
      tmin = gpuWakeOrderedTimeMean(np, Po, plan, "CWAKE ordered time-sum kernel") -
             wake->dt * bins / 2.0;
    else
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
  data.cpuOrder = cpuOrder ? 1 : 0;
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
    NULL, NULL,
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

/* ZLONGIT and ZTRANSVERSE on the device (November 14, 2026).  The device finds the time span, bins
 * (always in particle order: a stable sort by bin, then one block per bin) and kicks.  When output
 * needs the CPU's order, or smoothing or WAKES is asked for, the host makes the voltages from the
 * histograms with the operations of track_through_zlongit() and track_through_ztransverse():
 * Savitzky-Golay smoothing, realFFT, Z(f) times I(f), the inverse realFFT, and writes the WAKES
 * tables; the voltages are then the CPU's to the last bit.  Otherwise cuFFT makes them, as for
 * IMPEDANCE.  A verify build reruns the CPU routine, which writes the WAKES table itself. */

static const char *gpuCurrentElementName(void) {
  ELEMENT_LIST *eptr = (ELEMENT_LIST *)gpuBase.element;
  return eptr && eptr->name ? eptr->name : "?";
}

/* gpuBunchedWakePlan() found the rows out of bunch order.  They are gathered by bunch (a stable sort,
 * so each bunch keeps elegant's particle order, as index_bunch_assignments()'s bucket lists do) into a
 * scratch buffer, which stands in for the beam until gpuZimpedanceScatter() writes the rows back. */
static void *gpuZimpedanceSavedCoord = NULL;
static long *gpuZimpedanceBucketStart = NULL, *gpuZimpedanceBucketCount = NULL;
static long gpuZimpedanceBucketCapacity = 0;

static void gpuZimpedanceGather(long np, GPU_BUNCHED_WAKE_PLAN *plan, const char *operation) {
  void *gathered = NULL;
  float milliseconds = 0;
  int status;

  if (gpuZimpedanceSavedCoord)
    gpuRequiredFailure("ZLONGIT/ZTRANSVERSE bunch gather is already active");
  if (gpuZimpedanceBucketCapacity < plan->nBuckets) {
    free(gpuZimpedanceBucketStart);
    free(gpuZimpedanceBucketCount);
    gpuZimpedanceBucketStart = (long *)malloc(plan->nBuckets * sizeof(*gpuZimpedanceBucketStart));
    gpuZimpedanceBucketCount = (long *)malloc(plan->nBuckets * sizeof(*gpuZimpedanceBucketCount));
    if (!gpuZimpedanceBucketStart || !gpuZimpedanceBucketCount)
      gpuRequiredFailure("unable to allocate the ZLONGIT/ZTRANSVERSE bunch ranges");
    gpuZimpedanceBucketCapacity = plan->nBuckets;
  }
  status = gpuCudaGatherByBunch(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, (int)bunchIndex,
                                plan->minBunch, plan->nBuckets, &gathered, gpuZimpedanceBucketStart,
                                gpuZimpedanceBucketCount, &milliseconds);
  if (status != 0)
    gpuFatalStatus(operation, status);
  gpuRecordReductionKernel(milliseconds);
  gpuZimpedanceSavedCoord = gpuBase.deviceCoord;
  gpuBase.deviceCoord = gathered;
  plan->bucketStart = gpuZimpedanceBucketStart;
  plan->bucketCount = gpuZimpedanceBucketCount;
}

static void gpuZimpedanceScatter(long np, const char *operation) {
  float milliseconds = 0;
  int status;

  gpuBase.deviceCoord = gpuZimpedanceSavedCoord;
  gpuZimpedanceSavedCoord = NULL;
  status = gpuCudaScatterByBunch(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, &milliseconds);
  if (status != 0)
    gpuFatalStatus(operation, status);
  gpuRecordReductionKernel(milliseconds);
}

static void gpuZimpedanceExit(void) {
  if (exitElegant)
    exitElegant(1);
  exit(1);
}

typedef struct {
  ZLONGIT *zlongit;
  const GPU_COMBINED_WAKE_DATA *data;
  double rampFactor;
  long pass;      /* the pass number track_through_zlongit() is given, for WAKES */
  long bunch;     /* its iBucket */
  long particles; /* the bunch's particles, for WAKES' charge */
} GPU_ZLONGIT_HOST_VOLTAGE;

typedef struct {
  ZTRANSVERSE *ztransverse;
  const GPU_COMBINED_WAKE_DATA *data;
  double userFactor[2];
  long pass, particles;
} GPU_ZTRANSVERSE_HOST_VOLTAGE;

#ifndef GPU_VERIFY
/* track_through_zlongit()'s wake output, statement for statement */
static void gpuZlongitWriteWakes(ZLONGIT *zlongit, long i_pass0, long iBucket, long np,
                                 const double *Itime, const double *Vtime, long nb, double dt) {
  double factor;
  long ib;

  if (!zlongit->SDDS_wake_initialized || !zlongit->wakes)
    return;
  factor = zlongit->macroParticleCharge * particleRelSign / dt;
  if (!((zlongit->wake_interval <= 0 || ((i_pass0 - zlongit->wake_start) % zlongit->wake_interval) == 0) &&
        i_pass0 >= zlongit->wake_start && (zlongit->wake_end < 0 || i_pass0 <= zlongit->wake_end)))
    return;
  if (!SDDS_StartTable(zlongit->SDDS_wake, nb)) {
    SDDS_SetError("Problem starting SDDS table for wake output (track_through_zlongit)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  for (ib = 0; ib < nb; ib++) {
    if (!SDDS_SetRowValues(zlongit->SDDS_wake, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, ib,
                           0, ib * dt, 1, Vtime[ib], 2, Itime[ib] * factor, -1)) {
      SDDS_SetError("Problem setting rows of SDDS table for wake output (track_through_zlongit)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  }
  if (!SDDS_SetParameters(zlongit->SDDS_wake, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                          "Pass", i_pass0, "Bunch", iBucket, "q",
                          zlongit->macroParticleCharge * particleRelSign * np, NULL)) {
    SDDS_SetError("Problem setting parameters of SDDS table for wake output (track_through_zlongit)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  if (zlongit->broad_band) {
    if (!SDDS_SetParameters(zlongit->SDDS_wake, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                            "Ra", zlongit->Ra, "fo", zlongit->freq, "Deltaf", 1. / dt, NULL)) {
      SDDS_SetError("Problem setting parameters of SDDS table for wake output (track_through_zlongit)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  }
  if (!SDDS_WriteTable(zlongit->SDDS_wake)) {
    SDDS_SetError("Problem writing SDDS table for wake output (track_through_zlongit)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  if (!inhibitFileSync)
    SDDS_DoFSync(zlongit->SDDS_wake);
}

/* track_through_ztransverse()'s wake output for one plane: the first starts the table, the second
 * writes it */
static void gpuZtransverseWriteWakes(ZTRANSVERSE *ztransverse, long i_pass0, long np, long plane,
                                     long first, const double *posItime, const double *Vtime,
                                     long nb, double dt) {
  double factor;
  long ib;

  if (!ztransverse->SDDS_wake_initialized || !ztransverse->wakes)
    return;
  factor = ztransverse->macroParticleCharge * particleRelSign / dt;
  if (!((ztransverse->wake_interval <= 0 ||
         ((i_pass0 - ztransverse->wake_start) % ztransverse->wake_interval) == 0) &&
        i_pass0 >= ztransverse->wake_start &&
        (ztransverse->wake_end < 0 || i_pass0 <= ztransverse->wake_end)))
    return;
  if (first && !SDDS_StartTable(ztransverse->SDDS_wake, nb)) {
    SDDS_SetError("Problem starting SDDS table for wake output (track_through_ztransverse)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  for (ib = 0; ib < nb; ib++) {
    if (!SDDS_SetRowValues(ztransverse->SDDS_wake, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, ib,
                           0, ib * dt,
                           1 + plane * 2, posItime[ib] * factor,
                           2 + plane * 2, Vtime[ib], -1)) {
      SDDS_SetError("Problem setting rows of SDDS table for wake output (track_through_ztransverse)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  }
  if (!SDDS_SetParameters(ztransverse->SDDS_wake, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                          "Pass", i_pass0, "q", ztransverse->macroParticleCharge * particleRelSign * np,
                          NULL)) {
    SDDS_SetError("Problem setting parameters of SDDS table for wake output (track_through_ztransverse)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  if (ztransverse->broad_band) {
    if (!SDDS_SetParameters(ztransverse->SDDS_wake, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                            "Rs", ztransverse->Rs, "fo", ztransverse->freq,
                            "Deltaf", ztransverse->bin_size, NULL)) {
      SDDS_SetError("Problem setting parameters of SDDS table for wake output (track_through_ztransverse)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  }
  if (!first) {
    if (!SDDS_WriteTable(ztransverse->SDDS_wake)) {
      SDDS_SetError("Problem writing SDDS table for wake output (track_through_ztransverse)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    if (!inhibitFileSync)
      SDDS_DoFSync(ztransverse->SDDS_wake);
  }
}
#endif

static void gpuZimpedanceSmooth(double *data, long nb, long order, long halfWidth) {
  if (!SavitzkyGolaySmooth)
    gpuRequiredFailure("Savitzky-Golay smoothing routine unavailable for ZLONGIT/ZTRANSVERSE");
  /* the CPU ignores the result, as here */
  SavitzkyGolaySmooth(data, nb, order, halfWidth, halfWidth, 0);
}

/* track_through_zlongit()'s voltage from the count histogram */
static void gpuZlongitHostVoltage(const double *histogram, double *voltage, void *context0) {
  static double *Itime = NULL, *Ifreq = NULL, *Vtime = NULL;
  static long capacity = 0;
  GPU_ZLONGIT_HOST_VOLTAGE *context = (GPU_ZLONGIT_HOST_VOLTAGE *)context0;
  ZLONGIT *zlongit = context->zlongit;
  long nb = context->data->bins;
  double dt = context->data->dt, factor;

  if (capacity < nb) {
    free(Itime);
    free(Ifreq);
    free(Vtime);
    Itime = (double *)malloc(2 * nb * sizeof(*Itime));
    Ifreq = (double *)malloc(2 * nb * sizeof(*Ifreq));
    Vtime = (double *)malloc(2 * (nb + 1) * sizeof(*Vtime));
    if (!Itime || !Ifreq || !Vtime)
      gpuRequiredFailure("unable to allocate the ZLONGIT host FFT buffers");
    capacity = nb;
  }
  memset(Itime, 0, 2 * nb * sizeof(*Itime));
  memcpy(Itime, histogram, nb * sizeof(*Itime));
  if (zlongit->smoothing)
    gpuZimpedanceSmooth(Itime, nb, zlongit->SGOrder, zlongit->SGHalfWidth);
  memcpy(Ifreq, Itime, 2 * nb * sizeof(*Ifreq));
  realFFT(Ifreq, nb, 0);
  factor = zlongit->macroParticleCharge * particleRelSign / dt * zlongit->factor * context->rampFactor;
  memset(Vtime, 0, 2 * (nb + 1) * sizeof(*Vtime));
  gpuApplyImpedanceMultiplication(Ifreq, zlongit->Z, Vtime, nb, factor);
  realFFT(Vtime, nb, INVERSE_FFT);
  memcpy(voltage, Vtime, nb * sizeof(*voltage));
#ifndef GPU_VERIFY
  gpuZlongitWriteWakes(zlongit, context->pass, context->bunch, context->particles, Itime, Vtime, nb, dt);
#endif
}

/* track_through_ztransverse()'s voltages, x into channel 1 and y into channel 2 */
static void gpuZtransverseHostVoltage(const double *histogram, double *voltage, void *context0) {
  static double *posItime = NULL, *posIfreq = NULL, *Vtime = NULL;
  static long capacity = 0;
  GPU_ZTRANSVERSE_HOST_VOLTAGE *context = (GPU_ZTRANSVERSE_HOST_VOLTAGE *)context0;
  ZTRANSVERSE *ztransverse = context->ztransverse;
  const GPU_COMBINED_WAKE_DATA *data = context->data;
  long nb = data->bins, plane, first = 1;
  double dt = data->dt, factor;

  if (capacity < nb) {
    free(posItime);
    free(posIfreq);
    free(Vtime);
    posItime = (double *)malloc(4 * nb * sizeof(*posItime));
    posIfreq = (double *)malloc(2 * nb * sizeof(*posIfreq));
    Vtime = (double *)malloc(2 * (nb + 1) * sizeof(*Vtime));
    if (!posItime || !posIfreq || !Vtime)
      gpuRequiredFailure("unable to allocate the ZTRANSVERSE host FFT buffers");
    capacity = nb;
  }
  /* binTransverseTimeDistribution() makes both planes before either voltage */
  for (plane = 0; plane < 2; plane++) {
    memset(posItime + 2 * nb * plane, 0, 2 * nb * sizeof(*posItime));
    memcpy(posItime + 2 * nb * plane, histogram + data->driver[1 + plane] * nb,
           nb * sizeof(*posItime));
  }
  for (plane = 0; plane < 2; plane++) {
    double *Itime = posItime + 2 * nb * plane;
    if (context->userFactor[plane] == 0) {
      for (long ib = 0; ib < nb; ib++)
        Vtime[ib] = 0;
    } else {
      if (ztransverse->smoothing)
        gpuZimpedanceSmooth(Itime, nb, ztransverse->SGOrder, ztransverse->SGHalfWidth);
      memcpy(posIfreq, Itime, 2 * nb * sizeof(*posIfreq));
      realFFT(posIfreq, nb, 0);
      factor = ztransverse->macroParticleCharge * particleRelSign / dt * context->userFactor[plane];
      memset(Vtime, 0, 2 * (nb + 1) * sizeof(*Vtime));
      gpuApplyImpedanceMultiplication(posIfreq, ztransverse->Z[plane], Vtime, nb, factor);
      realFFT(Vtime, nb, INVERSE_FFT);
      memcpy(voltage + (1 + plane) * nb, Vtime, nb * sizeof(*voltage));
    }
#ifndef GPU_VERIFY
    gpuZtransverseWriteWakes(ztransverse, context->pass, context->particles, plane, first,
                             Itime, Vtime, nb, dt);
#endif
    first = 0; (void)first; /* a verify build writes no WAKES here */
  }
}

/* One bunch of track_through_zlongit(): pass is the pass less START_ON_PASS, pass0 the pass itself,
 * bunch the CPU's iBucket */
static long gpuTrackZlongitBucket(long np, ZLONGIT *zlongit, double Po, RUN *run, long pass,
                                  long pass0, CHARGE *charge, double rampFactor, long bunch,
                                  const GPU_BUNCHED_WAKE_PLAN *plan) {
  GPU_BEAM_SUM_DATA sums;
  GPU_COMBINED_WAKE_DATA data;
  GPU_ZLONGIT_HOST_VOLTAGE hostVoltage;
  const double *tables[GPU_COMBINED_WAKE_CHANNELS] = {NULL};
  long expectedParticles, binnedCount = 0, trackedParticles, startParticle, nb, hostFft;
  long cpuOrder = gpuWakeCpuOrderEnabled();
  double *deviceCoord;
  double tmin, tmax, tmean = 0, dt;
  float milliseconds = 0;
  int status;

  startParticle = plan->useBunchFilter && plan->bucketStart ? plan->selectedStart : 0;
  trackedParticles = plan->useBunchFilter && plan->bucketStart ? plan->selectedCount : np;
  if (trackedParticles <= 0)
    return 0;
  deviceCoord = ((double *)gpuBase.deviceCoord) + startParticle * gpuBase.deviceStride;
  expectedParticles = gpuWakeTimeSums(np, Po, plan, &sums, !zlongit->bin_centered && !cpuOrder,
                                      "ZLONGIT time-coordinate reduction kernel");
  if (expectedParticles <= 0)
    return 0;
  tmin = sums.min[6];
  tmax = sums.max[6];
  if (!zlongit->bin_centered) {
    /* compute_average(): the times added in particle order, over the count */
    if (cpuOrder)
      tmean = gpuWakeOrderedTimeMean(np, Po, plan, "ZLONGIT ordered time-sum kernel");
    else
      tmean = sums.centroidSum[6] / sums.count;
  }
  /* the CPU passes the whole beam's particle count, for the macro-particle charge */
  set_up_zlongit(zlongit, run, pass, np, charge, tmax - tmin);
  nb = zlongit->n_bins;
  dt = zlongit->bin_size;
  if (nb < 2 || !(dt > 0) || !zlongit->Z)
    gpuRequiredFailure("ZLONGIT data was not initialized for CUDA tracking");

  if ((tmax - tmin) * 2 > nb * dt) {
    printf("%s %s: Time span of bunch %ld (%21.15le->%21.15le, span %21.15le s) is more than half the total time span (%21.15le s).\n",
           entity_name[T_ZLONGIT], gpuCurrentElementName(), bunch, tmin, tmax, tmax - tmin, nb * dt);
    printf("If using broad-band impedance, you should increase the number of bins (or use auto-scaling) and rerun.\n");
    printf("If using file-based impedance, you should increase the number of data points or decrease the frequency resolution.\n");
    if (!zlongit->allowLongBeam)
      gpuZimpedanceExit();
  }

  memset(&data, 0, sizeof(data));
  data.bins = nb;
  data.tablePoints = nb;
  data.mode = GPU_COMBINED_WAKE_MODE_IMPEDANCE;
  data.interpolate = (int)zlongit->interpolate;
  /* Always the ordered histogram (stable sort, block-per-bin sums): it is the CPU's, and unlike the
   * partial histograms its cost doesn't grow with the bins, which a broad-band resonator's auto-scaling
   * can take to 131,072.  Only the mean time above depends on the output's needs. */
  data.cpuOrder = 1;
  data.useBunchFilter = (int)(plan->useBunchFilter && !plan->bucketStart);
  data.bunchIndexColumn = bunchIndex;
  data.selectedBunch = plan->selectedBunch;
  if (zlongit->bin_centered)
    data.tmin = tmin - dt;
  else {
    /* the bunch centered in the grid, as zlongit.c forms it */
    data.tmin = tmean - dt * nb / 2.0;
    data.legacyGrid = 1;
    data.tminCenter = data.tmin + dt / 2.0;
  }
  data.dt = dt;
  data.pCentral = Po;
  data.particleMassMV = particleMassMV;
  data.particleRelSign = particleRelSign;
  data.cMks = c_mks;
  data.enabled[0] = 1;
  data.driver[0] = 0;
  data.kickPlane[0] = 0;
  data.factor[0] = zlongit->macroParticleCharge * particleRelSign / dt * zlongit->factor * rampFactor;
  tables[0] = zlongit->Z;
  hostVoltage.zlongit = zlongit;
  hostVoltage.data = &data;
  hostVoltage.rampFactor = rampFactor;
  hostVoltage.pass = pass0;
  hostVoltage.bunch = bunch;
  hostVoltage.particles = expectedParticles;
  /* The host's realFFT when output needs the CPU's numbers, for smoothing and for WAKES; cuFFT
   * otherwise (the host's FFT of an auto-scaled broad-band grid, 131,072 bins, took 2 ms a call) */
  hostFft = cpuOrder || zlongit->smoothing || gpuStringSet(zlongit->wakes);
  status = gpuCudaCombinedWakeTrack(deviceCoord, trackedParticles, (int)gpuBase.deviceStride, &data,
                                    tables, &binnedCount, NULL, NULL,
                                    hostFft ? gpuZlongitHostVoltage : NULL, &hostVoltage,
                                    &milliseconds);
  if (status != 0)
    gpuFatalStatus("ZLONGIT combined CUDA kernel", status);
  gpuRecordWakeKernel(milliseconds);
  if (binnedCount != expectedParticles) {
    char warningBuffer[1024];
    snprintf(warningBuffer, 1024, "Only %ld of %ld particles were binned; results suspect. Reduce impedance frequency spacing for file-based impedances or invoking auto-scaling for broad-band impedances.",
             binnedCount, expectedParticles);
    gpuWakeTrackingWarning("Not all particles binned in ZLONGIT.", warningBuffer);
  }
  return 1;
}

static long gpuTrackZtransverseBucket(long np, ZTRANSVERSE *ztransverse, double Po, RUN *run,
                                      long pass, long pass0, CHARGE *charge, double rampFactor,
                                      long bunch, const GPU_BUNCHED_WAKE_PLAN *plan) {
  GPU_BEAM_SUM_DATA sums;
  GPU_COMBINED_WAKE_DATA data;
  GPU_ZTRANSVERSE_HOST_VOLTAGE hostVoltage;
  const double *tables[GPU_COMBINED_WAKE_CHANNELS] = {NULL};
  long expectedParticles, binnedCount = 0, trackedParticles, startParticle, nb, plane, hostFft;
  double *deviceCoord;
  double tmin, tmax, dt;
  float milliseconds = 0;
  int status;

  startParticle = plan->useBunchFilter && plan->bucketStart ? plan->selectedStart : 0;
  trackedParticles = plan->useBunchFilter && plan->bucketStart ? plan->selectedCount : np;
  if (trackedParticles <= 0)
    return 0;
  deviceCoord = ((double *)gpuBase.deviceCoord) + startParticle * gpuBase.deviceStride;
  expectedParticles = gpuWakeTimeSums(np, Po, plan, &sums, 0,
                                      "ZTRANSVERSE time-coordinate reduction kernel");
  if (expectedParticles <= 0)
    return 0;
  tmin = sums.min[6];
  tmax = sums.max[6];
  set_up_ztransverse(ztransverse, run, pass, np, charge, tmax - tmin);
  nb = ztransverse->n_bins;
  dt = ztransverse->bin_size;
  if (nb < 2 || !(dt > 0) || !ztransverse->Z[0] || !ztransverse->Z[1])
    gpuRequiredFailure("ZTRANSVERSE data was not initialized for CUDA tracking");
  tmin -= dt;
  tmax += dt;
  if ((tmax - tmin) * 2 > nb * dt) {
    fprintf(stderr, "%s %s: Time span of bunch %ld (%21.15le s) is more than half the total time span (%21.15le s).\n",
            entity_name[T_ZTRANSVERSE], gpuCurrentElementName(), bunch, tmax - tmin, nb * dt);
    fprintf(stderr, "If using broad-band impedance, you should increase the number of bins and rerun.\n");
    fprintf(stderr, "If using file-based impedance, you should increase the number of data points or decrease the frequency resolution.\n");
    if (!ztransverse->allowLongBeam)
      gpuZimpedanceExit();
  }

  memset(&data, 0, sizeof(data));
  data.bins = nb;
  data.tablePoints = nb;
  data.mode = GPU_COMBINED_WAKE_MODE_IMPEDANCE;
  data.interpolate = (int)ztransverse->interpolate;
  /* always the ordered histogram, as for ZLONGIT */
  data.cpuOrder = 1;
  data.useBunchFilter = (int)(plan->useBunchFilter && !plan->bucketStart);
  data.bunchIndexColumn = bunchIndex;
  data.selectedBunch = plan->selectedBunch;
  data.tmin = tmin;
  data.dt = dt;
  data.pCentral = Po;
  data.offset[0] = ztransverse->dx;
  data.offset[1] = ztransverse->dy;
  data.particleMassMV = particleMassMV;
  data.particleRelSign = particleRelSign;
  data.cMks = c_mks;
  hostVoltage.ztransverse = ztransverse;
  hostVoltage.data = &data;
  hostVoltage.userFactor[0] = ztransverse->factor * ztransverse->xfactor * rampFactor;
  hostVoltage.userFactor[1] = ztransverse->factor * ztransverse->yfactor * rampFactor;
  hostVoltage.pass = pass0;
  hostVoltage.particles = expectedParticles;
  /* channel 1 kicks x' with the x voltage, channel 2 y'; a drive exponent of 1 bins (x - DX) or
   * (y - DY), one below bins 1 */
  for (plane = 0; plane < 2; plane++) {
    long channel = 1 + plane;
    long driveExponent = plane ? ztransverse->yDriveExponent : ztransverse->xDriveExponent;
    data.enabled[channel] = hostVoltage.userFactor[plane] != 0;
    data.driver[channel] = driveExponent == 1 ? 1 + plane : 0;
    data.kickPlane[channel] = 1 + plane;
    data.probeExponent[channel] = plane ? ztransverse->yProbeExponent : ztransverse->xProbeExponent;
    data.factor[channel] = ztransverse->macroParticleCharge * particleRelSign / dt *
                           hostVoltage.userFactor[plane];
    tables[channel] = ztransverse->Z[plane];
  }
  /* the host's realFFT when output needs the CPU's numbers, for smoothing and for WAKES; cuFFT otherwise */
  hostFft = gpuWakeCpuOrderEnabled() || ztransverse->smoothing || gpuStringSet(ztransverse->wakes);
  status = gpuCudaCombinedWakeTrack(deviceCoord, trackedParticles, (int)gpuBase.deviceStride, &data,
                                    tables, &binnedCount, NULL, NULL,
                                    hostFft ? gpuZtransverseHostVoltage : NULL, &hostVoltage,
                                    &milliseconds);
  if (status != 0)
    gpuFatalStatus("ZTRANSVERSE combined CUDA kernel", status);
  gpuRecordWakeKernel(milliseconds);
  return 1;
}

void gpu_track_through_zlongit(long np, void *zlongit0, double Po, void *run0, long iPass,
                               void *charge0) {
  ZLONGIT *zlongit = (ZLONGIT *)zlongit0;
  GPU_BUNCHED_WAKE_PLAN plan, bucketPlan;
  long pass, first, last, tracked = 0;
  double rampFactor;

  if (np <= 0 || !zlongit)
    return;
  pass = iPass - zlongit->startOnPass;
  if (pass < 0 || zlongit->factor == 0) {
    gpuRecordWallSeconds();
    return;
  }
  rampFactor = 0;
  if (pass >= (zlongit->rampPasses - 1))
    rampFactor = 1;
  else
    rampFactor = (pass + 1.0) / zlongit->rampPasses;
  if (!set_up_zlongit)
    gpuRequiredFailure("ZLONGIT setup routine is unavailable");
  if (gpuZlongitPlanCache.valid && gpuZlongitPlanCache.owner == zlongit &&
      gpuZlongitPlanCache.particles == np) {
    plan = gpuZlongitPlanCache.plan;
    gpuZlongitPlanCache.valid = 0;
  } else if (gpuBunchedWakePlan(np, zlongit->bunchedBeamMode, zlongit->startBunch, zlongit->endBunch,
                                (CHARGE *)charge0, "ZLONGIT", &plan) == GPU_BUNCHED_WAKE_UNSUPPORTED) {
    gpuRequiredFailure("unsupported ZLONGIT bunch layout reached CUDA path");
  }
  if (plan.action == GPU_BUNCHED_WAKE_SKIP) {
    gpuRecordWallSeconds();
    return;
  }
  gpuCopyHostToDevice(np);
  if (plan.needsGather)
    gpuZimpedanceGather(np, &plan, "ZLONGIT bunch gather kernels");
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
    if (gpuTrackZlongitBucket(np, zlongit, Po, (RUN *)run0, pass, iPass, (CHARGE *)charge0,
                              rampFactor, relative, &bucketPlan) > 0)
      tracked++;
  }
  if (plan.needsGather)
    gpuZimpedanceScatter(np, "ZLONGIT bunch scatter kernel");
  if (tracked)
    gpuMarkDeviceChanged(np);
  gpuRecordWallSeconds();
}

void gpu_track_through_ztransverse(long np, void *ztransverse0, double Po, void *run0, long iPass,
                                   void *charge0) {
  ZTRANSVERSE *ztransverse = (ZTRANSVERSE *)ztransverse0;
  GPU_BUNCHED_WAKE_PLAN plan, bucketPlan;
  long pass, first, last, tracked = 0;
  double rampFactor = 1;

  if (np <= 0 || !ztransverse)
    return;
  pass = iPass - ztransverse->startOnPass;
  if (pass < 0 || ztransverse->factor == 0) {
    gpuRecordWallSeconds();
    return;
  }
  if (pass >= (ztransverse->rampPasses - 1))
    rampFactor = 1;
  else
    rampFactor = (pass + 1.0) / ztransverse->rampPasses;
  if (!set_up_ztransverse)
    gpuRequiredFailure("ZTRANSVERSE setup routine is unavailable");
  if (gpuZtransversePlanCache.valid && gpuZtransversePlanCache.owner == ztransverse &&
      gpuZtransversePlanCache.particles == np) {
    plan = gpuZtransversePlanCache.plan;
    gpuZtransversePlanCache.valid = 0;
  } else if (gpuBunchedWakePlan(np, ztransverse->bunchedBeamMode, ztransverse->startBunch,
                                ztransverse->endBunch, (CHARGE *)charge0, "ZTRANSVERSE", &plan) ==
             GPU_BUNCHED_WAKE_UNSUPPORTED) {
    gpuRequiredFailure("unsupported ZTRANSVERSE bunch layout reached CUDA path");
  }
  if (plan.action == GPU_BUNCHED_WAKE_SKIP) {
    gpuRecordWallSeconds();
    return;
  }
  gpuCopyHostToDevice(np);
  if (plan.needsGather)
    gpuZimpedanceGather(np, &plan, "ZTRANSVERSE bunch gather kernels");
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
    if (gpuTrackZtransverseBucket(np, ztransverse, Po, (RUN *)run0, pass, iPass, (CHARGE *)charge0,
                                  rampFactor, relative, &bucketPlan) > 0)
      tracked++;
  }
  if (plan.needsGather)
    gpuZimpedanceScatter(np, "ZTRANSVERSE bunch scatter kernel");
  if (tracked)
    gpuMarkDeviceChanged(np);
  gpuRecordWallSeconds();
}

/* IBSCATTER on the device (November 14, 2026): track_IBS()'s serial path for SMOOTH=1.  The host copies
 * a bunch's rows down and runs elegant's own slicebeam() and computeSliceParameters() on the copy, then
 * correctedEmittance(), forth_propagate_twiss() and IBSRate(), as track_IBS() does; the device does the
 * inflation of inflateEmittance() and inflateEmittanceZ().  The beam stays on the device: one copy down
 * instead of the round trip.  The sums are the host's because an ordered sum is one chain of dependent
 * additions: about 28 ns each on the device, against 1 ns on the host.  The inflation's means (x, x', y,
 * y', delta, t over a slice) are computeSliceParameters()'s centroids: the same values added in the
 * same order. */
#define GPU_IBS_HOST_SLICE_PARAMETERS 11
#define GPU_IBS_SAFE_SQRT(x) ((x) < 0 ? 0.0 : sqrt(x))

static double *gpuIbsRows = NULL, **gpuIbsPart = NULL;
static long gpuIbsRowCapacity = 0;

/* The rows start .. start+np-1 on the host, as part[] */
static double **gpuIbsCopyRows(double *coord, long np) {
  long stride = gpuBase.deviceStride, ip;
  float milliseconds = 0;
  int status;

  if (gpuIbsRowCapacity < np) {
    free(gpuIbsRows);
    free(gpuIbsPart);
    gpuIbsRows = (double *)malloc(np * stride * sizeof(*gpuIbsRows));
    gpuIbsPart = (double **)malloc(np * sizeof(*gpuIbsPart));
    if (!gpuIbsRows || !gpuIbsPart)
      gpuRequiredFailure("unable to allocate the IBSCATTER host rows");
    gpuIbsRowCapacity = np;
  }
  status = gpuCudaCopyDeviceToHost(gpuIbsRows, coord, (unsigned long)(np * stride), &milliseconds);
  if (status != 0)
    gpuFatalStatus("cudaMemcpy(IBSCATTER rows to host)", status);
  gpuRecordMilliseconds(&gpuBase.gpuTransferToHostSeconds, milliseconds);
  for (ip = 0; ip < np; ip++)
    gpuIbsPart[ip] = gpuIbsRows + ip * stride;
  return gpuIbsPart;
}

/* One bucket of track_IBS(): rows start .. start+np-1 of the device's beam */
static void gpuIbsBucket(IBSCATTER *IBS, long start, long np, double eta[4], double Po, RUN *run,
                         CHARGE *charge) {
  double *coord = ((double *)gpuBase.deviceCoord) + start * gpuBase.deviceStride;
  long nslice = IBS->nslice, islice, istart, iend, *index, *count;
  unsigned int *sliceOfRow = NULL;
  double *centroid, *parameters, *time = NULL, **part;
  double tLength = 1, bLength = 0, aveCoord[6], S[6][6], zRate[3];
  double betax0, alphax0, betay0, alphay0;
  float milliseconds = 0;
  int status;

  IBS->charge = charge->macroParticleCharge * np;
  if (np == 0 || !IBS->charge)
    return;
  part = gpuIbsCopyRows(coord, np);
  index = (long *)malloc(np * sizeof(*index));
  count = (long *)malloc(nslice * sizeof(*count));
  centroid = (double *)calloc(6 * nslice, sizeof(*centroid));
  parameters = (double *)calloc(GPU_IBS_HOST_SLICE_PARAMETERS * nslice, sizeof(*parameters));
  if (!index || !count || !centroid || !parameters)
    gpuRequiredFailure("unable to allocate IBSCATTER slice arrays");
  if (nslice > 1) {
    /* index_bunch_assignments()'s times */
    if (!(time = (double *)malloc(np * sizeof(*time))))
      gpuRequiredFailure("unable to allocate IBSCATTER times");
    computeTimeCoordinatesOnly(time, Po, part, np);
  }
  slicebeam(part, np, time, Po, nslice, index, count, &tLength);
  if (nslice > 1)
    bLength = c_mks * tLength / sqrt(4 * PI);

  iend = 0;
  for (islice = 0; islice < nslice; islice++) {
    istart = iend;
    iend += count[islice];
    if (count[islice] < 10) {
      if (printWarningForTracking)
        printWarningForTracking("IBS suppressed due to fewer than 10 particles in slice.",
                                "This may happen for end slices and may be resolved using more particles");
      zeroslice(islice, IBS);
      /* the inflation still needs this slice's means */
      if (count[islice] > 0) {
        computeSliceParameters(aveCoord, S, part, index, istart, iend, Po);
        memcpy(centroid + 6 * islice, aveCoord, 6 * sizeof(*aveCoord));
      }
      continue;
    }
    computeSliceParameters(aveCoord, S, part, index, istart, iend, Po);
    memcpy(centroid + 6 * islice, aveCoord, 6 * sizeof(*aveCoord));
    IBS->emitx0[islice] = correctedEmittance(S, eta, 0, 1, &betax0, &alphax0);
    IBS->emity0[islice] = correctedEmittance(S, eta, 2, 3, &betay0, &alphay0);
    IBS->emitl0[islice] = GPU_IBS_SAFE_SQRT(S[4][4] * S[5][5] - sqr(S[4][5]));
    IBS->sigmaDelta0[islice] = sqrt(S[5][5]);
    if (IBS->nslice == 1)
      bLength = sqrt(S[4][4]) * c_mks;
    IBS->sigmaz0[islice] = bLength;
    if (!IBS->forceMatchedTwiss)
      forth_propagate_twiss(IBS, islice, betax0, alphax0, betay0, alphay0, run);
    IBS->icharge[islice] = IBS->charge * (double)count[islice] / (double)np;
    IBSRate(fabs(IBS->icharge[islice] / particleCharge),
            IBS->elements, 1, 0, IBS->isRing,
            IBS->emitx0[islice], IBS->emity0[islice], IBS->sigmaDelta0[islice], bLength,
            IBS->s, IBS->pCentral, IBS->betax[islice], IBS->alphax[islice], IBS->betay[islice],
            IBS->alphay[islice], IBS->etax, IBS->etaxp, IBS->etay, IBS->etayp,
            IBS->xRateVsS[islice], IBS->yRateVsS[islice], IBS->zRateVsS[islice],
            &(IBS->xGrowthRate[islice]), &(IBS->yGrowthRate[islice]), &(IBS->zGrowthRate[islice]),
            1, -1);
    IBS->xGrowthRate[islice] *= IBS->factor;
    IBS->yGrowthRate[islice] *= IBS->factor;
    IBS->zGrowthRate[islice] *= IBS->factor;
  }

  /* inflateEmittance() in x and y and inflateEmittanceZ(), slice by slice, on the device */
  if (nslice > 1) {
    if (!(sliceOfRow = (unsigned int *)malloc(np * sizeof(*sliceOfRow))))
      gpuRequiredFailure("unable to allocate IBSCATTER slice numbers");
    iend = 0;
    for (islice = 0; islice < nslice; islice++) {
      long k;
      istart = iend;
      iend += count[islice];
      for (k = istart; k < iend; k++)
        sliceOfRow[index[k]] = (unsigned int)islice;
    }
  }
  for (islice = 0; islice < nslice; islice++) {
    double *a = parameters + GPU_IBS_HOST_SLICE_PARAMETERS * islice;
    zRate[1] = 1. + IBS->dT * IBS->zGrowthRate[islice];
    if (islice == 0)
      zRate[0] = zRate[1];
    else
      zRate[0] = 1. + IBS->dT * IBS->zGrowthRate[islice - 1];
    if (islice == IBS->nslice - 1)
      zRate[2] = zRate[1];
    else
      zRate[2] = 1. + IBS->dT * IBS->zGrowthRate[islice + 1];
    a[0] = centroid[islice * 6 + 0];
    a[1] = centroid[islice * 6 + 1];
    a[2] = centroid[islice * 6 + 2];
    a[3] = centroid[islice * 6 + 3];
    a[4] = centroid[islice * 6 + 5];
    a[5] = centroid[islice * 6 + 4];
    a[6] = sqrt(1. + IBS->dT * IBS->xGrowthRate[islice]);
    a[7] = sqrt(1. + IBS->dT * IBS->yGrowthRate[islice]);
    a[8] = zRate[0];
    a[9] = zRate[1];
    a[10] = zRate[2];
  }
  status = gpuCudaIbsInflate(coord, np, (int)gpuBase.deviceStride, nslice, sliceOfRow, parameters,
                             eta, Po, c_mks, tLength, IBS->dT, &milliseconds);
  if (status != 0)
    gpuFatalStatus("IBSCATTER inflation kernel", status);
  gpuRecordHelperKernel(milliseconds);

  /* ISRING=0: computeSliceParameters() again after the inflation, for every slice */
  if (!IBS->isRing) {
    part = gpuIbsCopyRows(coord, np);
    iend = 0;
    for (islice = 0; islice < nslice; islice++) {
      istart = iend;
      iend += count[islice];
      computeSliceParameters(aveCoord, S, part, index, istart, iend, Po);
      IBS->emitx[islice] = correctedEmittance(S, eta, 0, 1, 0, 0);
      IBS->emity[islice] = correctedEmittance(S, eta, 2, 3, 0, 0);
      IBS->emitl[islice] = GPU_IBS_SAFE_SQRT(S[4][4] * S[5][5] - sqr(S[4][5]));
      IBS->sigmaDelta[islice] = sqrt(S[5][5]);
      IBS->sigmaz[islice] = sqrt(S[4][4]);
    }
  }
  free(index);
  free(count);
  free(centroid);
  free(parameters);
  free(time);
  free(sliceOfRow);
}

/* track_IBS() from the bunch assignment to the end of the bucket loop; the caller goes on to the output */
void gpu_track_IBS(long np0, void *ibs0, double Po, void *run0, void *charge0) {
  IBSCATTER *IBS = (IBSCATTER *)ibs0;
  GPU_BUNCHED_WAKE_PLAN plan;
  double eta[4];
  long nBuckets, relative;

  if (np0 <= 0 || !IBS || !charge0)
    return;
  if (!init_IBS || !IBSRate || !correctedEmittance || !zeroslice || !forth_propagate_twiss ||
      !slicebeam || !computeSliceParameters || !computeTimeCoordinatesOnly)
    gpuRequiredFailure("IBSCATTER host routines are unavailable");
  eta[0] = IBS->etax[IBS->elements - 1];
  eta[1] = IBS->etaxp[IBS->elements - 1];
  eta[2] = IBS->etay[IBS->elements - 1];
  eta[3] = IBS->etayp[IBS->elements - 1];
  if (gpuIbsPlanCache.valid && gpuIbsPlanCache.owner == IBS && gpuIbsPlanCache.particles == np0) {
    plan = gpuIbsPlanCache.plan;
    gpuIbsPlanCache.valid = 0;
  } else if (gpuBunchedWakePlan(np0, IBS->bunchedBeamMode, -1, -1, (CHARGE *)charge0, "IBSCATTER",
                                &plan) == GPU_BUNCHED_WAKE_UNSUPPORTED) {
    gpuRequiredFailure("unsupported IBSCATTER bunch layout reached CUDA path");
  }
  gpuCopyHostToDevice(np0);
  if (plan.needsGather)
    gpuZimpedanceGather(np0, &plan, "IBSCATTER bunch gather kernels");
  nBuckets = plan.useBunchFilter ? plan.nBuckets : 1;
  if (nBuckets > 1 && (!plan.bucketStart || !plan.bucketCount))
    gpuRequiredFailure("IBSCATTER bunch ranges missing on the CUDA path");
  for (relative = 0; relative < nBuckets; relative++) {
    long start = nBuckets > 1 ? plan.bucketStart[relative] : 0;
    long count = nBuckets > 1 ? plan.bucketCount[relative] : np0;
    gpuIbsBucket(IBS, start, count, eta, Po, (RUN *)run0, (CHARGE *)charge0);
  }
  if (plan.needsGather)
    gpuZimpedanceScatter(np0, "IBSCATTER bunch scatter kernel");
  gpuMarkDeviceChanged(np0);
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
                                      wakeData->n_bins && !gpuWakeCpuOrderEnabled(),
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
    if (gpuWakeCpuOrderEnabled())
      tmin = gpuWakeOrderedTimeMean(np0, Po, bunchedPlan,
                                    "WAKE ordered time-sum kernel") -
             wakeData->dt * nb / 2.0;
    else
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
                                      wakeData->n_bins && !gpuWakeCpuOrderEnabled(),
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
    if (gpuWakeCpuOrderEnabled())
      tmin = gpuWakeOrderedTimeMean(np0, Po, bunchedPlan,
                                    "TRWAKE ordered time-sum kernel") -
             wakeData->dt * nb / 2.0;
    else
      tmin = sums.centroidSum[6] / sums.count - wakeData->dt * nb / 2.0;
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
  /* rotateBeamCoordinatesForMisalignment() skips a whole turn */
  data.hasTilt = wakeData->tilt && fabs(fabs(wakeData->tilt) - PIx2) >= 1e-12 ? 1 : 0;
  gpuRotationTiltSinCos(wakeData->tilt, &data.cosTilt, &data.sinTilt);
  data.cpuOrderSums = gpuWakeCpuOrderEnabled() ? 1 : 0;
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

/* RAMPRF's and MODRF's particle loop (add_to_particle_energy() inline), with the CPU's sin for output in
 * CPU order */
static void gpuRfKickCavity(long np, double P_central, double length, double volt, double omega,
                            double phase, double tOffset, long driftOnly, long stopOnBad) {
  float milliseconds = 0;
  long badCount = 0, ip;
  int status;

  if (np <= 0 || (driftOnly && !length))
    return;
  startGpuTimer();
  gpuCopyHostToDevice(np);
  if (driftOnly) {
    status = gpuCudaRamprf(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, P_central,
                           length, 0, 0, 0, 0.0, c_mks, 3, NULL, NULL, NULL, &milliseconds);
    if (status != 0)
      gpuFatalStatus("RAMPRF drift kernel", status);
    gpuRecordHelperKernel(milliseconds);
  } else if (gpuRfcaReferenceMathEnabled()) {
    double *kick;

    gpuEnsureRfcaReferenceScratch(np);
    kick = gpuRfcaScratch.hostReferenceKick;
    status = gpuCudaRamprf(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, P_central,
                           length, volt, omega, phase, tOffset, c_mks, 1, gpuRfcaScratch.referenceKick,
                           NULL, NULL, &milliseconds);
    if (status != 0)
      gpuFatalStatus("RAMPRF phase kernel", status);
    gpuRecordHelperKernel(milliseconds);
    status = gpuCudaCopyDeviceToHost(kick, gpuRfcaScratch.referenceKick, np, &milliseconds);
    if (status != 0)
      gpuFatalStatus("RAMPRF phase transfer", status);
    gpuRecordMilliseconds(&gpuBase.gpuTransferToHostSeconds, milliseconds);
    for (ip = 0; ip < np; ip++)
      kick[ip] = volt * sin(kick[ip]);
    status = gpuCudaCopyHostToDevice(gpuRfcaScratch.referenceKick, kick, np, &milliseconds);
    if (status != 0)
      gpuFatalStatus("RAMPRF kick transfer", status);
    gpuRecordMilliseconds(&gpuBase.gpuTransferToDeviceSeconds, milliseconds);
    milliseconds = 0;
    status = gpuCudaRamprf(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, P_central,
                           length, volt, omega, phase, tOffset, c_mks, 2, NULL,
                           gpuRfcaScratch.referenceKick, &badCount, &milliseconds);
    if (status != 0)
      gpuFatalStatus("RAMPRF kick kernel", status);
    gpuRecordHelperKernel(milliseconds);
    gpuRfcaReferenceCalls++;
    gpuRfcaReferenceParticles += np;
  } else {
    status = gpuCudaRamprf(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, P_central,
                           length, volt, omega, phase, tOffset, c_mks, 0, NULL, NULL, &badCount,
                           &milliseconds);
    if (status != 0)
      gpuFatalStatus("RAMPRF kernel", status);
    gpuRecordHelperKernel(milliseconds);
  }
  gpuMarkDeviceChanged(np);
  gpuRecordWallSeconds();
  if (badCount && stopOnBad) {
    double **coord = copyParticlesToCpuReadOnly("RAMPRF bad coordinate");
    long i = 6;

    for (ip = 0; ip < np; ip++) {
      for (i = 0; i < 6; i++)
        if (isnan(coord[ip][i]) || isinf(coord[ip][i]))
          break;
      if (i != 6)
        break;
    }
    printf("error: bad coordinate for particle %ld in RAMPRF\n", i);
    fflush(stdout);
    for (i = 0; i < 6; i++)
      printf("%15.8e ", coord[ip][i]);
    fflush(stdout);
    printf("\n");
    fflush(stdout);
    abort();
  }
}

/* ramped_rf_cavity()'s particle loop, with the CPU's sin for output in CPU order.  The CPU stops on
 * a coordinate that isn't finite, and so does this, with the CPU's message. */
void gpu_ramped_rf_cavity(long np, double P_central, double length,
                          double volt, double omega, double phase,
                          long driftOnly) {
  gpuRfKickCavity(np, P_central, length, volt, omega, phase, 0.0, driftOnly, 1);
}

/* modulated_rf_cavity()'s particle loop: RAMPRF's with the phase taken from t - tAve; it doesn't stop on a
 * coordinate that isn't finite */
void gpu_modulated_rf_cavity(long np, double P_central, double length, double volt, double omega,
                             double phase, double tAve, long driftOnly) {
  gpuRfKickCavity(np, P_central, length, volt, omega, phase, tAve, driftOnly, 0);
}

/* Each particle's time as modulated_rf_cavity() forms it for its mean, s/(c beta_from_delta(p0, delta)), on
 * the host in particle order; the array is valid until the next call */
const double *gpu_rf_particle_times(long np, double pCentral) {
  static double *times = NULL;
  static long capacity = 0;
  float milliseconds = 0;
  int status;

  if (np <= 0)
    return NULL;
  if (capacity < np) {
    free(times);
    if (!(times = (double *)malloc(np * sizeof(*times))))
      gpuRequiredFailure("unable to allocate the RF particle times");
    capacity = np;
  }
  startGpuTimer();
  gpuCopyHostToDevice(np);
  status = gpuCudaRfParticleTimes(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, pCentral, c_mks, times,
                                  &milliseconds);
  if (status != 0)
    gpuFatalStatus("RF particle time kernel", status);
  gpuRecordHelperKernel(milliseconds);
  gpuRecordWallSeconds();
  return times;
}

/* With output in CPU order, the energy kicks use the CPU's sin, and the matrix method's R12 the
 * CPU's log, from phases and momenta the device makes as simple_rfca.c does.  length is the section
 * whose middle the phase is taken at, 0 for a thin kick.  Returns the device's kicks, or NULL when
 * the kernels make their own; a non-NULL matrixR12 asks for the matrix method's R12 too. */
static inline void gpuRfcaReferenceKickValue(double *kick, double *momentum, long ip,
                                             double volt, double length) {
  kick[ip] = volt * sin(kick[ip]);
  if (momentum) {
    double P = momentum[ip], gamma = sqrt(sqr(P) + 1);
    double dP = sqrt(sqr(gamma + kick[ip]) - 1) - P;
    momentum[ip] = fabs(dP / P) > 1e-14 ? length * (P / dP * log(1 + dP / P)) : length;
  }
}

typedef struct {
  double *kick, *momentum;
  double volt, length;
} GPU_RFCA_REFERENCE_KICKS;

static void gpuRfcaReferenceKickRange(long first, long last, void *context) {
  GPU_RFCA_REFERENCE_KICKS *reference = (GPU_RFCA_REFERENCE_KICKS *)context;
  long ip;

  for (ip = first; ip < last; ip++)
    gpuRfcaReferenceKickValue(reference->kick, reference->momentum, ip, reference->volt,
                              reference->length);
}

static void *gpuRfcaReferenceKicks(long np, double pCentral, double length,
                                   double volt, double omega, double phase,
                                   double timeShift, void **matrixR12) {
  GPU_RFCA_REFERENCE_KICKS reference;
  double *kick, *momentum;
  float milliseconds = 0;
  int status;

  if (matrixR12)
    *matrixR12 = NULL;
  if (np <= 0 || !gpuRfcaReferenceMathEnabled())
    return NULL;
  gpuEnsureRfcaReferenceScratch(np);
  kick = gpuRfcaScratch.hostReferenceKick;
  momentum = gpuRfcaScratch.hostReferenceAux;
  status = gpuCudaRfcaPhaseValues(gpuBase.deviceCoord, gpuRfcaScratch.referenceKick,
                                  matrixR12 ? gpuRfcaScratch.referenceAux : NULL,
                                  np, (int)gpuBase.deviceStride, pCentral, length,
                                  omega, phase, timeShift, c_mks, &milliseconds);
  if (status != 0)
    gpuFatalStatus("RFCA reference phase kernel", status);
  gpuRecordHelperKernel(milliseconds);
  status = gpuCudaCopyDeviceToHost(kick, gpuRfcaScratch.referenceKick, np, &milliseconds);
  if (status != 0)
    gpuFatalStatus("RFCA reference phase transfer", status);
  gpuRecordMilliseconds(&gpuBase.gpuTransferToHostSeconds, milliseconds);
  if (matrixR12) {
    status = gpuCudaCopyDeviceToHost(momentum, gpuRfcaScratch.referenceAux, np, &milliseconds);
    if (status != 0)
      gpuFatalStatus("RFCA reference momentum transfer", status);
    gpuRecordMilliseconds(&gpuBase.gpuTransferToHostSeconds, milliseconds);
  }
  /* Each particle's numbers depend on that particle alone, so the host threads the ISR numbers use
   * give the same result as one: 200,000 particles took 2 ms a cavity on one thread */
  reference.kick = kick;
  reference.momentum = matrixR12 ? momentum : NULL;
  reference.volt = volt;
  reference.length = length;
  gpuHostParallelFor(np, np >= GPU_HOST_THREAD_MIN_SINES ? gpuHostThreads() : 1,
                     gpuRfcaReferenceKickRange, &reference);
  status = gpuCudaCopyHostToDevice(gpuRfcaScratch.referenceKick, kick, np, &milliseconds);
  if (status != 0)
    gpuFatalStatus("RFCA reference kick transfer", status);
  gpuRecordMilliseconds(&gpuBase.gpuTransferToDeviceSeconds, milliseconds);
  if (matrixR12) {
    status = gpuCudaCopyHostToDevice(gpuRfcaScratch.referenceAux, momentum, np, &milliseconds);
    if (status != 0)
      gpuFatalStatus("RFCA reference R12 transfer", status);
    gpuRecordMilliseconds(&gpuBase.gpuTransferToDeviceSeconds, milliseconds);
    *matrixR12 = gpuRfcaScratch.referenceAux;
  }
  gpuRfcaReferenceCalls++;
  gpuRfcaReferenceParticles += np;
  return gpuRfcaScratch.referenceKick;
}

static long gpuRfcaThinKickOnDevice(long np, RFCA *rfca, double **accepted,
                                    double *P_central, double zEnd,
                                    double phase) {
  double omega, volt;
  float milliseconds = 0;
  int status;
  long remaining;
  void *referenceKick = NULL;

  if (np <= 0)
    return np;
  if (!rfca || !P_central)
    gpuRequiredFailure("NULL RFCA thin-kick input in CUDA path");

  omega = PIx2 * rfca->freq;
  volt = rfca->volt / (1e6 * particleMassMV * particleRelSign);
  /* The kick doesn't depend on x or y, but the CPU's offset and its removal round them */
  gpuRfcwApplyCoordinateOffset(np, 0, rfca->dx, "RFCA CUDA thin x-offset kernel");
  gpuRfcwApplyCoordinateOffset(np, 2, rfca->dy, "RFCA CUDA thin y-offset kernel");
  startGpuTimer();
  gpuCopyHostToDevice(np);
  referenceKick = gpuRfcaReferenceKicks(np, *P_central, 0, volt, omega, phase, 0, NULL);
  milliseconds = 0;
  status = gpuCudaRfcaThinKick(gpuBase.deviceCoord, np,
                               (int)gpuBase.deviceStride, *P_central,
                               volt, omega, phase, c_mks, referenceKick, &milliseconds);
  if (status != 0)
    gpuFatalStatus("RFCA thin kick kernel", status);
  gpuRecordHelperKernel(milliseconds);
  gpuMarkDeviceChanged(np);
  gpuRecordWallSeconds();
  gpuRfcwApplyCoordinateOffset(np, 0, -rfca->dx, "RFCA CUDA thin restore x-offset kernel");
  gpuRfcwApplyCoordinateOffset(np, 2, -rfca->dy, "RFCA CUDA thin restore y-offset kernel");
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
  void *referenceKick, *referenceR12;

  if (np <= 0)
    return np;
  if (!rfca || !P_central)
    gpuRequiredFailure("NULL RFCA RF-only matrix input in CUDA path");

  omega = PIx2 * rfca->freq;
  volt = rfca->volt / (1e6 * particleMassMV * particleRelSign);
  startGpuTimer();
  gpuCopyHostToDevice(np);
  gpuEnsureRfcaScratch();
  referenceKick = gpuRfcaReferenceKicks(np, *P_central, rfca->length, volt, omega,
                                        phase, 0, &referenceR12);
  milliseconds = 0;
  status = gpuCudaRfcwRfOnlyMatrixChecked(
    gpuBase.deviceCoord, np, (int)gpuBase.deviceStride, *P_central,
    rfca->length, volt, omega, phase, (int)rfca->end1Focus,
    (int)rfca->end2Focus, rfca->dx, rfca->dy, c_mks, referenceKick,
    referenceR12, gpuRfcaScratch.lostCount, &lostCount, &milliseconds);
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
    double timeShift = rfca->standingWave ? 0 : ik * dtLight;
    void *referenceKick;

    startGpuTimer();
    gpuCopyHostToDevice(np);
    referenceKick = gpuRfcaReferenceKicks(np, *P_central, sectionLength, sectionVolt,
                                          omega, phase, timeShift, NULL);
    milliseconds = 0;
    status = gpuCudaRfcwKickInitial(gpuBase.deviceCoord,
                                    gpuRfcwKickScratch.inverseF, np,
                                    (int)gpuBase.deviceStride, *P_central,
                                    sectionLength, sectionVolt, omega,
                                    phase, timeShift,
                                    (int)(rfca->end1Focus && ik == 0),
                                    c_mks, referenceKick, &milliseconds);
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
  /* DELTA_PHASE: additive offset, added after PHASE as simple_rfca.c does */
  phase += rfca->phase * PI / 180.0;
  phase += rfca->dphase * PI / 180.0;
  *phaseOut = phase;
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
  void *referenceKick, *referenceR12;

  if (np <= 0)
    return;
  if (!rfcw || !P_central)
    gpuRequiredFailure("NULL RFCW RF-only input in CUDA path");

  omega = PIx2 * rfcw->freq;
  volt = rfcw->volt / (1e6 * particleMassMV * particleRelSign);
  startGpuTimer();
  gpuCopyHostToDevice(np);
  referenceKick = gpuRfcaReferenceKicks(np, *P_central, rfcw->length, volt, omega,
                                        phase, 0, &referenceR12);
  milliseconds = 0;
  status = gpuCudaRfcwRfOnlyMatrix(gpuBase.deviceCoord, np,
                                   (int)gpuBase.deviceStride, *P_central,
                                   rfcw->length, volt, omega, phase,
                                   (int)rfcw->end1Focus,
                                   (int)rfcw->end2Focus, dx, dy,
                                   c_mks, referenceKick, referenceR12,
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
                                                double phase, double timeShift) {
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
    length, volt, omega, phase, timeShift, c_mks, &sums, &milliseconds);
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
  double absLengthScale, pBeam;
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
  if (data.dt <= 0) {
    /* The CPU code also can't continue: it exits for identical particles too */
    fprintf(stderr, "Error: RFCW LSC needs a spread in particle times, but all %ld particles have the same time.\n", np);
    exit(1);
  }

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

  pBeam = gpuLscBeamMomentum(&sums, &data, np, binnedCount, Po);

  absLengthScale = fabs(lengthScale);
  if (Imax > 0) {
    kSC = 2 / beamRadius * sqrt(Imax / ipow3(pBeam) / 17045.0);
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
                       data.backtrack, pBeam,
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
                         charge->macroParticleCharge, data.backtrack,
                         gpuLscCpuBeamMomentum(cpuCoord, cpuPbin, np, Po),
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
  return gpuSelectDevice();
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
                                        rfcw->length, volt, phase, 0);
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
    double timeShift = rfcw->standingWave ? 0 : ik * dtLight;
    double dgammaOverGammaAve = 0;
    void *referenceKick;

    if (rfcw->doLSC)
      dgammaOverGammaAve =
        gpuRfcwDgammaOverGammaAveOnDevice(np, rfcw, *P_central,
                                          sectionLength, sectionVolt,
                                          phase, timeShift);

    startGpuTimer();
    gpuCopyHostToDevice(np);
    referenceKick = gpuRfcaReferenceKicks(np, *P_central, sectionLength, sectionVolt,
                                          omega, phase, timeShift, NULL);
    milliseconds = 0;
    status = gpuCudaRfcwKickInitial(gpuBase.deviceCoord,
                                    gpuRfcwKickScratch.inverseF, np,
                                    (int)gpuBase.deviceStride, *P_central,
                                    sectionLength, sectionVolt, omega,
                                    phase, timeShift,
                                    (int)(rfcw->end1Focus && ik == 0),
                                    c_mks, referenceKick, &milliseconds);
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

/* findFiducialTime()'s result without bringing the beam to the host.  The device makes each particle's
 * value with the CPU's expression, and the host picks the first, the one of greatest delta, or adds
 * them in particle order with the CPU's Kahan sum, so the time is the CPU's exactly.  Returns 0 when
 * the CPU routine must do it: the host already has the beam, the beam is spread over MPI ranks, or
 * the mode isn't one of these three. */
static long gpuFiducialTimeFromDeviceValues(long np, double s0, double sOffset,
                                            double p0, unsigned long mode,
                                            double *tFid) {
  static double *value = NULL, *delta = NULL;
  static unsigned char *selected = NULL;
  static long capacity = 0;
  int64_t startPID = -1, endPID = -1;
  float milliseconds = 0;
  long ip, usePmax, verifying = 0;
  int status;

#ifdef GPU_VERIFY
  /* the verify build keeps the host current, and this must still be checked */
  verifying = gpuBase.verifyMode;
#endif
#if USE_MPI
  if (distributedBeam)
    return 0;
#endif
  if (np <= 0 || !tFid || !gpuBase.initialized || gpuBase.activeDevice < 0 ||
      !gpuBase.deviceCoord || !gpuBase.deviceCurrent ||
      (gpuBase.hostCurrent && !verifying))
    return 0;
  if (!(mode & (FID_MODE_FIRST | FID_MODE_PMAX | FID_MODE_TMEAN)))
    return 0;
  if ((mode & FID_MODE_TMEAN) && !(mode & (FID_MODE_FIRST | FID_MODE_PMAX)) && !KahanPlus)
    return 0;
  if (!gpuFiducialPidRange(mode, &startPID, &endPID))
    return 0;
  if (capacity < np) {
    free(value);
    free(delta);
    free(selected);
    value = malloc(sizeof(*value) * np);
    delta = malloc(sizeof(*delta) * np);
    selected = malloc(sizeof(*selected) * np);
    if (!value || !delta || !selected)
      gpuRequiredFailure("unable to allocate the fiducial time buffers");
    capacity = np;
  }
  usePmax = !(mode & FID_MODE_FIRST) && (mode & FID_MODE_PMAX);
  startGpuTimer();
  status = gpuCudaFiducialValues(gpuBase.deviceCoord, np, (int)gpuBase.deviceStride,
                                 p0, sOffset, c_mks, (int)particleIDIndex,
                                 startPID, endPID, value, selected,
                                 usePmax ? delta : NULL, &milliseconds);
  if (status != 0)
    gpuFatalStatus("RF fiducial value kernel", status);
  gpuRecordHelperKernel(milliseconds);
  gpuRecordWallSeconds();
  /* the same order of tests as findFiducialTime() */
  if (mode & FID_MODE_FIRST) {
    for (ip = 0; ip < np; ip++)
      if (selected[ip])
        break;
    if (ip == np)
      bombElegant("No available particle for the FID_MODE_FIRST mode in findFiducialTime", NULL);
    *tFid = value[ip];
  } else if (mode & FID_MODE_PMAX) {
    double best = -DBL_MAX;
    long ibest = -1;

    for (ip = 0; ip < np; ip++)
      if (selected[ip] && best < delta[ip]) {
        best = delta[ip];
        ibest = ip;
      }
    if (ibest < 0)
      bombElegant("No available particle for RF cavity fiducialization", NULL);
    *tFid = value[ibest];
  } else {
    double tsum = 0, error = 0;
    long nsum = 0;

    for (ip = 0; ip < np; ip++)
      if (selected[ip]) {
        tsum = KahanPlus(tsum, value[ip], &error);
        nsum++;
      }
    if (nsum <= 0)
      bombElegant("No available particle for RF cavity fiducilization", NULL);
    *tFid = tsum / nsum;
  }
#ifdef GPU_VERIFY
  if (gpuBase.verifyMode && findFiducialTime)
    compareGpuCpuValue(*tFid,
                       findFiducialTime(copyParticlesToCpuReadOnly("GPU_VERIFY fiducial time"),
                                        np, s0, sOffset, p0, mode),
                       0.0, "findFiducialTime");
#else
  (void)s0;
#endif
  return 1;
}

double gpu_findFiducialTime(long np, double s0, double sOffset,
                            double p0, unsigned long mode) {
  double tFid;
  GPU_BEAM_SUM_DATA result;
  float milliseconds = 0;
  int status = 0;
  int64_t startPID = -1, endPID = -1;

  if (gpuRfcaReferenceMathEnabled() && findFiducialTime) {
    double **coord;

    /* "light" mode reads no particle */
    if (mode & FID_MODE_LIGHT)
      return findFiducialTime(gpuBase.coord, np, s0, sOffset, p0, mode);
    if (gpuFiducialTimeFromDeviceValues(np, s0, sOffset, p0, mode, &tFid))
      return tFid;
    /* Fiducialization only reads coordinates. A mutable synchronization clears
     * elementOnGpu, causing subsequent beam sums to read pre-kick host data
     * even though the cavity continues tracking on the device. */
    coord = copyParticlesToCpuReadOnly("RF CPU-order reference preparation");
    return findFiducialTime(coord, np, s0, sOffset, p0, mode);
  }

  if (mode & FID_MODE_LIGHT)
    return (s0 + sOffset) / c_mks;

  /* The device's values with the host's sum, or the CPU routine when the host has the beam, give the
   * CPU's time exactly and sooner than the one-block sums below (2.8 ms a cavity for 200,000
   * particles), which are left for a beam spread over MPI ranks */
  if (findFiducialTime) {
    if (gpuFiducialTimeFromDeviceValues(np, s0, sOffset, p0, mode, &tFid))
      return tFid;
#if !USE_MPI
    if (np > 0 && gpuBase.hostCurrent)
      return findFiducialTime(copyParticlesToCpuReadOnly("RF reference time"), np, s0, sOffset, p0,
                              mode);
#endif
  }

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
#ifdef GPU_VERIFY
  /* The cavity's phase comes from this, and the CPU shadow then reuses it */
  if (gpuBase.verifyMode && findFiducialTime)
    compareGpuCpuValue(tFid,
                       findFiducialTime(copyParticlesToCpuReadOnly("GPU_VERIFY fiducial time"),
                                        np, s0, sOffset, p0, mode),
                       gpuEnvDouble("ELEGANT_GPU_FIDUCIAL_COMPARE_REL", 1e-12),
                       "findFiducialTime");
#endif
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
  /* DELTA_PHASE: additive offset, added after PHASE as simple_rfca.c does */
  phase += rfca->phase * PI / 180.0;
  phase += rfca->dphase * PI / 180.0;
  *phaseOut = phase;
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
  if (gpuRfcaNoOpElementSupported(eptr))
    return np;
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

/* Element types moved to the device on November 15, 2026 */
#include "gpu_ports_host.h"
