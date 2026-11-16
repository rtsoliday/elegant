#ifndef GPU_PORTS_H
#define GPU_PORTS_H

/* Element types moved to the device on November 15, 2026: the CPU-only rows of feature table 11
 * (history/2026-11-15_sub1x-eval/REPORT.md).  Shared by gpu_stub.c (gpu_ports_host.h) and
 * gpu_cuda_runtime.cu (gpu_ports_device.h).  Each kernel does the CPU routine's arithmetic in its order;
 * a particle the CPU removes is flagged, and the host rebuilds the CPU's swap order from the flags. */

#ifdef __cplusplus
extern "C" {
#endif

/* KPOLY: polynomial_kicks() (kick_poly.c) */
typedef struct GPU_KPOLY_DATA {
  double KL0;
  double cosTilt, sinTilt;
  double dx, dy, dz;
  double coordLimit, slopeLimit;
  double zStart, Po;
  long order;
  int yplane;
} GPU_KPOLY_DATA;

/* HKPOLY: polynomial_hamiltonian() (kick_poly.c) */
typedef struct GPU_HKPOLY_DATA {
  double K[7][7], D[7][7], E[7][7][7];
  double cosTilt, sinTilt;
  double dx, dy, dz;
  double factor; /* hkpoly->factor / hkpoly->nRepeats, as the CPU forms it */
  double coordLimit, slopeLimit;
  double zStart, Po;
  long nRepeats;
  int driftType;
  int ixMax, iyMax, iqxMax, iqyMax, ideltaMax;
  int expandHamiltonian;
} GPU_HKPOLY_DATA;

/* MBUMPER: track_through_mkicker() (kicker.c) after its set-up and phase reference */
#define GPU_MBUMPER_MAX_ORDER 20
typedef struct GPU_MBUMPER_DATA {
  const double *tWaveform, *ampWaveform; /* device copies */
  long nWaveform;
  double tmin, tmax;
  double strength, length;
  double pCentral, cMks;
  double clockOffset, timeOffset; /* trackingClockOffset(), and the phase reference plus TIME_OFFSET */
  double dx, dy, dz;
  int doOffset;
  int doTilt;
  double cosTilt, sinTilt;          /* rotateBeamCoordinatesForMisalignment(tilt) */
  double cosUntilt, sinUntilt;      /* rotateBeamCoordinatesForMisalignment(-tilt) */
  double coordLimit, slopeLimit;
  long order;
  long nParts;
  int periodic;
  int expandHamiltonian;
} GPU_MBUMPER_DATA;

/* RMDF: track_through_ramped_deflector() (track_ramp.c) */
typedef struct GPU_RMDF_DATA {
  double dx, dy;
  double pcCentral, cMks;
  double tNull, Estrength, length;
  double cosTilt, sinTilt;
  long nSections;
} GPU_RMDF_DATA;

/* MRFDF: track_through_multipole_deflector() (track_rf.c) */
typedef struct GPU_MRFDF_DATA {
  double a[5], b[5], frequency[5], phase[5];
  double factor, charge, massC2;
  double pcCentral, cMks, tFirst, pi;
  double cosTilt, sinTilt, cosUntilt, sinUntilt;
  long long startPID, endPID;
  int doTilt;
} GPU_MRFDF_DATA;

/* SHRFDF: track_through_space_harmonic_deflector() (shrfdf.c) */
typedef struct GPU_SHRFDF_DATA {
  double v[10], phase[10];
  double periodPhase, periodLength;
  double factor, charge, massC2;
  double pcCentral, cMks, tFirst, pi;
  double cosTilt, sinTilt, cosUntilt, sinUntilt;
  int doTilt;
} GPU_SHRFDF_DATA;

/* RFTM110: track_through_rftm110_deflector() (track_rf.c), after its host part (waveform, noise) */
typedef struct GPU_RFTM110_DATA {
  double voltTimes2, omega, k, phase0;
  double pcCentral, cMks;
  double cosTilt, sinTilt, cosUntilt, sinUntilt;
  int doTilt;
} GPU_RFTM110_DATA;

/* ALPH: alpha_magnet_tracking()'s puck and slit filters (alpha_matrix.c), with the limits as the CPU's floats */
typedef struct GPU_ALPH_FILTER_DATA {
  double xl, xu;
  double z, Po;
  int slits; /* 0: the puck removes xl <= x <= xu; 1: the slits remove x <= xl (doXl) and x >= xu (doXu) */
  int doXl, doXu;
} GPU_ALPH_FILTER_DATA;

/* KSBEND: track_through_kick_sbend() (kick_sbend.c), its file-level constants and the per-element trig */
typedef struct GPU_KSBEND_DATA {
  double Fy_0, Fy_x, Fy_x2, Fy_x3, Fy_x4, Fy_y2, Fy_x_y2, Fy_x2_y2, Fy_y4;
  double Fx_y, Fx_x_y, Fx_x2_y, Fx_x3_y, Fx_y3, Fx_x_y3;
  double h, ha, radCoef, rho0, n, length;
  double coordLimit, slopeLimit;
  double cosTtilt, sinTtilt;
  double dcoordEtilt[6];
  double dxi, dyi, dzi, dxf, dyf, dzf;
  double e1, e2, psi1, psi2, tanE1, tanE2, secE1, secE2, he1, he2;
  double zStart, Po;
  long nKicks;
  int modifiedMidpoint, paraxial, edge1, edge2, edgeOrder;
} GPU_KSBEND_DATA;

/* BOFFAXE: trackMagneticFieldOffAxisExpansion() (magneticFieldOffAxisExpansion.c) without ISR or output */
typedef struct GPU_BOFFAXE_DATA {
  const double *field; /* device copy of fieldData[nDz1][nz], row by row */
  long nz, nDz1;
  double dataDz;      /* the table's dz */
  double dz;          /* the step: dataDz*Z_INTERVAL/Z_SUBDIVISIONS */
  double zMin, zMax;  /* -/+ half the table's span */
  double zEntry, zExit;
  double pCentral;
  double q, relSign, mass, cMks; /* particleCharge, particleRelSign, particleMass, c_mks, multiplied in order */
  double radCoef;
  double strength;
  double dx, dy, dz0;
  double cosTilt, sinTilt, cosUntilt, sinUntilt;
  long zInterval, zSubdivisions;
  int order, expansionOrder;
  int synchRad, doOffset, doTilt;
} GPU_BOFFAXE_DATA;

/* GFWIGGLER (APPLE): APPLE_Track() and f_ijk() (apple.c); the harmonic tables in one device array */
typedef struct GPU_APPLE_DATA {
  const double *table;
  long NxHarm, NzHarm, ShimHarm;
  /* offsets into table of the [i][j] arrays (NxHarm*NzHarm each, i-major), kz and the shim arrays */
  long kx, ky, CoZ, CxXoZ, CxYoZ, CxXoYZ, CxX2oYZ, CxX2oZ, CxXYoZ, CxX3oYZ, CxY2oZ, kz, Ci_shim, Si_shim;
  double C1, C2, C3, C4, S1, S2, S3, S4;
  double x0, lz, kx_shim, dgap, shimScale;
  double pCentral, eomc;
  double dx, dy, dz;
  double cosTilt, sinTilt, cosUntilt, sinUntilt;
  long periods, step;
  int order, endPole, shimOn, doOffset, doTilt;
} GPU_APPLE_DATA;

/* PFILTER: track_through_pfilter()'s three passes (limit_amplitudes.c) */
#define GPU_PFILTER_PERCENTILE 0 /* delta above (upper) or below the percentile limit */
#define GPU_PFILTER_FIXED 1      /* momentum outside the fixed limits */
#define GPU_PFILTER_DELTA 2      /* |delta - reference| not below DELTALIMIT */
typedef struct GPU_PFILTER_DATA {
  double limit, pUpper, pLower, reference, deltaLimit;
  double z, Po;
  int mode, upper, hasUpper, hasLower;
} GPU_PFILTER_DATA;

/* CLEAN: remove_outlier_particles() (limit_amplitudes.c), after the host's centroids and deviations */
typedef struct GPU_CLEAN_DATA {
  double limit[6], centroid[6], stDev[6];
  double z, Po, cMks;
  int mode; /* 0 stdeviation, 1 absdeviation, 2 absvalue */
} GPU_CLEAN_DATA;

/* EMITTANCE: transformEmittances()'s transformation (do_tracking.c), after the host's beam sums */
typedef struct GPU_EMITTANCE_DATA {
  double factor[2], eta[2];
  int doPlane[2];
} GPU_EMITTANCE_DATA;

/* REMCOR: remove_correlations()'s subtraction (do_tracking.c), after the host's sums */
typedef struct GPU_REMCOR_DATA {
  double ratio[4];
  int active[4];
  int wc;
} GPU_REMCOR_DATA;

/* Packed columns for the host's sums, in particle order */
#define GPU_PORT_COLUMN_TIME 10         /* s/(beta c), beta from the particle's momentum */
#define GPU_PORT_COLUMN_CLEAN_S 11      /* CLEAN's s: the time when every coordinate is finite, else s */
#define GPU_PORT_COLUMN_FINITE 12       /* 1 when every coordinate is finite */

/* SCATTER, DSCATTER, PEPPOT: increments the host made with the CPU's random numbers, applied on the device.
 * Each particle has GPU_INCREMENT_VALUES values: a flag, then the increments of x, x', y, y' and delta. */
#define GPU_INCREMENT_VALUES 6
typedef struct GPU_INCREMENT_DATA {
  double Po;
  int active[4];  /* which of x, x', y, y' take an increment */
  int dpActive;   /* delta too, with the path length kept as the time (SCATTER's dp, DSCATTER's dp plane) */
} GPU_INCREMENT_DATA;

/* MATTER: track_through_matter() (matter.c) without slots, bremsstrahlung or electron recoil, with the
 * host's random numbers.  Each particle has GPU_MATTER_VALUES values: in the multiple-scattering modes
 * (1, 2) the Gaussian numbers (four, or two), in the single-scattering mode (0) the first of its events
 * and their count; then dgamma (energy decay) and 1 when lost.  An event is GPU_MATTER_EVENT_VALUES
 * values: its section, the scattering point in it, and the kicks to x' and y'. */
typedef struct GPU_MATTER_DATA {
  double L, thetaRms, Po, z0;
  double L1;     /* mode 0: a section's length */
  long sections; /* mode 0 */
  int multipleScattering; /* 0, 1 or 2 */
  int energyDecay;
  /* deviceDraws: the device makes the scattering numbers from random_2's sequence.  Modes 1 and 2: v[0] is
   * the state of a particle's first number, or, with no values at all (no energy decay), particle ip's
   * is first * stride^ip.  Mode 0: an event is its section and the state of its numbers. */
  int deviceDraws;
  double K2;     /* mode 0 */
  unsigned long long first, stride;
} GPU_MATTER_DATA;
#define GPU_MATTER_VALUES 6
#define GPU_MATTER_EVENT_VALUES 4

/* SCATTER and DSCATTER with the device making the numbers.  The host walks random_2's sequence for the
 * CPU's decisions (whether a particle scatters, and DSCATTER's limits) and finds where each scattered
 * particle's numbers start; the device makes them from there, as random_2, gauss_rn() and interp() do.
 * A state (as a double: it has 48 bits) of 0 means no scattering; with no states, particles 0 to count-1
 * scatter, particle ip's numbers starting at first * stride^ip. */
typedef struct GPU_DRAW_DATA {
  GPU_INCREMENT_DATA inc;
  double sigma[4], dp; /* SCATTER */
  int gaussian;        /* SCATTER */
  int dscatter;
  int column;          /* DSCATTER: 1 to 4 for x to y', 5 for delta */
  int randomSign;      /* DSCATTER */
  double factor;       /* DSCATTER */
  const double *indep, *cdf; /* DSCATTER's table */
  long nData;
  long count;
  unsigned long long first, stride;
} GPU_DRAW_DATA;

/* PEPPOT: pepper_pot_plate()'s hole test (pepper_pot.c) */
typedef struct GPU_PEPPOT_DATA {
  const double *x, *y; /* device copies of the hole centers */
  long nHoles;
  double radius2;
  double length; /* drift before the test (the second pass), 0 for none */
} GPU_PEPPOT_DATA;

/* MHISTOGRAM: one histogram for the device's fill (gpu_mhist_fill()) */
#define GPU_MHIST_BOOKS 6
typedef struct GPU_MHIST_BOOK {
  int nD, bins, kind; /* kind 0: chfill1m() (bins is the book's bins); 1: chfilln() at offset */
  int offset;
  int xbins[6];
  int column[6]; /* chfill1m(): the coordinate each dimension reads */
  double xmin[6], dx[6];
  long countOffset;          /* into the counts of all the books */
  unsigned long long *count; /* set on the device */
} GPU_MHIST_BOOK;

/* BUMPER (KICKER): track_through_kicker() (kicker.c), the waveform and any deflection map uploaded once */
typedef struct GPU_KICKER_DATA {
  const double *tWaveform, *ampWaveform; /* set on the device */
  const double *xpFactor, *ypFactor;     /* the deflection map, set on the device */
  long nWaveform, nKicks, nx, ny, mapValues;
  double tmin, tmax, angle, length, b2, pCentral, cMks, clockOffset, timeOffset;
  double xmin, ymin, dxg, dyg;
  double dx, dy, dz, cosTilt, sinTilt, cosUntilt, sinUntilt;
  int periodic, doOffset, doTilt, deflectionMap;
} GPU_KICKER_DATA;

/* The particles left after the element, or -1 when the CPU must track it (the host falls back). */
long gpu_track_kpoly(double **particle, long nParticles, void *element, double Po, double **accepted,
                     double zStart);
long gpu_track_hkpoly(double **particle, long nParticles, void *element, double Po, double **accepted,
                      double zStart);
/* 1 when the device tracked the call, 0 when the CPU must */
long gpu_track_mbumper(double **particle, long nParticles, void *element, double pCentral, long pass,
                       long defaultOrder);
long gpu_track_rmdf(double **particle, long nParticles, void *element, double pCentral);
long gpu_track_mrfdf(double **particle, long nParticles, void *element, double pCentral, long pass);
long gpu_track_shrfdf(double **particle, long nParticles, void *element, double pCentral);
long gpu_track_rftm110(double **particle, long nParticles, void *element, double pCentral, double lCentral,
                       double zEnd, long pass);
/* the particles left */
long gpu_track_alph(double **particle, void *matrix, void *element, long nParticles, double **accepted,
                    double pCentral, double z);
long gpu_track_ksbend(double **particle, long nParticles, void *element, double Po, double **accepted,
                      double zStart);
/* 1 when tracked, 0 when the CPU must */
long gpu_track_kicker(double **particle, long nParticles, void *element, double pCentral, long pass, long defaultOrder);
/* 1 when tracked, 0 when the CPU must */
long gpu_track_boffaxe(double **particle, long nParticles, void *element, double pCentral);
long gpu_track_apple(double **particle, long nParticles, void *element, double pCentral);
/* the particles left */
long gpu_track_pfilter(double **particle, void *element, long nParticles, double **accepted, double z, double Po);
long gpu_track_sample(double **particle, void *element, long nParticles, double **accepted, double z, double p0);
long gpu_track_clean(double **particle, void *element, long nParticles, double **accepted, double z, double Po);
/* 1 when tracked */
long gpu_track_emittance(double **particle, long nParticles, double pCentral, void *element);
long gpu_track_remcor(double **particle, void *element, long nParticles);
/* MHISTOGRAM's min, max and time mean (findMinMax()), and its fill loop (mhist_table()), on the device;
 * 0 when the CPU must */
long gpu_mhist_minmax(long nParticles, double Po, double *min, double *max, double *c0);
long gpu_mhist_fill(long nParticles, double Po, double t0, void *book1d, int32_t *bins1d, void *x2d, void *y2d,
                    void *z2d, void *tr4d, void *full6d);
/* After the host changed one coordinate of every particle of a beam it holds a current copy of (CKICKER):
 * the column to the device, so both stay current */
void gpu_port_upload_column(long nParticles, long column);
/* Around a CPU routine given a host copy of a beam whose element is on the device: its own helpers
 * (compute_centroids(), accumulate_beam_sums()) must not go to the device.  begin returns what end
 * restores. */
long gpu_port_host_begin(void);
void gpu_port_host_end(long saved);
/* SCATTER and DSCATTER (the host draws elegant's random numbers in its order); a case the device doesn't
 * take runs the CPU routine on the host */
void gpu_track_scatter(double **part, long np, double Po, void *element, long iPass);
void gpu_track_dscatter(double **part, long np, double Po, void *element, long iPass);
/* the particles left */
long gpu_track_matter(double **part, long np, long iPass, void *element, double Po, double **accepted, double z0);
long gpu_track_peppot(double **part, void *element, long np, double **accepted);
/* After the host changed columns of a beam whose current copy it holds (CBSCAT, IONEFFECTS) */
void gpu_port_upload_columns(long nParticles, const long *columns, long nColumns);
/* IONEFFECTS: the Gaussian ion field's kicks to a bunch of electrons (applyIonKicksToElectronBunch()) on the
 * device's beam; rows are the bunch's rows in the beam, NULL for all */
void gpu_ioneffects_kicks(const int64_t *rows, long np, const double *centroid, const double *sigma, double charge,
                          double Po);
/* After the device changed a beam whose host copy an element was reading (IONEFFECTS) */
void gpu_port_device_changed(long nParticles);

/* SLICE: dump_slice_analysis()'s numbers (slicePoint.c), the CPU's to the bit: the times, the extremes and each
 * slice's members on the device, their sums in particle order on the host.  The columns are x, x', y, y', s, delta
 * and the time; the pairs are the second moments accumulate_beam_sums() makes with BEAM_SUMS_SPARSE. */
#define GPU_SLICE_COLUMNS 7
#define GPU_SLICE_SIGMA_PAIRS 15
#define GPU_SLICE_SIGMA_PAIR_LIST                                                                                \
  {{0, 0}, {0, 1}, {0, 5}, {1, 1}, {1, 5}, {2, 2}, {2, 3}, {2, 5}, {3, 3}, {3, 5}, {4, 4}, {4, 5}, {4, 6}, {5, 5}, \
   {6, 6}}
/* The particles' times (with clockOffset added, PID range startPID..endPID as dump_slice_analysis() takes it)
 * and the bunch's earliest and latest; 0 when the host must make them */
long gpu_slice_point_times(long nParticles, double Po, double clockOffset, long startPID, long endPID,
                           double *tMin, double *tMax);
/* After gpu_slice_point_times(): for each slice t0[k] to t0[k]+dt0, its count, its columns' sums and the
 * GPU_SLICE_SIGMA_PAIRS sums of products about the means as accumulate_beam_sums() divides them (PID range as
 * accumulate_beam_sums() takes it); 0 when the host must make them */
long gpu_slice_point_sums(long nParticles, long startPID, long endPID, long nSlices, const double *t0, double dt0,
                          long *count, double *columnSum, double *sigmaSum);

#ifdef __cplusplus
}
#endif

#endif
