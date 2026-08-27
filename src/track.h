/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: track.h
 *       contains definitions used to manipulate mad-format lattices,
 *       and to do tracking through such lattices.
 *       Lengths and multipoles are expressed in meters^n, angles in
 *       radians.
 * The external arrays are in track_data.c
 *
 * Michael Borland, 1987
 */

#include "pressureData.h"

/* Define USE_MPI to be 1 in the Makefile for compiling parallel elegant. */

#ifndef USE_MPI
#define USE_MPI 0
#endif

#if USE_MPI 
#include "mpi.h" /* Defines the interface to MPI allowing the use of all MPI functions. */
#include "pgapack.h"
#if USE_MPE
#include "mpe.h" /* Defines the MPE library */ 
#endif
#include <float.h>
#if !defined(_WIN32)
#include <unistd.h>
#endif
#endif 

#include <stdio.h>
#include <stdbool.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "namelist.h"
#include "SDDS.h"
#include "rpn.h"
#include "table.h"
#include "chbook.h"
#include "matrixOp.h"
#include "matlib.h"

#include "manual.h"

#if defined(_WIN32)
#include <float.h>
#include <math.h>
#include <io.h>

/*#define isnan(x) _isnan(x)*/
#define dup2(x,y) _dup2(x,y)
#endif

#ifndef STANDARD
#include "standard.h"
#endif
#ifndef HASHTAB
#include "hashtab.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Convert p=beta*gamma to T*m with this factor */
#define RIGIDITY_FACTOR ((me_mks*c_mks)/e_mks)
/* Convert B*lambda to K with this factor */
#define UNDULATOR_K_FACTOR (1/(RIGIDITY_FACTOR*PIx2))

extern double particleMass, particleCharge, particleMassMV, particleRadius, particleRelSign,
   particleAnomalousMagneticMoment;
extern long particleIsElectron;

  /* coordinates are (x, xp, y, yp, s, delta) */
#define COORDINATES_PER_PARTICLE 6
  /* input/output coordinates are (x, xp, y, yp, s, delta, particle ID) */
#define IO_COORDINATES_PER_PARTICLE 7
  /* Basic properties are particle ID, pass of loss, bunch number. They are always present. 
   * If pass of loss is negative, particle is active */
#define BASIC_PROPERTIES_PER_PARTICLE 3
#define particleIDIndex COORDINATES_PER_PARTICLE
#define lossPassIndex (COORDINATES_PER_PARTICLE+1)
#define bunchIndex (COORDINATES_PER_PARTICLE+2)
  /* Global loss properties are X, Z, thetaX, of loss. Memory for these won't be allocated unless requested. *
   * NB: we presently assume that the beamline is in the Y=0 plane, so Y=y and thetaY=atan(yp) */
#define GLOBAL_LOSS_PROPERTIES_PER_PARTICLE 3
  /* Spin properties are (Sx, Sy, Sz) */
#define SPIN_PROPERTIES_PER_PARTICLE 3
  /* This can be used to safely over-size arrays when we don't care about saving space */
#define MAX_PROPERTIES_PER_PARTICLE (COORDINATES_PER_PARTICLE+BASIC_PROPERTIES_PER_PARTICLE+GLOBAL_LOSS_PROPERTIES_PER_PARTICLE+SPIN_PROPERTIES_PER_PARTICLE)
  /* These will be computed based on run-time needs */
extern int totalPropertiesPerParticle, globalLossCoordOffset, spinCoordOffset;
extern size_t sizeOfParticle;

/* number of sigmas for gaussian random numbers in radiation emission simulation in CSBEND, KQUAD, etc. */
  extern double srGaussianLimit;

/* various user-controlled global flags and settings (global_settings namelist) */
extern long inhibitFileSync;
extern long allowOverwrite;
extern long echoNamelists;
extern long mpiRandomizationMode;
extern long exactNormalizedEmittance;
extern long shareTrackingBasedMatrices;
extern long trackingBasedMatricesStoreLimit;
extern long trackingBasedDiffusionMatrixParticles;
/* If 1, the per-element tracking-based matrix computation (used for twiss
 * output, optimization, etc.) is parallelized across ranks.  Set to 0
 * (forced) by do_parallel_optimization_setup, because rank-parallel
 * optimization already uses every rank for its own beam and adding
 * parallelism inside matrix-determination tracking would either deadlock
 * or do nothing useful. */
extern long parallelTrackingBasedMatrices;
extern double trackingMatrixStepFactor;
extern long trackingMatrixPoints, trackingMatrixMaxFitOrder;
extern double trackingMatrixStepSize[6];
extern long warningCountLimit;
extern short misalignmentMethod, trackingMatrixCleanUp;
extern double slopeLimit, coordLimit, sStart;
extern char *searchPath;

/*------------------------------------------------------------------------------
 * Pelegant parallelism model -- a reference for the flags below.
 *
 * Pelegant uses a master-slave layout: rank 0 is the "master" and ranks
 * 1..N-1 are "slaves" (a.k.a. workers).  The master normally handles
 * namelist parsing, controls flow, and writes output files; slaves do
 * the heavy tracking arithmetic in parallel.
 *
 * Two orthogonal "beam-distribution mode" choices exist.  These describe
 * *how the user's beam is partitioned across MPI ranks*.  Both modes are
 * decided at beam-setup time (sdds_beam / bunched_beam) and held for the
 * duration of a tracking run:
 *
 *   distributedBeam -- the beam-partition flavour.
 *     1: "true parallel" -- the particles of one beam are *distributed*
 *        across the slave ranks (each slave holds a subset of the
 *        particles; master holds none for tracking).  This is the
 *        standard Pelegant mode when you have many particles and want
 *        them split N ways across the slaves.
 *     0: "replicated-beam" -- every rank holds a complete copy of the
 *        beam.  Used both when the same beam is intentionally tracked
 *        redundantly on every rank (closed-orbit search,
 *        orbit/tune/chromaticity correction, fiducial beam) AND when
 *        each rank holds its OWN beam (rank-parallel optimization and
 *        dynamic-aperture search; see independentRunPerRank below).
 *     Note: distributedBeam=0 does NOT mean the beam has one particle.
 *     It means the beam is NOT split across ranks -- each rank holds a
 *     full beam (which itself may have many particles).
 *
 *   independentRunPerRank -- distinguishes the two distributedBeam=0 cases.
 *     1: each rank holds its OWN beam and tracks INDEPENDENTLY with
 *        different parameter settings (parallel optimization,
 *        rank-parallel dynamic-aperture search).  No inter-rank
 *        particle data flow during tracking.
 *     0: every rank tracks the SAME beam redundantly (closed-orbit
 *        search, orbit correction) OR true-parallel (distributedBeam=1).
 *     Set in do_parallel_optimization_setup for any optimization method
 *     other than simplex.  Used by track_through_X routines to decide
 *     whether cross-rank MPI collectives over MPI_COMM_WORLD / workers
 *     would deadlock (when 1, they would, because ranks are diverging).
 *
 * Two runtime "particle-location" state variables track WHERE the
 * particle arrays are AT THIS MOMENT during a tracking pass.  They are
 * meaningful only when distributedBeam=1; otherwise particles are
 * replicated on every rank and the concept doesn't apply.  Their values
 * can change multiple times during a single tracking call as elements
 * scatter to slaves (parallel elements) and gather to master
 * (UNIPROCESSOR elements):
 *
 *   parallelStatus -- the current scatter/gather state.
 *     trueParallel: particles are CURRENTLY distributed across the
 *       slaves (each slave has its subset; master has none).  The
 *       previous tracking element finished with scattered particles
 *       and the next element can run in parallel directly.
 *     notParallel: particles are CURRENTLY gathered onto the master
 *       (or in the process of having been gathered).  A subsequent
 *       parallel element will need to scatter them again.  Entered
 *       when a UNIPROCESSOR element runs.
 *     initialMode: transitional initial state before the first
 *       scatter has happened (rarely seen in practice; the initial
 *       value is trueParallel).
 *
 *   partOnMaster -- does the master rank currently hold particle data?
 *     1: yes (either we just gathered for a UNIPROCESSOR element, or
 *        we are in distributedBeam=0 / independentRunPerRank mode where the
 *        master also holds a beam).
 *     0: no (particles are on slaves only; this is the typical
 *        distributedBeam=1 + parallelStatus=trueParallel state).
 *
 *   The two are closely related but not redundant: parallelStatus
 *   describes whether a SCATTER is needed before the next parallel
 *   element runs, while partOnMaster describes whether the master's
 *   particle buffer is currently meaningful.
 *
 * Truth table (decided at beam setup, then held):
 *
 *   distributedBeam  independentRunPerRank   typical use
 *   -------------  -------------------   -----------
 *        1                 0             standard Pelegant: particles
 *                                        split across slaves; master
 *                                        does NOT track
 *        0                 0             closed-orbit, orbit/tune/chrom
 *                                        correction, fiducial beam:
 *                                        every rank tracks the same
 *                                        replicated beam redundantly
 *        0                 1             rank-parallel optimization
 *                                        (sample, randomsample, swarm,
 *                                        genetic, hybsimplex) and
 *                                        rank-parallel aperture search:
 *                                        every rank tracks its OWN
 *                                        independent beam with its OWN
 *                                        parameter point
 *        1                 1             not used in practice
 *
 *----------------------------------------------------------------------------*/

/* Permission to write to output files (only on the rank elected to write). */
extern long writePermitted;
/* Per-rank role flags.  isMaster <=> (myid==0); isSlave <=> (myid!=0).  Set
 * once at MPI startup. */
extern long isMaster;
extern long isSlave;
/* Beam-partition mode and rank-parallel-optimization mode.  See the
 * truth table above. */
extern long distributedBeam;
extern long independentRunPerRank;
/* A hash table for loading parameters effectively */
extern htab *load_hash;
/* A factor used for distibuting memory on slave processors */
extern double memDistFactor;
/* Set when do_tracking() is called with TEST_PARTICLES flag (closed-orbit
 * search, single-particle trajectory probes).  Some elements use this to
 * skip collective bookkeeping that isn't meaningful for a probe particle. */
extern long trajectoryTracking;

#if USE_MPI
/* parallelStatus values: see the comment block above for full semantics.
 *   notParallel  -- particles are gathered on master right now
 *   initialMode  -- transitional pre-first-scatter
 *   trueParallel -- particles are scattered across slaves right now
 */
typedef enum pMode {notParallel, initialMode, trueParallel} parallelMode;
extern parallelMode parallelStatus;
/* MPI rank-count and this rank's id. */
extern int n_processors;
extern int myid;
/* Does the master hold a meaningful particle buffer right now?  See
 * comment block above. */
extern int partOnMaster;
/* If 1, the n_particles >= n_processors-1 requirement is relaxed.  Set
 * for fiducial-beam tracking, rank-parallel runs (each rank has its own
 * small beam), and aperture search (each rank probes one test particle). */
extern long lessPartAllowed;
/* Sub-communicator containing slave ranks ONLY (rank 0 excluded).  Used
 * by collective-effect tracking routines for inter-slave reductions that
 * shouldn't include the master in true-parallel mode. */
extern MPI_Comm workers;
extern int fdStdout; /* save the duplicated file descriptor stdout to use it latter */
extern int dumpAcceptance; /* A flag to indicate if the initial coordinates of transmitted particles will be dumped */
extern long do_find_aperture; /* A flag to set singlePart tracking in dynamic aperture optimization for Pelegant */
extern long watch_not_allowed; /* A flag to indicate the watch point is not allowed for aperture searching for Pelegant */
extern long last_optimize_function_call; /* A flag used to indicate if it is the last optimization function call (after exiting the optimization loop)  */
extern int min_value_location; /* The location (which processor) of the optimization value */
/* A flag used to specify if the output will be enabled for some special cases. E.g., simplex method in Pelegant */
extern long enableOutput;
/* used to abort MPI run in cases where master is not running the same code */
#define MPI_ABORT_BUNCH_TOO_LONG_ZLONGIT 1
#define MPI_ABORT_BUNCH_TOO_LONG_ZTRANSVERSE 2
#define MPI_ABORT_BUNCH_TOO_LONG_RFMODE 3
#define MPI_ABORT_BUNCH_ASSIGNMENT_ERROR 4
#define MPI_ABORT_POINTER_ISSUE 5
#define MPI_ABORT_BAD_PARTICLE_ID 6
#define MPI_ABORT_RF_FIDUCIALIZATION_ERROR 7
#define MPI_ABORT_SREFFECTS_ERROR1 8
#define MPI_ABORT_PARTICLE_TUNE_IO_ERROR 9
#define MPI_ABORT_RESPONSE_MATRIX_SHARING 10  
#define N_MPI_ABORT_TYPES 11
extern char *mpiAbortDescription[N_MPI_ABORT_TYPES];
void doMpiAbort(int code, char *format, ...);
#endif
extern short mpiAbort;

#ifndef __cplusplus
extern long remaining_sequence_No, orig_sequence_No; /* For Pelegant regression test */
#endif


#ifdef SORT
extern int comp_IDs(const void *coord1, const void *coord2);
#endif

#define malloc_verify(n) 1

typedef struct {
  double *sz, *xMax, *yMax, *dx, *dy;
  long points;
  short periodic, initialized, persistent;
  short zmode; /* If non-zero, data is function of z not s */
} APERTURE_DATA;
extern void readApertureInput(APERTURE_DATA *apData, char *input, short zmode);

typedef struct {
  double *Z, *X;
  double Y;
  long points;
  int32_t canGo;
} OBSTRUCTION_DATASET;

typedef struct {
  short initialized, hasCanGoFlag;
  long periods;
  int32_t superperiodicity;
  double center[2]; /* Z, X */
  double YLimit[2]; /* low, high limits */
  double YSpacing;  /* if multiple planes of data are used */
  double *YValue;   /* for each plane */
  double YMax, YMin;
  long nY, iY0;
  OBSTRUCTION_DATASET **data;
  long *nDataSets;
} OBSTRUCTION_DATASETS;

/* Variable-order transport matrix structure */

typedef struct {
    double *C, **R, ***T, ****Q;
    long order;
    double *maxError, *meanAbsError;
    /* These are needed by radiation calculations */
    struct element_list *eptr;  /* address of element structure, if any */
    } VMATRIX;

/* structure for general multipole kicks */

typedef struct {
  long initialized;
  long randomized;         /* used for random multipoles */
  long orders;
  int32_t *order;
  double referenceRadius;  
  int32_t referenceOrder;
  char *filename;
  /* normal and skew terms */
  double *an, *bn;         /* input values for normal and skew terms, respectively (note difference from common use!),
			      for KQUAD, KSEXT, others */
  double *anMod, *bnMod;   /* computed values: anMod=an*n!/r^n, bnMod=bn*n!/r^n */
  double *KnL, *JnL;       /* provided as input for FMULT, but computed for KQUAD, KSEXT, etc. */
  short copy;              /* if non-zero, pointers are copies of another structure and shouldn't be freed or reallocated */
} MULTIPOLE_DATA ;

/* structure used by polynomial series map element */
typedef struct {
  long mapInitialized;
  long terms;
  int32_t *Ix, *Iqx, *Iy, *Iqy, *Is, *Idelta;
  double *Coefficient;
} POLYNOMIALSERIES_DATA ;

/* structure for storing Twiss parameters */
typedef struct {
  /* the order of the items in this structure should not be
   * changed! 
   */
  double betax, alphax, phix, etax, etapx, apx;
#define TWISS_Y_OFFSET 6
  double betay, alphay, phiy, etay, etapy, apy;
#define TWISS_CENT_OFFSET 12
  double Cx, Cy;
#define TWISS_RAD_INTEGRALS_OFFSET 13
  double dI[6];
  short periodic;  /* kind of wasteful... */
} TWISS;

/* extended structure for storing Twiss parameters (x, y, z) for a beam */
typedef struct {
  double beta[3], alpha[3], eta[4];
  double centroid[6], emit[3];
} TWISSBEAM;

/* Sigma matrix for beam moments computations */
typedef struct {
  /* 21 unique elements from upper triangular part of the 
   * sigma matrix, stored in row order */
  double sigma[21];  
} SIGMA_MATRIX;

/* structure for accumulating beam moments */

#define DO_NORMEMIT_SUMS 0

typedef struct {
    double sigma[7][7];  /* sigma[i][j] = Sum((x[i]-c[i])*(x[j]-c[j])/n) */
    double sigman[7][7]; /* sigman[i][j] = Sum((x[i]-c[i])*(x[j]-c[j])/n), for normalized coordinates */
    double maxabs[7];    /* maximum values for x, xp, y, yp, max deviation for s, max value for dp/p, max deviation for t */
    double min[7];
    double max[7];
} BEAM_SUMS2;

typedef struct {
  double centroid[3];
  double sigma[3][3];
} SPIN_SUMS;
  
typedef struct {
    double centroid[7];  /* centroid[i] = Sum(x[i]/n), i=6 is time */
    BEAM_SUMS2 *beamSums2; /* second-order correlations, etc. */
    SPIN_SUMS *spinSums;   /* centroid and rms for spin */
    int64_t n_part;         /* number of particles */
    double z;            /* z location */
    double p0;           /* reference momentum (beta*gamma) */
    double charge;       /* charge in beam */
    long pass;           /* pass through system */
    } BEAM_SUMS;

typedef struct {
  double centroid[6]; /* X, theta, Y, phi, Z, delta */
  int64_t n_part;
} GLOBAL_BEAM_SUMS;
  
typedef struct {
    double c1, c2;          /* centroids for two coordinate planes */
    double min1, min2;      /* minimum value for each plane */
    double max1, max2;      /* maximum value for each plane */
    double s11, s12, s22;   /* <(xi-<xi>)*(xj-<xj>)> */
    double emittance;       /* sqrt(s11*s22-s22^2) */
    double S1, S2;          /* sqrt(<xi^2>) */
    } ONE_PLANE_PARAMETERS;

/* Node structure for linked-list of element definitions: */

typedef struct element_list {
    double beg_pos, end_pos, end_theta;
    double floorCoord[3], floorAngle[3]; /* (X, Y, Z), (theta, phi, psi) at exit */
    char *name, *group;
    char *definition_text;
    char *p_elem;        /* pointer to the element structure */
    char *p_elem0;       /* pointer to the reference element structure */
    long type;
    long occurence;     /* greater than 1, if assigned */
    long flags;
#define PARAMETERS_ARE_STATIC    0
#define PARAMETERS_ARE_VARIED    1
#define PARAMETERS_ARE_PERTURBED 2
#define VMATRIX_IS_VARIED 4
#define VMATRIX_IS_PERTURBED 8
    short chamberShape;
#define ROUND_CHAMBER 0
#define RECTANGULAR_CHAMBER 1
#define ELLIPTICAL_CHAMBER 2
#define SUPERELLIPTICAL_CHAMBER 3
#define UNKNOWN_CHAMBER 4
#define N_CHAMBER_SHAPES 5
    short reversed; /* used only when saving a beamline with output_seq!=0 */
    double Pref_input, Pref_output;
    double Pref_output_fiducial;
    VMATRIX *matrix;      /* pure matrix of this element */
    VMATRIX *savedMatrix; /* saved matrix of this element */
    VMATRIX *accumMatrix; /* accumulated matrix to the end of this element */
    TWISS *twiss;         /* computed from the above matrices */
    VMATRIX *Mld0;        /* individual linear damping matix */
    VMATRIX *Mld;         /* accumulated linear damping matrix (on-orbit, with trajectory ) */
    SIGMA_MATRIX *sigmaMatrix;
    char *part_of;     /* name of lowest-level line that this element is part of */
    struct element_list *pred, *succ;
    short ignore, firstOfDivGroup;
    double *Drad;         /* 21-element radiation diffusion matrix for this element */
    double *DIbs;         /* 21-element IBS diffusion matrix for this element */
    double coulombLog;
    double *Dgas;         /* 21-element elastic gas-scattering diffusion matrix for this element */
    double *D;            /* Sum of radiation, IBS, and gas diffusion matrix for this element */
    double *accumD;       /* accumulated radiation+IBS+gas diffusion matrix up to end of this element */
    long divisions;       /* if element was subdivided, how many times */
    size_t namelen;
    } ELEMENT_LIST;

extern char *chamberShapeChoice[N_CHAMBER_SHAPES];

typedef struct {
    double centroid[6];
    int64_t n_part;
    ELEMENT_LIST *elem;
    } TRAJECTORY;

/* structure to store data on links between elements */
typedef struct {
    char **target_name;            /* names of target elements */
    long *target_occurence;        /* occurence of target, if given */
    ELEMENT_LIST ***target_elem;   /* arrays of pointers to target element pointers */
    long *n_targets;               /* number of targets with given name */
    char **item;                   /* names of items to be changed */
    double *initial_value;         /* initial value of the parameter */
    double *minimum, *maximum;     /* user's limits on the parameter value */
    double **baseline_value;       /* baseline value after initial change/perturbation */
    long *target_param;            /* parameter (item) type code */
    char **source_name;            /* names of source elements, parallel to target_name */
    long *source_position;         /* before, after, etc. */
    long *flags;
#define STATIC_LINK 1
#define DYNAMIC_LINK 2
#define POST_CORRECTION_LINK 4
#define TURN_BY_TURN_LINK 8
#define LINK_ELEMENT_DEFINITION 16
#define EXCLUDE_SELF_LINK 32
    ELEMENT_LIST ***source_elem;   /* arrays of pointers to source element pointers, parallel to target_elem */
    long *n_parameters;            /* numbers of parameters for each source */
    char **equation;               /* rpn equations for items in terms of parameters of source */
    long n_links;
    } ELEMENT_LINKS;

typedef struct {
  short active;
  char *elementName;
  long elementOccurence, deltaPosition;
  short ringMode;
} CHANGE_START_SPEC;

typedef struct {
  short active;
  char *elementName;
  long elementOccurence, deltaPosition;
} CHANGE_END_SPEC;

/* radiation integrals and related values.  See SLAC 1193. */
typedef struct {
  short computed;
  double RI[6];
  double Jx, Jy, Jdelta;
  double taux, tauy, taudelta;
  double ex0, sigmadelta, Uo, Pref;
  /* Sokolov-Ternov self-polarization (planar-ring sign-correct DK).
   * I3pos / I3neg split RI[2] (= I3) by sign of bend angle so anti-parallel
   * dipoles and wigglers suppress Peq.  tauP is the polarization time
   * (s); Peq the equilibrium polarization; PstLimit = 8/(5*sqrt(3)) the
   * ideal Sokolov-Ternov limit for a single-direction planar ring. */
  double I3pos, I3neg;
  double tauP, Peq, PstLimit;
} RADIATION_INTEGRALS;

typedef struct {
  /* First-order geometric terms (abs, real, imag) */
  double h21000[3], h30000[3], h10110[3], h10020[3], h10200[3];
  /* First order chromatic terms */
  double h11001[3], h00111[3], h20001[3], h00201[3], h10002[3];
  /* First order coupling terms */
  double h10010[3], h10100[3];
  /* Second-order geometric terms */
  double h22000[3], h11110[3], h00220[3], h31000[3], h40000[3];
  double h20110[3], h11200[3], h20020[3], h20200[3], h00310[3], h00400[3];
  /* tune shifts with amplitude */
  double dnux_dJx, dnux_dJy, dnuy_dJy;
} DRIVING_TERMS;

typedef struct {
  double (*f10010)[3];  /* Skew quadrupole */
  double (*f10100)[3];
  double (*f30000)[3];  /* Normal sextupole */
  double (*f12000)[3];
  double (*f10200)[3];
  double (*f01200)[3];
  double (*f01110)[3];
  double (*f00300)[3];  /* Skew Sextupole */
  double (*f00120)[3];
  double (*f20100)[3];
  double (*f20010)[3];
  double (*f11010)[3];
} S_DRIVING_TERMS;

/* Node structure for linked-list of beamline definitions: */
#define N_TSWA 3
typedef struct line_list {
    char *name;
    char *definition;
    ELEMENT_LIST *elem;     /* linked list of elements that make up this beamline */
    long n_elems, ncat_elems;
    ELEMENT_LIST *ecat;     /* linked list of concatenated elements that are equivalent to the beamline */
    long i_recirc;                /* refers to element index in elem list */
    ELEMENT_LIST *elem_recirc;    /* pointer to element in elem list */
    long icat_recirc;             /* refers to element index in ecat list */
    ELEMENT_LIST *ecat_recirc;    /* pointer to element in ecat list */
    TWISS *twiss0;                /* initial Twiss parameters */
    SIGMA_MATRIX *sigmaMatrix0;   /* initial Sigma matrix */
    ELEMENT_LIST *elem_twiss;  /* pointer to element for which twiss0 holds entering Twiss parameters.
                              Usually &elem or elem_recirc. */
    ELEMENT_LIST *elast;    /* pointer to last element &elem list */
    double tune[2];          /* x and y tunes from start of elem_twiss to end of line */
    long waists[2];          /* number of sign changes in alpha */
    double chromaticity[2];  /* dNUx/d(p/p0) and dNUy/d(p/p0) */
    double eta2[4], eta3[4]; /* second- and third-order dispersion (x=x0+eta*delta+eta2*delta^2+eta3*delta^3 */
    double chrom2[2], chrom3[2]; /* second- and third-order chromaticity (derivatives, not polynomial coefs) */
    double chromDeltaHalfRange;    /* momentum error range for tune limits */
    double tuneChromUpper[2];  /* upper limit of tune due to chromaticity and momentum spread */
    double tuneChromLower[2];  /* lower limit of tune due to chromaticity and momentum spread */
    double dbeta_dPoP[2];    /* d/d(p/p0) of betax and betay */
    double dalpha_dPoP[2];   /* d/d(p/p0) of alphax and alphay */
    double alpha[3];         /* first, second, third order momentum compaction: Cs=Cs0+alpha*delta+alpha2*delta^2+alpha3*delta^3 */
    double dnux_dA[N_TSWA][N_TSWA];    /* tune shift with amplitude [i][j] = dnux/(dAx^i dAy^j) */
    double dnuy_dA[N_TSWA][N_TSWA];    /* tune shift with amplitude */
    double nuxTswaExtrema[2];  /* min, max tunes from TSWA calculations */ 
    double nuyTswaExtrema[2];  /* min, max tunes from TSWA calculations */ 
    double acceptance[4];      /* in pi-meter-radians for x and y, plus z locations of limits (4 doubles in all) */
    double couplingFactor[3];  /* kappa, delta, r from pg 187 of HAPE */
    DRIVING_TERMS drivingTerms;
    S_DRIVING_TERMS sDrivingTerms; /* s dependent driving terms */
    RADIATION_INTEGRALS radIntegrals;
    char *acc_limit_name[2];  /* names of elements at which acceptance is limited, in x and y */
    TRAJECTORY *closed_orbit;  /* closed orbit, if previously calculated, starting at recirc element */
    VMATRIX *matrix;       /* matrix from start of elem_twiss to end of line */
    VMATRIX *Mld;          /* linear damping matrix from start of elem_twiss to end of line */
    char *part_of;         /* name of lowest-level line that this line is part of */
    ELEMENT_LINKS *links;   /* pointer to element links for this beamline */
    struct line_list *pred, *succ;
    double revolution_length;
    unsigned long flags;
/* flags to indicate status of operations for beamline
 * X_CURRENT : operation is current
 * X_DONE    : operation has been previously done, but may not be current
 */
#define BEAMLINE_CONCAT_CURRENT   0x00000001UL
#define BEAMLINE_CONCAT_DONE      0x00000002UL
#define BEAMLINE_TWISS_CURRENT    0x00000004UL
#define BEAMLINE_TWISS_DONE       0x00000008UL
#define BEAMLINE_TWISS_WANTED     0x00000010UL
#define BEAMLINE_RADINT_WANTED    0x00000020UL
#define BEAMLINE_RADINT_CURRENT   0x00000040UL
#define BEAMLINE_RADINT_DONE      0x00000080UL
#define BEAMLINE_BACKTRACKING     0x00000100UL
#define BEAMLINE_MATRICES_NEEDED  0x00000200UL
    unsigned long fiducial_flag;  /* same bits as used by VARY, but only used to keep track of status */
    } LINE_LIST;

typedef struct {
    short valuesInitialized;
    long nItems;
    LINE_LIST *beamline;
    ELEMENT_LIST **element;      /* element to be modulated */
    char **item;                 /* name of item to vary for each element, e.g., "K1" */
    long *parameterNumber;       /* parameter number of varied value */
    unsigned long *flags;        /* flag bits follow: */      
#define DIFFERENTIAL_MOD   0x01UL
#define MULTIPLICATIVE_MOD 0x02UL
#define VERBOSE_MOD        0x04UL
#define REFRESH_MATRIX_MOD 0x08UL
    double *factor;              /* factor by which to multiply the modulation */
    double *verboseThreshold;    /* fractional change for verbose output */
    double *lastVerboseValue;    /* last value for which a change was announced */
    double *unperturbedValue;    /* value without modulation */
    char **expression;           /* rpn expression for A(t) */
    long *dataIndex;             /* used for sharing of data tables */
    long *nData;                 /* number of data elements */
    double **timeData;           /* time values */
    double *timeDelay;
    double **modulationData;     /* amplitude values */
    char **record;               /* output filenames */
    long *flushRecord;           /* passes between flushing the record file */
    FILE **fpRecord;             /* output file structures */
    long *startPass, *endPass, *passDelay;
    long *convertPassToTime;
  } MODULATION_DATA;

typedef struct {
    short valuesInitialized;
    long nItems;
    ELEMENT_LIST **element;      /* element to be modulated */
    char **item;                 /* name of item to vary for each element, e.g., "K1" */
    long *parameterNumber;       /* parameter number of varied value */
    unsigned long *flags;        /* flag bits follow: */      
#define DIFFERENTIAL_RAMP   0x01UL
#define MULTIPLICATIVE_RAMP 0x02UL
#define VERBOSE_RAMP        0x04UL
#define REFRESH_MATRIX_RAMP 0x08UL
    double *unperturbedValue;    /* value without modulation */
    long *startPass, *endPass;
    double *startValue, *endValue, *exponent;
    char **record;
    FILE **fpRecord;
  } RAMP_DATA;


/* structure containing information for variation of parameters */
typedef struct {
    long ready;                  /* indicates there is valid data here */
    long at_start;               /* indicates that present state is start of variation */
    long n_indices;              /* number of indices for variation */
    long *index, *index_limit;   /* current indices, index limits */
    long indexLimitProduct;      /* product of all the index limits */
    long n_elements_to_vary;    
    long *element_index;         /* index used for each element */
    char **element;              /* names of elements being varied, e.g., "Q1" */
    char **item;                 /* names of item to vary for each element, e.g., "K1" */
    double *initial, *final;     /* rangse to vary over */
    double *step;                /* step sizes */
    double **enumerated_value;   /* list of values to take on, if enumerated list given */
    char **varied_quan_name;     /* e.g., "Q1[K1]" */
    char **varied_quan_unit;
    long *varied_type;           /* type code of varied element, e.g., T_QUAD */
    double *varied_quan_value;   /* current value */
    long *varied_param;          /* parameter number of varied value */
    long *flags;                 /* flag bits for variation */
#define VARY_GEOMETRIC 1
    long i_vary;                 /* running count of number of vary steps taken */
    long i_step;
    long n_steps;                /* number of error sets/bunches levels */
    double bunch_frequency;      /* bunch interval, if timing is varied */
    double step_frequency;       /* used for macro-scale harmonic timing offsets that are not included in bunch coordinates */
    short reset_rf_each_step;     /* whether to reset rf element phases/timing */
    short reset_scattering_seed;  /* whether to reset random numbers for scattering for each step */
    unsigned long fiducial_flag; /* flags for fiducial control */
    long n_passes;               /* number of times to go through beamline */
    short new_data_read;          /* new data has been read for variation of elements */
    short terminate_on_failure;
    LINE_LIST *cell;               /* cell to be varied along with main beamline */
    char *waitForStepSemaphore;    /* optional filename for semaphore that must be present before starting a step */
    char *stepDoneSemaphore;       /* optional filename for semaphore to create when step is done */
    double semaphoreCheckInterval; /* in seconds */
    short restartFiles;            /* if non-zero, certain files are restarted for each step */
    } VARY;
void check_VARY_structure(VARY *_control, char *caller);

/* structures containing information for random errors */

typedef struct {
  double *sourceData; /* data supplied from file */
  long nValues;       /* number of values supplied from file */
  long *sequence;     /* sequence of indices for use of sourceData */
  long iSequence;     /* position in the sequence */
  short mode;         /* valid modes are defined in error.nl */
} ERROR_SAMPLES;

typedef struct {
    long n_items;
    char **name;                 /* names of elements being varied, e.g., "Q1" */
    char **item;                 /* name of item to vary for each element, e.g., "K1" */
    char **quan_name;            /* full name of perturbed quantity, e.g., dQ1[K1] */
    char **quan_unit;
    long *quan_final_index;      /* SDDS index in 'final' output file */
    long quan_final_duplicates;  /* also used with final output */
    double *error_level;         /* e.g., sigma for gaussian errors */
    double *error_cutoff;        /* e.g., 3 sigma */
    long *error_type;
    long *elem_type;             /* type code of element, e.g., T_QUAD */
    long *param_number;          /* parameter number of varied value */
    long *sampleIndex;           /* if non-negative, index of the sequence in errorSamples array */
    long *flags;                 /* flag bits follow: */      
#define BIND_ERRORS 1
#define FRACTIONAL_ERRORS 2
#define ANTIBIND_ERRORS 4
#define BIND_ERRORS_MASK (BIND_ERRORS|ANTIBIND_ERRORS)
#define PRE_CORRECTION 8
#define POST_CORRECTION 16
#define NAME_IS_LINE 32
#define NONADDITIVE_ERRORS 64
#define FORCE_ZERO_ERRORS 128
    long *bind_number;           /* how many consecutive elements to bind */
    long *boundTo;               /* index of prior error to which this error is bound */
    double *unperturbed_value;   /* current value without errors */
    double *error_value;         /* current error value */
    double *sMin, *sMax;         /* limits geographical region of application */
    FILE *fp_log;                /* file to log error values to */
    long new_data_read;          /* new data has been read for control of tracking */
    long no_errors_first_step;   /* if nonzero, first step is perfect lattice */
    ERROR_SAMPLES *errorSamples;
    long nErrorSampleSets;
    } ERRORVAL;

/* structure containing information for modulations */


/* structures containing information for optimization */

typedef struct {
    char **element;                       /* names of element being varied, e.g., "Q1" */
    char **item;                          /* names of item to vary, e.g., "K1" */
    double *lower_limit, *upper_limit;    /* ranges to allow variation over */
    double *step;                         /* step size */
    double *orig_step;                    /* original step size */
    char **varied_quan_name;              /* e.g., "Q1[K1]" */
    char **varied_quan_unit;
    long *varied_type;                    /* type codes of varied element, e.g., T_QUAD */
    double *varied_quan_value;            /* current values */
    long *varied_param;                   /* parameter numbers of varied parameters */
    double *initial_value;                /* initial values of varied parameters */
    long *memory_number;                  /* rpn memory numbers of varied parameters */
    short *force_inside;
    long n_variables;
    } OPTIM_VARIABLES ;

typedef struct {
    char **element;                       /* names of elements being varied, e.g., "Q1" */
    char **item;                          /* names of items to vary, e.g., "K1" */
    char **equation;                      /* rpn expressions for values of the variables */
    char **pcode;                         /* Pseudo-code versions of the equations */
    char **varied_quan_name;              /* e.g., "Q1[K1]" */
    char **varied_quan_unit;
    long *varied_type;                    /* type codes of varied element, e.g., T_QUAD */
    double *varied_quan_value;            /* current values */
    long *varied_param;                   /* parameter numbers of varied parameters */
    long *memory_number;                  /* rpn memory numbers of varied parameters */
    long n_covariables;
    } OPTIM_COVARIABLES ;

typedef struct {
    char **quantity;                    /* any quantity compiled by compute_final_parameters */
    long *index;                        /* indices in array compile by compute_final_parameters */
    double *lower, *upper;              /* upper and lower limits of allowed ranges */
    long n_constraints;
    } OPTIM_CONSTRAINTS ;

#define OPTIM_MODE_MINIMUM      0
#define OPTIM_MODE_MAXIMUM      1
#define N_OPTIM_MODES           2

#define OPTIM_METHOD_SIMPLEX    0
#define OPTIM_METHOD_GRID       1
#define OPTIM_METHOD_SAMPLE     2
#define OPTIM_METHOD_POWELL     3
#define OPTIM_METHOD_RANSAMPLE  4
#define OPTIM_METHOD_RANWALK    5
#define OPTIM_METHOD_GENETIC    6
#define OPTIM_METHOD_HYBSIMPLEX 7
#define OPTIM_METHOD_SWARM      8
#define OPTIM_METHOD_1DSCANS    9
#define OPTIM_METHOD_RCDS      10
#define N_OPTIM_METHODS        11


  /* The definitions of PGA_CROSSOVER_ONEPT, PGA_CROSSOVER_TWOPT, PGA_CROSSOVER_UNIFORM can be found at pgapack.h */
#define N_CROSSOVER_TYPES       3

typedef struct {
    long mode, method;
    double tolerance, target;
    long i_pass, n_passes, n_evaluations;
    long soft_failure, UDFcreated;
    FILE *fp_log;
    long verbose;
    long balance_terms;
    char *equation;              /* rpn equation for thing to optimize */    
    char **term;                 /* terms that make up the equation, if used */
    double *termWeight, *usersTermWeight, *termValue;
    long terms;                  /* number of terms */
    char *UDFname;
    OPTIM_VARIABLES variables;
    OPTIM_CONSTRAINTS constraints;    
    OPTIM_COVARIABLES covariables;
    long update_periodic_twiss_parameters;    /* flag: user must request this */
    long new_data_read;          /* new data has been read for optimization */
    long n_restarts;
    double restart_reset_threshold, restart_worst_term_factor;
    long restart_worst_terms;
    long matrix_order, *TijkMem, *UijklMem;
    double simplexDivisor, simplexPassRangeFactor;
    double random_factor, rcdsStepFactor;
    long includeSimplex1dScans, startFromSimplexVertex1;
    /* For parallel optimization only */
    long n_iterations;            /* The maximal number of iterations allowed */
    long max_no_change;           /* The number of iterations to stop when no change in the best solution found */
    long population_size;
    long print_all_individuals;
    double hybrid_simplex_tolerance;
    long hybrid_simplex_tolerance_count, hybrid_simplex_comparison_interval;
    SDDS_TABLE popLog;
    long crossover_type;          /* For genetic optimization only */
    long nParticlesToMatch;
    double **coordinatesToMatch, particleMatchingWeight[6];
    unsigned long particleMatchingMode;
    long statistic;
    } OPTIMIZATION_DATA;

/* structure to store particle coordinates */
typedef struct {
    double **original;      /* original particle data */
    int64_t n_original;        /* number of particles read from data file, also size of original array */
    int64_t n_saved;           /* number of particles saved in original array, if this is being done */
    double p0_original;     /* initial central momentum */
    double **particle;      /* current/final coordinates */
    int64_t n_to_track;        /* initial number of particles being tracked. */
    int64_t n_lost;
    int64_t id_slots_per_bunch;     /* if non-zero, the bunch # is (particleID-1)/id_slots_per_bunch */
#if SDDS_MPI_IO
  int64_t n_to_track_total;    /* The total number of particles being tracked on all the processors */
  int64_t n_original_total;    /* The total number of particles read from data file */
  int64_t n_saved_total;       /* The total number of particles saved for reuse on all the processors */
#endif
    int64_t n_particle;        /* size of particle and accepted arrays */
    double p0;              /* current/final central momentum */
    double **accepted;      /* coordinates of accepted particles */
    int64_t n_accepted;        /* final number of particles being tracked. */
    double bunchFrequency;
    } BEAM;
void free_beamdata(BEAM *beam);

typedef struct {
  long active;
  char *filename;
  SDDS_DATASET SDDSout;
  /* input data */
  double beta, undulatorK, undulatorPeriod;
  double sliceFraction;
  long nSlices, beamsizeMode;
  /* simulation data */
  double *betaToUse, *charge, *pCentral, *rmsBunchLength, *Sdelta, *emit;
  double *betaxBeam, *alphaxBeam, *betayBeam, *alphayBeam, *enx, *eny;
  double *Cx, *Cy, *Cxp, *Cyp, *Cdelta, *Ct;
  long *betaToUseIndex, *chargeIndex, *pCentralIndex, *rmsBunchLengthIndex, *SdeltaIndex, *emitIndex;
  long *betaxBeamIndex, *alphaxBeamIndex, *betayBeamIndex, *alphayBeamIndex, *enxIndex, *enyIndex;
  long *CxIndex, *CyIndex, *CxpIndex, *CypIndex, *CdeltaIndex, *CtIndex;
  /* computed FEL output */
  double *lightWavelength, *saturationLength, *gainLength, *noisePower;
  double *saturationPower, *PierceParameter, *etaDiffraction, *etaEmittance, *etaEnergySpread;
  long *lightWavelengthIndex, *saturationLengthIndex, *gainLengthIndex, *noisePowerIndex;
  long *saturationPowerIndex, *PierceParameterIndex, *etaDiffractionIndex, *etaEmittanceIndex, 
              *etaEnergySpreadIndex;
  long *sliceFound;
} SASEFEL_OUTPUT;

typedef struct {
  long active, rows;
  SDDS_DATASET SDDSout;
  /* input data */
  char *filename;
  long nSlices, finalValuesOnly;
  double sStart, sEnd;
  /* simulation data */
  double *enx, *eny, *ecnx, *ecny, eta[4];
  double *betacx, *betacy, *alphacx, *alphacy;
  double *Cx, *Cy, *Cxp, *Cyp, *Ct, *Cdelta, *Sdelta, *duration, *charge;
  double *particles;
  long *enxIndex, *enyIndex, *ecnxIndex, *ecnyIndex;
  long *betacxIndex, *betacyIndex, *alphacxIndex, *alphacyIndex;
  long *CxIndex, *CyIndex, *CxpIndex, *CypIndex, *CtIndex, *CdeltaIndex, *SdeltaIndex;
  long *durationIndex, *chargeIndex, *particlesIndex;
  long *enxMemNum, *enyMemNum, *ecnxMemNum, *ecnyMemNum;
  long *betacxMemNum, *betacyMemNum, *alphacxMemNum, *alphacyMemNum;
  long *CxMemNum, *CyMemNum, *CxpMemNum, *CypMemNum, *CtMemNum, *CdeltaMemNum, *SdeltaMemNum;
  long *durationMemNum, *chargeMemNum, *particlesMemNum;
  long *sliceFound;
} SLICE_OUTPUT;


/* structure for hold information for per-particle tune output */
typedef struct {
  /* user-provided parameters */
  char *filename;
  int64_t startPID, endPID;
  long PIDInterval;
  short include[4]; /* x, y, delta, spin */
  long segmentLength, startPass;
  /* Conversion of particle ID to particle index. This is needed because particles get reordered when losses occur. */
  htab *indexHash;
  int64_t np;
  /* data[coord][particle][turn], e.g., data[2][10][20] is y for 11th particle on 21st turn */
  double ***data;
  int64_t *particleID, *particleIndex;
  /* Pass from which we started accumulating data */
  long pass0; 
  /* Maximum number of turns recorded and buffer size for particle. May vary due to losses. */
  long *turnIndexLimit, maxBufferSize;
  SDDS_DATASET SDDSout;
  short initialized, dataPending;
  long nPagesOutput;
} PARTICLE_TUNES;
    
/* structure to hold information for output files specified in run_setup namelist */

typedef struct {
    SDDS_TABLE SDDS_output, SDDS_accept, SDDS_centroid, SDDS_bpmCentroid, SDDS_sigma, SDDS_final, SDDS_losses, SDDS_tunes;
    long output_initialized, accept_initialized, centroid_initialized, bpmCentroid_initialized, sigma_initialized,
      final_initialized, losses_initialized, tunes_initialized;
    BEAM_SUMS *sums_vs_z;
    long n_z_points;
    SASEFEL_OUTPUT sasefel;
    SLICE_OUTPUT sliceAnalysis;
    PARTICLE_TUNES particleTunes;
} OUTPUT_FILES;

  /* structure for passing information on run conditions */

typedef struct {
    double ideal_gamma, p_central;
    short default_order, concat_order, print_statistics;
    short combine_bunch_statistics, wrap_around, tracking_updates, final_pass; 
    short always_change_p0, stopTrackingParticleLimit, load_balancing_on, random_sequence_No, checkBeamStructure;
    short showElementTiming, monitorMemoryUsage, backtrack, lossesIncludeGlobalCoordinates, spinTracking, coupledSigmaOutput;
    double lossLimit[2]; /* loss recording only between these limits */
    char *runfile, *lattice, *acceptance, *centroid, *bpmCentroid, *sigma, 
      *final, *output, *rootname, *losses, *tuneFile;
    APERTURE_DATA apertureData;
    PRESSURE_DATA pressureData; /* Used by moments_output */
    MODULATION_DATA modulationData;
    RAMP_DATA rampData;
    OUTPUT_FILES outputFiles;
    char *trackingInterruptFile;
    time_t trackingInterruptFileMtime;
    long n_passes_fiducial;      /* if >0, the number of times to go through for the fiducial particle */
#if USE_MPI
    int n_processors;
#endif
} RUN;

#define CONTEXT_BUFSIZE 1023
#ifdef _MSC_VER
#define AL __declspec(align(16))
#else
#define AL __attribute__ ((aligned(16)))
#endif

typedef AL struct {
  char *elementName;
  long iPass, elementOccurrence, step, elementType;
  ELEMENT_LIST *element;
  SLICE_OUTPUT *sliceAnalysis;
  double zStart, zEnd;
  char *rootname;
  unsigned long flags;
#if USE_MPI
  int myid;
#endif
} TRACKING_CONTEXT;

/* data arrays for awe dumps, found in dump_particlesX.c */
#define N_BEAM_QUANTITIES 9
extern char *beam_quan[N_BEAM_QUANTITIES];
extern char *beam_unit[N_BEAM_QUANTITIES];
#define N_FINAL_QUANTITIES 80
extern char *final_quan[N_FINAL_QUANTITIES];
extern char *final_unit[N_FINAL_QUANTITIES];

/* entity type codes */
#define T_ECOPY -32767
#define T_FREEVAR -32766
#define T_KNOB    -32765 /* slot refers to a knob (see knobs.h); param_number holds the knob index */
#define T_RENAME -5
#define T_RETURN -4
#define T_TITLE -3
#define T_USE   -2
#define T_NODEF -1
#define N_MADCOMS 5
#define T_LINE  0
#define T_QUAD  1
#define T_SBEN  2
#define T_RBEN  3
#define T_DRIF  4
#define T_SEXT  5
#define T_OCT   6
#define T_MULT  7
#define T_SOLE  8
#define T_HCOR  9
#define T_VCOR  10
#define T_RFCA  11
#define T_ELSE  12
#define T_HMON  13
#define T_VMON  14
#define T_MONI  15
#define T_RCOL  16
#define T_ECOL  17
#define T_MARK  18
/* Explicit matrix input from a file: */
#define T_MATR  19
/* Alpha magnet, or one half thereof: */
#define T_ALPH  20
/* RF DeFlector */
#define T_RFDF  21
/* TM-mode RF cavity using Ez(z,r=0) from cavity simulation: */
#define T_RFTMEZ0  22
/* RaMped, spatially-constant electric DeFlector--semi-analytical solution: */
#define T_RMDF  23
/* TM-mode RF cavity with spatially Constant Fields, using NAG integrator: */
#define T_TMCF  24
/* Ramped, spatially-Constant Electric deflector PLates, using NAG integrator: */
#define T_CEPL  25
#define T_WATCH  26
/* Traveling-Wave deflector PLates (TEM fields), using NAG integrator: */
#define T_TWPL 27
#define T_MALIGN 28
/* Traveling-Wave Linear Accelerator, using NAG and first space harmonic only*/
#define T_TWLA 29
/* Pepper-pot plate */
#define T_PEPPOT 30
/* Energy matching */
#define T_ENERGY 31
/* Maximum amplitudes (after elements with matrices) */
#define T_MAXAMP 32
/* Coordinate rotation */
#define T_ROTATE 33
/* Define point from which transmission is to be calculated */
#define T_TRCOUNT 34
/* Define point from which recirculation is to begin */
#define T_RECIRC 35
/* Quadrupole triangular fringe-field element */
#define T_QFRING 36
/* Scraper insertable from one side */
#define T_SCRAPER 37
/* Trajectory correction */
#define T_CENTER 38
/* kicker magnet */
#define T_KICKER 39
/* kick sextupole */
#define T_KSEXT 40
/* kick sector bend */
#define T_KSBEND 41
/* kick quadrupole */
#define T_KQUAD 42
#define T_MAGNIFY 43
#define T_SAMPLE 44
#define T_HVCOR 45
#define T_SCATTER 46
#define T_NIBEND 47
#define T_KPOLY 48
#define T_NISEPT 49
#define T_RAMPRF 50
#define T_RAMPP  51
#define T_STRAY 52
#define T_CSBEND 53
#define T_TWMTA 54
#define T_MATTER 55
#define T_RFMODE 56
#define T_TRFMODE 57
#define T_ZLONGIT 58
#define T_SREFFECTS 59
#define T_MODRF 60
#define T_BMAPXY 61
#define T_ZTRANSVERSE 62
#define T_IBSCATTER 63
#define T_FMULT 64
#define T_WAKE 65
#define T_TRWAKE 66
#define T_TUBEND 67
#define T_CHARGE 68
#define T_PFILTER 69
#define T_HISTOGRAM 70
#define T_CSRCSBEND 71
#define T_CSRDRIFT 72
#define T_RFCW  73
#define T_REMCOR 74
#define T_MAPSOLENOID 75
#define T_REFLECT 76
#define T_CLEAN 77
#define T_TWISSELEMENT 78
#define T_WIGGLER 79
#define T_SCRIPT 80
#define T_FLOORELEMENT 81
#define T_LTHINLENS 82
#define T_LMIRROR   83
#define T_EMATRIX   84
#define T_FRFMODE   85
#define T_FTRFMODE  86
#define T_TFBPICKUP 87
#define T_TFBDRIVER 88
#define T_LSCDRIFT  89
#define T_DSCATTER  90
#define T_LSRMDLTR     91
#define T_POLYNOMIALSERIES 92
#define T_RFTM110  93
#define T_CWIGGLER 94
#define T_EDRIFT 95
#define T_SCMULT 96						
#define T_ILMATRIX 97
#define T_TSCATTER  98
#define T_KQUSE 99
#define T_UKICKMAP 100
#define T_MKICKER 101
#define T_EMITTANCE 102
#define T_MHISTOGRAM 103
#define T_FTABLE 104
#define T_KOCT 105
#define T_MRADINTEGRALS 106
#define T_APPLE 107
#define T_MRFDF 108
#define T_CORGPIPE 109
#define T_LRWAKE 110
#define T_EHCOR  111
#define T_EVCOR  112
#define T_EHVCOR  113
#define T_BMAPXYZ 114
#define T_BRAT 115
#define T_BGGEXP 116
#define T_BRANCH 117
#define T_IONEFFECTS 118
#define T_SLICE_POINT 119
#define T_SPEEDBUMP 120
#define T_CCBEND 121
#define T_HKPOLY 122
#define T_BOFFAXE 123
#define T_APCONTOUR 124
#define T_TAPERAPC 125
#define T_TAPERAPE 126
#define T_TAPERAPR 127
#define T_SHRFDF  128
#define T_KICKMAP 129
#define T_BEAMBEAM 130
#define T_CPICKUP 131
#define T_CKICKER 132
#define T_LGBEND 133
#define T_CORGPLATES 134
#define T_BEDGE 135
#define T_DQCOR 136
#define T_POLAR 137
#define T_IMPEDANCE 138
#define T_CWAKE 139
#define T_CBSCAT 140
#define N_TYPES 141

extern char *entity_name[N_TYPES];
extern char *madcom_name[N_MADCOMS];
extern char *entity_text[N_TYPES];

/* number of parameters for physical elements
 * a zero indicates an unsupported element
 */
#define N_QUAD_PARAMS 33
#define N_BEND_PARAMS 27
#define N_DRIFT_PARAMS 2
#define N_SEXT_PARAMS 11
#define N_OCTU_PARAMS 8
#define N_MULT_PARAMS 15
#define N_SOLE_PARAMS 7
#define N_HCOR_PARAMS 11
#define N_VCOR_PARAMS 11
#define N_RFCA_PARAMS 19
#define N_ELSE_PARAMS 0
#define N_HMON_PARAMS 12
#define N_VMON_PARAMS 12
#define N_MONI_PARAMS 15
#define N_RCOL_PARAMS 7
#define N_ECOL_PARAMS 9
#define N_MARK_PARAMS 4
#define N_MATR_PARAMS 4
#define N_ALPH_PARAMS 13
#define N_RFDF_PARAMS 29
#define N_RFTMEZ0_PARAMS 36
#define N_RMDF_PARAMS 10
#define N_TMCF_PARAMS 18
#define N_CEPL_PARAMS 16
#define N_TWPL_PARAMS 16
#define N_WATCH_PARAMS 21
#define N_MALIGN_PARAMS 14
#define N_TWLA_PARAMS 20
#define N_PEPPOT_PARAMS 6
#define N_ENERGY_PARAMS 4
#define N_MAXAMP_PARAMS 6
#define N_ROTATE_PARAMS 4
#define N_TRCOUNT_PARAMS 1
#define N_RECIRC_PARAMS 1
#define N_QFRING_PARAMS 9
#define N_SCRAPER_PARAMS 15
#define N_CENTER_PARAMS 9
#define N_KICKER_PARAMS 14
#define N_KSEXT_PARAMS 39
#define N_KSBEND_PARAMS 27
#define N_KQUAD_PARAMS 57
#define N_MAGNIFY_PARAMS 6
#define N_SAMPLE_PARAMS 2
#define N_HVCOR_PARAMS 13
#define N_SCATTER_PARAMS 9
#define N_NIBEND_PARAMS 24
#define N_KPOLY_PARAMS 8
#define N_RAMPRF_PARAMS 9
#define N_RAMPP_PARAMS 1
#define N_NISEPT_PARAMS 9
#define N_STRAY_PARAMS 7
#define N_CSBEND_PARAMS 84+14
#define N_MATTER_PARAMS 22
#define N_RFMODE_PARAMS 57
#define N_TRFMODE_PARAMS 25
#define N_TWMTA_PARAMS 17
#define N_ZLONGIT_PARAMS 31
#define N_MODRF_PARAMS 20
#define N_SREFFECTS_PARAMS 15
#define N_ZTRANSVERSE_PARAMS 39
#define N_IMPEDANCE_PARAMS 41
#define N_CWAKE_PARAMS 32
#define N_IBSCATTER_PARAMS 13
#define N_FMULT_PARAMS 13
#define N_BMAPXY_PARAMS 7
#define N_WAKE_PARAMS 17
#define N_TRWAKE_PARAMS 25
#define N_TUBEND_PARAMS 6
#define N_CHARGE_PARAMS 3
#define N_PFILTER_PARAMS 6
#define N_HISTOGRAM_PARAMS 15
#define N_CSRCSBEND_PARAMS 73
#define N_CSRDRIFT_PARAMS 27
#define N_REMCOR_PARAMS 6
#define N_MAPSOLENOID_PARAMS 18
#define N_RFCW_PARAMS 42
#define N_REFLECT_PARAMS 1
#define N_CLEAN_PARAMS 7
#define N_TWISSELEMENT_PARAMS 22
#define N_WIGGLER_PARAMS 16
#define N_SCRIPT_PARAMS 41
#define N_FLOORELEMENT_PARAMS 6
#define N_LTHINLENS_PARAMS 8
#define N_LMIRROR_PARAMS 9
#define N_EMATRIX_PARAMS (1+6+6*6+6*21+9)
#define N_FRFMODE_PARAMS  14
#define N_FTRFMODE_PARAMS 17
#define N_TFBPICKUP_PARAMS 40
#define N_TFBDRIVER_PARAMS 52
#define N_LSCDRIFT_PARAMS  14
#define N_DSCATTER_PARAMS 14
#define N_LSRMDLTR_PARAMS 27
#define N_POLYNOMIALSERIES_PARAMS 6
#define N_RFTM110_PARAMS 16
#define N_CWIGGLER_PARAMS 38
#define N_EDRIFT_PARAMS 1
#define N_SCMULT_PARAMS 0		
#define N_ILMATRIX_PARAMS 46
#define N_TSCATTER_PARAMS 1
#define N_KQUSE_PARAMS 17
#define N_UKICKMAP_PARAMS 17
#define N_MKICKER_PARAMS 13
#define N_EMITTANCEELEMENT_PARAMS 4
#define N_MHISTOGRAM_PARAMS 12
#define N_FTABLE_PARAMS 16
#define N_KOCT_PARAMS 22
#define N_MRADITEGRALS_PARAMS 1
#define N_APPLE_PARAMS 25
#define N_MRFDF_PARAMS 27
#define N_CORGPIPE_PARAMS 15
#define N_LRWAKE_PARAMS 15
#define N_EHCOR_PARAMS 16
#define N_EVCOR_PARAMS 16
#define N_EHVCOR_PARAMS 18
#define N_BMAPXYZ_PARAMS 32
#define N_BRAT_PARAMS 37
#define N_BGGEXP_PARAMS 35
#define N_BRANCH_PARAMS 7
#define N_SLICE_POINT_PARAMS 13
#define N_IONEFFECTS_PARAMS 18
#define N_SPEEDBUMP_PARAMS 9
#define N_CCBEND_PARAMS 74
#define N_HKPOLY_PARAMS (2*49+7*7*7+8)
#define N_BOFFAXE_PARAMS (19+5)
#define N_APCONTOUR_PARAMS 15
#define N_TAPERAPC_PARAMS 6
#define N_TAPERAPE_PARAMS 12
#define N_TAPERAPR_PARAMS 9
#define N_SHRFDF_PARAMS 25
#define N_KICKMAP_PARAMS 13
#define N_BEAMBEAM_PARAMS 6
#define N_CPICKUP_PARAMS 7
#define N_CKICKER_PARAMS 17
#define N_LGBEND_PARAMS 24
#define N_CORGPLATES_PARAMS 13
#define N_BEDGE_PARAMS 7
#define N_DQCOR_PARAMS 33
#define N_POLAR_PARAMS 6
#define N_CBSCAT_PARAMS 20

/* END OF LIST FOR NUMBERS OF PARAMETERS */

#define PARAM_CHANGES_MATRIX   0x0001UL
#define PARAM_DIVISION_RELATED 0x0002UL
#define PARAM_XY_WAVEFORM      0x0004UL
#define PARAM_IS_ALIAS         0x0008UL
#define PARAM_IS_DEPRECATED    0x0010UL
#define PARAM_IS_LOCKED        0x0020UL

typedef struct {
    char *name;            /* parameter name */
    char *unit;            /* parameter unit */
    long type;              /* parameter data type */
    unsigned long flags;
    long offset;           /* offset position in structure */
    /* one of the following will have a meaningful value--I don't use a union,
     * since these cannot be initialized (except the first member)
     */
    char *string;
    double number;
    long integer;
    char *description;
    } PARAMETER;

/* maximum slope and coordinate allowed for particles in certain routines
 * (for KSBENDs, KQUADs, KSEXTs, CSBENDs, and MULT elements)
 */
#define SLOPE_LIMIT 1.0L
#define COORD_LIMIT 10.0L

#define IS_DOUBLE 1
#define IS_LONG 2
#define IS_STRING 3
#define IS_SHORT 4
#define IS_INT64 5

#define DEFAULT_FREQUENCY 2856e6
#define DEFAULT_GAP 0.01
#define DEFAULT_FINT 0.5
#define DEFAULT_N_SECTIONS 10
#define DEFAULT_ACCURACY 1e-4
#define DEFAULT_INTEG_METHOD "runge-kutta"
#if USE_MPI
#define DEFAULT_FIDUCIAL_MODE "t,ave"
#else
#define DEFAULT_FIDUCIAL_MODE "t,median"
#endif
#define DEFAULT_RAMP_TIME 1e-9
#define DEFAULT_RADIAL_OFFSET 1.0
#define DEFAULT_BETA_WAVE 1.0
#define DEFAULT_THRESHOLD 1e-12
#define DEFAULT_NIBEND_TYPE "linear"
#define DEFAULT_N_KICKS 4
#define DEFAULT_N_SLICES 4

/* bit definitions for flags word in ELEMENT_DESCRIPTION */
#define HAS_MATRIX         0x00000001UL
#define HAS_LENGTH         0x00000002UL
#define DONT_CONCAT        0x00000004UL
#define OFFSETS_CHECKED    0x00000008UL
#define IS_MAGNET          0x00000010UL
#define MATRIX_TRACKING    0x00000020UL
#define HAS_RF_MATRIX      0x00000040UL
/* Element may change the reference energy */
#define MAY_CHANGE_ENERGY  0x00000080UL
/* Matrix changes with energy */
#define MAT_CHW_ENERGY     0x00000100UL
/* Element can be automatically divided */
#define DIVIDE_OK          0x00000200UL
/* set this flag to prevent dictionary output of experimental elements or elements that
 * are not intended to be inserted by users */
#define NO_DICT_OUTPUT     0x00000400UL
/* indicates element that can only be done by a single processor at this time */
#define UNIPROCESSOR       0x00000800UL
/* indicates that element is a uni-processor diagnostic (uniprocessor but requires no scatter) */
#define UNIDIAGNOSTIC     (0x00001000UL|UNIPROCESSOR)
/* indicates that element should be processed even with 0 particles */
#define RUN_ZERO_PARTICLES 0x00002000UL
/* indicates that element will be done on all of the processors */
#define MPALGORITHM (0x00004000UL|RUN_ZERO_PARTICLES)
#define GPU_SUPPORT  0x00008000UL
#define NO_APERTURE  0x00010000UL
#define BACKTRACK    0x00020000UL
  /* Indicates that a matrix is used, but also something else */
#define HYBRID_TRACKING  0x00040000UL
#define SPIN_TRACKING    0x00080000UL
  /* Element's tracking routine contains MPI collective calls
   * (MPI_Allreduce / MPI_Bcast over MPI_COMM_WORLD or workers) that
   * coordinate state across ranks within a single tracking pass --
   * e.g. wake-field histograms, beam-driven cavity modes, IBS, ion
   * effects, transverse feedback.  These elements are incompatible
   * with rank-parallel optimization methods (randomsample, randomwalk,
   * swarm, hybsimplex, genetic) because each rank tracks its own beam
   * independently and the collectives deadlock as soon as ranks
   * diverge.  Detected at parallel_optimization_setup time. */
#define COLLECTIVE_EFFECTS 0x00100000UL
  
typedef struct {
    long n_params;
    unsigned long flags;
    long structure_size;      /* in bytes */
    PARAMETER *parameter;
    long user_structure_size; /* in bytes, just the part of the structure that the user sets */
    } ELEMENT_DESCRIPTION;

extern ELEMENT_DESCRIPTION entity_description[N_TYPES];

#define QFRINGE_SIMPLE 0
#define QFRINGE_INSET  1
#define QFRINGE_INTEGRALS 2

/* names and storage structure for quadrupole physical parameters */
extern PARAMETER quad_param[N_QUAD_PARAMS];

typedef struct {
    double length, k1, tilt, pitch, yaw;
    double dx, dy, dz, fse, xkick, ykick;
    double xKickCalibration, yKickCalibration;
    short xSteering, ySteering, order;
    short edge1_effects, edge2_effects;
    char *fringeType;
    double ffringe, lEffective;
    double fringeIntP[5], fringeIntM[5];
    short radial, malignMethod;
    } QUAD;

/* names and storage structure for bending magnet physical parameters */
extern PARAMETER bend_param[N_BEND_PARAMS];

typedef struct {
    double length, angle, k1, e[2], tilt;
    double k2, h[2], hgap, fint;
    double dx, dy, dz;
    double fse, fseDipole, fseQuadrupole;     /* Fractional Strength Error (combined, dipole, quadrupole) */
    double etilt;   /* error tilt angle */
    short etiltSign, edge_effects[2];
    short order, edge_order, TRANSPORT;
    short use_bn;
    double b1, b2;
    /* for internal use only: */
    unsigned short edgeFlags;
    double k1_internal, k2_internal;
    short e1Index, e2Index;
    } BEND;

typedef struct {
  double beta, rho, hgap, fint, hPoleFace;
  short order, exitEdge;
} BEDGE;

/* names and storage structure for drift length physical parameters */
extern PARAMETER drift_param[N_DRIFT_PARAMS];

typedef struct {
    double length;
    short order;
    } DRIFT;

/* names and storage structure for exact drift physical parameters */
extern PARAMETER edrift_param[N_EDRIFT_PARAMS];

typedef struct {
    double length;
    } EDRIFT;

/* names and storage structure for sextupole physical parameters */
extern PARAMETER sext_param[N_SEXT_PARAMS];

typedef struct {
    double length, k2, k1, j1, tilt;
    double dx, dy, dz, fse;
    double ffringe;
    short order;
    } SEXT;

/* names and storage structure for octupole physical parameters */
extern PARAMETER octu_param[N_OCTU_PARAMS];

typedef struct {
    double length, k3, tilt;
    double dx, dy, dz, fse;
    short order;
    } OCTU;

/* names and storage structure for solenoid */
extern PARAMETER sole_param[N_SOLE_PARAMS] ;
   
typedef struct {
    double length, ks, B;
    double dx, dy, dz;
    short order;
    } SOLE;

/* names and storage structure for arbitrary multipole */
extern PARAMETER mult_param[N_MULT_PARAMS];
   
typedef struct {
    double length, KnL, tilt, bore, BTipL;
    double dx, dy, dz, factor;
    short order, synch_rad, expandHamiltonian, matrixOrder;
    long nSlices;
    } MULT;

/* names and storage structure for arbitary multipole from an SDDS File */
extern PARAMETER fmult_param[N_FMULT_PARAMS];

typedef struct {
  double length, tilt, dx, dy, dz, fse, factor;
  long n_kicks, nSlices;
  short synch_rad;
  char *filename;
  short sqrtOrder, untiltedMatrix;
  /* For internal use: */
  MULTIPOLE_DATA multData;
} FMULT;

/* names and storage structure for horizontal corrector physical parameters */
extern PARAMETER hcor_param[N_HCOR_PARAMS] ;
   
typedef struct {
    double length, kick, tilt, b2, calibration;
    short edge_effects, order, steering;
    short synchRad, isr;
    double lEffRad;
    } HCOR;

/* names and storage structure for exact corrector physical parameters */
extern PARAMETER ehcor_param[N_EHCOR_PARAMS] ;
extern PARAMETER evcor_param[N_EVCOR_PARAMS] ;
extern PARAMETER ehvcor_param[N_EHVCOR_PARAMS] ;
   
typedef struct {
    double length, kick, tilt, dx, dy, dz, calibration;
    double lEffRad;
    short steering, synchRad, isr, matrixOrder;
    char *steeringMultipoles, *randomMultipoles;
    double randomMultipoleFactor, steeringMultipoleFactor;
    /* for internal use */
    MULTIPOLE_DATA steeringMultipoleData;
    MULTIPOLE_DATA randomMultipoleData;
    short multipolesRandomized;
    MULTIPOLE_DATA totalMultipoleData;  /* generated when randomization takes place */
    } EHCOR;

typedef struct {
    double length, kick, tilt, dx, dy, dz, calibration;
    double lEffRad;
    short steering, synchRad, isr, matrixOrder;
    char *steeringMultipoles, *randomMultipoles;
    double randomMultipoleFactor, steeringMultipoleFactor;
    /* for internal use */
    MULTIPOLE_DATA steeringMultipoleData;
    MULTIPOLE_DATA randomMultipoleData;
    short multipolesRandomized;
    MULTIPOLE_DATA totalMultipoleData;  /* generated when randomization takes place */
    } EVCOR;

typedef struct {
    double length, xkick, ykick, tilt, dx, dy, dz, xcalibration, ycalibration;
    double lEffRad;
    short steering, synchRad, isr, matrixOrder;
    char *steeringMultipoles, *randomMultipoles;
    double randomMultipoleFactor, steeringMultipoleFactor;
    /* for internal use */
    MULTIPOLE_DATA steeringMultipoleData;
    MULTIPOLE_DATA randomMultipoleData;
    short multipolesRandomized;
    MULTIPOLE_DATA totalMultipoleData;  /* generated when randomization takes place */
  } EHVCOR;

/* names and storage structure for vertical corrector physical parameters */
extern PARAMETER vcor_param[N_VCOR_PARAMS] ;

typedef struct {
    double length, kick, tilt, b2, calibration;
    short edge_effects, order, steering;
    short synchRad, isr;
    double lEffRad;
    } VCOR;

/* names and storage structure for RF cavity physical parameters */
extern PARAMETER rfca_param[N_RFCA_PARAMS] ;
   
typedef struct {
    double length, volt, phase, freq, Q;
    long phase_reference;
    short change_p0, change_t;
    char *fiducial;
    short end1Focus, end2Focus, standingWave;
    char *bodyFocusModel;
    long nKicks;
    double dx, dy;
    double tReference;
    short linearize, lockPhase;
    /* for internal use only: */
    short fiducial_seen, backtrack;
    double phase_fiducial, t_fiducial;
    } RFCA;

/* names and storage structure for modulated RF cavity physical parameters */
extern PARAMETER modrf_param[N_MODRF_PARAMS] ;
   
typedef struct {
    double length, volt, phase, freq, Q;
    long phase_reference;
    double amMag, amPhase, amFreq, amOffset, amDecay;
    double pmMag, pmPhase, pmFreq, pmOffset, pmDecay;
    double startTime, endTime;
    char *fiducial, *record;
    /* for internal use only: */
    long fiducial_seen;
    double phase_fiducial; /* -omega0*t0 */
    SDDS_DATASET *SDDSrec;
    } MODRF;

/* names and storage structure for beam-position-monitor physical parameters */
/* NB: the length, dx, dy, and weight setpoint elements MUST
 * be in the same position and order in all these structures !
 */
extern PARAMETER hmon_param[N_HMON_PARAMS] ;
extern PARAMETER vmon_param[N_VMON_PARAMS] ;
extern PARAMETER moni_param[N_MONI_PARAMS] ;

#define IS_MONITOR(type) ((type)==T_MONI || (type)==T_HMON || (type)==T_VMON)

typedef struct {
    double length, dx, dy, dz, weight, tilt, calibration, setpoint;
    short order;
    char *readout;   /* rpn equation for x readout as function of x and y */
    short coFitpoint;
    short storeTurnByTurn;
    unsigned short initialized; /* 0x01: CO, 0x02: TBT */
    long coMemoryNumber[2], tbtMemoryNumber[2];
    } HMON;
typedef struct {
    double length, dx, dy, dz, weight, tilt, calibration, setpoint;
    short order;
    char *readout;   /* rpn equation for y readout as function of x and y */
    short coFitpoint;
    short storeTurnByTurn;
    unsigned short initialized; /* 0x01: CO, 0x02: TBT */
    long coMemoryNumber[2], tbtMemoryNumber[2];
    } VMON;
typedef struct {
    double length, dx, dy, dz, weight, tilt, xcalibration, ycalibration, xsetpoint, ysetpoint;
    short order;
    char *x_readout, *y_readout; /* rpn equations for x and y readouts as function of actual x and y */
    short coFitpoint;
    short storeTurnByTurn;
    unsigned short initialized; /* 0x01: CO, 0x02: TBT */
    long coMemoryNumber[4], tbtMemoryNumber[3];
    } MONI;

/* names and storage structure for rectangular collimator physical parameters */
extern PARAMETER rcol_param[N_RCOL_PARAMS] ;

/* used by RCOL, ECOL, and MAXAMP */
#define OPEN_PLUS_X 1
#define OPEN_PLUS_Y 2
#define OPEN_MINUS_X 3
#define OPEN_MINUS_Y 4

typedef struct {
    double length, x_max, y_max, dx, dy;
    char *openSide;
    short invert;
    } RCOL;

/* names and storage structure for elliptical collimator physical parameters */
extern PARAMETER ecol_param[N_ECOL_PARAMS] ;
   
typedef struct {
    double length, x_max, y_max, dx, dy;
    char *openSide;
    short exponent, yExponent, invert;
    } ECOL;

/* storage structure for beam cleaner */
extern PARAMETER clean_param[N_CLEAN_PARAMS];

typedef struct {
  char *mode;
  double xLimit, xpLimit, yLimit, ypLimit, tLimit, deltaLimit;
} CLEAN;

/* storage structure for twiss element (sets twiss parameters) */
typedef struct {
  TWISS twiss;
  short fromBeam, from0Values, computeOnce, applyOnce, verbose;
  short disable;
  TWISS twiss0;
  /* internal variables */
  long transformComputed;
} TWISSELEMENT;

/* storage structure for emittance element (sets emittances) */
typedef struct {
  double emit[2];
  double emitn[2];
} EMITTANCEELEMENT;

/* storage structure for floor element (sets floor coordinates) */
typedef struct {
  double position[3]; /* X, Y, Z */
  double angle[3];    /* theta, phi, psi */
} FLOORELEMENT;

/* storage structure for radiation integral multiplier */
typedef struct {
  double factor;
} MRADINTEGRALS;

/* storage structure for marker */

typedef struct {
  double dx, dy;   /* Useful for recording position of girder ends, for example */
  short fitpoint;
  short coupled_fitpoint; /* also supply coupled (eigen-)emittances and coupled lattice functions of the tracked beam */
  /* values for internal use: */
  unsigned long init_flags; /* 1:twiss_mem initialized,
                               2:centroid_mem, sigma_mem, emit_mem initialized,
                               4:floor_mem initialized
                               8:matrix_mem initialized
                               16:ctwiss_mem initialized
                               32:moments_mem initialized
                               64:coupledBeam_mem initialized */
  long *twiss_mem;       /* betax, alphax, NUx, etax, etaxp, betay, ... */
  long *centroid_mem;    /* (x, xp, y, yp, s, dp, Pcen, n) from tracking */
  long *sigma_mem;       /* (x, xp, y, yp, s, dp) from tracking */
  long *min_mem;         /* (x, xp, y, yp, s, dp) from tracking */
  long *max_mem;         /* (x, xp, y, yp, s, dp) from tracking */
  long *sij_mem;         /* <xi*xj> for 6>=j>i>=1 from tracking */
  long *emit_mem;        /* (x, y, z, cx, cy) from tracking */
  long *betaBeam_mem;    /* (x, y) from tracking */
  long *alphaBeam_mem;    /* (x, y) from tracking */
  long *floor_mem;       /* X, Z, theta */
  long *matrix_mem;
  long *co_mem;          /* closed orbit */
  long *ctwiss_mem;      /* coupled twiss parameters */
  long *moments_mem;     /* beam moments from moments propagation (not tracking) */
  long *coupledBeam_mem; /* coupled (eigen-)emittances and coupled lattice functions from tracking */
} MARK;

/* storage structure for alpha magnet */
extern PARAMETER alph_param[N_ALPH_PARAMS] ;

/* x_max(m) = ALPHA_CONST*sqrt(beta*gamma/grad_B(G/cm)) */
#define ALPHA_CONST 0.750498604674380
#define ALPHA_ANGLE (PI/180.0*40.709910707900)

typedef struct {
    double xmax;        /* 75.05*sqrt(beta*gamma/gradient) in meters */
    double xs1, xs2;    /* for momentum filtration */
    double dp1, dp2;    /* for momentum filtration */
    double xPuck, widthPuck;  /* for momentum filtration */
    double dx, dy, dz, tilt;
    short part;
    short order;
    double gradient;   
    } ALPH;

/* names and storage structure for RF deflector cavity
 */
extern PARAMETER rfdf_param[N_RFDF_PARAMS] ;
   
typedef struct {
  double length, phase, tilt, frequency, voltage, fse, b2;
  double time_offset;             /* equivalent to phase */
  long n_kicks, phase_reference;
  short standingWave;
  char *voltageWaveform;
  short voltageIsPeriodic, alignWaveforms;
  double voltageNoise, phaseNoise;
  double groupVoltageNoise, groupPhaseNoise;
  long voltageNoiseGroup, phaseNoiseGroup;
  long startPass, endPass;
  int64_t startPID, endPID;
  short driftMatrix;
  double dx, dy, dz;
  short magneticDeflection;
  /* for internal use only */
  double t_first_particle;        
  short initialized, fiducial_seen;
  double Ts;                          /* accumulated time-of-flight of central particle */
  double *t_Vf, *Vfactor, VPeriod;    /* (time, V/volt) pairs */
  double V_tFinal, V_tInitial;
  long n_Vpts;
} RFDF;

/* names and storage structure for multipole RF deflector cavity
 */
extern PARAMETER mrfdf_param[N_MRFDF_PARAMS] ;
   
typedef struct {
  double factor, tilt;
  double a[5], b[5], frequency[5], phase[5];
  long phase_reference;
  long startPass, endPass;
  int64_t startPID, endPID;
  /* for internal use only */
  double t_first_particle;        
  long initialized, fiducial_seen;
} MRFDF;


/* names and storage structure for RF deflector cavity done with
 * correct TM110 mode fields
 */
extern PARAMETER rftm110_param[N_RFTM110_PARAMS] ;
   
typedef struct {
  double phase, tilt, frequency, voltage;
  long phase_reference;
  char *voltageWaveform;
  short voltageIsPeriodic, alignWaveforms;
  double voltageNoise, phaseNoise;
  double groupVoltageNoise, groupPhaseNoise;
  long voltageNoiseGroup, phaseNoiseGroup;
  long startPass, endPass;
  /* for internal use only */
  double t_first_particle;        
  short   initialized, fiducial_seen;
  double Ts;                          /* accumulated time-of-flight of central particle */
  double *t_Vf, *Vfactor, VPeriod;    /* (time, V/volt) pairs */
  double V_tFinal, V_tInitial;
  long n_Vpts;
} RFTM110;

/* TM-mode RF-cavity using Ez(z,r=0)
 */
extern PARAMETER rftmez0_param[N_RFTMEZ0_PARAMS] ;

typedef struct {
    double length, frequency, phase, Ez_peak, time_offset;
    long phase_reference;
    double dx, dy, dzMA, eTilt, ePitch, eYaw;
    long n_steps;
    short radial_order, change_p0;
    char *inputFile, *zColumn, *EzColumn;
    char *solenoidFile, *solenoid_zColumn, *solenoid_rColumn, *solenoidBzColumn, *solenoidBrColumn;
    double solenoidFactor, dxSol, dySol, dzSolMA, eTiltSol, eYawSol, ePitchSol;
    double BxStray, ByStray, accuracy;
    char *method, *fiducial, *fieldTestFile;
    /* variables for internal use only: */
    long initialized;
    double *fiducial_part;
    double phase0;
    double k;  /* omega*c */
    long nz;
    double dz, dZ;  /* actual and scaled point spacing (dZ=k*dz) */
    double z0;      /* used to align solenoid field */
    double *Ez, *dEzdZ;  /* scaled field Ez*e/(mc) and derivative wrt Z */
    /* for solenoid: */
    long nzSol, nrSol;
    double dRSol, dZSol, Z0Sol;  /* scaled grid spacing and starting point */
    double **BrSol, **BzSol;
    /* for use in case where user gives only on-axis data */
    double *dBzdZSol;     /* derivative of scaled Bz wrt Z  */
    /* Stray field */
    double BxStrayScaled, ByStrayScaled;
    } RFTMEZ0;

/* names and storage structure for ramped deflector plates using 
 * semi-analytical method.
 */
extern PARAMETER rmdf_param[N_RMDF_PARAMS] ;
   
typedef struct {
    double length, tilt, ramp_time, voltage, gap;
    double time_offset;             
    long n_sections, phase_reference;
    double dx, dy;
    double t_first_particle;        /* not to be set by user! */
    long   initialized;             /* ditto */
    } RMDF;

/* Constant-field TM deflector cavity.  Er, Ez, and Bphi are all constant over
 * the length of the cavity.  Uses NAG integrator.
 */
extern PARAMETER tmcf_param[N_TMCF_PARAMS];

typedef struct {
    double length, frequency, phase, time_offset;
    double radial_offset, tilt;
    double Er, Bphi, Ez;
    double accuracy;
    double x_max, y_max;
    double dx, dy;
    long phase_reference, n_steps;
    char *method, *fiducial;
    /* variables for internal use only: */
    double *fiducial_part;            
    double phase0;        /* phase at which fiducial particle reaches center */
    double k;             /* omega/c */
    } TMCF_MODE;

/* Constant electric field deflector plates using NAG integrator.
 */
extern PARAMETER cepl_param[N_CEPL_PARAMS];

typedef struct {
    /* variables set by the user (assigned values in compute_matrices): */
    double length, ramp_time, time_offset;
    double voltage, gap, static_voltage, tilt;
    double accuracy;
    double x_max, y_max, dx, dy;
    long phase_reference, n_steps;
    char *method, *fiducial;
    /* variables for internal use only: */
    double *fiducial_part;
    double tau0;        /* t/ramp_time at which fiducial particle reaches center */
    double E_scaled;    /* e.voltage.ramp_time/(gap.m.c) */
    double E_static;    /* e.static_voltage.ramp_time/(gamp.m.c) */
    double k;           /* 1/(c.ramp_time) */
    double sin_tilt, cos_tilt;
    } CE_PLATES;

/* cylindrically-symmetric solenoid from a map of (Bz, Br) vs (z, r)
 */
extern PARAMETER mapSolenoid_param[N_MAPSOLENOID_PARAMS] ;

typedef struct {
    double length, dx, dy;
    double eTilt, eYaw, ePitch; /* misalignment angles */
    long n_steps;
    char *inputFile, *rColumn, *zColumn, *BrColumn, *BzColumn;
    double factor, BxUniform, ByUniform, lUniform, accuracy;
    char *method;
    /* variables for internal use only: */
    long initialized;
    long nz, nr; 
    double dz, dr;
    double **Bz, **Br, BxUniformScaled, ByUniformScaled;
    double zMap0, zMap1;
  } MAP_SOLENOID;

/* storage structure for watch points  */
extern PARAMETER watch_param[N_WATCH_PARAMS];

#define WATCH_COORDINATES 0
#define WATCH_PARAMETERS 1
#define WATCH_CENTROIDS 2
#define WATCH_FFT 3
#define N_WATCH_MODES 4
extern char *watch_mode[N_WATCH_MODES];

#define FFT_HANNING 0
#define FFT_PARZEN 1
#define FFT_WELCH 2
#define FFT_UNIFORM 3
#define N_FFT_WINDOWS 4
extern char *fft_window_name[N_FFT_WINDOWS];

typedef struct {
    double fraction;
    int64_t startPID, endPID;
    long interval, start_pass, end_pass;
    char *filename, *label, *mode;
    short xData, yData, longitData, excludeSlopes;
    long flushInterval, sparseInterval;
    short disable, useDisconnect;
    long indexOffset;
    double referenceFrequency;
    short autoReference, bunchSeries; 
    /* internal variables for SDDS output */
    short initialized;
    long count, mode_code, window_code;
    long xIndex[2], yIndex[2], longitIndex[3], spinIndex[3], IDIndex;
    SDDS_TABLE *SDDS_table;
    double t0Last, t0LastError;
    long passLast, flushSample;
    } WATCH;

/* histogram element */

extern PARAMETER histogram_param[N_HISTOGRAM_PARAMS];

typedef struct {
    char *filename;
    long interval, startPass, bins;
    short fixedBinSize, xData, yData, longitData;
    double binSizeFactor;
    short normalize, disable, sparse;
    int64_t startPID, endPID;
    short bunchSeries;
    /* internal variables for SDDS output */
    short initialized;
    long count;
    long columnIndex[7][2];  /* x, xp, y, yp, t, p, dt */
    double binSize[7];
    SDDS_TABLE *SDDS_table;
    } HISTOGRAM;

extern PARAMETER mhistogram_param[N_MHISTOGRAM_PARAMS];

typedef struct {
    char *file1d, *file2dH, *file2dV, *file2dL, *file4d, *file6d;
    char *inputBinFile;
    long interval, startPass;
    short normalize, disable, lumped;
    /* internal variables for SDDS output */
    int32_t *bins1d, *bins2d, *bins4d, *bins6d;
    book1m *x1d;
    ntuple *x2d, *y2d, *z2d;
    ntuple *Tr4d, *full6d;
    short initialized;
    long count;
    } MHISTOGRAM;

  /* slice analysis element */

typedef struct {
  long nSlices;
  int64_t startPID, endPID;
  long interval, start_pass, end_pass;
  char *filename, *label;
  long indexOffset;
  double referenceFrequency;
  short disable, useDisconnect, bunchSeries;
  /* internal variables for SDDS output */
  short initialized;
  SDDS_TABLE *SDDS_table;
  double t0Last, t0LastError;
  long passLast;
} SLICE_POINT;

/* Traveling wave (TEM) deflector plates using NAG integrator.
 */
extern PARAMETER twpl_param[N_TWPL_PARAMS];

typedef struct {
    /* variables set by the user (assigned values in compute_matrices): */
    double length, ramp_time, time_offset;
    double voltage, gap, static_voltage, tilt;
    double accuracy;
    double x_max, y_max;
    double dx, dy;
    long phase_reference, n_steps;
    char *method, *fiducial;
    /* variables for internal use only: */
    double *fiducial_part;
    double tau0;        /* t/ramp_time at which fiducial particle reaches center */
    double E_scaled;    /* e.voltage.ramp_time/(gap.m.c) */
    double E_static;    /* e.static_voltage.ramp_time/(gamp.m.c) */
    double k;           /* 1/(c.ramp_time) */
    double sin_tilt, cos_tilt;
    } TW_PLATES;

/* names and storage structure for misalignment physical parameters */
extern PARAMETER malign_param[N_MALIGN_PARAMS] ;

typedef struct {
    double dxp, dyp, dx, dy, dz, dt, dp, de;
    long on_pass;
    int64_t startPID, endPID;
    short forceModifyMatrix, floor, excludeOrbit; 
    } MALIGN;

/* Traveling-Wave Linear Accelerator, using NAG and first space harmonic 
 */
extern PARAMETER twla_param[N_TWLA_PARAMS];

typedef struct {
    /* variables set by the user (assigned values in compute_matrices): */
    double length, frequency, phase, time_offset;
    double Ez, B_solenoid, accuracy;
    double x_max, y_max;
    double dx, dy, beta_wave, alpha;
    long phase_reference, n_steps, focussing;
    char *method, *fiducial; 
    long change_p0;
    double sum_bn2;
  
    /* variables for internal use only: */
    double *fiducial_part;            
    double EzS, BsolS;
    double ErS, BphiS;    /* calculated from Ez */
    double FrP;           /* Ponderomotive force coeffcience, from sum_bn2, see Hartman et al, PRE 47 (3), 1993. */
    double phase0;        /* phase at which fiducial particle reaches center of
                             first cell */
    double kz;            /* omega/(c*beta_wave) */
    double alphaS;        /* alpha/k */
    } TW_LINAC;

/* storage structure for pepper-pot plates */
extern PARAMETER peppot_param[N_PEPPOT_PARAMS];

typedef struct {
    double length, radii, transmission, tilt;
    double theta_rms;
    long n_holes;
    /* internal variables */
    double *x, *y;
    } PEPPOT;

/* storage structure for energy matching */
extern PARAMETER energy_param[N_ENERGY_PARAMS];

typedef struct {
    double central_energy;
    double central_momentum;
    long match_beamline;
    long match_particles;
    } ENERGY;

/* storage structure for amplitude limits */
extern PARAMETER maxamp_param[N_MAXAMP_PARAMS];

typedef struct {
    double x_max, y_max;
    long elliptical, exponent, yExponent;
    char *openSide;
    } MAXAMP;

/* storage structure for beam rotation */
extern PARAMETER rotate_param[N_ROTATE_PARAMS];

typedef struct {
    double tilt;
    long on_pass;
    short excludeFloor, excludeOptics;
    } ROTATE;

/* storage structure for transmission count */
extern PARAMETER trcount_param[N_TRCOUNT_PARAMS];

typedef struct {
    long dummy;
    } TRCOUNT;

/* storage structure for reflection */
extern PARAMETER reflect_param[N_REFLECT_PARAMS];

typedef struct {
    long dummy;
    } REFLECT;

/* storage structure for recirculation point */
extern PARAMETER recirc_param[N_RECIRC_PARAMS];

typedef struct {
    long i_recirc_element;
    } RECIRC;

/* storage structure for conditional branch instruction */
extern PARAMETER branch_param[N_BRANCH_PARAMS];

typedef struct {
  long counter, interval, offset, verbosity, defaultToElse;
  char *branchTo, *elseTo;
  /* internal variables */
  ELEMENT_LIST *beptr1, *beptr2;
  double z;
  long privateCounter;
} BRANCH;

/* names and storage structure for quadrupole fringe field parameters */
extern PARAMETER qfring_param[N_QFRING_PARAMS];

typedef struct {
    double length, k1, tilt;
    double dx, dy, dz, fse;
    long direction, order;
    } QFRING;

/* names and storage structure for beam-scraper parameters */
extern PARAMETER scraper_param[N_SCRAPER_PARAMS];

typedef struct {
  double length;
  double Xo;
  long energyDecay, energyStraggle, nuclearBremsstrahlung, electronRecoil, Z;
  double A, rho, pLimit;
  double position;
  double dx, dy;
  char *insert_from;          /* one of +x, -x, +y, -y --- replaces direction */
  long oldDirection;
  /* internal only */
  unsigned long direction;
#define DIRECTION_PLUS_X 0x01
#define DIRECTION_MINUS_X 0x02
#define DIRECTION_X (DIRECTION_PLUS_X+DIRECTION_MINUS_X)
#define DIRECTION_PLUS_Y 0x04
#define DIRECTION_MINUS_Y 0x08
#define DIRECTION_Y (DIRECTION_PLUS_Y+DIRECTION_MINUS_Y)
} SCRAPER;

/* names and storage structure for beam-centering parameters */
extern PARAMETER center_param[N_CENTER_PARAMS];

typedef struct {
    long doCoord[7], onceOnly, onPass;
    /* for internal use only */
    long deltaSet[7];
    double delta[7];
    } CENTER;

/* names and storage structure for beam correlation remover parameters */
extern PARAMETER remcor_param[N_REMCOR_PARAMS];

typedef struct {
    short x, xp, y, yp, with, onceOnly;
    /* for internal use only */
    long ratioSet[4];
    double ratio[4];
    } REMCOR;

/* names and storage structure for time-dependent kicker */
extern PARAMETER kicker_param[N_KICKER_PARAMS];

typedef struct {
    double length, angle, tilt, dx, dy, dz, b2, time_offset;
    long periodic, phase_reference, fire_on_pass, n_kicks;
    char *waveform, *deflectionMap;
    /* for internal use only: */
    long initialized;
    double *t_wf, *amp_wf;         /* amplitude vs time for waveform */
    double tmin, tmax;
    long n_wf;                     /* number of points in waveform */
    long fiducial_seen;
    double t_fiducial;
    FILE *fpdebug;
    /* for use with optional deflection map */
    long points, nx, ny;
    double *xpFactor, *ypFactor;
    double xmin, xmax, dxg;
    double ymin, ymax, dyg;
    } KICKER;

/* names and storage structure for time-dependent multipole kicker */
extern PARAMETER mkicker_param[N_MKICKER_PARAMS];

typedef struct {
    double length, strength, tilt, dx, dy, dz, time_offset;
    long order, periodic, phase_reference, fire_on_pass, n_kicks;
    char *waveform;
    /* for internal use only: */
    double *t_wf, *amp_wf;         /* amplitude vs time for waveform */
    double tmin, tmax;
    long n_wf;                     /* number of points in waveform */
    long fiducial_seen;
    double t_fiducial;
    FILE *fpdebug;
    } MKICKER;

/* names and storage structure for kick sextupole physical parameters */
extern PARAMETER ksext_param[N_KSEXT_PARAMS];

typedef struct {
    double length, k2, k1, j1, tilt, pitch, yaw, bore, B;
    long n_kicks, nSlices;
    double dx, dy, dz, fse, xkick, ykick;
    double xKickCalibration, yKickCalibration;
    short xSteering, ySteering, synch_rad;
    char *systematic_multipoles, *edge_multipoles, *random_multipoles, *steering_multipoles;
    double systematicMultipoleFactor, randomMultipoleFactor, steeringMultipoleFactor;
    short minMultipoleOrder[2], maxMultipoleOrder[2]; /* normal, skew */
    short integration_order, sqrtOrder, isr, isr1Particle, expandHamiltonian, malignMethod;
    /* for internal use */
    short multipolesInitialized, totalMultipolesComputed;
    MULTIPOLE_DATA systematicMultipoleData; 
    MULTIPOLE_DATA edgeMultipoleData; 
    MULTIPOLE_DATA randomMultipoleData;      
    MULTIPOLE_DATA totalMultipoleData;  /* generated when randomization takes place */
    MULTIPOLE_DATA steeringMultipoleData;
    } KSEXT;

/* names and storage structure for kick octupole physical parameters */
extern PARAMETER koct_param[N_KOCT_PARAMS];

typedef struct {
    double length, k3, tilt, pitch, yaw, bore, B;
    double dx, dy, dz, fse;
    long n_kicks, nSlices;
    char *systematic_multipoles, *random_multipoles;
    short integration_order, sqrtOrder, synch_rad, isr, isr1Particle, expandHamiltonian, malignMethod;
    /* for internal use */
    short multipolesInitialized, totalMultipolesComputed;
    MULTIPOLE_DATA systematicMultipoleData; 
    MULTIPOLE_DATA randomMultipoleData;      
    MULTIPOLE_DATA totalMultipoleData;  /* generated when randomization takes place */
    } KOCT;

/* names and storage structure for symplectic bending magnet physical parameters */
extern PARAMETER ksbend_param[N_KSBEND_PARAMS];

typedef struct {
    double length, angle, k1, k2, k3, k4, e[2], tilt;
    double h[2], hgap, fint;
    double dx, dy, dz;
    double fse;     /* Fractional Strength Error */
    double etilt;   /* error tilt angle */
    long n_kicks, nonlinear, synch_rad;
    long edge_effects[2], edge_order, paraxial, TRANSPORT;
    char *method; 
    /* for internal use only */
    long flags;
    long e1Index, e2Index;
    } KSBEND;

/* names and storage structure for kick quadrupole physical parameters */
extern PARAMETER kquad_param[N_KQUAD_PARAMS];

typedef struct {
    double length, k1, tilt, pitch, yaw, bore, B;
    double dx, dy, dz, fse;
    long n_kicks, nSlices;
    double xkick, ykick;
    double xKickCalibration, yKickCalibration;
    short xSteering, ySteering, synch_rad;
    char *systematic_multipoles, *edge_multipoles, *random_multipoles, *steering_multipoles;
    double systematicMultipoleFactor, randomMultipoleFactor, steeringMultipoleFactor;
    short minMultipoleOrder[2], maxMultipoleOrder[2]; /* normal, skew */
    short integration_order, sqrtOrder, isr, isr1Particle, synchRadInOrdinaryMatrix;
    short edge1_effects, edge2_effects;
    double lEffective;
    double fringeIntP[5], fringeIntM[5];
    short edge1Linear, edge2Linear;
    double edge1NonlinearFactor, edge2NonlinearFactor;
    short radial, expandHamiltonian, trackingBasedMatrix, malignMethod;
    /* for internal use */
    short multipolesInitialized, totalMultipolesComputed;
    MULTIPOLE_DATA systematicMultipoleData; 
    MULTIPOLE_DATA edgeMultipoleData; 
    MULTIPOLE_DATA randomMultipoleData;
    MULTIPOLE_DATA totalMultipoleData;  /* generated when randomization takes place */
    MULTIPOLE_DATA steeringMultipoleData;
    } KQUAD;

/* names and storage structure for magnifier physical parameters */
typedef struct {
    double mx, mxp, my, myp, ms, mdp;
    } MAGNIFY;

/* names and storage structure for beam sample physical parameters */
typedef struct {
    double fraction;
    long interval;
    } SAMPLE;

/* names and storage structure for horizontal/vertical corrector physical parameters */
extern PARAMETER hvcor_param[N_HVCOR_PARAMS] ;
   
typedef struct {
    double length, xkick, ykick, tilt, b2, xcalibration, ycalibration;
    long edge_effects, order, steering;
    long synchRad, isr;
    double lEffRad;
    } HVCOR;

/* names and storage structure for explicit matrix input from a file */
extern PARAMETER matr_param[N_MATR_PARAMS] ;
typedef struct {
    double length, fraction;
    char *filename;
    short order;
    /* for internal use only */
    short matrix_read, fiducialSeen;
    double sReference;
    VMATRIX M;
    } MATR;

/* names and storage structure for explicit matrix input */
extern PARAMETER ematrix_param[N_EMATRIX_PARAMS] ;
typedef struct {
    double length, angle;
    double dx, dy, dz;
    double tilt, yaw, pitch;
    short order;
    double C[6];
    double deltaP;
    double R[6][6];
    double T[6][6][6];
    /* for internal use only */
    short fiducialSeen;
    double sReference;
    } EMATRIX;

/* names and storage structure for individualized linear matrix input */
extern PARAMETER ilmatrix_param[N_ILMATRIX_PARAMS] ;
typedef struct {
    double length;
    double tune[2], chrom[2], chrom2[2], chrom3[2];
    double tswax[2], tsway[2]; /* dnux/dAx, dnuy/dAx, dnux/dAy, dnuy/dAy */
    double tswax2[2], tsway2[2]; /* dnux/dAx^2, dnuy/dAx^2, dnux/dAy^2, dnuy/dAy^2 */
    double tswaxay[2]; /* dnux/dAx/dAy, dnuy/dAx/dAy */
    double beta[2], beta1[2], alpha[2], alpha1[2];
    double eta[4], eta1[4];
    double alphac[3];
    double dsdA[2];
    double dsdA2[2];
    double dsdAxAy;
    double tilt;
    short allowResonanceCrossing, verbosity;
    } ILMATRIX;

/* names and storage structure for scattering element physical parameters */
typedef struct {
  double x, xp, y, yp, dp, probability;
  long startOnPass, endOnPass;
  char *distribution;
} SCATTER;

/* names and storage structure for Compton-backscattering element physical parameters.
 * Models Compton back-scattering of a counter-propagating laser off the beam, following
 * Pan et al., Phys. Rev. Accel. Beams 22, 040702 (2019), Eq. 4.  Zero-length per-particle
 * stochastic kick applied once per pass. */
typedef struct {
  double laserWavelength;     /* LASER_WAVELENGTH, m  (E_h = h c / lambda) */
  double pulseEnergy;         /* PULSE_ENERGY, J */
  double sigmax, sigmay;      /* SIGMA_X, SIGMA_Y : laser rms sizes at focus, m */
  double sigmaz;              /* LASER_PULSE_LENGTH : laser rms length, m */
  double dx, dy;              /* DX, DY : laser transverse offset at focus, m */
  double focusPosition;       /* FOCUS_POSITION : S0, longitudinal focus offset from element, m */
  double delay;               /* LASER_DELAY : laser vs reference-particle timing offset, s */
  double theta0;              /* COLLISION_ANGLE, rad (default PI = head-on; !=PI => crossing angle) */
  double azimuth;             /* COLLISION_AZIMUTH, rad (default 0 = crossing in the x-z plane) */
  double factor;              /* FACTOR : scale on scatter rate (default 1) */
  double nPhotons;            /* N_PHOTONS : override pulse-energy photon count if > 0 */
  long nSteps;                /* N_STEPS : longitudinal quadrature points (default 21) */
  long startOnPass, endOnPass;
  long passInterval;          /* PASS_INTERVAL : act only every Nth pass counting from STARTONPASS (default 1) */
  char *crossSection;         /* "klein-nishina" | "thomson" */
  long exactRecoil;           /* EXACT_RECOIL : 0 = approximate Eq.(4) lab recoil (default);
                                 1 = exact per-particle rest-frame Compton boost */
  char *photonOutputFile;     /* NULL/"" => no photon output */
  /* for internal use only: */
  short photonFileActive, initialized;
  SDDS_DATASET *SDDSphotons;
} CBSCAT;


/* names and storage structure for distribution-based scattering element physical parameters */
typedef struct {
  long *particleIDScattered, nParticles, group, nScattered, allScattered;
} DSCATTER_GROUP;

typedef struct {
  char *plane, *fileName, *valueName, *cdfName, *pdfName;
  long oncePerParticle;
  double factor, probability;
  long group, randomSign, limitPerPass, limitTotal, startOnPass, endOnPass;
  /* internal variables */
  short initialized, firstInGroup;
  long iPlane, nData, groupIndex, nLeft;
  double *indepData, *cdfData;
} DSCATTER;

typedef struct {
  long nbins, distIn;
  long *ipage_his;
  double charge, frequency, ignoredPortion;
  double emitN[3], range[3];
  double sigz, delta_p0;
} TSCATTER_SPEC;

typedef struct {
  long dummy;
  /* internal variables */
  char *name;
  double s, betagamma, gamma,pCentral_mev, deltaP, deltaN;
  double AveR, p_rate, s_rate, i_rate, total_scatter;
  char *losFile, *bunFile, *disFile,*iniFile, *outFile;
  double twiss[3][3], disp[2][2];
  double sigx, sigy, sigz, sigxyz;
  double factor, totalWeight;
  double xmin[6], xmax[6];
  ntuple *thist, *fullhis, *xhis, *yhis, *zhis;
  long simuCount; 
} TSCATTER;

/* names and storage structure for polynomial kick terms */
extern PARAMETER kpoly_param[N_KPOLY_PARAMS];

typedef struct {
    double coefficient, tilt, dx, dy, dz, factor;
    long order;
    char *plane;
    /* for internal use only: */
    long yplane; 
    } KPOLY;

/* names and storage structure for polynomial kick terms */
extern PARAMETER hkpoly_param[N_HKPOLY_PARAMS];

typedef struct {
  double length;
  double K[7][7], D[7][7], E[7][7][7];
  double tilt, dx, dy, dz, factor;
  long nRepeats;
  short driftType;
  } HKPOLY;

/* names and storage structure for numerically integrated bending magnet physical parameters */
extern PARAMETER nibend_param[N_NIBEND_PARAMS];

typedef struct {
    double length, angle, e[2], tilt;
    double dx, dy, dz;
    double fint, hgap;          /* used to calculate flen */
    double fp1, fp2, fp3, fp4;  /* fringe-field parameters 1 and 2 */
    double fse;                 /* Fractional Strength Error */
    double etilt;               /* error tilt angle */
    double accuracy;
    char *model, *method;
    long synch_rad, adjustBoundary, adjustField, fudgePathLength, fringePosition;
    /* for internal use only: */
    long initialized;       /* initialization done */
    double flen;            /* distance from iron edge to end of fringe field */
    double rho0;            /* central bending radius */ 
    double fse_adjust;      /* if adjustField, drho/rho0 to get correct bending angle */
    double zeta_offset;     /* if adjustBoundary, offset of fringe field perpendicular 
                             * to pole face to get correct bend angle. Also affected by the
                             * setting of boundaryPosition */
    double x_correction;    /* used to fix spurious central trajectory offsets */
    double s_offset;        /* error in path length, which must be compensated with drifts before and after */
    long angleSign;         /* used internally to keep sign of angle separate from angle */
    unsigned long edgeFlags;
    long e1Index, e2Index;
    } NIBEND;

/* names and storage structure for numerically integrated septum magnet physical parameters */
extern PARAMETER nisept_param[N_NISEPT_PARAMS];

typedef struct {
    double length;          /* straight-line length */
    double angle;           /* desired bending angle */
    double e1;
    double b1;              /* cartesian gradient is Bo*b1 */
    double q1_ref;          /* reference coordinate for Bo, gradient */
    double flen;            /* fringe-field length */
    double accuracy;
    char *method;
    char *model;            /* linear only for this element */
    /* for internal use only: */
    double e2;              /* e2 = angle - e1 */
    double rho_ideal;       /* length/angle */
    double rho0;            /* bending radius along q1=q1_ref */ 
    double fse_opt;         /* optimum fse to get right bending angle */
    double last_fse_opt;    
    double q1_offset;
    long negative_angle;
    unsigned long edgeFlags;
    } NISEPT;

/* names and storage structure for ramped RF cavity physical parameters */
extern PARAMETER ramprf_param[N_RAMPRF_PARAMS] ;
   
typedef struct {
    double length, volt, phase, freq;
    long phase_reference;        /* only meaningful if no frequency ramping */
    char *vwaveform, *pwaveform, *fwaveform, *fiducial;
    /* for internal use only: */
    double Ts;                 /* accumulated time-of-flight of central particle */
    double *t_Vf, *Vfactor;    /* (time, V/volt) pairs */
    double *t_dP, *dPhase;     /* (time, delta phase) pairs */
    double *t_ff, *ffactor;    /* (time, frequency/freq) pairs */
    long n_Vpts, n_Ppts, n_fpts;
    double phase_fiducial;
    long fiducial_seen;
    } RAMPRF;

/* names and storage structure for momentum ramp */
extern PARAMETER rampp_param[N_RAMPP_PARAMS] ;
   
typedef struct {
    char *waveform;
    /* for internal use only: */
    double Po;                 /* set to central momentum for first beam seen */
    double *t_Pf, *Pfactor;    /* (time, P/Po) pairs */
    long n_pts;
    } RAMPP;

/* names and storage structure for stray fields */
extern PARAMETER stray_param[N_STRAY_PARAMS] ;
typedef struct {
    double length;
    double lBx, lBy;         /* local field, in Tesla */
    double gBx, gBy, gBz;    /* global field, in Tesla */
    long order;
    /* for internal use only */
    long WiInitialized;
    void *Wi;              /* pointer to inverse of survey rotation matrix */
                           /* use a void here because we can't load the matrix.h */
                           /* file due to conflicts between two matrix libraries */
    } STRAY;

/* names and storage structure for canonically-integrated bending magnet physical parameters */
extern PARAMETER csbend_param[N_CSBEND_PARAMS];

typedef struct {
    double length, angle;
    double k1, k2, k3, k4, k5, k6, k7, k8;
    double e[2], tilt;
    double h[2], hgap, fintBoth, fint[2];
    double fringeInt[2][7]; /* entrance and exit fringe integrals */
    double dx, dy, dz, xKick, yKick;
    double fse, fseDipole, fseQuadrupole;     /* Fractional Strength Error (combined, dipole, quadrupole) */
    double etilt, epitch, eyaw;   /* error tilt, pitch, yaw angle */
    long nSlices;
    short etiltSign, nonlinear, synch_rad, synchRadInOrdinaryMatrix;
    short edge_effects[2], edge_order;
    short integration_order, expandHamiltonian;
    double edge_kick_limit[2];
    short kick_limit_scaling;
    short use_bn, expansionOrder;
    double b1, b2, b3, b4, b5, b6, b7, b8;
    double xReference, f1, f2, f3, f4, f5, f6, f7, f8;
    double g1, g2, g3, g4, g5, g6, g7, g8;
    short isr, isr1Particle, sqrtOrder;
    short distributionBasedRadiation, includeOpeningAngle;
    char *photonOutputFile;
    double photonLowEnergyCutoff;
    short referenceCorrection, trackingMatrix, fseCorrection, malignMethod;
    short xSteering, ySteering;
    /* for internal use only: */
    unsigned short edgeFlags, edgeFlip;
    double b[9], c[9], fseCorrectionValue, fseCorrectionPathError;
    short refTrajectoryChangeSet;
    double refLength, refAngle, **refTrajectoryChange;
    long refSlices;
    short photonFileActive;
    SDDS_DATASET *SDDSphotons;
    short e1Index, e2Index;
    // memoization for field
    short expansionOrder_ref, nonlinear_ref;
    long expansionOrder1_ref, hasSkew_ref, hasNormal_ref;
    double h_ref;
    double b_ref[9], c_ref[9];
    double **Fx_xy_ref, **Fy_xy_ref;
    } CSBEND;

/* names and storage structure for canonically-integrated rectangular bending magnet physical parameters */
extern PARAMETER ccbend_param[N_CCBEND_PARAMS];
#define N_CCBEND_FRINGE_INT 8
#define MAX_EXTRA_ORDER 21

typedef struct {
    double length, angle;
    double K1, K2, K3, K4, K5, K6, K7, K8;
    double tilt, yaw;
    short fringeModel;
    double hgap, fint1, fint2;
    double fringeInt1[N_CCBEND_FRINGE_INT], fringeInt2[N_CCBEND_FRINGE_INT];
    double dx, dy, dz;
    double eTilt, ePitch, eYaw;
    short malignMethod;
    double fse, fseDipole, fseQuadrupole;     /* Fractional Strength Error (combined, dipole, quadrupole) */
    double xKick;
    long nSlices;
    short integration_order;
    char *systematic_multipoles, *edge_multipoles, *edge1_multipoles, *edge2_multipoles, *random_multipoles;
    double systematicMultipoleFactor, randomMultipoleFactor;
    short referenceOrder;
    short minMultipoleOrder[2], maxMultipoleOrder[2]; /* normal, skew */
    short synch_rad, isr, isr1Particle, distributionBasedRadiation, includeOpeningAngle, synchRadInOrdinaryMatrix;
    short optimizeFse, optimizeDx, optimizeFseOnce, optimizeDxOnce, compensateKn, referenceCorrection;
    short edgeOrder, dxdySign, verbose;
    char *centroidOutputFile;
    /* for internal use only: */
    short optimized, edgeFlip;
    double fseOffset, dxOffset, KnDelta, xAdjust, extraTilt;
    double referenceData[6]; /* length, angle, K1, K2, yaw, tilt */
    double referenceTrajectory[5];
    short multipolesInitialized;
    MULTIPOLE_DATA systematicMultipoleData; 
    MULTIPOLE_DATA edge1MultipoleData; 
    MULTIPOLE_DATA edge2MultipoleData; 
    MULTIPOLE_DATA randomMultipoleData;
    short totalMultipolesComputed;
    MULTIPOLE_DATA totalMultipoleData;  /* generated when randomization takes place */
    short centroidsRequested;  /* non-zero on all workers */
    SDDS_DATASET *SDDScen;     /* non-null only on myid==1 */
    } CCBEND;

/* names and storage structure for canonically-integrated bending magnet with CSR physical parameters */
extern PARAMETER csrcsbend_param[N_CSRCSBEND_PARAMS];

typedef struct {
    double length, angle, k1, k2, k3, k4, k5, k6, k7, k8;
    double e[2], tilt;
    double h[2], hgap, fint;
    double dx, dy, dz;
    double fse;     /* Fractional Strength Error */
    double etilt;   /* error tilt angle */
    long nSlices;
    short etiltSign, nonlinear, useMatrix, synch_rad;
    short edge_effects[2],  edge_order;
    short integration_order;
    long bins;
    double binSize;
    short binOnce;
    double binRangeFactor;
    short SGHalfWidth, SGOrder, SGDerivHalfWidth, SGDerivOrder, trapazoidIntegration;
    char *histogramFile;
    long outputInterval;
    short outputLastWakeOnly, steadyState, integratedGreensFunction;
    short use_bn, expansionOrder;
    double b1, b2, b3, b4, b5, b6, b7, b8;
    short isr, isr1Particle, csr, csrBlock;
    char *derbenevCriterionMode, *particleOutputFile;
    long particleOutputInterval, sliceAnalysisInterval;
    double lowFrequencyCutoff0, lowFrequencyCutoff1;
    double highFrequencyCutoff0, highFrequencyCutoff1;
    short clipNegativeBins;
    char *wakeFilterFile, *wffFreqColumn, *wffRealColumn, *wffImagColumn;
    /* for internal use only: */
    short wakeFileActive, particleFileActive, backtrack;
    SDDS_DATASET *SDDSout, *SDDSpart;
    double b[9], c[9];
    short xIndex, xpIndex, tIndex, pIndex;
    long wffValues;
    double *wffFreqValue, *wffRealFactor, *wffImagFactor;
    unsigned short edgeFlags;
    short e1Index, e2Index;
    } CSRCSBEND;

/* names and storage structure for drift with CSR */
extern PARAMETER csrdrift_param[N_CSRDRIFT_PARAMS];

typedef struct {
  double length, attenuationLength, dz;
  long nKicks;
  short spread, useOvertakingLength;
  double overtakingLengthMultiplier;
  short csr, useSaldin54;
  long nSaldin54Points;
  char *normMode, *spreadMode, *wavelengthMode, *bunchlengthMode, *Saldin54Output;
  short useStupakov;
  char *StupakovOutput;
  long StupakovOutputInterval, sliceAnalysisInterval;
  short linearOptics;
  short LSCInterpolate;
  long LSCBins;
  double LSCLowFrequencyCutoff0, LSCLowFrequencyCutoff1, LSCHighFrequencyCutoff0, LSCHighFrequencyCutoff1, LSCRadiusFactor;
  /* used internally only */
  FILE *fpSaldin;
} CSRDRIFT;

/* names and storage structure for top-up bending magnet physical parameters */
extern PARAMETER tubend_param[N_TUBEND_PARAMS];

typedef struct {
  double length, angle;
  double fse;
  double offset, magnet_width, magnet_angle;
} TUBEND;

/* names and storage structure for traveling-wave muffin-tin accelerator */
extern PARAMETER twmta_param[N_TWMTA_PARAMS];

typedef struct {
    /* variables set by the user (assigned values in compute_matrices): */
    double length, frequency, phase;
    double Ez, accuracy;
    double x_max, y_max;
    double dx, dy, kx;
    double beta_wave, Bsol, alpha;
    long phase_reference, n_steps;
    char *method, *fiducial; 
    /* variables for internal use only: */
    double *fiducial_part;            
    double ky;
    double ExS, EyS, EzS;
    double BxS, ByS, BsolS;
    double phase0;       /* phase at which fiducial particle reaches center of
                            first cell */
    double kz;           /* omega/(c*beta_wave) */
    double Kx, Ky, Kz;   /* kx/ko etc. */
    double alphaS;       /* alpha/k */
    } TWMTA;

/* names and storage structure for matter physical parameters */
extern PARAMETER matter_param[N_MATTER_PARAMS];

typedef struct {
    double length, lEffective;
    double Xo;       /* radiation length */
    long energyDecay, energyStraggle, nuclearBremsstrahlung, electronRecoil, Z;
    double A, rho, pressure, temperature;
    long multiplicity;
    double pLimit;
    double width, spacing, tilt, center;
    long nSlots;
    long startPass, endPass;
    short forceMCS;
    } MATTER;

/* names and storage structure for RF mode physical parameters */
extern PARAMETER rfmode_param[N_RFMODE_PARAMS];

typedef struct {
  /* Data for an IIR filter with input x[i] and output y[i]
     y[n] = Sum[i=1, N] a[i]*y[n-i] + Sum[i=0, N] b[i]*x[n-i]
     */
  long nTerms;      /* Number of terms in the filter. If zero, filter is ignored. */
  double *an, *bn;  /* filter coefficients */
  double *xn, *yn;  /* input, output buffers, treated as circular buffers */
  long iBuffer;     /* index of start location in buffer (1 delay value) */
} IIRFILTER;

int readIIRFilter(IIRFILTER *filterBank, long maxFilters, char *inputFile);
double applyIIRFilter(IIRFILTER *filterBank, long nFilters, double x);
void freeIIRFilterMemory(IIRFILTER *filterBank, long nFilters);

typedef struct {
    double Ra, Rs, Q, freq;    /* 2*Rs, Rs=shunt impedance, Q, mode resonant frequency */
    double charge;             /* total initial charge (use of CHARGE element is better) */
    double initial_V;          /* initial voltage */
    double initial_phase;      /* phase of initial voltage */
    double initial_t;          /* time offset (match with the beam arrival time) */
    double beta;               /* the cavity beta (default is 0) */
    double bin_size;           /* size of charge bins */
    long n_bins;               /* number of charge bins */
    long interpolate;          /* if nonzero, interpolate within bins */
    long preload;              /* preload with steady-state voltage for point bunch */
    double preloadCharge;      /* preload with steady-state voltage for point bunch using this total beam charge */
    double preload_factor;     /* factor to multiply preload voltage by--usually 1 */
    long preloadHarmonic;      /* allows determingin detuning if detuning is larger that frev/2 */
    long rigid_until_pass;     /* beam is "rigid" until this pass */
    long detuned_until_pass;   /* cavity is completely detuned until this pass */
    long sample_interval;      /* sample interval for record file */
    long flush_interval;       /* flush interval for record file */
    char *record;              /* name of file to record (t, V) in */
    long single_pass;          /* controls accumulation of voltage from turn-to-turn */
    long pass_interval;        /* number of passes between applications of wake */
    char *fwaveform, *Qwaveform;  /* waveforms for f/f0 and Q/Q0 vs time */
    long rampPasses;           /* If nonzero, the number of passes over which to ramp impedance up */
    long binless;              /* If nonozero, then use particle-by-particle algorithm. */
    long reset_for_each_step;  /* If nonzero (default), then mode voltage and phase are reset for each step */
    long long_range_only;      /* If nonzero, then only "long-range" effect is included (from previous passes) */
    long allowUnbinnedParticles; /* If nonzero, then program will keep running even if particles fall outside binning region. */
    long n_cavities;           /* multiply effect by this number */
    long bunchedBeamMode;
    double bunchInterval;      /* use when bunchedBeamMode>1 indicating pseudo bunches */
    double driveFrequency;     /* must be non-zero or no generator voltage */
    double voltageSetpoint;    /* desired total cavity voltage, to be achieved by feedback */
    double phaseSetpoint;      /* desired total cavity phase, to be achieved by feedback */
    long updateInterval;       /* feedback update interval in buckets */
    long readOffset;           /* offset of voltage, phase reading relative to filled bucket */
    long adjustmentStart, adjustmentEnd, adjustmentInterval;
    double adjustmentFraction;
    char *amplitudeFilterFile, *phaseFilterFile;
    char *IFilterFile, *QFilterFile;
    char *feedbackRecordFile;
    long muteGenerator;        /* if non-zero, generator output is muted */
    double generatorFactor;
  /** char *noiseAlphaGen, *noisePhiGen; */
#define I_NOISE_ALPHA_GEN 0
#define I_NOISE_PHI_GEN 1
#define I_NOISE_ALPHA_V 2
#define I_NOISE_PHI_V 3
#define I_NOISE_I_GEN 4
#define I_NOISE_Q_GEN 5
#define I_NOISE_I_V 6
#define I_NOISE_Q_V 7
    char *noiseData[8];
    /* for internal use: */
    double RaInternal;         /* used to store Ra or 2*Rs, whichever is nonzero */
    double mp_charge;          /* charge per macroparticle */
    long initialized;          /* indicates that beam has been seen */
    /* BEAM-INDUCED voltage data */
    double V;                  /* magnitude of voltage */
    double Vr, Vi;             /* real, imaginary components of voltage phasor at t=tlast */
    double last_t;             /* time at which last particle was seen (register-reduced) */
    double last_phase;         /* phase at t=last_t */
    double last_omega;         /* omega at t=last_t */
    double last_Q;             /* loaded Q at t=last_t */
    double last_clockOffset;   /* trackingClockOffset() at t=last_t; damping uses true elapsed time = (t-last_t)+(clockNow-last_clockOffset) */
    double last_clockPhase;    /* trackingClockPhase(omega) at t=last_t; phase advance adds (clockPhaseNow-last_clockPhase) for a non-harmonic mode */
    /* generator- and feedback-related data, see T. Berenc RF-TN-2015-001 */
    double setpointAdjustment, fbVCavity;
    double lambdaA;          /* 2/((Ra/Q)*Qloaded) */
    double Vg, phaseg, tg;   /* Used to determine the voltage and phase seen during the bunch passage according to Vg*cos(omega*(t-tg) + phaseg) */
    MATRIX *Viq;             /* (2x1) matrix giving I and Q components of generator voltage */
    MATRIX *Iiq;             /* (2x1) matrix giving I and Q components of generator current */
    MATRIX *Ig0;             /* I and Q components of reference generator current (gives desired voltage and phase when beam current is zero) */
    MATRIX *A, *B;           /* Berenc's matrices for evolution of the voltage, used between ticks*/
    MATRIX *At, *Bt;         /* Berenc's matrices for evolution of the voltage, used for particle bins */
    MATRIX *Mt1, *Mt2, *Mt3; /* temporary matrices for carrying out computations */
    long nAmplitudeFilters, nPhaseFilters, nIFilters, nQFilters;
    IIRFILTER amplitudeFilter[4]; /* output of filters is summed */
    IIRFILTER phaseFilter[4];     /* output of filters is summed */
    IIRFILTER IFilter[4]; /* output of filters is summed */
    IIRFILTER QFilter[4];     /* output of filters is summed */
    double V0, last_phase0;  /* needed for I/Q feedback */
    double fbLastTickTime;   /* time at which last FB tick occurred, adjusted when first bunch is seen */
    double fbNextTickTime;   /* time at which next FB tick will occur */
    double fbNextTickTimeError; /* Used with Kahan sum rule to reduce error */
    double tGenerator;       /* reference time for the generator voltage */
    short fbRunning;         /* if non-zero, fbFirstTickTime has been set */
    /* frequency table */
    double *tFreq, *fFreq;
    long nFreq;
    /* Q table */
    double *tQ, *fQ;
    long nQ;
    /* noise tables */
    double *tNoise[8], *fNoise[8];
    long nNoise[8];                /* If 0, no noise data */
    /* files for record output */
    SDDS_DATASET *SDDSrec  ;    /* seen by beam */
    long sample_counter;       /* row in the output record */
    SDDS_DATASET *SDDSfbrec;  /* seen by  feedback system */
    long fbSample;           /* row in the output record */
    long fileInitialized;
    } RFMODE;

/* names and storage structure for RF-mode-from-file physical parameters */
extern PARAMETER frfmode_param[N_FRFMODE_PARAMS];

typedef struct {
    char *filename;
    double bin_size;           /* size of charge bins */
    long n_bins;               /* number of charge bins */
    long rigid_until_pass;     /* beam is "rigid" until this pass */
    long useSymmData;          /* use Symm data from URMEL output? */
    double factor;             /* multiply impedance by this factor */
    double cutoffFrequency;    /* modes above this frequency are ignored */
    char *outputFile;          /* output file for voltage in each mode */
    long flushInterval;       /* interval at which data is flushed */
    long rampPasses;           /* If nonzero, the number of passes over which to ramp impedance up */
    long reset_for_each_step;  /* If nonzero (default), then mode voltage and phase are reset for each step */
    long long_range_only;      /* If nonzero, then only "long-range" effect is included (from previous passes) */
    long n_cavities;           /* multiply effect by this number */
    long bunchedBeamMode;
    /* for internal use: */
    double mp_charge;          /* charge per macroparticle */
    long initialized;          /* indicates that beam has been seen */
    long modes;                /* number of modes */
    double *V;                 /* magnitude of voltage */
    double *Vr, *Vi;           /* real, imaginary components of voltage phasor at t=tlast */
    double *omega;             /* frequency */
    double *Q;                 /* loaded quality factor */
    double *Rs;                /* shunt impedance */
    double *beta;              /* normalized load impedance */
    double last_t;             /* time at which last particle was seen (register-reduced) */
    double *last_phase;        /* phase at t=last_t */
    double last_clockOffset;   /* trackingClockOffset() at t=last_t; damping uses true elapsed time = (t-last_t)+(clockNow-last_clockOffset) */
    double *last_clockPhase;   /* per-mode trackingClockPhase(omega) at t=last_t; phase advance adds (clockPhaseNow-last_clockPhase) for a non-harmonic mode */
    SDDS_DATASET *SDDSout;
    long *modeIndex;           /* SDDS index of mode column in output file */
    } FRFMODE;

/* names and storage structure for transverse RF mode physical parameters */
extern PARAMETER trfmode_param[N_TRFMODE_PARAMS];

typedef struct {
    double Ra, Rs, Q, freq;    /* 2*Rs, Rs=shunt impedance, Q, frequency */
    double charge;             /* total initial charge */
    double beta;               /* the cavity beta (default is 0) */
    double bin_size;           /* size of charge bins */
    long n_bins;               /* number of charge bins */
    long interpolate;          /* interpolate within bins */
    char *plane;               /* "x", "y", or "both" */
    long sample_interval;      /* sample interval for record file */
    long perParticleOutput;    /* asks for per-particle output in record file */
    char *record;              /* name of file to record (t, V) in */
    long single_pass;          /* controls accumulation of voltage from turn-to-turn */
    long rigid_until_pass;     /* don't affect beam until this pass */
    double dx, dy;
    double xfactor, yfactor;
    long rampPasses;           /* If nonzero, the number of passes over which to ramp impedance up */
    long binless;
    long reset_for_each_step;  /* If nonzero (default), then mode voltage and phase are reset for each step */
    long long_range_only;      /* If nonzero, then only "long-range" effect is included (from previous passes) */
    long n_cavities;           /* multiply effect by this number */
    long bunchedBeamMode;
    /* for internal use: */
    double RaInternal;         /* used to store Ra or 2*Rs, whichever is nonzero */
    long doX, doY;
    double mp_charge;          /* charge per macroparticle */
    long initialized;          /* indicates that beam has been seen */
    double Vx;                 /* magnitude of voltage */
    double Vxr, Vxi;           /* real, imaginary components of voltage phasor at t=tlast */
    double Vy;                 /* magnitude of voltage */
    double Vyr, Vyi;           /* real, imaginary components of voltage phasor at t=tlast */
    double last_t;             /* time at which last particle was seen (register-reduced) */
    double last_xphase;        /* phase at t=last_t */
    double last_yphase;        /* phase at t=last_t */
    double last_clockOffset;   /* trackingClockOffset() at t=last_t; damping uses true elapsed time = (t-last_t)+(clockNow-last_clockOffset) */
    double last_clockPhase;    /* trackingClockPhase(omega) at t=last_t; phase advance adds (clockPhaseNow-last_clockPhase) for a non-harmonic mode (shared x/y) */
    SDDS_DATASET *SDDSrec;
    long fileInitialized;
    } TRFMODE;


/* names and storage structure for transverse-RF-mode-from-file physical parameters */
extern PARAMETER ftrfmode_param[N_FTRFMODE_PARAMS];

typedef struct {
    char *filename;
    double bin_size;           /* size of charge bins */
    long n_bins;               /* number of charge bins */
    long rigid_until_pass;     /* beam is "rigid" until this pass */
    long useSymmData;          /* use "Symm" columns from URMEL file? */
    double dx, dy;
    double xfactor;            /* multiply impedance by this factor */
    double yfactor;            /* multiply impedance by this factor */
    double cutoffFrequency;    /* modes above this frequency are ignored */
    char *outputFile;          /* output file for voltage in each mode */
    long flushInterval;       /* interval at which data is flushed */
    long rampPasses;           /* If nonzero, the number of passes over which to ramp impedance up */
    long reset_for_each_step;  /* If nonzero (default), then mode voltage and phase are reset for each step */
    long long_range_only;      /* If nonzero, then only "long-range" effect is included (from previous passes) */
    long n_cavities;           /* multiply effect by this number */
    long bunchedBeamMode;
    /* for internal use: */
    double mp_charge;          /* charge per macroparticle */
    long initialized;          /* indicates that beam has been seen */
    long modes;                /* number of modes */
    int32_t *doX, *doY;           /* plane of mode */
    double *Vx;                /* magnitude of voltage */
    double *Vxr, *Vxi;         /* real, imaginary components of voltage phasor at t=tlast */
    double *Vy;                /* magnitude of voltage */
    double *Vyr, *Vyi;         /* real, imaginary components of voltage phasor at t=tlast */
    double *omega;             /* frequency */
    double *Q;                 /* loaded quality factor */
    double *beta;              /* normalized load impedance */
    double *Rs;                /* shunt impedance */
    double last_t;             /* time at which last particle was seen (register-reduced) */
    double *lastPhasex;        /* phase at t=last_t */
    double *lastPhasey;        /* phase at t=last_t */
    double last_clockOffset;   /* trackingClockOffset() at t=last_t; damping uses true elapsed time = (t-last_t)+(clockNow-last_clockOffset) */
    double *last_clockPhase;   /* per-mode trackingClockPhase(omega) at t=last_t; phase advance adds (clockPhaseNow-last_clockPhase) for a non-harmonic mode (shared x/y) */
    SDDS_DATASET *SDDSout;
    long *xModeIndex, *yModeIndex;
    } FTRFMODE;

/* names and storage structure for longitudinal impedance physical parameters */
extern PARAMETER zlongit_param[N_ZLONGIT_PARAMS];

typedef struct {
    double charge;             /* total initial charge */
    long broad_band;           /* flag */
    double Ra, Rs, Q, freq;    /* 2*Rs, Rs=shunt impedance, Q, frequency */
    char *Zreal, *Zimag;       /* impedance vs frequency files */
    double bin_size;           /* size of charge bins */
    long n_bins;               /* number of charge bins--must be 2^n */
    long max_n_bins;
    char *wakes;               /* name of file to save wake potentials in */
    long wake_interval;        /* interval (in turns) between output of wakes */
    long wake_start, wake_end; /* pass on which to start, end output of wakes */
    long area_weight;          /* flag to turn on area-weighting */
    long interpolate;          /* flag to turn on interpolation */
    long smoothing;            /* flag to turn on smoothing */
    long SGOrder, SGHalfWidth; /* Savitzky-Golay smoothing parameters */
    long reverseTimeOrder;     /* use for "acausal" impedances like CSR */
    double factor;             /* multiply impedance by this factor */
    long startOnPass;          /* If nonzero, the pass on which impedance turns on. */
    long rampPasses;           /* If nonzero, the number of passes over which to ramp impedance up */
    double highFrequencyCutoff0, highFrequencyCutoff1;  /* start and stop frequency for smoothing filter */
    long bunchedBeamMode, startBunch, endBunch;
    long allowLongBeam;       /* If nonozero, then long bunches don't cause abort */
    long bin_centered;        /* If nonzero, use ZTRANSVERSE/IMPEDANCE-style bin placement (bunch at left
                                 edge of grid, bin centers at tmin+ib*dt).  Default (0) uses the legacy
                                 ZLONGIT convention (bunch centered in grid, bin left-edges at tmin+ib*dt). */
    /* for internal use: */
    long initialized;          /* indicates that files are loaded */
    double *Z;                 /* n_Z (Re Z, Im Z) pairs */
    /* variables for SDDS output of wakes */
    SDDS_TABLE *SDDS_wake;
    long SDDS_wake_initialized;
    double macroParticleCharge;
    } ZLONGIT;

/* names and storage structure for transverse impedance physical parameters */
extern PARAMETER ztransverse_param[N_ZTRANSVERSE_PARAMS];

typedef struct {
    double charge;             /* total initial charge */
    long broad_band;           /* flag */
    double Rs, Q, freq;        /* shunt impedance, Q, frequency */
    /* input file containing impedance functions */
    char *inputFile;
    /* columns from input file */
    char *freqColumn, *ZxReal, *ZxImag, *ZyReal, *ZyImag;
    double bin_size;           /* size of charge bins */
    long interpolate;          /* whether to interpolate voltage or not */
    long n_bins;               /* number of charge bins--must be 2^n */
    long max_n_bins;
    long smoothing;            /* flag to turn on smoothing */
    long SGOrder, SGHalfWidth; /* Savitzky-Golay smoothing parameters */
    double dx, dy;
    double factor, xfactor, yfactor;  /* multiply impedance by these factors */
    char *wakes;               /* name of file to save wake potentials to */
    long wake_interval;        /* interval (in turns) between output of wakes */
    long wake_start, wake_end; /* pass on which to start, end output of wakes */
    long startOnPass;          /* If nonzero, the pass on which impedance turns on. */
    long rampPasses;           /* If nonzero, the number of passes over which to ramp impedance up */
    double highFrequencyCutoff0, highFrequencyCutoff1;  /* start and stop frequency for smoothing filter */
    long xDriveExponent, yDriveExponent;
    long xProbeExponent, yProbeExponent;
    long bunchedBeamMode, startBunch, endBunch;
    long allowLongBeam;       /* If nonozero, then long bunches don't cause abort */
    /* for internal use */
    double *Z[2];             /* Z (Re Z, Im Z) pairs for each plane */
    long initialized;
    double macroParticleCharge;
    /* variables for SDDS output of wakes */
    SDDS_TABLE *SDDS_wake;
    long SDDS_wake_initialized;
    } ZTRANSVERSE;

/* names and storage structure for combined-impedance physical parameters */
extern PARAMETER impedance_param[N_IMPEDANCE_PARAMS];

#define IMPEDANCE_N_WAKES 5
#define IMPEDANCE_ZL  0
#define IMPEDANCE_ZXD 1
#define IMPEDANCE_ZYD 2
#define IMPEDANCE_ZXQ 3
#define IMPEDANCE_ZYQ 4

typedef struct {
    char *inputFile;
    char *freqColumn;
    /* per-wake column-name pairs (NULL => wake disabled) */
    char *ZL_real,  *ZL_imag;
    char *ZxD_real, *ZxD_imag;
    char *ZyD_real, *ZyD_imag;
    char *ZxQ_real, *ZxQ_imag;
    char *ZyQ_real, *ZyQ_imag;
    /* binning */
    double bin_size;
    long interpolate;
    long n_bins;
    long max_n_bins;
    long smoothing;
    long SGOrder, SGHalfWidth;
    double dx, dy;
    /* scale factors (LRWAKE convention) */
    double factor;
    double xFactor;       /* x dipole */
    double yFactor;       /* y dipole */
    double zFactor;       /* longitudinal */
    double qxFactor;      /* x quadrupole */
    double qyFactor;      /* y quadrupole */
    /* wake diagnostic output */
    char *wakes;
    long wake_interval;
    long wake_start, wake_end;
    /* pass-based controls */
    long startOnPass;
    long rampPasses;
    /* smoothing filter cutoffs */
    double highFrequencyCutoff0, highFrequencyCutoff1;
    /* bunched-beam controls */
    long bunchedBeamMode, startBunch, endBunch;
    long allowLongBeam;
    /* longitudinal-only options */
    long area_weight;
    long reverseTimeOrder;
    /* for internal use */
    long initialized;
    long enabled[IMPEDANCE_N_WAKES]; /* whether each wake type is active */
    double *Z[IMPEDANCE_N_WAKES];    /* per-wake impedance arrays */
    double macroParticleCharge;
    SDDS_TABLE *SDDS_wake;
    long SDDS_wake_initialized;
    long n_freq;            /* number of frequency points in the spectrum */
    } IMPEDANCE;

/* names and storage structure for longitudinal wake physical parameters */
extern PARAMETER wake_param[N_WAKE_PARAMS];

typedef struct {
    char *inputFile, *tColumn, *WColumn;
    double charge;             /* total initial charge */
    double factor;             /* factor to multiply by (e.g., number of cells) */
    long n_bins;               /* number of charge bins */
    long interpolate;          /* flag to turn on interpolation */
    long smoothing, SGHalfWidth, SGOrder;  /* flag to turn on smoothing plus control parameters */
    long change_p0, allowLongBeam;
    long rampPasses;           /* If nonzero, the number of passes over which to ramp wake up */
    long bunchedBeamMode;          /* If nonzero, then do calculations bunch-by-bunch */
    long startBunch, endBunch;
    long acausalAllowed;
    /* for internal use: */
    long initialized;          /* indicates that files are loaded */
    long wakePoints, isCopy, i0;
    double *W, *t, macroParticleCharge, dt;
  } WAKE;

/* names and storage structure for longitudinal corrugated-pipe wake physical parameters */
extern PARAMETER corgpipe_param[N_CORGPIPE_PARAMS];

typedef struct {
    double length, radius, period, gap, depth, dt, tmax;
    long n_bins;
    long interpolate;          /* flag to turn on interpolation */
    long smoothing, SGHalfWidth, SGOrder;  /* flag to turn on smoothing plus control parameters */
    long change_p0, allowLongBeam;
    long rampPasses;           /* If nonzero, the number of passes over which to ramp wake up */
  } CORGPIPE;

/* names and storage structure for long-range wake physical parameters */
extern PARAMETER lrwake_param[N_LRWAKE_PARAMS];

typedef struct {
  char *inputFile;
  char *WColumn[6]; /* t, Wx, Wy, Wz, Qx, Qy column names */
  double factor;             /* factor to multiply by (e.g., number of cells) */
  double xFactor, yFactor, zFactor, qxFactor, qyFactor;
  long turnsToKeep, rampPasses;
  /* for internal use: */
  long initialized;          /* indicates that files are loaded */
  long nBuckets;
  /* variables for holding data from users input file */
  long wakePoints;
  double *W[6];              /* t, Wx, Wy, Wz, Qx, Qy */
  double dt;
  /* bucket history data---centroids of prior buckets */
  long nHistory; /* number of historical buckets */
  double *tHistory, *QHistory, *xHistory, *yHistory;
  } LRWAKE;

/* names and storage structure for transverse wake physical parameters */
extern PARAMETER trwake_param[N_TRWAKE_PARAMS];

typedef struct {
    char *inputFile, *tColumn, *WxColumn, *WyColumn;
    double charge;             /* total initial charge */
    double factor;             /* factor to multiply by (e.g., number of cells) */
    double xfactor, yfactor;   /* individual factors for x and y */
    long n_bins;               /* number of charge bins */
    long interpolate;          /* flag to turn on interpolation */
    long smoothing, SGHalfWidth, SGOrder;  /* flag to turn on smoothing plus control parameters */
    double dx, dy, tilt;
    long xDriveExponent, yDriveExponent;
    long xProbeExponent, yProbeExponent;
    long rampPasses;           /* If nonzero, the number of passes over which to ramp wake up */
    long bunchedBeamMode;          /* If nonzero, then do calculations bunch-by-bunch */
    long startBunch, endBunch;
    long acausalAllowed;
    /* for internal use: */
    long initialized;          /* indicates that files are loaded */
    long wakePoints, isCopy, i0;
    double *W[2], *t, macroParticleCharge, dt;
  } TRWAKE;

/* names and storage structure for the combined-wake element (CWAKE):
 * one element loads up to seven wake channels (longitudinal monopole,
 * x/y dipole, x/y quadrupole, x/y constant) from one SDDS file and
 * shares the bin-assignment and bucket-iteration overhead across
 * every enabled channel.  Modelled on WAKE + TRWAKE; channel naming
 * follows the LRWAKE / IMPEDANCE convention.
 */
extern PARAMETER cwake_param[N_CWAKE_PARAMS];

#define CWAKE_N_WAKES 7
#define CWAKE_WZ 0
#define CWAKE_WX 1
#define CWAKE_WY 2
#define CWAKE_QX 3
#define CWAKE_QY 4
#define CWAKE_CX 5
#define CWAKE_CY 6

typedef struct {
    char  *inputFile, *tColumn;
    char  *WzColumn, *WxColumn, *WyColumn;
    char  *QxColumn, *QyColumn;
    char  *CxColumn, *CyColumn;
    double factor;
    double zFactor, xFactor, yFactor;
    double qxFactor, qyFactor;
    double cxFactor, cyFactor;
    long   n_bins;
    long   interpolate;
    long   smoothing, SGHalfWidth, SGOrder;
    double dx, dy, tilt;
    long   change_p0, allowLongBeam, acausalAllowed;
    long   rampPasses;
    long   bunchedBeamMode, startBunch, endBunch;
    /* internal use */
    long   initialized;
    long   enabled[CWAKE_N_WAKES];
    long   wakePoints, i0;
    double dt;
    double *t;
    double *W[CWAKE_N_WAKES];
    double macroParticleCharge;
} CWAKE;

/* names and storage structure for RF cavity with wake physical parameters */
extern PARAMETER rfcw_param[N_RFCW_PARAMS] ;

typedef struct {
  long bins, interpolate;
  double lowFrequencyCutoff0, lowFrequencyCutoff1;
  double highFrequencyCutoff0, highFrequencyCutoff1;
  double radiusFactor;
  /* internal use only */
  double backtrack;
} LSCKICK;

typedef struct {
    double length, cellLength, volt, phase, freq, Q;
    long phase_reference, change_p0, change_t;
    char *fiducial;
    short end1Focus, end2Focus, standingWave;
    char *bodyFocusModel;
    long nKicks;
    long includeZWake;          /* if zero, longitudial wake is disabled */
    long includeTrWake;          /* if zero, transverse wakes are disabled */
    char *wakeFile, *zWakeFile, *trWakeFile, *tColumn, *WxColumn, *WyColumn, *WzColumn;
    long n_bins;               /* number of charge bins */
    long interpolate;          /* flag to turn on interpolation */
    long smoothing, SGHalfWidth, SGOrder;  /* flag to turn on smoothing plus control parameters */
    double dx, dy;
    double tReference;
    long linearize, doLSC, LSCBins, LSCInterpolate;
    double LSCLowFrequencyCutoff0, LSCLowFrequencyCutoff1;
    double LSCHighFrequencyCutoff0, LSCHighFrequencyCutoff1, LSCRadiusFactor;
    long wakesAtEnd;
    /* for internal use only: */
    short initialized, backtrack;
    RFCA rfca;
    TRWAKE trwake;
    WAKE wake;
    LSCKICK LSCKick;
    } RFCW;

/* names and storage structure for SR effects */
extern PARAMETER sreffects_param[N_SREFFECTS_PARAMS];

typedef struct {
  double Jx, Jy, Jdelta, exRef, eyRef, SdeltaRef, DdeltaRef, pRef, coupling, fraction;
  long damp, qExcite, loss;
  double cutoff;
  long includeOffsets;
} SREFFECTS;

/* names and storage structure for intra-beam scattering */
extern PARAMETER IBSCATTER_param[N_IBSCATTER_PARAMS];

typedef struct {
  double factor;
  short do_x, do_y, do_z;
  short smooth, forceMatchedTwiss, isRing;
  long nslice, interval;
  char *filename; 
  short bunchedBeamMode, parallelIntegration, verbose;
  /* internal use only */
  double charge;
  char **name;
  double *s, *pCentral, *icharge;
  double *etax, *etaxp, *etay, *etayp; 
  double **betax, **alphax, **betay, **alphay;
  double **xRateVsS, **yRateVsS, **zRateVsS;
  double *emitx0, *emity0, *emitl0, *sigmaz0, *sigmaDelta0;
  double *emitx, *emity, *emitl, *sigmaz, *sigmaDelta;
  double *xGrowthRate, *yGrowthRate, *zGrowthRate;
  long elements, offset, output;
  double revolutionLength, dT;
  ELEMENT_LIST *elem;
  SDDS_DATASET outPage;
  long isInit, doOut;
} IBSCATTER;

/* space charge multipole element */
typedef struct {
    /* values for internal use: */
  long flag;
  double lastSigma[3]; /* values of sigmax, sigmay, sigmaz on last pass */
} SCMULT;          

extern PARAMETER bmapxy_param[N_BMAPXY_PARAMS];

typedef struct {
  double length, strength, accuracy;
  char *method, *filename;
  char *FxRpn, *FyRpn;
  /* these are set by the program when the file is read */
  long points, nx, ny, BGiven;
  double *Fx, *Fy;
  double xmin, xmax, dx;
  double ymin, ymax, dy;
  long rpnMem[2];
} BMAPXY;

/* aperture from a file giving (x, y) */
extern PARAMETER apcontour_param[N_APCONTOUR_PARAMS];

typedef struct {
  double length;
  double tilt;            /* roll angle */
  double dx, dy, dz;      /* misalignments */
  double resolution;
  double xFactor, yFactor; /* multiply x and y data by these factors */
  short invert;           /* If non-zero, the shape is an obstruction not an aperture */
  short sticky;           /* If non-zero, the shape is applied until canceled by another APCONTOUR element */
  short cancel;           /* If non-zero, cancel previous APCONTOUR. Valid even without other parameters */
  short holdOff;          /* If non-zero, applied only in the downstream elements (assuming sticky=1) */
  char *filename;         /* filename for generalized gradients vs z */
  char *xColumn;          /* name of column containing x data */
  char *yColumn;          /* name of column containing y data */
  /* these are set by the program when the file is read */
  short initialized, hasLogic;
  double **x, **y;
  char **logic;
  long nContours, *nPoints;
} APCONTOUR;

extern PARAMETER bmapxyz_param[N_BMAPXYZ_PARAMS];

typedef struct {
  short singlePrecision; 
  /* these are copies of pointers, potentially shared with other instances */
  double *Fx, *Fy, *Fz;    /* use if !singlePrecision */ 
  float *Fx1, *Fy1, *Fz1;  /* use if singlePrecision */
  double xmin, xmax, dx;
  double ymin, ymax, dy;
  double zmin, zmax, dz;
  long nx, ny, nz, points, BGiven;
  short magnetSymmetry[3]; /* 0=none, 1=even, 2=odd */
} BMAPXYZ_DATA;

typedef struct {
  double length;
  double dxError, dyError, dzError, tilt;
  double fieldLength, strength, fse, BFactor[3], BInside[3];
  double xInsideLimit[2], accuracy;
  char *method, *filename;
  short synchRad, checkFields, injectAtZero, driftMatrix, xyInterpolationOrder, xyGridExcess;
  short singlePrecision, discardMap, verbosity;
  char *particleOutputFile, *apContourElement;
  double zMinApContour, zMaxApContour;
  /* internal variables */
  BMAPXYZ_DATA *data; 
  SDDS_DATASET *SDDSpo;
  long poIndex[11];
  long poRow, poRows;
  APCONTOUR apContour;
} BMAPXYZ;

extern PARAMETER brat_param[N_BRAT_PARAMS];
typedef struct {
  double length, angle, fse;
  short flip;
  double accuracy;
  char *method, *filename, *filenameAdditional;
  double xVertex, zVertex;
  double xEntry, zEntry;
  double xExit, zExit;
  double dxMap, dyMap, dzMap;
  double yawMap;
  double mainFactor, additionalFactor;
  double fieldFactor, deltaByInside;
  short useFTABLE, xyInterpolationOrder, xyGridExcess, xyExtrapolate, useSbenMatrix, useDriftMatrix, singlePrecision;
  char *particleOutput;
  short particleOutputLostOnly;
  long particleOutputSelectionInterval, particleOutputSampleInterval;
  char *fieldOutputFile;
  long nOutput[3];
  /* these are set by the program when the file is read */
  short initialized, fieldOutputDone;
  long dataIndex, dataIndexAdditional;
  SDDS_DATASET *SDDSparticleOutput;
} BRAT;

/* magnetic field generalized gradient expansion */
extern PARAMETER bggexp_param[N_BGGEXP_PARAMS];

typedef struct {
  double length;          /* for floor coorindates, s coordinate, etc */
  double fieldLength;     /* expected length of the field map */
  char *filename;         /* filename for (normal) generalized gradients vs z */
  char *normalFilename;   /* filename for normal-orientation GG vs z */
  char *skewFilename;     /* filename for skew-orientation GG vs z */
  double strength;        /* multiply fields by a factor */
  double multipoleFactor[5]; /* for solenoid, dipole, quadrupole, sextupole, octupole terms */
  double BFactor[3];         /* Bx, By, Bz factors */
  double tilt;            /* roll angle */
  double dx, dy, dz;      /* misalignments */
  double Bx, By;          /* stray field */
  short mMaximum;          /* maximum value of m that is included */
  short maximum2n;         /* maximum value of 2*n that is included */
  short zInterval;         /* interval between z points used */
  short symplectic;        /* use symplectic integrator */
  short synchRad, isr;
  char *particleOutputFile;  
  short isBend;
  double xVertex, zVertex;
  double xEntry, zEntry;
  double xExit, zExit;
  double dxExpansion;
  /* these are set by the program when the file is read */
  short initialized;
  long dataIndex[2]; /* normal, skew */
  SDDS_DATASET *SDDSpo;
  long poIndex[11];
} BGGEXP;

/* magnetic field from on-axis data */
extern PARAMETER boffaxe_param[N_BOFFAXE_PARAMS];

typedef struct {
  double length;          /* for floor coorindates, s coordinate, etc */
  double fieldLength;     /* expected length of the field map */
  char *filename;         /* filename for generalized gradients vs z */
  char *zColumn;          /* name of column containing z data */
  char *fieldColumn;      /* name of column containing field data */
  short order;            /* 1=quadrupole, 2=sextupole, ... */
  short expansionOrder;   /* maximum order of expansion in (x, y) */
  double strength;        /* multiply fields by a factor */
  double tilt;            /* roll angle */
  double dx, dy, dz;      /* misalignments */
  double Bx, By;          /* stray field */
  short zInterval;        /* interval between z points used */
  short zSubdivisions;    /* whether to subdivide the z steps */
  short synchRad, isr;
  char *particleOutputFile;  
  char *fieldOutputFile;
  long nOutput[2];
  double halfSpanOutput[2];
  /* these are set by the program when the file is read */
  short initialized;
  long dataIndex;
  SDDS_DATASET *SDDSpo;
  long poIndex[9];
} BOFFAXE;

/* names and storage structure for CHARGE element */
extern PARAMETER charge_param[N_CHARGE_PARAMS];
typedef struct {
  double charge, chargePerParticle;
  long allowChangeWhileRunning;
  /* for internal use only */
  double macroParticleCharge;
  long idSlotsPerBunch; /* copied from BEAM structure by do_tracking */
} CHARGE;

/* Names and storage structure for POLARization element */  
extern PARAMETER polar_param[N_POLAR_PARAMS];
typedef struct {
  double polarization, e[3];
  char *mode;
  long onPass;
} POLAR;
  
/* names and storage structure for PFILTER element */
extern PARAMETER pfilter_param[N_PFILTER_PARAMS];
typedef struct {
  double deltaLimit, lowerFraction, upperFraction;
  long fixPLimits, beamCentered, bins;
  /* for internal use only */
  long limitsFixed;
  double pLower, pUpper;
  long hasLower, hasUpper;
} PFILTER;

/* names and storage structure for WIGGLER element */
extern PARAMETER wiggler_param[N_WIGGLER_PARAMS];
typedef struct {
  double length, radius, K, B;
  double dx, dy, dz, tilt;
  long poles;
  short focusing;
  char *model;
  double gap, jFraction, C[3];
  /* internal use only */
  double radiusInternal;  /* may be computed from K or gap */
} WIGGLER;

/* names and storage structure for CWIGGLER element */
extern PARAMETER cwiggler_param[N_CWIGGLER_PARAMS];
typedef struct {
  double length, BMax, BxMax, ByMax, tguGradient, tguCompFactor, poleFactor[3];
  double dx, dy, dz, tilt;
  long periods, stepsPerPeriod;
  short integrationOrder;
  char *ByFile, *BxFile;
  short BySplitPole, BxSplitPole;
  short sr, isr, isr1Particle, sinusoidal, vertical, helical, tgu;
  short forceMatched;
  char *fieldOutput;
  short verbosity;
  double BConstant[2];
  char *model;
  double gap, jFraction, C[3];
  /* for internal use */
  short initialized;
  double BMaxToUse, BxMaxToUse, ByMaxToUse; /* copies of user data, or computed from model */
  double *ByData, *BxData; 
  long ByHarmonics, BxHarmonics;
  double BPeak[2];              
  double radiusInternal[2];     
  double zEndPointH[2], zEndPointV[2];
  SDDS_DATASET *SDDSFieldOutput;
  short fieldOutputInitialized;
  long fieldOutputRow, fieldOutputRows;
  short endFlag[2]; /* indicate whether to include the pole factors at entrance and exit */
} CWIGGLER;

/* names and storage structure for APPLE-II element */
extern PARAMETER apple_param[N_APPLE_PARAMS];
typedef struct {
  double length, BMax, shimScale;
  double dx, dy, dz, tilt;
  long periods, step;
  short order, End_Pole, shimOn;
  char *Input;
  char *shimInput;
  short sr, isr, isr1Particle;
  double x0, gap0, dgap, phi1, phi2, phi3, phi4;
  short verbosity;
  /* for internal use */
  short initialized;
  long NxHarm, NzHarm, ShimHarm;
  double C1, C2, C3, C4;
  double S1, S2, S3, S4;
  double **Cij, **kx, **ky, *kz;
  double **CoZ, **CxXoZ, **CxYoZ, **CxXoYZ, **CxX2oYZ;
  double **CxX2oZ, **CxXYoZ, **CxX3oYZ, **CxY2oZ;
  double lz;
  double kx_shim, *Ci_shim, *Si_shim;
  short drift;
  double BPeak[2], radiusInternal[2];
} APPLE;

/* names and storage structure for SCRIPT element */
extern PARAMETER script_param[N_SCRIPT_PARAMS];
typedef struct {
  double length;
  char *command;
  short useCsh, verbosity, rpnParameters;
  long startPass, endPass, passInterval, onPass;
  char *directory, *rootname, *inputExtension, *outputExtension;
  short keepFiles, driftMatrix, driftTestParticles;
  short useParticleID, noNewParticles, determineLossesFromParticleID, softFailure;
  char *postCommandParameterFile;
  double NP[10];
  char *SP[10];
} SCRIPT;

/* Light Thin Lens element */
extern PARAMETER lthinlens_param[N_LTHINLENS_PARAMS];
typedef struct {
  double fx, fy;
  double dx, dy, dz;
  double tilt, yaw, pitch;
} LTHINLENS;

/* Light Mirror element */
extern PARAMETER lmirror_param[N_LMIRROR_PARAMS];
typedef struct {
  double Rx, Ry;
  double theta;
  double dx, dy, dz;
  double tilt, yaw, pitch;
} LMIRROR;


#define TFB_FILTER_LENGTH 30
/* Transverse Feedback Pickup element */
extern PARAMETER tfbpickup_param[N_TFBPICKUP_PARAMS];
typedef struct {
  char *ID, *plane;
  double rmsNoise, a[TFB_FILTER_LENGTH];
  long updateInterval, startPass, endPass;
  double referenceFrequency;
  double dx, dy;
  short bunchedBeamMode;
  /* internal parameters */
  short initialized, iPlane;
  long filterLength, pass0;
  double *filterOutput;
  double tReference;
  long nBunches, tReferenceSet;
  /* circular buffer for storing past readings */
  double **data;
} TFBPICKUP;

/* Transverse Feedback Driver element */
extern PARAMETER tfbdriver_param[N_TFBDRIVER_PARAMS];
typedef struct {
  char *ID;
  double strength, kickLimit, frequency, driveFrequency, clockFrequency, clockOffset, phase, RaOverQ, QLoaded;
  char *outputFile, *gainFactorFile, *gainFactorColumn;
  double gainFactor0, gainChargeScale;
  long delay;
  double a[TFB_FILTER_LENGTH];
  long updateInterval, outputInterval;
  long startPass, endPass;
  short longitudinal;
  short bunchedBeamMode; 
  /* internal parameters */
  unsigned short initialized;
#define TFBDRIVER_MAIN_INIT ((unsigned short)0x01)
#define TFBDRIVER_CLOCK_INIT ((unsigned short)0x02)
  short computeGeneratorCurrent;
  long filterLength, dataWritten, outputIndex, pass0;
  TFBPICKUP *pickup;
  SDDS_DATASET *SDDSout;
  long nBunches;
  double *gainFactor; /* for individual bunches */
  long nGainFactors;  /* should equal number of bunches */
  /* circular buffer for storing output signal */
  long maxDelay;
  double **driverSignal;
  /* variables needed for circuit model per Berenc, RF-TN-2018-005 */
  double lastV, lastVp, lastIg, lastTime, thisTime, VResidual;
  double sigma, k, omegao, omegan, omegag;
  double Zc[2];
} TFBDRIVER;


/* Longitudinal space-charge drift */
extern PARAMETER lscdrift_param[N_LSCDRIFT_PARAMS];
typedef struct {
  double length, lEffective;
  long bins;
  short smoothing, SGHalfWidth, SGOrder, interpolate, lsc, autoLEffective;
  double lowFrequencyCutoff0, lowFrequencyCutoff1;
  double highFrequencyCutoff0, highFrequencyCutoff1, radiusFactor;
  /* internal use only */
  short backtrack;
} LSCDRIFT;

/* PLanar Undulator with optional laser heater */
extern PARAMETER lsrMdltr_param[N_LSRMDLTR_PARAMS];
typedef struct {
  double length, Bu, tguGradient, tguCompFactor;
  long periods;
  char *method, *fieldExpansion;
  double accuracy; 
  long nSteps;
  double poleFactor1, poleFactor2, poleFactor3;
  double usersLaserWavelength, laserPeakPower, laserW0, laserPhase;
  double laserX0, laserY0, laserZ0, laserTilt;
  short laserM, laserN;  /* mode numbers TEM-m-n*/
  short synchRad, isr;
  short helical;
  char *timeProfileFile;
  double timeProfileOffset;
  /* internal variables */
  double laserWavelength, Ef0Laser, omega, k;
  double Escale, Bscale;
  double ku, xOffset, BuScaled, ZRayleigh;
  short fieldCode;
  long tProfilePoints;
  double *timeValue, *amplitudeValue;
  double t0;  /* reference time for arrival at center of device (v=c assumed) */
} LSRMDLTR;

/* names and storage structure for Polynomial-series map physical parameters */
extern PARAMETER polynomialSeries_param[N_POLYNOMIALSERIES_PARAMS];

typedef struct {
  double length;
  double tilt, dx, dy, dz;
  char *filename;
  /* for internal use */
  long elementInitialized;
  POLYNOMIALSERIES_DATA coord[6]; /* polynomial in (x, qx, y, qy, s, delta) to give ith coordinate */
  int32_t maxExponent[6]; /* maximum exponent of each input coordinate over all polynomials */
  double *power[6]; /* array of powers of the ith input coordinate */
} POLYNOMIALSERIES;

/* names and storage structure for quadrupole+sextupole physical parameters */
extern PARAMETER kquse_param[N_KQUSE_PARAMS];

typedef struct {
    double length, k1, k2, tilt;
    double dx, dy, dz, fse1, fse2;
    long n_kicks, nSlices;
    short integration_order, synch_rad, isr, isr1Particle, matrixTracking, expandHamiltonian;
  } KQUSE;

/* names and storage structure for dipole/quadrupole corrector physical parameters */
extern PARAMETER dqcor_param[N_DQCOR_PARAMS];

typedef struct {
  double length, k1, j1;
  long nSlices;
  short integration_order;
  double fse, tilt, pitch, yaw, dx, dy, dz;
  short malignMethod;
  double xkick, ykick;
  double xKickCalibration, yKickCalibration;
  short xSteering, ySteering;
  short synch_rad, isr, isr1Particle;
  char *dipoleSystematicMultipoles;
  double dipoleSystematicMultipoleFactor;
  char *quadrupoleSystematicMultipoles, *quadrupoleRandomMultipoles;
  double quadrupoleSystematicMultipoleFactor, quadrupoleRandomMultipoleFactor;
  short minMultipoleOrder[2], maxMultipoleOrder[2]; /* normal, skew */
  short trackingBasedMatrix;
  /* for internal use */
  double K1ReferenceFrame, xKickReferenceFrame, yKickReferenceFrame, phiReferenceFrame;
  short multipolesInitialized, totalMultipolesComputed;
  MULTIPOLE_DATA dipoleSystematicMultipoleData; 
  MULTIPOLE_DATA quadrupoleSystematicMultipoleData, quadrupoleRandomMultipoleData, totalMultipoleData;
} DQCOR;

/* names and storage structure for kick map physical parameters */
extern PARAMETER ukickmap_param[N_UKICKMAP_PARAMS];

typedef struct {
  double length, tilt, dx, dy, dz, fieldFactor, xyFactor, yaw;
  char *inputFile;
  long nKicks, periods;
  double Kreference, Kactual;
  short synchRad, isr, yawEnd, singlePeriodMap;
  /* for internal use only */
  short initialized;
  short flipSign; /* 0 for forward tracking, 1 for backward */
  long points, nx, ny;
  double *xpFactor, *ypFactor;
  double xmin, xmax, dxg;
  double ymin, ymax, dyg;
  double radiusInternal;
} UKICKMAP;  

/* names and storage structure for general kick map physical parameters */
extern PARAMETER kickmap_param[N_KICKMAP_PARAMS];

typedef struct {
  double length, tilt, dx, dy, dz, factor, xyFactor, yaw;
  char *inputFile;
  long nKicks;
  short synchRad, isr, yawEnd, singlePeriodMap;
  /* for internal use only */
  short initialized;
  short flipSign; /* 0 for forward tracking, 1 for backward */
  long points, nx, ny;
  double *xpFactor, *ypFactor;
  double xmin, xmax, dxg;
  double ymin, ymax, dyg;
} KICKMAP;  

/* names and storage structure for beam-beam element */
typedef struct {
  double charge;
  double centroid[2];
  double size[2];
  char *distribution;
  /* for internal use */
#define GAUSSIAN_BEAM_BEAM 0
#define UNIFORM_ELLIPSOIDAL_BEAM_BEAM 1
#define PARABOLIC_ELLIPSOIDAL_BEAM_BEAM 2
#define N_BEAM_BEAM_DISTRIBUTIONS 3
  short distributionCode;
} BEAMBEAM;
extern char *beamBeamDistributionOption[N_BEAM_BEAM_DISTRIBUTIONS];

/* names and storage structure for field table parameters */
extern PARAMETER ftable_param[N_FTABLE_PARAMS];

typedef struct {
  double l0, angle, l1, l2, e1, e2;
  double tilt, dx, dy, dz, factor, threshold;
  char *inputFile;
  long nKicks;
  short  verbose, simpleInput;
  /* for internal use only */
  short initialized;
  short dataIsCopy;
  double length, arcLength;
  ntuple *Bx, *By, *Bz;
} FTABLE;  


extern PARAMETER ionEffects_param[N_IONEFFECTS_PARAMS];

typedef struct {
  long disable;
  long macroIons, generationInterval;
  double span[2];              /* Actually the x, y half-span, centered on the center of the chamber */
  double poisson_span[2];      /* Span over which Poisson equation is solved */
  long n2dGridIon[2];          /* Grid size for 2d ion histogram (e.g., for Poisson solver for ion  fields) */
  double binDivisor[2];        /* x, y value. Used if bigaussian fitting wanted for field calculation */
  double rangeMultiplier[2];   /* x, y value. Used if bigaussian fitting wanted for field calculation */
  double sigmaLimitMultiplier[2]; /* x, y value. Used if bigaussian fitting wanted for field calculation */
  long startPass, endPass, passInterval;
  /* internal parameters */
  double sLocation;                /* location of the element */
  double sStart, sEnd;            /* coordinate range over which this element is modeling ions */
  double *pressure;               /* pressure for each species averaged over the effective length.
                                   * Indices line up with those used in the pressure input file.
                                   */
#define COORDINATES_PER_ION 5
  double t;                        /* time at which ion coordinates were last computed */
  double ***coordinate;            /* coordinate[i][j][k] is the kth coordinate of the jth ion of species i */
                                   /* coordinate order is (x, vx, y, vy, charge) */
  long *nIons;                     /* nIons[i] is the number of ions of species i */
  long nTotalIons;                 /* total over all processors */
  long nCoreIons;                  /* total over all processors */
  long nMin, nMax;                 /* min and max over all processors */
  double ionDelta[2];              /* delta x or y for ion histogram bins */
  double ionRange[2];              /* range in x or y for ion histogram */
  long ionBins[2];                 /* number of bins in each dimension for 1d histograms (may be changed by code) */
  double **ion2dDensity;           /* for Poisson solver */
  double **ionPotential;
  double **xKickPoisson;
  double **yKickPoisson;
  double *xyIonHistogram[2];       /* values for ion histogram independent coordinates (x, y) */
  double *ionHistogram[2];         /* charge histogram */
  double *ionHistogramFit[2];      /* fit to same */
  double ionHistogramMissed[2];    /* Charge of ions that are left out of the histogram */
  double qTotal;                   /* for normalization of ionHistogramMissed */
  /* save fit parameters between calls to speed up subsequent fitting, and for output */
  short ionFieldMethod;             /* last method used */
  unsigned short xyFitSet[2];       /* 0x01:2-function fit set, 0x02:3-function fit set */
  double xyFitParameter2[2][6];     /* [0]=x, [1]=y: fit parameters for two distributions (bigaussian or bilorentzian) */
  double xyFitParameter3[2][9];     /* [0]=x, [1]=y: fit parameters for three distributions (trigaussian or trilorentzian) */
  double xyFitResidual[2];          /* [0]=x, [1]=y: last residual */
  double ionChargeFromFit[2];
#if USE_MPI
  long nEvaluationsBest[2], nEvaluationsMin[2], nEvaluationsMax[2];
#else
  long nEvaluations[2];
#endif
} IONEFFECTS ;

/* names and storage structure for speed bump physical parameters */
extern PARAMETER speedbump_param[N_SPEEDBUMP_PARAMS];

typedef struct {
  double length, chord, dzCenter, height, position, dx, dy;
  char *insertFrom;
  short scraperConvention;
  /* Internal parameters --- See SCRAPER element for the direction codes */
  unsigned long direction;
} SPEEDBUMP;

/* aperture from section of right circular cylinder */
extern PARAMETER taperapc_param[N_TAPERAPC_PARAMS];

typedef struct {
  double length, r[2], dx, dy;
  short sticky;
  /* internal use only */
  short e1Index, e2Index;
} TAPERAPC;

/* tapered elliptical aperture */
extern PARAMETER taperape_param[N_TAPERAPE_PARAMS];

typedef struct {
  double length, a[2], b[2], dx, dy, tilt, resolution;
  short xExponent, yExponent, sticky;
  /* internal use only */
  short e1Index, e2Index;
} TAPERAPE;

/* tapered rectangular aperture */
extern PARAMETER taperapr_param[N_TAPERAPR_PARAMS];

typedef struct {
  double length, xmax[2], ymax[2], dx, dy, tilt;
  short sticky;
  /* internal use only */
  short e1Index, e2Index;
} TAPERAPR;

/* names and storage structure for space harmonic RF deflector cavity
 */
extern PARAMETER shrfdf_param[N_SHRFDF_PARAMS] ;
   
typedef struct {
  double factor, tilt, period_length, period_phase;
  double v[10], phase[10];
  long phase_reference;
  /* for internal use only */
  double t_first_particle;        
  long initialized, fiducial_seen;
} SHRFDF;

/* Transverse Feedback Pickup element */
extern PARAMETER cpickup_param[N_CPICKUP_PARAMS];
typedef struct {
  char *ID; 
  long updateInterval, startPass, endPass;
  double dx, dy;
  short bunchedBeamMode;
   
  /* internal parameters */
  short initialized;
  long nBunches, lastPass;
  double *tAverage;  /* Stores the average arrival time of each bunch */
  double *tMin; /* Time of earliest particle in each bunch */
  double *tMax; /* Time of latest particle in each bunch */
  long *npBunch; /* number of particles in each bunch */
  double **long_coords, **horz_coords, **vert_coords; /* t, x, y coordinates of each particle in each bunch */
  long **pid; /* particle ID for each particle in each bunch */
  htab **pidHashTable; /* allows kicker code to quickly find particle index by particle ID */
  long **index;  /* for storing indices for the hash table */
} CPICKUP;


/* Transverse Feedback Driver element */
extern PARAMETER ckicker_param[N_CKICKER_PARAMS];
typedef struct {
  char *ID; 
  double strength, kickLimit;
  double phase;
  long updateInterval;
  long startPass, endPass;
  short bunchedBeamMode; 
  double lambda_rad;
  short transverseMode, incoherentMode, numericalMode;
  short dtClosedOrbit;
  double angle_rad; 
  double magnification;
  double modulation_freq;
  long Nu; 

  /* internal parameters */
  short initialized;
  CPICKUP *pickup;
  double delta_t;
  long nBunches;
} CKICKER;

  /* names and storage structure for longitudinal-gradient dipoles */
extern PARAMETER lgbend_param[N_LGBEND_PARAMS];

#define N_LGBEND_FRINGE_INT 8

typedef struct {
  double length;    /* straight segment length */
  double arcLength; /* arc length of segment */
  double arcLengthStart; /* accumulated arc length at start of segment */
  double K1, K2;
  double zAccumulated;  /* at exit */
  double angle, entryX, entryAngle, exitX, exitAngle;
  short has1, has2;
  double fringeInt1K0, fringeInt1I0, fringeInt1K2, fringeInt1I1, fringeInt1K4, fringeInt1K5, fringeInt1K6,
    fringeInt1K7;
  double fringeInt2K0, fringeInt2I0, fringeInt2K2, fringeInt2I1, fringeInt2K4, fringeInt2K5, fringeInt2K6,
    fringeInt2K7;
} LGBEND_SEGMENT;

typedef struct {
  double length; /* arc length. It's actually set when the configuration file is read. */
  char *configurationFile, *apertureDataFile;
  double tilt;
  double dx, dy, dz, eyaw, epitch, etilt;
  double fse;
  long nSlices;   /* slices per segment */
  short integration_order, edgeOrder;
  short synch_rad, isr, isr1Particle, synchRadInOrdinaryMatrix, distributionBasedRadiation, includeOpeningAngle;
  short optimizeFse, compensateKn, verbose;
  char *centroidOutputFile;
  /* for internal use only: */
  double xVertex, zVertex, xEntry, zEntry, xExit, zExit; /* from configuration file */
  short initialized;
  long nSegments;
  double angle; /* total angle */
  LGBEND_SEGMENT *segment;
  double *fseOpt, *KnDelta;
  short optimized, wasFlipped;
  double predrift, postdrift;
  APERTURE_DATA *localApertureData;
  short centroidsRequested;  /* non-zero on all workers */
  SDDS_DATASET *SDDScen;     /* non-null only on myid==1 */
} LGBEND;

/* names and storage structure for longitudinal corrugated-plates wake physical parameters */
extern PARAMETER corgplates_param[N_CORGPLATES_PARAMS];
extern PARAMETER cbscat_param[N_CBSCAT_PARAMS];

typedef struct {
  double length, halfGap, period, depth, dt, tmax;
  long interpolate;          /* flag to turn on interpolation */
  long smoothing, SGHalfWidth, SGOrder;  /* flag to turn on smoothing plus control parameters */
  long change_p0, allowLongBeam;
  long rampPasses;           /* If nonzero, the number of passes over which to ramp wake up */
} CORGPLATES;

  /* END OF ELEMENT STRUCTURE DEFINITIONS */

/* macros for bending magnets */ 
long determine_bend_flags(ELEMENT_LIST *eptr, long edge1_effects, long edge2_effects);
#define SAME_BEND_PRECEDES 1 
#define SAME_BEND_FOLLOWS 2 
#define BEND_EDGE1_EFFECTS 4 
#define BEND_EDGE2_EFFECTS 8 
#define BEND_EDGE_EFFECTS (BEND_EDGE1_EFFECTS+BEND_EDGE2_EFFECTS)
#define BEND_EDGE_DETERMINED 16

#define IS_BEND(type) ((type)==T_SBEN || (type)==T_RBEN || (type)==T_CSBEND || (type)==T_KSBEND || (type)==T_CSRCSBEND || (type)==T_CCBEND || (type)==T_LGBEND)
#define IS_RADIATOR(type) ((type)==T_SBEN || (type)==T_RBEN || (type)==T_CSBEND || (type)==T_CSRCSBEND || \
                           (type)==T_QUAD || (type)==T_KQUAD || (type)==T_SEXT || (type)==T_KSEXT || \
			   (type)==T_WIGGLER || (type)==T_CWIGGLER || (type)==T_APPLE || \
                           (type)==T_HCOR || (type)==T_VCOR || (type)==T_HVCOR || (type)==T_BGGEXP || \
                           (type)==T_CCBEND || (type)==T_KICKMAP || (type)==T_LGBEND )

 
/* flags for run_awe_beam and run_bunched_beam */
#define TRACK_PREVIOUS_BUNCH 1

/* flags for do_tracking/track_beam flag word */
#define FINAL_SUMS_ONLY          0x00001UL
#define TEST_PARTICLES           0x00002UL
#define BEGIN_AT_RECIRC          0x00004UL
#define TEST_PARTICLE_LOSSES     0x00008UL
#define SILENT_RUNNING           0x00010UL
#define TIME_DEPENDENCE_OFF      0x00020UL
#define INHIBIT_FILE_OUTPUT      0x00040UL
#define LINEAR_CHROMATIC_MATRIX  0x00080UL
#define LONGITUDINAL_RING_ONLY   0x00100UL
#define FIRST_BEAM_IS_FIDUCIAL   0x00200UL
#define FIDUCIAL_BEAM_SEEN       0x00400UL
#define PRECORRECTION_BEAM       0x00800UL
#define RESTRICT_FIDUCIALIZATION 0x01000UL
#define IBS_ONLY_TRACKING        0x02000UL
#define CLOSED_ORBIT_TRACKING    0x04000UL
#define ALLOW_MPI_ABORT_TRACKING 0x08000UL
#define RESET_RF_FOR_EACH_STEP   0x10000UL
#define OPTIMIZING               0x20000UL
#define CENTROID_SUMS_ONLY       0x40000UL
#define LOSS_COORDINATES_NEEDED  0x80000UL
/* return values for get_reference_phase and check_reference_phase */
#define REF_PHASE_RETURNED 1
#define REF_PHASE_NOT_SET  2
#define REF_PHASE_NONEXISTENT 3

/* definitions for use by bunched beam routines */

#define GAUSSIAN_BEAM 0
#define HARD_EDGE_BEAM 1
#define UNIFORM_ELLIPSE 2
#define SHELL_BEAM 3
#define DYNAP_BEAM 4
#define LINE_BEAM 5
#define GAUSSIAN_HALO_BEAM 6
#define N_BEAM_TYPES 7

extern char *beam_type[N_BEAM_TYPES];

typedef struct {
    double emit, beta, alpha, eta, etap;
    double cutoff;
    long beam_type;
    double cent_posi, cent_slope;
    } TRANSVERSE;

typedef struct {
    double sigma_dp, sigma_s, dp_s_coupling;
    double beta, emit, alpha, chirp;
    double cutoff;
    long beam_type;
    double cent_s, cent_dp;
    } LONGITUDINAL;

void setup_bunched_beam_moments(
    BEAM *beam,
    NAMELIST_TEXT *nltext,
    RUN *run,
    VARY *control,
    ERRORVAL *errcon,
    OPTIM_VARIABLES *optim,
    OUTPUT_FILES *output,
    LINE_LIST *beamline,
    long n_elements,
    long save_original
    );
void zero_centroid(double **particle, int64_t n_particles, long coord);
long generate_bunch(double **particle, int64_t n_particles, TRANSVERSE *x_plane,  TRANSVERSE *y_plane,
                    LONGITUDINAL *longit, long *enforce_rms_params, long limit_invar, long symmetrize, 
                    long *haltonID, long haltonOpt, long *randomizeOrder, long elliptical_symmetry, double Po);
void set_beam_centroids(double **particle, long offset, int64_t n_particles, double cent_posi, 
    double cent_slope);
void gaussian_distribution(double **particle, int64_t n_particles, 
                           long offset, double s1, double s2, long symmetrize, long *haltonID, long haltonOpt, double limit,
                           double limit_invar, double beta, long halo);
void enforce_sigma_values(double **coord, int64_t n_part, long offset, double s1d, double s2d);
void polarizeBeam(double **part, int64_t np, POLAR *polar);

/* prototypes for alpha_matrix.c: */
extern VMATRIX *alpha_magnet_matrix(double gradient, double xgamma, long maximum_order,
    long part);
extern long alpha_magnet_tracking(double **particle, VMATRIX *M, ALPH *alpha, int64_t n_part,
    double **accepted, double P_central, double z);
 
/* prototypes for awe_beam13.c: */
/*
extern long get_particles(double ***particle, char **input, long n_input, long one_dump, long n_skip);
extern void setup_awe_beam(BEAM *beam, NAMELIST_TEXT *nltext, RUN *run, VARY *control,
    ERROR *errcon, OPTIM_VARIABLES *optim, OUTPUT_FILES *output, LINE_LIST *beamline, long n_elements);
extern long run_awe_beam(RUN *run, VARY *control, ERROR *errcon,
    LINE_LIST *beamline, long n_elements, BEAM *beam, OUTPUT_FILES *output, long flags);
extern long new_awe_beam(BEAM *beam, RUN *run, VARY *control, OUTPUT_FILES *output, long flags);
extern void finish_awe_beam(OUTPUT_FILES *output, RUN *run, VARY *control, ERROR *errcon,
    LINE_LIST *beamline, long n_elements, BEAM *beam);
*/

/*prototypes for sdds_beam.c */
void adjust_arrival_time_data(double **coord, int64_t np, double Po, long center_t, long flip_t);
 
/* prototypes for bend_matrix6.c: */
extern VMATRIX *bend_matrix(double length, double angle, double ea1, double ea2, double R1, double R2,
                            double k1, double k2, double tilt, double fint1, double fint2, double gap, 
                            double fse, double fseDipole, double fseQuadrupole, double etilt,
                            long order, long edge_order, long flags, long TRANSPORT,
                            double hkick, double vkick);
extern VMATRIX *edge_matrix(double beta, double h, double Rpole, double n, long which_edge,
                            double gK, long order, long all_terms, long TRANSPORT, double length);
extern VMATRIX *corrector_matrix(double length, double kick, double tilt, double b2, double calibration,
    long do_edges, long max_order);
extern VMATRIX *hvcorrector_matrix(double length, double xkick, double ykick, double tilt, double b2,
    double xcalibration, double ycalibration, long do_edges, long max_order);
extern VMATRIX *sbend_matrix(double t0, double h, double ha, double n,         
    double beta, double xgamma, long order);
void apply_edge_effects(
    double *x, double *xp, double *y, double *yp,
    double rho, double n, double beta, double R, double psi, long which_edge
    );
 
/* prototypes for bunched_beam12.c: */
extern void setup_bunched_beam(BEAM *beam, NAMELIST_TEXT *nltext, RUN *run, VARY *control,
                               ERRORVAL *errcon, OPTIM_VARIABLES *optim, OUTPUT_FILES *output, 
                               LINE_LIST *beamline, long n_elements,
                               long save_original);
extern void setup_bunched_beam_moments(BEAM *beam,NAMELIST_TEXT *nltext, RUN *run,
                                       VARY *control, ERRORVAL *errcon, OPTIM_VARIABLES *optim, OUTPUT_FILES *output,
                                       LINE_LIST *beamline, long n_elements, long save_original);
extern long new_bunched_beam(BEAM *beam, RUN *run, VARY *control, OUTPUT_FILES *output, long flags);
extern long run_bunched_beam(RUN *run, VARY *control, ERRORVAL *errcon, OPTIM_VARIABLES *optim, 
                             LINE_LIST *beamline, long n_elements,
                             BEAM *beam, OUTPUT_FILES *output, long flags);
extern void finish_bunched_beam(OUTPUT_FILES *output, RUN *run, VARY *control, ERRORVAL *errcon, 
                                OPTIM_VARIABLES *optim,
                                LINE_LIST *beamline, long n_elements, BEAM *beam);
extern char *brief_number(double x, char *buffer);

extern long track_beam(RUN *run, VARY *control, ERRORVAL *errcon, OPTIM_VARIABLES *optim,
                       LINE_LIST *beamline, BEAM *beam, OUTPUT_FILES *output, unsigned long flags,
                       long delayOutput, double *finalCharge);
extern BEAM *getBeamBeingTracked();
extern void do_track_beam_output(RUN *run, VARY *control, ERRORVAL *errcon, OPTIM_VARIABLES *optim,
                                 LINE_LIST *beamline, BEAM *beam, OUTPUT_FILES *output, unsigned long flags,
                                 double finalCharge);
extern void finish_output(OUTPUT_FILES *output, RUN *run, VARY *control,
                          ERRORVAL *errcon, OPTIM_VARIABLES *optim,
                          LINE_LIST *beamline, long n_elements, BEAM *beam,
                          double finalCharge);
extern void setup_output(OUTPUT_FILES *output, RUN *run, VARY *control, ERRORVAL *errcon, 
                         OPTIM_VARIABLES *optim,
                         LINE_LIST *beamline);

/* prototypes for cfgets.c: */
extern char *cfgets(char *s, long n, FILE *fpin);
extern void delete_spaces(char *s);
extern void str_to_upper_quotes(char *s);
 
/* prototypes for check_duplic.c: */
extern long check_duplic_elem(ELEMENT_LIST **elem, ELEMENT_LIST **new_elem, char *nameToCheck, long n_elems,
			      ELEMENT_LIST **existing_elem);
extern long check_duplic_line(LINE_LIST *line, char *new_line, long n_lines, long checkOnly);
 
/* prototypes for compute_centroids.c: */
extern BEAM_SUMS *allocateBeamSums(unsigned long flags, long nz);
extern void freeBeamSums(BEAM_SUMS *sums, long nz);
extern void compute_centroids(double *centroid, double **coordinates, int64_t n_part);
extern void compute_sigmas(double *emit, double *sigma, double *centroid, double **coordinates, int64_t n_part);
extern void zero_beam_sums(BEAM_SUMS *sums, long n);
#define BEAM_SUMS_SPARSE   0x0001UL
#define BEAM_SUMS_NOMINMAX 0x0002UL
#define BEAM_SUMS_EXACTEMIT 0x0004UL
extern void accumulate_beam_sums(BEAM_SUMS *sums, double **coords, int64_t n_part, double p_central, double mp_charge,
                                 double *timeValue, double tMin, double tMax,
				 int64_t startPID, int64_t endPID, unsigned long flags);
extern void accumulate_beam_sums1(BEAM_SUMS *sums,
                                  double **coord,
                                  int64_t n_part,
                                  double p_central, 
                                  double mp_charge,
                                  double *timeValue, double tMin, double tMax,
                                  int64_t startPID, int64_t endPID,
                                  unsigned long flags);
extern void copy_beam_sums(BEAM_SUMS *target, BEAM_SUMS *source);
extern long computeSliceMoments(double C[6], double S[6][6], 
			 double **part, int64_t np, 
			 double minValue, double maxValue);
double correctedEmittance(double S[6][6], double eta[4], long i1, long i2,
			  double *beta, double *alpha);
void performSliceAnalysisOutput(SLICE_OUTPUT *sliceOutput, double **particle, long particles, 
				long newPage, long step, double Po, double charge, 
				char *elementName, double elementPosition,
				long timeGiven);
void clearSliceAnalysis();
void performSliceAnalysis(SLICE_OUTPUT *sliceOutput, double **particle, long particles, 
			  double Po, double charge, long timeGiven);
void setupSliceAnalysis(NAMELIST_TEXT *nltext, RUN *run, 
			OUTPUT_FILES *output_data);

/* prototypes for compute_matrices13.c: */
extern void determineDQCorReferenceFrameTilt(DQCOR *dqcor);
extern VMATRIX *full_matrix(ELEMENT_LIST *elem, RUN *run, long order);
extern VMATRIX *append_full_matrix(ELEMENT_LIST *elem, RUN *run, VMATRIX *M0, long order);
extern VMATRIX *accumulate_matrices(ELEMENT_LIST *elem, RUN *run, VMATRIX *M0, long order, long full_matrix_only);
extern void checkMatrices(char *label, ELEMENT_LIST *elem);
extern long fill_in_matrices(ELEMENT_LIST *elem, RUN *run);
extern VMATRIX *accumulateRadiationMatrices(ELEMENT_LIST *elem, RUN *run, VMATRIX *M0, long order, long radiation, long nSlices, long sliceEtilted,
                                            short ibsUpdate);
extern long calculate_matrices(LINE_LIST *line, RUN *run);
extern VMATRIX *drift_matrix(double length, long order);
extern double computeUndulatorFieldFromModel(double gap, double fractionOfJc, double *C, double length, long poles, char *model);
extern VMATRIX *wiggler_matrix(double length, double radius, long poles, long order, long focusing);
extern void GWigSymplecticPass(double **coord, long num_particles, double pCentral,
			CWIGGLER *cwiggler, double *sigmaDelta2, long singleStep, double *ZwStart);
extern void determineCWigglerEndFlags(CWIGGLER *cwig, ELEMENT_LIST *eptr0);
extern void InitializeAPPLE(char *file, APPLE *apple);
extern void APPLE_Track(double **coord, int64_t num_particles, double pCentral,
			APPLE *apple);
extern VMATRIX *sextupole_matrix(double K2, double K1, double J1, double length, long maximum_order, double fse, double xkick, double ykick, double ffringe);
extern VMATRIX *solenoid_matrix(double length, double ks, long max_order);
extern VMATRIX *compute_matrix(ELEMENT_LIST *elem, RUN *run, VMATRIX *Mspace);
extern void startMatrixComputationTiming();
extern void reportMatrixComputationTiming();
extern VMATRIX *determineMatrix(RUN *run, ELEMENT_LIST *eptr, double *startingCoord, double *stepSize);
VMATRIX *determineMatrixHigherOrder(RUN *run, ELEMENT_LIST *eptr, double *startingCoord, double *stepSize, long order);
extern void determineRadiationMatrix(VMATRIX *Mr, RUN *run, ELEMENT_LIST *eptr, double *startingCoord, double *D, long slices, long sliceEtilted, long order);
extern void determineRadiationMatrix1(VMATRIX *Mr, RUN *run, ELEMENT_LIST *eptr, double *startingCoord, double *D, long ignoreRadiation, double *z, long iSlice);
extern void set_up_watch_point(WATCH *watch, RUN *run, long occurence, char *previousElementName, long previousElementOccurence,
			       long i_pass, ELEMENT_LIST *eptr, int64_t IDSlotsPerBunch);
extern void set_up_slice_point(SLICE_POINT *slice, RUN *run, long occurence, char *previousElementName, int64_t IDSlotsPerBunch);
void SDDS_SlicePointSetup(SLICE_POINT *slicePoint, char *command_file, char *lattice_file, char *caller, 
                          char *previousElementName);
void dump_slice_analysis(SLICE_POINT *slicePoint, long step, long pass, long n_passes, 
                         double **particle, long particles, double Po, 
                         double revolutionLength, double z, double mp_charge);
extern VMATRIX *magnification_matrix(MAGNIFY *magnif);
extern void reset_special_elements(LINE_LIST *beamline, unsigned long flags);
#define RESET_INCLUDE_RF     0x0001UL
#define RESET_INCLUDE_RANDOM 0x0002UL
#define RESET_INCLUDE_NIELEM 0x0004UL
#define RESET_INCLUDE_ALL 0xFFFFFFFFUL
extern VMATRIX *stray_field_matrix(double length, double *lB, double *gB, double theta, long order, double p_central, 
                                   void *Wi);
extern VMATRIX *rf_cavity_matrix(double length, double voltage, double frequency, double phase, double *P_central, 
                                 long order, long end1Focus, long end2Focus, char *bodyFocusModel, long sw, long change_p0, 
                                 double Preference, ELEMENT_LIST *elem, RUN *run);

/* prototypes for concat_beamline2.c: */
extern void copy_matrices1(VMATRIX *M1,  VMATRIX *M0);
extern void free_elements1(ELEMENT_LIST *elemlist);
extern void concatenate_beamline(LINE_LIST *beamline, RUN *run);
 
/* prototypes for concat_mat.c: */
extern void concat_matrices(VMATRIX *M2, VMATRIX *M1, VMATRIX *M0, unsigned long mode);
#define CONCAT_EXCLUDE_S0 0x0001UL

/* prototypes for copy_particles.c: */
extern void copy_particles(double **copy, double **original, int64_t n_particles);
 
/* prototypes for correct.c: */
extern void finish_response_output(void);
double computeMonitorReading(ELEMENT_LIST *elem, long coord, double *centroid, unsigned long flags);
#define COMPUTEMONITORREADING_TILT_0 0x0001UL
#define COMPUTEMONITORREADING_CAL_1  0x0002UL
void setMonitorCalibration(ELEMENT_LIST *elem, double calib, long coord);
double getMonitorCalibration(ELEMENT_LIST *elem, long coord);

extern long find_closed_orbit(TRAJECTORY *clorb, double clorb_acc, double clorb_acc_req, long clorb_iter, LINE_LIST *beamline, 
                              VMATRIX *M, RUN *run, double dp, long start_from_recirc, long fixed_length, 
                              double *starting_point, double iter_fraction, double iteration_multiplier, long multiplier_interval, 
                              double *deviation,
                              long track_for_orbit);
extern void rotate_xy(double *x, double *y, double angle);
extern void setupRotate3Matrix(void **Rv, double roll, double yaw, double pitch);
void rotate3(double *data, void *Rv);

/* prototypes for counter.c: */
extern long advance_values1(double *value, long n_values, long *value_index, double *initial, double *step, 
                            double **enumerated_value, long *counter, long *max_count, long *flags, long n_indices);

extern double beta_from_delta(double p, double delta);
extern long do_tracking(BEAM *beam, double **coord, int64_t n_original, long *effort, LINE_LIST *beamline, 
                        double *P_central, double **accepted, BEAM_SUMS **sums_vs_z, 
                        long *n_z_points, TRAJECTORY *traj_vs_z, RUN *run, long step,
                        unsigned long flags, long n_passes, long passOffset, SASEFEL_OUTPUT *sasefel,
			SLICE_OUTPUT *sliceAnalysis,
                        double *finalCharge, double **lostParticles, ELEMENT_LIST *startElem);
extern void recordLostParticles(BEAM *beam, double **coord, int64_t nLeft, int64_t nToTrack, long pass);
extern void resetElementTiming();
extern void reportElementTiming();

extern void setTrackingContext(char *name, long occurence, long type, char *rootname, ELEMENT_LIST *eptr, long iPass);
extern void getTrackingContext(TRACKING_CONTEXT *trackingContext);
extern TRACKING_CONTEXT trackingContext;

extern void offset_beam(double **coord, int64_t n_to_track, MALIGN *offset, double P_central);
extern void do_match_energy(double **coord, int64_t np, double *P_central, long change_beam);
extern void set_central_energy(double **coord, int64_t np, double new_energy, double *P_central);
extern void set_central_momentum(double **coord, int64_t np, double  P_new, double *P_central);
extern void center_beam(double **part, CENTER *center, int64_t np, long iPass, double P0);
void remove_correlations(double **part, REMCOR *remcor, int64_t np);
void drift_beam(double **part, int64_t np, double length, long order);
void exactDrift(double **part, int64_t np, double length);
void computeEtiltCentroidOffset(double *dcoord_etilt, double rho0, double angle, double etilt, double tilt);
void scatter_ele(double **part, int64_t np, double Po, SCATTER *scatter, long iPass);
void track_CBScat(double **part, int64_t np, double Po, CBSCAT *cb, long iPass, long iOccurence);
void store_fitpoint_twiss_parameters(MARK *fpt, char *name, long occurence, TWISS *twiss, RADIATION_INTEGRALS *radIntegrals);
void store_fitpoint_beam_parameters(MARK *fpt, char *name, long occurence, double **coord, int64_t np, double Po);
void setTrackingWedgeFunction(void (*wedgeFunc)(double **part, int64_t np, long pass, double *pCentral),
                              ELEMENT_LIST *eptr);
void setTrackingOmniWedgeFunction(void (*wedgeFunc)(double **part, int64_t np, long pass, long i_elem, long n_elem, ELEMENT_LIST *eptr, double *pCentral));
void setTrackingOmniWedgeGpuFunction(long (*wedgeFunc)(int64_t np, long pass,
                                                      long i_elem, long n_elem,
                                                      ELEMENT_LIST *eptr,
                                                      double *pCentral));
void gatherParticles(double ***coord, int64_t *nToTrack, int64_t *nLost, double ***accepted, long n_processors, int myid, double *round);
long transformBeamWithScript(SCRIPT *script, double pCentral, CHARGE *charge, BEAM *beam, double **part, 
                             int64_t np, char *mainRootname, long iPass, long driftOrder, double z, long forceSerial,
			     long occurence, long backtrack, LINE_LIST *beamline, RUN *run);
long transformBeamWithScript_s(SCRIPT *script, double pCentral, CHARGE *charge, BEAM *beam, double **part, 
			       int64_t np, char *mainRootname, long iPass, long driftOrder, double z, long occurence, long backtrack);
#ifdef USE_MPI
long transformBeamWithScript_p(SCRIPT *script, double pCentral, CHARGE *charge, BEAM *beam, double **part, 
                               int64_t np, char *mainRootname, long iPass, long driftOrder, double z, long occurence, long bracktrack);
#endif
void convertToCanonicalCoordinates(double **coord, int64_t np, double p0, long includeTimeCoordinate);
void convertFromCanonicalCoordinates(double **coord, int64_t np, double p0, long includeTimeCoordinate);

extern void track_through_kicker(double **part, int64_t np, KICKER *kicker, double p_central, long pass,
      long order);
void initializeFTable(FTABLE *ftable);
void readSimpleFtable(FTABLE *ftable);
void field_table_tracking(double **coord, int64_t np, FTABLE *ftable, double Po, RUN *run);
void rotate_coordinate(double **A, double *x, long inverse);
double choose_theta(double rho, double x0, double x1, double x2);
void track_through_mkicker(double **part, int64_t np, MKICKER *kicker, double p_central, long pass, long default_order);

extern long simple_rf_cavity(double **part, int64_t np, RFCA *rfca, double **accepted, double *P_central,
                             double zEnd);
extern long track_through_rfcw(double **part, int64_t np, RFCW *rfcw, double **accepted, double *P_central, 
                               double zEnd, RUN *run, long i_pass, CHARGE *charge);
extern long modulated_rf_cavity(double **part, int64_t np, MODRF *modrf, double P_central, double zEnd,
				long iPass, long nPasses);
extern void set_up_kicker(KICKER *kicker);
extern void add_to_particle_energy(double *coord, double timeOfFlight, double Po, double dgamma);
extern void identifyRfcaBodyFocusModel(void *pElem, long type, short *matrixMethod, short *useSRSModel, short *twFocusing1);

#define FID_MODE_LIGHT       0x0001UL
#define FID_MODE_TMEAN       0x0002UL
#define FID_MODE_FIRST       0x0004UL
#define FID_MODE_PMAX        0x0008UL
#define FID_MODE_FULLBEAM    0x1000UL
double findFiducialTime(double **part, int64_t np, double s0, double sOffset,
                        double p0, unsigned long mode);
extern unsigned long parseFiducialMode(char *mode);
void setFiducializationBunch(long b, int64_t n);
long getFiducializationPidRange(unsigned long mode, int64_t *startPID,
                                int64_t *endPID);

/* prototypes for final_props.c */
extern void SDDS_FinalOutputSetup(SDDS_TABLE *SDDS_table, char *filename, long mode, long lines_per_row,
                           char *contents, char *command_file, char *lattice_file, 
                           char **varied_quantity_name, char **varied_quantity_unit, long varied_quantities,
                           char **error_element_name, char **error_element_unit, 
                           long error_elements, long *error_element_index,
                           long *error_element_duplicates,
                           char **optimization_quantity_name, char **optimization_quantity_unit, long optimization_quantities,
                           char *caller);
extern void dump_final_properties(SDDS_TABLE *SDDS_table, BEAM_SUMS *sums,
     double *varied_quan, char *first_varied_quan_name, long n_varied_quan, long totalSteps,
     double *perturbed_quan, long *perturbed_quan_index, long perturbed_duplicates, long n_perturbed_quan,
     double *optim_quan, char *first_optim_quan_name, long n_optim_quan, double *optim_lower, double *optim_upper, 
     long step, double **particle, int64_t n_original, double p_central, VMATRIX *M,
     double finalCharge);
extern long compute_final_properties
    (double *data, BEAM_SUMS *sums, int64_t n_original, double p_central, VMATRIX *M, double **coord, 
     long step, long totalSteps, double finalCharge);
extern void rpn_store_final_properties(double *value, long number);
extern long get_final_property_index(char *name);
extern long count_final_properties(void);

extern double beam_width(double fraction, double **coord, int64_t n_part, long sort_coord);
extern double approximateBeamWidth(double fraction, double **part, int64_t nPart, long iCoord);
#if USE_MPI
extern double approximateBeamWidth_p(double fraction, double **part, int64_t nPart, long iCoord);
extern double rms_emittance_p(double **coord, long i1, long i2, long n,
			      double *S11Return, double *S12Return, double *S22Return, 
			      double *c1Return, double *c2Return, long *nTotal);
extern double rms_longitudinal_emittance_p(double **coord, int64_t n, double Po, int64_t startPID, int64_t endPID);
extern double computeAverage_p(double *data, int64_t np, MPI_Comm mpiComm);
#endif
void computeBeamTwissParameters(TWISS *twiss, double **data, long particles);
void computeBeamTwissParameters3(TWISSBEAM *twiss, double **data, long particles);
extern double rms_emittance(double **coord, long i1, long i2, long n,
                            double *S11Return, double *S12Return, double *S22Return, double *c1Return, double *c2Return);
extern double rms_longitudinal_emittance(double **coord, int64_t n, double Po, int64_t startPID, int64_t endPID);
extern double rms_norm_emittance(double **coord, long i1, long i2, int64_t ip, long n, double Po);
extern void compute_longitudinal_parameters(ONE_PLANE_PARAMETERS *bp, double **coord, long n, double Po);
extern void compute_transverse_parameters(ONE_PLANE_PARAMETERS *bp, double **coord, long n, long plane);

long binParticleCoordinate(double **hist, long *maxBins,
                           double *lower, double *upper, double *binSize, long *bins,
                           double expansionFactor,
                           double **particleCoord, long nParticles, long coordinateIndex);
#if USE_MPI
/* This is the same function as binParticleCoordinate except that only one processor needs to call it */  
long binParticleCoordinate_s(double **hist, long *maxBins,
                           double *lower, double *upper, double *binSize, long *bins,
                           double expansionFactor,
                           double **particleCoord, long nParticles, long coordinateIndex);
#endif

/* prototypes for matrix_output.c: */
void simplify_units(char *buffer, char **numer, long n_numer, char **denom, long n_denom);
void run_matrix_output(RUN *run, VARY *control, LINE_LIST *beamline);
void setup_matrix_output(NAMELIST_TEXT *nltext, RUN *run, VARY *control, LINE_LIST *beamline);

/* prototypes for twiss.c: */
VMATRIX *compute_periodic_twiss(double *betax, double *alphax, double *etax, double *etaxp,
                                double *phix, double *betay, double *alphay, double *etay, double *etayp, double *phiy,
                                ELEMENT_LIST *elem, double *clorb, RUN *run, unsigned long *unstable, double *eta2, double *eta3);
void propagate_twiss_parameters(TWISS *twiss0, double *tune, long *waists,
                                RADIATION_INTEGRALS *radIntegrals,
                                ELEMENT_LIST *elem, RUN *run, double *traj, double *finalTraj,
				double *couplingFactor);
long get_twiss_mode(long *mode, TWISS *twiss);
void compute_twiss_parameters(RUN *run, LINE_LIST *beamline, double *starting_coord, long matched, 
                              long radiation_integrals,
                              double beta_x, double alpha_x, double eta_x, double etap_x, 
                              double beta_y, double alpha_y, double eta_y, double etap_y,
                              unsigned long *unstable);
void update_twiss_parameters(RUN *run, LINE_LIST *beamline, unsigned long *unstable);
void compute_twiss_statistics(LINE_LIST *beamline, TWISS *twiss_ave, TWISS *twiss_min, TWISS *twiss_max);
void compute_twiss_percentiles(LINE_LIST *beamline, TWISS *twiss_p99, TWISS *twiss_p98, TWISS *twiss_p96);
void dump_twiss_parameters(LINE_LIST *beamline, long n_elem, 
                           double *tune, RADIATION_INTEGRALS *radIntegrals, double *chromaticity, 
                           double *dbeta, double *dalpha, double *acceptance, char **acceptanceElementName, double *alphac,
                           long final_values_only, long tune_corrected, RUN *run);
void setup_twiss_output(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline, long *do_twiss_output,
                        long default_order);
void setupTuneShiftWithAmplitude(NAMELIST_TEXT *nltext, RUN *run);
long run_twiss_output(RUN *run, LINE_LIST *beamline, double *starting_coord, long tune_corrected);
void finish_twiss_output(LINE_LIST *beamline);
void run_rf_setup(RUN *run, LINE_LIST *beamline, long writeToFile);
void setup_rf_setup(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline, long do_twiss_output, long *do_rf_setup) ;
double rfAcceptance_Fq(double q) ;
double solveForOverVoltage(double F, double q0);

void copy_doubles(double *source, double *target, long n);
void completeRadiationIntegralComputation(RADIATION_INTEGRALS *RI, LINE_LIST *beamline, double Po, double *coord);
void setupTwissAnalysisRequest(NAMELIST_TEXT *nltext, RUN *run, 
                               LINE_LIST *beamline);
#define CTFT_INCLUDE_X  0x0001UL
#define CTFT_INCLUDE_Y  0x0002UL
#define CTFT_USE_MATRIX 0x0004UL
long computeTunesFromTracking(double *tune, double *amp, VMATRIX *M, LINE_LIST *beamline, RUN *run,
			      double *startingCoord, 
			      double xAmplitude, double yAmplitude, double deltaOffset, long turns, long turnOffset,
                              double *endingCoord, double *lowerLimit, double *upperLimit,
			      long allowLosses, long nPeriods, unsigned long flags);
long computeTunesFromTrackingBatch(double *tune, double *amp, VMATRIX *M,
                                   LINE_LIST *beamline, RUN *run,
                                   double **startingCoord,
                                   const double *xAmplitude,
                                   const double *yAmplitude,
                                   const double *deltaOffset,
                                   long particles, long turns,
                                   long turnOffset, double **endingCoord,
                                   long *valid, long *survived,
                                   double *lowerLimit,
                                   double *upperLimit, long allowLosses,
                                   long nPeriods, unsigned long flags);
double adjustTuneHalfPlane(double frequency, double phase0, double phase1);
int lsf2dPolyUnweighted(double *x[2], double *y, long points, int32_t *order[2],
                        long nOrders, double *coef, double *chi, double *condition, 
                        double *diff);

/* particleTunes.c */
long setupParticleTunes(NAMELIST_TEXT *nltext, RUN *run, VARY *control, PARTICLE_TUNES *ptunes);
void accumulateParticleTuneData(double **coord, int64_t np, long pass, PARTICLE_TUNES *pTunes);
void outputParticleTunes(PARTICLE_TUNES *ptunes, long pass);
void finishParticleTunes(PARTICLE_TUNES *ptunes);

/* tune.c */
double *scanNumberList(char *list, long *nFound);

/* chrom.c */
void computeChromaticities(double *chromx, double *chromy, 
                           double *dbetax, double *dbetay,
                           double *dalphax, double *dalphay,
                           TWISS *twiss0, TWISS *twiss1, VMATRIX *M);

/* frequencyMap.c */
void setupFrequencyMap(NAMELIST_TEXT *nltext, RUN *run, VARY *control);
long doFrequencyMap(RUN *run, VARY *control, double *referenceCoord,
                    ERRORVAL *errcon, LINE_LIST *beamline);
void finishFrequencyMap();

/* tune footprint */
/*** Any changes to this structure require changes to setupTuneFootprintDataTypes in tuneFootprint.c ****/
typedef struct {
  double chromaticTuneRange[2]; /* range of tunes from momentum variation */
  double deltaRange[3]; /* span of delta (x, y, minimum of both) */
  double amplitudeTuneRange[2]; /* range of tunes from amplitude variation */
  double positionRange[2]; /* span of x and y */
  double chromaticDiffusionMaximum, amplitudeDiffusionMaximum;
  double xyArea;
  double nuxChromLimit[2], nuyChromLimit[2];
  double nuxAmpLimit[2], nuyAmpLimit[2];
  double chrom1[2];
} TUNE_FOOTPRINTS;
long setupTuneFootprint(NAMELIST_TEXT *nltext, RUN *run, VARY *control);
long doTuneFootprint(RUN *run, VARY *control, double *referenceCoord,
                     LINE_LIST *beamline, TUNE_FOOTPRINTS *tfOutput);
void finishTuneFootprint();
void outputTuneFootprint(VARY *control);

/* chaosMap.c */
void setupChaosMap(NAMELIST_TEXT *nltext, RUN *run, VARY *control);
long doChaosMap(RUN *run, VARY *control, double *referenceCoord,
                    ERRORVAL *errcon, LINE_LIST *beamline);
void finishChaosMap();

/* prototypes for elegant.c: */
extern void getRunControlContext(VARY *context);
extern void getRunSetupContext (RUN *context);
extern char *compose_filename(char *templateString, char *root_name);
extern char *compose_filename_occurence(char *templateString, char *root_name, long occurence);
#if USE_MPI
extern char *compose_filename_per_processor(char *templateString, char *root_name);
#endif
extern double find_beam_p_central(char *input);
void center_beam_on_coords(double **particle, int64_t n_part, double *coord, long center_momentum_also);
void offset_beam_by_coords(double **part, int64_t np, double *coord, long offset_dp);
void link_date(void);
void check_heap(void);
void do_print_dictionary(char *filename, long latex_form, long SDDS_form);
void print_dictionary_entry(FILE *fp, long type, long latex_form, long SDDS_form);
void bombElegant(const char *error, const char *usage);
void bombTracking(const char *error);
void bombElegantVA(char *ptemplate, ...);
void exitElegant(long status);
void printMessageAndTime(FILE *fp, char *message);

/* prototypes for error.c: */
extern void error_setup(ERRORVAL *errcon, NAMELIST_TEXT *nltext, RUN *run_cond, LINE_LIST *beamline);
extern void add_error_element(ERRORVAL *errcon, NAMELIST_TEXT *nltext, LINE_LIST *beamline);
extern double parameter_value(char *pname, long elem_type, long param, LINE_LIST *beamline);
extern double perturbation(double xamplitude, double xcutoff, long xerror_type, long sampleIndex, ERROR_SAMPLES *errorSamples);
 
/* prototypes for extend_list.c: */
extern void extend_line_list(LINE_LIST **lptr);
extern void extend_elem_list(ELEMENT_LIST **eptr);
 
/* prototypes for get_beamline5.c: */
extern void show_elem(ELEMENT_LIST *eptr, long type);
extern LINE_LIST *get_beamline(char *madfile, char *use_beamline, double p_central, long echo, long backtrack,
                               CHANGE_START_SPEC *css, CHANGE_END_SPEC *ces);
extern double compute_end_positions(LINE_LIST *lptr) ;
extern void show_elem(ELEMENT_LIST *eptr, long type);
extern void free_elements(ELEMENT_LIST *elemlist);
extern void free_beamlines(LINE_LIST *beamline);
extern void do_save_lattice(NAMELIST_TEXT *nl, RUN *run, LINE_LIST *beamline);
extern void print_with_continuation(FILE *fp, char *s, long endcol, char separator, char *continString);
extern void change_defined_parameter_values(char **elem_name, long *param_number, long *type, double *value, long n_elems);
extern void change_defined_parameter_divopt(char *elem_name, long param, long elem_type, 
                                     double value, char *valueString, unsigned long mode, 
                                     long checkDiv);
extern void change_defined_parameter(char *elem_name, long param_number, long type, double value, char *valueString, unsigned long mode);
extern void change_used_parameter_divopt(LINE_LIST *beamline, char *elem_name, long param, long elem_type, 
                                  double value, char *valueString, unsigned long mode, 
                                  long checkDiv);
extern void change_used_parameter(LINE_LIST *beamline, char *elem_name, long param, long elem_type, 
                           double value, char *valueString, unsigned long mode);

extern void delete_matrix_data(LINE_LIST *beamline);
extern void create_load_hash(ELEMENT_LIST *elem);

extern void add_element(ELEMENT_LIST *elem0, ELEMENT_LIST *elem1);
extern ELEMENT_LIST *rm_element(ELEMENT_LIST *elem); 
extern ELEMENT_LIST *replace_element(ELEMENT_LIST *elem0, ELEMENT_LIST *elem1); 

/* prototypes for limit_amplitudes4.c: */
extern long rectangular_collimator(double **initial, RCOL *rcol, int64_t np, double **accepted, double z, double P_central,
                                     ELEMENT_LIST *eptr);
extern long limit_amplitudes(double **coord, double xmax, double ymax, int64_t np, double **accepted, double z, double P_central,
                             long extrapolate_z, long openCode, ELEMENT_LIST *eptr);
extern long elliptical_collimator(double **initial, ECOL *ecol, int64_t np, double **accepted, double z, double P_central,
                                    ELEMENT_LIST *eptr);
extern long elimit_amplitudes(double **coord, double xmax, double ymax, int64_t np, double **accepted, double z,
                              double P_central, long extrapolate_z, long openCode, long exponent, long yexponent, ELEMENT_LIST *eptr);
extern long remove_outlier_particles(double **initial, CLEAN *clean, int64_t np, 
				     double **accepted, double z, double Po);  
extern long beam_scraper(double **initial, SCRAPER *scraper, int64_t np, double **accepted, double z,
                         double P_central, ELEMENT_LIST *eptr);
unsigned long interpretScraperDirection(char *insertFrom, long oldDir);
extern long track_through_pfilter(double **initial, PFILTER *pfilter, int64_t np, 
                                  double **accepted, double z, double Po);
long removeInvalidParticles(double **coord, int64_t np, double **accepted,
                            double z, double Po);
extern long determineOpenSideCode(char *openSide);
long interpolateApertureData(double z, APERTURE_DATA *apData,
                             double *xCenter, double *yCenter, double *xSize, double *ySize);
long imposeApertureData(double **coord, int64_t np, double **accepted,
                        double z, double Po, APERTURE_DATA *apData, ELEMENT_LIST *eptr);
void resetApertureData(APERTURE_DATA *apData);
long track_through_speedbump(double **initial, SPEEDBUMP *speedbump, int64_t np, double **accepted, double z,
                             double Po, ELEMENT_LIST *eptr);
int pointIsInsideContour(double x0, double y0, double *x, double *y, int64_t n, double *center, double theta);
void initializeApContour(APCONTOUR *apcontour);
long trackThroughApContour(double **initial, APCONTOUR *apcontour, int64_t np, double **accepted, double z,
                           double Po, ELEMENT_LIST *eptr);
long imposeApContour(double **coord, APCONTOUR *apcontour, int64_t np, double **accepted, double z,
                     double Po, ELEMENT_LIST *eptr);
long checkApContour(double x, double y, APCONTOUR *apcontour, ELEMENT_LIST *eptr);
long trackThroughTaperApCirc(double **initial, TAPERAPC *taperApC, int64_t np, double **accepted, double z,
                             double Po, ELEMENT_LIST *eptr);
long trackThroughTaperApElliptical(double **initial, TAPERAPE *taperApE, int64_t np, double **accepted, double zStartElem,
                                   double Po, ELEMENT_LIST *eptr);
long trackThroughTaperApRectangular(double **initial, TAPERAPR *taperApR, int64_t np, double **accepted, double zStartElem,
                                    double Po, ELEMENT_LIST *eptr);
double linear_interpolation(double *y, double *t, long n, double t0, long i);
long find_nearby_array_entry(double *entry, long n, double key);
 
/* prototypes for kick_sbend.c: */
long track_through_kick_sbend(double **part, int64_t n_part, KSBEND *ksbend, double p_error, double Po,
    double **accepted, double z_start);
void bend_edge_kicks(double *x, double *xp, double *y, double *yp, double rho, double n, double beta, 
    double psi,  long which_edge);

/* prototypes for mad_parse4.c: */
extern long is_simple(char *s);
extern void copy_p_elem(char *target, char *source, long type);
extern void fill_line(LINE_LIST *line, long nl, ELEMENT_LIST *elem, long ne, char *s);
extern ELEMENT_LIST *expand_line(ELEMENT_LIST *leptr, LINE_LIST *lptr,
    char *s, LINE_LIST *line, long nl, ELEMENT_LIST *elem, long ne, char *part_of);
extern long is_simple(char *s);
extern void fill_elem(ELEMENT_LIST *eptr, char *s, long type, FILE *fp_input);
extern long expand_phys(ELEMENT_LIST *leptr, char *entity, ELEMENT_LIST *elem_list,     
    long ne, LINE_LIST *line_list, long nl, long reverse, long multiplier, char *part_of);
extern void copy_element(ELEMENT_LIST *e1, ELEMENT_LIST *e2, long reverse, long division,
                         long divisions, char *editCmd);
extern void copy_named_element(ELEMENT_LIST *eptr, char *s, ELEMENT_LIST *elem);
extern long copy_line(ELEMENT_LIST *e1, ELEMENT_LIST *e2, long ne, long reverse, char *part_of, char *editCmd);
extern void modify_for_backtracking(ELEMENT_LIST *eptr);
extern long tell_type(char *s, ELEMENT_LIST *elem);
extern char *get_param_name(char *s);
extern char *find_param(char *s, char *param);
extern void unknown_parameter(char *parameter, char *element, char *type_name, char *caller);
extern void parse_element(char *p_elem, PARAMETER *parameter, long n_params,
    char *string, ELEMENT_LIST *eptr, char *type_name);
extern void parse_pepper_pot(PEPPOT *peppot, FILE *fp, char *name);
extern long set_max_name_length(long length);
void resetElementToDefaults(char *p_elem, long type);
 
/* prototypes for malign_mat.c: */
extern void misalign_matrix(VMATRIX *M, double dx, double dy, double dz, 
                            double pitch, double yaw, double tilt,
                            double designTilt, double thetaBend, double length, 
                            short method);
extern VMATRIX *misalignment_matrix(MALIGN *malign, long order);
extern void offsetBeamCoordinatesForMisalignment(double **part, int64_t np, double dx, double dy, double dz);
extern void offsetParticlesForMisalignment(long mode, double **coord, int64_t np, double dx, double dy, 
                                           double dz,  double ax, double ay, double az,
                                           double tilt, double thetaBend, double length,
                                           short face);
extern void offsetParticlesForEntranceCenteredMisalignmentExact(double **coord, int64_t np, double dx, double dy, 
                                                                double dz,  double ax, double ay, double az,
                                                                double tilt, double thetaBend, double length,
                                                                short face);
extern void offsetParticlesForBodyCenteredMisalignmentExact(double **coord, int64_t np, double dx0, double dy0, double dz0,
                                                            double ax0, double ay0, double az0,
                                                            double tilt, double thetaBend, double length,
                                                            short face);
extern void offsetParticlesForEntranceCenteredMisalignmentLinearized(VMATRIX **VM, double **coord, int64_t np, 
                                                              double dx, double dy, double dz,
                                                              double ax, double ay, double az, double tilt,
                                                              double thetaBend, double length, short face);
extern void offsetParticlesForBodyCenteredMisalignmentLinearized(VMATRIX **VM, double **coord, int64_t np, 
                                                          double dx, double dy, double dz,
                                                          double ax, double ay, double az, double tilt,
                                                          double thetaBend, double length, short face);
/* prototypes for matrix7.c: */
extern void print_matrices(FILE *fp, char *string, VMATRIX *M, double suppressBelow);
extern void print_matrices1(FILE *fp, char *string, char *format, VMATRIX *M, double suppressBelow);
extern void initialize_matrices(VMATRIX *M, long order);
extern void null_matrices(VMATRIX *M, unsigned long flags);
extern void remove_s_dependent_matrix_elements(VMATRIX *M, long order);
/* flags for null_matrices */
#define EXCLUDE_C  0x01
#define EXCLUDE_R  0x02
#define EXCLUDE_T  0x04
#define EXCLUDE_Q  0x08
#define SET_UNIT_R 0x10
extern void track_particles(double **final, VMATRIX *M, double  **initial, int64_t n_part);
extern void free_matrices(VMATRIX *M);
// Not used
// extern void free_nonlinear_matrices(VMATRIX *M);
extern void free_matrices_above_order(VMATRIX *M, long order);
extern void set_matrix_pointers(double **C, double ***R, double ****T, double *****Q, VMATRIX *M);
extern long read_matrices(VMATRIX *M, char *filename, FILE *fp);
extern void filter_matrices(VMATRIX *M, double threshold);
extern void random_matrices(VMATRIX *M, double C0, double R0, double T0, double Q0);
extern void copy_matrices(VMATRIX *M1, VMATRIX *M0);
extern long check_matrix(VMATRIX *M, char *comment);
extern long reverse_matrix(VMATRIX *Mr, VMATRIX *M);
extern double checkSymplecticity(VMATRIX *Mv, short canonical);
extern void checkSymplecticity3rdOrder(VMATRIX *M, double meanMax[3][2]);

/* prototypes for motion4.c: */
extern long motion(double **part, int64_t n_part, void *field, long field_type, double *P_central, double *dgamma,
    double *dP, double **accepted, double z_start);
 
/* prototypes for multipole.c: */
typedef struct {
  double xCen, yCen, xMax, yMax;
  double reverseTilt;
  double reverseTiltCS[2]; /* cos(tilt), sin(tilt): used to undo coordinate tilts when those are done for computational reasons */
  short elliptical, present, xExponent, yExponent;
  short openSide;
  APCONTOUR *apContour;
  APERTURE_DATA *localAperture;
  ELEMENT_LIST *eptr;
} MULT_APERTURE_DATA;
extern void setupMultApertureData(MULT_APERTURE_DATA *apertureData, double reverseTilt, APCONTOUR *apContour, MAXAMP *maxamp, 
                                  APERTURE_DATA *apFileData, APERTURE_DATA *localApFileData, double zPosition, ELEMENT_LIST *eptr);
extern long checkMultAperture(double x, double y, double sLocal, MULT_APERTURE_DATA *apData);
extern long multipole_tracking(double **particle, int64_t n_part, MULT *multipole, double p_error, double Po, double **accepted, double z_start);
extern long multipole_tracking2(double **particle, int64_t n_part, ELEMENT_LIST *elem, double p_error, 
                                double Po, double **accepted, double z_start,
                                MAXAMP *maxamp, APCONTOUR *apcontour, APERTURE_DATA *apData, double *sigmaDelta2,
                                long iSlice);
extern long fmultipole_tracking(double **particle,  int64_t n_part, FMULT *multipole,
                                double p_error, double Po, double **accepted, double z_start);
int integrate_kick_multipole_ordn(double *coord, double dx, double dy, double xkick, double ykick,
                                  double Po, double rad_coef, double isr_coef,
                                  long *order, double *KnL,  short *skew,
                                  long n_parts, int64_t i_part, double drift,
                                  long integration_order,
                                  MULTIPOLE_DATA *multData, MULTIPOLE_DATA *edgeMultData, MULTIPOLE_DATA *steeringMultData,
                                  MULT_APERTURE_DATA *apData, double *dzLoss, double *sigmaDelta2,
				  long radial, 
                                  double refTilt /* used for obstruction evaluation only */);
long findMaximumOrder(long order, long order2, MULTIPOLE_DATA *edgeMultData, MULTIPOLE_DATA *steeringMultData, 
                      MULTIPOLE_DATA *multData);

/* prototypes for polynomialseries.c: */
extern long polynomialSeries_tracking(double **particle,  int64_t n_part, POLYNOMIALSERIES *polynomialSeries,
                                double p_error, double Po, double **accepted, double z_start);

/* prototypes for kickmap.c */
long trackKickMap(double **particle, double **accepted, long nParticles, double pRef, KICKMAP *map,
                  double zStart, double *sigmaDelta2);

/* prototypes for ukickmap.c */
long trackUndulatorKickMap(double **particle, double **accepted, long nParticles, double pRef, UKICKMAP *map,
                  double zStart);

/* prototypes for output_magnets.c: */
extern void output_magnets(char *filename, char *line_name, LINE_LIST *beamline);
extern void output_profile(char *filename, char *line_name, LINE_LIST *beamline);
 
/* prototypes for pepper_pot2.c: */
extern long pepper_pot_plate(double **initial, PEPPOT *peppot, int64_t np, double **accepted);
 
/* prototypes for phase_reference.c: */
extern long get_phase_reference(double *phase, long phase_ref_number);
extern long set_phase_reference(long phase_ref_number, double phase);
extern void delete_phase_references(void);
extern long unused_phase_reference(void);
extern double get_reference_phase(long phase_ref, double phase0);

/* prototypes for print_line2.c: */
extern void print_line(FILE *fp, LINE_LIST *lptr);
extern void print_elem_list(FILE *fp, ELEMENT_LIST *eptr);
extern void print_elem(FILE *fp, ELEMENT_LIST *eptr);
extern void print_elem_names(FILE *fp, ELEMENT_LIST *eptr, long width);

/* prototypes for quad_matrix3.c: */
VMATRIX *quadrupole_matrix(double K1, double lHC, long maximum_order,
                           double fse,
                           double xkick, double ykick,
                           double edge1_effects, double edge2_effects,
                           char *fringeType, double ffringe, double lEffective,
                           double *fringeIntM, double *fringeIntP,
			   short radial
                           );
VMATRIX *dqcor_matrix(double K1, double L, long maximum_order, double fse, double xkick, double ykick);
void computeQuadSteeringStrengths(double *K0, double *J0, double xkick0, double ykick0, double L, double K1);
extern VMATRIX *quad_fringe(double l, double ko, long order, long reverse, double fse);
extern void qfringe_R_matrix(double *R11, double *R21, double *R12, double *R22, double dk_dz, double l);
extern void qfringe_T_matrix(double *T116, double *T126, double *T216, double *T226,
    double *T511, double *T512, double *T522, double dk_dz, double l, long reverse);
extern VMATRIX *qfringe_matrix(double K1, double l, long direction, long order, double fse);
extern VMATRIX *quse_matrix(double K1, double K2, double l, long maximum_order, double fse1, double fse2);

/* prototypes for fringe.c */
void quadFringe(double **coord, int64_t np, double K1, double *fringeIntM, double *fringeIntP, 
                int backtrack, int inFringe, int higherOrder,
                int linearFlag, double nonlinearFactor);
void dipoleFringe(double *vec, double h, long inFringe, long higherOrder, double K1);
VMATRIX *quadFringeMatrix(VMATRIX *Mu, double K1, long backtrack, long inFringe, double *fringeIntM, double *fringeIntP);
//VMATRIX *quadPartialFringeMatrix(VMATRIX *M, double K1, long inFringe, double *fringeInt, long part);

/* prototypes for tilt_matrices.c: */
extern void tilt_matrices0(VMATRIX *M, double tilt);
extern void tilt_matrices(VMATRIX *M, double tilt);
extern VMATRIX *rotation_matrix(double tilt);
extern void rotateCoordinatesForMisalignment(double *coord, double angle);
extern void rotateBeamCoordinatesForMisalignment(double **part, int64_t np, double angle);
void pitch_matrices(VMATRIX *M, double pitch);
void yaw_matrices(VMATRIX *M, double yaw);

/* prototypes for track_ramp.c: */
extern void track_through_ramped_deflector(double **final, RMDF *ramp_param, double **initial, int64_t n_particles, double pc_central);
extern void track_through_rftm110_deflector(double **final, RFTM110 *rf_param,
					    double **initial, int64_t n_particles,
					    double pc_central, double L_central, double zEnd,
					    long pass);
 
/* prototypes for track_rf2.c: */
extern void track_through_rf_deflector(double **final, RFDF *rf_param,
                                       double **initial, int64_t n_particles,
                                       double pc_central, double L_central, double zEnd,
                                       long pass);
extern void track_through_multipole_deflector(double **final, MRFDF *rf_param, double **initial,
                                              int64_t n_particles, double pc_central, long pass);

/* prototypes for vary4.c: */
extern void vary_setup(VARY *_control, NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline);
extern void add_varied_element(VARY *_control, NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline);
extern long vary_beamline(VARY *_control, ERRORVAL *errcon, RUN *run, LINE_LIST *beamline);
extern long perturb_beamline(VARY *_control, ERRORVAL *errcon, RUN *run, LINE_LIST *beamline);
extern ELEMENT_LIST *find_element(char *elem_name,  ELEMENT_LIST **context, ELEMENT_LIST *elem);
extern ELEMENT_LIST *find_element_hash(char *elem_name, long occurence,  ELEMENT_LIST **context,  ELEMENT_LIST *elem);
extern ELEMENT_LIST *wfind_element(char *elem_name,  ELEMENT_LIST **context, ELEMENT_LIST *elem);
ELEMENT_LIST *find_element_index(char *elem_name,  ELEMENT_LIST **context,  ELEMENT_LIST *elem, long *index);
extern long confirm_parameter(char *item_name, long type);
extern void set_element_flags(LINE_LIST *beamline, char **elem_name, long *elem_perturb_flags,
    long *type, long *param, long n_elems,
    long pflag, long mflag, long overwrite, long permit_flags);
extern void assert_parameter_values(char **elem_name, long *param_number, long *type, double *value, long n_elems,
    LINE_LIST *beamline);
long get_parameter_value(double *value, char *elem_name, long param_number, long type, LINE_LIST *beamline);
extern void assert_perturbations(char **elem_name, long *param_number, long *type, long n_elems,
    double *amplitude, double *cutoff, long *error_type, double *perturb, long *elem_perturb_flags,
    long *bind_number, long *boundTo, double *sMin, double *sMax, long *sampleIndex, ERROR_SAMPLES *errorSamples,
    FILE *fp_log, long step, LINE_LIST *beamline, long permit_flags);
extern long compute_changed_matrices(LINE_LIST *beamline, RUN *run);

/* prototypes for routines in optimize.c */

void do_optimization_setup(OPTIMIZATION_DATA *_optimize, NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline);
#if USE_MPI
void do_parallel_optimization_setup(OPTIMIZATION_DATA *optimization_data, NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline);
  void SDDS_PopulationSetup(char *population_log, SDDS_TABLE *popLogPtr, OPTIM_VARIABLES *optim, OPTIM_COVARIABLES *co_optim);
#endif
void add_optimization_variable(OPTIMIZATION_DATA *_optimize, NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline);
void add_optimization_term(OPTIMIZATION_DATA *optimization_data, NAMELIST_TEXT *nltext, RUN *run,
                           LINE_LIST *beamline);
void add_optimization_constraint(OPTIMIZATION_DATA *_optimize, NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline);
void summarize_optimization_setup(OPTIMIZATION_DATA *_optimize);
void do_optimize(NAMELIST_TEXT *nltext, RUN *run1, VARY *control1, ERRORVAL *error1, LINE_LIST *beamline1, 
                 BEAM *beam1, OUTPUT_FILES *output1, OPTIMIZATION_DATA *optimization_data1, 
                 void *chromData, long beam_type1, long doClosedOrbit, long doChromCorr,
                 void *correct, long correctMode, void *tuneData, long doTuneCorr, long doFindAperture,
                 long doResponse);
void add_optimization_covariable(OPTIMIZATION_DATA *_optimize, NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline);
double optimization_function(double *values, long *invalid);
void do_set_reference_particle_output(OPTIMIZATION_DATA *optimization_data, NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline);
#if USE_MPI
  long geneticMin(double *yReturn, double *xGuess, double *xLowerLimit, double *xUpperLimit, double *xStep,
		  long dimensions, double target, double (*func)(double *x, long *invalid), long maxIterations,
		  long maxNoChange, long populationSize, long printFrequency, long pirntAllPopulations,
		  char *populations_log, SDDS_TABLE *logPtr, long verbose, long crossoverType,
		  OPTIM_VARIABLES *optim, OPTIM_COVARIABLES *co_optim);
  
  long swarmMin(double *yReturn, double *xGuess, double *xLowerLimit, double *xUpperLimit, double *xStep, long dimensions, 
		double target,double (*func)(double *x, long *invalid), long populationSize, long n_iterations, long max_iterations);
#endif
/* prototype for sample.c */
long sample_particles(double **initial, SAMPLE *samp, int64_t np, double **accepted, double z, double p0);


/* prototype for run_rpnexpr.c */
void run_rpn_expression(NAMELIST_TEXT *nltext);

/* prototypes for trace.c */
void process_trace_request(NAMELIST_TEXT *nltext);
void log_entry(const char *routine);
void log_exit(const char *routine);
void show_traceback(FILE *fp);

/* flag word for trace mode */
extern long trace_mode;
#define TRACE_ENTRY 1
#define TRACE_HEAP_VERIFY 2
#define TRACE_MEMORY_LEVEL 4
#define TRACEBACK_ON 8

/* global particle ID counter */
extern long particleID;

/* prototypes for lorentz.c */
long lorentz(double **part, int64_t n_part, void *field, long field_type, double P_central, double **accepted, 
             MAXAMP *maxamp, APCONTOUR *apcontour, APERTURE_DATA *apData);
void lorentz_report(void);

/* prototypes for kick_poly.c */
long polynomial_kicks(double **particle, int64_t n_part, KPOLY *kpoly, double p_error, double Po,
    double **accepted, double z_start);
long polynomial_hamiltonian(double **particle,  int64_t n_part, HKPOLY *hkpoly, double p_error, double Po,
                            double **accepted, double z_start);

/* prototypes for ramp_p.c */
void ramp_momentum(double **coord, int64_t np, RAMPP *rampp, double *P_central, long pass);

/* prototypes for ramped_rfca.c */
long ramped_rf_cavity(double **part, int64_t np, RAMPRF *ramprf, double P_central, 
                      double L_central, double z_cavity, long pass);

/* prototypes for closed_orbit.c */
extern void dump_closed_orbit(TRAJECTORY *traj, long n_elems, long step, double *deviation, long badOrbit);
void finish_clorb_output(void);
#define CLOSED_ORBIT_OUTPUT 0x01UL
#define CLOSED_ORBIT_IGNORE_BEAM 0x02UL
long run_closed_orbit(RUN *run, LINE_LIST *beamline, double *starting_coord, BEAM *beam, unsigned long flags);
long setup_closed_orbit(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline);
long checkChangeT(LINE_LIST *beamline);

/* prototypes for aperture_search.c */
void setup_aperture_search(NAMELIST_TEXT *nltext, RUN *run, VARY *control, long *optimizationMode);
long do_aperture_search(RUN *run, VARY *control, double *referenceCoord, 
			ERRORVAL *errcon, LINE_LIST *beamline, double *apertureReturn);
long do_aperture_search_mp(RUN *run, VARY *control, double *referenceCoord,
			   ERRORVAL *errcon, LINE_LIST *beamline);
long do_aperture_search_sp(RUN *run, VARY *control, double *referenceCoord, 
			   ERRORVAL *errcon, LINE_LIST *beamline);
long do_aperture_search_line(RUN *run, VARY *control, double *referenceCoord,
			     ERRORVAL *errcon, LINE_LIST *beamline,
			     long number, double *apertureReturn);
void finish_aperture_search(RUN *run, VARY *control,  ERRORVAL *errcon, LINE_LIST *beamline);
double trimApertureSearchResult(long lines, double *xLimit, double *yLimit, double *dxFactor, double *dyFactor, long fullPlane);

/* prototypes for analyze.c */
void setup_transport_analysis(NAMELIST_TEXT *nltext, RUN *run, VARY *control, ERRORVAL *errcon);
void do_transport_analysis(RUN *run, VARY *control, ERRORVAL *errcon, LINE_LIST *beamline, double *orbit);
void finish_transport_analysis(RUN *run, VARY *control, ERRORVAL *errcon, LINE_LIST *beamline);

/* prototypes for link_elements.c */
void element_link_control(ELEMENT_LINKS *links, NAMELIST_TEXT *nltext, RUN *run_cond, LINE_LIST *beamline);
void add_element_links(ELEMENT_LINKS *links, NAMELIST_TEXT *nltext, LINE_LIST *beamline);
long assert_element_links(ELEMENT_LINKS *links, RUN *run_cond, LINE_LIST *beamline, long flags);
void reset_element_links(ELEMENT_LINKS *links, RUN *run_cond, LINE_LIST *beamline);
void rebaseline_element_links(ELEMENT_LINKS *links, RUN *run, LINE_LIST *beamline);

long track_through_matter(double **part, int64_t np, long iPass, MATTER *matter, double Po, double **accepted, double z0);

void track_through_rfmode(double **part, int64_t np, RFMODE *rfmode, double Po,
    char *element_name, double element_z, long pass, long n_passes, CHARGE *charge);
void set_up_rfmode(RFMODE *rfmode, char *element_name, double element_z, long n_passes, RUN *run, int64_t n_particles,
                   double Po, double Lo);

void track_through_trfmode(double **part, int64_t np, TRFMODE *trfmode, double Po,
    char *element_name, double element_z, long pass, long n_passes, CHARGE *charge);
void set_up_trfmode(TRFMODE *trfmode, char *element_name, double element_z, 
                    long n_passes, RUN *run, int64_t n_particles);
void track_through_zlongit(double **part, int64_t np, ZLONGIT *zlongit, double Po, RUN *run, long i_pass,
                           CHARGE *charge);
void applyLowPassFilterToImpedance(double *Z, long nfreq, double cutoff0, double cutoff1);
void track_through_lscdrift(double **part, int64_t np, LSCDRIFT *lscdrift, double Po, CHARGE *charge);
long checkPointSpacing(double *x, long n, double tolerance);
void track_through_ztransverse(double **part, int64_t np, ZTRANSVERSE *ztransverse,
                               double Po, RUN *run, long i_pass,
                               CHARGE *charge);
void track_through_impedance(double **part0, long np0, IMPEDANCE *imp,
                             double Po, RUN *run, long i_pass, CHARGE *charge);
void optimizeBinSettingsForImpedance(double timeSpan, double freq, double Q,
                                     double *binSize, long *nBins, long maxBins);
void convolveArrays(double *output, long outputs, 
                    double *a1, long n1,
                    double *a2, long n2, long di2);
void applyLongitudinalWakeKicks(double **part, double *time, long *pbin, int64_t np, double Po,
                                double *Vtime, long nb, double tmin, double dt,
                                long interpolate);
void applyTransverseWakeKicks(double **part, double *time, double *pz, long *pbin, int64_t np,
                              double Po, long plane,
                              double *Vtime, long nb, double tmin, double dt, 
                              long interpolate, long exponent);
void track_through_wake(double **part, int64_t np, WAKE *wakeData, double *Po,
                        RUN *run, long i_pass, CHARGE *charge);
void track_through_corgpipe(double **part, int64_t np, CORGPIPE *corgpipe, double *Pcentral, 
                            RUN *run, long i_pass, CHARGE *charge);
void track_through_corgplates(double **part, int64_t np, CORGPLATES *corgplates, double *Pcentral, 
                            RUN *run, long i_pass, CHARGE *charge);
void track_through_cwake(double **part0, long np0, CWAKE *cw,
                         double *PoInput, RUN *run, long i_pass, CHARGE *charge);
void track_through_trwake(double **part, int64_t np, TRWAKE *wakeData, double Po,
                          RUN *run, long i_pass, CHARGE *charge);
void track_through_lrwake(double **part, int64_t np, LRWAKE *wakeData, double *Po,
			  RUN *run, long i_pass, CHARGE *charge);
void index_bunch_assignments(double **part, int64_t np, long idSlotsPerBunch, double P0, double **time, long **ibParticle, int64_t ***ipBucket, int64_t **npBucket, long *nBuckets,
                                  long lastNBuckets);
void free_bunch_index_memory(double *time0, long *ibParticle, int64_t **ipBucket, int64_t *npBucket, long nBuckets);

void addLSCKick(double **part, int64_t np, LSCKICK *LSC, double Po, CHARGE *charge, 
                double lengthScale, double dgammaOverGamma);
void computeTimeCoordinatesOnly(double *time, double Po, double **part, int64_t np);
double computeTimeCoordinates(double *time, double Po, double **part, int64_t np);
void computeDistanceCoordinates(double *time, double Po, double **part, int64_t np);
/* tracking-clock register (simple_rfca.c): macro-scale absolute time in seconds
   that is not carried in part[][4]; true time = part[][4]/(c*beta) + trackingClockOffset() */
void setTrackingClockBase(double t_s);
void resetTrackingClock(void);
void accumulateTrackingClock(double dt_s);
double trackingClockOffset(void);
/* (omega*trueOffset) reduced to [0,2*PI) in extended precision; a non-harmonic
   time-dependent consumer adds the across-gap difference of this to its phase */
double trackingClockPhase(double omega);
long binTransverseTimeDistribution(double **posItime, double *pz, long *pbin, double tmin,
                                   double dt, long nb, double *time, double **part, double Po, int64_t np,
                                   double dx, double dy, long xPower, long yPower);
long binTimeDistribution(double *Itime, long *pbin, double tmin,
                         double dt, long nb, double *time, double **part, double Po, int64_t np);

long trackBGGExpansion(double **part, int64_t np, BGGEXP *bgg, double pCentral, double **accepted, double *sigmaDelta2);

long trackMagneticFieldOffAxisExpansion(double **part, int64_t np, BOFFAXE *boa, double pCentral, double **accepted, double *sigmaDelta2);

void track_SReffects(double **coord, int64_t n, SREFFECTS *SReffects, double Po,
                     TWISS *twiss, RADIATION_INTEGRALS *radIntegrals,
                     long lossesOnly);
VMATRIX *srEffectsMatrix(SREFFECTS *SReffects);

void track_IBS(double **coord, int64_t np, ELEMENT_LIST *eptr, double Po, 
               ELEMENT_LIST *element, CHARGE *charge, long i_pass, long n_passes, RUN *run);

void addCorrectorRadiationKick(double **coord, int64_t np, ELEMENT_LIST *elem, long type, double Po, double *sigmaDelta2, 
			       long disableISR);
long track_through_csbendCSR(double **part, int64_t n_part, CSRCSBEND *csbend, double p_error, double Po, double **accepted,
                             double z_start, double z_end, CHARGE *charge, char *rootname, MAXAMP *maxamp, 
                             APCONTOUR *apContour, APERTURE_DATA *apFileData, ELEMENT_LIST *eptr);
long track_through_csbend(double **part, int64_t n_part, CSBEND *csbend, double p_error, double Po, double **accepted,
                          double z_start, double *sigmaDelta2, char *rootname, MAXAMP *maxamp, 
                          APCONTOUR *apContour, APERTURE_DATA *apFileData, long iSlice, ELEMENT_LIST *eptr);
void csbend_update_fse_adjustment(CSBEND *csbend, ELEMENT_LIST *eptr);
long track_through_driftCSR(double **part, int64_t np, CSRDRIFT *csrDrift, 
                            double Po, double **accepted, double zStart, 
			    double revolutionLength, CHARGE *charge, char *rootname);
long reset_driftCSR();
long applyLowPassFilter(double *histogram, long bins, double start, double end);
long applyLHPassFilters(double *histogram, long bins, double startHP, double endHP,
			double startLP, double endLP, long clipNegative);

void updateSpinQuaternionLocalFS(double S[3], double Bx, double By, double Bz,
				 double x, double y, double xp, double yp, double ds, double h, double p,
				 double charge, double mass, double a);
void performQuaternionRotation(double S[3], double rx, double ry, double rz);
void rotateSpinCoordinateSystem(double *S, double cos_t, double sin_t);
void rotateSpinsCoordinateSystem(double **particle, int64_t np, double t);
void updateSpinForSolenoid(double **coord, int64_t np, double Po, SOLE *sole);

/* Spin helpers shared between CCBEND and LGBEND.  See ccbend.c for definitions. */
void applyFringeSpinKick(double *spin, double xp, double xp0, double yp, double yp0,
                         double y, double rho, double p, double a);
void rotateSpinsAboutYAxis(double **particle, int64_t np, double angle);
  
long track_through_ccbend(double **particle, int64_t n_part, ELEMENT_LIST *eptr, CCBEND *ccbend, double Po,
                          double **accepted, double z_start, double *sigmaDelta2, char *rootname,
                          MAXAMP *maxamp, APCONTOUR *apContour, APERTURE_DATA *apFileData, int64_t iPart, long iFinalSlice);
void addCcbendRadiationIntegrals(CCBEND *ccbend, double *startingCoord, double pCentral,
                                 double eta0, double etap0, double beta0, double alpha0,
                                 double *I1, double *I2, double *I3, double *I4, double *I5, ELEMENT_LIST *elem);
int integrate_kick_KnL(double *coord, const double dx, const double dy, const double Po, const double rad_coef, const double isr_coef, const double *KnLFull, const long nTerms,
                       const long integration_order, const long n_parts, const int64_t iPart, 
                       long iFinalSlice, double drift, MULTIPOLE_DATA *multData, MULTIPOLE_DATA *edge1MultData, MULTIPOLE_DATA *edge2MultData, MULT_APERTURE_DATA *apData,
                       double *dzLoss, double *sigmaDelta2, double *lastRho1, double refTilt, double ZOffset, ELEMENT_LIST *eptr, GLOBAL_BEAM_SUMS *beamSums);

long track_through_lgbend(double **particle, int64_t n_part, ELEMENT_LIST *eptr, LGBEND *lgbend, double Po,
                          double **accepted, double z_start, double *sigmaDelta2, char *rootname,
                          MAXAMP *maxamp, APCONTOUR *apContour, APERTURE_DATA *apFileData, int64_t iPart, long iFinalSlice);
void readLGBendConfiguration(LGBEND *lgbend, ELEMENT_LIST *eptr);
void copyLGBend(LGBEND *target, LGBEND *source);
void configureLGBendGeometry(LGBEND *lgbend);
void flipLGBEND(LGBEND *lgbend);
void addLgbendRadiationIntegrals(LGBEND *lgbend, double *startingCoord, double pCentral,
                                 double eta0, double etap0, double beta0, double alpha0,
                                 double *I1, double *I2, double *I3, double *I4, double *I5, ELEMENT_LIST *elem);


void output_floor_coordinates(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline);
void final_floor_coordinates(LINE_LIST *beamline, double *XYZ, double *Angle,
                             double *XYZMin, double *XYZMax);
#define GLOBAL_LOCAL_MODE_DZ  1
#define GLOBAL_LOCAL_MODE_SEG 2
#define GLOBAL_LOCAL_MODE_END 3
#define GLOBAL_LOCAL_MODE_DZPM  4
void convertLocalCoordinatesToGlobal(double *Z, double *X, double *Y, double *thetaX, short mode,
                                     double *coord, ELEMENT_LIST *eptr, double dZ,
                                     long segment, long nSegments);

long trackThroughExactCorrector(double **part, int64_t n_part, ELEMENT_LIST *eptr, double Po, double **accepted, double z_start, double *sigmaDelta2);

long setup_load_parameters(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline);
long setup_load_parameters_for_file(char *filename, RUN *run, LINE_LIST *beamline);
long do_load_parameters(LINE_LIST *beamline, long change_definitions, char *scriptLoadFile);
char **addPatterns(long *patterns, char *input0);
int matchesPatternList(char **pattern, long patterns, char *input);
#define NO_LOAD_PARAMETERS 0
#define PARAMETERS_LOADED 1
#define PARAMETERS_ENDED 2
void finish_load_parameters(void);
extern void dumpRfcReferenceData(char *filename, RUN *run, LINE_LIST *beamline);
extern void finishRfcDataFile() ;

/* load parameters modes and indices */
/* order here must be the same as load_mode array in load_parameters.c */
#define LOAD_MODE_ABSOLUTE     0
#define LOAD_MODE_DIFFERENTIAL 1
#define LOAD_MODE_IGNORE       2
#define LOAD_MODE_FRACTIONAL   3
#define LOAD_FLAG_ABSOLUTE     (1<<LOAD_MODE_ABSOLUTE)
#define LOAD_FLAG_DIFFERENTIAL (1<<LOAD_MODE_DIFFERENTIAL)
#define LOAD_FLAG_IGNORE       (1<<LOAD_MODE_IGNORE)
#define LOAD_FLAG_FRACTIONAL   (1<<LOAD_MODE_FRACTIONAL)
#define LOAD_FLAG_VERBOSE      (LOAD_FLAG_FRACTIONAL<<1)
extern long nearestInteger(double value);
extern int64_t nearestInteger64(double value);

#define SDDS_EOS_NEWFILE 1
#define SDDS_EOS_COMPLETE 2
extern long check_sdds_column(SDDS_TABLE *SDDS_table, char *name, char *units);
extern long check_sdds_parameter(SDDS_TABLE *SDDS_table, char *name, char *units);
extern void SDDS_ElegantOutputSetup(SDDS_TABLE *SDDS_table, char *filename, long mode, long lines_per_row,
                             char *contents, char *command_file, char *lattice_file, SDDS_DEFINITION *parameter_definition,
                             long n_parameters, SDDS_DEFINITION *column_definition, long n_columns,
                             char *caller, long flags);
extern void SDDS_PhaseSpaceSetup(SDDS_TABLE *SDDS_table, char *filename, long mode, long lines_per_row, char *contents,
                          char *command_file, char *lattice_file, char *caller);
extern void SDDS_BeamLossSetup(SDDS_TABLE *SDDS_table, char *filename, long mode, long lines_per_row, char *contents, 
                               char *command_file, char *lattice_file, long includeGlobalCoordinates, char *caller);
extern void SDDS_SigmaMatrixSetup(SDDS_TABLE *SDDS_table, char *filename, long mode, long lines_per_row,
                           char *command_file, char *lattice_file, char *caller);
  extern void SDDS_WatchPointSetup(WATCH *watch, char *filename, long mode, long lines_per_row,
                                 char *command_file, char *lattice_file, char *caller, char *qualifier, 
                                 char *previousElementName, long previousElementOccurence);
extern int32_t SDDS_InitializeOutputElegant(SDDS_DATASET *SDDS_dataset, int32_t data_mode,
                                            int32_t lines_per_row, const char *description,
                                            const char *contents, const char *filename);
extern int32_t SDDS_Parallel_InitializeOutputElegant(SDDS_DATASET *SDDS_dataset, const char *description,
                                              const char *contents, const char *filename);

void SDDS_HistogramSetup(HISTOGRAM *histogram, char *filename, long mode, long lines_per_row,
                         char *command_file, char *lattice_file, char *caller);
void dump_particle_histogram(HISTOGRAM *histogram, long step, long pass, double **particle, int64_t particles,
                             double Po, double length, double charge, double z);
extern void dump_watch_particles(WATCH *watch, long step, long pass, double **particle, int64_t particles, double Po,
                                 double length, double mp_charge, double z, int64_t idSlotsPerBunch);
extern void dump_watch_parameters(WATCH *watch, long step, long pass, long n_passes, double **particle, int64_t particles,
				  long original_particles,  double Po, double revolutionLength, double z, double mp_charge);
extern void dump_watch_FFT(WATCH *watch, long step, long pass, long n_passes, double **particle, long particles,
                           long original_particles,  double Po, double length);
extern void do_watch_FFT(double **data, long n_data, long slot, long window_code);
extern void dump_lost_particles(SDDS_TABLE *SDDS_table, double *sLimit, double pCentral,
		  double **particle, int64_t particles, long step, double length);
extern void dump_centroid(SDDS_TABLE *SDDS_table, BEAM_SUMS *sums, LINE_LIST *beamline, long n_elements, long bunch,
                          double p_central, short bpmsOnly);
extern void dump_phase_space(SDDS_TABLE *SDDS_table, double **particle, int64_t particles, long step, double Po,
                             double charge, int64_t idSlotsPerBunch);
extern void dump_sigma(SDDS_TABLE *SDDS_table, BEAM_SUMS *sums, LINE_LIST *beamline, long n_elements, long step,
		       double p_central, short coupledSigmaOutput);
void computeEmitTwissFromSigmaMatrix(double *emit, double *emitc, double *beta, double *alpha, double sigma[7][7], long plane);
extern void doSASEFELAtEndOutput(SASEFEL_OUTPUT *sasefelOutput, long step);
extern void computeSASEFELAtEnd(SASEFEL_OUTPUT *sasefelOutput, double **particle, long particles, 
                         double Po, double charge);
extern void setupSASEFELAtEnd(NAMELIST_TEXT *nltext, RUN *run, OUTPUT_FILES *output_data);
extern void storeSASEFELAtEndInRPN(SASEFEL_OUTPUT *sasefelOutput);
extern void SDDS_CentroidOutputSetup(SDDS_TABLE *SDDS_table, char *filename, long mode, long lines_per_row, char *contents, char *command_file, char *lattice_file, char *caller, short bpmsOnly);
extern void SDDS_SigmaOutputSetup(SDDS_TABLE *SDDS_table, char *filename, long mode, long lines_per_row,
				  char *command_file, char *lattice_file, char *caller, short coupledSigmaOutput);
extern void readErrorMultipoleData(MULTIPOLE_DATA *multData, char *multFile, long steering);
extern void set_up_histogram(HISTOGRAM *histogram, RUN *run, long occurence, int64_t IDSlotsPerBunch);
extern long track_through_tubend(double **part, int64_t n_part, TUBEND *tubend,
                          double p_error, double Po, double **accepted,
                          double z_start);
extern void setup_sdds_beam(BEAM *beam,NAMELIST_TEXT *nltext,RUN *run, VARY *control,ERRORVAL *errcon,OPTIM_VARIABLES *optim,OUTPUT_FILES *output,LINE_LIST *beamline,long n_elements,
                            long save_original);
extern long new_sdds_beam(BEAM *beam,RUN *run,VARY *control,OUTPUT_FILES *output,long flags);
void terminate_sdds_beam(void);
extern void dumpLatticeParameters(char *filename, RUN *run, LINE_LIST *beamline, long suppressDefaults);
extern void do_fit_trace_data(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline);
extern void compute_offsets(void);
extern void convert_to_xorbit(char *outputfile, LINE_LIST *beamline, long flip_k, 
                    char *header_file, char *ender_file);
extern void finishLatticeParametersFile(void);
extern long memoryUsage();

void executeCshCommand(char *cmd, char *rootname);
extern void doSubprocessCommand(char *command);
void run_subprocess(NAMELIST_TEXT *nltext, RUN *run);
void setSearchPath(char *path);
char *findFileInSearchPath(const char *filename);
long getTableFromSearchPath(TABLE *tab, char *file, long sampleInterval, long flags);
/*long SDDS_InitializeInputFromSearchPath(SDDS_DATASET *SDDSin, char *file);*/

void ComputeSASEFELParameters
  (double *lightWavelength, double *saturationLength, double *gainLength,  double *noisePower,
   double *saturationPower, double *PierceParameter, double *etaDiffraction, double *etaEmittance,
   double *etaEnergySpread, double charge, double rmsBunchLength, double undulatorPeriod, double undulatorK, 
   double beta,  double emittance, double sigmaDelta, double pCentral, short planar);
double FELScalingFunction
  (double *etaDiffraction, double *etaEmittance,
   double *etaEnergySpread, double L1D, double beta, double emittance,
   double lightWavelength, double undulatorPeriod, double sigmaDelta);

void setup_alter_element(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline);
void do_alter_elements(RUN *run, LINE_LIST *beamline, short before_load_parameters, short per_step);

VMATRIX *twissTransformMatrix(TWISSELEMENT *twissWanted, TWISS *twissInput);
VMATRIX *twissTransformMatrix1(TWISS *twissWanted, TWISS *twissInput);
VMATRIX *lightThinLensMatrix(LTHINLENS *ltl);
VMATRIX *lightMirrorMatrix(LMIRROR *lm);

void setupDivideElements(NAMELIST_TEXT *nltext, RUN *run, 
			 LINE_LIST *beamline);
void addDivisionSpec(char *name, char *type, char *excludeNamePattern, char *excludeTypePattern,
		     long divisions,
		     double maximum_length);
long elementDivisions(char *name, char *type, double length);

void addTransmutationSpec(char *name, char *type, char *exclude,
                          long newType);
void clearTransmutationSpecs() ;
long elementTransmutation(char *name, long type) ;
void setupTransmuteElements(NAMELIST_TEXT *nltext, RUN *run, 
                            LINE_LIST *beamline);

void setupIgnoreElements(NAMELIST_TEXT *nltext, RUN *run, 
			 LINE_LIST *beamline);
long countIgnoreElementsSpecs(long completely);
void addIgnoreElementsSpec(char *name, char *type, char *exclude, long completely0);
void clearIgnoreElementsSpecs();
long ignoreElement(char *name, long type, long completelyOnly);
void setupTransmuteElements(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline);

void setupSCEffect(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline); 
void addSCSpec(char *name, char *type, char *exclude, long skip, long verbosity);
void clearSCSpecs();
long getSCMULTSpecCount();  
extern char *scMultName;
long insertSCMULT(char *name, long type, long *occurrence);
void trackThroughSCMULT(double **part, int64_t np, double Po, long iPass, ELEMENT_LIST *eptr, CHARGE *charge);
void finishSCSpecs();
void initializeSCMULT(ELEMENT_LIST *eptr, double **part, int64_t np, double Po, long i_pass, long idSlotsPerbunch);
void accumulateSCMULT(double **part, int64_t np, double Po, ELEMENT_LIST *eptr, long idSlotsPerbunch);
double computeRmsCoordinate(double **coord, long i1, int64_t np, double *mean, long *countReturn);
#if USE_MPI
double computeRmsCoordinate_p(double **coord, long i1, int64_t np, double *centroid, int64_t *npTotal, unsigned long classFlags);
#endif

void do_insert_elements(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline);
long insertElem(char *name, long type, long *skip, long occurPosition, double endPosition);
long getAddElemFlag(); 
char *getElemDefinition();
long getAddEndFlag();

void do_replace_elements(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline);
long replaceElem(char *name, long type, long *skip, long occurPosition);
long getDelElemFlag();
char *getElemDefinition1();

void TouschekEffect(RUN *run, VARY *control, ERRORVAL *errcon, LINE_LIST *beamline, NAMELIST_TEXT *nltext); 
void SDDS_BeamScatterSetup(SDDS_TABLE *SDDS_table, char *filename, long mode, long lines_per_row, char *contents, 
                           char *command_file, char *lattice_file, char *caller);
void dump_scattered_particles(SDDS_TABLE *SDDS_table, double **particle, 
                              long particles, double *weight, TSCATTER *tsptr);
void SDDS_BeamScatterLossSetup(SDDS_TABLE *SDDS_table, char *filename, long mode, long lines_per_row, char *contents, 
                               char *command_file, char *lattice_file, char *caller);
void dump_scattered_loss_particles(SDDS_TABLE *SDDS_table, double **particleLos, double **particleOri,
                                   long *lostOnPass, long particles, double *weight, TSCATTER *tsptr, 
                                   ELEMENT_LIST *eptr);

void transverseFeedbackPickup(TFBPICKUP *tfbp, double **part, int64_t np, long pass, double Po, long idSlotsPerBunch);
void initializeTransverseFeedbackPickup(TFBPICKUP *tfbp);
void transverseFeedbackDriver(TFBDRIVER *tfbd, double **part, int64_t np, LINE_LIST *beamline, long pass, long n_passes, char *rootname, double Po, long idSlotsPerBunch, CHARGE *charge);
void initializeTransverseFeedbackDriver(TFBDRIVER *tfbd, LINE_LIST *beamline, long n_passes, char *rootname);

void coolerPickup(CPICKUP *tfbp, double **part, int64_t np, long pass, double Po, long idSlotsPerBunch);
void initializeCoolerPickup(CPICKUP *tfbp);
void coolerKicker(CKICKER *tfbd, double **part, int64_t np, LINE_LIST *beamline, long pass, long n_passes, char *rootname, double Po, long idSlotsPerBunch);
void initializeCoolerKicker(CKICKER *tfbd, LINE_LIST *beamline, long n_passes, char *rootname, double Po);

long computeEngeCoefficients(double *engeCoef, double rho, double length, double gap, double fint);

long DefineNoiseGroup(long groupId);
long ResetNoiseGroupValues();
double GetNoiseGroupValue(long groupId);

/* from elegant.c */
void swapParticles(double *p1, double *p2);

/* prototypes for momentumAperture.c */
void setupMomentumApertureSearch(NAMELIST_TEXT *nltext, RUN *run, VARY *control);
void finishMomentumApertureSearch();
long doMomentumApertureSearch(RUN *run, VARY *control, ERRORVAL *errcon, LINE_LIST *beamline, double *startingCoord);
#if USE_MPI
void gatherLostParticles(double ***lostParticles, int64_t *nLost, double **coord, int64_t nSurvived, long n_processors, int myid);
#endif

/* prototypes for drand_oag.c */
double random_1_elegant(long iseed);

#if SDDS_MPI_IO
/* prototypes for media_oag.c */
long approximate_percentiles_p(double *position, double *percent, long positions, double *x, long n, 
			       long bins);
#endif

#define RESTART_RN_BEAMLINE 0x0001
#define RESTART_RN_SCATTER  0x0002
#define RESTART_RN_BPMNOISE 0x0004
#define RESTART_RN_BEAMGEN  0x0008
#define RESTART_RN_ALL      (RESTART_RN_BEAMLINE|RESTART_RN_SCATTER|RESTART_RN_BPMNOISE|RESTART_RN_BEAMGEN)
void seedElegantRandomNumbers(long seed, unsigned long restart);

/* compute long sum with Kahan's algorithm */
double Kahan (long length, double a[], double *error);
double KahanPlus (double oldSum, double b, double *error);
#if USE_MPI
double KahanParallel (double sum,  double error, MPI_Comm comm);
void find_global_min_max (double *min, double *max, int64_t np, MPI_Comm comm);
#endif
typedef struct {
  double t;
  int64_t ip;
} TIMEDATA;
extern int compTimeData(const void *tv1, const void *tv2);

#define MAX_BUCKETS 16384
int comp_BucketNumbers(const void *coord1, const void *coord2);


void setStartingMoments(SIGMA_MATRIX *sm, 
                        double emit_x, double beta_x, double alpha_x, double eta_x, double etap_x,
                        double emit_y, double beta_y, double alpha_y, double eta_y, double etap_y,
                        double emit_z, double beta_z, double alpha_z);
void propagateBeamMoments(RUN *run, LINE_LIST *beamline, double *traj);
void setupMomentsOutput(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline, long *doMomentsOutput,
                        long default_order);
void finishMomentsOutput(void);
long runMomentsOutput(RUN *run, LINE_LIST *beamline, double *startingCoord, long tune_corrected,
                      long writeToFile);
void computeAllUndulatorBrightnesses(LINE_LIST *beamline);
void fillSigmaPropagationMatrix(double **Ms, double **R);
long getMoments(double M[6][6], double C[6], long matched0, long equilibrium0, long radiation0);

/* The sigma matrix s[i][j] is stored in a 21-element array.  These indices give the i and j values 
 * corresponding to an element the array.  We have i<=j (upper triangular).  Values are filled in
 * by setSigmaIndices, which is called on start-up.
 */
extern long sigmaIndex1[21], sigmaIndex2[21];

/* This array gives the index in the 21-element array for given i and j.  Values are filled in
 * by setSigmaIndices, which is called on start-up.
 */
extern long sigmaIndex3[6][6];

void setup_coupled_twiss_output(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline, long *do_coupled_twiss_output,
                                long default_order);
int run_coupled_twiss_output(RUN *run, LINE_LIST *beamline, double *starting_coord);
void finish_coupled_twiss_output();
void SortEigenvalues (double *WR, double *WI, double *VR, int matDim, int eigenModesNumber, int verbosity);

long applyElementModulations(MODULATION_DATA *modData, LINE_LIST *beamline, double pCentral, double **coord, int64_t np,
                             RUN *run, long i_pass);
void addModulationElements(MODULATION_DATA *modData, NAMELIST_TEXT *nltext, LINE_LIST *beamline, RUN *run);

void addRampElements(RAMP_DATA *rampData, NAMELIST_TEXT *nltext, LINE_LIST *beamline, RUN *run);
long applyElementRamps(RAMP_DATA *rampData, LINE_LIST *beamline, double pCentral, RUN *run, long iPass);

#if USE_MPI
void histogram_sums(long nonEmptyBins, long firstBin, long *lastBin, long *his);
#endif

extern void setupIonEffects(NAMELIST_TEXT *nltext, VARY *control, RUN *run);
extern void completeIonEffectsSetup(RUN *run, LINE_LIST *beamline);
extern void trackWithIonEffects(double **part0, long np0, IONEFFECTS *ionEffects, double Po, long iPass, long nPasses, CHARGE *charge);
extern void evaluateVoltageFromLorentzian(double *Eperp, double a, double b, double x, double y);
extern void gaussianBeamKick(double *coord, double *center, double *sigma, long fromBeam, double kick[2], double charge, 
		      double ionMass, double ionCharge);
extern void ellipsoidalBeamKick(double *coord, double P0, double pMass, double pCharge, double centroid[2],
                                double size[2], double charge, short parabolic);

extern VMATRIX *computeMatricesFromTracking(FILE *fpo_ma, double **initial, double **final, double **error, 
				 double *step_size, double *maximum_value, int n_points1, int n_points_total,
				 int max_order_of_fits, int verbose);
extern int makeInitialParticleEnsemble(double ***initial, double *reference, double ***final, 
				       double ***error, int n_points1, double *step);

extern time_t get_mtime(char *filename);  

extern long trackBRAT(double **part, int64_t np, BRAT *brat, double pCentral, double **accepted);
extern int interpolate2dFieldMapHigherOrder(double *Foutput, double x, double y,
                                            double dx, double dy,
                                            double xmin, double ymin,
                                            double xmax, double ymax,
                                            long nx, long ny,
                                            double *F0, double *F1, double *F2, short order, short gridExcess);
extern int interpolate2dFieldMapHigherOrder2(double *Foutput, double x, double y,
					     double dx, double dy,
					     double xmin, double ymin,
					     double xmax, double ymax,
					     long nx, long ny,
					     void *F0, void *F1, void *F2, long offset, 
					     short singlePrecision, short order, short gridExcess);
extern void printWarning(char *text,  char *detail);
extern void printWarningForTracking(char *text, char *detail);
extern void printWarningWithContext(char *context1, char  *context2, char *text,  char *detail);
extern void setWarningFilePointer(FILE *fp);
extern void summarizeWarnings();
/* Push/pop nestable suppression of printWarning output (used by correction
 * routines during speculative steps that may destabilise the lattice). */
extern void pushWarningSuppression(void);
extern void popWarningSuppression(void);

extern void setObstructionsMode(long state) ;
extern void resetObstructionData(OBSTRUCTION_DATASETS *obsData);
extern void readObstructionInput(NAMELIST_TEXT *nltext, RUN *run);
extern long filterParticlesWithObstructions(double **coord, int64_t np, double **accepted, double z, double P_central);
extern long insideObstruction(double *part, short mode, double dz, long segment, long nSegments);
extern long insideObstruction_xyz(double x, double xp, double y, double yp, long particleID, double *lossCoordinates, 
				  double tilt, short mode, double dz, long segment, long nSegments);
extern long insideObstruction_XYZ(double X, double Y, double Z, double dXi, double dYi, double dZi, 
                                  double thetai, double xp, double *lossCoordinates);


extern void processGlobalSettings(NAMELIST_TEXT *nltext);

// to indicate very unlikely conditions
#define likely(x)	__builtin_expect(!!(x), 1)
#define unlikely(x)	__builtin_expect(!!(x), 0)

// This allows to specify things that will be true always (by negative converse)
// Can be used to hint at range of variable
#define assume( condition ) { if(!(condition)) __builtin_unreachable(); }

#ifdef __cplusplus
}
#endif
