/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

#include "matrixOp.h"

/* see correction.c for additional explanation of the next three structures */

typedef struct {
  /* arrays for information on individual correcting elements */
  ELEMENT_LIST **elem;                   /* pointers to elements */
  char **corr_param;                     /* parameter names */
  long *param_offset;                    /* offset of correcting parameter in element structure */
  long *param_index;                     /* index of correcting parameter in entity description */
  double *corr_tweek;                    /* tweek values--amount to change parameter by to get dkick/dparam */
  double *corr_limit;                    /* limiting absolute value of the parameter */
  long n_corr_types;
} STEERING_LIST;

typedef struct {
    /* information on useful monitors and correctors */
    long *mon_index;                       /* index of monitor in trajectory array (which has trajectory at all elements) */
    int32_t nmon, ncor;                    /* numbers of monitors and correctors */
    ELEMENT_LIST **umoni, **ucorr;         /* arrays of pointers to monitor and corrector elements */
    double *kick_coef;                     /* dkick/dparam (==1 for hkick, vkick, and hvkick elements) */
    long *sl_index;                        /* index of steering list entry for each corrector */
    short *pegged;                         /* if non-zero, corrector is pegged */
    double *weight;                        /* weights for monitors */
    short equalW;                          /* if non-zero, all weights are equal */

    /* arrays for holding corrector information for output */
    double **kick, **posi;
    /* copies of input specifications for correction */
    double corr_fraction, corr_accuracy, corr_limit, bpm_noise, default_tweek, bpm_noise_cutoff;
    long bpm_noise_distribution, default_threading_divisor, threading_correctors;
    long fixed_length, fixed_length_matrix, use_perturbed_matrix;
    long remove_smallest_SVs, keep_largest_SVs, auto_limit_SVs, remove_pegged, Tikhonov_n;
    double minimum_SV_ratio, Tikhonov_relative_alpha;
    long n_iterations;
    /* correction matrix and inverse, respectively: */
    /* Mij(C, i, j) = dX(monitor i)/dK(corrector j) */
    MAT *C, *T; 
    /* Cp is the slope response. Not used for correction but useful for output. */
    /* Mij(Cp, i, j) = dX'(monitor i)/dK(corrector j) */
    MAT *Cp;
    /* information about last correction */
    long n_cycles_done;
    /* at present these are used only for response matrix for cross-plane correction */
    short bpmPlane, corrPlane;
    } CORMON_DATA;

typedef struct {
    short mode;
#define TRAJECTORY_CORRECTION 0
#define ORBIT_CORRECTION 1
    short method;
    short verbose, track_before_and_after, n_iterations, n_xy_cycles, minimum_cycles;
    short prezero_correctors, start_from_centroid, use_actual_beam, response_only, disable;
    short xplane, yplane, forceAlternation;
    short use_response_from_computed_orbits;
    short use_altered_matrices[2];
    double clorb_accuracy, clorb_accuracy_requirement, clorb_iterations, clorb_iter_fraction, clorb_fraction_multiplier;
  short clorb_multiplier_interval, clorb_track_for_orbit, rpn_store_response_matrix;
    STEERING_LIST SLx, SLy;
    TRAJECTORY **traj;
    /* These structures store Full data for all correctors */
    CORMON_DATA *CMFx, *CMFy;
    /* These structure store "altered" data after removal of pegged correctors */
    CORMON_DATA *CMAx, *CMAy;
    /* These just point to the verison in use */
    CORMON_DATA *CMx, *CMy;
    } CORRECTION ;
    

#define COMPUTE_RESPONSE_FINDONLY    0x0001UL
#define COMPUTE_RESPONSE_INVERT      0x0002UL
#define COMPUTE_RESPONSE_SILENT      0x0004UL
#define COMPUTE_RESPONSE_FIXEDLENGTH 0x0008UL
#define COMPUTE_RESPONSE_VERBOSE     0x0010UL
extern void compute_trajcor_matrices(CORMON_DATA *CM, STEERING_LIST *SL, long coord, RUN *run, LINE_LIST *beamline, unsigned long flags,
                                     short rpn_store_response_matrix);
void compute_coupled_trajcor_matrices(CORMON_DATA *CM, STEERING_LIST *SL, RUN *run, LINE_LIST *beamline, unsigned long flags);
extern void compute_orbcor_matrices(CORMON_DATA *CM, STEERING_LIST *SL, long coord, RUN *run, LINE_LIST *beamline, unsigned long flags,
                                    short rpn_store_response_matrix);
extern void compute_orbcor_matrices1(CORRECTION *_correct, long coord, RUN *run, LINE_LIST *beamline, unsigned long flags,
                                     long coupled, CORMON_DATA *CMxy, CORMON_DATA *CMyx);

#ifdef USE_MPI
void compute_orbcor_matrices1p(CORRECTION *_correct, RUN *run, LINE_LIST *beamline, unsigned long flags,
                               long coupled, CORMON_DATA *CMxy, CORMON_DATA *CMyx);
#endif

extern void setup_corrector_output(char *filename, RUN *run);
extern void dump_corrector_data(CORMON_DATA *CM, STEERING_LIST *SL, long index, char *plane, long step);
extern void setup_cormon_stats(char *filename, RUN *run);
extern void dump_cormon_stats(long verbose, long plane, double **kick, long n_kicks, 
    double **position, long n_positions, double *Cdp, long n_iterations, long cycle,
    long final_cycle, long step, long textOnly);
extern void setup_orb_traj_output(char *filename, char *mode, RUN *run);
extern void dump_orb_traj(TRAJECTORY *traj, long n_elems, char *description, long step);

extern void setup_correction_matrix_output(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline, 
                                           CORRECTION *correct,
                                           long *do_response, long BnLUnitsOk);
extern void run_response_output(RUN *run, LINE_LIST *beamline, CORRECTION *correct, long tune_corrected);
extern void update_response(RUN *run, LINE_LIST *beamline, CORRECTION *correct) ;
extern void correction_setup(CORRECTION *_correct, NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline);
extern long do_correction(CORRECTION *correct, RUN *run, LINE_LIST *beamline, double *starting_coords, 
        BEAM *beam, long sim_step, unsigned long correctionStepFlag);
#define INITIAL_CORRECTION   0x0001UL
#define FINAL_CORRECTION     0x0002UL
#define NO_OUTPUT_CORRECTION 0x0004UL
extern void add_steering_element(CORRECTION *correct, LINE_LIST *beamline, RUN *run, NAMELIST_TEXT *nltext);
void compute_amplification_factors(NAMELIST_TEXT *nltext, RUN *run, CORRECTION *correct,
    long closed_orbit, LINE_LIST *beamline);

long zero_correctors(ELEMENT_LIST *elem, RUN *run, CORRECTION *correct);

void finish_orb_traj_output();
void finish_bpm_output();
void finish_corrector_output();
void finish_cormon_stats();

#define GLOBAL_CORRECTION 0
#define ONE_TO_ONE_CORRECTION 1
#define THREAD_CORRECTION 2
#define ONE_TO_BEST_CORRECTION 3
#define ONE_TO_NEXT_CORRECTION 4
#define COUPLED_CORRECTION 5
#define N_CORRECTION_METHODS 6
