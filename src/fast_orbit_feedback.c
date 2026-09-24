/*************************************************************************\
* Copyright (c) 2026 The University of Chicago, as Operator of Argonne
* National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution.
\*************************************************************************/

/* file: fast_orbit_feedback.c
 * purpose: fast orbit feedback (FOFB) simulation command.
 *
 * The command combines realistic beam dynamics (turn-by-turn tracking, with
 * synchrotron radiation) with a digital orbit-feedback loop:
 *
 *   for each FOFB iteration (run_control n_steps):
 *     track the (single, persistent) beam for run_control n_passes turns.
 *       - each turn every BPM's reading is pushed through its own digital IIR
 *         filter to form a running "tick" readout;
 *       - each turn every steering actuator's physically-applied kick is
 *         advanced one step of its own z-transform (IIR step response) toward
 *         the currently-held setpoint u[] -- so a setpoint that changes at the
 *         tick is NOT seen instantly by the beam, it ramps in turn-by-turn.
 *     at the end of the iteration the BPM ticks are collected into an orbit
 *     vector Q, projected into actuator space through the inverse response
 *     matrix T=-C^-1, and passed through a per-actuator-class PID controller
 *     (with anti-windup) to update the held setpoints u[].
 *
 * The steering-corrector core is complemented by an optional RF-frequency
 * actuator (include_rf_frequency, rf_Kp/Ki/Kd): the selected RFCA cavities form
 * one joint frequency knob appended to the x-plane response matrix as the
 * analytic dispersive column dx_i/df = -eta_i/(alpha_c*f0).  It closes the
 * steering<->energy loop that pure horizontal steering cannot, removing the slow
 * residual horizontal drift.  Which cavities are driven is selected through
 * &steering_element target=fofb, item=FREQ (else every RFCA is auto-discovered).
 *
 * See fast_orbit_feedback.nl for the namelist and the plan file for the full
 * design rationale.
 */

#include "mdb.h"
#include "track.h"
#include "correctDefs.h"
#include "fast_orbit_feedback.h"

/* nonzero while a fast_orbit_feedback command is running; read by do_tracking
   to enable the per-turn BPM store, the actuator z-transform hook, and by
   fofbStoreBpmTick/fofbUpdateActuators as a guard. */
long fofbActive = 0;

/* Total passes tracked across all FOFB steps (control->n_passes*control->n_steps).
   i_pass is continuous across steps (passOffset accumulates), so a WATCH in
   centroid/parameter mode accumulates one row per pass over the whole feedback run,
   not just n_passes.  dump_watch_parameters sizes its SDDS table from this when
   nonzero; it is 0 whenever FOFB is not running. */
long fofbTotalPasses = 0;

#define FOFB_MAX_FILTERS 100

double noise_value(double xamplitude, double xcutoff, long xerror_type);
long add_steer_type_to_lists(STEERING_LIST *SL, long plane, long type, char *item, double tweek, double limit,
                             LINE_LIST *beamline, RUN *run, long forceQuads);
long add_steer_elem_to_lists(STEERING_LIST *SL, long plane, char *name, char *item,
                             char *element_type, double tweek, double limit,
                             long start_occurence, long end_occurence, long occurence_step,
                             double s_start, double s_end,
                             LINE_LIST *beamline, RUN *run, long forceQuadsBends, long verbose);

/* FOFB-owned steering lists, populated by fofbAddSteerElem when a &steering_element
   command sets target="fast_orbit_feedback".  These are independent of the global
   &correct SLx/SLy.  fofbSL[0]=x correctors, fofbSL[1]=y correctors; fofbRFsl holds
   the RF cavities (item=FREQ).  setupPlane consumes them (auto-adding the standard
   dedicated correctors / all RFCAs when a class is not declared). */
static STEERING_LIST fofbSL[2];
static STEERING_LIST fofbRFsl;
static long fofbSLDeclared[2] = {0, 0};
static long fofbRFDeclared = 0;

/* Copies of the closed-orbit-start namelist flags, set in setupFastOrbitFeedback and
   read by elegant.c (through the accessors below) to center/offset the beam on the
   computed closed orbit before tracking begins, mirroring the &track command. */
static long fofbCenterOnOrbitFlag = 0;
static long fofbCenterMomentumAlsoFlag = 1;
static long fofbOffsetByOrbitFlag = 0;
static long fofbOffsetMomentumAlsoFlag = 1;

long fofbCenterOnOrbit(void) { return fofbCenterOnOrbitFlag; }
long fofbCenterMomentumAlso(void) { return fofbCenterMomentumAlsoFlag; }
long fofbOffsetByOrbit(void) { return fofbOffsetByOrbitFlag; }
long fofbOffsetMomentumAlso(void) { return fofbOffsetMomentumAlsoFlag; }

/* ------------------------------------------------------------------ */
/* FOFB-owned steering-element intake: routed here from add_steering_element when a
   &steering_element command sets target="fast_orbit_feedback".  item=FREQ selects
   RF cavities (the joint RF-frequency knob); anything else is a plane corrector. */

long fofbAddSteerElem(long plane, char *name, char *item, char *element_type, double tweek, double limit,
                      long start_occurence, long end_occurence, long occurence_step,
                      double s_start, double s_end, LINE_LIST *beamline, RUN *run, long verbose) {
  long found;
  if (item && strcmp(item, "FREQ") == 0) {
    found = add_steer_elem_to_lists(&fofbRFsl, plane, name, item, element_type, tweek, limit,
                                    start_occurence, end_occurence, occurence_step, s_start, s_end,
                                    beamline, run, 0, verbose);
    if (found)
      fofbRFDeclared = 1;
  } else {
    long idx = (plane == 2) ? 1 : 0;
    found = add_steer_elem_to_lists(&fofbSL[idx], plane, name, item ? item : (idx ? "VKICK" : "HKICK"),
                                    element_type, tweek, limit,
                                    start_occurence, end_occurence, occurence_step, s_start, s_end,
                                    beamline, run, 0, verbose);
    if (found)
      fofbSLDeclared[idx] = 1;
  }
  return found;
}

/* release a STEERING_LIST's arrays (mirrors the reset branch of add_steer_elem_to_lists) */
static void freeSteeringList(STEERING_LIST *SL) {
  long i;
  if (SL->corr_param) {
    for (i = 0; i < SL->n_corr_types; i++)
      if (SL->corr_param[i])
        free(SL->corr_param[i]);
    free(SL->corr_param);
  }
  if (SL->elem)
    free(SL->elem);
  if (SL->corr_tweek)
    free(SL->corr_tweek);
  if (SL->corr_limit)
    free(SL->corr_limit);
  if (SL->param_offset)
    free(SL->param_offset);
  if (SL->param_index)
    free(SL->param_index);
  memset(SL, 0, sizeof(*SL));
}

/* ------------------------------------------------------------------ */
/* per-monitor runtime state, attached through ELEMENT_LIST->p_elem->fofbData */

typedef struct {
  IIRFILTER *xFilter, *yFilter; /* independent IIR banks (own state) for this BPM */
  long nxFilter, nyFilter;
  double xTick, yTick; /* running filtered readout (value at the last turn = the tick) */
} FOFB_BPM_DATA;

/* per-plane control state, all arrays indexed by corrector 0..CM.ncor-1 */

typedef struct {
  long coord;   /* 0 = x, 2 = y */
  long active;  /* nonzero if this plane has both monitors and correctors */
  CORMON_DATA CM;
  STEERING_LIST SL;
  double *base;        /* actuator base setpoint (parameter units, p_elem) */
  double *u;           /* currently-held actuator setpoint (parameter units) */
  double *Iacc;        /* PID integral accumulator (parameter units) */
  double *ePrev;       /* previous PID error (parameter units) */
  double *lastApplied; /* last physically-applied parameter value (for change test) */
  short *actPegged;    /* nonzero while actuator setpoint is pegged at the limit */
  long *kickOffset;    /* SL.param_offset[sl_index] for each corrector */
  IIRFILTER **actFilter; /* per-corrector actuator IIR bank (own state) */
  long *nActFilter;
  double Kp, Ki, Kd;   /* steering-class PID gains */
  /* RF-frequency actuator (x plane only): the selected RFCAs form one joint knob,
     registered as one extra actuator column at index rfIndex (-1 if inactive). */
  long rfIndex;        /* actuator index of the joint RF-frequency knob, or -1 */
  double rfKp, rfKi, rfKd; /* RF-class PID gains */
  double s_rf;         /* SVD conditioning scale applied to the RF response column */
  ELEMENT_LIST **rfElem; /* selected RFCA elements driven with a common detuning */
  double *rfBaseFreq;  /* each selected cavity's base frequency (Hz) */
  long *rfFreqOffset;  /* offset of FREQ within each cavity's p_elem */
  long nRF;            /* number of selected RF cavities */
} FOFB_PLANE;

static FOFB_PLANE planeData[2]; /* [0]=x, [1]=y */
static long fofbSetupDone = 0;

/* diagnostic output.  One SDDS page is written per feedback step and flushed to
   disk as the run proceeds (mirroring do_transport_analysis in analyze.c), so the
   file can be examined live while the code is still running rather than only at the
   end.  fofbHasDfrfColumn records whether the RF-frequency detuning column exists. */
static SDDS_DATASET SDDS_fofb;
static long fofbOutputActive = 0;
static long fofbHasDfrfColumn = 0;

/* master-only guard for output under MPI */
#if USE_MPI
#  define FOFB_IS_MASTER (myid == 0)
#else
#  define FOFB_IS_MASTER 1
#endif

/* ------------------------------------------------------------------ */

static void **fofbDataPtr(ELEMENT_LIST *eptr) {
  switch (eptr->type) {
  case T_HMON:
    return &(((HMON *)eptr->p_elem)->fofbData);
  case T_VMON:
    return &(((VMON *)eptr->p_elem)->fofbData);
  case T_MONI:
    return &(((MONI *)eptr->p_elem)->fofbData);
  default:
    return NULL;
  }
}

/* allocate a fresh IIR bank and load it from file (own state); returns count */
static IIRFILTER *loadFilterBank(char *file, long *nFilter) {
  IIRFILTER *bank;
  *nFilter = 0;
  if (!file)
    return NULL;
  bank = tmalloc(sizeof(*bank) * FOFB_MAX_FILTERS);
  memset(bank, 0, sizeof(*bank) * FOFB_MAX_FILTERS);
  *nFilter = readIIRFilter(bank, FOFB_MAX_FILTERS, file);
  if (*nFilter <= 0) {
    free(bank);
    return NULL;
  }
  return bank;
}

/* zero the running state of an IIR bank without discarding coefficients */
static void resetFilterBank(IIRFILTER *bank, long nFilter) {
  long i, j;
  for (i = 0; i < nFilter; i++) {
    bank[i].iBuffer = 0;
    for (j = 0; j < bank[i].nTerms; j++)
      bank[i].xn[j] = bank[i].yn[j] = 0;
  }
}

/* ------------------------------------------------------------------ */
/* attach FOFB_BPM_DATA to every monitor used by either plane */

static void attachMonitorData(FOFB_PLANE *plane) {
  long i;
  for (i = 0; i < plane->CM.nmon; i++) {
    ELEMENT_LIST *moni = plane->CM.umoni[i];
    void **pp = fofbDataPtr(moni);
    FOFB_BPM_DATA *bd;
    if (!pp)
      continue;
    if (!(bd = *pp)) {
      bd = tmalloc(sizeof(*bd));
      memset(bd, 0, sizeof(*bd));
      *pp = bd;
    }
    if (plane->coord == 0 && !bd->xFilter)
      bd->xFilter = loadFilterBank(bpm_filter_file, &bd->nxFilter);
    if (plane->coord == 2 && !bd->yFilter)
      bd->yFilter = loadFilterBank(bpm_filter_file, &bd->nyFilter);
  }
}

/* ------------------------------------------------------------------ */
/* invert CM->C into CM->T = -C^-1, mirroring the COMPUTE_RESPONSE_INVERT block of
   compute_orbcor_matrices (correct.c).  Used after the RF column is appended. */

static void fofbInvertCM(CORMON_DATA *CM) {
  double conditionNumber;
  if (CM->T) {
    matrix_free(CM->T);
    CM->T = NULL;
  }
  if (CM->auto_limit_SVs && (CM->C->m < CM->C->n) && CM->remove_smallest_SVs < (long)(CM->C->n - CM->C->m))
    CM->remove_smallest_SVs = CM->C->n - CM->C->m;
  CM->T = matrix_invert(CM->C, CM->equalW ? NULL : CM->weight,
                        (int32_t)CM->keep_largest_SVs, (int32_t)CM->remove_smallest_SVs,
                        CM->minimum_SV_ratio, CM->Tikhonov_relative_alpha, CM->Tikhonov_n,
                        0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &conditionNumber);
  matrix_scmul(CM->T, -1);
  if (verbosity)
    printf("fast_orbit_feedback: RF-augmented correction matrix condition number %e\n", conditionNumber);
}

/* Append the joint RF-frequency actuator to the x-plane response matrix.  Builds the
   analytic dispersive column dx_i/df = -eta_i/(alpha_c*f0), scales it by s_rf for SVD
   conditioning, appends it to CM->C, and extends the CM per-corrector arrays by one
   (kick_coef=1/s_rf, sl_index=-1 as the RF sentinel).  Populates plane->rf* state. */

static void appendRFActuator(FOFB_PLANE *plane, RUN *run, LINE_LIST *beamline) {
  CORMON_DATA *CM = &plane->CM;
  long nSteer = CM->ncor, nmon = CM->nmon, i, j, k;
  double alpha_c, f0, *rfcol, sC = 0, sR = 0, rmsC, rmsR, s_rf;
  MAT *Caug;

  /* resolve the cavity list: declared entries, else auto-add every RFCA.
     add_steer_elem_to_lists uppercases its name/item/element_type in place, so it
     must receive mutable copies (not string literals in read-only memory). */
  if (!fofbRFDeclared || fofbRFsl.n_corr_types == 0) {
    char *anyName, *freqItem, *rfcaType;
    cp_str(&anyName, "*");
    cp_str(&freqItem, "FREQ");
    cp_str(&rfcaType, "RFCA");
    if (!add_steer_elem_to_lists(&fofbRFsl, 0, anyName, freqItem, rfcaType, 1e-6, 0,
                                 0, 0, 1, -1, -1, beamline, run, 0, verbosity > 1))
      bombElegant("fast_orbit_feedback: include_rf_frequency set but no RFCA cavities found", NULL);
    free(anyName);
    free(freqItem);
    free(rfcaType);
  }
  plane->nRF = fofbRFsl.n_corr_types;

  alpha_c = beamline->alpha[0];
  if (alpha_c == 0)
    bombElegant("fast_orbit_feedback: momentum compaction alpha_c is zero; cannot form RF-frequency response (need &twiss_output)", NULL);
  f0 = ((RFCA *)(fofbRFsl.elem[0]->p_elem))->freq;
  if (f0 <= 0)
    bombElegant("fast_orbit_feedback: RF cavity has non-positive FREQ", NULL);

  /* analytic per-Hz dispersive column at the monitors */
  rfcol = tmalloc(sizeof(*rfcol) * nmon);
  for (i = 0; i < nmon; i++)
    rfcol[i] = -CM->umoni[i]->twiss->etax / (alpha_c * f0);

  /* conditioning scale so the RF column carries weight comparable to the correctors */
  for (j = 0; j < nSteer; j++)
    for (i = 0; i < nmon; i++)
      sC += Mij(CM->C, i, j) * Mij(CM->C, i, j);
  for (i = 0; i < nmon; i++)
    sR += rfcol[i] * rfcol[i];
  rmsC = nSteer ? sqrt(sC / ((double)nmon * nSteer)) : 1.0;
  rmsR = sqrt(sR / nmon);
  if (rf_response_scale > 0)
    s_rf = rf_response_scale;
  else
    s_rf = (rmsR > 0 && rmsC > 0) ? rmsC / rmsR : 1.0;
  plane->s_rf = s_rf;

  /* augment C with the scaled RF column as the last column */
  Caug = matrix_get(nmon, nSteer + 1);
  for (j = 0; j < nSteer; j++)
    memcpy(Caug->me[j], CM->C->me[j], sizeof(double) * nmon);
  for (i = 0; i < nmon; i++)
    Mij(Caug, i, nSteer) = rfcol[i] * s_rf;
  free(rfcol);
  matrix_free(CM->C);
  CM->C = Caug;

  /* extend the per-corrector CM arrays by one for the RF entry */
  CM->kick_coef = trealloc(CM->kick_coef, sizeof(*CM->kick_coef) * (nSteer + 1));
  CM->sl_index = trealloc(CM->sl_index, sizeof(*CM->sl_index) * (nSteer + 1));
  CM->ucorr = trealloc(CM->ucorr, sizeof(*CM->ucorr) * (nSteer + 1));
  CM->pegged = trealloc(CM->pegged, sizeof(*CM->pegged) * (nSteer + 1));
  CM->kick_coef[nSteer] = 1.0 / s_rf; /* e = Mij(dK,rf)/kick_coef is then Hz */
  CM->sl_index[nSteer] = -1;          /* RF sentinel */
  CM->ucorr[nSteer] = fofbRFsl.elem[0];
  CM->pegged[nSteer] = 0;
  CM->ncor = nSteer + 1;
  plane->rfIndex = nSteer;

  /* cache the cavity elements / base frequencies for the turn-by-turn apply */
  plane->rfElem = tmalloc(sizeof(*plane->rfElem) * plane->nRF);
  plane->rfBaseFreq = tmalloc(sizeof(*plane->rfBaseFreq) * plane->nRF);
  plane->rfFreqOffset = tmalloc(sizeof(*plane->rfFreqOffset) * plane->nRF);
  for (k = 0; k < plane->nRF; k++) {
    plane->rfElem[k] = fofbRFsl.elem[k];
    plane->rfFreqOffset[k] = fofbRFsl.param_offset[k];
    plane->rfBaseFreq[k] = *((double *)(fofbRFsl.elem[k]->p_elem + fofbRFsl.param_offset[k]));
  }

  if (verbosity)
    printf("fast_orbit_feedback: RF-frequency actuator active over %ld cavity(ies), f0=%.6e Hz, alpha_c=%.6e, s_rf=%.3e\n",
           plane->nRF, f0, alpha_c, s_rf);
  fflush(stdout);
}

/* ------------------------------------------------------------------ */

static void setupPlane(FOFB_PLANE *plane, long coord, RUN *run, LINE_LIST *beamline) {
  CORMON_DATA *CM;
  STEERING_LIST *SL;
  long i, found = 0, idx = (coord == 0) ? 0 : 1, rfActive;
  char *item;

  memset(plane, 0, sizeof(*plane));
  plane->coord = coord;
  plane->rfIndex = -1;
  CM = &plane->CM;

  /* select the corrector steering list: if this plane's correctors were declared
     through &steering_element target=fofb, use exactly those; otherwise auto-add the
     standard dedicated-corrector element types (mirrors correction_setup defaults,
     minus quad/bend steering). */
  if (fofbSLDeclared[idx]) {
    SL = &fofbSL[idx];
    found = SL->n_corr_types;
    if (verbosity)
      printf("fast_orbit_feedback: %s plane uses %ld user-declared corrector(s)\n",
             coord == 0 ? "x" : "y", found);
  } else {
    SL = &plane->SL;
    if (coord == 0) {
      cp_str(&item, "KICK");
      found += add_steer_type_to_lists(SL, 0, T_HCOR, item, 1e-6, corrector_limit, beamline, run, 0);
      found += add_steer_type_to_lists(SL, 0, T_EHCOR, item, 1e-6, corrector_limit, beamline, run, 0);
      cp_str(&item, "HKICK");
      found += add_steer_type_to_lists(SL, 0, T_HVCOR, item, 1e-6, corrector_limit, beamline, run, 0);
      found += add_steer_type_to_lists(SL, 0, T_EHVCOR, item, 1e-6, corrector_limit, beamline, run, 0);
    } else {
      cp_str(&item, "KICK");
      found += add_steer_type_to_lists(SL, 2, T_VCOR, item, 1e-6, corrector_limit, beamline, run, 0);
      found += add_steer_type_to_lists(SL, 2, T_EVCOR, item, 1e-6, corrector_limit, beamline, run, 0);
      cp_str(&item, "VKICK");
      found += add_steer_type_to_lists(SL, 2, T_HVCOR, item, 1e-6, corrector_limit, beamline, run, 0);
      found += add_steer_type_to_lists(SL, 2, T_EHVCOR, item, 1e-6, corrector_limit, beamline, run, 0);
    }
  }

  /* the RF-frequency actuator is a horizontal (dispersive) knob only */
  rfActive = (coord == 0 && include_rf_frequency);

  if (!found && !rfActive) {
    printWarning("No steering correctors found for a plane.",
                 coord == 0 ? "fast_orbit_feedback x plane inactive." : "fast_orbit_feedback y plane inactive.");
    plane->active = 0;
    return;
  }

  /* SV / inversion controls consumed by compute_orbcor_matrices */
  CM->nmon = CM->ncor = 0;
  CM->C = CM->T = NULL;
  CM->fixed_length = CM->fixed_length_matrix = 0;
  CM->auto_limit_SVs = 1;
  CM->keep_largest_SVs = keep_largest_SVs;
  CM->remove_smallest_SVs = remove_smallest_SVs;
  CM->minimum_SV_ratio = minimum_SV_ratio;
  CM->Tikhonov_relative_alpha = Tikhonov_relative_alpha;
  CM->Tikhonov_n = Tikhonov_n;

  /* When the RF actuator is active we append its column before inverting, so defer
     the inversion; otherwise invert here exactly as increment 1. */
  compute_orbcor_matrices(CM, SL, coord, run, beamline,
                          (rfActive ? 0 : COMPUTE_RESPONSE_INVERT) | (verbosity < 2 ? COMPUTE_RESPONSE_SILENT : 0), 0);

  if (CM->nmon == 0) {
    plane->active = 0;
    return;
  }

  if (rfActive) {
    appendRFActuator(plane, run, beamline);
    fofbInvertCM(CM);
  }

  if (CM->ncor == 0 || CM->T == NULL) {
    plane->active = 0;
    return;
  }
  plane->active = 1;

  /* copy BPM noise controls into CM (used by fofbStoreBpmTick) */
  CM->bpm_noise = bpm_noise;
  CM->bpm_noise_cutoff = bpm_noise_cutoff;
  CM->bpm_noise_distribution = 1; /* gaussian; uniform via bpm_noise_distribution string below */
  if (bpm_noise_distribution && strncmp(bpm_noise_distribution, "uniform", 7) == 0)
    CM->bpm_noise_distribution = 2;

  /* per-actuator state */
  plane->base = tmalloc(sizeof(*plane->base) * CM->ncor);
  plane->u = tmalloc(sizeof(*plane->u) * CM->ncor);
  plane->Iacc = tmalloc(sizeof(*plane->Iacc) * CM->ncor);
  plane->ePrev = tmalloc(sizeof(*plane->ePrev) * CM->ncor);
  plane->lastApplied = tmalloc(sizeof(*plane->lastApplied) * CM->ncor);
  plane->actPegged = tmalloc(sizeof(*plane->actPegged) * CM->ncor);
  plane->kickOffset = tmalloc(sizeof(*plane->kickOffset) * CM->ncor);
  plane->actFilter = tmalloc(sizeof(*plane->actFilter) * CM->ncor);
  plane->nActFilter = tmalloc(sizeof(*plane->nActFilter) * CM->ncor);
  for (i = 0; i < CM->ncor; i++) {
    long sl_index = CM->sl_index[i];
    if (sl_index < 0) {
      /* joint RF-frequency knob: base setpoint is the reference cavity frequency */
      plane->kickOffset[i] = plane->rfFreqOffset[0];
      plane->base[i] = plane->rfBaseFreq[0];
      plane->actFilter[i] = loadFilterBank(rf_filter_file, &plane->nActFilter[i]);
    } else {
      ELEMENT_LIST *corr = CM->ucorr[i];
      plane->kickOffset[i] = SL->param_offset[sl_index];
      plane->base[i] = *((double *)(corr->p_elem + plane->kickOffset[i]));
      plane->actFilter[i] = loadFilterBank(steering_filter_file, &plane->nActFilter[i]);
    }
    plane->u[i] = plane->base[i];
    plane->lastApplied[i] = plane->base[i];
    plane->Iacc[i] = plane->ePrev[i] = 0;
    plane->actPegged[i] = 0;
  }

  plane->Kp = steering_Kp;
  plane->Ki = steering_Ki;
  plane->Kd = steering_Kd;
  plane->rfKp = rf_Kp;
  plane->rfKi = rf_Ki;
  plane->rfKd = rf_Kd;

  attachMonitorData(plane);

  if (verbosity)
    printf("fast_orbit_feedback: %s plane has %ld monitors, %ld correctors\n",
           coord == 0 ? "x" : "y", (long)CM->nmon, (long)CM->ncor);
  fflush(stdout);
}

/* ------------------------------------------------------------------ */

void setupFastOrbitFeedback(NAMELIST_TEXT *nltext, RUN *run, VARY *control, LINE_LIST *beamline) {
  /* process namelist */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&fast_orbit_feedback, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &fast_orbit_feedback);

  if (output_interval < 1)
    output_interval = 1;

  /* stash the closed-orbit-start flags for elegant.c to act on before tracking */
  fofbCenterOnOrbitFlag = center_on_orbit;
  fofbCenterMomentumAlsoFlag = center_momentum_also;
  fofbOffsetByOrbitFlag = offset_by_orbit;
  fofbOffsetMomentumAlsoFlag = offset_momentum_also;

#if USE_MPI
  printWarning("fast_orbit_feedback is validated for serial (elegant) running.",
               "Under Pelegant with a distributed beam, BPM readings are not reduced across ranks.");
#endif

  /* Warn if the deck defined &steering_element correctors but none target the feedback.
     Such commands default to target="correct" and populate the &correct SLx/SLy lists,
     which fast_orbit_feedback does not read -- so the feedback silently falls back to
     auto-discovering dedicated correctors (and ignores DQCOR-type or other user-chosen
     correctors).  This is the common footgun when adapting an orbit-correction deck. */
  if (!fofbSLDeclared[0] && !fofbSLDeclared[1] && !fofbRFDeclared &&
      nNonFOFBSteeringElementsSeen() > 0)
    printWarning("fast_orbit_feedback: steering_element commands were given but none target the feedback.",
                 "Add target=\"fofb\" to the &steering_element commands so their correctors drive the feedback; "
                 "otherwise those definitions are ignored and the feedback auto-discovers dedicated correctors.");

  setupPlane(&planeData[0], 0, run, beamline);
  setupPlane(&planeData[1], 2, run, beamline);

  if (!planeData[0].active && !planeData[1].active)
    bombElegant("fast_orbit_feedback: no active correction plane (no monitors and/or correctors found)", NULL);

  /* diagnostic output setup (master only) */
  fofbOutputActive = 0;
  if (output && FOFB_IS_MASTER) {
    output = compose_filename(output, run->rootname);
    /* SDDS_DefineParameter/SDDS_DefineColumn return the new element's index (>=0)
       and -1 on error -- unlike the DefineSimple* forms (1/0), so the first
       element's index of 0 would make a "!" test spuriously fail; check "< 0". */
    if (!SDDS_InitializeOutput(&SDDS_fofb, SDDS_BINARY, 1, NULL, "fast orbit feedback", output) ||
        SDDS_DefineParameter(&SDDS_fofb, "nMonitorsX", NULL, NULL,
                             "Number of horizontal beam-position monitors read by the feedback",
                             NULL, SDDS_LONG, NULL) < 0 ||
        SDDS_DefineParameter(&SDDS_fofb, "nCorrectorsX", NULL, NULL,
                             "Number of horizontal actuators driven by the feedback",
                             NULL, SDDS_LONG, NULL) < 0 ||
        SDDS_DefineParameter(&SDDS_fofb, "nMonitorsY", NULL, NULL,
                             "Number of vertical beam-position monitors read by the feedback",
                             NULL, SDDS_LONG, NULL) < 0 ||
        SDDS_DefineParameter(&SDDS_fofb, "nCorrectorsY", NULL, NULL,
                             "Number of vertical steering correctors driven by the feedback",
                             NULL, SDDS_LONG, NULL) < 0 ||
        SDDS_DefineColumn(&SDDS_fofb, "Step", NULL, NULL,
                          "Feedback iteration index (0-based); one row per iteration",
                          NULL, SDDS_LONG, 0) < 0 ||
        SDDS_DefineColumn(&SDDS_fofb, "Pass", NULL, NULL,
                          "Cumulative tracking turns completed through the end of this iteration",
                          NULL, SDDS_LONG, 0) < 0 ||
        SDDS_DefineColumn(&SDDS_fofb, "xBpmRms", NULL, "m",
                          "RMS over the horizontal BPM errors of the filtered orbit reading measured at this iteration",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDS_fofb, "yBpmRms", NULL, "m",
                          "RMS over the vertical BPM errors of the filtered orbit reading measured at this iteration",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDS_fofb, "MaxCorrX", NULL, "rad",
                          "Largest horizontal steering-corrector kick excursion from its base value "
                          "at this iteration (the RF-frequency actuator is excluded)",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDS_fofb, "MaxCorrY", NULL, "rad",
                          "Largest vertical steering-corrector kick excursion from its base value "
                          "at this iteration",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDS_fofb, "PeggedX", NULL, NULL,
                          "Number of horizontal actuators saturated at their limit at this iteration",
                          NULL, SDDS_LONG, 0) < 0 ||
        SDDS_DefineColumn(&SDDS_fofb, "PeggedY", NULL, NULL,
                          "Number of vertical actuators saturated at their limit at this iteration",
                          NULL, SDDS_LONG, 0) < 0) {
      SDDS_SetError("Unable to set up fast_orbit_feedback output file");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    /* the RF-frequency detuning column exists only when the RF actuator is active,
       so include_rf_frequency=0 runs keep the increment-1 output layout unchanged */
    if (include_rf_frequency &&
        SDDS_DefineColumn(&SDDS_fofb, "DeltaFrf", NULL, "Hz",
                          "Applied RF-frequency detuning from the base frequency at this iteration "
                          "(one common fractional detuning shared by all selected cavities)",
                          NULL, SDDS_DOUBLE, 0) < 0) {
      SDDS_SetError("Unable to set up fast_orbit_feedback DeltaFrf column");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    if (!SDDS_WriteLayout(&SDDS_fofb)) {
      SDDS_SetError("Unable to write fast_orbit_feedback output layout");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    fofbOutputActive = 1;
    fofbHasDfrfColumn = include_rf_frequency ? 1 : 0;
  }

  fofbSetupDone = 1;
}

/* ------------------------------------------------------------------ */
/* per-turn hook: advance each actuator's applied kick one z-transform step
   toward its held setpoint u[].  Called from do_tracking's pass loop.
   Returns nonzero if any element matrix was changed. */

long fofbUpdateActuators(LINE_LIST *beamline, RUN *run, long i_pass) {
  long ip, i, changed = 0;
  for (ip = 0; ip < 2; ip++) {
    FOFB_PLANE *plane = &planeData[ip];
    if (!plane->active)
      continue;
    for (i = 0; i < plane->CM.ncor; i++) {
      double delta, applied, *pval;
      delta = plane->u[i] - plane->base[i];
      if (plane->nActFilter[i] > 0)
        applied = plane->base[i] + applyIIRFilter(plane->actFilter[i], plane->nActFilter[i], delta);
      else
        applied = plane->u[i]; /* no step response -> setpoint applied immediately */
      if (applied == plane->lastApplied[i])
        continue;

      if (i == plane->rfIndex) {
        /* joint RF-frequency knob: every selected cavity gets the same fractional
           detuning applied/base.  The RFCA reads freq live from p_elem during
           tracking, so no matrix recompute is needed; but a bare freq write would
           leave phase_fiducial (= -omega_old*t0) stale, so rescale it by the
           frequency ratio to hold the fiducial time t0 fixed (subtlety (a)). */
        double fidRatio = (plane->lastApplied[i] != 0) ? applied / plane->lastApplied[i] : 1.0;
        double frac = (plane->base[i] != 0) ? applied / plane->base[i] : 1.0;
        long k;
        for (k = 0; k < plane->nRF; k++) {
          RFCA *rfca = (RFCA *)(plane->rfElem[k]->p_elem);
          *((double *)(plane->rfElem[k]->p_elem + plane->rfFreqOffset[k])) = plane->rfBaseFreq[k] * frac;
          if (rfca->fiducial_seen) {
            rfca->phase_fiducial *= fidRatio; /* preserves t0 = -phase_fiducial/omega */
            set_phase_reference(rfca->phase_reference, rfca->phase_fiducial);
          }
        }
        plane->lastApplied[i] = applied;
        continue; /* no compute_matrix / assert_element_links for the RF entry */
      }

      {
        ELEMENT_LIST *corr = plane->CM.ucorr[i];
        pval = (double *)(corr->p_elem + plane->kickOffset[i]);
        *pval = applied;
        plane->lastApplied[i] = applied;
        /* refresh the element matrix so matrix tracking sees the new kick */
        if (corr->matrix) {
          free_matrices(corr->matrix);
          tfree(corr->matrix);
          corr->matrix = NULL;
        }
        compute_matrix(corr, run, NULL);
        changed = 1;
      }
    }
  }
  if (changed && beamline->links)
    assert_element_links(beamline->links, run, beamline, DYNAMIC_LINK);
  return changed;
}

/* ------------------------------------------------------------------ */
/* per-turn hook: push a BPM reading through its IIR filter to update the tick */

void fofbStoreBpmTick(ELEMENT_LIST *eptr, double xReading, double yReading) {
  void **pp = fofbDataPtr(eptr);
  FOFB_BPM_DATA *bd;
  if (!pp || !(bd = *pp))
    return;
  if (eptr->type == T_HMON || eptr->type == T_MONI) {
    if (bpm_noise)
      xReading += noise_value(bpm_noise, bpm_noise_cutoff,
                              (bpm_noise_distribution && strncmp(bpm_noise_distribution, "uniform", 7) == 0) ? 2 : 1);
    bd->xTick = bd->nxFilter > 0 ? applyIIRFilter(bd->xFilter, bd->nxFilter, xReading) : xReading;
  }
  if (eptr->type == T_VMON || eptr->type == T_MONI) {
    if (bpm_noise)
      yReading += noise_value(bpm_noise, bpm_noise_cutoff,
                              (bpm_noise_distribution && strncmp(bpm_noise_distribution, "uniform", 7) == 0) ? 2 : 1);
    bd->yTick = bd->nyFilter > 0 ? applyIIRFilter(bd->yFilter, bd->nyFilter, yReading) : yReading;
  }
}

/* ------------------------------------------------------------------ */
/* PID update in actuator space, run once per FOFB iteration */

static void updateSetpoints(FOFB_PLANE *plane, double *rmsOrbit, double *maxCorr, long *nPegged) {
  CORMON_DATA *CM = &plane->CM;
  MAT *Q, *dK;
  long i;
  double sum2 = 0, maxc = 0;
  *nPegged = 0;

  /* collect BPM ticks into the orbit vector */
  Q = matrix_get(CM->nmon, 1);
  for (i = 0; i < CM->nmon; i++) {
    void **pp = fofbDataPtr(CM->umoni[i]);
    FOFB_BPM_DATA *bd = pp ? *pp : NULL;
    double reading = 0;
    if (bd)
      reading = plane->coord == 0 ? bd->xTick : bd->yTick;
    Mij(Q, i, 0) = reading;
    sum2 += reading * reading;
  }
  *rmsOrbit = CM->nmon ? sqrt(sum2 / CM->nmon) : 0;

  /* dK = T * Q = -C^-1 * Q : deadbeat correction demand (kick units) */
  dK = matrix_mult(CM->T, Q);

  for (i = 0; i < CM->ncor; i++) {
    double e, d, uProp, uNew, applied, Kp, Ki, Kd, limitVal, limitRef;
    long isRF = (i == plane->rfIndex);
    if (isRF) {
      /* RF-frequency class: gains rf_*, and the limit is a detuning about the base
         frequency (|f-f0| <= rf_frequency_limit), not an absolute value. */
      Kp = plane->rfKp;
      Ki = plane->rfKi;
      Kd = plane->rfKd;
      limitVal = rf_frequency_limit;
      limitRef = plane->base[i];
    } else {
      Kp = plane->Kp;
      Ki = plane->Ki;
      Kd = plane->Kd;
      limitVal = corrector_limit;
      limitRef = 0; /* correctors clamp on the absolute kick */
    }
    e = Mij(dK, i, 0) / CM->kick_coef[i]; /* parameter-space error (Hz for the RF entry) */
    d = e - plane->ePrev[i];
    if (!(anti_windup && plane->actPegged[i]))
      plane->Iacc[i] += Ki * e;
    uProp = plane->base[i] + Kp * e + Kd * d;
    uNew = uProp + plane->Iacc[i];
    if (limitVal > 0 && fabs(uNew - limitRef) > limitVal) {
      double lim = limitRef + (uNew > limitRef ? 1.0 : -1.0) * limitVal;
      if (anti_windup)
        plane->Iacc[i] = lim - uProp; /* back-calculate: freeze integrator at saturation */
      uNew = lim;
      plane->actPegged[i] = 1;
      (*nPegged)++;
    } else
      plane->actPegged[i] = 0;
    plane->u[i] = uNew;
    plane->ePrev[i] = e;
    if (!isRF) {
      /* MaxCorr* reports the steering excursion (rad); the RF detuning is a
         different quantity and is excluded here. */
      applied = fabs(plane->u[i] - plane->base[i]);
      if (applied > maxc)
        maxc = applied;
    }
  }
  *maxCorr = maxc;

  matrix_free(dK);
  matrix_free(Q);
}

/* ------------------------------------------------------------------ */

long doFastOrbitFeedback(RUN *run, VARY *control, LINE_LIST *beamline, BEAM *beam, OUTPUT_FILES *output_files) {
  long i_step, ip, passOffset = 0;
  double pCentral, finalCharge = 0;
  double clockBase = 0.0;
  unsigned long baseFlags, savedBeamlineFidFlag;

  if (!fofbSetupDone)
    bombElegant("fast_orbit_feedback: setup was not performed", NULL);

  pCentral = run->p_central;
  fofbActive = 1;
  fofbTotalPasses = control->n_passes * control->n_steps;

  /* Unlike &track (one do_tracking call per step), FOFB tracks ONE persistent beam
     across all n_steps, calling do_tracking once per step so tracking is continuous.
     do_tracking OR's beamline->fiducial_flag into its flags (do_tracking.c), and that
     flag carries RESET_RF_FOR_EACH_STEP by default (run_control's reset_rf_for_each_step,
     copied to the beamline in elegant.c). That would make do_tracking delete the phase
     references and re-fiducialize the RF at the start of EVERY step -- recomputing each
     cavity's fiducial time t0 from the mid-flight beam and shifting the RF energy kick at
     the step boundary (a spurious centroid-energy jump). The RF must be fiducialized once,
     on the step-0 beam, and then held for the life of the continuous beam. We therefore
     save the flag, let step 0 fiducialize exactly as a standalone first step would, and
     clear RESET_RF_FOR_EACH_STEP before the remaining steps; the flag is restored on exit. */
  savedBeamlineFidFlag = beamline->fiducial_flag;

  /* SUPPRESS_TWISS_UPDATE: the actuators change corrector kicks and rf frequency
     every turn, which invalidates the cached twiss and would otherwise force
     do_tracking to re-derive the twiss parameters on every pass (see
     do_tracking.c).  Those thin kicks move only the closed orbit, not the linear
     optics, so the recomputation is wasted work; suppress it automatically. */
  baseFlags = FINAL_SUMS_ONLY | ALLOW_MPI_ABORT_TRACKING | SUPPRESS_TWISS_UPDATE;

  for (i_step = 0; i_step < control->n_steps; i_step++) {
    unsigned long flags;
    double xrms = 0, yrms = 0, xcor = 0, ycor = 0;
    long xpeg = 0, ypeg = 0;

    /* optional reset of the BPM filter running state (never the actuator/Iacc state) */
    if (reset_filters_each_step) {
      for (ip = 0; ip < 2; ip++) {
        FOFB_PLANE *plane = &planeData[ip];
        long i;
        if (!plane->active)
          continue;
        for (i = 0; i < plane->CM.nmon; i++) {
          void **pp = fofbDataPtr(plane->CM.umoni[i]);
          FOFB_BPM_DATA *bd = pp ? *pp : NULL;
          if (!bd)
            continue;
          if (bd->nxFilter > 0)
            resetFilterBank(bd->xFilter, bd->nxFilter);
          if (bd->nyFilter > 0)
            resetFilterBank(bd->yFilter, bd->nyFilter);
        }
      }
    }

    flags = baseFlags |
            (control->fiducial_flag &
             (FIRST_BEAM_IS_FIDUCIAL + FIDUCIAL_BEAM_SEEN + RESTRICT_FIDUCIALIZATION +
              LINEAR_CHROMATIC_MATRIX + LONGITUDINAL_RING_ONLY));

    /* FOFB tracks one persistent beam across n_steps do_tracking calls, but each
       do_tracking call resets the tracking clock accumulator (do_tracking.c
       resetTrackingClock()).  Standard &track makes a single do_tracking call, so its
       CHANGE_T/step_frequency macro-time offset accumulates continuously over all passes.
       To reproduce that continuity here, seed each step's base with the total macro time
       elapsed through the previous step; the upcoming do_tracking zeroes the accumulator
       before any consumer (e.g. applyElementModulations) reads trackingClockOffset(), so
       the offset stays continuous across step boundaries instead of sawtoothing to 0. */
    setTrackingClockBase(clockBase);

    do_tracking(beam, NULL, 0, NULL, beamline, &pCentral, beam->accepted, NULL, NULL, NULL,
                run, i_step, flags, control->n_passes, passOffset, NULL, NULL, &finalCharge, NULL, NULL);

    /* capture this step's accumulated macro time (base + per-step accum) so the next
       step continues from it */
    clockBase = trackingClockOffset();

    /* hold the RF fiducial across the remaining iterations */
    if (control->fiducial_flag & FIRST_BEAM_IS_FIDUCIAL) {
      control->fiducial_flag |= FIDUCIAL_BEAM_SEEN;
      beamline->fiducial_flag |= FIDUCIAL_BEAM_SEEN;
    }
    /* After step 0 has established the RF fiducial on the continuous beam, stop
       do_tracking from deleting phase references / re-fiducializing the RF at each
       subsequent step boundary (see the header comment above). */
    beamline->fiducial_flag &= ~RESET_RF_FOR_EACH_STEP;

    passOffset += control->n_passes;

    /* run the controller for each active plane */
    if (planeData[0].active)
      updateSetpoints(&planeData[0], &xrms, &xcor, &xpeg);
    if (planeData[1].active)
      updateSetpoints(&planeData[1], &yrms, &ycor, &ypeg);

    if (verbosity)
      printf("fast_orbit_feedback step %ld: xBpmRms=%.6e m, yBpmRms=%.6e m (pegged x=%ld, y=%ld)\n",
             i_step, xrms, yrms, xpeg, ypeg);

    /* Write (and flush) one output page for this step so the file can be examined
       while the run is still in progress, rather than only at the end. */
    if (fofbOutputActive && FOFB_IS_MASTER && (i_step % output_interval == 0)) {
      if (!SDDS_StartPage(&SDDS_fofb, 1) ||
          !SDDS_SetParameters(&SDDS_fofb, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                              "nMonitorsX", (int32_t)planeData[0].CM.nmon, "nCorrectorsX", (int32_t)planeData[0].CM.ncor,
                              "nMonitorsY", (int32_t)planeData[1].CM.nmon, "nCorrectorsY", (int32_t)planeData[1].CM.ncor, NULL) ||
          !SDDS_SetRowValues(&SDDS_fofb, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, 0,
                             "Step", (int32_t)i_step, "Pass", (int32_t)passOffset,
                             "xBpmRms", xrms, "yBpmRms", yrms, "MaxCorrX", xcor, "MaxCorrY", ycor,
                             "PeggedX", (int32_t)xpeg, "PeggedY", (int32_t)ypeg, NULL)) {
        SDDS_SetError("Unable to write fast_orbit_feedback output page");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
      if (fofbHasDfrfColumn) {
        double dfrf = (planeData[0].active && planeData[0].rfIndex >= 0)
                        ? planeData[0].u[planeData[0].rfIndex] - planeData[0].base[planeData[0].rfIndex]
                        : 0.0;
        if (!SDDS_SetRowValues(&SDDS_fofb, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, 0,
                               "DeltaFrf", dfrf, NULL)) {
          SDDS_SetError("Unable to set fast_orbit_feedback DeltaFrf value");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
      }
      if (!SDDS_WritePage(&SDDS_fofb)) {
        SDDS_SetError("Unable to write fast_orbit_feedback output page");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
      if (!inhibitFileSync)
        SDDS_DoFSync(&SDDS_fofb);
    }
  }

  fofbActive = 0;
  fofbTotalPasses = 0;
  beamline->fiducial_flag = savedBeamlineFidFlag;
  return 1;
}

/* ------------------------------------------------------------------ */

static void freePlane(FOFB_PLANE *plane) {
  long i;
  if (!plane->active) {
    if (plane->CM.C)
      matrix_free(plane->CM.C);
    if (plane->CM.T)
      matrix_free(plane->CM.T);
    plane->CM.C = plane->CM.T = NULL;
    return;
  }
  /* detach & free per-monitor data */
  for (i = 0; i < plane->CM.nmon; i++) {
    void **pp = fofbDataPtr(plane->CM.umoni[i]);
    FOFB_BPM_DATA *bd = pp ? *pp : NULL;
    if (!bd)
      continue;
    if (bd->xFilter) {
      freeIIRFilterMemory(bd->xFilter, bd->nxFilter);
      free(bd->xFilter);
    }
    if (bd->yFilter) {
      freeIIRFilterMemory(bd->yFilter, bd->nyFilter);
      free(bd->yFilter);
    }
    free(bd);
    *pp = NULL;
  }
  for (i = 0; i < plane->CM.ncor; i++) {
    if (plane->actFilter[i]) {
      freeIIRFilterMemory(plane->actFilter[i], plane->nActFilter[i]);
      free(plane->actFilter[i]);
    }
  }
  free(plane->base);
  free(plane->u);
  free(plane->Iacc);
  free(plane->ePrev);
  free(plane->lastApplied);
  free(plane->actPegged);
  free(plane->kickOffset);
  free(plane->actFilter);
  free(plane->nActFilter);
  if (plane->rfElem)
    free(plane->rfElem);
  if (plane->rfBaseFreq)
    free(plane->rfBaseFreq);
  if (plane->rfFreqOffset)
    free(plane->rfFreqOffset);
  plane->rfElem = NULL;
  plane->rfBaseFreq = NULL;
  plane->rfFreqOffset = NULL;
  if (plane->CM.C)
    matrix_free(plane->CM.C);
  if (plane->CM.T)
    matrix_free(plane->CM.T);
  plane->CM.C = plane->CM.T = NULL;
  plane->active = 0;
}

void finishFastOrbitFeedback(void) {
  fofbActive = 0;
  fofbTotalPasses = 0;
  if (!fofbSetupDone)
    return;

  if (fofbOutputActive && FOFB_IS_MASTER) {
    /* every step already wrote and flushed its own page during tracking; just
       close the file out here. */
    if (!SDDS_Terminate(&SDDS_fofb)) {
      SDDS_SetError("Unable to finish fast_orbit_feedback output file");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  }
  fofbOutputActive = 0;
  fofbHasDfrfColumn = 0;

  freePlane(&planeData[0]);
  freePlane(&planeData[1]);

  /* release the FOFB-owned steering intake so a later &fast_orbit_feedback
     re-declares its actuators cleanly */
  freeSteeringList(&fofbSL[0]);
  freeSteeringList(&fofbSL[1]);
  freeSteeringList(&fofbRFsl);
  fofbSLDeclared[0] = fofbSLDeclared[1] = 0;
  fofbRFDeclared = 0;

  fofbSetupDone = 0;
}
