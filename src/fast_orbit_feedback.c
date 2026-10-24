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
 *     track the (single, persistent) beam for passes_between_updates turns.
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
 * The command is organized by TIMESCALE.  run_control n_passes is NOT used:
 * the fast update cadence is passes_between_updates, and the slow RF-frequency
 * loop updates every rf_update_multiplier fast iterations.
 *
 * FAST loop (CM->C/T matrices): updates every iteration. The steering correctors plus an
 * optional RF-PHASE column (include_rf_phase, energy_Kp/Ki/Kd).  A held RF phase
 * offset produces, over a sub-synchrotron-period step, a meaningful energy/orbit
 * response linearized about the synchronous phase: dx_i/dphi = eta_i/phaseScale.
 * Appended as the last column (phaseIndex), it lets the deadbeat solve assign the
 * dispersive (eta-shaped) orbit to the phase knob and the betatron residual to the
 * correctors automatically.  Cavities: &steering_element target=fofb, item=DELTA_PHASE.
 *
 * SLOW loop (separate matrix plane->Crf/Trf, updated every rf_update_multiplier
 * iterations): the RF-FREQUENCY actuator (include_rf_frequency, rf_Kp/Ki/Kd).  The
 * selected RFCA cavities form one joint frequency knob whose static dispersive
 * response dx_i/df = -eta_i/(alpha_c*f0) is its own single-column pseudo-inverse.
 * Cavities: &steering_element target=fofb, item=FREQ.
 *
 * With both loops enabled they coexist: the fast phase column damps the AC synchrotron
 * transient (zero DC, the beam re-phases) and the slow frequency loop nails the DC
 * dispersive orbit.  When the frequency loop runs WITHOUT a phase column, the fast
 * corrector solve is automatically made eta-blind (the dispersion mode is projected
 * out of Q in updateSetpoints) so the correctors do not fight the slow frequency
 * integrator over the energy channel.
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

/* Total passes tracked across all FOFB steps (passes_between_updates*control->n_steps).
   i_pass is continuous across steps (passOffset accumulates), so a WATCH in
   centroid/parameter mode accumulates one row per pass over the whole feedback run,
   not just passes_between_updates.  dump_watch_parameters sizes its SDDS table from this when
   nonzero; it is 0 whenever FOFB is not running. */
long fofbTotalPasses = 0;

/* Maximum number of cascaded filters per filter file */
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
static STEERING_LIST fofbRFPhasesl;
static long fofbSLDeclared[2] = {0, 0};
static long fofbRFDeclared = 0;
static long fofbRFPhaseDeclared = 0;

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
   RF cavities for the joint RF-frequency knob; item=PHASE/DELTA_PHASE/PHASE_SETPOINT selects
   RF cavities for the RF-phase energy actuator (a separate scalar loop).  PHASE and DELTA_PHASE
   drive an RFCA's additive DELTA_PHASE channel (leaving PHASE free for injected phase errors);
   PHASE_SETPOINT drives an RFMODE's generator/feedback phase setpoint directly.  Anything else
   is a plane corrector. */

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
  } else if (item && (strcmp(item, "PHASE") == 0 || strcmp(item, "DELTA_PHASE") == 0 ||
                      strcmp(item, "PHASE_SETPOINT") == 0)) {
    /* All three select the RF-phase energy actuator.  "PHASE"/"DELTA_PHASE" drive an RFCA's additive
       DELTA_PHASE (not PHASE); "PHASE_SETPOINT" drives an RFMODE's phase setpoint literally.  Pass a
       mutable copy -- add_steer_elem_to_lists uppercases its item argument in place. */
    char *phaseItem;
    cp_str(&phaseItem, strcmp(item, "PHASE_SETPOINT") == 0 ? "PHASE_SETPOINT" : "DELTA_PHASE");
    found = add_steer_elem_to_lists(&fofbRFPhasesl, plane, name, phaseItem, element_type, tweek, limit,
                                    start_occurence, end_occurence, occurence_step, s_start, s_end,
                                    beamline, run, 0, verbose);
    free(phaseItem);
    if (found)
      fofbRFPhaseDeclared = 1;
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
  /* separate x-filter for the RF-frequency loop (x/dispersive plane only): the frequency
     integrator must average over the synchrotron period (notch f_s) while steering keeps a
     short bpm_filter_file; populated only when rf_bpm_filter_file is set. */
  IIRFILTER *xFilterRF;
  long nxFilterRF;
  double xTickRF; /* running RF-loop filtered readout; == raw reading when no RF bank */
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
  /* RF-FREQUENCY actuator (x plane only): the selected RFCAs form one joint knob with
     its OWN separate single-column (eta-shaped) response matrix Crf/Trf and scalar PID
     state (NOT a column of CM->C).  Updated on the slow cadence (rf_update_multiplier),
     held (zero-order) between updates. */
  MAT *Crf, *Trf;      /* standalone RF-frequency response (nmon x 1) and its -pinv (1 x nmon) */
  double rfKp, rfKi, rfKd; /* RF-class PID gains */
  double s_rf;         /* conditioning scale applied to the RF response column */
  double rfBase;       /* reference cavity base frequency (Hz) */
  double rfU;          /* currently-held RF-frequency setpoint (Hz) */
  double rfIacc;       /* RF-frequency PID integral accumulator (Hz) */
  double rfEPrev;      /* previous RF-frequency PID error (Hz) */
  double rfLastApplied;/* last physically-applied RF frequency (Hz, after any ramp) */
  short rfPegged;      /* nonzero while rfU is clamped at rf_frequency_limit */
  IIRFILTER *rfActFilter; /* optional RF-frequency actuator ramp bank (rf_frequency_filter_file) */
  long nRfActFilter;
  ELEMENT_LIST **rfElem; /* selected RFCA elements driven with a common detuning */
  double *rfBaseFreq;  /* each selected cavity's base frequency (Hz) */
  long *rfFreqOffset;  /* offset of FREQ within each cavity's p_elem */
  long nRF;            /* number of selected RF cavities */
  /* RF-PHASE energy actuator (x/dispersive plane only): now a COLUMN of the fast matrix
     CM->C at index phaseIndex (sl_index==-2 sentinel), driven by the deadbeat solve with
     its own PID class (energy_Kp/Ki/Kd, clamp rf_phase_limit).  The phase offset is
     u[phaseIndex]-base[phaseIndex] (deg), written to the cavity PHASE without touching
     phase_fiducial.  phaseScale (deg per unit delta) sets the column dx_i/dphi=eta_i/phaseScale. */
  long phaseIndex;            /* CM->ncor index of the RF-phase column, or -1 if inactive */
  long rfPhaseActive;         /* nonzero when include_rf_phase drives this plane */
  double s_phase;             /* conditioning scale applied to the RF-phase response column */
  ELEMENT_LIST **rfPhaseElem; /* selected RFCA cavities whose PHASE is modulated */
  double *rfBasePhase;        /* each cavity's base PHASE (deg) */
  long *rfPhaseOffset;        /* offset of PHASE within each cavity's p_elem */
  long nRFPhase;              /* number of selected cavities */
  double energyKp, energyKi, energyKd; /* RF-phase-class PID gains */
  double phaseScale;          /* deg of RF phase per unit delta (signed); builds the column */
  double deltaEst;            /* last dispersion-projected energy offset (diagnostic only) */
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
static long fofbHasPhaseColumns = 0;

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

/* allocate a fresh IIR bank and load it from file (own state); returns count.
   Each SDDS page in the file is one cascade stage: FOFB applies the bank with
   applyIIRFilterCascade (series), so stage n's output feeds stage n+1's input and
   the stages may have different numbers of coefficients (rows).  A single-page
   file is one filter, identical to the former single-stage behavior. */
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
    /* dedicated BPM filter for the RF-frequency loop (x plane only) */
    if (plane->coord == 0 && include_rf_frequency && rf_bpm_filter_file && !bd->xFilterRF)
      bd->xFilterRF = loadFilterBank(rf_bpm_filter_file, &bd->nxFilterRF);
  }
}

/* ------------------------------------------------------------------ */
/* invert CM->C into CM->T = -C^-1, mirroring the COMPUTE_RESPONSE_INVERT block of
   compute_orbcor_matrices (correct.c).  Used after the RF-phase column is appended. */

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
    printf("fast_orbit_feedback: fast correction matrix condition number %e\n", conditionNumber);
}

/* Build the standalone RF-frequency response matrix and its pseudo-inverse.  The selected
   RFCA cavities form one joint frequency knob whose static dispersive response at the
   monitors is dx_i/df = -eta_i/(alpha_c*f0).  This is its OWN single-column matrix
   plane->Crf (nmon x 1) with plane->Trf = -pinv(Crf) (1 x nmon) -- NOT a column of the
   fast CM->C.  The loop runs on the slow cadence against the notch-filtered orbit.
   Also initializes the scalar RF-frequency PID state and caches the cavity list. */

static void buildRFMatrix(FOFB_PLANE *plane, RUN *run, LINE_LIST *beamline) {
  CORMON_DATA *CM = &plane->CM;
  long nmon = CM->nmon, i, k;
  double alpha_c, f0, conditionNumber;

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

  /* single-column (nmon x 1) analytic per-Hz dispersive response; a conditioning scale is
     unnecessary for a standalone one-column pseudo-inverse (it normalizes out), so s_rf=1
     and the solved error is already in Hz (kick_coef == 1). */
  plane->s_rf = 1.0;
  plane->Crf = matrix_get(nmon, 1);
  for (i = 0; i < nmon; i++)
    Mij(plane->Crf, i, 0) = -CM->umoni[i]->twiss->etax / (alpha_c * f0);

  if (plane->Trf) {
    matrix_free(plane->Trf);
    plane->Trf = NULL;
  }
  plane->Trf = matrix_invert(plane->Crf, NULL, 0, 0, 0.0, 0.0, -1,
                             0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &conditionNumber);
  matrix_scmul(plane->Trf, -1);

  /* cache the cavity elements / base frequencies for the turn-by-turn apply */
  plane->rfElem = tmalloc(sizeof(*plane->rfElem) * plane->nRF);
  plane->rfBaseFreq = tmalloc(sizeof(*plane->rfBaseFreq) * plane->nRF);
  plane->rfFreqOffset = tmalloc(sizeof(*plane->rfFreqOffset) * plane->nRF);
  for (k = 0; k < plane->nRF; k++) {
    plane->rfElem[k] = fofbRFsl.elem[k];
    plane->rfFreqOffset[k] = fofbRFsl.param_offset[k];
    plane->rfBaseFreq[k] = *((double *)(fofbRFsl.elem[k]->p_elem + fofbRFsl.param_offset[k]));
  }

  /* scalar PID state for the single frequency knob (clamped about rfBase) */
  plane->rfBase = plane->rfBaseFreq[0];
  plane->rfU = plane->rfBase;
  plane->rfLastApplied = plane->rfBase;
  plane->rfIacc = plane->rfEPrev = 0;
  plane->rfPegged = 0;
  plane->rfActFilter = loadFilterBank(rf_frequency_filter_file, &plane->nRfActFilter);

  if (verbosity)
    printf("fast_orbit_feedback: RF-frequency actuator active over %ld cavity(ies), f0=%.6e Hz, alpha_c=%.6e (separate slow matrix)\n",
           plane->nRF, f0, alpha_c);
  fflush(stdout);
}

/* ------------------------------------------------------------------ */
/* Return an RF-phase actuator element's drive voltage (V), synchronous phase (deg), and RF
   frequency (Hz), reading the fields appropriate to its type.  RFCA drives its additive
   DELTA_PHASE about PHASE (volt/phase/freq); an RFMODE drives PHASE_SETPOINT, the target of
   its internal generator/direct-RF-feedback loop, whose realized total cavity voltage/phase at
   FOFB (sub-feedback-bandwidth) timescales track voltageSetpoint/phaseSetpoint, with the RF
   frequency given by driveFrequency.  PHASE_SETPOINT shares RFCA PHASE's V*sin(phi) convention
   (rfmode.cc converts sin->cos internally at set-up), so the same V*cos(phi) energy-response form
   applies.  Bombs on any other element type. */

static void fofbCavityDrive(ELEMENT_LIST *e, double *volt, double *phaseDeg, double *freq) {
  if (e->type == T_RFCA) {
    RFCA *r = (RFCA *)e->p_elem;
    *volt = r->volt;
    *phaseDeg = r->phase;
    *freq = r->freq;
  } else if (e->type == T_RFMODE) {
    RFMODE *r = (RFMODE *)e->p_elem;
    /* One RFMODE occurrence can represent N_CAVITIES identical cavities: the energy kick is
       n_cavities*V (rfmode.cc), so the per-degree energy authority -- and hence the voltage that
       enters sum(V*cos(phi_s)) -- is n_cavities*V_SETPOINT, not the per-cavity V_SETPOINT. */
    *volt = r->voltageSetpoint * (r->n_cavities > 0 ? r->n_cavities : 1);
    *phaseDeg = r->phaseSetpoint;
    *freq = r->driveFrequency;
  } else
    bombElegantVA("fast_orbit_feedback: element %s is not a supported RF-phase actuator "
                  "(only RFCA with DELTA_PHASE or RFMODE with PHASE_SETPOINT).\n",
                  e->name ? e->name : "?");
}

/* ------------------------------------------------------------------ */
/* Append the RF-phase energy actuator as the last column of the fast response matrix
   CM->C.  Over a sub-synchrotron-period step a held phase offset gives a meaningful
   energy/orbit response linearized about the synchronous phase: a command Delta-phi (deg)
   yields delta = Delta-phi/phaseScale and orbit x_i = eta_i*delta, so the column is
   dx_i/dphi = eta_i/phaseScale (m/deg).  Scaled by s_phase for conditioning, with
   kick_coef=1/s_phase so the solved error reads out in degrees and sl_index=-2 as the
   phase sentinel.  Called only for the x (dispersive) plane when include_rf_phase is set;
   the caller inverts CM afterwards.  Mirrors buildRFMatrix's cavity discovery. */

static void appendPhaseActuator(FOFB_PLANE *plane, RUN *run, LINE_LIST *beamline) {
  CORMON_DATA *CM = &plane->CM;
  long nSteer = CM->ncor, nmon = CM->nmon, i, j, k;
  double *phasecol, sC = 0, sP = 0, rmsC, rmsP, s_phase;
  double sumVcosV, sumVabsV, voltScaledSum, cosEff, rf0freq = 0;
  MAT *Caug;

  /* resolve the cavity list: declared entries, else auto-add every RFCA (mutable
     copies required -- add_steer_elem_to_lists uppercases its args in place). */
  if (!fofbRFPhaseDeclared || fofbRFPhasesl.n_corr_types == 0) {
    char *anyName, *phaseItem, *rfcaType;
    cp_str(&anyName, "*");
    /* Actuate the cavity's additive DELTA_PHASE (base 0), not PHASE: this leaves PHASE as the
       clean synchronous setpoint (which &modulate_elements may perturb to inject phase errors)
       and lets the feedback correction compose additively on the same cavity. */
    cp_str(&phaseItem, "DELTA_PHASE");
    cp_str(&rfcaType, "RFCA");
    if (!add_steer_elem_to_lists(&fofbRFPhasesl, 0, anyName, phaseItem, rfcaType, 1e-6, 0,
                                 0, 0, 1, -1, -1, beamline, run, 0, verbosity > 1))
      bombElegant("fast_orbit_feedback: include_rf_phase set but no RFCA cavities found", NULL);
    free(anyName);
    free(phaseItem);
    free(rfcaType);
  }
  plane->nRFPhase = fofbRFPhasesl.n_corr_types;

  plane->rfPhaseElem = tmalloc(sizeof(*plane->rfPhaseElem) * plane->nRFPhase);
  plane->rfBasePhase = tmalloc(sizeof(*plane->rfBasePhase) * plane->nRFPhase);
  plane->rfPhaseOffset = tmalloc(sizeof(*plane->rfPhaseOffset) * plane->nRFPhase);
  for (k = 0; k < plane->nRFPhase; k++) {
    plane->rfPhaseElem[k] = fofbRFPhasesl.elem[k];
    plane->rfPhaseOffset[k] = fofbRFPhasesl.param_offset[k];
    plane->rfBasePhase[k] = *((double *)(fofbRFPhasesl.elem[k]->p_elem + fofbRFPhasesl.param_offset[k]));
  }

  /* Validate the actuator element types.  RFCA (DELTA_PHASE) and RFMODE (PHASE_SETPOINT) are
     supported; any other type would misread fields below and write a meaningless parameter.  An
     RFMODE only honors PHASE_SETPOINT when its generator/feedback path is active (DRIVE_FREQUENCY>0),
     so reject an inert one explicitly rather than let the loop silently lose authority. */
  for (k = 0; k < plane->nRFPhase; k++) {
    ELEMENT_LIST *e = plane->rfPhaseElem[k];
    if (e->type != T_RFCA && e->type != T_RFMODE)
      bombElegantVA("fast_orbit_feedback: element %s is not a supported RF-phase actuator "
                    "(only RFCA with DELTA_PHASE or RFMODE with PHASE_SETPOINT).\n",
                    e->name ? e->name : "?");
    if (e->type == T_RFMODE) {
      RFMODE *rf = (RFMODE *)e->p_elem;
      /* PHASE_SETPOINT is only consumed during tracking by RFMODE's generator/feedback path, and
         then only when phase feedback is actually running: it is re-read in the amplitude/phase
         branch (needs a PHASE_FILTER on top of an AMPLITUDE_FILTER) and in the I/Q branch (needs an
         IN_PHASE_FILTER or QUADRATURE_FILTER).  With the generator off (DRIVE_FREQUENCY<=0) or no
         phase feedback configured, the per-turn PHASE_SETPOINT write is silently ignored and the
         loop would run with phantom authority -- so reject/flag that up front. */
      long haveAmp = rf->amplitudeFilterFile && rf->amplitudeFilterFile[0];
      long havePhase = rf->phaseFilterFile && rf->phaseFilterFile[0];
      long haveIQ = (rf->IFilterFile && rf->IFilterFile[0]) || (rf->QFilterFile && rf->QFilterFile[0]);
      long phaseFB = haveAmp ? havePhase : haveIQ;
      if (rf->driveFrequency <= 0)
        bombElegantVA("fast_orbit_feedback: RFMODE %s is used as an RF-phase actuator but has "
                      "DRIVE_FREQUENCY<=0, so PHASE_SETPOINT has no effect (the generator path is "
                      "inactive).  Set DRIVE_FREQUENCY>0.\n",
                      e->name ? e->name : "?");
      if (!phaseFB)
        printWarning("fast_orbit_feedback: RFMODE RF-phase actuator has no phase feedback configured; "
                     "PHASE_SETPOINT may have no tracking effect.",
                     "PHASE_SETPOINT is actuated only by RFMODE's generator feedback -- give it a "
                     "PHASE_FILTER (with AMPLITUDE_FILTER) or an IN_PHASE_FILTER/QUADRATURE_FILTER, "
                     "or the feedback command is ignored and the loop runs with phantom authority.");
    }
  }

  plane->energyKp = energy_Kp;
  plane->energyKi = energy_Ki;
  plane->energyKd = energy_Kd;
  plane->deltaEst = 0;

  /* phaseScale: deg of RF phase per unit energy offset delta.  energy_response_scale,
     if given, is authoritative (signed, user-calibrated).  Otherwise form the analytic
     estimate (180/pi)*pCentral/(passes_between_updates*SUM_k[voltScaled_k*cos phi_s,k]).  The
     actuator drives the SAME phase offset on EVERY included cavity (RFCA DELTA_PHASE, or RFMODE
     PHASE_SETPOINT -- see fofbCavityDrive), so the energy authority per degree is the SUM of
     V*cos(phi_s) over all of them -- not one cavity's.  voltScaled matches the dgamma=volt*sin(phase)
     convention of simple_rfca.c (volt in gamma units) and pCentral is gamma0.  Each (V,phi) is the
     cavity's drive voltage and fiducialized synchronous phase phi_s (set by &rf_setup; for RFMODE,
     the V_SETPOINT/PHASE_SETPOINT targets its internal feedback drives the cavity to),
     so the summed cos carries the transition state and the sign is self-correcting for BOTH
     regimes (above transition phi_s in (90,180) deg so cos<0; below transition the reverse).
     Because phaseScale ~ 1/sum(V cos phi_s), the closed-loop damping rate works out to
     g = energy_Kp/passes_between_updates (voltScaled, gamma, slip factor and the cos all cancel),
     i.e. damping in either regime with no transition branch needed.  sumVcosV is the sum in raw
     volts (signed); voltScaledSum is it in gamma units; cosEff is the net cos(phi_s) normalized
     by the total voltage (reduces exactly to one cavity's cos(phi_s) in the single-cavity case).
     Failure modes: (a) operating AT transition, net cos->0, phaseScale->inf and the phase knob
     has no energy authority (nu_s->0); (b) a deck whose phases are not the true stable phases
     (PHASE set by hand, or no &rf_setup) -- then set energy_response_scale explicitly. */
  sumVcosV = sumVabsV = 0;
  for (k = 0; k < plane->nRFPhase; k++) {
    double vk = 0, phik = 0, fk = 0;
    fofbCavityDrive(plane->rfPhaseElem[k], &vk, &phik, &fk);
    sumVcosV += vk * cos(phik * PI / 180.0);
    sumVabsV += fabs(vk);
    if (k == 0)
      rf0freq = fk; /* common RF frequency, used for the synchrotron-period efficacy estimate */
  }
  voltScaledSum = sumVcosV / (1e6 * particleMassMV * particleRelSign);
  cosEff = (sumVabsV != 0) ? sumVcosV / sumVabsV : 0;
  if (energy_response_scale != 0) {
    plane->phaseScale = energy_response_scale;
  } else {
    double etaSlip;
    if (voltScaledSum != 0 && passes_between_updates > 0)
      plane->phaseScale = (180.0 / PI) * run->p_central /
                          ((double)passes_between_updates * voltScaledSum);
    else
      plane->phaseScale = 0;
    /* Resolve/validate the analytic sign from the slip factor the lattice already provides.
       Longitudinal stability requires eta_slip*cos(phi_s) < 0 (above transition eta_slip>0 so
       cos phi_s<0; below transition the reverse), and that is exactly the condition under
       which the analytic phaseScale is a damping -- rather than anti-damping -- gain.  So when
       the phases are the true synchronous phases the sign is automatically correct in either
       regime; when they are not (PHASE set by hand, or &rf_setup not run) the consistency check
       below fires.  cosEff is the volt-weighted net cos(phi_s) across the cavities.  Skipped when
       alpha_c is unavailable (no twiss), where the transition state cannot be determined. */
    etaSlip = beamline->alpha[0] - 1.0 / (run->p_central * run->p_central);
    if (beamline->alpha[0] != 0 && etaSlip * cosEff >= 0)
      printWarning("fast_orbit_feedback: the analytic RF-phase actuator sign looks wrong for "
                   "this lattice (it will anti-damp).",
                   "eta_slip*cos(phi_s) >= 0, so the cavity phase(s) are not the stable "
                   "synchronous phase -- set with &rf_setup, or give energy_response_scale "
                   "explicitly (deg per unit delta).");
    else if (fabs(cosEff) < 0.02)
      printWarning("fast_orbit_feedback: the synchronous phase is near 90 deg (cos(phi_s) ~ 0).",
                   "The RF-phase knob has little energy authority there (phaseScale is very "
                   "large); prefer the RF-frequency loop, or set energy_response_scale from a "
                   "calibration run.");
    else if (verbosity)
      printf("fast_orbit_feedback: RF-phase sign resolved from slip factor (eta_slip=%.4e, "
             "cos(phi_s)=%.4f over %ld cavity(ies), %s transition); analytic phaseScale is a "
             "damping gain.  Set energy_response_scale to refine the magnitude.\n",
             etaSlip, cosEff, plane->nRFPhase, etaSlip > 0 ? "above" : "below");
  }
  if (plane->phaseScale == 0)
    bombElegant("fast_orbit_feedback: RF-phase response scale is zero; set energy_response_scale "
                "(deg per unit delta) or check the cavity VOLT/PHASE and passes_between_updates.", NULL);

  /* efficacy warning: a phase offset held across a step loses its energy-damping action
     once the beam begins to re-phase, i.e. once the step spans a sizable fraction of the
     synchrotron period.  Estimate nu_s from the standard formula and warn if the step is
     long compared with it.  Best-effort only (skipped silently if the estimate is not a
     sane positive number). */
  {
    double h, Eev, nus2, Tsynch;
    h = rf0freq * beamline->revolution_length / c_mks; /* beta ~ 1; common RF frequency */
    Eev = run->p_central * particleMassMV * 1e6;
    /* nu_s^2 ~ alpha_c*h*|sum_k V_k cos phi_s,k|/(2 pi E): the restoring force is the SUM over
       all included cavities, matching the phaseScale sum above (sumVcosV is in raw volts). */
    nus2 = (Eev > 0) ? beamline->alpha[0] * h * fabs(sumVcosV) / (PIx2 * Eev)
                     : 0;
    if (nus2 > 0) {
      Tsynch = 1.0 / sqrt(nus2);
      if (passes_between_updates > Tsynch / 5.0)
        printWarning("fast_orbit_feedback: the RF-phase actuator step is long compared with "
                     "the synchrotron period.",
                     "passes_between_updates exceeds ~T_synch/5; a held phase offset loses "
                     "energy-damping efficacy as the beam re-phases.  Use a faster loop "
                     "(smaller passes_between_updates).");
    }
  }

  /* analytic per-degree energy/orbit column at the monitors */
  phasecol = tmalloc(sizeof(*phasecol) * nmon);
  for (i = 0; i < nmon; i++)
    phasecol[i] = (CM->umoni[i]->twiss ? CM->umoni[i]->twiss->etax : 0) / plane->phaseScale;

  /* conditioning scale so the phase column carries weight comparable to the correctors */
  for (j = 0; j < nSteer; j++)
    for (i = 0; i < nmon; i++)
      sC += Mij(CM->C, i, j) * Mij(CM->C, i, j);
  for (i = 0; i < nmon; i++)
    sP += phasecol[i] * phasecol[i];
  rmsC = nSteer ? sqrt(sC / ((double)nmon * nSteer)) : 1.0;
  rmsP = sqrt(sP / nmon);
  if (rf_response_scale > 0)
    s_phase = rf_response_scale;
  else
    s_phase = (rmsP > 0 && rmsC > 0) ? rmsC / rmsP : 1.0;
  plane->s_phase = s_phase;

  /* augment C with the scaled phase column as the last column */
  Caug = matrix_get(nmon, nSteer + 1);
  for (j = 0; j < nSteer; j++)
    memcpy(Caug->me[j], CM->C->me[j], sizeof(double) * nmon);
  for (i = 0; i < nmon; i++)
    Mij(Caug, i, nSteer) = phasecol[i] * s_phase;
  free(phasecol);
  matrix_free(CM->C);
  CM->C = Caug;

  /* extend the per-corrector CM arrays by one for the phase entry */
  CM->kick_coef = trealloc(CM->kick_coef, sizeof(*CM->kick_coef) * (nSteer + 1));
  CM->sl_index = trealloc(CM->sl_index, sizeof(*CM->sl_index) * (nSteer + 1));
  CM->ucorr = trealloc(CM->ucorr, sizeof(*CM->ucorr) * (nSteer + 1));
  CM->pegged = trealloc(CM->pegged, sizeof(*CM->pegged) * (nSteer + 1));
  CM->kick_coef[nSteer] = 1.0 / s_phase; /* e = Mij(dK,phase)/kick_coef is then degrees */
  CM->sl_index[nSteer] = -2;             /* RF-phase sentinel */
  CM->ucorr[nSteer] = plane->rfPhaseElem[0];
  CM->pegged[nSteer] = 0;
  CM->ncor = nSteer + 1;
  plane->phaseIndex = nSteer;
  plane->rfPhaseActive = 1;

  if (verbosity)
    printf("fast_orbit_feedback: RF-phase energy actuator active over %ld cavity(ies), "
           "phaseScale=%.6e deg/delta%s, s_phase=%.3e (fast-matrix column)\n",
           plane->nRFPhase, plane->phaseScale,
           energy_response_scale != 0 ? " (user)" : " (analytic)", s_phase);
  fflush(stdout);
}

/* ------------------------------------------------------------------ */
/* Response-matrix file I/O.  FOFB can LOAD the forward steering response matrix from an
   SDDS file (response[0]/response[1]) or WRITE the one it computed (response_output[0]/[1]),
   in the exact layout &correction_matrix_output produces for the default (unscaled) units:
   a BPMName STRING column, an s DOUBLE column (monitor location), and one DOUBLE column per
   corrector named by the corrector's unique element name.  The cell value is
   Mij(C,i,j) = dX(monitor i)/dK(corrector j).  Loading applies to the steering correctors
   only; a loaded matrix is always inverted internally (fofbInvertCM), so no inverse-matrix
   loading is provided. */

/* Build display labels for a set of elements, matching &correction_matrix_output's naming
   rule (response.c, full_names=0): the plain element name when it is unique within the set,
   otherwise name#occurence.  Returns a malloc'd array of malloc'd strings. */
static char **fofbElementLabels(ELEMENT_LIST **list, long n) {
  char **label;
  long i, j;
  char buf[1024];
  label = tmalloc(sizeof(*label) * (n > 0 ? n : 1));
  for (i = 0; i < n; i++) {
    short unique = 1;
    for (j = 0; j < n; j++)
      if (j != i && strcmp(list[i]->name, list[j]->name) == 0) {
        unique = 0;
        break;
      }
    if (unique)
      cp_str(&label[i], list[i]->name);
    else {
      snprintf(buf, sizeof(buf), "%s#%ld", list[i]->name, list[i]->occurence);
      cp_str(&label[i], buf);
    }
  }
  return label;
}

static void fofbFreeLabels(char **label, long n) {
  long i;
  for (i = 0; i < n; i++)
    free(label[i]);
  free(label);
}

/* Write the forward steering response matrix (CM->C, dX_i/dK_j, m per kick-radian) to an
   SDDS file in &correction_matrix_output "Response" layout.  Master-only under MPI.  Must
   be called with the steering-only CM (before any RF-phase column is appended). */
static void fofbWriteResponseMatrix(CORMON_DATA *CM, char *filename, RUN *run, long coord) {
  SDDS_DATASET SDDSout;
  char **monLabel, **corLabel;
  long i, j, nameIndex = -1, sIndex = -1, *corIndex;
  char desc[256];
  if (!FOFB_IS_MASTER)
    return;
  filename = compose_filename(filename, run->rootname);
  monLabel = fofbElementLabels(CM->umoni, CM->nmon);
  corLabel = fofbElementLabels(CM->ucorr, CM->ncor);
  snprintf(desc, sizeof(desc), "%s-plane fast_orbit_feedback steering response matrix",
           coord == 0 ? "horizontal" : "vertical");
  if (!SDDS_InitializeOutputElegant(&SDDSout, SDDS_BINARY, 1, desc, desc, filename))
    SDDS_Bomb("fast_orbit_feedback: unable to initialize response_output file");
  if ((nameIndex = SDDS_DefineColumn(&SDDSout, "BPMName", NULL, NULL,
                                     "beam-position-monitor name", NULL, SDDS_STRING, 0)) < 0 ||
      (sIndex = SDDS_DefineColumn(&SDDSout, "s", NULL, "m",
                                  "beam-position-monitor location", NULL, SDDS_DOUBLE, 0)) < 0)
    SDDS_Bomb("fast_orbit_feedback: unable to define response_output columns");
  SDDS_DefineParameter(&SDDSout, "CorrectionMatrixType", NULL, NULL, "correction matrix type",
                       NULL, SDDS_STRING, "Response");
  SDDS_DefineParameter1(&SDDSout, "NCorrectors", NULL, NULL, "Number of correctors", NULL,
                        SDDS_LONG, &CM->ncor);
  SDDS_DefineParameter1(&SDDSout, "NMonitors", NULL, NULL, "Number of monitors", NULL,
                        SDDS_LONG, &CM->nmon);
  SDDS_DefineParameter(&SDDSout, "Stage", NULL, NULL, "Simulation stage", NULL, SDDS_STRING,
                       "fast_orbit_feedback response matrix");
  corIndex = tmalloc(sizeof(*corIndex) * (CM->ncor > 0 ? CM->ncor : 1));
  for (j = 0; j < CM->ncor; j++)
    if ((corIndex[j] = SDDS_DefineColumn(&SDDSout, corLabel[j], NULL, "m", "response",
                                         NULL, SDDS_DOUBLE, 0)) < 0)
      SDDS_Bomb("fast_orbit_feedback: unable to define response_output corrector column");
  if (!SDDS_WriteLayout(&SDDSout) || !SDDS_StartPage(&SDDSout, CM->nmon))
    SDDS_Bomb("fast_orbit_feedback: unable to start response_output page");
  for (i = 0; i < CM->nmon; i++) {
    if (!SDDS_SetRowValues(&SDDSout, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, i,
                           nameIndex, monLabel[i], sIndex, CM->umoni[i]->end_pos, -1))
      SDDS_Bomb("fast_orbit_feedback: unable to set response_output monitor row");
    for (j = 0; j < CM->ncor; j++)
      if (!SDDS_SetRowValues(&SDDSout, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, i,
                             corIndex[j], (double)Mij(CM->C, i, j), -1))
        SDDS_Bomb("fast_orbit_feedback: unable to set response_output matrix value");
  }
  if (!SDDS_WritePage(&SDDSout) || !SDDS_Terminate(&SDDSout))
    SDDS_Bomb("fast_orbit_feedback: unable to write response_output file");
  if (verbosity)
    printf("fast_orbit_feedback: wrote %s-plane forward response matrix (%ld monitors x %ld correctors) to %s\n",
           coord == 0 ? "horizontal" : "vertical", (long)CM->nmon, (long)CM->ncor, filename);
  fflush(stdout);
  free(corIndex);
  fofbFreeLabels(monLabel, CM->nmon);
  fofbFreeLabels(corLabel, CM->ncor);
}

/* Load the forward steering response matrix from an SDDS file into CM->C, overwriting the
   analytically computed entries.  Rows are matched to this plane's monitors by BPMName and
   columns to its correctors by unique element name (order-independent); any dimension or
   name mismatch is fatal.  Runs on all ranks (read-only), so every rank's subsequent
   inversion is consistent. */
static void fofbLoadResponseMatrix(CORMON_DATA *CM, char *filename, RUN *run, long coord) {
  SDDS_DATASET SDDSin;
  char **bpmName, **monLabel, **corLabel;
  long i, j, r, nrows, *rowOf;
  double *col;
  char msg[1024];
  filename = compose_filename(filename, run->rootname);
  if (!SDDS_InitializeInputFromSearchPath(&SDDSin, filename)) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    snprintf(msg, sizeof(msg), "fast_orbit_feedback: unable to open response file %s", filename);
    bombElegant(msg, NULL);
  }
  if (SDDS_ReadPage(&SDDSin) <= 0)
    bombElegant("fast_orbit_feedback: response file has no data page", NULL);
  nrows = SDDS_RowCount(&SDDSin);
  if (nrows != CM->nmon) {
    snprintf(msg, sizeof(msg),
             "fast_orbit_feedback: response file has %ld rows but the %s plane has %ld monitors",
             nrows, coord == 0 ? "horizontal" : "vertical", (long)CM->nmon);
    bombElegant(msg, NULL);
  }
  if (SDDS_CheckColumn(&SDDSin, "BPMName", NULL, SDDS_STRING, stderr) != SDDS_CHECK_OKAY)
    bombElegant("fast_orbit_feedback: response file lacks a BPMName string column", NULL);
  if (!(bpmName = (char **)SDDS_GetColumn(&SDDSin, "BPMName")))
    bombElegant("fast_orbit_feedback: unable to read BPMName column from response file", NULL);
  monLabel = fofbElementLabels(CM->umoni, CM->nmon);
  corLabel = fofbElementLabels(CM->ucorr, CM->ncor);
  /* map each plane monitor to its row in the file by BPMName */
  rowOf = tmalloc(sizeof(*rowOf) * (CM->nmon > 0 ? CM->nmon : 1));
  for (i = 0; i < CM->nmon; i++) {
    rowOf[i] = -1;
    for (r = 0; r < nrows; r++)
      if (bpmName[r] && strcmp(bpmName[r], monLabel[i]) == 0) {
        rowOf[i] = r;
        break;
      }
    if (rowOf[i] < 0) {
      snprintf(msg, sizeof(msg),
               "fast_orbit_feedback: monitor %s not found in response file BPMName column", monLabel[i]);
      bombElegant(msg, NULL);
    }
  }
  /* copy each corrector column, matched by unique element name */
  for (j = 0; j < CM->ncor; j++) {
    if (SDDS_CheckColumn(&SDDSin, corLabel[j], NULL, SDDS_ANY_NUMERIC_TYPE, stderr) != SDDS_CHECK_OKAY) {
      snprintf(msg, sizeof(msg),
               "fast_orbit_feedback: corrector column %s not found (numeric) in response file", corLabel[j]);
      bombElegant(msg, NULL);
    }
    if (!(col = SDDS_GetColumnInDoubles(&SDDSin, corLabel[j]))) {
      snprintf(msg, sizeof(msg),
               "fast_orbit_feedback: unable to read column %s from response file", corLabel[j]);
      bombElegant(msg, NULL);
    }
    for (i = 0; i < CM->nmon; i++)
      Mij(CM->C, i, j) = col[rowOf[i]];
    free(col);
  }
  if (verbosity)
    printf("fast_orbit_feedback: loaded %s-plane forward response matrix (%ld monitors x %ld correctors) from %s\n",
           coord == 0 ? "horizontal" : "vertical", (long)CM->nmon, (long)CM->ncor, filename);
  fflush(stdout);
  SDDS_FreeStringArray(bpmName, nrows);
  free(bpmName);
  fofbFreeLabels(monLabel, CM->nmon);
  fofbFreeLabels(corLabel, CM->ncor);
  free(rowOf);
  SDDS_Terminate(&SDDSin);
}

/* ------------------------------------------------------------------ */

static void setupPlane(FOFB_PLANE *plane, long coord, RUN *run, VARY *control, LINE_LIST *beamline) {
  CORMON_DATA *CM;
  STEERING_LIST *SL;
  long i, found = 0, idx = (coord == 0) ? 0 : 1, rfActive, phaseActive, responseLoad;
  unsigned long flags;
  char *item;

  memset(plane, 0, sizeof(*plane));
  plane->coord = coord;
  plane->phaseIndex = -1;
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

  /* the RF-frequency and RF-phase actuators are horizontal (dispersive) knobs only */
  rfActive = (coord == 0 && include_rf_frequency);
  phaseActive = (coord == 0 && include_rf_phase);
  /* a forward steering matrix may be loaded from file for this plane */
  responseLoad = (response[idx] != NULL && !SDDS_StringIsBlank(response[idx]));

  if (!found && !rfActive && !phaseActive) {
    printWarning("No steering correctors found for a plane.",
                 coord == 0 ? "fast_orbit_feedback x plane inactive." : "fast_orbit_feedback y plane inactive.");
    plane->active = 0;
    return;
  }

  /* SV / inversion controls consumed by compute_orbcor_matrices */
  CM->nmon = CM->ncor = 0;
  CM->C = CM->T = NULL;
  /* fixed_length is the correction-time orbit model (unused: FOFB tracks the beam and
     never runs a closed-orbit correction step). fixed_length_matrix selects the orbit
     model used to build the response matrix (0 vary RF frequency, 1 vary energy,
     2 full 6-D); honored by both the analytic and computed-orbit builders below. */
  CM->fixed_length = 0;
  CM->fixed_length_matrix = fixed_length_matrix;
  CM->auto_limit_SVs = 1;
  CM->keep_largest_SVs = keep_largest_SVs;
  CM->remove_smallest_SVs = remove_smallest_SVs;
  CM->minimum_SV_ratio = minimum_SV_ratio;
  CM->Tikhonov_relative_alpha = Tikhonov_relative_alpha;
  CM->Tikhonov_n = Tikhonov_n;

  /* When the RF-phase actuator is active we append its column to the fast matrix before
     inverting, and when a forward matrix is loaded from file we overwrite CM->C before
     inverting, so in both cases defer the inversion; otherwise invert here.  (The
     RF-frequency actuator now has its own separate matrix and does NOT affect the fast
     inversion.) */
  flags = ((phaseActive || responseLoad) ? 0 : COMPUTE_RESPONSE_INVERT) |
          (verbosity < 2 ? COMPUTE_RESPONSE_SILENT : 0);
  if (use_response_from_computed_orbits) {
    /* Build the response matrix by tweaking each corrector and differencing perturbed
       closed orbits (mirrors &correct use_response_from_computed_orbits). Reuse the
       serial per-plane builder compute_orbcor_matrices1 on all ranks: it is compiled
       unconditionally, matrix-based (single test particle) for closed_orbit_tracking_turns=0,
       and MPI-safe run redundantly -- the batched compute_orbcor_matrices1p does both
       planes in one collective call, incompatible with FOFB's per-plane + x-only RF-append
       flow. A local CORRECTION carries only the closed-orbit knobs and this plane's CM/SL. */
    CORRECTION corrTmp;
    memset(&corrTmp, 0, sizeof(corrTmp));
    corrTmp.clorb_accuracy = closed_orbit_accuracy;
    corrTmp.clorb_accuracy_requirement = closed_orbit_accuracy_requirement;
    corrTmp.clorb_iterations = closed_orbit_iterations;               /* struct field is double */
    corrTmp.clorb_iter_fraction = closed_orbit_iteration_fraction;
    corrTmp.clorb_fraction_multiplier = closed_orbit_fraction_multiplier;
    corrTmp.clorb_multiplier_interval = closed_orbit_multiplier_interval; /* struct field is short */
    corrTmp.clorb_track_for_orbit = closed_orbit_tracking_turns;      /* struct field is short */
    corrTmp.rpn_store_response_matrix = 0;
    /* shallow struct copy: SL's pointer members are read-only during response
       computation, and corrTmp is never freed. */
    if (coord == 0) {
      corrTmp.CMFx = CM;
      corrTmp.SLx = *SL;
    } else {
      corrTmp.CMFy = CM;
      corrTmp.SLy = *SL;
    }
    compute_orbcor_matrices1(&corrTmp, coord, run, beamline, flags, 0, NULL, NULL);
  } else {
    compute_orbcor_matrices(CM, SL, coord, run, beamline, flags, 0);
  }

  if (CM->nmon == 0) {
    plane->active = 0;
    return;
  }

  /* optionally overwrite the analytic steering matrix with one loaded from file, and/or
     dump the steering matrix (before any RF-phase column is appended) for inspection/reload */
  if (responseLoad)
    fofbLoadResponseMatrix(CM, response[idx], run, coord);
  if (response_output[idx] != NULL && !SDDS_StringIsBlank(response_output[idx]))
    fofbWriteResponseMatrix(CM, response_output[idx], run, coord);

  /* fast matrix: append the RF-phase column (if any) then invert.  The RF-frequency knob
     is built as a separate slow-cadence matrix that leaves CM->C/T untouched.  A loaded
     matrix with no phase column is inverted here. */
  if (phaseActive) {
    appendPhaseActuator(plane, run, beamline);
    fofbInvertCM(CM);
  } else if (responseLoad) {
    fofbInvertCM(CM);
  }
  if (rfActive)
    buildRFMatrix(plane, run, beamline);

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
    if (sl_index == -2) {
      /* RF-phase column: base setpoint is the reference cavity PHASE (deg).  With no
         rf_phase_filter_file the setpoint is written immediately each turn (direct PHASE
         write); with one, the written phase ramps toward the setpoint through this
         per-actuator IIR bank (its own state), modelling a finite-bandwidth phase shifter. */
      plane->kickOffset[i] = plane->rfPhaseOffset[0];
      plane->base[i] = plane->rfBasePhase[0];
      plane->actFilter[i] = loadFilterBank(rf_phase_filter_file, &plane->nActFilter[i]);
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

  /* flush_interval is the number of feedback steps between in-place flushes of the
     single-page output file to disk (WATCH-style); < 1 means flush only at the end. */
  if (flush_interval < 1)
    flush_interval = 0;

  /* feedback cadence: run_control n_passes is no longer used by FOFB.  passes_between_updates
     (turns tracked between fast updates) is required; the slow RF-frequency loop updates every
     rf_update_multiplier fast iterations. */
  if (passes_between_updates <= 0)
    bombElegant("fast_orbit_feedback: passes_between_updates must be > 0 (the number of turns tracked "
                "between fast feedback updates).  run_control n_passes is no longer used by FOFB.", NULL);
  if (rf_update_multiplier < 1)
    bombElegant("fast_orbit_feedback: rf_update_multiplier must be >= 1 (the slow RF-frequency loop "
                "updates every rf_update_multiplier fast iterations).", NULL);
  if (rf_update_multiplier > 1 && !include_rf_frequency)
    printWarning("fast_orbit_feedback: rf_update_multiplier > 1 but include_rf_frequency is 0.",
                 "rf_update_multiplier only paces the slow RF-frequency loop; it has no effect here.");
  /* A &steering_element can declare the RF cavities for the frequency/phase actuators, but the
     actuator itself is built only when its include_rf_* flag is set (setupPlane gates buildRFMatrix
     on include_rf_frequency and appendPhaseActuator on include_rf_phase).  Declaring the cavities
     without enabling the actuator silently ignores the declaration -- warn so the mismatch is seen. */
  if (fofbRFDeclared && !include_rf_frequency)
    printWarning("fast_orbit_feedback: steering_element item=FREQ cavities were declared but include_rf_frequency is 0.",
                 "The RF-frequency actuator is not built, so those cavity declarations are ignored.  "
                 "Set include_rf_frequency=1 to drive them, or remove the item=FREQ steering_element.");
  if (fofbRFPhaseDeclared && !include_rf_phase)
    printWarning("fast_orbit_feedback: steering_element item=PHASE cavities were declared but include_rf_phase is 0.",
                 "The RF-phase energy actuator is not built, so those cavity declarations are ignored.  "
                 "Set include_rf_phase=1 to drive them, or remove the item=PHASE steering_element.");
  /* Each actuator-lag filter is loaded only where its actuator is built; warn if one is set
     without its enabling flag so a silently-ignored filter file is seen. */
  if (rf_phase_filter_file && !include_rf_phase)
    printWarning("fast_orbit_feedback: rf_phase_filter_file is set but include_rf_phase is 0.",
                 "The RF-phase actuator is not built, so the phase filter file is ignored.  "
                 "Set include_rf_phase=1 to use it.");
  if (rf_frequency_filter_file && !include_rf_frequency)
    printWarning("fast_orbit_feedback: rf_frequency_filter_file is set but include_rf_frequency is 0.",
                 "The RF-frequency actuator is not built, so the frequency filter file is ignored.  "
                 "Set include_rf_frequency=1 to use it.");

  /* response-matrix orbit model (mirrors &correct fixed_length_matrix guards) */
  if (fixed_length_matrix < 0 || fixed_length_matrix > 2)
    bombElegant("fixed_length_matrix must be 0 (vary RF frequency), 1 (vary energy, momentum secant), or 2 (full 6-D closed orbit).", NULL);
  if (fixed_length_matrix == 1 && checkChangeT(beamline))
    bombElegant("change_t is nonzero on one or more RF cavities. This is incompatible with fixed_length_matrix=1 orbit computations.", NULL);
  if (fixed_length_matrix && include_rf_frequency)
    printWarning("fast_orbit_feedback: fixed_length_matrix is set together with include_rf_frequency.",
                 "The RF-frequency actuator moves the orbit by varying the RF frequency, which is inconsistent with a fixed-length response matrix. Use fixed_length_matrix=0 (variable-length, vary-RF-frequency model) or disable include_rf_frequency.");
  /* Gated on include_rf_frequency so this does not contradict the warning just above: when the
     RF-frequency actuator is active the loop really does vary the rf frequency, and a
     variable-length matrix is then the right model. */
  if (fixed_length_matrix == 0 && !include_rf_frequency)
    printWarning((char *)"fast_orbit_feedback: fixed_length_matrix=0 is unphysical for orbit feedback.",
                 (char *)"A fixed_length_matrix=0 response matrix lets the orbit length vary, which "
                 "corresponds to the rf frequency changing in response to each corrector.  The feedback "
                 "does not do that, so the predicted corrector-to-orbit response does not match the "
                 "machine and the loop may converge poorly or go unstable.  Use fixed_length_matrix=2 "
                 "(or 1).");
  /* subtract_dispersion is deprecated: the dispersive/betatron partition is now automatic.
     With an RF-phase column the deadbeat solve assigns the eta-shaped energy orbit to the
     phase knob; without one but with the RF-frequency loop active, the corrector solve is
     made eta-blind automatically (dispersion projected out of Q in updateSetpoints).  Either
     way the manual subtraction is obsolete. */
  if (subtract_dispersion)
    printWarning("fast_orbit_feedback: subtract_dispersion is deprecated and now a no-op.",
                 "The dispersive orbit is partitioned automatically: the RF-phase column owns "
                 "it when include_rf_phase is set, otherwise the correctors are made eta-blind "
                 "whenever the RF-frequency loop is active.");
  /* rf_bpm_filter_file gives the RF-frequency loop its own BPM filter (to notch the
     synchrotron line that destabilizes the frequency integrator at any gain) -- it only
     has meaning for that loop. */
  if (rf_bpm_filter_file && !include_rf_frequency)
    bombElegant("fast_orbit_feedback: rf_bpm_filter_file is set but include_rf_frequency is 0. "
                "The dedicated RF-loop BPM filter only applies to the RF-frequency actuator; "
                "enable include_rf_frequency or remove rf_bpm_filter_file.", NULL);
  if (use_response_from_computed_orbits) {
    if (closed_orbit_accuracy <= 0)
      bombElegant("closed_orbit_accuracy must be > 0", NULL);
    if (closed_orbit_accuracy_requirement <= 0)
      bombElegant("closed_orbit_accuracy_requirement must be > 0", NULL);
    if (closed_orbit_iteration_fraction <= 0 || closed_orbit_iteration_fraction > 1)
      bombElegant("closed_orbit_iteration_fraction must be on (0, 1]", NULL);
    if (closed_orbit_iterations <= 0)
      bombElegant("closed_orbit_iterations <= 0", NULL);
  }

  /* forward steering response-matrix loading (response[0]=horizontal, response[1]=vertical).
     Only these two planes exist for FOFB; a loaded matrix is always inverted internally,
     so there is no inverse-matrix loading and the cross-plane slots are unused. */
  if ((response[2] != NULL && !SDDS_StringIsBlank(response[2])) ||
      (response[3] != NULL && !SDDS_StringIsBlank(response[3])) ||
      (response_output[2] != NULL && !SDDS_StringIsBlank(response_output[2])) ||
      (response_output[3] != NULL && !SDDS_StringIsBlank(response_output[3])))
    bombElegant("fast_orbit_feedback: only response[0]/response_output[0] (horizontal) and "
                "response[1]/response_output[1] (vertical) are supported; cross-plane response "
                "matrices are not used by FOFB.", NULL);
  if (((response[0] != NULL && !SDDS_StringIsBlank(response[0])) ||
       (response[1] != NULL && !SDDS_StringIsBlank(response[1]))) &&
      use_response_from_computed_orbits)
    bombElegant("fast_orbit_feedback: response (loaded forward matrix) and "
                "use_response_from_computed_orbits are mutually exclusive.", NULL);

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

  setupPlane(&planeData[0], 0, run, control, beamline);
  setupPlane(&planeData[1], 2, run, control, beamline);

  if (!planeData[0].active && !planeData[1].active)
    bombElegant("fast_orbit_feedback: no active correction plane (no monitors and/or correctors found)", NULL);

  if (include_rf_phase && !planeData[0].rfPhaseActive)
    bombElegant("fast_orbit_feedback: include_rf_phase is set but the x plane is inactive (no monitors). "
                "The RF-phase energy actuator rides on the horizontal BPM set; declare H correctors "
                "(or include_rf_frequency) so the x plane is built.", NULL);

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
    /* Record every &fast_orbit_feedback setting as an SDDS parameter so the output file
       is self-documenting and the run is reproducible.  Values are filled in once below
       (SDDS_SetParameters after SDDS_StartPage).  These are always defined, independent of
       which optional loops are enabled. */
    if (!SDDS_DefineSimpleParameter(&SDDS_fofb, "passes_between_updates", NULL, SDDS_LONG) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "rf_update_multiplier", NULL, SDDS_LONG) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "flush_interval", NULL, SDDS_LONG) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "verbosity", NULL, SDDS_LONG) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "fixed_length_matrix", NULL, SDDS_LONG) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "use_response_from_computed_orbits", NULL, SDDS_LONG) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "response0", NULL, SDDS_STRING) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "response1", NULL, SDDS_STRING) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "response2", NULL, SDDS_STRING) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "response3", NULL, SDDS_STRING) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "response_output0", NULL, SDDS_STRING) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "response_output1", NULL, SDDS_STRING) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "response_output2", NULL, SDDS_STRING) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "response_output3", NULL, SDDS_STRING) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "closed_orbit_accuracy", "m", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "closed_orbit_accuracy_requirement", "m", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "closed_orbit_iterations", NULL, SDDS_LONG) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "closed_orbit_iteration_fraction", NULL, SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "closed_orbit_fraction_multiplier", NULL, SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "closed_orbit_multiplier_interval", NULL, SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "closed_orbit_tracking_turns", NULL, SDDS_LONG) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "keep_largest_SVs", NULL, SDDS_LONG) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "remove_smallest_SVs", NULL, SDDS_LONG) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "minimum_SV_ratio", NULL, SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "Tikhonov_relative_alpha", NULL, SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "Tikhonov_n", NULL, SDDS_LONG) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "steering_Kp", NULL, SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "steering_Ki", NULL, SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "steering_Kd", NULL, SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "corrector_limit", "rad", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "include_rf_frequency", NULL, SDDS_LONG) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "rf_Kp", NULL, SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "rf_Ki", NULL, SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "rf_Kd", NULL, SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "rf_frequency_limit", "Hz", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "rf_response_scale", NULL, SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "include_rf_phase", NULL, SDDS_LONG) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "energy_Kp", NULL, SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "energy_Ki", NULL, SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "energy_Kd", NULL, SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "rf_phase_limit", "deg", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "energy_response_scale", NULL, SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "subtract_dispersion", NULL, SDDS_LONG) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "anti_windup", NULL, SDDS_LONG) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "reset_filters_each_step", NULL, SDDS_LONG) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "bpm_noise", "m", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "bpm_noise_cutoff", "m", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "bpm_noise_distribution", NULL, SDDS_STRING) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "bpm_filter_file", NULL, SDDS_STRING) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "steering_filter_file", NULL, SDDS_STRING) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "rf_phase_filter_file", NULL, SDDS_STRING) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "rf_frequency_filter_file", NULL, SDDS_STRING) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "rf_bpm_filter_file", NULL, SDDS_STRING) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "output", NULL, SDDS_STRING) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "center_on_orbit", NULL, SDDS_LONG) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "center_momentum_also", NULL, SDDS_LONG) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "offset_by_orbit", NULL, SDDS_LONG) ||
        !SDDS_DefineSimpleParameter(&SDDS_fofb, "offset_momentum_also", NULL, SDDS_LONG)) {
      SDDS_SetError("Unable to define fast_orbit_feedback setting parameters");
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
    /* RF-phase energy-actuator diagnostics exist only when include_rf_phase is set, so
       include_rf_phase=0 runs keep the output layout unchanged */
    if (include_rf_phase &&
        (SDDS_DefineColumn(&SDDS_fofb, "DeltaPhase", NULL, "deg",
                           "Applied RF-phase offset from the base phase at this iteration "
                           "(one common offset shared by all selected cavities), the energy actuator command",
                           NULL, SDDS_DOUBLE, 0) < 0 ||
         SDDS_DefineColumn(&SDDS_fofb, "deltaEnergyEst", NULL, NULL,
                           "Dispersion-projected common energy offset sum(x_i*eta_i)/sum(eta_i^2) "
                           "estimated from the horizontal BPMs at this iteration",
                           NULL, SDDS_DOUBLE, 0) < 0)) {
      SDDS_SetError("Unable to set up fast_orbit_feedback RF-phase columns");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    if (!SDDS_WriteLayout(&SDDS_fofb)) {
      SDDS_SetError("Unable to write fast_orbit_feedback output layout");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    /* The whole run is written as ONE page (one row per step), updated in place
       periodically during tracking (see doFastOrbitFeedback / finishFastOrbitFeedback),
       so start the page once here sized to the full step count and set the page-constant
       monitor/corrector counts once.  planeData[] is populated by the setupPlane calls above. */
    if (!SDDS_StartPage(&SDDS_fofb, control->n_steps > 0 ? control->n_steps : 1) ||
        !SDDS_SetParameters(&SDDS_fofb, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                            "nMonitorsX", (int32_t)planeData[0].CM.nmon, "nCorrectorsX", (int32_t)planeData[0].CM.ncor,
                            "nMonitorsY", (int32_t)planeData[1].CM.nmon, "nCorrectorsY", (int32_t)planeData[1].CM.ncor, NULL)) {
      SDDS_SetError("Unable to start fast_orbit_feedback output page");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    /* Fill in the recorded settings (defined above).  LONG parameters are passed as int32_t
       and STRING parameters guard NULL with "".  Split across calls only for readability. */
    if (!SDDS_SetParameters(&SDDS_fofb, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
            "passes_between_updates", (int32_t)passes_between_updates,
            "rf_update_multiplier", (int32_t)rf_update_multiplier,
            "flush_interval", (int32_t)flush_interval,
            "verbosity", (int32_t)verbosity,
            "fixed_length_matrix", (int32_t)fixed_length_matrix,
            "use_response_from_computed_orbits", (int32_t)use_response_from_computed_orbits,
            "response0", response[0] ? response[0] : "",
            "response1", response[1] ? response[1] : "",
            "response2", response[2] ? response[2] : "",
            "response3", response[3] ? response[3] : "",
            "response_output0", response_output[0] ? response_output[0] : "",
            "response_output1", response_output[1] ? response_output[1] : "",
            "response_output2", response_output[2] ? response_output[2] : "",
            "response_output3", response_output[3] ? response_output[3] : "",
            "closed_orbit_accuracy", closed_orbit_accuracy,
            "closed_orbit_accuracy_requirement", closed_orbit_accuracy_requirement,
            "closed_orbit_iterations", (int32_t)closed_orbit_iterations,
            "closed_orbit_iteration_fraction", closed_orbit_iteration_fraction,
            "closed_orbit_fraction_multiplier", closed_orbit_fraction_multiplier,
            "closed_orbit_multiplier_interval", closed_orbit_multiplier_interval,
            "closed_orbit_tracking_turns", (int32_t)closed_orbit_tracking_turns, NULL) ||
        !SDDS_SetParameters(&SDDS_fofb, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
            "keep_largest_SVs", (int32_t)keep_largest_SVs,
            "remove_smallest_SVs", (int32_t)remove_smallest_SVs,
            "minimum_SV_ratio", minimum_SV_ratio,
            "Tikhonov_relative_alpha", Tikhonov_relative_alpha,
            "Tikhonov_n", (int32_t)Tikhonov_n,
            "steering_Kp", steering_Kp, "steering_Ki", steering_Ki, "steering_Kd", steering_Kd,
            "corrector_limit", corrector_limit,
            "include_rf_frequency", (int32_t)include_rf_frequency,
            "rf_Kp", rf_Kp, "rf_Ki", rf_Ki, "rf_Kd", rf_Kd,
            "rf_frequency_limit", rf_frequency_limit,
            "rf_response_scale", rf_response_scale,
            "include_rf_phase", (int32_t)include_rf_phase,
            "energy_Kp", energy_Kp, "energy_Ki", energy_Ki, "energy_Kd", energy_Kd,
            "rf_phase_limit", rf_phase_limit,
            "energy_response_scale", energy_response_scale,
            "subtract_dispersion", (int32_t)subtract_dispersion, NULL) ||
        !SDDS_SetParameters(&SDDS_fofb, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
            "anti_windup", (int32_t)anti_windup,
            "reset_filters_each_step", (int32_t)reset_filters_each_step,
            "bpm_noise", bpm_noise, "bpm_noise_cutoff", bpm_noise_cutoff,
            "bpm_noise_distribution", bpm_noise_distribution ? bpm_noise_distribution : "",
            "bpm_filter_file", bpm_filter_file ? bpm_filter_file : "",
            "steering_filter_file", steering_filter_file ? steering_filter_file : "",
            "rf_phase_filter_file", rf_phase_filter_file ? rf_phase_filter_file : "",
            "rf_frequency_filter_file", rf_frequency_filter_file ? rf_frequency_filter_file : "",
            "rf_bpm_filter_file", rf_bpm_filter_file ? rf_bpm_filter_file : "",
            "output", output ? output : "",
            "center_on_orbit", (int32_t)center_on_orbit,
            "center_momentum_also", (int32_t)center_momentum_also,
            "offset_by_orbit", (int32_t)offset_by_orbit,
            "offset_momentum_also", (int32_t)offset_momentum_also, NULL)) {
      SDDS_SetError("Unable to set fast_orbit_feedback setting parameters");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    fofbOutputActive = 1;
    fofbHasDfrfColumn = include_rf_frequency ? 1 : 0;
    fofbHasPhaseColumns = include_rf_phase ? 1 : 0;
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
        applied = plane->base[i] + applyIIRFilterCascade(plane->actFilter[i], plane->nActFilter[i], delta);
      else
        applied = plane->u[i]; /* no step response -> setpoint applied immediately */
      if (applied == plane->lastApplied[i])
        continue;

      if (i == plane->phaseIndex) {
        /* RF-phase column (NOT a corrector): write the cavity's DELTA_PHASE = rfBasePhase[k] +
           offset on every selected cavity, where offset = applied - base[i] (deg) and
           rfBasePhase[k] is the initial DELTA_PHASE (0).  DELTA_PHASE is an additive offset on
           top of PHASE (simple_rfca.c), so this composes with any phase error injected on PHASE
           via &modulate_elements.  Does NOT touch phase_fiducial (independent of DELTA_PHASE) and
           does NOT set changed=1: the RFCA live-reads DELTA_PHASE each pass (simple_rfca.c), so no
           compute_matrix is needed.  With no rf_phase_filter_file applied==u[i]; with one the
           written offset ramps toward u[i].  Either way u[i]==base[i]==0 on step 0, so the IIR
           input (delta=0) is zero and pass 0 writes DELTA_PHASE=0 -- fiducialization still
           happens at the clean PHASE. */
        double offset = applied - plane->base[i];
        long k;
        for (k = 0; k < plane->nRFPhase; k++)
          *((double *)(plane->rfPhaseElem[k]->p_elem + plane->rfPhaseOffset[k])) =
            plane->rfBasePhase[k] + offset;
        plane->lastApplied[i] = applied;
        continue; /* no compute_matrix / assert_element_links for the phase entry */
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

    /* standalone RF-frequency knob (NOT in CM->ncor): ramp rfU through its optional actuator
       filter every turn (a zero-order hold between the slow-cadence updates), then write the
       common fractional detuning applied/rfBase to every selected cavity.  The RFCA reads freq
       live from p_elem during tracking, so no matrix recompute is needed; but a bare freq write
       would leave phase_fiducial (= -omega_old*t0) stale, so rescale it by the frequency ratio
       to hold the fiducial time t0 fixed (subtlety (a)).  Does NOT set changed=1. */
    if (plane->nRF > 0) {
      double delta = plane->rfU - plane->rfBase;
      double applied = (plane->nRfActFilter > 0)
                         ? plane->rfBase + applyIIRFilterCascade(plane->rfActFilter, plane->nRfActFilter, delta)
                         : plane->rfU;
      if (applied != plane->rfLastApplied) {
        double fidRatio = (plane->rfLastApplied != 0) ? applied / plane->rfLastApplied : 1.0;
        double frac = (plane->rfBase != 0) ? applied / plane->rfBase : 1.0;
        long k;
        for (k = 0; k < plane->nRF; k++) {
          RFCA *rfca = (RFCA *)(plane->rfElem[k]->p_elem);
          *((double *)(plane->rfElem[k]->p_elem + plane->rfFreqOffset[k])) = plane->rfBaseFreq[k] * frac;
          if (rfca->fiducial_seen) {
            rfca->phase_fiducial *= fidRatio; /* preserves t0 = -phase_fiducial/omega */
            set_phase_reference(rfca->phase_reference, rfca->phase_fiducial);
          }
        }
        plane->rfLastApplied = applied;
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
    bd->xTick = bd->nxFilter > 0 ? applyIIRFilterCascade(bd->xFilter, bd->nxFilter, xReading) : xReading;
    /* same reading (incl. noise) through the RF-loop's own filter, when configured */
    bd->xTickRF = bd->nxFilterRF > 0 ? applyIIRFilterCascade(bd->xFilterRF, bd->nxFilterRF, xReading) : xReading;
  }
  if (eptr->type == T_VMON || eptr->type == T_MONI) {
    if (bpm_noise)
      yReading += noise_value(bpm_noise, bpm_noise_cutoff,
                              (bpm_noise_distribution && strncmp(bpm_noise_distribution, "uniform", 7) == 0) ? 2 : 1);
    bd->yTick = bd->nyFilter > 0 ? applyIIRFilterCascade(bd->yFilter, bd->nyFilter, yReading) : yReading;
  }
}

/* ------------------------------------------------------------------ */
/* FAST loop: PID update in actuator space over the correctors and (x plane) the RF-phase
   column, run once per FOFB iteration from the raw bpm_filter ticks.  The deadbeat solve
   dK = T*Q partitions the orbit automatically: the eta-shaped (energy) component is assigned
   to the phase column and the betatron residual to the correctors.  The slow RF-frequency
   loop is handled separately by updateRFLoop. */

static void updateSetpoints(FOFB_PLANE *plane, double *rmsOrbit, double *maxCorr, long *nPegged) {
  CORMON_DATA *CM = &plane->CM;
  MAT *Q, *dK;
  long i;
  double sum2 = 0, maxc = 0;
  *nPegged = 0;

  /* Diagnostic dispersion projection delta_est = sum(x_i*eta_i)/sum(eta_i^2) from the RAW x
     ticks (x plane, phase active only).  This is now DIAGNOSTIC-ONLY -- the phase command
     comes from the deadbeat solve below, not from this projection.  sums->centroid (hence
     xTick) is already MPI_Allreduce'd, so delta_est is bit-identical on every rank. */
  if (plane->coord == 0 && plane->rfPhaseActive) {
    double sxe = 0, see = 0;
    for (i = 0; i < CM->nmon; i++) {
      void **pp = fofbDataPtr(CM->umoni[i]);
      FOFB_BPM_DATA *bd = pp ? *pp : NULL;
      double eta = CM->umoni[i]->twiss ? CM->umoni[i]->twiss->etax : 0;
      double reading = bd ? bd->xTick : 0;
      sxe += reading * eta;
      see += eta * eta;
    }
    plane->deltaEst = (see > 0) ? sxe / see : 0;
  }

  /* collect BPM ticks (raw bpm_filter) into the orbit vector; xBpmRms reports this orbit */
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

  /* Auto eta-blind the correctors: when the SLOW RF-frequency loop owns the dispersive
     (eta-shaped) orbit and there is NO RF-phase column to absorb it in the fast deadbeat
     solve, project the dispersion mode out of Q before the solve.  Otherwise the fast
     correctors cancel the eta-shaped orbit with betatron bumps, hiding the error from -- and
     then fighting -- the slow frequency integrator (two integral loops competing over the
     same energy channel => windup/divergence).  Removing the eta component here makes the
     correctors blind to dispersion, leaving that DC channel to the frequency loop (which acts
     on xTickRF).  This is the classic radial-loop spatial decoupling, and is exactly what the
     deprecated subtract_dispersion did by hand -- now applied automatically only in the
     frequency-only case (with an RF-phase column present, that column already owns the fast
     dispersion mode, so we leave Q intact). */
  if (plane->coord == 0 && plane->nRF > 0 && !plane->rfPhaseActive) {
    double sxe = 0, see = 0;
    for (i = 0; i < CM->nmon; i++) {
      double eta = CM->umoni[i]->twiss ? CM->umoni[i]->twiss->etax : 0;
      sxe += Mij(Q, i, 0) * eta;
      see += eta * eta;
    }
    if (see > 0) {
      double proj = sxe / see;
      for (i = 0; i < CM->nmon; i++) {
        double eta = CM->umoni[i]->twiss ? CM->umoni[i]->twiss->etax : 0;
        Mij(Q, i, 0) -= proj * eta;
      }
    }
  }

  /* dK = T * Q = -C^-1 * Q : deadbeat correction demand (kick units; deg for the phase col) */
  dK = matrix_mult(CM->T, Q);

  for (i = 0; i < CM->ncor; i++) {
    double e, d, uProp, uNew, applied, Kp, Ki, Kd, limitVal, limitRef;
    long isPhase = (i == plane->phaseIndex);
    if (isPhase) {
      /* RF-phase class: gains energy_*, and the limit is a phase excursion about the base
         phase (|phi-phi0| <= rf_phase_limit). */
      Kp = plane->energyKp;
      Ki = plane->energyKi;
      Kd = plane->energyKd;
      limitVal = rf_phase_limit;
      limitRef = plane->base[i];
    } else {
      Kp = plane->Kp;
      Ki = plane->Ki;
      Kd = plane->Kd;
      limitVal = corrector_limit;
      limitRef = 0; /* correctors clamp on the absolute kick */
    }
    e = Mij(dK, i, 0) / CM->kick_coef[i]; /* param-space error (deg for the phase entry) */
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
    if (!isPhase) {
      /* MaxCorr* reports the steering excursion (rad); the RF-phase command is a
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
/* SLOW RF-frequency loop: a scalar PID on the standalone RF-frequency matrix, run only when
   the cadence gate fires (every rf_update_multiplier fast iterations).  It acts on the
   NOTCH-FILTERED full orbit (xTickRF, which == the raw x reading when no rf_bpm_filter_file
   is set) to null the DC dispersive orbit without being destabilized by the synchrotron line.
   The scalar error e = (Trf*Q_rf)[0] is in Hz; the held frequency rfU is clamped to
   |rfU-rfBase| <= rf_frequency_limit with the same back-calculation anti-windup as the
   correctors.  Runs on ALL ranks (xTickRF is MPI_Allreduce'd), so rfU is bit-identical. */

static void updateRFLoop(FOFB_PLANE *plane, long *nPegged) {
  CORMON_DATA *CM = &plane->CM;
  MAT *Q_rf, *e_rf;
  long i;
  double e, d, uProp, uNew;
  *nPegged = 0;
  if (!plane->active || plane->nRF == 0)
    return;

  Q_rf = matrix_get(CM->nmon, 1);
  for (i = 0; i < CM->nmon; i++) {
    void **pp = fofbDataPtr(CM->umoni[i]);
    FOFB_BPM_DATA *bd = pp ? *pp : NULL;
    Mij(Q_rf, i, 0) = bd ? bd->xTickRF : 0;
  }
  e_rf = matrix_mult(plane->Trf, Q_rf); /* 1 x 1: frequency error in Hz (s_rf=1) */
  e = Mij(e_rf, 0, 0) / plane->s_rf;

  d = e - plane->rfEPrev;
  if (!(anti_windup && plane->rfPegged))
    plane->rfIacc += plane->rfKi * e;
  uProp = plane->rfBase + plane->rfKp * e + plane->rfKd * d;
  uNew = uProp + plane->rfIacc;
  if (rf_frequency_limit > 0 && fabs(uNew - plane->rfBase) > rf_frequency_limit) {
    double lim = plane->rfBase + (uNew > plane->rfBase ? 1.0 : -1.0) * rf_frequency_limit;
    if (anti_windup)
      plane->rfIacc = lim - uProp;
    uNew = lim;
    plane->rfPegged = 1;
    (*nPegged)++;
  } else
    plane->rfPegged = 0;
  plane->rfU = uNew;
  plane->rfEPrev = e;

  matrix_free(e_rf);
  matrix_free(Q_rf);
}

/* ------------------------------------------------------------------ */
/* Flush every WATCH element's output page once, at the end of the feedback run.  do_tracking
   normally flushes WATCH pages at the end of each call, but FOFB suppresses that per-step
   flush (SUPPRESS_WATCH_FLUSH) to avoid per-step filesystem I/O.  During tracking each WATCH
   still flushes at its own FLUSH_INTERVAL (dump_watch_parameters), and on normal completion
   dump_watch's final-sample write lands the last rows; but on early beam-loss termination the
   final sample is never reached, and SDDS_Terminate does not flush unwritten in-memory rows.
   This single teardown flush guarantees the final rows reach disk in all cases.  Mirrors
   do_tracking's WATCH-flush block (initialized check, no rank guard). */
static void flushFofbWatchPages(LINE_LIST *beamline) {
  ELEMENT_LIST *eptr = beamline->elem;
  while (eptr) {
    if (eptr->type == T_WATCH) {
      WATCH *watch = (WATCH *)eptr->p_elem;
      if (watch->initialized && watch->SDDS_table) {
        if (!SDDS_UpdatePage(watch->SDDS_table, 0)) {
          SDDS_SetError("Unable to flush WATCH page at fast_orbit_feedback teardown");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
        if (!inhibitFileSync)
          SDDS_DoFSync(watch->SDDS_table);
      }
    }
    eptr = eptr->succ;
  }
}

/* ------------------------------------------------------------------ */
/* Bomb if an &modulate_elements modulation targets a parameter FOFB actuates.  Both the
   modulation and the proportional feedback reference the parameter's INITIAL value and write it
   absolutely every turn, so they would clobber each other (see fofbUpdateActuators).  A
   modulation item and an actuator conflict when they are the same element AND the same parameter;
   compare by element pointer (both come from beamline->elem) and byte offset (uniquely identifies
   the parameter within the element). */
static void fofbBombIfModulated(MODULATION_DATA *mod, ELEMENT_LIST *elem, long offset) {
  long j;
  if (!elem)
    return;
  for (j = 0; j < mod->nItems; j++) {
    if (mod->element[j] != elem)
      continue;
    if (entity_description[mod->element[j]->type].parameter[mod->parameterNumber[j]].offset == offset) {
      /* advice depends on where the free phase-error channel is for this actuator type */
      const char *advice =
        (elem->type == T_RFMODE)
          ? "To inject RF phase errors on this cavity, use RFMODE's own generator-noise "
            "channels (e.g. NOISE_PHASE_GEN) rather than modulating PHASE_SETPOINT."
          : "To inject RF phase errors, modulate PHASE instead (the feedback actuates the "
            "additive DELTA_PHASE).";
      bombElegantVA("fast_orbit_feedback: element %s parameter %s is modulated by &modulate_elements "
                    "but is also a fast_orbit_feedback actuator; both reference the parameter's initial "
                    "value and would overwrite each other every turn.  %s\n",
                    mod->element[j]->name,
                    entity_description[mod->element[j]->type].parameter[mod->parameterNumber[j]].name,
                    advice);
    }
  }
}

static void fofbCheckModulationConflicts(RUN *run) {
  MODULATION_DATA *mod = &run->modulationData;
  long ip, i, k;
  if (mod->nItems <= 0)
    return;
  for (ip = 0; ip < 2; ip++) {
    FOFB_PLANE *plane = &planeData[ip];
    if (!plane->active)
      continue;
    for (i = 0; i < plane->CM.ncor; i++) {
      if (i == plane->phaseIndex)
        continue; /* the phase column is covered via rfPhaseElem/rfPhaseOffset below */
      fofbBombIfModulated(mod, plane->CM.ucorr[i], plane->kickOffset[i]);
    }
    for (k = 0; k < plane->nRFPhase; k++)
      fofbBombIfModulated(mod, plane->rfPhaseElem[k], plane->rfPhaseOffset[k]); /* DELTA_PHASE */
    for (k = 0; k < plane->nRF; k++)
      fofbBombIfModulated(mod, plane->rfElem[k], plane->rfFreqOffset[k]); /* FREQ */
  }
}

/* ------------------------------------------------------------------ */

long doFastOrbitFeedback(RUN *run, VARY *control, LINE_LIST *beamline, BEAM *beam, OUTPUT_FILES *output_files) {
  long i_step, ip, passOffset = 0;
  double pCentral, finalCharge = 0;
  double clockBase = 0.0;
  unsigned long baseFlags, savedBeamlineFidFlag;

  if (!fofbSetupDone)
    bombElegant("fast_orbit_feedback: setup was not performed", NULL);

  /* refuse to run if any &modulate_elements modulation targets a parameter FOFB actuates */
  fofbCheckModulationConflicts(run);

  pCentral = run->p_central;
  fofbActive = 1;
  fofbTotalPasses = passes_between_updates * control->n_steps;

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
     optics, so the recomputation is wasted work; suppress it automatically.
     SUPPRESS_WATCH_FLUSH: FOFB calls do_tracking once per step, and do_tracking flushes
     every WATCH page at the end of each call -- a buffer flush + seeks per WATCH every
     passes_between_updates passes, on top of each WATCH's own FLUSH_INTERVAL flushing.
     Under many concurrent jobs that per-step I/O stalls the CPU on filesystem contention.
     Suppress it; WATCH pages are flushed by each WATCH's FLUSH_INTERVAL and once at
     teardown (flushFofbWatchPages). */
  baseFlags = FINAL_SUMS_ONLY | ALLOW_MPI_ABORT_TRACKING | SUPPRESS_TWISS_UPDATE | SUPPRESS_WATCH_FLUSH;

  for (i_step = 0; i_step < control->n_steps; i_step++) {
    unsigned long flags;
    double xrms = 0, yrms = 0, xcor = 0, ycor = 0;
    long xpeg = 0, ypeg = 0;
    long nLeft;

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
          if (bd->nxFilterRF > 0)
            resetFilterBank(bd->xFilterRF, bd->nxFilterRF);
        }
      }
    }

    /* FOFB tracks ONE continuous beam across all steps, so the energy profile vs s must be
       fiducialized once (on step 0) and then held -- exactly &track's single-do_tracking behavior.
       Force FIRST_BEAM_IS_FIDUCIAL regardless of run_control's first_is_fiducial, and set
       FIDUCIAL_BEAM_SEEN after step 0 (below); otherwise do_tracking re-establishes the energy
       profile -- printing "This step establishes energy profile vs s (fiducial beam)." -- at the
       start of every step and resets the reference momentum profile from the mid-flight beam. */
    flags = baseFlags | FIRST_BEAM_IS_FIDUCIAL |
            (control->fiducial_flag &
             (FIDUCIAL_BEAM_SEEN + RESTRICT_FIDUCIALIZATION +
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

    nLeft = do_tracking(beam, NULL, 0, NULL, beamline, &pCentral, beam->accepted, NULL, NULL, NULL,
                        run, i_step, flags, passes_between_updates, passOffset, NULL, NULL, &finalCharge, NULL, NULL);

    /* capture this step's accumulated macro time (base + per-step accum) so the next
       step continues from it */
    clockBase = trackingClockOffset();

    /* Terminate cleanly if the beam has been completely lost.  FOFB tracks one
       persistent beam across all n_steps; once no particles survive, every
       subsequent BPM reading is an empty-beam centroid (n_part==0), which feeds
       meaningless (potentially NaN) values into the feedback filters and makes the
       remaining steps pointless.  do_tracking returns the surviving count on this
       rank; reduce it so every rank reaches the same decision and the break is
       safe for the collective do_tracking call in the next iteration. */
    {
      long nSurviving = nLeft;
#if USE_MPI
      long nSurvivingGlobal = 0;
      MPI_Allreduce(&nSurviving, &nSurvivingGlobal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      nSurviving = nSurvivingGlobal;
#endif
      if (nSurviving <= 0) {
        if (FOFB_IS_MASTER)
          printf("fast_orbit_feedback: all particles lost at step %ld (after %ld turns); "
                 "terminating feedback cleanly.\n",
                 i_step, passOffset + passes_between_updates);
        fflush(stdout);
        break;
      }
    }

    /* After step 0 establishes the energy profile on the continuous beam, mark the fiducial beam
       seen (unconditionally -- FIRST_BEAM_IS_FIDUCIAL is forced above) so do_tracking holds it for
       every remaining step: no re-fiducialization, no repeated message.  beamline->fiducial_flag is
       restored on exit; the control bit is harmless afterward (FIRST_BEAM_IS_FIDUCIAL is not set
       there, so a later &track still fiducializes normally). */
    control->fiducial_flag |= FIDUCIAL_BEAM_SEEN;
    beamline->fiducial_flag |= FIDUCIAL_BEAM_SEEN;
    /* After step 0 has established the RF fiducial on the continuous beam, stop
       do_tracking from deleting phase references / re-fiducializing the RF at each
       subsequent step boundary (see the header comment above). */
    beamline->fiducial_flag &= ~RESET_RF_FOR_EACH_STEP;

    passOffset += passes_between_updates;

    /* run the FAST controller for each active plane every step; the RF-phase column is driven
       inside updateSetpoints for the x plane */
    if (planeData[0].active)
      updateSetpoints(&planeData[0], &xrms, &xcor, &xpeg);
    if (planeData[1].active)
      updateSetpoints(&planeData[1], &yrms, &ycor, &ypeg);

    /* run the SLOW RF-frequency loop only on the decimated cadence.  Gate on i_step (identical
       across MPI ranks) so the update instants are deterministic. */
    if (include_rf_frequency && planeData[0].active &&
        ((i_step + 1) % rf_update_multiplier) == 0) {
      long rfpeg = 0;
      updateRFLoop(&planeData[0], &rfpeg);
      xpeg += rfpeg;
    }

    if (verbosity)
      printf("fast_orbit_feedback step %ld: xBpmRms=%.6e m, yBpmRms=%.6e m (pegged x=%ld, y=%ld)\n",
             i_step, xrms, yrms, xpeg, ypeg);

    /* Record this step as one row of the single run-long page (started in
       setupFastOrbitFeedback).  The page is flushed to disk in place every
       flush_interval steps so the file can be examined while the run is still in
       progress; the final flush happens in finishFastOrbitFeedback. */
    if (fofbOutputActive && FOFB_IS_MASTER) {
      if (!SDDS_SetRowValues(&SDDS_fofb, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, i_step,
                             "Step", (int32_t)i_step, "Pass", (int32_t)passOffset,
                             "xBpmRms", xrms, "yBpmRms", yrms, "MaxCorrX", xcor, "MaxCorrY", ycor,
                             "PeggedX", (int32_t)xpeg, "PeggedY", (int32_t)ypeg, NULL)) {
        SDDS_SetError("Unable to set fast_orbit_feedback output row");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
      if (fofbHasDfrfColumn) {
        double dfrf = (planeData[0].active && planeData[0].nRF > 0)
                        ? planeData[0].rfU - planeData[0].rfBase
                        : 0.0;
        if (!SDDS_SetRowValues(&SDDS_fofb, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, i_step,
                               "DeltaFrf", dfrf, NULL)) {
          SDDS_SetError("Unable to set fast_orbit_feedback DeltaFrf value");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
      }
      if (fofbHasPhaseColumns) {
        double dphase = (planeData[0].active && planeData[0].phaseIndex >= 0)
                          ? planeData[0].u[planeData[0].phaseIndex] - planeData[0].base[planeData[0].phaseIndex]
                          : 0.0;
        double dest = (planeData[0].active && planeData[0].rfPhaseActive) ? planeData[0].deltaEst : 0.0;
        if (!SDDS_SetRowValues(&SDDS_fofb, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, i_step,
                               "DeltaPhase", dphase, "deltaEnergyEst", dest, NULL)) {
          SDDS_SetError("Unable to set fast_orbit_feedback RF-phase values");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
      }
      /* periodic in-place flush (first SDDS_UpdatePage writes the page, later ones update it) */
      if (flush_interval >= 1 && ((i_step + 1) % flush_interval) == 0) {
        if (!SDDS_UpdatePage(&SDDS_fofb, 0)) {
          SDDS_SetError("Unable to flush fast_orbit_feedback output page");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
        if (!inhibitFileSync)
          SDDS_DoFSync(&SDDS_fofb);
      }
    }
  }

  /* FOFB suppressed do_tracking's per-step WATCH flush (SUPPRESS_WATCH_FLUSH); flush the
     WATCH pages once here so the final rows reach disk on both normal completion and the
     beam-loss break above. */
  flushFofbWatchPages(beamline);

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
    if (bd->xFilterRF) {
      freeIIRFilterMemory(bd->xFilterRF, bd->nxFilterRF);
      free(bd->xFilterRF);
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
  /* standalone RF-frequency matrix + actuator ramp bank */
  if (plane->Crf)
    matrix_free(plane->Crf);
  if (plane->Trf)
    matrix_free(plane->Trf);
  plane->Crf = plane->Trf = NULL;
  if (plane->rfActFilter) {
    freeIIRFilterMemory(plane->rfActFilter, plane->nRfActFilter);
    free(plane->rfActFilter);
    plane->rfActFilter = NULL;
  }
  if (plane->rfPhaseElem)
    free(plane->rfPhaseElem);
  if (plane->rfBasePhase)
    free(plane->rfBasePhase);
  if (plane->rfPhaseOffset)
    free(plane->rfPhaseOffset);
  plane->rfPhaseElem = NULL;
  plane->rfBasePhase = NULL;
  plane->rfPhaseOffset = NULL;
  plane->rfPhaseActive = 0;
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
    /* The run accumulated one page of rows (one per step), periodically flushed during
       tracking.  Do a final in-place update so any rows written since the last periodic
       flush -- or the whole page, if flush_interval never fired or was disabled -- reach
       disk, then close the file.  SDDS_UpdatePage writes the page in full on its first
       call, so this is correct even if no periodic flush happened. */
    if (!SDDS_UpdatePage(&SDDS_fofb, 0)) {
      SDDS_SetError("Unable to write final fast_orbit_feedback output page");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    if (!SDDS_Terminate(&SDDS_fofb)) {
      SDDS_SetError("Unable to finish fast_orbit_feedback output file");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  }
  fofbOutputActive = 0;
  fofbHasDfrfColumn = 0;
  fofbHasPhaseColumns = 0;

  /* restore each phase-actuator cavity to its base phase for a clean end state (the
     RFCA otherwise retains the last commanded offset); do this before freePlane frees
     the cached arrays.  The RFCA live-reads rfca->phase, so a plain write suffices. */
  {
    long ip, k;
    for (ip = 0; ip < 2; ip++) {
      FOFB_PLANE *plane = &planeData[ip];
      if (plane->active && plane->rfPhaseActive && plane->rfPhaseElem)
        for (k = 0; k < plane->nRFPhase; k++)
          *((double *)(plane->rfPhaseElem[k]->p_elem + plane->rfPhaseOffset[k])) = plane->rfBasePhase[k];
    }
  }

  freePlane(&planeData[0]);
  freePlane(&planeData[1]);

  /* release the FOFB-owned steering intake so a later &fast_orbit_feedback
     re-declares its actuators cleanly */
  freeSteeringList(&fofbSL[0]);
  freeSteeringList(&fofbSL[1]);
  freeSteeringList(&fofbRFsl);
  freeSteeringList(&fofbRFPhasesl);
  fofbSLDeclared[0] = fofbSLDeclared[1] = 0;
  fofbRFDeclared = 0;
  fofbRFPhaseDeclared = 0;

  fofbSetupDone = 0;
}
