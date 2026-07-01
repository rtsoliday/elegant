/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution.
\*************************************************************************/

/* file: undulator_brightness.c
 *
 * Implements the &undulator_brightness setup namelist.  Each invocation
 * registers a brightness spec (undulator parameters, photon energy,
 * twiss source) and creates an RPN scalar named "<tag>.brightness",
 * initialized to zero.  computeAllUndulatorBrightnesses() is called
 * from the Twiss-refresh path (end of run_twiss_output and inside the
 * optimization function) and updates the RPN scalar in place using
 * computeBrightnessLindberg() from liboagphy, with the input beam
 * parameters first reduced to their dispersion-corrected effective
 * values via computeEffectiveBeamParameters().
 */

#include "mdb.h"
#include "track.h"
#include "oagphy.h"
#include "undulator_brightness.h"

typedef struct {
  char *tag;
  double radLambda;
  double radDet;
  long radHarm;
  double undPeriod;            /* meters */
  double undLength;            /* = undPeriod * undN */
  long undN;
  double current;
  short fromTwiss;
  double coupling;
  char *twissElementName;
  long twissOccurence;
  double emitx, emity;
  double betax, alphax, betay, alphay;
  double etax, etaxp, etay, etayp;
  double Sdelta;
  long rpnMem;
  short missingWarned;
  short unreachableWarned;
} UNDULATOR_BRIGHTNESS_SPEC;

static UNDULATOR_BRIGHTNESS_SPEC *specList = NULL;
static long nSpecs = 0;

void setup_undulator_brightness(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline) {
  UNDULATOR_BRIGHTNESS_SPEC *spec;
  char memName[256];
  long i;

  log_entry("setup_undulator_brightness");

  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&undulator_brightness, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &undulator_brightness);

  if (!tag || !strlen(tag))
    bombElegant("undulator_brightness: tag is required", NULL);
  if ((wavelength > 0) == (photon_energy > 0))
    bombElegant("undulator_brightness: exactly one of wavelength or photon_energy must be > 0", NULL);
  if (period_length <= 0)
    bombElegant("undulator_brightness: period_length must be > 0", NULL);
  if (total_length > 0) {
    long derivedN = (long)floor(total_length / period_length);
    if (derivedN < 1)
      bombElegant("undulator_brightness: total_length is less than one period_length", NULL);
    if (n_periods > 0 && n_periods != derivedN) {
      char detail[256];
      snprintf(detail, sizeof(detail),
               "tag=%s: total_length=%g and period_length=%g imply n_periods=%ld; "
               "the n_periods=%ld supplied on the namelist is being overridden",
               tag, total_length, period_length, derivedN, n_periods);
      printWarning("undulator_brightness: total_length overrides n_periods", detail);
    }
    n_periods = derivedN;
  } else if (n_periods <= 0) {
    bombElegant("undulator_brightness: either n_periods (> 0) or total_length (> period_length) is required", NULL);
  }
  if (current <= 0)
    bombElegant("undulator_brightness: current must be > 0", NULL);
  if (harmonic < 1)
    bombElegant("undulator_brightness: harmonic must be >= 1", NULL);
  if (use_twiss_output_values) {
    if (coupling <= 0)
      bombElegant("undulator_brightness: coupling must be > 0 when use_twiss_output_values=1", NULL);
  } else {
    if (emitx <= 0 || betax <= 0 || betay <= 0 || Sdelta <= 0)
      bombElegant("undulator_brightness: emitx, betax, betay, and Sdelta must all be > 0 (use_twiss_output_values=0)", NULL);
  }

  for (i = 0; i < nSpecs; i++) {
    if (strcmp(specList[i].tag, tag) == 0)
      bombElegant("undulator_brightness: duplicate tag", tag);
  }

  specList = (UNDULATOR_BRIGHTNESS_SPEC *)trealloc(specList, sizeof(*specList) * (nSpecs + 1));
  spec = &specList[nSpecs++];
  memset(spec, 0, sizeof(*spec));

  spec->tag = tmalloc(sizeof(*spec->tag) * (strlen(tag) + 1));
  strcpy(spec->tag, tag);

  if (photon_energy > 0)
    spec->radLambda = 12.39841984e-10 / photon_energy;
  else
    spec->radLambda = wavelength;
  spec->radHarm = harmonic;
  spec->radDet = detuning;
  spec->undPeriod = period_length;
  spec->undN = n_periods;
  spec->undLength = period_length * n_periods;
  spec->current = current;

  spec->fromTwiss = use_twiss_output_values ? 1 : 0;
  spec->coupling = coupling;
  if (twiss_element && strlen(twiss_element)) {
    spec->twissElementName = tmalloc(sizeof(*spec->twissElementName) * (strlen(twiss_element) + 1));
    strcpy(spec->twissElementName, twiss_element);
    spec->twissOccurence = twiss_occurence > 0 ? twiss_occurence : 1;
  } else {
    spec->twissElementName = NULL;
    spec->twissOccurence = 0;
  }

  spec->emitx = emitx;
  spec->emity = emity;
  spec->betax = betax;
  spec->alphax = alphax;
  spec->betay = betay;
  spec->alphay = alphay;
  spec->etax = etax;
  spec->etaxp = etaxp;
  spec->etay = etay;
  spec->etayp = etayp;
  spec->Sdelta = Sdelta;

  snprintf(memName, sizeof(memName), "%s.brightness", spec->tag);
  spec->rpnMem = rpn_create_mem(memName, 0);
  rpn_store(0.0, NULL, spec->rpnMem);

  log_exit("setup_undulator_brightness");
}

void computeAllUndulatorBrightnesses(LINE_LIST *beamline) {
  long iSpec;
  UNDULATOR_BRIGHTNESS_SPEC *spec;
  ELEMENT_LIST *eptr;
  double emitx, emity, betax, alphax, betay, alphay, etax, etaxp, etay, etayp, Sdelta;
  double exEff, betaxEff, alphaxEff;
  double eyEff, betayEff, alphayEff;
  double B;

  if (nSpecs == 0 || !beamline)
    return;

  for (iSpec = 0; iSpec < nSpecs; iSpec++) {
    spec = &specList[iSpec];

    if (spec->twissElementName) {
      eptr = NULL;
      for (ELEMENT_LIST *e = beamline->elem; e; e = e->succ) {
        if (strcmp(e->name, spec->twissElementName) == 0 && e->occurence == spec->twissOccurence) {
          eptr = e;
          break;
        }
      }
      if (!eptr) {
        if (!spec->missingWarned) {
          char detail[512];
          snprintf(detail, sizeof(detail),
                   "tag=%s, twiss_element=%s, twiss_occurence=%ld not found in beamline",
                   spec->tag, spec->twissElementName, spec->twissOccurence);
          printWarning("undulator_brightness: twiss source element not found", detail);
          spec->missingWarned = 1;
        }
        continue;
      }
    } else {
      eptr = beamline->elast;
    }

    if (!eptr || !eptr->twiss) {
      if (!spec->missingWarned) {
        char detail[256];
        snprintf(detail, sizeof(detail), "tag=%s: twiss not yet computed at source element", spec->tag);
        printWarning("undulator_brightness: twiss values not available", detail);
        spec->missingWarned = 1;
      }
      continue;
    }

    if (spec->fromTwiss) {
      if (!beamline->radIntegrals.computed) {
        if (!spec->missingWarned) {
          char detail[256];
          snprintf(detail, sizeof(detail),
                   "tag=%s: radiation_integrals=1 required in &twiss_output when use_twiss_output_values=1",
                   spec->tag);
          printWarning("undulator_brightness: radiation integrals not computed", detail);
          spec->missingWarned = 1;
        }
        continue;
      }
      /* Split natural emittance into x/y planes using damping partition
       * numbers Jx, Jy (see brightness.c:230-242 for the reference
       * computation in sddsbrightness): emitx*(1 + Jy*coupling/Jx) = ex0. */
      {
        double Jx = beamline->radIntegrals.Jx;
        double Jy = beamline->radIntegrals.Jy;
        emitx = beamline->radIntegrals.ex0 / (1 + Jy * spec->coupling / Jx);
        emity = spec->coupling * emitx;
      }
      Sdelta = beamline->radIntegrals.sigmadelta;
      betax = eptr->twiss->betax;
      alphax = eptr->twiss->alphax;
      etax = eptr->twiss->etax;
      etaxp = eptr->twiss->etapx;
      betay = eptr->twiss->betay;
      alphay = eptr->twiss->alphay;
      etay = eptr->twiss->etay;
      etayp = eptr->twiss->etapy;
    } else {
      emitx = spec->emitx;
      emity = spec->emity;
      betax = spec->betax;
      alphax = spec->alphax;
      betay = spec->betay;
      alphay = spec->alphay;
      etax = spec->etax;
      etaxp = spec->etaxp;
      etay = spec->etay;
      etayp = spec->etayp;
      Sdelta = spec->Sdelta;
    }

    computeEffectiveBeamParameters(&exEff, &betaxEff, &alphaxEff,
                                   emitx, betax, alphax, etax, etaxp, Sdelta);
    computeEffectiveBeamParameters(&eyEff, &betayEff, &alphayEff,
                                   emity, betay, alphay, etay, etayp, Sdelta);

    /* Derive K from the undulator resonance condition:
     *   lambda = (lambda_u / (2*n*gamma^2)) * (1 + K^2/2)
     * so K = sqrt(2 * (2*n*gamma^2*lambda/lambda_u - 1)).
     * gamma is computed from Pref_output = gamma*beta at the source element:
     *   gamma^2 = 1 + Pref^2. */
    {
      double pRef = eptr->Pref_output;
      double gamma2 = 1.0 + pRef * pRef;
      double KsqOver2 = 2.0 * spec->radHarm * gamma2 * spec->radLambda / spec->undPeriod - 1.0;
      double undK;
      if (KsqOver2 <= 0) {
        if (!spec->unreachableWarned) {
          char detail[256];
          snprintf(detail, sizeof(detail),
                   "tag=%s: wavelength %.6e m unreachable at harmonic %ld with period %.6e m and gamma %.6e",
                   spec->tag, spec->radLambda, spec->radHarm, spec->undPeriod, sqrt(gamma2));
          printWarning("undulator_brightness: wavelength below fundamental at K=0", detail);
          spec->unreachableWarned = 1;
        }
        continue;
      }
      undK = sqrt(2.0 * KsqOver2);

      B = computeBrightnessLindberg(spec->radLambda, (int)spec->radHarm, spec->radDet,
                                    spec->undLength, (int)spec->undN, undK,
                                    exEff, eyEff,
                                    betaxEff, betayEff,
                                    alphaxEff, alphayEff,
                                    Sdelta, spec->current);
    }
    rpn_store(B, NULL, spec->rpnMem);
  }
}
