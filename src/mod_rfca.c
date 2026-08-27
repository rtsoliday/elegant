/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: mod_rfca.c
 * contents: modulated_rf_cavity()
 *
 * Michael Borland, 1995
 */
#include "mdb.h"
#include "track.h"
#include "SDDS.h"

long modulated_rf_cavity(double **part, int64_t np, MODRF *modrf, double P_central, double zEnd,
			 long pass, long nPasses) {
  int64_t ip;
  double amPhase, pmPhase;
  double P, gamma, dgamma, phase, length, volt;
  double *coord, t, t0, omega0, omega, beta_i, tau, dt, tAve = 0;
#ifdef DEBUG
  static SDDS_TABLE debugTable;
  static long debugInitialized, debugCount = 0, debugLength;
#endif
#if USE_MPI
  long np_total, np_tmp;
#endif
#ifdef USE_KAHAN
  double error = 0.0, t1 = 0.0;
#endif

  log_entry("modulated_rf_cavity");

  if (modrf->freq < 1e3 && modrf->freq)
    printWarningForTracking("MODRF frequency is less than 1kHz", "This may be an error. Check manual for units.");
  if (fabs(modrf->volt) < 100 && modrf->volt)
    printWarning("MODRF voltage is less than 100V", "This may be an error. Check manual for units.");

#if USE_MPI
  if (!isMaster) {
#endif
    if (!part)
      bombElegant("NULL particle data pointer (modulated_rf_cavity)", NULL);
    for (ip = 0; ip < np; ip++)
      if (!part[ip]) {
        printf("NULL pointer for particle %" PRId64 " (modulated_rf_cavity)\n", ip);
        fflush(stdout);
#if USE_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        abort();
      }
#if USE_MPI
  }
#endif
  if (!modrf)
    bombElegant("NULL modrf pointer (modulated_rf_cavity)", NULL);

#ifdef DEBUG
  if (!debugInitialized) {
    debugInitialized = 1;
    debugCount = 0;
    if (!SDDS_InitializeOutputElegant(&debugTable, SDDS_BINARY, 0, NULL, NULL, "modrf.debug") ||
        SDDS_DefineColumn(&debugTable, "t", NULL, "s", NULL, NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&debugTable, "phase", NULL, "rad", NULL, NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&debugTable, "V", NULL, "V", NULL, NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&debugTable, "freq", NULL, "Hz", NULL, NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&debugTable, "RF", NULL, NULL, NULL, NULL, SDDS_DOUBLE, 0) < 0 ||
        !SDDS_WriteLayout(&debugTable) || !SDDS_StartTable(&debugTable, debugLength = 1024))
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
  }
#endif

#if (!USE_MPI)
  if (np <= 0) {
    log_exit("modulated_rf_cavity");
    return (np);
  }
#else
  if (distributedBeam) {
    if (isMaster)
      np_tmp = 0;
    else
      np_tmp = np;
    MPI_Allreduce(&np_tmp, &np_total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (np_total <= 0) {
      log_exit("modulated_rf_cavity");
      return (np_total);
    }
  } else if (np <= 0) {
    log_exit("modulated_rf_cavity");
    return (np);
  }
#endif

  length = modrf->length;

  if (modrf->volt == 0) {
    if (length) {
      if (isSlave || !distributedBeam) {
        for (ip = 0; ip < np; ip++) {
          coord = part[ip];
          coord[0] += coord[1] * length;
          coord[2] += coord[3] * length;
          coord[4] += length;
        }
      }
    }
    log_exit("modulated_rf_cavity");
    return (np);
  }

  if (!(omega0 = PIx2 * modrf->freq))
    bombElegant("FREQ=0 for MODRF element", NULL);

  if (modrf->phase_reference == 0)
    modrf->phase_reference = unused_phase_reference();

  switch (get_phase_reference(&phase, modrf->phase_reference)) {
  case REF_PHASE_RETURNED:
    break;
  case REF_PHASE_NOT_SET:
  case REF_PHASE_NONEXISTENT:
    if (!modrf->fiducial_seen) {
      /* set reference phase so that the center of this bunch goes through
                 * at the desired phase 
                 */
      unsigned long mode;
      if (!(mode = parseFiducialMode(modrf->fiducial)))
        bombElegant("invalid fiducial mode for MODRF element", NULL);
      t0 = findFiducialTime(part, np, zEnd - length, length / 2, P_central, mode);
      modrf->phase_fiducial = -omega0 * t0;
      modrf->fiducial_seen = 1;
    }
    set_phase_reference(modrf->phase_reference, phase = modrf->phase_fiducial);
    break;
  default:
    bombElegant("unknown return value from get_phase_reference()", NULL);
    break;
  }

  t0 = -modrf->phase_fiducial / omega0;
  if (isSlave || !distributedBeam) {
    for (ip = tAve = 0; ip < np; ip++) {
#ifndef USE_KAHAN
      tAve += part[ip][4] / (c_mks * beta_from_delta(P_central, part[ip][5]));
#else
      t1 = part[ip][4] / (c_mks * beta_from_delta(P_central, part[ip][5]));
      tAve = KahanPlus(tAve, t1, &error);
#endif
    }
  }
#if (!USE_MPI)
  tAve /= np;
#else
  if (distributedBeam) {
    if (USE_MPI) {
      double tAve_total = 0.0;
#  ifndef USE_KAHAN
      MPI_Allreduce(&tAve, &tAve_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      tAve = tAve_total / np_total;
#  else
      tAve_total = KahanParallel(tAve, error, MPI_COMM_WORLD);
      tAve = tAve_total / np_total;
#  endif
    }
  } else
    tAve /= np;
#endif

  dt = tAve - t0;
  if ((modrf->startTime >= 0 && tAve < modrf->startTime) || (modrf->endTime >= 0 && tAve > modrf->endTime)) {
    pmPhase = 0;
    phase = PI / 180 * modrf->phase + omega0 * dt;
    omega = omega0;
    amPhase = 0;
    volt = modrf->volt / (1e6 * particleMassMV * particleRelSign);
  } else {
    pmPhase = modrf->pmFreq * PIx2 * (dt-modrf->startTime) + modrf->pmPhase * PI / 180;
    phase = PI / 180 * (modrf->phase + modrf->pmOffset +
			modrf->pmMag * sin(pmPhase) * exp(-modrf->pmDecay * (dt-modrf->startTime)))
			+ omega0 * dt;
    if (modrf->pmMag)
      omega = omega0 + PIx2 * modrf->pmFreq * (PI / 180 * modrf->pmMag) * cos(pmPhase);
    else
      omega = omega0;
    amPhase = modrf->amFreq * PIx2 * (dt-modrf->startTime) + modrf->amPhase * PI / 180;
    volt = modrf->volt / (1e6 * particleMassMV * particleRelSign) *
      (1 + modrf->amOffset + modrf->amMag * sin(amPhase) * exp(-modrf->amDecay * (dt-modrf->startTime)));
  }

#if USE_MPI
  if (myid==0) {
#endif
  if (modrf->record) {
    if (!modrf->SDDSrec) {
      TRACKING_CONTEXT tcon;
      char *filename;
      getTrackingContext(&tcon);
      filename = compose_filename(modrf->record, tcon.rootname);
      modrf->SDDSrec = tmalloc(sizeof(*(modrf->SDDSrec)));
      if (!SDDS_InitializeOutput(modrf->SDDSrec, SDDS_BINARY, 1, NULL, NULL, filename) ||
	  !SDDS_DefineSimpleColumn(modrf->SDDSrec, "Pass", NULL, SDDS_LONG) ||
	  !SDDS_DefineSimpleColumn(modrf->SDDSrec, "Ct", "s", SDDS_DOUBLE) ||
	  !SDDS_DefineSimpleColumn(modrf->SDDSrec, "Phase", "deg", SDDS_DOUBLE) ||
	  !SDDS_DefineSimpleColumn(modrf->SDDSrec, "Voltage", "V", SDDS_DOUBLE) ||
	  !SDDS_WriteLayout(modrf->SDDSrec)) {
	SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
	SDDS_Bomb("problem setting up MODRF record file");
      }
      free(filename);
    }
    if (pass==0 && !SDDS_StartPage(modrf->SDDSrec, nPasses)) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      SDDS_Bomb((char *)"problem starting page for MODRF record file");
    }
    if (!SDDS_SetRowValues(modrf->SDDSrec, SDDS_SET_BY_NAME|SDDS_PASS_BY_VALUE,
			   pass, 
			   (char*)"Pass", pass,
			   (char*)"Ct", tAve,
			   (char*)"Phase", phase*180/PI,
			   (char*)"Voltage", volt*1e6*particleMassMV,
			   NULL)) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      SDDS_Bomb((char *)"problem setting up data for MODRF record file");
    }
    if ((pass%100==0 || pass==(nPasses-1)) && !SDDS_UpdatePage(modrf->SDDSrec, FLUSH_TABLE)) {
      SDDS_Bomb((char *)"problem setting up data for TRFMODE record file");
    }
  }
#if USE_MPI
  }
#endif
  
  if ((tau = modrf->Q / omega0))
    volt *= sqrt(1 - exp(-dt / tau));
  
  if (isSlave || !distributedBeam) {
    for (ip = 0; ip < np; ip++) {
      coord = part[ip];
      /* apply initial drift */
      coord[0] += coord[1] * length / 2;
      coord[2] += coord[3] * length / 2;
      coord[4] += length / 2 * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));

      /* compute energy kick */
      P = P_central * (1 + coord[5]);
      beta_i = P / (gamma = sqrt(sqr(P) + 1));
      t = coord[4] / (c_mks * beta_i);
      dgamma = volt * sin(omega * (t - tAve) + phase);

      /* apply energy kick */
      add_to_particle_energy(coord, t, P_central, dgamma);

      /* apply final drift */
      coord[0] += coord[1] * length / 2;
      coord[2] += coord[3] * length / 2;
      coord[4] += length / 2.0 * sqrt(1 + sqr(coord[1]) + sqr(coord[3]));

#ifdef DEBUG
      if (ip == 0) {
        double redPhase1;
        if ((debugCount + 1) > debugLength && !SDDS_LengthenTable(&debugTable, (debugLength += 1024)))
          SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
        if (!SDDS_SetRowValues(&debugTable, SDDS_BY_INDEX | SDDS_PASS_BY_VALUE,
                               debugCount,
                               0, t,
                               1, fmod(omega * (t - tAve) + phase, PIx2), 2, volt, 3, omega / PIx2,
                               4, volt * sin(omega * (t - tAve) + phase), -1) ||
            !SDDS_UpdateTable(&debugTable))
          SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
        debugCount++;
      }
#endif
    }
  }
  log_exit("modulated_rf_cavity");
  return (np);
}
