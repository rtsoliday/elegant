/*************************************************************************\
* Copyright (c) 2017 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2017 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

#include "mdb.h"
#include "track.h"
#include "SDDS.h"

#define SLICE_PARAMETERS 6
static SDDS_DEFINITION slice_parameter[SLICE_PARAMETERS] = {
  {"Step", "&parameter name=Step, type=long, description=\"Simulation step\" &end"},
  {"Pass", "&parameter name=Pass, type=long &end"},
  {"s", "&parameter name=s, units=m, type=double &end"},
  {"pCentral", "&parameter name=pCentral, symbol=\"p$bcen$n\", units=\"m$be$nc\", type=double, description=\"Reference beta*gamma\" &end"},
  {"dt", "&parameter name=dt, symbol=\"$gD$rt\", units=s, type=double, description=\"Slice duration\" &end"},
  {"SVNVersion", "&parameter name=SVNVersion, type=string, description=\"SVN version number\", fixed_value=" SVN_VERSION " &end"},
};

#define SLICE_COLUMNS 18
static SDDS_DEFINITION slice_column[SLICE_COLUMNS] = {
  {"Slice", "&column name=Slice, type=long &end"},
  {"Particles", "&column name=Particles, description=\"Number of simulation particles in slice\", type=long, &end"},
  {"Charge", "&column name=Charge, description=\"Charge in the slice\", units=C, type=double &end"},
  {"dCt", "&column name=dCt, symbol=\"$gD$rt\", units=s, type=double, description=\"relative time of flight\" &end"},
  {"Cx", "&column name=Cx, symbol=\"<x>\", units=m, type=double, description=\"x centroid\" &end"},
  {"Cxp", "&column name=Cxp, symbol=\"<x'>\", type=double, description=\"x' centroid\" &end"},
  {"Cy", "&column name=Cy, symbol=\"<y>\", units=m, type=double, description=\"y centroid\" &end"},
  {"Cyp", "&column name=Cyp, symbol=\"<y'>\", type=double, description=\"y' centroid\" &end"},
  {"Ct", "&column name=Ct, symbol=\"<t>\", units=s, type=double, description=\"central time of flight\" &end"},
  {"Cdelta", "&column name=Cdelta, symbol=\"<$gd$r>\", type=double, description=\"delta centroid\" &end"},
  {"Sx", "&column name=Sx, symbol=\"$gs$r$bx$n\", units=m, type=double, description=\"sqrt(<(x-<x>)^2>)\" &end"},
  {"Sxp", "&column name=Sxp, symbol=\"$gs$r$bx'$n\", type=double, description=\"sqrt(<(x'-<x'>)^2>)\" &end"},
  {"Sy", "&column name=Sy, symbol=\"$gs$r$by$n\", units=m, type=double, description=\"sqrt(<(y-<y>)^2>)\" &end"},
  {"Syp", "&column name=Syp, symbol=\"$gs$r$by'$n\", type=double, description=\"sqrt(<(y'-<y'>)^2>)\" &end"},
  {"St", "&column name=St, symbol=\"$gs$r$bt$n\", units=s, type=double, description=\"sqrt(<(t-<t>)^2>)\" &end"},
  {"Sdelta", "&column name=Sdelta, symbol=\"$gs$bd$n$r\", type=double, description=\"sqrt(<(delta-<delta>)^2>)\" &end"},
  {"ex", "&column name=ex, symbol=\"$ge$r$bx$n\", units=m, type=double, description=\"geometric horizontal emittance\" &end"},
  {"ey", "&column name=ey, symbol=\"$ge$r$by$n\", units=m, type=double, description=\"geometric vertical emittance\" &end"},
};

void set_up_slice_point(SLICE_POINT *slicePoint, RUN *run, long occurence, char *previousElementName, long IDSlotsPerBunch) {
  if (slicePoint->disable)
    return;
  if (slicePoint->interval <= 0)
    bombElegant("interval is non-positive for SLICE element", NULL);
  if (slicePoint->label && str_in(slicePoint->label, "%s")) {
    char *buffer;
    buffer = tmalloc(sizeof(*buffer) * (strlen(slicePoint->label) + strlen(run->rootname) + 1));
    sprintf(buffer, slicePoint->label, run->rootname);
    free(slicePoint->label);
    slicePoint->label = buffer;
  }
  slicePoint->filename = compose_filename_occurence(slicePoint->filename, run->rootname, occurence + slicePoint->indexOffset);
  SDDS_SlicePointSetup(slicePoint, run->runfile, run->lattice, "set_up_slice_point", previousElementName);
  slicePoint->initialized = 1;
  if (slicePoint->bunchSeries && IDSlotsPerBunch>0) {
    slicePoint->startPID = 1 + (occurence-1)*IDSlotsPerBunch;
    slicePoint->endPID = slicePoint->startPID + IDSlotsPerBunch - 1;
  }
}

void SDDS_SlicePointSetup(SLICE_POINT *slicePoint, char *command_file, char *lattice_file, char *caller,
                          char *previousElementName) {
  SDDS_TABLE *SDDS_table;
  char *filename;

#if MPI_DEBUG
  printf("SDDS_SlicePointSetup called\n");
  fflush(stdout);
#endif

#if USE_MPI
  if (myid < 0)
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (myid != 0)
    return;
#endif

  if (!slicePoint->SDDS_table)
    slicePoint->SDDS_table = tmalloc(sizeof(*(slicePoint->SDDS_table)));
  SDDS_table = slicePoint->SDDS_table;
  filename = slicePoint->filename;

  SDDS_ElegantOutputSetup(SDDS_table, filename, SDDS_BINARY, 0, "slice analysis",
                          command_file, lattice_file,
                          slice_parameter, SLICE_PARAMETERS,
                          slice_column, SLICE_COLUMNS,
                          caller, SDDS_EOS_NEWFILE);
  if (!SDDS_WriteLayout(SDDS_table)) {
    printf("Unable to write SDDS layout for file %s (%s)\n", filename, caller);
    fflush(stdout);
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    exitElegant(1);
  }
}

//static double tmp_safe_sqrt;
//#define SAFE_SQRT(x) ((tmp_safe_sqrt=(x))<0?(double)0.0:sqrt(tmp_safe_sqrt))

void dump_slice_analysis(SLICE_POINT *slicePoint, long step, long pass, long n_passes,
                         double **particle, long particles, double Po,
                         double revolutionLength, double z, double mp_charge) {
  long i, iSlice;
  double *timeCoord = NULL, tMin, tMax;
  double tc0, tc0Error;
  double emit[2], emitc[2];
  long Cx_index = 0, Sx_index = 0, ex_index = 0;
  double t0, dt0, dt;
  BEAM_SUMS *sums;
#if USE_MPI
  long particles_total;

  if (myid < 0)
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (myid == 0)
    particles = 0;

  MPI_Allreduce(&particles, &particles_total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif

  if (pass <= slicePoint->passLast) {
    slicePoint->t0Last = z * sqrt(Po * Po + 1) / (c_mks * (Po + 1e-32));
    slicePoint->t0LastError = 0;
  }
  tc0 = slicePoint->t0Last;
  tc0Error = slicePoint->t0LastError;
  /* This code mimics what happens to particles as they get ~T0 added on each turn with accumulating
   * round-off error. Prevents dCt from walking off too much.
   */
  for (i = 0; i < pass - slicePoint->passLast; i++) {
    dt = revolutionLength * sqrt(Po * Po + 1) / (c_mks * (Po + 1e-32));
    if (slicePoint->referenceFrequency > 0)
      dt = ((long)(dt * slicePoint->referenceFrequency + 0.5)) / slicePoint->referenceFrequency;
#ifndef USE_KAHAN
    tc0 += dt;
#else
    tc0 = KahanPlus(tc0, dt, &tc0Error);
#endif
  }
  slicePoint->t0Last = tc0;
  slicePoint->t0LastError = tc0Error;
  slicePoint->passLast = pass;

  if (!(pass >= slicePoint->start_pass && (pass - slicePoint->start_pass) % slicePoint->interval == 0 &&
        (slicePoint->end_pass < 0 || pass <= slicePoint->end_pass))) {
    return;
  }

  if (isMaster) {
    if (!slicePoint->initialized)
      bombElegant("uninitialized slice element encountered", NULL);
  } else {
    if (!particle && particles)
      bombElegant("NULL coordinate pointer passed to dump_slice_analysis", NULL);
  }

  timeCoord = tmalloc(sizeof(*timeCoord) * particles);
  tMax = -(tMin = DBL_MAX);
  for (i = 0; i < particles; i++) {
    double p, gamma, beta;
    if (!particle[i]) {
      printf("error: coordinate slot %ld is NULL (dump_watch_parameters)\n", i);
      fflush(stdout);
      abort();
    }
    if (slicePoint->startPID >= slicePoint->endPID ||
        (particle[i][6] >= slicePoint->startPID && particle[i][6] <= slicePoint->endPID)) {
      p = Po * (1 + particle[i][5]);
      gamma = sqrt(p * p + 1);
      beta = p / gamma;
      timeCoord[i] = particle[i][4] / (c_mks * beta);
      if (timeCoord[i] > tMax)
        tMax = timeCoord[i];
      if (timeCoord[i] < tMin)
        tMin = timeCoord[i];
    } else {
      timeCoord[i] = -DBL_MAX;
    }
  }
#if USE_MPI
  find_global_min_max(&tMin, &tMax, particles, MPI_COMM_WORLD);
#endif

#ifdef DEBUG
  printf("time limits: %le, %le\n", tMin, tMax);
#endif

  if (isMaster && !SDDS_StartTable(slicePoint->SDDS_table, slicePoint->nSlices)) {
    SDDS_SetError("Problem starting SDDS table (dump_slice_analysis)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  t0 = tMin;
  if (slicePoint->nSlices > 0)
    dt0 = (tMax - tMin) / slicePoint->nSlices;
  else
    dt0 = tMax - tMin;

#ifdef DEBU
  printf("dt0 = %le\n", dt0);
#endif

  sums = allocateBeamSums(0, 1);
  for (iSlice = 0; iSlice < slicePoint->nSlices; iSlice++, t0 += dt0) {
#ifdef DEBUG
    printf("iSlice = %ld, t:[%le, %le]\n", iSlice, t0, t0 + dt0);
    fflush(stdout);
#endif
    /* compute centroids, sigmas, and emittances for x, y, and s */
    zero_beam_sums(sums, 1);
    accumulate_beam_sums(sums, particle, particles, Po, mp_charge,
                         timeCoord, t0, t0 + dt0,
                         slicePoint->startPID, slicePoint->endPID, BEAM_SUMS_SPARSE | BEAM_SUMS_NOMINMAX);
    for (i = 0; i < 2; i++) {
      emitc[i] = emit[i] = 0;
      computeEmitTwissFromSigmaMatrix(emit + i, emitc + i, NULL, NULL, sums->beamSums2->sigma, i * 2);
    }
#ifdef DEBUG
    printf("ex = %le, ey = %le\n", emit[0], emit[1]);
    fflush(stdout);
#endif

    if (isMaster) {
#ifdef DEBUG
      printf("Setting row values\n");
      fflush(stdout);
#endif
      if ((Cx_index = SDDS_GetColumnIndex(slicePoint->SDDS_table, "Cx")) < 0) {
        SDDS_SetError("Problem getting index of SDDS columns (dump_slice_analysis)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
      if (!SDDS_SetRowValues(slicePoint->SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, iSlice,
                             0, iSlice, 1, sums->n_part, 2, sums->charge,
                             3, dt0 * (iSlice - (slicePoint->nSlices - 1) / 2.0),
                             -1)) {
        SDDS_SetError("Problem setting row values for SDDS table (dump_slice_analysis)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
      for (i = 0; i < 6; i++) {
        if (!SDDS_SetRowValues(slicePoint->SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, iSlice,
                               Cx_index + i, i == 4 ? (t0 + dt0 / 2) : sums->centroid[i],
                               -1)) {
          SDDS_SetError("Problem setting row values for SDDS table (dump_slice_analysis)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
      }

      if ((Sx_index = SDDS_GetColumnIndex(slicePoint->SDDS_table, "Sx")) < 0 ||
          (ex_index = SDDS_GetColumnIndex(slicePoint->SDDS_table, "ex")) < 0) {
        SDDS_SetError("Problem getting index of SDDS columns (dump_slice_analysis)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
      for (i = 0; i < 6; i++) {
        if (!SDDS_SetRowValues(slicePoint->SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, iSlice,
                               Sx_index + i, i == 4 ? sqrt(sums->beamSums2->sigma[6][6]) : sqrt(sums->beamSums2->sigma[i][i]),
                               -1)) {
          SDDS_SetError("Problem setting row values for SDDS table (dump_slice_analysis)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
      }
      if (!SDDS_SetRowValues(slicePoint->SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, iSlice,
                             ex_index, emit[0],
                             ex_index + 1, emit[1],
                             -1)) {
        SDDS_SetError("Problem setting row values for SDDS table (dump_slice_analysis)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
    }
  }
  freeBeamSums(sums, 1);

  if (isMaster) {
#ifdef DEBUG
    printf("Setting parameter values\n");
    fflush(stdout);
#endif
    if (!SDDS_SetParameters(slicePoint->SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                            "Step", step, "Pass", pass, "s", z,
                            "pCentral", Po, "dt", (tMax - tMin) / (slicePoint->nSlices > 1 ? slicePoint->nSlices - 1 : 1),
                            NULL)) {
      SDDS_SetError("Problem setting parameter values for SDDS table (dump_slice_analysis)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }

#ifdef DEBUG
    printf("Writing table\n");
    fflush(stdout);
#endif
    if (!SDDS_WriteTable(slicePoint->SDDS_table)) {
      SDDS_SetError("Problem writing data for SDDS table (dump_slice_analysis)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    if (!inhibitFileSync)
      SDDS_DoFSync(slicePoint->SDDS_table);
  }

#if USE_MPI
#  ifdef DEBUG
  printf("Waiting on final barrier\n");
  fflush(stdout);
#  endif
  MPI_Barrier(MPI_COMM_WORLD);
#  ifdef DEBUG
  printf("Passed final barrier\n");
  fflush(stdout);
#  endif
#endif

  free(timeCoord);
}
