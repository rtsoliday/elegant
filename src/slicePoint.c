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
#ifdef HAVE_GPU
#  include "gpu/gpu_base.h"
#endif

#define SLICE_PARAMETERS 6
static SDDS_DEFINITION slice_parameter[SLICE_PARAMETERS] = {
  {"Step", "&parameter name=Step, type=long, description=\"Simulation step\" &end"},
  {"Pass", "&parameter name=Pass, type=long &end"},
  {"s", "&parameter name=s, units=m, type=double &end"},
  {"pCentral", "&parameter name=pCentral, symbol=\"p$bcen$n\", units=\"m$be$nc\", type=double, description=\"Reference beta*gamma\" &end"},
  {"dt", "&parameter name=dt, symbol=\"$gD$rt\", units=s, type=double, description=\"Slice duration\" &end"},
  {"SVNVersion", "&parameter name=SVNVersion, type=string, description=\"SVN version number\", fixed_value=" SVN_VERSION " &end"},
};

#define SLICE_COLUMNS 20
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
  {"ecx", "&column name=ecx, symbol=\"$ge$r$bcx$n\", units=m, type=double, description=\"geometric horizontal emittance less dispersion contribution\" &end"},
  {"ecy", "&column name=ecy, symbol=\"$ge$r$bcy$n\", units=m, type=double, description=\"geometric vertical emittance less dispersion contribution\" &end"},
};

void set_up_slice_point(SLICE_POINT *slicePoint, RUN *run, long occurence, char *previousElementName, int64_t IDSlotsPerBunch) {
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

#if defined(HAVE_GPU) && !USE_MPI
/* gpu-elegant only.  dump_slice_analysis() below calls accumulate_beam_sums() once per slice, and each
 * call goes through the whole beam several dozen times to find the slice's few particles: 50 slices of a
 * 480,000-particle beam took 0.17 s a call, 58% of a ring run with six SLICE elements.  Here each slice's
 * particles are found once, in particle order, and accumulate_beam_sums() is given only those, with the
 * same limits.  It tests them again with the same comparisons and adds them in the same order, so the
 * sums are the same to the last bit.  ELEGANT_GPU_SLICE_SINGLE_PASS=0 uses the loop over the whole beam.
 */
typedef struct {
  long *start;      /* slice k's members are member[start[k]] ... member[start[k+1]-1] */
  double **coord;   /* the members' coordinates, slice after slice */
  double *time;     /* and their times */
  double *t0;       /* each slice's lower limit, as the loop below finds it by repeated addition */
} SLICE_MEMBERS;

static long sliceSinglePassEnabled(void) {
  static long setting = -1;

  if (setting < 0) {
    const char *value = getenv("ELEGANT_GPU_SLICE_SINGLE_PASS");
    setting = !(value && *value == '0');
  }
  return setting;
}

/* The slices particle ip is in: those with t0[k] <= t <= t0[k] + dt0, the test accumulate_beam_sums()
 * makes.  Both limits never decrease with k, so the slices are first..last (none if first > last). */
static void sliceRange(double t, const double *t0, double dt0, long nSlices, long *first, long *last) {
  long low, high;

  /* first slice whose upper limit is at or above t */
  low = 0;
  high = nSlices;
  while (low < high) {
    long middle = low + (high - low) / 2;
    if (t0[middle] + dt0 >= t)
      high = middle;
    else
      low = middle + 1;
  }
  *first = low;
  /* last slice whose lower limit is at or below t */
  low = -1;
  high = nSlices - 1;
  while (low < high) {
    long middle = low + (high - low + 1) / 2;
    if (t0[middle] <= t)
      low = middle;
    else
      high = middle - 1;
  }
  *last = low;
}

/* Returns 0 if the slices can't be told apart this way, and the caller then uses the whole beam */
static long findSliceMembers(SLICE_MEMBERS *members, SLICE_POINT *slicePoint, double **particle,
                             long particles, const double *timeCoord, double tMin, double dt0) {
  long nSlices = slicePoint->nSlices, iSlice, ip, first, last, pass;
  long allPIDs = slicePoint->startPID < 0 && slicePoint->endPID < 0;
  long *filled;
  double t0;

  memset(members, 0, sizeof(*members));
  if (nSlices < 1 || particles < 1 || !(dt0 > 0))
    return 0;
  members->t0 = tmalloc(sizeof(*members->t0) * nSlices);
  for (iSlice = 0, t0 = tMin; iSlice < nSlices; iSlice++, t0 += dt0) {
    members->t0[iSlice] = t0;
    /* accumulate_beam_sums() takes every particle when the limits don't differ */
    if (!(t0 < t0 + dt0)) {
      free(members->t0);
      members->t0 = NULL;
      return 0;
    }
  }
  members->start = tmalloc(sizeof(*members->start) * (nSlices + 1));
  filled = tmalloc(sizeof(*filled) * nSlices);
  for (pass = 0; pass < 2; pass++) {
    /* first the number in each slice, then the members in particle order */
    for (iSlice = 0; iSlice < nSlices; iSlice++)
      filled[iSlice] = 0;
    for (ip = 0; ip < particles; ip++) {
      double t = timeCoord[ip];
      if (!(allPIDs || (particle[ip][6] >= slicePoint->startPID && particle[ip][6] <= slicePoint->endPID)))
        continue;
      if (!(t == t))
        continue;
      sliceRange(t, members->t0, dt0, nSlices, &first, &last);
      for (iSlice = first; iSlice <= last; iSlice++) {
        if (pass) {
          long place = members->start[iSlice] + filled[iSlice];
          members->coord[place] = particle[ip];
          members->time[place] = t;
        }
        filled[iSlice]++;
      }
    }
    if (!pass) {
      long total = 0;
      for (iSlice = 0; iSlice < nSlices; iSlice++) {
        members->start[iSlice] = total;
        total += filled[iSlice];
      }
      members->start[nSlices] = total;
      members->coord = tmalloc(sizeof(*members->coord) * (total + 1));
      members->time = tmalloc(sizeof(*members->time) * (total + 1));
    }
  }
  free(filled);
  return 1;
}

/* With SLICE on the device: the slice's sums from the device's (gpu_slice_point_sums()), put together as
 * accumulate_beam_sums() puts its own together for a beam of at least one particle with BEAM_SUMS_SPARSE and
 * BEAM_SUMS_NOMINMAX.  Its spin sums and exactNormalizedEmittance's moments aren't output here. */
static void sliceSumsFromDevice(BEAM_SUMS *sums, long iSlice, const long *count, const double *columnSum,
                                const double *sigmaSum, double Po, double mp_charge) {
  static const int pair[GPU_SLICE_SIGMA_PAIRS][2] = GPU_SLICE_SIGMA_PAIR_LIST;
  int64_t npCount = count[iSlice];
  long i, j, k;
  double Sij;

  if (!sums->n_part)
    sums->p0 = Po;
  for (i = 0; i < 7; i++)
    if (npCount)
      sums->centroid[i] = (sums->centroid[i] * sums->n_part + columnSum[iSlice * GPU_SLICE_COLUMNS + i]) /
        (sums->n_part + npCount);
  for (i = 0; i < 7; i++)
    for (j = i; j < 7; j++)
      sums->beamSums2->sigma[j][i] = sums->beamSums2->sigma[i][j] = 0;
  for (k = 0; k < GPU_SLICE_SIGMA_PAIRS; k++) {
    i = pair[k][0];
    j = pair[k][1];
    Sij = sigmaSum[iSlice * GPU_SLICE_SIGMA_PAIRS + k];
    sums->beamSums2->sigma[j][i] = (sums->beamSums2->sigma[i][j] = (sums->beamSums2->sigma[i][j] * sums->n_part + Sij) /
                                    (sums->n_part + npCount));
  }
  sums->charge = mp_charge * npCount;
  sums->n_part += npCount;
}

#  ifdef GPU_VERIFY
static void compareSliceValue(double gpu, double cpu, const char *label) {
  if (!(gpu == cpu || (isnan(gpu) && isnan(cpu))))
    compareGpuCpuValue(gpu, cpu, 0.0, label);
}
#  endif

static void freeSliceMembers(SLICE_MEMBERS *members) {
  if (members->start)
    free(members->start);
  if (members->coord)
    free(members->coord);
  if (members->time)
    free(members->time);
  if (members->t0)
    free(members->t0);
  memset(members, 0, sizeof(*members));
}
#endif

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
#if defined(HAVE_GPU) && !USE_MPI
  SLICE_MEMBERS members;
  long membersFound = 0;
  /* SLICE on the device: its numbers from there (deviceSlices), or a read-only host copy */
  long deviceSlices = 0, hostCopy = 0, savedElementOnGpu = 0, hostTimes = 1, *deviceCount = NULL;
  double *deviceT0 = NULL, *deviceColumnSum = NULL, *deviceSigmaSum = NULL;
#  ifdef GPU_VERIFY
  double deviceTMin = 0, deviceTMax = 0;
  BEAM_SUMS *cpuSums = NULL;
#  endif
#endif
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

#if defined(HAVE_GPU) && !USE_MPI
  if (getElementOnGpu()) {
    if (slicePoint->nSlices > 0 &&
        gpu_slice_point_times(particles, Po, trackingClockOffset(), slicePoint->startPID, slicePoint->endPID,
                              &tMin, &tMax)) {
      /* the slices' lower limits as the loop below makes them; equal limits (every particle in the slice) are left
       * to the host, as findSliceMembers() leaves them */
      double t0Next = tMin, dt0Device = (tMax - tMin) / slicePoint->nSlices;
      long ok = dt0Device > 0;
      deviceT0 = tmalloc(sizeof(*deviceT0) * slicePoint->nSlices);
      for (iSlice = 0; ok && iSlice < slicePoint->nSlices; iSlice++, t0Next += dt0Device) {
        deviceT0[iSlice] = t0Next;
        ok = t0Next < t0Next + dt0Device;
      }
      if (ok) {
        deviceCount = tmalloc(sizeof(*deviceCount) * slicePoint->nSlices);
        deviceColumnSum = tmalloc(sizeof(*deviceColumnSum) * slicePoint->nSlices * GPU_SLICE_COLUMNS);
        deviceSigmaSum = tmalloc(sizeof(*deviceSigmaSum) * slicePoint->nSlices * GPU_SLICE_SIGMA_PAIRS);
        deviceSlices = gpu_slice_point_sums(particles, slicePoint->startPID, slicePoint->endPID, slicePoint->nSlices,
                                            deviceT0, dt0Device, deviceCount, deviceColumnSum, deviceSigmaSum);
      }
    }
#  ifdef GPU_VERIFY
    deviceTMin = tMin;
    deviceTMax = tMax;
#  endif
    if (!deviceSlices
#  ifdef GPU_VERIFY
        || 1 /* the CPU's numbers too, to compare */
#  endif
    ) {
      particle = copyParticlesToCpuReadOnly(deviceSlices ? "SLICE verification" : "SLICE the device can't slice");
      savedElementOnGpu = gpu_port_host_begin();
      hostCopy = 1;
    }
    hostTimes = !deviceSlices || hostCopy;
  }
#endif
  timeCoord = tmalloc(sizeof(*timeCoord) * particles);
#if defined(HAVE_GPU) && !USE_MPI
  if (hostTimes) {
#endif
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
      /* true absolute time; slice bounds derive from these min/max so slicing
         is invariant (see trackingClockOffset) */
      timeCoord[i] = particle[i][4] / (c_mks * beta) + trackingClockOffset();
      if (timeCoord[i] > tMax)
        tMax = timeCoord[i];
      if (timeCoord[i] < tMin)
        tMin = timeCoord[i];
    } else {
      timeCoord[i] = -DBL_MAX;
    }
  }
#if defined(HAVE_GPU) && !USE_MPI
  }
#  ifdef GPU_VERIFY
  if (deviceSlices) {
    compareSliceValue(deviceTMin, tMin, "SLICE earliest time");
    compareSliceValue(deviceTMax, tMax, "SLICE latest time");
  }
#  endif
#endif
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
#if defined(HAVE_GPU) && !USE_MPI
  if (!deviceSlices && sliceSinglePassEnabled() && !getElementOnGpu())
    membersFound = findSliceMembers(&members, slicePoint, particle, particles, timeCoord, tMin, dt0);
#  ifdef GPU_VERIFY
  if (deviceSlices)
    cpuSums = allocateBeamSums(0, 1);
#  endif
#endif
  for (iSlice = 0; iSlice < slicePoint->nSlices; iSlice++, t0 += dt0) {
#ifdef DEBUG
    printf("iSlice = %ld, t:[%le, %le]\n", iSlice, t0, t0 + dt0);
    fflush(stdout);
#endif
    /* compute centroids, sigmas, and emittances for x, y, and s */
    zero_beam_sums(sums, 1);
#if defined(HAVE_GPU) && !USE_MPI
    if (deviceSlices) {
      sliceSumsFromDevice(sums, iSlice, deviceCount, deviceColumnSum, deviceSigmaSum, Po, mp_charge);
#  ifdef GPU_VERIFY
      {
        long ic, jc;
        zero_beam_sums(cpuSums, 1);
        accumulate_beam_sums(cpuSums, particle, particles, Po, mp_charge, timeCoord, t0, t0 + dt0,
                             slicePoint->startPID, slicePoint->endPID, BEAM_SUMS_SPARSE | BEAM_SUMS_NOMINMAX);
        compareSliceValue((double)sums->n_part, (double)cpuSums->n_part, "SLICE particles in a slice");
        compareSliceValue(sums->charge, cpuSums->charge, "SLICE charge");
        for (ic = 0; ic < 7; ic++) {
          compareSliceValue(sums->centroid[ic], cpuSums->centroid[ic], "SLICE centroid");
          for (jc = 0; jc < 7; jc++)
            compareSliceValue(sums->beamSums2->sigma[ic][jc], cpuSums->beamSums2->sigma[ic][jc], "SLICE sigma matrix");
        }
      }
#  endif
    } else if (membersFound) {
      long count = members.start[iSlice + 1] - members.start[iSlice];
      if (count)
        accumulate_beam_sums(sums, members.coord + members.start[iSlice], count, Po, mp_charge,
                             members.time + members.start[iSlice], t0, t0 + dt0,
                             slicePoint->startPID, slicePoint->endPID, BEAM_SUMS_SPARSE | BEAM_SUMS_NOMINMAX);
      else {
        /* An empty slice still goes through the sums, which leave 0/0 in the sigma matrix: one particle
         * with a time below every slice does the same */
        double noTime = -DBL_MAX;
        accumulate_beam_sums(sums, particle, 1, Po, mp_charge, &noTime, t0, t0 + dt0,
                             slicePoint->startPID, slicePoint->endPID, BEAM_SUMS_SPARSE | BEAM_SUMS_NOMINMAX);
      }
    } else
#endif
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
                             ex_index    , emit[0],
                             ex_index + 1, emit[1],
                             ex_index + 2, emitc[0],
                             ex_index + 3, emitc[1],
                             -1)) {
        SDDS_SetError("Problem setting row values for SDDS table (dump_slice_analysis)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
    }
  }
  freeBeamSums(sums, 1);
#if defined(HAVE_GPU) && !USE_MPI
  if (membersFound)
    freeSliceMembers(&members);
#  ifdef GPU_VERIFY
  if (cpuSums)
    freeBeamSums(cpuSums, 1);
#  endif
  if (deviceT0)
    free(deviceT0);
  if (deviceCount)
    free(deviceCount);
  if (deviceColumnSum)
    free(deviceColumnSum);
  if (deviceSigmaSum)
    free(deviceSigmaSum);
  if (hostCopy)
    gpu_port_host_end(savedElementOnGpu);
#endif

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
