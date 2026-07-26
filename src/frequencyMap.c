/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: frequencyMap.c
 * purpose: Do frequency map tracking and analysis.
 *          See file frequencyMap.nl for input parameters.
 *
 * Michael Borland, 2004
 */
#include "mdb.h"
#include "track.h"
#include "frequencyMap.h"
#if defined(HAVE_GPU) && !USE_MPI
#  include "gpu_tune.h"
#endif

#define IC_X 0
#define IC_Y 1
#define IC_DELTA 2
#define IC_NUX 3
#define IC_NUY 4
#define IC_S 5
#define N_NOCHANGE_COLUMNS 6
#define IC_DNUX N_NOCHANGE_COLUMNS
#define IC_DNUY N_NOCHANGE_COLUMNS + 1
#define IC_DNU N_NOCHANGE_COLUMNS + 2
#define IC_DX N_NOCHANGE_COLUMNS + 3
#define IC_DY N_NOCHANGE_COLUMNS + 4
#define IC_DIFFUSION N_NOCHANGE_COLUMNS + 5
#define IC_DIFFUSION_RATE N_NOCHANGE_COLUMNS + 6
#define N_COLUMNS N_NOCHANGE_COLUMNS + 7
static SDDS_DEFINITION column_definition[N_COLUMNS] = {
  {"x", "&column name=x, symbol=x, units=m, type=double &end"},
  {"y", "&column name=y, symbol=y, units=m, type=double &end"},
  {"delta", "&column name=delta, type=double &end"},
  {"nux", "&column name=nux, symbol=$gn$r$bx$n, type=double &end"},
  {"nuy", "&column name=nuy, symbol=$gn$r$by$n, type=double &end"},
  {"s", "&column name=s, units=m/pass, type=double &end"},
  {"dnux", "&column name=dnux, symbol=$gDn$r$bx$n, type=double &end"},
  {"dnuy", "&column name=dnuy, symbol=$gDn$r$by$n, type=double &end"},
  {"dnu", "&column name=dnu, symbol=$gDn$r, type=double &end"},
  {"dx", "&column name=dx, symbol=$gD$rx, units=m, type=double &end"},
  {"dy", "&column name=dy, symbol=$gD$ry, units=m, type=double &end"},
  {"diffusion", "&column name=diffusion, symbol=\"log$b10$n($gDn$r$bx$n$a2$n+$gDn$r$bx$n$a2$n)\", type=double &end"},
  {"diffusionRate", "&column name=diffusionRate, symbol=\"log$b10$n($sr$e($gDn$r$bx$n$a2$n+$gDn$r$bx$n$a2$n)/Turns)\", type=double &end"}};

#define IP_STEP 0
#define N_PARAMETERS 2
static SDDS_DEFINITION parameter_definition[N_PARAMETERS] = {
  {"Step", "&parameter name=Step, type=long, description=\"Simulation step\" &end"},
  {"SVNVersion", "&parameter name=SVNVersion, type=string, description=\"SVN version number\", fixed_value=" SVN_VERSION " &end"},
};

static SDDS_DATASET SDDS_fmap;

#if defined(HAVE_GPU) && !USE_MPI
static long doFrequencyMapBatched(RUN *run, VARY *control,
                                  double *referenceCoord,
                                  LINE_LIST *beamline, long turns) {
  double **startingCoord, **endingCoord, **secondStartingCoord = NULL;
  double *firstTune, *secondTune, *firstAmplitude, *secondAmplitude;
  double *activeSecondTune = NULL, *activeSecondAmplitude = NULL;
  double *xAmplitude, *yAmplitude, *deltaOffset;
  double *gridX, *gridY, *gridDelta;
  double dx = 0, dy = 0, ddelta = 0, diffusion;
  long cpuTracking;
  long *firstValid, *secondValid, *firstSurvived;
  long *activeSecondValid = NULL, *secondIndex = NULL;
  long idelta, ix, iy, ip, points, row, secondParticles = 0, secondSlot;

  points = ndelta * nx * ny;
  if (turns <= 1 || quadratic_spacing ||
      !full_grid_output ||
      !gpu_batched_tune_tracking_enabled(points) ||
      !gpu_batched_frequency_map_beamline_supported(beamline))
    return 0;
  gpu_batched_tune_tracking_report("frequency map", points, turns, 1);
  cpuTracking = gpu_batched_frequency_map_cpu_tracking_required(beamline);
  startingCoord = (double **)czarray_2d(
    sizeof(**startingCoord), points, totalPropertiesPerParticle);
  endingCoord = (double **)czarray_2d(
    sizeof(**endingCoord), points, totalPropertiesPerParticle);
  firstTune = (double *)calloc(2 * points, sizeof(*firstTune));
  secondTune = include_changes ?
    (double *)calloc(2 * points, sizeof(*secondTune)) : NULL;
  firstAmplitude = (double *)calloc(2 * points, sizeof(*firstAmplitude));
  secondAmplitude = include_changes ?
    (double *)calloc(2 * points, sizeof(*secondAmplitude)) : NULL;
  xAmplitude = (double *)calloc(points, sizeof(*xAmplitude));
  yAmplitude = (double *)calloc(points, sizeof(*yAmplitude));
  deltaOffset = (double *)calloc(points, sizeof(*deltaOffset));
  gridX = (double *)calloc(points, sizeof(*gridX));
  gridY = (double *)calloc(points, sizeof(*gridY));
  gridDelta = (double *)calloc(points, sizeof(*gridDelta));
  firstValid = (long *)calloc(points, sizeof(*firstValid));
  firstSurvived = (long *)calloc(points, sizeof(*firstSurvived));
  secondValid = include_changes ?
    (long *)calloc(points, sizeof(*secondValid)) : NULL;
  if (!startingCoord || !endingCoord || !firstTune || !firstAmplitude ||
      !xAmplitude || !yAmplitude || !deltaOffset ||
      !gridX || !gridY || !gridDelta || !firstValid || !firstSurvived ||
      (include_changes &&
       (!secondTune || !secondAmplitude || !secondValid)))
    bombElegant("memory allocation failure (doFrequencyMapBatched)", NULL);

  if (!quadratic_spacing) {
    if (nx > 1)
      dx = (xmax - xmin) / (nx - 1);
    if (ny > 1)
      dy = (ymax - ymin) / (ny - 1);
  }
  if (ndelta > 1)
    ddelta = (delta_max - delta_min) / (ndelta - 1);

  ip = 0;
  for (idelta = 0; idelta < ndelta; idelta++) {
    for (ix = 0; ix < nx; ix++) {
      for (iy = 0; iy < ny; iy++, ip++) {
        memcpy(startingCoord[ip], referenceCoord,
               6 * sizeof(**startingCoord));
        gridX[ip] = xAmplitude[ip] = quadratic_spacing ?
          xmin + (xmax - xmin) * sqrt((ix + 1.) / nx) :
          xmin + ix * dx;
        gridY[ip] = yAmplitude[ip] = quadratic_spacing ?
          ymin + (ymax - ymin) * sqrt((iy + 1.) / ny) :
          ymin + iy * dy;
        gridDelta[ip] = deltaOffset[ip] = delta_min + idelta * ddelta;
      }
    }
  }
  gpu_batched_tune_tracking_set_cpu_only(cpuTracking);
  computeTunesFromTrackingBatch(
    firstTune, firstAmplitude, beamline->matrix, beamline, run,
    startingCoord, xAmplitude, yAmplitude, deltaOffset, points, turns, 0,
    endingCoord, firstValid, firstSurvived, NULL, NULL, 1, 1,
    CTFT_INCLUDE_X | CTFT_INCLUDE_Y);

  if (include_changes) {
    for (ip = 0; ip < points; ip++)
      if (firstValid[ip])
        secondParticles++;
    gpu_batched_tune_tracking_report(
      "frequency map second interval", secondParticles, turns, 1);
    if (secondParticles) {
      secondStartingCoord = (double **)calloc(
        secondParticles, sizeof(*secondStartingCoord));
      activeSecondTune = (double *)calloc(
        2 * secondParticles, sizeof(*activeSecondTune));
      activeSecondAmplitude = (double *)calloc(
        2 * secondParticles, sizeof(*activeSecondAmplitude));
      activeSecondValid = (long *)calloc(
        secondParticles, sizeof(*activeSecondValid));
      secondIndex = (long *)calloc(secondParticles, sizeof(*secondIndex));
      if (!secondStartingCoord || !activeSecondTune ||
          !activeSecondAmplitude || !activeSecondValid || !secondIndex)
        bombElegant("memory allocation failure "
                    "(doFrequencyMapBatched second interval)", NULL);
      for (ip = secondSlot = 0; ip < points; ip++) {
        if (!firstValid[ip])
          continue;
        secondStartingCoord[secondSlot] = endingCoord[ip];
        secondIndex[secondSlot++] = ip;
      }
      computeTunesFromTrackingBatch(
        activeSecondTune, activeSecondAmplitude, beamline->matrix, beamline,
        run, secondStartingCoord, NULL, NULL, NULL, secondParticles, turns,
        turns, NULL, activeSecondValid, NULL, NULL, NULL, 1, 1,
        CTFT_INCLUDE_X | CTFT_INCLUDE_Y);
      for (secondSlot = 0; secondSlot < secondParticles; secondSlot++) {
        ip = secondIndex[secondSlot];
        if (!(secondValid[ip] = activeSecondValid[secondSlot]))
          continue;
        secondTune[2 * ip] = activeSecondTune[2 * secondSlot];
        secondTune[2 * ip + 1] = activeSecondTune[2 * secondSlot + 1];
        secondAmplitude[2 * ip] =
          activeSecondAmplitude[2 * secondSlot];
        secondAmplitude[2 * ip + 1] =
          activeSecondAmplitude[2 * secondSlot + 1];
      }
    }
  }
  gpu_batched_tune_tracking_set_cpu_only(0);

  for (ip = row = 0; ip < points; ip++) {
    if (!firstValid[ip] && !full_grid_output)
      continue;
    if (!SDDS_SetRowValues(
          &SDDS_fmap, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, row,
          IC_X, gridX[ip], IC_Y, gridY[ip], IC_DELTA, gridDelta[ip],
          IC_NUX, firstValid[ip] ? firstTune[2 * ip] : -1.0,
          IC_NUY, firstValid[ip] ? firstTune[2 * ip + 1] : -1.0,
          IC_S, firstSurvived[ip] ? endingCoord[ip][4] / turns : 0.0,
          -1)) {
      SDDS_SetError("Problem setting SDDS row values "
                    "(doFrequencyMapBatched)");
      SDDS_PrintErrors(stderr,
                       SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    if (include_changes) {
      diffusion = 0;
      if (firstValid[ip] && secondValid[ip])
        diffusion = log10(
          sqr(secondTune[2 * ip] - firstTune[2 * ip]) +
          sqr(secondTune[2 * ip + 1] - firstTune[2 * ip + 1]));
      if (!SDDS_SetRowValues(
            &SDDS_fmap, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, row,
            IC_DNUX, firstValid[ip] && secondValid[ip] ?
              fabs(secondTune[2 * ip] - firstTune[2 * ip]) : 0.0,
            IC_DNUY, firstValid[ip] && secondValid[ip] ?
              fabs(secondTune[2 * ip + 1] - firstTune[2 * ip + 1]) : 0.0,
            IC_DNU, firstValid[ip] && secondValid[ip] ?
              sqrt(sqr(secondTune[2 * ip] - firstTune[2 * ip]) +
                   sqr(secondTune[2 * ip + 1] -
                       firstTune[2 * ip + 1])) : 0.0,
            IC_DX, firstValid[ip] && secondValid[ip] ?
              fabs(firstAmplitude[2 * ip] - secondAmplitude[2 * ip]) : 0.0,
            IC_DY, firstValid[ip] && secondValid[ip] ?
              fabs(firstAmplitude[2 * ip + 1] -
                   secondAmplitude[2 * ip + 1]) : 0.0,
            IC_DIFFUSION, diffusion,
            IC_DIFFUSION_RATE,
            diffusion == 0 ? 0 : diffusion / 2 - log10(turns),
            -1)) {
        SDDS_SetError("Problem setting batched frequency-map changes");
        SDDS_PrintErrors(stderr,
                         SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
    }
    row++;
  }

  free(firstTune);
  free(secondTune);
  free(firstAmplitude);
  free(secondAmplitude);
  free(xAmplitude);
  free(yAmplitude);
  free(deltaOffset);
  free(gridX);
  free(gridY);
  free(gridDelta);
  free(firstValid);
  free(firstSurvived);
  free(secondValid);
  free(activeSecondTune);
  free(activeSecondAmplitude);
  free(activeSecondValid);
  free(secondIndex);
  free(secondStartingCoord);
  free_czarray_2d((void **)startingCoord, points,
                  totalPropertiesPerParticle);
  free_czarray_2d((void **)endingCoord, points,
                  totalPropertiesPerParticle);
  return 1;
}
#endif

void setupFrequencyMap(
  NAMELIST_TEXT *nltext,
  RUN *run,
  VARY *control) {
  /* process namelist input */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&frequency_map, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &frequency_map);

  /* check for data errors */
  if (!output)
    bombElegant("no output filename specified", NULL);
  if (xmin > xmax)
    bombElegant("xmin > xmax", NULL);
  if (ymin > ymax)
    bombElegant("ymin > ymax", NULL);
  if (delta_min > delta_max)
    bombElegant("delta_min > delta_max", NULL);
  if (quadratic_spacing) {
    if (xmin < 0)
      xmin = 0;
    if (ymin < 0)
      ymin = 0;
  }
  if (nx < 1)
    nx = 1;
  if (ny < 1)
    ny = 1;
  if (ndelta < 1)
    ndelta = 1;

  output = compose_filename(output, run->rootname);
#if SDDS_MPI_IO
  SDDS_fmap.parallel_io = 1;
  SDDS_MPI_Setup(&SDDS_fmap, 1, n_processors, myid, MPI_COMM_WORLD, 1);
#endif
  SDDS_ElegantOutputSetup(&SDDS_fmap, output, SDDS_BINARY, 1, "frequency map analysis",
                          run->runfile, run->lattice, parameter_definition, N_PARAMETERS,
                          column_definition,
                          include_changes ? N_COLUMNS : N_NOCHANGE_COLUMNS,
                          "setup_frequencyMap", SDDS_EOS_NEWFILE);

  if (control->n_elements_to_vary)
    if (!SDDS_DefineSimpleParameters(&SDDS_fmap, control->n_elements_to_vary,
                                     control->varied_quan_name, control->varied_quan_unit, SDDS_DOUBLE)) {
      SDDS_SetError("Unable to define additional SDDS parameters (setup_aperture_search)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
#if !SDDS_MPI_IO
  if (!SDDS_WriteLayout(&SDDS_fmap)) {
    SDDS_SetError("Unable to write SDDS layout for aperture search");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
#else
  /* Open file here for parallel IO */
  if (!SDDS_MPI_File_Open(SDDS_fmap.MPI_dataset, SDDS_fmap.layout.filename, SDDS_MPI_WRITE_ONLY))
    SDDS_MPI_BOMB("SDDS_MPI_File_Open failed.", &SDDS_fmap.MPI_dataset->MPI_file);
  if (!SDDS_MPI_WriteLayout(&SDDS_fmap))
    SDDS_MPI_BOMB("SDDS_MPI_WriteLayout failed.", &SDDS_fmap.MPI_dataset->MPI_file);
#endif
}

long doFrequencyMap(
  RUN *run,
  VARY *control,
  double *referenceCoord,
  ERRORVAL *errcon,
  LINE_LIST *beamline) {
  double firstTune[2], secondTune[2], startingCoord[6], endingCoord[6];
  double firstAmplitude[2], secondAmplitude[2];
  double dx, dy, ddelta, x, y, delta;
  long ix, iy, idelta, ip, turns;
  static double **one_part;
  double p;
  long n_part, badPoint;
  double diffusion = 0;
  /* double diffusionRate; */
#if USE_MPI
  double oldPercentage = 0;
#endif

#if SDDS_MPI_IO
  long particles;
  FILE *fpd = NULL;
  if (verbosity < 0) {
    char s[100];
    sprintf(s, "fma-debug%03d.txt", myid);
    fpd = fopen(s, "w");
  }
  particles = ndelta * nx * ny / n_processors;
  if (myid < ndelta * nx * ny % n_processors)
    particles++;
  if (!SDDS_StartPage(&SDDS_fmap, particles) ||
      !SDDS_SetParameters(&SDDS_fmap, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, 0, control->i_step, -1)) {
    SDDS_SetError("Unable to start SDDS page (do_frequencyMap)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
#else
  if (!SDDS_StartPage(&SDDS_fmap, ndelta * nx * ny) ||
      !SDDS_SetParameters(&SDDS_fmap, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, 0, control->i_step, -1)) {
    SDDS_SetError("Unable to start SDDS page (do_frequencyMap)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
#endif

#if USE_MPI
  if (verbosity && myid == 1)
    dup2(fdStdout, fileno(stdout)); /* slave will provide warnings etc */
#endif

  if (control->n_elements_to_vary) {
    for (ip = 0; ip < control->n_elements_to_vary; ip++)
      if (!SDDS_SetParameters(&SDDS_fmap, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, ip + 1,
                              control->varied_quan_value[ip], -1)) {
        SDDS_SetError("Unable to start SDDS page (do_frequencyMap)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
  }

  /* Perform fiducialization by tracking one turn */
  if (!one_part)
    one_part = (double **)czarray_2d(sizeof(**one_part), 1, MAX_PROPERTIES_PER_PARTICLE);
  n_part = 1;
  if (referenceCoord) {
    long i;
    for (i = 0; i < 6; i++)
      one_part[0][i] = referenceCoord[i];
  }
  p = run->p_central;
  if (!do_tracking(NULL, one_part, n_part, NULL, beamline, &p, (double **)NULL, (BEAM_SUMS **)NULL, (long *)NULL,
                   NULL, run, 0, TEST_PARTICLES, 1, 0,
                   NULL, NULL, NULL, NULL, NULL)) {
    printf("Error: lost particle when fiducializing\n");
    exitElegant(1);
  }

  dx = dy = 0;
  if (!quadratic_spacing) {
    if (nx > 1)
      dx = (xmax - xmin) / (nx - 1);
    if (ny > 1)
      dy = (ymax - ymin) / (ny - 1);
  }
  if (ndelta > 1)
    ddelta = (delta_max - delta_min) / (ndelta - 1);
  else
    ddelta = 0;
  ip = 0;
  if (include_changes == 0)
    turns = control->n_passes;
  else
    turns = control->n_passes / 2;
#if defined(HAVE_GPU) && !USE_MPI
  if (doFrequencyMapBatched(run, control, referenceCoord, beamline, turns))
    goto frequencyMapTrackingDone;
#endif
  for (idelta = 0; idelta < ndelta; idelta++) {
    delta = delta_min + idelta * ddelta;
    for (ix = 0; ix < nx; ix++) {
      if (quadratic_spacing) {
        x = xmin + (xmax - xmin) * sqrt((ix + 1.) / nx);
      } else {
        x = xmin + ix * dx;
      }
      for (iy = 0; iy < ny; iy++) {
        if (quadratic_spacing) {
          y = ymin + (ymax - ymin) * sqrt((iy + 1.) / ny);
        } else {
          y = ymin + iy * dy;
        }
        memcpy(startingCoord, referenceCoord, sizeof(*startingCoord) * 6);
#if USE_MPI
        if (myid == (idelta * nx * ny + ix * ny + iy) % n_processors) /* Partition the job according to particle ID */
#endif
        {
#if USE_MPI
          if (fpd) {
            fprintf(fpd, "*** Starting tracking for idelta = %ld, ix = %ld, iy = %ld\n", idelta, ix, iy);
            fflush(fpd);
          }
#endif
          badPoint = 0;
          if (!computeTunesFromTracking(firstTune, firstAmplitude,
                                        beamline->matrix, beamline, run,
                                        startingCoord, x, y, delta, turns,
                                        0, endingCoord, NULL, NULL, 1, 1, CTFT_INCLUDE_X | CTFT_INCLUDE_Y) ||
              firstTune[0] > 1.0 || firstTune[0] < 0 || firstTune[1] > 1.0 || firstTune[1] < 0) {
            if (verbosity && !USE_MPI)
              printf("Problem with particle %ld tune determination\n", ip);
            badPoint = 1;
            firstTune[0] = firstTune[1] = -1;
            /* firstTune[1] = firstTune[1] = -1; */
            if (!full_grid_output)
              continue;
          }
          if (!SDDS_SetRowValues(&SDDS_fmap, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, ip,
                                 IC_X, x, IC_Y, y, IC_DELTA, delta,
                                 IC_NUX, firstTune[0],
                                 IC_NUY, firstTune[1],
                                 IC_S, endingCoord[4] / turns,
                                 -1)) {
            SDDS_SetError("Problem setting SDDS row values (doFrequencyMap)");
            SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
          }
          if (include_changes) {
#if USE_MPI
            if (fpd) {
              fprintf(fpd, "    Starting tracking for changes\n");
              fflush(fpd);
            }
#endif
            secondTune[0] = firstTune[0];
            secondTune[1] = firstTune[1];
            secondAmplitude[0] = firstAmplitude[0];
            secondAmplitude[1] = firstAmplitude[1];
            diffusion = 0;
            if (!badPoint) {
              memcpy(startingCoord, endingCoord, sizeof(*startingCoord) * 6);
              if (!computeTunesFromTracking(secondTune, secondAmplitude,
                                            beamline->matrix, beamline, run,
                                            startingCoord, 0.0, 0.0, 0.0, turns, turns,
                                            endingCoord, NULL, NULL, 1, 1,
                                            CTFT_INCLUDE_X | CTFT_INCLUDE_Y) ||
                  secondTune[0] > 1.0 || secondTune[0] < 0 || secondTune[1] > 1.0 || secondTune[1] < 0) {
                if (verbosity && !USE_MPI)
                  printf("Problem with particle %ld tune determination\n", ip);
                if (!full_grid_output) {
                  /* If the particle is lost, it will not show in the frequency map */
                  if (SDDS_fmap.n_rows)
                    SDDS_fmap.n_rows--;
                  continue;
                }
              } else
                diffusion = log10(sqr(secondTune[0] - firstTune[0]) + sqr(secondTune[1] - firstTune[1]));
            }
            if (!SDDS_SetRowValues(&SDDS_fmap, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, ip,
                                   IC_DNUX, fabs(secondTune[0] - firstTune[0]),
                                   IC_DNUY, fabs(secondTune[1] - firstTune[1]),
                                   IC_DNU,
                                   sqrt(sqr(secondTune[0] - firstTune[0]) + sqr(secondTune[1] - firstTune[1])),
                                   IC_DX, fabs(firstAmplitude[0] - secondAmplitude[0]),
                                   IC_DY, fabs(firstAmplitude[1] - secondAmplitude[1]),
                                   IC_DIFFUSION,
                                   diffusion,
                                   IC_DIFFUSION_RATE,
                                   diffusion == 0 ? 0 : diffusion / 2 - log10(turns),
                                   -1)) {
              SDDS_SetError("Problem setting SDDS row values (doFrequencyMap)");
              SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
            }
          }
          ip++;
          if (verbosity) {
#if USE_MPI
            if (fpd) {
              fprintf(fpd, "    Completed particle\n");
              fflush(fpd);
            }
            if (myid == 1) {
              double newPercentage = 100 * (idelta * nx * ny + ix * ny + iy + 1.0) / (ndelta * nx * ny);
              if ((newPercentage - oldPercentage) >= 1) {
                double dt = delapsed_time();
                printf("About %.1f%% done after %lg s wall time, completion expected in about %lg s\n", newPercentage, dt,
                       dt / (0.01 * newPercentage + 1e-16) - dt);
                if (oldPercentage > 0.5 && newPercentage < 0.5 && (newPercentage - oldPercentage) < 3)
                  printf("Don't you have something more interesting to do besides watching this?\n");
                oldPercentage = newPercentage;
                fflush(stdout);
              }
            }
#else
            printf("Done with particle %ld of %ld\n",
                   ix * ny * ndelta + iy * ndelta + idelta + 1, nx * ny * ndelta);
            fflush(stdout);
#endif
          }
        }
      }
    }
  }

#if defined(HAVE_GPU) && !USE_MPI
frequencyMapTrackingDone:
#endif
#if USE_MPI
  if (fpd) {
    fprintf(fpd, "*** Completed work for processor.\n");
    fflush(fpd);
  }
#endif

  if (!inhibitFileSync)
    SDDS_DoFSync(&SDDS_fmap);
#if SDDS_MPI_IO
  if (!SDDS_MPI_WriteTable(&SDDS_fmap)) {
#else
  if (!SDDS_WriteTable(&SDDS_fmap)) {
#endif
    SDDS_SetError("Problem writing SDDS table (doFrequencyMap)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

#if USE_MPI
  /* disable output from first slave */
  if (myid == 1) {
#  if defined(_WIN32)
    freopen("NUL", "w", stdout);
#  else
    if (!freopen("/dev/null", "w", stdout)) {
      perror("freopen failed");
      exit(EXIT_FAILURE);
    }
#  endif
  }
#endif

#if USE_MPI
  if (fpd) {
    fprintf(fpd, "*** Waiting at barrier.\n");
    fclose(fpd);
  }
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  return (1);
}

void finishFrequencyMap() {
  if (SDDS_IsActive(&SDDS_fmap) && !SDDS_Terminate(&SDDS_fmap)) {
    SDDS_SetError("Problem terminating SDDS output (finish_aperture_search)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
}
