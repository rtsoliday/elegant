/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

#define STORED_MATRIX_TYPE(type) ((type) == T_CCBEND || (type) == T_BRAT || (type) == T_BMAPXY || (type) == T_BMAPXYZ || (type) == T_CSBEND || (type) == T_FMULT || (type) == T_KQUAD || (type) == T_BGGEXP || (type) == T_LGBEND || type == (T_DQCOR))

/* file: analyze.c
 * purpose: Do tracking to find transfer matrix and tunes.
 *          See file analyze.nl for input parameters.
 *
 * Michael Borland, 1989
 */
#include "mdb.h"
#include "track.h"
#include "analyze.h"
#include "matlib.h"

#define X_BETA_OFFSET 0
/* 6: betax, alphax, nux, dbetax/ddelta, dalphax/ddelta, xix */
#define Y_BETA_OFFSET X_BETA_OFFSET + 6
/* 6: betay, alphay, nuy, dbetay/ddelta, dalphay/ddelta, xiy */
#define DETR_OFFSET Y_BETA_OFFSET + 6
/* 1: determinant */
#define X_ETA_OFFSET DETR_OFFSET + 1
/* 2: etax, etaxp */
#define Y_ETA_OFFSET X_ETA_OFFSET + 2
/* 2: etay, etayp */
#define CLORB_ETA_OFFSET Y_ETA_OFFSET + 2
/* 4: etax, etaxp, etay, etayp including closed orbit */
#define SYMPLECTICITY_OFFSET CLORB_ETA_OFFSET + 4
/* 3: symplecticity1, 2, 3 */
#define N_ANALYSIS_COLUMNS SYMPLECTICITY_OFFSET + 3

long addMatrixOutputColumns(SDDS_DATASET *SDDSout, long output_order);
long compareElements(ELEMENT_LIST *e1, ELEMENT_LIST *e2);

SDDS_DEFINITION analysis_column[N_ANALYSIS_COLUMNS] = {
  {"betax", "&column name=betax, units=m, symbol=\"$gb$r$bx$n\", type=double &end"},
  {"alphax", "&column name=alphax, units=m, symbol=\"$gb$r$bx$n\", type=double &end"},
  {"nux", "&column name=nux, symbol=\"$gn$r$bx$n\", type=double &end"},
  {"dbetax/dp", "&column name=dbetax/dp, units=m, type=double, description=\"Derivative of betax with momentum offset\" &end"},
  {"dalphax/dp", "&column name=dalphax/dp, type=double, description=\"Derivative of alphax with momentum offset\" &end"},
  {"dnux/dp", "&column name=dnux/dp, symbol=\"$gx$r$bx$n\", type=double, description=\"Horizontal chromaticity\" &end"},
  {"betay", "&column name=betay, units=m, symbol=\"$gb$r$by$n\", type=double &end"},
  {"alphay", "&column name=alphay, units=m, symbol=\"$gb$r$bx$n\", type=double &end"},
  {"nuy", "&column name=nuy, symbol=\"$gn$r$by$n\", type=double &end"},
  {"dbetay/dp", "&column name=dbetay/dp, units=m, type=double, description=\"Derivative of betay with momentum offset\" &end"},
  {"dalphay/dp", "&column name=dalphay/dp, type=double, description=\"Derivative of alphay with momentum offset\" &end"},
  {"dnuy/dp", "&column name=dnuy/dp, symbol=\"$gx$r$by$n\", type=double, description=\"Vertical chromaticity\" &end"},
  {"detR", "&column name=detR, symbol=\"detR\", type=double &end"},
  {"etax", "&column name=etax, units=m, symbol=\"$gc$r$bx$n\", type=double &end"},
  {"etapx", "&column name=etapx, symbol=\"$gc$r$bx$n$a'$n\", type=double &end"},
  {"etay", "&column name=etay, units=m, symbol=\"$gc$ry\", type=double &end"},
  {"etapy", "&column name=etapy, symbol=\"$gc$r$by$n$a'$n\", type=double &end"},
  {"cetax", "&column name=cetax, units=m, symbol=\"$gc$r$bxc$n\", type=double &end"},
  {"cetapx", "&column name=cetapx, symbol=\"$gc$r$bxc$n$a'$n\", type=double &end"},
  {"cetay", "&column name=cetay, units=m, symbol=\"$gc$r$byc$n\", type=double &end"},
  {"cetapy", "&column name=cetapy, symbol=\"$gc$r$byc$n$a'$n\", type=double &end"},
  {"Symplecticity1", "&column name=Symplecticity1, type=double, description=\"Symplecticity check for this element's first-order matrix\" &end"},
  {"Symplecticity2", "&column name=Symplecticity2, type=double, description=\"Symplecticity check for this element's second-order matrix\" &end"},
  {"Symplecticity3", "&column name=Symplecticity3, type=double, description=\"Symplecticity check for this element's third-order matrix\" &end"},
};

#define IP_STEP 0
#define IP_SUBSTEP 1
#define N_PARAMETERS 2
static SDDS_DEFINITION parameter_definition[N_PARAMETERS] = {
  {"Step", "&parameter name=Step, type=long, description=\"Simulation step\" &end"},
  {"Substep", "&parameter name=Substep, type=long, description=\"Simulation substep\" &end"},
};

static long SDDS_analyze_initialized = 0, n_output_columns;
static SDDS_TABLE SDDS_analyze;
static FILE *fpPrintout = NULL;

typedef struct {
  double tune1[2], beta1[2], alpha1[2];
} CHROM_DERIVS;

void copyParticles(double ***coordCopy, double **coord, long np);
void performChromaticAnalysisFromMap(VMATRIX *M, TWISS *twiss, CHROM_DERIVS *chromDeriv);
void printMapAnalysisResults(FILE *fp, long printoutOrder, char *printoutFormat,
                             VMATRIX *M, TWISS *twiss, CHROM_DERIVS *chromDeriv, double *data);
void propagateTwissParameters(TWISS *twiss1, TWISS *twiss0, VMATRIX *M);

void setup_transport_analysis(
  NAMELIST_TEXT *nltext,
  RUN *run,
  VARY *control,
  ERRORVAL *errcon) {
  log_entry("setup_transport_analysis");

  /* process namelist input */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&analyze_map, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &analyze_map);

  /* check for data errors */
  if (!output) 
    bombElegant("no output filename specified", NULL);
  if (printout_order > 3) {
    printWarning("analyze_map: maximum printout order is 3", NULL);
    printout_order = 3;
  }
  if (n_points <= max_fit_order)
    bombElegant("n_points <= max_fit_order", NULL);

  n_output_columns = N_ANALYSIS_COLUMNS + control->n_elements_to_vary + errcon->n_items;

#if USE_MPI
  if (myid == 0) {
    /* In MPI mode, all output is handled by the master processor */
#endif
    if (SDDS_analyze_initialized) {
      if (SDDS_IsActive(&SDDS_analyze) && !SDDS_Terminate(&SDDS_analyze)) {
        SDDS_SetError("Problem terminating SDDS output (finish_transport_analysis)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
      SDDS_analyze_initialized = 0;
    }

    if (output) {
      output = compose_filename(output, run->rootname);
      SDDS_ElegantOutputSetup(&SDDS_analyze, output, SDDS_BINARY, 1, "transport analysis",
                              run->runfile, run->lattice, parameter_definition, N_PARAMETERS,
                              analysis_column, N_ANALYSIS_COLUMNS, "setup_transport_analysis",
                              SDDS_EOS_NEWFILE);
      if (!SDDS_DefineSimpleColumns(&SDDS_analyze, control->n_elements_to_vary,
                                    control->varied_quan_name, control->varied_quan_unit, SDDS_DOUBLE) ||
          !SDDS_DefineSimpleColumns(&SDDS_analyze, errcon->n_items, errcon->quan_name, errcon->quan_unit,
                                    SDDS_DOUBLE)) {
        SDDS_SetError("Unable to define additional SDDS columns (setup_transport_analysis)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }

      n_output_columns += addMatrixOutputColumns(&SDDS_analyze, output_order);

      SDDS_analyze_initialized = 1;

      if (!SDDS_SaveLayout(&SDDS_analyze) || !SDDS_WriteLayout(&SDDS_analyze)) {
        SDDS_SetError("Unable to write SDDS layout for transport analysis");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
    }

    if (fpPrintout)
      fclose(fpPrintout);
    fpPrintout = NULL;
    if (printout) {
      printout = compose_filename(printout, run->rootname);
      fpPrintout = fopen(printout, "w");
    }
#if USE_MPI
  }
#endif

  log_exit("setup_transport_analysis");
}

void do_transport_analysis(
  RUN *run,
  VARY *control,
  ERRORVAL *errcon,
  LINE_LIST *beamline,
  double *orbit) {
  double **initialCoord, **finalCoord, **coordError, **finalCoordCopy;
  double *data, *offset, *orbit_p, *orbit_m;
  MATRIX *R;
  double p_central;
  long n_left, n_track, i, j, k, index, code;
  TRAJECTORY *clorb = NULL;
  double stepSize[6], maximumValue[6], sum2Difference, maxAbsDifference, difference;
  VMATRIX *M = NULL;
  TWISS twiss;
  CHROM_DERIVS chromDeriv;
  double symCheck[3][2];
#if USE_MPI
  long *nToTrackCounts, my_nTrack, my_offset, n_leftTotal, nItems;
  MPI_Status status;
  long nWorking;
  distributedBeam = 0;
#endif

  if (center_on_orbit && !orbit)
    bombElegant("you've asked to center the analysis on the closed orbit, but you didn't issue a closed_orbit command", NULL);

  data = (double *)tmalloc(sizeof(*data) * n_output_columns);
  offset = (double *)tmalloc(sizeof(*offset) * 6);
  m_alloc(&R, 6, 6);
  orbit_p = tmalloc(sizeof(*orbit_p) * 6);
  orbit_m = tmalloc(sizeof(*orbit_m) * 6);

  if (center_on_orbit)
    clorb = tmalloc(sizeof(*clorb) * (beamline->n_elems + 1));

  stepSize[0] = delta_x;
  stepSize[1] = delta_xp;
  stepSize[2] = delta_y;
  stepSize[3] = delta_yp;
  stepSize[4] = delta_s;
  stepSize[5] = delta_dp;

  n_track =
    makeInitialParticleEnsemble(&initialCoord,
                                (orbit && center_on_orbit ? orbit : NULL),
                                &finalCoord, &coordError,
                                n_points, stepSize);
#if DEBUG
  if (1) {
    long i, j;
    FILE *fp;
    fp = fopen("analyzeMap.in", "w");
    fprintf(fp, "SDDS1\n");
    fprintf(fp, "&column name=x0 type=double units=m &end\n");
    fprintf(fp, "&column name=xp0 type=double &end\n");
    fprintf(fp, "&column name=y0 type=double units=m &end\n");
    fprintf(fp, "&column name=yp0 type=double &end\n");
    fprintf(fp, "&column name=s0 type=double units=m &end\n");
    fprintf(fp, "&column name=delta0 type=double &end\n");
    fprintf(fp, "&data mode=ascii no_row_counts=1 &end\n");
    for (i = 0; i < n_track; i++) {
      for (j = 0; j < 6; j++)
        fprintf(fp, "%21.15e ", initialCoord[i][j]);
      fprintf(fp, "\n");
    }
  }
#endif

  if (canonical_variables)
    /* Assume particles are generated in canonical variables (x, px, y, py, -s, delta) and
       * convert to (x, x', y, y', s, delta) coordinates for tracking.
       */
    convertFromCanonicalCoordinates(finalCoord, n_track, run->p_central, 1);

  /* Track the reference particle for fiducialization. In MPI mode, all cores do this */
  beamline->fiducial_flag = 0;
  p_central = run->p_central;
  if (verbosity > 0) {
    printf("Tracking fiducial particle (do_transport_analysis)\n");
    fflush(stdout);
  }
  code = do_tracking(NULL, finalCoord, 1, NULL, beamline, &p_central,
                     NULL, NULL, NULL, NULL, run, control->i_step,
                     FIRST_BEAM_IS_FIDUCIAL + (verbosity > 1 ? 0 : SILENT_RUNNING) + INHIBIT_FILE_OUTPUT,
                     1, 0, NULL, NULL, NULL, NULL, NULL);

  if (!code) {
    printf("Fiducial particle lost. Don't know what to do.\n");
    exitElegant(1);
  }
  if (verbosity > 0) {
    printf("Fiducialization completed\n");
    fflush(stdout);
  }

  /* Track the other particles. */
  p_central = run->p_central;
#if USE_MPI
  /* We partition by processor in MPI mode.
       The arrays are oversized, but not so large that it will hurt. 
    */
  n_left = n_track - 1;
  if (n_left < n_processors)
    nWorking = n_left;
  else
    nWorking = n_processors;
  my_nTrack = (1.0 * n_left) / nWorking + 0.5;
  if (my_nTrack < 10) {
    my_nTrack = 10;
    nWorking = (1.0 * n_track) / my_nTrack + 0.5;
  }
  if (my_nTrack * nWorking > n_left) {
    nWorking = n_left / my_nTrack;
    my_nTrack = n_left / nWorking;
  }
  my_offset = myid * my_nTrack + 1;
  if (myid == (nWorking - 1))
    my_nTrack = n_left - my_offset + 1;
  else if (myid >= nWorking)
    my_nTrack = my_offset = n_left = 0;
#  if MPI_DEBUG
  printf("Tracking %ld particles, offset %ld, processor %d\n",
         my_nTrack, my_offset, myid);
  fflush(stdout);
#  endif
  if (my_nTrack)
    n_left = do_tracking(NULL, finalCoord + my_offset, my_nTrack, NULL, beamline, &p_central,
                         NULL, NULL, NULL, NULL, run, control->i_step,
                         FIDUCIAL_BEAM_SEEN + FIRST_BEAM_IS_FIDUCIAL + SILENT_RUNNING + INHIBIT_FILE_OUTPUT,
                         1, 0, NULL, NULL, NULL, NULL, NULL);
#  if MPI_DEBUG
  printf("Tracking done\n");
  fflush(stdout);
#  endif
  MPI_Allreduce(&n_left, &n_leftTotal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  if (n_leftTotal != (n_track - 1)) {
    char buffer[16834];
    snprintf(buffer, 16384, "%ld particles lost, continuing with next step", n_track - 1 - n_leftTotal);
    printWarning("analyze_map: particle(s) lost during transport analysis", buffer);
    return;
  }
#else
  n_left = do_tracking(NULL, finalCoord + 1, n_track - 1, NULL, beamline, &p_central,
                       NULL, NULL, NULL, NULL, run, control->i_step,
                       FIDUCIAL_BEAM_SEEN + FIRST_BEAM_IS_FIDUCIAL + SILENT_RUNNING + INHIBIT_FILE_OUTPUT,
                       1, 0, NULL, NULL, NULL, NULL, NULL);
  if (n_left != (n_track - 1)) {
    char buffer[16834];
    snprintf(buffer, 16384, "%ld particles lost, continuing with next step", n_track - 1 - n_left);
    printWarning("analyze_map: particle(s) lost during transport analysis", buffer);
    return;
  }
#endif

#if USE_MPI
  /* Gather final particles back to master */
  MPI_Barrier(MPI_COMM_WORLD);
  nToTrackCounts = tmalloc(sizeof(long) * n_processors);
  MPI_Gather(&my_nTrack, 1, MPI_LONG, nToTrackCounts, 1, MPI_LONG, 0, MPI_COMM_WORLD);
  if (myid == 0) {
    /* Copy data from each slave */
    my_nTrack += my_offset; /* to account for the fiducial particle */
    for (i = 1; i < nWorking; i++) {
      if (verbosity > 2) {
        printf("Pulling %ld particles from processor %ld\n", nToTrackCounts[i], i);
        fflush(stdout);
      }
      nItems = nToTrackCounts[i] * totalPropertiesPerParticle;
      MPI_Recv(&finalCoord[my_nTrack][0], nItems, MPI_DOUBLE, i, 100, MPI_COMM_WORLD, &status);
      my_nTrack += nToTrackCounts[i];
    }
  } else {
    /* Send data to master */
    if (my_nTrack)
      MPI_Send(&finalCoord[my_offset][0], my_nTrack * totalPropertiesPerParticle, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD);
  }
  free(nToTrackCounts);
#endif

#if USE_MPI
  if (myid == 0) {
    /* In MPI mode, only master does analysis and output */
#endif

#if DEBUG
    if (1) {
      long i, j;
      FILE *fp;
      fp = fopen("analyzeMap.out", "w");
      fprintf(fp, "SDDS1\n");
      fprintf(fp, "&column name=x type=double units=m &end\n");
      fprintf(fp, "&column name=xp type=double &end\n");
      fprintf(fp, "&column name=y type=double units=m &end\n");
      fprintf(fp, "&column name=yp type=double &end\n");
      fprintf(fp, "&column name=s type=double units=m &end\n");
      fprintf(fp, "&column name=delta type=double &end\n");
      fprintf(fp, "&data mode=ascii no_row_counts=1 &end\n");
      for (i = 0; i < n_track; i++) {
        for (j = 0; j < 6; j++)
          fprintf(fp, "%21.15e ", finalCoord[i][j]);
        fprintf(fp, "\n");
      }
    }
#endif
    if (canonical_variables)
      /* convert back to (x, px, y, py, -s, delta) for analysis */
      convertToCanonicalCoordinates(finalCoord, n_track, run->p_central, 1);

    copyParticles(&finalCoordCopy, finalCoord, n_track);
    if (p_central != run->p_central && verbosity > 0)
      printf("Central momentum changed from %e to %e\n", run->p_central, p_central);

    if (verbosity > 0) {
      if (orbit && center_on_orbit) {
        printf("closed orbit: \n");
        fflush(stdout);
        for (i = 0; i < 6; i++)
          printf("%16.8e ", orbit[i]);
        fputc('\n', stdout);
      }
      printf("final coordinates of reference particle: \n");
      fflush(stdout);
      for (i = 0; i < 6; i++)
        printf("%16.8e ", finalCoord[0][i]);
      fputc('\n', stdout);
      fflush(stdout);
      if (verbosity > 1) {
        for (i = 0; i < n_track - 1; i++) {
          printf("Particle %ld start/end/delta coordinates:\n", i);
          for (j = 0; j < 6; j++)
            printf("%21.15e%c", initialCoord[i][j], j == 5 ? '\n' : ' ');
          for (j = 0; j < 6; j++)
            printf("%21.15e%c", finalCoord[i][j], j == 5 ? '\n' : ' ');
          for (j = 0; j < 6; j++)
            printf("%21.15e%c", finalCoord[i][j] - finalCoord[0][j], j == 5 ? '\n' : ' ');
        }
        fflush(stdout);
      }
    }

    for (i = 0; i < 6; i++) {
      maximumValue[i] = -DBL_MAX;
      for (j = 0; j < n_track; j++) {
        if (maximumValue[i] < initialCoord[j][i])
          maximumValue[i] = initialCoord[j][i];
      }
    }

    /* Set errors as fraction of the absolute maximum coordinate */
    for (i = 0; i < 6; i++) {
      double min, max;
      max = -(min = DBL_MAX);
      for (j = 0; j < n_track; j++) {
        if (max < finalCoord[j][i])
          max = finalCoord[j][i];
        if (min > finalCoord[j][i])
          min = finalCoord[j][i];
      }
      max = fabs(max);
      min = fabs(min);
      max = max > min ? max : min;
      for (j = 0; j < n_track; j++)
        coordError[j][i] = max * accuracy_factor;
    }

    M = computeMatricesFromTracking(stdout, initialCoord, finalCoord, coordError, stepSize,
                                    maximumValue, n_points, n_track, max_fit_order, verbosity > 1 ? 1 : 0);
    performChromaticAnalysisFromMap(M, &twiss, &chromDeriv);

    data[X_BETA_OFFSET] = twiss.betax;
    data[X_BETA_OFFSET + 1] = twiss.alphax;
    data[X_BETA_OFFSET + 2] = twiss.phix / PIx2;
    data[X_BETA_OFFSET + 3] = chromDeriv.beta1[0];
    data[X_BETA_OFFSET + 4] = chromDeriv.alpha1[0];
    data[X_BETA_OFFSET + 5] = chromDeriv.tune1[0];
    data[X_ETA_OFFSET] = twiss.etax;
    data[X_ETA_OFFSET + 1] = twiss.etapx;

    data[Y_BETA_OFFSET] = twiss.betay;
    data[Y_BETA_OFFSET + 1] = twiss.alphay;
    data[Y_BETA_OFFSET + 2] = twiss.phiy / PIx2;
    data[Y_BETA_OFFSET + 3] = chromDeriv.beta1[1];
    data[Y_BETA_OFFSET + 4] = chromDeriv.alpha1[1];
    data[Y_BETA_OFFSET + 5] = chromDeriv.tune1[1];
    data[Y_ETA_OFFSET] = twiss.etay;
    data[Y_ETA_OFFSET + 1] = twiss.etapy;

    /* compute determinant of R */
    for (i = 0; i < 6; i++)
      for (j = 0; j < 6; j++)
        R->a[i][j] = M->R[i][j];
    data[DETR_OFFSET] = m_det(R);

    if (delta_dp && center_on_orbit) {
      if (!beamline->matrix)
        beamline->matrix = full_matrix(beamline->elem, run, 1);
      find_closed_orbit(clorb, 1e-12, 1e-11, 20, beamline, beamline->matrix, run, delta_dp, 1, 0, NULL, 0.5,
                        1.05, 5, NULL, 0);
      for (i = 0; i < 6; i++)
        orbit_p[i] = clorb[0].centroid[i];
      find_closed_orbit(clorb, 1e-12, 1e-11, 20, beamline, beamline->matrix, run, -delta_dp, 1, 0, NULL, 0.5,
                        1.05, 5, NULL, 0);
      for (i = 0; i < 6; i++)
        orbit_m[i] = clorb[0].centroid[i];
      for (i = 0; i < 4; i++)
        data[CLORB_ETA_OFFSET + i] = (orbit_p[i] - orbit_m[i]) / (2 * delta_dp);
    } else
      for (i = 0; i < 4; i++)
        data[CLORB_ETA_OFFSET + i] = 0;

    if (!canonical_variables) {
      data[SYMPLECTICITY_OFFSET] = checkSymplecticity(M, canonical_variables);
      data[SYMPLECTICITY_OFFSET + 1] = data[SYMPLECTICITY_OFFSET + 2] = -1;
    } else {
      checkSymplecticity3rdOrder(M, symCheck);
      data[SYMPLECTICITY_OFFSET + 0] = symCheck[0][1];
      data[SYMPLECTICITY_OFFSET + 1] = symCheck[1][1];
      data[SYMPLECTICITY_OFFSET + 2] = symCheck[2][1];
    }

    index = N_ANALYSIS_COLUMNS;
    for (i = 0; i < control->n_elements_to_vary; i++, index++)
      data[index] = control->varied_quan_value[i];
    for (i = 0; i < errcon->n_items; i++, index++)
      data[index] = errcon->error_value[i];

    for (i = 0; i < 6; i++, index++)
      data[index] = M->C[i];
    if (output_order >= 1) {
      for (i = 0; i < 6; i++)
        for (j = 0; j < 6; j++, index++)
          data[index] = M->R[i][j];
      if (output_order >= 2) {
        for (i = 0; i < 6; i++) {
          for (j = 0; j < 6; j++) {
            for (k = 0; k <= j; k++, index++) {
              data[index] = M->T[i][j][k];
            }
          }
        }
        if (output_order > 2) {
          for (i = 0; i < 6; i++) {
            for (j = 0; j < 6; j++) {
              for (k = 0; k <= j; k++) {
                long l;
                for (l = 0; l <= k; l++, index++) {
                  data[index] = M->Q[i][j][k][l];
                }
              }
            }
          }
        }
      }
    }

    if (SDDS_analyze_initialized) {
      if (!SDDS_StartTable(&SDDS_analyze, 1)) {
        printf("Unable to start SDDS table (do_transport_analysis)");
        fflush(stdout);
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        exitElegant(1);
      }
      if (!SDDS_SetParameters(&SDDS_analyze, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE,
                              IP_STEP, control->i_step, IP_SUBSTEP, control->i_vary, -1)) {
        SDDS_SetError("Unable to set SDDS parameter values (do_transport_analysis)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
      for (i = 0; i < SDDS_ColumnCount(&SDDS_analyze); i++)
        if (!SDDS_SetRowValues(&SDDS_analyze, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, 0,
                               i, data[i], -1)) {
          printf("Unable to set SDDS column %s (do_transport_analysis)\n",
                 analysis_column[i].name);
          fflush(stdout);
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          exitElegant(1);
        }
      if (!SDDS_WriteTable(&SDDS_analyze)) {
        printf("Unable to write SDDS table (do_transport_analysis)");
        fflush(stdout);
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        exitElegant(1);
      }
      if (!inhibitFileSync)
        SDDS_DoFSync(&SDDS_analyze);
    }

    if (fpPrintout)
      printMapAnalysisResults(fpPrintout, printout_order, printout_format, M, &twiss, &chromDeriv, data);
    if (verbosity > 0)
      printMapAnalysisResults(stdout, printout_order, printout_format, M, &twiss, &chromDeriv, data);

    /* check accuracy of matrix in reproducing coordinates */
    /* all processors do this for all particles, which shouldn't take much time */
    track_particles(initialCoord, M, initialCoord, n_track);
    for (k = 0; k < 6; k++) {
      sum2Difference = maxAbsDifference = 0;
      for (i = 0; i < n_track; i++) {
        difference = fabs(initialCoord[i][k] - finalCoordCopy[i][k]);
        if (difference > maxAbsDifference)
          maxAbsDifference = difference;
        sum2Difference += sqr(difference);
      }
      printf("Error for coordinate %ld: maximum = %le, rms = %le\n",
             k, maxAbsDifference, sqrt(sum2Difference / n_track));
    }

#if USE_MPI
  }
#endif

#if USE_MPI
  if (myid == 0) {
#endif
    if (M) {
      free_matrices(M);
      free(M);
    }
    free_czarray_2d((void **)finalCoordCopy, n_track, totalPropertiesPerParticle);
#if USE_MPI
  }
#endif

  free_czarray_2d((void **)initialCoord, n_track, totalPropertiesPerParticle);
  free_czarray_2d((void **)finalCoord, n_track, totalPropertiesPerParticle);
  free_czarray_2d((void **)coordError, n_track, totalPropertiesPerParticle);
  free(data);
  free(offset);
  free(orbit_p);
  free(orbit_m);
  m_free(&R);
}

void printMapAnalysisResults(FILE *fp, long printoutOrder, char *printoutFormat,
                             VMATRIX *M, TWISS *twiss, CHROM_DERIVS *chromDeriv, double *data) {
  double meanMax[3][2] = {{0, 0}, {0, 0}, {0, 0}};
  long saveOrder;
  saveOrder = M->order;
  M->order = printout_order > 3 ? 3 : printout_order;
  print_matrices1(fp, "Matrix from fitting:", printoutFormat, M, 0.0);
  M->order = saveOrder;
  fprintf(fp, "determinant of R = 1 + %14.6e\n", data[DETR_OFFSET] - 1);
  if (delta_dp && center_on_orbit)
    fprintf(fp, "dispersion functions from closed-orbit calculations:\nx: %e m    %e\ny: %e m    %e\n",
            data[CLORB_ETA_OFFSET], data[CLORB_ETA_OFFSET + 1],
            data[CLORB_ETA_OFFSET + 2], data[CLORB_ETA_OFFSET + 3]);
  if (periodic)
    fprintf(fp, "Lattice functions computed on assumption of periodic system:\n");
  else
    fprintf(fp, "Lattice functions computed on assumption of non-periodic system:\n");
  fprintf(fp, " horizontal:   tune = %14.6e  beta = %14.6e alpha = %14.6e  eta = %14.6e  eta' = %14.6e\n",
          twiss->phix / PIx2, twiss->betax, twiss->alphax, twiss->etax, twiss->etapx);
  fprintf(fp, "               dnu/dp = %14.6e  dbeta/dp = %14.6e  dalpha/dp = %14.6e\n",
          chromDeriv->tune1[0], chromDeriv->beta1[0], chromDeriv->alpha1[0]);
  fflush(fp);
  fprintf(fp, " vertical  :   tune = %14.6e  beta = %14.6e alpha = %14.6e  eta = %14.6e  eta' = %14.6e\n",
          twiss->phiy / PIx2, twiss->betay, twiss->alphay, twiss->etay, twiss->etapy);
  fprintf(fp, "               dnu/dp = %14.6e  dbeta/dp = %14.6e  dalpha/dp = %14.6e\n",
          chromDeriv->tune1[1], chromDeriv->beta1[1], chromDeriv->alpha1[1]);
  fflush(fp);

  if (canonical_variables) {
    checkSymplecticity3rdOrder(M, meanMax);
    fprintf(fp, "\nConstant part of the Jacobian differs in absolute value from the\n");
    fprintf(fp, "   symplectic condition by terms with:\n");
    fprintf(fp, "   Mean = %e, Maximum = %e.\n", meanMax[0][0], meanMax[0][1]);
    fprintf(fp, "First order part of the Jacobian differs in absolute value from the\n");
    fprintf(fp, "   symplectic condition by a coordinate times terms with:\n");
    fprintf(fp, "   Mean = %e, Maximum = %e.\n", meanMax[1][0], meanMax[1][1]);
    if (printoutOrder >= 3) {
      fprintf(fp, "Second order part of the Jacobian differs in absolute value from the\n");
      fprintf(fp, "   symplectic condition by a product of two coordinates times terms with:\n");
      fprintf(fp, "   Mean = %e, Maximum = %e.\n", meanMax[2][0], meanMax[2][1]);
      fflush(fp);
    }
  } else {
    fprintf(fp, "\nLinear-order symplecticity check (use canonical variables for higher order): %le\n",
            checkSymplecticity(M, canonical_variables));
  }
}

void finish_transport_analysis(
  RUN *run,
  VARY *control,
  ERRORVAL *errcon,
  LINE_LIST *beamline) {
  if (SDDS_analyze_initialized) {
    if (SDDS_IsActive(&SDDS_analyze) && !SDDS_Terminate(&SDDS_analyze)) {
      SDDS_SetError("Problem terminating SDDS output (finish_transport_analysis)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  }
  SDDS_analyze_initialized = 0;
  if (fpPrintout)
    fclose(fpPrintout);
}

VMATRIX *determineMatrix(RUN *run, ELEMENT_LIST *eptr, double *startingCoord, double *stepSize) {
  double **coord;
  long n_track, i, j, nLeft;
  VMATRIX *M;
  double **R, *C;
  double defaultStep[6] = {1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5};
  long ltmp1, ltmp2;
  double dgamma, dtmp1, dP[3];

  setTrackingContext(eptr->name, eptr->occurence, eptr->type, run->rootname, eptr, -1);

#if USE_MPI
  long distributedBeam_saved = distributedBeam;

  /* All the particles should do the same thing for this routine. */
  distributedBeam = 0;
#endif

  coord = (double **)czarray_2d(sizeof(**coord), 1 + 6 * 4, totalPropertiesPerParticle);

  if (stepSize) {
    for (i = 0; i < 6; i++)
      defaultStep[i] = stepSize[i];
  }
  stepSize = defaultStep;
  for (i = 0; i < 6; i++)
    stepSize[i] *= trackingMatrixStepFactor;

  n_track = 4 * 6 + 1;
  for (j = 0; j < totalPropertiesPerParticle; j++)
    for (i = 0; i < n_track; i++)
      coord[i][j] = startingCoord ? startingCoord[j] : 0;

  /* particles 0 and 1 are for d/dx */
  coord[0][0] += stepSize[0];
  coord[1][0] -= stepSize[0];
  /* particles 2 and 3 are for d/dxp */
  coord[2][1] += stepSize[1];
  coord[3][1] -= stepSize[1];
  /* particles 4 and 5 are for d/dy */
  coord[4][2] += stepSize[2];
  coord[5][2] -= stepSize[2];
  /* particles 6 and 7 are for d/dyp */
  coord[6][3] += stepSize[3];
  coord[7][3] -= stepSize[3];
  /* particles 8 and 9 are for d/ds */
  coord[8][4] += stepSize[4];
  coord[9][4] -= stepSize[4];
  /* particles 10 and 11 are for d/delta */
  coord[10][5] += stepSize[5];
  coord[11][5] -= stepSize[5];

  /* particles 12 and 13 are for d/dx */
  coord[12][0] += 3 * stepSize[0];
  coord[13][0] -= 3 * stepSize[0];
  /* particles 14 and 15 are for d/dxp */
  coord[14][1] += 3 * stepSize[1];
  coord[15][1] -= 3 * stepSize[1];
  /* particles 16 and 17 are for d/dy */
  coord[16][2] += 3 * stepSize[2];
  coord[17][2] -= 3 * stepSize[2];
  /* particles 18 and 19 are for d/dyp */
  coord[18][3] += 3 * stepSize[3];
  coord[19][3] -= 3 * stepSize[3];
  /* particles 20 and 21 are for d/ds */
  coord[20][4] += 3 * stepSize[4];
  coord[21][4] -= 3 * stepSize[4];
  /* particles 22 and 23 are for d/delta */
  coord[22][5] += 3 * stepSize[5];
  coord[23][5] -= 3 * stepSize[5];
  /* particle n_track-1 is the reference particle (coordinates set above) */

  setObstructionsMode(0);
  switch (eptr->type) {
  case T_CSBEND:
    ltmp1 = ((CSBEND *)eptr->p_elem)->isr;
    ltmp2 = ((CSBEND *)eptr->p_elem)->synch_rad;
    ((CSBEND *)eptr->p_elem)->isr = 0;
    if (!((CSBEND *)eptr->p_elem)->synchRadInOrdinaryMatrix)
      ((CSBEND *)eptr->p_elem)->synch_rad = 0;
    track_through_csbend(coord, n_track, (CSBEND *)eptr->p_elem, 0.0, run->p_central, NULL, 0.0,
                         NULL, NULL, NULL, NULL, NULL, -1, eptr);
    ((CSBEND *)eptr->p_elem)->isr = ltmp1;
    ((CSBEND *)eptr->p_elem)->synch_rad = ltmp2;
    break;
  case T_CCBEND:
    ltmp1 = ((CCBEND *)eptr->p_elem)->isr;
    ltmp2 = ((CCBEND *)eptr->p_elem)->synch_rad;
      ((CCBEND *)eptr->p_elem)->isr = 0;
      if (!((CCBEND *)eptr->p_elem)->synchRadInOrdinaryMatrix)
        ((CCBEND *)eptr->p_elem)->synch_rad = 0;
    track_through_ccbend(coord, n_track, eptr, (CCBEND *)eptr->p_elem, run->p_central, NULL, 0.0,
                         NULL, NULL, NULL, NULL, NULL, -1, -1);
    ((CCBEND *)eptr->p_elem)->isr = ltmp1;
    ((CCBEND *)eptr->p_elem)->synch_rad = ltmp2;
    break;
  case T_LGBEND:
    ltmp1 = ((LGBEND *)eptr->p_elem)->isr;
    ltmp2 = ((LGBEND *)eptr->p_elem)->synch_rad;
    ((LGBEND *)eptr->p_elem)->isr = 0;
    if (!((LGBEND *)eptr->p_elem)->synchRadInOrdinaryMatrix)
      ((LGBEND *)eptr->p_elem)->synch_rad = 0;
    ((LGBEND *)eptr->p_elem)->isr = ((LGBEND *)eptr->p_elem)->synch_rad = 0;
    track_through_lgbend(coord, n_track, eptr, (LGBEND *)eptr->p_elem, run->p_central, NULL, 0.0,
                         NULL, NULL, NULL, NULL, NULL, -1, -1);
    ((LGBEND *)eptr->p_elem)->isr = ltmp1;
    ((LGBEND *)eptr->p_elem)->synch_rad = ltmp2;
    break;
  case T_CWIGGLER:
    ltmp1 = ((CWIGGLER *)eptr->p_elem)->isr;
    ltmp2 = ((CWIGGLER *)eptr->p_elem)->sr;
    ((CWIGGLER *)eptr->p_elem)->isr = ((CWIGGLER *)eptr->p_elem)->sr = 0;
    GWigSymplecticPass(coord, n_track, run->p_central, (CWIGGLER *)eptr->p_elem, NULL, 0, NULL);
    ((CWIGGLER *)eptr->p_elem)->isr = ltmp1;
    ((CWIGGLER *)eptr->p_elem)->sr = ltmp2;
    break;
  case T_BRAT:
    trackBRAT(coord, n_track, (BRAT *)eptr->p_elem, run->p_central, NULL);
    break;
  case T_APPLE:
    ltmp1 = ((APPLE *)eptr->p_elem)->isr;
    ltmp2 = ((APPLE *)eptr->p_elem)->sr;
    ((APPLE *)eptr->p_elem)->isr = ((APPLE *)eptr->p_elem)->sr = 0;
    APPLE_Track(coord, n_track, run->p_central, (APPLE *)eptr->p_elem);
    ((APPLE *)eptr->p_elem)->isr = ltmp1;
    ((APPLE *)eptr->p_elem)->sr = ltmp2;
    break;
  case T_UKICKMAP:
    ltmp1 = ((UKICKMAP *)eptr->p_elem)->isr;
    ltmp2 = ((UKICKMAP *)eptr->p_elem)->synchRad;
    ((UKICKMAP *)eptr->p_elem)->isr = ((UKICKMAP *)eptr->p_elem)->synchRad = 0;
    if (trackUndulatorKickMap(coord, NULL, n_track, run->p_central, (UKICKMAP *)eptr->p_elem, 0) != n_track) {
      printf("*** Error: particles lost in determineMatrix call for UKICKMAP\n");
      exitElegant(1);
    }
    ((UKICKMAP *)eptr->p_elem)->isr = ltmp1;
    ((UKICKMAP *)eptr->p_elem)->synchRad = ltmp2;
    break;
  case T_KICKMAP:
    ltmp1 = ((KICKMAP *)eptr->p_elem)->isr;
    ltmp2 = ((KICKMAP *)eptr->p_elem)->synchRad;
    ((KICKMAP *)eptr->p_elem)->isr = ((KICKMAP *)eptr->p_elem)->synchRad = 0;
    if (trackKickMap(coord, NULL, n_track, run->p_central, (KICKMAP *)eptr->p_elem, 0, NULL) != n_track) {
      printf("*** Error: particles lost in determineMatrix call for KICKMAP\n");
      exitElegant(1);
    }
    ((KICKMAP *)eptr->p_elem)->isr = ltmp1;
    ((KICKMAP *)eptr->p_elem)->synchRad = ltmp2;
    break;
  case T_BGGEXP:
    ltmp1 = ((BGGEXP *)eptr->p_elem)->isr;
    ltmp2 = ((BGGEXP *)eptr->p_elem)->synchRad;
    ((BGGEXP *)eptr->p_elem)->isr = ((BGGEXP *)eptr->p_elem)->synchRad = 0;
    trackBGGExpansion(coord, n_track, (BGGEXP *)eptr->p_elem, run->p_central, NULL, NULL);
    ((BGGEXP *)eptr->p_elem)->isr = ltmp1;
    ((BGGEXP *)eptr->p_elem)->synchRad = ltmp2;
    break;
  case T_BOFFAXE:
    ltmp1 = ((BOFFAXE *)eptr->p_elem)->isr;
    ltmp2 = ((BOFFAXE *)eptr->p_elem)->synchRad;
    ((BOFFAXE *)eptr->p_elem)->isr = ((BOFFAXE *)eptr->p_elem)->synchRad = 0;
    trackMagneticFieldOffAxisExpansion(coord, n_track, (BOFFAXE *)eptr->p_elem, run->p_central, NULL, NULL);
    ((BOFFAXE *)eptr->p_elem)->isr = ltmp1;
    ((BOFFAXE *)eptr->p_elem)->synchRad = ltmp2;
    break;
  case T_SCRIPT:
#if USE_MPI
    if (myid == 0) {
#  if MPI_DEBUG
      printf("Calling transformBeamWithScript with force serial=1 \n");
      fflush(stdout);
#  endif
      transformBeamWithScript((SCRIPT *)eptr->p_elem, run->p_central, NULL, NULL, coord, n_track,
                              NULL, 0, 2, -1.0, 1, eptr->occurence, run->backtrack, NULL, NULL);
    } else {
#  if MPI_DEBUG
      printf("myid=%d in determineMatrix for SCRIPT\n", myid);
      fflush(stdout);
#  endif
    }
#  if MPI_DEBUG
    printf("Preparing MPI_Bcast coordinates from serial tracking\n");
    fflush(stdout);
#  endif
    MPI_Bcast(&(coord[0][0]), n_track * totalPropertiesPerParticle, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#  if MPI_DEBUG
    printf("Finished MPI_Bcast coordinates from serial tracking\n");
    fflush(stdout);
    if (1) {
      for (i = 0; i < n_track; i++)
        printf("%le %le %le %le %le %le\n",
               coord[i][0], coord[i][1], coord[i][2], coord[i][3], coord[i][4], coord[i][5]);
      fflush(stdout);
    }
#  endif
#else
    /* Serial version */
    transformBeamWithScript((SCRIPT *)eptr->p_elem, run->p_central, NULL, NULL, coord, n_track,
                            NULL, 0, 2, -1, 0, eptr->occurence, run->backtrack,
			    NULL, NULL);
#endif
    break;
  case T_TWMTA:
  case T_MAPSOLENOID:
    nLeft=motion(coord, n_track, eptr->p_elem, eptr->type, &run->p_central, &dgamma, dP, NULL, 0.0);
    if (nLeft!=n_track) {
      char buffer[1024];
      snprintf(buffer, 1024, "determineMatrix: %ld of %ld probe particles lost in matrix determination for %s#%ld (pCentral=%le)",
	       n_track-nLeft, n_track, eptr->name, eptr->occurence, run->p_central);
      printWarning(buffer, NULL);
    }
    break;
  case T_TWLA:
    /*
    printf("TWLA %s#%ld: before run->p_central=%le, Pref_input=%le, Pref_output=%le\n",
	   eptr->name, eptr->occurence, run->p_central, eptr->Pref_input, eptr->Pref_output);
    */
    nLeft=motion(coord, n_track, eptr->p_elem, eptr->type, &run->p_central, &dgamma, dP, NULL, 0.0);
    if (((TW_LINAC*)(eptr->p_elem))->change_p0)
      eptr->Pref_output = eptr->Pref_output_fiducial = run->p_central;
    /*
    printf("TWLA %s#%ld: after run->p_central=%le, Pref_input=%le, Pref_output=%le\n",
	   eptr->name, eptr->occurence, run->p_central, eptr->Pref_input, eptr->Pref_output);
    */
    if (nLeft!=n_track) {
      char buffer[1024];
      snprintf(buffer, 1024, "determineMatrix: %ld of %ld probe particles lost in matrix determination for %s#%ld (pCentral=%le)",
	       n_track-nLeft, n_track, eptr->name, eptr->occurence, run->p_central);
      printWarning(buffer, NULL);
    }
    break;
  case T_RFCA:
    simple_rf_cavity(coord, n_track, (RFCA *)eptr->p_elem, NULL, &run->p_central, 0);
    break;
  case T_RFCW:
    track_through_rfcw(coord, n_track, (RFCW *)eptr->p_elem, NULL, &run->p_central, 0, run, 0, NULL);
    break;
  case T_RFDF:
    /* Don't actually use this */
    track_through_rf_deflector(coord, (RFDF *)eptr->p_elem,
                               coord, n_track, run->p_central, 0, eptr->end_pos, 0);
    break;
  case T_LSRMDLTR:
    ltmp1 = ((LSRMDLTR *)eptr->p_elem)->isr;
    ltmp2 = ((LSRMDLTR *)eptr->p_elem)->synchRad;
    dtmp1 = ((LSRMDLTR *)eptr->p_elem)->laserPeakPower;
    ((LSRMDLTR *)eptr->p_elem)->isr = ((LSRMDLTR *)eptr->p_elem)->synchRad = 0;
    ((LSRMDLTR *)eptr->p_elem)->laserPeakPower = 0;
    motion(coord, n_track, eptr->p_elem, eptr->type, &run->p_central, &dgamma, dP, NULL, 0.0);
    ((LSRMDLTR *)eptr->p_elem)->isr = ltmp1;
    ((LSRMDLTR *)eptr->p_elem)->synchRad = ltmp2;
    ((LSRMDLTR *)eptr->p_elem)->laserPeakPower = dtmp1;
    break;
  case T_FTABLE:
    field_table_tracking(coord, n_track, (FTABLE *)eptr->p_elem, run->p_central, run);
    break;
  default:
    printf("*** Error: determineMatrix called for element that is not supported!\n");
    printf("***        Please report to developers.\n");
    exitElegant(1);
    break;
  }
  setObstructionsMode(1);

  M = tmalloc(sizeof(*M));
  M->order = 1;
  initialize_matrices(M, M->order);
  R = M->R;
  C = M->C;

  for (i = 0; i < 6; i++) {
    /* i indexes the dependent quantity */

    /* Determine C[i] */
    C[i] = coord[n_track - 1][i];

    /* Compute R[i][j] */
    for (j = 0; j < 6; j++) {
      /* j indexes the initial coordinate value */
      /*
      if (n_points==2) 
        R[i][j] = (coord[2*j][i]-coord[2*j+1][i])/(2*stepSize[j]);
      else
      */
      R[i][j] =
        (27 * (coord[2 * j][i] - coord[2 * j + 1][i]) - (coord[2 * j + 12][i] - coord[2 * j + 13][i])) / (48 * stepSize[j]);
    }
  }

  free_czarray_2d((void **)coord, 1 + 4 * 6, totalPropertiesPerParticle);

  /*
  if (1) {
    char s[1024];
    if (strlen(eptr->name)<900)
      sprintf(s, "\nElement %s#%ld matrix determined from tracking:\n", eptr->name, eptr->occurence);
    print_matrices(stdout, s, M);
  }
  */

#if USE_MPI
  distributedBeam = distributedBeam_saved;
#endif
  return M;
}

#if USE_MPI
#  ifdef DEBUG
static FILE *fpdeb = NULL;
#  endif
#endif

VMATRIX *determineMatrixHigherOrder(RUN *run, ELEMENT_LIST *eptr, double *startingCoord, double *stepSize, long order) {
  double **initialCoord, **finalCoord, **coordError, **finalCopy;
  long n_track, i, j, n_left;
  VMATRIX *M  = NULL;
  double defaultStep[6];
  double maximumValue[6];
  long ltmp1, ltmp2;
  double dgamma, dtmp1, dP[3];
  long nPoints1 = trackingMatrixPoints;
  long maxFitOrder = trackingMatrixMaxFitOrder; 
  double *maxError, *maError;
#if USE_MPI
  long nWorking = 0, n_leftTotal, k, *nToTrackCounts, fiducialOnly = 0;
#endif
  /* We'll store some of the matrices to avoid recomputing them */
  static long nStoredMatrices = 0, iStoredMatrices = -1;
  static ELEMENT_LIST **storedElement = NULL;
  static VMATRIX **storedMatrix = NULL;
  static bool announcedNTracked = false;
  long my_nTrack, my_offset;
  int hasSDependence = 0;

  setTrackingContext(eptr->name, eptr->occurence, eptr->type, run->rootname, eptr, -1);

  memcpy(defaultStep, trackingMatrixStepSize, sizeof(double) * 6);

  if (nPoints1 % 2 == 0)
    nPoints1 += 1;
  if (nPoints1 < 5)
    nPoints1 = 5;

#if USE_MPI
#  ifdef DEBUG
  if (fpdeb == NULL) {
    char s[100];
    sprintf(s, "debug-%03d.txt", myid);
    fpdeb = fopen(s, "w");
  }
#  endif
#endif
  if (shareTrackingBasedMatrices && STORED_MATRIX_TYPE(eptr->type)) {
    if (storedElement == NULL) {
      storedElement = tmalloc(sizeof(*storedElement) * trackingBasedMatricesStoreLimit);
      storedMatrix = tmalloc(sizeof(*storedMatrix) * trackingBasedMatricesStoreLimit);
    }

    for (i = 0; i < nStoredMatrices; i++) {
      CCBEND *crbptr0, *crbptr1;
      BRAT *brat0, *brat1;
      BGGEXP *bgg0, *bgg1;
      LGBEND *lgbptr0, *lgbptr1;
      short copied = 0;
      if (eptr->type == storedElement[i]->type && compareElements(storedElement[i], eptr) == 0 &&
          storedMatrix[i] && storedMatrix[i]->order > 0) {
        int ii;
        M = tmalloc(sizeof(*M));
        copy_matrices(M, storedMatrix[i]);
        switch (eptr->type) {
        case T_CCBEND:
          crbptr0 = (CCBEND *)storedElement[i]->p_elem;
          crbptr1 = (CCBEND *)eptr->p_elem;
          crbptr1->optimized = crbptr0->optimized;
          crbptr1->fseOffset = crbptr0->fseOffset;
          crbptr1->dxOffset = crbptr0->dxOffset;
          crbptr1->xAdjust = crbptr0->xAdjust;
          crbptr1->KnDelta = crbptr0->KnDelta;
          for (ii=0; ii<5; ii++)
                 crbptr1->referenceData[ii] = crbptr0->referenceData[ii];
          for (ii=0; ii<4; ii++)
            crbptr1->referenceTrajectory[ii] = crbptr0->referenceTrajectory[ii];
          copied = 1;
#ifdef DEBUG_CCBEND
          printf("Using stored matrix for CCBEND %s#%ld from %s#%ld\n", eptr->name, eptr->occurence,
                 storedElement[i]->name, storedElement[i]->occurence);
          printf("optimized = %hd, fseOffset=%le, dxOffset=%le, KnDelta=%le, lengthCorrection=%le\n",
                 crbptr1->optimized,
                 crbptr1->fseOffset,
                 crbptr1->dxOffset,
                 crbptr1->KnDelta, crbptr1->lengthCorrection);
          printf("refData = %le, %le, %le, %le, %le\n",
                 crbptr1->referenceData[0],
                 crbptr1->referenceData[1],
                 crbptr1->referenceData[2],
                 crbptr1->referenceData[3],
                 crbptr1->referenceData[4]);
          fflush(stdout);
#endif
          break;
        case T_LGBEND:
          lgbptr0 = (LGBEND *)storedElement[i]->p_elem;
          lgbptr1 = (LGBEND *)eptr->p_elem;
          lgbptr1->optimized = lgbptr0->optimized;
          for (i = 0; i < lgbptr0->nSegments; i++) {
            lgbptr1->fseOpt[i] = lgbptr0->fseOpt[i];
            lgbptr1->KnDelta[i] = lgbptr0->KnDelta[i];
          }
          copied = 1;
          break;
        case T_CSBEND:
          copied = 1;
          /*
	  printf("Using stored matrix for CSBEND %s#%ld from %s#%ld\n", eptr->name, eptr->occurence,
		 storedElement[i]->name, storedElement[i]->occurence);
	  fflush(stdout);
          */
          break;
        case T_BRAT:
          brat0 = (BRAT *)storedElement[i]->p_elem;
          brat1 = (BRAT *)eptr->p_elem;
          brat1->initialized = brat0->initialized;
          brat1->dataIndex = brat0->dataIndex;
          copied = 1;
          /*
	  printf("Using stored matrix for BRAT %s#%ld from %s#%ld\n", eptr->name, eptr->occurence,
		 storedElement[i]->name, storedElement[i]->occurence);
	  fflush(stdout);
          */
          break;
        case T_BGGEXP:
          bgg0 = (BGGEXP *)storedElement[i]->p_elem;
          bgg1 = (BGGEXP *)eptr->p_elem;
          bgg1->initialized = bgg0->initialized;
          bgg1->dataIndex[0] = bgg0->dataIndex[0];
          bgg1->dataIndex[1] = bgg0->dataIndex[1];
          copied = 1;
          printf("Using stored matrix for BGGEXP %s#%ld from %s#%ld\n", eptr->name, eptr->occurence,
                 storedElement[i]->name, storedElement[i]->occurence);
          fflush(stdout);
          break;
        case T_FMULT:
          /*
	  printf("Using stored matrix for FMULT %s#%ld from %s#%ld\n", eptr->name, eptr->occurence,
		 storedElement[i]->name, storedElement[i]->occurence);
	  fflush(stdout);
          */
          copied = 1;
          break;
        case T_HKPOLY:
        case T_KQUAD:
        case T_DQCOR:
        case T_BMAPXY:
        case T_BMAPXYZ:
          copied = 1;
          break;
        default:
          break;
        }
        if (copied) {
          free_matrices_above_order(M, order);
          /*
#if USE_MPI
          print_matrices1(fpdeb, eptr->name, "%13.8e ", M);
#endif
          */
          return M;
        }
      }
    }
  }

  if (stepSize) {
    for (i = 0; i < 6; i++)
      defaultStep[i] = stepSize[i];
  }
  stepSize = defaultStep;
  for (i = 0; i < 6; i++)
    stepSize[i] *= trackingMatrixStepFactor;

  n_track = makeInitialParticleEnsemble(&initialCoord, startingCoord, &finalCoord, &coordError, nPoints1, stepSize);
  finalCopy = (double**)czarray_2d(sizeof(**finalCopy), n_track, totalPropertiesPerParticle);
    
  n_left = n_track;
  if (!announcedNTracked) {
    printf("Tracking %ld particles in total for matrix determination\n", n_track);
    fflush(stdout);
    announcedNTracked = true;
  }
#if USE_MPI
#  ifdef DEBUG
  if (fpdeb) {
    fprintf(fpdeb, "Tracking %ld particles in total for matrix determination\n", n_track);
    fflush(fpdeb);
  }
#  endif
  /* We partition by processor in MPI mode.
     The arrays are oversized, but not so large that it will hurt. 
  */
  if (parallelTrackingBasedMatrices) {
#  ifdef DEBUG
    if (fpdeb) {
      fprintf(fpdeb, "Doing parallel matrix determination (n_track=%ld)\n", n_track);
      fflush(fpdeb);
    }
#  endif
    n_left = n_track;
    nWorking = n_track < n_processors ? n_track : n_processors;
    my_nTrack = (1.0 * n_track) / nWorking + 0.5;
    if (my_nTrack < 10) {
      my_nTrack = 10;
      nWorking = (1.0 * n_track) / my_nTrack + 0.5;
    }
#  ifdef DEBUG
    if (fpdeb) {
      fprintf(fpdeb, "1: nWorking = %ld, my_nTrack = %ld\n", nWorking, my_nTrack);
      fflush(fpdeb);
    }
#  endif
    if (my_nTrack * nWorking > n_track) {
      nWorking = n_track / my_nTrack;
      my_nTrack = n_track / nWorking;
    }
#  ifdef DEBUG
    if (fpdeb) {
      fprintf(fpdeb, "2: nWorking = %ld, my_nTrack = %ld\n", nWorking, my_nTrack);
      fflush(fpdeb);
    }
#  endif
    my_offset = myid * my_nTrack;
    fiducialOnly = 0;
    if (myid == (nWorking - 1))
      my_nTrack = n_track - my_offset;
    else if (myid >= nWorking) {
      /* Force tracking of at least one particle in case the element has internal fiducialization etc. */
      my_nTrack = 1;
      my_offset = 0;
      fiducialOnly = 1;
    }
    n_left = my_nTrack; /* In case tracking routine doesn't set this */
#  ifdef DEBUG
    if (fpdeb) {
      fprintf(fpdeb, "Tracking %ld particles (offset %ld) for matrix determination\n", my_nTrack, my_offset);
      fflush(fpdeb);
    }
#  endif
  } else {
#endif
    my_offset = 0;
    my_nTrack = n_track;
#if USE_MPI
  }
#endif

  if (my_nTrack) {
    setObstructionsMode(0);
    switch (eptr->type) {
    case T_CSBEND:
      ltmp1 = ((CSBEND *)eptr->p_elem)->isr;
      ltmp2 = ((CSBEND *)eptr->p_elem)->synch_rad;
      ((CSBEND *)eptr->p_elem)->isr = 0;
      if (!((CSBEND *)eptr->p_elem)->synchRadInOrdinaryMatrix)
        ((CSBEND *)eptr->p_elem)->synch_rad = 0;
      n_left = track_through_csbend(finalCoord + my_offset, my_nTrack, (CSBEND *)eptr->p_elem, 0.0, run->p_central, NULL, 0.0,
                                    NULL, NULL, NULL, NULL, NULL, -1, eptr);
      ((CSBEND *)eptr->p_elem)->isr = ltmp1;
      ((CSBEND *)eptr->p_elem)->synch_rad = ltmp2;
      break;
    case T_CCBEND:
      ltmp1 = ((CCBEND *)eptr->p_elem)->isr;
      ltmp2 = ((CCBEND *)eptr->p_elem)->synch_rad;
      ((CCBEND *)eptr->p_elem)->isr = 0;
      if (!((CCBEND *)eptr->p_elem)->synchRadInOrdinaryMatrix)
        ((CCBEND *)eptr->p_elem)->synch_rad = 0;
#ifdef DEBUG_CCBEND
      printf("Computing tracking-based matrix for CCBEND %s#%ld\n", eptr->name, eptr->occurence);
      fflush(stdout);
#endif
      n_left = track_through_ccbend(finalCoord + my_offset, my_nTrack, eptr, (CCBEND *)eptr->p_elem, run->p_central, NULL, 0.0,
                                    NULL, NULL, NULL, NULL, NULL, -1, -1);
      ((CCBEND *)eptr->p_elem)->isr = ltmp1;
      ((CCBEND *)eptr->p_elem)->synch_rad = ltmp2;
      break;
    case T_LGBEND:
      ltmp1 = ((LGBEND *)eptr->p_elem)->isr;
      ltmp2 = ((LGBEND *)eptr->p_elem)->synch_rad;
      ((LGBEND *)eptr->p_elem)->isr = 0;
      if (!((LGBEND *)eptr->p_elem)->synchRadInOrdinaryMatrix)
        ((LGBEND *)eptr->p_elem)->synch_rad = 0;
#ifdef DEBUG_LGBEND
      printf("Computing tracking-based matrix for LGBEND %s#%ld\n", eptr->name, eptr->occurence);
      fflush(stdout);
#endif
      n_left = track_through_lgbend(finalCoord + my_offset, my_nTrack, eptr, (LGBEND *)eptr->p_elem, run->p_central, NULL, 0.0,
                                    NULL, NULL, NULL, NULL, NULL, -1, -1);
      ((LGBEND *)eptr->p_elem)->isr = ltmp1;
      ((LGBEND *)eptr->p_elem)->synch_rad = ltmp2;
      break;
    case T_CWIGGLER:
      ltmp1 = ((CWIGGLER *)eptr->p_elem)->isr;
      ltmp2 = ((CWIGGLER *)eptr->p_elem)->sr;
      ((CWIGGLER *)eptr->p_elem)->isr = ((CWIGGLER *)eptr->p_elem)->sr = 0;
      GWigSymplecticPass(finalCoord + my_offset, my_nTrack, run->p_central, (CWIGGLER *)eptr->p_elem, NULL, 0, NULL);
      ((CWIGGLER *)eptr->p_elem)->isr = ltmp1;
      ((CWIGGLER *)eptr->p_elem)->sr = ltmp2;
      break;
    case T_BRAT:
      printf("Computing tracking-based matrix for BRAT %s#%ld\n", eptr->name, eptr->occurence);
      fflush(stdout);
      n_left = trackBRAT(finalCoord + my_offset, my_nTrack, (BRAT *)eptr->p_elem, run->p_central, NULL);
      break;
    case T_BMAPXY:
      printf("Computing tracking-based matrix for BMAPXY %s#%ld\n", eptr->name, eptr->occurence);
      fflush(stdout);
      n_left = lorentz(finalCoord + my_offset, my_nTrack, (BMAPXY *)eptr->p_elem, T_BMAPXY, run->p_central, NULL,
                       NULL, NULL, NULL);
      break;
    case T_BMAPXYZ:
      printf("Computing tracking-based matrix for BMXYZ %s#%ld\n", eptr->name, eptr->occurence);
      fflush(stdout);
      ltmp1 = ((BMAPXYZ *)eptr->p_elem)->synchRad;
      n_left = lorentz(finalCoord + my_offset, my_nTrack, (BMAPXYZ *)eptr->p_elem, T_BMAPXYZ, run->p_central, NULL,
                       NULL, NULL, NULL);
      ((BMAPXYZ *)eptr->p_elem)->synchRad = ltmp1;
      break;
    case T_APPLE:
      ltmp1 = ((APPLE *)eptr->p_elem)->isr;
      ltmp2 = ((APPLE *)eptr->p_elem)->sr;
      ((APPLE *)eptr->p_elem)->isr = ((APPLE *)eptr->p_elem)->sr = 0;
      APPLE_Track(finalCoord + my_offset, my_nTrack, run->p_central, (APPLE *)eptr->p_elem);
      ((APPLE *)eptr->p_elem)->isr = ltmp1;
      ((APPLE *)eptr->p_elem)->sr = ltmp2;
      break;
    case T_UKICKMAP:
      ltmp1 = ((UKICKMAP *)eptr->p_elem)->isr;
      ltmp2 = ((UKICKMAP *)eptr->p_elem)->synchRad;
      ((UKICKMAP *)eptr->p_elem)->isr = ((UKICKMAP *)eptr->p_elem)->synchRad = 0;
      if (trackUndulatorKickMap(finalCoord + my_offset, NULL, my_nTrack, run->p_central, (UKICKMAP *)eptr->p_elem, 0) != my_nTrack) {
        printf("*** Error: particles lost in determineMatrix call for UKICKMAP\n");
        exitElegant(1);
      }
      ((UKICKMAP *)eptr->p_elem)->isr = ltmp1;
      ((UKICKMAP *)eptr->p_elem)->synchRad = ltmp2;
      break;
    case T_KICKMAP:
      ltmp1 = ((KICKMAP *)eptr->p_elem)->isr;
      ltmp2 = ((KICKMAP *)eptr->p_elem)->synchRad;
      ((KICKMAP *)eptr->p_elem)->isr = ((KICKMAP *)eptr->p_elem)->synchRad = 0;
      if (trackKickMap(finalCoord + my_offset, NULL, my_nTrack, run->p_central, (KICKMAP *)eptr->p_elem, 0, NULL) != my_nTrack) {
        printf("*** Error: particles lost in determineMatrix call for KICKMAP\n");
        exitElegant(1);
      }
      ((KICKMAP *)eptr->p_elem)->isr = ltmp1;
      ((KICKMAP *)eptr->p_elem)->synchRad = ltmp2;
      break;
    case T_BGGEXP:
      ltmp1 = ((BGGEXP *)eptr->p_elem)->isr;
      ltmp2 = ((BGGEXP *)eptr->p_elem)->synchRad;
      ((BGGEXP *)eptr->p_elem)->isr = ((BGGEXP *)eptr->p_elem)->synchRad = 0;
      trackBGGExpansion(finalCoord + my_offset, my_nTrack, (BGGEXP *)eptr->p_elem, run->p_central, NULL, NULL);
      ((BGGEXP *)eptr->p_elem)->isr = ltmp1;
      ((BGGEXP *)eptr->p_elem)->synchRad = ltmp2;
      break;
    case T_BOFFAXE:
      ltmp1 = ((BOFFAXE *)eptr->p_elem)->isr;
      ltmp2 = ((BOFFAXE *)eptr->p_elem)->synchRad;
      ((BOFFAXE *)eptr->p_elem)->isr = ((BOFFAXE *)eptr->p_elem)->synchRad = 0;
      trackMagneticFieldOffAxisExpansion(finalCoord + my_offset, my_nTrack, (BOFFAXE *)eptr->p_elem, run->p_central, NULL, NULL);
      ((BOFFAXE *)eptr->p_elem)->isr = ltmp1;
      ((BOFFAXE *)eptr->p_elem)->synchRad = ltmp2;
      break;
    case T_MAPSOLENOID:
      motion(finalCoord + my_offset, my_nTrack, eptr->p_elem, eptr->type, &run->p_central, &dgamma, dP, NULL, 0.0);
      break;
    case T_TWMTA:
    case T_TWLA:
      hasSDependence = 1;
      motion(finalCoord + my_offset, my_nTrack, eptr->p_elem, eptr->type, &run->p_central, &dgamma, dP, NULL, 0.0);
      break;
    case T_RFDF:
      /* Don't actually use this */
      hasSDependence = 1;
      track_through_rf_deflector(finalCoord + my_offset, (RFDF *)eptr->p_elem,
                                 finalCoord + my_offset, my_nTrack, run->p_central, 0, eptr->end_pos, 0);
      break;
    case T_LSRMDLTR:
      hasSDependence = 1;
      ltmp1 = ((LSRMDLTR *)eptr->p_elem)->isr;
      ltmp2 = ((LSRMDLTR *)eptr->p_elem)->synchRad;
      dtmp1 = ((LSRMDLTR *)eptr->p_elem)->laserPeakPower;
      ((LSRMDLTR *)eptr->p_elem)->isr = ((LSRMDLTR *)eptr->p_elem)->synchRad = 0;
      ((LSRMDLTR *)eptr->p_elem)->laserPeakPower = 0;
      motion(finalCoord + my_offset, my_nTrack, eptr->p_elem, eptr->type, &run->p_central, &dgamma, dP, NULL, 0.0);
      ((LSRMDLTR *)eptr->p_elem)->isr = ltmp1;
      ((LSRMDLTR *)eptr->p_elem)->synchRad = ltmp2;
      ((LSRMDLTR *)eptr->p_elem)->laserPeakPower = dtmp1;
      break;
    case T_FTABLE:
      field_table_tracking(finalCoord + my_offset, my_nTrack, (FTABLE *)eptr->p_elem, run->p_central, run);
      break;
    case T_FMULT:
      fmultipole_tracking(finalCoord + my_offset, my_nTrack, (FMULT *)eptr->p_elem, 0, run->p_central, NULL, 0);
      break;
    case T_HKPOLY:
      polynomial_hamiltonian(finalCoord + my_offset, my_nTrack, (HKPOLY *)eptr->p_elem, 0, run->p_central, NULL, 0);
      break;
    case T_KQUAD:
      ltmp1 = ((KQUAD *)eptr->p_elem)->isr;
      ltmp2 = ((KQUAD *)eptr->p_elem)->synch_rad;
      ((KQUAD *)eptr->p_elem)->isr = 0;
      if (!((KQUAD *)eptr->p_elem)->synchRadInOrdinaryMatrix)
        ((KQUAD *)eptr->p_elem)->synch_rad = 0;
      multipole_tracking2(finalCoord + my_offset, my_nTrack, eptr, 0, run->p_central, NULL, 0.0, NULL, NULL, NULL, NULL, -1);
      ((KQUAD *)eptr->p_elem)->isr = ltmp1;
      ((KQUAD *)eptr->p_elem)->synch_rad = ltmp2;
      break;
    case T_DQCOR:
      ltmp1 = ((DQCOR *)eptr->p_elem)->isr;
      ltmp2 = ((DQCOR *)eptr->p_elem)->synch_rad;
      ((DQCOR *)eptr->p_elem)->isr = 0;
      multipole_tracking2(finalCoord + my_offset, my_nTrack, eptr, 0, run->p_central, NULL, 0.0, NULL, NULL, NULL, NULL, -1);
      ((DQCOR *)eptr->p_elem)->isr = ltmp1;
      ((DQCOR *)eptr->p_elem)->synch_rad = ltmp2;
      break;
    default:
      printf("*** Error: determineMatrixHigherOrder called for element that is not supported!\n");
      printf("***        Please report to developers.\n");
      exitElegant(1);
      break;
    }
    setObstructionsMode(1);
  }
#if USE_MPI
  if (parallelTrackingBasedMatrices) {
#  ifdef DEBUG
    if (fpdeb) {
      fprintf(fpdeb, "Collecting particle counts\n");
      fflush(fpdeb);
    }
#  endif
    if (fiducialOnly)
      n_left = my_nTrack = 0;
    MPI_Allreduce(&n_left, &n_leftTotal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
#  ifdef DEBUG
    if (fpdeb) {
      fprintf(fpdeb, "Total number left is %ld, expected %ld\n", n_leftTotal, n_track);
      fflush(fpdeb);
    }
#  endif
    if (n_leftTotal != n_track)
      bombElegantVA("lost particles (%ld -> %ld) when tracking to determine matrix for %s\n",
                    n_track, n_leftTotal, eptr->name);
  } else {
#endif
    if (n_left != n_track) {
      bombElegantVA("lost particles when tracking to determine matrix for %s\n", eptr->name);
    }
#if USE_MPI
  }
#endif

#if USE_MPI
  if (parallelTrackingBasedMatrices) {
    /* Gather final particles back to master */
    MPI_Barrier(MPI_COMM_WORLD);
#  ifdef DEBUG
    if (fpdeb) {
      fprintf(fpdeb, "Passed barrier just before particle collection\n");
      fflush(fpdeb);
    }
#  endif
    nToTrackCounts = tmalloc(sizeof(long) * n_processors);
    MPI_Gather(&my_nTrack, 1, MPI_LONG, nToTrackCounts, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    if (myid == 0) {
      MPI_Status status;
      long nItems;
      /* Copy data from each slave */
      my_nTrack += my_offset; /* to account for the fiducial particle */
      for (i = 1; i < nWorking; i++) {
#  ifdef DEBUG
        if (fpdeb) {
          fprintf(fpdeb, "Trying to get %ld particles from core %ld\n", nToTrackCounts[i], i);
          fflush(fpdeb);
        }
#  endif
        if (verbosity > 2) {
          printf("Pulling %ld particles from processor %ld\n", nToTrackCounts[i], i);
          fflush(stdout);
        }
        nItems = nToTrackCounts[i] * totalPropertiesPerParticle;
        MPI_Recv(&finalCoord[my_nTrack][0], nItems, MPI_DOUBLE, i, 100, MPI_COMM_WORLD, &status);
        my_nTrack += nToTrackCounts[i];
      }
    } else {
      /* Send data to master */
      if (my_nTrack) {
#  ifdef DEBUG
        if (fpdeb) {
          fprintf(fpdeb, "Trying to send %ld particles to master core\n", my_nTrack);
          fflush(fpdeb);
        }
#  endif
        MPI_Send(&finalCoord[my_offset][0], my_nTrack * totalPropertiesPerParticle, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD);
      }
    }
    free(nToTrackCounts);
    nToTrackCounts = NULL;
  }

  if (!parallelTrackingBasedMatrices || myid == 0) {
    /* In this case, only master does analysis */
#endif

    /* Set errors as fraction of the absolute maximum coordinate */
    for (i = 0; i < 6; i++) {
      double min, max;
      max = -(min = DBL_MAX);
      maximumValue[i] = -DBL_MAX;
      for (j = 0; j < n_track; j++) {
        if (max < finalCoord[j][i])
          max = finalCoord[j][i];
        if (min > finalCoord[j][i])
          min = finalCoord[j][i];
        if (maximumValue[i] < initialCoord[j][i])
          maximumValue[i] = initialCoord[j][i];
      }
      max = fabs(max);
      min = fabs(min);
      max = max > min ? max : min;
      for (j = 0; j < n_track; j++)
        coordError[j][i] = max * accuracy_factor;

    }

    /* We need to copy finalCoord for further use, because computeMatricesFromTracking modifies it
     * to contain the fit residuals. Those are similar to, but not the same as, the residuals
     * of the matrix itself.
     */
    for (j=0; j<n_track; j++)
      memcpy(finalCopy[j], finalCoord[j], sizeof(**finalCopy)*totalPropertiesPerParticle);
    M = computeMatricesFromTracking(stdout, initialCoord, finalCoord, coordError, stepSize,
                                    maximumValue, nPoints1, n_track, maxFitOrder, 0);

    /* Compute maximum and mean absolute error for each coordinate for the matrix vs tracking */
    track_particles(finalCoord, M, initialCoord, n_track);
    maxError = calloc(6, sizeof(*maxError));
    maError = calloc(6, sizeof(*maError));
    for (i=0; i<n_track; i++) {
      double deviation;
      for (j=0; j<6; j++) {
        deviation = fabs(finalCoord[i][j]-finalCopy[i][j]);
        if (deviation>maxError[j])
          maxError[j] = deviation;
        maError[j] += deviation;
      }
    }
    for (j=0; j<6; j++)
      maError[j] /= n_track;
    M->maxError = maxError;
    M->meanAbsError = maError;

    free_matrices_above_order(M, order);
    if (!hasSDependence && trackingMatrixCleanUp)
      /* Clean up spurious s-dependency in the matrix */
      remove_s_dependent_matrix_elements(M, order);

#if USE_MPI
  }

  if (parallelTrackingBasedMatrices) {
#  ifdef DEBUG
    if (fpdeb) {
      fprintf(fpdeb, "Preparing to share matrices with workers\n");
      fflush(fpdeb);
    }
#  endif
    if (myid != 0) {
      /* distribute matrices to other processors */
      M = tmalloc(sizeof(*M));
      initialize_matrices(M, M->order = order);
    }
    MPI_Bcast(M->C, 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (i = 0; i < 6; i++)
      MPI_Bcast(M->R[i], 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (order >= 2) {
      for (i = 0; i < 6; i++)
        for (j = 0; j < 6; j++)
          MPI_Bcast(M->T[i][j], j + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      if (order >= 3)
        for (i = 0; i < 6; i++)
          for (j = 0; j < 6; j++)
            for (k = 0; k <= j; k++)
              MPI_Bcast(M->Q[i][j][k], k + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
#  ifdef DEBUG
    if (fpdeb) {
      fprintf(fpdeb, "Done sharing matrices with workers\n");
      fflush(fpdeb);
    }
#  endif
  }
#endif

  if (shareTrackingBasedMatrices && STORED_MATRIX_TYPE(eptr->type)) {
    ELEMENT_LIST *eptrCopy;
    VMATRIX *matrixCopy;

    if (nStoredMatrices < trackingBasedMatricesStoreLimit) {
      iStoredMatrices++;
      nStoredMatrices++;
    } else {
      iStoredMatrices++;
      if (iStoredMatrices == nStoredMatrices)
        iStoredMatrices = 0;
    }

#ifdef DEBUG_CCBEND
    printf("Storing tracking-based matrix for %s#%ld\n", eptr->name, eptr->occurence);
    if (eptr->type == T_CCBEND) {
      CCBEND *ccb;
      ccb = (CCBEND *)(eptr->p_elem);
      printf("optimized = %hd, fseOffset=%le, dxOffset=%le, KnDelta=%le, lengthCorrection=%le\n",
             ccb->optimized, ccb->fseOffset, ccb->dxOffset, ccb->KnDelta, ccb->lengthCorrection);
      printf("refData = %le, %le, %le, %le, %le\n",
             ccb->referenceData[0], ccb->referenceData[1], ccb->referenceData[2], ccb->referenceData[3],
             ccb->referenceData[4]);
    }
    fflush(stdout);
#endif

    if (storedElement[iStoredMatrices]) {
      fflush(stdout);
      if (storedElement[iStoredMatrices]) {
        free_elements(storedElement[iStoredMatrices]);
      }
      storedElement[iStoredMatrices] = NULL;
    }
    if (storedMatrix[iStoredMatrices]) {
      fflush(stdout);
      if (storedMatrix[iStoredMatrices]) {
        free_matrices(storedMatrix[iStoredMatrices]);
        free(storedMatrix[iStoredMatrices]);
      }
      storedMatrix[iStoredMatrices] = NULL;
    }

    eptrCopy = tmalloc(sizeof(*eptrCopy));
    copy_element(eptrCopy, eptr, 0, 0, 0, NULL);
    storedElement[iStoredMatrices] = eptrCopy;
    storedElement[iStoredMatrices]->pred =
      storedElement[iStoredMatrices]->succ = NULL;
    storedElement[iStoredMatrices]->occurence = eptr->occurence;

    matrixCopy = tmalloc(sizeof(*matrixCopy));
    copy_matrices(matrixCopy, M);
    storedMatrix[iStoredMatrices] = matrixCopy;
    fflush(stdout);
  }

  free_czarray_2d((void **)initialCoord, n_track, totalPropertiesPerParticle);
  free_czarray_2d((void **)finalCoord, n_track, totalPropertiesPerParticle);
  free_czarray_2d((void **)coordError, n_track, totalPropertiesPerParticle);
  free_czarray_2d((void **)finalCopy, n_track, totalPropertiesPerParticle);

#if USE_MPI
#  ifdef DEBUG
  print_matrices1(fpdeb, eptr->name, "%13.8e ", M, 0.0);
#  endif
#endif

  return M;
}

/* FILE *fpdeb = NULL; */

void determineRadiationMatrix(VMATRIX *Mr, RUN *run, ELEMENT_LIST *eptr, double *startingCoord, double *Dr, long nSlices, long sliceEtilted, long order) {
  CSBEND csbend;
  CSRCSBEND *csrcsbend;
  BEND *sbend;
  WIGGLER *wig;
  CCBEND ccbend;
  LGBEND lgbend;
  KQUAD kquad;
  DQCOR dqcor;
  QUAD *quad;
  CWIGGLER cwig;
  BGGEXP bggexp;
  BOFFAXE boffaxe;
  KSEXT ksext;
  SEXT *sext;
  HCOR hcor;
  VCOR vcor;
  HVCOR hvcor;
  EHCOR ehcor;
  EVCOR evcor;
  EHVCOR ehvcor;
  KICKMAP kickmap;
  double length, z;
  long i, j, k, slice, nSlices0;
  double *accumD1, *accumD2, *dtmp;
  VMATRIX *M1, *M2, *Ml1, *Mtmp;
  ELEMENT_LIST elem;
  MATRIX *Ms;
  long ignoreRadiation = 0;

  /*
  fpdeb = fopen("analyze.sdds", "w");
  fprintf(fpdeb, "SDDS1\n&column name=z type=double &end\n");
  fprintf(fpdeb, "&column name=x type=double &end\n&column name=xp type=double &end\n");
  fprintf(fpdeb, "&column name=y type=double &end\n&column name=yp type=double &end\n");
  fprintf(fpdeb, "&column name=s type=double &end\n&column name=p type=double &end\n");
  fprintf(fpdeb, "&data mode=ascii no_row_counts=1 &end\n");
*/

  setTrackingContext(eptr->name, eptr->occurence, eptr->type, run->rootname, eptr, -1);

  /* Accumulated diffusion matrix */
  accumD1 = tmalloc(21 * sizeof(*(accumD1)));
  memset(accumD1, 0, 21 * sizeof(*(accumD1)));
  accumD2 = tmalloc(21 * sizeof(*(accumD2)));
  memset(accumD2, 0, 21 * sizeof(*(accumD2)));

  /* Matrices for C, R matrix propagation: */
  initialize_matrices(M1 = tmalloc(sizeof(*M1)), order);
  initialize_matrices(M2 = tmalloc(sizeof(*M2)), order);
  /* Temporary variable for linear matrix with radiation: */
  initialize_matrices(Ml1 = tmalloc(sizeof(*Ml1)), 1);
  for (i = 0; i < 6; i++) {
    M1->R[i][i] = 1;
    M1->C[i] = startingCoord[i];
  }
  /* Matrix for sigma matrix (D matrix) propagation */
  m_alloc(&Ms, 21, 21);

  nSlices0 = nSlices;

  switch (eptr->type) {
  case T_CWIGGLER:
    /* NB: CWIGGLER will be used in single-step mode, which is why this slicing ends up
     * sub-dividing the periods 
     */
    nSlices = ((CWIGGLER *)eptr->p_elem)->periods * ((CWIGGLER *)eptr->p_elem)->stepsPerPeriod;
    break;
  case T_WIGGLER:
    nSlices0 *= 4;
    nSlices = nSlices0*(((WIGGLER *)eptr->p_elem)->poles)/2;
    break;
  case T_CCBEND:
    nSlices = fabs(((CCBEND *)eptr->p_elem)->angle / 0.005) + 1;
    if (((CCBEND *)eptr->p_elem)->nSlices > nSlices)
      nSlices = ((CCBEND *)eptr->p_elem)->nSlices;
    break;
  case T_LGBEND:
    nSlices = fabs(((LGBEND *)eptr->p_elem)->angle / ((LGBEND *)eptr->p_elem)->nSegments / 0.005) + 1;
    if (((LGBEND *)eptr->p_elem)->nSlices > nSlices)
      nSlices = ((LGBEND *)eptr->p_elem)->nSlices;
    break;
  case T_CSBEND:
    nSlices = fabs(((CSBEND *)eptr->p_elem)->angle / 0.005) + 1;
    if (((CSBEND *)eptr->p_elem)->nSlices > nSlices)
      nSlices = ((CSBEND *)eptr->p_elem)->nSlices;
    break;
  case T_SBEN:
    nSlices = fabs(((BEND *)eptr->p_elem)->angle / 0.005) + 1;
    break;
  case T_CSRCSBEND:
    nSlices = fabs(((CSRCSBEND *)eptr->p_elem)->angle / 0.005) + 1;
    if (((CSRCSBEND *)eptr->p_elem)->nSlices > nSlices)
      nSlices = ((CSRCSBEND *)eptr->p_elem)->nSlices;
    break;
  case T_KQUAD:
    nSlices = fabs(((KQUAD *)eptr->p_elem)->k1 * ((KQUAD *)eptr->p_elem)->length * 10);
    if (nSlices < ((KQUAD *)eptr->p_elem)->nSlices)
      nSlices = ((KQUAD *)eptr->p_elem)->nSlices;
    break;
  case T_DQCOR:
    nSlices = fabs(((DQCOR *)eptr->p_elem)->k1 * ((DQCOR *)eptr->p_elem)->length * 10);
    if (nSlices < ((DQCOR *)eptr->p_elem)->nSlices)
      nSlices = ((DQCOR *)eptr->p_elem)->nSlices;
    break;
  case T_KSEXT:
    nSlices = fabs(((KSEXT *)eptr->p_elem)->k2 * ((KSEXT *)eptr->p_elem)->length * 10);
    if (nSlices < ((KSEXT *)eptr->p_elem)->nSlices)
      nSlices = ((KSEXT *)eptr->p_elem)->nSlices;
    break;
  case T_SEXT:
    nSlices = fabs(((SEXT *)eptr->p_elem)->k2 * ((SEXT *)eptr->p_elem)->length * 10);
    break;
  default:
    break;
  }
  if (nSlices < nSlices0)
    nSlices = nSlices0;

  z = 0;

  elem.end_pos = eptr->end_pos;
  elem.name = NULL;
  elem.occurence = 0;
  elem.type = eptr->type;
  length = 0;
  if (eptr->type == T_LGBEND)
    nSlices *= ((LGBEND *)eptr->p_elem)->nSegments; /* nSlices is per segment for LGBEND */

  for (slice = 0; slice < nSlices; slice++) {
    switch (eptr->type) {
    case T_CSBEND:
      if (slice == 0) {
        static short printed = 1;
        memcpy(&csbend, (CSBEND *)eptr->p_elem, sizeof(CSBEND));
        csbend.isr = 0;
        csbend.nSlices = nSlices;
        csbend.refTrajectoryChangeSet = 0;
        csbend.refTrajectoryChange = NULL;
        elem.type = T_CSBEND;
        elem.p_elem = (void *)&csbend;
        csbend.integration_order = 6;
        if (csbend.referenceCorrection) {
          double **coord;
          coord = (double **)czarray_2d(sizeof(**coord), 1, totalPropertiesPerParticle);
          track_through_csbend(coord, 1, &csbend, 0, run->p_central, NULL, 0.0,
                               NULL, run->rootname, NULL, NULL, NULL, -1, &elem);
          free_czarray_2d((void **)coord, 1, totalPropertiesPerParticle);
        }
        if (!printed) {
          print_elem(stdout, &elem);
          printed = 1;
        }
      }
      break;
    case T_CCBEND:
      if (slice == 0) {
        memcpy(&ccbend, (CCBEND *)eptr->p_elem, sizeof(CCBEND));
        ccbend.isr = 0;
        ccbend.nSlices = nSlices;
        elem.type = T_CCBEND;
        elem.p_elem = (void *)&ccbend;
        ccbend.integration_order = 6;
      }
      break;
    case T_LGBEND:
      if (slice == 0) {
        memcpy(&lgbend, (LGBEND *)eptr->p_elem, sizeof(LGBEND));
        copyLGBend(&lgbend, (LGBEND *)eptr->p_elem);
	lgbend.apertureDataFile = NULL;
	lgbend.localApertureData = NULL;
        lgbend.isr = 0;
        lgbend.nSlices = nSlices / lgbend.nSegments; /* nSlices is the total for all segments */
        lgbend.integration_order = 6;
        memcpy(&elem, eptr, sizeof(elem));
        elem.p_elem = (void *)&lgbend;
      }
      break;
    case T_SBEN:
      if (slice == 0) {
        static short printed  = 1;
        elem.type = T_CSBEND;
        elem.p_elem = (void *)&csbend;
        sbend = (BEND *)eptr->p_elem;
        memset(&csbend, 0, sizeof(csbend));
        csbend.isr = 0;
        csbend.synch_rad = 1;
        csbend.length = sbend->length;
        csbend.angle = sbend->angle;
        csbend.k1 = sbend->k1;
        csbend.e[0] = sbend->e[0];
        csbend.e[1] = sbend->e[1];
        csbend.k2 = sbend->k2;
        csbend.h[0] = sbend->h[0];
        csbend.h[1] = sbend->h[1];
        csbend.hgap = sbend->hgap;
        csbend.fintBoth = sbend->fint;
        csbend.dx = sbend->dx;
        csbend.dy = sbend->dy;
        csbend.dz = sbend->dz;
        csbend.fse = sbend->fse;
        csbend.tilt = sbend->tilt;
        csbend.etilt = sbend->etilt;
        csbend.edge_effects[0] = sbend->edge_effects[0];
        csbend.edge_effects[1] = sbend->edge_effects[1];
        csbend.edge_order = sbend->edge_order;
        csbend.edgeFlags = sbend->edgeFlags;
        csbend.refTrajectoryChangeSet = 0;
        csbend.refLength = 0;
        csbend.refAngle = 0;
        csbend.refTrajectoryChange = NULL;
        csbend.e1Index = sbend->e1Index;
        csbend.e2Index = sbend->e2Index;
        csbend.integration_order = 6;
        csbend.nSlices = nSlices;
        csbend.nonlinear = 1;
        if (!printed) {
          print_elem(stdout, &elem);
          printed = 1;
        }
      }
      break;
    case T_CSRCSBEND:
      if (slice == 0) {
        elem.type = T_CSBEND;
        elem.p_elem = (void *)&csbend;
        csrcsbend = (CSRCSBEND *)eptr->p_elem;
        memset(&csbend, 0, sizeof(csbend));
        csbend.isr = 0;
        csbend.synch_rad = 1;
        csbend.length = csrcsbend->length;
        csbend.angle = csrcsbend->angle;
        csbend.k1 = csrcsbend->k1;
        csbend.e[0] = csrcsbend->e[0];
        csbend.e[1] = csrcsbend->e[1];
        csbend.k2 = csrcsbend->k2;
        csbend.h[0] = csrcsbend->h[0];
        csbend.h[1] = csrcsbend->h[1];
        csbend.hgap = csrcsbend->hgap;
        csbend.fintBoth = csrcsbend->fint;
        csbend.dx = csrcsbend->dx;
        csbend.dy = csrcsbend->dy;
        csbend.dz = csrcsbend->dz;
        csbend.fse = csrcsbend->fse;
        csbend.tilt = csrcsbend->tilt;
        csbend.etilt = csrcsbend->etilt;
        csbend.edge_effects[0] = csrcsbend->edge_effects[0];
        csbend.edge_effects[1] = csrcsbend->edge_effects[1];
        csbend.edge_order = csrcsbend->edge_order;
        csbend.edgeFlags = csrcsbend->edgeFlags;
        csbend.refTrajectoryChangeSet = 0;
        csbend.refLength = 0;
        csbend.refAngle = 0;
        csbend.refTrajectoryChange = NULL;
        csbend.e1Index = csrcsbend->e1Index;
        csbend.e2Index = csrcsbend->e2Index;
        csbend.integration_order = 6;
        if ((csbend.nSlices = fabs(csbend.angle / 0.005) + 1) < nSlices0)
          csbend.nSlices = nSlices0;
      }
      break;
    case T_KQUAD:
      if (slice == 0) {
        memcpy(&kquad, (KQUAD *)eptr->p_elem, sizeof(KQUAD));
        kquad.isr = 0;
        kquad.nSlices = nSlices;
        kquad.n_kicks = 0;
        elem.type = T_KQUAD;
        elem.p_elem = (void *)&kquad;
        kquad.integration_order = 6;
      }
      break;
    case T_DQCOR:
      if (slice == 0) {
        memcpy(&dqcor, (DQCOR *)eptr->p_elem, sizeof(DQCOR));
        dqcor.isr = 0;
        dqcor.nSlices = nSlices;
        elem.type = T_DQCOR;
        elem.p_elem = (void *)&dqcor;
        dqcor.integration_order = 6;
      }
      break;
    case T_QUAD:
      if (slice == 0) {
        quad = (QUAD *)eptr->p_elem;
        memset(&kquad, 0, sizeof(KQUAD));
        kquad.isr = 0;
        kquad.synch_rad = 1;
        kquad.length = quad->length;
        kquad.k1 = quad->k1;
        kquad.tilt = quad->tilt;
        if (quad->ffringe)
          bombElegant("Can't perform radiation matrix calculations when QUAD has nonzero FFRINGE parameter", NULL);
        kquad.dx = quad->dx;
        kquad.dy = quad->dy;
        kquad.dz = quad->dz;
        kquad.fse = quad->fse;
        kquad.xkick = quad->xkick;
        kquad.ykick = quad->ykick;
        kquad.xKickCalibration = quad->xKickCalibration;
        kquad.yKickCalibration = quad->yKickCalibration;
        kquad.nSlices = nSlices;
        kquad.integration_order = 6;
        elem.type = T_KQUAD;
        elem.p_elem = (void *)&kquad;
      }
      break;
    case T_KSEXT:
      if (slice == 0) {
        memcpy(&ksext, (KSEXT *)eptr->p_elem, sizeof(KSEXT));
        ksext.isr = 0;
        ksext.n_kicks = 0;
        ksext.nSlices = nSlices;
        if (length < 1e-6) {
          ignoreRadiation = 1;
          ksext.synch_rad = 0;
        }
        elem.type = T_KSEXT;
        elem.p_elem = (void *)&ksext;
      }
      break;
    case T_SEXT:
      if (slice == 0) {
        sext = (SEXT *)eptr->p_elem;
        memset(&ksext, 0, sizeof(KSEXT));
        ksext.isr = 0;
        ksext.synch_rad = 1;
        ksext.length = sext->length;
        if (length < 1e-6) {
          ignoreRadiation = 1;
          ksext.synch_rad = 0;
        }
        ksext.k2 = sext->k2;
        ksext.tilt = sext->tilt;
        ksext.dx = sext->dx;
        ksext.dy = sext->dy;
        ksext.dz = sext->dz;
        ksext.fse = sext->fse;
        ksext.nSlices = nSlices;
        ksext.integration_order = 6;
        elem.type = T_KSEXT;
        elem.p_elem = (void *)&ksext;
      }
      break;
    case T_RFCA:
      nSlices = 1;
      elem.type = T_RFCA;
      elem.p_elem = (void *)eptr->p_elem;
      length = ((RFCA *)eptr->p_elem)->length;
      break;
    case T_TWLA:
      nSlices = 1;
      elem.type = T_TWLA;
      elem.p_elem = (void *)eptr->p_elem;
      length = ((TW_LINAC *)eptr->p_elem)->length / nSlices;
      break;
    case T_HCOR:
      memcpy(&elem, eptr, sizeof(elem));
      elem.matrix = NULL;
      elem.p_elem = (void *)&hcor;
      memcpy(&hcor, eptr->p_elem, sizeof(hcor));
      length = (hcor.length /= nSlices);
      hcor.lEffRad /= nSlices;
      hcor.kick /= nSlices;
      hcor.isr = 0;
      break;
    case T_VCOR:
      memcpy(&elem, eptr, sizeof(elem));
      elem.matrix = NULL;
      elem.p_elem = (void *)&vcor;
      memcpy(&vcor, eptr->p_elem, sizeof(vcor));
      length = (vcor.length /= nSlices);
      vcor.lEffRad /= nSlices;
      vcor.kick /= nSlices;
      vcor.isr = 0;
      break;
    case T_HVCOR:
      memcpy(&elem, eptr, sizeof(elem));
      elem.matrix = NULL;
      elem.p_elem = (void *)&hvcor;
      memcpy(&hvcor, eptr->p_elem, sizeof(hvcor));
      length = (hvcor.length /= nSlices);
      hvcor.lEffRad /= nSlices;
      hvcor.xkick /= nSlices;
      hvcor.ykick /= nSlices;
      hvcor.isr = 0;
      break;
    case T_EHCOR:
      memcpy(&elem, eptr, sizeof(elem));
      elem.matrix = NULL;
      elem.p_elem = (void *)&ehcor;
      memcpy(&ehcor, eptr->p_elem, sizeof(ehcor));
      length = (ehcor.length /= nSlices);
      ehcor.lEffRad /= nSlices;
      ehcor.kick /= nSlices;
      ehcor.isr = 0;
      break;
    case T_EVCOR:
      memcpy(&elem, eptr, sizeof(elem));
      elem.matrix = NULL;
      elem.p_elem = (void *)&evcor;
      memcpy(&evcor, eptr->p_elem, sizeof(evcor));
      length = (evcor.length /= nSlices);
      evcor.lEffRad /= nSlices;
      evcor.kick /= nSlices;
      evcor.isr = 0;
      break;
    case T_EHVCOR:
      memcpy(&elem, eptr, sizeof(elem));
      elem.matrix = NULL;
      elem.p_elem = (void *)&ehvcor;
      memcpy(&ehvcor, eptr->p_elem, sizeof(ehvcor));
      length = (ehvcor.length /= nSlices);
      ehvcor.lEffRad /= nSlices;
      ehvcor.xkick /= nSlices;
      ehvcor.ykick /= nSlices;
      ehvcor.isr = 0;
      break;
    case T_CWIGGLER:
      if (slice == 0) {
        memcpy(&cwig, (CWIGGLER *)eptr->p_elem, sizeof(CWIGGLER));
        cwig.isr = 0;
        elem.type = T_CWIGGLER;
        elem.p_elem = (void *)&cwig;
        length = cwig.length / nSlices;
      }
      break;
    case T_WIGGLER:
      if (slice == 0) {
        wig = (WIGGLER *)eptr->p_elem;
        memset(&cwig, 0, sizeof(cwig));
        cwig.sinusoidal = 1;
        cwig.length = wig->length;
        cwig.tilt = wig->tilt;
        cwig.dx = wig->dx;
        cwig.dy = wig->dy;
        cwig.dz = wig->dz;
        cwig.periods = wig->poles / 2;
        cwig.stepsPerPeriod = nSlices0;
        cwig.poleFactor[0] = cwig.poleFactor[1] = cwig.poleFactor[2] = 1;
        cwig.sr = 1;
        cwig.integrationOrder = 4;
        if (wig->K) {
          double lambda;
          lambda = cwig.length / cwig.periods;
          cwig.ByMax = wig->K / (UNDULATOR_K_FACTOR * lambda);
        } else if (wig->radius) {
          double H;
          H = run->p_central * RIGIDITY_FACTOR;
          cwig.ByMax = H / wig->radius;
        } else
          cwig.ByMax = wig->B;
        elem.type = T_CWIGGLER;
        elem.p_elem = (void *)&cwig;
        length = cwig.length / nSlices;
      }
      break;
    case T_BGGEXP:
      if (slice == 0) {
        nSlices = 1;
        memcpy(&bggexp, (BGGEXP *)eptr->p_elem, sizeof(BGGEXP));
        bggexp.isr = 0;
        elem.type = T_BGGEXP;
        elem.p_elem = (void *)&bggexp;
      }
      break;
    case T_BOFFAXE:
      if (slice == 0) {
        nSlices = 1;
        memcpy(&boffaxe, (BOFFAXE *)eptr->p_elem, sizeof(BOFFAXE));
        boffaxe.isr = 0;
        elem.type = T_BOFFAXE;
        elem.p_elem = (void *)&boffaxe;
      }
      break;
    case T_KICKMAP:
      if (slice == 0) {
        nSlices = 1;
        memcpy(&kickmap, (KICKMAP *)eptr->p_elem, sizeof(KICKMAP));
        kickmap.isr = 0;
        elem.type = T_KICKMAP;
        elem.p_elem = (void *)&kickmap;
      }
      break;
    default:
      printf("*** Error: determineRadiationMatrix called for element (%s) that is not supported!\n", eptr->name);
      printf("***        Please report to developers.\n");
      exit(1);
      break;
    }

    /* Step 1: determine effective R matrix for this element, as well as the diffusion matrix */
    determineRadiationMatrix1(Ml1, run, &elem, M1->C, accumD2, ignoreRadiation, &z, slice);
    /*
    if (elem.type==T_LGBEND || elem.type==T_CCBEND) {
      print_matrices(stdout, "matrix1:", Ml1);
      fflush(stdout);
    }
    */

    /* Step 2: Propagate the diffusion matrix */
    fillSigmaPropagationMatrix(Ms->a, Ml1->R);
    for (i = 0; i < 21; i++)
      for (j = 0; j < 21; j++)
        accumD2[i] += Ms->a[i][j] * accumD1[j];
    dtmp = accumD1;
    accumD1 = accumD2;
    accumD2 = dtmp;

    /* Step 3: Copy the propagated C vector */
    memcpy(M2->C, Ml1->C, 6 * sizeof(*(M2->C)));

    /* Step 4: Multiply the R matrices */
    for (i = 0; i < 6; i++)
      for (j = 0; j < 6; j++) {
        M2->R[i][j] = 0;
        for (k = 0; k < 6; k++)
          M2->R[i][j] += Ml1->R[i][k] * M1->R[k][j];
      }

    Mtmp = M2;
    M2 = M1;
    M1 = Mtmp;

    elem.end_pos += length;
  }

  /* Copy the matrix to the caller's buffer */
  for (i = 0; i < 6; i++) {
    Mr->C[i] = M1->C[i];
    for (j = 0; j < 6; j++) {
      Mr->R[i][j] = M1->R[i][j];
    }
  }
  for (i = 0; i < 21; i++)
    Dr[i] = accumD1[i];

  if (eptr->type == T_BGGEXP && trackingBasedDiffusionMatrixParticles > 1) {
    BEAM_SUMS sums;
    long nPart;
    /* track an ensemble to get approximation to total diffusion matrix */
    double **part;
    bggexp.isr = 1;
    nPart = trackingBasedDiffusionMatrixParticles;
#if USE_MPI
    nPart = nPart / n_processors;
#endif
    if (nPart < 10)
      nPart = 10; /* why 10? */
    /*
    printf("Performing tracking-based determination of BGGEXP diffusion matrix with %ld particles per core\n", nPart);
    fflush(stdout);
    */
    part = (double **)czarray_2d(sizeof(**part), nPart, totalPropertiesPerParticle);
    for (i = 0; i < nPart; i++)
      for (j = 0; j < 6; j++)
        part[i][j] = 0;
    trackBGGExpansion(part, nPart, &bggexp, run->p_central, NULL, NULL);
    sums.spinSums = NULL;
    sums.beamSums2 = tmalloc(sizeof(*(sums.beamSums2)));
    zero_beam_sums(&sums, 1);
    accumulate_beam_sums(&sums, part, nPart, run->p_central, 0.0, NULL, 0.0, 0.0, -1, -1, BEAM_SUMS_NOMINMAX);
    free_czarray_2d((void **)part, nPart, totalPropertiesPerParticle);
    for (i = 0; i < 6; i++)
      for (j = 0; j < 6; j++)
        Dr[sigmaIndex3[i][j]] = sums.beamSums2->sigma[i][j];
    free(sums.beamSums2);
    /*
    printf("Tracking-based BGGEXP diffusion matrix:\n");
    for (i=0; i<6; i++) {
      for (j=0; j<6; j++)
        printf("%13.6e ", Dr[sigmaIndex3[i][j]]);
      printf("\n");
    }
    fflush(stdout);
    */
  }

  if (eptr->type == T_CSBEND) {
    if (csbend.referenceCorrection) {
      free_czarray_2d((void **)csbend.refTrajectoryChange, csbend.refSlices, 5);
      csbend.refTrajectoryChange = NULL;
    }
  }

  free(accumD1);
  free(accumD2);
  free_matrices(M2);
  tfree(M2);
  free_matrices(M1);
  tfree(M1);
  free_matrices(Ml1);
  tfree(Ml1);
  M1 = M2 = Ml1 = NULL;
  m_free(&Ms);
}

void determineRadiationMatrix1(VMATRIX *Mr, RUN *run, ELEMENT_LIST *elem, double *startingCoord, double *D, long ignoreRadiation, double *z,
                               long iSlice) {
  CSBEND *csbend;
  CCBEND *ccbend;
  LGBEND *lgbend;
  KQUAD *kquad;
  DQCOR *dqcor;
  KSEXT *ksext;
  double **coord, pCentral;
  long n_track, i, j;
  double **R, *C, Cs0;
  double stepSize[6] = {1e-5, 1e-5, 1e-5, 1e-5, 1e-3, 1e-5};
  double sigmaDelta2;
  double dP[3], dgamma;
  VMATRIX *matrix;

  coord = (double **)czarray_2d(sizeof(**coord), 1 + 6 * 2, totalPropertiesPerParticle);

  Cs0 = startingCoord ? startingCoord[4] : 0;
  n_track = 2 * 6 + 1;
  for (j = 0; j < 6; j++)
    for (i = 0; i < n_track; i++)
      coord[i][j] = (startingCoord && j != 4) ? startingCoord[j] : 0;

  /* particles 0 and 1 are for d/dx, 2 and 3 are for d/dxp, etc */
  /* particle n_track-1 is the reference particle (coordinates set above) */
  for (i = 0; i < 6; i++) {
    coord[2 * i + 0][i] += stepSize[i] * trackingMatrixStepFactor;
    coord[2 * i + 1][i] -= stepSize[i] * trackingMatrixStepFactor;
  }

  if (D) {
    for (i = 0; i < 21; i++)
      D[i] = 0;
  }
  sigmaDelta2 = 0;
  setTrackingContext(elem->name, elem->occurence, elem->type, run->rootname, elem, -1);
  setObstructionsMode(0);
  switch (elem->type) {
  case T_CSBEND:
    csbend = (CSBEND *)elem->p_elem;
    track_through_csbend(coord, n_track, csbend, 0, run->p_central, NULL, elem->end_pos - csbend->length,
                         &sigmaDelta2, run->rootname, NULL, NULL, NULL, iSlice, elem);
    break;
  case T_CCBEND:
    ccbend = (CCBEND *)elem->p_elem;
    track_through_ccbend(coord, n_track, elem, ccbend, run->p_central, NULL, elem->end_pos - ccbend->length, &sigmaDelta2,
                         run->rootname, NULL, NULL, NULL, iSlice, -1);
    break;
  case T_LGBEND:
    lgbend = (LGBEND *)elem->p_elem;
    track_through_lgbend(coord, n_track, elem, lgbend, run->p_central, NULL, elem->end_pos - lgbend->length, &sigmaDelta2,
                         run->rootname, NULL, NULL, NULL, iSlice, -1);
    break;
  case T_SBEN:
    bombElegant("This shouldn't happen", NULL);
    track_particles(coord, elem->matrix, coord, n_track);
    break;
  case T_KQUAD:
    kquad = (KQUAD *)elem->p_elem;
    multipole_tracking2(coord, n_track, elem, 0.0, run->p_central, NULL, elem->end_pos - kquad->length,
                        NULL, NULL, NULL, &sigmaDelta2, iSlice);
    break;
  case T_DQCOR:
    dqcor = (DQCOR *)elem->p_elem;
    multipole_tracking2(coord, n_track, elem, 0.0, run->p_central, NULL, elem->end_pos - dqcor->length,
                        NULL, NULL, NULL, &sigmaDelta2, iSlice);
    break;
  case T_KSEXT:
    ksext = (KSEXT *)elem->p_elem;
    multipole_tracking2(coord, n_track, elem, 0.0, run->p_central, NULL, elem->end_pos - ksext->length,
                        NULL, NULL, NULL, &sigmaDelta2, iSlice);
    break;
  case T_RFCA:
    pCentral = run->p_central;
    for (i = 0; i < n_track; i++)
      coord[i][4] += Cs0;
    Cs0 = 0;
    simple_rf_cavity(coord, n_track, (RFCA *)elem->p_elem, NULL, &pCentral, elem->end_pos);
    break;
  case T_HCOR:
  case T_VCOR:
  case T_HVCOR:
    matrix = compute_matrix(elem, run, NULL);
    track_particles(coord, matrix, coord, n_track);
    addCorrectorRadiationKick(coord, n_track, elem, elem->type, run->p_central, &sigmaDelta2, 1);
    free_matrices(matrix);
    free(matrix);
    matrix = NULL;
    break;
  case T_EHCOR:
  case T_EVCOR:
  case T_EHVCOR:
    trackThroughExactCorrector(coord, n_track, elem, run->p_central, NULL, 0, &sigmaDelta2);
    break;
  case T_KICKMAP:
    trackKickMap(coord, NULL, n_track, run->p_central, (KICKMAP *)(elem->p_elem), 0, &sigmaDelta2);
    break;
  case T_TWLA:
    pCentral = run->p_central;
    motion(coord, n_track, elem->p_elem, elem->type, &pCentral, &dgamma, dP, NULL, 0.0);
    break;
  case T_CWIGGLER:
    GWigSymplecticPass(coord, n_track, run->p_central, (CWIGGLER *)elem->p_elem, &sigmaDelta2, 1, z);
    break;
  case T_BGGEXP:
    trackBGGExpansion(coord, n_track, (BGGEXP *)elem->p_elem, run->p_central, NULL, &sigmaDelta2);
    break;
  case T_BOFFAXE:
    trackMagneticFieldOffAxisExpansion(coord, n_track, (BOFFAXE *)elem->p_elem, run->p_central, NULL, &sigmaDelta2);
    break;
  default:
    printf("*** Error: determineRadiationMatrix1 called for element (%s) that is not supported!\n", elem->name);
    printf("***        Please report to developers.\n");
    exitElegant(1);
    break;
  }
  setObstructionsMode(1);
  if (ignoreRadiation)
    sigmaDelta2 = 0;

  /*  fprintf(fpdeb, "%le %le %le %le %le %le %le\n",
          *z,
          (coord[0][0]+coord[1][0])/2,
          (coord[0][1]+coord[1][1])/2,
          (coord[0][2]+coord[1][2])/2,
          (coord[0][3]+coord[1][3])/2,
          (coord[0][4]+coord[1][4])/2,
          (coord[0][5]+coord[1][5])/2);
          */

  R = Mr->R;
  C = Mr->C;
  for (i = 0; i < 6; i++) {
    /* i indexes the dependent quantity */

    /* Determine C[i] */
    C[i] = coord[n_track - 1][i];

    /* Compute R[i][j] */
    for (j = 0; j < 6; j++) {
      /* j indexes the initial coordinate value */
      R[i][j] = (coord[2 * j][i] - coord[2 * j + 1][i]) / (2 * stepSize[j] * trackingMatrixStepFactor);
    }
  }

  D[sigmaIndex3[5][5]] = sigmaDelta2;

  C[4] += Cs0;
  free_czarray_2d((void **)coord, 1 + 2 * 6, totalPropertiesPerParticle);
}

void copyParticles(double ***coordCopy, double **coord, long np) {
  long i;
  *coordCopy = (double **)zarray_2d(sizeof(***coordCopy), np, totalPropertiesPerParticle);
  for (i = 0; i < np; i++)
    memcpy((*coordCopy)[i], coord[i], sizeof(double) * totalPropertiesPerParticle);
}

void performChromaticAnalysisFromMap(VMATRIX *M, TWISS *twiss, CHROM_DERIVS *chromDeriv) {
  long i;
  double beta, alpha, cos_phi, sin_phi, phi, det, eta, etap;

  if (periodic) {
    for (i = 0; i < 4; i += 2) {
      /* find nu, beta, alpha */
      if (fabs(cos_phi = (M->R[i][i] + M->R[i + 1][i + 1]) / 2) > 1) {
        if (i == 0)
          printWarning("analyze_map: beamline unstable for x plane", NULL);
        else
          printWarning("analyze_map: beamline unstable for y plane", NULL);
        beta = alpha = phi = 0;
      } else {
        beta = fabs(M->R[i][i + 1] / sin(acos(cos_phi)));
        sin_phi = M->R[i][i + 1] / beta;
        if ((phi = atan2(sin_phi, cos_phi)) < 0)
          phi += PIx2;
        alpha = (M->R[i][i] - M->R[i + 1][i + 1]) / (2 * sin_phi);
      }

      /* compute eta and eta' */
      if ((det = (2 - M->R[i][i] - M->R[i + 1][i + 1])) <= 0) {
        printf("error: beamline unstable for %c plane--can't match dispersion functions\n",
               i == 0 ? 'x' : 'y');
        fflush(stdout);
        det = 1e-6;
      }
      eta = ((1 - M->R[i + 1][i + 1]) * M->R[i][5] + M->R[i][i + 1] * M->R[i + 1][5]) / det;
      etap = (M->R[i + 1][i] * M->R[i][5] + (1 - M->R[i][i]) * M->R[i + 1][5]) / det;
      if (i == 0) {
        twiss->betax = beta;
        twiss->alphax = alpha;
        twiss->phix = phi;
        twiss->etax = eta;
        twiss->etapx = etap;
      } else {
        twiss->betay = beta;
        twiss->alphay = alpha;
        twiss->phiy = phi;
        twiss->etay = eta;
        twiss->etapy = etap;
      }
    }

    twiss->periodic = 1;
    computeChromaticities(&(chromDeriv->tune1[0]), &(chromDeriv->tune1[1]),
                          &(chromDeriv->beta1[0]), &(chromDeriv->beta1[1]),
                          &(chromDeriv->alpha1[0]), &(chromDeriv->alpha1[1]),
                          twiss, twiss, M);
  } else {
    TWISS twiss0;
    twiss0.betax = beta_x;
    twiss0.alphax = alpha_x;
    twiss0.phix = 0;
    twiss0.etax = eta_x;
    twiss0.etapx = etap_x;
    twiss0.betay = beta_y;
    twiss0.alphay = alpha_y;
    twiss0.phiy = 0;
    twiss0.etay = eta_y;
    twiss0.etapy = etap_y;

    propagateTwissParameters(twiss, &twiss0, M);

    twiss->periodic = 0;

    computeChromaticities(&(chromDeriv->tune1[0]), &(chromDeriv->tune1[1]),
                          &(chromDeriv->beta1[0]), &(chromDeriv->beta1[1]),
                          &(chromDeriv->alpha1[0]), &(chromDeriv->alpha1[1]),
                          &twiss0, twiss, M);
  }
}

void propagateTwissParameters(TWISS *twiss1, TWISS *twiss0, VMATRIX *M) {
  size_t offset;
  double C, S, Cp, Sp;
  double beta1;
  /* double alpha1; */
  double beta0, alpha0, eta0, etap0, gamma0;
  double cos_dphi, sin_dphi, dphi;
  long plane;

  for (plane = offset = 0; plane < 2; plane++) {
    beta0 = *(&(twiss0->betax) + offset);
    alpha0 = *(&(twiss0->alphax) + offset);
    eta0 = *(&(twiss0->etax) + offset);
    etap0 = *(&(twiss0->etapx) + offset);
    gamma0 = (1 + alpha0 * alpha0) / beta0;
    C = M->R[plane * 2][plane * 2];
    S = M->R[plane * 2][plane * 2 + 1];
    Cp = M->R[plane * 2 + 1][plane * 2];
    Sp = M->R[plane * 2 + 1][plane * 2 + 1];
    beta1 = *(&(twiss1->betax) + offset) = C * C * beta0 - 2 * S * C * alpha0 + S * S * gamma0;
    /* alpha1 = *(&(twiss1->alphax)+offset) = -C*Cp*beta0 + (Sp*C+S*Cp)*alpha0 - S*Sp*gamma0; */
    *(&(twiss1->alphax) + offset) = -C * Cp * beta0 + (Sp * C + S * Cp) * alpha0 - S * Sp * gamma0;
    *(&(twiss1->etax) + offset) = eta0 * C + etap0 * S + M->R[plane * 2][5];
    *(&(twiss1->etapx) + offset) = eta0 * Cp + etap0 * Sp + M->R[plane * 2 + 1][5];
    if ((sin_dphi = S / sqrt(beta0 * beta1)) > 1) {
      sin_dphi = 1;
      cos_dphi = 0;
    } else if (sin_dphi < -1) {
      sin_dphi = -1;
      cos_dphi = 0;
    } else
      cos_dphi = sqrt(beta0 / beta1) * C - alpha0 * sin_dphi;
    if ((dphi = atan2(sin_dphi, cos_dphi)) < 0)
      dphi += PIx2;
    *(&(twiss1->phix) + offset) = dphi + *(&(twiss0->phix) + offset);
    offset = TWISS_Y_OFFSET;
  }
}

long addMatrixOutputColumns(SDDS_DATASET *SDDSout, long output_order) {
  char *unit[6] = {"m", "rad", "m", "rad", "m", "1"};
  long n_numer, n_denom;
  long i, j, k, l;
  char *denom[4], *numer[4];
  char s[100], t[1024];
  char buffer[SDDS_MAXLINE];
  long nTotal = 0;

  for (i = 0; i < 6; i++) {
    sprintf(buffer, "&column name=C%ld, symbol=\"C$b%ld$n\", type=double ", i + 1, i + 1);
    if (SDDS_StringIsBlank(unit[i]))
      strcpy_ss(t, " &end");
    else
      sprintf(t, "units=%s &end", unit[i]);
    strcat(buffer, t);
    if (!SDDS_ProcessColumnString(SDDSout, buffer, 0)) {
      SDDS_SetError("Problem defining SDDS matrix output Rij columns (addMatrixOutputColumns)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  }
  nTotal += 6;

  if (output_order >= 1) {
    for (i = 0; i < 6; i++) {
      for (j = 0; j < 6; j++) {
        sprintf(buffer, "&column name=R%ld%ld, symbol=\"R$b%ld%ld$n\", type=double ", i + 1, j + 1, i + 1, j + 1);
        if (i == j)
          strcpy_ss(t, " &end");
        else
          sprintf(t, "units=%s/%s &end", unit[i], unit[j]);
        strcat(buffer, t);
        if (!SDDS_ProcessColumnString(SDDSout, buffer, 0)) {
          SDDS_SetError("Problem defining SDDS matrix output Rij columns (addMatrixOutputColumns)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
      }
    }
    nTotal += 36;
  }
  if (output_order >= 2) {
    n_numer = 1;
    n_denom = 2;
    for (i = 0; i < 6; i++) {
      for (j = 0; j < 6; j++) {
        numer[0] = unit[i];
        denom[0] = unit[j];
        for (k = 0; k <= j; k++) {
          sprintf(buffer, "&column name=T%ld%ld%ld, symbol=\"T$b%ld%ld%ld$n\", type=double ",
                  i + 1, j + 1, k + 1, i + 1, j + 1, k + 1);
          numer[0] = unit[i];
          denom[0] = unit[j];
          denom[1] = unit[k];
          simplify_units(s, numer, n_numer, denom, n_denom);
          if (SDDS_StringIsBlank(s))
            sprintf(t, " &end");
          else
            sprintf(t, "units=%s &end", s);
          strcat(buffer, t);
          if (!SDDS_ProcessColumnString(SDDSout, buffer, 0)) {
            SDDS_SetError("Problem defining SDDS matrix output Tijk columns (addMatrixOutputColumns)");
            SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
          }
          nTotal++;
        }
      }
    }
  }

  if (output_order >= 3) {
    n_numer = 1;
    n_denom = 3;
    for (i = 0; i < 6; i++) {
      for (j = 0; j < 6; j++) {
        for (k = 0; k <= j; k++) {
          for (l = 0; l <= k; l++) {
            sprintf(buffer, "&column name=U%ld%ld%ld%ld, symbol=\"U$b%ld%ld%ld%ld$n\", type=double ",
                    i + 1, j + 1, k + 1, l + 1, i + 1, j + 1, k + 1, l + 1);
            numer[0] = unit[i];
            denom[0] = unit[j];
            denom[1] = unit[k];
            denom[2] = unit[l];
            simplify_units(t, numer, n_numer, denom, n_denom);
            if (SDDS_StringIsBlank(s))
              sprintf(t, " &end");
            else
              sprintf(t, "units=%s &end", s);
            strcat(buffer, t);
            if (!SDDS_ProcessColumnString(SDDSout, buffer, 0)) {
              SDDS_SetError("Problem defining SDDS matrix output Uijk columns (addMatrixOutputColumns)");
              SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
            }
            nTotal++;
          }
        }
      }
    }
  }

  return nTotal;
}

long compareElements(ELEMENT_LIST *e1, ELEMENT_LIST *e2) {
  long i;
  double d1, d2;
  long l1, l2;
  short s1, s2;
  char *cs1, *cs2;
  if (e1->type != e2->type)
    return 1;
  for (i = 0; i < entity_description[e1->type].n_params; i++) {
    switch (entity_description[e1->type].parameter[i].type) {
    case IS_DOUBLE:
      d1 = *((double *)((e1->p_elem) + entity_description[e1->type].parameter[i].offset));
      d2 = *((double *)((e2->p_elem) + entity_description[e2->type].parameter[i].offset));
      if (d1 != d2)
        return d1 < d2 ? -1 : 1;
      break;
    case IS_LONG:
      l1 = *((long *)((e1->p_elem) + entity_description[e1->type].parameter[i].offset));
      l2 = *((long *)((e2->p_elem) + entity_description[e2->type].parameter[i].offset));
      if (l1 != l2)
        return l1 < l2 ? -1 : 1;
      break;
    case IS_SHORT:
      s1 = *((short *)((e1->p_elem) + entity_description[e1->type].parameter[i].offset));
      s2 = *((short *)((e2->p_elem) + entity_description[e2->type].parameter[i].offset));
      if (s1 != s2)
        return s1 < s2 ? -1 : 1;
      break;
    case IS_STRING:
      cs1 = *((char **)((e1->p_elem) + entity_description[e1->type].parameter[i].offset));
      cs2 = *((char **)((e2->p_elem) + entity_description[e2->type].parameter[i].offset));
      if (cs1 != cs2) {
        if (!cs1)
          return 1;
        if (!cs2)
          return -1;
        if ((l1 = strcmp(cs1, cs2)))
          return l1;
      }
      break;
    default:
      fprintf(stderr, "Error: invalid item type code %ld for %s parameter of %s\n",
              entity_description[e1->type].parameter[i].type,
              entity_description[e1->type].parameter[i].name,
              entity_name[e1->type]);
      exit(1);
      break;
    }
  }

  if (e1->type == T_CSBEND) {
    CSBEND *csb1, *csb2;
    csb1 = (CSBEND *)e1->p_elem;
    csb2 = (CSBEND *)e2->p_elem;
    if (csb1->e1Index != csb2->e1Index || csb1->e2Index != csb2->e2Index || csb1->edgeFlip!=csb2->edgeFlip || 
        csb1->edgeFlags != csb2->edgeFlags) {
      /* elements are reflections of each other */
      return -1;
    }
  }
  if (e1->type == T_CCBEND) {
    CCBEND *ccb1, *ccb2;
    ccb1 = (CCBEND *)e1->p_elem;
    ccb2 = (CCBEND *)e2->p_elem;
    if (ccb1->edgeFlip != ccb2->edgeFlip)
      /* elements are reflections of each other */
      return -1;
  }
  if (e1->type == T_LGBEND) {
    LGBEND *lgb1, *lgb2;
    lgb1 = (LGBEND *)e1->p_elem;
    lgb2 = (LGBEND *)e2->p_elem;
    if (lgb1->wasFlipped != lgb2->wasFlipped)
      /* elements are reflections of each other */
      return -1;
  }
  return 0;
}
