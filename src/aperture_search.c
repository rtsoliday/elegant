/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: aperture_search.c
 * purpose: Do tracking to find machine aperture.
 *          See file aperture_search.nl for input parameters.
 *
 * Michael Borland, 1989
 */
#include "mdb.h"
#include "track.h"
#include "aperture_search.h"
#if defined(HAVE_GPU) && !USE_MPI
#  include "gpu_search.h"
#endif

#define APERTURE_TRACKING_FLAGS (SILENT_RUNNING | INHIBIT_FILE_OUTPUT | LOSS_COORDINATES_NEEDED)

#define IC_X 0
#define IC_Y 1
#define IC_SLOST 2
#define IC_XLOST 3
#define IC_YLOST 4
#define IC_XC 5
#define IC_YC 6
#define N_COLUMNS 7
static SDDS_DEFINITION column_definition[N_COLUMNS] = {
  {"x", "&column name=x, symbol=x, units=m, type=double &end"},
  {"y", "&column name=y, symbol=y, units=m, type=double &end"},
  {"sLost", "&column name=sLost, units=m, type=double &end"},
  {"xLost", "&column name=xLost, units=m, type=double &end"},
  {"yLost", "&column name=yLost, units=m type=double &end"},
  {"xClipped", "&column name=xClipped, symbol=xClipped, units=m, type=double &end"},
  {"yClipped", "&column name=yClipped, symbol=yClipped, units=m, type=double &end"},
};

static SDDS_DEFINITION columnp_definition[N_COLUMNS] = {
  {"xp", "&column name=xp, type=double &end"},
  {"yp", "&column name=yp, type=double &end"},
  {"sLost", "&column name=sLost, units=m, type=double &end"},
  {"xLost", "&column name=xLost, units=m, type=double &end"},
  {"yLost", "&column name=yLost, units=m type=double &end"},
  {"xpClipped", "&column name=xpClipped, type=double &end"},
  {"ypClipped", "&column name=ypClipped, type=double &end"},
};

#define IC_U 0
#define IC_V 1
#define IC_SLOSTD 2
#define IC_XLOSTD 3
#define IC_YLOSTD 4
#define IC_DELTALOSTD 5
#define N_DELTA_COLUMNS 6
static SDDS_DEFINITION column_x_delta_definition[N_DELTA_COLUMNS] = {
  {"delta", "&column name=delta, symbol=$gd$r, type=double &end"},
  {"x", "&column name=x, symbol=x, units=m, type=double &end"},
  {"sLost", "&column name=sLost, units=m, type=double &end"},
  {"xLost", "&column name=xLost, units=m, type=double &end"},
  {"yLost", "&column name=yLost, units=m type=double &end"},
  {"deltaLost", "&column name=deltaLost, type=double &end"},
};
static SDDS_DEFINITION column_y_delta_definition[N_DELTA_COLUMNS] = {
  {"delta", "&column name=delta, symbol=$gd$r, type=double &end"},
  {"y", "&column name=y, symbol=y, units=m, type=double &end"},
  {"sLost", "&column name=sLost, units=m, type=double &end"},
  {"xLost", "&column name=xLost, units=m, type=double &end"},
  {"yLost", "&column name=yLost, units=m type=double &end"},
  {"deltaLost", "&column name=deltaLost, type=double &end"},
};

#define IP_STEP 0
#define IP_SVN 1
#define IP_AREA 2
#define N_PARAMETERS 3
static SDDS_DEFINITION parameter_definition[N_PARAMETERS] = {
  {"Step", "&parameter name=Step, type=long, description=\"Simulation step\" &end"},
  {"SVNVersion", "&parameter name=SVNVersion, type=string, description=\"SVN version number\", fixed_value=" SVN_VERSION " &end"},
  {"Area", "&parameter name=Area, type=double, units=\"m$a2$n\" &end"},
};
static SDDS_DEFINITION parameterp_definition[N_PARAMETERS] = {
  {"Step", "&parameter name=Step, type=long, description=\"Simulation step\" &end"},
  {"SVNVersion", "&parameter name=SVNVersion, type=string, description=\"SVN version number\", fixed_value=" SVN_VERSION " &end"},
  {"Area", "&parameter name=Area, type=double, &end"},
};

static SDDS_DATASET SDDS_aperture;
static FILE *fpSearchOutput = NULL;

#define MP_MODE 0
#define SP_MODE 1
#define GRID_MODE 2
#define ONE_LINE_MODE (GRID_MODE+1)
/* This one needs to be the first line mode */
#define TWO_LINE_MODE (GRID_MODE+2)
#define THREE_LINE_MODE (GRID_MODE+3)
#define FIVE_LINE_MODE (GRID_MODE+4)
#define SEVEN_LINE_MODE (GRID_MODE+5)
#define NINE_LINE_MODE (GRID_MODE+6)
#define ELEVEN_LINE_MODE (GRID_MODE+7)
#define N_LINE_MODE (GRID_MODE+8)
/* This one needs to be the last line mode */
#define LINE_MODE (GRID_MODE+9)
#define N_SEARCH_MODES (GRID_MODE+10)
static char *search_mode[N_SEARCH_MODES] = {
  "many-particle",
  "single-particle",
  "grid",
  "one-line",
  "two-line",
  "three-line",
  "five-line",
  "seven-line",
  "nine-line",
  "eleven-line",
  "n-line",
  "particle-line",
};
static long mode_code = 0;
static long slope_mode = 0;
#define DELTA_X_MODE 1
#define DELTA_Y_MODE 2
static long delta_mode = 0;

#if USE_MPI
long do_aperture_search_line_p(RUN *run, VARY *control, double *referenceCoord, ERRORVAL *errcon,
                               LINE_LIST *beamline, long lines, double *returnValue);
long do_aperture_search_grid_p(RUN *run, VARY *control, double *referenceCoord, ERRORVAL *errcon,
                             LINE_LIST *beamline, double *returnValue);
#endif
long do_aperture_search_grid(RUN *run, VARY *control, double *referenceCoord, ERRORVAL *errcon,
                             LINE_LIST *beamline, double *returnValue);

void setup_aperture_search(
  NAMELIST_TEXT *nltext,
  RUN *run,
  VARY *control,
  long *optimizationMode) {
  char description[200];

  log_entry("setup_aperture_search");

  /* process namelist input */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&find_aperture, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &find_aperture);

  /* check for data errors */
  if (!output && !optimization_mode)
    bombElegant("no output filename specified (required if optimization_mode=0)", NULL);
  delta_mode = 0;
  if (deltamin!=0 || deltamax!=0) {
    if (ndelta==0) 
      bombElegant("deltamin and/or deltamax non-zero but ndelta = 0", NULL);
    if (xpmin!=xpmax || ypmin!=ypmax) 
      bombElegant("deltamin and/or deltamax non-zero but slope search is requested. Not supported.", NULL);
    if (xmin!=xmax) {
      delta_mode = DELTA_X_MODE;
      if (ymin!=ymax)
        bombElegant("deltamin and/or deltamax non-zero but parameters given for x and y scans as well. Only one is allowed at a time.", NULL);
    } else if (ymin!=ymax) 
      delta_mode = DELTA_Y_MODE;
    else 
      bombElegant("deltamin and/or deltamax non-zero but not given for x or y scans as well.", NULL);
  } else {
    if (xmin==0 && xmax==0 && xpmin==xpmax)
      xmin = -(xmax = 0.1);
    if (ymin==0 && ymax==0 && ypmin==ypmax)
      ymax = 0.1;

    if (xmin == xmax) {
      slope_mode = 1;
      if (xpmin == xpmax || ypmin == ypmax)
        bombElegant("if xmin==xmax, must not have xpmin==xpmax or ypmin==ypmax", NULL);
      if (ymin != ymax)
        bombElegant("if xmin==xmax, must also have ymin==ymax", NULL);
    } else {
      if (xmin >= xmax)
        bombElegant("xmin >= xmax", NULL);
      if (ymin >= ymax)
        bombElegant("ymin >= ymax", NULL);
      if (xpmin != xpmax || ypmin != ypmax)
        bombElegant("if xmin!=xmax, must have xpmin==xpmax and ypmin==ypmax", NULL);
    }
  }

  if (nx < 3)
    bombElegant("nx < 3", NULL);
  if (ny < 2)
    bombElegant("ny < 2", NULL);
  if (n_splits && n_splits < 1)
    bombElegant("n_splits is non-zero, but less than 1", NULL);
  if (n_splits) {
    if (split_fraction <= 0 || split_fraction >= 1)
      bombElegant("split_fraction must be greater than 0 and less than 1", NULL);
    if (desired_resolution <= 0 || desired_resolution >= 1)
      bombElegant("desired_resolution must be greater than 0 and less than 1", NULL);
    if ((desired_resolution *= (xmax - xmin)) > (xmax - xmin) / (nx - 1))
      bombElegant("desired_resolution is larger than coarse mesh", NULL);
  }
  if ((mode_code = match_string(mode, search_mode, N_SEARCH_MODES, 0)) < 0)
    bombElegant("unknown search mode", NULL);
  if (slope_mode && mode_code != N_LINE_MODE)
    bombElegant("slope-based aperture search only supported for mode=\"n-line\"", NULL);
  if (delta_mode  && mode_code != GRID_MODE)
    bombElegant("delta aperture search only supported for mode=\"grid\"", NULL);
#if USE_MPI
  if (mode_code == SP_MODE ) {
    bombElegant("mode=\"single-particle\" is not supported by parallel elegant", NULL);
  }
#endif
  if (mode_code < GRID_MODE) {
    printWarning("find_aperture: using one of the line- or grid-based search methods is strongly recommended",
                 "other methods, which search from large amplitude to small amplitude, may overstate the aperture because of stable islands");
  }
  if (full_plane != 0 && mode_code < ONE_LINE_MODE)
    bombElegant("full_plane=1 is only supported for line-based modes at present", NULL);

  if (optimization_mode && (mode_code <= TWO_LINE_MODE || (mode_code == LINE_MODE && n_lines < 3)))
    bombElegant("dynamic aperture optimization requires use of n-line mode with at least 3 lines", NULL);
  if (offset_by_orbit && mode_code == SP_MODE)
    bombElegant("can't presently offset_by_orbit for that mode", NULL);

#if USE_MPI
  watch_not_allowed = 1;
  if (isMaster) {
#endif
    if (!optimization_mode) {
      output = compose_filename(output, run->rootname);
      sprintf(description, "%s aperture search", search_mode[mode_code]);
      if (mode_code!=GRID_MODE) {
        SDDS_ElegantOutputSetup(&SDDS_aperture, output, SDDS_BINARY, 1,
                                description, run->runfile, run->lattice, slope_mode ? parameterp_definition : parameter_definition,
                                N_PARAMETERS - (mode_code >= GRID_MODE && mode_code <= LINE_MODE ? 0 : 1),
                                slope_mode ? columnp_definition : column_definition,
                                N_COLUMNS - (mode_code >= GRID_MODE && mode_code <= LINE_MODE ? 0 : 3),
                                "setup_aperture_search", SDDS_EOS_NEWFILE);
      } else {
        if (delta_mode==DELTA_X_MODE) {
          SDDS_ElegantOutputSetup(&SDDS_aperture, output, SDDS_BINARY, 1,
                                  description, run->runfile, run->lattice, parameter_definition, N_PARAMETERS, 
                                  column_x_delta_definition, N_DELTA_COLUMNS, "setup_aperture_search", SDDS_EOS_NEWFILE);
        } else if (delta_mode==DELTA_Y_MODE) {
          SDDS_ElegantOutputSetup(&SDDS_aperture, output, SDDS_BINARY, 1,
                                  description, run->runfile, run->lattice, parameter_definition, N_PARAMETERS, 
                                  column_y_delta_definition, N_DELTA_COLUMNS, "setup_aperture_search", SDDS_EOS_NEWFILE);
        } else {
          SDDS_ElegantOutputSetup(&SDDS_aperture, output, SDDS_BINARY, 1,
                                  description, run->runfile, run->lattice, parameter_definition, N_PARAMETERS, column_definition,
                                  N_COLUMNS-2, "setup_aperture_search", SDDS_EOS_NEWFILE);
        }
      }
      if (control->n_elements_to_vary)
        if (!SDDS_DefineSimpleParameters(&SDDS_aperture, control->n_elements_to_vary,
                                         control->varied_quan_name, control->varied_quan_unit, SDDS_DOUBLE)) {
          SDDS_SetError("Unable to define additional SDDS parameters (setup_aperture_search)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }

      if (!SDDS_WriteLayout(&SDDS_aperture)) {
        SDDS_SetError("Unable to write SDDS layout for aperture search");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }

      if (boundary && (mode_code == SP_MODE || mode_code == MP_MODE)) {
        FILE *fp;
        boundary = compose_filename(boundary, run->rootname);
        fp = fopen_e(boundary, "w", 0);
        fputs("SDDS1\n&column name=x, units=m, type=double &end\n", fp);
        fputs("&column name=y, units=m, type=double &end\n", fp);
        fprintf(fp, "&parameter name=MplTitle, type=string, fixed_value=\"Aperture search boundary for run %s\", &end\n",
                run->runfile);
        fputs("&data mode=ascii, no_row_counts=1 &end\n", fp);
        fprintf(fp, "%e\t%e\n", xmin, ymin);
        fprintf(fp, "%e\t%e\n", xmin, ymax);
        fprintf(fp, "%e\t%e\n", xmax, ymax);
        fprintf(fp, "%e\t%e\n", xmax, ymin);
        fprintf(fp, "%e\t%e\n", xmin, ymin);
        fclose(fp);
      }

      fpSearchOutput = NULL;
      if (search_output) {
        if (mode_code != SP_MODE) {
          printf("Error: search_output field can only be used with single-particle mode\n");
          exitElegant(1);
        }
        search_output = compose_filename(search_output, run->rootname);
        fpSearchOutput = fopen_e(search_output, "w", 0);
        fputs("SDDS1\n&parameter name=Step, type=long &end\n", fpSearchOutput);
        fputs("&parameter name=x0, type=double, units=m &end\n", fpSearchOutput);
        fputs("&parameter name=y0, type=double, units=m &end\n", fpSearchOutput);
        fputs("&parameter name=SearchFromRight, type=short &end\n", fpSearchOutput);
        fputs("&parameter name=IsStable, type=short &end\n", fpSearchOutput);
        fputs("&data mode=ascii no_row_counts=1 &end\n", fpSearchOutput);
      }
    }
#if USE_MPI
  } else /* set output to NULL for the slave processors */
    output = NULL;

  if (optimization_mode) {
    distributedBeam = 0; /* All the processors will track independently */
    lessPartAllowed = 1;
  }
#endif

  *optimizationMode = optimization_mode;

  log_exit("setup_aperture_search");
}

long do_aperture_search(
  RUN *run,
  VARY *control,
  double *referenceCoord,
  ERRORVAL *errcon,
  LINE_LIST *beamline,
  double *returnValue) {
  long retcode;
  *returnValue = 0;

  log_entry("do_aperture_search");
  switch (mode_code) {
  case N_LINE_MODE:
    if (n_lines <= 2)
      bombElegant("n_lines must be greater than 2 for aperture search", NULL);
    retcode = do_aperture_search_line(run, control, referenceCoord, errcon, beamline, n_lines, returnValue);
    break;
  case MP_MODE:
    retcode = do_aperture_search_mp(run, control, referenceCoord, errcon, beamline);
    break;
  case GRID_MODE:
    retcode = do_aperture_search_grid(run, control, referenceCoord, errcon, beamline, returnValue);
    break;
  case ONE_LINE_MODE:
  case LINE_MODE:
    retcode = do_aperture_search_line(run, control, referenceCoord, errcon, beamline, 1, returnValue);
    break;
  case TWO_LINE_MODE:
    retcode = do_aperture_search_line(run, control, referenceCoord, errcon, beamline, 2, returnValue);
    break;
  case THREE_LINE_MODE:
    retcode = do_aperture_search_line(run, control, referenceCoord, errcon, beamline, 3, returnValue);
    break;
  case FIVE_LINE_MODE:
    retcode = do_aperture_search_line(run, control, referenceCoord, errcon, beamline, 5, returnValue);
    break;
  case SEVEN_LINE_MODE:
    retcode = do_aperture_search_line(run, control, referenceCoord, errcon, beamline, 7, returnValue);
    break;
  case NINE_LINE_MODE:
    retcode = do_aperture_search_line(run, control, referenceCoord, errcon, beamline, 9, returnValue);
    break;
  case ELEVEN_LINE_MODE:
    retcode = do_aperture_search_line(run, control, referenceCoord, errcon, beamline, 11, returnValue);
    break;
  case SP_MODE:
  default:
    retcode = do_aperture_search_sp(run, control, referenceCoord, errcon, beamline);
    break;
  }
  return (retcode);
}

/* many-particle search routine */

long do_aperture_search_mp(
  RUN *run,
  VARY *control,
  double *referenceCoord,
  ERRORVAL *errcon,
  LINE_LIST *beamline) {
  double **coord, **accepted;
  double y, dx, dy;
  double **xy_left, **xy_right;
  long *found;
  long n_left, n_right, n_survived, ic;
  double p_central;
  long n_trpoint, ix, iy, is, ny1;
  long effort, n_stable;
  double orbit[6] = {0, 0, 0, 0, 0, 0};

  log_entry("do_aperture_search_mp");

  log_entry("do_aperture_search_mp.1");
  coord = (double **)czarray_2d(sizeof(**coord), ny + 1, totalPropertiesPerParticle);
  coord[ny] = NULL;
  accepted = (double **)czarray_2d(sizeof(**accepted), ny + 1, totalPropertiesPerParticle);
  accepted[ny] = NULL;
  xy_left = (double **)czarray_2d(sizeof(**xy_left), ny + 1, 2);
  xy_left[ny] = NULL;
  xy_right = (double **)czarray_2d(sizeof(**xy_right), ny + 1, 2);
  xy_right[ny] = NULL;
  found = (long *)tmalloc(sizeof(*found) * ny);
  n_left = n_right = 0;

  if (offset_by_orbit)
    memcpy(orbit, referenceCoord, sizeof(*referenceCoord) * 6);

  log_exit("do_aperture_search_mp.1");

  log_entry("do_aperture_search_mp.2");
  dx = (xmax - xmin) / (nx - 1);
  dy = (ymax - ymin) / (ny - 1);
  effort = 0;
  n_stable = 0;

  for (iy = 0, y = ymin; iy < ny; iy++, y += dy) {
    xy_left[iy][1] = xy_right[iy][1] = y;
    xy_left[iy][0] = xmin - dx;
    xy_right[iy][0] = xmax + dx;
  }

  ny1 = ny;
  fill_long_array(found, ny, 0L);
  log_exit("do_aperture_search_mp.2");

  log_entry("do_aperture_search_mp.3");
  while (ny1) {
    /* search from left */
    ny1 = 0;
    for (iy = 0; iy < ny; iy++) {
      if (!found[iy]) {
        if ((xy_left[iy][0] += dx) > xmax)
          found[iy] = -1;
        else {
          coord[ny1][0] = xy_left[iy][0];
          coord[ny1][2] = xy_left[iy][1];
          coord[ny1][1] = coord[ny1][3] = coord[ny1][4] = coord[ny1][5] = 0;
          coord[ny1][6] = iy;
          ny1++;
        }
      }
    }
    if (!ny1)
      break;
    if (verbosity > 1) {
      printf("tracking %ld particles with x = %e:  \n", ny1, coord[0][0]);
      fflush(stdout);
      if (verbosity > 2) {
        for (iy = 0; iy < ny1; iy++)
          printf("    y = %e ", coord[iy][2]);
        fflush(stdout);
        fputc('\n', stdout);
      }
    }
    p_central = run->p_central;
    n_trpoint = ny1;
    for (iy = 0; iy < ny1; iy++)
      for (ic = 0; ic < 6; ic++)
        coord[iy][ic] += orbit[ic];
    n_survived = do_tracking(NULL, coord, n_trpoint, &effort, beamline, &p_central,
                             accepted, NULL, NULL, NULL, run, control->i_step,
                             APERTURE_TRACKING_FLAGS, control->n_passes, 0, NULL, NULL, NULL, NULL, NULL);
    if (verbosity > 1) {
      printf("    %ld particles survived\n", n_survived);
      fflush(stdout);
    }

    for (is = 0; is < n_survived; is++) {
      if (verbosity > 2)
        printf("survivor: x = %e, y = %e\n",
               accepted[is][0] - orbit[0], accepted[is][2] - orbit[2]);
      fflush(stdout);

      iy = accepted[is][6];
      if (iy < 0 || iy >= ny)
        bombElegant("invalid index (do_aperture_search.1)", NULL);
      found[iy] = 1;
    }
    n_stable += n_survived;
    ny1 -= n_survived;
  }
  n_left = ny;
  for (iy = 0; iy < n_left; iy++) {
    if (found[iy] != 1) {
      for (ix = iy + 1; ix < ny; ix++) {
        found[ix - 1] = found[ix];
        xy_left[ix - 1][0] = xy_left[ix][0];
        xy_left[ix - 1][1] = xy_left[ix][1];
      }
      iy--;
      n_left--;
    }
  }
  if (verbosity > 1) {
    printf("results for scan from left\n");
    fflush(stdout);
    for (iy = 0; iy < n_left; iy++)
      printf("    stable particle at x=%e, y=%e\n", xy_left[iy][0], xy_left[iy][1]);
    fflush(stdout);
  }
  log_exit("do_aperture_search_mp.3");

  log_entry("do_aperture_search_mp.4");
  ny1 = ny;
  fill_long_array(found, ny, 0L);
  while (ny1) {
    /* search from right */
    ny1 = 0;
    for (iy = 0; iy < ny; iy++) {
      if (!found[iy]) {
        if ((xy_right[iy][0] -= dx) < xmin)
          found[iy] = -1;
        else {
          coord[ny1][0] = xy_right[iy][0];
          coord[ny1][2] = xy_right[iy][1];
          coord[ny1][1] = coord[ny1][3] = coord[ny1][4] = coord[ny1][5] = 0;
          coord[ny1][6] = iy;
          ny1++;
        }
      }
    }
    if (!ny1)
      break;
    if (verbosity > 1) {
      printf("tracking %ld particles with x = %e:  \n", ny1, coord[0][0]);
      fflush(stdout);
      if (verbosity > 2) {
        for (iy = 0; iy < ny1; iy++)
          printf("    y = %e ", coord[iy][2]);
        fflush(stdout);
        fputc('\n', stdout);
      }
    }
    p_central = run->p_central;
    n_trpoint = ny1;
    for (iy = 0; iy < ny1; iy++)
      for (ic = 0; ic < 6; ic++)
        coord[iy][ic] += orbit[ic];
    n_survived = do_tracking(NULL, coord, n_trpoint, &effort, beamline, &p_central,
                             accepted, NULL, NULL, NULL, run, control->i_step,
                             APERTURE_TRACKING_FLAGS, control->n_passes, 0, NULL, NULL, NULL, NULL, NULL);
    if (verbosity > 1) {
      printf("    %ld particles survived\n", n_survived);
      fflush(stdout);
    }

    for (is = 0; is < n_survived; is++) {
      if (verbosity > 2)
        printf("survivor: x = %e, y = %e\n",
               accepted[is][0] - orbit[0], accepted[is][2] - orbit[2]);
      fflush(stdout);

      iy = accepted[is][6];
      if (iy < 0 || iy >= ny)
        bombElegant("invalid index (do_aperture_search.1)", NULL);
      found[iy] = 1;
    }
    n_stable += n_survived;
    ny1 -= n_survived;
  }
  n_right = ny;
  for (iy = 0; iy < n_right; iy++) {
    if (found[iy] != 1) {
      for (ix = iy + 1; ix < ny; ix++) {
        found[ix - 1] = found[ix];
        if (!xy_right[ix - 1]) {
          printf("error: xy_right[%ld] is NULL\n", ix - 1);
          fflush(stdout);
          abort();
        }
        if (!xy_right[ix]) {
          printf("error: xy_right[%ld] is NULL\n", ix);
          fflush(stdout);
          abort();
        }
        xy_right[ix - 1][0] = xy_right[ix][0];
        xy_right[ix - 1][1] = xy_right[ix][1];
      }
      iy--;
      n_right--;
    }
  }
  if (verbosity > 1) {
    printf("results for scan from right\n");
    fflush(stdout);
    for (iy = 0; iy < n_right; iy++)
      printf("    stable particle at x=%e, y=%e\n", xy_right[iy][0], xy_right[iy][1]);
    fflush(stdout);
  }
  log_exit("do_aperture_search_mp.4");

  if (verbosity > 0) {
    printf("total effort:  %ld particle-turns   %ld stable particles were tracked\n", effort, n_stable);
    fflush(stdout);
  }

  log_entry("do_aperture_search_mp.5");

  SDDS_ClearErrors();
  if (!SDDS_StartTable(&SDDS_aperture, n_left + n_right)) {
    SDDS_SetError("Unable to start SDDS table (do_aperture_search)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  SDDS_SetParameters(&SDDS_aperture, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, IP_STEP, control->i_step, -1);
  if (control->n_elements_to_vary) {
    for (ix = 0; ix < control->n_elements_to_vary; ix++)
      if (!SDDS_SetParameters(&SDDS_aperture, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, ix + N_PARAMETERS,
                              control->varied_quan_value[ix], -1))
        break;
  }
  if (SDDS_NumberOfErrors()) {
    SDDS_SetError("Problem setting SDDS parameter values (do_aperture_search)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  log_entry("do_aperture_search_mp.6");
  for (iy = 0; iy < n_left; iy++)
    if (!SDDS_SetRowValues(&SDDS_aperture, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, iy,
                           IC_X, xy_left[iy][0], IC_Y, xy_left[iy][1], -1)) {
      SDDS_SetError("Problem setting SDDS row values (do_aperture_search)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  for (iy = 0; iy < n_right; iy++) {
    if (!SDDS_SetRowValues(&SDDS_aperture, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, iy + n_left,
                           IC_X, xy_right[n_right - iy - 1][0], IC_Y, xy_right[n_right - iy - 1][1], -1)) {
      SDDS_SetError("Problem setting SDDS row values (do_aperture_search)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  }
  log_exit("do_aperture_search_mp.6");
  if (!SDDS_WriteTable(&SDDS_aperture)) {
    SDDS_SetError("Problem writing SDDS table (do_aperture_search)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  if (!inhibitFileSync)
    SDDS_DoFSync(&SDDS_aperture);
  log_exit("do_aperture_search_mp.5");

  log_entry("do_aperture_search_mp.8");
  free_czarray_2d((void **)coord, ny, totalPropertiesPerParticle);
  free_czarray_2d((void **)accepted, ny, totalPropertiesPerParticle);
  free_czarray_2d((void **)xy_left, ny, 2);
  free_czarray_2d((void **)xy_right, ny, 2);
  free(found);
  log_exit("do_aperture_search_mp.8");

  log_exit("do_aperture_search_mp");
  return (1);
}

long do_aperture_search_sp(
  RUN *run,
  VARY *control,
  double *referenceCoord,
  ERRORVAL *errcon,
  LINE_LIST *beamline) {
  double **coord;
  double x, y, dx, dy;
  double **xy_left, **xy_right;
  long n_left, n_right;
  double last_x_left, last_x_right, x1, x2;
  double p_central;
  long n_trpoint, ix, iy, is;
  long effort, n_stable;

  log_entry("do_aperture_search_sp");

  coord = (double **)czarray_2d(sizeof(**coord), 1, totalPropertiesPerParticle);
  xy_left = (double **)czarray_2d(sizeof(**xy_left), ny, 2);
  xy_right = (double **)czarray_2d(sizeof(**xy_right), ny, 2);
  n_left = n_right = 0;

  dx = (xmax - xmin) / (nx - 1);
  dy = (ymax - ymin) / (ny - 1);
  last_x_left = xmin;
  last_x_right = xmax;
  effort = 0;
  n_stable = 0;
  for (iy = 0, y = ymin; iy < ny; iy++, y += dy) {
#if USE_MPI
    /* A y value will be searched with one CPU */
    if (myid != iy % n_processors)
      continue;
#endif
    if (verbosity > 0)
      printf("searching for aperture for y = %e m\n", y);
    fflush(stdout);
    if (verbosity > 1) {
      printf("    searching from left to right\n");
      fflush(stdout);
    }
    if (assume_nonincreasing && iy != 0) {
      x = last_x_left;
      ix = (x - xmin) / dx + 0.5;
      if (ix > nx - 1)
        ix = nx - 1;
    } else {
      x = xmin;
      ix = 0;
    }
    for (; ix < nx; ix++, x += dx) {
      if (verbosity > 1) {
        printf("    tracking for x = %e m\n", x);
        fflush(stdout);
      }
      coord[0][0] = x;
      coord[0][2] = y;
      p_central = run->p_central;
      coord[0][1] = coord[0][3] = coord[0][4] = coord[0][5] = 0;
      n_trpoint = 1;
      if (do_tracking(NULL, coord, n_trpoint, &effort, beamline, &p_central,
                      NULL, NULL, NULL, NULL, run, control->i_step,
                      APERTURE_TRACKING_FLAGS, control->n_passes, 0, NULL, NULL, NULL, NULL, NULL)) {
        /* stable */
        if (fpSearchOutput)
          fprintf(fpSearchOutput, "%ld\n%le\n%le\n0\n1\n", control->i_step, x, y);
        break;
      } else {
        /* unstable */
        if (fpSearchOutput)
          fprintf(fpSearchOutput, "%ld\n%le\n%le\n0\n0\n", control->i_step, x, y);
      }
    }
    if (ix != nx) {
      n_stable++;
      last_x_left = x;
      if (ix != 0 && n_splits) {
        /* do secondary search */
        x1 = x;      /* stable   */
        x2 = x - dx; /* unstable */
        for (is = 0; is < n_splits; is++) {
          if (fabs(x1 - x2) < desired_resolution)
            break;
          x = (1 - split_fraction) * x1 + split_fraction * x2;
          if (verbosity > 1) {
            printf("    splitting:  %e, %e --> %e \n", x1, x2, x);
            fflush(stdout);
          }
          coord[0][0] = x;
          coord[0][2] = y;
          p_central = run->p_central;
          coord[0][1] = coord[0][3] = coord[0][4] = coord[0][5] = 0;
          n_trpoint = 1;
          if (do_tracking(NULL, coord, n_trpoint, &effort, beamline, &p_central,
                          NULL, NULL, NULL, NULL, run, control->i_step,
                          APERTURE_TRACKING_FLAGS, control->n_passes, 0, NULL, NULL, NULL, NULL, NULL)) {
            n_stable++;
            x1 = x; /* stable */
            if (fpSearchOutput)
              fprintf(fpSearchOutput, "%ld\n%le\n%le\n0\n1\n", control->i_step, x, y);
          } else {
            x2 = x; /* unstable */
            if (fpSearchOutput)
              fprintf(fpSearchOutput, "%ld\n%le\n%le\n0\n0\n", control->i_step, x, y);
          }
        }
        x = x1;
      }
      xy_left[n_left][0] = x;
      xy_left[n_left][1] = y;
      if (verbosity > 0) {
        printf("    x = %e m is stable\n", x);
        fflush(stdout);
      }
      n_left++;
    } else {
      if (verbosity > 0) {
        printf("    no stable particles seen\n");
        fflush(stdout);
      }
      continue;
    }
    if (fpSearchOutput)
      fflush(fpSearchOutput);
    /* search from right */
    if (verbosity > 1) {
      printf("    searching from right to left\n");
      fflush(stdout);
    }
    if (assume_nonincreasing && iy != 0) {
      x = last_x_right;
      ix = (xmax - x) / dx + 0.5;
      if (ix > nx - 1)
        ix = nx - 1;
    } else {
      x = xmax;
      ix = 0;
    }
    for (; ix < nx; ix++, x -= dx) {
      if (verbosity > 1) {
        printf("    tracking for x = %e m\n", x);
        fflush(stdout);
      }
      coord[0][0] = x;
      coord[0][2] = y;
      p_central = run->p_central;
      coord[0][1] = coord[0][3] = coord[0][4] = coord[0][5] = 0;
      n_trpoint = 1;
      if (do_tracking(NULL, coord, n_trpoint, &effort, beamline, &p_central,
                      NULL, NULL, NULL, NULL, run, control->i_step,
                      APERTURE_TRACKING_FLAGS, control->n_passes, 0, NULL, NULL, NULL, NULL, NULL)) {
        /* stable */
        if (fpSearchOutput)
          fprintf(fpSearchOutput, "%ld\n%le\n%le\n1\n1\n", control->i_step, x, y);
        break;
      } else {
        /* unstable */
        if (fpSearchOutput)
          fprintf(fpSearchOutput, "%ld\n%le\n%le\n1\n0\n", control->i_step, x, y);
      }
    }
    if (ix != nx) {
      n_stable++;
      last_x_right = x;
      if (ix != 0 && n_splits) {
        /* do secondary search */
        x1 = x;      /* stable   */
        x2 = x + dx; /* unstable */
        for (is = 0; is < n_splits; is++) {
          if (fabs(x1 - x2) < desired_resolution)
            break;
          x = (1 - split_fraction) * x1 + split_fraction * x2;
          if (verbosity > 1) {
            printf("    splitting:  %e, %e --> %e \n", x1, x2, x);
            fflush(stdout);
          }
          coord[0][0] = x;
          coord[0][2] = y;
          p_central = run->p_central;
          coord[0][1] = coord[0][3] = coord[0][4] = coord[0][5] = 0;
          n_trpoint = 1;
          if (do_tracking(NULL, coord, n_trpoint, &effort, beamline, &p_central,
                          NULL, NULL, NULL, NULL, run, control->i_step,
                          APERTURE_TRACKING_FLAGS, control->n_passes, 0, NULL, NULL, NULL, NULL, NULL)) {
            n_stable++;
            x1 = x; /* stable */
            if (fpSearchOutput)
              fprintf(fpSearchOutput, "%ld\n%le\n%le\n1\n1\n", control->i_step, x, y);
          } else {
            x2 = x; /* unstable */
            if (fpSearchOutput)
              fprintf(fpSearchOutput, "%ld\n%le\n%le\n1\n0\n", control->i_step, x, y);
          }
        }
        x = x1;
      }
      xy_right[n_right][0] = x;
      xy_right[n_right][1] = y;
      if (verbosity > 0) {
        printf("    x = %e m is stable\n", x);
        fflush(stdout);
      }
      n_right++;
    }
  }
  if (fpSearchOutput)
    fflush(fpSearchOutput);
#if !USE_MPI
  if (verbosity > 0) {
    printf("total effort:  %ld particle-turns   %ld stable particles were tracked\n", effort, n_stable);
    fflush(stdout);
  }
#else
  /* Gather all the simulation result to master to write into a file */
  if (USE_MPI) {
    int *n_vector = (int *)tmalloc(n_processors * sizeof(*n_vector));
    int *offset = (int *)tmalloc(n_processors * sizeof(*offset));
    long i;

    MPI_Allgather(&n_left, 1, MPI_LONG, n_vector, 1, MPI_INT, MPI_COMM_WORLD);
    offset[0] = 0;
    for (i = 0; i < n_processors - 1; i++) {
      n_vector[i] *= 2;
      offset[i + 1] = offset[i] + n_vector[i];
    }
    n_vector[n_processors - 1] *= 2;
    MPI_Allgatherv(xy_left[0], 2 * n_left, MPI_DOUBLE, xy_left[0], n_vector, offset, MPI_DOUBLE, MPI_COMM_WORLD);

    MPI_Allgather(&n_right, 1, MPI_LONG, n_vector, 1, MPI_INT, MPI_COMM_WORLD);
    offset[0] = 0;
    for (i = 0; i < n_processors - 1; i++) {
      n_vector[i] *= 2;
      offset[i + 1] = offset[i] + n_vector[i];
    }
    n_vector[n_processors - 1] *= 2;
    MPI_Allgatherv(xy_right[0], 2 * n_right, MPI_DOUBLE, xy_right[0], n_vector, offset, MPI_DOUBLE, MPI_COMM_WORLD);
  }

  if (USE_MPI) {
    long effort_total, n_stable_total;
    long tmp[4], total[4];

    tmp[0] = effort;
    tmp[1] = n_stable;
    tmp[2] = n_left;
    tmp[3] = n_right;
    MPI_Reduce(tmp, (long*)&total, 4, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (isMaster) {
      effort_total = total[0];
      n_stable_total = total[1];
      n_left = total[2];
      n_right = total[3];
      if (verbosity > 0) {
        printf("total effort:  %ld particle-turns   %ld stable particles were tracked\n", effort_total, n_stable_total);
        fflush(stdout);
      }
    }
  }
  if (isMaster) {
#endif
  SDDS_ClearErrors();
  if (!SDDS_StartTable(&SDDS_aperture, n_left + n_right)) {
    SDDS_SetError("Unable to start SDDS table (do_aperture_search)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  SDDS_SetParameters(&SDDS_aperture, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, IP_STEP, control->i_step, -1);
  if (control->n_elements_to_vary) {
    for (ix = 0; ix < control->n_elements_to_vary; ix++)
      if (!SDDS_SetParameters(&SDDS_aperture, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, ix + N_PARAMETERS,
                              control->varied_quan_value[ix], -1))
        break;
  }
  if (SDDS_NumberOfErrors()) {
    SDDS_SetError("Problem setting SDDS parameter values (do_aperture_search)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  for (iy = 0; iy < n_left; iy++)
    if (!SDDS_SetRowValues(&SDDS_aperture, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, iy,
                           IC_X, xy_left[iy][0], IC_Y, xy_left[iy][1], -1)) {
      SDDS_SetError("Problem setting SDDS row values (do_aperture_search)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  for (iy = 0; iy < n_right; iy++) {
    if (!SDDS_SetRowValues(&SDDS_aperture, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, iy + n_left,
                           IC_X, xy_right[n_right - iy - 1][0], IC_Y, xy_right[n_right - iy - 1][1], -1)) {
      SDDS_SetError("Problem setting SDDS row values (do_aperture_search)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  }
  if (!SDDS_WriteTable(&SDDS_aperture)) {
    SDDS_SetError("Problem writing SDDS table (do_aperture_search)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  if (!inhibitFileSync)
    SDDS_DoFSync(&SDDS_aperture);
#if USE_MPI
}
#endif
free_czarray_2d((void **)coord, 1, totalPropertiesPerParticle);
free_czarray_2d((void **)xy_left, ny, 2);
free_czarray_2d((void **)xy_right, ny, 2);

log_exit("do_aperture_search_sp");
return (1);
}

void finish_aperture_search(
  RUN *run,
  VARY *control,
  ERRORVAL *errcon,
  LINE_LIST *beamline) {
  if (output) {
    if (SDDS_IsActive(&SDDS_aperture) && !SDDS_Terminate(&SDDS_aperture)) {
      SDDS_SetError("Problem terminating SDDS output (finish_aperture_search)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    if (fpSearchOutput) {
      fclose(fpSearchOutput);
      fpSearchOutput = NULL;
    }
  }
}

#ifdef USE_MPI
int comp_index(const void *idx1, const void *idx2) {
  double a1 = *((long *)idx1), a2 = *((long *)idx2);

  if (a1 < a2)
    return -1;
  else if (a1 > a2)
    return 1;
  else
    return 0;
}
#endif

/* line search routine */

#if defined(HAVE_GPU) && !USE_MPI
static long do_aperture_search_line_batched(
  RUN *run, VARY *control, double *referenceCoord, LINE_LIST *beamline,
  long lines, double *returnValue) {
  double **coord;
  double *dxFactor, *dyFactor, *dx, *dy, *x0, *y0;
  double *xLimit, *yLimit, *xLost, *yLost, *sLost;
  double *trialXLost, *trialYLost, *trialSLost;
  long *trialLossPass;
  unsigned char *survived, *seen;
  double orbit[6] = {0, 0, 0, 0, 0, 0};
  double pCentral, area = 0, dtheta;
  long line, step, split, index, ip, id, nSteps, maxSteps;
  long trials, nSurvived, effort = 0, originStable = 0;

  maxSteps = (long)(1 / split_fraction - 0.5);
  if (maxSteps < nx)
    maxSteps = nx;
  if (maxSteps < 1)
    maxSteps = 1;
  if (!gpu_batched_search_tracking_enabled(lines * nx) ||
      !gpu_batched_search_beamline_supported(beamline))
    return 0;

  coord = (double **)czarray_2d(sizeof(**coord), lines * maxSteps,
                                totalPropertiesPerParticle);
  dxFactor = (double *)tmalloc(sizeof(*dxFactor) * lines);
  dyFactor = (double *)tmalloc(sizeof(*dyFactor) * lines);
  dx = (double *)tmalloc(sizeof(*dx) * lines);
  dy = (double *)tmalloc(sizeof(*dy) * lines);
  x0 = (double *)tmalloc(sizeof(*x0) * lines);
  y0 = (double *)tmalloc(sizeof(*y0) * lines);
  xLimit = (double *)calloc((size_t)lines, sizeof(*xLimit));
  yLimit = (double *)calloc((size_t)lines, sizeof(*yLimit));
  xLost = (double *)tmalloc(sizeof(*xLost) * lines);
  yLost = (double *)tmalloc(sizeof(*yLost) * lines);
  sLost = (double *)tmalloc(sizeof(*sLost) * lines);
  survived = (unsigned char *)calloc((size_t)lines * maxSteps,
                                     sizeof(*survived));
  seen = (unsigned char *)calloc((size_t)lines * maxSteps, sizeof(*seen));
  trialXLost = (double *)calloc((size_t)lines * maxSteps,
                                sizeof(*trialXLost));
  trialYLost = (double *)calloc((size_t)lines * maxSteps,
                                sizeof(*trialYLost));
  trialSLost = (double *)calloc((size_t)lines * maxSteps,
                                sizeof(*trialSLost));
  trialLossPass = (long *)calloc((size_t)lines * maxSteps,
                                 sizeof(*trialLossPass));
  if (!coord || !dxFactor || !dyFactor || !dx || !dy || !x0 || !y0 ||
      !xLimit || !yLimit || !xLost || !yLost || !sLost || !survived ||
      !seen || !trialXLost || !trialYLost || !trialSLost ||
      !trialLossPass)
    bombElegant("memory allocation failure (batched aperture search)", NULL);

  switch (lines) {
  case 1:
    dxFactor[0] = dyFactor[0] = 1;
    break;
  case 2:
    dxFactor[0] = -1;
    dyFactor[0] = 1;
    dxFactor[1] = dyFactor[1] = 1;
    break;
  case 3:
    dxFactor[0] = 0;
    dyFactor[0] = 1;
    dxFactor[1] = dyFactor[1] = 1 / sqrt(2.0);
    dxFactor[2] = 1;
    dyFactor[2] = 0;
    break;
  default:
    dtheta = full_plane == 0 ? PI / (lines - 1) : PIx2 / (lines - 1);
    for (line = 0; line < lines; line++) {
      dxFactor[line] = sin(-PI / 2 + dtheta * line);
      dyFactor[line] = cos(-PI / 2 + dtheta * line);
      if (fabs(dxFactor[line]) < 1e-6)
        dxFactor[line] = 0;
      if (fabs(dyFactor[line]) < 1e-6)
        dyFactor[line] = 0;
    }
    break;
  }
  for (line = 0; line < lines; line++)
    xLost[line] = yLost[line] = sLost[line] = DBL_MAX;
  if (offset_by_orbit)
    memcpy(orbit, referenceCoord, sizeof(orbit));

  if (verbosity >= 1) {
    printf("** Starting batched %ld-line aperture search\n", lines);
    fflush(stdout);
  }
  if (output) {
    SDDS_ClearErrors();
    if (!SDDS_StartTable(&SDDS_aperture, lines)) {
      SDDS_SetError("Unable to start SDDS table (batched aperture search)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    SDDS_SetParameters(&SDDS_aperture,
                       SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE,
                       IP_STEP, control->i_step, -1);
    if (control->n_elements_to_vary) {
      for (index = 0; index < control->n_elements_to_vary; index++)
        if (!SDDS_SetParameters(
              &SDDS_aperture, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE,
              index + N_PARAMETERS, control->varied_quan_value[index], -1))
          break;
    }
    if (SDDS_NumberOfErrors()) {
      SDDS_SetError("Problem setting SDDS parameters (batched aperture search)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  }

  for (split = 0; split <= n_splits; split++) {
    nSteps = split == 0 ? nx : (long)(1 / split_fraction - 0.5);
    if (nSteps < 1)
      nSteps = 1;
    trials = lines * nSteps;
    memset(survived, 0, (size_t)trials * sizeof(*survived));
    memset(seen, 0, (size_t)trials * sizeof(*seen));
    for (line = 0; line < lines; line++) {
      if (split == 0) {
        if (!slope_mode) {
          dx[line] = xmax / (nx - 1) * dxFactor[line];
          dy[line] = ymax / (nx - 1) * dyFactor[line];
        } else {
          dx[line] = xpmax / (nx - 1) * dxFactor[line];
          dy[line] = ypmax / (nx - 1) * dyFactor[line];
        }
        x0[line] = y0[line] = 0;
      } else {
        x0[line] = xLimit[line];
        y0[line] = yLimit[line];
        dx[line] *= split_fraction;
        dy[line] *= split_fraction;
        x0[line] += dx[line];
        y0[line] += dy[line];
      }
      for (step = 0; step < nSteps; step++) {
        id = line * nSteps + step;
        memcpy(coord[id], orbit, sizeof(orbit));
        if (!slope_mode) {
          coord[id][0] = x0[line] + step * dx[line] + orbit[0];
          coord[id][2] = y0[line] + step * dy[line] + orbit[2];
        } else {
          coord[id][1] = x0[line] + step * dx[line] + orbit[1];
          coord[id][3] = y0[line] + step * dy[line] + orbit[3];
        }
        coord[id][particleIDIndex] = id + 1;
      }
    }
    pCentral = run->p_central;
    nSurvived = do_tracking(
      NULL, coord, trials, &effort, beamline, &pCentral,
      NULL, NULL, NULL, NULL, run, control->i_step,
      APERTURE_TRACKING_FLAGS, control->n_passes, 0,
      NULL, NULL, NULL, NULL, NULL);
    for (ip = 0; ip < trials; ip++) {
      id = (long)coord[ip][particleIDIndex] - 1;
      if (id < 0 || id >= trials || seen[id])
        bombElegant("invalid or duplicate search ID in batched aperture search",
                    NULL);
      seen[id] = 1;
      survived[id] = ip < nSurvived;
      trialXLost[id] = coord[ip][0];
      trialYLost[id] = coord[ip][2];
      trialSLost[id] = coord[ip][4];
      trialLossPass[id] = (long)coord[ip][lossPassIndex];
    }
    for (id = 0; id < trials; id++)
      if (!seen[id])
        bombElegant("missing search ID in batched aperture search", NULL);

    if (split == 0 && !survived[0]) {
      *returnValue = 0;
      goto batchedApertureCleanup;
    }
    originStable = 1;
    for (line = 0; line < lines; line++) {
      long firstLost = nSteps;
      long lastSurvived;
      for (step = 0; step < nSteps; step++) {
        id = line * nSteps + step;
        if (!survived[id]) {
          firstLost = step;
          break;
        }
      }
      lastSurvived = firstLost - 1;
      if (lastSurvived >= 0) {
        double candidateX = x0[line] + lastSurvived * dx[line];
        double candidateY = y0[line] + lastSurvived * dy[line];
        if (split == 0 ||
            (dx[line] > 0 && xLimit[line] < candidateX) ||
            (dx[line] == 0 && yLimit[line] < candidateY) ||
            (dx[line] < 0 && xLimit[line] > candidateX)) {
          xLimit[line] = candidateX;
          yLimit[line] = candidateY;
        }
      }
      /* The legacy line search reports the first loss from the latest split,
       * even when that split does not move the surviving boundary.  Keep the
       * same metadata semantics while looking the particle up by its stable
       * batched-search ID after loss compaction. */
      if (firstLost < nSteps) {
        id = line * nSteps + firstLost;
        xLost[line] = trialXLost[id];
        yLost[line] = trialYLost[id];
        sLost[line] = trialSLost[id];
        /* Retained by stable ID even though the legacy SDDS schema has no
         * loss-pass column for dynamic aperture output. */
        (void)trialLossPass[id];
      }
    }
  }

  if (output) {
    /* Preserve the legacy schema semantics: x/y are the raw search limits,
     * while xClipped/yClipped are populated only after island clipping. */
    for (line = 0; line < lines; line++)
      if (!SDDS_SetRowValues(
            &SDDS_aperture, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, line,
            IC_X, xLimit[line], IC_Y, yLimit[line],
            IC_XLOST, xLost[line], IC_YLOST, yLost[line],
            IC_SLOST, sLost[line], -1)) {
        SDDS_SetError("Problem setting rows (batched aperture search)");
        SDDS_PrintErrors(stderr,
                         SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
  }
  if (originStable && lines > 1)
    area = trimApertureSearchResult(lines, xLimit, yLimit,
                                    dxFactor, dyFactor, full_plane);
  *returnValue = area;
  if (output) {
    if (!SDDS_SetColumn(&SDDS_aperture, SDDS_SET_BY_INDEX,
                        xLimit, lines, IC_XC) ||
        !SDDS_SetColumn(&SDDS_aperture, SDDS_SET_BY_INDEX,
                        yLimit, lines, IC_YC) ||
        !SDDS_SetParameters(&SDDS_aperture,
                            SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                            "Area", area, NULL) ||
        !SDDS_WriteTable(&SDDS_aperture)) {
      SDDS_SetError("Problem writing batched aperture search output");
      SDDS_PrintErrors(stderr,
                       SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    if (!inhibitFileSync)
      SDDS_DoFSync(&SDDS_aperture);
  }

batchedApertureCleanup:
  free_czarray_2d((void **)coord, lines * maxSteps,
                  totalPropertiesPerParticle);
  free(dxFactor);
  free(dyFactor);
  free(dx);
  free(dy);
  free(x0);
  free(y0);
  free(xLimit);
  free(yLimit);
  free(xLost);
  free(yLost);
  free(sLost);
  free(survived);
  free(seen);
  free(trialXLost);
  free(trialYLost);
  free(trialSLost);
  free(trialLossPass);
  return 1;
}
#endif

long do_aperture_search_line(
  RUN *run,
  VARY *control,
  double *referenceCoord,
  ERRORVAL *errcon,
  LINE_LIST *beamline,
  long lines,
  double *returnValue) {
  double **coord;
  double x0, y0, dx, dy;
  double p_central;
  long index, split, nSteps;
  long effort, n_trpoint, line;
  double xSurvived, ySurvived, area, dtheta;
  double orbit[6] = {0, 0, 0, 0, 0, 0};
  double *dxFactor, *dyFactor;
  double *xLimit, *yLimit;
  double xLost, yLost, sLost;
  long originStable = 0;

#if USE_MPI
  return do_aperture_search_line_p(run, control, referenceCoord, errcon, beamline, lines, returnValue);
#endif
#if defined(HAVE_GPU) && !USE_MPI
  if (do_aperture_search_line_batched(run, control, referenceCoord,
                                      beamline, lines, returnValue))
    return 1;
#endif

  coord = (double **)czarray_2d(sizeof(**coord), 1, totalPropertiesPerParticle);

  dxFactor = tmalloc(sizeof(*dxFactor) * lines);
  dyFactor = tmalloc(sizeof(*dyFactor) * lines);
  xLimit = tmalloc(sizeof(*xLimit) * lines);
  yLimit = tmalloc(sizeof(*yLimit) * lines);

  switch (lines) {
  case 1:
    dxFactor[0] = 1;
    dyFactor[0] = 1;
    break;
  case 2:
    dxFactor[0] = -1;
    dyFactor[0] = 1;
    dxFactor[1] = 1;
    dyFactor[1] = 1;
    break;
  case 3:
    dxFactor[0] = 0;
    dyFactor[0] = 1;
    dxFactor[1] = dyFactor[1] = 1 / sqrt(2);
    dxFactor[2] = 1;
    dyFactor[2] = 0;
    break;
  default:
    if (full_plane == 0)
      dtheta = PI / (lines - 1);
    else
      dtheta = PIx2 / (lines - 1);
    for (line = 0; line < lines; line++) {
      dxFactor[line] = sin(-PI / 2 + dtheta * line);
      dyFactor[line] = cos(-PI / 2 + dtheta * line);
      if (fabs(dxFactor[line]) < 1e-6)
        dxFactor[line] = 0;
      if (fabs(dyFactor[line]) < 1e-6)
        dyFactor[line] = 0;
    }
    break;
  }

  effort = 0;
  xSurvived = ySurvived = -1;
  dx = dy = 0;
  if (offset_by_orbit) {
    /* N.B.: for an off-momentum orbit that is created with an initial
     * MALIGN element, the momentum offset will not appear in the
     * referenceCoord array.  So this works if the user sets ON_PASS=0
     * for the MALIGN.
     */
    memcpy(orbit, referenceCoord, sizeof(*referenceCoord) * 6);
    /*
      fprintf(stderr, "offseting by orbit: %e, %e, %e, %e, %e, %e \n",
      orbit[0], orbit[1], orbit[2],
	    orbit[3], orbit[4], orbit[5]);
    */
  }

  if (verbosity >= 1) {
    printf("** Starting %ld-line aperture search\n", lines);
    fflush(stdout);
  }

  if (output) {
    SDDS_ClearErrors();
    if (!SDDS_StartTable(&SDDS_aperture, lines)) {
      SDDS_SetError("Unable to start SDDS table (do_aperture_search)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    SDDS_SetParameters(&SDDS_aperture, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, IP_STEP, control->i_step, -1);
    if (control->n_elements_to_vary) {
      for (index = 0; index < control->n_elements_to_vary; index++)
        if (!SDDS_SetParameters(&SDDS_aperture, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, index + N_PARAMETERS,
                                control->varied_quan_value[index], -1))
          break;
    }
    if (SDDS_NumberOfErrors()) {
      SDDS_SetError("Problem setting SDDS parameter values (do_aperture_search)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  }

  for (line = 0; line < lines; line++) {
    if (dxFactor[line] > 0)
      xSurvived = ySurvived = -1;
    else
      xSurvived = -(ySurvived = -1);
    printf("* Searching line %ld\n", line);
    fflush(stdout);
    xLost = yLost = sLost = DBL_MAX;
    for (split = 0; split <= n_splits; split++) {
      if (split == 0) {
        if (!slope_mode) {
          dx = xmax / (nx - 1) * dxFactor[line];
          dy = ymax / (nx - 1) * dyFactor[line];
        } else {
          dx = xpmax / (nx - 1) * dxFactor[line];
          dy = ypmax / (nx - 1) * dyFactor[line];
        }
        x0 = y0 = 0;
        nSteps = nx;
      } else {
        x0 = xSurvived;
        y0 = ySurvived;
        dx *= split_fraction;
        dy *= split_fraction;
        x0 += dx;
        y0 += dy;
        nSteps = 1 / split_fraction - 0.5;
        if (nSteps < 1)
          nSteps = 1;
        if (verbosity >= 1) {
          printf("divided search interval to %e, %e\n", dx, dy);
          fflush(stdout);
        }
      }
      for (index = 0; index < nSteps; index++) {
        if (index != 0 || split != 0 || !originStable) {
          memcpy(coord[0], orbit, sizeof(*orbit) * 6);
          if (!slope_mode) {
            coord[0][0] = index * dx + x0 + orbit[0];
            coord[0][2] = index * dy + y0 + orbit[2];
          } else {
            coord[0][1] = index * dx + x0 + orbit[1];
            coord[0][3] = index * dy + y0 + orbit[3];
          }
          
          p_central = run->p_central;
          n_trpoint = 1;
          if (do_tracking(NULL, coord, n_trpoint, &effort, beamline, &p_central,
                          NULL, NULL, NULL, NULL, run, control->i_step,
                          APERTURE_TRACKING_FLAGS, control->n_passes, 0, NULL, NULL, NULL, NULL, NULL) != 1) {
            if (verbosity >= 2) {
              if (!slope_mode)
                printf("particle lost for x=%e, y=%e\n", index * dx + x0, index * dy + y0);
              else
                printf("particle lost for xp=%e, yp=%e\n", index * dx + x0, index * dy + y0);
              fflush(stdout);
            }
            xLost = coord[0][0];
            yLost = coord[0][2];
            sLost = coord[0][4];
            break;
          }
        }
        if (index == 0 && split == 0)
          originStable = 1;
        if (verbosity >= 2) {
          if (!slope_mode)
            printf("particle survived for x=%e, y=%e\n", x0 + index * dx, y0 + index * dy);
          else
            printf("particle survived for xp=%e, yp=%e\n", x0 + index * dx, y0 + index * dy);
          fflush(stdout);
        }
        if (dxFactor[line]) {
          if ((dxFactor[line] > 0 && (xSurvived < (x0 + index * dx))) ||
              (dxFactor[line] < 0 && (xSurvived > (x0 + index * dx)))) {
            xSurvived = x0 + index * dx;
            ySurvived = y0 + index * dy;
          }
        } else {
          if (ySurvived < (y0 + index * dy)) {
            xSurvived = x0 + index * dx;
            ySurvived = y0 + index * dy;
          }
        }
      }

      if (verbosity >= 1) {
        printf("Sweep done, particle survived up to x=%e, y=%e\n", xSurvived, ySurvived);
        fflush(stdout);
      }
    }
    xLimit[line] = xSurvived;
    yLimit[line] = ySurvived;
    if (output) {
      if (!SDDS_SetRowValues(&SDDS_aperture, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, line,
                             IC_X, xSurvived, IC_Y, ySurvived,
                             IC_XLOST, xLost, IC_YLOST, yLost, IC_SLOST, sLost, -1)) {
        SDDS_SetError("Problem setting SDDS row values (do_aperture_search)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
    }
  }
  
  area = 0;
  if (originStable && lines > 1) {
    /* compute the area */

    area = trimApertureSearchResult(lines, xLimit, yLimit, dxFactor, dyFactor, full_plane);
  }
  *returnValue = area;

  if (output) {
    if (!SDDS_SetColumn(&SDDS_aperture, SDDS_SET_BY_INDEX, xLimit, lines, IC_XC) ||
        !SDDS_SetColumn(&SDDS_aperture, SDDS_SET_BY_INDEX, yLimit, lines, IC_YC) ||
        !SDDS_SetParameters(&SDDS_aperture, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                            "Area", area, NULL)) {
      SDDS_SetError("Problem setting parameters values in SDDS table (do_aperture_search)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    if (control->n_elements_to_vary) {
      long i;
      for (i = 0; i < control->n_elements_to_vary; i++)
        if (!SDDS_SetParameters(&SDDS_aperture, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, i + N_PARAMETERS,
                                control->varied_quan_value[i], -1))
          break;
    }
    
    if (!SDDS_WriteTable(&SDDS_aperture)) {
      SDDS_SetError("Problem writing SDDS table (do_aperture_search)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    if (!inhibitFileSync)
      SDDS_DoFSync(&SDDS_aperture);
  }

  free_czarray_2d((void **)coord, 1, totalPropertiesPerParticle);
  free(dxFactor);
  free(dyFactor);
  free(xLimit);
  free(yLimit);
  
  return (1);
}

/* multi-particle search including inside particles */

long do_aperture_search_grid
(
 RUN *run,
 VARY *control,
 double *referenceCoord,
 ERRORVAL *errcon,
 LINE_LIST *beamline,
 double *returnValue
 ) {
  double **coord = NULL;
  double p_central;
  long ip, np=0, nLeft;
  long effort;
  double area;
  double orbit[6] = {0, 0, 0, 0, 0, 0};
  double *uLimit, *vLimit, *vLimit1, *xLost, *yLost, *sLost, *deltaLost;
  long side, sides;
  long icu, icv; /* indices of u and v coordinates. E.g., icu=0 and icv=2 for an (x, y) search */
  long iu, iv; /* counters for u and v coordinates */
  long nu, nv; /* number of steps for u and v coordinates */
  double du, dv; /* step sizes for u and v */
  double umin, umax, vmin, vmax;

  uLimit = vLimit = vLimit1 = xLost = yLost = sLost = deltaLost = NULL;
  
#if USE_MPI
  return do_aperture_search_grid_p(run, control, referenceCoord, errcon, beamline, returnValue);
#endif

  if (delta_mode==DELTA_X_MODE) {
    icu = 5;
    icv = 0;
    nu = ndelta;
    umin = deltamin;
    umax = deltamax;
    nv = nx;
    vmin = xmin;
    vmax = xmax;
    if (verbosity >= 1) {
      printf("** Starting multi-particle aperture search with ndelta=%ld, nx=%ld over [%le, %le] x [%le, %le]\n", nu, nv,
             umin, umax, vmin, vmax);
      fflush(stdout);
    }
  } else if (delta_mode==DELTA_Y_MODE) {
    icu = 5;
    icv = 2;
    nu = ndelta;
    umin = deltamin;
    umax = deltamax;
    nv = ny;
    vmin = ymin;
    vmax = ymax;
    if (verbosity >= 1) {
      printf("** Starting multi-particle aperture search with ny=%ld, ndelta=%ld\n", ny, ndelta);
      fflush(stdout);
    }
  } else {
    icu = 0;
    icv = 2;
    nu = nx;
    nv = ny;
    umin = xmin;
    umax = xmax;
    vmin = ymin;
    vmax = ymax;
    if (verbosity >= 1) {
      printf("** Starting multi-particle aperture search with nx=%ld, ny=%ld\n", nx, ny);
      fflush(stdout);
    }
  }

  coord = (double **)czarray_2d(sizeof(**coord), nu*nv, totalPropertiesPerParticle);

  if (offset_by_orbit) {
    /* N.B.: for an off-momentum orbit that is created with an initial
     * MALIGN element, the momentum offset will not appear in the
     * referenceCoord array.  So this works if the user sets ON_PASS=0
     * for the MALIGN.
     */
    memcpy(orbit, referenceCoord, sizeof(*referenceCoord) * 6);
  }

  if (output) {
    SDDS_ClearErrors();
    if (!SDDS_StartTable(&SDDS_aperture, 2*nu*nv)) {
      SDDS_SetError("Unable to start SDDS table (do_aperture_search)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    SDDS_SetParameters(&SDDS_aperture, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, IP_STEP, control->i_step, -1);
    if (control->n_elements_to_vary) {
      long index;
      for (index = 0; index < control->n_elements_to_vary; index++)
        if (!SDDS_SetParameters(&SDDS_aperture, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, index + N_PARAMETERS,
                                control->varied_quan_value[index], -1))
          break;
    }
    if (SDDS_NumberOfErrors()) {
      SDDS_SetError("Problem setting SDDS parameter values (do_aperture_search)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    if (verbosity >= 1) {
      printf("Opened output file %s\n", output);
      fflush(stdout);
    }
  }

  if (nu<1)
    nu = 2;
  if (nv<1)
    nv = 2;
  du = (umax - umin)/(nu-1);

  sides = vmin<0 ? 2 : 1;
  area = 0;
  for (side=0; side<sides; side++) {
    ip = 0;
    if (side==0)
      dv = vmax/(nv-1);
    else
      /* dv is negative in this case */
      dv = vmin/(nv-1);
    if (verbosity >= 2) {
      printf("du = %le, dv = %le\n", du, dv);
      fflush(stdout);
    }
    for (iu=0; iu<nu; iu++) {
      for (iv=0; iv<nv; iv++) {
        memset(coord[ip], 0, totalPropertiesPerParticle*sizeof(double));
        coord[ip][icu] = iu*du + umin;
        coord[ip][icv] = iv*dv;
        coord[ip][particleIDIndex] = ip+1;
        ip++;
      }
    }
    np = nu*nv;
    
    if (verbosity >= 1) {
      printf("Set up initial coordinates\n");
      fflush(stdout);
    }
    
    if (offset_by_orbit) {
      long i;
      for (ip=0; ip<np; ip++)
        for (i=0; i<6; i++)
          coord[ip][i] += orbit[i];
      if (verbosity >= 1) {
        printf("Offset by orbit\n");
        fflush(stdout);
      }
    }
    
    if (verbosity >= 1) {
      printf("Tracking %ld particles\n", np);
      fflush(stdout);
    }
    p_central = run->p_central;
    effort = 0;
    nLeft = do_tracking(NULL, coord, np, &effort, beamline, &p_central,
                        NULL, NULL, NULL, NULL, run, control->i_step,
                        APERTURE_TRACKING_FLAGS, control->n_passes, 0, NULL, NULL, NULL, NULL, NULL);
    if (verbosity >= 1) {
      long pid;
      printf("%ld particles left\n", nLeft);
      fflush(stdout);
      for (ip=0; ip<nLeft; ip++) {
        pid = coord[ip][particleIDIndex];
        iu = (pid-1)/nv;
        iv = (pid-1)%nv;
      }
    }
    
    vLimit = tmalloc(sizeof(*vLimit)*nu);
    vLimit1 = tmalloc(sizeof(*vLimit1)*nu);
    uLimit = tmalloc(sizeof(*uLimit)*nu);
    xLost = tmalloc(sizeof(*xLost)*nu);
    yLost = tmalloc(sizeof(*yLost)*nu);
    sLost = tmalloc(sizeof(*sLost)*nu);
    deltaLost = tmalloc(sizeof(*deltaLost)*nu);
    for (iu=0; iu<nu; iu++) {
      vLimit[iu] = 0;
      uLimit[iu] = iu*du + umin;
      xLost[iu] = yLost[iu] = sLost[iu] = deltaLost[iu] = DBL_MAX;
      if (side==0) {
          vLimit1[iu] = vmax + dv;
      } else {
          vLimit1[iu] = vmin + dv;
      }
    }
    for (ip=0; ip<nLeft; ip++) {
      /* surviving particles */
      long pid;
      double v;
      pid = coord[ip][particleIDIndex];
      iu = (pid-1)/nv;
      iv = (pid-1)%nv;
      v = iv*dv;
      if ((side==0 && v>vLimit[iu]) || (side==1 && v<vLimit[iu]))
        vLimit[iu] = v;
    }
    for (ip=nLeft; ip<np; ip++) {
      /* lost particles */
      long pid;
      double v;
      pid = coord[ip][particleIDIndex];
      iu = (pid-1)/nv;
      iv = (pid-1)%nv;
      v = iv*dv;
      if ((side==0 && v<vLimit[iu]) || (side==1 && v>vLimit[iu]))
        vLimit[iu] = v;
      if ((side==0 && v<vLimit1[iu]) || (side==1 && v>vLimit1[iu])) {
        vLimit1[iu] = v;
        xLost[iu] = coord[ip][0];
        yLost[iu] = coord[ip][2];
        sLost[iu] = coord[ip][4];
        deltaLost[iu] = coord[ip][5];
      }
    }
    if (verbosity >= 1) {
      printf("Found limits\n");
      fflush(stdout);
    }
    if (nLeft) {
      for (iu=0; iu<nu; iu++)
        area += fabs(vLimit[iu]*du);
    }
    if (output) {
      if (delta_mode) {
        for (iu=0; iu<nu; iu++) {
          if (!SDDS_SetRowValues(&SDDS_aperture, SDDS_SET_BY_INDEX|SDDS_PASS_BY_VALUE, iu+side*nu,
                                 IC_U, uLimit[iu], IC_V, vLimit[iu],
                                 IC_XLOSTD, xLost[iu], IC_YLOSTD, yLost[iu], IC_SLOSTD, sLost[iu], 
                                 IC_DELTALOSTD, deltaLost[iu], -1)) {
            SDDS_SetError("Problem setting row values in SDDS table (do_aperture_search)");
            SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
          }
        }
      } else {
        for (iu=0; iu<nu; iu++) {
          if (!SDDS_SetRowValues(&SDDS_aperture, SDDS_SET_BY_INDEX|SDDS_PASS_BY_VALUE, iu+side*nu,
                                 IC_X, uLimit[iu], IC_Y, vLimit[iu],
                                 IC_XLOST, xLost[iu], IC_YLOST, yLost[iu], IC_SLOST, sLost[iu], -1)) {
            SDDS_SetError("Problem setting row values in SDDS table (do_aperture_search)");
            SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
          }
        }
      }
    }
  }
  if (output) {
    if (!SDDS_SetParameters(&SDDS_aperture, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                              "Area", area, NULL)) {
        SDDS_SetError("Problem setting parameters values in SDDS table (do_aperture_search)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    if (control->n_elements_to_vary) {
      long i;
      for (i = 0; i < control->n_elements_to_vary; i++)
        if (!SDDS_SetParameters(&SDDS_aperture, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, i + N_PARAMETERS,
                                control->varied_quan_value[i], -1))
          break;
    }
    if (!SDDS_WriteTable(&SDDS_aperture)) {
      SDDS_SetError("Problem writing SDDS table (do_aperture_search)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    if (!inhibitFileSync)
      SDDS_DoFSync(&SDDS_aperture);
  }
  
  if (verbosity >= 1) {
    printf("Area = %le\n", area);
    fflush(stdout);
  }
  
  *returnValue = area;

  if (uLimit) free(uLimit);
  if (vLimit) free(vLimit);
  if (vLimit1) free(vLimit1);
  if (xLost) free(xLost);
  if (yLost) free(yLost);
  if (sLost) free(sLost);
  if (deltaLost) free(deltaLost);
  if (coord)
    free_czarray_2d((void **)coord, np, totalPropertiesPerParticle);

  return (1);
}

#if USE_MPI
long do_aperture_search_grid_p
(
 RUN *run,
 VARY *control,
 double *referenceCoord,
 ERRORVAL *errcon,
 LINE_LIST *beamline,
 double *returnValue
 ) {
  double **coord;
  double p_central;
  long ip, np = 0, nLeft, index, npLocal;
  double area;
  double orbit[6] = {0, 0, 0, 0, 0, 0};
  double *uLimit, *vLimit, *vLimit1, *xLost, *yLost, *sLost, *deltaLost;
  long side, sides;
  long icu, icv; /* indices of u and v coordinates. E.g., icu=0 and icv=2 for an (x, y) search */
  long iu, iv; /* counters for u and v coordinates */
  long nu, nv; /* number of steps for u and v coordinates */
  double du, dv; /* step sizes for u and v */
  double umin, umax, vmin, vmax;

  if (verbosity >= 1 && myid==0) {
    printf("** Starting multi-particle aperture search with nx=%ld, ny=%ld\n", nx, ny);
    fflush(stdout);
  }

  if (delta_mode==DELTA_X_MODE) {
    icu = 5;
    icv = 0;
    nu = ndelta;
    umin = deltamin;
    umax = deltamax;
    nv = nx;
    vmin = xmin;
    vmax = xmax;
    if (verbosity >= 1) {
      printf("** Starting multi-particle aperture search with ndelta=%ld, nx=%ld over [%le, %le] x [%le, %le]\n", nu, nv,
             umin, umax, vmin, vmax);
      fflush(stdout);
    }
  } else if (delta_mode==DELTA_Y_MODE) {
    icu = 5;
    icv = 2;
    nu = ndelta;
    umin = deltamin;
    umax = deltamax;
    nv = ny;
    vmin = ymin;
    vmax = ymax;
    if (verbosity >= 1) {
      printf("** Starting multi-particle aperture search with ny=%ld, ndelta=%ld\n", ny, ndelta);
      fflush(stdout);
    }
  } else {
    icu = 0;
    icv = 2;
    nu = nx;
    nv = ny;
    umin = xmin;
    umax = xmax;
    vmin = ymin;
    vmax = ymax;
    if (verbosity >= 1) {
      printf("** Starting multi-particle aperture search with nx=%ld, ny=%ld\n", nx, ny);
      fflush(stdout);
    }
  }

  if (offset_by_orbit) {
    /* N.B.: for an off-momentum orbit that is created with an initial
     * MALIGN element, the momentum offset will not appear in the
     * referenceCoord array.  So this works if the user sets ON_PASS=0
     * for the MALIGN.
     */
    memcpy(orbit, referenceCoord, sizeof(*referenceCoord) * 6);
  }

  if (output && myid==0) {
    SDDS_ClearErrors();
    if (!SDDS_StartTable(&SDDS_aperture, 2*nu*nv)) {
      SDDS_SetError("Unable to start SDDS table (do_aperture_search)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    SDDS_SetParameters(&SDDS_aperture, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, IP_STEP, control->i_step, -1);
    if (control->n_elements_to_vary) {
      long index;
      for (index = 0; index < control->n_elements_to_vary; index++)
        if (!SDDS_SetParameters(&SDDS_aperture, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, index + N_PARAMETERS,
                                control->varied_quan_value[index], -1))
          break;
    }
    if (SDDS_NumberOfErrors()) {
      SDDS_SetError("Problem setting SDDS parameter values (do_aperture_search)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    if (verbosity >= 1) {
      printf("Opened output file %s\n", output);
      fflush(stdout);
    }
  }

  if (nu<1)
    nu = 2;
  if (nv<1)
    nv = 2;
  du = (umax - umin)/(nu-1);

  np = nu * nv;
  coord = (double **)czarray_2d(sizeof(**coord), np, totalPropertiesPerParticle);

  sides = vmin<0 ? 2 : 1;
  area = 0;
  for (side=0; side<sides; side++) {
    np = nu*nv;
    if (side==0)
      dv = vmax/(nv-1);
    else
      /* dv is negative in this case */
      dv = vmin/(nv-1);

    if (verbosity >= 1 && myid==0) {
      printf("du = %le, dv = %le\n", du, dv);
      fflush(stdout);
    }

    ip = 0;
    index = 0;
    for (iu=0; iu<nu; iu++) {
      for (iv=0; iv<nv; iv++) {
        if (myid==ip%n_processors) {
          memset(coord[index], 0, totalPropertiesPerParticle*sizeof(double));
          coord[index][icu] = iu*du + umin;
          coord[index][icv] = iv*dv;
          coord[index][particleIDIndex] = ip+1;
          index++;
        }
        ip++;
      }
    }
    npLocal = index;
#if MPI_DEBUG
    printf("%ld particles will be tracked\n", npLocal);
    for (ip=0; ip<npLocal; ip++)
      printf("%ld: %le, %le, %ld\n", ip, coord[ip][icu], coord[ip][icv], (long)coord[ip][particleIDIndex]);
    fflush(stdout);
#endif
    
    if (verbosity >= 1 && myid==0) {
      printf("Set up initial coordinates\n");
      fflush(stdout);
    }
    
    if (offset_by_orbit) {
      long i;
      for (ip=0; ip<npLocal; ip++)
        for (i=0; i<6; i++)
          coord[ip][i] += orbit[i];
      if (verbosity >= 1 && myid==0) {
        printf("Offset by orbit\n");
        fflush(stdout);
      }
    }
    
    if (verbosity >= 1 && myid==0) {
      printf("Tracking %ld particles\n", np);
      fflush(stdout);
    }
    p_central = run->p_central;
    
    long nLost=0, nLeftLocal=0;
    
    nLeftLocal = do_tracking(NULL, coord, npLocal, NULL, beamline, &p_central,
                             NULL, NULL, NULL, NULL, run, control->i_step,
                             APERTURE_TRACKING_FLAGS, control->n_passes, 0, NULL, NULL, NULL, NULL, NULL);
    
    long *npVector = (long*)malloc(sizeof(long)*n_processors);
    long *nLeftVector = (long*)malloc(sizeof(long)*n_processors);
    MPI_Status mpiStatus;
    
#if MPI_DEBUG
    FILE *fpSurvived, *fpLost;
    char buffer[100];
    snprintf(buffer, 100, "survived-%d.sdds", myid);
    fpSurvived = fopen(buffer, "w");
    fprintf(fpSurvived, "SDDS1\n&column name=PID type=long &end\n");
    fprintf(fpSurvived, "&column name=u0 type=double &end\n");
    fprintf(fpSurvived, "&column name=v0 type=double &end\n");
    fprintf(fpSurvived, "&data mode=ascii no_row_counts=1 &end\n");
    for (ip=0; ip<nLeftLocal; ip++) {
      long pid;
      pid = coord[ip][particleIDIndex];
      iu = (pid-1)/nv;
      iv = (pid-1)%nv;
      fprintf(fpSurvived, "%ld %le %le\n", pid, iu*du + umin, iv*dv);
    }
    fclose(fpSurvived);

    snprintf(buffer, 100, "lost-%d.sdds", myid);
    fpLost = fopen(buffer, "w");
    fprintf(fpLost, "SDDS1\n&column name=PID type=long &end\n");
    fprintf(fpLost, "&column name=u0 type=double &end\n");
    fprintf(fpLost, "&column name=v0 type=double &end\n");
    fprintf(fpLost, "&column name=xLost type=double &end\n");
    fprintf(fpLost, "&column name=yLost type=double &end\n");
    fprintf(fpLost, "&column name=sLost type=double &end\n");
    fprintf(fpLost, "&data mode=ascii no_row_counts=1 &end\n");
    for (ip=nLeftLocal; ip<npLocal; ip++) {
      long pid;
      pid = coord[ip][particleIDIndex];
      iu = (pid-1)/nv;
      iv = (pid-1)%nv;
      fprintf(fpSurvived, "%ld %le %le %le %le %le\n", pid, iu*du + umin, iv*dv,
              coord[ip][0], coord[ip][2], coord[ip][4]);
    }
    fclose(fpLost);
#endif
    
#if MPI_DEBUG
    printf("%ld particles left\n", nLeftLocal);
    for (ip=0; ip<nLeftLocal; ip++)
      printf("%ld: %le, %le, %ld\n", ip, coord[ip][0], coord[ip][2], (long)coord[ip][particleIDIndex]);
    fflush(stdout);
#endif
    
    MPI_Gather(&npLocal, 1, MPI_LONG, npVector, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Gather(&nLeftLocal, 1, MPI_LONG, nLeftVector, 1, MPI_LONG, 0, MPI_COMM_WORLD);
#if MPI_DEBUG
    if (myid==0) {
      for (ip=0; ip<n_processors; ip++)
        printf("processor %ld reports %ld particles left, %ld particles total\n",
               ip, nLeftVector[ip], npVector[ip]);
      fflush(stdout);
    }
#endif
    double **lostParticles = NULL;
    long offset1, offset2;
    nLeft = np = offset1 = offset2 = 0;
    if (myid==0) {
      int islave;
      for (islave=0; islave<n_processors; islave++) {
        nLeft += nLeftVector[islave];
        np += npVector[islave];
      }
      if (verbosity >= 1 && myid==0) {
        printf("%ld particles left\n", nLeft);
        fflush(stdout);
      }
      lostParticles = (double **)czarray_2d(sizeof(**coord), np-nLeft, totalPropertiesPerParticle);
#if MPI_DEBUG
      printf("totals: %ld particles left, %ld particles tracked\n", nLeft, np);
#endif
      for (islave=0; islave<n_processors; islave++) {
        if (islave==0) {
          /* master */
          /* copy lost particle data */
          for (ip=nLeftLocal; ip<npLocal; ip++)
            memcpy(lostParticles[ip-nLeftLocal], coord[ip], totalPropertiesPerParticle*sizeof(double));
        } else {
          MPI_Recv(&coord[offset1][0], nLeftVector[islave]*totalPropertiesPerParticle, MPI_DOUBLE, islave, 1, MPI_COMM_WORLD, &mpiStatus);
          MPI_Recv(&lostParticles[offset2][0], (npVector[islave]-nLeftVector[islave])*totalPropertiesPerParticle, MPI_DOUBLE, islave, 
                   1, MPI_COMM_WORLD, &mpiStatus);
        }
        offset1 += nLeftVector[islave];
        offset2 += npVector[islave]-nLeftVector[islave];
      }
      nLost = np - nLeft;
    } else {
      MPI_Send(&coord[0][0], nLeftLocal*totalPropertiesPerParticle, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
      MPI_Send(&coord[nLeftLocal][0], (npLocal-nLeftLocal)*totalPropertiesPerParticle, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }
    
    free(npVector);
    free(nLeftVector);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (myid==0) {
#if MPI_DEBUG
      printf("%ld particles tracked in total:\n", np);
      printf("%ld particles left in total:\n", nLeft);
      for (ip=0; ip<nLeft; ip++)
        printf("%ld: %le, %le, %ld\n", ip, coord[ip][0], coord[ip][2], (long)coord[ip][particleIDIndex]);
      fflush(stdout);
      printf("%ld particles lost in total:\n", nLost);
      for (ip=0; ip<nLost; ip++)
        printf("%ld: %le, %le, %ld\n", ip, lostParticles[ip][0], lostParticles[ip][2], (long)lostParticles[ip][particleIDIndex]);
      fflush(stdout);
      fpSurvived = fopen("survived.sdds", "w");
      fprintf(fpSurvived, "SDDS1\n&column name=PID type=long &end\n");
      fprintf(fpSurvived, "&column name=u0 type=double &end\n");
      fprintf(fpSurvived, "&column name=v0 type=double &end\n");
      fprintf(fpSurvived, "&data mode=ascii no_row_counts=1 &end\n");
      fpLost = fopen("lost.sdds", "w");
      fprintf(fpLost, "SDDS1\n&column name=PID type=long &end\n");
      fprintf(fpLost, "&column name=u0 type=double &end\n");
      fprintf(fpLost, "&column name=v0 type=double &end\n");
      fprintf(fpLost, "&column name=xLost type=double &end\n");
      fprintf(fpLost, "&column name=yLost type=double &end\n");
      fprintf(fpLost, "&column name=sLost type=double &end\n");
      fprintf(fpLost, "&data mode=ascii no_row_counts=1 &end\n");
#endif
      vLimit = tmalloc(sizeof(*vLimit)*nu);
      vLimit1 = tmalloc(sizeof(*vLimit1)*nu);
      uLimit = tmalloc(sizeof(*uLimit)*nu);
      xLost = tmalloc(sizeof(*xLost)*nu);
      yLost = tmalloc(sizeof(*yLost)*nu);
      sLost = tmalloc(sizeof(*sLost)*nu);
      deltaLost = tmalloc(sizeof(*deltaLost)*nu);
      for (iu=0; iu<nu; iu++) {
        vLimit[iu] = 0;
        uLimit[iu] = iu*du + umin;
        xLost[iu] = yLost[iu] = sLost[iu] = deltaLost[iu] = DBL_MAX;
        if (side==0) {
          vLimit1[iu] = vmax + dv;
        } else {
          vLimit1[iu] = vmin + dv;
        }
      }
      for (ip=0; ip<nLeft; ip++) {
        /* surviving particles */
        long pid;
        double v;
        pid = coord[ip][particleIDIndex];
        iu = (pid-1)/nv;
        iv = (pid-1)%nv;
        v = iv*dv;
        if ((side==0 && v>vLimit[iu]) || (side==1 && v<vLimit[iu]))
          vLimit[iu] = v;
#if MPI_DEBUG
        fprintf(fpSurvived, "%ld %le %le\n", pid, iu*du+umin, v);
        fflush(fpSurvived);
#endif
      }
      for (ip=0; ip<nLost; ip++) {
        /* lost particles */
        long pid;
        double v;
        pid = lostParticles[ip][particleIDIndex];
        iu = (pid-1)/nv;
        iv = (pid-1)%nv;
        v = iv*dv;
        if ((side==0 && v<vLimit[iu]) || (side==1 && v>vLimit[iu]))
          vLimit[iu] = v;
        if ((side==0 && v<vLimit1[iu]) || (side==1 && v>vLimit1[iu])) {
          vLimit1[iu] = v;
          xLost[iu] = coord[ip][0];
          yLost[iu] = coord[ip][2];
          sLost[iu] = coord[ip][4];
          deltaLost[iu] = coord[ip][5];
        }
#if MPI_DEBUG
        fprintf(fpLost, "%ld %le %le %le %le %le\n", pid, iu*du+umin, v, lostParticles[ip][0], lostParticles[ip][2], 
                lostParticles[ip][4]);
        fflush(fpLost);
#endif
      }
#if MPI_DEBUG
      fclose(fpSurvived);
      fclose(fpLost);
#endif

      if (verbosity >= 1) {
        printf("Found limits\n");
        fflush(stdout);
      }
    
      if (nLeft) {
        for (iu=0; iu<nu; iu++)
          area += fabs(vLimit[iu]*du);
      }
      if (verbosity >= 1) {
        printf("Area = %le\n", area);
        fflush(stdout);
      }
      
      if (output) {
        if (delta_mode) {
          for (iu=0; iu<nu; iu++) {
            if (!SDDS_SetRowValues(&SDDS_aperture, SDDS_SET_BY_INDEX|SDDS_PASS_BY_VALUE, iu+side*nu,
                                   IC_U, uLimit[iu], IC_V, vLimit[iu],
                                   IC_XLOSTD, xLost[iu], IC_YLOSTD, yLost[iu], IC_SLOSTD, sLost[iu], 
                                   IC_DELTALOSTD, deltaLost[iu], -1)) {
              SDDS_SetError("Problem setting row values in SDDS table (do_aperture_search)");
              SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
            }
          }
        } else {
          for (iu=0; iu<nu; iu++) {
            if (!SDDS_SetRowValues(&SDDS_aperture, SDDS_SET_BY_INDEX|SDDS_PASS_BY_VALUE, iu+side*nu,
                                   IC_X, uLimit[iu], IC_Y, vLimit[iu],
                                   IC_XLOST, xLost[iu], IC_YLOST, yLost[iu], IC_SLOST, sLost[iu], -1)) {
              SDDS_SetError("Problem setting row values in SDDS table (do_aperture_search)");
              SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
            }
          }
        }
        if (sides==1 || side==1) {
          if (!SDDS_SetParameters(&SDDS_aperture, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                                  "Area", area, NULL)) {
            SDDS_SetError("Problem setting parameters values in SDDS table (do_aperture_search)");
            SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
          }
          if (!SDDS_WriteTable(&SDDS_aperture)) {
            SDDS_SetError("Problem writing SDDS table (do_aperture_search)");
            SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
          }
          if (!inhibitFileSync)
            SDDS_DoFSync(&SDDS_aperture);
        }
      }
      free(uLimit);
      free(vLimit);
      free(vLimit1);
      free(xLost);
      free(yLost);
      free(sLost);
      free(deltaLost);
      free_czarray_2d((void **)lostParticles, np-nLeft, totalPropertiesPerParticle);
    } else
      area += 0;
  }

  double areaBuffer;
  areaBuffer = area;
  MPI_Allreduce(&areaBuffer, &area, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  *returnValue = area;

  free_czarray_2d((void **)coord, np, totalPropertiesPerParticle);

  return (1);
}

/* Line aperture search that makes use of larger parallel resources by tracking all particles for all
   lines at the same time (if enough processors are available
*/

long do_aperture_search_line_p(
  RUN *run,
  VARY *control,
  double *referenceCoord,
  ERRORVAL *errcon,
  LINE_LIST *beamline,
  long lines,
  double *returnValue) {
  double **coord;
  double p_central;
  long index, split, step, nSteps;
  long effort, n_trpoint, line;
  double area, dtheta;
  double orbit[6] = {0, 0, 0, 0, 0, 0};
  double *x0, *y0, *dx, *dy, *dxFactor, *dyFactor;
  double *xLimit, *yLimit, *xLost, *yLost, *sLost;
  double **xLost2, **yLost2, **sLost2;
  double **survived, **buffer;
  long maxSteps;
#  if DEBUG
  FILE *fpd, *fpd2, *fpd3;
  char s[100];
  sprintf(s, "pda-%03d.sdds", myid);
  fpd = fopen(s, "w");
  fprintf(fpd, "SDDS1\n&column name=x type=double &end\n&column name=y type=double &end\n&column name=survived type=short &end\n");
  fprintf(fpd, "&column name=split type=short &end\n&column name=line type=short &end\n&column name=step type=short &end\n");
  fprintf(fpd, "&column name=index type=short &end\n&column name=ID type=short &end\n&data mode=ascii no_row_counts=1 &end\n");
  sprintf(s, "pda-%03d.txt", myid);
  fpd2 = fopen(s, "w");
  fpd3 = fopen("pda.sdds", "w");
  fprintf(fpd3, "SDDS1\n&column name=x type=double &end\n&column name=y type=double &end\n&column name=survived type=short &end\n");
  fprintf(fpd3, "&column name=split type=short &end\n&column name=line type=short &end\n&column name=step type=short &end\n");
  fprintf(fpd3, "&column name=index type=short &end\n&column name=ID type=short &end\n&data mode=ascii no_row_counts=1 &end\n");
#  endif

#if MPI_DEBUG
  printf("Starting n-line aperture search\n");
  fflush(stdout);
#endif

  maxSteps = 1 / split_fraction + 0.5;
  if (nx > maxSteps)
    maxSteps = nx;

  coord = (double **)czarray_2d(sizeof(**coord), 1, totalPropertiesPerParticle);

  /* These arrays are used to store results for each line and each step on the line */
  survived = (double **)czarray_2d(sizeof(**survived), lines, maxSteps);
  xLost2 = (double **)czarray_2d(sizeof(**xLost2), lines, maxSteps);
  yLost2 = (double **)czarray_2d(sizeof(**yLost2), lines, maxSteps);
  sLost2 = (double **)czarray_2d(sizeof(**sLost2), lines, maxSteps);
  /* buffer for sharing data among processors */
  buffer = (double **)czarray_2d(sizeof(**buffer), lines, maxSteps);

  /* Some of these were just scalars in the previous version.  We use arrays to avoid recomputing values */
  dx = tmalloc(sizeof(*dx) * lines);
  dy = tmalloc(sizeof(*dy) * lines);
  x0 = tmalloc(sizeof(*x0) * lines);
  y0 = tmalloc(sizeof(*y0) * lines);
  xLost = tmalloc(sizeof(*xLost) * lines);
  yLost = tmalloc(sizeof(*yLost) * lines);
  sLost = tmalloc(sizeof(*sLost) * lines);
  dxFactor = tmalloc(sizeof(*dxFactor) * lines);
  dyFactor = tmalloc(sizeof(*dyFactor) * lines);
  xLimit = calloc(sizeof(*xLimit), lines);
  yLimit = calloc(sizeof(*yLimit), lines);

  for (line = 0; line < lines; line++)
    xLost[line] = yLost[line] = sLost[line] = DBL_MAX;

  switch (lines) {
  case 1:
    dxFactor[0] = 1;
    dyFactor[0] = 1;
    break;
  case 2:
    dxFactor[0] = -1;
    dyFactor[0] = 1;
    dxFactor[1] = 1;
    dyFactor[1] = 1;
    break;
  case 3:
    dxFactor[0] = 0;
    dyFactor[0] = 1;
    dxFactor[1] = dyFactor[1] = 1 / sqrt(2);
    dxFactor[2] = 1;
    dyFactor[2] = 0;
    break;
  default:
    if (full_plane == 0)
      dtheta = PI / (lines - 1);
    else
      dtheta = PIx2 / (lines - 1);
    for (line = 0; line < lines; line++) {
      dxFactor[line] = sin(-PI / 2 + dtheta * line);
      dyFactor[line] = cos(-PI / 2 + dtheta * line);
      if (fabs(dxFactor[line]) < 1e-6)
        dxFactor[line] = 0;
      if (fabs(dyFactor[line]) < 1e-6)
        dyFactor[line] = 0;
    }
    break;
  }

  effort = 0;
  if (offset_by_orbit) {
    /* N.B.: for an off-momentum orbit that is created with an initial
     * MALIGN element, the momentum offset will not appear in the
     * referenceCoord array.  So this works if the user sets ON_PASS=0
     * for the MALIGN.
     */
    memcpy(orbit, referenceCoord, sizeof(*referenceCoord) * 6);
  }

  if (verbosity >= 1) {
    printf("** Starting %ld-line aperture search\n", lines);
    fflush(stdout);
  }

  if (isMaster && output) {
    SDDS_ClearErrors();
    if (!SDDS_StartTable(&SDDS_aperture, lines)) {
      SDDS_SetError("Unable to start SDDS table (do_aperture_search)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    SDDS_SetParameters(&SDDS_aperture, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, IP_STEP, control->i_step, -1);
    if (control->n_elements_to_vary) {
      for (index = 0; index < control->n_elements_to_vary; index++)
        if (!SDDS_SetParameters(&SDDS_aperture, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, index + N_PARAMETERS,
                                control->varied_quan_value[index], -1))
          break;
    }
    if (SDDS_NumberOfErrors()) {
      SDDS_SetError("Problem setting SDDS parameter values (do_aperture_search)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  }

  /* For each split, loop over lines and steps on lines to evaluate which particles survive */
  for (split = 0; split <= n_splits; split++) {
    if (split == 0)
      nSteps = nx;
    else
      nSteps = 1 / split_fraction - 0.5;
    if (nSteps < 1)
      nSteps = 1;
    /* Mark all particles as not surviving */
    for (line = 0; line < lines; line++) {
      for (step = 0; step < nSteps; step++)
        survived[line][step] = 0;
    }
    /* loop over lines.  'index' counts the particles to be tracked */
    for (index = line = 0; line < lines; line++) {
      /* Set start and delta values for the line */
      if (split == 0) {
        if (!slope_mode) {
          dx[line] = xmax / (nx - 1) * dxFactor[line];
          dy[line] = ymax / (nx - 1) * dyFactor[line];
        } else {
          dx[line] = xpmax / (nx - 1) * dxFactor[line];
          dy[line] = ypmax / (nx - 1) * dyFactor[line];
        }
        x0[line] = y0[line] = 0;
      } else {
        x0[line] = xLimit[line];
        y0[line] = yLimit[line];
        dx[line] *= split_fraction;
        dy[line] *= split_fraction;
        x0[line] += dx[line];
        y0[line] += dy[line];
      }
      /* step along the line */
      for (step = 0; step < nSteps; step++, index++) {
        /* initialize 2d arrays to zero even if we won't track this particle */
        xLost2[line][step] = 0;
        yLost2[line][step] = 0;
        sLost2[line][step] = 0;
        survived[line][step] = 0;
#  if DEBUG
        fprintf(fpd2, "myid=%d, split=%ld, line=%ld, step=%ld, index=%ld, n_processors=%d---",
                myid, split, line, step, index, n_processors);
        fflush(fpd2);
#  endif

        /* decide if we are going to track this particle */
        if (myid != index % n_processors) {
#  if DEBUG
          fprintf(fpd2, "not tracking\n");
          fflush(fpd2);
#  endif
          continue;
        }
#  if DEBUG
        fprintf(fpd2, "tracking---");
        fflush(fpd2);
#  endif

        /* Track a particle */
        memcpy(coord[0], orbit, sizeof(*orbit) * 6);
        if (!slope_mode) {
          coord[0][0] = step * dx[line] + x0[line] + orbit[0];
          coord[0][2] = step * dy[line] + y0[line] + orbit[2];
        } else {
          coord[0][1] = step * dx[line] + x0[line] + orbit[1];
          coord[0][3] = step * dy[line] + y0[line] + orbit[3];
        }
        p_central = run->p_central;
        n_trpoint = 1;
        if (do_tracking(NULL, coord, n_trpoint, &effort, beamline, &p_central,
                        NULL, NULL, NULL, NULL, run, control->i_step,
                        APERTURE_TRACKING_FLAGS, control->n_passes, 0, NULL, NULL, NULL, NULL, NULL) != 1) {
          /* Particle lost, so record information */
          xLost2[line][step] = coord[0][0];
          yLost2[line][step] = coord[0][2];
          sLost2[line][step] = coord[0][4];
          survived[line][step] = 0;
#  if DEBUG
          fprintf(fpd2, "lost\n");
          fflush(fpd2);
#  endif
        } else {
          /* Particle survived */
          survived[line][step] = 1;
#  if DEBUG
          fprintf(fpd2, "survived\n");
          fflush(fpd2);
#  endif
        }
#  if DEBUG
        fprintf(fpd, "%le %le %.0f %ld %ld %ld %ld %d\n", step * dx[line] + x0[line], step * dy[line] + y0[line], survived[line][step], split, line, step,
                index, myid);
        fflush(fpd);
#  endif
      }
    }
    /* Wait for all processors to exit the loop */
#  if MPI_DEBUG
    printf("Waiting on barrier after loop over all lines\n");
    fflush(stdout);
#  endif
    MPI_Barrier(MPI_COMM_WORLD);

    /* Sum values of arrays over all processors */
    MPI_Allreduce(survived[0], buffer[0], lines * maxSteps, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    memcpy(survived[0], buffer[0], sizeof(double) * lines * maxSteps);

    MPI_Allreduce(xLost2[0], buffer[0], lines * maxSteps, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    memcpy(xLost2[0], buffer[0], sizeof(double) * lines * maxSteps);

    MPI_Allreduce(yLost2[0], buffer[0], lines * maxSteps, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    memcpy(yLost2[0], buffer[0], sizeof(double) * lines * maxSteps);

    MPI_Allreduce(sLost2[0], buffer[0], lines * maxSteps, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    memcpy(sLost2[0], buffer[0], sizeof(double) * lines * maxSteps);

#  if DEBUG
    for (line = 0; line < lines; line++)
      for (step = 0; step < nSteps; step++) {
        fprintf(fpd3, "%le %le %.0f %ld %ld %ld %ld %d\n", step * dx[line] + x0[line], step * dy[line] + y0[line], survived[line][step], split, line, step,
                index, myid);
        fflush(fpd3);
      }
#  endif

    /* Scan array to determine x and y stability limits */
    for (line = 0; line < lines; line++) {
      if (split == 0) {
#  if DEBUG
        printf("Checking line=%ld, split = 0\n", line);
        fflush(stdout);
#  endif
        if (!survived[line][0]) {
          *returnValue = 0;
          free_czarray_2d((void **)coord, 1, totalPropertiesPerParticle);
          free_czarray_2d((void **)survived, lines, maxSteps);
          free_czarray_2d((void **)xLost2, lines, maxSteps);
          free_czarray_2d((void **)yLost2, lines, maxSteps);
          free_czarray_2d((void **)sLost2, lines, maxSteps);
          free_czarray_2d((void **)buffer, lines, maxSteps);
          
          free(dx);
          free(dy);
          free(x0);
          free(y0);
          free(xLost);
          free(yLost);
          free(dxFactor);
          free(dyFactor);
          free(xLimit);
          free(yLimit);
          return 1;
        }
        for (step = 1; step < nSteps; step++) {
          if (!survived[line][step])
            break;
        }
#  if DEBUG
        printf("split=0, lost at step=%ld of %ld\n", step, nSteps);
#  endif
      } else {
#  if DEBUG
        printf("Checking line=%ld, split = %ld\n", line, split);
        fflush(stdout);
#  endif
        for (step = 0; step < nSteps; step++) {
          if (!survived[line][step])
            break;
        }
#  if DEBUG
        printf("split=%ld, lost at step=%ld of %ld\n", split, step, nSteps);
        fflush(stdout);
#  endif
        if (step == 0)
          continue;
      }
      step--;

#  if DEBUG
      printf("line=%ld, split=%ld, x0=%le, y0=%le, dx=%le, dy=%le\n", line, split,
             x0[line], y0[line], dx[line], dy[line]);
      printf("Particle survived at step=%ld, x=%le, y=%le\n", step, x0[line] + step * dx[line], y0[line] + step * dy[line]);
      fflush(stdout);
#  endif

      if ((dx[line] > 0 && xLimit[line] < (step * dx[line] + x0[line])) ||
          (dx[line] == 0 && yLimit[line] < (step * dy[line] + y0[line])) ||
          (dx[line] < 0 && xLimit[line] > (step * dx[line] + x0[line]))) {
#  if DEBUG
        printf("Change limit from (%le, %le) to (%le, %le)\n",
               xLimit[line], yLimit[line], step * dx[line] + x0[line], step * dy[line] + y0[line]);
        fflush(stdout);
#  endif
        xLimit[line] = step * dx[line] + x0[line];
        yLimit[line] = step * dy[line] + y0[line];
        if (step < (nSteps - 1)) {
          xLost[line] = xLost2[line][step + 1];
          yLost[line] = yLost2[line][step + 1];
          sLost[line] = sLost2[line][step + 1];
        }
      }
    }
  }

#if MPI_DEBUG
  printf("Finished tracking for n-line aperture search\n");
  fflush(stdout);
#endif

  area = 0;
  if (isMaster) {
    /* Analyze and store results */

    if (output) {
      for (line = 0; line < lines; line++)
        if (!SDDS_SetRowValues(&SDDS_aperture, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, line,
                               IC_X, xLimit[line], IC_Y, yLimit[line],
                               IC_XLOST, xLost[line], IC_YLOST, yLost[line], IC_SLOST, sLost[line], -1)) {
          SDDS_SetError("Problem setting SDDS row values (do_aperture_search)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
    }

    if (lines > 1) {
      /* compute the area */
      area = trimApertureSearchResult(lines, xLimit, yLimit, dxFactor, dyFactor, full_plane);
    }
    *returnValue = area;

    if (output) {
      if (!SDDS_SetColumn(&SDDS_aperture, SDDS_SET_BY_INDEX, xLimit, lines, IC_XC) ||
          !SDDS_SetColumn(&SDDS_aperture, SDDS_SET_BY_INDEX, yLimit, lines, IC_YC) ||
          !SDDS_SetParameters(&SDDS_aperture, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                              "Area", area, NULL)) {
        SDDS_SetError("Problem setting parameters values in SDDS table (do_aperture_search)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
      if (control->n_elements_to_vary) {
        long i;
        for (i = 0; i < control->n_elements_to_vary; i++)
          if (!SDDS_SetParameters(&SDDS_aperture, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, i + N_PARAMETERS,
                                  control->varied_quan_value[i], -1))
            break;
      }

      if (!SDDS_WriteTable(&SDDS_aperture)) {
        SDDS_SetError("Problem writing SDDS table (do_aperture_search)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
      if (!inhibitFileSync)
        SDDS_DoFSync(&SDDS_aperture);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
#if MPI_DEBUG
  printf("Finished collation of results for n-line aperture search\n");
  fflush(stdout);
#endif

  free_czarray_2d((void **)coord, 1, totalPropertiesPerParticle);
  free_czarray_2d((void **)survived, lines, maxSteps);
  free_czarray_2d((void **)xLost2, lines, maxSteps);
  free_czarray_2d((void **)yLost2, lines, maxSteps);
  free_czarray_2d((void **)sLost2, lines, maxSteps);
  free_czarray_2d((void **)buffer, lines, maxSteps);

  free(dx);
  free(dy);
  free(x0);
  free(y0);
  free(xLost);
  free(yLost);
  free(dxFactor);
  free(dyFactor);
  free(xLimit);
  free(yLimit);

  return (1);
}

#endif
