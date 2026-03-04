/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: orbtraj_output.c
 * purpose: trajectory/orbit-correction trajectory/orbit output for elegant
 * 
 * Michael Borland, 1993.
 */
#include "mdb.h"
#include "track.h"
#include "SDDS.h"
#include "math.h"

/* Variables for output of corrector data */
static long SDDS_orb_initialized = 0;
static SDDS_TABLE SDDS_orb;

#define IC_S 0
#define IC_X 1
#define IC_Y 2
#define IC_ELEMENT 3
#define IC_OCCURENCE 4
#define IC_TYPE 5
#define IC_N 6
#define N_COLUMNS 7
static SDDS_DEFINITION column_definition[N_COLUMNS] = {
  {"s", "&column name=s, units=m, type=double, description=\"Distance\" &end"},
  {"x", "&column name=x, units=m, type=double, description=\"Horizontal position\" &end"},
  {"y", "&column name=y, units=m, type=double, description=\"Vertical position\" &end"},
  {"ElementName", "&column name=ElementName, type=string, description=\"Element name\", format_string=%10s &end"},
  {"ElementOccurence",
   "&column name=ElementOccurence, type=long, description=\"Occurence of element\", format_string=%6ld &end"},
  {"ElementType", "&column name=ElementType, type=string, description=\"Element-type name\", format_string=%10s &end"},
  {"Particles", "&column name=Particles, units=m, type=long, description=\"Number of particles\" &end"},
};

#define IP_STEP 0
#define IP_STAGE 1
#define N_PARAMETERS 2
static SDDS_DEFINITION parameter_definition[N_PARAMETERS] = {
  {"Step", "&parameter name=Step, type=long, description=\"Simulation step\" &end"},
  {"Stage", "&parameter name=Stage, type=string, description=\"Stage of correction\" &end"},
};

void setup_orb_traj_output(char *filename, char *mode, RUN *run) {
  char s[200];

  log_entry("setup_orb_traj_output");
  if (!SDDS_orb_initialized) {
    SDDS_orb_initialized = 1;
    zero_memory(&SDDS_orb, sizeof(SDDS_orb));
  }

  if (!filename)
    bombElegant("NULL filename passed (setup_orb_traj_output)", NULL);
  if (!run)
    bombElegant("NULL RUN pointer passed (setup_orb_traj_output)", NULL);
  if (SDDS_IsActive(&SDDS_orb) == 1) {
    if (!SDDS_Terminate(&SDDS_orb)) {
      printf("Unable to terminate SDDS output for correctors (setup_orb_traj_output)\n");
      fflush(stdout);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
  }

  sprintf(s, "%s-correction output", mode);
  SDDS_ElegantOutputSetup(&SDDS_orb, filename, SDDS_BINARY, 1, s, run->runfile, run->lattice,
                          parameter_definition, N_PARAMETERS, column_definition, N_COLUMNS, "setup_orb_traj_output",
                          SDDS_EOS_NEWFILE | SDDS_EOS_COMPLETE);

  log_exit("setup_orb_traj_output");
}

void dump_orb_traj(TRAJECTORY *traj, long n_elems, char *description, long step) {
  long i, n, occurence;
  double position;
  char *name;

  if (!SDDS_orb_initialized)
    return;

  log_entry("dump_orb_traj");

  /* count number of trajectory elements actually used */
  for (i = 1; i < n_elems + 1; i++) {
    if (!traj[i].elem)
      break;
  }
  n = i;

  if (!SDDS_StartTable(&SDDS_orb, n)) {
    SDDS_SetError("Unable to start SDDS table (dump_orb_traj)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  if (!SDDS_SetParameters(&SDDS_orb, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE,
                          IP_STEP, step, IP_STAGE, description, -1)) {
    SDDS_SetError("Unable to set SDDS parameters (dump_orb_traj)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  position = traj[1].elem->end_pos -
             (entity_description[traj[1].elem->type].flags & HAS_LENGTH ? *((double *)traj[1].elem->p_elem) : 0.0);
  name = "_BEG_";
  occurence = 1;

  for (i = 0; i < n; i++) {
    if (i) {
      position = traj[i].elem->end_pos;
      name = traj[i].elem->name;
      occurence = traj[i].elem->occurence;
    }
    if (!SDDS_SetRowValues(&SDDS_orb, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, i,
                           IC_S, position, IC_X, traj[i].centroid[0], IC_Y, traj[i].centroid[2],
                           IC_N, traj[i].n_part, IC_ELEMENT, name, IC_OCCURENCE, occurence,
                           IC_TYPE, i == 0 ? "MARK" : entity_name[traj[i].elem->type], -1)) {
      printf("Unable to set row %ld values (dump_orb_traj)\n", i);
      fflush(stdout);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
  }

  if (!SDDS_WriteTable(&SDDS_orb)) {
    SDDS_SetError("Unable to write orbit/trajectory data (dump_orb_traj)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  if (!inhibitFileSync)
    SDDS_DoFSync(&SDDS_orb);
  if (!SDDS_EraseData(&SDDS_orb)) {
    SDDS_SetError("Unable to erase orbit/trajectory data (dump_orb_traj)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  log_exit("dump_orb_traj");
}

void finish_orb_traj_output() {
  if (!SDDS_orb_initialized)
    return;
  if (!SDDS_Terminate(&SDDS_orb)) {
    printf("Unable to terminate SDDS output for orbit/trajectory (finish_orb_traj_output)\n");
    fflush(stdout);
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    exitElegant(1);
  }
  SDDS_orb_initialized = 0;
}

/* Variables for output of bpm data */
static long SDDS_bpm_initialized = 0;
static SDDS_TABLE SDDS_bpm;

#define ICB_S 0
#define ICB_X 1
#define ICB_Y 2
#define ICB_ELEMENT 3
#define ICB_OCCURENCE 4
#define ICB_TYPE 5
#define NB_COLUMNS 6
static SDDS_DEFINITION column_definition_bpm[N_COLUMNS] = {
  {"s", "&column name=s, units=m, type=double, description=\"Distance\" &end"},
  {"xError", "&column name=xError, units=m, type=double, description=\"x-dx-setpoint\" &end"},
  {"yError", "&column name=yError, units=m, type=double, description=\"y-dy-setpoint\" &end"},
  {"ElementName", "&column name=ElementName, type=string, description=\"Element name\", format_string=%10s &end"},
  {"ElementOccurence",
   "&column name=ElementOccurence, type=long, description=\"Occurence of element\", format_string=%6ld &end"},
  {"ElementType", "&column name=ElementType, type=string, description=\"Element-type name\", format_string=%10s &end"},
};

#define IPB_STEP 0
#define NB_PARAMETERS 1
static SDDS_DEFINITION parameter_definition_bpm[N_PARAMETERS] = {
  {"Step", "&parameter name=Step, type=long, description=\"Simulation step\" &end"},
};

void setup_bpm_output(char *filename, RUN *run) {
  if (!SDDS_bpm_initialized) {
    SDDS_bpm_initialized = 1;
    zero_memory(&SDDS_bpm, sizeof(SDDS_bpm));
  }

  if (!filename)
    bombElegant("NULL filename passed (setup_bpm_output)", NULL);
  if (!run)
    bombElegant("NULL RUN pointer passed (setup_bpm_output)", NULL);
  if (SDDS_IsActive(&SDDS_bpm) == 1) {
    if (!SDDS_Terminate(&SDDS_bpm)) {
      printf("Unable to terminate SDDS output for correctors (setup_bpm_output)\n");
      fflush(stdout);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
  }

  SDDS_ElegantOutputSetup(&SDDS_bpm, filename, SDDS_BINARY, 1, NULL, run->runfile, run->lattice,
                          parameter_definition_bpm, NB_PARAMETERS, column_definition_bpm, NB_COLUMNS, "setup_bpm_output",
                          SDDS_EOS_NEWFILE | SDDS_EOS_COMPLETE);
}

void dump_bpm_data(TRAJECTORY *traj, long n_elems, char *description, long step) {
  long i, j, n, occurence;
  double position, xReading, yReading;
  char *name;

  if (!SDDS_bpm_initialized)
    return;

  /* count number of monitor elements */
  n = 0;
  for (i = 1; i < n_elems + 1; i++) {
    if (!traj[i].elem)
      break;
    if (traj[i].elem->type == T_MONI || traj[i].elem->type == T_HMON || traj[i].elem->type == T_VMON)
      n++;
  }

  if (!SDDS_StartTable(&SDDS_bpm, n)) {
    SDDS_SetError("Unable to start SDDS table (dump_bpm_data)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  if (!SDDS_SetParameters(&SDDS_bpm, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE,
                          IPB_STEP, step, -1)) {
    SDDS_SetError("Unable to set SDDS parameters (dump_bpm_data)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  j = 0;
  for (i = 1; i < n_elems + 1; i++) {
    if (!traj[i].elem || !(traj[i].elem->type == T_MONI || traj[i].elem->type == T_HMON || traj[i].elem->type == T_VMON))
      continue;
    position = traj[i].elem->end_pos;
    name = traj[i].elem->name;
    occurence = traj[i].elem->occurence;
    xReading = yReading = 0;
    if (traj[i].elem->type == T_MONI || traj[i].elem->type == T_HMON)
      xReading = computeMonitorReading(traj[i].elem, 0, traj[i].centroid, 0);
    if (traj[i].elem->type == T_MONI || traj[i].elem->type == T_VMON)
      yReading = computeMonitorReading(traj[i].elem, 2, traj[i].centroid, 0);

    if (!SDDS_SetRowValues(&SDDS_bpm, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, j,
                           ICB_S, position, ICB_X, xReading, ICB_Y, yReading,
                           ICB_ELEMENT, name, ICB_OCCURENCE, occurence,
                           ICB_TYPE, i == 0 ? "MARK" : entity_name[traj[i].elem->type], -1)) {
      printf("Unable to set row %ld values (dump_bpm_data)\n", i);
      fflush(stdout);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
    j++;
  }

  if (!SDDS_WriteTable(&SDDS_bpm)) {
    SDDS_SetError("Unable to write orbit/trajectory data (dump_bpm_data)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  if (!inhibitFileSync)
    SDDS_DoFSync(&SDDS_bpm);
  if (!SDDS_EraseData(&SDDS_bpm)) {
    SDDS_SetError("Unable to erase orbit/trajectory data (dump_bpm_data)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  log_exit("dump_bpm_data");
}

void finish_bpm_output() {
  if (!SDDS_bpm_initialized)
    return;
  if (!SDDS_Terminate(&SDDS_bpm)) {
    printf("Unable to terminate SDDS output for orbit/trajectory (finish_bpm_output)\n");
    fflush(stdout);
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    exitElegant(1);
  }
  SDDS_bpm_initialized = 0;
}
