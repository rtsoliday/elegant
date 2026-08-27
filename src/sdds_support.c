/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: sdds_support.c
 * purpose: routines for working with SDDS files
 * 
 * Michael Borland, 1993.
 */
#include "mdb.h"
#include "fftpackC.h"
#include "track.h"
#include "SDDS.h"
#include "oagphy.h" /* COUPLED_RESULTS, ComputeCoupledParameters() for coupled_sigma output */
#ifdef HAVE_GPU
#  include <gpu_base.h>
#endif

void SDDS_ElegantOutputSetup(SDDS_TABLE *SDDS_table, char *filename, long mode, long lines_per_row,
                             char *contents, char *command_file, char *lattice_file, SDDS_DEFINITION *parameter_definition,
                             long n_parameters, SDDS_DEFINITION *column_definition, long n_columns,
                             char *caller, long flags) {
  static char description[SDDS_MAXLINE];
  long i, last_index, index;

  log_entry("SDDS_ElegantOutputSetup");
  last_index = -1;

#if MPI_DEBUG
  printf("SDDS_ElegantOutputSetup 0 for filename = %s\n", filename);
  fflush(stdout);
#endif
#if USE_MPI && !SDDS_MPI_IO
  if (myid != 0)
    return;
#endif

  if (flags & SDDS_EOS_NEWFILE) {
    if (SDDS_IsActive(SDDS_table) == 1) {
      if (!SDDS_Terminate(SDDS_table)) {
        printf("Unable to terminate SDDS output (%s)\n", caller);
        fflush(stdout);
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        exitElegant(1);
      }
    }
    sprintf(description, "%s--input: %s  lattice: %s", contents, command_file, lattice_file);
#if SDDS_MPI_IO
    /* We still need serial IO for some operations */
    if ((SDDS_table->parallel_io && (!SDDS_Parallel_InitializeOutputElegant(SDDS_table, description, contents, filename))) ||
        (!SDDS_table->parallel_io && (!SDDS_InitializeOutputElegant(SDDS_table, mode, 1, description, contents, filename))))
#else
    if (!SDDS_InitializeOutputElegant(SDDS_table, mode, 1, description, contents, filename))
#endif
    {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
  }

#if MPI_DEBUG
  printf("SDDS_ElegantOutputSetup 1 for filename = %s\n", filename);
  fflush(stdout);
#endif

  /* define SDDS parameters */
  for (i = 0; i < n_parameters; i++) {
    index = -1;
    if (!SDDS_ProcessParameterString(SDDS_table, parameter_definition[i].text, 0) ||
        (index = SDDS_GetParameterIndex(SDDS_table, parameter_definition[i].name)) < 0) {
      printf("Unable to define SDDS parameter for %s--string was:\n%s\n", caller,
             parameter_definition[i].text);
      fflush(stdout);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
    if (last_index == -1 && flags & SDDS_EOS_NEWFILE && index != 0)
      printWarning("First-defined parameter index for SDDS file is not 0.", filename);
    if (last_index != -1 && index != (last_index + 1))
      printWarning("Parameter indices for SDDS file are not sequential.", filename);
    last_index = index;
  }

#if MPI_DEBUG
  printf("SDDS_ElegantOutputSetup 2 for filename = %s\n", filename);
  fflush(stdout);
#endif

  last_index = -1;
  /* define SDDS columns */
  for (i = 0; i < n_columns; i++) {
    index = -1;
    if (!SDDS_ProcessColumnString(SDDS_table, column_definition[i].text, 0) ||
        (index = SDDS_GetColumnIndex(SDDS_table, column_definition[i].name)) < 0) {
      printf("Unable to define SDDS column for %s--string was:\n%s\n", caller,
             column_definition[i].text);
      fflush(stdout);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
    if (last_index == -1 && flags & SDDS_EOS_NEWFILE && index != 0)
      printWarning("First-defined column index for SDDS file is not 0", filename);
    if (last_index != -1 && index != (last_index + 1))
      printWarning("Column indices for SDDS file are not sequential", filename);
    last_index = index;
  }

#if MPI_DEBUG
  printf("SDDS_ElegantOutputSetup 3 for filename = %s\n", filename);
  fflush(stdout);
#endif

#if SDDS_MPI_IO
  /* In the case of parallel IO, the WiteLayout will be called at the time it dumps data or setups the output,
     as the communicator information is required */
  if (!SDDS_table->parallel_io && isMaster)
#endif
    if (flags & SDDS_EOS_COMPLETE) {
      if (!SDDS_WriteLayout(SDDS_table)) {
        printf("Unable to write SDDS layout for file %s (%s)\n",
               filename?filename:"?", caller?caller:"?");
        fflush(stdout);
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        exitElegant(1);
      }
    }

#if MPI_DEBUG
  printf("SDDS_ElegantOutputSetup 4 for filename = %s\n", filename);
  fflush(stdout);
#endif

  log_exit("SDDS_ElegantOutputSetup");
}

#define STANDARD_PARAMETERS 3
static SDDS_DEFINITION standard_parameter[STANDARD_PARAMETERS] = {
  {"Step", "&parameter name=Step, type=long, description=\"Simulation step\" &end"},
  {"SVNVersion", "&parameter name=SVNVersion, type=string, description=\"SVN version number\", fixed_value=" SVN_VERSION " &end"},
  {"PassLength", (char *)"&parameter name=PassLength, type=double, description=\"Length of a single pass\" &end"},
};

#define ELEMENT_COLUMNS 4
static SDDS_DEFINITION element_column[ELEMENT_COLUMNS] = {
  {"s", "&column name=s, units=m, type=double, description=\"Distance\" &end"},
  {"ElementName", "&column name=ElementName, type=string, description=\"Element name\", format_string=%10s &end"},
  {"ElementOccurence",
   "&column name=ElementOccurence, type=long, description=\"Occurence of element\", format_string=%6ld &end"},
  {"ElementType", "&column name=ElementType, type=string, description=\"Element-type name\", format_string=%10s &end"},
};

#define PHASE_SPACE_COLUMNS 7
#define SPIN_COLUMNS 3
static SDDS_DEFINITION phase_space_column[PHASE_SPACE_COLUMNS+SPIN_COLUMNS] = {
  {"x", "&column name=x, units=m, type=double &end"},
  {"xp", "&column name=xp, symbol=\"x'\", type=double &end"},
  {"y", "&column name=y, units=m, type=double &end"},
  {"yp", "&column name=yp, symbol=\"y'\", type=double &end"},
  {"t", "&column name=t, units=s, type=double &end"},
  {"p", "&column name=p, units=\"m$be$nc\", type=double &end"},
  {"particleID", "&column name=particleID, type=ulong64 &end"},
  {"spx", "&column name=spx, type=double &end"},
  {"spy", "&column name=spy, type=double &end"},
  {"spz", "&column name=spz, type=double &end"},
};

#define PHASE_SPACE_PARAMETERS 9
static SDDS_DEFINITION phase_space_parameter[PHASE_SPACE_PARAMETERS] = {
  {"Step", "&parameter name=Step, type=long, description=\"Simulation step\" &end"},
  {"pCentral", "&parameter name=pCentral, symbol=\"p$bcen$n\", units=\"m$be$nc\", type=double, description=\"Reference beta*gamma\" &end"},
  {"Charge", "&parameter name=Charge, type=double, units=C, description=\"Bunch charge before sampling\" &end"},
  {"Particles", "&parameter name=Particles, type=long64, description=\"Number of particles before sampling\" &end"},
  {"IDSlotsPerBunch", "&parameter name=IDSlotsPerBunch, type=long64, description=\"Number of particle ID slots reserved to a bunch\" &end"},
  {"SVNVersion", "&parameter name=SVNVersion, type=string, description=\"SVN version number\", fixed_value=" SVN_VERSION " &end"},
  {"tOffset", "&parameter name=tOffset type=double, description=\"Per-step offset of t values\" &end"},
  {"SampledCharge", "&parameter name=SampledCharge, type=double, units=C, description=\"Sampled charge\" &end"},
  {"SampledParticles", "&parameter name=SampledParticles, type=long64, description=\"Sampled number of particles\" &end"},
};

void SDDS_PhaseSpaceSetup(SDDS_TABLE *SDDS_table, char *filename, long mode, long lines_per_row, char *contents,
                          char *command_file, char *lattice_file, char *caller) {
  log_entry("SDDS_PhaseSpaceSetup");
#if SDDS_MPI_IO
  if (distributedBeam)
    SDDS_table->parallel_io = 1;
  else
    SDDS_table->parallel_io = 0;
  /* set up parallel IO information */
  SDDS_MPI_Setup(SDDS_table, SDDS_table->parallel_io, n_processors, myid, MPI_COMM_WORLD, 0);
#  if MPI_DEBUG
  printf("SDDS_PhaseSpaceSetup, filename = %s, SDDS_MPI_Setup done with parallel_io=%d\n", filename, SDDS_table->parallel_io);
  fflush(stdout);
#  endif
#endif
  SDDS_ElegantOutputSetup(SDDS_table, filename, mode, lines_per_row, contents, command_file, lattice_file,
                          phase_space_parameter, PHASE_SPACE_PARAMETERS - 2,
			  phase_space_column, PHASE_SPACE_COLUMNS + (spinCoordOffset>0 ? SPIN_COLUMNS : 0),
                          caller, SDDS_EOS_NEWFILE | SDDS_EOS_COMPLETE);
#if MPI_DEBUG
  printf("SDDS_ElegantOutputSetup done\n");
  fflush(stdout);
#endif
  log_exit("SDDS_PhaseSpaceSetup");
}

#define BEAM_LOSS_PARAMETERS 4
static SDDS_DEFINITION beam_loss_parameter[BEAM_LOSS_PARAMETERS] = {
  {"Step", "&parameter name=Step, type=long, description=\"Simulation step\" &end"},
  {"SVNVersion", "&parameter name=SVNVersion, type=string, description=\"SVN version number\", fixed_value=" SVN_VERSION " &end"},
  {"PassLength", (char *)"&parameter name=PassLength, type=double, description=\"Length of a single pass\" &end"},
  {"pCentral", "&parameter name=pCentral, symbol=\"p$bcen$n\", units=\"m$be$nc\", type=double, description=\"Reference beta*gamma\" &end"},
};

#define BEAM_LOSS_COLUMNS_BASIC 8
#define BEAM_LOSS_COLUMNS_EXTRA 3

static SDDS_DEFINITION beam_loss_column[BEAM_LOSS_COLUMNS_BASIC + BEAM_LOSS_COLUMNS_EXTRA] = {
  {"x", "&column name=x, units=m, type=double &end"},
  {"xp", "&column name=xp, symbol=\"x'\", type=double &end"},
  {"y", "&column name=y, units=m, type=double &end"},
  {"yp", "&column name=yp, symbol=\"y'\", type=double &end"},
  {"s", "&column name=s, units=m, type=double &end"},
  {"p", "&column name=p, units=\"m$be$nc\", type=double &end"},
  {"particleID", "&column name=particleID, type=ulong64 &end"},
  {"Pass", "&column name=Pass, type=long &end"},
  {"X", "&column name=X, units=m, type=double &end"},
  {"Z", "&column name=Z, units=m, type=double &end"},
  {"thetaX", "&column name=thetaX, type=double &end"},
};

static SDDS_DEFINITION spin_column[3] = {
  {"spx", "&column name=spx, type=double &end"},
  {"spy", "&column name=spy, type=double &end"},
  {"spz", "&column name=spz, type=double &end"},
};

void SDDS_BeamLossSetup(SDDS_TABLE *SDDS_table, char *filename, long mode, long lines_per_row, char *contents,
                        char *command_file, char *lattice_file, long includeGlobal, char *caller) {
  log_entry("SDDS_BeamLossSetup");
#if SDDS_MPI_IO
  SDDS_table->parallel_io = 1;
  /* set up parallel IO information */
  SDDS_MPI_Setup(SDDS_table, 1, n_processors, myid, MPI_COMM_WORLD, 0);
#endif

  SDDS_ElegantOutputSetup(SDDS_table, filename, mode, lines_per_row, contents, command_file, lattice_file,
                          beam_loss_parameter, BEAM_LOSS_PARAMETERS, beam_loss_column,
                          BEAM_LOSS_COLUMNS_BASIC + (includeGlobal ? BEAM_LOSS_COLUMNS_EXTRA : 0),
                          caller, SDDS_EOS_NEWFILE | (spinCoordOffset?0:SDDS_EOS_COMPLETE));
  if (spinCoordOffset)
    SDDS_ElegantOutputSetup(SDDS_table, filename, mode, lines_per_row, contents, command_file, lattice_file,
			    NULL, 0, spin_column, 3, caller, SDDS_EOS_COMPLETE);
  
  log_exit("SDDS_BeamLossSetup");
}

#define CENTROID_COLUMNS 11
#define CENTROID_COLUMNS_WITH_WEIGHTS 13
static SDDS_DEFINITION centroid_column[CENTROID_COLUMNS_WITH_WEIGHTS] = {
  {"Cx", "&column name=Cx, symbol=\"<x>\", units=m, type=double, description=\"x centroid\" &end"},
  {"Cxp", "&column name=Cxp, symbol=\"<x'>\", type=double, description=\"x' centroid\" &end"},
  {"Cy", "&column name=Cy, symbol=\"<y>\", units=m, type=double, description=\"y centroid\" &end"},
  {"Cyp", "&column name=Cyp, symbol=\"<y'>\", type=double, description=\"y' centroid\" &end"},
  {"Cs", "&column name=Cs, symbol=\"<s>\", units=m, type=double, description=\"mean distance traveled\" &end"},
  {"Cdelta", "&column name=Cdelta, symbol=\"<$gd$r>\", type=double, description=\"delta centroid\" &end"},
  {"Ct", "&column name=Ct, symbol=\"<t>\", units=s, type=double, description=\"mean arrival time\" &end"},
  {"Particles", "&column name=Particles, description=\"Number of particles\", type=long64 &end"},
  {"pCentral", "&column name=pCentral, symbol=\"p$bcen$n\", units=\"m$be$nc\", type=double, description=\"Reference beta*gamma\" &end"},
  {"Charge", "&column name=Charge, description=\"Charge in the beam\", units=C, type=double &end"},
  {"Pass", "&column name=Pass, type=long &end"},
  {"xBPMWeight", "&column name=xBPMWeight, description=\"Horizontal-plane BPM weight\", type=double &end"},
  {"yBPMWeight", "&column name=yBPMWeight, description=\"Vertical-plane BPM weight\", type=double &end"},
};

#define SPIN_CENTROID_COLUMNS 3
static SDDS_DEFINITION spin_centroid_column[SPIN_CENTROID_COLUMNS] = {
  {"Cspx", "&column name=Cspx, symbol=\"<Spinx>\", type=double, description=\"Mean x-direction spin component\" &end"},
  {"Cspy", "&column name=Cspy, symbol=\"<Spiny>\", type=double, description=\"Mean y-direction spin component\" &end"},
  {"Cspz", "&column name=Cspz, symbol=\"<Spinz>\", type=double, description=\"Mean z-direction spin component\" &end"},
};

void SDDS_CentroidOutputSetup(SDDS_TABLE *SDDS_table, char *filename, long mode, long lines_per_row, char *contents,
                              char *command_file, char *lattice_file, char *caller, short bpmsOnly) {
  log_entry("SDDS_CentroidOutputSetup");
#if USE_MPI
  if (myid != 0)
    return;
#  if SDDS_MPI_IO
  SDDS_table->parallel_io = 0;
#  endif
#endif

  SDDS_ElegantOutputSetup(SDDS_table, filename, mode, lines_per_row, contents, command_file, lattice_file,
                          standard_parameter, STANDARD_PARAMETERS, element_column, ELEMENT_COLUMNS,
                          caller, SDDS_EOS_NEWFILE);
  if (spinCoordOffset) {
    SDDS_ElegantOutputSetup(SDDS_table, filename, mode, lines_per_row, contents, command_file, lattice_file,
                            NULL, 0, spin_centroid_column, SPIN_CENTROID_COLUMNS,
                            caller, 0);
  }
  SDDS_ElegantOutputSetup(SDDS_table, NULL, 0, 0, NULL, NULL, NULL, NULL, 0,
                          centroid_column,
                          bpmsOnly?CENTROID_COLUMNS_WITH_WEIGHTS:CENTROID_COLUMNS, caller, SDDS_EOS_COMPLETE);
  log_exit("SDDS_CentroidOutputSetup");
}

#define SIGMA_MATRIX_COLUMNS 68
static SDDS_DEFINITION sigma_matrix_column[SIGMA_MATRIX_COLUMNS] = {
  {"s1", "&column name=s1, symbol=\"$gs$r$b1$n\", units=m, type=double, description=\"sqrt(<x*x>)\" &end"},
  {"s12", "&column name=s12, symbol=\"$gs$r$b12$n\", units=m, type=double, description=\"<x*xp'>\" &end"},
  {"s13", "&column name=s13, symbol=\"$gs$r$b13$n\", units=\"m$a2$n\", type=double, description=\"<x*y>\" &end"},
  {"s14", "&column name=s14, symbol=\"$gs$r$b14$n\", units=m, type=double, description=\"<x*y'>\" &end"},
  {"s15", "&column name=s15, symbol=\"$gs$r$b15$n\", units=\"m$a2$n\", type=double, description=\"<x*s>\" &end"},
  {"s16", "&column name=s16, symbol=\"$gs$r$b16$n\", units=m, type=double, description=\"<x*delta>\" &end"},
  {"s17", "&column name=s17, symbol=\"$gs$r$b17$n\", units=m*s, type=double, description=\"<x*t>\" &end"},
  {"s2", "&column name=s2, symbol=\"$gs$r$b2$n\", type=double, description=\"sqrt(<x'*x'>)\" &end"},
  {"s23", "&column name=s23, symbol=\"$gs$r$b23$n\", units=m, type=double, description=\"<x'*y>\" &end"},
  {"s24", "&column name=s24, symbol=\"$gs$r$b24$n\", type=double, description=\"<x'*y'>\" &end"},
  {"s25", "&column name=s25, symbol=\"$gs$r$b25$n\", units=m, type=double, description=\"<x'*s>\" &end"},
  {"s26", "&column name=s26, symbol=\"$gs$r$b26$n\", type=double, description=\"<x'*delta>\" &end"},
  {"s27", "&column name=s27, symbol=\"$gs$r$b27$n\", units=s, type=double, description=\"<x'*t>\" &end"},
  {"s3", "&column name=s3, symbol=\"$gs$r$b3$n\", units=m, type=double, description=\"sqrt(<y*y>)\" &end"},
  {"s34", "&column name=s34, symbol=\"$gs$r$b34$n\", units=m, type=double, description=\"<y*y'>\" &end"},
  {"s35", "&column name=s35, symbol=\"$gs$r$b35$n\", units=\"m$a2$n\", type=double, description=\"<y*s>\" &end"},
  {"s36", "&column name=s36, symbol=\"$gs$r$b36$n\", units=m, type=double, description=\"<y*delta>\" &end"},
  {"s37", "&column name=s37, symbol=\"$gs$r$b37$n\", units=m*s, type=double, description=\"<y*t>\" &end"},
  {"s4", "&column name=s4, symbol=\"$gs$r$b4$n\", type=double, description=\"sqrt(<y'*y')>\" &end"},
  {"s45", "&column name=s45, symbol=\"$gs$r$b45$n\", units=m, type=double, description=\"<y'*s>\" &end"},
  {"s46", "&column name=s46, symbol=\"$gs$r$b46$n\", type=double, description=\"<y'*delta>\" &end"},
  {"s47", "&column name=s47, symbol=\"$gs$r$b47$n\", units=s, type=double, description=\"<y'*t>\" &end"},
  {"s5", "&column name=s5, symbol=\"$gs$r$b5$n\", units=m, type=double, description=\"sqrt(<s*s>)\" &end"},
  {"s56", "&column name=s56, symbol=\"$gs$r$b56$n\", units=m, type=double, description=\"<s*delta>\" &end"},
  {"s57", "&column name=s57, symbol=\"$gs$r$b57$n\", units=m*s, type=double, description=\"<s*t>\" &end"},
  {"s6", "&column name=s6, symbol=\"$gs$r$b6$n\", type=double, description=\"sqrt(<delta*delta>)\" &end"},
  {"s67", "&column name=s67, symbol=\"$gs$r$b67$n\", units=s, type=double, description=\"<delta*t>\" &end"},
  {"s7", "&column name=s7, symbol=\"$gs$r$b7$n\", type=double, description=\"sqrt(<t*t>)\" &end"},
  {"ma1", "&column name=ma1, symbol=\"max$sb$ex$sb$e\", units=m, type=double, description=\"maximum absolute value of x\" &end"},
  {"ma2", "&column name=ma2, symbol=\"max$sb$ex'$sb$e\", type=double, description=\"maximum absolute value of x'\" &end"},
  {"ma3", "&column name=ma3, symbol=\"max$sb$ey$sb$e\", units=m, type=double, description=\"maximum absolute value of y\" &end"},
  {"ma4", "&column name=ma4, symbol=\"max$sb$ey'$sb$e\", type=double, description=\"maximum absolute value of y'\" &end"},
  {"ma5", "&column name=ma5, symbol=\"max$sb$e$gD$rs$sb$e\", type=double, units=m, description=\"maximum absolute deviation of s\" &end"},
  {"ma6", "&column name=ma6, symbol=\"max$sb$e$gd$r$sb$e\", type=double, description=\"maximum absolute value of delta\" &end"},
  {"ma7", "&column name=ma7, symbol=\"max$sb$e$gD$rt$sb$e\", units=s, type=double, description=\"maximum absolute deviation of t\" &end"},
  {"minimum1", "&column name=minimum1, symbol=\"x$bmin$n\", type=double, units=m &end"},
  {"minimum2", "&column name=minimum2, symbol=\"x'$bmin$n\", type=double, units=m &end"},
  {"minimum3", "&column name=minimum3, symbol=\"y$bmin$n\", type=double, units=m &end"},
  {"minimum4", "&column name=minimum4, symbol=\"y'$bmin$n\", type=double, units=m &end"},
  {"minimum5", "&column name=minimum5, symbol=\"$gD$rs$bmin$n\", type=double, units=m &end"},
  {"minimum6", "&column name=minimum6, symbol=\"$gd$r$bmin$n\", type=double, units=m &end"},
  {"minimum7", "&column name=minimum7, symbol=\"t$bmin$n\", type=double, units=s &end"},
  {"maximum1", "&column name=maximum1, symbol=\"x$bmax$n\", type=double, units=m &end"},
  {"maximum2", "&column name=maximum2, symbol=\"x'$bmax$n\", type=double, units=m &end"},
  {"maximum3", "&column name=maximum3, symbol=\"y$bmax$n\", type=double, units=m &end"},
  {"maximum4", "&column name=maximum4, symbol=\"y'$bmax$n\", type=double, units=m &end"},
  {"maximum5", "&column name=maximum5, symbol=\"$gD$rs$bmax$n\", type=double, units=m &end"},
  {"maximum6", "&column name=maximum6, symbol=\"$gd$r$bmax$n\", type=double, units=m &end"},
  {"maximum7", "&column name=maximum7, symbol=\"t$bmax$n\", type=double, units=s &end"},
  {"Sx", "&column name=Sx, symbol=\"$gs$r$bx$n\", units=m, type=double, description=\"sqrt(<(x-<x>)^2>)\" &end"},
  {"Sxp", "&column name=Sxp, symbol=\"$gs$r$bx'$n\", type=double, description=\"sqrt(<(x'-<x'>)^2>)\" &end"},
  {"Sy", "&column name=Sy, symbol=\"$gs$r$by$n\", units=m, type=double, description=\"sqrt(<(y-<y>)^2>)\" &end"},
  {"Syp", "&column name=Syp, symbol=\"$gs$r$by'$n\", type=double, description=\"sqrt(<(y'-<y'>)^2>)\" &end"},
  {"Ss", "&column name=Ss, units=m, symbol=\"$gs$r$bs$n\", type=double, description=\"sqrt(<(s-<s>)^2>)\" &end"},
  {"Sdelta", "&column name=Sdelta, symbol=\"$gs$bd$n$r\", type=double, description=\"sqrt(<(delta-<delta>)^2>)\" &end"},
  {"St", "&column name=St, units=s, symbol=\"$gs$r$bt$n\", type=double, description=\"sqrt(<(t-<t>)^2>)\" &end"},
  {"ex", "&column name=ex, symbol=\"$ge$r$bx$n\", units=m, type=double, description=\"geometric horizontal emittance\" &end"},
  {"enx", "&column name=enx, symbol=\"$ge$r$bx,n$n\", type=double, units=m, description=\"normalized horizontal emittance\"  &end"},
  {"ecx", "&column name=ecx, symbol=\"$ge$r$bx,c$n\", units=m, type=double, description=\"geometric horizontal emittance less dispersive contributions\" &end"},
  {"ecnx", "&column name=ecnx, symbol=\"$ge$r$bx,cn$n\", type=double, units=m, description=\"normalized horizontal emittance less dispersive contributions\" &end"},
  {"ey", "&column name=ey, symbol=\"$ge$r$by$n\", units=m, type=double, description=\"geometric vertical emittance\" &end"},
  {"eny", "&column name=eny, symbol=\"$ge$r$by,n$n\", type=double, units=m, description=\"normalized vertical emittance\" &end"},
  {"ecy", "&column name=ecy, symbol=\"$ge$r$by,c$n\", units=m, type=double, description=\"geometric vertical emittance less dispersive contributions\" &end"},
  {"ecny", "&column name=ecny, symbol=\"$ge$r$by,cn$n\", type=double, units=m, description=\"normalized vertical emittance less dispersive contributions\" &end"},
  {"betaxBeam", "&column name=betaxBeam, symbol=\"$gb$r$bx,beam$n\", type=double, units=m, description=\"betax for the beam, excluding dispersive contributions\" &end"},
  {"alphaxBeam", "&column name=alphaxBeam, symbol=\"$ga$r$bx,beam$n\", type=double, description=\"alphax for the beam, excluding dispersive contributions\" &end"},
  {"betayBeam", "&column name=betayBeam, symbol=\"$gb$r$by,beam$n\", type=double, units=m, description=\"betay for the beam, excluding dispersive contributions\" &end"},
  {"alphayBeam", "&column name=alphayBeam, symbol=\"$ga$r$by,beam$n\", type=double, description=\"alphay for the beam, excluding dispersive contributions\" &end"},
};

#define SIGMA_MATRIX_SPIN_COLUMNS 6
static SDDS_DEFINITION sigma_matrix_spin_column[SIGMA_MATRIX_SPIN_COLUMNS] = {
  {"Sspxx", "&column name=Sspxx, symbol=\"$gs$r$bxx$n\", units=m, type=double, description=\"Spin sigma matrix element x-x\" &end"},
  {"Sspxy", "&column name=Sspxy, symbol=\"$gs$r$bxy$n\", units=m, type=double, description=\"Spin sigma matrix element x-y\" &end"},
  {"Sspxz", "&column name=Sspxz, symbol=\"$gs$r$bxz$n\", units=m, type=double, description=\"Spin sigma matrix element x-z\" &end"},
  {"Sspyy", "&column name=Sspyy, symbol=\"$gs$r$byy$n\", units=m, type=double, description=\"Spin sigma matrix element y-y\" &end"},
  {"Sspyz", "&column name=Sspyz, symbol=\"$gs$r$byz$n\", units=m, type=double, description=\"Spin sigma matrix element y-z\" &end"},
  {"Sspzz", "&column name=Sspzz, symbol=\"$gs$r$bzz$n\", units=m, type=double, description=\"Spin sigma matrix element z-z\" &end"},
};

/* Coupled (eigen-)emittances and coupled lattice functions of the full 6x6 beam
   matrix (Sigma.J / Wolski method), appended to the sigma file when the run_setup
   coupled_sigma flag is set.  Column names, symbols, and units match the -coupled
   output of sddsanalyzebeam (elegantTools/sddsanalyzebeam.c).  Three modes m=1,2,3:
   mode 1 is x-dominated, 2 is y-dominated, 3 is longitudinal-dominated. */
#define COUPLED_SIGMA_COLUMNS 45
static SDDS_DEFINITION coupled_sigma_column[COUPLED_SIGMA_COLUMNS] = {
  {"e1", "&column name=e1, symbol=\"$ge$r$b1$n\", units=m, type=double, description=\"coupled eigen-emittance, mode 1\" &end"},
  {"en1", "&column name=en1, symbol=\"$ge$r$bn1$n\", units=m, type=double, description=\"normalized coupled eigen-emittance, mode 1\" &end"},
  {"e2", "&column name=e2, symbol=\"$ge$r$b2$n\", units=m, type=double, description=\"coupled eigen-emittance, mode 2\" &end"},
  {"en2", "&column name=en2, symbol=\"$ge$r$bn2$n\", units=m, type=double, description=\"normalized coupled eigen-emittance, mode 2\" &end"},
  {"e3", "&column name=e3, symbol=\"$ge$r$b3$n\", units=m, type=double, description=\"coupled eigen-emittance, mode 3\" &end"},
  {"en3", "&column name=en3, symbol=\"$ge$r$bn3$n\", units=m, type=double, description=\"normalized coupled eigen-emittance, mode 3\" &end"},
  {"betax1", "&column name=betax1, symbol=\"$gb$r$bx,1$n\", units=m, type=double, description=\"coupled beta_x, mode 1\" &end"},
  {"betay1", "&column name=betay1, symbol=\"$gb$r$by,1$n\", units=m, type=double, description=\"coupled beta_y, mode 1\" &end"},
  {"betaz1", "&column name=betaz1, symbol=\"$gb$r$bs,1$n\", units=m, type=double, description=\"coupled beta_s, mode 1\" &end"},
  {"alphax1", "&column name=alphax1, symbol=\"$ga$r$bx,1$n\", type=double, description=\"coupled alpha_x, mode 1\" &end"},
  {"alphay1", "&column name=alphay1, symbol=\"$ga$r$by,1$n\", type=double, description=\"coupled alpha_y, mode 1\" &end"},
  {"alphaz1", "&column name=alphaz1, symbol=\"$ga$r$bs,1$n\", type=double, description=\"coupled alpha_s, mode 1\" &end"},
  {"gammax1", "&column name=gammax1, symbol=\"$gg$r$bx,1$n\", units=1/m, type=double, description=\"coupled gamma_x, mode 1\" &end"},
  {"gammay1", "&column name=gammay1, symbol=\"$gg$r$by,1$n\", units=1/m, type=double, description=\"coupled gamma_y, mode 1\" &end"},
  {"gammaz1", "&column name=gammaz1, symbol=\"$gg$r$bs,1$n\", units=1/m, type=double, description=\"coupled gamma_s, mode 1\" &end"},
  {"betax2", "&column name=betax2, symbol=\"$gb$r$bx,2$n\", units=m, type=double, description=\"coupled beta_x, mode 2\" &end"},
  {"betay2", "&column name=betay2, symbol=\"$gb$r$by,2$n\", units=m, type=double, description=\"coupled beta_y, mode 2\" &end"},
  {"betaz2", "&column name=betaz2, symbol=\"$gb$r$bs,2$n\", units=m, type=double, description=\"coupled beta_s, mode 2\" &end"},
  {"alphax2", "&column name=alphax2, symbol=\"$ga$r$bx,2$n\", type=double, description=\"coupled alpha_x, mode 2\" &end"},
  {"alphay2", "&column name=alphay2, symbol=\"$ga$r$by,2$n\", type=double, description=\"coupled alpha_y, mode 2\" &end"},
  {"alphaz2", "&column name=alphaz2, symbol=\"$ga$r$bs,2$n\", type=double, description=\"coupled alpha_s, mode 2\" &end"},
  {"gammax2", "&column name=gammax2, symbol=\"$gg$r$bx,2$n\", units=1/m, type=double, description=\"coupled gamma_x, mode 2\" &end"},
  {"gammay2", "&column name=gammay2, symbol=\"$gg$r$by,2$n\", units=1/m, type=double, description=\"coupled gamma_y, mode 2\" &end"},
  {"gammaz2", "&column name=gammaz2, symbol=\"$gg$r$bs,2$n\", units=1/m, type=double, description=\"coupled gamma_s, mode 2\" &end"},
  {"betax3", "&column name=betax3, symbol=\"$gb$r$bx,3$n\", units=m, type=double, description=\"coupled beta_x, mode 3\" &end"},
  {"betay3", "&column name=betay3, symbol=\"$gb$r$by,3$n\", units=m, type=double, description=\"coupled beta_y, mode 3\" &end"},
  {"betaz3", "&column name=betaz3, symbol=\"$gb$r$bs,3$n\", units=m, type=double, description=\"coupled beta_s, mode 3\" &end"},
  {"alphax3", "&column name=alphax3, symbol=\"$ga$r$bx,3$n\", type=double, description=\"coupled alpha_x, mode 3\" &end"},
  {"alphay3", "&column name=alphay3, symbol=\"$ga$r$by,3$n\", type=double, description=\"coupled alpha_y, mode 3\" &end"},
  {"alphaz3", "&column name=alphaz3, symbol=\"$ga$r$bs,3$n\", type=double, description=\"coupled alpha_s, mode 3\" &end"},
  {"gammax3", "&column name=gammax3, symbol=\"$gg$r$bx,3$n\", units=1/m, type=double, description=\"coupled gamma_x, mode 3\" &end"},
  {"gammay3", "&column name=gammay3, symbol=\"$gg$r$by,3$n\", units=1/m, type=double, description=\"coupled gamma_y, mode 3\" &end"},
  {"gammaz3", "&column name=gammaz3, symbol=\"$gg$r$bs,3$n\", units=1/m, type=double, description=\"coupled gamma_s, mode 3\" &end"},
  {"A_xy_1", "&column name=A_xy_1, units=m, type=double, description=\"x-y coupling term, mode 1\" &end"},
  {"A_xpy_1", "&column name=A_xpy_1, type=double, description=\"x'-y coupling term, mode 1\" &end"},
  {"A_xyp_1", "&column name=A_xyp_1, type=double, description=\"x-y' coupling term, mode 1\" &end"},
  {"A_xpyp_1", "&column name=A_xpyp_1, units=1/m, type=double, description=\"x'-y' coupling term, mode 1\" &end"},
  {"A_xy_2", "&column name=A_xy_2, units=m, type=double, description=\"x-y coupling term, mode 2\" &end"},
  {"A_xpy_2", "&column name=A_xpy_2, type=double, description=\"x'-y coupling term, mode 2\" &end"},
  {"A_xyp_2", "&column name=A_xyp_2, type=double, description=\"x-y' coupling term, mode 2\" &end"},
  {"A_xpyp_2", "&column name=A_xpyp_2, units=1/m, type=double, description=\"x'-y' coupling term, mode 2\" &end"},
  {"A_xy_3", "&column name=A_xy_3, units=m, type=double, description=\"x-y coupling term, mode 3\" &end"},
  {"A_xpy_3", "&column name=A_xpy_3, type=double, description=\"x'-y coupling term, mode 3\" &end"},
  {"A_xyp_3", "&column name=A_xyp_3, type=double, description=\"x-y' coupling term, mode 3\" &end"},
  {"A_xpyp_3", "&column name=A_xpyp_3, units=1/m, type=double, description=\"x'-y' coupling term, mode 3\" &end"},
};

void SDDS_SigmaOutputSetup(SDDS_TABLE *SDDS_table, char *filename, long mode, long lines_per_row,
                           char *command_file, char *lattice_file, char *caller, short coupledSigmaOutput) {
  log_entry("SDDS_SigmaOutputSetup");
#if USE_MPI
  if (myid < 0)
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (myid != 0)
    return;
#endif

  SDDS_ElegantOutputSetup(SDDS_table, filename, mode, lines_per_row, "sigma matrix", command_file, lattice_file,
                          standard_parameter, STANDARD_PARAMETERS, element_column, ELEMENT_COLUMNS,
                          caller, SDDS_EOS_NEWFILE);
  SDDS_ElegantOutputSetup(SDDS_table, filename, mode, lines_per_row, NULL, NULL, NULL,
                          NULL, 0, sigma_matrix_column, SIGMA_MATRIX_COLUMNS,
                          caller, (spinCoordOffset||coupledSigmaOutput)?0:SDDS_EOS_COMPLETE);
  if (spinCoordOffset)
    SDDS_ElegantOutputSetup(SDDS_table, filename, mode, lines_per_row, "sigma matrix", command_file, lattice_file,
                            NULL, 0, sigma_matrix_spin_column, SIGMA_MATRIX_SPIN_COLUMNS,
                            caller, coupledSigmaOutput?0:SDDS_EOS_COMPLETE);
  if (coupledSigmaOutput)
    SDDS_ElegantOutputSetup(SDDS_table, filename, mode, lines_per_row, NULL, NULL, NULL,
                            NULL, 0, coupled_sigma_column, COUPLED_SIGMA_COLUMNS,
                            caller, SDDS_EOS_COMPLETE);
  log_exit("SDDS_SigmaOutputSetup");
}

#define WATCH_CENTROID_MODE_COLUMNS 19
static SDDS_DEFINITION watch_centroid_mode_column[WATCH_CENTROID_MODE_COLUMNS+3] = {
  {"Step", "&column name=Step, type=long &end"},
  {"Pass", "&column name=Pass, type=long &end"},
  {"ElapsedTime", "&column name=ElapsedTime, type=double units=s &end"},
  {"ElapsedCoreTime", "&column name=ElapsedCoreTime, type=double units=s &end"},
  {"MemoryUsage", "&column name=MemoryUsage, type=long, units=kB &end"},
  {"Cx", "&column name=Cx, symbol=\"<x>\", units=m, type=double, description=\"x centroid\" &end"},
  {"Cxp", "&column name=Cxp, symbol=\"<x'>\", type=double, description=\"x' centroid\" &end"},
  {"Cy", "&column name=Cy, symbol=\"<y>\", units=m, type=double, description=\"y centroid\" &end"},
  {"Cyp", "&column name=Cyp, symbol=\"<y'>\", type=double, description=\"y' centroid\" &end"},
  {"Cs", "&column name=Cs, symbol=\"<s>\", units=m, type=double, description=\"mean distance traveled\" &end"},
  {"Cdelta", "&column name=Cdelta, symbol=\"<$gd$r>\", type=double, description=\"delta centroid\" &end"},
  {"Ct", "&column name=Ct, symbol=\"<t>\", units=s, type=double, description=\"mean time of flight\" &end"},
  {"dCt", "&column name=dCt, symbol=\"$gD$r<t>\", units=s, type=double, description=\"mean time of flight relative to ideal\" &end"},
  {"Particles", "&column name=Particles, description=\"Number of particles\", type=long64, &end"},
  {"Charge", "&column name=Charge, description=\"Charge in the beam\", units=C, type=double &end"},
  {"Transmission", "&column name=Transmission, description=Transmission, type=double &end"},
  {"pCentral", "&column name=pCentral, symbol=\"p$bcen$n\", units=\"m$be$nc\", type=double, description=\"Reference beta*gamma\" &end"},
  {"pAverage", "&column name=pAverage, symbol=\"p$bave$n\", units=\"m$be$nc\", type=double, description=\"Mean beta*gamma\" &end"},
  {"KAverage", "&column name=KAverage, symbol=\"K$bave$n\", units=MeV, type=double, description=\"Mean kinetic energy\" &end"},
  {"Cspx", "&column name=Cspx, symbol=\"<Spinx>\", type=double, description=\"Mean x-direction spin component\" &end"},
  {"Cspy", "&column name=Cspy, symbol=\"<Spiny>\", type=double, description=\"Mean y-direction spin component\" &end"},
  {"Cspz", "&column name=Cspz, symbol=\"<Spinz>\", type=double, description=\"Mean z-direction spin component\" &end"},
};

#define WATCH_PARAMETER_MODE_COLUMNS 31
static SDDS_DEFINITION watch_parameter_mode_column[WATCH_PARAMETER_MODE_COLUMNS+9] = {
  {"Step", "&column name=Step, type=long &end"},
  {"Pass", "&column name=Pass, type=long &end"},
  {"ElapsedTime", "&column name=ElapsedTime, type=double units=s &end"},
  {"ElapsedCoreTime", "&column name=ElapsedCoreTime, type=double units=s &end"},
  {"MemoryUsage", "&column name=MemoryUsage, type=long, units=kB &end"},
  {"Cx", "&column name=Cx, symbol=\"<x>\", units=m, type=double, description=\"x centroid\" &end"},
  {"Cxp", "&column name=Cxp, symbol=\"<x'>\", type=double, description=\"x' centroid\" &end"},
  {"Cy", "&column name=Cy, symbol=\"<y>\", units=m, type=double, description=\"y centroid\" &end"},
  {"Cyp", "&column name=Cyp, symbol=\"<y'>\", type=double, description=\"y' centroid\" &end"},
  {"Cs", "&column name=Cs, symbol=\"<s>\", units=m, type=double, description=\"mean distance traveled\" &end"},
  {"Cdelta", "&column name=Cdelta, symbol=\"<$gd$r>\", type=double, description=\"delta centroid\" &end"},
  {"Ct", "&column name=Ct, symbol=\"<t>\", units=s, type=double, description=\"mean time of flight\" &end"},
  {"dCt", "&column name=dCt, symbol=\"$gD$r<t>\", units=s, type=double, description=\"mean time of flight relative to ideal\" &end"},
  {"Particles", "&column name=Particles, description=\"Number of particles\", type=long64, &end"},
  {"Charge", "&column name=Charge, description=\"Charge in the beam\", units=C, type=double &end"},
  {"Transmission", "&column name=Transmission, description=Transmission, type=double &end"},
  {"pCentral", "&column name=pCentral, symbol=\"p$bcen$n\", units=\"m$be$nc\", type=double, description=\"Reference beta*gamma\" &end"},
  {"pAverage", "&column name=pAverage, symbol=\"p$bave$n\", units=\"m$be$nc\", type=double, description=\"Mean beta*gamma\" &end"},
  {"KAverage", "&column name=KAverage, symbol=\"K$bave$n\", units=MeV, type=double, description=\"Mean kinetic energy\" &end"},
  {"Sx", "&column name=Sx, symbol=\"$gs$r$bx$n\", units=m, type=double, description=\"sqrt(<(x-<x>)^2>)\" &end"},
  {"Sxp", "&column name=Sxp, symbol=\"$gs$r$bx'$n\", type=double, description=\"sqrt(<(x'-<x'>)^2>)\" &end"},
  {"Sy", "&column name=Sy, symbol=\"$gs$r$by$n\", units=m, type=double, description=\"sqrt(<(y-<y>)^2>)\" &end"},
  {"Syp", "&column name=Syp, symbol=\"$gs$r$by'$n\", type=double, description=\"sqrt(<(y'-<y'>)^2>)\" &end"},
  {"Ss", "&column name=Ss, units=m, symbol=\"$gs$r$bs$n\", type=double, description=\"sqrt(<(s-<s>)^2>)\" &end"},
  {"Sdelta", "&column name=Sdelta, symbol=\"$gs$bd$n$r\", type=double, description=\"sqrt(<(delta-<delta>)^2>)\" &end"},
  {"St", "&column name=St, symbol=\"$gs$r$bt$n\", units=s, type=double, description=\"sqrt(<(t-<t>)^2>)\" &end"},
  {"ex", "&column name=ex, symbol=\"$ge$r$bx$n\", units=m, type=double, description=\"geometric horizontal emittance\" &end"},
  {"ey", "&column name=ey, symbol=\"$ge$r$by$n\", units=m, type=double, description=\"geometric vertical emittance\" &end"},
  {"el", "&column name=el, symbol=\"$ge$r$bl$n\", units=s, type=double, description=\"longitudinal emittance\" &end"},
  {"ecx", "&column name=ecx, symbol=\"$ge$r$bx,c$n\", units=m, type=double, description=\"geometric horizontal emittance less dispersive contributions\" &end"},
  {"ecy", "&column name=ecy, symbol=\"$ge$r$by,c$n\", units=m, type=double, description=\"geometric vertical emittance less dispersive contributions\" &end"},
  {"Cspx", "&column name=Cspx, symbol=\"<Spinx>\", type=double, description=\"Mean x-direction spin component\" &end"},
  {"Cspy", "&column name=Cspy, symbol=\"<Spiny>\", type=double, description=\"Mean y-direction spin component\" &end"},
  {"Cspz", "&column name=Cspz, symbol=\"<Spinz>\", type=double, description=\"Mean z-direction spin component\" &end"},
  {"Sspxx", "&column name=Sspxx, symbol=\"$gs$r$bxx$n\", units=m, type=double, description=\"Spin sigma matrix element x-x\" &end"},
  {"Sspxy", "&column name=Sspxy, symbol=\"$gs$r$bxy$n\", units=m, type=double, description=\"Spin sigma matrix element x-y\" &end"},
  {"Sspxz", "&column name=Sspxz, symbol=\"$gs$r$bxz$n\", units=m, type=double, description=\"Spin sigma matrix element x-z\" &end"},
  {"Sspyy", "&column name=Sspyy, symbol=\"$gs$r$byy$n\", units=m, type=double, description=\"Spin sigma matrix element y-y\" &end"},
  {"Sspyz", "&column name=Sspyz, symbol=\"$gs$r$byz$n\", units=m, type=double, description=\"Spin sigma matrix element y-z\" &end"},
  {"Sspzz", "&column name=Sspzz, symbol=\"$gs$r$bzz$n\", units=m, type=double, description=\"Spin sigma matrix element z-z\" &end"},
};

#define WATCH_POINT_FFT_COLUMNS 4
static SDDS_DEFINITION watch_point_fft_column[WATCH_POINT_FFT_COLUMNS] = {
  {"f", "&column name=f, symbol=\"f/f$bsample$n\", description=\"Normalized frequency\", type=double &end"},
  {"Cx", "&column name=Cx, symbol=\"FFT <x>\", units=m, type=double, description=\"FFT of x centroid\" &end"},
  {"Cy", "&column name=Cy, symbol=\"FFT <y>\", units=m, type=double, description=\"FFT of y centroid\" &end"},
  {"Cdelta", "&column name=Cdelta, symbol=\"FFT <$gd$r>\", type=double, description=\"FFT of delta centroid\" &end"},
};

void SDDS_WatchPointSetup(WATCH *watch, char *filename, long mode, long lines_per_row,
                          char *command_file, char *lattice_file, char *caller, char *qualifier,
                          char *previousElementName, long previousElementOccurence) {
  char s[100];
  SDDS_TABLE *SDDS_table;
  long watch_mode;
  char buffer[16384];

#if USE_MPI && !SDDS_MPI_IO
  if (myid < 0)
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (myid != 0)
    return;
#endif

  if (!watch->SDDS_table)
    watch->SDDS_table = tmalloc(sizeof(*(watch->SDDS_table)));
  SDDS_table = watch->SDDS_table;
  watch_mode = watch->mode_code;

#if SDDS_MPI_IO
  SDDS_table->parallel_io = 0; /* By default, I/O will be done in serial */
#endif
  switch (watch_mode) {
  case WATCH_COORDINATES:
#if SDDS_MPI_IO
    SDDS_table->parallel_io = 1;
    SDDS_MPI_Setup(SDDS_table, 1, n_processors, myid, MPI_COMM_WORLD, 0);
#endif
    SDDS_ElegantOutputSetup(SDDS_table, filename, mode, lines_per_row, "watch-point phase space",
                            command_file, lattice_file,
                            phase_space_parameter, PHASE_SPACE_PARAMETERS,
                            NULL, 0, caller, SDDS_EOS_NEWFILE);
#if SDDS_MPI_IO
    /* Open file here for parallel IO */
    if (!SDDS_MPI_File_Open(SDDS_table->MPI_dataset, SDDS_table->layout.filename, SDDS_MPI_WRITE_ONLY))
      SDDS_MPI_BOMB("SDDS_MPI_File_Open failed.", &SDDS_table->MPI_dataset->MPI_file);
#endif
    if (watch->xData) {
      if ((watch->xIndex[0] = SDDS_DefineColumn(SDDS_table, "x", NULL, "m", NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          (!watch->excludeSlopes &&
           (watch->xIndex[1] = SDDS_DefineColumn(SDDS_table, "xp", NULL, NULL, NULL, NULL, SDDS_DOUBLE, 0)) < 0)) {
        printf("Unable to define SDDS columns x and/or xp for file %s (%s)\n",
               filename, caller);
        fflush(stdout);
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        exitElegant(1);
      }
    }
    if (watch->yData) {
      if ((watch->yIndex[0] = SDDS_DefineColumn(SDDS_table, "y", NULL, "m", NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          (!watch->excludeSlopes &&
           (watch->yIndex[1] = SDDS_DefineColumn(SDDS_table, "yp", NULL, NULL, NULL, NULL, SDDS_DOUBLE, 0)) < 0)) {
        printf("Unable to define SDDS columns y and yp for file %s (%s)\n",
               filename, caller);
        fflush(stdout);
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        exitElegant(1);
      }
    }
    if (watch->longitData) {
      if ((watch->longitIndex[0] = SDDS_DefineColumn(SDDS_table, "t", NULL, "s", NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          (watch->longitIndex[1] = SDDS_DefineColumn(SDDS_table, "p", NULL, "m$be$nc", NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          (watch->longitIndex[2] = SDDS_DefineColumn(SDDS_table, "dt", NULL, "s", NULL, NULL, SDDS_DOUBLE, 0)) < 0) {
        printf("Unable to define SDDS columns t, dt, and p for file %s (%s)\n",
               filename, caller);
        fflush(stdout);
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        exitElegant(1);
      }
    }
    if (spinCoordOffset) {
      if ((watch->spinIndex[0] = SDDS_DefineColumn(SDDS_table, "spx", NULL, NULL, NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          (watch->spinIndex[1] = SDDS_DefineColumn(SDDS_table, "spy", NULL, NULL, NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          (watch->spinIndex[2] = SDDS_DefineColumn(SDDS_table, "spz", NULL, NULL, NULL, NULL, SDDS_DOUBLE, 0)) < 0) {
        printf("Unable to define SDDS columns spx, spy, and spz for file %s (%s)\n",
               filename, caller);
        fflush(stdout);
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        exitElegant(1);
      }
    }
    if ((watch->IDIndex = SDDS_DefineColumn(SDDS_table, "particleID", NULL, NULL, NULL, NULL, SDDS_ULONG64, 0)) < 0) {
      printf("Unable to define SDDS columns t, dt, and p for file %s (%s)\n",
             filename, caller);
      fflush(stdout);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }

    if (!SDDS_DefineSimpleParameter(SDDS_table, "Pass", NULL, SDDS_LONG) ||
        !SDDS_DefineSimpleParameter(SDDS_table, "PassLength", "m", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(SDDS_table, "PassCentralTime", "s", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(SDDS_table, "ElapsedTime", "s", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(SDDS_table, "ElapsedCoreTime", "s", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(SDDS_table, "MemoryUsage", "kB", SDDS_LONG) ||
        !SDDS_DefineSimpleParameter(SDDS_table, "s", "m", SDDS_DOUBLE)) {
      printf("Unable define SDDS parameter for file %s (%s)\n", filename, caller);
      fflush(stdout);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
    if (SDDS_DefineParameter(SDDS_table, "Description", NULL, NULL, NULL, "%s", SDDS_STRING, watch->label) < 0) {
      printf("Unable define SDDS parameter for file %s (%s)\n", filename, caller);
      fflush(stdout);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
    if (SDDS_DefineParameter(SDDS_table, "PreviousElementName", NULL, NULL, NULL, "%s", SDDS_STRING,
                             previousElementName ? previousElementName : "_BEG_") < 0) {
      printf("Unable define SDDS parameter for file %s (%s)\n", filename, caller);
      fflush(stdout);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
    snprintf(buffer, 16384, "%ld", previousElementOccurence);
    if (SDDS_DefineParameter(SDDS_table, "PreviousElementOccurence", NULL, NULL, NULL, "%ld", SDDS_LONG, buffer)<0) {
      printf("Unable define SDDS parameter for file %s (%s)\n", filename, caller);
      fflush(stdout);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
    snprintf(buffer, 16384, "%s#%ld", previousElementName?previousElementName:"_BEG_", previousElementOccurence);
    if (SDDS_DefineParameter(SDDS_table, "PreviousElementTag", NULL, NULL, NULL, "%s", SDDS_STRING, buffer)<0) {
      printf("Unable define SDDS parameter for file %s (%s)\n", filename, caller);
      fflush(stdout);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
    
#if SDDS_MPI_IO
    if (!SDDS_MPI_WriteLayout(SDDS_table))
#else
    if (!SDDS_WriteLayout(SDDS_table))
#endif
    {
      printf("Unable to write SDDS layout for file %s (%s)\n", filename, caller);
      fflush(stdout);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
    if ((watch->useDisconnect) && (!SDDS_DisconnectFile(SDDS_table)))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    break;
  case WATCH_CENTROIDS:
    if (isMaster)
      SDDS_ElegantOutputSetup(SDDS_table, filename, mode, lines_per_row, "watch-point centroids",
                              command_file, lattice_file,
                              standard_parameter, STANDARD_PARAMETERS,
                              watch_centroid_mode_column, WATCH_CENTROID_MODE_COLUMNS+(spinCoordOffset>0?3:0),
                              caller, SDDS_EOS_NEWFILE);
    if (!SDDS_DefineSimpleParameter(SDDS_table, "s", "m", SDDS_DOUBLE) ||
        SDDS_DefineParameter(SDDS_table, "PreviousElementName", NULL, NULL, NULL, "%s", SDDS_STRING,
                             previousElementName ? previousElementName : "_BEG_") < 0) {
      printf("Unable define SDDS parameter for file %s (%s)\n", filename, caller);
      fflush(stdout);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
    if (isMaster)
      if (!SDDS_WriteLayout(SDDS_table)) {
        printf("Unable to write SDDS layout for file %s (%s)\n", filename, caller);
        fflush(stdout);
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        exitElegant(1);
      }
    break;
  case WATCH_PARAMETERS:
    if (isMaster)
      SDDS_ElegantOutputSetup(SDDS_table, filename, mode, lines_per_row, "watch-point parameters",
                              command_file, lattice_file,
                              standard_parameter, STANDARD_PARAMETERS,
                              watch_parameter_mode_column, WATCH_PARAMETER_MODE_COLUMNS+(spinCoordOffset?9:0),
                              caller, SDDS_EOS_NEWFILE);
    if (!SDDS_DefineSimpleParameter(SDDS_table, "s", "m", SDDS_DOUBLE) ||
        SDDS_DefineParameter(SDDS_table, "PreviousElementName", NULL, NULL, NULL, "%s", SDDS_STRING,
                             previousElementName ? previousElementName : "_BEG_") < 0) {
      printf("Unable define SDDS parameter for file %s (%s)\n", filename, caller);
      fflush(stdout);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
    if (isMaster)
      if (!SDDS_WriteLayout(SDDS_table)) {
        printf("Unable to write SDDS layout for file %s (%s)\n", filename, caller);
        fflush(stdout);
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        exitElegant(1);
      }
    break;
  case WATCH_FFT:
    if (isMaster) {
      if (!qualifier)
        watch->window_code = FFT_HANNING;
      else if ((watch->window_code = match_string(qualifier, fft_window_name, N_FFT_WINDOWS, 0)) < 0)
        bombElegant("invalid window mode for WATCH fft", NULL);
      sprintf(s, "watch-point centroid FFT (%s window)", fft_window_name[watch->window_code]);
      SDDS_ElegantOutputSetup(SDDS_table, filename, mode, lines_per_row, s,
                              command_file, lattice_file, standard_parameter, STANDARD_PARAMETERS,
                              watch_point_fft_column, WATCH_POINT_FFT_COLUMNS,
                              caller, SDDS_EOS_NEWFILE | SDDS_EOS_COMPLETE);
    }
    break;
  default:
    break;
  }
}

void SDDS_HistogramSetup(HISTOGRAM *histogram, char *filename, long mode, long lines_per_row,
                         char *command_file, char *lattice_file, char *caller) {
  SDDS_TABLE *SDDS_table = NULL;
  long column, columns;

  if (histogram->sparse &&
      ((histogram->longitData ? 1 : 0) + (histogram->xData ? 1 : 0) + (histogram->yData ? 1 : 0)) > 1)
    bombElegant("When using HISTOGRAM in SPARSE mode, only one of LONGIT_DATA, X_DATA, or Y_DATA may be requested.", NULL);

  if (isMaster) {
    if (!histogram->SDDS_table)
      histogram->SDDS_table = tmalloc(sizeof(*(histogram->SDDS_table)));
    SDDS_table = histogram->SDDS_table;
    SDDS_ElegantOutputSetup(SDDS_table, filename, mode, lines_per_row, "histograms of phase-space coordinates",
                            command_file, lattice_file,
                            phase_space_parameter, PHASE_SPACE_PARAMETERS,
                            NULL, 0, caller, SDDS_EOS_NEWFILE);
  }
  for (column = 0; column < 7; column++)
    histogram->columnIndex[column][0] =
      histogram->columnIndex[column][1] = -1;

  if (isMaster) {
    columns = 0;
    if (histogram->xData) {
      if ((histogram->columnIndex[0][0] = SDDS_DefineColumn(SDDS_table, "x", NULL, "m", NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          (histogram->columnIndex[0][1] = SDDS_DefineColumn(SDDS_table, "xFrequency", NULL,
                                                            histogram->normalize ? "1/m" : "Particles/Bin", NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          (!histogram->sparse &&
           ((histogram->columnIndex[1][0] = SDDS_DefineColumn(SDDS_table, "xp", NULL, NULL, NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
            (histogram->columnIndex[1][1] = SDDS_DefineColumn(SDDS_table, "xpFrequency", NULL,
                                                              histogram->normalize ? NULL : "Particles/Bin", NULL, NULL, SDDS_DOUBLE, 0)) < 0))) {
        printf("Unable to define SDDS columns x and xp for file %s (%s)\n",
               filename, caller);
        fflush(stdout);
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        exit(1);
      }
      columns += histogram->sparse ? 2 : 4;
    }
    if (histogram->yData) {
      if ((histogram->columnIndex[2][0] = SDDS_DefineColumn(SDDS_table, "y", NULL, "m", NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          (histogram->columnIndex[2][1] = SDDS_DefineColumn(SDDS_table, "yFrequency", NULL,
                                                            histogram->normalize ? "1/m" : "Particles/Bin", NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          (!histogram->sparse &&
           ((histogram->columnIndex[3][0] = SDDS_DefineColumn(SDDS_table, "yp", NULL, NULL, NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
            (histogram->columnIndex[3][1] = SDDS_DefineColumn(SDDS_table, "ypFrequency", NULL,
                                                              histogram->normalize ? NULL : "Particles/Bin", NULL, NULL, SDDS_DOUBLE, 0)) < 0))) {
        printf("Unable to define SDDS columns y and yp for file %s (%s)\n",
               filename, caller);
        fflush(stdout);
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        exit(1);
      }
      columns += histogram->sparse ? 2 : 4;
    }
    if (histogram->longitData) {
      if ((histogram->columnIndex[4][0] = SDDS_DefineColumn(SDDS_table, "t", NULL, "s", NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          (histogram->columnIndex[4][1] = SDDS_DefineColumn(SDDS_table, "tFrequency", NULL,
                                                            histogram->normalize ? "1/s" : "Particles/Bin", NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          (!histogram->sparse &&
           ((histogram->columnIndex[5][0] = SDDS_DefineColumn(SDDS_table, "delta", NULL, NULL, NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
            (histogram->columnIndex[5][1] = SDDS_DefineColumn(SDDS_table, "deltaFrequency", NULL,
                                                              histogram->normalize ? NULL : "Particles/Bin", NULL, NULL, SDDS_DOUBLE, 0)) < 0)) ||
          (histogram->columnIndex[6][0] = SDDS_DefineColumn(SDDS_table, "dt", NULL, "s", NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          (histogram->columnIndex[6][1] = SDDS_DefineColumn(SDDS_table, "dtFrequency", NULL,
                                                            histogram->normalize ? "1/s" : "Particles/Bin", NULL, NULL, SDDS_DOUBLE, 0)) < 0) {
        printf("Unable to define SDDS columns t, dt, and p for file %s (%s)\n",
               filename, caller);
        fflush(stdout);
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        exit(1);
      }
      columns += histogram->sparse ? 4 : 6;
    }
    if (!columns)
      bombElegant("no output selected for histogram", NULL);
    if (!SDDS_DefineSimpleParameter(SDDS_table, "Pass", NULL, SDDS_LONG) ||
        !SDDS_DefineSimpleParameter(SDDS_table, "PassLength", "m", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(SDDS_table, "PassCentralTime", "s", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(SDDS_table, "s", "m", SDDS_DOUBLE)) {
      printf("Unable define SDDS parameter for file %s (%s)\n", filename, caller);
      fflush(stdout);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
    if (!SDDS_WriteLayout(SDDS_table)) {
      printf("Unable to write SDDS layout for file %s (%s)\n", filename, caller);
      fflush(stdout);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
  }
#if USE_MPI
  /* Broadcast the column index information to all the slaves */
  MPI_Bcast(&histogram->columnIndex[0][0], 14, MPI_LONG, 0, MPI_COMM_WORLD);
#endif
}

void dump_watch_particles(WATCH *watch, long step, long pass, double **particle, int64_t particles,
                          double Po, double length, double mp_charge, double z, int64_t slotsPerBunch) {
  int64_t i, row, count;
  double p, t0, t0Error, t;
  long memoryUsed;
#if SDDS_MPI_IO
  int64_t total_row = 0, total_count;
#endif

  log_entry("dump_watch_particles");

  if (pass <= watch->passLast) {
    watch->t0Last = z * sqrt(Po * Po + 1) / (c_mks * (Po + 1e-32));
    watch->t0LastError = 0;
  }
  t0 = watch->t0Last;
  t0Error = watch->t0LastError;
  /* This code mimics what happens to particles as they get ~T0 added on each turn with accumulating
   * round-off error. Prevents dCt from walking off too much.
   */
  for (i = 0; i < pass - watch->passLast; i++) {
    double dt;
    dt = length * sqrt(Po * Po + 1) / (c_mks * (Po + 1e-32));
    if (watch->referenceFrequency > 0)
      dt = ((long)(dt * watch->referenceFrequency + 0.5)) / watch->referenceFrequency;
#ifndef USE_KAHAN
    t0 += dt;
#else
    t0 = KahanPlus(t0, dt, &t0Error);
#endif
  }
  watch->t0Last = t0;
  watch->t0LastError = t0Error;
  watch->passLast = pass;

  if (!(pass >= watch->start_pass && (pass - watch->start_pass) % watch->interval == 0 &&
        (watch->end_pass < 0 || pass <= watch->end_pass))) {
    return;
  }

#if SDDS_MPI_IO
  if (isMaster && distributedBeam) /* No particle will be dumped by master */
    particles = 0;
  else
#endif
  {
    if (!watch->initialized)
      bombElegant("uninitialized watch-point (coordinate mode) encountered (dump_watch_particles)", NULL);
    if (!particle)
      bombElegant("NULL coordinate pointer passed to dump_watch_particles", NULL);
    if (watch->fraction > 1)
      bombElegant("Error: fraction>1 in dump_watch_particles", NULL);
    if (watch->startPID >= 0 && watch->startPID > watch->endPID)
      bombElegant("Error: startPID>endPID in dump_watch_particles", NULL);
    if ((watch->endPID >= 0 && watch->startPID < 0) || (watch->startPID >= 0 && watch->endPID < 0))
      bombElegant("Error: Invalid startPID, endPID in dump_watch_particles", NULL);
    for (i = 0; i < particles; i++)
      if (!particle[i]) {
        printf("error: coordinate slot %" PRId64 " is NULL (dump_watch_particles)\n", i);
        fflush(stdout);
        abort();
      }
  }
  if (watch->sparseInterval <= 0)
    watch->sparseInterval = 1;
  if (!SDDS_StartTable(watch->SDDS_table, particles / watch->sparseInterval + 1)) {
    SDDS_SetError("Problem starting SDDS table (dump_watch_particles)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  row = 0;
  count = 0;
#if SDDS_MPI_IO
  if ((isSlave && distributedBeam) || (!distributedBeam && isMaster))
#endif
    for (i = 0; i < particles; i += watch->sparseInterval) {
      if (!((watch->startPID < 0 && watch->endPID < 0) || (particle[i][6] >= watch->startPID && particle[i][6] <= watch->endPID)))
        continue;
      count++;
      if (watch->fraction == 1 || random_2(0) < watch->fraction) {
        if (watch->xData &&
            !SDDS_SetRowValues(watch->SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, row,
                               watch->xIndex[0], particle[i][0],
                               watch->excludeSlopes ? -1 : watch->xIndex[1], particle[i][1],
                               -1)) {
          SDDS_SetError("Problem setting SDDS row values (dump_watch_particles)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
        if (watch->yData &&
            !SDDS_SetRowValues(watch->SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, row,
                               watch->yIndex[0], particle[i][2],
                               watch->excludeSlopes ? -1 : watch->yIndex[1], particle[i][3],
                               -1)) {
          SDDS_SetError("Problem setting SDDS row values (dump_watch_particles)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
        if (watch->longitData) {
          p = Po * (1 + particle[i][5]);
          t = particle[i][4] / (c_mks * p / sqrt(sqr(p) + 1));
          if (!SDDS_SetRowValues(watch->SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, row,
                                 watch->longitIndex[0], t,
                                 watch->longitIndex[1], p,
                                 watch->longitIndex[2], t - t0,
                                 -1)) {
            SDDS_SetError("Problem setting SDDS row values (dump_watch_particles)");
            SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
          }
        }
        if (spinCoordOffset) {
          if (!SDDS_SetRowValues(watch->SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, row,
                                 watch->spinIndex[0], particle[i][spinCoordOffset+0],
                                 watch->spinIndex[1], particle[i][spinCoordOffset+1],
                                 watch->spinIndex[2], particle[i][spinCoordOffset+2],
                                 -1)) {
            SDDS_SetError("Problem setting SDDS row values (dump_watch_particles)");
            SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
          }
        }
        if (!SDDS_SetRowValues(watch->SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, row++,
                               watch->IDIndex, (uint64_t)particle[i][6], -1)) {
          SDDS_SetError("Problem setting SDDS row values (dump_watch_particles)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
      }
    }

#if USE_MPI
  if (USE_MPI && distributedBeam) {
    MPI_Allreduce(&row, &total_row, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&count, &total_count, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    if (isMaster)
      row = total_row;
    if (isMaster)
      count = total_count;
  }
  /* if (total_row)  */
#endif
  memoryUsed = memoryUsage();
  if (!SDDS_SetParameters(watch->SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                          "Step", step, "Pass", pass,
                          "Particles", count,
                          "Charge", mp_charge * count,
                          "pCentral", Po,
                          "PassLength", length,
                          "IDSlotsPerBunch", slotsPerBunch,
                          "PassCentralTime", t0, "s", z,
                          "ElapsedTime", delapsed_time(),
                          "MemoryUsage", memoryUsed,
#if USE_MPI
                          "ElapsedCoreTime", delapsed_time() * n_processors,
#else
                          "ElapsedCoreTime", delapsed_time(),
#endif
                          "SampledParticles", row,
                          "SampledCharge", mp_charge * row,
			  "tOffset", trackingClockOffset(),
                          NULL)) {
    SDDS_SetError("Problem setting SDDS parameters (dump_watch_particles)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

#if SDDS_MPI_IO
  if ((watch->useDisconnect) && (!SDDS_ReconnectFile(watch->SDDS_table)))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  if (total_row)
    if (!SDDS_MPI_WriteTable(watch->SDDS_table))
#else
  if (SDDS_IsDisconnected(watch->SDDS_table) && (!SDDS_ReconnectFile(watch->SDDS_table)))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  if (!SDDS_WriteTable(watch->SDDS_table))
#endif
    {
      SDDS_SetError("Problem writing SDDS table (dump_watch_particles)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  if (!inhibitFileSync)
    SDDS_DoFSync(watch->SDDS_table);
  if ((watch->useDisconnect) && (!SDDS_DisconnectFile(watch->SDDS_table)))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  if (!SDDS_ShortenTable(watch->SDDS_table, 1)) {
    SDDS_SetError("Problem shortening SDDS table (dump_watch_particles)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  log_exit("dump_watch_particles");
}

static double tmp_safe_sqrt;
#define SAFE_SQRT(x) ((tmp_safe_sqrt = (x)) < 0 ? (double)0.0 : sqrt(tmp_safe_sqrt))

void dump_watch_parameters(WATCH *watch, long step, long pass, long n_passes, double **particle,
                           int64_t particles, long original_particles, double Po,
                           double revolutionLength, double z, double mp_charge) {
  long sample, watchStartPass = watch->start_pass;
  int64_t i;
  double tc, tc0, tc0Error, p_sum, gamma_sum, sum, error_sum, p = 0.0;
  double emit[2], emitc[2];
  long Cx_index = 0, Sx_index = 0, ex_index = 0, ecx_index = 0, Cspx_index= 0;
  int64_t npCount;
  BEAM_SUMS *sums;
  double emittance_l;
  long memoryUsed;
#ifdef HAVE_GPU
  long useGpuParameters = 0;
#endif
#if USE_MPI
  int64_t particles_total, npCount_total = 0;

#  ifdef USE_MPE /* use the MPE library */
  int event1a, event1b;
  event1a = MPE_Log_get_event_number();
  event1b = MPE_Log_get_event_number();
  if (isMaster) {
    MPE_Describe_state(event1a, event1b, "End Watch", "pink");
  }
#  endif

  if (myid < 0)
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (myid == 0)
    particles = 0;

  MPI_Allreduce(&particles, &particles_total, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
#  ifdef DEBUG
  printf("particles = %" PRId64 ", particles_total = %" PRId64 "\n",
         particles, particles_total);
  fflush(stdout);
#  endif
#endif

  if (pass <= watch->passLast) {
    watch->t0Last = z * sqrt(Po * Po + 1) / (c_mks * (Po + 1e-32));
    watch->t0LastError = 0;
  }
  tc0 = watch->t0Last;
  tc0Error = watch->t0LastError;
  /* This code mimics what happens to particles as they get ~T0 added on each turn with accumulating
     * round-off error. Prevents dCt from walking off too much.
     */
  for (i = 0; i < pass - watch->passLast; i++) {
    double dt;
    dt = revolutionLength * sqrt(Po * Po + 1) / (c_mks * (Po + 1e-32));
    if (watch->referenceFrequency > 0)
      dt = ((long)(dt * watch->referenceFrequency + 0.5)) / watch->referenceFrequency;
#ifndef USE_KAHAN
    tc0 += dt;
#else
    tc0 = KahanPlus(tc0, dt, &tc0Error);
#endif
  }
  watch->t0Last = tc0;
  watch->t0LastError = tc0Error;
  watch->passLast = pass;

  if (!(pass >= watch->start_pass && (pass - watch->start_pass) % watch->interval == 0 &&
        (watch->end_pass < 0 || pass <= watch->end_pass))) {
    return;
  }

  if (isMaster) {
    log_entry("dump_watch_parameters");

    if (!watch->initialized)
      bombElegant("uninitialized watch-point (coordinate mode) encountered (dump_watch_parameters)", NULL);

    if (watch->fraction > 1)
      bombElegant("logic error--fraction>1 in dump_watch_parameters", NULL);
  }
  if (isSlave) {
    if (!particle && particles)
      bombElegant("NULL coordinate pointer passed to dump_watch_parameters", NULL);
  }

  for (i = 0; i < particles; i++)
    if (!particle[i]) {
      printf("error: coordinate slot %" PRId64 " is NULL (dump_watch_parameters)\n", i);
      fflush(stdout);
      abort();
    }

#ifdef DEBUG
  printf("dump_watch_parameters checkpoint 1\n");
  fflush(stdout);
#endif

  sample = (pass - watchStartPass) / watch->interval;

#ifdef HAVE_GPU
  useGpuParameters = gpu_watch_parameters_supported(watch, particles);
#endif

  if (isMaster)
    if ((watchStartPass == pass) && !SDDS_StartTable(watch->SDDS_table, (n_passes - watchStartPass) / watch->interval + 1)) {
      SDDS_SetError("Problem starting SDDS table (dump_watch_parameters)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }

#ifdef DEBUG
  printf("dump_watch_parameters checkpoint 2\n");
  fflush(stdout);
#endif

  /* compute centroids, sigmas, and emittances for x, y, and s */
  sums = allocateBeamSums(0, 1);
  zero_beam_sums(sums, 1);
  accumulate_beam_sums(sums, particle, particles, Po, mp_charge, NULL, 0.0, 0.0,
                       watch->startPID, watch->endPID, BEAM_SUMS_SPARSE | BEAM_SUMS_NOMINMAX);
#ifdef HAVE_GPU
  if (useGpuParameters &&
      !gpu_watch_parameter_sums(particles, &npCount, &p_sum, &gamma_sum)) {
    particle = copyParticlesToCpuReadOnly("WATCH parameter aggregate fallback");
    useGpuParameters = 0;
    zero_beam_sums(sums, 1);
    accumulate_beam_sums(sums, particle, particles, Po, mp_charge, NULL,
                         0.0, 0.0, watch->startPID, watch->endPID,
                         BEAM_SUMS_SPARSE | BEAM_SUMS_NOMINMAX);
  }
#endif
  if (isMaster) {
    if ((Cx_index = SDDS_GetColumnIndex(watch->SDDS_table, "Cx")) < 0) {
      SDDS_SetError("Problem getting index of SDDS columns (dump_watch_parameters)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    if (spinCoordOffset && (Cspx_index = SDDS_GetColumnIndex(watch->SDDS_table, "Cspx")) < 0) {
      SDDS_SetError("Problem getting index of SDDS columns (dump_watch_parameters)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    for (i = 0; i < 6; i++) {
      if (!SDDS_SetRowValues(watch->SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, sample,
                             Cx_index + i, sums->centroid[i],
                             -1)) {
        SDDS_SetError("Problem setting row values for SDDS table (dump_watch_parameters)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
    }
    if (spinCoordOffset) {
      for (i=0; i<3; i++) {
	if (!SDDS_SetRowValues(watch->SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, sample,
			       Cspx_index+i, sums->spinSums->centroid[i], -1)) {
          SDDS_SetError("Problem setting row values for SDDS table (dump_watch_parameters)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
	}
      }
    }
  }

#ifdef DEBUG
  printf("dump_watch_parameters checkpoint 3\n");
  fflush(stdout);
#endif

  if (watch->mode_code == WATCH_PARAMETERS) {
    for (i = 0; i < 2; i++) {
      emitc[i] = emit[i] = 0;
      computeEmitTwissFromSigmaMatrix(emit + i, emitc + i, NULL, NULL, sums->beamSums2->sigma, i * 2);
    }
    if (isMaster) {
      if ((Sx_index = SDDS_GetColumnIndex(watch->SDDS_table, "Sx")) < 0 ||
          (ex_index = SDDS_GetColumnIndex(watch->SDDS_table, "ex")) < 0 ||
          (ecx_index = SDDS_GetColumnIndex(watch->SDDS_table, "ecx")) < 0) {
        SDDS_SetError("Problem getting index of SDDS columns (dump_watch_parameters)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
      for (i = 0; i < 7; i++) {
        if (!SDDS_SetRowValues(watch->SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, sample,
                               Sx_index + i, SAFE_SQRT(sums->beamSums2->sigma[i][i]),
                               -1)) {
          SDDS_SetError("Problem setting row values for SDDS table (dump_watch_parameters)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
      }
      if (spinCoordOffset) {
        long j, offset=3;
        for (i=0; i<3; i++) {
	  for (j=i; j<3; j++, offset++) {
	    if (!SDDS_SetRowValues(watch->SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, sample,
				   Cspx_index+offset, sums->spinSums->sigma[i][j], -1)) {
              SDDS_SetError("Problem setting row values for SDDS table (dump_watch_parameters)");
              SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
	    }
	  }
	}
      }
      if (!SDDS_SetRowValues(watch->SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, sample,
                             ex_index, emit[0],
                             ex_index + 1, emit[1],
                             ecx_index, emitc[0],
                             ecx_index + 1, emitc[1],
                             -1)) {
        SDDS_SetError("Problem setting row values for SDDS table (dump_watch_parameters)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
    }
  }

#ifdef DEBUG
  printf("dump_watch_parameters checkpoint 4\n");
  fflush(stdout);
#endif

#if SDDS_MPI_IO
  emittance_l = rms_longitudinal_emittance_p(particle, particles, Po, watch->startPID, watch->endPID);
#  ifdef DEBUG
  printf("dump_watch_parameters checkpoint 4.1\n");
  fflush(stdout);
#  endif
  if (isMaster) {
    if (watch->mode_code == WATCH_PARAMETERS)
      if (!SDDS_SetRowValues(watch->SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, sample,
                             "el", emittance_l, NULL)) {
        SDDS_SetError("Problem setting row values for SDDS table (dump_watch_parameters)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
  }
#  ifdef DEBUG
  printf("dump_watch_parameters checkpoint 4.2\n");
  fflush(stdout);
#  endif
#else
#  ifdef HAVE_GPU
  if (useGpuParameters)
    emittance_l = SAFE_SQRT(sums->beamSums2->sigma[6][6] *
                              sums->beamSums2->sigma[5][5] -
                            sqr(sums->beamSums2->sigma[5][6]));
  else
#  endif
    emittance_l = rms_longitudinal_emittance(particle, particles, Po,
                                             watch->startPID, watch->endPID);
  if (watch->mode_code == WATCH_PARAMETERS)
    if (!SDDS_SetRowValues(watch->SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, sample,
                           "el", emittance_l, NULL)) {
      SDDS_SetError("Problem setting row values for SDDS table (dump_watch_parameters)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
#endif
#ifdef DEBUG
  printf("dump_watch_parameters checkpoint 5\n");
  fflush(stdout);
#endif

  /* time centroid and sigma */
  sum = error_sum = 0;
#ifdef HAVE_GPU
  if (useGpuParameters) {
    tc = sums->centroid[6];
  } else
#endif
  {
    for (i = npCount = p_sum = gamma_sum = 0; i < particles; i++) {
      if ((watch->startPID<0 && watch->endPID<0) || (particle[i][6] >= watch->startPID && particle[i][6] <= watch->endPID)) {
        p = Po * (1 + particle[i][5]);
        p_sum += p;
        gamma_sum += sqrt(sqr(p) + 1);
#ifndef USE_KAHAN
        sum += particle[i][4] / (p / sqrt(sqr(p) + 1) * c_mks);
#else
        sum = KahanPlus(sum, particle[i][4] / (p / sqrt(sqr(p) + 1) * c_mks), &error_sum);
#endif
        npCount++;
      }
    }
  }
#if USE_MPI
  if (USE_MPI) {
    double p_sum_total, gamma_sum_total, sum_total, error_sum_total;
    double outBuffer[5], inBuffer[5];
    inBuffer[0] = sum;
    inBuffer[1] = error_sum;
    inBuffer[2] = p_sum;
    inBuffer[3] = gamma_sum;
    inBuffer[4] = npCount;
    MPI_Allreduce(&inBuffer[0], &outBuffer[0], 5, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    sum_total = outBuffer[0];
    error_sum_total = outBuffer[1];
    p_sum_total = outBuffer[2];
    gamma_sum_total = outBuffer[3];
    npCount_total = outBuffer[4];
#  ifdef USE_KAHAN
    sum_total += error_sum_total;
#  endif
    if (npCount_total)
      tc = sum_total / npCount_total;
    else
      tc = 0;
    if (isMaster) {
      if (!SDDS_SetRowValues(watch->SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, sample,
                             "Ct", tc, "dCt", tc - tc0,
                             "pAverage", p_sum_total / npCount_total, "pCentral", Po,
                             "KAverage", (gamma_sum_total / npCount_total - 1) * me_mev, NULL)) {
        SDDS_SetError("Problem setting row values for SDDS table (dump_watch_parameters)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
    }
  }
#else
  if (
#  ifdef HAVE_GPU
      !useGpuParameters &&
#  endif
      particles)
    tc = sum / npCount;
  else if (
#  ifdef HAVE_GPU
           !useGpuParameters
#  else
           1
#  endif
          )
    tc = 0;

  if (!SDDS_SetRowValues(watch->SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, sample,
                         "Ct", tc, "dCt", tc - tc0,
                         "pAverage", p_sum / npCount, "pCentral", Po,
                         "KAverage", (gamma_sum / npCount - 1) * particleMassMV, NULL)) {
    SDDS_SetError("Problem setting row values for SDDS table (dump_watch_parameters)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
#endif

#ifdef DEBUG
  printf("dump_watch_parameters checkpoint 6\n");
  fflush(stdout);
#endif

  memoryUsed = memoryUsage();
  if (isMaster) {
    /* number of particles */
    if (!SDDS_SetRowValues(watch->SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, sample,
#if USE_MPI
                           "Particles", npCount_total,
                           "Transmission", (original_particles ? ((double)particles_total) / original_particles : (double)0.0),
                           "ElapsedCoreTime", delapsed_time() * n_processors,
                           "Charge", mp_charge * npCount_total,
#else
                           "Particles", npCount,
                           "Transmission", (original_particles ? ((double)particles) / original_particles : (double)0.0),
                           "ElapsedCoreTime", delapsed_time(),
                           "Charge", mp_charge * npCount,
#endif
                           "ElapsedTime", delapsed_time(),
                           "MemoryUsage", memoryUsed,
                           "Pass", pass,
                           "Step", step, NULL)) {
      SDDS_SetError("Problem setting row values for SDDS table (dump_watch_parameters)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    if (!SDDS_SetParameters(watch->SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                            "Step", step, "s", z, "PassLength", revolutionLength, NULL)) {
      SDDS_SetError("Problem setting parameter values for SDDS table (dump_watch_parameters)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }

    if (sample == (n_passes - 1) / watch->interval) {
      if (watch->flushInterval > 0) {
        if (sample != watch->flushSample && !SDDS_UpdatePage(watch->SDDS_table, 0)) {
          SDDS_SetError("Problem writing data for SDDS table (dump_watch_parameters)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
      } else if (!SDDS_WriteTable(watch->SDDS_table)) {
        SDDS_SetError("Problem writing data for SDDS table (dump_watch_parameters)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
    } else {
      if (watch->flushInterval > 0 && sample % watch->flushInterval == 0 &&
          !SDDS_UpdatePage(watch->SDDS_table, 0)) {
        SDDS_SetError("Problem flushing data for SDDS table (dump_watch_parameters)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
      watch->flushSample = sample;
    }
#ifdef USE_MPE
    MPE_Log_event(event1a, 0, "start watch"); /* record time spent on I/O operations */
#endif
    if (!inhibitFileSync)
      SDDS_DoFSync(watch->SDDS_table);
#ifdef USE_MPE
    MPE_Log_event(event1b, 0, "end watch");
#endif
    log_exit("dump_watch_parameters");
  }
#ifdef DEBUG
  printf("dump_watch_parameters checkpoint 7\n");
  fflush(stdout);
#endif
  freeBeamSums(sums, 1);
  }

void dump_watch_FFT(WATCH *watch, long step, long pass, long n_passes, double **particle, long particles,
                    long original_particles, double Po, double revolutionLength) {
  long sample_index, i, samples;
  double sum_x, sum_y, sum_dp, *sample[4];

#if USE_MPI
  if (myid < 0)
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (myid != 0)
    return;
#endif

  log_entry("dump_watch_FFT");

  samples = n_passes / watch->interval;
  if (!power_of_2(samples))
    bombElegant("number of samples must be a power of 2 for FFT (dump_watch_FFT)", NULL);
  if (!watch->initialized)
    bombElegant("uninitialized watch-point (coordinate mode) encountered (dump_watch_FFT)", NULL);
  if (!particle)
    bombElegant("NULL coordinate pointer passed to dump_watch_FFT", NULL);
  for (i = 0; i < particles; i++)
    if (!particle[i]) {
      printf("error: coordinate slot %ld is NULL (dump_watch_FFT)\n", i);
      fflush(stdout);
      abort();
    }

  if (pass == 0 && !SDDS_StartTable(watch->SDDS_table, 2 * samples)) {
    SDDS_SetError("Problem starting SDDS table (dump_watch_FFT)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  sample_index = pass / watch->interval;
  for (i = sum_x = sum_y = sum_dp = 0; i < particles; i++) {
    sum_x += particle[i][0];
    sum_y += particle[i][2];
    sum_dp += particle[i][5];
  }
  if (particles) {
    sum_x /= particles;
    sum_y /= particles;
    sum_dp /= particles;
  }
  /* SDDS is used to store the centroid vs pass data prior to FFT */
  if (!SDDS_SetRowValues(watch->SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, sample_index,
                         0, ((double)sample_index) / samples, 1, sum_x, 2, sum_y, 3, sum_dp, -1)) {
    SDDS_SetError("Problem setting row values for SDDS table (dump_watch_FFT)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  if (sample_index == samples - 1) {
    /* sample/samples stored temporarily in frequency column: */
    if (!(sample[0] = (double *)SDDS_GetColumn(watch->SDDS_table, "f")) ||
        !(sample[1] = (double *)SDDS_GetColumn(watch->SDDS_table, "Cx")) ||
        !(sample[2] = (double *)SDDS_GetColumn(watch->SDDS_table, "Cy")) ||
        !(sample[3] = (double *)SDDS_GetColumn(watch->SDDS_table, "Cdelta"))) {
      SDDS_SetError("Problem retrieving sample values from SDDS table (dump_watch_FFT)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    do_watch_FFT(sample, samples, 1, watch->window_code);
    do_watch_FFT(sample, samples, 2, watch->window_code);
    do_watch_FFT(sample, samples, 3, watch->window_code);
    if (!SDDS_EraseData(watch->SDDS_table) ||
        !SDDS_SetColumn(watch->SDDS_table, SDDS_SET_BY_INDEX, sample[0], samples / 2, 0) ||
        !SDDS_SetColumn(watch->SDDS_table, SDDS_SET_BY_INDEX, sample[1], samples / 2, 1) ||
        !SDDS_SetColumn(watch->SDDS_table, SDDS_SET_BY_INDEX, sample[2], samples / 2, 2) ||
        !SDDS_SetColumn(watch->SDDS_table, SDDS_SET_BY_INDEX, sample[3], samples / 2, 3)) {
      SDDS_SetError("Problem setting columns of SDDS table (dump_watch_FFT)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    for (i = 0; i < 4; i++)
      free(sample[i]);
    if (!SDDS_SetParameters(watch->SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                            "Step", step, "PassLength", revolutionLength, NULL)) {
      SDDS_SetError("Problem setting parameter values for SDDS table (dump_watch_FFT)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    if (!SDDS_WriteTable(watch->SDDS_table)) {
      SDDS_SetError("Problem writing data to SDDS file (dump_watch_FFT)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    if (!inhibitFileSync)
      SDDS_DoFSync(watch->SDDS_table);
  }
  log_exit("dump_watch_FFT");
}

void do_watch_FFT(double **data, long n_data, long slot, long window_code) {
  double *real_imag, r1, r2;
  long i;

#if USE_MPI
  if (myid < 0)
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (myid != 0)
    return;
#endif

  log_entry("do_watch_FFT");
  if (!data[slot])
    bombElegant("data pointer for specified slot is NULL (do_watch_FFT)", NULL);

  /* load into temporary array */
  real_imag = tmalloc(sizeof(*real_imag) * (n_data + 2));
  for (i = 0; i < n_data; i++)
    real_imag[i] = data[slot][i];

  /* apply the window */
  switch (window_code) {
  case FFT_HANNING:
    r1 = PIx2 / (n_data - 1);
    for (i = 0; i < n_data; i++)
      real_imag[i] *= (1 - cos(i * r1)) / 2;
    break;
  case FFT_PARZEN:
    r1 = (n_data - 1) / 2.0;
    for (i = 0; i < n_data; i++)
      real_imag[i] *= 1 - FABS((i - r1) / r1);
    break;
  case FFT_WELCH:
    r1 = (n_data - 1) / 2.0;
    r2 = sqr((n_data + 1) / 2.0);
    for (i = 0; i < n_data; i++)
      real_imag[i] *= 1 - sqr(i - r1) / r2;
    break;
  case FFT_UNIFORM:
    break;
  default:
    break;
  }

  /* do the FFT */
  if (!realFFT2(real_imag, real_imag, n_data, 0)) {
    printf("error: FFT fails for slot %ld of watch point\n", slot);
    fflush(stdout);
    abort();
  }

  /* put the amplitudes back into the data array.  Multiply the non-DC
     * terms by 2 to account for amplitude in negative frequencies. 
     */
  data[slot][0] = fabs(real_imag[0]);
  for (i = 1; i < n_data / 2; i++)
    data[slot][i] = sqrt(sqr(real_imag[2 * i]) + (i == 0 ? 0 : sqr(real_imag[2 * i + 1]))) * 2;

  free(real_imag);
  log_exit("do_watch_FFT");
}

void dump_particle_histogram(HISTOGRAM *histogram, long step, long pass, double **particle, int64_t particles,
                             double Po, double length, double charge, double z) {
  long icoord, ibin;
  int64_t ipart, jpart;
  double p, t0;
  static long maxBins = 0;
  static int64_t maxParticles = 0;
  static double *frequency = NULL, *coordinate = NULL, *histData = NULL;
  static double *gpuFrequency = NULL;
  double center, range, lower, upper;
  static short *chosen = NULL;
  int64_t nChosen = 0;
#ifdef HAVE_GPU
  double gpuMinimum[7], gpuMaximum[7], gpuLower[7], gpuUpper[7];
  unsigned int gpuCoordinateMask = 0;
  long useGpu = getElementOnGpu();
#endif

#if USE_MPI
  static double *buffer = NULL;
  int64_t particles_total, nChosen_total;
#endif

  log_entry("dump_particle_histogram");
  if (!histogram->initialized)
    bombElegant("uninitialized histogram encountered (dump_particle_histogram)", NULL);
  if (isSlave)
    if (!particle)
      bombElegant("NULL coordinate pointer passed to dump_particle_histogram", NULL);
  /*
  for (ipart=0; ipart<particles; ipart++)
    if (!particle[ipart]) {
      printf("error: coordinate slot %ld is NULL (dump_particle_histogram)\n", ipart);
      fflush(stdout);
      abort();
    }
  */
  if (isMaster) {
    if (!SDDS_StartTable(histogram->SDDS_table, histogram->bins)) {
      SDDS_SetError("Problem starting SDDS table (dump_particle_histogram)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  }
  if (histogram->bins > maxBins) {
    maxBins = histogram->bins;
    if (!(frequency = SDDS_Realloc(frequency, sizeof(*frequency) * maxBins)) ||
        !(coordinate = SDDS_Realloc(coordinate, sizeof(*coordinate) * maxBins)) ||
        !(gpuFrequency = SDDS_Realloc(gpuFrequency, sizeof(*gpuFrequency) * 7 * maxBins))) {
      SDDS_Bomb("Memory allocation failure (dump_particle_histogram)");
    }
#if USE_MPI
    if (!(buffer = SDDS_Realloc(buffer, sizeof(*buffer) * maxBins))) {
      SDDS_Bomb("Memory allocation failure (dump_particle_histogram)");
    }
    MPI_Reduce(&particles, &particles_total, 1, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
  }
  if (isSlave) {
    if (particles > maxParticles) {
      maxParticles = particles;
      if (!(histData = SDDS_Realloc(histData, sizeof(*histData) * maxParticles)) ||
          !(chosen = SDDS_Realloc(chosen, sizeof(*chosen) * maxParticles)))
        SDDS_Bomb("Memory allocation failure (dump_particle_histogram)");
    }
  }

  t0 = pass * length * sqrt(Po * Po + 1) / (c_mks * (Po + 1e-32));

#ifdef HAVE_GPU
  if (useGpu) {
    for (icoord = 0; icoord < 7; icoord++) {
      if (histogram->columnIndex[icoord][0] == -1 ||
          (histogram->sparse && (icoord == 1 || icoord == 3 || icoord == 5)))
        continue;
      gpuCoordinateMask |= 1U << icoord;
    }
    nChosen = gpu_histogram_ranges(particles, Po,
                                   histogram->startPID, histogram->endPID,
                                   gpuCoordinateMask,
                                   gpuMinimum, gpuMaximum);
    if (!nChosen) {
      particle = copyParticlesToCpuReadOnly("HISTOGRAM empty-selection fallback");
      useGpu = 0;
    } else {
      if (gpuCoordinateMask & (1U << 6)) {
        gpuMinimum[6] -= t0;
        gpuMaximum[6] -= t0;
      }
      for (icoord = 0; icoord < 7; icoord++) {
        gpuLower[icoord] = gpuUpper[icoord] = 0;
        if (!(gpuCoordinateMask & (1U << icoord)))
          continue;
        lower = gpuMinimum[icoord];
        upper = gpuMaximum[icoord];
        if (lower == upper) {
          lower -= 1e-16;
          upper += 1e-16;
          /* Preserve the CPU zero-width-range convention for the later
           * coordinate and normalization output calculations as well as
           * for the GPU binning bounds computed here. */
          gpuMinimum[icoord] = lower;
          gpuMaximum[icoord] = upper;
        }
        if (histogram->count == 0 || !histogram->fixedBinSize) {
          range = histogram->binSizeFactor * (upper - lower);
          histogram->binSize[icoord] = range / histogram->bins;
        } else
          range = histogram->binSize[icoord] * histogram->bins;
        center = (lower + upper) / 2;
        range *= (1 + 1e-12);
        gpuLower[icoord] = center - range / 2;
        gpuUpper[icoord] = center + range / 2;
      }
      gpu_histogram_bins(particles, Po,
                         histogram->startPID, histogram->endPID,
                         histogram->bins, gpuCoordinateMask,
                         t0, gpuLower, gpuUpper, gpuFrequency);
    }
  }
#endif
  if (
#ifdef HAVE_GPU
      !useGpu
#else
      1
#endif
  ) {
    for (ipart = 0; ipart < particles; ipart++) {
      if ((histogram->startPID<0 && histogram->endPID<0) ||
          (particle[ipart][6] >= histogram->startPID && particle[ipart][6] <= histogram->endPID)) {
        chosen[ipart] = 1;
        nChosen++;
      } else
        chosen[ipart] = 0;
    }
  }

  for (icoord = 0; icoord < 7; icoord++) {
    upper = -(lower = DBL_MAX);
    if (histogram->columnIndex[icoord][0] == -1)
      continue;
    if (histogram->sparse && (icoord == 1 || icoord == 3 || icoord == 5))
      continue;
    if (
#ifdef HAVE_GPU
        useGpu
#else
        0
#endif
    ) {
#ifdef HAVE_GPU
      lower = gpuMinimum[icoord];
      upper = gpuMaximum[icoord];
#endif
    } else if (isSlave) {
      if (icoord == 4) {
        /* t */
        for (ipart = jpart = 0; ipart < particles; ipart++) {
          if (chosen[ipart]) {
            p = Po * (1 + particle[ipart][5]);
            histData[jpart++] = particle[ipart][4] / (c_mks * p / sqrt(sqr(p) + 1));
          }
        }
      } else if (icoord == 6) {
        /* dt (could reuse computations for t, but would need to store another array) */
        for (ipart = jpart = 0; ipart < particles; ipart++) {
          if (chosen[ipart]) {
            p = Po * (1 + particle[ipart][5]);
            histData[jpart++] = particle[ipart][4] / (c_mks * p / sqrt(sqr(p) + 1)) - t0;
          }
        }
      } else {
        for (ipart = jpart = 0; ipart < particles; ipart++)
          if (chosen[ipart])
            histData[jpart++] = particle[ipart][icoord];
      }
      if (jpart != nChosen)
        bombElegant("fatal error: particle counting problem making histogram", NULL);
      find_min_max(&lower, &upper, histData, nChosen);
      if (lower == upper) {
        lower -= 1e-16;
        upper += 1e-16;
      }
    }
#if USE_MPI
    if (isMaster) /* The particles are on slave */
      nChosen = 0;
    find_global_min_max(&lower, &upper, nChosen, MPI_COMM_WORLD);
#endif
    if (histogram->count == 0 || !histogram->fixedBinSize) {
      range = histogram->binSizeFactor * (upper - lower);
      histogram->binSize[icoord] = range / histogram->bins;
    } else
      range = histogram->binSize[icoord] * histogram->bins;
    center = (lower + upper) / 2;
    range *= (1 + 1e-12); /* Rounding error might cause particle lost in the first and last bins */
    lower = center - range / 2;
    upper = center + range / 2;
    for (ibin = 0; ibin < histogram->bins; ibin++)
      coordinate[ibin] = (ibin + 0.5) * histogram->binSize[icoord] + lower;
    if (
#ifdef HAVE_GPU
        useGpu
#else
        0
#endif
    ) {
#ifdef HAVE_GPU
      memcpy(frequency, gpuFrequency + icoord * histogram->bins,
             sizeof(*frequency) * histogram->bins);
#endif
    } else
      make_histogram(frequency, histogram->bins, lower, upper, histData, nChosen, 1);
#if USE_MPI
    if (USE_MPI) /* This will update the number of particles locally on master if there are lost on slaves */
      MPI_Reduce(&nChosen, &nChosen_total, 1, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    if (isMaster)
      nChosen = nChosen_total;
    MPI_Allreduce(frequency, buffer, histogram->bins, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    memcpy(frequency, buffer, sizeof(*buffer) * histogram->bins);
    if (isMaster && nChosen_total)
#endif
    {
      if (histogram->normalize)
        for (ibin = 0; ibin < histogram->bins; ibin++)
          frequency[ibin] /= nChosen * (upper - lower) / histogram->bins;
      if (histogram->sparse) {
        long row;
        for (ibin = row = 0; ibin < histogram->bins; ibin++) {
          if (frequency[ibin] &&
              !SDDS_SetRowValues(histogram->SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, row++,
                                 histogram->columnIndex[icoord][0], coordinate[ibin],
                                 histogram->columnIndex[icoord][1], frequency[ibin], -1)) {
            SDDS_Bomb("Problem setting row values (dump_particle_histogram)");
          }
        }
      } else {
        if (!SDDS_SetColumn(histogram->SDDS_table, SDDS_SET_BY_INDEX,
                            coordinate, histogram->bins, histogram->columnIndex[icoord][0]) ||
            !SDDS_SetColumn(histogram->SDDS_table, SDDS_SET_BY_INDEX,
                            frequency, histogram->bins, histogram->columnIndex[icoord][1])) {
          SDDS_Bomb("Problem setting column values (dump_particle_histogram)");
        }
      }
    }
  }

#if SDDS_MPI_IO
  if (isMaster && nChosen_total)
#endif
  {
    if (!SDDS_SetParameters(histogram->SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                            "Step", step, "Pass", pass, "Particles", nChosen, "pCentral", Po,
                            "PassLength", length, "Charge", charge,
                            "PassCentralTime", t0, "s", z,
                            NULL)) {
      SDDS_SetError("Problem setting SDDS parameters (dump_particle_histogram)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    if (!SDDS_WriteTable(histogram->SDDS_table)) {
      SDDS_SetError("Problem writing SDDS table (dump_particle_histogram)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    if (!inhibitFileSync)
      SDDS_DoFSync(histogram->SDDS_table);
  }
  histogram->count++;
}

void dump_phase_space(SDDS_TABLE *SDDS_table, double **particle, int64_t particles, long step, double Po,
                      double charge, int64_t slotsPerBunch) {
  int64_t i;
  double p;

#if SDDS_MPI_IO
  int64_t total_particles;

  if (!SDDS_table->parallel_io && isSlave)
    return;

  /* Open file here for parallel IO */
  if (!SDDS_table->layout.layout_written) { /* Check if the file has been opened already */
    if (!SDDS_MPI_File_Open(SDDS_table->MPI_dataset, SDDS_table->layout.filename, SDDS_MPI_WRITE_ONLY))
      SDDS_MPI_BOMB("SDDS_MPI_File_Open failed.", &SDDS_table->MPI_dataset->MPI_file);
    if (!SDDS_MPI_WriteLayout(SDDS_table))
      SDDS_MPI_BOMB("SDDS_MPI_WriteLayout failed.", &SDDS_table->MPI_dataset->MPI_file);
  }

#endif

  log_entry("dump_phase_space");
  if ((distributedBeam && isSlave) || (!distributedBeam && isMaster)) {
    if (!particle)
      bombElegant("NULL coordinate pointer passed to dump_phase_space", NULL);

    for (i = 0; i < particles; i++)
      if (!particle[i]) {
        printf("error: coordinate slot %" PRId64 " is NULL (dump_phase_space)\n", i);
        fflush(stdout);
        abort();
      }
  }
  if (!SDDS_StartTable(SDDS_table, particles)) {
    SDDS_SetError("Problem starting SDDS table (dump_phase_space)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
#if USE_MPI
  if (distributedBeam) {
    MPI_Reduce(&particles, &total_particles, 1, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    if (isMaster) {
      particles = total_particles;
      if (slotsPerBunch==0)
        slotsPerBunch = total_particles;
    }
  }
#endif
  if (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                          "Step", step, "pCentral", Po, "Particles", particles,
                          "Charge", charge, "IDSlotsPerBunch", slotsPerBunch,
			  "tOffset", trackingClockOffset(), NULL)) {
    SDDS_SetError("Problem setting parameter values for SDDS table (dump_phase_space)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
#if SDDS_MPI_IO
  if ((distributedBeam && isSlave) || (!distributedBeam && isMaster))
#endif
    for (i = 0; i < particles; i++) {
      p = Po * (1 + particle[i][5]);
      if (!SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, i,
                             0, particle[i][0], 1, particle[i][1], 2, particle[i][2], 3, particle[i][3],
                             4, particle[i][4] / (c_mks * p / sqrt(sqr(p) + 1)), 5, p,
                             6, (uint64_t)particle[i][6], -1) ||
	  (spinCoordOffset>0 &&
	  !SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, i,
                             7, particle[i][spinCoordOffset+0], 8, particle[i][spinCoordOffset+1],
			     9, particle[i][spinCoordOffset+2],
                             -1))) {
        SDDS_SetError("Problem setting SDDS row values (dump_phase_space)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
    }
  if (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "Step", step, NULL)) {
    SDDS_SetError("Problem setting SDDS parameters (dump_phase_space)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
#if SDDS_MPI_IO
  if ((distributedBeam && !SDDS_MPI_WriteTable(SDDS_table)) || (!distributedBeam && !SDDS_WriteTable(SDDS_table)) || !SDDS_ShortenTable(SDDS_table, 1))
#else
  if (!SDDS_WriteTable(SDDS_table) || !SDDS_ShortenTable(SDDS_table, 1))
#endif
  {
    SDDS_SetError("Problem writing SDDS table (dump_phase_space)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  if (!inhibitFileSync)
    SDDS_DoFSync(SDDS_table);

  log_exit("dump_phase_space");
}

#ifdef SORT
static int comp_IDs1(const void **coord1, const void **coord2) {
  long v1, v2;
  v1 = (*((long **)coord1))[1];
  v2 = (*((long **)coord2))[1];
  if (v1 < v2)
    return -1;
  else if (v1 > v2)
    return 1;
  else
    return 0;
}
#endif

void dump_lost_particles(SDDS_TABLE *SDDS_table, double *sLimit, double pCentral, double **particle, int64_t particles,
			 long step, double length) {
  int64_t i, row, badPID;
  long spxIndex=-1;
#if USE_MPI && MPI_DEBUG
  printf("dump_lost_particles: running\n");
  fflush(stdout);
#endif
#if SDDS_MPI_IO
  /* Open file here for parallel IO */
  if (!SDDS_table->layout.layout_written) { /* Check if the file has been opened already */
    if (!SDDS_MPI_File_Open(SDDS_table->MPI_dataset, SDDS_table->layout.filename, SDDS_MPI_WRITE_ONLY))
      SDDS_MPI_BOMB("SDDS_MPI_File_Open failed.", &SDDS_table->MPI_dataset->MPI_file);
    if (!SDDS_MPI_WriteLayout(SDDS_table))
      SDDS_MPI_BOMB("SDDS_MPI_WriteLayout failed.", &SDDS_table->MPI_dataset->MPI_file);
  }
#endif

  log_entry("dump_lost_particles");
  if (isSlave) {
    if (!particle && particles)
      bombElegant("NULL coordinate pointer passed to dump_lost_particles", NULL);
  }

#if USE_MPI && MPI_DEBUG
  printf("dump_lost_particles: pointers ok\n");
  fflush(stdout);
#endif

#ifdef SORT /* Sort for comparing the serial and parallel versions. Disabled for parallel I/O version */
  if (SORT && particles) {
    /* This call looks strange, but works because the particle data is contiguous.  The sort
         * actually moves the data around without changing the pointers.
         */
    qsort(particle[0], particles, (COORDINATES_PER_PARTICLE + 1) * sizeof(double), comp_IDs);
  }
#endif
#if USE_MPI && MPI_DEBUG
  printf("checking PIDs on %" PRId64 " particles\n", particles);
  fflush(stdout);
#endif
  for (i = badPID = 0; i < particles; i++) {
#if USE_MPI && MPI_DEBUG
    printf("checking PID on particle %" PRId64 " of %" PRId64 "\n", i, particles);
    fflush(stdout);
#endif
    if (!particle[i]) {
      printf("error: coordinate slot %" PRId64 " is NULL (dump_lost_particles)\n", i);
      fflush(stdout);
#if USE_MPI && MPI_DEBUG
      printf("bad particle pointer for particle %" PRId64 "\n", i);
#endif
      abort();
    }
    if (particle[i][6] <= 0) {
      /*
#if USE_MPI
	  if (!badPID) 
	    dup2(fdStdout, fileno(stdout));
#endif
	  printf("particle %ld has bad PID: %ld\n", i, (long)particle[i][6]); 
	  */
      badPID++;
    }
  }
  if (badPID) {
#if USE_MPI
    printf("%" PRId64 " particles with \"bad\" PID on processor %d\n", badPID, myid);
#else
    printf("%" PRId64 " particles with \"bad\" PID\n", badPID);
#endif
    fflush(stdout);
  }
#if USE_MPI && MPI_DEBUG
  printf("PIDs checked.\n");
  fflush(stdout);
#endif

  if (spinCoordOffset) {
    if ((spxIndex=SDDS_GetColumnIndex(SDDS_table, "spx"))<0) {
      SDDS_SetError("Problem finding column index for spx (dump_lost_particles)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  }
  if (!SDDS_StartTable(SDDS_table, particles)) {
    SDDS_SetError("Problem starting SDDS table (dump_lost_particles)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  for (i = row = 0; i < particles; i++) {
#if USE_MPI && MPI_DEBUG
    printf("Setting row values for particle %" PRId64 "\n", i);
    fflush(stdout);
#endif
    if (particle[i][4] >= sLimit[0] && particle[i][4] <= sLimit[1]) {
      if (!SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, row,
                             0, particle[i][0], 1, particle[i][1], 2, particle[i][2], 3, particle[i][3],
                             4, particle[i][4], 5, particle[i][5],
                             6, (uint64_t)particle[i][6], 7, (long)particle[i][lossPassIndex], -1)) {
        SDDS_SetError("Problem setting SDDS row values (dump_lost_particles)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
      if (globalLossCoordOffset > 0) {
        /* global loss coordinates are available */
        if (!SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, row,
                               8, particle[i][globalLossCoordOffset + 0],
                               9, particle[i][globalLossCoordOffset + 1],
                               10, particle[i][globalLossCoordOffset + 2],
                               -1)) {
          SDDS_SetError("Problem setting SDDS row values for global coordinates (dump_lost_particles)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
      }
      if (spinCoordOffset) {
        /* spin coordinates are available */
        if (!SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, row,
                               spxIndex+0, particle[i][spinCoordOffset + 0],
                               spxIndex+1, particle[i][spinCoordOffset + 1],
                               spxIndex+2, particle[i][spinCoordOffset + 2],
                               -1)) {
          SDDS_SetError("Problem setting SDDS row values for spin (dump_lost_particles)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
      }
      row++;
    }
  }
#if USE_MPI && MPI_DEBUG
  printf("dump_lost_particles: set row values for %" PRId64 " particles\n", particles);
  fflush(stdout);
#endif
  if (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "Step", step, "PassLength", length,
			  "pCentral", pCentral, NULL)) {
    SDDS_SetError("Problem setting SDDS parameters (dump_lost_particles)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
#if SDDS_MPI_IO
  if (!SDDS_MPI_WriteTable(SDDS_table))
#else
  if (!SDDS_WriteTable(SDDS_table))
#endif
  {
    SDDS_SetError("Problem writing SDDS table (dump_lost_particles)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  if (!inhibitFileSync)
    SDDS_DoFSync(SDDS_table);

  log_exit("dump_lost_particles");
#if USE_MPI && MPI_DEBUG
  printf("dump_lost_particles: returning\n");
  fflush(stdout);
#endif
}

void dump_centroid(SDDS_TABLE *SDDS_table, BEAM_SUMS *sums, LINE_LIST *beamline, long n_elements, long step,
                   double p_central, short bpmsOnly) {
  long i, j, row;
  BEAM_SUMS *beam;
  ELEMENT_LIST *eptr, *eptrLast;
  double *cent;
  char *name, *type_name;
  long s_index, Cx_index = -1, occurence, type, Cspx_index=-1;
  long wIndex = -1;

#if USE_MPI
  if (myid < 0)
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (myid != 0)
    return;
#endif

  log_entry("dump_centroid");

  if (!SDDS_table)
    bombElegant("SDDS_TABLE pointer is NULL (dump_centroid)", NULL);
  if (!sums)
    bombElegant("BEAM_SUMS pointer is NULL (dump_centroid)", NULL);
  if (!beamline)
    bombElegant("LINE_LIST pointer is NULL (dump_centroid)", NULL);
  if ((s_index = SDDS_GetColumnIndex(SDDS_table, "s")) < 0 ||
      (Cx_index = SDDS_GetColumnIndex(SDDS_table, "Cx")) < 0) {
    SDDS_SetError("Problem getting index of SDDS columns (dump_centroid)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  if (spinCoordOffset &&
      (Cspx_index=SDDS_GetColumnIndex(SDDS_table, "Cspx"))<0) {
    SDDS_SetError("Problem getting index of SDDS spin columns (dump_centroid)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  if (!SDDS_StartTable(SDDS_table, n_elements + 1)) {
    SDDS_SetError("Problem starting SDDS table (dump_centroid)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  if (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                          "Step", step, "PassLength", beamline->revolution_length, NULL)) {
    SDDS_SetError("Problem setting parameter values for SDDS table (dump_centroid)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  if (!(beamline->flags & BEAMLINE_CONCAT_DONE))
    eptr = beamline->elem;
  else
    eptr = beamline->ecat;
  eptrLast = eptr;
  name = "_BEG_";
  type_name = "MARK";
  occurence = 1;
  type = -1;
  cent = tmalloc(sizeof(*cent) * 7);
  for (i = row = 0; i < n_elements; i++) {
    if (!eptr) {
      printf("element pointer is NULL, i=%ld (dump_centroid)", i);
      fflush(stdout);
      abort();
    }
    beam = sums + i;
    if (beam->n_part)
      for (j = 0; j < 7; j++) {
        cent[j] = beam->centroid[j];
        if (isnan(cent[j]) || isinf(cent[j]))
          cent[j] = DBL_MAX;
      }
    else
      for (j = 0; j < 7; j++)
        cent[j] = 0;
    if (!bpmsOnly || type == T_MONI || type == T_HMON || type == T_VMON) {
      if (spinCoordOffset) {
	if (!SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, row,
			       Cspx_index+0, beam->spinSums->centroid[0],
			       Cspx_index+1, beam->spinSums->centroid[1],
			       Cspx_index+2, beam->spinSums->centroid[2],
			       -1)) {
          SDDS_SetError("Problem setting spin row values for SDDS table (dump_centroid)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
	}
      }
      if (!SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, row,
                             Cx_index, cent[0], Cx_index + 1, cent[1], Cx_index + 2, cent[2],
                             Cx_index + 3, cent[3], Cx_index + 4, cent[4], Cx_index + 5, cent[5],
			     Cx_index + 6, cent[6], Cx_index + 7, beam->n_part,
			     Cx_index + 8, beam->p0, Cx_index + 9, beam->charge, Cx_index+10, beam->pass,
                             s_index, beam->z, s_index + 1, name, s_index + 2, occurence,
                             s_index + 3, type_name, -1)) {
        SDDS_SetError("Problem setting row values for SDDS table (dump_centroid)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
      if (bpmsOnly) {
        double xWeight, yWeight;
        if (wIndex==-1 &&
            ((wIndex = SDDS_GetColumnIndex(SDDS_table, "xBPMWeight")) < 0)) {
          SDDS_SetError("Problem getting index of SDDS columns (dump_centroid)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
        xWeight = yWeight = -1;
        if (type == T_MONI) {
          xWeight = yWeight = ((MONI*)(eptrLast->p_elem))->weight;
        } else if (type == T_HMON) {
          //print_elem(stdout, eptrLast);
          xWeight = ((HMON*)(eptrLast->p_elem))->weight;
        } else if (type == T_VMON) {
          //print_elem(stdout, eptrLast);
          yWeight = ((VMON*)(eptrLast->p_elem))->weight;
        }
        //printf("%s %le %le\n", name, xWeight, yWeight);
        if (!SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_INDEX|SDDS_PASS_BY_VALUE, row,
                               wIndex, xWeight,
                               wIndex+1, yWeight,
                               -1)) {
          SDDS_SetError("Problem setting row values for SDDS table (dump_centroid)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
      }
      row++;
    }
    name = eptr->name;
    type_name = entity_name[eptr->type];
    type = eptr->type;
    occurence = eptr->occurence;
    eptrLast = eptr;
    if (eptr->succ)
      eptr = eptr->succ;
    else if (beamline->elem_recirc)
      eptr = beamline->elem_recirc;
    else
      eptr = beamline->elem;
  }

  if (!SDDS_WriteTable(SDDS_table)) {
    SDDS_SetError("Unable to write centroid data (dump_centroid)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  if (!inhibitFileSync)
    SDDS_DoFSync(SDDS_table);
  if (!SDDS_EraseData(SDDS_table)) {
    SDDS_SetError("Unable to erase centroid data (dump_centroid)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  free(cent);
  log_exit("dump_centroid");
}

void dump_sigma(SDDS_TABLE *SDDS_table, BEAM_SUMS *sums, LINE_LIST *beamline, long n_elements, long step,
                double p_central, short coupledSigmaOutput) {
  long i, j, ie, offset, plane, index;
  BEAM_SUMS *beam;
  ELEMENT_LIST *eptr;
  double emit, emitNorm, emitc, emitcNorm, beta, alpha;
  char *name, *type_name;
  long s_index = 0, ma1_index = 0, min1_index = 0, max1_index = 0, Sx_index = 0, occurence, ex_index = 0, betax_index = 0;
  long sNIndex[7] = {0, 0, 0, 0, 0, 0, 0}, spinIndex=-1;

#if USE_MPI
  if (myid < 0)
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if (myid != 0)
    return;
#endif

  if (!SDDS_table)
    bombElegant("SDDS_TABLE pointer is NULL (dump_sigma)", NULL);
  if (!sums)
    bombElegant("BEAM_SUMS pointer is NULL (dump_centroid)", NULL);
  if (!beamline)
    bombElegant("LINE_LIST pointer is NULL (dump_centroid)", NULL);
  if ((s_index = SDDS_GetColumnIndex(SDDS_table, "s")) < 0 ||
      (sNIndex[0] = SDDS_GetColumnIndex(SDDS_table, "s1")) < 0 ||
      (sNIndex[1] = SDDS_GetColumnIndex(SDDS_table, "s2")) < 0 ||
      (sNIndex[2] = SDDS_GetColumnIndex(SDDS_table, "s3")) < 0 ||
      (sNIndex[3] = SDDS_GetColumnIndex(SDDS_table, "s4")) < 0 ||
      (sNIndex[4] = SDDS_GetColumnIndex(SDDS_table, "s5")) < 0 ||
      (sNIndex[5] = SDDS_GetColumnIndex(SDDS_table, "s6")) < 0 ||
      (sNIndex[6] = SDDS_GetColumnIndex(SDDS_table, "s7")) < 0 ||
      (ma1_index = SDDS_GetColumnIndex(SDDS_table, "ma1")) < 0 ||
      (min1_index = SDDS_GetColumnIndex(SDDS_table, "minimum1")) < 0 ||
      (max1_index = SDDS_GetColumnIndex(SDDS_table, "maximum1")) < 0 ||
      (Sx_index = SDDS_GetColumnIndex(SDDS_table, "Sx")) < 0 ||
      (ex_index = SDDS_GetColumnIndex(SDDS_table, "ex")) < 0 ||
      (betax_index = SDDS_GetColumnIndex(SDDS_table, "betaxBeam")) < 0) {
    SDDS_SetError("Problem getting index of SDDS columns (dump_sigma)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  if (spinCoordOffset &&
      (spinIndex=SDDS_GetColumnIndex(SDDS_table, "Sspxx"))<0) {
    SDDS_SetError("Problem getting index of SDDS spin-related columns (dump_sigma)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  
  if (!SDDS_StartTable(SDDS_table, n_elements + 1)) {
    SDDS_SetError("Problem starting SDDS table (dump_sigma)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  if (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                          "Step", step, "PassLength", beamline->revolution_length, NULL)) {
    SDDS_SetError("Problem setting parameter values for SDDS table (dump_sigma)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  if (!(beamline->flags & BEAMLINE_CONCAT_DONE))
    eptr = beamline->elem;
  else
    eptr = beamline->ecat;
  name = "_BEG_";
  type_name = "MARK";
  occurence = 1;
  for (ie = 0; ie < n_elements; ie++) {
    beam = sums + ie;
    if (!beam->beamSums2)
      bombElegant("beamSums2 pointer is NULL in dump_sigma. This is a bug!", NULL);
    if (beam->n_part) {
      for (i = 0; i < 7; i++) {
        double sigmaValue = SAFE_SQRT(beam->beamSums2->sigma[i][i]);
        if (!SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, ie,
                               Sx_index + i, sigmaValue,
                               sNIndex[i], sigmaValue,
                               -1)) {
          SDDS_SetError("Problem setting SDDS row values (dump_sigma 1)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
        offset = 1;
        for (j = i + 1; j < 7; j++, offset++) {
          if (!SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, ie,
                                 sNIndex[i] + offset, beam->beamSums2->sigma[i][j], -1)) {
            SDDS_SetError("Problem setting SDDS row values (dump_sigma 2)");
            SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
          }
        }
      }
      if (spinCoordOffset) {
	for (i=offset=0; i<3; i++) {
	  for (j=i; j<3; j++, offset++) {
	    if (!SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, ie,
				   spinIndex+offset, beam->spinSums->sigma[i][j], -1)) {
              SDDS_SetError("Problem setting SDDS spin row values");
              SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
	    }
	  }
	}
      }
      /* Set values for maximum amplitudes of coordinates */
      for (i = 0; i < 7; i++) {
        if (!SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, ie, ma1_index + i, beam->beamSums2->maxabs[i], -1)) {
          SDDS_SetError("Problem setting SDDS row values (dump_sigma 3)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
        if (!SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, ie, min1_index + i, beam->beamSums2->min[i], -1)) {
          SDDS_SetError("Problem setting SDDS row values (dump_sigma 4)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
        if (!SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, ie, max1_index + i, beam->beamSums2->max[i], -1)) {
          SDDS_SetError("Problem setting SDDS row values (dump_sigma 5)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
      }
      for (plane = 0; plane <= 2; plane += 2) {
        double dummy;
        /* emittance and beam twiss parameters */
        computeEmitTwissFromSigmaMatrix(&emit, &emitc, &beta, &alpha, beam->beamSums2->sigma, plane);
        if (exactNormalizedEmittance)
          computeEmitTwissFromSigmaMatrix(&emitNorm, &emitcNorm, &dummy, &dummy, beam->beamSums2->sigman, plane);
        else {
          emitNorm = emit * beam->p0 * (1 + beam->centroid[5]);
          emitcNorm = emitc * beam->p0 * (1 + beam->centroid[5]);
        }
        if (!SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, ie,
                               ex_index + 2 * plane, emit,
                               ex_index + 1 + 2 * plane, emitNorm,
                               ex_index + 2 + 2 * plane, emitc,
                               ex_index + 3 + 2 * plane, emitcNorm,
                               betax_index + plane + 0, beta,
                               betax_index + plane + 1, alpha,
                               -1)) {
          SDDS_SetError("Problem setting SDDS row values (dump_sigma 6)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
      }
    } else {
      for (index = sNIndex[0]; index < sNIndex[0] + 28; index++)
        if (!SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, ie, index, (double)0.0, -1)) {
          SDDS_SetError("Problem setting SDDS row values (dump_sigma 7)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
      for (index = ma1_index; index < ma1_index + 4; index++)
        if (!SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, ie, index, (double)0.0, -1)) {
          SDDS_SetError("Problem setting SDDS row values (dump_sigma 8)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
      for (index = Sx_index; index < Sx_index + 7; index++)
        if (!SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, ie, index, (double)0.0, -1)) {
          SDDS_SetError("Problem setting SDDS row values (dump_sigma 9)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
      for (index = ex_index; index < ex_index + 4; index++)
        if (!SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, ie, index, (double)0.0, -1)) {
          SDDS_SetError("Problem setting SDDS row values (dump_sigma 10)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
    }

    /* coupled (eigen-)emittances and coupled lattice functions of the 6x6 beam matrix */
    if (coupledSigmaOutput) {
      COUPLED_RESULTS cr;
      long ok = 1, mode;
      char cn[32];
      memset(&cr, 0, sizeof(cr));
      if (beam->n_part) {
        /* beamSums2->sigma is (x,x',y,y',s,delta,t); use the 6x6 (x,x',y,y',s,delta)
           block.  s is already in meters, so sLongScale=1. */
        double S6[6][6];
        for (i = 0; i < 6; i++)
          for (j = 0; j < 6; j++)
            S6[i][j] = beam->beamSums2->sigma[i][j];
        ComputeCoupledParameters(&cr, S6, 1.0);
      }
      for (mode = 0; mode < 3; mode++) {
        long m1 = mode + 1;
        double enm = cr.emit[mode] * beam->p0 * (1 + beam->centroid[5]);
        sprintf(cn, "e%ld", m1);
        ok &= SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, ie, cn, cr.emit[mode], NULL);
        sprintf(cn, "en%ld", m1);
        ok &= SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, ie, cn, enm, NULL);
        sprintf(cn, "betax%ld", m1);
        ok &= SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, ie, cn, cr.betax[mode], NULL);
        sprintf(cn, "betay%ld", m1);
        ok &= SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, ie, cn, cr.betay[mode], NULL);
        sprintf(cn, "betaz%ld", m1);
        ok &= SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, ie, cn, cr.betaz[mode], NULL);
        sprintf(cn, "alphax%ld", m1);
        ok &= SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, ie, cn, cr.alphax[mode], NULL);
        sprintf(cn, "alphay%ld", m1);
        ok &= SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, ie, cn, cr.alphay[mode], NULL);
        sprintf(cn, "alphaz%ld", m1);
        ok &= SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, ie, cn, cr.alphaz[mode], NULL);
        sprintf(cn, "gammax%ld", m1);
        ok &= SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, ie, cn, cr.gammax[mode], NULL);
        sprintf(cn, "gammay%ld", m1);
        ok &= SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, ie, cn, cr.gammay[mode], NULL);
        sprintf(cn, "gammaz%ld", m1);
        ok &= SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, ie, cn, cr.gammaz[mode], NULL);
        sprintf(cn, "A_xy_%ld", m1);
        ok &= SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, ie, cn, cr.A_xy[mode], NULL);
        sprintf(cn, "A_xpy_%ld", m1);
        ok &= SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, ie, cn, cr.A_xpy[mode], NULL);
        sprintf(cn, "A_xyp_%ld", m1);
        ok &= SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, ie, cn, cr.A_xyp[mode], NULL);
        sprintf(cn, "A_xpyp_%ld", m1);
        ok &= SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, ie, cn, cr.A_xpyp[mode], NULL);
      }
      if (!ok) {
        SDDS_SetError("Problem setting coupled sigma row values (dump_sigma)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
    }

    if (!SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, ie,
                           s_index, beam->z, s_index + 1, name, s_index + 2, occurence,
                           s_index + 3, type_name, -1)) {
      SDDS_SetError("Problem setting row values for SDDS table (dump_sigma 11)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }

    name = eptr->name;
    type_name = entity_name[eptr->type];
    occurence = eptr->occurence;
    if (eptr->succ)
      eptr = eptr->succ;
    else if (beamline->elem_recirc)
      eptr = beamline->elem_recirc;
    else
      eptr = beamline->elem;
  }

  if (!SDDS_WriteTable(SDDS_table)) {
    SDDS_SetError("Unable to write sigma data (dump_sigma 12)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  if (!inhibitFileSync)
    SDDS_DoFSync(SDDS_table);
  if (!SDDS_EraseData(SDDS_table)) {
    SDDS_SetError("Unable to erase sigma data (dump_sigma 13)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
}

#define BEAM_SCATTER_PARAMETERS 8
static SDDS_DEFINITION beam_scatter_parameter[BEAM_SCATTER_PARAMETERS] = {
  {"Particles", "&parameter name=Particles, type=long, description=\"Total simulated scatted particles\" &end"},
  {"pCentral", "&parameter name=pCentral, type=double, units=\"m$be$nc\", description=\"Central momentum\" &end"},
  {"AveRate", "&parameter name=AveRate, type=double, units=\"1/s\", description=\"Average Scattering Rate between 2 TSCATTER elements of a single bunch\" &end"},
  {"NScatter", "&parameter name=NScatter, type=double, units=\"1/s\", description=\"Total scattered particles per s between 2 TSCATTER elements of a beam\" &end"},
  {"PLocalRate", "&parameter name=PLocalRate, type=double, units=\"1/s\", description=\"Piwinski's Local Rate\" &end"},
  {"SLocalRate", "&parameter name=SLocalRate, type=double, units=\"1/s\", description=\"Simulated Local Rate\" &end"},
  {"IgnoredRate", "&parameter name=IgnoredRate, type=double, units=\"1/s\", description=\"Ignored Scattering Rate\" &end"},
  {"SVNVersion", "&parameter name=SVNVersion, type=string, description=\"SVN version number\", fixed_value=" SVN_VERSION " &end"},
};
#define BEAM_SCATTER_COLUMNS 9
static SDDS_DEFINITION beam_scatter_column[BEAM_SCATTER_COLUMNS] = {
  {"x", "&column name=x, units=m, type=double &end"},
  {"xp", "&column name=xp, symbol=\"x'\", type=double &end"},
  {"y", "&column name=y, units=m, type=double &end"},
  {"yp", "&column name=yp, symbol=\"y'\", type=double &end"},
  {"t", "&column name=t, units=s, type=double &end"},
  {"p", "&column name=p, units=\"m$be$nc\", type=double &end"},
  {"particleID", "&column name=particleID, type=ulong64 &end"},
  {"LRate", "&column name=LRate, units=\"1/s\", type=double, description=\"represents local scattering rate\" &end"},
  {"TRate", "&column name=TRate, units=\"1/s\", type=double, description=\"represents total scattering rate between 2 TSCATTER elements of a beam\" &end"},
};

void SDDS_BeamScatterSetup(SDDS_TABLE *SDDS_table, char *filename, long mode, long lines_per_row, char *contents,
                           char *command_file, char *lattice_file, char *caller) {
  log_entry("SDDS_BeamScatterSetup");

  SDDS_ElegantOutputSetup(SDDS_table, filename, mode, lines_per_row, contents, command_file, lattice_file,
                          beam_scatter_parameter, BEAM_SCATTER_PARAMETERS, beam_scatter_column, BEAM_SCATTER_COLUMNS,
                          caller, SDDS_EOS_NEWFILE | SDDS_EOS_COMPLETE);

  log_exit("SDDS_BeamScatterSetup");
}

void dump_scattered_particles(SDDS_TABLE *SDDS_table, double **particle,
                              long particles, double *weight, TSCATTER *tsptr) {
  long i;
  double rate;

  log_entry("dump_scattered_particles");
  if (!particle)
    bombElegant("NULL coordinate pointer passed to dump_scattered_particles", NULL);

  for (i = 0; i < particles; i++)
    if (!particle[i]) {
      printf("error: coordinate slot %ld is NULL (dump_scattered_particles)\n", i);
      fflush(stdout);
      abort();
    }

  if (!SDDS_StartTable(SDDS_table, particles)) {
    SDDS_SetError("Problem starting SDDS table (dump_scattered_particles)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  for (i = 0; i < particles; i++) {
    rate = weight[i] / tsptr->s_rate * tsptr->total_scatter;
    if (!SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, i,
                           0, particle[i][0], 1, particle[i][1], 2, particle[i][2], 3, particle[i][3],
                           4, particle[i][4] / c_mks, 5, (particle[i][5] + 1.) * tsptr->betagamma,
                           6, (uint64_t)particle[i][6], 7, weight[i], 8, rate, -1)) {
      SDDS_SetError("Problem setting SDDS row values (dump_scattered_particles)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  }

  if ((!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "Particles", particles, NULL)) ||
      (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "pCentral", tsptr->betagamma, NULL)) ||
      (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "AveRate", tsptr->AveR, NULL)) ||
      (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "NScatter", tsptr->total_scatter, NULL)) ||
      (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "PLocalRate", tsptr->p_rate, NULL)) ||
      (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "SLocalRate", tsptr->s_rate, NULL)) ||
      (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "IgnoredRate", tsptr->i_rate, NULL))) {
    SDDS_SetError("Problem setting SDDS parameters (dump_scattered_particles)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  if (!SDDS_WriteTable(SDDS_table)) {
    SDDS_SetError("Problem writing SDDS table (dump_scattered_particles)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  if (!inhibitFileSync)
    SDDS_DoFSync(SDDS_table);
  log_exit("dump_scattered_particles");
}

#define BEAM_SCATTER_LOSS_X0 0
#define BEAM_SCATTER_LOSS_X1 (BEAM_SCATTER_LOSS_X0 + 8)
#define BEAM_SCATTER_LOSS_PASS (BEAM_SCATTER_LOSS_X1 + 6)
#define BEAM_SCATTER_LOSS_COLUMNS 17
static SDDS_DEFINITION beam_scatter_loss_column[BEAM_SCATTER_LOSS_COLUMNS] = {
  {"x0", "&column name=x0, units=m, type=double &end"},
  {"xp0", "&column name=xp0, symbol=\"x'\", type=double &end"},
  {"y0", "&column name=y0, units=m, type=double &end"},
  {"yp0", "&column name=yp0, symbol=\"y'\", type=double &end"},
  {"t0", "&column name=t0, units=s, type=double &end"},
  {"p0", "&column name=p0, units=\"m$be$nc\", type=double &end"},
  {"particleID", "&column name=particleID, type=ulong64 &end"},
  {"s0", "&column name=s0, units=m, type=double &end"},
  {"x", "&column name=x, units=m, type=double &end"},
  {"xp", "&column name=xp, type=double &end"},
  {"y", "&column name=y, units=m, type=double &end"},
  {"yp", "&column name=yp, units=m, type=double &end"},
  {"s", "&column name=s, units=m, type=double &end"},
  {"p", "&column name=p, units=\"m$be$nc\", type=double &end"},
  {"Pass", "&column name=Pass, type=long &end"},
  {"LRate", "&column name=LRate, units=\"1/s\", type=double, description=\"represents local scattering rate\" &end"},
  {"TRate", "&column name=TRate, units=\"1/s\", type=double, description=\"represents total scattering rate between 2 TSCATTER elements of a beam\" &end"},
};
#define BEAM_SCATTER_LOSS_PARAMETERS 16
static SDDS_DEFINITION beam_scatter_loss_parameter[BEAM_SCATTER_LOSS_PARAMETERS] = {
  {"Particles", "&parameter name=Particles, type=long, description=\"Total lost simulated scattered particles\" &end"},
  {"S0", "&parameter name=S0, type=double, units=m, description=\"Scatter location\" &end"},
  {"ElementName", "&parameter name=ElementName, type=string &end"},
  {"ElementOccurence", "&parameter name=ElementOccurence, type=long &end"},
  {"PreviousElementName", "&parameter name=PreviousElementName, type=string &end"},
  {"PreviousElementOccurence", "&parameter name=PreviousElementOccurence, type=long &end"},
  {"NextElementName", "&parameter name=NextElementName, type=string &end"},
  {"NextElementOccurence", "&parameter name=NextElementOccurence, type=long &end"},
  {"PLocalRate", "&parameter name=PLocalRate, type=double, units=\"1/s\", description=\"Piwinski's Local Rate\" &end"},
  {"SLocalRate", "&parameter name=SLocalRate, type=double, units=\"1/s\", description=\"Simulated Local Rate\" &end"},
  {"IgnoredRate", "&parameter name=IgnoredRate, type=double, units=\"1/s\", description=\"Ignored Scattering Rate\" &end"},
  {"AveRate", "&parameter name=AveRate, type=double, units=\"1/s\", description=\"Average Scattering Rate\" &end"},
  {"NScatter", "&parameter name=NScatter, type=double, units=\"1/s\", description=\"Total scattered particles/s between 2 TSCATTER elements for a beam\" &end"},
  {"TotalLoss", "&parameter name=TotalLoss, type=double, units=\"1/s\", description=\"Total lost scattered particles/s between 2 TSCATTER elements for a beam\" &end"},
  {"pCentral", "&parameter name=pCentral, type=double, units=\"m$be$nc\", description=\"Central momentum\" &end"},
  {"SVNVersion", "&parameter name=SVNVersion, type=string, description=\"SVN version number\", fixed_value=" SVN_VERSION " &end"},
};

void SDDS_BeamScatterLossSetup(SDDS_TABLE *SDDS_table, char *filename, long mode, long lines_per_row, char *contents,
                               char *command_file, char *lattice_file, char *caller) {
  log_entry("SDDS_BeamScatterLossSetup");

  SDDS_ElegantOutputSetup(SDDS_table, filename, mode, lines_per_row, contents, command_file, lattice_file,
                          beam_scatter_loss_parameter, BEAM_SCATTER_LOSS_PARAMETERS,
                          beam_scatter_loss_column, BEAM_SCATTER_LOSS_COLUMNS,
                          caller, SDDS_EOS_NEWFILE | SDDS_EOS_COMPLETE);
  log_exit("SDDS_BeamScatterLossSetup");
}

void dump_scattered_loss_particles(SDDS_TABLE *SDDS_table, double **particleLos, double **particleOri,
                                   long *lostOnPass, long particles, double *weight, TSCATTER *tsptr, ELEMENT_LIST *eptr) {
  long i, j;
  double rate, intLossRate;

#ifdef DEBUG
  printf("dump_scattered_loss_particles: particles = %ld\n", particles);
#endif

  log_entry("dump_scattered_loss_particles");
  if (!particleLos)
    bombElegant("NULL coordinate pointer passed to dump_scattered_loss_particles", NULL);

  for (i = 0; i < particles; i++)
    if (!particleLos[i]) {
      printf("error: coordinate slot %ld is NULL (dump_scattered_loss_particles)\n", i);
      fflush(stdout);
      abort();
    }

  if (!SDDS_StartTable(SDDS_table, particles)) {
    SDDS_SetError("Problem starting SDDS table (dump_scattered_loss_particles)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  intLossRate = 0.;
  for (i = 0; i < particles; i++) {
    j = (long)particleLos[i][6] - 1;
#ifdef DEBUG
    printf("Lost particle %ld came from source %ld\n", i, j);
    fflush(stdout);
#endif
    rate = weight[j] / tsptr->s_rate * tsptr->total_scatter;
    intLossRate += rate;
    if (!SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, i,
                           BEAM_SCATTER_LOSS_X0, particleOri[j][0],
                           BEAM_SCATTER_LOSS_X0 + 1, particleOri[j][1],
                           BEAM_SCATTER_LOSS_X0 + 2, particleOri[j][2],
                           BEAM_SCATTER_LOSS_X0 + 3, particleOri[j][3],
                           BEAM_SCATTER_LOSS_X0 + 4, particleOri[j][4] / c_mks,
                           BEAM_SCATTER_LOSS_X0 + 5, (1. + particleOri[j][5]) * tsptr->betagamma,
                           BEAM_SCATTER_LOSS_X0 + 6, (uint64_t)particleOri[j][6],
                           BEAM_SCATTER_LOSS_X0 + 7, tsptr->s,
                           BEAM_SCATTER_LOSS_X1 + 0, particleLos[i][0],
                           BEAM_SCATTER_LOSS_X1 + 1, particleLos[i][1],
                           BEAM_SCATTER_LOSS_X1 + 2, particleLos[i][2],
                           BEAM_SCATTER_LOSS_X1 + 3, particleLos[i][3],
                           BEAM_SCATTER_LOSS_X1 + 4, particleLos[i][4],
                           BEAM_SCATTER_LOSS_X1 + 5, particleLos[i][5],
                           BEAM_SCATTER_LOSS_PASS + 0, (long)particleLos[i][7],
                           BEAM_SCATTER_LOSS_PASS + 1, weight[j],
                           BEAM_SCATTER_LOSS_PASS + 2, rate, -1)) {
      SDDS_SetError("Problem setting SDDS row values (dump_scattered_loss_particles)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  }
#ifdef DEBUG
  printf("Done storing lost particle coordinates\n");
  fflush(stdout);
#endif
  if ((!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "Particles", particles, NULL)) ||
      (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "S0", tsptr->s, NULL)) ||
      (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "ElementName", eptr->name, NULL)) ||
      (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "ElementOccurence", eptr->occurence, NULL)) ||
      (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "PreviousElementName", 
                           eptr->pred?eptr->pred->name:"_BEGIN_", NULL)) ||
      (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "PreviousElementOccurence", 
                           eptr->pred?eptr->pred->occurence:-1, NULL)) ||
      (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "NextElementName", 
                           eptr->succ?eptr->succ->name:"_BEGIN_", NULL)) ||
      (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "NextElementOccurence", 
                           eptr->succ?eptr->succ->occurence:-1, NULL)) ||
      (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "PLocalRate", tsptr->p_rate, NULL)) ||
      (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "SLocalRate", tsptr->s_rate, NULL)) ||
      (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "IgnoredRate", tsptr->i_rate, NULL)) ||
      (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "NScatter", tsptr->total_scatter, NULL)) ||
      (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "AveRate", tsptr->AveR, NULL)) ||
      (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "TotalLoss", intLossRate, NULL)) ||
      (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "pCentral", tsptr->betagamma, NULL))) {
    SDDS_SetError("Problem setting SDDS parameters (dump_scattered_loss_particles)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  if (!SDDS_WriteTable(SDDS_table)) {
    SDDS_SetError("Problem writing SDDS table (dump_scattered_loss_particles)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  if (!inhibitFileSync)
    SDDS_DoFSync(SDDS_table);

#ifdef DEBUG
  printf("Returning from dump_scattered_loss_particles\n");
  fflush(stdout);
#endif

  log_exit("dump_scattered_loss_particles");
}

void computeEmitTwissFromSigmaMatrix(double *emit, double *emitc, double *beta, double *alpha, double sigma[7][7], long plane) {
  if (plane < 0 || plane > 4 || plane % 2 != 0)
    bombElegant("invalid value for plane in computeEmitTwissFromSigmaMatrix", NULL);
  if (!emit || (!emitc && (alpha || beta)))
    bombElegant("invalid pointers passed to computeEmitTwissFromSigmaMatrix", NULL);

  /* emittance */
  *emit = SAFE_SQRT(sigma[0 + plane][0 + plane] * sigma[1 + plane][1 + plane] - sqr(sigma[0 + plane][1 + plane]));

  if (emitc) {
    /* corrected emittance */
    if (sigma[5][5] && plane != 4)
      *emitc = SAFE_SQRT(sqr(*emit) -
                         (sqr(sigma[0 + plane][5]) * sigma[1 + plane][1 + plane] -
                          2 * sigma[0 + plane][1 + plane] * sigma[0 + plane][5] * sigma[1 + plane][5] +
                          sqr(sigma[1 + plane][5]) * sigma[0 + plane][0 + plane]) /
                           sigma[5][5]);
    else
      *emitc = *emit;

    /* beta and alpha */
    if (beta && alpha) {
      *beta = *alpha = 0;
      if (*emitc) {
        if (sigma[5][5] && plane != 4) {
          double s11c, s12c;
          s11c = sigma[0 + plane][0 + plane] - sqr(sigma[0 + plane][5]) / sigma[5][5];
          s12c = sigma[0 + plane][1 + plane] - sigma[0 + plane][5] * sigma[1 + plane][5] / sigma[5][5];
          *beta = s11c / (*emitc);
          *alpha = -s12c / (*emitc);
        } else {
          *beta = sigma[0 + plane][0 + plane] / (*emitc);
          *alpha = -sigma[0 + plane][1 + plane] / (*emitc);
        }
      }
    }
  }
}
