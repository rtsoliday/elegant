/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: final_props.c
 * purpose: routines used to compute and output values for
 *          'final' file.
 * M. Borland, 1993.
 */

#include "mdb.h"
//#include "mdbsun.h"
#include "track.h"
#include "matlib.h"

#define ANALYSIS_BINS 10000

/* For time and momentum percentiles in final properties output.
 * Warning: increasing this number may seriously impact performance of
 * optimizations.
 */
#define ANALYSIS_BINS2 10000

static double tmp_safe_sqrt;
#define SAFE_SQRT(x) ((tmp_safe_sqrt = (x)) < 0 ? 0.0 : sqrt(tmp_safe_sqrt))

#define FINAL_PROPERTY_PARAMETERS (96 + 9 + 4 + 6 + 1 + 2 + 7 + 3 + 2 * 7)
#define FINAL_PROPERTY_LONG_PARAMETERS 5
#define FINAL_PROPERTY_OTHER_PARAMETERS 1
#define F_SIGMA_OFFSET 0
#define F_SIGMA_QUANS 7
#define F_CENTROID_OFFSET F_SIGMA_OFFSET + F_SIGMA_QUANS
#define F_CENTROID_QUANS 7
#define F_SIGMAT_OFFSET F_CENTROID_OFFSET + F_CENTROID_QUANS
#define F_SIGMAT_QUANS 22
#define F_T_OFFSET F_SIGMAT_OFFSET + F_SIGMAT_QUANS
#define F_T_QUANS 5
#define F_EMIT_OFFSET F_T_OFFSET + F_T_QUANS
#define F_EMIT_QUANS 5
#define F_NEMIT_OFFSET F_EMIT_OFFSET + F_EMIT_QUANS
#define F_NEMIT_QUANS 4
#define F_MINVALUE_OFFSET F_NEMIT_OFFSET + F_NEMIT_QUANS
#define F_MINVALUE_QUANS 7
#define F_MAXVALUE_OFFSET F_MINVALUE_OFFSET + F_MINVALUE_QUANS
#define F_MAXVALUE_QUANS 7
#define F_MAXABS_OFFSET F_MAXVALUE_OFFSET + F_MAXVALUE_QUANS
#define F_MAXABS_QUANS 7
#define F_WIDTH_OFFSET F_MAXABS_OFFSET + F_MAXABS_QUANS
#define F_WIDTH_QUANS 16
#define F_PERC_OFFSET F_WIDTH_OFFSET + F_WIDTH_QUANS
#define F_PERC_QUANS 9
#define F_RMAT_OFFSET F_PERC_OFFSET + F_PERC_QUANS
#define F_RMAT_QUANS 37
#define F_STATS_OFFSET F_RMAT_OFFSET + F_RMAT_QUANS
#define F_STATS_QUANS 7
#define F_N_OFFSET F_STATS_OFFSET + F_STATS_QUANS
#define F_N_QUANS 2
#if (F_N_QUANS + F_N_OFFSET) != FINAL_PROPERTY_PARAMETERS
#  error "FINAL_PROPERTY_PARAMETERS is inconsistent with parameter offsets"
#endif

static SDDS_DEFINITION final_property_parameter[FINAL_PROPERTY_PARAMETERS] = {
  /* beginning of type=double parameters */
  {"Sx", "&parameter name=Sx, symbol=\"$gs$r$bx$n\", units=m, type=double, description=\"sqrt(<(x-<x>)^2>)\"  &end"},
  {"Sxp", "&parameter name=Sxp, symbol=\"$gs$r$bx'$n\", type=double, description=\"sqrt(<(x'-<x'>)^2>)\"  &end"},
  {"Sy", "&parameter name=Sy, symbol=\"$gs$r$by$n\", units=m, type=double, description=\"sqrt(<(y-<y>)^2>)\"  &end"},
  {"Syp", "&parameter name=Syp, symbol=\"$gs$r$by'$n\", type=double, description=\"sqrt(<(y'-<y'>)^2>)\"  &end"},
  {"Ss", "&parameter name=Ss, units=m, symbol=\"$gs$r$bs$n\", type=double, description=\"sqrt(<(s-<s>)^2>)\"  &end"},
  {"Sdelta", "&parameter name=Sdelta, symbol=\"$gs$bd$n$r\", type=double, description=\"sqrt(<(delta-<delta>)^2>)\"  &end"},
  {"St", "&parameter name=St, symbol=\"$gs$r$bt$n\", type=double, units=s,, description=\"sqrt(<(t-<t>)^2>)\"  &end"},
  {"Cx", "&parameter name=Cx, symbol=\"<x>\", units=m, type=double, description=\"x centroid\" &end"},
  {"Cxp", "&parameter name=Cxp, symbol=\"<x'>\", type=double, description=\"x' centroid\" &end"},
  {"Cy", "&parameter name=Cy, symbol=\"<y>\", units=m, type=double, description=\"y centroid\" &end"},
  {"Cyp", "&parameter name=Cyp, symbol=\"<y'>\", type=double, description=\"y' centroid\" &end"},
  {"Cs", "&parameter name=Cs, symbol=\"<s>\", units=m, type=double, description=\"mean distance traveled\" &end"},
  {"Cdelta", "&parameter name=Cdelta, symbol=\"<$gd$r>\", type=double, description=\"delta centroid\" &end"},
  {"Ct", "&parameter name=Ct, symbol=\"<t>\", units=s, type=double, description=\"mean time of flight\" &end"},
  {"s12", "&parameter name=s12, symbol=\"$gs$r$b12$n\", units=m, type=double, description=\"<x*xp'>\" &end"},
  {"s13", "&parameter name=s13, symbol=\"$gs$r$b13$n\", units=\"m$a2$n\", type=double, description=\"<x*y>\" &end"},
  {"s14", "&parameter name=s14, symbol=\"$gs$r$b14$n\", units=m, type=double, description=\"<x*y'>\" &end"},
  {"s15", "&parameter name=s15, symbol=\"$gs$r$b15$n\", units=\"m^a2$n\", type=double, description=\"<x*s>\" &end"},
  {"s16", "&parameter name=s16, symbol=\"$gs$r$b16$n\", units=m, type=double, description=\"<x*delta>\" &end"},
  {"s17", "&parameter name=s17, symbol=\"$gs$r$b17$n\", units=m*s, type=double, description=\"<x*t>\" &end"},
  {"s23", "&parameter name=s23, symbol=\"$gs$r$b23$n\", units=m, type=double, description=\"<x'*y>\" &end"},
  {"s24", "&parameter name=s24, symbol=\"$gs$r$b24$n\", type=double, description=\"<x'*y'>\" &end"},
  {"s25", "&parameter name=s25, symbol=\"$gs$r$b25$n\", units=m, type=double, description=\"<x'*s>\" &end"},
  {"s26", "&parameter name=s26, symbol=\"$gs$r$b26$n\", type=double, description=\"<x'*delta>\" &end"},
  {"s27", "&parameter name=s27, symbol=\"$gs$r$b27$n\", units=s, type=double, description=\"<x'*t>\" &end"},
  {"s34", "&parameter name=s34, symbol=\"$gs$r$b34$n\", units=m, type=double, description=\"<y*y'>\" &end"},
  {"s35", "&parameter name=s35, symbol=\"$gs$r$b35$n\", units=\"m^a2$n\", type=double, description=\"<y*s>\" &end"},
  {"s36", "&parameter name=s36, symbol=\"$gs$r$b36$n\", units=m, type=double, description=\"<y*delta>\" &end"},
  {"s37", "&parameter name=s37, symbol=\"$gs$r$b37$n\", units=m*s, type=double, description=\"<y*t>\" &end"},
  {"s45", "&parameter name=s45, symbol=\"$gs$r$b45$n\", units=m, type=double, description=\"<y'*s>\" &end"},
  {"s46", "&parameter name=s46, symbol=\"$gs$r$b46$n\", type=double, description=\"<s'*delta>\" &end"},
  {"s47", "&parameter name=s47, symbol=\"$gs$r$b47$n\", units=s, type=double, description=\"<y'*t>\" &end"},
  {"s56", "&parameter name=s56, symbol=\"$gs$r$b56$n\", units=m, type=double, description=\"<s*delta>\" &end"},
  {"s57", "&parameter name=s57, symbol=\"$gs$r$b57$n\", units=m*s, type=double, description=\"<s*t>\" &end"},
  {"s67", "&parameter name=s67, symbol=\"$gs$r$b67$n\", units=s, type=double, description=\"<delta*t>\" &end"},
  {"s7", "&parameter name=s7, symbol=\"$gs$r$b7$n\", type=double, description=\"sqrt(<t*t>)\" &end"},
  {"Transmission", "&parameter name=Transmission, description=Transmission, type=double &end"},
  {"pCentral", "&parameter name=pCentral, symbol=\"p$bcen$n\", units=\"m$be$nc\", type=double, description=\"Reference beta*gamma\" &end"},
  {"pAverage", "&parameter name=pAverage, symbol=\"p$bave$n\", units=\"m$be$nc\", type=double, description=\"Mean beta*gamma\" &end"},
  {"KAverage", "&parameter name=KAverage, symbol=\"K$bave$n\", units=MeV, type=double, description=\"Mean kinetic energy\" &end"},
  {"Charge", "&parameter name=Charge, units=C, type=double, description=\"Beam charge\" &end"},
  {"ex", "&parameter name=ex, symbol=\"$ge$r$bx$n\", units=m, type=double, description=\"geometric horizontal emittance\" &end"},
  {"ey", "&parameter name=ey, symbol=\"$ge$r$by$n\", units=m, type=double, description=\"geometric vertical emittance\" &end"},
  {"ecx", "&parameter name=ecx, symbol=\"$ge$r$bx,c$n\", units=m, type=double, description=\"geometric horizontal emittance less dispersive contributions\" &end"},
  {"ecy", "&parameter name=ecy, symbol=\"$ge$r$by,c$n\", units=m, type=double, description=\"geometric vertical emittance less dispersive contributions\" &end"},
  {"el", "&parameter name=el, symbol=\"$ge$r$bl$n\", units=s, type=double &end"},
  {"enx", "&parameter name=enx, symbol=\"$ge$r$bx,n$n\", type=double, units=m, description=\"normalized horizontal emittance\"  &end"},
  {"eny", "&parameter name=eny, symbol=\"$ge$r$by,n$n\", type=double, units=m, description=\"normalized vertical emittance\" &end"},
  {"ecnx", "&parameter name=ecnx, symbol=\"$ge$r$bx,cn$n\", type=double, units=m, description=\"normalized horizontal emittance less dispersive contributions\" &end"},
  {"ecny", "&parameter name=ecny, symbol=\"$ge$r$by,cn$n\", type=double, units=m, description=\"normalized vertical emittance less dispersive contributions\" &end"},
  {"xMinimum", "&parameter name=xMinimum, type=double, units=m, description=\"minimum value of x\" &end"},
  {"xpMinimum", "&parameter name=xpMinimum, type=double,, description=\"minimum value of x'\" &end"},
  {"yMinimum", "&parameter name=yMinimum, type=double, units=m, description=\"minimum value of y\" &end"},
  {"ypMinimum", "&parameter name=ypMinimum, type=double, description=\"minimum value of y'\" &end"},
  {"dsMinimum", "&parameter name=dsMinimum, type=double, units=m, description=\"minimum value of ds\" &end"},
  {"deltaMinimum", "&parameter name=deltaMinimum, type=double, description=\"minimum value of delta\" &end"},
  {"dtMinimum", "&parameter name=dtMinimum, type=double, units=s, description=\"minimum value of dt\" &end"},
  {"xMaximum", "&parameter name=xMaximum, type=double, units=m, description=\"maximum value of x\" &end"},
  {"xpMaximum", "&parameter name=xpMaximum, type=double,, description=\"maximum value of x'\" &end"},
  {"yMaximum", "&parameter name=yMaximum, type=double, units=m, description=\"maximum value of y\" &end"},
  {"ypMaximum", "&parameter name=ypMaximum, type=double, description=\"maximum value of y'\" &end"},
  {"sMaximum", "&parameter name=sMaximum, type=double, units=m, description=\"maximum value of ds\" &end"},
  {"deltaMaximum", "&parameter name=deltaMaximum, type=double, description=\"maximum value of delta\" &end"},
  {"dtMaximum", "&parameter name=dtMaximum, type=double, units=s, description=\"maximum value of dt\" &end"},
  {"ma1", "&parameter name=ma1, type=double, units=m, description=\"maximum absolute value of x\" &end"},
  {"ma2", "&parameter name=ma2, type=double, description=\"maximum absolute value of x'\" &end"},
  {"ma3", "&parameter name=ma3, type=double, units=m, description=\"maximum absolute value of y\" &end"},
  {"ma4", "&parameter name=ma4, type=double, description=\"maximum absolute value of y'\" &end"},
  {"ma5", "&parameter name=ma5, type=double, units=m, description=\"maximum absolute value of ds\" &end"},
  {"ma6", "&parameter name=ma6, type=double, description=\"maximum absolute value of delta\" &end"},
  {"ma7", "&parameter name=ma7, type=double, units=s, description=\"maximum absolute value of dt\" &end"},
  {"Wx", "&parameter name=Wx, type=double, units=m, symbol=\"W$bx$n\", description=\"68.26% width in x\" &end"},
  {"Wy", "&parameter name=Wy, type=double, units=m, symbol=\"W$by$n\", description=\"68.26% width in y\" &end"},
  {"Dt", "&parameter name=Dt, type=double, units=s, symbol=\"$gD$rt\", description=\"Total time-of-flight span\" &end"},
  {"Ddelta", "&parameter name=Ddelta, type=double, symbol=\"$gDd$r\", description=\"Total delta span\" &end"},
  {"Dt50", "&parameter name=Dt50, type=double, units=s, symbol=\"$gD$rt$b50$n\",, description=\"50% time-of-flight span\" &end"},
  {"Dt60", "&parameter name=Dt60, type=double, units=s, symbol=\"$gD$rt$b60$n\",, description=\"60% time-of-flight span\" &end"},
  {"Dt70", "&parameter name=Dt70, type=double, units=s, symbol=\"$gD$rt$b70$n\",, description=\"70% time-of-flight span\" &end"},
  {"Dt80", "&parameter name=Dt80, type=double, units=s, symbol=\"$gD$rt$b80$n\",, description=\"80% time-of-flight span\" &end"},
  {"Dt90", "&parameter name=Dt90, type=double, units=s, symbol=\"$gD$rt$b90$n\",, description=\"90% time-of-flight span\" &end"},
  {"Dt95", "&parameter name=Dt95, type=double, units=s, symbol=\"$gD$rt$b95$n\",, description=\"95% time-of-flight span\" &end"},
  {"Ddelta50", "&parameter name=Ddelta50, type=double, symbol=\"$gDd$r$b50$n\",, description=\"50% delta span\" &end"},
  {"Ddelta60", "&parameter name=Ddelta60, type=double, symbol=\"$gDd$r$b60$n\",, description=\"60% delta span\" &end"},
  {"Ddelta70", "&parameter name=Ddelta70, type=double, symbol=\"$gDd$r$b70$n\",, description=\"70% delta span\" &end"},
  {"Ddelta80", "&parameter name=Ddelta80, type=double, symbol=\"$gDd$r$b80$n\",, description=\"80% delta span\" &end"},
  {"Ddelta90", "&parameter name=Ddelta90, type=double, symbol=\"$gDd$r$b90$n\",, description=\"90% delta span\" &end"},
  {"Ddelta95", "&parameter name=Ddelta95, type=double, symbol=\"$gDd$r$b95$n\",, description=\"95% delta span\" &end"},
  {"DtPerc10", "&parameter name=DtPerc10, type=double, units=s,, description=\"Time offset to 10% point\" &end"},
  {"DtPerc20", "&parameter name=DtPerc20, type=double, units=s,, description=\"Time offset to 20% point\" &end"},
  {"DtPerc30", "&parameter name=DtPerc30, type=double, units=s,, description=\"Time offset to 30% point\" &end"},
  {"DtPerc40", "&parameter name=DtPerc40, type=double, units=s,, description=\"Time offset to 40% point\" &end"},
  {"DtPerc50", "&parameter name=DtPerc50, type=double, units=s,, description=\"Time offset to 50% point\" &end"},
  {"DtPerc60", "&parameter name=DtPerc60, type=double, units=s,, description=\"Time offset to 60% point\" &end"},
  {"DtPerc70", "&parameter name=DtPerc70, type=double, units=s,, description=\"Time offset to 70% point\" &end"},
  {"DtPerc80", "&parameter name=DtPerc80, type=double, units=s,, description=\"Time offset to 80% point\" &end"},
  {"DtPerc90", "&parameter name=DtPerc90, type=double, units=s,, description=\"Time offset to 90% point\" &end"},
  {"R11", "&parameter name=R11, type=double, symbol=\"R$b11$n\", description=\"Transport matrix element from start of system\" &end"},
  {"R12", "&parameter name=R12, type=double, units=m, symbol=\"R$b12$n\", description=\"Transport matrix element from start of system\" &end"},
  {"R13", "&parameter name=R13, type=double, symbol=\"R$b13$n\", description=\"Transport matrix element from start of system\" &end"},
  {"R14", "&parameter name=R14, type=double, units=m, symbol=\"R$b14$n\", description=\"Transport matrix element from start of system\" &end"},
  {"R15", "&parameter name=R15, type=double, units=m, symbol=\"R$b15$n\", description=\"Transport matrix element from start of system\" &end"},
  {"R16", "&parameter name=R16, type=double, units=m, symbol=\"R$b16$n\", description=\"Transport matrix element from start of system\" &end"},
  {"R21", "&parameter name=R21, type=double, units=1/m, symbol=\"R$b21$n\", description=\"Transport matrix element from start of system\" &end"},
  {"R22", "&parameter name=R22, type=double, symbol=\"R$b22$n\", description=\"Transport matrix element from start of system\" &end"},
  {"R23", "&parameter name=R23, type=double, units=1/m, symbol=\"R$b23$n\", description=\"Transport matrix element from start of system\" &end"},
  {"R24", "&parameter name=R24, type=double, symbol=\"R$b24$n\", description=\"Transport matrix element from start of system\" &end"},
  {"R25", "&parameter name=R25, type=double, symbol=\"R$b25$n\", description=\"Transport matrix element from start of system\" &end"},
  {"R26", "&parameter name=R26, type=double, symbol=\"R$b26$n\", description=\"Transport matrix element from start of system\" &end"},
  {"R31", "&parameter name=R31, type=double, symbol=\"R$b31$n\", description=\"Transport matrix element from start of system\" &end"},
  {"R32", "&parameter name=R32, type=double, units=m, symbol=\"R$b32$n\", description=\"Transport matrix element from start of system\" &end"},
  {"R33", "&parameter name=R33, type=double, symbol=\"R$b33$n\", description=\"Transport matrix element from start of system\" &end"},
  {"R34", "&parameter name=R34, type=double, units=m, symbol=\"R$b34$n\", description=\"Transport matrix element from start of system\" &end"},
  {"R35", "&parameter name=R35, type=double, units=m, symbol=\"R$b35$n\", description=\"Transport matrix element from start of system\" &end"},
  {"R36", "&parameter name=R36, type=double, units=m, symbol=\"R$b36$n\", description=\"Transport matrix element from start of system\" &end"},
  {"R41", "&parameter name=R41, type=double, units=1/m, symbol=\"R$b41$n\", description=\"Transport matrix element from start of system\" &end"},
  {"R42", "&parameter name=R42, type=double, symbol=\"R$b42$n\", description=\"Transport matrix element from start of system\" &end"},
  {"R43", "&parameter name=R43, type=double, units=1/m, symbol=\"R$b43$n\", description=\"Transport matrix element from start of system\" &end"},
  {"R44", "&parameter name=R44, type=double, symbol=\"R$b44$n\", description=\"Transport matrix element from start of system\" &end"},
  {"R45", "&parameter name=R45, type=double, symbol=\"R$b45$n\", description=\"Transport matrix element from start of system\" &end"},
  {"R46", "&parameter name=R46, type=double, symbol=\"R$b46$n\", description=\"Transport matrix element from start of system\" &end"},
  {"R51", "&parameter name=R51, type=double, symbol=\"R$b51$n\", description=\"Transport matrix element from start of system\" &end"},
  {"R52", "&parameter name=R52, type=double, units=m, symbol=\"R$b52$n\", description=\"Transport matrix element from start of system\" &end"},
  {"R53", "&parameter name=R53, type=double, symbol=\"R$b53$n\", description=\"Transport matrix element from start of system\" &end"},
  {"R54", "&parameter name=R54, type=double, units=m, symbol=\"R$b54$n\", description=\"Transport matrix element from start of system\" &end"},
  {"R55", "&parameter name=R55, type=double, symbol=\"R$b55$n\", description=\"Transport matrix element from start of system\" &end"},
  {"R56", "&parameter name=R56, type=double, units=m, symbol=\"R$b56$n\", description=\"Transport matrix element from start of system\" &end"},
  {"R61", "&parameter name=R61, type=double, units=1/m, symbol=\"R$b61$n\", description=\"Transport matrix element from start of system\" &end"},
  {"R62", "&parameter name=R62, type=double, symbol=\"R$b62$n\", description=\"Transport matrix element from start of system\" &end"},
  {"R63", "&parameter name=R63, type=double, units=1/m, symbol=\"R$b63$n\", description=\"Transport matrix element from start of system\" &end"},
  {"R64", "&parameter name=R64, type=double, symbol=\"R$b64$n\", description=\"Transport matrix element from start of system\" &end"},
  {"R65", "&parameter name=R65, units=1/m, type=double, symbol=\"R$b65$n\", description=\"Transport matrix element from start of system\" &end"},
  {"R66", "&parameter name=R66, type=double, symbol=\"R$b66$n\", description=\"Transport matrix element from start of system\" &end"},
  {"detR", "&parameter name=detR, type=double, symbol=\"det R\", description=\"Determinant of transport matrix from start of system\" &end"},
  {"CPU", "&parameter name=CPU, type=double, units=s &end"},
  {"ElapsedTime", "&parameter name=ElapsedTime, type=double, units=s &end"},
  {"ElapsedCoreTime", "&parameter name=ElapsedCoreTime, type=double, units=s &end"},
  /* beginning of type=long parameters */
  {"MEM", "&parameter name=MEM, type=long, units=pages &end"},
  {"PF", "&parameter name=PF, type=long, units=pages &end"},
  {"Step", "&parameter name=Step, type=long &end"},
  {"Steps", "&parameter name=Steps, type=long &end"},
  {"Particles", "&parameter name=Particles, description=\"Number of particles\", type=long &end"},
  /* other */
  {"SVNVersion", "&parameter name=SVNVersion, type=string, description=\"SVN version number\", fixed_value=" SVN_VERSION " &end"},
};

void SDDS_FinalOutputSetup(SDDS_TABLE *SDDS_table, char *filename, long mode, long lines_per_row,
                           char *contents, char *command_file, char *lattice_file,
                           char **varied_quantity_name, char **varied_quantity_unit, long varied_quantities,
                           char **error_element_name, char **error_element_unit, long error_elements,
                           long *error_element_index, long *error_element_duplicates,
                           char **optimization_quantity_name, char **optimization_quantity_unit,
                           long optimization_quantities,
                           char *caller) {
  long i, duplicates = 0;
#if USE_MPI
  if (myid != 0)
    return;
  SDDS_table->parallel_io = 0;
#endif

  SDDS_ElegantOutputSetup(SDDS_table, filename, mode, lines_per_row, contents, command_file,
                          lattice_file, final_property_parameter, FINAL_PROPERTY_PARAMETERS, NULL, 0,
                          caller, SDDS_EOS_NEWFILE);
  if (spinCoordOffset) {
    if (!SDDS_DefineSimpleParameter(SDDS_table, "Cspx", NULL, SDDS_DOUBLE) ||
	!SDDS_DefineSimpleParameter(SDDS_table, "Cspy", NULL, SDDS_DOUBLE) ||
	!SDDS_DefineSimpleParameter(SDDS_table, "Cspz", NULL, SDDS_DOUBLE)) {
      printf("Problem defining SDDS spin parameters in file %s (%s)\n", filename, caller);
      fflush(stdout);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
  }

  if ((varied_quantities &&
       !SDDS_DefineSimpleParameters(SDDS_table, varied_quantities, varied_quantity_name, varied_quantity_unit, SDDS_DOUBLE))) {
    printf("Problem defining extra SDDS parameters in file %s (%s)\n", filename, caller);
    fflush(stdout);
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    exitElegant(1);
  }
  if (optimization_quantities) {
    for (i=0; i<optimization_quantities; i++) {
      char *buffer;
      long length;
      if (!SDDS_DefineSimpleParameter(SDDS_table, optimization_quantity_name[i], optimization_quantity_unit[i], SDDS_DOUBLE)) {
        printf("Problem defining extra SDDS parameters in file %s (%s)\n", filename, caller);
        fflush(stdout);
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        exitElegant(1);
      }
      if (i<(optimization_quantities-3)) {
        /* Add lower and upper limits for all but the three special quantities */
        buffer = tmalloc(sizeof(*buffer)*(length=strlen(optimization_quantity_name[i])+100));
        snprintf(buffer, length, "%s.LowerLimit", optimization_quantity_name[i]);
        if (!SDDS_DefineSimpleParameter(SDDS_table, buffer, optimization_quantity_unit[i], SDDS_DOUBLE)) {
          printf("Problem defining extra SDDS parameters in file %s (%s)\n", filename, caller);
          fflush(stdout);
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          exitElegant(1);
        }
        snprintf(buffer, length, "%s.UpperLimit", optimization_quantity_name[i]);
        if (!SDDS_DefineSimpleParameter(SDDS_table, buffer, optimization_quantity_unit[i], SDDS_DOUBLE)) {
          printf("Problem defining extra SDDS parameters in file %s (%s)\n", filename, caller);
          fflush(stdout);
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          exitElegant(1);
        }
        free(buffer);
      }
    }
  }
  for (i = 0; i < error_elements; i++) {
    if (!SDDS_DefineSimpleParameter(SDDS_table, error_element_name[i],
                                    error_element_unit[i], SDDS_DOUBLE)) {
      /* determine if the error is just because of a duplication */
      long j;
      for (j = 0; j < i; j++)
        if (strcmp(error_element_name[i], error_element_name[j]) == 0)
          break;
      if (i == j) {
        printf("Problem defining extra SDDS parameter %s in file %s (%s)\n", error_element_name[i],
               filename, caller);
        fflush(stdout);
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        exitElegant(1);
      }
      duplicates++;
    }
    if ((error_element_index[i] = SDDS_GetParameterIndex(SDDS_table, error_element_name[i])) < 0) {
      printf("Problem defining extra SDDS parameter %s in file %s (%s): couldn't retrieve index\n",
             error_element_name[i],
             filename, caller);
      fflush(stdout);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
  }
  *error_element_duplicates = duplicates;

  if (!SDDS_WriteLayout(SDDS_table)) {
    printf("Unable to write SDDS layout for file %s (%s)\n", filename, caller);
    fflush(stdout);
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    exitElegant(1);
  }
}

void dump_final_properties(SDDS_TABLE *SDDS_table, BEAM_SUMS *sums,
                           double *varied_quan, char *first_varied_quan_name, long n_varied_quan,
                           long totalSteps,
                           double *perturbed_quan, long *perturbed_quan_index,
                           long perturbed_quan_duplicates, long n_perturbed_quan,
                           double *optim_quan, char *first_optim_quan_name, 
                           long n_optim_quan, double *optim_lower, double *optim_upper,
                           long step, double **particle, long n_original, double p_central, VMATRIX *M,
                           double charge) {
  long n_computed, n_properties = 0;
  double *computed_properties;
  long index, i;

  log_entry("dump_final_properites");

  if (!SDDS_table)
    bombElegant("NULL SDDS_TABLE pointer (dump_final_properties)", NULL);
  if (!sums)
    bombElegant("NULL beam sums pointer (dump_final_properites)", NULL);
  if (n_varied_quan && (!varied_quan || !first_varied_quan_name))
    bombElegant("Unexpected NULL pointer for varied quanitity values/names (dump_final_properties)", NULL);
  if (n_perturbed_quan && (!perturbed_quan || !perturbed_quan_index))
    bombElegant("Unexpected NULL pointer for perturbed quantity values/names (dump_final_properties)", NULL);
  if (n_optim_quan && (!optim_quan || !first_optim_quan_name || !optim_lower || !optim_upper))
    bombElegant("Unexpected NULL pointer for optimization quantity values/names (dump_final_properties)", NULL);
  if (!M)
    bombElegant("NULL matrix pointer (dump_final_properties)", NULL);
  if (isSlave)
    if (!particle)
      bombElegant("NULL particle coordinates pointer (dump_final_properties)", NULL);

  if (isMaster)
    if (!SDDS_StartTable(SDDS_table, 0)) {
      SDDS_SetError("Problem starting SDDS table (dump_final_properties)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }

  if (isMaster)
    if ((n_properties = SDDS_ParameterCount(SDDS_table)) !=
        (FINAL_PROPERTY_PARAMETERS + n_varied_quan + n_perturbed_quan + (spinCoordOffset?3:0) +
         (n_optim_quan?3*(n_optim_quan-3) + 3:0) - perturbed_quan_duplicates)) {
      printf("error: the number of parameters (%ld) defined for the SDDS table for the final properties file is not equal to the number of quantities (%ld) for which information is provided (dump_final_properties)\n",
             n_properties,
             FINAL_PROPERTY_PARAMETERS + n_varied_quan + n_perturbed_quan + 
             (n_optim_quan?3*(n_optim_quan-3) + 3:0) + (spinCoordOffset?3:0) - perturbed_quan_duplicates);
      fflush(stdout);
      abort();
    }
#if USE_MPI
  /* This is required to let all the processors get right n_properties and perturbed_quan_duplicates */
  MPI_Bcast(&n_properties, 1, MPI_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(&perturbed_quan_duplicates, 1, MPI_LONG, 0, MPI_COMM_WORLD);
#endif
  computed_properties = tmalloc(sizeof(*computed_properties) * n_properties);
  if ((n_computed = compute_final_properties(computed_properties, sums, n_original, p_central, M, particle, step,
                                             totalSteps, charge)) !=
      (n_properties - (n_varied_quan + n_perturbed_quan + (spinCoordOffset?3:0) +
                       (n_optim_quan?3*(n_optim_quan -3 ) + 3:0) - perturbed_quan_duplicates))) {
    printf("error: compute_final_properties computed %ld quantities--%ld expected. (dump_final_properties)\n",
           n_computed, n_properties - (n_varied_quan + n_perturbed_quan + 
                                       (n_optim_quan?3*(n_optim_quan-3) + 3:0) - perturbed_quan_duplicates));
    printf("n_varied_quan = %ld, n_perturbed_quan = %ld, perturbed_quan_duplicates = %ld\n",
           n_varied_quan, n_perturbed_quan + (n_optim_quan?3*(n_optim_quan-3) + 3:0), perturbed_quan_duplicates);
    fflush(stdout);
    abort();
  }
  if (isMaster) {
    if ((index = SDDS_GetParameterIndex(SDDS_table, "MEM")) < 0) {
      SDDS_SetError("Problem getting SDDS index of Step parameter (dump_final_properties)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    for (i = 0; i < FINAL_PROPERTY_LONG_PARAMETERS; i++)
      if (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE,
                              i + index, (long)computed_properties[i + index], -1)) {
        SDDS_SetError("Problem setting SDDS parameter values (dump_final_properties)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
    if ((index = SDDS_GetParameterIndex(SDDS_table, "Sx")) < 0) {
      SDDS_SetError("Problem getting SDDS index of Sx parameter (dump_final_properties)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    for (i = 0; i < FINAL_PROPERTY_PARAMETERS - FINAL_PROPERTY_LONG_PARAMETERS - FINAL_PROPERTY_OTHER_PARAMETERS; i++)
      if (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE,
                              i + index, computed_properties[i + index], -1)) {
        SDDS_SetError("Problem setting SDDS parameter values for computed properties (dump_final_properties)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }

    if (first_varied_quan_name) {
      if ((index = SDDS_GetParameterIndex(SDDS_table, first_varied_quan_name)) < 0) {
        SDDS_SetError("Problem getting SDDS index of first varied quantity parameter (dump_final_properties)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
      for (i = 0; i < n_varied_quan; i++)
        if (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE,
                                i + index, varied_quan[i], -1)) {
          SDDS_SetError("Problem setting SDDS parameter values for varied quantities (dump_final_properties)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
    }

    if (perturbed_quan_index) {
      for (i = 0; i < n_perturbed_quan; i++)
        if (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE,
                                perturbed_quan_index[i], (double)0.0, -1)) {
          SDDS_SetError("Problem setting SDDS parameter values for perturbed quantities (dump_final_properties)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
      for (i = 0; i < n_perturbed_quan; i++) {
        double value;
        if (!SDDS_GetParameterByIndex(SDDS_table, perturbed_quan_index[i], &value) ||
            !SDDS_SetParameters(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE,
                                perturbed_quan_index[i], perturbed_quan[i] + value, -1)) {
          SDDS_SetError("Problem setting SDDS parameter values for perturbed quantities (dump_final_properties)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
      }
    }

    if (first_optim_quan_name) {
      if ((index = SDDS_GetParameterIndex(SDDS_table, first_optim_quan_name)) < 0) {
        SDDS_SetError("Problem getting SDDS index of first optimization quantity parameter (dump_final_properties)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
      for (i = 0; i < n_optim_quan-3; i++)
        if (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE,
                                3*i + index    , optim_quan[i],
                                3*i + index + 1, optim_lower[i],
                                3*i + index + 2, optim_upper[i],
                                -1)) {
          SDDS_SetError("Problem setting SDDS parameter values for optimization quantities (dump_final_properties)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
      index = 3*(i-1) + index+3;
      /* These are the 3 extra items */
      for (i = 0; i < 3; i++)
        if (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE,
                                i + index, optim_quan[i+n_optim_quan-3], -1)) {
          SDDS_SetError("Problem setting SDDS parameter values for optimization quantities (dump_final_properties)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
    }

    if (spinCoordOffset) {
      long isp;
      if ((isp = SDDS_GetParameterIndex(SDDS_table, "Cspx"))<0) {
        SDDS_SetError("Problem with definition of SDDS spin parameters (dump_final_properties)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
      if (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE,
			      isp+0, sums->spinSums->centroid[0], 
			      isp+1, sums->spinSums->centroid[1], 
			      isp+2, sums->spinSums->centroid[2],
			      -1)) {
	SDDS_SetError("Problem setting SDDS parameter values for spin quantities (dump_final_properties)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
    }
    if (!SDDS_WriteTable(SDDS_table)) {
      SDDS_SetError("Problem writing SDDS data for final properties (dump_final_properties)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    if (!inhibitFileSync)
      SDDS_DoFSync(SDDS_table);
  }

  free(computed_properties);
  log_exit("dump_final_properties");
}

/* routine: compute_final_properties
 * purpose: compute sigmas, centroids, number of particles, transmission,
 *          plus Rij (i<=4, j<=6) at end of beamline
 *
 */

long compute_final_properties(double *data, BEAM_SUMS *sums, long n_original, double p_central, VMATRIX *M, double **coord,
                              long step, long steps, double charge) {
  register long i, j;
  long i_data, offset;
  /* long index; */
  double dp_min, dp_max, Ddp = 0;
  double p_sum, gamma_sum, p, sum, tc = 0, tmin, tmax, dt = 0, t, pAverage;
  double **R;
  /* double centroid[6]; */
  MATRIX Rmat;
  static double *tData = NULL, *deltaData = NULL;
  static long percDataMax = 0;
  double percLevel[12] = {25, 20, 15, 10, 5, 2.5, 75, 80, 85, 90, 95, 97.5};
  double tPosition[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  double deltaPosition[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  double percLevel2[9] = {10, 20, 30, 40, 50, 60, 70, 80, 90};
  double tPosition2[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
#if USE_MPI
  double tmp;
  long n_part_total = 0, n_orig_total = 0;
#endif
  log_entry("compute_final_properties");
#ifdef DEBUG
  printf("compute_final_properties\n");
  fflush(stdout);
#endif

  if (!data)
    bombElegant("return data array is null (compute_final_properties)", NULL);
  if (!sums)
    bombElegant("beam sums element is null (compute_final_properties)", NULL);
  /*
  if (!M || !M->C || !M->R)
    bombElegant("invalid/null transport map (compute_final_properties)", NULL);
  */
  if (isSlave)
    if (!coord)
      bombElegant("particle coordinate array is null (compute_final_properties)", NULL);

      /* compute centroids and sigmas */
#if USE_MPI
  if (distributedBeam) {
    /* Establish the two global particle counts used below (notably for the
     * Transmission = n_part_total/n_orig_total ratio):
     *   n_orig_total  = global number of particles initially tracked.  n_original
     *                   is passed in as beam->n_to_track_total, i.e. the MPI_Reduce
     *                   of each processor's initial beam->n_to_track performed by
     *                   the caller (see bunched_beam.c, finish_output).  That reduce
     *                   writes only the root, so on the slaves n_original is stale
     *                   here; the Bcast below repairs it.
     *   n_part_total  = global number of surviving particles.  We read it from the
     *                   master's sums->n_part rather than reducing the slaves'
     *                   counts on purpose: accumulate_beam_sums() (compute_centroids.c)
     *                   already stores the *global* survivor total in the master's
     *                   sums->n_part for BOTH true-parallel (particles on slaves)
     *                   and gathered (particles on master) modes.  A slave-side
     *                   reduction would be wrong in the gathered case, so do not
     *                   "simplify" this to an Allreduce of slave counts.
     * The Bcasts are also load-bearing for synchronization, not just for the value:
     * the guard just below (... && n_part_total) gates a block that performs
     * collective operations (approximate_percentiles_p, the momentum-sum reduce,
     * etc.).  All processors must agree on n_part_total to enter or skip that block
     * together; otherwise some would skip the collectives and the run would hang
     * (the same class of deadlock as the lost-particle bug in new_sdds_beam). */
    n_orig_total = n_original;
    if (myid == 0)
      n_part_total = sums->n_part;
    MPI_Bcast(&n_part_total, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_orig_total, 1, MPI_LONG, 0, MPI_COMM_WORLD);
  }
  if ((distributedBeam && n_part_total) || (!distributedBeam && sums->n_part))
  /* We have to check the total number of particles, otherwise it will cause
	 synchronization problem as no particle left on some processors */
#else
  if (sums->n_part)
#endif
  {
    for (i = 0; i < 6; i++) {
      /* centroid[i] = data[i+F_CENTROID_OFFSET] = sums->centroid[i]; */
      data[i + F_CENTROID_OFFSET] = sums->centroid[i];
    }
    if (sums->beamSums2) {
      for (i = 0; i < 6; i++)
        data[i + F_SIGMA_OFFSET] = sqrt(sums->beamSums2->sigma[i][i]);
      for (i = 0; i < 7; i++) {
        data[i + F_MINVALUE_OFFSET] = sums->beamSums2->min[i];
        data[i + F_MAXVALUE_OFFSET] = sums->beamSums2->max[i];
        data[i + F_MAXABS_OFFSET] = sums->beamSums2->maxabs[i];
      }
    } else {
      for (i = 0; i < 6; i++)
        data[i + F_SIGMA_OFFSET] = -DBL_MAX;
      for (i = 0; i < 7; i++) {
        data[i + F_MAXABS_OFFSET] = data[i + F_MAXVALUE_OFFSET] = -DBL_MAX;
        data[i + F_MINVALUE_OFFSET] = DBL_MAX;
      }
    }
    offset = F_SIGMAT_OFFSET;
    /* index = 0; */
    /* sigma matrix elements sij */
    if (sums->beamSums2) {
      for (i = 0; i < 7; i++) {
        /* skip the diagonal element */
        for (j = i + 1; j < 7; j++)
          data[offset++] = sums->beamSums2->sigma[i][j];
      }
    } else {
      for (i = 0; i < 7; i++) {
        /* skip the diagonal element */
        for (j = i + 1; j < 7; j++)
          data[offset++] = -DBL_MAX;
      }
    }
    /* time centroid, sigma, and delta */
    tmax = dp_max = -(tmin = dp_min = DBL_MAX);
    if ((!tData || sums->n_part > percDataMax) &&
        (!(tData = malloc(sizeof(*tData) * (percDataMax = sums->n_part))) ||
         !(deltaData = malloc(sizeof(*deltaData) * (percDataMax = sums->n_part)))))
      bombElegant("memory allocation failure (compute_final_properties)", NULL);
#if USE_MPI
    if (isSlave || !distributedBeam)
#endif
      for (i = sum = 0; i < sums->n_part; i++) {
        if (!coord[i]) {
          printf("coordinate element for particle %ld is null (compute_final_properties)\n", i);
          fflush(stdout);
          abort();
        }
        if (coord[i][5] > dp_max)
          dp_max = coord[i][5];
        if (coord[i][5] < dp_min)
          dp_min = coord[i][5];
        p = p_central * (1 + coord[i][5]);
        deltaData[i] = coord[i][5];
        sum += (t = tData[i] = coord[i][4] / (p / sqrt(sqr(p) + 1) * c_mks));
        if (t < tmin)
          tmin = t;
        if (t > tmax)
          tmax = t;
      }
#if USE_MPI
    if (distributedBeam) {
      if (isMaster) {
        tmax = dp_max = -DBL_MAX;
        tmin = dp_min = DBL_MAX;
        sum = 0;
      }
      tmp = tmax;
      MPI_Reduce(&tmp, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      tmp = tmin;
      MPI_Reduce(&tmp, &tmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
      tmp = dp_max;
      MPI_Reduce(&tmp, &dp_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      tmp = dp_min;
      MPI_Reduce(&tmp, &dp_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
      tmp = sum;
      MPI_Reduce(&tmp, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
#endif
    if (isMaster || !distributedBeam) {
      dt = tmax - tmin;
      Ddp = dp_max - dp_min;
      data[6 + F_CENTROID_OFFSET] = (tc = sum / sums->n_part);
    }
#if USE_MPI
    if (distributedBeam)
      MPI_Bcast(&tc, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
    if (isSlave || !distributedBeam)
      for (i = sum = 0; i < sums->n_part; i++)
        sum += sqr(tData[i] - tc);
#if USE_MPI
    if (distributedBeam) {
      if (isMaster)
        sum = 0.0;
      tmp = sum;
      MPI_Reduce(&tmp, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
#endif
    data[6 + F_SIGMA_OFFSET] = sqrt(sum / sums->n_part);
#if !USE_MPI
    /* results of these calls used below */
    approximate_percentiles(tPosition, percLevel, 12, tData, sums->n_part, ANALYSIS_BINS2);
    approximate_percentiles(tPosition2, percLevel2, 9, tData, sums->n_part, ANALYSIS_BINS2);
    approximate_percentiles(deltaPosition, percLevel, 12, deltaData, sums->n_part, ANALYSIS_BINS2);
#else
    if (distributedBeam) {
      if (n_part_total) {
        approximate_percentiles_p(tPosition, percLevel, 12, tData, sums->n_part, ANALYSIS_BINS2);
        approximate_percentiles_p(tPosition2, percLevel2, 9, tData, sums->n_part, ANALYSIS_BINS2);
        approximate_percentiles_p(deltaPosition, percLevel, 12, deltaData, sums->n_part, ANALYSIS_BINS2);
      }
    } else {
      approximate_percentiles(tPosition, percLevel, 12, tData, sums->n_part, ANALYSIS_BINS2);
      approximate_percentiles(tPosition2, percLevel2, 9, tData, sums->n_part, ANALYSIS_BINS2);
      approximate_percentiles(deltaPosition, percLevel, 12, deltaData, sums->n_part, ANALYSIS_BINS2);
    }
#endif
  } else {
    for (i = 0; i < 7; i++)
      data[i + F_CENTROID_OFFSET] = data[i + F_SIGMA_OFFSET] = 0;
    dt = Ddp = tmin = tmax = 0;
    for (i = 0; i < 12; i++)
      tPosition[i] = deltaPosition[i] = 0;
  }

#ifdef DEBUG
  printf("compute_final_properties (10)\n");
  fflush(stdout);
#endif

  /* transmission */
#if USE_MPI
  if (distributedBeam) {
    /* n_part_total/n_orig_total are the reduced counts computed above */
    if (n_orig_total)
      data[F_T_OFFSET] = ((double)n_part_total) / n_orig_total;
    else
      data[F_T_OFFSET] = 0;
  } else {
    /* Non-distributed (e.g. single-particle) beam: n_part_total/n_orig_total were
     * never set, and the master holds the whole beam, so use the local counts as
     * in the serial case.  Using the uninitialized totals here gave a nonsensical
     * Transmission for a single-particle beam from bunched_beam. */
    if (n_original)
      data[F_T_OFFSET] = ((double)sums->n_part) / n_original;
    else
      data[F_T_OFFSET] = 0;
  }
#else
  if (n_original)
    data[F_T_OFFSET] = ((double)sums->n_part) / n_original;
  else
    data[F_T_OFFSET] = 0;
#endif
  
  /* lattice momentum :*/
  data[F_T_OFFSET + 1] = p_central;

  /* compute average momentum and kinetic energy */
  p_sum = gamma_sum = 0;
  pAverage = p_central;

#if USE_MPI
  if (isSlave || !distributedBeam)
#endif
    for (i = 0; i < sums->n_part; i++) {
      p_sum += (p = (1 + coord[i][5]) * p_central);
      gamma_sum += sqrt(sqr(p) + 1);
    }

#if USE_MPI
  if (distributedBeam) {
    double *tmp_sum = malloc(2 * sizeof(*tmp_sum)),
           *tmp = malloc(2 * sizeof(*tmp_sum));
    tmp[0] = p_sum;
    tmp[1] = gamma_sum;
    MPI_Reduce(tmp, tmp_sum, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    p_sum = tmp_sum[0];
    gamma_sum = tmp_sum[1];
    free(tmp);
    free(tmp_sum);
  }
#endif
  if (isMaster || !distributedBeam) {
    if (sums->n_part) {
      pAverage = data[F_T_OFFSET + 2] = p_sum / sums->n_part;
      data[F_T_OFFSET + 3] = (gamma_sum / sums->n_part - 1) * particleMassMV;
    } else
      data[F_T_OFFSET + 2] = data[F_T_OFFSET + 3] = 0;
    /* beam charge */
    data[F_T_OFFSET + 4] = charge;
  }
  /* compute "sigma" from width of particle distributions for x and y */
#if !USE_MPI
  if (coord && sums->n_part > 3) {
    data[F_WIDTH_OFFSET] = approximateBeamWidth(0.6826F, coord, sums->n_part, 0L) / 2.;
    data[F_WIDTH_OFFSET + 1] = approximateBeamWidth(0.6826F, coord, sums->n_part, 2L) / 2.;
#else
  if ((distributedBeam && n_part_total > 3) || (!distributedBeam && coord && sums->n_part > 3)) { /* In the version with parallel IO, coord on master points to NULL */
    if (!distributedBeam) {
      /* All the processors compute the final properties independently. */
      data[F_WIDTH_OFFSET] = approximateBeamWidth_p(0.6826F, coord, sums->n_part, 0L) / 2.;
      data[F_WIDTH_OFFSET + 1] = approximateBeamWidth_p(0.6826F, coord, sums->n_part, 2L) / 2.;
    } else {
      data[F_WIDTH_OFFSET] = approximateBeamWidth(0.6826F, coord, sums->n_part, 0L) / 2.;
      data[F_WIDTH_OFFSET + 1] = approximateBeamWidth(0.6826F, coord, sums->n_part, 2L) / 2.;
    }
#endif
    data[F_WIDTH_OFFSET + 2] = dt;
    data[F_WIDTH_OFFSET + 3] = Ddp;
    for (i = 0; i < 6; i++) {
      data[F_WIDTH_OFFSET + 4 + i] = tPosition[6 + i] - tPosition[0 + i];
      data[F_WIDTH_OFFSET + 10 + i] = deltaPosition[6 + i] - deltaPosition[0 + i];
    }
    for (i = 0; i < 9; i++)
      data[F_PERC_OFFSET + i] = tPosition2[i] - tmin;
  } else {
    for (i = 0; i < 10; i++)
      data[F_WIDTH_OFFSET] = 0;
    for (i = 0; i < 9; i++)
      data[F_PERC_OFFSET + i] = tPosition2[i] - tmin;
  }

#ifdef DEBUG
  printf("compute_final_properties (20)\n");
  fflush(stdout);
#endif

  if (sums->beamSums2) {
    /* compute emittances */
    computeEmitTwissFromSigmaMatrix(data + F_EMIT_OFFSET + 0, data + F_EMIT_OFFSET + 2, NULL, NULL, sums->beamSums2->sigma, 0);
    computeEmitTwissFromSigmaMatrix(data + F_EMIT_OFFSET + 1, data + F_EMIT_OFFSET + 3, NULL, NULL, sums->beamSums2->sigma, 2);
  }

#if !USE_MPI
  data[F_EMIT_OFFSET + 4] = rms_longitudinal_emittance(coord, sums->n_part, p_central, 0, 0);
#else
  if (distributedBeam)
    data[F_EMIT_OFFSET + 4] = rms_longitudinal_emittance_p(coord, sums->n_part, p_central, 0, 0);
  else
    data[F_EMIT_OFFSET + 4] = rms_longitudinal_emittance(coord, sums->n_part, p_central, 0, 0);
#endif

  /* compute normalized emittances */
  if (!exactNormalizedEmittance) {
    for (i = 0; i < 4; i++)
      data[F_NEMIT_OFFSET + i] = pAverage * data[F_EMIT_OFFSET + i];
  } else {
    double dummy;
    if (sums->beamSums2)
      for (i = 0; i < 2; i++)
        computeEmitTwissFromSigmaMatrix(data + F_NEMIT_OFFSET + i + 0, data + F_NEMIT_OFFSET + i + 2, &dummy, &dummy, sums->beamSums2->sigman, 2 * i);
  }

#ifdef DEBUG
  printf("compute_final_properties (30)\n");
  fflush(stdout);
#endif

  i_data = F_RMAT_OFFSET;
  if (M && (R=M->R)) {
    R = M->R;
    for (i = 0; i < 6; i++)
      for (j = 0; j < 6; j++)
        data[i_data++] = R[i][j];
    Rmat.a = R;
    Rmat.n = Rmat.m = 6;
    data[i_data] = m_det(&Rmat);
  } else {
    for (i = 0; i < 6; i++)
      for (j = 0; j < 6; j++)
        data[i_data++] = 0;
    data[i_data] = 0;
  }
  
  /* run time statistics */
  i_data = F_STATS_OFFSET;
#if defined(__linux__) || defined(__APPLE__)
  data[i_data++] = cpu_time() / 100.0;
  data[i_data++] = delapsed_time();
#  if USE_MPI
  data[i_data++] = delapsed_time() * n_processors;
#  else
  data[i_data++] = delapsed_time();
#  endif
  data[i_data++] = memoryUsage();
  data[i_data++] = page_faults();
#else
  data[i_data++] = 0;
  data[i_data++] = delapsed_time();
#  if USE_MPI
  data[i_data++] = delapsed_time() * n_processors;
#  else
  data[i_data++] = delapsed_time();
#  endif
  data[i_data++] = 0;
  data[i_data++] = 0;
#endif
  data[i_data++] = step;
  data[i_data++] = steps;

  /* number of particles */
  data[i_data = F_N_OFFSET] = sums->n_part;
#ifdef DEBUG
  printf("compute_final_properties (40)\n");
  fflush(stdout);
#endif
  log_exit("compute_final_properties");

  return (i_data + F_N_QUANS);
}

double beam_width(double fraction, double **coord, long n_part,
                  long sort_coord) {
  static long i_median, i_lo, i_hi;
  static double dx0, dx1;

  log_entry("beam_width");

  /* sort data in ascending order by the coordinate for which the
   * beam width is to be determined */
  set_up_row_sort(sort_coord, 6L, sizeof(**coord), double_cmpasc);
  qsort((void *)coord, n_part, sizeof(*coord), row_compare);
  for (i_median = 1; i_median < n_part; i_median++) {
    if (coord[i_median][sort_coord] < coord[i_median - 1][sort_coord])
      bombElegant("sort failure in beam_width()", NULL);
  }

  /* find indices of particles that are at +/- fraction/2 from the median */
  i_median = n_part / 2;
  if ((i_lo = i_median - fraction / 2. * n_part) < 0) {
    printWarning("Difficulty determining beam width using percentiles", "May be resolved using more particles");
  }
  if ((i_hi = i_median + fraction / 2. * n_part) >= n_part) {
    printWarning("Difficulty determining beam width using percentiles", "May be resolved using more particles");
  }

  if (i_lo != 0 && i_hi != (n_part - 1)) {
    dx0 = coord[i_hi][sort_coord] - coord[i_lo][sort_coord];
    dx1 = coord[i_hi + 1][sort_coord] - coord[i_lo - 1][sort_coord];
    log_exit("beam_width");
    return (INTERPOLATE(dx0, dx1, ((double)i_hi - i_lo + 1) / n_part, ((double)i_hi - i_lo + 3) / n_part, fraction));
  } else {
    log_exit("beam_width");
    return (coord[i_hi][sort_coord] - coord[i_lo][sort_coord]);
  }
}

double rms_emittance(double **coord, long i1, long i2, long n,
                     double *s11Return, double *s12Return, double *s22Return,
                     double *c1Return, double *c2Return) {
  double s11, s12, s22, x, xp;
  double xc, xpc;
  long i;

  if (!n)
    return (0.0);

  /* compute centroids */
  for (i = xc = xpc = 0; i < n; i++) {
    xc += coord[i][i1];
    xpc += coord[i][i2];
  }
  xc /= n;
  xpc /= n;
  if (c1Return)
    *c1Return = xc;
  if (c2Return)
    *c2Return = xpc;

  for (i = s11 = s12 = s22 = 0; i < n; i++) {
    x = coord[i][i1] - xc;
    xp = coord[i][i2] - xpc;
    s11 += sqr(x);
    s22 += sqr(xp);
    s12 += x * xp;
  }
  if (s11Return)
    *s11Return = s11 / n;
  if (s22Return)
    *s22Return = s22 / n;
  if (s12Return)
    *s12Return = s12 / n;
  return (SAFE_SQRT(s11 * s22 - sqr(s12)) / n);
}

#if USE_MPI
double rms_emittance_p(double **coord, long i1, long i2, long n,
                       double *s11Return, double *s12Return, double *s22Return,
                       double *c1Return, double *c2Return, long *nTotal) {
  double s11, s12, s22, x, xp, s11_local = 0.0, s22_local = 0.0, s12_local = 0.0;
  double xc, xpc, xc_local = 0.0, xpc_local = 0.0;
  long i, n_total = 0;

  if (distributedBeam) {
    if (isMaster)
      n = 0; /* The master will not contribute anything in this routine */
    MPI_Allreduce(&n, &n_total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  } else {
    if (!n)
      return (0.0);
  }

  if (nTotal)
    *nTotal = n_total;

  if (!n_total)
    return (0.0);

  /* compute centroids */
  for (i = 0; i < n; i++) {
    xc_local += coord[i][i1];
    xpc_local += coord[i][i2];
  }

  MPI_Allreduce(&xc_local, &xc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&xpc_local, &xpc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&n, &n_total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

  xc /= n_total;
  xpc /= n_total;
  if (c1Return)
    *c1Return = xc;
  if (c2Return)
    *c2Return = xpc;

  for (i = 0; i < n; i++) {
    x = coord[i][i1] - xc;
    xp = coord[i][i2] - xpc;
    s11_local += sqr(x);
    s22_local += sqr(xp);
    s12_local += x * xp;
  }

  MPI_Allreduce(&s11_local, &s11, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&s22_local, &s22, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&s12_local, &s12, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  if (s11Return)
    *s11Return = s11 / n_total;
  if (s22Return)
    *s22Return = s22 / n_total;
  if (s12Return)
    *s12Return = s12 / n_total;
  return (SAFE_SQRT(s11 * s22 - sqr(s12)) / n_total);
}
#endif

double rms_longitudinal_emittance(double **coord, long n, double Po, long startPID, long endPID) {
  double s11, s12, s22, dt, ddp;
  double tc, dpc, beta, P;
  long i, npCount;
  static double *time = NULL;
  static long max_n = 0;

  if (!n)
    return (0.0);

  for (i = npCount = 0; i < n; i++) {
    if (startPID >= endPID || (coord[i][6] >= startPID && coord[i][6] <= endPID)) {
      npCount++;
    }
  }
  if (!npCount)
    return 0.0;

  if (n > max_n)
    time = trealloc(time, sizeof(*time) * (max_n = n));

  log_entry("rms_logitudinal_emittance");

  /* compute centroids */
  for (i = tc = dpc = 0; i < n; i++) {
    if (startPID >= endPID || (coord[i][6] >= startPID && coord[i][6] <= endPID)) {
      P = Po * (1 + coord[i][5]);
      beta = P / sqrt(P * P + 1);
      time[i] = coord[i][4] / (beta * c_mks);
      tc += time[i];
      dpc += coord[i][5];
    }
  }
  tc /= npCount;
  dpc /= npCount;

  for (i = s11 = s12 = s22 = 0; i < n; i++) {
    if (startPID >= endPID || (coord[i][6] >= startPID && coord[i][6] <= endPID)) {
      dt = time[i] - tc;
      s11 += sqr(dt);
      ddp = coord[i][5] - dpc;
      s22 += sqr(ddp);
      s12 += dt * ddp;
    }
  }

  log_exit("rms_longitudinal_emittance");
  return (SAFE_SQRT(s11 * s22 - sqr(s12)) / npCount);
}

#if USE_MPI
double rms_longitudinal_emittance_p(double **coord, long n, double Po, long startPID, long endPID) {
  double s11, s12, s22, dt, ddp, s[3], s_total[3];
  double tc, dpc, beta, P, tmp[2], tmp_total[2];
  long i, npCount, npCount_total;
  double *time = NULL;
  long n_total = 0;

  if (distributedBeam) {
    if (isMaster)
      n = 0;
    MPI_Allreduce(&n, &n_total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  } else {
    if (!n)
      return (0.0);
  }
#  ifdef DEBUG
  printf("rms_longitudinal_emittance_p, n=%ld, n_total=%ld\n", n, n_total);
  fflush(stdout);
#  endif

  if (!n_total)
    return (0.0);

  for (i = npCount = 0; i < n; i++) {
    if (startPID >= endPID || (coord[i][6] >= startPID && coord[i][6] <= endPID)) {
      npCount++;
    }
  }
#  ifdef DEBUG
  printf("rms_longitudinal_emittance_p, npCount=%ld\n", npCount);
  fflush(stdout);
#  endif

  if (isMaster)
    npCount = 0;
  MPI_Allreduce(&npCount, &npCount_total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  if (!npCount_total)
    return 0.0;

#  ifdef DEBUG
  printf("rms_longitudinal_emittance_p, npCount_total=%ld\n", npCount_total);
  fflush(stdout);
#  endif
  time = tmalloc(sizeof(*time) * n);

#  ifdef DEBUG
  printf("rms_longitudinal_emittance_p, realloc of time finished\n");
  fflush(stdout);
#  endif
  if (isMaster)
    log_entry("rms_logitudinal_emittance");

  /* compute centroids */
  for (i = tc = dpc = 0; i < n; i++) {
    if (startPID >= endPID || (coord[i][6] >= startPID && coord[i][6] <= endPID)) {
      P = Po * (1 + coord[i][5]);
      beta = P / sqrt(P * P + 1);
      time[i] = coord[i][4] / (beta * c_mks);
      tc += time[i];
      dpc += coord[i][5];
    }
  }

#  ifdef DEBUG
  printf("rms_longitudinal_emittance_p, centroid computation finished\n");
  fflush(stdout);
#  endif

  tmp[0] = tc;
  tmp[1] = dpc;
  MPI_Allreduce((double*)&tmp, (double*)&tmp_total, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  tc = tmp_total[0] / npCount_total;
  dpc = tmp_total[1] / npCount_total;

  for (i = s11 = s12 = s22 = 0; i < n; i++) {
    if (startPID >= endPID || (coord[i][6] >= startPID && coord[i][6] <= endPID)) {
      dt = time[i] - tc;
      s11 += sqr(dt);
      ddp = coord[i][5] - dpc;
      s22 += sqr(ddp);
      s12 += dt * ddp;
    }
  }
  s[0] = s11;
  s[1] = s12;
  s[2] = s22;
  s_total[0] = s_total[1] = s_total[2] = 0; /* suppresses valgrind complaint */
  MPI_Reduce(s, s_total, 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

#  ifdef DEBUG
  printf("rms_longitudinal_emittance_p, sigma computation finished\n");
  fflush(stdout);
#  endif

  if (isMaster)
    log_exit("rms_longitudinal_emittance");

  free(time);

#  ifdef DEBUG
  printf("rms_longitudinal_emittance_p, returning\n");
  fflush(stdout);
#  endif

  /* Only the master will return meaningful result */
  return (SAFE_SQRT(s_total[0] * s_total[2] - sqr(s_total[1])) / npCount_total);
}

#endif

void compute_longitudinal_parameters(ONE_PLANE_PARAMETERS *bp, double **coord, long n, double Po) {
  double s11, s12, s22, S1, S2, dt, ddp;
  double tc, dpc, beta, P;
  double tmin, tmax, dpmin, dpmax;
  long i;
  static double *time = NULL;
  static long max_n = 0;

  log_entry("compute_longitudinal_parameters");

  if (!n)
    return;
  if (!bp) {
    printf("NULL ONE_PLANE_PARAMETERS pointer passed to compute_longitudinal_parameters\n");
    fflush(stdout);
    abort();
  }

  if (n > max_n)
    time = trealloc(time, sizeof(*time) * (max_n = n));

  /* compute centroids */
  tmax = -(tmin = DBL_MAX);
  dpmax = -(dpmin = DBL_MAX);
  for (i = tc = dpc = 0; i < n; i++) {
    P = Po * (1 + coord[i][5]);
    beta = P / sqrt(P * P + 1);
    time[i] = coord[i][4] / (beta * c_mks);
    tc += time[i];
    dpc += coord[i][5];
    if (coord[i][5] > dpmax)
      dpmax = coord[i][5];
    if (coord[i][5] < dpmin)
      dpmin = coord[i][5];
    if (time[i] > tmax)
      tmax = time[i];
    if (time[i] < tmin)
      tmin = time[i];
  }
  bp->c1 = (tc /= n);
  bp->c2 = (dpc /= n);
  bp->min1 = tmin;
  bp->min2 = dpmin;
  bp->max1 = tmax;
  bp->max2 = dpmax;

  for (i = S1 = S2 = s11 = s12 = s22 = 0; i < n; i++) {
    dt = time[i] - tc;
    s11 += sqr(dt);
    ddp = coord[i][5] - dpc;
    s22 += sqr(ddp);
    s12 += dt * ddp;
    S1 += sqr(time[i]);
    S2 += sqr(coord[i][5]);
  }
  bp->s11 = (s11 /= n);
  bp->s12 = (s12 /= n);
  bp->s22 = (s22 /= n);
  bp->S1 = sqrt(S1 / n);
  bp->S2 = sqrt(S2 / n);
  bp->emittance = SAFE_SQRT(s11 * s22 - sqr(s12));

  log_exit("compute_longitudinal_properties");
}

void compute_transverse_parameters(ONE_PLANE_PARAMETERS *bp, double **coord, long n, long plane) {
  double s11, s12, s22, S1, S2, dx, dxp;
  double xc, xpc;
  double xmin, xmax, xpmin, xpmax;
  long i, offset;

  if (!n)
    return;
  if (!bp) {
    printf("NULL ONE_PLANE_PARAMETERS pointer passed to compute_transverse_parameters\n");
    fflush(stdout);
    abort();
  }
  offset = plane ? 2 : 0;

  /* compute centroids */
  xmax = -(xmin = DBL_MAX);
  xpmax = -(xpmin = DBL_MAX);
  for (i = xc = xpc = 0; i < n; i++) {
    xc += coord[i][offset + 0];
    xpc += coord[i][offset + 1];
    if (coord[i][offset] > xmax)
      xmax = coord[i][offset];
    if (coord[i][offset] < xmin)
      xmin = coord[i][offset];
    if (coord[i][offset + 1] > xpmax)
      xpmax = coord[i][offset + 1];
    if (coord[i][offset + 1] < xpmin)
      xpmin = coord[i][offset + 1];
  }

  bp->c1 = (xc /= n);
  bp->c2 = (xpc /= n);
  bp->min1 = xmin;
  bp->min2 = xpmin;
  bp->max1 = xmax;
  bp->max2 = xpmax;

  for (i = S1 = S2 = s11 = s12 = s22 = 0; i < n; i++) {
    dx = coord[i][offset] - xc;
    s11 += sqr(dx);
    dxp = coord[i][offset + 1] - xpc;
    s22 += sqr(dxp);
    s12 += dx * dxp;
    S1 += sqr(coord[i][offset]);
    S2 += sqr(coord[i][offset + 1]);
  }
  bp->s11 = (s11 /= n);
  bp->s12 = (s12 /= n);
  bp->s22 = (s22 /= n);
  bp->S1 = sqrt(S1 / n);
  bp->S2 = sqrt(S2 / n);
  bp->emittance = SAFE_SQRT(s11 * s22 - sqr(s12));
}

void rpn_store_final_properties(double *value, long number) {
  static long *memory_number = NULL;
  long i;
  log_entry("rpn_store_final_parameters");
  if (number != FINAL_PROPERTY_PARAMETERS) {
    printf("error: number of values (%ld) being stored != FINAL_PROPERTY_PARAMETERS (rpn_store_final_parameters)\n",
           number);
    fflush(stdout);
    abort();
  }
  if (!memory_number) {
    memory_number = tmalloc(sizeof(*memory_number) * number);
    for (i = 0; i < number; i++)
      memory_number[i] = rpn_create_mem(final_property_parameter[i].name, 0);
  }
  for (i = 0; i < number; i++)
    rpn_store(value[i], NULL, memory_number[i]);
  log_exit("rpn_store_final_parameters");
}

long get_final_property_index(char *name) {
  long i;
  for (i = 0; i < FINAL_PROPERTY_PARAMETERS; i++)
    if (strcmp(name, final_property_parameter[i].name) == 0)
      return (i);
  return (-1);
}

long count_final_properties() {
  return (FINAL_PROPERTY_PARAMETERS);
}

double approximateBeamWidth(double fraction, double **part, long nPart, long iCoord) {
  double *hist, *cdf;
  long maxBins = ANALYSIS_BINS, bins = ANALYSIS_BINS, i50, iLo, iHi, i;
  double xMin, xMax, dx;
#if USE_MPI
  double *buffer;
#endif
  xMin = xMax = dx = 0;

  /* make histogram of the coordinate */
  hist = tmalloc(sizeof(*hist) * bins);

  binParticleCoordinate(&hist, &maxBins, &xMin, &xMax, &dx, &bins,
                        1.01, part, nPart, iCoord);
#if USE_MPI
  if (distributedBeam) { /* Master needs to know the information to write the result */
    buffer = malloc(sizeof(double) * bins);
    MPI_Allreduce(hist, buffer, bins, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    memcpy(hist, buffer, sizeof(double) * bins);
    free(buffer);
  }
#endif
  /* sum histogram to get CDF */
  cdf = hist;
  for (i = 1; i < bins; i++)
    cdf[i] += cdf[i - 1];
  /* normalize CDF and find 50% point */
  i50 = bins / 2;
  for (i = 0; i < bins; i++) {
    cdf[i] /= cdf[bins - 1];
    if (cdf[i] < 0.50)
      i50 = i;
  }
  /* find locations containing half the indicated area around the 50% point */
  iLo = iHi = i50;
  for (i = i50; i < bins; i++) {
    if ((cdf[i] - 0.5) < fraction / 2)
      iHi = i;
    else
      break;
  }
  for (i = i50; i >= 0; i--) {
    if ((0.5 - cdf[i]) < fraction / 2)
      iLo = i;
    else
      break;
  }
  free(hist);
  return (iHi - iLo) * dx;
}

#if USE_MPI
/* This function is added as we need do final statistics with Pelegant on one processor at the end */
double approximateBeamWidth_p(double fraction, double **part, long nPart, long iCoord) {
  double *hist, *cdf;
  long maxBins = ANALYSIS_BINS, bins = ANALYSIS_BINS, i50, iLo, iHi, i;
  double xMin, xMax, dx;
  xMin = xMax = dx = 0;

  /* make histogram of the coordinate */
  hist = tmalloc(sizeof(*hist) * bins);

  /* We expect it behaves like in a serial version */
  binParticleCoordinate_s(&hist, &maxBins, &xMin, &xMax, &dx, &bins,
                          1.01, part, nPart, iCoord);

  /* sum histogram to get CDF */
  cdf = hist;
  for (i = 1; i < bins; i++)
    cdf[i] += cdf[i - 1];
  /* normalize CDF and find 50% point */
  i50 = bins / 2;
  for (i = 0; i < bins; i++) {
    cdf[i] /= cdf[bins - 1];
    if (cdf[i] < 0.50)
      i50 = i;
  }
  /* find locations containing half the indicated area around the 50% point */
  iLo = iHi = i50;
  for (i = i50; i < bins; i++) {
    if ((cdf[i] - 0.5) < fraction / 2)
      iHi = i;
    else
      break;
  }
  for (i = i50; i >= 0; i--) {
    if ((0.5 - cdf[i]) < fraction / 2)
      iLo = i;
    else
      break;
  }
  free(hist);
  return (iHi - iLo) * dx;
}
#endif

void computeBeamTwissParameters(TWISS *twiss, double **data, long particles) {
  double C[6], S[6][6], eta[6], Sbeta[6][6], emitcor[3], betacor[3], alphacor[3];
  /* double beamsize[6]; */
  long i, j, iPart;
  double sum;
#if USE_MPI
  long particles_total, index = 0;
  double S_p[21], S_p_sum[21], Sbeta_p[10], Sbeta_p_sum[10];

  MPI_Allreduce(&particles, &particles_total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif

  compute_centroids(C, data, particles);

  /* compute correlations */
  for (i = 0; i < 6; i++) {
    for (j = 0; j <= i; j++) {
      for (iPart = sum = 0; iPart < particles; iPart++)
        sum += (data[iPart][i] - C[i]) * (data[iPart][j] - C[j]);
#if (!USE_MPI)
      S[j][i] = S[i][j] = sum / particles;
#else
      S_p[index++] = sum;
#endif
    }
  }

#if USE_MPI
  MPI_Allreduce(S_p, S_p_sum, 21, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  index = 0;
  for (i = 0; i < 6; i++) {
    for (j = 0; j <= i; j++) {
      S[i][j] = S[j][i] = S_p_sum[index++] / particles_total;
    }
  }
#endif

  /* for (i=0; i<6; i++) */
  /* beamsize[i] = sqrt(S[i][i]); */

  /* compute correlations with energy correlations removed */
  for (i = 0; i < 6; i++)
    eta[i] = 0;
  if (S[5][5])
    for (i = 0; i < 4; i++)
      eta[i] = S[i][5] / S[5][5];

#if USE_MPI
  index = 0;
#endif
  for (i = 0; i < 4; i++) {
    for (j = 0; j <= i; j++) {
      for (iPart = sum = 0; iPart < particles; iPart++)
        sum += ((data[iPart][i] - C[i]) - eta[i] * (data[iPart][5] - C[5])) * ((data[iPart][j] - C[j]) - eta[j] * (data[iPart][5] - C[5]));
#if (!USE_MPI)
      Sbeta[j][i] = Sbeta[i][j] = sum / particles;
#else
      Sbeta_p[index++] = sum;
#endif
    }
  }
#if USE_MPI
  MPI_Allreduce(Sbeta_p, Sbeta_p_sum, 10, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  index = 0;
  for (i = 0; i < 4; i++) {
    for (j = 0; j <= i; j++) {
      Sbeta[i][j] = Sbeta[j][i] = Sbeta_p_sum[index++] / particles_total;
    }
  }
#endif

  /* compute beta functions etc */
  for (i = 0; i < 2; i++) {
    emitcor[i] = betacor[i] = alphacor[i] = 0;
    if ((emitcor[i] = Sbeta[2 * i + 0][2 * i + 0] * Sbeta[2 * i + 1][2 * i + 1] - sqr(Sbeta[2 * i + 0][2 * i + 1])) > 0) {
      emitcor[i] = sqrt(emitcor[i]);
      betacor[i] = Sbeta[2 * i + 0][2 * i + 0] / emitcor[i];
      alphacor[i] = -Sbeta[2 * i + 0][2 * i + 1] / emitcor[i];
    } else
      emitcor[i] = 0;
  }

  twiss->betax = betacor[0];
  twiss->betay = betacor[1];
  twiss->alphax = alphacor[0];
  twiss->alphay = alphacor[1];
  twiss->etax = eta[0];
  twiss->etapx = eta[1];
  twiss->etay = eta[2];
  twiss->etapy = eta[3];
}

void computeBeamTwissParameters3(TWISSBEAM *twiss, double **data, long particles) {
  double C[6], S[6][6], eta[6], Sbeta[6][6];
  /* double beamsize[6]; */
  long i, j, iPart;
  double sum;
#if USE_MPI
  long particles_total, index = 0;
  double S_p[21], S_p_sum[21], Sbeta_p[21], Sbeta_p_sum[21];
  if (distributedBeam)
    MPI_Allreduce(&particles, &particles_total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif

  compute_centroids(C, data, particles);
  memcpy(twiss->centroid, C, sizeof(*(twiss->centroid)) * 6);

  /* compute correlations */
  for (i = 0; i < 6; i++) {
    for (j = 0; j <= i; j++) {
      for (iPart = sum = 0; iPart < particles; iPart++)
        sum += (data[iPart][i] - C[i]) * (data[iPart][j] - C[j]);
#if (!USE_MPI)
      S[j][i] = S[i][j] = sum / particles;
#else
      if (distributedBeam)
        S_p[index++] = sum;
      else
        S[j][i] = S[i][j] = sum / particles;
#endif
    }
  }

#if USE_MPI
  if (distributedBeam) {
    MPI_Allreduce(S_p, S_p_sum, 21, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    index = 0;
    for (i = 0; i < 6; i++) {
      for (j = 0; j <= i; j++) {
        S[i][j] = S[j][i] = S_p_sum[index++] / particles_total;
      }
    }
  }
#endif

  /* for (i=0; i<6; i++) */
  /* beamsize[i] = sqrt(S[i][i]); */

  /* compute correlations with energy offset */
  for (i = 0; i < 6; i++)
    eta[i] = 0;
  if (S[5][5])
    for (i = 0; i < 4; i++)
      eta[i] = S[i][5] / S[5][5];
  memcpy(twiss->eta, eta, sizeof(*(twiss->eta)) * 4);

#if USE_MPI
  if (distributedBeam)
    index = 0;
#endif
  for (i = 0; i < 6; i++) {
    for (j = 0; j <= i; j++) {
      for (iPart = sum = 0; iPart < particles; iPart++)
        sum += ((data[iPart][i] - C[i]) - eta[i] * (data[iPart][5] - C[5])) * ((data[iPart][j] - C[j]) - eta[j] * (data[iPart][5] - C[5]));
#if (!USE_MPI)
      Sbeta[j][i] = Sbeta[i][j] = sum / particles;
#else
      if (distributedBeam)
        Sbeta_p[index++] = sum;
      else
        Sbeta[j][i] = Sbeta[i][j] = sum / particles;
#endif
    }
  }
#if USE_MPI
  if (distributedBeam) {
    MPI_Allreduce(Sbeta_p, Sbeta_p_sum, 21, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    index = 0;
    for (i = 0; i < 6; i++) {
      for (j = 0; j <= i; j++) {
        Sbeta[i][j] = Sbeta[j][i] = Sbeta_p_sum[index++] / particles_total;
      }
    }
  }
#endif

  /* compute beta functions etc */
  for (i = 0; i < 3; i++) {
    twiss->emit[i] = twiss->beta[i] = twiss->alpha[i] = 0;
    if ((twiss->emit[i] = Sbeta[2 * i + 0][2 * i + 0] * Sbeta[2 * i + 1][2 * i + 1] - sqr(Sbeta[2 * i + 0][2 * i + 1])) > 0) {
      twiss->emit[i] = sqrt(twiss->emit[i]);
      twiss->beta[i] = Sbeta[2 * i + 0][2 * i + 0] / twiss->emit[i];
      twiss->alpha[i] = -Sbeta[2 * i + 0][2 * i + 1] / twiss->emit[i];
    } else
      twiss->emit[i] = 0;
  }
}
