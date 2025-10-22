/*************************************************************************\
 * Copyright (c) 2007 The University of Chicago, as Operator of Argonne
 * National Laboratory.
 * Copyright (c) 2007 The Regents of the University of California, as
 * Operator of Los Alamos National Laboratory.
 * This file is distributed subject to a Software License Agreement found
 * in the file LICENSE that is included with this distribution.
\*************************************************************************/

/* file: moments.c
 * purpose: computation of beam moments
 *
 * Michael Borland, 2007
 */
#define _USE_MATH_DEFINES
#include "mdb.h"
#include "track.h"
#include "moments.h"
#include "matlib.h"
#include <stddef.h>

void determineEquilibriumMoments(double **R, double *D, SIGMA_MATRIX *sigma0);
void propagateBeamMoments(RUN *run, LINE_LIST *beamline, double *traj);
void storeFitpointMomentsParameters(ELEMENT_LIST *elem);
void prepareMomentsArray(double *data, ELEMENT_LIST *elem, double *sigma);
void computeNaturalEmittances(VMATRIX *M, double *sigma, double *emittance);

static long momentsInitialized = 0;
static long SDDSMomentsInitialized = 0;
static SDDS_TABLE SDDSMoments;
static long momentsCount = 0;

#define IC_ELEMENT 0
#define IC_OCCURENCE 1
#define IC_TYPE 2
#define IC_S 3
#define IC_PCENTRAL 4
#define IC_EMITTANCE (IC_PCENTRAL + 1 + 21 + 6)
#define IC_SBETA (IC_EMITTANCE + 3)
#define IC_EMITBETA (IC_SBETA + 10)
#define IC_COULOMB_LOG (IC_EMITBETA+2)
#define N_COLUMNS (IC_PCENTRAL + 1 + 21 + 6 + 3 + 10 + 2 + 1)
static SDDS_DEFINITION column_definition[N_COLUMNS] = {
  {"ElementName", "&column name=ElementName, type=string, description=\"Element name\", format_string=%10s &end"},
  {"ElementOccurence", "&column name=ElementOccurence, type=long, description=\"Occurence of element\", format_string=%6ld &end"},
  {"ElementType", "&column name=ElementType, type=string, description=\"Element-type name\", format_string=%10s &end"},
  {"s", "&column name=s, type=double, units=m, description=Distance &end"},
  {"pCentral0", "&column name=pCentral0, type=double, units=\"m$be$nc\", symbol=\"p$bcent$n\", description=\"Initial central momentum\" &end"},
  {"s1", "&column name=s1, symbol=\"$gs$r$b1$n\", units=m, type=double, description=\"sqrt(<x*x>)\" &end"},
  {"s12", "&column name=s12, symbol=\"$gs$r$b12$n\", units=m, type=double, description=\"<x*xp'>\" &end"},
  {"s13", "&column name=s13, symbol=\"$gs$r$b13$n\", units=\"m$a2$n\", type=double, description=\"<x*y>\" &end"},
  {"s14", "&column name=s14, symbol=\"$gs$r$b14$n\", units=m, type=double, description=\"<x*y'>\" &end"},
  {"s15", "&column name=s15, symbol=\"$gs$r$b15$n\", units=\"m$a2$n\", type=double, description=\"<x*s>\" &end"},
  {"s16", "&column name=s16, symbol=\"$gs$r$b16$n\", units=m, type=double, description=\"<x*delta>\" &end"},
  {"s2", "&column name=s2, symbol=\"$gs$r$b2$n\", type=double, description=\"sqrt(<x'*x'>)\" &end"},
  {"s23", "&column name=s23, symbol=\"$gs$r$b23$n\", units=m, type=double, description=\"<x'*y>\" &end"},
  {"s24", "&column name=s24, symbol=\"$gs$r$b24$n\", type=double, description=\"<x'*y'>\" &end"},
  {"s25", "&column name=s25, symbol=\"$gs$r$b25$n\", units=m, type=double, description=\"<x'*s>\" &end"},
  {"s26", "&column name=s26, symbol=\"$gs$r$b26$n\", type=double, description=\"<x'*delta>\" &end"},
  {"s3", "&column name=s3, symbol=\"$gs$r$b3$n\", units=m, type=double, description=\"sqrt(<y*y>)\" &end"},
  {"s34", "&column name=s34, symbol=\"$gs$r$b34$n\", units=m, type=double, description=\"<y*y'>\" &end"},
  {"s35", "&column name=s35, symbol=\"$gs$r$b35$n\", units=\"m$a2$n\", type=double, description=\"<y*s>\" &end"},
  {"s36", "&column name=s36, symbol=\"$gs$r$b36$n\", units=m, type=double, description=\"<y*delta>\" &end"},
  {"s4", "&column name=s4, symbol=\"$gs$r$b4$n\", type=double, description=\"sqrt(<y'*y')>\" &end"},
  {"s45", "&column name=s45, symbol=\"$gs$r$b45$n\", units=m, type=double, description=\"<y'*s>\" &end"},
  {"s46", "&column name=s46, symbol=\"$gs$r$b46$n\", type=double, description=\"<s'*delta>\" &end"},
  {"s5", "&column name=s5, symbol=\"$gs$r$b5$n\", units=m, type=double, description=\"sqrt(<s*s>)\" &end"},
  {"s56", "&column name=s56, symbol=\"$gs$r$b56$n\", units=m, type=double, description=\"<s*delta>\" &end"},
  {"s6", "&column name=s6, symbol=\"$gs$r$b6$n\", type=double, description=\"sqrt(<delta*delta>)\" &end"},
  {"c1", "&column name=c1, symbol=\"c$b1$n\", units=m, type=double, description=\"<x>\" &end"},
  {"c2", "&column name=c2, symbol=\"c$b2$n\", units=, type=double, description=\"<x'>\" &end"},
  {"c3", "&column name=c3, symbol=\"c$b3$n\", units=m, type=double, description=\"<y>\" &end"},
  {"c4", "&column name=c4, symbol=\"c$b4$n\", units=, type=double, description=\"<y'>\" &end"},
  {"c5", "&column name=c5, symbol=\"c$b5$n\", units=m, type=double, description=\"<s>\" &end"},
  {"c6", "&column name=c6, symbol=\"c$b6$n\", units=, type=double, description=\"<delta>\" &end"},
  {"ex", "&column name=ex, symbol=\"$ge$r$bx$n\", units=m, type=double, description=\"Projected horizontal emittance\" &end"},
  {"ey", "&column name=ey, symbol=\"$ge$r$by$n\", units=m, type=double, description=\"Projected vertical emittance\" &end"},
  {"ez", "&column name=ez, symbol=\"$ge$r$bz$n\", units=m, type=double, description=\"Projected longitudinal emittance\" &end"},
  {"s1beta", "&column name=s1beta, symbol=\"$gs$r$b1,$gb$r$n\", units=m, type=double, description=\"sqrt(<x*x>) (betatron)\" &end"},
  {"s12beta", "&column name=s12beta, symbol=\"$gs$r$b12,$gb$r$n\", units=m, type=double, description=\"<x*xp'> (betatron)\" &end"},
  {"s13beta", "&column name=s13beta, symbol=\"$gs$r$b13,$gb$r$n\", units=\"m$a2$n\", type=double, description=\"<x*y> (betatron)\" &end"},
  {"s14beta", "&column name=s14beta, symbol=\"$gs$r$b14,$gb$r$n\", units=m, type=double, description=\"<x*y'> (betatron)\" &end"},
  {"s2beta", "&column name=s2beta, symbol=\"$gs$r$b2,$gb$r$n\", type=double, description=\"sqrt(<x'*x'>) (betatron)\" &end"},
  {"s23beta", "&column name=s23beta, symbol=\"$gs$r$b23,$gb$r$n\", units=m, type=double, description=\"<x'*y> (betatron)\" &end"},
  {"s24beta", "&column name=s24beta, symbol=\"$gs$r$b24,$gb$r$n\", type=double, description=\"<x'*y'> (betatron)\" &end"},
  {"s3beta", "&column name=s3beta, symbol=\"$gs$r$b3,$gb$r$n\", units=m, type=double, description=\"sqrt(<y*y>) (betatron)\" &end"},
  {"s34beta", "&column name=s34beta, symbol=\"$gs$r$b34,$gb$r$n\", units=m, type=double, description=\"<y*y'> (betatron)\" &end"},
  {"s4beta", "&column name=s4beta, symbol=\"$gs$r$b4,$gb$r$n\", type=double, description=\"sqrt(<y'*y')> (betatron)\" &end"},
  {"exbeta", "&column name=exbeta, symbol=\"$ge$r$bx,$gb$r$n\", units=m, type=double, description=\"Projected horizontal betatron emittance\" &end"},
  {"eybeta", "&column name=eybeta, symbol=\"$ge$r$by,$gb$r$n\", units=m, type=double, description=\"Projected vertical betatron emittance\" &end"},
  {"CoulombLog", "&column name=CoulombLog, type=double, description=\"Coulomb log if IBS calculations invoked\" &end\n"},
};

#define IP_STEP 0
#define IP_STAGE 1
#define IP_PCENTRAL 2
#define IP_E1 3
#define IP_E2 4
#define IP_E3 5
#define IP_IBS_ITERATION 6
#define IP_IBS_ITERATIONS 7
#define IP_E1_CONVERGENCE 8
#define IP_E2_CONVERGENCE 9
#define IP_E3_CONVERGENCE 10
#define IP_CHARGE 11
#define N_PARAMETERS IP_CHARGE + 1
static SDDS_DEFINITION parameter_definition[N_PARAMETERS] = {
  {"Step", "&parameter name=Step, type=long, description=\"Simulation step\" &end"},
  {"Stage", "&parameter name=Stage, type=string, description=\"Stage of computation\" &end"},
  {"pCentral", "&parameter name=pCentral, type=double, units=\"m$be$nc\", description=\"Central momentum\" &end"},
  {"e1", "&parameter name=e1, symbol=\"$ge$r$b1$n\", type=double, units=m,  description=\"Emittance of mode 1\" &end"},
  {"e2", "&parameter name=e2, symbol=\"$ge$r$b2$n\", type=double, units=m,  description=\"Emittance of mode 2\" &end"},
  {"e3", "&parameter name=e3, symbol=\"$ge$r$b3$n\", type=double, units=m,  description=\"Emittance of mode 3\" &end"},
  {"IBSIteration", "&parameter name=IBSIteration, type=short, description=\"Number of IBS iterations performed\" &end"},
  {"IBSIterations", "&parameter name=IBSIterations, type=short, description=\"Number of IBS iterations requested\" &end"},
  {"e1Convergence", "&parameter name=e1Convergence, symbol=\"$gDe$r$b1$n/$ge$r$b1$n\", type=double, units=m,  description=\"Convergence for IBS for emittance of mode 1\" &end"},
  {"e2Convergence", "&parameter name=e2Convergence, symbol=\"$gDe$r$b2$n/$ge$r$b2$n\", type=double, units=m,  description=\"Convergence for IBS for emittance of mode 2\" &end"},
  {"e3Convergence", "&parameter name=e3Convergence, symbol=\"$gDe$r$b3$n/$ge$r$b3$n\", type=double, units=m,  description=\"Convergence for IBS for emittance of mode 3\" &end"},
  {"Charge", "&parameter name=Charge, type=double, units=C, description=\"Charge if IBS included otherwise zero.\" &end"},
};

static double savedFinalMoments[6][6];
static double savedFinalCentroid[6];
static SDDS_DATASET SDDSmatrix;
static short matrixOutputInitialized = 0;
void updateIbsScatteringMatrices(LINE_LIST *beamline, double charge, double *eNatural);


void setUpMomentsMatrixOutput(RUN *run, char *outputFilename) {
  char buffer[1024], t[1024];
  long i, j;
  static char *unit[6] = {"m", "rad", "m", "rad", "m", "1"};
#if USE_MPI
  if (myid == 0) {
#endif
    if (!SDDS_InitializeOutputElegant(&SDDSmatrix, SDDS_BINARY, 0, "transfer and diffusion matrix from moments calculation",
                                      NULL, outputFilename))
      bombElegant("problem setting up output file for transfer and diffusion matrix", NULL);

    for (i = 0; i < 6; i++) {
      sprintf(buffer, "&column name=C%ld, symbol=\"C$b%ld$n\", type=double ", i + 1, i + 1);
      if (SDDS_StringIsBlank(unit[i]))
        strcpy_ss(t, " &end");
      else
        sprintf(t, "units=%s &end", unit[i]);
      strcat(buffer, t);
      if (!SDDS_ProcessColumnString(&SDDSmatrix, buffer, 0)) {
        SDDS_SetError("Problem defining SDDS matrix output Rij columns (setUpMomentsMatrixOutput)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
    }
    for (i = 0; i < 6; i++) {
      for (j = 0; j < 6; j++) {
        sprintf(buffer, "&column name=R%ld%ld, symbol=\"R$b%ld%ld$n\", type=double ", i + 1, j + 1, i + 1, j + 1);
        if (i == j)
          strcpy_ss(t, " &end");
        else
          sprintf(t, "units=%s/%s &end", unit[i], unit[j]);
        strcat(buffer, t);
        if (!SDDS_ProcessColumnString(&SDDSmatrix, buffer, 0)) {
          SDDS_SetError("Problem defining SDDS matrix output Rij columns (setUpMatrixOutput)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
      }
    }
    for (i = 0; i < 6; i++) {
      for (j = 0; j < 6; j++) {
        sprintf(buffer, "&column name=D%ld%ld, symbol=\"D$b%ld%ld$n\", type=double ", i + 1, j + 1, i + 1, j + 1);
        if (i == j)
          strcpy_ss(t, " &end");
        else
          sprintf(t, "units=%s/%s &end", unit[i], unit[j]);
        strcat(buffer, t);
        if (!SDDS_ProcessColumnString(&SDDSmatrix, buffer, 0)) {
          SDDS_SetError("Problem defining SDDS matrix output Dij columns (setUpMatrixOutput)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
      }
    }
    if (!SDDS_WriteLayout(&SDDSmatrix)) {
      SDDS_SetError("Problem writing SDDS layout (setUpMatrixOutput)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    matrixOutputInitialized = 1;
#if USE_MPI
  }
#endif
}

void outputMomentsMatrices(VMATRIX *M, double *D) {
  long i, j, index;
#if USE_MPI
  if (myid == 0) {
#endif
    if (!SDDS_StartPage(&SDDSmatrix, 1))
      bombElegant("problem starting page in output file for transfer and diffusion matrix", NULL);
    index = 0;
    for (i = 0; i < 6; i++) {
      if (!SDDS_SetRowValues(&SDDSmatrix, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, 0,
                             index++, M->C[i], -1)) {
        SDDS_SetError("Problem setting SDDS matrix output Ci columns (outputMomentsMatrices)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
    }
    for (i = 0; i < 6; i++) {
      for (j = 0; j < 6; j++) {
        if (!SDDS_SetRowValues(&SDDSmatrix, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, 0,
                               index++, M->R[i][j], -1)) {
          SDDS_SetError("Problem setting SDDS matrix output Rij columns (outputMomentsMatrices)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
      }
    }
    for (i = 0; i < 6; i++) {
      for (j = 0; j < 6; j++) {
        if (!SDDS_SetRowValues(&SDDSmatrix, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, 0,
                               index++, D[sigmaIndex3[i][j]], -1)) {
          SDDS_SetError("Problem setting SDDS matrix output Sij columns (outputMomentsMatrices)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
      }
    }
    if (!SDDS_WritePage(&SDDSmatrix)) {
      SDDS_SetError("Problem writing SDDS matrix output (outputMomentsMatrices)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
#if USE_MPI
  }
#endif
}

void dumpBeamMoments(
                     LINE_LIST *beamline,
                     long n_elem,
                     long final_values_only,
                     long tune_corrected,
                     RUN *run,
                     double *eNatural,
                     double *eConvergence,
                     long ibsIteration,
                     long ibsIterations,
                     double charge) {
  double data[N_COLUMNS];
  /* double *emit; */
  long j, row_count, elemCheck;
  char *stage;
  ELEMENT_LIST *elem;
  SIGMA_MATRIX *sigma0;

  if (tune_corrected == 1)
    stage = "tunes corrected";
  else
    stage = "tunes uncorrected";

  if (!SDDS_StartTable(&SDDSMoments, final_values_only ? 1 : n_elem + 1)) {
    SDDS_SetError("Problem starting SDDS table (dumpBeamMoments)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  if (!SDDS_SetParameters(&SDDSMoments, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE,
                          IP_STEP, momentsCount, IP_STAGE, stage,
                          IP_PCENTRAL, run->p_central, -1)) {
    SDDS_SetError("Problem setting SDDS parameters (dumpBeamMoments 1)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  elem = beamline->elem_twiss;
  sigma0 = beamline->sigmaMatrix0;
  /* emit = data+IC_EMITTANCE; */
  if (!final_values_only) {
    row_count = 0;
    data[IC_PCENTRAL] = elem->Pref_input;
    prepareMomentsArray(data, elem, (double *)sigma0->sigma);
    data[IC_S] = sStart; /* position for initial values */
    for (j = IC_S; j < N_COLUMNS; j++)
      if (!SDDS_SetRowValues(&SDDSMoments, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, row_count, j, data[j], -1)) {
        SDDS_SetError("Problem setting SDDS rows (dumpBeamMoments)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
    if (!SDDS_SetRowValues(&SDDSMoments, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, row_count++,
                           IC_ELEMENT, "_BEG_", IC_OCCURENCE, (long)1, IC_TYPE, "MARK", 
                           IC_COULOMB_LOG, elem->succ->coulombLog, -1)) {
      SDDS_SetError("Problem setting SDDS rows (dumpBeamMoments)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }

    elemCheck = 0;
    while (elem) {
      data[IC_S] = elem->end_pos; /* position */
      data[IC_PCENTRAL] = elem->Pref_output;
      if (!elem->sigmaMatrix)
        bombElegant("Sigma matrix data not computed prior to dumpBeamMoments() call (2)", NULL);
      prepareMomentsArray(data, elem, elem->sigmaMatrix->sigma);
      for (j = IC_S; j < N_COLUMNS; j++)
        if (!SDDS_SetRowValues(&SDDSMoments, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, row_count, j, data[j], -1)) {
          SDDS_SetError("Problem setting SDDS rows (dumpBeamMoments)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
      if (!SDDS_SetRowValues(&SDDSMoments, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, row_count,
                             IC_ELEMENT, elem->name, IC_OCCURENCE, elem->occurence,
                             IC_TYPE, entity_name[elem->type],
                             IC_COULOMB_LOG, elem->coulombLog,  -1)) {
        SDDS_SetError("Problem setting SDDS rows (dumpBeamMoments)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
      elemCheck++;
      row_count++;
      elem = elem->succ;
    }
    if (elemCheck != n_elem)
      bombElegant("element count error in dumpBeamMoments()", NULL);
  } else {
    /* find final element */
    elemCheck = 0;
    while (1) {
      if (!elem->sigmaMatrix)
        bombElegant("Sigma matrix data not computed prior to dumpBeamMoments() call (3)", NULL);
      elemCheck++;
      if (!elem->succ)
        break;
      elem = elem->succ;
    }
    if (elemCheck != n_elem)
      bombElegant("element count error in dumpBeamMoments()", NULL);
    prepareMomentsArray(data, elem, elem->sigmaMatrix->sigma);
    for (j = IC_S; j < N_COLUMNS; j++)
      if (!SDDS_SetRowValues(&SDDSMoments, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, 0, j, data[j], -1)) {
        SDDS_SetError("Problem setting SDDS rows (dumpBeamMoments)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
    if (!SDDS_SetRowValues(&SDDSMoments, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, 0,
                           IC_ELEMENT, elem->name, IC_OCCURENCE, elem->occurence,
                           IC_TYPE, entity_name[elem->type],
                           IC_COULOMB_LOG, elem->coulombLog, -1)) {
      SDDS_SetError("Problem setting SDDS rows (dumpBeamMoments)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  }

  if (!SDDS_SetParameters(&SDDSMoments, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE,
                          IP_E1, eNatural[0],
                          IP_E2, eNatural[1],
                          IP_E3, eNatural[2],
                          IP_E1_CONVERGENCE, eConvergence[0],
                          IP_E2_CONVERGENCE, eConvergence[1],
                          IP_E3_CONVERGENCE, eConvergence[2],
                          IP_IBS_ITERATION, ibsIteration+1,
                          IP_IBS_ITERATIONS, ibsIterations,
                          IP_CHARGE, charge, -1)) {
    SDDS_SetError("Problem setting SDDS emittance parameters (dumpBeamMoments)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  if (!SDDS_WriteTable(&SDDSMoments)) {
    SDDS_SetError("Unable to write Twiss parameter data (dumpBeamMoments)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  if (!inhibitFileSync)
    SDDS_DoFSync(&SDDSMoments);
  if (!SDDS_EraseData(&SDDSMoments)) {
    SDDS_SetError("Unable to erase Twiss parameter data (dumpBeamMoments)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  if (tune_corrected)
    momentsCount++;
}

void setupMomentsOutput(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline, long *doMomentsOutput,
                        long default_order) {
  /* process namelist input */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&moments_output, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &moments_output);

#if USE_MPI
  if (!writePermitted)
    filename = matrix_output = NULL;
#endif

  if (filename)
    filename = compose_filename(filename, run->rootname);
  if (matrix_output)
    matrix_output = compose_filename(matrix_output, run->rootname);
  *doMomentsOutput = output_at_each_step;
  trackingBasedDiffusionMatrixParticles = tracking_based_diffusion_matrix_particles;
  if (ibs_iterations) {
    ELEMENT_LIST *eptr;
    short chargePresent = 0;
    eptr = beamline->elem;
    if (ibs_iterations<5)
      printWarning("ibs_iterations < 5", "This is not recommended.");
    while (eptr) {
      if (eptr->type == T_CHARGE) {
	chargePresent = 1;
	break;
      }
      eptr = eptr->succ;
    }
    if (!chargePresent)
      bombElegant("ibs_iterations is non-zero but no CHARGE element in the beamline", NULL);
  }
  if (reference_file && matched)
    bombElegant("reference_file and matched=1 are incompatible", NULL);
  if (!matched) {
    if (reference_file) {
      if (reference_element && reference_element_occurrence < 0)
        bombElegant("invalid value of reference_element_occurrence---use 0 for last occurrence, >=1 for specific occurrence.", NULL);
      bombElegant("reference file feature not implemented yet.", NULL);
    }
    if (beta_x <= 0 || beta_y <= 0 || beta_z <= 0 || emit_x < 0 || emit_y < 0 || emit_z < 0)
      bombElegant("invalid initial beta-functions given in moments_output namelist", NULL);
  }

  if (filename) {
    SDDS_ElegantOutputSetup(&SDDSMoments, filename, SDDS_BINARY, 1, "Beam moments",
                            run->runfile, run->lattice,
                            parameter_definition, N_PARAMETERS,
                            column_definition, N_COLUMNS, "setupMomentsOutput",
                            SDDS_EOS_NEWFILE | SDDS_EOS_COMPLETE);
    SDDSMomentsInitialized = 1;
    momentsCount = 0;
  } else
    SDDSMomentsInitialized = 0;
  momentsInitialized = 1;
  if (matrix_output)
    setUpMomentsMatrixOutput(run, matrix_output);
}

void finishMomentsOutput(void) {
  if (SDDSMomentsInitialized && !SDDS_Terminate(&SDDSMoments)) {
    SDDS_SetError("Problem terminating SDDS output (finishMomentsOutput)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  SDDSMomentsInitialized = momentsCount = momentsInitialized = 0;
  if (matrixOutputInitialized && !SDDS_Terminate(&SDDSmatrix)) {
    SDDS_SetError("Problem terminating SDDS output (finishMomentsOutput)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  matrixOutputInitialized = 0;
}

long runMomentsOutput(RUN *run, LINE_LIST *beamline, double *startingCoord, long tune_corrected, long writeToFile) {
  ELEMENT_LIST *eptr, *elast;
  long n_elem, last_n_elem, i;
  double eNatural[3] = {0, 0, 0};
  double eNaturalPrev[3], eConvergence[3]={0,0,0};
  double charge = 0, charge0 = 0;
  
#ifdef DEBUG
  printf("now in runMomentsOutput\n");
  fflush(stdout);
#endif

  if (!momentsInitialized)
    return 1;

  if (tune_corrected == 0 && !output_before_tune_correction)
    return 1;

  if (ibs_iterations && (ibs_coulomb_log<=0 && !(beamline->radIntegrals.computed)))
    bombElegant("IBS computations requested in moments_output, but Coulomb log is zero and radiation integrals were not computed in twiss_output", NULL);
                
  /* Computations will start at the beginning of the beamline, or at the
   * first recirculation element
   */
  eptr = beamline->elem_twiss = beamline->elem;
  n_elem = last_n_elem = beamline->n_elems;
  while (eptr) {
    if (eptr->type == T_CHARGE)
      charge0 = ((CHARGE*)(eptr->p_elem))->charge;
    if (eptr->type == T_RECIRC) {
      last_n_elem = n_elem;
      beamline->elem_twiss = beamline->elem_recirc = eptr;
    }
    eptr = eptr->succ;
    n_elem--;
  }
  n_elem = last_n_elem;

  if (!beamline->sigmaMatrix0)
    beamline->sigmaMatrix0 = tmalloc(sizeof(*(beamline->sigmaMatrix0)));

  if (verbosity > 0) {
    if (!ibs_iterations)
      printf("\nPerforming beam moments computation.\n");
    else
      printf("\nPerforming beam moments computation including IBS with total charge of %le C per bunch.\n", charge0);
    fflush(stdout);
  }

  for (int iteration=0; (ibs_iterations==0 && iteration==0) || (ibs_iterations>0 && iteration<ibs_iterations); iteration++) {
    if (verbosity>1 && ibs_iterations)
      report_stats(stdout, "Performing ibs iteration: ");
    if (beamline->Mld) {
      free_matrices(beamline->Mld);
      free(beamline->Mld);
      beamline->Mld = NULL;
    }

    if (startingCoord) {
      VMATRIX *M1;
      M1 = tmalloc(sizeof(*M1));
      initialize_matrices(M1, 1);
      for (i = 0; i < 6; i++) {
	M1->C[i] = startingCoord[i];
	M1->R[i][i] = 1;
      }
      if (verbosity > 1)
	printf("Computing matrix with starting coordinates (%e, %e, %e, %e, %e, %e)\n",
	       startingCoord[0],
	       startingCoord[1],
	       startingCoord[2],
	       startingCoord[3],
	       startingCoord[4],
	       startingCoord[5]);
      beamline->Mld = accumulateRadiationMatrices(beamline->elem_twiss, run, M1, 1, radiation, n_slices, slice_etilted, iteration);
      free_matrices(M1);
      free(M1);
      M1 = NULL;
    } else {
      if (verbosity > 1) {
	printf("Computing matrix without starting coordinates\n");
	fflush(stdout);
      }
      beamline->Mld = accumulateRadiationMatrices(beamline->elem_twiss, run, NULL, 1, radiation, n_slices, slice_etilted, iteration);
    }
    
    if (verbosity > 0) {
      char text[200];
      sprintf(text, "** One-turn, on-orbit matrix with%sradiation:", radiation ? " " : "out ");
      print_matrices(stdout, text, beamline->Mld);
      fflush(stdout);
    }
    if (verbosity > 1) {
      long j;
      printf("** One-turn diffusion matrix: \n");
      for (i = 0; i < 6; i++)
	for (j = 0; j < 6; j++)
	  printf("%13.6e%c", beamline->elast->accumD[sigmaIndex3[i][j]], j == 5 ? '\n' : ' ');
      fflush(stdout);
    }
    
    if (matrixOutputInitialized)
      outputMomentsMatrices(beamline->Mld, beamline->elast->accumD);

    if (equilibrium) {
      if (verbosity > 1) {
        printf("Computing equilibrium moments\n");
        fflush(stdout);
      }
      /* Compute equilibrium moments */
      determineEquilibriumMoments(beamline->Mld->R, beamline->elast->accumD, beamline->sigmaMatrix0);
    } else {
      if (matched) {
	/* Use periodic lattice functions */
	if (!(beamline->flags & BEAMLINE_TWISS_DONE))
	  bombElegant("no twiss parameters computed for matched moments propogation", NULL);
	/* Determine starting moments from twiss parameter values */
	setStartingMoments(beamline->sigmaMatrix0,
			   emit_x, beamline->twiss0->betax, beamline->twiss0->alphax, beamline->twiss0->etax, beamline->twiss0->etapx,
			   emit_y, beamline->twiss0->betay, beamline->twiss0->alphay, beamline->twiss0->etay, beamline->twiss0->etapy,
			   emit_z, beta_z, alpha_z);
      } else
	/* Determine starting moments from twiss parameter values */
	setStartingMoments(beamline->sigmaMatrix0,
			   emit_x, beta_x, alpha_x, eta_x, etap_x,
			   emit_y, beta_y, alpha_y, eta_y, etap_y,
			   emit_z, beta_z, alpha_z);
    }
    if (verbosity > 1) {
      long j;
      printf("** Starting sigma matrix: \n");
      for (i = 0; i < 6; i++)
	for (j = 0; j < 6; j++)
	  printf("%13.6e%c", beamline->sigmaMatrix0->sigma[sigmaIndex3[i][j]], j == 5 ? '\n' : ' ');
      printf("Propagating beam moments\n");
      fflush(stdout);
    }

    /* Propagate moments to each element */
    propagateBeamMoments(run, beamline, startingCoord);

    elast = beamline->elast;
    if (verbosity > 1) {
      long j;
      printf("** Final sigma matrix: \n");
      for (i = 0; i < 6; i++)
	for (j = 0; j < 6; j++)
	  printf("%13.6e%c", elast->sigmaMatrix->sigma[sigmaIndex3[i][j]], j == 5 ? '\n' : ' ');
      fflush(stdout);
    }

    // set eNaturalPrev to the value of eNatural on the previous iteration
    eNaturalPrev[0] = eNatural[0];
    eNaturalPrev[1] = eNatural[1];
    eNaturalPrev[2] = eNatural[2];
    if (equilibrium) {
      if (verbosity>1) {
        printf("Computing natural emittances\n");
        fflush(stdout);
      }
      computeNaturalEmittances(beamline->Mld, beamline->sigmaMatrix0->sigma, eNatural);
      if (verbosity>1) {
        printf("e1 = %le, e2 = %le, e3 = %le\n", eNatural[0], eNatural[1], eNatural[2]);
        fflush(stdout);
      }
    }
    if (ibs_iterations) {
      if (iteration==0) {
        eNaturalPrev[0] = eNatural[0];
        eNaturalPrev[1] = eNatural[1];
        eNaturalPrev[2] = eNatural[2];	// for first iteration set the two to be equal
      }
      for (int i=0; i<3; i++)
        eConvergence[i] = fabs(eNaturalPrev[i]-eNatural[i])/eNatural[i];
    }
    if (ibs_iterations && iteration!=(ibs_iterations-1)) {
      if (iteration>=(ibs_iterations-3)) {
        charge = charge0;
      } else {
        // slowly change charge and emittance used to compute ibs to aid convergence
        charge = charge*(1-ibs_iteration_fraction) + charge0*ibs_iteration_fraction;
        eNatural[0] = eNatural[0]*(1.0 - ibs_iteration_fraction) + eNaturalPrev[0]*ibs_iteration_fraction;
        eNatural[1] = eNatural[1]*(1.0 - ibs_iteration_fraction) + eNaturalPrev[1]*ibs_iteration_fraction;
        eNatural[2] = eNatural[2]*(1.0 - ibs_iteration_fraction) + eNaturalPrev[2]*ibs_iteration_fraction;
      }
      if (verbosity>1) {
        printf("Computing IBS scattering matrices for q=%le C\n",charge);
        fflush(stdout);
      }
      updateIbsScatteringMatrices(beamline, charge, eNatural);
    }
    
    if (SDDSMomentsInitialized && writeToFile &&
        (!ibs_iterations || ibs_output_iterations || (!ibs_output_iterations && iteration==(ibs_iterations-1))))
      dumpBeamMoments(beamline, n_elem, final_values_only, tune_corrected, run, eNatural, eConvergence,
                      iteration, ibs_iterations, charge0);
    
    if (beamline->elem->sigmaMatrix) {
      for (i = 0; i < 6; i++) {
	long j;
	for (j = 0; j < 6; j++)
	  savedFinalMoments[i][j] = beamline->elem->sigmaMatrix->sigma[sigmaIndex3[i][j]];
	savedFinalCentroid[i] = beamline->elem->Mld->C[i];
      }
    }
  }

  if (verbosity>1 && ibs_iterations)
    report_stats(stdout, "Finished ibs iterations: ");

#ifdef DEBUG
  eptr = beamline->elem_twiss;
  while (eptr) {
    int i, j;
    if (strcmp(eptr->name, "B1")==0) {
      for (i=0; i<6; i++) {
	printf("D[%d] for %s#%ld: ", i, eptr->name, eptr->occurence);
	for (j=0; j<=i; j++)
	  printf("%13.6e ", eptr->DIbs[sigmaIndex3[i][j]]);
	printf("\n");
      }
    }
    eptr = eptr->succ;
  }
#endif

  return 1;
}

void setStartingMoments(SIGMA_MATRIX *sm,
                        double emit_x, double beta_x, double alpha_x, double eta_x, double etap_x,
                        double emit_y, double beta_y, double alpha_y, double eta_y, double etap_y,
                        double emit_z, double beta_z, double alpha_z) {
  double sDelta;

  sm->sigma[sigmaIndex3[4][4]] = emit_z * beta_z;
  sm->sigma[sigmaIndex3[5][5]] = emit_z / beta_z * (1 + sqr(alpha_z));
  sm->sigma[sigmaIndex3[4][5]] = -emit_z * alpha_z;
  sDelta = sqrt(sm->sigma[sigmaIndex3[5][5]]);

  sm->sigma[sigmaIndex3[0][0]] = emit_x * beta_x + sqr(sDelta * eta_x);
  sm->sigma[sigmaIndex3[1][1]] = emit_x / beta_x * (1 + sqr(alpha_x)) + sqr(sDelta * etap_x);
  sm->sigma[sigmaIndex3[0][1]] = -emit_x * alpha_x + sqr(sDelta) * eta_x * etap_x;
  sm->sigma[sigmaIndex3[0][5]] = sqr(sDelta) * eta_x;
  sm->sigma[sigmaIndex3[1][5]] = sqr(sDelta) * etap_x;

  sm->sigma[sigmaIndex3[2][2]] = emit_y * beta_y + sqr(sDelta * eta_y);
  sm->sigma[sigmaIndex3[3][3]] = emit_y / beta_y * (1 + sqr(alpha_y)) + sqr(sDelta * etap_y);
  sm->sigma[sigmaIndex3[2][3]] = -emit_y * alpha_y + sqr(sDelta) * eta_y * etap_y;
  sm->sigma[sigmaIndex3[2][5]] = sqr(sDelta) * eta_y;
  sm->sigma[sigmaIndex3[3][5]] = sqr(sDelta) * etap_y;
}

void fillSigmaPropagationMatrix(double **Ms, double **R) {
  long i, j, k, l, m;
  double Rik;

  for (i = 0; i < 6; i++) {
    for (j = i; j < 6; j++) {
      m = sigmaIndex3[i][j];
      for (k = 0; k < 6; k++) {
        Rik = R[i][k];
        for (l = k; l < 6; l++) {
          Ms[m][sigmaIndex3[k][l]] = Rik * R[j][l] + (k != l ? R[i][l] * R[j][k] : 0);
        }
      }
    }
  }
}

void propagateBeamMoments(RUN *run, LINE_LIST *beamline, double *traj) {
  long i, j;
  ELEMENT_LIST *elem;
  VMATRIX *M1, *M2, *Me;
  SIGMA_MATRIX *S1, *S2;
  /* double path[6]; */
  MATRIX *Ms;

  /* Allocate memory to store sigma matrix as we propagate, copy initial matrix */
  S1 = tmalloc(sizeof(*S1));
  S2 = tmalloc(sizeof(*S1));
  memcpy(S1, beamline->sigmaMatrix0, sizeof(*S1));

  M1 = tmalloc(sizeof(*M1));
  M2 = tmalloc(sizeof(*M2));
  initialize_matrices(M1, 1);
  initialize_matrices(M2, 1);
  if (traj) {
    for (i = 0; i < 6; i++) {
      /* path[i] = traj[i]; */
      M1->R[i][i] = 1;
    }
  } else {
    for (i = 0; i < 6; i++) {
      /* path[i] = 0; */
      M1->R[i][i] = 1;
    }
  }

  elem = beamline->elem_twiss;
  m_alloc(&Ms, 21, 21);
  while (elem) {
    if (!(elem->sigmaMatrix))
      elem->sigmaMatrix = tmalloc(sizeof(*(elem->sigmaMatrix)));

    if ((Me = elem->Mld)) {
      fillSigmaPropagationMatrix(Ms->a, Me->R);

      for (i = 0; i < 21; i++) {
        S2->sigma[i] = 0;
        for (j = 0; j < 21; j++) {
          S2->sigma[i] += Ms->a[i][j] * S1->sigma[j];
        }
      }
      if (elem->D)
        for (i = 0; i < 21; i++)
          S2->sigma[i] += elem->D[i];
      memcpy(elem->sigmaMatrix, S2, sizeof(*S2));
      memcpy(S1, S2, sizeof(*S2));
    } else
      /* Assume it doesn't modify the sigma matrix */
      memcpy(elem->sigmaMatrix, S1, sizeof(*S1));
    if (elem->type == T_MARK && ((MARK *)elem->p_elem)->fitpoint)
      storeFitpointMomentsParameters(elem);
    elem = elem->succ;
  }

  free_matrices(M1);
  free(M1);
  M1 = NULL;
  free_matrices(M2);
  free(M2);
  M2 = NULL;
  free(S1);
  free(S2);
  m_free(&Ms);
}

void determineEquilibriumMoments(
                                 double **R,                /* revolution matrix (input) */
                                 double *D,                 /* diffusion matrix (input) */
                                 SIGMA_MATRIX *sigmaMatrix0 /* sigma matrix (output) */
                                 ) {
  MATRIX *Ms, *Md, *M1, *M2, *M3;
  long i;

  m_alloc(&Ms, 21, 21); /* sigma matrix propagator */
  m_alloc(&Md, 21, 1);  /* diffiusion matrix */
  m_alloc(&M1, 21, 21); /* work matrix */
  m_alloc(&M2, 21, 21); /* work matrix */
  m_alloc(&M3, 21, 1);  /* work matrix */

  fillSigmaPropagationMatrix(Ms->a, R);

  m_zero(Md);
  for (i = 0; i < 21; i++)
    Md->a[i][0] = D[i];

  /* S = Inv(I-Ms) * D */
  m_identity(M1);
  m_subtract(M2, M1, Ms);
  m_invert(M1, M2);
  m_mult(M3, M1, Md);

  for (i = 0; i < 21; i++)
    sigmaMatrix0->sigma[i] = M3->a[i][0];

  m_free(&Ms);
  m_free(&Md);
  m_free(&M1);
  m_free(&M2);
  m_free(&M3);
}

void storeFitpointMomentsParameters(ELEMENT_LIST *elem) {
  MARK *mark;
  char *name;
  long occurence;
  SIGMA_MATRIX *sigma0;
  double *centroid, data[N_COLUMNS];
  char s[1000];
  long i, j, offset, c;
  char plane[3] = "xy";

  mark = (MARK *)elem->p_elem;
  name = elem->name;
  occurence = elem->occurence;
  sigma0 = elem->sigmaMatrix;
  centroid = elem->Mld->C;
  prepareMomentsArray(data, elem, (double *)sigma0->sigma);

  if (!(mark->init_flags & 32)) {
    mark->moments_mem = tmalloc(sizeof(*(mark->moments_mem)) * (21 + 10 + 6 + 2));
    mark->init_flags |= 32;
    offset = 0;
    for (i = 0; i < 21; i++) {
      sprintf(s, "%s#%ld.s%ld%ldm", name, occurence,
              sigmaIndex1[i] + 1, sigmaIndex2[i] + 1);
      mark->moments_mem[offset++] = rpn_create_mem(s, 0);
    }
    for (i = 0; i < 4; i++) {
      for (j = i; j < 4; j++) {
        sprintf(s, "%s#%ld.s%ld%ldbetam", name, occurence, i + 1, j + 1);
        mark->moments_mem[offset] = rpn_create_mem(s, 0);
        offset++;
      }
    }
    for (i = 0; i < 6; i++) {
      sprintf(s, "%s#%ld.c%ldm", name, occurence, i + 1);
      mark->moments_mem[offset++] = rpn_create_mem(s, 0);
    }
    for (i = 0; i < 2; i++) {
      sprintf(s, "%s#%ld.e%cbetam", name, occurence, plane[i]);
      mark->moments_mem[offset++] = rpn_create_mem(s, 0);
    }
    if (offset != (21 + 10 + 6 + 2))
      bombElegant("Counting issue (1) for moments-related RPN variables", NULL);
  }

  offset = 0;
  for (i = 0; i < 21; i++)
    rpn_store(sigma0->sigma[i], NULL, mark->moments_mem[offset++]);
  for (i = c = 0; i < 4; i++) {
    for (j = i; j < 4; j++, c++) {
      rpn_store(i == j ? sqr(data[IC_SBETA + c]) : data[IC_SBETA + c], NULL, mark->moments_mem[offset++]);
    }
  }
  for (i = 0; i < 6; i++)
    rpn_store(centroid[i], NULL, mark->moments_mem[offset++]);
  for (i = 0; i < 2; i++)
    rpn_store(data[IC_EMITBETA + i], NULL, mark->moments_mem[offset++]);
  if (offset != (21 + 10 + 6 + 2))
    bombElegant("Counting issue (2) for moments-related RPN variables", NULL);

  /*
    printf("*********************\n");
    offset = 0;
    for (i=0; i<21; i++) {
    sprintf(s, "%s#%ld.s%ld%ldm", name, occurence,
    sigmaIndex1[i]+1, sigmaIndex2[i]+1);
    printf("%s#%ld.s%ld%ldm = %le = %le = %le\n", name, occurence, sigmaIndex1[i]+1, sigmaIndex2[i]+1, sigma0->sigma[i],
    rpn_recall(rpn_create_mem(s, 0)), rpn_recall(mark->moments_mem[offset++]));
    }
    for (i=c=0; i<4; i++) {
    for (j=i; j<4; j++, c++) {
    sprintf(s, "%s#%ld.s%ld%ldbetam", name, occurence, i+1, j+1);
    printf("%s#%ld.s%ld%ldbetam = %le = %le = %le\n", name, occurence, i+1, j+1,
    i==j?sqr(data[IC_SBETA+c]):data[IC_SBETA+c],
    rpn_recall(rpn_create_mem(s, 0)),
    rpn_recall(mark->moments_mem[offset++]));
    }
    }
    for (i=0; i<6; i++) {
    sprintf(s, "%s#%ld.c%ldm", name, occurence, i+1);
    printf("%s#%ld.c%ld = %le = %le = %le\n",
    name, occurence, i+1, centroid[i],
    rpn_recall(rpn_create_mem(s, 0)),
    rpn_recall(mark->moments_mem[offset++]));
    }
    for (i=0; i<2; i++) {
    sprintf(s, "%s#%ld.e%cbetam", name, occurence, plane[i]);
    printf("%s#%ld.e%cbetam = %le = %le = %le\n", name, occurence, plane[i],
    data[IC_EMITBETA+i], rpn_recall(rpn_create_mem(s, 0)), rpn_recall(mark->moments_mem[offset++]));
    }
  */
}

void prepareMomentsArray(double *data, ELEMENT_LIST *elem, double *sigma) {
  long i, j, k, l, plane;
  double *emit;
  double sBeta[4][4];

  for (i = 0; i < N_COLUMNS; i++)
    data[i] = -1;

  data[IC_S] = elem->end_pos; /* position */
  data[IC_PCENTRAL] = elem->Pref_output;
  copy_doubles(data + IC_PCENTRAL + 1, sigma, 21);
  copy_doubles(data + IC_PCENTRAL + 1 + 21, elem->Mld->C, 6);
  for (i = 0; i < 6; i++) {
    k = sigmaIndex3[i][i];
    data[IC_PCENTRAL + 1 + k] = sqrt(data[IC_PCENTRAL + 1 + k]);
  }

  emit = data + IC_EMITTANCE;
  for (plane = 0; plane < 3; plane++) {
    emit[plane] = sigma[sigmaIndex3[0 + plane * 2][0 + plane * 2]] * sigma[sigmaIndex3[1 + plane * 2][1 + plane * 2]] -
      sqr(sigma[sigmaIndex3[0 + plane * 2][1 + plane * 2]]);
    if (emit[plane] > 0)
      emit[plane] = sqrt(emit[plane]);
    else
      emit[plane] = -1;
  }

  /* compute betatron quantities */
  for (i = l = 0; i < 4; i++)
    for (j = i; j < 4; j++, l++) {
      data[IC_SBETA + l] = sigma[sigmaIndex3[i][j]] -
        sigma[sigmaIndex3[i][5]] * sigma[sigmaIndex3[j][5]] / sigma[sigmaIndex3[5][5]];
      sBeta[i][j] = sBeta[j][i] = data[IC_SBETA + l];
      if (i == j) {
        if (data[IC_SBETA + l] > 0)
          data[IC_SBETA + l] = sqrt(data[IC_SBETA + l]);
        else
          data[IC_SBETA + l] = 0;
      }
    }

  emit = data + IC_EMITBETA;
  for (plane = 0; plane < 2; plane++) {
    emit[plane] = sBeta[2 * plane][2 * plane] * sBeta[2 * plane + 1][2 * plane + 1] -
      sqr(sBeta[2 * plane][2 * plane + 1]);
    if (emit[plane] > 0)
      emit[plane] = sqrt(emit[plane]);
    else
      emit[plane] = -1;
  }
}

/* Following code is from V. Sajaev's calculateEnvelopes.c program, adapted to
 * elegant by M. Borland and V. Sajaev.
 */

#define MATDIM 6
#define MATDIM2 36

void NormalizeEigenvectors(int dim, double *V, int debug);
double *AddMM1(int sum, int rows, int cols, double *M1, double *M2);
double *MatrixProduct1(int rows1, int cols1, double *T1, int rows2, int cols2, double *T2);
double *TransposeM(int rows, int cols, double *M);
void MatrixPrintout1(char *string, double *AA, int N, int M);
#if defined(LAPACK) || defined(CLAPACK) || defined(MKL)
  int dgeev_(char *JOBVL, char *JOBVR, int *N, double *A,
         int *LDA, double *WR, double *WI, double *VL,
         int *LDVL, double *VR, int *LDVR, double *work,
         int *lwork, int *info);
#endif

void computeNaturalEmittances(VMATRIX *Mld, double *sigmaMatrix, double *emittance) {
  int i, j, k, eigenModesNumber, dim = MATDIM;
  double WR[MATDIM], WI[MATDIM], VL[MATDIM2], VR[MATDIM2], M[MATDIM2], Mcopy[MATDIM2], work[1000];
  double *Rdiag;
  char JOBVL, JOBVR;
  int N, LDA, LDVL, LDVR, lwork, info;
  double ReV[MATDIM2], ImV[MATDIM2];
  double *M1, *M2, *M3, *M4;
  static char *enRpnName[3] = {
    "e1m", "e2m", "e3m"};
  static long enRpnMemory[3] = {
    -1, -1, -1};

  eigenModesNumber = 3;

  /* Copy the revolution matrix into the working buffer */
  for (i = 0; i < MATDIM; i++)
    for (j = 0; j < MATDIM; j++) {
      M[i * MATDIM + j] = Mld->R[i][j];
    }

  memcpy(Mcopy, M, sizeof(*Mcopy) * MATDIM2);

  /*--- Calculating eigenvectors using dgeev ... */
  /* VR is right-hand side eigenvectors such that: VR^transp * M = lamdba * VR^transp
     VR[0 to 5] are real components of vector 1, VR[6 to 11] are imaginary components of vector 1 and so on.
  */
#if defined(LAPACK) || defined(CLAPACK) || defined(MKL)
  JOBVL = 'N';
  JOBVR = 'V';
  N = LDA = LDVR = MATDIM;
  LDVL = 1;
  lwork = 1000;
  dgeev_((char *)&JOBVL, (char *)&JOBVR, (int *)&N, (double *)&Mcopy,
         (int *)&LDA, (double *)&WR, (double *)&WI, (double *)&VL,
         (int *)&LDVL, (double *)&VR, (int *)&LDVR, (double *)&work,
         (int *)&lwork, (int *)&info);
#else
  fprintf(stderr, "Error calling dgeev. You will need to install LAPACK and rebuild elegant\n");
  exitElegant(1);
#endif

  if (info != 0) {
    if (info < 0) {
      printf("Error calling dgeev, argument %d.\n", abs(info));
    }
    if (info > 0) {
      printf("Error running dgeev, calculation of eigenvalue number %d failed.\n", info);
    }
    exitElegant(1);
  }

  /*--- Sorting of eigenvalues and eigenvectors according to (x,y,z)... */
  SortEigenvalues(WR, WI, VR, dim, eigenModesNumber, 0);

  /*--- Normalization of eigenvectors... */
  NormalizeEigenvectors(dim, VR, 0);

  /*--- Assembling diagonalizing matrix V ---*/
  for (k = 0; k < eigenModesNumber; k++) {
    for (i = 0; i < dim; i++) {
      ReV[k * 2 * dim + i] = VR[k * 2 * dim + i];
      ReV[(k * 2 + 1) * dim + i] = -VR[(k * 2 + 1) * dim + i];
    }
  }
  for (k = 0; k < eigenModesNumber; k++) {
    for (i = 0; i < dim; i++) {
      ImV[2 * k * dim + i] = VR[(2 * k + 1) * dim + i];
      ImV[(2 * k + 1) * dim + i] = -VR[2 * k * dim + i];
    }
  }

  /* Copy the sigma matrix into the working buffer */
  for (i = 0; i < MATDIM; i++)
    for (j = 0; j < MATDIM; j++)
      Mcopy[i * MATDIM + j] = sigmaMatrix[sigmaIndex3[i][j]];

  M3 = MatrixProduct1(dim, dim, ReV, dim, dim, Mcopy);
  M4 = TransposeM(dim, dim, ReV);
  M1 = MatrixProduct1(dim, dim, M3, dim, dim, M4);
  free(M3);
  free(M4);

  M3 = MatrixProduct1(dim, dim, ImV, dim, dim, Mcopy);
  M4 = TransposeM(dim, dim, ImV);
  M2 = MatrixProduct1(dim, dim, M3, dim, dim, M4);
  free(M3);
  free(M4);

  Rdiag = AddMM1(1, dim, dim, M1, M2);
  free(M1);
  free(M2);

  emittance[0] = Rdiag[0];
  emittance[1] = Rdiag[2 * dim + 2];
  emittance[2] = Rdiag[4 * dim + 4];

  /* Store these in rpn memories in case needed by optimizer */
  for (i = 0; i < 3; i++) {
    if (enRpnMemory[i] == -1)
      enRpnMemory[i] = rpn_create_mem(enRpnName[i], 0);
    rpn_store(emittance[i], NULL, enRpnMemory[i]);
  }

  free(Rdiag);
}

void NormalizeEigenvectors(int dim, double *V, int debug) {
  int k, i, eigenModesNumber;
  double Norm[3], Vcopy[MATDIM2];
  eigenModesNumber = 3;

  for (i = 0; i < MATDIM2; i++)
    Vcopy[i] = V[i];
  for (k = 0; k < eigenModesNumber; k++) {
    Norm[k] = 0;
    for (i = 0; i < eigenModesNumber; i++) {
      /* Index = Irow*dim + Icolumn */
      Norm[k] += (V[2 * k * dim + 2 * i + 1] * V[(2 * k + 1) * dim + 2 * i] - V[2 * k * dim + 2 * i] * V[(2 * k + 1) * dim + 2 * i + 1]) * 2;
    }
    Norm[k] = -1.0 / sqrt(fabs(Norm[k]));
    if (debug == 4)
      printf("Normalization coefficient[%d]= %12.4e \n", k, Norm[k]);
  }
  for (k = 0; k < eigenModesNumber; k++) {
    for (i = 0; i < dim * 2; i++) {
      V[k * 2 * dim + i] = Vcopy[k * 2 * dim + i] * Norm[k];
    }
  }
}

double *AddMM1(int sum, int rows, int cols, double *M1, double *M2)
/* Adds two matrices */
{
  int i, j;
  double *M3;
  M3 = malloc(sizeof(*M3) * cols * rows);
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      if (sum == 1) {
        M3[i * cols + j] = M1[i * cols + j] + M2[i * cols + j];
      } else {
        M3[i * cols + j] = M1[i * cols + j] - M2[i * cols + j];
      }
    }
  }
  return M3;
}

double *MatrixProduct1(int rows1, int cols1, double *T1, int rows2, int cols2, double *T2)
/* Calculates T3=T1*T2 */
{
  double *T3;
  int i, j, k;
  T3 = malloc(sizeof(*T3) * rows1 * cols2);
  if (cols1 != rows2) {
    printf("Wrong matrix dimension!\n");
    exitElegant(1);
  }
  for (i = 0; i < rows1; i++) {
    for (j = 0; j < cols2; j++) {
      T3[i * cols2 + j] = 0;
      for (k = 0; k < cols1; k++) {
        T3[i * cols2 + j] += T1[i * cols1 + k] * T2[k * cols2 + j];
      }
    }
  }
  return T3;
}

double *TransposeM(int rows, int cols, double *M) {
  int i, j;
  double *Mt;
  Mt = malloc(sizeof(*Mt) * cols * rows);
  for (i = 0; i < rows; i++)
    for (j = 0; j < cols; j++)
      Mt[j * cols + i] = M[i * cols + j];
  return Mt;
}

void MatrixPrintout1(char *string, double *AA, int Nrow, int Ncol) {
  int i, j;
  printf("%s\n", string);
  for (i = 0; i < Nrow; i++) {
    for (j = 0; j < Ncol; j++) {
      printf("%16.8e", AA[i * Ncol + j]);
    }
    printf("\n");
  }
  printf("\n");
}

long getMoments(double M[6][6], double C[6], long matched0, long equilibrium0, long radiation0) {
  long i, j;

  if (!momentsInitialized)
    bombElegant("Error: moments calculation not initialized. Did you set output_at_each_step=1?\n", NULL);
  if (matched0 != matched || equilibrium0 != equilibrium || radiation0 != radiation)
    bombElegantVA("Error: moments calculation not run with matched=%ld, equilibrium=%ld, and radiation=%ld\n",
                  matched0, equilibrium0, radiation0);

  for (i = 0; i < 6; i++)
    for (j = 0; j < 6; j++)
      M[i][j] = savedFinalMoments[i][j];
  for (i = 0; i < 6; i++)
    C[i] = savedFinalCentroid[i];

  return 1;
}


static inline void diagonalization_matrix_3x3(MATRIX *F, double eval[3], MATRIX *Mevec, int do_sort)
/*
  Fast diagonalization of a symmetric 3x3 matrix F (2D array).
  
  Input:
    F        - symmetric matrix (row-major in C). Only upper/lower triangle is used.
    do_sort  - if nonzero, eigenvalues are sorted ascending and eigenvectors permuted.

  Output:
    eval[3]  - eigenvalues
    Mevec    - matrix of eigenvectors Mevec->a[i] is the ith eigenvector

  Author: ChatGPT 5, with modification for numerical stability by M. Borland
*/
{
    double evec[3][3]; /* local storage of the eigen vectors */
    // Copy & symmetrize
    double A[3][3], maxAbs;
    for (int i=0;i<3;++i)
        for (int j=0;j<3;++j)
            A[i][j] = 0.5*(F->a[i][j] + F->a[j][i]);

    // normalize for numerical stability
    maxAbs = 0;
    for (int i=0;i<3;++i)
      for (int j=0;j<3;++j) {
        if (fabs(A[i][j])>maxAbs)
          maxAbs = fabs(A[i][j]);
      }
    for (int i=0;i<3;++i)
      for (int j=0;j<3;++j)
        A[i][j] /= maxAbs;
    
    // Initialize eigenvectors as identity
    memset(evec, 0, 9*sizeof(double));
    evec[0][0]=1.0; evec[1][1]=1.0; evec[2][2]=1.0;

    const int max_sweeps = 10;
    const double tol = 1e-15;

    for (int sweep=0; sweep<max_sweeps; ++sweep) {
        double off = fabs(A[0][1]) + fabs(A[0][2]) + fabs(A[1][2]);
        if (off < tol * (fabs(A[0][0])+fabs(A[1][1])+fabs(A[2][2]))) break;

        int pairs[3][2] = {{0,1},{0,2},{1,2}};
        for (int k=0;k<3;++k) {
            int p = pairs[k][0], q = pairs[k][1];
            double apq = A[p][q];
            if (fabs(apq) < tol) continue;

            double app = A[p][p], aqq = A[q][q];
            double tau = (aqq - app)/(2.0*apq);
            double t = copysign(1.0/(fabs(tau)+sqrt(1.0+tau*tau)), tau);
            double c = 1.0/sqrt(1.0+t*t);
            double s = t*c;
            double theta = s/(1.0+c);

            // Zero out p,q off-diagonal
            A[p][q] = A[q][p] = 0.0;
            A[p][p] = app - t*apq;
            A[q][q] = aqq + t*apq;

            // Update other elements
            for (int r=0;r<3;++r) if (r!=p && r!=q) {
                double arp = A[r][p], arq = A[r][q];
                double nrp = arp - s*(arq + theta*arp);
                double nrq = arq + s*(arp - theta*arq);
                A[r][p] = A[p][r] = nrp;
                A[r][q] = A[q][r] = nrq;
            }

            // Update eigenvectors
            for (int r=0;r<3;++r) {
                double vrp = evec[r][p];
                double vrq = evec[r][q];
                evec[r][p] = vrp - s*(vrq + theta*vrp);
                evec[r][q] = vrq + s*(vrp - theta*vrq);
            }
        }
    }

    // Eigenvalues
    eval[0]=A[0][0]; eval[1]=A[1][1]; eval[2]=A[2][2];

    if (do_sort) {
      // Sort eigenvalues ascending and reorder eigenvectors
      int idx[3]={0,1,2};
      for (int i=0;i<3;++i) for (int j=i+1;j<3;++j) {
	  if (eval[j] < eval[i]) {
            double te=eval[i]; eval[i]=eval[j]; eval[j]=te;
            int ti=idx[i]; idx[i]=idx[j]; idx[j]=ti;
	  }
	}
      
      double V[3][3];
      memcpy(V, evec, 9*sizeof(double));
      for (int r=0;r<3;++r) {
        evec[r][0]=V[r][idx[0]];
        evec[r][1]=V[r][idx[1]];
        evec[r][2]=V[r][idx[2]];
      }
    }

    for (int i=0; i<3; i++) {
      eval[i] *= maxAbs;
      for (int j=0; j<3; j++)
	Mevec->a[i][j] = evec[j][i];
    }
}

void m_schur(MATRIX *F, MATRIX *C)
/* Compute the 3x3 momentum moments matrix conditional on locality using the Schur-complement identity
   F = C[pp] - C[px] Inv(C[xx]) C[xp]
   where C[pp][ij] = <pi*pj>, C[xx][ij] = <xi*xj>, and C[px][ij] = C[xp][ji] = <pi*xj>
*/
{
  static short initialized = 0;
  static MATRIX *Cxx, *Cpp, *Cxp, *Cpx, *InvCxx, *temp1, *temp2;
  long i, j;
  if (!initialized) {
    initialized = 1;
    m_alloc(&Cxx, 3, 3);
    m_alloc(&Cpp, 3, 3);
    m_alloc(&Cxp, 3, 3);
    m_alloc(&Cpx, 3, 3);
    m_alloc(&InvCxx, 3, 3);
    m_alloc(&temp1, 3, 3);
    m_alloc(&temp2, 3, 3);
    initialized = 1;
  }
  /* Extract submatrices */
  for (i=0; i<3; i++) {
    for (j=0; j<3; j++) {
      Cxx->a[i][j] = C->a[2*i][2*j];
      Cpp->a[i][j] = C->a[2*i+1][2*j+1];
      Cxp->a[i][j] = Cpx->a[j][i] = C->a[2*i][2*j+1];
    }
  }
  m_invert(InvCxx, Cxx);
  m_mult(temp1, Cpx, InvCxx);
  m_mult(temp2, temp1, Cxp);
  m_subtract(F, Cpp, temp2);
}

/* Code to compute integrals in K&O equation 70, by ChatGPT.
 * Performs non-adapative gaussian quadrature.
 */

static const double x32[16] = {
 0.0483076656877383, 0.1444719615827965, 0.2392873622521371, 0.3318686022821277,
 0.4213512761306353, 0.5068999089322294, 0.5877157572407623, 0.6630442669302152,
 0.7321821187402897, 0.7944837959679424, 0.8493676137325700, 0.8963211557660521,
 0.9349060759377390, 0.9647622555875064, 0.9856115115452684, 0.9972638618494816
};
static const double w32[16] = {
 0.0965400885147278, 0.0956387200792749, 0.0938443990808046, 0.0911738786957639,
 0.0876520930044038, 0.0833119242269467, 0.0781938957870703, 0.0723457941088485,
 0.0658222227763618, 0.0586840934785355, 0.0509980592623762, 0.0428358980222267,
 0.0342738629130214, 0.0253920653092621, 0.0162743947309057, 0.0070186100094701
};

// integrand for gi
static double integrand(double s, int i, const double u[3]) {
    double ss = sin(s), cc = cos(s);
    double num = 2.0*u[i]*ss*ss*cc;
    double d1,d2;
    if (i==0) {
        d1 = ss*ss + (u[0]/u[1])*cc*cc;
        d2 = ss*ss + (u[0]/u[2])*cc*cc;
    } else if (i==1) {
        d1 = ss*ss + (u[1]/u[0])*cc*cc;
        d2 = ss*ss + (u[1]/u[2])*cc*cc;
    } else {
        d1 = ss*ss + (u[2]/u[0])*cc*cc;
        d2 = ss*ss + (u[2]/u[1])*cc*cc;
    }
    return num / sqrt(d1*d2);
}

double compute_gi(int i, const double u[3])
{
    const double a = 0.0, b = M_PI_2;
    const double c1 = 0.5*(b-a);
    const double c2 = 0.5*(b+a);
    double sum = 0.0;
    for (int k=0;k<16;++k) {
        double dx = c1 * x32[k];
        double s1 = c2 - dx, s2 = c2 + dx;
        sum += w32[k]*( integrand(s1,i,u) + integrand(s2,i,u) );
    }
    return c1*sum;
}

/* Based on Kubo and Oide, PRST-AB 4, 124401 (2001) */
/* A human wrote this part! */
void updateIbsScatteringMatrices(LINE_LIST *beamline, double charge, double *eGeometric)
{

  ELEMENT_LIST *eptr;
  double coulombLog;
  double Ci, betaGamma, gamma, p0;
  double g[3], dw2dts[3], length, duration, eigVal[3], u[3];
  int i, j;
  static short initialized = 0;
  static MATRIX *Sigma, *Cp, *C, *D, *Dt, *ToMom, *Boost, *DeBoost, *F, *dppdt, *dppdtLab, *temp6x6, *temp3x3, *dw2dt,
    *xyzSigma;
  static double *diagElements;
  eptr = beamline->elem_twiss;

  if (ibs_coulomb_log>0)
    coulombLog = ibs_coulomb_log;

  if (!initialized) {
    initialized = 1;
    m_alloc(&Sigma, 6, 6);
    m_alloc(&xyzSigma, 6, 6);
    m_alloc(&Cp, 6, 6);
    m_alloc(&C, 6, 6);
    m_alloc(&ToMom, 6, 6);
    m_alloc(&Boost, 6, 6);
    m_alloc(&DeBoost, 3, 3);
    m_alloc(&temp6x6, 6, 6);
    m_alloc(&D, 3, 3);
    m_alloc(&Dt, 3, 3);
    m_alloc(&F, 3, 3);
    m_alloc(&dw2dt, 3, 3);
    m_alloc(&dppdt, 3, 3);
    m_alloc(&dppdtLab, 3, 3);
    m_alloc(&temp3x3, 3, 3);
    diagElements = malloc(sizeof(*diagElements)*6);
    initialized = 1;
  }

  while (eptr) {
#ifdef DEBUG
    printf("Working on %s\n", eptr->name);
#endif
    eptr->coulombLog = -1;
    if (!eptr->DIbs) {
      eptr->DIbs = malloc(sizeof(*eptr->DIbs)*21);
#ifdef DEBUG
      printf("DIbs not set for %s#%ld. Allocating\n", eptr->name, eptr->occurence);
#endif
    }
    if (!(entity_description[eptr->type].flags&HAS_LENGTH) || (length=((DRIFT*)(eptr->p_elem))->length) <= 0) {
      /* Diffusion matrix is zero */
#ifdef DEBUG
      printf("Element %s has zero length.\n", eptr->name);
#endif
      memset(eptr->DIbs, 0, 21*sizeof(*eptr->DIbs));
      if (eptr->pred)
        eptr->coulombLog = eptr->pred->coulombLog;
    } else {
      /* Assume for now that the beam properties at the end of an element are representative of those throughout the element.
	 This needs to be re-examined for elements (e.g., LGBEND, CCBEND) that can't be split.
       */

#ifdef DEBUG
      printf("Element %s has length %le.\n", eptr->name, length);
#endif
      
      /* 1. Assemble the lab-frame sigma matrix Sigma in the usual (x, x', y, y', z, delta) coordinates */
      for (int i=0; i<6; i++)
	for (int j=0; j<6; j++)
	  Sigma->a[i][j] = eptr->sigmaMatrix->sigma[sigmaIndex3[i][j]];
#ifdef DEBUG
      m_show(Sigma, "%13.6e ", "Sigma:\n", stdout);
#endif

      /* 2. The sigma matrix is in terms of geometric quantities (x, x', y, y', s, deltaP/P) 
	 Transform approximately to (x, px, y, py, dz, dpz) coordinates using C' = P*Sigma*Trans(P), where P = diag(1, p0, 1, p0, 1, p0)
	 and p0 = m*c*beta*gamma.
      */
      memset(diagElements, 0, 6*sizeof(*diagElements));
      diagElements[0] = diagElements[2] = diagElements[4] = 1;
      betaGamma = (eptr->Pref_input+eptr->Pref_output)/2; /* beta*gamma dimensionless momentum */
      p0 = me_mks*c_mks*betaGamma; /* reference momentum in SI units */
#ifdef DEBUG
      printf("betaGamma = %le, p0 = %le\n", betaGamma, p0);
#endif
      diagElements[1] = diagElements[3] = diagElements[5] = p0;
      m_diag(ToMom, diagElements);
      m_mult(temp6x6, ToMom, Sigma);
      m_mult(Cp, temp6x6, ToMom);
#ifdef DEBUG
      m_show(Cp, "%13.6e ", "Cp:\n", stdout);
#endif
	     
      /* 3. Transform C to the co-moving frame using C = T*C'*Trans(T), where T = diag(1, 1, 1, 1, gamma, 1/gamma) */
      memset(diagElements, 0, 6*sizeof(*diagElements));
      diagElements[0] = diagElements[1] = diagElements[2] = diagElements[3] = 1;
      diagElements[4] = gamma = sqrt(betaGamma*betaGamma+1);
      diagElements[5] = 1/gamma;
      m_diag(Boost, diagElements);
      m_mult(temp6x6, Boost, Cp);
      m_mult(C, temp6x6, Boost);
#ifdef DEBUG
      m_show(C, "%13.6e ", "C:\n", stdout);
#endif
      
      /* 4. Compute the 3x3 momentum moments matrix conditional on locality using the Schur-complement identity
	 F = C[pp] - C[px] Inv(C[xx]) C[xp]
	 where C[pp][ij] = <pi*pj>, C[xx][ij] = <xi*xj>, and C[px][ij] = C[xp][ji] = <pi*xj>
      */
      m_schur(F, C);
#ifdef DEBUG
      m_show(F, "%13.6e ", "F:\n", stdout);
#endif
      
      /* 5. Find matrix D (K&O call this matrix R) to diagonalize F. We'll need D and Trans(D) later,
       * and the eigenvalues immediately.
       */
      diagonalization_matrix_3x3(F, u, D, 0);
      m_trans(Dt, D);
#ifdef DEBUG
      m_show(D, "%13.6e ", "D:\n", stdout);
#endif

      /* 6. Compute g[i] integrals (Eq. 70 from K&O). These depend on the diagonal elements of G, which
       * are just the eigenvalues of F 
       */
      for (i=0; i<3; i++)
	g[i] = compute_gi(i, u);

      if (ibs_coulomb_log>0) {
        coulombLog = ibs_coulomb_log;
      } else {
        double meanDensity, bMax, bMin1, bMin2, bMin, v, tau;
        /* 6.5 Compute the Coulomb log */
        /* First, compute bMax */
        /* form the spatial sigma matrix in the co-moving frame */
        /* printf("Coulomb log calculation:\n"); */
        for (i=0; i<6; i+=2)
          for (j=0; j<6; j+=2)
            xyzSigma->a[i/2][j/2] = C->a[i][j];
        /* compute the eigenvalues */
        diagonalization_matrix_3x3(xyzSigma, eigVal, temp3x3, 0);
        /* printf("eigenvalues %le, %le %le\n", eigVal[0], eigVal[1], eigVal[2]); */
        /* mean particle meanDensity */
        meanDensity = charge/particleCharge/8/sqrt(ipow(PI,3)*eigVal[0]*eigVal[1]*eigVal[2]);
        /* printf("meanDensity = %le\n", meanDensity); */
        /* K&O Eq. 78 */
        bMax = min_double(4, sqrt(eigVal[0]), sqrt(eigVal[1]), sqrt(eigVal[2]), pow(meanDensity, -1./3.));
        /* printf("bMax = %le\n", bMax); */
        /* Second, compute bMin */
        /* maxmium of damping times in co-moving frame */
        tau = max_double(3, beamline->radIntegrals.taux, beamline->radIntegrals.tauy, beamline->radIntegrals.taudelta)/gamma;
        /* printf("tau = %le\n", tau); */
        /* typical velocity */
        v = sqrt(u[0]+u[1]+u[2])/me_mks;
        bMin1 = 1/sqrt(PI*meanDensity*v*tau);
        bMin2 = sqrt(2)*particleRadius*sqr(particleMass*c_mks)/(u[0]+u[1]+u[2]);
        /* printf("v = %le, bMin1 = %le, bMin2 = %le\n", v, bMin1, bMin2); */
        bMin = max_double(2, bMin1, bMin2);
        /* printf("bMin = %le\n", bMin); */
        coulombLog = log(bMax/bMin);
      }
      /* printf("Coulomb log = %le\n", coulombLog); */
      eptr->coulombLog = coulombLog;
      
#ifdef DEBUG
      printf("particleCharge = %le, charge = %le, Gamma = %le, CL = %le \n => Ci = %le\n",
	     particleCharge, charge,  Gamma, coulombLog, Ci);
#endif
      
      /* 7. Compute d<wi^2>/dt for i=1,2,3 (Eq. 65) */
      Ci = sqr(particleRadius)*(charge/particleCharge)*coulombLog/
        (4*PI*ipow(gamma,3)*eGeometric[0]*eGeometric[1]*(eGeometric[2]/c_mks));
      dw2dts[0] = Ci*((g[1] - g[0]) + (g[2] - g[0]));
      dw2dts[1] = Ci*((g[0] - g[1]) + (g[2] - g[1]));
      dw2dts[2] = Ci*((g[0] - g[2]) + (g[1] - g[2]));
      m_diag(dw2dt, dw2dts);
#ifdef DEBUG
      m_show(dw2dt, "%13.6e ", "d<w2>/dt:\n", stdout);
#endif
      
      /* 8. Compute d<pi*pj>/dt for i=1,2,3 j=1,2,3 (Eq. 71)
       * d<pi*pj>/dt = D*d<w^2>/dt*Trans(D)
       */
      m_mult(temp3x3, D, dw2dt);
      m_mult(dppdt, temp3x3, Dt);
#ifdef DEBUG
      m_show(dppdt, "%13.6e ", "d<pi*pj>/dt:\n", stdout);
#endif

      /* 9. Transform d<pi*pj>/dt to lab frame. 
       * (d<pi*pj>/dt)Lab = (M d<pi*pj>/dt Trans(M))/gamma
       */
      memset(diagElements, 0, 6*sizeof(*diagElements));
      diagElements[0] = 1;
      diagElements[1] = 1;
      diagElements[2] = gamma;
      m_diag(DeBoost, diagElements);
      m_mult(temp3x3, DeBoost, dppdt);
      m_mult(dppdtLab, temp3x3, DeBoost);
      m_scmul(dppdtLab, dppdtLab, 1./gamma);
#ifdef DEBUG
      m_show(dppdtLab, "%13.6e ", "d<pi*pj>/dt(Lab):\n", stdout);
#endif

      /* 10. Compute change over element length and assign to IBS diffusion matrix */
      /* Also, convert p^2 to x', y', delta */
      duration = length/(c_mks*betaGamma/gamma)/sqr(betaGamma*me_mks*c_mks);
#ifdef DEBUG
      printf("duration = %le s\n", duration);
#endif
      memset(eptr->DIbs, 0, 21*sizeof(*eptr->DIbs));
      for (i=0; i<3; i++)
	for (j=0; j<3; j++)
	  eptr->DIbs[sigmaIndex3[2*i+1][2*j+1]] = dppdtLab->a[i][j]*duration;

#ifdef DEBUG
      for (i=0; i<6; i++) {
	printf("D[%d] for %s: ", i, eptr->name);
	for (j=0; j<=i; j++)
	  printf("%13.6e ", eptr->DIbs[sigmaIndex3[i][j]]);
	printf("\n");
      }
#endif
    }
    eptr = eptr->succ;
  }

  /* Supply CL values for initial zero-length elements, if any */
  eptr = beamline->elem_twiss;
  while (eptr && eptr->coulombLog<=0) {
    eptr->coulombLog = coulombLog;
    eptr = eptr->succ;
  }
}

