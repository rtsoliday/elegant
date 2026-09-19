/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: response.c
 * purpose: trajectory/orbit correction response matrix output for elegant
 * 
 * Michael Borland, 1993
 */
#include "mdb.h"
#include "track.h"
#include "correctDefs.h"

extern char *correction_mode[2];

typedef struct {
  SDDS_TABLE SDDSout;
  long monitors, correctors;
  char **monitorName;
  long sIndex, monitorNameIndex, *correctorIndex;
} RESPONSE_OUTPUT;
RESPONSE_OUTPUT xRespOutput, yRespOutput, xInvRespOutput, yInvRespOutput;
RESPONSE_OUTPUT xyRespOutput, yxRespOutput;
RESPONSE_OUTPUT xpRespOutput, ypRespOutput;

#define NORMAL_UNITS 0
#define KNL_UNITS 1
#define BNL_UNITS 2

static CORMON_DATA CMyx, CMxy;

void setup_response_output(RESPONSE_OUTPUT *respOutput,
                           char *filename, char *type, RUN *run, char *beamline_name, CORMON_DATA *CM, STEERING_LIST *SL,
                           long bpmPlane, long corrPlane, long inverse, long unitsType, long slopeCode);
void do_response_output(RESPONSE_OUTPUT *respOutput, CORMON_DATA *CM, STEERING_LIST *SL, long plane,
                        long inverse, long KnL_units, long tune_corrected);
#include "response.h"

void setup_correction_matrix_output(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline, CORRECTION *correct,
                                    long *do_response,
                                    long BnLUnitsOK) {
  long unitsCode;
  log_entry("setup_correction_matrix_output");
  if (correct->mode == -1) {
    printf("Error: you must request orbit/trajectory correction before requesting response matrix output.\n");
    printf("Otherwise, elegant doesn't have the needed information about correctors and monitors.\n");
    exitElegant(1);
  }

  /* process the namelist text */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&correction_matrix_output, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &correction_matrix_output);
  *do_response = output_at_each_step;

  if (correct->CMFx->fixed_length_matrix != fixed_length)
    bombElegantVA("Inconsistent values of parameters: &correct fixed_length_matrix=%ld and &response_matrix_output fixed_length=%ld",
		  correct->CMFx->fixed_length_matrix,fixed_length);
  if (correct->CMFy->fixed_length_matrix != fixed_length)
    bombElegantVA("Inconsistent values of parameters: &correct fixed_length_matrix=%ld and &response_matrix_output fixed_length=%ld",
		  correct->CMFy->fixed_length_matrix,fixed_length);

  unitsCode = KnL_units ? KNL_UNITS : (BnL_units ? BNL_UNITS : 0);
  if (unitsCode == BNL_UNITS && !BnLUnitsOK)
    bombElegant("At present you must give the matrix_output or twiss_output command to use BnL_units=1.  Sorry.", NULL);

  if (coupled) {
    /* set up the coupled response matrix data structures */

    /* CMyx is the vertical response to horizontal correctors */
    /* --- copy corrector information */
    CMyx.ncor = correct->CMFx->ncor;
    CMyx.kick_coef = correct->CMFx->kick_coef;
    CMyx.sl_index = correct->CMFx->sl_index;
    CMyx.ucorr = correct->CMFx->ucorr;
    /* --- copy BPM information */
    CMyx.nmon = correct->CMFy->nmon;
    CMyx.umoni = correct->CMFy->umoni;
    CMyx.mon_index = correct->CMFy->mon_index;
    CMyx.C = NULL;
    CMyx.Cp = NULL;
    CMyx.bpmPlane = 1;
    CMyx.corrPlane = 0;
    CMyx.fixed_length = CMyx.fixed_length_matrix = fixed_length;
    CMyx.weight = correct->CMFy->weight;
    CMyx.use_perturbed_matrix = 0;

    /* CMxy is the horizontal response to vertical correctors */
    /* --- copy corrector information */
    CMxy.ncor = correct->CMFy->ncor;
    CMxy.kick_coef = correct->CMFy->kick_coef;
    CMxy.sl_index = correct->CMFy->sl_index;
    CMxy.ucorr = correct->CMFy->ucorr;

    /* --- copy BPM information */
    CMxy.nmon = correct->CMFx->nmon;
    CMxy.umoni = correct->CMFx->umoni;
    CMxy.mon_index = correct->CMFx->mon_index;
    CMxy.C = NULL;
    CMxy.Cp = NULL;
    CMxy.bpmPlane = 0;
    CMxy.corrPlane = 1;
    CMxy.fixed_length = CMxy.fixed_length_matrix = fixed_length;
    CMxy.weight = correct->CMFx->weight;
    CMxy.use_perturbed_matrix = 0;
  }

#if USE_MPI
  if (isSlave)
    return;
#endif

  if (correct->method==COUPLED_CORRECTION) {
    if (response[0])
      setup_response_output(&xRespOutput, response[0], correction_mode[correct->mode], run, beamline->name,
                            correct->CMFx, &correct->SLx, 2, 2, 0, unitsCode, 0);
    if (slope_response[0])
      setup_response_output(&xpRespOutput, slope_response[0], correction_mode[correct->mode], run, beamline->name,
                            correct->CMFx, &correct->SLx, 2, 2, 0, unitsCode, 1);
    if (inverse[0])
      setup_response_output(&xInvRespOutput, inverse[0], correction_mode[correct->mode], run, beamline->name,
                            correct->CMFx, &correct->SLx, 2, 2, 1, unitsCode, 0);
  } else {
    if (response[0])
      setup_response_output(&xRespOutput, response[0], correction_mode[correct->mode], run, beamline->name,
                            correct->CMFx, &correct->SLx, 0, 0, 0, unitsCode, 0);
    if (response[1])
      setup_response_output(&yRespOutput, response[1], correction_mode[correct->mode], run, beamline->name,
                            correct->CMFy, &correct->SLy, 1, 1, 0, unitsCode, 0);
    
    if (response[2])
      setup_response_output(&yxRespOutput, response[2], correction_mode[correct->mode], run, beamline->name,
                            &CMyx, &correct->SLx, 1, 0, 0, unitsCode, 0);
    if (response[3])
      setup_response_output(&xyRespOutput, response[3], correction_mode[correct->mode], run, beamline->name,
                            &CMxy, &correct->SLy, 0, 1, 0, unitsCode, 0);
    
    if (slope_response[0])
      setup_response_output(&xpRespOutput, slope_response[0], correction_mode[correct->mode], run, beamline->name,
                            correct->CMFx, &correct->SLx, 0, 0, 0, unitsCode, 1);
    if (slope_response[1])
      setup_response_output(&ypRespOutput, slope_response[1], correction_mode[correct->mode], run, beamline->name,
                            correct->CMFy, &correct->SLy, 1, 1, 0, unitsCode, 1);
    
    if (inverse[0])
      setup_response_output(&xInvRespOutput, inverse[0], correction_mode[correct->mode], run, beamline->name,
                            correct->CMFx, &correct->SLx, 0, 0, 1, unitsCode, 0);
    if (inverse[1])
      setup_response_output(&yInvRespOutput, inverse[1], correction_mode[correct->mode], run, beamline->name,
                            correct->CMFy, &correct->SLy, 1, 1, 1, unitsCode, 0);
  }
  
  log_exit("setup_correction_matrix_output");
}

void setup_response_output(RESPONSE_OUTPUT *respOutput,
                           char *filename, char *type, RUN *run, char *beamline_name, CORMON_DATA *CM, STEERING_LIST *SL,
                           long bpmPlane, long correctorPlane, long inverse, long unitsCode, long slopeCode) {
  ELEMENT_LIST *eptr;
  static char s[256], t[256], units[32];
  long i, j, *unique_name, sl_index;
  long matrixType;
  char *matrixTypeName[5] = {"horizontal", "vertical", "v->h", "h->v", "coupled"};

  if (bpmPlane==2) {
    /* two-plane BPM data */
    if (correctorPlane!=2)
      bombElegant("Coding error: setup_response_output has two-plane BPM data by one-plane corrector data", NULL);
    matrixType = 4;
  } else
    matrixType = bpmPlane == correctorPlane ? bpmPlane : 2 + bpmPlane;

  log_entry("setup_response_output");
  filename = compose_filename(filename, run->rootname);
  if (!inverse) {
    sprintf(s, "%s-plane %s %sfixed path-length response matrix for beamline %s of lattice %s",
            matrixTypeName[matrixType], type,
            fixed_length ? "" : "non-", beamline_name, run->lattice);
    sprintf(t, "%s-plane %s %sfixed path-length response matrix", matrixTypeName[matrixType], type,
            fixed_length ? "" : "non-");
  } else {
    sprintf(s, "%s-plane %s transposed inverse %sfixed path-length response matrix for beamline %s of lattice %s",
            matrixTypeName[matrixType], type,
            fixed_length ? "" : "non-", beamline_name, run->lattice);
    sprintf(t, "%s-plane %s transposed inverse %sfixed path-length response matrix", matrixTypeName[matrixType],
            type, fixed_length ? "" : "non-");
  }

  if (!SDDS_InitializeOutputElegant(&respOutput->SDDSout, SDDS_BINARY, 0, s, t, filename)) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    exitElegant(1);
  }
#if RW_ASSOCIATES
  SDDS_DefineAssociate(&respOutput->SDDSout, "elegantInput",
                       run->runfile, getenv("PWD"), "elegant input file used to create this file",
                       "elegant input, parent", 0);
  SDDS_DefineAssociate(&respOutput->SDDSout, "elegantLattice",
                       run->lattice, getenv("PWD"), "elegant lattice file used to create this file",
                       "elegant lattice, parent", 0);
#endif
  respOutput->monitorNameIndex =
    SDDS_DefineColumn(&respOutput->SDDSout, "BPMName", NULL, NULL, "beam-position-monitor name", NULL, SDDS_STRING, 0);
  respOutput->sIndex =
    SDDS_DefineColumn(&respOutput->SDDSout, "s", NULL, "m", "beam-position-monitor location", NULL, SDDS_DOUBLE, 0);
  SDDS_DefineParameter(&respOutput->SDDSout, "CorrectionMatrixType", NULL, NULL, "correction matrix type",
                       NULL, SDDS_STRING, inverse ? "Inverse" : "Response");
  SDDS_DefineParameter1(&respOutput->SDDSout, "NCorrectors", NULL, NULL, "Number of correctors", NULL, SDDS_LONG,
                        &CM->ncor);
  SDDS_DefineParameter1(&respOutput->SDDSout, "NMonitors", NULL, NULL, "Number of correctors", NULL, SDDS_LONG,
                        &CM->nmon);
  SDDS_DefineParameter(&respOutput->SDDSout, "Stage", NULL, NULL, "Simulation stage", NULL, SDDS_STRING, NULL);
  SDDS_DefineParameter(&respOutput->SDDSout, "SVNVersion", NULL, NULL, "SVN version number", NULL, SDDS_STRING, SVN_VERSION);
  if (SDDS_NumberOfErrors())
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  respOutput->correctors = CM->ncor;
  respOutput->monitors = CM->nmon;
  respOutput->correctorIndex = tmalloc(sizeof(*respOutput->correctorIndex) * respOutput->correctors);
  respOutput->monitorName = tmalloc(sizeof(*respOutput->monitorName) * respOutput->monitors);

  unique_name = tmalloc(sizeof(*unique_name) * (CM->ncor));
  for (i = 0; i < CM->ncor; i++)
    unique_name[i] = 1;
  for (i = 0; i < CM->ncor; i++) {
    for (j = 0; j < i; j++) {
      if (strcmp(CM->ucorr[i]->name, CM->ucorr[j]->name) == 0)
        unique_name[i] = unique_name[j] = 0;
    }
  }


  for (i = 0; i < CM->ncor; i++) {
    sl_index = CM->sl_index[i];
    eptr = CM->ucorr[i];
    if (!inverse) {
      if (is_blank(entity_description[eptr->type].parameter[SL->param_index[sl_index]].unit)) {
        if (slopeCode)
          strcpy_ss(units, "rad");
        else
        strcpy_ss(units, "m");
      } else {
        if (unitsCode == KNL_UNITS && (eptr->type == T_HCOR || eptr->type == T_HVCOR || eptr->type == T_VCOR ||
                                       eptr->type == T_EHCOR || eptr->type == T_EHVCOR || eptr->type == T_EVCOR)) {
          if (slopeCode)
            sprintf(units, "rad/K0L");
          else
          sprintf(units, "m/K0L");
        }
        else if (unitsCode == BNL_UNITS && (eptr->type == T_HCOR || eptr->type == T_HVCOR || eptr->type == T_VCOR ||
                                            eptr->type == T_EHCOR || eptr->type == T_EHVCOR || eptr->type == T_EVCOR)) {
          if (slopeCode)
            sprintf(units, "rad/(T*m)");
          else
          sprintf(units, "1/T");
        } else {
          if (slopeCode)
            sprintf(units, "rad/%s", entity_description[eptr->type].parameter[SL->param_index[sl_index]].unit);
          else
          sprintf(units, "m/%s", entity_description[eptr->type].parameter[SL->param_index[sl_index]].unit);
          str_tolower(units);
        }
      }
      sprintf(s, "response to %s#%ld.%s", eptr->name, eptr->occurence, SL->corr_param[sl_index]);
    } else {
      if (is_blank(entity_description[eptr->type].parameter[SL->param_index[sl_index]].unit)) {
        if (slopeCode)
          strcpy_ss(units, "1/rad");
        else
        strcpy_ss(units, "1/m");
      } else {
        if (unitsCode == KNL_UNITS && (eptr->type == T_HCOR || eptr->type == T_HVCOR || eptr->type == T_VCOR ||
                                       eptr->type == T_EHCOR || eptr->type == T_EHVCOR || eptr->type == T_EVCOR)) {
          if (slopeCode)
            sprintf(units, "K0L/rad");
          else
          sprintf(units, "K0L/m");
        } else if (unitsCode == BNL_UNITS && (eptr->type == T_HCOR || eptr->type == T_HVCOR || eptr->type == T_VCOR ||
                                              eptr->type == T_EHCOR || eptr->type == T_EHVCOR || eptr->type == T_EVCOR)) {
          if (slopeCode)
            sprintf(units, "(T*m)/rad");
          else
          sprintf(units, "T");
        } else {
          if (slopeCode) 
            sprintf(units, "%s/rad", entity_description[eptr->type].parameter[SL->param_index[sl_index]].unit);
          else 
          sprintf(units, "%s/m", entity_description[eptr->type].parameter[SL->param_index[sl_index]].unit);
          str_tolower(units);
        }
      }
      sprintf(s, "inverse response to %s#%ld.%s", eptr->name, eptr->occurence, SL->corr_param[sl_index]);
    }
    if (unique_name[i]) {
      if (full_names)
        sprintf(t, "%s.%s", CM->ucorr[i]->name, entity_description[eptr->type].parameter[SL->param_index[sl_index]].name);
      else
        sprintf(t, "%s", CM->ucorr[i]->name);
    } else {
      if (full_names) {
        sprintf(t, "%s#%ld.%s", CM->ucorr[i]->name, CM->ucorr[i]->occurence, entity_description[eptr->type].parameter[SL->param_index[sl_index]].name);
      } else 
       sprintf(t, "%s#%ld", CM->ucorr[i]->name, CM->ucorr[i]->occurence);
    }
    respOutput->correctorIndex[i] =
      SDDS_DefineColumn(&respOutput->SDDSout, t, NULL, units, s, NULL, SDDS_DOUBLE, 0);
  }
  if (SDDS_NumberOfErrors()) {
    SDDS_PrintErrors(stderr, 1);
    exitElegant(1);
  }
  if (!SDDS_WriteLayout(&respOutput->SDDSout)) {
    SDDS_PrintErrors(stderr, 1);
    exitElegant(1);
  }

  unique_name = trealloc(unique_name, sizeof(*unique_name) * (CM->nmon));
  for (i = 0; i < CM->nmon; i++)
    unique_name[i] = 1;
  for (i = 0; i < CM->nmon; i++) {
    for (j = 0; j < i; j++) {
      if (strcmp(CM->umoni[i]->name, CM->umoni[j]->name) == 0)
        unique_name[i] = unique_name[j] = 0;
    }
  }

  for (i = 0; i < CM->nmon; i++) {
    if (!unique_name[i]) {
      sprintf(s, "%s#%ld", CM->umoni[i]->name, CM->umoni[i]->occurence);
      SDDS_CopyString(&respOutput->monitorName[i], s);
    } else
      SDDS_CopyString(&respOutput->monitorName[i], CM->umoni[i]->name);
  }
  set_namelist_processing_flags(0);
  set_print_namelist_flags(0);
  log_exit("setup_response_output");
}

/* This routine is only run by the optimizer.  It results in values being placed in
 * rpn memories for use in optimization.
 */

void update_response(RUN *run, LINE_LIST *beamline, CORRECTION *correct) {
  MAT *Cx, *Cy, *Tx, *Ty;

  /* Copy the matrices from the correction structure so we can put them back when we are done. 
     * We have to do this because the correction command may have different settings (e.g., fixed-length
     * constraint or non-perturbed matrix.
     */

  Cx = correct->CMFx->C;
  Tx = correct->CMFx->T;
  correct->CMFx->C = correct->CMFx->T = NULL;

  Cy = correct->CMFy->C;
  Ty = correct->CMFy->T;
  correct->CMFy->C = correct->CMFy->T = NULL;

  if (correct->method == COUPLED_CORRECTION) {
    /* The correction matrix is coupled */
    if (correct->mode != TRAJECTORY_CORRECTION) 
      bombElegant("correction method is COUPLED_CORRECTION but mode is not TRAJECTORY_CORRECTION", NULL);
    compute_coupled_trajcor_matrices(correct->CMFx, &correct->SLx, run, beamline, COMPUTE_RESPONSE_SILENT);
  } else {
    if (correct->mode == TRAJECTORY_CORRECTION) {
      compute_trajcor_matrices(correct->CMFx, &correct->SLx, 0, run, beamline, COMPUTE_RESPONSE_SILENT, correct->rpn_store_response_matrix);
      compute_trajcor_matrices(correct->CMFy, &correct->SLy, 2, run, beamline, COMPUTE_RESPONSE_SILENT, correct->rpn_store_response_matrix);
      if (coupled) {
        compute_trajcor_matrices(&CMxy, &correct->SLy, 0, run, beamline, COMPUTE_RESPONSE_SILENT, correct->rpn_store_response_matrix);
        compute_trajcor_matrices(&CMyx, &correct->SLx, 2, run, beamline, COMPUTE_RESPONSE_SILENT, correct->rpn_store_response_matrix);
      }
    } else if (correct->mode == ORBIT_CORRECTION) {
      if (!use_response_from_computed_orbits) {
        compute_orbcor_matrices(correct->CMFx, &correct->SLx, 0, run, beamline, COMPUTE_RESPONSE_SILENT | (fixed_length ? COMPUTE_RESPONSE_FIXEDLENGTH : 0), correct->rpn_store_response_matrix);
        compute_orbcor_matrices(correct->CMFy, &correct->SLy, 2, run, beamline, COMPUTE_RESPONSE_SILENT | (fixed_length ? COMPUTE_RESPONSE_FIXEDLENGTH : 0), correct->rpn_store_response_matrix);
      } else {
#if USE_MPI
        compute_orbcor_matrices1p(correct, run, beamline, COMPUTE_RESPONSE_SILENT | (fixed_length ? COMPUTE_RESPONSE_FIXEDLENGTH : 0), coupled, &CMxy, &CMyx);
#else
        compute_orbcor_matrices1(correct, 0, run, beamline, COMPUTE_RESPONSE_SILENT | (fixed_length ? COMPUTE_RESPONSE_FIXEDLENGTH : 0), coupled, &CMxy, &CMyx);
        compute_orbcor_matrices1(correct, 2, run, beamline, COMPUTE_RESPONSE_SILENT | (fixed_length ? COMPUTE_RESPONSE_FIXEDLENGTH : 0), coupled, &CMxy, &CMyx);
#endif
      }
    } else
      bombElegant("bad correction mode (update_response)", NULL);
  }
  
  /* copy matrices back to the correction structure and free memory */
  matrix_free(correct->CMFx->C);
  matrix_free(correct->CMFx->T);
  correct->CMFx->C = Cx;
  correct->CMFx->T = Tx;

  matrix_free(correct->CMFy->C);
  matrix_free(correct->CMFy->T);
  correct->CMFy->C = Cy;
  correct->CMFy->T = Ty;
}

void run_response_output(RUN *run, LINE_LIST *beamline, CORRECTION *correct, long tune_corrected) {
  long unitsCode;
  MAT *Cx, *Cy, *Tx, *Ty;
  unsigned long flags[2];
  
  unitsCode = KnL_units ? KNL_UNITS : (BnL_units ? BNL_UNITS : 0);
  if (tune_corrected == 0 && !output_before_tune_correction)
    return;

  /* Copy the matrices from the correction structure so we can put them back when we are done. 
     * We have to do this because the correction command may have different settings (e.g., fixed-length
     * constraint or non-perturbed matrix.
     */

  Cx = correct->CMFx->C;
  Tx = correct->CMFx->T;
  correct->CMFx->C = correct->CMFx->T = NULL;

  Cy = correct->CMFy->C;
  Ty = correct->CMFy->T;
  correct->CMFy->C = correct->CMFy->T = NULL;

  if (correct->method == COUPLED_CORRECTION) {
    /* The correction matrix is coupled */
    if (correct->mode != TRAJECTORY_CORRECTION) 
      bombElegant("correction method is COUPLED_CORRECTION but mode is not TRAJECTORY_CORRECTION", NULL);
    compute_coupled_trajcor_matrices(correct->CMFx, &correct->SLx, run, beamline,
                                     (!(inverse[0] == NULL || SDDS_StringIsBlank(inverse[0])) ? COMPUTE_RESPONSE_INVERT : 0) | COMPUTE_RESPONSE_SILENT);
  } else {
    if (correct->mode == TRAJECTORY_CORRECTION) {
      printf("Computing coupled trajectory correction matrices for output.\n");
      fflush(stdout);
      compute_trajcor_matrices(correct->CMFx, &correct->SLx, 0, run, beamline,
                               (!(inverse[0] == NULL || SDDS_StringIsBlank(inverse[0])) ? COMPUTE_RESPONSE_INVERT : 0) | COMPUTE_RESPONSE_SILENT, correct->rpn_store_response_matrix);
      compute_trajcor_matrices(correct->CMFy, &correct->SLy, 2, run, beamline,
                               (!(inverse[1] == NULL || SDDS_StringIsBlank(inverse[1])) ? COMPUTE_RESPONSE_INVERT : 0) | COMPUTE_RESPONSE_SILENT, correct->rpn_store_response_matrix);
      
      if (coupled) {
        compute_trajcor_matrices(&CMxy, &correct->SLy, 0, run, beamline, COMPUTE_RESPONSE_SILENT, correct->rpn_store_response_matrix);
        compute_trajcor_matrices(&CMyx, &correct->SLx, 2, run, beamline, COMPUTE_RESPONSE_SILENT, correct->rpn_store_response_matrix);
      }
    } else if (correct->mode == ORBIT_CORRECTION) {
      for (int i=0; i<2; i++)
	flags[i] = (!(inverse[i] == NULL || SDDS_StringIsBlank(inverse[i])) ? COMPUTE_RESPONSE_INVERT : 0) | COMPUTE_RESPONSE_SILENT |
	  (fixed_length ? COMPUTE_RESPONSE_FIXEDLENGTH : 0);
      if (!use_response_from_computed_orbits) {
        printf("Computing orbit correction matrices for output using beta functions, with %s length.\n",
               fixed_length ? "fixed" : "variable");
        fflush(stdout);
        compute_orbcor_matrices(correct->CMFx, &correct->SLx, 0, run, beamline, flags[0], correct->rpn_store_response_matrix);
        compute_orbcor_matrices(correct->CMFy, &correct->SLy, 2, run, beamline, flags[1], correct->rpn_store_response_matrix);
      } else {
        printf("Computing orbit correction matrices for output using closed orbits, with %s length.\n",
               fixed_length ? "fixed" : "variable");
        fflush(stdout);
#if USE_MPI
        compute_orbcor_matrices1p(correct, run, beamline, flags[0]|flags[1], coupled, &CMxy, &CMyx);
#else
        compute_orbcor_matrices1(correct, 0, run, beamline, flags[0]|flags[1], coupled, &CMxy, &CMyx);
        compute_orbcor_matrices1(correct, 2, run, beamline, flags[0]|flags[1], coupled, &CMxy, &CMyx);
#endif
      }
    } else
      bombElegant("bad correction mode (run_response_output)", NULL);
  }
  
#if USE_MPI
  if (!isSlave) {
#endif

    if (response[0])
      do_response_output(&xRespOutput, correct->CMFx, &correct->SLx, 0, 0, unitsCode, tune_corrected);
    if (inverse[0])
      do_response_output(&xInvRespOutput, correct->CMFx, &correct->SLx, 0, 1, unitsCode, tune_corrected);
    if (slope_response[0])
      do_response_output(&xpRespOutput, correct->CMFx, &correct->SLx, 1, 0, unitsCode, tune_corrected);
    if (correct->method!=COUPLED_CORRECTION) {
      if (response[1])
        do_response_output(&yRespOutput, correct->CMFy, &correct->SLy, 2, 0, unitsCode, tune_corrected);
      if (response[2])
        do_response_output(&yxRespOutput, &CMyx, &correct->SLx, 0, 0, unitsCode, tune_corrected);
      if (response[3])
        do_response_output(&xyRespOutput, &CMxy, &correct->SLy, 2, 0, unitsCode, tune_corrected);
      if (slope_response[1])
        do_response_output(&ypRespOutput, correct->CMFy, &correct->SLy, 3, 0, unitsCode, tune_corrected);
      if (inverse[1])
        do_response_output(&yInvRespOutput, correct->CMFy, &correct->SLy, 2, 1, unitsCode, tune_corrected);
    }
#if USE_MPI
  }
#endif

  /* copy matrices back to the correction structure and free memory */
  matrix_free(correct->CMFx->C);
  matrix_free(correct->CMFx->T);
  correct->CMFx->C = Cx;
  correct->CMFx->T = Tx;

  matrix_free(correct->CMFy->C);
  matrix_free(correct->CMFy->T);
  correct->CMFy->C = Cy;
  correct->CMFy->T = Ty;
}

void do_response_output(RESPONSE_OUTPUT *respOutput, CORMON_DATA *CM, STEERING_LIST *SL, long corrPlane,
                        long inverse, long unitsCode, long tune_corrected) {
  long i, j;
  ELEMENT_LIST *eptr;
  double value;

  if (!SDDS_StartTable(&respOutput->SDDSout, CM->nmon) ||
      !SDDS_SetParameters(&respOutput->SDDSout, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                          "Stage",
                          tune_corrected == 0 ? "tune uncorrected" : (tune_corrected == 1 ? "tune corrected" : "input lattice response"), NULL))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  for (i = 0; i < CM->nmon; i++) {
    if (!SDDS_SetRowValues(&respOutput->SDDSout, SDDS_PASS_BY_VALUE | SDDS_SET_BY_INDEX, i,
                           respOutput->monitorNameIndex, respOutput->monitorName[i],
                           respOutput->sIndex, CM->umoni[i]->end_pos, -1))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    if (unitsCode == BNL_UNITS)
      for (j = 0; j < CM->ncor; j++) {
        eptr = CM->ucorr[j];
        value = (inverse ? Mij(CM->T, j, i) : (corrPlane%2==0 ? Mij(CM->C, i, j) : Mij(CM->Cp, i, j)));
        if (eptr->type == T_HCOR || eptr->type == T_HVCOR || eptr->type == T_VCOR ||
            eptr->type == T_EHCOR || eptr->type == T_EHVCOR || eptr->type == T_EVCOR) {
          value *= (inverse ? eptr->Pref_output / 586.679 : 586.679 / (eptr->Pref_output + 1e-10));
        }
        if (!SDDS_SetRowValues(&respOutput->SDDSout, SDDS_PASS_BY_VALUE | SDDS_SET_BY_INDEX, i,
                               respOutput->correctorIndex[j], value, -1))
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
    else if (unitsCode == KNL_UNITS && corrPlane < 2)
      for (j = 0; j < CM->ncor; j++) {
        eptr = CM->ucorr[j];
        value = (inverse ? Mij(CM->T, j, i) : (corrPlane%2==0 ? Mij(CM->C, i, j) : Mij(CM->Cp, i, j)));
        if (eptr->type == T_HCOR || eptr->type == T_HVCOR ||
            eptr->type == T_EHCOR || eptr->type == T_EHVCOR)
          value = -value;
        if (!SDDS_SetRowValues(&respOutput->SDDSout, SDDS_PASS_BY_VALUE | SDDS_SET_BY_INDEX, i,
                               respOutput->correctorIndex[j], value, -1))
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
    else
      for (j = 0; j < CM->ncor; j++) {
        value = (inverse ? Mij(CM->T, j, i) : (corrPlane%2==0 ? Mij(CM->C, i, j) : Mij(CM->Cp, i, j)));
        if (!SDDS_SetRowValues(&respOutput->SDDSout, SDDS_PASS_BY_VALUE | SDDS_SET_BY_INDEX, i,
                               respOutput->correctorIndex[j], value, -1))
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
  }
  if (!SDDS_WriteTable(&respOutput->SDDSout))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
}

void finish_response_output(void) {
  log_entry("finish_response_output");
  if (response[0] && !SDDS_Terminate(&xRespOutput.SDDSout))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  if (response[1] && !SDDS_Terminate(&yRespOutput.SDDSout))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  if (inverse[0] && !SDDS_Terminate(&xInvRespOutput.SDDSout))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  if (inverse[1] && !SDDS_Terminate(&yInvRespOutput.SDDSout))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  log_exit("finish_response_output");
}
