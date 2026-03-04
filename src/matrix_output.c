/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: matrix_output()
 * purpose: handle user requests for matrix output
 *
 * Michael Borland, 1991
 */
#include "mdb.h"
#include "track.h"
#include "match_string.h"

static long output_now = 0;
static long n_outputs = 0;
static FILE **fp_printout; /* for printout */
static long *print_full_only = NULL;
static char **printoutFormat = NULL;
static long *SDDS_order = NULL;
static long *print_order = NULL;
static double *suppress_below_value = NULL;
static char *unit[6] = {"m", "rad", "m", "rad", "m", "1"};
static char **start_name = NULL;
static long *start_occurence = NULL;
static char **SDDS_match = NULL;

static SDDS_TABLE *SDDS_matrix = NULL;
static long *SDDS_matrix_initialized = NULL;
static long *SDDS_matrix_count = NULL;
static long *individualMatrices = NULL, *printElementData = NULL;

static long *mathematicaFullMatrix = NULL;
static FILE **fpMathematica = NULL;
static char **mathematicaMatrixName = NULL;

#define IC_S 0
#define IC_ELEMENT 1
#define IC_OCCURENCE 2
#define IC_TYPE 3
#define IC_SYMPLECTICITY1 4
#define IC_SYMPLECTICITY1_FULL 5
#define N_COLUMNS 6
static SDDS_DEFINITION column_definition[N_COLUMNS] = {
  {"s", "&column name=s, units=m, type=double, description=\"Distance\" &end"},
  {"ElementName", "&column name=ElementName, type=string, description=\"Element name\", format_string=%10s &end"},
  {"ElementOccurence",
   "&column name=ElementOccurence, type=long, description=\"Occurence of element\", format_string=%6ld &end"},
  {"ElementType", "&column name=ElementType, type=string, description=\"Element-type name\", format_string=%10s &end"},
  {"Symplecticity1", "&column name=Symplecticity1, type=double, description=\"Symplecticity check for this element's first-order matrix\" &end"},
  {"SymplecticityFull1", "&column name=SymplecticityFull1, type=double, description=\"Symplecticity check for the concatenated first-order matrix\" &end"},
};

#define IP_STEP 0
#define N_PARAMETERS 1
static SDDS_DEFINITION parameter_definition[N_PARAMETERS] = {
  {"Step", "&parameter name=Step, type=long, description=\"Simulation step\" &end"},
};

void SDDS_set_matrices(SDDS_TABLE *SDDS_table, VMATRIX *M, VMATRIX *M1, long order, long step,
                       ELEMENT_LIST *elem, long i_element, long n_elements, long individual_matrices);

void setup_matrix_output(
  NAMELIST_TEXT *nltext,
  RUN *run,
  VARY *control,
  LINE_LIST *beamline) {
#include "matrix_output.h"
  long i, j, k, l;
  char *denom[4], *numer[4];
  long n_denom, n_numer;
  char s[100], t[200];
  char buffer[SDDS_MAXLINE];

  log_entry("setup_matrix_output");

  /* process namelist input */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&matrix_output, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &matrix_output);

  /* check for validity of namelist inputs */
  if (printout == NULL && SDDS_output == NULL)
    bombElegant("no output filenames given in namelist matrix_output", NULL);
  if (printout && !(printout_order > 0 && printout_order < 4))
    bombElegant("printout_order is invalid", NULL);
  if (SDDS_output && !(SDDS_output_order > 0 && SDDS_output_order < 4))
    bombElegant("SDDS_output_order is invalid", NULL);

  printout = compose_filename(printout, run->rootname);
  SDDS_output = compose_filename(SDDS_output, run->rootname);
  fp_printout = trealloc(fp_printout, sizeof(*fp_printout) * (n_outputs + 1));
  print_full_only = trealloc(print_full_only, sizeof(*print_full_only) * (n_outputs + 1));
  printoutFormat = trealloc(printoutFormat, sizeof(*printoutFormat) * (n_outputs + 1));
  print_order = trealloc(print_order, sizeof(*print_order) * (n_outputs + 1));
  suppress_below_value = trealloc(suppress_below_value, sizeof(*suppress_below_value) * (n_outputs + 1));
  start_name = trealloc(start_name, sizeof(*start_name) * (n_outputs + 1));
  start_occurence = trealloc(start_occurence, sizeof(*start_occurence) * (n_outputs + 1));
  SDDS_match = trealloc(SDDS_match, sizeof(*SDDS_match) * (n_outputs + 1));
  SDDS_order = trealloc(SDDS_order, sizeof(*SDDS_order) * (n_outputs + 1));
  SDDS_matrix = trealloc(SDDS_matrix, sizeof(*SDDS_matrix) * (n_outputs + 1));
  SDDS_matrix_initialized = trealloc(SDDS_matrix_initialized, sizeof(*SDDS_matrix_initialized) * (n_outputs + 1));
  SDDS_matrix_count = trealloc(SDDS_matrix_count, sizeof(*SDDS_matrix_count) * (n_outputs + 1));
  if (individual_matrices && full_matrix_only)
    bombElegant("individual_matrices and full_matrix_only are incompatible", NULL);

  individualMatrices = trealloc(individualMatrices, sizeof(*individualMatrices) * (n_outputs + 1));
  printElementData = trealloc(printElementData, sizeof(*printElementData) * (n_outputs + 1));

  fpMathematica = trealloc(fpMathematica, sizeof(*fpMathematica) * (n_outputs + 1));
  mathematicaMatrixName = trealloc(mathematicaMatrixName, sizeof(*mathematicaMatrixName) * (n_outputs + 1));
  mathematicaFullMatrix = trealloc(mathematicaFullMatrix, sizeof(*mathematicaFullMatrix) * (n_outputs + 1));

  individualMatrices[n_outputs] = individual_matrices;
  printElementData[n_outputs] = print_element_data;

  if (start_from)
    cp_str(start_name + n_outputs, start_from);
  else
    start_name[n_outputs] = NULL;
  start_occurence[n_outputs] = start_from_occurence;
  print_order[n_outputs] = printout ? printout_order : 0;
  print_full_only[n_outputs] = full_matrix_only;
  suppress_below_value[n_outputs] = suppress_below;
  cp_str(&printoutFormat[n_outputs], printout_format);

  SDDS_order[n_outputs] = SDDS_output ? SDDS_output_order : 0;
  if (SDDS_output_match)
    cp_str(SDDS_match + n_outputs, SDDS_output_match);
  else
    SDDS_match[n_outputs] = NULL;
  SDDS_matrix_initialized[n_outputs] = SDDS_matrix_count[n_outputs] = 0;
  SDDS_ZeroMemory(SDDS_matrix + n_outputs, sizeof(*SDDS_matrix));

#if USE_MPI
  /* This has to be here, otherwise some values will not be set correctly */
  if (isSlave)
    printout = SDDS_output = NULL;
#endif
  if (printout) {
    fp_printout[n_outputs] = fopen_e(printout, "w", 0);
  } else
    fp_printout[n_outputs] = NULL;

  mathematicaFullMatrix[n_outputs] = mathematica_full_matrix;
  fpMathematica[n_outputs] = NULL;
  mathematicaMatrixName[n_outputs] = NULL;
  if ((mathematicaFullMatrix[n_outputs] = mathematica_full_matrix)) {
    cp_str(&mathematicaMatrixName[n_outputs], mathematica_matrix_name);
    if (mathematica_matrix_file && strlen(mathematica_matrix_file)) {
      mathematica_matrix_file = compose_filename(mathematica_matrix_file, run->rootname);
      fpMathematica[n_outputs] = fopen_e(mathematica_matrix_file, "w", 0);
      free(mathematica_matrix_file);
    }
  }

  if (SDDS_output) {
    SDDS_ElegantOutputSetup(SDDS_matrix + n_outputs, SDDS_output, SDDS_BINARY, 1, "matrix",
                            run->runfile, run->lattice, parameter_definition, N_PARAMETERS,
                            column_definition, N_COLUMNS, "setup_matrix_output",
                            SDDS_EOS_NEWFILE);

    for (i = 0; i < 6; i++) {
      sprintf(buffer, "&column name=C%ld, symbol=\"C$b%ld$n\", type=double ", i + 1, i + 1);
      if (SDDS_StringIsBlank(unit[i]))
        strcpy_ss(t, " &end");
      else
        sprintf(t, "units=%s &end", unit[i]);
      strcat(buffer, t);
      if (!SDDS_ProcessColumnString(SDDS_matrix + n_outputs, buffer, 0)) {
        SDDS_SetError("Problem defining SDDS matrix output Ci columns (setup_matrix_output)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
    }
    if (individual_matrices) {
      for (i = 0; i < 6; i++) {
        sprintf(buffer, "&column name=MaxError%ld, type=double, units=%s &end", i, unit[i]);
        if (!SDDS_ProcessColumnString(SDDS_matrix + n_outputs, buffer, 0)) {
          SDDS_SetError("Problem defining SDDS matrix output max. error columns (setup_matrix_output)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
      }
    }
    if (SDDS_output_order >= 1) {
      for (i = 0; i < 6; i++) {
        for (j = 0; j < 6; j++) {
          sprintf(buffer, "&column name=R%ld%ld, symbol=\"R$b%ld%ld$n\", type=double ", i + 1, j + 1, i + 1, j + 1);
          if (i == j)
            strcpy_ss(t, " &end");
          else
            sprintf(t, "units=%s/%s &end", unit[i], unit[j]);
          strcat(buffer, t);
          if (!SDDS_ProcessColumnString(SDDS_matrix + n_outputs, buffer, 0)) {
            SDDS_SetError("Problem defining SDDS matrix output Rij columns (setup_matrix_output)");
            SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
          }
        }
      }
    }
    if (SDDS_output_order >= 2) {
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
            if (!SDDS_ProcessColumnString(SDDS_matrix + n_outputs, buffer, 0)) {
              SDDS_SetError("Problem defining SDDS matrix output Tijk columns (setup_matrix_output)");
              SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
            }
          }
        }
      }
    }

    if (SDDS_output_order >= 3) {
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
              if (!SDDS_ProcessColumnString(SDDS_matrix + n_outputs, buffer, 0)) {
                SDDS_SetError("Problem defining SDDS matrix output Uijk columns (setup_matrix_output)");
                SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
              }
            }
          }
        }
      }
    }

    if (!SDDS_WriteLayout(SDDS_matrix + n_outputs)) {
      SDDS_SetError("Unable to write SDDS layout for matrix output");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    SDDS_matrix_initialized[n_outputs] = 1;
  } else
    SDDS_matrix_initialized[n_outputs] = 0;

  beamline->flags |= BEAMLINE_MATRICES_NEEDED;
  if (!output_at_each_step) {
    /* user wants output now */
    output_now = n_outputs;
    n_outputs++;
    run_matrix_output(run, control, beamline);
    n_outputs--;
    if (SDDS_matrix_initialized[n_outputs]) {
      if (!SDDS_Terminate(SDDS_matrix + n_outputs)) {
        SDDS_SetError("Problem terminating SDDS output (setup_matrix_output)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
      SDDS_matrix_initialized[n_outputs] = 0;
    }
    if (fp_printout[n_outputs])
      fclose(fp_printout[n_outputs]);
    fp_printout[n_outputs] = NULL;
    if (fpMathematica[n_outputs])
      fclose(fpMathematica[n_outputs]);
    fpMathematica[n_outputs] = NULL;
  } else {
    /* otherwise, the information is saved, so advance array counter */
    n_outputs++;
  }
  output_now = 0;

  log_exit("setup_matrix_output");
}

void run_matrix_output(
  RUN *run,
  VARY *control,
  LINE_LIST *beamline) {
  ELEMENT_LIST *member, *first_member;
  long i, n_elem_no_matrix, sfo;
  //long n_elements;
  long i_SDDS_output, n_SDDS_output = 0;
  long i_output, output_order;
  VMATRIX *M1, *M2, *tmp;
  char s[256];
  /* double z0; */
  VARY rcContext;
  double Ccopy[6];

  if (n_outputs == 0)
    return;

  log_entry("run_matrix_output");

  calculate_matrices(beamline, run);

  for (i_output = output_now; i_output < n_outputs; i_output++) {
    output_order = MAX(print_order[i_output], SDDS_order[i_output]);

    initialize_matrices(M1 = tmalloc(sizeof(*M1)), output_order);
    initialize_matrices(M2 = tmalloc(sizeof(*M2)), output_order);
    for (i = 0; i < 6; i++)
      M1->R[i][i] = 1;

    //n_elements = 0;
    n_elem_no_matrix = 0;
    first_member = member = beamline->elem;
    /* z0 = 0; */
    sfo = start_occurence[i_output];
    if (start_name[i_output] != NULL) {
      member = NULL;
      while (sfo--)
        if (!(first_member = find_element(start_name[i_output], &member, beamline->elem)))
          bombElegant("can't find specified occurence of given element for matrix output", NULL);
      printf("starting matrix output from element %s at z=%e m\n", first_member->name, first_member->end_pos);
      fflush(stdout);
      member = first_member;
      /*if (first_member->pred)
        z0 = first_member->pred->end_pos;*/
      getRunControlContext(&rcContext);
      if (rcContext.ready && (rcContext.fiducial_flag & FIRST_BEAM_IS_FIDUCIAL)) {
        VMATRIX *Mfull;
        /* Need to concantenate the matrix up to this point to find the centroids
	 * for propagation downstream 
	 */
        Mfull = accumulate_matrices(beamline->elem, run, NULL, 2, 0);
        for (i = 0; i < 6; i++)
          M1->C[i] = member->accumMatrix->C[i];
        free_matrices(Mfull);
        free(Mfull);
        Mfull = NULL;
      }
    }
    beamline->elem_recirc = NULL;
    beamline->i_recirc = 0;
    while (member) {
#if DEBUG
      printf("working on matrix for %s\n", member->name);
      fflush(stdout);
      print_elem(stdout, member);
#endif
      if (!member->matrix)
        compute_matrix(member, run, NULL);
      //n_elements++;
      member = member->succ;
    }
    if (SDDS_matrix_initialized[i_output]) {
      member = first_member;
      n_SDDS_output = 1; /* for the beginning element */
      while (member) {
        if (!SDDS_match[i_output] || wild_match(member->name, SDDS_match[i_output]))
          n_SDDS_output++;
        member = member->succ;
      }
    }
    if (fp_printout[i_output] && printElementData[i_output])
      print_line(fp_printout[i_output], beamline);
    if (SDDS_matrix_initialized[i_output] && !print_full_only[i_output]) {
      ELEMENT_LIST start_elem;
      start_elem.end_pos = sStart;
      start_elem.name = "_BEG_";
      start_elem.type = T_MARK;
      start_elem.occurence = 1;
      SDDS_set_matrices(SDDS_matrix + i_output, M1, NULL, SDDS_order[i_output],
                        control ? control->i_step : -1, &start_elem, 0, n_SDDS_output, individualMatrices[i_output]);
    }
    i_SDDS_output = 1;
    member = first_member;
    while (member) {
      if (entity_description[member->type].flags & HAS_MATRIX) {
        if (fp_printout[i_output] && !print_full_only[i_output]) {
#ifdef DEBUG
          printf("doing printout for matrix of %s %s\n",
                 entity_name[member->type], member->name);
          fflush(stdout);
#endif
          if (member->matrix->order > print_order[i_output]) {
            SWAP_LONG(member->matrix->order, print_order[i_output]);
            print_matrices1(fp_printout[i_output], member->name, printoutFormat[i_output], member->matrix, suppress_below_value[i_output]);
            SWAP_LONG(member->matrix->order, print_order[i_output]);
          } else
            print_matrices1(fp_printout[i_output], member->name, printoutFormat[i_output], member->matrix, suppress_below_value[i_output]);
        }
#ifdef DEBUG
        printf("concatenating matrix of %s %s\n",
               entity_name[member->type], member->name);
        fflush(stdout);
#endif
        if (individualMatrices[i_output])
          /* exclude C so that the effect of the trajectory is included */
          null_matrices(M1, EXCLUDE_C | SET_UNIT_R);
        concat_matrices(M2, member->matrix, M1,
                        entity_description[member->type].flags & HAS_RF_MATRIX ? CONCAT_EXCLUDE_S0 : 0);
        tmp = M2;
        M2 = M1;
        M1 = tmp;
        if (fp_printout[i_output] && !print_full_only[i_output]) {
          sprintf(s, "%s matrix after last element",
                  individualMatrices[i_output] ? "Effective element" : "Concatenated");
          if (M1->order > print_order[i_output]) {
            SWAP_LONG(M1->order, print_order[i_output]);
            print_matrices1(fp_printout[i_output], s, printoutFormat[i_output], M1, suppress_below_value[i_output]);
            SWAP_LONG(M1->order, print_order[i_output]);
          } else {
            print_matrices1(fp_printout[i_output], s, printoutFormat[i_output], M1, suppress_below_value[i_output]);
          }
        }
      } else {
        n_elem_no_matrix++;
        if (individualMatrices[i_output]) {
          copy_matrices1(M2, M1);
          null_matrices(M1, EXCLUDE_C | SET_UNIT_R);
        }
      }
      if (SDDS_matrix_initialized[i_output] &&
          (!SDDS_match[i_output] || wild_match(member->name, SDDS_match[i_output]))) {
        if (!print_full_only[i_output]) {
          if (individualMatrices[i_output]) {
            /* output change in C, rather than C itself */
            copy_doubles(Ccopy, M1->C, 6);
            for (i = 0; i < 6; i++)
              M1->C[i] -= M2->C[i];
          }
          SDDS_set_matrices(SDDS_matrix + i_output, M1, member->matrix, SDDS_order[i_output],
                            control ? control->i_step : -1, member,
                            i_SDDS_output++, n_SDDS_output, individualMatrices[i_output]);
          if (individualMatrices[i_output])
            copy_doubles(M1->C, Ccopy, 6);
        } else if (member->succ == NULL) {
          SDDS_set_matrices(SDDS_matrix + i_output, M1, member->matrix, SDDS_order[i_output],
                            control ? control->i_step : -1, member,
                            0, n_SDDS_output, individualMatrices[i_output]);
        }
      }
      member = member->succ;
    }
    if (SDDS_matrix_initialized[i_output]) {
      if (!SDDS_WriteTable(SDDS_matrix + i_output)) {
        SDDS_SetError("Unable to write matrix data (run_matrix_output)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
      if (!inhibitFileSync)
        SDDS_DoFSync(SDDS_matrix + i_output);
      if (!SDDS_EraseData(SDDS_matrix + i_output)) {
        SDDS_SetError("Unable to erase matrix data (run_matrix_output)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
      if (n_elem_no_matrix) {
        char warningText[1024];
        snprintf(warningText, 1024, "%ld elements had no matrix", n_elem_no_matrix);
        printWarning("Some elements had no matrix.", warningText);
      }
    }
    if (fp_printout[i_output]) {
      if (n_elem_no_matrix)
        snprintf(s, 256, "full matrix---WARNING: %ld ELEMENTS HAD NO MATRIX",
                 n_elem_no_matrix);
      else
        strcpy(s, "full matrix");
      SWAP_LONG(M1->order, print_order[i_output]);
      print_matrices1(fp_printout[i_output], s, printoutFormat[i_output], M1, suppress_below_value[i_output]);
      SWAP_LONG(M1->order, print_order[i_output]);

      if (mathematicaFullMatrix[i_output]) {
        long i, j;
        char sbuffer[1024];
        FILE *fptmp;
        if (fpMathematica[i_output])
          fptmp = fpMathematica[i_output];
        else
          fptmp = fp_printout[i_output];
        fprintf(fptmp, "%s={\n", mathematicaMatrixName[i_output]);
        for (i = 0; i < 6; i++) {
          fprintf(fptmp, "{");
          for (j = 0; j < 6; j++) {
            snprintf(sbuffer, 1024, "%21.15e", M1->R[i][j]);
            edit_string(sbuffer, "%/e/*10^(/ei/)");
            fprintf(fptmp, "%s%s",
                    sbuffer, j == 5 ? (i == 5 ? "}\n" : "},\n") : ",");
          }
        }
        fprintf(fptmp, "}\n");
      }
    }
  }
  log_exit("run_matrix_output");
}

void simplify_units(char *buffer, char **numer, long n_numer, char **denom, long n_denom) {
  long in, id, i, j;
  long *mult_numer, *mult_denom;
  char t[256];

  log_entry("simplify_units");

#ifdef DEBUG
  printf("*** simplify_units called\nnumerators: ");
  fflush(stdout);
  for (in = 0; in < n_numer; in++)
    printf("%s ", numer[in]);
  fflush(stdout);
  printf("\ndenominators: ");
  fflush(stdout);
  for (id = 0; id < n_denom; id++)
    printf("%s ", denom[id]);
  fflush(stdout);
  fputc('\n', stdout);
#endif

  for (in = 0; n_numer > 0 && in < n_numer; in++) {
    for (id = 0; n_denom > 0 && id < n_denom && n_numer > 0 && in < n_numer; id++) {
      if (strcmp(numer[in], denom[id]) == 0) {
        for (i = in; i < n_numer - 1; i++)
          numer[i] = numer[i + 1];
        for (i = id; i < n_denom - 1; i++)
          denom[i] = denom[i + 1];
        n_numer--;
        in--;
        n_denom--;
        id--;
      }
    }
  }
  for (id = 0; n_denom > 0 && id < n_denom; id++) {
    if (strcmp(denom[id], "1") == 0) {
      for (i = id; i < n_denom - 1; i++)
        denom[i] = denom[i + 1];
      n_denom--;
      id--;
    }
  }

#ifdef DEBUG
  printf("common factors divided out\nnumerators: ");
  fflush(stdout);
  for (in = 0; in < n_numer; in++)
    printf("%s ", numer[in]);
  fflush(stdout);
  printf("\ndenominators: ");
  fflush(stdout);
  for (id = 0; id < n_denom; id++)
    printf("%s ", denom[id]);
  fflush(stdout);
  fputc('\n', stdout);
#endif

  mult_numer = tmalloc(sizeof(*mult_numer) * (n_numer ? n_numer : 1));
  for (in = 0; in < n_numer; in++) {
    mult_numer[in] = 1;
    for (i = in + 1; i < n_numer; i++) {
      if (strcmp(numer[in], numer[i]) == 0) {
        mult_numer[in]++;
        for (j = i; j < n_numer - 1; j++)
          numer[j] = numer[j + 1];
        n_numer--;
        i--;
      }
    }
  }

  mult_denom = tmalloc(sizeof(*mult_denom) * (n_denom ? n_denom : 1));
  for (id = 0; id < n_denom; id++) {
    mult_denom[id] = 1;
    for (i = id + 1; i < n_denom; i++) {
      if (strcmp(denom[id], denom[i]) == 0) {
        mult_denom[id]++;
        for (j = i; j < n_denom - 1; j++)
          denom[j] = denom[j + 1];
        n_denom--;
        i--;
      }
    }
  }

#ifdef DEBUG
  printf("multiplicities established\nnumerators: ");
  fflush(stdout);
  for (in = 0; in < n_numer; in++)
    printf("%s^%ld ", numer[in], mult_numer[in]);
  fflush(stdout);
  printf("\ndenominators: ");
  fflush(stdout);
  for (id = 0; id < n_denom; id++)
    printf("%s^%ld ", denom[id], mult_denom[id]);
  fflush(stdout);
  fputc('\n', stdout);
#endif

  if (n_numer == 0 && n_denom == 0)
    strcpy_ss(buffer, "");
  else {
    buffer[0] = 0;
    if (n_numer < 1)
      strcpy_ss(buffer, "1");
    else {
      for (in = 0; in < n_numer; in++) {
        if (mult_numer[in] == 1)
          sprintf(t, "%s%s", numer[in], (in == n_numer - 1 ? "" : "-"));
        else
          sprintf(t, "%s$a%ld$n%s", numer[in], mult_numer[in], (in == n_numer - 1 ? "" : "-"));
        strcat(buffer, t);
      }
    }
    if (n_denom == 1)
      strcat(buffer, "/");
    else if (n_denom > 1)
      strcat(buffer, "/(");
    if (n_denom >= 1) {
      for (id = 0; id < n_denom; id++) {
        if (mult_denom[id] == 1)
          sprintf(t, "%s%s", denom[id], (id == n_denom - 1 ? "" : "-"));
        else
          sprintf(t, "%s$a%ld$n%s", denom[id], mult_denom[id], (id == n_denom - 1 ? "" : "-"));
        strcat(buffer, t);
      }
      if (n_denom > 1)
        strcat(buffer, ")");
    }
  }
  tfree(mult_numer);
  mult_numer = NULL;
  tfree(mult_denom);
  mult_denom = NULL;
  log_exit("simplify_units");
}

void finish_matrix_output() {
  long i_output;
  for (i_output = 0; i_output < n_outputs; i_output++) {
    if (fp_printout[i_output])
      fclose(fp_printout[i_output]);
    if (fpMathematica[i_output])
      fclose(fpMathematica[i_output]);
    if (SDDS_matrix_initialized[i_output] && !SDDS_Terminate(SDDS_matrix + i_output)) {
      SDDS_SetError("Problem terminating SDDS matrix output (finish_matrix_output)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    SDDS_matrix_initialized[i_output] = 0;
    fp_printout[i_output] = NULL;
  }
  n_outputs = 0;
}

void SDDS_set_matrices(SDDS_TABLE *SDDS_table, VMATRIX *M, VMATRIX *M0, long order, long step,
                       ELEMENT_LIST *elem, long i_element, long n_elements, long individual_matrices) {
  register long i, j, k, l, index;

  log_entry("SDDS_set_matrices");

  if (!M || !(M->C) || !(M->R) || (order > 1 && !(M->T)) || (order > 2 && !(M->Q)))
    bombElegant("bad matrix passed to SDDS_set_matrices()", NULL);

  if (i_element == 0) {
    if (!SDDS_StartTable(SDDS_table, n_elements)) {
      SDDS_SetError("Problem starting SDDS table for matrix output (SDDS_set_matrices)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    if (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, "Step", step, NULL)) {
      SDDS_SetError("Problem setting parameters for matrix_output (SDDS_set_matrices)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  }

  if (!SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, i_element,
                         IC_S, elem->end_pos, IC_ELEMENT, elem->name, IC_OCCURENCE, elem->occurence,
                         IC_TYPE, entity_name[elem->type],
                         IC_SYMPLECTICITY1, M0 ? checkSymplecticity(M0, 0) : (double)0.0,
                         IC_SYMPLECTICITY1_FULL, checkSymplecticity(M, 0),
                         -1)) {
    SDDS_SetError("Problem setting row values for SDDS matrix output (SDDS_set_matrices, 1)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  index = N_COLUMNS;
  for (i = 0; i < 6; i++)
    if (!SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, i_element,
                           index++, M->C[i], -1)) {
      SDDS_SetError("Problem setting row values for SDDS matrix output (SDDS_set_matrices, 2)");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }

  if (individual_matrices) {
    for (i = 0; i < 6; i++)
      if (!SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, i_element,
                             index++, M0 && M0->maxError?M0->maxError[i]:-1.0, -1)) {
        SDDS_SetError("Problem setting row values for SDDS matrix output (SDDS_set_matrices, 2)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
  }

  for (i = 0; i < 6; i++)
    for (j = 0; j < 6; j++)
      if (!SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, i_element,
                             index++, M->R[i][j], -1)) {
        SDDS_SetError("Problem setting row values for SDDS matrix output (SDDS_set_matrices, 3)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }

  if (order > 1)
    for (i = 0; i < 6; i++)
      for (j = 0; j < 6; j++)
        for (k = 0; k <= j; k++)
          if (!SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, i_element,
                                 index++, M->T[i][j][k], -1)) {
            SDDS_SetError("Problem setting row values for SDDS matrix output (SDDS_set_matrices, 4)");
            SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
          }

  if (order > 2)
    for (i = 0; i < 6; i++)
      for (j = 0; j < 6; j++)
        for (k = 0; k <= j; k++)
          for (l = 0; l <= k; l++)
            if (!SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, i_element,
                                   index++, M->Q[i][j][k][l], -1)) {
              SDDS_SetError("Problem setting row values for SDDS matrix output (SDDS_set_matrices, 5)");
              SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
            }

  log_exit("SDDS_set_matrices");
}
