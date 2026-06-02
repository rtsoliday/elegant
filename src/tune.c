/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: tune.c
 * purpose: tune correction by adjustment of quadrupoles
 *
 * Michael Borland, 1992
 */
#include "mdb.h"
#include "track.h"
#include "tuneDefs.h"
#include "correctorStash.h"

static FILE *fp_sl = NULL;
static long alter_defined_values;
static FILE *fp_response = NULL;
static FILE *fp_correction = NULL;
static short fseUnits = 0;

static CORRECTOR_STASH tuneStash = { NULL, NULL, NULL, NULL, 0, 0, 1, 0, "tune" };
static TUNE_CORRECTION *tune_stash_corr = NULL;

double *scanNumberList(char *list, long *nFound);

void setup_tune_correction(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline, TUNE_CORRECTION *tune) {
  /* VMATRIX *M; */
  ELEMENT_LIST *eptr;
  /* ELEMENT_LIST *elast; */
  double beta_x, alpha_x, eta_x, etap_x;
  double beta_y, alpha_y, eta_y, etap_y;
  long n_elem, last_n_elem;
  unsigned long unstable;
  long nFSE = 0;
  
#include "tune.h"

  log_entry("setup_tune_correction");

  if (fp_sl) {
    fclose(fp_sl);
    fp_sl = NULL;
  }
  if (fp_response) {
    fclose(fp_response);
    fp_response = NULL;
  }
  if (fp_correction) {
    fclose(fp_correction);
    fp_correction = NULL;
  }

  /* process namelist input */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&correct_tunes, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  str_toupper(quadrupoles);
  if (has_wildcards(quadrupoles) && strchr(quadrupoles, '-'))
    quadrupoles = expand_ranges(quadrupoles);
  if (echoNamelists)
    print_namelist(stdout, &correct_tunes);

  if (tune->name)
    tfree(tune->name);
  tune->name = tmalloc(sizeof(*tune->name) * (tune->n_families = 1));
  while ((tune->name[tune->n_families - 1] = get_token(quadrupoles)))
    tune->name = trealloc(tune->name, sizeof(*tune->name) * (tune->n_families += 1));
  if ((--tune->n_families) < 2)
    bombElegant("too few quadrupoles given for tune correction", NULL);

  if (tune->item)
    tfree(tune->item);
  if (tune->itemIsFSE)
    tfree(tune->itemIsFSE);
  tune->item = tmalloc(sizeof(*tune->item) * tune->n_families);
  tune->itemIsFSE = tmalloc(sizeof(*tune->itemIsFSE) * tune->n_families);
  if (items && strlen(items)) {
    long ii;
#define N_KNOWN_ITEMS 3
    char *knownItem[N_KNOWN_ITEMS] = {"K1", "FSE", "FSE_QUADRUPOLE"};
    for (ii=0; ii<tune->n_families; ii++) {
      if (!(tune->item[ii] = get_token(items)) || !strlen(tune->item[ii]))
        bombElegant("too few items given for tune correction", NULL);
      if ((tune->itemIsFSE[ii]=match_string(tune->item[ii], knownItem, N_KNOWN_ITEMS, EXACT_MATCH))<0) 
        bombElegant("item is not recognized", NULL);
      nFSE += tune->itemIsFSE[ii] ? 1 : 0;
    }
    if (nFSE!=0 && nFSE!=tune->n_families)
      fprintf(stderr, "Warning: some tune controls are physical units (K1) and others are FSE\n");
  } else {
    long ii;
    tune->item[0] = "K1";
    for (ii=1; ii<tune->n_families; ii++)
      tune->item[ii] = tune->item[0];
  }
  
  tune->lowerLimit = tune->upperLimit = NULL;
  if (lower_limits) {
    long nll = 0;
    tune->lowerLimit = scanNumberList(lower_limits, &nll);
    if (nll != tune->n_families)
      bombElegantVA("number of items in lower_limits list (%ld) not the same as number of quadrupole names (%ld)",
                    nll, tune->n_families);
  }

  if (upper_limits) {
    long nul = 0;
    tune->upperLimit = scanNumberList(upper_limits, &nul);
    if (nul != tune->n_families)
      bombElegantVA("number of items in upper_limits list (%ld) not the same as number of quadrupole names (%ld)",
                    nul, tune->n_families);
  }

  tune->length = calloc(tune->n_families, sizeof(*tune->length));
  tune->K1 = calloc(tune->n_families, sizeof(*tune->K1));

  tune->tunex = tune_x;
  tune->tuney = tune_y;
  tune->gain = correction_fraction;
  tune->n_iterations = n_iterations;
  tune->use_perturbed_matrix = use_perturbed_matrix;
  tune->tolerance = tolerance;
  tune->maximum_gain = max_correction_fraction;
  tune->step_up_interval = step_up_interval;
  tune->delta_gain = delta_correction_fraction;
  if ((tune->dK1_weight = dK1_weight) < 0)
    tune->dK1_weight = 0;
  tune->verbosity = verbosity;
  tune->update_orbit = update_orbit;
  alter_defined_values = change_defined_values;
  tune_stash_corr = tune;
  tuneStash.reassert = reset_correctors_each_step ? 1 : 0;

  tune->exclude = NULL;
  tune->n_exclude = 0;
  if (exclude) {
    char *excludec;
    cp_str(&excludec, exclude);
    tune->exclude = tmalloc(sizeof(*tune->exclude) * (tune->n_exclude = 1));
    while ((tune->exclude[tune->n_exclude - 1] = get_token(excludec)))
      tune->exclude = trealloc(tune->exclude, sizeof(*tune->exclude) * (tune->n_exclude += 1));
    tune->n_exclude--;
    free(excludec);
  }

  if (!beamline->twiss0 || !beamline->matrix) {
    if (!beamline->twiss0)
      beamline->twiss0 = tmalloc(sizeof(*beamline->twiss0));

    eptr = beamline->elem_twiss = beamline->elem;
    n_elem = last_n_elem = beamline->n_elems;
    /* elast = eptr; */
    while (eptr) {
      if (eptr->type == T_RECIRC) {
        last_n_elem = n_elem;
        beamline->elem_twiss = beamline->elem_recirc = eptr;
      }
      /* elast = eptr; */
      eptr = eptr->succ;
      n_elem--;
    }
    n_elem = last_n_elem;
  }

  if (!tune->use_perturbed_matrix || tune->tunex < 0 || tune->tuney < 0) {
    printf("Computing periodic Twiss parameters.\n");
    fflush(stdout);
    /* M =  */
    beamline->matrix = compute_periodic_twiss(&beta_x, &alpha_x, &eta_x, &etap_x, beamline->tune,
                                              &beta_y, &alpha_y, &eta_y, &etap_y, beamline->tune + 1,
                                              beamline->elem_twiss, NULL, run, &unstable, NULL, NULL);
    beamline->twiss0->betax = beta_x;
    beamline->twiss0->alphax = alpha_x;
    beamline->twiss0->phix = 0;
    beamline->twiss0->etax = eta_x;
    beamline->twiss0->etapx = etap_x;
    beamline->twiss0->betay = beta_y;
    beamline->twiss0->alphay = alpha_y;
    beamline->twiss0->phiy = 0;
    beamline->twiss0->etay = eta_y;
    beamline->twiss0->etapy = etap_y;

    propagate_twiss_parameters(beamline->twiss0, beamline->tune, beamline->waists,
                               NULL, beamline->elem_twiss, run, NULL, NULL,
                               beamline->couplingFactor);
    if (tune->tunex < 0) {
      printf("horizontal tune will be held at %f\n", tune->tunex = beamline->tune[0]);
      fflush(stdout);
    }
    if (tune->tuney < 0) {
      printf("vertical tune will be held at %f\n", tune->tuney = beamline->tune[1]);
      fflush(stdout);
    }
  }

#if USE_MPI
  if (!writePermitted)
    strength_log = response_matrix_output = correction_matrix_output = NULL;
#endif
  if (strength_log) {
    strength_log = compose_filename(strength_log, run->rootname);
    fp_sl = fopen_e(strength_log, "w", 0);
    fprintf(fp_sl, "SDDS1\n&parameter name=Step, type=long, description=\"Simulation step\" &end\n");
    fprintf(fp_sl, "&column name=ElementName, type=string  &end\n");
    fprintf(fp_sl, "&column name=ElementParameter, type=string  &end\n");
    fprintf(fp_sl, "&column name=ElementOccurence, type=long  &end\n");
    fprintf(fp_sl, "&column name=ParameterValue, type=double, units=\"1/m$a2$n\" &end\n");
    fprintf(fp_sl, "&data mode=ascii, no_row_counts=1 &end\n");
    fflush(fp_sl);
  }
  if (response_matrix_output) {
    response_matrix_output = compose_filename(response_matrix_output, run->rootname);
    fp_response = fopen_e(response_matrix_output, "w", 0);
    fprintf(fp_response, "SDDS1\n&parameter name=Step, type=long, description=\"Simulation step\" &end\n");
    fprintf(fp_response, "&parameter name=Excluded, type=string, fixed_value=\"%s\" &end\n",
            exclude && strlen(exclude) ? exclude : "");
    fprintf(fp_response, "&column name=FamilyName, type=string  &end\n");
    fseUnits = fse_units;
    if (!fse_units) {
      fprintf(fp_response, "&column name=dnux/dK1L, type=double, units=m,  &end\n");
      fprintf(fp_response, "&column name=dnuy/dK1L, type=double, units=m,  &end\n");
    } else {
      fprintf(fp_response, "&column name=dnux/FSE, type=double, units=m,  &end\n");
      fprintf(fp_response, "&column name=dnuy/FSE, type=double, units=m,  &end\n");
    }
    fprintf(fp_response, "&data mode=ascii, no_row_counts=1 &end\n");
    fflush(fp_response);
  }
  if (correction_matrix_output) {
    correction_matrix_output = compose_filename(correction_matrix_output, run->rootname);
    fp_correction = fopen_e(correction_matrix_output, "w", 0);
    fprintf(fp_correction, "SDDS1\n&parameter name=Step, type=long, description=\"Simulation step\" &end\n");
    fprintf(fp_correction, "&parameter name=Excluded, type=string, fixed_value=\"%s\" &end\n",
            exclude && strlen(exclude) ? exclude : "");
    fprintf(fp_correction, "&column name=FamilyName, type=string  &end\n");
    if (!fseUnits) {
      fprintf(fp_correction, "&column name=dK1L/dnux, type=double, units=1/m,  &end\n");
      fprintf(fp_correction, "&column name=dK1L/dnuy, type=double, units=1/m,  &end\n");
    } else {
      fprintf(fp_correction, "&column name=FSE/dnux, type=double, &end\n");
      fprintf(fp_correction, "&column name=FSE/dnuy, type=double, &end\n");
    }
    fprintf(fp_correction, "&data mode=ascii, no_row_counts=1 &end\n");
    fflush(fp_correction);
  }

  if (!tune->use_perturbed_matrix)
    computeTuneCorrectionMatrix(run, NULL, beamline, tune, 1);
}

void computeTuneCorrectionMatrix(RUN *run, VARY *control, LINE_LIST *beamline, TUNE_CORRECTION *tune, long printout) {
  MATRIX *C, *Ct, *CtC, *inv_CtC;
  VMATRIX *M;
  long i, count;
  double betax_L_sum, betay_L_sum;
  ELEMENT_LIST *context;
  long step;

  if (!(M = beamline->matrix) || !M->C || !M->R)
    bombElegant("something wrong with transfer map for beamline (setup_tune_correction)", NULL);

  step = control ? control->i_step : 0;

  /* Solve dnu = C*dK1 for dK1 */
  m_alloc(&C, 2 + tune->n_families, tune->n_families);
  m_zero(C);
  m_alloc(&Ct, tune->n_families, 2 + tune->n_families);
  m_alloc(&CtC, tune->n_families, tune->n_families);
  m_alloc(&inv_CtC, tune->n_families, tune->n_families);
  m_alloc(&(tune->T), tune->n_families, 2 + tune->n_families);
  m_alloc(&(tune->dK1), tune->n_families, 1);
  m_alloc(&(tune->dtune), 2 + tune->n_families, 1);

  printf("Computing tune influence matrix for all named quadrupoles.\n");
  fflush(stdout);

  for (i = 0; i < tune->n_families; i++) {
    count = 0;
    context = NULL;
    betax_L_sum = betay_L_sum = 0;
    while ((context = wfind_element(tune->name[i], &context, beamline->elem))) {
      if (tune->n_exclude) {
        long j, excluded;
        for (j = excluded = 0; j < tune->n_exclude; j++)
          if (wild_match(context->name, tune->exclude[j])) {
            excluded = 1;
            break;
          }
        if (excluded)
          continue;
      }
      if (count == 0) {
        long K1_param;
        double *K1ptr;
        if (!(K1_param = confirm_parameter("K1", context->type))) {
          printf("error: element %s does not have K1 parameter\n", context->name);
          fflush(stdout);
          exitElegant(1);
        }
        if (entity_description[context->type].flags & HAS_LENGTH)
          tune->length[i] = ((QUAD *)context->p_elem)->length;
        else
          tune->length[i] = 1;
        if (!(K1ptr = (double*)(context->p_elem + entity_description[context->type].parameter[K1_param].offset)))
          bombElegant("K1ptr NULL in computeTuneCorrectionMatrix", NULL);
        tune->K1[i] = *K1ptr;
        if (tune->itemIsFSE[i]) {
          if (!confirm_parameter(tune->item[i], context->type)) {
            printf("error: element %s does not have %s parameter\n", context->name, tune->item[i]);
            fflush(stdout);
            exitElegant(1);
          }
        }
      }
      if (context->pred && context->pred->twiss) {
        betax_L_sum += (context->pred->twiss->betax + context->twiss->betax) / 2 * tune->length[i];
        betay_L_sum += (context->pred->twiss->betay + context->twiss->betay) / 2 * tune->length[i];
      } else {
        betax_L_sum += context->twiss->betax * tune->length[i];
        betay_L_sum += context->twiss->betay * tune->length[i];
      }
      count++;
    }
    if (count == 0) {
      printf("error: element %s is not part of the beamline\n", tune->name[i]);
      fflush(stdout);
      exitElegant(1);
    }
    if (printout) {
      printf("%ld instances of %s found\n", count, tune->name[i]);
      fflush(stdout);
    }
    C->a[0][i] = betax_L_sum / (4 * PI);
    C->a[1][i] = -betay_L_sum / (4 * PI);
    if (tune->itemIsFSE[i]) {
      C->a[0][i] *= tune->K1[i];
      C->a[1][i] *= tune->K1[i];
    }
    if (C->a[0][i] == 0 || C->a[1][i] == 0) {
      printf("error: parameter %s of element %s does not change the tune!\n", tune->item[i], tune->name[i]);
      fflush(stdout);
      exitElegant(1);
    }
  }

  if (printout) {
    printf("\nfamily               dNUx/dK1                  dNUy/dK1\n");
    fflush(stdout);
    for (i = 0; i < tune->n_families; i++) {
      if (!tune->itemIsFSE[i])
        printf("%10s:    %22.15e     %22.15e\n", tune->name[i], C->a[0][i], C->a[1][i]);
      else
        printf("%10s:    %22.15e     %22.15e\n", tune->name[i], C->a[0][i]/tune->K1[i], C->a[1][i]/tune->K1[i]);
    }
    fflush(stdout);
  }
  if (fp_response) {
    fprintf(fp_response, "%ld\n", step);
    for (i = 0; i < tune->n_families; i++) {
      if (!fseUnits) {
        if (!tune->itemIsFSE[i])
          fprintf(fp_response, "%s %22.15e %22.15e\n",
                  tune->name[i], C->a[0][i] / tune->length[i], C->a[1][i] / tune->length[i]);
        else 
          fprintf(fp_response, "%s %22.15e %22.15e\n",
                  tune->name[i], C->a[0][i] / tune->K1[i] / tune->length[i], C->a[1][i] / tune->K1[i] / tune->length[i]);
      } else {
        if (!tune->itemIsFSE[i])
          fprintf(fp_response, "%s %22.15e %22.15e\n",
                  tune->name[i], C->a[0][i] * tune->K1[i], C->a[1][i] * tune->K1[i]);
        else 
          fprintf(fp_response, "%s %22.15e %22.15e\n",
                  tune->name[i], C->a[0][i], C->a[1][i]);
      }
    }
    fprintf(fp_response, "\n");
    fflush(fp_response);
  }

  for (i = 0; i < tune->n_families; i++)
    C->a[i + 2][i] = tune->n_families > 2 ? tune->dK1_weight : 0;

  m_trans(Ct, C);
  m_mult(CtC, Ct, C);
  m_invert(inv_CtC, CtC);
  m_mult(tune->T, inv_CtC, Ct);

  if (printout) {
    printf("\nfamily               dK1/dNUx                  dK1/dNUy\n");
    for (i = 0; i < tune->n_families; i++) {
      if (!tune->itemIsFSE[i])
        printf("%10s:    %22.15e     %22.15e\n", tune->name[i], tune->T->a[i][0], tune->T->a[i][1]);
      else
        printf("%10s:    %22.15e     %22.15e\n", tune->name[i], tune->T->a[i][0]*tune->K1[i], tune->T->a[i][1]*tune->K1[i]);
    }
    printf("\n");
    fflush(stdout);
  }
  if (fp_correction) {
    fprintf(fp_correction, "%ld\n", step);
    for (i = 0; i < tune->n_families; i++) {
      if (!fseUnits) {
        if (!tune->itemIsFSE[i])
          fprintf(fp_correction, "%s %22.15e %22.15e\n",
                  tune->name[i], tune->T->a[i][0] * tune->length[i], tune->T->a[i][1] * tune->length[i]);
        else 
          fprintf(fp_correction, "%s %22.15e %22.15e\n",
                  tune->name[i], tune->T->a[i][0] * tune->length[i] * tune->K1[i], tune->T->a[i][1] * tune->length[i] * tune->K1[i]);
      } else {
        if (!tune->itemIsFSE[i])
          fprintf(fp_correction, "%s %22.15e %22.15e\n",
                  tune->name[i], tune->T->a[i][0] / tune->K1[i], tune->T->a[i][1] / tune->K1[i]);
        else 
          fprintf(fp_correction, "%s %22.15e %22.15e\n",
                  tune->name[i], tune->T->a[i][0], tune->T->a[i][1]);
      }
    }
    fprintf(fp_correction, "\n");
    fflush(fp_correction);
  }

  m_free(&C);
  m_free(&Ct);
  m_free(&CtC);
  m_free(&inv_CtC);

  log_exit("setup_tune_correction");
}

long do_tune_correction(TUNE_CORRECTION *tune, RUN *run, VARY *control, LINE_LIST *beamline,
                        double *clorb, long do_closed_orbit, long step, long last_iteration) {
  VMATRIX *M;
  double K1 = 0.0, FSE = 0.0, gain, LastMsError, MsError;
  long steps_since_gain_change;
  ELEMENT_LIST *context;
  long i, K1_param, FSE_param, type = 0, iter;
  double beta_x, alpha_x, eta_x, etap_x;
  double beta_y, alpha_y, eta_y, etap_y, dtunex, dtuney;
  /* double factor; */
  static long tunes_saved = 0;
  unsigned long unstable;
  double *K1ptr, *FSEptr;
  char warningBuffer[1024];

#ifdef DEBUG
  printf("do_tune_correction\n");
  fflush(stdout);
#endif

  M = beamline->matrix = compute_periodic_twiss(&beta_x, &alpha_x, &eta_x, &etap_x, beamline->tune,
                                                &beta_y, &alpha_y, &eta_y, &etap_y, beamline->tune + 1,
                                                beamline->elem_twiss, clorb, run, &unstable, NULL, NULL);
#ifdef DEBUG
  printf("   updated periodic twiss solution\n");
  fflush(stdout);
#endif

  beamline->twiss0->betax = beta_x;
  beamline->twiss0->alphax = alpha_x;
  beamline->twiss0->phix = 0;
  beamline->twiss0->etax = eta_x;
  beamline->twiss0->etapx = etap_x;
  beamline->twiss0->betay = beta_y;
  beamline->twiss0->alphay = alpha_y;
  beamline->twiss0->phiy = 0;
  beamline->twiss0->etay = eta_y;
  beamline->twiss0->etapy = etap_y;

  propagate_twiss_parameters(beamline->twiss0, beamline->tune, beamline->waists,
                             NULL, beamline->elem_twiss, run, clorb, NULL,
                             beamline->couplingFactor);

#ifdef DEBUG
  printf("   updated full twiss solution\n");
  fflush(stdout);
#endif

  if (!M || !M->C || !M->R)
    bombElegant("something wrong with transfer map for beamline (do_tune_correction.1)", NULL);

  if (tune->verbosity > 0) {
    printf("\nAdjusting tunes:\n");
    printf("initial tunes:  %e  %e\n", beamline->tune[0], beamline->tune[1]);
    fflush(stdout);
  }

  MsError = sqr(beamline->tune[0] - tune->tunex) + sqr(beamline->tune[1] - tune->tuney);

  if (!tunes_saved) {
    /* nux_orig = beamline->tune[0]; */
    /* nuy_orig = beamline->tune[1]; */
    tunes_saved = 1;
  }

  steps_since_gain_change = 0;
  gain = tune->gain;

  for (iter = 0; iter < tune->n_iterations; iter++) {
    long nTotal, nLimit, nChanged;
#ifdef DEBUG
    printf("   Starting iteration %ld\n", iter);
    fflush(stdout);
#endif
    dtunex = tune->tunex - beamline->tune[0];
    dtuney = tune->tuney - beamline->tune[1];
    if (tune->tolerance > 0 &&
        tune->tolerance > fabs(dtunex) &&
        tune->tolerance > fabs(dtuney)) {
      printf("Tunes are acceptable---stopping tune correction.\n");
      break;
    }

    if (tune->use_perturbed_matrix)
      computeTuneCorrectionMatrix(run, control, beamline, tune, 0);
    m_zero(tune->dtune);
    tune->dtune->a[0][0] = dtunex;
    tune->dtune->a[1][0] = dtuney;

    LastMsError = MsError;
    if (iter) {
      MsError = sqr(beamline->tune[0] - tune->tunex) + sqr(beamline->tune[1] - tune->tuney);
      if (MsError > LastMsError) {
        /* reset to minimum gain */
        printWarning("Tune correction diverging.", "Gain reduced 10-fold.");
        gain /= 10;
        steps_since_gain_change = 0;
      }
    }

    m_mult(tune->dK1, tune->T, tune->dtune);
    m_scmul(tune->dK1, tune->dK1, gain);
    for (i = 0; i < tune->n_families; i++) {
      if (isnan(tune->dK1->a[i][0]) || isinf(tune->dK1->a[i][0]))
        break;
    }
    if (i != tune->n_families) {
      printf("Unable to correct tune---diverged.\n");
      fflush(stdout);
      return 0;
    }

    nTotal = nLimit = nChanged = 0;
    for (i = 0; i < tune->n_families; i++) {
      short has_wc;
      double K10, FSE0;
      context = NULL;
      has_wc = has_wildcards(tune->name[i]);
      K1_param = -1;
      FSE_param = -1;
      while ((context = wfind_element(tune->name[i], &context, beamline->elem))) {
        if (tune->n_exclude) {
          long j, excluded;
          for (j = excluded = 0; j < tune->n_exclude; j++)
            if (wild_match(context->name, tune->exclude[j])) {
              excluded = 1;
              break;
            }
          if (excluded)
            continue;
        }
        if (!(K1_param = confirm_parameter("K1", context->type))) {
          printf("error: element %s does not have K1 parameter\n",
                 context->name);
          fflush(stdout);
          exitElegant(1);
        }
        FSE = FSE0 = 0;
        if (!(K1ptr = (double *)(context->p_elem + entity_description[context->type].parameter[K1_param].offset)))
          bombElegant("K1ptr NULL in do_tune_correction", NULL);
        K1 = K10 = *K1ptr;
        if (tune->itemIsFSE[i]) {
          if (!(FSE_param = confirm_parameter(tune->item[i], context->type))) {
            printf("error: element %s does not have %s parameter\n",
                   context->name, tune->item[i]);
            fflush(stdout);
            exitElegant(1);
          }
          if (!(FSEptr = (double *)(context->p_elem + entity_description[context->type].parameter[FSE_param].offset)))
            bombElegant("K1ptr NULL in do_tune_correction", NULL);
          FSE0 = *FSEptr;
          FSE = (*FSEptr += tune->dK1->a[i][0]);
          if (tune->lowerLimit && FSE < tune->lowerLimit[i]) {
            FSE = *FSEptr = tune->lowerLimit[i];
            nLimit++;
            snprintf(warningBuffer, 1024,
                     "%s#%ld is at the lower limit (FSE=%le, limit=%le).",
                     context->name, context->occurence, FSE, tune->lowerLimit[i]);
            printWarning("Quadrupole at lower limit during tune correction.", warningBuffer);
          } else if (tune->upperLimit && FSE > tune->upperLimit[i]) {
            FSE = *FSEptr = tune->upperLimit[i];
            nLimit++;
            snprintf(warningBuffer, 1024,
                     "%s#%ld is at the upper limit (FSE=%le, limit=%le).",
                     context->name, context->occurence, FSE, tune->upperLimit[i]);
            printWarning("Quadrupole at upper limit during tune correction.", warningBuffer);
          }
        } else {
          K1 = (*K1ptr += tune->dK1->a[i][0]);
          if (tune->lowerLimit && K1 < tune->lowerLimit[i]) {
            K1 = *K1ptr = tune->lowerLimit[i];
            nLimit++;
            snprintf(warningBuffer, 1024,
                     "%s#%ld is at the lower limit (K1=%le, limit=%le).",
                     context->name, context->occurence, K1, tune->lowerLimit[i]);
            printWarning("Quadrupole at lower limit during tune correction.", warningBuffer);
          } else if (tune->upperLimit && K1 > tune->upperLimit[i]) {
            K1 = *K1ptr = tune->upperLimit[i];
            nLimit++;
            snprintf(warningBuffer, 1024,
                     "%s#%ld is at the upper limit (K1=%le, limit=%le).",
                     context->name, context->occurence, K1, tune->upperLimit[i]);
            printWarning("Quadrupole at upper limit during tune correction.", warningBuffer);
          }
        }
        nTotal++;
        if (context->matrix) {
          free_matrices(context->matrix);
          free(context->matrix);
          context->matrix = NULL;
        }
        compute_matrix(context, run, NULL);
        type = context->type;
        if (has_wc && alter_defined_values) {
          change_defined_parameter(context->name, K1_param, type, K1, NULL, LOAD_FLAG_ABSOLUTE);
        }
        if (tune->verbosity > 1) {
          printf("change of %s#%ld[K1] is  %.15g 1/m^3\n", context->name, context->occurence, K1 - K10);
          fflush(stdout);
        }
        if (K1!=K10 || FSE!=FSE0)
          nChanged++;
      }
      if (!has_wc && alter_defined_values) {
        if (tune->verbosity > 2) {
          printf("new value of %s[K1] is  %.15g 1/m^3\n", tune->name[i], K1);
          fflush(stdout);
        }
        if (K1_param == -1)
          bombElegant("Error: K1_param==-1 in do_tune_correction---seek expert help!\n", NULL);
        change_defined_parameter(tune->name[i], K1_param, type, K1, NULL, LOAD_FLAG_ABSOLUTE);
      }
    }
    if (++steps_since_gain_change == tune->step_up_interval) {
      if ((gain += tune->delta_gain) > tune->maximum_gain)
        gain = tune->maximum_gain;
      steps_since_gain_change = 0;
    }

    if (beamline->links)
      assert_element_links(beamline->links, run, beamline,
                           STATIC_LINK + DYNAMIC_LINK + (alter_defined_values ? LINK_ELEMENT_DEFINITION : 0));

    if (do_closed_orbit && (tune->update_orbit != 0 && i % tune->update_orbit == 0)) {
      if (tune->verbosity > 1) {
        printf("Updating closed orbit\n");
        fflush(stdout);
      }
      run_closed_orbit(run, beamline, clorb, NULL, 0);
    }
    M = beamline->matrix = compute_periodic_twiss(&beta_x, &alpha_x, &eta_x, &etap_x, beamline->tune,
                                                  &beta_y, &alpha_y, &eta_y, &etap_y, beamline->tune + 1,
                                                  beamline->elem_twiss, clorb, run, &unstable, NULL,
                                                  NULL);

    beamline->twiss0->betax = beta_x;
    beamline->twiss0->alphax = alpha_x;
    beamline->twiss0->phix = 0;
    beamline->twiss0->etax = eta_x;
    beamline->twiss0->etapx = etap_x;
    beamline->twiss0->betay = beta_y;
    beamline->twiss0->alphay = alpha_y;
    beamline->twiss0->phiy = 0;
    beamline->twiss0->etay = eta_y;
    beamline->twiss0->etapy = etap_y;

    if (!M || !M->C || !M->R)
      bombElegant("something wrong with transfer map for beamline (do_tune_correction.2)", NULL);

    propagate_twiss_parameters(beamline->twiss0, beamline->tune, beamline->waists,
                               NULL, beamline->elem_twiss, run, clorb, NULL,
                               beamline->couplingFactor);
    if (tune->verbosity > 1) {
      printf("new tunes: %e %e\n", beamline->tune[0], beamline->tune[1]);
      fflush(stdout);
    }

    /*
    if (factor!=1)
      break;
    */

    if (nTotal == nLimit || nChanged == 0)
      break;
  }

#ifdef DEBUG
  printf("   Finished iterations\n");
  fflush(stdout);
#endif

  if (tune->verbosity > 0)
    printf("Tune correction completed after %ld iterations\n", iter);
  if (tune->verbosity == 1)
    printf("final tunes  : %e %e\n", beamline->tune[0], beamline->tune[1]);
  if (tune->verbosity > 0)
    fflush(stdout);

  if (fp_sl && last_iteration) {
    tunes_saved = 0;
    if (step != 1)
      fprintf(fp_sl, "\n%ld\n", step);
    else
      fprintf(fp_sl, "%ld\n", step);
    for (i = 0; i < tune->n_families; i++) {
      context = NULL;
      while ((context = wfind_element(tune->name[i], &context, beamline->elem))) {
        if (tune->n_exclude) {
          long j, excluded;
          for (j = excluded = 0; j < tune->n_exclude; j++)
            if (wild_match(context->name, tune->exclude[j])) {
              excluded = 1;
              break;
            }
          if (excluded)
            continue;
        }
        K1_param = confirm_parameter(tune->item[i], context->type);
        if (!(K1ptr = (double *)(context->p_elem + entity_description[context->type].parameter[K1_param].offset)))
          bombElegant("K1ptr NULL in do_tune_correction", NULL);
        fprintf(fp_sl, "%s %s %ld %21.15e\n", context->name, tune->item[i], context->occurence, *K1ptr);
      }
    }
    fflush(fp_sl);
  }

#ifdef DEBUG
  printf("   Done with tune correction\n");
  fflush(stdout);
#endif

  return 1;
}

double *scanNumberList(char *list, long *nFound) {
  long nll;
  char *ptr;
  double *dList;
  nll = 0;
  dList = NULL;
  while ((ptr = get_token(list))) {
    dList = SDDS_Realloc(dList, sizeof(*dList) * (nll + 1));
    if (sscanf(ptr, "%le", &dList[nll]) != 1)
      return NULL;
    nll++;
  }
  *nFound = nll;
  return dList;
}

/* ---------------------------------------------------------------- */
/* Per-step corrector stash hooks (tune correction).  Walks each tune
 * family pattern (tune_stash_corr->name[i]) over the beamline and
 * records the parameter index named by tune_stash_corr->item[i]. */

void correct_tunes_save_correctors(RUN *run, LINE_LIST *beamline) {
  long i;
  (void)run;
  if (tune_stash_corr == NULL || tune_stash_corr->n_families <= 0) return;
  corstash_clear(&tuneStash);
  for (i = 0; i < tune_stash_corr->n_families; i++) {
    ELEMENT_LIST *context = NULL;
    long paramIndex = -1;
    long lastType = -1;
    while ((context = wfind_element(tune_stash_corr->name[i], &context, beamline->elem))) {
      if (tune_stash_corr->n_exclude) {
        long j, excluded;
        for (j = excluded = 0; j < tune_stash_corr->n_exclude; j++)
          if (wild_match(context->name, tune_stash_corr->exclude[j])) {
            excluded = 1;
            break;
          }
        if (excluded) continue;
      }
      if (context->type != lastType) {
        paramIndex = confirm_parameter(tune_stash_corr->item[i], context->type);
        lastType = context->type;
      }
      if (paramIndex < 0) continue;
      corstash_add(&tuneStash, context, paramIndex);
    }
  }
  corstash_snapshot(&tuneStash);
}

long correct_tunes_reassert_correctors(RUN *run, LINE_LIST *beamline) {
  return corstash_reassert(&tuneStash, run, beamline);
}

void correct_tunes_invalidate_correctors(void) {
  corstash_clear(&tuneStash);
  tune_stash_corr = NULL;
}
