/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* contents: compute_matrices(), drift_matrix(), sextupole_matrix() 
 *           plus more...
 * Michael Borland, 1989.
 */
#include "mdb.h"
#include "track.h"
#include "matlib.h"

void InitializeCWiggler(CWIGGLER *cwiggler, char *name);
VMATRIX *matrixFromExplicitMatrix(EMATRIX *emat, long order);
VMATRIX *matrixForILMatrix(ILMATRIX *ilmat, long order);
VMATRIX *rfdf_matrix(RFDF *rfdf, double Preference);
VMATRIX *sextupoleFringeMatrix(double K2, double length, long maxOrder, long side);
VMATRIX *mult_matrix(MULT *mult, double P, long maxOrder);
VMATRIX *interpolateMatrixWithIdentityMatrix(VMATRIX *M0, double fraction, long order);

static double timeCounter[N_TYPES];
static long runCounter[N_TYPES];
static long timingActive = 0;
void startMatrixComputationTiming() {
  long i;
  for (i = 0; i < N_TYPES; i++)
    timeCounter[i] = runCounter[i] = 0;
  timingActive = 1;
}
void reportMatrixComputationTiming() {
  if (timingActive) {
    long i;
    printf("Time spent computing matrices for different elements:\n");
    for (i = 0; i < N_TYPES; i++) {
      if (runCounter[i] != 0)
        printf("%16s: %10ld times, %10.3e s, %10.3e s/element\n", entity_name[i], runCounter[i], timeCounter[i], timeCounter[i] / runCounter[i]);
    }
    fflush(stdout);
  }
  timingActive = 0;
}

void checkMatrices(char *label, ELEMENT_LIST *elem) {
  ELEMENT_LIST *eptr;
  eptr = elem;
  printf("Matrix check %s\n", label);
  while (eptr) {
    if ((entity_description[eptr->type].flags & HAS_MATRIX) && !(eptr->matrix)) {
      printf("%s #%ld has no matrix\n", eptr->name, eptr->occurence);
    }
    eptr = eptr->succ;
  }
}

void determineDQCorReferenceFrameTilt(DQCOR *dqcor)
{
  double phi;
  /* atan(2*phi) = J1/K1. The 2 is because it is a quadrupole (r^2*exp(2*i*phi)) */
  phi = atan2(dqcor->j1, dqcor->k1)/2;
  dqcor->K1ReferenceFrame = sqrt(sqr(dqcor->j1) + sqr(dqcor->k1));

  dqcor->xKickReferenceFrame = dqcor->xkick*dqcor->xKickCalibration;
  dqcor->yKickReferenceFrame = dqcor->ykick*dqcor->yKickCalibration;
  rotate_xy(&(dqcor->xKickReferenceFrame), &(dqcor->yKickReferenceFrame), phi);

  dqcor->phiReferenceFrame = phi;
}

double computeUndulatorFieldFromModel(double gap, double fractionOfJc, double *C, double length, long poles, char *model)
{
#define UND_MODEL_DEJUS_NdFeB 0
#define UND_MODEL_DEJUS_SmCo 1
#define UND_MODEL_MOOG_PrFeB_77K 2
#define UND_MODEL_MOOG_NdFeB_150K 3
#define UND_MODEL_HALBACH 4
#define UND_MODEL_SC_NbTi 5
#define UND_MODEL_SC_Nb3Sn 6
#define UND_MODEL_CUSTOM 7
#define N_MODELS 8
  double r;
  short type = 0; /* Halbach-type expression for HPMU */
  /* Arrays for S. H. Kim model of SCU */
#define SCU_ARRAY_SIZE 100
  static double *jn = NULL, *BmRef = NULL, *B0Ref = NULL, *j = NULL, *B0 = NULL, *deltaB = NULL;
  static char *modelOption[N_MODELS] = {
    "NdFeB-RT", "SmCo-RT", "PrFeB-77K", "NdFeB-150K", "Halbach", "NbTi", "Nb3Sn", "Custom"
  };
  double C1, C2, C3;
  double Bc2, ja, period0, jMax, jOp, B0Op;
  long i, interpCode;
  double period = 0;
  if (poles>2)
    period = 2*(length/poles);
  if (period<=0 || gap<=0)
    bombElegantVA("Period (%le) or gap (%le) invalid for planar undulator models\n", period, gap);
  C1 = C2 = C3 = 0;
  switch (match_string(model, modelOption, N_MODELS, 0)) {
  case UND_MODEL_DEJUS_NdFeB:
    C1 = 3.276;
    C2 = -4.51;
    C3 = 1.20;
    break;
  case UND_MODEL_DEJUS_SmCo:
    C1 = 2.94;
    C2 = -4.62;
    C3 = 1.37;
    break;
  case UND_MODEL_MOOG_PrFeB_77K:
    C1 = 3.502;
    C2 = -3.604;
    C3 = 0.359;
    break;
  case UND_MODEL_MOOG_NdFeB_150K:
    C1 = 3.341;
    C2 = -3.606;
    C3 = 0.300;
    break;
  case UND_MODEL_HALBACH:
    C1 = 3.33;
    C2 = -5.47;
    C3 = 1.8;
    break;
  case UND_MODEL_CUSTOM:
    C1 = C[0];
    C2 = C[1];
    C3 = C[2];
    break;
  case UND_MODEL_SC_NbTi:
    type = 1;
    break;
  case UND_MODEL_SC_Nb3Sn:
    type = 2;
    break;
  default:
    bombElegantVA("Unknown undulator model \"%s\"\n", model);
    break;
  }
  if (!type) {
    r = gap/period;
    return C1*exp((C3*r+C2)*r);
  }
  if (fractionOfJc>1 || fractionOfJc<0) 
    bombElegantVA("Superconducting undulator model evaluation invalid for j/jC = %le\n", fractionOfJc);
  if (jn==NULL) {
    double jn1, jn0, djn;
    /* initialize arrays for interpolation of S. H. Kim SCU model */
    jn = tmalloc(sizeof(*jn)*SCU_ARRAY_SIZE);
    BmRef = tmalloc(sizeof(double)*SCU_ARRAY_SIZE);
    B0Ref = tmalloc(sizeof(double)*SCU_ARRAY_SIZE);
    j = tmalloc(sizeof(double)*SCU_ARRAY_SIZE);
    B0 = tmalloc(sizeof(double)*SCU_ARRAY_SIZE);
    deltaB = tmalloc(sizeof(double)*SCU_ARRAY_SIZE);
    jn1 = 2.5;
    jn0 = 0.5;
    djn = (jn1 - jn0) / (SCU_ARRAY_SIZE - 1);
    for (i = 0; i < SCU_ARRAY_SIZE; i++) {
      jn[i] = jn0 + djn * i;
      B0Ref[i] = 0.28588 + 0.69021 * jn[i] - 0.022496 * jn[i] * jn[i];
      BmRef[i] = 1.8228 + 0.67253 * jn[i] + 0.88641 * jn[i] * jn[i] - 0.2227 * ipow(jn[i], 3);
    }
  }
  Bc2 = 10.4;
  ja = 2.111;
  period0 = 16e-3;
  for (i = 0; i < SCU_ARRAY_SIZE ; i++) {
    j[i] = jn[i] * period0 / period;
    B0[i] = B0Ref[i] / exp(PI *(gap / period - 0.5));
    deltaB[i] = (1 - j[i] / ja) * Bc2 - BmRef[i];
  }
  jMax = interp(j, deltaB, SCU_ARRAY_SIZE, 0.0, 0, 1, &interpCode);
  if (!interpCode) {
    bombElegantVA("Problem (1) with superconducting undulator model evaluation for period = %le, j/jC = %le, model = %s. Code = %ld\n",
		  period, fractionOfJc, model, interpCode);
  } else {
    jOp = fractionOfJc * jMax;
    B0Op = interp(B0, j, SCU_ARRAY_SIZE, jOp, 0, 1, &interpCode);
    if (!interpCode) {
      bombElegantVA("Problem (2) with superconducting undulator model evaluation for period = %le, j/jC = %le, model = %s. Code = %ld\n",
		    period, fractionOfJc, model, interpCode);
    } else {
      if (type==2) /* Nb3Sn */
	B0Op *= 1.3;
      return B0Op;
    }
  }
  return 0.0;
}

VMATRIX *full_matrix(ELEMENT_LIST *elem, RUN *run, long order) {
  VMATRIX *M;
  log_entry("full_matrix");

  if (!elem) {
    fputs("error: NULL element pointer passed to full_matrix", stdout);
    abort();
  }

#ifdef WATCH_MEMORY
  printf("start full_matrix: CPU: %6.2lf  PF: %6ld  MEM: %6ld\n",
         cpu_time() / 100.0, page_faults(), memory_count());
  fflush(stdout);
#endif

  M = accumulate_matrices(elem, run, NULL, order, 1);
  log_exit("full_matrix");
  return M;
}

VMATRIX *accumulate_matrices(ELEMENT_LIST *elem, RUN *run, VMATRIX *M0, long order, long full_matrix_only) {
  VMATRIX *M1, *M2, *tmp;
  ELEMENT_LIST *member;
  double Pref_input;
  long i;

  log_entry("accumulate_matrices");

  if (!elem) {
    fputs("error: NULL element pointer passed to accumulate_matrices", stdout);
    abort();
  }

  initialize_matrices(M1 = tmalloc(sizeof(*M1)), order);
  initialize_matrices(M2 = tmalloc(sizeof(*M2)), order);
  for (i = 0; i < 6; i++)
    M1->R[i][i] = M2->R[i][i] = 1;
  if (M0)
    copy_matrices1(M1, M0);

  member = elem;

  while (member) {
    if (member->type < 0 || member->type >= N_TYPES) {
      printf("error: bad element type %ld (accumulate_matrices)\n", member->type);
      fflush(stdout);
      printf("element name is %s and end position is %em\n",
             (member->name ? member->name : "{null}"), member->end_pos);
      fflush(stdout);
      abort();
    }
    if (member->pred)
      Pref_input = member->pred->Pref_output;
    else
      Pref_input = member->Pref_input;
    if (!member->matrix || Pref_input != member->Pref_input)
      compute_matrix(member, run, NULL);
    if (entity_description[member->type].flags & HAS_MATRIX) {
      if (!member->matrix) {
        printf("programming error: matrix not computed for element %s\n",
               member->name);
        fflush(stdout);
        abort();
      }
    }
    if (member->matrix) {
      concat_matrices(M2, member->matrix, M1,
                      entity_description[member->type].flags & HAS_RF_MATRIX ? CONCAT_EXCLUDE_S0 : 0);
      tmp = M2;
      M2 = M1;
      M1 = tmp;
      if (!full_matrix_only) {
        if (member->accumMatrix)
          free_matrices(member->accumMatrix);
        else
          member->accumMatrix = tmalloc(sizeof(*(member->accumMatrix)));
        copy_matrices(member->accumMatrix, M1);
      }
    } else if (!full_matrix_only) {
      if (member->accumMatrix)
        free_matrices(member->accumMatrix);
      else
        member->accumMatrix = tmalloc(sizeof(*(member->accumMatrix)));
      copy_matrices(member->accumMatrix, M1);
    }
    member = member->succ;
  }
  if (M2) {
    free_matrices(M2);
    tfree(M2);
    M2 = NULL;
  }
  log_exit("accumulate_matrices");
  return M1;
}

VMATRIX *append_full_matrix(ELEMENT_LIST *elem, RUN *run, VMATRIX *M0, long order) {
  return accumulate_matrices(elem, run, M0, order, 1);
}

VMATRIX *accumulateRadiationMatrices(ELEMENT_LIST *elem, RUN *run, VMATRIX *M0, long order, long radiation, long nSlices, long sliceEtilted,
                                     short ibsUpdate) {
  VMATRIX *M1, *M2, *Ml1, *Ml2, *tmp;
  ELEMENT_LIST *member;
  double Pref_input;
  long i, j, k;
  MATRIX *Ms;
#if USE_MPI
  long notSinglePart_saved;
  notSinglePart_saved = notSinglePart;
  notSinglePart = 0;
#endif

  if (!elem) {
    fputs("error: NULL element pointer passed to accumulateRadiationMatrices", stdout);
    abort();
  }

  initialize_matrices(M1 = tmalloc(sizeof(*M1)), order);
  initialize_matrices(M2 = tmalloc(sizeof(*M2)), order);
  initialize_matrices(Ml1 = tmalloc(sizeof(*Ml1)), 1);
  initialize_matrices(Ml2 = tmalloc(sizeof(*Ml2)), 1);
  for (i = 0; i < 6; i++)
    M1->R[i][i] = M2->R[i][i] = Ml1->R[i][i] = 1;
  if (M0)
    copy_matrices1(M1, M0);
  m_alloc(&Ms, 21, 21);

  member = elem;

  while (member) {
    if (member->type < 0 || member->type >= N_TYPES) {
      printf("error: bad element type %ld (accumulateRadiationMatrices)\n", member->type);
      fflush(stdout);
      printf("element name is %s and end position is %em\n",
             (member->name ? member->name : "{null}"), member->end_pos);
      fflush(stdout);
      abort();
    }
    if (member->pred)
      Pref_input = member->pred->Pref_output;
    else
      Pref_input = member->Pref_input;
    if (!member->matrix || Pref_input != member->Pref_input) {
      if (member->matrix) {
        free_matrices(member->matrix);
        free(member->matrix);
        member->matrix = NULL;
      }
      compute_matrix(member, run, NULL);
    }
    if ((entity_description[member->type].flags & HAS_MATRIX) && !member->matrix) {
      printf("programming error: matrix not computed for element %s\n",
             member->name);
      fflush(stdout);
      abort();
    }
    if (!(member->D))
      member->D = tmalloc(21 * sizeof(*(member->D)));
    if (!ibsUpdate)
      memset(member->D, 0, 21 * sizeof(*(member->D)));
    if (!(member->accumD))
      member->accumD = tmalloc(21 * sizeof(*(member->accumD)));
    memset(member->accumD, 0, 21 * sizeof(*(member->accumD)));
    if (member->matrix) {
      /* The matrix variables are as follows:
         M1:  the concatenated matrix up to this point
         M2:  working variable for updating M1
         Ml1: unit R matrix, but centroid matrix may be changed for use in concatenation.
         Ml2: holds result of computing the linearized matrix for the element we are working on,
              but with the full trajectory in C (not just the contribution).
       */
      /* Step 1: determine effective R matrix for this element 
       * either by tracking through the element or by concatenating
       * the incoming trajectory with the element's matrix.
       */
      if (radiation && (IS_RADIATOR(member->type) || member->type == T_RFCA || member->type == T_TWLA)) {
        if (!(member->Mld0)) {
          member->Mld0 = tmalloc(sizeof(*(member->Mld0)));
          initialize_matrices(member->Mld0, 1);
        }
        if (!ibsUpdate) {
          /* Must include radiation, so do tracking */
          determineRadiationMatrix(Ml2, run, member, M1->C, member->D, nSlices, sliceEtilted, order);
          copy_matrices(member->Mld0, Ml2);
        } else
          copy_matrices(Ml2, member->Mld0);
        memcpy(member->accumD, member->D, 21 * sizeof(*(member->D)));
      } else if (member->type == T_SREFFECTS) {
        /* Must not use the matrix for these elements, as it may double-count radiation losses */
        for (i = 0; i < 6; i++) {
          /* Copy the centroid */
          Ml2->C[i] = M1->C[i];
          /* Copy the unit R matrix */
          memcpy(Ml2->R[i], Ml1->R[i], 6 * sizeof(*(Ml2->R[i])));
        }
      } else {
        /* Just use the matrix computed above.
         * Concatenate incoming centroid with the matrix to get the on-orbit R matrix 
         */
        memcpy(Ml1->C, M1->C, 6 * sizeof(*(Ml1->C)));
        concat_matrices(Ml2, member->matrix, Ml1,
                        entity_description[member->type].flags & HAS_RF_MATRIX ? CONCAT_EXCLUDE_S0 : 0);
      }
      if (member->DIbs) {
	/* Add the IBS contribution to the diffusion matrix */
#ifdef DEBUG
	printf("Element %s diffusion matrix without IBS:\n", member->name);
	for (i=0; i<6; i++) {
	  printf("D[%ld]: ", i);
	  for (j=0; j<=i; j++)
	    printf("%13.6e ", member->accumD[sigmaIndex3[i][j]]);
	  printf("\n");
	}
	fflush(stdout);
	printf("Element %s IBS contribution:\n", member->name);
	for (i=0; i<6; i++) {
	  printf("D[%ld]: ", i);
	  for (j=0; j<=i; j++)
	    printf("%13.6e ", member->DIbs[sigmaIndex3[i][j]]);
	  printf("\n");
	}
#endif
	for (i=0; i<21; i++)
	  member->accumD[i] += member->DIbs[i];
#ifdef DEBUG
	printf("Element %s diffusion matrix with IBS:\n", member->name);
	for (i=0; i<6; i++) {
	  printf("D[%ld]: ", i);
	  for (j=0; j<=i; j++)
	    printf("%13.6e ", member->accumD[sigmaIndex3[i][j]]);
	  printf("\n");
	}
	fflush(stdout);
#endif
      }
      /* Step 2: Copy the C vector */
      memcpy(M2->C, Ml2->C, 6 * sizeof(*(M2->C)));
      /* Step 3: Propagate the diffusion matrix */
      if (member->pred && member->pred->accumD) {
        fillSigmaPropagationMatrix(Ms->a, Ml2->R);
        for (i = 0; i < 21; i++)
          for (j = 0; j < 21; j++)
            member->accumD[i] += Ms->a[i][j] * member->pred->accumD[j];
      }
#ifdef DEBUG
      printf("Accumulated diffusion matrix:\n");
      for (i=0; i<6; i++) {
	printf("D[%ld]: ", i);
	for (j=0; j<=i; j++)
	  printf("%13.6e ", member->accumD[sigmaIndex3[i][j]]);
	printf("\n");
      }
      fflush(stdout);
#endif
      /* Step 5: Multiply the R matrices */
      for (i = 0; i < 6; i++)
        for (j = 0; j < 6; j++) {
          M2->R[i][j] = 0;
          for (k = 0; k < 6; k++)
            M2->R[i][j] += Ml2->R[i][k] * M1->R[k][j];
        }
      tmp = M2;
      M2 = M1;
      M1 = tmp;
      /* Step 6: Store the linear damping matrix and trajectory. */
      /* This matrix must be used carefully because the C component contains the full
       * trajectory, not just the contribution.  It should only be used for moments
       * propagation.
       */
      if (!(member->Mld)) {
        member->Mld = tmalloc(sizeof(*(member->Mld)));
        initialize_matrices(member->Mld, 1);
      }
      for (i = 0; i < 6; i++) {
        member->Mld->C[i] = Ml2->C[i];
        memcpy(member->Mld->R[i], Ml2->R[i], 6 * sizeof(*(Ml2->R[i])));
      }
    } else {
      if (member->pred && member->pred->accumD)
        memcpy(member->accumD, member->pred->accumD, 21 * sizeof(*(member->accumD)));
      if (!(member->Mld)) {
        member->Mld = tmalloc(sizeof(*(member->Mld)));
        initialize_matrices(member->Mld, 1);
      }
      for (i = 0; i < 6; i++) {
        member->Mld->C[i] = M1->C[i];
        member->Mld->R[i][i] = 1;
      }
      if (member->type == T_ENERGY) {
        ENERGY *energy;
        energy = (ENERGY *)member->p_elem;
        if (energy->match_beamline) {
          /* Reference momentum is changed to match the centroid of the particles,
           * so set momentum centroid to zero */
          member->Mld->C[5] = M1->C[5] = 0;
        } else {
          printf("Error: ENERGY element with MATCH_BEAMLINE not set encountered in moments analysis.\n");
          printf("ENERGY element can only be included in MATCH_BEAMLINE mode at present.\n");
          exitElegant(1);
        }
      }
    }
    member = member->succ;
  }
  if (M2) {
    free_matrices(M2);
    tfree(M2);
    M2 = NULL;
  }
  if (Ml1) {
    free_matrices(Ml1);
    tfree(Ml1);
    Ml1 = NULL;
  }
  if (Ml2) {
    free_matrices(Ml2);
    tfree(Ml2);
    Ml2 = NULL;
  }
  m_free(&Ms);
#if USE_MPI
  notSinglePart = notSinglePart_saved;
#endif
  return M1;
}

long fill_in_matrices(
  ELEMENT_LIST *elem,
  RUN *run) {
  ELEMENT_LIST *member;
  long n_elements;

  n_elements = 0;
  member = elem;
  while (member) {
    if (member->type < 0 || member->type >= N_TYPES) {
      printf("error: bad element type %ld (fill_in_matrices)\n", member->type);
      fflush(stdout);
      printf("element name is %s and end position is %em\n",
             (member->name ? member->name : "{null}"), member->end_pos);
      fflush(stdout);
      abort();
    }
    if ((member->matrix == NULL || (member->pred && member->pred->Pref_output != member->Pref_input)) &&
        entity_description[member->type].flags & HAS_MATRIX) {
      compute_matrix(member, run, NULL);
      n_elements++;
    }
    member = member->succ;
  }

  return (n_elements);
}

long calculate_matrices(
  LINE_LIST *line, /* Beamline to calculate matrices for. */
  RUN *run) {
  ELEMENT_LIST *member;
  long n_elements;

  log_entry("calculate_matrices");

  n_elements = 0;
  member = line->elem;
  line->elem_recirc = NULL;
  line->i_recirc = 0;
  while (member) {
    if (!member->matrix || (member->pred && member->pred->Pref_output != member->Pref_input))
      compute_matrix(member, run, NULL);
    if (member->type == T_RECIRC) {
      line->elem_recirc = member;
      line->i_recirc = n_elements;
    }
    n_elements++;
    member = member->succ;
  }
  log_exit("calculate_matrices");
  return (n_elements);
}

VMATRIX *drift_matrix(double length, long order) {
  VMATRIX *M;
  double *C, **R, ***T;

  log_entry("drift_matrix");

  M = tmalloc(sizeof(*M));
  M->order = (order > 2 ? 2 : order);
  initialize_matrices(M, M->order);
  R = M->R;
  C = M->C;
  T = M->T;

  C[4] = length;
  R[0][0] = R[1][1] = R[2][2] = R[3][3] = R[4][4] = R[5][5] = 1;
  R[0][1] = R[2][3] = length;

  if (order > 1)
    T[4][1][1] = T[4][3][3] = length / 2;

  log_exit("drift_matrix");
  return (M);
}

VMATRIX *wiggler_matrix(double length, double radius, long poles,
                        long order, long focusing) {
  VMATRIX *M;
  double **R, *C;
  double kl;

  M = tmalloc(sizeof(*M));
  M->order = 1;
  initialize_matrices(M, M->order);
  R = M->R;
  C = M->C;

  R[0][0] = R[1][1] = R[2][2] = R[3][3] = R[4][4] = R[5][5] = 1;

  if (length) {
    C[4] = length;
    R[0][1] = length;
    if (focusing) {
      kl = length / (SQRT2 * fabs(radius));
      R[2][2] = R[3][3] = cos(kl);
      R[2][3] = sin(kl) / (kl / length);
      R[3][2] = -(kl / length) * sin(kl);
    } else {
      R[2][3] = length;
    }
    R[4][5] = -(poles / 2) * ipow2(1 / radius) * ipow3(length / poles) / ipow2(PI);
  }

  return (M);
}

VMATRIX *sextupole_matrix(double K2, double K1, double J1, double length, long maximum_order, double fse, double xkick, double ykick, double ffringe) {
  VMATRIX *M, *Medge1, *Medge2;
  VMATRIX *Mfringe, *Mtot;
  double *C, **R, ***T = NULL, ****U = NULL;
  double temp, lf = 0;

  Medge1 = Medge2 = NULL;

  K1 *= (1 + fse);
  J1 *= (1 + fse);
  K2 *= (1 + fse);

  M = tmalloc(sizeof(*M));
  initialize_matrices(M, M->order = MIN(3, maximum_order));
  R = M->R;
  C = M->C;

  if (ffringe > 0) {
    lf = length * ffringe / 2;
    length *= (1 - ffringe / 2);
    Medge1 = sextupoleFringeMatrix(K2, lf, maximum_order, -1);
    Medge2 = sextupoleFringeMatrix(K2, lf, maximum_order, 1);
  }

  R[0][0] = R[1][1] = R[2][2] = R[3][3] = R[4][4] = R[5][5] = 1;
  C[4] = R[0][1] = R[2][3] = length;

  if (M->order >= 2) {
    T = M->T;
    temp = K2 * length / 2; /* temp = ks^2*l */
    T[1][0][0] = -(T[1][2][2] = temp);
    T[3][2][0] = 2 * temp;
    temp *= length; /* temp = ks^2*l^2 */
    T[0][0][0] = -temp / 2;
    T[0][2][2] = temp / 2;
    T[1][1][0] = -temp;
    T[1][3][2] = temp;
    T[2][2][0] = T[3][3][0] = T[3][2][1] = temp;
    temp *= length; /* temp = ks^2*l^3 */
    T[0][1][0] = -temp / 3.;
    T[1][1][1] = -temp / 3.;
    T[0][3][2] = T[2][3][0] = T[2][2][1] = temp / 3.;
    T[1][3][3] = temp / 3.;
    T[3][3][1] = 2 * temp / 3;
    temp *= length; /* temp = ks^2*l^4 */
    T[0][1][1] = -temp / 12;
    T[0][3][3] = temp / 12;
    T[2][3][1] = temp / 6;
    /* path length terms--same as for drift */
    T[4][1][1] = T[4][3][3] = length / 2;

    if (M->order >= 3) {
      U = M->Q;
      U[0][0][0][0] = ipow2(K2) * ipow4(length) / 48.0;
      U[0][1][0][0] = ipow2(K2) * ipow5(length) / 48.0;
      U[0][1][1][0] = ipow2(K2) * ipow6(length) / 144.0;
      U[0][1][1][1] = ipow2(K2) * ipow7(length) / 1008.0;
      U[0][2][2][0] = ipow2(K2) * ipow4(length) / 48.0;
      U[0][2][2][1] = -ipow2(K2) * ipow5(length) / 240.0;
      U[0][3][2][0] = ipow2(K2) * ipow5(length) / 40.0;
      U[0][3][2][1] = ipow2(K2) * ipow6(length) / 360.0;
      U[0][3][3][0] = ipow2(K2) * ipow6(length) / 240.0;
      U[0][3][3][1] = ipow2(K2) * ipow7(length) / 1008.0;
      U[0][5][0][0] = K2 * ipow2(length) / 4.0;
      U[0][5][1][0] = K2 * ipow3(length) / 6.0;
      U[0][5][1][1] = K2 * ipow4(length) / 24.0;
      U[0][5][2][2] = -K2 * ipow2(length) / 4.0;
      U[0][5][3][2] = -K2 * ipow3(length) / 6.0;
      U[0][5][3][3] = -K2 * ipow4(length) / 24.0;
      U[1][0][0][0] = ipow2(K2) * ipow3(length) / 12.0;
      U[1][1][0][0] = 5.0 * ipow2(K2) * ipow4(length) / 48.0;
      U[1][1][1][0] = ipow2(K2) * ipow5(length) / 24.0;
      U[1][1][1][1] = ipow2(K2) * ipow6(length) / 144.0;
      U[1][2][2][0] = ipow2(K2) * ipow3(length) / 12.0;
      U[1][2][2][1] = -ipow2(K2) * ipow4(length) / 48.0;
      U[1][3][2][0] = ipow2(K2) * ipow4(length) / 8.0;
      U[1][3][2][1] = ipow2(K2) * ipow5(length) / 60.0;
      U[1][3][3][0] = ipow2(K2) * ipow5(length) / 40.0;
      U[1][3][3][1] = ipow2(K2) * ipow6(length) / 144.0;
      U[1][5][0][0] = K2 * length / 2.0;
      U[1][5][1][0] = K2 * ipow2(length) / 2.0;
      U[1][5][1][1] = K2 * ipow3(length) / 6.0;
      U[1][5][2][2] = -K2 * length / 2.0;
      U[1][5][3][2] = -K2 * ipow2(length) / 2.0;
      U[1][5][3][3] = -K2 * ipow3(length) / 6.0;
      U[2][2][0][0] = ipow2(K2) * ipow4(length) / 48.0;
      U[2][2][1][0] = ipow2(K2) * ipow5(length) / 40.0;
      U[2][2][1][1] = ipow2(K2) * ipow6(length) / 240.0;
      U[2][2][2][2] = ipow2(K2) * ipow4(length) / 48.0;
      U[2][3][0][0] = -ipow2(K2) * ipow5(length) / 240.0;
      U[2][3][1][0] = ipow2(K2) * ipow6(length) / 360.0;
      U[2][3][1][1] = ipow2(K2) * ipow7(length) / 1008.0;
      U[2][3][2][2] = ipow2(K2) * ipow5(length) / 48.0;
      U[2][3][3][2] = ipow2(K2) * ipow6(length) / 144.0;
      U[2][3][3][3] = ipow2(K2) * ipow7(length) / 1008.0;
      U[2][5][2][0] = -K2 * ipow2(length) / 2.0;
      U[2][5][2][1] = -K2 * ipow3(length) / 6.0;
      U[2][5][3][0] = -K2 * ipow3(length) / 6.0;
      U[2][5][3][1] = -K2 * ipow4(length) / 12.0;
      U[3][2][0][0] = ipow2(K2) * ipow3(length) / 12.0;
      U[3][2][1][0] = ipow2(K2) * ipow4(length) / 8.0;
      U[3][2][1][1] = ipow2(K2) * ipow5(length) / 40.0;
      U[3][2][2][2] = ipow2(K2) * ipow3(length) / 12.0;
      U[3][3][0][0] = -ipow2(K2) * ipow4(length) / 48.0;
      U[3][3][1][0] = ipow2(K2) * ipow5(length) / 60.0;
      U[3][3][1][1] = ipow2(K2) * ipow6(length) / 144.0;
      U[3][3][2][2] = 5.0 * ipow2(K2) * ipow4(length) / 48.0;
      U[3][3][3][2] = ipow2(K2) * ipow5(length) / 24.0;
      U[3][3][3][3] = ipow2(K2) * ipow6(length) / 144.0;
      U[3][5][2][0] = -K2 * length;
      U[3][5][2][1] = -K2 * ipow2(length) / 2.0;
      U[3][5][3][0] = -K2 * ipow2(length) / 2.0;
      U[3][5][3][1] = -K2 * ipow3(length) / 3.0;
      U[4][1][0][0] = -K2 * ipow2(length) / 4.0;
      U[4][1][1][0] = -K2 * ipow3(length) / 6.0;
      U[4][1][1][1] = -K2 * ipow4(length) / 24.0;
      U[4][2][2][1] = K2 * ipow2(length) / 4.0;
      U[4][3][2][0] = K2 * ipow2(length) / 2.0;
      U[4][3][2][1] = K2 * ipow3(length) / 3.0;
      U[4][3][3][0] = K2 * ipow3(length) / 6.0;
      U[4][3][3][1] = K2 * ipow4(length) / 8.0;
    }
  }

  if (K1 != 0 || J1 != 0) {
    /* Add terms related to skew quadrupole, valid only if skew quad is "weak" */
    double L = length;
    double K1p2;
    double J1p2;
    double Lp2, Lp3, Lp4, Lp5;
    K1p2 = pow(K1, 2);
    J1p2 = pow(J1, 2);
    Lp2 = pow(L, 2);
    Lp3 = pow(L, 3);
    Lp4 = pow(L, 4);
    Lp5 = pow(L, 5);
    R[0][0] += -(K1 * Lp2) / 2. + (J1p2 * Lp4) / 24. + (K1p2 * Lp4) / 24.;
    R[0][1] += -(K1 * Lp3) / 6. + (J1p2 * Lp5) / 120. + (K1p2 * Lp5) / 120.;
    R[0][2] += -(J1 * Lp2) / 2.;
    R[0][3] += -(J1 * Lp3) / 6.;
    R[1][0] += -(K1 * L) + (J1p2 * Lp3) / 6. + (K1p2 * Lp3) / 6.;
    R[1][1] += -(K1 * Lp2) / 2. + (J1p2 * Lp4) / 24. + (K1p2 * Lp4) / 24.;
    R[1][2] += -(J1 * L);
    R[1][3] += -(J1 * Lp2) / 2.;
    R[2][0] += -(J1 * Lp2) / 2.;
    R[2][1] += -(J1 * Lp3) / 6.;
    R[2][2] += (K1 * Lp2) / 2. + (J1p2 * Lp4) / 24. + (K1p2 * Lp4) / 24.;
    R[2][3] += (K1 * Lp3) / 6. + (J1p2 * Lp5) / 120. + (K1p2 * Lp5) / 120.;
    R[3][0] += -(J1 * L);
    R[3][1] += -(J1 * Lp2) / 2.;
    R[3][2] += K1 * L + (J1p2 * Lp3) / 6. + (K1p2 * Lp3) / 6.;
    R[3][3] += (K1 * Lp2) / 2. + (J1p2 * Lp4) / 24. + (K1p2 * Lp4) / 24.;

    if (M->order >= 2) {
      double Lp6, Lp7, Lp8, Lp9;
      double K1p3, K1p4;
      double J1p3, J1p4;
      double K2p2;
      Lp6 = pow(L, 6);
      Lp7 = pow(L, 7);
      Lp8 = pow(L, 8);
      Lp9 = pow(L, 9);
      K1p3 = pow(K1, 3);
      K1p4 = pow(K1, 4);
      J1p3 = pow(J1, 3);
      J1p4 = pow(J1, 4);
      K2p2 = pow(K2, 2);

      T[0][0][0] += (K1 * K2 * Lp4) / 16. + (J1p2 * K2 * Lp6) / 240. - (K1p2 * K2 * Lp6) / 240.;
      T[0][1][0] += (K1 * K2 * Lp5) / 24. + (J1p2 * K2 * Lp7) / 504. - (K1p2 * K2 * Lp7) / 504.;
      T[0][1][1] += (K1 * K2 * Lp6) / 144. + (J1p2 * K2 * Lp8) / 4032. - (K1p2 * K2 * Lp8) / 4032.;
      T[0][2][0] += -(J1 * K2 * Lp4) / 24. - (J1 * K1 * K2 * Lp6) / 60.;
      T[0][2][1] += (J1 * K2 * Lp5) / 120. - (J1 * K1 * K2 * Lp7) / 252.;
      T[0][2][2] += (K1 * K2 * Lp4) / 48. - (J1p2 * K2 * Lp6) / 240. + (K1p2 * K2 * Lp6) / 240.;
      T[0][3][0] += -(J1 * K2 * Lp5) / 40. - (J1 * K1 * K2 * Lp7) / 252.;
      T[0][3][1] += -(J1 * K2 * Lp6) / 360. - (J1 * K1 * K2 * Lp8) / 1008.;
      T[0][3][2] += (K1 * K2 * Lp5) / 40. - (J1p2 * K2 * Lp7) / 504. + (K1p2 * K2 * Lp7) / 504.;
      T[0][3][3] += (K1 * K2 * Lp6) / 240. - (J1p2 * K2 * Lp8) / 4032. + (K1p2 * K2 * Lp8) / 4032.;
      T[0][5][0] += (K1 * Lp2) / 2. - (J1p2 * Lp4) / 12. - (K1p2 * Lp4) / 12.;
      T[0][5][1] += (K1 * Lp3) / 6. - (J1p2 * Lp5) / 60. - (K1p2 * Lp5) / 60.;
      T[0][5][2] += (J1 * Lp2) / 2.;
      T[0][5][3] += (J1 * Lp3) / 6.;
      T[1][0][0] += (K1 * K2 * Lp3) / 4. + (J1p2 * K2 * Lp5) / 40. - (K1p2 * K2 * Lp5) / 40.;
      T[1][1][0] += (5 * K1 * K2 * Lp4) / 24. + (J1p2 * K2 * Lp6) / 72. - (K1p2 * K2 * Lp6) / 72.;
      T[1][1][1] += (K1 * K2 * Lp5) / 24. + (J1p2 * K2 * Lp7) / 504. - (K1p2 * K2 * Lp7) / 504.;
      T[1][2][0] += -(J1 * K2 * Lp3) / 6. - (J1 * K1 * K2 * Lp5) / 10.;
      T[1][2][1] += (J1 * K2 * Lp4) / 24. - (J1 * K1 * K2 * Lp6) / 36.;
      T[1][2][2] += (K1 * K2 * Lp3) / 12. - (J1p2 * K2 * Lp5) / 40. + (K1p2 * K2 * Lp5) / 40.;
      T[1][3][0] += -(J1 * K2 * Lp4) / 8. - (J1 * K1 * K2 * Lp6) / 36.;
      T[1][3][1] += -(J1 * K2 * Lp5) / 60. - (J1 * K1 * K2 * Lp7) / 126.;
      T[1][3][2] += (K1 * K2 * Lp4) / 8. - (J1p2 * K2 * Lp6) / 72. + (K1p2 * K2 * Lp6) / 72.;
      T[1][3][3] += (K1 * K2 * Lp5) / 40. - (J1p2 * K2 * Lp7) / 504. + (K1p2 * K2 * Lp7) / 504.;
      T[1][5][0] += K1 * L - (J1p2 * Lp3) / 3. - (K1p2 * Lp3) / 3.;
      T[1][5][1] += (K1 * Lp2) / 2. - (J1p2 * Lp4) / 12. - (K1p2 * Lp4) / 12.;
      T[1][5][2] += J1 * L;
      T[1][5][3] += (J1 * Lp2) / 2.;
      T[2][0][0] += -(J1 * K2 * Lp4) / 48. + (J1 * K1 * K2 * Lp6) / 120.;
      T[2][1][0] += -(J1 * K2 * Lp5) / 40. + (J1 * K1 * K2 * Lp7) / 252.;
      T[2][1][1] += -(J1 * K2 * Lp6) / 240. + (J1 * K1 * K2 * Lp8) / 2016.;
      T[2][2][0] += (K1 * K2 * Lp4) / 24. + (J1p2 * K2 * Lp6) / 120. - (K1p2 * K2 * Lp6) / 120.;
      T[2][2][1] += (K1 * K2 * Lp5) / 40. + (J1p2 * K2 * Lp7) / 504. - (K1p2 * K2 * Lp7) / 504.;
      T[2][2][2] += -(J1 * K2 * Lp4) / 16. - (J1 * K1 * K2 * Lp6) / 120.;
      T[2][3][0] += -(K1 * K2 * Lp5) / 120. + (J1p2 * K2 * Lp7) / 504. - (K1p2 * K2 * Lp7) / 504.;
      T[2][3][1] += (K1 * K2 * Lp6) / 360. + (J1p2 * K2 * Lp8) / 2016. - (K1p2 * K2 * Lp8) / 2016.;
      T[2][3][2] += -(J1 * K2 * Lp5) / 24. - (J1 * K1 * K2 * Lp7) / 252.;
      T[2][3][3] += -(J1 * K2 * Lp6) / 144. - (J1 * K1 * K2 * Lp8) / 2016.;
      T[2][5][0] += (J1 * Lp2) / 2.;
      T[2][5][1] += (J1 * Lp3) / 6.;
      T[2][5][2] += -(K1 * Lp2) / 2. - (J1p2 * Lp4) / 12. - (K1p2 * Lp4) / 12.;
      T[2][5][3] += -(K1 * Lp3) / 6. - (J1p2 * Lp5) / 60. - (K1p2 * Lp5) / 60.;
      T[3][0][0] += -(J1 * K2 * Lp3) / 12. + (J1 * K1 * K2 * Lp5) / 20.;
      T[3][1][0] += -(J1 * K2 * Lp4) / 8. + (J1 * K1 * K2 * Lp6) / 36.;
      T[3][1][1] += -(J1 * K2 * Lp5) / 40. + (J1 * K1 * K2 * Lp7) / 252.;
      T[3][2][0] += (K1 * K2 * Lp3) / 6. + (J1p2 * K2 * Lp5) / 20. - (K1p2 * K2 * Lp5) / 20.;
      T[3][2][1] += (K1 * K2 * Lp4) / 8. + (J1p2 * K2 * Lp6) / 72. - (K1p2 * K2 * Lp6) / 72.;
      T[3][2][2] += -(J1 * K2 * Lp3) / 4. - (J1 * K1 * K2 * Lp5) / 20.;
      T[3][3][0] += -(K1 * K2 * Lp4) / 24. + (J1p2 * K2 * Lp6) / 72. - (K1p2 * K2 * Lp6) / 72.;
      T[3][3][1] += (K1 * K2 * Lp5) / 60. + (J1p2 * K2 * Lp7) / 252. - (K1p2 * K2 * Lp7) / 252.;
      T[3][3][2] += (-5 * J1 * K2 * Lp4) / 24. - (J1 * K1 * K2 * Lp6) / 36.;
      T[3][3][3] += -(J1 * K2 * Lp5) / 24. - (J1 * K1 * K2 * Lp7) / 252.;
      T[3][5][0] += J1 * L;
      T[3][5][1] += (J1 * Lp2) / 2.;
      T[3][5][2] += -(K1 * L) - (J1p2 * Lp3) / 3. - (K1p2 * Lp3) / 3.;
      T[3][5][3] += -(K1 * Lp2) / 2. - (J1p2 * Lp4) / 12. - (K1p2 * Lp4) / 12.;

      if (M->order >= 3) {
        U[0][0][0][0] += -(J1p2 * K1 * Lp4) / 8. - (K1p3 * Lp4) / 8. + (J1p4 * Lp6) / 120. + (J1p2 * K1p2 * Lp6) / 30. + (K1p4 * Lp6) / 40. - (K1 * K2p2 * Lp6) / 240.;
        U[0][1][0][0] += (J1p2 * Lp3) / 6. + (K1p2 * Lp3) / 2. - (9 * J1p2 * K1 * Lp5) / 40. - (9 * K1p3 * Lp5) / 40. + (J1p4 * Lp7) / 126. + (2 * J1p2 * K1p2 * Lp7) / 63. + (K1p4 * Lp7) / 42. - (K1 * K2p2 * Lp7) / 336.;
        U[0][1][1][0] += (-3 * K1 * Lp2) / 4. + (5 * J1p2 * Lp4) / 24. + (K1p2 * Lp4) / 2. - (5 * J1p2 * K1 * Lp6) / 48. - (5 * K1p3 * Lp6) / 48. + (J1p4 * Lp8) / 384. + (J1p2 * K1p2 * Lp8) / 96. + (K1p4 * Lp8) / 128. - (K1 * K2p2 * Lp8) / 1152.;
        U[0][1][1][1] += -(K1 * Lp3) / 4. + (J1p2 * Lp5) / 24. + (K1p2 * Lp5) / 10. - (5 * J1p2 * K1 * Lp7) / 336. - (5 * K1p3 * Lp7) / 336. + (J1p4 * Lp9) / 3456. + (J1p2 * K1p2 * Lp9) / 864. + (K1p4 * Lp9) / 1152. - (K1 * K2p2 * Lp9) / 10368.;
        U[0][2][0][0] += -(J1p3 * Lp4) / 8. - (J1 * K1p2 * Lp4) / 8. + (J1p3 * K1 * Lp6) / 20. + (J1 * K1p3 * Lp6) / 20. - (J1 * K2p2 * Lp6) / 80.;
        U[0][2][1][0] += (2 * J1 * K1 * Lp3) / 3. - (J1p3 * Lp5) / 5. - (J1 * K1p2 * Lp5) / 5. + (2 * J1p3 * K1 * Lp7) / 63. + (2 * J1 * K1p3 * Lp7) / 63. - (J1 * K2p2 * Lp7) / 168.;
        U[0][2][1][1] += (-3 * J1 * Lp2) / 4. + (7 * J1 * K1 * Lp4) / 24. - (11 * J1p3 * Lp6) / 240. - (11 * J1 * K1p2 * Lp6) / 240. + (J1p3 * K1 * Lp8) / 192. + (J1 * K1p3 * Lp8) / 192. - (J1 * K2p2 * Lp8) / 1152.;
        U[0][2][2][0] += -(J1p2 * K1 * Lp4) / 8. - (K1p3 * Lp4) / 8. + (J1p4 * Lp6) / 24. + (J1p2 * K1p2 * Lp6) / 30. - (K1p4 * Lp6) / 120. + (K1 * K2p2 * Lp6) / 80.;
        U[0][2][2][1] += (J1p2 * Lp3) / 2. + (K1p2 * Lp3) / 6. - (J1p2 * K1 * Lp5) / 40. - (K1p3 * Lp5) / 40. + (J1p4 * Lp7) / 84. + (J1p2 * K1p2 * Lp7) / 126. - (K1p4 * Lp7) / 252. + (K1 * K2p2 * Lp7) / 336.;
        U[0][2][2][2] += -(J1p3 * Lp4) / 8. - (J1 * K1p2 * Lp4) / 8. - (J1p3 * K1 * Lp6) / 60. - (J1 * K1p3 * Lp6) / 60. + (J1 * K2p2 * Lp6) / 240.;
        U[0][3][0][0] += (J1 * K1 * Lp3) / 3. - (3 * J1p3 * Lp5) / 40. - (3 * J1 * K1p2 * Lp5) / 40. + (J1p3 * K1 * Lp7) / 63. + (J1 * K1p3 * Lp7) / 63. - (J1 * K2p2 * Lp7) / 336.;
        U[0][3][1][0] += -(J1 * Lp2) / 2. + (7 * J1 * K1 * Lp4) / 12. - (31 * J1p3 * Lp6) / 360. - (31 * J1 * K1p2 * Lp6) / 360. + (J1p3 * K1 * Lp8) / 96. + (J1 * K1p3 * Lp8) / 96. - (J1 * K2p2 * Lp8) / 576.;
        U[0][3][1][1] += (-5 * J1 * Lp3) / 12. + (7 * J1 * K1 * Lp5) / 40. - (19 * J1p3 * Lp7) / 1008. - (19 * J1 * K1p2 * Lp7) / 1008. + (J1p3 * K1 * Lp9) / 576. + (J1 * K1p3 * Lp9) / 576. - (J1 * K2p2 * Lp9) / 3456.;
        U[0][3][2][0] += (J1p2 * Lp3) / 3. - (K1p2 * Lp3) / 3. - (J1p2 * K1 * Lp5) / 10. - (K1p3 * Lp5) / 10. + (J1p4 * Lp7) / 36. + (J1p2 * K1p2 * Lp7) / 42. - (K1p4 * Lp7) / 252. + (K1 * K2p2 * Lp7) / 168.;
        U[0][3][2][1] += (K1 * Lp2) / 2. + (7 * J1p2 * Lp4) / 12. - (11 * J1p2 * K1 * Lp6) / 360. - (11 * K1p3 * Lp6) / 360. + (11 * J1p4 * Lp8) / 1344. + (J1p2 * K1p2 * Lp8) / 168. - (K1p4 * Lp8) / 448. + (K1 * K2p2 * Lp8) / 576.;
        U[0][3][2][2] += -(J1 * K1 * Lp3) / 3. - (7 * J1p3 * Lp5) / 40. - (7 * J1 * K1p2 * Lp5) / 40. - (J1p3 * K1 * Lp7) / 63. - (J1 * K1p3 * Lp7) / 63. + (J1 * K2p2 * Lp7) / 336.;
        U[0][3][3][0] += -(K1 * Lp2) / 4. + (5 * J1p2 * Lp4) / 24. - (K1p2 * Lp4) / 12. - (13 * J1p2 * K1 * Lp6) / 720. - (13 * K1p3 * Lp6) / 720. + (13 * J1p4 * Lp8) / 2688. + (J1p2 * K1p2 * Lp8) / 224. - (K1p4 * Lp8) / 2688. + (K1 * K2p2 * Lp8) / 1152.;
        U[0][3][3][1] += (K1 * Lp3) / 12. + (19 * J1p2 * Lp5) / 120. - (K1p2 * Lp5) / 60. - (J1p2 * K1 * Lp7) / 144. - (K1p3 * Lp7) / 144. + (5 * J1p4 * Lp9) / 3456. + (J1p2 * K1p2 * Lp9) / 864. - (K1p4 * Lp9) / 3456. + (K1 * K2p2 * Lp9) / 3456.;
        U[0][3][3][2] += -(J1 * Lp2) / 4. - (7 * J1 * K1 * Lp4) / 24. - (11 * J1p3 * Lp6) / 144. - (11 * J1 * K1p2 * Lp6) / 144. - (J1p3 * K1 * Lp8) / 192. - (J1 * K1p3 * Lp8) / 192. + (J1 * K2p2 * Lp8) / 1152.;
        U[0][3][3][3] += -(J1 * Lp3) / 12. - (7 * J1 * K1 * Lp5) / 120. - (11 * J1p3 * Lp7) / 1008. - (11 * J1 * K1p2 * Lp7) / 1008. - (J1p3 * K1 * Lp9) / 1728. - (J1 * K1p3 * Lp9) / 1728. + (J1 * K2p2 * Lp9) / 10368.;
        U[0][5][0][0] += -(K1 * K2 * Lp4) / 8. - (J1p2 * K2 * Lp6) / 80. + (K1p2 * K2 * Lp6) / 80.;
        U[0][5][1][0] += -(K1 * K2 * Lp5) / 12. - (J1p2 * K2 * Lp7) / 168. + (K1p2 * K2 * Lp7) / 168.;
        U[0][5][1][1] += -(K1 * K2 * Lp6) / 72. - (J1p2 * K2 * Lp8) / 1344. + (K1p2 * K2 * Lp8) / 1344.;
        U[0][5][2][0] += (J1 * K2 * Lp4) / 12. + (J1 * K1 * K2 * Lp6) / 20.;
        U[0][5][2][1] += -(J1 * K2 * Lp5) / 60. + (J1 * K1 * K2 * Lp7) / 84.;
        U[0][5][2][2] += -(K1 * K2 * Lp4) / 24. + (J1p2 * K2 * Lp6) / 80. - (K1p2 * K2 * Lp6) / 80.;
        U[0][5][3][0] += (J1 * K2 * Lp5) / 20. + (J1 * K1 * K2 * Lp7) / 84.;
        U[0][5][3][1] += (J1 * K2 * Lp6) / 180. + (J1 * K1 * K2 * Lp8) / 336.;
        U[0][5][3][2] += -(K1 * K2 * Lp5) / 20. + (J1p2 * K2 * Lp7) / 168. - (K1p2 * K2 * Lp7) / 168.;
        U[0][5][3][3] += -(K1 * K2 * Lp6) / 120. + (J1p2 * K2 * Lp8) / 1344. - (K1p2 * K2 * Lp8) / 1344.;
        U[0][5][5][0] += -(K1 * Lp2) / 2. + (J1p2 * Lp4) / 8. + (K1p2 * Lp4) / 8.;
        U[0][5][5][1] += -(K1 * Lp3) / 6. + (J1p2 * Lp5) / 40. + (K1p2 * Lp5) / 40.;
        U[0][5][5][2] += -(J1 * Lp2) / 2.;
        U[0][5][5][3] += -(J1 * Lp3) / 6.;
        U[1][0][0][0] += -(J1p2 * K1 * Lp3) / 2. - (K1p3 * Lp3) / 2. + (J1p4 * Lp5) / 20. + (J1p2 * K1p2 * Lp5) / 5. + (3 * K1p4 * Lp5) / 20. - (K1 * K2p2 * Lp5) / 40.;
        U[1][1][0][0] += (J1p2 * Lp2) / 2. + (3 * K1p2 * Lp2) / 2. - (9 * J1p2 * K1 * Lp4) / 8. - (9 * K1p3 * Lp4) / 8. + (J1p4 * Lp6) / 18. + (2 * J1p2 * K1p2 * Lp6) / 9. + (K1p4 * Lp6) / 6. - (K1 * K2p2 * Lp6) / 48.;
        U[1][1][1][0] += (-3 * K1 * L) / 2. + (5 * J1p2 * Lp3) / 6. + 2 * K1p2 * Lp3 - (5 * J1p2 * K1 * Lp5) / 8. - (5 * K1p3 * Lp5) / 8. + (J1p4 * Lp7) / 48. + (J1p2 * K1p2 * Lp7) / 12. + (K1p4 * Lp7) / 16. - (K1 * K2p2 * Lp7) / 144.;
        U[1][1][1][1] += (-3 * K1 * Lp2) / 4. + (5 * J1p2 * Lp4) / 24. + (K1p2 * Lp4) / 2. - (5 * J1p2 * K1 * Lp6) / 48. - (5 * K1p3 * Lp6) / 48. + (J1p4 * Lp8) / 384. + (J1p2 * K1p2 * Lp8) / 96. + (K1p4 * Lp8) / 128. - (K1 * K2p2 * Lp8) / 1152.;
        U[1][2][0][0] += -(J1p3 * Lp3) / 2. - (J1 * K1p2 * Lp3) / 2. + (3 * J1p3 * K1 * Lp5) / 10. + (3 * J1 * K1p3 * Lp5) / 10. - (3 * J1 * K2p2 * Lp5) / 40.;
        U[1][2][1][0] += 2 * J1 * K1 * Lp2 - J1p3 * Lp4 - J1 * K1p2 * Lp4 + (2 * J1p3 * K1 * Lp6) / 9. + (2 * J1 * K1p3 * Lp6) / 9. - (J1 * K2p2 * Lp6) / 24.;
        U[1][2][1][1] += (-3 * J1 * L) / 2. + (7 * J1 * K1 * Lp3) / 6. - (11 * J1p3 * Lp5) / 40. - (11 * J1 * K1p2 * Lp5) / 40. + (J1p3 * K1 * Lp7) / 24. + (J1 * K1p3 * Lp7) / 24. - (J1 * K2p2 * Lp7) / 144.;
        U[1][2][2][0] += -(J1p2 * K1 * Lp3) / 2. - (K1p3 * Lp3) / 2. + (J1p4 * Lp5) / 4. + (J1p2 * K1p2 * Lp5) / 5. - (K1p4 * Lp5) / 20. + (3 * K1 * K2p2 * Lp5) / 40.;
        U[1][2][2][1] += (3 * J1p2 * Lp2) / 2. + (K1p2 * Lp2) / 2. - (J1p2 * K1 * Lp4) / 8. - (K1p3 * Lp4) / 8. + (J1p4 * Lp6) / 12. + (J1p2 * K1p2 * Lp6) / 18. - (K1p4 * Lp6) / 36. + (K1 * K2p2 * Lp6) / 48.;
        U[1][2][2][2] += -(J1p3 * Lp3) / 2. - (J1 * K1p2 * Lp3) / 2. - (J1p3 * K1 * Lp5) / 10. - (J1 * K1p3 * Lp5) / 10. + (J1 * K2p2 * Lp5) / 40.;
        U[1][3][0][0] += J1 * K1 * Lp2 - (3 * J1p3 * Lp4) / 8. - (3 * J1 * K1p2 * Lp4) / 8. + (J1p3 * K1 * Lp6) / 9. + (J1 * K1p3 * Lp6) / 9. - (J1 * K2p2 * Lp6) / 48.;
        U[1][3][1][0] += -(J1 * L) + (7 * J1 * K1 * Lp3) / 3. - (31 * J1p3 * Lp5) / 60. - (31 * J1 * K1p2 * Lp5) / 60. + (J1p3 * K1 * Lp7) / 12. + (J1 * K1p3 * Lp7) / 12. - (J1 * K2p2 * Lp7) / 72.;
        U[1][3][1][1] += (-5 * J1 * Lp2) / 4. + (7 * J1 * K1 * Lp4) / 8. - (19 * J1p3 * Lp6) / 144. - (19 * J1 * K1p2 * Lp6) / 144. + (J1p3 * K1 * Lp8) / 64. + (J1 * K1p3 * Lp8) / 64. - (J1 * K2p2 * Lp8) / 384.;
        U[1][3][2][0] += J1p2 * Lp2 - K1p2 * Lp2 - (J1p2 * K1 * Lp4) / 2. - (K1p3 * Lp4) / 2. + (7 * J1p4 * Lp6) / 36. + (J1p2 * K1p2 * Lp6) / 6. - (K1p4 * Lp6) / 36. + (K1 * K2p2 * Lp6) / 24.;
        U[1][3][2][1] += K1 * L + (7 * J1p2 * Lp3) / 3. - (11 * J1p2 * K1 * Lp5) / 60. - (11 * K1p3 * Lp5) / 60. + (11 * J1p4 * Lp7) / 168. + (J1p2 * K1p2 * Lp7) / 21. - (K1p4 * Lp7) / 56. + (K1 * K2p2 * Lp7) / 72.;
        U[1][3][2][2] += -(J1 * K1 * Lp2) - (7 * J1p3 * Lp4) / 8. - (7 * J1 * K1p2 * Lp4) / 8. - (J1p3 * K1 * Lp6) / 9. - (J1 * K1p3 * Lp6) / 9. + (J1 * K2p2 * Lp6) / 48.;
        U[1][3][3][0] += -(K1 * L) / 2. + (5 * J1p2 * Lp3) / 6. - (K1p2 * Lp3) / 3. - (13 * J1p2 * K1 * Lp5) / 120. - (13 * K1p3 * Lp5) / 120. + (13 * J1p4 * Lp7) / 336. + (J1p2 * K1p2 * Lp7) / 28. - (K1p4 * Lp7) / 336. + (K1 * K2p2 * Lp7) / 144.;
        U[1][3][3][1] += (K1 * Lp2) / 4. + (19 * J1p2 * Lp4) / 24. - (K1p2 * Lp4) / 12. - (7 * J1p2 * K1 * Lp6) / 144. - (7 * K1p3 * Lp6) / 144. + (5 * J1p4 * Lp8) / 384. + (J1p2 * K1p2 * Lp8) / 96. - (K1p4 * Lp8) / 384. + (K1 * K2p2 * Lp8) / 384.;
        U[1][3][3][2] += -(J1 * L) / 2. - (7 * J1 * K1 * Lp3) / 6. - (11 * J1p3 * Lp5) / 24. - (11 * J1 * K1p2 * Lp5) / 24. - (J1p3 * K1 * Lp7) / 24. - (J1 * K1p3 * Lp7) / 24. + (J1 * K2p2 * Lp7) / 144.;
        U[1][3][3][3] += -(J1 * Lp2) / 4. - (7 * J1 * K1 * Lp4) / 24. - (11 * J1p3 * Lp6) / 144. - (11 * J1 * K1p2 * Lp6) / 144. - (J1p3 * K1 * Lp8) / 192. - (J1 * K1p3 * Lp8) / 192. + (J1 * K2p2 * Lp8) / 1152.;
        U[1][5][0][0] += -(K1 * K2 * Lp3) / 2. - (3 * J1p2 * K2 * Lp5) / 40. + (3 * K1p2 * K2 * Lp5) / 40.;
        U[1][5][1][0] += (-5 * K1 * K2 * Lp4) / 12. - (J1p2 * K2 * Lp6) / 24. + (K1p2 * K2 * Lp6) / 24.;
        U[1][5][1][1] += -(K1 * K2 * Lp5) / 12. - (J1p2 * K2 * Lp7) / 168. + (K1p2 * K2 * Lp7) / 168.;
        U[1][5][2][0] += (J1 * K2 * Lp3) / 3. + (3 * J1 * K1 * K2 * Lp5) / 10.;
        U[1][5][2][1] += -(J1 * K2 * Lp4) / 12. + (J1 * K1 * K2 * Lp6) / 12.;
        U[1][5][2][2] += -(K1 * K2 * Lp3) / 6. + (3 * J1p2 * K2 * Lp5) / 40. - (3 * K1p2 * K2 * Lp5) / 40.;
        U[1][5][3][0] += (J1 * K2 * Lp4) / 4. + (J1 * K1 * K2 * Lp6) / 12.;
        U[1][5][3][1] += (J1 * K2 * Lp5) / 30. + (J1 * K1 * K2 * Lp7) / 42.;
        U[1][5][3][2] += -(K1 * K2 * Lp4) / 4. + (J1p2 * K2 * Lp6) / 24. - (K1p2 * K2 * Lp6) / 24.;
        U[1][5][3][3] += -(K1 * K2 * Lp5) / 20. + (J1p2 * K2 * Lp7) / 168. - (K1p2 * K2 * Lp7) / 168.;
        U[1][5][5][0] += -(K1 * L) + (J1p2 * Lp3) / 2. + (K1p2 * Lp3) / 2.;
        U[1][5][5][1] += -(K1 * Lp2) / 2. + (J1p2 * Lp4) / 8. + (K1p2 * Lp4) / 8.;
        U[1][5][5][2] += -(J1 * L);
        U[1][5][5][3] += -(J1 * Lp2) / 2.;
        U[2][0][0][0] += -(J1p3 * Lp4) / 8. - (J1 * K1p2 * Lp4) / 8. + (J1p3 * K1 * Lp6) / 60. + (J1 * K1p3 * Lp6) / 60. + (J1 * K2p2 * Lp6) / 240.;
        U[2][1][0][0] += (J1 * K1 * Lp3) / 3. - (7 * J1p3 * Lp5) / 40. - (7 * J1 * K1p2 * Lp5) / 40. + (J1p3 * K1 * Lp7) / 63. + (J1 * K1p3 * Lp7) / 63. + (J1 * K2p2 * Lp7) / 336.;
        U[2][1][1][0] += -(J1 * Lp2) / 4. + (7 * J1 * K1 * Lp4) / 24. - (11 * J1p3 * Lp6) / 144. - (11 * J1 * K1p2 * Lp6) / 144. + (J1p3 * K1 * Lp8) / 192. + (J1 * K1p3 * Lp8) / 192. + (J1 * K2p2 * Lp8) / 1152.;
        U[2][1][1][1] += -(J1 * Lp3) / 12. + (7 * J1 * K1 * Lp5) / 120. - (11 * J1p3 * Lp7) / 1008. - (11 * J1 * K1p2 * Lp7) / 1008. + (J1p3 * K1 * Lp9) / 1728. + (J1 * K1p3 * Lp9) / 1728. + (J1 * K2p2 * Lp9) / 10368.;
        U[2][2][0][0] += (J1p2 * K1 * Lp4) / 8. + (K1p3 * Lp4) / 8. + (J1p4 * Lp6) / 24. + (J1p2 * K1p2 * Lp6) / 30. - (K1p4 * Lp6) / 120. - (K1 * K2p2 * Lp6) / 80.;
        U[2][2][1][0] += (J1p2 * Lp3) / 3. - (K1p2 * Lp3) / 3. + (J1p2 * K1 * Lp5) / 10. + (K1p3 * Lp5) / 10. + (J1p4 * Lp7) / 36. + (J1p2 * K1p2 * Lp7) / 42. - (K1p4 * Lp7) / 252. - (K1 * K2p2 * Lp7) / 168.;
        U[2][2][1][1] += (K1 * Lp2) / 4. + (5 * J1p2 * Lp4) / 24. - (K1p2 * Lp4) / 12. + (13 * J1p2 * K1 * Lp6) / 720. + (13 * K1p3 * Lp6) / 720. + (13 * J1p4 * Lp8) / 2688. + (J1p2 * K1p2 * Lp8) / 224. - (K1p4 * Lp8) / 2688. - (K1 * K2p2 * Lp8) / 1152.;
        U[2][2][2][0] += -(J1p3 * Lp4) / 8. - (J1 * K1p2 * Lp4) / 8. - (J1p3 * K1 * Lp6) / 20. - (J1 * K1p3 * Lp6) / 20. - (J1 * K2p2 * Lp6) / 80.;
        U[2][2][2][1] += -(J1 * K1 * Lp3) / 3. - (3 * J1p3 * Lp5) / 40. - (3 * J1 * K1p2 * Lp5) / 40. - (J1p3 * K1 * Lp7) / 63. - (J1 * K1p3 * Lp7) / 63. - (J1 * K2p2 * Lp7) / 336.;
        U[2][2][2][2] += (J1p2 * K1 * Lp4) / 8. + (K1p3 * Lp4) / 8. + (J1p4 * Lp6) / 120. + (J1p2 * K1p2 * Lp6) / 30. + (K1p4 * Lp6) / 40. + (K1 * K2p2 * Lp6) / 240.;
        U[2][3][0][0] += (J1p2 * Lp3) / 2. + (K1p2 * Lp3) / 6. + (J1p2 * K1 * Lp5) / 40. + (K1p3 * Lp5) / 40. + (J1p4 * Lp7) / 84. + (J1p2 * K1p2 * Lp7) / 126. - (K1p4 * Lp7) / 252. - (K1 * K2p2 * Lp7) / 336.;
        U[2][3][1][0] += -(K1 * Lp2) / 2. + (7 * J1p2 * Lp4) / 12. + (11 * J1p2 * K1 * Lp6) / 360. + (11 * K1p3 * Lp6) / 360. + (11 * J1p4 * Lp8) / 1344. + (J1p2 * K1p2 * Lp8) / 168. - (K1p4 * Lp8) / 448. - (K1 * K2p2 * Lp8) / 576.;
        U[2][3][1][1] += -(K1 * Lp3) / 12. + (19 * J1p2 * Lp5) / 120. - (K1p2 * Lp5) / 60. + (J1p2 * K1 * Lp7) / 144. + (K1p3 * Lp7) / 144. + (5 * J1p4 * Lp9) / 3456. + (J1p2 * K1p2 * Lp9) / 864. - (K1p4 * Lp9) / 3456. - (K1 * K2p2 * Lp9) / 3456.;
        U[2][3][2][0] += (-2 * J1 * K1 * Lp3) / 3. - (J1p3 * Lp5) / 5. - (J1 * K1p2 * Lp5) / 5. - (2 * J1p3 * K1 * Lp7) / 63. - (2 * J1 * K1p3 * Lp7) / 63. - (J1 * K2p2 * Lp7) / 168.;
        U[2][3][2][1] += -(J1 * Lp2) / 2. - (7 * J1 * K1 * Lp4) / 12. - (31 * J1p3 * Lp6) / 360. - (31 * J1 * K1p2 * Lp6) / 360. - (J1p3 * K1 * Lp8) / 96. - (J1 * K1p3 * Lp8) / 96. - (J1 * K2p2 * Lp8) / 576.;
        U[2][3][2][2] += (J1p2 * Lp3) / 6. + (K1p2 * Lp3) / 2. + (9 * J1p2 * K1 * Lp5) / 40. + (9 * K1p3 * Lp5) / 40. + (J1p4 * Lp7) / 126. + (2 * J1p2 * K1p2 * Lp7) / 63. + (K1p4 * Lp7) / 42. + (K1 * K2p2 * Lp7) / 336.;
        U[2][3][3][0] += (-3 * J1 * Lp2) / 4. - (7 * J1 * K1 * Lp4) / 24. - (11 * J1p3 * Lp6) / 240. - (11 * J1 * K1p2 * Lp6) / 240. - (J1p3 * K1 * Lp8) / 192. - (J1 * K1p3 * Lp8) / 192. - (J1 * K2p2 * Lp8) / 1152.;
        U[2][3][3][1] += (-5 * J1 * Lp3) / 12. - (7 * J1 * K1 * Lp5) / 40. - (19 * J1p3 * Lp7) / 1008. - (19 * J1 * K1p2 * Lp7) / 1008. - (J1p3 * K1 * Lp9) / 576. - (J1 * K1p3 * Lp9) / 576. - (J1 * K2p2 * Lp9) / 3456.;
        U[2][3][3][2] += (3 * K1 * Lp2) / 4. + (5 * J1p2 * Lp4) / 24. + (K1p2 * Lp4) / 2. + (5 * J1p2 * K1 * Lp6) / 48. + (5 * K1p3 * Lp6) / 48. + (J1p4 * Lp8) / 384. + (J1p2 * K1p2 * Lp8) / 96. + (K1p4 * Lp8) / 128. + (K1 * K2p2 * Lp8) / 1152.;
        U[2][3][3][3] += (K1 * Lp3) / 4. + (J1p2 * Lp5) / 24. + (K1p2 * Lp5) / 10. + (5 * J1p2 * K1 * Lp7) / 336. + (5 * K1p3 * Lp7) / 336. + (J1p4 * Lp9) / 3456. + (J1p2 * K1p2 * Lp9) / 864. + (K1p4 * Lp9) / 1152. + (K1 * K2p2 * Lp9) / 10368.;
        U[2][5][0][0] += (J1 * K2 * Lp4) / 24. - (J1 * K1 * K2 * Lp6) / 40.;
        U[2][5][1][0] += (J1 * K2 * Lp5) / 20. - (J1 * K1 * K2 * Lp7) / 84.;
        U[2][5][1][1] += (J1 * K2 * Lp6) / 120. - (J1 * K1 * K2 * Lp8) / 672.;
        U[2][5][2][0] += -(K1 * K2 * Lp4) / 12. - (J1p2 * K2 * Lp6) / 40. + (K1p2 * K2 * Lp6) / 40.;
        U[2][5][2][1] += -(K1 * K2 * Lp5) / 20. - (J1p2 * K2 * Lp7) / 168. + (K1p2 * K2 * Lp7) / 168.;
        U[2][5][2][2] += (J1 * K2 * Lp4) / 8. + (J1 * K1 * K2 * Lp6) / 40.;
        U[2][5][3][0] += (K1 * K2 * Lp5) / 60. - (J1p2 * K2 * Lp7) / 168. + (K1p2 * K2 * Lp7) / 168.;
        U[2][5][3][1] += -(K1 * K2 * Lp6) / 180. - (J1p2 * K2 * Lp8) / 672. + (K1p2 * K2 * Lp8) / 672.;
        U[2][5][3][2] += (J1 * K2 * Lp5) / 12. + (J1 * K1 * K2 * Lp7) / 84.;
        U[2][5][3][3] += (J1 * K2 * Lp6) / 72. + (J1 * K1 * K2 * Lp8) / 672.;
        U[2][5][5][0] += -(J1 * Lp2) / 2.;
        U[2][5][5][1] += -(J1 * Lp3) / 6.;
        U[2][5][5][2] += (K1 * Lp2) / 2. + (J1p2 * Lp4) / 8. + (K1p2 * Lp4) / 8.;
        U[2][5][5][3] += (K1 * Lp3) / 6. + (J1p2 * Lp5) / 40. + (K1p2 * Lp5) / 40.;
        U[3][0][0][0] += -(J1p3 * Lp3) / 2. - (J1 * K1p2 * Lp3) / 2. + (J1p3 * K1 * Lp5) / 10. + (J1 * K1p3 * Lp5) / 10. + (J1 * K2p2 * Lp5) / 40.;
        U[3][1][0][0] += J1 * K1 * Lp2 - (7 * J1p3 * Lp4) / 8. - (7 * J1 * K1p2 * Lp4) / 8. + (J1p3 * K1 * Lp6) / 9. + (J1 * K1p3 * Lp6) / 9. + (J1 * K2p2 * Lp6) / 48.;
        U[3][1][1][0] += -(J1 * L) / 2. + (7 * J1 * K1 * Lp3) / 6. - (11 * J1p3 * Lp5) / 24. - (11 * J1 * K1p2 * Lp5) / 24. + (J1p3 * K1 * Lp7) / 24. + (J1 * K1p3 * Lp7) / 24. + (J1 * K2p2 * Lp7) / 144.;
        U[3][1][1][1] += -(J1 * Lp2) / 4. + (7 * J1 * K1 * Lp4) / 24. - (11 * J1p3 * Lp6) / 144. - (11 * J1 * K1p2 * Lp6) / 144. + (J1p3 * K1 * Lp8) / 192. + (J1 * K1p3 * Lp8) / 192. + (J1 * K2p2 * Lp8) / 1152.;
        U[3][2][0][0] += (J1p2 * K1 * Lp3) / 2. + (K1p3 * Lp3) / 2. + (J1p4 * Lp5) / 4. + (J1p2 * K1p2 * Lp5) / 5. - (K1p4 * Lp5) / 20. - (3 * K1 * K2p2 * Lp5) / 40.;
        U[3][2][1][0] += J1p2 * Lp2 - K1p2 * Lp2 + (J1p2 * K1 * Lp4) / 2. + (K1p3 * Lp4) / 2. + (7 * J1p4 * Lp6) / 36. + (J1p2 * K1p2 * Lp6) / 6. - (K1p4 * Lp6) / 36. - (K1 * K2p2 * Lp6) / 24.;
        U[3][2][1][1] += (K1 * L) / 2. + (5 * J1p2 * Lp3) / 6. - (K1p2 * Lp3) / 3. + (13 * J1p2 * K1 * Lp5) / 120. + (13 * K1p3 * Lp5) / 120. + (13 * J1p4 * Lp7) / 336. + (J1p2 * K1p2 * Lp7) / 28. - (K1p4 * Lp7) / 336. - (K1 * K2p2 * Lp7) / 144.;
        U[3][2][2][0] += -(J1p3 * Lp3) / 2. - (J1 * K1p2 * Lp3) / 2. - (3 * J1p3 * K1 * Lp5) / 10. - (3 * J1 * K1p3 * Lp5) / 10. - (3 * J1 * K2p2 * Lp5) / 40.;
        U[3][2][2][1] += -(J1 * K1 * Lp2) - (3 * J1p3 * Lp4) / 8. - (3 * J1 * K1p2 * Lp4) / 8. - (J1p3 * K1 * Lp6) / 9. - (J1 * K1p3 * Lp6) / 9. - (J1 * K2p2 * Lp6) / 48.;
        U[3][2][2][2] += (J1p2 * K1 * Lp3) / 2. + (K1p3 * Lp3) / 2. + (J1p4 * Lp5) / 20. + (J1p2 * K1p2 * Lp5) / 5. + (3 * K1p4 * Lp5) / 20. + (K1 * K2p2 * Lp5) / 40.;
        U[3][3][0][0] += (3 * J1p2 * Lp2) / 2. + (K1p2 * Lp2) / 2. + (J1p2 * K1 * Lp4) / 8. + (K1p3 * Lp4) / 8. + (J1p4 * Lp6) / 12. + (J1p2 * K1p2 * Lp6) / 18. - (K1p4 * Lp6) / 36. - (K1 * K2p2 * Lp6) / 48.;
        U[3][3][1][0] += -(K1 * L) + (7 * J1p2 * Lp3) / 3. + (11 * J1p2 * K1 * Lp5) / 60. + (11 * K1p3 * Lp5) / 60. + (11 * J1p4 * Lp7) / 168. + (J1p2 * K1p2 * Lp7) / 21. - (K1p4 * Lp7) / 56. - (K1 * K2p2 * Lp7) / 72.;
        U[3][3][1][1] += -(K1 * Lp2) / 4. + (19 * J1p2 * Lp4) / 24. - (K1p2 * Lp4) / 12. + (7 * J1p2 * K1 * Lp6) / 144. + (7 * K1p3 * Lp6) / 144. + (5 * J1p4 * Lp8) / 384. + (J1p2 * K1p2 * Lp8) / 96. - (K1p4 * Lp8) / 384. - (K1 * K2p2 * Lp8) / 384.;
        U[3][3][2][0] += -2 * J1 * K1 * Lp2 - J1p3 * Lp4 - J1 * K1p2 * Lp4 - (2 * J1p3 * K1 * Lp6) / 9. - (2 * J1 * K1p3 * Lp6) / 9. - (J1 * K2p2 * Lp6) / 24.;
        U[3][3][2][1] += -(J1 * L) - (7 * J1 * K1 * Lp3) / 3. - (31 * J1p3 * Lp5) / 60. - (31 * J1 * K1p2 * Lp5) / 60. - (J1p3 * K1 * Lp7) / 12. - (J1 * K1p3 * Lp7) / 12. - (J1 * K2p2 * Lp7) / 72.;
        U[3][3][2][2] += (J1p2 * Lp2) / 2. + (3 * K1p2 * Lp2) / 2. + (9 * J1p2 * K1 * Lp4) / 8. + (9 * K1p3 * Lp4) / 8. + (J1p4 * Lp6) / 18. + (2 * J1p2 * K1p2 * Lp6) / 9. + (K1p4 * Lp6) / 6. + (K1 * K2p2 * Lp6) / 48.;
        U[3][3][3][0] += (-3 * J1 * L) / 2. - (7 * J1 * K1 * Lp3) / 6. - (11 * J1p3 * Lp5) / 40. - (11 * J1 * K1p2 * Lp5) / 40. - (J1p3 * K1 * Lp7) / 24. - (J1 * K1p3 * Lp7) / 24. - (J1 * K2p2 * Lp7) / 144.;
        U[3][3][3][1] += (-5 * J1 * Lp2) / 4. - (7 * J1 * K1 * Lp4) / 8. - (19 * J1p3 * Lp6) / 144. - (19 * J1 * K1p2 * Lp6) / 144. - (J1p3 * K1 * Lp8) / 64. - (J1 * K1p3 * Lp8) / 64. - (J1 * K2p2 * Lp8) / 384.;
        U[3][3][3][2] += (3 * K1 * L) / 2. + (5 * J1p2 * Lp3) / 6. + 2 * K1p2 * Lp3 + (5 * J1p2 * K1 * Lp5) / 8. + (5 * K1p3 * Lp5) / 8. + (J1p4 * Lp7) / 48. + (J1p2 * K1p2 * Lp7) / 12. + (K1p4 * Lp7) / 16. + (K1 * K2p2 * Lp7) / 144.;
        U[3][3][3][3] += (3 * K1 * Lp2) / 4. + (5 * J1p2 * Lp4) / 24. + (K1p2 * Lp4) / 2. + (5 * J1p2 * K1 * Lp6) / 48. + (5 * K1p3 * Lp6) / 48. + (J1p4 * Lp8) / 384. + (J1p2 * K1p2 * Lp8) / 96. + (K1p4 * Lp8) / 128. + (K1 * K2p2 * Lp8) / 1152.;
        U[3][5][0][0] += (J1 * K2 * Lp3) / 6. - (3 * J1 * K1 * K2 * Lp5) / 20.;
        U[3][5][1][0] += (J1 * K2 * Lp4) / 4. - (J1 * K1 * K2 * Lp6) / 12.;
        U[3][5][1][1] += (J1 * K2 * Lp5) / 20. - (J1 * K1 * K2 * Lp7) / 84.;
        U[3][5][2][0] += -(K1 * K2 * Lp3) / 3. - (3 * J1p2 * K2 * Lp5) / 20. + (3 * K1p2 * K2 * Lp5) / 20.;
        U[3][5][2][1] += -(K1 * K2 * Lp4) / 4. - (J1p2 * K2 * Lp6) / 24. + (K1p2 * K2 * Lp6) / 24.;
        U[3][5][2][2] += (J1 * K2 * Lp3) / 2. + (3 * J1 * K1 * K2 * Lp5) / 20.;
        U[3][5][3][0] += (K1 * K2 * Lp4) / 12. - (J1p2 * K2 * Lp6) / 24. + (K1p2 * K2 * Lp6) / 24.;
        U[3][5][3][1] += -(K1 * K2 * Lp5) / 30. - (J1p2 * K2 * Lp7) / 84. + (K1p2 * K2 * Lp7) / 84.;
        U[3][5][3][2] += (5 * J1 * K2 * Lp4) / 12. + (J1 * K1 * K2 * Lp6) / 12.;
        U[3][5][3][3] += (J1 * K2 * Lp5) / 12. + (J1 * K1 * K2 * Lp7) / 84.;
        U[3][5][5][0] += -(J1 * L);
        U[3][5][5][1] += -(J1 * Lp2) / 2.;
        U[3][5][5][2] += K1 * L + (J1p2 * Lp3) / 2. + (K1p2 * Lp3) / 2.;
        U[3][5][5][3] += (K1 * Lp2) / 2. + (J1p2 * Lp4) / 8. + (K1p2 * Lp4) / 8.;
      }
    }
  }

  if (Medge1 && Medge2) {
    VMATRIX *M1, *M2, *Md;
    M1 = tmalloc(sizeof(*M1));
    initialize_matrices(M1, M->order);
    M2 = tmalloc(sizeof(*M2));
    initialize_matrices(M2, M->order);

    /* drift back to fringe starting point */
    Md = drift_matrix(-lf / 2, M->order);

    concat_matrices(M1, Medge1, Md, 0);
    concat_matrices(M2, M, M1, 0);
    concat_matrices(M1, Medge2, M2, 0);
    concat_matrices(M, Md, M1, 0);

    free_matrices(Md);
    tfree(Md);
    Md = NULL;
    free_matrices(M1);
    tfree(M1);
    M1 = NULL;
    free_matrices(M2);
    tfree(M2);
    M2 = NULL;
    free_matrices(Medge1);
    tfree(Medge1);
    Medge1 = NULL;
    free_matrices(Medge2);
    tfree(Medge2);
    Medge2 = NULL;
  }

  if (xkick || ykick) {
    /* put identical kicks at the entrance and exit */
    Mtot = tmalloc(sizeof(*Mtot));
    initialize_matrices(Mtot, M->order);
    Mfringe = hvcorrector_matrix(0, xkick / 2, ykick / 2, 0.0, 0.0, 1.0, 1.0, 0, M->order);
    concat_matrices(Mtot, Mfringe, M, 0);
    concat_matrices(M, Mtot, Mfringe, 0);
    free_matrices(Mfringe);
    tfree(Mfringe);
    Mfringe = NULL;
    free_matrices(Mtot);
    tfree(Mtot);
    Mtot = NULL;
  }

  return (M);
}

VMATRIX *octupole_matrix(double K3, double length, long maximum_order, double fse) {
  VMATRIX *M;
  double *C, **R, ***T, ****U;

  K3 *= (1 + fse);

  M = tmalloc(sizeof(*M));
  initialize_matrices(M, M->order = MIN(3, maximum_order));
  R = M->R;
  C = M->C;

  R[0][0] = R[1][1] = R[2][2] = R[3][3] = R[4][4] = R[5][5] = 1;
  C[4] = R[0][1] = R[2][3] = length;
  if (M->order >= 2) {
    /* path length terms--same as for drift */
    T = M->T;
    T[4][1][1] = T[4][3][3] = length / 2;
  }
  if (M->order >= 3) {
    double L, L2, L3, L4, L5;
    U = M->Q;
    L = length;
    L2 = L * L;
    L3 = L2 * L;
    L4 = L3 * L;
    L5 = L4 * L;

    U[0][0][0][0] = -(K3 * L2) / 12.;
    U[0][1][0][0] = -(K3 * L3) / 12.;
    U[0][1][1][0] = -(K3 * L4) / 24.;
    U[0][1][1][1] = -(K3 * L5) / 120.;
    U[0][2][2][0] = (K3 * L2) / 4.;
    U[0][2][2][1] = (K3 * L3) / 12.;
    U[0][3][2][0] = (K3 * L3) / 6.;
    U[0][3][2][1] = (K3 * L4) / 12.;
    U[0][3][3][0] = (K3 * L4) / 24.;
    U[0][3][3][1] = (K3 * L5) / 40.;
    U[1][0][0][0] = -(K3 * L) / 6.;
    U[1][1][0][0] = -(K3 * L2) / 4.;
    U[1][1][1][0] = -(K3 * L3) / 6.;
    U[1][1][1][1] = -(K3 * L4) / 24.;
    U[1][2][2][0] = (K3 * L) / 2.;
    U[1][2][2][1] = (K3 * L2) / 4.;
    U[1][3][2][0] = (K3 * L2) / 2.;
    U[1][3][2][1] = (K3 * L3) / 3.;
    U[1][3][3][0] = (K3 * L3) / 6.;
    U[1][3][3][1] = (K3 * L4) / 8.;
    U[2][2][0][0] = (K3 * L2) / 4.;
    U[2][2][1][0] = (K3 * L3) / 6.;
    U[2][2][1][1] = (K3 * L4) / 24.;
    U[2][2][2][2] = -(K3 * L2) / 12.;
    U[2][3][0][0] = (K3 * L3) / 12.;
    U[2][3][1][0] = (K3 * L4) / 12.;
    U[2][3][1][1] = (K3 * L5) / 40.;
    U[2][3][2][2] = -(K3 * L3) / 12.;
    U[2][3][3][2] = -(K3 * L4) / 24.;
    U[2][3][3][3] = -(K3 * L5) / 120.;
    U[3][2][0][0] = (K3 * L) / 2.;
    U[3][2][1][0] = (K3 * L2) / 2.;
    U[3][2][1][1] = (K3 * L3) / 6.;
    U[3][2][2][2] = -(K3 * L) / 6.;
    U[3][3][0][0] = (K3 * L2) / 4.;
    U[3][3][1][0] = (K3 * L3) / 3.;
    U[3][3][1][1] = (K3 * L4) / 8.;
    U[3][3][2][2] = -(K3 * L2) / 4.;
    U[3][3][3][2] = -(K3 * L3) / 6.;
    U[3][3][3][3] = -(K3 * L4) / 24.;
  }

  return (M);
}

VMATRIX *solenoid_matrix(double length, double ks, long max_order) {
  VMATRIX *M;
  double *C, **R, ***T;
  double cos_ksl;
  double sin_ksl;
  double ksl;
  double CS2;
  double S2;
  double temp;

  log_entry("solenoid_matrix");

  if (ks == 0 || length == 0)
    return (drift_matrix(length, max_order));

  /* defined Ks = -B/(B.rho) as in MAD, which is different from TRANSPORT Ks = -B/(2*B.rho) */
  ks /= 2;
  ksl = ks * length;

  M = tmalloc(sizeof(*M));
  M->order = MIN(max_order, 2);
  initialize_matrices(M, M->order);
  R = M->R;
  C = M->C;
  T = M->T;

  cos_ksl = cos(ksl);
  sin_ksl = sin(ksl);
  C[4] = length;

  R[0][0] = R[1][1] = R[2][2] = R[3][3] = sqr(cos_ksl);
  R[2][0] = R[3][1] = -(R[0][2] = R[1][3] = CS2 = sin_ksl * cos_ksl);
  R[4][4] = R[5][5] = 1;

  R[0][1] = R[2][3] = CS2 / ks;
  R[1][0] = R[3][2] = -ks * CS2;

  S2 = sqr(sin_ksl);
  R[1][2] = -(R[3][0] = ks * S2);
  R[2][1] = -(R[0][3] = S2 / ks);

  if (M->order == 2) {
    double sin_2ksl, cos_2ksl;
    sin_2ksl = sin(2 * ksl);
    cos_2ksl = cos(2 * ksl);

    temp = ksl * sin_2ksl;
    T[0][5][0] = T[1][5][1] = T[2][5][2] = T[3][5][3] = temp;

    T[0][5][1] = T[2][5][3] = sin_2ksl / (2 * ks) - length * cos_2ksl;

    temp = -ksl * cos_2ksl;
    T[0][5][2] = T[1][5][3] = temp;
    T[3][5][1] = T[2][5][0] = -temp;

    T[2][5][1] = -(T[0][5][3] = (1.0 - cos_2ksl) / (2 * ks) - length * sin_2ksl);
    T[1][5][0] = T[3][5][2] = 0.5 * ks * (2 * ksl * cos_2ksl + sin_2ksl);
    T[1][5][2] = T[3][5][0] = 0.5 * ks * (1.0 - cos_2ksl + 2 * ksl * sin_2ksl);
    T[3][5][0] = -T[1][5][2];

    T[4][1][1] = T[4][3][3] = length / 2;
    /* These terms per P. Emma */
    T[4][0][0] = T[4][2][2] = sqr(ks) * length / 2;
    T[4][3][0] = -(T[4][2][1] = ks * length);
  }
  log_exit("solenoid_matrix");
  return (M);
}

VMATRIX *compute_matrix(
  ELEMENT_LIST *elem,
  RUN *run,
  VMATRIX *Mspace /* currently ignored--intended to allow memory to be passed to put matrix in */
) {
  QUAD *quad;
  BEND *bend;
  SEXT *sext;
  HCOR *hcor;
  HVCOR *hvcor;
  VCOR *vcor;
  ALPH *alph;
  DRIFT *drift;
  OCTU *oct;
  SOLE *sole;
  ROTATE *rot;
  QFRING *qfring;
  MONI *moni;
  HMON *hmon;
  VMON *vmon;
  KSEXT *ksext;
  KOCT *koct;
  KSBEND *ksbend;
  KQUAD *kquad;
  DQCOR *dqcor;
  NIBEND *nibend;
  NISEPT *nisept;
  KQUSE *kquse;
  MULT *mult;
  FMULT *fmult;
  SAMPLE *sample;
  STRAY *stray;
  CSBEND *csbend;
  CCBEND *ccbend;
  RFCA *rfca;
  ENERGY *energy;
  RFCW *rfcw;
  LGBEND *lgbend;
  MATTER *matter;
  MALIGN *malign;
  MATR *matr;
  MODRF *modrf;
  CSRCSBEND *csrcsbend;
  BRAT *brat;
  CSRDRIFT *csrdrift;
  LSCDRIFT *lscdrift;
  EDRIFT *edrift;
  WIGGLER *wiggler;
  CWIGGLER *cwiggler;
  APPLE *apple;
  UKICKMAP *ukmap;
  SCRIPT *script;
  EHCOR *ehcor;
  EVCOR *evcor;
  EHVCOR *ehvcor;
  BEDGE *bEdge;
  double ks, Pref_output, pSave;
  VARY rcContext;
  long fiducialize;
  double tStart = 0;

  setTrackingContext(elem->name, elem->occurence, elem->type, run->rootname, elem, -1);

  getRunControlContext(&rcContext);
  fiducialize = 1;
  if (rcContext.ready) {
    if ((rcContext.fiducial_flag & FIRST_BEAM_IS_FIDUCIAL) &&
        rcContext.i_step != 0)
      fiducialize = 0;
  }
  if (elem->pred)
    elem->Pref_input = elem->pred->Pref_output;
  else
    elem->Pref_input = run->p_central;

  /* Pref_output is the assumed output value of Pref */
  Pref_output = elem->Pref_input;
  if (!fiducialize)
    /* Assume we've already fiducialized and use the previous Pref output value */
    Pref_output = elem->Pref_output;
  /* This variable is used to pass the input momentum to some elements and 
     * get the output momentum back.  It will be reset to the local variable
     * Pref_output if fiducializing 
     */
  elem->Pref_output = elem->Pref_input;

  if (elem->matrix) {
    free_matrices(elem->matrix);
    free(elem->matrix);
  }
  elem->matrix = NULL;

  if (timingActive)
    tStart = getTimeInSecs();
#if DEBUG
  printf("computing matrix for %s#%ld, type %s\n", elem->name, elem->occurence,
         entity_name[elem->type]);
  fflush(stdout);
#endif
  switch (elem->type) {
  case T_DRIF:
    drift = (DRIFT *)elem->p_elem;
    elem->matrix = drift_matrix(drift->length, (drift->order ? drift->order : run->default_order));
    break;
  case T_WIGGLER:
    wiggler = (WIGGLER *)elem->p_elem;
    wiggler->radiusInternal = wiggler->radius;
    if (wiggler->K > 0) {
      double period;
      /* poles = 2*(wiggler->poles/2)+1; */
      period = 2 * (wiggler->length / wiggler->poles);
      wiggler->radiusInternal = elem->Pref_input * period / (PIx2 * wiggler->K);
    } else if (wiggler->B > 0)
      wiggler->radiusInternal = elem->Pref_input / (particleCharge / particleMass / c_mks) / wiggler->B;
    else if (wiggler->gap > 0) {
      double B;
      if ((B = computeUndulatorFieldFromModel(wiggler->gap, wiggler->jFraction, &(wiggler->C[0]), wiggler->length, wiggler->poles, wiggler->model))>0)
	wiggler->radiusInternal = elem->Pref_input / (particleCharge / particleMass / c_mks) / B;
    }
    if (wiggler->radiusInternal == 0) {
      fprintf(stderr, "Error: wiggler radius is zero\n");
      fprintf(stderr, "Parameters are length=%e, poles=%ld, radius=%e, K=%e, B=%e, gap=%e, j/jC=%e, model=%s\n",
              wiggler->length, wiggler->poles, wiggler->radiusInternal, wiggler->K, wiggler->B, wiggler->gap, wiggler->jFraction, wiggler->model);
    }
    elem->matrix = wiggler_matrix(wiggler->length, wiggler->radiusInternal, wiggler->poles,
                                  run->default_order, wiggler->focusing);
    if (wiggler->dx || wiggler->dy || wiggler->dz || wiggler->tilt) {
      misalign_matrix(elem->matrix,
                      wiggler->dx, wiggler->dy, wiggler->dz,
                      0.0, 0.0, 0.0, wiggler->tilt,
                      0.0, wiggler->length,
                      misalignmentMethod);
    }
    break;
  case T_CWIGGLER:
    cwiggler = (CWIGGLER *)elem->p_elem;
    InitializeCWiggler(cwiggler, elem->name);
    if (cwiggler->BPeak[0] == 0 && cwiggler->BPeak[1] == 0) {
      printWarningForTracking("CWIGGLER has zero field in both planes.", NULL);
      elem->matrix = drift_matrix(cwiggler->length, run->default_order);
      cwiggler->radiusInternal[0] = cwiggler->radiusInternal[1] = HUGE_VAL;
    } else {
      cwiggler->radiusInternal[0] = cwiggler->radiusInternal[1] = HUGE_VAL;
      if (cwiggler->BPeak[0])
        cwiggler->radiusInternal[0] = elem->Pref_input / (particleCharge / particleMass / c_mks) / cwiggler->BPeak[0];
      if (cwiggler->BPeak[1])
        cwiggler->radiusInternal[1] = elem->Pref_input / (particleCharge / particleMass / c_mks) / cwiggler->BPeak[1];
      /* printf("Wiggler %s radii are %le  and %le,  length is %le, with %ld periods\n",
             elem->name, cwiggler->radiusInternal[0], cwiggler->radiusInternal[1], 
             cwiggler->length, cwiggler->periods);
             */
      pSave = run->p_central;
      run->p_central = elem->Pref_input;
      elem->matrix = determineMatrixHigherOrder(run, elem, NULL, NULL, MIN(run->default_order, 3));
      run->p_central = pSave;
    }
    break;
  case T_APPLE:
    apple = (APPLE *)elem->p_elem;
    InitializeAPPLE(apple->Input, apple);
    if (apple->BPeak[0] == 0 && apple->BPeak[1] == 0) {
      printWarningForTracking("GFWIGGLER has zero field in both planes.", NULL);
      elem->matrix = drift_matrix(apple->length, run->default_order);
      apple->radiusInternal[0] = apple->radiusInternal[1] = HUGE_VAL;
    } else {
      apple->radiusInternal[0] = apple->radiusInternal[1] = HUGE_VAL;
      if (apple->BPeak[0])
        apple->radiusInternal[0] = elem->Pref_input / (particleCharge / particleMass / c_mks) / apple->BPeak[0];
      if (apple->BPeak[1])
        apple->radiusInternal[1] = elem->Pref_input / (particleCharge / particleMass / c_mks) / apple->BPeak[1];
      /* printf("Wiggler %s radii are %le  and %le,  length is %le, with %ld periods\n",
             elem->name, apple->radiusInternal[0], apple->radiusInternal[1], 
             apple->length, apple->periods);
             */
      pSave = run->p_central;
      run->p_central = elem->Pref_input;
      elem->matrix = determineMatrix(run, elem, NULL, NULL);
      run->p_central = pSave;
    }
    break;
  case T_UKICKMAP:
    ukmap = ((UKICKMAP *)elem->p_elem);
    ukmap->radiusInternal = -1;
    if (ukmap->Kreference || ukmap->Kactual) {
      double K;
      if (ukmap->Kreference && ukmap->Kactual)
        bombElegant("UKICKMAP has nonzero value for both KREF and KACTUAL", NULL);
      if (ukmap->periods <= 0)
        bombElegant("UKICKMAP has PERIODS<=0 and KREFERENCE non-zero", NULL);
      if ((K = ukmap->Kreference ? ukmap->Kreference * ukmap->fieldFactor : ukmap->Kactual) > 0) {
        ukmap->radiusInternal =
          elem->Pref_input * (ukmap->length / ukmap->periods) / (PIx2 * K);
      }
    }
    pSave = run->p_central;
    run->p_central = elem->Pref_input;
    elem->matrix = determineMatrix(run, elem, NULL, NULL);
    run->p_central = pSave;
    break;
  case T_KICKMAP:
    pSave = run->p_central;
    run->p_central = elem->Pref_input;
    elem->matrix = determineMatrix(run, elem, NULL, NULL);
    run->p_central = pSave;
    break;
  case T_SCRIPT:
    script = ((SCRIPT *)elem->p_elem);
    if (script->driftMatrix)
      elem->matrix = drift_matrix(script->length, run->default_order);
    else {
      pSave = run->p_central;
      run->p_central = elem->Pref_input;
      elem->matrix = determineMatrix(run, elem, NULL, NULL);
      run->p_central = pSave;
    }
    break;
  case T_FTABLE:
    /* ftable = ((FTABLE*)elem->p_elem); */
    pSave = run->p_central;
    run->p_central = elem->Pref_input;
    elem->matrix = determineMatrix(run, elem, NULL, NULL);
    run->p_central = pSave;
    break;
  case T_BGGEXP:
  case T_BOFFAXE:
    pSave = run->p_central;
    run->p_central = elem->Pref_input;
    elem->matrix = determineMatrixHigherOrder(run, elem, NULL, NULL, MIN(run->default_order, 3));
    run->p_central = pSave;
    break;
  case T_BEDGE:
    bEdge = (BEDGE*)elem->p_elem;
    if (bEdge->rho==0)
      elem->matrix = drift_matrix(0.0, 1);
    else
      elem->matrix = edge_matrix(bEdge->beta, 1./(bEdge->rho), bEdge->hPoleFace, 0.0, bEdge->exitEdge?1:-1, 2*bEdge->fint*bEdge->hgap, bEdge->order, 1, 0, 0.0);
    break;
  case T_RBEN:
  case T_SBEN:
    bend = (BEND *)elem->p_elem;
    bend->edgeFlags = determine_bend_flags(elem, bend->edge_effects[bend->e1Index], bend->edge_effects[bend->e2Index]);
    if (bend->use_bn) {
      bend->k1_internal = bend->b1 / (bend->length / bend->angle);
      bend->k2_internal = bend->b2 / (bend->length / bend->angle);
    } else {
      bend->k1_internal = bend->k1;
      bend->k2_internal = bend->k2;
    }
    elem->matrix =
      bend_matrix(bend->length, bend->angle,
                  bend->e[bend->e1Index], bend->e[bend->e2Index],
                  bend->h[bend->e1Index], bend->h[bend->e2Index],
                  bend->k1_internal, bend->k2_internal,
                  misalignmentMethod == 0 ? bend->tilt : 0.0,
                  bend->fint, bend->fint,
                  bend->hgap * 2, bend->fse, bend->fseDipole, bend->fseQuadrupole,
                  misalignmentMethod == 0 ? bend->etilt * bend->etiltSign : 0.0,
                  bend->order ? bend->order : run->default_order, bend->edge_order,
                  bend->edgeFlags, bend->TRANSPORT, 0.0, 00);
    if (bend->dx || bend->dy || bend->dz || bend->tilt || bend->etilt) {
      misalign_matrix(elem->matrix,
                      bend->dx, bend->dy, bend->dz,
                      0.0, 0.0,
                      misalignmentMethod == 0 ? 0 : bend->etiltSign * bend->etilt,
                      misalignmentMethod == 0 ? 0 : bend->tilt,
                      bend->angle, bend->length,
                      misalignmentMethod);
    }
    break;
  case T_QUAD:
    quad = (QUAD *)elem->p_elem;
    elem->matrix = quadrupole_matrix(quad->k1, quad->length,
                                     quad->order ? quad->order : run->default_order,
                                     quad->fse, quad->xkick*quad->xKickCalibration, quad->ykick*quad->yKickCalibration,
                                     quad->edge1_effects, quad->edge2_effects,
                                     quad->fringeType, quad->ffringe, quad->lEffective,
                                     quad->fringeIntM, quad->fringeIntP, quad->radial);
    if (quad->dx || quad->dy || quad->dz || quad->tilt || quad->pitch || quad->yaw) {
      misalign_matrix(elem->matrix, quad->dx, quad->dy, quad->dz,
                      quad->pitch, quad->yaw, 0.0, quad->tilt, 0.0, quad->length,
                      quad->malignMethod);
    }
    break;
  case T_SEXT:
    sext = (SEXT *)elem->p_elem;
    elem->matrix = sextupole_matrix(sext->k2, sext->k1, sext->j1, sext->length,
                                    sext->order ? sext->order : run->default_order,
                                    sext->fse, 0.0, 0.0, sext->ffringe);
    if (sext->dx || sext->dy || sext->dz || sext->tilt)
      misalign_matrix(elem->matrix, sext->dx, sext->dy, sext->dz,
                      0.0, 0.0, 0.0, sext->tilt, 0.0, sext->length,
                      misalignmentMethod);
    break;
  case T_OCT:
    oct = (OCTU *)elem->p_elem;
    elem->matrix = octupole_matrix(oct->k3, oct->length,
                                   oct->order ? oct->order : run->default_order, oct->fse);
    if (oct->dx || oct->dy || oct->dz || oct->tilt)
      misalign_matrix(elem->matrix, oct->dx, oct->dy, oct->dz,
                      0.0, 0.0, 0.0, oct->tilt, 0.0, oct->length,
                      misalignmentMethod);
    break;
  case T_ALPH:
    alph = (ALPH *)elem->p_elem;
    if (alph->xmax)
      alph->gradient = elem->Pref_input * sqr(ALPHA_CONST / alph->xmax);
    else
      bombElegant("supply xmax for alpha magnet", NULL);
    elem->matrix = alpha_magnet_matrix(alph->gradient, sqrt(sqr(run->p_central) + 1),
                                       (alph->order ? alph->order : run->default_order), alph->part);
    if ((alph->xs1 || alph->xs2 || alph->dp1 != -1 || alph->dp2 != 1 ||
         alph->xPuck != -1 || alph->widthPuck != 0) &&
        alph->part == 0)
      bombElegant("alpha-magnet scraper not supported for full magnet", NULL);
    if (alph->tilt)
      tilt_matrices(elem->matrix, alph->tilt);
    if (alph->dx || alph->dy || alph->dz)
      misalign_matrix(elem->matrix, alph->dx, alph->dy, alph->dz,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0);
    break;
  case T_ROTATE:
    rot = (ROTATE *)elem->p_elem;
    if (rot->excludeOptics)
      elem->matrix = drift_matrix(0.0, 1);
    else
      elem->matrix = rotation_matrix(rot->tilt);
    break;
  case T_MONI:
    moni = (MONI *)elem->p_elem;
    elem->matrix = drift_matrix(moni->length, moni->order ? moni->order : run->default_order);
    break;
  case T_HMON:
    hmon = (HMON *)elem->p_elem;
    elem->matrix = drift_matrix(hmon->length, hmon->order ? hmon->order : run->default_order);
    break;
  case T_VMON:
    vmon = (VMON *)elem->p_elem;
    elem->matrix = drift_matrix(vmon->length, vmon->order ? vmon->order : run->default_order);
    break;
  case T_SOLE:
    sole = (SOLE *)elem->p_elem;
    ks = sole->ks;
    if (sole->B && !ks)
      ks = -sole->B * particleCharge / (particleMass * c_mks * elem->Pref_input);
    elem->matrix = solenoid_matrix(sole->length, ks,
                                   sole->order ? sole->order : run->default_order);
    if (sole->dx || sole->dy || sole->dz)
      misalign_matrix(elem->matrix, sole->dx, sole->dy, sole->dz,
                      0.0, 0.0, 0.0, 0.0, 0.0, sole->length,
                      misalignmentMethod);
    break;
  case T_VCOR:
    vcor = (VCOR *)elem->p_elem;
    elem->matrix = corrector_matrix(vcor->length, vcor->kick, vcor->tilt + PI / 2,
                                    vcor->b2, vcor->calibration,
                                    vcor->edge_effects, vcor->order ? vcor->order : run->default_order);
    break;
  case T_HCOR:
    hcor = (HCOR *)elem->p_elem;
    elem->matrix = corrector_matrix(hcor->length, hcor->kick, hcor->tilt,
                                    hcor->b2, hcor->calibration,
                                    hcor->edge_effects, hcor->order ? hcor->order : run->default_order);
    break;
  case T_MATR:
    matr = (MATR *)elem->p_elem;
    if (!matr->matrix_read) {
      char *filename;
      FILE *fpm;
      if (!(filename = findFileInSearchPath(matr->filename))) {
        fprintf(stderr, "Unable to find MATR file %s\n", matr->filename);
        exitElegant(1);
      }
      printf("File %s found: %s\n", matr->filename, filename);
      fpm = fopen_e(filename, "r", 0);
      matr->M.order = matr->order;
      initialize_matrices(&(matr->M), matr->order);
      if (!read_matrices(&(matr->M), filename, fpm)) {
        printf("error reading matrix from file %s\n", filename);
        fflush(stdout);
        abort();
      }
      free(filename);
      fclose(fpm);
      matr->matrix_read = 1;
      matr->length = matr->M.C[4];
    }
    elem->matrix = interpolateMatrixWithIdentityMatrix(&(((MATR *)elem->p_elem)->M), matr->fraction, matr->order);
    break;
  case T_WATCH:
    break;
  case T_HISTOGRAM:
    break;
  case T_QFRING:
    qfring = (QFRING *)elem->p_elem;
    elem->matrix = qfringe_matrix(qfring->k1, qfring->length,
                                  qfring->direction, qfring->order ? qfring->order : run->default_order,
                                  qfring->fse);
    if (qfring->dx || qfring->dy || qfring->dz || qfring->tilt)
      misalign_matrix(elem->matrix, qfring->dx, qfring->dy, qfring->dz,
                      0.0, 0.0, 0.0, qfring->tilt, 0.0, qfring->length,
                      misalignmentMethod);
    break;
  case T_MALIGN:
    malign = (MALIGN *)elem->p_elem;
    if (malign->on_pass == -1 || malign->forceModifyMatrix)
      elem->matrix = misalignment_matrix((MALIGN *)elem->p_elem, run->default_order);
    else
      elem->matrix = drift_matrix(0.0, run->default_order);
    break;
  case T_KSBEND:
    ksbend = (KSBEND *)elem->p_elem;
    if (ksbend->n_kicks < 1)
      bombElegant("n_kicks must be > 0 for KSBEND element", NULL);
    ksbend->flags = determine_bend_flags(elem, ksbend->edge_effects[ksbend->e1Index], ksbend->edge_effects[ksbend->e2Index]);
    elem->matrix =
      bend_matrix(ksbend->length, ksbend->angle,
                  ksbend->e[ksbend->e1Index], ksbend->e[ksbend->e2Index],
                  ksbend->h[ksbend->e1Index], ksbend->h[ksbend->e2Index],
                  ksbend->k1, ksbend->k2,
                  misalignmentMethod == 0 ? ksbend->tilt : 0.0,
                  ksbend->fint, ksbend->fint,
                  ksbend->hgap * 2, ksbend->fse, 0, 0,
                  misalignmentMethod == 0 ? ksbend->etilt : 0.0,
                  ksbend->nonlinear ? 2 : (run->default_order ? run->default_order : 1),
                  ksbend->edge_order, ksbend->flags,
                  ksbend->TRANSPORT, 0.0, 0.0);
    if (ksbend->dx || ksbend->dy || ksbend->dz || ksbend->tilt || ksbend->etilt)
      misalign_matrix(elem->matrix, ksbend->dx, ksbend->dy, ksbend->dz,
                      0.0, 0.0,
                      misalignmentMethod == 0 ? 0 : ksbend->etilt,
                      misalignmentMethod == 0 ? 0 : ksbend->tilt,
                      ksbend->angle, ksbend->length,
                      misalignmentMethod);
    break;
  case T_KQUAD:
    kquad = (KQUAD *)elem->p_elem;
    if (kquad->bore)
      kquad->k1 = kquad->B / kquad->bore * (particleCharge / (particleMass * c_mks * elem->Pref_input));
    if (kquad->nSlices < 1)
      bombElegant("n_slices must be > 0 for KQUAD element", NULL);
    readErrorMultipoleData(&(kquad->systematicMultipoleData),
                           kquad->systematic_multipoles, 0);
    readErrorMultipoleData(&(kquad->randomMultipoleData),
                           kquad->random_multipoles, 0);
    readErrorMultipoleData(&(kquad->steeringMultipoleData),
                           kquad->steering_multipoles, 1);
    if (kquad->trackingBasedMatrix <= 0) {
      elem->matrix = quadrupole_matrix(kquad->k1, kquad->length,
                                       (run->default_order ? run->default_order : 1),
                                       kquad->fse, kquad->xkick * kquad->xKickCalibration, kquad->ykick * kquad->yKickCalibration,
                                       kquad->edge1Linear ? kquad->edge1_effects : 0,
                                       kquad->edge2Linear ? kquad->edge2_effects : 0,
                                       "integrals", 0.0, kquad->lEffective,
                                       kquad->fringeIntM, kquad->fringeIntP, kquad->radial);
      if (kquad->dx || kquad->dy || kquad->dz || kquad->tilt || kquad->pitch || kquad->yaw)
        misalign_matrix(elem->matrix, kquad->dx, kquad->dy, kquad->dz,
                        kquad->pitch, kquad->yaw, 0.0, kquad->tilt, 0.0, kquad->length,
                        kquad->malignMethod);
    } else {
      if (kquad->trackingBasedMatrix > 3)
        kquad->trackingBasedMatrix = 3;
      elem->matrix = determineMatrixHigherOrder(run, elem, NULL, NULL, MIN(run->default_order, kquad->trackingBasedMatrix));
    }
    break;
  case T_DQCOR:
    dqcor = (DQCOR *)elem->p_elem;
    if (dqcor->nSlices < 1)
      bombElegant("n_slices must be > 0 for DQCOR element", NULL);
    readErrorMultipoleData(&(dqcor->quadrupoleSystematicMultipoleData),
                           dqcor->quadrupoleSystematicMultipoles, 0);
    readErrorMultipoleData(&(dqcor->quadrupoleRandomMultipoleData),
                           dqcor->quadrupoleRandomMultipoles, 0);
    readErrorMultipoleData(&(dqcor->dipoleSystematicMultipoleData),
                           dqcor->dipoleSystematicMultipoles, 1);
    if (dqcor->trackingBasedMatrix <= 0) {
      determineDQCorReferenceFrameTilt(dqcor);
      elem->matrix = dqcor_matrix(dqcor->K1ReferenceFrame, dqcor->length,
				  (run->default_order ? run->default_order : 2),
				  dqcor->fse, dqcor->xKickReferenceFrame, dqcor->yKickReferenceFrame);
      if (dqcor->phiReferenceFrame)
	tilt_matrices(elem->matrix, dqcor->phiReferenceFrame);
      if (dqcor->dx || dqcor->dy || dqcor->dz || dqcor->tilt || dqcor->pitch || dqcor->yaw)
        misalign_matrix(elem->matrix, dqcor->dx, dqcor->dy, dqcor->dz,
                        dqcor->pitch, dqcor->yaw, 0.0, dqcor->tilt, 0.0, dqcor->length,
                        dqcor->malignMethod);
    } else {
      if (dqcor->trackingBasedMatrix > 3)
        dqcor->trackingBasedMatrix = 3;
      elem->matrix = determineMatrixHigherOrder(run, elem, NULL, NULL, MIN(run->default_order, dqcor->trackingBasedMatrix));
    }
    break;
  case T_KSEXT:
    ksext = (KSEXT *)elem->p_elem;
    if (ksext->bore)
      ksext->k2 = 2 * ksext->B / sqr(ksext->bore) * (particleCharge / (particleMass * c_mks * elem->Pref_input));
    if (ksext->nSlices < 1)
      bombElegant("n_slices must be > 0 for KSEXT element", NULL);
    elem->matrix = sextupole_matrix(ksext->k2, ksext->k1, ksext->j1, ksext->length,
                                    (run->default_order ? run->default_order : 2),
                                    ksext->fse,
                                    ksext->xkick * ksext->xKickCalibration, ksext->ykick * ksext->yKickCalibration,
                                    0.0);
    if (ksext->dx || ksext->dy || ksext->dz || ksext->tilt || ksext->yaw || ksext->pitch)
      misalign_matrix(elem->matrix, ksext->dx, ksext->dy, ksext->dz,
                      ksext->pitch, ksext->yaw, 0.0, ksext->tilt, 0.0, ksext->length,
                      ksext->malignMethod);
    readErrorMultipoleData(&(ksext->systematicMultipoleData),
                           ksext->systematic_multipoles, 0);
    readErrorMultipoleData(&(ksext->randomMultipoleData),
                           ksext->random_multipoles, 0);
    break;
  case T_KOCT:
    koct = (KOCT *)elem->p_elem;
    if (koct->bore)
      koct->k3 = 6 * koct->B / ipow3(koct->bore) * (particleCharge / (particleMass * c_mks * elem->Pref_input));
    if (koct->nSlices < 1)
      bombElegant("n_slices must be > 0 for KOCT element", NULL);
    elem->matrix = octupole_matrix(koct->k3, koct->length, (run->default_order ? run->default_order : 2), koct->fse);
    if (koct->dx || koct->dy || koct->dz || koct->tilt || koct->yaw || koct->pitch)
      misalign_matrix(elem->matrix, koct->dx, koct->dy, koct->dz,
                      koct->pitch, koct->yaw, 0.0, koct->tilt, 0.0, koct->length,
                      koct->malignMethod);
    readErrorMultipoleData(&(koct->systematicMultipoleData),
                           koct->systematic_multipoles, 0);
    readErrorMultipoleData(&(koct->randomMultipoleData),
                           koct->random_multipoles, 0);
    break;
  case T_KQUSE:
    kquse = (KQUSE *)elem->p_elem;
    if (kquse->nSlices < 1)
      bombElegant("n_slices must be > 0 for KQUSE element", NULL);
    elem->matrix = quse_matrix(kquse->k1, kquse->k2, kquse->length,
                               (run->default_order ? run->default_order : 1),
                               kquse->fse1, kquse->fse2);
    if (kquse->dx || kquse->dy || kquse->dz || kquse->tilt)
      misalign_matrix(elem->matrix, kquse->dx, kquse->dy, kquse->dz,
                      0.0, 0.0, 0.0, kquse->tilt, 0.0, kquse->length,
                      misalignmentMethod);
    break;
  case T_MAGNIFY:
    elem->matrix = magnification_matrix((MAGNIFY *)elem->p_elem);
    break;
  case T_SAMPLE:
    sample = (SAMPLE *)elem->p_elem;
    if (sample->interval <= 0)
      bombElegant("sample interval invalid", NULL);
    if (sample->fraction > 1 || sample->fraction <= 0)
      bombElegant("sample fraction invalid", NULL);
    break;
  case T_HVCOR:
    hvcor = (HVCOR *)elem->p_elem;
    elem->matrix = hvcorrector_matrix(hvcor->length, hvcor->xkick, hvcor->ykick,
                                      hvcor->tilt, hvcor->b2, hvcor->xcalibration, hvcor->ycalibration,
                                      hvcor->edge_effects, hvcor->order ? hvcor->order : run->default_order);
    break;
  case T_NIBEND:
    nibend = (NIBEND *)elem->p_elem;
    nibend->edgeFlags = determine_bend_flags(elem, 1, 1);
    elem->matrix =
      bend_matrix(nibend->length, nibend->angle,
                  nibend->e[nibend->e1Index], nibend->e[nibend->e2Index],
                  0.0, 0.0, 0.0, 0.0, nibend->tilt, nibend->fint, nibend->fint, nibend->hgap * 2,
                  nibend->fse, 0.0, 0.0, nibend->etilt,
                  (run->default_order ? run->default_order : 1), 0L, nibend->edgeFlags, 0, 0.0, 0.0);
    break;
  case T_NISEPT:
    nisept = (NISEPT *)elem->p_elem;
    nisept->edgeFlags = determine_bend_flags(elem, 1, 1);
    elem->matrix =
      bend_matrix(nisept->length, nisept->angle, nisept->e1, nisept->angle - nisept->e1,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  (run->default_order ? run->default_order : 1), 0, nisept->edgeFlags, 0, 0.0, 0.0);
    break;
  case T_STRAY:
    stray = (STRAY *)elem->p_elem;
    elem->matrix =
      stray_field_matrix(stray->length, &stray->lBx, &stray->gBx, elem->end_theta,
                         (stray->order ? stray->order : run->default_order),
                         elem->Pref_input, stray->WiInitialized ? stray->Wi : NULL);
    break;
  case T_CSBEND:
    csbend = (CSBEND *)elem->p_elem;
    if (csbend->nSlices < 1)
      bombElegant("n_slices must be > 0 for CSBEND element", NULL);
    csbend->edgeFlags = determine_bend_flags(elem, csbend->edge_effects[csbend->e1Index], csbend->edge_effects[csbend->e2Index]);
    csbend_update_fse_adjustment(csbend, elem);
    if (csbend->length == 0 && (csbend->use_bn || csbend->xReference))
      bombElegant("Can't compute matrix for CSBEND with L=0 and USE_BN!=0 or XREFERENCE!=0", NULL);
    if (csbend->trackingMatrix > 0) {
      if (csbend->trackingMatrix > 3)
        csbend->trackingMatrix = 3;
      elem->matrix = determineMatrixHigherOrder(run, elem, NULL, NULL, MIN(run->default_order, csbend->trackingMatrix));
    } else {
      long edgeOrder;
      edgeOrder = csbend->edge_order;
      if (csbend->edge_effects[csbend->e1Index] > 2 || csbend->edge_effects[csbend->e2Index] > 2)
        edgeOrder = 2;
      elem->matrix = bend_matrix(csbend->length, csbend->angle, csbend->e[csbend->e1Index], csbend->e[csbend->e2Index],
                                 csbend->h[csbend->e1Index], csbend->h[csbend->e2Index],
                                 (csbend->use_bn ? csbend->b1 * csbend->angle / csbend->length : csbend->k1) + (csbend->xReference > 0 ? csbend->f1 * csbend->angle / csbend->length / csbend->xReference : 0),
                                 (csbend->use_bn ? csbend->b2 / (csbend->length / csbend->angle) : csbend->k2) + (csbend->xReference > 0 ? 2 * csbend->f2 * csbend->angle / csbend->length / sqr(csbend->xReference) : 0),
                                 csbend->malignMethod == 0 ? csbend->tilt : 0.0,
                                 csbend->fint[csbend->e1Index] >= 0 ? csbend->fint[csbend->e1Index] : csbend->fintBoth,
                                 csbend->fint[csbend->e2Index] >= 0 ? csbend->fint[csbend->e2Index] : csbend->fintBoth,
                                 csbend->hgap * 2,
                                 csbend->fse + (csbend->fseCorrection ? csbend->fseCorrectionValue : 0),
                                 csbend->fseDipole, csbend->fseQuadrupole,
                                 csbend->malignMethod == 0 ? csbend->etilt * csbend->etiltSign : 0.0,
                                 csbend->nonlinear ? 2 : (run->default_order ? run->default_order : 1),
                                 edgeOrder, csbend->edgeFlags, 0,
                                 csbend->xKick, csbend->yKick);
      if (csbend->dx || csbend->dy || csbend->dz || csbend->tilt || csbend->etilt || csbend->epitch || csbend->eyaw) {
        misalign_matrix(elem->matrix, csbend->dx, csbend->dy, csbend->dz,
                        csbend->epitch, csbend->eyaw,
                        csbend->malignMethod == 0 ? 0 : csbend->etiltSign * csbend->etilt,
                        csbend->malignMethod == 0 ? 0 : csbend->tilt,
                        csbend->angle, csbend->length,
                        csbend->malignMethod);
      }
    }
    break;
  case T_BRAT:
    brat = (BRAT *)elem->p_elem;
    if (brat->useDriftMatrix)
      elem->matrix = drift_matrix(brat->length, 1);
    else if (brat->useSbenMatrix)
      elem->matrix = bend_matrix(brat->length, brat->angle, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0,
                                 brat->fse, 0.0, 0.0, 0.0, 1, 0, 0, 0, 0.0, 0.0);
    else {
      printf("Using tracking to determine matrix for %s\n", elem->name);
      elem->matrix = determineMatrixHigherOrder(run, elem, NULL, NULL, MIN(run->default_order, 3));
    }
    break;
  case T_BMAPXY:
    elem->matrix = determineMatrixHigherOrder(run, elem, NULL, NULL, MIN(run->default_order, 3));
    break;
  case T_BMAPXYZ:
    if (((BMAPXYZ *)elem->p_elem)->driftMatrix)
      elem->matrix = drift_matrix(((BMAPXYZ *)elem->p_elem)->length, MIN(run->default_order, 3));
    else
      elem->matrix = determineMatrixHigherOrder(run, elem, NULL, NULL, MIN(run->default_order, 3));
    break;
  case T_CCBEND:
    ccbend = (CCBEND *)elem->p_elem;
#ifdef DEBUG_CCBEND
    printf("Computing matrix for CCBEND %s#%ld\n", elem->name, elem->occurence);
    printf("optimized = %hd\n", ccbend->optimized);
    fflush(stdout);
#endif
    if (ccbend->nSlices < 1)
      bombElegant("N_SLICES must be > 0 for CCBEND element", NULL);
    elem->matrix = determineMatrixHigherOrder(run, elem, NULL, NULL, MIN(run->default_order, 3));
    break;
  case T_LGBEND:
    lgbend = (LGBEND *)elem->p_elem;
#ifdef DEBUG_LGBEND
    printf("Computing matrix for LGBEND %s#%ld\n", elem->name, elem->occurence);
    printf("optimized = %hd\n", lgbend->optimized);
    fflush(stdout);
#endif
    if (lgbend->nSlices < 1)
      bombElegant("N_SLICES must be > 0 for LGBEND element", NULL);
    elem->matrix = determineMatrixHigherOrder(run, elem, NULL, NULL, MIN(run->default_order, 3));
    break;
  case T_CSRCSBEND:
    csrcsbend = (CSRCSBEND *)elem->p_elem;
    if (csrcsbend->nSlices < 1)
      bombElegant("N_SLICES must be > 0 for CSRCSBEND element", NULL);
    csrcsbend->edgeFlags = determine_bend_flags(elem, csrcsbend->edge_effects[csrcsbend->e1Index], csrcsbend->edge_effects[csrcsbend->e2Index]);
    elem->matrix =
      bend_matrix(csrcsbend->length, csrcsbend->angle, csrcsbend->e[csrcsbend->e1Index], csrcsbend->e[csrcsbend->e2Index],
                  csrcsbend->h[csrcsbend->e1Index], csrcsbend->h[csrcsbend->e2Index],
                  csrcsbend->use_bn ? csrcsbend->b1 / (csrcsbend->length / csrcsbend->angle) : csrcsbend->k1,
                  csrcsbend->use_bn ? csrcsbend->b2 / (csrcsbend->length / csrcsbend->angle) : csrcsbend->k2,
                  misalignmentMethod == 0 ? csrcsbend->tilt : 0.0,
                  csrcsbend->fint, csrcsbend->fint,
                  csrcsbend->hgap * 2, csrcsbend->fse, 0, 0,
                  misalignmentMethod == 0 ? csrcsbend->etilt * csrcsbend->etiltSign : 0.0,
                  csrcsbend->nonlinear ? 2 : (run->default_order ? run->default_order : 1),
                  csrcsbend->edge_order, csrcsbend->edgeFlags, 0, 0.0, 0.0);
    if (csrcsbend->dx || csrcsbend->dy || csrcsbend->dz || csrcsbend->tilt || csrcsbend->etilt) {
      misalign_matrix(elem->matrix, csrcsbend->dx, csrcsbend->dy, csrcsbend->dz,
                      0.0, 0.0,
                      misalignmentMethod == 0 ? 0 : csrcsbend->etiltSign * csrcsbend->etilt,
                      misalignmentMethod == 0 ? 0 : csrcsbend->tilt,
                      csrcsbend->angle, csrcsbend->length,
                      misalignmentMethod);
    }
    break;
  case T_RFCA:
    rfca = (RFCA *)elem->p_elem;
    elem->matrix = rf_cavity_matrix(rfca->length, rfca->volt, rfca->freq, rfca->phase,
                                    &elem->Pref_output, run->default_order ? run->default_order : 1,
                                    rfca->end1Focus, rfca->end2Focus,
                                    rfca->bodyFocusModel, rfca->standingWave,
                                    fiducialize * (rfca->change_p0 || run->always_change_p0),
                                    Pref_output, elem, run);
    if (rfca->dx || rfca->dy)
      misalign_matrix(elem->matrix, rfca->dx, rfca->dy, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, rfca->length,
                      misalignmentMethod);
    break;
  case T_RFCW:
    rfcw = (RFCW *)elem->p_elem;
    elem->matrix = rf_cavity_matrix(rfcw->length, rfcw->volt, rfcw->freq, rfcw->phase,
                                    &elem->Pref_output, run->default_order ? run->default_order : 1,
                                    rfcw->end1Focus, rfcw->end2Focus,
                                    rfcw->bodyFocusModel, rfcw->standingWave,
                                    fiducialize * (rfcw->change_p0 || run->always_change_p0),
                                    Pref_output, elem, run);
    if (rfcw->dx || rfcw->dy)
      misalign_matrix(elem->matrix, rfcw->dx, rfcw->dy, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, rfcw->length,
                      misalignmentMethod);
    break;
  case T_MODRF:
    modrf = (MODRF *)elem->p_elem;
    elem->matrix = rf_cavity_matrix(modrf->length, modrf->volt, modrf->freq, modrf->phase,
                                    &elem->Pref_output, run->default_order ? run->default_order : 1,
                                    0, 0, NULL, 0,
                                    fiducialize * run->always_change_p0, Pref_output, elem, run);
    break;
  case T_ENERGY:
    energy = (ENERGY *)elem->p_elem;
    if (energy->central_energy || energy->central_momentum) {
      if (energy->central_energy)
        elem->Pref_output = sqrt(sqr(energy->central_energy) - 1);
      else
        elem->Pref_output = energy->central_momentum;
    }
    break;
  case T_MATTER:
    matter = (MATTER *)elem->p_elem;
    elem->matrix = drift_matrix(matter->length, run->default_order);
    break;
  case T_CSRDRIFT:
    csrdrift = (CSRDRIFT *)elem->p_elem;
    if ((csrdrift->useOvertakingLength ? 1 : 0) +
          (csrdrift->spread ? 1 : 0) + (csrdrift->attenuationLength > 0 ? 1 : 0) >
        1)
      bombElegant("Give one and only one of SPREAD, ATTENUATION_LENGTH, or USE_OVERTAKING_LENGTH for CSRDRIFT", NULL);
    elem->matrix = drift_matrix(csrdrift->length, run->default_order);
    break;
  case T_LSCDRIFT:
    lscdrift = (LSCDRIFT *)elem->p_elem;
    elem->matrix = drift_matrix(lscdrift->length, run->default_order);
    break;
  case T_EDRIFT:
    edrift = (EDRIFT *)elem->p_elem;
    elem->matrix = drift_matrix(edrift->length, run->default_order);
    break;
  case T_TWISSELEMENT:
    if (((TWISSELEMENT *)elem->p_elem)->disable)
      elem->matrix = drift_matrix(0.0, 1);
    else if (((TWISSELEMENT *)elem->p_elem)->from0Values)
      elem->matrix = twissTransformMatrix1(&(((TWISSELEMENT *)elem->p_elem)->twiss), &(((TWISSELEMENT *)elem->p_elem)->twiss0));
    else
      elem->matrix = twissTransformMatrix((TWISSELEMENT *)elem->p_elem, NULL);
    break;
  case T_REFLECT:
    elem->matrix = tmalloc(sizeof(*(elem->matrix)));
    initialize_matrices(elem->matrix, elem->matrix->order = 1);
    elem->matrix->R[0][0] = elem->matrix->R[2][2] =
      elem->matrix->R[4][4] = elem->matrix->R[5][5] = 1;
    elem->matrix->R[1][1] = elem->matrix->R[3][3] = -1;
    break;
  case T_LTHINLENS:
    elem->matrix = lightThinLensMatrix((LTHINLENS *)elem->p_elem);
    break;
  case T_LMIRROR:
    elem->matrix = lightMirrorMatrix((LMIRROR *)elem->p_elem);
    break;
  case T_EMATRIX:
    elem->matrix = matrixFromExplicitMatrix((EMATRIX *)elem->p_elem, run->default_order);
    if (((EMATRIX *)elem->p_elem)->deltaP)
      elem->Pref_output += ((EMATRIX *)elem->p_elem)->deltaP;
    break;
  case T_ILMATRIX:
    elem->matrix = matrixForILMatrix((ILMATRIX *)elem->p_elem, run->default_order);
    break;
  case T_SREFFECTS:
    elem->matrix = srEffectsMatrix((SREFFECTS *)elem->p_elem);
    break;
  case T_TWLA:
  case T_TWMTA:
  case T_MAPSOLENOID:
  case T_LSRMDLTR:
    pSave = run->p_central;
    run->p_central = elem->Pref_input;
    elem->matrix = determineMatrix(run, elem, NULL, NULL);
    elem->Pref_output = run->p_central;
    run->p_central = pSave;
    break;
  case T_RFDF:
    /* elem->matrix = determineMatrix(run, elem, NULL, NULL); */
    if (((RFDF *)elem->p_elem)->driftMatrix)
      elem->matrix = drift_matrix(((RFDF *)elem->p_elem)->length, run->default_order);
    else
      elem->matrix = rfdf_matrix((RFDF *)elem->p_elem, Pref_output);
    break;
  case T_MULT:
    elem->matrix = mult_matrix(mult = (MULT *)elem->p_elem, Pref_output, run->default_order);
    if (mult->dx || mult->dy || mult->dz || mult->tilt)
      misalign_matrix(elem->matrix, mult->dx, mult->dy, mult->dz,
                      0.0, 0.0, 0.0, mult->tilt, 0.0, mult->length,
                      misalignmentMethod);
    break;
  case T_FMULT:
    fmult = (FMULT *)elem->p_elem;
    if (fmult->untiltedMatrix) {
      double saveTilt;
      saveTilt = fmult->tilt;
      fmult->tilt = 0;
      elem->matrix = determineMatrixHigherOrder(run, elem, NULL, NULL, 3);
      fmult->tilt = saveTilt;
    } else
      elem->matrix = determineMatrixHigherOrder(run, elem, NULL, NULL, 3);
    break;
  case T_EHCOR:
    ehcor = (EHCOR *)elem->p_elem;
    elem->matrix = corrector_matrix(ehcor->length, ehcor->kick, ehcor->tilt,
                                    0, ehcor->calibration,
                                    0, ehcor->matrixOrder ? ehcor->matrixOrder : run->default_order);
    break;
  case T_EVCOR:
    evcor = (EVCOR *)elem->p_elem;
    elem->matrix = corrector_matrix(evcor->length, evcor->kick, evcor->tilt + PI / 2,
                                    0, evcor->calibration,
                                    0, evcor->matrixOrder ? evcor->matrixOrder : run->default_order);
    break;
  case T_EHVCOR:
    ehvcor = (EHVCOR *)elem->p_elem;
    elem->matrix = hvcorrector_matrix(ehvcor->length, ehvcor->xkick, ehvcor->ykick,
                                      ehvcor->tilt, 0, ehvcor->xcalibration, ehvcor->ycalibration,
                                      0, ehvcor->matrixOrder ? ehvcor->matrixOrder : run->default_order);
    break;
  case T_HKPOLY:
    pSave = run->p_central;
    run->p_central = elem->Pref_input;
    elem->matrix = determineMatrixHigherOrder(run, elem, NULL, NULL, MIN(run->default_order, 3));
    run->p_central = pSave;
    break;
    /* INSERT NEW CASES ABOVE THIS POINT */
  case T_KPOLY:
  case T_RFTMEZ0:
  case T_RMDF:
  case T_TMCF:
  case T_CEPL:
  case T_TWPL:
  case T_RCOL:
  case T_PEPPOT:
  case T_MAXAMP:
  case T_ECOL:
  case T_TRCOUNT:
  case T_RECIRC:
  case T_SCRAPER:
  case T_CENTER:
  case T_SCATTER:
  case T_RAMPRF:
  case T_RAMPP:
  case T_KICKER:
  case T_RFMODE:
  case T_REMCOR:
  case T_DSCATTER:
  case T_MKICKER:
  case T_SPEEDBUMP:
  case T_APCONTOUR:
  case T_TAPERAPC:
  case T_TAPERAPE:
  case T_TAPERAPR:
  default:
    if (entity_description[elem->type].flags & HAS_LENGTH)
      elem->matrix = drift_matrix(*((double *)elem->p_elem), run->default_order);
    if ((entity_description[elem->type].flags & HAS_MATRIX) && !elem->matrix) {
      printf("error: failed to compute matrix for %s, which should have a matrix.\n",
             elem->name);
      fflush(stdout);
      abort();
    }
    break;
  }

  if (timingActive) {
    timeCounter[elem->type] += getTimeInSecs() - tStart;
    runCounter[elem->type] += 1;
  }

  if (elem->matrix)
    /* This allows us to find the element if we have the matrix */
    elem->matrix->eptr = elem;

  return (elem->matrix);
}

char *watch_mode[N_WATCH_MODES] = {
  "coordinates",
  "parameters",
  "centroids",
  "fft",
};
char *fft_window_name[N_FFT_WINDOWS] = {
  "hanning",
  "parzen",
  "welch",
  "uniform",
};

void set_up_watch_point(WATCH *watch, RUN *run, long occurence, char *previousElementName, long previousElementOccurence,
                        long i_pass, ELEMENT_LIST *eptr, long IDSlotsPerBunch) {
  char *mode, *qualifier;

#if MPI_DEBUG
  printf("set_up_watch_point called\n");
  fflush(stdout);
#endif
  if (watch->disable)
    return;
  if (watch->interval <= 0 || watch->fraction <= 0)
    bombElegant("interval or fraction is non-positive for WATCH element", NULL);
  if (!watch->mode || watch->mode[0] == 0)
    bombElegant("mode must be given for WATCH element", NULL);
  mode = watch->mode;
  if ((qualifier = strchr(mode, ' ')))
    *qualifier++ = 0;
  if ((watch->mode_code = match_string(watch->mode, watch_mode, N_WATCH_MODES, 0)) < 0)
    bombElegant("unknown watch mode", NULL);
  if (watch->label && str_in(watch->label, "%s")) {
    char *buffer;
    buffer = tmalloc(sizeof(*buffer) * (strlen(watch->label) + strlen(run->rootname) + 1));
    sprintf(buffer, watch->label, run->rootname);
    free(watch->label);
    watch->label = buffer;
  }
  watch->filename = compose_filename_occurence(watch->filename, run->rootname, occurence + watch->indexOffset);
  SDDS_WatchPointSetup(watch, SDDS_BINARY, 1, run->runfile, run->lattice, "set_up_watch_point", qualifier, 
                       previousElementName, previousElementOccurence);
  watch->initialized = 1;
  watch->count = 0;
  watch->flushSample = -1;
  if (watch->autoReference && eptr) {
    double fmax = -DBL_MAX, f;
    while (eptr) {
      f = -1;
      if (eptr->type == T_RFCA) {
        f = ((RFCA *)eptr->p_elem)->freq;
      } else if (eptr->type == T_RFCW) {
        f = ((RFCW *)eptr->p_elem)->freq;
      }
      if (f > fmax)
        fmax = f;
      eptr = eptr->succ;
    }
    printf("Using frequency %21.15e Hz for rf reference for WATCH file %s\n",
           fmax, watch->filename);
    watch->referenceFrequency = fmax;
  }
  if (watch->start_pass < i_pass)
    /* Need this for WATCH points on branches that don't get executed on pass 0 */
    watch->start_pass = i_pass;
  if (watch->bunchSeries && IDSlotsPerBunch>0) {
    watch->startPID = 1 + (occurence-1)*IDSlotsPerBunch;
    watch->endPID = watch->startPID + IDSlotsPerBunch - 1;
  }
}

void set_up_histogram(HISTOGRAM *histogram, RUN *run, long occurence, long IDSlotsPerBunch) {
  if (histogram->disable)
    return;
  if (histogram->interval <= 0)
    bombElegant("interval is non-positive for HISTOGRAM element", NULL);
  if (histogram->bins <= 2)
    bombElegant("number of bins is less than 2 for HISTOGRAM element", NULL);
  if (!histogram->xData && !histogram->yData && !histogram->longitData)
    bombElegant("no data selected for HISTOGRAM element", NULL);
  if (histogram->binSizeFactor <= 0)
    bombElegant("bin_size_factor is non-positive for HISTOGRAM element", NULL);

  histogram->filename = compose_filename_occurence(histogram->filename, run->rootname, occurence);

  SDDS_HistogramSetup(histogram, SDDS_BINARY, 1, run->runfile, run->lattice, "set_up_histogram");
  histogram->initialized = 1;
  histogram->count = 0;
  if (histogram->bunchSeries && IDSlotsPerBunch>0) {
    histogram->startPID = 1 + (occurence-1)*IDSlotsPerBunch;
    histogram->endPID = histogram->startPID + IDSlotsPerBunch - 1;
  }
}

VMATRIX *magnification_matrix(MAGNIFY *magnif) {
  VMATRIX *M;

  log_entry("magnification_matrix");

  M = tmalloc(sizeof(*M));
  initialize_matrices(M, M->order = 1);
  M->R[0][0] = magnif->mx;
  M->R[1][1] = magnif->mxp;
  M->R[2][2] = magnif->my;
  M->R[3][3] = magnif->myp;
  M->R[4][4] = magnif->ms;
  M->R[5][5] = magnif->mdp;

  log_exit("magnification_matrix");
  return (M);
}

void reset_special_elements(LINE_LIST *beamline, unsigned long flags) {
  ELEMENT_LIST *eptr;
  NIBEND *nibend;
  NISEPT *nisept;
  short includeRF = 0, includeRandom = 0, includeNiElements = 0;

  log_entry("reset_special_elements");

  if (flags & RESET_INCLUDE_RF) {
#ifdef DEBUG_RESET
    printf("Resetting special rf elements\n");
#endif
    includeRF = 1;
  }
  if (flags & RESET_INCLUDE_RANDOM) {
#ifdef DEBUG_RESET
    printf("Resetting special random elements\n");
#endif
    includeRandom = 1;
  }
  if (flags & RESET_INCLUDE_NIELEM)
    includeNiElements = 1;
  eptr = beamline->elem;
  while (eptr) {
    switch (eptr->type) {
    case T_KICKER:
      ((KICKER *)eptr->p_elem)->fiducial_seen = 0;
      break;
    case T_NIBEND:
      if (includeNiElements) {
        nibend = (NIBEND *)eptr->p_elem;
        nibend->initialized = 0;
      }
      break;
    case T_NISEPT:
      if (includeNiElements) {
        nisept = (NISEPT *)eptr->p_elem;
        nisept->fse_opt = 0;
      }
      break;
    case T_TMCF:
      if (includeRF) {
        ((TMCF_MODE *)eptr->p_elem)->fiducial_part = NULL;
      }
      break;
    case T_RFTMEZ0:
      if (includeRF) {
        ((RFTMEZ0 *)eptr->p_elem)->fiducial_part = NULL;
      }
      break;
    case T_CEPL:
      ((CE_PLATES *)eptr->p_elem)->fiducial_part = NULL;
      break;
    case T_TWPL:
      ((TW_PLATES *)eptr->p_elem)->fiducial_part = NULL;
      break;
    case T_TWLA:
      if (includeRF) {
        ((TW_LINAC *)eptr->p_elem)->fiducial_part = NULL;
      }
      break;
    case T_TWMTA:
      if (includeRF) {
        ((TWMTA *)eptr->p_elem)->fiducial_part = NULL;
      }
      break;
    case T_RFCA:
      if (includeRF) {
        ((RFCA *)eptr->p_elem)->fiducial_seen = 0;
      }
      break;
    case T_RFCW:
      if (includeRF) {
        ((RFCW *)eptr->p_elem)->rfca.fiducial_seen = 0;
      }
      break;
    case T_RFDF:
      if (includeRF) {
        ((RFDF *)eptr->p_elem)->fiducial_seen = 0;
      }
      break;
    case T_RFTM110:
      if (includeRF) {
        ((RFTM110 *)eptr->p_elem)->fiducial_seen = 0;
      }
      break;
    case T_MODRF:
      if (includeRF) {
        ((MODRF *)eptr->p_elem)->fiducial_seen = 0;
      }
      break;
    case T_RFMODE:
      if (includeRF && ((RFMODE *)eptr->p_elem)->reset_for_each_step)
        ((RFMODE *)eptr->p_elem)->initialized = 0;
      break;
    case T_FRFMODE:
      if (includeRF && ((FRFMODE *)eptr->p_elem)->reset_for_each_step)
        ((FRFMODE *)eptr->p_elem)->initialized = 0;
      break;
    case T_TRFMODE:
      if (includeRF && ((TRFMODE *)eptr->p_elem)->reset_for_each_step)
        ((TRFMODE *)eptr->p_elem)->initialized = 0;
      break;
    case T_FTRFMODE:
      if (includeRF && ((FTRFMODE *)eptr->p_elem)->reset_for_each_step)
        ((FTRFMODE *)eptr->p_elem)->initialized = 0;
      break;
    case T_RAMPRF:
      if (includeRF) {
        ((RAMPRF *)eptr->p_elem)->Ts = 0;
        ((RAMPRF *)eptr->p_elem)->fiducial_seen = 0;
      }
      break;
    case T_KQUAD:
      if (includeRandom)
        ((KQUAD *)eptr->p_elem)->randomMultipoleData.randomized = 0;
      ((KQUAD *)eptr->p_elem)->totalMultipolesComputed = 0;
      break;
    case T_CCBEND:
      if (includeRandom)
        ((CCBEND *)eptr->p_elem)->randomMultipoleData.randomized = 0;
      ((CCBEND *)eptr->p_elem)->totalMultipolesComputed = 0;
      break;
    case T_KSEXT:
      if (includeRandom)
        ((KSEXT *)eptr->p_elem)->randomMultipoleData.randomized = 0;
      ((KSEXT *)eptr->p_elem)->totalMultipolesComputed = 0;
      break;
    case T_KOCT:
      if (includeRandom)
        ((KOCT *)eptr->p_elem)->randomMultipoleData.randomized = 0;
      ((KOCT *)eptr->p_elem)->totalMultipolesComputed = 0;
      break;
    case T_MATR:
      if (includeRF)
        ((MATR *)eptr->p_elem)->fiducialSeen = 0;
      break;
    case T_EMATRIX:
      if (includeRF)
        ((EMATRIX *)eptr->p_elem)->fiducialSeen = 0;
      break;
    case T_EHCOR:
      if (includeRandom)
        ((EHCOR *)eptr->p_elem)->multipolesRandomized = 0;
      break;
    case T_EVCOR:
      if (includeRandom)
        ((EVCOR *)eptr->p_elem)->multipolesRandomized = 0;
      break;
    case T_EHVCOR:
      if (includeRandom)
        ((EHVCOR *)eptr->p_elem)->multipolesRandomized = 0;
      break;
    default:
      break;
    }
    eptr = eptr->succ;
  }
#ifdef DEBUG_RESET
  printf("Special elements have been reset.\n");
  fflush(stdout);
#endif
  log_exit("reset_special_elements");
}

VMATRIX *stray_field_matrix(double length, double *lB, double *gB, double theta, long order, double p_central,
                            void *Wi) {
  /* factor in equation theta = CTHETA*B(T)*L(m)/(beta*gamma): */
#define CTHETA 5.8667907921396181e+02
  VMATRIX *M;
  double xkick, ykick, Bx, By;
  double rho;
#ifdef DEBUG_STRAY
  static FILE *fp = NULL;
  if (!fp) {
    fp = fopen_e("stray.erl", "w", 0);
    fprintf(fp, "SDDS1\n&column name=xKick type=double &end\n");
    fprintf(fp, "&column name=yKick type=double &end\n");
    fprintf(fp, "&column name=p type=double &end\n");
    fprintf(fp, "&column name=By type=double &end\n");
    fprintf(fp, "&column name=Bx type=double &end\n");
    fprintf(fp, "&data mode=ascii no_row_counts=1 &end\n");
  }
#endif

  if (Wi) {
    /* BLocalTotal = BLocal + Wi*GBlobal */
    MATRIX *Bg, *Bl;
    m_alloc(&Bg, 3, 1);
    m_alloc(&Bl, 3, 1);
    Bg->a[0][0] = gB[0];
    Bg->a[1][0] = gB[1];
    Bg->a[2][0] = gB[2];
    m_mult(Bl, (MATRIX *)Wi, Bg);
    Bx = Bl->a[0][0] + lB[0];
    By = Bl->a[1][0] + lB[1];
#ifdef DEBUG_STRAY
    m_show(Bg, "%10.3e", "Bg: \n", stdout);
    m_show(Wi, "%10.3e", "Wi: \n", stdout);
    m_show(Bl, "%10.3e", "Wi*Bg: \n", stdout);
    printf("Bx = %e, By = %e\n", Bx, By);
#endif
    m_free(&Bg);
    m_free(&Bl);
  } else {
    if (gB[0] || gB[1] || gB[2])
      bombElegant("to use global stray fields, you must do a floor coordinate computation", NULL);
    Bx = lB[0];
    By = lB[1];
  }
  if (By) {
    rho = p_central / (CTHETA * By);
    xkick = asin(length / rho);
  } else
    xkick = 0;
  if (Bx) {
    rho = p_central / (CTHETA * Bx);
    ykick = -asin(length / rho);
  } else
    ykick = 0;
#ifdef DEBUG_STRAY
  fprintf(fp, "%le %le %le %le %le\n", xkick, ykick, p_central, By, Bx);
  fflush(fp);
#endif

  M = hvcorrector_matrix(length, xkick, ykick, 0.0, 0.0, 1.0, 1.0, 0, order);

  return (M);
}

VMATRIX *rf_cavity_matrix(double length, double voltage, double frequency, double phase,
                          double *P_central, long order, long end1Focus, long end2Focus,
                          char *bodyFocusModel, long standingWave,
                          long change_p0, double Preference, ELEMENT_LIST *elem, RUN *run)
{
  VMATRIX *M, *Medge, *Mtot, *tmp;
  double *C, **R, ***T, dP, gamma, dgamma, dgammaMax;
  double cos_phase, sin_phase;
  double inverseF[2] = {0, 0};
  short useSRSModel=0, useTWModel=0, matrixMethod=0;

  if (voltage == 0)
    return drift_matrix(length, order);

  M = tmalloc(sizeof(*M));
  M->order = order > 2 ? 2 : order;
  initialize_matrices(M, M->order);
  T = M->T;
  R = M->R;
  C = M->C;

  if (*P_central <= 0) {
    printf("error: P_central = %g\n", *P_central);
    fflush(stdout);
    abort();
  }

  C[4] = length;
  voltage /= 1e6; /* convert to MV */
  sin_phase = sin(PI * phase / 180.0);
  cos_phase = cos(PI * phase / 180.0);
  dgamma = (dgammaMax = voltage / particleMassMV * particleRelSign) * sin_phase;
  gamma = sqrt(sqr(*P_central) + 1);
  dP = sqrt(sqr(gamma + dgamma) - 1) - *P_central;

  if (elem->type==T_RFCA || elem->type==T_RFCW)
    identifyRfcaBodyFocusModel(elem->p_elem, elem->type, &matrixMethod, &useSRSModel, &useTWModel);

  if (useTWModel) {
    double pSave;
    long cp0Save;
    pSave = run->p_central;
    cp0Save = run->always_change_p0;
    run->p_central = *P_central;
    run->always_change_p0 = change_p0;
    M = determineMatrix(run, elem, NULL, NULL);
    elem->Pref_output = run->p_central;
    run->p_central = pSave;
    run->always_change_p0 = cp0Save;
    return M;
  } else if (!useSRSModel) {
    /* old TRANSPORT matrix */
    double p00s, sinPhi0, cosPhi0, cotPhi0,
      cscPhi0, f1, sf1, f2, cos2phi0, f4, f2s, f5, logf4, sf5,
      dgMaxs, sqrt2, f5_1p5, cosPhi0s, sinPhi0s, lambdas, f6, logf6,
      sinPhi0c, dgMaxc, f1_1p5, f7, f7s;
    /* double PIs, f3; */
    double p00, dgMax, phi0, lambda, L;

    phi0 = phase * PI / 180;
    if (fabs(sin(phi0)) < 1e-6)
      phi0 += 2e-6;

    p00 = *P_central;
    dgMax = dgammaMax;
    lambda = c_mks / frequency;
    L = length;

    p00s = sqr(p00);
    sinPhi0 = sin(phi0);
    cosPhi0 = cos(phi0);
    cotPhi0 = 1 / tan(phi0);
    cscPhi0 = 1 / sin(phi0);
    f1 = 1 + p00s;
    sf1 = sqrt(f1);
    f2 = sf1 + dgMax * sinPhi0;
    /* f3 = p00s+sf1; */
    cos2phi0 = cos(2 * phi0);
    f4 = p00 + sf1;
    f2s = sqr(f2);
    f5 = -1 + f2s;
    logf4 = log(f4);
    sf5 = sqrt(f5);
    dgMaxs = sqr(dgMax);
    sqrt2 = sqrt(2);
    f5_1p5 = pow(f5, 1.5);
    cosPhi0s = sqr(cosPhi0);
    sinPhi0s = sqr(sinPhi0);
    lambdas = sqr(lambda);
    f6 = f2 + sf5;
    logf6 = log(f6);
    sinPhi0c = ipow3(sinPhi0);
    dgMaxc = ipow3(dgMax);
    f1_1p5 = pow(f1, 1.5);
    f7 = p00s + 2 * dgMax * sf1 * sinPhi0 + dgMaxs * sinPhi0s;
    f7s = sqr(f7);
    /* PIs = sqr(PI); */

    R[0][0] = R[2][2] = R[4][4] = 1;
    R[0][1] = R[2][3] = (L * p00 * cscPhi0 * (-logf4 + logf6)) / dgMax;
    R[1][1] = R[3][3] = p00 / sf5;
    R[5][4] = (2 * dgMax * PI * cosPhi0 * (f2)) / (lambda * (f5));
    R[5][5] = (p00s * (f2)) / (sf1 * (f5));

    if (order > 1) {
      T[0][4][1] = T[2][4][3] = (2 * L * p00 * PI * cotPhi0 *
                                 (dgMax + (cscPhi0 *
                                           (logf4 - logf6) *
                                           sqrt(dgMaxs + 2 * p00s - dgMaxs * cos2phi0 + 4 * dgMax * sf1 * sinPhi0)) /
                                            sqrt2)) /
                                (dgMax * lambda * sf5);

      T[0][5][1] = T[2][5][3] =
        (L * p00 * cscPhi0 * (-((p00 + p00s / sf1) / (f4)) - logf4 + logf6 + p00s / (sf1 * sf5))) / dgMax;

      T[1][4][1] = T[3][4][3] = (-2 * dgMax * p00 * PI * cosPhi0 * (f2)) / (lambda * f5_1p5);

      T[1][5][1] = T[3][5][3] = (dgMax * p00 * sinPhi0 * (2 + p00s + dgMax * sf1 * sinPhi0)) / (sf1 * f5_1p5);

      T[5][1][1] = T[5][3][3] =
        (-0.25 * L * p00s * cscPhi0 * (log(-1 + sf1) - 1. * log(1 + sf1) - 1. * log(-1 + f2) + log(1 + f2))) / dgMax;

      T[5][4][4] =
        (-19.739208802178716 * dgMax * (dgMax * cosPhi0s + sinPhi0 * (p00s * sf1 + dgMax * (2 + 3 * p00s) * sinPhi0 + 3 * dgMaxs * sf1 * sinPhi0s + dgMaxc * sinPhi0c))) / (lambdas * f7s);

      T[5][5][4] = (-2 * dgMax * p00s * PI * cosPhi0) / (lambda * sf1 * f7s);

      T[5][5][5] =
        (0.5 * dgMax * p00s * sinPhi0 * (2 + 3 * p00s + 3 * dgMax * sf1 * sinPhi0 + dgMaxs * sinPhi0s)) / (f1_1p5 * f7s);
    }
  } else {
    /* note that Rosenzweig and Serafini use gamma in places
       * where they should probably use momentum, but I'll keep
       * their expressions for now.
       */
    double alpha, sin_alpha, gammaf;
    gammaf = gamma + dgamma;
    if (fabs(sin_phase) > 1e-6)
      alpha = log(gammaf / gamma) / (2 * SQRT2 * sin_phase);
    else
      alpha = dgammaMax / gamma / (2 * SQRT2);
    R[0][0] = R[2][2] = cos(alpha);
    R[1][1] = R[3][3] = R[0][0] * gamma / gammaf;
    R[0][1] = R[2][3] = 2 * SQRT2 * gamma * length / dgammaMax * (sin_alpha = sin(alpha));
    R[1][0] = R[3][2] = -sin_alpha * dgammaMax / (length * gammaf * 2 * SQRT2);
    R[4][4] = 1;
    R[5][4] = (voltage / particleMassMV * particleRelSign) * cos_phase / (gamma + dgamma) * (PIx2 * frequency / c_mks);
    R[5][5] = 1 / (1 + dP / (*P_central));
  }

  if (length && (end1Focus || end2Focus)) {
    short end;
    if (end1Focus) {
      inverseF[0] = dgamma / (2 * length * gamma);
    }
    if (end2Focus)
      inverseF[1] = -dgamma / (2 * length * (gamma + dgamma));
    Medge = tmalloc(sizeof(*Medge));
    initialize_matrices(Medge, Medge->order = M->order);
    Medge->R[0][0] = Medge->R[1][1] = Medge->R[2][2] = Medge->R[3][3] =
      Medge->R[4][4] = Medge->R[5][5] = 1;
    Mtot = tmalloc(sizeof(*Mtot));
    initialize_matrices(Mtot, Mtot->order = M->order);
    for (end = 0; end < 2; end++) {
      if (inverseF[end]) {
        Medge->R[1][0] = Medge->R[3][2] = -inverseF[end];
      } else
        continue;
      if (end == 0) {
        concat_matrices(Mtot, M, Medge, CONCAT_EXCLUDE_S0);
      } else {
        concat_matrices(Mtot, Medge, M, CONCAT_EXCLUDE_S0);
      }
      tmp = Mtot;
      Mtot = M;
      M = tmp;
    }
    free_matrices(Medge);
    tfree(Medge);
    Medge = NULL;
    free_matrices(Mtot);
    tfree(Mtot);
    Mtot = NULL;
  }

  if (change_p0)
    Preference = *P_central + dP;

  if (!change_p0) {
    /* The matrix implicitly included acceleration. 
       * Must change the reference momentum of the matrix to exclude this.
       */
    Mtot = tmalloc(sizeof(*Mtot));
    Medge = tmalloc(sizeof(*Medge));
    initialize_matrices(Mtot, Mtot->order = M->order);
    initialize_matrices(Medge, Medge->order = M->order);
    Medge->R[0][0] = Medge->R[1][1] = Medge->R[2][2] = Medge->R[3][3] =
      Medge->R[4][4] = 1;
    Medge->R[5][5] = (*P_central + dP) / Preference;
    Medge->C[5] = (*P_central + dP - Preference) / Preference;
    concat_matrices(Mtot, Medge, M, CONCAT_EXCLUDE_S0);
    tmp = Mtot;
    Mtot = M;
    M = tmp;
    free_matrices(Medge);
    free_matrices(Mtot);
    tfree(Medge);
    tfree(Mtot);
    Medge = Mtot = NULL;
  }

  *P_central = Preference;

  return (M);
}

VMATRIX *twissTransformMatrix1(TWISS *twissWanted, TWISS *twissInput) {

  VMATRIX *M1, *M2, *M3, *Mtot;
  double beta1, beta2, alpha1, alpha2;

  M2 = tmalloc(sizeof(*M2));
  initialize_matrices(M2, M2->order = 1);
  M2->R[0][0] = M2->R[1][1] = M2->R[2][2] = M2->R[3][3] =
    M2->R[4][4] = M2->R[5][5] = 1;

  if (twissInput == NULL)
    return M2;

  beta1 = twissInput->betax;
  beta2 = twissWanted->betax;
  alpha1 = twissInput->alphax;
  alpha2 = twissWanted->alphax;
  M2->R[0][0] = beta2 / sqrt(beta1 * beta2);
  M2->R[1][0] = (alpha1 - alpha2) / sqrt(beta1 * beta2);
  M2->R[1][1] = beta1 / sqrt(beta1 * beta2);

  beta1 = twissInput->betay;
  beta2 = twissWanted->betay;
  alpha1 = twissInput->alphay;
  alpha2 = twissWanted->alphay;
  M2->R[2][2] = beta2 / sqrt(beta1 * beta2);
  M2->R[3][2] = (alpha1 - alpha2) / sqrt(beta1 * beta2);
  M2->R[3][3] = beta1 / sqrt(beta1 * beta2);

  M1 = tmalloc(sizeof(*M1));
  initialize_matrices(M1, M1->order = 1);
  M1->R[0][0] = M1->R[1][1] = M1->R[2][2] = M1->R[3][3] =
    M1->R[4][4] = M1->R[5][5] = 1;
  M1->R[0][5] = -twissInput->etax;
  M1->R[1][5] = -twissInput->etapx;
  M1->R[2][5] = -twissInput->etay;
  M1->R[3][5] = -twissInput->etapy;

  M3 = tmalloc(sizeof(*M3));
  initialize_matrices(M3, M3->order = 1);
  M3->R[0][0] = M3->R[1][1] = M3->R[2][2] = M3->R[3][3] =
    M3->R[4][4] = M3->R[5][5] = 1;
  M3->R[0][5] = twissWanted->etax;
  M3->R[1][5] = twissWanted->etapx;
  M3->R[2][5] = twissWanted->etay;
  M3->R[3][5] = twissWanted->etapy;

  Mtot = tmalloc(sizeof(*Mtot));
  initialize_matrices(Mtot, Mtot->order = 1);
  concat_matrices(Mtot, M2, M1, 0);
  concat_matrices(M1, M3, Mtot, 0);

  free_matrices(Mtot);
  free_matrices(M2);
  free_matrices(M3);
  free(Mtot);
  free(M2);
  free(M3);
  Mtot = M2 = M3 = NULL;

  return M1;
}

VMATRIX *twissTransformMatrix(TWISSELEMENT *twissWanted,
                              TWISS *twissInput) {
  return twissTransformMatrix1(&(twissWanted->twiss), twissInput);
}

/* thin lens for light optics */
VMATRIX *lightThinLensMatrix(LTHINLENS *ltl) {
  VMATRIX *M;
  double *C, **R;
  long i;

  M = tmalloc(sizeof(*M));
  M->order = 1;
  initialize_matrices(M, M->order);
  R = M->R;
  C = M->C;

  for (i = 0; i < 6; i++) {
    C[i] = 0;
    R[i][i] = 1;
  }
  if (ltl->fx)
    R[1][0] = -1 / ltl->fx;
  if (ltl->fy)
    R[3][2] = -1 / ltl->fy;
  if (ltl->tilt)
    tilt_matrices(M, ltl->tilt);
  if (ltl->pitch)
    pitch_matrices(M, ltl->pitch);
  if (ltl->yaw)
    yaw_matrices(M, ltl->yaw);
  if (ltl->dx || ltl->dy || ltl->dz)
    misalign_matrix(M, ltl->dx, ltl->dy, ltl->dz,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    misalignmentMethod);
  return M;
}

/* mirror for light optics */
VMATRIX *lightMirrorMatrix(LMIRROR *lm) {
  VMATRIX *M;
  double *C, **R;
  long i;

  M = tmalloc(sizeof(*M));
  M->order = 1;
  initialize_matrices(M, M->order);
  R = M->R;
  C = M->C;

  for (i = 0; i < 6; i++) {
    C[i] = 0;
    R[i][i] = 1;
  }
  if (lm->Rx)
    R[1][0] = -2 / (lm->Rx * cos(lm->theta));
  if (lm->Ry)
    R[3][2] = -2 / (lm->Ry / cos(lm->theta));
  if (lm->tilt)
    tilt_matrices(M, lm->tilt);
  if (lm->pitch)
    pitch_matrices(M, lm->pitch);
  if (lm->yaw)
    yaw_matrices(M, lm->yaw);
  if (lm->dx || lm->dy || lm->dz)
    misalign_matrix(M, lm->dx, lm->dy, lm->dz,
                    0.0, 0.0, 0.0, 0.0, 2 * PI - 2 * lm->theta, 0.0,
                    misalignmentMethod);
  return M;
}

/* explicit matrix input from user */
VMATRIX *matrixFromExplicitMatrix(EMATRIX *emat, long order) {
  long i, j, k;
  VMATRIX *M;
  double *C, **R, ***T;

  if (emat->order != 0)
    order = emat->order;
  M = tmalloc(sizeof(*M));
  initialize_matrices(M, M->order = order);
  C = M->C;
  R = M->R;
  T = M->T;

  for (i = 0; i < 6; i++)
    C[i] = emat->C[i];

  if (order >= 1)
    for (i = 0; i < 6; i++)
      for (j = 0; j < 6; j++)
        R[i][j] = emat->R[i][j];

  if (order >= 2)
    for (i = 0; i < 6; i++)
      for (j = 0; j < 6; j++)
        for (k = 0; k <= j; k++)
          T[i][j][k] = emat->T[i][j][k];

  if (emat->tilt)
    tilt_matrices(M, emat->tilt);
  if (emat->pitch)
    pitch_matrices(M, emat->pitch);
  if (emat->yaw)
    yaw_matrices(M, emat->yaw);
  if (emat->dx || emat->dy || emat->dz)
    misalign_matrix(M, emat->dx, emat->dy, emat->dz,
                    0.0, 0.0, 0.0, 0.0, emat->angle, C[5],
                    misalignmentMethod);

  return M;
}

/* ilmatrix */
VMATRIX *matrixForILMatrix(ILMATRIX *ilmat, long order) {
  long i, offset, plane;
  VMATRIX *M1;
  double tune2pi, R11, R22, R12, sin_phi, cos_phi, alpha, beta, eta, etap;

  if (order < 1)
    order = 1;
  M1 = tmalloc(sizeof(*M1));
  initialize_matrices(M1, M1->order = order);
  for (i = 0; i < 6; i++)
    M1->C[i] = 0;

  for (plane = 0; plane < 2; plane++) {
    tune2pi = PIx2 * ilmat->tune[plane];
    offset = 2 * plane;
    beta = ilmat->beta[plane];
    eta = ilmat->eta[0 + offset];
    etap = ilmat->eta[1 + offset];
    /* R11=R22 or R33=R44 */
    sin_phi = sin(tune2pi);
    cos_phi = cos(tune2pi);
    alpha = ilmat->alpha[plane];
    /* R11 or R33 */
    R11 = M1->R[0 + offset][0 + offset] = cos_phi + alpha * sin_phi;
    /* R22 or R44 */
    R22 = M1->R[1 + offset][1 + offset] = cos_phi - alpha * sin_phi;
    /* R12 or R34 */
    if ((R12 = M1->R[0 + offset][1 + offset] = beta * sin_phi)) {
      /* R21 or R43 */
      M1->R[1 + offset][0 + offset] = (R11 * R22 - 1) / R12;
    } else {
      printf("ILMATRIX problem: divide by zero\n");
      return NULL;
    }
    /* R16 or R36 */
    M1->R[0 + offset][5] = eta - eta * cos_phi - (alpha * eta + beta * etap) * sin_phi;
    /* R26 or R46 */
    M1->R[1 + offset][5] = etap - etap * cos_phi + (eta + sqr(alpha) * eta + alpha * beta * etap) * sin_phi / beta;
    /* R51 or R53 */
    M1->R[4][0 + offset] = -etap + etap * cos_phi + (eta + sqr(alpha) * eta + alpha * beta * etap) * sin_phi / beta;
    /* R52 or R54 */
    M1->R[4][1 + offset] = eta - eta * cos_phi + (alpha * eta + beta * etap) * sin_phi;
  }
  /* Other path-length terms */
  M1->R[4][4] = 1;
  M1->R[4][5] = ilmat->alphac[0] * ilmat->length - M1->R[4][0] * ilmat->eta[0] - M1->R[4][1] * ilmat->eta[1] - M1->R[4][2] * ilmat->eta[2] - M1->R[4][3] * ilmat->eta[3];

  M1->R[5][5] = 1;
  M1->C[4] = ilmat->length;

  if (ilmat->tilt)
    tilt_matrices(M1, ilmat->tilt);

  return M1;
}

VMATRIX *rfdf_matrix(RFDF *rfdf, double pReference) {
  VMATRIX *M;
  double *C, **R, ***T;
  double k, theta, omega, L, phi, beta;
  double cphi, sphi;

  if (rfdf->voltage == 0 || rfdf->fse == -1)
    return drift_matrix(rfdf->length, 1);

  beta = pReference / sqrt(sqr(pReference) + 1);
  /*  theta = (particleCharge*rfdf->voltage*(1+rfdf->fse))/(particleMass*sqr(c_mks)*pReference*beta); */
  theta = (particleCharge * rfdf->voltage * (1 + rfdf->fse)) / (particleMass * sqr(c_mks) * pReference);
  omega = rfdf->frequency * PIx2;
  k = omega / c_mks;
  L = rfdf->length;

  if (L == 0) {
    phi = rfdf->phase * PI / 180.;
    cphi = cos(phi);
    sphi = sin(phi);
    M = tmalloc(sizeof(*M));
    M->order = 1;
    initialize_matrices(M, M->order);

    R = M->R;
    C = M->C;
    C[1] = theta * cphi;
    C[5] = sqrt(sqr(C[1]) + 1) - 1;
    R[0][0] = R[1][1] = R[2][2] = R[3][3] = R[4][4] = R[5][5] = 1;
    R[1][0] = -cphi * sphi * k * sqr(theta);
    R[1][4] = -k * sphi * theta / beta;
    R[1][5] = -theta * cphi;
    R[5][5] = 1 / sqrt(1 + sqr(cphi * theta));
    R[5][0] = k * sphi * theta * R[5][5];
    R[5][1] = theta * cphi * R[5][5];
    R[5][4] = -k * cphi * sphi * sqr(theta) * R[5][5] / beta;

    if (rfdf->dx || rfdf->dy || rfdf->dz || rfdf->tilt)
      misalign_matrix(M, rfdf->dx, rfdf->dy, rfdf->dz,
                      0.0, 0.0, 0.0, rfdf->tilt,  0.0, rfdf->length,
                      misalignmentMethod);
    return (M);
  } else {
    VMATRIX *Mt, *Mc, *Md, *Ms, *Mtmp;
    double dz, z, phase;
    long i, n;
    double cphi2, sphi2, k2, theta2;
    double theta3;
    double theta4;

    if ((n = (rfdf->length / (PIx2 / k)) * 100) < 10)
      n = 10;
    dz = rfdf->length / (2 * n);
    Md = drift_matrix(dz, 2);
    Mt = drift_matrix(0, 2);
    Ms = drift_matrix(0, 2);
    Mc = drift_matrix(0, 2);
    C = Mc->C;
    R = Mc->R;
    T = Mc->T;

    theta /= n;
    theta2 = theta * theta;
    theta3 = theta * theta2;
    theta4 = theta2 * theta2;
    k2 = k * k;
    z = -rfdf->length / (2 * beta) + dz;
    Mt->C[4] = -rfdf->length / 2;
    for (i = 0; i < n; i++) {
      if (rfdf->standingWave)
        phase = rfdf->phase * PI / 180 + k * (Mt->C[4] + dz) / beta;
      else
        phase = rfdf->phase * PI / 180 + k * (Mt->C[4] + dz) * (1 / beta - 1);

      cphi = cos(phase);
      sphi = sin(phase);
      cphi2 = cphi * cphi;
      sphi2 = sphi * sphi;

      C[1] = cphi * theta;
      C[5] = sqrt(cphi2 * theta2 + 1) - 1;

      R[1][0] = -k * cphi * sphi * theta2;
      R[1][4] = -k * sphi * theta / beta;
      R[1][5] = -cphi * theta;

      R[5][0] = k * sphi * theta / sqrt(cphi2 * theta2 + 1);
      R[5][1] = cphi * theta / sqrt(cphi2 * theta2 + 1);

      R[5][4] = -k * cphi * sphi * theta2 / sqrt(cphi2 * theta2 + 1) / beta;

      R[5][5] = 1 / sqrt(cphi2 * theta2 + 1);

      T[1][0][0] = 2 * k2 * cphi * sphi2 * theta3;
      T[1][1][0] = -k * sphi * theta;
      T[1][1][1] = cphi * theta;
      T[1][4][0] = k2 * sphi2 * theta2 - k2 * cphi2 * theta2;
      T[1][4][4] = -k2 * cphi * theta;
      T[1][5][0] = 2 * k * cphi * sphi * theta2;
      T[1][5][4] = k * sphi * theta;
      T[1][5][5] = 2 * cphi * theta;
      T[5][0][0] = k2 * sphi2 * theta2 / sqrt(cphi2 * theta2 + 1) - k2 * sphi2 * theta2 / pow(cphi2 * theta2 + 1, 1.5);
      T[5][1][0] = -k * cphi * sphi * theta2 / pow(cphi2 * theta2 + 1, 1.5);
      T[5][1][1] = -cphi2 * theta2 / pow(cphi2 * theta2 + 1, 1.5);
      T[5][4][0] = k2 * cphi * theta / sqrt(cphi2 * theta2 + 1) + k2 * cphi * sphi2 * theta3 / pow(cphi2 * theta2 + 1, 1.5);
      T[5][4][1] = k * cphi2 * sphi * theta3 / pow(cphi2 * theta2 + 1, 1.5) - k * sphi * theta / sqrt(cphi2 * theta2 + 1);
      T[5][4][4] = (2 * k2 * sphi2 * theta2 - 2 * k2 * cphi2 * theta2) / (2 * sqrt(cphi2 * theta2 + 1)) - k2 * cphi2 * sphi2 * theta4 / pow(cphi2 * theta2 + 1, 1.5);
      T[5][5][0] = k * sphi * theta / sqrt(cphi2 * theta2 + 1) - k * sphi * theta / pow(cphi2 * theta2 + 1, 1.5);
      T[5][5][1] = cphi * theta / sqrt(cphi2 * theta2 + 1) - cphi * theta / pow(cphi2 * theta2 + 1, 1.5);
      T[5][5][4] = k * cphi * sphi * theta2 / pow(cphi2 * theta2 + 1, 1.5);
      T[5][5][5] = 1 / sqrt(cphi2 * theta2 + 1) - 1 / pow(cphi2 * theta2 + 1, 1.5);

      concat_matrices(Ms, Md, Mt, 0);
      Ms->C[4] -= z;
      concat_matrices(Mt, Mc, Ms, 0);
      Mt->C[4] += z;
      concat_matrices(Ms, Md, Mt, 0);
      Mtmp = Ms;
      Ms = Mt;
      Mt = Mtmp;
      z += 2 * dz;
    }
    free_matrices(Mc);
    free_matrices(Ms);
    free_matrices(Md);
    free(Mc);
    free(Ms);
    free(Md);
    Mc = Ms = Md = NULL;
    Mt->C[4] += rfdf->length / 2;
    null_matrices(Mt, EXCLUDE_R + EXCLUDE_C);
    Mt->order = 1;

    if (rfdf->dx || rfdf->dy || rfdf->dz || rfdf->tilt)
      misalign_matrix(Mt, rfdf->dx, rfdf->dy, rfdf->dz,
                      0.0, 0.0, 0.0, rfdf->tilt, 0.0, rfdf->length,
                      misalignmentMethod);
    return (Mt);
  }
}

VMATRIX *sextupoleFringeMatrix(double K2, double length, long maxOrder, long side) {
  VMATRIX *M;
  double *C, **R, ***T, ****U;

  M = tmalloc(sizeof(*M));
  initialize_matrices(M, M->order = MIN(3, maxOrder));
  R = M->R;
  C = M->C;

  R[0][0] = R[1][1] = R[2][2] = R[3][3] = R[4][4] = R[5][5] = 1;
  C[4] = R[0][1] = R[2][3] = length;

  if (side == -1) {
    /* entrance */
    if (M->order >= 2) {
      T = M->T;
      T[0][0][0] = -(K2 * ipow2(length)) / 12.;
      T[0][1][0] = -(K2 * ipow3(length)) / 12.;
      T[0][1][1] = -(K2 * ipow4(length)) / 40.;
      T[0][2][2] = (K2 * ipow2(length)) / 12.;
      T[0][3][2] = (K2 * ipow3(length)) / 12.;
      T[0][3][3] = (K2 * ipow4(length)) / 40.;
      T[1][0][0] = -(K2 * length) / 4.;
      T[1][1][0] = -(K2 * ipow2(length)) / 3.;
      T[1][1][1] = -(K2 * ipow3(length)) / 8.;
      T[1][2][2] = (K2 * length) / 4.;
      T[1][3][2] = (K2 * ipow2(length)) / 3.;
      T[1][3][3] = (K2 * ipow3(length)) / 8.;
      T[2][2][0] = (K2 * ipow2(length)) / 6.;
      T[2][2][1] = (K2 * ipow3(length)) / 12.;
      T[2][3][0] = (K2 * ipow3(length)) / 12.;
      T[2][3][1] = (K2 * ipow4(length)) / 20.;
      T[3][2][0] = (K2 * length) / 2.;
      T[3][2][1] = (K2 * ipow2(length)) / 3.;
      T[3][3][0] = (K2 * ipow2(length)) / 3.;
      T[3][3][1] = (K2 * ipow3(length)) / 4.;
      T[4][1][1] = length / 2.;
      T[4][3][3] = length / 2.;
      if (M->order >= 3) {
        U = M->Q;
        U[0][0][0][0] = (ipow2(K2) * ipow4(length)) / 360.;
        U[0][1][0][0] = (ipow2(K2) * ipow5(length)) / 252.;
        U[0][1][1][0] = (13 * ipow2(K2) * ipow6(length)) / 6720.;
        U[0][1][1][1] = (ipow2(K2) * ipow7(length)) / 2880.;
        U[0][2][2][0] = (ipow2(K2) * ipow4(length)) / 360.;
        U[0][3][2][0] = (ipow2(K2) * ipow5(length)) / 252.;
        U[0][3][2][1] = (ipow2(K2) * ipow6(length)) / 1120.;
        U[0][3][3][0] = (ipow2(K2) * ipow6(length)) / 960.;
        U[0][3][3][1] = (ipow2(K2) * ipow7(length)) / 2880.;
        U[0][5][0][0] = (K2 * ipow2(length)) / 12.;
        U[0][5][1][0] = (K2 * ipow3(length)) / 12.;
        U[0][5][1][1] = (K2 * ipow4(length)) / 40.;
        U[0][5][2][2] = -(K2 * ipow2(length)) / 12.;
        U[0][5][3][2] = -(K2 * ipow3(length)) / 12.;
        U[0][5][3][3] = -(K2 * ipow4(length)) / 40.;
        U[1][0][0][0] = (ipow2(K2) * ipow3(length)) / 60.;
        U[1][1][0][0] = (ipow2(K2) * ipow4(length)) / 36.;
        U[1][1][1][0] = (13 * ipow2(K2) * ipow5(length)) / 840.;
        U[1][1][1][1] = (ipow2(K2) * ipow6(length)) / 320.;
        U[1][2][2][0] = (ipow2(K2) * ipow3(length)) / 60.;
        U[1][3][2][0] = (ipow2(K2) * ipow4(length)) / 36.;
        U[1][3][2][1] = (ipow2(K2) * ipow5(length)) / 140.;
        U[1][3][3][0] = (ipow2(K2) * ipow5(length)) / 120.;
        U[1][3][3][1] = (ipow2(K2) * ipow6(length)) / 320.;
        U[1][5][0][0] = (K2 * length) / 4.;
        U[1][5][1][0] = (K2 * ipow2(length)) / 3.;
        U[1][5][1][1] = (K2 * ipow3(length)) / 8.;
        U[1][5][2][2] = -(K2 * length) / 4.;
        U[1][5][3][2] = -(K2 * ipow2(length)) / 3.;
        U[1][5][3][3] = -(K2 * ipow3(length)) / 8.;
        U[2][2][0][0] = (ipow2(K2) * ipow4(length)) / 360.;
        U[2][2][1][0] = (ipow2(K2) * ipow5(length)) / 252.;
        U[2][2][1][1] = (ipow2(K2) * ipow6(length)) / 960.;
        U[2][2][2][2] = (ipow2(K2) * ipow4(length)) / 360.;
        U[2][3][1][0] = (ipow2(K2) * ipow6(length)) / 1120.;
        U[2][3][1][1] = (ipow2(K2) * ipow7(length)) / 2880.;
        U[2][3][2][2] = (ipow2(K2) * ipow5(length)) / 252.;
        U[2][3][3][2] = (13 * ipow2(K2) * ipow6(length)) / 6720.;
        U[2][3][3][3] = (ipow2(K2) * ipow7(length)) / 2880.;
        U[2][5][2][0] = -(K2 * ipow2(length)) / 6.;
        U[2][5][2][1] = -(K2 * ipow3(length)) / 12.;
        U[2][5][3][0] = -(K2 * ipow3(length)) / 12.;
        U[2][5][3][1] = -(K2 * ipow4(length)) / 20.;
        U[3][2][0][0] = (ipow2(K2) * ipow3(length)) / 60.;
        U[3][2][1][0] = (ipow2(K2) * ipow4(length)) / 36.;
        U[3][2][1][1] = (ipow2(K2) * ipow5(length)) / 120.;
        U[3][2][2][2] = (ipow2(K2) * ipow3(length)) / 60.;
        U[3][3][1][0] = (ipow2(K2) * ipow5(length)) / 140.;
        U[3][3][1][1] = (ipow2(K2) * ipow6(length)) / 320.;
        U[3][3][2][2] = (ipow2(K2) * ipow4(length)) / 36.;
        U[3][3][3][2] = (13 * ipow2(K2) * ipow5(length)) / 840.;
        U[3][3][3][3] = (ipow2(K2) * ipow6(length)) / 320.;
        U[3][5][2][0] = -(K2 * length) / 2.;
        U[3][5][2][1] = -(K2 * ipow2(length)) / 3.;
        U[3][5][3][0] = -(K2 * ipow2(length)) / 3.;
        U[3][5][3][1] = -(K2 * ipow3(length)) / 4.;
      }
    }
  } else {
    /* exit */
    if (M->order >= 2) {
      T = M->T;
      T[0][0][0] = -(K2 * ipow2(length)) / 6.;
      T[0][1][0] = -(K2 * ipow3(length)) / 12.;
      T[0][1][1] = -(K2 * ipow4(length)) / 60.;
      T[0][2][2] = (K2 * ipow2(length)) / 6.;
      T[0][3][2] = (K2 * ipow3(length)) / 12.;
      T[0][3][3] = (K2 * ipow4(length)) / 60.;
      T[1][0][0] = -(K2 * length) / 4.;
      T[1][1][0] = -(K2 * ipow2(length)) / 6.;
      T[1][1][1] = -(K2 * ipow3(length)) / 24.;
      T[1][2][2] = (K2 * length) / 4.;
      T[1][3][2] = (K2 * ipow2(length)) / 6.;
      T[1][3][3] = (K2 * ipow3(length)) / 24.;
      T[2][2][0] = (K2 * ipow2(length)) / 3.;
      T[2][2][1] = (K2 * ipow3(length)) / 12.;
      T[2][3][0] = (K2 * ipow3(length)) / 12.;
      T[2][3][1] = (K2 * ipow4(length)) / 30.;
      T[3][2][0] = (K2 * length) / 2.;
      T[3][2][1] = (K2 * ipow2(length)) / 6.;
      T[3][3][0] = (K2 * ipow2(length)) / 6.;
      T[3][3][1] = (K2 * ipow3(length)) / 12.;
      T[4][1][1] = length / 2.;
      T[4][3][3] = length / 2.;
      if (M->order >= 3) {
        U = M->Q;
        U[0][0][0][0] = (ipow2(K2) * ipow4(length)) / 144.;
        U[0][1][0][0] = (3 * ipow2(K2) * ipow5(length)) / 560.;
        U[0][1][1][0] = (3 * ipow2(K2) * ipow6(length)) / 2240.;
        U[0][1][1][1] = (ipow2(K2) * ipow7(length)) / 6720.;
        U[0][2][2][0] = (ipow2(K2) * ipow4(length)) / 144.;
        U[0][2][2][1] = -(ipow2(K2) * ipow5(length)) / 720.;
        U[0][3][2][0] = (17 * ipow2(K2) * ipow5(length)) / 2520.;
        U[0][3][2][1] = (ipow2(K2) * ipow6(length)) / 2016.;
        U[0][3][3][0] = (17 * ipow2(K2) * ipow6(length)) / 20160.;
        U[0][3][3][1] = (ipow2(K2) * ipow7(length)) / 6720.;
        U[0][5][0][0] = (K2 * ipow2(length)) / 6.;
        U[0][5][1][0] = (K2 * ipow3(length)) / 12.;
        U[0][5][1][1] = (K2 * ipow4(length)) / 60.;
        U[0][5][2][2] = -(K2 * ipow2(length)) / 6.;
        U[0][5][3][2] = -(K2 * ipow3(length)) / 12.;
        U[0][5][3][3] = -(K2 * ipow4(length)) / 60.;
        U[1][0][0][0] = (ipow2(K2) * ipow3(length)) / 60.;
        U[1][1][0][0] = (11 * ipow2(K2) * ipow4(length)) / 720.;
        U[1][1][1][0] = (11 * ipow2(K2) * ipow5(length)) / 2520.;
        U[1][1][1][1] = (11 * ipow2(K2) * ipow6(length)) / 20160.;
        U[1][2][2][0] = (ipow2(K2) * ipow3(length)) / 60.;
        U[1][2][2][1] = -(ipow2(K2) * ipow4(length)) / 240.;
        U[1][3][2][0] = (7 * ipow2(K2) * ipow4(length)) / 360.;
        U[1][3][2][1] = (ipow2(K2) * ipow5(length)) / 630.;
        U[1][3][3][0] = (ipow2(K2) * ipow5(length)) / 360.;
        U[1][3][3][1] = (11 * ipow2(K2) * ipow6(length)) / 20160.;
        U[1][5][0][0] = (K2 * length) / 4.;
        U[1][5][1][0] = (K2 * ipow2(length)) / 6.;
        U[1][5][1][1] = (K2 * ipow3(length)) / 24.;
        U[1][5][2][2] = -(K2 * length) / 4.;
        U[1][5][3][2] = -(K2 * ipow2(length)) / 6.;
        U[1][5][3][3] = -(K2 * ipow3(length)) / 24.;
        U[2][2][0][0] = (ipow2(K2) * ipow4(length)) / 144.;
        U[2][2][1][0] = (17 * ipow2(K2) * ipow5(length)) / 2520.;
        U[2][2][1][1] = (17 * ipow2(K2) * ipow6(length)) / 20160.;
        U[2][2][2][2] = (ipow2(K2) * ipow4(length)) / 144.;
        U[2][3][0][0] = -(ipow2(K2) * ipow5(length)) / 720.;
        U[2][3][1][0] = (ipow2(K2) * ipow6(length)) / 2016.;
        U[2][3][1][1] = (ipow2(K2) * ipow7(length)) / 6720.;
        U[2][3][2][2] = (3 * ipow2(K2) * ipow5(length)) / 560.;
        U[2][3][3][2] = (3 * ipow2(K2) * ipow6(length)) / 2240.;
        U[2][3][3][3] = (ipow2(K2) * ipow7(length)) / 6720.;
        U[2][5][2][0] = -(K2 * ipow2(length)) / 3.;
        U[2][5][2][1] = -(K2 * ipow3(length)) / 12.;
        U[2][5][3][0] = -(K2 * ipow3(length)) / 12.;
        U[2][5][3][1] = -(K2 * ipow4(length)) / 30.;
        U[3][2][0][0] = (ipow2(K2) * ipow3(length)) / 60.;
        U[3][2][1][0] = (7 * ipow2(K2) * ipow4(length)) / 360.;
        U[3][2][1][1] = (ipow2(K2) * ipow5(length)) / 360.;
        U[3][2][2][2] = (ipow2(K2) * ipow3(length)) / 60.;
        U[3][3][0][0] = -(ipow2(K2) * ipow4(length)) / 240.;
        U[3][3][1][0] = (ipow2(K2) * ipow5(length)) / 630.;
        U[3][3][1][1] = (11 * ipow2(K2) * ipow6(length)) / 20160.;
        U[3][3][2][2] = (11 * ipow2(K2) * ipow4(length)) / 720.;
        U[3][3][3][2] = (11 * ipow2(K2) * ipow5(length)) / 2520.;
        U[3][3][3][3] = (11 * ipow2(K2) * ipow6(length)) / 20160.;
        U[3][5][2][0] = -(K2 * length) / 2.;
        U[3][5][2][1] = -(K2 * ipow2(length)) / 6.;
        U[3][5][3][0] = -(K2 * ipow2(length)) / 6.;
        U[3][5][3][1] = -(K2 * ipow3(length)) / 12.;
      }
    }
  }

  /* 
  if (side==-1) 
    print_matrices(stdout, "entrance fringe matrix:\n", M);
  else
    print_matrices(stdout, "exit fringe matrix:\n", M);
    */

  return M;
}

VMATRIX *mult_matrix(MULT *mult, double P, long defaultOrder) {
  VMATRIX *M;
  double length, H, KnL;
  long matrixOrder;
  matrixOrder = mult->matrixOrder ? mult->matrixOrder: defaultOrder;
  if ((length = mult->length) <= 1e-24)
    length = 1e-24;
  H = P * me_mks * c_mks / e_mks;
  if (mult->bore) {
    KnL = dfactorial(mult->order) * mult->BTipL * mult->factor / (H * ipow(mult->bore, mult->order));
  } else {
    KnL = mult->KnL * mult->factor;
  }
  switch (mult->order) {
  case 0: /* dipole */
    M = bend_matrix(length, KnL / length, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, matrixOrder, matrixOrder, 0, 0,
                    0.0, 0.0);
    break;
  case 1: /* quadrupole */
    M = quadrupole_matrix(KnL / length, length, matrixOrder, 0.0, 0.0, 0.0, 0.0, 0.0, NULL, 0.0, -1.0, NULL, NULL, 0);
    break;
  case 2: /* sextupole */
    M = sextupole_matrix(KnL / length, 0, 0, length, matrixOrder, 0.0, 0.0, 0.0, 0.0);
    break;
  case 3: /* octupole */
    M = octupole_matrix(KnL / length, length, matrixOrder, 0.0);
    break;
  default: /* something else, use drift matrix */
    M = drift_matrix(length, matrixOrder);
    break;
  }
  return M;
}

VMATRIX *interpolateMatrixWithIdentityMatrix(VMATRIX *M0, double fraction, long order) {
  VMATRIX *M;
  long i, j, k, l;

  M = tmalloc(sizeof(*M));
  M->order = order;
  initialize_matrices(M, M->order);

  for (i=0; i<6; i++)
    M->C[i] = M0->C[i]*fraction;
  M->C[4] = M0->C[4];

  for (i = 0; i < 6; i++)
    for (j = 0; j < 6; j++)
      M->R[i][j] = (i == j ? 1 - fraction : 0) + M0->R[i][j] * fraction;

  if (order > 1) {
    for (i = 0; i < 6; i++)
      for (j = 0; j < 6; j++)
        for (k = 0; k <= j; k++)
          M->T[i][j][k] = M0->T[i][j][k] * fraction;
    if (order > 2) {
      for (i = 0; i < 6; i++)
        for (j = 0; j < 6; j++)
          for (k = 0; k <= j; k++)
            for (l = 0; l <= k; l++)
              M->Q[i][j][k][l] = M0->Q[i][j][k][l] * fraction;
    }
  }

  return (M);
}

