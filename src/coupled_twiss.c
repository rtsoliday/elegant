/*************************************************************************\
 * Copyright (c) 2006 The University of Chicago, as Operator of Argonne
 * National Laboratory.
 * Copyright (c) 2006 The Regents of the University of California, as
 * Operator of Los Alamos National Laboratory.
 * This file is distributed subject to a Software License Agreement found
 * in the file LICENSE that is included with this distribution.
\*************************************************************************/

/* file: coupled_twiss.c
 * purpose: computation of coupled Twiss parameters
 *
 * Vadim Sajaev, 2006.
 * Incorporated into elegant by Michael Borland
 */
#include "mdb.h"
#include "track.h"
#include "coupled_twiss.h"

#if defined(LAPACK) || defined(CLAPACK) || defined(MKL)
  int dgeev_(char *JOBVL, char *JOBVR, int *N, double *A,
         int *LDA, double *WR, double *WI, double *VL,
         int *LDVL, double *VR, int *LDVR, double *work,
         int *lwork, int *info);
#endif
void store_fitpoint_ctwiss_parameters(MARK *fpt, char *name, long occurence,
                                      double betax1, double betax2,
                                      double betay1, double betay2,
                                      double etax, double etay,
                                      double tilt);

static SDDS_DATASET SDDScoupled;
static short SDDScoupledInitialized = 0;
static short initialized = 0;

void SortEigenvalues(double *WR, double *WI, double *VR, int matDim, int eigenModesNumber, int verbosity);
int GetMaxIndex(double *V, int N);
void GetAMatrix(double *V, double *transferMatrix, double *A, int *eigenModesNumber, int *matDim);
void MatrixPrintout(double *AA, int *NA, int *MA, int dim);
void MatrixProduct(int *N1, int *M1, double *T1, int *N2, int *M2, double *T2, double *T3);

void setup_coupled_twiss_output(
                                NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline, long *do_coupled_twiss_output,
                                long default_order) {
  /* process namelist input */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&coupled_twiss_output, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &coupled_twiss_output);

#if USE_MPI
  if (!writePermitted)
    filename = NULL;
#endif

  if (filename)
    filename = compose_filename(filename, run->rootname);

  *do_coupled_twiss_output = output_at_each_step;

  if (!emittances_from_twiss_command && emit_x == 0 && sigma_dp == 0)
    bombElegant("supply emit_x, sigma_dp, or set emittances_from_twiss_command=1", NULL);
  if (!emittances_from_twiss_command) {
    if (emit_x < 0)
      bombElegant("emit_x < 0", NULL);
    if (sigma_dp < 0)
      bombElegant("sigma_dp < 0", NULL);
    if (emittance_ratio < 0)
      bombElegant("emittance_ratio < 0", NULL);
  }

  if (filename) {
    if (!SDDS_InitializeOutputElegant(&SDDScoupled, SDDS_BINARY, 0, NULL, NULL, filename) ||
        !SDDS_DefineSimpleColumn(&SDDScoupled, "ElementName", NULL, SDDS_STRING) ||
        !SDDS_DefineSimpleColumn(&SDDScoupled, "s", "m", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleColumn(&SDDScoupled, "Sx", "m", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleColumn(&SDDScoupled, "Sxp", "", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleColumn(&SDDScoupled, "Sy", "m", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleColumn(&SDDScoupled, "Syp", "", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleColumn(&SDDScoupled, "xyTilt", "", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleColumn(&SDDScoupled, "Ss", "m", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleColumn(&SDDScoupled, "betax1", "m", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleColumn(&SDDScoupled, "betax2", "m", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleColumn(&SDDScoupled, "betay1", "m", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleColumn(&SDDScoupled, "betay2", "m", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleColumn(&SDDScoupled, "etax", "m", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleColumn(&SDDScoupled, "etay", "m", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDScoupled, "nux", "", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDScoupled, "nuy", "", SDDS_DOUBLE)) {
      printf("Unable to set up file %s\n", filename);
      fflush(stdout);
      SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }

    if (output_sigma_matrix) {
      long maxDimension, i, j;
      char name[100], units[10];
      if (calculate_3d_coupling)
        maxDimension = 6;
      else
        maxDimension = 4;
      for (i = 0; i < maxDimension; i++)
        for (j = i; j < maxDimension; j++) {
          if ((i == 0 || i == 2 || i == 4) && (j == 0 || j == 2 || j == 4))
            strcpy_ss(units, "m$a2$n");
          else if ((!(i == 0 || i == 2 || i == 4) && (j == 0 || j == 2 || j == 4)) ||
                   ((i == 0 || i == 2 || i == 4) && !(j == 0 || j == 2 || j == 4)))
            strcpy_ss(units, "m");
          else
            strcpy_ss(units, "");
          sprintf(name, "S%ld%ld", i + 1, j + 1);
          if (!SDDS_DefineSimpleColumn(&SDDScoupled, name, units, SDDS_DOUBLE)) {
            printf("Unable to set up file %s\n", filename);
            fflush(stdout);
            SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
            exitElegant(1);
          }
        }
    }

    if (!SDDS_WriteLayout(&SDDScoupled)) {
      printf("Unable to set up file %s\n", filename);
      fflush(stdout);
      SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
    SDDScoupledInitialized = 1;
  }

  initialized = 1;
}

int run_coupled_twiss_output(RUN *run, LINE_LIST *beamline, double *starting_coord) {
  char JOBVL, JOBVR;
  int N, LDA, LDVL, LDVR, lwork, info, i, j, k;
  double A[36], WR[6], WI[6], VL[36], VR[36], work[1000];
  double emit[3], Norm[3], Vnorm[36];
  double Amatrix[108], SigmaMatrix[6][6];
  int matDim, eigenModesNumber;
  double transferMatrix[36];
  VMATRIX *M, *M1;
  double **R;
  ELEMENT_LIST *eptr, *eptr0;
  long nElements, lastNElements, iElement;
  double betax1, betax2, betay1, betay2, etax, etay, tilt;
  double nux, nuy;
  static long cnuMemory[2] = {-1, -1};
  
  if (!initialized)
    return 0;
  if (cnuMemory[0]==-1) {
    cnuMemory[0] = rpn_create_mem("cnux", 0);
    cnuMemory[1] = rpn_create_mem("cnuy", 0);
  }
      
  if (verbosity > 2)
    printf("\n* Computing coupled sigma matrix\n");
  beamline->flags |= BEAMLINE_MATRICES_NEEDED;
  if (emittances_from_twiss_command) {
    if (!(beamline->flags & BEAMLINE_TWISS_DONE)) {
      printWarning("coupled_twiss_output has emittances_from_twiss_command=1, but twiss calculations not seen", NULL);
      return (1);
    }
    if (!(beamline->flags & BEAMLINE_RADINT_DONE)) {
      printWarning("coupled_twiss_output has emittances_from_twiss_command=1, but twiss calculations don't include radiation integrals", NULL);
      return (1);
    }
    emit_x = beamline->radIntegrals.ex0;
    sigma_dp = beamline->radIntegrals.sigmadelta;
    if (verbosity > 2)
      printf("Raw emittance = %e, momentum spread = %e\n", emit_x, sigma_dp);
  }
  fflush(stdout);

  emit[0] = emit_x;
  emit[1] = emit_x * emittance_ratio;
  emit[2] = 0;

  /* Count the number of elements from the recirc element to the end. */
  /* Also store the pointer to the recirc element. */
  eptr = eptr0 = beamline->elem;
  nElements = lastNElements = beamline->n_elems;
  while (eptr) {
    if (eptr->type == T_RECIRC) {
      lastNElements = nElements;
      eptr0 = eptr;
    }
    eptr = eptr->succ;
    nElements--;
  }
  nElements = lastNElements;

  if (starting_coord) {
    /* use the closed orbit to compute the on-orbit R matrix */
    M1 = tmalloc(sizeof(*M1));
    initialize_matrices(M1, 1);
    for (i = 0; i < 6; i++) {
      M1->C[i] = starting_coord[i];
      M1->R[i][i] = 1;
    }
    M = accumulate_matrices(eptr0, run, M1, concat_order, 0);
    free_matrices(M1);
    free(M1);
    M1 = NULL;
  } else
    M = accumulate_matrices(eptr0, run, NULL, concat_order, 0);
  R = M->R;

  if (verbosity > 3) {
    long order;
    order = M->order;
    M->order = 1;
    print_matrices(stdout, "One-turn matrix:", M);
    M->order = order;
  }

  /* Determination of matrix dimension for these calculations. */
  if (calculate_3d_coupling != 1) {
    matDim = 4;
  } else {
    if (abs(R[4][4]) + abs(R[5][5]) >= 2) {
      printf("Either there is no cavity or 3rd mode is unstable. Only 2 modes will be calculated.\n");
      matDim = 4;
    } else {
      matDim = 6;
    }
  }
  eigenModesNumber = matDim / 2;

  /*--- Reducing matrix dimensions, A is reduced R */
  for (i = 0; i < matDim; i++) {
    for (j = 0; j < matDim; j++) {
      A[i * matDim + j] = R[j][i];
    }
  }
  free_matrices(M);
  free(M);
  M = NULL;

  /*--- Changing time sign for symplecticity... */
  if (matDim == 6) {
    for (i = 0; i < 6; i++) {
      A[24 + i] = -1.0 * A[24 + i];
      A[i * 6 + 4] = -1.0 * A[i * 6 + 4];
    }
  }
  if (verbosity > 4) {
    MatrixPrintout((double *)&A, &matDim, &matDim, 1);
  }

  /*--- Calculating eigenvectors using dgeev_ ... */
  JOBVL = 'N';
  JOBVR = 'V';
  N = matDim;
  LDA = matDim;
  LDVL = 1;
  LDVR = matDim;
  lwork = 204;
#if defined(LAPACK) || defined(CLAPACK) || defined(MKL)
  dgeev_((char *)&JOBVL, (char *)&JOBVR, (int *)&N, (double *)&A,
         (int *)&LDA, (double *)&WR, (double *)&WI, (double *)&VL,
         (int *)&LDVL, (double *)&VR, (int *)&LDVR, (double *)&work,
         (int *)&lwork, (int *)&info);
#else
  fprintf(stderr, "Error calling dgeev. You will need to install LAPACK and rebuild elegant\n");
  return (1);
#endif
  if (info != 0) {
    if (info < 0) {
      printf("Error calling dgeev, argument %d.\n", abs(info));
    }
    if (info > 0) {
      printf("Error running dgeev, calculation of eigenvalue number %d failed.\n", info);
    }
    return (1);
  }
  if (verbosity > 1) {
    printf("Info: %d ; %f \n", info, work[0]);
    for (i = 0; i < matDim; i++) {
      printf("%d: %9.6f + i* %10.6f\n", i, WR[i], WI[i]);
    }
    fflush(stdout);
  }
  if (verbosity > 2) {
    printf("Non-normalized vectors:\n");
    MatrixPrintout((double *)&VR, &matDim, &matDim, 1);
    fflush(stdout);
  }

  /*--- Sorting of eigenvalues and eigenvectors according to (x,y,z)... */
  SortEigenvalues((double *)&WR, (double *)&WI, (double *)&VR, matDim, eigenModesNumber, verbosity);
  nux = fabs(atan2(WI[0], WR[0]) / PIx2);
  nuy = fabs(atan2(WI[2], WR[2]) / PIx2);
  if (verbosity>0) {
    printf("coupled twiss nux = %le, nuy=%le\n", nux, nuy);
    fflush(stdout);
  }
  rpn_store(nux, NULL, cnuMemory[0]);
  rpn_store(nuy, NULL, cnuMemory[1]);

  /*--- Normalization of eigenvectors... */
  for (k = 0; k < eigenModesNumber; k++) {
    Norm[k] = 0;
    for (i = 0; i < eigenModesNumber; i++) {
      /* Index = Irow*matDim + Icolumn */
      Norm[k] += VR[2 * k * matDim + 2 * i + 1] * VR[(2 * k + 1) * matDim + 2 * i] - VR[2 * k * matDim + 2 * i] * VR[(2 * k + 1) * matDim + 2 * i + 1];
    }
    Norm[k] = 1.0 / sqrt(fabs(Norm[k]));
    if (verbosity > 3) {
      printf("Norm[%d]= %12.4e \n", k, Norm[k]);
    }
  }
  for (k = 0; k < eigenModesNumber; k++) {
    for (i = 0; i < matDim; i++) {
      Vnorm[k * 2 * matDim + i] = VR[k * 2 * matDim + i] * Norm[k];
      Vnorm[(k * 2 + 1) * matDim + i] = VR[(k * 2 + 1) * matDim + i] * Norm[k];
    }
  }
  if (verbosity > 2) {
    printf("Normalized vectors:\n");
    MatrixPrintout((double *)&Vnorm, &matDim, &matDim, 1);
  }

  if (SDDScoupledInitialized) {
    /*--- Prepare the output file */
    if (!SDDS_StartPage(&SDDScoupled, nElements) ||
        !SDDS_SetParameters(&SDDScoupled, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                            "nux", nux, "nuy", nuy, NULL)) {
      fflush(stdout);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      return (1);
    }
  }

  /*--- Loop over elements */
  iElement = 0;
  eptr = eptr0;
  while (eptr) {
    if (verbosity > 1) {
      printf("\nElement number %ld: %s\n", iElement, eptr->name);
      fflush(stdout);
    }

    if (!eptr->accumMatrix) {
      fprintf(stderr, "Error: no accumulated matrix found for element %s", eptr->name);
      return (1);
    }

    /*--- Reducing matrix dimensions */
    R = eptr->accumMatrix->R;
    for (i = 0; i < matDim; i++) {
      for (j = 0; j < matDim; j++) {
        transferMatrix[i * matDim + j] = R[j][i];
      }
    }

    /*--- Changing time sign for symplecticity... */
    if (matDim == 6) {
      for (i = 0; i < 6; i++) {
        transferMatrix[24 + i] = -1.0 * transferMatrix[24 + i];
        transferMatrix[i * 6 + 4] = -1.0 * transferMatrix[i * 6 + 4];
      }
    }

    /*--- Calculating A matrices (product of eigenvectors)... */
    GetAMatrix((double *)&Vnorm, (double *)&transferMatrix, (double *)&Amatrix, &eigenModesNumber, &matDim);
    if (verbosity > 2) {
      for (k = 0; k < eigenModesNumber; k++) {
        printf("A matrix for mode %d\n", k);
        MatrixPrintout((double *)&Amatrix[k * matDim * matDim], &matDim, &matDim, 1);
      }
    }

    /*--- Calculating sigma matrix... */
    if (eigenModesNumber == 3) {
      emit[2] = sigma_dp * sigma_dp * Amatrix[2 * matDim * matDim + 4 * matDim + 4];
    }
    for (i = 0; i < matDim; i++) {
      for (j = 0; j < matDim; j++) {
        SigmaMatrix[i][j] = 0;
        for (k = 0; k < eigenModesNumber; k++) {
          SigmaMatrix[i][j] += emit[k] * Amatrix[k * matDim * matDim + i * matDim + j];
        }
      }
    }
    if (verbosity > 1) {
      printf("Sigma matrix:\n");
      MatrixPrintout((double *)&SigmaMatrix, &matDim, &matDim, 2);
    }

    tilt = 0.5 * atan(2 * SigmaMatrix[0][2] / (SigmaMatrix[0][0] - SigmaMatrix[2][2]));
    if (SDDScoupledInitialized) {
      /*--- Calculating beam sizes: 0-SigmaX, 1-SigmaXP, 2-SigmaY, 3-SigmaYP, 4-BeamTilt, 5-BunchLength */
      if (!SDDS_SetRowValues(&SDDScoupled, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                             iElement,
                             "ElementName", eptr->name,
                             "s", eptr->end_pos,
                             "Sx", sqrt(SigmaMatrix[0][0]),
                             "Sxp", sqrt(SigmaMatrix[1][1]),
                             "Sy", sqrt(SigmaMatrix[2][2]),
                             "Syp", sqrt(SigmaMatrix[3][3]),
                             "xyTilt", tilt,
                             "Ss", eigenModesNumber == 3 ? sqrt(SigmaMatrix[4][4]) : -1,
                             NULL)) {
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        return (1);
      }
    }

    if (verbosity > 1) {
      printf("SigmaX  = %12.4e, SigmaY  = %12.4e, Beam tilt = %12.4e \n",
             sqrt(SigmaMatrix[0][0]), sqrt(SigmaMatrix[2][2]),
             0.5 * atan(2 * SigmaMatrix[0][2] / (SigmaMatrix[0][0] - SigmaMatrix[2][2])));
      printf("SigmaXP = %12.4e, SigmaYP = %12.4e, \n", sqrt(SigmaMatrix[1][1]), sqrt(SigmaMatrix[3][3]));
      if (eigenModesNumber == 3) {
        printf("Bunch length = %12.4e \n", sqrt(SigmaMatrix[4][4]));
      }
    }

    betax1 = Amatrix[0];
    betax2 = Amatrix[1 * matDim * matDim];
    betay1 = Amatrix[2 * matDim + 2];
    betay2 = Amatrix[1 * matDim * matDim + 2 * matDim + 2];
    etax = sqrt(Amatrix[2 * matDim * matDim] * Amatrix[2 * matDim * matDim + 4 * matDim + 4]);
    etay = sqrt(Amatrix[2 * matDim * matDim + 2 * matDim + 2] * Amatrix[2 * matDim * matDim + 4 * matDim + 4]);
    if (SDDScoupledInitialized) {
      if (!SDDS_SetRowValues(&SDDScoupled, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                             iElement,
                             "betax1", betax1,
                             "betax2", betax2,
                             "betay1", betay1,
                             "betay2", betay2,
                             "etax", etax,
                             "etay", etay,
                             NULL)) {
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        return (1);
      }

      if (output_sigma_matrix) {
        char name[100];
        for (i = 0; i < matDim; i++)
          for (j = i; j < matDim; j++) {
            sprintf(name, "S%d%d", i + 1, j + 1);
            if (!SDDS_SetRowValues(&SDDScoupled, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                                   iElement, name, SigmaMatrix[i][j], NULL)) {
              SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
              return (1);
            }
          }
      }
    }

    if (verbosity > 1) {
      printf("betax_1 = %12.4e, betax_2 = %12.4e \n",
             Amatrix[0], Amatrix[1 * matDim * matDim]);
      printf("betay_1 = %12.4e, betay_2 = %12.4e \n",
             Amatrix[2 * matDim + 2], Amatrix[1 * matDim * matDim + 2 * matDim + 2]);
      printf("etax    = %12.4e, etay    = %12.4e \n",
             sqrt(Amatrix[2 * matDim * matDim] * Amatrix[2 * matDim * matDim + 4 * matDim + 4]),
             sqrt(Amatrix[2 * matDim * matDim + 2 * matDim + 2] * Amatrix[2 * matDim * matDim + 4 * matDim + 4]));
      fflush(stdout);
    }

    if (eptr->type == T_MARK && ((MARK *)eptr->p_elem)->fitpoint)
      store_fitpoint_ctwiss_parameters((MARK *)eptr->p_elem, eptr->name, eptr->occurence, betax1, betax2, betay1, betay2, etax, etay,
                                       tilt);

    
    iElement++;
    eptr = eptr->succ;
  }

  if (SDDScoupledInitialized && !SDDS_WritePage(&SDDScoupled)) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    return (1);
  }
  return (0);
}

void finish_coupled_twiss_output() {
  if (SDDScoupledInitialized && !SDDS_Terminate(&SDDScoupled)) {
    SDDS_SetError("Problem terminating SDDS output (finish_twiss_output)");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  SDDScoupledInitialized = 0;
  initialized = 0;
}

/****************************************************************************************************************/

void SortEigenvalues(double *WR, double *WI, double *VR, int matDim, int eigenModesNumber, int verbosity) {
  int N, i, j, index;

  double WRcopy[6], WIcopy[6], VRcopy[36];
  double **VV;
  int *MaxIndex;
  N = eigenModesNumber;
  if (N < 2)
    return;

  MaxIndex = malloc(sizeof(*MaxIndex) * N);
  VV = malloc(sizeof(*VV) * N);
  for (i = 0; i < N; i++)
    VV[i] = malloc(sizeof(**VV) * N);

  /*--- Finding biggest components of vectors... */
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      VV[i][j] = pow(VR[i * 2 * matDim + j * 2], 2) + pow(VR[i * 2 * matDim + j * 2 + 1], 2) +
        pow(VR[(i * 2 + 1) * matDim + j * 2], 2) + pow(VR[(i * 2 + 1) * matDim + j * 2 + 1], 2);
    }
  }
  /*
    for (i=0; i<N; i++) {
    MaxIndex[i]=GetMaxIndex(VV[i], N);
    }
  */
  MaxIndex[0] = GetMaxIndex(VV[0], N);
  VV[1][MaxIndex[0]] = -1.0;
  MaxIndex[1] = GetMaxIndex(VV[1], N);
  if (N > 2) {
    VV[2][MaxIndex[0]] = -1.0;
    VV[2][MaxIndex[1]] = -1.0;
    MaxIndex[2] = GetMaxIndex(VV[2], N);
  }

  /*--- Copying arrays... */
  for (i = 0; i < matDim; i++) {
    WRcopy[i] = WR[i];
    WIcopy[i] = WI[i];
    for (j = 0; j < matDim; j++) {
      VRcopy[i * matDim + j] = VR[i * matDim + j];
    }
  }

  /*--- Copying back according to MaxIndex... */
  for (i = 0; i < N; i++) {
    index = MaxIndex[i];
    WR[i * 2] = WRcopy[index * 2];
    WR[i * 2 + 1] = WRcopy[index * 2 + 1];
    WI[i * 2] = WIcopy[index * 2];
    WI[i * 2 + 1] = WIcopy[index * 2 + 1];
    for (j = 0; j < matDim; j++) {
      VR[i * 2 * matDim + j] = VRcopy[index * 2 * matDim + j];
      VR[(i * 2 + 1) * matDim + j] = VRcopy[(index * 2 + 1) * matDim + j];
    }
  }

  if (verbosity > 1) {
    printf("Eigenvalues after sorting:\n");
    for (i = 0; i < matDim; i++) {
      printf("%d: %9.6f + i* %10.6f\n", i, WR[i], WI[i]);
    }
  }
  if (verbosity > 2) {
    printf("Vectors after sorting:\n");
    MatrixPrintout((double *)&VR[0], &matDim, &matDim, 1);
  }

  free(MaxIndex);
  for (i = 0; i < N; i++)
    free(VV[i]);
  free(VV);
}

int GetMaxIndex(double *V, int N) {
  int i, maxIndex;
  double maxNumber;
  maxNumber = -1e99;
  maxIndex = -1;
  for (i = 0; i < N; i++) {
    if (V[i] > maxNumber) {
      maxNumber = V[i];
      maxIndex = i;
    }
  }
  if (maxIndex == -1) {
    printf("Error finding maximum number.\n");
    exitElegant(1);
  }
  return maxIndex;
}

/****************************************************************************************************************/

void GetAMatrix(double *V, double *transferMatrix, double *A, int *eigenModesNumber, int *matDim) {
  int i, j, k, K, N;
  double *E;
  K = *eigenModesNumber;
  N = *matDim;

  E = malloc(sizeof(*E) * N * N);

  /*--- Vector rotation */
  MatrixProduct(&N, &N, (double *)&V[0], &N, &N, &transferMatrix[0], E);

  /*--- A is 3d array. First index is eigenmode, other 2 are rows and columns of A matrix */
  for (k = 0; k < K; k++) {
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        A[k * N * N + i * N + j] = E[i + 2 * k * N] * E[j + 2 * k * N] + E[i + (2 * k + 1) * N] * E[j + (2 * k + 1) * N];
      }
    }
  }
  free(E);
}

/*********************************************************************************************************/

void MatrixPrintout(double *AA, int *NA, int *MA, int dim) {
  int i, j, N, M, Mult;

  N = *NA;
  M = *MA;
  if (dim == 1) {
    Mult = N;
  } else {
    Mult = 6;
  }
  for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
      printf("%14.6e", AA[i + Mult * j]);
    }
    printf("\n");
  }
  printf("\n");
  fflush(stdout);
}

/*********************************************************************************************************/

void MatrixProduct(int *N1, int *M1, double *T1, int *N2, int *M2, double *T2, double *T3) {
  int i, j, k, nRows1, nCols1, nRows2, nCols2;
  nRows1 = *N1;
  nCols1 = *M1;
  nRows2 = *N2;
  nCols2 = *M2;

  if (nCols1 != nRows2) {
    printf("Wrong matrix dimension!\n");
    exitElegant(1);
  }

  for (i = 0; i < nRows1; i++) {
    for (j = 0; j < nCols2; j++) {
      T3[i * nRows1 + j] = 0;
      for (k = 0; k < nCols1; k++) {
        T3[i * nRows1 + j] += T1[i * nRows1 + k] * T2[k * nRows2 + j];
      }
    }
  }
}

void store_fitpoint_ctwiss_parameters(MARK *fpt, char *name, long occurence,
                                      double betax1, double betax2,
                                      double betay1, double betay2,
                                      double etax, double etay,
                                      double tilt) {
  long i;
  double data[7];
  static char *suffix[7] = {
    "betax1", "betax2", "betay1", "betay2", "cetax", "cetay", "tilt"};
  static char s[200];

  data[0] = betax1;
  data[1] = betax2;
  data[2] = betay1;
  data[3] = betay2;
  data[4] = etax;
  data[5] = etay;
  data[6] = tilt;

  if (!(fpt->init_flags & 16)) {
    fpt->ctwiss_mem = tmalloc(sizeof(*(fpt->ctwiss_mem)) * 12);
    fpt->init_flags |= 16;
    for (i = 0; i < 7; i++) {
      sprintf(s, "%s#%ld.%s", name, occurence, suffix[i]);
      fpt->ctwiss_mem[i] = rpn_create_mem(s, 0);
    }
  }
  for (i = 0; i < 7; i++)
    rpn_store(data[i], NULL, fpt->ctwiss_mem[i]);
}
