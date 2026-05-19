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
                                      double alphax1, double alphax2,
                                      double alphay1, double alphay2,
                                      double etax, double etay,
                                      double tilt);
static void LoadStartingCoupledTwissFromFile(char *filename_inner_scope,
                                             char *elementName,
                                             long elementOccurrence);

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

  if (reference_file && matched)
    bombElegant("coupled_twiss_output: reference_file and matched=1 are incompatible", NULL);

  if (!matched) {
    /* User-supplied initial conditions are propagated; periodic eigenvector
       calculation is skipped, so the third (longitudinal) mode cannot be set up. */
    if (calculate_3d_coupling) {
      printWarning("coupled_twiss_output: matched=0 does not support calculate_3d_coupling=1; forcing calculate_3d_coupling=0",
                   NULL);
      calculate_3d_coupling = 0;
    }

    if (reference_file) {
      if (reference_element && reference_element_occurrence < 0)
        bombElegant("coupled_twiss_output: invalid reference_element_occurrence---use 0 for last occurrence, >=1 for specific occurrence", NULL);
      LoadStartingCoupledTwissFromFile(reference_file, reference_element,
                                       reference_element_occurrence);
      if (reflect_reference_values) {
        /* Time-reversal: positions unchanged, momenta flip sign. For a symmetric
           matrix A, A[i][j] -> R[i][i] R[j][j] A[i][j] with R = diag(1,-1,1,-1)
           in the transverse subspace. So alpha (xx'/yy' off-diagonal) and the
           cross-block entries with one momentum index flip sign; everything else
           is unchanged. */
        alpha_x1 = -alpha_x1;  alpha_x2 = -alpha_x2;
        alpha_y1 = -alpha_y1;  alpha_y2 = -alpha_y2;
        A_xpy_1  = -A_xpy_1;   A_xyp_1  = -A_xyp_1;
        A_xpy_2  = -A_xpy_2;   A_xyp_2  = -A_xyp_2;
        etap_x   = -etap_x;    etap_y   = -etap_y;
      }
      printf("Starting coupled twiss parameters loaded from %s:\n", reference_file);
      printf("  betax1=%le alphax1=%le gammax1=%le\n", beta_x1, alpha_x1, gamma_x1);
      printf("  betax2=%le alphax2=%le gammax2=%le\n", beta_x2, alpha_x2, gamma_x2);
      printf("  betay1=%le alphay1=%le gammay1=%le\n", beta_y1, alpha_y1, gamma_y1);
      printf("  betay2=%le alphay2=%le gammay2=%le\n", beta_y2, alpha_y2, gamma_y2);
      printf("  eta_x=%le etap_x=%le eta_y=%le etap_y=%le\n", eta_x, etap_x, eta_y, etap_y);
      fflush(stdout);
    }

    if (beta_x1 < 0 || beta_x2 < 0 || beta_y1 < 0 || beta_y2 < 0)
      bombElegant("coupled_twiss_output: beta_x1/x2/y1/y2 must be non-negative when matched=0", NULL);
    if (beta_x1 == 0 && beta_x2 == 0)
      bombElegant("coupled_twiss_output: at least one of beta_x1, beta_x2 must be > 0 when matched=0", NULL);
    if (beta_y1 == 0 && beta_y2 == 0)
      bombElegant("coupled_twiss_output: at least one of beta_y1, beta_y2 must be > 0 when matched=0", NULL);
  }

  if (filename) {
    if (!SDDS_InitializeOutputElegant(&SDDScoupled, SDDS_BINARY, 0, NULL, NULL, filename) ||
        SDDS_DefineColumn(&SDDScoupled, "ElementName", NULL, NULL, "Element name",
                          NULL, SDDS_STRING, 0) < 0 ||
        SDDS_DefineColumn(&SDDScoupled, "s", "s", "m", "Distance from start of beamline",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDScoupled, "Sx", "$gs$r$bx$n", "m", "Horizontal RMS beam size",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDScoupled, "Sxp", "$gs$r$bx'$n", "", "Horizontal RMS beam divergence",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDScoupled, "Sy", "$gs$r$by$n", "m", "Vertical RMS beam size",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDScoupled, "Syp", "$gs$r$by'$n", "", "Vertical RMS beam divergence",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDScoupled, "xyTilt", NULL, "rad",
                          "Tilt angle of the (x,y) beam ellipse principal axis",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDScoupled, "Ss", "$gs$r$bs$n", "m",
                          "RMS bunch length (3D mode only; -1 if not computed)",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDScoupled, "betax1", "$gb$r$bx,1$n", "m",
                          "Mode-1 contribution to horizontal beta function",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDScoupled, "betax2", "$gb$r$bx,2$n", "m",
                          "Mode-2 contribution to horizontal beta function",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDScoupled, "betay1", "$gb$r$by,1$n", "m",
                          "Mode-1 contribution to vertical beta function",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDScoupled, "betay2", "$gb$r$by,2$n", "m",
                          "Mode-2 contribution to vertical beta function",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDScoupled, "alphax1", "$ga$r$bx,1$n", NULL,
                          "Mode-1 contribution to horizontal alpha function",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDScoupled, "alphax2", "$ga$r$bx,2$n", NULL,
                          "Mode-2 contribution to horizontal alpha function",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDScoupled, "alphay1", "$ga$r$by,1$n", NULL,
                          "Mode-1 contribution to vertical alpha function",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDScoupled, "alphay2", "$ga$r$by,2$n", NULL,
                          "Mode-2 contribution to vertical alpha function",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDScoupled, "gammax1", "$gg$r$bx,1$n", "1/m",
                          "Mode-1 contribution to horizontal gamma function",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDScoupled, "gammax2", "$gg$r$bx,2$n", "1/m",
                          "Mode-2 contribution to horizontal gamma function",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDScoupled, "gammay1", "$gg$r$by,1$n", "1/m",
                          "Mode-1 contribution to vertical gamma function",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDScoupled, "gammay2", "$gg$r$by,2$n", "1/m",
                          "Mode-2 contribution to vertical gamma function",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDScoupled, "A_xy_1", NULL, "m",
                          "Mode-1 cross-plane xy covariance per unit emittance",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDScoupled, "A_xpy_1", NULL, NULL,
                          "Mode-1 cross-plane x'y covariance per unit emittance",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDScoupled, "A_xyp_1", NULL, NULL,
                          "Mode-1 cross-plane xy' covariance per unit emittance",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDScoupled, "A_xpyp_1", NULL, "1/m",
                          "Mode-1 cross-plane x'y' covariance per unit emittance",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDScoupled, "A_xy_2", NULL, "m",
                          "Mode-2 cross-plane xy covariance per unit emittance",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDScoupled, "A_xpy_2", NULL, NULL,
                          "Mode-2 cross-plane x'y covariance per unit emittance",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDScoupled, "A_xyp_2", NULL, NULL,
                          "Mode-2 cross-plane xy' covariance per unit emittance",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDScoupled, "A_xpyp_2", NULL, "1/m",
                          "Mode-2 cross-plane x'y' covariance per unit emittance",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDScoupled, "etax", "$gc$r$bx$n", "m",
                          "Horizontal dispersion (signed)",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDScoupled, "etaxp", "$gc$r$bx$n'", NULL,
                          "Horizontal dispersion slope (signed)",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDScoupled, "etay", "$gc$r$by$n", "m",
                          "Vertical dispersion (signed)",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        SDDS_DefineColumn(&SDDScoupled, "etayp", "$gc$r$by$n'", NULL,
                          "Vertical dispersion slope (signed)",
                          NULL, SDDS_DOUBLE, 0) < 0 ||
        !SDDS_DefineSimpleParameter(&SDDScoupled, "nux", "", SDDS_DOUBLE) ||
        !SDDS_DefineSimpleParameter(&SDDScoupled, "nuy", "", SDDS_DOUBLE)) {
      printf("Unable to set up file %s\n", filename);
      fflush(stdout);
      SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }

    if (output_sigma_matrix) {
      long maxDimension, i, j;
      char name[100], units[10], description[100];
      /* labels for phase-space indices 1..6 used in column descriptions */
      static const char *coordLabel[6] = {"x", "x'", "y", "y'", "s", "delta"};
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
          sprintf(description, "Sigma matrix element <%s %s>", coordLabel[i], coordLabel[j]);
          if (SDDS_DefineColumn(&SDDScoupled, name, NULL, units, description,
                                NULL, SDDS_DOUBLE, 0) < 0) {
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
  double betax1, betax2, betay1, betay2, etax, etay, etaxp, etayp, tilt;
  double alphax1, alphax2, alphay1, alphay2;
  double nux, nuy;
  static long cnuMemory[2] = {-1, -1};
  /* For non-periodic (matched=0) propagation of user-supplied initial conditions:
     Astart[k] holds the initial 4x4 A matrix (sigma-matrix shape) for mode k=0,1
     stored row-major; dispStart/dispCur hold the 6D dispersion vector. */
  double Astart[2][16];
  double dispStart[6]={0,0,0,0,0,0}, dispCur[6]={0,0,0,0,0,0};
  
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
    print_matrices(stdout, "One-turn matrix:", M, 0.0);
    M->order = order;
  }

  if (matched) {
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
    {
      long long N_ll = (long long)N;
      long long LDA_ll = (long long)LDA;
      long long LDVL_ll = (long long)LDVL;
      long long LDVR_ll = (long long)LDVR;
      long long lwork_ll = (long long)lwork;
      long long info_ll = 0;
      dgeev_((char *)&JOBVL, (char *)&JOBVR, (int *)&N_ll, (double *)&A,
             (int *)&LDA_ll, (double *)&WR, (double *)&WI, (double *)&VL,
             (int *)&LDVL_ll, (double *)&VR, (int *)&LDVR_ll, (double *)&work,
             (int *)&lwork_ll, (int *)&info_ll);
      info = (int)info_ll;
    }
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
  } else {
    /* Non-periodic: propagate user-supplied initial twiss values.
       Force 4x4 transverse computation; dispersion is propagated as a separate 6-vector.
       The one-turn matrix is not used; free it now (accumulate_matrices already filled
       eptr->accumMatrix for each element). */
    matDim = 4;
    eigenModesNumber = 2;
    free_matrices(M);
    free(M);
    M = NULL;

    /* Build initial A matrices for the two transverse modes.
       The user supplies (beta, alpha) per plane per mode -- four numbers per A_k that
       set A_k[0][0], A_k[2][2], A_k[0][1], A_k[2][3]. To uniquely propagate A_k we
       also need six more entries: the diagonal gammas A_k[1][1], A_k[3][3] and the
       four cross-block entries A_k[0][2], A_k[0][3], A_k[1][2], A_k[1][3].

       gamma_x{1,2}, gamma_y{1,2}: if the user supplies a positive value, use it
       directly. Otherwise (default -1) fall back to a rank-2 eigenvector heuristic --
       primary plane gets (1+alpha^2)/beta (full Twiss), secondary plane gets
       alpha^2/beta (rank-1 reduction). The heuristic is correct for uncoupled
       initial conditions; for coupled ones it is only approximate and the user
       should supply gamma explicitly from a reference (matched) ctwi file.

       A_xy_{1,2}, A_xyp_{1,2}, A_xpy_{1,2}, A_xpyp_{1,2}: the cross-block entries
       of A_k. Default 0 (block-diagonal). For exact reproduction of a coupled
       matched solution the user should supply these from the reference file. */
    for (i = 0; i < 16; i++) {
      Astart[0][i] = 0;
      Astart[1][i] = 0;
    }
    for (k = 0; k < 2; k++) {
      double bx = (k == 0) ? beta_x1   : beta_x2;
      double ax = (k == 0) ? alpha_x1  : alpha_x2;
      double by = (k == 0) ? beta_y1   : beta_y2;
      double ay = (k == 0) ? alpha_y1  : alpha_y2;
      double gxu = (k == 0) ? gamma_x1 : gamma_x2;
      double gyu = (k == 0) ? gamma_y1 : gamma_y2;
      double Axy   = (k == 0) ? A_xy_1   : A_xy_2;
      double Axpy  = (k == 0) ? A_xpy_1  : A_xpy_2;
      double Axyp  = (k == 0) ? A_xyp_1  : A_xyp_2;
      double Axpyp = (k == 0) ? A_xpyp_1 : A_xpyp_2;
      int primary_is_x = (k == 0);
      double gx_heur = 0, gy_heur = 0;
      if (bx > 0)
        gx_heur = primary_is_x ? (1.0 + ax * ax) / bx : (ax * ax) / bx;
      if (by > 0)
        gy_heur = primary_is_x ? (ay * ay) / by : (1.0 + ay * ay) / by;
      double gx = (gxu > 0) ? gxu : gx_heur;
      double gy = (gyu > 0) ? gyu : gy_heur;
      /* x-plane block */
      if (bx > 0) {
        Astart[k][0 * 4 + 0] = bx;
        Astart[k][1 * 4 + 1] = gx;
        Astart[k][0 * 4 + 1] = -ax;
        Astart[k][1 * 4 + 0] = -ax;
      }
      /* y-plane block */
      if (by > 0) {
        Astart[k][2 * 4 + 2] = by;
        Astart[k][3 * 4 + 3] = gy;
        Astart[k][2 * 4 + 3] = -ay;
        Astart[k][3 * 4 + 2] = -ay;
      }
      /* cross-block (x,xp) x (y,yp), symmetric */
      Astart[k][0 * 4 + 2] = Axy;    Astart[k][2 * 4 + 0] = Axy;
      Astart[k][0 * 4 + 3] = Axyp;   Astart[k][3 * 4 + 0] = Axyp;
      Astart[k][1 * 4 + 2] = Axpy;   Astart[k][2 * 4 + 1] = Axpy;
      Astart[k][1 * 4 + 3] = Axpyp;  Astart[k][3 * 4 + 1] = Axpyp;
    }

    /* Initial dispersion vector: applying R(s) to (eta, etap, eta, etap, 0, 1)
       propagates the closed-orbit offset per unit dp/p. */
    dispStart[0] = eta_x;
    dispStart[1] = etap_x;
    dispStart[2] = eta_y;
    dispStart[3] = etap_y;
    dispStart[4] = 0;
    dispStart[5] = 1;

    /* Tunes are not defined for a non-periodic propagation. */
    nux = 0;
    nuy = 0;
  }

  rpn_store(nux, NULL, cnuMemory[0]);
  rpn_store(nuy, NULL, cnuMemory[1]);
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

    R = eptr->accumMatrix->R;

    if (matched) {
      /*--- Reducing matrix dimensions */
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
    } else {
      /*--- Non-periodic: transport user-supplied initial A matrices via
            A_k(s) = M * A_k(0) * M^T using the 4x4 transverse block. */
      double M4[16], MA[16];
      int p;
      for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++) {
          M4[i * 4 + j] = R[i][j];
        }
      }
      for (k = 0; k < 2; k++) {
        for (i = 0; i < 4; i++) {
          for (j = 0; j < 4; j++) {
            double s = 0;
            for (p = 0; p < 4; p++)
              s += M4[i * 4 + p] * Astart[k][p * 4 + j];
            MA[i * 4 + j] = s;
          }
        }
        for (i = 0; i < 4; i++) {
          for (j = 0; j < 4; j++) {
            double s = 0;
            for (p = 0; p < 4; p++)
              s += MA[i * 4 + p] * M4[j * 4 + p]; /* M^T[p][j] = M[j][p] */
            Amatrix[k * 16 + i * 4 + j] = s;
          }
        }
      }
      /* Propagate dispersion vector with full 6x6: dispCur = R * dispStart. */
      for (i = 0; i < 6; i++) {
        double s = 0;
        for (j = 0; j < 6; j++)
          s += R[i][j] * dispStart[j];
        dispCur[i] = s;
      }
    }

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
    /* A[i][j] for each mode k stores the per-mode contribution to the sigma matrix
       (up to emittance), so A[0][1] = -alpha_x and A[2][3] = -alpha_y for that mode. */
    alphax1 = -Amatrix[0 * matDim * matDim + 0 * matDim + 1];
    alphax2 = -Amatrix[1 * matDim * matDim + 0 * matDim + 1];
    alphay1 = -Amatrix[0 * matDim * matDim + 2 * matDim + 3];
    alphay2 = -Amatrix[1 * matDim * matDim + 2 * matDim + 3];
    if (matched) {
      if (eigenModesNumber == 3) {
        /* For the longitudinal-mode eigenvector v_2 with the standard structure
           v_2 = (eta_x*v_5, eta_xp*v_5, eta_y*v_5, eta_yp*v_5, v_4, v_5):
               A_mode2[i][5] = eta_i * |v_5|^2 = eta_i * A_mode2[5][5]
           so eta_i = A_mode2[i][5] / A_mode2[5][5] (signed). This is exact and
           replaces the older sqrt(A[0][0]*A[4][4]) formula which gave |eta|
           inflated by sqrt(1+alpha_z^2). */
        double Azz = Amatrix[2 * matDim * matDim + 5 * matDim + 5];
        if (Azz != 0) {
          etax  = Amatrix[2 * matDim * matDim + 0 * matDim + 5] / Azz;
          etaxp = Amatrix[2 * matDim * matDim + 1 * matDim + 5] / Azz;
          etay  = Amatrix[2 * matDim * matDim + 2 * matDim + 5] / Azz;
          etayp = Amatrix[2 * matDim * matDim + 3 * matDim + 5] / Azz;
        } else {
          etax = etaxp = etay = etayp = 0;
        }
      } else {
        /* No longitudinal mode present, so coupled-twiss dispersion is not available. */
        etax = etaxp = etay = etayp = 0;
      }
    } else {
      /* Use the separately-propagated dispersion vector. */
      etax  = dispCur[0];
      etaxp = dispCur[1];
      etay  = dispCur[2];
      etayp = dispCur[3];
    }
    if (SDDScoupledInitialized) {
      /* Extra A-matrix entries that aren't recoverable from (beta,alpha) alone in the coupled case.
         Output them so a non-periodic run can reproduce a periodic solution exactly via rpn_load. */
      double gammax1_out = Amatrix[0 * matDim * matDim + 1 * matDim + 1];
      double gammax2_out = Amatrix[1 * matDim * matDim + 1 * matDim + 1];
      double gammay1_out = Amatrix[0 * matDim * matDim + 3 * matDim + 3];
      double gammay2_out = Amatrix[1 * matDim * matDim + 3 * matDim + 3];
      double Axy1_out   = Amatrix[0 * matDim * matDim + 0 * matDim + 2];
      double Axpy1_out  = Amatrix[0 * matDim * matDim + 1 * matDim + 2];
      double Axyp1_out  = Amatrix[0 * matDim * matDim + 0 * matDim + 3];
      double Axpyp1_out = Amatrix[0 * matDim * matDim + 1 * matDim + 3];
      double Axy2_out   = Amatrix[1 * matDim * matDim + 0 * matDim + 2];
      double Axpy2_out  = Amatrix[1 * matDim * matDim + 1 * matDim + 2];
      double Axyp2_out  = Amatrix[1 * matDim * matDim + 0 * matDim + 3];
      double Axpyp2_out = Amatrix[1 * matDim * matDim + 1 * matDim + 3];
      if (!SDDS_SetRowValues(&SDDScoupled, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                             iElement,
                             "betax1", betax1,
                             "betax2", betax2,
                             "betay1", betay1,
                             "betay2", betay2,
                             "alphax1", alphax1,
                             "alphax2", alphax2,
                             "alphay1", alphay1,
                             "alphay2", alphay2,
                             "gammax1", gammax1_out,
                             "gammax2", gammax2_out,
                             "gammay1", gammay1_out,
                             "gammay2", gammay2_out,
                             "A_xy_1",   Axy1_out,
                             "A_xpy_1",  Axpy1_out,
                             "A_xyp_1",  Axyp1_out,
                             "A_xpyp_1", Axpyp1_out,
                             "A_xy_2",   Axy2_out,
                             "A_xpy_2",  Axpy2_out,
                             "A_xyp_2",  Axyp2_out,
                             "A_xpyp_2", Axpyp2_out,
                             "etax",  etax,
                             "etaxp", etaxp,
                             "etay",  etay,
                             "etayp", etayp,
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
      printf("betax_1 = %12.4e, betax_2 = %12.4e \n", betax1, betax2);
      printf("betay_1 = %12.4e, betay_2 = %12.4e \n", betay1, betay2);
      printf("etax    = %12.4e, etay    = %12.4e \n", etax, etay);
      fflush(stdout);
    }

    if (eptr->type == T_MARK && ((MARK *)eptr->p_elem)->fitpoint)
      store_fitpoint_ctwiss_parameters((MARK *)eptr->p_elem, eptr->name, eptr->occurence,
                                       betax1, betax2, betay1, betay2,
                                       alphax1, alphax2, alphay1, alphay2,
                                       etax, etay, tilt);

    
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
                                      double alphax1, double alphax2,
                                      double alphay1, double alphay2,
                                      double etax, double etay,
                                      double tilt) {
  long i;
  double data[11];
  static char *suffix[11] = {
    "betax1", "betax2", "betay1", "betay2",
    "alphax1", "alphax2", "alphay1", "alphay2",
    "cetax", "cetay", "tilt"};
  static char s[200];

  data[0] = betax1;
  data[1] = betax2;
  data[2] = betay1;
  data[3] = betay2;
  data[4] = alphax1;
  data[5] = alphax2;
  data[6] = alphay1;
  data[7] = alphay2;
  data[8] = etax;
  data[9] = etay;
  data[10] = tilt;

  if (!(fpt->init_flags & 16)) {
    fpt->ctwiss_mem = tmalloc(sizeof(*(fpt->ctwiss_mem)) * 12);
    fpt->init_flags |= 16;
    for (i = 0; i < 11; i++) {
      sprintf(s, "%s#%ld.%s", name, occurence, suffix[i]);
      fpt->ctwiss_mem[i] = rpn_create_mem(s, 0);
    }
  }
  for (i = 0; i < 11; i++)
    rpn_store(data[i], NULL, fpt->ctwiss_mem[i]);
}

/*********************************************************************************************************/
/* Load the 24 initial-condition values for a non-periodic coupled twiss propagation from a previously
 * produced coupled_twiss_output SDDS file. Modeled on LoadStartingTwissFromFile in twiss.cc. The target
 * variables are the file-scope namelist globals (beta_x1, alpha_x1, gamma_x1, A_xy_1, ..., etap_y).
 *
 * If elementName is non-NULL, the row from that element is used (elementOccurrence>=1 selects which
 * occurrence; 0 selects the last). Otherwise the last row of the file is used.
 */
static void LoadStartingCoupledTwissFromFile(char *filename_inner_scope,
                                             char *elementName,
                                             long elementOccurrence) {
  SDDS_DATASET SDDSin;
  long rows = 0, rowOfInterest, i;
  static struct {
    const char *colName;
    double *target;
    const char *units; /* expected units, or NULL */
  } table[] = {
    { "betax1",   &beta_x1,   "m"   },
    { "betax2",   &beta_x2,   "m"   },
    { "betay1",   &beta_y1,   "m"   },
    { "betay2",   &beta_y2,   "m"   },
    { "alphax1",  &alpha_x1,  NULL  },
    { "alphax2",  &alpha_x2,  NULL  },
    { "alphay1",  &alpha_y1,  NULL  },
    { "alphay2",  &alpha_y2,  NULL  },
    { "gammax1",  &gamma_x1,  "1/m" },
    { "gammax2",  &gamma_x2,  "1/m" },
    { "gammay1",  &gamma_y1,  "1/m" },
    { "gammay2",  &gamma_y2,  "1/m" },
    { "A_xy_1",   &A_xy_1,    "m"   },
    { "A_xpy_1",  &A_xpy_1,   NULL  },
    { "A_xyp_1",  &A_xyp_1,   NULL  },
    { "A_xpyp_1", &A_xpyp_1,  "1/m" },
    { "A_xy_2",   &A_xy_2,    "m"   },
    { "A_xpy_2",  &A_xpy_2,   NULL  },
    { "A_xyp_2",  &A_xyp_2,   NULL  },
    { "A_xpyp_2", &A_xpyp_2,  "1/m" },
    { "etax",     &eta_x,     "m"   },
    { "etaxp",    &etap_x,    NULL  },
    { "etay",     &eta_y,     "m"   },
    { "etayp",    &etap_y,    NULL  },
  };
  long nCols = sizeof(table) / sizeof(table[0]);

  if (!SDDS_InitializeInputFromSearchPath(&SDDSin, filename_inner_scope) ||
      SDDS_ReadPage(&SDDSin) != 1)
    SDDS_Bomb((char *)"problem reading coupled twiss reference file");

  for (i = 0; i < nCols; i++) {
    if (SDDS_CheckColumn(&SDDSin, (char *)table[i].colName, (char *)table[i].units,
                         SDDS_ANY_FLOATING_TYPE, stdout) != SDDS_CHECK_OK) {
      fprintf(stderr, "coupled_twiss_output: required column \"%s\" not found in reference file\n",
              table[i].colName);
      SDDS_Bomb((char *)"invalid/missing columns in coupled twiss reference file");
    }
  }
  if (SDDS_CheckColumn(&SDDSin, (char *)"ElementName", NULL, SDDS_STRING, stdout) != SDDS_CHECK_OK)
    SDDS_Bomb((char *)"missing ElementName column in coupled twiss reference file");

  if (elementName) {
    if (!SDDS_SetRowFlags(&SDDSin, 1) ||
        (rows = SDDS_MatchRowsOfInterest(&SDDSin, (char *)"ElementName", elementName, SDDS_AND)) <= 0)
      SDDS_Bomb((char *)"Could not find specified reference_element in coupled twiss reference file");
    if (elementOccurrence > 0 && elementOccurrence > rows)
      SDDS_Bomb((char *)"Too few occurrences of reference_element in coupled twiss reference file");
  }
  if ((rows = SDDS_CountRowsOfInterest(&SDDSin)) < 1)
    SDDS_Bomb((char *)"No data in coupled twiss reference file");
  if (elementName && elementOccurrence > 0)
    rowOfInterest = elementOccurrence - 1;
  else
    rowOfInterest = rows - 1;

  for (i = 0; i < nCols; i++) {
    double *data = SDDS_GetColumnInDoubles(&SDDSin, (char *)table[i].colName);
    if (!data)
      SDDS_Bomb((char *)"Problem reading column from coupled twiss reference file");
    *(table[i].target) = data[rowOfInterest];
    free(data);
  }
  SDDS_Terminate(&SDDSin);
}
