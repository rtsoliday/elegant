/* Memory.c, rewriten from meschach (H. Shang)
 *
 * These versions assume that matrices are stored in column-major order, to be compatible with
 * LAPACK SVD routines.  This may introduce confusion in accessing the elements of a matrix.
 *
 * Example:
 * A = matrix_get(m, n) creates a matrix with m rows and n columns
 * However, A->me[i][j] is the ith column and jth row, 0<=i<=n, 0<=j<=m
 *
 * Using the macro Mij(A, i, j) accesses the ith row and jth column, avoiding confusion.
 */

/**************************************************************************
 **
 ** Copyright (C) 1993 David E. Steward & Zbigniew Leyk, all rights reserved.
 **
 **                          Meschach Library
 **
 ** This Meschach Library is provided "as is" without any express
 ** or implied warranty of any kind with respect to this software.
 ** In particular the authors shall not be liable for any direct,
 ** indirect, special, incidental or consequential damages arising
 ** in any way from use of the software.
 **
 ** Everyone is granted permission to copy, modify and redistribute this
 ** Meschach Library, provided:
 **  1.  All copies contain this copyright notice.
 **  2.  All modified copies shall carry a notice stating who
 **      made the last modification and the date of such modification.
 **  3.  No charge is made for this software or works derived from it.
 **      This clause shall not be construed as constraining other software
 **      distributed on the same medium as this software, nor is a
 **      distribution fee considered a charge.
 **
 ***************************************************************************/

#if defined(__APPLE__)
#  define ACCELERATE_NEW_LAPACK
#  include <Accelerate/Accelerate.h>
#endif
#include "SDDS.h"
#include "mdb.h"

#ifndef HUGE_VAL
#  define HUGE_VAL HUGE
#else
#  ifndef HUGE
#    define HUGE HUGE_VAL
#  endif
#endif

#ifdef MKL
#  include "mkl.h"
int dgemm_(char *transa, char *transb, const MKL_INT *m, const MKL_INT *n, const MKL_INT *k, double *alpha, double *a, const MKL_INT *lda, double *b, const MKL_INT *ldb, double *beta, double *c, const MKL_INT *ldc);
#endif

#ifdef CLAPACK
#  if !defined(_WIN32) && !defined(__APPLE__)
#    include "cblas.h"
#  endif
#  ifdef F2C
#    include "f2c.h"
#    if defined(_WIN32)
typedef struct {
  real r, i;
} complex;
typedef struct {
  doublereal r, i;
} doublecomplex;
#    endif
#  endif
#  if !defined(__APPLE__)
#    include "clapack.h"
#  endif
#  if defined(_WIN32)
int f2c_dgemm(char *transA, char *transB, integer *M, integer *N, integer *K,
              doublereal *alpha,
              doublereal *A, integer *lda,
              doublereal *B, integer *ldb,
              doublereal *beta,
              doublereal *C, integer *ldc);
#  endif
#endif

#ifdef LAPACK
#  include "f2c.h"
int dgetrf_(integer *m, integer *n, doublereal *a, integer *lda, integer *ipiv, integer *info);
int dgemm_(char *transa, char *transb, integer *m, integer *n, integer *k, doublereal *alpha,
           doublereal *a, integer *lda, doublereal *b, integer *ldb, doublereal *beta,
           doublereal *c, integer *ldc);
int dgesvd_(char *jobu, char *jobvt, integer *m, integer *n, doublereal *a, integer *lda,
            doublereal *s, doublereal *u, integer *ldu, doublereal *vt, integer *ldvt,
            doublereal *work, integer *lwork, integer *info);
#endif

#include "matrixOp.h"

/* m_get -- gets an mxn matrix (in MAT form) by dynamic memory allocation in column major order
 */

MAT *matrix_get(int m, int n) {
  MAT *matrix;
  int i;

  if (m < 0 || n < 0)
    SDDS_Bomb("Error for matrix_get, invalid row and/or column value provided.");
  if ((matrix = NEW(MAT)) == (MAT *)NULL)
    SDDS_Bomb("Unable to allocate memory for matrix (matrix_get (1))");

  matrix->m = matrix->max_m = m;
  matrix->n = matrix->max_n = n;
  matrix->max_size = m * n;

  if ((matrix->base = NEW_A(m * n, Real)) == (Real *)NULL) {
    free(matrix);
    SDDS_Bomb("Unable to allocate memory for matrix->base (matrix_get (2))");
  }
  if ((matrix->me = (Real **)calloc(n, sizeof(Real *))) == (Real **)NULL) {
    free(matrix->base);
    free(matrix);
    SDDS_Bomb("Unable to allocate memory for matrix->me (matrix_get (3))");
  }

  /* The 0 column data is from base[0] to base[m-1]; and 1 column data is
     from base[m] to base[2m-1]; etc...*/
  for (i = 0; i < n; i++)
    matrix->me[i] = &(matrix->base[i * matrix->m]);

  return matrix;
}

VEC *vec_get(int size) {
  VEC *vector;

  if (size < 0)
    SDDS_Bomb("negative size provided for v_get");

  if ((vector = NEW(VEC)) == (VEC *)NULL)
    SDDS_Bomb("Unable to allocate memory for vector(v_get)");

  vector->dim = vector->max_dim = size;
  if ((vector->ve = NEW_A(size, Real)) == (Real *)NULL) {
    free(vector);
    SDDS_Bomb("Unable to allocate memory for vector(v_get)");
  }
  return (vector);
}

int matrix_free(MAT *mat) {
  if (mat == (MAT *)NULL || (int)(mat->m) < 0 ||
      (int)(mat->n) < 0)
    /* don't trust it */
    return (-1);
  if (mat->me)
    free(mat->me);
  if (mat->base)
    free(mat->base);
  if (mat)
    free(mat);
  mat = (MAT *)NULL;
  return 0;
}

int vec_free(VEC *vec) {
  if (vec == (VEC *)NULL || (int)(vec->dim) < 0)
    /* don't trust it */
    return (-1);
  if (vec->ve)
    free((char *)vec->ve);
  free((char *)vec);
  vec = (VEC *)NULL;
  return (0);
}

MAT *matrix_copy(MAT *mat) {
  MAT *matrix = NULL;
  if (!mat)
    return NULL;
  matrix = matrix_get(mat->m, mat->n);
  memcpy((char *)matrix->base, (char *)mat->base, sizeof(*mat->base) * mat->m * mat->n);
  return matrix;
}

MAT *matrix_transpose(MAT *A) {
  register long i, j;
  MAT *matrix;

  if (!A)
    return NULL;
  matrix = matrix_get(A->n, A->m);
  for (i = 0; i < matrix->n; i++)
    for (j = 0; j < matrix->m; j++)
      matrix->me[i][j] = A->me[j][i];
  return matrix;
}

MAT *matrix_add(MAT *mat1, MAT *mat2) {
  int i;
  MAT *matrix = NULL;
  if (!mat1 || !mat2)
    SDDS_Bomb("Memory is not allocated for the addition matrices.\n");
  if (mat1->m != mat2->m || mat1->n != mat2->n)
    SDDS_Bomb("The input matrix rows and columns do not match!");
  matrix = matrix_get(mat1->m, mat1->n);
  for (i = 0; i < matrix->m * matrix->n; i++)
    matrix->base[i] = mat1->base[i] + mat2->base[i];
  return matrix;
}

/* add mat2 to mat1 without allocating new memory (save memory) */
int32_t matrix_add_sm(MAT *mat1, MAT *mat2) {
  int i;
  if (mat1->m != mat2->m || mat1->n != mat2->n)
    return 0;
  for (i = 0; i < mat1->n * mat1->m; i++)
    mat1->base[i] += mat2->base[i];
  return 1;
}

MAT *matrix_sub(MAT *mat1, MAT *mat2) {
  int i;
  MAT *matrix = NULL;

  if (!mat1 || !mat2)
    SDDS_Bomb("Memory is not allocated for the substraction matrices!");
  if (mat1->m != mat2->m || mat1->n != mat2->n)
    SDDS_Bomb("The rows and columns of input matrices do not match(matrix_suB)!");
  matrix = matrix_get(mat1->m, mat1->n);

  for (i = 0; i < mat1->n * mat1->m; i++)
    matrix->base[i] = mat1->base[i] - mat2->base[i];
  return matrix;
}

/* add mat2 to mat1 without allocating new memory (save memory) */
int32_t matrix_sub_sm(MAT *mat1, MAT *mat2) {
  int i;
  if (mat1->m != mat2->m || mat1->n != mat2->n)
    return 0;
  for (i = 0; i < mat1->n * mat1->m; i++)
    mat1->base[i] -= mat2->base[i];
  return 1;
}

/* Hadamard multiplication */
MAT *matrix_h_mult(MAT *mat1, MAT *mat2) {
  int i;
  MAT *new_mat = NULL;

  if (!mat1 || !mat2)
    SDDS_Bomb("Memory is not allocated for the hadamard multiplication matrices(matrix_h_mult)!");
  if (mat1->m != mat2->m || mat1->n != mat2->n)
    SDDS_Bomb("The rows and columns of input matrices do not match(matrix_h_mult)!");
  new_mat = matrix_get(mat1->m, mat1->n);
  for (i = 0; i < mat1->n * mat1->m; i++)
    new_mat->base[i] = mat1->base[i] * mat2->base[i];
  return new_mat;
}

/* self hadamard multiply (to save memory), multiply the mat2 to mat1 without allocating new matrix */
int32_t matrix_h_mult_sm(MAT *mat1, MAT *mat2) {
  int i;
  if (!mat1 || !mat2)
    return 0;
  if (mat1->m != mat2->m || mat1->n != mat2->n)
    return 0;
  for (i = 0; i < mat1->n * mat1->m; i++)
    mat1->base[i] *= mat2->base[i];
  return 1;
}

/* Hadamard division */
MAT *matrix_h_divide(MAT *mat1, MAT *mat2) {
  int i;
  MAT *new_mat = NULL;

  if (!mat1 || !mat2)
    SDDS_Bomb("Memory is not allocated for the hadamard division matrices(matrix_h_divide)!");
  if (mat1->m != mat2->m || mat1->n != mat2->n)
    SDDS_Bomb("The rows and columns of input matrices do not match(matrix_h_divide)!");
  new_mat = matrix_get(mat1->m, mat1->n);

  for (i = 0; i < mat1->n * mat1->m; i++)
    new_mat->base[i] = mat1->base[i] / mat2->base[i];
  return new_mat;
}

/* self divide (to save memory), dividing the mat1 by mat2 without allocating new matrix */
int32_t matrix_h_divide_sm(MAT *mat1, MAT *mat2) {
  int i;
  if (!mat1 || !mat2)
    return 0;
  if (mat1->m != mat2->m || mat1->n != mat2->n)
    return 0;
  for (i = 0; i < mat1->n * mat1->m; i++)
    mat1->base[i] /= mat2->base[i];
  return 1;
}

/*m is the row number, n is the column number */
/*base is allocated by column order */
MAT *op_matrix_mult(MAT *A, MAT *B) {
  register long i, j, k, m, n, p;
  Real *a_j;
  MAT *matrix = NULL;

  if ((p = A->n) != B->m)
    SDDS_Bomb("The columns of A and rows of B do not match, can not do multiplication of A X B (op_matrix_mult)!");
  m = A->m;
  n = B->n;

  matrix = matrix_get(m, n);
  /*A->base is in column order, change it to row order */
  p = A->n;
  a_j = (Real *)malloc(sizeof(Real) * m * p);
  for (i = 0; i < m; i++)
    for (j = 0; j < p; j++)
      a_j[i * p + j] = A->base[j * m + i];

  for (j = 0; j < n; j++)
    for (i = 0; i < m; i++) {
      matrix->me[j][i] = 0;
      for (k = 0; k < p; k++)
        matrix->me[j][i] += a_j[i * p + k] * B->base[k + j * p];
    }
  free(a_j);
  return matrix;
}

MAT *matrix_mult(MAT *mat1, MAT *mat2) {

  MAT *new_mat = NULL;
#if defined(CLAPACK) || defined(MKL)
#  if defined(ACCELERATE_NEW_LAPACK)
  __LAPACK_int lda, kk, ldb;
  double alpha = 1.0, beta = 0.0;
#  else
  long lda, kk, ldb;
  double alpha = 1.0, beta = 0.0;
#  endif
#endif
#if defined(LAPACK)
  integer lda, kk, ldb;
  doublereal alpha = 1.0, beta = 0.0;
#endif
  if (!mat1 || !mat2)
    SDDS_Bomb("Memory is not allocated (matrix_mult)!");
  if (mat1->n != mat2->m)
    SDDS_Bomb("The columns of A and rows of B do not match, can not do A X B operation(matrix_mult)!");

#if defined(CLAPACK) || defined(LAPACK) || defined(MKL)
  kk = MAX(1, mat1->n);
  lda = MAX(1, mat1->m);
  ldb = MAX(1, mat2->m);
  new_mat = matrix_get(mat1->m, mat2->n);
#  if defined(_WIN32) && defined(CLAPACK)
  {
    long long m_ll = (long long)new_mat->m;
    long long n_ll = (long long)new_mat->n;
    long long k_ll = (long long)kk;
    long long lda_ll = (long long)lda;
    long long ldb_ll = (long long)ldb;
    long long ldc_ll = (long long)new_mat->m;
  f2c_dgemm("N", "N",
            (integer *)&m_ll, (integer *)&n_ll, (integer *)&k_ll, &alpha, mat1->base,
            (integer *)&lda_ll, mat2->base, (integer *)&ldb_ll, &beta, new_mat->base, (integer *)&ldc_ll);
  }
#  else
#    if defined(LAPACK)
  {
    long long m_ll = (long long)new_mat->m;
    long long n_ll = (long long)new_mat->n;
    long long k_ll = (long long)kk;
    long long lda_ll = (long long)lda;
    long long ldb_ll = (long long)ldb;
    long long ldc_ll = (long long)new_mat->m;
  dgemm_("N", "N",
         (integer *)&m_ll, (integer *)&n_ll, (integer *)&k_ll, &alpha, mat1->base,
         (integer *)&lda_ll, mat2->base, (integer *)&ldb_ll, &beta, new_mat->base, (integer *)&ldc_ll);
  }
#    else
#      if defined(MKL)
  {
    MKL_INT m_mkl = (MKL_INT)new_mat->m;
    MKL_INT n_mkl = (MKL_INT)new_mat->n;
    MKL_INT k_mkl = (MKL_INT)kk;
    MKL_INT lda_mkl = (MKL_INT)lda;
    MKL_INT ldb_mkl = (MKL_INT)ldb;
    MKL_INT ldc_mkl = (MKL_INT)new_mat->m;
    dgemm_("N", "N",
           &m_mkl, &n_mkl, &k_mkl, &alpha, mat1->base,
           &lda_mkl, mat2->base, &ldb_mkl, &beta, new_mat->base, &ldc_mkl);
  }
#      else
#        if defined(ACCELERATE_NEW_LAPACK)
  {
    long long m_ll = (long long)new_mat->m;
    long long n_ll = (long long)new_mat->n;
    long long k_ll = (long long)kk;
    long long lda_ll = (long long)lda;
    long long ldb_ll = (long long)ldb;
    long long ldc_ll = (long long)new_mat->m;
  dgemm_("N", "N",
         (__LAPACK_int *)&m_ll, (__LAPACK_int *)&n_ll, (__LAPACK_int *)&k_ll, &alpha, mat1->base,
         (__LAPACK_int *)&lda_ll, mat2->base, (__LAPACK_int *)&ldb_ll, &beta, new_mat->base, (__LAPACK_int *)&ldc_ll);
  }
#        else

  {
    long long m_ll = (long long)new_mat->m;
    long long n_ll = (long long)new_mat->n;
    long long k_ll = (long long)kk;
    long long lda_ll = (long long)lda;
    long long ldb_ll = (long long)ldb;
    long long ldc_ll = (long long)new_mat->m;
    dgemm_("N", "N",
           (integer *)&m_ll, (integer *)&n_ll, (integer *)&k_ll, (doublereal *)&alpha, mat1->base,
           (integer *)&lda_ll, mat2->base, (integer *)&ldb_ll, (doublereal *)&beta, new_mat->base, (integer *)&ldc_ll);
  }
#        endif
#      endif
#    endif
#  endif
#else
  new_mat = op_matrix_mult(mat1, mat2);
#endif
  return new_mat;
}

MAT *matrix_invert(
                   MAT *Ain,                     /* input matrix */
                   double *weight,               /* weights for each SV */
                   int32_t largestSValue,        /* number of largest SVs to keep */
                   int32_t smallestSValue,       /* number of smallest SVs to discard */
                   double minRatio,              /* keep small SVs if SV/Max(SV)>minRatio */
                   double tikhonovRelativeAlpha, /* Tikhonov alpha parameter */
                   int32_t tikhonovN,            /* set Tikhonov alpha parameter to the nth SV */
                   int32_t deleteVectors,        /* 0/1 if caller supplying list of SVs to delete */
                   int32_t *deleteVector,        /* indices of vectors to delete */
                   char **deletedVector,         /* list of deleted vectors as a string */
                   VEC **S_Vec,                  /* return of all singular values */
                   int32_t *sValues,             /* return of number of SVs */
                   VEC **S_Vec_used,             /* return of all SVs used */
                   int32_t *usedSValues,         /* return of number of SVs used */
                   MAT **U_matrix,               /* return of U matrix */
                   MAT **Vt_matrix,              /* return of transpose of V */
                   double *conditionNum          /* return of condition number (max(SV)/min(SV)) */
                   ) {
  MAT *Inv = NULL;
  MAT *A;
#if defined(CLAPACK) || defined(LAPACK) || defined(MKL)
  MAT *U = NULL, *V = NULL, *Vt = NULL, *Invt = NULL;
  int32_t i, j, NSVUsed = 0, m, n, firstdelete = 1;
  VEC *SValue = NULL, *SValueUsed = NULL, *InvSValue = NULL;
  double max, min;
  char deletedVectors[1024];
  /*use economy svd method i.e. U matrix is rectangular not square matrix */
  char calcMode = 'S';
#endif
#if defined(CLAPACK) || defined(MKL)
  double *work;
  long lda;
  double alpha = 1.0, beta = 0.0;
  int kk, ldb;
#endif
#if defined(LAPACK)
  doublereal *work;
  integer lda;
  double alpha = 1.0, beta = 0.0;
  int kk, ldb;
#endif

  A = matrix_copy(Ain); /* To avoid changing users matrix */

#if defined(CLAPACK) || defined(LAPACK) || defined(MKL)
  if (!A || A->m <= 0 || A->n <= 0)
    SDDS_Bomb("Invalid matrix provided for invert (matrix_invert)!");
  n = A->n;
  m = A->m;
  deletedVectors[0] = '\0';
  if (S_Vec)
    SValue = *S_Vec;
  if (S_Vec_used)
    SValueUsed = *S_Vec_used;
  if (U_matrix)
    U = *U_matrix;
  if (Vt_matrix)
    V = *Vt_matrix;
  if (!SValue)
    SValue = vec_get(A->n);
  if (!SValueUsed)
    SValueUsed = vec_get(A->n);
  if (!InvSValue)
    InvSValue = vec_get(A->n);
  if (!Vt)
    Vt = matrix_get(A->n, A->n);
  if (!U)
    U = matrix_get(A->m, MIN(A->m, A->n));
  if (weight) {
    /* multiply ith row of A by weight[i] */
    for (i = 0; i < A->m; i++)
      for (j = 0; j < A->n; j++)
        Mij(A, i, j) *= weight[i];
  }
#  if defined(CLAPACK) && !defined(ACCELERATE_NEW_LAPACK)
  work = (double *)malloc(sizeof(double) * 1);
  {
    long long m_ll = (long long)A->m;
    long long n_ll = (long long)A->n;
    long long lda_ll = (long long)MAX(1, A->m);
    long long ldu_ll = (long long)A->m;
    long long ldvt_ll = (long long)A->n;
    long long lwork_ll = -1;
    long long info_ll = 0;
    dgesvd_((char *)&calcMode, (char *)&calcMode, (long *)&m_ll, (long *)&n_ll,
            (double *)A->base, (long *)&lda_ll,
            (double *)SValue->ve,
            (double *)U->base, (long *)&ldu_ll,
            (double *)Vt->base, (long *)&ldvt_ll,
            (double *)work, (long *)&lwork_ll,
            (long *)&info_ll);
    if (info_ll < 0) {
      char msg[256];
      snprintf(msg, sizeof(msg), "Error calling dgesvd (workspace query) in matrix_invert: argument %lld illegal (possible LAPACK ABI mismatch LP64/ILP64).", -info_ll);
      SDDS_Bomb(msg);
    }
    lwork_ll = (long long)work[0];
    if (lwork_ll < 1)
      SDDS_Bomb("Invalid workspace size returned by dgesvd in matrix_invert.");
    work = (double *)realloc(work, sizeof(double) * (size_t)lwork_ll);
    dgesvd_((char *)&calcMode, (char *)&calcMode, (long *)&m_ll, (long *)&n_ll,
            (double *)A->base, (long *)&lda_ll,
            (double *)SValue->ve,
            (double *)U->base, (long *)&ldu_ll,
            (double *)Vt->base, (long *)&ldvt_ll,
            (double *)work, (long *)&lwork_ll,
            (long *)&info_ll);
    if (info_ll < 0) {
      char msg[256];
      snprintf(msg, sizeof(msg), "Error calling dgesvd in matrix_invert: argument %lld illegal (possible LAPACK ABI mismatch LP64/ILP64).", -info_ll);
      SDDS_Bomb(msg);
    }
  }
  free(work);
#  endif
#  if defined(CLAPACK) && defined(ACCELERATE_NEW_LAPACK)
  work = (double *)malloc(sizeof(double) * 1);
  {
    long long m_ll = (long long)A->m;
    long long n_ll = (long long)A->n;
    long long lda_ll = (long long)MAX(1, A->m);
    long long ldu_ll = (long long)A->m;
    long long ldvt_ll = (long long)A->n;
    long long lwork_ll = -1;
    long long info_ll = 0;

    dgesvd_((char *)&calcMode, (char *)&calcMode, (__LAPACK_int *)&m_ll, (__LAPACK_int *)&n_ll,
            (double *)A->base, (__LAPACK_int *)&lda_ll,
            (double *)SValue->ve,
            (double *)U->base, (__LAPACK_int *)&ldu_ll,
            (double *)Vt->base, (__LAPACK_int *)&ldvt_ll,
            (double *)work, (__LAPACK_int *)&lwork_ll,
            (__LAPACK_int *)&info_ll);

    lwork_ll = (long long)work[0];
    if (lwork_ll < 1)
      SDDS_Bomb("Invalid workspace size returned by dgesvd in matrix_invert.");
    work = (double *)realloc(work, sizeof(double) * (size_t)lwork_ll);

    dgesvd_((char *)&calcMode, (char *)&calcMode, (__LAPACK_int *)&m_ll, (__LAPACK_int *)&n_ll,
            (double *)A->base, (__LAPACK_int *)&lda_ll,
            (double *)SValue->ve,
            (double *)U->base, (__LAPACK_int *)&ldu_ll,
            (double *)Vt->base, (__LAPACK_int *)&ldvt_ll,
            (double *)work, (__LAPACK_int *)&lwork_ll,
            (__LAPACK_int *)&info_ll);
  }
  free(work);
#  endif
#  if defined(MKL)
  {
    MKL_INT m_mkl = (MKL_INT)A->m;
    MKL_INT n_mkl = (MKL_INT)A->n;
    MKL_INT lda_mkl = (MKL_INT)MAX(1, A->m);
    MKL_INT ldu_mkl = (MKL_INT)A->m;
    MKL_INT ldvt_mkl = (MKL_INT)A->n;
    MKL_INT lwork_mkl = (MKL_INT)-1;
    MKL_INT info_mkl = 0;

    work = (double *)malloc(sizeof(double) * 1);
    dgesvd_((char *)&calcMode, (char *)&calcMode, &m_mkl, &n_mkl,
            (double *)A->base, &lda_mkl,
            (double *)SValue->ve,
            (double *)U->base, &ldu_mkl,
            (double *)Vt->base, &ldvt_mkl,
            (double *)work, &lwork_mkl,
            &info_mkl);

    if (info_mkl != 0)
      SDDS_Bomb("Error calling dgesvd (workspace query) in matrix_invert.");

    lwork_mkl = (MKL_INT)work[0];
    if (lwork_mkl < 1)
      SDDS_Bomb("Invalid workspace size returned by dgesvd in matrix_invert.");
    work = (double *)realloc(work, sizeof(double) * (size_t)lwork_mkl);

    dgesvd_((char *)&calcMode, (char *)&calcMode, &m_mkl, &n_mkl,
            (double *)A->base, &lda_mkl,
            (double *)SValue->ve,
            (double *)U->base, &ldu_mkl,
            (double *)Vt->base, &ldvt_mkl,
            (double *)work, &lwork_mkl,
            &info_mkl);
    free(work);

  }
#  endif
#  if defined(LAPACK)
  work = (doublereal *)malloc(sizeof(doublereal) * 1);
  {
    long long m_ll = (long long)A->m;
    long long n_ll = (long long)A->n;
    long long lda_ll = (long long)MAX(1, A->m);
    long long ldu_ll = (long long)A->m;
    long long ldvt_ll = (long long)A->n;
    long long lwork_ll = -1;
    long long info_ll = 0;

    dgesvd_((char *)&calcMode, (char *)&calcMode, (integer *)&m_ll, (integer *)&n_ll,
            (doublereal *)A->base, (integer *)&lda_ll,
            (doublereal *)SValue->ve,
            (doublereal *)U->base, (integer *)&ldu_ll,
            (doublereal *)Vt->base, (integer *)&ldvt_ll,
            (doublereal *)work, (integer *)&lwork_ll,
            (integer *)&info_ll);

    if (info_ll < 0) {
      char msg[256];
      snprintf(msg, sizeof(msg), "Error calling dgesvd (workspace query) in matrix_invert: argument %lld illegal (possible LAPACK ABI mismatch LP64/ILP64).", -info_ll);
      SDDS_Bomb(msg);
    }

    lwork_ll = (long long)work[0];
    if (lwork_ll < 1)
      SDDS_Bomb("Invalid workspace size returned by dgesvd in matrix_invert.");
    work = (doublereal *)realloc(work, sizeof(doublereal) * (size_t)lwork_ll);

    dgesvd_((char *)&calcMode, (char *)&calcMode, (integer *)&m_ll, (integer *)&n_ll,
            (doublereal *)A->base, (integer *)&lda_ll,
            (doublereal *)SValue->ve,
            (doublereal *)U->base, (integer *)&ldu_ll,
            (doublereal *)Vt->base, (integer *)&ldvt_ll,
            (doublereal *)work, (integer *)&lwork_ll,
            (integer *)&info_ll);

    if (info_ll < 0) {
      char msg[256];
      snprintf(msg, sizeof(msg), "Error calling dgesvd in matrix_invert: argument %lld illegal (possible LAPACK ABI mismatch LP64/ILP64).", -info_ll);
      SDDS_Bomb(msg);
    }

  }
  free(work);
#  endif

  NSVUsed = 1;
  for (i = 0; i < n; i++) {
    if ((SValueUsed->ve[i] = SValue->ve[i]) != 0)
      InvSValue->ve[i] = 1 / SValue->ve[i];
    else
      InvSValue->ve[i] = 0;
  }

  /* Apply Tikhonov regularization if requested */
  if (tikhonovRelativeAlpha > 0 || (tikhonovN >= 0 && tikhonovN < n)) {
    double tikhonovAlpha;
    if (tikhonovRelativeAlpha > 0) {
      double maxSV;
      maxSV = 0;
      for (i = 0; i < n; i++)
        if (SValue->ve[i] > maxSV)
          maxSV = SValue->ve[i];
      tikhonovAlpha = maxSV * tikhonovRelativeAlpha;
    } else
      tikhonovAlpha = SValue->ve[tikhonovN];
    if (tikhonovAlpha > 0) {
      for (i = 0; i < n; i++) {
        if (SValue->ve[i]) {
          InvSValue->ve[i] = SValue->ve[i] / (sqr(SValue->ve[i]) + sqr(tikhonovAlpha));
          SValueUsed->ve[i] = 1 / InvSValue->ve[i];
        }
      }
    }
  }

  max = 0;
  min = HUGE;
  max = MAX(SValueUsed->ve[0], max);
  min = MIN(SValueUsed->ve[0], min);

  /*
    1) first remove SVs that are exactly zero
    2) remove SV according to ratio option
    3) remove SV according to largests option
    4) remove SV of user-selected vectors
  */

  for (i = 1; i < n; i++) {
    if (!SValueUsed->ve[i]) {
      InvSValue->ve[i] = 0;
    } else if ((SValueUsed->ve[i] / SValueUsed->ve[0]) < minRatio) {
      InvSValue->ve[i] = 0;
      SValueUsed->ve[i] = 0;
    } else if (largestSValue && i >= largestSValue) {
      InvSValue->ve[i] = 0;
      SValueUsed->ve[i] = 0;
    } else if (smallestSValue && i >= (n - smallestSValue)) {
      InvSValue->ve[i] = 0;
      SValueUsed->ve[i] = 0;
    } else {
      max = MAX(SValueUsed->ve[i], max);
      min = MIN(SValueUsed->ve[i], min);
      NSVUsed++;
    }
  }

  /*4) remove SV of user-selected vectors -
    delete vector given in the -deleteVectors option
    by setting the inverse singular values to 0*/
  if (deleteVector) {
    char buffer[100];
    for (i = 0; i < deleteVectors; i++) {
      if (0 <= deleteVector[i] && deleteVector[i] < n) {
        if (firstdelete)
          sprintf(deletedVectors, "%d", deleteVector[i]);
        else {
          sprintf(buffer, " %d", deleteVector[i]);
          strcat(deletedVectors, buffer);
        }
        firstdelete = 0;
        InvSValue->ve[deleteVector[i]] = 0;
        SValueUsed->ve[deleteVector[i]] = 0;
        if (largestSValue && deleteVector[i] >= largestSValue)
          break;
        NSVUsed--;
      }
    }
    if (deletedVector)
      SDDS_CopyString(deletedVector, deletedVectors);
  }

  if (conditionNum)
    *conditionNum = max / min;
  /* R = U S Vt and Inv = V SInv Ut, Inv is nxm matrix*/
  /* then Invt = U Sinv Vt , Invt is mxn matrix*/
  Invt = matrix_get(m, n);
  V = matrix_get(Vt->m, Vt->n);
  /* U ant Vt matrix (in column order) are available */
  for (i = 0; i < Vt->n; i++) {
    for (kk = 0; kk < n; kk++)
      V->base[i * V->m + kk] = Vt->base[i * V->m + kk] * InvSValue->ve[kk];
  }
  vec_free(InvSValue);
  /* Rinvt = U Sinv Vt = U V */
  /* note that, the arguments of dgemm should be
     U->m -- should be the row number of product matrix, which is U x V
     V->n -- the column number of the product matrix
     kk   -- should be the column number of U and row number of V, whichever is smaller (otherwise, memory access error)
     lda  -- should be the row number of U matrix
     ldb  -- should be the row number of V matrix */
  kk = MIN(U->n, V->m);
  lda = MAX(1, U->m);
  ldb = MAX(1, V->m);
#  if defined(MKL)
  {
    MKL_INT um_mkl = (MKL_INT)U->m;
    MKL_INT vn_mkl = (MKL_INT)V->n;
    MKL_INT kk_mkl = (MKL_INT)kk;
    MKL_INT lda_mkl = (MKL_INT)lda;
    MKL_INT ldb_mkl = (MKL_INT)ldb;
    MKL_INT ldc_mkl = (MKL_INT)U->m;
    dgemm_("N", "N",
           &um_mkl, &vn_mkl, &kk_mkl, &alpha, U->base,
           &lda_mkl, V->base, &ldb_mkl, &beta, Invt->base, &ldc_mkl);
  }
#  else
  {
    long long um_ll = (long long)U->m;
    long long vn_ll = (long long)V->n;
    long long kk_ll = (long long)kk;
    long long lda_ll = (long long)lda;
    long long ldb_ll = (long long)ldb;
    long long ldc_ll = (long long)U->m;
#    if defined(_WIN32) && defined(CLAPACK)
    f2c_dgemm("N", "N",
              (integer *)&um_ll, (integer *)&vn_ll, (integer *)&kk_ll, &alpha, U->base,
              (integer *)&lda_ll, V->base, (integer *)&ldb_ll, &beta, Invt->base, (integer *)&ldc_ll);
#    else
#      if defined(LAPACK)
    dgemm_("N", "N",
           (integer *)&um_ll, (integer *)&vn_ll, (integer *)&kk_ll, &alpha, U->base,
           (integer *)&lda_ll, V->base, (integer *)&ldb_ll, &beta, Invt->base, (integer *)&ldc_ll);
#      else
#        if defined(ACCELERATE_NEW_LAPACK)
    dgemm_("N", "N",
           (__LAPACK_int *)&um_ll, (__LAPACK_int *)&vn_ll, (__LAPACK_int *)&kk_ll, &alpha, U->base,
           (__LAPACK_int *)&lda_ll, V->base, (__LAPACK_int *)&ldb_ll, &beta, Invt->base, (__LAPACK_int *)&ldc_ll);
#        else
    dgemm_("N", "N",
           (integer *)&um_ll, (integer *)&vn_ll, (integer *)&kk_ll, (doublereal *)&alpha, U->base,
           (integer *)&lda_ll, V->base, (integer *)&ldb_ll, (doublereal *)&beta, Invt->base, (integer *)&ldc_ll);
#        endif
#      endif
#    endif
  }
#  endif
  matrix_free(V);
  if (Vt_matrix) {
    if (*Vt_matrix)
      matrix_free(*Vt_matrix);
    *Vt_matrix = Vt;
  } else
    matrix_free(Vt);
  if (U_matrix) {
    if (*U_matrix)
      matrix_free(*U_matrix);
    *U_matrix = U;
  } else
    matrix_free(U);
  if (S_Vec) {
    if (*S_Vec)
      vec_free(*S_Vec);
    *S_Vec = SValue;
  } else
    vec_free(SValue);
  if (sValues)
    *sValues = SValue->dim;
  if (S_Vec_used) {
    if (*S_Vec_used)
      vec_free(*S_Vec_used);
    *S_Vec_used = SValueUsed;
    *usedSValues = NSVUsed;
  } else
    vec_free(SValueUsed);
  /*transpose Invt to Inv */
  Inv = matrix_transpose(Invt);
  matrix_free(Invt);
  if (weight) {
    /* multiply ith column of inverse by weight[i] */
    for (i = 0; i < Inv->n; i++)
      for (j = 0; j < Inv->m; j++)
        Mij(Inv, j, i) *= weight[i];
  }
#else
  SDDS_Bomb("Matrix inversion is unavailable for non LAPACK/CLAPACK versions of elegant.");
#endif
  matrix_free(A);
  return Inv;
}

MAT *matrix_identity(int32_t m, int32_t n) {
  register long i, j;
  register double *a_j;
  MAT *matrix = NULL;
  matrix = matrix_get(m, n);
  for (j = 0; j < n; j++) {
    a_j = matrix->me[j];
    for (i = 0; i < m; i++)
      a_j[i] = (i == j ? 1 : 0);
  }
  return matrix;
}

/* the data is one dimension array, ordered by column */
void *SDDS_GetCastMatrixOfRowsByColumn(SDDS_DATASET *SDDS_dataset, int32_t *n_rows, long sddsType) {
  void *data;
  long i, j, k, size;
  /*  long type;*/
  if (!SDDS_CheckDataset(SDDS_dataset, "SDDS_GetCastMatrixOfRowsByColumn"))
    return (NULL);
  if (!SDDS_NUMERIC_TYPE(sddsType)) {
    SDDS_SetError("Unable to get matrix of rows--no columns selected (SDDS_GetCastMatrixOfRowsByColumn) (1)");
    return NULL;
  }
  if (SDDS_dataset->n_of_interest <= 0) {
    SDDS_SetError("Unable to get matrix of rows--no columns selected (SDDS_GetCastMatrixOfRowsByColumn) (2)");
    return (NULL);
  }
  if (!SDDS_CheckTabularData(SDDS_dataset, "SDDS_GetCastMatrixOfRowsByColumn"))
    return (NULL);
  size = SDDS_type_size[sddsType - 1];
  if ((*n_rows = SDDS_CountRowsOfInterest(SDDS_dataset)) <= 0) {
    SDDS_SetError("Unable to get matrix of rows--no rows of interest (SDDS_GetCastMatrixOfRowsByColumn) (3)");
    return (NULL);
  }
  for (i = 0; i < SDDS_dataset->n_of_interest; i++) {
    if (!SDDS_NUMERIC_TYPE(SDDS_dataset->layout.column_definition[SDDS_dataset->column_order[i]].type)) {
      SDDS_SetError("Unable to get matrix of rows--not all columns are numeric ()SDDS_GetCastMatrixOfRowsByColumn (4)");
      return NULL;
    }
  }
  if (!(data = (void *)SDDS_Malloc(size * (*n_rows) * SDDS_dataset->n_of_interest))) {
    SDDS_SetError("Unable to get matrix of rows--memory allocation failure (SDDS_GetCastMatrixOfRowsByColumn) (5)");
    return (NULL);
  }
  for (j = k = 0; j < SDDS_dataset->n_rows; j++) {
    if (SDDS_dataset->row_flag[j]) {
      for (i = 0; i < SDDS_dataset->n_of_interest; i++)
        SDDS_CastValue(SDDS_dataset->data[SDDS_dataset->column_order[i]], j,
                       SDDS_dataset->layout.column_definition[SDDS_dataset->column_order[i]].type,
                       sddsType, (char *)data + (k + i * SDDS_dataset->n_rows) * sizeof(Real));
      k++;
    }
  }
  return (data);
}

double matrix_det(MAT *A) {
  double det = 1.0;
  MAT *B;

#if defined(CLAPACK) || defined(MKL)
#  if defined(ACCELERATE_NEW_LAPACK)
  __LAPACK_int i, lda, n, m, *ipvt, info;
#  else
  long i, lda, n, m, *ipvt, info;
#  endif
#endif
#if defined(LAPACK)
  integer i, lda, n, m, *ipvt, info;
#endif

  if (A->m != A->n)
    return 0;
#if defined(LAPACK) || defined(CLAPACK) || defined(MKL)
  lda = A->m;
  n = A->n;
  m = A->m;
  /* Allocate pivots large enough for ILP64 runtimes (INTEGER*8). */
  ipvt = calloc((size_t)n, 8);
  if (!ipvt)
    SDDS_Bomb("Memory allocation failure in matrix_det (ipvt). Possible out-of-memory.");
  B = matrix_copy(A);
  /*LU decomposition*/
#  if defined(MKL)
  {
    MKL_INT m_mkl = (MKL_INT)m;
    MKL_INT n_mkl = (MKL_INT)n;
    MKL_INT lda_mkl = (MKL_INT)lda;
    MKL_INT info_mkl = 0;
    MKL_INT *ipvt_mkl = (MKL_INT *)calloc((size_t)n_mkl, sizeof(*ipvt_mkl));
    if (!ipvt_mkl)
      SDDS_Bomb("Memory allocation failure in matrix_det (ipvt_mkl).");
    dgetrf_(&m_mkl, &n_mkl, B->base, &lda_mkl, ipvt_mkl, &info_mkl);
    info = (long)info_mkl;
    free(ipvt_mkl);
  }
#  else
  dgetrf_(&m, &n, B->base, &lda, ipvt, &info);
#  endif

  if (info < 0) {
    fprintf(stderr, "Error in LU decomposition, the %d-th argument had an illegal value.\n", (int)(-info));
    return 0;
  } else if (info > 0) {
    fprintf(stderr, "Error in LU decomposition, U(%d,%d) is exactly zero. The factorization has been completed, but the factor U is exactly singular, and division by zero will occur if it is used to solve a system of equations.\n", (int)n, (int)m);
    return 0;
  }
  for (i = 0; i < n; i++)
    det *= B->me[i][i];
  matrix_free(B);
#else
  SDDS_Bomb("The matrix determinant is not implemented for non LAPACK/CLAPACK library.");
#endif
  return det;
}

/* routine: m_show()
 * purpose: to display a matrix on the screen
 * usage:
 *   MAT *A;
 *   ...
 *   m_show(A, "%.4f  ", "label", stdout);   ! print w/a certain format
 *
 * Michael Borland, 1986, 2008.
 */

void matrix_show(
                 MAT *A,
                 char *format,
                 char *label,
                 FILE *fp) {
  register long i, j;

  if (label)
    fputs(label, fp);
  for (i = 0; i < A->m; i++) {
    for (j = 0; j < A->n; j++)
      fprintf(fp, format, Mij(A, i, j));
    fputc('\n', fp);
  }
}

int32_t matrix_check(MAT *A) {
  long i;
  if (!A->base)
    return 0;
  if (!A->me)
    return 0;
  for (i = 0; i < A->n; i++)
    if (!A->me[i])
      return 0;
  return 1;
}

/* Multiply matrix by a scalar */
void matrix_scmul(MAT *mat1, double scalar) {
  long i;
  for (i = 0; i < mat1->m * mat1->n; i++)
    mat1->base[i] *= scalar;
}

VEC *vec_addition(VEC *a, VEC *b, double as, double bs, VEC *sum)
/* sum = as*a + bs*b */
{
  long i;
  if (a->dim != b->dim || a->dim != sum->dim)
    return NULL;
  for (i = 0; i < a->dim; i++)
    sum->ve[i] = as * a->ve[i] + bs * b->ve[i];
  return sum;
}

double vec_dot(VEC *a, VEC *b) {
  double sum;
  long i;

  if (a->dim != b->dim)
    return -1;
  for (i = sum = 0; i < a->dim; i++)
    sum += a->ve[i] * b->ve[i];
  return sum;
}

VEC *vec_cross(VEC *a, VEC *b, VEC *out)
/* out = a x b  for 3-vectors only */
{
  double o[3];
  if (a->dim != 3 || b->dim != 3 || out->dim != 3)
    return NULL;
  o[0] = a->ve[1] * b->ve[2] - a->ve[2] * b->ve[1];
  o[1] = a->ve[2] * b->ve[0] - a->ve[0] * b->ve[2];
  o[2] = a->ve[0] * b->ve[1] - a->ve[1] * b->ve[0];
  memcpy(out->ve, o, 3 * sizeof(out->ve[0]));
  return out;
}

int lsf2dPolyUnweighted(
                        /* input */
                        double *x[2],      /* independent variable values */
                        double *y,         /* dependent variable values */
                        long points,       /* number of points */
                        int32_t *order[2], /* pairs of orders for terms */
                        long nOrders,      /* number of pairs of orders for terms */
                        /* output  */
                        /* main */
                        double *coef,      /* polynomial coefficients */
                        double *chi,       /* reduced chi squared */
                        double *condition, /* condition number from SVD matrix inversion */
                        double *diff       /* array of differences between fit and y values */
                        ) {
  MAT *X, *A, *Y, *K, *Xc;
  double *weight;
  long i, j;

  X = matrix_get(points, nOrders);
  Y = matrix_get(points, 1);
  weight = tmalloc(sizeof(*weight) * points);

  for (i = 0; i < points; i++) {
    weight[i] = 1;
    Mij(Y, i, 0) = y[i];
    for (j = 0; j < nOrders; j++)
      Mij(X, i, j) = ipow(x[0][i], order[0][j]) * ipow(x[1][i], order[1][j]);
  }

  /* Equation is Y=X*K
   * A = Inv(X)
   * K = A*Y
   */
  Xc = matrix_copy(X);
  A = matrix_invert(X, NULL, 0, 0, 0, 0, -1, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, condition);
  K = matrix_mult(A, Y);

  for (i = 0; i < nOrders; i++)
    coef[i] = Mij(K, i, 0);

  /* evaluate the polynomial at the fit points */
  matrix_free(Y);
  Y = matrix_mult(Xc, K);
  *chi = 0;
  for (i = 0; i < points; i++) {
    diff[i] = y[i] - Mij(Y, i, 0);
    *chi += sqr(diff[i]) * weight[i];
  }
  if (nOrders < points)
    *chi /= (points - nOrders);
  else
    *chi = -1;

  matrix_free(X);
  matrix_free(Xc);
  matrix_free(Y);
  matrix_free(A);
  matrix_free(K);

  return 1;
}
