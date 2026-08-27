/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: matrix.c--3rd-order matrix routines for tracking.
 * contents:  print_matrices(), initialize_matrices(), null_matrices(),
 *            free_matrices(), track_particles(), 
 *
 * It is assumed that Tijk and Tikj for k<j have been combined into 
 * one element, namely, Tijk, which has been multiplied by 2.
 * Similarly:
 *    -- since Qijkk==Qikjk==Qikkj, for j>k, these elements have been combined
 *       as 3*Qijkk.
 *    -- since Qijjk==Qijkj==Qikjj, for j>k, these elements have been combined
 *       as 3*Qijjk.
 *    -- since Qijkl==Qijlk, for j>k and k>l, these elements have been combined
 *       as 6*Qijkl.
 *
 * Michael Borland, 1989.
 */
#include "mdb.h"
#include "track.h"
#ifdef HAVE_GPU
#  include <gpu_matrix.h>
#endif

void print_matrices(FILE *fp, char *string, VMATRIX *M, double suppressBelow) {
  print_matrices1(fp, string, "%22.15e ", M, suppressBelow);
}

void print_matrices1(FILE *fp, char *string, char *format, VMATRIX *M, double suppressBelow) {
  long i, j, k, l;
  double *C;
  double **R;
  double ***T;
  double ****Q;

  log_entry("print_matrices");

  set_matrix_pointers(&C, &R, &T, &Q, M);

  fprintf(fp, "%s\nC:   ", string);
  for (i = 0; i < 6; i++)
    fprintf(fp, format, fabs(C[i])<suppressBelow ? 0.0 : C[i]);
  fputc('\n', fp);

  for (i = 0; i < 6; i++) {
    fprintf(fp, "R%ld: ", i + 1);
    for (j = 0; j < 6; j++)
      fprintf(fp, format, fabs(R[i][j])<suppressBelow ? 0.0 : R[i][j]);
    fputc('\n', fp);
  }

  if (M->T && M->order >= 2) {
    for (i = 0; i < 6; i++) {
      for (j = 0; j < 6; j++) {
        fprintf(fp, "T%ld%ld: ", i + 1, j + 1);
        for (k = 0; k <= j; k++)
          fprintf(fp, format, fabs(T[i][j][k])<suppressBelow ? 0.0 : T[i][j][k]);
        fputc('\n', fp);
      }
    }
  }

  if (M->Q && M->order >= 3) {
    for (i = 0; i < 6; i++) {
      for (j = 0; j < 6; j++) {
        for (k = 0; k <= j; k++) {
          fprintf(fp, "Q%ld%ld%ld: ", i + 1, j + 1, k + 1);
          for (l = 0; l <= k; l++)
            fprintf(fp, format, fabs(Q[i][j][k][l])<suppressBelow ? 0.0 : Q[i][j][k][l]);
          fputc('\n', fp);
        }
      }
    }
  }
  
  if (M->maxError) {
    fprintf(fp, "MaxErr: ");
    for (j=0; j<6; j++)
      fprintf(fp, format, M->maxError[j]);
    fprintf(fp, "\n");
    fprintf(fp, "MAErr:  ");
    for (j=0; j<6; j++)
      fprintf(fp, format, M->meanAbsError[j]);
    fprintf(fp, "\n");
  }

  log_exit("print_matrices");
}

void initialize_matrices(VMATRIX *M, long order) {
  long i, j, k;
  double *C, **R;
  double ***T;
  double ****Q;

  log_entry("initialize_matrices");
  M->eptr = NULL;
  M->maxError = M->meanAbsError = NULL;

  // Contiguous memory regions will allow strided access and vectorization
  // Actually...maybe not, this ragged array stuff in order 2/3 is not good

  if (order < 1 || order > 3) {
    printf("invalid order: %ld  (initialize_matrices)\n", order);
    fflush(stdout);
    exitElegant(1);
  }

  M->C = C = tmalloc(sizeof(*C) * 6);
  double *Rb = (double *)tmalloc(sizeof(double) * (6+36));
  M->R = R = (double **)Rb;
  double *Rbuf = Rb+6;
//  int s = (6+36);
//  if (order >= 2)
//    s += (6+36+126);
//  if (order >= 3)
//    s += (6+36+126+336);

  for (i = 0; i < 6; i++)
    R[i] = Rbuf + i*6;

  M->T = NULL;
  if (order >= 2) {
    // Tijk is ragged, Tijk and Tikj for k<j have been combined
    // So we need sum(i:0-5,j:0-5,k:j+1) = 6*sum((j+1),j:0-5) = 6 * (1+2+3+4+5+6) = 126
    double *Tb = tmalloc(sizeof(double) * (126) + sizeof(char*)*(6+36));
    M->T = T = (double ***)Tb;
    double **T2 = (double **)(Tb+6);
    double *Tbuf = (Tb+6+36);
    long pos = 0;
    for (i = 0; i < 6; i++) {
      T[i] = T2 + i*6;
      for (j = 0; j < 6; j++) {
        T[i][j] = Tbuf + pos;
        pos += j+1;
      }
    }
//    if (pos >= 126+1) {
//      printf("invalid state: %ld  (T)\n", pos);
//      bombElegant("I am bad at T matrices, send help", NULL);
//    }
  }

  M->Q = NULL;
  if (order >= 3) {
    // sum(i:0-5;j:0-5;k:0-j;l:k+1) = 336 (data)
    // sum(i:0-5;j:0-5;k:j+1) = 126 (last level pointers)
    double *Qb = tmalloc(sizeof(double) * (336) + sizeof(char*)*(6+36+126));
    M->Q = Q = (double ****)Qb;
    double ***Q2 = (double ***)(Qb+6);
    double **Q3 = (double **)(Qb+6+36);
    double *Qbuf = (Qb+6+36+126);
    long pos = 0;
    long pos3 = 0;
    for (i = 0; i < 6; i++) {
      Q[i] = Q2 + i*6;
      for (j = 0; j < 6; j++) {
        Q[i][j] = Q3 + pos3;
        pos3 += j+1;
        for (k = 0; k <= j; k++) {
          Q[i][j][k] = Qbuf + pos;
          pos += k+1;
//          if (Qposition >= 336+1 || Qpos_level3 >= 126+1) {
//            printf("invalid state: %ld %ld (Q)\n", Qposition, Qpos_level3);
//            bombElegant("I am bad at Q matrices, send help", NULL);
//          }
        }
      }
    }
  }

  M->order = order;
  log_exit("initialize_matrices");
}

void null_matrices(VMATRIX *M, unsigned long flags) {
  long i, j;
  double *C, **R, ***T, ****Q;

  M->eptr = NULL;

  set_matrix_pointers(&C, &R, &T, &Q, M);

  if (M->order == 3 && !(flags & EXCLUDE_Q) && Q)
    memset(***Q, 0, 336);

  if (M->order >= 2 && !(flags & EXCLUDE_T) && T)
    memset(**T, 0, 126);

  if (M->order >= 1 && !(flags & EXCLUDE_R) && R)
    for (i = 0; i < 6; i++)
      for (j = 0; j < 6; j++)
        R[i][j] = (flags & SET_UNIT_R) && (i == j) ? 1 : 0;

  if (!(flags & EXCLUDE_C) && C)
    for (i = 0; i < 6; i++)
      C[i] = 0;
}

void track_particles(double **final, VMATRIX *M, double **initial,
                     int64_t n_part) {
  double sum1;
  double *Tij, sum, coord_j;
  long k, j;
  long i, l;
  int64_t i_part;
  double coord_k, coord_jk;
  double **Qij, *Qijk;
  double *C, **R, ***T, ****Q;
  double *fin, *ini;
  static double temp[6];

#ifdef HAVE_GPU
#  ifdef GPU_VERIFY
  char fname[20];
#  endif /* GPU_VERIFY */
  if (getElementOnGpu()) {
    if (gpu_matrix_supported(M)) {
      startGpuTimer();
      gpu_track_particles(M, n_part);
#  ifdef GPU_VERIFY
      startCpuTimer();
      track_particles(final, M, initial, n_part);
      sprintf(fname, "track_particles_M%ld", M->order);
      compareGpuCpu(n_part, fname);
#  endif /* GPU_VERIFY */
      return;
    }
    forceParticlesToCpu("unsupported matrix for CUDA track_particles");
  }
#endif /* HAVE_GPU */

  log_entry("track_particles");

  if (!M) {
    TRACKING_CONTEXT tc;
    getTrackingContext(&tc);
    bombElegantVA("NULL VMATRIX pointer in track_particles (%s #%ld)",
                  tc.elementName, tc.elementOccurrence);
  }
#if !SDDS_MPI_IO
  if (!final)
    bombElegant("NULL final coordinates pointer in track_particles", NULL);
  if (!initial)
    bombElegant("NULL initial coordinates pointer in track_particles", NULL);
#endif

  set_matrix_pointers(&C, &R, &T, &Q, M);

  // TODO: move all the null checks outside the per-particle loop

  switch (M->order) {
  case 3:
    if (!C)
      bombElegant("NULL C pointer (track_particles)", NULL);
    if (!R)
      bombElegant("NULL R pointer (track_particles)", NULL);
    if (!T)
      bombElegant("NULL T pointer (track_particles)", NULL);
    if (!Q)
      bombElegant("NULL Q pointer (track_particles)", NULL);
    for (i_part = n_part - 1; i_part >= 0; i_part--) {
      if (!(fin = final[i_part])) {
        printf("error: final coordinate pointer is NULL for particle %ld (track_particles)\n", i_part);
        fflush(stdout);
        abort();
      }
      if (!(ini = initial[i_part])) {
        printf("error: final coordinate pointer is NULL for particle %ld (track_particles)\n", i_part);
        fflush(stdout);
        abort();
      }
      fin[6] = ini[6]; /* copy particle ID # */
      for (i = 5; i >= 0; i--) {
        sum = C[i];
        for (j = 5; j >= 0; j--) {
          if ((coord_j = ini[j]) != 0) {
            sum += R[i][j] * coord_j;
            Tij = T[i][j] + j;
            Qij = Q[i][j] + j;
            for (k = j; k >= 0; k--, Tij--) {
              Qijk = *(Qij--) + k;
              if ((coord_k = ini[k]) != 0) {
                coord_jk = coord_j * coord_k;
                sum += *Tij * coord_jk;
                for (l = k; l >= 0; l--)
                  sum += *(Qijk--) * coord_jk * ini[l];
              }
            }
          }
        }
        temp[i] = sum; /* to prevent changing initial values in 
                                  the event initial and final are same*/
      }
      for (i = 5; i >= 0; i--)
        fin[i] = temp[i];
    }
    break;
  case 2:
    for (i_part = 0; i_part < n_part; i_part++) {
      fin = final[i_part];
      ini = initial[i_part];
      fin[6] = ini[6]; /* copy particle ID # */

      //Because Tijk is not a standard matrix we have to do pointer walking
      //Sum[(x), x=[0,1,...n]] = (n)(n+1)/2 - triangular sequence
      //(1), (1+2), (1+2+3), (1+2+3+4), (1+2+3+4+5), ...
      //1, 3, 6, 10, 15, 21
      //sum() j*(j+1)/2
      //sum(sum()) i*(i+1)*(i+2)/6
      //sum(sum(sum())) i*(i+1)*(i+2)*(i+3)/24
      //offset = i*(i+1)*(i+2)*(i+3)/24 + j*(j+1)*(j+2)/6 + k*(k+1)/2 + k
      //offset_simple = i*21 + j*(j+1)/2 + k
      // Force unroll backwards loop
      // This halves runtime
      for (i = 5; i >= 0; i--) {
        sum = C[i];
        for (j = 5; j >= 0; j--) {
          sum1 = R[i][j];
          for (k = j; k >= 0; k--)
            sum1 += T[i][j][k] * ini[k];
          sum += sum1 * ini[j];
        }
        temp[i] = sum;
      }
      for (i = 5; i >= 0; i--)
        fin[i] = temp[i];
    }
    break;
  case 1:
    if (!C)
      bombElegant("NULL C pointer (track_particles)", NULL);
    if (!R)
      bombElegant("NULL R pointer (track_particles)", NULL);
    for (i_part = n_part - 1; i_part >= 0; i_part--) {
      if (!(fin = final[i_part])) {
        printf("error: final coordinate pointer is NULL for particle %ld (track_particles)\n", i_part);
        fflush(stdout);
        abort();
      }
      if (!(ini = initial[i_part])) {
        printf("error: final coordinate pointer is NULL for particle %ld (track_particles)\n", i_part);
        fflush(stdout);
        abort();
      }
      fin[6] = ini[6]; /* copy particle ID # */
      for (i = 5; i >= 0; i--) {
        sum = C[i];
        for (j = 5; j >= 0; j--)
          sum += R[i][j] * ini[j];
        temp[i] = sum;
      }
      for (i = 5; i >= 0; i--)
        fin[i] = temp[i];
    }
    break;
  default:
    printf("invalid order: %ld  (track_particle)\n",
           M->order);
    fflush(stdout);
    exitElegant(1);
    break;
  }

  log_exit("track_particles");
}

void free_matrices(VMATRIX *M) {
  double *C, **R;
  double ***T;
  double ****Q;

  log_entry("free_matrices");
  if (!M)
    bombElegant("NULL matrix passed to free_matrices", NULL);

  set_matrix_pointers(&C, &R, &T, &Q, M);

  if (M->maxError)
    free(M->maxError);
  if (M->meanAbsError)
    free(M->meanAbsError);
  M->maxError = M->meanAbsError = NULL;

  if (M->order > 3 || M->order < 1) {
    printf("invalid order: %ld  (free_matrices)\n", M->order);
    fflush(stdout);
    exitElegant(1);
  }

  if (M->order >= 1) {
    if (!R || !C)
      bombElegant("NULL R or C entry for matrix (free_matrices)", NULL);
    tfree(C);
    tfree(R);
  }

  if (M->order >= 2) {
    if (!T)
      bombElegant("NULL T entry for matrix (free_matrices)", NULL);
    if (!T[0])
      bombElegant("NULL T[0] entry for matrix (free_matrices)", NULL);
    tfree(T);
  }

  if (M->order >= 3) {
    if (!Q)
      bombElegant("NULL Q entry for matrix (free_matrices)", NULL);
    if (!Q[0])
      bombElegant("NULL Q[0] entry for matrix (free_matrices)", NULL);
    if (!Q[0][0])
      bombElegant("NULL Q[0][0] entry for matrix (free_matrices)", NULL);
    tfree(Q);
  }

  M->C = NULL;
  M->R = NULL;
  M->T = NULL;
  M->Q = NULL;
  log_exit("free_matrices");
}

void free_matrices_above_order(VMATRIX *M, long order) {
  double *C, **R;
  double ***T;
  double ****Q;

  set_matrix_pointers(&C, &R, &T, &Q, M);
  if (M->order < order)
    return;

  if (M->order == 3 && order < 3) {
    tfree(Q);
    M->Q = NULL;
    M->order = 2;
  }

  if (M->order == 2 && order < 2) {
    tfree(T);
    M->T = NULL;
    M->order = 1;
  }
}

void set_matrix_pointers(double **C, double ***R, double ****T, double *****Q, VMATRIX *M) {
  if (!M) {
    printf("error: NULL VMATRIX pointer\n");
    fflush(stdout);
    abort();
  }
  *C = M->C;
  *R = M->R;
  *T = M->T;
  *Q = M->Q;
}

long read_matrices(VMATRIX *M, char *filename, FILE *fp) {
  long order, i, j, k, l;
  short found;
  char s[256], *ptr;
  double *C, **R, ***T, ****Q;

  log_entry("read_matrices");

  set_matrix_pointers(&C, &R, &T, &Q, M);
  order = M->order;

  if (order >= 1) {
    found = 0;
    while (!feof(fp) && !found) {
      s[0] = 0;
      if (!fgets(s, 256, fp)) {
        log_exit("read_matrices");
        return (1);
      }
      if (strncmp(s, "C: ", 3) == 0) {
        found = 1;
      }
    }
    if (!found || strncmp(s, "C: ", 3) != 0 || !(ptr = strchr(s, ':'))) {
      printf("Error: did not find \"C:\" tag, which marks the start of a matrix, in file %s\n", filename);
      printf("Note that the matrix format changed in version 2019.1.2, which may be the source of the problem.\n");
      log_exit("read_matrices");
      return (0);
    }
    for (i = 0; i < 6; i++)
      if (!get_double(C + i, ptr)) {
        log_exit("read_matrices");
        return (0);
      }
    for (i = 0; i < 6; i++) {
      if (!fgets(s, 256, fp) || !(ptr = strchr(s, ':'))) {
        log_exit("read_matrices");
        return (0);
      }
      for (j = 0; j < 6; j++)
        if (!get_double(R[i] + j, ptr)) {
          log_exit("read_matrices");
          return (0);
        }
    }
  }

  if (order >= 2) {
    for (i = 0; i < 6; i++) {
      for (j = 0; j < 6; j++) {
        if (!fgets(s, 256, fp) || !(ptr = strchr(s, ':'))) {
          log_exit("read_matrices");
          return (1);
        }
        for (k = 0; k <= j; k++)
          if (!get_double(T[i][j] + k, ptr)) {
            log_exit("read_matrices");
            return (1);
          }
      }
    }
  }

  if (order >= 3) {
    for (i = 0; i < 6; i++) {
      for (j = 0; j < 6; j++) {
        for (k = 0; k <= j; k++) {
          if (!fgets(s, 256, fp) || !(ptr = strchr(s, ':'))) {
            log_exit("read_matrices");
            return (2);
          }
          for (l = 0; l <= k; l++)
            if (!get_double(Q[i][j][k] + l, ptr)) {
              log_exit("read_matrices");
              return (2);
            }
        }
      }
    }
  }

  log_exit("read_matrices");
  return (order);
}

void filter_matrices(VMATRIX *M, double threshold) {
  register long i, j, k, l;
  double *Tij, **Qij, *Qijk;
  double *C, **R, ***T, ****Q;

  log_entry("filter_matrices");

  set_matrix_pointers(&C, &R, &T, &Q, M);

  switch (M->order) {
  case 3:
    for (i = 0; i < 6; i++) {
      if (fabs(C[i]) < threshold)
        C[i] = 0;
      for (j = 0; j < 6; j++) {
        if (fabs(R[i][j]) < threshold)
          R[i][j] = 0;
        Tij = T[i][j];
        Qij = Q[i][j];
        for (k = 0; k <= j; k++) {
          if (fabs(Tij[k]) < threshold)
            Tij[k] = 0;
          Qijk = *Qij++;
          for (l = 0; l <= k; l++)
            if (fabs(Qijk[l]) < 0)
              Qijk[l] = 0;
        }
      }
    }
    break;
  case 2:
    for (i = 0; i < 6; i++) {
      if (fabs(C[i]) < threshold)
        C[i] = 0;
      for (j = 0; j < 6; j++) {
        if (fabs(R[i][j]) < threshold)
          R[i][j] = 0;
        Tij = T[i][j];
        for (k = 0; k <= j; k++) {
          if (fabs(Tij[k]) < threshold)
            Tij[k] = 0;
        }
      }
    }
    break;
  case 1:
    for (i = 0; i < 6; i++) {
      if (fabs(C[i]) < threshold)
        C[i] = 0;
      for (j = 0; j < 6; j++) {
        if (fabs(R[i][j]) < threshold)
          R[i][j] = 0;
      }
    }
    break;
  default:
    printf("invalid order: %ld  (filter_matrices)\n",
           M->order);
    fflush(stdout);
    exitElegant(1);
    break;
  }
  log_exit("filter_matrices");
}

void random_matrices(VMATRIX *M, double C0, double R0, double T0, double Q0) {
  register long i, j, k, l;
  double *Tij, **Qij, *Qijk;
  double *C, **R, ***T, ****Q;

  log_entry("random_matrices");

  set_matrix_pointers(&C, &R, &T, &Q, M);

  switch (M->order) {
  case 3:
    for (i = 0; i < 6; i++) {
      C[i] = C0 * (2 * random_1_elegant(1) - 1);
      for (j = 0; j < 6; j++) {
        R[i][j] = R0 * (2 * random_1_elegant(1) - 1);
        Tij = T[i][j];
        Qij = Q[i][j];
        for (k = 0; k <= j; k++) {
          *Tij++ = T0 * (2 * random_1_elegant(1) - 1);
          Qijk = *Qij++;
          for (l = 0; l <= k; l++)
            *Qijk++ = Q0 * (2 * random_1_elegant(1) - 1);
        }
      }
    }
    break;
  case 2:
    for (i = 0; i < 6; i++) {
      C[i] = C0 * (2 * random_1_elegant(1) - 1);
      for (j = 0; j < 6; j++) {
        R[i][j] = R0 * (2 * random_1_elegant(1) - 1);
        Tij = T[i][j];
        for (k = 0; k <= j; k++) {
          *Tij++ = T0 * (2 * random_1_elegant(1) - 1);
        }
      }
    }
    break;
  case 1:
    for (i = 0; i < 6; i++) {
      C[i] = C0 * (2 * random_1_elegant(1) - 1);
      for (j = 0; j < 6; j++) {
        R[i][j] = R0 * (2 * random_1_elegant(1) - 1);
      }
    }
    break;
  default:
    printf("invalid order: %ld  (null_matrices)\n",
           M->order);
    fflush(stdout);
    exitElegant(1);
    break;
  }
  log_exit("random_matrices");
}

void copy_matrices(VMATRIX *M1, VMATRIX *M0) {
  long i, j, k, l;

  log_entry("copy_matrices");

  initialize_matrices(M1, M1->order = M0->order);

  // TODO: check if memcpy better now that we have large chunks?

  for (i = 0; i < 6; i++) {
    M1->C[i] = M0->C[i];
    for (j = 0; j < 6; j++)
      M1->R[i][j] = M0->R[i][j];
  }

  if (M1->order >= 2) {
    for (i = 0; i < 6; i++)
      for (j = 0; j < 6; j++)
        for (k = 0; k <= j; k++)
          M1->T[i][j][k] = M0->T[i][j][k];
  }

  if (M1->order >= 3) {
    for (i = 0; i < 6; i++)
      for (j = 0; j < 6; j++)
        for (k = 0; k <= j; k++)
          for (l = 0; l <= k; l++)
            M1->Q[i][j][k][l] = M0->Q[i][j][k][l];
  }

  if (M1->maxError && M0->maxError)
    memcpy(M1->maxError, M0->maxError, sizeof(*(M1->maxError))*6);
  if (M1->meanAbsError && M0->meanAbsError)
    memcpy(M1->meanAbsError, M0->meanAbsError, sizeof(*(M1->meanAbsError))*6);
  
  log_exit("copy_matrices");
}

long check_matrix(VMATRIX *M, char *comment) {
  long i, j, k;

  if (M == NULL) {
    printf("error: NULL matrix pointer---%s\n", comment);
    fflush(stdout);
    abort();
  }
  if (M->order <= 0 || M->order > 3) {
    printf("error: matrix order out of range---%s\n", comment);
    fflush(stdout);
    abort();
  }
  if (M->R == NULL) {
    printf("error: NULL R matrix---%s\n", comment);
    fflush(stdout);
    abort();
  }
  for (i = 0; i < 6; i++) {
    if (M->R[i] == NULL) {
      printf("error: NULL R[%ld] row---%s\n", i, comment);
      fflush(stdout);
      abort();
    }
  }
  if (M->order == 1) {
    return (1);
  }
  if (M->T == NULL) {
    printf("error: NULL Tmatrix---%s\n", comment);
    fflush(stdout);
    abort();
  }
  for (i = 0; i < 6; i++) {
    if (M->T[i] == NULL) {
      printf("error: NULL T[%ld] row---%s\n", i, comment);
      fflush(stdout);
      abort();
    }
    for (j = 0; j < 6; j++) {
      if (M->T[i][j] == NULL) {
        printf("error: NULL T[%ld][%ld] row---%s\n", i, j, comment);
        fflush(stdout);
        abort();
      }
    }
  }
  if (M->order == 2) {
    return (2);
  }
  if (M->Q == NULL) {
    printf("error: NULL Q matrix---%s\n", comment);
    fflush(stdout);
    abort();
  }
  for (i = 0; i < 6; i++) {
    if (M->Q[i] == NULL) {
      printf("error: NULL T[%ld] row---%s\n", i, comment);
      fflush(stdout);
      abort();
    }
    for (j = 0; j < 6; j++) {
      if (M->Q[i][j] == NULL) {
        printf("error: NULL T[%ld][%ld] row---%s\n", i, j, comment);
        fflush(stdout);
        abort();
      }
      for (k = 0; k <= j; k++) {
        if (M->Q[i][j][k] == NULL) {
          printf("error: NULL Q[%ld][%ld][%ld] row---%s\n", i, j, k, comment);
          fflush(stdout);
          abort();
        }
      }
    }
  }
  return (3);
}

long reverse_matrix(VMATRIX *Mr, VMATRIX *M) {
  /* the reverse matrix of R is J*Inv(R)*J, where
   * Jij =  0  i!=j
   * Jii =  1  i=0,2,5
   * Jii = -1  i=1,3,4
   */
  long i, j;
  static MATRIX *R = NULL, *Rr = NULL, *J = NULL, *InvR = NULL, *Tmp = NULL;
  if (!R) {
    m_alloc(&R, 6, 6);
    m_alloc(&Rr, 6, 6);
    m_alloc(&J, 6, 6);
    m_alloc(&InvR, 6, 6);
    m_alloc(&Tmp, 6, 6);

    for (i = 0; i < 6; i++)
      for (j = 0; j < 6; j++) {
        if (i != j)
          J->a[i][j] = 0;
        else if (i == 0 || i == 2 || i == 5) /* yes, 5 is correct here due to sign conventions in elegant */
          J->a[i][i] = 1;
        else
          J->a[i][i] = -1;
      }
  }

  for (i = 0; i < 6; i++)
    for (j = 0; j < 6; j++)
      R->a[i][j] = M->R[i][j];

  if (!m_invert(InvR, R)) {
    printf("Error: matrix inversion failed when forming reverse matrix\n");
    return 0;
  }
  m_mult(Tmp, InvR, J);
  m_mult(Rr, J, Tmp);
  for (i = 0; i < 6; i++)
    for (j = 0; j < 6; j++)
      Mr->R[i][j] = Rr->a[i][j];

  return 1;
}

double checkSymplecticity(VMATRIX *Mv, short canonical) {
  static MATRIX *U = NULL, *Mtmp = NULL, *R = NULL, *Rt = NULL;
  double deltaMax, delta;
  long i, j;

  if (!U) {
    m_alloc(&U, 6, 6);
    m_alloc(&Mtmp, 6, 6);
    m_alloc(&R, 6, 6);
    m_alloc(&Rt, 6, 6);
    m_zero(U);
    if (!canonical) {
      /* Note that the 5,4 and 4,5 elements have a minus sign because
       * elegant uses s rather than -s as the coordinate
       */
      U->a[0][1] = U->a[2][3] = U->a[5][4] = 1;
      U->a[1][0] = U->a[3][2] = U->a[4][5] = -1;
    } else {
      U->a[0][1] = U->a[2][3] = U->a[4][5] = 1;
      U->a[1][0] = U->a[3][2] = U->a[5][4] = -1;
    }
  }

  for (i = 0; i < 6; i++)
    for (j = 0; j < 6; j++)
      R->a[i][j] = Mv->R[i][j];

  /* Need R*U*Transpose(R) */
  m_trans(Rt, R);
  m_mult(Mtmp, R, U);
  m_mult(R, Mtmp, Rt);

  deltaMax = -DBL_MAX;
  for (i = 0; i < 6; i++)
    for (j = 0; j < 6; j++) {
      if ((delta = fabs(R->a[i][j] - U->a[i][j])) > deltaMax)
        deltaMax = delta;
    }

  return deltaMax;
}

void checkSymplecticity3rdOrder(VMATRIX *M, double meanMax[3][2]) {
  double *C;
  double **R;
  double ***T;
  double ****Q;

  double ****matQ;
  double ****jacCubic;
  double ****cubicTerms1, ****cubicTerms2;

  double ***matT2;
  double ***jacQuad;
  double ***quadraticTerms1, ***quadraticTerms2;

  double **sympJmat;
  double **jacLinMat1, **jacLinMat2;

  double mean, max;

  int iz, jz, kz, lz, m, n, Ndim = 6;

  set_matrix_pointers(&C, &R, &T, &Q, M);

  sympJmat = calloc(Ndim, sizeof(double *));
  jacLinMat1 = calloc(Ndim, sizeof(double *));
  jacLinMat2 = calloc(Ndim, sizeof(double *));
  for (iz = 0; iz < Ndim; iz++) {
    sympJmat[iz] = calloc(Ndim, sizeof(double));
    jacLinMat1[iz] = calloc(Ndim, sizeof(double));
    jacLinMat2[iz] = calloc(Ndim, sizeof(double));
  }

  for (iz = 0; iz < Ndim; iz++)
    for (jz = 0; jz < Ndim; jz++) {
      if (jz == iz + 1 && iz % 2 == 0)
        sympJmat[iz][jz] = 1.0;
      if (jz == iz - 1 && (iz + 1) % 2 == 0)
        sympJmat[iz][jz] = -1.0;
    }

  /* Compute matrix JR^{tr}JR */
  for (iz = 0; iz < Ndim; iz++)
    for (jz = 0; jz < Ndim; jz++)
      for (kz = 0; kz < Ndim; kz++)
        jacLinMat1[iz][jz] += sympJmat[iz][kz] * R[kz][jz];
  for (iz = 0; iz < Ndim; iz++)
    for (jz = 0; jz < Ndim; jz++)
      for (kz = 0; kz < Ndim; kz++)
        jacLinMat2[iz][jz] += R[kz][iz] * jacLinMat1[kz][jz];
  for (iz = 0; iz < Ndim; iz++)
    for (jz = 0; jz < Ndim; jz++) {
      jacLinMat1[iz][jz] = 0.0;
      for (kz = 0; kz < Ndim; kz++)
        jacLinMat1[iz][jz] += sympJmat[iz][kz] * jacLinMat2[kz][jz];
    }
  /* add identity */
  for (iz = 0; iz < Ndim; iz++)
    jacLinMat1[iz][iz] += 1.0;

  for (iz = 0; iz < Ndim; iz++)
    for (jz = 0; jz < Ndim; jz++)
      jacLinMat2[iz][jz] = fabs(jacLinMat1[iz][jz]);

  n = 0;
  mean = 0.0;
  max = 0;
  for (iz = 0; iz < Ndim; iz++)
    for (jz = 0; jz < Ndim; jz++) {
      mean += jacLinMat2[iz][jz];
      if (max < jacLinMat2[iz][jz])
        max = jacLinMat2[iz][jz];
      n++;
    }
  mean = mean / 30.0;
  meanMax[0][0] = mean;
  meanMax[0][1] = max;

  for (iz = 0; iz < Ndim; iz++) {
    free(jacLinMat1[iz]);
    free(jacLinMat2[iz]);
  }
  free(jacLinMat1);
  free(jacLinMat2);

  //  second = {x,px,y,py,ct,delta}
  matT2 = calloc(Ndim, sizeof(double **));
  quadraticTerms1 = calloc(Ndim, sizeof(double **));
  quadraticTerms2 = calloc(Ndim, sizeof(double **));
  jacQuad = calloc(Ndim, sizeof(double **));
  for (n = 0; n < Ndim; n++) {
    matT2[n] = calloc(Ndim, sizeof(double *));
    quadraticTerms1[n] = calloc(Ndim, sizeof(double *));
    quadraticTerms2[n] = calloc(Ndim, sizeof(double *));
    jacQuad[n] = calloc(Ndim, sizeof(double *));
    for (iz = 0; iz < Ndim; iz++) {
      matT2[n][iz] = calloc(Ndim, sizeof(double));
      quadraticTerms1[n][iz] = calloc(Ndim, sizeof(double));
      quadraticTerms2[n][iz] = calloc(Ndim, sizeof(double));
      jacQuad[n][iz] = calloc(Ndim, sizeof(double));
    }
  }

  // find matrix of terms ~{x,px,y,py,ct,delta}
  for (iz = 0; iz < Ndim; iz++)
    for (jz = 0; jz < Ndim; jz++)
      for (kz = 0; kz <= jz; kz++)
        quadraticTerms1[kz][iz][jz] = T[iz][jz][kz];
  for (iz = 0; iz < Ndim; iz++)
    for (jz = 0; jz < Ndim; jz++)
      for (kz = jz; kz < Ndim; kz++)
        quadraticTerms1[kz][iz][jz] += T[iz][kz][jz];

  for (n = 0; n < Ndim; n++)
    for (iz = 0; iz < Ndim; iz++)
      for (jz = 0; jz < Ndim; jz++) {
        matT2[n][iz][jz] = quadraticTerms1[n][iz][jz];
        quadraticTerms1[n][iz][jz] = 0.0;
      }

  /* Compute matrix JR^{tr}JT */
  for (n = 0; n < Ndim; n++)
    for (iz = 0; iz < Ndim; iz++)
      for (jz = 0; jz < Ndim; jz++)
        for (kz = 0; kz < Ndim; kz++)
          quadraticTerms1[n][iz][jz] += sympJmat[iz][kz] * matT2[n][kz][jz];
  for (n = 0; n < Ndim; n++)
    for (iz = 0; iz < Ndim; iz++)
      for (jz = 0; jz < Ndim; jz++)
        for (kz = 0; kz < Ndim; kz++)
          quadraticTerms2[n][iz][jz] += R[kz][iz] * quadraticTerms1[n][kz][jz];
  for (n = 0; n < Ndim; n++)
    for (iz = 0; iz < Ndim; iz++)
      for (jz = 0; jz < Ndim; jz++)
        for (kz = 0; kz < Ndim; kz++)
          jacQuad[n][iz][jz] += sympJmat[iz][kz] * quadraticTerms2[n][kz][jz];

  /* Add matrix JT^{tr}JR */
  for (n = 0; n < Ndim; n++)
    for (iz = 0; iz < Ndim; iz++)
      for (jz = 0; jz < Ndim; jz++) {
        quadraticTerms1[n][iz][jz] = 0.0;
        for (kz = 0; kz < Ndim; kz++)
          quadraticTerms1[n][iz][jz] += sympJmat[iz][kz] * matT2[n][jz][kz];
      }
  for (n = 0; n < Ndim; n++)
    for (iz = 0; iz < Ndim; iz++)
      for (jz = 0; jz < Ndim; jz++) {
        quadraticTerms2[n][iz][jz] = 0.0;
        for (kz = 0; kz < Ndim; kz++)
          quadraticTerms2[n][iz][jz] += quadraticTerms1[n][iz][kz] * sympJmat[kz][jz];
      }
  for (n = 0; n < Ndim; n++)
    for (iz = 0; iz < Ndim; iz++)
      for (jz = 0; jz < Ndim; jz++)
        for (kz = 0; kz < Ndim; kz++)
          jacQuad[n][iz][jz] += quadraticTerms2[n][iz][kz] * R[kz][jz];

  n = 0;
  mean = 0.0;
  max = 0.0;
  for (kz = 0; kz < Ndim; kz++)
    for (iz = 0; iz < Ndim; iz++)
      for (jz = 0; jz < Ndim; jz++) {
        mean += fabs(jacQuad[kz][iz][jz]);
        if (max < fabs(jacQuad[kz][iz][jz]))
          max = fabs(jacQuad[kz][iz][jz]);
        n++;
      }
  mean = mean / 180.0;
  meanMax[1][0] = mean;
  meanMax[1][1] = max;

  for (n = 0; n < Ndim; n++) {
    for (iz = 0; iz < Ndim; iz++) {
      free(jacQuad[n][iz]);
      free(matT2[n][iz]);
    }
    free(jacQuad[n]);
    free(matT2[n]);
  }
  free(jacQuad);
  free(matT2);

  if (M->order >= 3) {
    matQ = calloc(Ndim, sizeof(double ***));
    cubicTerms1 = calloc(Ndim, sizeof(double ***));
    cubicTerms2 = calloc(Ndim, sizeof(double ***));
    jacCubic = calloc(Ndim, sizeof(double ***));
    for (n = 0; n < Ndim; n++) {
      matQ[n] = calloc(Ndim, sizeof(double **));
      cubicTerms1[n] = calloc(Ndim, sizeof(double **));
      cubicTerms2[n] = calloc(Ndim, sizeof(double **));
      jacCubic[n] = calloc(Ndim, sizeof(double **));
      for (m = 0; m < Ndim; m++) {
        matQ[n][m] = calloc(Ndim, sizeof(double *));
        cubicTerms1[n][m] = calloc(Ndim, sizeof(double *));
        cubicTerms2[n][m] = calloc(Ndim, sizeof(double *));
        jacCubic[n][m] = calloc(Ndim, sizeof(double *));
        for (iz = 0; iz < Ndim; iz++) {
          matQ[n][m][iz] = calloc(Ndim, sizeof(double));
          cubicTerms1[n][m][iz] = calloc(Ndim, sizeof(double));
          cubicTerms2[n][m][iz] = calloc(Ndim, sizeof(double));
          jacCubic[n][m][iz] = calloc(Ndim, sizeof(double));
        }
      }
    }

    // find matrix of quadratic terms ~{x^2,x*px,...}
    for (iz = 0; iz < Ndim; iz++)
      for (jz = 0; jz < Ndim; jz++)
        for (kz = 0; kz <= jz; kz++)
          for (lz = 0; lz <= kz; lz++)
            cubicTerms1[kz][lz][iz][jz] = Q[iz][jz][kz][lz];
    for (iz = 0; iz < Ndim; iz++)
      for (jz = 0; jz < Ndim; jz++)
        for (kz = jz; kz < Ndim; kz++)
          for (lz = 0; lz <= jz; lz++)
            cubicTerms1[kz][lz][iz][jz] += Q[iz][kz][jz][lz];
    for (iz = 0; iz < Ndim; iz++)
      for (jz = 0; jz < Ndim; jz++)
        for (kz = jz; kz < Ndim; kz++)
          for (lz = jz; lz <= kz; lz++)
            cubicTerms1[kz][lz][iz][jz] += Q[iz][kz][lz][jz];

    for (n = 0; n < Ndim; n++)
      for (m = 0; m < Ndim; m++)
        for (iz = 0; iz < Ndim; iz++)
          for (jz = 0; jz < Ndim; jz++) {
            matQ[n][m][iz][jz] = cubicTerms1[n][m][iz][jz];
            cubicTerms1[n][m][iz][jz] = 0.0;
          }

    /* Compute matrix JR^{tr}JU */
    for (n = 0; n < Ndim; n++)
      for (m = 0; m < Ndim; m++)
        for (iz = 0; iz < Ndim; iz++)
          for (jz = 0; jz < Ndim; jz++)
            for (kz = 0; kz < Ndim; kz++)
              cubicTerms1[n][m][iz][jz] += sympJmat[iz][kz] * matQ[n][m][kz][jz];
    for (n = 0; n < Ndim; n++)
      for (m = 0; m < Ndim; m++)
        for (iz = 0; iz < Ndim; iz++)
          for (jz = 0; jz < Ndim; jz++)
            for (kz = 0; kz < Ndim; kz++)
              cubicTerms2[n][m][iz][jz] += R[kz][iz] * cubicTerms1[n][m][kz][jz];
    for (n = 0; n < Ndim; n++)
      for (m = 0; m < Ndim; m++)
        for (iz = 0; iz < Ndim; iz++)
          for (jz = 0; jz < Ndim; jz++)
            for (kz = 0; kz < Ndim; kz++)
              jacCubic[n][m][iz][jz] += sympJmat[iz][kz] * cubicTerms2[n][m][kz][jz];

    /* Add matrix JU^{tr}JR */
    for (n = 0; n < Ndim; n++)
      for (m = 0; m < Ndim; m++)
        for (iz = 0; iz < Ndim; iz++)
          for (jz = 0; jz < Ndim; jz++) {
            cubicTerms1[n][m][iz][jz] = 0.0;
            for (kz = 0; kz < Ndim; kz++)
              cubicTerms1[n][m][iz][jz] += sympJmat[iz][kz] * matQ[n][m][jz][kz];
          }
    for (n = 0; n < Ndim; n++)
      for (m = 0; m < Ndim; m++)
        for (iz = 0; iz < Ndim; iz++)
          for (jz = 0; jz < Ndim; jz++) {
            cubicTerms2[n][m][iz][jz] = 0.0;
            for (kz = 0; kz < Ndim; kz++)
              cubicTerms2[n][m][iz][jz] += cubicTerms1[n][m][iz][kz] * sympJmat[kz][jz];
          }
    for (n = 0; n < Ndim; n++)
      for (m = 0; m < Ndim; m++)
        for (iz = 0; iz < Ndim; iz++)
          for (jz = 0; jz < Ndim; jz++)
            for (kz = 0; kz < Ndim; kz++)
              jacCubic[n][m][iz][jz] += cubicTerms2[n][m][iz][kz] * R[kz][jz];

    /* Compute JT^{tr}JT */
    for (n = 0; n < Ndim; n++)
      for (iz = 0; iz < Ndim; iz++)
        for (jz = 0; jz < Ndim; jz++)
          quadraticTerms1[n][iz][jz] = 0.0;
    for (iz = 0; iz < Ndim; iz++)
      for (jz = 0; jz < Ndim; jz++)
        for (n = 0; n <= jz; n++)
          quadraticTerms1[n][iz][jz] = T[iz][jz][n];
    for (iz = 0; iz < Ndim; iz++)
      for (jz = 0; jz < Ndim; jz++)
        for (n = jz; n < Ndim; n++)
          quadraticTerms1[n][iz][jz] += T[iz][n][jz];
    for (n = 0; n < Ndim; n++)
      for (m = 0; m < Ndim; m++)
        for (iz = 0; iz < Ndim; iz++)
          for (jz = 0; jz < Ndim; jz++)
            cubicTerms1[n][m][iz][jz] = 0.0;
    for (n = 0; n < Ndim; n++)
      for (iz = 0; iz < Ndim; iz++)
        for (jz = 0; jz < Ndim; jz++)
          for (kz = 0; kz < Ndim; kz++)
            cubicTerms1[n][0][iz][jz] += sympJmat[iz][kz] * quadraticTerms1[n][jz][kz];

    for (m = 0; m < Ndim; m++)
      for (iz = 0; iz < Ndim; iz++)
        for (jz = 0; jz < Ndim; jz++)
          quadraticTerms2[m][iz][jz] = 0.0;
    for (iz = 0; iz < Ndim; iz++)
      for (jz = 0; jz < Ndim; jz++)
        for (m = 0; m <= jz; m++)
          quadraticTerms2[m][iz][jz] = T[iz][jz][m];
    for (iz = 0; iz < Ndim; iz++)
      for (jz = 0; jz < Ndim; jz++)
        for (m = jz; m < Ndim; m++)
          quadraticTerms2[m][iz][jz] += T[iz][m][jz];
    for (n = 0; n < Ndim; n++)
      for (m = 0; m < Ndim; m++)
        for (iz = 0; iz < Ndim; iz++)
          for (jz = 0; jz < Ndim; jz++)
            cubicTerms2[n][m][iz][jz] = 0.0;
    for (m = 0; m < Ndim; m++)
      for (iz = 0; iz < Ndim; iz++)
        for (jz = 0; jz < Ndim; jz++)
          for (kz = 0; kz < Ndim; kz++)
            cubicTerms2[0][m][iz][jz] += sympJmat[iz][kz] * quadraticTerms2[m][kz][jz];

    for (n = 0; n < Ndim; n++)
      for (m = 0; m < Ndim; m++)
        for (iz = 0; iz < Ndim; iz++)
          for (jz = 0; jz < Ndim; jz++)
            matQ[n][m][iz][jz] = 0.0;
    for (n = 0; n < Ndim; n++)
      for (m = 0; m <= n; m++)
        for (iz = 0; iz < Ndim; iz++)
          for (jz = 0; jz < Ndim; jz++)
            for (kz = 0; kz < Ndim; kz++)
              matQ[n][m][iz][jz] += cubicTerms1[n][0][iz][kz] * cubicTerms2[0][m][kz][jz];
    for (n = 0; n < Ndim; n++)
      for (m = 0; m < n; m++)
        for (iz = 0; iz < Ndim; iz++)
          for (jz = 0; jz < Ndim; jz++)
            for (kz = 0; kz < Ndim; kz++)
              matQ[n][m][iz][jz] += cubicTerms1[m][0][iz][kz] * cubicTerms2[0][n][kz][jz];

    for (n = 0; n < Ndim; n++)
      for (m = 0; m < Ndim; m++)
        for (iz = 0; iz < Ndim; iz++)
          for (jz = 0; jz < Ndim; jz++)
            jacCubic[n][m][iz][jz] += matQ[n][m][iz][jz];

    mean = 0.0;
    max = 0.0;
    for (n = 0; n < Ndim; n++)
      for (m = 0; m < Ndim; m++)
        for (iz = 0; iz < Ndim; iz++)
          for (jz = 0; jz < Ndim; jz++) {
            mean += fabs(jacCubic[n][m][iz][jz]);
            if (max < fabs(jacCubic[n][m][iz][jz]))
              max = fabs(jacCubic[n][m][iz][jz]);
          }
    mean = mean / 630.0;
    meanMax[2][0] = mean;
    meanMax[2][1] = max;

    for (n = 0; n < Ndim; n++) {
      for (m = 0; m < Ndim; m++) {
        for (iz = 0; iz < Ndim; iz++) {
          free(matQ[n][m][iz]);
          free(cubicTerms1[n][m][iz]);
          free(cubicTerms2[n][m][iz]);
          free(jacCubic[n][m][iz]);
        }
        free(matQ[n][m]);
        free(cubicTerms1[n][m]);
        free(cubicTerms2[n][m]);
        free(jacCubic[n][m]);
      }
      free(matQ[n]);
      free(cubicTerms1[n]);
      free(cubicTerms2[n]);
      free(jacCubic[n]);
    }
    free(matQ);
    free(cubicTerms1);
    free(cubicTerms2);
    free(jacCubic);
  }

  for (n = 0; n < Ndim; n++) {
    for (iz = 0; iz < Ndim; iz++) {
      free(quadraticTerms1[n][iz]);
      free(quadraticTerms2[n][iz]);
    }
    free(quadraticTerms1[n]);
    free(quadraticTerms2[n]);
    free(sympJmat[n]);
  }
  free(quadraticTerms1);
  free(quadraticTerms2);
  free(sympJmat);
}

void remove_s_dependent_matrix_elements(VMATRIX *M, long order)
{
  long i, j, k, l;

  for (i=0; i<6; i++) {
    if (i==4) continue;
    M->R[i][4] = 0;

    for (j=0; j<6; j++) {
      for (k=0; k<=j; k++)
        if (k==4 || j==4)
          M->T[i][j][k] = 0;
    }
    
    for (j=0; j<6; j++) {
      for (k=0; k<=j; k++)
        for (l=0; l<=k; l++)
          if (k==4 || j==4  || l==4)
            M->Q[i][j][k][l] = 0;
    }
  }
}

      
      
