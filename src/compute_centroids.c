/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: compute_centroids()
 * contents: compute_centroids(), compute_sigmas(), zero_beam_sums(),
 *           accumulate_beam_sums()
 *
 * Michael Borland, 1989
 */

/* routine: compute_centroids()
 * purpose: compute centroids for (x, x', y, y', s, deltap/p)
 *
 * Michael Borland, 1989
 */
#include "mdb.h"
#include "track.h"
#ifdef HAVE_GPU
#  include <gpu_compute_centroids.h>
#endif

BEAM_SUMS *allocateBeamSums(unsigned long flags, long nz) {
  BEAM_SUMS *ptr;
  long i;
  ptr = tmalloc(sizeof(BEAM_SUMS) * nz);
  if (!(flags & CENTROID_SUMS_ONLY)) {
    for (i = 0; i < nz; i++)
      ptr[i].beamSums2 = tmalloc(sizeof(BEAM_SUMS2));
  } else {
    for (i = 0; i < nz; i++)
      ptr[i].beamSums2 = NULL;
  }
  if (spinCoordOffset) {
    for (i=0; i<nz; i++)
      ptr[i].spinSums = tmalloc(sizeof(SPIN_SUMS));
  } else {
    for (i=0; i<nz; i++)
      ptr[i].spinSums = NULL;
  }
  /* always compute beamSums2 for the final point */
  if (flags & CENTROID_SUMS_ONLY)
    ptr[nz - 1].beamSums2 = tmalloc(sizeof(BEAM_SUMS2));
  return ptr;
}

void freeBeamSums(BEAM_SUMS *sums, long nz) {
  long i;
  for (i = 0; i < nz; i++) {
    if (sums[i].beamSums2)
      free(sums[i].beamSums2);
    if (sums[i].spinSums)
      free(sums[i].spinSums);
  }
  free(sums);
}

void compute_centroids(
  double *centroid,
  double **coordinates,
  long n_part) {
  long i_part, i_coord;
  double sum[6], *part;
  long active = 1;
#ifdef USE_KAHAN
  double error[6];
#endif

#ifdef HAVE_GPU
  if (getElementOnGpu()) {
    startGpuTimer();
    gpu_compute_centroids(centroid, n_part);
#  ifdef GPU_VERIFY
    startCpuTimer();
    copyReductionArrays(centroid, NULL);
    compute_centroids(centroid, coordinates, n_part);
    compareReductionArrays(centroid, NULL, NULL, "compute_centroids");
#  endif /* GPU_VERIFY */
    return;
  }
#endif /* HAVE_GPU */

#if USE_MPI /* In the non-parallel mode, it will be same with the serial version */
  long n_total;
#  ifdef USE_KAHAN
  long j;
  double error_sum = 0.0, error_total = 0.0,
         **sumMatrix, **errorMatrix,
         *sumArray, *errorArray;
  sumMatrix = (double **)czarray_2d(sizeof(**sumMatrix), n_processors, 6);
  errorMatrix = (double **)czarray_2d(sizeof(**errorMatrix), n_processors, 6);
  sumArray = malloc(sizeof(double) * n_processors);
  errorArray = malloc(sizeof(double) * n_processors);
#  endif
  if (notSinglePart) {
    if (((parallelStatus == trueParallel) && isSlave) || ((parallelStatus != trueParallel) && isMaster))
      active = 1;
    else
      active = 0;
  }
#endif

  for (i_coord = 0; i_coord < 6; i_coord++) {
    sum[i_coord] = centroid[i_coord] = 0;
#ifdef USE_KAHAN
    error[i_coord] = 0.0;
#endif
  }

  if (active) {
    for (i_part = 0; i_part < n_part; i_part++) {
      part = coordinates[i_part];
      for (i_coord = 0; i_coord < 6; i_coord++)
#ifndef USE_KAHAN
        sum[i_coord] += part[i_coord];
#else
        sum[i_coord] = KahanPlus(sum[i_coord], part[i_coord], &error[i_coord]);
#endif
    }
  }
  if (!USE_MPI || !notSinglePart) {
    if (n_part)
      for (i_coord = 0; i_coord < 6; i_coord++)
        centroid[i_coord] = sum[i_coord] / n_part;
  }
#if USE_MPI
  if (notSinglePart) {
    if (parallelStatus != trueParallel) {
      if (isMaster && n_part)
        for (i_coord = 0; i_coord < 6; i_coord++)
          centroid[i_coord] = sum[i_coord] / n_part;
    } else {
      if (isMaster) {
        n_part = 0;
      }
      /* compute centroid sum over processors */
#  ifndef USE_KAHAN
      MPI_Allreduce(sum, centroid, 6, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#  else
      MPI_Allgather(sum, 6, MPI_DOUBLE, &sumMatrix[0][0], 6, MPI_DOUBLE, MPI_COMM_WORLD);
      /* compute error sum over processors */
      MPI_Allgather(error, 6, MPI_DOUBLE, &errorMatrix[0][0], 6, MPI_DOUBLE, MPI_COMM_WORLD);

      for (i_coord = 0; i_coord < 6; i_coord++) {
        error_sum = 0.0;
        /* extract the columnwise array from the matrix */
        for (j = 0; j < n_processors; j++) {
          sumArray[j] = sumMatrix[j][i_coord];
          errorArray[j] = errorMatrix[j][i_coord];
        }
        centroid[i_coord] = Kahan(n_processors - 1, &sumArray[1], &error_sum);
        error_total = Kahan(n_processors - 1, &errorArray[1], &error_sum);
        centroid[i_coord] += error_total;
      }
#  endif
      /* compute total number of particles over processors */
      MPI_Allreduce(&n_part, &n_total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      if (n_total)
        for (i_coord = 0; i_coord < 6; i_coord++)
          centroid[i_coord] /= n_total;
    }
  }
#endif

#if USE_MPI /* In the non-parallel mode, it will be same with the serial version */
#  ifdef USE_KAHAN
  free_czarray_2d((void **)sumMatrix, n_processors, 6);
  free_czarray_2d((void **)errorMatrix, n_processors, 6);
  free(sumArray);
  free(errorArray);
#  endif
#endif
}

/* routine: compute_sigmas()
 * purpose: compute sigmas for (x, x', y, y', s, deltap/p)
 *
 * Michael Borland, 1989
 */

void compute_sigmas(
  double *emit,
  double *sigma,
  double *centroid,
  double **coordinates,
  long n_part) {
  long i_part, i_coord;
  double sum2[6], *part, value;
  long active = 1;
#if USE_MPI /* In the non-parallel mode, it will be same with the serial version */
  long n_total = 0;
  double sum2_total[6];

  if (notSinglePart) {
    if (isMaster && parallelStatus == trueParallel)
      n_part = 0;
    if (((parallelStatus == trueParallel) && isSlave) || ((parallelStatus != trueParallel) && isMaster))
      active = 1;
    else
      active = 0;
  }
#endif

  if (active) {
    for (i_coord = 0; i_coord < 6; i_coord++)
      sum2[i_coord] = 0;
    if (emit)
      for (i_coord = 0; i_coord < 3; i_coord++)
        emit[i_coord] = 0;

    for (i_part = 0; i_part < n_part; i_part++) {
      part = coordinates[i_part];
      for (i_coord = 0; i_coord < 6; i_coord++)
        sum2[i_coord] += sqr(part[i_coord] - centroid[i_coord]);
    }
#if USE_MPI
    if (notSinglePart) {
      /* compute total number of particles over processors */
      MPI_Allreduce(sum2, sum2_total, 6, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&n_part, &n_total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    }
    if (n_total) {
      for (i_coord = 0; i_coord < 6; i_coord++)
        if ((value = sum2_total[i_coord]) > 0)
          sigma[i_coord] = sqrt(value / n_total);
        else
          sigma[i_coord] = 0;
      if (emit) {
        for (i_coord = 0; i_coord < 6; i_coord += 2) {
          double sum12, sum12_total;
          sum12 = 0;
          for (i_part = 0; i_part < n_part; i_part++) {
            part = coordinates[i_part];
            sum12 += (part[i_coord] - centroid[i_coord]) * (part[i_coord + 1] - centroid[i_coord + 1]);
          }
          MPI_Allreduce(&sum12, &sum12_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
          if ((emit[i_coord / 2] = sqr(sigma[i_coord] * sigma[i_coord + 1]) - sqr(sum12_total / n_total)) > 0)
            emit[i_coord / 2] = sqrt(emit[i_coord / 2]);
          else
            emit[i_coord / 2] = 0;
        }
      }
#else
    if (n_part) {
      for (i_coord = 0; i_coord < 6; i_coord++)
        if ((value = sum2[i_coord]) > 0)
          sigma[i_coord] = sqrt(value / n_part);
        else
          sigma[i_coord] = 0;
      if (emit) {
        for (i_coord = 0; i_coord < 6; i_coord += 2) {
          double sum12;
          sum12 = 0;
          for (i_part = 0; i_part < n_part; i_part++) {
            part = coordinates[i_part];
            sum12 += (part[i_coord] - centroid[i_coord]) * (part[i_coord + 1] - centroid[i_coord + 1]);
          }
          if ((emit[i_coord / 2] = sqr(sigma[i_coord] * sigma[i_coord + 1]) - sqr(sum12 / n_part)) > 0)
            emit[i_coord / 2] = sqrt(emit[i_coord / 2]);
          else
            emit[i_coord / 2] = 0;
        }
      }
#endif
    }
  }
}

void zero_beam_sums(
  BEAM_SUMS *sums,
  long n) {
  long i, j, k;
  for (i = 0; i < n; i++) {
    if (sums[i].beamSums2) {
      for (j = 0; j < 7; j++)
        sums[i].beamSums2->maxabs[j] = 0;
      for (j = 0; j < 7; j++)
        sums[i].beamSums2->min[j] = -(sums[i].beamSums2->max[j] = -DBL_MAX);
      for (j = 0; j < 7; j++)
        for (k = 0; k < 7; k++)
          sums[i].beamSums2->sigma[j][k] = sums[i].beamSums2->sigman[j][k] = 0;
    }
    if (sums[i].spinSums) {
      for (j=0; j<3; j++)
	sums[i].spinSums->centroid[j] = 0;
      for (j=0; j<3; j++)
	for (k=0; k<3; k++)
	  sums[i].spinSums->sigma[j][k] = 0;
    }
    for (j = 0; j < 7; j++)
      sums[i].centroid[j] = 0;
    sums[i].n_part = sums[i].z = sums[i].p0 = 0;
    sums[i].charge = 0;
    sums[i].pass = -1;
  }
}

#if !USE_MPI
void accumulate_beam_sums(
  BEAM_SUMS *sums,
  double **coord,
  long n_part,
  double p_central,
  double mp_charge,
  double *timeValue, double tMin, double tMax, /* time filter */
  long startPID, long endPID,
  unsigned long flags) {
  long i_part, i, j;
  double centroid[7], centroidn[7], *timeCoord, *pz = NULL;
  double spinCentroid[3];
  double value, Sij, Sijn;
  long npCount = 0;
  short *chosen = NULL;
  short sparse[7][7] = {
    {1, 1, 0, 0, 0, 1, 0},
    {0, 1, 0, 0, 0, 1, 0},
    {0, 0, 1, 1, 0, 1, 0},
    {0, 0, 0, 1, 0, 1, 0},
    {0, 0, 0, 0, 1, 1, 1},
    {0, 0, 0, 0, 0, 1, 0},
    {0, 0, 0, 0, 0, 0, 1}};
#  ifdef USE_KAHAN
  double errorCen[7], errorCenn[7], errorSig, errorSign;
  double spinErrorCen[3];
#  endif

#  ifdef HAVE_GPU
  if (getElementOnGpu()) {
#    ifdef GPU_VERIFY
    BEAM_SUMS *gpusums = (BEAM_SUMS *)getGpuBeamSums(sums);
    startGpuTimer();
    gpu_accumulate_beam_sums(gpusums, n_part, p_central, mp_charge, timeValue, tMin, tMax, startPID, endPID, flags);
    startCpuTimer();
    accumulate_beam_sums(sums, coord, n_part, p_central, mp_charge, timeValue, tMin, tMax, startPID, endPID, flags);
    compareReductionArrays(NULL, NULL, sums, "accumulate_beam_sums");
#    else
    gpu_accumulate_beam_sums(sums, n_part, p_central, mp_charge, timeValue, tMin, tMax, startPID, endPID, flags);
#    endif /* GPU_VERIFY */
    return;
  }
#  endif /* HAVE_GPU */

  timeCoord = malloc(sizeof(*timeCoord) * n_part);
  if (!timeValue)
    computeTimeCoordinates(timeCoord, p_central, coord, n_part);
  else
    memcpy(timeCoord, timeValue, sizeof(*timeCoord) * n_part);

  chosen = malloc(sizeof(short) * n_part);

  if (exactNormalizedEmittance) {
    pz = malloc(sizeof(double) * n_part);
    for (i = 0; i < n_part; i++)
      pz[i] = p_central * (1 + coord[i][5]) / sqrt(1 + sqr(coord[i][1]) + sqr(coord[i][3]));
  }

  if (!sums->n_part)
    sums->p0 = p_central;

  if (n_part) {
    for (i_part = npCount = 0; i_part < n_part; i_part++) {
      if (((startPID<0 && endPID<0) || (coord[i_part][6] >= startPID && coord[i_part][6] <= endPID)) &&
          (tMin >= tMax || (timeCoord[i_part] >= tMin && timeCoord[i_part] <= tMax))) {
        chosen[i_part] = 1;
        npCount++;
      } else
        chosen[i_part] = 0;
    }
    if (!(flags & BEAM_SUMS_NOMINMAX) && sums->beamSums2) {
      /* maximum amplitudes */
      for (i = 0; i < 6; i++) {
        if (i == 4)
          continue; /* done below */
        for (i_part = 0; i_part < n_part; i_part++) {
          if (chosen[i_part]) {
            if ((value = fabs(coord[i_part][i])) > sums->beamSums2->maxabs[i])
              sums->beamSums2->maxabs[i] = value;
            if ((value = coord[i_part][i]) > sums->beamSums2->max[i])
              sums->beamSums2->max[i] = value;
            if ((value = coord[i_part][i]) < sums->beamSums2->min[i])
              sums->beamSums2->min[i] = value;
          }
        }
      }
    }

    /* compute centroids for present beam and add in to existing centroid data */
    for (i = 0; i < 7; i++) {
#  ifdef USE_KAHAN
      errorCen[i] = errorCenn[i] = 0.0;
#  endif
      for (centroid[i] = centroidn[i] = i_part = 0; i_part < n_part; i_part++) {
        if (chosen[i_part]) {
#  ifndef USE_KAHAN
          if (exactNormalizedEmittance && (i == 1 || i == 3)) {
            /* centroidn[1] and centroidn[3] will be px and py */
            centroidn[i] += coord[i_part][i] * pz[i_part];
          }
          centroid[i] += i < 6 ? coord[i_part][i] : timeCoord[i_part];
#  else
          if (exactNormalizedEmittance && (i == 1 || i == 3)) {
            /* centroidn[1] and centroidn[3] will be px and py */
            centroidn[i] = KahanPlus(centroidn[i], coord[i_part][i] * pz[i_part], &errorCenn[i]);
          }
          centroid[i] = KahanPlus(centroid[i], i < 6 ? coord[i_part][i] : timeCoord[i_part], &errorCen[i]);
#  endif
        }
      }
      if (npCount) {
        sums->centroid[i] = (sums->centroid[i] * sums->n_part + centroid[i]) / (sums->n_part + npCount);
        centroid[i] /= npCount;
        centroidn[i] /= npCount;
      }
    }
    for (i = 0; i < 7; i++) {
      if (i != 1 && i != 3)
        centroidn[i] = centroid[i];
    }

    for (i_part = 0; i_part < n_part; i_part++)
      timeCoord[i_part] -= centroid[6];

    if (spinCoordOffset) {
      /* compute centroids for present beam and add in to existing centroid data */
      for (i = 0; i < 3; i++) {
#  ifdef USE_KAHAN
        spinErrorCen[i] = 0.0;
#  endif
	spinCentroid[i] = 0.0;
        for (i_part = 0; i_part < n_part; i_part++) {
          if (chosen[i_part]) {
#  ifndef USE_KAHAN
	    spinCentroid[i] += coord[i_part][spinCoordOffset+i];
#  else
            spinCentroid[i] = KahanPlus(spinCentroid[i], coord[i_part][spinCoordOffset+i], &spinErrorCen[i]);
#  endif
	  }
        }
        if (npCount) {
          sums->spinSums->centroid[i] = (sums->spinSums->centroid[i] * sums->n_part + spinCentroid[i]) / (sums->n_part + npCount);
          spinCentroid[i] /= npCount;
        }
      }
    }

    if (!(flags & BEAM_SUMS_NOMINMAX) && sums->beamSums2) {
      i = 4;
      for (i_part = 0; i_part < n_part; i_part++) {
        if (chosen[i_part]) {
          if ((value = fabs(coord[i_part][i] - centroid[i])) > sums->beamSums2->maxabs[i])
            sums->beamSums2->maxabs[i] = value;
          if ((value = coord[i_part][i] - centroid[i]) > sums->beamSums2->max[i])
            sums->beamSums2->max[i] = value;
          if ((value = coord[i_part][i] - centroid[i]) < sums->beamSums2->min[i])
            sums->beamSums2->min[i] = value;
        }
      }
      i = 6; /* time coordinate */
      for (i_part = 0; i_part < n_part; i_part++) {
        if (chosen[i_part]) {
          if ((value = fabs(timeCoord[i_part])) > sums->beamSums2->maxabs[i])
            sums->beamSums2->maxabs[i] = value;
          if ((value = timeCoord[i_part]) > sums->beamSums2->max[i])
            sums->beamSums2->max[i] = value;
          if ((value = timeCoord[i_part]) < sums->beamSums2->min[i])
            sums->beamSums2->min[i] = value;
        }
      }
    }

    if (sums->beamSums2) {
      /* compute Sigma[i][j] for present beam and add to existing data */
      for (i = 0; i < 7; i++) {
        for (j = i; j < 7; j++) {
#  ifdef USE_KAHAN
          double Y, b, SijOld;
          double Yn, bn, SijOldn;
          errorSig = errorSign = 0.0;
#  endif
          Sij = Sijn = 0;
          if (flags & BEAM_SUMS_SPARSE && !sparse[i][j]) {
            /* Only compute the diagonal blocks and dispersive correlations */
            sums->beamSums2->sigma[j][i] = sums->beamSums2->sigma[i][j] = 0;
            sums->beamSums2->sigman[j][i] = sums->beamSums2->sigman[i][j] = 0;
            continue;
          }
          for (i_part = 0; i_part < n_part; i_part++) {
            if (chosen[i_part]) {
              b = ((i < 6 ? coord[i_part][i] - centroid[i] : timeCoord[i_part])) * ((j < 6 ? coord[i_part][j] - centroid[j] : timeCoord[i_part]));
              if (exactNormalizedEmittance) {
                bn = (coord[i_part][i] * (i == 1 || i == 3 ? pz[i_part] : 1) - centroidn[i]) * (coord[i_part][j] * (j == 1 || j == 3 ? pz[i_part] : 1));
              }
#  ifndef USE_KAHAN
              Sij += b;
              if (exactNormalizedEmittance)
                Sijn += bn;
#  else
              /* In-line KahanPlus to improve performance */
              /* Sij = KahanPlus(Sij, b, &errorSig); */
              Y = b + errorSig;
              SijOld = Sij;
              Sij += Y;
              errorSig = Y - (Sij - SijOld);
              if (exactNormalizedEmittance) {
                Yn = bn + errorSign;
                SijOldn = Sijn;
                Sijn += Yn;
                errorSign = Yn - (Sijn - SijOldn);
              }
#  endif
            }
          }
          sums->beamSums2->sigma[j][i] = (sums->beamSums2->sigma[i][j] = (sums->beamSums2->sigma[i][j] * sums->n_part + Sij) / (sums->n_part + npCount));
          if (exactNormalizedEmittance) {
            sums->beamSums2->sigman[j][i] = (sums->beamSums2->sigman[i][j] = (sums->beamSums2->sigman[i][j] * sums->n_part + Sijn) / (sums->n_part + npCount));
          }
        }
      }
    }

    if (sums->spinSums) {
      /* compute Sigma[i][j] for present beam and add to existing data */
      for (i = 0; i < 3; i++) {
        for (j = i; j < 3; j++) {
#  ifdef USE_KAHAN
          double Y, b, SijOld;
          errorSig = errorSign = 0.0;
#  endif
          Sij = 0;
          for (i_part = 0; i_part < n_part; i_part++) {
            if (chosen[i_part]) {
              b = (coord[i_part][i+spinCoordOffset] - spinCentroid[i]) * (coord[i_part][j+spinCoordOffset] - spinCentroid[j]);
#  ifndef USE_KAHAN
              Sij += b;
#  else
              /* In-line KahanPlus to improve performance */
              /* Sij = KahanPlus(Sij, b, &errorSig); */
              Y = b + errorSig;
              SijOld = Sij;
              Sij += Y;
              errorSig = Y - (Sij - SijOld);
#  endif
            }
          }
          sums->spinSums->sigma[i][j] = (sums->spinSums->sigma[i][j] * sums->n_part + Sij) / (sums->n_part + npCount);
	  sums->spinSums->sigma[j][i] = sums->spinSums->sigma[i][j];
        }
      }
    }
  }

  sums->charge = mp_charge * npCount;
  sums->n_part += npCount;
  free(timeCoord);
  free(chosen);
  if (pz)
    free(pz);
}

#else
/* USE_MPI=1 */

void accumulate_beam_sums(
  BEAM_SUMS *sums,
  double **coord,
  long n_part,
  double p_central,
  double mp_charge,
  double *timeValue, double tMin, double tMax, /* time filter */
  long startPID, long endPID,
  unsigned long flags) {
  /* The order of these calls must not be changed, since the second call uses results from the first */
  accumulate_beam_sums1(sums, coord, n_part, p_central, mp_charge, timeValue, tMin, tMax, startPID, endPID, flags);
  if (exactNormalizedEmittance)
    accumulate_beam_sums1(sums, coord, n_part, p_central, mp_charge, timeValue, tMin, tMax,
                          startPID, endPID, flags | BEAM_SUMS_EXACTEMIT | BEAM_SUMS_NOMINMAX);
}

void accumulate_beam_sums1(
  BEAM_SUMS *sums,
  double **coord,
  long n_part,
  double p_central,
  double mp_charge,
  double *timeValue, double tMin, double tMax, /* time filter */
  long startPID, long endPID,
  unsigned long flags) {
  long i_part, i, j;
  double centroid[7], *timeCoord;
  double spinCentroid[3];
  double value;
  long active = 1;
  double Sij;
  long npCount = 0, npCount_total = 0;
  short *chosen = NULL;
  short sparse[7][7] = {
    {1, 1, 0, 0, 0, 1, 0},
    {0, 1, 0, 0, 0, 1, 0},
    {0, 0, 1, 1, 0, 1, 0},
    {0, 0, 0, 1, 0, 1, 0},
    {0, 0, 0, 0, 1, 1, 1},
    {0, 0, 0, 0, 0, 1, 0},
    {0, 0, 0, 0, 0, 0, 1}};
#  ifdef USE_KAHAN
  double errorCen[7], errorSig[28];
  double errorSpinCen[3]={0,0,0};
#  endif

  if (!coord)
    n_part = 0;

#  ifdef HAVE_GPU
  if (getElementOnGpu()) {
#    ifdef GPU_VERIFY
    BEAM_SUMS *gpusums = (BEAM_SUMS *)getGpuBeamSums(sums);
    startGpuTimer();
    gpu_accumulate_beam_sums(gpusums, n_part, p_central, mp_charge, timeValue, tMin, tMax, startPID, endPID, flags);
    startCpuTimer();
    accumulate_beam_sums1(sums, coord, n_part, p_central, mp_charge, timeValue, tMin, tMax, startPID, endPID, flags);
    compareReductionArrays(NULL, NULL, sums, "accumulate_beam_sums");
#    else
    //gpu_accumulate_beam_sums(sums, n_part, p_central, mp_charge, startPID, endPID, flags);
    gpu_accumulate_beam_sums(sums, n_part, p_central, mp_charge, timeValue, tMin, tMax, startPID, endPID, flags);
#    endif /* GPU_VERIFY */
    return;
  }
#  endif   /* HAVE_GPU */

  double buffer[7], Sij_p[28], Sij_total[28], *pz = NULL;
  long offset = 0, index;
#  ifdef USE_KAHAN
  double error_sum = 0.0, error_total = 0.0,
         **sumMatrixCen, **errorMatrixCen,
         **sumMatrixSig, **errorMatrixSig,
         **sumMatrixSpinCen, **errorMatrixSpinCen,
         *sumArray, *errorArray;
  long k;
  sumMatrixCen = (double **)czarray_2d(sizeof(**sumMatrixCen), n_processors, 7);
  errorMatrixCen = (double **)czarray_2d(sizeof(**errorMatrixCen), n_processors, 7);
  sumMatrixSig = (double **)czarray_2d(sizeof(**sumMatrixSig), n_processors, 28);
  errorMatrixSig = (double **)czarray_2d(sizeof(**errorMatrixSig), n_processors, 28);
  sumMatrixSpinCen = errorMatrixSpinCen = NULL;
  if (spinCoordOffset) {
    sumMatrixSpinCen = (double **)czarray_2d(sizeof(**sumMatrixSpinCen), n_processors, 3);
    errorMatrixSpinCen = (double **)czarray_2d(sizeof(**errorMatrixSpinCen), n_processors, 3);
  }
  sumArray = malloc(sizeof(double) * n_processors);
  errorArray = malloc(sizeof(double) * n_processors);
#  endif

  if (notSinglePart) {
    if (((parallelStatus == trueParallel) && isSlave) || ((parallelStatus != trueParallel) && isMaster))
      active = 1;
    else
      active = 0;
  }

  timeCoord = malloc(sizeof(double) * n_part);
  if (!timeValue)
    computeTimeCoordinatesOnly(timeCoord, p_central, coord, n_part);
  else
    memcpy(timeCoord, timeValue, sizeof(*timeCoord) * n_part);

  chosen = malloc(sizeof(short) * n_part);

  if (flags & BEAM_SUMS_EXACTEMIT) {
    pz = malloc(sizeof(double) * n_part);
    for (i = 0; i < n_part; i++)
      pz[i] = p_central * (1 + coord[i][5]) / sqrt(1 + sqr(coord[i][1]) + sqr(coord[i][3]));
  }

  if (!sums->n_part)
    sums->p0 = p_central;
  for (i_part = npCount = 0; i_part < n_part; i_part++) {
    if (((startPID<0 && endPID<0) || (coord[i_part][6] >= startPID && coord[i_part][6] <= endPID)) &&
        (tMin >= tMax || (timeCoord[i_part] >= tMin && timeCoord[i_part] <= tMax))) {
      chosen[i_part] = 1;
      npCount++;
    } else
      chosen[i_part] = 0;
  }
  if (active) {
    if (!(flags & BEAM_SUMS_NOMINMAX) && !(flags & BEAM_SUMS_EXACTEMIT) && sums->beamSums2) {
      /* maximum amplitudes */
      for (i = 0; i < 6; i++) {
        if (i == 4)
          continue; /* done below */
        for (i_part = 0; i_part < n_part; i_part++) {
          if (chosen[i_part]) {
            if ((value = fabs(coord[i_part][i])) > sums->beamSums2->maxabs[i])
              sums->beamSums2->maxabs[i] = value;
            if ((value = coord[i_part][i]) > sums->beamSums2->max[i])
              sums->beamSums2->max[i] = value;
            if ((value = coord[i_part][i]) < sums->beamSums2->min[i])
              sums->beamSums2->min[i] = value;
          }
        }
      }
    }
    /* compute centroids for present beam and add in to existing centroid data */
    for (i = 0; i < 7; i++) {
#  ifdef USE_KAHAN
      errorCen[i] = 0.0;
#  endif
      for (centroid[i] = i_part = 0; i_part < n_part; i_part++) {
        if (chosen[i_part]) {
#  ifndef USE_KAHAN
          if (flags & BEAM_SUMS_EXACTEMIT && (i == 1 || i == 3))
            centroid[i] += coord[i_part][i] * pz[i_part];
          else
            centroid[i] += i < 6 ? coord[i_part][i] : timeCoord[i_part];
#  else
          if (flags & BEAM_SUMS_EXACTEMIT && (i == 1 || i == 3))
            centroid[i] = KahanPlus(centroid[i], coord[i_part][i] * pz[i_part], &errorCen[i]);
          else
            centroid[i] = KahanPlus(centroid[i], i < 6 ? coord[i_part][i] : timeCoord[i_part], &errorCen[i]);
#  endif
        }
      }
      if (!notSinglePart && npCount) {
        /* single-particle mode and this process has a particle */
        if (!(flags & BEAM_SUMS_EXACTEMIT))
          sums->centroid[i] = (sums->centroid[i] * sums->n_part + centroid[i]) / (sums->n_part + npCount);
        centroid[i] /= npCount;
      }
      if (notSinglePart && (parallelStatus != trueParallel) && isMaster) {
        /* multi-particle mode and, but not true parallel mode, so master has the particles */
        if (npCount) {
          if (!(flags & BEAM_SUMS_EXACTEMIT))
            sums->centroid[i] = (sums->centroid[i] * sums->n_part + centroid[i]) / (sums->n_part + npCount);
          centroid[i] /= npCount;
        }
      }
    }
    if (spinCoordOffset) {    
      /* compute spin centroids for present beam and add in to existing centroid data */
      for (i = 0; i < 3; i++) {
#  ifdef USE_KAHAN
        errorSpinCen[i] = 0.0;
#  endif
	spinCentroid[i] = 0.0;
        for (i_part = 0; i_part < n_part; i_part++) {
          if (chosen[i_part]) {
#  ifndef USE_KAHAN
            spinCentroid[i] += coord[i_part][i+spinCoordOffset];
#  else
            spinCentroid[i] = KahanPlus(spinCentroid[i], coord[i_part][i+spinCoordOffset], &errorSpinCen[i]);
#  endif
          }
        }
        if (!notSinglePart && npCount) {
          /* single-particle mode and this process has a particle */
          sums->spinSums->centroid[i] = (sums->spinSums->centroid[i] * sums->n_part + spinCentroid[i]) / (sums->n_part + npCount);
          spinCentroid[i] /= npCount;
        }
        if (notSinglePart && (parallelStatus != trueParallel) && isMaster) {
          /* multi-particle mode and, but not true parallel mode, so master has the particles */
          if (npCount) {
            sums->spinSums->centroid[i] = (sums->spinSums->centroid[i] * sums->n_part + spinCentroid[i]) / (sums->n_part + npCount);
            spinCentroid[i] /= npCount;
          }
        }
      }
    }
  } /* active */

  if (notSinglePart) {
    /* compute sums over processors */
    if (parallelStatus == trueParallel) {
      if (isMaster) {
        n_part = 0; /* All the particles have been distributed to the slave processors */
        npCount = 0;
        memset(centroid, 0.0, sizeof(double) * 7);
      }
      /* compute centroid sum over processors */
#  ifndef USE_KAHAN
      MPI_Allreduce(centroid, buffer, 7, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      memcpy(centroid, buffer, sizeof(double) * 7);
#  else
      MPI_Allgather(centroid, 7, MPI_DOUBLE, &sumMatrixCen[0][0], 7, MPI_DOUBLE, MPI_COMM_WORLD);
      /* compute error sum over processors */
      MPI_Allgather(errorCen, 7, MPI_DOUBLE, &errorMatrixCen[0][0], 7, MPI_DOUBLE, MPI_COMM_WORLD);
      for (i = 0; i < 7; i++) {
        error_sum = 0.0;
        centroid[i] = 0.0;
        /* extract the columnwise array from the matrix */
        for (j = 1; j < n_processors; j++) {
          centroid[i] = KahanPlus(centroid[i], sumMatrixCen[j][i], &error_sum);
          centroid[i] = KahanPlus(centroid[i], errorMatrixCen[j][i], &error_sum);
        }
      }
#  endif
      /* compute total number of particles over processors */
      MPI_Allreduce(&npCount, &npCount_total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      if (npCount_total)
        for (i = 0; i < 7; i++) {
          if (!(flags & BEAM_SUMS_EXACTEMIT))
            sums->centroid[i] = (sums->centroid[i] * sums->n_part + centroid[i]) / (sums->n_part + npCount_total);
          centroid[i] /= npCount_total;
        }
      if (spinCoordOffset) {
        if (parallelStatus == trueParallel) {
          if (isMaster) {
            n_part = 0; /* All the particles have been distributed to the slave processors */
            npCount = 0;
            memset(spinCentroid, 0.0, sizeof(double) * 3);
          }
          /* compute centroid sum over processors */
#  ifndef USE_KAHAN
          MPI_Allreduce(spinCentroid, buffer, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
          memcpy(spinCentroid, buffer, sizeof(double) * 3);
#  else
          MPI_Allgather(spinCentroid, 3, MPI_DOUBLE, &sumMatrixSpinCen[0][0], 3, MPI_DOUBLE, MPI_COMM_WORLD);
          /* compute error sum over processors */
          MPI_Allgather(errorSpinCen, 3, MPI_DOUBLE, &errorMatrixSpinCen[0][0], 3, MPI_DOUBLE, MPI_COMM_WORLD);
          for (i = 0; i < 3; i++) {
            error_sum = 0.0;
            spinCentroid[i] = 0.0;
            /* extract the columnwise array from the matrix */
            for (j = 1; j < n_processors; j++) {
              spinCentroid[i] = KahanPlus(spinCentroid[i], sumMatrixSpinCen[j][i], &error_sum);
              spinCentroid[i] = KahanPlus(spinCentroid[i], errorMatrixSpinCen[j][i], &error_sum);
            }
          }
#  endif
          if (npCount_total)
            for (i = 0; i < 3; i++) {
              sums->spinSums->centroid[i] = (sums->spinSums->centroid[i] * sums->n_part + spinCentroid[i]) / (sums->n_part + npCount_total);
              spinCentroid[i] /= npCount_total;
            }
	}
      }
    } else if (isMaster) {
      npCount_total = npCount;
    }
  }

  if (active) {
    for (i_part = 0; i_part < n_part; i_part++)
      timeCoord[i_part] -= centroid[6];
    if (!(flags & BEAM_SUMS_NOMINMAX) && !(flags & BEAM_SUMS_EXACTEMIT) && sums->beamSums2) {
      i = 4;
      for (i_part = 0; i_part < n_part; i_part++) {
        if (chosen[i_part]) {
          if ((value = fabs(coord[i_part][i] - centroid[i])) > sums->beamSums2->maxabs[i])
            sums->beamSums2->maxabs[i] = value;
          if ((value = coord[i_part][i] - centroid[i]) > sums->beamSums2->max[i])
            sums->beamSums2->max[i] = value;
          if ((value = coord[i_part][i] - centroid[i]) < sums->beamSums2->min[i])
            sums->beamSums2->min[i] = value;
        }
      }
      i = 6; /* time coordinate */
      for (i_part = 0; i_part < n_part; i_part++) {
        if (chosen[i_part]) {
          if ((value = fabs(timeCoord[i_part])) > sums->beamSums2->maxabs[i])
            sums->beamSums2->maxabs[i] = value;
          if ((value = timeCoord[i_part]) > sums->beamSums2->max[i])
            sums->beamSums2->max[i] = value;
          if ((value = timeCoord[i_part]) < sums->beamSums2->min[i])
            sums->beamSums2->min[i] = value;
        }
      }
    }
  }

  if (!(flags & BEAM_SUMS_NOMINMAX) && !(flags & BEAM_SUMS_EXACTEMIT) && sums->beamSums2) {
    if (notSinglePart) {
      if (parallelStatus == trueParallel) {
        /* compute sums->beamSums2->maxabs over processors*/
        MPI_Allreduce(sums->beamSums2->maxabs, buffer, 7, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        memcpy(sums->beamSums2->maxabs, buffer, sizeof(double) * 7);
        /* compute sums->beamSums2->max over processors */
        MPI_Allreduce(sums->beamSums2->max, buffer, 7, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        memcpy(sums->beamSums2->max, buffer, sizeof(double) * 7);
        /* compute sums->beamSums2->min over processors */
        MPI_Allreduce(sums->beamSums2->min, buffer, 7, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        memcpy(sums->beamSums2->min, buffer, sizeof(double) * 7);
      }
    }
  }

  if (sums->beamSums2) {
    /* compute Sigma[i][j] for present beam and add to existing data */
    if (active) {
      for (i = 0; i < 7; i++) {

        /* if ((parallelStatus==trueParallel) && notSinglePart)  */
        if (i >= 1)
          offset += i - 1;

        for (j = i; j < 7; j++) {
          double b;
#  ifdef USE_KAHAN
          double Y, SijOld;
          errorSig[j] = 0.0;
#  endif
          if (flags & BEAM_SUMS_SPARSE && !sparse[i][j]) {
            /* Only compute the diagonal blocks and dispersive correlations */
            if (flags & BEAM_SUMS_EXACTEMIT)
              sums->beamSums2->sigman[j][i] = sums->beamSums2->sigman[i][j] = 0;
            else
              sums->beamSums2->sigma[j][i] = sums->beamSums2->sigma[i][j] = 0;
            continue;
          }

          if (notSinglePart) {
            if (parallelStatus == trueParallel) {
              index = 6 * i + j - offset;
              Sij_p[index] = 0;
#  ifdef USE_KAHAN
              errorSig[index] = 0.0;
#  endif
              for (i_part = 0; i_part < n_part; i_part++) {
                if (chosen[i_part]) {
                  b = ((i < 6 ? ((flags & BEAM_SUMS_EXACTEMIT) && (i == 1 || i == 3) ? pz[i_part] : 1) * coord[i_part][i] - centroid[i] : timeCoord[i_part])) *
                      ((j < 6 ? ((flags & BEAM_SUMS_EXACTEMIT) && (j == 1 || j == 3) ? pz[i_part] : 1) * coord[i_part][j] - centroid[j] : timeCoord[i_part]));
#  ifndef USE_KAHAN
                  Sij_p[index] += b;
#  else
                  /* Sij_p[index] = KahanPlus(Sij_p[index], b, &errorSig[index]);  */
                  Y = b + errorSig[index];
                  SijOld = Sij_p[index];
                  Sij_p[index] += Y;
                  errorSig[index] = Y - (Sij_p[index] - SijOld);
#  endif
                }
              }
            } else if (isMaster) {
              for (Sij = i_part = 0; i_part < n_part; i_part++) {
                if (chosen[i_part]) {
#  ifndef USE_KAHAN
                  Sij += ((i < 6 ? ((flags & BEAM_SUMS_EXACTEMIT) && (i == 1 || i == 3) ? pz[i_part] : 1) * coord[i_part][i] : timeCoord[i_part]) - centroid[i]) *
                         ((j < 6 ? ((flags & BEAM_SUMS_EXACTEMIT) && (j == 1 || j == 3) ? pz[i_part] : 1) * coord[i_part][j] : timeCoord[i_part]) - centroid[j]);
#  else
                  Sij = KahanPlus(Sij,
                                  ((i < 6 ? ((flags & BEAM_SUMS_EXACTEMIT) && (i == 1 || i == 3) ? pz[i_part] : 1) * coord[i_part][i] - centroid[i] : timeCoord[i_part])) *
                                    ((j < 6 ? ((flags & BEAM_SUMS_EXACTEMIT) && (j == 1 || j == 3) ? pz[i_part] : 1) * coord[i_part][j] - centroid[j] : timeCoord[i_part])),
                                  &errorSig[j]);
#  endif
                }
              }
              if (npCount) {
                if (flags & BEAM_SUMS_EXACTEMIT)
                  /* sums->n_part was updated already, so we need to *subtract* npCount, not add it */
                  sums->beamSums2->sigman[j][i] = (sums->beamSums2->sigman[i][j] = (sums->beamSums2->sigman[i][j] * (sums->n_part - npCount) + Sij) / sums->n_part);
                else
                  sums->beamSums2->sigma[j][i] = (sums->beamSums2->sigma[i][j] = (sums->beamSums2->sigma[i][j] * sums->n_part + Sij) / (sums->n_part + npCount));
              }
            }
          } else { /* Single particle case */
            for (Sij = i_part = 0; i_part < n_part; i_part++) {
              if (chosen[i_part]) {
#  ifndef USE_KAHAN
                Sij += ((i < 6 ? ((flags & BEAM_SUMS_EXACTEMIT) && (i == 1 || i == 3) ? pz[i_part] : 1) * coord[i_part][i] - centroid[i] : timeCoord[i_part])) *
                       ((j < 6 ? ((flags & BEAM_SUMS_EXACTEMIT) && (j == 1 || j == 3) ? pz[i_part] : 1) * coord[i_part][j] - centroid[j] : timeCoord[i_part]));
#  else
                Sij = KahanPlus(Sij,
                                ((i < 6 ? ((flags & BEAM_SUMS_EXACTEMIT) && (i == 1 || i == 3) ? pz[i_part] : 1) * coord[i_part][i] - centroid[i] : timeCoord[i_part])) *
                                  ((j < 6 ? ((flags & BEAM_SUMS_EXACTEMIT) && (j == 1 || j == 3) ? pz[i_part] : 1) * coord[i_part][j] - centroid[j] : timeCoord[i_part])),
                                &errorSig[j]);
#  endif
              }
            }
            if (n_part) {
              if (flags & BEAM_SUMS_EXACTEMIT)
                /* sums->n_part was updated already, so we need to *subtract* npCount, not add it */
                sums->beamSums2->sigman[j][i] = (sums->beamSums2->sigman[i][j] = (sums->beamSums2->sigman[i][j] * (sums->n_part - npCount) + Sij) / sums->n_part);
              else
                sums->beamSums2->sigma[j][i] = (sums->beamSums2->sigma[i][j] = (sums->beamSums2->sigma[i][j] * sums->n_part + Sij) / (sums->n_part + npCount));
            }
          }
        }
      }
    }

    if (notSinglePart) {
      if (parallelStatus == trueParallel) {
        if (isMaster) {
          memset(Sij_p, 0, sizeof(double) * 28);
#  ifdef USE_KAHAN
          memset(errorSig, 0, sizeof(double) * 28);
#  endif
        }
        /* compute Sij sum over processors */
#  ifndef USE_KAHAN
        MPI_Allreduce(Sij_p, Sij_total, 28, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#  else
        MPI_Allgather(Sij_p, 28, MPI_DOUBLE, &sumMatrixSig[0][0], 28, MPI_DOUBLE, MPI_COMM_WORLD);
        /* compute error sum over processors */
        MPI_Allgather(errorSig, 28, MPI_DOUBLE, &errorMatrixSig[0][0], 28, MPI_DOUBLE, MPI_COMM_WORLD);
        offset = 0;
        for (i = 0; i < 7; i++) {
          if (i >= 1)
            offset += i - 1;
          for (j = i; j < 7; j++) {
            index = 6 * i + j - offset;
            if (flags & BEAM_SUMS_SPARSE && !sparse[i][j]) {
              /* Only compute the diagonal blocks and dispersive correlations */
              Sij_total[index] = 0;
              continue;
            }
            error_sum = 0.0;
            /* extract the columnwise array from the matrix */
            for (k = 0; k < n_processors; k++) {
              sumArray[k] = sumMatrixSig[k][index];
              errorArray[k] = errorMatrixSig[k][index];
            }
            Sij_total[index] = Kahan(n_processors - 1, &sumArray[1], &error_sum);
            error_total = Kahan(n_processors - 1, &errorArray[1], &error_sum);
            Sij_total[index] += error_total;
          }
        }

#  endif
        if (npCount_total) {
          offset = 0;
          for (i = 0; i < 7; i++) {
            if (i >= 1)
              offset += i - 1;
            for (j = i; j < 7; j++) {
              if (flags & BEAM_SUMS_SPARSE && !sparse[i][j]) {
                /* Only compute the diagonal blocks and dispersive correlations */
                if (flags & BEAM_SUMS_EXACTEMIT)
                  sums->beamSums2->sigman[j][i] = sums->beamSums2->sigman[i][j] = 0;
                else
                  sums->beamSums2->sigma[j][i] = sums->beamSums2->sigma[i][j] = 0;
                continue;
              }
              index = 6 * i + j - offset;
              if (flags & BEAM_SUMS_EXACTEMIT)
                /* sums->n_part was updated already, so we need to *subtract* npCount, not add it */
                sums->beamSums2->sigman[j][i] = (sums->beamSums2->sigman[i][j] = (sums->beamSums2->sigman[i][j] * (sums->n_part - npCount_total) + Sij_total[index]) / sums->n_part);
              else
                sums->beamSums2->sigma[j][i] = (sums->beamSums2->sigma[i][j] = (sums->beamSums2->sigma[i][j] * sums->n_part + Sij_total[index]) / (sums->n_part + npCount_total));
            }
          }
        }
      }
    }
  }

  if (!(flags & BEAM_SUMS_EXACTEMIT)) {

    if (!notSinglePart) {
      sums->n_part += npCount;
      sums->charge = mp_charge * npCount;
    } else if (!SDDS_MPI_IO) {
      if (parallelStatus == trueParallel) {
        sums->n_part += npCount_total;
        sums->charge = mp_charge * npCount_total;
      } else if (isMaster) {
        sums->n_part += npCount;
        sums->charge = mp_charge * npCount;
      }
    } else {
      if (isMaster) {
        sums->n_part += npCount_total;
        sums->charge = mp_charge * npCount_total;
      } else {
        sums->n_part += npCount;
        sums->charge = mp_charge * npCount;
      }
    }
  }

#  ifdef USE_KAHAN
  free_czarray_2d((void **)sumMatrixCen, n_processors, 7);
  free_czarray_2d((void **)errorMatrixCen, n_processors, 7);
  free_czarray_2d((void **)sumMatrixSig, n_processors, 28);
  free_czarray_2d((void **)errorMatrixSig, n_processors, 28);
  if (spinCoordOffset) {
    free_czarray_2d((void **)sumMatrixSpinCen, n_processors, 3);
    free_czarray_2d((void **)errorMatrixSpinCen, n_processors, 3);
  }
  free(sumArray);
  free(errorArray);
#  endif
  if (timeCoord != timeValue)
    free(timeCoord);
  free(chosen);
  if (pz)
    free(pz);
}
#endif

void copy_beam_sums(
  BEAM_SUMS *target,
  BEAM_SUMS *source) {
  long i, j;

  if (target->beamSums2 && source->beamSums2) {
    for (i = 0; i < 7; i++) {
      target->beamSums2->maxabs[i] = source->beamSums2->maxabs[i];
      target->beamSums2->max[i] = source->beamSums2->max[i];
      target->beamSums2->min[i] = source->beamSums2->min[i];
    }
  }
  for (i = 0; i < 7; i++)
    target->centroid[i] = source->centroid[i];
  if (target->beamSums2 && source->beamSums2) {
    for (i = 0; i < 7; i++)
      for (j = 0; j < 7; j++)
        target->beamSums2->sigma[i][j] = source->beamSums2->sigma[i][j];
  }
  target->n_part = source->n_part;
  target->z = source->z;
  target->p0 = source->p0;
}

long computeSliceMoments(double C[6], double S[6][6],
                         double **part, long np,
                         double sMin, double sMax) {
  long i, j, k, count = 0;
  if (!part)
    bombElegant("NULL pointer passed to computeSliceMoments", NULL);
  for (j = 0; j < 6; j++) {
    C[j] = 0;
    for (k = 0; k < 6; k++)
      S[j][k] = 0;
  }
  if (!np)
    return 0;

  for (i = 0; i < np; i++) {
    if (sMin != sMax && (part[i][4] < sMin || part[i][4] > sMax))
      continue;
    count++;
    for (j = 0; j < 6; j++) {
      C[j] += part[i][j];
    }
  }
  if (!count)
    return 0;

  for (j = 0; j < 6; j++)
    C[j] /= count;

  for (i = 0; i < np; i++) {
    if (sMin != sMax && (part[i][4] < sMin || part[i][4] > sMax))
      continue;
    for (j = 0; j < 6; j++)
      for (k = 0; k <= j; k++)
        S[j][k] += (part[i][j] - C[j]) * (part[i][k] - C[k]);
  }

  for (j = 0; j < 6; j++)
    for (k = 0; k <= j; k++) {
      S[j][k] /= count;
      S[k][j] = S[j][k];
    }
  return count;
}
