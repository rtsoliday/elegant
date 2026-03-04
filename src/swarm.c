/************************************************************************* \
* Copyright (c) 2010 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2010 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
 This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: swarm.c
 * Implementation of Particle Swarm Optimization
 * Y. Wang, 2010
 * Reference: James Kennedy and Russell Eberhart, Particle Swarm Optimization, 
 *            Proc. IEEE Int'l. Conf. on Neural Networks (Perth, Australia)
 */

#if defined(USE_GSL)
#include "gsl/gsl_vector.h"
#include "gsl/gsl_matrix.h"
#endif
#include "track.h"
#include "mdb.h"

#if !defined(USE_GSL)
long swarmMin(
  double *yReturn,
  double *xGuess,
  double *xLowerLimit,
  double *xUpperLimit,
  double *stepSize,
  long dimensions,
  double target,
  double (*func)(double *x, long *invalid),
  long populationSize,
  long n_iterations,
  long max_iterations) {
    fprintf(stderr, "Error: cannot run swarmMin because elegant was not compiled with GSL support.\n");
    exit(1);
  }
#else
long swarmMin(
  double *yReturn,
  double *xGuess,
  double *xLowerLimit,
  double *xUpperLimit,
  double *stepSize,
  long dimensions,
  double target,
  double (*func)(double *x, long *invalid),
  long populationSize,
  long n_iterations,
  long max_iterations) {
  long local_populations = populationSize / n_processors;
  long remaining_populations = populationSize - n_processors * local_populations;
  long i, j, isInvalid;
  static long best_index = -1;
  static double best_result = DBL_MAX;
  double local_rand, result, xLow, xHigh;
  //double xDiff;
  /* See reference:  Good Parameters for Particle Swarm Optimization By Magnus Erik
  double omiga = -0.4438, phi_p =-0.2699, phi_g = 3.3950; */
  double phi_p = 2.0, phi_g = 2.0, w_max = 0.9, w_min = 0.2;
  static gsl_matrix *coord_matrix = NULL, *velocity_matrix = NULL, *local_best_coord = NULL, *global_best_coord = NULL, *tmp_coord_matrix = NULL;
  static gsl_vector *local_best = NULL, *random_vector = NULL;
  static long initialized = 0;
  static double *Input = NULL;
  static char *agent_invalid = NULL;

  if (!initialized) {
    if (myid < remaining_populations)
      local_populations++;

    if (!agent_invalid)
      agent_invalid = calloc(local_populations, sizeof(*agent_invalid));

    if (!coord_matrix)
      coord_matrix = gsl_matrix_alloc(local_populations, dimensions);

    if (!tmp_coord_matrix)
      tmp_coord_matrix = gsl_matrix_alloc(local_populations, dimensions);

    if (!velocity_matrix)
      velocity_matrix = gsl_matrix_alloc(local_populations, dimensions);

    if (!local_best_coord)
      local_best_coord = gsl_matrix_alloc(local_populations, dimensions);

    if (!global_best_coord)
      global_best_coord = gsl_matrix_alloc(local_populations, dimensions);

    if (!local_best)
      local_best = gsl_vector_alloc(local_populations);

    if (!random_vector)
      random_vector = gsl_vector_alloc(local_populations);

    if (!Input)
      Input = trealloc(Input, sizeof(*Input) * dimensions);

    /* Initialize the population with uniformly distributed random numbers within the range.
       The master processor will evaluate the intitial vectors */
    for (j = 0; j < dimensions; j++) {
      xLow = xLowerLimit[j];
      xHigh = xUpperLimit[j];
      //xDiff = xHigh - xLow;
      for (i = 0; i < local_populations; i++) {
        if (isMaster && (i == 0))
          gsl_matrix_set(coord_matrix, i, j, xGuess[j]);
        else {
          double value;
          /* gsl_matrix_set (coord_matrix, i, j, xGuess[j]+xDiff*(random_2(0)-0.5)); */
          value = xGuess[j] + stepSize[i] * (2 * random_2(0) - 1);
          if (value < xLow)
            value = xLow;
          if (value > xHigh)
            value = xHigh;
          gsl_matrix_set(coord_matrix, i, j, value);
        }
        gsl_matrix_set(velocity_matrix, i, j, 0);
      }
    }
  } else {
    /* vx[][] = vx[][] + 2 * rand() * (pbestx[][] - presentx[][]) +
       2 * rand() * (pbestx[][gbest] - presentx[][]) -- see the reference */
    gsl_matrix_memcpy(tmp_coord_matrix, local_best_coord);
    gsl_matrix_sub(tmp_coord_matrix, coord_matrix);
    for (i = 0; i < local_populations; i++) {
      local_rand = random_2(0) * phi_p;
      for (j = 0; j < dimensions; j++)
        gsl_matrix_set(tmp_coord_matrix, i, j, local_rand * gsl_matrix_get(tmp_coord_matrix, i, j));
    }
    /* Creates the global_best_coord matrix by repeating the coordinates of the best particle */
    for (i = 0; i < local_populations; i++) {
      for (j = 0; j < dimensions; j++)
        gsl_matrix_set(global_best_coord, i, j, xGuess[j]);
    }
    /* Finds difference between global best and coordinates of all particles */
    gsl_matrix_sub(global_best_coord, coord_matrix);
    /* multiply this difference by a random value chosen for each particle */
    for (i = 0; i < local_populations; i++) {
      local_rand = random_2(0) * phi_g;
      for (j = 0; j < dimensions; j++)
        gsl_matrix_set(global_best_coord, i, j, local_rand * gsl_matrix_get(global_best_coord, i, j));
    }
    /* add differences to local best */
    gsl_matrix_add(tmp_coord_matrix, global_best_coord);
    /* add scaled velocity to differences */
    gsl_matrix_scale(velocity_matrix, w_max - (w_max - w_min) * n_iterations / max_iterations);
    gsl_matrix_add(velocity_matrix, tmp_coord_matrix);
    /* add all this to the particle coordinates */
    gsl_matrix_add(coord_matrix, velocity_matrix);
    /* gsl_matrix_memcpy (coord_matrix, velocity_matrix); */
  }

  /* Evaluates the populations */
  for (i = 0; i < local_populations; i++) {
    if (agent_invalid[i])
      continue;
    for (j = 0; j < dimensions; j++) {
      Input[j] = gsl_matrix_get(coord_matrix, i, j);
    }
    enforceVariableLimits(Input, xLowerLimit, xUpperLimit, dimensions);
    /* Set the changed coordinate back to the original matrix */
    for (j = 0; j < dimensions; j++)
      gsl_matrix_set(coord_matrix, i, j, Input[j]);
#if MPI_DEBUG
    for (j = 0; j < dimensions; j++)
      fprintf(stdout, "on %d, Input[%ld]=%lf, velocity = %lf, step_limit=%lf\n", myid, j, Input[j], gsl_matrix_get(velocity_matrix, 0, j), xUpperLimit[j] - xLowerLimit[j]);
#endif
    result = func(Input, &isInvalid);
    if (isInvalid) {
      agent_invalid[i] = 1;
      result = sqrt(DBL_MAX);
    }
#if MPI_DEBUG
    fprintf(stdout, "on %d, result=%lg\n", myid, result);
#endif
    if (result < best_result) { /* Updates the best result on this processor */
      best_result = result;
      best_index = i;
    }

    if ((result < gsl_vector_get(local_best, i)) || !initialized) { /* Updates the best result for this agent */
      gsl_vector_set(local_best, i, result);

      for (j = 0; j < dimensions; j++)
        gsl_matrix_set(local_best_coord, i, j, Input[j]);
    }
  }

  initialized = 1;

  if (best_index != -1) {
    for (j = 0; j < dimensions; j++)
      xGuess[j] = gsl_matrix_get(local_best_coord, best_index, j);

    *yReturn = best_result;

    /* If agent had invalid, result replace with best */
    for (i = 0; i < local_populations; i++) {
      if (agent_invalid[i]) {
        for (j = 0; j < dimensions; j++)
          gsl_matrix_set(coord_matrix, i, j, gsl_matrix_get(local_best_coord, best_index, j));
        agent_invalid[i] = 0;
      }
    }

#if MPI_DEBUG
    printf("Result=%lf\n", result);
#endif
  }

  return local_populations;

}
#endif
