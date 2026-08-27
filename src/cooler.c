/*************************************************************************\
 * Copyright (c) 2003 The University of Chicago, as Operator of Argonne
 * National Laboratory.
 * Copyright (c) 2003 The Regents of the University of California, as
 * Operator of Los Alamos National Laboratory.
 * This file is distributed subject to a Software License Agreement found
 * in the file LICENSE that is included with this distribution.
 \*************************************************************************/

/*  file: cooler.cc
 *  based on: tfeedback.c
 *
 */
#include "mdb.h"
#include "track.h"
#if defined(USE_GSL)
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_integration.h>
#endif
#include <float.h>

#define cms c_mks
#define twopi PIx2

#define DEBUG 0

void coolerPickup(CPICKUP *cpickup, double **part0, long np0, long pass, double Po, long idSlotsPerBunch) {

  /*Initialize some variables */
  double t_sum, t_min, t_max;
  int64_t i;
  double *time0 = NULL;    /* array to record arrival time of each particle */
  long *ibParticle = NULL; /* array to record which bucket each particle is in */
  int64_t **ipBucket = NULL;  /* array to record particle indices in part0 array for all particles in each bucket */
  int64_t *npBucket = NULL;   /* array to record how many particles are in each bucket */
  long nBuckets = 0;
  long ib;
  int64_t npTotal;
  double t_sumTotal, t_minTotal, t_maxTotal;

#if USE_MPI
  if (!partOnMaster) {
    // this element does nothing in single particle mode (e.g., trajectory, orbit, ..)
    if (distributedBeam == 0)
      return;
  }
#endif

#if DEBUG
  printf("Continuing cooler pickup\n");
  fflush(stdout);
#endif

  // makes sure element is initialized
  if (cpickup->initialized == 0)
    initializeCoolerPickup(cpickup);
  else {
    for (ib = 0; ib < cpickup->nBunches; ++ib) {
      free(cpickup->long_coords[ib]);
      free(cpickup->horz_coords[ib]);
      free(cpickup->vert_coords[ib]);
      free(cpickup->pid[ib]);
      hdestroy(cpickup->pidHashTable[ib]);
      free(cpickup->index[ib]);
    }
    free(cpickup->long_coords);
    free(cpickup->horz_coords);
    free(cpickup->vert_coords);
    free(cpickup->pid);
    free(cpickup->npBunch);
    free(cpickup->pidHashTable);
    free(cpickup->index);

    cpickup->long_coords = cpickup->horz_coords = cpickup->vert_coords = NULL;
    cpickup->pid = NULL;
    cpickup->npBunch = NULL;
    cpickup->pidHashTable = NULL;
    cpickup->index = NULL;
  }

  // runs only between user specified passes
  if ((cpickup->startPass > 0 && pass < cpickup->startPass) || (cpickup->endPass > 0 && pass > cpickup->endPass))
    return;
  if (cpickup->updateInterval > 1 && pass % cpickup->updateInterval != 0)
    return;
  cpickup->lastPass = pass;

#if DEBUG
  printf("Initialized cooler pickup\n");
  fflush(stdout);
#endif

  // assign buckets normal
  if (isSlave || !distributedBeam
#if USE_MPI
      || (partOnMaster && myid == 0)
#endif
      )
    index_bunch_assignments(part0, np0, cpickup->bunchedBeamMode?idSlotsPerBunch:0, Po,
                            &time0, &ibParticle, &ipBucket, &npBucket, &nBuckets, -1);

#if DEBUG
  printf("indexed bunch assignments\n");
  fflush(stdout);
#endif

#if USE_MPI
  if (partOnMaster || myid != 0) {
#endif
#if DEBUG
    printf("%ld buckets found\n", nBuckets);
    fflush(stdout);
#endif

    if (cpickup->nBunches == 0 || cpickup->nBunches != nBuckets) {
      if (cpickup->nBunches != nBuckets) {
	printf("Number of bunches has changed, re-initializing cooler pickup\n");
	fflush(stdout);
      }
      cpickup->nBunches = nBuckets;
      cpickup->tAverage = (double*)SDDS_Realloc(cpickup->tAverage, sizeof(*cpickup->tAverage)*nBuckets);
      cpickup->tMin = (double*)SDDS_Realloc(cpickup->tMin, sizeof(*cpickup->tMin)*nBuckets);
      cpickup->tMax = (double*)SDDS_Realloc(cpickup->tMax, sizeof(*cpickup->tMax)*nBuckets);
      for (i = 0; i < nBuckets; ++i) {
	cpickup->tAverage[i] = 0;
	cpickup->tMin[i] = 0;
	cpickup->tMax[i] = 0;
      }
    }

    // Allocate space for particle coordinates, which may be needed by the CKICKER
    cpickup->long_coords = (double **)malloc(sizeof(double *) * nBuckets);
    cpickup->horz_coords = (double **)malloc(sizeof(double *) * nBuckets);
    cpickup->vert_coords = (double **)malloc(sizeof(double *) * nBuckets);
    cpickup->pid = (long **)malloc(sizeof(long *) * nBuckets);
    cpickup->npBunch = (long *)malloc(sizeof(long) * nBuckets);
    cpickup->pidHashTable = (htab **)malloc(sizeof(htab *) * nBuckets);
    cpickup->index = (long **)malloc(sizeof(long *) * nBuckets);

    // Fill the variables in CPICKUP
    for (ib = 0; ib < nBuckets; ++ib) {
      cpickup->long_coords[ib] = (double *)malloc(sizeof(double) * npBucket[ib]);
      cpickup->horz_coords[ib] = (double *)malloc(sizeof(double) * npBucket[ib]);
      cpickup->vert_coords[ib] = (double *)malloc(sizeof(double) * npBucket[ib]);
      cpickup->pid[ib] = (long *)malloc(sizeof(long) * npBucket[ib]);
      cpickup->npBunch[ib] = npBucket[ib];
      cpickup->pidHashTable[ib] = hcreate(12);
      cpickup->index[ib] = (long *)malloc(sizeof(long) * npBucket[ib]);

      // Record information about the particles in the bucket
      t_sum = 0;
      t_min = DBL_MAX;
      t_max = 0;

      // Save data for all particles
      for (i = 0; i < npBucket[ib]; ++i) {
	
	t_sum += time0[ipBucket[ib][i]];
	if (time0[ipBucket[ib][i]] < t_min)
	  t_min = time0[ipBucket[ib][i]];
	if (time0[ipBucket[ib][i]] > t_max)
	  t_max = time0[ipBucket[ib][i]];

	// Particle time used in computing kicks; x and y for transverse effects
	cpickup->long_coords[ib][i] = time0[ipBucket[ib][i]];
	cpickup->horz_coords[ib][i] = part0[ipBucket[ib][i]][0] + cpickup->dx;
	cpickup->vert_coords[ib][i] = part0[ipBucket[ib][i]][2] + cpickup->dy;

	// For double-checking ordering of particles between pickup and kicker
	char pidString[32];
	cpickup->pid[ib][i] = part0[ipBucket[ib][i]][particleIDIndex];
	snprintf(pidString, 32, "%ld", cpickup->pid[ib][i]);
	cpickup->index[ib][i] = i;
	if (!hadd(cpickup->pidHashTable[ib], (unsigned char*)pidString, strlen(pidString), &(cpickup->index[ib][i])))
	  bombElegantVA("Problem creating PID hash table: duplicate PID %ld\n", cpickup->pid[ib][i]);
      } // particle

      // Bucket particle properties, across all nodes
      npTotal = npBucket[ib];
      t_sumTotal = t_sum;
      t_minTotal = t_min;
      t_maxTotal = t_max;
#if USE_MPI
      if (!partOnMaster) {
	MPI_Allreduce(&npBucket[ib], &npTotal, 1, MPI_INT64_T, MPI_SUM, workers);
	MPI_Allreduce(&t_sum, &t_sumTotal, 1, MPI_DOUBLE, MPI_SUM, workers);
	MPI_Allreduce(&t_min, &t_minTotal, 1, MPI_DOUBLE, MPI_MIN, workers);
	MPI_Allreduce(&t_max, &t_maxTotal, 1, MPI_DOUBLE, MPI_MAX, workers);
      }
#endif
      cpickup->tAverage[ib] = t_sumTotal / npTotal;
      cpickup->tMin[ib] = t_minTotal;
      cpickup->tMax[ib] = t_maxTotal;
    } // bucket

#if USE_MPI
  } /* if (!partOnMaster || myid==0) */
#endif

  //memory managment
  if (isSlave || !distributedBeam
#if USE_MPI
      || (partOnMaster && myid==0)
#endif
      )
    free_bunch_index_memory(time0, ibParticle, ipBucket, npBucket, nBuckets);

  // Wait for sync
#if USE_MPI
#if DEBUG
  printf("Waiting on barrier before returning from coolerPickup\n");
  fflush(stdout);
#endif
  if (!partOnMaster)
    MPI_Barrier(MPI_COMM_WORLD);
#endif

#if DEBUG
  printf("Returning from coolerPickup\n");
  fflush(stdout);
#endif

}

// Pickup initialization function
void initializeCoolerPickup(CPICKUP *cpickup) {
  if (cpickup->ID == NULL || !strlen(cpickup->ID))
    bombElegant("you must give an ID string for CPICKUP", NULL);

  cpickup->nBunches = 0;

  if (cpickup->updateInterval < 1)
    cpickup->updateInterval = 1;

  cpickup->initialized = 1;
  cpickup->lastPass = -1;
  cpickup->long_coords = cpickup->horz_coords = cpickup->vert_coords = NULL;
  cpickup->pid = NULL;
  cpickup->npBunch = NULL;
}

struct E_params {
  double x;
  double y;
  double gamma;
  double lambda;
};

#if defined(USE_GSL)
double Ex(double theta, void *params) {
  //unpacking parameters
  struct E_params *p = (struct E_params *)params;
  double x = (p->x);
  double y = (p->y);
  double gamma = (p->gamma);
  double lambda = (p->lambda);

  // define some parameters for calc
  double k0 = twopi / lambda;
  double rho = sqrt(x * x + y * y);
  double phi = atan2(y, x);

  double J0_term = gsl_sf_bessel_Jn(0, rho * k0 * theta / (1 + pow(gamma * theta, 2)));
  double J2_term = gsl_sf_bessel_Jn(2, rho * k0 * theta / (1 + pow(gamma * theta, 2))) * pow(gamma * theta, 2) * cos(2 * phi);
  double I0_term = theta / pow(1 + pow(gamma * theta, 2), 4);

  return (J0_term + J2_term) * I0_term;
}
#endif

#if !defined(USE_GSL)
void coolerKicker(CKICKER *ckicker, double **part0, long np0, LINE_LIST *beamline,
  long pass, long nPasses, char *rootname, double Po, long idSlotsPerBunch) {
    fprintf(stderr, "Error: Cannot run generateBunchForMoments, elegant was not build with GSL support.\n");
   exit(1);
}
#else
void coolerKicker(CKICKER *ckicker, double **part0, long np0, LINE_LIST *beamline,
		  long pass, long nPasses, char *rootname, double Po, long idSlotsPerBunch) {
  int64_t i;
  int64_t j;
  double *time0 = NULL;    /* array to record arrival time of each particle */
  long *ibParticle = NULL; /* array to record which bucket each particle is in */
  int64_t **ipBucket = NULL;  /* array to record particle indices in part0 array for all particles in each bucket */
  int64_t *npBucket = NULL;   /* array to record how many particles are in each bucket */
  long nBuckets = 0;
  long updateInterval;
  double Ex0;
  long ib;
  int64_t npTotal;

#if USE_MPI
  if (!partOnMaster) {
    if (distributedBeam == 0)
      /* this element does nothing in single particle mode (e.g., trajectory, orbit, ..) */
      return;
  }
#endif

  if ((ckicker->startPass>0 && pass<ckicker->startPass) ||
      (ckicker->endPass>0 && pass>ckicker->endPass))
    return;

#if DEBUG
  printf("Continuing cooler kicker\n");
  fflush(stdout);
#endif

  // Determine 0th integral for transverse mode
  if (ckicker->transverseMode != 0) {
    // integration variables
    double error;
    size_t neval;
    gsl_function F;

    struct E_params params = {0, 0, Po, ckicker->lambda_rad}; // {x, y, gamma, lambda}

    F.function = &Ex;
    F.params = &params;

    // (gsl_function F, a, b, epsabs, epsrel, result, error, neval)
    gsl_integration_qng(&F, 0, ckicker->angle_rad / Po, 0, 1e-7, &Ex0, &error, &neval);
  }

  if (ckicker->initialized == 0)
    initializeCoolerKicker(ckicker, beamline, nPasses * nBuckets, rootname, Po);

#if DEBUG
  printf("Initialized cooler kicker\n");
  fflush(stdout);
#endif

  if (ckicker->startPass > 0 && ckicker->startPass != ckicker->pickup->startPass)
    bombElegantVA((char *)"CKICKER linked to CPICKUP with different START_PASS value (%ld vs %ld).",
                  ckicker->startPass, ckicker->pickup->startPass);
  if (ckicker->endPass > 0 && ckicker->endPass != ckicker->pickup->endPass)
    bombElegantVA((char *)"CKICKER linked to CPICKUP with different END_PASS value (%ld vs %ld).",
                  ckicker->endPass, ckicker->pickup->endPass);

  if ((updateInterval = ckicker->pickup->updateInterval * ckicker->updateInterval) <= 0)
    bombElegantVA((char *)"CKICKER and CPICKUP with ID=%s have UPDATE_INTERVAL product of %d", ckicker->ID, updateInterval);
  if (pass % updateInterval != 0)
    return;

  if (ckicker->pickup->lastPass != pass)
    bombElegantVA("CKICKER and CPICKUP with ID=%s are not synchronized to the same pass (%ld vs %ld)\n",
                  pass, ckicker->pickup->lastPass);

  if (isSlave || !distributedBeam
#if USE_MPI
      || (partOnMaster && myid == 0)
#endif
  )
    index_bunch_assignments(part0, np0, ckicker->bunchedBeamMode ? idSlotsPerBunch : 0, Po,
                            &time0, &ibParticle, &ipBucket, &npBucket, &nBuckets, -1);

#if DEBUG
  printf("indexed bunch assignments\n");
  fflush(stdout);
#endif

#if USE_MPI
  if (partOnMaster || myid != 0) {
#endif

    if (ckicker->nBunches == 0 || ckicker->nBunches != nBuckets) {
      if (ckicker->nBunches != nBuckets) {
	printf("Number of bunches has changed, re-initializing cooler driver.\n");
	fflush(stdout);
      }
      ckicker->nBunches = nBuckets;
    }

    if (ckicker->nBunches != ckicker->pickup->nBunches)
      bombElegantVA("mismatch in number of buckets between CKICKER (%ld) and CPICKUP (%ld)",
		    ckicker->nBunches, ckicker->pickup->nBunches);

    // Loop over buckets
    for (ib = 0; ib < nBuckets; ++ib) {
      // ================================= //
      // This is where the kick is applied //
      // ================================= //

      // Get the index of the particles in the pickup
      long *sourceIndex, *sitmp;
      char pidString[32];
      sourceIndex = (long*)malloc(sizeof(long)*npBucket[ib]);
      for (i = 0; i < npBucket[ib]; i++) {
	snprintf(pidString, 32, "%ld", (long)part0[ipBucket[ib][i]][particleIDIndex]);
	if (!hfind(ckicker->pickup->pidHashTable[ib], (unsigned char*)pidString, strlen(pidString)))
	  bombElegantVA("PID of particle in CKICKER not present at CPICKUP! Cooling ID=%s\n",
			ckicker->ID);
	if (!(sitmp = (long*)hstuff(ckicker->pickup->pidHashTable[ib])))
	  bombElegantVA("Problem retrieving index of PID %ld. Cooling ID=%s\n",
			(long)part0[ipBucket[ib][i]][particleIDIndex], ckicker->ID);
	sourceIndex[i] = *sitmp;
      }

      // Total number of particles across all nodes
      npTotal = npBucket[ib];
#if USE_MPI
      if (!partOnMaster)
	MPI_Allreduce(&npBucket[ib], &npTotal, 1, MPI_INT64_T, MPI_SUM, workers);
#endif

      // Get the dt between CPICKUP and CKICKER
      double dt_ref = 0;
      if (ckicker->dtClosedOrbit != 0)
	dt_ref = ckicker->delta_t;
      else {
	double t_sum = 0;
	for (i = 0; i < npBucket[ib]; ++i)
	  t_sum += time0[ipBucket[ib][i]];
	double t_sumTotal = t_sum;
#if USE_MPI
	if (!partOnMaster)
	  MPI_Allreduce(&t_sum, &t_sumTotal, 1, MPI_DOUBLE, MPI_SUM, workers);
#endif
	double t_avg = t_sumTotal / npTotal;
	dt_ref = t_avg - ckicker->pickup->tAverage[ib];
      }

#if DEBUG
      printf("Bunch %ld has %ld particles, dt_ref = %le\n", ib, npBucket[ib], dt_ref);
      fflush(stdout);
#endif

      // For incoherent mode, need the times from all bucket particles from across all nodes
      // and the translation from particle ID on this node to the bucket
      double* all_pickup_times = NULL;
      long ibOffset = 0;
      if (ckicker->incoherentMode != 0) {

        // Get list of all particles on this node
        double* node_pickup_times = (double*)malloc(npBucket[ib]*sizeof(double));
        for (i = 0; i < npBucket[ib]; ++i)
          node_pickup_times[i] = ckicker->pickup->long_coords[ib][sourceIndex[i]];

        // Get list of all particles across all nodes
#if USE_MPI
        if (!partOnMaster) {
	  // Get number of worker and rank
	  int num_workers = -1, worker_rank = -1;
	  MPI_Comm_size(workers, &num_workers);
	  MPI_Comm_rank(workers, &worker_rank);

          // Number of bunch particles on all nodes
          int64_t* node_bunch_particles = (int64_t*)malloc(num_workers*sizeof(int64_t));
          MPI_Allgather(&npBucket[ib], 1, MPI_INT64_T, node_bunch_particles, 1, MPI_INT64_T, workers);

	  // Offsets for where the particles on each node sit wrt bucket
          long* node_array_offsets = (long*)malloc(num_workers*sizeof(long));
          for (int nodeIt = 0; nodeIt < num_workers; ++nodeIt) { // node refers to workers
            node_array_offsets[nodeIt] = 0;
            for (int otherNodesIt = nodeIt-1; otherNodesIt >= 0; --otherNodesIt)
              node_array_offsets[nodeIt] += node_bunch_particles[otherNodesIt];
          }
          ibOffset = node_array_offsets[worker_rank];

	  // The way the MPI gather function works requires the number of elements being
	  // passed around be described as an int.  This imposes an effective limit to how
	  // many particles can be simulated in parallel (2^31).  It is expected that most
	  // situations would be below this, so the code is left as-is right now, with a
	  // warning.  In the future if more particles are required to be run in this
	  // routine, this can be revisited.
	  if (npTotal > pow(2,31))
	    printWarning("CKICKER cannot handle >2^31 particles in the simulation.",
			 "The number of particles simulated (>2^31) may be too many to be correctly "
			 "handled by the MPI routines in the CKICKER incoherent interactions.");

	  // Convert longs to ints for the purposes of the MPI method
	  int* node_bunch_particles_i = (int*)malloc(num_workers*sizeof(int));
	  int* node_array_offsets_i = (int*)malloc(num_workers*sizeof(int));
	  for (int nodeIt = 0; nodeIt < num_workers; ++nodeIt) {
	    node_bunch_particles_i[nodeIt] = (int)node_bunch_particles[nodeIt];
	    node_array_offsets_i[nodeIt] = (int)node_array_offsets[nodeIt];
	  }

          // Full list of particle times
          all_pickup_times = (double*)malloc(npTotal*sizeof(double));
          MPI_Allgatherv(node_pickup_times, node_bunch_particles_i[worker_rank], MPI_DOUBLE,
                         all_pickup_times, node_bunch_particles_i, node_array_offsets_i, MPI_DOUBLE,
			 workers);

          free(node_pickup_times); node_pickup_times = NULL;
          free(node_bunch_particles); node_bunch_particles = NULL;
          free(node_array_offsets); node_array_offsets = NULL;
          free(node_bunch_particles_i); node_bunch_particles_i = NULL;
          free(node_array_offsets_i); node_array_offsets_i = NULL;
        } else {
          all_pickup_times = node_pickup_times; node_pickup_times = NULL;
        }
#else
        all_pickup_times = node_pickup_times; node_pickup_times = NULL;
#endif
      } // incoherent

      // In 'numerical mode', the wave packet at the pickup is determined a priori for
      // the full beam bunch, to create a look-up function to be used when applying kicks.
      // Need to store the size of the wave and combine across all nodes.
      double* pickup_rad = NULL;
      double pickup_rad_start = 0., rad_bin_length = 0.;
      if (ckicker->numericalMode != 0) {

	// Container to hold kick function across full pulse wavepacket
	double rad_period_length = ckicker->lambda_rad / c_mks;
	double rad_particle_length = 2 * ckicker->Nu * rad_period_length;
	int rad_period_bins = 20;
	rad_bin_length = rad_period_length / rad_period_bins;
	int rad_particle_nbins = rad_particle_length / rad_bin_length;

	double pickup_pulse_length = ckicker->pickup->tMax[ib] - ckicker->pickup->tMin[ib];
	int pickup_rad_nbins
	  = (pickup_pulse_length + rad_particle_length) / rad_bin_length;
	pickup_rad_start = ckicker->pickup->tMin[ib] - rad_particle_length/2.;
#if DEBUG
	printf("Radiation wave packet starts at %le and has %d bins (of length %le): total %lf MB\n",
	       pickup_rad_start, pickup_rad_nbins, rad_bin_length, pickup_rad_nbins*8e-6);
	fflush(stdout);
#endif

	// Define the full wave packet at the pickup for this node
	double* node_pickup_rad = (double*)malloc(pickup_rad_nbins*sizeof(double));
	for (int pickup_rad_bin = 0; pickup_rad_bin < pickup_rad_nbins; ++pickup_rad_bin)
	  node_pickup_rad[pickup_rad_bin] = 0;
	double time_pu, time_pu_start;
	int bin_start;
	double phi, envelope;
	for (i = 0; i < npBucket[ib]; ++i) {
	  // Time at pickup
	  time_pu = ckicker->pickup->long_coords[ib][sourceIndex[i]];
	  time_pu_start = time_pu - rad_particle_length/2.;
	  bin_start = (time_pu_start - pickup_rad_start) / rad_bin_length;

	  // Radiation at pickup
	  for (int rad_bin = 0; rad_bin < rad_particle_nbins; ++rad_bin) {
	    phi = (2. * ckicker->Nu * rad_bin/rad_particle_nbins) + ckicker->phase;
	    envelope = (rad_particle_nbins/2. - fabs(rad_particle_nbins/2.-rad_bin)) / (rad_particle_nbins/2.);
	    node_pickup_rad[bin_start+rad_bin] += ckicker->strength * envelope * sin(twopi*phi);
	  } // bin
	} // particle

	// Get the full wave packet at the pickup across all nodes
#if USE_MPI
        if (!partOnMaster) {
	  pickup_rad = (double*)malloc(pickup_rad_nbins*sizeof(double));
	  MPI_Allreduce(node_pickup_rad, pickup_rad, pickup_rad_nbins, MPI_DOUBLE, MPI_SUM, workers);
	  free(node_pickup_rad); node_pickup_rad = NULL;
        } else {
          pickup_rad = node_pickup_rad; node_pickup_rad = NULL;
        }
#else
        pickup_rad = node_pickup_rad; node_pickup_rad = NULL;
#endif
      } // numerical wave packet

      // Apply the kicks to each particle in turn
      double deltax_i, deltay_i, dt_i, dt_ij_pu;
      double time_i_pu, time_i_ku, time_j_pu;
      double total_phi, kick, wave_packet_kick;
      double coherent_phi, envelope_strength, coherent_kick;
      double incoherent_phi, incoherent_strength, incoherent_kick;
      double modulation_phase, modulation_strength;
      int phase_bin;

      // Numerical mode
      if (ckicker->numericalMode != 0) {
	for (i = 0; i < npBucket[ib]; ++i) {
	  time_i_ku = time0[ipBucket[ib][i]];
	  phase_bin = (time_i_ku - (pickup_rad_start + dt_ref)) / rad_bin_length;
	  wave_packet_kick = pickup_rad[phase_bin];
#if DEBUG
	  printf("Giving particle %ld a kick %le from the numerical wave packet\n", i, wave_packet_kick);
	  fflush(stdout);
#endif
	  part0[ipBucket[ib][i]][5] += wave_packet_kick;
	}

      // Analytical mode
      } else {
	for (i = 0; i < npBucket[ib]; ++i) {

	  kick = 0.; coherent_kick = 0.; incoherent_kick = 0.;

	  // Get the times the particle went through the pickup and kicker
	  time_i_pu = ckicker->pickup->long_coords[ib][sourceIndex[i]];
	  time_i_ku = time0[ipBucket[ib][i]];
	  dt_i = time_i_ku - time_i_pu;

	  // Coherent kick
	  coherent_phi = (dt_i - dt_ref) * c_mks / ckicker->lambda_rad;
	  total_phi = coherent_phi + ckicker->phase;
	  /* double envelope_strength = (2.0/twopi)*(atan(100*(total_phi+ckicker->Nu)) */
	  /*                                         + atan(-100*(total_phi-ckicker->Nu))) * (1 - abs(total_phi)/ckicker->Nu); */
	  envelope_strength = fabs(total_phi) <= ckicker->Nu ? 1 - fabs(total_phi)/ckicker->Nu : 0;
	  modulation_phase = ckicker->modulation_freq * pass * twopi;
	  modulation_strength = pow(cos(modulation_phase), 2);
	  coherent_kick = ckicker->strength * envelope_strength * modulation_strength * sin(total_phi*twopi);
#if DEBUG
	  printf("Giving particle %ld (global id %ld) a coherent kick %le\n", i, i+ibOffset, coherent_kick);
	  fflush(stdout);
#endif

	  // Transverse component
	  if (ckicker->transverseMode != 0) {
	    double Exi, error;
	    size_t neval;
	    gsl_function F;
	    deltax_i = part0[ipBucket[ib][i]][0] -
	      ckicker->magnification * ckicker->pickup->horz_coords[ib][sourceIndex[i]];
	    deltay_i = part0[ipBucket[ib][i]][2] -
	      ckicker->magnification * ckicker->pickup->vert_coords[ib][sourceIndex[i]];
	    struct E_params params = {deltax_i, deltay_i, Po, ckicker->lambda_rad}; // {x, y, gamma, lambda}
	    F.function = &Ex;
	    F.params = &params;

	    gsl_integration_qng(&F, 0, ckicker->angle_rad/Po, 0, 1e-7, &Exi, &error, &neval);

	    kick += coherent_kick * Exi/Ex0;
	  } else
	    kick += coherent_kick;

	  // Incoherent kick
	  if (ckicker->incoherentMode != 0) {

	    // Convert to 'global' bucket index
	    // NB this relies on MPI consistently ordering particles by node,
	    // which I believe it does!
	    long i_gbucket = i + ibOffset;

	    // Consider the kicks from each of the other particles
	    for (j = 0; j < npTotal; ++j) {

	      // Time of this other particle at the pickup
	      time_j_pu = all_pickup_times[j];
	      dt_ij_pu = time_i_pu - time_j_pu;

	      // Only consider other particles, and only those which are within the radiation wavelength envelope
	      if ((i_gbucket == j) || fabs(dt_ij_pu) >= (ckicker->Nu*ckicker->lambda_rad)/c_mks)
		continue;

	      incoherent_phi = dt_ij_pu * c_mks / ckicker->lambda_rad;
	      total_phi = coherent_phi + incoherent_phi + ckicker->phase;
	      /* double incoherent_strength = (2.0/twopi)*(atan(100*(total_phi+ckicker->Nu)) + */
	      /*                                           atan(-100*(total_phi-ckicker->Nu))) * (1 - abs(total_phi)/(ckicker->Nu)); */
	      incoherent_strength = fabs(total_phi) <= ckicker->Nu ? 1 - fabs(total_phi)/ckicker->Nu : 0;
	      incoherent_kick = ckicker->strength * incoherent_strength * sin(total_phi*twopi);

#if DEBUG
	      printf("Giving particle %ld (global id %ld) a kick %le from particle %ld\n",
		     i, i_gbucket, incoherent_kick, j);
	      fflush(stdout);
#endif
	      kick += incoherent_kick;
	    }
	  } // incoherent

	  part0[ipBucket[ib][i]][5] += kick;
	} // particle
      } // kick mode

      // Clean up memory associated with the bucket
      if (ckicker->incoherentMode != 0) {
	free(all_pickup_times); all_pickup_times = NULL;
      }
      if (ckicker->numericalMode != 0) {
	free(pickup_rad); pickup_rad = NULL;
      }
      free(sourceIndex); sourceIndex = NULL;
    } // bucket

    // Clean up bucket
    if (isSlave || !distributedBeam)
      free_bunch_index_memory(time0, ibParticle, ipBucket, npBucket, nBuckets);

    // Sync all nodes
#if USE_MPI
  } /* if (partOnMaster || myid!=0) */
  if (!partOnMaster)
    MPI_Barrier(MPI_COMM_WORLD);
#endif
}
#endif

// Kicker initialization function
void initializeCoolerKicker(CKICKER *ckicker, LINE_LIST *beamline, long nPasses, char *rootname, double Po) {
  if (ckicker->ID == NULL || !strlen(ckicker->ID))
    bombElegant("CKICKER requires an ID string", NULL);

  // Find the CPICKUP associated with this CKICKER, and the
  // reference particle location at each element
  ELEMENT_LIST *eptr = beamline->elem;
  long pickupFound = 0;
  double pickup_coord[6], kicker_coord[6];
  for (long beamlineElemIt = 0; beamlineElemIt < beamline->n_elems; ++beamlineElemIt) {
    if (eptr->type == T_CPICKUP && strcmp(ckicker->ID, ((CPICKUP*)eptr->p_elem)->ID) == 0) {
      pickupFound = 1;
      ckicker->pickup = ((CPICKUP*)eptr->p_elem);
      if (ckicker->dtClosedOrbit != 0)
	memcpy(pickup_coord,
	       beamline->closed_orbit[beamlineElemIt - beamline->i_recirc].centroid,
	       sizeof(double)*6);
    }
    if (eptr->type == T_CKICKER && strcmp(ckicker->ID, ((CKICKER*)eptr->p_elem)->ID) == 0 &&
	ckicker->dtClosedOrbit != 0)
      memcpy(kicker_coord,
	     beamline->closed_orbit[beamlineElemIt - beamline->i_recirc].centroid,
	     sizeof(double)*6);
    eptr = eptr->succ;
  }
  if (!pickupFound)
    bombElegant("CPICKUP not found for CKICKER", NULL);

  // Save the reference delta time between pickup and kicker
  if (ckicker->dtClosedOrbit != 0) {
    double beta = Po / (sqrt(pow(Po, 2) + 1));
    ckicker->delta_t = (kicker_coord[4] - pickup_coord[4]) / (beta * c_mks);
  } else
    printWarning("CPICKUP -> CKICKER dt",
		 "The default behavior is to take the difference between the average bunch times; "
		 "consider using the reference dt from the closed orbit calculation for better accuracy "
		 "(ensure closed_orbit is in the elegant configuration and use CKICKER::dtClosedOrbit != 0).");

  // Initialize the variables
  ckicker->nBunches = 0;
  if (ckicker->updateInterval < 1)
    ckicker->updateInterval = 1;
  ckicker->initialized = 1;
}
