/************************************************************************* \
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

#if USE_MPI
#  include "mpi.h" /* Defines the interface to MPI allowing the use of all MPI functions. */
#  if USE_MPE
#    include "mpe.h" /* Defines the MPE library */
#  endif
#endif
#include <complex>
#include "mdb.h"
#include "track.h"
#include "match_string.h"
#ifdef HAVE_GPU
#  include "gpu_space_charge.h"
#endif

typedef struct {
  double dmux, dmuy;
  double center[3];
  double sigma[3];
} BUNCH_DATA;  

/* There's no reason to use a structure here since these are all local variables.
 * Note that they can't be part of the SCMULT elements since the information is needed
 * across such elements.
 */
typedef struct {
  long verbosity;
  long horizontal, vertical, longitudinal, uniform;
  long nonlinear, sliceThreshold, sliceInterpolation;
  double sliceDuration;
  double averagingFactor;
  double c0;   			/* c0=re*np/(2*Pi)^(3/2) -> calculate once.  */
  double c1;	 		/* c1=c0/p0^3 */
  double length;
  long nBunches;
  BUNCH_DATA *bunchData;
} SPACE_CHARGE;

static short firstCall = 1;
static SPACE_CHARGE sc;

void linearSCKick(double *coord, ELEMENT_LIST *eptr, double *centroid, double *sigma, double charge, long iBunch);
int nonlinearSCKick(double *coord, ELEMENT_LIST *eptr, double *centroid, double *sigma, double *kick, double charge, long iBunch);

static void applySCKick(double *coord, ELEMENT_LIST *eptr,
                        double *centroid, double *sigma,
                        double charge, long iBunch) {
  if (!sc.nonlinear) {
    linearSCKick(coord, eptr, centroid, sigma, charge, iBunch);
  } else {
    double kick[2];
    if (!nonlinearSCKick(coord, eptr, centroid, sigma, kick, charge, iBunch)) {
      linearSCKick(coord, eptr, centroid, sigma, charge, iBunch);
      return;
    }
    coord[1] += kick[0];
    coord[3] += kick[1];
  }
}

void setupSCEffect(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline) {
#include "insertSCeffects.h"
  long i;

  /* process the namelist text */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&insert_sceffects, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &insert_sceffects);

  if (clear) {
    clearSCSpecs();
    if (!name && !type)
      return;
  }
  if (disable)
    return;

  if (!name || !strlen(name))
    bombElegant((char *)"no name given", NULL);
  str_toupper(name);
  if (has_wildcards(name) && strchr(name, '-'))
    name = expand_ranges(name);
  if (type) {
    str_toupper(type);
    if (has_wildcards(type) && strchr(type, '-'))
      type = expand_ranges(type);
    for (i = 0; i < N_TYPES; i++)
      if (wild_match(entity_name[i], type))
        break;
    if (i == N_TYPES) {
      fprintf(stderr, (char *)"type pattern %s does not match any known type", type);
      exitElegant(1);
    }
  }
  if (exclude) {
    str_toupper(exclude);
    if (has_wildcards(exclude) && strchr(exclude, '-'))
      exclude = expand_ranges(exclude);
  }

  if (getSCMULTSpecCount())
    printWarning((char*)"Multiple insert_sceffects commands given.",
		 (char*)"Calculation settings will be taken from the last command.");
  
  addSCSpec(name, type, exclude, skip, verbosity);
  cp_str(&(scMultName), element_prefix);

  if (!firstCall) {
    if (sc.bunchData)
      free(sc.bunchData);
    sc.bunchData = NULL;
  }
  sc.vertical = vertical;
  sc.horizontal = horizontal;
  sc.longitudinal = longitudinal;
  sc.nonlinear = nonlinear;
  sc.uniform = uniform_distribution;
  sc.sliceDuration = slice_duration;
  sc.sliceThreshold = slice_threshold;
  sc.sliceInterpolation = slice_interpolation;
  sc.bunchData = NULL;
  sc.nBunches = 0;
  firstCall = 0;
  
  if ((sc.averagingFactor = averaging_factor) <= 0 || averaging_factor > 1)
    bombElegant("averaging_factor must be on (0, 1]", NULL);
  if (averaging_factor < 1 && !nonlinear)
    bombElegant("averaging_factor is ignore for linear mode", NULL);

  if (longitudinal)
    printWarning((char *)"The longitudinal space-charge effect is not implemented by SCMULT.",
                 (char *)"Consider using LSCDRIFT elements.");

}

/* track through space charge element */
void trackThroughSCMULT(double **part0, int64_t np0, double Po, long iPass, ELEMENT_LIST *eptr, CHARGE *charge) {
  int64_t i;
  long *pbin = NULL;       /* array to record which bin each particle is in */
  double *time0 = NULL;    /* array to record arrival time of each particle */
  double *time = NULL;     /* array to record arrival time of each particle, for working bucket */
  double **part = NULL;    /* particle buffer for working bucket */
  long *ibParticle = NULL; /* array to record which bucket each particle is in */
  int64_t **ipBucket = NULL;  /* array to record particle indices in part0 array for all particles in each bucket */
  int64_t *npBucket = NULL;   /* array to record how many particles are in each bucket */
  /* long ib, nb = 0, n_binned = 0; */
  long iBucket, nBuckets;
  int64_t max_np = 0, ip, np;
  double tmin, tmax;
  double totalCharge;
#ifdef HAVE_GPU
  long scmultOnGpu;
#endif

#ifdef DEBUG
  printf("entered trackThroughSCMULT\n");
  fflush(stdout);
#endif

#ifdef HAVE_GPU
  if ((isSlave || !distributedBeam) && charge && eptr && eptr->twiss &&
      sc.bunchData && sc.nBunches == 1 &&
      gpu_scmult_single_bunch_supported(np0, charge->idSlotsPerBunch,
                                        sc.nonlinear, sc.sliceDuration,
                                        sc.horizontal, sc.vertical)) {
#  ifdef GPU_VERIFY
    if (getElementOnGpu())
      part0 = forceParticlesToCpu("trackThroughSCMULT verification");
#  endif
    if (gpu_scmult_compute_centroid_sigma(np0, Po,
                                          sc.bunchData[0].center,
                                          sc.bunchData[0].sigma)) {
      totalCharge = np0 * charge->macroParticleCharge;
      if (!(iPass == 0 || sc.averagingFactor == 1)) {
        for (int j = 0; j < 3; j++)
          sc.bunchData[0].sigma[j] =
            (1 - sc.averagingFactor) * ((SCMULT *)eptr->p_elem)->lastSigma[j] +
            sc.averagingFactor * sc.bunchData[0].sigma[j];
      }
      for (int j = 0; j < 3; j++)
        ((SCMULT *)eptr->p_elem)->lastSigma[j] = sc.bunchData[0].sigma[j];
      if (sc.nonlinear)
        gpu_track_through_scmult_nonlinear(np0, totalCharge, sc.c1,
                                           sc.horizontal, sc.vertical,
                                           sc.uniform,
                                           sc.bunchData[0].center,
                                           sc.bunchData[0].sigma,
                                           sc.bunchData[0].dmux,
                                           sc.bunchData[0].dmuy,
                                           eptr->twiss->betax,
                                           eptr->twiss->betay);
      else
        gpu_track_through_scmult_linear(np0, totalCharge, sc.c1,
                                        sc.horizontal, sc.vertical,
                                        sc.uniform,
                                        sc.bunchData[0].center,
                                        sc.bunchData[0].sigma,
                                        sc.bunchData[0].dmux,
                                        sc.bunchData[0].dmuy,
                                        eptr->twiss->betax,
                                        eptr->twiss->betay);
#  ifdef GPU_VERIFY
#    if defined(_OPENMP)
#      pragma omp parallel for num_threads(gpuGetOmpTrackingThreads()) schedule(static) if(gpuOmpTrackingRequested(np0))
#    endif
      for (i = 0; i < np0; i++) {
        applySCKick(part0[i], eptr, sc.bunchData[0].center,
                    sc.bunchData[0].sigma, totalCharge, 0);
      }
      compareGpuCpu(np0, sc.nonlinear ?
                    "trackThroughSCMULT nonlinear resident" :
                    "trackThroughSCMULT linear resident");
#  endif
      sc.bunchData[0].dmux = sc.bunchData[0].dmuy = 0.0;
      return;
    }
  }
  if (getElementOnGpu())
    part0 = forceParticlesToCpu("trackThroughSCMULT fallback");
#endif

  if (isSlave || !distributedBeam) {
    index_bunch_assignments(part0, np0, charge->idSlotsPerBunch, Po, &time0, &ibParticle, &ipBucket, &npBucket, &nBuckets, -1);

#ifdef DEBUG
    if (nBuckets > 1) {
      printf("%ld buckets\n", nBuckets);
      fflush(stdout);
      for (iBucket = 0; iBucket < nBuckets; iBucket++) {
        printf("bucket %ld: %ld particles\n", iBucket, npBucket[iBucket]);
        fflush(stdout);
      }
    }
#endif

    if (sc.nBunches!=0 && sc.nBunches!=nBuckets)
      bombElegantVA((char*)"Change in number of bunches from %ld to %ld seen in SCMULT. Not supported.\n",
		    sc.nBunches, nBuckets);
    if (!sc.bunchData)
      bombElegant("Bucket data not allocated for SCMULT (2). This is a bug!", NULL);
      
    for (iBucket = 0; iBucket < nBuckets; iBucket++) {
      if (nBuckets == 1) {
        time = time0;
        part = part0;
        np = np0;
        pbin = (long*)trealloc(pbin, sizeof(*pbin) * (max_np = np));
      } else {
        if ((np = npBucket[iBucket]) == 0)
          continue;
#ifdef DEBUG
        printf("SCMULT: copying data to work array, iBucket=%ld, np=%ld\n", iBucket, np);
        fflush(stdout);
#endif
        if (np > max_np) {
          if (part)
            free_czarray_2d((void **)part, max_np, totalPropertiesPerParticle);
          part = (double **)czarray_2d(sizeof(double), np, totalPropertiesPerParticle);
          time = (double *)tmalloc(sizeof(*time) * np);
          pbin = (long *)trealloc(pbin, sizeof(*pbin) * np);
          max_np = np;
        }
#if defined(HAVE_GPU) && defined(_OPENMP)
#  pragma omp parallel for num_threads(gpuGetOmpTrackingThreads()) schedule(static) if(gpuOmpTrackingRequested(np))
#endif
        for (ip=0; ip < np; ip++) {
          time[ip] = time0[ipBucket[iBucket][ip]];
          memcpy(part[ip], part0[ipBucket[iBucket][ip]], sizeof(double) * totalPropertiesPerParticle);
        }
      }
	
      tmax = -(tmin = DBL_MAX);
      find_min_max(&tmin, &tmax, time, np);
#ifdef DEBUG
      printf("SCMULT: tmin=%21.15le, tmax=%21.15le, np=%ld\n", tmin, tmax, np);
      fflush(stdout);
#endif
#if USE_MPI
      totalCharge = 0;
      if (isSlave && distributedBeam) {
	int64_t np_total;
	find_global_min_max(&tmin, &tmax, np, workers);
	MPI_Allreduce(&np, &np_total, 1, MPI_INT64_T, MPI_SUM, workers);
	totalCharge = np_total*charge->macroParticleCharge;
#ifdef DEBUG
	printf("SCMULT: global tmin=%21.15le, tmax=%21.15le, np=%ld, Q=%le C\n", tmin, tmax, np_total, totalCharge);
	fflush(stdout);
#endif
      }
#else
      totalCharge = np*charge->macroParticleCharge;
#ifdef DEBUG
      printf("SCMULT: global tmin=%21.15le, tmax=%21.15le, np=%ld, Q=%le C\n", tmin, tmax, np, totalCharge);
      fflush(stdout);
#endif
#endif

      /* Compute rms sizes */
      for (int j=0; j<3; j++)
	sc.bunchData[iBucket].sigma[j] = computeRmsCoordinate(part, 2*j, np, &(sc.bunchData[iBucket].center[j]), NULL);
#ifdef DEBUG
      printf("SCMULT: sigmax, y, z = %le, %le, %le; %ld particles, %le C\n", sc.bunchData[iBucket].sigma[0], sc.bunchData[iBucket].sigma[1], sc.bunchData[iBucket].sigma[2], np, totalCharge);
      fflush(stdout);
#endif
      if (!(iPass == 0 || sc.averagingFactor == 1)) {
	/* average over turns if requested */
	for (int j=0; j<3; j++)
	  sc.bunchData[iBucket].sigma[j] = (1 - sc.averagingFactor) * ((SCMULT *)eptr->p_elem)->lastSigma[j] + sc.averagingFactor * sc.bunchData[iBucket].sigma[j];
      }
	      /* Save values in case we need them for future averaging */
	      for (int j=0; j<3; j++)
		((SCMULT *)eptr->p_elem)->lastSigma[j] = sc.bunchData[iBucket].sigma[j];
#ifdef HAVE_GPU
	      scmultOnGpu = 0;
	      if (gpu_scmult_linear_supported(np, nBuckets, sc.nonlinear,
	                                      sc.sliceDuration, sc.horizontal,
	                                      sc.vertical) ||
	          gpu_scmult_nonlinear_supported(np, nBuckets, sc.nonlinear,
	                                         sc.sliceDuration, sc.horizontal,
	                                         sc.vertical)) {
		if (sc.nonlinear)
		  gpu_track_through_scmult_nonlinear(np, totalCharge, sc.c1,
		                                   sc.horizontal, sc.vertical,
		                                   sc.uniform,
		                                   sc.bunchData[iBucket].center,
		                                   sc.bunchData[iBucket].sigma,
		                                   sc.bunchData[iBucket].dmux,
		                                   sc.bunchData[iBucket].dmuy,
		                                   eptr->twiss->betax,
		                                   eptr->twiss->betay);
		else
		  gpu_track_through_scmult_linear(np, totalCharge, sc.c1,
		                                sc.horizontal, sc.vertical,
		                                sc.uniform,
		                                sc.bunchData[iBucket].center,
		                                sc.bunchData[iBucket].sigma,
		                                sc.bunchData[iBucket].dmux,
		                                sc.bunchData[iBucket].dmuy,
		                                eptr->twiss->betax,
		                                eptr->twiss->betay);
#  ifdef GPU_VERIFY
#    if defined(_OPENMP)
#      pragma omp parallel for num_threads(gpuGetOmpTrackingThreads()) schedule(static) if(gpuOmpTrackingRequested(np))
#    endif
		for (i = 0; i < np; i++) {
		  applySCKick(part[i], eptr, sc.bunchData[iBucket].center,
			      sc.bunchData[iBucket].sigma, totalCharge, iBucket);
		}
		compareGpuCpu(np, sc.nonlinear ?
		              "trackThroughSCMULT nonlinear" :
		              "trackThroughSCMULT linear");
#  endif
		scmultOnGpu = 1;
	      }
	      if (!scmultOnGpu) {
#endif
		if (sc.sliceDuration<=0) {
		  /* compute kicks using unsliced method */
#if defined(HAVE_GPU) && defined(_OPENMP)
#  pragma omp parallel for num_threads(gpuGetOmpTrackingThreads()) schedule(static) if(gpuOmpTrackingRequested(np))
#endif
		  for (i = 0; i < np; i++) {
		    applySCKick(part[i], eptr, sc.bunchData[iBucket].center,
				sc.bunchData[iBucket].sigma, totalCharge, iBucket);
		  }
		} else {
		  /* compute kicks using sliced method */
		  long nSlices, iSlice;
		  double *QTime, *xyCentroidTime[2], *xySizeTime[2];
		  tmin -= sc.sliceDuration/2;
		  tmax += sc.sliceDuration/2;
		  if ((nSlices = (tmax-tmin)/sc.sliceDuration+1)<=0)
		    bombElegantVA((char*)"Error in trackThroughSCMULT: number of slices is %ld, t:[%le, %le], dt=%le\n",
				  nSlices, tmin, tmax, sc.sliceDuration);
		  QTime = (double*)calloc(nSlices, sizeof(*QTime));
		  xyCentroidTime[0] = (double*)calloc(nSlices, sizeof(*xyCentroidTime[0]));
		  xyCentroidTime[1] = (double*)calloc(nSlices, sizeof(*xyCentroidTime[1]));
		  xySizeTime[0] = (double*)calloc(nSlices, sizeof(*xySizeTime[0]));
		  xySizeTime[1] = (double*)calloc(nSlices, sizeof(*xySizeTime[1]));
		  binTimeDistribution(QTime, pbin, tmin, sc.sliceDuration, nSlices, time, part, Po, np);
		  binTransverseTimeDistribution(xyCentroidTime, NULL, pbin, tmin, sc.sliceDuration, nSlices, time, part, Po, np, 0.0, 0.0, 1, 1);
		  binTransverseTimeDistribution(xySizeTime, NULL, pbin, tmin, sc.sliceDuration, nSlices, time, part, Po, np, 0.0, 0.0, 2, 2);
#if USE_MPI
	if (isSlave && distributedBeam) {
	  /* Sum charge distribution across all processors */
	  double *buffer;
	  buffer = (double*)malloc(sizeof(double) * nSlices);
	  MPI_Allreduce(QTime, buffer, nSlices, MPI_DOUBLE, MPI_SUM, workers);
	  memcpy(QTime, buffer, sizeof(double) * nSlices);
	  for (int plane=0; plane<2; plane++) {
	    MPI_Allreduce(xyCentroidTime[plane], buffer, nSlices, MPI_DOUBLE, MPI_SUM, workers);
	    memcpy(xyCentroidTime[plane], buffer, sizeof(double) * nSlices);
	    MPI_Allreduce(xySizeTime[plane], buffer, nSlices, MPI_DOUBLE, MPI_SUM, workers);
	    memcpy(xySizeTime[plane], buffer, sizeof(double) * nSlices);
	  }
	  free(buffer);
	}
#endif
		  for (iSlice=0; iSlice<nSlices; iSlice++) {
		    if (QTime[iSlice]) {
		      double d;
		      for (int plane=0; plane<2; plane++) {
		        if (QTime[iSlice]>=sc.sliceThreshold)
			  /* compute slice centroid */
			  xyCentroidTime[plane][iSlice] /= QTime[iSlice];
		        else
			  xyCentroidTime[plane][iSlice] = sc.bunchData[iBucket].center[plane];
		        /* compute slice rms size. If invalid or too few particles, use whole-beam value */
		        if (QTime[iSlice]>=sc.sliceThreshold &&
			    (d = xySizeTime[plane][iSlice]/QTime[iSlice] - sqr(xyCentroidTime[plane][iSlice]))>=0)
			  xySizeTime[plane][iSlice] = sqrt(d);
		        else
			  xySizeTime[plane][iSlice] = sc.bunchData[iBucket].sigma[plane];
		      }
		    }
		    QTime[iSlice] *= charge->macroParticleCharge;
		  }
#if defined(HAVE_GPU) && defined(_OPENMP)
#  pragma omp parallel for num_threads(gpuGetOmpTrackingThreads()) schedule(static) if(gpuOmpTrackingRequested(np))
#endif
		  for (i = 0; i < np; i++) {
		    double centroid[3], sigma[3], sliceCharge;
		    long particleSlice = pbin[i];
		    double *particleCoord = part[i];
		    centroid[2] = 0;
		    sigma[2] = sc.sliceDuration*c_mks; /* not actually used */
		    if (sc.sliceInterpolation==0) {
		      for (int plane=0; plane<2; plane++) {
		        centroid[plane] = xyCentroidTime[plane][particleSlice];
		        sigma[plane] = xySizeTime[plane][particleSlice];
		      }
		      sliceCharge = QTime[particleSlice];
		    } else {
		      double timeOffset;
		      long ib;
		      short interpolate = sc.sliceInterpolation;
		      if ((ib = particleSlice)<0 || ib>(nSlices-1)) {
		        interpolate = 0;
		        timeOffset = 0;
		      }
		      else
		        timeOffset = time[i] - (tmin + ib*sc.sliceDuration); /* distance to bin center */
		      if ((timeOffset<0 && ib) || ib==nSlices-1) {
		        ib--;
		        timeOffset += sc.sliceDuration;
		      }
		      for (int plane=0; plane<2; plane++) {
		        if (!interpolate) {
			  centroid[plane] = xyCentroidTime[plane][particleSlice];
			  sigma[plane] = xySizeTime[plane][particleSlice];
			  sliceCharge = QTime[particleSlice];
		        } else {
			  centroid[plane] = xyCentroidTime[plane][ib] + (xyCentroidTime[plane][ib+1]-xyCentroidTime[plane][ib])/sc.sliceDuration*timeOffset;
			  sigma[plane] = xySizeTime[plane][ib] + (xySizeTime[plane][ib+1]-xySizeTime[plane][ib])/sc.sliceDuration*timeOffset;
			  sliceCharge = QTime[ib] + (QTime[ib+1]-QTime[ib])/sc.sliceDuration*timeOffset;
		        }
		      }
		    }
		    applySCKick(particleCoord, eptr, centroid, sigma,
				sliceCharge, iBucket);
		  }
		  free(QTime);
		  free(xyCentroidTime[0]);
		  free(xyCentroidTime[1]);
		  free(xySizeTime[0]);
		  free(xySizeTime[1]);
		}
#ifdef HAVE_GPU
	      }
#endif

      if (nBuckets != 1) {
	/* Move data back to input array */
#if defined(HAVE_GPU) && defined(_OPENMP)
#  pragma omp parallel for num_threads(gpuGetOmpTrackingThreads()) schedule(static) if(gpuOmpTrackingRequested(np))
#endif
        for (ip = 0; ip < np; ip++)
          memcpy(part0[ipBucket[iBucket][ip]], part[ip], sizeof(double) * totalPropertiesPerParticle);
      }
      sc.bunchData[iBucket].dmux = sc.bunchData[iBucket].dmuy = 0.0; /* reset space charge strength */
    }
  }

  if (pbin)
    free(pbin);
  if (time && time!=time0)
    free(time);
  if (part && part!=part0)
    free(part);
  if (isSlave || !distributedBeam)
    free_bunch_index_memory(time0, ibParticle, ipBucket, npBucket, nBuckets);

#ifdef DEBUG
  printf("returning from trackThroughSCMULT\n");
  fflush(stdout);
#endif
}

void linearSCKick(double *coord, ELEMENT_LIST *eptr, double *centroid, double *sigma, double charge, long iBunch) {
  double k0, kx, ky;
  if (sc.sliceDuration>0) {
    k0 = sc.c1 /(sc.sliceDuration*c_mks) * charge * sqrt(PIx2);
  } else {
    if (sc.uniform) {
      k0 = sc.c1 / sigma[2] * charge * sqrt(PI / 6.0);
    } else {
      k0 = sc.c1 / sigma[2] * charge * exp(-sqr(coord[4] - centroid[2]) / sqr(sigma[2]) / 2.0);
    }
  }
  if (sc.horizontal) {
    kx = k0 * sc.bunchData[iBunch].dmux / eptr->twiss->betax; /* From dmux to KL */
    coord[1] += kx * (coord[0] - centroid[0]);
  }
  if (sc.vertical) {
    ky = k0 * sc.bunchData[iBunch].dmuy / eptr->twiss->betay; /* From dmuy to KL */
    coord[3] += ky * (coord[2] - centroid[1]);
  }
}

int nonlinearSCKick(double *coord, ELEMENT_LIST *eptr, double *centroid, double *sigma, double *kick, double charge, long iBunch) {
  double k0, kx, ky;
  std::complex<double> wa, wb, w1, w2, w;
  long overflow;
  double x, y, z;

  x = coord[0] - centroid[0];
  y = coord[2] - centroid[1];
  z = coord[4] - centroid[2];

  if (sc.sliceDuration>0) {
    k0 = sc.c1/(sc.sliceDuration*c_mks) * charge * PI;
  } else {
    if (sc.uniform) {
      k0 = sc.c1/sigma[2] * charge * PI / 12.0;
    } else {
      k0 = sc.c1/sigma[2] * charge * exp(-sqr(z / sigma[2]) / 2.0) * sqrt(PI / 2.0);
    }
  }

  // See V. Ziemann, SLAC-PUB-5582
  if (fabs(sigma[0]-sigma[1])/sigma[0]<1e-6) {
    // special case for round beams
    double sig, r, dp, theta;
    sig = (sigma[0] + sigma[1])/2;
    dp = theta = 0;
    if ((r = sqrt(sqr(x)+sqr(y)))>0) {
      dp = (1- exp (-sqr(r/sig)/2))/r;
      theta = atan2(y, x);
    }
    k0 *= sqrt(2/PI);
    kick[0] = dp*cos(theta)*k0;
    kick[1] = dp*sin(theta)*k0;
  } else {
    short swapXY = 0;
    double sigmaMajor = sigma[0], sigmaMinor = sigma[1];
    if (sigmaMajor<sigmaMinor) {
      double tmp;
      swapXY = 1;
      SWAP_DOUBLE(sigmaMajor, sigmaMinor);
      tmp = x;
      x = y;
      y = -tmp;
    }

    double ay = fabs(y); // This allows handling y<0 case without numerical issues. See Ziemann SLAC-PUB-5582
    double sd = sqrt(2.0*(sqr(sigmaMajor) - sqr(sigmaMinor)));
    w1 = std::complex<double>(x / sd, ay / sd);
    w2 = std::complex<double>(x / sd * sigmaMinor / sigmaMajor, ay / sd * sigmaMajor / sigmaMinor);
    
    wa = complexErf(w1, &overflow);
    if (overflow)
      return (0);
    wb = complexErf(w2, &overflow);
    if (overflow)
      return (0);

    double C3 = exp(-sqr(x) / (2 * sqr(sigmaMajor)) - sqr(y) / (2 * sqr(sigmaMinor)));
    w = wa - C3 * wb;

    if (swapXY) {
      kx = k0 * sc.bunchData[iBunch].dmux * sigma[0] * sqrt((sigma[0] + sigma[1]) / fabs(sigma[0] - sigma[1])) / eptr->twiss->betax;
      ky = k0 * sc.bunchData[iBunch].dmuy * sigma[1] * sqrt((sigma[0] + sigma[1]) / fabs(sigma[0] - sigma[1])) / eptr->twiss->betay;

      kick[0] = -kx * w.real() * (y>0 ? 1 : -1); /* note that y is really x */
      kick[1] =  ky * w.imag();
    } else {
      kx = k0 * sc.bunchData[iBunch].dmux * sigma[0] * sqrt((sigma[0] + sigma[1]) / fabs(sigma[0] - sigma[1])) / eptr->twiss->betax;
      ky = k0 * sc.bunchData[iBunch].dmuy * sigma[1] * sqrt((sigma[0] + sigma[1]) / fabs(sigma[0] - sigma[1])) / eptr->twiss->betay;

      kick[0] = kx * w.imag();
      kick[1] = ky * w.real() * (y>0 ? 1 : -1);
    }      
  }
  return (1);
}

void initializeSCMULT(ELEMENT_LIST *eptr, double **part0, int64_t np0, double Po, long i_pass, long idSlotsPerBunch) {
  long *pbin = NULL;       /* array to record which bin each particle is in */
  double *time0 = NULL;    /* array to record arrival time of each particle */
  double *time = NULL;     /* array to record arrival time of each particle, for working bucket */
  double **part = NULL;    /* particle buffer for working bucket */
  long *ibParticle = NULL; /* array to record which bucket each particle is in */
  int64_t **ipBucket = NULL;  /* array to record particle indices in part0 array for all particles in each bucket */
  int64_t *npBucket = NULL;   /* array to record how many particles are in each bucket */
  /* long ib, nb = 0, n_binned = 0; */
  long iBucket, nBuckets;
  int64_t max_np = 0, ip, np;

  if (!eptr->twiss)
    bombElegant((char *)"Twiss parameters must be calculated before SC tracking.", NULL);

#ifdef DEBUG
  printf("initializeSCMULT 1, np=%ld\n", np0);
  fflush(stdout);
#endif

#ifdef HAVE_GPU
  if ((isSlave || !distributedBeam) &&
      (sc.nBunches == 0 || sc.nBunches == 1) &&
      gpu_scmult_can_initialize_on_gpu(np0)) {
    long gpuBuckets = 0;
    if (gpu_scmult_count_bunches(np0, idSlotsPerBunch, &gpuBuckets) &&
        gpuBuckets == 1) {
      if (sc.nBunches == 0) {
        sc.nBunches = 1;
        sc.bunchData = (BUNCH_DATA *)calloc(1, sizeof(BUNCH_DATA));
        if (!sc.bunchData)
          bombElegant("Unable to allocate bucket data for SCMULT.", NULL);
      }
      if (gpu_scmult_compute_centroid_sigma(np0, Po,
                                            sc.bunchData[0].center,
                                            sc.bunchData[0].sigma)) {
        sc.bunchData[0].dmux = sc.bunchData[0].dmuy = 0.0;
        sc.c0 = fabs(sqrt(2.0 / PI) * particleRadius / particleCharge);
        sc.c1 = sc.c0 / sqr(Po) / sqrt(sqr(Po) + 1.0);
        sc.length = 0.0;
        return;
      }
    }
  }
  if (gpu_scmult_can_skip_cpu(np0, idSlotsPerBunch) || getElementOnGpu())
    part0 = forceParticlesToCpu("initializeSCMULT fallback");
#endif

  if (isSlave || !distributedBeam) {
#ifdef DEBUG
    printf("indexing bucket assignments\n");
    fflush(stdout);
#endif
    index_bunch_assignments(part0, np0, idSlotsPerBunch, Po, &time0, &ibParticle, &ipBucket, &npBucket, &nBuckets, -1);
#ifdef DEBUG
    printf("%ld bunches found\n", nBuckets);
    fflush(stdout);
#endif
  
    if (sc.nBunches!=0 && sc.nBunches!=nBuckets)
      bombElegantVA((char*)"Change in number of bunches from %ld to %ld seen in SCMULT. Not supported.\n",
		    sc.nBunches, nBuckets);
    if (sc.nBunches==0) {
      sc.nBunches = nBuckets;
      sc.bunchData = (BUNCH_DATA*)calloc(nBuckets, sizeof(BUNCH_DATA));
    }

    for (iBucket = 0; iBucket < nBuckets; iBucket++) {
#ifdef DEBUG
      printf("Working on bucket %ld\n", iBucket);
      fflush(stdout);
#endif
      if (nBuckets == 1) {
        time = time0;
        part = part0;
        np = np0;
        pbin = (long*)trealloc(pbin, sizeof(*pbin) * (max_np = np));
      } else {
        if ((np = npBucket[iBucket]) == 0)
          continue;
#ifdef DEBUG
        printf("SCMULT: copying data to work array, iBucket=%ld, np=%ld\n", iBucket, np);
        fflush(stdout);
#endif
        if (np > max_np) {
          if (part)
            free_czarray_2d((void **)part, max_np, totalPropertiesPerParticle);
          part = (double **)czarray_2d(sizeof(double), np, totalPropertiesPerParticle);
          time = (double *)tmalloc(sizeof(*time) * np);
          pbin = (long *)trealloc(pbin, sizeof(*pbin) * np);
          max_np = np;
        }
        for (ip=0; ip < np; ip++) {
          time[ip] = time0[ipBucket[iBucket][ip]];
          memcpy(part[ip], part0[ipBucket[iBucket][ip]], sizeof(double) * totalPropertiesPerParticle);
        }
      }
      
#if USE_MPI
      for (int j=0; j<3; j++)
	sc.bunchData[iBucket].sigma[j] = computeRmsCoordinate(part, 2*j, np, &(sc.bunchData[iBucket].center[j]), NULL);
#else
      for (int j=0; j<3; j++)
	sc.bunchData[iBucket].sigma[j] = computeRmsCoordinate(part, 2*j, np, &(sc.bunchData[iBucket].center[j]), NULL);
#endif
      sc.bunchData[iBucket].dmux = sc.bunchData[iBucket].dmuy = 0.0;
    }
  }

#if USE_MPI
  // Share data with master?
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (pbin)
    free(pbin);
  if (time && time!=time0)
    free(time);
  if (part && part!=part0)
    free(part);
  if (isSlave || !distributedBeam)
    free_bunch_index_memory(time0, ibParticle, ipBucket, npBucket, nBuckets);
  
#ifdef DEBUG
  printf("initializeSCMULT 2\n");
  fflush(stdout);
#endif

  sc.c0 = fabs(sqrt(2.0 / PI) * particleRadius / particleCharge);
  sc.c1 = sc.c0 / sqr(Po) / sqrt(sqr(Po) + 1.0);

  sc.length = 0.0;

#ifdef DEBUG
  printf("initializeSCMULT 3\n");
  fflush(stdout);
#endif
}

void accumulateSCMULT(double **part0, int64_t np0, double Po, ELEMENT_LIST *eptr, long idSlotsPerBunch) {
  long *pbin = NULL;       /* array to record which bin each particle is in */
  double *time0 = NULL;    /* array to record arrival time of each particle */
  double *time = NULL;     /* array to record arrival time of each particle, for working bucket */
  double **part = NULL;    /* particle buffer for working bucket */
  long *ibParticle = NULL; /* array to record which bucket each particle is in */
  int64_t **ipBucket = NULL;  /* array to record particle indices in part0 array for all particles in each bucket */
  int64_t *npBucket = NULL;   /* array to record how many particles are in each bucket */
  /* long ib, nb = 0, n_binned = 0; */
  long iBucket, nBuckets;
  int64_t max_np = 0, ip, np;
  TWISS *twiss0;
  double dmux, dmuy, temp;
  double length;

  twiss0 = (eptr->pred)->twiss;
  
#ifdef DEBUG
  printf("accumulateSCMULT 1, np=%ld\n", np0);
  fflush(stdout);
#endif

#ifdef HAVE_GPU
  if ((isSlave || !distributedBeam) && sc.nBunches == 1 && sc.bunchData &&
      eptr && eptr->twiss && eptr->pred && eptr->pred->twiss &&
      gpu_scmult_can_skip_cpu(np0, idSlotsPerBunch)) {
    temp = sc.bunchData[0].sigma[0] + sc.bunchData[0].sigma[1];
    dmux = twiss0->betax / sc.bunchData[0].sigma[0] / temp;
    dmuy = twiss0->betay / sc.bunchData[0].sigma[1] / temp;
    if (gpu_scmult_compute_centroid_sigma(np0, Po,
                                          sc.bunchData[0].center,
                                          sc.bunchData[0].sigma)) {
      twiss0 = eptr->twiss;
      temp = sc.bunchData[0].sigma[0] + sc.bunchData[0].sigma[1];
      dmux += twiss0->betax / sc.bunchData[0].sigma[0] / temp;
      dmuy += twiss0->betay / sc.bunchData[0].sigma[1] / temp;
      length = ((DRIFT *)eptr->p_elem)->length;
      sc.bunchData[0].dmux += dmux * length / 2.0;
      sc.bunchData[0].dmuy += dmuy * length / 2.0;
      return;
    }
  }
  if (gpu_scmult_can_skip_cpu(np0, idSlotsPerBunch) || getElementOnGpu())
    part0 = forceParticlesToCpu("accumulateSCMULT fallback");
#endif

  if (isSlave || !distributedBeam) {
#ifdef DEBUG
    printf("indexing bucket assignments\n");
    fflush(stdout);
#endif
    index_bunch_assignments(part0, np0, idSlotsPerBunch, Po, &time0, &ibParticle, &ipBucket, &npBucket, &nBuckets, -1);
#ifdef DEBUG
    printf("%ld bunches found\n", nBuckets);
    fflush(stdout);
#endif
    
    if (sc.nBunches!=0 && sc.nBunches!=nBuckets)
      bombElegantVA((char*)"Change in number of bunches from %ld to %ld seen in SCMULT. Not supported.\n",
		    sc.nBunches, nBuckets);
    if (!sc.bunchData)
      bombElegant("Bucket data not allocated for SCMULT (1). This is a bug!", NULL);
    
    for (iBucket = 0; iBucket < nBuckets; iBucket++) {
      if (nBuckets == 1) {
        time = time0;
        part = part0;
        np = np0;
        pbin = (long*)trealloc(pbin, sizeof(*pbin) * (max_np = np));
      } else {
        if ((np = npBucket[iBucket]) == 0)
          continue;
#ifdef DEBUG
        printf("SCMULT: copying data to work array, iBucket=%ld, np=%ld\n", iBucket, np);
        fflush(stdout);
#endif
        if (np > max_np) {
          if (part)
            free_czarray_2d((void **)part, max_np, totalPropertiesPerParticle);
          part = (double **)czarray_2d(sizeof(double), np, totalPropertiesPerParticle);
          time = (double *)tmalloc(sizeof(*time) * np);
          pbin = (long *)trealloc(pbin, sizeof(*pbin) * np);
          max_np = np;
        }
        for (ip=0; ip < np; ip++) {
          time[ip] = time0[ipBucket[iBucket][ip]];
          memcpy(part[ip], part0[ipBucket[iBucket][ip]], sizeof(double) * totalPropertiesPerParticle);
        }
      }
      
      temp = sc.bunchData[iBucket].sigma[0] + sc.bunchData[iBucket].sigma[1];
      dmux = twiss0->betax / sc.bunchData[iBucket].sigma[0] / temp;
      dmuy = twiss0->betay / sc.bunchData[iBucket].sigma[1] / temp;
      
      
#if USE_MPI
      for (int j=0; j<3; j++)
	sc.bunchData[iBucket].sigma[j] = computeRmsCoordinate(part, 2*j, np, &(sc.bunchData[iBucket].center[j]), NULL);
#else
      for (int j=0; j<3; j++)
	sc.bunchData[iBucket].sigma[j] = computeRmsCoordinate(part, 2*j, np, &(sc.bunchData[iBucket].center[j]), NULL);
#endif
      twiss0 = eptr->twiss;
      temp = sc.bunchData[iBucket].sigma[0] + sc.bunchData[iBucket].sigma[1];
      dmux += twiss0->betax / sc.bunchData[iBucket].sigma[0] / temp;
      dmuy += twiss0->betay / sc.bunchData[iBucket].sigma[1] / temp;
      
      length = ((DRIFT *)eptr->p_elem)->length;
      sc.bunchData[iBucket].dmux += dmux * length / 2.0;
      sc.bunchData[iBucket].dmuy += dmuy * length / 2.0;
    }
  }
#if USE_MPI
  // Share data with master?
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (pbin)
    free(pbin);
  if (time && time!=time0)
    free(time);
  if (part && part!=part0)
    free(part);
  if (isSlave || !distributedBeam)
    free_bunch_index_memory(time0, ibParticle, ipBucket, npBucket, nBuckets);
}

double computeRmsCoordinate(double **coord, long i1, int64_t np, double *meanReturn, long *countReturn)
/* Confusingly, this routine works fine in parallel, provided all processors except the master participate */
{
  double vrms = 0.0, xc = 0.0;
  int64_t i;
#if USE_MPI
  double xc_sum = 0.0, vrms_sum = 0.0;
  int64_t np_total;
#endif

  if (!USE_MPI || !distributedBeam) {
    if (!np)
      return (0.0);
  }
#if USE_MPI
  else {
    if (isMaster)
      np = 0;
    MPI_Allreduce(&np, &np_total, 1, MPI_INT64_T, MPI_SUM, workers);
    if (!np_total)
      return (0.0);
  }
#endif

  /* compute centroids */
  for (i = xc = 0; i < np; i++) {
    xc += coord[i][i1];
  }
  /* Compute the sum of xc across all the processors */
  if (!USE_MPI || !distributedBeam)
    xc /= np;
#if USE_MPI
  else {
    MPI_Allreduce(&xc, &xc_sum, 1, MPI_DOUBLE, MPI_SUM, workers);
    xc = xc_sum / np_total;
  }
#endif
  for (i = vrms = 0; i < np; i++) {
    vrms += sqr(coord[i][i1] - xc);
  }
  if (!USE_MPI || !distributedBeam)
    vrms /= np;
#if USE_MPI
  else {
    MPI_Allreduce(&vrms, &vrms_sum, 1, MPI_DOUBLE, MPI_SUM, workers);
    vrms = vrms_sum / np_total;
  }
#endif
  if (meanReturn)
    *meanReturn = xc;
#if USE_MPI
  if (countReturn)
    *countReturn = np_total;
#else
  if (countReturn)
    *countReturn = np;
#endif

  return (sqrt(vrms));
}

#if USE_MPI
/* We have this new function as we need treat the parallel and serial element separately */
double computeRmsCoordinate_p(double **coord, long i1, int64_t np, double *centroid, int64_t *npTotal, unsigned long classFlags) {
  double vrms = 0.0, xc = 0.0;
  int64_t np_total;
  int64_t i;

  if (centroid)
    *centroid = 0;
  if (npTotal)
    *npTotal = 0;

  if (classFlags & UNIPROCESSOR) { /* serial element, only master works */
    MPI_Bcast(&np, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    if (npTotal)
      *npTotal = np;
    if (!np)
      return (0.0);

    /* compute centroids */
    if (isMaster) {
      for (i = xc = 0; i < np; i++) {
        xc += coord[i][i1];
      }
      xc /= np;
    }
    if (centroid)
      *centroid = xc;
    /* Broadcast the xc from master to all the slaves */
    MPI_Bcast(&xc, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (isMaster) {
      for (i = vrms = 0; i < np; i++) {
        vrms += sqr(coord[i][i1] - xc);
      }
      vrms /= np;
    }
    /* Broadcast the vrms from master to all the slaves */
    MPI_Bcast(&vrms, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else { /* parallel element, only slaves works */
    double xc_sum = 0.0, vrms_sum = 0.0;

    if (isMaster)
      np = 0;
    MPI_Allreduce(&np, &np_total, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    if (npTotal)
      *npTotal = np_total;

    /* compute centroids */
    if (isSlave) {
      for (i = xc = 0; i < np; i++) {
        xc += coord[i][i1];
      }
    }
    /* Compute the sum of xc across all the processors */
    MPI_Allreduce(&xc, &xc_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    xc = xc_sum / np_total;
    if (centroid)
      *centroid = xc;

    if (isSlave) {
      for (i = vrms = 0; i < np; i++) {
        vrms += sqr(coord[i][i1] - xc);
      }
    }
    /* Compute the sum of vrms across all the processors */
    MPI_Allreduce(&vrms, &vrms_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    vrms = vrms_sum / np_total;
  }
  return sqrt(vrms);
}
#endif
