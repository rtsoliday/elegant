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

/* There's no reason to use a structure here since these are all local variables.
 * Note that they can't be part of the SCMULT elements since the information is needed
 * across such elements.
 */
typedef struct {
  long verbosity;
  long horizontal, vertical, longitudinal, uniform;
  long nonlinear, sliceThreshold, sliceInterpolation;
  double sliceDuration;
  double averagingFactor, center[3];
  double sigma[3];
  double c0;   			/* c0=re*np/(2*Pi)^(3/2) -> calculate once.  */
  double c1;	 		/* c1=c0/p0^3 */
  double dmux, dmuy;
  double length;
} SPACE_CHARGE;

static SPACE_CHARGE sc;

void linearSCKick(double *coord, ELEMENT_LIST *eptr, double *centroid, double charge);
int nonlinearSCKick(double *coord, ELEMENT_LIST *eptr, double *centroid,
                    double sigmax, double sigmay, double *kick, double charge);


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

  sc.vertical = vertical;
  sc.horizontal = horizontal;
  sc.longitudinal = longitudinal;
  sc.nonlinear = nonlinear;
  sc.uniform = uniform_distribution;
  sc.sliceDuration = slice_duration;
  sc.sliceThreshold = slice_threshold;
  sc.sliceInterpolation = slice_interpolation;
  
  if ((sc.averagingFactor = averaging_factor) <= 0 || averaging_factor > 1)
    bombElegant("averaging_factor must be on (0, 1]", NULL);
  if (averaging_factor < 1 && !nonlinear)
    bombElegant("averaging_factor is ignore for linear mode", NULL);

  if (longitudinal)
    printWarning((char *)"The longitudinal space-charge effect is not implemented by SCMULT.",
                 (char *)"Consider using LSCDRIFT elements.");

}

/* track through space charge element */
void trackThroughSCMULT(double **part0, long np0, double Po, long iPass, ELEMENT_LIST *eptr, CHARGE *charge) {
  long i;
  long *pbin = NULL;       /* array to record which bin each particle is in */
  double *time0 = NULL;    /* array to record arrival time of each particle */
  double *time = NULL;     /* array to record arrival time of each particle, for working bucket */
  double **part = NULL;    /* particle buffer for working bucket */
  long *ibParticle = NULL; /* array to record which bucket each particle is in */
  long **ipBucket = NULL;  /* array to record particle indices in part0 array for all particles in each bucket */
  long *npBucket = NULL;   /* array to record how many particles are in each bucket */
  /* long ib, nb = 0, n_binned = 0; */
  long iBucket, nBuckets, max_np = 0, ip, np;
  double tmin, tmax;
  double *coord;
  double kick[2];
  double totalCharge;
  int flag;

#ifdef DEBUG
  printf("entered trackThroughSCMULT\n");
  fflush(stdout);
#endif

  if (isSlave || !notSinglePart) {
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
	
      tmax = -(tmin = DBL_MAX);
      find_min_max(&tmin, &tmax, time, np);
#ifdef DEBUG
      printf("SCMULT: tmin=%21.15le, tmax=%21.15le, np=%ld\n", tmin, tmax, np);
      fflush(stdout);
#endif
#if USE_MPI
      totalCharge = 0;
      if (isSlave && notSinglePart) {
	long np_total;
	find_global_min_max(&tmin, &tmax, np, workers);
	MPI_Allreduce(&np, &np_total, 1, MPI_LONG, MPI_SUM, workers);
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
	sc.sigma[j] = computeRmsCoordinate(part, 2*j, np, &(sc.center[j]), NULL);
#ifdef DEBUG
      printf("SCMULT: sigmax, y, z = %le, %le, %le; %ld particles, %le C\n", sc.sigma[0], sc.sigma[1], sc.sigma[2], np, totalCharge);
      fflush(stdout);
#endif
      if (!(iPass == 0 || sc.averagingFactor == 1)) {
	/* average over turns if requested */
	for (int j=0; j<3; j++)
	  sc.sigma[j] = (1 - sc.averagingFactor) * ((SCMULT *)eptr->p_elem)->lastSigma[j] + sc.averagingFactor * sc.sigma[j];
      }
      /* Save values in case we need them for future averaging */
      for (int j=0; j<3; j++)
	((SCMULT *)eptr->p_elem)->lastSigma[j] = sc.sigma[j];
      if (sc.sliceDuration<=0) {
	/* compute kicks using unsliced method */
	for (i = 0; i < np; i++) {
	  coord = part[i];
	  if (!sc.nonlinear) {
	    linearSCKick(coord, eptr, sc.center, totalCharge);
	  } else {
	    flag = nonlinearSCKick(coord, eptr, sc.center, sc.sigma[0], sc.sigma[1], kick, totalCharge);
	    if (!flag) {
	      linearSCKick(coord, eptr, sc.center, totalCharge);
	      continue;
	    }
	    coord[1] += kick[0];
	    coord[3] += kick[1];
	  }
	}
      } else {
	/* compute kicks using sliced method */
	long nSlices, iSlice;
	double *QTime, *xyCentroidTime[2], *xySizeTime[2], centroid[3], sigma[3], sliceCharge;
	tmin -= sc.sliceDuration/2;
	tmax += sc.sliceDuration/2;
	if ((nSlices = (tmax-tmin)/sc.sliceDuration+1)<=0)
	  bombElegantVA("Error in trackThroughSCMULT: number of slices is %ld, t:[%le, %le], dt=%le\n",
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
	if (isSlave && notSinglePart) {
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
	sc.sigma[2] = sc.sliceDuration*c_mks;
	for (iSlice=0; iSlice<nSlices; iSlice++) {
	  if (QTime[iSlice]) {
	    double d;
	    for (int plane=0; plane<2; plane++) {
	      if (QTime[iSlice]>=sc.sliceThreshold)
		/* compute slice centroid */
		xyCentroidTime[plane][iSlice] /= QTime[iSlice];
	      else
		xyCentroidTime[plane][iSlice] = sc.center[plane];
	      /* compute slice rms size. If invalid or too few particles, use whole-beam value */
	      if (QTime[iSlice]>=sc.sliceThreshold &&
		  (d = xySizeTime[plane][iSlice]/QTime[iSlice] - sqr(xyCentroidTime[plane][iSlice]))>=0)
		xySizeTime[plane][iSlice] = sqrt(d);
	      else
		xySizeTime[plane][iSlice] = sc.sigma[plane];
	    }
	  }
	  QTime[iSlice] *= charge->macroParticleCharge;
	}
	centroid[2] = 0;
	sigma[2] = sc.sliceDuration*c_mks;
	for (i = 0; i < np; i++) {
	  coord = part[i];
	  iSlice = pbin[i];
	  if (sc.sliceInterpolation==0) {
	    for (int plane=0; plane<2; plane++) {
	      centroid[plane] = xyCentroidTime[plane][iSlice];
	      sigma[plane] = xySizeTime[plane][iSlice];
	    }
	    sliceCharge = QTime[iSlice];
	  } else {
	    double timeOffset;
	    long ib;
	    short interpolate = sc.sliceInterpolation;
	    if ((ib = iSlice)<0 || ib>(nSlices-1)) {
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
		centroid[plane] = xyCentroidTime[plane][iSlice];
		sigma[plane] = xySizeTime[plane][iSlice];
		sliceCharge = QTime[iSlice];
	      } else {
		centroid[plane] = xyCentroidTime[plane][ib] + (xyCentroidTime[plane][ib+1]-xyCentroidTime[plane][ib])/sc.sliceDuration*timeOffset;
		sigma[plane] = xySizeTime[plane][ib] + (xySizeTime[plane][ib+1]-xySizeTime[plane][ib])/sc.sliceDuration*timeOffset;
		sliceCharge = QTime[ib] + (QTime[ib+1]-QTime[ib])/sc.sliceDuration*timeOffset;
	      }
	    }
	  }
	  if (!sc.nonlinear) {
	    linearSCKick(coord, eptr, centroid, sliceCharge);
	  } else {
	    flag = nonlinearSCKick(coord, eptr, centroid, sigma[0], sigma[1], kick, sliceCharge);
	    if (!flag) {
	      linearSCKick(coord, eptr, centroid, sliceCharge);
	      continue;
	    }
	    coord[1] += kick[0];
	    coord[3] += kick[1];
	  }
	}
	free(QTime);
	free(xyCentroidTime[0]);
	free(xyCentroidTime[1]);
	free(xySizeTime[0]);
	free(xySizeTime[1]);
      }
	
      if (nBuckets != 1) {
	/* Move data back to input array */
        for (ip = 0; ip < np; ip++)
          memcpy(part0[ipBucket[iBucket][ip]], part[ip], sizeof(double) * totalPropertiesPerParticle);
      }
    }
  }

  if (pbin)
    free(pbin);
  if (time && time!=time0)
    free(time);
  if (part && part!=part0)
    free(part);
  if (isSlave || !notSinglePart)
    free_bunch_index_memory(time0, ibParticle, ipBucket, npBucket, nBuckets);

  sc.dmux = sc.dmuy = 0.0; /* reset space charge strength */
#ifdef DEBUG
  printf("returning from trackThroughSCMULT\n");
  fflush(stdout);
#endif
}

void linearSCKick(double *coord, ELEMENT_LIST *eptr, double *centroid, double charge) {
  double k0, kx, ky;
  if (sc.sliceDuration>0) {
    k0 = sc.c1 / sc.sigma[2] * charge * sqrt(PIx2);
  } else {
    if (sc.uniform) {
      k0 = sc.c1 / sc.sigma[2] * charge * sqrt(PI / 6.0);
    } else {
      k0 = sc.c1 / sc.sigma[2] * charge * exp(-sqr(coord[4] - centroid[2]) / sqr(sc.sigma[2]) / 2.0);
    }
  }
  if (sc.horizontal) {
    kx = k0 * sc.dmux / eptr->twiss->betax; /* From dmux to KL */
    coord[1] += kx * (coord[0] - centroid[0]);
  }
  if (sc.vertical) {
    ky = k0 * sc.dmuy / eptr->twiss->betay; /* From dmuy to KL */
    coord[3] += ky * (coord[2] - centroid[1]);
  }
}

int nonlinearSCKick(double *coord, ELEMENT_LIST *eptr, double *centroid,
                    double sigmax, double sigmay, double *kick, double charge) {
  double k0, kx, ky;
  std::complex<double> wa, wb, w1, w2, w;
  long overflow;
  double x, y, z;

  x = coord[0] - centroid[0];
  y = coord[2] - centroid[1];
  z = coord[4] - centroid[2];

  if (sc.sliceDuration>0) {
    k0 = sc.c1/sc.sigma[2] * charge * PI;
  } else {
    if (sc.uniform) {
      k0 = sc.c1/sc.sigma[2] * charge * PI / 12.0;
    } else {
      k0 = sc.c1/sc.sigma[2] * charge * exp(-sqr(z / sc.sigma[2]) / 2.0) * sqrt(PI / 2.0);
    }
  }

  if (fabs(sigmax-sigmay)/sigmax<1e-6) {
    // special case for round beams
    double sig = (sigmax + sigmay)/2;
    double r = sqrt(sqr(x)+sqr(y));
    double dp = (1- exp (-sqr(r/sig)/2))/r;
    double theta = atan2(y, x);
    k0 *= sqrt(2/PI);
    kick[0] = dp*cos(theta)*k0;
    kick[1] = dp*sin(theta)*k0;
  } else {
    short swapXY = 0;
    if (sigmax<sigmay) {
      double tmp;
      swapXY = 1;
      SWAP_DOUBLE(sigmax, sigmay);
      tmp = x;
      x = y;
      y = -tmp;
    }

    double ay = fabs(y);
    double sd = sqrt(2.0*(sqr(sigmax) - sqr(sigmay)));
    w1 = std::complex<double>(x / sd, ay / sd);
    w2 = std::complex<double>(x / sd * sigmay / sigmax, ay / sd * sigmax / sigmay);
    
    wa = complexErf(w1, &overflow);
    if (overflow)
      return (0);
    wb = complexErf(w2, &overflow);
    if (overflow)
      return (0);

    double C3 = exp(-sqr(x) / (2 * sqr(sigmax)) - sqr(y) / (2 * sqr(sigmay)));
    w = wa + C3 * wb;

    kx = k0 * sc.dmux * sigmax * sqrt(sigmax + sigmay) / sqrt(fabs(sigmax - sigmay)) / eptr->twiss->betax;
    ky = k0 * sc.dmuy * sigmay * sqrt(sigmax + sigmay) / sqrt(fabs(sigmax - sigmay)) / eptr->twiss->betay;

    kick[0] = kx * w.imag();
    kick[1] = ky * w.real() * (y>0 ? 1 : -1);

    if (swapXY) {
      double tmp;
      tmp = kick[0];
      kick[0] = -kick[1];
      kick[1] = tmp;
    }
  }
  return (1);
}

void initializeSCMULT(ELEMENT_LIST *eptr, double **part, long np, double Po, long i_pass) {
  if (!eptr->twiss)
    bombElegant((char *)"Twiss parameters must be calculated before SC tracking.", NULL);

#ifdef DEBUG
  printf("initializeSCMULT 0\n");
  fflush(stdout);
#endif

#ifdef DEBUG
  printf("initializeSCMULT 1\n");
  fflush(stdout);
#endif

#if USE_MPI
  /* We set it as single particle case, as the particles have not been distributed 
     when the function is called, all the processors will do the same */
  notSinglePart = 0;
#endif
  for (int j=0; j<3; j++)
    sc.sigma[j] = computeRmsCoordinate(part, 2*j, np, NULL, NULL);
#ifdef DEBUG
  printf("initializeSCMULT 2\n");
  fflush(stdout);
#endif
#if USE_MPI
  /* set it back to parallel execution */
  notSinglePart = 1;
#endif
  sc.c0 = fabs(sqrt(2.0 / PI) * particleRadius / particleCharge);
  sc.c1 = sc.c0 / sqr(Po) / sqrt(sqr(Po) + 1.0);
  /* printf("r=%g, chargeperparticle=%g, particleCharge=%g\n",  particleRadius,  sc.chargePerParticle, particleCharge); 
     printf("c0=%g, c1=%g, sz=%.6g\n", sc.c0, sc.c1, sc.sigma[2]); */

  sc.dmux = sc.dmuy = 0.0;
  sc.length = 0.0;
#ifdef DEBUG
  printf("initializeSCMULT 3\n");
  fflush(stdout);
#endif
}

void accumulateSCMULT(double **part, long np, ELEMENT_LIST *eptr) {
  TWISS *twiss0;
  double dmux, dmuy, temp;
  double length;

  twiss0 = (eptr->pred)->twiss;
  temp = sc.sigma[0] + sc.sigma[1];
  dmux = twiss0->betax / sc.sigma[0] / temp;
  dmuy = twiss0->betay / sc.sigma[1] / temp;
#if USE_MPI
  for (int j=0; j<3; j++)
    sc.sigma[j] = computeRmsCoordinate_p(part, 2*j, np, &(sc.center[j]), NULL, entity_description[eptr->type].flags);
#else
  for (int j=0; j<3; j++)
    sc.sigma[j] = computeRmsCoordinate(part, 2*j, np, &(sc.center[j]), NULL);
#endif
  twiss0 = eptr->twiss;
  temp = sc.sigma[0] + sc.sigma[1];
  dmux += twiss0->betax / sc.sigma[0] / temp;
  dmuy += twiss0->betay / sc.sigma[1] / temp;

  length = ((DRIFT *)eptr->p_elem)->length;
  sc.dmux += dmux * length / 2.0;
  sc.dmuy += dmuy * length / 2.0;
}

double computeRmsCoordinate(double **coord, long i1, long np, double *meanReturn, long *countReturn)
/* Confusingly, this routine works fine in parallel, provided all processors except the master participate */
{
  double vrms = 0.0, xc = 0.0;
  long i;
#if USE_MPI
  double xc_sum = 0.0, vrms_sum = 0.0;
  long np_total;
#endif

  if (!USE_MPI || !notSinglePart) {
    if (!np)
      return (0.0);
  }
#if USE_MPI
  else {
    if (isMaster)
      np = 0;
    MPI_Allreduce(&np, &np_total, 1, MPI_LONG, MPI_SUM, workers);
    if (!np_total)
      return (0.0);
  }
#endif

  /* compute centroids */
  for (i = xc = 0; i < np; i++) {
    xc += coord[i][i1];
  }
  /* Compute the sum of xc across all the processors */
  if (!USE_MPI || !notSinglePart)
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
  if (!USE_MPI || !notSinglePart)
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
double computeRmsCoordinate_p(double **coord, long i1, long np, double *centroid, long *npTotal, unsigned long classFlags) {
  double vrms = 0.0, xc = 0.0;
  long i, np_total;

  if (centroid)
    *centroid = 0;
  if (npTotal)
    *npTotal = 0;

  if (classFlags & UNIPROCESSOR) { /* serial element, only master works */
    MPI_Bcast(&np, 1, MPI_LONG, 0, MPI_COMM_WORLD);
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
    MPI_Allreduce(&np, &np_total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
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
