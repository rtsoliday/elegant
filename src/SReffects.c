/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* file: SReffects.c
 * contents: track_SReffects()
 *
 * Michael Borland, 1995
 */
#include "mdb.h"
#include "track.h"
#ifdef HAVE_GPU
#  include "gpu/gpu_sreffects.h"
#endif

void track_SReffects(double **coord, long np, SREFFECTS *SReffects0, double Po,
                     TWISS *twiss, RADIATION_INTEGRALS *radIntegrals,
                     long lossOnly) {
  long ip;
  double Fx, Fy, Fdelta, Ddelta, P, t;
  double gamma2Ratio, gammaRatio;
  double Srxp, Sryp, Srdelta;
  double xpEta, ypEta;
  double *part, beta;
  static long first = 1;
  double deltaChange, cutoff;
  long active = 1;
  SREFFECTS SReffects;
#if defined(HAVE_GPU) && defined(GPU_VERIFY)
  long gpuTracked = 0;
#endif

#if USE_MPI
  mpiAbort = 0;
#endif

#if USE_MPI
  if (isMaster && distributedBeam)
    /* This is a parallel element, the master will not track unless it is a single particle simulation */
    active = 0;
#endif

  if (!twiss) {
    printf("Problem with SREFFECTS: no twiss parameters were computed\n");
    printf("It may be necessary to pre-compute twiss parameters using an additional\n");
    printf("twiss_output command with output_at_each_step=0\n");
    fflush(stdout);
    exitElegant(1);
  }

  memcpy(&SReffects, SReffects0, sizeof(SReffects));

  if (SReffects.pRef == 0) {
    if (!radIntegrals || !radIntegrals->computed) {
      bombElegant("Problem with SREFFECTS element: pRef=0 but no radiation integrals computed.  Use the twiss_output command to compute these.", NULL);
    }
    /* take data from radiation integrals */
    SReffects.pRef = radIntegrals->Pref;
    SReffects.Jx = radIntegrals->Jx;
    SReffects.Jy = radIntegrals->Jy;
    SReffects.Jdelta = radIntegrals->Jdelta;
    SReffects.exRef = radIntegrals->ex0 / (1 + SReffects.coupling * SReffects.Jy / SReffects.Jx);
    SReffects.eyRef = SReffects.exRef * SReffects.coupling;
    SReffects.SdeltaRef = radIntegrals->sigmadelta;
    SReffects.DdeltaRef = -radIntegrals->Uo / particleMassMV / Po;
    if (first) {
      printf("SReffects set up:\nPref=%g, Jx/y/delta=%g/%g/%g\nex/ey=%g/%g  Sdelta=%g  Ddelta=%g\n",
             SReffects.pRef, SReffects.Jx, SReffects.Jy, SReffects.Jdelta,
             SReffects.exRef, SReffects.eyRef, SReffects.SdeltaRef, SReffects.DdeltaRef);
      fflush(stdout);
    }
  } else {
    if (SReffects.eyRef != 0 && SReffects.coupling != 0) {
#if USE_MPI
      mpiAbort = MPI_ABORT_SREFFECTS_ERROR1;
      return;
#else
      bombElegant("Error: SREFFECTS element has nonzero values for both EYREF and COUPLING", NULL);
#endif
    }
    if (SReffects.eyRef == 0 || SReffects.coupling != 0) {
      SReffects.exRef = SReffects.exRef / (1 + SReffects.coupling * SReffects.Jy / SReffects.Jx);
      SReffects.eyRef = SReffects.exRef * SReffects.coupling;
    }
  }

  gamma2Ratio = (sqr(Po) + 1) / (sqr(SReffects.pRef) + 1);
  gammaRatio = sqrt(gamma2Ratio);

#if (!USE_MPI) /* The parallel version doesn't check here, as it needs communications every time */
  if (SReffects.DdeltaRef > 0) {
    /* this is a temporary kludge to ensure that all particles get lost when this happens */
#  ifdef HAVE_GPU
    if (getElementOnGpu())
      coord = forceParticlesToCpu("unphysical SREFFECTS parameters");
#  endif
    for (ip = 0; ip < np; ip++)
      coord[ip][5] = -1;
    printWarningForTracking("SReffects parameters are unphysical.", "Particles are being dumped.");
    return;
  }
#endif
  /* compute P/Po change per turn due to SR losses at the present momentum */
  Ddelta = SReffects.DdeltaRef * gammaRatio * gamma2Ratio;

  if (!lossOnly) {
    /* RMS values for random added components */
    if (twiss->betax <= 0 || twiss->betay <= 0)
      bombElegant("Twiss parameters invalid in track_SReffects", NULL);
    /* damping rates less RF contributions */
    Fx = 1 + (SReffects.Jx - 1) * Ddelta;
    Fy = 1 + (SReffects.Jy - 1) * Ddelta;
    Fdelta = 1 + SReffects.Jdelta * Ddelta;
    if (SReffects.qExcite) {
      Srdelta = sqrt(1 - sqr(Fdelta)) * SReffects.SdeltaRef * gammaRatio;
      Srxp = sqrt((1 - sqr(1 + SReffects.Jx * Ddelta)) * SReffects.exRef * gamma2Ratio / twiss->betax);
      Sryp = sqrt((1 - sqr(1 + SReffects.Jy * Ddelta)) * SReffects.eyRef * gamma2Ratio / twiss->betay);
    } else {
      Srdelta = Srxp = Sryp = 0;
    }
  } else {
    Fx = Fy = Fdelta = 1;
    Srdelta = Srxp = Sryp = 0;
  }

  /* damping rates less RF contributions */
  if (!SReffects.damp)
    Fx = Fy = Fdelta = 1;
  if (!SReffects.loss)
    Ddelta = 0;

  if (SReffects.fraction != 1) {
    /* scale with fraction of effect */
    Fx = 1 + (Fx - 1) * SReffects.fraction;
    Fy = 1 + (Fy - 1) * SReffects.fraction;
    Fdelta = 1 + (Fdelta - 1) * SReffects.fraction;
    Ddelta *= SReffects.fraction;
    Srdelta *= sqrt(SReffects.fraction);
    Srxp *= sqrt(SReffects.fraction);
    Sryp *= sqrt(SReffects.fraction);
  }

  if (first) {
    /*
        printf("Damping/QE constants:\nFx = %e, Fy = %e, Fd = %e\nSrxp = %e, Sryp = %e, Srd = %e\n",
               Fx, Fy, Fdelta, Srxp, Sryp, Srdelta);
        fflush(stdout);
        printf("Twiss parameters: betax = %e, etapx = %e, betay = %e\n",
               twiss->betax, twiss->etapx, twiss->betay);
        fflush(stdout);
*/
  }

  if (first)
    first = 0;

  cutoff = SReffects.cutoff;
  if (active) {
#ifdef HAVE_GPU
    if (getElementOnGpu()) {
      GPU_SREFFECTS_DATA gpuData;
      memset(&gpuData, 0, sizeof(gpuData));
      gpuData.lossOnly = lossOnly ? 1 : 0;
      gpuData.includeOffsets = SReffects.includeOffsets ? 1 : 0;
      gpuData.Fx = Fx;
      gpuData.Fy = Fy;
      gpuData.Fdelta = Fdelta;
      gpuData.Ddelta = Ddelta;
      gpuData.pCentral = Po;
      gpuData.etax = twiss->etax;
      gpuData.etapx = twiss->etapx;
      gpuData.etay = twiss->etay;
      gpuData.etapy = twiss->etapy;
      gpu_track_sreffects(np, &gpuData);
#  ifdef GPU_VERIFY
      startCpuTimer();
      gpuTracked = 1;
#  else
      return;
#  endif
    }
#endif
    if (!lossOnly) {
      for (ip = 0; ip < np; ip++) {
        part = coord[ip];
        xpEta = part[5] * twiss->etapx;
        part[1] = (part[1] - xpEta) * Fx + Srxp * gauss_rn_lim(0.0, 1.0, cutoff, random_2) + xpEta;
        ypEta = part[5] * twiss->etapy;
        part[3] = (part[3] - ypEta) * Fy + Sryp * gauss_rn_lim(0.0, 1.0, cutoff, random_2) + ypEta;
        P = (1 + part[5]) * Po;
        beta = P / sqrt(sqr(P) + 1);
        t = part[4] / beta;
        deltaChange = -part[5];
        part[5] = Ddelta + part[5] * Fdelta + Srdelta * gauss_rn_lim(0.0, 1.0, cutoff, random_2);
        deltaChange += part[5];
        if (SReffects.includeOffsets) {
          /* This is required to keep the beam at the same distance from the off-momentum closed orbit,
             * to avoid additional quantum excitation.  It isn't needed if the rf cavity is right next door
             * and is set properly.
             */
          part[0] += twiss->etax * deltaChange;
          part[1] += twiss->etapx * deltaChange;
          part[2] += twiss->etay * deltaChange;
          part[3] += twiss->etapy * deltaChange;
        }
        P = (1 + part[5]) * Po;
        beta = P / sqrt(sqr(P) + 1);
        part[4] = t * beta;
      }
    } else {
      for (ip = 0; ip < np; ip++) {
        part = coord[ip];
        P = (1 + part[5]) * Po;
        beta = P / sqrt(sqr(P) + 1);
        t = part[4] / beta;
        part[5] += Ddelta;
        P = (1 + part[5]) * Po;
        beta = P / sqrt(sqr(P) + 1);
        part[4] = t * beta;
      }
    }
  }
#if defined(HAVE_GPU) && defined(GPU_VERIFY)
  if (gpuTracked)
    compareGpuCpu(np, "track_SReffects");
#endif
}

VMATRIX *srEffectsMatrix(SREFFECTS *SReffects) {
  double Ddelta;
  VMATRIX *M1;
  long i;

  Ddelta = SReffects->DdeltaRef;

  if (!SReffects->loss)
    Ddelta = 0;

  if (SReffects->fraction != 1) {
    /* scale with fraction of effect */
    Ddelta *= SReffects->fraction;
  }

  M1 = tmalloc(sizeof(*M1));
  initialize_matrices(M1, M1->order = 1);
  for (i = 0; i < 6; i++)
    M1->R[i][i] = 1;
  M1->C[5] = Ddelta;
  return M1;
}
