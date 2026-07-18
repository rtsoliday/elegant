/* GWigSymplecticPass.c for
   elegant
*/

/*
 *---------------------------------------------------------------------------
 * Modification Log:
 * -----------------
 * .03  2004-11-01     M. Borland, APS, borland@aps.anl.gov
 *                              Converted from AT to elegant.
 *
 * .02  2003-06-18     J. Li, jing@fel.duke.edu
 *				Cleanup the code
 *
 * .01  2003-04-20     YK Wu, wu@fel.duke.edu
 *				GWiggler interface
 *
 *---------------------------------------------------------------------------
 *  Accelerator Physics Group, Duke FEL Lab, www.fel.duke.edu  
 */

#include <stdlib.h>
#include <math.h>
#include "gwig.h"
#ifdef HAVE_GPU
#  include "gpu/gpu_base.h"
#  include "gpu/gpu_cwiggler.h"
#endif

/******************************************************************************/
/* PHYSICS SECTION ************************************************************/

void UpdateCWigglerFields(CWIGGLER *cwiggler) {
  /* Default to the user-provided values */
  cwiggler->BMaxToUse = cwiggler->BMax;
  cwiggler->ByMaxToUse = cwiggler->ByMax;
  cwiggler->BxMaxToUse = cwiggler->BxMax;
  if (cwiggler->gap<=0)
    return;
  cwiggler->BMaxToUse = computeUndulatorFieldFromModel(cwiggler->gap, cwiggler->jFraction, &(cwiggler->C[0]), cwiggler->length,
							2*cwiggler->periods, cwiggler->model);
  cwiggler->BxMaxToUse = cwiggler->ByMaxToUse = 0;
}
  
void GWigInit(struct gwig *Wig,
              double Ltot,         /* total length of wiggler */
              double Lw,           /* wiggler period (m) */
              double Bmax,         /* peak magnetic field (Tesla) */
              double BHmax,        /* peak magnetic field (Tesla) for H wiggler expansion, ignored if Bmax is nonzero */
              double BVmax,        /* peak magnetic field (Tesla) for V wiggler expansion, ignored if Bmax is nonzero */
              double normGradient, /* normalized gradient, such that fields are multiplied by (1+gradient*x) */
              int Nstep,           /* number of integration steps (per period?) */
              int Nmeth,           /* integration method (2, 4 , or 6 for integration order) */
              int NHharm,          /* number of horizontal harmonics (By) */
              int NVharm,          /* number of vertical harmonics (Bx) */
              int HSplitPole,      /* use split-pole expansion for horizontal harmonics? */
              int VSplitPole,      /* use split-pole expansion for vertical harmonics? */
              double *pBy,         /* data for horizontal harmonics (By) */
                                   /* (harmonic, strength, kx/kw, ky/kw, kz/kw, phase, ...) */
              double *pBx,         /* data for vertical harmonics (Bx) */
              double *zEndPointH,  /* endpoint data for horizontal harmonics (By) */
              double *zEndPointV,  /* endpoint data for vertical harmonics (Bx) */
              double pCentral,     /* central momentum (beta*gamma) */
              long synchRad,       /* classical radiation ? */
              long isr,            /* quantum (incoherent) radiation ? */
              short endFlag0, short endFlag1,
              double BConstant[2]
) {
  double *tmppr;
  int i;
  double kw;

  Wig->Po = pCentral;
  Wig->E0 = pCentral * XMC2;
  Wig->Pmethod = Nmeth;
  Wig->PN = Nstep;
  Wig->Nw = (int)(Ltot / Lw + 0.5); /* In case Lw is slightly inaccurate */
  Wig->NHharm = NHharm;
  Wig->NVharm = NVharm;
  Wig->PB0 = Bmax;
  Wig->PB0H = BHmax;
  Wig->PB0V = BVmax;
  Wig->Lw = Lw;
  Wig->srCoef = 0;
  Wig->HSplitPole = HSplitPole;
  Wig->VSplitPole = VSplitPole;
  Wig->zStartH = zEndPointH[0];
  Wig->zEndH = zEndPointH[1];
  Wig->zStartV = zEndPointV[0];
  Wig->zEndV = zEndPointV[1];
  Wig->BConstant[0] = BConstant[0];
  Wig->BConstant[1] = BConstant[1];

  /* Boundaries of first, second, third poles */
  if (endFlag0) {
    Wig->z1 = Wig->zStartH + 1 * Wig->Lw / 4.0 - 0.25 * Wig->Lw / Nstep;
    Wig->z2 = Wig->zStartH + 3 * Wig->Lw / 4.0 - 0.25 * Wig->Lw / Nstep;
    Wig->z3 = Wig->zStartH + 5 * Wig->Lw / 4.0 - 0.25 * Wig->Lw / Nstep;
  } else {
    Wig->z1 = Wig->z2 = Wig->z3 = -DBL_MAX;
    Wig->zStartH = Wig->zStartV = MIN(Wig->zStartH, Wig->zStartV);
  }

  /* Boundaries of last, second-to-last, and third-to-last poles */
  if (endFlag1) {
    /* The 0.25 Lw/N term ensures that rounding error doesn't mess things up in comparisons in GWigPoleFactor() */
    Wig->z4 = Wig->zEndH - 5 * Wig->Lw / 4.0 - 0.25 * Wig->Lw / Nstep;
    Wig->z5 = Wig->zEndH - 3 * Wig->Lw / 4.0 - 0.25 * Wig->Lw / Nstep;
    Wig->z6 = Wig->zEndH - Wig->Lw / 4.0 - 0.25 * Wig->Lw / Nstep;
  } else {
    Wig->z4 = Wig->z5 = Wig->z6 = DBL_MAX;
    Wig->zEndH = Wig->zEndV = MAX(Wig->zEndH, Wig->zEndV);
  }

  if ((Wig->sr = synchRad))
    Wig->srCoef = sqr(particleCharge) * ipow3(pCentral) / (6 * PI * epsilon_o * particleMass * sqr(c_mks));

  Wig->isr = isr;
  Wig->isrCoef = particleRadius * sqrt(55 / (24.0 * sqrt(3)) * ipow5(pCentral) * 137.0359895);

  kw = 2.0e0 * PI / (Wig->Lw);
  Wig->Zw = 0.0;
  Wig->Aw = 0.0;
  tmppr = pBy;
  if (NHharm > WHmax || NVharm > WHmax) {
    printf("*** Error: too many harmonics for CWIGGLER. Maximum is %ld.\n", (long)WHmax);
    exitElegant(1);
  }
  for (i = 0; i < NHharm; i++) {
    tmppr++;
    Wig->HCw[i] = 0.0;
    Wig->HCw_raw[i] = *tmppr;

    tmppr++;
    Wig->Hkx[i] = (*tmppr) * kw;

    tmppr++;
    Wig->Hky[i] = (*tmppr) * kw;

    tmppr++;
    Wig->Hkz[i] = (*tmppr) * kw;

    tmppr++;
    Wig->Htz[i] = *tmppr;

    tmppr++;
  }

  tmppr = pBx;
  for (i = 0; i < NVharm; i++) {
    tmppr++;
    Wig->VCw[i] = 0.0;
    Wig->VCw_raw[i] = *tmppr;

    tmppr++;
    Wig->Vkx[i] = (*tmppr) * kw;

    tmppr++;
    Wig->Vky[i] = (*tmppr) * kw;

    tmppr++;
    Wig->Vkz[i] = (*tmppr) * kw;

    tmppr++;
    Wig->Vtz[i] = *tmppr;

    tmppr++;
  }

  for (i = NHharm; i < WHmax; i++) {
    Wig->HCw[i] = 0.0;
    Wig->HCw_raw[i] = 0.0;
    Wig->Hkx[i] = 0.0;
    Wig->Hky[i] = 0.0;
    Wig->Hkz[i] = 0.0;
    Wig->Htz[i] = 0.0;
  }
  for (i = NVharm; i < WHmax; i++) {
    Wig->VCw[i] = 0.0;
    Wig->VCw_raw[i] = 0.0;
    Wig->Vkx[i] = 0.0;
    Wig->Vky[i] = 0.0;
    Wig->Vkz[i] = 0.0;
    Wig->Vtz[i] = 0.0;
  }
}

#define second 2
#define fourth 4

#ifdef HAVE_GPU
static long GWigTrackIdealSinusoidalOnGpu(struct gwig *Wig,
                                           long numParticles) {
  GPU_CWIGGLER_DATA data;
  double beta0, gamma0, kw;

  if (!Wig || !Wig->cwiggler || numParticles <= 0 ||
      Wig->NHharm < 0 || Wig->NHharm > 1 ||
      Wig->NVharm < 0 || Wig->NVharm > 1)
    return 0;
  kw = 2 * PI / Wig->Lw;
  if ((Wig->NHharm &&
       (Wig->HCw_raw[0] != 1 || Wig->Hkx[0] != 0 ||
        Wig->Hky[0] != kw || Wig->Hkz[0] != kw)) ||
      (Wig->NVharm &&
       (Wig->VCw_raw[0] != 1 || Wig->Vkx[0] != kw ||
        Wig->Vky[0] != 0 || Wig->Vkz[0] != kw)))
    return 0;

  memset(&data, 0, sizeof(data));
  gamma0 = Wig->E0 / XMC2;
  beta0 = sqrt(1 - 1 / (gamma0 * gamma0));
  data.periods = Wig->Nw;
  data.stepsPerPeriod = Wig->PN;
  data.integrationOrder = Wig->Pmethod;
  data.hasHorizontal = Wig->NHharm == 1;
  data.hasVertical = Wig->NVharm == 1;
  data.synchRad = Wig->sr;
  data.length = Wig->cwiggler->length;
  data.periodLength = Wig->Lw;
  data.kw = kw;
  data.pCentral = Wig->Po;
  data.srCoef = Wig->srCoef;
  if (data.hasHorizontal) {
    double field = Wig->PB0H != 0 ? Wig->PB0H : Wig->PB0;
    data.horizontalField = field;
    data.horizontalCoefficient =
      (q_e / m_e / clight) / (2 * PI) * Wig->Lw * field /
      (gamma0 * beta0);
    data.horizontalPhase = Wig->Htz[0];
  }
  if (data.hasVertical) {
    double field = Wig->PB0V != 0 ? Wig->PB0V : Wig->PB0;
    data.verticalField = field;
    data.verticalCoefficient =
      (q_e / m_e / clight) / (2 * PI) * Wig->Lw * field /
      (gamma0 * beta0);
    data.verticalPhase = Wig->Vtz[0];
  }
  data.zStartHorizontal = Wig->zStartH;
  data.zEndHorizontal = Wig->zEndH;
  data.zStartVertical = Wig->zStartV;
  data.zEndVertical = Wig->zEndV;
  memcpy(data.poleFactor, Wig->cwiggler->poleFactor,
         sizeof(data.poleFactor));
  data.z1 = Wig->z1;
  data.z2 = Wig->z2;
  data.z3 = Wig->z3;
  data.z4 = Wig->z4;
  data.z5 = Wig->z5;
  data.z6 = Wig->z6;
  gpu_track_cwiggler(numParticles, &data);
  return 1;
}
#endif

void GWigSymplecticPass(double **coord, long num_particles, double pCentral,
                        CWIGGLER *cwiggler, double *sigmaDelta2,
                        long singleStep,
                        double *ZwStart) {

  int c;
  double r6[6], denom, ZwEnd, pf;
  static struct gwig Wig;
  MALIGN malign;
  TRACKING_CONTEXT tContext;
#if defined(HAVE_GPU) && defined(GPU_VERIFY)
  long gpuTracked = 0;
#endif

  getTrackingContext(&tContext);

  if (!singleStep || (singleStep && ZwStart && *ZwStart == 0)) {
    InitializeCWiggler(cwiggler, tContext.elementName);
  }

  if (cwiggler->stepsPerPeriod%4!=0) {
    char buffer[100];
    long spp;
    spp = cwiggler->stepsPerPeriod;
    cwiggler->stepsPerPeriod = (cwiggler->stepsPerPeriod/4+1)*4;
    snprintf(buffer, 100, "Changed from %ld to %ld\n", spp, cwiggler->stepsPerPeriod);
    printWarningForTracking("CWIGGLER STEPS_PER_PERIOD must be 4*integer", buffer);
  }

  UpdateCWigglerFields(cwiggler);
  
  GWigInit(&Wig, cwiggler->length, cwiggler->length / cwiggler->periods,
           cwiggler->BMaxToUse, cwiggler->ByMaxToUse, cwiggler->BxMaxToUse,
           cwiggler->tgu ? cwiggler->tguGradient : 0,
           cwiggler->stepsPerPeriod,
           cwiggler->integrationOrder,
           cwiggler->ByHarmonics, cwiggler->BxHarmonics,
           cwiggler->BySplitPole, cwiggler->BxSplitPole,
           cwiggler->ByData, cwiggler->BxData,
           cwiggler->zEndPointH, cwiggler->zEndPointV,
           pCentral, 
           cwiggler->sr, cwiggler->isr && (num_particles > 1 || cwiggler->isr1Particle),
           cwiggler->endFlag[0], cwiggler->endFlag[1], &cwiggler->BConstant[0]);

  Wig.cwiggler = cwiggler;

#ifdef HAVE_GPU
  if (getElementOnGpu()) {
    if (!singleStep && !ZwStart && !sigmaDelta2 &&
        GWigTrackIdealSinusoidalOnGpu(&Wig, num_particles)) {
#  ifdef GPU_VERIFY
      startCpuTimer();
      gpuTracked = 1;
#  else
      return;
#  endif
    } else {
      coord = forceParticlesToCpu("unsupported CWIGGLER tracking call");
    }
  }
#endif

  if (ZwStart && (*ZwStart - cwiggler->length) > 1e-6 * cwiggler->length)
    bombElegant("ZwStart>cwiggler->length  This is a bug!", NULL);
  if (ZwStart)
    ZwEnd = *ZwStart + (singleStep ? cwiggler->length / (cwiggler->periods * cwiggler->stepsPerPeriod) : cwiggler->length);
  else
    ZwEnd = cwiggler->length;

  if (cwiggler->tilt)
    rotateBeamCoordinatesForMisalignment(coord, num_particles, cwiggler->tilt);
  if (cwiggler->dx || cwiggler->dy || cwiggler->dz) {
    memset(&malign, 0, sizeof(malign));
    malign.dx = -cwiggler->dx;
    malign.dy = -cwiggler->dy;
    malign.dz = cwiggler->dz;
    malign.startPID = malign.endPID = -1;
    offset_beam(coord, num_particles, &malign, pCentral);
  }

  if (cwiggler->fieldOutputInitialized) {
    if (!SDDS_StartPage(cwiggler->SDDSFieldOutput, 1000)) {
      printf("*** Error: unable to start SDDS page for CWIGGLER field output\n");
      SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    cwiggler->fieldOutputRow = 0;
    cwiggler->fieldOutputRows = 1000;
  }

  for (c = 0; c < num_particles; c++) {
    /* convert from (x, x', y, y', s, delta) to Canonical coordinates 
     * (x, qx, y, qy, delta, s) 
     * d =  sqrt(1+sqr(xp)+sqr(yp))
     * qx = (1+delta)*xp/d
     * qy = (1+delta)*yp/d
     */
    r6[0] = coord[c][0];
    r6[2] = coord[c][2];
    r6[1] = (1 + coord[c][5]) * coord[c][1] / (denom = sqrt(1 + sqr(coord[c][1]) + sqr(coord[c][3])));
    r6[3] = (1 + coord[c][5]) * coord[c][3] / denom;
    /* For some reason, they swap the order here */
    r6[4] = coord[c][5];
    r6[5] = coord[c][4];
    if (ZwStart) {
      Wig.Zw = *ZwStart;
      if (singleStep && Wig.Zw != 0) {
        double ax, ay, axpy, aypx;
        GWigAx(&Wig, r6, &ax, &axpy, pf = GWigPoleFactor(&Wig, *ZwStart));
        GWigAy(&Wig, r6, &ay, &aypx, pf);
        r6[1] += ax;
        r6[3] += ay;
      }
    } else
      Wig.Zw = 0;

    /* Track through the wiggler */
    switch (cwiggler->integrationOrder) {
    case second:
      GWigPass_2nd(&Wig, r6, sigmaDelta2, singleStep);
      break;
    case fourth:
      GWigPass_4th(&Wig, r6, sigmaDelta2, singleStep);
      break;
    default:
      printf("Error: Invalid method integration order for CWIGGLER (use 2 or 4)\n");
      exit(1);
      break;
    }

    if (singleStep) {
      /* convert back to elegant coordinates (special code for interior of device to account for vector potential) */
      double ax, ay, axpy, aypx;
      GWigAx(&Wig, r6, &ax, &axpy, pf = GWigPoleFactor(&Wig, ZwEnd));
      GWigAy(&Wig, r6, &ay, &aypx, pf);
      coord[c][0] = r6[0];
      coord[c][2] = r6[2];
      coord[c][5] = r6[4];
      coord[c][4] = r6[5];
      /* d = sqrt(sqr(1+delta)-sqr(qx)-sqr(qy))
       * xp = qx/d, yp=qy/d
       */
      r6[1] -= ax;
      r6[3] -= ay;
      denom = sqrt(sqr(1 + coord[c][5]) - sqr(r6[1]) - sqr(r6[3]));
      coord[c][1] = r6[1] / denom;
      coord[c][3] = r6[3] / denom;
    } else {
      /* convert back to elegant coordinates */
      coord[c][0] = r6[0];
      coord[c][2] = r6[2];
      coord[c][5] = r6[4];
      coord[c][4] = r6[5];
      /* d = sqrt(sqr(1+delta)-sqr(qx)-sqr(qy))
       * xp = qx/d, yp=qy/d
       */
      denom = sqrt(sqr(1 + coord[c][5]) - sqr(r6[1]) - sqr(r6[3]));
      if (Wig.BConstant[0] || Wig.BConstant[1]) {
        /* subtract off vector-potential contribution to the momentum for the constant field component. 
         * we don't need this for the wiggler terms because they are assumed to integrate to zero.
         */
        double ax, ay, gamma0, beta0;
        gamma0 = Wig.E0 / XMC2;
        beta0 = sqrt(1e0 - 1e0 / (gamma0 * gamma0));
        ax = -Wig.BConstant[1]*ZwEnd/(gamma0*beta0)*(q_e/m_e/clight);
        ay = Wig.BConstant[0]*ZwEnd/(gamma0*beta0)*(q_e/m_e/clight);
        r6[1] -= ax;
        r6[3] -= ay;
      }
      coord[c][1] = r6[1] / denom;
      coord[c][3] = r6[3] / denom;
    }
  }

  if (!singleStep && cwiggler->fieldOutputInitialized) {
    if (!SDDS_WritePage(cwiggler->SDDSFieldOutput)) {
      printf("*** Error: unable to write SDDS page for CWIGGLER field output\n");
      SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  }

  if (cwiggler->dx || cwiggler->dy || cwiggler->dz) {
    memset(&malign, 0, sizeof(malign));
    malign.dx = cwiggler->dx;
    malign.dy = cwiggler->dy;
    malign.dz = -cwiggler->dz;
    offset_beam(coord, num_particles, &malign, pCentral);
  }
  if (cwiggler->tilt)
    rotateBeamCoordinatesForMisalignment(coord, num_particles, -cwiggler->tilt);
  if (sigmaDelta2)
    *sigmaDelta2 /= num_particles;
  if (ZwStart)
    *ZwStart += singleStep ? cwiggler->length / (cwiggler->periods * cwiggler->stepsPerPeriod) : cwiggler->length;
#if defined(HAVE_GPU) && defined(GPU_VERIFY)
  if (gpuTracked)
    compareGpuCpu(num_particles, "GWigSymplecticPass");
#endif
}

void InitializeCWiggler(CWIGGLER *cwiggler, char *name) {
  double sumCmn[2] = {0, 0};
  long i;

  UpdateCWigglerFields(cwiggler);
  if (cwiggler->BMaxToUse && (cwiggler->BxMaxToUse || cwiggler->ByMaxToUse)) {
    printf("*** Error: Non-zero BMAX for CWIGGLER when BXMAX or BYMAX also non-zero\n");
    exitElegant(1);
  }
  if (!cwiggler->initialized) {
    if (cwiggler->fieldOutput) {
      if (!cwiggler->SDDSFieldOutput)
        cwiggler->SDDSFieldOutput = tmalloc(sizeof(*(cwiggler->SDDSFieldOutput)));
      if (!SDDS_InitializeOutputElegant(cwiggler->SDDSFieldOutput, SDDS_BINARY, 0, NULL, NULL, cwiggler->fieldOutput) ||
          !SDDS_DefineSimpleColumn(cwiggler->SDDSFieldOutput, "x", "m", SDDS_FLOAT) ||
          !SDDS_DefineSimpleColumn(cwiggler->SDDSFieldOutput, "y", "m", SDDS_FLOAT) ||
          !SDDS_DefineSimpleColumn(cwiggler->SDDSFieldOutput, "z", "m", SDDS_FLOAT) ||
          !SDDS_DefineSimpleColumn(cwiggler->SDDSFieldOutput, "px", "", SDDS_FLOAT) ||
          !SDDS_DefineSimpleColumn(cwiggler->SDDSFieldOutput, "py", "", SDDS_FLOAT) ||
          !SDDS_DefineSimpleColumn(cwiggler->SDDSFieldOutput, "Bx", "T", SDDS_FLOAT) ||
          !SDDS_DefineSimpleColumn(cwiggler->SDDSFieldOutput, "By", "T", SDDS_FLOAT) ||
          !SDDS_DefineSimpleColumn(cwiggler->SDDSFieldOutput, "PoleFactor", NULL, SDDS_FLOAT) ||
          !SDDS_WriteLayout(cwiggler->SDDSFieldOutput)) {
        printf("*** Error: problem setting up field output file for CWIGGLER\n");
        SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
      cwiggler->fieldOutputInitialized = 1;
    }
    if (cwiggler->sinusoidal) {
      if (cwiggler->BxFile || cwiggler->ByFile) {
        printf("*** Error: CWIGGLER element has SINUSOIDAL=1, but also has filenames\n");
        exitElegant(1);
      }
      cwiggler->BxHarmonics = 0;
      cwiggler->BxData = NULL;
      if (cwiggler->tgu) {
        if (!cwiggler->BMaxToUse && !cwiggler->ByMaxToUse) {
          printf("*** Error: CWIGGLER element has TGU=1, but also has B_MAX=%le and BY_MAX=%le\n",
                 cwiggler->BMaxToUse, cwiggler->ByMaxToUse);
          exitElegant(1);
        }
        if (cwiggler->vertical || cwiggler->helical) {
          printf("*** Error: CWIGGLER element has TGU=1, but also has VERTICAL=1 or HELICAL=1\n");
          exitElegant(1);
        }
      }
      if (!cwiggler->vertical || cwiggler->helical) {
        cwiggler->ByHarmonics = 1;
        cwiggler->ByData = tmalloc(sizeof(*(cwiggler->ByData)) * 6);
        cwiggler->ByData[0] = 0; /* row */
        cwiggler->ByData[1] = 1; /* Cmn */
        cwiggler->ByData[2] = 0; /* kx */
        cwiggler->ByData[3] = 1; /* ky */
        cwiggler->ByData[4] = 1; /* kz */
        cwiggler->ByData[5] = 0; /* phase */
      }
      if (cwiggler->vertical || cwiggler->helical) {
        cwiggler->BxHarmonics = 1;
        cwiggler->BxData = tmalloc(sizeof(*(cwiggler->BxData)) * 6);
        cwiggler->BxData[0] = 0; /* row */
        cwiggler->BxData[1] = 1; /* Cmn */
        cwiggler->BxData[2] = 1; /* kx */
        cwiggler->BxData[3] = 0; /* ky */
        cwiggler->BxData[4] = 1; /* kz */
        if (cwiggler->helical)
          cwiggler->BxData[5] = PI / 2; /* phase */
        else
          cwiggler->BxData[5] = 0;
      }
    } else {
      if (cwiggler->tgu) {
        printf("*** Error: CWIGGLER element has TGU=1, but doesn't have SINUSOIDAL=1\n");
        exitElegant(1);
      }
      if (cwiggler->helical) {
        printf("*** Error: CWIGGLER element has HELICAL=1, but doesn't have SINUSOIDAL=1\n");
        exitElegant(1);
      }
      if (cwiggler->vertical) {
        printf("*** Error: CWIGGLER element has VERTICAL=1, but doesn't have SINUSOIDAL=1\n");
        exitElegant(1);
      }
      ReadCWigglerHarmonics(&cwiggler->ByData, &cwiggler->ByHarmonics,
                            cwiggler->ByFile, "By", 0, cwiggler->BySplitPole, cwiggler);
      ReadCWigglerHarmonics(&cwiggler->BxData, &cwiggler->BxHarmonics,
                            cwiggler->BxFile, "Bx", 1, cwiggler->BxSplitPole, cwiggler);
    }
  }

  for (i = 0; i < cwiggler->ByHarmonics; i++)
    sumCmn[1] += cwiggler->ByData[6 * i + 1];
  for (i = 0; i < cwiggler->BxHarmonics; i++)
    sumCmn[0] += cwiggler->BxData[6 * i + 1];
  if (cwiggler->BMaxToUse) {
    cwiggler->BPeak[0] = cwiggler->BMaxToUse * sumCmn[0];
    cwiggler->BPeak[1] = cwiggler->BMaxToUse * sumCmn[1];
  } else {
    cwiggler->BPeak[0] = cwiggler->BxMaxToUse * sumCmn[0];
    cwiggler->BPeak[1] = cwiggler->ByMaxToUse * sumCmn[1];
  }

  if (cwiggler->ByHarmonics) {
    double phase;
    phase = fmod(cwiggler->ByData[5], PIx2);
    if (cwiggler->BMaxToUse == 0 && cwiggler->ByMaxToUse == 0)
      printWarningForTracking("BMAX=0 and BYMAX=0 for CWIGGLER with BY Harmonics", NULL);
    if (phase == 0 || phase == PI || !cwiggler->forceMatched) {
      cwiggler->zEndPointH[0] = 0;
      cwiggler->zEndPointH[1] = cwiggler->length;
    } else if (phase < PI) {
      cwiggler->zEndPointH[0] = (PI - phase) / (PIx2) * (cwiggler->length / cwiggler->periods);
      cwiggler->zEndPointH[1] = cwiggler->length - (cwiggler->length / cwiggler->periods - cwiggler->zEndPointH[0]);
    } else {
      cwiggler->zEndPointH[0] = (PIx2 - phase) / (PIx2) * (cwiggler->length / cwiggler->periods);
      cwiggler->zEndPointH[1] = cwiggler->length - (cwiggler->length / cwiggler->periods - cwiggler->zEndPointH[0]);
    }
    if (phase != 0 && phase != PI && cwiggler->forceMatched)
      printf("Inset endpoints for CWIGGLER %s By Harmonic Data: %le, %le\n",
             name, cwiggler->zEndPointH[0], cwiggler->zEndPointH[1]);
    /* This helps ensure the field is right for the last point even with summation errors for the z position */
    cwiggler->zEndPointH[0] -= (cwiggler->length / cwiggler->periods) / cwiggler->stepsPerPeriod / 16;
    cwiggler->zEndPointH[1] += (cwiggler->length / cwiggler->periods) / cwiggler->stepsPerPeriod / 16;
  }
  if (cwiggler->BxHarmonics) {
    double phase;
    phase = fmod(cwiggler->BxData[5], PIx2);
    if (cwiggler->BMaxToUse == 0 && cwiggler->BxMaxToUse == 0)
      printWarningForTracking("BMAX=0 and BXMAX=0 for CWIGGLER WITH BX Harmonics", NULL);
    if (phase == 0 || phase == PI || !cwiggler->forceMatched) {
      cwiggler->zEndPointV[0] = 0;
      cwiggler->zEndPointV[1] = cwiggler->length;
    } else if (phase < PI) {
      cwiggler->zEndPointV[0] = (PI - phase) / (PIx2) * (cwiggler->length / cwiggler->periods);
      cwiggler->zEndPointV[1] = cwiggler->length - (cwiggler->length / cwiggler->periods - cwiggler->zEndPointV[0]);
    } else {
      cwiggler->zEndPointV[0] = (PIx2 - phase) / (PIx2) * (cwiggler->length / cwiggler->periods);
      cwiggler->zEndPointV[1] = cwiggler->length - (cwiggler->length / cwiggler->periods - cwiggler->zEndPointV[0]);
    }
    if (phase != 0 && phase != PI && cwiggler->forceMatched)
      printf("Inset endpoints for CWIGGLER %s Bx Harmonic Data: %le, %le\n",
             name, cwiggler->zEndPointV[0], cwiggler->zEndPointV[1]);
    /* This helps ensure the field is right for the last point even with summation errors for the z position */
    cwiggler->zEndPointV[0] -= (cwiggler->length / cwiggler->periods) / cwiggler->stepsPerPeriod / 16;
    cwiggler->zEndPointV[1] += (cwiggler->length / cwiggler->periods) / cwiggler->stepsPerPeriod / 16;
  }

  cwiggler->initialized = 1;
}

long ReadCWigglerHarmonics(double **BData, long *harmonics, char *file, char *name, long verticalWiggler, long splitPole, CWIGGLER *cwiggler) {
  SDDS_DATASET SDDSin;
  double *Cmn, *kx, *ky, *kz, *phase;
  long row, rows;

  Cmn = kx = ky = kz = phase = NULL;

  if (!file) {
    *harmonics = 0;
    return 0;
  }
  if (!SDDS_InitializeInput(&SDDSin, file) || !SDDS_ReadPage(&SDDSin)) {
    printf("Error: problem initializing file %s\n", file);
    exitElegant(1);
  }
  if (!(rows = SDDS_RowCount(&SDDSin))) {
    printf("Error: no rows in file %s\n", file);
    exitElegant(1);
  }
  if (!(Cmn = SDDS_GetColumnInDoubles(&SDDSin, "Cmn")) ||
      !(kx = SDDS_GetColumnInDoubles(&SDDSin, "KxOverKw")) ||
      !(ky = SDDS_GetColumnInDoubles(&SDDSin, "KyOverKw")) ||
      !(kz = SDDS_GetColumnInDoubles(&SDDSin, "KzOverKw")) ||
      !(phase = SDDS_GetColumnInDoubles(&SDDSin, "Phase"))) {
    printf("Error: problem reading file %s\n", file);
    printf("Check for existence of Cmn, KxOverKw, KyOverKw, KzOverKw, and Phase\n");
    exitElegant(1);
  }

  *harmonics = rows;
  if (!(*BData = calloc(rows * 6, sizeof(**BData)))) {
    printf("ReadCWigglerHarmonics: memory allocation failure (%ld harmonics)\n",
           rows);
    exitElegant(1);
  }

  for (row = 0; row < rows; row++) {
    if (kz[row] <= 0) {
      printf("*** Error: Problem with KzOverKw in %s: value %e is not positive\n", file, kz[row]);
      exitElegant(1);
    }
    if ((!verticalWiggler && !splitPole) || (verticalWiggler && splitPole)) {
      if (ky[row] < kz[row]) {
        printf("*** Error: Problem with KyOverKw<KzOverKw in %s\n", file);
        exitElegant(1);
      }
      if (!fabs(kx[row]) && verticalWiggler) {
        printf("*** Error: KxOverKw = 0 is not supported for vertical split pole wigglers\n");
        exitElegant(1);
      }
      if (fabs(sqrt(sqr(kx[row]) + sqr(kz[row])) / ky[row] - 1) > 1e-6) {
        printf("*** Error: KyOverKw == sqrt(KxOverKw^2+KzOverKw^2) not satisfied to sufficient accuracy (1e-6) in %s\n", file);
        exitElegant(1);
      }
    } else {
      if (kx[row] < kz[row]) {
        printf("*** Error: Problem with KxOverKw<KzOverKw in %s\n", file);
        exitElegant(1);
      }
      if (!fabs(ky[row]) && !verticalWiggler) {
        printf("*** Error: KyOverKw = 0 is not supported for horizontal split pole wigglers\n");
        exitElegant(1);
      }
      if (fabs(sqrt(sqr(ky[row]) + sqr(kz[row])) / kx[row] - 1) > 1e-6) {
        printf("*** Error: KxOverKw == sqrt(KyOverKw^2+KzOverKw^2) not satisfied to sufficient accuracy (1e-6) in %s\n", file);
        exitElegant(1);
      }
    }
    if (PIx2 * kz[row] > 2 * cwiggler->stepsPerPeriod) {
      printf("*** Error: KzOverKw = %le is too large for only %ld steps per period\n",
             kz[row], cwiggler->stepsPerPeriod);
      exitElegant(1);
    }
    (*BData)[row * 6] = row;
    (*BData)[row * 6 + 1] = Cmn[row];
    (*BData)[row * 6 + 2] = kx[row];
    (*BData)[row * 6 + 3] = ky[row];
    (*BData)[row * 6 + 4] = kz[row];
    (*BData)[row * 6 + 5] = phase[row];
  }
  if (phase[0] < 0) {
    printf("Error: Phase value for CWIGGLER first harmonic is negative.  This isn't allowed.\n");
    exitElegant(1);
  }
  if (!SDDS_Terminate(&SDDSin))
    printWarning("Problem terminating CWIGGLER input file", NULL);

  free(Cmn);
  free(kx);
  free(ky);
  free(kz);
  free(phase);

  return rows;
}
