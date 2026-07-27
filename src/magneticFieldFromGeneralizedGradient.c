/* Copyright 2016 by Michael Borland, Ryan Lindberg, and Argonne National Laboratory,
 * all rights reserved.
 */
#include "mdb.h"
#include "SDDS.h"
#include "track.h"
#ifdef HAVE_GPU
#  include "gpu/gpu_base.h"
#  include "gpu/gpu_bggexp.h"
#endif

typedef struct {
  short skew;              /* if non-zero, these are skew terms */
  long nz;                 /* number of z points */
  double dz;               /* z spacing */
  double xCenter, yCenter; /* center of the expansion in magnet coordinate system */
  double xMax, yMax;       /* half-aperture of the field expansion in expansion coordinate system */
  double zMin, zMax;       /* minimum and maximum z values */
  long nm;                 /* number of values of m (angular harmonic) */
  long *m;                 /* value of m */
  double nGradients;       /* number of gradient functions per m */
  /* See M. Venturini and A. Dragt, NIM A 427 (1999) 387-392, Eq. 9. */
  double ***Cmn;     /* generalized gradient: Cnms[im][in][iz] */
  double ***dCmn_dz; /* z derivative of generalized gradient */
} STORED_BGGEXP_DATA;

static STORED_BGGEXP_DATA *storedBGGExpData = NULL;
static long nBGGExpDataSets = 0;
static htab *fileHashTable = NULL;

long computeGGEMagneticFields(double *Bx, double *By, double *Bz,
                              double x, double y, long iz, BGGEXP *bgg, STORED_BGGEXP_DATA *bggData[2], long igLimit[2]);

long computeGGEVectorPotential(double *Ax, double *dAx_dx, double *dAx_dy,
                               double *Ay, double *dAy_dx, double *dAy_dy,
                               double *dAz_dx, double *dAz_dy,
                               double x, double y, long iz,
                               BGGEXP *bgg, STORED_BGGEXP_DATA *bggData[2], long igLimit[2]);

#ifdef HAVE_GPU
static long trackBGGExpansionOnGpu(long np, BGGEXP *bgg, double pCentral,
                                   STORED_BGGEXP_DATA *bggData[2],
                                   long igLimit[2]) {
  GPU_BGGEXP_DATA data;
  const double **Cmn = NULL, **dCmnDz = NULL;
  double *coefficient = NULL, *multipoleFactor = NULL;
  int *m = NULL, *gradient = NULL, *radialPower = NULL;
  long im, ig, iterm, termCount;
  unsigned long long tableSignature = 1469598103934665603ULL;

  if (!bggData[0] || bggData[1] || igLimit[0] <= 0 ||
      bggData[0]->nz <= 1 ||
      bggData[0]->xMax > 0 || bggData[0]->yMax > 0)
    return 0;

  termCount = bggData[0]->nm * igLimit[0];
  if (termCount <= 0)
    return 0;
  m = malloc(sizeof(*m) * termCount);
  gradient = malloc(sizeof(*gradient) * termCount);
  radialPower = malloc(sizeof(*radialPower) * termCount);
  coefficient = malloc(sizeof(*coefficient) * termCount);
  multipoleFactor = malloc(sizeof(*multipoleFactor) * termCount);
  Cmn = malloc(sizeof(*Cmn) * termCount);
  dCmnDz = malloc(sizeof(*dCmnDz) * termCount);
  if (!m || !gradient || !radialPower || !coefficient ||
      !multipoleFactor || !Cmn || !dCmnDz)
    bombElegant("memory allocation failure preparing BGGEXP CUDA data", NULL);

  for (im = iterm = 0; im < bggData[0]->nm; im++) {
    long harmonic = bggData[0]->m[im];
    double mfact = dfactorial(harmonic);
    double harmonicFactor =
      harmonic < 5 ? bgg->multipoleFactor[harmonic] : 1;

    if (bgg->mMaximum > 0 && harmonic > bgg->mMaximum)
      continue;
    for (ig = 0; ig < igLimit[0]; ig++, iterm++) {
      m[iterm] = harmonic;
      gradient[iterm] = ig;
      radialPower[iterm] = 2 * ig + harmonic - 1;
      tableSignature ^= (unsigned long long)(unsigned int)harmonic;
      tableSignature *= 1099511628211ULL;
      tableSignature ^= (unsigned long long)(unsigned int)ig;
      tableSignature *= 1099511628211ULL;
      coefficient[iterm] =
        ipow(-1, ig) * mfact /
        (ipow(2, 2 * ig) * factorial(ig) * factorial(ig + harmonic));
      multipoleFactor[iterm] = harmonicFactor;
      Cmn[iterm] = bggData[0]->Cmn[im][ig];
      dCmnDz[iterm] = bggData[0]->dCmn_dz[im][ig];
    }
  }
  termCount = iterm;
  if (!termCount) {
    free(m);
    free(gradient);
    free(radialPower);
    free(coefficient);
    free(multipoleFactor);
    free(Cmn);
    free(dCmnDz);
    return 0;
  }

  memset(&data, 0, sizeof(data));
  data.tableOwner = bggData[0];
  data.tableSignature = tableSignature;
  data.nz = bggData[0]->nz;
  data.termCount = termCount;
  data.m = m;
  data.gradient = gradient;
  data.radialPower = radialPower;
  data.coefficient = coefficient;
  data.multipoleFactor = multipoleFactor;
  data.Cmn = Cmn;
  data.dCmnDz = dCmnDz;
  data.dz = bggData[0]->dz;
  data.zMin = bggData[0]->zMin;
  data.zMax = bggData[0]->zMax;
  data.xCenter = bggData[0]->xCenter;
  data.yCenter = bggData[0]->yCenter;
  data.length = bgg->length;
  data.dxExpansion = bgg->dxExpansion;
  data.pCentral = pCentral;
  data.strength = bgg->strength;
  data.Bx = bgg->Bx;
  data.By = bgg->By;
  memcpy(data.BFactor, bgg->BFactor, sizeof(data.BFactor));
  data.particleCharge = particleCharge;
  data.particleRelSign = particleRelSign;
  data.particleMass = particleMass;
  data.cMks = c_mks;

  gpu_track_bggexp(np, &data);

  free(m);
  free(gradient);
  free(radialPower);
  free(coefficient);
  free(multipoleFactor);
  free(Cmn);
  free(dCmnDz);
  return 1;
}
#endif

#define BUFSIZE 16834

long addBGGExpData(char *filename, char *nameFragment, short skew) {
  SDDS_DATASET SDDSin;
  TRACKING_CONTEXT tcontext;
  char buffer[BUFSIZE];
  long im, ic, nc, readCode, nz;
  int32_t m;
  short xCenterPresent, yCenterPresent, xMaxPresent, yMaxPresent;
  long *nstore;

  if (!fileHashTable)
    fileHashTable = hcreate(12);

  if (hfind(fileHashTable, (unsigned char*)filename, strlen(filename))) {
    printf("Using previously-stored BGGEXP data for filename %s\n", filename);
    fflush(stdout);
    im = *((long *)hstuff(fileHashTable));
    printf("Using BGGEXP table %ld for filename %s\n", im, filename);
    fflush(stdout);
    return im;
  }

  getTrackingContext(&tcontext);
  printf("Adding BGGEXP data from file %s for element %s #%ld\n", filename, tcontext.elementName, tcontext.elementOccurrence);
  fflush(stdout);

  if (!SDDS_InitializeInputFromSearchPath(&SDDSin, filename))
    bombElegantVA("Unable to read file %s for BGGEXP %s #%ld\n", filename, tcontext.elementName, tcontext.elementOccurrence);

  if (SDDS_CheckParameter(&SDDSin, "m", NULL, SDDS_ANY_INTEGER_TYPE, stderr) != SDDS_CHECK_OK)
    bombElegantVA("Unable to find integer parameter \"m\" in file %s for BGGEXP %s #%ld\n", filename, tcontext.elementName, tcontext.elementOccurrence);

  xCenterPresent = yCenterPresent = 0;
  if (SDDS_CheckParameter(&SDDSin, "xCenter", "m", SDDS_ANY_FLOATING_TYPE, NULL) == SDDS_CHECK_OK)
    xCenterPresent = 1;
  if (SDDS_CheckParameter(&SDDSin, "yCenter", "m", SDDS_ANY_FLOATING_TYPE, NULL) == SDDS_CHECK_OK)
    yCenterPresent = 1;

  xMaxPresent = yMaxPresent = 0;
  if (SDDS_CheckParameter(&SDDSin, "xMax", "m", SDDS_ANY_FLOATING_TYPE, NULL) == SDDS_CHECK_OK)
    xMaxPresent = 1;
  if (SDDS_CheckParameter(&SDDSin, "yMax", "m", SDDS_ANY_FLOATING_TYPE, NULL) == SDDS_CHECK_OK)
    yMaxPresent = 1;

  /* Check presence of z column */
  if (SDDS_CheckColumn(&SDDSin, "z", "m", SDDS_ANY_FLOATING_TYPE, stderr) != SDDS_CHECK_OK)
    bombElegantVA("Unable to find floating-point column \"z\" with units \"m\" in file %s for BGGEXP %s #%ld\n", filename, tcontext.elementName, tcontext.elementOccurrence);

  /* Check presence of Cnm* columns */
  ic = 0;
  while (1) {
    snprintf(buffer, BUFSIZE, "%s%ld", nameFragment, 2 * ic);
    if (SDDS_CheckColumn(&SDDSin, buffer, NULL, SDDS_ANY_FLOATING_TYPE, NULL) != SDDS_CHECK_OK)
      break;
    ic++;
  }
  if (ic == 0)
    bombElegantVA("Unable to find any floating-point columns %s* in file %s for BGGEXP %s #%ld\n",
                  nameFragment, filename, tcontext.elementName, tcontext.elementOccurrence);
  nc = ic;
  printf("Found %ld %s* columns\n", nc, nameFragment);

  /* Check for presence of matching dCnmXXX/dz columns */
  for (ic = 0; ic < nc; ic++) {
    snprintf(buffer, BUFSIZE, "d%s%ld/dz", nameFragment, 2 * ic);
    if (SDDS_CheckColumn(&SDDSin, buffer, NULL, SDDS_ANY_FLOATING_TYPE, stderr) != SDDS_CHECK_OK)
      break;
  }
  if (ic != nc)
    bombElegantVA("Unable to find matching floating-point columns dCnm*/dz in file %s for BGGEXP %s #%ld\n", filename, tcontext.elementName, tcontext.elementOccurrence);

  if (!(storedBGGExpData = SDDS_Realloc(storedBGGExpData, sizeof(*storedBGGExpData) * (nBGGExpDataSets + 1))))
    bombElegantVA("Memory allocation error reading data from file %s for BGGEXP %s #%ld\n", filename, tcontext.elementName, tcontext.elementOccurrence);

  storedBGGExpData[nBGGExpDataSets].nm = storedBGGExpData[nBGGExpDataSets].nz = 0;
  storedBGGExpData[nBGGExpDataSets].nGradients = nc;
  storedBGGExpData[nBGGExpDataSets].m = NULL;
  storedBGGExpData[nBGGExpDataSets].Cmn = NULL;
  storedBGGExpData[nBGGExpDataSets].dCmn_dz = NULL;

  im = nz = 0;
  storedBGGExpData[nBGGExpDataSets].zMin = DBL_MAX;
  storedBGGExpData[nBGGExpDataSets].zMax = -DBL_MAX;
  storedBGGExpData[nBGGExpDataSets].xCenter = storedBGGExpData[nBGGExpDataSets].yCenter = 0;
  storedBGGExpData[nBGGExpDataSets].xMax = storedBGGExpData[nBGGExpDataSets].yMax = -1;
  while ((readCode = SDDS_ReadPage(&SDDSin)) > 0) {
    if (!SDDS_GetParameterAsLong(&SDDSin, "m", &m) || (m < 1 && !skew) || (m < 0 && skew))
      bombElegantVA("Problem with value of m (m<%ld) for page %ld of file %s for BGGEXP %s #%ld\n",
                    (skew ? 0 : 1), readCode, filename, tcontext.elementName, tcontext.elementOccurrence);
    if (readCode == 1) {
      long iz;
      double dz0, dz, *z, zMin, zMax;
      if ((xCenterPresent && !SDDS_GetParameterAsDouble(&SDDSin, "xCenter", &(storedBGGExpData[nBGGExpDataSets].xCenter))) ||
          (yCenterPresent && !SDDS_GetParameterAsDouble(&SDDSin, "yCenter", &(storedBGGExpData[nBGGExpDataSets].yCenter))))
        bombElegantVA("Problem getting xCenter or yCenter values from file %s for BGGEXP %s #%ld\n",
                      filename, tcontext.elementName, tcontext.elementOccurrence);
      if ((xMaxPresent && !SDDS_GetParameterAsDouble(&SDDSin, "xMax", &(storedBGGExpData[nBGGExpDataSets].xMax))) ||
          (yMaxPresent && !SDDS_GetParameterAsDouble(&SDDSin, "yMax", &(storedBGGExpData[nBGGExpDataSets].yMax))))
        bombElegantVA("Problem getting xMax or yMax values from file %s for BGGEXP %s #%ld\n",
                      filename, tcontext.elementName, tcontext.elementOccurrence);
      if ((nz = SDDS_RowCount(&SDDSin)) <= 1)
        bombElegantVA("Too few z values in file %s for BGGEXP %s #%ld\n", filename, tcontext.elementName, tcontext.elementOccurrence);
      if (!(z = SDDS_GetColumnInDoubles(&SDDSin, "z")))
        bombElegantVA("Problem reading column z from %s for BGGEXP %s #%ld\n", buffer, filename, tcontext.elementName, tcontext.elementOccurrence);
      find_min_max(&zMin, &zMax, z, nz);
      if (zMin < storedBGGExpData[nBGGExpDataSets].zMin)
        storedBGGExpData[nBGGExpDataSets].zMin = zMin;
      if (zMax > storedBGGExpData[nBGGExpDataSets].zMax)
        storedBGGExpData[nBGGExpDataSets].zMax = zMax;
      dz0 = z[1] - z[0];
      for (iz = 1; iz < nz; iz++) {
        dz = z[iz] - z[iz - 1];
        if (dz <= 0 || fabs(dz0 / dz - 1) > 1e-6)
          bombElegantVA("Data not uniformly and monotonically increasing in z column from %s for BGGEXP %s #%ld\n", filename, tcontext.elementName, tcontext.elementOccurrence);
      }
      free(z);
      storedBGGExpData[nBGGExpDataSets].dz = dz0;
    } else {
      if (nz != SDDS_RowCount(&SDDSin))
        bombElegantVA("Inconsistent number of z values in file %s for BGGEXP %s #%ld\n", filename, tcontext.elementName, tcontext.elementOccurrence);
    }
    if (!(storedBGGExpData[nBGGExpDataSets].m = SDDS_Realloc(storedBGGExpData[nBGGExpDataSets].m,
                                                             sizeof(*storedBGGExpData[nBGGExpDataSets].m) * (im + 1))) ||
        !(storedBGGExpData[nBGGExpDataSets].Cmn = SDDS_Realloc(storedBGGExpData[nBGGExpDataSets].Cmn,
                                                               sizeof(*storedBGGExpData[nBGGExpDataSets].Cmn) * (im + 1))) ||
        !(storedBGGExpData[nBGGExpDataSets].dCmn_dz = SDDS_Realloc(storedBGGExpData[nBGGExpDataSets].dCmn_dz,
                                                                   sizeof(*storedBGGExpData[nBGGExpDataSets].dCmn_dz) * (im + 1))))
      bombElegantVA("Memory allocation failure (1) loading data from file %s for BGGEXP %s #%ld\n", filename, tcontext.elementName, tcontext.elementOccurrence);
    storedBGGExpData[nBGGExpDataSets].Cmn[im] = NULL;
    storedBGGExpData[nBGGExpDataSets].dCmn_dz[im] = NULL;
    if (!(storedBGGExpData[nBGGExpDataSets].Cmn[im] = malloc(sizeof(*storedBGGExpData[nBGGExpDataSets].Cmn[im]) * nc)) ||
        !(storedBGGExpData[nBGGExpDataSets].dCmn_dz[im] = malloc(sizeof(*storedBGGExpData[nBGGExpDataSets].dCmn_dz[im]) * nc)))
      bombElegantVA("Memory allocation failure (2) loading data from file %s for BGGEXP %s #%ld\n", filename, tcontext.elementName, tcontext.elementOccurrence);

    for (ic = 0; ic < nc; ic++) {
      snprintf(buffer, BUFSIZE, "%s%ld", nameFragment, 2 * ic);
      if (!(storedBGGExpData[nBGGExpDataSets].Cmn[im][ic] = SDDS_GetColumnInDoubles(&SDDSin, buffer))) {
        SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
        bombElegantVA("Problem reading column %s from %s for BGGEXP %s #%ld\n", buffer, filename, tcontext.elementName, tcontext.elementOccurrence);
      }
      snprintf(buffer, BUFSIZE, "d%s%ld/dz", nameFragment, 2 * ic);
      if (!(storedBGGExpData[nBGGExpDataSets].dCmn_dz[im][ic] = SDDS_GetColumnInDoubles(&SDDSin, buffer))) {
        SDDS_PrintErrors(stdout, SDDS_VERBOSE_PrintErrors);
        bombElegantVA("Problem reading column %s from %s for BGGEXP %s #%ld\n", buffer, filename, tcontext.elementName, tcontext.elementOccurrence);
      }
    }

    storedBGGExpData[nBGGExpDataSets].m[im] = m;
    im++;
  }
  SDDS_Terminate(&SDDSin);

  storedBGGExpData[nBGGExpDataSets].nm = im;
  storedBGGExpData[nBGGExpDataSets].nz = nz;

  nstore = tmalloc(sizeof(*nstore));
  *nstore = nBGGExpDataSets;
  hadd(fileHashTable, (unsigned char *)filename, strlen(filename), (void *)nstore);

  printf("Done adding BGGEXP data from file %s\n", filename);
  fflush(stdout);

  return nBGGExpDataSets++;
}

long trackBGGExpansion(double **part, long np, BGGEXP *bgg, double pCentral, double **accepted, double *sigmaDelta2) {
  long ip, iz, irow, igLimit[2], nz;
  STORED_BGGEXP_DATA *bggData[2];
  double ds, dz, x, y, xp, yp, delta, s, phi, denom;
  double step, length;
  TRACKING_CONTEXT tcontext;
  double radCoef = 0, isrCoef = 0;
  double zMin, zMax, xVertex, zVertex, xEntry, zEntry, xExit, zExit;
  short isLost;
#if defined(HAVE_GPU) && defined(GPU_VERIFY)
  long gpuTracked = 0;
#endif

#ifdef DEBUG
  static FILE *fpdebug = NULL;
  if (!fpdebug) {
    fpdebug = fopen("bggexp.deb", "w");
    fprintf(fpdebug, "SDDS1\n");
    fprintf(fpdebug, "&column name=particleID type=long units=m &end\n");
    fprintf(fpdebug, "&column name=ds type=double units=m &end\n");
    fprintf(fpdebug, "&column name=x type=double units=m &end\n");
    fprintf(fpdebug, "&column name=y type=double units=m &end\n");
    fprintf(fpdebug, "&column name=z type=double units=m &end\n");
    fprintf(fpdebug, "&column name=Bx type=double units=T &end\n");
    fprintf(fpdebug, "&column name=By type=double units=T &end\n");
    fprintf(fpdebug, "&column name=Bz type=double units=T &end\n");
    fprintf(fpdebug, "&column name=px type=double &end\n");
    fprintf(fpdebug, "&column name=py type=double &end\n");
    fprintf(fpdebug, "&column name=pz type=double &end\n");
    fprintf(fpdebug, "&column name=dpx type=double &end\n");
    fprintf(fpdebug, "&column name=dpy type=double &end\n");
    fprintf(fpdebug, "&column name=dpz type=double &end\n");
    fprintf(fpdebug, "&data mode=ascii no_row_counts=1 &end\n");
  }
#endif
  if (bgg->synchRad) {
    radCoef = ipow4(particleCharge) / (6 * PI * epsilon_o * ipow4(c_mks) * ipow3(particleMass));
    isrCoef = sqrt(55 / (24 * sqrt(3)) * particleRadius * hbar_mks * ipow3(particleCharge)) / sqr(particleMass * c_mks);
  }
  if (sigmaDelta2)
    *sigmaDelta2 = 0;

  getTrackingContext(&tcontext);

  if (!bgg->initialized) {
    /* char *outputFile; */
    bgg->initialized = 1;
    bgg->dataIndex[0] = bgg->dataIndex[1] = -1;
    if (bgg->filename && strlen(bgg->filename))
      bgg->dataIndex[0] = addBGGExpData(bgg->filename, "Cnm", 0);
    if (bgg->normalFilename && strlen(bgg->normalFilename)) {
      if (bgg->dataIndex[0] != -1)
        bombElegantVA("Give FILENAME or NORMAL_FILENAME for BGGEXP, not both (element %s)", tcontext.elementName);
      bgg->dataIndex[0] = addBGGExpData(bgg->normalFilename, "CnmS", 0);
    }
    if (bgg->skewFilename && strlen(bgg->skewFilename))
      bgg->dataIndex[1] = addBGGExpData(bgg->skewFilename, "CnmC", 1);

    if (bgg->dataIndex[0] != -1 && bgg->dataIndex[1] != -1) {
      if (storedBGGExpData[bgg->dataIndex[0]].xCenter != storedBGGExpData[bgg->dataIndex[1]].xCenter)
        bombElegantVA("Mismatch of xCenter between normal (%le) and skew (%le) data for BGGEXP %s\n",
                      storedBGGExpData[bgg->dataIndex[0]].xCenter, storedBGGExpData[bgg->dataIndex[1]].xCenter,
                      tcontext.elementName);
      if (storedBGGExpData[bgg->dataIndex[0]].yCenter != storedBGGExpData[bgg->dataIndex[1]].yCenter)
        bombElegantVA("Mismatch of yCenter between normal (%le) and skew (%le) data for BGGEXP %s\n",
                      storedBGGExpData[bgg->dataIndex[0]].yCenter, storedBGGExpData[bgg->dataIndex[1]].yCenter,
                      tcontext.elementName);
      if (storedBGGExpData[bgg->dataIndex[0]].xMax != storedBGGExpData[bgg->dataIndex[1]].xMax)
        bombElegantVA("Mismatch of xMax between normal (%le) and skew (%le) data for BGGEXP %s\n",
                      storedBGGExpData[bgg->dataIndex[0]].xMax, storedBGGExpData[bgg->dataIndex[1]].xMax,
                      tcontext.elementName);
      if (storedBGGExpData[bgg->dataIndex[0]].yMax != storedBGGExpData[bgg->dataIndex[1]].yMax)
        bombElegantVA("Mismatch of yCenter between normal (%le) and skew (%le) data for BGGEXP %s\n",
                      storedBGGExpData[bgg->dataIndex[0]].yMax, storedBGGExpData[bgg->dataIndex[1]].yMax,
                      tcontext.elementName);
      if (storedBGGExpData[bgg->dataIndex[0]].dz != storedBGGExpData[bgg->dataIndex[1]].dz)
        bombElegantVA("Mismatch of z spacing between normal (%le) and skew (%le) data for BGGEXP %s\n",
                      storedBGGExpData[bgg->dataIndex[0]].dz, storedBGGExpData[bgg->dataIndex[1]].dz,
                      tcontext.elementName);
      if (storedBGGExpData[bgg->dataIndex[0]].zMin != storedBGGExpData[bgg->dataIndex[1]].zMin)
        bombElegantVA("Mismatch of minimum z value between normal (%le) and skew (%le) data for BGGEXP %s\n",
                      storedBGGExpData[bgg->dataIndex[0]].zMin, storedBGGExpData[bgg->dataIndex[1]].zMin,
                      tcontext.elementName);
      if (storedBGGExpData[bgg->dataIndex[0]].zMax != storedBGGExpData[bgg->dataIndex[1]].zMax)
        bombElegantVA("Mismatch of maximum z value between normal (%le) and skew (%le) data for BGGEXP %s\n",
                      storedBGGExpData[bgg->dataIndex[0]].zMax, storedBGGExpData[bgg->dataIndex[1]].zMax,
                      tcontext.elementName);
    }

#if !USE_MPI
    if (bgg->particleOutputFile && !bgg->SDDSpo) {
      char *filename;
      bgg->SDDSpo = tmalloc(sizeof(*(bgg->SDDSpo)));
      filename = compose_filename(bgg->particleOutputFile, tcontext.rootname);
      if (!SDDS_InitializeOutputElegant(bgg->SDDSpo, SDDS_BINARY, 1,
                                        NULL, NULL, filename) ||
          0 > SDDS_DefineParameter(bgg->SDDSpo, "SVNVersion", NULL, NULL, "SVN version number", NULL, SDDS_STRING, SVN_VERSION) ||
          !SDDS_DefineSimpleParameter(bgg->SDDSpo, "ElementName", NULL, SDDS_STRING) ||
          !SDDS_DefineSimpleParameter(bgg->SDDSpo, "particleID", NULL, SDDS_ULONG64) ||
          !SDDS_DefineSimpleParameter(bgg->SDDSpo, "Strength", NULL, SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameter(bgg->SDDSpo, "dX", NULL, SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameter(bgg->SDDSpo, "dY", NULL, SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameter(bgg->SDDSpo, "dZ", NULL, SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameter(bgg->SDDSpo, "pCentral", "m$be$nc", SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameter(bgg->SDDSpo, "XEntry", "m", SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameter(bgg->SDDSpo, "ZEntry", "m", SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameter(bgg->SDDSpo, "XVertex", "m", SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameter(bgg->SDDSpo, "ZVertex", "m", SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameter(bgg->SDDSpo, "XExit", "m", SDDS_DOUBLE) ||
          !SDDS_DefineSimpleParameter(bgg->SDDSpo, "ZExit", "m", SDDS_DOUBLE) ||
          (bgg->poIndex[0] = SDDS_DefineColumn(bgg->SDDSpo, "x", NULL, "m", NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          (bgg->poIndex[1] = SDDS_DefineColumn(bgg->SDDSpo, "px", NULL, NULL, NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          (bgg->poIndex[2] = SDDS_DefineColumn(bgg->SDDSpo, "y", NULL, "m", NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          (bgg->poIndex[3] = SDDS_DefineColumn(bgg->SDDSpo, "py", NULL, NULL, NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          (bgg->poIndex[4] = SDDS_DefineColumn(bgg->SDDSpo, "z", NULL, "m", NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          (bgg->poIndex[5] = SDDS_DefineColumn(bgg->SDDSpo, "pz", NULL, NULL, NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          (bgg->poIndex[6] = SDDS_DefineColumn(bgg->SDDSpo, "Bx", NULL, "T", NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          (bgg->poIndex[7] = SDDS_DefineColumn(bgg->SDDSpo, "By", NULL, "T", NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          (bgg->poIndex[8] = SDDS_DefineColumn(bgg->SDDSpo, "Bz", NULL, "T", NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          (bgg->poIndex[9] = SDDS_DefineColumn(bgg->SDDSpo, "particleID", NULL, NULL, NULL, NULL, SDDS_ULONG64, 0)) < 0 ||
          (bgg->poIndex[10] = SDDS_DefineColumn(bgg->SDDSpo, "s", NULL, "m", NULL, NULL, SDDS_DOUBLE, 0)) < 0 ||
          !SDDS_WriteLayout(bgg->SDDSpo)) {
        SDDS_SetError("Problem setting up particle output file for BGGEXP");
        SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
      }
      free(filename);
    }
#endif
  }

  bggData[0] = bggData[1] = NULL;
  igLimit[0] = igLimit[1] = -1;
  zMin = zMax = dz = 0;
  nz = 0;
  if (bgg->dataIndex[0] != -1) {
    bggData[0] = storedBGGExpData + bgg->dataIndex[0];
    zMin = bggData[0]->zMin;
    zMax = bggData[0]->zMax;
    nz = bggData[0]->nz;
    dz = bggData[0]->dz;
    igLimit[0] = bggData[0]->nGradients;
    if (bgg->maximum2n >= 0) {
      igLimit[0] = bgg->maximum2n / 2 + 1;
      if (igLimit[0] > bggData[0]->nGradients)
        igLimit[0] = bggData[0]->nGradients;
    }
  }
  if (bgg->dataIndex[1] != -1) {
    bggData[1] = storedBGGExpData + bgg->dataIndex[1];
    zMin = bggData[1]->zMin;
    zMax = bggData[1]->zMax;
    nz = bggData[1]->nz;
    dz = bggData[1]->dz;
    igLimit[1] = bggData[1]->nGradients;
    if (bgg->maximum2n >= 0) {
      igLimit[1] = bgg->maximum2n / 2 + 1;
      if (igLimit[1] > bggData[1]->nGradients)
        igLimit[1] = bggData[1]->nGradients;
    }
  }

  length = bgg->length;

  if (bgg->isBend) {
    xVertex = bgg->xVertex;
    xEntry = bgg->xEntry;
    xExit = bgg->xExit;
    zVertex = bgg->zVertex;
    zEntry = bgg->zEntry;
    zExit = bgg->zExit;
  } else {
    xEntry = xExit = xVertex = 0;
    zVertex = (zMax + zMin) / 2;
    zEntry = zVertex - length / 2;
    zExit = zVertex + length / 2;
    /*
    printf("fieldLength = %le, z:[%le, %le]\n", fieldLength, zMin, zMax);
    printf("vertex: (%le, %le)\n", zVertex, xVertex);
    printf("entry: (%le, %le)\n", zEntry, xEntry);
    printf("exit: (%le, %le)\n", zExit, xExit);
    */
  }

  step = (zMax - zMin) / (nz - 1);
  /*
  if (fabs(step/dz-1)>1e-6) 
    bombElegantVA("Length mismatch for BGGEXP %s #%ld: %le vs %le\nlength=%le m, nz=%ld\n",
                  tcontext.elementName, tcontext.elementOccurrence, step, bggData->dz,
                  bgg->fieldLength, nz);
  */

  /* Do misalignments */
  if (bgg->dx || bgg->dy || bgg->dz)
    offsetBeamCoordinatesForMisalignment(part, np, bgg->dx, bgg->dy, bgg->dz);
  if (bgg->tilt)
    rotateBeamCoordinatesForMisalignment(part, np, bgg->tilt);

  if (bgg->zInterval <= 0)
    bombElegantVA("zInterval %ld is invalid for BGGEXP %s #%ld\n", bgg->zInterval, tcontext.elementName, tcontext.elementOccurrence);
  /* izLast = nz-bgg->zInterval; */

#ifdef HAVE_GPU
  if (getElementOnGpu()) {
    if (trackBGGExpansionOnGpu(np, bgg, pCentral, bggData, igLimit)) {
#  ifdef GPU_VERIFY
      startCpuTimer();
      gpuTracked = 1;
#  else
      return np;
#  endif
    } else {
      part = forceParticlesToCpu("unsupported BGGEXP field table");
    }
  }
#endif

  if (bgg->symplectic) {
    long iImpLoop;
    double xMid, yMid, xNext, yNext, xLoop, yLoop, delta_s;
    double px, py, pxNext, pyNext, pxLoop, pyLoop, ux = 0, uy = 0;
    double delta, phi, denom, scaleA, sin_phi, cos_phi;
    double Ax, dAx_dx, dAx_dy, Ay, dAy_dx, dAy_dy, dAz_dx, dAz_dy;
    double epsImplConverge = 1.e-14 * 1.0e-3;
    double Bx, By, Bz;

    double magnet_s;

    /*
    if (!bggData[0])
      bombElegantVA("No normal data supplied for BGGEXP %s in symplectic mode. Don't know what to do since skew has not been implemented.",
                    tcontext.elementName);
    */

    scaleA = -bgg->strength * particleCharge * particleRelSign / (pCentral * particleMass * c_mks); /** [factor in parentheses of a = (q/p_0)*A] **/
    /* Element body */
#if defined(HAVE_GPU) && defined(_OPENMP) && !defined(DEBUG)
#  pragma omp parallel for num_threads(gpuGetOmpTrackingThreads()) schedule(static) \
    if(gpuOmpTrackingRequested(np) && !bgg->SDDSpo && !bgg->synchRad && \
       !bgg->isr && !sigmaDelta2) \
    firstprivate(ux, uy) \
    private(ip, iz, irow, isLost, ds, x, y, delta, s, phi, denom, \
            iImpLoop, xMid, yMid, xNext, yNext, xLoop, yLoop, delta_s, \
            px, py, pxNext, pyNext, pxLoop, pyLoop, sin_phi, \
            cos_phi, Ax, dAx_dx, dAx_dy, Ay, dAy_dx, dAy_dy, dAz_dx, \
            dAz_dy, Bx, By, Bz, magnet_s)
#endif
    for (ip = 0; ip < np; ip++) {
#if !USE_MPI
      if (bgg->SDDSpo && np < 1000) {
        if (!SDDS_StartPage(bgg->SDDSpo, (bggData[0] ? bggData[0]->nz : bggData[1]->nz) + 1) ||
            !SDDS_SetParameters(bgg->SDDSpo, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                                "particleID", (uint64_t)(part[ip][6]), "pCentral", pCentral,
                                "XVertex", bgg->xVertex, "ZVertex", bgg->zVertex,
                                "XEntry", bgg->xEntry, "ZEntry", bgg->zEntry,
                                "XExit", bgg->xExit, "ZExit", bgg->zExit,
                                "Strength", bgg->strength, "dX", bgg->dx, "dY", bgg->dy, "dZ", bgg->dz,
                                "ElementName", tcontext.elementName,
                                NULL)) {
          SDDS_SetError("Problem setting up particle output page for BGGEXP");
          SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
        }
      }
#endif
      isLost = 0;

      /* Transform to canonical coordinates */
      delta = part[ip][5];
      denom = 1.0 / sqrt(1.0 + part[ip][1] * part[ip][1] + part[ip][3] * part[ip][3]);
      x = part[ip][0];
      px = part[ip][1] * (1.0 + delta) * denom;
      y = part[ip][2];
      py = part[ip][3] * (1.0 + delta) * denom;
      s = part[ip][4];
      delta_s = 0.0;
      Bx = By = Bz = 0;

      /* Compute transformation from canonical accelerator coordinates -> magnet grid if bending magnet */
      denom = (1.0 + delta) * (1.0 + delta) - px * px - py * py;
      denom = 1.0 / sqrt(denom);
      if (bgg->isBend) {
        phi = atan2(xVertex - xEntry, zVertex - zEntry); /* angle of incoming x-plane w.r.t. magnet x */
        sin_phi = sin(phi);
        cos_phi = cos(phi);
        magnet_s = x * sin_phi / (cos_phi - sin_phi * px * denom);
      } else {
        phi = sin_phi = magnet_s = 0;
        cos_phi = 1;
      }

      x = (x + px * magnet_s * denom * denom) / cos_phi;
      y = (y + py * magnet_s * denom * denom);
      px = px * cos_phi + sin_phi / denom;
      delta_s = (1 + delta) * magnet_s * denom;

      x += xEntry - bgg->dxExpansion; /* Shift x according to entrance point & map shift */

      /* Shift according to central point of the expansion, if given in input file */
      if (bggData[0] && (bggData[0]->xCenter || bggData[0]->yCenter)) {
        x -= bggData[0]->xCenter;
        y -= bggData[0]->yCenter;
      } else if (bggData[1] && (bggData[1]->xCenter || bggData[1]->yCenter)) {
        x -= bggData[1]->xCenter;
        y -= bggData[1]->yCenter;
      }

      /* Drift backward from entrance plane to beginning of field map */
      denom = (1.0 + delta) * (1.0 + delta) - px * px - py * py;
      denom = 1.0 / sqrt(denom);
      x -= (zEntry - zMin) * px * denom;
      y -= (zEntry - zMin) * py * denom;
      s -= (zEntry - zMin) * (1.0 + delta) * denom;

      /* Drift particles back 1/2 step from start of field map to prepare for implicit midpoint rule integration */
      denom = (1.0 + delta) * (1.0 + delta) - px * px - py * py;
      denom = 1.0 / sqrt(denom);
      x -= 0.5 * step * bgg->zInterval * px * denom;
      y -= 0.5 * step * bgg->zInterval * py * denom;
      s -= 0.5 * step * bgg->zInterval * (1.0 + delta) * denom;

      /* Integrate through the magnet */
      for (iz = irow = 0; iz < bggData[0]->nz; iz += bgg->zInterval) {
#if !USE_MPI
        if (bgg->SDDSpo && np < 1000 &&
            !SDDS_SetRowValues(bgg->SDDSpo, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, irow++,
                               bgg->poIndex[0], x + bggData[0]->xCenter,
                               bgg->poIndex[1], px * pCentral,
                               bgg->poIndex[2], y,
                               bgg->poIndex[3], py * pCentral,
                               bgg->poIndex[4], (iz - 1) * dz + bggData[0]->zMin,
                               bgg->poIndex[5], sqrt(sqr(pCentral * (1 + delta)) - (sqr(px) + sqr(py)) * sqr(pCentral)),
                               bgg->poIndex[6], Bx,
                               bgg->poIndex[7], By,
                               bgg->poIndex[8], Bz,
                               bgg->poIndex[9], (uint64_t)part[ip][particleIDIndex],
                               bgg->poIndex[10], s, 
                               -1)) {
          SDDS_SetError("Problem setting particle output data for BGGEXP");
          SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
        }
#endif
        /* r = sqrt(sqr(x)+sqr(y)); */
        phi = atan2(y, x);
        cos_phi = cos(phi);
        sin_phi = sin(phi);

        /** Calculate vector potential A and its relevant derivatives from the generalized gradients **/
        computeGGEVectorPotential(&Ax, &dAx_dx, &dAx_dy, &Ay, &dAy_dx, &dAy_dy, &dAz_dx, &dAz_dy,
                                  x, y, iz, bgg, bggData, igLimit);

        /** Start with first order guess for the 'Next' coordinates **/
        ux = px - scaleA * Ax;
        uy = py - scaleA * Ay;
        denom = (1.0 + delta) * (1.0 + delta) - ux * ux - uy * uy;
        denom = 1.0 / sqrt(denom);
        xNext = x + step * bgg->zInterval * ux * denom;
        yNext = y + step * bgg->zInterval * uy * denom;
        pxNext = px + step * bgg->zInterval * scaleA * ((ux * dAx_dx + uy * dAy_dx) * denom + dAz_dx);
        pyNext = py + step * bgg->zInterval * scaleA * ((ux * dAx_dy + uy * dAy_dy) * denom + dAz_dy);
        iImpLoop = 0;
        do {
          /** Convergence determined when updated 'Next' coordinates match previous 'Loop' coords.  **/
          xLoop = xNext;
          yLoop = yNext;
          pxLoop = pxNext;
          pyLoop = pyNext;
          xMid = 0.5 * (x + xLoop);
          yMid = 0.5 * (y + yLoop);
          /* r = sqrt(sqr(xMid)+sqr(yMid)); */
          phi = atan2(yMid, xMid);
          cos_phi = cos(phi);
          sin_phi = sin(phi);

          /** Calculate vector potential A and its relevant derivatives from the generalized gradients **/
          computeGGEVectorPotential(&Ax, &dAx_dx, &dAx_dy, &Ay, &dAy_dx, &dAy_dy, &dAz_dx, &dAz_dy,
                                    xMid, yMid, iz, bgg, bggData, igLimit);

          /** Update coordinates **/
          ux = 0.5 * (px + pxLoop) - scaleA * Ax;
          uy = 0.5 * (py + pyLoop) - scaleA * Ay;
          denom = (1.0 + delta) * (1.0 + delta) - ux * ux - uy * uy;
          denom = 1.0 / sqrt(denom);
          xNext = x + step * bgg->zInterval * ux * denom;
          yNext = y + step * bgg->zInterval * uy * denom;
          pxNext = px + step * bgg->zInterval * scaleA * ((ux * dAx_dx + uy * dAy_dx) * denom + dAz_dx);
          pyNext = py + step * bgg->zInterval * scaleA * ((ux * dAx_dy + uy * dAy_dy) * denom + dAz_dy);
          iImpLoop++;
          if (iImpLoop > 10) {
            /* printf("HERE: daz_dx = %e\t By = %e\n", scaleA*dAz_dx, B[2]); */
            break;
          }
        } while ((fabs(xNext - xLoop) > epsImplConverge) || (fabs(yNext - yLoop) > epsImplConverge) ||
                 (fabs(pxNext - pxLoop) > epsImplConverge) || (fabs(pyNext - pyLoop) > epsImplConverge));
        x = xNext;
        y = yNext;
        px = pxNext;
        py = pyNext;
        delta_s += step * bgg->zInterval * ((1.0 + delta) * denom - 1.0);

        if (bgg->synchRad || bgg->SDDSpo) {
          ds = step * bgg->zInterval * (1.0 + delta) * denom;
          /* compute Bx, By */
          computeGGEMagneticFields(&Bx, &By, &Bz, x, y, iz, bgg, bggData, igLimit);

          if (bgg->synchRad) {
            double deltaTemp, B2, F;
            /* This is only valid for ultra-relatistic particles that radiate a small fraction of its energy in any step */
            /* It only uses Bx, By and ignores opening angle effects ~1/\gamma */
            B2 = sqr(Bx) + sqr(By);
            deltaTemp = delta - radCoef * pCentral * sqr(1.0 + delta) * B2 * ds;
            F = isrCoef * pCentral * sqr(1.0 + delta) * sqrt(ds) * pow(B2, 3. / 4.);
            if (bgg->isr && np != 1)
              deltaTemp += F * gauss_rn_lim(0.0, 1.0, srGaussianLimit, random_2);
            if (sigmaDelta2)
              *sigmaDelta2 += sqr(F);
            px *= (1 + deltaTemp) / (1 + delta);
            py *= (1 + deltaTemp) / (1 + delta);
            ux *= (1 + deltaTemp) / (1 + delta);
            uy *= (1 + deltaTemp) / (1 + delta);
            delta = deltaTemp;
          }
        }

        if (iz != (bggData[0]->nz))
          s += step * bgg->zInterval;

#ifdef DEBUG
        fprintf(fpdebug, "%le %le %le %le %le %le %le %le %i\n",
                iz * dz, x, y, px, py, s, dAz_dy, dAz_dx, iImpLoop);
#endif
      }
      /* Drift particles back 1/2 step to end on last magnet grid point used */
      px = ux; /* Assuming Ax -> 0 */
      py = uy; /* Assuming Ay -> 0 */
      denom = (1.0 + delta) * (1.0 + delta) - px * px - py * py;
      denom = 1.0 / sqrt(denom);
      x -= 0.5 * step * bgg->zInterval * px * denom;
      y -= 0.5 * step * bgg->zInterval * py * denom;
      s -= 0.5 * step * bgg->zInterval * (1.0 + delta) * denom;
      denom = (1.0 + delta) * (1.0 + delta) - px * px - py * py;
      denom = 1.0 / sqrt(denom);
      if (iz < bggData[0]->nz) {
        /* Drift forward */
        x += dz * (bggData[0]->nz - 1 - (iz - 1)) * px * denom;
        y += dz * (bggData[0]->nz - 1 - (iz - 1)) * py * denom;
        s += dz * (bggData[0]->nz - 1 - (iz - 1)) * (1.0 + delta) * denom;
      }

#if !USE_MPI
      if (bgg->SDDSpo && np < 1000) {
        if (!SDDS_SetRowValues(bgg->SDDSpo, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, irow++,
                               bgg->poIndex[0], x + bggData[0]->xCenter,
                               bgg->poIndex[1], px * pCentral,
                               bgg->poIndex[2], y,
                               bgg->poIndex[3], py * pCentral,
                               bgg->poIndex[4], (bggData[0]->nz - 1) * dz + bggData[0]->zMin,
                               bgg->poIndex[5], sqrt(sqr(pCentral * (1 + delta)) - (sqr(px) + sqr(py)) * sqr(pCentral)),
                               bgg->poIndex[6], Bx,
                               bgg->poIndex[7], By,
                               bgg->poIndex[8], Bz,
                               bgg->poIndex[9], (long)part[ip][particleIDIndex],
                               bgg->poIndex[10], s,
                               -1) ||
            !SDDS_WritePage(bgg->SDDSpo)) {
          SDDS_SetError("Problem setting particle output data for BGGEXP");
          SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
        }
      }
#endif

      /* Drift backward from end of field map to exit plane */
      denom = (1.0 + delta) * (1.0 + delta) - px * px - py * py;
      denom = 1.0 / sqrt(denom);
      x -= (zMax - zExit) * px * denom;
      y -= (zMax - zExit) * py * denom;
      s -= (zMax - zExit) * (1.0 + delta) * denom;

      /* Shift according to central point of the expansion, if given in input file */
      if (bggData[0] && (bggData[0]->xCenter || bggData[0]->yCenter)) {
        x += bggData[0]->xCenter;
        y += bggData[0]->yCenter;
      } else if (bggData[1] && (bggData[1]->xCenter || bggData[1]->yCenter)) {
        x += bggData[1]->xCenter;
        y += bggData[1]->yCenter;
      }

      x -= xExit - bgg->dxExpansion; /* Shift x according to exit point & map shift */

      /* Compute transformation from magnet grid -> accelerator coordinates if bending magnet */
      if (bgg->isBend) {
        phi = atan2(xVertex - xExit, zExit - zVertex); /* angle of outgoing x-plane w.r.t. magnet x */
        sin_phi = sin(phi);
        cos_phi = cos(phi);
      } else {
        sin_phi = 0;
        cos_phi = 1;
      }
      denom = (1.0 + delta) * (1.0 + delta) - px * px - py * py;
      denom = 1.0 / sqrt(denom);
      magnet_s = x * sin_phi / (cos_phi - sin_phi * px * denom);

      x = (x + px * magnet_s * denom * denom) / cos_phi;
      y = (y + py * magnet_s * denom * denom);
      px = px * cos_phi + sin_phi / denom;
      delta_s += (1 + delta) * magnet_s * denom;

      /* Update (non-canonical) particle coordinates */
      part[ip][0] = x;
      denom = (1.0 + delta) * (1.0 + delta) - px * px - py * py;
      denom = 1.0 / sqrt(denom);
      part[ip][1] = px * denom;
      part[ip][2] = y;
      part[ip][3] = py * denom;
      part[ip][4] = s + delta_s;
      part[ip][5] = delta;
      if (isLost) {
        /* TODO: do something about lost particles */
      }
    }
  } else {
    /* Non-symplectic */
    double B[3], p[3], B2Max, pErr[3];
    double pOrig;
    double xpTemp, ypTemp, xpNew, ypNew;
    double xTemp, yTemp, xNew, yNew, preFactorDz, deltaTemp;
    double magnet_s;

#if defined(HAVE_GPU) && defined(_OPENMP) && !defined(DEBUG)
#  pragma omp parallel for num_threads(gpuGetOmpTrackingThreads()) schedule(static) \
    if(gpuOmpTrackingRequested(np) && !bgg->SDDSpo && !bgg->synchRad && \
       !bgg->isr && !sigmaDelta2) \
    firstprivate(dz) \
    private(ip, iz, irow, isLost, ds, x, y, xp, yp, delta, s, phi, denom, \
            B, p, B2Max, pErr, pOrig, xpTemp, ypTemp, xpNew, ypNew, \
            xTemp, yTemp, xNew, yNew, preFactorDz, deltaTemp, magnet_s)
#endif
    for (ip = 0; ip < np; ip++) {
      B2Max = 0;
      x = part[ip][0];
      xp = part[ip][1];
      y = part[ip][2];
      yp = part[ip][3];
      s = part[ip][4];
      delta = part[ip][5];

      /* Compute transformation from accelerator coordinates */
      if (bgg->isBend) {
        phi = atan2(xVertex - xEntry, zVertex - zEntry); /* angle of incoming x-plane w.r.t. magnet x */
        magnet_s = x * sin(phi) / (cos(phi) - xp * sin(phi));
        x = (x + xp * magnet_s) / cos(phi);
        y = y + yp * magnet_s;
        s += sqrt(1.0 + xp * xp + yp * yp) * magnet_s;
        yp = yp / (cos(phi) - xp * sin(phi));
        xp = (xp * cos(phi) + sin(phi)) / (cos(phi) - xp * sin(phi));
      } else {
        phi = magnet_s = 0;
      }

      x += xEntry - bgg->dxExpansion; /* Shift x according to entrance point & map shift */

      /* Shift according to central point of the expansion, if given in input file */
      if (bggData[0] && (bggData[0]->xCenter || bggData[0]->yCenter)) {
        x -= bggData[0]->xCenter;
        y -= bggData[0]->yCenter;
      } else if (bggData[1] && (bggData[1]->xCenter || bggData[1]->yCenter)) {
        x -= bggData[1]->xCenter;
        y -= bggData[1]->yCenter;
      }

      /* Drift backward from entrance plane to beginning of field map */
      x -= (zEntry - zMin) * xp;
      y -= (zEntry - zMin) * yp;
      s -= (zEntry - zMin) * sqrt(1.0 + xp * xp + yp * yp);

      /* compute momenta (x, y, z) */
      denom = sqrt(1 + sqr(xp) + sqr(yp));
      pOrig = pCentral * (1 + delta);
      p[2] = pOrig / denom;
      p[0] = xp * p[2];
      p[1] = yp * p[2];
      /* gamma = sqrt(sqr(p[0]) + sqr(p[1]) + sqr(p[2]) + 1); */
      pErr[0] = pErr[1] = pErr[2] = 0;

#if !USE_MPI
      if (bgg->SDDSpo && np < 1000) {
        if (!SDDS_StartPage(bgg->SDDSpo, (bggData[0] ? bggData[0]->nz : bggData[1]->nz) + 1) ||
            !SDDS_SetParameters(bgg->SDDSpo, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                                "particleID", (uint64_t)(part[ip][6]), "pCentral", pCentral,
                                "XVertex", bgg->xVertex, "ZVertex", bgg->zVertex,
                                "XEntry", bgg->xEntry, "ZEntry", bgg->zEntry,
                                "XExit", bgg->xExit, "ZExit", bgg->zExit,
                                "Strength", bgg->strength, "dX", bgg->dx, "dY", bgg->dy, "dZ", bgg->dz,
                                "ElementName", tcontext.elementName,
                                NULL)) {
          SDDS_SetError("Problem setting up particle output page for BGGEXP");
          SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
        }
      }
#endif

      /* Integrate through the magnet */
      B[0] = B[1] = B[2] = 0;
      isLost = 0;
      for (iz = irow = 0; iz < nz - 1 && !isLost; iz += bgg->zInterval) {
        if ((isLost += computeGGEMagneticFields(&B[0], &B[1], &B[2], x, y, iz, bgg, bggData, igLimit)))
          break;
        denom = sqrt(1 + sqr(xp) + sqr(yp));
        p[2] = pCentral * (1 + delta) / denom;
        p[0] = xp * p[2];
        p[1] = yp * p[2];
        /* gamma = sqrt(sqr(p[0]) + sqr(p[1]) + sqr(p[2]) + 1); */
#if !USE_MPI
        if (bgg->SDDSpo && np < 1000 &&
            !SDDS_SetRowValues(bgg->SDDSpo, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, irow++,
                               bgg->poIndex[0], x,
                               bgg->poIndex[1], p[0],
                               bgg->poIndex[2], y,
                               bgg->poIndex[3], p[1],
                               bgg->poIndex[4], iz * dz + bggData[0]->zMin,
                               bgg->poIndex[5], p[2],
                               bgg->poIndex[6], B[0],
                               bgg->poIndex[7], B[1],
                               bgg->poIndex[8], B[2],
                               bgg->poIndex[9], (uint64_t)part[ip][particleIDIndex],
                               bgg->poIndex[10], s,
                               -1)) {
          SDDS_SetError("Problem setting particle output data for BGGEXP");
          SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
        }
#endif
        /* r = sqrt(sqr(x)+sqr(y)); */
        /* phi = atan2(y, x); */

        dz = dz * bgg->zInterval;
        preFactorDz = -dz * particleCharge * particleRelSign / (pCentral * particleMass * c_mks * (1.0 + delta));
        preFactorDz = preFactorDz * sqrt(1.0 + xp * xp + yp * yp);
        /* Apply prediction step */
        xTemp = x + dz * xp;
        yTemp = y + dz * yp;
        xpTemp = xp + preFactorDz * ((yp * B[2] - (1.0 + xp * xp) * B[1]) + xp * yp * B[0]);
        ypTemp = yp + preFactorDz * (((1.0 + yp * yp) * B[0] - xp * B[2]) - xp * yp * B[1]);
        ds = dz * sqrt(1 + sqr(xp) + sqr(yp));

        /* r = sqrt(sqr(xTemp)+sqr(yTemp)); */
        /* phi = atan2(yTemp, xTemp); */

        /* Compute fields at next z location */
        if ((isLost += computeGGEMagneticFields(&B[0], &B[1], &B[2], xTemp, yTemp, iz + 1, bgg, bggData, igLimit)))
          break;

        dz = dz * bgg->zInterval;
        preFactorDz = -dz * particleCharge * particleRelSign / (pCentral * particleMass * c_mks * (1.0 + delta));
        preFactorDz = preFactorDz * sqrt(1.0 + xpTemp * xpTemp + ypTemp * ypTemp);

        /* Apply correction step */
        xNew = 0.5 * (x + xTemp + dz * xpTemp);
        yNew = 0.5 * (y + yTemp + dz * ypTemp);
        xpNew = 0.5 * (xp + xpTemp + preFactorDz * ((ypTemp * B[2] - (1.0 + xpTemp * xpTemp) * B[1]) + xpTemp * ypTemp * B[0]));
        ypNew = 0.5 * (yp + ypTemp + preFactorDz * (((1.0 + ypTemp * ypTemp) * B[0] - xpTemp * B[2]) - xpTemp * ypTemp * B[1]));
        ds = 0.5 * (ds + dz * sqrt(1 + sqr(xpTemp) + sqr(ypTemp)));

        x = xNew;
        y = yNew;
        xp = xpNew;
        yp = ypNew;
        s += ds;

        denom = sqrt(1 + sqr(xp) + sqr(yp));
        p[2] = pCentral * (1 + delta) / denom;
        p[0] = xp * p[2];
        p[1] = yp * p[2];

#ifdef DEBUG
        fprintf(fpdebug, "%.0f %le %le %le %le %le %le %le %le %le %le\n",
                part[ip][6], ds, x, y, iz * dz,
                B[0], B[1], B[2],
                p[0], p[1], p[2]);
#endif

        if (bgg->synchRad) {
          /* This is only valid for ultra-relatistic particles */
          double B2, F;
          B2 = sqr(B[0]) + sqr(B[1]);
          if (B2 > B2Max)
            B2Max = B2;
          deltaTemp = delta - radCoef * pCentral * sqr(1.0 + delta) * B2 * ds;
          F = isrCoef * pCentral * sqr(1.0 + delta) * sqrt(ds) * pow(B2, 3. / 4.);
          if (bgg->isr && np != 1)
            deltaTemp += F * gauss_rn_lim(0.0, 1.0, srGaussianLimit, random_2);
          if (sigmaDelta2)
            *sigmaDelta2 += sqr(F);
          delta = deltaTemp;
        }
      }
      if (iz < nz - 1) {
        /* Drift forward */
        x += xp * dz * (nz - (iz - 1));
        y += yp * dz * (nz - (iz - 1));
        s += dz * (nz - (iz - 1)) * sqrt(1 + sqr(xp) + sqr(yp));
      }

#if !USE_MPI
      if (bgg->SDDSpo && np < 1000) {
        if (!SDDS_SetRowValues(bgg->SDDSpo, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, irow++,
                               bgg->poIndex[0], x,
                               bgg->poIndex[1], p[0],
                               bgg->poIndex[2], y,
                               bgg->poIndex[3], p[1],
                               bgg->poIndex[4], (nz - 1) * dz + bggData[0]->zMin,
                               bgg->poIndex[5], p[2],
                               bgg->poIndex[6], B[0],
                               bgg->poIndex[7], B[1],
                               bgg->poIndex[8], B[2],
                               bgg->poIndex[9], (uint64_t)part[ip][particleIDIndex],
                               bgg->poIndex[10], s,
                               -1) ||
            !SDDS_WritePage(bgg->SDDSpo)) {
          SDDS_SetError("Problem setting particle output data for BGGEXP");
          SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
        }
      }
#endif

      /* Drift backward from end of field map to exit plane */
      x -= (zMax - zExit) * xp;
      y -= (zMax - zExit) * yp;
      s -= (zMax - zExit) * sqrt(1.0 + xp * xp + yp * yp);

      /* Shift according to central point of the expansion, if given in input file */
      if (bggData[0] && (bggData[0]->xCenter || bggData[0]->yCenter)) {
        x += bggData[0]->xCenter;
        y += bggData[0]->yCenter;
      } else if (bggData[1] && (bggData[1]->xCenter || bggData[1]->yCenter)) {
        x += bggData[1]->xCenter;
        y += bggData[1]->yCenter;
      }

      x -= xExit - bgg->dxExpansion; /* Shift x according to exit point & map shift */

      /* Compute transformation from magnet grid -> accelerator coordinates if bending magnet */
      if (bgg->isBend) {
        phi = atan2(xVertex - xExit, zExit - zVertex); /* angle of outgoing x-plane w.r.t. magnet x */
        magnet_s = x * sin(phi) / (cos(phi) - xp * sin(phi));
        x = (x + xp * magnet_s) / cos(phi);
        y = y + yp * magnet_s;
        s += sqrt(1.0 + xp * xp + yp * yp) * magnet_s;
        yp = yp / (cos(phi) - xp * sin(phi));
        xp = (xp * cos(phi) + sin(phi)) / (cos(phi) - xp * sin(phi));
      } else {
        phi = magnet_s = 0;
      }

      part[ip][0] = x;
      part[ip][1] = xp; /*  p[0]/p[2]; */
      part[ip][2] = y;
      part[ip][3] = yp; /*  p[1]/p[2]; */
      part[ip][4] = s;
      if (isLost) {
        /* TODO: do something about lost particles */
      }
      /*
      printf("P: %le -> %le, change = %le, B2Max = %le\n",
             pOrig,
             sqrt(sqr(p[0])+sqr(p[1])+sqr(p[2])),
             sqrt(sqr(p[0])+sqr(p[1])+sqr(p[2]))-pOrig,
             B2Max);
      */
      if (bgg->synchRad) {
        double gamma0, beta0, gamma1, p1, beta1;
        gamma0 = sqrt(sqr(pCentral * (1 + part[ip][5])) + 1);
        beta0 = pCentral * (1 + part[ip][5]) / gamma0;
        p1 = sqrt(sqr(p[0]) + sqr(p[1]) + sqr(p[2]));
        gamma1 = sqrt(p1 * p1 + 1);
        beta1 = p1 / gamma1;
        part[ip][4] *= beta1 / beta0;
        part[ip][5] = p1 / pCentral - 1;
      }
    }
  }

  if (sigmaDelta2 && np)
    *sigmaDelta2 /= np;

  /* Undo misalignments */
  if (bgg->tilt)
    rotateBeamCoordinatesForMisalignment(part, np, -bgg->tilt);
  if (bgg->dx || bgg->dy || bgg->dz)
    offsetBeamCoordinatesForMisalignment(part, np, -bgg->dx, -bgg->dy, -bgg->dz);

#ifdef DEBUG
  fflush(fpdebug);
#endif

#if defined(HAVE_GPU) && defined(GPU_VERIFY)
  if (gpuTracked)
    compareGpuCpu(np, "trackBGGExpansion");
#endif
  return np;
}

long computeGGEMagneticFields(double *Bx, double *By, double *Bz,
                              double x, double y, long iz, BGGEXP *bgg, STORED_BGGEXP_DATA *bggData[2], long igLimit[2]) {
  double Br, Bphi, mfactor;
  long isLost, ns, im, m, ig;
  double r, phi;

  Br = Bphi = *Bx = *By = *Bz = 0;
  isLost = 0;
  r = sqrt(sqr(x) + sqr(y));
  phi = atan2(y, x);

  for (ns = 0; ns < 2; ns++) {
    /* ns=0 => normal, ns=1 => skew */
    if (bggData[ns]) {
      if ((bggData[ns]->xMax > 0 && fabs(x) > bggData[ns]->xMax) &&
          (bggData[ns]->yMax > 0 && fabs(y) > bggData[ns]->yMax)) {
        isLost = 1;
        break;
      }
      for (im = 0; im < bggData[ns]->nm; im++) {
        double mfact, term, sin_mphi, cos_mphi;
        m = bggData[ns]->m[im];
        mfactor = 1;
        if (m < 5)
          mfactor = bgg->multipoleFactor[m];
        if (bgg->mMaximum > 0 && m > bgg->mMaximum)
          continue;
        mfact = dfactorial(m);
        sin_mphi = sin(m * phi);
        cos_mphi = cos(m * phi);
        if (ns == 0) {
          /* normal */
          for (ig = 0; ig < igLimit[ns]; ig++) {
            term = ipow(-1, ig) * mfact / (ipow(2, 2 * ig) * factorial(ig) * factorial(ig + m)) * ipow(r, 2 * ig + m - 1);
            *Bz += term * bggData[ns]->dCmn_dz[im][ig][iz] * r * sin_mphi * mfactor;
            term *= bggData[ns]->Cmn[im][ig][iz];
            Br += term * (2 * ig + m) * sin_mphi * mfactor;
            Bphi += m * term * cos_mphi * mfactor;
          }
        } else {
          /* skew */
          if (m == 0) {
            *Bz += bggData[ns]->dCmn_dz[im][0][iz] * mfactor; // on-axis Bz from m=ig=0 term
            for (ig = 1; ig < igLimit[ns]; ig++) {
              term = ipow(-1, ig) * mfact / (ipow(2, 2 * ig) * factorial(ig) * factorial(ig + m)) * ipow(r, 2 * ig + m - 1);
              *Bz += term * bggData[ns]->dCmn_dz[im][ig][iz] * r * mfactor;
              Br += term * (2 * ig + m) * bggData[ns]->Cmn[im][ig][iz] * mfactor;
            }
          } else {
            for (ig = 0; ig < igLimit[ns]; ig++) {
              term = ipow(-1, ig) * mfact / (ipow(2, 2 * ig) * factorial(ig) * factorial(ig + m)) * ipow(r, 2 * ig + m - 1);
              *Bz += term * bggData[ns]->dCmn_dz[im][ig][iz] * r * cos_mphi * mfactor;
              term *= bggData[ns]->Cmn[im][ig][iz];
              Br += term * (2 * ig + m) * cos_mphi * mfactor;
              Bphi -= m * term * sin_mphi * mfactor;
            }
          }
        }
      }
    }
  }
  *Bx = ((bgg->Bx + (Br * cos(phi) - Bphi * sin(phi))) * bgg->strength) * bgg->BFactor[0];
  *By = ((bgg->By + (Br * sin(phi) + Bphi * cos(phi))) * bgg->strength) * bgg->BFactor[1];
  *Bz *= bgg->strength * bgg->BFactor[2];
  return isLost;
}

long computeGGEVectorPotential(
  double *Ax, double *dAx_dx, double *dAx_dy,
  double *Ay, double *dAy_dx, double *dAy_dy,
  double *dAz_dx, double *dAz_dy,
  double x, double y, long iz,
  BGGEXP *bgg, STORED_BGGEXP_DATA *bggData[2], long igLimit[2]) {
  long isLost, ns, im, m, ig;
  double r, phi, sin_phi, cos_phi;

  *Ax = *dAx_dx = *dAx_dy = *Ay = *dAy_dx = *dAy_dy = *dAz_dx = *dAz_dy = 0;
  isLost = 0;
  r = sqrt(sqr(x) + sqr(y));
  phi = atan2(y, x);
  cos_phi = cos(phi);
  sin_phi = sin(phi);

  for (ns = 0; ns < 2; ns++) {
    /* ns=0 => normal, ns=1 => skew */
    if (bggData[ns]) {
      double mfactor;
      if ((bggData[ns]->xMax > 0 && fabs(x) > bggData[ns]->xMax) &&
          (bggData[ns]->yMax > 0 && fabs(y) > bggData[ns]->yMax))
        isLost = 1;
      for (im = 0; im < bggData[ns]->nm; im++) {
        double mfact, term, rDeriv, phiDeriv, sin_m1phi, cos_m1phi, sin_mphi, cos_mphi;
        m = bggData[ns]->m[im];
        mfactor = 1;
        if (m < 5)
          mfactor = bgg->multipoleFactor[m];
        if (bgg->mMaximum > 0 && m > bgg->mMaximum)
          continue;
        mfact = dfactorial(m);
        sin_m1phi = sin((m + 1) * phi);
        cos_m1phi = cos((m + 1) * phi);
        sin_mphi = sin(m * phi);
        cos_mphi = cos(m * phi);
        if (ns == 0) {
          /* normal components in symmetric Coulomb gauge */
          for (ig = 0; ig < igLimit[ns]; ig++) {
            term = ipow(-1, ig) * mfact * ipow(r, 2 * ig + m) / (2.0 * ipow(4, ig) * factorial(ig) * factorial(ig + m + 1));

            *Ax += term * cos_m1phi * bggData[ns]->dCmn_dz[im][ig][iz] * mfactor;
            *Ay += term * sin_m1phi * bggData[ns]->dCmn_dz[im][ig][iz] * mfactor;

            rDeriv = (2 * ig + m + 1) * cos_m1phi * bggData[ns]->dCmn_dz[im][ig][iz];
            phiDeriv = -(m + 1) * sin_m1phi * bggData[ns]->dCmn_dz[im][ig][iz];
            *dAx_dx += term * (cos_phi * rDeriv - sin_phi * phiDeriv) * mfactor;
            *dAx_dy += term * (sin_phi * rDeriv + cos_phi * phiDeriv) * mfactor;

            rDeriv = (2 * ig + m + 1) * sin_m1phi * bggData[ns]->dCmn_dz[im][ig][iz];
            phiDeriv = (m + 1) * cos_m1phi * bggData[ns]->dCmn_dz[im][ig][iz];
            *dAy_dx += term * (cos_phi * rDeriv - sin_phi * phiDeriv) * mfactor;
            *dAy_dy += term * (sin_phi * rDeriv + cos_phi * phiDeriv) * mfactor;

            term = ipow(-1, ig) * mfact * ipow(r, 2 * ig + m - 1) / (ipow(4, ig) * factorial(ig) * factorial(ig + m));
            rDeriv = -(2 * ig + m) * cos_mphi * bggData[ns]->Cmn[im][ig][iz];
            phiDeriv = m * sin_mphi * bggData[ns]->Cmn[im][ig][iz];
            *dAz_dx += term * (cos_phi * rDeriv - sin_phi * phiDeriv) * mfactor;
            *dAz_dy += term * (sin_phi * rDeriv + cos_phi * phiDeriv) * mfactor;
          }
        } else {
          /* skew components in symmetric Coulomb gauge */
          for (ig = 0; ig < igLimit[ns]; ig++) {
            term = ipow(-1, ig) * mfact * ipow(r, 2 * ig + m) / (2.0 * ipow(4, ig) * factorial(ig) * factorial(ig + m + 1));
            //dGenGrad_s = bggData->dCmns_dz[im][ig][iz];
            //GenGrad_s = bggData->Cmns[im][ig][iz];

            *Ax -= term * sin_m1phi * bggData[ns]->dCmn_dz[im][ig][iz] * mfactor;
            *Ay += term * cos_m1phi * bggData[ns]->dCmn_dz[im][ig][iz] * mfactor;

            rDeriv = -(2 * ig + m + 1) * sin_m1phi * bggData[ns]->dCmn_dz[im][ig][iz];
            phiDeriv = -(m + 1) * cos_m1phi * bggData[ns]->dCmn_dz[im][ig][iz];
            *dAx_dx += term * (cos_phi * rDeriv - sin_phi * phiDeriv) * mfactor;
            *dAx_dy += term * (sin_phi * rDeriv + cos_phi * phiDeriv) * mfactor;

            rDeriv = (2 * ig + m + 1) * cos_m1phi * bggData[ns]->dCmn_dz[im][ig][iz];
            phiDeriv = -(m + 1) * sin_m1phi * bggData[ns]->dCmn_dz[im][ig][iz];
            *dAy_dx += term * (cos_phi * rDeriv - sin_phi * phiDeriv) * mfactor;
            *dAy_dy += term * (sin_phi * rDeriv + cos_phi * phiDeriv) * mfactor;

            if (m > 0) {
              term = ipow(-1, ig) * mfact * ipow(r, 2 * ig + m - 1) / (ipow(4, ig) * factorial(ig) * factorial(ig + m));
              rDeriv = (2 * ig + m) * sin_mphi * bggData[ns]->Cmn[im][ig][iz];
              phiDeriv = m * cos_mphi * bggData[ns]->Cmn[im][ig][iz];
              *dAz_dx += term * (cos_phi * rDeriv - sin_phi * phiDeriv) * mfactor;
              *dAz_dy += term * (sin_phi * rDeriv + cos_phi * phiDeriv) * mfactor;
            }
          }
        }
      }
    }
  }
  *Ax *= r;
  *Ay *= r;
  *dAz_dx = *dAz_dx - bgg->By;
  *dAz_dy = *dAz_dy + bgg->Bx;
  return isLost;
}
