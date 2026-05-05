/* Copyright 2015 by Michael Borland and Argonne National Laboratory,
 * all rights reserved.
 */
/* program: bratSubroutines.c
 * purpose: Bending magnet RAy Trace program for gridded data for dipole magnet.
 *           
 * The equations of motion are
 *        -
 *      d w    -1  -   - -
 *      --- = ---- w x B(q)
 *      d s    H
 *
 *        -
 *      d q    -
 *      --- =  w
 *      d s
 *
 * where B is the magnetic field, s is the distance,
 * q is the coordinate, w is the rate of change of q with
 * distance, and H is the rigidity
 * q(0) = z in the field input file
 * q(1) = x in the field input file
 * q(2) = y
 */
#include "mdb.h"
#include "SDDS.h"
#include "track.h"

void BRAT_B_field(double *B, double *Q);
void BRAT_B_field_permuted(double *B, double *Q);
double BRAT_setup_field_data(char *input, double xCenter, double zCenter, char *interpolationParameter);
double BRAT_setup_arc_field_data(char *input, char *sName, char *fieldName, double xCenter);
void BRAT_setup_single_scan_field_data(char *input);
double BRAT_exit_function(double *qp, double *q, double s);
void BRAT_deriv_function(double *qp, double *q, double s);
long BRAT_lorentz_integration(double *accelCoord, double *q, long doStoreData, double *BMaxReturn);
void BRAT_store_data(double *qp, double *q, double s, double exval);
void BRAT_optimize_magnet(unsigned long flags);
double refineAngle(double theta, double z0, double x0, double zv, double xv,
                   double z1, double x1);
void BRAT_store_data_permuted(double p0, double *p, double *q1, double *B, double s, double exval);
static long verbose_optimize = 0;
void clear2dMapList();
void add2dMapList(double interpValue);
void readBratFieldFile(BRAT *brat, char *filename, short additionalFile);
void writeBratFieldOutput(BRAT *brat, char *rootname);

/* parameters of element needed for integration */
static double theta, fse = 0, interpParameter = 0;
static long interpOrder = 1;
static double dXOffset = 0, dYOffset = 0, dZOffset = 0;
static double magnetYaw = 0;
static double fieldFactor = 1, mainFactor = 1;
static long zDuplicate = 0;
static double rhoMax = 0;
static double fieldSign = 1;
static double deltaByInside = 0;
static double BMaxOnTrajectory = -DBL_MAX;
static double BSignOnTrajectory = 0;
static char *interpolationParameterUnits = NULL;
static double xfWeight = 1, xfpWeight = 1;
static double dxDzFactor;
static short ySymmetryCodeGlobal=0; /* 0 = none, 1 = odd, 2 = even */
static short zSymmetryCodeGlobal=0; /* 0 = none, 1 = odd, 2 = even */

#define BRAT_INTERP_EXTRAPOLATE 0x001UL
#define BRAT_INTERP_PERMISSIVE 0x002UL
static unsigned long bratInterpFlags = 0;

/* parameters for field calculation: */
/* Bnorm is misnamed here, based on earlier versions of the program */
static double **Bnorm, **dBnormdz, **dBnormdx;
static double *BxNorm, *ByNorm, *BzNorm;
static float *Bx1Norm, *By1Norm, *Bz1Norm;
static short singlePrecision = 0;
static double additionalFactor = 0;
static double *BxNormAdditional, *ByNormAdditional, *BzNormAdditional;
static float *Bx1NormAdditional, *By1NormAdditional, *Bz1NormAdditional;
/* these are used to store multiple 2D field maps for interpolation for abrat program */
/* the results of interpolation are loaded into Bnorm, dBnormdz, and dBnormdx */
static double *interpolationParameterValue = NULL;
static double ***BnormList = NULL, ***dBnormdzList = NULL, ***dBnormdxList = NULL;
static long length2dMapList = 0;
/* static double Breference = 0; */
static double xi, xf, dx;
static double yi, yf, dy;
static double zi, zf, dz, z_outer;
static long nx, ny, nz;
static long extend_data, particle_inside, single_scan = 0, arc_scan = 0;
static double extend_edge_angle;
static double gap = 0.04;
static double central_length;

static double *BArcNorm, *dBArcNormds, *sArc, rhoArc;
static long nArcPoints;

/* tolerance for integration, zero-finding */
static double integ_tol, zero_tol;

/* variables for storing integration history */
static short storeData = 0; /* 0: no data stored, 1: only (X, Y, Z, BX, BY, BZ) are stored, 2: add (wX, Wy, wZ), 3: everything */
static double *X_stored = NULL, *Z_stored = NULL, *Y_stored = NULL;
static double *wX_stored = NULL, *wZ_stored = NULL, *wY_stored = NULL;
static double *aX_stored = NULL, *aZ_stored = NULL, *aY_stored = NULL;
static double *FX_stored = NULL, *FZ_stored = NULL, *FY_stored = NULL, *Fint_stored = NULL;
static double *s_stored;
static long n_stored, max_store;

/* particle trajectory parameters */
static double zStart, zEnd;         /* starting and ending point of integration */
static double xNomEntry, zNomEntry; /* coordinates of hard-edge entry */
static double xNomExit, zNomExit;   /* coordinates of hard-edge exit */
static double xVertex, zVertex;
static double thetaEntry; /* angle of the initial nominal trajectory */
static double rigidity;
static double xCenter, zCenter;
static double global_delta;

static long isLost = 0;
#ifndef ABRAT_PROGRAM
static double lossCoordinates[4]; /* X, Z, thetaX, y */
#endif

#define TOLERANCE_FACTOR 1e-14
#define OPTIMIZE_ON 0x0001
#define OPTIMIZE_FSE 0x0002
#define OPTIMIZE_DX 0x0004
#define OPTIMIZE_VERBOSE 0x0008
#define OPTIMIZE_QUIET 0x0010
#define OPTIMIZE_DZ 0x0020
#define OPTIMIZE_YAW 0x0040

static long quiet = 0;
static long useFTABLE = 0;
static double Po;

static short idealMode = 0, fieldMapDimension = 2, xyInterpolationOrder, xyGridExcess = 0, xyExtrapolate = 1;
/* static double idealB; */
static double idealChord, idealEdgeAngle;

static double fseLimit[2] = {-1, 1};
static double dxLimit[2] = {-1, 1};
static double dzLimit[2] = {-1, 1};
static double yawLimit[2] = {-1, 1};

typedef struct {
  char *filename;
  long nx, ny, nz;
  short singlePrecision, additionalMap;
  short ySymmetryCode; /* 0 = none, 1 = odd, 2 = even */
  short zSymmetryCode; /* 0 = none, 1 = odd, 2 = even */
  /* used if singlePrecision=0 */
  double *Bx, *By, *Bz;
  double *BxAdditional, *ByAdditional, *BzAdditional;
  /* used if singlePrecision!=0 */
  float *Bx1, *By1, *Bz1;
  float *Bx1Additional, *By1Additional, *Bz1Bend;
  double xi, xf, dx;
  double yi, yf, dy;
  double zi, zf, dz;
  double Bmin, Bmax;
} BRAT_3D_DATA;
static BRAT_3D_DATA *brat3dData = NULL;
static long nBrat3dData = 0;

long trackBRAT(double **part, long np, BRAT *brat, double pCentral, double **accepted) {
  long ip, ic, itop;
  TRACKING_CONTEXT tcontext;

  getTrackingContext(&tcontext);
  Po = pCentral;

  if (!brat->initialized) {
    /* double Bmin, Bmax; */
    /* long i, rows, idata; */
    /* SDDS_DATASET SDDS_table; */

    brat->dataIndex = brat->dataIndexAdditional = -1;
    brat->fieldOutputDone = 0;
    readBratFieldFile(brat, brat->filename, 0);
    if (brat->filenameAdditional && strlen(brat->filenameAdditional))
      readBratFieldFile(brat, brat->filenameAdditional, 1);

#ifndef ABRAT_PROGRAM
    if (brat->particleOutput && strlen(brat->particleOutput) && isSlave) {
      char *filename;
#  if USE_MPI
      filename = compose_filename_per_processor(brat->particleOutput, tcontext.rootname);
#  else
      filename = compose_filename(brat->particleOutput, tcontext.rootname);
#  endif
      brat->SDDSparticleOutput = tmalloc(sizeof(SDDS_DATASET));
      if (!SDDS_InitializeOutput(brat->SDDSparticleOutput, SDDS_BINARY, 0, NULL, NULL, filename)) {
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        exitElegant(1);
      }
      free(filename);
      if (!SDDS_DefineSimpleColumn(brat->SDDSparticleOutput, "x", "m", SDDS_FLOAT) ||
          !SDDS_DefineSimpleColumn(brat->SDDSparticleOutput, "y", "m", SDDS_FLOAT) ||
          !SDDS_DefineSimpleColumn(brat->SDDSparticleOutput, "z", "m", SDDS_FLOAT) ||
          !SDDS_DefineSimpleColumn(brat->SDDSparticleOutput, "wx", NULL, SDDS_FLOAT) ||
          !SDDS_DefineSimpleColumn(brat->SDDSparticleOutput, "wy", NULL, SDDS_FLOAT) ||
          !SDDS_DefineSimpleColumn(brat->SDDSparticleOutput, "wz", NULL, SDDS_FLOAT) ||
          !SDDS_DefineSimpleColumn(brat->SDDSparticleOutput, "Bx", "T", SDDS_FLOAT) ||
          !SDDS_DefineSimpleColumn(brat->SDDSparticleOutput, "By", "T", SDDS_FLOAT) ||
          !SDDS_DefineSimpleColumn(brat->SDDSparticleOutput, "Bz", "T", SDDS_FLOAT) ||
          !SDDS_DefineSimpleParameter(brat->SDDSparticleOutput, "particleID", NULL, SDDS_ULONG64) ||
          !SDDS_DefineSimpleParameter(brat->SDDSparticleOutput, "ySymmetryCode", NULL, SDDS_SHORT) ||
          !SDDS_DefineSimpleParameter(brat->SDDSparticleOutput, "zSymmetryCode", NULL, SDDS_SHORT) ||
          !SDDS_DefineSimpleParameter(brat->SDDSparticleOutput, "XLoss", "m", SDDS_FLOAT) ||
          !SDDS_DefineSimpleParameter(brat->SDDSparticleOutput, "yLoss", "m", SDDS_FLOAT) ||
          !SDDS_DefineSimpleParameter(brat->SDDSparticleOutput, "ZLoss", "m", SDDS_FLOAT) ||
          !SDDS_DefineSimpleParameter(brat->SDDSparticleOutput, "thetaLoss", NULL, SDDS_FLOAT)) {
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        exitElegant(1);
      }
      if (!SDDS_WriteLayout(brat->SDDSparticleOutput)) {
        printf("Unable to write SDDS layout for file %s (%s)\n", brat->particleOutput, tcontext.elementName);
        fflush(stdout);
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        exitElegant(1);
      }
    } else
      brat->SDDSparticleOutput = NULL;
#endif /* not abrat program */

    brat->initialized = 1;
  }

  /* Store nothing if running test particles, otherwise store (x, y, z, Wx, Wy, Wz, Bx, By, Bz) */
  storeData = tcontext.flags & TEST_PARTICLES ? 0 : 2;

  if (brat->dataIndex < 0 || brat->dataIndex >= nBrat3dData)
    bombElegant("BRAT data indexing bug (1). Please report.", NULL);
  if (strcmp(brat->filename, brat3dData[brat->dataIndex].filename) != 0)
    bombElegant("BRAT data indexing bug (2). Please report.", NULL);
  if (brat->filenameAdditional && strlen(brat->filenameAdditional)) {
    if (brat->dataIndexAdditional < 0 || brat->dataIndexAdditional >= nBrat3dData)
      bombElegant("BRAT data indexing bug (3). Please report.", NULL);

    if (strcmp(brat->filenameAdditional, brat3dData[brat->dataIndexAdditional].filename) != 0)
      bombElegant("BRAT data indexing bug (4). Please report.", NULL);

    if (brat3dData[brat->dataIndex].nx != brat3dData[brat->dataIndexAdditional].nx)
      bombElegantVA("BRAT main (%s) and additional (%s) files have different number of x grid points (%ld vs %ld).\n",
                    brat3dData[brat->dataIndex].filename,
                    brat3dData[brat->dataIndexAdditional].filename,
                    brat3dData[brat->dataIndex].nx, brat3dData[brat->dataIndexAdditional].nx);
    if (brat3dData[brat->dataIndex].ny != brat3dData[brat->dataIndexAdditional].ny)
      bombElegantVA("BRAT main (%s) and additional (%s) files have different number of y grid points.\n",
                    brat3dData[brat->dataIndex].filename,
                    brat3dData[brat->dataIndexAdditional].filename);
    if (brat3dData[brat->dataIndex].nz != brat3dData[brat->dataIndexAdditional].nz)
      bombElegantVA("BRAT main (%s) and additional (%s) files have different number of z grid points.\n",
                    brat3dData[brat->dataIndex].filename,
                    brat3dData[brat->dataIndexAdditional].filename);

    if (brat3dData[brat->dataIndex].xi != brat3dData[brat->dataIndexAdditional].xi)
      bombElegantVA("BRAT main (%s) and additional (%s) files have different x lower limit.\n",
                    brat3dData[brat->dataIndex].filename,
                    brat3dData[brat->dataIndexAdditional].filename);
    if (brat3dData[brat->dataIndex].yi != brat3dData[brat->dataIndexAdditional].yi)
      bombElegantVA("BRAT main (%s) and additional (%s) files have different y lower limit.\n",
                    brat3dData[brat->dataIndex].filename,
                    brat3dData[brat->dataIndexAdditional].filename);
    if (brat3dData[brat->dataIndex].zi != brat3dData[brat->dataIndexAdditional].zi)
      bombElegantVA("BRAT main (%s) and additional (%s) files have different z lower limit.\n",
                    brat3dData[brat->dataIndex].filename,
                    brat3dData[brat->dataIndexAdditional].filename);

    if (brat3dData[brat->dataIndex].xf != brat3dData[brat->dataIndexAdditional].xf)
      bombElegantVA("BRAT main (%s) and additional (%s) files have different x upper limit.\n",
                    brat3dData[brat->dataIndex].filename,
                    brat3dData[brat->dataIndexAdditional].filename);
    if (brat3dData[brat->dataIndex].yf != brat3dData[brat->dataIndexAdditional].yf)
      bombElegantVA("BRAT main (%s) and additional (%s) files have different y upper limit.\n",
                    brat3dData[brat->dataIndex].filename,
                    brat3dData[brat->dataIndexAdditional].filename);
    if (brat3dData[brat->dataIndex].zf != brat3dData[brat->dataIndexAdditional].zf)
      bombElegantVA("BRAT main (%s) and additional (%s) files have different z upper limit.\n",
                    brat3dData[brat->dataIndex].filename,
                    brat3dData[brat->dataIndexAdditional].filename);

    if (brat3dData[brat->dataIndex].dx != brat3dData[brat->dataIndexAdditional].dx)
      bombElegantVA("BRAT main (%s) and additional (%s) files have different x grid spacing.\n",
                    brat3dData[brat->dataIndex].filename,
                    brat3dData[brat->dataIndexAdditional].filename);
    if (brat3dData[brat->dataIndex].dy != brat3dData[brat->dataIndexAdditional].dy)
      bombElegantVA("BRAT main (%s) and additional (%s) files have different y grid spacing.\n",
                    brat3dData[brat->dataIndex].filename,
                    brat3dData[brat->dataIndexAdditional].filename);
    if (brat3dData[brat->dataIndex].dz != brat3dData[brat->dataIndexAdditional].dz)
      bombElegantVA("BRAT main (%s) and additional (%s) files have different z grid spacing.\n",
                    brat3dData[brat->dataIndex].filename,
                    brat3dData[brat->dataIndexAdditional].filename);
    if (brat3dData[brat->dataIndex].ySymmetryCode != brat3dData[brat->dataIndexAdditional].ySymmetryCode)
      bombElegantVA("BRAT main (%s) and additional (%s) files have different y symmetry.\n",
                    brat3dData[brat->dataIndex].filename,
                    brat3dData[brat->dataIndexAdditional].filename);
    if (brat3dData[brat->dataIndex].zSymmetryCode != brat3dData[brat->dataIndexAdditional].zSymmetryCode)
      bombElegantVA("BRAT main (%s) and additional (%s) files have different z symmetry.\n",
                    brat3dData[brat->dataIndex].filename,
                    brat3dData[brat->dataIndexAdditional].filename);
  }

  BxNorm = brat3dData[brat->dataIndex].Bx;
  Bx1Norm = brat3dData[brat->dataIndex].Bx1;
  xi = brat3dData[brat->dataIndex].xi;
  xf = brat3dData[brat->dataIndex].xf;
  dx = brat3dData[brat->dataIndex].dx;
  nx = brat3dData[brat->dataIndex].nx;

  ByNorm = brat3dData[brat->dataIndex].By;
  By1Norm = brat3dData[brat->dataIndex].By1;
  yi = brat3dData[brat->dataIndex].yi;
  yf = brat3dData[brat->dataIndex].yf;
  dy = brat3dData[brat->dataIndex].dy;
  ny = brat3dData[brat->dataIndex].ny;

  BzNorm = brat3dData[brat->dataIndex].Bz;
  Bz1Norm = brat3dData[brat->dataIndex].Bz1;
  zi = brat3dData[brat->dataIndex].zi;
  zf = brat3dData[brat->dataIndex].zf;
  dz = brat3dData[brat->dataIndex].dz;
  nz = brat3dData[brat->dataIndex].nz;

  additionalFactor = 0;
  if (brat->filenameAdditional && strlen(brat->filenameAdditional)) {
    BxNormAdditional = brat3dData[brat->dataIndexAdditional].Bx;
    Bx1NormAdditional = brat3dData[brat->dataIndexAdditional].Bx1;
    ByNormAdditional = brat3dData[brat->dataIndexAdditional].By;
    By1NormAdditional = brat3dData[brat->dataIndexAdditional].By1;
    BzNormAdditional = brat3dData[brat->dataIndexAdditional].Bz;
    Bz1NormAdditional = brat3dData[brat->dataIndexAdditional].Bz1;
    additionalFactor = brat->additionalFactor;
  }

  xyInterpolationOrder = brat->xyInterpolationOrder;
  xyGridExcess = brat->xyGridExcess;
  xyExtrapolate = brat->xyExtrapolate;
  singlePrecision = brat3dData[brat->dataIndex].singlePrecision;
  ySymmetryCodeGlobal = brat3dData[brat->dataIndex].ySymmetryCode;
  zSymmetryCodeGlobal = brat3dData[brat->dataIndex].zSymmetryCode;

  if (brat3dData[brat->dataIndex].zSymmetryCode) {
    zi = 0;
    zStart = -zf - dz;
  }  else {
    zStart = zi - dz;
  }
  z_outer = MAX(fabs(zi), fabs(zf));

  fieldMapDimension = 3;

  fieldFactor = brat->fieldFactor;
  mainFactor = brat->mainFactor;
  theta = brat->angle;
  fse = brat->fse;
  dXOffset = brat->dxMap;
  dYOffset = brat->dyMap;
  dZOffset = brat->dzMap;
  magnetYaw = brat->yawMap;

  xVertex = brat->xVertex;
  zVertex = brat->zVertex;
  xNomEntry = brat->xEntry;
  zNomEntry = brat->zEntry;
  xNomExit = brat->xExit;
  zNomExit = brat->zExit;
  zCenter = (zNomEntry + zNomExit) / 2;
  xCenter = (xNomEntry + xNomExit) / 2;
  central_length = 100 * (zNomExit - zNomEntry);

  theta = refineAngle(theta, zNomEntry, xNomEntry, zVertex, xVertex,
                      zNomExit, xNomExit);
  thetaEntry = atan2(xVertex - xNomEntry, zVertex - zNomEntry);

  rigidity = pCentral * particleMass * c_mks / particleCharge * particleRelSign;
  rhoMax = fabs(rigidity / brat3dData[brat->dataIndex].Bmax);
  deltaByInside = brat->deltaByInside;

  integ_tol = brat->accuracy;
  zero_tol = integ_tol * 10;
  useFTABLE = brat->useFTABLE;

  if (brat->flip) {
    for (ip=0; ip<np; ip++) {
      part[ip][0] *= -1;
      part[ip][1] *= -1;
      part[ip][2] *= -1;
      part[ip][3] *= -1;
    }
  }

#ifndef ABRAT_PROGRAM
  if (!brat->fieldOutputDone && brat->fieldOutputFile && strlen(brat->fieldOutputFile))
    writeBratFieldOutput(brat, tcontext.rootname);
  brat->fieldOutputDone = 1;
#endif

  itop = np - 1;
  for (ip = 0; ip <= itop; ip++) {
    double accelCoord[6], q[10];
#ifndef ABRAT_PROGRAM
    long i, j;
    long iOut;
#endif
    for (ic = 0; ic < 6; ic++)
      accelCoord[ic] = part[ip][ic];
    n_stored = 0;
    BRAT_lorentz_integration(accelCoord, q, brat->SDDSparticleOutput ? 1 : 0, NULL);
    for (ic = 0; ic < 6; ic++)
      part[ip][ic] = accelCoord[ic];
#ifndef ABRAT_PROGRAM
    iOut = ip;
    if (isLost) {
      if (globalLossCoordOffset > 0)
        for (ic = 0; ic < GLOBAL_LOSS_PROPERTIES_PER_PARTICLE; ic++)
          part[ip][globalLossCoordOffset + ic] = lossCoordinates[ic];
      part[ip][2] = lossCoordinates[3];
      part[ip][5] = pCentral * (1 + global_delta);
      swapParticles(part[ip], part[itop]);
      if (accepted)
        swapParticles(accepted[ip], accepted[itop]);
      iOut = itop;
      itop--;
      ip--;
    }
    if (isSlave && brat->SDDSparticleOutput && (!brat->particleOutputLostOnly || isLost) && (brat->particleOutputSelectionInterval <= 1 || ((long)part[iOut][particleIDIndex]) % brat->particleOutputSelectionInterval == 0)) {
      SDDS_DATASET *SDDS_table;
      SDDS_table = brat->SDDSparticleOutput;
      if (!SDDS_StartTable(SDDS_table, n_stored)) {
        SDDS_SetError("Problem starting SDDS table (BRAT)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      }
      if (!SDDS_SetParameters(SDDS_table, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                              "particleID",
                              (uint64_t)part[iOut][particleIDIndex],
                              "ySymmetryCode", ySymmetryCodeGlobal,
                              "zSymmetryCode", zSymmetryCodeGlobal,
                              "XLoss", (float)lossCoordinates[0],
                              "yLoss", (float)lossCoordinates[3],
                              "ZLoss", (float)lossCoordinates[1],
                              "thetaLoss", (float)lossCoordinates[2],
                              NULL)) {
        SDDS_SetError("Problem setting data in SDDS table (BRAT)");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      }
      if (brat->particleOutputSampleInterval < 1)
        brat->particleOutputSampleInterval = 1;
      if (storeData>1) {
        for (i = j = 0; i < n_stored; i += brat->particleOutputSampleInterval, j++) {
          if (!SDDS_SetRowValues(SDDS_table, SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE, j,
                                 0, (float)X_stored[i], 1, (float)Y_stored[i], 2, (float)Z_stored[i],
                                 3, (float)wX_stored[i], 4, (float)wY_stored[i], 5, (float)wZ_stored[i],
                                 6, (float)FX_stored[i], 7, (float)FY_stored[i], 8, (float)FZ_stored[i], -1)) {
            SDDS_SetError("Problem setting data in SDDS table (BRAT)");
            SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          }
        }
        if (!SDDS_WriteTable(SDDS_table)) {
          SDDS_SetError("Problem writing data in SDDS table (BRAT)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        }
      }
    }
#endif /* If not ABRAT program */
  }

  if (brat->flip) {
    for (ip=0; ip<np; ip++) {
      part[ip][0] *= -1;
      part[ip][1] *= -1;
      part[ip][2] *= -1;
      part[ip][3] *= -1;
    }
  }
  
  return itop + 1;
}

double BRAT_optim_function(double *param, long *invalid) {
  double q[9], result;
  double accelCoord[6] = {0, 0, 0, 0, 0, 0};
  double *w;

  fse = param[0];
  dXOffset = param[1];
  dZOffset = param[2];
  magnetYaw = param[3];
  if (length2dMapList > 1) {
    /* interpolate field maps */
    long i, ix, iz;
    unsigned long interpCode;
    double *xbuffer, *ybuffer;
    OUTRANGE_CONTROL outRange;
    xbuffer = tmalloc(sizeof(*xbuffer) * length2dMapList);
    ybuffer = tmalloc(sizeof(*ybuffer) * length2dMapList);
    if (bratInterpFlags & BRAT_INTERP_EXTRAPOLATE)
      outRange.flags = OUTRANGE_EXTRAPOLATE;
    else
      outRange.flags = 0;
    for (i = 0; i < length2dMapList; i++)
      xbuffer[i] = interpolationParameterValue[i];
    for (ix = 0; ix < nx; ix++)
      for (iz = 0; iz < nz; iz++) {
        for (i = 0; i < length2dMapList; i++)
          ybuffer[i] = BnormList[i][ix][iz];
        Bnorm[ix][iz] = interpolate(ybuffer, xbuffer, length2dMapList, param[4], &outRange, &outRange, interpOrder,
                                    &interpCode, 1);
        for (i = 0; i < length2dMapList; i++)
          ybuffer[i] = dBnormdxList[i][ix][iz];
        dBnormdx[ix][iz] = interpolate(ybuffer, xbuffer, length2dMapList, param[4], &outRange, &outRange, interpOrder,
                                       &interpCode, 1);
        for (i = 0; i < length2dMapList; i++)
          ybuffer[i] = dBnormdzList[i][ix][iz];
        dBnormdz[ix][iz] = interpolate(ybuffer, xbuffer, length2dMapList, param[4], &outRange, &outRange, interpOrder,
                                       &interpCode, 1);
      }
  }
  BRAT_lorentz_integration(accelCoord, q, 0, NULL);
  *invalid = 0;
  w = q + 3;
  result = sqrt(xfWeight * sqr(accelCoord[0]) + xfpWeight * sqr(accelCoord[1]));
  if (verbose_optimize && !quiet)
    printf("optim_function called: FSE = %e   dX = %e   w1/w0 = %e  sf = %e  penalty function = %e\n",
           param[0], param[1], w[1] / w[0], fabs(accelCoord[4]), result);
  return (result);
}

void BRAT_report_function(double result, double *param, long pass,
                          long n_evals, long n_dim) {
  if (!quiet) {
    printf("report for pass %ld of optimization:\n", pass);
    printf("    %ld evaluations so far\n", n_evals);
    printf("    penalty function is %e\n", result);
    printf("    FSE      = %.15e\n", param[0]);
    printf("    dX       = %.15e\n", param[1]);
    printf("    dZ       = %.15e\n", param[2]);
    printf("    yaw      = %.15e\n", param[3]);
    if (length2dMapList > 1) {
      printf(" parameter   = %.15e\n", param[4]);
    }
  }
}

#define N_EVAL_MAX 1500
#define N_PASS_MAX 3

void BRAT_optimize_magnet(unsigned long flags) {
  double x[5], dx[5], xlo[5], xhi[5];
  short disable[5] = {0, 0, 0, 0, 1}; /* fse, dx, dz, yaw, parameter */
  double tolerance, result;
  long dummy;

  tolerance = 1e-10;
  x[0] = fse;
  x[1] = dXOffset;
  x[2] = dZOffset;
  x[3] = magnetYaw;
  x[4] = 0;
  dx[0] = dx[1] = dx[2] = dx[3] = dx[4] = 1e-4;
  xlo[0] = fseLimit[0];
  xhi[0] = fseLimit[1];
  xlo[1] = dxLimit[0];
  xhi[1] = dxLimit[1];
  xlo[2] = dzLimit[0];
  xhi[2] = dzLimit[1];
  xlo[3] = yawLimit[0];
  xhi[3] = yawLimit[1];
  xlo[4] = -1;
  xhi[4] = 1;
  if (length2dMapList > 1) {
    double range;
    find_min_max(&xlo[4], &xhi[4], interpolationParameterValue, length2dMapList);
    dx[4] = (range = xhi[4] - xlo[4]) * 1e-4;
    x[4] = (xhi[4] + xlo[4]) / 2;
    if (bratInterpFlags & BRAT_INTERP_EXTRAPOLATE) {
      xlo[4] = x[4] - range * 5;
      xhi[4] = x[4] + range * 5;
    }
    disable[4] = 0;
  }
  if (!(flags & OPTIMIZE_FSE))
    disable[0] = 1;
  if (!(flags & OPTIMIZE_DX))
    disable[1] = 1;
  if (!(flags & OPTIMIZE_DZ))
    disable[2] = 1;
  if (!(flags & OPTIMIZE_YAW))
    disable[3] = 1;
  if (simplexMin(&result, x, dx, xlo, xhi, disable, 5, tolerance,
                 tolerance, BRAT_optim_function, BRAT_report_function,
                 N_EVAL_MAX, N_PASS_MAX, 12, 3.0, 1.0, 0) < 0) {
    printWarningForTracking("optimization of magnet failed", NULL);
  }
  dx[0] = dx[1] = dx[2] = dx[3] = 1e-4;
  if (length2dMapList > 1)
    dx[4] = (xhi[4] - xlo[4]) * 1e-4;
  if (simplexMin(&result, x, dx, xlo, xhi, disable, 5, tolerance,
                 tolerance, BRAT_optim_function, BRAT_report_function,
                 N_EVAL_MAX, N_PASS_MAX, 12, 3.0, 1.0, 0) < 0) {
    printWarningForTracking("optimization of magnet failed", NULL);
  }
  fse = x[0];
  dXOffset = x[1];
  dZOffset = x[2];
  magnetYaw = x[3];
  if (length2dMapList > 1)
    interpParameter = x[4];
  BMaxOnTrajectory = -DBL_MAX;
  result = BRAT_optim_function(x, &dummy);
  if (!quiet) {
    printf("penalty function after optimization: %e\n", result);
    printf("FSE = %.15e\n", x[0]);
    printf("dX  = %.15e\n", x[1] + x[2] * dxDzFactor);
    printf("dZ  = %.15e\n", x[2]);
    printf("Yaw = %.15e\n", x[3]);
    if (length2dMapList > 1)
      printf("Parameter = %.15e\n", x[4]);
  }
}

#define MIN_N_STEPS 100

#ifdef USE_GSL
#  include "gsl/gsl_poly.h"
#endif

long BRAT_lorentz_integration(
  double *accelCoord,
  double *q, /* z, x, y, dz/dS, dx/dS, dy/dS, dF/dS */
  long doStoreData,
  double *BMaxReturn) {
  long n_eq = 10;
  double exvalue, qptest[10];
  long misses[10], accmode[10];
  double accuracy[10], tiny[10];
  double hrec, hmax, dSds;
  double s_start = 0, s_end, exit_toler, ds, dz;
  long int_return;
  double *w, *IF;
  /* double xStart, dx; */
  double zStart, slope, phi;

  BMaxOnTrajectory = -DBL_MAX;
  BSignOnTrajectory = 0;

  global_delta = accelCoord[5];
  particle_inside = 0;
  n_stored = 0;
  /* drift to parallel plane for start of integration */
  slope = (xVertex - xNomEntry) / (zVertex - zNomEntry);
  phi = atan(slope);
  if (idealMode) {
    dz = idealChord;
    /* dx = slope*dz; */
  } else {
    dz = zf - zi;
    /* dx = slope*dz; */
  }
  /* xStart = xNomEntry - dx; */
  zStart = zNomEntry - dz;

  /* Compute drift-back distance and apply to path length */
  /* ds = (zStart-(zNomEntry+accelCoord[0]*sin(phi)))/(cos(phi+atan(accelCoord[1]))*cos(atan(accelCoord[3]))); */
  /* New expression from R. Lindberg */
  ds = (zStart - (zNomEntry - accelCoord[0] * sin(phi))) * sqrt(1 + sqr(accelCoord[1]) + sqr(accelCoord[3])) / (cos(phi) - accelCoord[1] * sin(phi));
  accelCoord[4] += ds;

#ifdef DEBUG
  fprintf(stderr, "drift-back: %e\n", ds);
#endif

  /* transform from accel coords to cartesian coords */

  /* note that w0^2+w1^2+w2^2 = 1 */
  /* dS/ds is the rate of change of distance traveled w.r.t. distance of the fiducial particle */
  dSds = sqrt(1 + sqr(accelCoord[1]) + sqr(accelCoord[3]));
  w = q + 3;
  w[0] = (-accelCoord[1] * sin(phi) + cos(phi)) / dSds;
  w[1] = (accelCoord[1] * cos(phi) + sin(phi)) / dSds;
  w[2] = accelCoord[3] / dSds;

  q[0] = zNomEntry - accelCoord[0] * sin(phi) + ds * w[0];
  q[1] = xNomEntry + accelCoord[0] * cos(phi) + ds * w[1];
  q[2] = accelCoord[2] + ds * w[2];

#ifdef DEBUG
  fprintf(stderr, "initial: q[0] = %21.15e, q[1] = %21.15e, q[2] = %21.15e\n",
          q[0], q[1], q[2]);
#endif

  IF = q + 6;
  IF[0] = IF[1] = IF[2] = IF[3] = 0;
  BRAT_deriv_function(qptest, q, 0.0L);
  if (qptest[3] != 0 || qptest[4] != 0 || qptest[5] != 0)
    bomb("particle started inside the magnet!", NULL);
  isLost = 0;

  if (!useFTABLE) {
    fill_long_array(accmode, n_eq, 2);
    fill_double_array(tiny, n_eq, TOLERANCE_FACTOR);
    fill_double_array(accuracy, n_eq, integ_tol);
    fill_long_array(misses, n_eq, 0);

    /* use adaptive integration */
    s_start = 0;
    s_end = central_length * 2;
    hmax = fabs(rhoMax * theta) / MIN_N_STEPS;
    hrec = hmax / 10;
    if (idealMode)
      zEnd = 2 * idealChord;
    else
      zEnd = zf + (zf - zi);
    if ((exit_toler = sqr(integ_tol) * s_end) < central_length * TOLERANCE_FACTOR)
      exit_toler = central_length * TOLERANCE_FACTOR;
    switch (int_return = bs_odeint(q, BRAT_deriv_function, n_eq, accuracy,
                                   accmode, tiny, misses,
                                   &s_start, s_end, exit_toler, hrec, hmax, &hrec, BRAT_exit_function,
                                   exit_toler, 0, doStoreData ? BRAT_store_data : NULL)) {
    case DIFFEQ_ZERO_STEPSIZE:
    case DIFFEQ_CANT_TAKE_STEP:
    case DIFFEQ_OUTSIDE_INTERVAL:
    case DIFFEQ_XI_GT_XF:
      printf("Integration failure---may be program bug\n");
      exit(1);
      break;
    case DIFFEQ_END_OF_INTERVAL:
      break;
    default:
      if (!isLost) {
        if ((exvalue = BRAT_exit_function(NULL, q, 0.0L)) > exit_toler)
          bomb("error: exit value of exceeds tolerance for BRAT (please report to developers)", NULL);
      }
      break;
    }
#ifdef DEBUG
    fprintf(stderr, "final: q[0] = %21.15e, q[1] = %21.15e, q[2] = %21.15e\n",
            q[0], q[1], q[2]);
#endif
  } else {
    /* FTABLE mode */
#define FTABLE_WORKING 1
#ifdef FTABLE_WORKING
    double eomc;
    double **A, BA, pA, step, p[3], xyz0[3], xyz[3], p0, B[3], rho;
    double theta, theta0, theta1, theta2, tm_a, tm_b, tm_c, pathLength, zStop;
    long nKicks;
    nKicks = useFTABLE;
    eomc = -particleCharge / particleMass / c_mks;
    step = 2 * (zf - zi) / nKicks;
    A = (double **)czarray_2d(sizeof(double), 3, 3);

    /* 1. get particle's coordinates */
    p0 = (1. + accelCoord[5]) * Po;
    /* working coordinate order is (X, Y, Z), whereas BRAT uses (Z, X, Y) */
    p[2] = w[0] * p0;
    p[0] = w[1] * p0;
    p[1] = w[2] * p0;
    xyz0[2] = q[0];
    xyz0[0] = q[1];
    xyz0[1] = q[2];
    pathLength = 0;
    B[0] = B[1] = B[2] = 0;
    zStop = -q[0];
    if (zStop < zf)
      zStop = zf;

    while (xyz0[2] <= zStop && !isLost) {
      BRAT_store_data_permuted(p0, p, xyz0, B, pathLength, 0.0);
      /* 2. get field at the middle point */
      xyz[0] = xyz0[0] + p[0] / p[2] * step / 2.0;
      xyz[1] = xyz0[1] + p[1] / p[2] * step / 2.0;
      xyz[2] = xyz0[2] + step / 2.0;
      BRAT_B_field_permuted(B, xyz);
#  ifdef DEBUG
      fprintf(stderr, "xyz: %le, %le, %le   pxyz: %le, %le, %le  B: %le, %le, %le\n",
              xyz[0], xyz[1], xyz[2],
              p[0], p[1], p[2],
              B[0], B[1], B[2]);
#  endif
      BA = sqrt(sqr(B[0]) + sqr(B[1]) + sqr(B[2]));
      if (BA > BMaxOnTrajectory) {
        BMaxOnTrajectory = BA;
        BSignOnTrajectory = SIGN(B[2]);
      }
      /* 3. calculate the rotation matrix */
      A[0][0] = -(p[1] * B[2] - p[2] * B[1]);
      A[0][1] = -(p[2] * B[0] - p[0] * B[2]);
      A[0][2] = -(p[0] * B[1] - p[1] * B[0]);
      pA = sqrt(sqr(A[0][0]) + sqr(A[0][1]) + sqr(A[0][2]));
      /* When field not equal to zero or not parallel to the particles motion */
      if (BA > 1e-8 && pA) {
        A[0][0] /= pA;
        A[0][1] /= pA;
        A[0][2] /= pA;
        A[1][0] = B[0] / BA;
        A[1][1] = B[1] / BA;
        A[1][2] = B[2] / BA;
        A[2][0] = A[0][1] * A[1][2] - A[0][2] * A[1][1];
        A[2][1] = A[0][2] * A[1][0] - A[0][0] * A[1][2];
        A[2][2] = A[0][0] * A[1][1] - A[0][1] * A[1][0];

        /* 4. rotate coordinates from (x,y,z) to (u,v,w) with u point to BxP, v point to B */
        rotate_coordinate(A, p, 0);
        if (p[2] < 0)
          bombElegant("Table function doesn't support particle going backward", NULL);
        rotate_coordinate(A, B, 0);

        /* 5. apply kick */
        rho = p[2] / (eomc * B[1]);
        theta0 = theta1 = theta2 = 0.;
        if (A[2][2]) {
          tm_a = 3.0 * A[0][2] / A[2][2];
          tm_b = -6.0 * A[1][2] * p[1] / p[2] / A[2][2] - 6.0;
          tm_c = 6.0 * step / rho / A[2][2];
#  ifdef USE_GSL
          gsl_poly_solve_cubic(tm_a, tm_b, tm_c, &theta0, &theta1, &theta2);
#  else
          bombElegant("gsl_poly_solve_cubic function is not available because this version of elegant was not built with the gsl library", NULL);
#  endif
        } else if (A[0][2]) {
          tm_a = A[1][2] * p[1] / p[2] + A[2][2];
          theta0 = (tm_a - sqrt(sqr(tm_a) - 2. * A[0][2] * step / rho)) / A[0][2];
          theta1 = (tm_a + sqrt(sqr(tm_a) - 2. * A[0][2] * step / rho)) / A[0][2];
        } else {
          tm_a = A[1][2] * p[1] / p[2] + A[2][2];
          theta0 = step / rho / tm_a;
        }
        theta = choose_theta(rho, theta0, theta1, theta2);

        p[0] = -p[2] * sin(theta);
        p[2] *= cos(theta);
        xyz[0] = rho * (cos(theta) - 1);
        xyz[1] = (p[1] / p[2]) * rho * theta;
        xyz[2] = rho * sin(theta);

        /* 6. rotate back to (x,y,z) */
        rotate_coordinate(A, xyz, 1);
        rotate_coordinate(A, p, 1);
        xyz0[0] += xyz[0];
        xyz0[1] += xyz[1];
        pathLength += sqrt(sqr(rho * theta) + sqr(xyz[1]));
        xyz0[2] += step;
      } else {
        xyz0[0] += p[0] / p[2] * step;
        xyz0[1] += p[1] / p[2] * step;
        pathLength += step;
        xyz0[2] += step;
      }
    }
#endif
    BRAT_store_data_permuted(p0, p, xyz0, B, pathLength, 0.0);
    /* convert back to (Z, X, Y, WZ, WX, WY) */
    w[0] = p[2] / p0;
    w[1] = p[0] / p0;
    w[2] = p[1] / p0;
    q[0] = xyz0[2];
    q[1] = xyz0[0];
    q[2] = xyz0[1];
    s_start = pathLength;
  }

  if (!isLost) {
    /* drift back to reference plane */
    slope = (xVertex - xNomExit) / (zVertex - zNomExit);
    phi = -atan(slope);
    ds = ((zNomExit - q[0]) * cos(phi) - (xNomExit - q[1]) * sin(phi)) / (w[0] * cos(phi) - w[1] * sin(phi));
#ifdef DEBUG
    printf("exit: phi = %le, ds = %le\n", phi, ds);
#endif
    q[0] += ds * w[0];
    q[1] += ds * w[1];
    q[2] += ds * w[2];

    /* convert back to accelerator coordinates */
    dSds = 1. / (-w[1] * sin(phi) + w[0] * cos(phi));

    accelCoord[1] = (w[1] * cos(phi) + w[0] * sin(phi)) * dSds;
    accelCoord[3] = w[2] * dSds;
    accelCoord[0] = (q[0] - zNomExit) / sin(phi);
    accelCoord[2] = q[2];
    accelCoord[4] += s_start + ds;
  } else {
    /* some accelerator coordinates are unknown because a loss occurred inside
     * the integration region and we haven't worked out how to compute the
     * accelerator coordinates for those cases
     */
    accelCoord[0] = accelCoord[1] = 999; /* my modest homage to fortran */
    accelCoord[4] += s_start;
  }

  if (BMaxReturn)
    *BMaxReturn = BMaxOnTrajectory * BSignOnTrajectory;

#ifdef DEBUG
  fprintf(stderr, "final: x=%e, xp=%e, y=%e, yp=%e\n",
          accelCoord[0], accelCoord[1], accelCoord[2], accelCoord[3]);
#endif

#ifdef DEBUG
  fprintf(stderr, "path length: %e\n", s_start);
  fprintf(stderr, "drift back exit: %e\n", ds);
  fprintf(stderr, "ref plane: x=%e, xp=%e, y=%e, yp=%e, s=%e\n",
          accelCoord[0], accelCoord[1], accelCoord[2], accelCoord[3], accelCoord[4]);
#endif

  return isLost;
}

#undef DEBUG

double BRAT_exit_function(double *qp, double *q, double s) {
  if (isLost)
    return 0.0;
  return q[0] - zEnd;
}

void BRAT_deriv_function(double *qp, double *q, double s) {
  double *F;
  static double *w, *wp, factor, BA;

  F = qp + 6;
  BRAT_B_field(F, q);
  BA = sqrt(F[0] * F[0] + F[1] * F[1] + F[2] * F[2]);
  if (BA > BMaxOnTrajectory) {
    BMaxOnTrajectory = BA;
    BSignOnTrajectory = SIGN(F[2]);
  }
  F[3] = (1 - F[2]) * F[2];
  w = q + 3;
  wp = qp + 3;
  qp[0] = w[0];
  qp[1] = w[1];
  qp[2] = w[2];
  factor = 1 / (rigidity * (1 + global_delta));
  wp[0] = (w[2] * F[1] - w[1] * F[2]) * factor;
  wp[1] = (w[0] * F[2] - w[2] * F[0]) * factor;
  wp[2] = (w[1] * F[0] - w[0] * F[1]) * factor;
}

void BRAT_store_data_permuted(double p0, double *p, double *q1, double *B, double s, double exval) {
  double q[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, qp[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  q[0] = q1[2];
  q[1] = q1[0];
  q[2] = q1[1];
  q[4] = p[2] / p0;
  q[5] = p[0] / p0;
  q[6] = p[1] / p0;
  qp[6] = B[2];
  qp[7] = B[0];
  qp[8] = B[1];
  BRAT_store_data(qp, q, s, 0.0);
}

void BRAT_store_data(double *qp, double *q, double s, double exval) {
  if (fabs(q[0]) > 2 * z_outer || !storeData)
    return;
#if DEBUG
  printf("Storing particle and field data on trajectory\n");
  fflush(stdout);
#endif
  if (n_stored >= max_store) {
    max_store += 100;
    X_stored = trealloc(X_stored, sizeof(*X_stored) * max_store);
    Y_stored = trealloc(Y_stored, sizeof(*Y_stored) * max_store);
    Z_stored = trealloc(Z_stored, sizeof(*Z_stored) * max_store);
    FX_stored = trealloc(FX_stored, sizeof(*FX_stored) * max_store);
    FY_stored = trealloc(FY_stored, sizeof(*FY_stored) * max_store);
    FZ_stored = trealloc(FZ_stored, sizeof(*FZ_stored) * max_store);
    if (storeData > 1) {
      wX_stored = trealloc(wX_stored, sizeof(*X_stored) * max_store);
      wY_stored = trealloc(wY_stored, sizeof(*Y_stored) * max_store);
      wZ_stored = trealloc(wZ_stored, sizeof(*Z_stored) * max_store);
      if (storeData > 2) {
        aX_stored = trealloc(aX_stored, sizeof(*aX_stored) * max_store);
        aY_stored = trealloc(aY_stored, sizeof(*aY_stored) * max_store);
        aZ_stored = trealloc(aZ_stored, sizeof(*aZ_stored) * max_store);
        Fint_stored = trealloc(Fint_stored, sizeof(*Fint_stored) * max_store);
        s_stored = trealloc(s_stored, sizeof(*s_stored) * max_store);
      }
    }
  }
  Z_stored[n_stored] = q[0];
  X_stored[n_stored] = q[1];
  Y_stored[n_stored] = q[2];
  FZ_stored[n_stored] = qp[6];
  FX_stored[n_stored] = qp[7];
  FY_stored[n_stored] = qp[8];
  if (storeData > 1) {
    wZ_stored[n_stored] = q[3];
    wX_stored[n_stored] = q[4];
    wY_stored[n_stored] = q[5];
    if (storeData > 2) {
      aZ_stored[n_stored] = qp[3];
      aX_stored[n_stored] = qp[4];
      aY_stored[n_stored] = qp[5];
      Fint_stored[n_stored] = q[9] / gap / 2;
      s_stored[n_stored] = s;
    }
  }
  n_stored++;
#if DEBUG
  printf("%ld data points now stored\n", n_stored);
  fflush(stdout);
#endif
}

double BRAT_setup_arc_field_data(char *input, char *sName, char *fieldName, double xCenter) {
  SDDS_DATASET SDDS_table;
  long iData, increasing;
  double BRef, BMin, BMax, sMin, sMax;

  if (!SDDS_InitializeInputFromSearchPath(&SDDS_table, input) || !SDDS_ReadPage(&SDDS_table)) {
    SDDS_SetError("Unable to read data file");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  if (!(nArcPoints = SDDS_CountRowsOfInterest(&SDDS_table)))
    bomb("no data in field file", NULL);
  if (nArcPoints < 2)
    bomb("too little data in field file", NULL);

  if (SDDS_CheckColumn(&SDDS_table, sName, "m", SDDS_ANY_FLOATING_TYPE, stderr) != SDDS_CHECK_OKAY)
    exit(1);
  if (SDDS_CheckColumn(&SDDS_table, fieldName, "T", SDDS_ANY_FLOATING_TYPE, stderr) != SDDS_CHECK_OKAY)
    exit(1);

  if (!(sArc = SDDS_GetColumnInDoubles(&SDDS_table, sName)) ||
      !(BArcNorm = SDDS_GetColumnInDoubles(&SDDS_table, fieldName))) {
    SDDS_SetError("Unable to retrieve data columns");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  if (!(dBArcNormds = malloc(sizeof(*dBArcNormds) * nArcPoints))) {
    fprintf(stderr, "brat: memory allocation failure\n");
    exit(1);
  }

  increasing = 0;
  if (sArc[0] < sArc[1])
    increasing = 1;
  BMin = -(BMax = -DBL_MAX);
  sMin = -(sMax = -DBL_MAX);
  for (iData = 0; iData < nArcPoints; iData++) {
    if (BArcNorm[iData] > BMax)
      BMax = BArcNorm[iData];
    if (BArcNorm[iData] < BMin)
      BMin = BArcNorm[iData];
    if (sArc[iData] > sMax)
      sMax = sArc[iData];
    if (sArc[iData] < sMin)
      sMin = sArc[iData];
    if (iData == 0)
      continue;
    if ((increasing && sArc[iData - 1] >= sArc[iData]) ||
        (!increasing && sArc[iData - 1] <= sArc[iData]))
      bomb("arc s values not monotonic", NULL);
  }

  if (fabs(BMin) > fabs(BMax))
    BRef = BMin;
  else
    BRef = BMax;
  if (sMin != 0) {
    fprintf(stderr, "minimum value of s is %le, but must be zero for arc data\n",
            sMin);
    exit(1);
  }

  if (BRef == 0)
    bomb("field appears to be zero in arc data", NULL);
  /*
  for (iData=0; iData<nArcPoints; iData++)
    BArcNorm[iData] /= BRef;
  */

  dBArcNormds[0] = (BArcNorm[1] - BArcNorm[0]) / (sArc[1] - sArc[0]);
  dBArcNormds[nArcPoints - 1] = (BArcNorm[nArcPoints - 1] - BArcNorm[nArcPoints - 2]) / (sArc[nArcPoints - 1] - sArc[nArcPoints - 2]);

  for (iData = 1; iData < (nArcPoints - 1); iData++) {
    dBArcNormds[iData] = (BArcNorm[iData + 1] - BArcNorm[iData - 1]) / (sArc[iData + 1] - sArc[iData - 1]);
  }

  return BRef;
}

double BRAT_setup_field_data(char *input, double xCenter, double zCenter, char *interpolationParameter) {
  SDDS_TABLE SDDS_table;
  long idata, ix, iz, rows;
  double x, z, B, *xd = NULL, *zd = NULL, *Bd = NULL, xCheck, zCheck;
  double *yd = NULL, *Bxd = NULL, *Byd = NULL, *Bzd = NULL;
  double Bmin, Bmax;
  long ix0, ix1, iz0, iz1;
  double dz0 = 0, dx0 = 0;
#ifdef DEBUG
  double xc;
#endif

#if USE_MPI
  if (myid==1)
    dup2(fdStdout, fileno(stdout));
#endif
  
  printf("Reading BRAT field data from %s\n", input);
  fflush(stdout);

  if (!SDDS_InitializeInputFromSearchPath(&SDDS_table, input) || !SDDS_ReadPage(&SDDS_table)) {
    SDDS_SetError("Unable to read data file");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  if (!(rows = SDDS_CountRowsOfInterest(&SDDS_table)))
    bomb("no data in field file", NULL);

  clear2dMapList();

  if (fieldMapDimension == 2) {
    long firstPage = 1;
    double interpParamValue;
    interpolationParameterUnits = NULL;
    interpParameter = 0;
    if (!quiet) {
      printf("2D field map mode\n");
      fflush(stdout);
    }
    do {
      if (!single_scan) {
        if (firstPage) {
          if (interpolationParameter) {
            if (SDDS_CheckParameter(&SDDS_table, interpolationParameter, NULL, SDDS_ANY_FLOATING_TYPE, stderr) != SDDS_CHECK_OKAY)
              exit(1);
            if (SDDS_GetParameterInformation(&SDDS_table, "units", &interpolationParameterUnits, SDDS_GET_BY_NAME, interpolationParameter) != SDDS_STRING)
              bomb("unable to get units for interpolation parameter", NULL);
          }
          if (SDDS_CheckColumn(&SDDS_table, "x", "m", SDDS_ANY_FLOATING_TYPE, stderr) != SDDS_CHECK_OKAY ||
              SDDS_CheckColumn(&SDDS_table, "z", "m", SDDS_ANY_FLOATING_TYPE, stderr) != SDDS_CHECK_OKAY ||
              SDDS_CheckColumn(&SDDS_table, "B", "T", SDDS_ANY_FLOATING_TYPE, stderr) != SDDS_CHECK_OKAY) {
            exit(1);
          }
          if (SDDS_CheckParameter(&SDDS_table, "xMinimum", "m", SDDS_DOUBLE, stderr) != SDDS_CHECK_OKAY ||
              SDDS_CheckParameter(&SDDS_table, "xInterval", "m", SDDS_DOUBLE, stderr) != SDDS_CHECK_OKAY ||
              SDDS_CheckParameter(&SDDS_table, "xDimension", NULL, SDDS_LONG, stderr) != SDDS_CHECK_OKAY ||
              SDDS_CheckParameter(&SDDS_table, "zMinimum", "m", SDDS_DOUBLE, stderr) != SDDS_CHECK_OKAY ||
              SDDS_CheckParameter(&SDDS_table, "zInterval", "m", SDDS_DOUBLE, stderr) != SDDS_CHECK_OKAY ||
              SDDS_CheckParameter(&SDDS_table, "zDimension", NULL, SDDS_LONG, stderr) != SDDS_CHECK_OKAY)
            exit(1);
        }
        if (!(xd = (double *)SDDS_GetColumnInDoubles(&SDDS_table, "x")) ||
            !(zd = (double *)SDDS_GetColumnInDoubles(&SDDS_table, "z")) ||
            !((Bd = (double *)SDDS_GetColumnInDoubles(&SDDS_table, "B")))) {
          SDDS_SetError("Unable to retrieve data column(s)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
        xi = xf = dx = 0;
        zi = zf = dz = 0;
        nx = nz = 0;

        if (!SDDS_GetParameter(&SDDS_table, "xMinimum", &xi) || !SDDS_GetParameter(&SDDS_table, "xDimension", &nx) ||
            !SDDS_GetParameter(&SDDS_table, "xInterval", &dx) ||
            !SDDS_GetParameter(&SDDS_table, "zMinimum", &zi) || !SDDS_GetParameter(&SDDS_table, "zDimension", &nz) ||
            !SDDS_GetParameter(&SDDS_table, "zInterval", &dz)) {
          SDDS_SetError("Unable to retrieve parameter value(s)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
        if (interpolationParameter && !SDDS_GetParameterAsDouble(&SDDS_table, interpolationParameter, &interpParamValue)) {
          SDDS_SetError("Unable to retrieve parameter value(s)");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }

        if (nx <= 0 || nz <= 0)
          bomb("non-positive value for dx and/or dz", NULL);
        xf = (nx - 1) * dx + xi;
        zf = (nz - 1) * dz + zi;
        if (nx == 1) {
          single_scan = 1;
          xi = xf = 0;
          if (!quiet) {
            printf("data treated as single scan\n");
            fflush(stdout);
          }
        } else if (!quiet) {
          printf("data grid:\n    x: [%.8f, %.8f]m   dx = %.8fm   nx = %ld\n",
                 xi, xf, dx, nx);
          fflush(stdout);
        }
        if (!quiet) {
          printf("     z: [%.8f, %.8f]m   dz = %.8fm   nz = %ld\n",
                 zi, zf, dz, nz);
          fflush(stdout);
        }
      } else {
        if (interpolationParameter)
          bomb("interpolation not supported for single scan", NULL);
        if (SDDS_CheckColumn(&SDDS_table, "z", "m", SDDS_ANY_FLOATING_TYPE, stderr) != SDDS_CHECK_OKAY)
          exit(1);
        if (SDDS_CheckColumn(&SDDS_table, "B", "T", SDDS_ANY_FLOATING_TYPE, stderr) == SDDS_CHECK_OKAY)
          exit(1);
        if (!(zd = (double *)SDDS_GetColumnInDoubles(&SDDS_table, "z"))) {
          SDDS_SetError("Unable to retrieve data z column");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
        if (!(Bd = (double *)SDDS_GetColumnInDoubles(&SDDS_table, "B"))) {
          SDDS_SetError("Unable to retrieve data field column");
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        }
        nx = 1;
        xi = xf = 0;
        zi = zd[0];
        zf = zd[rows - 1];
        nz = rows;
        if (nz < 1)
          bomb("nz < 1 in single scan data", NULL);
        dz = (zf - zi) / (nz - 1);
        if (!quiet) {
          printf("data grid:\n    ");
          printf("    z: [%.8f, %.8f]m   dz = %.8fm   nz = %ld\n",
                 zi, zf, dz, nz);
          fflush(stdout);
        }
      }

#ifdef DEBUG
      fprintf(stderr, "Allocating %ld x %ld field and gradient arrays\n",
              nx, nz);
#endif
      Bnorm = (double **)zarray_2d(sizeof(**Bnorm), nx, nz);
      dBnormdz = (double **)zarray_2d(sizeof(**dBnormdz), nx, nz);
      dBnormdx = (double **)zarray_2d(sizeof(**dBnormdx), nx, nz);
#ifdef DEBUG
      fprintf(stderr, "Initializing Bnorm array\n");
#endif
      for (ix = 0; ix < nx; ix++)
        for (iz = 0; iz < nz; iz++)
          Bnorm[ix][iz] = DBL_MAX;
      idata = 0;
#ifdef DEBUG
      xc = xi;
#endif
      Bmin = -(Bmax = -DBL_MAX);
      ix = 0;
      if (single_scan)
        dx = 1;
      for (idata = 0; idata < rows; idata++) {
#ifdef DEBUG
        fprintf(stderr, "Checking idata=%ld\n", idata);
#endif
        x = single_scan ? 0 : xd[idata];
        z = zd[idata];
        B = Bd[idata];
#ifdef DEBUG
        fprintf(stderr, "x=%e, z=%e, B=%e\n", x, z, B);
#endif
        if (!single_scan)
          ix = (x - xi) / dx + 0.5;
        iz = (z - zi) / dz + 0.5;
        xCheck = ix * dx + xi;
        zCheck = iz * dz + zi;
#ifdef DEBUG
        fprintf(stderr, "ix = %ld, iz = %ld, xc=%e, zc=%e\n", ix, iz,
                xCheck, zCheck);
#endif
        if (fabs(x - xCheck) / dx > 1e-2 || fabs(z - zCheck) / dz > 1e-2) {
          fprintf(stderr, "Grid data not uniform: (%e, %e) -> (%e, %e)\n",
                  x, z, xCheck, zCheck);
          exit(1);
        }
        if (ix < 0 || ix > nx || iz < 0 || iz > nz) {
          fprintf(stderr, "Point outside grid: (%e, %e) -> (%ld, %ld)\n",
                  x, z, ix, iz);
          exit(1);
        }
        if (Bnorm[ix][iz] != DBL_MAX)
          bomb("two data points map to one grid point", NULL);
        Bnorm[ix][iz] = B;
        if (B < Bmin)
          Bmin = B;
        if (B > Bmax)
          Bmax = B;
      }
      if (!single_scan)
        free(xd);
      free(zd);
      free(Bd);

      idata = 0;
      for (ix = 0; ix < nx; ix++)
        for (iz = 0; iz < nz; iz++) {
          if (Bnorm[ix][iz] == DBL_MAX) {
            idata++;
            Bnorm[ix][iz] = 0;
          }
        }
      if (idata && !quiet) {
        char buffer[16384];
        snprintf(buffer, 16384, "No data for %ld of %ld points\n", idata, nx * nz);
        printWarningForTracking(buffer, NULL);
      }

      if (fabs(Bmin) > fabs(Bmax))
        Bmax = Bmin;
      if (Bmax == 0)
        bomb("extremal B value is zero", NULL);
      if (!quiet) {
        printf("Maximum value of B is %.8f T\n", Bmax);
        fflush(stdout);
      }
      /* 
         if (!quiet)
         printf("reference B value is %.8f T\n", Bmax);
         for (ix=0; ix<nx; ix++)
         for (iz=0; iz<nz; iz++)
         Bnorm[ix][iz] /= Bmax;
      */

      /* compute derivatives */
      for (ix = 0; ix < nx; ix++) {
        if (nx == 1)
          ix0 = ix1 = 0;
        else {
          if (ix == 0) {
            dx0 = dx;
            ix1 = ix + 1;
            ix0 = ix;
          } else if (ix == (nx - 1)) {
            dx0 = dx;
            ix1 = ix;
            ix0 = ix - 1;
          } else {
            dx0 = 2 * dx;
            ix1 = ix + 1;
            ix0 = ix - 1;
          }
        }
        for (iz = 0; iz < nz; iz++) {
          if (iz == 0) {
            dz0 = dz;
            iz1 = iz + 1;
            iz0 = iz;
          } else if (iz == (nz - 1)) {
            dz0 = dz;
            iz1 = iz;
            iz0 = iz - 1;
          } else {
            dz0 = 2 * dz;
            iz1 = iz + 1;
            iz0 = iz - 1;
          }
          dBnormdz[ix][iz] = (Bnorm[ix][iz1] - Bnorm[ix][iz0]) / dz0;
          if (!single_scan)
            dBnormdx[ix][iz] = (Bnorm[ix1][iz] - Bnorm[ix0][iz]) / dx0;
          else
            dBnormdx[ix][iz] = 0;
        }
      }
      if (interpolationParameter)
        add2dMapList(interpParamValue);
      if (!quiet) {
        printf("Finished processing page from field file\n");
        fflush(stdout);
      }
    } while (interpolationParameter && SDDS_ReadPage(&SDDS_table) > 0 && (rows = SDDS_CountRowsOfInterest(&SDDS_table)) > 0);
    if (interpolationParameter) {
      if (length2dMapList <= 1)
        bomb("interpolation parameter was given but only one page was found in the input file", NULL);
      /* these will be used for holding the interpolated map */
      Bnorm = (double **)zarray_2d(sizeof(**Bnorm), nx, nz);
      dBnormdz = (double **)zarray_2d(sizeof(**dBnormdz), nx, nz);
      dBnormdx = (double **)zarray_2d(sizeof(**dBnormdx), nx, nz);
    }
    if (!SDDS_Terminate(&SDDS_table))
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
  } else {
    /* 3d field map */
    if (interpolationParameter)
      bomb("interpolation not supported for 3D maps", NULL);
    if (SDDS_CheckColumn(&SDDS_table, "x", "m", SDDS_ANY_FLOATING_TYPE, stderr) != SDDS_CHECK_OKAY ||
        SDDS_CheckColumn(&SDDS_table, "y", "m", SDDS_ANY_FLOATING_TYPE, stderr) != SDDS_CHECK_OKAY ||
        SDDS_CheckColumn(&SDDS_table, "z", "m", SDDS_ANY_FLOATING_TYPE, stderr) != SDDS_CHECK_OKAY) {
      exit(1);
    }
    if (SDDS_CheckColumn(&SDDS_table, "Bx", "T", SDDS_ANY_FLOATING_TYPE, stderr) != SDDS_CHECK_OKAY ||
        SDDS_CheckColumn(&SDDS_table, "By", "T", SDDS_ANY_FLOATING_TYPE, stderr) != SDDS_CHECK_OKAY ||
        SDDS_CheckColumn(&SDDS_table, "Bz", "T", SDDS_ANY_FLOATING_TYPE, stderr) != SDDS_CHECK_OKAY) {
      exit(1);
    }
    if (!(xd = SDDS_GetColumnInDoubles(&SDDS_table, "x")) ||
        !(yd = SDDS_GetColumnInDoubles(&SDDS_table, "y")) ||
        !(zd = SDDS_GetColumnInDoubles(&SDDS_table, "z")) ||
        !(Bxd = SDDS_GetColumnInDoubles(&SDDS_table, "Bx")) ||
        !(Byd = SDDS_GetColumnInDoubles(&SDDS_table, "By")) ||
        !(Bzd = SDDS_GetColumnInDoubles(&SDDS_table, "Bz"))) {
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    }

    /* It is assumed that the data is ordered so that x changes fastest.
     * This can be accomplished with sddssort -column=z,incr -column=y,incr -column=x,incr
     * The points are assumed to be equipspaced.
     */
    nx = 1;
    xi = xd[0];
    while (nx < rows) {
      if (xd[nx - 1] > xd[nx])
        break;
      nx++;
    }
    if (nx == rows) {
      fprintf(stderr, "file doesn't have correct structure or amount of data (x)\n");
      fprintf(stderr, "Use sddssort -column=z -column=y -column=x to sort the file\n");
      exit(1);
    }
    xf = xd[nx - 1];
    dx = (xf - xi) / (nx - 1);

    ny = 1;
    yi = yd[0];
    while (ny < (rows / nx)) {
      if (yd[(ny - 1) * nx] > yd[ny * nx])
        break;
      ny++;
    }
    if (ny == rows) {
      fprintf(stderr, "file doesn't have correct structure or amount of data (y)\n");
      fprintf(stderr, "Use sddssort -column=z -column=y -column=x to sort the file\n");
      exit(1);
    }
    yf = yd[(ny - 1) * nx];
    dy = (yf - yi) / (ny - 1);

    if (nx <= 1 || ny <= 1 || (nz = rows / (nx * ny)) <= 1) {
      fprintf(stderr, "file doesn't have correct structure or amount of data (z)\n");
      fprintf(stderr, "Use sddssort -column=z -column=y -column=x to sort the file\n");
      exit(1);
    }
    zi = zd[0];
    zf = zd[rows - 1];
    dz = (zf - zi) / (nz - 1);

    Bmin = -(Bmax = -DBL_MAX);
    for (idata = 0; idata < rows; idata++) {
      /* if (fabs(yd[idata])<dy/2 && fabs(xd[idata])<dx/2) { */
      if (Byd[idata] > Bmax)
        Bmax = Byd[idata];
      if (Byd[idata] < Bmin)
        Bmin = Byd[idata];
      /* } */
    }
    free(xd);
    free(yd);
    free(zd);
    if (Bmax == -DBL_MAX) {
      fprintf(stderr, "BRAT file doesn't have valid Bmax value\n");
      fprintf(stderr, "Bmax = %le, Bmin = %le\n", Bmax, Bmin);
      exit(1);
    }
    if (fabs(Bmin) > fabs(Bmax))
      SWAP_DOUBLE(Bmin, Bmax);
    BxNorm = Bxd;
    ByNorm = Byd;
    BzNorm = Bzd;

    printf("3D field map data: nx=%ld, ny=%ld, nz=%ld\ndx=%e, dy=%e, dz=%e\nx:[%e, %e], y:[%e, %e], z:[%e, %e]\nBy:[%e, %e]\n",
           nx, ny, nz, dx, dy, dz,
           xi, xf, yi, yf, zi, zf,
           Bmin, Bmax);
  }

#if USE_MPI
  if (myid==1) {
#if defined(_WIN32)
    freopen("NUL", "w", stdout);
#else
    if (!freopen("/dev/null", "w", stdout)) {
      perror("freopen failed");
      exit(EXIT_FAILURE);
    }
#endif
  }
#endif
  
  return Bmax;
}

void BRAT_B_field_permuted(
  double *Fp, /* Output: Bx, By, Bz */
  double *Qp  /* Input: X, Y, Z */
) {
  double Q[3];
  double F[3];
  Q[0] = Qp[2];
  Q[1] = Qp[0];
  Q[2] = Qp[1];
  BRAT_B_field(F, Q); 
  Fp[0] = F[1]; /* Bx */
  Fp[1] = F[2]; /* By */
  Fp[2] = F[0]; /* Bz */
}

void BRAT_B_field(double *F, double *Qg) {
  long ix, iy, iz, j, outside;
  double fx, fy, fz, val_z1, val_z2, f[3];
#ifndef ABRAT_PROGRAM
  double xSlope;
#endif
  double x, y, z, derivSign = 1;
  double Q[3];
  short symmetryFactor[3] = {1, 1, 1}; /* Bz, Bx, By */

  memcpy(Q, Qg, 3 * sizeof(*Q));
  if (idealMode) {
    double Lx;
    Lx = idealChord + 2 * Qg[1] * tan(idealEdgeAngle * PI / 180);
    if (fabs(Qg[0]) > Lx / 2) {
      F[0] = F[1] = F[2] = 0;
    } else {
      F[0] = F[1] = 0;
      F[2] = 1;
    }
    for (j = 0; j < 3; j++)
      F[j] *= fieldSign * (1 + fse);
    return;
  }
  if (arc_scan) {
    double phi, s, dByds;
    /* double R; */
    long inStraight;
    OUTRANGE_CONTROL belowRange = {0.0, OUTRANGE_SATURATE};
    OUTRANGE_CONTROL aboveRange = {0.0, OUTRANGE_VALUE};
    unsigned long code;
    phi = fabs(atan2(Q[0], Q[1]));
#ifdef DEBUG
    fprintf(stderr, "computing arc field for Z=%e, X=%e, phi=%e: ",
            Q[0], Q[1], phi);
#endif
    if (phi <= theta / 2) {
      s = phi * rhoArc;
      inStraight = 0;
    } else {
      /* R = sqrt(sqr(Q[0])+sqr(Q[1])); */
      s = rhoArc * ((theta / 2) + tan(phi - theta / 2));
      inStraight = 1;
    }
    F[2] = interpolate(BArcNorm, sArc, nArcPoints, s, &belowRange, &aboveRange, 1, &code, 1.0);
    dByds = interpolate(dBArcNormds, sArc, nArcPoints, s, &belowRange, &aboveRange, 1, &code, 1.0);
    if (inStraight) {
      phi = theta / 2;
    }
    F[0] = SIGN(Q[0]) * Q[2] * dByds * cos(phi);
    F[1] = -Q[2] * dByds * sin(phi);
#ifdef DEBUG
    fprintf(stderr, "s=%e, B=%e\n", s, F[2]);
#endif
    if (Q[0] >= zNomEntry && Q[0] <= zNomExit && deltaByInside)
      F[2] += deltaByInside;
    for (j = 0; j < 3; j++)
      F[j] *= fieldSign * (1 + fse);
    return;
  }

#ifndef ABRAT_PROGRAM
  z = Q[0];
  x = Q[1];
  y = Q[2];
  lossCoordinates[3] = y; /* just in case it is needed */
  xSlope = 0;
  if (Qg[3] != 0)
    xSlope = Qg[4] / Qg[3];
  if (!isLost &&
      (z >= zNomEntry && z <= zNomExit) &&
      insideObstruction_XYZ(x, y, z, xNomEntry, 0.0, zNomEntry, thetaEntry, xSlope, lossCoordinates)) {
    /*
    printf("Loss coordinates: X = %le, Z = %le, theta = %le, Q[3] = %le, Q[4] = %le, Q[5] = %le\n",
           lossCoordinates[0], lossCoordinates[1], lossCoordinates[2],
           Qg[3], Qg[4], Qg[5]);
    fflush(stdout);
    */
    isLost = 1;
  }
#endif

  if (magnetYaw) {
    double Qy[2];
    Qy[0] = Q[0];
    Qy[1] = Q[1];
    Q[0] = Qy[0] * cos(magnetYaw) + Qy[1] * sin(magnetYaw);
    Q[1] = -Qy[0] * sin(magnetYaw) + Qy[1] * cos(magnetYaw);
  }
  z = Q[0];
  if (zDuplicate && z > 0) {
    z = -fabs(z);
    derivSign = -1;
  }

  x = Q[1] - dXOffset + dZOffset * dxDzFactor;
  y = Q[2] - dYOffset;
  if (y<0 && ySymmetryCodeGlobal) {
    if (ySymmetryCodeGlobal==1) {
      /* odd magnet symmetry w.r.t. y, e.g., a normal quad */
      symmetryFactor[1] *= -1; /* Bx */
      symmetryFactor[0] *= -1; /* Bz */
    } else if (ySymmetryCodeGlobal==2) { 
      /* even magnet symmetry w.r.t. y, e.g., a skew quad */
      symmetryFactor[2] *= -1; /* By */
    }
    y *= -1;
  }

  z -= dZOffset;
  if (z<0 && zSymmetryCodeGlobal) {
    if (zSymmetryCodeGlobal==1) {
      /* odd magnet symmetry w.r.t. z */
      symmetryFactor[1] *= -1; /* Bx */
      symmetryFactor[2] *= -1; /* By */
    } else if (zSymmetryCodeGlobal==2) { 
      /* even magnet symmetry w.r.t. z */
      symmetryFactor[0] *= -1; /* Bz */
    }
    z *= -1;
  }

  F[0] = F[1] = F[2] = 0;

#ifdef DEBUG
  fprintf(stderr, "Finding field at x=%e (%e) y=%e z=%e\n", x, Q[1], y, z);
#endif

  if (!single_scan) {
    if (!particle_inside) {
      if ((xi - x) > dx * 1e-6 || (x - xf) > dx * 1e-6) {
#ifdef DEBUG
        fprintf(stderr, "Particle outside grid [%le, %le] in x\n", xi, xf);
#endif
        return;
      }
      if ((zi - z) > dz * 1e-6 || (z - zf) > dz * 1e-6) {
#ifdef DEBUG
        fprintf(stderr, "Particle outside grid [%le, %le] in z\n", zi, zf);
#endif
        return;
      }
      if (fieldMapDimension == 3 &&
          ((yi - y) > dy * 1e-6 || (y - yf) > dy * 1e-6)) {
#ifdef DEBUG
        fprintf(stderr, "Particle outside grid [%le, %le] in y\n", yi, yf);
#endif
        return;
      }
    }

    particle_inside = 1;
    if (!extend_data) {
      if ((xi - x) > dx * 1e-6 || (x - xf) > dx * 1e-6)
        return;
      if ((zi - z) > dz * 1e-6 || (z - zf) > dz * 1e-6)
        return;
      if (fieldMapDimension == 3 &&
          ((yi - y) > dy * 1e-6 || (y - yf) > dy * 1e-6))
        return;
    }
  } else {
    if (extend_data && extend_edge_angle)
      z = z - x * tan(extend_edge_angle);
    if ((zi - z) > dz * 1e-6 || (z - zf) > dz * 1e-6)
      return;
  }

  iz = (z - zi) / dz;
  if (iz >= nz - 1) {
    iz = nz - 2;
    z = zf;
  }
  if (iz < 0) {
    iz = 0;
    z = zi;
  }
  fz = (z - zi) / dz - iz;

  if (single_scan) {
    F[2] = Bnorm[0][iz] + fz * (Bnorm[0][iz + 1] - Bnorm[0][iz]);
    if (y) {
      F[0] = derivSign * y * (dBnormdz[0][iz] + fz * (dBnormdz[0][iz + 1] - dBnormdz[0][iz]));
      F[1] = y * (dBnormdx[0][iz] + fz * (dBnormdx[0][iz + 1] - dBnormdx[0][iz]));
    }
    if (z >= zNomEntry && z <= zNomExit && deltaByInside)
      F[2] += deltaByInside;
    for (j = 0; j < 3; j++)
      F[j] *= fieldSign * (1 + fse);
    return;
  }

  outside = 0;
  ix = (x - xi) / dx;
  if (ix >= nx - 1) {
    ix = nx - 2;
    x = xf;
    outside = 1;
  }
  if (ix < 0) {
    ix = 0;
    x = xi;
    outside = 1;
  }
  fx = (x - xi) / dx - ix;

  if (fieldMapDimension == 3) {
    iy = (y - yi) / dy;
    if (iy >= ny - 1) {
      iy = ny - 2;
      y = yf;
      outside = 1;
    }
    if (iy < 0) {
      iy = 0;
      y = yi;
      outside = 1;
    }
    fy = (y - yi) / dy - iy;
  } else {
    iy = 0;
    fy = 0;
  }

  if (outside && !xyExtrapolate) {
    for (j = 0; j < 3; j++)
      F[j] = 0;
    return;
  }

  if (fieldMapDimension == 2) {
    val_z1 = Bnorm[ix][iz] + fx * (Bnorm[ix + 1][iz] - Bnorm[ix][iz]);
    val_z2 = Bnorm[ix][iz + 1] + fx * (Bnorm[ix + 1][iz + 1] - Bnorm[ix][iz + 1]);
    F[2] = val_z1 + fz * (val_z2 - val_z1);
    val_z1 = derivSign * (dBnormdz[ix][iz] + fx * (dBnormdz[ix + 1][iz] - dBnormdz[ix][iz]));
    val_z2 = derivSign * (dBnormdz[ix][iz + 1] + fx * (dBnormdz[ix + 1][iz + 1] - dBnormdz[ix][iz + 1]));
    f[0] = y * (val_z1 + fz * (val_z2 - val_z1));
    val_z1 = dBnormdx[ix][iz] + fx * (dBnormdx[ix + 1][iz] - dBnormdx[ix][iz]);
    val_z2 = dBnormdx[ix][iz + 1] + fx * (dBnormdx[ix + 1][iz + 1] - dBnormdx[ix][iz + 1]);
    f[1] = y * (val_z1 + fz * (val_z2 - val_z1));
    F[0] = f[0];
    F[1] = f[1];
    if (z >= zNomEntry && z <= zNomExit && deltaByInside)
      F[2] += deltaByInside;
  } else {
    long iq;
    double Finterp1[2][2], Finterp2[2];
    double *Fq[3], Freturn[3] = {0, 0, 0};
    float *Fq1[3];
    Fq[0] = BzNorm;
    Fq[1] = BxNorm;
    Fq[2] = ByNorm;
    Fq1[0] = Bz1Norm;
    Fq1[1] = Bx1Norm;
    Fq1[2] = By1Norm;

#ifdef DEBUG
    fprintf(stderr, "Doing 3d interpolation: x=(%le, %le, %le), i=(%ld, %ld, %ld), f=(%le, %le, %le)\n",
            x, y, z, ix, iy, iz, fx, fy, fz);
#endif

    if (xyInterpolationOrder > 1) {
      double FOutput1[3], FOutput2[3];
      long offset;
      offset = iz * nx * ny;
      if (!singlePrecision)
        interpolate2dFieldMapHigherOrder2(&FOutput1[0], x, y, dx, dy, xi, yi, xf, yf, nx, ny,
                                          Fq[0], Fq[1], Fq[2], offset, singlePrecision, xyInterpolationOrder, xyGridExcess);
      else
        interpolate2dFieldMapHigherOrder2(&FOutput1[0], x, y, dx, dy, xi, yi, xf, yf, nx, ny,
                                          Fq1[0], Fq1[1], Fq1[2], offset, singlePrecision, xyInterpolationOrder, xyGridExcess);

      offset = (iz + 1) * nx * ny;
      if (!singlePrecision)
        interpolate2dFieldMapHigherOrder2(&FOutput2[0], x, y, dx, dy, xi, yi, xf, yf, nx, ny,
                                          Fq[0], Fq[1], Fq[2], offset, singlePrecision, xyInterpolationOrder, xyGridExcess);
      else
        interpolate2dFieldMapHigherOrder2(&FOutput2[0], x, y, dx, dy, xi, yi, xf, yf, nx, ny,
                                          Fq1[0], Fq1[1], Fq1[2], offset, singlePrecision, xyInterpolationOrder, xyGridExcess);

      for (iq = 0; iq < 3; iq++)
        Freturn[iq] = (1 - fz) * FOutput1[iq] + fz * FOutput2[iq];
    } else {
      for (iq = 0; iq < 3; iq++) {
        /* interpolate vs z to get four points in a x-y grid */
        if (!singlePrecision) {
          /* (ix, iy) */
          Finterp1[0][0] = (1 - fz) * (*(Fq[iq] + (ix + 0) + iy * nx + iz * nx * ny)) +
                           fz * (*(Fq[iq] + (ix + 0) + iy * nx + (iz + 1) * nx * ny));
          /* (ix+1, iy) */
          Finterp1[1][0] = (1 - fz) * (*(Fq[iq] + (ix + 1) + (iy + 0) * nx + iz * nx * ny)) +
                           fz * (*(Fq[iq] + (ix + 1) + (iy + 0) * nx + (iz + 1) * nx * ny));
          /* (ix, iy+1) */
          Finterp1[0][1] = (1 - fz) * (*(Fq[iq] + (ix + 0) + (iy + 1) * nx + iz * nx * ny)) +
                           fz * (*(Fq[iq] + (ix + 0) + (iy + 1) * nx + (iz + 1) * nx * ny));
          /* (ix+1, iy+1) */
          Finterp1[1][1] = (1 - fz) * (*(Fq[iq] + (ix + 1) + (iy + 1) * nx + iz * nx * ny)) +
                           fz * (*(Fq[iq] + (ix + 1) + (iy + 1) * nx + (iz + 1) * nx * ny));
        } else {
          Finterp1[0][0] = (1 - fz) * (*(Fq1[iq] + (ix + 0) + iy * nx + iz * nx * ny)) +
                           fz * (*(Fq1[iq] + (ix + 0) + iy * nx + (iz + 1) * nx * ny));
          /* (ix+1, iy) */
          Finterp1[1][0] = (1 - fz) * (*(Fq1[iq] + (ix + 1) + (iy + 0) * nx + iz * nx * ny)) +
                           fz * (*(Fq1[iq] + (ix + 1) + (iy + 0) * nx + (iz + 1) * nx * ny));
          /* (ix, iy+1) */
          Finterp1[0][1] = (1 - fz) * (*(Fq1[iq] + (ix + 0) + (iy + 1) * nx + iz * nx * ny)) +
                           fz * (*(Fq1[iq] + (ix + 0) + (iy + 1) * nx + (iz + 1) * nx * ny));
          /* (ix+1, iy+1) */
          Finterp1[1][1] = (1 - fz) * (*(Fq1[iq] + (ix + 1) + (iy + 1) * nx + iz * nx * ny)) +
                           fz * (*(Fq1[iq] + (ix + 1) + (iy + 1) * nx + (iz + 1) * nx * ny));
        }

        /* interpolate vs y to get two points spaced by dx */
        /* ix */
        Finterp2[0] = (1 - fy) * Finterp1[0][0] + fy * Finterp1[0][1];
        /* ix+1 */
        Finterp2[1] = (1 - fy) * Finterp1[1][0] + fy * Finterp1[1][1];

        /* interpolate vs x to get 1 value */
        Freturn[iq] = (1 - fx) * Finterp2[0] + fx * Finterp2[1];
      }
    }
    for (iq = 0; iq < 3; iq++)
      Freturn[iq] *= mainFactor * fieldFactor;

    if (additionalFactor) {
      Fq[0] = BzNormAdditional;
      Fq[1] = BxNormAdditional;
      Fq[2] = ByNormAdditional;
      Fq1[0] = Bz1NormAdditional;
      Fq1[1] = Bx1NormAdditional;
      Fq1[2] = By1NormAdditional;

#ifdef DEBUG
      fprintf(stderr, "Doing 3d interpolation: x=(%le, %le, %le), i=(%ld, %ld, %ld), f=(%le, %le, %le)\n",
              x, y, z, ix, iy, iz, fx, fy, fz);
#endif

      if (xyInterpolationOrder > 1) {
        double FOutput1[3], FOutput2[3];
        long offset;
        offset = iz * nx * ny;
        if (!singlePrecision)
          interpolate2dFieldMapHigherOrder2(&FOutput1[0], x, y, dx, dy, xi, yi, xf, yf, nx, ny,
                                            Fq[0], Fq[1], Fq[2], offset, singlePrecision, xyInterpolationOrder, xyGridExcess);
        else
          interpolate2dFieldMapHigherOrder2(&FOutput1[0], x, y, dx, dy, xi, yi, xf, yf, nx, ny,
                                            Fq1[0], Fq1[1], Fq1[2], offset, singlePrecision, xyInterpolationOrder, xyGridExcess);

        offset = (iz + 1) * nx * ny;
        if (!singlePrecision)
          interpolate2dFieldMapHigherOrder2(&FOutput2[0], x, y, dx, dy, xi, yi, xf, yf, nx, ny,
                                            Fq[0], Fq[1], Fq[2], offset, singlePrecision, xyInterpolationOrder, xyGridExcess);
        else
          interpolate2dFieldMapHigherOrder2(&FOutput2[0], x, y, dx, dy, xi, yi, xf, yf, nx, ny,
                                            Fq1[0], Fq1[1], Fq1[2], offset, singlePrecision, xyInterpolationOrder, xyGridExcess);

        for (iq = 0; iq < 3; iq++)
          Freturn[iq] += additionalFactor * fieldFactor * ((1 - fz) * FOutput1[iq] + fz * FOutput2[iq]);
      } else {
        for (iq = 0; iq < 3; iq++) {
          /* interpolate vs z to get four points in a x-y grid */
          if (!singlePrecision) {
            /* (ix, iy) */
            Finterp1[0][0] = (1 - fz) * (*(Fq[iq] + (ix + 0) + iy * nx + iz * nx * ny)) +
                             fz * (*(Fq[iq] + (ix + 0) + iy * nx + (iz + 1) * nx * ny));
            /* (ix+1, iy) */
            Finterp1[1][0] = (1 - fz) * (*(Fq[iq] + (ix + 1) + (iy + 0) * nx + iz * nx * ny)) +
                             fz * (*(Fq[iq] + (ix + 1) + (iy + 0) * nx + (iz + 1) * nx * ny));
            /* (ix, iy+1) */
            Finterp1[0][1] = (1 - fz) * (*(Fq[iq] + (ix + 0) + (iy + 1) * nx + iz * nx * ny)) +
                             fz * (*(Fq[iq] + (ix + 0) + (iy + 1) * nx + (iz + 1) * nx * ny));
            /* (ix+1, iy+1) */
            Finterp1[1][1] = (1 - fz) * (*(Fq[iq] + (ix + 1) + (iy + 1) * nx + iz * nx * ny)) +
                             fz * (*(Fq[iq] + (ix + 1) + (iy + 1) * nx + (iz + 1) * nx * ny));
          } else {
            Finterp1[0][0] = (1 - fz) * (*(Fq1[iq] + (ix + 0) + iy * nx + iz * nx * ny)) +
                             fz * (*(Fq1[iq] + (ix + 0) + iy * nx + (iz + 1) * nx * ny));
            /* (ix+1, iy) */
            Finterp1[1][0] = (1 - fz) * (*(Fq1[iq] + (ix + 1) + (iy + 0) * nx + iz * nx * ny)) +
                             fz * (*(Fq1[iq] + (ix + 1) + (iy + 0) * nx + (iz + 1) * nx * ny));
            /* (ix, iy+1) */
            Finterp1[0][1] = (1 - fz) * (*(Fq1[iq] + (ix + 0) + (iy + 1) * nx + iz * nx * ny)) +
                             fz * (*(Fq1[iq] + (ix + 0) + (iy + 1) * nx + (iz + 1) * nx * ny));
            /* (ix+1, iy+1) */
            Finterp1[1][1] = (1 - fz) * (*(Fq1[iq] + (ix + 1) + (iy + 1) * nx + iz * nx * ny)) +
                             fz * (*(Fq1[iq] + (ix + 1) + (iy + 1) * nx + (iz + 1) * nx * ny));
          }

          /* interpolate vs y to get two points spaced by dx */
          /* ix */
          Finterp2[0] = (1 - fy) * Finterp1[0][0] + fy * Finterp1[0][1];
          /* ix+1 */
          Finterp2[1] = (1 - fy) * Finterp1[1][0] + fy * Finterp1[1][1];

          /* interpolate vs x to get 1 value */
          Freturn[iq] += additionalFactor * fieldFactor * ((1 - fx) * Finterp2[0] + fx * Finterp2[1]);
        }
      }
    }

    F[0] = Freturn[0];
    F[1] = Freturn[1];
    F[2] = Freturn[2];
    if (z >= zNomEntry && z <= zNomExit && deltaByInside)
      F[2] += deltaByInside*symmetryFactor[2]; /* undoes factor for this component of By */
  }

  for (j = 0; j < 3; j++)
    F[j] *= fieldSign * (1 + fse)*symmetryFactor[j];

#ifdef DEBUG
  fprintf(stderr, "F[0] = %e, F[1] = %e, F[2] = %e, FSE = %e\n", F[0], F[1], F[2], fse);
#endif

  return;
}

double refineAngle(double theta,
                   double z0, double x0,
                   double zv, double xv,
                   double z1, double x1) {
  double dot, mag0, mag1;
  double cos_theta, theta1;

  if ((xv > x0 && theta < 0) || (xv < x0 && theta > 0)) {
    printf("Error for BRAT: angle and vertex-entry data are inconsistent in sign\n");
    exit(1);
  }

  dot = (xv - x0) * (x1 - xv) + (zv - z0) * (z1 - zv);
  mag0 = sqrt(sqr(xv - x0) + sqr(zv - z0));
  mag1 = sqrt(sqr(xv - x1) + sqr(zv - z1));
  cos_theta = dot / (mag0 * mag1);

  theta1 = acos(cos_theta) * SIGN(theta);
  if (fabs(theta1 - theta) < 1e-4) {
    /*
    printf("Refined BRAT angle from %21.15e to %21.15e (1)\n",
	   theta, theta1);
    */
    return theta1;
  }

  theta1 = PIx2 - theta1;
  if (fabs(theta1 - theta) < 1e-4) {
    /*
    printf("Refined BRAT angle from %21.15e to %21.15e (2)\n",
	   theta, theta1);
    */
    return theta1;
  }

  theta1 = acos(cos_theta);

  printf("Error for BRAT: failed to figure out refined angle:\nx0 = %le, z0 = %le\nxv = %le, zv = %le\nx1 = %le, z1 = %le\ntheta = %le, theta1 = %le\n",
         x0, z0, xv, zv, x1, z1, theta, theta1);
  exit(1);
}

void rotate_coordinate(double **A, double *x, long inverse) {
  long i, j;
  double temp[3] = {0, 0, 0}; /* prevent spurious compiler warning */

  for (i = 0; i < 3; i++) {
    temp[i] = 0;
    for (j = 0; j < 3; j++) {
      if (!inverse)
        temp[i] += A[i][j] * x[j];
      else
        temp[i] += A[j][i] * x[j];
    }
  }

  for (i = 0; i < 3; i++)
    x[i] = temp[i];

  return;
}

/* choose suitable value from cubic solver */
double choose_theta(double rho, double x0, double x1, double x2) {
  double temp = 0;

  if (rho < 0) {
    temp = -DBL_MAX;
    if (x0 < 0 && x0 > temp)
      temp = x0;
    if (x1 < 0 && x1 > temp)
      temp = x1;
    if (x2 < 0 && x2 > temp)
      temp = x2;
  } else if (rho > 0) {
    temp = DBL_MAX;
    if (x0 > 0 && x0 < temp)
      temp = x0;
    if (x1 > 0 && x1 < temp)
      temp = x1;
    if (x2 > 0 && x2 < temp)
      temp = x2;
  } else {
    fprintf(stderr, "choose_theta: rho = %21.15e, x0 = %21.15e,  x1 = %21.15e, x2 = %21.15e\n", rho, x0, x1, x2);
    bombElegant("rho = 0 in choose_theta (FTABLE). Seek expert help.", NULL);
  }

  return temp;
}

static long last_nx = -1, last_nz = -1;
static double last_xi, last_xf, last_dx;
static double last_zi, last_zf, last_dz;

void clear2dMapList() {
  long i;
  for (i = 0; i < length2dMapList; i++) {
    free_zarray_2d((void **)BnormList[i], nx, nz);
    free_zarray_2d((void **)dBnormdzList[i], nx, nz);
    free_zarray_2d((void **)dBnormdxList[i], nx, nz);
  }
  if (BnormList)
    free(BnormList);
  if (dBnormdzList)
    free(dBnormdzList);
  if (dBnormdxList)
    free(dBnormdxList);
  if (interpolationParameterValue)
    free(interpolationParameterValue);
  BnormList = dBnormdzList = dBnormdxList = NULL;
  interpolationParameterValue = NULL;
  length2dMapList = 0;
}

void add2dMapList(double interpValue) {
  BnormList = SDDS_Realloc(BnormList, sizeof(*BnormList) * (length2dMapList + 1));
  dBnormdxList = SDDS_Realloc(dBnormdxList, sizeof(*dBnormdxList) * (length2dMapList + 1));
  dBnormdzList = SDDS_Realloc(dBnormdzList, sizeof(*dBnormdzList) * (length2dMapList + 1));
  interpolationParameterValue = SDDS_Realloc(interpolationParameterValue, sizeof(*interpolationParameterValue) * (length2dMapList + 1));
  /* the map is stored in the global variables, so just copy it */
  BnormList[length2dMapList] = Bnorm;
  Bnorm = NULL;
  dBnormdxList[length2dMapList] = dBnormdx;
  dBnormdx = NULL;
  dBnormdzList[length2dMapList] = dBnormdz;
  dBnormdz = NULL;
  interpolationParameterValue[length2dMapList] = interpValue;
  length2dMapList++;
  if (length2dMapList > 1) {
    if (interpolationParameterValue[length2dMapList - 2] >= interpolationParameterValue[length2dMapList - 1])
      bomb("interpolation parameter values are not monotonically increasing in the input file", NULL);
    if (last_nx != nx)
      bomb("x grid size changed between pages", NULL);
    if (last_nz != nz)
      bomb("z grid size changed between pages", NULL);
    if (!(bratInterpFlags & BRAT_INTERP_PERMISSIVE)) {
      if (last_xi != xi || last_xf != xf || last_dx != dx) {
        printf("x grid parameters changed from %ld, %le, %le, %le to %ld, %le, %le, %le!\n",
               last_nx, last_xi, last_xf, last_dx,
               nx, xi, xf, dx);
        exit(1);
      }
      if (last_nz != nz || last_zi != zi || last_zf != zf || last_dz != dz) {
        printf("z grid parameters changed from %ld, %le, %le, %le to %ld, %le, %le, %le!\n",
               last_nz, last_zi, last_zf, last_dz,
               nz, zi, zf, dz);
        exit(1);
      }
    }
  } else {
    last_nx = nx;
    last_nz = nz;
    last_xi = xi;
    last_xf = xf;
    last_dx = dx;
    last_zi = zi;
    last_zf = zf;
    last_dz = dz;
  }
}

int interpolate2dFieldMapHigherOrder(
  double *Foutput, /* output of interpolation */
  double x, double y,
  double dx, double dy,
  double xmin, double ymin,
  double xmax, double ymax,
  long nx, long ny,
  double *F0, double *F1, double *F2, /* maps to interpolate, ignored if NULL */
  short order,
  short gridExcess)
/* Performs 2nd- and  higher order interpolation of uniformly-spaced 2d field maps.
   Method is to solve XY*A = F, where for 2nd order
   XY is a 16x6 matrix for the grid such that XY[i][j] = (1, y[i], y[i]^2, x[i], x[i]*y[i], x[i]^2), i=0, ..., 15
   F is 16x1 with F = Trans[(f[0][0], f[0][1], f[0][2], f[0][3], f[1][0], f[1][1], f[1][2], f[1][3], ... f[3][3])]
   A is determined from Inv[XY*Transpose[XY]]*Tranpose[XY]*F, then used to determine F values of x and y not on the grid
 */

{
  static MATRIX *XY = NULL, *xy = NULL, *XYTrans = NULL, *XYTransXY = NULL, *S = NULL, *T = NULL, *U = NULL;
  double *Field[3] = {NULL, NULL, NULL};
  long i, j, l, k, m, n, f, nc;
  static double *xPow = NULL, *yPow = NULL;
  static long lastOrder = -1, dim = -1, ng = -1, gridOffset = -1;
  long ix, iy, minGrid;
  double fx, fy;

  if (lastOrder != order) {
    if (XY) {
      m_free(&XY);
      m_free(&XYTrans);
      m_free(&T);
      m_free(&S);
      m_free(&U);
      m_free(&xy);
      free(xPow);
      free(yPow);
    }
    lastOrder = order;

    nc = (order + 2) * (order + 1) / 2; /* number of polynomial coefficients */
    /* ensure that we have sufficient data for the required number of coefficients */
    minGrid = sqrt(nc);
    while (nc > (minGrid * minGrid))
      minGrid++;
    /* add the user's "excess" number of rows and columns */
    if (gridExcess < 0)
      gridExcess = 0;
    ng = minGrid + gridExcess; /* number of rows and columns of data to use */
    if (ng>nx || ng>ny)
      ng = nx>ny ? ny : nx;
    dim = sqr(ng);             /* number of points in the ng x ng grid*/
    if (dim < nc)
      bombElegant("Something wrong with setting up the number of rows and columns of data for higher-order x-y fitting", NULL);
    gridOffset = (ng - 1) / 2;

    m_alloc(&XY, dim, nc);       /* array of polynomial terms */
    m_alloc(&XYTrans, nc, dim);  /* transpose of same */
    m_alloc(&XYTransXY, nc, nc); /* product of transpose and XY */
    m_alloc(&T, nc, nc);         /* inverse of (XYTrans*XY) */
    m_alloc(&S, nc, dim);        /* T*(XYTrans*XY)^{-1} */
    m_alloc(&xy, 1, nc);         /* vector of polynomial terms for fit evaluation */
    m_alloc(&U, 1, dim);         /* xy*S */
    /*
    printf("Using %ld x %ld grid for order=%hd interpolation in BRAT/BMXYZ elements (%ld coefficients, %ld fit points)\n",
	   ng, ng, order, nc, dim);
    fflush(stdout);
    */
    /* arrays of stored powers of x and y */
    xPow = tmalloc(sizeof(*xPow) * (order + 1));
    yPow = tmalloc(sizeof(*yPow) * (order + 1));

    for (i = l = 0; i < ng; i++) {
      /* i is (xGrid-x0)/dx */
      for (j = 0; j < ng; j++, l++) {
        /* j is (yGrid-y0)/dy */
        xPow[0] = yPow[0] = 1;
        for (m = 1; m <= order; m++) {
          xPow[m] = i * xPow[m - 1];
          yPow[m] = j * yPow[m - 1];
        }
        for (m = k = 0; m <= order; m++) {
          for (n = 0; n <= (order - m); n++, k++)
            XY->a[l][k] = xPow[m] * yPow[n];
        }
      }
    }
    if (!m_trans(XYTrans, XY))
      return 0;
    if (!m_mult(XYTransXY, XYTrans, XY))
      return 0;
    if (!m_invert(T, XYTransXY))
      return 0;
    if (!m_mult(S, T, XYTrans))
      return 0;
  }

  /* calculate minimum x index for the grid points */
  ix = (x - xmin) / dx - gridOffset;
  /* ensure that all the points are within the full grid */
  if ((ix + ng) > (nx - 1))
    ix = nx - 1 - ng;
  if (ix < 0)
    ix = 0;
  /* fractional position of the particle for the reduced (local) grid */
  fx = (x - (ix * dx + xmin)) / dx;

  /* calculate minimum y index for the grid points */
  iy = (y - ymin) / dy - gridOffset;
  /* ensure that all the points are within the full grid */
  if ((iy + ng) > (ny - 1))
    iy = ny - 1 - ng;
  if (iy < 0)
    iy = 0;
  /* fractional position of the particle for the reduced (local) grid */
  fy = (y - (iy * dy + ymin)) / dy;

  xPow[0] = yPow[0] = 1;
  for (m = 1; m <= order; m++) {
    xPow[m] = fx * xPow[m - 1];
    yPow[m] = fy * yPow[m - 1];
  }
  for (m = k = 0; m <= order; m++) {
    for (n = 0; n <= (order - m); n++, k++)
      xy->a[0][k] = xPow[m] * yPow[n];
  }

  m_mult(U, xy, S);

  Field[0] = F0;
  Field[1] = F1;
  Field[2] = F2;
  for (f = 0; f < 3; f++) {
    if (!Field[f])
      continue;
    Foutput[f] = 0;
    for (i = k = 0; i < ng; i++) {
      for (j = 0; j < ng; j++, k++) {
        Foutput[f] += *(Field[f] + (ix + i) + (iy + j) * nx) * U->a[0][k];
      }
    }
  }

  return 1;
}

int interpolate2dFieldMapHigherOrder2
  /* allows single- or double-precision field maps */
  (
    double *Foutput, /* output of interpolation */
    double x, double y,
    double dx, double dy,
    double xmin, double ymin,
    double xmax, double ymax,
    long nx, long ny,
    void *F0, void *F1, void *F2, /* maps to interpolate, ignored if NULL */
    long offset,
    short singlePrecision,
    short order,
    short gridExcess)
/* Performs 2nd- and  higher order interpolation of uniformly-spaced 2d field maps.
   Method is to solve XY*A = F, where for 2nd order
   XY is a 16x6 matrix for the grid such that XY[i][j] = (1, y[i], y[i]^2, x[i], x[i]*y[i], x[i]^2), i=0, ..., 15
   F is 16x1 with F = Trans[(f[0][0], f[0][1], f[0][2], f[0][3], f[1][0], f[1][1], f[1][2], f[1][3], ... f[3][3])]
   A is determined from Inv[XY*Transpose[XY]]*Tranpose[XY]*F, then used to determine F values of x and y not on the grid
 */

{
  static MATRIX *XY = NULL, *xy = NULL, *XYTrans = NULL, *XYTransXY = NULL, *S = NULL, *T = NULL, *U = NULL;
  long i, j, l, k, m, n, f, nc;
  static double *xPow = NULL, *yPow = NULL;
  static long lastOrder = -1, dim = -1, ng = -1, gridOffset = -1;
  long ix, iy, minGrid;
  double fx, fy;

  if (lastOrder != order) {
    if (XY) {
      m_free(&XY);
      m_free(&XYTrans);
      m_free(&T);
      m_free(&S);
      m_free(&U);
      m_free(&xy);
      free(xPow);
      free(yPow);
    }
    lastOrder = order;

    nc = (order + 2) * (order + 1) / 2; /* number of polynomial coefficients */
    /* ensure that we have sufficient data for the required number of coefficients */
    minGrid = sqrt(nc);
    while (nc > (minGrid * minGrid))
      minGrid++;
    /* add the user's "excess" number of rows and columns */
    if (gridExcess < 0)
      gridExcess = 0;
    ng = minGrid + gridExcess; /* number of rows and columns of data to use */
    if (ng>nx || ng>ny)
      ng = nx>ny ? ny : nx;
    dim = sqr(ng);             /* number of points in the ng x ng grid*/
    if (dim < nc)
      bombElegant("Something wrong with setting up the number of rows and columns of data for higher-order x-y fitting", NULL);
    gridOffset = (ng - 1) / 2;

    m_alloc(&XY, dim, nc);       /* array of polynomial terms */
    m_alloc(&XYTrans, nc, dim);  /* transpose of same */
    m_alloc(&XYTransXY, nc, nc); /* product of transpose and XY */
    m_alloc(&T, nc, nc);         /* inverse of (XYTrans*XY) */
    m_alloc(&S, nc, dim);        /* T*(XYTrans*XY)^{-1} */
    m_alloc(&xy, 1, nc);         /* vector of polynomial terms for fit evaluation */
    m_alloc(&U, 1, dim);         /* xy*S */

    printf("Using %ld x %ld grid for order=%hd interpolation in BRAT/BMXYZ elements (%ld coefficients, %ld fit points)\n",
	   ng, ng, order, nc, dim);
    fflush(stdout);

    /* arrays of stored powers of x and y */
    xPow = tmalloc(sizeof(*xPow) * (order + 1));
    yPow = tmalloc(sizeof(*yPow) * (order + 1));

    for (i = l = 0; i < ng; i++) {
      /* i is (xGrid-x0)/dx */
      for (j = 0; j < ng; j++, l++) {
        /* j is (yGrid-y0)/dy */
        xPow[0] = yPow[0] = 1;
        for (m = 1; m <= order; m++) {
          xPow[m] = i * xPow[m - 1];
          yPow[m] = j * yPow[m - 1];
        }
        for (m = k = 0; m <= order; m++) {
          for (n = 0; n <= (order - m); n++, k++)
            XY->a[l][k] = xPow[m] * yPow[n];
        }
      }
    }
    if (!m_trans(XYTrans, XY))
      return 0;
    if (!m_mult(XYTransXY, XYTrans, XY))
      return 0;
    if (!m_invert(T, XYTransXY))
      return 0;
    if (!m_mult(S, T, XYTrans))
      return 0;
  }

  /* calculate minimum x index for the grid points */
  ix = (x - xmin) / dx - gridOffset;
  /* ensure that all the points are within the full grid */
  if ((ix + ng) > (nx - 1))
    ix = nx - 1 - ng;
  if (ix < 0)
    ix = 0;
  /* fractional position of the particle for the reduced (local) grid */
  fx = (x - (ix * dx + xmin)) / dx;

  /* calculate minimum y index for the grid points */
  iy = (y - ymin) / dy - gridOffset;
  /* ensure that all the points are within the full grid */
  if ((iy + ng) > (ny - 1))
    iy = ny - 1 - ng;
  if (iy < 0)
    iy = 0;
  /* fractional position of the particle for the reduced (local) grid */
  fy = (y - (iy * dy + ymin)) / dy;

  xPow[0] = yPow[0] = 1;
  for (m = 1; m <= order; m++) {
    xPow[m] = fx * xPow[m - 1];
    yPow[m] = fy * yPow[m - 1];
  }
  for (m = k = 0; m <= order; m++) {
    for (n = 0; n <= (order - m); n++, k++)
      xy->a[0][k] = xPow[m] * yPow[n];
  }

  m_mult(U, xy, S);

  if (!singlePrecision) {
    double *Field[3] = {NULL, NULL, NULL};
    Field[0] = ((double *)F0) + offset;
    Field[1] = ((double *)F1) + offset;
    Field[2] = ((double *)F2) + offset;
    for (f = 0; f < 3; f++) {
      if (!Field[f])
        continue;
      Foutput[f] = 0;
      for (i = k = 0; i < ng; i++) {
        for (j = 0; j < ng; j++, k++) {
          Foutput[f] += *(Field[f] + (ix + i) + (iy + j) * nx) * U->a[0][k];
        }
      }
    }
  } else {
    float *Field[3] = {NULL, NULL, NULL};
    Field[0] = ((float *)F0) + offset;
    Field[1] = ((float *)F1) + offset;
    Field[2] = ((float *)F2) + offset;
    for (f = 0; f < 3; f++) {
      if (!Field[f])
        continue;
      Foutput[f] = 0;
      for (i = k = 0; i < ng; i++) {
        for (j = 0; j < ng; j++, k++) {
          Foutput[f] += *(Field[f] + (ix + i) + (iy + j) * nx) * U->a[0][k];
        }
      }
    }
  }

  return 1;
}

void readBratFieldFile(BRAT *brat, char *filename, short additionalFile) {
  long i, idata, rows;
  long nx, ny, nz;
  SDDS_DATASET SDDS_table;
  double Bmin, Bmax;
  short ySymmetryCode = 0;
  short zSymmetryCode = 0;

  /* See if we've read this file already---should use a hash table */
  for (i = 0; i < nBrat3dData; i++) {
    if (strcmp(filename, brat3dData[i].filename) == 0) {
      printf("Using previously-read data for BRAT file %s\n",
             filename);
      break;
    }
  }

  if (i != nBrat3dData) {
    if ((brat->singlePrecision && !brat3dData[i].singlePrecision) ||
        (!brat->singlePrecision && brat3dData[i].singlePrecision))
      bombElegantVA("Error: inconsistent SINGLE_PRECISION settings for use of field map %s", filename);
    if (!additionalFile)
      brat->dataIndex = i;
    else
      brat->dataIndexAdditional = i;
    return;
  } else if (!brat->singlePrecision) {
    double *xd, *yd, *zd, *Bxd, *Byd, *Bzd;
    xd = yd = zd = Bxd = Byd = Bzd = NULL;

    /* read data from file */

    printf("Reading BRAT field data from %s\n", filename);
    fflush(stdout);

    if (!SDDS_InitializeInputFromSearchPath(&SDDS_table, filename) || !SDDS_ReadPage(&SDDS_table)) {
      SDDS_SetError("Unable to read BRAT data file");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    SDDS_SetColumnMemoryMode(&SDDS_table, DONT_TRACK_COLUMN_MEMORY_AFTER_ACCESS);
    if (!(rows = SDDS_CountRowsOfInterest(&SDDS_table)))
      bomb("no data in BRAT field file", NULL);
    printf("%ld rows of data\n", rows);
    fflush(stdout);
    if (SDDS_CheckColumn(&SDDS_table, "x", "m", SDDS_ANY_FLOATING_TYPE, stderr) != SDDS_CHECK_OKAY ||
        SDDS_CheckColumn(&SDDS_table, "y", "m", SDDS_ANY_FLOATING_TYPE, stderr) != SDDS_CHECK_OKAY ||
        SDDS_CheckColumn(&SDDS_table, "z", "m", SDDS_ANY_FLOATING_TYPE, stderr) != SDDS_CHECK_OKAY) {
      printf("units wrong on x, y, or z\n");
      fflush(stdout);
      exit(1);
    }
    printf("x, y, and z found with expected units\n");
    fflush(stdout);

    /* It is assumed that the data is ordered so that x changes fastest.
     * This can be accomplished with sddssort -column=z,incr -column=y,incr -column=x,incr
     * The points are assumed to be equipspaced.
     */
    if (!(xd = SDDS_GetColumnInDoubles(&SDDS_table, "x")))
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    nx = 1;
    xi = xd[0];
    while (nx < rows) {
      if (xd[nx - 1] > xd[nx])
        break;
      nx++;
    }
    if (nx == rows) {
      fprintf(stderr, "BRAT file doesn't have correct structure or amount of data (x)\n");
      fprintf(stderr, "Use sddssort -column=z -column=y -column=x to sort the file\n");
      exit(1);
    }
    xf = xd[nx - 1];
    dx = (xf - xi) / (nx - 1);
    free(xd);
    
    if (!(yd = SDDS_GetColumnInDoubles(&SDDS_table, "y")))
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    ny = 1;
    yi = yd[0];
    while (ny < (rows / nx)) {
      if (yd[(ny - 1) * nx] > yd[ny * nx])
        break;
      ny++;
    }
    if (ny == rows) {
      fprintf(stderr, "BRAT file doesn't have correct structure or amount of data (y)\n");
      fprintf(stderr, "Use sddssort -column=z -column=y -column=x to sort the file\n");
      exit(1);
    }
    yf = yd[(ny - 1) * nx];
    dy = (yf - yi) / (ny - 1);
    free(yd);
    
    if (!(zd = SDDS_GetColumnInDoubles(&SDDS_table, "z")))
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    if (nx <= 1 || ny <= 1 || (nz = rows / (nx * ny)) <= 1) {
      fprintf(stderr, "BRAT file doesn't have correct structure or amount of data (z)\n");
      fprintf(stderr, "Use sddssort -column=z -column=y -column=x to sort the file\n");
      exit(1);
    }
    zi = zd[0];
    zf = zd[rows - 1];
    dz = (zf - zi) / (nz - 1);
    free(zd);
    
    if (SDDS_CheckColumn(&SDDS_table, "Bx", "T", SDDS_ANY_FLOATING_TYPE, stderr) != SDDS_CHECK_OKAY ||
        SDDS_CheckColumn(&SDDS_table, "By", "T", SDDS_ANY_FLOATING_TYPE, stderr) != SDDS_CHECK_OKAY ||
        SDDS_CheckColumn(&SDDS_table, "Bz", "T", SDDS_ANY_FLOATING_TYPE, stderr) != SDDS_CHECK_OKAY) {
      printf("units wrong on Bx, By, or Bz\n");
      fflush(stdout);
      exit(1);
    }
    printf("Bx, By, and Bz found with expected units\n");
    fflush(stdout);
    if (!(Bxd = SDDS_GetColumnInDoubles(&SDDS_table, "Bx")) ||
        !(Byd = SDDS_GetColumnInDoubles(&SDDS_table, "By")) ||
        !(Bzd = SDDS_GetColumnInDoubles(&SDDS_table, "Bz"))) {
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    }


    Bmin = -(Bmax = -DBL_MAX);
    for (idata = 0; idata < rows; idata++) {
      /* if (fabs(yd[idata])<dy/2 && fabs(xd[idata])<dx/2) { */
      if (Byd[idata] > Bmax)
        Bmax = Byd[idata];
      if (Byd[idata] < Bmin)
        Bmin = Byd[idata];
      /* } */
    }

    if (Bmax == -DBL_MAX) {
      fprintf(stderr, "BRAT file doesn't have valid Bmax value\n");
      fprintf(stderr, "Bmax = %le, Bmin = %le\n", Bmax, Bmin);
      exit(1);
    }
    if (fabs(Bmin) > fabs(Bmax))
      SWAP_DOUBLE(Bmin, Bmax);

    if (!quiet)
      printf("3D BRAT field map data: nx=%ld, ny=%ld, nz=%ld\ndx=%21.15e, dy=%21.15e, dz=%21.15e\nx:[%21.15e, %21.15e], y:[%21.15e, %21.15e], z:[%21.15e, %21.15e]\nBy:[%21.15e, %21.15e]\n",
             nx, ny, nz, dx, dy, dz,
             xi, xf, yi, yf, zi, zf,
             Bmin, Bmax);

    /* copy to storage area */
    if (!(brat3dData = SDDS_Realloc(brat3dData, sizeof(*brat3dData) * (nBrat3dData + 1))))
      bombElegant("memory allocation failure storage BRAT data", NULL);

    brat3dData[nBrat3dData].Bmin = Bmin;
    brat3dData[nBrat3dData].Bmax = Bmax;

    brat3dData[nBrat3dData].Bx1 = brat3dData[nBrat3dData].By1 = brat3dData[nBrat3dData].Bz1 = NULL;

    brat3dData[nBrat3dData].Bx = Bxd;
    brat3dData[nBrat3dData].xi = xi;
    brat3dData[nBrat3dData].xf = xf;
    brat3dData[nBrat3dData].dx = dx;
    brat3dData[nBrat3dData].nx = nx;

    brat3dData[nBrat3dData].By = Byd;
    brat3dData[nBrat3dData].yi = yi;
    brat3dData[nBrat3dData].yf = yf;
    brat3dData[nBrat3dData].dy = dy;
    brat3dData[nBrat3dData].ny = ny;

    brat3dData[nBrat3dData].Bz = Bzd;
    brat3dData[nBrat3dData].zi = zi;
    brat3dData[nBrat3dData].zf = zf;
    brat3dData[nBrat3dData].dz = dz;
    brat3dData[nBrat3dData].nz = nz;
    brat3dData[nBrat3dData].singlePrecision = 0;
  } else {
    /* single-precision data */
    float *xd, *yd, *zd, *Bxd, *Byd, *Bzd;
    xd = yd = zd = Bxd = Byd = Bzd = NULL;

    /* read data from file */

    printf("Reading BRAT field data from %s\n", filename);
    fflush(stdout);

    if (!SDDS_InitializeInputFromSearchPath(&SDDS_table, filename) || !SDDS_ReadPage(&SDDS_table)) {
      SDDS_SetError("Unable to read BRAT data file");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    SDDS_SetColumnMemoryMode(&SDDS_table, DONT_TRACK_COLUMN_MEMORY_AFTER_ACCESS);
    if (!(rows = SDDS_CountRowsOfInterest(&SDDS_table)))
      bomb("no data in BRAT field file", NULL);
    printf("%ld rows of data\n", rows);
    fflush(stdout);
    if (SDDS_CheckColumn(&SDDS_table, "x", "m", SDDS_ANY_FLOATING_TYPE, stderr) != SDDS_CHECK_OKAY ||
        SDDS_CheckColumn(&SDDS_table, "y", "m", SDDS_ANY_FLOATING_TYPE, stderr) != SDDS_CHECK_OKAY ||
        SDDS_CheckColumn(&SDDS_table, "z", "m", SDDS_ANY_FLOATING_TYPE, stderr) != SDDS_CHECK_OKAY) {
      printf("units wrong on x, y, or z\n");
      fflush(stdout);
      exit(1);
    }
    printf("x, y, and z found with expected units\n");
    fflush(stdout);

    /* It is assumed that the data is ordered so that x changes fastest.
     * This can be accomplished with sddssort -column=z,incr -column=y,incr -column=x,incr
     * The points are assumed to be equipspaced.
     */
    if (!(xd = SDDS_GetColumnInFloats(&SDDS_table, "x"))) 
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    nx = 1;
    xi = xd[0];
    while (nx < rows) {
      if (xd[nx - 1] > xd[nx])
        break;
      nx++;
    }
    if (nx == rows) {
      fprintf(stderr, "BRAT file doesn't have correct structure or amount of data (x)\n");
      fprintf(stderr, "Use sddssort -column=z -column=y -column=x to sort the file\n");
      exit(1);
    }
    xf = xd[nx - 1];
    dx = (xf - xi) / (nx - 1);
    free(xd);

    if (!(yd = SDDS_GetColumnInFloats(&SDDS_table, "y")))
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    ny = 1;
    yi = yd[0];
    while (ny < (rows / nx)) {
      if (yd[(ny - 1) * nx] > yd[ny * nx])
        break;
      ny++;
    }
    if (ny == rows) {
      fprintf(stderr, "BRAT file doesn't have correct structure or amount of data (y)\n");
      fprintf(stderr, "Use sddssort -column=z -column=y -column=x to sort the file\n");
      exit(1);
    }
    yf = yd[(ny - 1) * nx];
    dy = (yf - yi) / (ny - 1);
    free(yd);
    
    if (!(zd = SDDS_GetColumnInFloats(&SDDS_table, "z")))
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    if (nx <= 1 || ny <= 1 || (nz = rows / (nx * ny)) <= 1) {
      fprintf(stderr, "BRAT file doesn't have correct structure or amount of data (z)\n");
      fprintf(stderr, "Use sddssort -column=z -column=y -column=x to sort the file\n");
      exit(1);
    }
    zi = zd[0];
    zf = zd[rows - 1];
    dz = (zf - zi) / (nz - 1);
    free(zd);

    if (SDDS_CheckColumn(&SDDS_table, "Bx", "T", SDDS_ANY_FLOATING_TYPE, stderr) != SDDS_CHECK_OKAY ||
        SDDS_CheckColumn(&SDDS_table, "By", "T", SDDS_ANY_FLOATING_TYPE, stderr) != SDDS_CHECK_OKAY ||
        SDDS_CheckColumn(&SDDS_table, "Bz", "T", SDDS_ANY_FLOATING_TYPE, stderr) != SDDS_CHECK_OKAY) {
      printf("units wrong on Bx, By, or Bz\n");
      fflush(stdout);
      exit(1);
    }
    printf("Bx, By, and Bz found with expected units\n");
    fflush(stdout);
    if (!(Bxd = SDDS_GetColumnInFloats(&SDDS_table, "Bx")) ||
        !(Byd = SDDS_GetColumnInFloats(&SDDS_table, "By")) ||
        !(Bzd = SDDS_GetColumnInFloats(&SDDS_table, "Bz"))) {
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    }


    Bmin = -(Bmax = -DBL_MAX);
    for (idata = 0; idata < rows; idata++) {
      /* if (fabs(yd[idata])<dy/2 && fabs(xd[idata])<dx/2) { */
      if (Byd[idata] > Bmax)
        Bmax = Byd[idata];
      if (Byd[idata] < Bmin)
        Bmin = Byd[idata];
      /* } */
    }

    if (Bmax == -DBL_MAX) {
      fprintf(stderr, "BRAT file doesn't have valid Bmax value\n");
      fprintf(stderr, "Bmax = %le, Bmin = %le\n", Bmax, Bmin);
      exit(1);
    }
    if (fabs(Bmin) > fabs(Bmax))
      SWAP_DOUBLE(Bmin, Bmax);

    if (!quiet)
      printf("3D BRAT field map data: nx=%ld, ny=%ld, nz=%ld\ndx=%21.15e, dy=%21.15e, dz=%21.15e\nx:[%21.15e, %21.15e], y:[%21.15e, %21.15e], z:[%21.15e, %21.15e]\nBy:[%21.15e, %21.15e]\n",
             nx, ny, nz, dx, dy, dz,
             xi, xf, yi, yf, zi, zf,
             Bmin, Bmax);

    /* copy to storage area */
    if (!(brat3dData = SDDS_Realloc(brat3dData, sizeof(*brat3dData) * (nBrat3dData + 1))))
      bombElegant("memory allocation failure storage BRAT data", NULL);

    brat3dData[nBrat3dData].Bx = brat3dData[nBrat3dData].By = brat3dData[nBrat3dData].Bz = NULL;

    brat3dData[nBrat3dData].Bmin = Bmin;
    brat3dData[nBrat3dData].Bmax = Bmax;

    brat3dData[nBrat3dData].Bx1 = Bxd;
    brat3dData[nBrat3dData].xi = xi;
    brat3dData[nBrat3dData].xf = xf;
    brat3dData[nBrat3dData].dx = dx;
    brat3dData[nBrat3dData].nx = nx;

    brat3dData[nBrat3dData].By1 = Byd;
    brat3dData[nBrat3dData].yi = yi;
    brat3dData[nBrat3dData].yf = yf;
    brat3dData[nBrat3dData].dy = dy;
    brat3dData[nBrat3dData].ny = ny;

    brat3dData[nBrat3dData].Bz1 = Bzd;
    brat3dData[nBrat3dData].zi = zi;
    brat3dData[nBrat3dData].zf = zf;
    brat3dData[nBrat3dData].dz = dz;
    brat3dData[nBrat3dData].nz = nz;
    brat3dData[nBrat3dData].singlePrecision = 1;
  }

  if (SDDS_GetParameterIndex(&SDDS_table, "ySymmetry")>=0) {
    char *symmetry[3] = {"none", "odd", "even"};
    char *ySymmetry = NULL;
    if (SDDS_CheckParameter(&SDDS_table, "ySymmetry", NULL, SDDS_STRING, stdout)!=SDDS_CHECK_OK  ||
	!SDDS_GetParameter(&SDDS_table, "ySymmetry", (void*)&ySymmetry)) 
      bombElegantVA("Problem with \"ySymmetry\" parameter in BRAT file %s: not string type \n", filename);
    if ((ySymmetryCode = match_string(ySymmetry, symmetry, 3, 0))<0)
      bombElegantVA("Problem with \"ySymmetry\" parameter in BRAT file %s: value %s not recognized\n", filename, ySymmetry);
    printf("Recognized y-plane magnet symmetry of %s\n", symmetry[ySymmetryCode]);
  }
  if (SDDS_GetParameterIndex(&SDDS_table, "zSymmetry")>=0) {
    char *symmetry[3] = {"none", "odd", "even"};
    char *zSymmetry = NULL;
    if (SDDS_CheckParameter(&SDDS_table, "zSymmetry", NULL, SDDS_STRING, stdout)!=SDDS_CHECK_OK  ||
	!SDDS_GetParameter(&SDDS_table, "zSymmetry", (void*)&zSymmetry)) 
      bombElegantVA("Problem with \"zSymmetry\" parameter in BRAT file %s: not string type \n", filename);
    if ((zSymmetryCode = match_string(zSymmetry, symmetry, 3, 0))<0)
      bombElegantVA("Problem with \"zSymmetry\" parameter in BRAT file %s: value %s not recognized\n", filename, zSymmetry);
    printf("Recognized z-plane magnet symmetry of %s\n", symmetry[zSymmetryCode]);
  }
  brat3dData[nBrat3dData].ySymmetryCode = ySymmetryCode;
  brat3dData[nBrat3dData].zSymmetryCode = zSymmetryCode;
  if (!SDDS_Terminate(&SDDS_table))
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
  printf("Finished reading data from file\n");
  fflush(stdout);

  cp_str(&brat3dData[nBrat3dData].filename, filename);
  if (additionalFile)
    brat->dataIndexAdditional = nBrat3dData;
  else
    brat->dataIndex = nBrat3dData;
  nBrat3dData += 1;
  printf("Done reading BRAT field data from %s\n", filename);
  fflush(stdout);
}

void writeBratFieldOutput(BRAT *brat, char *rootname)
{
#ifndef ABRAT_PROGRAM
  long ix, iy, iz, row;
  long nx1, ny1, nz1;
  double xyz[3], B[3], dx1, dy1, dz1;
  SDDS_DATASET SDDSout;
  BRAT_3D_DATA *data;
  char *filename;

#if USE_MPI
  if (myid==0) {
#endif
    data = &brat3dData[brat->dataIndex];
  
    nx1 = brat->nOutput[0]<=1 ? data->nx :  brat->nOutput[0];
    ny1 = brat->nOutput[1]<=1 ? data->ny :  brat->nOutput[1];
    nz1 = brat->nOutput[2]<=1 ? data->nz :  brat->nOutput[2];
    
    filename = compose_filename(brat->fieldOutputFile, rootname);
    printf("Writing BRAT field output to %s\n", filename);
    fflush(stdout);

    if (!SDDS_InitializeOutput(&SDDSout, SDDS_BINARY, 0, NULL, NULL, filename)) {
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        exitElegant(1);
    }
    if (!SDDS_DefineSimpleColumn(&SDDSout, "x", "m", SDDS_FLOAT) ||
        !SDDS_DefineSimpleColumn(&SDDSout, "y", "m", SDDS_FLOAT) ||
        !SDDS_DefineSimpleColumn(&SDDSout, "z", "m", SDDS_FLOAT) ||
        !SDDS_DefineSimpleColumn(&SDDSout, "Bx", "T", SDDS_FLOAT) ||
        !SDDS_DefineSimpleColumn(&SDDSout, "By", "T", SDDS_FLOAT) ||
        !SDDS_DefineSimpleColumn(&SDDSout, "Bz", "T", SDDS_FLOAT) ||
        !SDDS_WriteLayout(&SDDSout) ||
        !SDDS_StartPage(&SDDSout, nx1*ny1*nz1)) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }

    dx1 = (data->xf-data->xi)/(nx1-1);
    dy1 = (data->yf-data->yi)/(ny1-1);
    dz1 = (data->zf-data->zi)/(nz1-1);
    
    row = 0;
    for (iz=0; iz<nz1; iz++) {
      xyz[2] = data->zi + dz1*iz;
      if (xyz[2]>=data->zf)
        xyz[2] = data->zf - data->dz/1e6;
      for (iy=0; iy<ny1; iy++) {
        xyz[1] = data->yi + dy1*iy;
        if (xyz[1]>=data->yf)
          xyz[1] = data->yf - data->dy/1e6;
        for (ix=0; ix<nx1; ix++) {
          xyz[0] = data->xi + dx1*ix;
          if (xyz[0]>=data->xf)
            xyz[0] = data->xf - data->dx/1e6;
          BRAT_B_field_permuted(B, xyz);
          if (!SDDS_SetRowValues(&SDDSout, SDDS_SET_BY_INDEX|SDDS_PASS_BY_VALUE, row++,
                                 0, (float)xyz[0], 1, (float)xyz[1], 2, (float)xyz[2], 
                                 3, (float)B[0], 4, (float)B[1], 5, (float)B[2], -1)) {
            SDDS_SetError("Problem setting data in SDDS table for BRAT field output.");
            SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
          }
        }
      }
    }
    if (!SDDS_WriteTable(&SDDSout) || !SDDS_Terminate(&SDDSout)) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    }
#if USE_MPI
  }
  MPI_Barrier(MPI_COMM_WORLD);
#endif
#endif
}
