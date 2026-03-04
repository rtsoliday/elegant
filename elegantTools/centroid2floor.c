/* centroid2floor.c
   This program will read two files: a floor-coordinates output file and a centroid output file .
   The former gives the reference coordinates (s, Z, X, Y, theta, phi, psi) of a beamline while the
   latter gives the offsets (s, Cx, Cy) of the beam path in Frenet-Serret coordinates.
   The output from the program is the beam path in (Z, X, Y, theta, phi, psi) coordinates.

   This software was written entirely by GPT-5.3-Codex using instructions from M. Borland,
   who performed testing.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "SDDS.h"
#include "mdb.h"

typedef struct {
  /* Floor-coordinate reference trajectory sample at arc length s. */
  double s, X, Y, Z, theta, phi, psi;
} FLOOR_POINT;

static int compare_floor_points(const void *item1, const void *item2) {
  const FLOOR_POINT *p1 = (const FLOOR_POINT *)item1;
  const FLOOR_POINT *p2 = (const FLOOR_POINT *)item2;
  if (p1->s < p2->s)
    return -1;
  if (p1->s > p2->s)
    return 1;
  return 0;
}

static char *USAGE = "centroid2floor <floorFile> <centroidFile> <outputFile>\n\n"
                     "Converts centroid offsets from Frenet-Serret coordinates to global floor coordinates.\n"
                     "The floor file must provide columns s, X, Y, Z, theta, phi, psi.\n"
                     "The centroid file must provide columns s, Cx, Cy; Cxp/Cyp are optional.\n\n"
                     "Program by GPT-5.2-Codex and Michael Borland.\n";

/*
  Make angular data continuous near a reference value so linear interpolation
  does not jump by ±2*pi at wrap boundaries.
*/
static double unwrap_angle(double value, double reference) {
  while (value - reference > PI)
    value -= PIx2;
  while (value - reference < -PI)
    value += PIx2;
  return value;
}

static double wrap_angle(double value) {
  while (value > PI)
    value -= PIx2;
  while (value < -PI)
    value += PIx2;
  return value;
}

/*
  Build a right-handed local triad from elegant floor angles:
    es = local tangent (+s direction)
    ex = local horizontal (x in Frenet-Serret system)
    ey = local vertical   (y in Frenet-Serret system)

  theta and phi define pointing direction of the tangent.  psi is roll about
  the tangent and rotates (ex,ey) around es.
*/
static void make_local_basis(double theta, double phi, double psi,
                             double ex[3], double ey[3], double es[3]) {
  double cth = cos(theta), sth = sin(theta);
  double cph = cos(phi), sph = sin(phi);
  double cps = cos(psi), sps = sin(psi);
  double ex0[3], ey0[3];

  es[0] = sth * cph;
  es[1] = sph;
  es[2] = cth * cph;

  ex0[0] = cth;
  ex0[1] = 0;
  ex0[2] = -sth;

  ey0[0] = -sph * sth;
  ey0[1] = cph;
  ey0[2] = -sph * cth;

  ex[0] = ex0[0] * cps + ey0[0] * sps;
  ex[1] = ex0[1] * cps + ey0[1] * sps;
  ex[2] = ex0[2] * cps + ey0[2] * sps;

  ey[0] = -ex0[0] * sps + ey0[0] * cps;
  ey[1] = -ex0[1] * sps + ey0[1] * cps;
  ey[2] = -ex0[2] * sps + ey0[2] * cps;
}

static void interpolate_floor(double sValue, long *iFloor, long nFloor,
                              double *sFloor, double *XFloor, double *YFloor, double *ZFloor,
                              double *thetaFloor, double *phiFloor, double *psiFloor,
                              double *X, double *Y, double *Z,
                              double *theta, double *phi, double *psi) {
  long i1, i2;
  double f, ds;

  /* Clamp outside range instead of extrapolating. */

  if (sValue <= sFloor[0]) {
    *X = XFloor[0];
    *Y = YFloor[0];
    *Z = ZFloor[0];
    *theta = thetaFloor[0];
    *phi = phiFloor[0];
    *psi = psiFloor[0];
    *iFloor = 0;
    return;
  }

  if (sValue >= sFloor[nFloor - 1]) {
    *X = XFloor[nFloor - 1];
    *Y = YFloor[nFloor - 1];
    *Z = ZFloor[nFloor - 1];
    *theta = thetaFloor[nFloor - 1];
    *phi = phiFloor[nFloor - 1];
    *psi = psiFloor[nFloor - 1];
    *iFloor = nFloor - 1;
    return;
  }

  if (*iFloor < 0)
    *iFloor = 0;
  while (*iFloor > 0 && sValue < sFloor[*iFloor])
    (*iFloor)--;
  while (*iFloor + 1 < nFloor && sValue > sFloor[*iFloor + 1])
    (*iFloor)++;

  i1 = *iFloor;
  i2 = i1 + 1;
  if (i2 >= nFloor)
    i2 = nFloor - 1;

  ds = sFloor[i2] - sFloor[i1];
  if (ds <= 0)
    f = 0;
  else
    f = (sValue - sFloor[i1]) / ds;

  /* Linear interpolation in arc length s for position and orientation. */
  *X = XFloor[i1] + f * (XFloor[i2] - XFloor[i1]);
  *Y = YFloor[i1] + f * (YFloor[i2] - YFloor[i1]);
  *Z = ZFloor[i1] + f * (ZFloor[i2] - ZFloor[i1]);
  *theta = thetaFloor[i1] + f * (thetaFloor[i2] - thetaFloor[i1]);
  *phi = phiFloor[i1] + f * (phiFloor[i2] - phiFloor[i1]);
  *psi = psiFloor[i1] + f * (psiFloor[i2] - psiFloor[i1]);
}

int main(int argc, char **argv) {
  SDDS_DATASET SDDSfloor, SDDScen, SDDSout;
  char *floorFile, *centroidFile, *outputFile;
  long nFloor, rows, page;
  long i, iFloor;
  int32_t hasCxp, hasCyp;
  FLOOR_POINT *floorPoint;
  /* Reference floor arrays from floorFile. */
  double *sFloor = NULL, *XFloor = NULL, *YFloor = NULL, *ZFloor = NULL, *thetaFloor = NULL, *phiFloor = NULL, *psiFloor = NULL;
  /* Centroid arrays from centroidFile. */
  double *sCen = NULL, *Cx = NULL, *Cy = NULL, *Cxp = NULL, *Cyp = NULL;
  /* Output global path arrays. */
  double *XOut, *YOut, *ZOut, *thetaOut, *thetaRefOut, *phiOut, *psiOut;

  SDDS_RegisterProgramName(argv[0]);

  if (argc != 4)
    bomb(NULL, USAGE);

  floorFile = argv[1];
  centroidFile = argv[2];
  outputFile = argv[3];

  if (!SDDS_InitializeInput(&SDDSfloor, floorFile) || SDDS_ReadPage(&SDDSfloor) <= 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  if ((nFloor = SDDS_CountRowsOfInterest(&SDDSfloor)) < 2)
    SDDS_Bomb("floor file has too few rows");

  if (!(sFloor = SDDS_GetColumnInDoubles(&SDDSfloor, "s")) ||
      !(XFloor = SDDS_GetColumnInDoubles(&SDDSfloor, "X")) ||
      !(YFloor = SDDS_GetColumnInDoubles(&SDDSfloor, "Y")) ||
      !(ZFloor = SDDS_GetColumnInDoubles(&SDDSfloor, "Z")) ||
      !(thetaFloor = SDDS_GetColumnInDoubles(&SDDSfloor, "theta")) ||
      !(phiFloor = SDDS_GetColumnInDoubles(&SDDSfloor, "phi")) ||
      !(psiFloor = SDDS_GetColumnInDoubles(&SDDSfloor, "psi")))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  /*
    Some floor files may not be monotonically ordered in s.  Sort by s so
    interpolation is well-defined and we can use a forward-running interval index.
  */
  floorPoint = tmalloc(sizeof(*floorPoint) * nFloor);
  for (i = 0; i < nFloor; i++) {
    floorPoint[i].s = sFloor[i];
    floorPoint[i].X = XFloor[i];
    floorPoint[i].Y = YFloor[i];
    floorPoint[i].Z = ZFloor[i];
    floorPoint[i].theta = thetaFloor[i];
    floorPoint[i].phi = phiFloor[i];
    floorPoint[i].psi = psiFloor[i];
  }

  qsort((void *)floorPoint, nFloor, sizeof(*floorPoint), compare_floor_points);

  for (i = 0; i < nFloor; i++) {
    sFloor[i] = floorPoint[i].s;
    XFloor[i] = floorPoint[i].X;
    YFloor[i] = floorPoint[i].Y;
    ZFloor[i] = floorPoint[i].Z;
    thetaFloor[i] = floorPoint[i].theta;
    phiFloor[i] = floorPoint[i].phi;
    psiFloor[i] = floorPoint[i].psi;
  }
  free(floorPoint);

  for (i = 1; i < nFloor; i++) {
    /* Make angular arrays continuous for interpolation across wrap boundaries. */
    thetaFloor[i] = unwrap_angle(thetaFloor[i], thetaFloor[i - 1]);
    phiFloor[i] = unwrap_angle(phiFloor[i], phiFloor[i - 1]);
    psiFloor[i] = unwrap_angle(psiFloor[i], psiFloor[i - 1]);
  }

  if (!SDDS_InitializeInput(&SDDScen, centroidFile))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  hasCxp = SDDS_GetColumnIndex(&SDDScen, "Cxp") >= 0;
  hasCyp = SDDS_GetColumnIndex(&SDDScen, "Cyp") >= 0;

  if (!SDDS_InitializeOutput(&SDDSout, SDDS_BINARY, 1,
                             "global beam path coordinates", "global beam path coordinates",
                             outputFile) ||
      !SDDS_TransferAllParameterDefinitions(&SDDSout, &SDDScen, SDDS_TRANSFER_KEEPOLD) ||
      !SDDS_DefineSimpleColumn(&SDDSout, "s", "m", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSout, "X", "m", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSout, "Y", "m", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSout, "Z", "m", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSout, "thetaRef", "radians", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSout, "theta", "radians", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSout, "phi", "radians", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(&SDDSout, "psi", "radians", SDDS_DOUBLE) ||
      !SDDS_WriteLayout(&SDDSout))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  page = 0;
  while (SDDS_ReadPage(&SDDScen) > 0) {
    page++;
    if ((rows = SDDS_CountRowsOfInterest(&SDDScen)) <= 0)
      continue;

    if (!(sCen = SDDS_GetColumnInDoubles(&SDDScen, "s")) ||
        !(Cx = SDDS_GetColumnInDoubles(&SDDScen, "Cx")) ||
        !(Cy = SDDS_GetColumnInDoubles(&SDDScen, "Cy")))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

    Cxp = Cyp = NULL;
    if (hasCxp && !(Cxp = SDDS_GetColumnInDoubles(&SDDScen, "Cxp")))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    if (hasCyp && !(Cyp = SDDS_GetColumnInDoubles(&SDDScen, "Cyp")))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

    XOut = tmalloc(sizeof(*XOut) * rows);
    YOut = tmalloc(sizeof(*YOut) * rows);
    ZOut = tmalloc(sizeof(*ZOut) * rows);
    thetaOut = tmalloc(sizeof(*thetaOut) * rows);
    thetaRefOut = tmalloc(sizeof(*thetaRefOut) * rows);
    phiOut = tmalloc(sizeof(*phiOut) * rows);
    psiOut = tmalloc(sizeof(*psiOut) * rows);

    iFloor = 0;
    for (i = 0; i < rows; i++) {
      double XRef, YRef, ZRef, thetaRef, phiRef, psiRef;
      /* ex/ey/es are local unit vectors at this s location. */
      double ex[3], ey[3], es[3], t[3], norm;
      double cxp = 0, cyp = 0;

      interpolate_floor(sCen[i], &iFloor, nFloor,
                        sFloor, XFloor, YFloor, ZFloor, thetaFloor, phiFloor, psiFloor,
                        &XRef, &YRef, &ZRef, &thetaRef, &phiRef, &psiRef);
      thetaRefOut[i] = thetaRef;

      make_local_basis(thetaRef, phiRef, psiRef, ex, ey, es);

      /*
        Convert local centroid offsets to global Cartesian position:
          r = r_ref + Cx*ex + Cy*ey
      */
      XOut[i] = XRef + Cx[i] * ex[0] + Cy[i] * ey[0];
      YOut[i] = YRef + Cx[i] * ex[1] + Cy[i] * ey[1];
      ZOut[i] = ZRef + Cx[i] * ex[2] + Cy[i] * ey[2];

      if (Cxp)
        cxp = Cxp[i];
      if (Cyp)
        cyp = Cyp[i];

      /*
        Build approximate tangent using local slopes:
          t ~ es + Cxp*ex + Cyp*ey
        then normalize to unit length.
      */
      t[0] = es[0] + cxp * ex[0] + cyp * ey[0];
      t[1] = es[1] + cxp * ex[1] + cyp * ey[1];
      t[2] = es[2] + cxp * ex[2] + cyp * ey[2];
      norm = sqrt(sqr(t[0]) + sqr(t[1]) + sqr(t[2]));
      if (norm <= 0)
        norm = 1;
      t[0] /= norm;
      t[1] /= norm;
      t[2] /= norm;

      /* Convert tangent direction back to (theta,phi). */
      thetaOut[i] = wrap_angle(atan2(t[0], t[2]));
      phiOut[i] = asin(t[1]);

      /* Roll is inherited from the local floor frame at this s location. */
      psiOut[i] = wrap_angle(psiRef);
    }

    if (!SDDS_StartPage(&SDDSout, rows) ||
        !SDDS_CopyParameters(&SDDSout, &SDDScen) ||
        !SDDS_SetColumnFromDoubles(&SDDSout, SDDS_SET_BY_NAME, sCen, rows, "s") ||
        !SDDS_SetColumnFromDoubles(&SDDSout, SDDS_SET_BY_NAME, XOut, rows, "X") ||
        !SDDS_SetColumnFromDoubles(&SDDSout, SDDS_SET_BY_NAME, YOut, rows, "Y") ||
        !SDDS_SetColumnFromDoubles(&SDDSout, SDDS_SET_BY_NAME, ZOut, rows, "Z") ||
        !SDDS_SetColumnFromDoubles(&SDDSout, SDDS_SET_BY_NAME, thetaOut, rows, "theta") ||
        !SDDS_SetColumnFromDoubles(&SDDSout, SDDS_SET_BY_NAME, thetaRefOut, rows, "thetaRef") ||
        !SDDS_SetColumnFromDoubles(&SDDSout, SDDS_SET_BY_NAME, phiOut, rows, "phi") ||
        !SDDS_SetColumnFromDoubles(&SDDSout, SDDS_SET_BY_NAME, psiOut, rows, "psi") ||
        !SDDS_WritePage(&SDDSout))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

    free(sCen);
    free(Cx);
    free(Cy);
    if (Cxp)
      free(Cxp);
    if (Cyp)
      free(Cyp);
    free(XOut);
    free(YOut);
    free(ZOut);
    free(thetaOut);
    free(thetaRefOut);
    free(phiOut);
    free(psiOut);
  }

  free(sFloor);
  free(XFloor);
  free(YFloor);
  free(ZFloor);
  free(thetaFloor);
  free(phiFloor);
  free(psiFloor);

  if (!SDDS_Terminate(&SDDSfloor) || !SDDS_Terminate(&SDDScen) || !SDDS_Terminate(&SDDSout))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  return 0;
}
