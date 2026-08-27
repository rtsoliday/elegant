/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* routine: misalign_matrix()
 * purpose: alter a matrix to reflect misalignments
 * N.B.:  to be used correctly, this routine should only be called AFTER 
 *        the tilt or rotation of the element is applied.
 *
 * Michael Borland, 1991.
 */
#include "mdb.h"
#include "track.h"

VMATRIX *transformMatrixBetweenMomentumAndSlopes(VMATRIX *VM);


//typedef struct {
//  double **a;
//  int n, m;
//} MATRIX;

#undef m_mult
#define m_mult(C,A,B) local_mat_mult(C,A,B)
static int local_mat_mult(MATRIX *C, const MATRIX *A, const MATRIX *B) {
  //double *a_i, *c_i, c_i_j;

  const int m = A->m;
  const int n = A->n;
  const int p = B->m;

  if (m != B->n || n != C->n || p != C->m)
    return 0;

  double c_i_j;
  for (int i=0; i<n; i++) {
    for (int j=0; j<p; j++) {
      c_i_j = 0;
      for (int k=0; k<m; k++) {
        c_i_j += (A->a[i][k])*(B->a[k][j]);
      }
      C->a[i][j] = c_i_j;
    }
  }



  return 1;
}



/* Modify a matrix to include misalignments */
void misalign_matrix(
  VMATRIX *M,                            /* matrix to modify, will be changed */
  double dx, double dy, double dz,       /* displacements */
  double pitch, double yaw, double tilt, /* angular errors, tilt is non-zero only for dipoles */
  double designTilt,                     /* design tilt for dipoles, tilt for straight elements */
  double thetaBend,                      /* bending angle */
  double length,                         /* length */
  short method                           /* 0 = "traditional", 1 = Venturini Linear Entrance-centered
                                            2 = Venturini linear body-centered, 3 = Venturini exact entrance-centered
                                            4 = Venturini exact body-centered
                                         */
) {
  log_entry("misalign_matrix");

  if (method == 0) {
    VMATRIX *M1, *M2, *tmp;
    static MALIGN mal;

    tilt_matrices(M, tilt + designTilt);

    /* entrance misalignment corresponds to opposite tranverse offset of the beam
     * plus propagation of beam forward/backward to the new entrance of the magnet 
     */
    memset(&mal, 0, sizeof(mal));
    mal.dx = -dx;
    mal.dxp = 0;
    mal.dy = -dy;
    mal.dyp = 0;
    mal.dz = dz;
    mal.dp = mal.dt = 0;
    M1 = misalignment_matrix(&mal, M->order);

    /* exit misalignment corresponds to undoing the entrance misalignment of the
     * beam, with consideration given to the fact that the curvilinear coordinate
     * system at the exit may be rotated relative to that at the exit.  This
     * code is incorrect if the bending magnet is tilted!
     */
    mal.dx = dx * cos(thetaBend) + dz * sin(thetaBend);
    mal.dz = dx * sin(thetaBend) - dz * cos(thetaBend);
    mal.dy = dy;
    M2 = misalignment_matrix(&mal, M->order);

    tmp = tmalloc(sizeof(*tmp));
    initialize_matrices(tmp, tmp->order = M->order);

    concat_matrices(tmp, M, M1, 0);
    concat_matrices(M, M2, tmp, 0);

    free_matrices(tmp);
    tfree(tmp);
    tmp = NULL;
    free_matrices(M1);
    tfree(M1);
    M1 = NULL;
    free_matrices(M2);
    tfree(M2);
    M2 = NULL;
  } else {
    VMATRIX *MEntrance, *MExit, *tmp;
    /* We don't use the exact methods here as they don't return the required matrix */
    if (method==2 || method==4) {
      offsetParticlesForBodyCenteredMisalignmentLinearized(&MEntrance, NULL, 0,
                                                           dx, dy, dz,
                                                           pitch, yaw, tilt,
                                                           designTilt, thetaBend, length, 1);
      offsetParticlesForBodyCenteredMisalignmentLinearized(&MExit, NULL, 0,
                                                           dx, dy, dz,
                                                           pitch, yaw, tilt,
                                                           designTilt, thetaBend, length, 2);
    } else {
      offsetParticlesForEntranceCenteredMisalignmentLinearized(&MEntrance, NULL, 0,
                                                               dx, dy, dz,
                                                               pitch, yaw, tilt,
                                                               designTilt, thetaBend, length, 1);
      offsetParticlesForEntranceCenteredMisalignmentLinearized(&MExit, NULL, 0,
                                                               dx, dy, dz,
                                                               pitch, yaw, tilt,
                                                               designTilt, thetaBend, length, 2);
    }

    tmp = tmalloc(sizeof(*tmp));
    initialize_matrices(tmp, tmp->order = M->order);

    concat_matrices(tmp, M, MEntrance, 0);
    concat_matrices(M, MExit, tmp, 0);

    free_matrices(tmp);
    tfree(tmp);
    free_matrices(MEntrance);
    tfree(MEntrance);
    free_matrices(MExit);
    tfree(MExit);
  }
}

/* routine: misalignment_matrix()
 * purpose: provide a matrix for a misalignment
 *
 * Michael Borland, 1991.
 */

VMATRIX *misalignment_matrix(MALIGN *malign, long order) {
  VMATRIX *M;
  static VMATRIX *M1 = NULL;
  double *C, **R;

  log_entry("misalignment_matrix");

  M = tmalloc(sizeof(*M));
  M->order = order;
  initialize_matrices(M, M->order);
  R = M->R;
  C = M->C;

  C[0] = malign->dx;
  C[1] = malign->dxp;
  C[2] = malign->dy;
  C[3] = malign->dyp;
  C[4] = 0;
  /* ignore malign->dt.  Not sure what it really means. */
  C[5] = malign->dp + malign->de; /* missing 1/beta^2 on malign->de term here ... */

  R[0][0] = R[1][1] = R[2][2] = R[3][3] = R[4][4] = R[5][5] = 1;

  if (malign->dz) {
    /* do longitudinal misalignment */
    VMATRIX *tmp;

    if (!M1) {
      M1 = tmalloc(sizeof(*M1));
      initialize_matrices(M1, M1->order = M->order);
    }
    tmp = drift_matrix(malign->dz, M->order);
    concat_matrices(M1, M, tmp, 0);
    free_matrices(tmp);
    tfree(tmp);
    tmp = NULL;
    copy_matrices1(M, M1);
  }
  log_exit("misalignment_matrix");
  return (M);
}

/* We offset the coordinates of a beam to implement the mislignment of the upcoming element by (dx, dy, dz).
 * Hence, the particles are offset by the negative of the given (dx, dy) values, and are drifted an extra
 * distance dz.
 */

void offsetBeamCoordinatesForMisalignment(double **coord, int64_t np, double dx, double dy, double dz) {
  int64_t ip;
  double *part;

  for (ip = np - 1; ip >= 0; ip--) {
    part = coord[ip];
    part[4] += dz * sqrt(1 + sqr(part[1]) + sqr(part[3]));
    part[0] = part[0] - dx + dz * part[1];
    part[2] = part[2] - dy + dz * part[3];
  }
}

/* Compute Hadamard product of two matrices */

double m_hproduct(MATRIX *M1, MATRIX *M2) {
  double dot;
  long i, j;
  if ((M1->n != M2->n) || (M1->m != M2->m))
    bombElegant("matrix mismatch in m_hproduct", NULL);
  dot = 0;
  for (i = 0; i < M1->n; i++)
    for (j = 0; j < M1->m; j++)
      dot += M1->a[i][j] * M2->a[i][j];
  return dot;
}

#define SMALL_ANGLE 1e-12

/* Algorithm due to M. Venturini, ALSU-AP-TN-2021-001. */
/* This is a fairly literal translation of his Mathematica code */
void offsetParticlesForEntranceCenteredMisalignmentLinearized(
  VMATRIX **VM,                    /* if matrix return is desired */
  double **coord, int64_t np,         /* if particle transformation is desired */
  double dx, double dy, double dz, /* error displacements */
  double ax,                       /* error pitch */
  double ay,                       /* error yaw */
  double az,                       /* error roll or tilt */
  double tilt,                     /* design tilt */
  double thetaBend,
  double length,
  short face /* 1 => entry, 2 => exit */
) {
  double cx, cy, cz, sx, sy, sz;
  static MATRIX *xAxis, *yAxis, *zAxis, *dVector, *OP, *OPp;
  static MATRIX *XaxiSxyz, *YaxiSxyz, *ZaxiSxyz;
  static MATRIX *RX, *R, *tmp, *LinMat;
  static MATRIX *tA, *tE, *tD0;
  static MATRIX *V1;           /* working 3x1 matrix */
  static MATRIX *M1, *M2;      /* working 3x3 matrices */
  static MATRIX *t0, *t1, *t2; /* working 6x1 matrices */
  static MATRIX *RB, *RBT, *RXT, *OOp, *OpPp;
  static short initialized = 0;
  static VMATRIX *LVM;
  double LD = 0;
  int64_t ip;
  long i, j;

  if (!initialized) {
    m_alloc(&xAxis, 3, 1);
    m_alloc(&yAxis, 3, 1);
    m_alloc(&zAxis, 3, 1);
    m_alloc(&dVector, 3, 1);
    m_alloc(&OP, 3, 1);
    m_alloc(&R, 3, 3);
    m_alloc(&XaxiSxyz, 3, 1);
    m_alloc(&YaxiSxyz, 3, 1);
    m_alloc(&ZaxiSxyz, 3, 1);
    m_alloc(&tmp, 3, 1);
    m_alloc(&OPp, 3, 1);
    m_alloc(&RX, 3, 3);
    m_alloc(&RB, 3, 3);
    m_alloc(&RBT, 3, 3);
    m_alloc(&RXT, 3, 3);
    m_alloc(&M1, 3, 3);
    m_alloc(&M2, 3, 3);
    m_alloc(&OOp, 3, 1);
    m_alloc(&V1, 3, 1);
    m_alloc(&OpPp, 3, 1);
    m_alloc(&tD0, 6, 1);
    m_alloc(&LinMat, 6, 6);
    m_alloc(&tA, 6, 1);
    m_alloc(&tE, 6, 1);
    m_alloc(&t0, 6, 1);
    m_alloc(&t1, 6, 1);
    m_alloc(&t2, 6, 1);
    LVM = tmalloc(sizeof(*LVM));
    initialize_matrices(LVM, LVM->order = 1);
    initialized = 1;
  }

  /* unit vectors of the xyz coordinate-system axes */
  xAxis->a[0][0] = 1;
  xAxis->a[1][0] = 0;
  xAxis->a[2][0] = 0;
  yAxis->a[0][0] = 0;
  yAxis->a[1][0] = 1;
  yAxis->a[2][0] = 0;
  zAxis->a[0][0] = 0;
  zAxis->a[1][0] = 0;
  zAxis->a[2][0] = 1;

  /* displacement-error vector in the xyz coordinate system */
  dVector->a[0][0] = dx;
  dVector->a[1][0] = dy;
  dVector->a[2][0] = dz;
  if (!m_copy(OP, dVector))
    bombElegant("m_copy(OP, dVector);", NULL);

  /* rotational-error matrix in the xyz coordinate system */
  cx = cos(ax);
  cy = cos(ay);
  cz = cos(az);
  sx = sin(ax);
  sy = sin(ay);
  sz = sin(az);

  RX->a[0][0] = cy * cz;
  RX->a[0][1] = -cy * sz;
  RX->a[0][2] = sy;
  RX->a[1][0] = cz * sx * sy + cx * sz;
  RX->a[1][1] = cx * cz - sx * sy * sz;
  RX->a[1][2] = -cy * sx;
  RX->a[2][0] = -cx * cz * sy + sx * sz;
  RX->a[2][1] = cz * sx + cx * sy * sz;
  RX->a[2][2] = cx * cy;

  if (face == 1) {
    if (!m_copy(R, RX))
      bombElegant("m_copy(R, RX);", NULL);
    if (!m_mult(XaxiSxyz, R, xAxis))
      bombElegant("m_mult(XaxiSxyz, R, xAxis);", NULL);
    if (!m_mult(YaxiSxyz, R, yAxis))
      bombElegant("m_mult(YaxiSxyz, R, yAxis);", NULL);
    if (!m_mult(ZaxiSxyz, R, zAxis))
      bombElegant("m_mult(ZaxiSxyz, R, zAxis);", NULL);
    LD = m_hproduct(ZaxiSxyz, OP);
    if (!m_copy(tmp, OP))
      bombElegant("m_copy(tmp, OP);", NULL);
  } else if (face == 2) {
    RB->a[0][0] = cos(thetaBend);
    RB->a[0][1] = 0;
    RB->a[0][2] = -sin(thetaBend);
    RB->a[1][0] = 0;
    RB->a[1][1] = 1;
    RB->a[1][2] = 0;
    RB->a[2][0] = sin(thetaBend);
    RB->a[2][1] = 0;
    RB->a[2][2] = cos(thetaBend);
    /* rotational-error matrix in the x'y'z' coordinate system */
    if (!m_trans(RBT, RB))
      bombElegant("m_trans(RBT, RB);", NULL);
    if (!m_trans(RXT, RX))
      bombElegant("m_trans(RXT, RX);", NULL);
    if (!m_mult(M1, RBT, RXT))
      bombElegant("m_mult(M1, RBT, RXT);", NULL);
    if (!m_mult(R, M1, RB))
      bombElegant("m_mult(R, M1, RB);", NULL);

    /* XYZ-axes unit-vectors expressed in the x'y'z' coordinate system */
    if (!m_mult(XaxiSxyz, RB, xAxis))
      bombElegant("m_mult(XaxiSxyz, RB, xAxis);", NULL);
    if (!m_mult(YaxiSxyz, RB, yAxis))
      bombElegant("m_mult(YaxiSxyz, RB, yAxis);", NULL);
    if (!m_mult(ZaxiSxyz, RB, zAxis))
      bombElegant("m_mult(ZaxiSxyz, RB, zAxis);", NULL);

    if (fabs(thetaBend) < SMALL_ANGLE) {
      OPp->a[0][0] = 0;
      OPp->a[1][0] = 0;
      OPp->a[2][0] = length;
    } else {
      OPp->a[0][0] = length / thetaBend * (cos(thetaBend) - 1);
      OPp->a[1][0] = 0;
      OPp->a[2][0] = length * sin(thetaBend) / thetaBend;
    }
    if (!m_mult(V1, RX, OPp))
      bombElegant("m_mult(V1, RX, OPp);", NULL);
    if (!m_add(OOp, V1, OP))
      bombElegant("m_add(OOp, V1, OP);", NULL);
    if (!m_subtract(OpPp, OPp, OOp))
      bombElegant("m_subtract(OpPp, OPp, OOp);", NULL);
    LD = m_hproduct(ZaxiSxyz, OpPp);
    if (!m_copy(tmp, OpPp))
      bombElegant("m_copy(tmp, OpPp);", NULL);
  } else
    bombElegantVA("Invalid face code %hd in call to offsetParticlesForEntranceCenteredMisalignmentLinearized", face);

  /* m_show(R, "%13.6e ", "R:\n", stdout); */

  tD0->a[0][0] = -m_hproduct(tmp, XaxiSxyz);
  tD0->a[1][0] = 0;
  tD0->a[2][0] = -m_hproduct(tmp, YaxiSxyz);
  tD0->a[3][0] = tD0->a[4][0] = tD0->a[5][0] = 0;

  LinMat->a[0][0] = R->a[1][1] / R->a[2][2];
  LinMat->a[0][1] = LD * sqrt(R->a[1][1] / R->a[2][2]);
  LinMat->a[0][2] = -R->a[0][1] / R->a[2][2];
  LinMat->a[0][3] = -LD * sqr(R->a[0][1] / R->a[2][2]);
  LinMat->a[0][4] = 0;
  LinMat->a[0][5] = 0;
  LinMat->a[1][0] = 0;
  LinMat->a[1][1] = R->a[0][0];
  LinMat->a[1][2] = 0;
  LinMat->a[1][3] = R->a[1][0];
  LinMat->a[1][4] = 0;
  LinMat->a[1][5] = R->a[2][0];
  LinMat->a[2][0] = -R->a[1][0] / R->a[2][2];
  LinMat->a[2][1] = -LD * sqr(R->a[1][0] / R->a[2][2]);
  LinMat->a[2][2] = R->a[0][0] / R->a[2][2];
  LinMat->a[2][3] = LD * sqr(R->a[0][0] / R->a[2][2]);
  LinMat->a[2][4] = 0;
  LinMat->a[2][5] = 0;
  LinMat->a[3][0] = 0;
  LinMat->a[3][1] = R->a[0][1];
  LinMat->a[3][2] = 0;
  LinMat->a[3][3] = R->a[1][1];
  LinMat->a[3][4] = 0;
  LinMat->a[3][5] = R->a[2][1];
  LinMat->a[4][0] = -R->a[0][2] / R->a[2][2];
  LinMat->a[4][1] = -LD * sqr(R->a[0][2] / R->a[2][2]);
  LinMat->a[4][2] = -R->a[1][2] / R->a[2][2];
  LinMat->a[4][3] = -LD * sqr(R->a[1][2] / R->a[2][2]);
  LinMat->a[4][4] = 1;
  LinMat->a[4][5] = 0;
  LinMat->a[5][0] = 0;
  LinMat->a[5][1] = 0;
  LinMat->a[5][2] = 0;
  LinMat->a[5][3] = 0;
  LinMat->a[5][4] = 0;
  LinMat->a[5][5] = 1;

  t0->a[0][0] = LD * R->a[2][0] / R->a[2][2];
  t0->a[1][0] = R->a[2][0];
  t0->a[2][0] = LD * R->a[2][1] / R->a[2][2];
  t0->a[3][0] = R->a[2][1];
  t0->a[4][0] = LD / R->a[2][2];
  t0->a[5][0] = 0;

  for (i = 0; i < 6; i++) {
    LVM->C[i] = t0->a[i][0] + tD0->a[i][0];
    for (j = 0; j < 6; j++) {
      LVM->R[i][j] = LinMat->a[i][j];
    }
  }

  if (coord && np != 0) {
    double qx, qy, factor;

    if (face==2 && tilt)
      rotateBeamCoordinatesForMisalignment(coord, np, -tilt);

    for (ip = 0; ip < np; ip++) {
      /* convert from (x, xp, y, yp, s, delta) to (x, qx, y, qy, s, delta) */
      factor = (1 + coord[ip][5]) / sqrt(1 + sqr(coord[ip][1]) + sqr(coord[ip][3]));
      coord[ip][1] *= factor;
      coord[ip][3] *= factor;
    }

    track_particles(coord, LVM, coord, np);

    for (ip = 0; ip < np; ip++) {
      /* convert from (x, qx, y, qy, s, delta) to (x, xp, y, yp, s, delta) */
      qx = coord[ip][1];
      qy = coord[ip][3];
      factor = 1 / sqrt(sqr(1 + coord[ip][5]) - sqr(qx) - sqr(qy));
      coord[ip][1] *= factor;
      coord[ip][3] *= factor;
    }

    if (face==1 && tilt)
      rotateBeamCoordinatesForMisalignment(coord, np, tilt);

  }

  /* These matrices are in (x, qx, y, qy, s, delta) coordinates.
   * Transform to (x, xp, y, yp, s, delta).
   * Use 3rd order to get higher precision.
   */
  if (VM) {
    (*VM) = tmalloc(sizeof(**VM));
    initialize_matrices((*VM), (*VM)->order = 3);
    for (i = 0; i < 6; i++) {
      (*VM)->C[i] = t0->a[i][0] + tD0->a[i][0];
      for (j = 0; j < 6; j++) {
        (*VM)->R[i][j] = LinMat->a[i][j];
      }
    }
    /* Transform to slope coordinates */
    transformMatrixBetweenMomentumAndSlopes(*VM);

    if (tilt) {
      VMATRIX *Mt1, *Mt2;
      Mt2 = tmalloc(sizeof(*Mt2));
      initialize_matrices(Mt2, (*VM)->order);
      if (face==2) {
        Mt1 = rotation_matrix(-tilt);
        concat_matrices(Mt2, *VM, Mt1, 0);
        *VM = Mt2;
        free_matrices(Mt1);
      } else {
        Mt1 = rotation_matrix(tilt);
        concat_matrices(Mt2, Mt1, *VM, 0);
        *VM = Mt2;
        free_matrices(Mt1);
      }
    }
  }
}

void offsetParticlesForMisalignment
(
 long mode, /* 0 = original, 1 = linear entrance, 2 = linear body, 3 = exact entrance, 4 = exact body */
 double **coord, int64_t np,
 double dx, double dy, double dz, /* error displacements */
 double ax,                       /* error pitch */
 double ay,                       /* error yaw */
 double az,                       /* error roll or tilt */
 double tilt,                     /* design tilt */
 double thetaBend,
 double length,
 short face /* 1 => entry, 2 => exit */
 ) {
  switch (mode) {
  case 0:
    /* not valid for dipoles! */
    if (face==1) {
      if (dx || dy || dz)
        offsetBeamCoordinatesForMisalignment(coord, np, dx, dy, dz);
      if (tilt || az)
        rotateBeamCoordinatesForMisalignment(coord, np, tilt+az);
    } else {
      if (tilt || az)
        rotateBeamCoordinatesForMisalignment(coord, np, -(tilt+az));
      if (dx || dy || dz)
        offsetBeamCoordinatesForMisalignment(coord, np, -dx, -dy, -dz);
    }
    break;
  case 1:
    offsetParticlesForEntranceCenteredMisalignmentLinearized
      (
       NULL, coord, np, dx, dy, dz, ax, ay, az, tilt, thetaBend, length, face);
    break;
    
  case 2:
    offsetParticlesForBodyCenteredMisalignmentLinearized
      (
       NULL, coord, np, dx, dy, dz, ax, ay, az, tilt, thetaBend, length, face);
    break;

  case 3:
    offsetParticlesForEntranceCenteredMisalignmentExact
      (
       coord, np, dx, dy, dz, ax, ay, az, tilt, thetaBend, length, face);
    break;
    
  case 4:
    offsetParticlesForBodyCenteredMisalignmentExact
      (
       coord, np, dx, dy, dz, ax, ay, az, tilt, thetaBend, length, face);
    break;
    
  }

}

/* Algorithm due to M. Venturini, ALSU-AP-TN-2021-001. */
/* This is a fairly literal translation of his Mathematica code */
void offsetParticlesForEntranceCenteredMisalignmentExact(
  double **coord, int64_t np,
  double dx, double dy, double dz, /* error displacements */
  double ax,                       /* error pitch */
  double ay,                       /* error yaw */
  double az,                       /* error roll or tilt */
  double tilt,                     /* design tilt */
  double thetaBend,
  double length,
  short face /* 1 => entry, 2 => exit */
) {
  double cx, cy, cz, sx, sy, sz;
  static MATRIX *xAxis, *yAxis, *zAxis, *dVector, *OP, *OPp;
  static MATRIX *XaxiSxyz, *YaxiSxyz, *ZaxiSxyz;
  static MATRIX *RX, *R, *tmp, *LinMat;
  static MATRIX *tA, *tE, *tD0;
  static MATRIX *V1, *V2;      /* working 3x1 matrix */
  static MATRIX *M1, *M2;      /* working 3x3 matrices */
  static MATRIX *t0, *t1, *t2; /* working 6x1 matrices */
  static MATRIX *RB, *RBT, *RXT, *OOp, *OpPp;
  static short initialized = 0;
  double LD = 0;
  int64_t ip;
  double sinThetaW, cosThetaW, tanThetaW;

  static MATRIX *etaAxis, *etaPAxis, *etaPPAxis;
  static MATRIX *csiAxis, *csiPAxis, *csiPPAxis;
  static MATRIX *Xaxis, *Yaxis, *Zaxis;
  static MATRIX *rA, *pA, *rD, *pD;
  double tB1, tB2, tB3, tB4, tB5, tB6;
  double tC1, tC2, tC3, tC4, tC5, tC6;
  double tD1, tD2, tD3, tD4, tD5;
  double factor, factor1, factor2, pz, pzP;

  if (!initialized) {
    m_alloc(&tA, 6, 1);
    m_alloc(&xAxis, 3, 1);
    m_alloc(&yAxis, 3, 1);
    m_alloc(&zAxis, 3, 1);
    m_alloc(&dVector, 3, 1);
    m_alloc(&OP, 3, 1);
    m_alloc(&R, 3, 3);
    m_alloc(&XaxiSxyz, 3, 1);
    m_alloc(&YaxiSxyz, 3, 1);
    m_alloc(&ZaxiSxyz, 3, 1);
    m_alloc(&tmp, 3, 1);
    m_alloc(&OPp, 3, 1);
    m_alloc(&RX, 3, 3);
    m_alloc(&RB, 3, 3);
    m_alloc(&RBT, 3, 3);
    m_alloc(&RXT, 3, 3);
    m_alloc(&M1, 3, 3);
    m_alloc(&M2, 3, 3);
    m_alloc(&OOp, 3, 1);
    m_alloc(&V1, 3, 1);
    m_alloc(&V2, 3, 1);
    m_alloc(&OpPp, 3, 1);
    m_alloc(&tD0, 6, 1);
    m_alloc(&LinMat, 6, 6);
    m_alloc(&tE, 6, 1);
    m_alloc(&t0, 6, 1);
    m_alloc(&t1, 6, 1);
    m_alloc(&t2, 6, 1);
    m_alloc(&etaAxis, 3, 1);
    m_alloc(&etaPAxis, 3, 1);
    m_alloc(&etaPPAxis, 3, 1);
    m_alloc(&csiAxis, 3, 1);
    m_alloc(&csiPAxis, 3, 1);
    m_alloc(&csiPPAxis, 3, 1);
    m_alloc(&Xaxis, 3, 1);
    m_alloc(&Yaxis, 3, 1);
    m_alloc(&Zaxis, 3, 1);
    m_alloc(&rA, 3, 1);
    m_alloc(&pA, 3, 1);
    m_alloc(&rD, 3, 1);
    m_alloc(&pD, 3, 1);
    initialized = 1;
  }

  /* unit vectors of the xyz coordinate-system axes */
  xAxis->a[0][0] = 1;
  xAxis->a[1][0] = 0;
  xAxis->a[2][0] = 0;
  yAxis->a[0][0] = 0;
  yAxis->a[1][0] = 1;
  yAxis->a[2][0] = 0;
  zAxis->a[0][0] = 0;
  zAxis->a[1][0] = 0;
  zAxis->a[2][0] = 1;

  /* displacement-error vector in the xyz coordinate system */
  dVector->a[0][0] = dx;
  dVector->a[1][0] = dy;
  dVector->a[2][0] = dz;
  if (!m_copy(OP, dVector))
    bombElegant("m_copy(OP, dVector);", NULL);

  /* rotational-error matrix in the xyz coordinate system */
  cx = cos(ax);
  cy = cos(ay);
  cz = cos(az);
  sx = sin(ax);
  sy = sin(ay);
  sz = sin(az);

  RX->a[0][0] = cy * cz;
  RX->a[0][1] = -cy * sz;
  RX->a[0][2] = sy;
  RX->a[1][0] = cz * sx * sy + cx * sz;
  RX->a[1][1] = cx * cz - sx * sy * sz;
  RX->a[1][2] = -cy * sx;
  RX->a[2][0] = -cx * cz * sy + sx * sz;
  RX->a[2][1] = cz * sx + cx * sy * sz;
  RX->a[2][2] = cx * cy;

  if (face == 1) {
    if (!m_copy(R, RX))
      bombElegant("m_copy(R, RX);", NULL);
  } else if (face == 2) {
    RB->a[0][0] = cos(thetaBend);
    RB->a[0][1] = 0;
    RB->a[0][2] = -sin(thetaBend);
    RB->a[1][0] = 0;
    RB->a[1][1] = 1;
    RB->a[1][2] = 0;
    RB->a[2][0] = sin(thetaBend);
    RB->a[2][1] = 0;
    RB->a[2][2] = cos(thetaBend);

    /* rotational-error matrix in the x'y'z' coordinate system */
    if (!m_trans(RBT, RB))
      bombElegant("m_trans(RBT, RB);", NULL);
    if (!m_trans(RXT, RX))
      bombElegant("m_trans(RXT, RX);", NULL);
    if (!m_mult(M1, RBT, RXT))
      bombElegant("m_mult(M1, RBT, RXT);", NULL);
    if (!m_mult(R, M1, RB))
      bombElegant("m_mult(R, M1, RB);", NULL);
  } else
    bombElegantVA("Invalid face code %hd in call to offsetParticlesForMisalignment", face);

  if (!m_mult(Xaxis, R, xAxis))
    bombElegant("m_mult(Xaxis, R, xAxis);", NULL);
  if (!m_mult(Yaxis, R, yAxis))
    bombElegant("m_mult(Yaxis, R, yAxis);", NULL);
  if (!m_mult(Zaxis, R, zAxis))
    bombElegant("m_mult(Zaxis, R, zAxis);", NULL);

  if (R->a[0][2] == 0) {
    /* R13 == 0 */
    if (R->a[1][2] != 0) {
      /* R23 !=0 */
      if (!m_mult(ZaxiSxyz, R, zAxis))
        bombElegant("m_mult(ZaxiSxyz, R, zAxis)", NULL);
      if (!m_scmul(etaAxis, xAxis, -SIGN(R->a[1][2])))
        bombElegant("m_scmul(etaAxis, xAxis, -SIGN(R->a[1][2]));", NULL);
      if (!m_scmul(csiAxis, yAxis, SIGN(R->a[1][2])))
        bombElegant("m_scmul(csiAxis, yAxis, SIGN(R->a[1][2]));", NULL);
      if (!m_scmul(V1, zAxis, -SIGN(R->a[1][2]) * R->a[1][2]))
        bombElegant("m_scmul(V1, zAxis, -SIGN(R->a[1][2])*R->a[1][2)];", NULL);
      if (!m_scmul(V2, yAxis, SIGN(R->a[1][2]) * R->a[2][2]))
        bombElegant("m_scmul(V2, yAxis, SIGN(R->a[1][2])*R->a[2][2)];", NULL);
      if (!m_add(csiPAxis, V1, V2))
        bombElegant("m_add(csiPAxis, V1, V)2;", NULL);
      sinThetaW = -SIGN(R->a[1][2]) * R->a[1][2];
      cosThetaW = sqrt(1 - sqr(sinThetaW));
      tanThetaW = sinThetaW / cosThetaW;
    } else {
      sinThetaW = tanThetaW = 0;
      cosThetaW = 1;
      if (!m_copy(etaAxis, xAxis))
        bombElegant("m_copy(etaAxis, xAxi)s;", NULL);
      if (!m_copy(csiAxis, yAxis))
        bombElegant("m_copy(csiAxis, yAxi)s;", NULL);
      if (!m_copy(csiPAxis, yAxis))
        bombElegant("m_copy(csiPAxis, yAxi)s;", NULL);
    }
  } else {
    /* R13 !=0 */
    sinThetaW = -SIGN(R->a[0][2]) * sqrt(sqr(R->a[0][2]) + sqr(R->a[1][2]));
    cosThetaW = sqrt(1 - sqr(sinThetaW));
    tanThetaW = sinThetaW / cosThetaW;
    etaAxis->a[0][0] = R->a[1][2] / sinThetaW;
    etaAxis->a[1][0] = -R->a[0][2] / sinThetaW;
    etaAxis->a[2][0] = 0;
    csiAxis->a[0][0] = -R->a[0][2] / sinThetaW;
    csiAxis->a[1][0] = -R->a[1][2] / sinThetaW;
    csiAxis->a[2][0] = 0;
    csiPAxis->a[0][0] = -R->a[0][2] / tanThetaW;
    csiPAxis->a[1][0] = -R->a[1][2] / tanThetaW;
    csiPAxis->a[2][0] = sinThetaW;
  }
  if (!m_copy(etaPAxis, etaAxis))
    bombElegant("m_copy(etaPAxis, etaAxis);", NULL);
  if (!m_copy(etaPPAxis, etaAxis))
    bombElegant("m_copy(etaPPAxis, etaAxis);", NULL);
  if (!m_copy(csiPPAxis, csiPAxis))
    bombElegant("m_copy(csiPPAxis, csiPAxis);", NULL);

  if (face == 1) {
    /* XYZ-axes unit-vectors expresed in the xyz coordinate system */
    if (!m_copy(XaxiSxyz, Xaxis))
      bombElegant("m_copy(XaxiSxyz, Xaxis);", NULL);
    if (!m_copy(YaxiSxyz, Yaxis))
      bombElegant("m_copy(YaxiSxyz, Yaxis);", NULL);
    LD = m_hproduct(Zaxis, OP);
    if (!m_copy(tmp, OP))
      bombElegant("m_copy(tmp, OP);", NULL);
  } else {
    if (fabs(thetaBend) < SMALL_ANGLE) {
      OPp->a[0][0] = 0;
      OPp->a[1][0] = 0;
      OPp->a[2][0] = length;
    } else {
      OPp->a[0][0] = length / thetaBend * (cos(thetaBend) - 1);
      OPp->a[1][0] = 0;
      OPp->a[2][0] = length * sin(thetaBend) / thetaBend;
    }
    if (!m_mult(XaxiSxyz, RB, xAxis))
      bombElegant("m_mult(XaxiSxyz, RB, xAxis);", NULL);
    if (!m_mult(YaxiSxyz, RB, yAxis))
      bombElegant("m_mult(YaxiSxyz, RB, yAxis);", NULL);
    if (!m_mult(ZaxiSxyz, RB, zAxis))
      bombElegant("m_mult(ZaxiSxyz, RB, zAxis);", NULL);
    if (!m_mult(V1, RX, OPp))
      bombElegant("m_mult(V1, Rx, OPp);", NULL);
    if (!m_add(OOp, V1, OP))
      bombElegant("m_add(OOp, V1, OP);", NULL);
    if (!m_subtract(OpPp, OPp, OOp))
      bombElegant("m_subtract(OpPp, OPp, OOp);", NULL);
    LD = m_hproduct(ZaxiSxyz, OpPp);
    if (!m_copy(tmp, OpPp))
      bombElegant("m_copy(tmp, OpPp);", NULL);
  }

  if (coord && np != 0) {
    double qx, qy;

    if (face==2 && tilt)
      rotateBeamCoordinatesForMisalignment(coord, np, -tilt);

    for (ip = 0; ip < np; ip++) {
      /*
      if (face==1) {
        coord[ip][4] += dz*sqrt(1 + sqr(coord[ip][1]) + sqr(coord[ip][3]));
        coord[ip][0] += -dx + dz*coord[ip][1];
        coord[ip][2] += -dy + dz*coord[ip][3];
      }
      */

      /* convert from (x, xp, y, yp, s, delta) to (x, qx, y, qy, s, delta) */
      factor = (1 + coord[ip][5]) / sqrt(1 + sqr(coord[ip][1]) + sqr(coord[ip][3]));
      qx = coord[ip][1] * factor;
      qy = coord[ip][3] * factor;

      /* This step has different meaning for face=1 and face=2:
         face==1 : change from xyz coordinates to csi-eta coordinates
         face==2 : change from x'y'z' coordinates to csi-eta coordinates
      */

      rA->a[0][0] = coord[ip][0];
      rA->a[1][0] = coord[ip][2];
      rA->a[2][0] = 0;

      pA->a[0][0] = qx;
      pA->a[1][0] = qy;
      pA->a[2][0] = 0;

      tB1 = m_hproduct(csiAxis, rA);
      tB2 = m_hproduct(csiAxis, pA);
      tB3 = m_hproduct(etaAxis, rA);
      tB4 = m_hproduct(etaAxis, pA);
      tB5 = coord[ip][4];
      tB6 = coord[ip][5];

      /* wedge-drift propagation */
      pz = sqrt(sqr(1 + tB6) - sqr(tB2) - sqr(tB4));
      factor1 = tB1 / (1 - tB2 * tanThetaW / pz);
      factor2 = factor1 * tanThetaW / pz;

      tC1 = factor1 / cosThetaW;
      tC2 = tB2 * cosThetaW + pz * sinThetaW;
      tC3 = tB3 + tB4 * factor2;
      tC4 = tB4;
      tC5 = tB5 + (1 + tB6) * factor2;
      tC6 = tB6;

      /* conventional-drift propagation */
      pzP = sqrt(sqr(1 + tC6) - sqr(tC2) - sqr(tC4));
      tD1 = tC1 + LD * tC2 / pzP;
      tD2 = tC2;
      tD3 = tC3 + LD * tC4 / pzP;
      tD4 = tC4;
      tD5 = tC5 + LD * (1 + tC6) / pzP;

      /* This step has different meaing for face=1 and face=2
         face==1: change from csi''-eta'' to XYZ coordinates
         face==2: change from csi''-eta'' to X'Y'Z' coordinates 
      */
      if (!m_scmul(V1, csiPPAxis, tD1))
        bombElegant("m_scmul(V1, csiPPAxis, tD1);", NULL);
      if (!m_scmul(V2, etaPPAxis, tD3))
        bombElegant("m_scmul(V2, etaPPAxis, tD3);", NULL);
      if (!m_add(rD, V1, V2))
        bombElegant("m_add(rD, V1, V2);", NULL);
      if (!m_scmul(V1, csiPPAxis, tD2))
        bombElegant("m_scmul(V1, csiPPAxis, tD2);", NULL);
      if (!m_scmul(V2, etaPPAxis, tD4))
        bombElegant("m_scmul(V2, etaPPAxis, tD4);", NULL);
      if (!m_add(pD, V1, V2))
        bombElegant("m_add(pD, V1, V2);", NULL);
      /* if face==2, X/Y axis are the X/Y-axis unit vectors in the x'y'z' coordinate system
         while X/YaxiSxyz are in the xyz coordinate system
      */
      coord[ip][0] = m_hproduct(rD, Xaxis) - m_hproduct(tmp, XaxiSxyz);
      coord[ip][2] = m_hproduct(rD, Yaxis) - m_hproduct(tmp, YaxiSxyz);
      coord[ip][4] = tD5;
      qx = m_hproduct(pD, Xaxis);
      qy = m_hproduct(pD, Yaxis);
      factor = 1 / sqrt(sqr(1 + coord[ip][5]) - sqr(qx) - sqr(qy));
      coord[ip][1] = qx * factor;
      coord[ip][3] = qy * factor;

      /*
      if (face==2) {
        coord[ip][4] -= dz*sqrt(1 + sqr(coord[ip][1]) + sqr(coord[ip][3]));
        coord[ip][0] -= -dx + dz*coord[ip][1];
        coord[ip][2] -= -dy + dz*coord[ip][3];
      }
      */
    }
    if (face==1 && tilt)
      rotateBeamCoordinatesForMisalignment(coord, np, tilt);

  }
}

/* Algorithm due to M. Venturini, ALSU-AP-TN-2021-001. */
/* This is a fairly literal translation of his Mathematica code */
void offsetParticlesForBodyCenteredMisalignmentLinearized(
  VMATRIX **VM,            /* if matrix return is desired */
  double **coord, int64_t np, /* if particle transformation is desired */
  /* error displacements */
  double dx0, double dy0, double dz0,
  double ax0,  /* error pitch */
  double ay0,  /* error yaw */
  double az0,  /* error roll or tilt */
  double tilt, /* design tilt */
  double thetaBend,
  double length,
  short face /* 1 => entry, 2 => exit */
) {
  double cx, cy, cz, sx, sy, sz;
  static MATRIX *xAxis, *yAxis, *zAxis, *d0Vector, *OP, *OPp;
  static MATRIX *XaxiSxyz, *YaxiSxyz, *ZaxiSxyz;
  static MATRIX *RX, *R, *R0, *RB2, *tmp, *LinMat;
  static MATRIX *tA, *tE, *tD0;
  static MATRIX *V1, *V2;      /* working 3x1 matrices */
  static MATRIX *M1, *M2;      /* working 3x3 matrices */
  static MATRIX *t0, *t1, *t2; /* working 6x1 matrices */
  static MATRIX *RB, *RBT, *RXT, *OOp, *OpPp, *OO0, *P0P;
  static short initialized = 0;
  double LD = 0, Rc;
  int64_t ip;
  long i, j;
  static VMATRIX *LVM;

  if (!initialized) {
    m_alloc(&xAxis, 3, 1);
    m_alloc(&yAxis, 3, 1);
    m_alloc(&zAxis, 3, 1);
    m_alloc(&d0Vector, 3, 1);
    m_alloc(&OP, 3, 1);
    m_alloc(&R, 3, 3);
    m_alloc(&XaxiSxyz, 3, 1);
    m_alloc(&YaxiSxyz, 3, 1);
    m_alloc(&ZaxiSxyz, 3, 1);
    m_alloc(&tmp, 3, 1);
    m_alloc(&OPp, 3, 1);
    m_alloc(&RX, 3, 3);
    m_alloc(&R0, 3, 3);
    m_alloc(&RB, 3, 3);
    m_alloc(&RB2, 3, 3);
    m_alloc(&RBT, 3, 3);
    m_alloc(&RXT, 3, 3);
    m_alloc(&M1, 3, 3);
    m_alloc(&M2, 3, 3);
    m_alloc(&OOp, 3, 1);
    m_alloc(&V1, 3, 1);
    m_alloc(&V2, 3, 1);
    m_alloc(&OpPp, 3, 1);
    m_alloc(&tD0, 6, 1);
    m_alloc(&LinMat, 6, 6);
    m_alloc(&tA, 6, 1);
    m_alloc(&tE, 6, 1);
    m_alloc(&t0, 6, 1);
    m_alloc(&t1, 6, 1);
    m_alloc(&t2, 6, 1);
    m_alloc(&OO0, 3, 1);
    m_alloc(&P0P, 3, 1);
    LVM = tmalloc(sizeof(*LVM));
    initialize_matrices(LVM, LVM->order = 1);
    initialized = 1;
  }

  /* unit vectors of the xyz coordinate-system axes */
  xAxis->a[0][0] = 1;
  xAxis->a[1][0] = 0;
  xAxis->a[2][0] = 0;
  yAxis->a[0][0] = 0;
  yAxis->a[1][0] = 1;
  yAxis->a[2][0] = 0;
  zAxis->a[0][0] = 0;
  zAxis->a[1][0] = 0;
  zAxis->a[2][0] = 1;

  /* displacement-error vector in the xyz coordinate system */
  d0Vector->a[0][0] = dx0;
  d0Vector->a[1][0] = dy0;
  d0Vector->a[2][0] = dz0;
  if (!m_copy(OP, d0Vector))
    bombElegant("m_copy(OP, d0Vector);", NULL);
  if (fabs(thetaBend) >= SMALL_ANGLE)
    Rc = length / thetaBend;
  else
    Rc = 0; /* not actually used */

  /* rotational-error matrix in the x0y0z0 coordinate system */
  cx = cos(ax0);
  cy = cos(ay0);
  cz = cos(az0);
  sx = sin(ax0);
  sy = sin(ay0);
  sz = sin(az0);

  R0->a[0][0] = cy * cz;
  R0->a[0][1] = -cy * sz;
  R0->a[0][2] = sy;
  R0->a[1][0] = cz * sx * sy + cx * sz;
  R0->a[1][1] = cx * cz - sx * sy * sz;
  R0->a[1][2] = -cy * sx;
  R0->a[2][0] = -cx * cz * sy + sx * sz;
  R0->a[2][1] = cz * sx + cx * sy * sz;
  R0->a[2][2] = cx * cy;

  RB2->a[0][0] = cos(thetaBend / 2);
  RB2->a[0][1] = 0;
  RB2->a[0][2] = -sin(thetaBend / 2);
  RB2->a[1][0] = 0;
  RB2->a[1][1] = 1;
  RB2->a[1][2] = 0;
  RB2->a[2][0] = sin(thetaBend / 2);
  RB2->a[2][1] = 0;
  RB2->a[2][2] = cos(thetaBend / 2);

  if (!m_trans(M2, RB2))
    bombElegant("m_trans(M2, RB2);", NULL);
  if (!m_mult(M1, R0, M2))
    bombElegant("m_mult(M1, R0, M2);", NULL);
  if (!m_mult(RX, RB2, M1))
    bombElegant("m_mult(RX, RB2, M1);", NULL);

  if (fabs(thetaBend) >= SMALL_ANGLE) {
    if (!m_mult(V1, RB2, zAxis))
      bombElegant("m_mult(V1, RB2, zAxis);", NULL);
    if (!m_scmul(OO0, V1, Rc * sin(thetaBend / 2)))
      bombElegant("m_scmul(OO0, V1, Rc*sin(thetaBend/2));", NULL);
    if (!m_mult(M1, RX, RB2))
      bombElegant("m_mult(M1, RX, RB2);", NULL);
    if (!m_mult(V1, M1, zAxis))
      bombElegant("m_mult(V1, M1, zAxis);", NULL);
    if (!m_scmul(P0P, V1, -Rc * sin(thetaBend / 2)))
      bombElegant("m_scmul(P0P, V1, -Rc*sin(thetaBend/2));", NULL);
  } else {
    if (!m_scmul(OO0, zAxis, length / 2))
      bombElegant("m_scmul(OO0, zAxis, length/2);", NULL);
    if (!m_mult(V1, RX, zAxis))
      bombElegant("m_mult(V1, RX, zAxis);", NULL);
    if (!m_scmul(P0P, V1, -length / 2))
      bombElegant("m_scmul(P0P, V1, -length/2);", NULL);
  }
  if (!m_add(V1, OO0, P0P))
    bombElegant("m_add(V1, OO0, P0P);", NULL);
  if (!m_mult(V2, RB2, d0Vector))
    bombElegant("m_mult(V2, RB2, d0Vector);", NULL);
  if (!m_add(OP, V1, V2))
    bombElegant("m_add(OP, V1, V2);", NULL);

  if (face == 1) {
    if (!m_copy(R, RX))
      bombElegant("m_copy(R, RX);", NULL);
    if (!m_mult(XaxiSxyz, R, xAxis))
      bombElegant("m_mult(XaxiSxyz, R, xAxis);", NULL);
    if (!m_mult(YaxiSxyz, R, yAxis))
      bombElegant("m_mult(YaxiSxyz, R, yAxis);", NULL);
    if (!m_mult(ZaxiSxyz, R, zAxis))
      bombElegant("m_mult(ZaxiSxyz, R, zAxis);", NULL);
    LD = m_hproduct(ZaxiSxyz, OP);
    if (!m_copy(tmp, OP))
      bombElegant("m_copy(tmp, OP);", NULL);
  } else if (face == 2) {
    RB->a[0][0] = cos(thetaBend);
    RB->a[0][1] = 0;
    RB->a[0][2] = -sin(thetaBend);
    RB->a[1][0] = 0;
    RB->a[1][1] = 1;
    RB->a[1][2] = 0;
    RB->a[2][0] = sin(thetaBend);
    RB->a[2][1] = 0;
    RB->a[2][2] = cos(thetaBend);

    /* rotational-error matrix in the x'y'z' coordinate system */
    if (!m_trans(RBT, RB))
      bombElegant("m_trans(RBT, RB);", NULL);
    if (!m_trans(RXT, RX))
      bombElegant("m_trans(RXT, RX);", NULL);
    if (!m_mult(M1, RBT, RXT))
      bombElegant("m_mult(M1, RBT, RXT);", NULL);
    if (!m_mult(R, M1, RB))
      bombElegant("m_mult(R, M1, RB);", NULL);

    /* XYZ-axes unit-vectors expressed in the x'y'z' coordinate system */
    if (!m_mult(XaxiSxyz, RB, xAxis))
      bombElegant("m_mult(XaxiSxyz, RB, xAxis);", NULL);
    if (!m_mult(YaxiSxyz, RB, yAxis))
      bombElegant("m_mult(YaxiSxyz, RB, yAxis);", NULL);
    if (!m_mult(ZaxiSxyz, RB, zAxis))
      bombElegant("m_mult(ZaxiSxyz, RB, zAxis);", NULL);

    if (fabs(thetaBend) < SMALL_ANGLE) {
      OPp->a[0][0] = 0;
      OPp->a[1][0] = 0;
      OPp->a[2][0] = length;
    } else {
      OPp->a[0][0] = length / thetaBend * (cos(thetaBend) - 1);
      OPp->a[1][0] = 0;
      OPp->a[2][0] = length * sin(thetaBend) / thetaBend;
    }
    if (!m_mult(V1, RX, OPp))
      bombElegant("m_mult(V1, RX, OPp);", NULL);
    if (!m_add(OOp, V1, OP))
      bombElegant("m_add(OOp, V1, OP);", NULL);
    if (!m_subtract(OpPp, OPp, OOp))
      bombElegant("m_subtract(OpPp, OPp, OOp);", NULL);
    LD = m_hproduct(ZaxiSxyz, OpPp);
    if (!m_copy(tmp, OpPp))
      bombElegant("m_copy(tmp, OpPp);", NULL);
  } else
    bombElegantVA("Invalid face code %hd in call to offsetParticlesForMisalignment", face);

  /* m_show(R, "%13.6e ", "R:\n", stdout); */

  tD0->a[0][0] = -m_hproduct(tmp, XaxiSxyz);
  tD0->a[1][0] = 0;
  tD0->a[2][0] = -m_hproduct(tmp, YaxiSxyz);
  tD0->a[3][0] = tD0->a[4][0] = tD0->a[5][0] = 0;

  LinMat->a[0][0] = R->a[1][1] / R->a[2][2];
  LinMat->a[0][1] = LD * sqrt(R->a[1][1] / R->a[2][2]);
  LinMat->a[0][2] = -R->a[0][1] / R->a[2][2];
  LinMat->a[0][3] = -LD * sqr(R->a[0][1] / R->a[2][2]);
  LinMat->a[0][4] = 0;
  LinMat->a[0][5] = 0;
  LinMat->a[1][0] = 0;
  LinMat->a[1][1] = R->a[0][0];
  LinMat->a[1][2] = 0;
  LinMat->a[1][3] = R->a[1][0];
  LinMat->a[1][4] = 0;
  LinMat->a[1][5] = R->a[2][0];
  LinMat->a[2][0] = -R->a[1][0] / R->a[2][2];
  LinMat->a[2][1] = -LD * sqr(R->a[1][0] / R->a[2][2]);
  LinMat->a[2][2] = R->a[0][0] / R->a[2][2];
  LinMat->a[2][3] = LD * sqr(R->a[0][0] / R->a[2][2]);
  LinMat->a[2][4] = 0;
  LinMat->a[2][5] = 0;
  LinMat->a[3][0] = 0;
  LinMat->a[3][1] = R->a[0][1];
  LinMat->a[3][2] = 0;
  LinMat->a[3][3] = R->a[1][1];
  LinMat->a[3][4] = 0;
  LinMat->a[3][5] = R->a[2][1];
  LinMat->a[4][0] = -R->a[0][2] / R->a[2][2];
  LinMat->a[4][1] = -LD * sqr(R->a[0][2] / R->a[2][2]);
  LinMat->a[4][2] = -R->a[1][2] / R->a[2][2];
  LinMat->a[4][3] = -LD * sqr(R->a[1][2] / R->a[2][2]);
  LinMat->a[4][4] = 1;
  LinMat->a[4][5] = 0;
  LinMat->a[5][0] = 0;
  LinMat->a[5][1] = 0;
  LinMat->a[5][2] = 0;
  LinMat->a[5][3] = 0;
  LinMat->a[5][4] = 0;
  LinMat->a[5][5] = 1;

  t0->a[0][0] = LD * R->a[2][0] / R->a[2][2];
  t0->a[1][0] = R->a[2][0];
  t0->a[2][0] = LD * R->a[2][1] / R->a[2][2];
  t0->a[3][0] = R->a[2][1];
  t0->a[4][0] = LD / R->a[2][2];
  t0->a[5][0] = 0;

  for (i = 0; i < 6; i++) {
    LVM->C[i] = t0->a[i][0] + tD0->a[i][0];
    for (j = 0; j < 6; j++) {
      LVM->R[i][j] = LinMat->a[i][j];
    }
  }

  if (coord && np != 0) {
    double qx, qy, factor;

    if (face==2 && tilt)
      rotateBeamCoordinatesForMisalignment(coord, np, -tilt);

    for (ip = 0; ip < np; ip++) {
      /* convert from (x, xp, y, yp, s, delta) to (x, qx, y, qy, s, delta) */
      factor = (1 + coord[ip][5]) / sqrt(1 + sqr(coord[ip][1]) + sqr(coord[ip][3]));
      coord[ip][1] *= factor;
      coord[ip][3] *= factor;
    }

    track_particles(coord, LVM, coord, np);

    for (ip = 0; ip < np; ip++) {
      /* convert from (x, qx, y, qy, s, delta) to (x, xp, y, yp, s, delta) */
      qx = coord[ip][1];
      qy = coord[ip][3];
      factor = 1 / sqrt(sqr(1 + coord[ip][5]) - sqr(qx) - sqr(qy));
      coord[ip][1] *= factor;
      coord[ip][3] *= factor;
    }

    if (face==1 && tilt)
      rotateBeamCoordinatesForMisalignment(coord, np, tilt);

  }

  /* These matrices are in (x, qx, y, qy, s, delta) coordinates.
   * Transform to (x, xp, y, yp, s, delta).
   * Use 3rd order to get higher precision.
   */
  if (VM) {
    (*VM) = tmalloc(sizeof(**VM));
    initialize_matrices((*VM), (*VM)->order = 3);
    for (i = 0; i < 6; i++) {
      (*VM)->C[i] = t0->a[i][0] + tD0->a[i][0];
      for (j = 0; j < 6; j++) {
        (*VM)->R[i][j] = LinMat->a[i][j];
      }
    }
    /* Transform to slope coordinates */
    transformMatrixBetweenMomentumAndSlopes(*VM);

    if (tilt) {
      VMATRIX *Mt1, *Mt2;
      Mt2 = tmalloc(sizeof(*Mt2));
      initialize_matrices(Mt2, (*VM)->order);
      if (face==2) {
        Mt1 = rotation_matrix(-tilt);
        concat_matrices(Mt2, *VM, Mt1, 0);
        *VM = Mt2;
        free_matrices(Mt1);
      } else {
        Mt1 = rotation_matrix(tilt);
        concat_matrices(Mt2, Mt1, *VM, 0);
        *VM = Mt2;
        free_matrices(Mt1);
      }
    }
  }
}

VMATRIX *transformMatrixBetweenMomentumAndSlopes(VMATRIX *VM) {
  VMATRIX *X, *tmp;
  long i, order;

  order = VM->order;

  X = tmalloc(sizeof(*X));
  initialize_matrices(X, X->order = 3);
  tmp = tmalloc(sizeof(*tmp));
  initialize_matrices(tmp, tmp->order = 3);

  for (i = 0; i < 6; i++)
    X->R[i][i] = 1;

  /* From (x, xp, y, yp, s, delta) to (x, qx, y, qy, s, delta)
     T262 = 1
     T464 = 1
     U2222 = -3
     U2442 = -1
     U4422 = -1
     U4444 = -3
  */
  X->T[1][5][1] = X->T[3][5][3] = 1;
  X->Q[1][1][1][1] = X->Q[3][3][3][3] = -3;
  X->Q[1][3][3][1] = X->Q[3][3][1][1] = -1;
  concat_matrices(tmp, VM, X, 0);

  /* From (x, qx, y, qy, s, delta) to (x, xp, y, yp, s, delta)
     T262 = -1
     T464 = -1
     U2222 = 3
     U2442 = 1
     U2662 = 2
     U4422 = 1
     U4444 = 3
     U4664 = 2
  */

  X->T[1][5][1] = X->T[3][5][3] = -1;
  X->Q[1][1][1][1] = X->Q[3][3][3][3] = 3;
  X->Q[1][3][3][1] = X->Q[3][3][1][1] = 1;
  X->Q[1][5][5][1] = X->Q[3][5][5][3] = 2;
  concat_matrices(VM, X, tmp, 0);

  free_matrices(X);
  free_matrices(tmp);
  free(X);

  if (VM->order > order) {
    long tmpOrder;
    /* copy just the desired order */
    tmpOrder = VM->order;
    VM->order = order;
    copy_matrices(tmp, VM);
    VM->order = tmpOrder;
    free_matrices(VM);
    free(VM);
    return tmp;
  } else {
    free(tmp);
    return VM;
  }
}

/* Algorithm due to M. Venturini, ALSU-AP-TN-2021-001. */
/* This is a fairly literal translation of his Mathematica code, modified
 * to perform the misalignment in the body-centered frame 
 */
void offsetParticlesForBodyCenteredMisalignmentExact(
  double **coord, int64_t np,
  double dx0, double dy0, double dz0,
  double ax0,  /* error pitch */
  double ay0,  /* error yaw */
  double az0,  /* error roll or tilt */
  double tilt, /* design tilt */
  double thetaBend,
  double length,
  short face /* 1 => entry, 2 => exit */
) {
  double cx, cy, cz, sx, sy, sz;
  static MATRIX *xAxis, *yAxis, *zAxis, *d0Vector, *OP, *OPp;
  static MATRIX *XaxiSxyz, *YaxiSxyz, *ZaxiSxyz;
  static MATRIX *RX, *R, *tmp, *LinMat;
  static MATRIX *tA, *tE, *tD0;
  static MATRIX *V1, *V2;      /* working 3x1 matrix */
  static MATRIX *M1, *M2;      /* working 3x3 matrices */
  static MATRIX *t0, *t1, *t2; /* working 6x1 matrices */
  static MATRIX *R0, *RB, *RB2, *RBT, *RXT, *OOp, *OpPp, *OO0, *O0P0, *P0P;
  static short initialized = 0;
  double LD = 0;
  int64_t ip;
  double sinThetaW, cosThetaW, tanThetaW;

  static MATRIX *etaAxis, *etaPAxis, *etaPPAxis;
  static MATRIX *csiAxis, *csiPAxis, *csiPPAxis;
  static MATRIX *Xaxis, *Yaxis, *Zaxis;
  static MATRIX *rA, *pA, *rD, *pD;
  double tB1, tB2, tB3, tB4, tB5, tB6;
  double tC1, tC2, tC3, tC4, tC5, tC6;
  double tD1, tD2, tD3, tD4, tD5;
  double factor1, factor2, pz, pzP, Rc;

  if (!initialized) {
    m_alloc(&tA, 6, 1);
    m_alloc(&xAxis, 3, 1);
    m_alloc(&yAxis, 3, 1);
    m_alloc(&zAxis, 3, 1);
    m_alloc(&d0Vector, 3, 1);
    m_alloc(&OP, 3, 1);
    m_alloc(&R, 3, 3);
    m_alloc(&XaxiSxyz, 3, 1);
    m_alloc(&YaxiSxyz, 3, 1);
    m_alloc(&ZaxiSxyz, 3, 1);
    m_alloc(&tmp, 3, 1);
    m_alloc(&OPp, 3, 1);
    m_alloc(&R0, 3, 3);
    m_alloc(&RX, 3, 3);
    m_alloc(&RB, 3, 3);
    m_alloc(&RB2, 3, 3);
    m_alloc(&RBT, 3, 3);
    m_alloc(&RXT, 3, 3);
    m_alloc(&M1, 3, 3);
    m_alloc(&M2, 3, 3);
    m_alloc(&OOp, 3, 1);
    m_alloc(&OpPp, 3, 1);
    m_alloc(&OO0, 3, 1);
    m_alloc(&O0P0, 3, 1);
    m_alloc(&P0P, 3, 1);
    m_alloc(&V1, 3, 1);
    m_alloc(&V2, 3, 1);
    m_alloc(&tD0, 6, 1);
    m_alloc(&LinMat, 6, 6);
    m_alloc(&tE, 6, 1);
    m_alloc(&t0, 6, 1);
    m_alloc(&t1, 6, 1);
    m_alloc(&t2, 6, 1);
    m_alloc(&etaAxis, 3, 1);
    m_alloc(&etaPAxis, 3, 1);
    m_alloc(&etaPPAxis, 3, 1);
    m_alloc(&csiAxis, 3, 1);
    m_alloc(&csiPAxis, 3, 1);
    m_alloc(&csiPPAxis, 3, 1);
    m_alloc(&Xaxis, 3, 1);
    m_alloc(&Yaxis, 3, 1);
    m_alloc(&Zaxis, 3, 1);
    m_alloc(&rA, 3, 1);
    m_alloc(&pA, 3, 1);
    m_alloc(&rD, 3, 1);
    m_alloc(&pD, 3, 1);
    initialized = 1;
  }

  /* unit vectors of the xyz coordinate-system axes */
  xAxis->a[0][0] = 1;
  xAxis->a[1][0] = 0;
  xAxis->a[2][0] = 0;
  yAxis->a[0][0] = 0;
  yAxis->a[1][0] = 1;
  yAxis->a[2][0] = 0;
  zAxis->a[0][0] = 0;
  zAxis->a[1][0] = 0;
  zAxis->a[2][0] = 1;

  /* displacement-error vector in the xyz coordinate system */
  d0Vector->a[0][0] = dx0;
  d0Vector->a[1][0] = dy0;
  d0Vector->a[2][0] = dz0;
  if (!m_copy(OP, d0Vector))
    bombElegant("m_copy(OP, d0Vector);", NULL);
  if (fabs(thetaBend) >= SMALL_ANGLE)
    Rc = length / thetaBend;
  else
    Rc = 0; /* not actually used */

  /* rotational-error matrix in the x0y0z0 coordinate system */
  cx = cos(ax0);
  cy = cos(ay0);
  cz = cos(az0);
  sx = sin(ax0);
  sy = sin(ay0);
  sz = sin(az0);

  R0->a[0][0] = cy * cz;
  R0->a[0][1] = -cy * sz;
  R0->a[0][2] = sy;
  R0->a[1][0] = cz * sx * sy + cx * sz;
  R0->a[1][1] = cx * cz - sx * sy * sz;
  R0->a[1][2] = -cy * sx;
  R0->a[2][0] = -cx * cz * sy + sx * sz;
  R0->a[2][1] = cz * sx + cx * sy * sz;
  R0->a[2][2] = cx * cy;

  RB2->a[0][0] = cos(thetaBend / 2);
  RB2->a[0][1] = 0;
  RB2->a[0][2] = -sin(thetaBend / 2);
  RB2->a[1][0] = 0;
  RB2->a[1][1] = 1;
  RB2->a[1][2] = 0;
  RB2->a[2][0] = sin(thetaBend / 2);
  RB2->a[2][1] = 0;
  RB2->a[2][2] = cos(thetaBend / 2);

  if (!m_trans(M2, RB2))
    bombElegant("m_trans(M2, RB2);", NULL);
  if (!m_mult(M1, R0, M2))
    bombElegant("m_mult(M1, R0, M2);", NULL);
  if (!m_mult(RX, RB2, M1))
    bombElegant("m_mult(RX, RB2, M1);", NULL);

  if (fabs(thetaBend) >= SMALL_ANGLE) {
    if (!m_mult(V1, RB2, zAxis))
      bombElegant("m_mult(V1, RB2, zAxis);", NULL);
    if (!m_scmul(OO0, V1, Rc * sin(thetaBend / 2)))
      bombElegant("m_scmul(OO0, V1, Rc*sin(thetaBend/2));", NULL);
    if (!m_mult(M1, RX, RB2))
      bombElegant("m_mult(M1, RX, RB2);", NULL);
    if (!m_mult(V1, M1, zAxis))
      bombElegant("m_mult(V1, M1, zAxis);", NULL);
    if (!m_scmul(P0P, V1, -Rc * sin(thetaBend / 2)))
      bombElegant("m_scmul(P0P, V1, -Rc*sin(thetaBend/2));", NULL);
  } else {
    if (!m_scmul(OO0, zAxis, length / 2))
      bombElegant("m_scmul(OO0, zAxis, length/2);", NULL);
    if (!m_mult(V1, RX, zAxis))
      bombElegant("m_mult(V1, RX, zAxis);", NULL);
    if (!m_scmul(P0P, V1, -length / 2))
      bombElegant("m_scmul(P0P, V1, -length/2);", NULL);
  }
  if (!m_add(V1, OO0, P0P))
    bombElegant("m_add(V1, OO0, P0P);", NULL);
  if (!m_mult(V2, RB2, d0Vector))
    bombElegant("m_mult(V2, RB2, d0Vector);", NULL);
  if (!m_add(OP, V1, V2))
    bombElegant("m_add(OP, V1, V2);", NULL);

  if (face == 1) {
    if (!m_copy(R, RX))
      bombElegant("m_copy(R, RX);", NULL);
    if (!m_mult(XaxiSxyz, R, xAxis))
      bombElegant("m_mult(XaxiSxyz, R, xAxis);", NULL);
    if (!m_mult(YaxiSxyz, R, yAxis))
      bombElegant("m_mult(YaxiSxyz, R, yAxis);", NULL);
    if (!m_mult(ZaxiSxyz, R, zAxis))
      bombElegant("m_mult(ZaxiSxyz, R, zAxis);", NULL);
    LD = m_hproduct(ZaxiSxyz, OP);
    if (!m_copy(tmp, OP))
      bombElegant("m_copy(tmp, OP);", NULL);
  } else if (face == 2) {
    RB->a[0][0] = cos(thetaBend);
    RB->a[0][1] = 0;
    RB->a[0][2] = -sin(thetaBend);
    RB->a[1][0] = 0;
    RB->a[1][1] = 1;
    RB->a[1][2] = 0;
    RB->a[2][0] = sin(thetaBend);
    RB->a[2][1] = 0;
    RB->a[2][2] = cos(thetaBend);

    /* rotational-error matrix in the x'y'z' coordinate system */
    if (!m_trans(RBT, RB))
      bombElegant("m_trans(RBT, RB);", NULL);
    if (!m_trans(RXT, RX))
      bombElegant("m_trans(RXT, RX);", NULL);
    if (!m_mult(M1, RBT, RXT))
      bombElegant("m_mult(M1, RBT, RXT);", NULL);
    if (!m_mult(R, M1, RB))
      bombElegant("m_mult(R, M1, RB);", NULL);

    /* XYZ-axes unit-vectors expressed in the x'y'z' coordinate system */
    if (!m_mult(XaxiSxyz, RB, xAxis))
      bombElegant("m_mult(XaxiSxyz, RB, xAxis);", NULL);
    if (!m_mult(YaxiSxyz, RB, yAxis))
      bombElegant("m_mult(YaxiSxyz, RB, yAxis);", NULL);
    if (!m_mult(ZaxiSxyz, RB, zAxis))
      bombElegant("m_mult(ZaxiSxyz, RB, zAxis);", NULL);

    if (fabs(thetaBend) < SMALL_ANGLE) {
      OPp->a[0][0] = 0;
      OPp->a[1][0] = 0;
      OPp->a[2][0] = length;
    } else {
      OPp->a[0][0] = length / thetaBend * (cos(thetaBend) - 1);
      OPp->a[1][0] = 0;
      OPp->a[2][0] = length * sin(thetaBend) / thetaBend;
    }
    if (!m_mult(V1, RX, OPp))
      bombElegant("m_mult(V1, RX, OPp);", NULL);
    if (!m_add(OOp, V1, OP))
      bombElegant("m_add(OOp, V1, OP);", NULL);
    if (!m_subtract(OpPp, OPp, OOp))
      bombElegant("m_subtract(OpPp, OPp, OOp);", NULL);
    LD = m_hproduct(ZaxiSxyz, OpPp);
    if (!m_copy(tmp, OpPp))
      bombElegant("m_copy(tmp, OpPp);", NULL);
  } else
    bombElegantVA("Invalid face code %hd in call to offsetParticlesForMisalignment", face);

  if (R->a[0][2] == 0) {
    /* R13 == 0 */
    if (R->a[1][2] != 0) {
      /* R23 !=0 */
      if (!m_mult(ZaxiSxyz, R, zAxis))
        bombElegant("m_mult(ZaxiSxyz, R, zAxis)", NULL);
      if (!m_scmul(etaAxis, xAxis, -SIGN(R->a[1][2])))
        bombElegant("m_scmul(etaAxis, xAxis, -SIGN(R->a[1][2]));", NULL);
      if (!m_scmul(csiAxis, yAxis, SIGN(R->a[1][2])))
        bombElegant("m_scmul(csiAxis, yAxis, SIGN(R->a[1][2]));", NULL);
      if (!m_scmul(V1, zAxis, -SIGN(R->a[1][2]) * R->a[1][2]))
        bombElegant("m_scmul(V1, zAxis, -SIGN(R->a[1][2])*R->a[1][2)];", NULL);
      if (!m_scmul(V2, yAxis, SIGN(R->a[1][2]) * R->a[2][2]))
        bombElegant("m_scmul(V2, yAxis, SIGN(R->a[1][2])*R->a[2][2)];", NULL);
      if (!m_add(csiPAxis, V1, V2))
        bombElegant("m_add(csiPAxis, V1, V)2;", NULL);
      sinThetaW = -SIGN(R->a[1][2]) * R->a[1][2];
      cosThetaW = sqrt(1 - sqr(sinThetaW));
      tanThetaW = sinThetaW / cosThetaW;
    } else {
      sinThetaW = tanThetaW = 0;
      cosThetaW = 1;
      if (!m_copy(etaAxis, xAxis))
        bombElegant("m_copy(etaAxis, xAxi)s;", NULL);
      if (!m_copy(csiAxis, yAxis))
        bombElegant("m_copy(csiAxis, yAxi)s;", NULL);
      if (!m_copy(csiPAxis, yAxis))
        bombElegant("m_copy(csiPAxis, yAxi)s;", NULL);
    }
  } else {
    /* R13 !=0 */
    sinThetaW = -SIGN(R->a[0][2]) * sqrt(sqr(R->a[0][2]) + sqr(R->a[1][2]));
    cosThetaW = sqrt(1 - sqr(sinThetaW));
    tanThetaW = sinThetaW / cosThetaW;
    etaAxis->a[0][0] = R->a[1][2] / sinThetaW;
    etaAxis->a[1][0] = -R->a[0][2] / sinThetaW;
    etaAxis->a[2][0] = 0;
    csiAxis->a[0][0] = -R->a[0][2] / sinThetaW;
    csiAxis->a[1][0] = -R->a[1][2] / sinThetaW;
    csiAxis->a[2][0] = 0;
    csiPAxis->a[0][0] = -R->a[0][2] / tanThetaW;
    csiPAxis->a[1][0] = -R->a[1][2] / tanThetaW;
    csiPAxis->a[2][0] = sinThetaW;
  }
  if (!m_copy(etaPAxis, etaAxis))
    bombElegant("m_copy(etaPAxis, etaAxis);", NULL);
  if (!m_copy(etaPPAxis, etaAxis))
    bombElegant("m_copy(etaPPAxis, etaAxis);", NULL);
  if (!m_copy(csiPPAxis, csiPAxis))
    bombElegant("m_copy(csiPPAxis, csiPAxis);", NULL);

  if (!m_mult(Xaxis, R, xAxis))
    bombElegant("m_mult(Xaxis, R, xAxis);", NULL);
  if (!m_mult(Yaxis, R, yAxis))
    bombElegant("m_mult(Yaxis, R, yAxis);", NULL);
  if (!m_mult(Zaxis, R, zAxis))
    bombElegant("m_mult(Zaxis, R, zAxis);", NULL);

  if (face == 1) {
    /* XYZ-axes unit-vectors expresed in the xyz coordinate system */
    if (!m_copy(XaxiSxyz, Xaxis))
      bombElegant("m_copy(XaxiSxyz, Xaxis);", NULL);
    if (!m_copy(YaxiSxyz, Yaxis))
      bombElegant("m_copy(YaxiSxyz, Yaxis);", NULL);
    LD = m_hproduct(Zaxis, OP);
    if (!m_copy(tmp, OP))
      bombElegant("m_copy(tmp, OP);", NULL);
  } else {
    if (fabs(thetaBend) < SMALL_ANGLE) {
      OPp->a[0][0] = 0;
      OPp->a[1][0] = 0;
      OPp->a[2][0] = length;
    } else {
      OPp->a[0][0] = length / thetaBend * (cos(thetaBend) - 1);
      OPp->a[1][0] = 0;
      OPp->a[2][0] = length * sin(thetaBend) / thetaBend;
    }
    if (!m_mult(XaxiSxyz, RB, xAxis))
      bombElegant("m_mult(XaxiSxyz, RB, xAxis);", NULL);
    if (!m_mult(YaxiSxyz, RB, yAxis))
      bombElegant("m_mult(YaxiSxyz, RB, yAxis);", NULL);
    if (!m_mult(ZaxiSxyz, RB, zAxis))
      bombElegant("m_mult(ZaxiSxyz, RB, zAxis);", NULL);
    if (!m_mult(V1, RX, OPp))
      bombElegant("m_mult(V1, Rx, OPp);", NULL);
    if (!m_add(OOp, V1, OP))
      bombElegant("m_add(OOp, V1, OP);", NULL);
    if (!m_subtract(OpPp, OPp, OOp))
      bombElegant("m_subtract(OpPp, OPp, OOp);", NULL);
    LD = m_hproduct(ZaxiSxyz, OpPp);
    if (!m_copy(tmp, OpPp))
      bombElegant("m_copy(tmp, OpPp);", NULL);
  }

  if (coord && np != 0) {
    double qx, qy, factor;
    if (face==2 && tilt)
      rotateBeamCoordinatesForMisalignment(coord, np, -tilt);

    for (ip = 0; ip < np; ip++) {
      /*
      if (face==1) {
        coord[ip][4] += dz*sqrt(1 + sqr(coord[ip][1]) + sqr(coord[ip][3]));
        coord[ip][0] += -dx + dz*coord[ip][1];
        coord[ip][2] += -dy + dz*coord[ip][3];
      }
      */

      /* convert from (x, xp, y, yp, s, delta) to (x, qx, y, qy, s, delta) */
      factor = (1 + coord[ip][5]) / sqrt(1 + sqr(coord[ip][1]) + sqr(coord[ip][3]));
      qx = coord[ip][1] * factor;
      qy = coord[ip][3] * factor;

      /* This step has different meaning for face=1 and face=2:
         face==1 : change from xyz coordinates to csi-eta coordinates
         face==2 : change from x'y'z' coordinates to csi-eta coordinates
      */

      rA->a[0][0] = coord[ip][0];
      rA->a[1][0] = coord[ip][2];
      rA->a[2][0] = 0;

      pA->a[0][0] = qx;
      pA->a[1][0] = qy;
      pA->a[2][0] = 0;

      tB1 = m_hproduct(csiAxis, rA);
      tB2 = m_hproduct(csiAxis, pA);
      tB3 = m_hproduct(etaAxis, rA);
      tB4 = m_hproduct(etaAxis, pA);
      tB5 = coord[ip][4];
      tB6 = coord[ip][5];

      /* wedge-drift propagation */
      pz = sqrt(sqr(1 + tB6) - sqr(tB2) - sqr(tB4));
      factor1 = tB1 / (1 - tB2 * tanThetaW / pz);
      factor2 = factor1 * tanThetaW / pz;

      tC1 = factor1 / cosThetaW;
      tC2 = tB2 * cosThetaW + pz * sinThetaW;
      tC3 = tB3 + tB4 * factor2;
      tC4 = tB4;
      tC5 = tB5 + (1 + tB6) * factor2;
      tC6 = tB6;

      /* conventional-drift propagation */
      pzP = sqrt(sqr(1 + tC6) - sqr(tC2) - sqr(tC4));
      tD1 = tC1 + LD * tC2 / pzP;
      tD2 = tC2;
      tD3 = tC3 + LD * tC4 / pzP;
      tD4 = tC4;
      tD5 = tC5 + LD * (1 + tC6) / pzP;

      /* This step has different meaing for face=1 and face=2
         face==1: change from csi''-eta'' to XYZ coordinates
         face==2: change from csi''-eta'' to X'Y'Z' coordinates 
      */
      if (!m_scmul(V1, csiPPAxis, tD1))
        bombElegant("m_scmul(V1, csiPPAxis, tD1);", NULL);
      if (!m_scmul(V2, etaPPAxis, tD3))
        bombElegant("m_scmul(V2, etaPPAxis, tD3);", NULL);
      if (!m_add(rD, V1, V2))
        bombElegant("m_add(rD, V1, V2);", NULL);
      if (!m_scmul(V1, csiPPAxis, tD2))
        bombElegant("m_scmul(V1, csiPPAxis, tD2);", NULL);
      if (!m_scmul(V2, etaPPAxis, tD4))
        bombElegant("m_scmul(V2, etaPPAxis, tD4);", NULL);
      if (!m_add(pD, V1, V2))
        bombElegant("m_add(pD, V1, V2);", NULL);

      /* if face==2, X/Y axis are the X/Y-axis unit vectors in the x'y'z' coordinate system
         while X/YaxiSxyz are in the xyz coordinate system
      */
      coord[ip][0] = m_hproduct(rD, Xaxis) - m_hproduct(tmp, XaxiSxyz);
      coord[ip][2] = m_hproduct(rD, Yaxis) - m_hproduct(tmp, YaxiSxyz);
      coord[ip][4] = tD5;
      qx = m_hproduct(pD, Xaxis);
      qy = m_hproduct(pD, Yaxis);
      factor = 1 / sqrt(sqr(1 + coord[ip][5]) - sqr(qx) - sqr(qy));
      coord[ip][1] = qx * factor;
      coord[ip][3] = qy * factor;
    }
    if (face==1 && tilt)
      rotateBeamCoordinatesForMisalignment(coord, np, tilt);
    
  }
}
