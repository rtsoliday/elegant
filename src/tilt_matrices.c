/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* routine: tilt_matrices()
 * purpose: rotate matrices through a given angle 
 *
 * Michael Borland, 1989
 */
#include "mdb.h"
#include "track.h"

void tilt_matrices(VMATRIX *M, double tilt) {
  static VMATRIX Rot, IRot;
  static long initialized = 0;
  VMATRIX Mr;
  double sin_tilt, cos_tilt;

  log_entry("tilt_matrices");

  if (tilt == 0) {
    log_exit("tilt_matrices");
    return;
  }

  if (!initialized) {
    initialized = 1;
    initialize_matrices(&Rot, 1);
    initialize_matrices(&IRot, 1);
  }

  initialize_matrices(&Mr, M->order);

  if (fabs(tilt - PI / 2) < 1e-12) {
    sin_tilt = 1;
    cos_tilt = 0;
  } else if (fabs(tilt + PI / 2) < 1e-12) {
    sin_tilt = -1;
    cos_tilt = 0;
  } else if (fabs(tilt - PI) < 1e-12 || fabs(tilt + PI) < 1e-12) {
    sin_tilt = 0;
    cos_tilt = -1;
  } else {
    sin_tilt = sin(tilt);
    cos_tilt = cos(tilt);
  }

  /* Rotation matrix for (x, y) */
  IRot.R[0][0] = Rot.R[0][0] = cos_tilt;
  IRot.R[0][2] = -(Rot.R[0][2] = sin_tilt);
  IRot.R[2][0] = -(Rot.R[2][0] = -sin_tilt);
  IRot.R[2][2] = Rot.R[2][2] = cos_tilt;

  /* Rotation matrix for (x', y') */
  IRot.R[1][1] = Rot.R[1][1] = cos_tilt;
  IRot.R[1][3] = -(Rot.R[1][3] = sin_tilt);
  IRot.R[3][1] = -(Rot.R[3][1] = -sin_tilt);
  IRot.R[3][3] = Rot.R[3][3] = cos_tilt;

  IRot.R[4][4] = IRot.R[5][5] = Rot.R[4][4] = Rot.R[5][5] = 1;

  concat_matrices(&Mr, M, &Rot, 0);
  concat_matrices(M, &IRot, &Mr, 0);

  free_matrices(&Mr);

  log_exit("tilt_matrices");
}

/* routine: rotation_matrix()
 * purpose: return R matrix for rotation of beamline about given angle
 *
 * Michael Borland, 1989
 */

VMATRIX *rotation_matrix(double tilt) {
  VMATRIX *Rot;
  double sin_tilt, cos_tilt;

  log_entry("rotation_matrix");

  if (fabs(tilt - PI / 2) < 1e-12) {
    sin_tilt = 1;
    cos_tilt = 0;
  } else if (fabs(tilt - PI) < 1e-12) {
    sin_tilt = 0;
    cos_tilt = -1;
  } else {
    sin_tilt = sin(tilt);
    cos_tilt = cos(tilt);
  }

  Rot = tmalloc(sizeof(*Rot));
  initialize_matrices(Rot, 1);

  /* Rotation matrix for (x, y) */
  Rot->R[0][0] = cos_tilt;
  Rot->R[0][2] = sin_tilt;
  Rot->R[2][0] = -sin_tilt;
  Rot->R[2][2] = cos_tilt;

  /* Rotation matrix for (x', y') */
  Rot->R[1][1] = cos_tilt;
  Rot->R[1][3] = sin_tilt;
  Rot->R[3][1] = -sin_tilt;
  Rot->R[3][3] = cos_tilt;

  Rot->R[4][4] = Rot->R[5][5] = 1;

  log_exit("rotation_matrix");
  return (Rot);
}

/* The name is a misnomer: we rotate the coordinates of a particle
 * to emulate rotation of the upcoming element by the given angle
 */

void rotateCoordinatesForMisalignment(double *coord, double angle) {
  static double x, xp, y, yp;
  static double sin_a, cos_a;

  if (!angle || fabs(fabs(angle) - PIx2) < 1e-12)
    return;
  if (fabs(fabs(angle) - PI) < 1e-12) {
    cos_a = -1;
    sin_a = 0;
  } else if (fabs(angle - PIo2) < 1e-12) {
    cos_a = 0;
    sin_a = 1;
  } else if (fabs(angle + PIo2) < 1e-12) {
    cos_a = 0;
    sin_a = -1;
  } else {
    cos_a = cos(angle);
    sin_a = sin(angle);
  }
  x = coord[0];
  xp = coord[1];
  y = coord[2];
  yp = coord[3];
  coord[0] = x * cos_a + y * sin_a;
  coord[2] = -x * sin_a + y * cos_a;
  coord[1] = xp * cos_a + yp * sin_a;
  coord[3] = -xp * sin_a + yp * cos_a;
  if (spinCoordOffset) {
    double Sx, Sy;
    Sx = coord[spinCoordOffset+0];
    Sy = coord[spinCoordOffset+1];
    coord[spinCoordOffset+0] =  Sx * cos_a + Sy * sin_a;
    coord[spinCoordOffset+1] = -Sx * sin_a + Sy * cos_a;
  }
}

/* We rotate the coordinates of a beam to implement the rotation of the upcoming element by the given angle.
 * Hence, the particles are rotated by the negative of the angle.
 */

void rotateBeamCoordinatesForMisalignment(double **part, long np, double angle) {
  double x, xp, y, yp, *coord;
  double sin_a, cos_a;
  long i;

  if (!angle || fabs(fabs(angle) - PIx2) < 1e-12)
    return;
  if (fabs(fabs(angle) - PI) < 1e-12) {
    cos_a = -1;
    sin_a = 0;
  } else if (fabs(angle - PIo2) < 1e-12) {
    cos_a = 0;
    sin_a = 1;
  } else if (fabs(angle + PIo2) < 1e-12) {
    cos_a = 0;
    sin_a = -1;
  } else {
    cos_a = cos(angle);
    sin_a = sin(angle);
  }

  for (i = 0; i < np; i++) {
    coord = part[i];
    x = coord[0];
    xp = coord[1];
    y = coord[2];
    yp = coord[3];
    coord[0] = x * cos_a + y * sin_a;
    coord[2] = -x * sin_a + y * cos_a;
    coord[1] = xp * cos_a + yp * sin_a;
    coord[3] = -xp * sin_a + yp * cos_a;
  }
  if (spinCoordOffset) {
    double Sx, Sy;
    for (i = 0; i < np; i++) {
      coord = part[i];
      Sx = coord[spinCoordOffset+0];
      Sy = coord[spinCoordOffset+1];
      coord[spinCoordOffset+0] =  Sx * cos_a + Sy * sin_a;
      coord[spinCoordOffset+1] = -Sx * sin_a + Sy * cos_a;
    }
  }
}

#include "matlib.h"

void setupRotate3Matrix(void **Rv, double roll, double yaw, double pitch) {
  MATRIX *R;
  double angle[3];   /* yaw=phi, pitch=theta, roll=psi */
  double s[3], c[3]; /* sine, cosine of theta, phi, psi */
  long ip;

  /* sequence of rotations is: yaw, pitch, roll 
   * this is the 3-2-1 seqeunce from Appendix B of Goldstein's Classical Mechanics 
   * angles: phi=yaw, theta=pitch, psi=roll
   * coordinates: Z = y, X = z, Y = x,
   * where capital letters are Goldstein's (x,y,z) and small letters are accelerator
   * coordinates.
   */
  angle[0] = yaw;
  angle[1] = pitch;
  angle[2] = roll;
  for (ip = 0; ip < 3; ip++) {
    s[ip] = sin(angle[ip]);
    c[ip] = cos(angle[ip]);
  }
  m_alloc(&R, 3, 3);
#define C_PHI c[0]
#define S_PHI s[0]
#define C_THETA c[1]
#define S_THETA s[1]
#define C_PSI c[2]
#define S_PSI s[2]
  R->a[0][0] = C_THETA * C_PHI;
  R->a[0][1] = C_THETA * S_PHI;
  R->a[0][2] = -S_THETA;
  R->a[1][0] = S_PSI * S_THETA * C_PHI - C_PSI * S_PHI;
  R->a[1][1] = S_PSI * S_THETA * S_PHI + C_PSI * C_PHI;
  R->a[1][2] = C_THETA * S_PSI;
  R->a[2][0] = C_PSI * S_THETA * C_PHI + S_PSI * S_PHI;
  R->a[2][1] = C_PSI * S_THETA * S_PHI - S_PSI * C_PHI;
  R->a[2][2] = C_THETA * C_PSI;
  *Rv = R;
}

void rotate3(double *data, void *Rv) {
  static MATRIX *Q1 = NULL, *Q0 = NULL; /* (z, x, y) */
  MATRIX *R;
  R = (MATRIX *)Rv;

  if (!Q0) {
    m_alloc(&Q0, 3, 1);
    m_alloc(&Q1, 3, 1);
  }

  Q0->a[0][0] = data[2]; /* accelerator z is Goldstein's x */
  Q0->a[1][0] = data[0]; /* accelerator x is Goldstein's y */
  Q0->a[2][0] = data[1]; /* accelerator y is Goldstein's z */
  m_mult(Q1, R, Q0);
  data[2] = Q1->a[0][0];
  data[0] = Q1->a[1][0];
  data[1] = Q1->a[2][0];
}

void yaw_matrices(VMATRIX *M, double yaw)
/* transform element matrix to reflect yawing of element (rotation
 * about vertical axis)
 */
{
  static VMATRIX input, output;
  static long initialized = 0;
  static VMATRIX Mr;

  if (yaw == 0)
    return;

  if (!initialized) {
    long i;
    initialized = 1;
    initialize_matrices(&input, 1);
    initialize_matrices(&output, 1);
    initialize_matrices(&Mr, M->order);
    for (i = 0; i < 6; i++)
      input.R[i][i] = output.R[i][i] = 1;
  }

  input.C[1] = yaw;
  output.C[1] = -yaw;
  concat_matrices(&Mr, M, &input, 0);
  concat_matrices(M, &output, &Mr, 0);
}

void pitch_matrices(VMATRIX *M, double pitch)
/* transform element matrix to reflect pitching of element (rotation
 * about horizontal transverse axis)
 */
{
  static VMATRIX input, output;
  static long initialized = 0;
  static VMATRIX Mr;

  if (pitch == 0)
    return;

  if (!initialized) {
    long i;
    initialized = 1;
    initialize_matrices(&input, 1);
    initialize_matrices(&output, 1);
    initialize_matrices(&Mr, M->order);
    for (i = 0; i < 6; i++)
      input.R[i][i] = output.R[i][i] = 1;
  }

  input.C[3] = pitch;
  output.C[3] = -pitch;

  concat_matrices(&Mr, M, &input, 0);
  concat_matrices(M, &output, &Mr, 0);
}
