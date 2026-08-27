/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

#include "mdb.h"
#include "track.h"

int FindLineCircleIntersections1(double *x, double *y,
                                 double x0, double y0, double theta,
                                 double xc, double yc, double r);
#define DEBUG 0

long track_through_tubend(double **part, int64_t n_part, TUBEND *tubend,
                          double p_error, double Po, double **accepted,
                          double z_start) {
#if DEBUG
  static FILE *fp = NULL;
  double x0, X0, X1, X2, X3, X4;
  double xp0, Y0, Y1, Y2, Y3, Y4;
#endif
  int64_t ip, i_top;
  long particleLost, solutions;
  double rhoRefTraj, thetaRefTraj;
  double rhoMagnet, thetaMagnet, w2, dY;
  double distanceToEdge;
  double XInitial, YInitial, phiInitial, YRI, CEY;
  double Xint[2], Yint[2], XInt = 0.0, YInt = 0.0;
  double XMagnetEnd, YPoleCenter;
  /* double YRefFinal; */
  double phiExit = 0.0, XFinal, YFinal;
  double *coord;
  double rho_particle, CPX, CPY;

#if DEBUG
  if (!fp) {
    fp = fopen("tubend.debug", "w");
    fprintf(fp, "SDDS1\n&column name=x0, type=double &end\n&column name=xp0, type=double &end\n");
    fprintf(fp, "&column name=X0, type=double &end\n&column name=Y0, type=double &end\n");
    fprintf(fp, "&column name=X1, type=double &end\n&column name=Y1, type=double &end\n");
    fprintf(fp, "&column name=X2, type=double &end\n&column name=Y2, type=double &end\n");
    fprintf(fp, "&column name=X3, type=double &end\n&column name=Y3, type=double &end\n");
    fprintf(fp, "&column name=X4, type=double &end\n&column name=Y4, type=double &end\n");
    fprintf(fp, "&column name=Lost type=long &end\n");
    fprintf(fp, "&data mode=ascii &end\n");
  }
  fprintf(fp, "%ld\n", n_part);
#endif

  rhoRefTraj = tubend->length / tubend->angle;
  thetaRefTraj = tubend->angle;
  thetaMagnet = tubend->magnet_angle;
  w2 = tubend->magnet_width / 2;
  dY = tubend->offset;

  /* A coordinate system is used that places the center of the arc of the reference
     trajectory at the origin.
     */

  /* Ends of the magnet are at X = +/- this value */
  XMagnetEnd = rhoRefTraj * sin(thetaRefTraj / 2);
  /* Radius of the magnet itself */
  rhoMagnet = XMagnetEnd / sin(thetaMagnet / 2);
  /* Coordinate of the center of the circle defining the outer edge of the dipole */
  CEY = rhoRefTraj * cos(thetaRefTraj / 2) - dY + w2 - rhoMagnet * cos(thetaMagnet / 2);
  /* Center of the pole at either end of the magnet */
  YPoleCenter = (rhoMagnet * cos(thetaMagnet / 2) + CEY - w2);
  /* The Y coordinate of the reference trajectory at the exit (and entrance) */
  /* YRefFinal = rhoRefTraj*cos(thetaRefTraj/2); */

  i_top = n_part - 1;
  for (ip = 0; ip <= i_top; ip++) {
    coord = part[ip];
#if DEBUG
    X0 = X1 = X2 = X3 = X4 = 0;
    Y0 = Y1 = Y2 = Y3 = Y4 = 0;
    x0 = coord[0];
    xp0 = coord[1];
#endif

    /* global coordinates of particle at the reference plane */
    XInitial = (rhoRefTraj + coord[0]) * sin(thetaRefTraj / 2);
    YInitial = (rhoRefTraj + coord[0]) * cos(thetaRefTraj / 2);
#if DEBUG
    X0 = XInitial;
    Y0 = YInitial;
#endif

    /* angle of particle trajectory.
     * slope in global coordinates is tan(phiInitial).
     */
    phiInitial = -thetaRefTraj / 2 - atan(coord[1]);

    /* intersection of particle trajectory with rectangular edge of entrance */
    YRI = YInitial + (XMagnetEnd - XInitial) * tan(phiInitial);
#if DEBUG
    X1 = XMagnetEnd;
    Y1 = YRI;
#endif
    distanceToEdge = YRI - YPoleCenter;
    particleLost = 0;
    if (distanceToEdge < -w2)
      particleLost = 1;
    else if (tubend->fse <= -1) {
      tubend->fse = -1; /* in case it is <-1 */
      /* final slope */
      phiExit = phiInitial;
      /* Copy initial coordinates for use in computing coordinates at
       * the exit reference plane 
       */
      XInt = XInitial;
      YInt = YInitial;
    } else {
      if (fabs(distanceToEdge) < w2) {
        /* Go through the magnet starting at the end plane.
         * Record the intersection with the end plane for use below in
         * computing the circle of the particle motion.
         */
        XInt = XMagnetEnd;
        YInt = YRI;
#if DEBUG
        X2 = XInt;
        Y2 = YInt;
#endif
      } else {
        /* go through the magnet starting at the upper edge */
        /* find intersection with the upper edge */
        if ((solutions = FindLineCircleIntersections1(Xint, Yint, XInitial, YInitial, phiInitial, 0, CEY, rhoMagnet)) < 2) {
          /* less than two solutions means we didn't enter the magnet */
          particleLost = 1;
#if DEBUG
          printf("Circle intersection not found: %ld solutions\n", solutions);
          fflush(stdout);
#endif
        } else {
          /* choose the solution with largest Xint */
          if (Xint[0] < Xint[1]) {
            XInt = Xint[1];
            YInt = Yint[1];
          } else {
            XInt = Xint[0];
            YInt = Yint[0];
          }
#if DEBUG
          X2 = XInt;
          Y2 = YInt;
#endif
        }
      }

      if (!particleLost) {
        /* At this point, (XInt, YInt) is a point on the circle. 
         * The starting slope is tan(phiInitial).
         */

        /* find parameters of circle that particle travels on */
        rho_particle = rhoRefTraj * (1 + coord[5]) / (1 + tubend->fse);
        CPX = XInt + rho_particle * sin(phiInitial);
        CPY = YInt - rho_particle * cos(phiInitial);

        /* find intersection of that circle with the line at the exit edge */
        if (FindLineCircleIntersections1(Xint, Yint, -XMagnetEnd, 0, PIo2,
                                         CPX, CPY, rho_particle) < 2) {
          /* Less than two solutions means we don't exit the magnet */
          particleLost = 1;
        } else {
          /* take the solution with the largest Y */
          if (Yint[0] > Yint[1]) {
            XInt = Xint[0];
            YInt = Yint[0];
          } else {
            XInt = Xint[1];
            YInt = Yint[1];
          }
#if DEBUG
          X3 = XInt;
          Y3 = YInt;
#endif
          distanceToEdge = YInt - YPoleCenter;
          if (fabs(distanceToEdge) > w2) {
            /* Particle did not exit through the downstream pole face */
            particleLost = 1;
          } else {
            /* Compute final slope.  Used with XInt and YInt below to
               * get coordinates at exit reference plane. */
            phiExit = -atan((XInt - CPX) / (YInt - CPY));
          }
        }
      }
    }

    if (!particleLost) {
      /* At this point, (XInt, YInt) is the intersection of the particle
       * trajectory with the end plane.  The slope of the trajectory at
       * that point is tan(phiExit).
       */

      /* Find intersection of line of trajectory from end of magnet with the
       * exit reference plane, X=-Y*tan(thetaRefTraj/2)
       */
      YFinal = (YInt - XInt * tan(phiExit)) / (1 + tan(thetaRefTraj / 2) * tan(phiExit));
      XFinal = -YFinal * tan(thetaRefTraj / 2);
#if DEBUG
      X4 = XFinal;
      Y4 = YFinal;
#endif

      /* transform to curvilinear coordinates */
      coord[0] = sqrt(sqr(XFinal) + sqr(YFinal)) - rhoRefTraj;
      coord[1] = tan(thetaRefTraj / 2 - phiExit);
      /* this could be done right but for top-up tracking we don't care: */
      coord[2] = tubend->length * coord[3];
      coord[4] += tubend->length;
    } else {
      /* lost particle */
      if (!part[i_top]) {
        printf(
          "error: couldn't swap particles %ld and %ld--latter is null pointer (track_through_tubend)\n",
          ip, i_top);
        fflush(stdout);
        abort();
      }
      swapParticles(part[ip], part[i_top]);
      if (accepted) {
        if (!accepted[i_top]) {
          printf(
            "error: couldn't swap acceptance data for particles %ld and %ld--latter is null pointer (track_through_tubend)\n",
            ip, i_top);
          fflush(stdout);
          abort();
        }
        swapParticles(accepted[ip], accepted[i_top]);
      }
      part[i_top][4] = z_start;
      part[i_top][5] = Po * (1 + part[i_top][5]);
      i_top--;
      ip--;
    }
#if DEBUG
    fprintf(fp, "%le %le ", x0, xp0);
    fprintf(fp, "%le %le ", X0, Y0);
    fprintf(fp, "%le %le ", X1, Y1);
    fprintf(fp, "%le %le ", X2, Y2);
    fprintf(fp, "%le %le ", X3, Y3);
    fprintf(fp, "%le %le ", X4, Y4);
    fprintf(fp, "%ld\n", particleLost);
#endif
  }
#if DEBUG
  fflush(fp);
#endif
  return i_top + 1;
}
