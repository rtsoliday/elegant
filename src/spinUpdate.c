#include "mdb.h"
#include "track.h"

/* Update spin vector S[3] in the local Frenet-Serret frame.
 *
 * Inputs:
 *   S[3]      Spin vector in local frame, updated in place.
 *   Bx,By,Bz  Magnetic field components in local frame [Tesla].
 *             Here Bz is the longitudinal component along the local s-axis.
 *   x,y       Transverse coordinates [m].
 *   xp,yp     Slopes dx/ds and dy/ds relative to reference path.
 *   ds        Step in reference-path coordinate s [m]. May be negative.
 *   h         Signed reference curvature = 1/rho [1/m].
 *             Convention for h>0 is clockwise rotation when viewed from y>0.
 *   gamma     Lorentz factor of the particle.
 *   beta      v/c of the particle.
 *   charge    Particle charge [C].
 *   mass      Particle mass [kg].
 *   a         Anomalous magnetic moment = (g-2)/2.
 *
 * Notes:
 *   1. Magnetic-field-only Thomas-BMT is used.
 *   2. Since the independent variable is reference s, not true path length l,
 *      the BMT angular velocity is multiplied by dl/ds.
 *   3. The particle direction in the local frame is taken as proportional to
 *
 *        (xp, yp, 1 + h*x)
 *
 *      which is appropriate for curvilinear Frenet-Serret coordinates.
 *   4. The local frame rotation is taken as:
 *
 *        Omega_frame = (0, h, 0)
 *
 *      in units of rad/m, using the common convention for a planar bend with
 *      local y vertical. Depending on your sign convention for h and By, you
 *      may need to flip the sign of this term. See note below.
 *
 * Sign-check:
 *   For an ideal planar reference electron in vertical field By and initial
 *   spin longitudinal, the relative spin precession should be proportional
 *   to a*gamma*h, not (1+a*gamma)h. If the sign is wrong, flip h in the
 *   frame-rotation subtraction.
 *
 * Written by ChatGPT.
 */

void updateSpinQuaternionLocalFS(
    double S[3],
    double Bx, double By, double Bz,
    double x, double y,
    double xp, double yp,
    double ds,
    double h,
    double p,
    double charge,
    double mass,
    double a)
{
    const double tiny = 1e-300;
    double beta, gamma;

    double B[3], betav[3], OmegaBMT[3], OmegaEff[3];

    gamma = sqrt(p*p+1);
    beta = p/gamma;

    B[0] = Bx;
    B[1] = By;
    B[2] = Bz;
#ifdef DEBUG
    printf("Spin update: B=(%le, %le, %le), gamma = %le, beta = %le, ds = %le\n", B[0], B[1], B[2], gamma, beta, ds);
    printf("             S=(%le, %le, %le)\n", S[0], S[1], S[2]);
    fflush(stdout);
#endif
    
    /* Geometry factor: dr/ds in local FS coordinates is proportional to
     *   (xp, yp, 1 + h*x)
     * so dl/ds = sqrt((1+h*x)^2 + xp^2 + yp^2)
     */
    double one_plus_hx = 1.0 + h * x;
    double dl_ds = sqrt(one_plus_hx*one_plus_hx + xp*xp + yp*yp);

    if (dl_ds < tiny)
        return;

    /* Velocity vector / c */
    betav[0] = xp / dl_ds*beta;
    betav[1] = yp / dl_ds*beta;
    betav[2] = one_plus_hx / dl_ds*beta;

    /* Magnetic-field-only Thomas-BMT in l-form:
     *
     *   dS/dl = Omega_l x S
     *
     *   Omega_l = -(q/(m c |beta|)) [ (a + 1/gamma) B
     *                              - a*gamma/(gamma+1) (beta·B) beta ]
     *
     * Convert to s-form with dS/ds = (dl/ds) * Omega_l x S
     */
    double beta_dot_B =
        betav[0]*B[0] + betav[1]*B[1] + betav[2]*B[2];

    double coeff = -(charge / (mass * c_mks * beta)) * dl_ds;
    double kpar  = a * gamma / (gamma + 1.0);

    for (int i = 0; i < 3; i++) {
        OmegaBMT[i] = coeff * (
            (a + 1.0/gamma) * B[i] - kpar * beta_dot_B * betav[i]
        );
    }

    /* Subtract local frame rotation.
     * Common planar-FS choice: Omega_frame = (0, -h, 0)
     */
    OmegaEff[0] = OmegaBMT[0];
    OmegaEff[1] = OmegaBMT[1] + h;
    OmegaEff[2] = OmegaBMT[2];

    /* Rotation vector over this step */
    double rx = OmegaEff[0] * ds;
    double ry = OmegaEff[1] * ds;
    double rz = OmegaEff[2] * ds;

    performQuaternionRotation(S,rx,ry,rz);
}

void performQuaternionRotation(double S[3], double rx, double ry, double rz)
{
    const double tiny = 1e-300;
    double theta = sqrt(rx*rx + ry*ry + rz*rz);
    if (theta < tiny)
        return;
    
    /* Quaternion q = [qw, qx, qy, qz] */
    double half = 0.5 * theta;
    double s = sin(half) / theta;
    double qw = cos(half);
    double qx = rx * s;
    double qy = ry * s;
    double qz = rz * s;

    /* Rotate S using q S q^{-1}, vector form */
    double vx = S[0], vy = S[1], vz = S[2];

    double tx = 2.0 * (qy*vz - qz*vy);
    double ty = 2.0 * (qz*vx - qx*vz);
    double tz = 2.0 * (qx*vy - qy*vx);

    S[0] = vx + qw*tx + (qy*tz - qz*ty);
    S[1] = vy + qw*ty + (qz*tx - qx*tz);
    S[2] = vz + qw*tz + (qx*ty - qy*tx);

    /* Renormalize to suppress roundoff drift */
    double snorm = sqrt(S[0]*S[0] + S[1]*S[1] + S[2]*S[2]);
    if (snorm > tiny) {
        S[0] /= snorm;
        S[1] /= snorm;
        S[2] /= snorm;
    }
#ifdef DEBUG
    printf("             S=(%le, %le, %le)\n", S[0], S[1], S[2]);
    fflush(stdout);
#endif
}

void rotateSpinCoordinateSystem(double *S, double cos_t, double sin_t)
{
  double S0[2];
  memcpy(S0, S, sizeof(*S)*2);
  S[0] = cos_t*S0[0] + sin_t*S0[1];
  S[1] = -sin_t*S0[0] + cos_t*S0[1];
}

void rotateSpinsCoordinateSystem(double **particle, long np, double t)
{
  double cos_t, sin_t;
  double S0[2];
  long i;
  if (!spinCoordOffset || !t)
    return;
  cos_t = cos(t);
  sin_t = sin(t);
  for (i=0; i<np; i++) {
    memcpy(S0, particle[i]+spinCoordOffset, sizeof(double)*2);
    particle[i][spinCoordOffset+0] =  cos_t*S0[0] + sin_t*S0[1];
    particle[i][spinCoordOffset+1] = -sin_t*S0[0] + cos_t*S0[1];
  }
}

void updateSpinForSolenoid(double **coord, long np, double Po, SOLE *sole)
{
  double Bz;
  long i;
  if (sole->B)
    Bz = sole->B;
  else
    Bz = -sole->ks*Po*particleMass*c_mks/particleCharge*particleRelSign;
  if (Bz==0)
    return;
  for (i=0; i<np; i++)
    updateSpinQuaternionLocalFS(coord[i]+spinCoordOffset, 0.0, 0.0, Bz,
				coord[i][0], coord[i][2], coord[i][1], coord[i][3],
				sole->length, 0.0, Po*(1+coord[i][5]), 
				-particleCharge*particleRelSign,
				particleMass, particleAnomalousMagneticMoment);
}
