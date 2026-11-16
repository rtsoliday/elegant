/*
 *----------------------------------------------------------------------------
 * Modification Log:
 * -----------------
 * .03  2004-11-01      M. Borland, borland@aps.anl.gov
 *                      Moved function prototypes from gwig.c
 *
 * .02  2003-04-29      YK Wu, Duke University, wu@fel.duke.edu
 *                      use scientific notation for constants
 *
 * .01  2003-04-28      YK Wu, Duke University, wu@fel.duke.edu
 *                      header file for gwig.c
 *----------------------------------------------------------------------------
 *  Accelerator Physics Group, Duke FEL Lab, www.fel.duke.edu
*/

#include "SDDS.h"
#include "mdb.h"
#include "track.h"

#ifndef GWIG
#define GWIG

#define WHmax 1000
#define GWIG_EPS (1e-6)

static const int Elem_Entrance = 1;
static const int Elem_Exit =-1;

static const double q_e    = 1.602176462e-19; /* electron charge, [C] */
static const double m_e    = 9.10938188e-31;  /* electron mass, [kg] */
static const double clight = 2.99792458e8;    /* speed of light [m/s] */
static const double r_e    = 2.817940285e-15; /* electron classic radius,[m]*/
static const double XMC2   = 0.510998902e-03; /* mc^2 in GeV */

struct gwig {
  int Pmethod;      /* Integration Method */
  int PN;           /* Number of integration steps */
  double E0;        /* Energy of ring, [GeV] */
  double Po;        /* beta*gamma for reference particle */
  double PB0;       /* B0 in [Tesla] */
  double PB0H, PB0V; /* Ignored if B0 is nonzero, but used otherwise for H and V wiggler fields */
  int Nw;           /* Number of periods */
  double Lw;        /* Wiggler Period [m] */
  int NHharm;       /* No. of horizontal harmonics */
  int NVharm;       /* No. of vertical harmonics */
  double Aw;        /* Wiggler parameter */
  double Zw;        /* Longitudinal variable [m] */
  double zStartH, zStartV;  /* Start and end z coordinates of the wiggler field, which are computed */
  double zEndH, zEndV;      /* based on the phase of the first harmonic to get matched dispersion. */
  short sr, isr;    /* flags for classical and incoherent SR */
  double srCoef, isrCoef;
  double z1, z2, z3; /* cut-over positions for entrance poles */
  double z4, z5, z6; /* cut-over positions for exit poles */
  double BConstant[2]; /* constant fields (for correction) */

  int HSplitPole, VSplitPole;
  double HCw[WHmax];
  double VCw[WHmax];
  double HCw_raw[WHmax];
  double VCw_raw[WHmax];
  double Hkx[WHmax];
  double Hky[WHmax];
  double Hkz[WHmax];
  double Htz[WHmax];
  double Vkx[WHmax];
  double Vky[WHmax];
  double Vkz[WHmax];
  double Vtz[WHmax];
  CWIGGLER *cwiggler;
};

double GWigPoleFactor(struct gwig *pWig, double Z);
void GWigPass_2nd(struct gwig *pWig, double *X, double *sigmaDelta2, long singleStep);
void GWigPass_4th(struct gwig *pWig, double *X, double *sigmaDelta2, long singleStep);
long GWigCountRadiationKicks(struct gwig *pWig);
void GWigAddToFieldOutput(CWIGGLER *cwiggler, double Z, double *X, double *B, double poleFactor);
void GWigMap_2nd(struct gwig *pWig, double *X, double dl, double poleFactor);
void GWigAx(struct gwig *pWig, double *Xvec, double *pax, double *paxpy, double poleFactor);
void GWigAy(struct gwig *pWig, double *Xvec, double *pay, double *paypx, double poleFactor);
double sinc(double x );
void InitializeCWiggler(CWIGGLER *cwiggler, char *name);
long ReadCWigglerHarmonics(double **BData, long *harmonics, char *file, char *name, long vertical,
                           long splitPole, CWIGGLER *cwiggler);
#endif
