/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/*
 $Log: not supported by cvs2svn $
 Revision 1.1  2006/03/15 17:10:22  shang
 first version, contains functions for physics apllications

*/

/* define structure for use with scanargs(), scanlist() */
#if !defined(PHYSICS_INCLUDED)
#define PHYSICS_INCLUDED 1

#include <assert.h>
#include "SDDS.h"
#include "mdb.h"
#undef epicsShareFuncOAGPHY

#if (defined(_WIN32) && defined(_MSC_VER) && defined(EXPORT_OAGPHYLIB))
#define epicsShareFuncOAGPHY  __declspec(dllexport)
#else
#define epicsShareFuncOAGPHY
#endif

#define TWISS_DATA_TYPES 16
typedef struct {
  long beams; /* number of eletron beams = number of twiss input file pages.*/
  double **data; /* contains all the data for following types */
  double *betax, *alphax, *etax, *etaxp; 
  double *betay, *alphay, *etay, *etayp;
  double *ex0, *ey0, *Sdelta0, *pCentral;
  double *sigmax, *sigmaxp, *sigmay, *sigmayp;
} TWISS_PARAMETER;

epicsShareFuncOAGPHY extern long ReadTwissInput(char *inputfile,
                                                TWISS_PARAMETER *twiss,
                                                double coupling,
                                                double emitRatio,
                                                double period,
                                                long Nu);

epicsShareFuncOAGPHY extern long GetTwissValues(SDDS_DATASET *SDDSin,
                                                double *betax, double *alphax,
                                                double *etax, double *etaxp, 
                                                double *betay, double *alphay,
                                                double *etay, double *etayp, 
                                                double *ex0, double *ey0, 
                                                double *Sdelta0, double *pCentral, 
                                                double emitRatio, double coupling);
/* fwhm is optional: pass NULL to skip FWHM computation. */
epicsShareFuncOAGPHY extern void FindPeak(double *E,double *spec,double *ep,double *sp,double *fwhm,long n);
epicsShareFuncOAGPHY extern void ComputeBeamSize(double period, long Nu, double ex, 
                                                 double ey, double Sdelta0, 
                                                 double betax, double alphax, 
                                                 double etax, double etaxp,
                                                 double betay, double alphay, 
                                                 double etay, double etayp,
                                                 double *Sx, double *Sy, double *Sxp, double *Syp);

/* nstart and nend are optional output pointers: pass NULL to skip.
 * Returns 0 on success, 1 on failure. */
epicsShareFuncOAGPHY extern int Gauss_Convolve(double *E,double *spec,long *ns,double sigmaE, long *nstart, long *nend);

	/*****  Computes the undulator brightness produced by a Gaussian electron beam  ***/
epicsShareFuncOAGPHY extern double computeBrightnessLindberg(double radLambda, int radHarm, double radDet,
			 double undLength, int undN, double undK,
			 double emitx, double emity, double betax, double betay,
			 double alphax, double alphay, double sigmaDelta, double current);

	/*****  Computes the undulator flux in the central cone near an odd harmonic  ***/
epicsShareFuncOAGPHY extern double computeFluxLindberg(int radHarm, int undN, double undK,
			   double radDet, double sigmaDelta, double current);

epicsShareFuncOAGPHY extern void computeEffectiveBeamParameters(double *exEff, double *betaxEff, double *alphaxEff,
								double ex, double betax, double alphax,
								double etax, double etaxp, double Sdelta);

/* results of the coupled (eigen-)emittance analysis of the full 6x6 beam matrix;
   mode index 0 is x-dominated, 1 is y-dominated, 2 is longitudinal-dominated
   (see ComputeCoupledParameters).  The longitudinal coordinate is s=beta*c*t (m),
   matching elegant's moments_output, so emit[2] is a geometric emittance in m. */
typedef struct {
  double emit[3];                          /* eigen-emittances e1, e2, e3 (geometric, m) */
  double betax[3], alphax[3], gammax[3];   /* per-mode Twiss functions, x plane */
  double betay[3], alphay[3], gammay[3];   /* per-mode Twiss functions, y plane */
  double betaz[3], alphaz[3], gammaz[3];   /* per-mode Twiss functions, longitudinal (s,delta) plane */
  double A_xy[3], A_xpy[3], A_xyp[3], A_xpyp[3]; /* per-mode x-y cross-plane coupling terms */
} COUPLED_RESULTS;

/* Coupled (eigen-)emittance and coupled lattice-function analysis of the 6x6 beam
   matrix S via the Sigma.J (Wolski) method.  sLongScale=beta*c rescales row/col 4
   from t (s) to s=beta*c*t (m); pass 1 if S is already in s.  Returns 1 on success
   (result filled), 0 on failure.  Requires compilation with GSL (-DUSE_GSL). */
epicsShareFuncOAGPHY extern long ComputeCoupledParameters(COUPLED_RESULTS *result, double S[6][6], double sLongScale);
#endif
