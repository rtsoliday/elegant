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
 Revision 1.2  2006/03/15 17:09:07  shang
 added ReadTwissInput function

 Revision 1.1  2006/01/09 22:05:22  shang
 first version

 these routines are used in computing undulator brightness 
*/

#include "oagphy.h"

long ReadTwissInput(char *inputfile, TWISS_PARAMETER *twiss, double coupling, double emitRatio,
                    double period, long Nu)
{
  SDDS_DATASET SDDSin;
  long readCode;
  double **data=NULL;
  int i;
  char *Units=NULL;
  
  enum TWISS_DATA {betax=0, alphax, etax, etaxp, 
                     betay, alphay, etay, etayp,
                     ex0, ey0, Sdelta0, pCentral,
                     sigmax, sigmaxp, sigmay, sigmayp};
  
  if (!SDDS_InitializeInput(&SDDSin, inputfile)) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
    return 0;
  }
  /* check the input file for valid data */
  if (SDDS_CheckColumn(&SDDSin, "betax", "m", SDDS_ANY_FLOATING_TYPE, stderr)!=SDDS_CHECK_OK || 
      SDDS_CheckColumn(&SDDSin, "alphax", "", SDDS_ANY_FLOATING_TYPE, stderr)!=SDDS_CHECK_OK || 
      SDDS_CheckColumn(&SDDSin, "etax", "m", SDDS_ANY_FLOATING_TYPE, stderr)!=SDDS_CHECK_OK ||
      SDDS_CheckColumn(&SDDSin, "etaxp", "", SDDS_ANY_FLOATING_TYPE, stderr)!=SDDS_CHECK_OK ||
      SDDS_CheckColumn(&SDDSin, "betay", "m", SDDS_ANY_FLOATING_TYPE, stderr)!=SDDS_CHECK_OK || 
      SDDS_CheckColumn(&SDDSin, "alphay", "", SDDS_ANY_FLOATING_TYPE, stderr)!=SDDS_CHECK_OK ||
      SDDS_CheckColumn(&SDDSin, "etay", "m", SDDS_ANY_FLOATING_TYPE, stderr)!=SDDS_CHECK_OK ||
      SDDS_CheckColumn(&SDDSin, "etayp", "", SDDS_ANY_FLOATING_TYPE, stderr)!=SDDS_CHECK_OK ) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
    fprintf(stderr,"something wrong with twiss parameter columns.\n");
    return 0;
  }
  if (SDDS_CheckParameter(&SDDSin, "ex0", "$gp$rm", SDDS_ANY_FLOATING_TYPE, NULL)!=SDDS_CHECK_OK) {
    if (SDDS_CheckColumn(&SDDSin,"ex", "$gp$rm", SDDS_ANY_FLOATING_TYPE, NULL)!=SDDS_CHECK_OK &&
        SDDS_CheckColumn(&SDDSin,"ex", "m", SDDS_ANY_FLOATING_TYPE, NULL)!=SDDS_CHECK_OK) {
      fprintf(stderr,"Something wrong with both ex0 parameter and ex column, one of them has to exist.\n");
      return 0;
    }
  }
  if (SDDS_CheckParameter(&SDDSin, "pCentral", NULL, SDDS_ANY_FLOATING_TYPE, NULL)==SDDS_CHECK_OK) {
    if (!SDDS_GetParameterInformation(&SDDSin, "units", &Units, SDDS_GET_BY_NAME, "pCentral")) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
      return 0;
    }
    if (Units && !is_blank(Units) && strcmp(Units, "m$be$nc")) {
      fprintf(stderr,"Invalid units of pCentral parameter.\n");
      return 0;
    }
  } else if (SDDS_CheckColumn(&SDDSin, "pCentral", NULL, SDDS_ANY_FLOATING_TYPE, NULL)==SDDS_CHECK_OK) {
    if (!SDDS_GetColumnInformation(&SDDSin, "units", &Units, SDDS_GET_BY_NAME, "pCentral")) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
      return 0;
    }
    if (Units && !is_blank(Units) && strcmp(Units, "m$be$nc")) {
      fprintf(stderr, "Invalid units of pCentral column.\n");
      return 0;
    }
  } else {
    fprintf(stderr, "Something wrong with both pCentral parameter and pCentral column, one of them has to exist.\n");
    return 0;
  }
  if (SDDS_CheckParameter(&SDDSin, "Sdelta0", "", SDDS_ANY_FLOATING_TYPE, NULL)!=SDDS_CHECK_OK &&
      SDDS_CheckColumn(&SDDSin, "Sdelta", "", SDDS_ANY_FLOATING_TYPE, NULL)!=SDDS_CHECK_OK) {
    fprintf(stderr, "Something wrong with both Sdelta0 parameter and Sdelta column, one of them has to exist.\n");
    return 0;
  }
  twiss->beams = 0;
  data = malloc(sizeof(*data)*TWISS_DATA_TYPES);
  for (i=0; i<TWISS_DATA_TYPES; i++)
    data[i] = NULL;
  
  while ((readCode=SDDS_ReadPage(&SDDSin))>0) {
    for (i=0; i<TWISS_DATA_TYPES; i++)
      data[i] = SDDS_Realloc(data[i], sizeof(**data) * (twiss->beams + 1));
    if (!(GetTwissValues(&SDDSin, &data[0][twiss->beams], &data[1][twiss->beams], 
                         &data[2][twiss->beams], &data[3][twiss->beams],
                         &data[4][twiss->beams], &data[5][twiss->beams], 
                         &data[6][twiss->beams], &data[7][twiss->beams],
                         &data[8][twiss->beams], &data[9][twiss->beams],
                         &data[10][twiss->beams], &data[11][twiss->beams],
                         emitRatio, coupling))) {
      fprintf(stderr, "Problem in getting twiss values.\n");
      return 0;
    }
    ComputeBeamSize(period, Nu, data[ex0][twiss->beams], data[ey0][twiss->beams], 
                    data[Sdelta0][twiss->beams],
                    data[betax][twiss->beams], data[alphax][twiss->beams], 
                    data[etax][twiss->beams], data[etaxp][twiss->beams],
                    data[betay][twiss->beams], data[alphay][twiss->beams],
                    data[etay][twiss->beams], data[etayp][twiss->beams],
                    data[sigmax]+twiss->beams, data[sigmay]+twiss->beams,
                    data[sigmaxp] + twiss->beams, data[sigmayp]+ twiss->beams);
    twiss->beams ++;   
  }
  if (!SDDS_Terminate(&SDDSin)) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
    return 0;
  }
  
  twiss->data = data;
  twiss->betax = data[betax];
  twiss->alphax = data[alphax];
  twiss->etax = data[etax];
  twiss->etaxp = data[etaxp];
  twiss->betay = data[betay];
  twiss->alphay = data[alphay];
  twiss->etay = data[etay];
  twiss->etayp = data[etayp];
  twiss->ex0 = data[ex0];
  twiss->ey0 = data[ey0];
  twiss->Sdelta0 = data[Sdelta0];
  twiss->pCentral = data[pCentral];
  twiss->sigmax = data[sigmax];
  twiss->sigmaxp = data[sigmaxp];
  twiss->sigmay = data[sigmay];
  twiss->sigmayp = data[sigmayp];
  return 1;
}


long GetTwissValues(SDDS_DATASET *SDDSin,
                    double *betax, double *alphax, double *etax, double *etaxp,
                    double *betay, double *alphay, double *etay, double *etayp,
                    double *ex0, double *ey0, double *Sdelta0, double *pCentral,
                    double emitRatio, double coupling) {
  double *data;
  long rows, ey0Exist = 0;

  if (!(rows = SDDS_RowCount(SDDSin)))
    return 0;

  if (!(data = SDDS_GetColumnInDoubles(SDDSin, "betax")))
    SDDS_Bomb("unable to get betax");
  *betax = data[rows - 1];
  free(data);

  if (!(data = SDDS_GetColumnInDoubles(SDDSin, "alphax")))
    SDDS_Bomb("unable to get alphax");
  *alphax = data[rows - 1];
  free(data);

  if (!(data = SDDS_GetColumnInDoubles(SDDSin, "etax")))
    SDDS_Bomb("unable to get etax");
  *etax = data[rows - 1];
  free(data);

  if (!(data = SDDS_GetColumnInDoubles(SDDSin, "etaxp")))
    SDDS_Bomb("unable to get etax");
  *etaxp = data[rows - 1];
  free(data);

  if (!(data = SDDS_GetColumnInDoubles(SDDSin, "betay")))
    SDDS_Bomb("unable to get betay");
  *betay = data[rows - 1];
  free(data);

  if (!(data = SDDS_GetColumnInDoubles(SDDSin, "alphay")))
    SDDS_Bomb("unable to get alphay");
  *alphay = data[rows - 1];
  free(data);

  if (!(data = SDDS_GetColumnInDoubles(SDDSin, "etay")))
    SDDS_Bomb("unable to get etay");
  *etay = data[rows - 1];
  free(data);

  if (!(data = SDDS_GetColumnInDoubles(SDDSin, "etayp")))
    SDDS_Bomb("unable to get etay");
  *etayp = data[rows - 1];
  free(data);
  if (SDDS_CheckParameter(SDDSin, "ex0", "$gp$rm", SDDS_ANY_FLOATING_TYPE, NULL) == SDDS_CHECK_OK ||
      SDDS_CheckParameter(SDDSin, "ex0", "m", SDDS_ANY_FLOATING_TYPE, NULL) == SDDS_CHECK_OK) {
    if (!SDDS_GetParameterAsDouble(SDDSin, "ex0", ex0))
      SDDS_Bomb("unable to get ex0 parameter from input file");
  } else {
    if (!(data = SDDS_GetColumnInDoubles(SDDSin, "ex")))
      SDDS_Bomb("unable to get ex");
    *ex0 = data[0];
    free(data);
  }
  if (SDDS_CheckParameter(SDDSin, "pCentral", NULL, SDDS_ANY_FLOATING_TYPE, NULL) == SDDS_CHECK_OK) {
    if (!SDDS_GetParameterAsDouble(SDDSin, "pCentral", pCentral))
      SDDS_Bomb("unable to get pCentral parameter from input file");
  } else {
    if (!(data = SDDS_GetColumnInDoubles(SDDSin, "pCentral")))
      SDDS_Bomb("unable to get pCentral");
    *pCentral = data[0];
    free(data);
  }
  if (SDDS_CheckParameter(SDDSin, "Sdelta0", "", SDDS_ANY_FLOATING_TYPE, NULL) == SDDS_CHECK_OK) {
    if (!SDDS_GetParameterAsDouble(SDDSin, "Sdelta0", Sdelta0))
      SDDS_Bomb("unable to get Sdelta0 parameter from input file");
  } else {
    if (!(data = SDDS_GetColumnInDoubles(SDDSin, "Sdelta")))
      SDDS_Bomb("unable to get pCentral");
    *Sdelta0 = data[0];
    free(data);
  }
  /* get ey0 if it is there */
  if (SDDS_GetParameterAsDouble(SDDSin, "ey0", ey0)) {
    ey0Exist = 1;
  } else if ((data = SDDS_GetColumnInDoubles(SDDSin, "ey"))) {
    ey0Exist = 1;
    *ey0 = data[0];
    free(data);
  }
  if (*ex0 <= 0.0)
    SDDS_Bomb("ex0 should be greater than zero.");
  if (!ey0Exist) {
    double Jx, Jy;
    if (SDDS_CheckParameter(SDDSin, "Jx", NULL, SDDS_ANY_FLOATING_TYPE, NULL) != SDDS_CHECK_OK ||
        SDDS_CheckParameter(SDDSin, "Jy", NULL, SDDS_ANY_FLOATING_TYPE, NULL) != SDDS_CHECK_OK ||
        !SDDS_GetParameterAsDouble(SDDSin, "Jx", &Jx) ||
        !SDDS_GetParameterAsDouble(SDDSin, "Jy", &Jy)) {
      SDDS_Bomb("unable to get Jx and/or Jy parameter from input file");
    }
    if (emitRatio == 0 && coupling == 0)
      SDDS_Bomb("No vertical emittance data in file: give -emittanceRatio or -coupling");
    if (coupling) {
      *ex0 = *ex0 / (1 + Jy * coupling / Jx);
      *ey0 = coupling * (*ex0);
    } else {
      *ey0 = *ex0 * emitRatio;
    }
  }
  return 1;
}


/*compute the beam size from twiss parameters
  output: Sx --- sigmaX
  Sy --- sigmaY
  Sxp --- sigmaX prime
  Syp --- sigmaY prime
  */
void ComputeBeamSize(double period, long Nu, double ex, double ey, double Sdelta0, 
		     double betax, double alphax, double etax, double etaxp,
		     double betay, double alphay, double etay, double etayp,
		     double *Sx, double *Sy, double *Sxp, double *Syp)
{
  double gammax, gammay;
  /*double length;*/
  
  /*length = Nu*period;*/
  
  gammax = (1+sqr(alphax))/betax;
  gammay = (1+sqr(alphay))/betay;
  if (Sxp)
    *Sxp = sqrt(ex*gammax + sqr(Sdelta0*etaxp))*1.0e3;
  if (Syp)
    *Syp = sqrt(ey*gammay + sqr(Sdelta0*etayp))*1.0e3;
  if (Sx)
    *Sx = sqrt(ex*betax + sqr(Sdelta0*etax))*1.0e3;
  if (Sy)
    *Sy = sqrt(ey*betay + sqr(Sdelta0*etay))*1.0e3;
}

/* Find the peak of (E,spec) over n points.  If fwhm is non-NULL, also
 * write the FWHM (in E units) to *fwhm, or -1 if it cannot be resolved. */
void FindPeak(double *E, double *spec, double *ep, double *sp, double *fwhm, long n)
{
  long i, ip;
  double e1, e2;

  *sp = spec[0];
  *ep = E[0];
  ip = -1;
  for (i = 1; i < n; i++) {
    if (*sp < spec[i]) {
      *sp = spec[i];
      *ep = E[i];
      ip = i;
    }
  }
  if (fwhm && ip != -1) {
    *fwhm = -1;
    e1 = e2 = -1;
    for (i = ip; i >= 1; i--) {
      if (spec[i] < (spec[ip] / 2)) {
        e1 = E[i] + (E[i + 1] - E[i]) / (spec[i + 1] - spec[i]) * (spec[ip] / 2 - spec[i]);
        break;
      }
    }
    for (i = ip; i < (n - 1); i++) {
      if (spec[i + 1] < (spec[ip] / 2)) {
        e2 = E[i] + (E[i + 1] - E[i]) / (spec[i + 1] - spec[i]) * (spec[ip] / 2 - spec[i]);
        break;
      }
    }
    if (e1 > 0 && e2 > 0)
      *fwhm = e2 - e1;
  }
}

/* Convolve (E,spec) of length *ns with a Gaussian of relative width sigmaE.
 * The peak is located internally to scale the kernel.  Output: *ns is
 * shrunk to the convolved range; convolved arrays are written back into
 * E and spec.  nstart/nend are optional: if non-NULL, they receive the
 * begin/end indices into the original input arrays.
 * Returns 0 on success, 1 on failure. */
int Gauss_Convolve(double *E, double *spec, long *ns, double sigmaE, long *nstart, long *nend)
{
  long nSigma = 3, nppSigma = 6, ne1, ne2, ns1, np;
  long i, j;
  double ep, sp, sigp, de, sum, *gs, x, *spec2;

  ns1 = *ns;
  gs = spec2 = NULL;

  if (!E || !spec) {
    fprintf(stderr, "No energy or spectra points!\n");
    return 1;
  }
  FindPeak(E, spec, &ep, &sp, NULL, ns1);
  /* generate Gaussian with correct sigma in units of x-axis */
  de = E[1] - E[0];
  sigp = 2.0 * sigmaE * ep / de;

  if (sigp < (nppSigma - 1)) {
    fprintf(stderr, "too few data points for Gaussian convolution\n");
    fprintf(stderr, "de = %e, sigmaE = %e, ep = %e, sigp=%le\n", de, sigmaE, ep, sigp);
    return 1;
  }
  np = (2 * nSigma) * sigp + 1;
  if (np % 2 == 0)
    np = np + 1; /* make odd */
  gs = (double *)calloc(np, sizeof(*gs));
  spec2 = (double *)calloc(ns1, sizeof(*spec2));
  sum = 0.0;
  for (i = 0; i < np; i++) {
    x = i * 1.0 - 0.5 * (np - 1);
    gs[i] = exp(-x * x / 2.0 / (sigp * sigp));
    sum = sum + gs[i];
  }
  /* make convolution */
  ne1 = np / 2;
  ne2 = ns1 - ne1 - 1;
  if (ne2 < 0) {
    fprintf(stderr, "Error: Check the number of peak search points\n");
    free(spec2);
    free(gs);
    return 1;
  }
  for (i = ne1; i <= ne2; i++) {
    spec2[i] = 0.0;
    for (j = 0; j < np; j++)
      spec2[i] = spec2[i] + spec[i + ne1 - j] * gs[j];
  }
  /* return in original array and make adjustment of array sizes */
  *ns = ne2 - ne1 + 1;
  if (nstart)
    *nstart = ne1;
  if (nend)
    *nend = ne2;
  for (i = ne1; i <= ne2; i++) {
    E[i - ne1] = E[i];
    spec[i - ne1] = spec2[i] / sum;
  }
  free(spec2);
  free(gs);
  return 0;
}
