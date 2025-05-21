#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265358979323846264338328
#define TWOPI 6.28318530717958647692528676656

typedef struct COMPLEX {double re; double im;} COMPLEX;

	/*****  Computes the undulator brightness produced by a Gaussian electron beam  ***/
double computeBrightnessLindberg(double radLambda, int radHarm, double radDet,
			 double undLength, int undN, double undK,
			 double emitx, double emity, double betax, double betay,
			 double alphax, double alphay, double sigmaDelta, double current);
	/*****  Computes the complex function to be integrated by computeBrightness()  ***/
COMPLEX computeIntegrand(double twoPiSigmapNu, double twoPiDetuneNu,
			double epsx_epsr, double epsy_epsr,
			double betax_undL, double betay_undL,
			double alphax, double alphay, double phaseFactor, double xi, double tauq);

	/*****  Computes the undulator flux in the central cone near an odd harmonic  ***/
double computeFLuxLindberg(int radHarm, int undN, double undK,
			   double radDet, double sigmaDelta, double current);

        /*****  Simple functions for complex numbers  ***/
COMPLEX ComplexAdd(COMPLEX x, COMPLEX y);
COMPLEX ComplexMult(COMPLEX x, COMPLEX y);
COMPLEX ComplexDivide(COMPLEX x, COMPLEX y);
COMPLEX ComplexSqrt(COMPLEX x);

        /*****  Calculates the Bessel function J(x,n)  ***/
double BesselFuncJ(double x, int n);

/*********************************************************************************/
/****  Computes the undulator brightness produced by a Gaussian electron beam  ***/
/*********************************************************************************/
/* Inputs are radiation wavelength (m), undulator harmonic (odd integer),        */
/* scaled detuning from resonance: \Delta\lambda/\lambda = (2pi*Nu/h)radDet      */
/*		[good choices: radDet = 0 (resonance), -0.5, or -1.0 (max flux)] */
/* undulator length (m), # periods (integer), undulator K, emitances in x,y (m), */
/* beta_x,y functions (m) and correlation alpha_x,y at the undulator middle,     */
/* electron beam energy spread dE/E, and current (A)                             */
/*********************************************************************************/
double computeBrightnessLindberg(double radLambda, int radHarm, double radDet,
			 double undLength, int undN, double undK,
			 double emitx, double emity, double betax, double betay,
			 double alphax, double alphay, double sigmaDelta, double current)
{
  double *xi, *tau, *weight;
  COMPLEX integral, integrand1, integrand2;

  // define relevant detuning, energy spread paramegters
  double twoPiDetuneNu = TWOPI*radDet;
  double twoPiSigmapNu = TWOPI*sigmaDelta*(double)(radHarm*undN);

  // define dimensionless emitx,y/emitrad and betax,y/Lu
  double epsr = radLambda/(4.0*PI);
  double epsx_epsr = emitx/epsr;
  double betax_undL = betax/undLength;
  double epsy_epsr = emity/epsr;
  double betay_undL = betay/undLength;

  double dxi, fluxFactor, phaseFactor;
  int i, j, Nxi, Ntau;

  dxi = undK*undK;
  dxi = (double)radHarm*dxi/(4.0 + 2.0*dxi);
  if((radHarm-1)%2 == 0) {
    dxi = undK*( BesselFuncJ(dxi, (radHarm-1)/2) - BesselFuncJ(dxi, (radHarm+1)/2) );
    dxi = (double)(undN*radHarm)*dxi*dxi;	// Nu*h*(K[JJ]_h)^2
  }
  else {
    printf("harmonic needs to be odd!\n");
    return(-1.0);
  }
  fluxFactor = 0.5*radLambda*1.e6;				// radiaton phase space area in mm*mrad
  fluxFactor = 0.5*1.4308852545e14/(fluxFactor*fluxFactor);	// (pi/2)(fine structure)/e in 0.1%BW
  fluxFactor = current*fluxFactor*dxi/(1.0 + 0.5*undK*undK);	// multiply by I*Nu*h(K[JJ]_h)^2/(1+K^2/2)

  Ntau = 6;
  // We integrate along tau using 6 point Gauss-Legendre quadrature
  tau = calloc(Ntau, sizeof(double));
  weight = calloc(Ntau, sizeof(double));
  tau[0] =  0.2386191860831969;  weight[0] = 0.4679139345726910;
  tau[1] = -0.2386191860831969;  weight[1] = 0.4679139345726910;
  tau[2] =  0.6612093864662648;  weight[2] = 0.36076157304813783;
  tau[3] = -0.6612093864662648;  weight[3] = 0.36076157304813783;
  tau[4] =  0.9324695142031520;  weight[4] = 0.1713244923791703;
  tau[5] = -0.9324695142031520;  weight[5] = 0.1713244923791703;

  // We integrate along xi using trapezoidal rule and a number of sampling points that
  // depends upon the beam emittances/(lambda/4\pi), targeting a max error < 0.5%
  if((epsx_epsr < 3.0e2) && (epsy_epsr < 3.0e2)) {
    if((epsx_epsr > 3.0e-3) && (epsy_epsr > 3.0e-3))
      Nxi = 100;
    else
      Nxi = 200;
  }
  else {
    if((epsx_epsr < 1.0e4) && (epsy_epsr < 1.0e4))
      Nxi = 300;
    else {
      if((epsx_epsr < 3.0e5) && (epsy_epsr < 3.0e5))
	Nxi = 800;
      else
	Nxi = 2000;
    }
  }
  xi = calloc(Nxi, sizeof(double));
  // The integration spacing dxi increases quadratically as xi increases from 0 to 1
  xi[0] = 0.0;
  for(i=1; i<Nxi; i++) {
    dxi = (double)((Nxi-1)*(2*Nxi-1));
    dxi = (double)(6*i*i)/(dxi*(double)Nxi);
    xi[i] = xi[i-1] + dxi;
  }

  // add small value to avoid singularity when emittance -> 0...changes calculation by <0.4%
  epsx_epsr += 2.0e-4;
  epsy_epsr += 2.0e-4;
  integral.re = 0.0;
  integral.im = 0.0;

  phaseFactor = undK*undK;
  phaseFactor = (double)radHarm*phaseFactor/( (2.0 + phaseFactor)*PI*(double)undN );
  // 6 point Gauss-Legendre quadrature
  for(j=0; j<Ntau; j++) {
    // trapezoidal rule in xi with a variable dxi
    integrand1 = computeIntegrand(twoPiSigmapNu, twoPiDetuneNu, epsx_epsr, epsy_epsr,
				betax_undL, betay_undL, alphax, alphay, phaseFactor, xi[0], tau[j]);
    for(i=1; i<Nxi; i++) {
      integrand2 = computeIntegrand(twoPiSigmapNu, twoPiDetuneNu, epsx_epsr, epsy_epsr,
				betax_undL, betay_undL, alphax, alphay, phaseFactor, xi[i], tau[j]);
      integral.re += weight[j]*(xi[i]-xi[i-1])*(integrand1.re + integrand2.re);
      integrand1 = integrand2;
    }
  }
  integral.re *= 0.5*fluxFactor;
  //printf("%17.10e %17.10e %17.10e \n", undK, fluxFactor, epsx_epsr*epsy_epsr);

  free(xi);
  free(tau);  free(weight);

  return(integral.re);
}

/* computes the integrand of the function in computeBrightness */
COMPLEX computeIntegrand(double twoPiSigmapNu, double twoPiDetuneNu,
			double epsx_epsr, double epsy_epsr,
			double betax_undL, double betay_undL,
			double alphax, double alphay, double phaseFactor, double xi, double tauq)
{
  COMPLEX integrand;
  COMPLEX numer, numerx, denomx, denomy, denom;
  double temp, tau = (1.0 - xi)*tauq;

  numer.im = twoPiSigmapNu*xi;
  numer.im = (1.0 - xi)*exp(-2.0*numer.im*numer.im)/PI;
  numer.re =  numer.im*cos(twoPiDetuneNu*xi);
  numer.im = -numer.im*sin(twoPiDetuneNu*xi);

  denomx.re = (1.0 + alphax*alphax)*0.25*(tau*tau - xi*xi)/betax_undL;
  denomx.re += betax_undL - alphax*tau;
  denomx.re = 2.0*epsx_epsr*denomx.re;
  denomx.im = (1.0 + epsx_epsr*epsx_epsr)*xi;

  numerx.re = -phaseFactor*(1.0 + epsx_epsr*epsx_epsr);
  numerx.im =  0.0;
  numerx = ComplexDivide(numerx, denomx);
  temp = exp(numerx.re);
  numerx.re = temp*cos(numerx.im);
  numerx.im = temp*sin(numerx.im);
  numer = ComplexMult(numer, numerx);

  denomy.re = (1.0 + alphay*alphay)*0.25*(tau*tau - xi*xi)/betay_undL;
  denomy.re += betay_undL - alphay*tau;
  denomy.re = 2.0*epsy_epsr*denomy.re;
  denomy.im = (1.0 + epsy_epsr*epsy_epsr)*xi;

  denom = ComplexSqrt(ComplexMult(denomx, denomy));
  integrand = ComplexDivide(numer, denom);

  return(integrand);
}

/*********************************************************************************/
/***  Simple calculation of the flux in the central cone for undular radiation ***/
/*********************************************************************************/
/* Inputs are undulator harmonic (odd integer), number of periods, K parameter,  */
/* and scaled detuning from resonance: \Delta\lambda/\lambda = (Nu/h)radDet      */
/*	     [good choices: radDet = 0 (resonance), -0.5, or -1.0 (max flux)]    */
/* electron beam energy spread dE/E, and current (A)                             */
/*********************************************************************************/
/* This calculation includes lowest order effects of energy spread but neglects  */
/* off-axis contributions, so we require |radDet| < 2 or so about an odd integer */
/*********************************************************************************/

double computeFluxLindberg(int radHarm, int undN, double undK,
			   double radDet, double sigmaDelta, double current)
{
// parameters for Gauss-Kronod integration
  double *xiK15, *weightK15, *weightG7;
// output flux, 1st and 2nd estimate, and prefactor
  double flux, flux1, flux2, fluxFactor;
  double weight, tempXi, temp1, temp2, error;
// define relevant detuning, energy spread paramegters
  double twoPiDetuneNu = TWOPI*radDet;
  double twoPiSigmapNu = TWOPI*sigmaDelta*(double)(radHarm*undN);

// sub-divisions of the initial integration interval
  int isub, Nsubdiv = 1;
  int ix, Nxi;

  fluxFactor = undK*undK;
  fluxFactor = (double)radHarm*fluxFactor/(4.0 + 2.0*fluxFactor);
  if((radHarm-1)%2 == 0) {
    fluxFactor = undK*( BesselFuncJ(fluxFactor, (radHarm-1)/2)
			- BesselFuncJ(fluxFactor, (radHarm+1)/2) );
    fluxFactor = (double)(undN*radHarm)*fluxFactor*fluxFactor;	// Nu*h*(K[JJ]_h)^2
  }
  else {
    printf("harmonic needs to be odd!\n");
    return(-1.0);
  }
  fluxFactor = 0.5*1.4308852545e14*current*fluxFactor/(1.0 + 0.5*undK*undK);
		// multiply by (pi/2)(fine structure)(I/e)/(1+K^2/2) in /0.1%BW/s

// We integrate using the 15-point Kronrod rule nested in a 7-point Gauss rule
  Nxi = 15;
  xiK15 = calloc(Nxi, sizeof(double));
  weightK15 = calloc(Nxi, sizeof(double));
  weightG7 = calloc(Nxi, sizeof(double));

  xiK15[0] = -0.991455371120813;  weightK15[0] = 0.022935322010529;
  xiK15[1] = -0.949107912342759;  weightK15[1] = 0.063092092629979;
  xiK15[2] = -0.864864423359769;  weightK15[2] = 0.104790010322250;
  xiK15[3] = -0.741531185599394;  weightK15[3] = 0.140653259715525;
  xiK15[4] = -0.586087235467691;  weightK15[4] = 0.169004726639267;
  xiK15[5] = -0.405845151377397;  weightK15[5] = 0.190350578064785;
  xiK15[6] = -0.207784955007898;  weightK15[6] = 0.204432940075298;
  xiK15[7] =  0.0;		  weightK15[7] = 0.209482141084728;
  xiK15[8] =  0.207784955007898;  weightK15[8] = 0.204432940075298;
  xiK15[9] =  0.405845151377397;  weightK15[9] = 0.190350578064785;
  xiK15[10] = 0.586087235467691;  weightK15[10] = 0.169004726639267;
  xiK15[11] = 0.741531185599394;  weightK15[11] = 0.140653259715525;
  xiK15[12] = 0.864864423359769;  weightK15[12] = 0.104790010322250;
  xiK15[13] = 0.949107912342759;  weightK15[13] = 0.063092092629979;
  xiK15[14] = 0.991455371120813;  weightK15[14] = 0.022935322010529;

// even weights are zero to get 7 point rule
  weightG7[0] = 0.0;  weightG7[1] = 0.129484966168870;
  weightG7[2] = 0.0;  weightG7[3] = 0.279705391489277;
  weightG7[4] = 0.0;  weightG7[5] = 0.381830050505119;
  weightG7[6] = 0.0;  weightG7[7] = 0.417959183673469;
  weightG7[8] = 0.0;  weightG7[9] = 0.381830050505119;
  weightG7[10] = 0.0;  weightG7[11] = 0.279705391489277;
  weightG7[12] = 0.0;  weightG7[13] = 0.129484966168870;

  while(Nsubdiv > 0) {
    flux1 = 0.0;
    flux2 = 0.0;
    for(isub=0; isub<Nsubdiv; isub++) {
      for(ix=0; ix<Nxi; ix++) {
	// define point within subdivision
	tempXi = ( xiK15[ix] + (double)(2*isub+1-Nsubdiv) )/(double)Nsubdiv;
	tempXi = 0.5*(tempXi + 1.0);
	// integrand things
	temp1  = twoPiSigmapNu*tempXi;
	temp2  = sin(twoPiDetuneNu*tempXi);
	flux = exp(-2.0*temp1*temp1)*(temp2 - temp2/tempXi);
	// weight within subdivision
	weight = weightK15[ix]/(double)Nsubdiv;
	flux1 += weight*flux;
	weight = weightG7[ix]/(double)Nsubdiv;
	flux2 += weight*flux;
      }
    }
    // compute flux from both rules
    flux1 = fluxFactor*(1.0 + (1.0/PI)*flux1);
    flux2 = fluxFactor*(1.0 + (1.0/PI)*flux2);
    error = fabs(flux1-flux2)/flux1;
    // exit if difference is < 1/10^5 (this calculation is only approximate anyway)
    if(error < 1.e-5)
      Nsubdiv = -1;
    else {
      if(Nsubdiv < 513)
	Nsubdiv *= 2;
      else {
	printf("Warning, after 9 subdivisions the fractional errror is %e\n", error);
	Nsubdiv = -1;
      }
    }
  }
  return(flux1);
}


/****************************************************************/
/* Addition of complex numbers				        */
/****************************************************************/
COMPLEX ComplexAdd(COMPLEX x, COMPLEX y)
{
  COMPLEX z;

  z.re = x.re + y.im;
  z.im = x.im + y.im;

  return(z);
}

/****************************************************************/
/* Multiplication of complex numbers				*/
/****************************************************************/
COMPLEX ComplexMult(COMPLEX x, COMPLEX y)
{
  COMPLEX z;

  z.re = x.re*y.re - x.im*y.im;
  z.im = x.re*y.im + x.im*y.re;

  return(z);
}

/****************************************************************/
/* Division of complex numbers (z=x/y)				*/
/****************************************************************/
COMPLEX ComplexDivide(COMPLEX x, COMPLEX y)
{
  COMPLEX z;
  double inv_mag;

  inv_mag = 1.0/(y.re*y.re + y.im*y.im);
  z.re = (x.re*y.re + x.im*y.im)*inv_mag;
  z.im = (x.im*y.re - x.re*y.im)*inv_mag;

  return(z);
}

/****************************************************************/
/* Square root of complex number				*/
/****************************************************************/
COMPLEX ComplexSqrt(COMPLEX x)
{
  COMPLEX z;
  double phase, sqrt_mag;

  phase = atan2(x.im, x.re);
  sqrt_mag = sqrt(sqrt(x.re*x.re + x.im*x.im));
  z.re = sqrt_mag*cos(0.5*phase);
  z.im = sqrt_mag*sin(0.5*phase);

  return(z);
}

/****************************************************************/
/* Bessel function of integer order n				*/
/****************************************************************/
double BesselFuncJ(double x, int n)
{
  double besselJ = 1.0;
  double d = 1.0;
  double r = 1.0;

  int f;

  n += 1;

  f = 2 + 2*(int)(sqrt( 0.5*(fabs(x) + 7.0)*(fabs(x) + 7.0) + 0.25*(double)(n*n) ));

  do {
    r = x/( 2.0*(double)f - r*x );
    if(f < n)
      besselJ = besselJ*r;

    d = d*r + 2.0*( 0.5*(double)f - (double)(f/2) );
    f = f - 1;
  }  while(f != 0);

  besselJ = besselJ/(2.0*d - 1.0);

  return(besselJ);
}
