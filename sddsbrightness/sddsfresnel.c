/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/**
 * @file sddsfresnel.c
 * @brief One-dimensional Fresnel diffraction pattern computation program
 * 
 * This program computes one-dimensional Fresnel diffraction patterns of a 
 * general mask design for point and Gaussian sources. It uses the Fresnel
 * integrals to calculate the diffraction amplitude and intensity at various
 * points in the image plane.
 * 
 * The program supports:
 * - Multiple apertures with different geometries
 * - Polychromatic sources with bandwidth effects
 * - Gaussian source profiles
 * - Custom source and spectrum profiles from SDDS files
 * - Convolution with source size effects
 * 
 * @author Bingxin Yang
 * @author Hairong Shang
 * @version 1.0
 * @date 2025
 * @organization Argonne National Laboratory
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <ctype.h>
#include <math.h>
#include <complex.h>
#include "fresnel.h"

#include "fftpackC.h"
#include "SDDS.h"
#include "mdb.h"
#include "scan.h"
#include "rpn.h"

#define SET_SPECTRUM 0
#define SET_SOURCE 1
#define SET_APERTURE 2
#define SET_IMAGE 3
#define SET_VERBOSE 4
#define N_OPTIONS 5

static char *option[N_OPTIONS]={"spectrum", "source", "aperture", "image", "verbose"};

static char *USAGE="sddsfresnel computes one-dimensional Fresnel diffraction patterns of a \n\
 * general mask design for point and Gaussian sources. It uses the Fresnel \n\
 * integrals to calculate the diffraction amplitude and intensity at various \n\
 * points in the image plane. \n\
 * \n\
 * The program supports:\n\
 * - Multiple apertures with different geometries \n\
 * - Polychromatic sources with bandwidth effects \n\
 * - Gaussian source profiles \n\
 * - Custom source and spectrum profiles from SDDS files \n\
 * - Convolution with source size effects/. \n\
\n\
The usages are: \n\
sddsfresnel <inputFile> <outputFile> \n\
                   -spectrum=wavelength=nm,energy=eV,bandWidth=ss,nwaves=ss,file=<filename> \n\
                   -source=distance=m,center=mm,width=mm,file=<filename> \n\
                   -aperture=center=mm,width=mm,peakamplitude=tt,phaseshift=pp,symmetry=ss \n\
                   -image=distance=m,center=mm,width=mm,gridpoints=gg \n\
                   [-verbose=nn] \n\
<inputFile>    optional, if provided, it is used to supply default specifications of the source, apertures, and the image: \n\
                   (1) the apertures are specified by data columns: ApertureCenter, ApertureWidth, Amplitude, and Phase \n\
                   (2) defaults of other parameters are replaced by SDDS parameters if present: \n\
                       SourceDistance, SouceCenter, SourceWidth, PhotonEnergy, PhotonWavelength,  \n\
                       PhotonBandWidth, ImageDistance, ImageWidth, nImagePoints. The parameters \n\
                       specified in the command line options will replace the values provided by the input file.\n\
<outputFile>   required, filename for data output. The output file contains all input data as parameters and the following data columns: \n\
                   xImage     -- transverse grid data points in the image plane. \n\
                   xSource    -- grid data points in the source plane, mapped from the image plan grid. \n\
                   ImageAmp0  -- image amplitude at the center wavelength for zero source size. \n\
                   ImageIntensity0  -- multi-wavelength image intensity for zero source size. \n\
                   ImageIntensity   -- image intensity after convolution with Gaussian source profile. \n\
spectrum       photon spectrum specifications: \n\
                   wavelength -- photon wavelength in units of nm, default=0.1nm\n\
                   energy     -- photon energy in eV, used to calculate wavelength,default=0\n\
                   bandwidth  -- rms width of the Gaussian spectrum Sigma(Lamda)/Lamda, defualt=0\n\
                   nwaves     -- (odd) number of data points in the wavelength grid, default=1. \n\
                   profile    -- provide the photn wavelenth distribution profile through an SDDS file \n\
                                 with two columns PhotonWavelength and Weight.\n\
                                 if not provided and bandwidth>0, a gaussian distribution will be used.\n\
source         source geometry specifications:\n\
                   distance   -- source distance from the diffraction mask in units of m, default = 10.\n\
                   center     -- position of the source center in the units of mm, default = 0.\n\
                   width      -- rms width of the source in units of mm, default = 0. \n\
                   wavelength -- photon wavelength in units of nm, default = 0.1nm\n\
                   profile    -- provide the source geometry distribution profile through an SDDS file \n\
                                 with two columns: xSource and SourceIntensity. \n\
aperture       diffraction aperture specification(s). Use N specifications for N apertures.\n\
                   center     -- position of the aperture center in units of mm, default = 0 \n\
                   width      -- full width of the aperture in units of mm,  default = 0.01 \n\
                   amplitude  -- peak amplitude after the aperture, no units, default = 1.\n\
                   phase      -- phase shift through the aperture in units of degrees, default = 0; \n\
                   symmetry   -- add another aperture symmetric to point x = 0, default = 0; \n\
image          image detector specifications. \n\
                   distance   -- detector distance from the diffraction mask in units of m, default = 10. \n\
                   center     -- center of the detector grid in units of mm, default = 0 \n\
                   width      -- total width of the detector screen, default = 1. \n\
                   gridpoints -- number of points in the detector plane grid, default = 1024. \n\
verbose          screen output control. controls how detailed the screen output is. \n\n\
sddsxra computes one-dimensional Fresnel diffraction patterns of a general mask design for point and Gaussian sources.\n\n\
Program by Bingxin Yang and Hairong Shang.  ANL(This is version 1.0, "__DATE__")\n";


/**
 * @struct APERTURE
 * @brief Structure to define aperture properties for Fresnel diffraction
 * 
 * This structure contains all the geometric and optical properties
 * needed to define a single aperture in the diffraction mask.
 */
typedef struct {
  double center;     /**< Position of aperture center in mm */
  double width;      /**< Full width of aperture in mm */
  double amplitude;  /**< Peak amplitude transmission (dimensionless) */
  double phase;      /**< Phase shift through aperture in degrees */
  double complex Ap; /**< Complex amplitude factor (amplitude * exp(-i*phase)) */
} APERTURE;

/* Function prototypes */

/**
 * @brief Calculate the complex amplitude contribution from a single aperture
 * @param aperture The aperture structure containing geometry and optical properties
 * @param wavelength Photon wavelength in nm
 * @param phi Observation angle in radians
 * @param f Focal length in mm
 * @return Complex amplitude contribution from this aperture
 */
double complex one_aperture_amplitude(APERTURE aperture, double wavelength, double phi, double f);

/**
 * @brief Initialize the output SDDS file with proper column and parameter definitions
 * @param outTable Pointer to SDDS dataset structure
 * @param outputFile Name of the output file
 */
void SetupOutputFile(SDDS_DATASET *outTable, char *outputFile);

/**
 * @brief Initialize an aperture structure with default values
 * @param aperture Pointer to aperture structure to initialize
 */
void initialize_aperture(APERTURE *aperture);

/**
 * @brief Read input parameters and aperture data from SDDS file
 * @param inputFile Name of input SDDS file
 * @param aperture Pointer to aperture array (will be reallocated)
 * @param apertures Pointer to number of apertures (will be updated)
 * @param sourceDistance Pointer to source distance parameter
 * @param sourceCenter Pointer to source center parameter
 * @param sourceWidth Pointer to source width parameter
 * @param photonWavelength Pointer to photon wavelength parameter
 * @param photonEnergy Pointer to photon energy parameter
 * @param photonBandwidth Pointer to photon bandwidth parameter
 * @param nWaves Pointer to number of wavelength points parameter
 * @param imageDistance Pointer to image distance parameter
 * @param imageCenter Pointer to image center parameter
 * @param imageWidth Pointer to image width parameter
 * @param nImagePoints Pointer to number of image points parameter
 */
void ReadInputFile(char *inputFile, APERTURE *aperture, int32_t *apertures, 
                   double *sourceDistance, double *sourceCenter, double *sourceWidth,
                   double *photonWavelength, double *photonEnergy, double *photonBandwidth, int32_t *nWaves,
                   double *imageDistance, double *imageCenter, double *imageWidth, int32_t *nImagePoints);

/**
 * @brief Reorder array for FFT convolution (wrap-around ordering)
 * @param response1 Output reordered array (size 2*nsig+2)
 * @param t Independent variable array
 * @param response Input response array
 * @param nres Number of response points
 * @param nsig Number of signal points
 */
void wrap_around_order(double *response1, double *t, double *response,
                       long nres, long nsig);

/**
 * @brief Multiply two complex numbers stored as separate real and imaginary parts
 * @param r0 Pointer to real part of result
 * @param i0 Pointer to imaginary part of result
 * @param r1 Real part of first complex number
 * @param i1 Imaginary part of first complex number
 * @param r2 Real part of second complex number
 * @param i2 Imaginary part of second complex number
 */
void complex_multiply(double *r0, double *i0, double r1, double i1,
                      double r2, double i2);

/**
 * @brief Convolve two signals using FFT
 * @param signal1 First signal array (diffraction intensity)
 * @param indep1 Independent variable for first signal
 * @param signal2 Second signal array (source profile)
 * @param indep2 Independent variable for second signal
 * @param points Number of points in arrays
 * @return Pointer to convolved signal array (caller must free)
 */
double *convolve(double *signal1, double *indep1, double *signal2, double *indep2, long points);

/**
 * @brief Main function for sddsfresnel program
 * 
 * This function processes command line arguments, reads input files,
 * performs Fresnel diffraction calculations, and writes results to
 * an output SDDS file.
 * 
 * @param argc Number of command line arguments
 * @param argv Array of command line argument strings
 * @return 0 on success, non-zero on error
 */
int main ( int argc, char *argv[] )
{
  long    i_arg, i, verbose=0, k, m;
  SDDS_DATASET SDDSout;
  char *inputfile=NULL, *outputfile=NULL, *sourceProfile=NULL, *spectrumProfile=NULL;
  SCANNED_ARG *s_arg=NULL;
  /*parameters for photon spectrum. Overiding rules: Command Line (CL) > file input > default */
  double photonEnergy = 0, photonWavelength = 0.1, photonBandwidth=0;
  double photonEnergyCL = 0, photonWavelengthCL = 0, photonBandwidthCL=0;
  int32_t nWavesCL = 0, nWaves = 1;
  /*parameters for source. Overiding rules: Command Line (CL) > file input > default */
  double sourceDistance = 10, sourceCenter = 0, sourceWidth = 0;
  double sourceDistanceCL = 0, sourceCenterCL = -9999, sourceWidthCL = 0;
  /* image parameters Overiding rules: Command Line (CL) > file input > default */
  double imageDistance = 10, imageCenter = 0, imageWidth = 1;
  double imageDistanceCL = 0, imageCenterCL = -9999, imageWidthCL = 0;
  int32_t nImagePointsCL = 0, nImagePoints = 1024;
  /* aperture parameters: no override, all apertures are counted */
  int32_t apertures = 0, nSourcePoints=0; /*default to compute one aperture only */
  APERTURE *aperture=NULL;
  /*temporarily only compute one aperture, in the future, it would be an array */
  unsigned long dummyFlags=0, symmetryFlag=0, interpCode;
  double *wavelength=NULL, *xSource=NULL, *intensity0 = NULL, *weight=NULL, *xSource0=NULL, *sourceSig0=NULL;
  double *sourceSig=NULL, *intensity=NULL;
  double miny, stepy, f, M, imageAngle, deltaWave, sum, xImage;
  double complex Amp0, Amp1;
  OUTRANGE_CONTROL aboveRange, belowRange;
  /* double *phi = NULL, *imageAmp=NULL, *imageInten=NULL; */
  
  verbose = 1;
  aboveRange.flags = belowRange.flags = OUTRANGE_VALUE;
  belowRange.value=aboveRange.value=0;
  
  SDDS_RegisterProgramName(argv[0]);
  argc = scanargs(&s_arg, argc, argv);
  if (argc<2) {
    fprintf(stdout, "%s", USAGE);
    exit(0);
  }
  
  for (i_arg=1; i_arg<argc; i_arg++) {
    if (s_arg[i_arg].arg_type==OPTION) {
      switch (match_string(s_arg[i_arg].list[0], option, N_OPTIONS, 0)) {
      case SET_SPECTRUM:
        s_arg[i_arg].n_items--;
        if (!scanItemList(&dummyFlags, s_arg[i_arg].list+1, &s_arg[i_arg].n_items, 0,
                          "wavelength", SDDS_DOUBLE, &photonWavelengthCL, 1, 0,
                          "energy", SDDS_DOUBLE, &photonEnergyCL, 1, 0,
                          "bandwidth", SDDS_DOUBLE, &photonBandwidthCL, 1, 0,
			  "nwaves", SDDS_LONG, &nWavesCL, 1, 0, 
                          "profile", SDDS_STRING, &spectrumProfile, 1, 0,
                          NULL))
          SDDS_Bomb("invalid -spectrum syntax");
        if (nWaves<=0)
          SDDS_Bomb("invalid -spectrum syntax: nWaves <= 0.");
        break;
      case SET_SOURCE:
	s_arg[i_arg].n_items--;
        if (!scanItemList(&dummyFlags, s_arg[i_arg].list+1, &s_arg[i_arg].n_items, 0,
                          "distance", SDDS_DOUBLE, &sourceDistanceCL, 1, 0,
                          "center", SDDS_DOUBLE, &sourceCenterCL, 1, 0,
                          "width", SDDS_DOUBLE, &sourceWidthCL, 1, 0,
                          "profile", SDDS_STRING, &sourceProfile, 1, 0,
                          NULL))
          SDDS_Bomb("invalid -source syntax");
	break;
      case SET_APERTURE:
        symmetryFlag = 0;
        s_arg[i_arg].n_items--;
        aperture = (APERTURE*)SDDS_Realloc(aperture, sizeof(*aperture)*(apertures + 1));
        initialize_aperture(aperture+apertures);
        if (!scanItemList(&symmetryFlag, s_arg[i_arg].list+1, &s_arg[i_arg].n_items, 0,
                          "center", SDDS_DOUBLE, &aperture[apertures].center, 1, 0,
                          "width", SDDS_DOUBLE, &aperture[apertures].width, 1, 0,
                          "amplitude", SDDS_DOUBLE, &aperture[apertures].amplitude, 1, 0,
                          "phase", SDDS_DOUBLE, &aperture[apertures].phase, 1, 0,
                          "symmetry", SDDS_LONG, &symmetryFlag, 1, 0,
                          NULL))
          SDDS_Bomb("invalid -aperture syntax");
        apertures ++;
        if (symmetryFlag) {
          /*fprintf(stderr, "apply symmetry.\n"); */
          aperture = (APERTURE*)SDDS_Realloc(aperture, sizeof(*aperture)*(apertures + 1));
          aperture[apertures].center = - aperture[apertures-1].center;
          aperture[apertures].width = aperture[apertures-1].width;
          aperture[apertures].amplitude = aperture[apertures-1].amplitude;
          aperture[apertures].phase = aperture[apertures-1].phase;
          apertures++;
        }
        break;
      case SET_IMAGE:
        s_arg[i_arg].n_items--;
        if (!scanItemList(&dummyFlags, s_arg[i_arg].list+1, &s_arg[i_arg].n_items, 0,
                          "distance", SDDS_DOUBLE, &imageDistanceCL, 1, 0,
                          "center", SDDS_DOUBLE, &imageCenterCL, 1, 0,
                          "width", SDDS_DOUBLE, &imageWidthCL, 1, 0,
			  "ngridpoints", SDDS_LONG, &nImagePointsCL, 1, 0, 
                          NULL))
          SDDS_Bomb("invalid -image syntax");
        if (nImagePoints<=0)
          SDDS_Bomb("invalid -image syntax: nImagePoints <= 0.");
	break;
      case SET_VERBOSE:
 	verbose = 1;
 	if (s_arg[i_arg].n_items > 1) {
 	  if ( !get_long(&verbose, s_arg[i_arg].list[1]) )
 	    SDDS_Bomb("invalid -verbose value");
 	}
	break;
      default:
	fprintf(stdout, "error: unknown switch: %s\n", s_arg[i_arg].list[0]);
	fflush(stdout);
	exit(1);
	break;
      }
    } else {
      if (inputfile==NULL)
        inputfile = s_arg[i_arg].list[0];
      else if (outputfile==NULL)
        outputfile = s_arg[i_arg].list[0];
      else
        SDDS_Bomb("too many filenames");
    }
  }
  if (!outputfile) {
    outputfile = inputfile;
    inputfile = NULL;
  }
  if (!outputfile)
    SDDS_Bomb("Output file not provided.");
  if (inputfile) 
    ReadInputFile(inputfile, aperture, &apertures, &sourceDistance, &sourceCenter, &sourceWidth, &photonWavelength,
                  &photonEnergy, &photonBandwidth, &nWaves,
                  &imageDistance, &imageCenter, &imageWidth, &nImagePoints);

  /* Command line inputs override file inputs */
  if (photonWavelengthCL > 0) photonWavelength = photonWavelengthCL;
  if (photonEnergyCL > 0)     photonEnergy = photonEnergyCL;
  if (photonBandwidthCL > 0)  photonBandwidth = photonBandwidthCL;
  if (nWavesCL > 0)           nWaves = nWavesCL;
  if (sourceDistanceCL > 0)   sourceDistance = sourceDistanceCL;
  if (imageDistanceCL > 0)    imageDistance = imageDistanceCL;
  if (sourceCenterCL > -9998) sourceCenter = sourceCenterCL;
  if (imageCenterCL > -9998)  imageCenter = imageCenterCL;
  if (sourceWidthCL > 0)      sourceWidth = sourceWidthCL;
  if (imageWidthCL > 0)       imageWidth = imageWidthCL;
  if (nImagePointsCL > 0)     nImagePoints = nImagePointsCL;

  if (photonEnergy>0)
    photonWavelength = 1239.8418 / photonEnergy;
  
  /* compute focal length f*/
  f = (sourceDistance * imageDistance)/(sourceDistance + imageDistance);
  /* compute magnification */
  M = imageDistance/sourceDistance;
  if (apertures==0) {
    apertures=1;
    aperture = malloc(sizeof(*aperture));
    initialize_aperture(aperture);
  } 
  /*calculate Ap */
  for (m=0; m<apertures; m++) {
    aperture[m].Ap = aperture[m].amplitude * exp(-I * aperture[m].phase);
  }
  if (spectrumProfile) {
    SDDS_DATASET profileData;
    if (!SDDS_InitializeInput(&profileData, spectrumProfile))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
    if (SDDS_ReadPage( &profileData)<0)
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
    nWaves = SDDS_CountRowsOfInterest( &profileData);
    wavelength = weight = NULL;
    if (!(wavelength = SDDS_GetColumnInDoubles(&profileData, "Wavelength")) ||
        !(weight = SDDS_GetColumnInDoubles(&profileData, "Weight")))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
    if (!SDDS_Terminate(&profileData))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
    for (i=0; i<nWaves; i++)
      fprintf(stderr, "wavelength=%f, weight=%f\n", wavelength[i], weight[i]);
  } else {
    /* Make sure that nWaves is a positive odd number */
    if (nWaves<1)
      nWaves = 1;
    if (nWaves%2==0)
      nWaves +=1;
    if (photonBandwidth>0 && nWaves<3)
      nWaves = 25;
    wavelength = calloc(sizeof(*wavelength), nWaves);
    weight = calloc(sizeof(*weight),  nWaves);
    /* Center wavelength and weight factor */
    wavelength[0] = photonWavelength;
    sum = weight[0] = 1;
    /* Other wavelength and normalized weight factor */
    if (photonBandwidth>0) {
      deltaWave = 7 * photonBandwidth / (nWaves -1);
      wavelength[(nWaves-1)/2] = photonWavelength;
      weight[(nWaves-1)/2] = 1.0;
      fprintf(stderr, "deltaWave=%f\n", deltaWave);
      for (k=(nWaves-1)/2-1;k>=0;k--) {
        wavelength[k]  =  photonWavelength - deltaWave * photonWavelength * ((nWaves-1)/2 - k);
        wavelength[nWaves - 1 - k] =  photonWavelength +  deltaWave * photonWavelength * ((nWaves-1)/2 - k);
        weight[k] =  weight[nWaves-1 - k] = exp(-0.5 * sqr(((nWaves-1)/2 - k) * deltaWave/photonBandwidth));
        sum += 2*weight[k]; 
      }
      /* normalize weight factors and make positive wavelength */
      for (k=0; k<nWaves; k++) {
        /* wavelength[k] = fabs(wavelength[k]); */
        weight[k] = weight[k] / sum;
      }
      /* for (i=0; i<nWaves; i++)
        fprintf(stderr, "i=%d, wavelength=%f, weight=%f\n", i, wavelength[i], weight[i]); */
    }
  }

  /* Show input parameters */
  if (verbose > 0) {
    fprintf(stdout, " photonWavelength = %0.3f (nm), photonEnergy = %0.1f (eV), photonBandwidth = %0.3f, nWaves = %d \n", 
      photonWavelength, photonEnergy, photonBandwidth, nWaves );
    fprintf(stdout, " sourceDistance = %0.2f (m), sourceCenter = %0.3f (mm), sourceWidth = %0.3f (mm)\n", 
      sourceDistance, sourceCenter, sourceWidth);
    fprintf(stdout, " imageDistance  = %0.2f (m), imageCenter  = %0.3f (mm), imageWidth  = %0.3f (mm), nImagePoints = %d \n", 
      imageDistance, imageCenter, imageWidth, nImagePoints);
    fprintf(stdout, " Total apertures = %d\n   Center(mm)  Width(mm)    Amplitude      Phase\n", apertures);
    for (m=0; m<apertures; m++)
      fprintf(stdout, "%10.3f%10.3f%14.3e%14.3e \n", aperture[m].center, aperture[m].width, aperture[m].amplitude, aperture[m].phase);
  }

  /* Save parameters */
  SetupOutputFile(&SDDSout, outputfile);
  if (!SDDS_StartPage(&SDDSout, nImagePoints))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
  if (!SDDS_SetParameters(&SDDSout, SDDS_SET_BY_NAME|SDDS_PASS_BY_VALUE,
                          "SourceDistance", sourceDistance,
                          "SourceWidth", sourceWidth,
                          "PhotonWavelength", photonWavelength,
                          "PhotonEnergy", photonEnergy,
                          "PhotonBandWidth", photonBandwidth,
                          "NWaves", nWaves,
                          "ImageDistance", imageDistance,
                          "ImageWidth", imageWidth,
                          "ImageCenter", imageCenter,
                          "NImagePoints", nImagePoints,
                          "FocalLength", f,
                          "Magnification", M,
                          "NApertures", apertures,
                          NULL))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);

  /* Core loops: multio-wavelength diffraction amplitude and intensity from multiple apertures */
  xSource = malloc(sizeof(*xSource)*nImagePoints);
  
  if (sourceProfile) {
    SDDS_DATASET profileData;
    if (!SDDS_InitializeInput(&profileData, sourceProfile))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
    if (SDDS_ReadPage( &profileData)<0)
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
    nSourcePoints = SDDS_CountRowsOfInterest( &profileData);
    sourceSig0 = xSource0 = NULL;
    if (!(xSource0 = SDDS_GetColumnInDoubles(&profileData, "xSource")) ||
        !(sourceSig0 = SDDS_GetColumnInDoubles(&profileData, "SourceIntensity")))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
    if (!SDDS_Terminate(&profileData))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
    sourceSig = malloc(sizeof(*sourceSig)*nImagePoints);
  } else if (sourceWidth>0) 
    sourceSig = malloc(sizeof(*sourceSig)*nImagePoints);
  
  miny = -(imageWidth/2) + imageCenter;
  stepy = imageWidth/(nImagePoints - 1);
  intensity0 = malloc(sizeof(*intensity0) * nImagePoints);
  for (i=0; i<nImagePoints; i++) {
    /* compute coordinates in source and image plane */
    xImage = miny + stepy * i;
    imageAngle = xImage/imageDistance;
    xSource[i] = xImage/M;
    /*compute Gaussian source profile */
    if (sourceProfile) {
      /*do interpolate to find the source signal */
      sourceSig[i] = interpolate(sourceSig0, xSource0, nSourcePoints, xSource[i], &belowRange, &aboveRange, 1, &interpCode, 1);
    } else {
      if (sourceWidth>0)
        sourceSig[i] = 1 / (sqrt(2*PI) * sourceWidth) * exp(-0.5 * sqr( (xSource[i] - sourceCenter) / sourceWidth ));
    }
    /* compute amplitude at center wavelength */
    Amp0 = 0;
    for (m=0; m<apertures; m++) 
      Amp0 += one_aperture_amplitude(aperture[m], photonWavelength, imageAngle, f);
    /* compute amplitude at other wavelengths and intensity */
    sum = weight[0] * (sqr(creal(Amp0)) + sqr(cimag(Amp0)));
    if (nWaves>1 && (photonBandwidth>0 || spectrumProfile)) {
      for (k=1; k<nWaves; k++) {
        if (verbose && i==0) fprintf(stdout, "k=%ld, wavelength=%f, weight=%f\n", k, wavelength[k], weight[k]);
        Amp1 = 0;
        for (m=0; m<apertures; m++) 
          Amp1 += one_aperture_amplitude(aperture[m], wavelength[k], imageAngle, f);
        sum += weight[k] * (sqr(creal(Amp1)) + sqr(cimag(Amp1)));
      }
    }
    intensity0[i] = sum;
    /* Save data */
    if (!SDDS_SetRowValues(&SDDSout, SDDS_SET_BY_NAME|SDDS_PASS_BY_VALUE, i,
                           "xImage", xImage,
                           "xSource", xSource[i],
                           "ImageIntensity0", intensity0[i],
                           "ImageAmp0Real", creal(Amp0),
                           "ImageAmp0Imag",cimag(Amp0),
                            NULL))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
  }
  free(aperture);
  /*do convolution of intensity with the source to obtain the total intensity */
  if (sourceWidth>0 || sourceProfile) {
    intensity = convolve(intensity0, xSource, sourceSig, xSource, nImagePoints);
    if (!SDDS_SetColumn(&SDDSout, SDDS_SET_BY_NAME|SDDS_PASS_BY_REFERENCE, 
                        intensity, nImagePoints, "ImageIntensity"))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
    free(intensity);
    free(sourceSig);
    if (xSource0) free(xSource0);
    if (sourceSig0) free(sourceSig0);
  } else {
    if (!SDDS_SetColumn(&SDDSout, SDDS_SET_BY_NAME|SDDS_PASS_BY_REFERENCE, 
                        intensity0, nImagePoints, "ImageIntensity"))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
  }
  
  if (!SDDS_WritePage(&SDDSout) || !SDDS_Terminate(&SDDSout))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
  
  free(xSource);
  free(intensity0);
  if (wavelength) free(wavelength);
  if (weight) free(weight);
  free_scanargs(&s_arg, argc);
  return 0;
}

/**
 * @brief Multiply two complex numbers stored as separate real and imaginary parts
 * 
 * Performs complex multiplication: (r1 + i1*i) * (r2 + i2*i) = (r0 + i0*i)
 * where r0 = r1*r2 - i1*i2 and i0 = r1*i2 + i1*r2
 * 
 * @param r0 Pointer to real part of result
 * @param i0 Pointer to imaginary part of result
 * @param r1 Real part of first complex number
 * @param i1 Imaginary part of first complex number
 * @param r2 Real part of second complex number
 * @param i2 Imaginary part of second complex number
 */
void complex_multiply(double *r0, double *i0, double r1, double i1,
                      double r2, double i2)
{
  *r0 = r1 * r2 - i1 * i2;  /* Real part: (a+bi)(c+di) = ac-bd */
  *i0 = r1 * i2 + i1 * r2;  /* Imaginary part: (a+bi)(c+di) = ad+bc */
}

double complex one_aperture_amplitude(APERTURE aperture, double wavelength, double phi, double f) 
{
  double complex E1, E2, E;
  double x1, x2;
  
  x1 = sqrt(2000.0 / (wavelength * f)) * (f * phi - aperture.center + 0.5 * aperture.width);
  x2 = sqrt(2000.0 / (wavelength * f)) * (f * phi - aperture.center - 0.5 * aperture.width);
   
  E1 = fresnel_c(x1) + I * fresnel_s(x1);
  E2 = fresnel_c(x2) + I * fresnel_s(x2);
 
  E = 0.5 * (1 - I) * aperture.Ap * (E1 - E2);
  return E;
}

void SetupOutputFile(SDDS_DATASET *outTable, char *outputFile) {
  if (!SDDS_InitializeOutput(outTable, SDDS_BINARY, 1, NULL, NULL, outputFile))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
  
  if (SDDS_DefineParameter(outTable, "SourceDistance", NULL, "m", NULL,NULL,SDDS_DOUBLE, 0)<0 ||
      SDDS_DefineParameter(outTable, "SourceWidth", NULL, "mm", NULL,NULL,SDDS_DOUBLE, 0)<0 ||
      SDDS_DefineParameter(outTable, "SourceCenter", NULL, "mm", NULL,NULL,SDDS_DOUBLE, 0)<0 ||
      SDDS_DefineParameter(outTable, "PhotonWavelength", NULL, "nm", NULL,NULL,SDDS_DOUBLE, 0)<0 ||
      SDDS_DefineParameter(outTable, "PhotonEnergy", NULL, "eV", NULL,NULL,SDDS_DOUBLE, 0)<0 ||
      SDDS_DefineParameter(outTable, "PhotonBandWidth", NULL, NULL, NULL,NULL,SDDS_DOUBLE, 0)<0 ||
      SDDS_DefineParameter(outTable, "NWaves", NULL, NULL, NULL,NULL,SDDS_LONG, 0)<0 ||
      SDDS_DefineParameter(outTable, "ImageDistance", NULL, "m", NULL,NULL,SDDS_DOUBLE, 0)<0 ||
      SDDS_DefineParameter(outTable, "ImageWidth", NULL, "mm", NULL,NULL,SDDS_DOUBLE, 0)<0 ||
      SDDS_DefineParameter(outTable, "ImageCenter", NULL, "mm", NULL,NULL,SDDS_DOUBLE, 0)<0 ||
      SDDS_DefineParameter(outTable, "NImagePoints", NULL, "nm", NULL,NULL,SDDS_LONG, 0)<0 ||
      SDDS_DefineParameter(outTable, "FocalLength", NULL, "m", NULL,NULL,SDDS_DOUBLE, 0)<0 ||
      SDDS_DefineParameter(outTable, "Magnification", NULL, NULL, NULL,NULL,SDDS_DOUBLE, 0)<0 ||
      SDDS_DefineParameter(outTable, "NApertures", NULL, NULL, NULL,NULL,SDDS_LONG, 0)<0 ||
      SDDS_DefineColumn(outTable, "xImage", NULL, "mm", NULL, NULL, SDDS_DOUBLE, 0)<0 ||
      SDDS_DefineColumn(outTable, "xSource", NULL, "mm", NULL, NULL, SDDS_DOUBLE, 0)<0 ||
      SDDS_DefineColumn(outTable, "ImageIntensity", NULL, NULL, NULL, NULL, SDDS_DOUBLE, 0)<0 ||
      SDDS_DefineColumn(outTable, "ImageIntensity0", NULL, NULL, NULL, NULL, SDDS_DOUBLE, 0)<0 ||
      SDDS_DefineColumn(outTable, "ImageAmp0Real", NULL, NULL, NULL, NULL, SDDS_DOUBLE, 0)<0 ||
      SDDS_DefineColumn(outTable, "ImageAmp0Imag", NULL, NULL, NULL, NULL, SDDS_DOUBLE, 0)<0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
 
  if (!SDDS_WriteLayout(outTable))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
}

void initialize_aperture(APERTURE *aperture)
{
  aperture->center = 0;
  aperture->width = 0.01;
  aperture->amplitude = 1;
  aperture->phase = 0;
}

void ReadInputFile(char *inputFile, APERTURE *aperture, int32_t *apertures, 
                   double *sourceDistance, double *sourceCenter, double *sourceWidth,
                   double *photonWavelength, double *photonEnergy, double *photonBandwidth, int32_t *nWaves,
                   double *imageDistance, double *imageCenter, double *imageWidth, int32_t *nImagePoints)
{
  SDDS_DATASET SDDSin;
  double *center, *width, *transmission, *phase;
  int32_t rows, i;
  
  center = width = transmission = phase = NULL;
  rows = 0;
  
  if (!SDDS_InitializeInput(&SDDSin, inputFile))
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors|SDDS_VERBOSE_PrintErrors);
  
  switch(SDDS_CheckColumn( &SDDSin, "ApertureCenter", NULL, SDDS_DOUBLE, stderr)) {
  case SDDS_CHECK_OKAY:
    break;
  default:
    fprintf( stderr, "Something wrong with column %s in file %s.\n", "ApertureCenter", inputFile);
    exit(1);
  }
  switch(SDDS_CheckColumn( &SDDSin, "ApertureWidth", NULL, SDDS_DOUBLE, stderr)) {
  case SDDS_CHECK_OKAY:
    break;
  default:
    fprintf( stderr, "Something wrong with column %s in file %s.\n", "ApertureWidth", inputFile);
    exit(1);
  }
  switch(SDDS_CheckColumn( &SDDSin, "Transimission", NULL, SDDS_DOUBLE, stderr)) {
  case SDDS_CHECK_OKAY:
    break;
  default:
    fprintf( stderr, "Something wrong with column %s in file %s.\n", "Transimission", inputFile);
    exit(1);
  }
  switch(SDDS_CheckColumn( &SDDSin, "PhaseShift", NULL, SDDS_DOUBLE, stderr)) {
  case SDDS_CHECK_OKAY:
    break;
  default:
    fprintf( stderr, "Something wrong with column %s in file %s.\n", "PhaseShift", inputFile);
    exit(1);
  }
  
  if (SDDS_ReadPage( &SDDSin)<0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
  rows = SDDS_CountRowsOfInterest( &SDDSin);
  
  if (SDDS_CheckParameter(&SDDSin, "SourceDistance", NULL, SDDS_DOUBLE, stderr)==SDDS_CHECK_OKAY) {
    if (!SDDS_GetParameter(&SDDSin, "SourceDistance", sourceDistance))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
  }
  if (SDDS_CheckParameter(&SDDSin, "SourceCenter", NULL, SDDS_DOUBLE, stderr)==SDDS_CHECK_OKAY) {
    if (!SDDS_GetParameter(&SDDSin, "SourceCenter", sourceCenter))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
  }
  if (SDDS_CheckParameter(&SDDSin, "SourceWidth", NULL, SDDS_DOUBLE, stderr)==SDDS_CHECK_OKAY) {
    if (!SDDS_GetParameter(&SDDSin, "SourceWidth", sourceWidth))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
  }
  if (SDDS_CheckParameter(&SDDSin, "PhotonEnergy", NULL, SDDS_DOUBLE, stderr)==SDDS_CHECK_OKAY) {
    if (!SDDS_GetParameter(&SDDSin, "PhotonEnergy", photonEnergy))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
  }
  if (SDDS_CheckParameter(&SDDSin, "PhotonWavelength", NULL, SDDS_DOUBLE, stderr)==SDDS_CHECK_OKAY) {
    if (!SDDS_GetParameter(&SDDSin, "PhotonWavelength", photonWavelength))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
  }
  if (SDDS_CheckParameter(&SDDSin, "PhotonBandWidth", NULL, SDDS_DOUBLE, stderr)==SDDS_CHECK_OKAY) {
    if (!SDDS_GetParameter(&SDDSin, "PhotonBandWidth", photonBandwidth))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
  }
  if (SDDS_CheckParameter(&SDDSin, "ImageDistance", NULL, SDDS_DOUBLE, stderr)==SDDS_CHECK_OKAY) {
    if (!SDDS_GetParameter(&SDDSin, "ImageDistance", imageDistance))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
  }
  if (SDDS_CheckParameter(&SDDSin, "ImageCenter", NULL, SDDS_DOUBLE, stderr)==SDDS_CHECK_OKAY) {
    if (!SDDS_GetParameter(&SDDSin, "ImageCenter", imageCenter))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
  }
  if (SDDS_CheckParameter(&SDDSin, "ImageWidth", NULL, SDDS_DOUBLE, stderr)==SDDS_CHECK_OKAY) {
    if (!SDDS_GetParameter(&SDDSin, "ImageWidth", imageWidth))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
  }
  if (SDDS_CheckParameter(&SDDSin, "NImagePoints", NULL, SDDS_DOUBLE, stderr)==SDDS_CHECK_OKAY) {
    if (!SDDS_GetParameter(&SDDSin, "NImagePoints", nImagePoints))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
  }
  if (!rows)
    return;
  if (!(center = SDDS_GetColumnInDoubles(&SDDSin, "ApertureCenter")) ||
      !(width = SDDS_GetColumnInDoubles(&SDDSin, "ApertureWidth")) ||
      !(transmission = SDDS_GetColumnInDoubles(&SDDSin, "Amplitude")) ||
      !(phase = SDDS_GetColumnInDoubles(&SDDSin, "Phase")))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);

  aperture = SDDS_Realloc(aperture, sizeof(*aperture)*(*apertures+rows));
  for (i=0; i<rows; i++) {
    aperture[*apertures+i].center = center[i];
    aperture[*apertures+i].width = width[i];
    aperture[*apertures+i].phase = phase[i];
    aperture[*apertures+i].amplitude = transmission[i];
  }
  *apertures += rows;
  free(center);
  free(width);
  free(phase);
  free(transmission);
  if (!SDDS_Terminate(&SDDSin))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
}

void wrap_around_order
  (
   double *response1,
   double *t,
   double *response,
   long nres,
   long nsig                       
   )
{
  long i;
  long iz;

  for (iz=0; iz<nres; iz++)
    if (t[iz]>=0)
      break;
  if (iz==nres)
    bomb("response function is acausal", NULL);

  fill_double_array(response1, 2*nsig+2, 0.0L);
  for (i=iz; i<nres; i++)
    response1[i-iz] = response[i];
  for (i=0; i<iz; i++)
    response1[2*nsig-(iz-i)] = response[i];
}

double *convolve(double *signal1, double *indep1, double *signal2, double *indep2, long points)
{
  long i, nfreq, rows1, rows2;
  double range;
  double *fft_sig, *fft_res;
  
  rows1 = rows2 = points;
  
  if (!(fft_sig = SDDS_Calloc(sizeof(*fft_sig), (2*points+2))) ||
      !(fft_res = SDDS_Calloc(sizeof(*fft_res), (2*points+2)))) {
    SDDS_SetError("memory allocation failure");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    exit(1);
  }
  /* return array fft_res with size 2*rows1 + 2 */
  wrap_around_order(fft_res, indep2, signal2, points, points);
  for (i=0; i<points; i++)
    fft_sig[i] = signal1[i];  
  realFFT2(fft_sig, fft_sig, 2*points, 0); 
  realFFT2(fft_res, fft_res, 2*points, 0); 
  nfreq = rows1 + 1;
  range = 2 * rows1 * (indep1[rows1-1] - indep1[0])/ (rows1 - 1);
  for (i=0; i<nfreq; i++) {
    complex_multiply(fft_sig+2*i,  fft_sig+2*i+1,
                     fft_sig[2*i], fft_sig[2*i+1],
                     fft_res[2*i], fft_res[2*i+1]);
  }
  realFFT2(fft_sig, fft_sig, 2*rows1, INVERSE_FFT);
  for (i=0; i<rows1; i++)
    fft_sig[i] *= range;
  free(fft_res);
  return fft_sig;
}
