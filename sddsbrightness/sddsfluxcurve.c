/*************************************************************************\
* Copyright (c) 2009 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2009 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* program: sddsfluxcurve.c
 * purpose: take twiss output from elegant and compute undulator
 *          flux tuning curves
 *
 * Michael Borland, 2009
 * Using code from sddsbrightness (Shang, Dejus, Borland) and sddsurgent (Shang, Dejus)
 *
 $Log: not supported by cvs2svn $
 Revision 1.11  2011/02/17 18:38:21  shang
 changed the default pinhole size to 2.0 instead of 0 to avoid giving 0 insentities when the pinhole sizes are not provided.

 Revision 1.10  2010/10/26 17:32:38  borland
 Fixed bugs in parsing the emittanceRatio parameter.

 Revision 1.9  2010/03/08 03:02:59  borland
 Fixed units in "current" parameter.

 Revision 1.8  2009/05/01 21:31:40  borland
 Default distance is now 100 m.

 Revision 1.7  2009/04/14 13:19:48  borland
 Now accepts units of "m" for ex0.

 Revision 1.6  2009/04/09 22:19:56  borland
 Removed some invalid testing done on arguments.

 Revision 1.5  2009/04/09 22:10:28  borland
 Fixed bug in checking arguments.

 Revision 1.4  2009/04/09 21:55:01  borland
 Added output of total power and on-axis power density.

 Revision 1.3  2009/04/09 16:21:15  borland
 Fixed usage message.

 Revision 1.2  2009/04/09 14:48:26  borland
 Added total flux computation and output.

 Revision 1.1  2009/04/09 14:21:26  borland
 First version in CVS.

 */
#include "mdb.h"
#include "scan.h"
#include "SDDS.h"
#include "sddsbrightness.h"
#include "oagphy.h"

#define SET_PIPE 0
#define SET_HARMONICS 1
#define SET_UNDULATOR 2
#define SET_ELECTRON_BEAM 3
#define SET_METHOD 4
#define SET_PINHOLE 5
#define SET_MODE 6
#define N_OPTIONS 7

char *option[N_OPTIONS] = {
  "pipe", "harmonics", "undulator", "electronbeam", "method", "pinhole",
  "mode",
} ;

char *USAGE1="sddsfluxcurve [-pipe=[input][,output]] [<twissFile>] [<SDDSoutputfile>]\n\
    [-harmonics=<integer>] [-method=<methodName>[,neks=<integer>]]\n\
    [-mode={pinhole|density|total}]\n\
    -undulator=period=<meters>,numberOfPeriods=<integer>{,kmin=<value>,kmax=<value>[,points=<number>]kfilename=<string>,kcolumn=<string>[,{helical|planar}]]}\n\
    [-electronBeam=current=<amps>,[,{coupling=<value> | emittanceRatio=<value>}]]\n\
    [-pinhole=distance=<meters>,xsize=<meters>,ysize=<meters>[,xnumber=<integer>][,ynumber=<integer>][,xposition=<meters>][,yposition=<meters>]]\n\
    [-nowarnings]\n\n\
harmonics        number of harmonics to compute\n\
pinhole          specify the distance to the pinhole, plus the full size of\n\
                 the pinhole horizontal and vertical apertures.\n\
                 Defaults to 50x50 grid centered on 0,0.\n\
method           choose method for calculating flux \n\
                 method=dejus    Non-zero emittance, \n\
                                 infinite-N +convolution (Dejus' approach) (default) \n\
                 method=walkerinfinite        Non-zero emittance, \n\
                                              infinite-N +convolution (Walker's approach) \n\
                 neks=<value> number of points for peaking search. \n\
mode             Choose calculation mode: \n\
                 mode=pinhole         Flux through defined pinhole (default)\n\
                 mode=density         Flux density (includes central-cone flux).\n\
                 mode=total           Total flux.\n\
undulator        specify undulator parameters\n\
electronBeam     specify electron beam parameters that are not in the twiss file.\n\
                 Defaults to 0.1A.\n\
Computes on-axis aperture-limited flux tuning curve for an undulator centered on the\n\
end of the beamline the Twiss parameters for which are in the input file.  You should generate the\n\
input file using elegant's twiss_output command with radiation_integrals=1 .\n\n\
Program by Michael Borland.  (This is version 1.12, "__DATE__")\n";

#define MODE_PINHOLE 0
#define MODE_DENSITY 1
#define MODE_TOTAL   2
#define MODE_OPTIONS 3
static char *mode_option[MODE_OPTIONS] = {
  "pinhole", "density", "total"
  } ;

typedef struct {
  char *kfilename, *kcolumn;
  double *kvalue;
  long nPeriods, nPoints;
  double period, KMin, KMax;
  unsigned long flags;
} UNDULATOR_PARAM;

typedef struct {
  double current, coupling, emittanceRatio;
  double ex0, ey0;
  double energy, gamma;
  double sigmax, sigmay, sigmaxp, sigmayp, energySpread;
  unsigned long flags;
} ELECTRON_BEAM_PARAM;

typedef struct {
  double xPC, yPC, xPS, yPS, distance;
  long nXP, nYP;
  unsigned long flags;
} PINHOLE_PARAM;

#define UNDULATOR_KMIN_GIVEN              0x00001U
#define UNDULATOR_KMAX_GIVEN              0x00002U
#define UNDULATOR_PERIOD_GIVEN            0x00004U
#define UNDULATOR_PERIODS_GIVEN           0x00008U
#define UNDULATOR_POINTS_GIVEN            0x00010U
#define UNDULATOR_KFILENAME_GIVEN         0x00020U
#define UNDULATOR_KCOLUMN_GIVEN           0x00040U
#define UNDULATOR_HELICAL_GIVEN           0x00080U
#define UNDULATOR_PLANAR_GIVEN            0x00100U

#define ELECTRON_CURRENT_GIVEN            0x00080U
#define ELECTRON_COUPLING_GIVEN           0x00100U
#define ELECTRON_ERATIO_GIVEN             0x00200U

#define PINHOLE_DISTANCE_GIVEN            0x00400U
#define PINHOLE_XPC_GIVEN                 0x00800U
#define PINHOLE_YPC_GIVEN                 0x01000U
#define PINHOLE_XPS_GIVEN                 0x02000U
#define PINHOLE_YPS_GIVEN                 0x04000U
#define PINHOLE_NXP_GIVEN                 0x08000U
#define PINHOLE_NYP_GIVEN                 0x10000U

/*fortran subroutine*/
void urgent_();
void us_();
#if defined(_WIN32) && !defined(_MINGW)
#define urgent_() URGENT()
#define us_() US()
#endif

long SetUpOutputFile(SDDS_DATASET *SDDSout, SDDS_DATASET *SDDSin, char *outputfile, long harmonics, long mode);
long GetBeamParameters(ELECTRON_BEAM_PARAM *eBeam, SDDS_DATASET *SDDSin);
void CheckInputParameters(long method, UNDULATOR_PARAM undulator_param, ELECTRON_BEAM_PARAM electron_param, 
                            PINHOLE_PARAM pinhole_param, long harmonics);
void InitializeInputFile(SDDS_DATASET *SDDSin, char *inputfile);
/* FindPeak and Gauss_Convolve are provided by liboagphy (physics/oagphy.h). */

void CalculateFlux(ELECTRON_BEAM_PARAM eBeam,
                   UNDULATOR_PARAM undulator,
                   PINHOLE_PARAM pinhole,
                   long ihMin, long ihMax, long ihStep,
                   long neks,
                   long method, long mode,
                   double **K, double **TotalPower, double **OnAxisPowerDensity, double ***FnOut,
                   double ***Energy, double ***Flux, double ***LambdarOut);
void UndulatorCentralConeFlux(ELECTRON_BEAM_PARAM eBeam, UNDULATOR_PARAM undulator, int h, double *K, double *CentralConeFlux);
void getKValueDataFromFile(UNDULATOR_PARAM *undulator_param);

int main(int argc, char **argv)
{
  SDDS_DATASET SDDSin, SDDSout;
  char *inputfile, *outputfile;
  SCANNED_ARG *s_arg;
  unsigned long pipeFlags;
  long harmonics, tmpFileUsed, i_arg, readCode, h, ih;
  unsigned long dummyFlags;
  long method, nE, ihMin, ihMax, mode = 4;
  double *KK, **FnOut, **Energy, **Flux, **LambdarOut, *TotalPower, *OnAxisPowerDensity, *CentralConeFlux;
  int32_t neks;
  PINHOLE_PARAM pinhole_param;
  UNDULATOR_PARAM undulator_param;
  ELECTRON_BEAM_PARAM electron_param;

  SDDS_RegisterProgramName(argv[0]);
  argc = scanargs(&s_arg, argc, argv);
  if (argc<2) {
    fprintf(stderr, "%s\n", USAGE1);
    exit(1);
  }

  KK = TotalPower = OnAxisPowerDensity = CentralConeFlux = NULL;
  FnOut = Energy = Flux = LambdarOut = NULL;

  inputfile = outputfile = NULL;
  pipeFlags = dummyFlags=0;
  harmonics = 0;
  method = 4;
  neks = 100;

  pinhole_param.flags = 0;
  pinhole_param.distance = 30;
  pinhole_param.xPC = pinhole_param.yPC = 0;
  pinhole_param.xPS = pinhole_param.yPS = 2.0;
  pinhole_param.nXP = pinhole_param.nYP = 20;

  electron_param.flags = 0;
  electron_param.current = 0.1;
  electron_param.coupling = electron_param.emittanceRatio = 0;
  
  undulator_param.flags = 0;
  undulator_param.nPeriods = 0;
  undulator_param.nPoints = 100;
  undulator_param.period = undulator_param.KMin = undulator_param.KMax = 0;
  undulator_param.kfilename = undulator_param.kcolumn = NULL;
  undulator_param.kvalue = NULL;
  
  for (i_arg=1; i_arg<argc; i_arg++) {
    if (s_arg[i_arg].arg_type==OPTION) {
      switch (match_string(s_arg[i_arg].list[0], option, N_OPTIONS, 0)) {
      case SET_PIPE:
        if (!processPipeOption(s_arg[i_arg].list+1, s_arg[i_arg].n_items-1, &pipeFlags))
          SDDS_Bomb("invalid -pipe syntax");
        break;
      case SET_HARMONICS:
        if (s_arg[i_arg].n_items!=2 ||
            !sscanf(s_arg[i_arg].list[1], "%ld", &harmonics) ||
            harmonics<1)
          SDDS_Bomb("invalid -harmonics value");
        break;
      case SET_UNDULATOR:
        s_arg[i_arg].n_items--;
        if (!scanItemList(&undulator_param.flags, s_arg[i_arg].list+1, &s_arg[i_arg].n_items, 0,
                          "period", SDDS_DOUBLE, &undulator_param.period, 1, UNDULATOR_PERIOD_GIVEN,
                          "kmin", SDDS_DOUBLE, &undulator_param.KMin, 1, UNDULATOR_KMIN_GIVEN,
                          "kmax", SDDS_DOUBLE, &undulator_param.KMax, 1, UNDULATOR_KMAX_GIVEN,
                          "numberofperiods", SDDS_LONG, &undulator_param.nPeriods, 1, UNDULATOR_PERIODS_GIVEN,
                          "points", SDDS_LONG, &undulator_param.nPoints, 1, UNDULATOR_POINTS_GIVEN,
			  "kfilename", SDDS_STRING, &undulator_param.kfilename, 1, UNDULATOR_KFILENAME_GIVEN,
			  "kcolumn", SDDS_STRING, &undulator_param.kcolumn, 1, UNDULATOR_KCOLUMN_GIVEN,
                          "helical", -1, NULL, 0, UNDULATOR_HELICAL_GIVEN,
                          "planar", -1, NULL, 0, UNDULATOR_PLANAR_GIVEN,
                          NULL) ||
            undulator_param.period<=0 || undulator_param.nPeriods<10)
          SDDS_Bomb("invalid -undulator parameters/values");
	if (!(undulator_param.flags&UNDULATOR_KFILENAME_GIVEN)) {
	  if (undulator_param.KMin<=0 || undulator_param.KMax<undulator_param.KMax ||
	      undulator_param.nPoints<1)
	    SDDS_Bomb("invalid -undulator syntax: invalid k values provided.");
	} else {
	  if (!(undulator_param.flags&UNDULATOR_KCOLUMN_GIVEN))
	    SDDS_Bomb("invalid -undulator syntax: kcolumn provided.");
	  getKValueDataFromFile(&undulator_param);
	}
        if (undulator_param.flags&UNDULATOR_HELICAL_GIVEN &&
            undulator_param.flags&UNDULATOR_PLANAR_GIVEN)
          SDDS_Bomb("give helical or planar, not both");
        break;
      case SET_PINHOLE:
        if (s_arg[i_arg].n_items<2)
          SDDS_Bomb("invalid -pinhole syntax.");
        s_arg[i_arg].n_items--;
        if (!scanItemList(&pinhole_param.flags, s_arg[i_arg].list+1, &s_arg[i_arg].n_items, 0,
                          "distance", SDDS_DOUBLE, &pinhole_param.distance, 1, PINHOLE_DISTANCE_GIVEN,
                          "xposition", SDDS_DOUBLE, &pinhole_param.xPC, 1, PINHOLE_XPC_GIVEN,
                          "yposition", SDDS_DOUBLE, &pinhole_param.yPC, 1, PINHOLE_YPC_GIVEN,
                          "xsize", SDDS_DOUBLE, &pinhole_param.xPS, 1, PINHOLE_XPS_GIVEN,
                          "ysize", SDDS_DOUBLE, &pinhole_param.yPS, 1, PINHOLE_YPS_GIVEN,
                          "xnumber", SDDS_LONG, &pinhole_param.nXP, 1, PINHOLE_NXP_GIVEN,
                          "ynumber", SDDS_LONG, &pinhole_param.nYP, 1, PINHOLE_NYP_GIVEN,
                          NULL))
          SDDS_Bomb("invalid -pinhole syntax.");
        if (pinhole_param.nXP>500 || pinhole_param.nYP>500) 
          SDDS_Bomb("Number of intervals of x/y acceptance (nXP/nYP) should be between 0 and 500");
        /* convert units to mm */
        pinhole_param.xPC *= 1e3;
        pinhole_param.xPS *= 1e3;
        pinhole_param.yPC *= 1e3;
        pinhole_param.yPS *= 1e3;
        s_arg[i_arg].n_items++;
        break;
      case SET_ELECTRON_BEAM:
        if (s_arg[i_arg].n_items<2)
          SDDS_Bomb("invalid -electronBeam syntax.");
        s_arg[i_arg].n_items--;
        if (!scanItemList(&electron_param.flags, s_arg[i_arg].list+1, &s_arg[i_arg].n_items, 0,
                          "current", SDDS_DOUBLE, &electron_param.current, 1, ELECTRON_CURRENT_GIVEN,
                          "coupling", SDDS_DOUBLE, &electron_param.coupling, 1, ELECTRON_COUPLING_GIVEN,
                          "emittanceratio", SDDS_DOUBLE, &electron_param.emittanceRatio, 1, ELECTRON_ERATIO_GIVEN,
                          NULL))
          SDDS_Bomb("invalid -electronBeam syntax.");
        if (electron_param.flags&ELECTRON_COUPLING_GIVEN && electron_param.flags&ELECTRON_ERATIO_GIVEN)
          SDDS_Bomb("give only one of coupling and emittanceRatio to -electronBeam option");
        s_arg[i_arg].n_items++;
        break;
      case SET_MODE:
        if (s_arg[i_arg].n_items<2) 
          SDDS_Bomb("invalid -mode syntax/values");
        switch (match_string(s_arg[i_arg].list[1], mode_option, MODE_OPTIONS, 0)) {
        case MODE_PINHOLE:
          mode = 4;
          break;
        case MODE_DENSITY:
          mode = 2;
          break;
        case MODE_TOTAL:
          mode = 5;
          break;
        default:
          SDDS_Bomb("Invalid mode given (use pinhole or density)");
          break;
        }
        break;
      case SET_METHOD:
        if (s_arg[i_arg].n_items<2) 
          SDDS_Bomb("invalid -method syntax/values");
        switch (match_string(s_arg[i_arg].list[1], method_option, METHOD_OPTIONS, 0)) {
        case DEJUS:
          method = 4;
          break;
        case WALKER_INF:
          method = 14;
          break;
        default:
          SDDS_Bomb("Invalid method given (use dejus or walkerinfinite)");
          break;
        }
        s_arg[i_arg].n_items -=2;
        if (s_arg[i_arg].n_items>0 &&
            !scanItemList(&dummyFlags, s_arg[i_arg].list+2, &s_arg[i_arg].n_items, 0,
                          "neks", SDDS_LONG, &neks, 1, 0, NULL,
                          NULL))
          SDDS_Bomb("invalid -method syntax/values");
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
#ifdef DEBUG
  fprintf(stderr, "Argument parsing done.\n");
#endif

  CheckInputParameters(method, undulator_param, electron_param, pinhole_param, harmonics);
  
  processFilenames("sddsfluxcurve", &inputfile, &outputfile, pipeFlags, 0, &tmpFileUsed);
  if (tmpFileUsed)
    SDDS_Bomb("can't overwrite input file");
  
#ifdef DEBUG
  fprintf(stderr, "Checking input file...\n");
#endif

  InitializeInputFile(&SDDSin, inputfile);
  
#ifdef DEBUG
  fprintf(stderr, "Setting up output file...\n");
#endif
  if (!SetUpOutputFile(&SDDSout, &SDDSin, outputfile, harmonics, mode))
    SDDS_Bomb("problem setting up output file");

#ifdef DEBUG
  fprintf(stderr, "Entering main loop\n");
#endif
  while ((readCode=SDDS_ReadPage(&SDDSin))>0) {
    if (!GetBeamParameters(&electron_param, &SDDSin))
      SDDS_Bomb("problem getting twiss parameters and other values from input file");
#ifdef DEBUG
    fprintf(stderr, "Twiss values read in and beam parameters computed.\n");
#endif

    if (!SDDS_StartPage(&SDDSout, undulator_param.nPoints))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
#ifdef DEBUG
    fprintf(stderr, "Started output page.\n");
#endif
    ihMin=1;
    ihMax=2*(harmonics-1)+1;
    nE=undulator_param.nPoints;
    
#ifdef DEBUG
    fprintf(stderr, "Calling CalculateFlux\n");
#endif
    electron_param.current *= 1e3;  /* want mA */
    CalculateFlux(electron_param, undulator_param, pinhole_param,
                  ihMin, ihMax, 2, 
                  neks,
                  method, mode,
                  &KK, &TotalPower, &OnAxisPowerDensity,
                  &FnOut, &Energy, &Flux, &LambdarOut);
    electron_param.current /= 1e3; 
    if (mode==2) {
      CentralConeFlux = tmalloc(sizeof(double)*nE);
    }
    
#ifdef DEBUG
    fprintf(stderr, "Returned from CalculateFlux\n");
#endif
    for (ih=0; ih<harmonics; ih++) {
      h = ih*2+1;
      if (mode==2)
        UndulatorCentralConeFlux(electron_param, undulator_param, h, KK, CentralConeFlux);
      if (h==1 && 
          (!SDDS_SetColumn(&SDDSout, SDDS_SET_BY_INDEX, KK, nE, 0) ||
           !SDDS_SetColumn(&SDDSout, SDDS_SET_BY_INDEX, TotalPower, nE, 1) ||
           !SDDS_SetColumn(&SDDSout, SDDS_SET_BY_INDEX, OnAxisPowerDensity, nE, 2)))
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
      if (mode==2) {
        if (!SDDS_SetColumn(&SDDSout, SDDS_SET_BY_INDEX, Flux[ih], nE, ih*5+3) ||
            !SDDS_SetColumn(&SDDSout, SDDS_SET_BY_INDEX, CentralConeFlux, nE, ih*5+4) ||
            !SDDS_SetColumn(&SDDSout, SDDS_SET_BY_INDEX, FnOut[ih], nE, ih*5+5) ||
            !SDDS_SetColumn(&SDDSout, SDDS_SET_BY_INDEX, LambdarOut[ih], nE, ih*5+6) ||
            !SDDS_SetColumn(&SDDSout, SDDS_SET_BY_INDEX, Energy[ih], nE, ih*5+7))
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
      } else {
        if (!SDDS_SetColumn(&SDDSout, SDDS_SET_BY_INDEX, Flux[ih], nE, ih*4+3) ||
            !SDDS_SetColumn(&SDDSout, SDDS_SET_BY_INDEX, FnOut[ih], nE, ih*4+4) ||
            !SDDS_SetColumn(&SDDSout, SDDS_SET_BY_INDEX, LambdarOut[ih], nE, ih*4+5) ||
            !SDDS_SetColumn(&SDDSout, SDDS_SET_BY_INDEX, Energy[ih], nE, ih*4+6))
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
      }
    }
    if (!SDDS_SetParameters(&SDDSout, SDDS_BY_NAME|SDDS_PASS_BY_VALUE, "current", electron_param.current, 
                            "EnergySpread", electron_param.energySpread, 
                            "sigmax", electron_param.sigmax, "sigmay", electron_param.sigmay, 
                            "sigmaxprime", electron_param.sigmaxp, "sigmayprime", electron_param.sigmayp, 
                            "period", undulator_param.period, "numberOfPeriods", undulator_param.nPeriods, 
                            "emitx", electron_param.ex0, "emity", electron_param.ey0, NULL))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
    
#ifdef DEBUG
    fprintf(stderr, "Writing output page.\n");
#endif
    if (!SDDS_WritePage(&SDDSout))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
    
#ifdef DEBUG
    fprintf(stderr, "Exiting main loop.\n");
#endif
  }
  
  if (!SDDS_Terminate(&SDDSin) || !SDDS_Terminate(&SDDSout))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
  if (undulator_param.kfilename) free(undulator_param.kfilename);
  if (undulator_param.kcolumn) free(undulator_param.kcolumn);
  if (undulator_param.kvalue) free(undulator_param.kvalue);
  free_scanargs(&s_arg,argc);
  if (method) {
    for (ih=0;ih<harmonics;ih++) {
      free(Energy[ih]);
      free(FnOut[ih]);
      free(Flux[ih]);
      free(LambdarOut[ih]);
    }
    free(Energy);
    free(FnOut);
    free(Flux);
    free(LambdarOut);
    free(KK);
    free(TotalPower);
    free(OnAxisPowerDensity);
    free(CentralConeFlux);
  }
  return 0;
}

long SetUpOutputFile(SDDS_DATASET *SDDSout, SDDS_DATASET *SDDSin, char *outputfile, long harmonics, long mode)
{
  long h;
  char buffer[1024];
  
  if (!SDDS_InitializeOutput(SDDSout, SDDS_BINARY, 1, NULL, NULL, outputfile) ||
      !SDDS_DefineSimpleColumn(SDDSout, "K", NULL, SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(SDDSout, "TotalPower", "W", SDDS_DOUBLE) ||
      !SDDS_DefineSimpleColumn(SDDSout, "OnAxisPowerDensity", "W/mrad$a2$n", SDDS_DOUBLE))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
  if (SDDS_DefineParameter(SDDSout,"current",NULL, "A", NULL,NULL,SDDS_DOUBLE, 0)<0 ||
      SDDS_DefineParameter(SDDSout,"EnergySpread",NULL, NULL, NULL,NULL,SDDS_DOUBLE, 0)<0 ||
      SDDS_DefineParameter(SDDSout,"sigmax",NULL, "mm", NULL,NULL,SDDS_DOUBLE, 0)<0 ||
      SDDS_DefineParameter(SDDSout,"sigmay",NULL, "mm", NULL,NULL,SDDS_DOUBLE, 0)<0 ||
      SDDS_DefineParameter(SDDSout,"sigmaxprime",NULL,"mrad", NULL,NULL,SDDS_DOUBLE, 0)<0 ||
      SDDS_DefineParameter(SDDSout,"sigmayprime",NULL,"mrad", NULL,NULL,SDDS_DOUBLE, 0)<0 ||
      SDDS_DefineParameter(SDDSout,"period",NULL, "m", NULL,NULL,SDDS_DOUBLE, 0)<0 ||
      SDDS_DefineParameter(SDDSout,"numberOfPeriods",NULL, NULL, NULL,NULL,SDDS_LONG, 0)<0 ||
      SDDS_DefineParameter(SDDSout,"emitx","m",NULL,NULL,NULL,SDDS_DOUBLE, 0)<0 ||
      SDDS_DefineParameter(SDDSout,"emity","m",NULL,NULL,NULL,SDDS_DOUBLE, 0)<0 )
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
  for (h=1; h<2*harmonics; h+=2) {
    switch (mode) {
    case 4:
      /* pinhole flux */
      sprintf(buffer, "PinholeFlux%ld", h);
      if (!SDDS_DefineSimpleColumn(SDDSout, buffer, "photons/s/0.1%BW", SDDS_DOUBLE))
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
      break;
    case 2:
      /* flux density */
      sprintf(buffer, "FluxDensity%ld", h);
      if (!SDDS_DefineSimpleColumn(SDDSout, buffer, "photons/s/mrad$a2$n/0.1%BW", SDDS_DOUBLE))
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
      sprintf(buffer, "CentralConeFlux%ld", h);
      if (!SDDS_DefineSimpleColumn(SDDSout, buffer, "photons/s/0.1%BW", SDDS_DOUBLE))
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
      break;
    case 5:
      /* total flux */
      sprintf(buffer, "TotalFlux%ld", h);
      if (!SDDS_DefineSimpleColumn(SDDSout, buffer, "photons/s/0.1%BW", SDDS_DOUBLE))
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
      break;
    default:
      SDDS_Bomb("Invalid mode seen (SetUpOutputFile)");
      break;
    }
    sprintf(buffer, "F%ld", h);
    if (!SDDS_DefineSimpleColumn(SDDSout, buffer, NULL, SDDS_DOUBLE))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
    sprintf(buffer, "wavelength%ld", h);
    if (!SDDS_DefineSimpleColumn(SDDSout, buffer, "m", SDDS_DOUBLE))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
    sprintf(buffer, "photonEnergy%ld", h);
    if (!SDDS_DefineSimpleColumn(SDDSout, buffer, "keV", SDDS_DOUBLE))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
  }
  if (!SDDS_WriteLayout(SDDSout))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
  return 1;
}

long GetBeamParameters(ELECTRON_BEAM_PARAM *eBeam, SDDS_DATASET *SDDSin)
{
  double *data;
  long rows, ey0Exist=0;
  double betax, betay, alphax, alphay, etax, etaxp, etay, etayp, gammax, gammay;
  double ex0, pCentral, Sdelta0, ey0;
  
  if (!(rows=SDDS_RowCount(SDDSin)))
    return 0;

  if (!(data=SDDS_GetColumnInDoubles(SDDSin, "betax")))
    SDDS_Bomb("unable to get betax");
  betax = data[rows-1];
  free(data);

  if (!(data=SDDS_GetColumnInDoubles(SDDSin, "alphax")))
    SDDS_Bomb("unable to get alphax");
  alphax = data[rows-1];
  free(data);
  
  if (!(data=SDDS_GetColumnInDoubles(SDDSin, "etax")))
    SDDS_Bomb("unable to get etax");
  etax = data[rows-1];
  free(data);

  if (!(data=SDDS_GetColumnInDoubles(SDDSin, "etaxp")))
    SDDS_Bomb("unable to get etax");
  etaxp = data[rows-1];
  free(data);

  if (!(data=SDDS_GetColumnInDoubles(SDDSin, "betay")))
    SDDS_Bomb("unable to get betay");
  betay = data[rows-1];
  free(data);

  if (!(data=SDDS_GetColumnInDoubles(SDDSin, "alphay")))
    SDDS_Bomb("unable to get alphay");
  alphay = data[rows-1];
  free(data);

  if (!(data=SDDS_GetColumnInDoubles(SDDSin, "etay")))
    SDDS_Bomb("unable to get etay");
  etay = data[rows-1];
  free(data);

  if (!(data=SDDS_GetColumnInDoubles(SDDSin, "etayp")))
    SDDS_Bomb("unable to get etay");
  etayp = data[rows-1];
  free(data);

  if (SDDS_CheckParameter(SDDSin, "ex0", "$gp$rm", SDDS_ANY_FLOATING_TYPE, NULL)==SDDS_CHECK_OK ||
      SDDS_CheckParameter(SDDSin, "ex0", "m", SDDS_ANY_FLOATING_TYPE, NULL)==SDDS_CHECK_OK) {
    if (!SDDS_GetParameterAsDouble(SDDSin, "ex0", &ex0))
      SDDS_Bomb("unable to get ex0 parameter from input file");
  } else {
    if (!(data=SDDS_GetColumnInDoubles(SDDSin, "ex")))
      SDDS_Bomb("unable to get ex");
    ex0 = data[0];
    free(data);
  }
  if (SDDS_CheckParameter(SDDSin, "pCentral", NULL, SDDS_ANY_FLOATING_TYPE, NULL)==SDDS_CHECK_OK) {
    if (!SDDS_GetParameterAsDouble(SDDSin, "pCentral", &pCentral))
      SDDS_Bomb("unable to get pCentral parameter from input file");
  } else {
    if (!(data=SDDS_GetColumnInDoubles(SDDSin, "pCentral")))
      SDDS_Bomb("unable to get pCentral");
    pCentral = data[0];
    free(data);
  }
  if (SDDS_CheckParameter(SDDSin, "Sdelta0", "", SDDS_ANY_FLOATING_TYPE, NULL)==SDDS_CHECK_OK) {
    if (!SDDS_GetParameterAsDouble(SDDSin, "Sdelta0", &Sdelta0))
      SDDS_Bomb("unable to get Sdelta0 parameter from input file");
  } else {
    if (!(data=SDDS_GetColumnInDoubles(SDDSin, "Sdelta")))
      SDDS_Bomb("unable to get pCentral");
    Sdelta0 = data[0];
    free(data);
  }
  /* get ey0 if it is there */
  if (SDDS_GetParameterAsDouble(SDDSin, "ey0", &ey0)) {
    ey0Exist=1;
  } else if ((data=SDDS_GetColumnInDoubles(SDDSin, "ey"))) {
    ey0Exist=1;
    ey0 = data[0];
    free(data);
  }

  if (ex0<=0.0)
    SDDS_Bomb("ex0 should be greater than zero.");
  if (!ey0Exist) {
    if (eBeam->emittanceRatio==0 && eBeam->coupling==0)
      SDDS_Bomb("No vertical emittance data in file: give emittanceRatio or coupling using -electronBeam");
    if (eBeam->coupling) {
      ex0 = ex0/(1+eBeam->coupling);
      ey0 = eBeam->coupling*ex0;
    } else {
      ey0 = ex0*eBeam->emittanceRatio;
    }
  }

  gammax = (1 + sqr(alphax))/betax;
  gammay = (1 + sqr(alphay))/betay;

  eBeam->gamma   = sqrt(sqr(pCentral)+1);
  eBeam->energy  = pCentral*me_mev/1e3;  /* in GeV */

  /* Beam sizes in mm and mrad */
  eBeam->sigmax  = sqrt(ex0*betax  + sqr(Sdelta0*etax))*1e3;
  eBeam->sigmaxp = sqrt(ex0*gammax + sqr(Sdelta0*etaxp))*1e3;
  eBeam->sigmay  = sqrt(ey0*betay  + sqr(Sdelta0*etay))*1e3;
  eBeam->sigmayp = sqrt(ey0*gammay + sqr(Sdelta0*etayp))*1e3;

  eBeam->energySpread = Sdelta0;

  eBeam->ex0 = ex0;
  eBeam->ey0 = ey0;
  
  return 1;
}

void CheckInputParameters(long method, UNDULATOR_PARAM undulator_param, ELECTRON_BEAM_PARAM electron_param, 
                            PINHOLE_PARAM pinhole_param, long harmonics)
{
  if (harmonics<=0)
    SDDS_Bomb("give -harmonics option");
  
  if (undulator_param.flags==0)
    SDDS_Bomb("give -undulator option");

  /* Should be more thorough here. */
}

void InitializeInputFile(SDDS_DATASET *SDDSin, char *inputfile)
{
  char *Units = NULL;
  
  if (!SDDS_InitializeInput(SDDSin, inputfile))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);

  /* check the input file for valid data */
  if (SDDS_CheckColumn(SDDSin, "betax", "m", SDDS_ANY_FLOATING_TYPE, stderr)!=SDDS_CHECK_OK || 
      SDDS_CheckColumn(SDDSin, "alphax", "", SDDS_ANY_FLOATING_TYPE, stderr)!=SDDS_CHECK_OK || 
      SDDS_CheckColumn(SDDSin, "etax", "m", SDDS_ANY_FLOATING_TYPE, stderr)!=SDDS_CHECK_OK ||
      SDDS_CheckColumn(SDDSin, "etaxp", "", SDDS_ANY_FLOATING_TYPE, stderr)!=SDDS_CHECK_OK ||
      SDDS_CheckColumn(SDDSin, "betay", "m", SDDS_ANY_FLOATING_TYPE, stderr)!=SDDS_CHECK_OK || 
      SDDS_CheckColumn(SDDSin, "alphay", "", SDDS_ANY_FLOATING_TYPE, stderr)!=SDDS_CHECK_OK ||
      SDDS_CheckColumn(SDDSin, "etay", "m", SDDS_ANY_FLOATING_TYPE, stderr)!=SDDS_CHECK_OK ||
      SDDS_CheckColumn(SDDSin, "etayp", "", SDDS_ANY_FLOATING_TYPE, stderr)!=SDDS_CHECK_OK ) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
    SDDS_Bomb("something wrong with twiss parameter columns");
  }
  if (SDDS_CheckParameter(SDDSin, "ex0", "$gp$rm", SDDS_ANY_FLOATING_TYPE, NULL)!=SDDS_CHECK_OK &&
      SDDS_CheckParameter(SDDSin, "ex0", "m", SDDS_ANY_FLOATING_TYPE, NULL)!=SDDS_CHECK_OK) {
    if (SDDS_CheckColumn(SDDSin,"ex", "$gp$rm", SDDS_ANY_FLOATING_TYPE, NULL)!=SDDS_CHECK_OK &&
        SDDS_CheckColumn(SDDSin,"ex", "m", SDDS_ANY_FLOATING_TYPE, NULL)!=SDDS_CHECK_OK) {
      SDDS_Bomb("Something wrong with both ex0 parameter and ex column, one of them has to exist");
    }
  }
  if (SDDS_CheckParameter(SDDSin, "pCentral", NULL, SDDS_ANY_FLOATING_TYPE, NULL)==SDDS_CHECK_OK) {
    if (!SDDS_GetParameterInformation(SDDSin, "units", &Units, SDDS_GET_BY_NAME, "pCentral"))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
    if (Units && !is_blank(Units) && strcmp(Units, "m$be$nc"))
      SDDS_Bomb("Invalid units of pCentral parameter");
  } else if (SDDS_CheckColumn(SDDSin, "pCentral", NULL, SDDS_ANY_FLOATING_TYPE, NULL)==SDDS_CHECK_OK) {
    if (!SDDS_GetColumnInformation(SDDSin, "units", &Units, SDDS_GET_BY_NAME, "pCentral"))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
    if (Units && !is_blank(Units) && strcmp(Units, "m$be$nc"))
      SDDS_Bomb("Invalid units of pCentral column");
  } else 
    SDDS_Bomb("Something wrong with both pCentral parameter and pCentral column, one of them has to exist");
  if (SDDS_CheckParameter(SDDSin, "Sdelta0", "", SDDS_ANY_FLOATING_TYPE, NULL)!=SDDS_CHECK_OK &&
      SDDS_CheckColumn(SDDSin, "Sdelta", "", SDDS_ANY_FLOATING_TYPE, NULL)!=SDDS_CHECK_OK) {
    SDDS_Bomb("Something wrong with both Sdelta0 parameter and Sdelta column, one of them has to exist");
  }
}


void CalculateFlux(ELECTRON_BEAM_PARAM eBeam,
                   UNDULATOR_PARAM undulator,
                   PINHOLE_PARAM pinhole,
                   long ihMin, long ihMax, long ihStep,
                   long neks,
                   long method, long mode,
                   double **K,  double **TotalPower, double **OnAxisPowerDensity, 
                   double ***FnOut, double ***Energy, double ***Flux, double ***LambdarOut)
{
  double lambdar, reducedE, kx, ky, eMin, eMax, ekMin, ekMax, ep, sp, dep1, dep2, fc, fc2, de, smax;
  long ih, i, j, je;
  long nSigma=3, nppSigma=6, nek, exitLoop=0;
  double JArg, sigmaE, sigmaEE, dek, ek;
  double **ei, *kyb, **eb, **sb;
  double e, k, period;
  long nPhi, nAlpha, nOmega, points, nE;
  double dAlpha, dOmega;
  long nXP0, nYP0;
  double minF = 1e4;
  double densityFactor = 1;
  double lambda1, E1, ptot, pd, *EE, *xPMM, *yPMM, *irradiance;
  double *L1, *L2, *L3, *L4, *spec1;
  double ptot1, ftot;
  long isub, iang, min_harmonic=0, max_harmonic=0, nEE;
  double Kmin, Kmax;
  static double *Kvalue = NULL;
  
#ifdef DEBUG
  fprintf(stderr, "In CalculateFlux\n");
#endif

  nPhi = 20;
  nAlpha=15;
  dAlpha=2.0;
  nOmega=16;
  dOmega=2.0;
  sigmaE = eBeam.energySpread;
  if (mode==2)
    densityFactor = sqr(pinhole.distance);
  
  ei = eb = sb = NULL;
  period = undulator.period*1.0e2; /*use cm as units */

  if (*K) free(*K);
  if (*TotalPower) free(*TotalPower);
  if (*OnAxisPowerDensity) free(*OnAxisPowerDensity);
  if (*FnOut) free(*FnOut);
  if (*Energy) free(*Energy);
  if (*Flux) free(*Flux);
  if (*LambdarOut) free(*LambdarOut);
  *K = *TotalPower = *OnAxisPowerDensity = NULL;
  *FnOut = *Energy = *Flux = *LambdarOut = NULL;

  nE = undulator.nPoints;

#ifdef DEBUG
  fprintf(stderr, "Allocating memory...\n");
#endif

  kyb = (double*)calloc(nE+1, sizeof(*kyb));
 
  ei = (double**)malloc(sizeof(*ei)*(MAXIMUM_E+1));
  eb = (double**)malloc(sizeof(*eb)*(MAXIMUM_E+1));
  sb = (double**)malloc(sizeof(*sb)*(MAXIMUM_E+1));
  
  for (i = 0;i<(MAXIMUM_E+1);i++) {
    ei[i] = (double*)calloc(MAXIMUM_H, sizeof(**ei));
    eb[i] = (double*)calloc(MAXIMUM_H, sizeof(**eb));
    sb[i] = (double*)calloc(MAXIMUM_H, sizeof(**sb));
  }
  
  if (MAXIMUM_E>=(pinhole.nXP+1)*(pinhole.nYP+1))
    points = MAXIMUM_E + 100;
  else
    points=(pinhole.nXP+1)*(pinhole.nYP+1)+1;

  EE=(double*)malloc(sizeof(*EE)*points);
  xPMM=(double*)malloc(sizeof(*xPMM)*points); 
  yPMM=(double*)malloc(sizeof(*yPMM)*points); 
  irradiance=(double*)malloc(sizeof(*irradiance)*points);
  L1=(double*)malloc(sizeof(*L1)*points);
  L2=(double*)malloc(sizeof(*L2)*points);
  L3=(double*)malloc(sizeof(*L3)*points);
  L4=(double*)malloc(sizeof(*L4)*points);
  spec1=(double*)malloc(sizeof(*spec1)*points);
  *TotalPower = (double*)calloc(nE, sizeof(**TotalPower));
  *OnAxisPowerDensity = (double*)calloc(nE, sizeof(**OnAxisPowerDensity));
  
#ifdef DEBUG
  fprintf(stderr, "Done\n");
#endif

  lambdar = period*1.0E8/(2.0*sqr(eBeam.gamma)); /*reduced wavelength A */
  reducedE = c_evang/lambdar; /*reduced energy ev */
  kx = 0.0;
  Kmin = undulator.KMin;
  Kmax = undulator.KMax;
  Kvalue = undulator.kvalue;
  if (undulator.flags&UNDULATOR_HELICAL_GIVEN) {
    Kmin = sqrt(2)*Kmin;
    Kmax = sqrt(2)*Kmax;
    Kvalue = SDDS_Realloc(Kvalue, sizeof(*Kvalue)*undulator.nPoints);
    if (undulator.flags&UNDULATOR_KFILENAME_GIVEN) 
      for (i=0; i<undulator.nPoints; i++)
        Kvalue[i] = undulator.kvalue[i]*sqrt(2);
  }
  eMin = reducedE/(1+sqr(Kmax)/2.0);
  eMax = reducedE/(1+sqr(Kmin)/2.0);
#ifdef DEBUG
  fprintf(stderr, "eMin = %e eV, eMax = %e eV, lambda = %e A\n", eMin, eMax, lambdar);
#endif

  /* nXP and nYP are changed by US and urgent program
     the return value of nXP = nXP + 1, nYP = nYP +1
     so nXP, nYp is changing when page number increasing
     nXP0, nYP0 are to remember the input value of nXP and nYP
   */
  nXP0 = pinhole.nXP;
  nYP0 = pinhole.nYP;

  /* Compute peak shifts at minimum K */
  ky = Kmin;

  /* First we figure out the offsets, dep1 and dep2, between the ideal
   * position of the flux peak vs. photon energy and the actual position.
   * It differs for even and odd harmonics
   */
  dep1 = dep2 = 0;
  for (i = 1;i<3;i++) {
    if (i == 1) {
      ekMin = 0.95*i*eMax;
      ekMax = 1.01*i*eMax;
    } else {
      ekMin = 0.820*i*eMax;
      ekMax = 1.002*i*eMax;
    }
#ifdef DEBUG
    fprintf(stderr, "Computing initial spectrum for i=%ld, ek: [%e, %e]\n",
            i, ekMin, ekMax);
#endif
    /* call fortran subroutine us */
    pinhole.nXP = nXP0;
    pinhole.nYP = nYP0; 
    nEE = 0;
    us_(&eBeam.energy, &eBeam.current, &eBeam.sigmax, 
        &eBeam.sigmay, &eBeam.sigmaxp, &eBeam.sigmayp, 
        &period, &undulator.nPeriods, &kx, &ky,
        &ekMin, &ekMax, &nE, 
        &pinhole.distance, &pinhole.xPC, &pinhole.yPC, 
        &pinhole.xPS, &pinhole.yPS, &pinhole.nXP, &pinhole.nYP, 
        &mode, &method, &i, 
        &nPhi, &nAlpha, &dAlpha, &nOmega, &dOmega, 
        &nSigma, /*end of input parameters */
        &E1,     /* energy of first harmonic */
        &lambda1, /* wavelength of same */
        &ptot,   /* total power */
        &pd,     /* on-axis power density */
        &ptot1,  /* total power */
        &isub,   /* indicates which analysis routine was used.  isub=2 corresponds to mode=4 */
        &iang,   /* if 1, angular units are used, otherwise spatial */
        &ftot,   /* total flux */
        &max_harmonic, &min_harmonic,   /* ignored */
        L1, L2, L3, L4, /* ? */
        &nEE, /* number of energy points */
        xPMM, /* x values (ignored) */
        yPMM, /* y values (ignored) */
        irradiance, /* ignored */
        spec1, /* flux or flux density */
        EE /* photon energy */
        );
    /*find the peak */
    FindPeak(EE, spec1, &ep, &sp, NULL, nE);
#ifdef DEBUG
    fprintf(stderr, "found peak at energy=%e, value=%e\n", ep, sp);
#endif
    if (i == 1)
      dep1 = eMax*i-ep;
    else
      dep2 = eMax*i-ep;
  }

#ifdef DEBUG
  fprintf(stderr, "Entering loop for harmonics\n");
#endif
  /*Main loop over harmonics and K-values */
  ih = 0;
  de = (eMax-eMin)/(nE-1);
  for (i=ihMin; i<=ihMax; i=i+ihStep) {
#ifdef DEBUG
    fprintf(stderr, "Working on harmonic %ld\n", i);
#endif
    /* Try Kmin initially to find fc for each harmonic */
    smax = 0;
    je = nE;
    nek = neks;
    do {
      je--;
      if (!undulator.kvalue)
	ek = eMin+je*de;
      else
	ek = reducedE/(1+sqr(undulator.kvalue[je])/2.0);
      ky = sqrt(2.0*(reducedE/ek-1.0));
      kx = 0;
      if (undulator.flags&UNDULATOR_HELICAL_GIVEN)
        kx = ky = ky/sqrt(2);
      if (ih == 0)
        kyb[je] = ky;
      if (i%2) {
        /*odd harmonics */
        ekMin = i*ek-i*dep1;
        ekMax = i*ek+i*dep1/2.0;
        if (i == 1) ekMin = ekMin-dep1;
        if (i> (ek/dep1)) {
          fprintf(stderr, "Warning: overlapping range for initial peak search for harmonic %ld\n", i);
          exitLoop = 1;
          ih++;
          break;
        }
      } else {
        /*even harmonics */
        ekMin = i*ek-4.0*dep2;
        ekMax = i*ek;
      }
      /* call us */
      pinhole.nXP = nXP0;
      pinhole.nYP = nYP0; 
      us_(&eBeam.energy, &eBeam.current, &eBeam.sigmax, 
          &eBeam.sigmay, &eBeam.sigmaxp, &eBeam.sigmayp, 
          &period, &undulator.nPeriods, &kx, &ky,
          &ekMin, &ekMax, &nE, 
          &pinhole.distance, &pinhole.xPC, &pinhole.yPC, 
          &pinhole.xPS, &pinhole.yPS, &pinhole.nXP, &pinhole.nYP, 
          &mode, &method, &i, 
          &nPhi, &nAlpha, &dAlpha, &nOmega, &dOmega, 
          &nSigma, /*end of input parameters */
          &E1,     /* energy of first harmonic */
          &lambda1, /* wavelength of same */
          &ptot,   /* total power */
          &pd,     /* on-axis power density */
          &ptot1,  /* total power */
          &isub,   /* indicates which analysis routine was used.  isub=2 corresponds to mode=4 */
          &iang,   /* if 1, angular units are used, otherwise spatial */
          &ftot,   /* total flux */
          &max_harmonic, &min_harmonic,   /* ignored */
          L1, L2, L3, L4, /* ? */
          &nEE, /* number of energy points */
          xPMM, /* x values (ignored) */
          yPMM, /* y values (ignored) */
          irradiance, /* ignored */
          spec1, /* flux or flux density */
          EE /* photon energy */
          );
      smax = 0;
      for (j=0; j<nE; j++)
        if (spec1[j]>smax)
          smax = spec1[j];
#ifdef DEBUG
      fprintf(stderr, "Saving values for je=%ld, ih=%ld, ei[je]=%p, eb[je]=%p, sb[je]=%p\n",
              je, ih, ei[je], eb[je], sb[je]);
#endif
      ei[je][ih] = i*ek;
      eb[je][ih] = i*ek;
      sb[je][ih] = 0.0;
    } while (smax<minF && je>0);
    if (exitLoop)
      break;
    if (smax < minF ) {
      fprintf(stderr, "Warning, Harmonic intensity %e too small, for harmonic number %ld\n", smax, i);
      break;
    }
    FindPeak(EE, spec1, &ep, &sp, NULL, nE);
#ifdef DEBUG
    fprintf(stderr, "Peak located at %e\n", ep);
#endif
    /*define fc */
    fc = 0.985*ep/ek/i;
    if (i > 1/(1-fc) ) {
      fprintf(stderr, "Warning: overlapping range for peak search for harmonics %ld\n", i);
      break;
    }
    je = nE-1;
    for (j=0; j<=je; j++) {
      if (!undulator.kvalue)
	ek = eMin+j*de;
      else
	ek = reducedE/(1+sqr(undulator.kvalue[j])/2.0);
      ky = sqrt(2.0*(reducedE/ek-1.0));
      kx = 0;
      if (undulator.flags&UNDULATOR_HELICAL_GIVEN)
        kx = ky = ky/sqrt(2);
      if (ih == 0) 
        kyb[j] = ky;
      if (i%2) 
        fc2 = 1.002;
      else
        fc2 = 1.000;
      ekMin = fc*i*ek;
      ekMax = fc2*i*ek;
      /* adjust ekmin, ekmax, and number of points if beam energy spread applied */
#ifdef DEBUG
        fprintf(stderr, "values for scan: ekMin = %e, ekMax = %e\n",
                ekMin, ekMax);
#endif
      if (sigmaE > 0) {
        nek = neks;
        dek = (ekMax-ekMin)/nek;
        sigmaEE = 2.0*sigmaE*i*ek; /*  estimated width (eV) */
        ekMin = ekMin-sigmaEE*nSigma; /* adjust for convolution */
        ekMax = ekMax+sigmaEE*nSigma;  /* adjust for convolution */
#ifdef DEBUG
        fprintf(stderr, "adjusted for convolution: ekMin = %e, ekMax = %e\n",
                ekMin, ekMax);
#endif
        if (sigmaEE/nppSigma < dek) {
          dek = sigmaEE/nppSigma;
        }
        nek = (ekMax-ekMin)/dek+1; /*number of points */
#ifdef DEBUG
          fprintf(stderr, "adjusted for convolution: nek = %ld, dek = %e\n", nek, dek);
#endif
        if (nek>MAXIMUM_E) {
          fprintf(stderr, "Energy points out of boundary (constrainted by FORTRAN usb subroutine).\n");
          exit(1);
        }
      }
      /* get undulator on-axis flux for given K value in energy range ekmin to ekmax
         returns energy array e (eV) and spec1 (ph/s/0.1%bw) */
      pinhole.nXP = nXP0;
      pinhole.nYP = nYP0; 
#ifdef DEBUG
      fprintf(stderr, "ky = %e, looking between %e and %e eV\n", ky, ekMin, ekMax);
#endif
      us_(&eBeam.energy, &eBeam.current, &eBeam.sigmax, 
          &eBeam.sigmay, &eBeam.sigmaxp, &eBeam.sigmayp, 
          &period, &undulator.nPeriods, &kx, &ky,
          &ekMin, &ekMax, &nek, 
          &pinhole.distance, &pinhole.xPC, &pinhole.yPC, 
          &pinhole.xPS, &pinhole.yPS, &pinhole.nXP, &pinhole.nYP, 
          &mode, &method, &i, 
          &nPhi, &nAlpha, &dAlpha, &nOmega, &dOmega, 
          &nSigma, /*end of input parameters */
          &E1,     /* energy of first harmonic */
          &lambda1, /* wavelength of same */
          &ptot,   /* total power */
          &pd,     /* on-axis power density */
          &ptot1,  /* total power */
          &isub,   /* indicates which analysis routine was used.  isub=2 corresponds to mode=4 */
          &iang,   /* if 1, angular units are used, otherwise spatial */
          &ftot,   /* total flux */
          &max_harmonic, &min_harmonic, 
          L1, L2, L3, L4, /* ? */
          &nEE, /* number of energy points */
          xPMM, /* x values (ignored) */
          yPMM, /* y values (ignored) */
          irradiance, /* ignored */
          spec1, /* flux or flux density */
          EE /* photon energy */
          );
      if (sigmaE>0) {
        /* gauss convolve */
        if (Gauss_Convolve(EE, spec1, &nek, sigmaE, NULL, NULL))
          exit(1);
        FindPeak(EE, spec1, &ep, &sp, NULL, nek);
#ifdef DEBUG
        fprintf(stderr, "after convolution, peak found at %e eV\n", ep);
#endif    
      } else {
        FindPeak(EE, spec1, &ep, &sp, NULL, nek);
#ifdef DEBUG
        fprintf(stderr, "peak found at %e eV\n", ep);
#endif
      }
#ifdef DEBUG
      fprintf(stderr, "%e eV, E1 = %e, lambda1 = %e, ptot = %e, pd = %e, ptot1 = %e\n", 
              ep, E1, lambda1, ptot, pd, ptot1);
      fprintf(stderr, "Saving values for j=%ld, ih=%ld, ei[j]=%p, eb[j]=%p, sb[j]=%p\n",
              j, ih, ei[j], eb[j], sb[j]);
#endif

      ei[j][ih] = i*ek;
      eb[j][ih] = ep;
      sb[j][ih] = sp*densityFactor;
      if (ih==0) {
        (*TotalPower)[nE-1-j] = ptot;
        (*OnAxisPowerDensity)[nE-1-j] = pd;
      }
    }
   /* fprintf(stderr, "Harmonics %d completed.\n", i); */
    ih++;
  } /*end for harmonics loop */
#ifdef DEBUG
  fprintf(stderr, "Exited loop for harmonics\n");
#endif
  
  /*output the result */
  *K = (double*)calloc(nE, sizeof(**K));
  *FnOut = (double**)malloc(sizeof(**FnOut)*ih);
  *Energy = (double**)malloc(sizeof(**Energy)*ih);
  *Flux = (double**)malloc(sizeof(**Flux)*ih);
  *LambdarOut = (double**)malloc(sizeof(**LambdarOut)*ih);
  for (i=0; i<ih; i++) {
    (*FnOut)[i] = calloc(nE, sizeof(***FnOut));
    (*Energy)[i] = calloc(nE, sizeof(***Energy));
    (*Flux)[i] = calloc(nE, sizeof(***Flux));
    (*LambdarOut)[i] = calloc(nE, sizeof(***LambdarOut));
  }
  for (j=0; j<nE; j++) {
    (*K)[j] = kyb[nE-1-j];
  }
  
  ih = 0;
  for (i=ihMin; i<=ihMax; i=i+ihStep) {
    for (j=0; j<nE; j++) {
      k = (*K)[j];
      JArg = i*k*k/(4+2*k*k);
      (*FnOut)[ih][j] = pow(i*k/(1+k*k/2)*(jn((i+1)/2, JArg) - jn((i-1)/2, JArg)), 2);
      (*Energy)[ih][j] = eb[nE-1-j][ih]*1.0e-3;
      (*Flux)[ih][j] = sb[nE-1-j][ih];
      e = (*Energy)[ih][j];
      (*LambdarOut)[ih][j] = 12.39/(e*1.0e10);
    }
    ih++;
  }
#ifdef DEBUG
  fprintf(stderr, "Copied results to output arrays\n");
#endif

  free(kyb);
  
  for (i=0; i<(MAXIMUM_E+1); i++) {
    free(ei[i]);
    free(eb[i]);
    free(sb[i]);
  }
  free(ei);
  free(eb);
  free(sb);
#ifdef DEBUG
  fprintf(stderr, "Freed memory.\n");
#endif
  return;
}


void getKValueDataFromFile(UNDULATOR_PARAM *undulator_param)
{
  SDDS_DATASET SDDSin;
  
  if (!SDDS_InitializeInput(&SDDSin, undulator_param->kfilename))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
  if (SDDS_CheckColumn(&SDDSin, undulator_param->kcolumn, NULL, SDDS_ANY_FLOATING_TYPE, stderr)!=SDDS_CHECK_OK) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
    fprintf(stderr, "Something wrong with column %s in %s\n", undulator_param->kcolumn, undulator_param->kfilename);
    exit(1);
  }
  if (SDDS_ReadPage(&SDDSin)<1 || (undulator_param->nPoints = SDDS_RowCount(&SDDSin))<1 || 
      !(undulator_param->kvalue=SDDS_GetColumnInDoubles(&SDDSin, undulator_param->kcolumn)))
    SDDS_Bomb("unable to read error factor file");
  if (!SDDS_Terminate(&SDDSin))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors|SDDS_EXIT_PrintErrors);
  /*sort kvalue in decreasing order to make the energy in increasing order */
  qsort(undulator_param->kvalue, undulator_param->nPoints, sizeof(*undulator_param->kvalue), double_cmpdes);
  undulator_param->KMin = undulator_param->kvalue[undulator_param->nPoints-1];
  undulator_param->KMax = undulator_param->kvalue[0];
  if (undulator_param->KMax<undulator_param->KMin || undulator_param->KMin<0 || undulator_param->nPoints<1)
    SDDS_Bomb("Invalid k values provided for undulator-- has to at least two points, and in increasing order."); 
  
}

double Undulator_QnK(int n, double K) {
  double x, sJJ;
  
  x = n*K*K/(4 + 2*K*K);
  sJJ = jn((n-1)/2, x) - jn((n+1)/2, x);

  return n*sqr(K*sJJ)/(1+K*K/2);
}

double Undulator_QnKK(int n, double Kx, double Ky) {
  /* After J. Clarke's 2010 Lecture */
  double Y, K2p, K2m, Jp, Jm;

  K2p = sqr(Kx)+sqr(Ky);
  K2m = sqr(Ky)-sqr(Kx);
  Y = n*K2m/(4+2*K2p);
  Jp = jn((n+1)/2, Y);
  Jm = jn((n-1)/2, Y);
  
  return n*(sqr(Kx*(Jp+Jm)) + sqr(Ky*(Jp-Jm)))/(1+K2p/2);
}


void UndulatorCentralConeFlux(ELECTRON_BEAM_PARAM eBeam, UNDULATOR_PARAM undulator, int h, double *K, double *CentralConeFlux) 
{
  long iK;

  if (undulator.flags&UNDULATOR_HELICAL_GIVEN) {
    for (iK=0; iK<undulator.nPoints; iK++) {
      CentralConeFlux[iK] = 1.431e14*eBeam.current*undulator.nPeriods*Undulator_QnKK(h, K[iK], K[iK]);
    }
  } else {
    for (iK=0; iK<undulator.nPoints; iK++) 
      CentralConeFlux[iK] = 1.431e14*eBeam.current*undulator.nPeriods*Undulator_QnK(h, K[iK]);
  }
  
}



