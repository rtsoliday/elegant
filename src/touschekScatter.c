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
#include "match_string.h"
#include "touschekScatter.h"
#include "SDDS.h"
#include "constants.h"

#define DEBUG 0
#define MAXREGION 30
#define STEPS 100

/* global variable */
static TSCATTER_SPEC *tsSpec = NULL;

/* Initialization of the simulation */
void init_TSPEC(RUN *run, LINE_LIST *beamlin, long nElement);
TSCATTER *initTSCATTER(ELEMENT_LIST *eptr, long iElement);
long get_MAInput(char *filename, LINE_LIST *beamline, long nElement);
long Check_HisInput(char *filename, LINE_LIST *beamline, long nElement, long *pIndex, long flag);

/* Calculate local and integrated Touschek scattering rate */
int TouschekRate(LINE_LIST *beamline, long nElement);
void FIntegral(double tm, double b1, double b2, double *F);
double Fvalue(double t, double tm, double b1, double b2);

/* Monte Carlo simulation of Touschek scattering */
void TouschekDistribution(RUN *run, VARY *control, LINE_LIST *beamline);

void selectPartGauss(TSCATTER *tsptr, double *p1, double *p2,
                     double *dens1, double *dens2, double *ran1);
void selectPartReal(TSCATTER *tsptr, double *p1, double *p2,
                    double *dens1, double *dens2, double *ran1);

void bunch2cm(double *p1, double *p2, double *q, double *beta, double *gamma);
void eulertrans(double *v0, double theta, double phi, double *v1, double *v);
void cm2bunch(double *p1, double *p2, double *q, double *beta, double *gamma);
double moeller(double beta0, double theta);
void pickPart(double *weight, long *index, long start, long end,
              long *iTotal, double *wTotal, double weight_limit, double weight_ave);
void determineOccurenceInFilenames(short *occurenceSeen, short *noOccurenceSeen,
                                   char *initial, char *distribution, char *output, char *loss, char *bunch);

void TouschekEffect(RUN *run,
                    VARY *control,
                    ERRORVAL *errcon,
                    LINE_LIST *beamline,
                    NAMELIST_TEXT *nltext) {
  ELEMENT_LIST *eptr;
  long nElement;
  double TSCATTER_Start, TSCATTER_End = 0, sEndLine = 0;

  /* Check if there is TScatter element along beamline. */
  if (!(eptr = beamline->elem_recirc))
    eptr = beamline->elem;
  nElement = 0;
  TSCATTER_Start = eptr->end_pos;
  while (eptr) {
    if (eptr->type == T_TSCATTER) {
      nElement++;
      TSCATTER_End = eptr->end_pos;
      if (nElement == 1) {
        if (TSCATTER_End != TSCATTER_Start)
          bombElegant("You should have the first TSCATTER element at the beginning of beamline", NULL);
      }
    }
    sEndLine = eptr->end_pos;
    eptr = eptr->succ;
  }
  if (sEndLine != TSCATTER_End)
    bombElegant("You should have the last TSCATTER element at the end of beamline", NULL);
  if (!nElement)
    bombElegant("No TSCATTER element along beamline", NULL);

  /* process the namelist text */
  set_namelist_processing_flags(STICKY_NAMELIST_DEFAULTS);
  set_print_namelist_flags(0);
  if (processNamelist(&touschek_scatter, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &touschek_scatter);

  if (!charge)
    bombElegant("charge has to be given", NULL);
  if (frequency < 1)
    bombElegant("frequency has to >=1", NULL);
  if (!emit_x && !emit_nx)
    bombElegant("bunch emittance-x has to be given", NULL);
  if (!emit_nx)
    emit_nx = sqrt(1. + sqr(run->p_central)) * emit_x;
  if (!emit_y && !emit_ny)
    bombElegant("bunch emittance-y has to be given", NULL);
  if (!emit_ny)
    emit_ny = sqrt(1. + sqr(run->p_central)) * emit_y;
  if (!sigma_dp)
    bombElegant("energy spread has to be given", NULL);
  if (!sigma_s)
    bombElegant("bunch length has to be given", NULL);
  if (sbin_step <= 0)
    bombElegant("sbin can not be less than 0", NULL);
  if (do_track)
    if (control->ready != 1)
      bombElegant("run_control must precede touschek_scatter namelists for doing tracking", NULL);

  if (!Momentum_Aperture)
    bombElegant("Momentum_Aperture file needed before performing simulation", NULL);
  if (get_MAInput(Momentum_Aperture, beamline, nElement) < 0)
    bombElegant("The input Momentum_Aperture file is not valid for this calculation - not same element location!", NULL);

  /* Initialize Touschek calculation */
  init_TSPEC(run, beamline, nElement);

  /* Calculate Piwinski's scattering rate. */
  if (TouschekRate(beamline, nElement))
    bombElegant("Touschek scattering rate calculation error", NULL);
  /* Generate scattered particles at TScatter. 
     And track the scattered particles down to beamline. */
  TouschekDistribution(run, control, beamline);

  return;
}

int TouschekRate(LINE_LIST *beamline, long nElement) {
  double NP;
  double tmP, tmN, B1, B2, F, rateP, rateN, IntR, IntLength, length;
  double rate0, rate;
  double *tmPList, *tmNList, *posList;
  ELEMENT_LIST *eptr;

  double betagamma, gamma;
  double sp2, sp4, beta2, betagamma2, sh2;
  double sx2, sxb2, dx2, dx_2;
  double sy2, syb2, dy2, dy_2;
  double a0, c0, c1, c2, c3;
  TWISS *twiss0;

  long i = 0;

  NP = tsSpec->charge / e_mks;
  a0 = sqr(re_mks) * c_mks * NP * NP / (8 * sqrt(PI) * sigma_s);

  tmPList = (double *)malloc(sizeof(double) * nElement);
  tmNList = (double *)malloc(sizeof(double) * nElement);
  posList = (double *)malloc(sizeof(double) * nElement);
  i = 0;
  if (!(eptr = beamline->elem_recirc))
    eptr = beamline->elem;
  while (eptr) {
    if (eptr->type == T_TSCATTER) {
      tmPList[i] = sqr(eptr->Pref_output) / (sqr(eptr->Pref_output) + 1.) * sqr(((TSCATTER *)eptr->p_elem)->deltaP);
      tmNList[i] = sqr(eptr->Pref_output) / (sqr(eptr->Pref_output) + 1.) * sqr(((TSCATTER *)eptr->p_elem)->deltaN);
      posList[i] = eptr->end_pos;
      i++;
    }
    eptr = eptr->succ;
  }

  IntR = 0.;
  IntLength = 0.;
  rate0 = rate = 0.;
  i = 0;
  if (!(eptr = beamline->elem_recirc))
    eptr = beamline->elem;
  while (eptr) {
    if (eptr->type == T_TSCATTER) {
      if (IntLength != 0) {
        ((TSCATTER *)eptr->p_elem)->AveR = IntR / IntLength;
      } else {
        ((TSCATTER *)eptr->p_elem)->AveR = 0;
      }
      ((TSCATTER *)eptr->p_elem)->p_rate = rate;
      ((TSCATTER *)eptr->p_elem)->total_scatter = IntR / c_mks * tsSpec->frequency;
      IntR = 0.;
      IntLength = 0.;
      i++;
    }
    /* calculate scattering rate for the first element, which should always have zero length */
    if (rate0 == 0) {
      tmP = tmPList[0];
      tmN = tmNList[0];
      length = 0;
    } else {
      if (!(entity_description[eptr->type].flags & HAS_LENGTH) ||
          (length = ((DRIFT *)eptr->p_elem)->length) <= 0) {
        eptr = eptr->succ;
        continue;
      }
      if (i == 0 || i == nElement)
        bombElegant("Something wrong, ask expert for help", NULL);

      tmP = ((eptr->end_pos - posList[i - 1]) * tmPList[i] + (posList[i] - eptr->end_pos) * tmPList[i - 1]) / (posList[i] - posList[i - 1]);
      tmN = ((eptr->end_pos - posList[i - 1]) * tmNList[i] + (posList[i] - eptr->end_pos) * tmNList[i - 1]) / (posList[i] - posList[i - 1]);
    }

    betagamma = eptr->Pref_output;
    gamma = sqrt(sqr(betagamma) + 1.);
    beta2 = sqr(betagamma) / sqr(gamma);
    betagamma2 = sqr(betagamma);

    sp2 = sqr(tsSpec->delta_p0 / betagamma);
    sp4 = sqr(sp2);

    twiss0 = eptr->twiss;
    sxb2 = twiss0->betax * tsSpec->emitN[0] / gamma;
    dx2 = sqr(twiss0->etax);
    sx2 = sxb2 + dx2 * sp2;
    dx_2 = sqr(twiss0->alphax * twiss0->etax + twiss0->betax * twiss0->etapx);

    syb2 = twiss0->betay * tsSpec->emitN[1] / gamma;
    dy2 = sqr(twiss0->etay);
    sy2 = syb2 + dy2 * sp2;
    dy_2 = sqr(twiss0->alphay * twiss0->etay + twiss0->betay * twiss0->etapy);

    sh2 = 1 / (1 / sp2 + (dx2 + dx_2) / sxb2 + (dy2 + dy_2) / syb2);
    c0 = sh2 / (sqr(beta2 * tsSpec->emitN[0] * tsSpec->emitN[1]) * sp2);
    c1 = sqr(twiss0->betax) / (2 * betagamma2 * sxb2);
    c2 = sqr(twiss0->betay) / (2 * betagamma2 * syb2);
    c3 = sx2 * sy2 - sp4 * dx2 * dy2;
    B1 = c1 * (1 - sh2 * dx_2 / sxb2) + c2 * (1 - sh2 * dy_2 / syb2);
    B2 = sqr(B1) - c0 * c3;
    if (B2 < 0) {
      if (fabs(B2 / sqr(B1)) < 1e-7)
        printWarningForTracking("touschek_scatter: B2^2<0.",
                                "Please seek expert help.");
      else
        B2 = 0;
    }
    B2 = sqrt(B2);

    FIntegral(tmP, B1, B2, &F);
    rateP = a0 * sqrt(c0) * F / gamma / gamma / 2.;
    FIntegral(tmN, B1, B2, &F);
    rateN = a0 * sqrt(c0) * F / gamma / gamma / 2.;
    /* first element has length=0, this only set rate0 to a non zero value */
    rate = rateP + rateN;
    IntR += (rate0 + rate) * length / 2;
    IntLength += length;
    rate0 = rate;
    eptr = eptr->succ;
  }
  free(tmPList);
  free(tmNList);
  free(posList);
  return (0);
}

void FIntegral(double tm, double b1, double b2, double *F) {
  long maxRegion = MAXREGION;
  long steps = STEPS; /* number of integration steps per decade */
  long i, j;
  double test = 1e-5, simpsonCoeff[2] = {2., 4.};
  double h, tau0, tau, intF;
  double cof, sum, fun;
  /* Using simpson's rule to do the integral. 
     split integral into region with "steps" steps per region.
     integration intervals be [1,3], [3,9], [9,27], and so on. 
     i is the index over these intervals. j is the index over each step.
  */
  tau0 = tm;
  intF = 0.0;
  for (i = 0; i < maxRegion; i++) {
    if (i == 0) {
      h = tau0 * 2. / steps;
    } else {
      h = tau0 * 3. / steps;
    }
    sum = 0.0;
    for (j = 0; j <= steps; j++) {
      tau = tau0 + h * j;
      cof = simpsonCoeff[j % 2];
      if (j == 0 || j == steps)
        cof = 1.;
      fun = Fvalue(tau, tm, b1, b2);
      sum += cof * fun;
    }
    tau0 = tau;
    sum = (sum / 3.0) * h;
    intF += sum;
    if (FABS(sum / intF) < test)
      break;
    if (i == maxRegion)
      printWarning("touschek_scatter: integral did not converge.", NULL);
  }

  *F = intF;
  return;
}

double Fvalue(double t, double tm, double b1, double b2) {
  double c0, c1, c2, result;

  c0 = (sqr(2. + 1. / t) * (t / tm / (1. + t) - 1.) + 1. - sqrt(tm * (1. + t) / t) - (4. + 1. / t) / (2. * t) * log(t / tm / (1. + t))) * sqrt(t / (1. + t));
  c1 = exp(-b1 * t);
  c2 = dbesi0(b2 * t);
  result = c0 * c1 * c2;
  /* If overflow/underflow use approximate equation for modified bessel function. */
  if (isnan(result) || fabs(result) > FLT_MAX) {
    result = c0 * exp(b2 * t - b1 * t) / sqrt(PIx2 * b2 * t);
  }
  return result;
}

#define PART_DIST_PARAMETERS 7
static SDDS_DEFINITION part_dist_para[PART_DIST_PARAMETERS] = {
  {"Element_Name", "&parameter name=Element_Name, type=string &end"},
  {"s", "&parameter name=s, units=\"m\", description=\"Location from beginning of beamline\", type=double &end"},
  {"Piwinski_AveR", "&parameter name=Piwinski_AveR, description=\"Average scattering rate from Piwinski's formula\", type=double &end"},
  {"Piwinski_Rate", "&parameter name=Piwinski_Rate, description=\"Local scattering rate from Piwinski's formula\",  type=double &end"},
  {"MC_Rate", "&parameter name=MC_Rate, description=\"Local scattering rate from Monte Carlo simulation\", type=double &end"},
  {"Ignored_Rate", "&parameter name=Ignored_Rate, description=\"Ignored rate from tracking\", type=double &end"},
  {"Bins", "&parameter name=Bins, type=long &end"},
};
static void *part_dist_paraValue[PART_DIST_PARAMETERS];

void TouschekDistribution(RUN *run, VARY *control, LINE_LIST *beamline) {
  long i, j, total_event, n_left, iElement = 0, elementIndex;
  ELEMENT_LIST *eptr;
  TSCATTER *tsptr;
  double pTemp[6], p1[6], p2[6], dens1, dens2;
  double theta, phi, qa[3], qb[3], beta[3], qabs, gamma;
  double beta0, cross, temp;
  static SDDS_TABLE SDDS_bunch, SDDS_loss;
  BEAM Beam0, Beam, *beam0, *beam;
  char *Name[6] = {"x", "y", "s", "xp", "yp", "dp/p"};
  char *Units[6] = {"m", "m", "m", "", "", ""};
  int32_t bookBins[6];
  book1 *lossDis = NULL;
  book1m *iniBook = NULL, *disBook = NULL;
  double *weight;
  double ran1[11];
  long *index, iTotal, sTotal;
  double weight_limit, weight_ave, wTotal;
  double **fiducialParticle, pCentral;
  long iProcessing = 0;
  short occurenceSeen = 0, noOccurenceSeen = 0, skip;
  double **lostParticle = NULL;
  long nLost = 0;

  fiducialParticle = (double **)czarray_2d(sizeof(**fiducialParticle), 1, totalPropertiesPerParticle);
  if (!(eptr = beamline->elem_recirc))
    eptr = beamline->elem;
  beam0 = &Beam0;
  beam = &Beam;
  memset(beam, 0, sizeof(*beam));
  memset(beam0, 0, sizeof(*beam0));
  sTotal = (long)(beamline->revolution_length / sbin_step) + 1;

  determineOccurenceInFilenames(&occurenceSeen, &noOccurenceSeen,
                                initial, distribution, output, loss, bunch);
  if (occurenceSeen && noOccurenceSeen)
    bombElegant("Some output files have occurence field and some don't.  Use only one method.", NULL);
  if (verbosity) {
    printf("Output filenames will%s have occurence index\n", occurenceSeen ? "" : " not");
    fflush(stdout);
  }

  iTotal = 0; /* Suppress compiler warning */

  elementIndex = 0; /* index relative to start or recirculation point */
  while (eptr) {
    if (eptr->type == T_TSCATTER) {
      iElement++; /* index of TSCATTER elements */
      if (i_start >= 0 && iElement < i_start) {
        if (verbosity) {
          printf("Skipping %s#%ld at s=%le (outside index range)\n",
                 eptr->name, eptr->occurence, eptr->end_pos);
          fflush(stdout);
        }
        eptr = eptr->succ;
        elementIndex++;
        continue;
      }
      if (i_end >= 0 && iElement > i_end) {
        if (verbosity) {
          printf("Breaking at %s#%ld at s=%le (outside index range)\n",
                 eptr->name, eptr->occurence, eptr->end_pos);
          fflush(stdout);
        }
        break;
      }

      tsptr = initTSCATTER(eptr, iElement);
      skip = 0;
      if (initial) {
        tsptr->iniFile = compose_filename_occurence(initial, run->rootname, eptr->occurence);
        if (occurenceSeen && !overwrite_files && fexists(tsptr->iniFile))
          skip = 1;
      }
      if (distribution) {
        tsptr->disFile = compose_filename_occurence(distribution, run->rootname, eptr->occurence);
        if (occurenceSeen && !overwrite_files && fexists(tsptr->disFile))
          skip = 1;
      }
      if (output) {
        tsptr->outFile = compose_filename_occurence(output, run->rootname, eptr->occurence);
        if (occurenceSeen && !overwrite_files && fexists(tsptr->outFile))
          skip = 1;
      }
      if (loss) {
        tsptr->losFile = compose_filename_occurence(loss, run->rootname, eptr->occurence);
        if (occurenceSeen && !overwrite_files && fexists(tsptr->losFile))
          skip = 1;
      }
      if (bunch) {
        tsptr->bunFile = compose_filename_occurence(bunch, run->rootname, eptr->occurence);
        if (occurenceSeen && !overwrite_files && fexists(tsptr->bunFile))
          skip = 1;
      }
      if (skip) {
        if (verbosity) {
          printf("Skipping %s#%ld at s=%le (some files exist)\n",
                 eptr->name, eptr->occurence, eptr->end_pos);
          fflush(stdout);
        }
        eptr = eptr->succ;
        elementIndex++;
        continue;
      }
      if (verbosity) {
        printf("Working on %s#%ld at s=%le, n_simulated = %ld\n",
               eptr->name, eptr->occurence, eptr->end_pos, n_simulated);
        fflush(stdout);
      }

      iProcessing++;

      weight = (double *)malloc(sizeof(double) * n_simulated);
      beam0->particle = (double **)czarray_2d(sizeof(double), n_simulated, totalPropertiesPerParticle);
      /* beam0->original = (double**)czarray_2d(sizeof(double), n_simulated, totalPropertiesPerParticle); */
      beam0->original = NULL;
      beam0->accepted = NULL;
      beam0->n_original = beam0->n_to_track = beam0->n_particle = n_simulated;
      beam0->n_accepted = beam0->n_saved = 0;
      beam0->p0_original = beam0->p0 = tsptr->betagamma;
      beam0->bunchFrequency = 0.;
      beam0->n_lost = 0;

      if (verbosity > 1) {
        printf("Working arrays allocated\n");
        fflush(stdout);
      }

      if (initial) {
        for (i = 0; i < 6; i++)
          bookBins[i] = tsSpec->nbins;
        iniBook = chbook1m(Name, Units, tsptr->xmin, tsptr->xmax, bookBins, 6);
        if (verbosity > 1) {
          printf("iniBook set up\n");
          fflush(stdout);
        }
      }
      if (distribution) {
        for (i = 0; i < 6; i++)
          bookBins[i] = tsSpec->nbins;
        tsptr->xmin[5] = -0.1;
        tsptr->xmax[5] = 0.1;
        disBook = chbook1m(Name, Units, tsptr->xmin, tsptr->xmax, bookBins, 6);
        if (verbosity > 1) {
          printf("disBook set up\n");
          fflush(stdout);
        }
      }
      if (output) {
        lossDis = chbook1("s", "m", 0, beamline->revolution_length, sTotal);
        if (verbosity > 1) {
          printf("lossDis set up\n");
          fflush(stdout);
        }
      }
#if USE_MPI
      if (isMaster)
#endif
        if (loss) {
          if (occurenceSeen || iProcessing == 1) {
            if (verbosity)
              printf("Setting up loss file\n");
            SDDS_BeamScatterLossSetup(&SDDS_loss, tsptr->losFile, SDDS_BINARY, 1,
                                      "lost particle coordinates", run->runfile,
                                      run->lattice, "touschek_scatter");
          }
        }
#if USE_MPI
      if (isMaster)
#endif
        if (bunch) {
          if (occurenceSeen || iProcessing == 1) {
            SDDS_BeamScatterSetup(&SDDS_bunch, tsptr->bunFile, SDDS_BINARY, 1,
                                  "scattered-beam phase space", run->runfile,
                                  run->lattice, "touschek_scatter");
            if (verbosity > 1) {
              printf("bunch output set up\n");
              fflush(stdout);
            }
          }
        }
      if (verbosity)
        report_stats(stdout, "Before particle generation: ");
      i = 0;
      j = 0;
      total_event = 0;
      while (1) {
        if (i >= n_simulated)
          break;
        /* Select the 11 random number then mix them. Use elegant run_setup seed */
        for (j = 0; j < 11; j++) {
          ran1[j] = random_1_elegant(1);
        }
        randomizeOrder((char *)ran1, sizeof(ran1[0]), 11, 0, random_4);

        total_event++;
        if (!tsSpec->distIn)
          selectPartGauss(tsptr, p1, p2, &dens1, &dens2, ran1);
        else
          selectPartReal(tsptr, p1, p2, &dens1, &dens2, ran1);
        if (initial) {
          chfill1m(iniBook, p1, dens1, bookBins, 6);
          chfill1m(iniBook, p2, dens2, bookBins, 6);
        }
        if (!dens1 || !dens2) {
          continue;
        }
        /* This is very important. Change from slop to MeV */
        for (j = 3; j < 5; j++) {
          p1[j] *= tsptr->pCentral_mev;
          p2[j] *= tsptr->pCentral_mev;
        }
        p1[5] = (p1[5] + 1) * tsptr->pCentral_mev;
        p2[5] = (p2[5] + 1) * tsptr->pCentral_mev;

        bunch2cm(p1, p2, qa, beta, &gamma);

        theta = (ran1[9] * 0.9999 + 0.00005) * PI;
        phi = ran1[10] * PI;

        temp = dens1 * dens2 * sin(theta);
        eulertrans(qa, theta, phi, qb, &qabs);
        cm2bunch(p1, p2, qb, beta, &gamma);
        p1[5] = (p1[5] - tsptr->pCentral_mev) / tsptr->pCentral_mev;
        p2[5] = (p2[5] - tsptr->pCentral_mev) / tsptr->pCentral_mev;

        if (p1[5] > p2[5]) {
          for (j = 0; j < 6; j++) {
            pTemp[j] = p2[j];
            p2[j] = p1[j];
            p1[j] = pTemp[j];
          }
        }

        if (p1[5] < tsptr->deltaN || p2[5] > tsptr->deltaP) {
          beta0 = qabs / sqrt(qabs * qabs + me_mev * me_mev);
          cross = moeller(beta0, theta);
          temp *= cross * beta0 / gamma / gamma;

          if (p1[5] < tsptr->deltaN) {
            tsptr->totalWeight += temp;
            p1[3] /= tsptr->pCentral_mev;
            p1[4] /= tsptr->pCentral_mev;
            if (distribution) {
              chfill1m(disBook, p1, temp, bookBins, 6);
            }
            tsptr->simuCount++;

            beam0->particle[i][0] = p1[0];
            beam0->particle[i][1] = p1[3];
            beam0->particle[i][2] = p1[1];
            beam0->particle[i][3] = p1[4];
            beam0->particle[i][4] = p1[2];
            beam0->particle[i][5] = p1[5];
            beam0->particle[i][6] = i + 1;
            weight[i] = temp;
            i++;
          }

          if (i >= n_simulated)
            break;

          if (p2[5] > tsptr->deltaP) {
            tsptr->totalWeight += temp;
            p2[3] /= tsptr->pCentral_mev;
            p2[4] /= tsptr->pCentral_mev;
            if (distribution) {
              chfill1m(disBook, p2, temp, bookBins, 6);
            }
            tsptr->simuCount++;

            beam0->particle[i][0] = p2[0];
            beam0->particle[i][1] = p2[3];
            beam0->particle[i][2] = p2[1];
            beam0->particle[i][3] = p2[4];
            beam0->particle[i][4] = p2[2];
            beam0->particle[i][5] = p2[5];
            beam0->particle[i][6] = i + 1;
            weight[i] = temp;
            i++;
          }
          /*          printf("scatted particles %ld.\n", tsptr->simuCount); */
        }
        if (total_event * 11 > (long)2e9) {
          printWarning("touschek_scatter: the total number of random numbers used exceeded 2e9.",
                       "Use smaller n_simulated or smaller delta.");
          break;
        }
      }
      if (verbosity)
        report_stats(stdout, "After particle generation: ");
      if (tsptr->simuCount == 0)
        bombElegant("It appears that the Touschek lifetime is extremely long and there is no need to perform Touschek simulation.\n If you think this is a wrong statement, send input to developers for evaluation.\n", NULL);

      if (total_event / tsptr->simuCount > 20) {
        if (distribution_cutoff[0] < 5 || distribution_cutoff[1] < 5)
          printWarning("touschek_scatter: scattering rate is low.",
                       "Please use >=5 sigma beam for better simulation.");
        else
          printWarning("touschek_scatter: Sscattering rate is very low.",
                       "Please ignore the rate from Monte Carlo simulation. Use Piwinski's rate only.");
      }
      tsptr->factor = tsptr->factor / (double)(total_event);
      tsptr->s_rate = tsptr->totalWeight * tsptr->factor;
      tsptr->i_rate = tsptr->s_rate * tsSpec->ignoredPortion;

      /* Pick tracking particles from the simulated scattered particles */
      index = (long *)malloc(sizeof(long) * tsptr->simuCount);
      for (i = 0; i < tsptr->simuCount; i++)
        index[i] = i;
      if (tsSpec->ignoredPortion <= 1e-9) {
        iTotal = tsptr->simuCount;
        wTotal = tsptr->totalWeight;
      } else {
        iTotal = 0;
        wTotal = 0.;
        weight_limit = tsptr->totalWeight * (1 - tsSpec->ignoredPortion);
        weight_ave = tsptr->totalWeight / tsptr->simuCount;
#if USE_MPI
        if (isMaster)
#endif
          pickPart(weight, index, 0, tsptr->simuCount,
                   &iTotal, &wTotal, weight_limit, weight_ave);
      }
      if (verbosity) {
        printf("%ld of %ld particles selected for tracking\n", iTotal, tsptr->simuCount);
        fflush(stdout);
      }
      if (verbosity)
        report_stats(stdout, "After particle selection: ");
#if USE_MPI
      if (USE_MPI) {
        /* Share the number of particles to track for purposes of setting array sizes on the slaves */
        long iTotal_orig;
        long work_processors = n_processors - 1;

        MPI_Bcast(&iTotal, 1, MPI_LONG, 0, MPI_COMM_WORLD);
        iTotal_orig = iTotal;
        if (isSlave) {
          iTotal = iTotal_orig / work_processors;
          if (myid <= iTotal_orig % work_processors)
            iTotal++;
        }
      }
#endif
      beam->particle = (double **)czarray_2d(sizeof(double), iTotal, totalPropertiesPerParticle);
      beam->original = (double **)czarray_2d(sizeof(double), iTotal, totalPropertiesPerParticle);
      beam->accepted = NULL;
      beam->n_original = beam->n_to_track = beam->n_particle = iTotal;
      beam->n_accepted = beam->n_saved = 0;
      beam->p0_original = beam->p0 = tsptr->betagamma;
      beam->bunchFrequency = 0.;
      beam->n_lost = 0;

#if USE_MPI
      if (myid != 0)
        iTotal = 0; /* This will be set when we scatter in do_tracking */
#endif
      for (i = 0; i < iTotal; i++) {
        beam->original[i][0] = beam->particle[i][0] = beam0->particle[index[i]][0];
        beam->original[i][1] = beam->particle[i][1] = beam0->particle[index[i]][1];
        beam->original[i][2] = beam->particle[i][2] = beam0->particle[index[i]][2];
        beam->original[i][3] = beam->particle[i][3] = beam0->particle[index[i]][3];
        beam->original[i][4] = beam->particle[i][4] = beam0->particle[index[i]][4];
        beam->original[i][5] = beam->particle[i][5] = beam0->particle[index[i]][5];
        beam->original[i][6] = beam->particle[i][6] = i + 1;
        weight[i] *= tsptr->factor;
      }
      if (tsSpec->distIn)
        tsptr->total_scatter *= tsptr->s_rate / tsptr->p_rate;
#if USE_MPI
      if (isMaster)
#endif
        if (bunch) {
          dump_scattered_particles(&SDDS_bunch, beam->particle, (long)iTotal,
                                   weight, tsptr);
          if (occurenceSeen && !SDDS_Terminate(&SDDS_bunch)) {
            SDDS_SetError("Problem terminating 'bunch' file (finish_output)");
            SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
          }
        }
      if (distribution || initial) {
        part_dist_paraValue[0] = (void *)(&tsptr->name);
        part_dist_paraValue[1] = (void *)(&tsptr->s);
        part_dist_paraValue[2] = (void *)(&tsptr->AveR);
        part_dist_paraValue[3] = (void *)(&tsptr->p_rate);
        part_dist_paraValue[4] = (void *)(&tsptr->s_rate);
        part_dist_paraValue[5] = (void *)(&tsptr->i_rate);
        part_dist_paraValue[6] = (void *)(&bookBins[0]);
#if USE_MPI
        if (isMaster)
#endif
          if (distribution) {
            chprint1m(disBook, tsptr->disFile, "Simulated scattered particle final distribution", part_dist_para,
                      part_dist_paraValue, PART_DIST_PARAMETERS, 1, 0, noOccurenceSeen && iProcessing != 1);
            free_hbook1m(disBook);
          }
#if USE_MPI
        if (isMaster)
#endif
          if (initial) {
            chprint1m(iniBook, tsptr->iniFile, "Simulated scattered particle original distribution", part_dist_para,
                      part_dist_paraValue, PART_DIST_PARAMETERS, 1, 0, noOccurenceSeen && iProcessing != 1);
            free_hbook1m(iniBook);
          }
      }
#if USE_MPI
      notSinglePart = 0;
      partOnMaster = 1;
      parallelStatus = notParallel;
#endif
      if (do_track) {
        if (verbosity > 1) {
          printf("Tracking fiducial particle\n");
          fflush(stdout);
        }
        memset(fiducialParticle[0], 0, sizeof(**fiducialParticle) * totalPropertiesPerParticle);
        delete_phase_references();
        reset_special_elements(beamline, RESET_INCLUDE_ALL & ~RESET_INCLUDE_RANDOM);
        pCentral = run->p_central;
        if (!do_tracking(NULL, fiducialParticle, 1, NULL, beamline,
                         &pCentral, NULL, NULL, NULL, NULL, run, control->i_step,
                         FIRST_BEAM_IS_FIDUCIAL + RESTRICT_FIDUCIALIZATION + (verbosity > 1 ? 0 : SILENT_RUNNING) + INHIBIT_FILE_OUTPUT, 1, 0, NULL, NULL, NULL, NULL, NULL)) {
          bombElegant("Fiducial particle was lost", NULL);
        }
        if (verbosity > 1) {
          printf("fiducial particle tracked.\n");
          fflush(stdout);
        }
        if (eptr->pred) {
          /* particle must look as if it traveled from start of beamline to this location in order to get the right
           * phase at the rf cavities set up by the fiducial particle
           */
          long ip;
          for (ip = 0; ip < iTotal; ip++)
            beam->particle[ip][4] += eptr->pred->end_pos;
        }
#if USE_MPI && MPI_DEBUG
        printf("%ld particles on processor %d\n", iTotal, myid);
        fflush(stdout);
#endif
#if USE_MPI
        notSinglePart = 1;
        partOnMaster = 1;
        parallelStatus = notParallel;
#endif
        if (beamline->closed_orbit) {
          /* If a closed orbit was computed, scatter with respect to that. */
          long ip, k;
          for (ip = 0; ip < iTotal; ip++) {
            for (k = 0; k < 4; k++)
              beam->particle[ip][k] += beamline->closed_orbit[elementIndex].centroid[k];
          }
        }
        n_left = do_tracking(beam, NULL, (long)iTotal, NULL, beamline,
                             &beam->p0, NULL, NULL, NULL, NULL, run, control->i_step,
                             FIRST_BEAM_IS_FIDUCIAL + FIDUCIAL_BEAM_SEEN + RESTRICT_FIDUCIALIZATION + (verbosity > 2 ? 0 : SILENT_RUNNING + INHIBIT_FILE_OUTPUT), control->n_passes, 0, NULL,
                             NULL, NULL, NULL, eptr);
        nLost = beam->n_lost;
        notSinglePart = 0;
        if (verbosity > 1)
          report_stats(stdout, "Tracking completed (1): ");
#if USE_MPI
        partOnMaster = 1;
        parallelStatus = notParallel;
#  if MPI_DEBUG
        printf("%ld particles on processor %d\n", iTotal, myid);
        fflush(stdout);
#  endif
        gatherLostParticles(&lostParticle, &nLost, beam->particle, n_left, n_processors, myid);
        if (verbosity > 1) {
          printf("Gathered %ld lost particles to master\n", nLost);
          fflush(stdout);
        }
        if (isMaster)
          beam->n_lost = nLost;

        if (verbosity > 1) {
          long nLeftTotal;
          if (isMaster)
            n_left = 0;
          MPI_Reduce(&n_left, &nLeftTotal, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
          if (isMaster)
            printf("%ld particles survived tracking\n", nLeftTotal);
#  if MPI_DEBUG
          if (!isMaster)
            printf("%ld particles survived tracking\n", n_left);
#  endif
          fflush(stdout);
        }

#else
        lostParticle = beam->particle + n_left;
        nLost = beam->n_lost;
        if (verbosity > 1) {
          printf("%ld of %ld particles survived tracking\n", n_left, iTotal);
          fflush(stdout);
        }
#endif

#if USE_MPI
        if (isMaster)
#endif
          if (loss) {
            if (verbosity > 2) {
              char s[1000];
              sprintf(s, "dumping \"loss\" file, %ld particles of %ld", nLost, beam->n_to_track);
              report_stats(stdout, s);
            }
            dump_scattered_loss_particles(&SDDS_loss, lostParticle, beam->original,
                                          NULL, nLost, weight, tsptr, eptr);
            if (occurenceSeen && !SDDS_Terminate(&SDDS_loss)) {
              SDDS_SetError("Problem terminating 'losses' file (finish_output)");
              SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
            }
            if (verbosity > 3) {
              char s[1000];
              sprintf(s, "Finished dumping \"loss\" file, %ld particles of %ld", n_left, beam->n_to_track);
              report_stats(stdout, s);
            }
          }

#if USE_MPI
        if (isMaster)
#endif
          if (output) {
            if (verbosity > 2) {
              report_stats(stdout, "Dumping \"output\" file:");
            }
            for (i = 0; i < nLost; i++) {
              j = lostParticle[i][6] - 1;
              /*
            if (verbosity>10)
              printf("s[%ld] = %e from w[%ld] = %e\n", i, lostParticle[i][4], j, weight[j]);
            */
#if USE_MPI
              /* In parallel mode, the master gathers lost particles only, so no offset */
              chfill1(lossDis, lostParticle[i][4], weight[j] * tsptr->total_scatter / tsptr->s_rate);
#else
            /* In serial mode, lost particles are stored in the upper part of the tracking array */
            chfill1(lossDis, beam->particle[i + n_left][4], weight[j] * tsptr->total_scatter / tsptr->s_rate);
#endif
            }
            chprint1(lossDis, tsptr->outFile, "Beam loss distribution in particles/s in a bin of sbin_step(m) for particles scattered between i-1 and i TSCATTER elements", "particles/s", NULL,
                     NULL, 0, 0, verbosity, noOccurenceSeen && iProcessing != 1);
            free_hbook1(lossDis);
            if (verbosity > 3) {
              report_stats(stdout, "Finished dumping \"output\" file:");
            }
          }
      }

      if (verbosity > 2) {
        report_stats(stdout, "Freeing beam data: ");
        fflush(stdout);
      }
      free_beamdata(beam);
      free_beamdata(beam0);
#if USE_MPI
      if (isMaster) {
        free_czarray_2d((void **)lostParticle, nLost, totalPropertiesPerParticle);
        lostParticle = NULL;
      }
#endif

      if (verbosity > 2) {
        report_stats(stdout, "Freeing other data: ");
      }
      free(weight);
      free(index);
      if (tsSpec->distIn == 1) {
        free_hbookn(tsptr->fullhis);
      } else if (tsSpec->distIn == 2) {
        free_hbookn(tsptr->thist);
        free_hbookn(tsptr->zhis);
      } else if (tsSpec->distIn == 3) {
        free_hbookn(tsptr->xhis);
        free_hbookn(tsptr->yhis);
        free_hbookn(tsptr->zhis);
      }
    }
#if USE_MPI
    if (isMaster)
#endif
      if (verbosity > 2) {
        printf("Advancing to next element\n");
        fflush(stdout);
      }
    elementIndex++;
    eptr = eptr->succ;
  }
  if (verbosity > 1)
    report_stats(stdout, "Main touschek loop completed: ");

  if (!occurenceSeen && !SDDS_Terminate(&SDDS_loss)) {
    SDDS_SetError("Problem terminating 'losses' file");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  if (!occurenceSeen && !SDDS_Terminate(&SDDS_bunch)) {
    SDDS_SetError("Problem terminating 'bunch' file");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }

  if (verbosity > 1)
    report_stats(stdout, "Touschek I/O completed: ");

  return;
}

/* Initialize beam parameter at each Scatter element */
TSCATTER *initTSCATTER(ELEMENT_LIST *eptr, long iElement) {
  TSCATTER *tsptr;
  double temp;
  long i;

  tsptr = ((TSCATTER *)eptr->p_elem);

  tsptr->betagamma = eptr->Pref_output;
  tsptr->gamma = sqrt(sqr(tsptr->betagamma) + 1.);
  tsptr->pCentral_mev = tsptr->gamma * me_mev;
  tsptr->s_rate = tsptr->i_rate = tsptr->totalWeight = 0;
  tsptr->simuCount = 0;
  tsptr->name = eptr->name;
  tsptr->s = eptr->end_pos;

  tsptr->twiss[0][0] = eptr->twiss->alphax;
  tsptr->twiss[0][1] = eptr->twiss->betax;
  tsptr->twiss[0][2] = (1. + sqr(eptr->twiss->alphax)) / eptr->twiss->betax;
  tsptr->twiss[1][0] = eptr->twiss->alphay;
  tsptr->twiss[1][1] = eptr->twiss->betay;
  tsptr->twiss[1][2] = (1. + sqr(eptr->twiss->alphay)) / eptr->twiss->betay;
  tsptr->twiss[2][0] = 0.0;
  tsptr->twiss[2][1] = tsSpec->sigz * eptr->Pref_output / tsSpec->delta_p0;
  tsptr->twiss[2][2] = 1. / tsptr->twiss[2][1];
  tsptr->disp[0][0] = eptr->twiss->etax;
  tsptr->disp[0][1] = eptr->twiss->etapx;
  tsptr->disp[1][0] = eptr->twiss->etay;
  tsptr->disp[1][1] = eptr->twiss->etapy;

  tsptr->sigx = sqrt(tsSpec->emitN[0] / tsptr->gamma * tsptr->twiss[0][1]);
  tsptr->sigy = sqrt(tsSpec->emitN[1] / tsptr->gamma * tsptr->twiss[1][1]);
  tsptr->sigz = tsSpec->sigz;
  tsptr->sigxyz = tsptr->sigx * tsptr->sigy * tsptr->sigz;

  temp = sqr(tsSpec->charge / e_mks) * sqr(PI) * sqr(re_mks) * c_mks / 4.;
  if (!tsSpec->distIn) {
    tsptr->xmin[0] = -(tsptr->xmax[0] = 0.5 * tsSpec->range[0] * tsptr->sigx);
    tsptr->xmin[1] = -(tsptr->xmax[1] = 0.5 * tsSpec->range[1] * tsptr->sigy);
    tsptr->xmin[2] = -(tsptr->xmax[2] = 0.5 * tsSpec->range[2] * tsptr->sigz);
    tsptr->xmin[3] = -(tsptr->xmax[3] = 0.5 * tsSpec->range[0] * sqrt(tsSpec->emitN[0] / tsptr->gamma / tsptr->twiss[0][1]) * (1. + fabs(tsptr->twiss[0][0])));
    tsptr->xmin[4] = -(tsptr->xmax[4] = 0.5 * tsSpec->range[1] * sqrt(tsSpec->emitN[1] / tsptr->gamma / tsptr->twiss[1][1]) * (1. + fabs(tsptr->twiss[1][0])));
    tsptr->xmin[0] = -(tsptr->xmax[0] += fabs(0.5 * tsSpec->range[2] * tsSpec->delta_p0 / tsptr->betagamma * tsptr->disp[0][0]));
    tsptr->xmin[1] = -(tsptr->xmax[1] += fabs(0.5 * tsSpec->range[2] * tsSpec->delta_p0 / tsptr->betagamma * tsptr->disp[1][0]));
    tsptr->xmin[3] = -(tsptr->xmax[3] += fabs(0.5 * tsSpec->range[2] * tsSpec->delta_p0 / tsptr->betagamma * tsptr->disp[0][1]));
    tsptr->xmin[4] = -(tsptr->xmax[4] += fabs(0.5 * tsSpec->range[2] * tsSpec->delta_p0 / tsptr->betagamma * tsptr->disp[1][1]));
    tsptr->xmin[5] = -(tsptr->xmax[5] = 0.5 * tsSpec->range[2] * tsSpec->delta_p0 / tsptr->betagamma);
  } else if (tsSpec->distIn == 1) {
    tsptr->fullhis = readbookn(FullDist, tsSpec->ipage_his[iElement - 1]);
    for (i = 0; i < 3; i++) {
      tsptr->xmin[i] = tsptr->fullhis->xmin[i * 2];
      tsptr->xmax[i] = tsptr->fullhis->xmax[i * 2];
      tsptr->xmin[i + 3] = tsptr->fullhis->xmin[i * 2 + 1];
      tsptr->xmax[i + 3] = tsptr->fullhis->xmax[i * 2 + 1];
    }
    tsptr->xmin[2] *= c_mks;
    tsptr->xmax[2] *= c_mks;
  } else if (tsSpec->distIn == 2) {
    tsptr->thist = readbookn(TranDist, tsSpec->ipage_his[iElement - 1]);
    tsptr->zhis = readbookn(ZDist, tsSpec->ipage_his[iElement - 1]);
    for (i = 0; i < 2; i++) {
      tsptr->xmin[i] = tsptr->thist->xmin[i * 2];
      tsptr->xmax[i] = tsptr->thist->xmax[i * 2];
      tsptr->xmin[i + 3] = tsptr->thist->xmin[i * 2 + 1];
      tsptr->xmax[i + 3] = tsptr->thist->xmax[i * 2 + 1];
    }
    tsptr->xmin[2] = tsptr->zhis->xmin[0] * c_mks;
    tsptr->xmax[2] = tsptr->zhis->xmax[0] * c_mks;
    tsptr->xmin[5] = tsptr->zhis->xmin[1];
    tsptr->xmax[5] = tsptr->zhis->xmax[1];
  } else if (tsSpec->distIn == 3) {
    tsptr->xhis = readbookn(XDist, tsSpec->ipage_his[iElement - 1]);
    tsptr->yhis = readbookn(YDist, tsSpec->ipage_his[iElement - 1]);
    tsptr->zhis = readbookn(ZDist, tsSpec->ipage_his[iElement - 1]);

    tsptr->xmin[0] = tsptr->xhis->xmin[0];
    tsptr->xmin[3] = tsptr->xhis->xmin[1];
    tsptr->xmin[1] = tsptr->yhis->xmin[0];
    tsptr->xmin[4] = tsptr->yhis->xmin[1];
    tsptr->xmin[2] = tsptr->zhis->xmin[0] * c_mks;
    tsptr->xmin[5] = tsptr->zhis->xmin[1];
    tsptr->xmax[0] = tsptr->xhis->xmax[0];
    tsptr->xmax[3] = tsptr->xhis->xmax[1];
    tsptr->xmax[1] = tsptr->yhis->xmax[0];
    tsptr->xmax[4] = tsptr->yhis->xmax[1];
    tsptr->xmax[2] = tsptr->zhis->xmax[0] * c_mks;
    tsptr->xmax[5] = tsptr->zhis->xmax[1];
  }

  if (tsSpec->distIn) {
    tsptr->factor = temp * (tsptr->xmax[0] - tsptr->xmin[0]) * sqr(tsptr->xmax[3] - tsptr->xmin[3]) * (tsptr->xmax[1] - tsptr->xmin[1]) * sqr(tsptr->xmax[4] - tsptr->xmin[4]) * (tsptr->xmax[2] - tsptr->xmin[2]) * sqr(tsptr->xmax[5] - tsptr->xmin[5]);
  } else {
    tsptr->factor = temp * pow(tsSpec->range[0], 3.0) * pow(tsSpec->range[1], 3.0) * pow(tsSpec->range[2], 3.0) / pow(2 * PI, 6.0) / tsptr->sigxyz;
  }

  return (tsptr);
}

/* Initialize Touschek Scattering Calculation */
void init_TSPEC(RUN *run, LINE_LIST *beamline, long nElement) {
  if (!(tsSpec = SDDS_Malloc(sizeof(*tsSpec))))
    bombElegant("memory allocation failure at setup Touscheck scatter", NULL);

  tsSpec->nbins = nbins;
  tsSpec->charge = charge;
  tsSpec->frequency = frequency;
  tsSpec->ignoredPortion = ignored_portion;
  tsSpec->emitN[0] = emit_nx;
  tsSpec->emitN[1] = emit_ny;
  tsSpec->emitN[2] = sigma_s * sigma_dp * sqrt(sqr(run->p_central) + 1.);
  tsSpec->range[0] = 2 * distribution_cutoff[0];
  tsSpec->range[1] = 2 * distribution_cutoff[1];
  tsSpec->range[2] = 2 * distribution_cutoff[2];
  tsSpec->sigz = sigma_s;
  tsSpec->delta_p0 = sigma_dp * run->p_central;

  tsSpec->distIn = 0;
  if (FullDist) {
    tsSpec->ipage_his = calloc(sizeof(long), nElement);
    tsSpec->distIn = 1;
    if (Check_HisInput(FullDist, beamline, nElement, tsSpec->ipage_his, 0) < 0)
      bombElegant("The input FullDist is not valid for this calculation - not same element location!", NULL);
  } else if (TranDist) {
    if (!ZDist) {
      printWarning("touschek_scatter: ZDist needs be given with TranDist.", "The input distribution file is ignored.");
      tsSpec->distIn = 0;
    } else {
      tsSpec->ipage_his = calloc(sizeof(long), nElement);
      tsSpec->distIn = 2;
      if (Check_HisInput(ZDist, beamline, nElement, tsSpec->ipage_his, 0) < 0 ||
          Check_HisInput(TranDist, beamline, nElement, tsSpec->ipage_his, 1) < 0)
        bombElegant("The input ZDist or TranDist is not valid for this calculation - not same element location!", NULL);
    }
  } else if (XDist || YDist || ZDist) {
    if (!XDist || !YDist || !ZDist) {
      printWarning("touschek_scatter: [XYZ]Dist need be given at the same time.", "The input distribution file is ignored.");
      tsSpec->distIn = 0;
    } else {
      tsSpec->ipage_his = calloc(sizeof(long), nElement);
      tsSpec->distIn = 3;
      if (Check_HisInput(XDist, beamline, nElement, tsSpec->ipage_his, 0) < 0 ||
          Check_HisInput(YDist, beamline, nElement, tsSpec->ipage_his, 1) < 0 ||
          Check_HisInput(ZDist, beamline, nElement, tsSpec->ipage_his, 1) < 0)
        bombElegant("The input ZDist or TranDist is not valid for this calculation - not same element location!", NULL);
    }
  }

  return;
}

/* Check validation of histogram input file, flag>1 for second file */
long Check_HisInput(char *filename, LINE_LIST *beamline, long nElement, long *pIndex, long flag) {
  long result;
  long i, iPage, Occurence;
  char **Name, **Type;
  ELEMENT_LIST *eptr;
  SDDS_DATASET input;
  double s, eps = 1e-7;

  if (!SDDS_InitializeInput(&input, filename))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  i = 0;
  result = -nElement;
  if (!(eptr = beamline->elem_recirc))
    eptr = beamline->elem;
  while (eptr) {
    if (eptr->type == T_TSCATTER) {
      while (1) {
        if ((iPage = SDDS_ReadPage(&input)) <= 0)
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        if (!(Name = (char **)SDDS_GetParameter(&input, "ElementName", NULL)))
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        if (!(Type = (char **)SDDS_GetParameter(&input, "ElementType", NULL)))
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        if (!SDDS_GetParameter(&input, "ElementOccurence", (void *)&Occurence))
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        if (!SDDS_GetParameter(&input, "s", (void *)&s))
          SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
        if (fabs(s - eptr->end_pos) < eps &&
            Occurence == eptr->occurence &&
            strcmp(*Name, eptr->name) == 0 &&
            strcmp(*Type, entity_name[eptr->type]) == 0)
          break;
      }
      if (!flag)
        pIndex[i] = iPage;
      else if (pIndex[i] != iPage)
        bombElegant("The input ZDist and TranDist file has different element location!", NULL);
      i++;
      result++;
    }
    eptr = eptr->succ;
  }

  if (!SDDS_Terminate(&input))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  return result;
}

/* Get Momentum Aperture Input */
long get_MAInput(char *filename, LINE_LIST *beamline, long nElement) {
  long result, i, iTotal;
  ELEMENT_LIST *eptr;
  SDDS_DATASET input;
  char **Name = NULL, **Type = NULL;
  double *s = NULL, *dpp = NULL, *dpm = NULL, eps = 1e-7;
  int32_t *Occurence = NULL;

  if (!SDDS_InitializeInput(&input, filename))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  if (SDDS_ReadPage(&input) <= 0)
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  iTotal = SDDS_CountRowsOfInterest(&input);
  if (!(s = SDDS_GetColumnInDoubles(&input, "s")) ||
      !(dpp = SDDS_GetColumnInDoubles(&input, "deltaPositive")) ||
      !(dpm = SDDS_GetColumnInDoubles(&input, "deltaNegative")) ||
      !(Occurence = SDDS_GetColumnInLong(&input, "ElementOccurence")) ||
      !(Name = (char **)SDDS_GetColumn(&input, "ElementName")) ||
      !(Type = (char **)SDDS_GetColumn(&input, "ElementType"))) {
    bombElegant("Momentum aperture file lacks required columns (s, deltaPositive, deltaNegative, ElementOccurence, ElementName, ElementType)", NULL);
  }

  for (i = 1; i < iTotal; i++)
    if (s[i - 1] > s[i])
      bombElegant("Momentum aperture data is not sorted in order of increasing s.", NULL);

  i = 0;
  result = -nElement;
  if (!(eptr = beamline->elem_recirc))
    eptr = beamline->elem;

  while (eptr) {
    if (eptr->type == T_TSCATTER) {
      if (verbosity > 1) {
        printf("Looking for data for TSCATTER %s#%ld at s=%21.15e\n",
               eptr->name, eptr->occurence, eptr->end_pos);
        fflush(stdout);
      }
      while (1) {
        if (fabs(*s - eptr->end_pos) < eps &&
            (match_position_only ||
             (*Occurence == eptr->occurence &&
              strcmp(*Name, eptr->name) == 0 &&
              strcmp(*Type, entity_name[eptr->type]) == 0))) {
          ((TSCATTER *)eptr->p_elem)->deltaN = (*dpm) * Momentum_Aperture_scale;
          ((TSCATTER *)eptr->p_elem)->deltaP = (*dpp) * Momentum_Aperture_scale;
          if (verbosity > 1) {
            printf("Matched by \"%s\"#%ld at s=%21.15e\n",
                   *Name, (long)*Occurence, *s);
            fflush(stdout);
          }
          i++;
          s++;
          dpp++;
          dpm++;
          Name++;
          Type++;
          Occurence++;
          break;
        }
        s++;
        dpp++;
        dpm++;
        Name++;
        Type++;
        Occurence++;
        if (i++ > iTotal) {
          bombElegantVA("Momentum aperture file ends earlier than required---no data for %s#%ld at s=%lem\n",
                        eptr->name, eptr->occurence, eptr->end_pos);
        }
      }
      result++;
    }
    eptr = eptr->succ;
  }

  if (!SDDS_Terminate(&input))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  return result;
}

/************************************************************************************\
 * Modified from S. Khan's code.                                                    *
 *      select two electrons (labelled a and b) for Touschek scattering             *
 * p1[x,y,s,xp,yp,dp/p]                                                             *
\************************************************************************************/

void selectPartGauss(TSCATTER *tsptr, double *p1, double *p2,
                     double *dens1, double *dens2, double *ran1) {
  int i;
  double U[3], V1[3], V2[3], densa[3], densb[3];

  /* Select random particle's parameter in normal phase space */
  for (i = 0; i < 3; i++) {
    U[i] = (ran1[i] - 0.5) * tsSpec->range[i] * sqrt(tsSpec->emitN[i] / tsptr->gamma);
    V1[i] = (ran1[i + 3] - 0.5) * tsSpec->range[i] * sqrt(tsSpec->emitN[i] / tsptr->gamma);
    V2[i] = (ran1[i + 6] - 0.5) * tsSpec->range[i] * sqrt(tsSpec->emitN[i] / tsptr->gamma);
    densa[i] = exp(-0.5 * (U[i] * U[i] + V1[i] * V1[i]) / tsSpec->emitN[i] * tsptr->gamma);
  }
  densb[2] = exp(-0.5 * (U[2] * U[2] + V2[2] * V2[2]) / tsSpec->emitN[2] * tsptr->gamma);
  /* Change particle's parameter from normal phase space to real phase space */
  for (i = 0; i < 3; i++) {
    p1[i] = p2[i] = sqrt(tsptr->twiss[i][1]) * U[i];
    p1[i + 3] = (V1[i] - tsptr->twiss[i][0] * U[i]) / sqrt(tsptr->twiss[i][1]);
    p2[i + 3] = (V2[i] - tsptr->twiss[i][0] * U[i]) / sqrt(tsptr->twiss[i][1]);
  }
  /* Dispersion correction */
  p1[0] = p1[0] + p1[5] * tsptr->disp[0][0];
  p1[1] = p1[1] + p1[5] * tsptr->disp[1][0];
  p1[3] = p1[3] + p1[5] * tsptr->disp[0][1];
  p1[4] = p1[4] + p1[5] * tsptr->disp[1][1];

  p2[0] = p1[0] - p2[5] * tsptr->disp[0][0];
  p2[1] = p1[1] - p2[5] * tsptr->disp[1][0];
  U[0] = p2[0] / sqrt(tsptr->twiss[0][1]);
  U[1] = p2[1] / sqrt(tsptr->twiss[1][1]);
  p2[3] = (V2[0] - tsptr->twiss[0][0] * U[0]) / sqrt(tsptr->twiss[0][1]);
  p2[4] = (V2[1] - tsptr->twiss[1][0] * U[1]) / sqrt(tsptr->twiss[1][1]);
  densb[0] = exp(-0.5 * (U[0] * U[0] + V2[0] * V2[0]) / tsSpec->emitN[0] * tsptr->gamma);
  densb[1] = exp(-0.5 * (U[1] * U[1] + V2[1] * V2[1]) / tsSpec->emitN[1] * tsptr->gamma);

  p2[0] = p1[0];
  p2[1] = p1[1];
  p2[3] = p2[3] + p2[5] * tsptr->disp[0][1];
  p2[4] = p2[4] + p2[5] * tsptr->disp[1][1];

  *dens1 = densa[0] * densa[1] * densa[2];
  *dens2 = densb[0] * densb[1] * densb[2];

  return;
}

/* select particle from input particle distribution function */
void selectPartReal(TSCATTER *tsptr, double *p1, double *p2,
                    double *dens1, double *dens2, double *ran1) {
  double x10[6], x20[6], x1[6], x2[6];
  long i;

  *dens1 = (*dens2 = 0);
  for (i = 0; i < 6; i++)
    x10[i] = (x20[i] = ran1[i]);
  x20[1] = ran1[6];
  x20[3] = ran1[7];
  x20[5] = ran1[8];

  if (tsSpec->distIn == 1) {
    *dens1 = interpolate_bookn(tsptr->fullhis, x10, x1, 0, 1, 1, 1, 0);
    *dens2 = interpolate_bookn(tsptr->fullhis, x20, x2, 0, 1, 1, 1, 0);
  }
  if (tsSpec->distIn == 2) {
    *dens1 = interpolate_bookn(tsptr->thist, x10, x1, 0, 1, 1, 1, 0);
    *dens2 = interpolate_bookn(tsptr->thist, x20, x2, 0, 1, 1, 1, 0);
    *dens1 *= interpolate_bookn(tsptr->zhis, x10, x1, 4, 1, 1, 1, 0);
    *dens2 *= interpolate_bookn(tsptr->zhis, x20, x2, 4, 1, 1, 1, 0);
  }
  if (tsSpec->distIn == 3) {
    *dens1 = interpolate_bookn(tsptr->xhis, x10, x1, 0, 1, 1, 1, 0);
    *dens2 = interpolate_bookn(tsptr->xhis, x20, x2, 0, 1, 1, 1, 0);
    *dens1 *= interpolate_bookn(tsptr->yhis, x10, x1, 2, 1, 1, 1, 0);
    *dens2 *= interpolate_bookn(tsptr->yhis, x20, x2, 2, 1, 1, 1, 0);
    *dens1 *= interpolate_bookn(tsptr->zhis, x10, x1, 4, 1, 1, 1, 0);
    *dens2 *= interpolate_bookn(tsptr->zhis, x20, x2, 4, 1, 1, 1, 0);
  }

  for (i = 0; i < 3; i++) {
    p1[i] = x1[i * 2];
    p1[i + 3] = x1[i * 2 + 1];
    p2[i] = x2[i * 2];
    p2[i + 3] = x2[i * 2 + 1];
  }
  p1[2] = (p2[2] *= c_mks);
  *dens1 /= c_mks;
  *dens2 /= c_mks;

  return;
}

/* This function transfer particle's momentum from lab system to c.o.m system */
/* p1[3]=p1x*c, p1[4]=p1y*c, p1[5]=p1z*c in MeV */
/* p2[3]=p2x*c, p2[4]=p2y*c, p2[5]=p1z*c in MeV */
void bunch2cm(double *p1, double *p2, double *q, double *beta, double *gamma) {
  double pp1, pp2, e1, e2, ee;
  int i;
  double bb, betap1, factor;

  pp1 = 0.0;
  pp2 = 0.0;
  for (i = 3; i < 6; i++) {
    pp1 = pp1 + sqr(p1[i]);
    pp2 = pp2 + sqr(p2[i]);
  }
  e1 = sqrt(me_mev * me_mev + pp1);
  e2 = sqrt(me_mev * me_mev + pp2);
  ee = e1 + e2;

  betap1 = 0.0;
  bb = 0.0;
  for (i = 0; i < 3; i++) {
    beta[i] = (p1[i + 3] + p2[i + 3]) / ee;
    betap1 = betap1 + beta[i] * p1[i + 3];
    bb = bb + beta[i] * beta[i];
  }

  *gamma = 1. / sqrt(1. - bb);
  factor = ((*gamma) - 1.) * betap1 / bb;

  for (i = 0; i < 3; i++) {
    q[i] = p1[i + 3] + factor * beta[i] - (*gamma) * e1 * beta[i];
  }

  return;
}

/* Rotate scattered p in c.o.m system */
void eulertrans(double *v0, double theta, double phi, double *v1, double *v) {
  double th, ph, s1, s2, c1, c2;
  double x0, y0, z0;

  *v = sqrt(v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2]);
  th = acos(v0[2] / (*v));
  ph = atan2(v0[1], v0[0]);

  s1 = sin(th);
  s2 = sin(ph);
  c1 = cos(th);
  c2 = cos(ph);

  x0 = cos(theta);
  y0 = sin(theta) * cos(phi);
  z0 = sin(theta) * sin(phi);

  v1[0] = (*v) * (s1 * c2 * x0 - s2 * y0 - c1 * c2 * z0);
  v1[1] = (*v) * (s1 * s2 * x0 + c2 * y0 - c1 * s2 * z0);
  v1[2] = (*v) * (c1 * x0 + s1 * z0);

  return;
}

void cm2bunch(double *p1, double *p2, double *q, double *beta, double *gamma) {
  int i;
  double pq, e, betaq, bb, factor;

  pq = 0.0;
  for (i = 0; i < 3; i++) {
    pq = pq + q[i] * q[i];
  }

  e = sqrt(me_mev * me_mev + pq);

  betaq = 0.0;
  bb = 0.0;
  for (i = 0; i < 3; i++) {
    betaq = betaq + beta[i] * q[i];
    bb = bb + beta[i] * beta[i];
  }

  factor = ((*gamma) - 1) * betaq / bb;
  for (i = 0; i < 3; i++) {
    p1[i + 3] = q[i] + (*gamma) * beta[i] * e + factor * beta[i];
    p2[i + 3] = -q[i] + (*gamma) * beta[i] * e - factor * beta[i];
  }

  return;
}

double moeller(double beta0, double theta) {
  double cross;
  double beta2, st2;

  beta2 = beta0 * beta0;
  st2 = sqr(sin(theta));

  cross = (1. - beta2) * (sqr(1. + 1. / beta2) * (4. / st2 / st2 - 3. / st2) + 1. + 4. / st2);

  return cross;
}

void pickPart(double *weight, long *index, long start, long end,
              long *iTotal, double *wTotal, double weight_limit, double weight_ave) {
  long i, i1, i2, N;
  double w1, w2;
  long *index1, *index2;
  double *weight1, *weight2;

  i1 = i2 = 0;
  w1 = w2 = 0.;
  N = end - start;
  if (N < 3)
    return; /* scattered particles normally appear in pair */
  index2 = (long *)malloc(sizeof(long) * N);
  weight2 = (double *)malloc(sizeof(double) * N);
  index1 = (long *)malloc(sizeof(long) * N);
  weight1 = (double *)malloc(sizeof(double) * N);

  for (i = start; i < end; i++) {
    if (weight[i] > weight_ave) {
      weight2[i2] = weight[i];
      index2[i2++] = index[i];
      w2 += weight[i];
    } else {
      weight1[i1] = weight[i];
      index1[i1++] = index[i];
      w1 += weight[i];
    }
  }
  if ((w2 + (*wTotal)) > weight_limit) {
    weight_ave = w2 / (double)i2;
    for (i = 0; i < i2; i++) {
      index[start + i] = index2[i];
      weight[start + i] = weight2[i];
    }
    free(weight1);
    free(index1);
    free(weight2);
    free(index2);
    pickPart(weight, index, start, start + i2,
             iTotal, wTotal, weight_limit, weight_ave);
    return;
  }

  *iTotal += i2;
  *wTotal += w2;
  weight_ave = w1 / (double)i1;
  for (i = 0; i < i2; i++) {
    index[start + i] = index2[i];
    weight[start + i] = weight2[i];
  }
  for (i = 0; i < i1; i++) {
    index[start + i2 + i] = index1[i];
    weight[start + i2 + i] = weight1[i];
  }
  free(weight1);
  free(index1);
  free(weight2);
  free(index2);
  pickPart(weight, index, i2 + start, end,
           iTotal, wTotal, weight_limit, weight_ave);
  return;
}

#if defined(_WIN32)
short has_occurence_string(char *template);
#else
#  include <sys/types.h>
#  include <regex.h>

short has_occurence_string(char *template) {
  static regex_t regExpr;
  static short first = 1;
  if (first) {
    int code;
    if ((code = regcomp(&regExpr, ".*%[0-9]*ld.*", REG_NOSUB))) {
      char errbuf[1000];
      regerror(code, &regExpr, errbuf, (size_t)1000);
      printf("%s\n", errbuf);
      bombElegant("problem compiling regular expression for filename occurence determination (has_occurence_string)", NULL);
    }
    first = 0;
  }
  return !regexec(&regExpr, template, 0, NULL, 0);
}
#endif

void determineOccurenceInFilenames(short *occurenceSeen, short *noOccurenceSeen,
                                   char *initial, char *distribution, char *output, char *loss, char *bunch) {
  if (initial) {
    if (has_occurence_string(initial))
      *occurenceSeen = 1;
    else
      *noOccurenceSeen = 1;
  }
  if (distribution) {
    if (has_occurence_string(distribution))
      *occurenceSeen = 1;
    else
      *noOccurenceSeen = 1;
  }
  if (output) {
    if (has_occurence_string(output))
      *occurenceSeen = 1;
    else
      *noOccurenceSeen = 1;
  }
  if (loss) {
    if (has_occurence_string(loss))
      *occurenceSeen = 1;
    else
      *noOccurenceSeen = 1;
  }
  if (bunch) {
    if (has_occurence_string(bunch))
      *occurenceSeen = 1;
    else
      *noOccurenceSeen = 1;
  }
}
