/*************************************************************************\
 * Copyright (c) 2002 The University of Chicago, as Operator of Argonne
 * National Laboratory.
 * Copyright (c) 2002 The Regents of the University of California, as
 * Operator of Los Alamos National Laboratory.
 * This file is distributed subject to a Software License Agreement found
 * in the file LICENSE that is included with this distribution. 
\*************************************************************************/

#include "mdb.h"
//#include "mdbsun.h"
#include "track.h"
#ifdef USE_GSL
#  include "gsl/gsl_poly.h"
#endif

long transformBeamWithScript(SCRIPT *script, double pCentral, CHARGE *charge,
                             BEAM *beam, double **part, int64_t np,
                             char *mainRootname, long iPass, long driftOrder, double z,
                             long forceSerial, long occurence, long backtrack,
			     LINE_LIST *beamline, RUN *run) {
  short doDrift;
  long npReturn;
  doDrift = 0;
  npReturn = 0;
  
  if (script->onPass >= 0) {
    if (script->onPass != iPass)
      doDrift = 1;
  } else {
    if (script->startPass >= 0 && script->startPass > iPass)
      doDrift = 1;
    if (script->endPass >= 0 && iPass > script->endPass)
      doDrift = 1;
    if (script->passInterval > 0) {
      if (script->startPass < 0)
        bombElegant("Problem with script element: START_PASS<0 but PASS_INTERVAL>0", NULL);
      if ((iPass - script->startPass) % script->passInterval != 0)
        doDrift = 1;
    }
  }

  if (doDrift) {
    drift_beam(part, np, script->length, driftOrder);
    npReturn = np;
  } else {
#if USE_MPI
#  if MPI_DEBUG
    printf("In transformBeamWithScript, forceSerial=%ld\n", forceSerial);
    fflush(stdout);
#  endif
    if (!forceSerial) {
      if (script->useParticleID && script->determineLossesFromParticleID) {
	if (isMaster)
	  printWarningForTracking("The DETERMINE_LOSSES_FROM_PID flag of the SCRIPT element is ignored in Pelegant.", NULL);
      }
      npReturn = transformBeamWithScript_p(script, pCentral, charge, beam, part, np, mainRootname, iPass, driftOrder, z, occurence, backtrack);
    } else {
      npReturn = transformBeamWithScript_s(script, pCentral, charge, beam, part, np, mainRootname, iPass, driftOrder, z, occurence, backtrack);
    }
#else
    npReturn = transformBeamWithScript_s(script, pCentral, charge, beam, part, np, mainRootname, iPass, driftOrder, z, occurence, backtrack);
#endif
  }
  
  if (script->postCommandParameterFile && strlen(script->postCommandParameterFile) && beamline && run) {
    /* load data from the file into element parameters */
    do_load_parameters(beamline, 0, script->postCommandParameterFile);
    /* update matrices etc. */
    fill_in_matrices(beamline->elem, run);
  }

  return npReturn;
}

void determineScriptNames(SCRIPT *script, char **rootname0, char **input0, char **output0, long *nameLength0,
                          char *mainRootname, long forceSerial, long pass, long occurence) {
  char *rootname = NULL, *input, *output = NULL;
  long nameLength;
#if USE_MPI
  long rootnameLength;
#endif

  if (!script->rootname || !strlen(script->rootname)) {
    /* generate random rootname */
#ifdef USE_MPI
    if (isMaster && !(rootname = tmpname(NULL)))
      bombElegant("problem generating temporary filename for script", NULL);
#  if USE_MPI
    if (isMaster)
      rootnameLength = strlen(rootname) + 1;
    if (!forceSerial) {
      MPI_Bcast(&rootnameLength, 1, MPI_LONG, 0, MPI_COMM_WORLD);
      if (isSlave) /* Master and slave could have different rootname length if use C-shell, which calls tmpname one more time on master for execution.*/
        rootname = malloc(rootnameLength);
      /* As different processors will have different process names, we need
         make sure they have the same file name for parallel I/O */
      MPI_Bcast(rootname, rootnameLength, MPI_CHAR, 0, MPI_COMM_WORLD);
#    if MPI_DEBUG
      printf("rootname = %s\n", rootname);
#    endif
    }
#  endif
#else
    if (!(rootname = tmpname(NULL)))
      bombElegant("problem generating temporary filename for script", NULL);
#endif
  } else {
    char *rootname2;
    char buffer[100];
    rootname = tmalloc(sizeof(*rootname) * (2 * strlen(script->rootname) + 100));
    rootname2 = tmalloc(sizeof(*rootname) * (2 * strlen(script->rootname) + 100));
    strcpy(rootname, script->rootname);
    sprintf(buffer, "%06ld", pass);
    replaceString(rootname2, rootname, "%p", buffer, 1, 0);
    replaceString(rootname, rootname2, "%s", mainRootname, 1, 0);
    sprintf(rootname2, rootname, occurence);
    free(rootname);
    rootname = rootname2;
  }
  if (!rootname)
    bombElegant("problem generating temporary rootname for script", NULL);
  nameLength = (script->directory ? strlen(script->directory) : 0) +
               strlen(rootname) + strlen(script->inputExtension) +
               strlen(script->outputExtension) + 4;
  if (!(input = malloc(sizeof(*input) * nameLength)) ||
      !(output = malloc(sizeof(*output) * nameLength)))
    bombElegant("problem generating temporary filename for script", NULL);

  *rootname0 = rootname;
  *input0 = input;
  *output0 = output;
  *nameLength0 = nameLength;
}

void prepareScriptCommand(SCRIPT *script, long iPass, char *rootname, char *input, char *output, char **cmdBufferRet, long occurence,
                          long backtrack) {
  char *cmdBuffer0, *cmdBuffer1 = NULL;
  long i;
  char passString[20], occurenceString[20];

  /* prepare command */
  if (script->directory && strlen(script->directory)) {
#if defined(_WIN32)
    sprintf(input, "%s\\%s.%s", script->directory, rootname, script->inputExtension);
    sprintf(output, "%s\\%s.%s", script->directory, rootname, script->outputExtension);
#else
    sprintf(input, "%s/%s.%s", script->directory, rootname, script->inputExtension);
    sprintf(output, "%s/%s.%s", script->directory, rootname, script->outputExtension);
#endif
  } else {
    sprintf(input, "%s.%s", rootname, script->inputExtension);
    sprintf(output, "%s.%s", rootname, script->outputExtension);
  }
  if (rootname != script->rootname)
    free(rootname);

  sprintf(passString, "%ld", iPass);

  if (!(cmdBuffer0 = malloc(sizeof(char) * (strlen(script->command) + 10 * strlen(input) + 10 * strlen(output) + strlen(passString)))) ||
      !(cmdBuffer1 = malloc(sizeof(char) * (strlen(script->command) + 10 * strlen(input) + 10 * strlen(output) + strlen(passString)))))
    bombElegant("memory allocation failure making command buffer for script", NULL);
  replaceString(cmdBuffer0, script->command, "%i", input, 9, 0);
  replaceString(cmdBuffer1, cmdBuffer0, "%o", output, 9, 0);
  replaceString(cmdBuffer0, cmdBuffer1, "%p", passString, 9, 0);
  replaceString(cmdBuffer1, cmdBuffer0, "%b", backtrack ? "1" : "0", 9, 0);
  sprintf(occurenceString, "%ld", occurence);
  replaceString(cmdBuffer0, cmdBuffer1, "%c", occurenceString, 9, 0);
  strcpy(cmdBuffer1, cmdBuffer0);

  /* substitute numerical parameters */
  for (i = 0; i < 10; i++) {
    long count = 0;
    char tag[10], value[25], *ptr;
    sprintf(tag, "%%np%ld", i);
    ptr = cmdBuffer1;
    while ((ptr = strstr(ptr, tag))) {
      count++;
      ptr += 3;
    }
    if (!count)
      continue;
    sprintf(value, "%21.15e", script->NP[i]);
    if (!(cmdBuffer0 = SDDS_Realloc(cmdBuffer0, sizeof(*cmdBuffer1) * (strlen(cmdBuffer1) + count * 25 + 1))) ||
        !(cmdBuffer1 = SDDS_Realloc(cmdBuffer1, sizeof(*cmdBuffer1) * (strlen(cmdBuffer1) + count * 25 + 1))))
      SDDS_Bomb("memory allocation failure");
    replaceString(cmdBuffer0, cmdBuffer1, tag, value, count, 0);
    strcpy_ss(cmdBuffer1, cmdBuffer0);
  }
  /* substitute string parameters */
  for (i = 0; i < 10; i++) {
    long count = 0;
    char tag[10], *ptr;
    if (!script->SP[i] || strlen(script->SP[i]) == 0)
      continue;
    sprintf(tag, "%%sp%ld", i);
    ptr = cmdBuffer1;
    while ((ptr = strstr(ptr, tag))) {
      count++;
      ptr += 3;
    }
    if (!count)
      continue;
    if (!(cmdBuffer0 =
            SDDS_Realloc(cmdBuffer0, sizeof(*cmdBuffer1) * (strlen(cmdBuffer1) + count * strlen(script->SP[i]) + 1))) ||
        !(cmdBuffer1 =
            SDDS_Realloc(cmdBuffer1, sizeof(*cmdBuffer1) * (strlen(cmdBuffer1) + count * strlen(script->SP[i]) + 1))))
      SDDS_Bomb("memory allocation failure");
    replaceString(cmdBuffer0, cmdBuffer1, tag, script->SP[i], count, 0);
    strcpy_ss(cmdBuffer1, cmdBuffer0);
  }
  interpret_escaped_quotes(cmdBuffer1);

  free(cmdBuffer0);
  *cmdBufferRet = cmdBuffer1;
}

long transformBeamWithScript_s(SCRIPT *script, double pCentral, CHARGE *charge,
                               BEAM *beam, double **part, int64_t np,
                               char *mainRootname, long iPass, long driftOrder, double z,
                               long occurence, long backtrack) {
  char *rootname = NULL, *input, *output = NULL;
  char *cmdBuffer1 = NULL;
  SDDS_DATASET SDDSout, SDDSin;
  double *data = NULL;
  char *dataname[6] = {"x", "xp", "y", "yp", "t", "p"};
  long i, j, npNew, nameLength;
  long k;
  double *pID = NULL;
  short failSoftly = 0;
  TRACKING_CONTEXT trackingContext;
  double **lostParticle;
  int64_t nLost;
  long nLost2;

  if (np == 0)
    return 0;
  getTrackingContext(&trackingContext);

  determineScriptNames(script, &rootname, &input, &output, &nameLength, mainRootname, 1, iPass, occurence);
  if (script->verbosity > 0) {
    printf("SCRIPT rootname: %s\n", rootname);
    fflush(stdout);
  }
  prepareScriptCommand(script, iPass, rootname, input, output, &cmdBuffer1, occurence, backtrack);
  if (script->verbosity > 0) {
    printf("%s\n", cmdBuffer1);
    fflush(stdout);
  }

  /* dump the data to script input file */
  SDDS_ForceInactive(&SDDSout);
  SDDS_PhaseSpaceSetup(&SDDSout, input, SDDS_BINARY, 1, "script input",
                       "unknown", "unknown",
                       "transformBeamWithScript");
  dump_phase_space(&SDDSout, part, np, 0, pCentral, charge ? charge->macroParticleCharge * np : 0.0, beam ? beam->id_slots_per_bunch : 0);

  if (!SDDS_Terminate(&SDDSout))
    SDDS_Bomb("problem terminating script input file");

  /* copy the lost particles to a separate buffer so we can easily put them back */
  if (beam && beam->n_lost) {
    int64_t il;
    lostParticle = (double **)czarray_2d(sizeof(double), beam->n_lost, totalPropertiesPerParticle);
    for (il = 0; il < beam->n_lost; il++)
      memcpy(lostParticle[il], beam->particle[np + il], totalPropertiesPerParticle * sizeof(double));
    nLost = beam->n_lost;
  } else {
    lostParticle = NULL;
    nLost = 0;
  }
  nLost2 = 0;

#if defined(CONDOR_COMPILE)
  _condor_ckpt_disable();
#endif

  /* run the script */
  if (script->useCsh)
    executeCshCommand(cmdBuffer1, mainRootname);
  else
    if (!system(cmdBuffer1)) {
    }

#if defined(CONDOR_COMPILE)
  _condor_ckpt_enable();
#endif

  if (script->verbosity > 0) {
    printf("Command completed\n");
    fflush(stdout);
  }

  /* read the data from script output file */
  if (!fexists(output)) {
    if (!script->softFailure)
      SDDS_Bomb("unable to find script output file");
    failSoftly = 1;
  }
  if (!SDDS_InitializeInput(&SDDSin, output)) {
    if (!script->softFailure) {
      SDDS_SetError("Unable to read script output file");
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    }
    failSoftly = 1;
  }

  if (!failSoftly) {
    if (!check_sdds_column(&SDDSin, "x", "m") ||
        !check_sdds_column(&SDDSin, "y", "m") ||
        !check_sdds_column(&SDDSin, "xp", NULL) ||
        !check_sdds_column(&SDDSin, "yp", NULL) ||
        !check_sdds_column(&SDDSin, "p", "m$be$nc") ||
        !check_sdds_column(&SDDSin, "t", "s")) {
      if (!check_sdds_column(&SDDSin, "p", "m$be$nc") &&
          check_sdds_column(&SDDSin, "p", NULL)) {
        printWarningForTracking("p has no units in script output file.", "Expected m$be$nc.");
      } else {
        printf("Error: necessary data quantities (x, x', y, y', t, p) have the wrong units or are not present in script output");
        exitElegant(1);
      }
    }

    if (SDDS_ReadPage(&SDDSin) != 1) {
      SDDS_SetError("Unable to read script output file");
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
      return 0;
    }
    npNew = SDDS_RowCount(&SDDSin);
  } else {
    npNew = 0;
  }

  if (npNew > np)
    if (script->noNewParticles)
      bombElegant("The number of particles increased after the SCRIPT element, even though NO_NEW_PARTICLES=0.", NULL);

  if (script->verbosity > 0) {
    if (failSoftly)
      printf("Problem reading script output file---all particles considered lost (SOFT_FAILURE=1)\n");
    else
      printf("%ld particles in script output file (was %ld)\n", npNew, np);
    fflush(stdout);
  }

  pID = NULL;
  if (npNew && script->useParticleID) {
    /* User wants us to use the particleID data from the script output file */
    if (!(pID = SDDS_GetColumnInDoubles(&SDDSin, "particleID"))) {
      SDDS_SetError("Unable to read particleID from script output file. Please set USE_PARTICLE_ID=0 if this is desired.\n");
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    }
    if (script->determineLossesFromParticleID) {
      long found, itop;
      double *pID2;
      pID2 = tmalloc(sizeof(double) * npNew);
      memcpy(pID2, pID, sizeof(double) * npNew);
      qsort((void *)pID2, npNew, sizeof(*pID2), double_cmpasc);
      found = 0;
      for (k = 1; k < npNew; k++)
        if (pID2[k - 1] == pID2[k])
          found++;
      if (found)
        printWarning("Duplicate particleID values in script output file.", "Loss accounting is suspect!");
      free(pID2);
      /* Figure out which particles if any were lost by matching particleID from the input and output */
      /* Otherwise, we just load the particleID data with no loss accounting. */
      itop = np - 1;
      for (j = 0; j <= itop; j++) {
        found = 0;
        for (k = 0; k < npNew; k++) {
          if (part[j][6] == pID[k]) {
            found = 1;
            break;
          }
        }
        if (!found) {
          /* particle was lost */
          swapParticles(part[itop], part[j]);
          part[itop][4] = z;
          part[itop][5] = pCentral * (1 + part[itop][5]);
          nLost2++;
          itop--;
          j--;
        }
      }
      if (script->verbosity > 1)
        printf("%ld particles of %ld lost based on particleID matching\n", nLost2, np);
      if (beam && nLost2) {
        long il;
        if (lostParticle) {
          lostParticle = (double **)resize_czarray_2d((void **)lostParticle, sizeof(double),
                                                      nLost2 + nLost, totalPropertiesPerParticle);
          for (il = 0; il < nLost2; il++) {
            memcpy(lostParticle[nLost + il], part[itop + 1 + il], totalPropertiesPerParticle * sizeof(double));
            lostParticle[nLost + il][lossPassIndex] = iPass;
          }
        } else {
          lostParticle = (double **)czarray_2d(sizeof(double), nLost2, totalPropertiesPerParticle);
          for (il = 0; il < nLost2; il++) {
            memcpy(lostParticle[il], part[itop + 1 + il], totalPropertiesPerParticle * sizeof(double));
            lostParticle[il][lossPassIndex] = iPass;
          }
        }
      }
    }
  }

  /* TODO: should this be np+nLost ? */
  if ((npNew + nLost + nLost2) > np) {
    printf("Resizing the array in transformBeamWithScript_s\n");
    fflush(stdout);
    /* Resize the input array, preserving the data */
    part = (double **)resize_czarray_2d((void **)part, sizeof(double),
                                        npNew + nLost + nLost2, totalPropertiesPerParticle);
    if (beam) {
      beam->particle = part;
      beam->n_particle = npNew;
    }
  }

  /* copy phase space coordinates */
  if (npNew) {
    for (i = 0; i < 6; i++) {
      if (!(data = SDDS_GetColumnInDoubles(&SDDSin, dataname[i]))) {
        SDDS_SetError("Unable to read script output file");
        SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
      }
      for (j = 0; j < npNew; j++)
        part[j][i] = data[j];
      free(data);
      data = NULL;
    }
  }

  /* fill in lost particle data */
  if (beam && lostParticle && (nLost || nLost2)) {
    long il;
    for (il = 0; il < (nLost + nLost2); il++)
      memcpy(part[npNew + il], lostParticle[il], sizeof(double) * totalPropertiesPerParticle);
    beam->n_lost = nLost + nLost2;
    free_czarray_2d((void **)lostParticle, nLost + nLost2, totalPropertiesPerParticle);
    lostParticle = NULL;
  }

  /* copy or create particle ID */
  if (pID) {
    /* The new particle IDs replace the pre-existing ones */
    for (j = 0; j < npNew; j++)
      part[j][6] = pID[j];
    free(pID);
    pID = NULL;
  } else {
    /* No particle ID data in the file, so generate some */
    for (j = 0; j < npNew; j++)
      part[j][6] = j + 1;
    printf("Changing particle ID for new particles to make them sequential\n");
  }

  /* update bunch assignments using PID */
  if (beam && beam->id_slots_per_bunch > 0)
    for (j = 0; j < npNew; j++)
      part[j][bunchIndex] = (part[j][6] - 1) / beam->id_slots_per_bunch;

  /* Figure out the charge */
  if (charge && npNew) {
    double totalCharge, oldMacroParticleCharge;
    oldMacroParticleCharge = charge->macroParticleCharge;
    if (!SDDS_GetParameterAsDouble(&SDDSin, "Charge", &totalCharge)) {
      SDDS_SetError("Unable to read Charge parameter from script output file");
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    }
    charge->charge = totalCharge;
    charge->macroParticleCharge = 0;
    if (npNew)
      charge->macroParticleCharge = totalCharge / npNew;
    if (oldMacroParticleCharge != 0 && fabs((charge->macroParticleCharge - oldMacroParticleCharge) / oldMacroParticleCharge) > 1e-8)
      printWarningForTracking("Macro-particle charge changed after SCRIPT element.",
                              "This may indicate a problem with the script.");
  }

  if (script->rpnParameters) {
    char **pName;
    int32_t pNames, iName;
    if (!(pName = SDDS_GetParameterNames(&SDDSin, &pNames))) {
      SDDS_SetError("Unable to get names of parameters from script output file");
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    }
    for (iName = 0; iName < pNames; iName++) {
      double paramValue;
      char rpnNameBuffer[16384];
      if (SDDS_NUMERIC_TYPE(SDDS_GetParameterType(&SDDSin, iName))) {
        if (!(SDDS_GetParameterAsDouble(&SDDSin, pName[iName], &paramValue))) {
          SDDS_SetError("Unable to get values of parameters from script output file");
          SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
        }
        snprintf(rpnNameBuffer, 16384, "%s#%ld.%s", trackingContext.elementName,
                 trackingContext.elementOccurrence, pName[iName]);
        rpn_store(paramValue, NULL, rpn_create_mem(rpnNameBuffer, 0));
      }
    }
    free(pName);
  }

  /* Close files */
  if (!failSoftly) {
    if (SDDS_ReadPage(&SDDSin) != -1)
      SDDS_Bomb("Script output file has multiple pages");
    if (!SDDS_Terminate(&SDDSin))
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
  }

  if (script->verbosity) {
    printf("Done processing particle input file from script\n");
    fflush(stdout);
  }

  /* convert (t, p) data to (s, delta) */
  for (j = 0; j < npNew; j++) {
    double p, beta;
    p = part[j][5];
    part[j][5] = (p - pCentral) / pCentral;
    beta = p / sqrt(sqr(p) + 1);
    part[j][4] *= beta * c_mks;
  }

  if (!script->keepFiles) {
    /* delete the input and output files */
    remove(input);
    remove(output);
  }

  /* clean up */
  free(cmdBuffer1);

  return npNew;
}

#if USE_MPI
long transformBeamWithScript_p(SCRIPT *script, double pCentral, CHARGE *charge,
                               BEAM *beam, double **part, int64_t np,
                               char *mainRootname, long iPass, long driftOrder, double z, long occurence,
                               long backtrack) {
  char *rootname = NULL, *input, *output = NULL;
  char *cmdBuffer1 = NULL;
  SDDS_DATASET SDDSout, SDDSin;
  double *data = NULL;
  char *dataname[6] = {"x", "xp", "y", "yp", "t", "p"};
  long i, j, npNew, npTotal, npNewTotal, nameLength;
  double *pID = NULL;
  TRACKING_CONTEXT trackingContext;
  double **lostParticle;
  int64_t nLost;

  getTrackingContext(&trackingContext);

#  if MPI_DEBUG
  printf("transformBeamWithScript_p: isMaster = %ld, distributedBeam = %ld, isSlave = %ld\n",
         isMaster, distributedBeam, isSlave);
#  endif

  if (distributedBeam) {
    MPI_Allreduce(&np, &npTotal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (npTotal == 0)
      return 0;
  } else
    npTotal = np;
  if (npTotal == 0)
    return 0;

  determineScriptNames(script, &rootname, &input, &output, &nameLength, mainRootname, 0, iPass, occurence);

  prepareScriptCommand(script, iPass, rootname, input, output, &cmdBuffer1, occurence, backtrack);
  if (script->verbosity > 0) {
    printf("%s\n", cmdBuffer1);
    fflush(stdout);
  }

#  if MPI_DEBUG
  printf("npTotal = %ld\n", npTotal);
  fflush(stdout);
#  endif

  /* dump the data to script input file */
  if (script->verbosity > 1) {
    printf("Writing data to script input file %s\n", input);
    fflush(stdout);
  }
  if (distributedBeam || (!distributedBeam && isMaster)) {
#  if MPI_DEBUG
    printf("dumping data to the script input file %s\n", input);
    fflush(stdout);
#  endif
    SDDS_ForceInactive(&SDDSout);
#  if MPI_DEBUG
    printf("SDDS_ForceInactive done\n");
    fflush(stdout);
#  endif
    SDDS_PhaseSpaceSetup(&SDDSout, input, SDDS_BINARY, 1, "script input",
                         "unknown", "unknown",
                         "transformBeamWithScript");
#  if MPI_DEBUG
    printf("SDDS_PhaseSpaceSetup done\n");
    fflush(stdout);
#  endif

    dump_phase_space(&SDDSout, part, np, 0, pCentral, charge ? charge->macroParticleCharge * npTotal : 0.0, beam ? beam->id_slots_per_bunch : 0);

#  if MPI_DEBUG
    printf("dump_phase_space done\n");
    fflush(stdout);
#  endif

    if (!SDDS_Terminate(&SDDSout))
      SDDS_Bomb("problem terminating script input file");
#  if MPI_DEBUG
    printf("SDDS_Terminate done\n");
    fflush(stdout);
#  endif
  }
#  if defined(CONDOR_COMPILE)
  _condor_ckpt_disable();
#  endif
#  if MPI_DEBUG
  printf("dumped the data to the script input file\n");
  fflush(stdout);
#  endif
  if (script->verbosity > 2) {
    printf("Done writing data to script input file %s\n", input);
    fflush(stdout);
  }

  /* copy the lost particles to a separate buffer so we can easily put them back */
  if (beam && beam->n_lost) {
    int64_t il;
    lostParticle = (double **)czarray_2d(sizeof(double), beam->n_lost, totalPropertiesPerParticle);
    for (il = 0; il < beam->n_lost; il++)
      memcpy(lostParticle[il], beam->particle[np + il], totalPropertiesPerParticle * sizeof(double));
    nLost = beam->n_lost;
  } else {
    lostParticle = NULL;
    nLost = 0;
  }

  /* run the script */
  if (isMaster) /* This will be done on the master */
  {
    if (script->useCsh)
      executeCshCommand(cmdBuffer1, mainRootname);
    else
      if (!system(cmdBuffer1)) {
      }
  }
#  if defined(CONDOR_COMPILE)
  _condor_ckpt_enable();
#  endif

  if (script->verbosity > 0) {
    printf("Command completed\n");
    fflush(stdout);
  }

  /* read the data from script output file */
#  if SDDS_MPI_IO
#    ifdef MPI_DEBUG
  printf("Waiting on MPI barrier, distributedBeam = %ld\n", distributedBeam);
  fflush(stdout);
#    endif

  MPI_Barrier(MPI_COMM_WORLD);

  if (distributedBeam) {
    if (!fexists(output))
      SDDS_Bomb("unable to find script output file");
    SDDSin.parallel_io = 1;
    /* set up parallel IO information */
    SDDS_MPI_Setup(&SDDSin, 1, n_processors, myid, MPI_COMM_WORLD, 0);
    if (!SDDS_MPI_InitializeInput(&SDDSin, output)) {
      SDDS_SetError("Unable to read script output file");
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    }
  } else if (isMaster) {
    if (!fexists(output))
      SDDS_Bomb("unable to find script output file");
    if (!SDDS_InitializeInput(&SDDSin, output)) {
      SDDS_SetError("Unable to read script output file");
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    }
  } else
    bombElegant("Invalid branch for SCRIPT element. Seek expert help.", NULL);
#  else
  if (!fexists(output))
    SDDS_Bomb("unable to find script output file");
  if (!SDDS_InitializeInput(&SDDSin, output)) {
    SDDS_SetError("Unable to read script output file");
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
  }
#  endif
#  if MPI_DEBUG
  printf("Initalized input file\n");
  fflush(stdout);
#  endif

  if (distributedBeam || (!distributedBeam && isMaster)) {
    if (!check_sdds_column(&SDDSin, "x", "m") ||
        !check_sdds_column(&SDDSin, "y", "m") ||
        !check_sdds_column(&SDDSin, "xp", NULL) ||
        !check_sdds_column(&SDDSin, "yp", NULL) ||
        !check_sdds_column(&SDDSin, "p", "m$be$nc") ||
        !check_sdds_column(&SDDSin, "t", "s")) {
      if (!check_sdds_column(&SDDSin, "p", "m$be$nc") &&
          check_sdds_column(&SDDSin, "p", NULL)) {
        printWarningForTracking("p has no units in script output file.", "Expected m$be$nc.");
      } else {
        printf(
          "necessary data quantities (x, x', y, y', t, p) have the wrong units or are not present in script output");
        fflush(stdout);
        exitElegant(1);
      }
    }
  }
  if (script->verbosity > 2) {
    printf("About to read file %s\n", output);
    fflush(stdout);
  }

  if (SDDS_ReadPage(&SDDSin) != 1) {
    SDDS_SetError("Unable to read script output file");
    SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    return 0;
  }
  npNew = SDDS_RowCount(&SDDSin);
#  if MPI_DEBUG
  printf("npNew = %ld\n", npNew);
  fflush(stdout);
#  endif
  if (!distributedBeam && npNew > 1)
    bombElegant("More than 1 particle returned for SCRIPT element when one particle provided in single-particle mode.", NULL);

  if (!distributedBeam) {
    MPI_Bcast(&npNew, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    npNewTotal = npNew;
  } else
    MPI_Allreduce(&npNew, &npNewTotal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

  if (npNewTotal > npTotal)
    if (script->noNewParticles)
      bombElegantVA("The number of particles increased from %ld to %ld after the SCRIPT element, even though NO_NEW_PARTICLES=0.",
                    npTotal, npNewTotal);

  if (script->verbosity > 0 && isMaster) {
    printf("%ld particles in script output file (was %ld)\n", npNewTotal, npTotal);
    fflush(stdout);
  } else {
#  if MPI_DEBUG
    printf("%ld particles in script output file (was %ld)\n", npNewTotal, npTotal);
    fflush(stdout);
#  endif
  }

  pID = NULL;
  /* See if user wants us to use the particleID data from the script output file */
  if (!isMaster && npNew > 0 && script->useParticleID) {
    if (!(pID = SDDS_GetColumnInDoubles(&SDDSin, "particleID"))) {
      SDDS_SetError("Unable to read particleID from script output file. Please set USE_PARTICLE_ID=0 if this is desired.\n");
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    }
#  if MPI_DEBUG
    printf("Read particle ID data\n");
    fflush(stdout);
#  endif
  }

  if (!isMaster && distributedBeam && (npNew + nLost) > np) {
    /* Resize the input array, preserving the data */
    part = (double **)resize_czarray_2d((void **)part, sizeof(double), npNew + nLost, totalPropertiesPerParticle);
    if (beam) {
      /* TODO: should free beam->particle first ! */
      beam->particle = part;
      beam->n_particle = npNew;
    }
#  if MPI_DEBUG
    printf("Resized part array\n");
    fflush(stdout);
#  endif
  }

  if (!isMaster && npNew > 0 && distributedBeam) {
#  if MPI_DEBUG
    printf("Copying particle data\n");
    fflush(stdout);
#  endif
    /* copy phase space coordinates */
    for (i = 0; i < 6; i++) {
      if (!(data = SDDS_GetColumnInDoubles(&SDDSin, dataname[i]))) {
        SDDS_SetError("Unable to read script output file");
        SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
      }
      for (j = 0; j < npNew; j++)
        part[j][i] = data[j];
      free(data);
      data = NULL;
    }
#  if MPI_DEBUG
    printf("Resized part array\n");
    fflush(stdout);
#  endif
  }
  if (script->verbosity > 3) {
    printf("Arrays resized\n");
    fflush(stdout);
  }

  /* put lost particle data back at the top of the array */
  if (beam && lostParticle && nLost) {
    int64_t il;
    for (il = 0; il < nLost; il++)
      memcpy(part[npNew + il], lostParticle[il], sizeof(double) * totalPropertiesPerParticle);
    beam->n_lost = nLost;
    free_czarray_2d((void **)lostParticle, nLost, totalPropertiesPerParticle);
    lostParticle = NULL;
  }

  /* copy or create particle ID */
  if (script->useParticleID) {
    if (pID) {
      /* The new particle IDs replace the pre-existing ones */
#  if MPI_DEBUG
      printf("Copying particle ID data\n");
      fflush(stdout);
#  endif
      for (j = 0; j < npNew; j++)
        part[j][6] = pID[j];
      pID = NULL;
    }
  } else {
    /* No particle ID data in the file, so generate some */
    if (isSlave && distributedBeam) {
      long sum = 0, tmp, my_offset = 0, *offset = tmalloc(n_processors * sizeof(*offset));
      MPI_Allgather(&npNew, 1, MPI_LONG, offset, 1, MPI_LONG, workers);
      tmp = offset[0];
      for (i = 1; i < n_processors; i++) {
        sum += tmp;
        tmp = offset[i];
        offset[i] = sum;
      }
      offset[0] = 0;
      my_offset = offset[myid - 1];
#  if MPI_DEBUG
      for (i = 0; i < n_processors; i++)
        printf("offset[%ld] = %ld\n", i, offset[i]);
      printf("my_offset = %ld\n", my_offset);
      fflush(stdout);
#  endif
      tfree(offset);
      for (j = 0; j < npNew; j++) {
        part[j][6] = j + 1 + my_offset;
#  if MPI_DEBUG
        printf("PID = %ld for j=%ld on myid=%d\n", (long)part[j][6], j, myid);
#  endif
      }
    }
  }
  if (script->verbosity > 3) {
    printf("Particle IDs (optionally) reset\n");
    fflush(stdout);
  }

  /* update bunch assignments using PID */
  if (beam && beam->id_slots_per_bunch > 0)
    for (j = 0; j < npNew; j++)
      part[j][bunchIndex] = (part[j][6] - 1) / beam->id_slots_per_bunch;

  /* update charge */
  if ((!distributedBeam && isMaster) || distributedBeam) {
    if (charge) {
      double totalCharge, oldMacroParticleCharge;
      oldMacroParticleCharge = charge->macroParticleCharge;
      if (!SDDS_GetParameterAsDouble(&SDDSin, "Charge", &totalCharge)) {
        SDDS_SetError("Unable to read Charge parameter from script output file");
        SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
      }
      if (totalCharge!=charge->charge && script->verbosity>1) 
	printf("Charge changed from %21.15e to %21.15e\n", charge->charge, totalCharge);
      charge->charge = totalCharge;
      charge->macroParticleCharge = 0;
      if (!distributedBeam) {
        if (npNew)
          charge->macroParticleCharge = totalCharge / npNew;
	if (script->verbosity>3) 
	  printf("Macro particle charge was %21.15e, now %21.15e, npNew = %ld, total charge = %21.15e\n",
		 oldMacroParticleCharge, charge->macroParticleCharge, npNew, charge->charge);
      } else if (npNewTotal) {
        charge->macroParticleCharge = totalCharge / npNewTotal;
	if (script->verbosity>3)
	  printf("Macro particle charge was %21.15e, now %21.15e, npNewTotal = %ld, total charge = %21.15e\n",
		 oldMacroParticleCharge, charge->macroParticleCharge, npNewTotal, charge->charge);
      }
      if (oldMacroParticleCharge != 0 && fabs((charge->macroParticleCharge - oldMacroParticleCharge) / oldMacroParticleCharge) > 1e-8) {
        printWarningForTracking("Macro-particle charge changed after SCRIPT element.",
                                "This may indicate a problem with the script.");
      }
    }
    if (script->verbosity > 3) {
      printf("Charge value (optionally) set\n");
      fflush(stdout);
    }
  }

  if (charge && distributedBeam) {
    if (MPI_Bcast(&(charge->charge), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD)!=MPI_SUCCESS ||
	MPI_Bcast(&charge->macroParticleCharge, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD)!=MPI_SUCCESS)
      bombElegant("MPI failed broadcasting charge data", NULL);
    if (script->verbosity > 3) {
      printf("Charge value broadcast (charge = %21.15e, mpCharge = %21.15e)\n",
	     charge->charge, charge->macroParticleCharge);
      fflush(stdout);
    }
  }

  if (script->rpnParameters) {
    char **pName;
    int32_t pNames, iName;
    if (!(pName = SDDS_GetParameterNames(&SDDSin, &pNames))) {
      SDDS_SetError("Unable to get names of parameters from script output file");
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
    }
    for (iName = 0; iName < pNames; iName++) {
      double paramValue;
      char rpnNameBuffer[16384];
      if (SDDS_NUMERIC_TYPE(SDDS_GetParameterType(&SDDSin, iName))) {
        if (!(SDDS_GetParameterAsDouble(&SDDSin, pName[iName], &paramValue))) {
          SDDS_SetError("Unable to get values of parameters from script output file");
          SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
        }
        snprintf(rpnNameBuffer, 16384, "%s#%ld.%s", trackingContext.elementName,
                 trackingContext.elementOccurrence, pName[iName]);
        rpn_store(paramValue, NULL, rpn_create_mem(rpnNameBuffer, 0));
      }
    }
    free(pName);
  }

  /* Close files */
  if (distributedBeam || (isMaster && !distributedBeam)) {
    if (SDDS_ReadPage(&SDDSin) != -1)
      SDDS_Bomb("Script output file has multiple pages");
    if (!SDDS_Terminate(&SDDSin))
      SDDS_PrintErrors(stderr, SDDS_EXIT_PrintErrors | SDDS_VERBOSE_PrintErrors);
  }
  if (script->verbosity) {
    printf("Done processing particle input file from script\n");
    fflush(stdout);
  }

  /* convert (t, p) data to (s, delta) */
  for (j = 0; j < npNew; j++) {
    double p, beta;
    p = part[j][5];
    part[j][5] = (p - pCentral) / pCentral;
    beta = p / sqrt(sqr(p) + 1);
    part[j][4] *= beta * c_mks;
  }

  if (isMaster && !script->keepFiles) {
    /* delete the input and output files */
    remove(input);
    remove(output);
  }

  /* clean up */
  free(cmdBuffer1);

  if (beam)
    beam->n_to_track_total = npNewTotal;
#  if MPI_DEBUG
  printf("returning npNew=  %ld\n", npNew);
  fflush(stdout);
#  endif
  return npNew;
}

#endif
