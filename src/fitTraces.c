/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/*
 * $Log: not supported by cvs2svn $
 * Revision 1.28  2010/02/04 15:17:26  borland
 * No longer use the bomb() routine.  Instead, use bombElegant(), which allows
 * better control of what happens when exiting.  Added "failed" semaphore option.
 * Switched from process_namelist() to processNamelist() for better response
 * to errors.
 * Includes Y. Wang's changes to parallelize shell and line beam types.
 *
 * Revision 1.27  2009/02/12 22:52:45  borland
 * Added ability to turn off echoing of namelists.
 *
 * Revision 1.26  2008/10/22 18:30:51  borland
 * Added global_settings command and means to inhibit file sync calls.
 *
 * Revision 1.25  2006/05/31 16:02:54  ywang25
 * The first release of Pelegant. It has passed a regression test of 100 cases.
 *
 * Revision 1.24  2005/11/22 23:21:20  borland
 * Added momentum aperture search, which necessitated adding an argument to
 * do_tracking, resulting in changes in many files.
 * Also improved convergence of orbit finder, adding a second iteration using
 * tracking if the matrix-based method fails.
 *
 * Revision 1.23  2005/08/18 02:49:10  borland
 * Particle allocation is now guaranteed to put particles in contiguous memory.
 * This means we don't swap pointers anymore, but must copy coordinates.
 *
 * Revision 1.22  2004/08/19 06:42:36  borland
 * Should fix bugs in particle creation in SCRIPT element, but needs
 * more testing.
 *
 * Revision 1.21  2004/03/28 17:00:55  borland
 * Added output of pass on which particles are lost to the lost particle file.
 * Not fully tested.
 *
 * Revision 1.20  2003/05/07 14:48:32  soliday
 * Removed fsync warning message.
 *
 * Revision 1.19  2003/02/15 22:57:48  borland
 * Added SDDS_DoFSync() calls to make sure output files get updated on
 * file server.
 *
 * Revision 1.18  2003/01/27 16:46:02  borland
 * Fixed problems with tune-shift-with-amplitude computation in presence
 * of MALIGN elements with ON_PASS!=-1.
 * Added LTHINLENS and LMIRROR elements.
 *
 * Revision 1.17  2002/08/14 20:23:38  soliday
 * Added Open License
 *
 * Revision 1.16  2002/01/02 14:18:57  borland
 * Revised due to addition of slice_analysis command.
 *
 * Revision 1.15  2001/10/15 20:37:02  soliday
 * Cleaned up for Linux.
 *
 * Revision 1.14  2001/10/15 15:42:28  soliday
 * Cleaned up for WIN32.
 *
 * Revision 1.13  2001/08/02 14:45:27  borland
 * Added search_path feature to run_setup namelist and to many (all?) elements and
 * commands that take input files.
 * FITPOINT beam statistics data now includes x, y, and z emittances.
 *
 * Revision 1.12  2000/10/25 20:52:00  borland
 * Added FLUSH_INTERVAL parameter to WATCH element.
 * Improvements to trace fitting.
 * Optimization includes ability to supply weights for each term.
 *
 * Revision 1.11  1999/10/12 21:49:54  borland
 * All printouts now go to the stdout rather than stderr.  fflush statements,
 * some unnecessary, were added in a mostly automated fashion.
 *
 * Revision 1.10  1999/09/09 04:41:30  borland
 * Modified tracking procedure (track_beam) to separate tracking stage
 * and output stage. This allowed adding optimization function output to
 * the final output file, but required lots of (minor) changes to
 * other routines.
 * Added floor coordinate output for CSRCSBENDS.
 *
 * Revision 1.9  1999/08/05 15:35:31  soliday
 * Added WIN32 and Linux support
 *
 * Revision 1.8  1999/07/01 01:52:38  borland
 * Made computation of sigma matrix for sigma and final output files consistent.
 * In both cases, now remove the centroid contributions.
 *
 * Revision 1.7  1998/08/25 23:54:24  borland
 * Latest version.  Still has trouble with real SR data.
 *
 * Revision 1.6  1998/06/08 16:50:43  borland
 * Fixed problems with SVD.  Added more reporting.
 *
 * Revision 1.5  1998/04/17 22:15:36  borland
 * Use Meschach matrix library.  Supports SVD or non-SVD.  Removes BPM
 * common mode.
 *
 * Revision 1.4  1998/03/19 21:00:55  borland
 * Removed items from track.h and put them in three new files (correctDefs.h,
 * tuneDefs.h, chromDefs.h) to isolate references to matlib.h; this is to
 * allow use of the Meschach matrix library in parts of elegant.
 *
 * Revision 1.3  1998/03/17 20:02:26  borland
 * All printouts now go to stdout, rather than the previous mix of
 * stdout and stdout.
 *
 * Revision 1.2  1997/10/20 14:57:10  borland
 * Improved trace fitting and related routines.  Added output of traces
 * after fitting.  Fixed some output-related bugs.
 *
 * Revision 1.1  1997/08/13 20:03:46  borland
 * First version.
 *
 */
/* file: fitTraces.c 
 *
 * Michael Borland, 1997
 */
#include "mdb.h"
#include "track.h"
#include "fitTraces.h"
#include "meschach.h"

typedef struct {
  long BPMs, traces;               /* number of BPMs and number of traces */
  char **BPMName;                  /* BPM names */
  long *xParamIndex, *yParamIndex; /* user parameter index of this BPM for x/y, if used */
  double **x, **y;                 /* measured beam position readout at BPMs */
  double **xSim, **ySim;           /* simulated coordinates at BPMs---*not* simulated readout! */
  ELEMENT_LIST **element;          /* pointer to the element structure */
  long *latticeIndex;              /* index of BPM in the lattice */
  double **startingCoord;          /* starting coordinates of each trace, varied in fitting */
} FIT_TRACE_DATA;

typedef struct {
  long parameters;
  char **elementName, **parameterName;
  long *parameterIndex, *elementType;
  double *delta, *changeLimit, *lowerLimit, *upperLimit;
  double **paramData; /* will point to the actual location used to store the parameter value in
                         the element structure */
  ELEMENT_LIST **target;
  double *definedValue;
} FIT_TRACE_PARAMETERS;

typedef struct {
  SDDS_TABLE SDDStable;
  long **traceDataIndex, *paramDataIndex;
} FIT_OUTPUT_DATA;

FIT_TRACE_PARAMETERS *fit_traces_readFitParametersFile(char *dataFile, LINE_LIST *beamline,
                                                       long bpmCalibrations);
FIT_TRACE_DATA *fit_traces_readTraceDataFile(char *dataFile, LINE_LIST *beamline);
void find_trajectory_bpm_readouts(double *xReadout, double *yReadout, double *xActual,
                                  double *yActual, ELEMENT_LIST **bpmElement, long *BPMIndex,
                                  long BPMs, LINE_LIST *beamline, RUN *run,
                                  TRAJECTORY *trajBuffer, double *startingCoordinate, double momentum);
void fit_traces_findDerivatives(FIT_TRACE_DATA *traceData, FIT_TRACE_PARAMETERS *fitParam,
                                MAT *D, LINE_LIST *beamline, RUN *run);
double fit_trace_findReadbackErrors(MAT *readbackError, FIT_TRACE_DATA *traceData,
                                    LINE_LIST *beamline, RUN *run);
FIT_OUTPUT_DATA *fit_trace_setUpOutputFile(char *filename,
                                           FIT_TRACE_PARAMETERS *fitParam,
                                           FIT_TRACE_PARAMETERS *bpmCalParam,
                                           FIT_TRACE_DATA *traceData, long iterations);
void fit_trace_saveParamValues(double *buffer, FIT_TRACE_DATA *traceData,
                               FIT_TRACE_PARAMETERS *fitParam,
                               FIT_TRACE_PARAMETERS *bpmCalParam);
void fit_trace_restoreParamValues(double *buffer, FIT_TRACE_DATA *traceData,
                                  FIT_TRACE_PARAMETERS *fitParam,
                                  FIT_TRACE_PARAMETERS *bpmCalParam, RUN *run,
                                  LINE_LIST *beamline);
double fit_trace_takeStep(MAT *D, MAT *readbackVector, MAT *paramVector, FIT_TRACE_DATA *traceData,
                          FIT_TRACE_PARAMETERS *fitParam, LINE_LIST *beamline, RUN *run,
                          double convergenceFactor, double position_change_limit,
                          double slope_change_limit, long use_SVD,
                          long SVs_to_keep, long SVs_to_remove,
                          double *minSV, double *maxSV);
void fit_trace_setRowValues(FIT_OUTPUT_DATA *outputData, long iteration,
                            double rmsError, double lastRmsError, double pass,
                            double convergenceFactor, FIT_TRACE_DATA *traceData,
                            FIT_TRACE_PARAMETERS *fitParam, FIT_TRACE_PARAMETERS *bpmCalParam);
double fit_trace_calibrateMonitors(MAT *readbackVector, FIT_TRACE_DATA *traceData,
                                   FIT_TRACE_PARAMETERS *fitParam, FIT_TRACE_PARAMETERS *bpmCalData,
                                   LINE_LIST *beamline,
                                   RUN *run, double convergence_factor, long reject_common_mode);
long fit_trace_setUpTraceOutput(char *filename);
void fit_trace_writeTraceOutput(char *filename, FIT_TRACE_DATA *traceData,
                                LINE_LIST *beamline, RUN *run, MAT *readbackError);
void fit_trace_randomizeValues(MAT *paramVector, FIT_TRACE_DATA *traceData,
                               FIT_TRACE_PARAMETERS *fitParam, LINE_LIST *beamline, RUN *run, double level);

MAT *m_diag(VEC *diagElements, MAT *A) {
  long i;
  if (!diagElements)
    bombElegant("Problem with allocation of vector of diagonal elements.\n", NULL);
  if (!A)
    A = m_get(diagElements->dim, diagElements->dim);
  m_zero(A);
  for (i = 0; i < MIN(A->n, A->m); i++)
    A->me[i][i] = diagElements->ve[i];
  return A;
}

void do_fit_trace_data(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline) {
  long i, iteration, subIteration, parameters, readbacks, outputRow, i_restart;
  long iBPM, iTrace, goodSteps;
  double rmsError = 0.0, lastRmsError = 0.0;
  FIT_TRACE_PARAMETERS *fitParam, *bpmCalParam;
  FIT_TRACE_DATA *traceData;
  MAT *D, *readbackVector, *paramVector;
  MAT *D0, *paramVector0;
  double *lastParameterValues, *startParameterValues, minSV, maxSV;
  FIT_OUTPUT_DATA *outputData;
  long pass = 0, passCount, monitorCalsDone;

  /* process namelist text */
  if (processNamelist(&fit_traces, nltext) == NAMELIST_ERROR)
    bombElegant(NULL, NULL);
  if (echoNamelists)
    print_namelist(stdout, &fit_traces);

  if (!trace_data_file || !fexists(trace_data_file)) {
    printf("fit_traces: trace_data_file file not given or not found\n");
    fflush(stdout);
    exitElegant(1);
  }
  if (!fit_parameters_file || !fexists(fit_parameters_file)) {
    printf("fit_traces: fit_parameters_file file not given or not found\n");
    fflush(stdout);
    exitElegant(1);
  }
  if (iterations < 1) {
    printf("fit_traces: iterations<1\n");
    fflush(stdout);
    exitElegant(1);
  }
  if (!fit_output_file) {
    printf("fit_traces: fit_output_file file not given\n");
    fflush(stdout);
    exitElegant(1);
  }
  if (use_SVD && SVs_to_remove && SVs_to_keep) {
    printf("Can't have both SVs_to_remove and SVs_to_keep nonzero.\n");
    fflush(stdout);
    exitElegant(1);
  }

  /* trace_data_file file contains column data giving
   * x readings, y readings, and BPM name.
   * Each page is a separate trace.
   */
  traceData = fit_traces_readTraceDataFile(trace_data_file, beamline);
  if (!traceData->traces) {
    printf("fit_traces: no trace data in %s\n", trace_data_file);
    fflush(stdout);
    exitElegant(1);
  }
  printf("%ld traces with %ld BPMs:\n", traceData->traces, traceData->BPMs);
  fflush(stdout);
  for (iTrace = 0; iTrace < traceData->traces; iTrace++) {
    printf("Trace %ld: ", iTrace);
    fflush(stdout);
    for (iBPM = 0; iBPM < traceData->BPMs; iBPM++) {
      printf("(%10.3e, %10.3e)  ",
             traceData->x[iTrace][iBPM],
             traceData->y[iTrace][iBPM]);
      fflush(stdout);
    }
    printf("\n");
    fflush(stdout);
  }

  /* fit_parameters_file file contains column data giving ElementName, ElementParameter
   */
  fitParam = fit_traces_readFitParametersFile(fit_parameters_file, beamline, 0);
  bpmCalParam = fit_traces_readFitParametersFile(fit_parameters_file, beamline, 1);

  outputData = fit_trace_setUpOutputFile(fit_output_file, fitParam, bpmCalParam, traceData, iterations);

  parameters = 4 * traceData->traces + fitParam->parameters;
  readbacks = 2 * traceData->BPMs * traceData->traces;
  lastParameterValues = tmalloc(sizeof(*lastParameterValues) * (parameters + bpmCalParam->parameters));
  startParameterValues = tmalloc(sizeof(*startParameterValues) * (parameters + bpmCalParam->parameters));

  /* check that there are enough traces for the given number of parameters */
  if (readbacks < (parameters + bpmCalParam->parameters)) {
    printf("fit_traces: too few traces for given number of fit parameters.\n");
    fflush(stdout);
    exitElegant(1);
  }

  /* allocate matrices to solve the problem:
   * readbackVector = D*paramVector 
   * where D is the matrix of derivatives of readbacks wrt parameters 
   * least squares solution is
   * paramVector = Inv(Tr(D)*D)*Tr(D)*readbackVector
   * D is organized as follows:
   *   D(i, j) = derivative of ith readout w.r.t. jth parameter where
   *   i = 2*t*B + 2*b + c with
   *       t = trace index [0, T-1], b = BPM index [0, B-1], c = x/y coordinate index [0, 1]
   *   j = p+4*t on [0, 4T-1] with
   *         p = phase-space coord index, [0, 3]
   *         t = trace index [0, T-1]
   *     = u+4T on [4T, 4T+U-1] with
   *         u = user-parameter index, [0, U-1]
   * readbackVector is organized as follows:
   *   rV(i,0) = ith readout, where i is as above
   * parameterVector is organized as follows:
   *   pV(j,0) = jth parameter, where j is as above
   */
  D = m_get(readbacks, parameters);
  readbackVector = m_get(readbacks, 1); /* really the vector of readback errors */
  paramVector = m_get(parameters, 1);   /* really the vector of parameter deltas */

  D0 = m_get(readbacks, parameters - fitParam->parameters);
  paramVector0 = m_get(parameters - fitParam->parameters, 1);

  if (n_restarts < 0)
    n_restarts = 0;
  outputRow = 0;
  for (i_restart = 0; i_restart <= n_restarts; i_restart++) {
    printf("Starting optimization pass %ld\n", i_restart);
    if (n_restarts && i_restart > 0)
      fit_trace_randomizeValues(paramVector0, traceData, fitParam, beamline, run, restart_randomization_level);
    else
      lastRmsError = rmsError = 0;
    for (iteration = goodSteps = 0; iteration < iterations; iteration++) {
      rmsError = fit_trace_findReadbackErrors(readbackVector, traceData, beamline, run);
      fit_trace_setRowValues(outputData, outputRow++, rmsError, lastRmsError, pass,
                             convergence_factor, traceData, fitParam, bpmCalParam);

      minSV = DBL_MAX;
      maxSV = -DBL_MAX;
      if (iteration == 0) {
        for (i = passCount = goodSteps = 0; i < trace_sub_iterations; i++) {
          lastRmsError = rmsError;
          rmsError = fit_trace_takeStep(D0, readbackVector, paramVector0,
                                        traceData, NULL, beamline, run,
                                        trace_convergence_factor, position_change_limit,
                                        slope_change_limit, 0, 0, 0,
                                        &minSV, &maxSV);
          passCount++;
          if (lastRmsError < rmsError) {
            trace_convergence_factor /= convergence_factor_divisor;
            if (trace_convergence_factor < convergence_factor_min)
              trace_convergence_factor = convergence_factor_min;
            goodSteps = 0;
          } else {
            goodSteps++;
            if (goodSteps >= convergence_increase_steps) {
              trace_convergence_factor *= convergence_factor_multiplier;
              goodSteps = 0;
              if (trace_convergence_factor > convergence_factor_max)
                trace_convergence_factor = convergence_factor_max;
            }
          }

          if (i && fabs(lastRmsError - rmsError) / (rmsError + 1e-10) < trace_fractional_target)
            break;
        }
        printf("RMS error is %e after trace optimization  %ld passes\n",
               rmsError, passCount);
        fflush(stdout);
      }

      /*
        if (use_SVD) 
        printf("  min/max inverse SVs: %le %le\n",
        minSV, maxSV);
        fflush(stdout);
        */

      passCount = 0;
      subIteration = sub_iterations;
      minSV = DBL_MAX;
      maxSV = -DBL_MAX;
      monitorCalsDone = 0;
      do {
        lastRmsError = rmsError;
        /* fit_trace_saveParamValues(lastParameterValues, traceData, fitParam); */
        rmsError = fit_trace_takeStep(D, readbackVector, paramVector,
                                      traceData, fitParam, beamline, run,
                                      convergence_factor, position_change_limit,
                                      slope_change_limit, use_SVD,
                                      SVs_to_keep, SVs_to_remove,
                                      &minSV, &maxSV);
        if ((passCount && lastRmsError > rmsError && (lastRmsError - rmsError) < BPM_threshold) || rmsError < BPM_threshold) {
          monitorCalsDone++;
          rmsError = fit_trace_calibrateMonitors(readbackVector, traceData, fitParam, bpmCalParam,
                                                 beamline, run, convergence_factor,
                                                 reject_BPM_common_mode);
        }
        passCount++;
        if (rmsError < target)
          break;
        if (fabs(lastRmsError - rmsError) < tolerance)
          break;
        if (lastRmsError < rmsError) {
          convergence_factor /= convergence_factor_divisor;
          if (convergence_factor < convergence_factor_min)
            convergence_factor = convergence_factor_min;
          goodSteps = 0;
        } else {
          goodSteps++;
          if (goodSteps >= convergence_increase_steps) {
            convergence_factor *= convergence_factor_multiplier;
            if (convergence_factor > convergence_factor_max)
              convergence_factor = convergence_factor_max;
            goodSteps = 0;
          }
        }
        pass++;
      } while (--subIteration > 0);
      rmsError = fit_trace_findReadbackErrors(readbackVector, traceData, beamline, run);
      printf("RMS error is %e after full  optimization  %ld passes,  %ld monitor calibrations, C=%e\n",
             rmsError, passCount, monitorCalsDone, convergence_factor);
      fflush(stdout);
      if (use_SVD) {
        printf("  min/max inverse SVs: %e %e\n",
               minSV, maxSV);
        fflush(stdout);
      }
      if (rmsError < target) {
        printf("rms error less than target %e, terminating.\n", target);
        break;
      }
      if (fabs(lastRmsError - rmsError) < tolerance) {
        printf("rms error change less than tolerance %e, terminating.\n", tolerance);
        break;
      }
    }
    fflush(stdout);
    if (rmsError < target)
      break;
  }

  fit_trace_setRowValues(outputData, outputRow++, rmsError, lastRmsError, pass,
                         convergence_factor, traceData, fitParam, bpmCalParam);
  if (!SDDS_Terminate(&outputData->SDDStable))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);

  if (trace_output_file)
    fit_trace_writeTraceOutput(trace_output_file, traceData, beamline, run, readbackVector);
}

void fit_trace_writeTraceOutput(
  char *filename,
  FIT_TRACE_DATA *traceData,
  LINE_LIST *beamline,
  RUN *run,
  MAT *readbackError) {
  SDDS_TABLE SDDSout;
  long iTrace, iBPM, row;
  double x, y;

  if (!SDDS_InitializeOutputElegant(&SDDSout, SDDS_BINARY, 0, NULL, NULL,
                                    filename) ||
      0 > SDDS_DefineColumn(&SDDSout, "xFit", NULL, "m", "Predicted x coordinate",
                            NULL, SDDS_DOUBLE, 0) ||
      0 > SDDS_DefineColumn(&SDDSout, "yFit", NULL, "m", "Predicted y coordinate",
                            NULL, SDDS_DOUBLE, 0) ||
      0 > SDDS_DefineColumn(&SDDSout, "BPMName", NULL, NULL, "BPM Name", NULL,
                            SDDS_STRING, 0) ||
      !SDDS_WriteLayout(&SDDSout)) {
    SDDS_SetError("Problem setting up trace output.");
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  fit_trace_findReadbackErrors(readbackError, traceData, beamline, run);
  for (iTrace = 0; iTrace < traceData->traces; iTrace++) {
    if (!SDDS_StartPage(&SDDSout, traceData->BPMs)) {
      SDDS_SetError("Problem starting page for trace output.");
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
    for (iBPM = 0; iBPM < traceData->BPMs; iBPM++) {
      row = 2 * iTrace * traceData->BPMs + 2 * iBPM;
      x = readbackError->me[row][0] + traceData->x[iTrace][iBPM];
      y = readbackError->me[row + 1][0] + traceData->y[iTrace][iBPM];
      if (!SDDS_SetRowValues(&SDDSout, SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE, iBPM,
                             "xFit", x, "yFit", y,
                             "BPMName", traceData->BPMName[iBPM], NULL)) {
        SDDS_SetError("Problem setting row values for trace output.");
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
    }
    if (!SDDS_WritePage(&SDDSout))
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    if (!inhibitFileSync)
      SDDS_DoFSync(&SDDSout);
  }
  if (!SDDS_Terminate(&SDDSout))
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
}

double fit_trace_findReadbackErrors(
  MAT *readbackError,
  FIT_TRACE_DATA *traceData,
  LINE_LIST *beamline,
  RUN *run) {
  long iTrace, iBPM, iCoord, row, count;
  static double *x = NULL, *y = NULL;
  static long lastBPMs = 0, lastElements = 0;
  double startingCoord[4], p, sum;
  static TRAJECTORY *trajectory = NULL;

  assert_element_links(beamline->links, run, beamline, DYNAMIC_LINK | LINK_ELEMENT_DEFINITION);

  if (!lastBPMs || lastBPMs != traceData->BPMs) {
    lastBPMs = traceData->BPMs;
    if (!(x = SDDS_Realloc(x, sizeof(*x) * lastBPMs)) ||
        !(y = SDDS_Realloc(y, sizeof(*y) * lastBPMs))) {
      printf("Memory allocation failure in fit_trace_findReadbackErrors (1)\n");
      fflush(stdout);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
  }
  if (!lastElements || lastElements != beamline->n_elems) {
    if (!(trajectory = SDDS_Realloc(trajectory, sizeof(*trajectory) * beamline->n_elems))) {
      printf("Memory allocation failure in fit_trace_findReadbackErrors (2)\n");
      fflush(stdout);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
    lastElements = beamline->n_elems;
  }

  /* track each trace to predict position */
  p = sqrt(sqr(run->ideal_gamma) + 1);
  sum = 0;
  count = 0;
  for (iTrace = 0; iTrace < traceData->traces; iTrace++) {
    for (iCoord = 0; iCoord < 4; iCoord++)
      startingCoord[iCoord] = traceData->startingCoord[iTrace][iCoord];
    find_trajectory_bpm_readouts(x, y,
                                 traceData->xSim[iTrace],
                                 traceData->ySim[iTrace],
                                 traceData->element,
                                 traceData->latticeIndex, traceData->BPMs,
                                 beamline, run, trajectory, startingCoord, p);
    for (iBPM = 0; iBPM < traceData->BPMs; iBPM++) {
      row = 2 * iTrace * traceData->BPMs + 2 * iBPM;
      readbackError->me[row][0] = x[iBPM] - traceData->x[iTrace][iBPM];
      readbackError->me[row + 1][0] = y[iBPM] - traceData->y[iTrace][iBPM];
      sum += sqr(readbackError->me[row][0]) + sqr(readbackError->me[row + 1][0]);
      count++;
    }
  }
  return sqrt(sum / count);
}

void fit_traces_findDerivatives(
  FIT_TRACE_DATA *traceData,
  FIT_TRACE_PARAMETERS *fitParam,
  MAT *D,
  LINE_LIST *beamline,
  RUN *run) {
  long iTrace, iBPM, iUserParam, iCoord, row, column;
  static double *x0 = NULL, *y0 = NULL, *x = NULL, *y = NULL;
  static long lastBPMs = 0, lastElements = 0;
  double startingCoord[4], refValue, p;
  static TRAJECTORY *trajectory = NULL;

  if (!lastBPMs || lastBPMs != traceData->BPMs) {
    lastBPMs = traceData->BPMs;
    if (!(x0 = SDDS_Realloc(x0, sizeof(*x0) * lastBPMs)) ||
        !(y0 = SDDS_Realloc(y0, sizeof(*y0) * lastBPMs)) ||
        !(x = SDDS_Realloc(x, sizeof(*x) * lastBPMs)) ||
        !(y = SDDS_Realloc(y, sizeof(*y) * lastBPMs))) {
      printf("Memory allocation failure in fit_traces_findDerivatives\n");
      fflush(stdout);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
  }
  if (!lastElements || lastElements != beamline->n_elems) {
    if (!(trajectory = SDDS_Realloc(trajectory, sizeof(*trajectory) * beamline->n_elems))) {
      printf("Memory allocation failure in fit_traces_findDerivatives\n");
      fflush(stdout);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
    lastElements = beamline->n_elems;
  }

  for (iCoord = 0; iCoord < D->n; iCoord++) {
    for (iBPM = 0; iBPM < D->m; iBPM++) {
      D->me[iCoord][iBPM] = 0;
    }
  }

  /* two types of parameters: 
     1. initial trajectory coordinates 
     2. element parameters declared by user
     */

  /* compute and store derivative data for initial trajectory coordintes */
  p = sqrt(sqr(run->ideal_gamma) + 1);
  for (iTrace = 0; iTrace < traceData->traces; iTrace++) {
    /* find reference traj for derivatives */
    for (iCoord = 0; iCoord < 4; iCoord++)
      startingCoord[iCoord] = traceData->startingCoord[iTrace][iCoord];
    find_trajectory_bpm_readouts(x0, y0, NULL, NULL, traceData->element,
                                 traceData->latticeIndex, traceData->BPMs,
                                 beamline, run, trajectory, startingCoord, p);
    for (iCoord = 0; iCoord < 4; iCoord++) {
      refValue = startingCoord[iCoord];
      startingCoord[iCoord] += 1e-6;
      find_trajectory_bpm_readouts(x, y, NULL, NULL, traceData->element,
                                   traceData->latticeIndex, traceData->BPMs,
                                   beamline, run, trajectory, startingCoord, p);
      startingCoord[iCoord] = refValue;
      column = iCoord + 4 * iTrace;
      for (iBPM = 0; iBPM < traceData->BPMs; iBPM++) {
        row = 2 * iTrace * traceData->BPMs + 2 * iBPM;
        D->me[row][column] = (x[iBPM] - x0[iBPM]) / 1e-6;
        D->me[row + 1][column] = (y[iBPM] - y0[iBPM]) / 1e-6;
      }
    }
  }

  for (iTrace = 0; iTrace < traceData->traces; iTrace++) {
    for (iCoord = 0; iCoord < 4; iCoord++)
      startingCoord[iCoord] = traceData->startingCoord[iTrace][iCoord];
    find_trajectory_bpm_readouts(x0, y0, NULL, NULL, traceData->element,
                                 traceData->latticeIndex, traceData->BPMs,
                                 beamline, run, trajectory, startingCoord, p);
    for (iUserParam = 0; iUserParam < fitParam->parameters; iUserParam++) {
      refValue = *(fitParam->paramData[iUserParam]);
      *(fitParam->paramData[iUserParam]) += fitParam->delta[iUserParam];
      if (fitParam->target[iUserParam]->matrix) {
        free_matrices(fitParam->target[iUserParam]->matrix);
        free(fitParam->target[iUserParam]->matrix);
      }
      compute_matrix(fitParam->target[iUserParam], run, NULL);
      assert_element_links(beamline->links, run, beamline, DYNAMIC_LINK);
      find_trajectory_bpm_readouts(x, y, NULL, NULL, traceData->element,
                                   traceData->latticeIndex, traceData->BPMs,
                                   beamline, run, trajectory, startingCoord, p);
      *(fitParam->paramData[iUserParam]) = refValue;
      free_matrices(fitParam->target[iUserParam]->matrix);
      free(fitParam->target[iUserParam]->matrix);
      compute_matrix(fitParam->target[iUserParam], run, NULL);
      assert_element_links(beamline->links, run, beamline, DYNAMIC_LINK);
      column = 4 * traceData->traces + iUserParam;
      for (iBPM = 0; iBPM < traceData->BPMs; iBPM++) {
        row = 2 * iTrace * traceData->BPMs + 2 * iBPM;
        D->me[row][column] = (x[iBPM] - x0[iBPM]) / fitParam->delta[iUserParam];
        D->me[row + 1][column] = (y[iBPM] - y0[iBPM]) / fitParam->delta[iUserParam];
      }
    }
  }
}

void find_trajectory_bpm_readouts(
  double *xReadout, double *yReadout, /* arrays in which to return x and y readouts */
  double *xActual, double *yActual,   /* arrays in which to return real x and y positions */
  ELEMENT_LIST **bpmElement,          /* pointers to BPM elements in the beamline */
  long *latticeIndex,                 /* index in the lattice of BPMs */
  long BPMs,
  LINE_LIST *beamline, RUN *run,
  TRAJECTORY *trajBuffer, double *startingCoordinate, double momentum) {
  static double **particle = NULL;
  long tracking_flags = TEST_PARTICLES, nPart = 1;
  long iBPM, i;

  if (!particle) {
    particle = (double **)czarray_2d(sizeof(**particle), 1, MAX_PROPERTIES_PER_PARTICLE);
    particle[0][4] = particle[0][5] = 0;
    particle[0][6] = 1;
  }
  for (i = 0; i < 4; i++)
    particle[0][i] = startingCoordinate[i];

  if (!do_tracking(NULL, particle, nPart, NULL, beamline, &momentum,
                   (double **)NULL, (BEAM_SUMS **)NULL, (long *)NULL,
                   trajBuffer, run, 0, tracking_flags, 1, 0, NULL, NULL, NULL, NULL, NULL)) {
    printf("Error tracking particle to find trajectory at BPMs.\n");
    fflush(stdout);
    exitElegant(1);
  }
  for (iBPM = 0; iBPM < BPMs; iBPM++) {
    if (xActual)
      xActual[iBPM] = trajBuffer[latticeIndex[iBPM]].centroid[0];
    if (yActual)
      yActual[iBPM] = trajBuffer[latticeIndex[iBPM]].centroid[2];
    xReadout[iBPM] = computeMonitorReading(bpmElement[iBPM], 0,
                                           trajBuffer[latticeIndex[iBPM]].centroid, 0);
    yReadout[iBPM] = computeMonitorReading(bpmElement[iBPM], 2,
                                           trajBuffer[latticeIndex[iBPM]].centroid, 0);
  }
}

FIT_TRACE_PARAMETERS *fit_traces_readFitParametersFile(
  char *dataFile,
  LINE_LIST *beamline,
  long bpmCalibrations) {
  SDDS_TABLE SDDSin;
  int64_t i;
  FIT_TRACE_PARAMETERS *ftp;
  long parameterIndex, elementType, changeLimitsPresent;
  long lowerLimitsPresent, upperLimitsPresent;
  char **name0;
  long names0;
  ELEMENT_LIST *elem;

  if (!SDDS_InitializeInputFromSearchPath(&SDDSin, dataFile)) {
    printf("Error: couldn't read file %s\n", dataFile);
    fflush(stdout);
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    exitElegant(1);
  }
  if (SDDS_CHECK_OKAY != SDDS_CheckColumn(&SDDSin, "ElementName", NULL, SDDS_STRING, stdout) ||
      SDDS_CHECK_OKAY != SDDS_CheckColumn(&SDDSin, "ElementParameter", NULL, SDDS_STRING, stdout) ||
      SDDS_CHECK_OKAY != SDDS_CheckColumn(&SDDSin, "Delta", NULL, SDDS_ANY_NUMERIC_TYPE, stdout)) {
    printf("Problem with column(s) in file %s\n", dataFile);
    fflush(stdout);
    exitElegant(1);
  }

  switch (SDDS_CheckColumn(&SDDSin, "ChangeLimit", NULL, SDDS_ANY_NUMERIC_TYPE, NULL)) {
  case SDDS_CHECK_OKAY:
    changeLimitsPresent = 1;
    break;
  case SDDS_CHECK_NONEXISTENT:
    changeLimitsPresent = 0;
    break;
  default:
    printf("Problem with column ChangeLimit.");
    fflush(stdout);
    exitElegant(1);
    break;
  }

  switch (SDDS_CheckColumn(&SDDSin, "LowerLimit", NULL, SDDS_ANY_NUMERIC_TYPE, NULL)) {
  case SDDS_CHECK_OKAY:
    lowerLimitsPresent = 1;
    break;
  case SDDS_CHECK_NONEXISTENT:
    lowerLimitsPresent = 0;
    break;
  default:
    printf("Problem with column LowerLimit.");
    fflush(stdout);
    exitElegant(1);
    break;
  }
  switch (SDDS_CheckColumn(&SDDSin, "UpperLimit", NULL, SDDS_ANY_NUMERIC_TYPE, NULL)) {
  case SDDS_CHECK_OKAY:
    upperLimitsPresent = 1;
    break;
  case SDDS_CHECK_NONEXISTENT:
    upperLimitsPresent = 0;
    break;
  default:
    printf("Problem with column UpperLimit.");
    fflush(stdout);
    exitElegant(1);
    break;
  }

  if (SDDS_ReadPage(&SDDSin) <= 0) {
    printf("Problem reading data from file %s\n", dataFile);
    fflush(stdout);
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    exitElegant(1);
  }
  if (!(ftp = malloc(sizeof(*ftp)))) {
    printf("Error: memory allocation failure (fit_traces_readFitParametersFile)\n");
    fflush(stdout);
    exitElegant(1);
  }
  ftp->changeLimit = ftp->lowerLimit = ftp->upperLimit = NULL;
  if (!(name0 = SDDS_GetColumn(&SDDSin, "ElementName"))) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    exitElegant(1);
  }
  if ((names0 = SDDS_RowCount(&SDDSin)) <= 0) {
    printf("No data in file (fit_traces_readFitParametersFile)\n");
    fflush(stdout);
    exitElegant(1);
  }
  SDDS_SetRowFlags(&SDDSin, 0);
  for (i = 0; i < names0; i++) {
    if (!(elem = find_element(name0[i], NULL, beamline->elem))) {
      printf("Element %s not in lattice (fit_traces_readFitParametersFile)\n",
             name0[i]);
      fflush(stdout);
      exitElegant(1);
    }
    if ((bpmCalibrations &&
         (elem->type == T_MONI || elem->type == T_HMON || elem->type == T_VMON)) ||
        (!bpmCalibrations &&
         !(elem->type == T_MONI || elem->type == T_HMON || elem->type == T_VMON))) {
      if (!SDDS_AssertRowFlags(&SDDSin, SDDS_INDEX_LIMITS, i, i, 1)) {
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        exitElegant(1);
      }
    }
  }
  free(name0);

  if (!(ftp->parameters = SDDS_CountRowsOfInterest(&SDDSin))) {
    return ftp;
  }

  if (!(ftp->paramData = malloc(sizeof(*ftp->paramData) * ftp->parameters))) {
    printf("Error: memory allocation failure (fit_traces_readFitParametersFile)\n");
    fflush(stdout);
    exitElegant(1);
  }
  ftp->changeLimit = ftp->lowerLimit = ftp->upperLimit = NULL;
  if (!(ftp->elementName = SDDS_GetColumn(&SDDSin, "ElementName")) ||
      !(ftp->parameterName = SDDS_GetColumn(&SDDSin, "ElementParameter")) ||
      !(ftp->delta = SDDS_GetColumnInDoubles(&SDDSin, "Delta")) ||
      (changeLimitsPresent && !(ftp->changeLimit = SDDS_GetColumnInDoubles(&SDDSin, "ChangeLimit"))) ||
      (lowerLimitsPresent && !(ftp->lowerLimit = SDDS_GetColumnInDoubles(&SDDSin, "LowerLimit"))) ||
      (upperLimitsPresent && !(ftp->upperLimit = SDDS_GetColumnInDoubles(&SDDSin, "UpperLimit")))) {
    printf("Problem reading data from file %s\n", dataFile);
    fflush(stdout);
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    exitElegant(1);
  }
  if (!(ftp->target = malloc(sizeof(*ftp->target) * ftp->parameters)) ||
      !(ftp->parameterIndex = malloc(sizeof(*ftp->parameterIndex) * ftp->parameters)) ||
      !(ftp->elementType = malloc(sizeof(*ftp->elementType) * ftp->parameters)) ||
      !(ftp->definedValue = malloc(sizeof(*ftp->definedValue) * ftp->parameters))) {
    printf("Error: memory allocation problem reading parameters file\n");
    fflush(stdout);
    exitElegant(1);
  }
  for (i = 0; i < ftp->parameters; i++) {
    if (!(ftp->target[i] = find_element(ftp->elementName[i], NULL, beamline->elem))) {
      printf("Error: element %s not found in beamline\n", ftp->elementName[i]);
      fflush(stdout);
      exitElegant(1);
    }
    ftp->elementType[i] = elementType = ftp->target[i]->type;
    if ((ftp->parameterIndex[i] = parameterIndex = confirm_parameter(ftp->parameterName[i], elementType)) < 0) {
      printf("Error: element %s does not have a parameter called %s\n",
             ftp->elementName[i], ftp->parameterName[i]);
      fflush(stdout);
      exitElegant(1);
    }
    if (entity_description[elementType].parameter[parameterIndex].type != IS_DOUBLE) {
      printf("Error: parameter %s of element %s is not a double value\n",
             ftp->parameterName[i], ftp->elementName[i]);
      fflush(stdout);
      exitElegant(1);
    }
    ftp->paramData[i] = ((double *)(ftp->target[i]->p_elem +
                                    entity_description[elementType].parameter[parameterIndex].offset));
    ftp->definedValue[i] = *(ftp->paramData[i]);
    if (ftp->lowerLimit || ftp->upperLimit) {
      if (ftp->lowerLimit && ftp->upperLimit) {
        if (ftp->lowerLimit[i] == ftp->upperLimit[i])
          continue;
        if (ftp->lowerLimit[i] > ftp->upperLimit[i]) {
          printf("Error: upperLimit<lowerLimit for parameter %s of element %s\n",
                 ftp->parameterName[i], ftp->elementName[i]);
          fflush(stdout);
          exitElegant(1);
        }
      }
      if (ftp->lowerLimit && ftp->lowerLimit[i] > ftp->definedValue[i]) {
        char warningText[16834];
        snprintf(warningText, 16384, "Value %e of %s.%s is below lower limit of %e.",
                 ftp->definedValue[i], ftp->elementName[i], ftp->parameterName[i], ftp->lowerLimit[i]);
        printWarning("Initial value is less than lower limit for trace fitting.",
                     warningText);
      }
      if (ftp->upperLimit && ftp->upperLimit[i] < ftp->definedValue[i]) {
        snprintf(warningText, 16384, "Value %e of %s.%s is above lower limit of %e.",
                 ftp->definedValue[i], ftp->elementName[i], ftp->parameterName[i], ftp->upperLimit[i]);
        printWarning("Initial value is greater than upper limit for trace fitting.",
                     warningText);
      }
    }
  }

  if (SDDS_ReadPage(&SDDSin) > 1)
    printfWarning("Only the first page of fit_traces input file is used.",
                  "Input file provided has multiple pages.");
  SDDS_Terminate(&SDDSin);

  if (bpmCalibrations)
    printf("%ld bpm calibration parameters defined.\n", ftp->parameters);
  else
    printf("%ld fit parameters defined.\n", ftp->parameters);
  fflush(stdout);

  return ftp;
}

FIT_TRACE_DATA *fit_traces_readTraceDataFile(
  char *dataFile,
  LINE_LIST *beamline) {
  SDDS_TABLE SDDSin;
  long maxTraces, iTrace, indexFirst;
  long iBPMFirst, iBPM;
  FIT_TRACE_DATA *trace;
  char **BPMName;

  if (!SDDS_InitializeInputFromSearchPath(&SDDSin, dataFile)) {
    printf("Error: couldn't read file %s\n", dataFile);
    fflush(stdout);
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
    exitElegant(1);
  }
  if (SDDS_CHECK_OKAY != SDDS_CheckColumn(&SDDSin, "x", "m", SDDS_ANY_NUMERIC_TYPE, stdout) ||
      SDDS_CHECK_OKAY != SDDS_CheckColumn(&SDDSin, "y", "m", SDDS_ANY_NUMERIC_TYPE, stdout) ||
      SDDS_CHECK_OKAY != SDDS_CheckColumn(&SDDSin, "BPMName", NULL, SDDS_STRING, stdout)) {
    printf("Problem with column(s) x, y, or BPMName in file %s\n",
           dataFile);
    fflush(stdout);
    exitElegant(1);
  }

  if (!(trace = malloc(sizeof(*trace)))) {
    printf("Error trying to allocate space for traces.\n");
    fflush(stdout);
    exitElegant(1);
  }
  maxTraces = 0;
  iTrace = 0;
  trace->x = trace->y = trace->xSim = trace->ySim = NULL;
  while (SDDS_ReadPage(&SDDSin) > 0) {
    if (maxTraces <= iTrace) {
      if (!(trace->x = SDDS_Realloc(trace->x, (maxTraces + 10) * sizeof(*trace->x))) ||
          !(trace->y = SDDS_Realloc(trace->y, (maxTraces + 10) * sizeof(*trace->y))) ||
          !(trace->xSim = SDDS_Realloc(trace->xSim, (maxTraces + 10) * sizeof(*trace->xSim))) ||
          !(trace->ySim = SDDS_Realloc(trace->ySim, (maxTraces + 10) * sizeof(*trace->ySim)))) {
        printf("Error trying to allocate space for more traces.\n");
        fflush(stdout);
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        exitElegant(1);
      }
      maxTraces += 10;
    }
    if (!iTrace) {
      if (!(trace->BPMs = SDDS_CountRowsOfInterest(&SDDSin))) {
        printf("No traces on first page of trace file\n");
        fflush(stdout);
        exitElegant(1);
      }
      /* allocate arrays to hold indices of BPM in users list of parameters, if it is used */
      if (!(trace->xParamIndex = SDDS_Malloc(sizeof(*trace->xParamIndex) * trace->BPMs)) ||
          !(trace->yParamIndex = SDDS_Malloc(sizeof(*trace->yParamIndex) * trace->BPMs))) {
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
        exitElegant(1);
      }
      for (iBPM = 0; iBPM < trace->BPMs; iBPM++)
        /* indicate that no search has been done */
        trace->xParamIndex[iBPM] = trace->yParamIndex[iBPM] = -2;
    } else {
      if (trace->BPMs != SDDS_CountRowsOfInterest(&SDDSin)) {
        printf("Fit traces have different numbers of data points.");
        fflush(stdout);
        exitElegant(1);
      }
    }
    if (!(trace->x[iTrace] = SDDS_GetColumnInDoubles(&SDDSin, "x")) ||
        !(trace->y[iTrace] = SDDS_GetColumnInDoubles(&SDDSin, "y"))) {
      printf("Error trying to read x or y values for trace\n");
      fflush(stdout);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
    if (!(trace->xSim[iTrace] = malloc(sizeof(*trace->xSim[iTrace]) * trace->BPMs)) ||
        !(trace->ySim[iTrace] = malloc(sizeof(*trace->ySim[iTrace]) * trace->BPMs))) {
      printf("Memory allocation failure (xSim/ySim arrays)\n");
      fflush(stdout);
      exitElegant(1);
    }
    if (!(BPMName = SDDS_GetColumn(&SDDSin, "BPMName"))) {
      printf("Error trying to read BPM names for trace\n");
      fflush(stdout);
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors);
      exitElegant(1);
    }
    if (iTrace == 0) {
      trace->BPMName = BPMName;
    } else {
      /* require the same BPM names on every page */
      for (iBPM = 0; iBPM < trace->BPMs; iBPM++) {
        if (strcmp(trace->BPMName[iBPM], BPMName[iBPM])) {
          printf("Fit traces have mismatched BPM names---all pages must have the names in the same order.\n");
          fflush(stdout);
          exitElegant(1);
        }
      }
    }
    iTrace++;
  }

  trace->traces = iTrace;
  trace->startingCoord = (double **)czarray_2d(sizeof(**trace->startingCoord), trace->traces, 4);

  indexFirst = LONG_MAX;
  iBPMFirst = 0;
  if (!(trace->latticeIndex = malloc(sizeof(*trace->latticeIndex) * trace->BPMs)) ||
      !(trace->element = malloc(sizeof(*trace->element) * trace->BPMs))) {
    printf("Memory allocation failure storing trace data.\n");
    fflush(stdout);
    exitElegant(1);
  }
  for (iBPM = 0; iBPM < trace->BPMs; iBPM++) {
    if (!(trace->element[iBPM] = find_element_index(trace->BPMName[iBPM], NULL, beamline->elem, trace->latticeIndex + iBPM)) || trace->element[iBPM]->type != T_MONI) {
      printf("Element %s not found or not of type MONI\n", trace->BPMName[iBPM]);
      fflush(stdout);
      exitElegant(1);
    }
    if (trace->latticeIndex[iBPM] < indexFirst) {
      indexFirst = trace->latticeIndex[iBPM];
      iBPMFirst = iBPM;
    }
  }

  for (iTrace = 0; iTrace < trace->traces; iTrace++) {
    trace->startingCoord[iTrace][0] = trace->x[iTrace][iBPMFirst];
    trace->startingCoord[iTrace][2] = trace->y[iTrace][iBPMFirst];
    trace->startingCoord[iTrace][1] =
      trace->startingCoord[iTrace][3] = 0;
  }

  printf("%ld traces found with %ld BPMs.\n", trace->traces, trace->BPMs);
  fflush(stdout);
  fflush(stdout);
  return trace;
}

FIT_OUTPUT_DATA *fit_trace_setUpOutputFile(char *filename,
                                           FIT_TRACE_PARAMETERS *fitParam,
                                           FIT_TRACE_PARAMETERS *bpmCalParam,
                                           FIT_TRACE_DATA *traceData,
                                           long iterations) {
  long iTrace, iCoord, iUserParam;
  FIT_OUTPUT_DATA *outputData;
  char name[128];
  char *coordName[4] = {
    "x",
    "xp",
    "y",
    "yp",
  };
  char *coordUnits[4] = {
    "m",
    "rad",
    "m",
    "rad",
  };

  if (!(outputData = malloc(sizeof(*outputData))) ||
      !(outputData->traceDataIndex = (long **)czarray_2d(sizeof(**outputData->traceDataIndex),
                                                         traceData->traces, 4)) ||
      !(outputData->paramDataIndex = malloc(sizeof(*outputData->paramDataIndex) *
                                            (fitParam->parameters + bpmCalParam->parameters)))) {
    printf("memory allocation failure in fit_trace_setUpOutputFile");
    fflush(stdout);
    exitElegant(1);
  }

  if (!SDDS_InitializeOutputElegant(&outputData->SDDStable, SDDS_BINARY, 0, NULL, NULL,
                                    filename) ||
      0 > SDDS_DefineColumn(&outputData->SDDStable, "Iteration", NULL, NULL,
                            "Major iteration number",
                            NULL, SDDS_LONG, 0) ||
      0 > SDDS_DefineColumn(&outputData->SDDStable, "Passes", NULL, NULL,
                            "Total number of iteration passes",
                            NULL, SDDS_LONG, 0) ||
      0 > SDDS_DefineColumn(&outputData->SDDStable, "RMSError", NULL, "m",
                            "RMS error of predicted BPM readouts",
                            NULL, SDDS_DOUBLE, 0) ||
      0 > SDDS_DefineColumn(&outputData->SDDStable, "RMSErrorImprovement", NULL, NULL,
                            "Fractional change in RMS error",
                            NULL, SDDS_DOUBLE, 0) ||
      0 > SDDS_DefineColumn(&outputData->SDDStable, "ConvergenceFactor", NULL, NULL,
                            NULL, NULL, SDDS_DOUBLE, 0)) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  for (iTrace = 0; iTrace < traceData->traces; iTrace++) {
    for (iCoord = 0; iCoord < 4; iCoord++) {
      sprintf(name, "Trace%ldInitial%s", iTrace, coordName[iCoord]);
      if (0 > (outputData->traceDataIndex[iTrace][iCoord] =
                 SDDS_DefineColumn(&outputData->SDDStable, name, NULL, coordUnits[iCoord],
                                   NULL, NULL, SDDS_DOUBLE, 0))) {
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
    }
  }
  for (iUserParam = 0; iUserParam < fitParam->parameters; iUserParam++) {
    sprintf(name, "%s.%s", fitParam->elementName[iUserParam],
            fitParam->parameterName[iUserParam]);
    if (0 > (outputData->paramDataIndex[iUserParam] =
               SDDS_DefineColumn(&outputData->SDDStable, name, NULL, NULL,
                                 NULL, NULL, SDDS_DOUBLE, 0))) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  }
  for (iUserParam = 0; iUserParam < bpmCalParam->parameters; iUserParam++) {
    sprintf(name, "%s.%s", bpmCalParam->elementName[iUserParam],
            bpmCalParam->parameterName[iUserParam]);
    if (0 > (outputData->paramDataIndex[iUserParam + fitParam->parameters] =
               SDDS_DefineColumn(&outputData->SDDStable, name, NULL, NULL,
                                 NULL, NULL, SDDS_DOUBLE, 0))) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  }

  if (!SDDS_WriteLayout(&outputData->SDDStable) || !SDDS_StartPage(&outputData->SDDStable, iterations * 2)) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  return outputData;
}

void fit_trace_saveParamValues(
  double *buffer,
  FIT_TRACE_DATA *traceData,
  FIT_TRACE_PARAMETERS *fitParam,
  FIT_TRACE_PARAMETERS *bpmCalParam) {
  long iTrace, iCoord, iUserParam, offset;
  for (iTrace = 0; iTrace < traceData->traces; iTrace++)
    for (iCoord = 0; iCoord < 4; iCoord++)
      buffer[iTrace * 4 + iCoord] = traceData->startingCoord[iTrace][iCoord];
  offset = traceData->traces * 4;
  for (iUserParam = 0; iUserParam < fitParam->parameters; iUserParam++)
    buffer[iUserParam + offset] = *(fitParam->paramData[iUserParam]);
  for (; iUserParam < fitParam->parameters + bpmCalParam->parameters; iUserParam++)
    buffer[iUserParam + offset] = *(bpmCalParam->paramData[iUserParam]);
}

void fit_trace_restoreParamValues(
  double *buffer,
  FIT_TRACE_DATA *traceData,
  FIT_TRACE_PARAMETERS *fitParam,
  FIT_TRACE_PARAMETERS *bpmCalParam,
  RUN *run,
  LINE_LIST *beamline) {
  long iTrace, iCoord, iUserParam, offset;

  /* assert new trajectory starting values */
  for (iTrace = 0; iTrace < traceData->traces; iTrace++) {
    for (iCoord = 0; iCoord < 4; iCoord++) {
      traceData->startingCoord[iTrace][iCoord] = buffer[iTrace * 4 + iCoord];
    }
  }

  /* assert new element parameter values and recompute matrices */
  offset = traceData->traces * 4;
  for (iUserParam = 0; iUserParam < fitParam->parameters; iUserParam++) {
    *(fitParam->paramData[iUserParam]) = buffer[offset + iUserParam];
    if (fitParam->target[iUserParam]->matrix) {
      free_matrices(fitParam->target[iUserParam]->matrix);
      free(fitParam->target[iUserParam]->matrix);
    }
    compute_matrix(fitParam->target[iUserParam], run, NULL);
  }
  assert_element_links(beamline->links, run, beamline, DYNAMIC_LINK | LINK_ELEMENT_DEFINITION);
}

double fit_trace_takeStep(
  MAT *D,
  MAT *readbackVector,
  MAT *paramVector,
  FIT_TRACE_DATA *traceData,
  FIT_TRACE_PARAMETERS *fitParam,
  LINE_LIST *beamline,
  RUN *run,
  double convergenceFactor,
  double positionChangeLimit,
  double slopeChangeLimit,
  long use_SVD,
  long SVs_to_keep, long SVs_to_remove,
  double *minSV, double *maxSV) {
  double rmsError, factor;
  long iTrace, iCoord, offset, iUserParam, checkLimits, i;
  FIT_TRACE_PARAMETERS fitParam0 = {
    0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
  static MAT *DInv = NULL, *U = NULL, *V = NULL, *T = NULL, *S = NULL;
  static MAT *DtD = NULL, *DtDInv = NULL, *DtDInvDt = NULL, *Dt = NULL;
  static VEC *SingValue = NULL;

  if (!fitParam) {
    fitParam = &fitParam0;
    fitParam->parameters = 0;
  }

  /* find derivatives with respect to all parameters, both the user-declared ones
   * and the starting trajectory values 
   */
  fit_traces_findDerivatives(traceData, fitParam, D, beamline, run);

#ifdef DEBUG
  printf("D:\n");
  fflush(stdout);
  m_foutput(stdout, D);
#endif

  if (use_SVD) {
    /* invert the matrix D using SVD */
    DInv = m_resize(DInv, D->n, D->m);
    U = m_resize(U, D->m, D->m);
    V = m_resize(V, D->n, D->n);
    T = m_resize(T, D->n, D->m);
    S = m_resize(S, D->n, D->m);
    SingValue = v_resize(SingValue, D->n);
    svd(D, U, V, SingValue);
#ifdef DEBUG
    printf("U:\n");
    fflush(stdout);
    m_foutput(stdout, U);
    printf("V:\n");
    fflush(stdout);
    m_foutput(stdout, V);
    printf("SingValue (raw): \n");
    fflush(stdout);
    v_foutput(stdout, SingValue);
#endif
    for (i = 0; i < SingValue->dim; i++)
      if (SingValue->ve[i])
        SingValue->ve[i] = 1. / SingValue->ve[i];
    SVs_to_keep = SingValue->dim;
    if (SVs_to_remove > 0)
      SVs_to_keep -= SVs_to_remove;
    for (i = SVs_to_keep; i < SingValue->dim; i++)
      SingValue->ve[i] = 0;
    for (i = 0; i < SVs_to_keep; i++) {
      if (SingValue->ve[i] < *minSV)
        *minSV = SingValue->ve[i];
      if (SingValue->ve[i] > *maxSV)
        *maxSV = SingValue->ve[i];
    }
    m_diag(SingValue, S);
    m_mlt(S, U, T);
    mtrm_mlt(V, T, DInv);
    m_mlt(DInv, readbackVector, paramVector);

#ifdef DEBUG
    printf("SingValue:\n");
    fflush(stdout);
    v_foutput(stdout, SingValue);
    printf("S:\n");
    fflush(stdout);
    m_foutput(stdout, S);
    printf("T:\n");
    fflush(stdout);
    m_foutput(stdout, T);
    printf("DInv:\n");
    fflush(stdout);
    m_foutput(stdout, DInv);
#endif
  } else {
    DtD = m_resize(DtD, D->n, D->n);
    Dt = m_resize(Dt, D->n, D->m);
    DtDInv = m_resize(DtDInv, D->n, D->n);
    DtDInvDt = m_resize(DtDInvDt, D->n, D->m);
    mtrm_mlt(D, D, DtD);
    m_inverse(DtD, DtDInv);
    m_transp(D, Dt);
    m_mlt(DtDInv, Dt, DtDInvDt);
    m_mlt(DtDInvDt, readbackVector, paramVector);
#ifdef DEBUG
    printf("DtDInv:\n");
    fflush(stdout);
    m_foutput(stdout, DtDInv);
#endif
  }
#ifdef DEBUG
  printf("readbackVector:\n");
  fflush(stdout);
  m_foutput(stdout, readbackVector);
  printf("paramVector:\n");
  fflush(stdout);
  m_foutput(stdout, paramVector);
#endif

  if (convergenceFactor > 0)
    sm_mlt(convergenceFactor, paramVector, paramVector);

  /* check for changes that exceed allowed limits */
  factor = 1;
  offset = traceData->traces * 4;
  if (fitParam->changeLimit) {
    for (iUserParam = 0; iUserParam < fitParam->parameters; iUserParam++) {
      if (fitParam->changeLimit[iUserParam] &&
          fabs(paramVector->me[offset + iUserParam][0]) > fitParam->changeLimit[iUserParam]) {
        factor = MIN(factor, fabs(fitParam->changeLimit[iUserParam] / paramVector->me[offset + iUserParam][0]));
      }
    }
  }

  if (positionChangeLimit) {
    for (iTrace = 0; iTrace < traceData->traces; iTrace++) {
      for (iCoord = 0; iCoord < 4; iCoord += 2) {
        if (fabs(paramVector->me[iTrace * 4 + iCoord][0]) > positionChangeLimit) {
          factor = MIN(factor, fabs(positionChangeLimit / paramVector->me[iTrace * 4 + iCoord][0]));
        }
      }
    }
  }

  if (slopeChangeLimit) {
    for (iTrace = 0; iTrace < traceData->traces; iTrace++) {
      for (iCoord = 1; iCoord < 4; iCoord += 2) {
        if (fabs(paramVector->me[iTrace * 4 + iCoord][0]) > slopeChangeLimit) {
          factor = MIN(factor, fabs(slopeChangeLimit / paramVector->me[iTrace * 4 + iCoord][0]));
        }
      }
    }
  }

  if (factor < 1) {
    printf("changes reduced by factor %f to stay within change limits\n",
           factor);
    fflush(stdout);
    sm_mlt(factor, paramVector, paramVector);
  } else if (factor > 1) {
    printf("Error: limiting factor exceeds 1.\n");
    fflush(stdout);
  }

  /* assert new trajectory starting values */
  for (iTrace = 0; iTrace < traceData->traces; iTrace++) {
    for (iCoord = 0; iCoord < 4; iCoord++) {
      traceData->startingCoord[iTrace][iCoord] -=
        paramVector->me[iTrace * 4 + iCoord][0];
    }
  }

  /* assert new element parameter values and recompute matrices */
  for (iUserParam = 0; iUserParam < fitParam->parameters; iUserParam++) {
    *(fitParam->paramData[iUserParam]) -=
      paramVector->me[offset + iUserParam][0];
    /*
    printf("New value for %s[%s] is %e, change of %le\n",
            fitParam->elementName[iUserParam], fitParam->parameterName[iUserParam],
            *(fitParam->paramData[iUserParam]), paramVector->me[offset+iUserParam][0]);
    fflush(stdout);
*/
    checkLimits = 1;
    if (fitParam->lowerLimit && fitParam->upperLimit &&
        fitParam->lowerLimit[iUserParam] == fitParam->upperLimit[iUserParam])
      checkLimits = 0;
    if (checkLimits && fitParam->lowerLimit &&
        *(fitParam->paramData[iUserParam]) < fitParam->lowerLimit[iUserParam]) {
      *(fitParam->paramData[iUserParam]) = fitParam->lowerLimit[iUserParam];
      printf("Parameter %s of %s limited to %e\n",
             fitParam->parameterName[iUserParam], fitParam->elementName[iUserParam],
             *(fitParam->paramData[iUserParam]));
      fflush(stdout);
    }
    if (checkLimits && fitParam->upperLimit) {
      if (*(fitParam->paramData[iUserParam]) > fitParam->upperLimit[iUserParam]) {
        *(fitParam->paramData[iUserParam]) = fitParam->upperLimit[iUserParam];
        printf("Parameter %s of %s limited to %e\n",
               fitParam->parameterName[iUserParam], fitParam->elementName[iUserParam],
               *(fitParam->paramData[iUserParam]));
        fflush(stdout);
      }
    }
    if (fitParam->target[iUserParam]->matrix) {
      free_matrices(fitParam->target[iUserParam]->matrix);
      free(fitParam->target[iUserParam]->matrix);
    }
    change_defined_parameter_values(&fitParam->elementName[iUserParam],
                                    &fitParam->parameterIndex[iUserParam],
                                    &fitParam->elementType[iUserParam],
                                    fitParam->paramData[iUserParam], 1);
    compute_matrix(fitParam->target[iUserParam], run, NULL);
  }

  /* compute RMS error and return it */
  rmsError = fit_trace_findReadbackErrors(readbackVector, traceData, beamline, run);
  return rmsError;
}

void fit_trace_randomizeValues(
  MAT *paramVector,
  FIT_TRACE_DATA *traceData,
  FIT_TRACE_PARAMETERS *fitParam,
  LINE_LIST *beamline,
  RUN *run,
  double level) {
  FIT_TRACE_PARAMETERS fitParam0 = {
    0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
  long iTrace, iCoord, iUserParam;

  if (!fitParam) {
    fitParam = &fitParam0;
    fitParam->parameters = 0;
  }

  /* randomize trajectory starting values */
  for (iTrace = 0; iTrace < traceData->traces; iTrace++) {
    for (iCoord = 0; iCoord < 4; iCoord++)
      traceData->startingCoord[iTrace][iCoord] *= 1 + (random_1_elegant(-1) - 0.5) * level * 2;
  }

  /* randomize element parameter values and recompute matrices */
  for (iUserParam = 0; iUserParam < fitParam->parameters; iUserParam++) {
    *(fitParam->paramData[iUserParam]) *= 1 + (random_1_elegant(-1) - 0.5) * level * 2;
    change_defined_parameter_values(&fitParam->elementName[iUserParam],
                                    &fitParam->parameterIndex[iUserParam],
                                    &fitParam->elementType[iUserParam],
                                    fitParam->paramData[iUserParam], 1);
    compute_matrix(fitParam->target[iUserParam], run, NULL);
  }
}

double fit_trace_calibrateMonitors(
  MAT *readbackVector,
  FIT_TRACE_DATA *traceData,
  FIT_TRACE_PARAMETERS *fitParam,
  FIT_TRACE_PARAMETERS *bpmCalParam,
  LINE_LIST *beamline,
  RUN *run,
  double convergence_factor,
  long reject_common_mode) {
  long iBPM, iTrace, iParam;
  double rmsError, calibration;
  long checkLimits, nxCals, nyCals;
  double sum1, sum2, reading, xCalSum, yCalSum, xdCalAverage = 0.0, ydCalAverage = 0.0;
  double centroid[6] = {0,0,0,0,0,0};

  /* find the new trajectory */
  rmsError = fit_trace_findReadbackErrors(readbackVector, traceData, beamline, run);
  nxCals = nyCals = 0;
  xCalSum = yCalSum = 0;
  for (iBPM = 0; iBPM < traceData->BPMs; iBPM++) {
    /* look at each BPM for which there is trace data */
    if (traceData->element[iBPM]->type == T_MONI ||
        traceData->element[iBPM]->type == T_HMON) {
      /* horizontal BPMs */
      if (traceData->xParamIndex[iBPM] == -2) {
        for (iParam = 0; iParam < bpmCalParam->parameters; iParam++) {
          /* see if this bpm is in the list of parameters to be fit */
          if (traceData->element[iBPM] != bpmCalParam->target[iParam])
            continue;
          if ((traceData->element[iBPM]->type == T_MONI &&
               strcmp(bpmCalParam->parameterName[iParam], "XCALIBRATION") == 0) ||
              (traceData->element[iBPM]->type == T_HMON &&
               strcmp(bpmCalParam->parameterName[iParam], "CALIBRATION") == 0))
            break;
        }
        if (iParam != bpmCalParam->parameters)
          traceData->xParamIndex[iBPM] = iParam;
        else
          traceData->xParamIndex[iBPM] = -1; /* search done, no match found */
      }
      if ((iParam = traceData->xParamIndex[iBPM]) >= 0) {
        for (iTrace = sum1 = sum2 = 0; iTrace < traceData->traces; iTrace++) {
          centroid[0] = traceData->xSim[iTrace][iBPM];
          centroid[2] = traceData->ySim[iTrace][iBPM];
          reading = computeMonitorReading(traceData->element[iBPM], 0, centroid, COMPUTEMONITORREADING_CAL_1);
          sum1 += traceData->x[iTrace][iBPM] * reading;
          sum2 += reading * reading;
        }
        if (sum2) {
          calibration = sum1 / sum2 * convergence_factor +
                        (1 - convergence_factor) * getMonitorCalibration(traceData->element[iBPM], 0);
          checkLimits = 1;
          if (bpmCalParam->lowerLimit && bpmCalParam->upperLimit &&
              bpmCalParam->lowerLimit[iParam] == bpmCalParam->upperLimit[iParam])
            checkLimits = 0;
          if (checkLimits && bpmCalParam->lowerLimit &&
              bpmCalParam->lowerLimit[iParam] > calibration)
            calibration = bpmCalParam->lowerLimit[iParam];
          if (checkLimits && bpmCalParam->upperLimit &&
              bpmCalParam->upperLimit[iParam] < calibration)
            calibration = bpmCalParam->upperLimit[iParam];
          setMonitorCalibration(traceData->element[iBPM], calibration, 0);
          nxCals++;
          xCalSum += calibration;
        }
      }
    }
    if (traceData->element[iBPM]->type == T_MONI ||
        traceData->element[iBPM]->type == T_VMON) {
      /* vertical BPMs */
      if (traceData->yParamIndex[iBPM] == -2) {
        for (iParam = 0; iParam < bpmCalParam->parameters; iParam++) {
          /* see if this bpm is in the list of parameters to be fit */
          if (traceData->element[iBPM] != bpmCalParam->target[iParam])
            continue;
          if ((traceData->element[iBPM]->type == T_MONI &&
               strcmp(bpmCalParam->parameterName[iParam], "YCALIBRATION") == 0) ||
              (traceData->element[iBPM]->type == T_VMON &&
               strcmp(bpmCalParam->parameterName[iParam], "CALIBRATION") == 0))
            break;
        }
        if (iParam != bpmCalParam->parameters)
          traceData->yParamIndex[iBPM] = iParam;
        else
          traceData->yParamIndex[iBPM] = -1; /* search done, no match found */
      }
      if ((iParam = traceData->yParamIndex[iBPM]) >= 0) {
        for (iTrace = sum1 = sum2 = 0; iTrace < traceData->traces; iTrace++) {
          centroid[0] = traceData->xSim[iTrace][iBPM];
          centroid[2] = traceData->ySim[iTrace][iBPM];
          reading = computeMonitorReading(traceData->element[iBPM], 2, centroid, COMPUTEMONITORREADING_CAL_1);
          sum1 += traceData->y[iTrace][iBPM] * reading;
          sum2 += reading * reading;
        }
        if (sum2) {
          calibration = sum1 / sum2 * convergence_factor +
                        (1 - convergence_factor) * getMonitorCalibration(traceData->element[iBPM], 1);
          checkLimits = 1;
          if (bpmCalParam->lowerLimit && bpmCalParam->upperLimit &&
              bpmCalParam->lowerLimit[iParam] == bpmCalParam->upperLimit[iParam])
            checkLimits = 0;
          if (checkLimits && bpmCalParam->lowerLimit &&
              bpmCalParam->lowerLimit[iParam] > calibration)
            calibration = bpmCalParam->lowerLimit[iParam];
          if (checkLimits && bpmCalParam->upperLimit &&
              bpmCalParam->upperLimit[iParam] < calibration)
            calibration = bpmCalParam->upperLimit[iParam];
          setMonitorCalibration(traceData->element[iBPM], calibration, 1);
          nyCals++;
          yCalSum += calibration;
        }
      }
    }
  }

  if (reject_common_mode && (nxCals || nyCals)) {
    if (nxCals)
      xdCalAverage = xCalSum / nxCals - 1;
    if (nyCals)
      ydCalAverage = yCalSum / nyCals - 1;
    convergence_factor = 1;
    for (iBPM = 0; iBPM < traceData->BPMs; iBPM++) {
      /* look at each BPM for which there is trace data */
      if (nxCals && (traceData->element[iBPM]->type == T_MONI ||
                     traceData->element[iBPM]->type == T_HMON)) {
        /* horizontal BPMs */
        if ((iParam = traceData->xParamIndex[iBPM]) >= 0) {
          calibration = getMonitorCalibration(traceData->element[iBPM], 0);
          calibration = (calibration - xdCalAverage) * convergence_factor +
                        (1 - convergence_factor) * calibration;
          checkLimits = 1;
          if (bpmCalParam->lowerLimit && bpmCalParam->upperLimit &&
              bpmCalParam->lowerLimit[iParam] == bpmCalParam->upperLimit[iParam])
            checkLimits = 0;
          if (checkLimits && bpmCalParam->lowerLimit &&
              bpmCalParam->lowerLimit[iParam] > calibration)
            calibration = bpmCalParam->lowerLimit[iParam];
          if (checkLimits && bpmCalParam->upperLimit &&
              bpmCalParam->upperLimit[iParam] < calibration)
            calibration = bpmCalParam->upperLimit[iParam];
          setMonitorCalibration(traceData->element[iBPM], calibration, 0);
        }
      }
      if (nyCals && (traceData->element[iBPM]->type == T_MONI ||
                     traceData->element[iBPM]->type == T_VMON)) {
        /* vertical BPMs */
        if ((iParam = traceData->yParamIndex[iBPM]) >= 0) {
          calibration = getMonitorCalibration(traceData->element[iBPM], 1);
          calibration = (calibration - ydCalAverage) * convergence_factor +
                        (1 - convergence_factor) * calibration;
          checkLimits = 1;
          if (bpmCalParam->lowerLimit && bpmCalParam->upperLimit &&
              bpmCalParam->lowerLimit[iParam] == bpmCalParam->upperLimit[iParam])
            checkLimits = 0;
          if (checkLimits && bpmCalParam->lowerLimit &&
              bpmCalParam->lowerLimit[iParam] > calibration)
            calibration = bpmCalParam->lowerLimit[iParam];
          if (checkLimits && bpmCalParam->upperLimit &&
              bpmCalParam->upperLimit[iParam] < calibration)
            calibration = bpmCalParam->upperLimit[iParam];
          setMonitorCalibration(traceData->element[iBPM], calibration, 1);
        }
      }
    }
  }

  /* find the new trajectory */
  rmsError = fit_trace_findReadbackErrors(readbackVector, traceData, beamline, run);
  return rmsError;
}

void fit_trace_setRowValues(
  FIT_OUTPUT_DATA *outputData,
  long iteration,
  double rmsError,
  double lastRmsError,
  double pass,
  double convergenceFactor,
  FIT_TRACE_DATA *traceData,
  FIT_TRACE_PARAMETERS *fitParam,
  FIT_TRACE_PARAMETERS *bpmCalParam) {
  long iTrace, iCoord, iUserParam;

  if (!SDDS_SetRowValues(&outputData->SDDStable,
                         SDDS_SET_BY_NAME | SDDS_PASS_BY_VALUE,
                         iteration,
                         "Iteration", iteration,
                         "RMSError", rmsError,
                         "Passes", pass,
                         "RMSErrorImprovement", (rmsError - lastRmsError) / lastRmsError,
                         "ConvergenceFactor", convergenceFactor,
                         NULL)) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
  for (iTrace = 0; iTrace < traceData->traces; iTrace++) {
    for (iCoord = 0; iCoord < 4; iCoord++) {
      if (!SDDS_SetRowValues(&outputData->SDDStable,
                             SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE,
                             iteration,
                             outputData->traceDataIndex[iTrace][iCoord],
                             traceData->startingCoord[iTrace][iCoord],
                             -1)) {
        SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
      }
    }
  }
  for (iUserParam = 0; iUserParam < fitParam->parameters; iUserParam++) {
    if (!SDDS_SetRowValues(&outputData->SDDStable,
                           SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE,
                           iteration,
                           outputData->paramDataIndex[iUserParam],
                           *(fitParam->paramData[iUserParam]),
                           -1)) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  }
  for (iUserParam = 0; iUserParam < bpmCalParam->parameters; iUserParam++) {
    if (!SDDS_SetRowValues(&outputData->SDDStable,
                           SDDS_SET_BY_INDEX | SDDS_PASS_BY_VALUE,
                           iteration,
                           outputData->paramDataIndex[iUserParam + fitParam->parameters],
                           *(bpmCalParam->paramData[iUserParam]),
                           -1)) {
      SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
    }
  }
  if (!SDDS_UpdatePage(&outputData->SDDStable, 0)) {
    SDDS_PrintErrors(stderr, SDDS_VERBOSE_PrintErrors | SDDS_EXIT_PrintErrors);
  }
}
