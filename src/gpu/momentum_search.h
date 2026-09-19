/* Internal momentumAperture.c implementation. Each lane advances the original
 * scalar search state machine: do not assume monotonic survival or replace
 * repeated addition by start + index*step (the grid endpoints can differ).
 * This path is opt-in until the full fresh-baseline certification passes. */
typedef struct {
  ELEMENT_LIST *element;
  double pref, phase, time;
  long seen, reference;
} MOMENTUM_REFERENCE_ELEMENT;

typedef struct {
  MOMENTUM_REFERENCE_ELEMENT *element;
  long count;
  void *phases;
} MOMENTUM_REFERENCE;

typedef struct {
  long target, side, split, done, survivor, loser, actualSurvivor, actualLoser;
  double delta, interval, limit, survivedDelta, lostDelta;
  double xLost, yLost, sLost, deltaLost, tune[2];
  long lostPass;
} MOMENTUM_LANE;

static void momentumReferenceCapture(MOMENTUM_REFERENCE *state, LINE_LIST *beamline) {
  ELEMENT_LIST *eptr;
  long i = 0;
  memset(state, 0, sizeof(*state));
  for (eptr = beamline->elem; eptr; eptr = eptr->succ)
    state->count++;
  state->element = tmalloc(state->count * sizeof(*state->element));
  for (eptr = beamline->elem; eptr; eptr = eptr->succ, i++) {
    MOMENTUM_REFERENCE_ELEMENT *entry = state->element + i;
    entry->element = eptr;
    entry->pref = eptr->Pref_output_fiducial;
    if (eptr->type == T_RFCA) {
      RFCA *rf = (RFCA *)eptr->p_elem;
      entry->seen = rf->fiducial_seen;
      entry->reference = rf->phase_reference;
      entry->phase = rf->phase_fiducial;
      entry->time = rf->t_fiducial;
    }
  }
  state->phases = save_phase_references();
}

static void momentumReferenceRestore(MOMENTUM_REFERENCE *state, LINE_LIST *beamline) {
  long i;
  reset_special_elements(beamline, RESET_INCLUDE_ALL & ~RESET_INCLUDE_RANDOM);
  restore_phase_references(state->phases);
  for (i = 0; i < state->count; i++) {
    MOMENTUM_REFERENCE_ELEMENT *entry = state->element + i;
    entry->element->Pref_output_fiducial = entry->pref;
    if (entry->element->type == T_RFCA) {
      RFCA *rf = (RFCA *)entry->element->p_elem;
      rf->fiducial_seen = entry->seen;
      rf->phase_reference = entry->reference;
      rf->phase_fiducial = entry->phase;
      rf->t_fiducial = entry->time;
    }
  }
}

static void momentumReferenceFree(MOMENTUM_REFERENCE *state) {
  free_phase_references(state->phases);
  free(state->element);
}

static void momentumLaneInit(MOMENTUM_LANE *lane, long target, long side) {
  memset(lane, 0, sizeof(*lane));
  lane->target = target;
  lane->side = side;
  lane->delta = side ? delta_positive_start : delta_negative_start;
  lane->limit = side ? delta_positive_limit : delta_negative_limit;
  lane->interval = (side ? 1 : -1) * delta_step_size;
  lane->lostDelta = (side ? 1 : -1) * DBL_MAX / 2;
  lane->lostPass = -1;
}

/* Exactly the split transition in the scalar loop below. */
static void momentumLaneFinishSplit(MOMENTUM_LANE *lane) {
  if (lane->split == 0) {
    if (!lane->survivor) {
      if (!soft_failure)
        bombElegant("No survivor found for initial momentum scan", NULL);
      lane->survivedDelta = 0;
      lane->survivor = 1;
      lane->done = 1;
    }
    if (!lane->loser) {
      if (!soft_failure)
        bombElegant("No loss found for initial momentum scan", NULL);
      lane->loser = 1;
      lane->done = 1;
    }
  }
  if (lane->split++ >= splits)
    lane->done = 1;
  if (lane->done)
    return;
  lane->delta = lane->survivedDelta - steps_back * lane->interval;
  lane->interval /= split_step_divisor;
  lane->delta += lane->interval;
  lane->limit = lane->lostDelta;
  if ((lane->delta < 0 && lane->side) || (lane->delta > 0 && !lane->side))
    lane->delta = 0;
}

static void momentumLaneResult(MOMENTUM_LANE *lane, long survived,
                               double *coord, double pCentral) {
  if (!survived) {
    lane->lostPass = (long)coord[lossPassIndex];
    lane->xLost = coord[0];
    lane->yLost = coord[2];
    lane->sLost = coord[4];
    lane->deltaLost = (coord[5] - pCentral) / pCentral;
    lane->lostDelta = lane->delta;
    lane->loser = lane->actualLoser = 1;
    momentumLaneFinishSplit(lane);
  } else {
    lane->survivedDelta = lane->delta;
    lane->survivor = lane->actualSurvivor = 1;
    lane->delta += lane->interval;
  }
  while (!lane->done && fabs(lane->delta) > fabs(lane->limit))
    momentumLaneFinishSplit(lane);
}

static long momentumScalarTrial(RUN *run, VARY *control, LINE_LIST *beamline,
                                double *startingCoord, ELEMENT_LIST *target,
                                double delta, double **coord, double *pCentral,
                                MOMENTUM_REFERENCE *prepared) {
  long code;
  memset(coord[0], 0, totalPropertiesPerParticle * sizeof(**coord));
  if (startingCoord)
    memcpy(coord[0], startingCoord, 6 * sizeof(**coord));
  coord[0][particleIDIndex] = 1;
  turnsStored = 0;
  momentumOffsetValue = delta;
  if (!fiducialize) {
    delete_phase_references();
    reset_special_elements(beamline, RESET_INCLUDE_ALL & ~RESET_INCLUDE_RANDOM);
  } else
    momentumReferenceRestore(prepared, beamline);
  setTrackingWedgeFunction(momentumOffsetFunction, target);
  *pCentral = run->p_central;
  code = do_tracking(NULL, coord, 1, NULL, beamline, pCentral,
                     NULL, NULL, NULL, NULL, run, control->i_step,
                     (fiducialize ? FIDUCIAL_BEAM_SEEN : 0) + FIRST_BEAM_IS_FIDUCIAL +
                       SILENT_RUNNING + INHIBIT_FILE_OUTPUT + MOMENTUM_APERTURE_TRACKING_FLAGS,
                     control->n_passes, 0, NULL, NULL, NULL, NULL, NULL);
  setTrackingWedgeFunction(NULL, NULL);
  return code;
}

static long momentumScalarLane(MOMENTUM_LANE *lane, RUN *run, VARY *control,
                               LINE_LIST *beamline, double *startingCoord,
                               ELEMENT_LIST *target, double **coord,
                               MOMENTUM_REFERENCE *prepared, long *replays) {
  long index = lane->target, side = lane->side, tuneKnown = 0;
  double tune[2] = {-1, -1};
  momentumLaneInit(lane, index, side);
  while (!lane->done) {
    double pCentral;
    long code;
    if (fabs(lane->delta) > fabs(lane->limit)) {
      momentumLaneFinishSplit(lane);
      continue;
    }
    code = momentumScalarTrial(run, control, beamline, startingCoord,
                               target, lane->delta, coord, &pCentral, prepared);
    (*replays)++;
    /* Scalar tune extraction leaves the previous trial's tunes untouched
     * when NAFF fails. Preserve that history across this whole search. */
    if (code && turnsStored > 2) {
      if (determineTunesFromTrackingData(tune, turnByTurnCoord, turnsStored, lane->delta))
        tuneKnown = 1;
      else if (!tuneKnown)
        /* The original scalar variable may carry a prior direction's tune.
         * Delegate the whole command rather than invent that history. */
        return 0;
    } else {
      tune[0] = tune[1] = -1;
      tuneKnown = 1;
    }
    if (code) {
      lane->tune[0] = tune[0];
      lane->tune[1] = tune[1];
    }
    momentumLaneResult(lane, code, coord[0], pCentral);
  }
  return 1;
}

static long momentumTurnLinksActive(LINE_LIST *beamline) {
  long i;
  if (beamline->links)
    for (i = 0; i < beamline->links->n_links; i++)
      if (beamline->links->flags[i] & TURN_BY_TURN_LINK)
        return 1;
  return 0;
}

static long doMomentumApertureSearchIndependent(
  RUN *run, VARY *control, LINE_LIST *beamline, double *startingCoord,
  ELEMENT_LIST *elem0, long nElem,
  int32_t **lostOnPass, short **loserFound, short **survivorFound,
  double **deltaSurvived, double **xTuneSurvived,
  double **yTuneSurvived, double **xLost, double **yLost,
  double **deltaWhenLost, double **sLost,
  double *sStart, char **ElementName, char **ElementType,
  int32_t *ElementOccurence, short *direction) {
  const char *option = getenv("ELEGANT_GPU_ENABLE_NONFIDUCIAL_MOMENTUM_SEARCH");
  const char *batchBoundaryOption = getenv("ELEGANT_GPU_BATCH_MOMENTUM_BOUNDARY_REPLAY");
  const char *profileBoundaryOption = getenv("ELEGANT_GPU_PROFILE_MOMENTUM_BOUNDARY_REPLAY");
  const char *reason = NULL;
  MOMENTUM_REFERENCE entry, prepared;
  MOMENTUM_LANE *lane;
  ELEMENT_LIST **target;
  double **coord, **resultCoord, **scalar;
  double *delta, *history, *historyCount;
  long *targetById, *laneById, *splitById;
  unsigned char *seen;
  unsigned char *boundaryMismatch;
  long capacity, tasks, i, offset, count, active, left, ip, id, code;
  long suppressed, replays = 0, fallbacks = 0, gpuElements = 0, batches = 0;
  long scalarCommandFallback = 0;
  long batchBoundaryReplay = !batchBoundaryOption ||
    strtol(batchBoundaryOption, NULL, 10) != 0;
  long profileBoundaryReplay = profileBoundaryOption &&
    strtol(profileBoundaryOption, NULL, 10) != 0;
  long savedShowElementTiming = run->showElementTiming;
  double pCentral;

  /* Default on for eligible searches; retain an explicit scalar-path override. */
  if (option && strtol(option, NULL, 10) == 0)
    return -1;
  if (fiducialize && gpu_batched_search_beamline_supported(beamline))
    return -1;
  if (control->n_passes <= 1)
    reason = "one-turn search has no shared unperturbed first turn";
  else if (run->n_passes_fiducial > 0)
    reason = "additional fiducial passes are not supported";
  else if (run->always_change_p0)
    reason = "always_change_p0 changes the reference momentum";
  else if (run->modulationData.nItems)
    reason = "element modulation has unsupported time-dependent state";
  else if (run->rampData.nItems)
    reason = "element ramps have unsupported mutable state";
  else if (momentumTurnLinksActive(beamline))
    reason = "turn-by-turn element links have unsupported mutable state";
  else if (run->stopTrackingParticleLimit > 0)
    reason = "particle-count stopping condition depends on batch size";
  else if (forbid_resonance_crossing)
    reason = "resonance-crossing search requires scalar history decisions";
  else if (allow_watch_file_output)
    reason = "watch-file output is enabled";
  else if (output_mode > 1)
    reason = "unsupported momentum output mode";
  else if (nElem <= 0 || nElem > LONG_MAX / 2)
    reason = "invalid or oversized selected location count";
  else if (trackingWedgeFunctionsActive())
    reason = "external tracking callbacks are already installed";
  else if (gpuGetTrackingSuppressed())
    reason = "GPU tracking is explicitly suppressed";
  else if (!gpu_batched_search_tracking_enabled(2 * nElem))
    reason = "batch below threshold or batched search disabled";
  else
    gpu_momentum_search_beamline_supported(beamline, &reason);
  if (reason) {
    fprintf(stderr, "elegant CUDA: independent momentum search fallback: %s.\n", reason);
    return -1;
  }
  capacity = gpu_momentum_search_batch_capacity(control->n_passes, totalPropertiesPerParticle);
  if (capacity > 0) {
    long side, initial = 0;
    for (side = 0; side < 2 && initial < capacity; side++) {
      double value = side ? delta_positive_start : delta_negative_start;
      double limit = side ? delta_positive_limit : delta_negative_limit;
      double step = side ? delta_step_size : -delta_step_size;
      while (fabs(value) <= fabs(limit) && initial < capacity) {
        initial++;
        if (value + step == value)
          bombElegant("momentum step does not advance the search", NULL);
        value += step;
      }
    }
    capacity = initial > capacity / nElem ? capacity : initial * nElem;
  }
  if (capacity <= 0) {
    fprintf(stderr, "elegant CUDA: independent momentum search fallback: no batch memory budget.\n");
    return -1;
  }
  momentumReferenceCapture(&entry, beamline);
  suppressed = gpuGetTrackingSuppressed();
  scalar = (double **)czarray_2d(sizeof(**scalar), 1, totalPropertiesPerParticle);
  if (!fiducialize) {
    /* All trials have the same unperturbed first turn. Compute RF fiducials
     * with the scalar particle, not a floating-point reduction over copies.
     * Eligibility guarantees constant reference momentum and excludes clocks,
     * wakefields, noise, and other candidate-dependent mutable state. */
    if (startingCoord)
      memcpy(scalar[0], startingCoord, 6 * sizeof(**scalar));
    scalar[0][particleIDIndex] = 1;
    delete_phase_references();
    reset_special_elements(beamline, RESET_INCLUDE_ALL & ~RESET_INCLUDE_RANDOM);
    gpuSetTrackingSuppressed(1);
    pCentral = run->p_central;
    code = do_tracking(NULL, scalar, 1, NULL, beamline, &pCentral,
                       NULL, NULL, NULL, NULL, run, control->i_step,
                       FIRST_BEAM_IS_FIDUCIAL + SILENT_RUNNING + INHIBIT_FILE_OUTPUT +
                         MOMENTUM_APERTURE_TRACKING_FLAGS,
                       1, 0, NULL, NULL, NULL, NULL, NULL);
    gpuSetTrackingSuppressed(suppressed);
    if (!code || pCentral != run->p_central) {
      momentumReferenceRestore(&entry, beamline);
      momentumReferenceFree(&entry);
      free_czarray_2d((void **)scalar, 1, totalPropertiesPerParticle);
      fprintf(stderr, "elegant CUDA: independent momentum search fallback: reference trial lost or momentum changed.\n");
      return -1;
    }
  }
  momentumReferenceCapture(&prepared, beamline);
  tasks = 2 * nElem;
  lane = tmalloc(tasks * sizeof(*lane));
  target = tmalloc(nElem * sizeof(*target));
  for (i = 0; i < nElem; i++) {
    target[i] = elementArray[i]->succ ? elementArray[i]->succ : elem0;
    momentumLaneInit(lane + 2 * i, i, 0);
    momentumLaneInit(lane + 2 * i + 1, i, 1);
  }
  coord = (double **)czarray_2d(sizeof(**coord), capacity, totalPropertiesPerParticle);
  resultCoord = (double **)czarray_2d(sizeof(**resultCoord), capacity, totalPropertiesPerParticle);
  delta = tmalloc(capacity * sizeof(*delta));
  targetById = tmalloc(capacity * sizeof(*targetById));
  laneById = tmalloc(capacity * sizeof(*laneById));
  splitById = tmalloc(capacity * sizeof(*splitById));
  seen = tmalloc(capacity * sizeof(*seen));
  boundaryMismatch = tmalloc(tasks * sizeof(*boundaryMismatch));
  history = tmalloc((size_t)capacity * 5 * control->n_passes * sizeof(*history));
  historyCount = tmalloc(capacity * sizeof(*historyCount));
  fprintf(stderr, "elegant CUDA: independent momentum search: %ld lanes, capacity %ld, fiducialize=%ld.\n",
          tasks, capacity, (long)fiducialize);
  do {
    active = 0;
    for (offset = 0; offset < tasks;) {
      count = 0;
      while (offset < tasks && count < capacity) {
        double value;
        i = offset;
        while (!lane[i].done && fabs(lane[i].delta) > fabs(lane[i].limit))
          momentumLaneFinishSplit(lane + i);
        if (lane[i].done) {
          offset++;
          continue;
        }
        value = lane[i].delta;
        /* Speculate within this split only. Results are consumed in scalar
         * order; candidates beyond the first loss never change the state. */
        while (count < capacity && fabs(value) <= fabs(lane[i].limit)) {
          memset(coord[count], 0, totalPropertiesPerParticle * sizeof(**coord));
          if (startingCoord)
            memcpy(coord[count], startingCoord, 6 * sizeof(**coord));
          coord[count][particleIDIndex] = count + 1;
          laneById[count] = i;
          splitById[count] = lane[i].split;
          delta[count] = value;
          targetById[count] = lane[i].target;
          count++;
          if (value + lane[i].interval == value)
            bombElegant("momentum step does not advance the search", NULL);
          value += lane[i].interval;
        }
        if (fabs(value) > fabs(lane[i].limit))
          offset++;
      }
      if (!count)
        continue;
      active += count;
      memset(seen, 0, count * sizeof(*seen));
      momentumReferenceRestore(&prepared, beamline);
      batchedMomentumTargetElement = target;
      batchedMomentumTargets = nElem;
      batchedMomentumTargetById = targetById;
      batchedMomentumDeltaById = delta;
      batchedMomentumHistory = history;
      batchedMomentumHistoryCount = historyCount;
      batchedMomentumParticles = count;
      batchedMomentumTurns = control->n_passes;
      gpu_configure_batched_momentum_search(delta, targetById, count,
                                             control->n_passes, fireOnPass,
                                             history, historyCount);
      setTrackingOmniWedgeFunction(momentumOffsetFunctionBatched);
      setTrackingOmniWedgeGpuFunction(momentumOffsetFunctionBatchedGpu);
      gpu_momentum_search_batch_scope(1);
      pCentral = run->p_central;
#if defined(_OPENMP)
      if (gpuOmpTrackingRequested(count)) {
        /* Keep one worker team alive across GPU tracking so that any
         * loss-sensitive CPU fallback can reuse it. */
#  pragma omp parallel num_threads(gpuGetOmpTrackingThreads())
        {
#  pragma omp single
          left = do_tracking(NULL, coord, count, NULL, beamline, &pCentral,
                             NULL, NULL, NULL, NULL, run, control->i_step,
                             FIRST_BEAM_IS_FIDUCIAL + FIDUCIAL_BEAM_SEEN + SILENT_RUNNING +
                               INHIBIT_FILE_OUTPUT + MOMENTUM_APERTURE_TRACKING_FLAGS,
                             control->n_passes, 0, NULL, NULL, NULL, NULL, NULL);
        }
      } else
#endif
        left = do_tracking(NULL, coord, count, NULL, beamline, &pCentral,
                           NULL, NULL, NULL, NULL, run, control->i_step,
                           FIRST_BEAM_IS_FIDUCIAL + FIDUCIAL_BEAM_SEEN + SILENT_RUNNING +
                             INHIBIT_FILE_OUTPUT + MOMENTUM_APERTURE_TRACKING_FLAGS,
                           control->n_passes, 0, NULL, NULL, NULL, NULL, NULL);
      gpuElements += getGpuBase()->gpuElementCount;
      batches++;
      fprintf(stderr, "elegant CUDA: independent momentum batch %ld completed: trials=%ld survivors=%ld gpuElements=%ld.\n",
              batches, count, left, getGpuBase()->gpuElementCount);
      gpu_momentum_search_batch_scope(0);
      setTrackingOmniWedgeGpuFunction(NULL);
      setTrackingOmniWedgeFunction(NULL);
      gpu_clear_batched_momentum_search();
      clearBatchedMomentumCallbackState();
      for (ip = 0; ip < count; ip++) {
        id = (long)coord[ip][particleIDIndex] - 1;
        if (id < 0 || id >= count || seen[id])
          bombElegantVA("invalid independent momentum lane ID: row=%ld id=%ld count=%ld seen=%ld raw=%.17g",
                        ip, id, count,
                        id >= 0 && id < count ? (long)seen[id] : -1L,
                        coord[ip][particleIDIndex]);
        seen[id] = ip < left ? 2 : 1;
        memcpy(resultCoord[id], coord[ip], totalPropertiesPerParticle * sizeof(**coord));
      }
      for (id = 0; id < count; id++) {
        MOMENTUM_LANE *item = lane + laneById[id];
        if (!seen[id])
          bombElegant("missing independent momentum lane ID", NULL);
        if (item->done || item->split != splitById[id])
          continue;
        if (item->delta != delta[id])
          bombElegant("independent momentum grid changed order", NULL);
        momentumLaneResult(item, seen[id] == 2, resultCoord[id], pCentral);
      }
    }
  } while (active);

  gpuSetTrackingSuppressed(1);
  memset(boundaryMismatch, 0, tasks * sizeof(*boundaryMismatch));
  if (batchBoundaryReplay) {
    count = 0;
    for (i = 0; i < tasks; i++) {
      MOMENTUM_LANE *item = lane + i;
      if (item->actualSurvivor) {
        if (count >= capacity) {
          batchBoundaryReplay = 0;
          break;
        }
        memset(coord[count], 0, totalPropertiesPerParticle * sizeof(**coord));
        if (startingCoord)
          memcpy(coord[count], startingCoord, 6 * sizeof(**coord));
        coord[count][particleIDIndex] = count + 1;
        targetById[count] = item->target;
        delta[count] = item->survivedDelta;
        laneById[count] = i;
        splitById[count] = 0;
        count++;
      }
      if (item->actualLoser) {
        if (count >= capacity) {
          batchBoundaryReplay = 0;
          break;
        }
        memset(coord[count], 0, totalPropertiesPerParticle * sizeof(**coord));
        if (startingCoord)
          memcpy(coord[count], startingCoord, 6 * sizeof(**coord));
        coord[count][particleIDIndex] = count + 1;
        targetById[count] = item->target;
        delta[count] = item->lostDelta;
        laneById[count] = i;
        splitById[count] = 1;
        count++;
      }
    }
    if (batchBoundaryReplay) {
      fprintf(stderr, "elegant CUDA: confirming %ld momentum boundary points in one OpenMP CPU ensemble.\n", count);
      memset(seen, 0, count * sizeof(*seen));
      momentumReferenceRestore(&prepared, beamline);
      batchedMomentumTargetElement = target;
      batchedMomentumTargets = nElem;
      batchedMomentumTargetById = targetById;
      batchedMomentumDeltaById = delta;
      batchedMomentumHistory = history;
      batchedMomentumHistoryCount = historyCount;
      batchedMomentumParticles = count;
      batchedMomentumTurns = control->n_passes;
      memset(historyCount, 0, count * sizeof(*historyCount));
      setTrackingOmniWedgeFunction(momentumOffsetFunctionBatched);
      gpu_momentum_search_batch_scope(1);
      pCentral = run->p_central;
      if (profileBoundaryReplay) {
        resetElementTiming();
        run->showElementTiming = 1;
      }
#if defined(_OPENMP)
      if (gpuOmpTrackingRequested(count)) {
        /* Keep one worker team alive across the complete CPU confirmation.
         * CSBEND, CCBEND, LGBEND, exact drifts, and multipole taskloops then
         * reuse this team instead of creating a team for every element. */
#  pragma omp parallel num_threads(gpuGetOmpTrackingThreads())
        {
#  pragma omp single
          left = do_tracking(NULL, coord, count, NULL, beamline, &pCentral,
                             NULL, NULL, NULL, NULL, run, control->i_step,
                             FIRST_BEAM_IS_FIDUCIAL + FIDUCIAL_BEAM_SEEN + SILENT_RUNNING +
                               INHIBIT_FILE_OUTPUT + MOMENTUM_APERTURE_TRACKING_FLAGS,
                             control->n_passes, 0, NULL, NULL, NULL, NULL, NULL);
        }
      } else
#endif
        left = do_tracking(NULL, coord, count, NULL, beamline, &pCentral,
                           NULL, NULL, NULL, NULL, run, control->i_step,
                           FIRST_BEAM_IS_FIDUCIAL + FIDUCIAL_BEAM_SEEN + SILENT_RUNNING +
                             INHIBIT_FILE_OUTPUT + MOMENTUM_APERTURE_TRACKING_FLAGS,
                           control->n_passes, 0, NULL, NULL, NULL, NULL, NULL);
      if (profileBoundaryReplay) {
        reportElementTiming();
        run->showElementTiming = savedShowElementTiming;
      }
      gpu_momentum_search_batch_scope(0);
      setTrackingOmniWedgeFunction(NULL);
      clearBatchedMomentumCallbackState();
      replays += count;
      for (ip = 0; ip < count; ip++) {
        id = (long)coord[ip][particleIDIndex] - 1;
        if (id < 0 || id >= count || seen[id])
          bombElegant("invalid boundary replay lane ID", NULL);
        seen[id] = ip < left ? 2 : 1;
        memcpy(resultCoord[id], coord[ip],
               totalPropertiesPerParticle * sizeof(**coord));
      }
      for (id = 0; id < count; id++) {
        MOMENTUM_LANE *item = lane + laneById[id];
        if (!seen[id])
          bombElegant("missing boundary replay lane ID", NULL);
        if (splitById[id] == 0) {
          double *turnCoord[5];
          long ic;
          if (seen[id] != 2)
            boundaryMismatch[laneById[id]] = 1;
          item->tune[0] = item->tune[1] = -1;
          if (seen[id] == 2 && (long)historyCount[id] > 2) {
            for (ic = 0; ic < 5; ic++)
              turnCoord[ic] = history +
                ((size_t)id * 5 + ic) * control->n_passes;
            if (!determineTunesFromTrackingData(item->tune, turnCoord,
                                                (long)historyCount[id], delta[id]))
              boundaryMismatch[laneById[id]] = 1;
          }
        } else {
          if (seen[id] == 2)
            boundaryMismatch[laneById[id]] = 1;
          else {
            item->lostPass = (long)resultCoord[id][lossPassIndex];
            item->xLost = resultCoord[id][0];
            item->yLost = resultCoord[id][2];
            item->sLost = resultCoord[id][4];
            item->deltaLost = (resultCoord[id][5] - pCentral) / pCentral;
          }
        }
      }
    } else
      fprintf(stderr, "elegant CUDA: boundary ensemble exceeds batch capacity; using scalar CPU confirmation.\n");
  }
  for (i = 0; i < tasks; i++) {
    MOMENTUM_LANE *item = lane + i;
    long mismatch = batchBoundaryReplay ? boundaryMismatch[i] : 0;
    long row = output_mode ? i : i / 2;
    long slot = output_mode ? 0 : item->side;
    if (!batchBoundaryReplay && item->actualSurvivor) {
      code = momentumScalarTrial(run, control, beamline, startingCoord,
                                 target[item->target], item->survivedDelta,
                                 scalar, &pCentral, &prepared);
      replays++;
      mismatch = !code;
      item->tune[0] = item->tune[1] = -1;
      if (code && turnsStored > 2 &&
          !determineTunesFromTrackingData(item->tune, turnByTurnCoord, turnsStored, item->survivedDelta)) {
        /* The scalar output may depend on an earlier successful tune fit. */
        fprintf(stderr, "elegant CUDA: momentum boundary tune replay failed at %s#%ld; scalar search required.\n",
                target[item->target]->name, target[item->target]->occurence);
        mismatch = 1;
      }
    }
    if (!batchBoundaryReplay && item->actualLoser) {
      code = momentumScalarTrial(run, control, beamline, startingCoord,
                                 target[item->target], item->lostDelta,
                                 scalar, &pCentral, &prepared);
      replays++;
      mismatch |= code != 0;
      if (!code) {
        item->lostPass = (long)scalar[0][lossPassIndex];
        item->xLost = scalar[0][0];
        item->yLost = scalar[0][2];
        item->sLost = scalar[0][4];
        item->deltaLost = (scalar[0][5] - pCentral) / pCentral;
      }
    }
    if (mismatch) {
      fprintf(stderr, "elegant CUDA: momentum boundary replay requires fallback at %s#%ld direction=%ld; rerunning complete scalar search.\n",
              target[item->target]->name, target[item->target]->occurence,
              item->side ? 1L : -1L);
      fallbacks++;
      if (!momentumScalarLane(item, run, control, beamline, startingCoord,
                              target[item->target], scalar, &prepared, &replays)) {
        scalarCommandFallback = 1;
        fprintf(stderr, "elegant CUDA: prior-direction tune history required; restoring state and rerunning the original scalar command.\n");
        break;
      }
    }
    lostOnPass[slot][row] = item->lostPass;
    loserFound[slot][row] = item->loser;
    survivorFound[slot][row] = item->survivor;
    deltaSurvived[slot][row] = item->survivedDelta;
    xTuneSurvived[slot][row] = item->tune[0];
    yTuneSurvived[slot][row] = item->tune[1];
    xLost[slot][row] = item->xLost;
    yLost[slot][row] = item->yLost;
    deltaWhenLost[slot][row] = item->deltaLost;
    sLost[slot][row] = item->sLost;
    sStart[row] = elementArray[item->target]->end_pos;
    ElementName[row] = elementArray[item->target]->name;
    ElementType[row] = entity_name[elementArray[item->target]->type];
    ElementOccurence[row] = elementArray[item->target]->occurence;
    if (output_mode)
      direction[row] = item->side ? 1 : -1;
  }
  gpuSetTrackingSuppressed(suppressed);
  momentumReferenceRestore(&entry, beamline);
  momentumReferenceFree(&entry);
  momentumReferenceFree(&prepared);
  free_czarray_2d((void **)scalar, 1, totalPropertiesPerParticle);
  free_czarray_2d((void **)coord, capacity, totalPropertiesPerParticle);
  free_czarray_2d((void **)resultCoord, capacity, totalPropertiesPerParticle);
  free(lane); free(target); free(delta); free(targetById); free(laneById); free(splitById);
  free(seen); free(boundaryMismatch); free(history); free(historyCount);
  fprintf(stderr, "elegant CUDA: independent momentum summary: batches=%ld gpuElements=%ld cpuReplays=%ld scalarFallbacks=%ld.\n",
          batches, gpuElements, replays, fallbacks);
  return scalarCommandFallback ? -1 : (output_mode ? tasks : nElem) - 1;
}
