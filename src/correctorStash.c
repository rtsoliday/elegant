/*************************************************************************\
 * Copyright (c) 2026 The University of Chicago, as Operator of Argonne
 * National Laboratory.
 * This file is distributed subject to a Software License Agreement found
 * in the file LICENSE that is included with this distribution.
\*************************************************************************/

/* file: correctorStash.c
 * purpose: implementation of the per-correction-type corrector stash that
 *          drives per-step reassertion in vary_beamline().  See
 *          correctorStash.h for the API.
 */

#include "mdb.h"
#include "track.h"
#include "correctorStash.h"

/* Shared flag: set to 0 by &run_setup, set to 1 on the first vary_beamline()
 * tick that takes a snapshot. */
short corrector_stash_snapshotted = 0;

/* ---------------------------------------------------------------- */

void corstash_init(CORRECTOR_STASH *s, const char *label) {
  s->elem        = NULL;
  s->paramIndex  = NULL;
  s->valuePtr    = NULL;
  s->savedValue  = NULL;
  s->n           = 0;
  s->capacity    = 0;
  s->reassert    = 1;
  s->saved       = 0;
  s->label       = label;
}

void corstash_clear(CORRECTOR_STASH *s) {
  s->n = 0;
  s->saved = 0;
}

void corstash_free(CORRECTOR_STASH *s) {
  if (s->elem)       { free(s->elem);       s->elem = NULL; }
  if (s->paramIndex) { free(s->paramIndex); s->paramIndex = NULL; }
  if (s->valuePtr)   { free(s->valuePtr);   s->valuePtr = NULL; }
  if (s->savedValue) { free(s->savedValue); s->savedValue = NULL; }
  s->n = 0;
  s->capacity = 0;
  s->saved = 0;
}

void corstash_add(CORRECTOR_STASH *s, ELEMENT_LIST *elem, long paramIndex) {
  if (elem == NULL || paramIndex < 0) return;
  if (entity_description[elem->type].parameter[paramIndex].type != IS_DOUBLE)
    return;
  if (s->n == s->capacity) {
    s->capacity = s->capacity ? 2 * s->capacity : 32;
    s->elem       = SDDS_Realloc(s->elem,       sizeof(*s->elem)       * s->capacity);
    s->paramIndex = SDDS_Realloc(s->paramIndex, sizeof(*s->paramIndex) * s->capacity);
    s->valuePtr   = SDDS_Realloc(s->valuePtr,   sizeof(*s->valuePtr)   * s->capacity);
    s->savedValue = SDDS_Realloc(s->savedValue, sizeof(*s->savedValue) * s->capacity);
  }
  s->elem[s->n]       = elem;
  s->paramIndex[s->n] = paramIndex;
  s->valuePtr[s->n]   = (double *)(elem->p_elem +
      entity_description[elem->type].parameter[paramIndex].offset);
  s->savedValue[s->n] = 0;
  s->n++;
  /* Adding a new corrector invalidates any prior snapshot. */
  s->saved = 0;
}

void corstash_snapshot(CORRECTOR_STASH *s) {
  long i;
  for (i = 0; i < s->n; i++)
    s->savedValue[i] = *s->valuePtr[i];
  s->saved = 1;
}

long corstash_reassert(CORRECTOR_STASH *s, RUN *run, LINE_LIST *beamline) {
  long i, restored = 0;
  if (!s->saved || !s->reassert || s->n == 0) return 0;
  for (i = 0; i < s->n; i++) {
    ELEMENT_LIST *e = s->elem[i];
    *s->valuePtr[i] = s->savedValue[i];
    if (e->matrix) {
      free_matrices(e->matrix);
      free(e->matrix);
      e->matrix = NULL;
    }
    compute_matrix(e, run, NULL);
    /*
    Claude put this here, but it is not correct.
    change_defined_parameter(e->name, s->paramIndex[i], e->type,
                             s->savedValue[i], NULL, LOAD_FLAG_ABSOLUTE);
    */
    restored++;
  }
  return restored;
}

/* ---------------------------------------------------------------- */
/* Dispatch entry points.  Each correction module's .c file provides one
 * implementation pair (save + reassert + invalidate); the dispatcher
 * below just routes to them.  The five save/reassert routines are
 * defined in correct_*.c (per the API in correctorStash.h); the
 * invalidator walks all of them in turn here. */

extern void correct_invalidate_correctors(void);
extern void correct_tunes_invalidate_correctors(void);
extern void correct_chromaticity_invalidate_correctors(void);
extern void correct_coupling_invalidate_correctors(void);
extern void correct_lattice_invalidate_correctors(void);

void corrector_stash_invalidate_all(void) {
  correct_invalidate_correctors();
  correct_tunes_invalidate_correctors();
  correct_chromaticity_invalidate_correctors();
  correct_coupling_invalidate_correctors();
  correct_lattice_invalidate_correctors();
  corrector_stash_snapshotted = 0;
}

long corrector_stash_step_tick(RUN *run, LINE_LIST *beamline) {
  long restored = 0;
  if (!corrector_stash_snapshotted) {
    correct_save_correctors(run, beamline);
    correct_tunes_save_correctors(run, beamline);
    correct_chromaticity_save_correctors(run, beamline);
    correct_coupling_save_correctors(run, beamline);
    correct_lattice_save_correctors(run, beamline);
    corrector_stash_snapshotted = 1;
  } else {
    restored += correct_reassert_correctors(run, beamline);
    restored += correct_tunes_reassert_correctors(run, beamline);
    restored += correct_chromaticity_reassert_correctors(run, beamline);
    restored += correct_coupling_reassert_correctors(run, beamline);
    restored += correct_lattice_reassert_correctors(run, beamline);
  }
  return restored;
}
