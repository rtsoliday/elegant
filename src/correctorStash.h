/*************************************************************************\
 * Copyright (c) 2026 The University of Chicago, as Operator of Argonne
 * National Laboratory.
 * This file is distributed subject to a Software License Agreement found
 * in the file LICENSE that is included with this distribution.
\*************************************************************************/

/* file: correctorStash.h
 * purpose: per-correction-type save/restore of corrector values across the
 *          steps loop in vary_beamline().
 *
 * Each of the five correction commands (correct, correct_tunes,
 * chromaticity, correct_coupling, correct_lattice) owns one
 * CORRECTOR_STASH that records the *valuePtr addresses of every element
 * parameter it adjusts.  On the first vary_beamline() tick after a fresh
 * &run_setup the stash is snapshotted (corstash_snapshot); on every
 * subsequent tick the snapshot is reasserted (corstash_reassert) before
 * perturb_beamline() runs.  This restores the corrector values to their
 * post-&run_setup state so each step starts with a clean slate rather
 * than inheriting the previous step's converged correctors.  Per-command
 * "reset_correctors_each_step" namelist flags control reassertion at
 * the granularity of one correction type.
 *
 * Ordering inside vary_beamline(): the stash tick runs BEFORE
 * do_load_parameters() so that &load_parameters with change_defined_values=0
 * (which executes inside the steps loop) overrides the reasserted snapshot
 * for the current step.  &load_parameters with change_defined_values=1
 * takes effect at command-issue time (outside the loop) and is therefore
 * baked into the first-tick snapshot.
 *
 * The caller must include "track.h" before this header.
 */

#ifndef CORRECTOR_STASH_H
#define CORRECTOR_STASH_H

typedef struct {
  ELEMENT_LIST **elem;       /* corrector elements (one entry per item)     */
  long  *paramIndex;         /* index in entity_description[type].parameter */
  double **valuePtr;         /* direct pointer into elem->p_elem            */
  double *savedValue;        /* value at snapshot time                      */
  long n;                    /* number of populated entries                 */
  long capacity;             /* allocated capacity                          */
  short reassert;            /* 1 = restore at step start (default);        *
                              * 0 = inherit previous-step values            */
  short saved;               /* 1 = savedValue[] populated                  */
  const char *label;         /* short tag for diagnostic prints             */
} CORRECTOR_STASH;

/* Initialize an empty stash with a short text label (e.g. "coupling"). */
void corstash_init(CORRECTOR_STASH *s, const char *label);

/* Append (elem, paramIndex) to the stash.  Computes valuePtr from elem->p_elem
 * and entity_description[].parameter[].offset.  Does NOT snapshot the value
 * yet -- call corstash_snapshot() once the corrector inventory is finalized. */
void corstash_add(CORRECTOR_STASH *s, ELEMENT_LIST *elem, long paramIndex);

/* Drop the element list AND the saved values.  Used when a correction
 * command re-runs setup (different inventory) or when &run_setup invalidates
 * everything. */
void corstash_clear(CORRECTOR_STASH *s);

/* Snapshot the current *valuePtr[i] into savedValue[i].  Marks saved=1. */
void corstash_snapshot(CORRECTOR_STASH *s);

/* If saved && reassert: write savedValue[i] back through valuePtr[i], free
 * + compute_matrix() the element, and change_defined_parameter() so subsequent
 * twiss/parameter output reflects the restored value.  Returns the number of
 * elements actually restored (0 if reassert==0 or never saved). */
long corstash_reassert(CORRECTOR_STASH *s, RUN *run, LINE_LIST *beamline);

/* Free internal buffers; safe to call multiple times. */
void corstash_free(CORRECTOR_STASH *s);

/* ---- Cross-module dispatch from vary_beamline() ------------------------ */

/* Tell each correction module to snapshot its current corrector values. */
void correct_save_correctors(RUN *run, LINE_LIST *beamline);
void correct_tunes_save_correctors(RUN *run, LINE_LIST *beamline);
void correct_chromaticity_save_correctors(RUN *run, LINE_LIST *beamline);
void correct_coupling_save_correctors(RUN *run, LINE_LIST *beamline);
void correct_lattice_save_correctors(RUN *run, LINE_LIST *beamline);

/* Tell each correction module to reassert its saved values.  Each returns
 * the count of correctors actually restored. */
long correct_reassert_correctors(RUN *run, LINE_LIST *beamline);
long correct_tunes_reassert_correctors(RUN *run, LINE_LIST *beamline);
long correct_chromaticity_reassert_correctors(RUN *run, LINE_LIST *beamline);
long correct_coupling_reassert_correctors(RUN *run, LINE_LIST *beamline);
long correct_lattice_reassert_correctors(RUN *run, LINE_LIST *beamline);

/* Invalidate every correction module's stash.  Called from &run_setup. */
void corrector_stash_invalidate_all(void);

/* Dispatcher used by vary_beamline().  On the first tick after &run_setup
 * (corrector_stash_snapshotted==0) this calls every module's save hook and
 * sets the flag; on subsequent ticks it calls every module's reassert hook.
 * Returns the total number of correctors restored. */
long corrector_stash_step_tick(RUN *run, LINE_LIST *beamline);

/* Snapshot-state flag shared between vary.c and the invalidator. */
extern short corrector_stash_snapshotted;

#endif /* CORRECTOR_STASH_H */
