/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* Copyright (c) 2002 The Regents of the University of California, as
* Operator of Los Alamos National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution. 
\*************************************************************************/

/* routine: get_phase_reference()
 * purpose: store/retrieve reference phases for time-dependent elements
 *
 * Michael Borland, 1989
 */
#include "mdb.h"
#include "track.h"

/* flag bit values */
#define FL_REF_PHASE_SET 1

struct phase_reference {
  long ref_number;
  long flags;
  double phase;
} *reference = NULL;
static long n_references = 0;

long get_phase_reference(
  double *phase,
  long phase_ref_number) {
  long i;
#ifdef DEBUG
  printf("get_phase_reference(%le, %ld)\n", *phase, phase_ref_number);
  fflush(stdout);
#endif

  log_entry("get_phase_reference");

  if (phase_ref_number == 0) {
    log_exit("get_phase_reference");
    return (0);
  }

  for (i = 0; i < n_references; i++) {
    if (reference[i].ref_number == phase_ref_number) {
      if (reference[i].flags & FL_REF_PHASE_SET) {
        *phase = reference[i].phase;
#ifdef DEBUG
        printf("returning REF_PHASE_RETURNED\n");
        fflush(stdout);
#endif
        log_exit("get_phase_reference");
        return (REF_PHASE_RETURNED);
      }
#ifdef DEBUG
      printf("returning REF_PHASE_NOT_SET\n");
      fflush(stdout);
#endif
      log_exit("get_phase_reference");
      return (REF_PHASE_NOT_SET);
    }
  }
#ifdef DEBUG
  printf("returning REF_PHASE_NONEXISTENT\n");
  fflush(stdout);
#endif
  log_exit("get_phase_reference");
  return (REF_PHASE_NONEXISTENT);
}

long set_phase_reference(
  long phase_ref_number, /* number of the phase reference group */
  double phase           /* phase to assert for fiducial particle */
) {
  long i;

  log_entry("set_phase_reference");

#ifdef DEBUG
  printf("set_phase_reference(%ld, %le)\n", phase_ref_number, phase);
  fflush(stdout);
#endif

  if (phase_ref_number == 0) {
    log_exit("set_phase_reference");
    return (0);
  }

  for (i = 0; i < n_references; i++) {
    if (reference[i].ref_number == phase_ref_number) {
      reference[i].phase = phase;
      reference[i].flags = FL_REF_PHASE_SET;
#ifdef DEBUG
      printf("existing phase reference set\n");
      fflush(stdout);
#endif
      log_exit("set_phase_reference");
      return (1);
    }
  }
  if (phase_ref_number > LONG_MAX / 2)
    bombElegant("please use a small integer for the phase_reference number", NULL);
  reference = trealloc(reference, sizeof(*reference) * (++n_references));
  reference[i].ref_number = phase_ref_number;
  reference[i].phase = phase;
  reference[i].flags = FL_REF_PHASE_SET;
#ifdef DEBUG
  printf("new phase reference set\n");
  fflush(stdout);
#endif
  log_exit("set_phase_reference");
  return (1);
}

void delete_phase_references(void) {
  long i;

  log_entry("delete_phase_references");

  for (i = 0; i < n_references; i++)
    reference[i].flags = 0;
  /*
    printf("Phase references deleted\n");
    fflush(stdout);
    */
  log_exit("delete_phase_references");
}

long unused_phase_reference() {
  static long big = LONG_MAX;

  log_entry("unused_phase_reference");

  reference = trealloc(reference, sizeof(*reference) * (n_references + 1));
  reference[n_references].ref_number = big;
  reference[n_references].phase = 0;
  reference[n_references].flags = 0;
  n_references++;
  log_exit("unused_phase_reference");
  return (big--);
}

double get_reference_phase(long phase_ref, double phase0)
/* routine to maintain emulate old get_reference_phase(),
     * which is obsolete and should be phased out (pun intended)
     */
{
  double phase;

  log_entry("get_reference_phase");
#ifdef DEBUG
  printf("obsolete routine get_reference_phase called\n");
  fflush(stdout);
#endif
  switch (get_phase_reference(&phase, phase_ref)) {
  case REF_PHASE_RETURNED:
    log_exit("get_reference_phase");
    return (phase);
  case REF_PHASE_NOT_SET:
  case REF_PHASE_NONEXISTENT:
  default:
    set_phase_reference(phase_ref, phase0);
    log_exit("get_reference_phase");
    return (phase0);
  }
}

/* Snapshot only scalar table entries, never element structures or their owned
 * pointers. Restoring preserves the auto-generated reference-number allocator. */
typedef struct {
  long count;
  struct phase_reference *entry;
} PHASE_REFERENCE_SNAPSHOT;

void *save_phase_references(void) {
  PHASE_REFERENCE_SNAPSHOT *state = tmalloc(sizeof(*state));
  state->count = n_references;
  state->entry = n_references ? tmalloc(n_references * sizeof(*reference)) : NULL;
  if (n_references)
    memcpy(state->entry, reference, n_references * sizeof(*reference));
  return state;
}

void restore_phase_references(const void *snapshot) {
  const PHASE_REFERENCE_SNAPSHOT *state = snapshot;
  long i;
  if (!state)
    return;
  /* Auto-assigned references can be added after capture. Keep their numbers
   * allocated, but unset, just as delete_phase_references would. */
  for (i = 0; i < n_references; i++)
    reference[i].flags = 0;
  if (n_references < state->count) {
    reference = trealloc(reference, state->count * sizeof(*reference));
    n_references = state->count;
  }
  if (state->count)
    memcpy(reference, state->entry, state->count * sizeof(*reference));
}

void free_phase_references(void *snapshot) {
  PHASE_REFERENCE_SNAPSHOT *state = snapshot;
  if (state) {
    free(state->entry);
    free(state);
  }
}
