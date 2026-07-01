/*************************************************************************\
* Copyright (c) 2002 The University of Chicago, as Operator of Argonne
* National Laboratory.
* This file is distributed subject to a Software License Agreement found
* in the file LICENSE that is included with this distribution.
\*************************************************************************/

/* file: knobs.h
 *
 * Public API for the knob registry populated by the &load_knobs setup
 * command.  A knob is a scalar variable; setting its value V offsets
 * each linked element parameter by (V - V_previous) * Factor, so the
 * lattice carries the cumulative contribution of every knob without
 * needing a separate baseline array.  Multiple knobs may touch the
 * same parameter; their contributions add linearly.
 *
 * Knobs are made addressable from &vary_element, &optimization_variable,
 * and &error_element by storing T_KNOB (from track.h) in the per-slot
 * "type" array and the knob's index in knobList in the per-slot
 * "param_number" array.  assert_parameter_values() and
 * assert_perturbations() branch on T_KNOB and call set_knob_value() /
 * perturb_knob() instead of writing to a single element parameter.
 */

#ifndef KNOBS_H_INCLUDED
#define KNOBS_H_INCLUDED

/* Callers of this header must have already included "namelist.h" and
 * "track.h" -- track.h has no include guard and re-including it produces
 * conflicting struct definitions. */

typedef struct {
  char *name;                   /* knob scalar name; "<name>.value" is the RPN slot */
  long nTargets;                /* number of (element, parameter) targets after wildcard expansion */
  ELEMENT_LIST **eptr;          /* cached element pointers,                          length nTargets */
  long *paramIndex;             /* cached entity_description parameter index,        length nTargets */
  long *paramOffset;            /* cached byte offset inside p_elem,                 length nTargets */
  short *paramType;             /* IS_DOUBLE / IS_LONG / IS_SHORT,                   length nTargets */
  double *factor;               /* multiplier per target,                            length nTargets */
  double currentValue;          /* current scalar V; starts at 0 */
  long rpnMem;                  /* rpn_create_mem("<name>.value", 0) */
  short matrixChanges;          /* nonzero if any target carries PARAM_CHANGES_MATRIX */
} KNOB;

extern KNOB *knobList;
extern long nKnobs;

void setup_load_knobs(NAMELIST_TEXT *nltext, RUN *run, LINE_LIST *beamline);

/* Returns the index of the knob named `name`, or -1 if no knob has that name.
 * Match is case-sensitive AFTER str_toupper on both sides (we upper-case
 * the registered name at load time, mirroring the rest of elegant). */
long find_knob(const char *name);

double get_knob_value(long knobIndex);

/* Apply (newValue - knob->currentValue) * Factor to every linked element
 * parameter, then update knob->currentValue = newValue and refresh the
 * "<name>.value" RPN scalar.  Invalidates the beamline's matrix/twiss/radint
 * caches if any target carries PARAM_CHANGES_MATRIX. */
void set_knob_value(long knobIndex, double newValue, LINE_LIST *beamline);

/* Convenience wrapper for &error_element: equivalent to
 *   set_knob_value(knobIndex, get_knob_value(knobIndex) + delta, beamline). */
void perturb_knob(long knobIndex, double delta, LINE_LIST *beamline);

#endif
